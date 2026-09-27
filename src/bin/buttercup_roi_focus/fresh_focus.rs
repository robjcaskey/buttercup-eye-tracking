//! Area admission BEFORE circle interpretations, region estimation and scoring.
use super::{
    archive,
    area_consistency::{self, Admission, Observation},
    Result,
};
use buttercup_eye_tracking::{focus_region::*, geometry::Ellipse};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

const PROVIDERS: [&str; 2] = ["sam", "obelisk"];
fn load(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn u(v: &Value) -> Result<u64> {
    v.as_u64().ok_or_else(|| "missing unsigned identity".into())
}
fn add(c: &mut BTreeMap<String, usize>, key: &str) {
    *c.entry(key.into()).or_default() += 1;
}
fn counts() -> BTreeMap<String, usize> {
    [
        "multiple",
        "one",
        "zero",
        "unresolved_region",
        "invalid_projection",
    ]
    .map(|s| (s.into(), 0))
    .into_iter()
    .collect()
}
fn ellipse(row: &Value, provider: &str) -> Result<Option<Ellipse>> {
    let fit = &row[provider]["fit"];
    if fit.is_null() || row[provider]["admissible"] != true {
        return Ok(None);
    }
    let origin: [f64; 2] = serde_json::from_value(fit["origin"].clone())?;
    if origin
        != [
            u(&row["frame"]["sensor_x"])? as f64,
            u(&row["frame"]["sensor_y"])? as f64,
        ]
    {
        return Err("fit crop changed".into());
    }
    let e: [f64; 5] = serde_json::from_value(fit["ellipse"].clone())?;
    if !e.iter().all(|x| x.is_finite()) || e[3] <= 0. || e[2] < e[3] {
        return Ok(None);
    }
    Ok(Some(Ellipse {
        center: (e[0] + origin[0], e[1] + origin[1]),
        major_radius: e[2],
        minor_radius: e[3],
        angle: e[4],
    }))
}

/// Only constructed after area admission. Downstream estimation accepts this
/// retained list, so excluded rows cannot contribute an intersection or vote.
struct Retained {
    row: usize,
    id: u64,
    source: u64,
    epoch: u64,
    eye: u64,
    ns: u64,
    ellipse: Ellipse,
    area: Admission,
}

fn evaluate(
    name: &str,
    retained: &[Retained],
    original: &[Value],
    out: &Path,
    class_writer: &mut BufWriter<fs::File>,
    pair_writer: &mut BufWriter<fs::File>,
) -> Result<Value> {
    let mut groups = BTreeMap::<(u64, u64), Vec<usize>>::new();
    let mut rays = vec![];
    let mut max_parity = 0f64;
    for (i, row) in retained.iter().enumerate() {
        assert!(
            row.area.accepted,
            "excluded area reached the geometry stage"
        );
        groups.entry((row.source, row.epoch)).or_default().push(i);
        let geometry =
            TheoreticalEllipseExplanations::from_ellipse(row.ellipse, [4000.; 2], [4000., 3000.]);
        if let Some(g) = geometry {
            let fit = &original[row.row][name]["fit"];
            let normals: [[f64; 3]; 2] = serde_json::from_value(fit["normals"].clone())?;
            let centers: [[f64; 3]; 2] = serde_json::from_value(fit["centers_per_radius"].clone())?;
            for j in 0..2 {
                max_parity = max_parity
                    .max(norm(sub(g.rays[j].direction, normals[j])))
                    .max(norm(sub(g.rays[j].origin_iris_radii, centers[j])));
            }
        }
        rays.push(geometry);
    }
    if max_parity > 1e-8 {
        return Err("native circle projection parity failed".into());
    }
    let mut totals = counts();
    let mut group_reports = vec![];
    let mut used_in_fit = BTreeSet::new();
    let mut fit_pairs = 0usize;
    let mut groups_with_regions = 0usize;
    let mut prefix_counts = counts();
    for ((source, epoch), ids) in groups {
        let start_ns = ids.iter().map(|&i| retained[i].ns).min().unwrap();
        let mut pairs = BTreeMap::<u64, [Option<usize>; 2]>::new();
        for &i in &ids {
            let row = &retained[i];
            if rays[i].is_some() && (1..=2).contains(&row.eye) {
                let slot = &mut pairs.entry(row.ns).or_insert([None; 2])[row.eye as usize - 1];
                if slot.replace(i).is_some() {
                    return Err("duplicate retained eye exposure".into());
                }
            }
        }
        let pairs = pairs
            .into_iter()
            .filter_map(|(ns, p)| Some((ns, [p[0]?, p[1]?])))
            .collect::<Vec<_>>();
        let mut estimator = FocusRegionEstimator::new(FocusOptions::default());
        let cutoff = pairs.len() * 7 / 10;
        let cutoff_ns = pairs.get(cutoff).map(|p| p.0).unwrap_or(u64::MAX);
        let mut prefix_regions = vec![];
        for (pi, &(ns, pair)) in pairs.iter().enumerate() {
            if pi == cutoff {
                prefix_regions = estimator.regions();
            }
            let a = &retained[pair[0]];
            let b = &retained[pair[1]];
            assert!(
                a.area.accepted && b.area.accepted,
                "area rejection reached region fitting"
            );
            used_in_fit.insert(a.id);
            used_in_fit.insert(b.id);
            let intersections = estimator.observe(
                (ns - start_ns) as f64 / 1e9,
                rays[pair[0]].as_ref().unwrap(),
                rays[pair[1]].as_ref().unwrap(),
            );
            serde_json::to_writer(
                &mut *pair_writer,
                &json!({"provider":name,"source":source,"epoch":epoch,"source_ns":ns.to_string(),"records":[a.id,b.id],"possibilities":intersections}),
            )?;
            writeln!(pair_writer)?;
        }
        fit_pairs += pairs.len();
        let regions = estimator.regions();
        groups_with_regions += usize::from(!regions.is_empty());
        let mut classes = counts();
        for &i in &ids {
            let input = &retained[i];
            let (class, interpretations) = verdict(rays[i], &regions);
            add(&mut classes, class);
            add(&mut totals, class);
            let prefix_class = if input.ns >= cutoff_ns {
                let p = verdict(rays[i], &prefix_regions).0;
                add(&mut prefix_counts, p);
                Some(p)
            } else {
                None
            };
            serde_json::to_writer(
                &mut *class_writer,
                &json!({"record":input.id,"provider":name,"source":source,"epoch":epoch,"eye":input.eye,"sequence":original[input.row]["sequence"],"source_ns":input.ns.to_string(),"raw_sha256":original[input.row]["raw_sha256"],"area_admission":input.area,"class":class,"geometry":rays[i],"interpretations":interpretations,"retrospective_region_fit":true,"prefix70_class":prefix_class}),
            )?;
            writeln!(class_writer)?;
        }
        group_reports.push(json!({"source":source,"epoch":epoch,"retained_frames":ids.len(),"paired_exposures":pairs.len(),"geometric_pairs":estimator.geometric_pairs,"discarded_cluster_births":estimator.discarded_cluster_births,"regions":regions,"classes":classes,"prefix70_regions":prefix_regions}));
    }
    let retained_ids = retained.iter().map(|r| r.id).collect::<BTreeSet<_>>();
    assert!(used_in_fit.is_subset(&retained_ids));
    assert_eq!(totals.values().sum::<usize>(), retained.len());
    let result = json!({"retained_frames":retained.len(),"retained_records":retained_ids,"classes":totals,
        "paired_exposures":fit_pairs,"records_used_for_regions":used_in_fit,"groups_with_regions":groups_with_regions,
        "groups":group_reports,"prefix70_last30_classes":prefix_counts,"max_fresh_ray_error":max_parity,
        "excluded_frames_in_geometry":0,"excluded_frames_in_region_fit":0,"excluded_frames_in_classification":0});
    fs::write(
        out.join(format!("{name}-regions.json")),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}

fn verdict(
    rays: Option<TheoreticalEllipseExplanations>,
    regions: &[FocusRegion],
) -> (&'static str, Vec<RayClassification>) {
    let Some(rays) = rays else {
        return ("invalid_projection", vec![]);
    };
    if regions.is_empty() {
        return ("unresolved_region", vec![]);
    }
    let interpretations = rays.rays.map(|r| classify(r, regions, 2.)).to_vec();
    let n = interpretations
        .iter()
        .filter(|r| matches!(r.status, "inside" | "nearby"))
        .count();
    (["zero", "one", "multiple"][n], interpretations)
}

pub fn run(fresh_dir: &str, baseline_dir: &str, output: &str) -> Result<()> {
    let fresh = Path::new(fresh_dir);
    let baseline = Path::new(baseline_dir);
    let out = Path::new(output);
    if out.exists() {
        return Err("output exists".into());
    }
    let inference = load(&fresh.join("summary.json"))?;
    let provenance = load(&fresh.join("provenance.json"))?;
    // Original metadata establishes cohort identity only. Its focus regions,
    // classifications and stored ray explanations never enter this pipeline.
    let prior = load(&baseline.join("summary.json"))?;
    if inference["complete"] != true
        || prior["schema"] != "buttercup-focus-volume-experiment-v1"
        || provenance["binary_sha256"] != prior["binary_sha256"]
    {
        return Err("incomplete inference or mismatched source archive".into());
    }
    let input_bytes = fs::read(fresh.join("frames.jsonl"))?;
    let rows = BufReader::new(input_bytes.as_slice())
        .lines()
        .map(|l| Ok(serde_json::from_str::<Value>(&l?)?))
        .collect::<Result<Vec<_>>>()?;
    let expected = u(&inference["counts_total_sam_fit_admitted_obelisk_fit_admitted"][0])? as usize;
    if rows.len() != expected || expected != u(&prior["classes"]["multiple"])? as usize {
        return Err("incomplete original ambiguous cohort".into());
    }
    let mut seen = BTreeSet::new();
    let mut exposures = BTreeSet::new();
    let mut identities = vec![];
    let mut shapes = [vec![], vec![]];
    let mut observations = [vec![], vec![]];
    for row in &rows {
        let id = u(&row["record"])?;
        let source = u(&row["source"])?;
        let epoch = u(&row["epoch"])?;
        let eye = u(&row["eye"])?;
        let ns = row["source_ns"]
            .as_str()
            .ok_or("missing source time")?
            .parse::<u64>()?;
        if !seen.insert(id) || !exposures.insert((source, epoch, eye, ns)) {
            return Err("duplicate exposure".into());
        }
        for key in [
            "record",
            "source",
            "epoch",
            "eye",
            "sequence",
            "source_ns",
            "raw_sha256",
        ] {
            if row[key] != row["previous_evaluation"][key] {
                return Err(format!("identity changed: {key}").into());
            }
        }
        identities.push((id, source, epoch, eye, ns));
        for (pi, name) in PROVIDERS.iter().enumerate() {
            let e = ellipse(row, name)?;
            observations[pi].push(Observation {
                group: (source, epoch, eye),
                ns,
                area_px2: e.map(|s| std::f64::consts::PI * s.major_radius.powi(2)),
            });
            shapes[pi].push(e);
        }
    }
    // First substantive stage: area admission. No candidate rays have been
    // computed and no focus-region estimator exists yet.
    let admission = observations.each_ref().map(|o| area_consistency::assess(o));
    fs::create_dir_all(out)?;
    let mut eligibility = BufWriter::new(fs::File::create(out.join("eligibility.jsonl"))?);
    let mut retained_writer = BufWriter::new(fs::File::create(out.join("retained-inputs.jsonl"))?);
    let mut rejected = BufWriter::new(fs::File::create(out.join("excluded-inputs.jsonl"))?);
    let mut retained: [Vec<Retained>; 2] = [vec![], vec![]];
    let mut admissions = [BTreeMap::<String, usize>::new(), BTreeMap::new()];
    for (i, &(id, source, epoch, eye, ns)) in identities.iter().enumerate() {
        for (pi, name) in PROVIDERS.iter().enumerate() {
            let gate = &admission[pi][i];
            add(&mut admissions[pi], gate.reason);
            let record = json!({"record":id,"provider":name,"source":source,"epoch":epoch,"eye":eye,"source_ns":ns.to_string(),"area_admission":gate});
            serde_json::to_writer(&mut eligibility, &record)?;
            writeln!(eligibility)?;
            if !gate.accepted {
                serde_json::to_writer(&mut rejected, &record)?;
                writeln!(rejected)?;
                continue;
            }
            retained[pi].push(Retained {
                row: i,
                id,
                source,
                epoch,
                eye,
                ns,
                ellipse: shapes[pi][i].ok_or("accepted missing shape")?,
                area: gate.clone(),
            });
            serde_json::to_writer(
                &mut retained_writer,
                &json!({"record":id,"provider":name,"source":source,"epoch":epoch,"eye":eye,"sequence":rows[i]["sequence"],"source_ns":ns.to_string(),"raw_sha256":rows[i]["raw_sha256"],"raw_source":rows[i]["raw_source"],"stream_entry":rows[i]["stream_entry"],"frame":rows[i]["frame"],"fit":rows[i][name]["fit"],"area_admission":gate}),
            )?;
            writeln!(retained_writer)?;
        }
    }
    eligibility.flush()?;
    retained_writer.flush()?;
    rejected.flush()?;
    // Only the typed retained collections cross into the geometry stage.
    let mut classifications = BufWriter::new(fs::File::create(out.join("classifications.jsonl"))?);
    let mut intersections = BufWriter::new(fs::File::create(out.join("intersections.jsonl"))?);
    let mut results = vec![];
    for (pi, name) in PROVIDERS.iter().enumerate() {
        results.push(evaluate(
            name,
            &retained[pi],
            &rows,
            out,
            &mut classifications,
            &mut intersections,
        )?);
    }
    classifications.flush()?;
    intersections.flush()?;
    let summary = json!({"schema":"buttercup-area-first-focus-v1","complete":true,"input_exposures":rows.len(),
        "admission":{"sam":admissions[0],"obelisk":admissions[1]},"sam":results[0],"obelisk":results[1],
        "fresh_frames_sha256":archive::digest(&input_bytes),"fresh_inference_provenance":provenance,
        "original_metadata_sha256":archive::digest(&fs::read(baseline.join("summary.json"))?),
        "executable_sha256":archive::digest(&fs::read(std::env::current_exe()?)?),
        "policy":{"stage_order":["native fit/RAW gate","source-timed frontal-equivalent area admission","circle interpretations","new region estimates from admitted exact-time eye pairs","classify admitted frames"],
            "area":"pi*major_radius^2; leave-one-out 20%-trimmed mean within +/-1 source second; >=6 neighbors, >=2 on each side, >=100ms span; ratio in [2/3,1.5]",
            "unknown_area":"excluded before geometry, region fitting and ambiguity counting",
            "region_reuse":false,"focus_options":FocusOptions::default(),"nearby_degrees":2.},
        "scope":"Only the original ambiguous cohort freshly segmented by both models. Regions rebuilt separately per model from retained exact-time eye pairs. No bad/unknown-area row or archived focus region contributes. Symmetric area gate and full-region results are retrospective. Prefix70 regions still use retrospectively area-admitted observations, so this is not a causal online validation. No independent scale, sign or localization truth; stable size alone does not certify anatomical correctness."});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    eprintln!(
        "AREA-FIRST FOCUS {}",
        serde_json::to_string(
            &json!({"admission":summary["admission"],"sam":summary["sam"]["classes"],"obelisk":summary["obelisk"]["classes"]})
        )?
    );
    Ok(())
}
