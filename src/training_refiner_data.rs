//! Native RAW preparation and reporting. No learned dependencies or camera IO.
#![allow(dead_code)]
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const ROLES: [&str; 6] = [
    "rim",
    "band_inner",
    "band_outer",
    "iris_onset",
    "surface_apex",
    "subsurface_limit",
];
#[path = "refinement_target_report.rs"]
mod refinement_target_report;
fn read(path: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array().ok_or_else(|| "expected array".into())
}
fn string(v: &Value) -> Result<&str> {
    v.as_str().ok_or_else(|| "expected string".into())
}
fn int(v: &Value) -> Result<u64> {
    v.as_u64()
        .ok_or_else(|| "expected nonnegative integer".into())
}
fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Array(a) => !a.is_empty(),
        Value::Object(a) => !a.is_empty(),
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64() != Some(0.),
    }
}
fn visible(p: &Value) -> bool {
    p["visibility"] == "visible" && p["x"].is_number() && p["y"].is_number()
}
pub fn observations(label: &Value) -> Vec<Value> {
    let mut result = vec![];
    for p in label["annotation_points"].as_array().into_iter().flatten() {
        if p["kind"] != "iris_edge" || !visible(p) {
            continue;
        }
        let xy = json!([p["x"], p["y"]]);
        let mut targets = json!({"rim": xy});
        for (field, role) in [
            ("band_inner", "band_inner"),
            ("band_outer", "band_outer"),
            ("iris_side_onset", "iris_onset"),
            ("subsurface_visibility_limit", "subsurface_limit"),
        ] {
            if !p[field].is_null() {
                targets[role] = p[field].clone();
            }
        }
        if p["source"]
            .as_str()
            .unwrap_or("")
            .contains("visibility_triplet_apex")
        {
            targets["surface_apex"] = xy.clone();
        }
        let occluded = if p["possibly_occluded"]
            .as_array()
            .is_some_and(|a| a.contains(&json!("submerged")))
        {
            vec!["subsurface_limit"]
        } else {
            vec![]
        };
        result.push(json!({"anchor":xy,"targets":targets,"weights":{"rim":if p["source"]=="paired_midpoint" {0.3} else {1.0}},"occluded":occluded,"kind":p.get("source").unwrap_or(&json!("legacy_single"))}));
    }
    for p in label["limbus_partial_observations"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let m = &p["landmarks"];
        if !m["apex"].is_null() || !visible(&m["inner"]) {
            continue;
        }
        let xy = json!([m["inner"]["x"], m["inner"]["y"]]);
        let triplet = p["mode"] == "triplet";
        let mut targets = json!({});
        targets[if triplet { "iris_onset" } else { "band_inner" }] = xy.clone();
        result.push(json!({"anchor":xy,"targets":targets,"weights":{},"kind":"possibly_occluded","occluded":if triplet {vec!["rim","surface_apex","subsurface_limit"]} else {vec!["band_outer"]}}));
    }
    result
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn prepare(inventory: &Value) -> Result<Value> {
    let (mut rows, mut excluded, mut seen) = (vec![], vec![], HashSet::new());
    for name in array(&inventory["labels"])? {
        let name = string(name)?;
        let path = Path::new(name);
        if name.contains("assistant")
            || path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .contains("backup")
        {
            excluded.push(json!({"path":name,"reason":"assistant-or-backup"}));
            continue;
        }
        let label = read(path)?;
        if !truth(&label["reviewed"]) {
            excluded.push(json!({"path":name,"reason":"not-reviewed"}));
            continue;
        }
        let raw = Path::new(string(&label["source_raw"])?);
        let metadata = read(raw.with_extension("json"))?;
        let (w, h) = (int(&label["frame_width"])?, int(&label["frame_height"])?);
        let bytes = fs::read(raw)?;
        let sha = digest(&bytes);
        let identity = json!([sha, w, h, label["sensor_origin"]]).to_string();
        if !seen.insert(identity) {
            return Err("duplicate reviewed native label; resolve precedence explicitly".into());
        }
        if metadata["width"] != w
            || metadata["height"] != h
            || label["sensor_origin"] != json!([metadata["sensor_x"], metadata["sensor_y"]])
            || bytes.len() as u64 != w / 4 * 5 * h
        {
            return Err(format!("native RAW/label identity mismatch: {}", path.display()).into());
        }
        let mut frame = json!({});
        for key in [
            "sequence",
            "timestamp_ns",
            "sensor_x",
            "sensor_y",
            "width",
            "height",
            "stride",
            "pixel_format",
        ] {
            frame[key] = metadata
                .get(key)
                .ok_or_else(|| format!("missing metadata {key}"))?
                .clone();
        }
        frame["eye_id"] = metadata.get("eye_id").cloned().unwrap_or(json!(1));
        frame["label"] = json!(if frame["eye_id"] == 1 {
            "subject-right"
        } else {
            "subject-left"
        });
        rows.push(json!({"source":{"index":rows.len(),"raw_file":raw.canonicalize()?,"raw_offset":0,"raw_length":bytes.len(),"raw_sha256":sha,"frame":frame,"scale_hint":null,"clock_attested":false},"label":path.canonicalize()?,"observations":observations(&label)}));
    }
    for row in &rows {
        int(&row["source"]["frame"]["timestamp_ns"])?;
    }
    rows.sort_by_key(|r| r["source"]["frame"]["timestamp_ns"].as_u64().unwrap());
    let (mut group, mut previous) = (-1i64, None);
    for row in &mut rows {
        let t = int(&row["source"]["frame"]["timestamp_ns"])?;
        if previous.is_none_or(|p| t - p > 300_000_000_000) {
            group += 1;
        }
        row["group"] = json!(group);
        row["source"]["clock_lineage"] = json!(format!("limbus-human-conservative-group:{group}"));
        previous = Some(t);
    }
    Ok(
        json!({"schema":"buttercup-limbus-patch-dataset-v1","roles":ROLES,"frames":rows,"excluded":excluded,"grouping":"sensor-timestamp-neighborhoods <=300s conservatively merged; clock identity not independently attested","distance_scale_support":"missing; do not normalize area by candidate radius","landmark_semantics":"optical image landmarks; no anatomical 3D depth or signed gaze labels"}),
    )
}
pub fn prepare_sequence(outline_path: &Path, area_path: &Path) -> Result<(Value, Vec<Value>)> {
    let (outlines, area) = (read(outline_path)?, read(area_path)?);
    let (mut rows, mut references, mut motion_reports) = (vec![], vec![], BTreeMap::new());
    for case in array(&outlines["cases"])? {
        let f = &case["frame"];
        let raw = Path::new(string(&f["source_raw"])?);
        let payload = fs::read(raw)?;
        if payload.len() as u64 != int(&f["length"])? {
            return Err("archived RAW length mismatch".into());
        }
        let mut src = json!({"raw_file":raw.canonicalize()?,"raw_offset":0,"raw_length":payload.len(),"raw_sha256":digest(&payload),"frame":f,"clock_attested":false,"clock_lineage":f["lineage"],"scale_hint":null});
        let mut matched = vec![];
        for r in array(&area["frames"])? {
            if Path::new(string(&r["source"])?).canonicalize()? == outline_path.canonicalize()?
                && ["sequence", "timestamp_ns", "width", "height"]
                    .iter()
                    .all(|k| r[k] == f[k])
                && r["eye"] == f["label"]
            {
                matched.push(r);
            }
        }
        if matched.len() != 1 {
            return Err("ambiguous outline/motion source join".into());
        }
        let hint = &matched[0]["independent_scale"];
        if truth(hint) {
            let name = string(&hint["report"])?;
            if !motion_reports.contains_key(name) {
                motion_reports.insert(name.to_string(), read(name)?);
            }
            let motion = &motion_reports[name];
            let matched: Vec<_> = array(&motion["frames"])?
                .iter()
                .filter(|m| {
                    ["sequence", "timestamp_ns", "width", "height"]
                        .iter()
                        .all(|k| m[k] == f[k])
                        && m["sensor_origin"] == json!([f["sensor_x"], f["sensor_y"]])
                })
                .collect();
            if matched.len() != 1 || motion["source"]["label"] != f["label"] {
                return Err("ambiguous native-motion metadata".into());
            }
            let m = matched[0];
            let mut stream = fs::File::open(string(&motion["source"]["stream"])?)?;
            stream.seek(SeekFrom::Start(int(&m["source_offset"])?))?;
            let mut bytes = vec![];
            stream
                .take(int(&m["source_length"])?)
                .read_to_end(&mut bytes)?;
            if bytes != payload {
                return Err("native-motion bytes differ from contour exposure".into());
            }
            src["relative_scale_hint"] = hint.clone();
            src["clock_basis"] = json!("exact-RAW-matched-native-motion-chain");
            src["clock_lineage"] = json!(Path::new(name).canonicalize()?);
        }
        let selected = array(&case["candidates"])?
            .iter()
            .find(|c| truth(&c["baseline_raw_admitted"]) && truth(&c["baseline_ellipse"]));
        let label_path = f.get("canonical_label").filter(|v| truth(v));
        let label = match label_path {
            Some(p) => read(string(p)?)?,
            None => json!({}),
        };
        rows.push(json!({"source":src,"group":-4,"label":label_path.cloned().unwrap_or(json!(raw)),"observations":if truth(&label["reviewed"]) {observations(&label)} else {vec![]},"supervision":"evaluation-only-sequence"}));
        references.push(json!({"input":src,"accepted":selected.is_some(),"source_identity_verified":true,"candidates":selected.into_iter().collect::<Vec<_>>()}));
    }
    Ok((
        json!({"schema":"buttercup-limbus-patch-dataset-v1","roles":ROLES,"frames":rows,"evaluation_only":true,"scope":"archived contour replay; relative scale is not camera range"}),
        references,
    ))
}
fn stats(values: impl IntoIterator<Item = Option<f64>>) -> Value {
    let mut v: Vec<_> = values
        .into_iter()
        .flatten()
        .filter(|v| v.is_finite())
        .collect();
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return json!({"n":0,"mean":null,"median":null,"p95":null,"max":null});
    }
    let n = v.len();
    json!({"n":n,"mean":v.iter().sum::<f64>()/n as f64,"median":if n%2==0 {(v[n/2-1]+v[n/2])/2.} else {v[n/2]},"p95":v[(0.95*n as f64).ceil() as usize-1],"max":v[n-1]})
}
fn counts(values: impl IntoIterator<Item = String>) -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for s in values {
        *m.entry(s).or_default() += 1;
    }
    m
}
pub fn summarize_cv(reports: &[Value]) -> Result<Value> {
    let (mut frames, mut identities) = (vec![], HashSet::new());
    for report in reports {
        let train = array(&report["model_training"]["train_raw_sha256"])?;
        for row in array(&report["frames"])? {
            if row["split"] != "test" {
                continue;
            }
            let identity = string(&row["source"]["raw_sha256"])?;
            if train.contains(&json!(identity)) || !identities.insert(identity) {
                return Err("duplicate held-out source or training leakage".into());
            }
            if report["model_training"]
                .get("test_group")
                .is_some_and(|g| g != &row["group"])
            {
                return Err("held-out source group mismatch".into());
            }
            frames.push(row);
        }
    }
    let admitted: Vec<_> = frames
        .iter()
        .copied()
        .filter(|r| truth(&r["baseline_raw_admitted"]))
        .collect();
    let matched: Vec<_> = admitted
        .iter()
        .copied()
        .filter(|r| truth(&r["candidate"]) && truth(&r["unmodified_refit"]))
        .collect();
    let error = |r: &Value, k: &str| r[k]["rim"]["rms_px"].as_f64();
    let mut pairs = vec![];
    for r in &matched {
        // Missing labels remain missing, including for otherwise accepted fits.
        if let (Some(b), Some(c), Some(n)) = (
            error(r, "baseline_errors"),
            error(r, "unmodified_refit_errors"),
            error(r, "candidate_errors"),
        ) {
            pairs.push(json!({"source":r["source"],"baseline_rms_px":b,"unmodified_refit_rms_px":c,"candidate_rms_px":n,"delta_from_control_px":n-c}));
        }
    }
    Ok(
        json!({"scope":"development grouped cross-validation; whole-frame contour fixed before label scoring","frames":frames.len(),"groups":counts(frames.iter().map(|r| r["group"].to_string())),"baseline_raw_admitted":admitted.len(),"accepted_refinement":matched.len(),"baseline_retained_without_refinement":admitted.len()-matched.len(),
        "matched_baseline_rms_px":stats(matched.iter().map(|r| error(r,"baseline_errors"))),"matched_control_rms_px":stats(matched.iter().map(|r| error(r,"unmodified_refit_errors"))),"matched_refined_rms_px":stats(matched.iter().map(|r| error(r,"candidate_errors"))),
        "all_admitted_delta_including_unchanged_fallback_px":stats(admitted.iter().map(|r| { let b = error(r,"baseline_errors")?; let n = if truth(&r["candidate_errors"]) {error(r,"candidate_errors")?} else {b}; Some(n-b) })),"refine_cpu_ms":stats(admitted.iter().map(|r| r["elapsed_ms"].as_f64())),"matched":pairs,
        "failures":frames.iter().filter(|r| !truth(&r["candidate"])).map(|r| json!({"source":r["source"],"status":r["status"],"baseline_raw_admitted":r["baseline_raw_admitted"]})).collect::<Vec<_>>(),
        "limitations":["Generic rim includes lower-weight band midpoints, not exclusively surface apex.","Ten RAW-admitted labels are not ten independent people or sessions.","Fallback is the original baseline, not a fresh refined observation.","Model comparisons informed development; this is not a sealed final test."]}),
    )
}
pub fn sequence_pairs(rows: &[Value]) -> Result<Vec<Value>> {
    // Preserve first-seen lineage order, as in the reference reporter.
    let mut lineages: Vec<(Value, Vec<&Value>)> = vec![];
    for r in rows {
        let src = &r["source"];
        if !truth(&src["clock_attested"])
            && src["clock_basis"] != "exact-RAW-matched-native-motion-chain"
        {
            continue;
        }
        let f = &src["frame"];
        let key = json!([src["clock_lineage"], f["eye_id"], f["region"]["session"]]);
        int(&f["timestamp_ns"])?;
        if let Some((_, g)) = lineages.iter_mut().find(|(k, _)| *k == key) {
            g.push(r);
        } else {
            lineages.push((key, vec![r]));
        }
    }
    let mut pairs = vec![];
    for (key, mut group) in lineages {
        group.sort_by_key(|r| r["source"]["frame"]["timestamp_ns"].as_u64().unwrap());
        for pair in group.windows(2) {
            let (before, after) = (pair[0], pair[1]);
            let (a, b) = (&before["source"]["frame"], &after["source"]["frame"]);
            let delta = int(&b["timestamp_ns"])? - int(&a["timestamp_ns"])?;
            if delta == 0
                || delta > 500_000_000
                || int(&a["sequence"])? >= int(&b["sequence"])?
                || ![before, after].iter().all(|r| {
                    truth(&r["baseline_raw_admitted"])
                        && truth(&r["candidate"])
                        && truth(&r["sn_feida"])
                })
            {
                continue;
            }
            let (p, q) = (&before["sn_feida"], &after["sn_feida"]);
            if p["scale_reference"] != q["scale_reference"] || p["units"] != q["units"] {
                continue;
            }
            let mut result = json!({"lineage":key,"before":before["source"]["raw_sha256"],"after":after["source"]["raw_sha256"],"dt_ms":delta as f64/1e6,"roi_reframed":a["sensor_x"] != b["sensor_x"] || a["sensor_y"] != b["sensor_y"]});
            let mut modes = vec!["baseline", "candidate"];
            if truth(&p["unmodified_refit"]) && truth(&q["unmodified_refit"]) {
                modes.push("unmodified_refit");
            }
            let mut valid = true;
            for mode in modes {
                let v = |r: &Value| r.get(mode).unwrap_or(&r[format!("{mode}_mm2")]).as_f64();
                if let (Some(x), Some(y)) = (v(p), v(q)) {
                    if x > 0. && y > 0. && x.is_finite() && y.is_finite() {
                        result[format!("{mode}_absolute_log_step")] = json!((y / x).ln().abs());
                        continue;
                    }
                }
                valid = false;
                break;
            }
            if valid {
                pairs.push(result);
            }
        }
    }
    Ok(pairs)
}
pub fn summarize_sequence(report: &Value) -> Result<Value> {
    let frames = array(&report["frames"])?;
    let pairs = sequence_pairs(frames)?;
    Ok(
        json!({"scope":"conditional archived-contour diagnostic; includes development sources, not a sealed final test","physical_context_ablated":report["physical_context_ablated"],"frames":frames.len(),"raw_admitted":frames.iter().filter(|r| truth(&r["baseline_raw_admitted"])).count(),"refined":frames.iter().filter(|r| truth(&r["candidate"]) && truth(&r["baseline_raw_admitted"])).count(),"independent_scale_frames":frames.iter().filter(|r| truth(&r["sn_feida"])).count(),"scale_units":counts(frames.iter().filter(|r| truth(&r["sn_feida"])).map(|r| r["sn_feida"]["units"].as_str().unwrap_or("coarse_mm2").to_string())),"matched_area_pairs":pairs.len(),"reframed_pairs":pairs.iter().filter(|p| truth(&p["roi_reframed"])).count(),"baseline_abs_log_sn_feida_step":stats(pairs.iter().map(|p| p["baseline_absolute_log_step"].as_f64())),"refined_abs_log_sn_feida_step":stats(pairs.iter().map(|p| p["candidate_absolute_log_step"].as_f64())),"control_abs_log_sn_feida_step":stats(pairs.iter().map(|p| p["unmodified_refit_absolute_log_step"].as_f64())),"refine_cpu_ms":stats(frames.iter().map(|r| r["elapsed_ms"].as_f64())),"pairs":pairs,"limitations":["Coarse scale is independent of candidate radius but is not calibrated metric truth.","Scale-only bounds omit fitted-radius and optical/model uncertainty.","Sparse samples, not an exhaustive sequential video replay; no missing frames filled.","Constant or wrong ellipses are not validated by area stability."]}),
    )
}
fn output_path(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let resolved = if path.exists() {
        path.canonicalize()?
    } else {
        parent
            .canonicalize()?
            .join(path.file_name().ok_or("missing output filename")?)
    };
    if !resolved.starts_with(Path::new("outputs").canonicalize()?) {
        return Err("output must be beneath outputs".into());
    }
    Ok(resolved)
}
fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    writeln!(f)?;
    Ok(())
}
fn write_lines(path: &Path, values: impl IntoIterator<Item = Value>) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    for v in values {
        serde_json::to_writer(&mut f, &v)?;
        writeln!(f)?;
    }
    Ok(())
}
pub fn prepare_cli(args: Vec<String>) -> Result<()> {
    if args.len() != 2 && !(args.len() == 4 && args[2] == "--sequence-area-report") {
        return Err("usage: buttercup_prepare_limbus_refiner INVENTORY OUTPUT [--sequence-area-report REPORT]".into());
    }
    let output = output_path(Path::new(&args[1]))?;
    let (data, reference) = if args.len() == 4 {
        let (d, r) = prepare_sequence(Path::new(&args[0]), Path::new(&args[3]))?;
        (d, Some(r))
    } else {
        (prepare(&read(&args[0])?)?, None)
    };
    fs::create_dir(&output)?;
    if let Some(r) = reference {
        write_lines(&output.join("sam-baseline.jsonl"), r)?;
    }
    write_json(&output.join("dataset.json"), &data)?;
    let frames = array(&data["frames"])?;
    write_lines(
        &output.join("sam-inputs.jsonl"),
        frames.iter().map(|r| r["source"].clone()),
    )?;
    let mut landmarks = vec![];
    let mut occluded = vec![];
    for r in frames {
        for o in array(&r["observations"])? {
            landmarks.extend(
                o["targets"]
                    .as_object()
                    .ok_or("missing targets")?
                    .keys()
                    .cloned(),
            );
            for v in array(&o["occluded"])? {
                occluded.push(string(v)?.to_string());
            }
        }
    }
    println!(
        "{}",
        json!({"frames":frames.len(),"groups":counts(frames.iter().map(|r| r["group"].to_string())),"landmarks":counts(landmarks),"explicitly_occluded":counts(occluded)})
    );
    Ok(())
}
/// Compare end-to-end worker replays, not archived contour-only refits.
/// Labels are scoring-only, and unsupported scale/held contacts stay missing.
pub fn summarize_live_pair(baseline: &Value, candidate: &Value) -> Result<Value> {
    for key in ["capture", "label", "sampling", "model", "backend"] {
        if baseline[key].is_null() || baseline[key] != candidate[key] {
            return Err(format!("mismatched replay {key}").into());
        }
    }
    let left = array(&baseline["cases"])?;
    let right = array(&candidate["cases"])?;
    if left.len()!=right.len() || left.is_empty() { return Err("unmatched source census".into()); }
    let fresh = |c:&Value| c["contact_fresh_source"]==true
        && c["virtual_contact"]["sign_resolved"]==true
        && c["virtual_contact"]["source_timestamp_ns"]==c["frame"]["timestamp_ns"];
    let mut rows=vec![];
    for (a,b) in left.iter().zip(right) {
        if a["frame"]!=b["frame"] || a["processed_roi"]!=b["processed_roi"]
            || a["motion_clock"]!=b["motion_clock"] || a["independent_motion"]!=b["independent_motion"] {
            return Err("mismatched source, crop, clock or independent motion".into());
        }
        let matched_raw=a["raw_admitted"]==true && b["raw_admitted"]==true;
        let label_pair=matched_raw.then(||a["human_visible_limbus"]["rms_px"].as_f64()
            .zip(b["human_visible_limbus"]["rms_px"].as_f64())).flatten();
        let area_pair=a["sn_feida_log_step"].as_f64().zip(b["sn_feida_log_step"].as_f64());
        let gaze_delta=if fresh(a)&&fresh(b) {
            let x=&a["virtual_contact"]["relative_gaze_vector"];
            let y=&b["virtual_contact"]["relative_gaze_vector"];
            (0..3).map(|i|x[i].as_f64().zip(y[i].as_f64()).map(|(x,y)|x*y))
                .collect::<Option<Vec<_>>>().map(|v|v.iter().sum::<f64>().clamp(-1.0,1.0).acos().to_degrees())
        } else {None};
        let center_delta=if matched_raw {
            (0..2).map(|i|a["outer"]["center"][i].as_f64().zip(b["outer"]["center"][i].as_f64()).map(|(x,y)|x-y))
                .collect::<Option<Vec<_>>>().map(|v|v[0].hypot(v[1]))
        } else {None};
        rows.push(json!({"frame":a["frame"],"refinement":b["limbus_refinement"],
            "raw":[a["raw_admitted"],b["raw_admitted"]],"fresh_signed_contact":[fresh(a),fresh(b)],
            "source_reframed":a["processed_roi"]["reframed"],"label_rms_pair_px":label_pair,
            "sn_feida_abs_log_step_pair":area_pair.map(|(x,y)|(x.abs(),y.abs())),
            "center_delta_px":center_delta,"gaze_delta_degrees_not_accuracy":gaze_delta}));
    }
    let mut result=json!({"schema":"buttercup-limbus-live-comparison-v1",
        "capture":baseline["capture"],"label":baseline["label"],"sampling":baseline["sampling"],
        "sources":rows.len(),"refinement_applied":rows.iter().filter(|r|r["refinement"]["applied"]==true).count(),
        "refinement_status":counts(rows.iter().map(|r|r["refinement"]["status"].as_str().unwrap_or("NO ATTEMPT").to_string())),
        "cpu_refinement_ms":stats(rows.iter().map(|r|r["refinement"]["cpu_ms"].as_f64())),
        "center_delta_px":stats(rows.iter().map(|r|r["center_delta_px"].as_f64())),
        "gaze_delta_degrees_not_accuracy":stats(rows.iter().map(|r|r["gaze_delta_degrees_not_accuracy"].as_f64())),
        "label_rms_delta_px":stats(rows.iter().map(|r|r["label_rms_pair_px"][0].as_f64().zip(r["label_rms_pair_px"][1].as_f64()).map(|(a,b)|b-a))),
        "contract":"single-user implementation comparison, not cold-bootstrap proof, signed-gaze truth or cross-user validation; fresh source-matched contacts only; independent source-clock RAW texture scale, never candidate radius; no unsupported scale or dropout bridges; report label coverage and regressions with area",
    });
    for (i,name) in ["baseline","candidate"].iter().enumerate() {
        result[*name]=json!({
            "raw_admitted":rows.iter().filter(|r|r["raw"][i]==true).count(),
            "fresh_signed_contacts":rows.iter().filter(|r|r["fresh_signed_contact"][i]==true).count(),
            "matched_human_rms_px":stats(rows.iter().map(|r|r["label_rms_pair_px"][i].as_f64())),
            "matched_sn_feida_abs_log_step":stats(rows.iter().map(|r|r["sn_feida_abs_log_step_pair"][i].as_f64())),
            "matched_reframe_sn_feida_abs_log_step":stats(rows.iter().filter(|r|r["source_reframed"]==true).map(|r|r["sn_feida_abs_log_step_pair"][i].as_f64()))});
    }
    result["matched"]=json!(rows);
    Ok(result)
}

pub fn report_cli(args: Vec<String>) -> Result<()> {
    if args.first().is_some_and(|mode|mode=="targets") {
        return refinement_target_report::report(&args);
    }
    if args.len() < 3
        || !["cv", "sequence", "live"].contains(&args[0].as_str())
        || (args[0] == "sequence" && args.len() != 3)
        || (args[0] == "live" && args.len() != 4)
    {
        return Err("usage: buttercup_report_limbus_refiner cv|sequence OUTPUT REPORT...; or live OUTPUT BASELINE CANDIDATE".into());
    }
    let output = output_path(Path::new(&args[1]))?;
    let reports = args[2..].iter().map(read).collect::<Result<Vec<_>>>()?;
    let result = if args[0] == "cv" {
        summarize_cv(&reports)?
    } else if args[0] == "live" {
        let mut result=summarize_live_pair(&reports[0], &reports[1])?;
        result["inputs"]=json!(args[2..].iter().map(|path|Ok(json!({"path":path,
            "sha256":format!("{:x}",Sha256::digest(fs::read(path)?))}))).collect::<Result<Vec<_>>>()?);
        result
    } else {
        summarize_sequence(&reports[0])?
    };
    write_json(&output, &result)?;
    let mut short = result;
    for key in ["matched", "pairs", "failures"] {
        short.as_object_mut().unwrap().remove(key);
    }
    println!("{short}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_comparison_rejects_mismatched_clocks_and_does_not_count_held_gaze() {
        let report=json!({"capture":"same","label":"right","sampling":{},"model":"same","backend":"student",
            "cases":[{"frame":{"timestamp_ns":20},"processed_roi":{},"motion_clock":{"epoch":1},"independent_motion":{},
                "raw_admitted":false,"contact_fresh_source":true,
                "virtual_contact":{"sign_resolved":true,"source_timestamp_ns":10}}]});
        let summary=summarize_live_pair(&report,&report).unwrap();
        assert_eq!(summary["baseline"]["fresh_signed_contacts"],0);
        assert_eq!(summary["baseline"]["matched_sn_feida_abs_log_step"]["n"],0);
        assert_eq!(summary["baseline"]["matched_human_rms_px"]["n"],0);
        let mut other=report.clone();other["cases"][0]["motion_clock"]["epoch"]=json!(2);
        assert!(summarize_live_pair(&report,&other).is_err());
    }

    fn sample(i: u64) -> Value {
        json!({"source":{"raw_sha256":i.to_string(),"clock_attested":true,"clock_lineage":"sensor-a","frame":{"eye_id":0,"sequence":i,"timestamp_ns":i*100_000_000,"sensor_x":200,"sensor_y":800}},"baseline_raw_admitted":true,"candidate":{"major_radius":80},"sn_feida":{"baseline":100,"candidate":100,"scale_reference":12,"units":"reference_px2"}})
    }
    #[test]
    fn midpoint_is_not_apex() {
        let o = observations(
            &json!({"annotation_points":[{"kind":"iris_edge","visibility":"visible","x":10,"y":20,"source":"paired_midpoint","band_inner":[8,20],"band_outer":[12,20]}]}),
        );
        assert!(o[0]["targets"]["surface_apex"].is_null());
        assert_eq!(o[0]["weights"]["rim"], 0.3);
    }
    #[test]
    fn submerged_never_replaces_apex() {
        let o = observations(
            &json!({"annotation_points":[{"kind":"iris_edge","visibility":"visible","x":10,"y":20,"source":"visibility_triplet_apex","iris_side_onset":[8,20],"subsurface_visibility_limit":[15,20]}]}),
        );
        assert_eq!(o[0]["targets"]["rim"], json!([10, 20]));
        assert_eq!(o[0]["targets"]["surface_apex"], json!([10, 20]));
        assert_eq!(o[0]["targets"]["subsurface_limit"], json!([15, 20]));
    }
    #[test]
    fn unknown_and_guessed_not_imputed() {
        let o = observations(
            &json!({"limbus_partial_observations":[{"mode":"triplet","landmarks":{"inner":{"x":8,"y":20,"visibility":"visible"},"apex":null,"submerged":null}}]}),
        );
        assert_eq!(o[0]["targets"], json!({"iris_onset":[8,20]}));
        assert!(o[0]["occluded"]
            .as_array()
            .unwrap()
            .contains(&json!("surface_apex")));
        assert!(observations(&json!({"annotation_points":[{"kind":"iris_edge","visibility":"guessed","x":10,"y":20}]})).is_empty());
    }
    #[test]
    fn never_bridge_dropout_or_scale_reset() {
        let mut rows = vec![sample(1), sample(2), sample(3)];
        assert_eq!(sequence_pairs(&rows).unwrap().len(), 2);
        rows[1]["candidate"] = Value::Null;
        assert!(sequence_pairs(&rows).unwrap().is_empty());
        rows[1]["candidate"] = json!({"major_radius":80});
        rows[1]["sn_feida"]["scale_reference"] = json!(13);
        assert!(sequence_pairs(&rows).unwrap().is_empty());
    }
    #[test]
    fn crop_move_preserves_area_but_clock_change_breaks_pair() {
        let mut rows = vec![sample(1), sample(2)];
        rows[1]["source"]["frame"]["sensor_y"] = json!(824);
        let p = sequence_pairs(&rows).unwrap();
        assert_eq!(p[0]["roi_reframed"], true);
        assert_eq!(p[0]["candidate_absolute_log_step"], 0.);
        rows[1]["source"]["clock_lineage"] = json!("sensor-b");
        assert!(sequence_pairs(&rows).unwrap().is_empty());
    }
    #[test]
    fn held_out_leakage_and_wrong_group_rejected() {
        let mut row = sample(1);
        row["split"] = json!("test");
        row["group"] = json!(0);
        assert!(summarize_cv(&[
            json!({"model_training":{"train_raw_sha256":["1"]},"frames":[row]})
        ])
        .is_err());
        assert!(summarize_cv(&[
            json!({"model_training":{"train_raw_sha256":[],"test_group":1},"frames":[row]})
        ])
        .is_err());
    }
    #[test]
    fn missing_rim_error_remains_missing() {
        let mut row = sample(1);
        row["split"] = json!("test");
        row["group"] = json!(0);
        row["unmodified_refit"] = json!({"major_radius":80});
        let out = summarize_cv(&[json!({"model_training":{"train_raw_sha256":[]},"frames":[row]})])
            .unwrap();
        assert_eq!(out["matched_refined_rms_px"]["n"], 0);
        assert_eq!(
            out["all_admitted_delta_including_unchanged_fallback_px"]["mean"],
            Value::Null
        );
    }

    #[test]
    fn native_identity_and_duplicate_labels_checked() {
        let root = Path::new("outputs").canonicalize().unwrap().join(format!(
            "refiner-native-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let raw = root.join("frame.raw");
        fs::write(&raw, [0u8; 5]).unwrap();
        write_json(&root.join("frame.json"), &json!({"width":4,"height":1,"sensor_x":200,"sensor_y":800,"stride":5,"pixel_format":4,"sequence":1,"timestamp_ns":100})).unwrap();
        let label = root.join("label.json");
        write_json(&label, &json!({"reviewed":true,"source_raw":raw,"frame_width":4,"frame_height":1,"sensor_origin":[200,800]})).unwrap();
        let inventory = json!({"labels":[label]});
        let out = prepare(&inventory).unwrap();
        assert_eq!(out["frames"][0]["source"]["raw_sha256"], digest(&[0; 5]));
        assert_eq!(out["frames"][0]["source"]["scale_hint"], Value::Null);
        assert!(prepare(&json!({"labels":[label,label]}))
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        fs::write(&raw, [0u8; 4]).unwrap();
        assert!(prepare(&inventory)
            .unwrap_err()
            .to_string()
            .contains("identity mismatch"));
        // Retain the tiny runtime fixture as test evidence beneath outputs.
    }
}
