//! Matched model-to-model replay diagnostics, never a relabeling of a model as
//! SAM or human truth. Labels are read only after the inference runs finish.
use super::{boolean, eye, fresh_admitted, held, number, source_key, summary};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

fn admitted(row: &Value) -> Result<bool, String> {
    fresh_admitted(row, "accepted")
}
fn indexed(rows: &[Value]) -> Result<BTreeMap<String, &Value>, String> {
    let mut map = BTreeMap::new();
    for row in rows {
        if row["source_identity_verified"] != true {
            return Err("unverified source".into());
        }
        if map.insert(source_key(row)?, row).is_some() {
            return Err("duplicate source".into());
        }
    }
    Ok(map)
}
fn metrics(pairs: &[(&Value, &Value)]) -> Result<Value, String> {
    let mut counts = [0usize; 7];
    let mut centers = vec![];
    let mut radii = vec![];
    for &(a, b) in pairs {
        let (aa, bb) = (admitted(a)?, admitted(b)?);
        for (i, v) in [
            aa,
            bb,
            aa && bb,
            aa && !bb,
            bb && !aa,
            aa && boolean(&a["pupil_present"])?,
            bb && boolean(&b["pupil_present"])?,
        ]
        .into_iter()
        .enumerate()
        {
            counts[i] += usize::from(v);
        }
        if aa && bb {
            centers.push(super::distance(
                &a["outer_ellipse"]["center"],
                &b["outer_ellipse"]["center"],
            )?);
            radii.push(
                (number(&a["outer_ellipse"]["major_radius"])?
                    - number(&b["outer_ellipse"]["major_radius"])?)
                .abs(),
            );
        }
    }
    Ok(
        json!({"frames":pairs.len(),"baseline_admitted":counts[0],"candidate_admitted":counts[1],
        "both_admitted":counts[2],"baseline_only":counts[3],"candidate_only_unverified":counts[4],
        "baseline_pupil_with_admitted_outer":counts[5],"candidate_pupil_with_admitted_outer":counts[6],
        "common_outer_center_disagreement_px":summary(centers),"common_major_radius_disagreement_px":summary(radii)}),
    )
}
fn timing(rows: &[Value]) -> Result<Value, String> {
    let mut seen = BTreeSet::new();
    let (mut cold, mut warm) = (vec![], vec![]);
    for row in rows {
        if seen.insert(eye(row)) {
            cold.push(row)
        } else {
            warm.push(row)
        }
    }
    let mut stages = json!({});
    for (i, name) in [
        "prepare_cpu",
        "h2d_synchronized",
        "forward_synchronized",
        "mask_materialization",
        "remaining_downstream",
    ]
    .iter()
    .enumerate()
    {
        stages[name] = summary(
            warm.iter()
                .filter_map(|r| r["student_stages_ms"][i].as_f64()),
        );
    }
    Ok(
        json!({"warmup_first_per_eye_ms":cold.iter().map(|r|r["elapsed_ms"].clone()).collect::<Vec<_>>(),
        "warm_whole_worker_ms":summary(warm.iter().map(|r|number(&r["elapsed_ms"])).collect::<Result<Vec<_>,_>>()?),
        "warm_stages_ms":stages,"warm_rows_without_stage_timings":warm.iter().filter(|r|r["student_stages_ms"].is_null()).count(),
        "scope":"completion-paced per-ROI worker, excludes RAW archive IO/unpack, queue and camera/display; device synchronization enabled only by BUTTERCUP_EYE_STUDENT_TIMING=1; separate stage quantiles are not additive"}),
    )
}
fn dynamics(rows: &[Value]) -> Result<Value, String> {
    eligible_dynamics(rows, None)
}
fn eligible_dynamics(rows: &[Value], eligible: Option<&BTreeSet<String>>) -> Result<Value, String> {
    let observed = |row: &Value| -> Result<bool, String> {
        Ok(admitted(row)? && match eligible {Some(set)=>set.contains(&source_key(row)?),None=>true})
    };
    let mut groups = BTreeMap::<String, Vec<&Value>>::new();
    for row in rows {
        groups
            .entry(json!([row["input"]["clock_lineage"], eye(row)]).to_string())
            .or_default()
            .push(row);
    }
    let (mut times, mut displacements, mut reframe_displacements, mut scale_areas, mut area_deltas) =
        (vec![], vec![], vec![], vec![], vec![]);
    let (mut reframes, mut scale_missing) = (0, 0);
    for rows in groups.values_mut() {
        rows.sort_by_key(|r| r["input"]["frame"]["timestamp_ns"].as_u64().unwrap_or(0));
        for row in rows.iter() {
            if observed(row)? {
                if let Some(area) = area(row)? {
                    scale_areas.push(area)
                } else {
                    scale_missing += 1
                }
            }
        }
        for pair in rows.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let (fa, fb) = (&a["input"]["frame"], &b["input"]["frame"]);
            let ta = fa["timestamp_ns"].as_u64().ok_or("invalid time")?;
            let tb = fb["timestamp_ns"].as_u64().ok_or("invalid time")?;
            if tb <= ta {
                return Err("non-increasing per-eye source clock".into());
            }
            let dt = (tb - ta) as f64 / 1e9;
            times.push(dt);
            let reframe = ["sensor_x", "sensor_y", "width", "height"]
                .iter()
                .any(|k| fa[k] != fb[k]);
            reframes += usize::from(reframe);
            // Do not score sparse reacquisitions as adjacent motion evidence.
            if !observed(a)?
                || !observed(b)?
                || dt > 0.25
                || a["input"]["clock_attested"] != true
                || b["input"]["clock_attested"] != true
            {
                continue;
            }
            let (ea, eb) = (&a["outer_ellipse"], &b["outer_ellipse"]);
            let dx = number(&eb["center"][0])? + number(&fb["sensor_x"])?
                - number(&ea["center"][0])?
                - number(&fa["sensor_x"])?;
            let dy = number(&eb["center"][1])? + number(&fb["sensor_y"])?
                - number(&ea["center"][1])?
                - number(&fa["sensor_y"])?;
            displacements.push(dx.hypot(dy));
            if reframe {
                reframe_displacements.push(dx.hypot(dy));
            }
            if let (Some(aa), Some(ab)) = (area(a)?, area(b)?) {
                area_deltas.push((ab - aa).abs());
            }
        }
    }
    Ok(
        json!({"source_intervals_seconds":summary(times),"reframe_pairs":reframes,
        "fresh_adjacent_sensor_center_displacement_px":summary(displacements),
        "fresh_adjacent_reframe_sensor_center_displacement_px":summary(reframe_displacements),
        "sn_feida_mm2_independent_scale":summary(scale_areas),"adjacent_sn_feida_absolute_delta_mm2":summary(area_deltas),
        "admitted_frames_missing_independent_scale":scale_missing,
        "caution":"displacement is not error without reference motion; no held frames, no gaps >250ms, no unattested clocks in motion metrics; scale is external pixels_per_10mm, never fitted radius"}),
    )
}
fn area(row: &Value) -> Result<Option<f64>, String> {
    Ok(row["input"]["scale_hint"]["pixels_per_10mm"]
        .as_f64()
        .filter(|s| s.is_finite() && *s > 0.)
        .map(|s| {
            number(&row["outer_ellipse"]["major_radius"])
                .map(|r| std::f64::consts::PI * (r * 10. / s).powi(2))
        })
        .transpose()?)
}
// A 2048-edge polygon approximates the geometric nearest rim, without assuming
// that a human point was marked along the ellipse's center ray.
fn rim_distance(ellipse: &Value, p: &Value) -> Result<f64, String> {
    let (x, y) = (
        number(&p[0])? - number(&ellipse["center"][0])?,
        number(&p[1])? - number(&ellipse["center"][1])?,
    );
    let (s, c) = number(&ellipse["angle"])?.sin_cos();
    let p = [c * x + s * y, -s * x + c * y];
    let (a, b) = (
        number(&ellipse["major_radius"])?,
        number(&ellipse["minor_radius"])?,
    );
    if !(a >= b && b > 0.) {
        return Err("invalid ellipse radii".into());
    }
    let mut last = [a, 0.];
    let mut best = f64::INFINITY;
    for i in 1..=2048 {
        let (s, c) = (i as f64 * std::f64::consts::TAU / 2048.).sin_cos();
        let next = [a * c, b * s];
        let d = [next[0] - last[0], next[1] - last[1]];
        let t = (((p[0] - last[0]) * d[0] + (p[1] - last[1]) * d[1]) / (d[0] * d[0] + d[1] * d[1]))
            .clamp(0., 1.);
        best = best.min((p[0] - last[0] - t * d[0]).hypot(p[1] - last[1] - t * d[1]));
        last = next;
    }
    Ok(best)
}
fn labels(pairs: &[(&Value, &Value)], dataset: &Value) -> Result<Value, String> {
    let mut references = BTreeMap::new();
    for frame in dataset["frames"]
        .as_array()
        .ok_or("missing canonical label frames")?
    {
        let source = &frame["source"];
        let key = source_key(&json!({"input":source}))?;
        if references.insert(key, frame).is_some() {
            return Err("duplicate canonical label source".into());
        }
    }
    let mut rows = vec![];
    let mut distances = [vec![], vec![]];
    let mut matched = 0;
    for &(a, b) in pairs {
        let Some(label) = references.get(&source_key(a)?) else {
            continue;
        };
        matched += 1;
        if !admitted(a)? || !admitted(b)? {
            continue;
        }
        let mut per = [vec![], vec![]];
        for obs in label["observations"]
            .as_array()
            .ok_or("missing observations")?
        {
            if obs["targets"]["rim"].is_null()
                || obs["occluded"]
                    .as_array()
                    .is_some_and(|v| v.contains(&json!("rim")))
            {
                continue;
            }
            for (i, r) in [a, b].into_iter().enumerate() {
                per[i].push(rim_distance(&r["outer_ellipse"], &obs["targets"]["rim"])?);
            }
        }
        if per[0].is_empty() {
            continue;
        }
        for i in 0..2 {
            distances[i].extend_from_slice(&per[i]);
        }
        rows.push(json!({"label":label["label"],"source":a["input"],"baseline_rim_distance_px":summary(per[0].iter().copied()),"candidate_rim_distance_px":summary(per[1].iter().copied())}));
    }
    Ok(
        json!({"reference_frames":references.len(),"exact_identity_matched_frames":matched,"common_admitted_with_visible_rim_frames":rows.len(),
        "baseline_rim_distance_px":summary(distances[0].iter().copied()),"candidate_rim_distance_px":summary(distances[1].iter().copied()),
        "frames":rows,"scope":"post-inference canonical reviewed rim points; unweighted geometric distance to 2048-edge ellipse, paired midpoint is approximate and not a guaranteed surface apex; training overlap must be separately reported"}),
    )
}
pub fn compare(a: &[Value], b: &[Value], canonical: Option<&Value>) -> Result<Value, String> {
    let (am, bm) = (indexed(a)?, indexed(b)?);
    if am.keys().ne(bm.keys()) {
        return Err("sources differ; not a matched comparison".into());
    }
    let pairs: Vec<_> = am.iter().map(|(k, a)| (*a, bm[k])).collect();
    let mut common = BTreeSet::new();
    for &(a,b) in &pairs {if admitted(a)? && admitted(b)? {common.insert(source_key(a)?);}}
    let mut eyes = json!({});
    for id in pairs.iter().map(|(a, _)| eye(a)).collect::<BTreeSet<_>>() {
        eyes[&id] = metrics(
            &pairs
                .iter()
                .copied()
                .filter(|(a, _)| eye(a) == id)
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(
        json!({"schema":"buttercup-student-matched-comparison-v1","coverage":metrics(&pairs)?,"per_eye":eyes,
        "baseline_timing":timing(a)?,"candidate_timing":timing(b)?,"baseline_dynamics":dynamics(a)?,"candidate_dynamics":dynamics(b)?,
        "common_admission_baseline_dynamics":eligible_dynamics(a,Some(&common))?,
        "common_admission_candidate_dynamics":eligible_dynamics(b,Some(&common))?,
        "held_excluded":[a.iter().filter(|r|held(r)).count(),b.iter().filter(|r|held(r)).count()],
        "canonical_labels":canonical.map(|c|labels(&pairs,c)).transpose()?,
        "limitations":"single-user model agreement/coverage is not accuracy; timing is not offered-load throughput or presentation latency; independent scale and human-label availability are explicitly reported"}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row(seq: u64) -> Value {
        json!({"input":{"raw_sha256":format!("hash{seq}"),"clock_lineage":"clock","frame":{"eye_id":1,"sequence":seq,"timestamp_ns":seq*100000000,"sensor_x":10,"sensor_y":20}},"source_identity_verified":true,"accepted":true,"pupil_present":true,"outer_ellipse":{"center":[0,0],"major_radius":5,"minor_radius":4,"angle":0},"elapsed_ms":2})
    }
    #[test]
    fn compare_does_not_call_rgb_sam_and_excludes_held() {
        let a = vec![row(1), row(2)];
        let mut b = a.clone();
        b[1]["held"] = json!(true);
        let r = compare(&a, &b, None).unwrap();
        assert_eq!(r["coverage"]["baseline_only"], 1);
        assert_eq!(r["coverage"]["both_admitted"], 1);
        assert_eq!(r["baseline_timing"]["warm_whole_worker_ms"]["n"], 1);
        assert_eq!(
            r["baseline_dynamics"]["sn_feida_mm2_independent_scale"]["n"],
            0
        );
    }
    #[test]
    fn clock_origin_and_hash_mismatch_refuse_comparison() {
        let a = row(1);
        for field in ["sensor_x", "timestamp_ns"] {
            let mut b = a.clone();
            b["input"]["frame"][field] = json!(999);
            assert!(compare(&[a.clone()], &[b], None).is_err());
        }
        assert!(compare(&[a.clone(), a.clone()], &[a], None).is_err());
    }
    #[test]
    fn nearest_rim_and_sensor_reframe_not_crop_motion() {
        let mut a = row(1);
        a["input"]["clock_attested"] = json!(true);
        assert!((rim_distance(&a["outer_ellipse"], &json!([7, 0])).unwrap() - 2.).abs() < 1e-6);
        let mut b = row(2);
        b["input"]["clock_attested"] = json!(true);
        b["input"]["frame"]["sensor_x"] = json!(12);
        b["outer_ellipse"]["center"] = json!([-2, 0]);
        let r = dynamics(&[a, b]).unwrap();
        assert_eq!(r["reframe_pairs"], 1);
        assert_eq!(
            r["fresh_adjacent_reframe_sensor_center_displacement_px"]["mean"],
            0.
        );
    }
    #[test]
    fn common_area_pairs_never_bridge_rejected_or_unattested_sources() {
        let mut a=vec![row(1),row(2),row(3)];
        for row in &mut a {row["input"]["clock_attested"]=json!(true);row["input"]["scale_hint"]=json!({"pixels_per_10mm":20});}
        let mut b=a.clone();b[1]["accepted"]=json!(false);
        let r=compare(&a,&b,None).unwrap();
        assert_eq!(r["common_admission_baseline_dynamics"]["sn_feida_mm2_independent_scale"]["n"],2);
        assert_eq!(r["common_admission_baseline_dynamics"]["adjacent_sn_feida_absolute_delta_mm2"]["n"],0);
        a[1]["input"]["clock_attested"]=Value::Null;
        assert_eq!(dynamics(&a).unwrap()["adjacent_sn_feida_absolute_delta_mm2"]["n"],0);
    }
}
