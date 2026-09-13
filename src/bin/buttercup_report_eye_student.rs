//! CPU-only diagnostics. SAM agreement is never human-label ground truth.
use serde_json::{json, Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};
#[path = "../student_comparison.rs"]
mod comparison;

fn number(v: &Value) -> Result<f64, String> {
    v.as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(|| "missing/non-finite number".into())
}
fn boolean(v: &Value) -> Result<bool, String> {
    v.as_bool().ok_or_else(|| "missing boolean".into())
}
fn summary(values: impl IntoIterator<Item = f64>) -> Value {
    let mut v: Vec<_> = values.into_iter().filter(|x| x.is_finite()).collect();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n == 0 {
        return json!({"n":0,"median":null,"mean":null,"p95":null});
    }
    let median = if n % 2 == 0 {
        v[n / 2 - 1] / 2.0 + v[n / 2] / 2.0
    } else {
        v[n / 2]
    };
    json!({"n":n,"median":median,"mean":v.iter().sum::<f64>()/n as f64,"p95":v[(n*95).div_ceil(100)-1]})
}
fn distance(a: &Value, b: &Value) -> Result<f64, String> {
    let a = a
        .as_array()
        .filter(|v| v.len() == 2)
        .ok_or("invalid ellipse center")?;
    let b = b
        .as_array()
        .filter(|v| v.len() == 2)
        .ok_or("invalid ellipse center")?;
    Ok((number(&a[0])? - number(&b[0])?).hypot(number(&a[1])? - number(&b[1])?))
}
fn held(row: &Value) -> bool {
    row["held"] == true
        || row["is_held"] == true
        || row["fresh"] == false
        || row["held_prediction"] == true
        || row["fresh_observation"] == false
        || row["state"] == "held"
}
fn fresh_admitted(row: &Value, field: &str) -> Result<bool, String> {
    Ok(boolean(&row[field])? && !held(row))
}
fn eye(row: &Value) -> String {
    row["input"]["frame"]["eye_id"]
        .as_u64()
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unknown".into())
}
fn split_report(rows: &[&Value]) -> Result<Value, String> {
    let mut counts = BTreeMap::<&str, usize>::new();
    let (mut center, mut pupil, mut major, mut minor) = (vec![], vec![], vec![], vec![]);
    let mut areas = [vec![], vec![]];
    let mut scale_count = 0;
    for row in rows {
        let (t, s) = (&row["teacher"], &row["student"]);
        let (ta, sa) = (
            fresh_admitted(t, "raw_admitted")?,
            fresh_admitted(s, "raw_admitted")?,
        );
        for (name, flag) in [
            ("teacher_admitted", ta),
            ("student_admitted", sa),
            ("both_admitted", ta && sa),
            ("teacher_only", ta && !sa),
            ("student_only_unverified", sa && !ta),
            ("teacher_pupil", ta && !t["pupil_ellipse"].is_null()),
            ("student_pupil", sa && !s["pupil_ellipse"].is_null()),
        ] {
            *counts.entry(name).or_default() += usize::from(flag);
        }
        if ta && sa {
            let (te, se) = (&t["outer_ellipse"], &s["outer_ellipse"]);
            center.push(distance(&te["center"], &se["center"])?);
            major.push((number(&te["major_radius"])? - number(&se["major_radius"])?).abs());
            minor.push((number(&te["minor_radius"])? - number(&se["minor_radius"])?).abs());
            if !t["pupil_ellipse"].is_null() && !s["pupil_ellipse"].is_null() {
                pupil.push(distance(
                    &t["pupil_ellipse"]["center"],
                    &s["pupil_ellipse"]["center"],
                )?);
            }
        }
        if let Some(scale) = row["input"]["scale_hint"]["pixels_per_10mm"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            scale_count += 1;
            for (i, result, admitted) in [(0, t, ta), (1, s, sa)] {
                if admitted && !result["outer_ellipse"].is_null() {
                    areas[i].push(
                        std::f64::consts::PI
                            * (number(&result["outer_ellipse"]["major_radius"])? / (scale / 10.0))
                                .powi(2),
                    );
                }
            }
        }
    }
    let sessions: BTreeSet<_> = rows
        .iter()
        .map(|r| r["input"]["clock_lineage"].to_string())
        .collect();
    let mut result = json!({"frames":rows.len(),"sessions":sessions.len(),
        "center_disagreement_with_teacher_px":summary(center),
        "pupil_center_disagreement_with_teacher_px":summary(pupil),
        "major_radius_disagreement_with_teacher_px":summary(major),
        "minor_radius_disagreement_with_teacher_px":summary(minor),
        "sn_feida_mm2_heuristic_scale":{"teacher":summary(areas[0].iter().copied()),"student":summary(areas[1].iter().copied())},
        "student_cached_input_mask_and_geometry_ms":summary(rows.iter().map(|r|number(&r["student_ms"])).collect::<Result<Vec<_>,_>>()?),
        "teacher_six_prompt_export_ms":summary(rows.iter().map(|r|number(&r["teacher"]["teacher_ms"])).collect::<Result<Vec<_>,_>>()?),
        "independent_scale_available_frames":scale_count,"independent_scale_missing_frames":rows.len()-scale_count,
        "teacher_held_excluded":rows.iter().filter(|r|held(&r["teacher"])).count(),
        "student_held_excluded":rows.iter().filter(|r|held(&r["student"])).count(),
        "human_label_availability":"not inspected; this report measures no human-label localization"});
    for (k, v) in counts {
        result[k] = json!(v);
    }
    Ok(result)
}
fn source_key(row: &Value) -> Result<String, String> {
    let input = &row["input"];
    let hash = input["raw_sha256"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing RAW identity")?;
    let lineage = input["clock_lineage"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or("missing source clock lineage")?;
    let frame = input["frame"]
        .as_object()
        .ok_or("missing source frame identity")?;
    for k in ["eye_id", "sequence", "timestamp_ns"] {
        if frame.get(k).and_then(Value::as_u64).is_none() {
            return Err(format!("missing source frame {k}"));
        }
    }
    // Include all acquisition/ROI metadata, not only a potentially repeated RAW hash.
    Ok(json!([hash, lineage, frame]).to_string())
}
fn coverage(pairs: &[(&Value, &Value)]) -> Result<Value, String> {
    let mut counts = [0usize; 5];
    for &(a, b) in pairs {
        let (ta, sa) = (
            fresh_admitted(a, "accepted")?,
            fresh_admitted(b, "accepted")?,
        );
        for (i, v) in [
            ta && sa,
            ta && !sa,
            sa && !ta,
            ta && boolean(&a["pupil_present"])?,
            sa && boolean(&b["pupil_present"])?,
        ]
        .into_iter()
        .enumerate()
        {
            counts[i] += usize::from(v);
        }
    }
    Ok(
        json!({"frames":pairs.len(),"both_admitted":counts[0],"sam_only":counts[1],
        "student_only_unverified":counts[2],"sam_pupil":counts[3],"student_pupil":counts[4]}),
    )
}
fn report(rows: &[Value], replays: &[(String, Vec<Value>)]) -> Result<Value, String> {
    let mut report = json!({"scope":"730-frame pilot if using the initial dataset; actual counts below",
        "human_label_localization":"not measured; SAM agreement is not human-label accuracy",
        "timing_caution":"student_ms excludes cached input preprocessing; teacher_ms queries six prompts, not the normal two-head live path",
        "area_caution":"SN-FEIDA uses source-provided independent pixels_per_10mm only; sparse frames are not a temporal-stability evaluation",
        "splits":{},"completion_paced_live_worker_replays":{},
        "generalization_scope":"single-user development results; no cross-user readiness claim"});
    for row in rows {
        if !matches!(
            row["input"]["student_split"].as_str(),
            Some("train" | "validation" | "test")
        ) {
            return Err("missing or unknown student split".into());
        }
        if row["input"]["clock_lineage"].as_str().is_none() {
            return Err("missing clock lineage".into());
        }
    }
    for split in ["train", "validation", "test"] {
        let subset: Vec<_> = rows
            .iter()
            .filter(|r| r["input"]["student_split"] == split)
            .collect();
        let mut value = split_report(&subset)?;
        let mut per_eye = Map::new();
        for id in subset.iter().map(|r| eye(r)).collect::<BTreeSet<_>>() {
            per_eye.insert(
                id.clone(),
                split_report(
                    &subset
                        .iter()
                        .copied()
                        .filter(|r| eye(r) == id)
                        .collect::<Vec<_>>(),
                )?,
            );
        }
        value["per_eye"] = Value::Object(per_eye);
        report["splits"][split] = value;
    }
    let mut by_backend: BTreeMap<String, BTreeMap<String, &Value>> = BTreeMap::new();
    for (path, replay) in replays {
        let mut seen = BTreeSet::new();
        let mut warm = vec![];
        let backend = replay.first().and_then(|r| r["backend"].as_str());
        for row in replay {
            let name = row["backend"].as_str().ok_or("missing replay backend")?;
            if Some(name) != backend {
                return Err("mixed backends in replay file".into());
            }
            if by_backend
                .entry(name.into())
                .or_default()
                .insert(source_key(row)?, row)
                .is_some()
            {
                return Err("duplicate replay source identity".into());
            }
            if !seen.insert(eye(row)) {
                warm.push(row);
            }
        }
        let accepted = replay
            .iter()
            .map(|r| fresh_admitted(r, "accepted"))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|v| *v)
            .count();
        let verified = replay
            .iter()
            .map(|r| boolean(&r["source_identity_verified"]))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .all(|v| v);
        report["completion_paced_live_worker_replays"][path] = json!({"frames":replay.len(),"accepted":accepted,
            "source_identity_verified":verified,
            "held_excluded":replay.iter().filter(|r|held(r)).count(),
            "elapsed_ms_excluding_first_per_eye":summary(warm.iter().map(|r|number(&r["elapsed_ms"])).collect::<Result<Vec<_>,_>>()?),
            "encode_ms_excluding_first_per_eye":summary(warm.iter().map(|r|number(&r["encode_ms"])).collect::<Result<Vec<_>,_>>()?),
            "note":"includes live preprocessing and shared geometry, not camera/display latency or offered-load frame drops"});
    }
    if let (Some(sam), Some(student)) = (by_backend.get("sam"), by_backend.get("student")) {
        if sam.keys().ne(student.keys()) {
            return Err("replay sources differ; cannot report matched coverage".into());
        }
        let pairs: Vec<_> = sam.iter().map(|(k, a)| (*a, student[k])).collect();
        if pairs.iter().any(|(a, b)| {
            a["source_identity_verified"] != true || b["source_identity_verified"] != true
        }) {
            return Err("replay source identity unverified; cannot report matched coverage".into());
        }
        let mut value = coverage(&pairs)?;
        let mut per_eye = Map::new();
        for id in pairs.iter().map(|(a, _)| eye(a)).collect::<BTreeSet<_>>() {
            per_eye.insert(
                id.clone(),
                coverage(
                    &pairs
                        .iter()
                        .copied()
                        .filter(|(a, _)| eye(a) == id)
                        .collect::<Vec<_>>(),
                )?,
            );
        }
        value["per_eye"] = Value::Object(per_eye);
        report["matched_live_worker_coverage"] = value;
    }
    Ok(report)
}
fn read_jsonl(path: &str) -> Result<Vec<Value>, String> {
    BufReader::new(File::open(Path::new(path)).map_err(|e| format!("{path}: {e}"))?)
        .lines()
        .enumerate()
        .map(|(i, line)| {
            serde_json::from_str(&line.map_err(|e| e.to_string())?)
                .map_err(|e| format!("{path}:{}: {e}", i + 1))
        })
        .collect()
}
fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let evaluation = args
        .next()
        .ok_or("usage: buttercup_report_eye_student EVALUATION [--replay JSONL]...")?;
    if evaluation == "--compare" {
        let baseline = read_jsonl(&args.next().ok_or("missing baseline replay")?)?;
        let candidate = read_jsonl(&args.next().ok_or("missing candidate replay")?)?;
        let labels = args.next().map(|path| {
            serde_json::from_reader(File::open(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
        }).transpose()?;
        if args.next().is_some() { return Err("unexpected comparison argument".into()); }
        println!("{}", serde_json::to_string_pretty(&comparison::compare(&baseline, &candidate, labels.as_ref())?).map_err(|e| e.to_string())?);
        return Ok(());
    }
    if evaluation == "--help" || evaluation == "-h" {
        println!("Summarize matched SAM/student diagnostics without calling SAM ground truth.\nUsage: buttercup_report_eye_student EVALUATION [--replay JSONL]...");
        return Ok(());
    }
    let mut replays = vec![];
    while let Some(arg) = args.next() {
        if arg != "--replay" {
            return Err(format!("unexpected argument {arg}"));
        }
        let path = args.next().ok_or("missing --replay path")?;
        replays.push((path.clone(), read_jsonl(&path)?));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report(&read_jsonl(&evaluation)?, &replays)?)
            .map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn evaluation() -> Value {
        let ellipse = json!({"center":[1,2],"major_radius":4,"minor_radius":3});
        json!({"input":{"student_split":"train","clock_lineage":"session","scale_hint":{"pixels_per_10mm":20},"frame":{"eye_id":0}},
            "teacher":{"raw_admitted":true,"outer_ellipse":ellipse,"pupil_ellipse":null,"teacher_ms":12},
            "student":{"raw_admitted":true,"outer_ellipse":{"center":[4,6],"major_radius":5,"minor_radius":2},"pupil_ellipse":null},"student_ms":3})
    }
    fn replay(backend: &str, eye: u64, seq: u64) -> Value {
        json!({"backend":backend,"input":{"raw_sha256":format!("raw-{eye}-{seq}"),"clock_lineage":"session","frame":{"eye_id":eye,"sequence":seq,"timestamp_ns":seq+100}},
            "accepted":true,"source_identity_verified":true,"pupil_present":true,"elapsed_ms":seq+1,"encode_ms":seq+2})
    }
    #[test]
    fn summary_matches_python_quantiles() {
        assert_eq!(
            summary([1., 2., 3., 4.]),
            json!({"n":4,"mean":2.5,"median":2.5,"p95":4.})
        );
        assert_eq!(summary([f64::NAN])["n"], 0);
        assert_eq!(summary((1..=20).map(f64::from))["p95"], 19.);
    }
    #[test]
    fn evaluation_metrics_and_empty_splits() {
        let r = report(&[evaluation()], &[]).unwrap();
        let train = &r["splits"]["train"];
        assert_eq!(train["center_disagreement_with_teacher_px"]["mean"], 5.);
        assert_eq!(
            train["major_radius_disagreement_with_teacher_px"]["mean"],
            1.
        );
        assert_eq!(train["per_eye"]["0"]["both_admitted"], 1);
        assert_eq!(
            train["sn_feida_mm2_heuristic_scale"]["teacher"]["mean"],
            json!(4. * std::f64::consts::PI)
        );
        assert_eq!(r["splits"]["test"]["frames"], 0);
        assert!(r["splits"]["test"].get("both_admitted").is_none());
    }
    #[test]
    fn held_and_missing_scale_do_not_count() {
        let mut row = evaluation();
        row["student"]["held"] = json!(true);
        row["input"]["scale_hint"] = Value::Null;
        let r = report(&[row], &[]).unwrap();
        let t = &r["splits"]["train"];
        assert_eq!(t["teacher_only"], 1);
        assert_eq!(t["student_admitted"], 0);
        assert_eq!(t["student_held_excluded"], 1);
        assert_eq!(t["independent_scale_missing_frames"], 1);
        assert_eq!(t["sn_feida_mm2_heuristic_scale"]["student"]["n"], 0);
    }
    #[test]
    fn matched_replay_warmup_and_held() {
        let a = vec![
            replay("sam", 0, 0),
            replay("sam", 1, 0),
            replay("sam", 0, 1),
            replay("sam", 1, 1),
        ];
        let mut b = a.clone();
        for row in &mut b {
            row["backend"] = json!("student");
        }
        b[2]["held"] = json!(true);
        let r = report(&[], &[("sam".into(), a), ("student".into(), b)]).unwrap();
        assert_eq!(
            r["completion_paced_live_worker_replays"]["sam"]["elapsed_ms_excluding_first_per_eye"]
                ["n"],
            2
        );
        assert_eq!(r["matched_live_worker_coverage"]["both_admitted"], 3);
        assert_eq!(r["matched_live_worker_coverage"]["sam_only"], 1);
        assert_eq!(
            r["matched_live_worker_coverage"]["per_eye"]["0"]["sam_only"],
            1
        );
    }
    #[test]
    fn reject_duplicate_mismatched_and_unverified_sources() {
        let a = replay("sam", 0, 0);
        let b = replay("student", 0, 0);
        assert!(report(&[], &[("a".into(), vec![a.clone(), a.clone()])])
            .unwrap_err()
            .contains("duplicate"));
        let mut mismatch = b.clone();
        mismatch["input"]["frame"]["timestamp_ns"] = json!(999);
        assert!(report(
            &[],
            &[("a".into(), vec![a.clone()]), ("b".into(), vec![mismatch])]
        )
        .unwrap_err()
        .contains("sources differ"));
        let mut unverified = b;
        unverified["source_identity_verified"] = json!(false);
        assert!(report(
            &[],
            &[("a".into(), vec![a]), ("b".into(), vec![unverified])]
        )
        .unwrap_err()
        .contains("unverified"));
    }

    #[test]
    fn shared_raw_bytes_do_not_collapse_distinct_exposures() {
        let mut a = vec![replay("sam", 0, 0), replay("sam", 0, 1)];
        a[1]["input"]["raw_sha256"] = a[0]["input"]["raw_sha256"].clone();
        let mut b = a.clone();
        for row in &mut b {
            row["backend"] = json!("student");
        }
        let r = report(&[], &[("a".into(), a), ("b".into(), b)]).unwrap();
        assert_eq!(r["matched_live_worker_coverage"]["frames"], 2);
    }
}
