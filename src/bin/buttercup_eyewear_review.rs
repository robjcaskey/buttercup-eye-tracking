//! Save assistant-reviewed eyewear context without inventing reflection masks.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs, io::{BufReader, Read}, path::Path, process::Command};

fn hash(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut input = BufReader::new(fs::File::open(path)?);
    let mut h = Sha256::new(); let mut buffer = [0; 65536];
    loop { let n = input.read(&mut buffer)?; if n == 0 { break; } h.update(&buffer[..n]); }
    Ok(format!("{:x}", h.finalize()))
}
fn label(value: &str) -> Result<&str, &'static str> {
    match value { "probably-on" | "probably-off" | "mixed" | "unknown" => Ok(value),
        _ => Err("invalid eyewear review label") }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len()!=3 { return Err("usage: buttercup_eyewear_review REVIEWS.json OUTPUT_REPORT.json".into()); }
    let spec: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    if spec["schema"] != "buttercup-eyewear-review-input-v1" { return Err("wrong schema".into()); }
    let mut reports=Vec::new();
    for review in spec["reviews"].as_array().ok_or("missing reviews")? {
        let archive = Path::new(review["archive"].as_str().ok_or("missing archive")?);
        let state=label(review["state"].as_str().ok_or("missing state")?)?;
        let extraction=Command::new("tar").args(["-xOf"]).arg(archive).arg("thumbnails.jsonl").output()?;
        if !extraction.status.success() { return Err("cannot read archived thumbnail index".into()); }
        let records: Vec<Value> = String::from_utf8(extraction.stdout)?.lines()
            .map(serde_json::from_str).collect::<Result<_,_>>()?;
        let live: Vec<_> = records.iter().filter(|r| r["camera"]["frame_kind"] == "sensor_band"
            && r["recording_start_snapshot"] != true).collect();
        if live.is_empty() { return Err("no fresh sensor bands; do not use cached global snapshot".into()); }
        // Same deterministic five positions used by the visual review. These
        // are sample labels, not an assertion about every intermediate frame.
        let mut evidence=Vec::new(); let mut previous=None;
        for numerator in 0..5 {
            let index=(live.len()-1)*numerator/4;
            if previous==Some(index) { continue; } previous=Some(index);
            let r=live[index];
            evidence.push(json!({"sample_index":index,"offset":r["offset"],"length":r["length"],
                "stream":"thumbnails.oic1","camera":r["camera"],"suggested_state":state}));
        }
        let session_path=archive.with_extension("session.json");
        let session:Value=serde_json::from_slice(&fs::read(&session_path)?)?;
        let record=json!({"schema":"buttercup-eyewear-review-v1",
            "archive":archive,"archive_sha256":hash(archive)?,
            "session_sha256":hash(&session_path)?,"state":state,
            "annotation_source":"assistant-visual-review","human_reviewed":false,
            "scope":"recording-level tentative summary from five fresh sensor-band samples",
            "interval_host_unix_ns":[session["started_host_unix_ns"],session["ended_host_unix_ns"]],
            "evidence":evidence,"notes":review["notes"],
            "uncertainty":"Uninspected intervals may contain transitions; probably is not a calibrated probability.",
            "reflection_labels":[],"reflection_training_eligible":false,
            "limitations":["Glasses presence does not establish reflection over iris.",
                "Glasses-off is not a negative label for corneal glints or screen reflections.",
                "No dense masks, human labels, or model-derived iris geometry are invented."]});
        let destination=archive.with_extension("").join("annotator/labels/eyewear.assistant-review.json");
        fs::create_dir_all(destination.parent().unwrap())?;
        // Never silently overwrite an earlier human or assistant review.
        let bytes=serde_json::to_vec_pretty(&record)?;
        if destination.exists() {
            if fs::read(&destination)?!=bytes { return Err(format!("review already exists: {}",destination.display()).into()); }
        } else { fs::write(&destination,bytes)?; }
        reports.push(json!({"review":destination,"state":state,"observed_samples":evidence.len()}));
    }
    let report=json!({"schema":"buttercup-reflection-training-readiness-v1","reviews":reports,
        "status":"needs-localized-reflection-supervision",
        "training_started":false,"reflection_label_count":0,
        "reason":"Tentative assistant eyewear summaries are not admissible dense reflection targets under bootstrapability.md.",
        "allowed_next_steps":["Canonical human review of native RAW temporal triplets with reflection-specific labels",
            "Pinned SAM3 weak masks, independently validated and explicitly defeasible",
            "Versioned native duplicate-reflection matcher using RAW only, with validated sparse weak targets and explicit abstention"],
        "split_policy":"Keep both eyes, neighboring recordings from this sitting, duplicate RAW and augmentations together; this six-recording sitting is one development group, not a train/test split."});
    fs::write(&args[2],serde_json::to_vec_pretty(&report)?)?;
    println!("{}",serde_json::to_string_pretty(&report)?); Ok(())
}
fn main() { if let Err(e)=run() { eprintln!("eyewear review: {e}"); std::process::exit(1); } }
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn labels_are_explicit_not_boolean_reflection_truth() {
        for s in ["probably-on","probably-off","mixed","unknown"] { assert_eq!(label(s).unwrap(),s); }
        for s in ["reflection","negative","on","0.95"] { assert!(label(s).is_err()); }
    }
}
