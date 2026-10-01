//! Calibration-target strata source index for Eye Student / Butter Obelisk.
//!
//! Acquisition metadata only: the recorded presentation events say which
//! calibration target was on screen, and the native frame records address the
//! RAW bytes. No model, recorded prediction, gaze estimate or fitted geometry
//! is consulted, so selection adds no learned ancestor (bootstrapability.md).
//! A frame is tagged with a target only after the settle interval following
//! that target's presentation; the tag names what was shown, not where the
//! person actually looked.
//!
//! Output rows use the source-index schema read by
//! `buttercup_prepare_eye_student` plus `student_stratum`, so sampling can be
//! balanced per session, eye and target. Whole sessions share one lineage and
//! therefore one train/validation/test partition, both eyes included.
use buttercup_eye_tracking::recorded_bundle::{metadata_records, BundleSource};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Same value as the viewer's VIRTUAL_MOUSE_TARGET_SETTLE: frames earlier than
/// this after a target appears are excluded from calibration samples too.
const TARGET_SETTLE_NS: i128 = 2_100_000_000;

fn integer(value: &Value) -> Option<i128> {
    value
        .as_i64()
        .map(i128::from)
        .or_else(|| value.as_u64().map(i128::from))
        .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
}

/// (time, calibration target id and normalized position) for every change of
/// the first visible calibration target, from host buffer-submit times.
fn target_changes(bundle: &BundleSource) -> Result<Vec<(i128, Option<(String, [f64; 2])>)>> {
    let metadata = bundle.read_entry("metadata.oim1")?;
    let mut changes: Vec<(i128, Option<(String, [f64; 2])>)> = Vec::new();
    for record in metadata_records(metadata.as_slice()) {
        let record = record?;
        if record["event"] != "presentation" {
            continue;
        }
        let Some(time) = integer(&record["host_submit_end_unix_ns"]) else { continue };
        let visible = record["active_targets"]
            .as_array()
            .and_then(|targets| targets.iter().find(|t| t["visible"] == true));
        let target = visible.and_then(|t| {
            let id = t["id"].as_str()?;
            if !id.starts_with("calibration-") {
                return None;
            }
            let n = t["normalized"].as_array()?;
            Some((id.to_owned(), [n.first()?.as_f64()?, n.get(1)?.as_f64()?]))
        });
        if changes.last().is_none_or(|(_, last)| *last != target) {
            changes.push((time, target));
        }
    }
    Ok(changes)
}

fn stratum(position: [f64; 2]) -> String {
    format!("calibration-target-{:.2}-{:.2}", position[0], position[1])
}

struct Counts {
    sessions: usize,
    skipped_sessions: usize,
    rows: usize,
    duplicate_raw: usize,
}

fn index_session(
    session_path: &Path,
    out: &mut impl Write,
    seen: &mut HashSet<String>,
    counts: &mut Counts,
) -> Result<()> {
    let session: Value = serde_json::from_reader(File::open(session_path)?)?;
    let bundle_path = PathBuf::from(session["raw_bundle"].as_str().ok_or("session lacks raw_bundle")?);
    let bundle = BundleSource::open(&bundle_path)?;
    let changes = target_changes(&bundle)?;
    if changes.iter().all(|(_, target)| target.is_none()) {
        counts.skipped_sessions += 1;
        return Ok(());
    }
    let stem = session_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("session path lacks a file name")?
        .trim_end_matches(".session.json");
    let lineage = format!("calibration-session:{stem}");
    let frames = bundle.read_entry("frames.jsonl")?;
    let mut handle: Option<File> = None;
    for line in std::str::from_utf8(&frames)?.lines().filter(|l| !l.trim().is_empty()) {
        let frame: Value = serde_json::from_str(line)?;
        let Some(arrival) = integer(&frame["host_arrival_unix_ns"]) else { continue };
        // The target whose presentation most recently preceded this arrival.
        let current = changes.partition_point(|(time, _)| *time <= arrival);
        if current == 0 {
            continue;
        }
        let (shown_at, target) = &changes[current - 1];
        let Some((id, position)) = target else { continue };
        if arrival < shown_at + TARGET_SETTLE_NS {
            continue;
        }
        let stream = frame["stream"].as_str().ok_or("frame lacks stream")?;
        let (file, entry_offset, entry_size) = bundle.entry_location(stream)?;
        let offset = u64::try_from(integer(&frame["offset"]).ok_or("frame lacks offset")?)?;
        let length = u64::try_from(integer(&frame["length"]).ok_or("frame lacks length")?)?;
        if offset.checked_add(length).ok_or("RAW range overflow")? > entry_size {
            return Err(format!("{}: frame exceeds {stream}", session_path.display()).into());
        }
        let file_handle = match handle.as_mut() {
            Some(h) => h,
            None => handle.insert(File::open(&file)?),
        };
        file_handle.seek(SeekFrom::Start(entry_offset + offset))?;
        let mut bytes = vec![0u8; usize::try_from(length)?];
        file_handle.read_exact(&mut bytes)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if !seen.insert(digest.clone()) {
            counts.duplicate_raw += 1;
            continue;
        }
        let row = json!({
            "index": counts.rows,
            "capture_entry": 0,
            "clock_lineage": lineage,
            "clock_attested": false,
            "raw_file": file,
            "raw_offset": entry_offset + offset,
            "raw_length": length,
            "raw_sha256": digest,
            "frame": frame,
            "scale_hint": Value::Null,
            "student_stratum": stratum(*position),
            "calibration_target": {"id": id, "normalized": position,
                "presented_host_submit_unix_ns": shown_at.to_string(),
                "semantics": "displayed stimulus after settle; not measured gaze"},
            "calibration_session": session_path,
        });
        serde_json::to_writer(&mut *out, &row)?;
        out.write_all(b"\n")?;
        counts.rows += 1;
    }
    counts.sessions += 1;
    Ok(())
}

fn run(output: &Path, sessions: &[PathBuf]) -> Result<Value> {
    let root = fs::canonicalize("outputs")?;
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    if !fs::canonicalize(parent)?.starts_with(&root) {
        return Err("output must be under outputs".into());
    }
    let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(output)?);
    let mut seen = HashSet::new();
    let mut counts = Counts { sessions: 0, skipped_sessions: 0, rows: 0, duplicate_raw: 0 };
    let mut failed = Vec::new();
    for session in sessions {
        if let Err(error) = index_session(session, &mut out, &mut seen, &mut counts) {
            failed.push(json!({"session": session, "error": error.to_string()}));
        }
    }
    out.flush()?;
    Ok(json!({"output": output, "indexed_sessions": counts.sessions,
        "sessions_without_calibration_targets": counts.skipped_sessions,
        "failed_sessions": failed, "rows": counts.rows, "duplicate_raw": counts.duplicate_raw,
        "settle_ns": TARGET_SETTLE_NS.to_string()}))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: buttercup_calibration_strata OUTPUT.jsonl SESSION.session.json...\nIndex native RAW calibration frames by displayed target (after settle) for stratified student selection.");
        return;
    }
    let output = PathBuf::from(args.remove(0));
    let sessions: Vec<PathBuf> = args.into_iter().map(PathBuf::from).collect();
    match run(&output, &sessions) {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("buttercup_calibration_strata: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strata_names_are_stable_and_distinguish_the_grid() {
        assert_eq!(stratum([0.9, 0.1]), "calibration-target-0.90-0.10");
        assert_ne!(stratum([0.5, 0.9]), stratum([0.9, 0.5]));
    }

    #[test]
    fn integers_accept_numbers_and_decimal_strings() {
        assert_eq!(integer(&json!("1790556045225685425")), Some(1790556045225685425));
        assert_eq!(integer(&json!(42)), Some(42));
        assert_eq!(integer(&json!("x")), None);
    }
}
