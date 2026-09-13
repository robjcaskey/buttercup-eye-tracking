//! RAW-only corpus inventory. File discovery never reads labels/predictions/models.
use super::{hash, u, Bundle, Result};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

/// Absolute sensor clock + eye + actual native crop, independent of container
/// offsets, extraction names, optional viewer metadata and recorded predictions.
pub fn identity(r: &Value) -> String {
    json!([
        r["timestamp_ns"],
        r["sequence"],
        r["label"],
        r["sensor_x"],
        r["sensor_y"],
        r["width"],
        r["height"],
        r["stride"],
        r["pixel_format"]
    ])
    .to_string()
}
fn compatible(r: &Value) -> bool {
    [
        "sequence",
        "timestamp_ns",
        "sensor_x",
        "sensor_y",
        "width",
        "height",
        "stride",
        "offset",
        "length",
    ]
    .iter()
    .all(|k| u(r, k).is_ok())
        && matches!(r["label"].as_str(), Some("subject-right" | "subject-left"))
        && r["stream"] == format!("{}.raw10", r["label"].as_str().unwrap())
        && r["pixel_format"] == "RAW10_LE40_1X1"
        && r["recording_start_snapshot"] != true
}
pub struct Candidate {
    pub path: PathBuf,
    pub allowed: BTreeSet<String>,
}
pub struct Inventory {
    pub selected: Vec<Candidate>,
    pub report: Value,
}
fn discover(
    path: &Path,
    excluded: &Path,
    dirs: &mut BTreeSet<PathBuf>,
    bundles: &mut BTreeSet<PathBuf>,
    raws: &mut BTreeSet<PathBuf>,
    errors: &mut Vec<Value>,
) {
    let canonical = match fs::canonicalize(path) {
        Ok(p) => p,
        Err(e) => {
            errors.push(json!({"path":path,"reason":e.to_string()}));
            return;
        }
    };
    if canonical.starts_with(excluded) {
        return;
    }
    let meta = match fs::metadata(&canonical) {
        Ok(m) => m,
        Err(e) => {
            errors.push(json!({"path":canonical,"reason":e.to_string()}));
            return;
        }
    };
    if meta.is_dir() {
        if !dirs.insert(canonical.clone()) {
            return;
        }
        let entries = match fs::read_dir(&canonical) {
            Ok(e) => e,
            Err(e) => {
                errors.push(json!({"path":canonical,"reason":e.to_string()}));
                return;
            }
        };
        for entry in entries {
            match entry {
                Ok(e) => discover(&e.path(), excluded, dirs, bundles, raws, errors),
                Err(e) => errors.push(json!({"path":canonical,"reason":e.to_string()})),
            }
        }
    } else if meta.is_file() {
        if canonical.extension().is_some_and(|s| s == "tar") {
            bundles.insert(canonical);
        } else if canonical.file_name().is_some_and(|s| s == "frames.jsonl") {
            bundles.insert(canonical.parent().unwrap().to_owned());
        } else if canonical.extension().is_some_and(|s| s == "raw10") {
            raws.insert(canonical);
        }
    }
}
pub fn inventory(roots: &[PathBuf], out: &Path) -> Result<Inventory> {
    let mut dirs = BTreeSet::new();
    let mut bundles = BTreeSet::new();
    let mut raws = BTreeSet::new();
    let mut errors = Vec::new();
    let excluded = fs::canonicalize(out)?;
    for root in roots {
        discover(
            root,
            &excluded,
            &mut dirs,
            &mut bundles,
            &mut raws,
            &mut errors,
        );
    }
    let mut bundles: Vec<_> = bundles.into_iter().collect();
    bundles.sort_by_key(|p| {
        let text = p.to_string_lossy();
        (
            if p.extension().is_some_and(|s| s == "tar") {
                if text.contains("/calibration-corpus/") {
                    0
                } else {
                    1
                }
            } else {
                2
            },
            p.clone(),
        )
    });
    let discovered = bundles.len();
    let mut captures = Vec::new();
    let mut selected = Vec::new();
    let mut seen: BTreeMap<String, (String, PathBuf)> = BTreeMap::new();
    let mut known_hashes: BTreeSet<String> = BTreeSet::new();
    let mut read_bytes = 0u64;
    let mut input_records = 0usize;
    let mut duplicate_records = 0usize;
    let mut invalid_records = 0usize;
    let mut conflicting = 0usize;
    let mut progress = File::create(out.join("inventory-progress.jsonl"))?;
    for (i, path) in bundles.iter().enumerate() {
        eprintln!("RAW inventory {}/{discovered}: {}", i + 1, path.display());
        let mut allowed = BTreeSet::new();
        let mut duplicate = 0;
        let mut invalid = 0;
        let mut conflicts = Vec::new();
        let outcome = (|| -> Result<Value> {
            let bundle = Bundle::open(path)?;
            let index = bundle.read("frames.jsonl", 0, None)?;
            let rows: Vec<Value> = String::from_utf8(index.clone())?
                .lines()
                .map(serde_json::from_str)
                .collect::<std::result::Result<_, _>>()?;
            input_records += rows.len();
            for r in &rows {
                if !compatible(r) {
                    invalid += 1;
                    continue;
                }
                let bytes = match bundle.read(
                    r["stream"].as_str().unwrap(),
                    u(r, "offset")?,
                    Some(u(r, "length")? as usize),
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        invalid += 1;
                        conflicts.push(json!({"sequence":r["sequence"],"reason":e.to_string()}));
                        continue;
                    }
                };
                read_bytes += bytes.len() as u64;
                let digest = hash(&bytes);
                let id = identity(r);
                known_hashes.insert(digest.clone());
                if let Some((old, first)) = seen.get(&id) {
                    if old == &digest {
                        duplicate += 1;
                    } else {
                        conflicts.push(json!({"sequence":r["sequence"],"timestamp_ns":r["timestamp_ns"],"reason":"same_acquisition_identity_different_RAW_bytes","first_container":first,"first_sha256":old,"conflicting_sha256":digest}));
                        conflicting += 1;
                    }
                } else {
                    seen.insert(id.clone(), (digest, path.clone()));
                    allowed.insert(id);
                }
            }
            Ok(
                json!({"path":path,"frames_index_sha256":hash(&index),"index_records":rows.len(),"unique_compatible_sources":allowed.len(),"duplicate_compatible_sources":duplicate,"unsupported_or_missing_records":invalid,"input_conflicts":conflicts,"status":if !allowed.is_empty(){"selected_bounded_windows"}else if duplicate>0{"duplicate_sources_only"}else{"no_compatible_sources"}}),
            )
        })();
        let record = match outcome {
            Ok(v) => v,
            Err(e) => {
                json!({"path":path,"status":"unsupported_or_malformed_bundle","reason":e.to_string()})
            }
        };
        duplicate_records += duplicate;
        invalid_records += invalid;
        writeln!(progress, "{}", serde_json::to_string(&record)?)?;
        progress.flush()?;
        if !allowed.is_empty() {
            selected.push(Candidate {
                path: path.clone(),
                allowed,
            });
        }
        captures.push(record);
    }
    // Standalone files lack a sequence index; retain them as an explicit
    // ineligible inventory even when their content duplicates a known source.
    let mut isolated = Vec::new();
    for path in raws {
        let parent = path.parent().unwrap();
        if parent.join("frames.jsonl").is_file()
            && matches!(
                path.file_name().and_then(|s| s.to_str()),
                Some("subject-left.raw10" | "subject-right.raw10")
            )
        {
            continue;
        }
        let size = fs::metadata(&path)?.len();
        let digest = if size <= 64 * 1024 * 1024 {
            Some(hash(&fs::read(&path)?))
        } else {
            None
        };
        let duplicate = digest.as_ref().is_some_and(|h| known_hashes.contains(h));
        isolated.push(json!({"path":path,"bytes":size,"sha256":digest,"duplicates_indexed_frame_content":duplicate,"parallax_eligible":false,"reason":if size>64*1024*1024{"unindexed_large_RAW_stream_no_temporal_metadata"}else{"isolated_RAW_without_temporal_index"}}));
    }
    let report = json!({"schema":"buttercup-raw-parallax-inventory-v1","requested_roots":roots,"canonical_roots":roots.iter().map(|p|fs::canonicalize(p).ok()).collect::<Vec<_>>(),"excluded_current_output":excluded,"discovered_canonical_bundles":discovered,"selected_unique_source_containers":selected.len(),"indexed_source_records":input_records,"unique_compatible_source_crops":seen.len(),"duplicate_source_records":duplicate_records,"unsupported_or_missing_records":invalid_records,"conflicting_source_identities":conflicting,"raw_bytes_hashed":read_bytes,"captures":captures,"isolated_raw_files":isolated,"discovery_errors":errors,"deduplication":"canonical container paths, then sensor timestamp/sequence/eye/geometry/pixel-format identity verified by exact RAW byte hash; prefer original calibration tar over diagnostic/extracted copies","selection":"at most3 acquisition-only17-frame windows per eye per selected unique-source container; no full-frame exhaustive detection","not_read":"predictions,models,labels,review guesses or other checkout resources"});
    fs::write(
        out.join("inventory.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    // Separately persisted per-source manifest binds inventory to material bytes.
    let mut sources = File::create(out.join("unique-sources.jsonl"))?;
    for (id, (digest, path)) in &seen {
        writeln!(
            sources,
            "{}",
            json!({"acquisition_identity":serde_json::from_str::<Value>(id)?,"raw_sha256":digest,"first_container":path})
        )?;
    }
    Ok(Inventory { selected, report })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_source_ignores_container_offsets_and_optional_metadata() {
        let mut a = json!({"timestamp_ns":100,"sequence":1,"label":"subject-right","sensor_x":12,"sensor_y":16,"width":64,"height":64,"stride":80,"pixel_format":"RAW10_LE40_1X1","offset":0});
        let mut b = a.clone();
        b["offset"] = json!(2000);
        b["source_clock"] = json!({"other":"metadata"});
        assert_eq!(identity(&a), identity(&b));
        a["sensor_x"] = json!(16);
        assert_ne!(identity(&a), identity(&b));
    }
}
