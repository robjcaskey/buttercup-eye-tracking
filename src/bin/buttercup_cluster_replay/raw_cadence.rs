//! Read acquisition indexes to find retained high-cadence RAW, separately per eye.
//! This inventory does not infer optical exposure time from a host display rate.
use super::{json, quantiles, BundleSource, Error, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{self, BufRead},
    path::Path,
};
type Result<T> = std::result::Result<T, Error>;
fn number(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str()?.parse().ok())
}
fn summarize(rows: &[Value]) -> Option<Value> {
    if rows.len() < 2 {
        return None;
    }
    let times = rows
        .iter()
        .map(|r| number(&r["timestamp_ns"]))
        .collect::<Option<Vec<_>>>()?;
    if times.windows(2).any(|p| p[1] <= p[0]) {
        return None;
    }
    let intervals = times
        .windows(2)
        .map(|p| (p[1] - p[0]) as f64 / 1e6)
        .collect::<Vec<_>>();
    let interval_ms = quantiles(intervals.clone());
    let seconds = (times.last()? - times[0]) as f64 / 1e9;
    let seqs = rows
        .iter()
        .map(|r| number(&r["sequence"]))
        .collect::<Option<Vec<_>>>()?;
    let missing = seqs
        .windows(2)
        .map(|p| p[1].saturating_sub(p[0]).saturating_sub(1))
        .sum::<u64>();
    Some(
        json!({"frames":rows.len(),"seconds":seconds,"retained_fps":(rows.len()-1) as f64/seconds,
        "median_interval_fps":1000./interval_ms["p50"].as_f64()?,"interval_ms":interval_ms,"gaps_over_33ms":intervals.iter().filter(|&&v|v>33.333334).count(),
        "intervals_at_least_60fps":intervals.iter().filter(|&&v|v<=16.666667).count(),"missing_sequence_numbers":missing,
        "first_source":rows[0],"last_source":rows.last()?,"width":rows[0]["width"],"height":rows[0]["height"],"pixel_format":rows[0]["pixel_format"]}),
    )
}
fn inspect(path: &Path) -> Result<Vec<Value>> {
    let root = if path.file_name().is_some_and(|n| n == "frames.jsonl") {
        path.parent().ok_or("index parent")?
    } else {
        path
    };
    let bundle = BundleSource::open(root)?;
    let manifest: Value = serde_json::from_slice(&bundle.read_entry("manifest.json")?)?;
    if manifest["schema"] != "buttercup-raw-eye-bundle-v1" {
        return Err("not a native RAW-eye bundle manifest".into());
    }
    if root.is_dir() && fs::metadata(root.join("frames.jsonl"))?.len() > 64 * 1024 * 1024 {
        return Err("index exceeds bounded 64 MiB inventory limit".into());
    }
    let bytes = bundle.read_entry("frames.jsonl")?;
    let mut groups = BTreeMap::<(u64, String), Vec<Value>>::new();
    for line in std::str::from_utf8(&bytes)?.lines() {
        let row: Value = serde_json::from_str(line)?;
        let Some(eye) = number(&row["eye_id"]) else {
            continue;
        };
        if row["pixel_format"] != "RAW10_LE40_1X1"
            || number(&row["timestamp_ns"]).is_none()
            || row["stream"].as_str().is_none()
        {
            continue;
        }
        let epoch = row["source_clock"]["source_key"]["stream_epoch"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();
        groups.entry((eye, epoch)).or_default().push(row);
    }
    let mut result = Vec::new();
    for ((eye, epoch), rows) in groups {
        let mut unique = BTreeSet::new();
        let mut duplicates = 0;
        let mut segments = vec![Vec::new()];
        for row in rows {
            let t = number(&row["timestamp_ns"]).unwrap();
            if !unique.insert(t) {
                duplicates += 1;
                continue;
            }
            let current = segments.last_mut().unwrap();
            if current
                .last()
                .is_some_and(|r: &Value| number(&r["timestamp_ns"]).unwrap() > t)
            {
                segments.push(Vec::new());
            }
            segments.last_mut().unwrap().push(row);
        }
        for (segment, rows) in segments.iter().enumerate() {
            let Some(mut summary) = summarize(rows) else {
                continue;
            };
            // Verify indexed RAW ranges exist, without pretending this inventory
            // checks every pixel or establishes anatomical image quality.
            for row in rows {
                let start = number(&row["offset"]).ok_or("offset")?;
                let length = number(&row["length"]).ok_or("length")?;
                if length == 0 {
                    return Err("empty RAW frame".into());
                }
                bundle.read_range(row["stream"].as_str().unwrap(), start + length - 1, 1)?;
            }
            summary["bundle"] = json!(root);
            summary["eye_id"] = json!(eye);
            summary["epoch"] = json!(epoch);
            summary["segment"] = json!(segment);
            summary["duplicate_timestamps_in_epoch"] = json!(duplicates);
            result.push(summary);
        }
    }
    Ok(result)
}
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        return Err("--raw-cadence NEW_REPORT_JSON < paths.txt".into());
    }
    let out = Path::new(&args[2]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("new checked output required".into());
    }
    let mut rows = Vec::new();
    let mut errors = Vec::new();
    let mut seen = BTreeSet::new();
    let mut paths = 0;
    for line in io::stdin().lock().lines() {
        let path = line?;
        let p = Path::new(&path);
        let canonical = fs::canonicalize(p)?;
        if !seen.insert(canonical) {
            continue;
        }
        paths += 1;
        eprintln!("cadence {paths}: {}", p.display());
        match inspect(p) {
            Ok(mut a) => rows.append(&mut a),
            Err(e) => errors.push(json!({"path":path,"error":e.to_string()})),
        }
    }
    rows.sort_by(|a, b| {
        b["retained_fps"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["retained_fps"].as_f64().unwrap())
            .then_with(|| b["frames"].as_u64().cmp(&a["frames"].as_u64()))
    });
    fs::write(
        out,
        serde_json::to_vec_pretty(
            &json!({"schema":"buttercup-raw-cadence-v1","paths_examined":paths,"segments":rows,"errors":errors,
        "basis":"native source timestamp intervals per eye and stream epoch; duplicate timestamps excluded; backwards clocks split; RAW indexed ranges checked",
        "limitations":"Source timestamps are acquisition metadata, not independently measured optical exposure clocks. Content freshness, sharpness and vessel visibility are not established. Duplicate exports can appear as distinct bundle paths."}),
        )?,
    )?;
    println!("Cadence inventory: {paths} paths; {}", out.display());
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_cadence_does_not_turn_a_sequence_gap_into_fresh_frames() {
        let rows = [(0, 1), (8_000_000, 2), (16_000_000, 3), (40_000_000, 6)]
            .map(|(t, s)| json!({"timestamp_ns":t,"sequence":s}));
        let a = summarize(&rows).unwrap();
        assert_eq!(a["retained_fps"], 75.);
        assert_eq!(a["median_interval_fps"], 125.);
        assert_eq!(a["missing_sequence_numbers"], 2);
        assert!(summarize(&[rows[0].clone(), rows[0].clone()]).is_none());
    }
}
