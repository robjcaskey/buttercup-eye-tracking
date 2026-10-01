//! Source-native, source-clock-disjoint student example selection.
//! No model, GPU, decoded image, or recorded prediction is consulted.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn partition(lineage: &str) -> &'static str {
    let digest = Sha256::digest(lineage.as_bytes());
    match u32::from_be_bytes(digest[..4].try_into().unwrap()) % 10 {
        0 => "test",
        1 => "validation",
        _ => "train",
    }
}

fn string<'a>(row: &'a Value, field: &str) -> Result<&'a str> {
    row[field]
        .as_str()
        .ok_or_else(|| format!("{field} must be a string").into())
}

// Source index integers may be JSON numbers or decimal strings, as in Python int().
fn integer(value: &Value) -> Result<i128> {
    if let Some(n) = value.as_i64() {
        return Ok(n.into());
    }
    if let Some(n) = value.as_u64() {
        return Ok(n.into());
    }
    if let Some(s) = value.as_str() {
        return Ok(s.trim().parse()?);
    }
    Err("expected an integer or decimal integer string".into())
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Eye {
    Number(i128),
    Text(String),
}

fn eye(value: &Value) -> Result<Eye> {
    match value.as_str() {
        Some(s) => Ok(Eye::Text(s.to_owned())),
        None => Ok(Eye::Number(integer(value)?)),
    }
}

struct Options {
    index: PathBuf,
    output: PathBuf,
    per_eye: usize,
    max_sessions: usize,
}

fn prepare(options: &Options) -> Result<Value> {
    if !(1..=200).contains(&options.per_eye) {
        return Err("per-eye-session must be 1..200".into());
    }
    let root = fs::canonicalize("outputs")?;
    if !root.starts_with("/mnt/bulk_data/buttercup-eye-tracking") {
        return Err("outputs must resolve under /mnt/bulk_data/buttercup-eye-tracking".into());
    }
    let parent = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !fs::canonicalize(parent)?.starts_with(&root) {
        return Err("output must be under outputs".into());
    }
    // Optional acquisition stratum (e.g. displayed calibration target): the
    // per-eye cap then applies per session, eye and stratum. Absent strata keep
    // the original per session and eye behaviour.
    let mut groups: BTreeMap<(String, Eye, String), Vec<Value>> = BTreeMap::new();
    let mut seen = HashSet::new();
    let mut sizes = HashMap::new();
    let mut unavailable = 0usize;
    for line in BufReader::new(File::open(&options.index)?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let path = string(&row, "raw_file")?;
        let size = *sizes.entry(path.to_owned()).or_insert_with(|| {
            fs::metadata(path)
                .ok()
                .filter(|m| m.is_file())
                .map(|m| m.len())
                .unwrap_or(0)
        });
        let offset = integer(&row["raw_offset"])?;
        let length = integer(&row["raw_length"])?;
        if offset < 0 || length < 0 {
            return Err("RAW offset and length must be nonnegative".into());
        }
        if offset.checked_add(length).ok_or("RAW range overflow")? > i128::from(size) {
            unavailable += 1;
            continue;
        }
        let digest = string(&row, "raw_sha256")?.to_owned();
        if !seen.insert(digest) {
            continue;
        }
        let key = (
            string(&row, "clock_lineage")?.to_owned(),
            eye(&row["frame"]["eye_id"])?,
            row["student_stratum"].as_str().unwrap_or_default().to_owned(),
        );
        // Validate timestamps before sorting; malformed inputs must fail, never silently reorder.
        integer(&row["frame"]["timestamp_ns"])?;
        groups.entry(key).or_default().push(row);
    }
    let mut lineages: Vec<_> = groups
        .keys()
        .map(|(lineage, _, _)| lineage.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    lineages.sort_by_cached_key(|lineage| hash(lineage.as_bytes()));
    if options.max_sessions > 0 {
        lineages.truncate(options.max_sessions);
    }
    let allowed: HashSet<_> = lineages.into_iter().collect();
    let mut selected = Vec::new();
    for ((lineage, _, _), mut rows) in groups {
        if !allowed.contains(&lineage) {
            continue;
        }
        rows.sort_by_key(|row| integer(&row["frame"]["timestamp_ns"]).unwrap());
        let count = rows.len().min(options.per_eye);
        for i in 0..count {
            let mut row = rows[i * (rows.len() - 1) / (count - 1).max(1)].clone();
            row["student_split"] = json!(partition(&lineage));
            selected.push(row);
        }
    }
    let mut dest = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&options.output)?,
    );
    let mut handles = HashMap::new();
    let mut splits: BTreeMap<String, usize> = BTreeMap::new();
    for (index, row) in selected.iter_mut().enumerate() {
        let path = string(row, "raw_file")?.to_owned();
        if !handles.contains_key(&path) {
            handles.insert(path.clone(), File::open(&path)?);
        }
        let stream = handles.get_mut(&path).unwrap();
        let offset = u64::try_from(integer(&row["raw_offset"])?)?;
        let length = u64::try_from(integer(&row["raw_length"])?)?;
        stream.seek(SeekFrom::Start(offset))?;
        let mut limited = stream.take(length);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        let mut read = 0u64;
        loop {
            let n = limited.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            read += n as u64;
            hasher.update(&buffer[..n]);
        }
        if read != length || format!("{:x}", hasher.finalize()) != string(row, "raw_sha256")? {
            return Err("RAW hash mismatch".into());
        }
        row["student_index"] = json!(index);
        serde_json::to_writer(&mut dest, row)?;
        dest.write_all(b"\n")?;
        *splits
            .entry(string(row, "student_split")?.to_owned())
            .or_default() += 1;
    }
    dest.flush()?;
    Ok(
        json!({"frames": selected.len(), "sessions": allowed.len(), "splits": splits,
        "unavailable_source_rows": unavailable, "output": options.output}),
    )
}

fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Options>> {
    let mut args = args.into_iter();
    let mut positional = Vec::new();
    let mut per_eye = 20usize;
    let mut max_sessions = 0usize;
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("Usage: buttercup_prepare_eye_student INDEX OUTPUT [--per-eye-session 20] [--max-sessions 0]\nSelect source-native, source-clock-disjoint examples and verify selected RAW hashes.\nRows with student_stratum are capped per session, eye and stratum.");
            return Ok(None);
        }
        let (name, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(a, b)| (a, Some(b)));
        match name {
            "--per-eye-session" | "--max-sessions" => {
                let value = inline
                    .map(str::to_owned)
                    .or_else(|| args.next())
                    .ok_or("missing option value")?;
                let value: i64 = value.parse()?;
                if name == "--per-eye-session" {
                    if !(1..=200).contains(&value) {
                        return Err("per-eye-session must be 1..200".into());
                    }
                    per_eye = value as usize;
                } else {
                    max_sessions = value.max(0) as usize;
                }
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option: {arg}").into()),
            _ => positional.push(PathBuf::from(arg)),
        }
    }
    if positional.len() != 2 {
        return Err("expected INDEX OUTPUT (use --help)".into());
    }
    Ok(Some(Options {
        index: positional.remove(0),
        output: positional.remove(0),
        per_eye,
        max_sessions,
    }))
}

fn main() {
    let result = parse(std::env::args().skip(1)).and_then(|options| match options {
        Some(options) => {
            println!("{}", prepare(&options)?);
            Ok(())
        }
        None => Ok(()),
    });
    if let Err(error) = result {
        eprintln!("buttercup_prepare_eye_student: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = PathBuf::from("outputs").join(format!(
                "rust-student-prepare-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn options(&self) -> Options {
            Options {
                index: self.0.join("index.jsonl"),
                output: self.0.join("selected.jsonl"),
                per_eye: 2,
                max_sessions: 0,
            }
        }
        fn rows(&self, rows: &[Value]) {
            let mut file = File::create(self.options().index).unwrap();
            for row in rows {
                writeln!(file, "{row}").unwrap();
            }
        }
        fn row(&self, lineage: &str, eye: u8, stamp: u64) -> Value {
            let bytes = format!("{lineage}/{eye}/{stamp}");
            let path = self
                .0
                .join(format!("{}-{eye}-{stamp}.raw", hash(lineage.as_bytes())));
            fs::write(&path, &bytes).unwrap();
            json!({"raw_file": path,"raw_offset": "0","raw_length": bytes.len(), "raw_sha256": hash(bytes.as_bytes()),
                "clock_lineage": lineage,"frame": {"eye_id":eye,"timestamp_ns":stamp.to_string()}})
        }
        fn selected(&self) -> Vec<Value> {
            BufReader::new(File::open(self.options().output).unwrap())
                .lines()
                .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
                .collect()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn stratum_caps_each_target_separately_within_one_session_and_eye() {
        let f = Fixture::new();
        let mut rows = vec![];
        // One long-held target (six frames) and one short target (two frames).
        for stamp in 1..=6 {
            let mut row = f.row("session", 1, stamp);
            row["student_stratum"] = json!("calibration-target-0.90-0.10");
            rows.push(row);
        }
        for stamp in 7..=8 {
            let mut row = f.row("session", 1, stamp);
            row["student_stratum"] = json!("calibration-target-0.10-0.10");
            rows.push(row);
        }
        f.rows(&rows);
        let result = prepare(&f.options()).unwrap();
        assert_eq!(result["frames"], 4, "two per stratum, not four from the longer hold");
        let chosen = f.selected();
        for stratum in ["calibration-target-0.90-0.10", "calibration-target-0.10-0.10"] {
            assert_eq!(chosen.iter().filter(|r| r["student_stratum"] == stratum).count(), 2);
        }
        assert!(chosen.iter().all(|r| r["student_split"] == partition("session")));
    }

    #[test]
    fn native_sampling_dedup_and_both_eyes() {
        let f = Fixture::new();
        let mut rows = vec![];
        for eye in [1, 0] {
            for stamp in [30, 10, 20] {
                rows.push(f.row("source α", eye, stamp));
            }
        }
        let mut duplicate = rows[0].clone();
        duplicate["clock_lineage"] = json!("other-session");
        rows.push(duplicate);
        let mut unavailable = rows[0].clone();
        unavailable["raw_length"] = json!(9999);
        rows.push(unavailable);
        f.rows(&rows);
        let result = prepare(&f.options()).unwrap();
        assert_eq!(result["frames"], 4);
        assert_eq!(result["sessions"], 1);
        assert_eq!(result["unavailable_source_rows"], 1);
        let chosen = f.selected();
        for (i, row) in chosen.iter().enumerate() {
            assert_eq!(row["student_index"], i);
            assert_eq!(row["frame"]["eye_id"], i / 2);
            assert_eq!(
                row["frame"]["timestamp_ns"],
                if i % 2 == 0 { "10" } else { "30" }
            );
            assert_eq!(row["student_split"], partition("source α"));
        }
        assert!(
            prepare(&f.options()).is_err(),
            "must never overwrite output"
        );
    }

    #[test]
    fn deterministic_session_limit_and_hash_rejection() {
        let f = Fixture::new();
        let rows: Vec<_> = ["a", "b", "c"].iter().map(|s| f.row(s, 0, 0)).collect();
        f.rows(&rows);
        let mut options = f.options();
        options.max_sessions = 1;
        prepare(&options).unwrap();
        let expected = ["a", "b", "c"]
            .into_iter()
            .min_by_key(|s| hash(s.as_bytes()))
            .unwrap();
        assert_eq!(f.selected()[0]["clock_lineage"], expected);
        options.output = f.0.join("bad.jsonl");
        options.max_sessions = 0;
        fs::write(rows[0]["raw_file"].as_str().unwrap(), b"xxxxx").unwrap();
        assert!(prepare(&options)
            .unwrap_err()
            .to_string()
            .contains("RAW hash mismatch"));
    }

    #[test]
    fn unsafe_ranges_and_output_paths_rejected() {
        let f = Fixture::new();
        let mut row = f.row("a", 0, 0);
        row["raw_offset"] = json!(-1);
        f.rows(&[row]);
        assert!(prepare(&f.options())
            .unwrap_err()
            .to_string()
            .contains("nonnegative"));
        let mut options = f.options();
        options.output = PathBuf::from("/tmp/forbidden-student.jsonl");
        assert!(prepare(&options)
            .unwrap_err()
            .to_string()
            .contains("under outputs"));
    }

    #[test]
    fn cli_bounds() {
        assert!(parse(["i", "o", "--per-eye-session", "201"].map(str::to_owned)).is_err());
        let args = parse(["i", "o", "--max-sessions=-2", "--per-eye-session=1"].map(str::to_owned))
            .unwrap()
            .unwrap();
        assert_eq!(args.max_sessions, 0);
        assert_eq!(args.per_eye, 1);
    }

    #[test]
    fn all_split_roles_match_python_reference_and_do_not_split_eyes() {
        // Golden values obtained from the original Python partition function.
        assert_eq!(partition("0"), "train");
        assert_eq!(partition("6"), "validation");
        assert_eq!(partition("11"), "test");
        let f = Fixture::new();
        let mut rows = Vec::new();
        for lineage in ["0", "6", "11"] {
            for eye in [0, 1] {
                rows.push(f.row(lineage, eye, 0));
            }
        }
        f.rows(&rows);
        let result = prepare(&f.options()).unwrap();
        assert_eq!(
            result["splits"],
            json!({"train": 2, "validation": 2, "test": 2})
        );
        for row in f.selected() {
            assert_eq!(
                row["student_split"],
                partition(row["clock_lineage"].as_str().unwrap())
            );
        }
        let link = f.0.join("escape");
        std::os::unix::fs::symlink("/tmp", &link).unwrap();
        let mut options = f.options();
        options.output = link.join("student-escape.jsonl");
        assert!(prepare(&options)
            .unwrap_err()
            .to_string()
            .contains("under outputs"));
    }
}
