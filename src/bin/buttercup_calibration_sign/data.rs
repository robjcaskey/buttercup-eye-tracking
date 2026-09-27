use super::Result;
use buttercup_eye_tracking::recorded_bundle::{metadata_records, BundleSource};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn write(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
pub fn num(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str()?.parse().ok())
}
pub fn output(path: &str) -> Result<PathBuf> {
    let p = PathBuf::from(path);
    let parent = fs::canonicalize(p.parent().ok_or("output parent")?)?;
    if !parent.starts_with(fs::canonicalize("data")?) {
        return Err("outputs must use the checked runtime links".into());
    }
    fs::create_dir(&p)?;
    Ok(p)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Span {
    pub start: u64,
    pub end: u64,
    pub id: String,
    pub uv: [f32; 2],
    pub hidden_since: Option<u64>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Source {
    pub session: String,
    pub archive: String,
    pub session_sha256: Option<String>,
    pub frame_index_sha256: String,
    pub metadata_sha256: String,
    pub group: String,
    pub day: u64,
    pub frames: Vec<Value>,
    pub spans: Vec<Span>,
    pub eligible: Vec<[usize; 2]>,
    pub targets: Vec<usize>,
    pub excluded: BTreeMap<String, usize>,
}
fn count(m: &mut BTreeMap<String, usize>, reason: &str) {
    *m.entry(reason.into()).or_default() += 1;
}
fn target(v: &Value) -> Option<(String, [f32; 2])> {
    if v["mode"] != "mouse-calibration" {
        return None;
    }
    let t: Vec<_> = v["active_targets"]
        .as_array()?
        .iter()
        .filter(|t| t["visible"] == true && t["role"] == "calibration")
        .collect();
    if t.len() != 1 {
        return None;
    }
    let uv = [
        t[0]["normalized"][0].as_f64()? as f32,
        t[0]["normalized"][1].as_f64()? as f32,
    ];
    if uv
        .iter()
        .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
    {
        return None;
    }
    Some((t[0]["id"].as_str()?.into(), uv))
}
fn spans(bytes: &[u8]) -> Result<Vec<Span>> {
    let mut configs = HashMap::<(String, String), bool>::new();
    let mut result = Vec::new();
    let mut current: Option<Span> = None;
    let mut last = 0;
    for r in metadata_records(bytes) {
        let r = r?;
        let key = (
            r["viewer_session_id"].to_string(),
            r["configuration_revision"].to_string(),
        );
        let cfg = match r["event"].as_str() {
            Some("recording_start_snapshot" | "queue_recovery_snapshot" | "live_checkpoint") => {
                Some(&r["configuration"])
            }
            Some("configuration_changed") => Some(&r["data"]),
            _ => None,
        };
        if let Some(cfg) = cfg {
            let c = &cfg["calibration"];
            configs.insert(
                key.clone(),
                c["thumbnail_opacity"].as_f64() == Some(0.0)
                    && c["sample_gate_has_hidden_submission"] == true,
            );
        }
        if r["event"] != "presentation" {
            continue;
        }
        let time =
            num(&r["host_submit_end_unix_ns"]).ok_or("presentation missing host submission")?;
        if time < last {
            return Err("nonmonotonic presentation clock".into());
        }
        last = time;
        let t = target(&r);
        let same = current
            .as_ref()
            .zip(t.as_ref())
            .is_some_and(|(a, (id, uv))| a.id == *id && a.uv == *uv);
        if !same {
            if let Some(mut c) = current.take() {
                c.end = time;
                result.push(c);
            }
            current = t.map(|(id, uv)| Span {
                start: time,
                end: time,
                id,
                uv,
                hidden_since: None,
            });
        }
        if let Some(c) = current.as_mut() {
            c.end = time;
            if configs.get(&key) == Some(&true) {
                c.hidden_since.get_or_insert(time);
            } else {
                c.hidden_since = None;
            }
        }
    }
    if let Some(c) = current {
        result.push(c);
    }
    Ok(result)
}
pub fn pair_clock(a: &Value, b: &Value) -> bool {
    let ka = &a["source_clock"]["source_key"];
    let kb = &b["source_clock"]["source_key"];
    // Legacy archives are excluded when source identity is missing: a timestamp
    // or repeated sequence alone cannot establish consecutive fresh exposures.
    for key in [
        "viewer_session_id",
        "stream_epoch",
        "region_session",
        "roi_id",
    ] {
        if ka[key].is_null() || ka[key] != kb[key] {
            return false;
        }
    }
    let (Some(sa), Some(sb), Some(ta), Some(tb)) = (
        num(&a["sequence"]),
        num(&b["sequence"]),
        num(&a["timestamp_ns"]),
        num(&b["timestamp_ns"]),
    ) else {
        return false;
    };
    sb == sa + 1
        && tb > ta
        && tb - ta <= 250_000_000
        && num(&ka["sequence"]) == Some(sa)
        && num(&kb["sequence"]) == Some(sb)
        && num(&ka["sensor_timestamp_ns"]) == Some(ta)
        && num(&kb["sensor_timestamp_ns"]) == Some(tb)
        && num(&ka["roi_id"]) == num(&a["eye_id"])
        && num(&kb["roi_id"]) == num(&b["eye_id"])
}
fn load(path: &Path, corpus: &Path) -> Result<Source> {
    let orphan = path.extension().is_some_and(|v| v == "tar");
    let session_bytes = if orphan { None } else { Some(fs::read(path)?) };
    let session: Value = if let Some(bytes) = &session_bytes {
        serde_json::from_slice(bytes)?
    } else {
        json!({"raw_bundle":path})
    };
    let raw = session["raw_bundle"]
        .as_str()
        .ok_or("session has no RAW archive")?;
    let mut archive = PathBuf::from(raw);
    if !archive.exists() {
        archive = corpus.join(archive.file_name().ok_or("archive basename")?);
    }
    let bundle = BundleSource::open(&archive)?;
    let fb = bundle.read_entry("frames.jsonl")?;
    let mb = bundle.read_entry("metadata.oim1")?;
    let frames: Vec<Value> = std::str::from_utf8(&fb)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    let spans = spans(&mb)?;
    let day = num(&session["started_host_unix_ns"])
        .or_else(|| frames.first().and_then(|f| num(&f["host_arrival_unix_ns"])))
        .ok_or("missing session host clock")?
        / 86_400_000_000_000;
    let group = frames
        .iter()
        .find_map(|f| f["source_clock"]["source_key"]["viewer_session_id"].as_str())
        .unwrap_or("")
        .to_owned();
    let mut out = Source {
        session: path.display().to_string(),
        archive: archive.display().to_string(),
        session_sha256: session_bytes.as_ref().map(|b| digest(b)),
        frame_index_sha256: digest(&fb),
        metadata_sha256: digest(&mb),
        group,
        day,
        frames,
        spans,
        eligible: Vec::new(),
        targets: Vec::new(),
        excluded: BTreeMap::new(),
    };
    let mut previous = HashMap::<u64, usize>::new();
    for (i, f) in out.frames.iter().enumerate() {
        let Some(eye) = num(&f["eye_id"]) else {
            count(&mut out.excluded, "missing-eye");
            continue;
        };
        let before = previous.insert(eye, i);
        if f["pixel_format"] != "RAW10_LE40_1X1" {
            count(&mut out.excluded, "unsupported-native-format");
            continue;
        }
        let arrival = num(&f["host_arrival_unix_ns"])
            .or_else(|| num(&f["source_clock"]["host_arrival_unix_ns"]));
        let Some(time) = arrival else {
            count(&mut out.excluded, "missing-host-arrival");
            continue;
        };
        let owned: Vec<_> = out
            .spans
            .iter()
            .enumerate()
            .filter(|(_, s)| time >= s.start && time < s.end)
            .collect();
        if owned.len() != 1 {
            count(&mut out.excluded, "no-unique-visible-calibration-target");
            continue;
        }
        let (site, s) = owned[0];
        if time < s.start + 2_100_000_000 || time + 100_000_000 > s.end {
            count(&mut out.excluded, "acclimation-or-transition-guard");
            continue;
        }
        if !s.hidden_since.is_some_and(|h| time > h + 100_000_000) {
            count(&mut out.excluded, "hidden-thumbnail-submit-unattested");
            continue;
        }
        let Some(j) = before else {
            count(&mut out.excluded, "no-prior-eye-frame");
            continue;
        };
        let a = &out.frames[j];
        let at = num(&a["host_arrival_unix_ns"])
            .or_else(|| num(&a["source_clock"]["host_arrival_unix_ns"]))
            .unwrap_or(0);
        if at < s.start + 2_100_000_000
            || at >= s.end
            || !s.hidden_since.is_some_and(|h| at > h + 100_000_000)
        {
            count(&mut out.excluded, "prior-frame-outside-settled-target");
            continue;
        }
        if !pair_clock(a, f) {
            count(
                &mut out.excluded,
                "nonconsecutive-or-incompatible-source-clock",
            );
            continue;
        }
        out.eligible.push([j, i]);
        out.targets.push(site);
    }
    Ok(out)
}
pub fn scan(corpus: &str) -> Result<(Vec<Source>, Value)> {
    let corpus = Path::new(corpus);
    let mut paths: Vec<_> = fs::read_dir(corpus)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".session.json"))
        .collect();
    paths.sort();
    let session_count = paths.len();
    let mut orphan: Vec<_> = fs::read_dir(corpus)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "tar")
                && !paths.iter().any(|s| {
                    s.file_name()
                        .unwrap()
                        .to_string_lossy()
                        .replace(".session.json", ".tar")
                        == p.file_name().unwrap().to_string_lossy()
                })
        })
        .collect();
    orphan.sort();
    paths.extend(orphan.iter().cloned());
    let total = paths.len();
    let mut sources = Vec::new();
    let mut failures = Vec::new();
    for (i, path) in paths.iter().enumerate() {
        match load(path, corpus) {
            Ok(s) => {
                eprintln!(
                    "inventory {}/{total}: {} RAW, {} pairs, {} targets: {}",
                    i + 1,
                    s.frames.len(),
                    s.eligible.len(),
                    s.spans.len(),
                    path.file_name().unwrap().to_string_lossy()
                );
                sources.push(s);
            }
            Err(e) => {
                failures.push(json!({"session":path,"reason":e.to_string()}));
                eprintln!("inventory {}/{total}: excluded: {e}", i + 1);
            }
        }
    }
    let mut reasons = BTreeMap::<String, usize>::new();
    for s in &sources {
        for (k, v) in &s.excluded {
            *reasons.entry(k.clone()).or_default() += v;
        }
    }
    let report = json!({"schema":"buttercup-calibration-sign-inventory-v1","sessions_considered":session_count,"total_entries_considered":total,"entries_read":sources.len(),"archives_without_session_also_scanned":orphan,"session_failures":failures,"raw_frames":sources.iter().map(|s|s.frames.len()).sum::<usize>(),"eligible_pairs":sources.iter().map(|s|s.eligible.len()).sum::<usize>(),"excluded_frames":reasons,"sources":sources.iter().map(|s|json!({"session":s.session,"archive":s.archive,"day":s.day,"viewer_group":s.group,"raw_frames":s.frames.len(),"pairs":s.eligible.len(),"spans":s.spans,"exclusions":s.excluded,"metadata_sha256":s.metadata_sha256,"frame_index_sha256":s.frame_index_sha256,"session_sha256":s.session_sha256})).collect::<Vec<_>>(),"supervision":"recorded visible intended target, not verified fixation or signed 3D truth; no recorded predictions, accepted-fit flag, fitted display pose or custom weights used","pair_rule":"two consecutive fresh exposures of the same eye and source lineage, dt <=250ms; both after 2000ms acclimation +100ms display allowance and after hidden-thumbnail submit +100ms; current at least100ms before target removal","missing":"actual display scanout/exposure latency and framewise off-target fixation are not independently measured"});
    Ok((sources, report))
}
pub fn inventory(corpus: &str, out: &str) -> Result<()> {
    let out = output(out)?;
    let (_, report) = scan(corpus)?;
    write(out.join("inventory.json"), &report)
}
