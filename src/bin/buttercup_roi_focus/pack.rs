use super::archive::{self, DiskAreaRecord, Frame, Manifest, Source};
use crate::Result;
use buttercup_eye_tracking::recorded_bundle::BundleSource;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};
pub fn number(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str()?.parse().ok())
}
fn epoch(f: &Value) -> String {
    f.pointer("/source_clock/source_key/stream_epoch")
        .and_then(Value::as_str)
        .unwrap_or("")
        .into()
}
fn stamp(f: &Value) -> u64 {
    number(&f["source_clock"]["source_key"]["sensor_timestamp_ns"])
        .or_else(|| number(&f["timestamp_ns"]))
        .unwrap_or(0)
}
fn seq(f: &Value) -> u64 {
    number(&f["source_clock"]["source_key"]["sequence"])
        .or_else(|| number(&f["sequence"]))
        .unwrap_or(0)
}
fn walk(p: &Path, seen: &mut HashSet<PathBuf>, out: &mut Vec<PathBuf>) -> Result<()> {
    let real = fs::canonicalize(p)?;
    if !seen.insert(real) {
        return Ok(());
    }
    if p.is_file() {
        if p.extension().is_some_and(|x| x == "tar") {
            out.push(p.into());
        }
        return Ok(());
    }
    if p.join("frames.jsonl").is_file() {
        out.push(p.into());
    }
    for entry in fs::read_dir(p)? {
        let e = entry?;
        let p = e.path();
        let name = e.file_name();
        if [
            "target",
            "build",
            ".git",
            "models",
            "weights",
            "node_modules",
        ]
        .iter()
        .any(|x| name == *x)
        {
            continue;
        }
        if e.file_type()?.is_dir() || p.extension().is_some_and(|x| x == "tar") {
            walk(&p, seen, out)?;
        }
    }
    Ok(())
}
fn valid(e: [f64; 5], f: &Frame) -> bool {
    e.iter().all(|x| x.is_finite())
        && e[3] >= 4.
        && e[2] >= e[3]
        && e[2] < (f.width.max(f.height) as f64) * 0.8
        && e[3] / e[2] >= 0.10
}
fn assign(f: &mut Frame, e: [f64; 5], provider: u8, quality: f32) -> bool {
    if !valid(e, f) {
        return false;
    }
    let conflict = f.flags & 1 != 0 && f.ellipse.iter().zip(e).any(|(a, b)| (a - b).abs() > 0.1);
    if f.flags & 1 == 0 || provider < f.provider {
        f.ellipse = e;
        f.provider = provider;
        f.flags |= 1;
        f.quality = quality;
        f.area = DiskAreaRecord::from_shape(f.shape());
    }
    conflict
}
pub fn run(output: &str, native_path: &str, roots: &[String]) -> Result<()> {
    let output = Path::new(output);
    if !output.is_absolute() || !output.starts_with("/tmp") {
        return Err("temporary archive must have an absolute /tmp path".into());
    }
    if output.exists() {
        return Err("refusing to overwrite existing temporary archive".into());
    }
    let native_bytes = fs::read(native_path)?;
    let native_hash = archive::digest(&native_bytes);
    // Match all crop/clock/sequence fields, not a rounded JSON numeric timestamp.
    let mut native = HashMap::new();
    for line in native_bytes
        .split(|b| *b == b'\n')
        .filter(|x| !x.is_empty())
    {
        let v: Value = serde_json::from_slice(line)?;
        let f = &v["source"]["frame"];
        if let Some(a) = v["fit"]["ellipse"].as_array().filter(|a| a.len() == 5) {
            let mut e = [0.; 5];
            for i in 0..5 {
                e[i] = a[i].as_f64().ok_or("native ellipse")?;
            }
            e[0] += number(&f["sensor_x"]).ok_or("native origin")? as f64;
            e[1] += number(&f["sensor_y"]).ok_or("native origin")? as f64;
            native.insert(
                (
                    epoch(f),
                    number(&f["eye_id"]).unwrap_or(0) as u16,
                    stamp(f),
                    seq(f),
                    number(&f["sensor_x"]).unwrap_or(0),
                    number(&f["sensor_y"]).unwrap_or(0),
                    number(&f["width"]).unwrap_or(0),
                    number(&f["height"]).unwrap_or(0),
                ),
                (e, v["admissible"] == true),
            );
        }
    }
    let mut paths = vec![];
    let mut seen = HashSet::new();
    for root in roots {
        walk(Path::new(root), &mut seen, &mut paths)?;
    }
    paths.sort();
    paths.dedup();
    let mut manifest=Manifest{schema:"buttercup-roi-ellipse-binary-v2".into(),sources:vec![],epochs:vec![],excluded:vec![],aliases:vec![],scale_references:vec![],native_conics_path:native_path.into(),native_conics_sha256:native_hash,assumptions:vec![
        "128-byte little-endian per-frame records plus JSON source/epoch dictionary; native video payloads are referenced, not copied or re-encoded".into(),
        "Every indexed ROI frame is retained, including missing ellipses; identical ellipse-history/metadata copies are represented by explicit aliases and never pooled as independent evidence".into(),
        "Priority: exact-source historical pinned SAM conic, source-bound archived semantic ellipse, source-bound archived virtual-contact ellipse reconstruction".into(),
        "Historical model outputs are diagnostic inputs only; no training, bootstrap-current claim or model promotion".into(),
        "Unprojection uses nominal focal=(4000,4000), principal=(4000,3000); equal iris radii across eyes. +Z toward camera; metric depth and optical-to-visual-axis offset unmeasured".into(),
        "Frames/prediction metadata hashed; original RAW payloads remain externally referenced. Duplicate metadata does not prove duplicate RAW".into()]};
    let mut all = vec![];
    let mut epoch_ids = HashMap::new();
    for (pi, path) in paths.iter().enumerate() {
        let bundle = match BundleSource::open(path) {
            Ok(b) => b,
            Err(e) => {
                manifest.excluded.push(json!({"path":path,"reason":e}));
                continue;
            }
        };
        let mut prefixes = match &bundle {
            BundleSource::Directory(_) => vec![String::new()],
            BundleSource::Tar { entries, .. } => entries
                .keys()
                .filter(|n| n.ends_with("frames.jsonl"))
                .map(|n| n[..n.len() - "frames.jsonl".len()].to_owned())
                .collect(),
        };
        prefixes.sort();
        if prefixes.is_empty() {
            manifest
                .excluded
                .push(json!({"path":path,"reason":"no frames.jsonl"}));
        }
        for prefix in prefixes {
            let bytes = match bundle.read_entry(&format!("{prefix}frames.jsonl")) {
                Ok(b) => b,
                Err(e) => {
                    manifest
                        .excluded
                        .push(json!({"path":path,"prefix":prefix,"reason":e}));
                    continue;
                }
            };
            let mut source = Source {
                path: fs::canonicalize(path)?.display().to_string(),
                prefix: prefix.clone(),
                frames_sha256: archive::digest(&bytes),
                predictions_sha256: None,
                streams: vec![],
                frames: 0,
                ellipse_counts: [0; 4],
                unmatched_predictions: 0,
                conflicting_ellipses: 0,
            };
            let mut frames = vec![];
            let mut keys = BTreeMap::new();
            let mut stream_ids = HashMap::new();
            for (index, line) in bytes
                .split(|b| *b == b'\n')
                .filter(|x| !x.is_empty())
                .enumerate()
            {
                let f: Value = match serde_json::from_slice(line) {
                    Ok(f) => f,
                    Err(e) => {
                        manifest.excluded.push(
                            json!({"path":path,"prefix":prefix,"row":index,"reason":e.to_string()}),
                        );
                        continue;
                    }
                };
                let Some(eye) = number(&f["eye_id"]).filter(|x| *x > 0 && *x <= u16::MAX as u64)
                else {
                    manifest
                        .excluded
                        .push(json!({"path":path,"row":index,"reason":"not an identified ROI"}));
                    continue;
                };
                let ep = epoch(&f);
                let epkey = if ep.is_empty() {
                    format!("unattested:{}:{prefix}", path.display())
                } else {
                    ep.clone()
                };
                let eid = *epoch_ids.entry(epkey.clone()).or_insert_with(|| {
                    manifest.epochs.push(epkey);
                    (manifest.epochs.len() - 1) as u32
                });
                let Some(stream) = f["stream"].as_str() else {
                    manifest
                        .excluded
                        .push(json!({"path":path,"row":index,"reason":"missing stream"}));
                    continue;
                };
                let sid = *stream_ids.entry(stream.to_owned()).or_insert_with(|| {
                    source.streams.push(stream.into());
                    (source.streams.len() - 1) as u32
                });
                let get = |k: &str| number(&f[k]).unwrap_or(0);
                let frame = Frame {
                    source: manifest.sources.len() as u32,
                    epoch: eid,
                    eye: eye as u16,
                    provider: 0,
                    flags: if ep.is_empty() { 0 } else { 4 },
                    sequence: seq(&f),
                    ns: stamp(&f),
                    host_ns: number(&f["source_clock"]["host_arrival_unix_ns"])
                        .unwrap_or(get("host_arrival_unix_ns")),
                    origin: [get("sensor_x") as u32, get("sensor_y") as u32],
                    width: get("width") as u32,
                    height: get("height") as u32,
                    stride: get("stride") as u32,
                    offset: get("offset"),
                    length: get("length") as u32,
                    stream: sid,
                    ellipse: [0.; 5],
                    quality: 0.,
                    index: index as u32,
                    area: DiskAreaRecord::missing(),
                };
                keys.entry((eid, eye as u16, frame.ns))
                    .or_insert(frames.len());
                frames.push(frame);
            }
            if let Ok(pb) = bundle.read_entry(&format!("{prefix}predictions.jsonl")) {
                source.predictions_sha256 = Some(archive::digest(&pb));
                for line in pb.split(|b| *b == b'\n').filter(|x| !x.is_empty()) {
                    let r: Value = match serde_json::from_slice(line) {
                        Ok(r) => r,
                        Err(_) => {
                            source.unmatched_predictions += 1;
                            continue;
                        }
                    };
                    let ep = epoch(&r);
                    let Some(&eid) = epoch_ids.get(&ep) else {
                        source.unmatched_predictions += 1;
                        continue;
                    };
                    let eye = number(&r["source_clock"]["source_key"]["roi_id"])
                        .or_else(|| number(&r["roi_frame_key"]["roi_id"]))
                        .unwrap_or(0) as u16;
                    let e = &r["predictions"]["eye_candidate"];
                    let sem = &e["limbus"]["semantic_ellipse"];
                    if sem.is_object() {
                        let ns = number(&r["source_clock"]["source_key"]["sensor_timestamp_ns"]);
                        if let Some(&i) = ns.and_then(|ns| keys.get(&(eid, eye, ns))) {
                            let vals = [
                                sem["center"][0].as_f64(),
                                sem["center"][1].as_f64(),
                                sem["major_radius"].as_f64(),
                                sem["minor_radius"].as_f64(),
                                sem["angle_rad"].as_f64(),
                            ];
                            if vals.iter().all(Option::is_some) {
                                let mut shape = vals.map(Option::unwrap);
                                shape[0] += frames[i].origin[0] as f64;
                                shape[1] += frames[i].origin[1] as f64;
                                source.conflicting_ellipses +=
                                    usize::from(assign(&mut frames[i], shape, 2, 0.5));
                            }
                        }
                    }
                    let contact = ["virtual_contact_surface_gaze", "surface_gaze"]
                        .iter()
                        .find_map(|k| {
                            e["centers_and_gaze"][*k]
                                .is_object()
                                .then_some(&e["centers_and_gaze"][*k])
                        });
                    if let Some(c) = contact {
                        let Some(ns) = number(&c["source_timestamp_ns"]) else {
                            source.unmatched_predictions += 1;
                            continue;
                        };
                        let Some(&i) = keys.get(&(eid, eye, ns)) else {
                            source.unmatched_predictions += 1;
                            continue;
                        };
                        let v = [
                            c["relative_gaze_vector"][0].as_f64(),
                            c["relative_gaze_vector"][1].as_f64(),
                            c["relative_gaze_vector"][2].as_f64(),
                            c["rectified_area_px2"].as_f64(),
                            c["bucketed_face_radius_px"].as_f64(),
                            c["camera_near_point_sensor"][0].as_f64(),
                            c["camera_near_point_sensor"][1].as_f64(),
                        ];
                        if v.iter().all(Option::is_some) {
                            let [nx, ny, nz, area, r, cx, cy] = v.map(Option::unwrap);
                            let a = (area / std::f64::consts::PI).sqrt();
                            let shape = [cx - r * nx, cy - r * ny, a, a * nz, (-nx).atan2(ny)];
                            source.conflicting_ellipses +=
                                usize::from(assign(&mut frames[i], shape, 3, 0.25));
                        }
                    }
                }
            }
            for f in &mut frames {
                let ep = &manifest.epochs[f.epoch as usize];
                if let Some((shape, admissible)) = native.get(&(
                    ep.clone(),
                    f.eye,
                    f.ns,
                    f.sequence,
                    f.origin[0] as u64,
                    f.origin[1] as u64,
                    f.width as u64,
                    f.height as u64,
                )) {
                    assign(f, *shape, 1, if *admissible { 1. } else { 0.25 });
                    if *admissible {
                        f.flags |= 2;
                    }
                }
                source.ellipse_counts[f.provider as usize] += 1;
            }
            source.frames = frames.len();
            all.extend(frames);
            manifest.sources.push(source);
        }
        if pi % 25 == 0 {
            eprintln!(
                "packed {}/{} paths: {} indexed ROI frames",
                pi + 1,
                paths.len(),
                all.len()
            );
        }
    }
    archive::compact(&mut manifest, &mut all);
    archive::write(output, &manifest, &all)?;
    let (reloaded, roundtrip) = archive::read(output)?;
    if roundtrip.len() != all.len()
        || all
            .iter()
            .zip(&roundtrip)
            .any(|(a, b)| a.encode() != b.encode())
    {
        return Err("binary roundtrip mismatch".into());
    }
    let report = json!({"binary":output,"bytes":fs::metadata(output)?.len(),"sha256":archive::digest(&fs::read(output)?),"records":all.len(),"sources":manifest.sources.len(),"epochs":manifest.epochs.len(),"ellipse_counts":(0..4).map(|p|all.iter().filter(|f|f.provider==p).count()).collect::<Vec<_>>(),"roundtrip_exact":true,"manifest":reloaded});
    fs::write(
        output.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    eprintln!(
        "PACKED {} frames / {} sources into {} bytes at {}",
        all.len(),
        manifest.sources.len(),
        fs::metadata(output)?.len(),
        output.display()
    );
    Ok(())
}
