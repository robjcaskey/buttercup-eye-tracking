//! Offline matched recent-exclusion experiment on exported native contours.
//! No live viewer/model settings are changed. See docs/flat-tire-area-and-motion.md.
#![allow(dead_code)]

#[path = "../geometry.rs"]
mod geometry;
// The directory wrapper preserves Rust's normal child-module lookup for the
// shared files; a direct #[path = "../file.rs"] changes that lookup root.
#[path = "../"]
mod native {
    pub(crate) mod conic_solver;
    pub(crate) mod outline_conic_segments;
    pub(crate) mod roi_evidence;
}
use native::{conic_solver, outline_conic_segments, roi_evidence};
#[path = "../raw10.rs"]
mod raw10;

use geometry::{ellipse_axis_point, Ellipse};
use outline_conic_segments::recent_exclusion::*;
use outline_conic_segments::{
    best_flat_tire_run, best_impossible_conic_run, sample_closed_contour, smooth_closed_contour,
    FlatTireSide,
};
use roi_evidence::{ExposureKey, RoiId, SourceClock};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::{PI, TAU};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, String>;

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_slice(&fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| format!("{}: {e}", path.display()))
}
fn number(v: &Value, key: &str) -> Result<f64> {
    v[key].as_f64().ok_or_else(|| format!("missing {key}"))
}
fn integer(v: &Value, key: &str) -> Result<u64> {
    v[key].as_u64().ok_or_else(|| format!("missing {key}"))
}
fn point(v: &Value) -> Option<(f64, f64)> {
    Some((v[0].as_f64()?, v[1].as_f64()?))
}
fn points(v: &Value) -> Vec<(f64, f64)> {
    v.as_array()
        .map(|a| a.iter().filter_map(point).collect())
        .unwrap_or_default()
}
fn ellipse(v: &Value) -> Option<Ellipse> {
    Some(Ellipse {
        center: point(&v["center"])?,
        major_radius: v["major_radius"].as_f64()?,
        minor_radius: v["minor_radius"].as_f64()?,
        angle: v["angle"].as_f64()?,
    })
}
fn ellipse_json(e: Ellipse) -> Value {
    json!({"center": e.center, "major_radius":e.major_radius,
    "minor_radius":e.minor_radius, "angle":e.angle})
}
fn hash(text: &str) -> u64 {
    text.bytes().fold(14695981039346656037, |h, b| {
        (h ^ b as u64).wrapping_mul(1099511628211)
    })
}

fn lineage_root(aliases: &BTreeMap<String, String>, lineage: &str) -> String {
    let mut root = lineage.to_string();
    while let Some(next) = aliases.get(&root) {
        root = next.clone();
    }
    root
}

fn new_output(path: &Path) -> Result<File> {
    let parent = path.parent().ok_or("output needs a parent directory")?;
    let allowed = fs::canonicalize("outputs").map_err(|e| e.to_string())?;
    if parent
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("runtime output cannot contain parent-directory traversal".into());
    }
    let mut ancestor = parent;
    while !ancestor.exists() {
        ancestor = ancestor.parent().ok_or("output has no existing parent")?;
    }
    if !fs::canonicalize(ancestor)
        .map_err(|e| e.to_string())?
        .starts_with(&allowed)
    {
        return Err("runtime output must be beneath the checked outputs link".into());
    }
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Build only metadata and symlinks to canonical archived RAW; never copy RAW,
/// read recorded predictions, or count backup/assistant labels independently.
fn prepare_capture(output: &Path, label_paths: &[String]) -> Result<()> {
    if output.exists() {
        return Err("capture-view output already exists".into());
    }
    let manifest = output.join("frames.jsonl");
    let mut writer = new_output(&manifest)?;
    let mut frames = BTreeMap::<(String, u64), Value>::new();
    for path in label_paths {
        let label = Path::new(path);
        let filename = label
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or("invalid label name")?;
        if label
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|v| v.to_str())
            != Some("labels")
            || filename.contains("backup")
            || !filename.ends_with(".labels.json")
        {
            return Err(format!(
                "only direct canonical annotator/labels files are accepted: {path}"
            ));
        }
        let annotator = label
            .parent()
            .and_then(|p| p.parent())
            .ok_or("invalid annotator path")?;
        if annotator.file_name().and_then(|v| v.to_str()) != Some("annotator") {
            return Err("noncanonical label path".into());
        }
        let archive = annotator.join("archive");
        let stem = filename.trim_end_matches(".labels.json");
        let metadata_path = archive.join(format!("{stem}.json"));
        let metadata = read_json(&metadata_path)?;
        let target = integer(&metadata, "sequence")?;
        let lineage = fs::canonicalize(annotator.parent().ok_or("capture parent")?)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .to_string();
        let contexts = metadata["context_frames"]
            .as_array()
            .ok_or("missing native temporal context")?;
        for frame in contexts {
            let member = frame["path"].as_str().ok_or("missing RAW member")?;
            if Path::new(member).components().count() != 1 {
                return Err("invalid RAW member".into());
            }
            let seq = member
                .trim_end_matches(".raw10")
                .rsplit('-')
                .next()
                .ok_or("RAW sequence")?
                .parse::<u64>()
                .map_err(|e| e.to_string())?;
            let raw = fs::canonicalize(archive.join(member)).map_err(|e| e.to_string())?;
            let raw_text = raw.to_string_lossy().to_string();
            let timestamp = integer(frame, "timestamp_ns")?;
            let key = (raw_text.clone(), timestamp);
            let stream = format!("{:016x}-seq-{seq}.raw10", hash(&lineage));
            let length = integer(frame, "stride")? * integer(frame, "height")?;
            if fs::metadata(&raw).map_err(|e| e.to_string())?.len() != length {
                return Err(format!("RAW size mismatch: {raw_text}"));
            }
            let item = frames.entry(key).or_insert_with(|| {
                json!({
                    "sequence":seq,"timestamp_ns":timestamp,"eye_id":1,"label":"subject-right",
                    "sensor_x":frame["sensor_x"],"sensor_y":frame["sensor_y"],
                    "width":frame["width"],"height":frame["height"],"stride":frame["stride"],
                    "pixel_format":"RAW10_LE40_1X1","stream":stream,"offset":0,"length":length,
                    "source_raw":raw_text,"lineage":lineage,
                })
            });
            if seq == target {
                item["canonical_label"] =
                    json!(fs::canonicalize(label).map_err(|e| e.to_string())?);
            }
        }
    }
    let mut unique = BTreeMap::<String, Value>::new();
    let mut aliases = BTreeMap::<String, String>::new();
    for frame in frames.into_values() {
        let key = format!(
            "{}:{}:{}:{}:{}:{}:{}",
            frame["timestamp_ns"],
            frame["sequence"],
            frame["sensor_x"],
            frame["sensor_y"],
            frame["width"],
            frame["height"],
            frame["stride"]
        );
        if let Some(prior) = unique.get_mut(&key) {
            // Byte equality confirms that overlap across annotation archives
            // is one exposure, even though the archived paths differ.
            if fs::read(prior["source_raw"].as_str().unwrap()).map_err(|e| e.to_string())?
                != fs::read(frame["source_raw"].as_str().unwrap()).map_err(|e| e.to_string())?
            {
                return Err(
                    "ambiguous identical exposure metadata with different RAW bytes".into(),
                );
            }
            let a = lineage_root(&aliases, prior["lineage"].as_str().unwrap());
            let b = lineage_root(&aliases, frame["lineage"].as_str().unwrap());
            if a != b {
                aliases.insert(a.clone().max(b.clone()), a.min(b));
            }
            if frame["canonical_label"].is_string() {
                if prior["canonical_label"].is_string()
                    && prior["canonical_label"] != frame["canonical_label"]
                {
                    return Err("duplicate canonical annotations for one physical exposure".into());
                }
                prior["canonical_label"] = frame["canonical_label"].clone();
            }
        } else {
            unique.insert(key, frame);
        }
    }
    let mut records = unique.into_values().collect::<Vec<_>>();
    for record in &mut records {
        record["lineage"] = json!(lineage_root(&aliases, record["lineage"].as_str().unwrap()));
    }
    records.sort_by_key(|r| {
        (
            r["lineage"].as_str().unwrap_or("").to_string(),
            r["timestamp_ns"].as_u64().unwrap_or(0),
        )
    });
    for frame in &records {
        let source = frame["source_raw"].as_str().ok_or("source raw")?;
        let destination = output.join(frame["stream"].as_str().ok_or("stream")?);
        if !destination.exists() {
            std::os::unix::fs::symlink(source, destination).map_err(|e| e.to_string())?;
        }
        serde_json::to_writer(&mut writer, frame).map_err(|e| e.to_string())?;
        writer.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    println!(
        "capture_view={} frames={} canonical_targets={}",
        output.display(),
        records.len(),
        records
            .iter()
            .filter(|r| r["canonical_label"].is_string())
            .count()
    );
    Ok(())
}

#[derive(Clone)]
struct Variant {
    name: String,
    config: Option<RecentExclusionConfig>,
    polish: bool,
}
fn variants() -> Vec<Variant> {
    let mut variants = vec![
        Variant {
            name: "exported_baseline".into(),
            config: None,
            polish: false,
        },
        Variant {
            name: "matched_polish_no_memory".into(),
            config: None,
            polish: true,
        },
    ];
    for (decay, name) in [
        (DecayFunction::Exponential, "exponential"),
        (DecayFunction::Linear, "linear"),
        (DecayFunction::FiniteHorizon, "finite_horizon"),
    ] {
        for duration_ms in [100, 250, 500] {
            for (penalty, hard) in [(0.50, false), (0.75, false), (0.75, true)] {
                variants.push(Variant {
                    name: format!(
                        "{name}_{duration_ms}ms_p{}_{}",
                        (penalty * 100.0) as u32,
                        if hard { "hysteresis" } else { "soft" }
                    ),
                    polish: true,
                    config: Some(RecentExclusionConfig {
                        decay,
                        decay_ns: duration_ms * 1_000_000,
                        horizon_ns: duration_ms * 3_000_000,
                        maximum_penalty: penalty,
                        reexclude_thresholds: hard.then_some((0.60, 0.20)),
                    }),
                });
            }
        }
    }
    variants
}

fn signed_residual(p: (f64, f64), e: Ellipse) -> f64 {
    let (x, y) = ellipse_axis_point(p, e);
    ((x / e.major_radius).hypot(y / e.minor_radius) - 1.0)
        * (e.major_radius * e.minor_radius).sqrt()
}
fn parameters(e: Ellipse) -> [f64; 5] {
    [
        e.center.0,
        e.center.1,
        e.major_radius.ln(),
        e.minor_radius.ln(),
        e.angle,
    ]
}
fn from_parameters(p: [f64; 5]) -> Ellipse {
    Ellipse {
        center: (p[0], p[1]),
        major_radius: p[2].exp(),
        minor_radius: p[3].exp(),
        angle: p[4],
    }
}
fn solve5(mut m: [[f64; 6]; 5]) -> Option<[f64; 5]> {
    for i in 0..5 {
        let pivot = (i..5).max_by(|&a, &b| m[a][i].abs().total_cmp(&m[b][i].abs()))?;
        if !m[pivot][i].is_finite() || m[pivot][i].abs() < 1e-10 {
            return None;
        }
        m.swap(i, pivot);
        let denominator = m[i][i];
        for j in i..6 {
            m[i][j] /= denominator;
        }
        for r in 0..5 {
            if r != i {
                let f = m[r][i];
                for c in i..6 {
                    m[r][c] -= f * m[i][c];
                }
            }
        }
    }
    Some(std::array::from_fn(|i| m[i][5]))
}

/// Same bounded robust polish in every matched arm; no radius/area target.
/// The native baseline's support determines eligibility. This experiment cannot
/// recover candidates for which that stateless fitter supplied no ellipse.
fn polish(
    initial: Ellipse,
    points: &[(f64, f64)],
    weights: &[f64],
    pixel_scale: f64,
) -> Option<Ellipse> {
    if points.len() < 12 || points.len() != weights.len() {
        return None;
    }
    if weights.iter().filter(|&&w| w > 0.0).count() < 12 {
        return None;
    }
    let cutoff = 2.0 * pixel_scale;
    let mut p = parameters(initial);
    let steps = [0.05, 0.05, 0.0001, 0.0001, 0.0001];
    for _ in 0..8 {
        let e = from_parameters(p);
        let mut m = [[0.0; 6]; 5];
        for (&point, &prior_weight) in points.iter().zip(weights) {
            if prior_weight == 0.0 {
                continue;
            }
            let residual = signed_residual(point, e);
            let weight = prior_weight * (cutoff / residual.abs().max(cutoff));
            let gradient = std::array::from_fn::<_, 5, _>(|i| {
                let mut plus = p;
                plus[i] += steps[i];
                let mut minus = p;
                minus[i] -= steps[i];
                (signed_residual(point, from_parameters(plus))
                    - signed_residual(point, from_parameters(minus)))
                    / (2.0 * steps[i])
            });
            for i in 0..5 {
                for j in 0..5 {
                    m[i][j] += weight * gradient[i] * gradient[j];
                }
                m[i][5] -= weight * gradient[i] * residual;
            }
        }
        for (i, row) in m.iter_mut().enumerate() {
            row[i] += 1e-6;
        }
        let Some(update) = solve5(m) else {
            break;
        };
        for i in 0..5 {
            let limit = if i < 2 { 2.0 * pixel_scale } else { 0.02 };
            p[i] += update[i].clamp(-limit, limit);
        }
    }
    let mut e = from_parameters(p);
    if e.minor_radius > e.major_radius {
        std::mem::swap(&mut e.major_radius, &mut e.minor_radius);
        e.angle += PI * 0.5;
    }
    let radius_ratio = e.major_radius / initial.major_radius;
    let minor_ratio = e.minor_radius / initial.minor_radius;
    let model = Ellipse {
        center: (
            (e.center.0 + 0.5) / pixel_scale - 0.5,
            (e.center.1 + 0.5) / pixel_scale - 0.5,
        ),
        major_radius: e.major_radius / pixel_scale,
        minor_radius: e.minor_radius / pixel_scale,
        angle: e.angle,
    };
    if !conic_solver::plausible_ellipse(model)
        || !e.angle.is_finite()
        || !(0.90..=1.10).contains(&radius_ratio)
        || !(0.90..=1.10).contains(&minor_ratio)
        || (e.center.0 - initial.center.0).hypot(e.center.1 - initial.center.1)
            > 0.10 * initial.major_radius
        || !conic_solver::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE
            .admits_axes(e.major_radius, e.minor_radius)
    {
        return None;
    }
    // Preserve current-frame support and angular spread after any re-exclusion.
    let mut bins = BTreeSet::new();
    let mut supported = 0;
    for (&point, &weight) in points.iter().zip(weights) {
        if weight > 0.0 && signed_residual(point, e).abs() <= 4.0 * pixel_scale {
            supported += 1;
            if let Some((phase, _)) = canonical_point(point, e) {
                bins.insert((phase / TAU * 16.0) as usize);
            }
        }
    }
    (supported >= 12 && bins.len() >= 5).then_some(e)
}

struct CurrentEvidence {
    good: Vec<bool>,
    bad_points: Vec<(f64, f64)>,
    good_points: Vec<(f64, f64)>,
}
fn current_evidence(
    outline: &[(f64, f64)],
    retained: &[(f64, f64)],
    e: Ellipse,
    pixel_scale: f64,
    detect_chords: bool,
) -> CurrentEvidence {
    let samples = sample_closed_contour(outline, 128);
    let smooth = smooth_closed_contour(&samples, 2);
    let n = samples.len();
    if n < 24 {
        return CurrentEvidence {
            good: vec![false; retained.len()],
            bad_points: vec![],
            good_points: vec![],
        };
    }
    let mut bad = vec![false; n];
    if detect_chords {
        let model_samples = smooth
            .iter()
            .map(|&(x, y)| ((x + 0.5) / pixel_scale - 0.5, (y + 0.5) / pixel_scale - 0.5))
            .collect::<Vec<_>>();
        let model = Ellipse {
            center: (
                (e.center.0 + 0.5) / pixel_scale - 0.5,
                (e.center.1 + 0.5) / pixel_scale - 0.5,
            ),
            major_radius: e.major_radius / pixel_scale,
            minor_radius: e.minor_radius / pixel_scale,
            angle: e.angle,
        };
        for side in [FlatTireSide::Upper, FlatTireSide::Lower] {
            if let Some(run) = best_flat_tire_run(&model_samples, model, side) {
                for d in 1..run.length.saturating_sub(1) {
                    bad[(run.start + d) % n] = true;
                }
            }
        }
        for _ in 0..4 {
            let Some(run) = best_impossible_conic_run(&model_samples, model, &bad) else {
                break;
            };
            for d in 1..run.length.saturating_sub(1) {
                bad[(run.start + d) % n] = true;
            }
        }
    }
    let good = retained
        .iter()
        .map(|&p| {
            let i = (0..n)
                .min_by(|&a, &b| {
                    ((samples[a].0 - p.0).hypot(samples[a].1 - p.1))
                        .total_cmp(&(samples[b].0 - p.0).hypot(samples[b].1 - p.1))
                })
                .unwrap();
            if bad[i] || signed_residual(p, e).abs() > 1.5 * pixel_scale {
                return false;
            }
            let a = smooth[(i + n - 3) % n];
            let b = smooth[i];
            let c = smooth[(i + 3) % n];
            let u = (b.0 - a.0, b.1 - a.1);
            let v = (c.0 - b.0, c.1 - b.1);
            let curvature = 2.0 * (u.0 * v.1 - u.1 * v.0).abs()
                / (u.0.hypot(u.1) * v.0.hypot(v.1) * (c.0 - a.0).hypot(c.1 - a.1)).max(1e-9);
            let (x, y) = ellipse_axis_point(p, e);
            let phase = (y / e.minor_radius).atan2(x / e.major_radius);
            let expected = e.major_radius * e.minor_radius
                / ((e.major_radius * phase.sin()).powi(2) + (e.minor_radius * phase.cos()).powi(2))
                    .powf(1.5);
            let tangent = (c.0 - a.0, c.1 - a.1);
            let (sin, cos) = e.angle.sin_cos();
            let local = (-e.major_radius * phase.sin(), e.minor_radius * phase.cos());
            let predicted = (cos * local.0 - sin * local.1, sin * local.0 + cos * local.1);
            let agreement = (tangent.0 * predicted.0 + tangent.1 * predicted.1).abs()
                / (tangent.0.hypot(tangent.1) * predicted.0.hypot(predicted.1)).max(1e-9);
            agreement >= 0.96 && (0.4..=2.5).contains(&(curvature / expected))
        })
        .collect::<Vec<_>>();
    CurrentEvidence {
        good_points: retained
            .iter()
            .zip(&good)
            .filter_map(|(&p, &g)| g.then_some(p))
            .collect(),
        bad_points: samples
            .iter()
            .zip(bad)
            .filter_map(|(&p, b)| b.then_some(p))
            .collect(),
        good,
    }
}

struct Raw {
    pixels: Vec<u16>,
    packed: Vec<u8>,
    width: usize,
    height: usize,
}
fn load_raw(capture: &Path, record: &Value) -> Result<Raw> {
    let stream = record["stream"].as_str().ok_or("missing RAW stream")?;
    if Path::new(stream).components().count() != 1 || stream == ".." {
        return Err("invalid RAW member".into());
    }
    let mut file = File::open(capture.join(stream)).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(integer(record, "offset")?))
        .map_err(|e| e.to_string())?;
    let mut packed = vec![0; integer(record, "length")? as usize];
    file.read_exact(&mut packed).map_err(|e| e.to_string())?;
    let width = integer(record, "width")? as usize;
    let height = integer(record, "height")? as usize;
    Ok(Raw {
        pixels: raw10::try_unpack_raw10(
            &packed,
            width,
            height,
            integer(record, "stride")? as usize,
        )?,
        packed,
        width,
        height,
    })
}
fn sample_raw(raw: &Raw, p: (f64, f64)) -> Option<f64> {
    let x = p.0.round() as isize;
    let y = p.1.round() as isize;
    if x < 1 || y < 1 || x + 1 >= raw.width as isize || y + 1 >= raw.height as isize {
        return None;
    }
    let mut value = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            value += raw.pixels[(y + dy) as usize * raw.width + (x + dx) as usize] as f64;
        }
    }
    Some(value / (9.0 * 1023.0))
}
fn raw_arc_support(raw: &Raw, e: Ellipse) -> Option<f64> {
    let (sin, cos) = e.angle.sin_cos();
    let mut contrast = Vec::new();
    // Lateral image-space arcs reduce direct upper/lower lid contamination;
    // this simple signed contrast is a diagnostic, not the live RAW gate.
    for i in 0..96 {
        let phase = TAU * i as f64 / 96.0;
        let local = (e.major_radius * phase.cos(), e.minor_radius * phase.sin());
        let p = (
            e.center.0 + cos * local.0 - sin * local.1,
            e.center.1 + sin * local.0 + cos * local.1,
        );
        if (p.0 - e.center.0).abs() < 0.65 * e.major_radius {
            continue;
        }
        let normal = (
            cos * phase.cos() / e.major_radius - sin * phase.sin() / e.minor_radius,
            sin * phase.cos() / e.major_radius + cos * phase.sin() / e.minor_radius,
        );
        let length = normal.0.hypot(normal.1);
        let d = (3.0 * normal.0 / length, 3.0 * normal.1 / length);
        if let Some((inside, outside)) =
            sample_raw(raw, (p.0 - d.0, p.1 - d.1)).zip(sample_raw(raw, (p.0 + d.0, p.1 + d.1)))
        {
            contrast.push(outside - inside);
        }
    }
    percentile(&contrast, 0.5)
}
fn percentile(values: &[f64], p: f64) -> Option<f64> {
    let mut sorted = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    (!sorted.is_empty()).then(|| sorted[((sorted.len() - 1) as f64 * p).round() as usize])
}
fn distribution(values: &[f64]) -> Value {
    json!({"count":values.len(),"mean":(!values.is_empty()).then(||values.iter().sum::<f64>()/values.len() as f64),
        "p50":percentile(values,0.50),"p95":percentile(values,0.95),"max":percentile(values,1.0)})
}

/// Euclidean point-to-ellipse distance: coarse bracket then golden-section
/// minimization. Human labels are read only after all candidates are fixed.
fn label_distance(p: (f64, f64), e: Ellipse) -> f64 {
    let (x, y) = ellipse_axis_point(p, e);
    let x = x.abs();
    let y = y.abs();
    let squared =
        |t: f64| (e.major_radius * t.cos() - x).powi(2) + (e.minor_radius * t.sin() - y).powi(2);
    let count = 64_usize;
    let best = (0..=count)
        .min_by(|&a, &b| {
            squared(PI * 0.5 * a as f64 / count as f64)
                .total_cmp(&squared(PI * 0.5 * b as f64 / count as f64))
        })
        .unwrap();
    let mut lo = PI * 0.5 * best.saturating_sub(1) as f64 / count as f64;
    let mut hi = PI * 0.5 * (best + 1).min(count) as f64 / count as f64;
    for _ in 0..32 {
        let a = lo + (hi - lo) * 0.38196601125;
        let b = lo + (hi - lo) * 0.61803398875;
        if squared(a) < squared(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    squared((lo + hi) * 0.5).sqrt()
}

#[derive(Clone)]
struct ScaleEvidence {
    scale: IndependentScale,
    report: String,
    stream: PathBuf,
    offset: u64,
    length: usize,
    reference_timestamp_ns: u64,
    cumulative_fractional_uncertainty: f64,
}
fn scale_key(
    eye: &str,
    timestamp: u64,
    sequence: u64,
    origin: (f64, f64),
    width: usize,
    height: usize,
) -> String {
    format!(
        "{eye}:{timestamp}:{sequence}:{}:{}:{width}:{height}",
        origin.0, origin.1
    )
}

fn similarity_linear_scale(diagonal_coefficient_delta: f64, rotation_coefficient: f64) -> Option<f64> {
    if !diagonal_coefficient_delta.is_finite() || !rotation_coefficient.is_finite() {
        return None;
    }
    // Native SimilarityMotion uses [[1+d,-r],[r,1+d]], so sqrt(det A)
    // is hypot(1+d,r); r is a matrix coefficient, not a standalone angle.
    let scale = (1.0 + diagonal_coefficient_delta).hypot(rotation_coefficient);
    (scale.is_finite() && scale > 0.0).then_some(scale)
}
fn load_scales(paths: &[String]) -> Result<BTreeMap<String, ScaleEvidence>> {
    let mut index = BTreeMap::new();
    for path in paths {
        let report = read_json(Path::new(path))?;
        let frames = report["frames"].as_array().ok_or("scale report frames")?;
        let eye = report["source"]["label"]
            .as_str()
            .ok_or("scale report eye")?;
        let stream = PathBuf::from(
            report["source"]["stream"]
                .as_str()
                .ok_or("scale report stream")?,
        );
        let mut chain = None::<(f64, f64, u64)>;
        for i in 1..frames.len() {
            let current = &frames[i];
            let previous = &frames[i - 1];
            let motion = &current["shared_global_scale"];
            let delta = motion["scale_delta"].as_f64().unwrap_or(f64::NAN);
            let rotation_coefficient = motion["rotation"].as_f64().unwrap_or(f64::NAN);
            let residual = motion["motion_residual"].as_f64().unwrap_or(f64::NAN);
            let support = motion["motion_support"].as_u64().unwrap_or(0);
            let timestamp = integer(current, "timestamp_ns")?;
            let previous_timestamp = integer(previous, "timestamp_ns")?;
            let valid = motion["reliable"] == true
                && support >= 9
                && residual.is_finite()
                && residual <= 2.0
                && motion["stable_frames"].as_u64().unwrap_or(0) >= 2
                && motion["occupied_quadrants"].as_u64().unwrap_or(0) >= 3
                && delta.is_finite()
                && delta.abs() <= 0.04
                && rotation_coefficient.is_finite()
                && rotation_coefficient.abs() <= 0.10
                && current["width"] == previous["width"]
                && current["height"] == previous["height"]
                && timestamp
                    .checked_sub(previous_timestamp)
                    .is_some_and(|dt| dt > 0 && dt <= 500_000_000);
            if !valid {
                chain = None;
                continue;
            }
            let (mut old_scale, mut old_uncertainty, mut reference) =
                chain.unwrap_or((1.0, 0.0, previous_timestamp));
            // This is the existing heuristic per-step scale allowance, summed
            // conservatively because adjacent RAW patches are correlated.
            let uncertainty = (0.012 + residual.max(0.0) / 180.0 + 0.025 / (support as f64).sqrt())
                .clamp(0.012, 0.045);
            if timestamp.saturating_sub(reference) > 1_000_000_000
                || old_uncertainty + uncertainty > 0.25
            {
                old_scale = 1.0;
                old_uncertainty = 0.0;
                reference = previous_timestamp;
            }
            let new_scale =
                old_scale * similarity_linear_scale(delta, rotation_coefficient).expect("validated similarity");
            let provenance = hash(&format!("{path}:{reference}"));
            for (frame, scale, uncertainty) in [
                (previous, old_scale, old_uncertainty),
                (current, new_scale, old_uncertainty + uncertainty),
            ] {
                let origin = point(&frame["sensor_origin"]).ok_or("scale report ROI origin")?;
                let key = scale_key(
                    eye,
                    integer(frame, "timestamp_ns")?,
                    integer(frame, "sequence")?,
                    origin,
                    integer(frame, "width")? as usize,
                    integer(frame, "height")? as usize,
                );
                let item = ScaleEvidence {
                    scale: IndependentScale {
                        pixels_per_reference_unit: scale,
                        provenance,
                    },
                    report: path.clone(),
                    stream: stream.clone(),
                    offset: integer(frame, "source_offset")?,
                    length: integer(frame, "source_length")? as usize,
                    reference_timestamp_ns: reference,
                    cumulative_fractional_uncertainty: uncertainty,
                };
                if index
                    .get(&key)
                    .is_some_and(|old: &ScaleEvidence| old.report != *path)
                {
                    return Err(
                        "overlapping scale reports require an explicit source choice".into(),
                    );
                }
                index.insert(key, item);
            }
            chain = Some((new_scale, old_uncertainty + uncertainty, reference));
        }
    }
    Ok(index)
}
fn checked_scale(
    index: &BTreeMap<String, ScaleEvidence>,
    key: &str,
    raw: &Raw,
) -> Result<Option<ScaleEvidence>> {
    let Some(scale) = index.get(key) else {
        return Ok(None);
    };
    let mut source = File::open(&scale.stream).map_err(|e| e.to_string())?;
    source
        .seek(SeekFrom::Start(scale.offset))
        .map_err(|e| e.to_string())?;
    let mut packed = vec![0; scale.length];
    source.read_exact(&mut packed).map_err(|e| e.to_string())?;
    if packed != raw.packed {
        return Err(format!(
            "scale source RAW differs from contour exposure: {key}"
        ));
    }
    Ok(Some(scale.clone()))
}

fn evaluate(output: &Path, inputs: &[String], scale_reports: &[String]) -> Result<()> {
    let scale_index = load_scales(scale_reports)?;
    let variants = variants();
    let mut rows = Vec::new();
    let mut candidate_rows = Vec::new();
    let mut source_counts = Vec::new();
    let mut canonical_labels = BTreeSet::new();
    for input in inputs {
        let report = read_json(Path::new(input))?;
        if report["schema"] != "buttercup-native-sam-outlines-v1" {
            return Err(format!("unsupported outline schema: {input}"));
        }
        let capture = PathBuf::from(report["capture"].as_str().ok_or("outline capture path")?);
        let eye = report["label"].as_str().ok_or("outline eye label")?;
        let records = fs::read_to_string(capture.join("frames.jsonl"))
            .map_err(|e| e.to_string())?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let cases = report["cases"].as_array().ok_or("outline cases")?;
        let mut memories = variants
            .iter()
            .map(|v| v.config.and_then(RecentExclusionMemory::new))
            .collect::<Vec<_>>();
        let mut source_candidates = 0;
        let mut source_fitted = 0;
        for case in cases {
            let seq = integer(case, "sequence")?;
            let timestamp = integer(case, "timestamp_ns")?;
            let record = records
                .iter()
                .find(|r| {
                    r["label"] == eye
                        && r["sequence"].as_u64() == Some(seq)
                        && r["timestamp_ns"].as_u64() == Some(timestamp)
                })
                .ok_or_else(|| format!("missing exact source record {input} sequence {seq}"))?;
            let lineage = record["lineage"]
                .as_str()
                .unwrap_or(capture.to_str().ok_or("capture path")?);
            let label_path = record["canonical_label"].as_str();
            if let Some(label) = label_path {
                canonical_labels.insert(label.to_string());
            }
            let raw = load_raw(&capture, record)?;
            let candidates = case["candidates"].as_array().ok_or("candidates")?;
            source_candidates += candidates.len();
            source_fitted += candidates
                .iter()
                .filter(|c| ellipse(&c["baseline_ellipse"]).is_some())
                .count();
            // Preserve the export's semantic order and frame-local RAW gate.
            // Human labels and area never select the query in any arm.
            let selected = candidates.iter().position(|c| {
                c["baseline_raw_admitted"] == true && ellipse(&c["baseline_ellipse"]).is_some()
            });
            let reference = selected
                .and_then(|i| ellipse(&candidates[i]["baseline_ellipse"]))
                .or_else(|| {
                    candidates
                        .iter()
                        .find_map(|c| ellipse(&c["baseline_ellipse"]))
                })
                .unwrap_or_default();
            let origin = (number(record, "sensor_x")?, number(record, "sensor_y")?);
            let scale_evidence = checked_scale(
                &scale_index,
                &scale_key(eye, timestamp, seq, origin, raw.width, raw.height),
                &raw,
            )?;
            let scale = scale_evidence.as_ref().map(|e| e.scale);
            let frame = ExclusionFrame {
                exposure: ExposureKey {
                    roi: RoiId(if eye == "subject-left" { 0 } else { 1 }),
                    clock: SourceClock {
                        domain: hash(lineage),
                        epoch: 0,
                    },
                    sequence: seq,
                    timestamp_ns: timestamp,
                },
                geometry_lineage: hash(&format!("{}x{}", raw.width, raw.height)),
                sensor_origin_px: origin,
                reference,
                independent_scale: scale,
            };
            let dispositions = memories
                .iter_mut()
                .map(|m| m.as_mut().map(|m| format!("{:?}", m.begin_frame(frame))))
                .collect::<Vec<_>>();
            let pixel_scale = raw.width as f64 / conic_solver::LEGACY_FIT_WIDTH as f64;
            let mut selected_evidence = None;
            let mut selected_results = vec![Value::Null; variants.len()];
            for (index, candidate) in candidates.iter().enumerate() {
                let Some(base) = ellipse(&candidate["baseline_ellipse"]) else {
                    continue;
                };
                let retained = points(&candidate["baseline_retained"]);
                let evidence = current_evidence(
                    &points(&candidate["outline"]),
                    &retained,
                    base,
                    pixel_scale,
                    selected == Some(index),
                );
                for (vi, variant) in variants.iter().enumerate() {
                    let weights = retained
                        .iter()
                        .zip(&evidence.good)
                        .map(|(&p, &g)| memories[vi].as_ref().map_or(1.0, |m| m.weight(p, g)))
                        .collect::<Vec<_>>();
                    let fitted = if variant.polish {
                        polish(base, &retained, &weights, pixel_scale)
                    } else {
                        Some(base)
                    };
                    let downweighted = weights.iter().filter(|&&w| w < 1.0).count();
                    let hard_excluded = weights.iter().filter(|&&w| w == 0.0).count();
                    candidate_rows.push(json!({"source":input,"sequence":seq,"query":candidate["query"],
                        "variant":variant.name,"fitted":fitted.is_some(),"downweighted":downweighted,"hard_excluded":hard_excluded}));
                    if selected == Some(index) {
                        selected_results[vi] = json!({"ellipse":fitted.map(ellipse_json),
                            "frontal_equivalent_area_px2":fitted.map(|e|PI*e.major_radius.powi(2)),
                            "sn_feida":fitted.zip(scale).and_then(|(e,s)|sn_feida(e,s)),
                            "current_raw_lateral_contrast":fitted.and_then(|e|raw_arc_support(&raw,e)),
                            "current_retained_residual_px":fitted.map(|e|distribution(&retained.iter().map(|&p|signed_residual(p,e).abs()).collect::<Vec<_>>())),
                            "retained_samples":retained.len(),"fresh_good_samples":evidence.good_points.len(),
                            "downweighted":downweighted,"hard_excluded":hard_excluded,
                            "memory_disposition":dispositions[vi],
                        });
                    }
                }
                if selected == Some(index) {
                    selected_evidence = Some(evidence);
                }
            }
            for (vi, memory) in memories.iter_mut().enumerate() {
                if let Some(memory) = memory {
                    if let Some(e) = &selected_evidence {
                        memory.observe_current(&e.bad_points, &e.good_points);
                    } else {
                        memory.observe_current(&[], &[]);
                    }
                    if let Some(object) = selected_results[vi].as_object_mut() {
                        object.insert(
                            "active_memory_regions_after_observation".into(),
                            json!(memory.active_regions()),
                        );
                    }
                }
            }
            rows.push(json!({"source":input,"lineage":lineage,"eye":eye,"sequence":seq,
                "timestamp_ns":timestamp,"width":raw.width,"height":raw.height,"canonical_label":label_path,
                "selected_query":selected.map(|i|candidates[i]["query"].clone()),
                "independent_scale_available":scale.is_some(),
                "independent_scale":scale_evidence.as_ref().map(|e|json!({
                    "scale":e.scale.pixels_per_reference_unit,"provenance":e.scale.provenance,
                    "reference_timestamp_ns":e.reference_timestamp_ns,"report":e.report,
                    "raw_bytes_verified":true,"cumulative_fractional_uncertainty_heuristic":e.cumulative_fractional_uncertainty,
                })),"results":selected_results}));
        }
        source_counts.push(json!({"input":input,"frames":cases.len(),"candidates":source_candidates,
            "baseline_fitted_candidates":source_fitted,"model":report["model"],"configuration":report["configuration"]}));
    }
    // Scoring only: candidate generation, ranking and temporal updates above
    // are complete before opening any annotation file.
    for row in &mut rows {
        let Some(path) = row["canonical_label"].as_str() else {
            continue;
        };
        let labels = read_json(Path::new(path))?;
        let visible = labels["annotation_points"]
            .as_array()
            .ok_or("canonical annotation_points")?
            .iter()
            .filter(|p| p["visibility"] == "visible" && p["kind"] == "iris_edge")
            .filter_map(|p| Some((p["x"].as_f64()?, p["y"].as_f64()?)))
            .collect::<Vec<_>>();
        row["visible_label_points"] = json!(visible.len());
        for result in row["results"].as_array_mut().unwrap() {
            if let Some(e) = ellipse(&result["ellipse"]) {
                result["human_visible_point_distance_px"] = distribution(
                    &visible
                        .iter()
                        .map(|&p| label_distance(p, e))
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    let summaries = (0..variants.len())
        .map(|vi| {
            let mut summary = summarize(&variants[vi], vi, &rows, &candidate_rows);
            summary["by_source"] = json!(inputs
                .iter()
                .map(|source| {
                    let selected = rows
                        .iter()
                        .filter(|r| r["source"] == *source)
                        .cloned()
                        .collect::<Vec<_>>();
                    let candidates = candidate_rows
                        .iter()
                        .filter(|r| r["source"] == *source)
                        .cloned()
                        .collect::<Vec<_>>();
                    let mut result = summarize(&variants[vi], vi, &selected, &candidates);
                    result["source"] = json!(source);
                    result
                })
                .collect::<Vec<_>>());
            summary
        })
        .collect::<Vec<_>>();
    let report = json!({"schema":"buttercup-recent-flat-tire-evaluation-v1",
        "contract":"Offline fixed-query comparison on native RAW contour exports. Exported baseline is preserved; the matched control and all decay candidates use the identical robust polish. Labels are scored only after every candidate and temporal update is fixed. No live enablement or label oracle. Neighbor frames and duplicate alternatives are not independent human ground truth.",
        "scale_contract":"Relative sn_feida uses exact-RAW-matched native global similarity, independent of candidate radius. The linear reference scale is 1 at the start of each contiguous reliable motion chain, with a distinct provenance ID; no comparisons cross reference changes. This is uncertain image-scale compensation, not metric anatomy or a calibrated depth estimate. Remaining pixel area steps are explicitly unnormalized proxies. Absolute log-step differences do not cancel unknown scale; signed candidate/control log area ratios do.",
        "raw_contract":"Original packed RAW10 streams decoded by src/raw10.rs; 3x3 native RAW mean lateral contrast is only a diagnostic and is not the live RAW admission gate.",
        "parameter_contract":{"angular_bins":REGION_BINS,"maximum_observation_points":128,
            "polish_iterations":8,"horizon_multiple":3,"reexclude_enter":0.60,"reexclude_leave":0.20,
            "maximum_adjacent_metric_gap_ns":500_000_000},
        "sources":source_counts,"scale_reports":scale_reports,"unique_canonical_label_targets":canonical_labels.len(),
        "variants":summaries,"frames":rows,"candidate_audit":candidate_rows});
    let mut writer = new_output(output)?;
    serde_json::to_writer_pretty(&mut writer, &report).map_err(|e| e.to_string())?;
    println!(
        "report={} frames={} variants={} canonical_targets={}",
        output.display(),
        rows.len(),
        variants.len(),
        canonical_labels.len()
    );
    for summary in &summaries {
        println!(
            "{} coverage={}/{} area_step_p95={:?} label_mean={:?} affected={}",
            summary["name"].as_str().unwrap(),
            summary["selected_fitted_frames"],
            rows.len(),
            summary["unnormalized_frontal_area_abs_log_step"]["p95"].as_f64(),
            summary["human_visible_frame_mean_distance_px"]["mean"].as_f64(),
            summary["downweighted_selected_frames"]
        );
    }
    Ok(())
}

fn summarize(variant: &Variant, vi: usize, rows: &[Value], candidates: &[Value]) -> Value {
    let mut area_steps = Vec::new();
    let mut matched_steps = Vec::new();
    let mut scale_steps = Vec::new();
    let mut control_steps_on_common = Vec::new();
    let mut control_scale_steps_on_common = Vec::new();
    let mut signed_log_ratios = Vec::new();
    let mut signed_log_ratio_steps = Vec::new();
    let mut human_errors = Vec::new();
    let mut raw_support = Vec::new();
    let mut residuals = Vec::new();
    let mut fitted = 0;
    let mut labeled_fitted = 0;
    let mut affected = 0;
    let mut area_spread = BTreeMap::<String, Vec<f64>>::new();
    for (i, row) in rows.iter().enumerate() {
        let result = &row["results"][vi];
        if ellipse(&result["ellipse"]).is_some() {
            fitted += 1;
        }
        if let Some(value) = result["human_visible_point_distance_px"]["mean"].as_f64() {
            human_errors.push(value);
            labeled_fitted += 1;
        }
        if let Some(value) = result["current_raw_lateral_contrast"].as_f64() {
            raw_support.push(value);
        }
        if let Some(value) = result["current_retained_residual_px"]["mean"].as_f64() {
            residuals.push(value);
        }
        if result["downweighted"].as_u64().unwrap_or(0) > 0 {
            affected += 1;
        }
        if let Some(area) = result["frontal_equivalent_area_px2"].as_f64() {
            if let Some(control) = row["results"][1]["frontal_equivalent_area_px2"].as_f64() {
                signed_log_ratios.push((area / control).ln());
            }
            area_spread
                .entry(format!("{}:{}", row["lineage"], row["eye"]))
                .or_default()
                .push(area);
            if i == 0 {
                continue;
            }
            let previous = &rows[i - 1];
            let prior = &previous["results"][vi];
            let consecutive = row["lineage"] == previous["lineage"]
                && row["eye"] == previous["eye"]
                && row["source"] == previous["source"]
                && row["width"] == previous["width"]
                && row["height"] == previous["height"]
                && row["timestamp_ns"]
                    .as_u64()
                    .unwrap()
                    .checked_sub(previous["timestamp_ns"].as_u64().unwrap())
                    .is_some_and(|dt| dt > 0 && dt <= 500_000_000);
            if consecutive {
                if let Some(old) = prior["frontal_equivalent_area_px2"].as_f64() {
                    let step = (area / old).ln().abs();
                    area_steps.push(step);
                    if let Some((a, b)) = row["results"][1]["frontal_equivalent_area_px2"]
                        .as_f64()
                        .zip(previous["results"][1]["frontal_equivalent_area_px2"].as_f64())
                    {
                        let control_step = (a / b).ln().abs();
                        matched_steps.push(step - control_step);
                        control_steps_on_common.push(control_step);
                        signed_log_ratio_steps.push((area / a).ln() - (old / b).ln());
                    }
                }
                if row["independent_scale"]["provenance"]
                    == previous["independent_scale"]["provenance"]
                {
                    if let Some((a, b)) =
                        result["sn_feida"].as_f64().zip(prior["sn_feida"].as_f64())
                    {
                        scale_steps.push((a / b).ln().abs());
                        if let Some((a, b)) = row["results"][1]["sn_feida"]
                            .as_f64()
                            .zip(previous["results"][1]["sn_feida"].as_f64())
                        {
                            control_scale_steps_on_common.push((a / b).ln().abs());
                        }
                    }
                }
            }
        }
    }
    let candidate_fit = candidates
        .iter()
        .filter(|c| c["variant"] == variant.name && c["fitted"] == true)
        .count();
    let spans = area_spread
        .into_iter()
        .map(|(k, areas)| {
            let mean = areas.iter().sum::<f64>() / areas.len() as f64;
            let cv = (areas.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / areas.len() as f64)
                .sqrt()
                / mean;
            json!({"group":k,"frames":areas.len(),"unnormalized_area_cv":cv})
        })
        .collect::<Vec<_>>();
    json!({"name":variant.name,"selected_fitted_frames":fitted,"selected_dropouts":rows.len()-fitted,
        "fitted_candidates":candidate_fit,"labeled_fitted_frames":labeled_fitted,
        "downweighted_selected_frames":affected,"unnormalized_frontal_area_abs_log_step":distribution(&area_steps),
        "matched_control_area_step_delta":distribution(&matched_steps),
        "matched_control_unnormalized_area_step_on_same_pairs":distribution(&control_steps_on_common),
        "signed_log_area_ratio_to_matched_control":distribution(&signed_log_ratios),
        "signed_change_of_log_area_ratio_to_matched_control":distribution(&signed_log_ratio_steps),
        "sn_feida_abs_log_step":distribution(&scale_steps),
        "matched_control_sn_feida_step_on_same_pairs":distribution(&control_scale_steps_on_common),
        "independent_scale_frames":rows.iter().filter(|r|r["independent_scale_available"]==true).count(),
        "sn_feida_frames":rows.iter().filter(|r|r["results"][vi]["sn_feida"].is_number()).count(),
        "human_visible_frame_mean_distance_px":distribution(&human_errors),
        "current_raw_lateral_contrast":distribution(&raw_support),"current_retained_mean_residual_px":distribution(&residuals),
        "unnormalized_group_area_spread":spans})
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let result = if args.first().map(String::as_str) == Some("--prepare-label-capture")
        && args.len() >= 3
    {
        prepare_capture(Path::new(&args[1]), &args[2..])
    } else if args.len() >= 2 {
        let mut inputs = Vec::new();
        let mut scales = Vec::new();
        let mut i = 1;
        while i < args.len() {
            if args[i] == "--scale-report" && i + 1 < args.len() {
                scales.push(args[i + 1].clone());
                i += 2;
            } else {
                inputs.push(args[i].clone());
                i += 1;
            }
        }
        evaluate(Path::new(&args[0]), &inputs, &scales)
    } else {
        Err("usage: buttercup_flat_tire_eval OUTPUT.json OUTLINES.json [OUTLINES.json ...] [--scale-report RAW_REPLAY.json ...]\n       buttercup_flat_tire_eval --prepare-label-capture OUTPUT_DIR CANONICAL.labels.json [...]".into())
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;
    #[test]
    fn similarity_scale_uses_determinant_with_nonzero_rotation() {
        let scale = similarity_linear_scale(0.02, 0.08).unwrap();
        assert!((scale.powi(2) - (1.02_f64.powi(2) + 0.08_f64.powi(2))).abs() < 1e-12);
        assert!(scale > 1.02);
        let angle = 0.08_f64;
        assert!(
            (similarity_linear_scale(angle.cos() - 1.0, angle.sin()).unwrap() - 1.0).abs() < 1e-12
        );
        assert!(similarity_linear_scale(f64::NAN, 0.0).is_none());
    }
    #[test]
    fn weighted_polish_cannot_publish_without_current_support() {
        let e = Ellipse {
            center: (100.0, 100.0),
            major_radius: 70.0,
            minor_radius: 50.0,
            angle: 0.2,
        };
        let points = e.dense_points(80);
        assert!(polish(e, &points, &vec![0.0; 80], 1.0).is_none());
        let good = polish(e, &points, &vec![1.0; 80], 1.0).unwrap();
        assert!((good.major_radius - e.major_radius).abs() < 1e-8);
    }
    #[test]
    fn point_distance_handles_axis_endpoints_and_center() {
        let e = Ellipse {
            center: (0.0, 0.0),
            major_radius: 20.0,
            minor_radius: 10.0,
            angle: 0.0,
        };
        assert!((label_distance((23.0, 0.0), e) - 3.0).abs() < 1e-8);
        assert!((label_distance((0.0, 0.0), e) - 10.0).abs() < 1e-8);
        assert!(label_distance((20.0, 0.0), e) < 1e-7);
    }
}
