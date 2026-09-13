//! Read-only replay of archived estimates. No model loading, RAW rewriting or live authority.
#![allow(dead_code)]
#[path = "../eye_scene_model/sign_kinematic_beam.rs"]
mod sign_kinematic_beam;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sign_kinematic_beam::{
    HeadTransport, Hypothesis, Observation, SignKinematicBeam, SourceIdentity, Status,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_FEATURES: usize = 64;
const MAX_SOURCE_CACHE: usize = 96;
const ARM_COUNT: usize = 6;
const STRIDES: [usize; ARM_COUNT] = [1, 1, 1, 2, 4, 8];
const ARMS: [&str; ARM_COUNT] = [
    "surrounding_tissue_transport",
    "no_transport",
    "iris_contaminated_negative_control",
    "surrounding_tissue_stride2",
    "surrounding_tissue_stride4",
    "surrounding_tissue_stride8",
];

fn number(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str()?.parse().ok())
}
fn array<const N: usize>(v: &Value) -> Option<[f64; N]> {
    let mut out = [0.0; N];
    for (i, x) in out.iter_mut().enumerate() {
        *x = v[i].as_f64()?;
        if !x.is_finite() {
            return None;
        }
    }
    Some(out)
}
fn rows(path: &Path) -> Result<impl Iterator<Item = Result<Value, String>>, String> {
    let label = path.display().to_string();
    Ok(
        BufReader::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?)
            .lines()
            .enumerate()
            .filter_map(move |(i, line)| match line {
                Ok(line) if line.trim().is_empty() => None,
                line => Some(line.map_err(|e| e.to_string()).and_then(|line| {
                    serde_json::from_str(&line).map_err(|e| format!("{label} line {}: {e}", i + 1))
                })),
            }),
    )
}
fn discover(path: &Path, dirs: &mut BTreeSet<PathBuf>, depth: usize) -> Result<(), String> {
    if depth > 8 {
        return Err(format!("discovery depth exceeded: {}", path.display()));
    }
    if path.join("frames.jsonl").is_file() && path.join("predictions.jsonl").is_file() {
        dirs.insert(fs::canonicalize(path).map_err(|e| e.to_string())?);
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            discover(&entry.path(), dirs, depth + 1)?;
        }
    }
    Ok(())
}
fn digest(path: &Path) -> Result<String, String> {
    let mut hash = Sha256::new();
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    std::io::copy(&mut file, &mut hash).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", hash.finalize()))
}

fn discover_labels(
    path: &Path,
    labels: &mut BTreeSet<PathBuf>,
    depth: usize,
) -> Result<(), String> {
    if depth > 8 {
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        if p.to_string_lossy().contains("assistant-visual") {
            continue;
        }
        if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            discover_labels(&p, labels, depth + 1)?;
        } else if p.to_string_lossy().ends_with(".labels.json")
            && !p.to_string_lossy().contains("backup")
        {
            labels.insert(p);
        }
    }
    Ok(())
}
fn score_labels(
    capture: &Path,
    labels: &BTreeSet<PathBuf>,
    ellipses: &BTreeMap<(u64, u64), Ellipse>,
) -> Result<Value, String> {
    let mut index = BTreeMap::new();
    for row in rows(&capture.join("frames.jsonl"))? {
        let v = row?;
        if let (Some(roi), Some(seq)) = (number(&v["eye_id"]), number(&v["sequence"])) {
            index.insert((roi, seq), v);
        }
    }
    let mut results = Vec::new();
    let mut seen_raw = BTreeSet::new();
    for path in labels {
        let v: Value = serde_json::from_reader(File::open(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        if v["reviewed"] != true {
            continue;
        }
        let Some(raw) = v["source_raw"].as_str() else {
            continue;
        };
        let stem = Path::new(raw)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        let Some((eye, seq)) = stem.rsplit_once("-seq-") else {
            continue;
        };
        let Ok(seq) = seq.parse::<u64>() else {
            continue;
        };
        let roi = if eye == "subject-right" {
            1
        } else if eye == "subject-left" {
            2
        } else {
            continue;
        };
        let Some(f) = index.get(&(roi, seq)) else {
            continue;
        };
        let Some(ns) = number(&f["timestamp_ns"]) else {
            continue;
        };
        let Some(e) = ellipses.get(&(roi, ns)) else {
            continue;
        };
        if array::<2>(&v["sensor_origin"])
            != Some([
                f["sensor_x"].as_f64().unwrap_or(f64::NAN),
                f["sensor_y"].as_f64().unwrap_or(f64::NAN),
            ])
        {
            continue;
        }
        let Ok(raw_bytes) = fs::read(raw) else {
            continue;
        };
        if number(&f["length"]) != Some(raw_bytes.len() as u64) {
            continue;
        }
        let Some(stream) = f["stream"].as_str() else {
            continue;
        };
        // Only a local basename from the native capture index may select a RAW stream.
        if Path::new(stream).components().count() != 1 {
            continue;
        }
        let Ok(mut input) = File::open(capture.join(stream)) else {
            continue;
        };
        input
            .seek(SeekFrom::Start(
                number(&f["offset"]).ok_or("missing RAW offset")?,
            ))
            .map_err(|e| e.to_string())?;
        let mut bytes = vec![0; raw_bytes.len()];
        input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        if bytes != raw_bytes {
            continue;
        }
        let hash = format!("{:x}", Sha256::digest(&bytes));
        if !seen_raw.insert(hash.clone()) {
            continue;
        }
        let (s, c) = e.angle.sin_cos();
        let rim: Vec<_> = (0..3600)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 3600.0;
                [
                    e.center[0] + e.major * a.cos() * c - e.minor * a.sin() * s,
                    e.center[1] + e.major * a.cos() * s + e.minor * a.sin() * c,
                ]
            })
            .collect();
        let distances: Vec<_> = v["annotation_points"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| {
                if p["kind"] != "iris_edge" || p["visibility"] != "visible" {
                    return None;
                }
                let x = p["x_sensor"].as_f64()?;
                let y = p["y_sensor"].as_f64()?;
                Some(
                    rim.iter()
                        .map(|q| (q[0] - x).hypot(q[1] - y))
                        .fold(f64::INFINITY, f64::min),
                )
            })
            .collect();
        if distances.is_empty() {
            continue;
        }
        results.push(json!({"label":path,"roi":roi,"sequence":seq,"verified_raw_sha256":hash,"provenance":v["provenance"],"schema":v["schema"],
            "visible_human_points":distances.len(),"median_distance_px":percentile(&distances,0.5),"max_distance_px":percentile(&distances,1.0),
            "rim_discretization_error_upper_px":std::f64::consts::PI*e.major/3600.0,"baseline_candidate_delta_px":0.0}));
    }
    Ok(
        json!({"matched_labels":results,"metric":"nearest ellipse rim sampled at3600angles; supplied visible human midpoint/edge points, not sign labels","candidate_changes_input_ellipse":false}),
    )
}

#[derive(Clone)]
struct Frame {
    ns: u64,
    seq: u64,
    roi: u64,
    epoch: String,
    attested_epoch: bool,
    origin: [f64; 2],
    region_session: Option<u64>,
    region_generation: Option<u64>,
}
fn frame(v: &Value, capture: &Path) -> Option<Frame> {
    let epoch = v
        .pointer("/source_clock/source_key/stream_epoch")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| number(&v["region"]["session"]).map(|n| format!("region-session:{n}")));
    Some(Frame {
        ns: number(&v["timestamp_ns"])?,
        seq: number(&v["sequence"])?,
        roi: number(&v["eye_id"])?,
        attested_epoch: epoch.is_some(),
        epoch: epoch.unwrap_or_else(|| format!("unattested-archive:{}", capture.display())),
        origin: [v["sensor_x"].as_f64()?, v["sensor_y"].as_f64()?],
        region_session: number(&v["region"]["session"]),
        region_generation: number(&v["region"]["generation"]),
    })
}
#[derive(Clone, Copy)]
struct Ellipse {
    center: [f64; 2],
    major: f64,
    minor: f64,
    angle: f64,
}
impl Ellipse {
    fn outside(self, p: [f64; 2]) -> bool {
        let (s, c) = self.angle.sin_cos();
        let x = p[0] - self.center[0];
        let y = p[1] - self.center[1];
        // Exclude iris and its uncertain edge; this is a skin/background proxy, not measured rigid head pose.
        ((c * x + s * y) / (self.major * 1.3 + 10.0)).powi(2)
            + ((-s * x + c * y) / (self.minor * 1.3 + 10.0)).powi(2)
            > 1.0
    }
    fn valid(self) -> bool {
        self.center.iter().all(|x| x.is_finite())
            && self.angle.is_finite()
            && self.major.is_finite()
            && self.minor.is_finite()
            && self.major >= self.minor
            && self.minor > 0.0
    }
    fn hypotheses(self) -> [Hypothesis; 2] {
        let z = self.minor / self.major;
        let tilt = (1.0 - z * z).max(0.0).sqrt();
        let n = [-self.angle.sin() * tilt, self.angle.cos() * tilt, z];
        let depth = self.major * (1.83_f64.powi(2) - 1.0).sqrt();
        [1.0, -1.0].map(|s| Hypothesis {
            normal_camera: [n[0] * s, n[1] * s, n[2]],
            effective_pivot_sensor_px: [
                self.center[0] - depth * n[0] * s,
                self.center[1] - depth * n[1] * s,
            ],
            pivot_sigma_px: 1.0 + 0.04 * depth,
        })
    }
}
fn contact_geometry(eye: &Value) -> Option<(&Value, u64)> {
    ["virtual_contact_surface_gaze", "surface_gaze"]
        .into_iter()
        .find_map(|name| {
            let v = &eye["centers_and_gaze"][name];
            Some((v, number(&v["source_timestamp_ns"])?))
        })
}
fn geometry(
    eye: &Value,
    source: &Frame,
    contact: Option<&Value>,
) -> Option<(Ellipse, &'static str)> {
    if let Some(v) = contact {
        let n = array::<3>(&v["relative_gaze_vector"])?;
        if (n.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() > 1e-5 || !(0.0..=1.0).contains(&n[2])
        {
            return None;
        }
        let near = array::<2>(&v["camera_near_point_sensor"])?;
        let radius = (v["rectified_area_px2"].as_f64()? / std::f64::consts::PI).sqrt();
        let quantized = v["bucketed_face_radius_px"].as_f64()?;
        let e = Ellipse {
            center: [near[0] - quantized * n[0], near[1] - quantized * n[1]],
            major: radius,
            minor: radius * n[2],
            angle: (-n[0]).atan2(n[1]),
        };
        return e
            .valid()
            .then_some((e, "recorded_contact_weak_perspective_counterfactual"));
    }
    let v = &eye["limbus"]["semantic_ellipse"];
    let e = Ellipse {
        center: sign_kinematic_beam::sensor_from_roi(array(&v["center"])?, source.origin),
        major: v["major_radius"].as_f64()?,
        minor: v["minor_radius"].as_f64()?,
        angle: v["angle_rad"].as_f64()?,
    };
    e.valid()
        .then_some((e, "native_semantic_ellipse_weak_perspective"))
}
#[derive(Clone)]
struct SourceFeatures {
    frame: Frame,
    points: BTreeMap<u64, [f64; 2]>,
}
fn features(eye: &Value, f: &Frame) -> SourceFeatures {
    let mut points = BTreeMap::new();
    if let Some(entries) = eye["motion"]["persistent_features"].as_array() {
        for v in entries {
            if number(&v["motion_layer"]) != Some(0)
                || v["layer_evidence"] != true
                || v["match_score"].as_f64().unwrap_or(0.0) < 0.6
                || number(&v["matched_streak"]).unwrap_or(0) < 3
            {
                continue;
            }
            if let (Some(id), Some(p)) = (number(&v["id"]), array(&v["point"])) {
                points.insert(id, sign_kinematic_beam::sensor_from_roi(p, f.origin));
            }
        }
    }
    SourceFeatures {
        frame: f.clone(),
        points,
    }
}
#[derive(Clone, Copy)]
struct Fit {
    center: [f64; 2],
    translation: [f64; 2],
    scale: f64,
    angle: f64,
    sigma: f64,
    count: usize,
}
struct ExteriorStep {
    roi: u64,
    from_ns: u64,
    to_ns: u64,
    from_sequence: u64,
    to_sequence: u64,
    epoch: String,
    fit: Fit,
}
fn exterior_steps(
    capture: &Path,
    filename: &str,
) -> Result<BTreeMap<(u64, u64), ExteriorStep>, String> {
    let path = capture.join(filename);
    let mut steps = BTreeMap::new();
    if !path.is_file() {
        return Ok(steps);
    }
    for row in rows(&path)? {
        let v = row?;
        if v["reliable"] != true {
            continue;
        }
        let parse = || -> Option<ExteriorStep> {
            Some(ExteriorStep {
                roi: number(&v["roi"])?,
                from_ns: number(&v["from_source_ns"])?,
                to_ns: number(&v["to_source_ns"])?,
                from_sequence: number(&v["from_sequence"])?,
                to_sequence: number(&v["to_sequence"])?,
                epoch: v["stream_epoch"].as_str()?.into(),
                fit: Fit {
                    center: array(&v["center_sensor_px"])?,
                    translation: array(&v["translation_px"])?,
                    scale: v["scale"].as_f64()?,
                    angle: v["angle_rad"].as_f64()?,
                    sigma: v["sigma_px"].as_f64()?,
                    count: number(&v["support"])? as usize,
                },
            })
        };
        let s = parse().ok_or("malformed reliable exterior-motion step")?;
        if s.to_ns <= s.from_ns
            || s.fit.count < 8
            || !s.fit.scale.is_finite()
            || s.fit.scale <= 0.0
            || !s.fit.angle.is_finite()
            || !s.fit.sigma.is_finite()
            || s.fit.sigma < 0.0
        {
            return Err("invalid exterior-motion support".into());
        }
        if steps.insert((s.roi, s.from_ns), s).is_some() {
            return Err("duplicate exterior-motion source endpoint".into());
        }
    }
    Ok(steps)
}
fn exterior_between(
    steps: &BTreeMap<(u64, u64), ExteriorStep>,
    previous: &Frame,
    current: &Frame,
) -> Option<Fit> {
    if current.epoch != previous.epoch
        || current.ns <= previous.ns
        || current.ns - previous.ns > 750_000_000
    {
        return None;
    }
    let mut at = previous.ns;
    let mut seq = previous.seq;
    let mut a = 1.0;
    let mut b = 0.0;
    let mut tx = 0.0;
    let mut ty = 0.0;
    let mut sigma = 0.0;
    let mut count = usize::MAX;
    for _ in 0..MAX_SOURCE_CACHE {
        let s = steps.get(&(current.roi, at))?;
        if s.epoch != current.epoch || s.from_sequence != seq || s.to_ns > current.ns {
            return None;
        }
        let (sin, cos) = s.fit.angle.sin_cos();
        let aa = s.fit.scale * cos;
        let bb = s.fit.scale * sin;
        let [cx, cy] = s.fit.center;
        let [dx, dy] = s.fit.translation;
        let nx = aa * tx - bb * ty + cx + dx - aa * cx + bb * cy;
        let ny = bb * tx + aa * ty + cy + dy - bb * cx - aa * cy;
        let na = aa * a - bb * b;
        let nb = bb * a + aa * b;
        a = na;
        b = nb;
        tx = nx;
        ty = ny;
        sigma = s.fit.scale * sigma + s.fit.sigma;
        count = count.min(s.fit.count);
        at = s.to_ns;
        seq = s.to_sequence;
        if at == current.ns {
            return (seq == current.seq).then_some(Fit {
                center: [0.0, 0.0],
                translation: [tx, ty],
                scale: a.hypot(b),
                angle: b.atan2(a),
                sigma,
                count,
            });
        }
    }
    None
}
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn fit_pairs(pairs: &[([f64; 2], [f64; 2])]) -> Option<Fit> {
    if pairs.len() < 8 {
        return None;
    }
    let count = pairs.len() as f64;
    let center = [0, 1].map(|i| pairs.iter().map(|p| p.0[i]).sum::<f64>() / count);
    let next = [0, 1].map(|i| pairs.iter().map(|p| p.1[i]).sum::<f64>() / count);
    let mut denom = 0.0;
    let mut a = 0.0;
    let mut b = 0.0;
    for (p, q) in pairs {
        let x = p[0] - center[0];
        let y = p[1] - center[1];
        let u = q[0] - next[0];
        let v = q[1] - next[1];
        denom += x * x + y * y;
        a += x * u + y * v;
        b += x * v - y * u;
    }
    if denom / count < 100.0 {
        return None;
    }
    a /= denom;
    b /= denom;
    let residuals: Vec<_> = pairs
        .iter()
        .map(|(p, q)| {
            let x = p[0] - center[0];
            let y = p[1] - center[1];
            (next[0] + a * x - b * y - q[0]).hypot(next[1] + b * x + a * y - q[1])
        })
        .collect();
    let sigma = (residuals.iter().map(|r| r * r).sum::<f64>() / count)
        .sqrt()
        .max(0.5);
    let scale = a.hypot(b);
    (scale.is_finite() && (0.90..=1.10).contains(&scale) && sigma <= 8.0).then_some(Fit {
        center,
        translation: [next[0] - center[0], next[1] - center[1]],
        scale,
        angle: b.atan2(a),
        sigma,
        count: pairs.len(),
    })
}
fn estimate_transport(
    previous: &SourceFeatures,
    current: &SourceFeatures,
    old: Ellipse,
    now: Ellipse,
    outside: bool,
) -> Option<Fit> {
    if previous.frame.epoch != current.frame.epoch {
        return None;
    }
    let pairs: Vec<_> = previous
        .points
        .iter()
        .filter_map(|(id, p)| {
            let q = *current.points.get(id)?;
            (!outside || (old.outside(*p) && now.outside(q))).then_some((*p, q))
        })
        .take(MAX_FEATURES)
        .collect();
    let fit = fit_pairs(&pairs)?;
    let (s, c) = fit.angle.sin_cos();
    let errors: Vec<_> = pairs
        .iter()
        .map(|(p, q)| {
            let x = p[0] - fit.center[0];
            let y = p[1] - fit.center[1];
            (fit.center[0] + fit.translation[0] + fit.scale * (c * x - s * y) - q[0])
                .hypot(fit.center[1] + fit.translation[1] + fit.scale * (s * x + c * y) - q[1])
        })
        .collect();
    let cut = (2.5 * median(errors.clone())).max(1.5);
    let kept: Vec<_> = pairs
        .iter()
        .zip(errors)
        .filter(|(_, e)| *e <= cut)
        .map(|(p, _)| *p)
        .collect();
    let mut result = fit_pairs(&kept)?;
    result.sigma = 2.0 * result.sigma + 2.0;
    Some(result)
}
fn head(fit: Fit, previous: &Frame, current: &Frame, stream: u64) -> HeadTransport {
    HeadTransport {
        from: SourceIdentity {
            stream,
            frame: previous.seq,
        },
        to: SourceIdentity {
            stream,
            frame: current.seq,
        },
        from_source_ns: previous.ns,
        to_source_ns: current.ns,
        center_sensor_px: fit.center,
        translation_px: fit.translation,
        scale: fit.scale,
        angle_rad: fit.angle,
        sigma_px: fit.sigma,
        rotation_vector_camera_rad: None,
    }
}
#[derive(Default)]
struct Stats {
    sparse_skips: u64,
    calls: u64,
    fresh: u64,
    resolved: u64,
    rejected: u64,
    missing: u64,
    transports: u64,
    disagreements: u64,
    baseline_resolved: u64,
    elapsed_ns: u128,
    max_states: usize,
    max_transitions: usize,
    flips: u64,
    angular_steps: Vec<f64>,
    previous: Option<([f64; 3], u64)>,
    reject_reasons: BTreeMap<String, u64>,
}
impl Stats {
    fn value(&self) -> Value {
        json!({"calls":self.calls,"fresh":self.fresh,"resolved_fresh":self.resolved,"rejected":self.rejected,
        "raw_snapshots_missing_geometry":self.missing,"intentional_sparse_source_skips":self.sparse_skips,"reject_reasons":self.reject_reasons,"transport_intervals":self.transports,"baseline_resolved_fresh":self.baseline_resolved,
        "disagree_with_recorded_branch_not_truth":self.disagreements,"strong_transverse_reversals":self.flips,
        "mean_tracker_us":self.elapsed_ns as f64 / self.calls.max(1) as f64 / 1000.0,
        "maximum_states":self.max_states,"maximum_transition_evaluations":self.max_transitions,
        "adjacent_resolved_angle_p95_deg":percentile(&self.angular_steps,0.95),"adjacent_resolved_angle_max_deg":percentile(&self.angular_steps,1.0)})
    }
}
fn percentile(xs: &[f64], q: f64) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * q) as usize])
}
fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| x * y)
        .sum::<f64>()
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn run_capture(
    capture: &Path,
    output: &mut impl Write,
    global: &mut [Stats; ARM_COUNT],
    labels: &BTreeSet<PathBuf>,
    motion_sidecar: &str,
) -> Result<Value, String> {
    let mut frames: BTreeMap<(u64, u64), Frame> = BTreeMap::new();
    let mut collisions = BTreeSet::new();
    let mut raw_rows = 0;
    let mut unattested = 0;
    for row in rows(&capture.join("frames.jsonl"))? {
        let row = row?;
        raw_rows += 1;
        if let Some(f) = frame(&row, capture) {
            unattested += usize::from(!f.attested_epoch);
            let key = (f.roi, f.ns);
            // Ambiguous repeated timestamps across lineages/origins are excluded, never overwritten.
            if frames.get(&key).is_some_and(|old| {
                old.epoch != f.epoch
                    || old.seq != f.seq
                    || old.origin != f.origin
                    || old.region_session != f.region_session
                    || old.region_generation != f.region_generation
            }) {
                collisions.insert(key);
            }
            frames.entry(key).or_insert(f);
        }
    }
    for key in &collisions {
        frames.remove(key);
    }
    let exterior = exterior_steps(capture, motion_sidecar)?;
    let mut streams = BTreeMap::<String, u64>::new();
    let mut trackers = BTreeMap::<u64, [SignKinematicBeam; ARM_COUNT]>::new();
    let mut stats: BTreeMap<u64, [Stats; ARM_COUNT]> = BTreeMap::new();
    let mut sparse_previous = BTreeMap::<(u64, usize), (SourceFeatures, Ellipse)>::new();
    let mut source_counts = BTreeMap::<u64, usize>::new();
    let mut cache = BTreeMap::<(u64, u64), SourceFeatures>::new();
    let mut previous = BTreeMap::<u64, (SourceFeatures, Ellipse)>::new();
    let mut seen = BTreeSet::new();
    let mut missing_seen = BTreeSet::new();
    let mut last_epoch = BTreeMap::new();
    let mut kinds = BTreeMap::<&str, u64>::new();
    let mut missing_source = 0;
    let mut prediction_rows = 0;
    let mut lag = Vec::new();
    let mut reframes = 0;
    let mut last_origin = BTreeMap::new();
    let mut baseline_previous = BTreeMap::<u64, ([f64; 3], u64)>::new();
    let mut baseline_flips = 0;
    let mut baseline_steps = Vec::new();
    let mut scale_pairs = 0;
    let mut scale_log_steps = Vec::new();
    let mut no_geometry = 0;
    let mut source_ellipses = BTreeMap::new();
    for row in rows(&capture.join("predictions.jsonl"))? {
        let row = row?;
        prediction_rows += 1;
        let Some(roi) = number(&row["roi_frame_key"]["roi_id"]) else {
            missing_source += 1;
            continue;
        };
        let Some(presentation_ns) = number(&row["roi_frame_key"]["sensor_timestamp_ns"]) else {
            missing_source += 1;
            continue;
        };
        let Some(present) = frames.get(&(roi, presentation_ns)) else {
            missing_source += 1;
            continue;
        };
        if number(&row["roi_frame_key"]["sequence"]) != Some(present.seq)
            || row
                .pointer("/source_clock/source_key/stream_epoch")
                .and_then(Value::as_str)
                .is_some_and(|e| e != present.epoch)
        {
            missing_source += 1;
            continue;
        }
        if last_epoch
            .insert(roi, present.epoch.clone())
            .is_some_and(|e| e != present.epoch)
        {
            previous.remove(&roi);
            sparse_previous.retain(|(r, _), _| *r != roi);
            source_counts.remove(&roi);
            baseline_previous.remove(&roi);
            last_origin.remove(&roi);
            cache.retain(|(id, _), _| *id != roi);
            if let Some(arms) = stats.get_mut(&roi) {
                for s in arms {
                    s.previous = None;
                }
            }
        }
        if last_origin
            .insert(roi, present.origin)
            .is_some_and(|o| o != present.origin)
        {
            reframes += 1;
        }
        let eye = &row["predictions"]["eye_candidate"];
        cache.insert((roi, presentation_ns), features(eye, present));
        while cache.len() > MAX_SOURCE_CACHE * 2 {
            let k = *cache.keys().min_by_key(|(_, ns)| ns).unwrap();
            cache.remove(&k);
        }
        let contact = contact_geometry(eye);
        let source_ns = contact.map_or(presentation_ns, |(_, ns)| ns);
        let Some(source) = frames
            .get(&(roi, source_ns))
            .filter(|f| f.epoch == present.epoch && source_ns <= presentation_ns)
        else {
            missing_source += 1;
            continue;
        };
        let stream_next = streams.len() as u64 + 1;
        let stream = *streams.entry(source.epoch.clone()).or_insert(stream_next);
        let source_id = SourceIdentity {
            stream,
            frame: source.seq,
        };
        let geom = geometry(eye, source, contact.map(|(v, _)| v));
        let Some((ellipse, kind)) = geom else {
            no_geometry += 1;
            // A publication snapshot lacking asynchronous geometry is not a source-bound blink.
            // Count absence without advancing the estimate stream past answers that arrive later.
            if missing_seen.insert((roi, source.epoch.clone(), source_ns)) {
                let arm_stats = stats
                    .entry(roi)
                    .or_insert_with(|| std::array::from_fn(|_| Stats::default()));
                for i in 0..ARM_COUNT {
                    arm_stats[i].missing += 1;
                    global[i].missing += 1;
                }
            }
            continue;
        };
        let fresh = seen.insert((roi, source.epoch.clone(), source_ns));
        let source_count = source_counts.entry(roi).or_default();
        let admission: [bool; ARM_COUNT] =
            std::array::from_fn(|i| i < 3 || (fresh && *source_count % STRIDES[i] == 0));
        if fresh {
            *source_count += 1;
        }
        source_ellipses.entry((roi, source.ns)).or_insert(ellipse);
        if fresh {
            lag.push((presentation_ns - source_ns) as f64 / 1e6);
        }
        *kinds.entry(kind).or_default() += 1;
        let hypotheses = ellipse.hypotheses();
        let baseline = contact.and_then(|(v, _)| array::<3>(&v["relative_gaze_vector"]));
        let baseline_resolved = contact.is_some_and(|(v, _)| v["sign_resolved"] == true);
        if fresh {
            if let Some(n) = baseline {
                if let Some((p, t)) = baseline_previous.insert(roi, (n, source_ns)) {
                    if source_ns > t && source_ns - t <= 500_000_000 {
                        baseline_steps.push(angle(p, n));
                        baseline_flips +=
                            usize::from((0..2).any(|i| {
                                p[i] * n[i] < 0.0 && p[i].abs() > 0.1 && n[i].abs() > 0.1
                            }));
                    }
                }
            }
        }
        let current_features = cache.get(&(roi, source_ns));
        let old = previous.get(&roi);
        let fits = [true, false].map(|outside| {
            if outside {
                if let Some((p, _)) = old {
                    if let Some(fit) = exterior_between(&exterior, &p.frame, source) {
                        return Some(fit);
                    }
                }
            }
            current_features.zip(old).and_then(|(f, (p, e))| {
                (source_ns > p.frame.ns && source_ns - p.frame.ns <= 750_000_000)
                    .then(|| estimate_transport(p, f, *e, ellipse, outside))
                    .flatten()
            })
        });
        let base_heads = [fits[0], None, fits[1]].map(|fit| {
            fit.zip(old)
                .map(|(fit, (p, _))| head(fit, &p.frame, source, stream))
        });
        let heads: [Option<HeadTransport>; ARM_COUNT] = std::array::from_fn(|i| {
            if i < 3 {
                return base_heads[i];
            }
            if !admission[i] {
                return None;
            }
            let (p, e) = sparse_previous.get(&(roi, i))?;
            if let Some(fit) = exterior_between(&exterior, &p.frame, source) {
                return Some(head(fit, &p.frame, source, stream));
            }
            let f = current_features?;
            if source_ns <= p.frame.ns || source_ns - p.frame.ns > 750_000_000 {
                return None;
            }
            let fit = estimate_transport(p, f, *e, ellipse, true)?;
            Some(head(fit, &p.frame, source, stream))
        });
        if fresh {
            if let Some((fit, (_, old_ellipse))) = fits[0].zip(old) {
                scale_pairs += 1;
                scale_log_steps.push(
                    (2.0 * (ellipse.major / old_ellipse.major).ln() - 2.0 * fit.scale.ln()).abs(),
                );
            }
        }
        let states = trackers
            .entry(roi)
            .or_insert_with(|| std::array::from_fn(|_| SignKinematicBeam::default()));
        let eye_stats = stats
            .entry(roi)
            .or_insert_with(|| std::array::from_fn(|_| Stats::default()));
        let mut outcomes = Vec::new();
        for arm in 0..ARM_COUNT {
            if !admission[arm] {
                if fresh {
                    eye_stats[arm].sparse_skips += 1;
                    global[arm].sparse_skips += 1;
                }
                outcomes.push(json!({"arm":ARMS[arm],"status":"SparseIntervalSkipped","selected":null,"support":0,"independent_costs":null}));
                continue;
            }
            let before = Instant::now();
            let o = states[arm].observe(Observation {
                source: source_id,
                source_ns: Some(source_ns),
                fresh,
                visible: true,
                hypotheses,
                head: heads[arm],
                anchor: None,
            });
            let elapsed = before.elapsed().as_nanos();
            for s in [&mut eye_stats[arm], &mut global[arm]] {
                s.calls += 1;
                s.fresh += u64::from(fresh);
                s.resolved += u64::from(fresh && o.selected.is_some());
                s.rejected += u64::from(o.status == Status::Rejected);
                s.missing += u64::from(o.status == Status::MissingGeometry);
                if let Some(reason) = o.reject_reason {
                    *s.reject_reasons.entry(format!("{:?}", reason)).or_default() += 1;
                }
                s.transports += u64::from(fresh && heads[arm].is_some());
                s.baseline_resolved += u64::from(fresh && baseline_resolved);
                s.elapsed_ns += elapsed;
                s.max_states = s.max_states.max(o.states);
                s.max_transitions = s.max_transitions.max(o.transitions);
                if fresh {
                    if let (Some(i), Some(n)) = (o.selected, baseline) {
                        s.disagreements += u64::from(angle(hypotheses[i].normal_camera, n) > 1.0);
                    }
                }
            }
            // Per-eye adjacency only; never form cross-capture/cross-eye global kinematics.
            if fresh {
                let s = &mut eye_stats[arm];
                if let Some(i) = o.selected {
                    let n = hypotheses[i].normal_camera;
                    if let Some((p, t)) = s.previous {
                        if source_ns > t && source_ns - t <= 500_000_000 {
                            s.angular_steps.push(angle(p, n));
                            s.flips += u64::from((0..2).any(|j| {
                                p[j] * n[j] < 0.0 && p[j].abs() > 0.1 && n[j].abs() > 0.1
                            }));
                        }
                    }
                    s.previous = Some((n, source_ns));
                } else {
                    s.previous = None;
                }
            }
            outcomes.push(json!({"arm":ARMS[arm],"status":format!("{:?}",o.status),"reject_reason":o.reject_reason.map(|r|format!("{:?}",r)),"selected":o.selected,
                "support":o.support,"independent_costs":o.independent_costs,"beam_costs":o.beam_costs,
                "mode":format!("{:?}",o.mode),"states":o.states,"transitions":o.transitions}));
            if arm >= 3 && fresh {
                if let Some(f) = current_features {
                    sparse_previous.insert((roi, arm), (f.clone(), ellipse));
                } else {
                    sparse_previous.remove(&(roi, arm));
                }
            }
        }
        if fresh {
            if let Some(f) = current_features {
                previous.insert(roi, (f.clone(), ellipse));
            } else {
                previous.remove(&roi);
            }
        }
        serde_json::to_writer(&mut *output,&json!({"capture":capture,"roi":roi,"source_ns":source_ns.to_string(),"sequence":source.seq,
            "epoch_attested":source.attested_epoch,"source_epoch":source.epoch,"fresh":fresh,"geometry_origin":kind,
            "source_lag_ms":(presentation_ns-source_ns)as f64/1e6,"normal_candidates":hypotheses.map(|h|h.normal_camera),
            "baseline_normal":baseline,"baseline_resolved":baseline_resolved,"transport_support":fits.map(|f|f.map(|f|f.count)),
            "outcomes":outcomes})).map_err(|e|e.to_string())?;
        writeln!(output).map_err(|e| e.to_string())?;
    }
    for arms in stats.values() {
        for (i, s) in arms.iter().enumerate() {
            global[i].flips += s.flips;
            global[i].angular_steps.extend(&s.angular_steps);
        }
    }
    Ok(
        json!({"capture":capture,"frames_sha256":digest(&capture.join("frames.jsonl"))?,"predictions_sha256":digest(&capture.join("predictions.jsonl"))?,
        "raw_index_rows":raw_rows,"unattested_epoch_rows":unattested,"prediction_rows":prediction_rows,"missing_source_rows":missing_source,
        "ambiguous_source_timestamp_keys_excluded":collisions.len(),
        "native_raw_exterior_steps":exterior.len(),"native_raw_exterior_sidecar_sha256":if exterior.is_empty(){None}else{Some(digest(&capture.join(motion_sidecar))?)},
        "no_geometry_rows":no_geometry,"geometry_origins":kinds,"roi_origin_changes":reframes,
        "source_lag_p50_ms":percentile(&lag,0.5),"source_lag_p95_ms":percentile(&lag,0.95),
        "recorded_baseline_strong_transverse_reversals":baseline_flips,"recorded_baseline_max_adjacent_angle_deg":percentile(&baseline_steps,1.0),
        "independent_feature_scale_intervals":scale_pairs,"mean_absolute_log_sn_feida_step":if scale_pairs>0{Some(scale_log_steps.iter().sum::<f64>()/scale_pairs as f64)}else{None},
        "sn_feida_baseline_candidate_delta":0.0,"ellipse_localization_baseline_candidate_delta":0.0,
        "human_localization":score_labels(capture,labels,&source_ellipses)?,
        "eyes":stats.into_iter().map(|(roi,arms)|json!({"roi":roi,"arms":arms.iter().enumerate().map(|(i,s)|json!({"name":ARMS[i],"metrics":s.value()})).collect::<Vec<_>>()})).collect::<Vec<_>>()}),
    )
}
fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let output = PathBuf::from(args.next().ok_or(
        "usage: buttercup_kinematic_sign_replay OUTPUT.json CAPTURE_OR_EXTRACTED_ROOT...",
    )?);
    let mut roots = Vec::new();
    let mut label_roots = Vec::new();
    let mut motion_sidecar = "exterior-motion.jsonl".to_string();
    while let Some(arg) = args.next() {
        if arg == "--labels" {
            label_roots.push(PathBuf::from(
                args.next().ok_or("--labels needs directory")?,
            ));
        } else if arg == "--motion-sidecar" {
            motion_sidecar = args
                .next()
                .and_then(|s| s.into_string().ok())
                .ok_or("--motion-sidecar needs a local filename")?;
            if !matches!(
                Path::new(&motion_sidecar)
                    .components()
                    .collect::<Vec<_>>()
                    .as_slice(),
                [std::path::Component::Normal(_)]
            ) {
                return Err("motion sidecar must be a local basename".into());
            }
        } else {
            roots.push(PathBuf::from(arg));
        }
    }
    if roots.is_empty() {
        return Err("provide capture roots".into());
    }
    let mut labels = BTreeSet::new();
    for root in &label_roots {
        discover_labels(root, &mut labels, 0)?;
    }
    let mut dirs = BTreeSet::new();
    for root in &roots {
        discover(root, &mut dirs, 0)?;
    }
    if dirs.is_empty() {
        return Err("no extracted frames.jsonl + predictions.jsonl pairs found".into());
    }
    let mut cases = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output.with_extension("cases.jsonl"))
            .map_err(|e| e.to_string())?,
    );
    let mut report = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|e| e.to_string())?,
    );
    let mut global: [Stats; ARM_COUNT] = std::array::from_fn(|_| Stats::default());
    let mut captures = Vec::new();
    let mut errors = Vec::new();
    let mut copies = Vec::new();
    let mut hashes = BTreeSet::new();
    for dir in dirs {
        let hash = (
            digest(&dir.join("frames.jsonl"))?,
            digest(&dir.join("predictions.jsonl"))?,
        );
        if !hashes.insert(hash) {
            copies.push(dir);
            continue;
        }
        match run_capture(&dir, &mut cases, &mut global, &labels, &motion_sidecar) {
            Ok(result) => {
                eprintln!(
                    "replayed {}: {} prediction rows",
                    dir.display(),
                    result["prediction_rows"]
                );
                captures.push(result);
            }
            Err(e) => errors.push(json!({"capture":dir,"error":e})),
        }
    }
    cases.flush().map_err(|e| e.to_string())?;
    let value = json!({"schema":"buttercup-kinematic-sign-replay-v1","roots":roots,"label_roots":label_roots,"label_files_considered":labels.len(),"captures":captures,"errors":errors,"exact_index_prediction_copies_skipped":copies,
        "motion_sidecar_filename":motion_sidecar,
        "tracker_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../eye_scene_model/sign_kinematic_beam.rs"))),
        "replay_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("buttercup_kinematic_sign_replay.rs"))),
        "arms":global.iter().enumerate().map(|(i,s)|json!({"name":ARMS[i],"metrics":s.value()})).collect::<Vec<_>>(),
        "scope":"single-user recorded-estimate diagnostic; no fresh inference, training, live authority or signed truth",
        "limitations":["Weak-perspective +/- conic counterfactuals are not the exact two pinhole off-axis solutions of the joint solver.",
            "Surrounding-tissue general-layer correspondence excludes an expanded iris but remains defeasible non-rigid transport, not calibrated head pose.",
            "No independently measured 3D head rotation, angular uncertainty, metric scale, fixation truth or signed 3D human labels.",
            "Older missing stream epochs receive isolated archive-local namespaces; no cross-archive temporal continuity is asserted.",
            "Input ellipses are unchanged: localization and SN-FEIDA cannot improve by selecting a sign; scale intervals use matched surrounding features.",
            "Recorded baseline agreement, smoothness and higher resolved coverage are not correctness. Negative control permits iris contamination deliberately.",
            "Hash deduplication covers identical frame-index AND prediction files, not partially overlapping captures or RAW identity.",
            "Timing excludes JSON IO and feature fit and uses shared host resources; it is not an isolated performance benchmark."]});
    serde_json::to_writer_pretty(&mut report, &value).map_err(|e| e.to_string())?;
    writeln!(report).map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({"report":output,"captures":value["captures"].as_array().unwrap().len(),"errors":value["errors"],"arms":value["arms"]})
    );
    if value["errors"]
        .as_array()
        .is_some_and(|errors| !errors.is_empty())
    {
        return Err("one or more captures failed; inspect the partial report errors".into());
    }
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
    #[test]
    fn native_transport_recovers_similarity_and_rejects_iris_only_support() {
        let e = Ellipse {
            center: [100.0, 100.0],
            major: 20.0,
            minor: 15.0,
            angle: 0.0,
        };
        let f = Frame {
            ns: 1,
            seq: 1,
            roi: 1,
            epoch: "test".into(),
            attested_epoch: true,
            origin: [0.0, 0.0],
            region_session: None,
            region_generation: None,
        };
        let points: BTreeMap<_, _> = (0..16)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 16.0;
                (i, [100.0 + 60.0 * a.cos(), 100.0 + 60.0 * a.sin()])
            })
            .collect();
        let old = SourceFeatures {
            frame: f.clone(),
            points: points.clone(),
        };
        let new = SourceFeatures {
            frame: f,
            points: points
                .into_iter()
                .map(|(i, p)| (i, [p[0] + 4.0, p[1] - 3.0]))
                .collect(),
        };
        let fit = estimate_transport(&old, &new, e, e, true).unwrap();
        assert!((fit.translation[0] - 4.0).abs() < 1e-9);
        assert!((fit.scale - 1.0).abs() < 1e-9);
        let large = Ellipse {
            major: 200.0,
            minor: 150.0,
            ..e
        };
        assert!(estimate_transport(&old, &new, large, large, true).is_none());
        assert!(estimate_transport(&old, &new, large, large, false).is_some());
    }
    #[test]
    fn clocks_parse_losslessly_and_conic_signs_have_equal_area() {
        let ns = 1_690_004_708_093_859_187_u64;
        assert_eq!(number(&json!(ns.to_string())), Some(ns));
        let e = Ellipse {
            center: [110.0, 190.0],
            major: 80.0,
            minor: 60.0,
            angle: 0.4,
        };
        let h = e.hypotheses();
        assert_eq!(h[0].normal_camera[0], -h[1].normal_camera[0]);
        assert_eq!(h[0].normal_camera[2], h[1].normal_camera[2]);
    }
    #[test]
    fn exterior_composition_requires_exact_endpoints_and_propagates_uncertainty() {
        let first = Frame {
            ns: 100_000_000,
            seq: 1,
            roi: 1,
            epoch: "test".into(),
            attested_epoch: true,
            origin: [0.0, 0.0],
            region_session: None,
            region_generation: None,
        };
        let last = Frame {
            ns: 300_000_000,
            seq: 3,
            ..first.clone()
        };
        let mut steps = BTreeMap::new();
        for i in 1..3 {
            steps.insert(
                (1, i * 100_000_000),
                ExteriorStep {
                    roi: 1,
                    from_ns: i * 100_000_000,
                    to_ns: (i + 1) * 100_000_000,
                    from_sequence: i,
                    to_sequence: i + 1,
                    epoch: "test".into(),
                    fit: Fit {
                        center: [500.0, 700.0],
                        translation: [2.0, -3.0],
                        scale: 1.0,
                        angle: 0.0,
                        sigma: 0.5,
                        count: 12,
                    },
                },
            );
        }
        let result = exterior_between(&steps, &first, &last).unwrap();
        assert_eq!(result.translation, [4.0, -6.0]);
        assert_eq!(result.sigma, 1.0);
        let wrong = Frame {
            seq: 4,
            ..last.clone()
        };
        assert!(exterior_between(&steps, &first, &wrong).is_none());
        let foreign = Frame {
            epoch: "other".into(),
            ..last.clone()
        };
        assert!(exterior_between(&steps, &first, &foreign).is_none());
        steps.remove(&(1, 200_000_000));
        assert!(exterior_between(&steps, &first, &last).is_none());
    }
}
