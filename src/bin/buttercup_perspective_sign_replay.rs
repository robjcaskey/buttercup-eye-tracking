//! Offline conventional perspective-twin replay; no model loads or live actions.
#![allow(dead_code)]
#[path = "../geometry.rs"]
mod geometry;
#[path = "../raw10.rs"]
mod raw10;
#[path = "../"]
mod native {
    pub(crate) mod binocular_coordinator;
    pub(crate) mod conic_solver;
    pub(crate) mod outline_conic_segments;
    pub(crate) mod roi_evidence;
    pub(crate) mod eye_scene_model {
        pub(crate) mod binocular_pose;
    }
}
use native::{
    binocular_coordinator, conic_solver, eye_scene_model, outline_conic_segments, roi_evidence,
};
#[path = "../eye_scene_model/perspective_sign.rs"]
mod perspective_sign;
use conic_solver::joint::{circle_pose_hypotheses, PinholeCamera, ProjectedCircle};
use geometry::Ellipse;
use perspective_sign::{angle, Observation, Pose, Source, Tracker, Transport};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

fn num(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_str()?.parse().ok())
}
fn arr<const N: usize>(v: &Value) -> Option<[f64; N]> {
    let a = v.as_array()?;
    if a.len() != N {
        return None;
    }
    let mut out = [0.; N];
    for i in 0..N {
        out[i] = a[i].as_f64()?;
    }
    Some(out)
}
fn rows(p: &Path) -> Result<Vec<Value>, String> {
    BufReader::new(File::open(p).map_err(|e| format!("{}: {e}", p.display()))?)
        .lines()
        .map(|l| serde_json::from_str(&l.map_err(|e| e.to_string())?).map_err(|e| e.to_string()))
        .collect()
}
fn digest(p: &Path) -> Result<String, String> {
    let mut f = File::open(p).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", h.finalize()))
}
fn discover(p: &Path, d: &mut BTreeSet<PathBuf>) -> Result<(), String> {
    if p.join("frames.jsonl").is_file() && p.join("predictions.jsonl").is_file() {
        d.insert(fs::canonicalize(p).map_err(|e| e.to_string())?);
        return Ok(());
    }
    for e in fs::read_dir(p).map_err(|e| e.to_string())? {
        let e = e.map_err(|e| e.to_string())?;
        if e.file_type().map_err(|e| e.to_string())?.is_dir() {
            discover(&e.path(), d)?;
        }
    }
    Ok(())
}

/// Invert the exact saved forward projection conditional on C,n,K. The saved
/// area supplies projected major radius, not an invented minor/axis direction.
fn recover(camera: PinholeCamera, c: [f64; 3], n: [f64; 3], major: f64) -> Option<(Ellipse, f64)> {
    if !major.is_finite() || major <= 0. || c[2] >= 0. {
        return None;
    }
    let mut lo = 0.;
    let mut hi = -c[2] * 0.9;
    if ProjectedCircle::project(camera, c, n, hi, [0, 0])?
        .ellipse()?
        .major_radius
        < major
    {
        return None;
    }
    for _ in 0..64 {
        let mid = (lo + hi) * 0.5;
        let e = ProjectedCircle::project(camera, c, n, mid, [0, 0])?.ellipse()?;
        if e.major_radius < major {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r = (lo + hi) * 0.5;
    let e = ProjectedCircle::project(camera, c, n, r, [0, 0])?.ellipse()?;
    ((e.major_radius - major).abs() < 1e-5 * major.max(1.)).then_some((e, r))
}
fn exact_poses(camera: PinholeCamera, e: Ellipse, r: f64, allowance: f64) -> Option<[Pose; 2]> {
    let twins = circle_pose_hypotheses(camera, e, [0, 0])?;
    let mut out = [Pose {
        normal: [0.; 3],
        pivot_px: [0.; 2],
        pivot_grid_px: [[0.; 2]; 5],
        allowance_px: allowance,
        uncertainty: None,
    }; 2];
    for i in 0..2 {
        let p = twins[i];
        let c = p.center_per_radius.map(|x| x * r);
        let mut grid = [[0.; 2]; 5];
        for (d, ratio) in perspective_sign::DEPTH_RATIOS.into_iter().enumerate() {
            grid[d] = camera.project(std::array::from_fn(|j| c[j] - r * ratio * p.normal[j]))?;
        }
        out[i] = Pose {
            normal: p.normal,
            pivot_px: grid[2],
            pivot_grid_px: grid,
            allowance_px: allowance,
            uncertainty: None,
        };
    }
    Some(out)
}
/// S=t I + [[u,v],[v,-u]]. Its perturbation norm is |dt|+hypot(du,dv).
/// This representation remains meaningful when the major-axis angle does not.
fn perturb_shape(mut e: Ellipse, delta: [f64; 3]) -> Option<Ellipse> {
    let t = (e.major_radius + e.minor_radius) * 0.5 + delta[0];
    let gap = (e.major_radius - e.minor_radius) * 0.5;
    let u = gap * (2. * e.angle).cos() + delta[1];
    let v = gap * (2. * e.angle).sin() + delta[2];
    let h = u.hypot(v);
    e.major_radius = t + h;
    e.minor_radius = t - h;
    e.angle = 0.5 * v.atan2(u);
    (e.minor_radius > 0.).then_some(e)
}
fn associate(reference: [Pose; 2], mut pair: [Pose; 2]) -> [Pose; 2] {
    let direct =
        angle(reference[0].normal, pair[0].normal) + angle(reference[1].normal, pair[1].normal);
    let swap =
        angle(reference[0].normal, pair[1].normal) + angle(reference[1].normal, pair[0].normal);
    if swap < direct {
        pair.swap(0, 1);
    }
    pair
}
fn shape_stencil() -> Vec<[f64; 3]> {
    let mut result = Vec::with_capacity(42);
    for t in -1..=1 {
        for u in -1..=1 {
            for v in -1..=1 {
                let d = [t as f64, u as f64, v as f64];
                let norm = d[0].abs() + d[1].hypot(d[2]);
                if norm > 0. {
                    result.push(d.map(|x| x / norm));
                }
            }
        }
    }
    for k in 0..16 {
        let phi = std::f64::consts::TAU * k as f64 / 16.;
        result.push([0., phi.cos(), phi.sin()]);
    }
    result
}
fn sampled_envelope(
    camera: PinholeCamera,
    e: Ellipse,
    r: f64,
    reference: [Pose; 2],
    budget: f64,
) -> ([[f64; 5]; 2], [f64; 2], usize) {
    let mut pivot = [[[0_f64; 5]; 2]; 2];
    let mut angular = [[0_f64; 2]; 2];
    let mut invalid = 0;
    for family in 0..2 {
        let count = if family == 0 { 43 } else { 8 };
        let stencil = shape_stencil();
        for k in 0..count {
            let sample = if family == 0 && k == 42 {
                // Probe the interior circular shape explicitly: outer-shell
                // samples can jump across its singular axis gauge.
                let h = ((e.major_radius - e.minor_radius) * 0.5).min(budget);
                perturb_shape(
                    e,
                    [0., -h * (2. * e.angle).cos(), -h * (2. * e.angle).sin()],
                )
            } else if family == 0 {
                perturb_shape(e, stencil[k].map(|x| budget * x))
            } else {
                let mut sample = e;
                let phi = std::f64::consts::TAU * k as f64 / 8.;
                sample.center.0 += budget * phi.cos();
                sample.center.1 += budget * phi.sin();
                Some(sample)
            };
            let Some(pair) = sample.and_then(|s| exact_poses(camera, s, r, 0.)) else {
                invalid += 1;
                continue;
            };
            let pair = associate(reference, pair);
            for i in 0..2 {
                angular[family][i] =
                    angular[family][i].max(angle(reference[i].normal, pair[i].normal));
                for d in 0..5 {
                    let p = pair[i].pivot_grid_px[d];
                    let q = reference[i].pivot_grid_px[d];
                    pivot[family][i][d] = pivot[family][i][d].max((p[0] - q[0]).hypot(p[1] - q[1]));
                }
            }
        }
    }
    // Sum separate full-budget extrema: explicit relaxed budget, not quadrature
    // or confidence. The finite sampling still does not certify the continuum.
    (
        std::array::from_fn(|i| std::array::from_fn(|d| pivot[0][i][d] + pivot[1][i][d])),
        std::array::from_fn(|i| angular[0][i] + angular[1][i]),
        invalid,
    )
}
fn poses(camera: PinholeCamera, e: Ellipse, r: f64, allowance: f64) -> Option<[Pose; 2]> {
    poses_split(camera, e, r, allowance, 0.5)
}
fn poses_split(
    camera: PinholeCamera,
    e: Ellipse,
    r: f64,
    allowance: f64,
    shared_fraction: f64,
) -> Option<[Pose; 2]> {
    let mut out = exact_poses(camera, e, r, allowance)?;
    let (full_pivot, angular, mut invalid) = sampled_envelope(camera, e, r, out, allowance);
    let mut uncertainty = [perspective_sign::ConicUncertainty {
        normals: [[0.; 3]; perspective_sign::SHAPE_STATES],
        pivots: [[[0.; 2]; 5]; perspective_sign::SHAPE_STATES],
        innovation_px: [[0.; 5]; perspective_sign::SHAPE_STATES],
        angular_radius_rad: 0.,
        invalid_samples: 0,
        full_budget_pivot_radius_px: [0.; 5],
    }; 2];
    for state in 0..perspective_sign::SHAPE_STATES {
        let mut delta = [0.; 3];
        if state > 0 {
            delta[(state - 1) / 2] =
                if state % 2 == 1 { 1. } else { -1. } * allowance * shared_fraction;
        }
        let biased = perturb_shape(e, delta);
        let pair = biased.and_then(|b| exact_poses(camera, b, r, 0.));
        let (b, pair) = if let (Some(b), Some(pair)) = (biased, pair) {
            (b, associate(out, pair))
        } else {
            invalid += 1;
            (e, out)
        };
        let (envelope, _, failures) =
            sampled_envelope(camera, b, r, pair, allowance * (1. - shared_fraction));
        invalid += failures;
        for i in 0..2 {
            uncertainty[i].normals[state] = pair[i].normal;
            uncertainty[i].pivots[state] = pair[i].pivot_grid_px;
            uncertainty[i].innovation_px[state] = envelope[i];
        }
    }
    for i in 0..2 {
        uncertainty[i].angular_radius_rad = angular[i];
        uncertainty[i].invalid_samples = invalid;
        uncertainty[i].full_budget_pivot_radius_px = full_pivot[i];
        out[i].uncertainty = Some(uncertainty[i]);
    }
    Some(out)
}
#[derive(Clone)]
struct Frame {
    source: Source,
    roi: u64,
    epoch: String,
    origin: [f64; 2],
    raw: Value,
}
fn raw_hash(dir: &Path, f: &Frame) -> Result<String, String> {
    let name = f.raw["stream"].as_str().ok_or("missing RAW filename")?;
    if Path::new(name)
        .components()
        .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("invalid RAW path".into());
    }
    let mut file = File::open(dir.join(name)).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(num(&f.raw["offset"]).ok_or("RAW offset")?))
        .map_err(|e| e.to_string())?;
    let len = num(&f.raw["length"]).ok_or("RAW length")?;
    let mut h = Sha256::new();
    let read = std::io::copy(&mut file.take(len), &mut h).map_err(|e| e.to_string())?;
    if read != len {
        return Err("short RAW".into());
    }
    Ok(format!("{:x}", h.finalize()))
}
fn fullkey(v: &Value, f: &Frame) -> bool {
    num(&v["roi_id"]) == Some(f.roi)
        && num(&v["sequence"]) == Some(f.source.frame)
        && num(&v["sensor_timestamp_ns"]) == Some(f.source.ns)
        && v["stream_epoch"].as_str() == Some(f.epoch.as_str())
        && v["viewer_session_id"].as_str().is_some()
        && v["viewer_session_id"] == f.raw["source_clock"]["source_key"]["viewer_session_id"]
        && ["region_session", "region_generation"]
            .into_iter()
            .all(|k| {
                num(&v[k]).is_some() && num(&v[k]) == num(&f.raw["source_clock"]["source_key"][k])
            })
}
fn transport(
    v: &Value,
    dir: &Path,
    from: &Frame,
    to: &Frame,
    hashes: &mut BTreeMap<(u64, u64), String>,
) -> Option<Transport> {
    if v["reliable"] != true
        || num(&v["support"])? < 8
        || v["evidence_origin"] != "native-raw-exterior"
        || !fullkey(&v["from_source_key"], from)
        || !fullkey(&v["to_source_key"], to)
    {
        return None;
    }
    for (f, key) in [(from, "from_raw_sha256"), (to, "to_raw_sha256")] {
        let k = (f.roi, f.source.ns);
        if !hashes.contains_key(&k) {
            hashes.insert(k, raw_hash(dir, f).ok()?);
        }
        if v[key].as_str() != Some(hashes[&k].as_str()) {
            return None;
        }
    }
    let scale = v["scale"].as_f64()?;
    let (s, c) = v["angle_rad"].as_f64()?.sin_cos();
    let center = arr::<2>(&v["center_sensor_px"])?;
    let d = arr::<2>(&v["translation_px"])?;
    let a = scale * c;
    let b = scale * s;
    Some(Transport {
        from: from.source,
        to: to.source,
        a,
        b,
        translation: [
            center[0] + d[0] - a * center[0] + b * center[1],
            center[1] + d[1] - b * center[0] - a * center[1],
        ],
        allowance_px: v["sigma_px"].as_f64()?,
    })
}
fn geometry(
    eye: &Value,
    f: &Frame,
) -> Option<(
    Ellipse,
    f64,
    PinholeCamera,
    Option<[f64; 3]>,
    bool,
    &'static str,
)> {
    let cg = &eye["centers_and_gaze"];
    let contact = &cg["virtual_contact_surface_gaze"];
    if !contact.is_null() {
        let j = &cg["joint_conics"];
        let idx = j["sources"].as_array()?.iter().position(|v| {
            num(&v["roi_id"]) == Some(f.roi)
                && num(&v["sequence"]) == Some(f.source.frame)
                && num(&v["sensor_timestamp_ns"]) == Some(f.source.ns)
                && num(&v["clock_domain"]) == Some(1)
                && num(&v["clock_epoch"])
                    == Some(f.epoch.bytes().fold(14695981039346656037, |h, b| {
                        (h ^ b as u64).wrapping_mul(1099511628211)
                    }))
        })?;
        let camera = PinholeCamera {
            focal_px: arr(&j["intrinsics"]["focal_px"])?,
            principal_px: arr(&j["intrinsics"]["principal_px"])?,
        };
        let c = arr(&j["eye_centers"][idx])?;
        let n = arr::<3>(&j["surface_normals"][idx])?;
        let baseline = arr::<3>(&contact["relative_gaze_vector"])?;
        if [n, baseline]
            .iter()
            .any(|n| (n.iter().map(|x| x * x).sum::<f64>() - 1.).abs() > 1e-5)
        {
            return None;
        }
        if angle(n, baseline) > 1e-5 {
            return None;
        }
        let major = (contact["rectified_area_px2"].as_f64()? / std::f64::consts::PI).sqrt();
        let (e, r) = recover(camera, c, n, major)?;
        let near = arr::<2>(&contact["camera_near_point_sensor"])?;
        let q = contact["bucketed_face_radius_px"].as_f64()?;
        if (e.center.0 - (near[0] - q * n[0])).hypot(e.center.1 - (near[1] - q * n[1])) > 0.05 {
            return None;
        }
        return Some((
            e,
            r,
            camera,
            Some(n),
            contact["sign_resolved"] == true,
            "saved-joint-pose-forward-projection",
        ));
    }
    let v = &eye["limbus"]["semantic_ellipse"];
    let center = arr::<2>(&v["center"])?;
    let e = Ellipse {
        center: (center[0] + f.origin[0], center[1] + f.origin[1]),
        major_radius: v["major_radius"].as_f64()?,
        minor_radius: v["minor_radius"].as_f64()?,
        angle: v["angle_rad"].as_f64()?,
    };
    Some((
        e,
        6.,
        PinholeCamera {
            focal_px: [4000.; 2],
            principal_px: [4000., 3000.],
        },
        None,
        false,
        "stored-semantic-conic-engineering-intrinsics",
    ))
}
#[derive(Default)]
struct Stats {
    admitted: usize,
    preferred: usize,
    identified: usize,
    transport: usize,
    rejected: usize,
    changed: usize,
    steps: Vec<f64>,
    previous: BTreeMap<u64, (Source, [f64; 3])>,
    step_ns: Vec<f64>,
    uncertainty_ns: Vec<f64>,
    angular_spread: Vec<f64>,
    cone_overlap: usize,
    invalid_samples: usize,
    peak_states: usize,
    peak_transitions: usize,
}
fn quantile(v: &[f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut x = v.to_vec();
    x.sort_by(f64::total_cmp);
    Some(x[((x.len() - 1) as f64 * q).round() as usize])
}
impl Stats {
    fn json(&self) -> Value {
        json!({"admitted":self.admitted,"fresh_provisional_preferences":self.preferred,"motion_supported_under_model":self.identified,"independently_identified":0,"transport_intervals":self.transport,"rejected":self.rejected,"changed_from_recorded":self.changed,"normal_step_degrees_p95":quantile(&self.steps,0.95),"normal_steps_over_30_degrees":self.steps.iter().filter(|x|**x>30.).count(),"adjacent_step_count":self.steps.len(),"step_nanoseconds_p95":quantile(&self.step_ns,0.95),"uncertainty_adapter_nanoseconds_p95":quantile(&self.uncertainty_ns,0.95),"sampled_angular_radius_rad_p95":quantile(&self.angular_spread,0.95),"angular_cone_overlap_sources":self.cone_overlap,"invalid_uncertainty_samples":self.invalid_samples,"peak_states":self.peak_states,"peak_transitions":self.peak_transitions})
    }
}
fn capture(
    dir: &Path,
    out: &mut impl Write,
    sidecar_name: &str,
    stride: usize,
    phase: usize,
    membership_name: Option<&str>,
) -> Result<Value, String> {
    let mut frames = BTreeMap::new();
    let mut epochs = BTreeMap::new();
    let mut collisions = BTreeSet::new();
    for v in rows(&dir.join("frames.jsonl"))? {
        let Some(ns) = num(&v["timestamp_ns"]) else {
            continue;
        };
        let Some(roi) = num(&v["eye_id"]) else {
            continue;
        };
        let Some(seq) = num(&v["sequence"]) else {
            continue;
        };
        let Some(origin) = arr::<2>(&json!([v["sensor_x"], v["sensor_y"]])) else {
            continue;
        };
        let epoch = v
            .pointer("/source_clock/source_key/stream_epoch")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("unattested:{}", dir.display()));
        let id = epochs.len() as u64 + 1;
        let stream = *epochs.entry(epoch.clone()).or_insert(id);
        let f = Frame {
            source: Source {
                stream,
                frame: seq,
                ns,
            },
            roi,
            epoch,
            origin,
            raw: v,
        };
        if let Some(old) = frames.insert((roi, ns), f.clone()) {
            if old.source != f.source
                || old.origin != f.origin
                || old.raw["region"] != f.raw["region"]
            {
                collisions.insert((roi, ns));
            }
        }
    }
    for k in &collisions {
        frames.remove(k);
    }
    let mut hashes = BTreeMap::new();
    let mut membership = BTreeSet::new();
    if let Some(name) = membership_name {
        for v in rows(&dir.join(name))? {
            let raw = &v["frame"];
            let roi = num(&raw["eye_id"]).ok_or("membership ROI")?;
            let ns = num(&raw["timestamp_ns"]).ok_or("membership time")?;
            let f = frames
                .get(&(roi, ns))
                .ok_or("membership source absent or ambiguous")?;
            if !fullkey(&raw["source_clock"]["source_key"], f) {
                return Err("membership source mismatch".into());
            }
            let hash = raw_hash(dir, f)?;
            if v["raw_sha256"].as_str() != Some(hash.as_str()) {
                return Err("membership native RAW mismatch".into());
            }
            hashes.insert((roi, ns), hash);
            if !membership.insert((roi, ns)) {
                return Err("duplicate membership".into());
            }
        }
    }
    let sidecar = dir.join(sidecar_name);
    let mut motion = BTreeMap::new();
    if sidecar.is_file() {
        for v in rows(&sidecar)? {
            if let (Some(r), Some(t)) = (num(&v["roi"]), num(&v["to_source_ns"])) {
                if motion.insert((r, t), v).is_some() {
                    return Err("duplicate motion endpoint".into());
                }
            }
        }
    }
    let mut trackers = BTreeMap::<u64, [Tracker; 5]>::new();
    let mut stats: [Stats; 5] = std::array::from_fn(|_| Stats::default());
    let mut previous = BTreeMap::<u64, Frame>::new();
    let mut source_counts = BTreeMap::<u64, usize>::new();
    let mut intentional_skips = 0;
    let mut seen = BTreeSet::new();
    let mut held = 0;
    let mut missing = 0;
    let mut unavailable = 0;
    let mut baseline_resolved = 0;
    let mut baseline_steps = Vec::new();
    let mut baseline_previous = BTreeMap::<u64, (Source, [f64; 3])>::new();
    let mut kinds = BTreeMap::<&str, usize>::new();
    let mut scale_steps = Vec::new();
    let mut previous_area = BTreeMap::<u64, f64>::new();
    let mut maximum_center_error = 0_f64;
    let mut maximum_twin_normal_error = 0_f64;
    for row in rows(&dir.join("predictions.jsonl"))? {
        let Some(roi) = num(&row["roi_frame_key"]["roi_id"]) else {
            missing += 1;
            continue;
        };
        let Some(delivery) = num(&row["roi_frame_key"]["sensor_timestamp_ns"]) else {
            missing += 1;
            continue;
        };
        let Some(present) = frames.get(&(roi, delivery)) else {
            missing += 1;
            continue;
        };
        if num(&row["roi_frame_key"]["sequence"]) != Some(present.source.frame) {
            missing += 1;
            continue;
        }
        let eye = &row["predictions"]["eye_candidate"];
        let ns =
            num(&eye["centers_and_gaze"]["virtual_contact_surface_gaze"]["source_timestamp_ns"])
                .unwrap_or(delivery);
        let Some(f) = frames
            .get(&(roi, ns))
            .filter(|f| f.epoch == present.epoch && ns <= delivery)
        else {
            missing += 1;
            continue;
        };
        let Some((e, r, camera, baseline, resolved, kind)) = geometry(eye, f) else {
            unavailable += 1;
            continue;
        };
        if !seen.insert((roi, f.source.stream, ns)) {
            held += 1;
            continue;
        }
        let count = source_counts.entry(roi).or_default();
        let keep = if membership_name.is_some() {
            membership.contains(&(roi, ns))
        } else {
            *count % stride == phase
        };
        *count += 1;
        if !keep {
            intentional_skips += 1;
            continue;
        }
        *kinds.entry(kind).or_default() += 1;
        let Some(twins) = exact_poses(camera, e, r, 2.) else {
            unavailable += 1;
            continue;
        };
        let seed = baseline.map_or(0, |n| {
            usize::from(angle(n, twins[1].normal) < angle(n, twins[0].normal))
        });
        if let Some(n) = baseline {
            maximum_twin_normal_error = maximum_twin_normal_error.max(angle(n, twins[seed].normal));
            let v = &eye["centers_and_gaze"]["virtual_contact_surface_gaze"];
            let near = arr::<2>(&v["camera_near_point_sensor"]).unwrap();
            let q = v["bucketed_face_radius_px"].as_f64().unwrap();
            maximum_center_error = maximum_center_error
                .max((e.center.0 - near[0] + q * n[0]).hypot(e.center.1 - near[1] + q * n[1]));
        }
        if previous.get(&roi).is_some_and(|p| {
            p.source.stream == f.source.stream
                && (f.source.ns <= p.source.ns || f.source.frame <= p.source.frame)
        }) {
            for s in &mut stats {
                s.rejected += 1;
            }
            continue;
        }
        let t = previous.get(&roi).and_then(|p| {
            motion
                .get(&(roi, ns))
                .and_then(|v| transport(v, dir, p, f, &mut hashes))
        });
        let area = std::f64::consts::PI * e.major_radius.powi(2);
        if let (Some(t), Some(a)) = (t, previous_area.get(&roi)) {
            scale_steps.push((area / a / (t.a * t.a + t.b * t.b)).ln().abs());
        }
        baseline_resolved += usize::from(resolved);
        if let Some(n) = baseline {
            if let Some((s, p)) = baseline_previous.insert(roi, (f.source, n)) {
                if s.stream == f.source.stream && ns > s.ns && ns - s.ns <= 750_000_000 {
                    baseline_steps.push(angle(p, n).to_degrees());
                }
            }
        }
        let arms = trackers
            .entry(roi)
            .or_insert_with(|| std::array::from_fn(|_| Tracker::default()));
        let mut details = Vec::new();
        for i in 0..5 {
            let allowance = [2., 4., 8., 2., 4.][i];
            let uncertainty_start = std::time::Instant::now();
            let shared_fraction = if i == 4 { 0. } else { 0.5 };
            let p = poses_split(camera, e, r, allowance, shared_fraction).unwrap();
            let uncertainty_ns = uncertainty_start.elapsed().as_nanos() as f64;
            let angular_radius = p.map(|p| p.uncertainty.unwrap().angular_radius_rad);
            let cone_overlap = angle(p[0].normal, p[1].normal) <= angular_radius.iter().sum();
            let start = std::time::Instant::now();
            let o = arms[i].observe(Observation {
                source: f.source,
                fresh: true,
                poses: p,
                seed,
                transport: if i == 3 { None } else { t },
                head_rotation: None,
            });
            let s = &mut stats[i];
            s.uncertainty_ns.push(uncertainty_ns);
            s.angular_spread.extend(angular_radius);
            s.cone_overlap += usize::from(cone_overlap);
            s.invalid_samples += p[0].uncertainty.unwrap().invalid_samples;
            s.step_ns.push(start.elapsed().as_nanos() as f64);
            s.peak_states = s.peak_states.max(o.states);
            s.peak_transitions = s.peak_transitions.max(o.transitions);
            if let Some(k) = o.preferred {
                s.admitted += 1;
                s.preferred += 1;
                s.identified += usize::from(o.identified.is_some());
                s.transport += usize::from(o.independent.is_some());
                s.changed += usize::from(baseline.is_some() && k != seed);
                if let Some((prev, n)) = s.previous.get(&roi).copied() {
                    if prev.stream == f.source.stream && ns > prev.ns && ns - prev.ns <= 750_000_000
                    {
                        s.steps.push(angle(n, p[k].normal).to_degrees());
                    }
                }
                s.previous.insert(roi, (f.source, p[k].normal));
            } else {
                s.rejected += 1;
            }
            let kinematics = o.kinematics.map(|k| json!({"velocity_rad_s":k.velocity_rad_s,"acceleration_rad_s2":k.acceleration_rad_s2,"jerk_rad_s3":k.jerk_rad_s3,"head_compensated":k.head_compensated,"mode":k.mode}));
            let emitted_kinematics = o.emitted_kinematics.map(|k| json!({"velocity_rad_s":k.velocity_rad_s,"acceleration_rad_s2":k.acceleration_rad_s2,"jerk_rad_s3":k.jerk_rad_s3,"head_compensated":k.head_compensated,"mode":k.mode}));
            details.push(json!({"preferred":o.preferred,"preferred_normal":o.preferred.map(|k|p[k].normal),"motion_supported_under_model":o.identified,"identified":null,"costs":o.costs,"independent_bounded_residuals":o.independent,"profiled_history_max_residual":o.history_bounds,"common_window_mean_robust_residual":o.history_scores,"model_margin":o.margin,"window_start_ns":o.window_start.map(|s|s.ns.to_string()),"window_intervals":o.window_intervals,"fitted_path_kinematics":kinematics,"emitted_kinematics":emitted_kinematics,"pivot_residuals_px":o.residual_px,"pivot_bounds_px":o.bounds_px,"support":o.support,"reason":o.reason,"states":o.states,"transitions":o.transitions}));
            let detail = details.last_mut().unwrap();
            detail["boundary_allowance_px"] = json!(allowance);
            detail["shared_shape_fraction"] = json!(shared_fraction);
            detail["independent_profile_shared_state"] =
                json!(o.independent_nuisance.map(|d| d / 5));
            detail["independent_profile_depth_ratio"] = json!(o
                .independent_nuisance
                .map(|d| perspective_sign::DEPTH_RATIOS[d % 5]));
            detail["joint_profile_shared_state"] = json!(o.joint_nuisance.map(|d| d / 5));
            detail["joint_profile_depth_ratio"] = json!(o
                .joint_nuisance
                .map(|d| perspective_sign::DEPTH_RATIOS[d % 5]));
            detail["joint_profile_current_normals"] = json!(std::array::from_fn::<_, 2, _>(|k| p
                [k]
                .nuisance_normal(o.joint_nuisance[k])));
            detail["selected_latent_normal"] = json!(o
                .preferred
                .map(|k| p[k].nuisance_normal(o.joint_nuisance[k])));
            detail["normal_output_contract"] = json!("nominal exact current twin; fitted derivatives use joint latent state; emitted derivatives use nominal outputs");
            detail["pivot_diagnostic_state"] =
                json!("zero-shared-bias-depth-ratio-1.5; not profiled winner");
            detail["sampled_angular_radius_rad"] = json!(angular_radius);
            detail["angular_cones_overlap"] = json!(cone_overlap);
            detail["invalid_uncertainty_samples"] =
                json!(p[0].uncertainty.unwrap().invalid_samples);
            detail["sampled_full_budget_pivot_radius_px"] =
                json!(p.map(|p| p.uncertainty.unwrap().full_budget_pivot_radius_px));
        }
        writeln!(out,"{}",json!({"capture":dir,"roi":roi,"sequence":f.source.frame,"source_ns":ns.to_string(),"source_key":f.raw["source_clock"]["source_key"],"geometry":kind,"ellipse_sensor":{"center":e.center,"major":e.major_radius,"minor":e.minor_radius,"angle":e.angle},"radius_conditional_mm":r,"normals":twins.map(|p|p.normal),"twin_separation_rad":angle(twins[0].normal,twins[1].normal),"pivots_sensor":twins.map(|p|p.pivot_px),"pose_allowances_px":twins.map(|p|p.allowance_px),"recorded_branch":baseline.map(|_|seed),"recorded_resolved":resolved,"arms":details})).map_err(|e|e.to_string())?;
        previous.insert(roi, f.clone());
        previous_area.insert(roi, area);
    }
    let mut report=(
        json!({"capture":dir,"frame_sha256":digest(&dir.join("frames.jsonl"))?,"prediction_sha256":digest(&dir.join("predictions.jsonl"))?,"motion_sha256":if sidecar.is_file(){Some(digest(&sidecar)?)}else{None},"ambiguous_source_keys":collisions.len(),"held_publications":held,"missing_source":missing,"unavailable_geometry_publications":unavailable,"geometry_kinds":kinds,"baseline_resolved":baseline_resolved,"baseline_steps_p95_degrees":quantile(&baseline_steps,0.95),"baseline_steps_over_30_degrees":baseline_steps.iter().filter(|x|**x>30.).count(),"sn_feida_direct_scale_pairs":scale_steps.len(),"sn_feida_absolute_log_step_p95":quantile(&scale_steps,0.95),"arms":stats.iter().map(Stats::json).collect::<Vec<_>>()}),
    ).0;
    report["source_stride"] = json!(stride);
    report["maximum_reconstructed_sensor_center_error_px"] = json!(maximum_center_error);
    report["maximum_saved_normal_to_twin_error_rad"] = json!(maximum_twin_normal_error);
    report["source_phase"] = json!(phase);
    report["intentional_source_skips"] = json!(intentional_skips);
    report["motion_sidecar"] = json!(sidecar_name);
    report["source_membership"] = json!(membership_name);
    report["membership_count"] = json!(membership_name.map(|_| membership.len()));
    report["membership_sha256"] =
        json!(membership_name.map(|n| digest(&dir.join(n))).transpose()?);
    Ok(report)
}
fn run() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let output = PathBuf::from(
        args.next()
            .ok_or("usage: buttercup_perspective_sign_replay OUTPUT.json CAPTURE_ROOT...")?,
    );
    let mut dirs = BTreeSet::new();
    let mut sidecar = "exterior-motion-circle.jsonl".to_string();
    let mut stride = 1;
    let mut phase = 0;
    let mut membership_name = None;
    while let Some(p) = args.next() {
        if p == "--source-membership" {
            let name = args
                .next()
                .ok_or("missing membership basename")?
                .into_string()
                .map_err(|_| "invalid membership")?;
            if Path::new(&name).components().count() != 1
                || !matches!(
                    Path::new(&name).components().next(),
                    Some(std::path::Component::Normal(_))
                )
            {
                return Err("membership must be a basename".into());
            }
            membership_name = Some(name);
        } else if p == "--motion-sidecar" {
            sidecar = args
                .next()
                .ok_or("missing motion basename")?
                .into_string()
                .map_err(|_| "invalid basename")?;
            if Path::new(&sidecar).components().count() != 1
                || !matches!(
                    Path::new(&sidecar).components().next(),
                    Some(std::path::Component::Normal(_))
                )
            {
                return Err("motion sidecar must be a basename".into());
            }
        } else if p == "--source-stride" {
            stride = args
                .next()
                .ok_or("missing stride")?
                .to_str()
                .ok_or("invalid stride")?
                .parse::<usize>()
                .map_err(|_| "invalid stride")?;
        } else if p == "--source-phase" {
            phase = args
                .next()
                .ok_or("missing phase")?
                .to_str()
                .ok_or("invalid phase")?
                .parse::<usize>()
                .map_err(|_| "invalid phase")?;
        } else {
            discover(Path::new(&p), &mut dirs)?;
        }
    }
    if !(1..=8).contains(&stride) || phase >= stride {
        return Err("require stride1..8 and phase<stride".into());
    }
    if dirs.is_empty() {
        return Err("no captures".into());
    }
    let detail = output.with_extension("jsonl");
    let mut out = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&detail)
            .map_err(|e| e.to_string())?,
    );
    let mut reports = Vec::new();
    let mut duplicates = 0;
    let mut seen = BTreeSet::new();
    for dir in dirs {
        let key = (
            digest(&dir.join("frames.jsonl"))?,
            digest(&dir.join("predictions.jsonl"))?,
        );
        if !seen.insert(key) {
            duplicates += 1;
            continue;
        }
        reports.push(capture(
            &dir,
            &mut out,
            &sidecar,
            stride,
            phase,
            membership_name.as_deref(),
        )?);
    }
    out.flush().map_err(|e| e.to_string())?;
    let report = json!({"schema":"perspective-sign-shadow-v6","tracker_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../eye_scene_model/perspective_sign.rs"))),"replay_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("buttercup_perspective_sign_replay.rs"))),"arms":["boundary-2px-half-shared-shape","boundary-4px-half-shared-shape","boundary-8px-half-shared-shape","no-transport-2px-half-shared-shape","boundary-4px-full-independent"],"exact_archive_duplicates":duplicates,"limitations":["Provisional preference is not calibrated gaze coverage or verified sign identity.","Saved joint pose and radius are upstream conditional geometry, not independently measured scale.","All arms keep the exact input conic; label localization and SN-FEIDA are unchanged by sign choice.","No signed 3-D truth, independent 3-D head rotation, or calibrated intrinsics; finite uncertainty samples are not a continuous enclosure or calibrated probability.","Shared additive shape bias is fixed in sensor coordinates across the window, not a general anatomical model under changing scale or rotation.","Historical semantic conics use stated engineering intrinsics and radius; old contact-only exports without joint pose are excluded.","RAW exterior transport is validated against full source keys and native byte hashes; only direct adjacent admitted intervals are used."],"captures":reports});
    let mut report = report;
    report["schema"] = json!("perspective-sign-shadow-v7");
    report["trajectory_contract"] = json!("shared-state conic normals and pivots; residual plus source-timed motion plus 0.03 radians arc cost on every edge; nominal current output");
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut f, &report).map_err(|e| e.to_string())?;
    println!("{}", output.display());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod perspective_tests {
    use super::*;
    fn camera() -> PinholeCamera {
        PinholeCamera {
            focal_px: [4000., 3900.],
            principal_px: [4000., 3000.],
        }
    }
    #[test]
    fn independent_shape_directions_both_twin_envelope_coverage() {
        let mut checks = 0;
        let mut pivot_misses = 0;
        let mut angular_misses = 0;
        let mut maximum_pivot_excess = 0_f64;
        let mut maximum_angular_excess = 0_f64;
        for axis in 0..2 {
            for sign in [-1., 1.] {
                for theta in [0.001_f64, 0.03, 0.3, 0.7] {
                    let mut n = [0., 0., theta.cos()];
                    n[axis] = sign * theta.sin();
                    // Include centered nearfrontal and off-axis camera geometries.
                    for offset in [0., 35.] {
                        let e = ProjectedCircle::project(
                            camera(),
                            [offset, offset, -310.],
                            n,
                            6.,
                            [0, 0],
                        )
                        .unwrap()
                        .ellipse()
                        .unwrap();
                        for budget in [2., 4., 8.] {
                            let reference = poses(camera(), e, 6., budget).unwrap();
                            for k in 0..48 {
                                // Irrational directions and interior radii differ from
                                // the envelope stencil. Center and S share total budget.
                                let t = (k as f64 * 1.41421356237).sin();
                                let phi = k as f64 * 2.399963229728653;
                                let radius = budget * (0.3 + 0.7 * ((k * 17 % 47) as f64 / 46.));
                                let shape = [t, phi.cos(), phi.sin()]
                                    .map(|v| v * radius * 0.8 / (1. + t.abs()));
                                let mut perturbed = perturb_shape(e, shape).unwrap();
                                perturbed.center.0 += 0.2 * radius * (phi * 1.7).cos();
                                perturbed.center.1 += 0.2 * radius * (phi * 1.7).sin();
                                let pair = associate(
                                    reference,
                                    exact_poses(camera(), perturbed, 6., 0.).unwrap(),
                                );
                                for i in 0..2 {
                                    let u = reference[i].uncertainty.unwrap();
                                    let error = angle(pair[i].normal, reference[i].normal);
                                    angular_misses +=
                                        usize::from(error > u.angular_radius_rad + 1e-8);
                                    maximum_angular_excess =
                                        maximum_angular_excess.max(error - u.angular_radius_rad);
                                    for d in 0..5 {
                                        checks += 1;
                                        let p = reference[i].pivot_grid_px[d];
                                        let q = pair[i].pivot_grid_px[d];
                                        let excess = (p[0] - q[0]).hypot(p[1] - q[1])
                                            - u.full_budget_pivot_radius_px[d];
                                        pivot_misses += usize::from(excess > 1e-7);
                                        maximum_pivot_excess = maximum_pivot_excess.max(excess);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!(
            "independent_envelope {}",
            json!({"pivot_checks":checks,"angular_checks":checks/5,"pivot_undercoverage":pivot_misses,"angular_undercoverage":angular_misses,"maximum_pivot_excess_px":maximum_pivot_excess,"maximum_angular_excess_rad":maximum_angular_excess})
        );
        assert_eq!(
            pivot_misses, 0,
            "finite pivot sample envelope undercovers independent directions"
        );
        assert_eq!(
            angular_misses, 0,
            "finite angular sample envelope undercovers independent directions"
        );
    }
    #[test]
    fn shared_shape_bias_vs_changing_innovation_sequence_contract() {
        let mut differing_scores = 0;
        let mut wrong_supported = 0;
        for actual_shared in [0., 0.5, 1.] {
            for axis in 0..2 {
                for sign in [-1., 1.] {
                    for interval in [8_000_000_u64, 20_000_000, 100_000_000, 300_000_000] {
                        for budget in [2., 4., 8.] {
                            let mut trackers = [Tracker::default(), Tracker::default()];
                            let mut previous = [None, None];
                            let mut correct = [0; 2];
                            let mut supported = [0; 2];
                            let mut covered = [0; 2];
                            let mut recovery = [None; 2];
                            let mut error_max = [0_f64; 2];
                            let mut states = [[0; 7]; 2];
                            for frame in 1..=18 {
                                let theta = sign * (-0.75 + (frame - 1) as f64 * 0.085);
                                let mut n = [0., 0., theta.cos()];
                                n[axis] = theta.sin();
                                let ratio = 1.37 + 0.02 * (frame as f64 * 0.2).sin();
                                let pivot = [35., 40., -320.];
                                let center = std::array::from_fn(|j| pivot[j] + 6. * ratio * n[j]);
                                let clean =
                                    ProjectedCircle::project(camera(), center, n, 6., [0, 0])
                                        .unwrap()
                                        .ellipse()
                                        .unwrap();
                                // Fixed sensor-basis S bias at a non-stencil direction;
                                // bounded changing innovations use a different direction.
                                let phi = frame as f64 * 1.61803398875;
                                let shared = [0.2, 0.48, 0.64]; // op norm = 1
                                let innovation = [0.2 * phi.sin(), phi.cos(), phi.sin()];
                                let scale = 1. + innovation[0].abs();
                                let delta = std::array::from_fn(|j| {
                                    budget
                                        * (actual_shared * shared[j]
                                            + (1. - actual_shared) * innovation[j] / scale)
                                });
                                let measured = perturb_shape(clean, delta).unwrap();
                                let mut scores = [None; 2];
                                for arm in 0..2 {
                                    let p = poses_split(
                                        camera(),
                                        measured,
                                        6.,
                                        budget,
                                        if arm == 0 { 0.5 } else { 0. },
                                    )
                                    .unwrap();
                                    let truth =
                                        usize::from(angle(n, p[1].normal) < angle(n, p[0].normal));
                                    let mut o = Observation {
                                        source: Source {
                                            stream: 1,
                                            frame,
                                            ns: 1_000_000_000 + frame * interval,
                                        },
                                        fresh: true,
                                        poses: p,
                                        seed: 1 - truth,
                                        transport: None,
                                        head_rotation: None,
                                    };
                                    if let Some(prev) = previous[arm] {
                                        o = link(prev, o, [0., 0.], 0.5);
                                    }
                                    let result = trackers[arm].observe(o);
                                    let selected = result.preferred.unwrap();
                                    let error = angle(n, p[selected].normal);
                                    error_max[arm] = error_max[arm].max(error);
                                    correct[arm] += usize::from(selected == truth);
                                    if selected == truth {
                                        recovery[arm].get_or_insert((frame - 1) * interval);
                                    }
                                    if let Some(k) = result.identified {
                                        supported[arm] += 1;
                                        wrong_supported += usize::from(k != truth);
                                        covered[arm] += usize::from(
                                            angle(n, p[k].normal)
                                                <= p[k].uncertainty.unwrap().angular_radius_rad
                                                    + 1e-8,
                                        );
                                    }
                                    states[arm][result.independent_nuisance[selected] / 5] += 1;
                                    scores[arm] = result.history_scores;
                                    previous[arm] = Some(o);
                                }
                                differing_scores += usize::from(scores[0] != scores[1]);
                            }
                            println!(
                                "shared_bias_trial {}",
                                json!({"actual_shared_fraction":actual_shared,"axis":axis,"sign":sign,"interval_ns":interval,"budget_px":budget,"correct":correct,"supported":supported,"supported_angle_covered":covered,"first_recovery_ns":recovery,"maximum_error_rad":error_max,"profile_state_counts":states})
                            );
                        }
                    }
                }
            }
        }
        println!(
            "shared_bias_summary {}",
            json!({"mixed_vs_independent_different_scores":differing_scores,"wrong_supported":wrong_supported})
        );
        assert!(
            differing_scores > 0,
            "shared shape nuisance must affect scores"
        );
        assert_eq!(
            wrong_supported, 0,
            "shared shape controls supplied wrong branch support"
        );
    }
    fn fixture(frame: u64, tilt: f64, shift: f64) -> (Observation, usize) {
        let n = [tilt.sin(), 0., tilt.cos()];
        let r = 6.;
        let h = r * (1.83_f64.powi(2) - 1.).sqrt();
        let pivot = [35. + shift, 40., -320.];
        let c = std::array::from_fn(|j| pivot[j] + h * n[j]);
        let e = ProjectedCircle::project(camera(), c, n, r, [0, 0])
            .unwrap()
            .ellipse()
            .unwrap();
        let mut p = poses(camera(), e, r, 0.2).unwrap();
        // Narrow synthetic support represents independently known fixture depth.
        for x in &mut p {
            x.allowance_px = 0.2;
        }
        let truth = usize::from(angle(n, p[1].normal) < angle(n, p[0].normal));
        (
            Observation {
                source: Source {
                    stream: 1,
                    frame,
                    ns: 1_000_000_000 + frame * 20_000_000,
                },
                fresh: true,
                poses: p,
                seed: truth,
                transport: None,
                head_rotation: None,
            },
            truth,
        )
    }
    fn link(
        previous: Observation,
        mut current: Observation,
        shift: [f64; 2],
        allowance: f64,
    ) -> Observation {
        current.transport = Some(Transport {
            from: previous.source,
            to: current.source,
            a: 1.,
            b: 0.,
            translation: shift,
            allowance_px: allowance,
        });
        current
    }
    #[test]
    fn exact_off_axis_roundtrip_and_distinct_perspective_twins() {
        for x in [-55., 0., 55.] {
            for tilt in [-0.65_f64, -0.2, 0.3, 0.65] {
                let c = [x, 45., -300.];
                let n = [tilt.sin(), 0., tilt.cos()];
                let r = 6.3;
                let original = ProjectedCircle::project(camera(), c, n, r, [0, 0])
                    .unwrap()
                    .ellipse()
                    .unwrap();
                let (e, recovered) = recover(camera(), c, n, original.major_radius).unwrap();
                assert!((r - recovered).abs() < 1e-8);
                assert!((e.minor_radius - original.minor_radius).abs() < 1e-8);
                let pair = circle_pose_hypotheses(camera(), e, [0, 0]).unwrap();
                assert!(pair.iter().any(|p| angle(p.normal, n) < 1e-6));
                for p in pair {
                    let reconstructed = ProjectedCircle::project(
                        camera(),
                        p.center_per_radius.map(|v| v * r),
                        p.normal,
                        r,
                        [0, 0],
                    )
                    .unwrap();
                    for i in 0..36 {
                        let theta = i as f64 * std::f64::consts::TAU / 36.;
                        let (s, c) = e.angle.sin_cos();
                        let u = e.major_radius * theta.cos();
                        let v = e.minor_radius * theta.sin();
                        assert!(
                            reconstructed
                                .residual_px((
                                    e.center.0 + c * u - s * v,
                                    e.center.1 + s * u + c * v
                                ))
                                .abs()
                                < 1e-6
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn shared_ratio_is_invariant_to_upstream_metric_gauge() {
        let (o, _) = fixture(1, 0.55, 0.);
        let n = o.poses[0].normal;
        let c = [35., 40., -320.];
        let e = ProjectedCircle::project(camera(), c, n, 6., [0, 0])
            .unwrap()
            .ellipse()
            .unwrap();
        let reference = poses(camera(), e, 6., 2.).unwrap();
        for scale in [0.2, 0.7, 2., 9.] {
            let scaled = poses(camera(), e, 6. * scale, 2.).unwrap();
            for i in 0..2 {
                for d in 0..5 {
                    assert!(
                        (reference[i].pivot_grid_px[d][0] - scaled[i].pivot_grid_px[d][0]).hypot(
                            reference[i].pivot_grid_px[d][1] - scaled[i].pivot_grid_px[d][1]
                        ) < 1e-8
                    );
                }
            }
        }
    }
    #[test]
    fn off_grid_drifting_pivot_source_cadence_sweep() {
        let mut admitted = 0;
        let mut supported = 0;
        let mut correct = 0;
        let mut total = 0;
        let mut post_seed_failures = 0;
        let mut delayed_recovery_trials = 0;
        for axis in 0..2 {
            for sign in [-1., 1.] {
                for interval in [8_000_000_u64, 20_000_000, 100_000_000, 300_000_000] {
                    for allowance in [2., 4., 8.] {
                        let mut tracker = Tracker::default();
                        let mut previous: Option<Observation> = None;
                        let mut recovery = None;
                        let mut maximum_post_seed_error = 0_f64;
                        for frame in 1..19 {
                            let theta = sign * (-0.75 + (frame - 1) as f64 * 0.085);
                            let mut n = [0., 0., theta.cos()];
                            n[axis] = theta.sin();
                            let r = 6.;
                            let ratio = 1.37 + 0.02 * (frame as f64 * 0.2).sin();
                            let pivot = [35., 40., -320.];
                            let c = std::array::from_fn(|j| pivot[j] + r * ratio * n[j]);
                            let e = ProjectedCircle::project(camera(), c, n, r, [0, 0])
                                .unwrap()
                                .ellipse()
                                .unwrap();
                            let mut p = poses(camera(), e, r, allowance).unwrap();
                            if frame % 2 == 0 {
                                p.swap(0, 1);
                            }
                            let truth = usize::from(angle(n, p[1].normal) < angle(n, p[0].normal));
                            let mut o = Observation {
                                source: Source {
                                    stream: 1,
                                    frame,
                                    ns: 1_000_000_000 + frame * interval,
                                },
                                fresh: true,
                                poses: p,
                                seed: 1 - truth,
                                transport: None,
                                head_rotation: None,
                            };
                            if let Some(prev) = previous {
                                o = link(prev, o, [0., 0.], 0.5);
                            }
                            let result = tracker.observe(o);
                            admitted += usize::from(result.preferred.is_some());
                            total += 1;
                            if result.preferred == Some(truth) {
                                correct += 1;
                                recovery.get_or_insert((frame - 1) * interval);
                            }
                            let error = angle(n, o.poses[result.preferred.unwrap()].normal);
                            if frame > 1 {
                                maximum_post_seed_error = maximum_post_seed_error.max(error);
                                post_seed_failures += usize::from(result.preferred != Some(truth));
                            }
                            if std::env::var_os("PERSPECTIVE_SWEEP_DETAILS").is_some() {
                                println!(
                                    "sweep_window {}",
                                    json!({"axis":axis,"sign":sign,"interval_ns":interval,"allowance_px":allowance,"frame":frame,"source_ns":o.source.ns,"window_start_ns":result.window_start.map(|s|s.ns),"window_intervals":result.window_intervals,"twin_separation_rad":angle(o.poses[0].normal,o.poses[1].normal),"angular_error_rad":error,"preferred":result.preferred,"truth":truth,"model_supported":result.identified,"margin":result.margin,"scores":result.history_scores})
                                );
                            }
                            if let Some(k) = result.identified {
                                supported += 1;
                                assert_eq!(k,truth,"wrong model-supported branch axis{axis} interval{interval} allowance{allowance} frame{frame}");
                            }
                            assert!(
                                result.states <= perspective_sign::MAX_STATES
                                    && result.transitions <= perspective_sign::MAX_TRANSITIONS
                            );
                            previous = Some(o);
                        }
                        delayed_recovery_trials += usize::from(recovery != Some(interval));
                        println!(
                            "sweep_trial {}",
                            json!({"axis":axis,"sign":sign,"interval_ns":interval,"allowance_px":allowance,"first_recovery_ns":recovery,"maximum_post_seed_angular_error_rad":maximum_post_seed_error})
                        );
                    }
                }
            }
        }
        assert_eq!(admitted, total);
        println!(
            "offgrid_complete {}",
            json!({"total":total,"correct":correct,"supported":supported,"post_seed_failures":post_seed_failures,"delayed_recovery_trials":delayed_recovery_trials})
        );
        assert_eq!(
            post_seed_failures, 0,
            "retained v5 noiseless post-seed recovery acceptance"
        );
        assert_eq!(
            delayed_recovery_trials, 0,
            "retained v5 one-interval recovery acceptance"
        );
        assert_eq!(correct, 816);
        assert!(supported > 0);
        println!("offgrid_sweep total={total} preferred_correct={correct} model_supported_correct={supported}; unresolved is not success");
    }
    #[test]
    fn abrupt_real_eye_saccade_retains_current_geometry() {
        let mut tracker = Tracker::default();
        let (mut previous, _) = fixture(1, -0.55, 0.);
        tracker.observe(previous);
        for frame in 2..14 {
            let tilt = if frame < 5 {
                -0.55 + frame as f64 * 0.025
            } else {
                0.35 + (frame - 5) as f64 * 0.015
            };
            let (o, truth) = fixture(frame, tilt, 0.);
            let o = link(previous, o, [0., 0.], 0.2);
            let out = tracker.observe(o);
            assert!(out.preferred.is_some());
            if frame >= 5 {
                assert_eq!(
                    out.preferred,
                    Some(truth),
                    "legitimate50degree step frame{frame} must not freeze: current{:?} history{:?} costs{:?}",out.independent,out.history_bounds,out.costs
                );
            }
            previous = o;
        }
    }
    #[test]
    fn perturbed_conic_transport_and_moving_pivot_controls() {
        perturbed_controls(false);
    }
    #[test]
    fn perturbed_shape_transport_and_moving_pivot_controls() {
        perturbed_controls(true);
    }
    fn perturbed_controls(shape_noise: bool) {
        let mut totals = [[0_usize; 4]; 3];
        for mode in 0..3 {
            for axis in 0..2 {
                for sign in [-1., 1.] {
                    for interval in [8_000_000_u64, 20_000_000, 100_000_000, 300_000_000] {
                        for allowance in [2., 4., 8.] {
                            for noise in [0.3, 1.] {
                                let mut tracker = Tracker::default();
                                let mut previous = None;
                                let mut previous_pixel: Option<[f64; 2]> = None;
                                let mut maximum_error = 0_f64;
                                let mut maximum_post_seed_error = 0_f64;
                                let mut first_recovery = None;
                                let mut post_recovery_errors = 0;
                                let mut maximum_supported_error = 0_f64;
                                let mut maximum_pivot_perturbation = 0_f64;
                                let mut maximum_boundary_bound = 0_f64;
                                let mut correct = 0;
                                let mut supported = 0;
                                let mut wrong_supported = 0;
                                let mut supported_angle_covered = 0;
                                let mut closest_angle_covered = 0;
                                let mut maximum_angle_excess = 0_f64;
                                for frame in 1..=18 {
                                    let theta = if mode == 1 {
                                        sign * 0.4
                                    } else {
                                        sign * (-0.75 + (frame - 1) as f64 * 0.085)
                                    };
                                    let mut n = [0., 0., theta.cos()];
                                    n[axis] = theta.sin();
                                    let ratio = 1.37 + 0.02 * (frame as f64 * 0.2).sin();
                                    let pivot = [
                                        35. + 0.015 * frame as f64,
                                        40. + 0.01 * (frame as f64 * 0.3).sin(),
                                        -320.,
                                    ];
                                    let center =
                                        std::array::from_fn(|j| pivot[j] + 6. * ratio * n[j]);
                                    let mut ellipse =
                                        ProjectedCircle::project(camera(), center, n, 6., [0, 0])
                                            .unwrap()
                                            .ellipse()
                                            .unwrap();
                                    let clean = ellipse;
                                    // Real bounded image measurement perturbations; this is
                                    // not merely a larger declared allowance on exact data.
                                    let center_fraction = if shape_noise { 0.25 } else { 0.7 };
                                    let dx = allowance
                                        * noise
                                        * center_fraction
                                        * (frame as f64 * 1.7).sin();
                                    let dy = allowance
                                        * noise
                                        * center_fraction
                                        * (frame as f64 * 2.3).cos();
                                    ellipse.center.0 += dx;
                                    ellipse.center.1 += dy;
                                    let mut boundary_bound = dx.hypot(dy);
                                    if shape_noise {
                                        let da =
                                            allowance * noise * 0.3 * (frame as f64 * 1.9).sin();
                                        let db =
                                            allowance * noise * 0.3 * (frame as f64 * 2.7).cos();
                                        let gap = clean.major_radius - clean.minor_radius;
                                        // Axis direction is less determined near a circle.
                                        let dangle = 0.3
                                            * (allowance * noise).atan2(gap)
                                            * (frame as f64 * 1.5).sin();
                                        ellipse.major_radius += da;
                                        ellipse.minor_radius += db;
                                        ellipse.angle += dangle;
                                        boundary_bound +=
                                            da.abs().max(db.abs()) + gap * dangle.sin().abs();
                                        if ellipse.minor_radius > ellipse.major_radius {
                                            std::mem::swap(
                                                &mut ellipse.minor_radius,
                                                &mut ellipse.major_radius,
                                            );
                                            ellipse.angle += std::f64::consts::FRAC_PI_2;
                                        }
                                        assert!(ellipse.minor_radius > 0.);
                                    }
                                    // Operator-norm/Hausdorff upper bound for ellipse
                                    // boundary displacement, including the axis gauge.
                                    assert!(boundary_bound <= allowance * noise + 1e-9);
                                    maximum_boundary_bound =
                                        maximum_boundary_bound.max(boundary_bound);
                                    let mut p = poses(camera(), ellipse, 6., allowance).unwrap();
                                    if frame % 2 == 0 {
                                        p.swap(0, 1);
                                    }
                                    let truth =
                                        usize::from(angle(n, p[1].normal) < angle(n, p[0].normal));
                                    let clean_pair = poses(camera(), clean, 6., allowance).unwrap();
                                    let clean_truth = usize::from(
                                        angle(n, clean_pair[1].normal)
                                            < angle(n, clean_pair[0].normal),
                                    );
                                    for depth in 0..5 {
                                        let a = p[truth].pivot_grid_px[depth];
                                        let b = clean_pair[clean_truth].pivot_grid_px[depth];
                                        maximum_pivot_perturbation = maximum_pivot_perturbation
                                            .max((a[0] - b[0]).hypot(a[1] - b[1]));
                                    }
                                    let mut o = Observation {
                                        source: Source {
                                            stream: 1,
                                            frame,
                                            ns: 1_000_000_000 + frame * interval,
                                        },
                                        fresh: true,
                                        poses: p,
                                        seed: 1 - truth,
                                        transport: None,
                                        head_rotation: None,
                                    };
                                    let pixel = camera().project(pivot).unwrap();
                                    if let (Some(prev), Some(pp)) = (previous, previous_pixel) {
                                        if mode != 2 {
                                            // Independent exterior estimate with bounded error,
                                            // plus actual source-matched movable-pivot transport.
                                            o = link(
                                                prev,
                                                o,
                                                [
                                                    pixel[0] - pp[0]
                                                        + 0.3 * (frame as f64 * 1.1).sin(),
                                                    pixel[1] - pp[1]
                                                        + 0.3 * (frame as f64 * 1.3).cos(),
                                                ],
                                                0.5,
                                            );
                                        }
                                    }
                                    let result = tracker.observe(o);
                                    let u = p[truth].uncertainty.unwrap();
                                    let excess = angle(n, p[truth].normal) - u.angular_radius_rad;
                                    closest_angle_covered += usize::from(excess <= 1e-8);
                                    maximum_angle_excess = maximum_angle_excess.max(excess);
                                    let selected = result.preferred.unwrap();
                                    let error = angle(n, p[selected].normal);
                                    maximum_error = maximum_error.max(error);
                                    if frame > 1 {
                                        maximum_post_seed_error =
                                            maximum_post_seed_error.max(error);
                                    }
                                    if selected == truth {
                                        first_recovery.get_or_insert((frame - 1) * interval);
                                    } else if first_recovery.is_some() {
                                        post_recovery_errors += 1;
                                    }
                                    correct += usize::from(selected == truth);
                                    if let Some(k) = result.identified {
                                        supported += 1;
                                        maximum_supported_error =
                                            maximum_supported_error.max(angle(n, p[k].normal));
                                        wrong_supported += usize::from(k != truth);
                                        supported_angle_covered += usize::from(
                                            angle(n, p[k].normal)
                                                <= p[k].uncertainty.unwrap().angular_radius_rad
                                                    + 1e-8,
                                        );
                                    }
                                    previous = Some(o);
                                    previous_pixel = Some(pixel);
                                }
                                totals[mode][0] += 18;
                                totals[mode][1] += correct;
                                totals[mode][2] += supported;
                                totals[mode][3] += wrong_supported;
                                println!(
                                    "perturbed_trial {}",
                                    json!({"shape_noise":shape_noise,"mode":(["moving-eye","head-translation-only","unknown-transport"][mode]),"axis":axis,"sign":sign,"interval_ns":interval,"allowance_px":allowance,"image_noise_fraction":noise,"observations":18,"preferred_closest_true_normal":correct,"model_supported":supported,"wrong_supported":wrong_supported,"maximum_angular_error_rad":maximum_error,"maximum_post_seed_angular_error_rad":maximum_post_seed_error,"first_recovery_ns":first_recovery,"post_recovery_errors":post_recovery_errors,"maximum_supported_angular_error_rad":maximum_supported_error,"maximum_boundary_displacement_bound_px":maximum_boundary_bound,"maximum_projected_pivot_perturbation_px":maximum_pivot_perturbation,"pivot_amplification_over_declared_boundary_allowance":maximum_pivot_perturbation/allowance})
                                );
                                println!(
                                    "perturbed_angular_coverage {}",
                                    json!({"shape_noise":shape_noise,"mode":mode,"axis":axis,"sign":sign,"interval_ns":interval,"allowance_px":allowance,"noise":noise,"closest_angle_covered":closest_angle_covered,"supported_angle_covered":supported_angle_covered,"supported":supported,"maximum_angular_excess_rad":maximum_angle_excess})
                                );
                            }
                        }
                    }
                }
            }
        }
        println!("perturbed_totals {}", json!(totals));
        assert_eq!(
            totals[0][3], 0,
            "wrong model-supported perturbed moving-eye decisions"
        );
        assert_eq!(totals[1][2], 0, "head-only noise manufactured sign support");
        assert_eq!(
            totals[2][2], 0,
            "unknown transport manufactured sign support"
        );
    }
    #[test]
    fn perspective_meridian_continuation_uses_motion_order_without_sign_support() {
        for axis in 0..2 {
            for sign in [-1., 1.] {
                let mut model = Tracker::default();
                let mut ablated = Tracker::default();
                let mut differences = 0;
                for (i, theta) in [-0.061_f64, -0.031, -0.001, 0.029, 0.059]
                    .into_iter()
                    .enumerate()
                {
                    let theta = theta * sign;
                    let mut n = [0., 0., theta.cos()];
                    n[axis] = theta.sin();
                    let c = [9. * n[0], 9. * n[1], -320. + 9. * n[2]];
                    let ellipse = ProjectedCircle::project(camera(), c, n, 6., [0, 0])
                        .unwrap()
                        .ellipse()
                        .unwrap();
                    let p = poses(camera(), ellipse, 6., 4.).unwrap();
                    let truth = usize::from(angle(n, p[1].normal) < angle(n, p[0].normal));
                    let o = Observation {
                        source: Source {
                            stream: 1,
                            frame: i as u64 + 1,
                            ns: 1_000_000_000 + i as u64 * 20_000_000,
                        },
                        fresh: true,
                        poses: p,
                        seed: truth,
                        transport: None,
                        head_rotation: None,
                    };
                    let a = ablated.observe_without_motion_order(o);
                    let b = model.observe(o);
                    assert!(a.identified.is_none() && b.identified.is_none());
                    assert_eq!(b.preferred, Some(truth), "axis{axis} sign{sign} i{i}");
                    differences += usize::from(a.preferred != b.preferred);
                }
                assert!(
                    differences > 0,
                    "motion-order ablation must change actual perspective continuation"
                );
            }
        }
    }
    #[test]
    fn source_key_requires_complete_exact_metadata() {
        let f = Frame {
            source: Source {
                stream: 1,
                frame: 5,
                ns: 20,
            },
            roi: 2,
            epoch: "test".into(),
            origin: [0., 0.],
            raw: json!({"source_clock":{"source_key":{"viewer_session_id":"viewer","region_session":"9","region_generation":"3"}}}),
        };
        let v = json!({"roi_id":2,"sequence":"5","sensor_timestamp_ns":"20","stream_epoch":"test","viewer_session_id":"viewer","region_session":"9","region_generation":"3"});
        assert!(fullkey(&v, &f));
        for key in [
            "roi_id",
            "sequence",
            "sensor_timestamp_ns",
            "stream_epoch",
            "viewer_session_id",
            "region_session",
            "region_generation",
        ] {
            let mut bad = v.clone();
            bad.as_object_mut().unwrap().remove(key);
            assert!(!fullkey(&bad, &f));
        }
        let mut bad = v;
        bad["viewer_session_id"] = json!("other");
        assert!(!fullkey(&bad, &f));
    }
    #[test]
    fn raw_identity_mismatch_cannot_supply_transport() {
        let key = |frame, ns| json!({"roi_id":2,"sequence":frame,"sensor_timestamp_ns":ns,"stream_epoch":"test","viewer_session_id":"viewer","region_session":"9","region_generation":"3"});
        let f = Frame {
            source: Source {
                stream: 1,
                frame: 5,
                ns: 20,
            },
            roi: 2,
            epoch: "test".into(),
            origin: [0., 0.],
            raw: json!({"source_clock":{"source_key":key(5,20)}}),
        };
        let mut g = f.clone();
        g.source.frame = 6;
        g.source.ns = 30;
        g.raw["source_clock"]["source_key"] = key(6, 30);
        let mut hashes = BTreeMap::from([
            ((2, 20), "actual-first".into()),
            ((2, 30), "actual-second".into()),
        ]);
        let v = json!({"reliable":true,"support":9,"evidence_origin":"native-raw-exterior","from_source_key":key(5,20),"to_source_key":key(6,30),"from_raw_sha256":"wrong-first","to_raw_sha256":"actual-second","scale":1.,"angle_rad":0.,"center_sensor_px":[0.,0.],"translation_px":[0.,0.],"sigma_px":1.});
        assert!(transport(&v, Path::new("unused"), &f, &g, &mut hashes).is_none());
    }
    #[test]
    fn unknown_three_dimensional_head_rotation_with_broad_nuisance_is_ambiguous() {
        let mut tracker = Tracker::default();
        let mut previous = None;
        let mut previous_pivot = None;
        for frame in 1..25 {
            let theta = frame as f64 * 0.008;
            let (s, c) = theta.sin_cos();
            let rotate = |v: [f64; 3]| [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
            let n = rotate([0.4_f64.sin(), 0., 0.4_f64.cos()]);
            let pivot = rotate([35., 40., -320.]);
            let center = std::array::from_fn(|j| pivot[j] + 9. * n[j]);
            let e = ProjectedCircle::project(camera(), center, n, 6., [0, 0])
                .unwrap()
                .ellipse()
                .unwrap();
            let p = poses(camera(), e, 6., 8.).unwrap();
            let truth = usize::from(angle(n, p[1].normal) < angle(n, p[0].normal));
            let pixel = camera().project(pivot).unwrap();
            let mut o = Observation {
                source: Source {
                    stream: 1,
                    frame,
                    ns: 1_000_000_000 + frame * 20_000_000,
                },
                fresh: true,
                poses: p,
                seed: truth,
                transport: None,
                head_rotation: None,
            };
            if let (Some(prev), Some(pp)) = (previous, previous_pivot) {
                let pp: [f64; 2] = pp;
                o = link(prev, o, [pixel[0] - pp[0], pixel[1] - pp[1]], 2.);
            }
            let result = tracker.observe(o);
            assert!(result.identified.is_none());
            previous = Some(o);
            previous_pivot = Some(pixel);
        }
    }
    #[test]
    fn permutation_static_readout_does_not_chatter() {
        let mut tracker = Tracker::default();
        let mut chosen = None;
        for frame in 1..80 {
            let (mut o, _) = fixture(frame, 0.45, 0.);
            if frame % 2 == 0 {
                o.poses.swap(0, 1);
                o.seed = 1 - o.seed;
            }
            let out = tracker.observe(o);
            let n = o.poses[out.preferred.unwrap()].normal;
            if let Some(p) = chosen {
                assert!(angle(n, p) < 1e-6);
            }
            chosen = Some(n);
            assert!(out.identified.is_none());
            assert!(out.states <= perspective_sign::MAX_STATES);
            assert!(out.transitions <= perspective_sign::MAX_TRANSITIONS);
        }
    }
    #[test]
    fn wrong_seed_recovers_with_independent_transport_and_crosses_meridian() {
        let mut tracker = Tracker::default();
        let (mut previous, truth) = fixture(1, -0.65, 0.);
        previous.seed = 1 - truth;
        tracker.observe(previous);
        let mut recovered = false;
        let mut identified = false;
        for frame in 2..45 {
            let tilt = -0.65 + (frame - 1) as f64 * 0.03;
            let (o, truth) = fixture(frame, tilt, 0.);
            let o = link(previous, o, [0., 0.], 0.2);
            let out = tracker.observe(o);
            if frame > 6 {
                assert_eq!(
                    out.preferred,
                    Some(truth),
                    "frame {frame} costs {:?}",
                    out.costs
                );
                recovered = true;
            }
            identified |= out.identified == Some(truth);
            previous = o;
        }
        assert!(recovered);
        assert!(identified);
    }
    #[test]
    fn head_translation_and_uncertain_nuisance_do_not_identify() {
        for allowance in [2., 4., 8., 100.] {
            let mut tracker = Tracker::default();
            let (mut prev, _) = fixture(1, 0.45, 0.);
            tracker.observe(prev);
            for frame in 2..15 {
                let (o, _) = fixture(frame, 0.45, (frame - 1) as f64 * 0.1);
                let delta = [4000. * 0.1 / 320., 0.];
                let o = link(prev, o, delta, allowance);
                let result = tracker.observe(o);
                assert!(result.identified.is_none());
                prev = o;
            }
        }
    }
    #[test]
    fn held_late_invalid_and_stream_reset_preserve_freshness() {
        let mut tracker = Tracker::default();
        let (o, _) = fixture(10, 0.4, 0.);
        assert!(tracker.observe(o).preferred.is_some());
        assert!(tracker.observe(o).preferred.is_none());
        let (mut held, _) = fixture(11, 0.4, 0.);
        held.fresh = false;
        assert!(tracker.observe(held).preferred.is_none());
        let (mut invalid, _) = fixture(12, 0.4, 0.);
        invalid.poses[0].normal = [0.; 3];
        assert!(tracker.observe(invalid).preferred.is_none());
        held.fresh = true;
        assert!(tracker.observe(held).preferred.is_none());
        let (mut new, _) = fixture(1, 0.4, 0.);
        new.source.stream = 2;
        assert!(tracker.observe(new).preferred.is_some());
    }
}
