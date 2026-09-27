//! Matched native-RAW diagnostics for the live temporal clustering kernels.
//! No recorded predictions, learned checkpoints, or calibration targets enter the solve.
use buttercup_eye_tracking::{
    raw10, raw_iris_focus as native, raw_motion_octrees as motion, recorded_bundle::BundleSource,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};
#[path = "buttercup_cluster_replay/carrier.rs"]
mod carrier;
#[path = "buttercup_cluster_replay/cohorts.rs"]
mod cohorts;
#[path = "buttercup_cluster_replay/demo.rs"]
mod demo;
#[path = "buttercup_cluster_replay/partition.rs"]
mod partition;
#[path = "../raw_preview.rs"]
#[allow(dead_code)]
mod raw_preview;
#[path = "buttercup_cluster_replay/targets.rs"]
mod targets;
#[path = "buttercup_cluster_replay/vessels.rs"]
mod vessels;
#[path = "buttercup_cluster_replay/vessel_ridges.rs"]
mod vessel_ridges;
#[path = "buttercup_cluster_replay/sclera_splats.rs"]
mod sclera_splats;
#[path = "buttercup_cluster_replay/colmap_probe.rs"]
mod colmap_probe;
#[path = "buttercup_cluster_replay/colmap_groups.rs"]
mod colmap_groups;
#[path = "buttercup_cluster_replay/raw_cadence.rs"]
mod raw_cadence;
#[path = "buttercup_cluster_replay/warp_probe.rs"]
mod warp_probe;
type Error = Box<dyn std::error::Error>;
type Result<T, E = Error> = std::result::Result<T, E>;

fn ellipse(e: motion::IrisEllipseSeed) -> Value {
    json!({"center":[e.center.0,e.center.1],"major":e.major_radius,"minor":e.minor_radius,"angle":e.angle})
}

fn ppm(path: &Path, pixels: &[u32], w: usize, h: usize) -> Result<(), Error> {
    let mut f = fs::File::create(path)?;
    write!(f, "P6\n{w} {h}\n255\n")?;
    let rgb = pixels
        .iter()
        .flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8])
        .collect::<Vec<_>>();
    f.write_all(&rgb)?;
    Ok(())
}

fn point(pixels: &mut [u32], w: usize, h: usize, p: (f64, f64), color: u32, radius: i32) {
    if !p.0.is_finite() || !p.1.is_finite() {
        return;
    }
    let (x, y) = (p.0.round() as i32, p.1.round() as i32);
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (px, py) = (x.saturating_add(dx), y.saturating_add(dy));
            if px >= 0 && py >= 0 && px < (w as i32) && py < (h as i32) {
                pixels[py as usize * w + px as usize] = color;
            }
        }
    }
}

fn curve(pixels: &mut [u32], w: usize, h: usize, e: motion::IrisEllipseSeed, color: u32) {
    let (s, c) = e.angle.sin_cos();
    for i in 0..720 {
        let t = i as f64 * std::f64::consts::TAU / 720.;
        point(
            pixels,
            w,
            h,
            (
                e.center.0 + c * e.major_radius * t.cos() - s * e.minor_radius * t.sin(),
                e.center.1 + s * e.major_radius * t.cos() + c * e.minor_radius * t.sin(),
            ),
            color,
            0,
        );
    }
}

fn quantiles(mut values: Vec<f64>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let q = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
    json!({"mean":values.iter().sum::<f64>()/values.len() as f64,"p50":q(0.5),"p95":q(0.95),"max":q(1.)})
}

fn read_rows(path: &Path) -> Result<Vec<Value>, Error> {
    fs::read_to_string(path)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

// Match visible human points to the nearest point on the conic. Work in one
// ellipse quadrant and refine the best coarse bracket, including its endpoints.
fn label_distance(p: (f64, f64), e: &Value) -> f64 {
    let (s, c) = e["angle"].as_f64().unwrap().sin_cos();
    let dx = p.0 - e["center"][0].as_f64().unwrap();
    let dy = p.1 - e["center"][1].as_f64().unwrap();
    let (x, y) = ((c * dx + s * dy).abs(), (-s * dx + c * dy).abs());
    let (a, b) = (e["major"].as_f64().unwrap(), e["minor"].as_f64().unwrap());
    let squared = |t: f64| (a * t.cos() - x).powi(2) + (b * t.sin() - y).powi(2);
    let step = std::f64::consts::FRAC_PI_2 / 64.0;
    let best = (0..=64usize)
        .min_by(|&i, &j| squared(i as f64 * step).total_cmp(&squared(j as f64 * step)))
        .unwrap();
    let mut lo = best.saturating_sub(1) as f64 * step;
    let mut hi = (best + 1).min(64) as f64 * step;
    for _ in 0..40 {
        let u = lo + (hi - lo) * 0.38196601125;
        let v = lo + (hi - lo) * 0.61803398875;
        if squared(u) < squared(v) {
            hi = v;
        } else {
            lo = u;
        }
    }
    squared((lo + hi) * 0.5)
        .min(squared(0.0))
        .min(squared(std::f64::consts::FRAC_PI_2))
        .sqrt()
}

/// Read labels only after both native replays have fixed their candidates.
/// RAW byte hashes, dimensions and sensor origins must all match before scoring.
fn compare(args: &[String]) -> Result<(), Error> {
    if !(5..=6).contains(&args.len()) {
        return Err("usage: buttercup-cluster-replay --compare BASELINE_DIR CANDIDATE_DIR NEW_REPORT [LABELS_DIR]".into());
    }
    let baseline = read_rows(&Path::new(&args[2]).join("frames.jsonl"))?;
    let candidate = read_rows(&Path::new(&args[3]).join("frames.jsonl"))?;
    let out = Path::new(&args[4]);
    if out.exists()
        || !fs::canonicalize(out.parent().ok_or("report needs parent")?)?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("report must be new and below checked outputs link".into());
    }
    if baseline.len() != candidate.len()
        || baseline
            .iter()
            .zip(&candidate)
            .any(|(a, b)| a["source"] != b["source"] || a["raw_sha256"] != b["raw_sha256"])
    {
        return Err("replays are not exactly source-matched".into());
    }
    let mut labels = Vec::new();
    if let Some(directory) = args.get(5) {
        for entry in fs::read_dir(directory)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if !name.ends_with(".labels.json") || name.contains("backup") {
                continue;
            }
            let label: Value = serde_json::from_slice(&fs::read(&path)?)?;
            if label["schema"] != "buttercup-raw-limbus-band-label-v1" {
                continue;
            }
            let raw_path = label["source_raw"]
                .as_str()
                .ok_or("label source_raw missing")?;
            let digest = format!("{:x}", Sha256::digest(fs::read(raw_path)?));
            let Some(index) = baseline.iter().position(|r| r["raw_sha256"] == digest) else {
                continue;
            };
            let meta = &baseline[index]["source"];
            if label["frame_width"] != meta["width"]
                || label["frame_height"] != meta["height"]
                || label["sensor_origin"][0] != meta["sensor_x"]
                || label["sensor_origin"][1] != meta["sensor_y"]
            {
                return Err(
                    format!("label coordinate identity mismatch: {}", path.display()).into(),
                );
            }
            let points = label["annotation_points"]
                .as_array()
                .ok_or("label points missing")?
                .iter()
                .filter(|p| p["kind"] == "iris_edge" && p["visibility"] == "visible")
                .filter_map(|p| Some((p["x"].as_f64()?, p["y"].as_f64()?)))
                .collect::<Vec<_>>();
            let score = |r: &Value| {
                let fit = &r["fit"];
                if fit.is_null() {
                    Value::Null
                } else {
                    quantiles(
                        points
                            .iter()
                            .map(|&p| label_distance(p, &fit["ellipse"]))
                            .collect(),
                    )
                }
            };
            labels.push(json!({"path":path,"reviewed":label["reviewed"],"source":meta,"raw_sha256":digest,
                "visible_points":points.len(),"baseline_error_px":score(&baseline[index]),"candidate_error_px":score(&candidate[index]),
                "baseline_coverage":!baseline[index]["fit"].is_null(),"candidate_coverage":!candidate[index]["fit"].is_null(),
                "provenance":label["provenance"]}));
        }
    }
    let stats = |rows: &[Value]| {
        json!({
        "candidate_fits":rows.iter().filter(|r|!r["fit"].is_null()).count(),
        "temporal_plus_fit_ms":quantiles(rows.iter().map(|r|r["temporal_ms"].as_f64().unwrap()+r["fit_ms"].as_f64().unwrap()).collect()),
        "fit_ms":quantiles(rows.iter().map(|r|r["fit_ms"].as_f64().unwrap()).collect())})
    };
    let changed = baseline.iter().zip(&candidate).filter(|(a,b)|a["fit"]!=b["fit"]).map(|(a,b)|
        json!({"source":a["source"],"baseline":a["fit"],"candidate":b["fit"],"baseline_rejection":a["rejection"],"candidate_rejection":b["rejection"]})).collect::<Vec<_>>();
    let report = json!({"schema":"buttercup-cluster-comparison-v1","frames":baseline.len(),"baseline_dir":args[2],"candidate_dir":args[3],
        "all_source_identities_match":true,"fit_changes":changed,"all_motion_layers_identical":baseline.iter().zip(&candidate).all(|(a,b)|a["layers"]==b["layers"]),
        "baseline":stats(&baseline),"candidate":stats(&candidate),"labels":labels,
        "sn_feida":null,"scale":"Independent scale not supplied; no area-stability or 3D-gaze accuracy claim.",
        "scope":"Candidate fits only; excludes live queues, asynchronous acquisition, final anatomical gates and publication. Unreviewed labels remain provisional; missing candidates count as dropouts, never zero localization error."});
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{} source-matched frames; report {}",
        baseline.len(),
        out.display()
    );
    Ok(())
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args().collect::<Vec<_>>();
    if args.get(1).is_some_and(|s| s == "--z-motion3d" || s == "--z-motion-groups" || s == "--z-motion-history") {
        return warp_probe::z_motion3d_run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--z-discovery") {
        return warp_probe::z_discovery_run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--iris-layers") {
        return warp_probe::layers_run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--iris-pivot") {
        return warp_probe::iris_run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--warp-probe") {
        return warp_probe::run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--raw-cadence") {
        return raw_cadence::run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--colmap-probe") {
        return colmap_probe::run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--colmap-raw-probe") {
        return colmap_probe::raw_run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--colmap-model-review") {
        return colmap_probe::model_review(&args);
    }
    if args.get(1).is_some_and(|s| s == "--colmap-motion-groups") {
        return colmap_groups::run(&args);
    }
    if args.get(1).is_some_and(|s|s=="--sclera-splats") {
        return sclera_splats::run(&args);
    }
    if args.get(1).is_some_and(|s|s=="--vessel-assay") {
        return vessels::run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--compare") {
        return compare(&args);
    }
    if args.get(1).is_some_and(|s| s == "--cohorts") {
        return cohorts::run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--partition") {
        return partition::run(&args);
    }
    if args.get(1).is_some_and(|s| s == "--demo" || s == "--motion-demo") {
        return demo::run(&args);
    }
    let tensors = args.iter().any(|s| s == "--tensors");
    let motion_only = args.iter().any(|s| s == "--motion-only");
    if motion_only && !tensors { return Err("--motion-only requires --tensors".into()); }
    args.retain(|s| s != "--motion-only");
    let anchor_carrier = args.iter().any(|s| s == "--anchor-carrier");
    let carrier = args.iter().any(|s| s == "--carrier");
    if anchor_carrier && !carrier {
        return Err("--anchor-carrier requires --carrier".into());
    }
    if carrier && !tensors {
        return Err("--carrier requires --tensors".into());
    }
    args.retain(|s| s != "--carrier");
    args.retain(|s| s != "--anchor-carrier");
    let component_radius = args
        .iter()
        .find_map(|s| s.strip_prefix("--tensor-radius="))
        .map(str::parse::<f32>)
        .transpose()?;
    if component_radius.is_some_and(|v| !v.is_finite() || !(0.2..=1.0).contains(&v))
        || (component_radius.is_some() && !tensors)
    {
        return Err("--tensor-radius=0.2..1.0 requires --tensors".into());
    }
    args.retain(|s| !s.starts_with("--tensor-radius="));
    let component_limit = args
        .iter()
        .find_map(|s| s.strip_prefix("--tensor-components="))
        .map(str::parse::<usize>)
        .transpose()?;
    if component_limit.is_some_and(|v| !(4..=8).contains(&v))
        || (component_limit.is_some() && !tensors)
    {
        return Err("--tensor-components=4..8 requires --tensors".into());
    }
    args.retain(|s| s != "--tensors");
    args.retain(|s| !s.starts_with("--tensor-components="));
    if args.len() < 3 {
        return Err("usage: buttercup-cluster-replay BUNDLE NEW_OUTPUT [FRAMES_PER_EYE=80] [SKIP_PER_EYE=0] [--tensors]".into());
    }
    let bundle = BundleSource::open(Path::new(&args[1]))?;
    let out = Path::new(&args[2]);
    if out.exists() {
        return Err("output must be new".into());
    }
    if !fs::canonicalize(out.parent().ok_or("output needs parent")?)?
        .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("output must be below checked outputs link".into());
    }
    let limit = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(80usize);
    let skip = args
        .get(4)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0usize);
    let records = String::from_utf8(bundle.read_entry("frames.jsonl")?)?;
    let target_timeline = if tensors {
        Some(targets::Timeline::read(&bundle)?)
    } else {
        None
    };
    fs::create_dir(out)?;
    if let Some(timeline) = &target_timeline {
        fs::write(
            out.join("target-timeline.json"),
            serde_json::to_vec_pretty(&timeline.report())?,
        )?;
    }
    let mut log = fs::File::create(out.join("frames.jsonl"))?;
    let mut trackers: [motion::FourMotionOctrees; 2] = std::array::from_fn(|_| Default::default());
    for tracker in &mut trackers {
        tracker.retain_tensor_clusters_for_replay(tensors);
        tracker.set_motion_only_for_replay(motion_only);
        if let Some(radius) = component_radius {
            tracker.set_tensor_component_radius_for_replay(radius);
        }
        if let Some(limit) = component_limit {
            tracker.set_tensor_component_limit_for_replay(limit);
        }
    }
    let mut outer: [native::OuterIrisTracker; 2] = std::array::from_fn(|_| Default::default());
    let mut carriers: [carrier::Carrier; 2] = std::array::from_fn(|_| Default::default());
    let mut previous: [Option<(u64, Value)>; 2] = [None, None];
    let mut counts = [0usize; 2];
    let mut tracking_epochs = [0usize; 2];
    let mut totals = [0usize; 2];
    let mut rows = Vec::new();
    for line in records.lines() {
        let meta: Value = serde_json::from_str(line)?;
        let u = |k: &str| meta[k].as_u64().ok_or_else(|| format!("missing {k}"));
        let id = u("eye_id")? as usize;
        if !(1..=2).contains(&id) {
            continue;
        }
        let eye = id - 1;
        totals[eye] += 1;
        if totals[eye] <= skip || counts[eye] >= limit {
            continue;
        }
        let (w, h, sx, sy, stamp) = (
            u("width")? as usize,
            u("height")? as usize,
            u("sensor_x")? as u32,
            u("sensor_y")? as u32,
            u("timestamp_ns")?,
        );
        let lineage = meta["source_clock"]["source_key"]["stream_epoch"].clone();
        let discontinuity = previous[eye]
            .as_ref()
            .is_some_and(|(t, l)| stamp <= *t || stamp - *t > 1_500_000_000 || *l != lineage);
        let interval = previous[eye]
            .as_ref()
            .filter(|_| !discontinuity)
            .map(|(t, _)| (stamp - *t) as f64 / 1e6);
        if discontinuity {
            trackers[eye].clear();
            outer[eye] = Default::default();
            tracking_epochs[eye] += 1;
        }
        previous[eye] = Some((stamp, lineage));
        let packed = bundle.read_range(
            meta["stream"].as_str().ok_or("missing stream")?,
            u("offset")?,
            u("length")? as usize,
        )?;
        let raw = raw10::try_unpack_raw10(&packed, w, h, u("stride")? as usize)?;
        let carrier_started = Instant::now();
        let carrier_report =
            carrier.then(|| carriers[eye].observe(&raw, w, h, sx, sy, stamp, discontinuity));
        let carrier_ms = carrier_started.elapsed().as_secs_f64() * 1000.;
        if anchor_carrier {
            if let Some((from, to, evidence)) = carriers[eye].interval {
                trackers[eye].set_carrier_for_replay(from, to, evidence);
            }
        }
        let start = Instant::now();
        let (seed, upper, lower) = if motion_only {
            (None, Vec::new(), Vec::new())
        } else {
        let focus = native::score_stream_eye(&raw, w, h);
        let upper = native::detect_upper_eyelid_points(&raw, w, h, sx, sy, &focus);
        let lower = native::detect_lower_eyelid_points(&raw, w, h, sx, sy, &focus);
        let boundary = native::detect_outer_iris_boundary_between_eyelids_tracked(
            &raw,
            w,
            h,
            sx,
            sy,
            &focus,
            &upper,
            &lower,
            &mut outer[eye],
        );
        let seed = if !boundary.points.is_empty() {
            Some(motion::IrisEllipseSeed {
                center: boundary.center,
                major_radius: boundary.major_radius,
                minor_radius: boundary.minor_radius,
                angle: boundary.angle,
            })
        } else if focus.eye_basin_valid {
            Some(motion::IrisEllipseSeed::circle(focus.center, focus.radius))
        } else {
            None
        };
        (seed, upper, lower)
        };
        let native_ms = if motion_only { 0. } else { start.elapsed().as_secs_f64() * 1000. };
        let lids = |p: &[native::BorderPoint]| {
            p.iter()
                .map(|p| motion::LidMarginPoint {
                    x: p.x as f32,
                    y: p.y as f32,
                })
                .collect::<Vec<_>>()
        };
        let start = Instant::now();
        let overlay = trackers[eye].observe_with_iris_seed_at_with_canny_profile_and_lids(
            &raw,
            w,
            h,
            sx,
            sy,
            stamp,
            None,
            motion::LearningCannyProfile::CannyBalanced,
            seed,
            &lids(&upper),
            &lids(&lower),
        );
        let temporal_ms = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let (fit, diag) = if motion_only {
            (None, motion::FeatureClusterIrisDiagnostics { rejection:"motion-only: iris detection and naming disabled", ..Default::default() })
        } else {
            motion::feature_cluster_iris_hypothesis_with_diagnostics(&overlay, w, h, seed, 0.)
        };
        let fit_ms = if motion_only { 0. } else { start.elapsed().as_secs_f64() * 1000. };
        let d = &overlay.match_diagnostics;
        if motion_only && (seed.is_some() || overlay.semantic_iris.is_some() || overlay.nested_eye_boundaries.is_some()
            || overlay.radial_limbus_region.is_some() || !overlay.radial_limbus_probes.is_empty()
            || d.relation_iris_candidates.selector_calls != 0 || d.iris_layer_identified || fit.is_some()) {
            return Err("motion-only ablation leaked iris geometry or anatomical selection".into());
        }
        let mut row = json!({"source":meta,"raw_sha256":format!("{:x}",Sha256::digest(&packed)),"reset":discontinuity,
            "source_interval_ms":interval,"seed":seed.map(ellipse),"temporal_ms":temporal_ms,"fit_ms":fit_ms,"native_ms":native_ms,
            "fit":fit.as_ref().map(|f|json!({"ellipse":ellipse(motion::IrisEllipseSeed{center:f.center,major_radius:f.major_radius,
                minor_radius:f.minor_radius,angle:f.angle}),"layer":f.motion_layer,"score":f.score,"edges":f.edge_support,
                "coverage":f.angular_coverage,"opposition":f.opposing_meridians,"evaluations":f.iterations,"bridged":f.bridged_current_frame_edges})),
            "semantic_split":diag.semantic_split,"rejection":diag.rejection,"eligible_layers":diag.eligible_layers,
            "associated_edges":diag.associated_edges,"tracks":overlay.matched_features,
            "layers":overlay.layers.iter().map(|l|json!({"support":l.persistent_tracks,"coherence":l.coherence,"separation":l.separation,"centroid":l.centroid})).collect::<Vec<_>>(),
            "relation_iris_support":d.relation_iris_support,"relation_iris_identity_confirmed":d.relation_iris_identity_confirmed,
            "iris_layer_identified":d.iris_layer_identified,
            "motion_only":motion_only,
            "relation_state":{"components":d.relation_components,"persistent_components":d.relation_persistent_components,
                "max_shared_frames":d.relation_max_shared_frames,"max_coherence":d.relation_max_coherence,
                "nodes":d.relation_nodes,"edges":d.relation_edges,"coherent_edges":d.relation_coherent_edges,
                "selector_calls":d.relation_iris_candidates.selector_calls},
            "stages_ms":{"neutral":d.neutral_micros as f64/1000.,"canny":d.canny_micros as f64/1000.,"edges":d.edge_micros as f64/1000.,
                "nested_boundary":d.nested_boundary_micros as f64/1000.,"matching":d.matching_micros as f64/1000.,
                "relation":d.relation_micros as f64/1000.,"layering":d.layering_micros as f64/1000.,"maintenance":d.maintenance_micros as f64/1000.,
                "light_field":d.light_field_micros as f64/1000.,"radial_limbus":d.radial_limbus_micros as f64/1000.},
            "patch_evaluations":[d.coarse_patch_evaluations,d.native_patch_evaluations],
            "pyramid_refinement_cache_hits":d.pyramid_refinement_cache_hits,"edge_count":overlay.edges.len(),
            "semantic_iris":overlay.semantic_iris.map(ellipse),"nested_limbus":overlay.nested_eye_boundaries.as_ref().map(|p|ellipse(p.limbus)),
            "sn_feida":null,"independent_scale_support":"not supplied; candidate radius is never used for normalization"});
        if let Some(timeline) = &target_timeline {
            row["tracking_epoch"] = json!(tracking_epochs[eye]);
            row["target"] = timeline.at_arrival(u("host_arrival_unix_ns")?);
            let mut clusters = serde_json::to_value(trackers[eye].tensor_clusters_for_replay())?;
            for (cluster, native_cluster) in clusters
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .zip(trackers[eye].tensor_clusters_for_replay())
            {
                for (member, point) in cluster["members"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(&native_cluster.members)
                {
                    member["photometry"] =
                        targets::photometry(&raw, w, h, sx, sy, point.current_sensor);
                }
            }
            row["tensor_clusters"] = clusters;
            let mut points = serde_json::to_value(trackers[eye].tensor_points_for_replay())?;
            for (member, point) in points
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .zip(trackers[eye].tensor_points_for_replay())
            {
                member["photometry"] =
                    targets::photometry(&raw, w, h, sx, sy, point.current_sensor);
            }
            row["tensor_points"] = points;
            if let Some(carrier_report) = carrier_report {
                row["outer_band_carrier"] = carrier_report;
                row["carrier_ms"] = json!(carrier_ms);
                row["anchored_carrier_used"] = json!(trackers[eye].carrier_used_for_replay());
            }
        }
        serde_json::to_writer(&mut log, &row)?;
        writeln!(log)?;
        log.flush()?;
        if counts[eye] % 8 <= 2 || fit.is_some() {
            let mut rgb = raw_preview::color_preview(&raw, w, h, sx, sy, 100, None);
            let name = format!("eye-{id}-{:04}", counts[eye]);
            ppm(&out.join(format!("{name}-raw.ppm")), &rgb, w, h)?;
            for e in &overlay.edges {
                point(&mut rgb, w, h, (e.x as f64, e.y as f64), 0x547780, 0);
            }
            if let Some(s) = seed {
                curve(&mut rgb, w, h, s, 0x20bfff);
            }
            if let Some(p) = overlay.nested_eye_boundaries.as_ref() {
                curve(&mut rgb, w, h, p.limbus, 0xeebd40);
            }
            if let Some(f) = fit.as_ref() {
                curve(
                    &mut rgb,
                    w,
                    h,
                    motion::IrisEllipseSeed {
                        center: f.center,
                        major_radius: f.major_radius,
                        minor_radius: f.minor_radius,
                        angle: f.angle,
                    },
                    0xff4050,
                );
            }
            ppm(&out.join(format!("{name}-overlay.ppm")), &rgb, w, h)?;
            if tensors {
                let mut pixels = raw_preview::color_preview(&raw, w, h, sx, sy, 100, None);
                for (index, cluster) in trackers[eye]
                    .tensor_clusters_for_replay()
                    .iter()
                    .enumerate()
                {
                    let color = [
                        0xff6060, 0x40ff60, 0x4080ff, 0xffd040, 0xf060ff, 0x40ffff, 0xffffff,
                        0xff8040,
                    ][index % 8];
                    for m in &cluster.members {
                        let a = [
                            m.previous_sensor[0] - sx as f32,
                            m.previous_sensor[1] - sy as f32,
                        ];
                        let b = [
                            m.current_sensor[0] - sx as f32,
                            m.current_sensor[1] - sy as f32,
                        ];
                        for i in 0..=16 {
                            let t = i as f32 / 16.;
                            point(
                                &mut pixels,
                                w,
                                h,
                                (
                                    (a[0] * (1. - t) + b[0] * t) as f64,
                                    (a[1] * (1. - t) + b[1] * t) as f64,
                                ),
                                color,
                                0,
                            );
                        }
                        point(&mut pixels, w, h, (b[0] as f64, b[1] as f64), color, 1);
                    }
                    if let Some(origin) = cluster.shared_origin_sensor {
                        let p=(origin[0] as f64-sx as f64,origin[1] as f64-sy as f64);
                        for offset in -4..=4 {
                            point(&mut pixels,w,h,(p.0+offset as f64,p.1),color,0);
                            point(&mut pixels,w,h,(p.0,p.1+offset as f64),color,0);
                        }
                    }
                }
                ppm(&out.join(format!("{name}-tensors.ppm")), &pixels, w, h)?;
                if let Some(points) = row["outer_band_carrier"]["points"].as_array() {
                    let mut pixels = raw_preview::color_preview(&raw, w, h, sx, sy, 100, None);
                    for m in points {
                        let color = if m["inlier"] == true {
                            0x40ff80
                        } else {
                            0x999999
                        };
                        let p = &m["current_sensor"];
                        point(
                            &mut pixels,
                            w,
                            h,
                            (
                                p[0].as_f64().unwrap() - sx as f64,
                                p[1].as_f64().unwrap() - sy as f64,
                            ),
                            color,
                            2,
                        );
                    }
                    ppm(&out.join(format!("{name}-carrier.ppm")), &pixels, w, h)?;
                }
            }
        }
        rows.push(row);
        counts[eye] += 1;
        if counts.iter().all(|n| *n >= limit) {
            break;
        }
    }
    let mut summary = json!({"schema":"buttercup-cluster-replay-v1","bundle":args[1],"counts":counts,"skip_per_eye":skip,
        "tensor_component_limit":component_limit.unwrap_or(4),
        "tensor_component_radius":component_radius.unwrap_or(0.36),
        "outer_band_carrier":carrier,
        "anchor_carrier":anchor_carrier,
        "motion_only":motion_only,
        "fits":rows.iter().filter(|r|!r["fit"].is_null()).count(),
        "scope":"Temporal clustering and current-frame candidate fit with native RAW seeds. Excludes live queues, asynchronous cold multibank worker, shared radius posterior and final anatomy/publication gates. Candidate fits are not accepted live gaze.",
        "labels":"not supplied; no localization accuracy claim","scale":"unavailable; SN-FEIDA withheld",
        "overlay_legend":{"cyan":"native RAW seed","gold":"nested outer boundary proposal","red":"cluster ellipse candidate","gray_blue":"measured Canny points"}});
    if motion_only {
        summary["scope"]=json!("Existing shared-origin graph on generic native RAW tracks. No native iris detector, ellipse seed, inferred iris geometry, iris naming, or ellipse fitting. Anonymous components are not anatomical labels; colors are per-frame groups and crosses are inferred image-space fixed points.");
    }
    summary["group_frames"]=json!({"any":rows.iter().filter(|r|r["tensor_clusters"].as_array().is_some_and(|c|!c.is_empty())).count(),
        "persistent":rows.iter().filter(|r|r["relation_state"]["persistent_components"].as_u64().unwrap_or(0)>0).count(),
        "two_persistent":rows.iter().filter(|r|r["relation_state"]["persistent_components"].as_u64().unwrap_or(0)>=2).count(),
        "finite_origin":rows.iter().filter(|r|r["tensor_clusters"].as_array().is_some_and(|cs|cs.iter().any(|c|!c["shared_origin_sensor"].is_null()))).count()});
    for field in ["native_ms", "temporal_ms", "fit_ms"] {
        summary[field] = quantiles(rows.iter().filter_map(|r| r[field].as_f64()).collect());
    }
    summary["temporal_plus_fit_ms"] = quantiles(
        rows.iter()
            .map(|r| r["temporal_ms"].as_f64().unwrap() + r["fit_ms"].as_f64().unwrap())
            .collect(),
    );
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
