//! Matched single-eye sign-only replay: freeze recorded SAM geometry, recompute
//! candidate-independent RAW transport, compare legacy vs motion-window policy.
//! No eye-pair evidence or measured display target enters either tracker.

use super::*;
use std::io::{BufRead, BufReader};

pub(crate) fn run<I: Iterator<Item = String>>(mut args: I) -> Result<(), String> {
    let output = PathBuf::from(
        args.next()
            .ok_or("expected OUTPUT CAPTURE_DIR LABEL [START] [COUNT]")?,
    );
    let capture = PathBuf::from(args.next().ok_or("missing capture directory")?);
    let label = args.next().ok_or("missing eye label")?;
    let eye_id = match label.as_str() {
        "subject-right" => 1,
        "subject-left" => 2,
        _ => return Err("label must be subject-right or subject-left".into()),
    };
    let start = parse_usize(args.next(), 0, "start")?;
    let count = parse_usize(args.next(), usize::MAX, "count")?;
    if args.next().is_some() || output.exists() {
        return Err("unexpected argument or output already exists".into());
    }
    let mut predictions = BTreeMap::<u64, (Value, Option<(f64, f64)>)>::new();
    let reader =
        BufReader::new(File::open(capture.join("predictions.jsonl")).map_err(|e| e.to_string())?);
    for line in reader.lines() {
        let row: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if row["roi_frame_key"]["roi_id"].as_u64() != Some(eye_id) {
            continue;
        }
        let eye = &row["predictions"]["eye_candidate"];
        let gaze = &eye["centers_and_gaze"]["virtual_contact_surface_gaze"];
        let Some(source) = gaze["source_timestamp_ns"].as_u64() else {
            continue;
        };
        let pupil = &eye["pupil"]["sam31_proposal_void_fit"];
        let anchor = (pupil["source_timestamp_ns"].as_u64() == Some(source))
            .then(|| {
                Some((
                    pupil["limbus_anchor"][0].as_f64()?,
                    pupil["limbus_anchor"][1].as_f64()?,
                ))
            })
            .flatten();
        predictions
            .entry(source)
            .or_insert_with(|| (gaze.clone(), anchor));
    }
    let frames = fs::read_to_string(capture.join("frames.jsonl"))
        .map_err(|e| e.to_string())?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?
        .into_iter()
        .filter(|r| r["eye_id"].as_u64() == Some(eye_id))
        .skip(start)
        .take(count)
        .collect::<Vec<_>>();
    if frames.is_empty() {
        return Err("empty RAW subset".into());
    }
    let mut raw = File::open(capture.join(format!("{label}.raw10"))).map_err(|e| e.to_string())?;
    let mut motion = raw_motion_octrees::NativeGlobalSimilarityTracker::default();
    let mut timeline = GlobalSimilarityTimeline::default();
    let mut trackers: [SurfaceGazeTracker; 2] =
        std::array::from_fn(|_| SurfaceGazeTracker::default());
    let mut previous_session = None;
    let mut previous_source = None;
    let mut clock_start = (integer(&frames[0], "timestamp_ns")?, Instant::now());
    let mut fresh_seed = true;
    let mut cases = Vec::new();
    let mut reliable_raw = 0usize;
    let mut paired_motion = 0usize;
    let mut changed_signs = 0usize;
    let mut reframes = 0usize;
    let mut previous_origin = None;
    let mut maximum_area_difference = 0.0_f64;
    for (index, record) in frames.iter().enumerate() {
        let ns = integer(record, "timestamp_ns")?;
        let session = record["region"]["session"].as_u64();
        if session != previous_session || previous_source.is_some_and(|previous| ns <= previous) {
            trackers = std::array::from_fn(|_| SurfaceGazeTracker::default());
            motion.clear();
            timeline = GlobalSimilarityTimeline::default();
            clock_start = (ns, Instant::now());
            fresh_seed = true;
        }
        previous_session = session;
        previous_source = Some(ns);
        let origin = (
            integer(record, "sensor_x")? as u32,
            integer(record, "sensor_y")? as u32,
        );
        reframes += usize::from(previous_origin.is_some_and(|previous| previous != origin));
        previous_origin = Some(origin);
        let width = integer(record, "width")? as usize;
        let height = integer(record, "height")? as usize;
        raw.seek(SeekFrom::Start(integer(record, "offset")?))
            .map_err(|e| e.to_string())?;
        let mut bytes = vec![0; integer(record, "length")? as usize];
        raw.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let pixels = Arc::new(raw10::try_unpack_raw10(
            &bytes,
            width,
            height,
            integer(record, "stride")? as usize,
        )?);
        let transport = motion.observe(pixels, width, height, origin.0, origin.1);
        reliable_raw += usize::from(transport.reliable);
        timeline.observe_frame(ns, transport);
        let Some((recorded, anchor)) = predictions.get(&ns) else {
            continue;
        };
        let number = |field: &str| {
            recorded[field]
                .as_f64()
                .ok_or_else(|| format!("missing gaze {field}"))
        };
        let vector = [0, 1, 2].map(|i| {
            recorded["relative_gaze_vector"][i]
                .as_f64()
                .unwrap_or(f64::NAN)
        });
        if !vector.iter().all(|v| v.is_finite()) {
            return Err("invalid recorded vector".into());
        }
        let radius = (number("rectified_area_px2")? / std::f64::consts::PI).sqrt();
        let face_radius = number("bucketed_face_radius_px")?;
        let near = [0, 1].map(|i| {
            recorded["camera_near_point_sensor"][i]
                .as_f64()
                .unwrap_or(f64::NAN)
        });
        let center = (
            near[0] - face_radius * vector[0] - f64::from(origin.0),
            near[1] - face_radius * vector[1] - f64::from(origin.1),
        );
        // Erase the recorded sign: both normal branches describe this exact
        // ellipse. Only the initial checkpoint below inherits the live sign.
        let outer = raw_iris_focus::OuterIrisBoundary {
            center,
            major_radius: radius,
            minor_radius: radius * vector[2],
            angle: (-vector[0])
                .atan2(vector[1])
                .rem_euclid(std::f64::consts::PI),
            points: vec![raw_iris_focus::OuterIrisPoint::default(); 8],
            ..raw_iris_focus::OuterIrisBoundary::default()
        };
        let now = clock_start.1 + Duration::from_nanos(ns - clock_start.0);
        // Identical input and legacy tracker code in both arms. Clearing only
        // this new window is an OFFLINE ablation, never a live UI/env option.
        trackers[0].motion_sign_window = Default::default();
        let mut available = false;
        let samples: [Option<SurfaceGazeSample>; 2] = std::array::from_fn(|arm| {
            let source_motion = trackers[arm]
                .last_keyed_source_timestamp_ns
                .and_then(|previous| timeline.reliable_between(previous, ns));
            available |= source_motion.is_some();
            let mut sample = trackers[arm].observe_keyed_with_global_similarity(
                ns,
                now,
                origin,
                *anchor,
                &outer,
                source_motion,
            )?;
            if fresh_seed {
                let same = sample.relative_gaze.right * vector[0]
                    + sample.relative_gaze.down * vector[1]
                    >= 0.0;
                if !same {
                    trackers[arm].selected_sign_hypothesis =
                        1 - trackers[arm].selected_sign_hypothesis;
                    trackers[arm].kinematic_history.clear();
                    trackers[arm].floating_center_sensor = None;
                    trackers[arm].floating_near_point_sensor = None;
                    sample.relative_gaze =
                        RelativeGazeVector::from_projected(vector[0], vector[1])?;
                    sample.near_surface_point_sensor_px = (near[0], near[1]);
                }
                trackers[arm].sign_resolved = recorded["sign_resolved"].as_bool().unwrap_or(false);
                trackers[arm].sign_epoch = recorded["sign_epoch"].as_u64().unwrap_or(0);
                sample.sign_resolved = trackers[arm].sign_resolved;
                sample.sign_epoch = trackers[arm].sign_epoch;
                trackers[arm].last_keyed_sample = Some(sample);
            }
            Some(sample)
        });
        fresh_seed = false;
        paired_motion += usize::from(available);
        if let (Some(a), Some(b)) = (samples[0], samples[1]) {
            changed_signs += usize::from(
                a.relative_gaze.right * b.relative_gaze.right
                    + a.relative_gaze.down * b.relative_gaze.down
                    < 0.0,
            );
            maximum_area_difference =
                maximum_area_difference.max((a.frontal_equivalent_disk_area_px2 - b.frontal_equivalent_disk_area_px2).abs());
        }
        cases.push(json!({"source_ns":ns,"sequence":record["sequence"],"origin":origin,
            "source_motion_available":available,"transport_reliable":transport.reliable,
            "transport_residual_px":transport.motion.residual,"baseline":json_surface_gaze(samples[0]),
            "candidate":json_surface_gaze(samples[1]),"recorded":recorded}));
        if index % 500 == 0 {
            eprintln!(
                "CONTACT_SIGN_REPLAY {label} RAW={index}/{} samples={}",
                frames.len(),
                cases.len()
            );
        }
    }
    let report = json!({"capture":capture,"label":label,"start":start,"raw_frames":frames.len(),
        "reliable_raw_motion_frames":reliable_raw,"source_matched_contacts":cases.len(),
        "contacts_with_source_motion":paired_motion,"changed_sign_samples":changed_signs,
        "roi_origin_changes":reframes,"max_baseline_candidate_rectified_area_difference_px2":maximum_area_difference,
        "limitations":["Frozen recorded ellipse geometry recovered from published contact; no segmentation rerun or human sign labels.",
            "Initial recorded sign checkpoint is shared by both arms; subsequent predictions are independent single-eye rollouts.",
            "Both arms recompute independent native RAW transport on recorded frames; unavailable/dropped frames cannot invent transport.",
            "Only the new motion window is disabled in baseline. No other-eye evidence enters solving.",
            "Area/ellipse localization invariance is a sign-only contract, not proof of geometry accuracy or independent metric scale."],
        "cases":cases});
    fs::write(
        &output,
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    eprintln!("CONTACT_SIGN_REPLAY {label} complete: contacts={} changed={changed_signs} RAW-motion={reliable_raw}/{} paired-motion={paired_motion}",cases.len(),frames.len());
    Ok(())
}
