//! Offline matched bootstrap trials with an explicit historical baseline.
//!
//! Established remains the baseline here; live tracking uses MotionWindowFallback.
//!
//! Replay the same recorded conics with all policies; only RAW motion supplies
//! scale/transport. Arrival scheduling uses the frame which first published a
//! contact, not the contact's earlier exposure. Missing prehistory stays missing.
use super::*;
use crate::eye_scene_model::SignAcquisitionPolicy;
use crate::raw_motion_octrees::NativeGlobalSimilarityEvidence;
use crate::roi_continuity::{RoiContinuitySession, RoiSource};
use std::io::{BufRead, BufReader};

const POLICIES: [SignAcquisitionPolicy; 4] = [
    SignAcquisitionPolicy::Established,
    SignAcquisitionPolicy::MotionWindowFallback,
    SignAcquisitionPolicy::MotionWindowOnly,
    SignAcquisitionPolicy::ReliableMotionSeed,
];

fn number(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key(String, u64);

fn key(row: &Value, timestamp_ns: u64) -> Result<Key, String> {
    let clock = &row["source_clock"]["source_key"];
    let domain = clock["stream_epoch"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| {
            // Older archives have only a sensor session. Never compare two unknown
            // sensor epochs as though their timestamps were wall time.
            number(&row["region"]["session"]).map(|session| format!("region:{session}"))
        })
        .ok_or("frame has no clock/region identity")?;
    Ok(Key(domain, timestamp_ns))
}

fn rows(path: &Path) -> Result<Vec<Value>, String> {
    BufReader::new(File::open(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .lines()
        .map(|line| {
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())
        })
        .collect()
}

#[derive(Clone)]
struct Frame {
    row: Value,
    capture: PathBuf,
    key: Key,
    source: RoiSource,
}

fn same_exposure(a: &Frame, b: &Frame, label: &str) -> Result<bool, String> {
    if a.source != b.source
        || a.row["stride"] != b.row["stride"]
        || a.row["length"] != b.row["length"]
    {
        return Ok(false);
    }
    let read = |f: &Frame| -> Result<Vec<u8>, String> {
        let mut file =
            File::open(f.capture.join(format!("{label}.raw10"))).map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(integer(&f.row, "offset")?))
            .map_err(|e| e.to_string())?;
        let mut bytes = vec![0; integer(&f.row, "length")? as usize];
        file.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        Ok(bytes)
    };
    Ok(read(a)? == read(b)?)
}

#[derive(Clone)]
struct Contact {
    recorded: Value,
    anchor: Option<(f64, f64)>,
    source: Key,
    ready_host_ns: Option<u64>,
}

fn contact(row: &Value, frame: &Frame) -> Option<Contact> {
    let eye = &row["predictions"]["eye_candidate"];
    let recorded = &eye["centers_and_gaze"]["virtual_contact_surface_gaze"];
    let ns = number(&recorded["source_timestamp_ns"])?;
    let pupil = &eye["pupil"]["sam31_proposal_void_fit"];
    let anchor = (number(&pupil["source_timestamp_ns"]) == Some(ns))
        .then(|| {
            Some((
                pupil["limbus_anchor"][0].as_f64()?,
                pupil["limbus_anchor"][1].as_f64()?,
            ))
        })
        .flatten();
    Some(Contact {
        recorded: recorded.clone(),
        anchor,
        source: Key(frame.key.0.clone(), ns),
        ready_host_ns: number(&row["prediction_ready"]["host_monotonic_ns"]),
    })
}

/// The two signs describe the SAME ellipse: recover it without using the
/// recorded branch as a target. This cannot recover unpublished rejected fits.
fn ellipse(contact: &Contact, source: &Frame) -> Result<raw_iris_focus::OuterIrisBoundary, String> {
    let gaze = &contact.recorded;
    let vector = [0, 1, 2].map(|i| gaze["relative_gaze_vector"][i].as_f64().unwrap_or(f64::NAN));
    let radius =
        (gaze["rectified_area_px2"].as_f64().unwrap_or(f64::NAN) / std::f64::consts::PI).sqrt();
    let quantized = gaze["bucketed_face_radius_px"].as_f64().unwrap_or(f64::NAN);
    let center = [0, 1].map(|i| {
        gaze["camera_near_point_sensor"][i]
            .as_f64()
            .unwrap_or(f64::NAN)
            - quantized * vector[i]
            - [source.source.sensor_x, source.source.sensor_y][i] as f64
    });
    if !vector
        .into_iter()
        .chain(center)
        .chain([radius, quantized])
        .all(f64::is_finite)
        || radius <= 0.0
        || quantized <= 0.0
        || vector[2] <= 0.0
        || vector[2] > 1.0
        || (vector.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() > 1e-6
    {
        return Err("invalid serialized contact geometry".into());
    }
    Ok(raw_iris_focus::OuterIrisBoundary {
        center: (center[0], center[1]),
        major_radius: radius,
        minor_radius: radius * vector[2],
        angle: (-vector[0])
            .atan2(vector[1])
            .rem_euclid(std::f64::consts::PI),
        points: vec![raw_iris_focus::OuterIrisPoint::default(); 8],
        ..raw_iris_focus::OuterIrisBoundary::default()
    })
}

fn seed_recorded(
    tracker: &mut SurfaceGazeTracker,
    sample: &mut SurfaceGazeSample,
    recorded: &Value,
) {
    let x = recorded["relative_gaze_vector"][0].as_f64().unwrap();
    let y = recorded["relative_gaze_vector"][1].as_f64().unwrap();
    if sample.relative_gaze.right * x + sample.relative_gaze.down * y < 0.0 {
        tracker.selected_sign_hypothesis = 1 - tracker.selected_sign_hypothesis;
        tracker.kinematic_history.clear();
        tracker.floating_center_sensor = None;
        tracker.floating_near_point_sensor = None;
        sample.relative_gaze = RelativeGazeVector::from_projected(x, y).unwrap();
        sample.near_surface_point_sensor_px = (
            recorded["camera_near_point_sensor"][0].as_f64().unwrap(),
            recorded["camera_near_point_sensor"][1].as_f64().unwrap(),
        );
    }
    tracker.sign_resolved = recorded["sign_resolved"].as_bool().unwrap_or(false);
    tracker.sign_epoch = number(&recorded["sign_epoch"]).unwrap_or(0);
    sample.sign_resolved = tracker.sign_resolved;
    sample.sign_epoch = tracker.sign_epoch;
    tracker.last_keyed_sample = Some(*sample);
}

/// Short, candidate-independent RAW scale chains. Values are comparable ONLY
/// inside one reference. Uncertainty is a bounded engineering allowance.
#[derive(Default)]
struct ScaleChain {
    reference: u64,
    started_ns: u64,
    scale: f64,
    allowance: f64,
}

impl ScaleChain {
    fn observe(
        &mut self,
        ns: u64,
        motion: NativeGlobalSimilarityEvidence,
        reset: bool,
    ) -> Option<(u64, f64, f64)> {
        let m = motion.motion;
        let scale =
            f64::from(1.0 + m.diagonal_coefficient_delta).hypot(f64::from(m.rotation_coefficient));
        if reset
            || !motion.reliable
            || !(0.8..=1.25).contains(&scale)
            || ns.saturating_sub(self.started_ns) > 1_000_000_000
            || self.allowance > 0.25
        {
            self.reference += 1;
            self.started_ns = ns;
            self.scale = 1.0;
            self.allowance = 0.0;
            return None;
        }
        self.scale *= scale;
        self.allowance += 0.01 + f64::from(m.residual) / 100.0;
        (self.allowance <= 0.25).then_some((self.reference, self.scale, self.allowance))
    }
}

pub(crate) fn run<I: Iterator<Item = String>>(args: I) -> Result<(), String> {
    let mut args = args.peekable();
    let output = PathBuf::from(args.next().ok_or("OUTPUT EYE [--arrival|--source] [--cold|--seed-recorded] [--no-motion] [--limit N] CAPTURE_DIR...")?);
    if output.exists() {
        return Err("output already exists".into());
    }
    let label = args.next().ok_or("missing eye label")?;
    let eye_id = match label.as_str() {
        "subject-right" => 1,
        "subject-left" => 2,
        _ => return Err("invalid eye".into()),
    };
    let mut arrival = true;
    let mut seed = false;
    let mut no_motion = false;
    let mut limit = usize::MAX;
    let mut captures = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--arrival" => arrival = true,
            "--source" => arrival = false,
            "--cold" => seed = false,
            "--seed-recorded" => seed = true,
            "--no-motion" => no_motion = true,
            "--limit" => limit = parse_usize(args.next(), usize::MAX, "limit")?,
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}")),
            _ => captures.push(PathBuf::from(arg)),
        }
    }
    if captures.is_empty() {
        return Err("missing captures".into());
    }
    let mut frames = Vec::new();
    let mut predictions = BTreeMap::new();
    let mut keys = BTreeMap::<Key, Frame>::new();
    let mut verified_duplicate_exposures = 0;
    for capture in &captures {
        let mut local_keys = BTreeMap::new();
        for row in rows(&capture.join("frames.jsonl"))? {
            if row["eye_id"].as_u64() != Some(eye_id) {
                continue;
            }
            let source = RoiSource {
                sequence: integer(&row, "sequence")?,
                timestamp_ns: integer(&row, "timestamp_ns")?,
                sensor_x: integer(&row, "sensor_x")? as u32,
                sensor_y: integer(&row, "sensor_y")? as u32,
                width: integer(&row, "width")? as usize,
                height: integer(&row, "height")? as usize,
            };
            let key = key(&row, source.timestamp_ns)?;
            let frame = Frame {
                row,
                key: key.clone(),
                capture: capture.clone(),
                source,
            };
            local_keys.insert(source.timestamp_ns, frame.clone());
            if let Some(previous) = keys.get(&key) {
                if !same_exposure(previous, &frame, &label)? {
                    return Err(format!("conflicting RAW/geometry for exposure {key:?}"));
                }
                verified_duplicate_exposures += 1;
            } else {
                keys.insert(key, frame.clone());
                frames.push(frame);
            }
        }
        for row in rows(&capture.join("predictions.jsonl"))? {
            if row["roi_frame_key"]["roi_id"].as_u64() != Some(eye_id) {
                continue;
            }
            let ns = number(&row["roi_frame_key"]["sensor_timestamp_ns"])
                .ok_or("prediction without ROI time")?;
            if let Some(frame) = local_keys.get(&ns) {
                // Keep no-surface entries as well: they are measured dropouts,
                // not an invitation to backfill the next available solution.
                predictions
                    .entry(frame.key.clone())
                    .or_insert_with(|| contact(&row, frame));
            }
        }
    }
    // Clock-less old archives can be replayed within one epoch. Without host
    // arrival evidence we cannot infer the chronology of unrelated sensors.
    let arrival_unix = |f: &Frame| {
        number(&f.row["source_clock"]["host_arrival_unix_ns"])
            .or_else(|| number(&f.row["host_arrival_unix_ns"]))
    };
    if frames.iter().all(|f| arrival_unix(f).is_some()) {
        frames.sort_by_key(|f| arrival_unix(f).unwrap());
    } else if frames.iter().all(|f| f.key.0 == frames[0].key.0) {
        frames.sort_by_key(|f| f.source.timestamp_ns);
    } else {
        return Err("multiple source epochs without complete host-arrival chronology".into());
    }
    frames.truncate(limit);
    if frames.is_empty() {
        return Err("empty eye corpus".into());
    }
    let by_key: BTreeMap<_, _> = frames.iter().map(|f| (f.key.clone(), f.clone())).collect();
    let mut at_source = BTreeMap::new();
    for frame in &frames {
        if let Some(Some(c)) = predictions.get(&frame.key) {
            at_source
                .entry(c.source.clone())
                .or_insert_with(|| c.clone());
        }
    }
    let mut files = BTreeMap::new();
    for capture in &captures {
        files.insert(
            capture.clone(),
            File::open(capture.join(format!("{label}.raw10"))).map_err(|e| e.to_string())?,
        );
    }
    let make_trackers = || {
        POLICIES.map(|acquisition_policy| SurfaceGazeTracker {
            acquisition_policy,
            ..SurfaceGazeTracker::default()
        })
    };
    let mut trackers = make_trackers();
    let mut motion = raw_motion_octrees::NativeGlobalSimilarityTracker::default();
    let mut timeline = GlobalSimilarityTimeline::default();
    let mut session = RoiContinuitySession::default();
    let mut domain = String::new();
    let mut scale_chain = ScaleChain::default();
    let mut scales = BTreeMap::new();
    let mut first_samples = [true; 4];
    let mut cases = Vec::new();
    let mut missing_contact = 0;
    let mut missing_source = 0;
    let mut mismatched_age = 0;
    let mut reliable_raw = 0;
    let mut resets = 0;
    let mut reframes = 0;
    let mut missing_host_clock = 0;
    let started = Instant::now();
    let mut host_origin = None;
    let mut sensor_origin = frames[0].source.timestamp_ns;
    let mut matched_input_geometry_delta = 0.0_f64;
    let mut previous_sample = [None::<SurfaceGazeSample>; 4];
    let mut sign_changes = [0; 4];
    let mut runtime_ns = [0u128; 4];
    let mut calls = [0usize; 4];
    let mut attempts = [0usize; 4];
    for frame in &frames {
        let ns = frame.source.timestamp_ns;
        let clock_changed = domain != frame.key.0;
        if clock_changed {
            domain = frame.key.0.clone();
            session.invalidate();
            motion.clear();
            timeline = GlobalSimilarityTimeline::default();
            host_origin = None;
            sensor_origin = ns;
        }
        let transition = session.observe(frame.source, SAM31_RESULT_MAX_AGE_NS);
        let reset = transition.resets_tracking();
        if reset {
            trackers = make_trackers();
            first_samples = [true; 4];
            previous_sample = [None; 4];
            resets += 1;
            motion.clear();
            timeline = GlobalSimilarityTimeline::default();
        }
        reframes += usize::from(transition == RoiTransition::CompatibleTranslation);
        if !transition.accepts_source() {
            continue;
        }
        let raw = files.get_mut(&frame.capture).unwrap();
        raw.seek(SeekFrom::Start(integer(&frame.row, "offset")?))
            .map_err(|e| e.to_string())?;
        let mut bytes = vec![0; integer(&frame.row, "length")? as usize];
        raw.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        let pixels = Arc::new(raw10::try_unpack_raw10(
            &bytes,
            frame.source.width,
            frame.source.height,
            integer(&frame.row, "stride")? as usize,
        )?);
        let transport = motion.observe(
            pixels,
            frame.source.width,
            frame.source.height,
            frame.source.sensor_x,
            frame.source.sensor_y,
        );
        reliable_raw += usize::from(transport.reliable);
        timeline.observe_frame(ns, transport);
        scales.insert(frame.key.clone(), scale_chain.observe(ns, transport, reset));
        let contact = if arrival {
            predictions.get(&frame.key).and_then(Option::as_ref)
        } else {
            at_source.get(&frame.key)
        };
        let Some(contact) = contact else {
            missing_contact += 1;
            continue;
        };
        let Some(source) = by_key.get(&contact.source) else {
            missing_source += 1;
            continue;
        };
        if contact.source.1 > ns
            || ns - contact.source.1 > SAM31_RESULT_MAX_AGE_NS
            || contact.source.1 < session.selection_started_ns().unwrap_or(ns)
        {
            mismatched_age += 1;
            continue;
        }
        let outer = ellipse(contact, source)?;
        let host = arrival.then_some(contact.ready_host_ns).flatten();
        let elapsed = if let Some(host) = host {
            host.saturating_sub(*host_origin.get_or_insert(host))
        } else {
            missing_host_clock += usize::from(arrival);
            ns.saturating_sub(sensor_origin)
        };
        let now = started + Duration::from_nanos(elapsed);
        let mut motion_available = [false; 4];
        let mut motion_residual_px = [None; 4];
        let mut fresh = [false; 4];
        let samples = std::array::from_fn::<_, 4, _>(|arm| {
            let t = &mut trackers[arm];
            fresh[arm] = t
                .last_keyed_attempted_source_timestamp_ns
                .is_none_or(|last| contact.source.1 > last);
            let source_motion = (!no_motion)
                .then(|| {
                    t.last_keyed_source_timestamp_ns
                        .and_then(|previous| timeline.reliable_between(previous, contact.source.1))
                })
                .flatten();
            motion_available[arm] = source_motion.is_some();
            motion_residual_px[arm] = source_motion.map(|m| m.motion.residual);
            let before = Instant::now();
            let sample = t.observe_keyed_with_global_similarity(
                contact.source.1,
                now,
                (source.source.sensor_x, source.source.sensor_y),
                contact.anchor,
                &outer,
                source_motion,
            );
            runtime_ns[arm] += before.elapsed().as_nanos();
            calls[arm] += 1;
            attempts[arm] += usize::from(fresh[arm]);
            let mut sample = sample?;
            if first_samples[arm] && seed {
                seed_recorded(t, &mut sample, &contact.recorded);
            }
            first_samples[arm] = false;
            if fresh[arm] {
                if let Some(previous) = previous_sample[arm] {
                    sign_changes[arm] += usize::from(sample.sign_epoch != previous.sign_epoch);
                }
                previous_sample[arm] = Some(sample);
            }
            Some(sample)
        });
        for sample in samples.into_iter().flatten() {
            if !sample.relative_gaze.is_camera_facing() {
                return Err("non-convex candidate".into());
            }
            matched_input_geometry_delta = matched_input_geometry_delta.max(
                (sample.frontal_equivalent_disk_area_px2
                    - std::f64::consts::PI * outer.major_radius.powi(2))
                .abs(),
            );
        }
        let safe = |origin: (u32, u32)| {
            let cx = outer.center.0 + f64::from(source.source.sensor_x) - f64::from(origin.0);
            let cy = outer.center.1 + f64::from(source.source.sensor_y) - f64::from(origin.1);
            let (s, c) = outer.angle.sin_cos();
            let ex = (outer.major_radius * c).hypot(outer.minor_radius * s);
            let ey = (outer.major_radius * s).hypot(outer.minor_radius * c);
            let margin = (frame.source.width.min(frame.source.height) as f64 * 0.04).max(4.0);
            cx - ex >= margin
                && cy - ey >= margin
                && cx + ex <= frame.source.width as f64 - margin
                && cy + ey <= frame.source.height as f64 - margin
        };
        let scale = scales.get(&contact.source).copied().flatten();
        cases.push(json!({"clock":frame.key.0,"frame_source_ns":ns.to_string(),"sequence":frame.source.sequence,
            "contact_source_ns":contact.source.1.to_string(),"source_lag_ms":(ns-contact.source.1) as f64 / 1e6,
            "fresh":fresh,"source_motion_available":motion_available,"source_motion_residual_px":motion_residual_px,"input_anchor":contact.anchor,
            "source_and_presentation_safe":safe((source.source.sensor_x,source.source.sensor_y)) && safe((frame.source.sensor_x,frame.source.sensor_y)),
            "recorded_signed":contact.recorded["sign_resolved"],"recorded_vector":contact.recorded["relative_gaze_vector"],
            "source_scale":scale.map(|(reference,scale,allowance)|json!({"reference":reference,"linear_scale":scale,"heuristic_fractional_allowance":allowance,
                "sn_feida":std::f64::consts::PI*outer.major_radius.powi(2)/scale.powi(2)})),
            "samples":samples.map(json_surface_gaze),
            "states":trackers.each_ref().map(|t|json!({"ema":t.contact_sign_hypotheses.map(|h|h.map(|v|v.residual_ema)),
                "observations":t.contact_sign_hypotheses.map(|h|h[0].observations),"reliable_motion_observations":t.reliable_motion_observations,
                "pending_anchor_votes":t.pending_sign_frames}))}));
    }
    let summaries: Vec<_> = (0..4).map(|arm| {
        let unique: Vec<_> = cases.iter().filter(|c|c["fresh"][arm]==true).collect();
        let signed: Vec<_> = unique.iter().filter(|c|c["samples"][arm]["sign_resolved"]==true).collect();
        let eligible = signed.iter().filter(|c|c["source_and_presentation_safe"]==true).count();
        let agrees = signed.iter().filter(|c| {
            let a=&c["samples"][arm]["relative_gaze_vector"]; let b=&c["recorded_vector"];
            (0..2).map(|i|a[i].as_f64().unwrap_or(0.0)*b[i].as_f64().unwrap_or(0.0)).sum::<f64>()>=0.0
        }).count();
        json!({"policy":format!("{:?}",POLICIES[arm]),"unique_attempts":attempts[arm],"signed_unique":signed.len(),
            "eligible_unique":eligible,"sign_epoch_changes":sign_changes[arm],"agreement_with_recorded_branch_not_truth":agrees,
            "mean_tracker_us_per_call":runtime_ns[arm] as f64 / 1000.0 / calls[arm].max(1) as f64,
            "first_signed_frame_ns":signed.first().map(|c|&c["frame_source_ns"])})
    }).collect();
    let report = json!({"schema":"buttercup-sign-acquisition-trial-v1","captures":captures,"eye":label,
        "schedule":if arrival{"recorded-publication-frame"}else{"idealized-source-order"},"seed_recorded":seed,"no_motion":no_motion,
        "raw_frames":frames.len(),"verified_duplicate_exposures":verified_duplicate_exposures,
        "reliable_raw_motion_frames":reliable_raw,"source_resets":resets,"compatible_roi_moves":reframes,
        "missing_contact_rows":missing_contact,"missing_source_rows":missing_source,"ineligible_source_age_rows":mismatched_age,
        "publication_host_clock_missing_rows":missing_host_clock,"maximum_frontal_area_change_px2":matched_input_geometry_delta,
        "human_gaze_labels":false,"human_limbus_labels":false,"summaries":summaries,
        "eligible_definition":"Signed and inside source/presentation ROI with the calibration margin; NOT the complete calibration admission state machine.",
        "limitations":["Recorded fitted ellipses, not a fresh SAM run. Unpublished rejected fits and pre-recording state are unavailable.",
            "Recorded publication frames and available host-ready clocks approximate delivery; exact internal SAM consumption instants are not archived.",
            "More resolved samples and agreement with the recorded sign are NOT correctness measurements.",
            "Ellipse localization is invariant by construction; no new human-label localization score or measured metric scale is claimed.",
            "RAW scale is independent of candidate radius but not independent of all iris pixels; compare SN-FEIDA only inside the same short reference chain.",
            "Moving-target warm-up is an active intervention and cannot be validated from a stationary-target recording."],"cases":cases});
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer(&mut file, &report).map_err(|e| e.to_string())?;
    eprintln!(
        "SIGN_ACQUISITION_TRIAL {}",
        json!({"output":output,"eye":label,"raw_frames":frames.len(),"summaries":summaries})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_identity_includes_sensor_epoch_and_preserves_integer_nanoseconds() {
        let a = json!({"source_clock":{"source_key":{"stream_epoch":"a"}}});
        let b = json!({"source_clock":{"source_key":{"stream_epoch":"b"}}});
        let ns = 1_690_062_817_325_594_229;
        assert_ne!(key(&a, ns), key(&b, ns));
        assert_eq!(number(&json!(ns.to_string())), Some(ns));
        assert!(key(&json!({}), ns).is_err());
    }

    #[test]
    fn scale_never_bridges_missing_motion_or_candidate_radius() {
        let mut chain = ScaleChain::default();
        let motion = NativeGlobalSimilarityEvidence {
            reliable: true,
            motion: raw_motion_octrees::SimilarityMotion {
                diagonal_coefficient_delta: 0.01,
                rotation_coefficient: 0.02,
                residual: 0.2,
                support: 16,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(chain.observe(1, motion, true).is_none());
        let (reference, scale, _) = chain.observe(100_000_001, motion, false).unwrap();
        assert!((scale - 1.01_f64.hypot(0.02)).abs() < 1e-7);
        assert!(chain
            .observe(200_000_001, Default::default(), false)
            .is_none());
        assert_ne!(
            chain.observe(300_000_001, motion, false).unwrap().0,
            reference
        );
    }
}
