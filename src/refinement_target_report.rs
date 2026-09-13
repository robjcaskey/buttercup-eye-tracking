//! Offline, source-matched target efficacy. No fitting, model inference, target
//! feedback, calibration changes or blink selection from prediction error.
use super::*;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, ErrorKind};

type V3 = [f64; 3];
fn timestamp(v: &Value) -> Result<i128> {
    if let Some(n) = v.as_u64() {
        return Ok(n as i128);
    }
    let n: i128 = string(v)?.parse()?;
    // Leave ample headroom for offset, lag and acquisition arithmetic.
    if !(-(u64::MAX as i128)..=u64::MAX as i128).contains(&n) {
        return Err("clock value outside supported nanosecond range".into());
    }
    Ok(n)
}
fn vector(v: &Value) -> Result<V3> {
    let mut result = [0.0; 3];
    for i in 0..3 {
        result[i] = v[i]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or("invalid vector")?;
    }
    Ok(result)
}
fn dot(a: V3, b: V3) -> f64 {
    (0..3).map(|i| a[i] * b[i]).sum()
}
fn unit(a: V3) -> Option<V3> {
    let n = dot(a, a).sqrt();
    (n.is_finite() && n > 1e-9).then(|| a.map(|x| x / n))
}
fn angle(a: V3, b: V3) -> Option<f64> {
    Some(dot(unit(a)?, unit(b)?).clamp(-1.0, 1.0).acos().to_degrees())
}

struct Monitor {
    center: V3,
    right: V3,
    down: V3,
    width: f64,
    height: f64,
}
impl Monitor {
    fn parse(v: &Value) -> Result<Self> {
        let m = Self {
            center: vector(&v["center_inches"])?,
            right: vector(&v["right_axis"])?,
            down: vector(&v["down_axis"])?,
            width: v["width_inches"].as_f64().ok_or("missing monitor width")?,
            height: v["height_inches"]
                .as_f64()
                .ok_or("missing monitor height")?,
        };
        if !m.width.is_finite()
            || !m.height.is_finite()
            || m.width <= 0.0
            || m.height <= 0.0
            || (dot(m.right, m.right) - 1.0).abs() > 1e-5
            || (dot(m.down, m.down) - 1.0).abs() > 1e-5
            || dot(m.right, m.down).abs() > 1e-5
        {
            return Err("invalid monitor basis/size".into());
        }
        Ok(m)
    }
    // Deliberately retain the capture's fixed reference-eye mapping. No
    // current/personal calibration or candidate-dependent origin is borrowed.
    fn target(&self, xy: [f64; 2], viewport: [f64; 2]) -> V3 {
        std::array::from_fn(|i| {
            self.center[i]
                + self.right[i] * self.width * (xy[0] / (viewport[0] - 1.0) - 0.5)
                + self.down[i] * self.height * (xy[1] / (viewport[1] - 1.0) - 0.5)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Target {
    xy: [f64; 2],
    viewport: [f64; 2],
    site: usize,
    site_start: i128,
}
struct Presentation {
    time: i128,
    target: Option<Target>,
}
struct Evidence {
    session: String,
    monitor: Value,
    targets: Vec<Presentation>,
    first_receipt_sequence: u64,
}

fn pair(v: &Value) -> Result<[f64; 2]> {
    Ok([
        v[0].as_f64()
            .filter(|x| x.is_finite())
            .ok_or("invalid coordinate")?,
        v[1].as_f64()
            .filter(|x| x.is_finite())
            .ok_or("invalid coordinate")?,
    ])
}

fn read_evidence(path: &Path) -> Result<Evidence> {
    read_evidence_stream(BufReader::new(File::open(path)?))
}

fn read_evidence_stream(mut file: impl Read) -> Result<Evidence> {
    let mut session = None::<String>;
    let mut monitor = Value::Null;
    let mut targets: Vec<Presentation> = vec![];
    let mut anchor = None;
    let mut current = None::<Target>;
    let mut site = 0;
    let mut receipt = None::<u64>;
    loop {
        let mut header = [0u8; 24];
        match file.read_exact(&mut header[..1]) {
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
            other => other?,
        }
        file.read_exact(&mut header[1..])?;
        if &header[..4] != b"OIM1" {
            return Err("not native OIM1 metadata".into());
        }
        let n = u64::from_le_bytes(header[8..16].try_into().unwrap());
        if n > 4_000_000 {
            return Err("oversized metadata record".into());
        }
        let mut payload = vec![0; n as usize];
        file.read_exact(&mut payload)?;
        let row: Value = serde_json::from_slice(&payload)?;
        if row["event"] == "recording_start_snapshot" || row["event"] == "presentation" {
            let id = string(&row["viewer_session_id"])?;
            if id.is_empty() || session.as_ref().is_some_and(|old| old != id) {
                return Err("mixed/missing presentation session identity".into());
            }
            session = Some(id.to_owned());
        }
        let next = &row["configuration"]["mapping"]["monitor_plane"];
        if !next.is_null() {
            if !monitor.is_null() && monitor != *next {
                return Err("monitor changed during capture".into());
            }
            monitor = next.clone();
        }
        if row["event"] != "presentation" {
            continue;
        }
        let recovery = &row["optical_clock"]["latest_recovery"];
        if recovery["verified_temporal_word"] == true {
            let seq = u64::try_from(timestamp(&recovery["sequence"])?)?;
            receipt = Some(receipt.map_or(seq, |old| old.min(seq)));
        }
        let time = timestamp(&row["host_submit_end_unix_ns"])?;
        if targets.last().is_some_and(|p| time < p.time) {
            return Err("nonmonotonic host presentation clock".into());
        }
        let active = array(&row["active_targets"])?;
        let visible: Vec<_> = active
            .iter()
            .filter(|t| t["role"] == "relative-motion-stimulus" && t["visible"] == true)
            .collect();
        if visible.len() > 1 {
            return Err("multiple simultaneous fixation targets".into());
        }
        let target = if let Some(t) = visible.first() {
            let xy = pair(&t["appearance"]["physical_pixel_position"])?;
            let offset = pair(&t["appearance"]["offset_px"])?;
            let viewport = pair(&row["viewport_px"])?;
            if viewport.iter().any(|x| *x <= 1.0) {
                return Err("invalid viewport".into());
            }
            let next_anchor = [xy[0] - offset[0], xy[1] - offset[1]];
            let changed = anchor != Some(next_anchor)
                || current.is_none()
                || current.is_some_and(|c| c.viewport != viewport);
            let site_start = if changed {
                if anchor.is_some() {
                    site += 1;
                }
                time
            } else {
                current.unwrap().site_start
            };
            anchor = Some(next_anchor);
            Some(Target {
                xy,
                viewport,
                site,
                site_start,
            })
        } else {
            None
        };
        // Retain removal explicitly: never hold the final target after it is
        // hidden, nor carry acquisition eligibility through a hidden interval.
        if targets.last().is_none_or(|p| p.target != target) {
            targets.push(Presentation { time, target });
        }
        current = target;
    }
    Monitor::parse(&monitor)?;
    if targets.is_empty() {
        return Err("no target presentations".into());
    }
    Ok(Evidence {
        session: session.ok_or("missing presentation session")?,
        monitor,
        targets,
        first_receipt_sequence: receipt.ok_or("no verified in-record optical receipt")?,
    })
}

fn validate_source_identity(frame: &Value, session: &str) -> Result<String> {
    let key = &frame["source_clock"]["source_key"];
    let epoch = string(&key["stream_epoch"])?;
    if epoch.is_empty()
        || key["viewer_session_id"] != session
        || timestamp(&key["sequence"])? != timestamp(&frame["sequence"])?
        || timestamp(&key["sensor_timestamp_ns"])? != timestamp(&frame["timestamp_ns"])?
    {
        return Err("replay source clock is not attested to presentation session/frame".into());
    }
    Ok(epoch.to_owned())
}

fn target_at(targets: &[Presentation], time: i128) -> Option<Target> {
    let i = targets.partition_point(|p| p.time <= time);
    i.checked_sub(1).and_then(|i| targets[i].target)
}
fn acquired(targets: &[Presentation], sensor: i128, bounds: [i128; 2]) -> bool {
    let early = sensor + bounds[0] - 400_000_000;
    let late = sensor + bounds[1];
    target_at(targets, early)
        .zip(target_at(targets, late))
        .is_some_and(|(a, b)| a.site == b.site && early >= a.site_start + 1_000_000_000)
}
fn fresh(case: &Value) -> Option<V3> {
    let gaze = &case["virtual_contact"];
    if case["contact_fresh_source"] != true
        || gaze["sign_resolved"] != true
        || gaze["source_timestamp_ns"] != case["frame"]["timestamp_ns"]
    {
        return None;
    }
    let g = vector(&gaze["relative_gaze_vector"]).ok()?;
    (g[2] > 0.0 && (dot(g, g) - 1.0).abs() < 1e-5).then_some(g)
}
fn percentile_error(
    cases: &[Value],
    eligible: &BTreeSet<u64>,
    targets: &[Presentation],
    monitor: &Monitor,
    offset: i128,
    lag_ms: u64,
) -> Value {
    let mut errors = vec![];
    let mut sites = BTreeMap::<usize, Vec<f64>>::new();
    for c in cases {
        let seq = c["frame"]["sequence"].as_u64().unwrap();
        if !eligible.contains(&seq) {
            continue;
        }
        let Some(gaze) = fresh(c) else {
            continue;
        };
        let sensor = timestamp(&c["frame"]["timestamp_ns"]).unwrap();
        let Some(t) = target_at(targets, sensor + offset - lag_ms as i128 * 1_000_000) else {
            continue;
        };
        if let Some(error) = angle(gaze, monitor.target(t.xy, t.viewport)) {
            errors.push(Some(error));
            sites.entry(t.site).or_default().push(error);
        }
    }
    json!({"lag_ms":lag_ms,"target_discrepancy_degrees":stats(errors),
        "per_site":sites.into_iter().map(|(site,errors)|json!({"site":site,"degrees":stats(errors.into_iter().map(Some))})).collect::<Vec<_>>()})
}

pub(super) fn report(args: &[String]) -> Result<()> {
    if args.len() != 7 {
        return Err("usage: buttercup_report_limbus_refiner targets OUTPUT BASELINE CANDIDATE METADATA.oim1 CLOCK.json BLINK_REVIEW.json".into());
    }
    let output = output_path(Path::new(&args[1]))?;
    let baseline = read(&args[2])?;
    let candidate = read(&args[3])?;
    // Exact frame, crop, clock and independent-motion matching before scoring.
    let mut result = summarize_live_pair(&baseline, &candidate)?;
    let cases = [array(&baseline["cases"])?, array(&candidate["cases"])?];
    let evidence = read_evidence(Path::new(&args[4]))?;
    let monitor = Monitor::parse(&evidence.monitor)?;
    let clock = read(&args[5])?;
    let blink = read(&args[6])?;
    let bounds = [
        timestamp(&clock["robust_empirical_offset_band_ns"][0])?,
        timestamp(&clock["robust_empirical_offset_band_ns"][1])?,
    ];
    if bounds[0] > bounds[1] || bounds[1] - bounds[0] > 1_000_000_000 {
        return Err("invalid/unbounded clock interval".into());
    }
    let midpoint = bounds[0] + (bounds[1] - bounds[0]) / 2;
    let first = &cases[0][0]["frame"];
    if blink["eye"] != first["label"] {
        return Err("blink review is not for this eye".into());
    }
    let mut identities = BTreeSet::new();
    let mut sequence_times = BTreeMap::new();
    let mut stream = None;
    for c in cases[0] {
        let f = &c["frame"];
        let seq = int(&f["sequence"])?;
        let ns = timestamp(&f["timestamp_ns"])?;
        if !identities.insert((seq, ns)) || sequence_times.insert(seq, ns).is_some() {
            return Err("duplicate/ambiguous source identity".into());
        }
        let epoch = validate_source_identity(f, &evidence.session)?;
        if stream.as_ref().is_some_and(|old| old != &epoch) {
            return Err("target evaluation needs one attested sensor clock epoch".into());
        }
        stream = Some(epoch);
    }
    let mut intervals = vec![];
    for pair in array(&blink["core_sequence_intervals"])? {
        let from = *sequence_times
            .get(&int(&pair[0])?)
            .ok_or("blink start is absent from this replay")?;
        let to = *sequence_times
            .get(&int(&pair[1])?)
            .ok_or("blink end is absent from this replay")?;
        if to < from {
            return Err("reversed blink interval".into());
        }
        intervals.push((
            from - int(&blink["pre_padding_ms"])? as i128 * 1_000_000,
            to + int(&blink["post_padding_ms"])? as i128 * 1_000_000,
        ));
    }
    if !sequence_times.contains_key(&evidence.first_receipt_sequence) {
        return Err("first optical receipt source missing from replay".into());
    }
    let mut excluded = [0usize; 3];
    let mut eligible = BTreeSet::new();
    for (&seq, &ns) in &sequence_times {
        if intervals.iter().any(|(lo, hi)| (*lo..=*hi).contains(&ns)) {
            excluded[0] += 1;
            continue;
        }
        if !acquired(&evidence.targets, ns, bounds) {
            excluded[1] += 1;
            continue;
        }
        if seq < evidence.first_receipt_sequence {
            excluded[2] += 1;
            continue;
        }
        eligible.insert(seq);
    }
    let common: BTreeSet<_> = cases[0]
        .iter()
        .zip(cases[1])
        .filter(|(a, b)| fresh(a).is_some() && fresh(b).is_some())
        .map(|(a, _)| a["frame"]["sequence"].as_u64().unwrap())
        .filter(|seq| eligible.contains(seq))
        .collect();
    let mut scores = vec![];
    for (i, arm) in ["baseline", "candidate"].iter().enumerate() {
        scores.push(json!({"arm":arm,"own_admissions":percentile_error(cases[i],&eligible,&evidence.targets,&monitor,midpoint,200),
            "same_source_lag_sweep":([0,100,200,300,400].map(|lag|percentile_error(cases[i],&common,&evidence.targets,&monitor,midpoint,lag))),
            "clock_lower_200ms":percentile_error(cases[i],&common,&evidence.targets,&monitor,bounds[0],200),
            "clock_upper_200ms":percentile_error(cases[i],&common,&evidence.targets,&monitor,bounds[1],200)}));
    }
    result["target_efficacy"] = json!({"exclusions":{"blink":excluded[0],"acquisition":excluded[1],"before_first_optical_receipt":excluded[2]},
        "eligible":eligible.len(),"matched_fresh_signed_sources":common.len(),"scores":scores,
        "viewer_session_id":evidence.session,"stream_epoch":stream,
        "monitor_from_capture":evidence.monitor,"clock":clock,"blink_review":blink,
        "first_receipt_sequence":evidence.first_receipt_sequence,"eligible_sequences":eligible,
        "contract":"conditional angular discrepancy from commanded display target under captured fixed-reference-eye monitor pose; not independently measured ocular direction, fixation or physical pose; no calibration/target feedback; unchanged blind blink list; 1s acquisition after large relocations across entire clock band and 0..400ms lag sweep; tiny steps not excluded; source time only, never held output; 200ms is a sensitivity reference, not fitted latency; imported empirical clock band assumes unit clock rate and is not a confidence interval or fresh optical decoding; correlated frames, not calibrated confidence"});
    result["inputs"] = json!(
        args[2..]
            .iter()
            .map(|path| Ok(json!({"path":path,
        "sha256":format!("{:x}",Sha256::digest(fs::read(path)?))})))
            .collect::<Result<Vec<_>>>()?
    );
    write_json(&output, &result)?;
    println!("{}", result["target_efficacy"]["scores"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gaze_uses_source_time_and_rejects_held_unsigned_or_concave_vectors() {
        let c = json!({"contact_fresh_source":true,"frame":{"timestamp_ns":20},
            "virtual_contact":{"source_timestamp_ns":20,"sign_resolved":true,"relative_gaze_vector":[0.0,0.0,1.0]}});
        assert!(fresh(&c).is_some());
        let mut held = c.clone();
        held["contact_fresh_source"] = json!(false);
        assert!(fresh(&held).is_none());
        for (key, value) in [
            ("source_timestamp_ns", json!(10)),
            ("sign_resolved", json!(false)),
            ("relative_gaze_vector", json!([0, 0, -1])),
        ] {
            let mut invalid = c.clone();
            invalid["virtual_contact"][key] = value;
            assert!(fresh(&invalid).is_none());
        }
    }
    #[test]
    fn target_removal_and_full_clock_band_bound_acquisition() {
        let target = Target {
            xy: [10.0, 10.0],
            viewport: [20.0, 20.0],
            site: 0,
            site_start: 1_000_000_000,
        };
        let stream = [
            Presentation {
                time: 1_000_000_000,
                target: Some(target),
            },
            Presentation {
                time: 3_000_000_000,
                target: None,
            },
        ];
        assert!(!acquired(&stream, 2_000_000_000, [0, 0]));
        assert!(acquired(&stream, 2_500_000_000, [0, 0]));
        assert!(!acquired(&stream, 2_900_000_000, [0, 200_000_000]));
        assert!(target_at(&stream, 3_000_000_000).is_none());
    }
    #[test]
    fn fractional_target_is_not_rounded_and_tilted_monitor_is_retained() {
        let m = Monitor {
            center: [0.0, 0.0, 24.0],
            right: [1.0, 0.0, 0.0],
            down: [0.0, 1.0, 0.0],
            width: 20.0,
            height: 10.0,
        };
        assert_eq!(m.target([50.0, 25.0], [101.0, 51.0]), [0.0, 0.0, 24.0]);
        assert!(angle(m.target([50.25, 25.0], [101.0, 51.0]), [0.0, 0.0, 1.0]).unwrap() > 0.1);
        assert!(Monitor::parse(&json!({"center_inches":[0,0,24],"right_axis":[1,0,0],"down_axis":[1,0,0],"width_inches":20,"height_inches":10})).is_err());
        let tilted = Monitor::parse(&json!({"center_inches":[0,0,24],"right_axis":[0.8,0,0.6],"down_axis":[0,1,0],"width_inches":20,"height_inches":10})).unwrap();
        assert_eq!(
            tilted.target([100.0, 25.0], [101.0, 51.0]),
            [8.0, 0.0, 30.0]
        );
    }
    #[test]
    fn clock_offsets_never_round_through_f64() {
        assert_eq!(
            timestamp(&json!("1690546059193851669")).unwrap(),
            1_690_546_059_193_851_669
        );
        assert!(timestamp(&json!(i128::MAX.to_string())).is_err());
    }

    #[test]
    fn attestation_rejects_wrong_session_sequence_timestamp_and_missing_epoch() {
        let f = json!({"sequence":10,"timestamp_ns":20,"source_clock":{"source_key":{
            "sequence":"10","sensor_timestamp_ns":"20","viewer_session_id":"session-a","stream_epoch":"session-a:1"}}});
        assert!(validate_source_identity(&f, "session-a").is_ok());
        assert!(validate_source_identity(&f, "session-b").is_err());
        for (name, value) in [
            ("sequence", json!(11)),
            ("sensor_timestamp_ns", json!(21)),
            ("stream_epoch", Value::Null),
        ] {
            let mut wrong = f.clone();
            wrong["source_clock"]["source_key"][name] = value;
            assert!(validate_source_identity(&wrong, "session-a").is_err());
        }
    }

    #[test]
    fn native_metadata_rejects_truncation_session_splicing_and_monitor_changes() {
        fn encode(rows: &[Value]) -> Vec<u8> {
            let mut bytes = vec![];
            for row in rows {
                let payload = serde_json::to_vec(row).unwrap();
                let mut header = [0u8; 24];
                header[..4].copy_from_slice(b"OIM1");
                header[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes());
                bytes.extend(header);
                bytes.extend(payload);
            }
            bytes
        }
        let monitor = json!({"center_inches":[0,0,24],"right_axis":[1,0,0],"down_axis":[0,1,0],"width_inches":20,"height_inches":10});
        let rows = [
            json!({"event":"recording_start_snapshot","viewer_session_id":"s","configuration":{"mapping":{"monitor_plane":monitor}}}),
            json!({"event":"presentation","viewer_session_id":"s","host_submit_end_unix_ns":"1000000000","viewport_px":[101,51],"active_targets":[{"role":"relative-motion-stimulus","visible":true,"appearance":{"physical_pixel_position":[50.25,25],"offset_px":[0.25,0]}}],"optical_clock":{"latest_recovery":{"verified_temporal_word":true,"sequence":"10"}}}),
            json!({"event":"presentation","viewer_session_id":"s","host_submit_end_unix_ns":"3000000000","active_targets":[]}),
        ];
        let bytes = encode(&rows);
        let parsed = read_evidence_stream(&bytes[..]).unwrap();
        assert_eq!(parsed.session, "s");
        assert_eq!(parsed.first_receipt_sequence, 10);
        assert_eq!(
            target_at(&parsed.targets, 2_000_000_000).unwrap().xy,
            [50.25, 25.0]
        );
        assert!(target_at(&parsed.targets, 3_000_000_000).is_none());
        for truncated in [1, 23, bytes.len() - 1] {
            assert!(read_evidence_stream(&bytes[..truncated]).is_err());
        }
        let mut mixed = rows.clone();
        mixed[2]["viewer_session_id"] = json!("other");
        assert!(read_evidence_stream(&encode(&mixed)[..]).is_err());
        mixed = rows.clone();
        mixed[2]["configuration"] = rows[0]["configuration"].clone();
        mixed[2]["configuration"]["mapping"]["monitor_plane"]["center_inches"][2] = json!(25);
        assert!(read_evidence_stream(&encode(&mixed)[..]).is_err());
    }
}
