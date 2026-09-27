//! Diagnostic near-stationary pairs. Recorded/custom predictions can be
//! compared, but this output is deliberately not an admissible training set.
use super::*;
use buttercup_eye_tracking::{
    recorded_bundle::BundleSource,
    screen_reflection_raw::{estimate_native_frame_translations, PackedRaw10},
};

#[path = "calibration_frame_pairs.rs"]
mod calibration;
pub(super) fn calibration_report(args: &[String]) -> Result<()> {
    calibration::report(args)
}

const MAX_DT_NS: i128 = 150_000_000;
const GOOD_DEGREES: f64 = 1.0;
const BAD_DEGREES: f64 = 4.0;

fn raw(source: &BundleSource, frame: &Value) -> Result<Vec<u8>> {
    Ok(source.read_range(
        string(&frame["stream"])?,
        int(&frame["offset"])?,
        int(&frame["length"])? as usize,
    )?)
}
fn view<'a>(bytes: &'a [u8], f: &Value) -> Result<PackedRaw10<'a>> {
    Ok(PackedRaw10::new(
        bytes,
        int(&f["width"])? as usize,
        int(&f["height"])? as usize,
        int(&f["stride"])? as usize,
        u32::try_from(int(&f["sensor_x"])?)?,
        u32::try_from(int(&f["sensor_y"])?)?,
    )?)
}
fn csv(value: &Value) -> String {
    if value.is_null() {
        return String::new();
    }
    let s = value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string());
    format!("\"{}\"", s.replace('"', "\"\""))
}
fn write_csv(path: &Path, columns: &[&str], rows: &[Value]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    writeln!(file, "{}", columns.join(","))?;
    for row in rows {
        writeln!(
            file,
            "{}",
            columns
                .iter()
                .map(|k| csv(&row[*k]))
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    Ok(())
}
fn pair_sources(a: &Value, b: &Value) -> Result<bool> {
    let dt = timestamp(&b["timestamp_ns"])? - timestamp(&a["timestamp_ns"])?;
    let ka = &a["source_clock"]["source_key"];
    let kb = &b["source_clock"]["source_key"];
    Ok(dt > 0
        && dt <= MAX_DT_NS
        && int(&b["sequence"])? == int(&a["sequence"])? + 1
        && [
            "viewer_session_id",
            "stream_epoch",
            "roi_id",
            "region_session",
        ]
        .iter()
        .all(|k| !ka[*k].is_null() && ka[*k] == kb[*k])
        && ["width", "height", "stride", "pixel_format"]
            .iter()
            .all(|k| a[*k] == b[*k]))
}
fn stable_motion(v: buttercup_eye_tracking::screen_reflection_raw::NativeFrameTranslation) -> bool {
    v.reliable && v.support >= 10 && v.residual <= 1.0 && v.step.0.hypot(v.step.1) <= 1.0
}

pub(super) fn report(args: &[String]) -> Result<()> {
    if args.len() != 6 {
        return Err("usage: buttercup_report_limbus_refiner pairs OUTPUT_DIR REPLAY.json METADATA.oim1 CLOCK.json BLINK_REVIEW.json".into());
    }
    let output = Path::new(&args[1]);
    // Immutable output: failed attempts cannot silently replace prior findings.
    fs::create_dir(output)?;
    let replay = read(&args[2])?;
    let cases = array(&replay["cases"])?;
    if cases.is_empty() {
        return Err("empty replay".into());
    }
    let evidence = read_evidence(Path::new(&args[3]))?;
    let monitor = Monitor::parse(&evidence.monitor)?;
    let clock = read(&args[4])?;
    let blink = read(&args[5])?;
    let bounds = [
        timestamp(&clock["robust_empirical_offset_band_ns"][0])?,
        timestamp(&clock["robust_empirical_offset_band_ns"][1])?,
    ];
    if bounds[0] > bounds[1] || bounds[1] - bounds[0] > 1_000_000_000 {
        return Err("unbounded clock".into());
    }
    let midpoint = bounds[0] + (bounds[1] - bounds[0]) / 2;
    if blink["eye"] != cases[0]["frame"]["label"] {
        return Err("wrong eye blink review".into());
    }
    let mut times = BTreeMap::new();
    let mut epoch = None;
    for c in cases {
        let f = &c["frame"];
        let seq = int(&f["sequence"])?;
        let ns = timestamp(&f["timestamp_ns"])?;
        let e = validate_source_identity(f, &evidence.session)?;
        if epoch.as_ref().is_some_and(|old| old != &e) || times.insert(seq, ns).is_some() {
            return Err("mixed clock/duplicate source".into());
        }
        epoch = Some(e);
    }
    if times
        .values()
        .zip(times.values().skip(1))
        .any(|(a, b)| b <= a)
    {
        return Err("nonmonotonic source time".into());
    }
    let mut intervals = vec![];
    for p in array(&blink["core_sequence_intervals"])? {
        let a = *times.get(&int(&p[0])?).ok_or("blink source missing")?;
        let b = *times.get(&int(&p[1])?).ok_or("blink source missing")?;
        if b < a {
            return Err("reversed blink interval".into());
        }
        intervals.push((
            a - int(&blink["pre_padding_ms"])? as i128 * 1_000_000,
            b + int(&blink["post_padding_ms"])? as i128 * 1_000_000,
        ));
    }
    let bundle = BundleSource::open(Path::new(string(&replay["capture"])?))?;
    let index = bundle.read_entry("frames.jsonl")?;
    let mut native = BTreeMap::new();
    for line in std::str::from_utf8(&index)?
        .lines()
        .filter(|l| !l.trim().is_empty())
    {
        let f: Value = serde_json::from_str(line)?;
        let key = f["source_clock"]["source_key"].to_string();
        if native.insert(key, f).is_some() {
            return Err("duplicate archived source".into());
        }
    }
    let mut frame_rows = vec![];
    let mut pair_rows = vec![];
    let mut samples = BTreeMap::new();
    for c in cases {
        let f = &c["frame"];
        let seq = int(&f["sequence"])?;
        let ns = timestamp(&f["timestamp_ns"])?;
        if native.get(&f["source_clock"]["source_key"].to_string()) != Some(f) {
            return Err("replay frame differs from native archive index".into());
        }
        let mut row = json!({"sequence":seq,"source_timestamp_ns":ns.to_string(),"source_key":f["source_clock"]["source_key"],"frame":f,
            "outer":c["outer"],"virtual_contact":c["virtual_contact"],"raw_admitted":c["raw_admitted"],
            "training_eligible":false,"training_exclusion":"diagnostic recorded/custom solve ranking lacks cold-bootstrap ancestry and independent fixation truth"});
        let reason = if seq < evidence.first_receipt_sequence {
            Some("before-optical-receipt")
        } else if intervals.iter().any(|(a, b)| (*a..=*b).contains(&ns)) {
            Some("blink-padding")
        } else if !acquired(&evidence.targets, ns, bounds) {
            Some("target-acquisition")
        } else if fresh(c).is_none() {
            Some("missing-fresh-signed-solve")
        } else {
            None
        };
        row["exclusion"] = json!(reason);
        if reason.is_none() {
            let g = fresh(c).unwrap();
            let mut errs = vec![];
            let mut site = None;
            // Both ends of empirical clock band and the full predeclared lag sweep.
            for offset in bounds.into_iter().chain([midpoint]) {
                for lag in [0, 100, 200, 300, 400] {
                    let t = target_at(&evidence.targets, ns + offset - lag * 1_000_000)
                        .ok_or("target disappeared")?;
                    if site.is_some_and(|s| s != t.site) {
                        return Err("acquisition gate allowed a site transition".into());
                    }
                    site = Some(t.site);
                    errs.push(angle(g, monitor.target(t.xy, t.viewport)).ok_or("invalid gaze")?);
                }
            }
            let t = target_at(&evidence.targets, ns + midpoint - 200_000_000).unwrap();
            let error = angle(g, monitor.target(t.xy, t.viewport)).unwrap();
            let min = errs.iter().copied().fold(f64::INFINITY, f64::min);
            let max = errs.iter().copied().fold(0.0, f64::max);
            row["error_degrees"] = json!(error);
            row["error_min_degrees"] = json!(min);
            row["error_max_degrees"] = json!(max);
            row["target_site"] = json!(site);
            row["target_pixels"] = json!(t.xy);
            row["target_direction"] = json!(monitor.target(t.xy, t.viewport));
            samples.insert(seq, (c, row.clone()));
        }
        frame_rows.push(row);
    }
    for (&seq, (a, ra)) in &samples {
        let Some((b, rb)) = samples.get(&(seq + 1)) else {
            continue;
        };
        if !pair_sources(&a["frame"], &b["frame"])? || ra["target_site"] != rb["target_site"] {
            continue;
        }
        let ea = ra["error_degrees"].as_f64().unwrap();
        let eb = rb["error_degrees"].as_f64().unwrap();
        let candidate = (ra["error_max_degrees"].as_f64().unwrap() <= GOOD_DEGREES
            && rb["error_min_degrees"].as_f64().unwrap() >= BAD_DEGREES)
            || (rb["error_max_degrees"].as_f64().unwrap() <= GOOD_DEGREES
                && ra["error_min_degrees"].as_f64().unwrap() >= BAD_DEGREES);
        // Evaluate RAW motion for every comparable neighboring pair, not just favorable errors.
        let ba = raw(&bundle, &a["frame"])?;
        let bb = raw(&bundle, &b["frame"])?;
        let va = view(&ba, &a["frame"])?;
        let vb = view(&bb, &b["frame"])?;
        let center = (va.width as f64 * 0.5, va.height as f64 * 0.5);
        let motion = estimate_native_frame_translations(
            &[va, vb],
            center,
            va.height.min(va.width) as f64 * 0.32,
        )[1];
        let still = stable_motion(motion);
        let good = if ea <= eb { ra } else { rb };
        let bad = if ea <= eb { rb } else { ra };
        pair_rows.push(json!({"capture":replay["capture"],"backend":replay["backend"],"label":replay["label"],"sequence_a":seq,"sequence_b":seq+1,
            "source_timestamp_a_ns":ra["source_timestamp_ns"],"source_timestamp_b_ns":rb["source_timestamp_ns"],
            "dt_ms":(timestamp(&b["frame"]["timestamp_ns"])?-timestamp(&a["frame"]["timestamp_ns"])?) as f64/1e6,
            "target_site":ra["target_site"],"error_a_degrees":ea,"error_b_degrees":eb,"good_sequence":good["sequence"],"bad_sequence":bad["sequence"],
            "good_error_max_degrees":good["error_max_degrees"],"bad_error_min_degrees":bad["error_min_degrees"],
            "raw_motion_x_px":motion.step.0,"raw_motion_y_px":motion.step.1,"raw_motion_support":motion.support,"raw_motion_residual_px":motion.residual,
            "raw_motion_reliable":motion.reliable,"near_stationary":still,"good_bad_error_pair":candidate,"selected":still&&candidate,
            "raw_a_sha256":digest(&ba),"raw_b_sha256":digest(&bb),"source_key_a":ra["source_key"],"source_key_b":rb["source_key"],
            "sign_epoch_a":a["virtual_contact"]["sign_epoch"],"sign_epoch_b":b["virtual_contact"]["sign_epoch"],
            "outer_a":a["outer"],"outer_b":b["outer"],"rectified_area_a_px2":a["virtual_contact"]["rectified_area_px2"],"rectified_area_b_px2":b["virtual_contact"]["rectified_area_px2"],
            "independent_scale_a":a["motion_clock"],"independent_scale_b":b["motion_clock"],
            "training_eligible":false,"review_status":"needs-RAW-and-fit-visual-review","training_exclusion":ra["training_exclusion"]}));
    }
    let selected = pair_rows
        .iter()
        .filter(|r| r["selected"] == true)
        .cloned()
        .collect::<Vec<_>>();
    let columns = [
        "capture",
        "backend",
        "label",
        "sequence_a",
        "sequence_b",
        "dt_ms",
        "target_site",
        "good_sequence",
        "bad_sequence",
        "error_a_degrees",
        "error_b_degrees",
        "good_error_max_degrees",
        "bad_error_min_degrees",
        "raw_motion_x_px",
        "raw_motion_y_px",
        "raw_motion_support",
        "raw_motion_residual_px",
        "raw_motion_reliable",
        "near_stationary",
        "good_bad_error_pair",
        "selected",
        "sign_epoch_a",
        "sign_epoch_b",
        "raw_a_sha256",
        "raw_b_sha256",
        "source_key_a",
        "source_key_b",
        "training_eligible",
        "review_status",
        "training_exclusion",
    ];
    write_csv(&output.join("pairs.csv"), &columns, &selected)?;
    write_csv(
        &output.join("error-jump-candidates.csv"),
        &columns,
        &pair_rows
            .iter()
            .filter(|r| r["good_bad_error_pair"] == true)
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    write_csv(&output.join("all-neighbor-pairs.csv"), &columns, &pair_rows)?;
    write_json(&output.join("frames.json"), &json!(frame_rows))?;
    write_json(&output.join("pairs.json"), &json!(pair_rows))?;
    let summary = json!({"schema":"buttercup-diagnostic-limbus-frame-pairs-v1","frames":cases.len(),"eligible_fresh_frames":samples.len(),"neighbor_pairs":pair_rows.len(),
        "near_stationary_pairs":pair_rows.iter().filter(|r|r["near_stationary"]==true).count(),"good_bad_error_pairs":pair_rows.iter().filter(|r|r["good_bad_error_pair"]==true).count(),"selected_pairs":selected.len(),
        "limits":{"max_dt_ms":150,"good_max_degrees":GOOD_DEGREES,"bad_min_degrees":BAD_DEGREES,"max_RAW_integer_translation_px":1.0,"minimum_RAW_support":10,"max_RAW_residual_px":1.0},
        "contract":"diagnostic only; no training eligibility or model promotion; commanded target plus captured fitted monitor is conditional reference, not independently measured fixation or optical axis; strict error bounds cover empirical clock band and 0..400ms lag; native texture motion at integer-pixel resolution with crop origins accounted for; near-stationary candidates still require visual review; no frozen/held solve admission; no new cross-user claim",
        "checkpoint":"879710f","model":replay["model"],"monitor":evidence.monitor,"clock":clock,
        "inputs":args[2..].iter().map(|p|Ok(json!({"path":p,"sha256":digest(&fs::read(p)?)}))).collect::<Result<Vec<_>>>()?,"frame_index_sha256":digest(&index)});
    write_json(&output.join("summary.json"), &summary)?;
    println!(
        "frames={} eligible={} neighbor_pairs={} selected={}",
        cases.len(),
        samples.len(),
        pair_rows.len(),
        selected.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_zero_transport_is_not_stillness() {
        use buttercup_eye_tracking::screen_reflection_raw::NativeFrameTranslation as M;
        let plausible = M {
            support: 15,
            residual: 0.2,
            ..Default::default()
        };
        assert!(!stable_motion(plausible));
        assert!(stable_motion(M {
            reliable: true,
            ..plausible
        }));
        assert!(!stable_motion(M {
            reliable: true,
            step: (2.0, 0.0),
            ..plausible
        }));
    }
    #[test]
    fn cross_epoch_and_dropped_exposure_pairs_are_rejected() {
        let a = json!({"timestamp_ns":10,"sequence":4,"width":420,"height":280,"stride":525,"pixel_format":"RAW10_LE40_1X1","source_clock":{"source_key":{"stream_epoch":"a","roi_id":1,"viewer_session_id":"v","region_session":"r"}}});
        let mut b = a.clone();
        b["timestamp_ns"] = json!(100_000_010);
        b["sequence"] = json!(5);
        assert!(pair_sources(&a, &b).unwrap());
        b["source_clock"]["source_key"]["stream_epoch"] = json!("b");
        assert!(!pair_sources(&a, &b).unwrap());
        b["source_clock"]["source_key"]["stream_epoch"] = json!("a");
        b["sequence"] = json!(6);
        assert!(!pair_sources(&a, &b).unwrap());
    }
}
