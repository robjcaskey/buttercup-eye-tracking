//! Completed M calibration diagnostics, with exact analysis-source identity and
//! independently rechecked native target/hidden-thumbnail admission. The saved
//! fitted affine is a conditional diagnostic reference, never training truth.
use super::*;
use buttercup_eye_tracking::recorded_bundle::metadata_records;
#[derive(Clone)]
struct Submit {
    time: i128,
    target: Option<usize>,
    uv: Option<[f64; 2]>,
    hidden: bool,
}
fn json_lines(bytes: &[u8]) -> Result<Vec<Value>> {
    std::str::from_utf8(bytes)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
fn mapping(session: &Value, g: [f64; 3]) -> Result<[f64; 2]> {
    let affine = &session["gaze_affine"];
    // Historical sessions without input-space tags used unit direction XY.
    // Refuse explicit unfamiliar input spaces instead of misapplying a mapping.
    for key in ["input_space", "input"] {
        if !affine[key].is_null() && affine[key] != "projected-direction" {
            return Err("unsupported calibration affine input space".into());
        }
    }
    let apply = |v: &Value| -> Result<f64> {
        Ok(v[0].as_f64().ok_or("affine x")? * g[0]
            + v[1].as_f64().ok_or("affine y")? * g[1]
            + v[2].as_f64().ok_or("affine constant")?)
    };
    let p = [apply(&affine["screen_x"])?, apply(&affine["screen_y"])?];
    if p.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite calibration mapping".into());
    }
    Ok(p)
}
fn submission_allows(submit: &Submit, target: usize, uv: [f64; 2], arrival: i128) -> bool {
    submit.time < arrival && submit.target == Some(target) && submit.uv == Some(uv) && submit.hidden
}
pub(super) fn report(args: &[String]) -> Result<()> {
    if args.len() < 3 {
        return Err("usage: buttercup_report_limbus_refiner calibration-pairs NEW_OUTPUT_DIR SESSION.json...".into());
    }
    let out = Path::new(&args[1]);
    fs::create_dir(out)?;
    let mut all = vec![];
    let mut source_rows = vec![];
    let mut summaries = vec![];
    for path in &args[2..] {
        let session = read(path)?;
        if session["fit_accepted"] != true || session["sequence_completed"] != true {
            return Err(format!("{path}: requires a completed accepted calibration").into());
        }
        let calibration_eye = int(&session["calibration_eye"])? + 1;
        let archive = string(&session["raw_bundle"])?;
        let bundle = BundleSource::open(Path::new(archive))?;
        let frame_bytes = bundle.read_entry("frames.jsonl")?;
        let frames = json_lines(&frame_bytes)?;
        let mut by_key = BTreeMap::new();
        for f in &frames {
            let k = &f["source_clock"]["source_key"];
            validate_source_identity(f, string(&k["viewer_session_id"])?)?;
            if timestamp(&k["roi_id"])? != timestamp(&f["eye_id"])? {
                return Err("RAW eye identity differs from its source key".into());
            }
            if by_key.insert(k.to_string(), f).is_some() {
                return Err("duplicate native source key".into());
            }
        }
        let mut configs = BTreeMap::<(String, String), Value>::new();
        let mut submits = vec![];
        let mut scenes = BTreeMap::new();
        let metadata = bundle.read_entry("metadata.oim1")?;
        for r in metadata_records(metadata.as_slice()) {
            let r = r?;
            let id = (
                r["viewer_session_id"].to_string(),
                r["configuration_revision"].to_string(),
            );
            match r["event"].as_str().unwrap_or("") {
                "recording_start_snapshot" | "queue_recovery_snapshot" | "live_checkpoint" => {
                    configs.insert(id, r["configuration"].clone());
                }
                "configuration_changed" => {
                    configs.insert(id, r["data"].clone());
                }
                "presentation" => {
                    let config = configs
                        .get(&id)
                        .ok_or("presentation configuration unavailable")?;
                    let state = &config["calibration"];
                    let time = timestamp(&r["host_submit_end_unix_ns"])?;
                    if submits.last().is_some_and(|s: &Submit| time < s.time) {
                        return Err("nonmonotonic presentation clock".into());
                    }
                    let target = state["target_index"].as_u64().map(|v| v as usize);
                    let visible = array(&r["active_targets"])?
                        .iter()
                        .filter(|t| t["visible"] == true && t["role"] == "calibration")
                        .collect::<Vec<_>>();
                    let uv = if visible.len() == 1 {
                        pair(&visible[0]["normalized"]).ok()
                    } else {
                        None
                    };
                    let hidden = r["mode"] == "mouse-calibration"
                        && state["phase"] == "collecting"
                        && state["thumbnail_opacity"].as_f64() == Some(0.0)
                        && state["sample_gate_has_hidden_submission"] == true;
                    submits.push(Submit {
                        time,
                        target,
                        uv,
                        hidden,
                    });
                }
                "scene_sample" => {
                    for eye in array(&r["data"]["sample"]["eyes"])? {
                        if eye["roi_id"].as_u64() != Some(calibration_eye)
                            || eye["held_geometry"] != false
                            || eye["basis"]["sign_resolved"] != true
                        {
                            continue;
                        }
                        if timestamp(&eye["basis"]["sign_epoch"]).ok()
                            != timestamp(&session["sign_epoch"]).ok()
                        {
                            continue;
                        }
                        let Some(g) = vector(&eye["admitted_contact_axis"]).ok() else {
                            continue;
                        };
                        if g[2] <= 0.0 || (dot(g, g) - 1.).abs() > 1e-5 {
                            continue;
                        }
                        let key = &eye["analysis_source"]["key"];
                        if eye["analysis_source"]["status"] != "exact-source-key"
                            || !by_key.contains_key(&key.to_string())
                        {
                            continue;
                        }
                        scenes.entry(key.to_string()).or_insert((g, eye.clone()));
                    }
                }
                _ => {}
            }
        }
        let monitor = Monitor::parse(&session["display_pose"])?;
        let targets = array(&session["target_statistics"])?;
        for t in targets {
            let lo = timestamp(&t["source_window_ns"][0])?;
            let hi = timestamp(&t["source_window_ns"][1])?;
            if hi < lo {
                return Err("reversed completed calibration source window".into());
            }
        }
        let mut samples = BTreeMap::new();
        let mut excluded = BTreeMap::<&str, usize>::new();
        for (key, (g, eye)) in scenes {
            let f = by_key[&key];
            let ns = timestamp(&f["timestamp_ns"])?;
            let owned = targets
                .iter()
                .filter(|t| {
                    timestamp(&t["source_window_ns"][0])
                        .ok()
                        .zip(timestamp(&t["source_window_ns"][1]).ok())
                        .is_some_and(|(lo, hi)| ns >= lo + 500_000_000 && ns <= hi)
                })
                .collect::<Vec<_>>();
            if owned.len() != 1 {
                *excluded
                    .entry("outside-unique-settled-target-window")
                    .or_default() += 1;
                continue;
            }
            let t = owned[0];
            let target = int(&t["target_index"])? as usize;
            let uv = pair(&t["target"])?;
            let arrival = timestamp(&f["source_clock"]["host_arrival_unix_ns"])?;
            let i = submits.partition_point(|s| s.time < arrival);
            if i == 0 || !submission_allows(&submits[i - 1], target, uv, arrival) {
                *excluded
                    .entry("target-or-hidden-submit-not-attested")
                    .or_default() += 1;
                continue;
            }
            let xy = mapping(&session, g)?;
            let angle =
                angle(monitor.target_uv(xy), monitor.target_uv(uv)).ok_or("mapped target angle")?;
            let screen = (xy[0] - uv[0]).hypot(xy[1] - uv[1]);
            let seq = int(&f["sequence"])?;
            if samples.insert(seq, (f, angle, screen, target)).is_some() {
                return Err("duplicate analysis for native source".into());
            }
            source_rows.push(json!({"archive":archive,"session":path,"frame":f,"eye":eye,"mapped_uv":xy,"target_uv":uv,"error_degrees":angle,"error_screen_fraction":screen,"target_index":target,
                "reference":"same-session saved affine; conditional within-calibration diagnostic, not out-of-sample fixation truth","training_eligible":false}));
        }
        let mut pairs = 0;
        let mut selected = 0;
        for (&seq, &(a, ea, sa, site)) in &samples {
            let Some(&(b, eb, sb, nextsite)) = samples.get(&(seq + 1)) else {
                continue;
            };
            if site != nextsite || !pair_sources(a, b)? {
                continue;
            }
            let bytes_a = raw(&bundle, a)?;
            let bytes_b = raw(&bundle, b)?;
            let va = view(&bytes_a, a)?;
            let vb = view(&bytes_b, b)?;
            let motion = estimate_native_frame_translations(
                &[va, vb],
                (va.width as f64 * 0.5, va.height as f64 * 0.5),
                va.width.min(va.height) as f64 * 0.32,
            )[1];
            let error_pair = ea.min(eb) <= GOOD_DEGREES && ea.max(eb) >= BAD_DEGREES;
            let still = stable_motion(motion);
            pairs += 1;
            selected += usize::from(still && error_pair);
            all.push(json!({"capture":archive,"session":path,"label":a["label"],"sequence_a":seq,"sequence_b":seq+1,
                "source_key_a":a["source_clock"]["source_key"],"source_key_b":b["source_clock"]["source_key"],"target_index":site,
                "error_a_degrees":ea,"error_b_degrees":eb,"error_a_screen_fraction":sa,"error_b_screen_fraction":sb,
                "good_sequence":if ea<eb{seq}else{seq+1},"bad_sequence":if ea<eb{seq+1}else{seq},"good_bad_error_pair":error_pair,"near_stationary":still,"selected":still&&error_pair,
                "dt_ms":(timestamp(&b["timestamp_ns"])?-timestamp(&a["timestamp_ns"])?) as f64/1e6,
                "raw_motion_x_px":motion.step.0,"raw_motion_y_px":motion.step.1,"raw_motion_support":motion.support,"raw_motion_residual_px":motion.residual,"raw_motion_reliable":motion.reliable,
                "raw_a_sha256":digest(&bytes_a),"raw_b_sha256":digest(&bytes_b),"training_eligible":false,
                "reference":"saved same-session affine; commanded target, not independent fixation/held-out accuracy","clock_limit":"RAW host arrival and buffer submission; unknown optical transport and scanout latency","review_status":"needs-RAW-and-fit-review"}));
        }
        summaries.push(json!({"session":path,"session_sha256":digest(&fs::read(path)?),"archive":archive,"frame_index_sha256":digest(&frame_bytes),"metadata_sha256":digest(&metadata),"fresh_eligible_sources":samples.len(),"neighbor_pairs":pairs,"selected_pairs":selected,"excluded":excluded}));
    }
    let columns = [
        "capture",
        "session",
        "label",
        "sequence_a",
        "sequence_b",
        "target_index",
        "dt_ms",
        "error_a_degrees",
        "error_b_degrees",
        "error_a_screen_fraction",
        "error_b_screen_fraction",
        "good_sequence",
        "bad_sequence",
        "raw_motion_x_px",
        "raw_motion_y_px",
        "raw_motion_support",
        "raw_motion_residual_px",
        "raw_motion_reliable",
        "near_stationary",
        "good_bad_error_pair",
        "selected",
        "raw_a_sha256",
        "raw_b_sha256",
        "source_key_a",
        "source_key_b",
        "training_eligible",
        "reference",
        "clock_limit",
        "review_status",
    ];
    write_csv(&out.join("all-neighbor-pairs.csv"), &columns, &all)?;
    write_csv(
        &out.join("pairs.csv"),
        &columns,
        &all.iter()
            .filter(|r| r["selected"] == true)
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    write_csv(
        &out.join("error-jump-candidates.csv"),
        &columns,
        &all.iter()
            .filter(|r| r["good_bad_error_pair"] == true)
            .cloned()
            .collect::<Vec<_>>(),
    )?;
    write_json(&out.join("frames.json"), &json!(source_rows))?;
    write_json(&out.join("pairs.json"), &json!(all))?;
    write_json(
        &out.join("summary.json"),
        &json!({"schema":"buttercup-completed-calibration-pairs-v1","captures":summaries,"selected_pairs":all.iter().filter(|r|r["selected"]==true).count(),"error_jump_candidates":all.iter().filter(|r|r["good_bad_error_pair"]==true).count(),"training_eligible":false}),
    )?;
    println!(
        "calibration neighbor pairs={} selected={} error-jump candidates={}",
        all.len(),
        all.iter().filter(|r| r["selected"] == true).count(),
        all.iter()
            .filter(|r| r["good_bad_error_pair"] == true)
            .count()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn visible_thumbnails_or_other_target_cannot_own_an_exposure() {
        let s = Submit {
            time: 100,
            target: Some(2),
            uv: Some([0.5, 0.5]),
            hidden: true,
        };
        assert!(submission_allows(&s, 2, [0.5, 0.5], 101));
        assert!(!submission_allows(&s, 2, [0.5, 0.5], 100));
        assert!(!submission_allows(&s, 1, [0.5, 0.5], 101));
        assert!(!submission_allows(
            &Submit { hidden: false, ..s },
            2,
            [0.5, 0.5],
            101
        ));
    }
    #[test]
    fn unfamiliar_affine_space_is_not_treated_as_direction_xy() {
        let s = json!({"gaze_affine":{"input_space":"display-intersection","screen_x":[1,0,0],"screen_y":[0,1,0]}});
        assert!(mapping(&s, [0.2, 0.3, 0.9]).is_err());
    }
}
