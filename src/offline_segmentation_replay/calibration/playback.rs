//! Score fresh native stereo evidence against the stimulus that was actually
//! submitted. This does not invent target visits or simulate a changed live UI.
use super::*;
use std::collections::BTreeSet;

pub(super) fn ns(value: &Value) -> Result<u64, String> {
    value.as_u64().or_else(|| value.as_str()?.parse().ok())
        .ok_or_else(|| format!("missing or invalid clock/integer: {value}"))
}
pub(super) fn pair(value: &Value) -> Option<(f64, f64)> {
    let result = (value[0].as_f64()?, value[1].as_f64()?);
    (result.0.is_finite() && result.1.is_finite()).then_some(result)
}

pub(super) fn current_qualified_source(row: &Value) -> Result<Option<u64>, String> {
    if row["state"] != "Ready" || row["acquisition"] != "Ready" { return Ok(None); }
    let qualified = ns(&row["qualified_source_ns"])?;
    let offered = ns(&row["source_ns"])?;
    if qualified > offered { return Err("qualified gaze precedes its native source arrival".into()); }
    // The other ROI's first completion may still display the previous paired
    // result. Ready describes geometry availability, not a new observation.
    Ok((qualified == offered).then_some(qualified))
}

#[derive(Clone, Debug)]
struct Visit {
    target: usize,
    authority: String,
    start: u64,
    end: u64,
    hidden: Option<u64>,
    timer_start: Option<u64>,
}

fn result_within_visit(visit:&Visit, arrival:u64, available:u64)->bool {
    arrival>=visit.start && arrival<visit.end && available>=arrival && available<visit.end
}

fn coupled_availability(row:&Value, case:&Value)->Result<u64,String> {
    if row["coupled_native_processing"]!=true || row["worker_replay"]!=case["replay"] {
        return Err("worker-only timing cannot authorize native result availability".into());
    }
    let ready=ns(&row["native_ready_elapsed_ns"])?;
    if ready<ns(&case["replay"]["ready_elapsed_ns"])? {return Err("native result preceded worker completion".into());}
    ns(&case["replay"]["first_host_arrival_monotonic_ns"])?
        .checked_add(ready).ok_or("native availability clock overflow".into())
}

fn visit_reached_hold(visit: &Visit, next: Option<&Visit>) -> bool {
    visit.end.saturating_sub(visit.start) >= VIRTUAL_MOUSE_TARGET_HOLD.as_nanos() as u64
        // The first/last submitted buffer can differ from the UI timer by its
        // render/submit duration. A contiguous recorded target advance attests
        // that timer gate under the checked recording recipe. It supplies no
        // gaze sample, sign confidence or cluster to the candidate.
        || next.is_some_and(|next| next.target == visit.target+1 && next.start == visit.end
            && next.authority == visit.authority)
}

fn visits<I>(rows: I, targets: &[(f64, f64)]) -> Result<Vec<Visit>, String>
where I: IntoIterator<Item = Result<Value, String>> {
    let mut configurations = BTreeMap::new();
    let mut result = Vec::<Visit>::new();
    let mut active: Option<usize> = None;
    let mut last_submit = None;
    let mut last_time = 0;
    for row in rows {
        let row = row?;
        if let Ok(at) = ns(&row["host_monotonic_ns"]) { last_time = last_time.max(at); }
        let event = row["event"].as_str().unwrap_or("");
        if matches!(event, "recording_start_snapshot" | "configuration_changed") {
            let config = if event == "configuration_changed" { &row["data"] } else { &row["configuration"] };
            if let Some(revision) = row["configuration_revision"].as_str() {
                let calibration = &config["calibration"];
                configurations.insert(revision.to_owned(), (
                    calibration["phase"].as_str() == Some("collecting"),
                    calibration["target_index"].as_u64(),
                    calibration["thumbnail_opacity"].as_f64(),
                    config["gaze_basis"]["gaze_authority_generation"].to_string(),
                    calibration["first_hidden_submit_elapsed_ns"].clone(),
                ));
            }
        }
        if event != "presentation" { continue; }
        let at = ns(&row["host_submit_end_monotonic_ns"])?;
        if last_submit.is_some_and(|last| at < last) { return Err("presentation clock went backwards".into()); }
        last_submit = Some(at); last_time = last_time.max(at);
        let revision = row["configuration_revision"].as_str().ok_or("presentation lacks configuration revision")?;
        let config = configurations.get(revision).ok_or("presentation configuration not recorded before submission")?;
        let visible = row["active_targets"].as_array().ok_or("missing submitted targets")?.iter()
            .filter(|t| t["visible"] == true && t["role"] == "calibration").collect::<Vec<_>>();
        let current = if config.0 && visible.len() == 1 {
            let target = config.1.ok_or("collecting configuration lacks target index")? as usize;
            let expected = *targets.get(target).ok_or("submitted calibration target index is out of range")?;
            let actual = pair(&visible[0]["normalized"]).ok_or("submitted target lacks coordinates")?;
            if (expected.0-actual.0).abs() > 1e-9 || (expected.1-actual.1).abs() > 1e-9 {
                return Err("submitted target differs from the recorded calibration recipe".into());
            }
            Some((target, config.2.ok_or("thumbnail visibility is not recorded")?, config.3.clone()))
        } else { None };
        let continuing = active.zip(current.as_ref()).is_some_and(|(i, (target, opacity, authority))| {
            let old = &result[i];
            old.target == *target && old.authority == *authority && !(old.hidden.is_some() && *opacity > 0.0)
        });
        if !continuing {
            if let Some(i) = active.take() { result[i].end = at; }
            if let Some((target, _, ref authority)) = current {
                active = Some(result.len());
                result.push(Visit { target, authority: authority.clone(), start: at, end: at, hidden: None, timer_start: None });
            }
        }
        if let (Some(i), Some((_, opacity, _))) = (active, current) {
            if opacity == 0.0 { result[i].hidden.get_or_insert(at); }
            if !config.4.is_null() {
                let hidden = result[i].hidden.ok_or("hidden timer receipt without a hidden submission")?;
                let started = hidden.checked_sub(ns(&config.4)?).ok_or("invalid target timer origin")?;
                if started > result[i].start || result[i].timer_start.is_some_and(|old|old!=started) {
                    return Err("recorded target timer changed inside one visit".into());
                }
                result[i].timer_start = Some(started);
            }
        }
    }
    if let Some(i) = active { result[i].end = last_time; }
    if result.is_empty() { return Err("archive contains no submitted stationary calibration targets".into()); }
    Ok(result)
}

/// Exercise the production collector with externally fixed recorded visits.
/// Stop feeding a visit immediately when the collector advances: the subject
/// continued looking at the recorded cue, not at our counterfactual next cue.
fn collector_replay(timeline: &[Visit], raw_times: &[(u64,u64)],
    qualified: &BTreeMap<u64,(u64,u64,(f64,f64))>, states: &BTreeMap<u64,String>,
    dimensions: (f64,f64), ready_times:Option<&BTreeMap<u64,u64>>) -> Result<Value,String> {
    let origin=Instant::now();
    let at=|ns|origin+Duration::from_nanos(ns);
    let mut samples=vec![Vec::new();VIRTUAL_MOUSE_CALIBRATION_TARGETS.len()];
    let mut authority=None;
    let mut reports=Vec::new();
    let mut completed=vec![false;samples.len()];
    let mut final_fit=Value::Null;
    for (index,visit) in timeline.iter().enumerate() {
        if authority.as_ref().is_some_and(|old|old!=&visit.authority)
            || (visit.target==0 && index>0 && timeline[index-1].target!=0) {
            samples.iter_mut().for_each(Vec::clear);completed.fill(false);final_fit=Value::Null;
        }
        authority=Some(visit.authority.clone());
        samples[visit.target].clear();completed[visit.target]=false;
        let Some(start)=visit.timer_start else {
            reports.push(json!({"visit":index,"target_index":visit.target,"missing_timer_receipt":true}));
            continue;
        };
        let mut mode=VirtualMouseMode::new(at(start));
        mode.target_index=visit.target;
        mode.samples=samples.clone();
        mode.display_dimensions=Some(dimensions);
        // The archive explicitly attests a stationary collecting phase after
        // sign acquisition. No acquisition sources enter these samples.
        mode.calibration_sign_epoch=Some(0);
        mode.target_source_started_ns=raw_times.iter().filter(|(arrival,_)|*arrival<=visit.start)
            .map(|(_,source)|*source).max().or_else(||raw_times.iter()
                .filter(|(arrival,_)|*arrival>visit.start).map(|(_,source)|*source).min());
        let mut hidden_submitted=false;
        let mut advanced_at=None;
        let mut last_source=mode.target_source_started_ns;
        let mut events=Vec::new();
        let mut late=Vec::new();
        for &(arrival,source) in raw_times.iter().filter(|(arrival,_)|*arrival>=visit.start && *arrival<visit.end) {
            let available=match ready_times {
                Some(times)=>match times.get(&source) {Some(&time)=>time,None=>continue},
                None=>arrival,
            };
            if available<arrival {return Err("collector result precedes its recorded RAW arrival".into());}
            if !result_within_visit(visit,arrival,available) {
                if qualified.contains_key(&source) {late.push(source.to_string());}
                continue;
            }
            events.push((available,arrival,source));
        }
        events.sort_unstable();
        let mut offered_results=Vec::new();
        for (available,arrival,source) in events {
            if !hidden_submitted && visit.hidden.is_some_and(|hidden|hidden<=available) {
                let hidden=visit.hidden.unwrap();
                mode.calibration_presented(at(hidden),at(hidden));hidden_submitted=true;
            }
            last_source=if ready_times.is_some() {raw_times.iter().filter(|(received,_)|*received<=available)
                .map(|(_,time)|*time).max()} else {Some(source)};
            mode.sample_source_arrived_at=Some(at(arrival));
            mode.frame_state=match states.get(&source).map(String::as_str) {
                Some("UnresolvedSign")=>CalibrationFrameState::UnresolvedSign,
                Some("Ready")=>CalibrationFrameState::Ready,
                _=>CalibrationFrameState::NoSurface,
            };
            let observation=qualified.get(&source).map(|&(received,_,feature)| {
                assert_eq!(received,arrival,"validated native arrival must remain identical");
                (source,feature,0)
            });
            offered_results.push(json!({"source_ns":source.to_string(),"raw_arrival_ns":arrival.to_string(),
                "result_available_ns":available.to_string(),"qualified":observation.is_some()}));
            mode.observe_at_frame(at(available),last_source,observation);
            if mode.target_index!=visit.target || mode.sequence_completed {
                advanced_at=Some(available);break;
            }
            if mode.calibration_failure.is_some() || mode.sign_acquisition.active() {break;}
        }
        if advanced_at.is_none() && mode.calibration_failure.is_none() && !mode.sign_acquisition.active() {
            mode.sample_source_arrived_at=None;
            // A tick at the recorded exit can advance an existing cluster;
            // never extend the visit to force the hold to pass.
            mode.observe_at_frame(at(visit.end),last_source,None);
            if mode.target_index!=visit.target || mode.sequence_completed {advanced_at=Some(visit.end);}
        }
        completed[visit.target]=advanced_at.is_some();
        samples=mode.samples.clone();
        if mode.sequence_completed {
            final_fit=json!({"native_sequence_completed":true,"all_recorded_visits_completed":completed.iter().all(|v|*v),
                "accepted":mode.display_plane.is_some() && mode.gaze_affine.is_some() && mode.calibration_failure.is_none(),
                "plane":mode.display_plane.map(monitor_location::plane_json),
                "affine":mode.gaze_affine.map(|a|json!({"x":a.x,"y":a.y})),
                "qualified_source_predictions":qualified.iter().map(|(&source,&(_,_,feature))| {
                    let predicted=mode.display_plane.zip(mode.gaze_affine).and_then(|(plane,affine)|
                        RelativeGazeVector::from_projected(feature.0,feature.1)
                            .and_then(|gaze|mode.gaze_affine_input.target(affine,plane,gaze)));
                    (source.to_string(),json!(predicted))
                }).collect::<serde_json::Map<String,Value>>(),
                "mapping_input":mode.gaze_affine_input.label(),"failure":mode.calibration_failure});
        }
        reports.push(json!({"visit":index,"target_index":visit.target,"timer_start_ns":start.to_string(),
            "hidden_submission_accepted":hidden_submitted && mode.target_hidden_elapsed_ns[visit.target].is_some(),
            "advanced_at_ns":advanced_at.map(|v|v.to_string()),"samples":samples[visit.target],
            "offered_results":offered_results,"qualified_sources_ready_after_target_exit":late,
            "failure":mode.calibration_failure,"requested_motion_recovery":mode.sign_acquisition.active()}));
    }
    Ok(json!({"visits":reports,"completed_targets":completed,"final_fit":final_fit,
        "measured_result_availability":ready_times.is_some(),
        "contract":"production VirtualMouseMode collector and finalizer; recorded stationary visits externally imposed; stop at candidate advance; no invented target visits or extra source votes",
        "limitations":[if ready_times.is_some() {"measured coupled result times; recorded stationary visits imposed externally, not a counterfactual closed-loop UI"}
            else {"does not simulate inference latency or closed-loop target timing"},
            "stationary phase initialization follows recorded acquisition; recovery motion itself is not replayed",
            "timer origin inferred from recorded hidden submit offset; trace submission stamp follows the UI callback",
            "target source frontier is the last RAW arrival before first submitted target, not a recorded UI field"]}))
}

/// Describes admitted native geometry, not frame-pair completeness or accuracy.
fn geometry_support_summary(admitted: &[Value]) -> Value {
    let both_contribute=admitted.iter().filter(|row|
        row["contributing_eyes"]==json!([true,true])).count();
    let both_supported=admitted.iter().filter(|row|
        row["contributing_eyes"]==json!([true,true]) && row["direction_supported"]==json!([true,true])).count();
    json!({"admitted_sources":admitted.len(),"both_eyes_contribute":both_contribute,
        "both_eye_directions_supported":both_supported,
        "one_eye_contributes":admitted.len()-both_contribute,
        "contract":"fresh qualified admitted sources only; paired camera delivery is not binocular geometric support; conditional direction support is not measured gaze accuracy"})
}

fn fit(observations: &[((f64, f64), (f64, f64))], dimensions: (f64, f64),
    held_out: Option<((f64,f64),(f64,f64))>) -> Value {
    let coverage = calibration_targets_have_required_coverage(observations.iter().map(|o|o.1));
    let mut plane_diagnostics = gaze_target_solver::DisplayPlaneFitDiagnostics::default();
    let plane = coverage.then(|| gaze_target_solver::fit_virtual_display_plane_diagnosed(
        observations, dimensions, Some(&mut plane_diagnostics))).flatten();
    let partial = plane_diagnostics.best_partial.map(|(candidate, inliers, cost)| {
        let residuals = observations.iter().map(|observation| {
            let error = gaze_target_solver::display_plane_residual(candidate, observation)
                .map(|r| r[0].hypot(r[1]));
            json!({"target":observation.1,"error_screen_fraction":error,
                "inlier":error.is_some_and(|e| e <= gaze_target_solver::VIRTUAL_MOUSE_PLANE_INLIER_RESIDUAL)})
        }).collect::<Vec<_>>();
        json!({"plane":monitor_location::plane_json(candidate),"inliers":inliers,
            "robust_cost":cost,"target_residuals":residuals})
    });
    let mapping = plane.and_then(|p| fit_calibrated_gaze_mapping(p, observations));
    let accepted = plane.zip(mapping).is_some_and(|(p, (a, input))|
        calibration_mapping_has_shared_support(p, a, input, observations));
    let predictions = plane.zip(mapping).map(|(p, (a, input))| observations.iter().map(|&(feature, target)| {
        let predicted = RelativeGazeVector::from_projected(feature.0, feature.1).and_then(|g|input.target(a, p, g));
        json!({"target":target,"predicted":predicted,"error_screen_fraction":predicted.map(|v|(v.0-target.0).hypot(v.1-target.1))})
    }).collect::<Vec<_>>());
    let held_prediction = held_out.zip(plane.zip(mapping)).and_then(|((feature,_),(p,(a,input)))|
        RelativeGazeVector::from_projected(feature.0,feature.1).and_then(|g|input.target(a,p,g)));
    let mapping_basis_diagnostics = [GazeAffineInput::ProjectedDirection, GazeAffineInput::DisplayIntersection]
        .into_iter().map(|input| {
            let candidate = plane.and_then(|p| gaze_target_solver::fit_gaze_mapping_basis(p, observations, input));
            let supported = plane.zip(candidate).is_some_and(|(p,a)|
                calibration_mapping_has_shared_support(p,a,input,observations));
            let prediction = held_out.zip(plane.zip(candidate)).and_then(|((feature,_),(p,a))|
                RelativeGazeVector::from_projected(feature.0,feature.1).and_then(|g|input.target(a,p,g)));
            json!({"mapping_input":input.label(),"fit_available":candidate.is_some(),
                "shared_support":supported,"held_out_prediction":prediction,
                "held_out_error_screen_fraction":held_out.zip(prediction).map(|((_,t),p)|(p.0-t.0).hypot(p.1-t.1)),
                "contract":"same training-only native plane and robust affine; held cue excluded from fitting and selection; unsupported alternatives are diagnostic only"})
        }).collect::<Vec<_>>();
    json!({"accepted":accepted,"coverage":coverage,"stable_targets":observations.len(),
        "mapping_basis_diagnostics":mapping_basis_diagnostics,
        "plane_diagnostics":{"refined_candidates":plane_diagnostics.refined_candidates,
            "best_before_coverage_gate":partial,
            "inlier_threshold_screen_fraction":gaze_target_solver::VIRTUAL_MOUSE_PLANE_INLIER_RESIDUAL,
            "contract":"diagnostic only; ranked by inlier count then robust cost before coverage rejection and final consensus polish; never substitutes for an accepted plane"},
        "plane":plane.map(monitor_location::plane_json),"mapping_input":mapping.map(|(_,input)|input.label()),
        "affine":mapping.map(|(a,_)|json!({"x":a.x,"y":a.y})),"target_residuals":predictions,
        "failure":if !coverage {Some("insufficient stable target coverage")} else if plane.is_none() {
            Some("display pose could not be resolved")} else if !accepted {Some("gaze mapping lacks shared 2D/3D support")} else {None},
        "held_out_target":held_out.map(|(_,t)|t),"held_out_prediction":held_prediction,
        "held_out_error_screen_fraction":held_out.zip(held_prediction).map(|((_,t),p)|(p.0-t.0).hypot(p.1-t.1)),
        "contract":"unchanged native target coverage, display-plane, affine and shared-support gates; residuals are on calibration cues, not independent gaze truth"})
}

/// Persist a fully completed production-collector result in an isolated replay
/// directory, then exercise the actual loader and cursor mapping on its native
/// source stream. This never installs a calibration in the user's settings.
fn reload_check(collector: &Value, eye: usize, cache: &[Value],
    qualified: &BTreeMap<u64,(u64,u64,(f64,f64))>, output: &Path,
    provenance: Value) -> Result<Value,String> {
    let fit=&collector["final_fit"];
    if fit["accepted"]!=true || fit["all_recorded_visits_completed"]!=true {
        return Ok(json!({"performed":false,"reason":"recorded production collector did not complete and accept every target"}));
    }
    if qualified.is_empty() {return Err("accepted calibration lacks qualified replay sources".into());}
    let backend=cache.first().and_then(|r|r["backend"].as_str()).ok_or("missing replay detector")?;
    let mode=SegmentationMode::parse(backend).ok_or("unknown replay detector")?;
    if cache.iter().any(|r|r["backend"]!=backend || r["model"]!=cache[0]["model"]) {
        return Err("cannot persist a calibration across different replay detectors".into());
    }
    let value=json!({"schema":"buttercup-gaze-calibration-v2","eye":eye,"segmentation_mode":mode.label(),
        "plane":fit["plane"],"gaze_affine":{"screen_x":fit["affine"]["x"],
            "screen_y":fit["affine"]["y"],"input":fit["mapping_input"]},
        "training_prompt_generation":null,"training_authority_generation":0,"training_sign_epoch":0});
    let before=CalibratedDisplay::from_json(&value)?;
    let directory=output.with_extension("reload-check");
    fs::create_dir(&directory).map_err(|e|format!("create isolated reload directory: {e}"))?;
    let calibration_path=directory.join("gaze-calibration.json");
    before.save(&calibration_path)?;
    let restored=CalibratedDisplay::load(&calibration_path).ok_or("saved native calibration could not reload")?;
    let mut writer=BufWriter::new(fs::OpenOptions::new().create_new(true).write(true)
        .open(directory.join("mapped-sources.jsonl")).map_err(|e|e.to_string())?);
    let mut maximum_delta=0.0_f64;
    let mut unavailable=0;
    for (&source,&(arrival,sequence,feature)) in qualified {
        let gaze=RelativeGazeVector::from_projected(feature.0,feature.1).ok_or("invalid qualified replay gaze")?;
        let native=fit["qualified_source_predictions"].get(source.to_string())
            .ok_or("native finalizer omitted a qualified source prediction")?;
        let expected=if native.is_null() {None} else {Some(pair(native).ok_or("invalid native finalizer prediction")?)};
        let actual=restored.target(gaze);
        let delta=match (expected,actual) {
            (Some(a),Some(b))=>Some((a.0-b.0).hypot(a.1-b.1)),
            (None,None)=>{unavailable+=1;None},
            _=>return Err("save/reload changed cursor availability".into()),
        };
        if let Some(delta)=delta {
            if !delta.is_finite() || delta>1e-9 {return Err("save/reload changed cursor mapping".into());}
            maximum_delta=maximum_delta.max(delta);
        }
        serde_json::to_writer(&mut writer,&json!({"source_ns":source.to_string(),"sequence":sequence,
            "host_arrival_monotonic_ns":arrival.to_string(),"gaze_feature":feature,
            "before_reload":expected,"after_reload":actual,"difference_screen_fraction":delta}))
            .map_err(|e|e.to_string())?;
        writer.write_all(b"\n").map_err(|e|e.to_string())?;
    }
    writer.flush().map_err(|e|e.to_string())?;
    let receipt=json!({"performed":true,"directory":directory,"qualified_native_sources":qualified.len(),
        "unavailable_cursor_sources":unavailable,"maximum_cursor_difference_screen_fraction":maximum_delta,
        "model":cache[0]["model"],"backend":backend,"provenance":provenance,
        "contract":"native save/load and target mapping; isolated playback artifact, not installed in live settings",
        "limits":["round-trip consistency is not independent gaze accuracy",
            "replay authority and sign epochs use a diagnostic namespace, not live ownership",
            "the native calibration format does not carry camera intrinsics; the companion provenance is required to reproduce its gaze basis"]});
    fs::write(directory.join("provenance.json"),serde_json::to_vec_pretty(&receipt).map_err(|e|e.to_string())?)
        .map_err(|e|e.to_string())?;
    Ok(receipt)
}

/// Bounded input-coordinate ablation. The conic solve, samples and native
/// calibration gates are unchanged; this candidate never replaces a live fit.
fn tangent_mapping_diagnostic(observations: &[((f64,f64),(f64,f64))], dimensions:(f64,f64),
    held_out:Option<((f64,f64),(f64,f64))>) -> Value {
    let tangent=|feature:(f64,f64)| {
        let gaze=RelativeGazeVector::from_projected(feature.0,feature.1)?;
        (gaze.toward_camera>1e-6).then_some((gaze.right/gaze.toward_camera,gaze.down/gaze.toward_camera))
    };
    let transformed=observations.iter().map(|&(feature,target)|tangent(feature).map(|feature|(feature,target)))
        .collect::<Option<Vec<_>>>();
    let coverage=calibration_targets_have_required_coverage(observations.iter().map(|o|o.1));
    let plane=coverage.then(||gaze_target_solver::fit_virtual_display_plane_with_dimensions(observations,dimensions)).flatten();
    let affine=transformed.as_deref().and_then(fit_robust_gaze_affine);
    let predict=|feature:(f64,f64)| {
        let gaze=RelativeGazeVector::from_projected(feature.0,feature.1)?;
        plane?.target(gaze)?;
        let result=affine?.map(tangent(feature)?);
        (result.0.is_finite() && result.1.is_finite()).then_some(result)
    };
    let shared=plane.is_some_and(|p|gaze_target_solver::calibration_predictions_have_shared_support(
        observations.iter().map(|&(f,t)|(t,RelativeGazeVector::from_projected(f.0,f.1).and_then(|g|p.target(g)),predict(f)))));
    let accepted=affine.is_some() && shared;
    let held=held_out.and_then(|(feature,_)|accepted.then(||predict(feature)).flatten());
    json!({"input":"camera-tangent-xy-over-z","accepted":accepted,"coverage":coverage,
        "plane_available":plane.is_some(),"affine_available":affine.is_some(),"shared_support":shared,
        "held_target":held_out.map(|(_,t)|t),"held_prediction":held,
        "held_error_screen_fraction":held.zip(held_out).map(|(p,(_,t))|(p.0-t.0).hypot(p.1-t.1)),
        "contract":"offline diagnostic only; identical fresh source features, native robust affine and shared 2D/3D gates; no change to live mapping selection"})
}

pub(crate) fn run<I: Iterator<Item = String>>(args: I) -> Result<(), String> {
    run_with_availability(args,false)
}

pub(crate) fn run_availability<I: Iterator<Item = String>>(args: I) -> Result<(), String> {
    run_with_availability(args,true)
}

fn run_with_availability<I: Iterator<Item = String>>(mut args: I, coupled:bool) -> Result<(), String> {
    let session_path = PathBuf::from(args.next().ok_or("expected SESSION CACHE BRIDGE NEW_REPORT")?);
    let cache_path = PathBuf::from(args.next().ok_or("missing fresh worker cache")?);
    let bridge_path = PathBuf::from(args.next().ok_or("missing native bridge report")?);
    let output = PathBuf::from(args.next().ok_or("missing new output report")?);
    if args.next().is_some() || output.exists() { return Err("unexpected argument or existing playback report".into()); }
    let allowed = fs::canonicalize("outputs").map_err(|e|e.to_string())?;
    if !fs::canonicalize(output.parent().ok_or("report needs a parent directory")?).map_err(|e|e.to_string())?.starts_with(allowed) {
        return Err("playback output must be beneath outputs".into());
    }
    let session = read_json(&session_path)?;
    if session["recording_complete"] != true { return Err("recording is not finalized".into()); }
    let eye = ns(&session["calibration_eye"])? as usize;
    if eye >= 2 { return Err("invalid calibration eye".into()); }
    let targets = session["targets"].as_array().ok_or("missing target recipe")?.iter()
        .map(|v|pair(v).ok_or("invalid target coordinates")).collect::<Result<Vec<_>,_>>()?;
    if targets != VIRTUAL_MOUSE_CALIBRATION_TARGETS { return Err("target recipe differs from current native coverage policy".into()); }
    if ns(&session["target_hold_ms"])? != VIRTUAL_MOUSE_TARGET_HOLD.as_millis() as u64 {
        return Err("recorded target hold differs from the current native recipe".into());
    }
    let dimensions = pair(&session["display_aspect"]).filter(|&(w,h)|w>0.0 && h>0.0)
        .filter(|_|session["display_size_source"] == "selected-monitor-edid-dtd-mm")
        .unwrap_or_else(gaze_target_solver::nominal_display_dimensions_inches);
    let archive = Path::new(session["raw_bundle"].as_str().ok_or("missing archive")?);
    let bundle = BundleSource::open(archive)?;
    let metadata = bundle.read_entry("metadata.oim1")?;
    let mut recorded_cameras=RecordedCameraInventory::default();
    let timeline = visits(metadata_records(metadata.as_slice()).map(|row| {
        let row=row?;recorded_cameras.observe(&row)?;Ok(row)
    }), &targets)?;
    let frame_bytes = bundle.read_entry("frames.jsonl")?;
    let mut frames = BTreeMap::new();
    let mut lineage = None;
    for line in frame_bytes.split(|&b|b==b'\n').filter(|line|!line.is_empty()) {
        let frame: Value = serde_json::from_slice(line).map_err(|e|e.to_string())?;
        let key = &frame["source_clock"]["source_key"];
        let epoch = key["stream_epoch"].as_str().ok_or("RAW source lacks clock lineage")?;
        if lineage.as_deref().is_some_and(|old|old!=epoch) { return Err("RAW calibration spans different source clocks".into()); }
        lineage = Some(epoch.to_owned());
        let identity = (ns(&frame["eye_id"])?, ns(&frame["timestamp_ns"])?, ns(&frame["sequence"])?);
        if identity != (ns(&key["roi_id"])?, ns(&key["sensor_timestamp_ns"])?, ns(&key["sequence"])?)
            || frames.insert(identity, frame).is_some() { return Err("invalid or duplicate native RAW identity".into()); }
    }
    let clock = joint_gaze_live::clock(lineage.as_deref().ok_or("no native RAW sources")?);
    let cache = read_rows(&cache_path)?;
    if !coupled && cache.iter().any(|row|row["replay"]["timing_validation"]==true) {
        return Err("recorded-stimulus fit currently requires completion-paced evidence; offered-load completion times must not be treated as RAW arrival times".into());
    }
    if coupled && (cache.is_empty() || cache.iter().any(|row|
        row["replay"]["timing_validation"]!=true || row["replay"]["source_schedule"]!="recorded-host-arrival"
            || row["replay"]["first_host_arrival_monotonic_ns"]!=cache[0]["replay"]["first_host_arrival_monotonic_ns"]
            || row["replay"]["first_source_ns"]!=cache[0]["replay"]["first_source_ns"])) {
        return Err("availability replay requires coupled native processing scheduled by recorded host arrivals".into());
    }
    if coupled {
        let origin=ns(&cache[0]["replay"]["first_host_arrival_monotonic_ns"])?;
        let first=ns(&cache[0]["replay"]["first_source_ns"])?;
        if !frames.iter().any(|((_,time,_),frame)|*time==first
            && ns(&frame["source_clock"]["host_arrival_monotonic_ns"]).ok()==Some(origin)) {
            return Err("coupled replay clock anchor is absent from the native recording".into());
        }
    }
    let mut inputs = BTreeMap::new();
    for case in &cache {
        let frame = &case["input"]["frame"];
        let identity = (ns(&frame["eye_id"])?, ns(&frame["timestamp_ns"])?, ns(&frame["sequence"])?);
        if frames.get(&identity) != Some(frame) || ns(&case["timestamp_ns"])? != identity.1
            || ns(&case["sequence"])? != identity.2 || case["input"]["clock_lineage"] != lineage.as_deref().unwrap()
            || inputs.insert(ns(&case["input"]["index"])?, case).is_some() {
            return Err("worker cache is not the exact recorded RAW sequence".into());
        }
    }
    let mut qualified = BTreeMap::<u64, (u64, u64, (f64,f64))>::new();
    let mut availability = BTreeMap::<u64,u64>::new();
    let mut geometry_support = BTreeMap::<u64,([bool;2],[bool;2])>::new();
    let mut completion_times=BTreeMap::<u64,u64>::new();
    let mut source_states = BTreeMap::<u64,String>::new();
    let mut seen_rows = BTreeSet::new();
    let mut mounts = BTreeSet::new();
    for row in read_rows(&bridge_path)? {
        let index = ns(&row["input_index"])?;
        let case = inputs.get(&index).ok_or("bridge row lacks matching worker input")?;
        if !seen_rows.insert(index) || ns(&row["source_ns"])? != ns(&case["timestamp_ns"])?
            || ns(&row["calibration_eye"])? != eye as u64 || row["timing_validation"] != case["replay"]["timing_validation"] {
            return Err("bridge report differs from its worker source/timing receipt".into());
        }
        mounts.insert(row["camera_mount_assumption"].to_string());
        let time = ns(&row["source_ns"])?;
        let available=if coupled {Some(coupled_availability(&row,case)?)} else {None};
        if let Some(available)=available {completion_times.insert(time,available);}
        source_states.insert(time, row["state"].as_str().ok_or("bridge state missing")?.to_owned());
        if current_qualified_source(&row)?.is_none() { continue; }
        let joint = &row["joint"];
        let sources = joint["sources"].as_array().ok_or("ready bridge row lacks joint sources")?;
        if joint["active"] != true || joint["calibration_source_group_complete"] != true
            || joint["contributing_eyes"][eye] != true || joint["posterior"]["direction_supported"][eye] != true
            || sources.len()!=2 || ns(&row["qualified_source_ns"])? != time {
            return Err("ready bridge row lacks current supported paired geometry".into());
        }
        let mut own_frame = None;
        for (i, source) in sources.iter().enumerate() {
            let roi = ns(&source["roi_id"])?; let sequence = ns(&source["sequence"])?;
            if roi != i as u64+1 || ns(&source["sensor_timestamp_ns"])? != time
                || ns(&source["clock_domain"])? != clock.domain || ns(&source["clock_epoch"])? != clock.epoch {
                return Err("qualified joint publication is held or crosses source clocks".into());
            }
            let frame = frames.get(&(roi,time,sequence)).ok_or("qualified joint is absent from native RAW")?;
            if i == eye { own_frame = Some(frame); }
        }
        let frame = own_frame.unwrap();
        let feature = pair(&joint["gaze_directions"][eye]).ok_or("ready source has invalid gaze")?;
        if RelativeGazeVector::from_projected(feature.0,feature.1).is_none() { return Err("invalid camera-facing gaze feature".into()); }
        // The first fully qualified same-source result owns one vote.
        let arrival=ns(&frame["source_clock"]["host_arrival_monotonic_ns"])?;
        if available.is_some_and(|at|at<arrival) {return Err("native result preceded its RAW arrival".into());}
        let contributing: [bool;2]=serde_json::from_value(joint["contributing_eyes"].clone())
            .map_err(|_|"qualified source lacks two explicit eye contribution flags")?;
        let supported: [bool;2]=serde_json::from_value(joint["posterior"]["direction_supported"].clone())
            .map_err(|_|"qualified source lacks two explicit direction support flags")?;
        geometry_support.entry(time).or_insert((contributing,supported));
        availability.entry(time).or_insert(available.unwrap_or(arrival));
        qualified.entry(time).or_insert((arrival,
            ns(&frame["sequence"])?, feature));
    }
    if seen_rows.len()!=inputs.len() || (!coupled && inputs.len()!=frames.len()) || mounts.len()!=1 {
        return Err("playback requires a complete worker/bridge sequence with one mounting policy".into());
    }
    let raw_times = frames.iter().filter(|((roi,_,_),_)|*roi==eye as u64+1).map(|((_,time,_),frame)|
        Ok((ns(&frame["source_clock"]["host_arrival_monotonic_ns"])?, *time)))
        .collect::<Result<Vec<_>,String>>()?;
    let origin = Instant::now();
    let instant = |t|origin+Duration::from_nanos(t);
    let mut owned = BTreeSet::new();
    let mut selected = vec![None;targets.len()];
    let mut selected_authority = None;
    let mut reports = Vec::new();
    for (visit_index, visit) in timeline.iter().enumerate() {
        if selected_authority.as_ref().is_some_and(|old|old!=&visit.authority)
            || (visit.target==0 && timeline[..visit_index].last().is_some_and(|old|old.target!=0)) {
            selected.fill(None);
        }
        selected_authority = Some(visit.authority.clone());
        // A recovery restarts the interrupted target, retaining other targets.
        selected[visit.target] = None;
        let source_start = raw_times.iter().filter(|(arrival,_)|*arrival<=visit.start).map(|(_,time)|*time).max()
            .or_else(||raw_times.iter().filter(|(arrival,_)|*arrival>visit.start).map(|(_,time)|*time).min());
        let mut recent = Vec::new(); let mut first = None; let mut admitted = Vec::new();
        let mut late=Vec::new();
        let mut states = BTreeMap::<String,usize>::new();
        let mut events=raw_times.iter().filter(|(arrival,_)|*arrival>=visit.start && *arrival<visit.end).copied().collect::<Vec<_>>();
        events.sort_by_key(|&(_,source)|availability.get(&source).copied().unwrap_or(u64::MAX));
        for (arrival,time) in events {
            *states.entry(source_states.get(&time).cloned().unwrap_or_else(||"NoBridgeCompletion".into())).or_default()+=1;
            let Some(&(source_arrival, sequence, feature))=qualified.get(&time) else { continue; };
            if source_arrival != arrival { return Err("RAW arrival clock changed in the qualified ledger".into()); }
            let available=availability[&time];
            if !calibration_sample_is_eligible(Duration::from_nanos(available-visit.start), visit.hidden.map(instant),
                Some(instant(arrival)), instant(available), source_start, Some(time)) { continue; }
            if !result_within_visit(visit,arrival,available) {late.push(time.to_string());continue;}
            if !owned.insert(time) { return Err("one source cannot train two recorded target visits".into()); }
            let (contributing,supported)=geometry_support[&time];
            admitted.push(json!({"sequence":sequence,"source_ns":time.to_string(),"feature":feature,
                "contributing_eyes":contributing,"direction_supported":supported,
                "raw_arrival_monotonic_ns":arrival.to_string(),"result_available_monotonic_ns":available.to_string()}));
            recent.push(feature);
            if recent.len()>VIRTUAL_MOUSE_MAX_RECENT_SAMPLES {recent.remove(0);}
            if first.is_none() && available-visit.start>=VIRTUAL_MOUSE_TARGET_HOLD.as_nanos() as u64 {
                if let Some(estimate)=calibration_target_estimate(&recent) { first=Some((time,estimate)); }
            }
        }
        // Live observes UI ticks as well as source arrivals. At a recorded
        // target exit, an existing cluster can satisfy the hold without a new
        // exposure arriving after the timer. Never fabricate that exposure.
        let hold_reached = visit_reached_hold(visit, timeline.get(visit_index+1));
        let decided_on_source = first.is_some();
        if first.is_none() && hold_reached {
            if let Some(estimate) = calibration_target_estimate(&recent) {
                let time = admitted.last().map(|r|ns(&r["source_ns"])).transpose()?;
                first = time.map(|time|(time,estimate));
            }
        }
        selected[visit.target]=first.map(|(_,estimate)|estimate.feature);
        reports.push(json!({"visit":visit_index,"target_index":visit.target,"target":targets[visit.target],
            "host_submit_window_monotonic_ns":[visit.start.to_string(),visit.end.to_string()],
            "first_hidden_submit_monotonic_ns":visit.hidden.map(|v|v.to_string()),
            "source_settle_frontier_ns":source_start.map(|v|v.to_string()),"raw_source_states":states,
            "minimum_hold_attested":hold_reached,
            "cluster_decision":if decided_on_source {"source-arrival"} else {"recorded-visit-exit"},
            "admitted_geometry_support":geometry_support_summary(&admitted),
            "admitted_unique_sources":admitted,
            "qualified_sources_ready_after_target_exit":late,
            "first_stable_cluster":first.map(|(time,e)|json!({"source_ns":time.to_string(),"feature":e.feature,
                "samples":e.total,"inliers":e.inliers,"angular_rms_degrees":e.angular_rms.to_degrees()})),
            "final_cluster":calibration_target_estimate(&recent).map(|e|json!({"feature":e.feature,
                "inliers":e.inliers,"angular_rms_degrees":e.angular_rms.to_degrees()}))}));
    }
    let observations = selected.iter().zip(&targets).filter_map(|(f,t)|f.map(|f|(f,*t))).collect::<Vec<_>>();
    let fit_report = fit(&observations, dimensions, None);
    let held_out = selected.iter().enumerate().map(|(held,feature)| {
        let training=selected.iter().zip(&targets).enumerate().filter(|(i,_)|*i!=held)
            .filter_map(|(_, (f,t))|f.map(|f|(f,*t))).collect::<Vec<_>>();
        json!({"target_index":held,"held_estimate_available":feature.is_some(),
            "result":fit(&training,dimensions,feature.map(|f|(f,targets[held])))})
    }).collect::<Vec<_>>();
    let tangent_held_out=selected.iter().enumerate().map(|(held,feature)| {
        let training=selected.iter().zip(&targets).enumerate().filter(|(i,_)|*i!=held)
            .filter_map(|(_, (f,t))|f.map(|f|(f,*t))).collect::<Vec<_>>();
        json!({"target_index":held,"held_estimate_available":feature.is_some(),
            "result":tangent_mapping_diagnostic(&training,dimensions,feature.map(|f|(f,targets[held])))})
    }).collect::<Vec<_>>();
    for (&source,&available) in &availability {completion_times.insert(source,available);}
    let native_collector=collector_replay(&timeline,&raw_times,&qualified,&source_states,dimensions,
        coupled.then_some(&completion_times))?;
    let camera_rows=read_rows(&bridge_path)?;
    let pupil_ablation=&camera_rows[0]["pupil_evidence_ablation"];
    if !pupil_ablation.is_null() && !matches!(pupil_ablation.as_str(),Some("none"|"omit-measured-pupil"|"half-weight")) {
        return Err("unknown pupil evidence ablation".into());
    }
    if camera_rows.iter().any(|row| &row["pupil_evidence_ablation"]!=pupil_ablation) {
        return Err("pupil evidence ablation changed during replay".into());
    }
    if pupil_ablation=="omit-measured-pupil" && camera_rows.iter().any(|row|
        row["joint"]["arcs"].as_array().is_some_and(|arcs|arcs.iter().any(|arc|
            arc["kind"]=="PupillaryBoundary" && arc["used"]==true))) {
        return Err("pupil ablation contains used pupil boundary evidence".into());
    }
    let reload=reload_check(&native_collector,eye,&cache,&qualified,&output,json!({
        "session":session_path,"cache":cache_path,"bridge":bridge_path,"camera_mount_assumption":mounts,
        "recorded_joint_camera_intrinsics":recorded_cameras.report(),
        "pupil_evidence_ablation":camera_rows[0]["pupil_evidence_ablation"],
        "intrinsics_diagnostic":camera_rows[0]["intrinsics_diagnostic"],
        "radial_camera_diagnostic":camera_rows[0]["radial_camera_diagnostic"],
        "integration_recipe":camera_rows[0]["integration_recipe"],
        "integration_budget_override":camera_rows[0]["integration_budget_override"]}))?;
    let report = json!({"schema":"buttercup-recorded-stimulus-calibration-v1","session":session_path,
        "recorded_joint_camera_intrinsics":recorded_cameras.report(),
        "cache":cache_path,"bridge":bridge_path,"camera_mount_assumption":mounts,
        "native_frames":frames.len(),"selected_eye":eye,"recorded_fit_accepted":session["fit_accepted"],
        "coupled_result_availability":coupled,"worker_completed_roi_results":inputs.len(),
        "availability_contract":if coupled {"actual concurrent CPU worker and native bridge completion; source and result must belong to the recorded stationary visit; no display scanout or closed-loop UI claim"}
            else {"completion-paced geometry replay; RAW arrival used as the diagnostic observation clock"},
        "recorded_sequence_completed":session["sequence_completed"],"visits":reports,
        "selected_target_features":selected,"fit":fit_report,"leave_one_target_out":held_out,
        "native_collector":native_collector,
        "pupil_evidence_ablation":camera_rows[0]["pupil_evidence_ablation"],
        "native_reload_check":reload,
        "camera_tangent_diagnostic":{"fit":tangent_mapping_diagnostic(&observations,dimensions,None),"leave_one_target_out":tangent_held_out},
        "cross_validation_contract":"all samples of the held target excluded from the fit; unchanged native coverage gates; missing estimates or failed training folds remain explicit; fixation-cue assumption, not independent gaze truth",
        "contract":"fixed recorded stimulus; source-bound fresh stereo solves, shared native preview/settle gate, first stable recent cluster and native monitor-fit gates; no target feedback into geometry",
        "limitations":["host RAW arrival and buffer submission are not measured exposure/scanout times",
            if coupled {"measured coupled callback timing includes synchronous diagnostics; no rendering/scanout or changed closed-loop target progression claim"}
                else {"completion-paced offline inference; no claim of live deadline or changed closed-loop target progression"},
            "missing target visits remain missing; accepted offline fit does not complete an interrupted live sequence",
            "Rob-only calibration cues; no independent fixation or human contour truth",
            "source frontier uses last native RAW received by first target submit, or first following RAW if history is absent"]});
    let mut writer=fs::OpenOptions::new().create_new(true).write(true).open(output).map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut writer,&report).map_err(|e|e.to_string())?;
    eprintln!("recorded stimulus replay: {} stable targets, accepted={}",observations.len(),report["fit"]["accepted"]);
    Ok(())
}

fn training_rank(fit: &Value) -> Option<(usize, f64)> {
    if fit["accepted"] != true { return None; }
    let residuals=fit["target_residuals"].as_array()?;
    let squared=residuals.iter().map(|r|r["error_screen_fraction"].as_f64()
        .filter(|e|e.is_finite()).map(|e|e*e)).collect::<Option<Vec<_>>>()?;
    if squared.is_empty() { return None; }
    Some((squared.len(), squared.iter().sum::<f64>()/squared.len() as f64))
}

/// Select a camera trial using training cues only, then report the untouched
/// held cue. The grid must be declared externally; this does not calibrate a
/// physical camera or use held-target error to choose its parameters.
pub(crate) fn compare_camera<I: Iterator<Item=String>>(mut args:I) -> Result<(),String> {
    let output=PathBuf::from(args.next().ok_or("expected NEW_REPORT PLAYBACK_REPORT...")?);
    let allowed=fs::canonicalize("outputs").map_err(|e|e.to_string())?;
    if !fs::canonicalize(output.parent().ok_or("report needs a parent directory")?).map_err(|e|e.to_string())?.starts_with(allowed) {
        return Err("camera comparison must be beneath outputs".into());
    }
    let mut reports=Vec::<Value>::new();let mut trials=Vec::new();
    for path in args {
        let report=read_json(Path::new(&path))?;
        if report["schema"]!="buttercup-recorded-stimulus-calibration-v1" {
            return Err("camera comparison requires native playback reports".into());
        }
        if let Some(first)=reports.first() {
            for key in ["session","cache","camera_mount_assumption","selected_eye","native_frames","pupil_evidence_ablation"] {
                if report[key]!=first[key] {return Err(format!("camera trials differ in {key}"));}
            }
        }
        let rows=read_rows(Path::new(report["bridge"].as_str().ok_or("trial has no bridge")?))?;
        let camera=rows.first().ok_or("empty camera bridge")?["intrinsics_diagnostic"].clone();
        let radial=rows[0]["radial_camera_diagnostic"].clone();
        if !radial.is_null() && !radial.as_f64().is_some_and(|k|k.is_finite() && (0.0..=1.0).contains(&k)) {
            return Err("invalid radial camera trial".into());
        }
        let values: [f64;4]=serde_json::from_value(camera.clone()).map_err(|e|e.to_string())?;
        if !values.into_iter().all(f64::is_finite) || values[0]<=0.0 || values[1]<=0.0
            || rows.iter().any(|r|r["intrinsics_diagnostic"]!=camera || r["radial_camera_diagnostic"]!=radial) {
            return Err("camera trial must attest one fixed finite native-pixel camera".into());
        }
        let mut ready=BTreeSet::new();let mut episodes=0;
        for row in &rows {
            if let Some(source)=current_qualified_source(row)? {ready.insert(source);}
            episodes=episodes.max(row["ui_acquisition_episodes"].as_u64().unwrap_or(0));
            let intrinsics=&row["joint"]["intrinsics"];
            if !intrinsics.is_null() && radial.as_f64().unwrap_or(0.0)==0.0 && (pair(&intrinsics["focal_px"])!=Some((values[0],values[1]))
                || pair(&intrinsics["principal_px"])!=Some((values[2],values[3]))) {
                return Err("published geometry used different intrinsics from its trial".into());
            }
        }
        let held=report["leave_one_target_out"].as_array().ok_or("missing held-target fits")?;
        let errors=held.iter().filter(|r|r["held_estimate_available"]==true && r["result"]["accepted"]==true)
            .filter_map(|r|r["result"]["held_out_error_screen_fraction"].as_f64()).collect();
        trials.push(json!({"report":path,"camera_fx_fy_cx_cy_px":camera,"unique_ready_sources":ready.len(),
            "radial_k1":radial,"radial_contract":"when nonzero: local diagonal radial-lens approximation at measured ROI centers; omits cross-axis shear and within-ROI curvature",
            "ui_acquisition_episodes_max":episodes,"ui_acquisition_regression":episodes>1,
            "accepted":report["fit"]["accepted"],"stable_targets":report["fit"]["stable_targets"],
            "in_sample_rms":training_rank(&report["fit"]).map(|(_,mse)|mse.sqrt()),
            "fixed_camera_held_error_screen_fraction":stats(errors)}));
        reports.push(report);
    }
    if reports.len()<2 {return Err("camera comparison needs at least two fixed-camera trials".into());}
    let mut folds=Vec::new();let mut errors=Vec::new();
    for held in 0..VIRTUAL_MOUSE_CALIBRATION_TARGETS.len() {
        let mut chosen=None;
        for (index,report) in reports.iter().enumerate() {
            let row=&report["leave_one_target_out"][held];
            if row["target_index"].as_u64()!=Some(held as u64) {return Err("camera trial target order differs".into());}
            if let Some((count,mse))=training_rank(&row["result"]) {
                if chosen.is_none_or(|(_,old_count,old_mse)|count>old_count || (count==old_count && mse<old_mse)) {
                    chosen=Some((index,count,mse));
                }
            }
        }
        let outcome=chosen.map(|(index,count,mse)| {
            let row=&reports[index]["leave_one_target_out"][held];
            let error=row["result"]["held_out_error_screen_fraction"].as_f64();
            if let Some(error)=error {errors.push(error);}
            json!({"trial":index,"camera":trials[index]["camera_fx_fy_cx_cy_px"],
                "radial_k1":trials[index]["radial_k1"],
                "training_targets":count,"training_rms":mse.sqrt(),
                "held_estimate_available":row["held_estimate_available"],
                "held_prediction":row["result"]["held_out_prediction"],"held_error_screen_fraction":error})
        });
        folds.push(json!({"held_target_index":held,"selected":outcome}));
    }
    let report=json!({"schema":"buttercup-recorded-camera-comparison-v1","session":reports[0]["session"],
        "trials":trials,"camera_selected_without_held_target":folds,"selected_camera_held_error_screen_fraction":stats(errors),
        "selection":"native accepted training fit; maximize available training targets, then minimize mean squared training residual across all those targets; no held-target values enter selection",
        "limitations":["camera grid is an effective pinhole sensitivity experiment, not uniquely measured physical intrinsics",
            "all candidate geometry uses the same RAW/cache; fixed-stimulus completion-paced replay does not validate live timing",
            "conditional model direction support is not calibrated gaze accuracy; calibration cues are not measured fixation truth",
            "failed coverage/refit folds and missing held observations stay unavailable; no zero errors are imputed",
            "grid selection must also be checked across separate recordings and independent RAW localization/scale evidence"]});
    let mut writer=fs::OpenOptions::new().create_new(true).write(true).open(output).map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut writer,&report).map_err(|e|e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_prefers_supported_perspective_without_using_held_cue() {
        // Latest RAW replay centroids with upper-right excluded from training.
        // Both bases pass; passing consistency alone does not make them equivalent.
        let features = [(-0.3723546403270148,0.6925909078171469),
            (0.120155404272703,0.6824141230133307),
            (-0.13257269472565092,0.6434014752921361),
            (0.046815447363112626,0.5367028317758029),
            (-0.3327979832767164,0.5407557272178635),
            (-0.1415946285316061,0.7182705616476506),
            (0.08262615762247053,0.6253458724129629),
            (-0.1348654916326624,0.5554811374119262)];
        let observations=features.into_iter().zip(VIRTUAL_MOUSE_CALIBRATION_TARGETS)
            .enumerate().filter_map(|(i,v)|(i!=1).then_some(v)).collect::<Vec<_>>();
        let dimensions=(23.228346456692915,13.110236220472444);
        let report=fit(&observations,dimensions,Some((features[1],(0.9,0.1))));
        assert_eq!(report["accepted"],true);
        assert_eq!(report["mapping_input"],"display-intersection");
        for alternative in report["mapping_basis_diagnostics"].as_array().unwrap() {
            assert_eq!(alternative["shared_support"],true);
        }
        let other=fit(&observations,dimensions,Some(((-0.2,0.4),(0.0,1.0))));
        for key in ["plane","affine","mapping_input","accepted"] {
            assert_eq!(report[key],other[key],"held cue must not affect training: {key}");
        }
    }

    #[test]
    fn calibration_plane_diagnostics_do_not_promote_a_rejected_recorded_fit() {
        // Latest recorded candidate, target 8 held out. Derived gaze features,
        // not fixation truth; this regression preserves the production gate.
        let features = [(-0.3723546403270148,0.6925909078171469),
            (0.12391820482972428,0.6691958532225948),
            (-0.13257269472565092,0.6434014752921361),
            (0.046815447363112626,0.5367028317758029),
            (-0.3327979832767164,0.5406718179223261),
            (-0.1415946285316061,0.7182705616476506),
            (0.08262615762247053,0.6253458724129629)];
        let observations=features.into_iter().zip(VIRTUAL_MOUSE_CALIBRATION_TARGETS).collect::<Vec<_>>();
        let dimensions=(23.228346456692915,13.110236220472444);
        assert!(gaze_target_solver::fit_virtual_display_plane_with_dimensions(&observations,dimensions).is_none());
        let report=fit(&observations,dimensions,None);
        assert_eq!(report["accepted"],false);
        assert_eq!(report["plane"],Value::Null);
        assert_eq!(report["coverage"],true);
        for candidate in report["mapping_basis_diagnostics"].as_array().unwrap() {
            assert_eq!(candidate["fit_available"],false);
            assert_eq!(candidate["shared_support"],false);
            assert_eq!(candidate["held_out_prediction"],Value::Null);
        }
        assert!(report["plane_diagnostics"]["refined_candidates"].as_u64().unwrap()>0);
        let partial=&report["plane_diagnostics"]["best_before_coverage_gate"];
        assert!(!partial["plane"].is_null(),"rejected candidate remains diagnostic only");
        assert_eq!(partial["target_residuals"].as_array().unwrap().len(),observations.len());
    }

    #[test]
    fn calibration_geometry_support_distinguishes_one_eye_from_two_supported_eyes() {
        let admitted=vec![
            json!({"contributing_eyes":[false,true],"direction_supported":[false,true]}),
            json!({"contributing_eyes":[true,true],"direction_supported":[false,true]}),
            json!({"contributing_eyes":[true,true],"direction_supported":[true,true]})];
        let report=geometry_support_summary(&admitted);
        assert_eq!(report["admitted_sources"],3);
        assert_eq!(report["one_eye_contributes"],1);
        assert_eq!(report["both_eyes_contribute"],2);
        assert_eq!(report["both_eye_directions_supported"],1);
        assert_eq!(geometry_support_summary(&[])["admitted_sources"],0);
    }

    fn config(revision: usize, phase: &str, target: usize, opacity: f64) -> Value {
        json!({"event":"configuration_changed","configuration_revision":revision.to_string(),
            "data":{"calibration":{"phase":phase,"target_index":target,"thumbnail_opacity":opacity},
            "gaze_basis":{"gaze_authority_generation":"7"}}})
    }
    fn present(revision: usize, at: u64, role: &str, target: (f64,f64)) -> Value {
        json!({"event":"presentation","configuration_revision":revision.to_string(),
            "host_submit_end_monotonic_ns":at.to_string(),
            "active_targets":[{"visible":true,"role":role,"normalized":target}]})
    }
    #[test]
    fn recorded_submission_closes_old_target_and_excludes_recovery() {
        let targets=[(0.1,0.1),(0.9,0.1)];
        let rows=vec![config(1,"collecting",0,1.),present(1,100,"calibration",targets[0]),
            config(2,"collecting",0,0.),present(2,600,"calibration",targets[0]),
            config(3,"sign-acquisition",0,0.),present(3,900,"sign-acquisition",(0.5,0.5)),
            config(4,"collecting",0,1.),present(4,1200,"calibration",targets[0]),
            config(5,"collecting",1,1.),present(5,2000,"calibration",targets[1])];
        let v=visits(rows.into_iter().map(Ok),&targets).unwrap();
        assert_eq!(v.len(),3);assert_eq!((v[0].start,v[0].end,v[0].hidden),(100,900,Some(600)));
        assert_eq!((v[1].start,v[1].end,v[1].hidden),(1200,2000,None));
        assert_eq!(v[2].target,1);
    }
    #[test]
    fn unknown_or_mismatched_stimulus_is_not_fabricated() {
        let target=(0.1,0.1);
        assert!(visits(vec![Ok(present(1,100,"calibration",target))],&[target]).is_err());
        assert!(visits(vec![Ok(config(1,"collecting",0,0.)),Ok(present(1,100,"calibration",(0.2,0.2)))],&[target]).is_err());
    }
    #[test]
    fn measured_result_availability_cannot_move_sources_between_targets() {
        let visit=Visit {target:0,authority:"a".into(),start:100,end:200,hidden:Some(120),timer_start:Some(95)};
        assert!(result_within_visit(&visit,150,199));
        assert!(!result_within_visit(&visit,150,200),"result arrived after cue removal");
        assert!(!result_within_visit(&visit,99,150),"a previous cue's exposure cannot train this target");
        assert!(!result_within_visit(&visit,150,149),"a result cannot precede its native RAW arrival");
        assert!(!result_within_visit(&visit,200,201));
    }
    #[test]
    fn availability_requires_matching_native_completion_and_clock_receipts() {
        let case=json!({"replay":{"ready_elapsed_ns":"30","first_host_arrival_monotonic_ns":"100"}});
        let mut row=json!({"coupled_native_processing":true,"worker_replay":case["replay"],"native_ready_elapsed_ns":"50"});
        assert_eq!(coupled_availability(&row,&case).unwrap(),150);
        row["native_ready_elapsed_ns"]=json!("29");
        assert!(coupled_availability(&row,&case).is_err());
        row["native_ready_elapsed_ns"]=json!("50");row["coupled_native_processing"]=json!(false);
        assert!(coupled_availability(&row,&case).is_err());
        row["coupled_native_processing"]=json!(true);row["worker_replay"]["first_host_arrival_monotonic_ns"]=json!("101");
        assert!(coupled_availability(&row,&case).is_err());
    }
    #[test]
    fn recorded_timer_receipt_is_consistent_and_not_a_fabricated_hold() {
        let target=(0.1,0.1);
        let mut hidden=config(3,"collecting",0,0.);
        hidden["data"]["calibration"]["first_hidden_submit_elapsed_ns"]=json!("510");
        let rows=vec![config(1,"collecting",0,1.),present(1,100,"calibration",target),
            config(2,"collecting",0,0.),present(2,600,"calibration",target),
            hidden.clone(),present(3,700,"calibration",target)];
        assert_eq!(visits(rows.clone().into_iter().map(Ok),&[target]).unwrap()[0].timer_start,Some(90));
        hidden["configuration_revision"]=json!("4");
        hidden["data"]["calibration"]["first_hidden_submit_elapsed_ns"]=json!("511");
        assert!(visits(rows.into_iter().chain([hidden,present(4,800,"calibration",target)]).map(Ok),&[target]).is_err());
    }
    #[test]
    fn native_collector_does_not_extend_a_short_visit_or_count_duplicate_exposures() {
        let visit=Visit {target:0,authority:"a".into(),start:0,end:4_599_000_000,
            hidden:Some(500_000_000),timer_start:Some(0)};
        let raw=(0..8).map(|i|(i*100_000_000+2_200_000_000,i*100_000_000+2_200_000_000))
            .chain([(0,0)]).collect::<Vec<_>>();
        let qualified=raw.iter().map(|&(a,s)|(s,(a,s,(0.1,0.2)))).collect();
        let states=raw.iter().map(|&(_,s)|(s,"Ready".into())).collect();
        let short=collector_replay(&[visit.clone()],&raw,&qualified,&states,(20.,12.),None).unwrap();
        assert_eq!(short["completed_targets"][0],false);
        let full=Visit {end:4_2_200_000_000,..visit};
        let mut repeated=raw.clone();repeated.extend(raw);
        let report=collector_replay(&[full],&repeated,&qualified,&states,(20.,12.),None).unwrap();
        assert_eq!(report["completed_targets"][0],true);
        assert_eq!(report["visits"][0]["samples"].as_array().unwrap().len(),8);
        assert!(report["final_fit"].is_null(),"one target must not complete a calibration");
    }
    #[test]
    fn native_reload_requires_actual_collector_completion_and_source_evidence() {
        let destination=Path::new("outputs/unused-reload-test.json");
        let partial=json!({"final_fit":{"accepted":true,"all_recorded_visits_completed":false}});
        let result=reload_check(&partial,1,&[],&BTreeMap::new(),destination,Value::Null).unwrap();
        assert_eq!(result["performed"],false,"a summary fit cannot authorize a saved completed calibration");
        let empty=json!({"final_fit":{"accepted":true,"all_recorded_visits_completed":true}});
        assert!(reload_check(&empty,1,&[],&BTreeMap::new(),destination,Value::Null).is_err());
    }
    #[test]
    fn measured_collector_excludes_late_results_and_delayed_preview_sources() {
        let visit=Visit {target:0,authority:"a".into(),start:0,end:4_2_200_000_000,
            hidden:Some(500_000_000),timer_start:Some(0)};
        let raw=(0..8).map(|i|(2_200_000_000+i*100_000_000,2_200_000_000+i*100_000_000))
            .chain([(0,0)]).collect::<Vec<_>>();
        let qualified=raw.iter().map(|&(a,s)|(s,(a,s,(0.1,0.2)))).collect();
        let states=raw.iter().map(|&(_,s)|(s,"Ready".into())).collect();
        let times=raw.iter().map(|&(a,s)|(s,if a>=2_700_000_000 {visit.end} else {a+100_000_000})).collect();
        let control=collector_replay(&[visit.clone()],&raw,&qualified,&states,(20.,12.),None).unwrap();
        assert_eq!(control["completed_targets"][0],true);
        let timed=collector_replay(&[visit.clone()],&raw,&qualified,&states,(20.,12.),Some(&times)).unwrap();
        assert_eq!(timed["completed_targets"][0],false);
        assert_eq!(timed["visits"][0]["samples"].as_array().unwrap().len(),5);
        assert_eq!(timed["visits"][0]["qualified_sources_ready_after_target_exit"].as_array().unwrap().len(),3);
        let preview=raw.iter().map(|&(_,s)|(200_000_000,s)).collect::<Vec<_>>();
        let qualified=preview.iter().map(|&(a,s)|(s,(a,s,(0.1,0.2)))).collect();
        let times=preview.iter().map(|&(_,s)|(s,3_100_000_000)).collect();
        let result=collector_replay(&[visit],&preview,&qualified,&states,(20.,12.),Some(&times)).unwrap();
        assert!(result["visits"][0]["samples"].as_array().unwrap().is_empty(),
            "late inference cannot turn preview exposures into fixation samples");
    }
    #[test]
    fn shared_gate_rejects_preview_boundary_and_unsettled_sensor_source() {
        let start=Instant::now();let hidden=start+Duration::from_millis(700);let now=start+Duration::from_secs(3);
        let gate=|arrival,source|calibration_sample_is_eligible(Duration::from_secs(3),Some(hidden),Some(arrival),now,Some(1_000_000_000),Some(source));
        assert!(!gate(hidden,1_800_000_000));
        assert!(!gate(now+Duration::from_millis(1),1_800_000_000));
        assert!(!gate(now,3_099_999_999));
        assert!(gate(now,3_100_000_000));
    }
    #[test]
    fn available_previous_pair_cannot_become_a_fresh_calibration_vote() {
        let mut row=json!({"state":"Ready","acquisition":"Ready","source_ns":"42","qualified_source_ns":"41"});
        assert_eq!(current_qualified_source(&row).unwrap(),None);
        row["qualified_source_ns"]=json!("42");
        assert_eq!(current_qualified_source(&row).unwrap(),Some(42));
        row["qualified_source_ns"]=json!("43");
        assert!(current_qualified_source(&row).is_err());
        row["qualified_source_ns"]=Value::Null;
        assert!(current_qualified_source(&row).is_err());
    }
    #[test]
    fn hold_can_complete_on_a_ui_tick_without_a_new_raw_frame() {
        let duration = VIRTUAL_MOUSE_TARGET_HOLD.as_nanos() as u64;
        let old = Visit {target:0,authority:"same".into(),start:0,end:duration-1_000_000,hidden:Some(500_000_000),timer_start:None};
        let mut next = Visit {target:1,authority:"same".into(),start:old.end,end:duration*2,hidden:None,timer_start:None};
        assert!(visit_reached_hold(&old,Some(&next)),"recorded UI advance attests the hold; no extra RAW sample is created");
        next.target=0;
        assert!(!visit_reached_hold(&old,Some(&next)),"a recovery cannot attest completion");
        assert!(!visit_reached_hold(&old,None),"a truncated short visit cannot attest completion");
        let closed = Visit {end:duration,..old};
        assert!(visit_reached_hold(&closed,None));
    }
    #[test]
    fn camera_selection_ignores_held_error_and_rejects_failed_training_fit() {
        let mut fit=json!({"accepted":true,"target_residuals":[{"error_screen_fraction":0.04}],
            "held_out_error_screen_fraction":0.9});
        let rank=training_rank(&fit);
        fit["held_out_error_screen_fraction"]=json!(0.0);
        assert_eq!(training_rank(&fit),rank);
        fit["accepted"]=json!(false);
        assert!(training_rank(&fit).is_none());
        fit["accepted"]=json!(true);fit["target_residuals"][0]["error_screen_fraction"]=Value::Null;
        assert!(training_rank(&fit).is_none(),"missing training predictions cannot win the grid search");
    }
}
