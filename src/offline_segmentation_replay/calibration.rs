//! Inspect a recorded calibration and prepare exact native RAW for the shared
//! worker replay. Target locations and recorded predictions never seed fits.
use super::*;
use buttercup_eye_tracking::recorded_bundle::{metadata_records, BundleSource};
use std::io::{BufWriter, Write};
pub(super) mod playback;
pub(super) mod geometry;

/// Recorded camera declarations stay distinct from a replay's explicit trial.
#[derive(Default)]
struct RecordedCameraInventory {
    cameras: BTreeMap<String,usize>,
    missing: usize,
}

impl RecordedCameraInventory {
    fn observe(&mut self, row:&Value)->Result<(),String> {
        let config=match row["event"].as_str() {
            Some("recording_start_snapshot")=>&row["configuration"],
            Some("configuration_changed")=>&row["data"],
            _=>return Ok(()),
        };
        let camera=&config["joint_camera_intrinsics"];
        if camera.is_null() {self.missing+=1;return Ok(());}
        if camera["units"]!="native-sensor-pixels" {
            return Err("recorded joint intrinsics must declare native-sensor-pixels".into());
        }
        let focal:[f64;2]=serde_json::from_value(camera["focal_px"].clone())
            .map_err(|_|"invalid recorded focal lengths")?;
        let principal:[f64;2]=serde_json::from_value(camera["principal_px"].clone())
            .map_err(|_|"invalid recorded principal point")?;
        let values=json!([focal[0],focal[1],principal[0],principal[1]]).to_string();
        crate::joint_gaze_live::parse_camera_intrinsics(&values)?;
        *self.cameras.entry(values).or_default()+=1;
        Ok(())
    }

    fn report(&self)->Value {
        json!({"status":if self.cameras.is_empty() {"not-recorded"}
            else if self.cameras.len()>1 {"multiple-recorded-cameras"}
            else if self.missing>0 {"partially-recorded"} else {"one-recorded-camera"},
            "configurations_without_intrinsics":self.missing,
            "cameras":self.cameras.iter().map(|(values,count)|json!({
                "fx_fy_cx_cy":serde_json::from_str::<Value>(values).expect("stored camera JSON"),
                "configuration_records":count})).collect::<Vec<_>>(),
            "units":"native-sensor-pixels",
            "contract":"recorded declarations, not measured physical calibration or replay-selected intrinsics; missing declarations remain unknown, never inferred from the engineering default"})
    }
}

#[cfg(test)]
mod recorded_camera_tests {
    use super::*;
    fn config(fx:f64)->Value {json!({"event":"configuration_changed","data":{
        "joint_camera_intrinsics":{"focal_px":[fx,5712.0],"principal_px":[4000.0,3000.0],
            "units":"native-sensor-pixels"}}})}
    #[test]
    fn calibration_camera_inventory_preserves_unknown_and_multiple_declarations() {
        let mut inventory=RecordedCameraInventory::default();
        inventory.observe(&json!({"event":"recording_start_snapshot","configuration":{}})).unwrap();
        assert_eq!(inventory.report()["status"],"not-recorded");
        inventory.observe(&config(5488.0)).unwrap();
        assert_eq!(inventory.report()["status"],"partially-recorded");
        inventory.observe(&config(5712.0)).unwrap();
        assert_eq!(inventory.report()["status"],"multiple-recorded-cameras");
        assert_eq!(inventory.report()["cameras"].as_array().unwrap().len(),2);
        let mut known=RecordedCameraInventory::default();known.observe(&config(5488.0)).unwrap();
        assert_eq!(known.report()["status"],"one-recorded-camera");
    }
    #[test]
    fn calibration_camera_inventory_rejects_wrong_units_and_invalid_focal_lengths() {
        let mut inventory=RecordedCameraInventory::default();
        let mut row=config(5488.0);row["data"]["joint_camera_intrinsics"]["units"]=json!("model-pixels");
        assert!(inventory.observe(&row).is_err());
        assert!(inventory.observe(&config(0.0)).is_err());
        assert_eq!(inventory.report()["cameras"].as_array().unwrap().len(),0);
    }
}

fn count(counts: &mut BTreeMap<String, usize>, key: &Value) {
    *counts.entry(key.to_string()).or_default() += 1;
}

fn read_json(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}
fn read_rows(path: &Path) -> Result<Vec<Value>, String> {
    fs::read_to_string(path).map_err(|e| e.to_string())?.lines()
        .map(serde_json::from_str).collect::<Result<_, _>>().map_err(|e| e.to_string())
}
fn stats(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    if values.is_empty() { return json!({"n":0}); }
    json!({"n":values.len(),"mean":values.iter().sum::<f64>()/values.len() as f64,
        "median":values[values.len()/2],"p95":values[(values.len()-1)*95/100],"max":values.last()})
}

/// Report existing native evidence; no inference, parameter search or relaxation
/// of calibration coverage. Replay timestamps are never fabricated target slots.
pub(crate) fn summarize<I: Iterator<Item = String>>(mut args: I) -> Result<(), String> {
    let manifest = PathBuf::from(args.next().ok_or("expected MANIFEST.json NEW_OUTPUT.json")?);
    let output = PathBuf::from(args.next().ok_or("missing summary output")?);
    if args.next().is_some() || output.exists() { return Err("unexpected argument or existing summary output".into()); }
    let config = read_json(&manifest)?;
    let path = |key: &str| -> Result<&Path, String> { Ok(Path::new(config[key].as_str().ok_or_else(|| format!("missing {key}"))?)) };
    let audit = read_json(path("audit")?)?;
    let sources = read_rows(path("sources")?)?;
    let motion = read_rows(path("motion")?)?;
    if sources.len() != motion.len() || sources.iter().zip(&motion).any(|(s,m)| *s != m["input"]) {
        return Err("RAW motion is not source-matched to the input index".into());
    }
    let expected: std::collections::BTreeSet<u64> = sources.iter().filter(|s|s["frame"]["eye_id"]==2)
        .map(|s|integer(&s["frame"],"timestamp_ns")).collect::<Result<_, _>>()?;
    let mut replays = Vec::new();
    for replay in config["replays"].as_array().ok_or("missing replay list")? {
        let rows = read_rows(Path::new(replay["path"].as_str().ok_or("missing replay path")?))?;
        if rows.len()!=sources.len() { return Err("calibration replay omitted a source completion".into()); }
        let mut states = BTreeMap::new();
        let mut ready = std::collections::BTreeSet::new();
        let mut publications = BTreeMap::new();
        let mut acquired = None;
        for row in &rows {
            if !row["camera_mount_assumption"].is_null() && row["camera_mount_assumption"]!=replay["camera_mount"] {
                return Err("replay mounting receipt disagrees with comparison manifest".into());
            }
            count(&mut states,&row["state"]);
            if row["acquisition"]=="Ready" && acquired.is_none() { acquired=row["elapsed_ms"].as_u64(); }
            if row["state"]=="Ready" {
                let ns=row["qualified_source_ns"].as_str().ok_or("ready row lacks qualified source")?.parse::<u64>().map_err(|e|e.to_string())?;
                if !expected.contains(&ns) || row["joint"]["calibration_source_group_complete"]!=true
                    || row["joint"]["sources"][1]["sensor_timestamp_ns"]!=row["qualified_source_ns"] {
                    return Err("ready calibration vote lacks exact native paired source".into());
                }
                ready.insert(ns);
            }
            let joint=&row["joint"];
            if joint["sources"].is_array() {
                publications.entry(joint["sources"].to_string()).or_insert(!joint["posterior"].is_null());
            }
        }
        replays.push(json!({"name":replay["name"],"path":replay["path"],"camera_mount":replay["camera_mount"],
            "mount_attested_in_output":rows.iter().all(|r|r["camera_mount_assumption"]==replay["camera_mount"]),
            "completion_rows":rows.len(),"row_states_include_repeated_presentations":states,
            "unique_ready_left_sources":ready.len(),"recorded_left_sources":expected.len(),
            "acquired_at_source_elapsed_ms":acquired,"unique_joint_publications":publications.len(),
            "publications_missing_posterior":publications.values().filter(|v|!**v).count(),
            "ready_source_ns":ready.iter().map(u64::to_string).collect::<Vec<_>>()}));
    }
    let mut detectors = Vec::new();
    for pair in config["detector_pairs"].as_array().ok_or("missing detector pair list")? {
        let caches: [Vec<Value>;2] = ["off","augmented"].map(|arm|
            read_rows(Path::new(pair[arm].as_str().unwrap()))).into_iter().collect::<Result<Vec<_>,_>>()?.try_into().unwrap();
        for cache in &caches {
            if cache.len()!=sources.len() || cache.iter().zip(&sources).any(|(r,s)|r["input"]!=*s) {
                return Err("detector comparison is not the identical native source sequence".into());
            }
        }
        if caches[0][0]["backend"]!=caches[1][0]["backend"] || caches[0][0]["model"]!=caches[1][0]["model"] {
            return Err("refiner comparison changed its detector".into());
        }
        for eye in 1..=2 {
            let mut present=[0;2];let mut admitted=[0;2];let mut refined=[0;2];
            let mut steps: [Vec<f64>;2]=std::array::from_fn(|_|Vec::new());
            let mut previous: Option<(u64,[Option<f64>;2])>=None;
            for (i,source) in sources.iter().enumerate().filter(|(_,s)|s["frame"]["eye_id"]==eye) {
                let radius=std::array::from_fn::<_,2,_>(|arm| {
                    let case=&caches[arm][i];
                    let candidate=case["candidates"].as_array().and_then(|c|c.first());
                    if candidate.is_some() {present[arm]+=1;}
                    if case["limbus_refinement"]["applied"]==true {refined[arm]+=1;}
                    candidate.filter(|c|c["baseline_raw_admitted"]==true).and_then(|c| {
                        admitted[arm]+=1;c["baseline_ellipse"]["major_radius"].as_f64()
                    })
                });
                let current=integer(&source["frame"],"timestamp_ns")?;
                if let Some((previous_ns,old))=previous {
                    if motion[i]["reliable"]==true && motion[i]["from_source_ns"].as_str()==Some(previous_ns.to_string().as_str()) {
                        let m=&motion[i]["motion"];
                        let scale=(1.0+m["diagonal_coefficient_delta"].as_f64().ok_or("motion scale missing")?)
                            .hypot(m["rotation_coefficient"].as_f64().ok_or("motion rotation missing")?);
                        if let ([Some(a),Some(b)],[Some(c),Some(d)])=(old,radius) {
                            steps[0].push((2.0*((c/a).ln()-scale.ln())).abs());
                            steps[1].push((2.0*((d/b).ln()-scale.ln())).abs());
                        }
                    }
                }
                previous=Some((current,radius));
            }
            detectors.push(json!({"name":pair["name"],"backend":caches[0][0]["backend"],"model":caches[0][0]["model"],"roi_id":eye,
                "arms":["refiner-off","refiner-augmented"],"outer_proposals":present,"raw_admitted":admitted,
                "refinement_applied":refined,"matched_sn_feida_abs_log_steps":steps.map(stats),
                "human_contour_labels":0}));
        }
    }
    let stationary = audit["visible_target_role_presentations"]["\"calibration\""].as_u64().unwrap_or(0);
    let result=json!({"schema":"buttercup-calibration-recovery-comparison-v1","manifest":manifest,
        "recording_audit":config["audit"],"native_paired_exposures":expected.len(),"replays":replays,"detectors":detectors,
        "recorded_stationary_target_presentations":stationary,
        "complete_calibration_from_this_archive":if stationary==0 {"unavailable: only sign-acquisition stimulus was shown; no nine-target calibration observations exist"} else {"requires a separate recorded-target admission and native monitor fit"},
        "limits":["single-user clip; no independent fixation, contour truth or calibrated camera/scale",
            "completion-paced masks and source-time bridge replay; no attested end-to-end live latency",
            "exact-source deduplication; no redraws or held predictions counted as new observations",
            "SN-FEIDA uses adjacent RAW motion scale, never candidate radius as its own normalization",
            "mount-conditioned posterior support is defeasible model support, not calibrated gaze accuracy"]});
    fs::write(output,serde_json::to_vec_pretty(&result).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}

pub(crate) fn prepare<I: Iterator<Item = String>>(mut args: I) -> Result<(), String> {
    let session_path = PathBuf::from(args.next().ok_or("expected SESSION.json NEW_OUTPUT_DIR")?);
    let output = PathBuf::from(args.next().ok_or("missing new output directory")?);
    if args.next().is_some() { return Err("unexpected calibration prepare argument".into()); }
    let allowed = fs::canonicalize("outputs").map_err(|e| e.to_string())?;
    let parent = fs::canonicalize(output.parent().ok_or("output needs a parent")?).map_err(|e| e.to_string())?;
    if !parent.starts_with(&allowed) { return Err("calibration output must be beneath outputs".into()); }
    let session: Value = serde_json::from_slice(&fs::read(&session_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if session["recording_complete"] != true { return Err("calibration recording is not finalized".into()); }
    let archive = Path::new(session["raw_bundle"].as_str().ok_or("missing raw bundle")?);
    let bundle = BundleSource::open(archive)?;
    let frame_bytes = bundle.read_entry("frames.jsonl")?;
    let mut frames: Vec<Value> = std::str::from_utf8(&frame_bytes).map_err(|e| e.to_string())?.lines()
        .map(serde_json::from_str).collect::<Result<_, _>>().map_err(|e| e.to_string())?;
    if frames.is_empty() { return Err("no recorded frames".into()); }
    let mut recorded_cameras=RecordedCameraInventory::default();
    let mut mounts = BTreeMap::new();
    let mut solvers = BTreeMap::new();
    let mut phases = BTreeMap::new();
    let mut roles = BTreeMap::new();
    let mut publications = BTreeMap::new();
    let mut scales = BTreeMap::<u64, [Value; 2]>::new();
    let mut scale = [Value::Null, Value::Null];
    let metadata = bundle.read_entry("metadata.oim1")?;
    for row in metadata_records(metadata.as_slice()) {
        let row = row?;
        recorded_cameras.observe(&row)?;
        let event = row["event"].as_str().unwrap_or("");
        if matches!(event, "recording_start_snapshot" | "configuration_changed") {
            let config = if event == "configuration_changed" { &row["data"] } else { &row["configuration"] };
            count(&mut mounts, &config["camera_mount_assumption"]);
            count(&mut solvers, &config["gaze_basis"]["solver"]);
            count(&mut phases, &config["calibration"]["phase"]);
        }
        if event == "presentation" {
            for target in row["active_targets"].as_array().ok_or("missing presentation targets")? {
                if target["visible"] == true { count(&mut roles, &target["role"]); }
            }
        }
        let scene = match event {
            "scene_sample" => Some(&row["data"]["sample"]),
            "recording_start_snapshot" => Some(&row["scene"]),
            _ => None,
        };
        if let Some(scene) = scene {
            for eye in scene["eyes"].as_array().ok_or("missing scene eyes")? {
                let index = eye["roi_id"].as_u64().and_then(|n| n.checked_sub(1)).filter(|n| *n < 2).ok_or("invalid scene ROI")? as usize;
                scale[index] = eye["scale_hint"].clone();
                let joint = &eye["joint_conics"];
                if joint["sources"].is_array() {
                    publications.entry(joint["sources"].to_string()).or_insert_with(|| json!({
                        "sources":joint["sources"],"posterior_present":!joint["posterior"].is_null(),
                        "alternative_cost_margin":joint["alternative_cost_margin"],
                        "gaze_directions":joint["gaze_directions"],"contributing_eyes":joint["contributing_eyes"],
                        "posterior":joint["posterior"]}));
                }
            }
            let time = row["host_unix_ns"].as_str().ok_or("missing scene host timestamp")?.parse::<u64>().map_err(|e| e.to_string())?;
            scales.insert(time, scale.clone());
        }
    }
    // Validate native identities before materializing an index. Never create a
    // fictitious pairing across clock epochs or label a held solve as fresh.
    let mut identities = std::collections::BTreeSet::new();
    let mut lineages = std::collections::BTreeSet::new();
    for frame in &frames {
        let key = &frame["source_clock"]["source_key"];
        let roi = integer(frame, "eye_id")?;
        if !(1..=2).contains(&roi)
            || key["roi_id"].as_u64() != Some(roi)
            || key["sequence"].as_str() != Some(integer(frame, "sequence")?.to_string().as_str())
            || key["sensor_timestamp_ns"].as_str() != Some(integer(frame, "timestamp_ns")?.to_string().as_str())
            || !identities.insert(key.to_string()) {
            return Err("invalid or duplicate native frame identity".into());
        }
        let epoch = key["stream_epoch"].as_str().filter(|s| !s.is_empty()).ok_or("missing native clock epoch")?;
        lineages.insert(epoch.to_owned());
    }
    // A mid-run reacquisition opens a new clock epoch and the live
    // calibration discards every earlier sample, so replaying only the final
    // epoch reproduces what was fitted. Opt-in; never pairs across epochs.
    if lineages.len() > 1 && std::env::var("BUTTERCUP_CALIBRATION_REPLAY_EPOCH").is_ok_and(|v| v == "last") {
        let last_epoch = |f: &Value| f["source_clock"]["source_key"]["stream_epoch"].as_str().map(str::to_owned);
        let latest = frames.iter().max_by_key(|f| f["host_arrival_unix_ns"].as_u64()).and_then(last_epoch);
        frames.retain(|f| last_epoch(f) == latest);
        lineages.retain(|epoch| Some(epoch) == latest.as_ref());
        eprintln!("calibration replay kept only final clock epoch {latest:?} ({} sources)", frames.len());
    }
    if lineages.len() != 1 { return Err("calibration replay requires a single recorded clock epoch (BUTTERCUP_CALIBRATION_REPLAY_EPOCH=last keeps the final one)".into()); }
    frames.sort_by_key(|f| (f["timestamp_ns"].as_u64(), f["eye_id"].as_u64()));
    fs::create_dir(&output).map_err(|e| e.to_string())?;
    let capture = output.join("capture");
    fs::create_dir(&capture).map_err(|e| e.to_string())?;
    for name in ["manifest.json", "frames.jsonl", "subject-left.raw10", "subject-right.raw10"] {
        fs::write(capture.join(name), bundle.read_entry(name)?).map_err(|e| e.to_string())?;
    }
    fs::write(capture.join("metadata.oim1"), metadata).map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(fs::File::create(output.join("sources.jsonl")).map_err(|e| e.to_string())?);
    let mut missing_scale = 0;
    for (index, frame) in frames.iter().enumerate() {
        let eye = integer(frame, "eye_id")? as usize - 1;
        let arrival = integer(frame, "host_arrival_unix_ns")?;
        let scale = scales.range(..=arrival).next_back().map(|(_, s)| s[eye].clone()).unwrap_or(Value::Null);
        if scale.is_null() { missing_scale += 1; }
        let member = frame["stream"].as_str().ok_or("missing RAW stream")?;
        if !matches!(member, "subject-left.raw10" | "subject-right.raw10") { return Err("unexpected native RAW member".into()); }
        let row = json!({"index":index,"capture_entry":0,"frame":frame,
            "clock_lineage":frame["source_clock"]["source_key"]["stream_epoch"],"clock_attested":true,
            "raw_file":fs::canonicalize(capture.join(member)).map_err(|e| e.to_string())?,
            "raw_offset":frame["offset"],"raw_length":frame["length"],"scale_hint":scale,
            "scale_contract":"last recorded coarse scale available by native RAW host arrival; missing stays missing; no candidate-derived scale"});
        serde_json::to_writer(&mut writer, &row).map_err(|e| e.to_string())?;
        writeln!(writer).map_err(|e| e.to_string())?;
    }
    writer.flush().map_err(|e| e.to_string())?;
    let report = json!({"schema":"buttercup-calibration-recording-audit-v1","session":session_path,
        "archive":archive,"recorded_session":session,"native_sources":frames.len(),
        "clock_lineages":lineages,"missing_prior_scale_sources":missing_scale,
        "recorded_joint_camera_intrinsics":recorded_cameras.report(),
        "recorded_camera_mounts":mounts,"recorded_solvers":solvers,"recorded_phases":phases,
        "visible_target_role_presentations":roles,"unique_joint_publications":publications.values().collect::<Vec<_>>(),
        "contract":"archive audit and source preparation only; no inference, target-dependent geometry, label reuse or calibration acceptance"});
    fs::write(output.join("recording-audit.json"), serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    eprintln!("calibration prepared sources={} missing_prior_scale={} output={}", frames.len(), missing_scale, output.display());
    Ok(())
}
