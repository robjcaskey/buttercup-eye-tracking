//! Inventory recorded evidence without treating predicted geometry as truth.
use super::*;
use buttercup_eye_tracking::recorded_bundle::metadata_records;

fn selected(path: &str) -> bool {
    let p = path.to_ascii_lowercase();
    [
        "light",
        "illum",
        "radiom",
        "exposure",
        "gain",
        "intrinsic",
        "extrinsic",
        "clock_model",
        "camera_to_",
        "pose_provenance",
        "pose_uncertainty",
        "optical_center",
        "physical_size",
        "truth",
        "label",
    ]
    .iter()
    .any(|key| p.contains(key))
}

fn visit(
    v: &Value,
    path: &str,
    fields: &mut BTreeMap<String, Value>,
    lights: &mut BTreeMap<String, usize>,
) {
    if selected(path) {
        let entry = fields
            .entry(path.to_owned())
            .or_insert_with(|| json!({"occurrences":0,"nonnull":0,"examples":[]}));
        entry["occurrences"] = json!(entry["occurrences"].as_u64().unwrap() + 1);
        if !v.is_null() {
            entry["nonnull"] = json!(entry["nonnull"].as_u64().unwrap() + 1);
        }
        let encoded = v.to_string();
        if encoded.len() <= 1600 {
            let examples = entry["examples"].as_array_mut().unwrap();
            if examples.len() < 3 && !examples.contains(v) {
                examples.push(v.clone());
            }
        }
    }
    match v {
        Value::Object(map) => {
            for (key, v) in map {
                if key == "lightbox" && v.is_object() {
                    let summary = json!({"enabled":v["enabled"],"pattern":v["pattern"],"effective_pattern":v["effective_pattern"],"schema":v["schema"]});
                    count(lights, summary.to_string());
                }
                visit(v, &format!("{path}/{key}"), fields, lights);
            }
        }
        Value::Array(values) => {
            for v in values {
                visit(v, &format!("{path}/[]"), fields, lights);
            }
        }
        _ => {}
    }
}

pub(crate) fn run(area: &str, fresh: &str, output: &str) -> Result<()> {
    let out = Path::new(output);
    if out.exists() {
        return Err("evidence audit output exists".into());
    }
    let input = lid_circle::admitted_with_context(Path::new(area), fresh, false)?;
    let mut source_counts = BTreeMap::<String, BTreeMap<String, usize>>::new();
    let mut raw_keys = BTreeMap::<String, BTreeSet<String>>::new();
    for row in &input {
        let source = row["raw_source"].as_str().ok_or("RAW source")?.to_owned();
        count(
            source_counts.entry(source.clone()).or_default(),
            row["provider"].as_str().unwrap().to_owned(),
        );
        raw_keys
            .entry(source)
            .or_default()
            .insert(row["raw_sha256"].as_str().unwrap().to_owned());
    }
    fs::create_dir(out)?;
    let mut sources = vec![];
    for (path, counts) in source_counts {
        let bundle = BundleSource::open(Path::new(&path))?;
        let manifest_bytes = bundle.read_entry("manifest.json")?;
        let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
        let metadata = bundle.read_entry("metadata.oim1")?;
        let mut fields = BTreeMap::new();
        let mut lights = BTreeMap::new();
        let mut types = BTreeMap::new();
        let mut records = 0usize;
        let mut first_event = Value::Null;
        for event in metadata_records(metadata.as_slice()) {
            let event = event?;
            if records == 0 {
                first_event = event.clone();
            }
            records += 1;
            let kind = event["event"]
                .as_str()
                .or_else(|| event["type"].as_str())
                .unwrap_or("missing-event-type");
            count(&mut types, kind.to_owned());
            visit(&event, "metadata", &mut fields, &mut lights);
        }
        let frame_bytes = bundle.read_entry("frames.jsonl")?;
        let mut frames = 0usize;
        for line in frame_bytes.split(|&b| b == b'\n').filter(|v| !v.is_empty()) {
            let value: Value = serde_json::from_slice(line)?;
            visit(&value, "frame_index", &mut fields, &mut lights);
            frames += 1;
        }
        visit(&manifest, "manifest", &mut fields, &mut lights);
        let session_path = Path::new(&path).with_extension("session.json");
        let session_bytes = fs::read(&session_path)?;
        let session: Value = serde_json::from_slice(&session_bytes)?;
        visit(&session, "session", &mut fields, &mut lights);
        let light_counts=lights.into_iter().map(|(state,occurrences)|Ok(json!({"state":serde_json::from_str::<Value>(&state)?,"metadata_occurrences_not_exposures":occurrences}))).collect::<Result<Vec<_>>>()?;
        let source = json!({"archive":path,"ambiguous_provider_rows":counts,"unique_ambiguous_raws":raw_keys[&path].len(),"metadata_records":records,"event_types":types,"frame_index_records":frames,"lightbox_states":light_counts,
            "candidate_measurement_fields":fields,"session":{"display_pose":session["display_pose"],"display_size_source":session["display_size_source"],"display_aspect":session["display_aspect"],"segmentation_mode":session["segmentation_mode"],"thumbnail_sample_exclusion":session["thumbnail_sample_exclusion"]},
            "hashes":{"manifest":archive::digest(&manifest_bytes),"metadata":archive::digest(&metadata),"frame_index":archive::digest(&frame_bytes),"session":archive::digest(&session_bytes)},"first_metadata_event":first_event});
        sources.push(source);
    }
    let result = json!({"schema":"buttercup-sclera-measurement-audit-v1","complete":true,"sources":sources,"executable_sha256":archive::digest(&fs::read("/proc/self/exe")?),"retained_inputs_sha256":archive::digest(&fs::read(Path::new(area).join("retained-inputs.jsonl"))?),
        "scope":"Embedded metadata, manifest, frame index and session sidecar of the actual ambiguous-frame sources. Field occurrences include snapshots and repeats; they are not fresh exposures. Keyword discovery does not assert calibration, rank, physical timing or truth. No RAW, labels, camera connection or model is changed."});
    fs::write(out.join("audit.json"), serde_json::to_vec_pretty(&result)?)?;
    println!("{}", out.display());
    Ok(())
}
