//! Native RAW review preparation. No inferred labels or rejected-frame context.
use super::*;

pub(crate) fn prepare(area_dir: &str, fresh_dir: &str, name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err("review name must contain lowercase letters, digits and hyphens".into());
    }
    let area = Path::new(area_dir);
    let input = lid_circle::admitted_with_context(area, fresh_dir, true)?;
    let classes = rows(&area.join("classifications.jsonl"))?
        .into_iter()
        .map(|r| {
            (
                (n(&r["record"]), r["provider"].as_str().unwrap().to_owned()),
                r["class"].clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let originals = rows(&Path::new(fresh_dir).join("frames.jsonl"))?
        .into_iter()
        .map(|r| (n(&r["record"]), r))
        .collect::<BTreeMap<_, _>>();
    let mut unique = BTreeMap::<u64, (Value, BTreeSet<String>, bool)>::new();
    for row in input {
        let record = n(&row["record"]);
        let provider = row["provider"].as_str().unwrap().to_owned();
        for field in [
            "raw_sha256",
            "frame",
            "source_ns",
            "source",
            "epoch",
            "eye",
            "raw_source",
            "stream_entry",
        ] {
            if row[field] != originals[&record][field] {
                return Err(format!("lid review identity mismatch: {field}").into());
            }
        }
        let ambiguous = classes[&(record, provider.clone())] == "multiple";
        let entry = unique
            .entry(record)
            .or_insert((row, BTreeSet::new(), false));
        entry.1.insert(provider);
        entry.2 |= ambiguous;
    }
    let source_paths = unique
        .values()
        .map(|(r, _, _)| r["raw_source"].as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    if source_paths.len() != 1 {
        return Err(
            "review preparation currently requires one capture; split captures explicitly".into(),
        );
    }
    let source = Path::new(source_paths.iter().next().unwrap());
    let capture = if source.extension().is_some_and(|e| e == "tar") {
        source.with_extension("")
    } else {
        source.to_path_buf()
    };
    let root = capture.join("annotator").join(name);
    if root.exists() {
        return Err("review set exists".into());
    }
    let mut groups = BTreeMap::<_, Vec<_>>::new();
    for (record, (row, providers, ambiguous)) in unique {
        groups
            .entry((n(&row["source"]), n(&row["epoch"]), n(&row["eye"])))
            .or_default()
            .push((record, row, providers, ambiguous));
    }
    let mut targets = vec![];
    let mut eligible_counts = BTreeMap::new();
    for (key, mut group) in groups {
        group.sort_by_key(|(_, r, _, _)| r["source_ns"].as_str().unwrap().parse::<u64>().unwrap());
        let mut eligible = vec![];
        for i in 1..group.len().saturating_sub(1) {
            if !group[i].3 {
                continue;
            }
            let ns = |j: usize| {
                group[j].1["source_ns"]
                    .as_str()
                    .unwrap()
                    .parse::<u64>()
                    .unwrap()
            };
            if ns(i) <= ns(i - 1)
                || ns(i + 1) <= ns(i)
                || ns(i) - ns(i - 1) > 300_000_000
                || ns(i + 1) - ns(i) > 300_000_000
            {
                continue;
            }
            let common = group[i - 1]
                .2
                .intersection(&group[i].2)
                .filter(|p| group[i + 1].2.contains(*p))
                .cloned()
                .collect::<Vec<_>>();
            if !common.is_empty() {
                eligible.push((i, common));
            }
        }
        eligible_counts.insert(format!("{}:{}:{}", key.0, key.1, key.2), eligible.len());
        let count = eligible.len().min(6);
        for pick in 0..count {
            let (i, common) = &eligible[(2 * pick + 1) * eligible.len() / (2 * count)];
            targets.push((
                group[i - 1].1.clone(),
                group[*i].1.clone(),
                group[i + 1].1.clone(),
                common.clone(),
            ));
        }
    }
    if targets.is_empty() {
        return Err("no admitted before/target/after review windows".into());
    }
    let archive = root.join("archive");
    fs::create_dir_all(archive.join("context"))?;
    let labels = capture.join("annotator/labels");
    fs::create_dir_all(&labels)?;
    let bundle = BundleSource::open(source)?;
    let mut receipt = vec![];
    for (before, target, after, providers) in targets {
        let target_ns = target["source_ns"].as_str().unwrap().parse::<u64>()?;
        let stem = format!(
            "eye-{}-seq-{}-raw-{}",
            target["eye"],
            target["sequence"],
            &target["raw_sha256"].as_str().unwrap()[..12]
        );
        let mut context = vec![];
        for (i, row) in [&before, &target, &after].iter().enumerate() {
            let f = &row["frame"];
            let file = if i == 1 {
                format!("{stem}.raw10")
            } else {
                format!(
                    "context/record-{}-{}.raw10",
                    row["record"],
                    &row["raw_sha256"].as_str().unwrap()[..12]
                )
            };
            let bytes = bundle.read_range(
                row["stream_entry"].as_str().ok_or("stream")?,
                n(&f["offset"]),
                n(&f["length"]) as usize,
            )?;
            if archive::digest(&bytes) != row["raw_sha256"] {
                return Err("review RAW hash mismatch".into());
            }
            // Check native packing/shape without changing any recorded byte.
            raw10::try_unpack_raw10(
                &bytes,
                n(&f["width"]) as usize,
                n(&f["height"]) as usize,
                n(&f["stride"]) as usize,
            )?;
            let path = archive.join(&file);
            if path.exists() {
                if fs::read(&path)? != bytes {
                    return Err("context RAW name collision".into());
                }
            } else {
                fs::write(&path, &bytes)?;
            }
            let ns = row["source_ns"].as_str().unwrap().parse::<u64>()?;
            context.push(json!({"path":file,"record":row["record"],"sequence":row["sequence"],"timestamp_ns":ns,"relative_timing_ms":(ns as i128-target_ns as i128) as f64*1e-6,"role":(["before","target","after"][i]),"width":f["width"],"height":f["height"],"stride":f["stride"],"sensor_x":f["sensor_x"],"sensor_y":f["sensor_y"],"raw_sha256":row["raw_sha256"],"area_admitted_common_providers":providers,"source_frame":f}));
        }
        let f = &target["frame"];
        let meta = json!({"schema":"buttercup-native-eye-review-v1","annotation_scope":"eyelid_margins","annotation_group":name,"sequence":target["sequence"],"timestamp_ns":target_ns,"record":target["record"],"source":target["source"],"epoch":target["epoch"],"eye":target["eye"],"width":f["width"],"height":f["height"],"stride":f["stride"],"sensor_x":f["sensor_x"],"sensor_y":f["sensor_y"],"pixel_format":"RAW10_LE40_1X1","raw_sha256":target["raw_sha256"],"target_position":1,"context_frames":context,"recorded_prediction":null,"provenance":{"raw_source":source,"stream_entry":target["stream_entry"],"source_frame":f,"area_admitted_providers":providers,"selection":"six evenly spaced eligible ambiguous targets per eye, with nearest admitted before/after exposures within 300 ms and a common admitted provider; no fit error, sign or detector preference selects targets"}});
        fs::write(
            archive.join(format!("{stem}.json")),
            serde_json::to_vec_pretty(&meta)?,
        )?;
        receipt.push(meta);
    }
    let manifest = json!({"complete":true,"schema":"buttercup-native-lid-review-v1","capture":source,"archive":archive,"labels":labels,"targets":receipt,"eligible_windows_by_source_epoch_eye":eligible_counts,"canonical_labeler":"paired-limbus-annotator/server.py; canonical location specified by repository AGENTS.md","area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"retained_inputs_sha256":archive::digest(&fs::read(area.join("retained-inputs.jsonl"))?),"executable_sha256":archive::digest(&fs::read("/proc/self/exe")?),"human_labels_created":0,"physical_sign_truth":null});
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::write(root.join("README.md"),format!("# Native eyelid review set\n\n{} targets from the existing capture, each with native RAW10 before/target/after context. Every exposure passed the area gate for a provider shared across its triplet. Targets were selected uniformly among eligible ambiguous frames; no sign or lid result selected them. Original bytes and sensor/source metadata are preserved and hashed.\n\nUse the canonical annotator's explicit eyelid-margin scope. Mark only visibly supported upper/lower margin points; leave uncertain or hidden portions unknown. These are not limbus, globe-center, or gaze-sign labels. Recorded predictions are omitted from the review input. No human labels have been created.\n\nCanonical label output: `{}`.\n",receipt.len(),labels.display()))?;
    println!("{}", root.display());
    Ok(())
}
