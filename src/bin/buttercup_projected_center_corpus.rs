//! Source-bound extension of the independent projected eye-centre experiment.
//! Reuses exported native conics, never the previous replay's branch choices.
#![allow(dead_code)]
#[path = "../geometry.rs"]
mod geometry;
#[path = "../raw10.rs"]
mod raw10;
#[path = "../raw_preview.rs"]
mod raw_preview;
#[path = "../"]
mod native {
    pub(crate) mod binocular_coordinator;
    pub(crate) mod conic_solver;
    pub(crate) mod outline_conic_segments;
    pub(crate) mod roi_evidence;
    pub(crate) mod eye_scene_model {
        pub(crate) mod binocular_pose;
    }
}
use native::{
    binocular_coordinator, conic_solver, eye_scene_model, outline_conic_segments, roi_evidence,
};
#[path = "buttercup_motion_sign_sheet/eye_center.rs"]
mod eye_center;
use conic_solver::joint::{circle_pose_hypotheses, PinholeCamera};
use geometry::Ellipse;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};
type E = Box<dyn std::error::Error>;
fn num(v: &Value) -> f64 {
    v.as_f64().expect("numeric source field")
}
fn uint(v: &Value) -> u64 {
    v.as_u64()
        .or_else(|| v.as_str()?.parse().ok())
        .expect("integer source field")
}
fn rows(p: &Path) -> Result<Vec<Value>, E> {
    BufReader::new(File::open(p)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}
fn hash(p: &Path) -> Result<String, E> {
    let mut h = Sha256::new();
    std::io::copy(&mut File::open(p)?, &mut h)?;
    Ok(format!("{:x}", h.finalize()))
}
fn raw_hash(p: &Path, offset: u64, length: u64) -> Result<String, E> {
    let mut f = File::open(p)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut h = Sha256::new();
    let n = std::io::copy(&mut f.take(length), &mut h)?;
    if n != length {
        return Err("short native RAW exposure".into());
    }
    Ok(format!("{:x}", h.finalize()))
}
fn pair(v: &Value) -> [f64; 2] {
    [num(&v[0]), num(&v[1])]
}
fn vector(v: &Value) -> [f64; 3] {
    [num(&v[0]), num(&v[1]), num(&v[2])]
}
fn gate(e: Ellipse, f: &Value) -> Option<&'static str> {
    if ![
        e.center.0,
        e.center.1,
        e.major_radius,
        e.minor_radius,
        e.angle,
    ]
    .iter()
    .all(|x| x.is_finite())
    {
        return Some("nonfinite ellipse");
    }
    let local = [
        e.center.0 - num(&f["sensor_x"]),
        e.center.1 - num(&f["sensor_y"]),
    ];
    let w = num(&f["width"]);
    let h = num(&f["height"]);
    if local[0] < 0. || local[0] >= w || local[1] < 0. || local[1] >= h {
        return Some("ellipse centre outside current ROI");
    }
    if e.minor_radius < 4.
        || e.major_radius < e.minor_radius
        || e.major_radius > w.max(h) * 0.6
        || e.minor_radius / e.major_radius < 0.15
    {
        return Some("unusable conic extent");
    }
    None
}
fn label_audit(out: &Path, catalogue: &Path) -> Result<(), E> {
    let labels: Vec<Value> = serde_json::from_slice(&fs::read(catalogue)?)?;
    let sides = rows(&out.join("inputs/evaluation-sidecar.jsonl"))?;
    let decisions = rows(&out.join("evaluation/decisions.jsonl"))?
        .into_iter()
        .filter(|v| v["variant"] == "static-2s")
        .map(|v| (uint(&v["id"]), v))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut matched = vec![];
    let mut unmatched = vec![];
    let mut excluded = vec![];
    for entry in &labels {
        let path = Path::new(entry["path"].as_str().ok_or("label path")?);
        let label: Value = serde_json::from_slice(&fs::read(path)?)?;
        if label["reviewed"] != true
            || entry["assistant"] == true
            || path.to_string_lossy().contains("assistant")
        {
            excluded.push(json!({"label":path,"reason":"not reviewed human label"}));
            continue;
        }
        let native = Path::new(label["source_raw"].as_str().ok_or("label native RAW")?);
        let digest = hash(native)?;
        let origin = pair(&label["sensor_origin"]);
        let identity = (
            digest.clone(),
            uint(&label["frame_width"]),
            uint(&label["frame_height"]),
            origin.map(|x| x as u64),
        );
        if !seen.insert(identity) {
            continue;
        }
        let same = sides
            .iter()
            .filter(|s| {
                s["input"]["raw_sha256"] == digest
                    && s["input"]["frame"]["width"] == label["frame_width"]
                    && s["input"]["frame"]["height"] == label["frame_height"]
                    && num(&s["input"]["frame"]["sensor_x"]) == origin[0]
                    && num(&s["input"]["frame"]["sensor_y"]) == origin[1]
            })
            .collect::<Vec<_>>();
        if same.is_empty() {
            unmatched.push(json!({"label":path,"verified_native_raw_sha256":digest}));
            continue;
        }
        let points = label["annotation_points"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|p| p["kind"] == "iris_edge" && p["visibility"] == "visible")
            .map(|p| [num(&p["x_sensor"]), num(&p["y_sensor"])])
            .collect::<Vec<_>>();
        for source in same {
            let v = &source["source_ellipse"];
            if !v.is_object() {
                continue;
            }
            let c = pair(&v["center_sensor_px"]);
            let a = num(&v["a"]);
            let b = num(&v["b"]);
            let (sn, cs) = num(&v["angle"]).sin_cos();
            let rim = (0..3600)
                .map(|i| {
                    let t = i as f64 * std::f64::consts::TAU / 3600.;
                    [
                        c[0] + a * t.cos() * cs - b * t.sin() * sn,
                        c[1] + a * t.cos() * sn + b * t.sin() * cs,
                    ]
                })
                .collect::<Vec<_>>();
            let mut distances = points
                .iter()
                .map(|p| {
                    rim.iter()
                        .map(|q| (p[0] - q[0]).hypot(p[1] - q[1]))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect::<Vec<_>>();
            distances.sort_by(f64::total_cmp);
            let decision = decisions
                .get(&uint(&source["id"]))
                .ok_or("label source decision")?;
            matched.push(json!({"label":path,"id":source["id"],"capture":decision["capture"],"eye":decision["eye"],"sequence":decision["sequence"],"source_ns":decision["source_ns"],"verified_native_raw_sha256":digest,"visible_human_points":points.len(),"mean_distance_px":(!distances.is_empty()).then(||distances.iter().sum::<f64>()/distances.len() as f64),"median_distance_px":distances.get(distances.len()/2),"max_distance_px":distances.last(),"rim_discretization_upper_px":std::f64::consts::PI*a/3600.,"baseline_candidate_localization_difference_px":0.,"sign_status":decision["status"],"selected":decision["selected"],"ellipse_admitted":decision["ellipse"].is_object(),"ellipse":v,"provenance":label["provenance"]}));
        }
    }
    render_labels(out, &matched, &sides)?;
    fs::write(
        out.join("human-label-audit.json"),
        serde_json::to_vec_pretty(
            &json!({"catalogue":catalogue,"catalogue_sha256":hash(catalogue)?,"label_files_considered":labels.len(),"matched":matched,"unmatched_reviewed":unmatched,"excluded":excluded,"metric":"Native-RAW-hash + sensor-footprint matched visible human iris_edge landmarks to unchanged ellipse sampled at3600angles. Labels are localization support, never sign truth; no pseudo-labels or new training. Rejected conics remain scored where a native source conic exists.","candidate_changes_input_ellipse":false,"sign_accuracy_available":false}),
        )?,
    )?;
    Ok(())
}
fn render_labels(out: &Path, matched: &[Value], sides: &[Value]) -> Result<(), E> {
    use std::fmt::Write as _;
    let mut best_by_capture = BTreeMap::<String, &Value>::new();
    for v in matched {
        let key = v["capture"].as_str().unwrap().to_owned();
        if best_by_capture
            .get(&key)
            .is_none_or(|old| num(&v["mean_distance_px"]) < num(&old["mean_distance_px"]))
        {
            best_by_capture.insert(key, v);
        }
    }
    let mut selected = best_by_capture
        .values()
        .copied()
        .take(6)
        .collect::<Vec<_>>();
    selected.sort_by(|a, b| num(&a["mean_distance_px"]).total_cmp(&num(&b["mean_distance_px"])));
    let mut svg="<svg xmlns='http://www.w3.org/2000/svg' width='1800' height='1140'><rect width='100%' height='100%' fill='#171b24'/><g font-family='sans-serif' fill='white'><text x='25' y='34' font-size='25'>Source-matched RAW: archived ellipse versus reviewed human landmarks</text><text x='25' y='65' font-size='17'>White = archived conic; green = visible human iris-edge landmarks. Best labeled exposure per recording is shown.</text><text x='25' y='90' font-size='16'>The sign experiment does not move these ellipses. Landmarks measure localization; they do not label gaze sign.</text>".to_owned();
    for (i, v) in selected.iter().enumerate() {
        let side = sides
            .iter()
            .find(|s| s["id"] == v["id"])
            .ok_or("label review source")?;
        let input = &side["input"];
        let frame = &input["frame"];
        let w = uint(&frame["width"]) as usize;
        let h = uint(&frame["height"]) as usize;
        let origin = [num(&frame["sensor_x"]), num(&frame["sensor_y"])];
        let mut file = File::open(input["raw_file"].as_str().unwrap())?;
        file.seek(SeekFrom::Start(uint(&input["raw_offset"])))?;
        let mut bytes = vec![0; uint(&input["raw_length"]) as usize];
        file.read_exact(&mut bytes)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(v["verified_native_raw_sha256"], digest);
        let pixels = raw10::try_unpack_raw10(&bytes, w, h, uint(&frame["stride"]) as usize)?;
        let rgb = raw_preview::color_preview(
            &pixels,
            w,
            h,
            origin[0] as u32,
            origin[1] as u32,
            100,
            None,
        );
        let (x, y) = (25. + 600. * (i % 3) as f64, 155. + 500. * (i / 3) as f64);
        let scale = 550. / w as f64;
        let pt = |p: [f64; 2]| {
            [
                x + (p[0] - origin[0]) * scale,
                y + (p[1] - origin[1]) * scale,
            ]
        };
        write!(svg,"<text x='{x}' y='{}' font-size='15'>Eye{} / source{} / mean landmark error {:.1}px</text><clipPath id='clip{i}'><rect x='{x}' y='{y}' width='550' height='{}'/></clipPath><g clip-path='url(#clip{i})'><g transform='translate({x},{y}) scale({scale})' shape-rendering='crispEdges'>",y-17.,v["eye"],v["sequence"],num(&v["mean_distance_px"]),h as f64*scale)?;
        for yy in 0..h {
            for xx in 0..w {
                write!(
                    svg,
                    "<rect x='{xx}' y='{yy}' width='1' height='1' fill='#{:06x}'/>",
                    rgb[yy * w + xx]
                )?;
            }
        }
        svg.push_str("</g>");
        let e = &v["ellipse"];
        let center = pt(pair(&e["center_sensor_px"]));
        write!(svg,"<ellipse cx='{}' cy='{}' rx='{}' ry='{}' transform='rotate({} {} {})' fill='none' stroke='white' stroke-width='1.5'/>",center[0],center[1],num(&e["a"])*scale,num(&e["b"])*scale,num(&e["angle"]).to_degrees(),center[0],center[1])?;
        let label: Value = serde_json::from_slice(&fs::read(v["label"].as_str().unwrap())?)?;
        for p in label["annotation_points"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["kind"] == "iris_edge" && p["visibility"] == "visible")
        {
            let q = pt([num(&p["x_sensor"]), num(&p["y_sensor"])]);
            write!(svg,"<circle cx='{}' cy='{}' r='3.5' fill='#57ff91' stroke='#0b4220' stroke-width='0.8'/>",q[0],q[1])?;
        }
        write!(svg,"</g><text x='{x}' y='{}' font-size='13'>RAW {}</text><text x='{x}' y='{}' font-size='13'>{}</text>",y+h as f64*scale+26.,&digest[..24],y+h as f64*scale+49.,v["sign_status"].as_str().unwrap())?;
    }
    svg.push_str("</g></svg>");
    fs::write(out.join("human-label-raw-review.svg"), svg)?;
    Ok(())
}
fn run() -> Result<(), E> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|x| x == "--label-audit") {
        if args.len() != 3 {
            return Err("usage: --label-audit EXISTING_EXTENSION_OUT LABEL_CATALOGUE.json".into());
        }
        return label_audit(Path::new(&args[1]), Path::new(&args[2]));
    }
    if args.len() < 3 {
        return Err("usage: buttercup_projected_center_corpus OUT EXPORT.json EXPORT.json [..]; each manifest must have a matching .jsonl native-conic export".into());
    }
    let out = PathBuf::from(&args[0]);
    if out.exists() {
        return Err("use a new output directory".into());
    }
    if !out
        .parent()
        .ok_or("missing output parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let base = out.join("inputs");
    fs::create_dir_all(base.join("quality-gate"))?;
    fs::create_dir(base.join("expanded-corpus"))?;
    fs::write(base.join("expanded-corpus/admitted-shapes.jsonl"), "")?;
    fs::write(base.join("expanded-corpus/evaluation-sidecar.jsonl"), "")?;
    let mut shapes = BufWriter::new(File::create(
        base.join("quality-gate/admitted-shapes.jsonl"),
    )?);
    let mut sidecar = BufWriter::new(File::create(base.join("evaluation-sidecar.jsonl"))?);
    let mut extraction = BufWriter::new(File::create(out.join("extraction.jsonl"))?);
    let mut capture_audits = vec![];
    let mut seen_archives = BTreeSet::new();
    let mut seen_raw = BTreeSet::new();
    let mut id = 0_u64;
    let mut imported_conics = 0;
    for manifest in &args[1..] {
        let manifest_path = Path::new(manifest);
        let document: Value = serde_json::from_slice(&fs::read(manifest_path)?)?;
        let exported = rows(&manifest_path.with_extension("jsonl"))?;
        for audit in document["captures"]
            .as_array()
            .ok_or("capture manifest missing")?
        {
            let dir = PathBuf::from(audit["capture"].as_str().ok_or("capture path")?);
            if dir.to_string_lossy().contains("recording-staging") {
                return Err("volatile capture excluded".into());
            }
            let actual_frames_hash = hash(&dir.join("frames.jsonl"))?;
            let actual_predictions_hash = hash(&dir.join("predictions.jsonl"))?;
            if audit["frame_sha256"] != actual_frames_hash
                || audit["prediction_sha256"] != actual_predictions_hash
            {
                return Err(format!("source manifest changed: {}", dir.display()).into());
            }
            if !seen_archives.insert((actual_frames_hash.clone(), actual_predictions_hash.clone()))
            {
                continue;
            }
            if audit["source_stride"].as_u64().unwrap_or(1) != 1
                || audit["source_phase"].as_u64().unwrap_or(0) != 0
            {
                return Err("use full-cadence native conic export".into());
            }
            let saved = exported
                .iter()
                .filter(|x| x["capture"] == audit["capture"])
                .collect::<Vec<_>>();
            let mut indexed = BTreeMap::new();
            let mut kinds = BTreeSet::new();
            for row in &saved {
                let key = (
                    uint(&row["roi"]),
                    uint(&row["sequence"]),
                    uint(&row["source_ns"]),
                );
                if indexed.insert(key, *row).is_some() {
                    return Err("duplicate source conic in export".into());
                }
                kinds.insert(row["geometry"].as_str().ok_or("geometry kind")?);
            }
            if kinds.len() > 1 {
                return Err("split mixed upstream geometry sources before evaluating".into());
            }
            let joint = kinds.contains("saved-joint-pose-forward-projection");
            let provider = if joint {
                "historical-joint-conditioned"
            } else {
                "historical-semantic"
            };
            let stage = if joint {
                "joint-conditioned"
            } else {
                "stored-semantic-conic"
            };
            // The exported normals are checked against the current native unprojection.
            // For saved joint records also establish that their original K is the same
            // engineering prior, rather than guessing it from the export's pixels.
            if joint {
                let mut camera_count = 0;
                for row in rows(&dir.join("predictions.jsonl"))? {
                    let k = &row["predictions"]["eye_candidate"]["centers_and_gaze"]
                        ["joint_conics"]["intrinsics"];
                    if k.is_object() {
                        if pair(&k["focal_px"]) != [4000.; 2]
                            || pair(&k["principal_px"]) != [4000., 3000.]
                        {
                            return Err(
                                "different saved joint camera requires a separate evaluation"
                                    .into(),
                            );
                        }
                        camera_count += 1;
                    }
                }
                if camera_count == 0 {
                    return Err("missing saved joint camera".into());
                }
            }
            let camera = PinholeCamera {
                focal_px: [4000.; 2],
                principal_px: [4000., 3000.],
            };
            let mut frame_keys = BTreeSet::new();
            let mut raw_rows = 0;
            let mut accepted = 0;
            let mut missing = 0;
            let mut rejected = BTreeMap::<String, usize>::new();
            let mut raw_duplicates = 0;
            let mut unavailable_raw = 0;
            let mut unavailable_raw_conics = 0;
            let mut maximum_native_error = 0_f64;
            let mut last_semantic_ellipse = BTreeMap::<u64, Value>::new();
            let frame_rows = rows(&dir.join("frames.jsonl"))?;
            for frame in frame_rows {
                let eye = uint(&frame["eye_id"]);
                if eye != 1 && eye != 2 {
                    continue;
                }
                let sequence = uint(&frame["sequence"]);
                let ns = uint(&frame["timestamp_ns"]);
                let key = (eye, sequence, ns);
                if !frame_keys.insert(key) {
                    return Err("duplicate archive source exposure".into());
                }
                raw_rows += 1;
                let raw_name = frame["stream"]
                    .as_str()
                    .ok_or("missing native RAW stream")?;
                if Path::new(raw_name)
                    .components()
                    .any(|c| !matches!(c, std::path::Component::Normal(_)))
                {
                    return Err("invalid native RAW filename".into());
                }
                let raw_file = dir.join(raw_name);
                let offset = uint(&frame["offset"]);
                let length = uint(&frame["length"]);
                let raw_len = match fs::metadata(&raw_file) {
                    Ok(metadata) => Some(metadata.len()),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
                    Err(error) => return Err(error.into()),
                };
                if raw_len.is_none_or(|size| size < offset.saturating_add(length)) {
                    unavailable_raw += 1;
                    if indexed.remove(&key).is_some() {
                        unavailable_raw_conics += 1;
                    }
                    continue;
                }
                let mut ellipse = Value::Null;
                let mut source_ellipse = Value::Null;
                let mut source_hash = Value::Null;
                let mut reason = "no exported source conic".to_string();
                let mut baseline = Value::Null;
                if let Some(record) = indexed.remove(&key) {
                    imported_conics += 1;
                    let source_key = &record["source_key"];
                    if source_key.is_object() && *source_key != frame["source_clock"]["source_key"]
                    {
                        return Err("full source key mismatch".into());
                    }
                    let v = &record["ellipse_sensor"];
                    let c = pair(&v["center"]);
                    let e = Ellipse {
                        center: (c[0], c[1]),
                        major_radius: num(&v["major"]),
                        minor_radius: num(&v["minor"]),
                        angle: num(&v["angle"]),
                    };
                    source_ellipse = json!({"center_sensor_px":c,"a":e.major_radius,"b":e.minor_radius,"angle":e.angle});
                    let poses = circle_pose_hypotheses(camera, e, [0, 0])
                        .ok_or("export no longer unprojects")?;
                    let old = [vector(&record["normals"][0]), vector(&record["normals"][1])];
                    let error = poses
                        .iter()
                        .map(|p| {
                            old.iter()
                                .map(|n| {
                                    p.normal
                                        .iter()
                                        .zip(n)
                                        .map(|(x, y)| (x - y).powi(2))
                                        .sum::<f64>()
                                        .sqrt()
                                })
                                .fold(f64::INFINITY, f64::min)
                        })
                        .fold(0., f64::max);
                    maximum_native_error = maximum_native_error.max(error);
                    if error > 1e-6 {
                        return Err("export/native unprojection disagreement".into());
                    }
                    let digest = raw_hash(&raw_file, offset, length)?;
                    let unique = (eye, ns, sequence, digest.clone());
                    if !seen_raw.insert(unique) {
                        raw_duplicates += 1;
                        continue;
                    }
                    source_hash = json!(digest);
                    let repeated_unattested = !joint
                        && last_semantic_ellipse
                            .get(&eye)
                            .is_some_and(|old| *old == source_ellipse);
                    if !joint {
                        last_semantic_ellipse.insert(eye, source_ellipse.clone());
                    }
                    let failure = gate(e, &frame).or(repeated_unattested
                        .then_some("unchanged semantic estimate lacks independent freshness"));
                    if let Some(failure) = failure {
                        reason = failure.to_string();
                        *rejected.entry(reason.clone()).or_default() += 1;
                    } else {
                        ellipse = source_ellipse.clone();
                        accepted += 1;
                        reason = "admitted source conic".to_string();
                    }
                    baseline = json!({"recorded_branch":record["recorded_branch"],"recorded_resolved":record["recorded_resolved"],"not_ground_truth":true});
                } else {
                    missing += 1;
                }
                let clock = frame["source_clock"]["source_key"]["stream_epoch"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("legacy-archive:{}", actual_frames_hash));
                let input = json!({"frame":frame,"raw_file":raw_file,"raw_offset":offset,"raw_length":length,"raw_sha256":source_hash});
                let shape = json!({"id":id,"capture":dir,"provider":provider,"stage":stage,"clock":clock,"eye":eye,"source_ns":ns,"sequence":sequence,"ellipse":ellipse});
                writeln!(shapes, "{shape}")?;
                writeln!(
                    sidecar,
                    "{}",
                    json!({"id":id,"input":input,"retained":[],"source_ellipse":source_ellipse,"baseline_diagnostic_only":baseline})
                )?;
                writeln!(
                    extraction,
                    "{}",
                    json!({"id":id,"capture":dir,"eye":eye,"sequence":sequence,"source_ns":ns,"reason":reason,"raw_sha256":source_hash})
                )?;
                id += 1;
            }
            if !indexed.is_empty() {
                return Err("exported source conic absent from exact RAW index".into());
            }
            let result = json!({"capture":dir,"provider":provider,"stage":stage,"indexed_eye_exposures":raw_rows,"raw_eye_exposures":raw_rows-unavailable_raw,"unavailable_indexed_raw":unavailable_raw,"unavailable_raw_conics":unavailable_raw_conics,"exported_conics":saved.len(),"admitted_ellipses":accepted,"missing_exported_conic":missing,"geometry_rejections":rejected,"duplicate_raw_exposures_skipped":raw_duplicates,"maximum_native_normal_error":maximum_native_error,"frames_sha256":actual_frames_hash,"predictions_sha256":actual_predictions_hash,"source_export":manifest_path.with_extension("jsonl"),"source_export_sha256":hash(&manifest_path.with_extension("jsonl"))?});
            println!("{result}");
            capture_audits.push(result);
        }
    }
    shapes.flush()?;
    sidecar.flush()?;
    extraction.flush()?;
    let metadata = json!({"schema":"projected-eye-centre-source-extension-v2","captures":capture_audits,"rows":id,"imported_conics":imported_conics,"gate":"finite conic, centre inside current ROI, minor radius >=4px, major radius <=0.6 of larger ROI dimension, axis ratio >=0.15; exact unchanged semantic estimates lacking an explicit current source attestation are unavailable. Basic geometry sanity only, not human localization or segmentation validation","source_binding":"Original frame/prediction manifests SHA256-verified; full source key where recorded; otherwise archive+eye+sequence+sensor time. Native RAW hashes for every exported source conic. Recompute both native normals and require <=1e-6 vector error against export. Previous branch decisions are never solver input.","missing":"All physically present indexed eye RAW exposures retained as unavailable when no source conic. Unchanged unattested semantic publications are not counted as fresh; changing legacy semantic estimates still have unverified geometry-evidence time. Conditional counts are not accuracy. No calibration labels or independent head pose used."});
    fs::write(
        out.join("source-audit.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    eye_center::run(&base, &out.join("evaluation"))?;
    let path = out.join("evaluation/summary.json");
    let mut summary: Value = serde_json::from_slice(&fs::read(&path)?)?;
    summary["data"]=json!(format!("{} additional archives, {} native eye exposures; source-audit.json reports all missing/invalid geometry and upstream stages",metadata["captures"].as_array().unwrap().len(),id));
    summary["limitations"]=json!("Stored semantic conics and saved joint-conditioned forward projections, separately labeled. Basic ROI/conic sanity gate differs from the earlier four-recording provider gate. No true gaze/sign labels, measured intrinsics or independent head motion. Conditional branch choices are coverage, not accuracy; prior replay choices are not solver inputs. No SN-FEIDA/localization improvement is possible because conics are unchanged.");
    fs::write(path, serde_json::to_vec_pretty(&summary)?)?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("projected-centre corpus: {e}");
        std::process::exit(1);
    }
}
