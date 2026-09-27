//! Frozen historical image-model evaluation on the exact ambiguous ROI set.
//! No optimizer, new labels, promoted model, or live-tracker change.
use super::{
    archive::{self, Frame, Manifest},
    Result,
};
use buttercup_eye_tracking::{
    calibration_sign_model::{self as net, Model, Prediction},
    focus_region::*,
    raw10,
    recorded_bundle::BundleSource,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};
#[path = "../../bootstrapability.rs"]
#[allow(dead_code)]
mod boot;
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
#[path = "../buttercup_calibration_sign/data.rs"]
#[allow(dead_code)]
mod data;
use canvas::*;

const DAY_NS: u64 = 86_400_000_000_000;
const ARMS: [&str; 2] = ["current-only", "two-frame"];
struct Fold {
    day: u64,
    models: [Model; 2],
    validation_group: String,
    fitting_hashes: HashSet<String>,
    fitting_groups: HashSet<String>,
}
fn read_json(p: impl AsRef<Path>) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn json_lines(p: impl AsRef<Path>) -> Result<Vec<Value>> {
    BufReader::new(fs::File::open(p)?)
        .lines()
        .map(|s| Ok(serde_json::from_str(&s?)?))
        .collect()
}
fn write_json(p: impl AsRef<Path>, value: &Value) -> Result<()> {
    fs::write(p, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0
}
fn identity(hash: &str, row: &Value) -> String {
    let key = &row["source_clock"]["source_key"];
    format!(
        "{hash}:{}:{}:{}:{}:{}:{}:{}:{}:{}",
        row["eye_id"],
        key["stream_epoch"],
        key["sequence"],
        key["sensor_timestamp_ns"],
        row["sensor_x"],
        row["sensor_y"],
        row["width"],
        row["height"],
        row["stride"]
    )
}
fn load_models(
    run: &Path,
    report: &Value,
) -> Result<(Vec<Fold>, [HashMap<String, Value>; 2], Value)> {
    let mut folds = vec![];
    for d in report["days"].as_array().ok_or("model days")? {
        let day = d.as_u64().ok_or("day")?;
        let mut models = vec![];
        for (arm, name) in ARMS.iter().enumerate() {
            let name = format!("day-{day}-{name}.json");
            let bytes = fs::read(run.join(&name))?;
            let expected = report["model_artifacts"]
                .as_array()
                .unwrap()
                .iter()
                .find(|a| a["path"] == name)
                .ok_or("model hash manifest missing")?;
            if expected["sha256"] != archive::digest(&bytes) {
                return Err("saved model hash changed".into());
            }
            let model: Model = serde_json::from_slice(&bytes)?;
            model.validate()?;
            if model.branch.is_none()
                || model.provenance["test_day"] != day
                || model.provenance["source"] != report["source"]
                || model.provenance["current_only"] != (arm == 0)
            {
                return Err("wrong model/fold/input provenance".into());
            }
            models.push(model);
        }
        let validation_group = models[0].provenance["validation_group"]
            .as_str()
            .ok_or("validation group")?
            .to_owned();
        if models[1].provenance["validation_group"] != validation_group {
            return Err("model arms have different validation groups".into());
        }
        folds.push(Fold {
            day,
            models: models.try_into().ok().unwrap(),
            validation_group,
            fitting_hashes: HashSet::new(),
            fitting_groups: HashSet::new(),
        });
    }
    let pairs = json_lines(run.join("pairs.jsonl"))?;
    if pairs.len() as u64 != report["training_pairs"].as_u64().unwrap() {
        return Err("training pair inventory count changed".into());
    }
    let mut partitions = vec![];
    for fold in &mut folds {
        let mut training = 0;
        let mut validation = 0;
        let mut held_out = 0;
        for p in &pairs {
            let day = p["day"].as_u64().ok_or("pair day")?;
            let group = p["viewer_group"].as_str().ok_or("pair group")?;
            if day == fold.day {
                held_out += 1;
                continue;
            }
            if group == fold.validation_group {
                validation += 1;
            } else {
                training += 1;
            }
            fold.fitting_groups.insert(group.into());
            for h in p["raw_sha256"].as_array().ok_or("training RAW hashes")? {
                fold.fitting_hashes
                    .insert(h.as_str().ok_or("RAW hash")?.into());
            }
        }
        let original = report["folds"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["test_day"] == fold.day && f["arm"] == "current-only")
            .ok_or("fold report")?;
        if original["training_pairs"] != training
            || original["validation_pairs"] != validation
            || original["test_pairs"] != held_out
        {
            return Err("partition reconstruction differs from training report".into());
        }
        partitions.push(json!({"day":fold.day,"train":training,"validation":validation,"held_out":held_out,"fitting_raw_hashes":fold.fitting_hashes.len(),"fitting_viewer_groups":fold.fitting_groups.len()}));
    }
    let mut saved: [HashMap<String, Value>; 2] = [HashMap::new(), HashMap::new()];
    for fold in &folds {
        for (arm, name) in ARMS.iter().enumerate() {
            for row in json_lines(run.join(format!("predictions-day-{}-{name}.jsonl", fold.day)))? {
                let id = identity(
                    row["source"]["raw_sha256"][1]
                        .as_str()
                        .ok_or("saved RAW hash")?,
                    &row["source"]["current"],
                );
                if saved[arm].insert(id, row).is_some() {
                    return Err("duplicate saved current-exposure identity".into());
                }
            }
        }
    }
    Ok((folds, saved, json!(partitions)))
}

struct Native {
    bundle: BundleSource,
    path: String,
    day: u64,
    rows: HashMap<usize, Value>,
}
fn open_source(
    source: u32,
    ids: &BTreeSet<usize>,
    frames: &[Frame],
    manifest: &Manifest,
    paths: &BTreeMap<u32, String>,
) -> Result<Native> {
    let s = &manifest.sources[source as usize];
    let path = paths.get(&source).unwrap_or(&s.path).clone();
    let bundle = BundleSource::open(Path::new(&path))?;
    let index = bundle.read_entry(&format!("{}frames.jsonl", s.prefix))?;
    if archive::digest(&index) != s.frames_sha256 {
        return Err("original/recovered frame index changed".into());
    }
    let wanted = ids
        .iter()
        .map(|&id| (frames[id].index as usize, id))
        .collect::<HashMap<_, _>>();
    let mut rows = HashMap::new();
    let mut first_host = None;
    for (line, bytes) in index
        .split(|b| *b == b'\n')
        .filter(|b| !b.is_empty())
        .enumerate()
    {
        if line != 0 && !wanted.contains_key(&line) {
            continue;
        }
        let row: Value = serde_json::from_slice(bytes)?;
        if line == 0 {
            first_host = data::num(&row["host_arrival_unix_ns"]);
        }
        if let Some(&id) = wanted.get(&line) {
            let f = &frames[id];
            let n = data::num;
            for (key, want) in [
                ("eye_id", f.eye as u64),
                ("width", f.width as u64),
                ("height", f.height as u64),
                ("stride", f.stride as u64),
                ("sensor_x", f.origin[0] as u64),
                ("sensor_y", f.origin[1] as u64),
                ("sequence", f.sequence),
                ("timestamp_ns", f.ns),
                ("offset", f.offset),
                ("length", f.length as u64),
            ] {
                if n(&row[key]) != Some(want) {
                    return Err(format!("binary/native mismatch at record {id}: {key}").into());
                }
            }
            if row["stream"].as_str() != Some(s.streams[f.stream as usize].as_str())
                || row["source_clock"]["source_key"]["stream_epoch"].as_str()
                    != Some(manifest.epochs[f.epoch as usize].as_str())
            {
                return Err("RAW stream/epoch mismatch".into());
            }
            rows.insert(id, row);
        }
    }
    if rows.len() != ids.len() {
        return Err("requested source rows missing".into());
    }
    let session = Path::new(&s.path).with_extension("session.json");
    let start = if session.is_file() {
        data::num(&read_json(&session)?["started_host_unix_ns"]).or(first_host)
    } else {
        first_host
    };
    Ok(Native {
        bundle,
        path,
        day: start.ok_or("no source recording day")? / DAY_NS,
        rows,
    })
}
struct Pixels {
    raw: Vec<u8>,
    hash: String,
    features: std::result::Result<Vec<f32>, String>,
}
fn pixels(native: &Native, f: &Frame, manifest: &Manifest) -> Result<Pixels> {
    let s = &manifest.sources[f.source as usize];
    let raw = native.bundle.read_range(
        &format!("{}{}", s.prefix, s.streams[f.stream as usize]),
        f.offset,
        f.length as usize,
    )?;
    let hash = archive::digest(&raw);
    let features = net::image(&raw, f.width as usize, f.height as usize, f.stride as usize);
    Ok(Pixels {
        raw,
        hash,
        features,
    })
}
fn average(predictions: &[Prediction]) -> Prediction {
    let mut p = predictions[0].clone();
    let avg = |f: fn(&Prediction) -> Vec<f32>| {
        let mut a = vec![0.; f(&p).len()];
        for p in predictions {
            for (x, v) in a.iter_mut().zip(f(p)) {
                *x += v / predictions.len() as f32;
            }
        }
        a
    };
    let target = avg(|p| p.target_scores.to_vec());
    let branch = avg(|p| p.conditional_branch_scores.unwrap().to_vec());
    let uv = avg(|p| p.uv.to_vec());
    let h = avg(|p| p.horizontal.to_vec());
    let v = avg(|p| p.vertical.to_vec());
    p.target_scores = target.try_into().unwrap();
    p.conditional_branch_scores = Some(branch.try_into().unwrap());
    p.uv = uv.try_into().unwrap();
    p.horizontal = h.try_into().unwrap();
    p.vertical = v.try_into().unwrap();
    p
}
fn evaluate(
    arm: usize,
    f: &Frame,
    current: &Pixels,
    previous: Option<&Pixels>,
    pair_valid: bool,
    group: &str,
    day: u64,
    folds: &[Fold],
    rays: TheoreticalEllipseExplanations,
    saved: Option<&Value>,
    identity_matches_pair: bool,
) -> Result<Value> {
    let unavailable = |reason: &str| json!({"scored":false,"reason":reason});
    let Ok(cur) = &current.features else {
        return Ok(unavailable("current-RAW-insufficient-contrast"));
    };
    let prev = if arm == 0 {
        cur
    } else {
        if !pair_valid {
            return Ok(unavailable("no-consecutive-fresh-previous-frame"));
        }
        let Some(previous) = previous else {
            return Ok(unavailable("previous-RAW-missing"));
        };
        let Ok(features) = &previous.features else {
            return Ok(unavailable("previous-RAW-insufficient-contrast"));
        };
        features
    };
    let known_day = folds.iter().any(|f| f.day == day);
    let mut members = vec![];
    let mut predictions = vec![];
    let mut rejected = vec![];
    for fold in folds.iter().filter(|f| !known_day || f.day == day) {
        if fold.fitting_groups.contains(group)
            || fold.fitting_hashes.contains(&current.hash)
            || (arm == 1 && previous.is_some_and(|p| fold.fitting_hashes.contains(&p.hash)))
        {
            rejected.push(fold.day);
            continue;
        }
        let p = fold.models[arm].predict(prev, cur)?;
        let choice = p.choose_native_branches(Some(rays.rays.map(|r| r.direction)));
        members.push(json!({"fold_day":fold.day,"scores":p.conditional_branch_scores,"raw_class":argmax(&p.conditional_branch_scores.unwrap()),"choice":choice}));
        predictions.push(p);
    }
    if predictions.is_empty() {
        return Ok(
            json!({"scored":false,"reason":"no-model-excluding-source-from-training-and-validation","rejected_folds":rejected}),
        );
    }
    let p = average(&predictions);
    let choice = p.choose_native_branches(Some(rays.rays.map(|r| r.direction)));
    let raw_class = argmax(&p.conditional_branch_scores.unwrap());
    let mut parity = None;
    if let Some(saved) = saved.filter(|_| identity_matches_pair || arm == 0) {
        if !known_day || predictions.len() != 1 || saved["source"]["day"] != day {
            return Err("saved scored exposure is not using its held-out model".into());
        }
        let old: Vec<f32> = serde_json::from_value(saved["conditional_branch_scores"].clone())?;
        let targets: Vec<f32> = serde_json::from_value(saved["class_scores"].clone())?;
        let delta = p
            .conditional_branch_scores
            .unwrap()
            .iter()
            .chain(p.target_scores.iter())
            .zip(old.iter().chain(&targets))
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        if delta > 1e-6 {
            return Err(format!("frozen model parity failed: {delta}").into());
        }
        parity = Some(delta);
    }
    let teacher = saved.and_then(|s| s["conditional_branch_label"].as_u64());
    let lower = usize::from(rays.rays[1].direction[0] < rays.rays[0].direction[0]);
    let teacher_branch = teacher.map(|t| if t == 0 { lower } else { 1 - lower });
    let votes = p.conditional_branch_scores.unwrap();
    let unanimous = choice.selected.is_some_and(|selected| {
        members
            .iter()
            .all(|m| m["choice"]["selected"].as_u64() == Some(selected as u64))
    });
    Ok(
        json!({"scored":true,"policy":if known_day{"held-out-recording-day"}else{"mean-of-three-frozen-folds-on-unseen-day"},"members":members,"rejected_folds":rejected,
        "prediction":p,"raw_class":raw_class,"support":votes[raw_class],"diagnostic_choice_on_original_candidates":choice,
        "all_members_support_same_choice":unanimous,"members_disagree_raw_class":members.iter().any(|m|m["raw_class"].as_u64()!=Some(raw_class as u64)),
        "same_geometry_provider_as_training":f.provider==1,"contract_compatible_selection":if f.provider==1{choice.selected}else{None},
        "conditional_teacher_class":teacher,"conditional_teacher_candidate":teacher_branch,"teacher_agreement":teacher.map(|t|t==raw_class as u64),
        "selected_teacher_agreement":choice.selected.zip(teacher_branch).map(|(a,b)|a==b),"saved_score_max_difference":parity,
        "target_class":saved.map(|s|s["source"]["target_class"].clone()),"target_class_agreement":saved.and_then(|s|s["source"]["target_class"].as_u64()).map(|t|t==argmax(&p.target_scores) as u64)}),
    )
}

fn selected(arm: &Value) -> Option<u64> {
    arm["diagnostic_choice_on_original_candidates"]["selected"].as_u64()
}
fn stats(rows: &[Value], arm: usize) -> Value {
    let mut result = BTreeMap::<String, usize>::new();
    let mut reasons = BTreeMap::<String, usize>::new();
    for r in rows {
        let a = &r["arms"][arm];
        let c = |name: &str, result: &mut BTreeMap<String, usize>| {
            *result.entry(name.into()).or_default() += 1;
        };
        c("total", &mut result);
        if a["scored"] != true {
            *reasons
                .entry(a["reason"].as_str().unwrap_or("unavailable").into())
                .or_default() += 1;
            continue;
        }
        c("image_scored", &mut result);
        if a["support"].as_f64().unwrap_or(0.) >= 0.8 {
            c("image_support_at_least_0_8", &mut result);
        }
        if a["members_disagree_raw_class"] == true {
            c("ensemble_members_disagree", &mut result);
        }
        if a["teacher_agreement"].is_boolean() {
            c("teacher_comparable", &mut result);
            if a["teacher_agreement"] == true {
                c("teacher_matches", &mut result);
            }
        }
        if a["saved_score_max_difference"].is_number() {
            c("original_scored_frame_parity_checks", &mut result);
        }
        if a["target_class_agreement"].is_boolean() {
            c("target_comparable", &mut result);
            if a["target_class_agreement"] == true {
                c("target_matches", &mut result);
            }
        }
        if let Some(branch) = selected(a) {
            c("diagnostic_selected", &mut result);
            if a["all_members_support_same_choice"] == true {
                c("all_members_support_same_choice", &mut result);
            }
            if a["contract_compatible_selection"].is_number() {
                c("sam_geometry_selected", &mut result);
            } else {
                c("contact_geometry_diagnostic_only", &mut result);
            }
            if a["selected_teacher_agreement"].is_boolean() {
                c("selected_teacher_comparable", &mut result);
                if a["selected_teacher_agreement"] == true {
                    c("selected_teacher_matches", &mut result);
                }
            }
            for (name, key, age) in [
                ("continuity_2s", "most_recent_unambiguous", 2000.),
                ("continuity_immediate", "immediately_previous", 250.),
            ] {
                let m = &r["continuity"][key];
                if m["strong_5_20"] == true
                    && m["age_ms"].as_f64().is_some_and(|v| v > 0. && v <= age)
                {
                    c(&format!("{name}_comparable"), &mut result);
                    if m["current_branch"].as_u64() == Some(branch) {
                        c(&format!("{name}_matches"), &mut result);
                    }
                }
            }
        } else {
            *reasons
                .entry(
                    a["diagnostic_choice_on_original_candidates"]["reason"]
                        .as_str()
                        .unwrap_or("no-choice")
                        .into(),
                )
                .or_default() += 1;
        }
    }
    json!({"counts":result,"abstention_reasons":reasons})
}
fn review_image(
    out: &Path,
    row: &Value,
    current: &Pixels,
    previous: Option<&Pixels>,
    f: &Frame,
    prior: Option<&Frame>,
    rays: TheoreticalEllipseExplanations,
) -> Result<()> {
    let mut c = Canvas::new(1700, 1000)?;
    c.clear();
    c.text(
        22.,
        37.,
        27.,
        WHITE,
        &format!(
            "Frozen ROI sign models | recording {} | eye {} | seq {}",
            f.source, f.eye, f.sequence
        ),
    );
    c.text(22.,69.,18.,MUTED,&format!("{} | both focus-region interpretations valid | conditional scores, not measured correctness",row["evaluation_scope"].as_str().unwrap()));
    for (col, (pixels, f)) in [(previous, prior), (Some(current), Some(f))]
        .into_iter()
        .enumerate()
    {
        let x = 22. + col as f64 * 560.;
        let Some(p) = pixels else {
            continue;
        };
        let Some(f) = f else {
            continue;
        };
        let values = raw10::try_unpack_raw10(
            &p.raw,
            f.width as usize,
            f.height as usize,
            f.stride as usize,
        )?;
        // Display-only complete CFA-cell average; all inference uses net::image.
        let sw = f.width as usize / 4;
        let sh = f.height as usize / 4;
        let mut gray = vec![];
        for y in 0..sh {
            for x in 0..sw {
                let mut sum = 0.;
                for dy in 0..4 {
                    for dx in 0..4 {
                        sum += values[(y * 4 + dy) * f.width as usize + x * 4 + dx] as f64 / 16.;
                    }
                }
                gray.push(sum);
            }
        }
        let mut sorted = gray.clone();
        sorted.sort_by(f64::total_cmp);
        let lo = sorted[sorted.len() / 100];
        let hi = sorted[sorted.len() * 99 / 100];
        let bgra = gray
            .iter()
            .flat_map(|v| {
                let q = (255. * ((v - lo) / (hi - lo).max(1.)).clamp(0., 1.)) as u8;
                [q, q, q, 255]
            })
            .collect::<Vec<_>>();
        c.text(
            x,
            107.,
            20.,
            WHITE,
            if col == 0 {
                "Previous RAW (4x4 CFA mean)"
            } else {
                "Current RAW / original ellipse (4x4 CFA mean)"
            },
        );
        c.image(&bgra, sw, sh, x, 127., 530., 353.33);
        if col == 1 {
            let s = 530. / f.width as f64;
            let pp = |p: [f64; 2]| {
                [
                    x + (p[0] - f.origin[0] as f64) * s,
                    127. + (p[1] - f.origin[1] as f64) * s,
                ]
            };
            let points = f
                .shape()
                .unwrap()
                .dense_points(200)
                .iter()
                .map(|p| pp([p.0, p.1]))
                .collect::<Vec<_>>();
            c.clipped(x, 127., 530., 353.33, |c| {
                c.path(&points, 2., WHITE);
                let project =
                    |p: V3| pp([4000. + 4000. * p[0] / -p[2], 3000. + 4000. * p[1] / -p[2]]);
                for (b, r) in rays.rays.iter().enumerate() {
                    c.arrow(
                        project(r.origin_iris_radii),
                        project(add(r.origin_iris_radii, scale(r.direction, 0.8))),
                        [CYAN, PINK][b],
                    );
                }
            });
        }
        c.text(
            x,
            506.,
            17.,
            MUTED,
            &format!("RAW {}... | seq {}", &p.hash[..12], f.sequence),
        );
        if let Ok(features) = &p.features {
            let bgra = features
                .iter()
                .flat_map(|v| {
                    let q = ((v + 1.) * 127.5).clamp(0., 255.) as u8;
                    [q, q, q, 255]
                })
                .collect::<Vec<_>>();
            c.image(&bgra, net::WIDTH, net::HEIGHT, x, 534., 256., 192.);
            c.text(x + 273., 570., 16., MUTED, "Actual 32x24");
            c.text(x + 273., 596., 16., MUTED, "neural input");
        }
    }
    for (a, name) in ARMS.iter().enumerate() {
        let x = 1150.;
        let y = 116. + a as f64 * 270.;
        let r = &row["arms"][a];
        c.text(x, y, 25., WHITE, name);
        if r["scored"] == true {
            let scores = &r["prediction"]["conditional_branch_scores"];
            c.text(
                x,
                y + 36.,
                18.,
                MUTED,
                &format!(
                    "low-X {:.4} / high-X {:.4}",
                    scores[0].as_f64().unwrap(),
                    scores[1].as_f64().unwrap()
                ),
            );
            c.text(
                x,
                y + 76.,
                25.,
                if selected(r).is_some() { GREEN } else { ORANGE },
                &selected(r)
                    .map(|b| format!("Candidate {}", if b == 0 { "A (cyan)" } else { "B (pink)" }))
                    .unwrap_or("Abstain".into()),
            );
            c.text(
                x,
                y + 106.,
                15.,
                MUTED,
                r["diagnostic_choice_on_original_candidates"]["reason"]
                    .as_str()
                    .unwrap(),
            );
            c.text(
                x,
                y + 140.,
                17.,
                WHITE,
                &format!("{} frozen model(s)", r["members"].as_array().unwrap().len()),
            );
            c.text(
                x,
                y + 168.,
                17.,
                MUTED,
                &format!(
                    "All agree with support: {}",
                    r["all_members_support_same_choice"]
                ),
            );
            c.text(
                x,
                y + 200.,
                17.,
                MUTED,
                &format!("Teacher agreement: {}", r["teacher_agreement"]),
            );
        } else {
            c.text(
                x,
                y + 40.,
                16.,
                ORANGE,
                r["reason"].as_str().unwrap_or("unavailable"),
            );
        }
    }
    let recent = &row["continuity"]["most_recent_unambiguous"];
    c.text(
        22.,
        794.,
        20.,
        WHITE,
        &format!(
            "Geometry source: {} | post-affine area {:.0} px2 | branch separation {:.1} deg",
            if f.provider == 1 {
                "SAM"
            } else {
                "archived virtual-contact reconstruction"
            },
            f.area.frontal_equivalent_disk_px2,
            rays.separation_degrees
        ),
    );
    if recent["strong_5_20"] == true && recent["age_ms"].as_f64().is_some_and(|v| v <= 2000.) {
        c.text(
            22.,
            834.,
            20.,
            WHITE,
            &format!(
                "Prior unique direction ({:.0} ms ago): candidate {} at {:.2} deg; other {:.2} deg",
                recent["age_ms"].as_f64().unwrap(),
                if recent["current_branch"] == 0 {
                    "A"
                } else {
                    "B"
                },
                recent["near_degrees"].as_f64().unwrap(),
                recent["far_degrees"].as_f64().unwrap()
            ),
        );
    } else {
        c.text(
            22.,
            834.,
            20.,
            MUTED,
            "No strong <=2s continuity reference for this frame.",
        );
    }
    c.text(22.,900.,18.,MUTED,"Held-out folds exclude training/validation RAW hashes and viewer sessions. Contact-derived geometry is diagnostic only.");
    c.text(22.,934.,18.,MUTED,"The teacher and continuity references are fallible estimates. Independent physical sign truth is unavailable.");
    c.png(&out.join(format!(
        "review-source-{}-record-{}.png",
        f.source, row["record"]
    )))?;
    Ok(())
}

pub fn run(args: &[String]) -> Result<()> {
    let binary = Path::new(&args[0]);
    let replay = Path::new(&args[1]);
    let run = Path::new(&args[2]);
    let continuity = Path::new(&args[3]);
    let movie = Path::new(&args[4]);
    let out = data::output(&args[5])?;
    let report_bytes = fs::read(run.join("results.json"))?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    let graph_bytes = fs::read(run.join("bootstrap-graph.json"))?;
    let graph =
        boot::parse(&graph_bytes).map_err(|e| format!("historical graph: {}", e.message))?;
    let source = boot::current_source(Path::new("."))?;
    let historical = boot::validate(&graph, &graph.source)
        .map_err(|e| format!("invalid historical graph: {}", e.message))?;
    let current = boot::validate(&graph, &source);
    let roots = read_json(run.join("roots.json"))?;
    for (id, actual) in [
        (
            "raw",
            archive::digest(&serde_json::to_vec(&roots["native_raw"])?),
        ),
        (
            "targets",
            archive::digest(&serde_json::to_vec(&roots["target_measurements"])?),
        ),
        (
            "model",
            archive::digest(&serde_json::to_vec(&report["model_artifacts"])?),
        ),
        ("evaluation", archive::digest(&report_bytes)),
    ] {
        if graph
            .nodes
            .iter()
            .find(|n| n.id == id)
            .and_then(|n| n.sha256.as_deref())
            != Some(actual.as_str())
        {
            return Err(format!("historical {id} manifest hash mismatch").into());
        }
    }
    write_json(
        out.join("provenance-preflight.json"),
        &json!({"current_source":source,"training_source":graph.source,"historical_declared_graph":historical,"current_source_graph_check":current,"scope":"Explicit historical experimental comparison allowed by bootstrapability.md; no training, teacher use, selector training, model promotion or current-checkout cold-bootstrap claim."}),
    )?;
    let (folds, saved, partitions) = load_models(run, &report)?;
    let focus = read_json(replay.join("summary.json"))?;
    let movie_summary = read_json(movie.join("summary.json"))?;
    let binary_hash = archive::digest(&fs::read(binary)?);
    if focus["binary_sha256"] != binary_hash || movie_summary["binary_sha256"] != binary_hash {
        return Err("binary does not match original focus/video experiment".into());
    }
    let (manifest, frames) = archive::read(binary)?;
    let mut cases = BTreeMap::new();
    for c in json_lines(continuity.join("cases.jsonl"))? {
        let id = c["record"].as_u64().ok_or("case record")? as usize;
        if cases.insert(id, c).is_some() {
            return Err("duplicate ambiguity case".into());
        }
    }
    if cases.len() as u64 != focus["classes"]["multiple"].as_u64().unwrap() {
        return Err("ambiguous set count changed".into());
    }
    let mut original = HashMap::new();
    for event in json_lines(movie.join("movie-frames.jsonl"))? {
        for e in event["eyes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| !e.is_null())
        {
            let id = e["record"].as_u64().unwrap() as usize;
            if cases.contains_key(&id) {
                if e["class"] != "multiple" || original.insert(id, e.clone()).is_some() {
                    return Err("ambiguous movie identity changed".into());
                }
            }
        }
    }
    if original.len() != cases.len() {
        return Err("movie lacks ambiguous RAW identities".into());
    }
    let paths = movie_summary["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["source"].as_u64().unwrap() as u32,
                r["raw_source_path"]
                    .as_str()
                    .unwrap_or(r["path"].as_str().unwrap())
                    .to_owned(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let keys = cases
        .keys()
        .map(|&id| (frames[id].source, frames[id].epoch, frames[id].eye))
        .collect::<BTreeSet<_>>();
    let mut groups = BTreeMap::<(u32, u32, u16), Vec<usize>>::new();
    for (i, f) in frames.iter().enumerate() {
        if keys.contains(&(f.source, f.epoch, f.eye)) {
            groups
                .entry((f.source, f.epoch, f.eye))
                .or_default()
                .push(i);
        }
    }
    let mut predecessors = HashMap::new();
    let mut needed = BTreeMap::<u32, BTreeSet<usize>>::new();
    for (_, mut group) in groups {
        group.sort_by_key(|&i| (frames[i].ns, frames[i].sequence));
        for (pos, &id) in group.iter().enumerate() {
            if cases.contains_key(&id) {
                needed.entry(frames[id].source).or_default().insert(id);
                if pos > 0 {
                    let p = group[pos - 1];
                    predecessors.insert(id, p);
                    needed.entry(frames[id].source).or_default().insert(p);
                }
            }
        }
    }
    let mut rows = vec![];
    let mut source_receipts = vec![];
    let mut writer = BufWriter::new(fs::File::create(out.join("predictions.jsonl"))?);
    let mut preview_index = vec![];
    for (source_id, ids) in &needed {
        let native = open_source(*source_id, ids, &frames, &manifest, &paths)?;
        let mut cache = HashMap::new();
        let mut source_cases = 0;
        let mut preview_tags = BTreeSet::new();
        for &id in ids.iter().filter(|id| cases.contains_key(id)) {
            let f = &frames[id];
            let case = &cases[&id];
            let raw_row = &native.rows[&id];
            if case["source"].as_u64() != Some(f.source as u64)
                || case["sequence"].as_u64() != Some(f.sequence)
                || case["source_ns"].as_str() != Some(f.ns.to_string().as_str())
            {
                return Err("continuity case changed source identity".into());
            }
            let prev = predecessors.get(&id).copied();
            for record in prev.into_iter().chain(std::iter::once(id)) {
                if !cache.contains_key(&record) {
                    cache.insert(record, pixels(&native, &frames[record], &manifest)?);
                }
            }
            let current = &cache[&id];
            let previous = prev.map(|p| &cache[&p]);
            if original[&id]["raw_sha256"] != current.hash {
                return Err("RAW differs from ambiguity movie exposure".into());
            }
            let rays: TheoreticalEllipseExplanations =
                serde_json::from_value(original[&id]["explanations"].clone())?;
            let fresh = TheoreticalEllipseExplanations::from_ellipse(
                f.shape().ok_or("ellipse missing")?,
                [4000.; 2],
                [4000., 3000.],
            )
            .ok_or("conic disappeared")?;
            if fresh
                .rays
                .iter()
                .zip(rays.rays)
                .any(|(a, b)| norm(sub(a.direction, b.direction)) > 1e-10)
            {
                return Err("original candidate geometry changed".into());
            }
            let pair_valid = prev.is_some_and(|p| data::pair_clock(&native.rows[&p], raw_row));
            let group = raw_row["source_clock"]["source_key"]["viewer_session_id"]
                .as_str()
                .ok_or("native viewer group")?;
            let identity = identity(&current.hash, raw_row);
            let mut arms = vec![];
            for a in 0..2 {
                let scored = saved[a].get(&identity);
                let matched_pair = scored.is_some_and(|s| {
                    previous.is_some_and(|p| s["source"]["raw_sha256"][0] == p.hash)
                });
                arms.push(evaluate(
                    a,
                    f,
                    current,
                    previous,
                    pair_valid,
                    group,
                    native.day,
                    &folds,
                    rays,
                    scored,
                    matched_pair,
                )?);
            }
            let selected_agreement = selected(&arms[0])
                .zip(selected(&arms[1]))
                .map(|(a, b)| a == b);
            let row = json!({"record":id,"source":f.source,"epoch":f.epoch,"eye":f.eye,"sequence":f.sequence,"source_ns":f.ns.to_string(),"provider":f.provider,"recording_day":native.day,"evaluation_scope":if folds.iter().any(|fold|fold.day==native.day){"held-out recording day"}else{"recording day absent from all original training"},"raw_source":native.path,"raw_sha256":current.hash,"previous_record":prev,"previous_raw_sha256":previous.map(|p|&p.hash),"pair_valid":pair_valid,"pair_age_ms":prev.map(|p|(f.ns-frames[p].ns) as f64*1e-6),"source_clock_group":group,"geometry":rays,"disk_area":f.area.json(),"arms":arms,"arms_agree_when_both_select":selected_agreement,"continuity":case,"physical_sign_truth":null});
            serde_json::to_writer(&mut writer, &row)?;
            writeln!(writer)?;
            let recent = &case["most_recent_unambiguous"];
            let strong_recent = recent["strong_5_20"] == true
                && recent["age_ms"]
                    .as_f64()
                    .is_some_and(|v| v > 0. && v <= 2000.);
            let continuity_disagreement = strong_recent
                && row["arms"].as_array().unwrap().iter().any(|a| {
                    selected(a).is_some_and(|b| recent["current_branch"].as_u64() != Some(b))
                });
            let tag = if f.provider == 1 && continuity_disagreement {
                "sam-continuity-disagreement"
            } else if f.provider == 1 && strong_recent {
                "sam-continuity-reference"
            } else if f.provider == 1 {
                "sam-example"
            } else if selected_agreement == Some(false) {
                "models-disagree"
            } else if row["arms"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a["teacher_agreement"] == false)
            {
                "teacher-disagreement"
            } else {
                "first-example"
            };
            if preview_tags.insert(tag) {
                review_image(
                    &out,
                    &row,
                    current,
                    previous,
                    f,
                    prev.map(|p| &frames[p]),
                    rays,
                )?;
                preview_index.push(json!({"record":id,"source":f.source,"reason":tag,"image":format!("review-source-{}-record-{id}.png",f.source)}));
            }
            rows.push(row);
            source_cases += 1;
        }
        source_receipts.push(json!({"source":source_id,"archive":manifest.sources[*source_id as usize].path,"raw_path":native.path,"day":native.day,"frame_index_sha256":manifest.sources[*source_id as usize].frames_sha256,"ambiguous_frames":source_cases,"decoded_images":cache.len()}));
        eprintln!("MODEL EVAL source {source_id}: {source_cases} ambiguous RAW frames, {} unique images, {} / {} complete",cache.len(),rows.len(),cases.len());
    }
    writer.flush()?;
    if rows.len() != cases.len() {
        return Err("not every ambiguous frame was attempted".into());
    }
    let mut provider_stats = vec![];
    for p in [1, 3] {
        let subset = rows
            .iter()
            .filter(|r| r["provider"] == p)
            .cloned()
            .collect::<Vec<_>>();
        provider_stats.push(json!({"provider":p,"arms":ARMS.iter().enumerate().map(|(a,name)|json!({"arm":name,"metrics":stats(&subset,a)})).collect::<Vec<_>>()}));
    }
    let mut source_stats = vec![];
    for &s in needed.keys() {
        let subset = rows
            .iter()
            .filter(|r| r["source"] == s)
            .cloned()
            .collect::<Vec<_>>();
        source_stats.push(json!({"source":s,"arms":ARMS.iter().enumerate().map(|(a,name)|json!({"arm":name,"metrics":stats(&subset,a)})).collect::<Vec<_>>()}));
    }
    let mut day_stats = vec![];
    for day in source_receipts
        .iter()
        .map(|r| r["day"].as_u64().unwrap())
        .collect::<BTreeSet<_>>()
    {
        let subset = rows
            .iter()
            .filter(|r| r["recording_day"] == day)
            .cloned()
            .collect::<Vec<_>>();
        day_stats.push(json!({"day":day,"arms":ARMS.iter().enumerate().map(|(a,name)|json!({"arm":name,"metrics":stats(&subset,a)})).collect::<Vec<_>>()}));
    }
    let both = rows
        .iter()
        .filter(|r| r["arms_agree_when_both_select"].is_boolean())
        .count();
    let agree = rows
        .iter()
        .filter(|r| r["arms_agree_when_both_select"] == true)
        .count();
    let result = json!({"schema":"buttercup-frozen-sign-model-ambiguity-evaluation-v1","source":source,"training_source":report["source"],"binary_sha256":binary_hash,"training_results_sha256":archive::digest(&report_bytes),"model_artifacts":report["model_artifacts"],"partition_audit":partitions,"sources":source_receipts,"ambiguous_frames":rows.len(),"independent_sign_accuracy":null,"arms":ARMS.iter().enumerate().map(|(a,name)|json!({"arm":name,"metrics":stats(&rows,a)})).collect::<Vec<_>>(),"both_models_select":both,"both_models_agree":agree,"by_provider":provider_stats,"by_source":source_stats,"by_day":day_stats,
        "policy":{"held_out_days":"Use the original excluded-day fold; exclude fitting/validation viewer groups and RAW content.","unseen_days":"Unweighted mean probabilities from the three frozen fold models. Individual members and disagreement retained. No fitting or best-fold selection.","current_only":"Same current features in both image slots, matching its training recipe.","two_frame":"Immediate preceding original same-eye frame, strict native pair_clock, sequence+1 and dt<=250ms. Never fill missing history with held pixels.","selection":"Unchanged >=0.8 uncalibrated image support; distinct camera-X ordering >=0.08 and valid camera-facing normals.","geometry":"Original focus experiment normals unchanged. SAM-provider contract-compatible selections reported separately from contact-derived diagnostic mappings.","parity":"Every exact saved held-out exposure is compared against original target and branch scores; tolerance1e-6.","truth":"Saved conditional teacher and <=2s continuity are agreement references only; no independent physical sign truth.","area":"Input ellipses and their unnormalized frontal-equivalent disk areas unchanged; no independent scale or SN-FEIDA improvement claim."}});
    write_json(out.join("summary.json"), &result)?;
    write_json(out.join("visual-review.json"), &json!(preview_index))?;
    if boot::current_source(Path::new("."))? != source {
        return Err("source changed during evaluation; keep result marked unsealed".into());
    }
    write_json(
        out.join("complete.json"),
        &json!({"complete":true,"frames":rows.len(),"source_stable":true,"new_training":false,"live_changes":false}),
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"frames":rows.len(),"arms":result["arms"],"both_select":both,"both_agree":agree})
        )?
    );
    Ok(())
}
