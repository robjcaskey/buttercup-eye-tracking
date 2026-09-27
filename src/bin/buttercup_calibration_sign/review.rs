use super::{data, native, Result};
use buttercup_eye_tracking::{
    calibration_sign_model::{self as net, Model},
    recorded_bundle::BundleSource,
};
use serde_json::{json, Value};
use std::{fs, path::Path};

pub fn infer(args: &[String]) -> Result<()> {
    if !(4..=5).contains(&args.len()) {
        return Err(
            "infer MODEL.json PAIR.json OUTPUT.json [NATIVE_CONIC_OR_PROJECTED_BRANCHES.json]"
                .into(),
        );
    }
    let model: Model = serde_json::from_slice(&fs::read(&args[1])?)?;
    model.validate()?;
    let pair: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    if !data::pair_clock(&pair["previous"], &pair["current"]) {
        return Err(
            "inference requires consecutive fresh frames of one native eye/source lineage".into(),
        );
    }
    let archive = pair["archive"].as_str().ok_or("archive")?;
    let bundle = BundleSource::open(Path::new(archive))?;
    let mut images = Vec::new();
    let mut hashes = Vec::new();
    let mut conic = None;
    for (i, key) in ["previous", "current"].iter().enumerate() {
        let f = &pair[key];
        let n = |k: &str| data::num(&f[k]).ok_or_else(|| format!("missing {k}"));
        let raw = bundle.read_range(
            f["stream"].as_str().ok_or("stream")?,
            n("offset")?,
            n("length")? as usize,
        )?;
        let hash = data::digest(&raw);
        if pair["raw_sha256"][i].as_str() != Some(&hash) {
            return Err("inference RAW does not match declared content identity".into());
        }
        images.push(net::image(
            &raw,
            n("width")? as usize,
            n("height")? as usize,
            n("stride")? as usize,
        )?);
        hashes.push(hash);
        if i == 1
            && model.branch.is_some()
            && model.provenance["geometry_provider"] != "sam31-single"
            && args.len() == 4
        {
            conic = native::for_supervision(&native::unpack(&raw, f)?, f);
        }
    }
    let prediction = model.predict(&images[0], &images[1])?;
    let mut projected: Option<[[f32; 2]; 2]> = None;
    if let Some(path) = args.get(4) {
        let geometry: Value = serde_json::from_slice(&fs::read(path)?)?;
        if model.branch.is_some() {
            let hash = geometry["source"]["raw_sha256"]
                .as_str()
                .ok_or("native conic RAW identity")?;
            if native::identity(hash, &geometry["source"]["frame"])
                != native::identity(&hashes[1], &pair["current"])
            {
                return Err("conic metadata/RAW identity differs from current image".into());
            }
            conic = serde_json::from_value(geometry["fit"].clone())?;
            if model.provenance["geometry_provider"] == "sam31-single"
                && conic.as_ref().is_some_and(|f| f.provider != "sam31-single")
            {
                return Err(
                    "SAM-trained branch review requires the declared geometry provider".into(),
                );
            }
        } else {
            projected = Some(serde_json::from_value(geometry)?);
        }
    }
    let branch = if model.branch.is_some() {
        prediction
            .choose_native_branches(conic.as_ref().filter(|c| c.admissible()).map(|c| c.normals))
    } else {
        prediction.choose_projected_branches(projected)
    };
    let output = Path::new(&args[3]);
    let parent = fs::canonicalize(output.parent().ok_or("output parent")?)?;
    if !parent.starts_with(fs::canonicalize("data")?) {
        return Err("inference output must use checked runtime links".into());
    }
    if output.exists() {
        return Err("inference output already exists".into());
    }
    data::write(
        output,
        &json!({"schema":"buttercup-two-frame-sign-inference-v1","model_sha256":data::digest(&fs::read(&args[1])?),"raw_sha256":hashes,"prediction":prediction,"native_conic":conic,"conditional_conic_branch":branch,"note":"conditional model support is not calibrated physical sign probability; no target field was used for inference"}),
    )
}

/// Source-matched diagnostic; never an annotation UI or training-label source.
pub fn review(run: &str) -> Result<()> {
    let run = Path::new(run);
    let report: Value = serde_json::from_slice(&fs::read(run.join("results.json"))?)?;
    let mut selected = Vec::new();
    for day in report["days"].as_array().ok_or("days")? {
        let day = day.as_u64().ok_or("day")?;
        let bytes = fs::read_to_string(run.join(format!("predictions-day-{day}-two-frame.jsonl")))?;
        let mut rows: Vec<Value> = bytes
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        rows.sort_by(|a, b| {
            let max = |r: &Value| {
                r["class_scores"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(Value::as_f64)
                    .fold(0f64, f64::max)
            };
            max(b).total_cmp(&max(a))
        });
        let mut seen = std::collections::BTreeSet::new();
        let mut correct = 0;
        let mut wrong = 0;
        for row in rows {
            let good = row["correct_target_class"] == true;
            let identity = (
                row["source"]["session"].to_string(),
                row["source"]["target"]["id"].to_string(),
            );
            if (good && correct >= 3) || (!good && wrong >= 3) || !seen.insert(identity) {
                continue;
            }
            if good {
                correct += 1
            } else {
                wrong += 1
            };
            selected.push(row);
            if correct == 3 && wrong == 3 {
                break;
            }
        }
    }
    data::write(run.join("raw-review-selection.json"), &selected)?;
    for (page, rows) in selected.chunks(6).enumerate() {
        let w = 840usize;
        let rowh = 308usize;
        let h = rows.len() * rowh;
        let mut rgb = vec![22u8; w * h * 3];
        let mut filters = Vec::new();
        for (ri, row) in rows.iter().enumerate() {
            let source = &row["source"];
            let bundle =
                BundleSource::open(Path::new(source["archive"].as_str().ok_or("archive")?))?;
            for (j, key) in ["previous", "current"].iter().enumerate() {
                let f = &source[key];
                let n = |k: &str| data::num(&f[k]).unwrap() as usize;
                let bytes = bundle.read_range(
                    f["stream"].as_str().unwrap(),
                    n("offset") as u64,
                    n("length"),
                )?;
                if data::digest(&bytes) != source["raw_sha256"][j].as_str().unwrap() {
                    return Err("RAW review identity mismatch".into());
                }
                let raw = buttercup_eye_tracking::raw10::try_unpack_raw10(
                    &bytes,
                    n("width"),
                    n("height"),
                    n("stride"),
                )?;
                let mut sorted = raw.clone();
                sorted.sort_unstable();
                let lo = sorted[sorted.len() / 200] as f32;
                let hi = sorted[sorted.len() * 199 / 200] as f32;
                for y in 0..280 {
                    for x in 0..420 {
                        let xx = (x * n("width") / 420) / 4 * 4;
                        let yy = (y * n("height") / 280) / 4 * 4;
                        let mut sum = 0f32;
                        for dy in 0..4 {
                            for dx in 0..4 {
                                sum += raw[(yy + dy) * n("width") + xx + dx] as f32 / 16.;
                            }
                        }
                        let gray =
                            (255. * ((sum - lo) / (hi - lo).max(1.)).clamp(0., 1.).powf(0.7)) as u8;
                        let pos = ((ri * rowh + y) * w + j * 420 + x) * 3;
                        rgb[pos..pos + 3].fill(gray);
                    }
                }
            }
            let uv = &source["target"]["uv"];
            let p = &row["predicted_uv"];
            let text = format!(
                "{} | day {} eye {} seq {} | target {:.2},{:.2} predicted {:.2},{:.2}",
                if row["correct_target_class"] == true {
                    "AGREES"
                } else {
                    "DISAGREES"
                },
                source["day"],
                source["current"]["eye_id"],
                source["current"]["sequence"],
                uv[0].as_f64().unwrap(),
                uv[1].as_f64().unwrap(),
                p[0].as_f64().unwrap(),
                p[1].as_f64().unwrap()
            );
            filters.push(format!(
                "drawtext=text='{text}':x=6:y={}:fontsize=15:fontcolor=white",
                ri * rowh + 283
            ));
        }
        let name = format!("raw-review-{page}");
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        ppm.extend(rgb);
        fs::write(run.join(format!("{name}.ppm")), ppm)?;
        let status = std::process::Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-y",
                "-i",
            ])
            .arg(run.join(format!("{name}.ppm")))
            .args(["-vf", &filters.join(","), "-frames:v", "1", "-threads", "1"])
            .arg(run.join(format!("{name}.png")))
            .status()?;
        if !status.success() {
            return Err("native review rendering failed".into());
        }
    }
    Ok(())
}
