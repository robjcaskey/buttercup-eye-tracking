//! Face-first, RAW-backed layer-order sandbox. Masks are editable hypotheses,
//! not human labels or a fitted anatomical depth map.
use super::geometry::{refine_affine, score, smooth, Affine, Features, Samples};
use super::{
    binary, digest, image, json, load, origin, BundleSource, Canvas, Image, Result, Value,
};
use std::{fs, path::Path};

fn face_samples(f: &Features, fold: usize) -> Samples {
    let (w, h) = (f.image.w, f.image.h);
    let mut s = Samples {
        points: Vec::new(),
        values: Vec::new(),
        center: [w as f64 / 2., h as f64 / 2.],
    };
    for y in (16..h - 16).step_by(6) {
        if y as f64 >= h as f64 * 0.20 && (y as f64) <= h as f64 * 0.80 {
            continue;
        }
        for x in (16..w - 16).step_by(6) {
            if (x / 24 + y / 24) % 2 != fold {
                continue;
            }
            let p = [x as f64, y as f64];
            if let Some(v) = f.sample(p) {
                s.points.push(p);
                s.values.push(v);
            }
        }
    }
    s
}
// Evaluate every fitting sample so the coarse seed cannot overfit 64 pixels.
fn face_translation(
    s: &Samples,
    f: &Features,
    delta: [f64; 2],
) -> (Affine, super::geometry::Score) {
    let mut best = (
        Affine::translation(delta),
        score(s, f, Affine::translation(delta)),
    );
    for y in -16..=16 {
        for x in -16..=16 {
            let m = Affine::translation([delta[0] + 2. * x as f64, delta[1] + 2. * y as f64]);
            let v = score(s, f, m);
            if v.loss < best.1.loss {
                best = (m, v);
            }
        }
    }
    refine_affine(s, f, best.0, s.center, false)
}
fn bounded_face(m: Affine) -> bool {
    let det = m.a[0][0] * m.a[1][1] - m.a[0][1] * m.a[1][0];
    (0.85..=1.15).contains(&det)
        && m.a[0][1].abs() <= 0.15
        && m.a[1][0].abs() <= 0.15
        && m.a[0][0] > 0.85
        && m.a[1][1] > 0.85
}
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("usage: --iris-layers SAVED_IRIS_REVIEW_JSON NEW_OUTPUT".into());
    }
    let source = Path::new(&args[2]).canonicalize()?;
    let out = Path::new(&args[3]);
    let checked = fs::canonicalize("outputs")?;
    if !source.starts_with(&checked)
        || out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(&checked)
    {
        return Err("saved checked review and new checked output required".into());
    }
    let bytes = fs::read(&source)?;
    let review: Value = serde_json::from_slice(&bytes)?;
    let report = &review["report"];
    let cfg = &report["config"];
    let eye = cfg["eye"].as_u64().ok_or("eye")?;
    let bundle = BundleSource::open(Path::new(report["bundle"].as_str().ok_or("native bundle")?))?;
    let metas = String::from_utf8(bundle.read_entry("frames.jsonl")?)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|m| m["eye_id"] == eye)
        .collect::<Vec<_>>();
    let first = cfg["first"].as_u64().ok_or("first")? as usize;
    let last = cfg["last"].as_u64().ok_or("last")? as usize;
    if last >= metas.len() || last <= first || last - first > 60 {
        return Err("bounded source interval required".into());
    }
    let clock = metas[0]["timestamp_ns"].as_u64().ok_or("clock")?;
    let inputs = metas[first..=last]
        .iter()
        .map(|m| load(&bundle, m, clock))
        .collect::<Result<Vec<_>>>()?;
    if review["frames"].as_array().map(Vec::len) != Some(inputs.len()) {
        return Err("source length mismatch".into());
    }
    for (i, input) in inputs.iter().enumerate() {
        if review["frames"][i]["raw_sha256"] != input.hash
            || review["frames"][i]["timestamp_ns"]
                != input.meta["timestamp_ns"]
                    .as_u64()
                    .ok_or("timestamp")?
                    .to_string()
        {
            return Err("source hash/timestamp mismatch".into());
        }
    }
    fs::create_dir(out)?;
    let features = inputs
        .iter()
        .map(|i| Features::with_mask(&smooth(&i.image, 2), &i.image, 0.70))
        .collect::<Vec<_>>();
    let train = face_samples(&features[0], 0);
    let held = face_samples(&features[0], 1);
    if train.points.len() < 40 || held.points.len() < 40 {
        return Err("insufficient outer-band support".into());
    }
    let mut frames = Vec::new();
    let mut rows = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        let a = origin(&inputs[0].meta);
        let b = origin(&input.meta);
        let delta = [a[0] - b[0], a[1] - b[1]];
        let crop = Affine::translation(delta);
        let (translation, translation_score) = if i == 0 {
            (crop, score(&train, &features[i], crop))
        } else {
            face_translation(&train, &features[i], delta)
        };
        let (candidate, candidate_score) = if i == 0 {
            (crop, translation_score)
        } else {
            refine_affine(&train, &features[i], translation, train.center, true)
        };
        let selected =
            if bounded_face(candidate) && candidate_score.loss + 0.02 < translation_score.loss {
                candidate
            } else {
                translation
            };
        let measured = score(&held, &features[i], selected);
        let reliable = measured.ncc >= 0.50 && measured.coverage >= 0.70;
        let (w, h) = (input.image.w, input.image.h);
        let mask = vec![true; w * h];
        let gray = binary::downsample_masked(&input.image, &mask, 8).image;
        let (cw, ch) = (gray.w, gray.h);
        // Whole-image display only; RAW floating means above supply the probe.
        let channels: [Image; 3] = std::array::from_fn(|c| Image {
            w,
            h,
            v: input
                .preview
                .chunks_exact(4)
                .map(|p| p[c] as f64 / 255.)
                .collect(),
        });
        let display = channels.map(|im| binary::downsample_masked(&im, &mask, 8).image);
        let mut pixels = Vec::with_capacity(cw * ch * 4);
        let mut peaks = Vec::with_capacity(cw * ch);
        for y in 0..ch {
            for x in 0..cw {
                let k = y * cw + x;
                for c in 0..3 {
                    pixels.push((display[c].v[k].clamp(0., 1.) * 255.).round() as u8);
                }
                pixels.push(255);
                let mut peak = 0f64;
                for yy in (y * 8).saturating_sub(3)..(y * 8 + 5).min(h) {
                    for xx in (x * 8).saturating_sub(3)..(x * 8 + 5).min(w) {
                        peak = peak.max(input.image.v[yy * w + xx]);
                    }
                }
                peaks.push((peak.clamp(0., 1.) * 1023.).round() as u16);
            }
        }
        image(out, i, input)?;
        let mut canvas = Canvas::new(cw, ch)?;
        canvas.image(&pixels, cw, ch, 0., 0., cw as f64, ch as f64);
        canvas.png(&out.join(format!("small-{i:03}.png")))?;
        let row = json!({"index":i,"time_s":input.time,"raw_sha256":input.hash,"timestamp_ns":input.meta["timestamp_ns"].as_u64().unwrap().to_string(),
            "crop_warp":crop,"face_warp":selected,"face_train":score(&train,&features[i],selected),"face_held":measured,"face_reliable":reliable,
            "crop_held":score(&held,&features[i],crop),"translation_held":score(&held,&features[i],translation),"affine_selected":selected.a!=translation.a,
            "prior_iris_warp":review["frames"][i]["variants"][2]["warp"]});
        rows.push(row.clone());
        frames.push(json!({"evidence":row,"image":format!("small-{i:03}.png"),"native_image":format!("raw-{i:03}.png"),
            "gray":gray.v.iter().map(|v|(v*65535.).round() as u16).collect::<Vec<_>>(),"bright":peaks}));
    }
    let mean = |key: &str| {
        rows.iter()
            .skip(1)
            .map(|r| r[key]["ncc"].as_f64().unwrap())
            .sum::<f64>()
            / (rows.len() - 1) as f64
    };
    let summary = json!({"schema":"iris-layer-order-experiment-v1","eye":eye,"frames":frames.len(),"first":first,"last":last,"source_review":source,"source_review_sha256":digest(&bytes),
        "face_fit_locations":train.points.len(),"face_held_locations":held.points.len(),"mean_crop_held_ncc":mean("crop_held"),"mean_translation_held_ncc":mean("translation_held"),"mean_face_held_ncc":mean("face_held"),
        "reliable_after_reference":rows.iter().skip(1).filter(|r|r["face_reliable"]==true).count(),"affine_selected_after_reference":rows.iter().skip(1).filter(|r|r["affine_selected"]==true).count(),
        "provenance":{"recipe":digest(include_bytes!("iris_layers.rs")),"viewer":digest(include_bytes!("iris_layers_viewer.html")),"geometry":digest(include_bytes!("iris_pivot_math.rs")),"decode":digest(include_bytes!("warp_probe.rs"))},
        "scope":"Outer top/bottom 20% image bands are face-motion proxies, not human-labeled face tissue. Both source folds omit the inner image, and have disjoint 24px tiles but correlated preprocessing. Translation versus bounded affine is selected by fitting loss only; held NCC and coverage flag weak support without choosing a model. Sensor crop shifts initialize all fits. Every frame matches the original reference. Editable eye/face apertures and fresh brightness components are mask hypotheses; brightness rank is not a tracked glint identity, and layer order is not physical depth. Layer-order residual translation probes do not replace the prior 3D iris solve. One native Rob recording, no human tissue correspondence truth, independent scale or new-user validation. SN-FEIDA is not applicable to this mask/motion sandbox."});
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    fs::write(
        out.join("face-motion.json"),
        serde_json::to_vec_pretty(&rows)?,
    )?;
    let data = json!({"report":summary,"width":(inputs[0].image.w-1)/8+1,"height":(inputs[0].image.h-1)/8+1,"factor":8,"native_width":inputs[0].image.w,"native_height":inputs[0].image.h,
        "roi":review["support"]["roi"],"face_train":train.points,"face_held":held.points,"frames":frames});
    fs::write(out.join("layers.json"), serde_json::to_vec(&data)?)?;
    fs::write(
        out.join("viewer.html"),
        include_str!("iris_layers_viewer.html")
            .replace("LAYER_DATA", &serde_json::to_string(&data)?),
    )?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn face_search_recovers_translation_with_gain_and_offset_on_held_tiles() {
        let (w, h) = (160, 120);
        let texture = |x: f64, y: f64| {
            (x * 0.19 + y * 0.13).sin()
                + (x * 0.071 - y * 0.23).cos()
                + ((x * x + y * y) * 0.0017).sin()
        };
        let make = |dx: f64, dy: f64, gain: f64, offset: f64| Features {
            image: Image {
                w,
                h,
                v: (0..w * h)
                    .map(|k| gain * texture((k % w) as f64 - dx, (k / w) as f64 - dy) + offset)
                    .collect(),
            },
            valid: vec![true; w * h],
            center_hint: None,
        };
        let reference = make(0., 0., 1., 0.);
        let target = make(6., -4., 1.7, 0.3);
        let fit = face_samples(&reference, 0);
        let held = face_samples(&reference, 1);
        let (m, _) = face_translation(&fit, &target, [2., -2.]);
        assert!((m.t[0] - 6.).abs() < 0.1 && (m.t[1] + 4.).abs() < 0.1);
        assert!(score(&held, &target, m).ncc > 0.9999);
    }
}
