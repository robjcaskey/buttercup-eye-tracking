//! Native-RAW, reduced-resolution perspective motion comparison. Offline only.
use super::{
    binary, digest, image, json, load, origin, BundleSource, Canvas, Image, Result, Value, P,
};
use std::{fs, path::Path};
#[path = "z_motion3d_math.rs"]
mod motion;
use motion::{Level, Picture};
#[path = "z_motion_groups.rs"]
mod groups;
fn reduced(rgb: &Picture, factor: usize) -> Picture {
    let mask = vec![true; rgb[0].w * rgb[0].h];
    std::array::from_fn(|c| binary::downsample_masked(&rgb[c], &mask, factor).image)
}
fn png(out: &Path, name: &str, p: &Picture) -> Result<()> {
    let (w, h) = (p[0].w, p[0].h);
    let mut bytes = Vec::new();
    for k in 0..w * h {
        for c in [2, 1, 0] {
            bytes.push((p[c].v[k].clamp(0., 1.) * 255.).round() as u8);
        }
        bytes.push(255);
    }
    let mut canvas = Canvas::new(w, h)?;
    canvas.image(&bytes, w, h, 0., 0., w as f64, h as f64);
    canvas.png(&out.join(name))?;
    Ok(())
}
fn frame(out: &Path, i: usize, p: &Picture, f: &Picture, time: f64, hash: &str) -> Result<Value> {
    let ultra = format!("ultra-{i:03}.png");
    let coarse = format!("coarse-{i:03}.png");
    png(out, &ultra, p)?;
    png(out, &coarse, f)?;
    Ok(
        json!({"ultra":ultra,"coarse":coarse,"time":time,"raw_sha256":hash,"native":format!("raw-{i:03}.png")}),
    )
}
fn pair(
    a: &Picture,
    b: &Picture,
    fa: &Picture,
    fb: &Picture,
    crop: P,
    reference: usize,
    target: usize,
) -> Value {
    let ultra = motion::evaluate(a, b, crop, None);
    let coarse = motion::evaluate(fa, fb, crop.map(|v| v * 3.), Some(&ultra));
    eprintln!(
        "3D {reference}->{target}: {}x{} held 2D {:.4} / 3D {:.4}; 53px {:.4} / {:.4}",
        a[0].w,
        a[0].h,
        ultra.baseline_held.loss,
        ultra.candidate_held.loss,
        coarse.baseline_held.loss,
        coarse.candidate_held.loss
    );
    json!({"reference":reference,"target":target,"ultra":ultra,"coarse":coarse})
}
pub fn run(args: &[String]) -> Result<()> {
    if !(4..=5).contains(&args.len()) {
        return Err("usage: --z-motion3d SAVED_IRIS_REVIEW_JSON|synthetic NEW_OUTPUT [HISTORICAL_MOTION_JSON]".into());
    }
    let root = fs::canonicalize("outputs")?;
    let out = Path::new(&args[3]);
    if out.exists()
        || !out
            .parent()
            .ok_or("parent")?
            .canonicalize()?
            .starts_with(&root)
    {
        return Err("new checked output required".into());
    }
    fs::create_dir(out)?;
    if args[2].starts_with("synthetic") {
        return if args[1] == "--z-motion-groups" || args[1] == "--z-motion-history" {
            groups::synthetic(out, &args[2], args[1] == "--z-motion-history")
        } else {
            synthetic(out)
        };
    }
    let source = Path::new(&args[2]).canonicalize()?;
    if !source.starts_with(&root) {
        return Err("checked review required".into());
    }
    let bytes = fs::read(&source)?;
    let review: Value = serde_json::from_slice(&bytes)?;
    let cfg = &review["report"]["config"];
    let eye = cfg["eye"].as_u64().ok_or("eye")?;
    let first = cfg["first"].as_u64().ok_or("first")? as usize;
    let last = cfg["last"].as_u64().ok_or("last")? as usize;
    let bundle = BundleSource::open(Path::new(
        review["report"]["bundle"].as_str().ok_or("bundle")?,
    ))?;
    let metas = String::from_utf8(bundle.read_entry("frames.jsonl")?)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|m| m["eye_id"] == eye)
        .collect::<Vec<_>>();
    if first >= last || last >= metas.len() || last - first > 60 {
        return Err("bounded native interval required".into());
    }
    let clock = metas[0]["timestamp_ns"].as_u64().ok_or("clock")?;
    let inputs = metas[first..=last]
        .iter()
        .map(|m| load(&bundle, m, clock))
        .collect::<Result<Vec<_>>>()?;
    if review["frames"].as_array().map(Vec::len) != Some(inputs.len()) {
        return Err("frame count mismatch".into());
    }
    let mut ultra = Vec::new();
    let mut coarse = Vec::new();
    let mut frames = Vec::new();
    for (i, input) in inputs.iter().enumerate() {
        if review["frames"][i]["raw_sha256"] != input.hash
            || review["frames"][i]["timestamp_ns"]
                != input.meta["timestamp_ns"]
                    .as_u64()
                    .ok_or("timestamp")?
                    .to_string()
        {
            return Err("RAW lineage mismatch".into());
        }
        let rgb = input.rgb.as_ref().ok_or("RAW RGB")?;
        let u = reduced(rgb, 24);
        let f = reduced(rgb, 8);
        // Display the actual linear RGB fitted above; no separately processed thumbnails.
        frames.push(frame(out, i, &u, &f, input.time, &input.hash)?);
        image(out, i, input)?;
        ultra.push(u);
        coarse.push(f);
    }
    if args[1] == "--z-motion-groups" || args[1] == "--z-motion-history" {
        let first_origin = origin(&inputs[0].meta);
        let crops = inputs
            .iter()
            .map(|v| {
                let o = origin(&v.meta);
                [
                    (first_origin[0] - o[0]) / 24.,
                    (first_origin[1] - o[1]) / 24.,
                ]
            })
            .collect::<Vec<_>>();
        let runner = if args[1] == "--z-motion-history" {
            groups::run_history
        } else {
            groups::run
        };
        return runner(
            out,
            &ultra,
            &coarse,
            frames,
            &crops,
            json!({"review":source,"review_sha256":digest(&bytes),"eye":eye,"first":first,"last":last,"native_frames":inputs.len(),"synthetic":false}),
        );
    }
    let history = if let Some(path) = args.get(4) {
        let path = Path::new(path).canonicalize()?;
        if !path.starts_with(&root) {
            return Err("checked historical output required".into());
        }
        let bytes = fs::read(&path)?;
        let d: Value = serde_json::from_slice(&bytes)?;
        if d["frames"].as_array().map(Vec::len) != Some(frames.len())
            || frames
                .iter()
                .enumerate()
                .any(|(i, f)| d["frames"][i]["raw_sha256"] != f["raw_sha256"])
        {
            return Err("historical native source mismatch".into());
        }
        Some((d, json!({"path":path,"sha256":digest(&bytes)})))
    } else {
        None
    };
    let mut pairs = Vec::new();
    for i in 1..inputs.len() {
        let r = (i - 1) / 10 * 10;
        let a = origin(&inputs[r].meta);
        let b = origin(&inputs[i].meta);
        let mut result = pair(
            &ultra[r],
            &ultra[i],
            &coarse[r],
            &coarse[i],
            [(a[0] - b[0]) / 24., (a[1] - b[1]) / 24.],
            r,
            i,
        );
        if let Some((old, _)) = &history {
            let old = &old["pairs"][i - 1];
            if old["reference"] != r || old["target"] != i {
                return Err("historical pair alignment mismatch".into());
            }
            for (level, a, b) in [
                ("ultra", &ultra[r], &ultra[i]),
                ("coarse", &coarse[r], &coarse[i]),
            ] {
                let model: motion::Model =
                    serde_json::from_value(old[level]["retained"][0]["model"].clone())?;
                result[level]["historical_relative_held"] =
                    serde_json::to_value(motion::fidelity(a, b, model))?;
            }
        }
        pairs.push(result);
    }
    finish(
        out,
        frames,
        pairs,
        json!({"review":source,"review_sha256":digest(&bytes),"eye":eye,"first":first,"last":last,"native_frames":inputs.len(),"synthetic":false,"historical":history.as_ref().map(|h|&h.1)}),
    )
}
fn finish(out: &Path, frames: Vec<Value>, pairs: Vec<Value>, source: Value) -> Result<()> {
    let avg = |level: &str, method: &str, key: &str| {
        pairs
            .iter()
            .map(|p| p[level][method][key].as_f64().unwrap())
            .sum::<f64>()
            / pairs.len() as f64
    };
    let mut metrics = json!({});
    for level in ["ultra", "coarse"] {
        metrics[level] = json!({"baseline_loss":avg(level,"baseline_held","loss"),"candidate_loss":avg(level,"candidate_held","loss"),"baseline_support":avg(level,"baseline_held","support"),"candidate_support":avg(level,"candidate_held","support"),"layered_loss_diagnostic":avg(level,"layered_held","loss"),"improved_pairs":pairs.iter().filter(|p|p[level]["candidate_held"]["loss"].as_f64().unwrap()<p[level]["baseline_held"]["loss"].as_f64().unwrap()).count(),"pairs":pairs.len()});
    }
    let report = json!({"schema":"perspective-motion-experiment-v1","source":source,"metrics":metrics,"provenance":{"recipe":digest(include_bytes!("z_motion3d.rs")),"math":digest(include_bytes!("z_motion3d_math.rs")),"viewer":digest(include_bytes!("z_motion3d_viewer.html")),"downsample":digest(include_bytes!("iris_pivot_binary.rs")),"raw_decode":digest(include_bytes!("warp_probe.rs")),"rgb_decode":digest(include_bytes!("iris_pivot_color.rs"))},
        "limitations":"Offline pairwise experiment, not a live tracker or calibrated head/eye pose. Three-axis rigid 3D rotation plus 3D translation and a quadratic reference depth surface, projected by a pinhole camera. Fixed pivot is a coordinate gauge: free pivot and translation are not independently observable. Focal hypotheses 1/2/4 image widths and image-center principal point are assumptions; RAW ROI crop shifts are accounted separately. 18x12 initial Gaussian-blurred search (24x native), then 53x35 (8x native), exact coordinate scale 3, not rounded-dimension ratio. Local multistart coordinate refinement and score-slack retention are heuristic, not certified bounds. Dominant group is not labeled face. Alternate residual group has no anatomical identity or proven depth order; unmatched includes occlusion, lighting and model error. Fitting combines 60% globally channel-normalized Huber color differences with 40% local NCC, trims 10% of valid fit patches and penalizes lost overlap; at least half of the fit patches must overlap. This still admits repeated-texture aliases. Patches are spatially split for fitting/evaluation, but blur correlates folds. Layered score chooses the better residual model on evaluated patches and is diagnostic only. No human correspondence labels, metric scale, camera calibration, temporal identity persistence or new-user/FPS claims. SN-FEIDA not applicable because no limbus is fit."});
    let data = json!({"report":report,"frames":frames,"pairs":pairs});
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&data["report"])?,
    )?;
    fs::write(out.join("motion.json"), serde_json::to_vec(&data)?)?;
    fs::write(
        out.join("viewer.html"),
        include_str!("z_motion3d_viewer.html")
            .replace("MOTION_DATA", &serde_json::to_string(&data)?),
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&data["report"]["metrics"])?
    );
    Ok(())
}
// Independent ray/sphere renderer: no use of the fitted quadratic depth surface or map.
fn sphere(yaw: f64, pitch: f64, shift: f64) -> Picture {
    let (w, h) = (159, 105);
    let f = 318.;
    let center = [shift, 0., 1.7];
    let radius: f64 = 0.9;
    std::array::from_fn(|c| {
        let mut v = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let ray = [(x as f64 - 79.) / f, (y as f64 - 52.) / f, 1.];
                let aa = ray.iter().map(|v| v * v).sum::<f64>();
                let b = ray.iter().zip(center).map(|(a, b)| a * b).sum::<f64>();
                let cc = center.iter().map(|v| v * v).sum::<f64>() - radius * radius;
                let d = b * b - aa * cc;
                if d <= 0. {
                    v.push(0.);
                    continue;
                }
                let t = (b - d.sqrt()) / aa;
                let q: [f64; 3] = std::array::from_fn(|i| ray[i] * t - center[i]);
                let (sy, cy) = yaw.sin_cos();
                let (sp, cp) = pitch.sin_cos();
                let r = [cy * q[0] - sy * q[2], q[1], sy * q[0] + cy * q[2]];
                let u = r[0];
                let vv = cp * r[1] + sp * r[2];
                let z = -sp * r[1] + cp * r[2];
                let phase = c as f64 * 0.8;
                v.push(
                    (0.5 + 0.15 * (u * 58. + phase).sin()
                        + 0.12 * (vv * 47. - phase).cos()
                        + 0.1 * ((u + vv) * 83. + z * 7. + phase).sin())
                    .clamp(0., 1.),
                );
            }
        }
        Image { w, h, v }
    })
}
// Ground-truth projected correspondences from the independently rendered sphere.
fn truth_error(level: &Level, yaw: f64, pitch: f64, shift: f64) -> Value {
    let scale = if level.width == 18 { 9. } else { 3. };
    let mut e2 = Vec::new();
    let mut e3 = Vec::new();
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    for o in &level.observations {
        let ray = [
            (o.p[0] * scale - 79.) / 318.,
            (o.p[1] * scale - 52.) / 318.,
            1.,
        ];
        let aa = ray.iter().map(|v| v * v).sum::<f64>();
        let bb = 1.7;
        let d = bb * bb - aa * (1.7 * 1.7 - 0.9 * 0.9);
        if d <= 0. {
            continue;
        }
        let t = (bb - d.sqrt()) / aa;
        let q = [ray[0] * t, ray[1] * t, t - 1.7];
        let r = [q[0], cp * q[1] - sp * q[2], sp * q[1] + cp * q[2]];
        let n = [cy * r[0] + sy * r[2], r[1], -sy * r[0] + cy * r[2]];
        let target = [n[0] + shift, n[1], n[2] + 1.7];
        if n.iter().zip(target).map(|(a, b)| a * b).sum::<f64>() >= 0. {
            continue;
        }
        let truth = [
            (318. * target[0] / target[2] + 79.) / scale,
            (318. * target[1] / target[2] + 52.) / scale,
        ];
        if truth[0] < 1.
            || truth[1] < 1.
            || truth[0] >= level.width as f64 - 2.
            || truth[1] >= level.height as f64 - 2.
        {
            continue;
        }
        for (model, errors) in [
            (level.baseline.model, &mut e2),
            (level.retained[0].model, &mut e3),
        ] {
            let projected = model.prepare().map(o.p).unwrap();
            errors.push((projected[0] - truth[0]).hypot(projected[1] - truth[1]));
        }
    }
    json!({"visible_samples":e2.len(),"baseline_mean_pixel_error":e2.iter().sum::<f64>()/e2.len().max(1)as f64,"candidate_mean_pixel_error":e3.iter().sum::<f64>()/e3.len().max(1)as f64})
}
fn synthetic(out: &Path) -> Result<()> {
    let mut frames = Vec::new();
    let mut pairs = Vec::new();
    for (i, (yaw, pitch, shift)) in [
        (0.08, 0.05, 0.01),
        (-0.08, -0.05, -0.01),
        (0.18, 0.12, 0.015),
        (-0.18, -0.12, -0.015),
        (0., 0., 0.015),
    ]
    .into_iter()
    .enumerate()
    {
        let a = sphere(0., 0., 0.);
        let b = sphere(yaw, pitch, shift);
        let ua = reduced(&a, 9);
        let ub = reduced(&b, 9);
        let fa = reduced(&a, 3);
        let fb = reduced(&b, 3);
        for (j, (u, f, raw)) in [(&ua, &fa, &a), (&ub, &fb, &b)].into_iter().enumerate() {
            let n = 2 * i + j;
            frames.push(frame(
                out,
                n,
                u,
                f,
                n as f64 * 0.03,
                "independent-ray-sphere",
            )?);
            png(out, &format!("raw-{n:03}.png"), raw)?;
        }
        let ultra = motion::evaluate(&ua, &ub, [0.; 2], None);
        let coarse = motion::evaluate(&fa, &fb, [0.; 2], Some(&ultra));
        let error = truth_error(&coarse, yaw, pitch, shift);
        let ultra_error = truth_error(&ultra, yaw, pitch, shift);
        let mut p = json!({"reference":2*i,"target":2*i+1,"ultra":ultra,"coarse":coarse,"reprojection_truth":error,"ultra_reprojection_truth":ultra_error});
        p["truth"] = json!({"yaw":yaw,"pitch":pitch,"translation_x":shift,"surface":"sphere, independently ray-intersected"});
        pairs.push(p);
    }
    finish(
        out,
        frames,
        pairs,
        json!({"synthetic":true,"cases":"opposite small and larger out-of-plane rotations and translation-only control"}),
    )
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    #[test]
    fn small_known_turns_match_at_eighteen_pixels() {
        for sign in [-1., 1.] {
            let a = reduced(&sphere(0., 0., 0.), 9);
            let b = reduced(&sphere(sign * 0.08, sign * 0.05, sign * 0.01), 9);
            let level = motion::evaluate(&a, &b, [0.; 2], None);
            let error = truth_error(&level, sign * 0.08, sign * 0.05, sign * 0.01);
            assert!(
                error["candidate_mean_pixel_error"].as_f64().unwrap() < 0.08,
                "{error}"
            );
        }
    }
}
