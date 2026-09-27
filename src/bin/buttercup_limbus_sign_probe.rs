//! CPU-only, evaluation-only limbus asymmetry probe. Never changes live geometry.
use buttercup_eye_tracking::{
    geometry::Ellipse,
    limbus_refiner::{self, Context, Model},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn plane(p: &[f64], xy: [f64; 2]) -> Option<[f64; 3]> {
    let ray = [(xy[0] - 4000.) / 4000., (xy[1] - 3000.) / 4000., 1.];
    let c = [p[0], p[1], p[2]];
    let n = [p[3], p[4], p[5]];
    let den = dot(n, ray);
    if den.abs() < 1e-8 {
        return None;
    }
    let t = dot(n, c) / den;
    (t > 0.).then(|| ray.map(|v| v * t))
}
fn geometry(p: &[f64], xy: [f64; 2], normal: (f64, f64)) -> Option<[f64; 2]> {
    let x = plane(p, xy)?;
    let c = [p[0], p[1], p[2]];
    let view = c.map(|v| v / norm(c));
    let depth = -dot([x[0] - c[0], x[1] - c[1], x[2] - c[2]], view);
    let a = plane(p, [xy[0] - normal.0 * 0.5, xy[1] - normal.1 * 0.5])?;
    let b = plane(p, [xy[0] + normal.0 * 0.5, xy[1] + normal.1 * 0.5])?;
    let r = |v: [f64; 3]| norm([v[0] - c[0], v[1] - c[1], v[2] - c[2]]);
    Some([depth, (r(b) - r(a)).abs()])
}
fn sample(gray: &[f64], w: usize, h: usize, x: f64, y: f64) -> Option<f64> {
    let x = x / 4. - 0.5;
    let y = y / 4. - 0.5;
    if x < 0. || y < 0. || x >= (w - 1) as f64 || y >= (h - 1) as f64 {
        return None;
    }
    let ix = x as usize;
    let iy = y as usize;
    let a = x - ix as f64;
    let b = y - iy as f64;
    Some(
        (1. - b) * ((1. - a) * gray[iy * w + ix] + a * gray[iy * w + ix + 1])
            + b * ((1. - a) * gray[(iy + 1) * w + ix] + a * gray[(iy + 1) * w + ix + 1]),
    )
}
fn raw_profile(gray: &[f64], w: usize, h: usize, p: (f64, f64), n: (f64, f64)) -> Value {
    let mut profile = Vec::new();
    for u in -24..=24 {
        let mut v = 0.;
        for t in -2..=2 {
            let Some(s) = sample(
                gray,
                w,
                h,
                p.0 + n.0 * u as f64 - n.1 * t as f64 * 2.,
                p.1 + n.1 * u as f64 + n.0 * t as f64 * 2.,
            ) else {
                return json!({"accepted":false,"reason":"crop-edge"});
            };
            v += s / 5.;
        }
        profile.push(v);
    }
    let inner = profile[..8].iter().sum::<f64>() / 8.;
    let outer = profile[41..].iter().sum::<f64>() / 8.;
    let gradient: Vec<_> = profile.windows(2).map(|v| v[1] - v[0]).collect();
    let up = gradient.iter().map(|v| v.max(0.)).sum::<f64>();
    let down = gradient.iter().map(|v| (-v).max(0.)).sum::<f64>();
    let quant = |q: f64| {
        let mut s = 0.;
        for (i, v) in gradient.iter().enumerate() {
            s += v.max(0.);
            if s >= q * up {
                return i as f64 - 23.5;
            }
        }
        23.5
    };
    let lo = quant(0.1);
    let mid = quant(0.5);
    let hi = quant(0.9);
    let width = hi - lo;
    let reason = if outer - inner < 0.04 {
        "low-or-reversed-contrast"
    } else if profile.iter().any(|v| *v > 0.98) {
        "saturation"
    } else if down > 0.3 * up {
        "nonmonotonic"
    } else if mid.abs() > 8. || lo < -19. || hi > 19. {
        "edge-not-localized"
    } else if !(2. ..=30.).contains(&width) {
        "width-outside-probe"
    } else {
        "admitted"
    };
    json!({"accepted":reason=="admitted","reason":reason,"profile":profile,"contrast":outer-inner,"negative_gradient_fraction":down/up.max(1e-8),"low_px":lo,"high_px":hi,"mid_px":mid,"width_px":width})
}
fn regress(data: &[(f64, f64, usize)], exclude: Option<usize>) -> Value {
    let d: Vec<_> = data
        .iter()
        .filter(|(_, _, sector)| Some(sector / 6) != exclude)
        .collect();
    let n = d.len();
    if n < 8 {
        return json!({"supported":false,"sectors":n});
    }
    let x = d.iter().map(|r| r.0).sum::<f64>() / n as f64;
    let y = d.iter().map(|r| r.1).sum::<f64>() / n as f64;
    let xx = d.iter().map(|r| (r.0 - x).powi(2)).sum::<f64>();
    let yy = d.iter().map(|r| (r.1 - y).powi(2)).sum::<f64>();
    let xy = d.iter().map(|r| (r.0 - x) * (r.1 - y)).sum::<f64>();
    let slope = xy / xx.max(1e-12);
    let corr = xy / (xx * yy).sqrt().max(1e-12);
    let near = d.iter().filter(|r| r.0 > 0.2).count();
    let far = d.iter().filter(|r| r.0 < -0.2).count();
    json!({"supported":near>=3 && far>=3,"sectors":n,"near_sectors":near,"far_sectors":far,"slope":slope,"correlation":corr,"positive_r2":if slope>0.{corr*corr}else{0.},"mean_width_mm_prior":y})
}
fn summarize(samples: &[Value], method: usize) -> Value {
    let mut data = [Vec::new(), Vec::new()];
    for branch in 0..2 {
        for sector in 0..24 {
            let vals: Vec<_> = samples
                .iter()
                .filter(|s| s["sector"] == sector && s["widths_px"][method].is_number())
                .collect();
            if vals.is_empty() {
                continue;
            }
            let n = vals.len() as f64;
            let x = vals
                .iter()
                .map(|s| s["geometry"][branch][0].as_f64().unwrap())
                .sum::<f64>()
                / n;
            let y = vals
                .iter()
                .map(|s| {
                    s["geometry"][branch][1].as_f64().unwrap()
                        * s["widths_px"][method].as_f64().unwrap()
                })
                .sum::<f64>()
                / n;
            data[branch].push((x, y, sector));
        }
    }
    let fits = [regress(&data[0], None), regress(&data[1], None)];
    let pick = |f: &[Value; 2]| -> Option<usize> {
        if !f.iter().all(|v| v["supported"] == true) {
            return None;
        }
        let a = f[0]["positive_r2"].as_f64()?;
        let b = f[1]["positive_r2"].as_f64()?;
        if a.max(b) < 0.1225 {
            return None;
        }
        Some(usize::from(b > a))
    };
    let candidate = pick(&fits);
    let leave: Vec<_> = (0..4)
        .map(|q| pick(&[regress(&data[0], Some(q)), regress(&data[1], Some(q))]))
        .collect();
    let stable = candidate.is_some() && leave.iter().all(|v| *v == candidate);
    json!({"fits":fits,"candidate":candidate,"leave_quadrant_out":leave,"stable_candidate":if stable{candidate}else{None},"measurement_count":samples.iter().filter(|s|s["widths_px"][method].is_number()).count()})
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: buttercup_limbus_sign_probe ELLIPSE_DIR TARGET_SELECTION NEW_OUTPUT_DIR NEIGHBOR_RADIUS".into());
    }
    let base = Path::new(&args[1]);
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    let radius: i64 = args[4].parse()?;
    if !(0..=5).contains(&radius) {
        return Err("radius must be 0..5".into());
    }
    let read = |n: &str| -> Result<Vec<Value>> {
        fs::read_to_string(base.join(n))?
            .lines()
            .map(|s| Ok(serde_json::from_str(s)?))
            .collect()
    };
    let shapes = read("ellipse-only.jsonl")?;
    let sides = read("evaluation-sidecar.jsonl")?;
    let estimates = read("estimates.jsonl")?;
    let selection: Vec<Value> = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
    let paths = [
        "data/models/limbus_refiner_v1.json",
        "outputs/limbus-pairs-20260914/cold-cpu-v3/full-augmented.json",
    ];
    let models: Vec<_> = paths
        .iter()
        .map(|p| Model::load(Path::new(p)))
        .collect::<std::result::Result<_, _>>()?;
    let methods = [
        "raw_edge_spread",
        "legacy_band",
        "legacy_optical",
        "cpu_augmented_band",
        "cpu_augmented_optical",
        "legacy_pixels_suppressed",
        "cpu_pixels_suppressed",
        "constant_11px_augmented_support",
    ];
    let mut results = Vec::new();
    let mut missing = Vec::new();
    for sel in &selection {
        for delta in -radius..=radius {
            for eye in 1..=2 {
                let seq = sel["sequence"].as_i64().unwrap() + delta;
                let Some(s) = shapes.iter().find(|r| {
                    r["capture"] == sel["capture"] && r["sequence"] == seq && r["eye"] == eye
                }) else {
                    missing.push(json!({"selection":sel,"delta":delta,"eye":eye,"reason":"missing-ellipse-row"}));
                    continue;
                };
                let id = s["id"].as_u64().unwrap();
                let Some(est) = estimates
                    .iter()
                    .find(|e| e["id"] == id && e["poses"].is_array())
                else {
                    missing.push(json!({"id":id,"reason":"missing-poses"}));
                    continue;
                };
                let side = sides.iter().find(|r| r["id"] == id).unwrap();
                let inp = &side["input"];
                let f = &inp["frame"];
                assert_eq!(s["source_ns"], f["timestamp_ns"]);
                assert_eq!(s["clock"], inp["clock_lineage"]);
                let w = f["width"].as_u64().unwrap() as usize;
                let h = f["height"].as_u64().unwrap() as usize;
                let sx = f["sensor_x"].as_f64().unwrap();
                let sy = f["sensor_y"].as_f64().unwrap();
                let el = &s["ellipse"];
                let e = Ellipse {
                    center: (
                        el["center_sensor_px"][0].as_f64().unwrap() - sx,
                        el["center_sensor_px"][1].as_f64().unwrap() - sy,
                    ),
                    major_radius: el["a"].as_f64().unwrap(),
                    minor_radius: el["b"].as_f64().unwrap(),
                    angle: el["angle"].as_f64().unwrap(),
                };
                let mut file = fs::File::open(inp["raw_file"].as_str().unwrap())?;
                file.seek(SeekFrom::Start(inp["raw_offset"].as_u64().unwrap()))?;
                let mut bytes = vec![0; inp["raw_length"].as_u64().unwrap() as usize];
                file.read_exact(&mut bytes)?;
                assert_eq!(bytes.len(), w * h * 5 / 4);
                assert_eq!((w % 4, h % 4), (0, 0));
                let mut raw = Vec::with_capacity(w * h);
                for g in bytes.chunks_exact(5) {
                    let v = g
                        .iter()
                        .enumerate()
                        .fold(0u64, |a, (i, b)| a | ((*b as u64) << (8 * i)));
                    for k in 0..4 {
                        raw.push(((v >> (10 * k)) & 1023) as u16);
                    }
                }
                let mut gray = vec![0.; w * h / 16];
                for y in 0..h / 4 {
                    for x in 0..w / 4 {
                        for yy in 0..4 {
                            for xx in 0..4 {
                                gray[y * (w / 4) + x] +=
                                    raw[(y * 4 + yy) * w + x * 4 + xx] as f64 / (16. * 1023.);
                            }
                        }
                    }
                }
                let poses: Vec<Vec<f64>> = est["poses"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| {
                        p.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect()
                    })
                    .collect();
                let retained = side["retained"]
                    .as_array()
                    .ok_or("missing observed retained points")?;
                let count = retained.len().min(96);
                let mut samples = Vec::new();
                for k in 0..count {
                    let p = &retained[k * retained.len() / count];
                    let pt = (p[0].as_f64().unwrap(), p[1].as_f64().unwrap());
                    let Some(n) = limbus_refiner::normal_at(e, pt) else {
                        continue;
                    };
                    let Some(patch) =
                        limbus_refiner::extract_patch(&raw, w, h, e, pt, Context::default(), 2.)
                    else {
                        continue;
                    };
                    let Some(g0) = geometry(&poses[0], [pt.0 + sx, pt.1 + sy], n) else {
                        continue;
                    };
                    let Some(g1) = geometry(&poses[1], [pt.0 + sx, pt.1 + sy], n) else {
                        continue;
                    };
                    let angle = (pt.1 - e.center.1)
                        .atan2(pt.0 - e.center.0)
                        .rem_euclid(std::f64::consts::TAU);
                    let sector = (angle * 24. / std::f64::consts::TAU) as usize;
                    let profile = raw_profile(&gray, w / 4, h / 4, pt, n);
                    let mut widths = vec![if profile["accepted"] == true {
                        profile["width_px"].as_f64()
                    } else {
                        None
                    }];
                    let mut learned = Vec::new();
                    let mut suppressed = Vec::new();
                    for model in &models {
                        let pred = model.predict(&patch).ok_or("nonfinite model prediction")?;
                        let width = |a: usize, b: usize| -> Option<f64> {
                            let d = pred[b].offset_px - pred[a].offset_px;
                            (pred[a].supported()
                                && pred[b].supported()
                                && patch.contrast >= 0.015
                                && patch.saturated_fraction < 0.2
                                && d > 0.
                                && d < 30.)
                                .then_some(d)
                        };
                        widths.push(width(1, 2));
                        widths.push(width(3, 5));
                        let mut control = patch.clone();
                        control.features[..256].fill(0.);
                        let control_pred = model
                            .predict(&control)
                            .ok_or("nonfinite control prediction")?;
                        suppressed.push(
                            width(1, 2)
                                .map(|_| control_pred[2].offset_px - control_pred[1].offset_px),
                        );
                        learned.push(json!(pred.iter().map(|p|json!({"offset":p.offset_px,"spread":p.spread_px,"visible_mass":p.visible_mass,"supported":p.supported()})).collect::<Vec<_>>()));
                    }
                    widths.extend(suppressed);
                    widths.push(widths[3].map(|_| 11.));
                    samples.push(json!({"point":pt,"normal":n,"sector":sector,"geometry":[g0,g1],"raw":profile,"learned":learned,"widths_px":widths,"patch_contrast":patch.contrast}));
                }
                let summaries: Vec<_> =
                    (0..methods.len()).map(|m| summarize(&samples, m)).collect();
                let row = json!({"id":id,"selection":sel,"neighbor_delta":delta,"eye":eye,"sequence":seq,"input":inp,"raw_sha256":format!("{:x}",Sha256::digest(&bytes)),"ellipse":s["ellipse"],"poses":poses,"samples":samples,"methods":summaries});
                eprintln!(
                    "{} seq {seq} eye {eye}: counts {:?}, stable {:?}",
                    sel["capture"],
                    summaries
                        .iter()
                        .map(|v| &v["measurement_count"])
                        .collect::<Vec<_>>(),
                    summaries
                        .iter()
                        .map(|v| &v["stable_candidate"])
                        .collect::<Vec<_>>()
                );
                results.push(row);
            }
        }
    }
    let model_manifest:Vec<_>=paths.iter().zip(&models).map(|(p,m)|json!({"path":p,"sha256":format!("{:x}",Sha256::digest(fs::read(p).unwrap())),"training":m.manifest["training"]})).collect();
    fs::write(
        out.join("results.json"),
        serde_json::to_vec(
            &json!({"methods":methods,"models":model_manifest,"frames":results,"missing":missing,"scope":"evaluation only; legacy ellipse/custom model ancestry not promoted; widths are optical image features, not measured anatomical thickness; no independent sign ground truth; no geometry or area changed","geometry":"positive near-depth means closer than iris center; width converted through each candidate plane with radius6mm and fx=fy4000 prior; not independent physical scale","gate":"at least 8 angular sectors, at least 3 each side of +/-0.2mm depth; positive width-nearness correlation >0.35 and same candidate after leaving out each quadrant; heuristic, not probability"}),
        )?,
    )?;
    Ok(())
}
