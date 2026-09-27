//! RAW-backed low resolution motion/layer hypothesis experiment.
use super::{
    binary, digest, image, json, load, origin, BundleSource, Canvas, Image, Result, Value, P,
};
use std::{fs, path::Path};
#[path = "z_discovery_math.rs"]
mod engine;
use engine::{Motion, Picture};

fn png(out: &Path, name: &str, p: &Picture) -> Result<()> {
    let mut bytes = Vec::new();
    for k in 0..p.w() * p.h() {
        for c in [2, 1, 0] {
            bytes.push((p.channels[c].v[k].clamp(0., 1.) * 255.).round() as u8);
        }
        bytes.push(255);
    }
    let mut canvas = Canvas::new(p.w(), p.h())?;
    canvas.image(&bytes, p.w(), p.h(), 0., 0., p.w() as f64, p.h() as f64);
    canvas.png(&out.join(name))?;
    Ok(())
}
fn display(out: &Path, i: usize, raw: &super::Input, factor: usize) -> Result<()> {
    let (w, h) = (raw.image.w, raw.image.h);
    let mask = vec![true; w * h];
    let channels = std::array::from_fn(|c| {
        let im = Image {
            w,
            h,
            v: raw
                .preview
                .chunks_exact(4)
                .map(|p| p[2 - c] as f64 / 255.)
                .collect(),
        };
        binary::downsample_masked(&im, &mask, factor).image
    });
    png(
        out,
        &format!("level{factor}-{i:03}.png"),
        &Picture { channels },
    )
}
fn serialize_picture(p: &Picture) -> Value {
    json!(p
        .channels
        .iter()
        .map(|c| c
            .v
            .iter()
            .map(|v| (v * 65535.).round() as u16)
            .collect::<Vec<_>>())
        .collect::<Vec<_>>())
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("usage: --z-discovery SAVED_IRIS_REVIEW_JSON|synthetic NEW_OUTPUT".into());
    }
    let out = Path::new(&args[3]);
    let root = fs::canonicalize("outputs")?;
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
    if args[2] == "synthetic" {
        return synthetic(out);
    }
    let source = Path::new(&args[2]).canonicalize()?;
    if !source.starts_with(&root) {
        return Err("checked source review required".into());
    }
    let bytes = fs::read(&source)?;
    let review: Value = serde_json::from_slice(&bytes)?;
    let report = &review["report"];
    let cfg = &report["config"];
    let eye = cfg["eye"].as_u64().ok_or("eye")?;
    let first = cfg["first"].as_u64().ok_or("first")? as usize;
    let last = cfg["last"].as_u64().ok_or("last")? as usize;
    let bundle = BundleSource::open(Path::new(report["bundle"].as_str().ok_or("bundle")?))?;
    let metas = String::from_utf8(bundle.read_entry("frames.jsonl")?)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|m| m["eye_id"] == eye)
        .collect::<Vec<_>>();
    if first >= last || last >= metas.len() || last - first > 60 {
        return Err("bounded source interval required".into());
    }
    let clock = metas[0]["timestamp_ns"].as_u64().ok_or("clock")?;
    let inputs = metas[first..=last]
        .iter()
        .map(|m| load(&bundle, m, clock))
        .collect::<Result<Vec<_>>>()?;
    if review["frames"].as_array().map(Vec::len) != Some(inputs.len()) {
        return Err("source count mismatch".into());
    }
    let mut pictures = Vec::new();
    let mut fine_pictures = Vec::new();
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
        let mask = vec![true; input.image.w * input.image.h];
        let rgb = input.rgb.as_ref().ok_or("raw rgb")?;
        let picture = Picture {
            channels: std::array::from_fn(|c| binary::downsample_masked(&rgb[c], &mask, 8).image),
        };
        image(out, i, input)?;
        display(out, i, input, 8)?;
        display(out, i, input, 4)?;
        let fine = Picture {
            channels: std::array::from_fn(|c| binary::downsample_masked(&rgb[c], &mask, 4).image),
        };
        frames.push(json!({"index":i,"time_s":input.time,"timestamp_ns":input.meta["timestamp_ns"].as_u64().unwrap().to_string(),"raw_sha256":input.hash,"image":format!("level8-{i:03}.png"),"fine_image":format!("level4-{i:03}.png"),"fine_width":fine.w(),"fine_height":fine.h(),"native_image":format!("raw-{i:03}.png"),"channels":serialize_picture(&picture)}));
        pictures.push(picture);
        fine_pictures.push(fine);
    }
    let mut pairs = Vec::new();
    for i in 1..inputs.len() {
        let reference = ((i - 1) / 10) * 10;
        let a = origin(&inputs[reference].meta);
        let b = origin(&inputs[i].meta);
        let crop = [(a[0] - b[0]) / 8., (a[1] - b[1]) / 8.];
        let result = engine::discover(&pictures[reference], &pictures[i], crop, 3);
        eprintln!(
            "eye {eye} pair {reference}->{i}: {} matches, {} groups",
            result.matches.len(),
            result.iterations.last().unwrap().motions.len()
        );
        let fine_result = engine::refine_level(
            &fine_pictures[reference],
            &fine_pictures[i],
            crop.map(|v| v * 2.),
            &result,
            2.,
            3,
        );
        pairs.push(json!({"reference":reference,"target":i,"crop":crop,"result":result,"fine_result":fine_result}));
    }
    finish(
        out,
        json!({"source_review":source,"source_review_sha256":digest(&bytes),"eye":eye,"first":first,"last":last,"native_frames":inputs.len(),"synthetic":false}),
        pictures[0].w(),
        pictures[0].h(),
        frames,
        pairs,
    )
}
fn finish(
    out: &Path,
    source: Value,
    w: usize,
    h: usize,
    frames: Vec<Value>,
    pairs: Vec<Value>,
) -> Result<()> {
    let mean =
        |f: &dyn Fn(&Value) -> f64| pairs.iter().map(f).sum::<f64>() / pairs.len().max(1) as f64;
    let last = |p: &Value| {
        p["result"]["iterations"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .clone()
    };
    let report = json!({"schema":"whole-image-z-discovery-v1","source":source,"pairs":pairs.len(),"width":w,"height":h,
        "mean_candidate_mask_coverage":mean(&|p|last(p)["mask_coverage"].as_f64().unwrap()),
        "mean_fine_mask_coverage":mean(&|p|p["fine_result"]["iterations"].as_array().unwrap().last().unwrap()["mask_coverage"].as_f64().unwrap()),
        "mean_fine_candidate_loss":mean(&|p|p["fine_result"]["iterations"].as_array().unwrap().last().unwrap()["held_loss"].as_f64().unwrap()),
        "mean_fine_candidate_coverage":mean(&|p|p["fine_result"]["iterations"].as_array().unwrap().last().unwrap()["coverage"].as_f64().unwrap()),
        "mean_fine_crop_loss":mean(&|p|p["fine_result"]["baseline_loss"].as_f64().unwrap()),
        "mean_fine_crop_coverage":mean(&|p|p["fine_result"]["baseline_coverage"].as_f64().unwrap()),
        "mean_fine_translation_loss":mean(&|p|p["fine_result"]["translation_loss"].as_f64().unwrap()),
        "mean_fine_translation_coverage":mean(&|p|p["fine_result"]["translation_coverage"].as_f64().unwrap()),
        "mean_crop_held_loss":mean(&|p|p["result"]["baseline_loss"].as_f64().unwrap()),"mean_translation_held_loss":mean(&|p|p["result"]["translation_loss"].as_f64().unwrap()),"mean_candidate_held_loss":mean(&|p|last(p)["held_loss"].as_f64().unwrap()),
        "mean_crop_coverage":mean(&|p|p["result"]["baseline_coverage"].as_f64().unwrap()),"mean_translation_coverage":mean(&|p|p["result"]["translation_coverage"].as_f64().unwrap()),"mean_candidate_coverage":mean(&|p|last(p)["coverage"].as_f64().unwrap()),
        "supported_pair_orders":pairs.iter().map(|p|last(p)["ordering"].as_array().unwrap().iter().filter(|r|!r["front"].is_null()).count()).sum::<usize>(),
        "unknown_pair_orders":pairs.iter().map(|p|last(p)["ordering"].as_array().unwrap().iter().filter(|r|r["front"].is_null()).count()).sum::<usize>(),
        "provenance":{"recipe":digest(include_bytes!("z_discovery.rs")),"engine":digest(include_bytes!("z_discovery_math.rs")),"bounds":digest(include_bytes!("z_discovery_bounds.rs")),"viewer":digest(include_bytes!("z_discovery_viewer.html")),"decode":digest(include_bytes!("warp_probe.rs")),"rgb_decode":digest(include_bytes!("iris_pivot_color.rs")),"downsample":digest(include_bytes!("iris_pivot_binary.rs"))},
        "method":"No anatomical ROI, face band, pupil or glint position is read. Whole-image phase-aware RAW RGB, Gaussian sigma4 then factor8. 3x3 per-channel normalized patch correlation; full-image forward/backward local translation seeds; up to three rigid-motion groups; three rounds at factor8 then three at factor4, alternating motion-supported spatial assignments, retained angle/intercept box subdivision fitted only at measured correspondence locations, and ordering permutations. Analytic conservative pivot polygons are derived from b=(I-R)c+t with bounded translation over a whole-image plus half-width exterior domain. No single pivot is selected. Score-based box pruning is heuristic, slack .025 and beam16; pruned counts are reported. Ordering margins are conservative over retained box-center hypotheses, not certified over continuous boxes. Rotation within 15 degrees, residual translation components within 8 coarse pixels. Front/back requires overlapping projected support and same-sign fit/held photometric margins >.04 with >=6 samples each. Unknown relations remain unknown. Refinement iterations are optimization rounds, not additional observations. Fine resolution inherits the retained coarse boxes, source labels and conditional ordering; preceding ordering excludes occluded correspondence locations from the next motion fit, and ordering is recomputed rather than locked; unresolved depth relations are carried as unknown rather than preventing motion refinement.",
        "limitations":"2D projected rigid motion, not metric depth or a 3D eyeball solve. Free translation makes pivot origins nonunique; origin polygons are conditional geometric enclosures. Reference changes every ten targets; group IDs are local to each pair and may swap. Newly exposed texture is unknown. Coverage counts only current patch loss <.36; inferred mask coverage is separate and never counts as fresh support. Reflection/additive lighting may form spurious groups. Spatial folds are correlated by blur and patch overlap; assignments/proposals see all pixels, so held scores diagnose motion refinement but are not independent validation. No human tissue correspondences, labels, independent scale, new-user or FPS claims. Coarse shape agreement is not durable vessel tracking. SN-FEIDA not applicable."});
    let data = json!({"report":report,"width":w,"height":h,"frames":frames,"pairs":pairs});
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&data["report"])?,
    )?;
    fs::write(out.join("discovery.json"), serde_json::to_vec(&data)?)?;
    fs::write(
        out.join("viewer.html"),
        include_str!("z_discovery_viewer.html")
            .replace("DISCOVERY_DATA", &serde_json::to_string(&data)?),
    )?;
    println!("{}", serde_json::to_string_pretty(&data["report"])?);
    Ok(())
}

// Separate renderer; truth never enters the discovery function.
fn scene(w: usize, h: usize, motions: &[Motion], reverse: bool, scale: f64) -> (Picture, Vec<i8>) {
    let mut labels = vec![0i8; w * h];
    let mut channels: [Image; 3] = std::array::from_fn(|_| Image {
        w,
        h,
        v: vec![0.; w * h],
    });
    for y in 0..h {
        for x in 0..w {
            let q = [x as f64 / scale, y as f64 / scale];
            let mut owner = 0;
            let mut p = motions[0].inverse(q);
            for id in if reverse { [2, 1] } else { [1, 2] } {
                let r = motions[id].inverse(q);
                let inside = if id == 1 {
                    ((r[0] - 18.) / 11.).powi(2) + ((r[1] - 17.) / 12.).powi(2) < 1.
                } else {
                    r[0] > 30. && r[0] < 44. && r[1] > 7. && r[1] < 29.
                };
                if inside {
                    owner = id;
                    p = r;
                }
            }
            labels[y * w + x] = owner as i8;
            for c in 0..3 {
                let v = 0.4
                    + 0.12 * (p[0] * (0.35 + owner as f64 * 0.07) + p[1] * 0.23 + c as f64).sin()
                    + 0.11 * (p[1] * 0.71 - p[0] * 0.12 + owner as f64 * 2.).cos()
                    + 0.07 * ((p[0] * p[0] + p[1] * p[1]) * 0.018 + owner as f64).sin();
                channels[c].v[y * w + x] = v;
            }
        }
    }
    (Picture { channels }, labels)
}
fn mirror(p: &mut Picture, labels: &mut [i8]) {
    let (w, h) = (p.w(), p.h());
    for y in 0..h {
        labels[y * w..(y + 1) * w].reverse();
        for c in 0..3 {
            p.channels[c].v[y * w..(y + 1) * w].reverse();
        }
    }
}
fn truth_metrics(d: &engine::Discovery, labels: &[i8], w: usize, h: usize, front: usize) -> Value {
    let result = d.iterations.last().unwrap();
    let mut best = (0usize, [0, 1, 2]);
    let mut total = 0;
    for perm in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let mut correct = 0;
        total = 0;
        for y in 3..h - 3 {
            for x in 3..w - 3 {
                let k = y * w + x;
                total += 1;
                let id = result.labels[k];
                if id >= 0 && perm[id as usize] == labels[k] as usize {
                    correct += 1;
                }
            }
        }
        if correct > best.0 {
            best = (correct, perm);
        }
    }
    let mut correct = 0;
    let mut wrong = 0;
    let mut unknown = 0;
    for e in &result.ordering {
        if let Some(f) = e.front {
            let (a, b) = (best.1[e.a], best.1[e.b]);
            let expected = if a == 0 {
                b
            } else if b == 0 {
                a
            } else {
                front
            };
            if best.1[f] == expected {
                correct += 1;
            } else {
                wrong += 1;
            }
        } else {
            unknown += 1;
        }
    }
    json!({"label_accuracy":best.0 as f64/total as f64,"group_to_truth":best.1,"correct_order_constraints":correct,"wrong_order_constraints":wrong,"unknown_order_constraints":unknown})
}
fn synthetic(out: &Path) -> Result<()> {
    let (w, h) = (53, 35);
    let base = Motion::identity([0., 0.], w, h);
    let mut frames = Vec::new();
    let mut pairs = Vec::new();
    for (reverse, mirrored) in [(false, false), (true, false), (false, true), (true, true)] {
        let reference = frames.len();
        let (mut a, mut labels) = scene(w, h, &[base; 3], reverse, 1.);
        let mut motions = [
            Motion {
                translation: [-0.5, 0.25],
                angle: 1f64.to_radians(),
                ..base
            },
            Motion {
                pivot: [15., 19.],
                translation: [4., 0.],
                angle: 7f64.to_radians(),
                ..base
            },
            Motion {
                pivot: [37., 18.],
                translation: [-3., 0.],
                angle: -5f64.to_radians(),
                ..base
            },
        ];
        let (mut b, mut target_labels) = scene(w, h, &motions, reverse, 1.);
        let (mut af, mut fine_labels) = scene(105, 69, &[base; 3], reverse, 2.);
        let (mut bf, mut fine_target_labels) = scene(105, 69, &motions, reverse, 2.);
        if mirrored {
            mirror(&mut a, &mut labels);
            mirror(&mut b, &mut target_labels);
            mirror(&mut af, &mut fine_labels);
            mirror(&mut bf, &mut fine_target_labels);
            for m in &mut motions {
                m.pivot[0] = 52. - m.pivot[0];
                m.translation[0] = -m.translation[0];
                m.angle = -m.angle;
            }
        }

        for (p, fine) in [(&a, &af), (&b, &bf)] {
            let i = frames.len();
            png(out, &format!("small-{i:03}.png"), p)?;
            png(out, &format!("fine-{i:03}.png"), fine)?;
            frames.push(json!({"index":i,"time_s":i as f64,"image":format!("small-{i:03}.png"),"fine_image":format!("fine-{i:03}.png"),"fine_width":fine.w(),"fine_height":fine.h(),"native_image":format!("fine-{i:03}.png"),"channels":serialize_picture(p)}));
        }
        let result = engine::discover(&a, &b, [0., 0.], 3);
        let fine_result = engine::refine_level(&af, &bf, [0., 0.], &result, 2., 3);
        let coarse_truth = truth_metrics(&result, &labels, w, h, if reverse { 1 } else { 2 });
        let fine_truth = truth_metrics(
            &fine_result,
            &fine_labels,
            105,
            69,
            if reverse { 1 } else { 2 },
        );
        pairs.push(json!({"reference":reference,"target":reference+1,"crop":[0,0],"truth":{"motions":motions,"labels":labels,"target_labels":target_labels,"front":if reverse{1}else{2},"mirrored":mirrored,"best_permutation_label_accuracy":coarse_truth["label_accuracy"],"coarse":coarse_truth,"fine":fine_truth},"result":result,"fine_result":fine_result}));
    }
    finish(
        out,
        json!({"synthetic":true,"description":"Two crossing textured foreground regions plus moving background; both front/back orders and horizontal reflections rendered independently; truth withheld from discovery."}),
        w,
        h,
        frames,
        pairs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn discovered_constraints_are_correct_or_unresolved_in_both_orders_and_positions() {
        for (reverse, mirrored) in [(false, false), (true, false), (false, true), (true, true)] {
            let base = Motion::identity([0., 0.], 53, 35);
            let moves = [
                Motion {
                    translation: [-0.5, 0.25],
                    angle: 1f64.to_radians(),
                    ..base
                },
                Motion {
                    pivot: [15., 19.],
                    translation: [4., 0.],
                    angle: 7f64.to_radians(),
                    ..base
                },
                Motion {
                    pivot: [37., 18.],
                    translation: [-3., 0.],
                    angle: -5f64.to_radians(),
                    ..base
                },
            ];
            let (mut a, mut la) = scene(53, 35, &[base; 3], reverse, 1.);
            let (mut b, mut lb) = scene(53, 35, &moves, reverse, 1.);
            let (mut af, mut lf) = scene(105, 69, &[base; 3], reverse, 2.);
            let (mut bf, mut lbf) = scene(105, 69, &moves, reverse, 2.);
            if mirrored {
                mirror(&mut a, &mut la);
                mirror(&mut b, &mut lb);
                mirror(&mut af, &mut lf);
                mirror(&mut bf, &mut lbf);
            }
            let coarse = engine::discover(&a, &b, [0., 0.], 3);
            let fine = engine::refine_level(&af, &bf, [0., 0.], &coarse, 2., 3);
            let truth = truth_metrics(&fine, &lf, 105, 69, if reverse { 1 } else { 2 });
            assert_eq!(
                truth["wrong_order_constraints"], 0,
                "reverse={reverse},mirror={mirrored}: {truth}"
            );
            assert!(
                truth["correct_order_constraints"].as_u64().unwrap() >= 1,
                "{truth}"
            );
            assert!(truth["label_accuracy"].as_f64().unwrap() > 0.70, "{truth}");
            // Refinement never jumps out of a surviving coarse parameter box.
            for (id, bound) in fine.iterations.last().unwrap().bounds.iter().enumerate() {
                for cell in &bound.cells {
                    assert!(coarse.iterations.last().unwrap().bounds[id]
                        .cells
                        .iter()
                        .any(|parent| cell.angle[0] >= parent.angle[0] - 1e-10
                            && cell.angle[1] <= parent.angle[1] + 1e-10
                            && (0..2).all(|c| cell.intercept[c][0]
                                >= parent.intercept[c][0] * 2. - 1e-10
                                && cell.intercept[c][1] <= parent.intercept[c][1] * 2. + 1e-10)));
                }
            }
        }
    }
    #[test]
    fn renderer_truth_masks_and_motion_recover_both_occlusion_orders() {
        for reverse in [false, true] {
            let base = Motion::identity([0., 0.], 53, 35);
            let moves = [
                Motion {
                    translation: [-0.5, 0.25],
                    angle: 1f64.to_radians(),
                    ..base
                },
                Motion {
                    pivot: [15., 19.],
                    translation: [4., 0.],
                    angle: 7f64.to_radians(),
                    ..base
                },
                Motion {
                    pivot: [37., 18.],
                    translation: [-3., 0.],
                    angle: -5f64.to_radians(),
                    ..base
                },
            ];
            let (a, labels) = scene(105, 69, &[base; 3], reverse, 2.);
            let (b, _) = scene(105, 69, &moves, reverse, 2.);
            let ms = moves.map(|m| Motion {
                pivot: m.pivot.map(|v| v * 2.),
                translation: m.translation.map(|v| v * 2.),
                ..m
            });
            let bank = ms.iter().map(|m| vec![*m]).collect::<Vec<_>>();
            let order = engine::order_evidence(&a, &b, &labels, &ms, &bank);
            let pair = order.iter().find(|p| p.a == 1 && p.b == 2).unwrap();
            assert_eq!(
                pair.front,
                Some(if reverse { 1 } else { 2 }),
                "{}",
                serde_json::to_string(pair).unwrap()
            );
        }
    }
}
