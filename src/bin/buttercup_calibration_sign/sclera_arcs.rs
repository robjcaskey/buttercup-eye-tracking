//! Outer envelopes of predicted sclera, kept separate from their fitted arcs.
use super::{data, n, rows, sam_export, Canvas, Result, CYAN, H, MUTED, ORANGE, W, WHITE};
use buttercup_eye_tracking::raw_motion_octrees::{fit_lid_quadratic, lid_curve_y};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
};

const CENTER: [f64; 2] = [W as f64 / 2., H as f64 / 2.];
const SCALE: [f64; 2] = CENTER;
const BAND: f64 = 3.5;

#[derive(Clone, Serialize, Deserialize)]
struct ArcFit {
    accepted: bool,
    reason: String,
    points: Vec<[f64; 2]>,
    inliers: Vec<bool>,
    coefficients: Option<[f64; 3]>,
    x_range: Option<[f64; 2]>,
    rmse_px: Option<f64>,
    longest_unsupported_gap_px: Option<f64>,
}

fn y(c: [f64; 3], x: f64) -> f64 {
    lid_curve_y(c, CENTER, SCALE, x)
}

/// Remove only connected dust. No iris ellipse, predicted gaze, brightness
/// threshold, temporal state or other prompt's mask enters this extraction.
fn envelope(mask: &[u8]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>, usize) {
    let mut seen = vec![false; W * H];
    let mut keep = vec![false; W * H];
    let mut removed = 0;
    for i in 0..mask.len() {
        if seen[i] || mask[i] < 128 {
            continue;
        }
        let mut queue = vec![i];
        seen[i] = true;
        let mut read = 0;
        while read < queue.len() {
            let j = queue[read];
            read += 1;
            let (x, yy) = (j % W, j / W);
            for next in [
                (x > 0).then_some(j.wrapping_sub(1)),
                (x + 1 < W).then_some(j + 1),
                (yy > 0).then_some(j.wrapping_sub(W)),
                (yy + 1 < H).then_some(j + W),
            ]
            .into_iter()
            .flatten()
            {
                if !seen[next] && mask[next] >= 128 {
                    seen[next] = true;
                    queue.push(next);
                }
            }
        }
        if queue.len() >= 32 {
            for j in queue {
                keep[j] = true;
            }
        } else {
            removed += queue.len();
        }
    }
    let (mut upper, mut lower) = (vec![], vec![]);
    for x in (2..W - 2).step_by(3) {
        let ys: Vec<_> = (0..H).filter(|&yy| keep[yy * W + x]).collect();
        if ys.len() < 4 {
            continue;
        }
        let first = ys[0];
        let last = *ys.last().unwrap();
        // Image/crop boundaries are not eyelid observations.
        if first > 1 {
            let a = mask[(first - 1) * W + x] as f64;
            let b = mask[first * W + x] as f64;
            upper.push([x as f64, first as f64 - 1. + (127.5 - a) / (b - a)]);
        }
        if last + 2 < H {
            let a = mask[last * W + x] as f64;
            let b = mask[(last + 1) * W + x] as f64;
            lower.push([x as f64, last as f64 + (a - 127.5) / (a - b)]);
        }
    }
    (upper, lower, removed)
}

fn admissible(c: [f64; 3], upper: bool, range: [f64; 2]) -> bool {
    // Assumption for these approximately upright ROIs: upper curve opens
    // downward, lower upward (image y increases downward). This helps reject
    // the *inner* iris-facing edge of a lone scleral crescent. It is a prior,
    // not proof that the selected mask or its boundary is anatomical.
    let correct_bend = if upper { c[2] >= -0.005 } else { c[2] <= 0.005 };
    correct_bend
        && c.iter().all(|v| v.is_finite())
        && (0..=20).all(|i| {
            let x = range[0] + (range[1] - range[0]) * i as f64 / 20.;
            let yy = y(c, x);
            yy >= 0. && yy < H as f64
        })
}

fn inliers(c: [f64; 3], p: &[[f64; 2]]) -> Vec<bool> {
    p.iter()
        .map(|p| (p[1] - y(c, p[0])).abs() <= BAND)
        .collect()
}

fn fit(points: Vec<[f64; 2]>, upper: bool) -> ArcFit {
    let mut result = ArcFit {
        accepted: false,
        reason: "too few boundary points".into(),
        inliers: vec![false; points.len()],
        points,
        coefficients: None,
        x_range: None,
        rmse_px: None,
        longest_unsupported_gap_px: None,
    };
    let p = &result.points;
    if p.len() < 18 {
        return result;
    }
    let range = [p[0][0], p.last().unwrap()[0]];
    if range[1] - range[0] < 70. {
        result.reason = "boundary span below 70 px".into();
        return result;
    }
    let mut seed = 0x51C1E4A_u64 + u64::from(upper);
    let mut best: Option<([f64; 3], f64)> = None;
    for _ in 0..384 {
        let mut samples = vec![];
        // Three strata make the hypothesis use separated columns, including
        // both sclera lobes when they are returned in the same SAM mask.
        for part in 0..3 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let start = part * p.len() / 3;
            let end = (part + 1) * p.len() / 3;
            let a = p[start + (seed >> 32) as usize % (end - start)];
            samples.push((a[0], a[1], 1.));
        }
        let Some(c) = fit_lid_quadratic(&samples, CENTER, SCALE) else {
            continue;
        };
        if !admissible(c, upper, range) {
            continue;
        }
        let keep = inliers(c, p);
        let xs: Vec<_> = p
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(p, _)| p[0])
            .collect();
        if xs.len() < 3 {
            continue;
        }
        let score = xs.len() as f64 + 0.03 * (xs.last().unwrap() - xs[0]);
        if best.is_none_or(|(_, old)| score > old) {
            best = Some((c, score));
        }
    }
    let Some((mut c, _)) = best else {
        result.reason = "no curve with the expected bend".into();
        return result;
    };
    for _ in 0..4 {
        let samples: Vec<_> = p
            .iter()
            .zip(inliers(c, p))
            .filter(|(_, k)| *k)
            .map(|(p, _)| (p[0], p[1], 1.))
            .collect();
        let Some(next) = fit_lid_quadratic(&samples, CENTER, SCALE) else {
            break;
        };
        if !admissible(next, upper, range) {
            break;
        }
        c = next;
    }
    let keep = inliers(c, p);
    let retained: Vec<_> = p
        .iter()
        .zip(&keep)
        .filter(|(_, k)| **k)
        .map(|(p, _)| *p)
        .collect();
    let mut bins = [false; 6];
    for a in &retained {
        bins[(((a[0] - range[0]) / (range[1] - range[0]) * 6.).floor() as usize).min(5)] = true;
    }
    if retained.len() >= 2 {
        let xr = [retained[0][0], retained.last().unwrap()[0]];
        result.x_range = Some(xr);
        result.rmse_px = Some(
            (retained
                .iter()
                .map(|p| (p[1] - y(c, p[0])).powi(2))
                .sum::<f64>()
                / retained.len() as f64)
                .sqrt(),
        );
        result.longest_unsupported_gap_px = Some(
            retained
                .windows(2)
                .map(|v| (v[1][0] - v[0][0] - 3.).max(0.))
                .fold(0., f64::max),
        );
        result.accepted = retained.len() >= 18
            && retained.len() * 2 >= p.len()
            && xr[1] - xr[0] >= 70.
            && bins.iter().filter(|&&v| v).count() >= 4;
    }
    result.reason = if result.accepted {
        "supported by predicted mask points"
    } else {
        "insufficient distributed inliers"
    }
    .into();
    result.coefficients = Some(c);
    result.inliers = keep;
    result
}

fn analyze(mask: &[u8]) -> Value {
    let (upper, lower, dust) = envelope(mask);
    let mut upper = fit(upper, true);
    let mut lower = fit(lower, false);
    let mut crossed = false;
    if upper.accepted && lower.accepted {
        let (a, b) = (upper.x_range.unwrap(), lower.x_range.unwrap());
        let (lo, hi) = (a[0].max(b[0]), a[1].min(b[1]));
        if hi > lo {
            crossed = (0..=32).any(|i| {
                let x = lo + (hi - lo) * i as f64 / 32.;
                y(upper.coefficients.unwrap(), x) + 3. >= y(lower.coefficients.unwrap(), x)
            });
        }
        if crossed {
            for arc in [&mut upper, &mut lower] {
                arc.accepted = false;
                arc.reason = "upper/lower curves cross or leave no opening".into();
            }
        }
    }
    json!({"upper":upper,"lower":lower,"removed_dust_pixels":dust,"crossed":crossed})
}

pub(super) fn draw(canvas: &mut Canvas, x: f64, yy: f64, arcs: &Value) {
    draw_layers(canvas, x, yy, arcs, true, true);
}

fn draw_layers(canvas: &mut Canvas, x: f64, yy: f64, arcs: &Value, points: bool, curves: bool) {
    for (key, color) in [("upper", ORANGE), ("lower", CYAN)] {
        let Ok(a) = serde_json::from_value::<ArcFit>(arcs[key].clone()) else {
            continue;
        };
        if a.accepted && curves {
            let (c, range) = (a.coefficients.unwrap(), a.x_range.unwrap());
            for xx in range[0].ceil() as usize..range[1].floor() as usize {
                let supported = a
                    .points
                    .iter()
                    .zip(&a.inliers)
                    .any(|(p, k)| *k && (p[0] - xx as f64).abs() <= 4.);
                if supported || xx % 12 < 6 {
                    canvas.line(
                        [x + xx as f64 + 0.5, yy + y(c, xx as f64) + 0.5],
                        [x + xx as f64 + 1.5, yy + y(c, xx as f64 + 1.) + 0.5],
                        2.,
                        color,
                    );
                }
            }
        }
        for (p, &retained) in a.points.iter().zip(&a.inliers).filter(|_| points) {
            canvas.dot(
                x + p[0] + 0.5,
                yy + p[1] + 0.5,
                if retained && a.accepted { 1.9 } else { 1.1 },
                if retained && a.accepted { color } else { MUTED },
                true,
            );
        }
    }
}

pub(super) fn status(arcs: &Value) -> String {
    ["upper", "lower"]
        .iter()
        .map(|key| {
            let a = &arcs[key];
            let label = if *key == "upper" { "U" } else { "L" };
            if a["accepted"] == true {
                let count = a["inliers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|v| **v == true)
                    .count();
                format!(
                    "{label} {count}/{} {:.1}px",
                    a["points"].as_array().unwrap().len(),
                    a["rmse_px"].as_f64().unwrap()
                )
            } else {
                format!("{label}: no supported fit")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn self_check() -> Result<Value> {
    let u = |x: f64| 70. + 0.0013 * (x - 210.).powi(2) + 0.12 * (x - 210.);
    let l = |x: f64| 210. - 0.0010 * (x - 210.).powi(2) + 0.12 * (x - 210.);
    let mut mask = vec![0; W * H];
    for yy in 0..H {
        for x in 0..W {
            let (xx, y) = (x as f64, yy as f64);
            if y >= u(xx) && y <= l(xx) && (xx - 210.).hypot(y - 145.) > 90. {
                mask[yy * W + x] = 255;
            }
        }
    }
    let result = analyze(&mask);
    let mut errors = vec![];
    for (key, truth) in [
        ("upper", u as fn(f64) -> f64),
        ("lower", l as fn(f64) -> f64),
    ] {
        let a: ArcFit = serde_json::from_value(result[key].clone())?;
        if !a.accepted {
            return Err(format!("synthetic {key} arc not recovered: {}", a.reason).into());
        }
        let c = a.coefficients.unwrap();
        let error = [70., 180., 210., 300., 350.]
            .iter()
            .map(|&x| (y(c, x) - truth(x)).abs())
            .fold(0., f64::max);
        if error > 2.5 {
            return Err("synthetic iris-gap curve recovery exceeds 2.5px".into());
        }
        errors.push(error);
    }
    let empty = analyze(&vec![0; W * H]);
    if empty["upper"]["accepted"] == true || empty["lower"]["accepted"] == true {
        return Err("empty-mask fit must abstain".into());
    }
    Ok(
        json!({"known_two_quadratics_with_circular_iris_gap":true,"max_errors_upper_lower_px":errors,"predeclared_limit_px":2.5,"empty_mask_abstained":true,"role":"checks geometry implementation only; not real lid accuracy"}),
    )
}

fn focus_sheet(out: &Path, records: &[Value], filename: &str) -> Result<()> {
    // Fixed prompt index for this explanatory layout, not per-frame prompt
    // selection. The six-prompt comparison is retained alongside it.
    let mut c = Canvas::new(4 * 444 + 20, records.len() * 335 + 150)?;
    c.clear();
    c.text(
        18.,
        33.,
        27.,
        WHITE,
        "From sclera to eyelid arcs | prompt: white of the eye",
    );
    c.text(18., 64., 17., MUTED, "Amber upper / cyan lower. Gray points rejected. Dashed curves interpolate gaps; fit residual is not anatomical error.");
    for (i, label) in [
        "Original / matched blur",
        "SAM sclera mask (cyan tint)",
        "Start / end boundary samples",
        "Fitted arcs only",
    ]
    .iter()
    .enumerate()
    {
        c.text(18. + i as f64 * 444., 103., 21., WHITE, label);
    }
    for (i, row) in records.iter().enumerate() {
        let p = &row["prompts"][1];
        if p["prompt"] != "white of the eye" {
            return Err("focused sheet prompt changed".into());
        }
        let rgb = fs::read(out.join(row["rgb"].as_str().ok_or("rgb path")?))?;
        let mask = fs::read(out.join(p["mask"].as_str().ok_or("mask path")?))?;
        if data::digest(&rgb) != row["sam_input_rgb_sha256"]
            || data::digest(&mask) != p["mask_sha256"]
        {
            return Err("focused sheet input changed".into());
        }
        let base = super::bgra(&rgb);
        let mut tinted = base.clone();
        for (j, &v) in mask.iter().enumerate() {
            if v >= 128 {
                for channel in 0..3 {
                    tinted[j * 4 + channel] = (0.72 * base[j * 4 + channel] as f64
                        + 0.28 * CYAN[2 - channel] * 255.)
                        .round() as u8;
                }
            }
        }
        let y = 145. + i as f64 * 335.;
        c.text(
            18.,
            y - 12.,
            16.,
            WHITE,
            &format!(
                "{} | S{} eye{} seq{}",
                row["sample_id"].as_str().unwrap(),
                n(&row["source"]),
                n(&row["eye"]),
                n(&row["sequence"])
            ),
        );
        for col in 0..4 {
            let x = 18. + col as f64 * 444.;
            c.image(
                if col == 1 { &tinted } else { &base },
                W,
                H,
                x,
                y,
                W as f64,
                H as f64,
            );
            if col >= 2 {
                c.clipped(x, y, W as f64, H as f64, |c| {
                    draw_layers(c, x, y, &p["arcs"], col == 2, col == 3)
                });
            }
        }
        c.text(
            462.,
            y + H as f64 + 24.,
            17.,
            MUTED,
            &format!(
                "SAM score {:.3}; {} mask pixels",
                p["score"].as_f64().unwrap(),
                n(&p["area_pixels"])
            ),
        );
        c.text(1350., y + H as f64 + 24., 16., MUTED, &status(&p["arcs"]));
    }
    c.png(&out.join(filename))
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 3 {
        return Err("sclera-arcs COMPLETED_SCLERA_RUN NEW_OUT".into());
    }
    if args[1] == "--self-check" {
        let check = self_check()?;
        let out = data::output(&args[2])?;
        data::write(out.join("check.json"), &check)?;
        println!("{}", serde_json::to_string_pretty(&check)?);
        return Ok(());
    }
    let source = Path::new(&args[1]);
    let summary: Value = serde_json::from_slice(&fs::read(source.join("summary.json"))?)?;
    if summary["complete"] != true || summary["diagnostic"] != "sclera" {
        return Err("completed sclera prompt run required".into());
    }
    let check = self_check()?;
    let out = data::output(&args[2])?;
    fs::create_dir(out.join("masks"))?;
    let mut records = rows(&source.join("frames.jsonl"))?;
    let mut writer = BufWriter::new(fs::File::create(out.join("frames.jsonl"))?);
    for row in &mut records {
        row["diagnostic"] = json!("sclera-arcs");
        let rgb_path = row["rgb"].as_str().ok_or("RGB path")?;
        let rgb = fs::read(source.join(rgb_path))?;
        if data::digest(&rgb) != row["sam_input_rgb_sha256"] {
            return Err("SAM input RGB hash mismatch".into());
        }
        fs::write(out.join(rgb_path), rgb)?;
        for p in row["prompts"].as_array_mut().ok_or("prompts")? {
            let path = p["mask"].as_str().ok_or("mask path")?;
            let mask = fs::read(source.join(path))?;
            if mask.len() != W * H || data::digest(&mask) != p["mask_sha256"] {
                return Err("SAM mask changed".into());
            }
            fs::write(out.join(path), &mask)?;
            p["arcs"] = analyze(&mask);
        }
        serde_json::to_writer(&mut writer, row)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    super::render_all(
        &out,
        &records,
        &summary["selection"],
        "sclera boundary arc fits",
    )?;
    let display = if summary["selection"]["mode"] == "matched" {
        (0..6)
            .flat_map(|i| [records[6 + i].clone(), records[i].clone()])
            .collect::<Vec<_>>()
    } else {
        records.clone()
    };
    focus_sheet(&out, &display, "sclera-to-arcs.png")?;
    for (i, chunk) in display.chunks(6).enumerate() {
        focus_sheet(&out, chunk, &format!("sclera-to-arcs-page-{}.png", i + 1))?;
    }
    let results: Vec<_> = (0..6).map(|k| {
        let p: Vec<_> = records.iter().map(|r| &r["prompts"][k]).collect();
        json!({"prompt":p[0]["prompt"],"upper_fits":p.iter().filter(|p| p["arcs"]["upper"]["accepted"] == true).count(),"lower_fits":p.iter().filter(|p| p["arcs"]["lower"]["accepted"] == true).count(),"both_fits":p.iter().filter(|p| p["arcs"]["upper"]["accepted"] == true && p["arcs"]["lower"]["accepted"] == true).count(),"denominator":p.len()})
    }).collect();
    data::write(
        out.join("summary.json"),
        &json!({"schema":"buttercup-sclera-boundary-arcs-v1","complete":true,"source_run":fs::canonicalize(source)?,"source_summary_sha256":sam_export::hash(&source.join("summary.json"))?,"source_frames_sha256":sam_export::hash(&source.join("frames.jsonl"))?,"executable_sha256":sam_export::hash(Path::new("/proc/self/exe"))?,"synthetic_check":check,"results":results,"anatomical_accuracy":null,"model":"y=H/2+(H/2)*(c0+c1*t+c2*t*t), t=(x-W/2)/(W/2)","policy":{"mask_threshold":0.5,"remove_components_below_px":32,"column_stride":3,"minimum_sclera_pixels_per_column":4,"minimum_points":18,"minimum_span_px":70,"residual_band_px":BAND,"minimum_inlier_fraction":0.5,"minimum_x_bins_of_6":4,"ransac_trials":384,"curvature_prior":"upper c2 >= -0.005, lower c2 <= 0.005 for approximately upright images","extrapolation":"none beyond inlier span; gaps inside span dashed"},"source_role":"SAM mask boundary proposals, not measured RAW edges or reviewed anatomy","no_iris_ellipse_used":true}),
    )?;
    fs::write(out.join("README.md"), "# Sclera boundary arc experiment\n\nSame frozen six original/blurred pairs and six literal sclera prompts. Gray is the original highest-score SAM mask boundary. Every third image column supplies the first and last threshold crossings after deleting connected dust smaller than 32 pixels. Crop edges and columns with fewer than four foreground pixels are excluded.\n\nAmber dots/curves are upper-envelope points and a robust quadratic fit; cyan is lower. Gray dots were rejected or could not support a fit. These points come from SAM, not independent observed anatomical labels. Upper/lower bend directions are a declared prior for these approximately upright ROIs. It helps distinguish lid-facing edges from iris-facing edges, but cannot certify the mask. No iris ellipse, gaze or anatomical oracle is used.\n\nThe shared native lid least-squares primitive fits normalized quadratics inside deterministic RANSAC. A fit needs at least 18 inliers, half the sampled points, 70 pixels of span and four of six horizontal bins. Inlier residual tolerance is 3.5 pixels. Crossing curves are rejected. Counts and residuals measure agreement with the predicted mask, not anatomical accuracy or calibrated confidence.\n\nSolid curve segments have nearby retained boundary points. Dashed segments interpolate unsupported gaps (often the iris). Nothing is drawn outside the retained horizontal span. One-sided sclera can therefore support only a partial curve. Missing/incorrect masks remain failures; the highest-score query is retained without selecting a more attractive alternative.\n\nThe manifest includes a known synthetic pair of quadratics with a circular iris gap, an empty-mask abstention check, all real point arrays/inlier flags/coefficients, hashes and policy constants. The synthetic check validates fitting arithmetic, not real-world anatomy. No training or live pipeline behavior changes.\n\nRun: buttercup_calibration_sign sclera-arcs COMPLETED_SCLERA_RUN NEW_OUT\n")?;
    eprintln!("SCLERA ARCS DONE {}", out.display());
    Ok(())
}
