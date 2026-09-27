//! Diagnostic corneal-reflection / neighboring-sclera illumination consistency.
//! Native same-CFA-phase RAW codes; no fitted exposure correction, sign training
//! or anatomically calibrated radiometry. Both poses use identical observations.
use super::{archive, Result};
#[path = "../../bootstrapability.rs"]
#[allow(dead_code)]
mod boot;
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
#[path = "../../raw_preview.rs"]
#[allow(dead_code)]
mod preview;
use buttercup_eye_tracking::{
    focus_region::*, geometry::Ellipse, raw10, recorded_bundle::BundleSource,
};
use canvas::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

#[derive(Clone)]
struct Pixel {
    xy: [f64; 2],
    value: f64,
    weight: f64,
}
struct Input {
    w: usize,
    h: usize,
    origin: [u32; 2],
    raw: Vec<u16>,
    sclera: Vec<u8>,
    mw: usize,
    mh: usize,
}
fn unit(v: V3) -> V3 {
    scale(v, 1. / norm(v).max(1e-12))
}
fn camera_ray(p: [f64; 2]) -> V3 {
    unit([(p[0] - 4000.) / 4000., (p[1] - 3000.) / 4000., -1.])
}
fn quantile(values: &[f64], q: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q) as usize]
}
fn shape(f: &Value) -> Option<Ellipse> {
    let e = &f["ellipse"];
    Some(Ellipse {
        center: (e[0].as_f64()?, e[1].as_f64()?),
        major_radius: e[2].as_f64()?,
        minor_radius: e[3].as_f64()?,
        angle: e[4].as_f64()?,
    })
}
fn rho(e: Ellipse, p: [f64; 2]) -> f64 {
    let (s, c) = e.angle.sin_cos();
    let dx = p[0] - e.center.0;
    let dy = p[1] - e.center.1;
    ((dx * c + dy * s) / e.major_radius).hypot((-dx * s + dy * c) / e.minor_radius)
}
fn surface(center: V3, r: f64, p: [f64; 2]) -> Option<(V3, V3)> {
    let q = camera_ray(p);
    let b = dot(q, center);
    let d = b * b - dot(center, center) + r * r;
    if d <= 0. || b <= 0. {
        return None;
    }
    let t = b - d.sqrt();
    if t <= 0. {
        return None;
    }
    let hit = scale(q, t);
    Some((hit, unit(sub(hit, center))))
}
fn reflected_direction(hit: V3, n: V3) -> V3 {
    let v = unit(scale(hit, -1.));
    unit(sub(scale(n, 2. * dot(n, v)), v))
}
fn globe_center(p: GazeRay, r: f64) -> V3 {
    sub(p.origin_iris_radii, scale(p.direction, (r * r - 1.).sqrt()))
}
fn points(input: &Input, e: Ellipse, phase: usize, threshold: u8) -> (Vec<Pixel>, Vec<Pixel>) {
    let mut iris = vec![];
    let mut sclera = vec![];
    let (px, py) = if phase == 0 { (2, 0) } else { (0, 2) };
    let (s, c) = e.angle.sin_cos();
    let hx = (e.major_radius.powi(2) * c * c + e.minor_radius.powi(2) * s * s).sqrt();
    let hy = (e.major_radius.powi(2) * s * s + e.minor_radius.powi(2) * c * c).sqrt();
    for y in 0..input.h {
        if (y + input.origin[1] as usize) % 4 != py {
            continue;
        }
        for x in 0..input.w {
            if (x + input.origin[0] as usize) % 4 != px {
                continue;
            }
            let p = [x as f64, y as f64];
            let r = rho(e, p);
            let v = input.raw[y * input.w + x] as f64;
            let value = Pixel {
                xy: [p[0] + input.origin[0] as f64, p[1] + input.origin[1] as f64],
                value: v,
                weight: 1.,
            };
            if r < 0.87 {
                iris.push(value.clone());
            }
            let mx = ((x as f64 + 0.5) * input.mw as f64 / input.w as f64).floor() as usize;
            let my = ((y as f64 + 0.5) * input.mh as f64 / input.h as f64).floor() as usize;
            // Same learned mask, shared 2D lateral region and observed samples
            // for A/B; never select different bright pixels per sphere.
            if r > 1.10
                && r < 1.9
                && (p[0] - e.center.0).abs() > 0.65 * hx
                && (p[1] - e.center.1).abs() < 0.75 * hy
                && input.sclera[my.min(input.mh - 1) * input.mw + mx.min(input.mw - 1)] >= threshold
                && v > 4.
                && v < 1015.
            {
                sclera.push(value);
            }
        }
    }
    if iris.len() < 16 {
        return (sclera, vec![]);
    }
    let values = iris.iter().map(|p| p.value).collect::<Vec<_>>();
    let med = quantile(&values, 0.5);
    let hi = quantile(&values, 0.995);
    let cutoff = med + 0.65 * (hi - med);
    if hi - med < 50. {
        return (sclera, vec![]);
    }
    let possible = iris
        .iter()
        .filter(|p| p.value > cutoff)
        .map(|p| ((p.xy[0] as i32, p.xy[1] as i32), p))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut glints = vec![];
    for (&key, _) in &possible {
        if !seen.insert(key) {
            continue;
        }
        let mut queue = vec![key];
        let mut i = 0;
        while i < queue.len() {
            let (x, y) = queue[i];
            i += 1;
            for (dx, dy) in [(4, 0), (-4, 0), (0, 4), (0, -4)] {
                let n = (x + dx, y + dy);
                if possible.contains_key(&n) && seen.insert(n) {
                    queue.push(n);
                }
            }
        }
        if queue.len() >= 3 {
            for k in queue {
                let mut p = possible[&k].clone();
                p.weight = p.value - med;
                glints.push(p);
            }
        }
    }
    (sclera, glints)
}
#[derive(Clone)]
struct Fit {
    coef: Vec<f64>,
    mse: f64,
    pred: Vec<f64>,
}
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let p = (i..n).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs()))?;
        a.swap(i, p);
        b.swap(i, p);
        let d = a[i][i];
        if d.abs() < 1e-9 {
            return None;
        }
        for j in i..n {
            a[i][j] /= d;
        }
        b[i] /= d;
        for k in 0..n {
            if k == i {
                continue;
            }
            let t = a[k][i];
            for j in i..n {
                a[k][j] -= t * a[i][j];
            }
            b[k] -= t * b[i];
        }
    }
    Some(b)
}
fn regression(features: &[Vec<f64>], points: &[Pixel], positive_gain: bool) -> Option<Fit> {
    let p = features[0].len();
    let mut predicted = vec![0.; points.len()];
    let mut last = vec![];
    // Hold out alternating 24x24 sensor-coordinate blocks, not neighboring
    // individual pixels. Every score uses the same held-out pixels for A/B.
    for fold in 0..2 {
        let mut weights = vec![1.; points.len()];
        let train = (0..points.len())
            .filter(|&i| {
                ((points[i].xy[0] as usize / 24 + points[i].xy[1] as usize / 24) % 2) != fold
            })
            .collect::<Vec<_>>();
        if train.len() < 8 || points.len() - train.len() < 8 {
            return None;
        }
        let mut coef = vec![0.; p];
        for _ in 0..4 {
            let mut a = vec![vec![0.; p]; p];
            let mut b = vec![0.; p];
            for &i in &train {
                for j in 0..p {
                    b[j] += weights[i] * features[i][j] * points[i].value;
                    for k in 0..p {
                        a[j][k] += weights[i] * features[i][j] * features[i][k];
                    }
                }
            }
            coef = solve(a, b)?;
            if positive_gain && coef[1] < 0. {
                coef[1] = 0.;
                coef[0] = train.iter().map(|&i| points[i].value).sum::<f64>() / train.len() as f64;
            }
            let residual = train
                .iter()
                .map(|&i| {
                    (points[i].value
                        - features[i]
                            .iter()
                            .zip(&coef)
                            .map(|(x, c)| x * c)
                            .sum::<f64>())
                    .abs()
                })
                .collect::<Vec<_>>();
            let scale = quantile(&residual, 0.5).max(3.) * 1.5;
            for (&i, r) in train.iter().zip(residual) {
                weights[i] = (scale / r.max(1e-12)).min(1.);
            }
        }
        for i in 0..points.len() {
            if (points[i].xy[0] as usize / 24 + points[i].xy[1] as usize / 24) % 2 == fold {
                predicted[i] = features[i].iter().zip(&coef).map(|(x, c)| x * c).sum();
            }
        }
        last.push(coef);
    }
    let mse = points
        .iter()
        .zip(&predicted)
        .map(|(p, v)| (p.value - v).powi(2))
        .sum::<f64>()
        / points.len() as f64;
    Some(Fit {
        coef: (0..p).map(|i| (last[0][i] + last[1][i]) * 0.5).collect(),
        mse,
        pred: predicted,
    })
}

struct Review {
    points: Vec<Pixel>,
    glints: Vec<Pixel>,
    predictions: [Vec<f64>; 2],
    lights: [V3; 2],
    rays: TheoreticalEllipseExplanations,
    e: Ellipse,
    metrics: Value,
}
fn variant(
    input: &Input,
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
    phase: usize,
    threshold: u8,
    rg: f64,
    rc: f64,
) -> (Value, Option<Review>) {
    let (observed, glints) = points(input, e, phase, threshold);
    let base = json!({"phase":phase,"sclera_threshold":threshold as f64/255.,"globe_radius_per_iris":rg,"cornea_radius_per_iris":rc,"sclera_samples":observed.len(),"glint_samples":glints.len()});
    let unavailable = |why: &str| {
        let mut v = base.clone();
        v["available"] = json!(false);
        v["reason"] = json!(why);
        (v, None)
    };
    if observed.len() < 32 {
        return unavailable("insufficient-sclera-support");
    }
    if glints.len() < 3 {
        return unavailable("no-resolved-corneal-highlight");
    }
    let centers = rays.rays.map(|p| globe_center(p, rg));
    let corneas = rays.rays.map(|p| globe_center(p, rc));
    let points = observed
        .iter()
        .filter(|p| centers.iter().all(|c| surface(*c, rg, p.xy).is_some()))
        .cloned()
        .collect::<Vec<_>>();
    // Even when a sphere fails the containment gate, show the lighting on
    // their common visible support. Keep the original 90% admission rule;
    // omitted observed pixels never silently become evidence for a sign.
    if points.len() < 32 {
        return unavailable("insufficient-common-globe-support");
    }
    let coverage_ok = points.len() * 10 >= observed.len() * 9;
    let xs = points.iter().map(|p| p.xy[0]).collect::<Vec<_>>();
    let ys = points.iter().map(|p| p.xy[1]).collect::<Vec<_>>();
    if quantile(&ys, 0.95) - quantile(&ys, 0.05) < e.major_radius * 0.25 {
        return unavailable("insufficient-vertical-sclera-extent");
    }
    let mut lights = [[0.; 3]; 2];
    for k in 0..2 {
        let mut sum = [0.; 3];
        let mut weight = 0.;
        let mut hits = 0;
        for p in &glints {
            if let Some((h, n)) = surface(corneas[k], rc, p.xy) {
                if dot(
                    sub(h, rays.rays[k].origin_iris_radii),
                    rays.rays[k].direction,
                ) < -0.05
                {
                    continue;
                }
                sum = add(sum, scale(reflected_direction(h, n), p.weight));
                weight += p.weight;
                hits += 1;
            }
        }
        if hits * 10 < glints.len() * 9 || weight <= 0. || norm(sum) / weight < 0.8 {
            return unavailable("inconsistent-corneal-reflection-support");
        }
        lights[k] = unit(sum);
    }
    let values = points.iter().map(|p| p.value).collect::<Vec<_>>();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    if variance < 25. {
        return unavailable("insufficient-observed-shading-variation");
    }
    let x0 = xs.iter().sum::<f64>() / xs.len() as f64;
    let y0 = ys.iter().sum::<f64>() / ys.len() as f64;
    let plane_features = points
        .iter()
        .map(|p| {
            vec![
                1.,
                (p.xy[0] - x0) / e.major_radius,
                (p.xy[1] - y0) / e.major_radius,
            ]
        })
        .collect::<Vec<_>>();
    let Some(plane) = regression(&plane_features, &points, false) else {
        return unavailable("spatial-holdout-degenerate");
    };
    let Some(constant) = regression(&vec![vec![1.]; points.len()], &points, false) else {
        return unavailable("spatial-holdout-degenerate");
    };
    let mut fitted = vec![];
    for k in 0..2 {
        let features = points
            .iter()
            .map(|p| {
                let (_, n) = surface(centers[k], rg, p.xy).unwrap();
                vec![1., dot(n, lights[k]).max(0.)]
            })
            .collect::<Vec<_>>();
        let Some(f) = regression(&features, &points, true) else {
            return unavailable("reflection-predicts-constant-sclera-illumination");
        };
        fitted.push(f);
    }
    let a = &fitted[0];
    let b = &fitted[1];
    let winner = usize::from(b.mse < a.mse);
    let best = &fitted[winner];
    let worst = &fitted[1 - winner];
    let gap = (worst.mse - best.mse) / constant.mse.max(1.);
    let improvement = 1. - best.mse / constant.mse.max(1.);
    let selected = (coverage_ok
        && gap >= 0.10
        && improvement >= 0.20
        && best.coef[0] >= 0.
        && best.coef[1] > 0.)
        .then_some(winner);
    let mut v = base;
    v["available"] = json!(true);
    v["common_sclera_samples"] = json!(points.len());
    v["common_sclera_fraction"] = json!(points.len() as f64 / observed.len() as f64);
    v["containment_gate_passed"] = json!(coverage_ok);
    v["candidate_sclera_containment"] = json!(centers.map(|center| observed
        .iter()
        .filter(|p| surface(center, rg, p.xy).is_some())
        .count() as f64
        / observed.len() as f64));
    v["inferred_light_directions"] = json!(lights);
    v["iris_normals"] = json!(rays.rays.map(|p| p.direction));
    v["constant_cv_mse"] = json!(constant.mse);
    v["image_plane_cv_mse"] = json!(plane.mse);
    v["image_brightness_gradient_codes_per_iris_radius"] = json!([plane.coef[1], plane.coef[2]]);
    v["candidate_cv_mse"] = json!([a.mse, b.mse]);
    v["ambient_directional_coefficients"] = json!([a.coef, b.coef]);
    v["relative_error_gap"] = json!(gap);
    v["improvement_over_constant"] = json!(improvement);
    v["lower_error_candidate"] = json!(winner);
    v["conditional_choice"] = json!(selected);
    v["beats_image_plane"] = json!(best.mse < plane.mse * 0.95);
    let review = Review {
        points,
        glints,
        predictions: [a.pred.clone(), b.pred.clone()],
        lights,
        rays,
        e,
        metrics: v.clone(),
    };
    (v, Some(review))
}
fn analyze(input: &Input, fit: &Value) -> (Value, Option<Review>) {
    let Some(e) = shape(fit) else {
        return (json!({"reason":"no-fresh-fit","stable_choice":null}), None);
    };
    let mut sensor = e;
    sensor.center.0 += input.origin[0] as f64;
    sensor.center.1 += input.origin[1] as f64;
    let Some(rays) =
        TheoreticalEllipseExplanations::from_ellipse(sensor, [4000.; 2], [4000., 3000.])
    else {
        return (
            json!({"reason":"invalid-unprojection","stable_choice":null}),
            None,
        );
    };
    let mut variants = vec![];
    let mut nominal = None;
    for phase in 0..2 {
        for threshold in [179, 230] {
            for rg in [1.8, 2., 2.2] {
                for rc in [1.2, 1.3, 1.4] {
                    let (v, review) = variant(input, e, rays, phase, threshold, rg, rc);
                    if phase == 0 && threshold == 179 && rg == 2. && rc == 1.3 {
                        nominal = review;
                    }
                    variants.push(v);
                }
            }
        }
    }
    let available = variants.iter().filter(|v| v["available"] == true).count();
    let votes = std::array::from_fn::<_, 2, _>(|k| {
        variants
            .iter()
            .filter(|v| v["conditional_choice"].as_u64() == Some(k as u64))
            .count()
    });
    let winner = usize::from(votes[1] > votes[0]);
    let stable = (available >= 24
        && votes[winner] >= 24
        && votes[1 - winner] == 0
        && nominal
            .as_ref()
            .is_some_and(|r| r.metrics["conditional_choice"].as_u64() == Some(winner as u64)))
    .then_some(winner);
    let native_surface_supported = stable.filter(|_| {
        nominal
            .as_ref()
            .is_some_and(|r| r.metrics["beats_image_plane"] == true)
    });
    (
        json!({"geometry":rays,"variants":variants,"available_variants":available,"conditional_votes":votes,"stable_choice":stable,"surface_specific_choice":native_surface_supported,"nominal":nominal.as_ref().map(|r|&r.metrics),"choice_basis":"Uncalibrated photometric compatibility; no physical sign truth"}),
        nominal,
    )
}

fn render(
    out: &Path,
    row: &Value,
    provider: &str,
    input: &Input,
    r: &Review,
    analysis: &Value,
) -> Result<()> {
    let mut c = Canvas::new(1800, 1020)?;
    c.clear();
    c.text(
        20.,
        34.,
        24.,
        WHITE,
        &format!(
            "Reflected light and sclera shading | {provider} | source {} eye {} seq {}",
            row["source"], row["eye"], row["sequence"]
        ),
    );
    c.text(20.,66.,17.,MUTED,"Both hypotheses use the same observed pixels. Light direction comes from the corneal highlight; brightness is checked on held-out patches.");
    let color = preview::color_preview(
        &input.raw,
        input.w,
        input.h,
        input.origin[0],
        input.origin[1],
        100,
        None,
    );
    let color = color
        .iter()
        .flat_map(|p| {
            [
                (*p & 255) as u8,
                ((*p >> 8) & 255) as u8,
                ((*p >> 16) & 255) as u8,
                255,
            ]
        })
        .collect::<Vec<_>>();
    let observed = r.points.iter().map(|p| p.value).collect::<Vec<_>>();
    let lo = quantile(&observed, 0.05);
    let hi = quantile(&observed, 0.95);
    for col in 0..3 {
        let x = 20. + col as f64 * 594.;
        let y = 114.;
        let s = 560. / input.w as f64;
        let hh = input.h as f64 * s;
        c.text(
            x,
            100.,
            21.,
            WHITE,
            [
                "Observed RAW and sampled sclera",
                "Hypothesis A: predicted shading",
                "Hypothesis B: predicted shading",
            ][col],
        );
        c.image(&color, input.w, input.h, x, y, 560., hh);
        let pp = |p: [f64; 2]| {
            [
                x + (p[0] - input.origin[0] as f64) * s,
                y + (p[1] - input.origin[1] as f64) * s,
            ]
        };
        c.clipped(x, y, 560., hh, |c| {
            c.path(
                &r.e.dense_points(200)
                    .iter()
                    .map(|&(u, v)| [x + u * s, y + v * s])
                    .collect::<Vec<_>>(),
                1.5,
                WHITE,
            );
            for (i, p) in r.points.iter().enumerate() {
                let v = if col == 0 {
                    p.value
                } else {
                    r.predictions[col - 1][i]
                };
                let q = ((v - lo) / (hi - lo).max(1.)).clamp(0., 1.);
                let p = pp(p.xy);
                c.rect(p[0] - 1.3 * s, p[1] - 1.3 * s, 2.6 * s, 2.6 * s, [q, q, q]);
            }
            for p in &r.glints {
                let p = pp(p.xy);
                c.dot(p[0], p[1], 2., ORANGE, false);
            }
            if col > 0 {
                let ray = r.rays.rays[col - 1];
                let proj = |p: V3| pp([4000. + 4000. * p[0] / -p[2], 3000. + 4000. * p[1] / -p[2]]);
                c.arrow(
                    proj(ray.origin_iris_radii),
                    proj(add(ray.origin_iris_radii, scale(ray.direction, 0.7))),
                    [CYAN, PINK][col - 1],
                );
            }
        });
        if col == 0 {
            c.text(
                x,
                525.,
                18.,
                WHITE,
                &format!(
                    "{} sclera samples; {} highlight samples",
                    r.points.len(),
                    r.glints.len()
                ),
            );
            let grad = &r.metrics["image_brightness_gradient_codes_per_iris_radius"];
            c.text(
                x,
                553.,
                17.,
                MUTED,
                &format!(
                    "Image gradient: right {:+.1}, down {:+.1}",
                    grad[0].as_f64().unwrap(),
                    grad[1].as_f64().unwrap()
                ),
            );
            c.text(
                x,
                580.,
                17.,
                MUTED,
                "Brightness patches use one common RAW-code scale.",
            );
            c.text(
                x,
                604.,
                15.,
                MUTED,
                &format!(
                    "Common globe coverage: {:.1}% of proposed sclera",
                    100. * r.metrics["common_sclera_fraction"].as_f64().unwrap()
                ),
            );
        } else {
            let k = col - 1;
            let light = r.lights[k];
            let n = r.rays.rays[k].direction;
            c.text(
                x,
                525.,
                19.,
                [CYAN, PINK][k],
                &format!(
                    "Iris normal y {:+.3} ({})",
                    n[1],
                    if n[1] < 0. {
                        "camera-up"
                    } else {
                        "camera-down"
                    }
                ),
            );
            c.text(
                x,
                553.,
                17.,
                WHITE,
                &format!(
                    "Light direction [{:+.2}, {:+.2}, {:+.2}]",
                    light[0], light[1], light[2]
                ),
            );
            c.text(
                x,
                580.,
                17.,
                MUTED,
                &format!(
                    "Held-out shading error {:.1}",
                    r.metrics["candidate_cv_mse"][k].as_f64().unwrap()
                ),
            );
        }
    }
    let xx = 110.;
    let yy = 870.;
    let ww = 1510.;
    let hh = 210.;
    c.text(20.,622.,18.,WHITE,"Brightness versus image height: observed white, hypothesis A cyan, hypothesis B pink (same spatially held-out sclera samples)");
    c.line([xx, yy], [xx + ww, yy], 1., MUTED);
    c.line([xx, yy], [xx, yy - hh], 1., MUTED);
    let ylo = r
        .points
        .iter()
        .map(|p| p.xy[1])
        .fold(f64::INFINITY, f64::min);
    let yhi = r
        .points
        .iter()
        .map(|p| p.xy[1])
        .fold(f64::NEG_INFINITY, f64::max);
    for series in 0..3 {
        let mut curve = vec![];
        for bin in 0..16 {
            let ids = (0..r.points.len())
                .filter(|&i| {
                    (((r.points[i].xy[1] - ylo) / (yhi - ylo + 1.) * 16.) as usize).min(15) == bin
                })
                .collect::<Vec<_>>();
            if ids.len() < 3 {
                continue;
            }
            let v = ids
                .iter()
                .map(|&i| {
                    if series == 0 {
                        r.points[i].value
                    } else {
                        r.predictions[series - 1][i]
                    }
                })
                .sum::<f64>()
                / ids.len() as f64;
            curve.push([
                xx + (bin as f64 + 0.5) * ww / 16.,
                yy - ((v - lo) / (hi - lo).max(1.)).clamp(0., 1.) * hh,
            ]);
        }
        c.path(&curve, 3., [WHITE, CYAN, PINK][series]);
    }
    c.text(20.,920.,19.,WHITE,&format!("Across 36 geometry / mask / green-phase settings: votes {} | stable choice {} | beats image-plane control {}",analysis["conditional_votes"],analysis["stable_choice"],analysis["surface_specific_choice"]));
    c.text(20.,953.,17.,MUTED,"Nominal sphere radii: globe 2.0, cornea 1.3 iris radii. Unknown shadows, glasses, source extent and surface reflectance can invalidate this model.");
    c.text(20.,985.,17.,MUTED,"Camera-relative up/down, not world vertical. Masks and ellipses are predictions; physical sign truth and independent scale are unavailable.");
    c.png(&out.join(format!("lighting-{}-{provider}.png", row["record"])))
}
pub fn run(fresh: &str, output: &str) -> Result<()> {
    let start = Instant::now();
    let fresh = Path::new(fresh);
    let out = Path::new(output);
    if !fs::canonicalize(out.parent().ok_or("output parent")?)?
        .starts_with(fs::canonicalize("data")?)
    {
        return Err("output must use bulk-data link".into());
    }
    fs::create_dir(out)?;
    let source = boot::current_source(Path::new("."))?;
    let rows = BufReader::new(fs::File::open(fresh.join("frames.jsonl"))?)
        .lines()
        .map(|r| Ok(serde_json::from_str::<Value>(&r?)?))
        .collect::<Result<Vec<_>>>()?;
    let completed: Value = serde_json::from_slice(&fs::read(fresh.join("summary.json"))?)?;
    if completed["complete"] != true
        || completed["counts_total_sam_fit_admitted_obelisk_fit_admitted"][0].as_u64()
            != Some(rows.len() as u64)
    {
        return Err("fresh segmentation is incomplete or its coverage changed".into());
    }
    let provenance: Value = serde_json::from_slice(&fs::read(fresh.join("provenance.json"))?)?;
    let mut bundles = BTreeMap::new();
    let mut counts = BTreeMap::<String, usize>::new();
    for provider in ["sam", "obelisk"] {
        for key in [
            "total",
            "fresh_admitted",
            "nominal_available",
            "nominal_choice",
            "nominal_containment_pass",
            "nominal_beats_plane",
            "brightness_increases_down",
            "stable_choice",
            "surface_specific_choice",
        ] {
            counts.insert(format!("{provider}/{key}"), 0);
        }
    }
    let mut examples = BTreeSet::new();
    let mut writer = BufWriter::new(fs::File::create(out.join("lighting.jsonl"))?);
    let mut index = vec![];
    let mut all = vec![];
    for row in rows {
        let path = row["raw_source"].as_str().ok_or("raw source")?;
        if !bundles.contains_key(path) {
            bundles.insert(path.to_owned(), BundleSource::open(Path::new(path))?);
        }
        let frame = &row["frame"];
        let n = |k: &str| frame[k].as_u64().unwrap();
        let raw = bundles[path].read_range(
            row["stream_entry"].as_str().unwrap(),
            n("offset"),
            n("length") as usize,
        )?;
        if archive::digest(&raw) != row["raw_sha256"] {
            return Err("RAW identity mismatch".into());
        }
        let masks = fs::read(fresh.join(row["obelisk"]["masks"].as_str().unwrap()))?;
        if archive::digest(&masks) != row["obelisk"]["masks_sha256"] || masks.len() != 6 * 384 * 256
        {
            return Err("Obelisk masks changed".into());
        }
        let input = Input {
            w: n("width") as usize,
            h: n("height") as usize,
            origin: [n("sensor_x") as u32, n("sensor_y") as u32],
            raw: raw10::try_unpack_raw10(
                &raw,
                n("width") as usize,
                n("height") as usize,
                n("stride") as usize,
            )?,
            sclera: masks[3 * 384 * 256..4 * 384 * 256].to_vec(),
            mw: 384,
            mh: 256,
        };
        let mut arms = vec![];
        for provider in ["sam", "obelisk"] {
            let (analysis, review) = if row[provider]["admissible"] == true {
                analyze(&input, &row[provider]["fit"])
            } else {
                (
                    json!({"reason":"fresh-conic-did-not-pass-RAW-gate","stable_choice":null}),
                    None,
                )
            };
            for (name, yes) in [
                ("total", true),
                ("fresh_admitted", row[provider]["admissible"] == true),
                ("nominal_available", review.is_some()),
                (
                    "nominal_containment_pass",
                    analysis["nominal"]["containment_gate_passed"] == true,
                ),
                (
                    "nominal_beats_plane",
                    analysis["nominal"]["beats_image_plane"] == true,
                ),
                (
                    "brightness_increases_down",
                    analysis["nominal"]["image_brightness_gradient_codes_per_iris_radius"][1]
                        .as_f64()
                        .is_some_and(|y| y > 0.),
                ),
                (
                    "nominal_choice",
                    analysis["nominal"]["conditional_choice"].is_number(),
                ),
                ("stable_choice", analysis["stable_choice"].is_number()),
                (
                    "surface_specific_choice",
                    analysis["surface_specific_choice"].is_number(),
                ),
            ] {
                if yes {
                    *counts.entry(format!("{provider}/{name}")).or_default() += 1;
                }
            }
            if let Some(ref review) = review {
                let tag = if analysis["surface_specific_choice"].is_number() {
                    "surface-specific"
                } else if analysis["stable_choice"].is_number() {
                    "stable"
                } else if analysis["nominal"]["conditional_choice"].is_number() {
                    "nominal-only"
                } else {
                    "unstable"
                };
                let key = format!("{provider}/source-{}/{tag}", row["source"]);
                let first = examples.insert(key);
                if first || (tag == "nominal-only" && index.len() < 80) {
                    render(out, &row, provider, &input, review, &analysis)?;
                    index.push(json!({"record":row["record"],"source":row["source"],"provider":provider,"reason":tag,"image":format!("lighting-{}-{provider}.png",row["record"])}));
                }
            }
            arms.push(json!({"provider":provider,"analysis":analysis}));
        }
        let result = json!({"record":row["record"],"source":row["source"],"epoch":row["epoch"],"eye":row["eye"],"sequence":row["sequence"],"source_ns":row["source_ns"],"raw_sha256":row["raw_sha256"],"arms":arms,"physical_sign_truth":null});
        serde_json::to_writer(&mut writer, &result)?;
        writeln!(writer)?;
        all.push(result);
        if all.len() % 100 == 0 {
            writer.flush()?;
            eprintln!(
                "LIGHTING {} frames {:.1}s",
                all.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    writer.flush()?;
    // Analytic reflection-law control: the recovered direction must give the
    // original surface normal as the light/view half-vector.
    let control_n = unit([0.1, -0.35, 0.9]);
    let hit = [0.2, -0.3, -20.];
    let light = reflected_direction(hit, control_n);
    let err = norm(sub(unit(add(light, unit(scale(hit, -1.)))), control_n));
    if err > 1e-10 {
        return Err("reflection-law control failed".into());
    }
    let paired = all
        .iter()
        .filter(|r| {
            r["arms"][0]["analysis"]["stable_choice"].is_number()
                && r["arms"][1]["analysis"]["stable_choice"].is_number()
        })
        .count();
    let summary = json!({"complete":true,"frames":all.len(),"source":source,"source_finish":boot::current_source(Path::new("."))?,"counts":counts,"both_providers_stable":paired,"seconds":start.elapsed().as_secs_f64(),"reflection_halfvector_error":err,"fresh_inference_provenance":provenance,"independent_sign_accuracy":null,"policy":{"pixels":"Only measured RAW10 green photosites at global CFA phases (2,0) and (0,2), independently. No RGB display values or 4x4 spatial averaging in photometry.","sclera":"Fixed Obelisk predicted sclera mask thresholds 0.7 and 0.9, shared ellipse lateral annulus 1.10..1.90, excluding clipped codes. Same observations for both signs; >=90% must lie on both candidate globes.","lighting":"Bright connected corneal-highlight samples reflected through each hypothetical corneal sphere give the candidate light direction. Independent ambient plus nonnegative directional gain fitted on alternating 24px source-coordinate blocks; error evaluated on the other blocks.","shape_sensitivity":"Globe radius 1.8/2.0/2.2 and cornea radius 1.2/1.3/1.4 iris radii. These engineering priors are not measured anatomy.","selection":"Nominal gap>=0.10 times constant-control MSE and >=20% improvement; stable requires >=24/36 supported preferences for the same sign and no opposite qualifying vote. Surface-specific additionally beats an image-plane gradient control by 5%.","limits":"Single-user historical diagnostic; no ground truth, measured lighting, calibrated reflectance, shadow/glasses/refraction model or independent scale. This does not change ellipses or SN-FEIDA and cannot certify a sign."}});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    fs::write(
        out.join("visual-review.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"frames":all.len(),"counts":counts,"seconds":start.elapsed().as_secs_f64()})
        )?
    );
    Ok(())
}
