//! Independent sphere-center shading profile; no sign labels or training.
//! First-order illumination is an explicit approximation, not measured light.
use super::*;
#[path = "paired.rs"]
mod paired;
pub(super) use paired::PairStream;
#[path = "temporal.rs"]
pub(crate) mod temporal;
#[path = "sclera_motion.rs"]
pub(crate) mod sclera_motion;

#[derive(Clone)]
struct Sample {
    xy: [f64; 2],
    q: V3,
    value: f64,
    fold: usize,
}
#[derive(Clone, Copy, Default)]
struct Stats {
    a: [[f64; 4]; 4],
    b: [f64; 4],
    yy: f64,
    count: usize,
}
impl Stats {
    fn add(&mut self, x: [f64; 4], y: f64) {
        self.count += 1;
        self.yy += y * y;
        for j in 0..4 {
            self.b[j] += x[j] * y;
            for k in 0..4 {
                self.a[j][k] += x[j] * x[k];
            }
        }
    }
    fn plus(self, other: Self) -> Self {
        let mut r = self;
        r.count += other.count;
        r.yy += other.yy;
        for j in 0..4 {
            r.b[j] += other.b[j];
            for k in 0..4 {
                r.a[j][k] += other.a[j][k];
            }
        }
        r
    }
    fn fit(self, dim: usize) -> Option<[f64; 4]> {
        if self.count < dim {
            return None;
        }
        let mut a = self.a;
        let mut b = self.b;
        for i in 0..dim {
            let p = (i..dim).max_by(|&j, &k| a[j][i].abs().total_cmp(&a[k][i].abs()))?;
            a.swap(i, p);
            b.swap(i, p);
            if a[i][i].abs() < 1e-10 {
                return None;
            }
            let d = a[i][i];
            for k in i..dim {
                a[i][k] /= d;
            }
            b[i] /= d;
            for j in 0..dim {
                if j != i {
                    let v = a[j][i];
                    for k in i..dim {
                        a[j][k] -= v * a[i][k];
                    }
                    b[j] -= v * b[i];
                }
            }
        }
        if b.iter().all(|v| v.is_finite()) {
            Some(b)
        } else {
            None
        }
    }
    fn mse(self, c: [f64; 4]) -> f64 {
        let mut v = self.yy;
        for j in 0..4 {
            v -= 2. * c[j] * self.b[j];
            for k in 0..4 {
                v += c[j] * c[k] * self.a[j][k];
            }
        }
        (v / self.count.max(1) as f64).max(0.)
    }
}
fn normal(g: V3, q: V3) -> Option<V3> {
    let b = dot(q, g);
    let disc = b * b - dot(g, g) + 1.;
    if disc <= 0. || b <= 0. {
        return None;
    }
    let t = b - disc.sqrt();
    (t > 0.).then(|| sub(scale(q, t), g))
}
// Unit-radius sphere. p[2] is f*R/(-Gz), not an orthographic surface.
fn globe(p: [f64; 3]) -> V3 {
    [(p[0] - 4000.) / p[2], (p[1] - 3000.) / p[2], -4000. / p[2]]
}
fn parameters(g: V3, r: f64) -> [f64; 3] {
    let p = project(g);
    [p[0], p[1], -4000. * r / g[2]]
}
#[derive(Clone)]
struct Trial {
    p: [f64; 3],
    stats: [Stats; 2],
    coef: [[f64; 4]; 3],
    errors: [f64; 3],
    physical: bool,
    // Coarse grid is independent of brightness. Adaptive refinements may be
    // selected only by the objective that proposed them (full/left/right).
    eligible: [bool; 3],
}
impl Trial {
    fn cv(&self) -> f64 {
        (self.stats[1].mse(self.coef[1]) * self.stats[1].count as f64
            + self.stats[0].mse(self.coef[2]) * self.stats[0].count as f64)
            / (self.stats[0].count + self.stats[1].count) as f64
    }
    fn value(&self) -> Value {
        json!({"sphere_center_sensor_px":[self.p[0],self.p[1]],"projected_radius_proxy_px":self.p[2],"mse":self.errors[0],"fixed_shape_spatial_cv_mse":self.cv(),"light_coefficients":self.coef[0],"nonnegative_ambient_unshadowed_directional_model":self.physical})
    }
}
fn trial(p: [f64; 3], points: &[Sample]) -> Option<Trial> {
    let g = globe(p);
    let mut stats = [Stats::default(); 2];
    let mut normals = Vec::with_capacity(points.len());
    for s in points {
        let n = normal(g, s.q)?;
        let x = [1., n[0], n[1], n[2]];
        stats[s.fold].add(x, s.value);
        normals.push(n);
    }
    let all = stats[0].plus(stats[1]);
    let coef = [all.fit(4)?, stats[0].fit(4)?, stats[1].fit(4)?];
    let physical = coef
        .iter()
        .all(|c| c[0] >= 0. && normals.iter().all(|&n| dot(n, [c[1], c[2], c[3]]) >= 0.));
    Some(Trial {
        p,
        stats,
        coef,
        errors: [
            all.mse(coef[0]),
            stats[0].mse(coef[1]),
            stats[1].mse(coef[2]),
        ],
        physical,
        eligible: [true; 3],
    })
}
fn control(points: &[Sample], e: Ellipse, dim: usize) -> Option<f64> {
    let mut s = [Stats::default(); 2];
    for p in points {
        let x = if dim == 1 {
            [1., 0., 0., 0.]
        } else {
            [
                1.,
                (p.xy[0] - e.center.0) / e.major_radius,
                (p.xy[1] - e.center.1) / e.major_radius,
                0.,
            ]
        };
        s[p.fold].add(x, p.value);
    }
    let (a, b) = (s[0].fit(dim)?, s[1].fit(dim)?);
    Some((s[0].mse(b) * s[0].count as f64 + s[1].mse(a) * s[1].count as f64) / points.len() as f64)
}
fn search(points: &[Sample], e: Ellipse) -> Vec<Trial> {
    let mut trials = vec![];
    for ix in -6..=6 {
        for iy in -6..=6 {
            for ir in 0..=6 {
                let p = [
                    e.center.0 + ix as f64 * e.major_radius / 4.,
                    e.center.1 + iy as f64 * e.major_radius / 4.,
                    e.major_radius * (1.5 + ir as f64 / 4.),
                ];
                if let Some(t) = trial(p, points) {
                    trials.push(t);
                }
            }
        }
    }
    // Refine separately for the full data and each TRAINING fold. Held-out
    // brightness never chooses that fold's center or radius.
    for objective in 0..3 {
        let Some(mut best) = trials
            .iter()
            .filter(|t| t.eligible[objective])
            .min_by(|a, b| a.errors[objective].total_cmp(&b.errors[objective]))
            .cloned()
        else {
            continue;
        };
        let mut step = e.major_radius / 8.;
        for _ in 0..5 {
            let at = best.p;
            for x in -1..=1 {
                for y in -1..=1 {
                    for r in -1..=1 {
                        let p = [
                            at[0] + x as f64 * step,
                            at[1] + y as f64 * step,
                            at[2] + r as f64 * step,
                        ];
                        if (p[0] - e.center.0).abs() > 1.5 * e.major_radius
                            || (p[1] - e.center.1).abs() > 1.5 * e.major_radius
                            || p[2] < 1.5 * e.major_radius
                            || p[2] > 3. * e.major_radius
                        {
                            continue;
                        }
                        if let Some(mut t) = trial(p, points) {
                            t.eligible = [false; 3];
                            t.eligible[objective] = true;
                            if t.errors[objective] < best.errors[objective] {
                                best = t.clone();
                            }
                            trials.push(t);
                        }
                    }
                }
            }
            step *= 0.5;
        }
    }
    trials
}
fn selected_cv(trials: &[Trial]) -> Option<(f64, bool)> {
    let a = trials
        .iter()
        .filter(|t| t.eligible[1])
        .min_by(|a, b| a.errors[1].total_cmp(&b.errors[1]))?;
    let b = trials
        .iter()
        .filter(|t| t.eligible[2])
        .min_by(|a, b| a.errors[2].total_cmp(&b.errors[2]))?;
    let score = (a.stats[1].mse(a.coef[1]) * a.stats[1].count as f64
        + b.stats[0].mse(b.coef[2]) * b.stats[0].count as f64)
        / (a.stats[1].count + b.stats[0].count) as f64;
    Some((score, a.physical && b.physical))
}
fn points(
    raw: &[u16],
    masks: &[u8],
    frame: &Value,
    e: Ellipse,
    threshold: u8,
    phase: usize,
) -> Vec<Sample> {
    let selected = sample_mask(masks, frame, e, threshold, 2);
    let (w, h) = (n(&frame["width"]) as usize, n(&frame["height"]) as usize);
    let (sx, sy) = (
        n(&frame["sensor_x"]) as usize,
        n(&frame["sensor_y"]) as usize,
    );
    let (px, py) = if phase == 0 { (2, 0) } else { (0, 2) };
    let mut out = vec![];
    for y in 0..h {
        if (y + sy) % 4 != py {
            continue;
        }
        for x in 0..w {
            if (x + sx) % 4 != px {
                continue;
            }
            let mx = ((x as f64 + 0.5) * 384. / w as f64) as usize;
            let my = ((y as f64 + 0.5) * 256. / h as f64) as usize;
            let v = raw[y * w + x];
            if selected[my * 384 + mx] && v > 4 && v < 1015 {
                let xy = [(x + sx) as f64, (y + sy) as f64];
                out.push(Sample {
                    xy,
                    q: camera_ray(xy),
                    value: v as f64,
                    fold: ((x + sx) / 24 + (y + sy) / 24) % 2,
                });
            }
        }
    }
    out
}
fn profile(trials: &[Trial], noise_scale: f64) -> Value {
    let Some(best) = trials
        .iter()
        .min_by(|a, b| a.errors[0].total_cmp(&b.errors[0]))
    else {
        return Value::Null;
    };
    let support=[0.01,0.05,0.10].map(|fraction| {
        let mut xmin=f64::INFINITY;let mut xmax=f64::NEG_INFINITY;let mut ymin=xmin;let mut ymax=xmax;let mut count=0;
        for t in trials.iter().filter(|t|t.errors[0]<=best.errors[0]+fraction*noise_scale) {xmin=xmin.min(t.p[0]);xmax=xmax.max(t.p[0]);ymin=ymin.min(t.p[1]);ymax=ymax.max(t.p[1]);count+=1;}
        json!({"excess_mse_fraction_of_constant_cv":fraction,"sampled_shapes":count,"center_span_px":[xmax-xmin,ymax-ymin],"center_box":[xmin,ymin,xmax,ymax],"confidence_interval":false})
    });
    json!({"best":best.value(),"profile_support":support,"sampled_shapes":trials.len(),"nested_spatial_cv":selected_cv(trials),"continuous_exclusion_proof":false})
}
fn analyze_points(
    points: &[Sample],
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
) -> (Value, Vec<Trial>) {
    if points.len() < 48 {
        return (
            json!({"usable":false,"samples":points.len(),"reason":"fewer than 48 native green samples"}),
            vec![],
        );
    }
    let Some(constant) = control(points, e, 1) else {
        return (
            json!({"usable":false,"reason":"spatial split lacks support"}),
            vec![],
        );
    };
    let gradient = control(points, e, 3);
    let trials = search(points, e);
    let candidates = rays.rays.map(|pose| {
        let t = (0..=26)
            .filter_map(|i| {
                let r = 1.5 + i as f64 * 0.05;
                trial(parameters(center(pose, r), r), points)
            })
            .collect::<Vec<_>>();
        json!({"profile":profile(&t,constant),"predictive":selected_cv(&t)})
    });
    let a = candidates[0]["predictive"][0].as_f64();
    let b = candidates[1]["predictive"][0].as_f64();
    let physical = candidates.iter().all(|c| c["predictive"][1] == true);
    let choice = match (a, b) {
        (Some(a), Some(b))
            if physical && (a - b).abs() > 0.10 * constant && a.min(b) < 0.8 * constant =>
        {
            Some(usize::from(b < a))
        }
        _ => None,
    };
    let beats = choice
        .zip(gradient)
        .is_some_and(|(k, g)| candidates[k]["predictive"][0].as_f64().unwrap() < 0.95 * g);
    (
        json!({"usable":!trials.is_empty(),"samples":points.len(),"constant_cv_mse":constant,"gradient_cv_mse":gradient,"independent_center":profile(&trials,constant),"candidates":candidates,"conditional_preference":choice,"beats_gradient":beats}),
        trials,
    )
}

/// Known-light oracle on real sample layouts. Albedo varies per sample and is
/// canceled when the recovered rho*N is normalized. No real RAW is relabeled.
fn oracle(points: &[Sample], rays: TheoreticalEllipseExplanations) -> Value {
    let mut cases = vec![];
    let lights = [[120., 0., 200.], [0., 120., 200.], [-120., -120., 200.]];
    for branch in 0..2 {
        for (scenario, noise, rotation_degrees, gain_error) in [
            ("ideal", 0., 0f64, 0.),
            ("noise3", 3., 0., 0.),
            ("noise10", 10., 0., 0.),
            ("light_strength_error5pct", 3., 0., 0.05),
            ("light_direction_error3deg", 3., 3., 0.),
            ("light_direction_error10deg", 3., 10., 0.),
        ] {
            let (st, ct) = rotation_degrees.to_radians().sin_cos();
            let actual_lights =
                lights.map(|l| [ct * l[0] + st * l[2], l[1], -st * l[0] + ct * l[2]]);
            let g = scale(center(rays.rays[branch], 2.3), 1. / 2.3);
            let mut normal_system = Stats::default();
            let mut error = 0.;
            let mut count = 0;
            for (index, s) in points.iter().enumerate() {
                let Some(nn) = normal(g, s.q) else {
                    continue;
                };
                if actual_lights.iter().any(|l| dot(*l, nn) <= 0.) {
                    continue;
                }
                let mut photo = Stats::default();
                let albedo = 0.8 + 0.4 * ((index * 71 % 101) as f64 / 100.);
                for (k, l) in lights.iter().enumerate() {
                    let perturb = ((index * 31 + k * 47) % 101) as f64 / 50. - 1.;
                    photo.add(
                        [l[0], l[1], l[2], 0.],
                        albedo * dot(actual_lights[k], nn) * (1. + gain_error * [1., -1., 0.5][k])
                            + noise * perturb,
                    );
                }
                let Some(v) = photo.fit(3) else {
                    continue;
                };
                let observed = unit([v[0], v[1], v[2]]);
                error += dot(nn, observed).clamp(-1., 1.).acos().to_degrees();
                count += 1;
                // (I-qq^T) G/R = -(I-qq^T) N, exact perspective geometry.
                for j in 0..3 {
                    let mut p = [0.; 4];
                    for k in 0..3 {
                        p[k] = (j == k) as u8 as f64 - s.q[j] * s.q[k];
                    }
                    normal_system.add(p, -observed[j] + s.q[j] * dot(s.q, observed));
                }
            }
            let recovered = normal_system.fit(3).map(|v| [v[0], v[1], v[2]]);
            let true_px = project(g);
            let estimated = recovered.map(project);
            let center_error = estimated.map(|p| (p[0] - true_px[0]).hypot(p[1] - true_px[1]));
            if scenario == "ideal" && count >= 12 {
                assert!(
                    center_error.is_some_and(|v| v < 1e-5),
                    "noise-free normal-to-center reconstruction failed"
                );
            }
            cases.push(json!({"branch":branch,"scenario":scenario,"noise_code_amplitude":noise,"systematic_light_direction_error_deg":rotation_degrees,"systematic_light_strength_error_fraction":gain_error,"samples":count,"mean_normal_error_degrees":if count>0 {Some(error/count as f64)} else {None},"true_center_sensor_px":true_px,"estimated_center_sensor_px":estimated,"center_error_px":center_error,"true_globe_center_per_radius":g,"estimated_globe_center_per_radius":recovered}));
        }
    }
    let globes = rays.rays.map(|p| scale(center(p, 2.3), 1. / 2.3));
    let ambient_samples = points
        .iter()
        .filter(|p| globes.iter().all(|&g| normal(g, p.q).is_some()))
        .cloned()
        .map(|mut p| {
            p.value = 300.;
            p
        })
        .collect::<Vec<_>>();
    let ambient = globes.map(|g| trial(parameters(g, 1.), &ambient_samples).map(|t| t.cv()));
    for mse in ambient.into_iter().flatten() {
        assert!(
            mse < 1e-5,
            "constant illumination acquired false shape information"
        );
    }
    json!({"synthetic_only":true,"lights":lights,"ambient_subtracted_exactly":true,"albedo_varies_0_8_to_1_2":true,"cases":cases,"uniform_ambient_counterexample":{"same_samples_for_both_branches":ambient_samples.len(),"branch_cv_mse":ambient,"intensity":300.}})
}
fn render_photo(
    row: &Value,
    raw: &[u16],
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
    points: &[Sample],
    trials: &[Trial],
    v: &Value,
    out: &Path,
) -> Result<()> {
    let f = &row["frame"];
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
    let rgb = preview::color_preview(raw, w, h, origin[0] as u32, origin[1] as u32, 100, None);
    let bgra = rgb
        .into_iter()
        .flat_map(|c| {
            [
                (c & 255) as u8,
                ((c >> 8) & 255) as u8,
                ((c >> 16) & 255) as u8,
                255,
            ]
        })
        .collect::<Vec<_>>();
    let mut c = Canvas::new(1800, 1040)?;
    c.clear();
    c.text(
        20.,
        35.,
        25.,
        WHITE,
        &format!(
            "Sclera shading and independent globe center | {} record {}",
            row["provider"].as_str().unwrap(),
            row["record"]
        ),
    );
    c.text(20.,68.,17.,MUTED,"Native green photosites. Unknown ambient and three light coefficients. No iris-sign or focus-region labels enter the center fit.");
    let best = trials
        .iter()
        .min_by(|a, b| a.errors[0].total_cmp(&b.errors[0]));
    for panel in 0..2 {
        let x = 20. + 600. * panel as f64;
        let y = 115.;
        let s = 570. / w as f64;
        c.text(
            x,
            102.,
            21.,
            WHITE,
            if panel == 0 {
                "RAW and measured brightness"
            } else {
                "Free-center shading fit"
            },
        );
        c.image(&bgra, w, h, x, y, 570., h as f64 * s);
        c.clipped(x, y, 570., h as f64 * s, |c| {
            let mapped = |p: [f64; 2]| [x + (p[0] - origin[0]) * s, y + (p[1] - origin[1]) * s];
            for p in points {
                let value = if panel == 0 {
                    p.value
                } else {
                    best.and_then(|t| {
                        normal(globe(t.p), p.q).map(|n| {
                            t.coef[0][0] + dot(n, [t.coef[0][1], t.coef[0][2], t.coef[0][3]])
                        })
                    })
                    .unwrap_or(p.value)
                };
                let gray = (value / 1023.).clamp(0., 1.);
                let xy = mapped(p.xy);
                c.dot(xy[0], xy[1], 2.2, [gray, gray, gray], true);
            }
            c.path(
                &e.dense_points(180)
                    .iter()
                    .map(|&(u, v)| mapped([u, v]))
                    .collect::<Vec<_>>(),
                1.5,
                WHITE,
            );
            for (k, p) in rays.rays.iter().enumerate() {
                let xy = mapped(project(center(*p, 2.1)));
                c.dot(xy[0], xy[1], 7., if k == 0 { CYAN } else { PINK }, false);
            }
            if panel == 1 {
                if let Some(t) = best {
                    c.path(
                        &silhouette(globe(t.p), 1.)
                            .into_iter()
                            .map(mapped)
                            .collect::<Vec<_>>(),
                        2.,
                        GREEN,
                    );
                    let p = mapped([t.p[0], t.p[1]]);
                    c.dot(p[0], p[1], 5., GREEN, true);
                }
            }
        });
    }
    c.text(
        1220.,
        102.,
        21.,
        WHITE,
        "Center profile: brighter = lower error",
    );
    let (x0, y0, size) = (1240., 145., 500.);
    let a = e.major_radius;
    let map = |p: [f64; 2]| {
        [
            x0 + size * ((p[0] - e.center.0) / a + 1.5) / 3.,
            y0 + size * ((p[1] - e.center.1) / a + 1.5) / 3.,
        ]
    };
    let min = best.map_or(0., |t| t.errors[0]);
    let span = v["constant_cv_mse"].as_f64().unwrap_or(1.).max(1.);
    let mut grid = BTreeMap::<(i32, i32), f64>::new();
    for t in trials {
        let key = (
            ((t.p[0] - e.center.0) / a * 4.).round() as i32,
            ((t.p[1] - e.center.1) / a * 4.).round() as i32,
        );
        grid.entry(key)
            .and_modify(|v| *v = v.min(t.errors[0]))
            .or_insert(t.errors[0]);
    }
    for ((xx, yy), error) in grid {
        let q = map([
            e.center.0 + xx as f64 * a / 4.,
            e.center.1 + yy as f64 * a / 4.,
        ]);
        let l = (-10. * (error - min) / span).exp();
        c.rect(
            q[0] - 18.,
            q[1] - 18.,
            36.,
            36.,
            [0.12 + 0.75 * l, 0.14 + 0.65 * l, 0.22 + 0.3 * l],
        );
    }
    for (k, p) in rays.rays.iter().enumerate() {
        let pts = [1.5, 2.8].map(|r| map(project(center(*p, r))));
        c.line(pts[0], pts[1], 3., if k == 0 { CYAN } else { PINK });
    }
    if let Some(t) = best {
        let xy = map([t.p[0], t.p[1]]);
        c.dot(xy[0], xy[1], 7., GREEN, false);
    }
    c.text(
        20.,
        610.,
        19.,
        WHITE,
        &format!(
            "Samples {} | constant CV MSE {} | gradient CV MSE {}",
            points.len(),
            v["constant_cv_mse"]
                .as_f64()
                .map_or("unavailable".into(), |x| format!("{x:.1}")),
            v["gradient_cv_mse"]
                .as_f64()
                .map_or("unavailable".into(), |x| format!("{x:.1}"))
        ),
    );
    c.text(
        20.,
        648.,
        18.,
        WHITE,
        &format!(
            "A/B predictive MSE: {} / {} | conditional preference {}",
            v["candidates"][0]["predictive"][0],
            v["candidates"][1]["predictive"][0],
            v["conditional_preference"]
        ),
    );
    if let Some(t) = best {
        c.text(20.,686.,18.,WHITE,&format!("Free center ({:.1},{:.1}) | radius proxy {:.1}px | nonnegative unshadowed model: {}",t.p[0]-origin[0],t.p[1]-origin[1],t.p[2],t.physical));
    }
    c.text(20.,740.,18.,MUTED,"Cyan/pink: proposed globe centers. Green: brightness-only best fit. Heatmap covers +/-1.5 iris radii in image coordinates.");
    c.text(20.,775.,18.,MUTED,"Profile explores unknown shape AND illumination. A broad valley means many centers explain the same light pattern.");
    c.text(20.,810.,18.,MUTED,"Shape and illumination are selected only on training blocks; predictive scores use other 24x24 sensor blocks.");
    c.text(20.,845.,18.,MUTED,"Masks remain unverified. Real shadows, sclera albedo, glasses and reflections violate this simple diffuse model.");
    c.text(20.,880.,18.,MUTED,"Fit quality is not sign accuracy. Finite profile support is a sensitivity diagnostic, not a confidence interval or uniqueness proof.");
    c.text(20.,925.,18.,MUTED,"Only upstream area-admitted frames. The ellipse and frontal-equivalent area are unchanged; independent scale is unavailable.");
    c.png(out)
}
pub(super) fn analyze(
    row: &Value,
    raw: &[u16],
    masks: &[u8],
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
    out: &Path,
    show: bool,
    do_control: bool,
    thresholds: [u8; 2],
) -> Result<(Value, Option<Value>)> {
    let mut sensor = e;
    sensor.center.0 += n(&row["frame"]["sensor_x"]) as f64;
    sensor.center.1 += n(&row["frame"]["sensor_y"]) as f64;
    let mut variants = vec![];
    let mut choices = vec![];
    let mut controls = None;
    for threshold in thresholds {
        for phase in 0..2 {
            let points = points(raw, masks, &row["frame"], e, threshold, phase);
            let (mut v, trials) = analyze_points(&points, sensor, rays);
            v["mask_threshold"] = json!(threshold);
            v["green_phase"] = json!(phase);
            if threshold == thresholds[0] && phase == 0 {
                if show {
                    render_photo(
                        row,
                        raw,
                        sensor,
                        rays,
                        &points,
                        &trials,
                        &v,
                        &out.join(format!(
                            "photometry-{}-{}.png",
                            row["record"],
                            row["provider"].as_str().unwrap()
                        )),
                    )?;
                }
                if do_control && points.len() >= 48 {
                    controls = Some(
                        json!({"record":row["record"],"provider":row["provider"],"oracle":oracle(&points,rays)}),
                    );
                }
            }
            choices.push(v["conditional_preference"].clone());
            variants.push(v);
        }
    }
    let stable = choices.iter().all(|v| v.is_number() && v == &choices[0]);
    let beats = stable && variants.iter().all(|v| v["beats_gradient"] == true);
    Ok((
        json!({"record":row["record"],"provider":row["provider"],"source":row["source"],"eye":row["eye"],"sequence":row["sequence"],"source_ns":row["source_ns"],"raw_sha256":row["raw_sha256"],"area_admission":row["area_admission"],"usable":variants.iter().all(|v|v["usable"]==true),"stable_nominal_preference":stable,"beats_gradient_all_variants":beats,"variants":variants,"physical_sign_accuracy":null}),
        controls,
    ))
}
