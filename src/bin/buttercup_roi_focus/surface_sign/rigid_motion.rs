//! Proper rigid globe rotations constrained by both native iris interpretations.
//! Parameters use training correspondences only; patch-separated LOO evaluates
//! all tentative interior matches. These are diagnostic, conditional choices.
use super::*;

#[derive(Clone)]
struct Fit {
    source: usize,
    target: usize,
    radius: f64,
    torsion: f64,
    train_mse: f64,
    predictions: Vec<Option<[f64; 2]>>,
}
impl Fit {
    fn summary(&self) -> Value {
        json!({"source_branch":self.source,"target_branch":self.target,"radius":self.radius,"torsion_radians":self.torsion,"train_mse_px2":self.train_mse})
    }
    fn value(&self) -> Value {
        let mut value = self.summary();
        value["predicted_sensor"] = json!(self.predictions);
        value
    }
}
struct Shape {
    source: usize,
    target: usize,
    radius: f64,
    target_center: V3,
    target_basis: [V3; 3],
    source_local: Vec<Option<V3>>,
    target_local: Vec<Option<V3>>,
}
fn shape_trials(a: &Frame, b: &Frame, m: &[NativePatchCorrespondence]) -> Vec<Shape> {
    let mut trials = vec![];
    for sa in 0..2 {
        for sb in 0..2 {
            for ri in 0..=26 {
                let r = 1.5 + ri as f64 * 0.05;
                let ga = scale(center(a.rays.rays[sa], r), 1. / r);
                let gb = center(b.rays.rays[sb], r);
                let ba = basis(a.rays.rays[sa].direction);
                let bb = basis(b.rays.rays[sb].direction);
                let source_local = m
                    .iter()
                    .map(|p| {
                        normal(ga, camera_ray(p.previous_sensor_px.map(f64::from)))
                            .map(|v| ba.map(|axis| dot(axis, v)))
                    })
                    .collect();
                let target_local = m
                    .iter()
                    .map(|p| {
                        normal(
                            scale(gb, 1. / r),
                            camera_ray(p.current_sensor_px.map(f64::from)),
                        )
                        .map(|v| bb.map(|axis| dot(axis, v)))
                    })
                    .collect();
                trials.push(Shape {
                    source: sa,
                    target: sb,
                    radius: r,
                    target_center: gb,
                    target_basis: bb,
                    source_local,
                    target_local,
                });
            }
        }
    }
    trials
}
fn prediction(s: &Shape, i: usize, theta: f64) -> Option<[f64; 2]> {
    let p = s.source_local[i]?;
    let (sin, cos) = theta.sin_cos();
    let n = add(
        add(
            scale(s.target_basis[0], cos * p[0] - sin * p[1]),
            scale(s.target_basis[1], sin * p[0] + cos * p[1]),
        ),
        scale(s.target_basis[2], p[2]),
    );
    let point = add(s.target_center, scale(n, s.radius));
    if point[2] >= -1e-8 || dot(n, unit(point)) >= -1e-6 {
        return None;
    }
    Some(project(point))
}
fn square(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}
fn fit(s: &Shape, m: &[NativePatchCorrespondence], train: &[usize]) -> Option<Fit> {
    let (mut cosine, mut sine) = (0., 0.);
    for &i in train {
        let p = s.source_local[i]?;
        let q = s.target_local[i]?;
        cosine += p[0] * q[0] + p[1] * q[1];
        sine += p[0] * q[1] - p[1] * q[0];
    }
    if cosine.hypot(sine) < 1e-10 {
        return None;
    }
    // Exact least-squares torsion in the unit-surface-normal coordinates.
    // Radius and source branch are selected by training pixel error below.
    let theta = sine.atan2(cosine);
    let mut error = 0.;
    for &i in train {
        error += square(
            prediction(s, i, theta)?,
            m[i].current_sensor_px.map(f64::from),
        );
    }
    Some(Fit {
        source: s.source,
        target: s.target,
        radius: s.radius,
        torsion: theta,
        train_mse: error / train.len() as f64,
        predictions: (0..m.len()).map(|i| prediction(s, i, theta)).collect(),
    })
}
fn distant(a: [f32; 2], b: [f32; 2], guard: f32) -> bool {
    (a[0] - b[0]).abs() > guard || (a[1] - b[1]).abs() > guard
}

fn baseline(
    m: &[NativePatchCorrespondence],
    train: &[usize],
    held: usize,
    similarity: bool,
) -> Option<f64> {
    let mut a = [0.; 2];
    let mut b = a;
    for &i in train {
        for j in 0..2 {
            a[j] += m[i].previous_sensor_px[j] as f64;
            b[j] += m[i].current_sensor_px[j] as f64;
        }
    }
    for j in 0..2 {
        a[j] /= train.len() as f64;
        b[j] /= train.len() as f64;
    }
    let (mut c, mut d, mut denom) = (0., 0., 0.);
    if similarity {
        for &i in train {
            let p = [
                m[i].previous_sensor_px[0] as f64 - a[0],
                m[i].previous_sensor_px[1] as f64 - a[1],
            ];
            let q = [
                m[i].current_sensor_px[0] as f64 - b[0],
                m[i].current_sensor_px[1] as f64 - b[1],
            ];
            c += p[0] * q[0] + p[1] * q[1];
            d += p[0] * q[1] - p[1] * q[0];
            denom += p[0] * p[0] + p[1] * p[1];
        }
        if denom < 1e-8 {
            return None;
        }
        c /= denom;
        d /= denom;
    } else {
        c = 1.;
    }
    let p = [
        m[held].previous_sensor_px[0] as f64 - a[0],
        m[held].previous_sensor_px[1] as f64 - a[1],
    ];
    Some(square(
        [b[0] + c * p[0] - d * p[1], b[1] + d * p[0] + c * p[1]],
        m[held].current_sensor_px.map(f64::from),
    ))
}

pub(super) fn analyze(a: &Frame, b: &Frame, m: &[NativePatchCorrespondence], guard: f32) -> Value {
    if m.len() < 6 {
        return json!({"usable":false,"reason":"fewer than six tentative interior correspondences","matches":m.len(),"guard_px":guard});
    }
    let shapes = shape_trials(a, b, m);
    let mut folds = vec![];
    for held in 0..m.len() {
        let train = (0..m.len())
            .filter(|&i| {
                i != held
                    && distant(m[i].previous_sensor_px, m[held].previous_sensor_px, guard)
                    && distant(m[i].current_sensor_px, m[held].current_sensor_px, guard)
            })
            .collect::<Vec<_>>();
        if train.len() < 3 {
            continue;
        }
        let selected = [0, 1].map(|target| {
            shapes
                .iter()
                .filter(|s| s.target == target)
                .filter_map(|s| fit(s, m, &train))
                .min_by(|a, b| a.train_mse.total_cmp(&b.train_mse))
        });
        let fits = selected.each_ref().map(|f| f.as_ref().map(Fit::summary));
        let errors = selected.each_ref().map(|f| {
            f.as_ref()
                .and_then(|f| f.predictions[held])
                .map(|p| square(p, m[held].current_sensor_px.map(f64::from)))
        });
        folds.push(json!({"held_index":held,"training_indices":train,"candidate_fits":fits,"held_squared_errors":errors,"translation_squared_error":baseline(m,&train,held,false),"similarity_squared_error":baseline(m,&train,held,true)}));
    }
    let covered = folds.len();
    if covered < 6 || covered * 5 < m.len() * 4 {
        return json!({"usable":false,"reason":"insufficient patch-separated held-out coverage","matches":m.len(),"held_points":covered,"guard_px":guard});
    }
    let mut candidate_cv = [0.; 2];
    let mut base = [0.; 2];
    for f in &folds {
        for b in 0..2 {
            let Some(e) = f["held_squared_errors"][b].as_f64() else {
                return json!({"usable":false,"reason":"a trained globe family lacks held-out projection support","matches":m.len(),"held_points":covered,"guard_px":guard});
            };
            candidate_cv[b] += e;
        }
        for (i, key) in ["translation_squared_error", "similarity_squared_error"]
            .iter()
            .enumerate()
        {
            let Some(e) = f[key].as_f64() else {
                return json!({"usable":false,"reason":"baseline singular","guard_px":guard});
            };
            base[i] += e;
        }
    }
    candidate_cv = candidate_cv.map(|e| (e / covered as f64).sqrt());
    base = base.map(|e| (e / covered as f64).sqrt());
    let winner = usize::from(candidate_cv[1] < candidate_cv[0]);
    let training_agreement = folds
        .iter()
        .filter(|f| {
            f["candidate_fits"][winner]["train_mse_px2"]
                .as_f64()
                .unwrap()
                < f["candidate_fits"][1 - winner]["train_mse_px2"]
                    .as_f64()
                    .unwrap()
        })
        .count();
    let source_zero = folds
        .iter()
        .filter(|f| f["candidate_fits"][winner]["source_branch"] == 0)
        .count();
    let source = usize::from(source_zero * 2 < covered);
    let source_agreement = if source == 0 {
        source_zero
    } else {
        covered - source_zero
    };
    let selected = (candidate_cv[winner] <= 2.
        && candidate_cv[1 - winner] >= candidate_cv[winner] + 1.
        && candidate_cv[1 - winner] >= 1.5 * candidate_cv[winner]
        && candidate_cv[winner] <= base[0].min(base[1]) + 0.5
        && training_agreement * 5 >= covered * 4
        && source_agreement * 5 >= covered * 4)
        .then_some([source, winner]);
    let full_train = (0..m.len()).collect::<Vec<_>>();
    let full = [0, 1].map(|target| {
        shapes
            .iter()
            .filter(|s| s.target == target)
            .filter_map(|s| fit(s, m, &full_train))
            .min_by(|a, b| a.train_mse.total_cmp(&b.train_mse))
            .map(|f| f.value())
    });
    json!({"usable":true,"matches":m.len(),"held_points":covered,"guard_px":guard,"candidate_cv_rmse_px":candidate_cv,"baseline_cv_rmse_px":{"translation":base[0],"similarity":base[1]},"training_target_agreement_fraction":training_agreement as f64/covered as f64,"training_source_agreement_fraction":source_agreement as f64/covered as f64,"conditional_pair_preference":selected,"folds":folds,"full_data_fits_for_display_only":full,"physical_sign_truth":null})
}

pub(super) fn render_fit(
    a: &Frame,
    b: &Frame,
    m: &[NativePatchCorrespondence],
    result: &Value,
    path: &Path,
) -> Result<()> {
    let f = &b.row["frame"];
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
    let pixels =
        preview::color_preview(&b.raw, w, h, origin[0] as u32, origin[1] as u32, 100, None)
            .into_iter()
            .flat_map(|v| {
                [
                    (v & 255) as u8,
                    ((v >> 8) & 255) as u8,
                    ((v >> 16) & 255) as u8,
                    255,
                ]
            })
            .collect::<Vec<_>>();
    let mut c = Canvas::new(1800, 850)?;
    c.clear();
    c.text(
        22.,
        36.,
        25.,
        WHITE,
        &format!(
            "Rigid sclera | {} RAW {} to {} | measured vs fitted, no sign truth",
            b.row["provider"].as_str().unwrap(),
            a.row["record"],
            b.row["record"]
        ),
    );
    c.text(22.,74.,18.,MUTED,"Orange circles: native patch matches. Blue crosses: full-data predictions shown for inspection; scores use held-out points.");
    for target in 0..2 {
        let x = 22. + 884. * target as f64;
        let y = 125.;
        let s = 852. / w as f64;
        c.image(&pixels, w, h, x, y, 852., h as f64 * s);
        let fit = &result["full_data_fits_for_display_only"][target];
        let r = fit["radius"].as_f64().ok_or("display fit missing")?;
        let g = center(b.rays.rays[target], r);
        let ellipse = shape(&b.row["fit"]["ellipse"])?;
        c.clipped(x, y, 852., h as f64 * s, |c| {
            c.path(
                &silhouette(g, r)
                    .iter()
                    .map(|p| [x + (p[0] - origin[0]) * s, y + (p[1] - origin[1]) * s])
                    .collect::<Vec<_>>(),
                1.5,
                [0.7, 0.7, 0.7],
            );
            c.path(
                &ellipse
                    .dense_points(160)
                    .iter()
                    .map(|&(u, v)| [x + u * s, y + v * s])
                    .collect::<Vec<_>>(),
                1.,
                WHITE,
            );
            for (i, point) in m.iter().enumerate() {
                let q = [
                    x + (point.current_sensor_px[0] as f64 - origin[0]) * s,
                    y + (point.current_sensor_px[1] as f64 - origin[1]) * s,
                ];
                c.dot(q[0], q[1], 3., ORANGE, false);
                if let Some(p) = fit["predicted_sensor"][i].as_array() {
                    let p = [
                        x + (p[0].as_f64().unwrap() - origin[0]) * s,
                        y + (p[1].as_f64().unwrap() - origin[1]) * s,
                    ];
                    c.cross(p[0], p[1], 4., [0.25, 0.7, 1.]);
                    c.path(&[q, p], 1., [0.25, 0.7, 1.]);
                }
            }
        });
        c.text(
            x,
            108.,
            20.,
            WHITE,
            &format!(
                "Target {} | source {} | R {:.2} | torsion {:.2} degrees",
                target,
                fit["source_branch"],
                r,
                fit["torsion_radians"].as_f64().unwrap().to_degrees()
            ),
        );
        c.text(
            x,
            728.,
            20.,
            WHITE,
            &format!(
                "Held-out RMSE {:.2} px",
                result["candidate_cv_rmse_px"][target].as_f64().unwrap()
            ),
        );
    }
    c.text(
        22.,
        776.,
        18.,
        MUTED,
        &format!(
            "Baseline CV RMSE: translation {:.2}, similarity {:.2} px. Conditional pair {:?}.",
            result["baseline_cv_rmse_px"]["translation"]
                .as_f64()
                .unwrap(),
            result["baseline_cv_rmse_px"]["similarity"]
                .as_f64()
                .unwrap(),
            result["conditional_pair_preference"]
        ),
    );
    c.text(22.,815.,18.,MUTED,"Gray outlines are the projected native 3D globe hypotheses, not observed anatomical boundaries. Conics, masks and sphere model remain uncertain.");
    c.png(path)
}
