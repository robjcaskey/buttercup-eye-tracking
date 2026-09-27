//! Unknown light direction shared over short source-time windows. Diagnostic.
//! Per-frame ambient and nonnegative gain remain free. No sign labels enter.
use super::*;
use std::time::Instant;

#[derive(Clone, Copy)]
struct DirectionFit {
    error: f64,
    shape: usize,
    coefficient: [f64; 4],
}
struct Model {
    samples: Vec<Sample>,
    shapes: [Vec<Trial>; 2],
    // [light][training fold][branch]. Radius and radiometry are selected only
    // on that fold; held-out brightness is never part of this cache.
    fits: Vec<[[Option<DirectionFit>; 2]; 2]>,
    constant: f64,
    gradient: f64,
    train_scale: [f64; 2],
    independent: [Option<(f64, bool)>; 2],
}
struct Observation {
    row: Value,
    timestamp: u64,
    class: String,
    models: Vec<Option<Model>>,
}

// Geometry-only feasibility: do not let an unphysical lower-loss light hide
// a valid alternative and then reject the whole observation after fitting.
// Most directions are decided by a conservative normal bounding box; only
// directions crossing that box need the exact per-sample dot products.
struct NormalSupport {
    normals: Vec<V3>,
    lower: V3,
    upper: V3,
}
impl NormalSupport {
    fn new(t: &Trial, samples: &[Sample]) -> Self {
        let normals = samples
            .iter()
            .map(|s| normal(globe(t.p), s.q).expect("supported trial normal"))
            .collect::<Vec<_>>();
        let lower = [0, 1, 2].map(|j| normals.iter().map(|n| n[j]).fold(f64::INFINITY, f64::min));
        let upper = [0, 1, 2].map(|j| {
            normals
                .iter()
                .map(|n| n[j])
                .fold(f64::NEG_INFINITY, f64::max)
        });
        Self {
            normals,
            lower,
            upper,
        }
    }
    fn permits(&self, light: V3) -> bool {
        let bound = |lower: bool| {
            (0..3)
                .map(|j| {
                    light[j]
                        * if (light[j] >= 0.) == lower {
                            self.lower[j]
                        } else {
                            self.upper[j]
                        }
                })
                .sum::<f64>()
        };
        if bound(true) >= 0. {
            return true;
        }
        if bound(false) < 0. {
            return false;
        }
        self.normals.iter().all(|&n| dot(light, n) >= 0.)
    }
}

fn directions() -> Vec<V3> {
    // Fixed, brightness-independent full-sphere search. Finite-grid profiles
    // are heuristic support; they are not continuous exclusion certificates.
    (0..1024)
        .map(|i| {
            let z = 1. - 2. * (i as f64 + 0.5) / 1024.;
            let theta = i as f64 * std::f64::consts::PI * (3. - 5f64.sqrt());
            let r = (1. - z * z).sqrt();
            [r * theta.cos(), r * theta.sin(), z]
        })
        .collect()
}

fn directional(s: Stats, light: V3) -> Option<([f64; 4], f64)> {
    if s.count < 12 {
        return None;
    }
    let count = s.count as f64;
    let x = (0..3).map(|j| light[j] * s.a[0][j + 1]).sum::<f64>();
    let xx = (0..3)
        .flat_map(|j| (0..3).map(move |k| light[j] * light[k] * s.a[j + 1][k + 1]))
        .sum::<f64>();
    let xy = (0..3).map(|j| light[j] * s.b[j + 1]).sum::<f64>();
    let y = s.b[0];
    let det = count * xx - x * x;
    // Exact two-variable nonnegative least squares, including both edges.
    let mut options = vec![(y.max(0.) / count, 0.), (0., (xy / xx.max(1e-12)).max(0.))];
    if det > 1e-12 * (count * xx).abs().max(1.) {
        let gain = (count * xy - x * y) / det;
        let ambient = (y - gain * x) / count;
        if gain >= 0. && ambient >= 0. {
            options.push((ambient, gain));
        }
    }
    options
        .into_iter()
        .map(|(a, g)| {
            let coefficient = [a, g * light[0], g * light[1], g * light[2]];
            (coefficient, s.mse(coefficient))
        })
        .filter(|(_, e)| e.is_finite())
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

fn prepare(
    samples: Vec<Sample>,
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
    lights: &[V3],
) -> Option<Model> {
    if samples.len() < 48
        || (0..2).any(|fold| samples.iter().filter(|s| s.fold == fold).count() < 12)
    {
        return None;
    }
    let constant = control(&samples, e, 1)?;
    let gradient = control(&samples, e, 3)?;
    let shapes = rays.rays.map(|p| {
        (0..=26)
            .filter_map(|i| {
                let r = 1.5 + i as f64 * 0.05;
                trial(parameters(center(p, r), r), &samples)
            })
            .collect::<Vec<_>>()
    });
    if shapes.iter().any(Vec::is_empty) {
        return None;
    }
    let train_scale = [0, 1].map(|fold| {
        let s = shapes[0][0].stats[fold];
        s.mse([s.b[0] / s.count as f64, 0., 0., 0.]).max(4.)
    });
    let independent = shapes.each_ref().map(|t| selected_cv(t));
    let support = shapes.each_ref().map(|family| {
        family
            .iter()
            .map(|t| {
                let normals = NormalSupport::new(t, &samples);
                lights
                    .iter()
                    .map(|&l| normals.permits(l))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let fits = lights
        .iter()
        .enumerate()
        .map(|(li, &light)| {
            [0, 1].map(|fold| {
                [0, 1].map(|branch| {
                    shapes[branch]
                        .iter()
                        .enumerate()
                        .filter_map(|(shape, t)| {
                            let s = t.stats[fold];
                            let (coefficient, error) = if support[branch][shape][li] {
                                directional(s, light)?
                            } else {
                                // With a shadowed sample the only member of
                                // this explicitly unshadowed model is gain=0.
                                let c = [s.b[0].max(0.) / s.count as f64, 0., 0., 0.];
                                (c, s.mse(c))
                            };
                            Some(DirectionFit {
                                error,
                                shape,
                                coefficient,
                            })
                        })
                        .min_by(|a, b| a.error.total_cmp(&b.error))
                })
            })
        })
        .collect();
    Some(Model {
        samples,
        shapes,
        fits,
        constant,
        gradient,
        train_scale,
        independent,
    })
}

fn fit_value(
    m: &Model,
    branch: usize,
    fold: usize,
    light_index: usize,
    lights: &[V3],
) -> Option<Value> {
    let f = m.fits[light_index][fold][branch]?;
    let t = &m.shapes[branch][f.shape];
    let minimum_directional = m
        .samples
        .iter()
        .filter_map(|s| normal(globe(t.p), s.q))
        .map(|n| dot(n, [f.coefficient[1], f.coefficient[2], f.coefficient[3]]))
        .fold(f64::INFINITY, f64::min);
    Some(
        json!({"light_index":light_index,"light_direction":lights[light_index],"shape":t.p,
        "coefficients":f.coefficient,"train_mse":f.error,"held_mse":t.stats[1-fold].mse(f.coefficient),
        "held_count":t.stats[1-fold].count,"minimum_directional_raw_codes":minimum_directional,
        "unshadowed_physical":minimum_directional>=-1e-8}),
    )
}

fn independent_grid(m: &Model, lights: &[V3]) -> Value {
    let fits = [0, 1].map(|fold| {
        [0, 1].map(|branch| {
            let li = (0..lights.len())
                .min_by(|&a, &b| {
                    m.fits[a][fold][branch]
                        .unwrap()
                        .error
                        .total_cmp(&m.fits[b][fold][branch].unwrap().error)
                })
                .unwrap();
            fit_value(m, branch, fold, li, lights).unwrap()
        })
    });
    let cv = [0, 1].map(|branch| {
        let count = fits
            .iter()
            .map(|f| f[branch]["held_count"].as_f64().unwrap())
            .sum::<f64>();
        fits.iter()
            .map(|f| {
                f[branch]["held_count"].as_f64().unwrap() * f[branch]["held_mse"].as_f64().unwrap()
            })
            .sum::<f64>()
            / count
    });
    let best = usize::from(cv[1] < cv[0]);
    let energy_agrees = (0..2).all(|f| {
        (fits[f][1 - best]["train_mse"].as_f64().unwrap()
            - fits[f][best]["train_mse"].as_f64().unwrap())
            / m.train_scale[f]
            > 0.10
    });
    let physical = (0..2).all(|f| fits[f][best]["unshadowed_physical"] == true);
    let choice = (energy_agrees
        && physical
        && cv[1 - best] - cv[best] > 0.10 * m.constant.max(4.)
        && cv[best] < 0.8 * m.constant
        && cv[best] < 0.95 * m.gradient)
        .then_some(best);
    json!({"candidate_cv_mse":cv,"conditional_preference":choice,"physical":physical,"training_folds_agree":energy_agrees})
}

fn analyze(target: &Observation, context: &[&Observation], variant: usize, lights: &[V3]) -> Value {
    let Some(m) = &target.models[variant] else {
        return json!({"usable":false,"reason":"target lacks common sphere/sample support"});
    };
    let included = context
        .iter()
        .filter(|o| o.models[variant].is_some())
        .copied()
        .collect::<Vec<_>>();
    let distinct_times = included
        .iter()
        .map(|o| o.timestamp)
        .collect::<BTreeSet<_>>()
        .len();
    if distinct_times < 5 {
        return json!({"usable":false,"reason":"fewer than five fresh source times with both globe families","distinct_times":distinct_times});
    }
    let mut candidates = vec![];
    let mut fold_profiles = vec![];
    for fold in 0..2 {
        let costs = lights
            .iter()
            .enumerate()
            .map(|(li, _)| {
                // Separability eliminates the exponential sign-sequence search:
                // min_L sum_i min_(sign_i,radius_i,ambient_i,gain_i) E_i.
                let mut background = 0.;
                for o in &included {
                    if n(&o.row["record"]) == n(&target.row["record"]) {
                        continue;
                    }
                    let mm = o.models[variant].as_ref().unwrap();
                    let best = mm.fits[li][fold]
                        .iter()
                        .flatten()
                        .map(|f| f.error)
                        .fold(f64::INFINITY, f64::min);
                    background += best / mm.train_scale[fold];
                }
                [0, 1].map(|b| {
                    background
                        + m.fits[li][fold][b]
                            .map_or(f64::INFINITY, |f| f.error / m.train_scale[fold])
                })
            })
            .collect::<Vec<_>>();
        let best = [0, 1].map(|b| {
            (0..lights.len())
                .min_by(|&a, &c| costs[a][b].total_cmp(&costs[c][b]))
                .unwrap()
        });
        let energies = [costs[best[0]][0], costs[best[1]][1]];
        let alternatives = [0, 1].map(|b| {
            costs
                .iter()
                .enumerate()
                .filter(|(_, c)| c[b] <= energies[b] + 0.05)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        });
        let mut fits = vec![];
        for b in 0..2 {
            let mut f = fit_value(m, b, fold, best[b], lights).unwrap();
            let context_physical = included.iter().all(|o| {
                let mm = o.models[variant].as_ref().unwrap();
                let branch = if n(&o.row["record"]) == n(&target.row["record"]) {
                    b
                } else {
                    usize::from(
                        mm.fits[best[b]][fold][1].unwrap().error
                            < mm.fits[best[b]][fold][0].unwrap().error,
                    )
                };
                fit_value(mm, branch, fold, best[b], lights)
                    .is_some_and(|v| v["unshadowed_physical"] == true)
            });
            f["all_context_unshadowed_physical"] = json!(context_physical);
            f["joint_training_energy"] = json!(energies[b]);
            f["nearby_grid_directions"] = json!(alternatives[b].len());
            fits.push(f);
        }
        candidates.push(fits);
        fold_profiles.push(json!({"training_fold":fold,"forced_branch_energy":energies,"energy_gap":energies[1]-energies[0],
            "nearby_light_direction_indices":alternatives,"finite_direction_grid_only":true}));
    }
    let predictive = [0, 1].map(|b| {
        let mut sum = 0.;
        let mut count = 0.;
        for f in 0..2 {
            let n = candidates[f][b]["held_count"].as_f64().unwrap();
            sum += n * candidates[f][b]["held_mse"].as_f64().unwrap();
            count += n;
        }
        sum / count
    });
    let best = usize::from(predictive[1] < predictive[0]);
    let other = 1 - best;
    let energy_agrees = fold_profiles.iter().all(|p| {
        let e = p["forced_branch_energy"].as_array().unwrap();
        e[other].as_f64().unwrap() - e[best].as_f64().unwrap() > 0.10
    });
    let physical = (0..2).all(|f| candidates[f][best]["all_context_unshadowed_physical"] == true);
    let predictive_gap = (predictive[other] - predictive[best]) / m.constant.max(4.);
    let choice = (energy_agrees
        && physical
        && predictive_gap > 0.10
        && predictive[best] < 0.8 * m.constant
        && predictive[best] < 0.95 * m.gradient)
        .then_some(best);
    json!({"usable":true,"samples":m.samples.len(),"context_records":included.iter().map(|o|o.row["record"].clone()).collect::<Vec<_>>(),
        "context_distinct_source_times":distinct_times,"span_ms":(target.timestamp-included.iter().map(|o|o.timestamp).min().unwrap()) as f64/1e6,
        "constant_cv_mse":m.constant,"gradient_cv_mse":m.gradient,"old_unconstrained_independent_light_candidate_cv":m.independent,
        "matched_independent_grid":independent_grid(m,lights),
        "shared_light_candidate_cv_mse":predictive,"gap_over_constant_mse":predictive_gap,
        "training_folds_agree_with_cv":energy_agrees,"all_context_physical":physical,
        "passes_constant_control":predictive[best]<0.8*m.constant,"passes_gradient_control":predictive[best]<0.95*m.gradient,
        "conditional_preference":choice,
        "fold_profiles":fold_profiles,"candidate_fits_by_training_fold":candidates})
}

fn synthetic_controls(observations: &[Observation], lights: &[V3]) -> Result<Value> {
    let illumination = unit([0.3, -0.2, 1.]);
    let mut cases = vec![];
    let mut maximum_stats_error = 0f64;
    let mut support_checks = 0usize;
    for (name, upward, flat) in [
        ("upward_3code_noise", true, false),
        ("downward_3code_noise", false, false),
        ("uniform_ambient", true, true),
    ] {
        let mut generated = vec![];
        let mut truths = BTreeMap::new();
        for o in observations
            .iter()
            .filter(|o| o.row["provider"] == "sam" && o.models[0].is_some())
            .take(40)
        {
            let e = shape(&o.row["fit"]["ellipse"])?;
            let mut sensor = e;
            sensor.center.0 += n(&o.row["frame"]["sensor_x"]) as f64;
            sensor.center.1 += n(&o.row["frame"]["sensor_y"]) as f64;
            let rays =
                TheoreticalEllipseExplanations::from_ellipse(sensor, [4000.; 2], [4000., 3000.])
                    .ok_or("synthetic native candidates")?;
            let truth = usize::from(if upward {
                rays.rays[1].direction[1] < rays.rays[0].direction[1]
            } else {
                rays.rays[1].direction[1] > rays.rays[0].direction[1]
            });
            let g = rays.rays.map(|p| scale(center(p, 2.15), 1. / 2.15));
            // The identical retained layout must intersect BOTH nominal globes.
            // No candidate-specific mask or visibility advantage is introduced.
            let samples = o.models[0]
                .as_ref()
                .unwrap()
                .samples
                .iter()
                .filter_map(|s| {
                    let nn = [normal(g[0], s.q)?, normal(g[1], s.q)?];
                    if nn.iter().any(|&v| dot(v, illumination) <= 0.05) {
                        return None;
                    }
                    let mut s = s.clone();
                    let seed = (s.xy[0] as u64).wrapping_mul(137)
                        ^ (s.xy[1] as u64).wrapping_mul(173)
                        ^ n(&o.row["record"]);
                    let noise = ((seed % 41) as f64 - 20.) * 3. / 20.;
                    s.value = if flat {
                        300.
                    } else {
                        30. + 220. * dot(nn[truth], illumination) + noise
                    };
                    Some(s)
                })
                .collect::<Vec<_>>();
            let Some(model) = prepare(samples, sensor, rays, lights) else {
                continue;
            };
            // Cross-check sufficient-statistics loss against direct native
            // sample evaluation, at a light direction outside the fixed grid.
            for fold in 0..2 {
                let t = trial(
                    parameters(center(rays.rays[truth], 2.15), 2.15),
                    &model.samples,
                )
                .ok_or("known synthetic sphere")?;
                if fold == 0 {
                    let support = NormalSupport::new(&t, &model.samples);
                    for &light in lights {
                        if support.permits(light)
                            != support.normals.iter().all(|&n| dot(light, n) >= 0.)
                        {
                            return Err("normal-bound physical feasibility parity failed".into());
                        }
                        support_checks += 1;
                    }
                }
                let (coef, stats_error) =
                    directional(t.stats[fold], illumination).ok_or("synthetic directional fit")?;
                let points = model
                    .samples
                    .iter()
                    .filter(|s| s.fold == fold)
                    .collect::<Vec<_>>();
                let direct = points
                    .iter()
                    .map(|s| {
                        let nn = normal(g[truth], s.q).unwrap();
                        let residual = s.value - coef[0] - dot(nn, [coef[1], coef[2], coef[3]]);
                        residual * residual
                    })
                    .sum::<f64>()
                    / points.len() as f64;
                maximum_stats_error = maximum_stats_error.max((direct - stats_error).abs());
                if (direct - stats_error).abs() > 1e-6 {
                    return Err("directional sufficient-statistics parity failed".into());
                }
            }
            truths.insert(n(&o.row["record"]), truth);
            generated.push(Observation {
                row: o.row.clone(),
                timestamp: o.timestamp,
                class: "synthetic".into(),
                models: vec![Some(model)],
            });
        }
        let mut results = vec![];
        let mut start = 0;
        for (i, o) in generated.iter().enumerate() {
            if i > 0 && o.timestamp.saturating_sub(generated[i - 1].timestamp) > 300_000_000 {
                start = i;
            }
            let context = generated[start..]
                .iter()
                .take_while(|p| p.timestamp <= o.timestamp)
                .filter(|p| o.timestamp - p.timestamp <= 2_000_000_000)
                .collect::<Vec<_>>();
            let v = analyze(o, &context, 0, lights);
            let choice = v["conditional_preference"].as_u64();
            if flat && choice.is_some() {
                return Err("uniform-ambient control spuriously chose sign".into());
            }
            results.push(json!({"record":o.row["record"],"known_synthetic_truth":truths[&n(&o.row["record"])],"outcome":match choice {None=>"abstained",Some(c) if c as usize==truths[&n(&o.row["record"]) ]=>"correct",Some(_)=>"wrong"},"result":v}));
        }
        cases.push(json!({"scenario":name,"generated_rows":generated.len(),"results":results}));
    }
    Ok(
        json!({"synthetic_only":true,"known_illumination":illumination,"ambient":30.,"gain":220.,"nominal_radius":2.15,"bounded_noise_raw_codes":3.,"uniform_ambient_code":300.,"maximum_direct_vs_stats_mse_error":maximum_stats_error,"physical_feasibility_bound_direct_checks":support_checks,"cases":cases,"scope":"First forty usable SAM layouts, arbitrary synthetic globe motion from native candidates; not anatomical or recorded sign truth"}),
    )
}

pub(crate) fn read_raw(row: &Value, bundles: &mut BTreeMap<String, BundleSource>) -> Result<Vec<u16>> {
    let p = row["raw_source"].as_str().ok_or("RAW source")?;
    if !bundles.contains_key(p) {
        bundles.insert(p.to_owned(), BundleSource::open(Path::new(p))?);
    }
    let f = &row["frame"];
    let bytes = bundles[p].read_range(
        row["stream_entry"].as_str().ok_or("stream entry")?,
        n(&f["offset"]),
        n(&f["length"]) as usize,
    )?;
    if archive::digest(&bytes) != row["raw_sha256"] {
        return Err("RAW hash mismatch".into());
    }
    Ok(raw10::try_unpack_raw10(
        &bytes,
        n(&f["width"]) as usize,
        n(&f["height"]) as usize,
        n(&f["stride"]) as usize,
    )?)
}

fn render_temporal(
    o: &Observation,
    result: &Value,
    model: usize,
    raw: &[u16],
    path: &Path,
) -> Result<()> {
    let row = &o.row;
    let f = &row["frame"];
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
    let bgra = preview::color_preview(raw, w, h, origin[0] as u32, origin[1] as u32, 100, None)
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
    let e = shape(&row["fit"]["ellipse"])?;
    let mut c = Canvas::new(1800, 830)?;
    c.clear();
    c.text(
        20.,
        35.,
        25.,
        WHITE,
        &format!(
            "Shared short-window light | {} RAW {} | native samples, no sign truth",
            row["provider"].as_str().unwrap(),
            row["record"]
        ),
    );
    c.text(20.,70.,18.,MUTED,"Same RAW and mask for both candidates. Radius, light and radiometry selected on training blocks; filled dots are held-out blocks.");
    for panel in 0..3 {
        let x = 20. + panel as f64 * 594.;
        let y = 135.;
        let scale = 560. / w as f64;
        c.text(
            x,
            115.,
            20.,
            WHITE,
            if panel == 0 {
                "Measured RAW and fitted iris"
            } else if panel == 1 {
                "A: residuals under fitted shared light"
            } else {
                "B: residuals under fitted shared light"
            },
        );
        c.image(&bgra, w, h, x, y, 560., h as f64 * scale);
        c.path(
            &e.dense_points(160)
                .iter()
                .map(|&(u, v)| [x + u * scale, y + v * scale])
                .collect::<Vec<_>>(),
            1.2,
            WHITE,
        );
        if panel > 0 && result["usable"] == true {
            let fit = &result["candidate_fits_by_training_fold"][0][panel - 1];
            let p: [f64; 3] = serde_json::from_value(fit["shape"].clone())?;
            let coef: [f64; 4] = serde_json::from_value(fit["coefficients"].clone())?;
            let m = o.models[model].as_ref().unwrap();
            for s in &m.samples {
                let nn = normal(globe(p), s.q).ok_or("display sample outside fitted sphere")?;
                let error = (s.value - coef[0] - dot(nn, [coef[1], coef[2], coef[3]])) / 40.;
                let color = if error >= 0. {
                    [1., (1. - error).clamp(0., 1.), 0.1]
                } else {
                    [0.1, (1. + error).clamp(0., 1.), 1.]
                };
                c.dot(
                    x + (s.xy[0] - origin[0]) * scale,
                    y + (s.xy[1] - origin[1]) * scale,
                    2.,
                    color,
                    s.fold == 1,
                );
            }
            c.cross(
                x + (p[0] - origin[0]) * scale,
                y + (p[1] - origin[1]) * scale,
                8.,
                ORANGE,
            );
            c.text(
                x,
                552.,
                18.,
                WHITE,
                &format!(
                    "CV MSE {:.1}; first-fold held MSE {:.1}",
                    result["shared_light_candidate_cv_mse"][panel - 1]
                        .as_f64()
                        .unwrap(),
                    fit["held_mse"].as_f64().unwrap()
                ),
            );
            c.text(
                x,
                580.,
                16.,
                MUTED,
                &format!(
                    "L=({:.2}, {:.2}, {:.2}); physical {}",
                    fit["light_direction"][0].as_f64().unwrap(),
                    fit["light_direction"][1].as_f64().unwrap(),
                    fit["light_direction"][2].as_f64().unwrap(),
                    fit["all_context_unshadowed_physical"]
                ),
            );
        }
    }
    c.text(
        20.,
        630.,
        20.,
        WHITE,
        &format!(
            "Usable {} | fresh source times {} | span {} ms | provisional choice {}",
            result["usable"],
            result["context_distinct_source_times"],
            result["span_ms"],
            result["conditional_preference"]
        ),
    );
    c.text(20.,674.,18.,MUTED,"Orange crosses: candidate globe centers. Red/blue: signed RAW-code residuals, on an identical +/-40-code color scale.");
    c.text(20.,712.,18.,MUTED,"Unknown constant light direction is an assumption. Each exposure has its own nonnegative ambient and gain; shadows/albedo may violate it.");
    c.text(20.,750.,18.,MUTED,"No measured light, independent scale, reviewed sclera boundary or physical sign truth is supplied by this diagnostic.");
    c.text(20.,790.,18.,MUTED,&format!("Mask threshold {}; native green phase {}; window {} ms. A single-setting preference is not a stable sign choice.",result["mask_threshold"],result["green_phase"],result["window_ms"]));
    c.png(path)
}

pub(crate) fn run(
    area_dir: &str,
    fresh_dir: &str,
    output: &str,
    anatomy_dir: Option<&str>,
) -> Result<()> {
    let started = Instant::now();
    let area = Path::new(area_dir);
    let fresh = Path::new(fresh_dir);
    let out = Path::new(output);
    if out.exists() {
        return Err("output exists".into());
    }
    let summary = load(&area.join("summary.json"))?;
    if summary["complete"] != true || summary["schema"] != "buttercup-area-first-focus-v1" {
        return Err("completed area-first run required".into());
    }
    let fresh_bytes = fs::read(fresh.join("frames.jsonl"))?;
    if archive::digest(&fresh_bytes) != summary["fresh_frames_sha256"] {
        return Err("fresh source changed".into());
    }
    let originals = rows(&fresh.join("frames.jsonl"))?
        .into_iter()
        .map(|r| (n(&r["record"]), r))
        .collect::<BTreeMap<_, _>>();
    let class_rows = rows(&area.join("classifications.jsonl"))?;
    let sources = class_rows
        .iter()
        .filter(|r| r["class"] == "multiple")
        .map(|r| n(&r["source"]))
        .collect::<BTreeSet<_>>();
    let classes = class_rows
        .iter()
        .map(|r| {
            (
                (n(&r["record"]), r["provider"].as_str().unwrap().to_string()),
                r["class"].as_str().unwrap().to_string(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let anatomy = anatomy_dir
        .map(|p| {
            anatomy_masks::AnatomyMasks::open(
                p,
                &archive::digest(&fs::read(area.join("summary.json"))?),
            )
        })
        .transpose()?;
    let thresholds = if anatomy.is_some() {
        [128, 153]
    } else {
        [179, 230]
    };
    let inputs = rows(&area.join("retained-inputs.jsonl"))?
        .into_iter()
        .filter(|r| sources.contains(&n(&r["source"])))
        .filter(|r| anatomy.as_ref().is_none_or(|a| a.contains(n(&r["record"]))))
        .collect::<Vec<_>>();
    fs::create_dir(out)?;
    let lights = directions();
    let mut bundles = BTreeMap::new();
    let mut observations = vec![];
    for (i, row) in inputs.into_iter().enumerate() {
        let id = n(&row["record"]);
        let provider = row["provider"].as_str().ok_or("provider")?;
        if row["area_admission"]["accepted"] != true
            || row["raw_sha256"] != originals[&id]["raw_sha256"]
            || row["fit"] != originals[&id][provider]["fit"]
        {
            return Err("area/RAW/fit identity mismatch".into());
        }
        let masks = if let Some(a) = &anatomy {
            a.get(&row)?
        } else {
            let m =
                fs::read(fresh.join(originals[&id]["obelisk"]["masks"].as_str().ok_or("masks")?))?;
            if m.len() != 6 * 384 * 256
                || archive::digest(&m) != originals[&id]["obelisk"]["masks_sha256"]
            {
                return Err("mask hash/shape".into());
            }
            m
        };
        let raw = read_raw(&row, &mut bundles)?;
        let e = shape(&row["fit"]["ellipse"])?;
        let mut sensor = e;
        sensor.center.0 += n(&row["frame"]["sensor_x"]) as f64;
        sensor.center.1 += n(&row["frame"]["sensor_y"]) as f64;
        let rays = TheoreticalEllipseExplanations::from_ellipse(sensor, [4000.; 2], [4000., 3000.])
            .ok_or("native circle solve")?;
        let samples = thresholds
            .into_iter()
            .flat_map(|threshold| (0..2).map(move |phase| (threshold, phase)))
            .map(|(threshold, phase)| points(&raw, &masks, &row["frame"], e, threshold, phase))
            .collect::<Vec<_>>();
        let models = std::thread::scope(|scope| {
            let handles = samples
                .into_iter()
                .map(|p| {
                    let lights = &lights;
                    scope.spawn(move || prepare(p, sensor, rays, lights))
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|h| h.join().expect("photometry worker"))
                .collect::<Vec<_>>()
        });
        let class = classes[&(id, provider.to_owned())].clone();
        let timestamp = row["source_ns"].as_str().ok_or("source time")?.parse()?;
        observations.push(Observation {
            row,
            timestamp,
            class,
            models,
        });
        if i % 25 == 0 {
            eprintln!(
                "TEMPORAL LIGHT prepared {} observations, {:.1}s",
                i + 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    observations.sort_by_key(|o| {
        (
            o.row["provider"].as_str().unwrap().to_owned(),
            n(&o.row["source"]),
            n(&o.row["epoch"]),
            o.timestamp,
            n(&o.row["eye"]),
        )
    });
    let controls = synthetic_controls(&observations, &lights)?;
    fs::write(
        out.join("controls.json"),
        serde_json::to_vec_pretty(&controls)?,
    )?;
    let mut writer = BufWriter::new(fs::File::create(out.join("temporal.jsonl"))?);
    let mut counts = BTreeMap::new();
    let mut shown = BTreeSet::new();
    let mut review = vec![];
    let mut choice_reviews = 0usize;
    let mut segment_start = 0;
    for (i, o) in observations.iter().enumerate() {
        if i > 0 {
            let p = &observations[i - 1];
            if o.row["provider"] != p.row["provider"]
                || o.row["source"] != p.row["source"]
                || o.row["epoch"] != p.row["epoch"]
                || o.timestamp.saturating_sub(p.timestamp) > 300_000_000
            {
                segment_start = i;
            }
        }
        let end = (i + 1..observations.len())
            .find(|&j| {
                observations[j].timestamp != o.timestamp
                    || observations[j].row["provider"] != o.row["provider"]
                    || observations[j].row["source"] != o.row["source"]
                    || observations[j].row["epoch"] != o.row["epoch"]
            })
            .unwrap_or(observations.len());
        let mut variants = vec![];
        for window_ns in [1_000_000_000u64, 2_000_000_000] {
            let context = observations[segment_start..end]
                .iter()
                .filter(|p| o.timestamp.saturating_sub(p.timestamp) <= window_ns)
                .collect::<Vec<_>>();
            for v in 0..4 {
                let mut result = analyze(o, &context, v, &lights);
                result["window_ms"] = json!(window_ns / 1_000_000);
                result["mask_threshold"] = json!(thresholds[v / 2]);
                result["green_phase"] = json!(v % 2);
                variants.push(result);
            }
        }
        let stable = variants[0]["conditional_preference"].is_number()
            && variants
                .iter()
                .all(|v| v["conditional_preference"] == variants[0]["conditional_preference"]);
        let provider = o.row["provider"].as_str().unwrap();
        count(
            &mut counts,
            format!(
                "{provider}/{}/{}",
                o.class,
                if stable {
                    "stable_conditional_choice"
                } else {
                    "not_stable"
                }
            ),
        );
        for (v, r) in variants.iter().enumerate() {
            count(
                &mut counts,
                format!(
                    "{provider}/{}/v{v}/{}",
                    o.class,
                    if r["usable"] != true {
                        "unavailable"
                    } else if r["conditional_preference"].is_number() {
                        "conditional_choice"
                    } else {
                        "inconclusive"
                    }
                ),
            );
        }
        let record = json!({"record":o.row["record"],"provider":provider,"class":o.class,"raw_sha256":o.row["raw_sha256"],"area_admission":o.row["area_admission"],"source_ns":o.row["source_ns"],"stable_conditional_preference":if stable{variants[0]["conditional_preference"].clone()}else{Value::Null},"variants":variants,"physical_sign_truth":null});
        writeln!(writer, "{}", serde_json::to_string(&record)?)?;
        let key = format!(
            "{provider}/{}/{}/{stable}/{}",
            o.row["eye"], o.class, variants[4]["usable"]
        );
        if shown.insert(key) && review.len() < 16 {
            let name = format!("temporal-{}-{provider}.png", o.row["record"]);
            let raw = read_raw(&o.row, &mut bundles)?;
            render_temporal(o, &variants[4], 0, &raw, &out.join(&name))?;
            review.push(json!({"image":name,"record":o.row["record"],"provider":provider}));
        }
        if choice_reviews < 8 {
            if let Some(v) = variants
                .iter()
                .position(|v| v["conditional_preference"].is_number())
            {
                let raw = read_raw(&o.row, &mut bundles)?;
                // Inspect the selected native CFA phase and the companion
                // phase on the same source exposure; never draw phase zero
                // samples with phase one's radiometric fit.
                for variant in [v, v ^ 1] {
                    let name = format!(
                        "temporal-preference-{}-{provider}-v{variant}.png",
                        o.row["record"]
                    );
                    render_temporal(o, &variants[variant], variant % 4, &raw, &out.join(&name))?;
                    review.push(json!({"image":name,"record":o.row["record"],"provider":provider,"variant":variant,"single_setting_preference_review":true}));
                }
                choice_reviews += 1;
            }
        }
    }
    writer.flush()?;
    fs::write(out.join("review.json"), serde_json::to_vec_pretty(&review)?)?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"counts":counts,"observations":observations.len(),"seconds":started.elapsed().as_secs_f64(),"light_directions":lights,"radius_grid":[1.5,2.8,0.05],"windows_ms":[1000,2000],"maximum_source_gap_ms":300,"minimum_distinct_source_times":5,"mask_thresholds":thresholds,"anatomy":anatomy.as_ref().map(|a|&a.meta),"area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"retained_inputs_sha256":archive::digest(&fs::read(area.join("retained-inputs.jsonl"))?),"executable_sha256":archive::digest(&fs::read("/proc/self/exe")?),"policy":"Shared unknown direction; per-frame nonnegative ambient/gain/radius, unshadowed feasibility enforced during search, arbitrary branch sequence; source-causal photometry after retrospective area gate; no physical sign truth or calibrated confidence; historical custom masks diagnostic only"}),
        )?,
    )?;
    Ok(())
}
