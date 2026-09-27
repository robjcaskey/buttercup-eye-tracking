//! Shared illumination across two exact-source eye exposures. Still diagnostic.
use super::*;
struct Observation {
    row: Value,
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
    samples: Vec<Vec<Sample>>,
    thresholds: [u8; 2],
}
type Key = (String, u64, u64, u64, String);
#[derive(Default)]
pub(crate) struct PairStream {
    waiting: BTreeMap<Key, Observation>,
}
impl PairStream {
    pub(crate) fn pending(&self) -> usize {
        self.waiting.len()
    }
    pub(crate) fn push(
        &mut self,
        row: &Value,
        raw: &[u16],
        masks: &[u8],
        e: Ellipse,
        rays: TheoreticalEllipseExplanations,
        thresholds: [u8; 2],
    ) -> Option<Value> {
        assert_eq!(
            row["area_admission"]["accepted"], true,
            "non-admitted row in paired shading"
        );
        let mut samples = vec![];
        for threshold in thresholds {
            for phase in 0..2 {
                samples.push(points(raw, masks, &row["frame"], e, threshold, phase));
            }
        }
        let key = (
            row["provider"].as_str().unwrap().into(),
            n(&row["source"]),
            n(&row["epoch"]),
            n(&row["sequence"]),
            row["source_ns"].as_str().unwrap().into(),
        );
        let mut sensor = e;
        sensor.center.0 += n(&row["frame"]["sensor_x"]) as f64;
        sensor.center.1 += n(&row["frame"]["sensor_y"]) as f64;
        let current = Observation {
            row: row.clone(),
            e: sensor,
            rays,
            samples,
            thresholds,
        };
        let Some(previous) = self.waiting.remove(&key) else {
            self.waiting.insert(key, current);
            return None;
        };
        assert_eq!(
            previous.thresholds, thresholds,
            "inconsistent paired mask thresholds"
        );
        assert_ne!(
            previous.row["eye"], current.row["eye"],
            "duplicate same-eye source in joint shading"
        );
        let eyes = if n(&previous.row["eye"]) < n(&current.row["eye"]) {
            [previous, current]
        } else {
            [current, previous]
        };
        assert_eq!([n(&eyes[0].row["eye"]), n(&eyes[1].row["eye"])], [1, 2]);
        let mut variants = vec![];
        for v in 0..4 {
            for flexible_gain in [false, true] {
                let mut r = analyze_pair(&eyes, v, flexible_gain);
                r["mask_threshold"] = json!(thresholds[v / 2]);
                r["green_phase"] = json!(v % 2);
                r["flexible_relative_albedo"] = json!(flexible_gain);
                variants.push(r);
            }
        }
        let stable = variants.iter().all(|v| {
            v["conditional_preference"].is_number()
                && v["conditional_preference"] == variants[0]["conditional_preference"]
        });
        let beats = stable && variants.iter().all(|v| v["beats_gradient"] == true);
        Some(
            json!({"records":[eyes[0].row["record"],eyes[1].row["record"]],"provider":key.0,"source":key.1,"epoch":key.2,"sequence":key.3,"source_ns":key.4,"raw_sha256":[eyes[0].row["raw_sha256"],eyes[1].row["raw_sha256"]],"both_upstream_area_admitted":true,"stable_preference":stable,"stable_beats_gradient":beats,"variants":variants,"physical_sign_truth":null}),
        )
    }
}
#[derive(Clone, Copy, Default)]
struct JointStats {
    a: [[f64; 5]; 5],
    b: [f64; 5],
    yy: f64,
    count: usize,
}
impl JointStats {
    // Each eye's sufficient statistics are formed once per sphere. Reuse them
    // for every paired shape/gain hypothesis; no repeated per-pixel fitting.
    fn add(&mut self, s: Stats, eye: usize, gain: f64) {
        let map = [eye, 2, 3, 4];
        let w = [1., gain, gain, gain];
        self.yy += s.yy;
        self.count += s.count;
        for j in 0..4 {
            self.b[map[j]] += w[j] * s.b[j];
            for k in 0..4 {
                self.a[map[j]][map[k]] += w[j] * w[k] * s.a[j][k];
            }
        }
    }
    fn fit(&self) -> Option<[f64; 5]> {
        let mut a = self.a;
        let mut b = self.b;
        for i in 0..5 {
            let p = (i..5).max_by(|&j, &k| a[j][i].abs().total_cmp(&a[k][i].abs()))?;
            a.swap(i, p);
            b.swap(i, p);
            if a[i][i].abs() < 1e-10 {
                return None;
            }
            let d = a[i][i];
            for k in i..5 {
                a[i][k] /= d;
            }
            b[i] /= d;
            for j in 0..5 {
                if j != i {
                    let v = a[j][i];
                    for k in i..5 {
                        a[j][k] -= v * a[i][k];
                    }
                    b[j] -= v * b[i];
                }
            }
        }
        b.iter().all(|v| v.is_finite()).then_some(b)
    }
    fn mse(&self, c: [f64; 5]) -> f64 {
        let mut v = self.yy;
        for j in 0..5 {
            v -= 2. * c[j] * self.b[j];
            for k in 0..5 {
                v += c[j] * c[k] * self.a[j][k];
            }
        }
        (v / self.count.max(1) as f64).max(0.)
    }
}
#[derive(Clone)]
struct Fit {
    shapes: [[f64; 3]; 2],
    gain: f64,
    train: [f64; 2],
    held: [f64; 2],
    coef: [[f64; 5]; 2],
}
fn fit(a: &Trial, b: &Trial, gain: f64) -> Option<Fit> {
    let mut stats = [JointStats::default(); 2];
    for fold in 0..2 {
        stats[fold].add(a.stats[fold], 0, 1.);
        stats[fold].add(b.stats[fold], 1, gain);
    }
    let coef = [stats[0].fit()?, stats[1].fit()?];
    Some(Fit {
        shapes: [a.p, b.p],
        gain,
        train: [stats[0].mse(coef[0]), stats[1].mse(coef[1])],
        held: [stats[1].mse(coef[0]), stats[0].mse(coef[1])],
        coef,
    })
}
fn physical(f: &Fit, fold: usize, eyes: &[Observation; 2], variant: usize) -> bool {
    let c = f.coef[fold];
    c[0] >= 0.
        && c[1] >= 0.
        && (0..2).all(|eye| {
            eyes[eye].samples[variant].iter().all(|s| {
                normal(globe(f.shapes[eye]), s.q)
                    .is_some_and(|nn| dot(nn, [c[2], c[3], c[4]]) >= 0.)
            })
        })
}
fn analyze_pair(eyes: &[Observation; 2], v: usize, flexible_gain: bool) -> Value {
    let counts = eyes.each_ref().map(|e| e.samples[v].len());
    if counts.iter().any(|&n| n < 48) {
        return json!({"usable":false,"samples":counts,"reason":"both eyes need at least 48 native samples"});
    }
    let total = (counts[0] + counts[1]) as f64;
    let weighted = |a: f64, b: f64| (a * counts[0] as f64 + b * counts[1] as f64) / total;
    let baselines: Vec<_> = eyes
        .iter()
        .map(|e| {
            (
                control(&e.samples[v], e.e, 1),
                control(&e.samples[v], e.e, 3),
            )
        })
        .collect();
    let Some((constant, gradient)) = baselines[0]
        .0
        .zip(baselines[1].0)
        .zip(baselines[0].1.zip(baselines[1].1))
        .map(|((a, b), (x, y))| (weighted(a, b), weighted(x, y)))
    else {
        return json!({"usable":false,"reason":"insufficient spatial blocks"});
    };
    let shapes: Vec<_> = eyes
        .iter()
        .map(|e| {
            e.rays.rays.map(|pose| {
                (0..=13)
                    .filter_map(|i| {
                        let r = 1.5 + i as f64 * 0.1;
                        trial(parameters(center(pose, r), r), &e.samples[v])
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let fold_counts = [0, 1].map(|fold| {
        eyes.iter()
            .map(|e| e.samples[v].iter().filter(|s| s.fold == fold).count())
            .sum::<usize>()
    });
    let mut candidates = vec![];
    for left in 0..2 {
        for right in 0..2 {
            let mut best: [Option<Fit>; 2] = [None, None];
            let mut fitted = 0;
            for a in &shapes[0][left] {
                for b in &shapes[1][right] {
                    for gain in if flexible_gain {
                        &[0.75, 1., 1.25][..]
                    } else {
                        &[1.][..]
                    } {
                        if let Some(f) = fit(a, b, *gain) {
                            fitted += 1;
                            for fold in 0..2 {
                                if best[fold]
                                    .as_ref()
                                    .is_none_or(|p| f.train[fold] < p.train[fold])
                                {
                                    best[fold] = Some(f.clone());
                                }
                            }
                        }
                    }
                }
            }
            let a = best[0].as_ref();
            let b = best[1].as_ref();
            let score = a.zip(b).map(|(a, b)| {
                (a.held[0] * fold_counts[1] as f64 + b.held[1] * fold_counts[0] as f64) / total
            });
            let plausible = a
                .zip(b)
                .is_some_and(|(a, b)| physical(a, 0, eyes, v) && physical(b, 1, eyes, v));
            let independent = selected_cv(&shapes[0][left])
                .zip(selected_cv(&shapes[1][right]))
                .map(|(a, b)| weighted(a.0, b.0));
            candidates.push(json!({"branches":[left,right],"shared_light_predictive_mse":score,"independent_lights_predictive_mse":independent,"physical_directional_light":plausible,"fitted_shapes":fitted,"selected_relative_gains":[a.map(|f|f.gain),b.map(|f|f.gain)],"training_fold_globe_parameters":[a.map(|f|f.shapes),b.map(|f|f.shapes)],"training_fold_lights":[a.map(|f|f.coef[0]),b.map(|f|f.coef[1])]}));
        }
    }
    let mut ranked = candidates
        .iter()
        .enumerate()
        .filter_map(|(k, v)| v["shared_light_predictive_mse"].as_f64().map(|x| (k, x)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let choice = if ranked.len() == 4
        && candidates[ranked[0].0]["physical_directional_light"] == true
        && ranked[1].1 - ranked[0].1 > 0.10 * constant
        && ranked[0].1 < 0.8 * constant
    {
        Some(ranked[0].0)
    } else {
        None
    };
    json!({"usable":ranked.len()==4,"samples":counts,"constant_predictive_mse":constant,"gradient_predictive_mse":gradient,"candidates":candidates,"conditional_preference":choice,"beats_gradient":choice.is_some_and(|_|ranked[0].1<0.95*gradient),"complete_four_way_gap_over_constant_mse":if ranked.len()==4 {Some((ranked[1].1-ranked[0].1)/constant)}else{None}})
}
