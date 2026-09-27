//! Persistent reference memberships and shared 3D surfaces across training frames.
//! Odd target images are never consulted until the complete discovery fit is frozen.
use super::super::motion::{self, Model, Picture};
use super::super::P;
use super::pixels::{self, Patch};
use serde::Serialize;

fn error(p: &Patch, b: &Picture, m: Model) -> f64 {
    pixels::error(p, b, m)
}
fn score(ps: &[&Patch], b: &Picture, m: Model) -> f64 {
    if ps.len() < 6 {
        return 1.;
    }
    let mut vs = ps.iter().map(|p| error(p, b, m)).collect::<Vec<_>>();
    vs.sort_by(f64::total_cmp);
    let n = (vs.len() * 9 / 10).max(1);
    vs[..n].iter().sum::<f64>() / n as f64
}
fn pose(ps: &[&Patch], b: &Picture, mut m: Model) -> Model {
    let mut best = score(ps, b, m);
    for round in 0..6 {
        let shrink = 0.5f64.powi(round);
        for _ in 0..2 {
            for i in 0..6 {
                let step = match i {
                    0..=2 => 0.12,
                    3..=4 => 0.65,
                    _ => 0.04,
                } * shrink;
                let old = m;
                for sign in [-1., 1.] {
                    let mut c = old;
                    c.set(i, old.get(i) + sign * step);
                    if !c.valid() {
                        continue;
                    }
                    let loss = score(ps, b, c);
                    if loss < best {
                        best = loss;
                        m = c;
                    }
                }
            }
        }
    }
    m
}
#[derive(Clone, Serialize)]
pub struct Sequence {
    pub poses: Vec<Model>,
    pub shape: [f64; 3],
    pub members: usize,
}
fn fit_sequence(
    ps: &[Patch],
    ids: &[usize],
    pictures: &[Picture],
    train: &[usize],
    mut seq: Sequence,
) -> Sequence {
    let points = ids.iter().map(|&i| &ps[i]).collect::<Vec<_>>();
    seq.members = ids.len();
    for cycle in 0..3 {
        let previous = seq.poses.clone();
        for (j, &t) in train.iter().enumerate().skip(1) {
            let mut best = pose(&points, &pictures[t], previous[t]);
            // Neighboring fitted exposures offer independent starts for the
            // same training image, preventing isolated local-minimum jumps.
            for neighbor in [
                j.checked_sub(1),
                if j + 1 < train.len() {
                    Some(j + 1)
                } else {
                    None
                },
            ]
            .into_iter()
            .flatten()
            {
                let seed = Model {
                    crop: previous[t].crop,
                    surface: seq.shape,
                    ..previous[train[neighbor]]
                };
                let candidate = pose(&points, &pictures[t], seed);
                if score(&points, &pictures[t], candidate) < score(&points, &pictures[t], best) {
                    best = candidate;
                }
            }
            seq.poses[t] = best;
        }
        // The same depth surface must explain every training exposure.
        for d in 0..3 {
            let mut best = train
                .iter()
                .skip(1)
                .map(|&t| score(&points, &pictures[t], seq.poses[t]))
                .sum::<f64>();
            let old = seq.shape;
            for sign in [-1., 1.] {
                let mut candidate = old;
                candidate[d] += sign * 0.3 * 0.5f64.powi(cycle);
                if candidate[d].abs() > if d == 2 { 1.2 } else { 0.7 } {
                    continue;
                }
                let loss = train
                    .iter()
                    .skip(1)
                    .map(|&t| {
                        score(
                            &points,
                            &pictures[t],
                            Model {
                                surface: candidate,
                                ..seq.poses[t]
                            },
                        )
                    })
                    .sum::<f64>();
                if loss < best {
                    best = loss;
                    seq.shape = candidate;
                }
            }
            for m in &mut seq.poses {
                m.surface = seq.shape;
            }
        }
    }
    seq
}
// Stronger null: two different depth surfaces but one common rigid pose.
// This separates a shape-model advantage from evidence for independent motion.
fn fit_shared(
    ps: &[Patch],
    labels: &[usize],
    pictures: &[Picture],
    train: &[usize],
    baseline: &Sequence,
    groups: &[Sequence],
    supported: &[bool],
) -> Vec<Sequence> {
    let mut out = groups.to_vec();
    for g in 0..2 {
        for &t in train {
            out[g].poses[t] = Model {
                surface: out[g].shape,
                ..baseline.poses[t]
            };
        }
    }
    for cycle in 0..4 {
        for &t in train.iter().skip(1) {
            let shapes = [out[0].shape, out[1].shape];
            let joint = |m: Model| {
                let mut values = ps
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| supported[*i])
                    .map(|(i, p)| {
                        error(
                            p,
                            &pictures[t],
                            Model {
                                surface: shapes[labels[i]],
                                ..m
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                let n = (values.len() * 9 / 10).max(1);
                values[..n].iter().sum::<f64>() / n as f64
            };
            let mut m = out[0].poses[t];
            let mut best = joint(m);
            // Give the common-motion null access to either independent pose
            // as a start too; do not handicap it with the original baseline.
            for group in groups {
                let candidate = Model {
                    surface: m.surface,
                    ..group.poses[t]
                };
                let value = joint(candidate);
                if value < best {
                    m = candidate;
                    best = value;
                }
            }
            for round in 0..5 {
                for d in 0..6 {
                    let old = m;
                    let step = match d {
                        0..=2 => 0.1,
                        3..=4 => 0.5,
                        _ => 0.03,
                    } * 0.5f64.powi(round);
                    for sign in [-1., 1.] {
                        let mut c = old;
                        c.set(d, c.get(d) + sign * step);
                        if c.valid() {
                            let e = joint(c);
                            if e < best {
                                best = e;
                                m = c;
                            }
                        }
                    }
                }
            }
            for g in 0..2 {
                out[g].poses[t] = Model {
                    surface: out[g].shape,
                    ..m
                };
            }
        }
        for g in 0..2 {
            let points = ps
                .iter()
                .enumerate()
                .filter(|(i, _)| labels[*i] == g && supported[*i])
                .map(|(_, p)| p)
                .collect::<Vec<_>>();
            for dim in 0..3 {
                let old = out[g].shape;
                let mut best = train
                    .iter()
                    .skip(1)
                    .map(|&t| score(&points, &pictures[t], out[g].poses[t]))
                    .sum::<f64>();
                for sign in [-1., 1.] {
                    let mut shape = old;
                    shape[dim] += sign * 0.25 * 0.5f64.powi(cycle);
                    if shape[dim].abs() > if dim == 2 { 1.2 } else { 0.7 } {
                        continue;
                    }
                    let e = train
                        .iter()
                        .skip(1)
                        .map(|&t| {
                            score(
                                &points,
                                &pictures[t],
                                Model {
                                    surface: shape,
                                    ..out[g].poses[t]
                                },
                            )
                        })
                        .sum::<f64>();
                    if e < best {
                        best = e;
                        out[g].shape = shape;
                    }
                }
                let shape = out[g].shape;
                for m in &mut out[g].poses {
                    m.surface = shape;
                }
            }
        }
    }
    out
}
fn local_offset(p: &Patch, b: &Picture, m: Model) -> P {
    let mut best = error(p, b, m);
    let mut delta = [0., 0.];
    for y in -4..=4 {
        for x in -4..=4 {
            let d = [x as f64 * 0.35, y as f64 * 0.35];
            let mut c = m;
            for a in 0..2 {
                c.translation[a] += d[a];
            }
            let e = error(p, b, c);
            if e < best {
                best = e;
                delta = d;
            }
        }
    }
    for step in [0.15, 0.05] {
        for a in 0..2 {
            let old = delta;
            for sign in [-1., 1.] {
                let mut d = old;
                d[a] += sign * step;
                let mut c = m;
                for k in 0..2 {
                    c.translation[k] += d[k];
                }
                let e = error(p, b, c);
                if e < best {
                    best = e;
                    delta = d;
                }
            }
        }
    }
    delta
}
fn kmedians(v: &[Vec<f64>]) -> Vec<usize> {
    let distance = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>();
    let mut best_labels = vec![0; v.len()];
    let mut best = f64::INFINITY;
    for start in 0..8 {
        let a = start * v.len() / 8;
        // Deterministic stratified distance-weighted seeds for robust L1 clustering. Always choosing the
        // farthest signature repeatedly seeds an occlusion outlier instead
        // of the second persistent region.
        let weights = v.iter().map(|x| distance(&v[a], x)).collect::<Vec<_>>();
        let threshold = weights.iter().sum::<f64>() * ((start * 5 % 8) as f64 + 0.5) / 8.;
        let mut cumulative = 0.;
        let b = weights
            .iter()
            .position(|&w| {
                cumulative += w;
                cumulative >= threshold
            })
            .unwrap_or(v.len() - 1);
        let mut centers = [v[a].clone(), v[b].clone()];
        let mut labels = vec![0; v.len()];
        for _ in 0..20 {
            for (i, x) in v.iter().enumerate() {
                labels[i] = usize::from(distance(x, &centers[1]) < distance(x, &centers[0]));
            }
            for g in 0..2 {
                let n = labels.iter().filter(|&&x| x == g).count();
                if n == 0 {
                    continue;
                }
                for d in 0..v[0].len() {
                    let mut values = v
                        .iter()
                        .zip(&labels)
                        .filter(|(_, l)| **l == g)
                        .map(|(x, _)| x[d])
                        .collect::<Vec<_>>();
                    values.sort_by(f64::total_cmp);
                    centers[g][d] = (values[(n - 1) / 2] + values[n / 2]) * 0.5;
                }
            }
        }
        if (0..2).any(|g| labels.iter().filter(|&&l| l == g).count() < 12) {
            continue;
        }
        let loss = v
            .iter()
            .zip(&labels)
            .map(|(x, &g)| distance(x, &centers[g]))
            .sum::<f64>();
        if loss < best {
            best = loss;
            best_labels = labels;
        }
    }
    best_labels
}
pub(super) fn interpolate(a: Model, b: Model, u: f64, crop: P) -> Model {
    // SO(3) interpolation of axis-angle quaternions, not Euler interpolation.
    let quaternion = |v: [f64; 3]| {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let f = if n < 1e-12 { 0.5 } else { (n / 2.).sin() / n };
        [(n / 2.).cos(), v[0] * f, v[1] * f, v[2] * f]
    };
    let qa = quaternion(a.omega);
    let mut qb = quaternion(b.omega);
    let mut dot = qa.iter().zip(qb).map(|(x, y)| x * y).sum::<f64>();
    if dot < 0. {
        qb = qb.map(|x| -x);
        dot = -dot;
    }
    let theta = dot.clamp(-1., 1.).acos();
    let (wa, wb) = if theta.abs() < 1e-6 {
        (1. - u, u)
    } else {
        (
            ((1. - u) * theta).sin() / theta.sin(),
            (u * theta).sin() / theta.sin(),
        )
    };
    let q: [f64; 4] = std::array::from_fn(|i| wa * qa[i] + wb * qb[i]);
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let q = q.map(|x| x / norm);
    let angle = 2. * q[0].clamp(-1., 1.).acos();
    let sn = (1. - q[0] * q[0]).max(0.).sqrt();
    Model {
        omega: if sn < 1e-9 {
            [0.; 3]
        } else {
            [q[1] * angle / sn, q[2] * angle / sn, q[3] * angle / sn]
        },
        translation: std::array::from_fn(|i| a.translation[i] * (1. - u) + b.translation[i] * u),
        crop,
        ..a
    }
}
#[derive(Serialize)]
pub struct FrameResult {
    pub index: usize,
    pub held: bool,
    pub single: Model,
    pub groups: Vec<Model>,
    pub single_error: f64,
    pub shared_error: f64,
    pub shared_groups: Vec<Model>,
    pub shared_losses: Vec<f64>,
    pub grouped_error: f64,
    pub swapped_error: f64,
    pub support: f64,
    pub separation: f64,
    pub single_losses: Vec<f64>,
    pub grouped_losses: Vec<f64>,
}
#[derive(Serialize)]
pub struct Discovery {
    pub points: Vec<P>,
    pub labels: Vec<usize>,
    pub initial_labels: Vec<usize>,
    pub membership_margin: Vec<f64>,
    pub membership_consistency: Vec<f64>,
    pub membership_fit_support: Vec<f64>,
    pub core: Vec<bool>,
    pub single: Sequence,
    pub groups: Vec<Sequence>,
    pub shared: Vec<Sequence>,
    pub frames: Vec<FrameResult>,
    pub train: Vec<usize>,
    pub iterations: Vec<Vec<usize>>,
}
pub fn discover(input: &[Picture], crops: &[P], times: &[f64]) -> Discovery {
    let pictures = input.iter().map(motion::normalized).collect::<Vec<_>>();
    let ps = pixels::patches(&pictures[0], 1);
    let all = (0..ps.len()).collect::<Vec<_>>();
    let train = (0..pictures.len()).step_by(2).collect::<Vec<_>>();
    let mut models = vec![Model::new(input[0][0].w, input[0][0].h, [0.; 2]); pictures.len()];
    let refs = ps.iter().collect::<Vec<_>>();
    for &t in train.iter().skip(1) {
        let mut seed = models[t - 2];
        seed.crop = crops[t];
        seed.surface = [0., 0., 0.3];
        models[t] = pose(&refs, &pictures[t], seed);
    }
    for m in &mut models {
        m.surface = [0., 0., 0.3];
    }
    let baseline = fit_sequence(
        &ps,
        &all,
        &pictures,
        &train,
        Sequence {
            poses: models,
            shape: [0., 0., 0.3],
            members: ps.len(),
        },
    );
    let signatures = ps
        .iter()
        .map(|p| {
            train
                .iter()
                .skip(1)
                .flat_map(|&t| local_offset(p, &pictures[t], baseline.poses[t]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut labels = kmedians(&signatures);
    let initial_labels = labels.clone();
    let mut groups = vec![baseline.clone(); 2];
    for g in 0..2 {
        let ids = (0..ps.len())
            .filter(|&i| labels[i] == g)
            .collect::<Vec<_>>();
        for (j, &t) in train.iter().skip(1).enumerate() {
            for d in 0..2 {
                groups[g].poses[t].translation[d] +=
                    ids.iter().map(|&i| signatures[i][2 * j + d]).sum::<f64>()
                        / ids.len().max(1) as f64;
            }
        }
    }
    let mut iterations = Vec::new();
    for _ in 0..4 {
        for g in 0..2 {
            let ids = (0..ps.len())
                .filter(|&i| labels[i] == g)
                .collect::<Vec<_>>();
            groups[g] = fit_sequence(&ps, &ids, &pictures, &train, groups[g].clone());
        }
        let old = labels.clone();
        for (i, p) in ps.iter().enumerate() {
            let mut costs = [0.; 2];
            let mut votes = 0;
            for &t in train.iter().skip(1) {
                if let Some(e) =
                    pixels::paired_error(p, &pictures[t], groups[0].poses[t], groups[1].poses[t])
                {
                    for g in 0..2 {
                        costs[g] += e[g];
                    }
                    votes += 1;
                }
            }
            if votes < (train.len() - 1) / 2 {
                continue;
            }
            for c in &mut costs {
                *c /= votes as f64;
            }
            let neighbors = ps
                .iter()
                .enumerate()
                .filter(|(j, q)| {
                    *j != i && (q.p[0] - p.p[0]).abs() + (q.p[1] - p.p[1]).abs() < 1.01
                })
                .map(|(j, _)| j)
                .collect::<Vec<_>>();
            let regularized: [f64; 2] = std::array::from_fn(|g| {
                costs[g]
                    + 0.02 * neighbors.iter().filter(|&&j| old[j] != g).count() as f64
                        / neighbors.len().max(1) as f64
            });
            labels[i] = usize::from(regularized[1] < regularized[0]);
        }
        if (0..2).any(|g| labels.iter().filter(|&&x| x == g).count() < 12) {
            labels = old;
            break;
        }
        iterations.push(labels.clone());
        if old == labels {
            break;
        }
    }
    // Refit on final memberships; validation frames still not read.
    for g in 0..2 {
        let ids = (0..ps.len())
            .filter(|&i| labels[i] == g)
            .collect::<Vec<_>>();
        groups[g] = fit_sequence(&ps, &ids, &pictures, &train, groups[g].clone());
    }
    let mut supported = vec![true; ps.len()];
    // Persistently occluded or unsupported patches must not pull a surface's
    // final pose. This mask uses only training residuals; all patches still
    // participate in held evaluation and membership auditing below.
    for g in 0..2 {
        let ids = (0..ps.len())
            .filter(|&i| {
                if labels[i] != g {
                    return false;
                }
                let errors = train
                    .iter()
                    .skip(1)
                    .map(|&t| error(&ps[i], &pictures[t], groups[g].poses[t]))
                    .collect::<Vec<_>>();
                errors.iter().filter(|&&e| e < 0.08).count() as f64 >= 0.8 * errors.len() as f64
                    && errors.iter().sum::<f64>() / (errors.len() as f64) < 0.08
            })
            .collect::<Vec<_>>();
        if ids.len() >= 12 {
            for i in 0..ps.len() {
                if labels[i] == g {
                    supported[i] = ids.contains(&i);
                }
            }
            groups[g] = fit_sequence(&ps, &ids, &pictures, &train, groups[g].clone());
        }
    }
    // Abstention is determined on training frames only. Evaluation below still
    // scores every patch so discarding uncertain membership cannot inflate gain.
    let mut membership_margin = Vec::new();
    let mut membership_consistency = Vec::new();
    let mut membership_fit_support = Vec::new();
    let mut core = Vec::new();
    for (i, p) in ps.iter().enumerate() {
        let mut margins = Vec::new();
        let mut own_errors = Vec::new();
        for &t in train.iter().skip(1) {
            let own = error(p, &pictures[t], groups[labels[i]].poses[t]);
            if let Some(e) = pixels::paired_error(
                p,
                &pictures[t],
                groups[labels[i]].poses[t],
                groups[1 - labels[i]].poses[t],
            ) {
                margins.push(e[1] - e[0]);
            }
            own_errors.push(own);
        }
        let mean = margins.iter().sum::<f64>() / margins.len().max(1) as f64;
        let decisive = margins
            .iter()
            .filter(|m| m.abs() > 0.003)
            .collect::<Vec<_>>();
        let agreement =
            decisive.iter().filter(|&&m| *m > 0.).count() as f64 / decisive.len().max(1) as f64;
        membership_margin.push(mean);
        membership_consistency.push(agreement);
        let support =
            own_errors.iter().filter(|&&e| e < 0.08).count() as f64 / own_errors.len() as f64;
        let own_mean = own_errors.iter().sum::<f64>() / own_errors.len() as f64;
        membership_fit_support.push(support);
        core.push(
            mean >= 0.015
                && agreement >= 0.8
                && support >= 0.8
                && own_mean < 0.08
                && margins.len() as f64 >= 0.8 * (train.len() - 1) as f64,
        );
    }
    let shared = fit_shared(
        &ps, &labels, &pictures, &train, &baseline, &groups, &supported,
    );
    let mut frames = Vec::new();
    for t in 1..pictures.len() {
        let held = t % 2 == 1;
        let pose_at = |s: &Sequence| {
            if !held {
                s.poses[t]
            } else {
                let a = t - 1;
                let b = (t + 1).min(pictures.len() - 1);
                interpolate(
                    s.poses[a],
                    s.poses[b],
                    (times[t] - times[a]) / (times[b] - times[a]),
                    crops[t],
                )
            }
        };
        let one = pose_at(&baseline);
        let gs = groups.iter().map(pose_at).collect::<Vec<_>>();
        let shared_groups = shared.iter().map(pose_at).collect::<Vec<_>>();
        let mut shared_losses = Vec::new();
        let mut e1 = Vec::new();
        let mut e2 = Vec::new();
        let mut swap = 0.;
        let mut sep = 0.;
        let mut support = 0;
        for (i, p) in ps.iter().enumerate() {
            e1.push(error(p, &pictures[t], one));
            shared_losses.push(error(p, &pictures[t], shared_groups[labels[i]]));
            let e = error(p, &pictures[t], gs[labels[i]]);
            e2.push(e);
            swap += error(p, &pictures[t], gs[1 - labels[i]]);
            if e < 0.08 {
                support += 1;
            }
            if let (Some(a), Some(b)) = (gs[0].prepare().map(p.p), gs[1].prepare().map(p.p)) {
                sep += (a[0] - b[0]).hypot(a[1] - b[1]);
            }
        }
        let n = ps.len().max(1) as f64;
        frames.push(FrameResult {
            index: t,
            held,
            single: one,
            groups: gs,
            single_error: e1.iter().sum::<f64>() / n,
            shared_error: shared_losses.iter().sum::<f64>() / n,
            shared_groups,
            shared_losses,
            grouped_error: e2.iter().sum::<f64>() / n,
            swapped_error: swap / n,
            support: support as f64 / n,
            separation: sep / n,
            single_losses: e1,
            grouped_losses: e2,
        });
    }
    Discovery {
        points: ps.iter().map(|p| p.p).collect(),
        labels,
        initial_labels,
        membership_margin,
        membership_consistency,
        membership_fit_support,
        core,
        single: baseline,
        shared,
        groups,
        frames,
        train,
        iterations,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn withheld_pixels_cannot_change_membership_or_fitted_motion() {
        let pictures = (0..9)
            .map(|t| {
                std::array::from_fn(|c| {
                    let (w, h) = (18, 12);
                    let v = (0..w * h)
                        .map(|k| {
                            let x = (k % w) as f64 - t as f64 * 0.04;
                            let y = (k / w) as f64;
                            0.4 + 0.12 * (x * 0.7 + c as f64).sin()
                                + 0.1 * (y * 0.9).cos()
                                + 0.08 * (x * 0.4 + y * 0.6).sin()
                        })
                        .collect();
                    super::super::super::Image { w, h, v }
                })
            })
            .collect::<Vec<Picture>>();
        let times = (0..9).map(|i| i as f64 / 30.).collect::<Vec<_>>();
        let crops = vec![[0.; 2]; 9];
        let a = discover(&pictures, &crops, &times);
        let mut corrupted = pictures.clone();
        for t in (1..9).step_by(2) {
            for c in 0..3 {
                for (k, v) in corrupted[t][c].v.iter_mut().enumerate() {
                    *v = ((k * 17 + t * 5 + c * 3) % 29) as f64 / 29.;
                }
            }
        }
        let b = discover(&corrupted, &crops, &times);
        assert_eq!(a.labels, b.labels);
        assert_eq!(a.core, b.core);
        assert_eq!(
            serde_json::to_string(&a.groups).unwrap(),
            serde_json::to_string(&b.groups).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&a.shared).unwrap(),
            serde_json::to_string(&b.shared).unwrap()
        );
        assert!(a
            .frames
            .iter()
            .zip(&b.frames)
            .filter(|(f, _)| f.held)
            .any(|(x, y)| (x.grouped_error - y.grouped_error).abs() > 0.01));
        for &t in &a.train {
            assert_eq!(a.shared[0].poses[t].omega, a.shared[1].poses[t].omega);
            assert_eq!(
                a.shared[0].poses[t].translation,
                a.shared[1].poses[t].translation
            );
        }
    }
    #[test]
    fn rotation_interpolation_preserves_endpoints() {
        let a = Model::new(18, 12, [0.; 2]);
        let mut b = a;
        b.omega = [0.1, -0.2, 0.3];
        let c = interpolate(a, b, 1., [0.; 2]);
        for i in 0..3 {
            assert!((c.omega[i] - b.omega[i]).abs() < 1e-10);
        }
    }
}
