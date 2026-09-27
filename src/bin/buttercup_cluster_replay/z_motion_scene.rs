//! Articulated hypothesis: one SE(3) parent plus rotation of an attached child.
//! The parent origin is the fixed reference gauge [0,0,1]. The child pivot is
//! fixed in that parent's coordinates; it is not a free per-frame translation.
use super::{
    engine,
    motion::{self, Model, Picture},
    P,
};
use serde::Serialize;
type V = [f64; 3];
type M = [[f64; 3]; 3];
fn mv(a: M, b: V) -> V {
    std::array::from_fn(|i| (0..3).map(|j| a[i][j] * b[j]).sum())
}
fn mm(a: M, b: M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}
fn tr(a: M) -> M {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
fn log(r: M) -> V {
    let angle = ((r[0][0] + r[1][1] + r[2][2] - 1.) * 0.5)
        .clamp(-1., 1.)
        .acos();
    let v = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    let scale = if angle < 1e-8 {
        0.5
    } else {
        angle / (2. * angle.sin())
    };
    v.map(|x| x * scale)
}
fn child(parent: Model, relative: V, pivot: V, shape: V, depth: f64) -> Model {
    let rp = motion::rotation(parent.omega);
    let rc = motion::rotation(relative);
    let rq = mv(rc, pivot);
    let shift = mv(rp, std::array::from_fn(|i| pivot[i] - rq[i]));
    Model {
        omega: log(mm(rp, rc)),
        translation: std::array::from_fn(|i| {
            parent.translation[i] + shift[i] * if i < 2 { parent.focal } else { 1. }
        }),
        surface: shape,
        depth_scale: depth,
        ..parent
    }
}
#[derive(Clone, Serialize)]
pub struct Scene {
    pub parent: Vec<Model>,
    pub relative_rotation: Vec<V>,
    /// Offset from the fixed root origin, expressed in the parent reference frame.
    pub child_pivot: V,
    pub child_depth_scale: f64,
    pub shapes: [V; 2],
    pub labels: Vec<usize>,
    pub parent_seed_group: usize,
}
impl Scene {
    pub fn models(&self, t: usize) -> [Model; 2] {
        let p = Model {
            surface: self.shapes[0],
            depth_scale: 1.,
            ..self.parent[t]
        };
        [
            p,
            child(
                p,
                self.relative_rotation[t],
                self.child_pivot,
                self.shapes[1],
                self.child_depth_scale,
            ),
        ]
    }
}
#[derive(Clone, Serialize)]
pub struct Render {
    pub values: Vec<[f64; 3]>,
    pub depth: Vec<f64>,
    pub owner: Vec<Option<usize>>,
    pub collisions: usize,
}
fn sample(im: &Picture, p: P) -> [f64; 3] {
    let (w, h) = (im[0].w, im[0].h);
    let px = p[0].clamp(0., (w - 1) as f64);
    let py = p[1].clamp(0., (h - 1) as f64);
    let x = (px as usize).min(w - 2);
    let y = (py as usize).min(h - 2);
    let u = px - x as f64;
    let v = py - y as f64;
    std::array::from_fn(|c| {
        (1. - v) * ((1. - u) * im[c].v[y * w + x] + u * im[c].v[y * w + x + 1])
            + v * ((1. - u) * im[c].v[(y + 1) * w + x] + u * im[c].v[(y + 1) * w + x + 1])
    })
}
/// Forward render reference cells with perspective-correct depth and texture.
/// Both bodies compete for each target pixel; nearest positive depth wins.
pub fn render(
    source: &Picture,
    points: &[P],
    labels: &[usize],
    models: [Model; 2],
    scale: f64,
) -> Render {
    let (w, h) = (source[0].w, source[0].h);
    let mut out = Render {
        values: vec![[0.; 3]; w * h],
        depth: vec![f64::INFINITY; w * h],
        owner: vec![None; w * h],
        collisions: 0,
    };
    let models = models.map(|m| m.scaled(scale).prepare());
    for (i, p) in points.iter().enumerate() {
        let g = labels[i];
        let mut xy = [[0.; 2]; 4];
        let mut projected = [[0.; 3]; 4];
        let mut valid = true;
        for (k, d) in [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
            .iter()
            .enumerate()
        {
            xy[k] = std::array::from_fn(|a| {
                ((p[a] + d[a]) * scale).clamp(
                    -0.5,
                    if a == 0 {
                        w as f64 - 0.5
                    } else {
                        h as f64 - 0.5
                    },
                )
            });
            if let Some(q) = models[g].map_depth(xy[k]) {
                projected[k] = q;
            } else {
                valid = false;
            }
        }
        if !valid {
            continue;
        }
        for ids in [[0, 1, 2], [0, 2, 3]] {
            let q = ids.map(|k| projected[k]);
            let s = ids.map(|k| xy[k]);
            let area = (q[1][1] - q[2][1]) * (q[0][0] - q[2][0])
                + (q[2][0] - q[1][0]) * (q[0][1] - q[2][1]);
            if area.abs() < 1e-9 {
                continue;
            }
            let x0 = q
                .iter()
                .map(|q| q[0])
                .fold(f64::INFINITY, f64::min)
                .ceil()
                .max(0.) as usize;
            let x1 = q
                .iter()
                .map(|q| q[0])
                .fold(f64::NEG_INFINITY, f64::max)
                .floor()
                .min((w - 1) as f64)
                .max(-1.) as isize;
            let y0 = q
                .iter()
                .map(|q| q[1])
                .fold(f64::INFINITY, f64::min)
                .ceil()
                .max(0.) as usize;
            let y1 = q
                .iter()
                .map(|q| q[1])
                .fold(f64::NEG_INFINITY, f64::max)
                .floor()
                .min((h - 1) as f64)
                .max(-1.) as isize;
            for y in y0..=(y1.max(-1) as usize).min(h - 1) {
                if y1 < 0 {
                    break;
                }
                for x in x0..=(x1.max(-1) as usize).min(w - 1) {
                    if x1 < 0 {
                        break;
                    }
                    let a = ((q[1][1] - q[2][1]) * (x as f64 - q[2][0])
                        + (q[2][0] - q[1][0]) * (y as f64 - q[2][1]))
                        / area;
                    let b = ((q[2][1] - q[0][1]) * (x as f64 - q[2][0])
                        + (q[0][0] - q[2][0]) * (y as f64 - q[2][1]))
                        / area;
                    let bc = [a, b, 1. - a - b];
                    if bc.iter().any(|&v| v < -1e-7) {
                        continue;
                    }
                    let invz = (0..3).map(|j| bc[j] / q[j][2]).sum::<f64>();
                    if invz <= 0. {
                        continue;
                    }
                    let z = 1. / invz;
                    let k = y * w + x;
                    if out.owner[k].is_some_and(|old| old != g) {
                        out.collisions += 1;
                    }
                    if z + 1e-9 >= out.depth[k] {
                        continue;
                    }
                    let uv = std::array::from_fn(|a| {
                        (0..3).map(|j| bc[j] * s[j][a] / q[j][2]).sum::<f64>() * z
                    });
                    out.depth[k] = z;
                    out.owner[k] = Some(g);
                    out.values[k] = sample(source, uv);
                }
            }
        }
    }
    out
}
fn loss(
    source: &Picture,
    target: &Picture,
    points: &[P],
    labels: &[usize],
    models: [Model; 2],
    scale: f64,
) -> (f64, f64, usize) {
    let r = render(source, points, labels, models, scale);
    let mut sum = 0.;
    let mut n = 0;
    for k in 0..r.values.len() {
        if r.owner[k].is_none() {
            continue;
        }
        n += 1;
        for c in 0..3 {
            let d = (r.values[k][c] - target[c].v[k]).abs();
            sum += if d < 0.5 { d * d } else { d - 0.25 };
        }
    }
    let coverage = n as f64 / r.values.len() as f64;
    if coverage < 0.5 {
        return (1., coverage, r.collisions);
    }
    (
        sum / (3 * n) as f64 + 0.2 * (1. - coverage),
        coverage,
        r.collisions,
    )
}
fn fit_frame(
    scene: &mut Scene,
    t: usize,
    source: &Picture,
    target: &Picture,
    points: &[P],
    attached: bool,
) {
    let mut best = loss(source, target, points, &scene.labels, scene.models(t), 1.).0;
    for round in 0..4 {
        for d in 0..if attached { 9 } else { 6 } {
            let old = scene.clone();
            let step = match d {
                0..=2 | 6..=8 => 0.06,
                3..=4 => 0.3,
                _ => 0.025,
            } * 0.5f64.powi(round);
            for sign in [-1., 1.] {
                let mut candidate = old.clone();
                if d < 6 {
                    let value = candidate.parent[t].get(d);
                    candidate.parent[t].set(d, value + sign * step);
                } else {
                    candidate.relative_rotation[t][d - 6] += sign * step;
                }
                if !candidate.parent[t].valid()
                    || candidate.relative_rotation[t].iter().any(|v| v.abs() > 0.6)
                {
                    continue;
                }
                let v = loss(
                    source,
                    target,
                    points,
                    &candidate.labels,
                    candidate.models(t),
                    1.,
                )
                .0;
                if v < best {
                    best = v;
                    *scene = candidate;
                }
            }
        }
    }
}
fn pivot_seed(parent: &[Model], child_models: &[Model], relative: &[V], train: &[usize]) -> V {
    let mut a = [[0.; 3]; 3];
    let mut b = [0.; 3];
    for &t in train.iter().skip(1) {
        let rp = motion::rotation(parent[t].omega);
        let rc = motion::rotation(relative[t]);
        let mat = mm(
            rp,
            std::array::from_fn(|i| std::array::from_fn(|j| f64::from(i == j) - rc[i][j])),
        );
        let delta: V = std::array::from_fn(|i| {
            (child_models[t].translation[i] - parent[t].translation[i])
                / if i < 2 { parent[t].focal } else { 1. }
        });
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += (0..3).map(|k| mat[k][i] * mat[k][j]).sum::<f64>();
            }
            b[i] += (0..3).map(|k| mat[k][i] * delta[k]).sum::<f64>();
        }
    }
    for i in 0..3 {
        a[i][i] += 1e-4;
    }
    b[2] += 1e-4 * 0.5;
    for i in 0..3 {
        let pivot = (i..3)
            .max_by(|&u, &v| a[u][i].abs().total_cmp(&a[v][i].abs()))
            .unwrap();
        a.swap(i, pivot);
        b.swap(i, pivot);
        let f = a[i][i];
        if f.abs() < 1e-12 {
            return [0., 0., 0.5];
        }
        for j in i..3 {
            a[i][j] /= f;
        }
        b[i] /= f;
        for k in 0..3 {
            if k == i {
                continue;
            }
            let f = a[k][i];
            for j in i..3 {
                a[k][j] -= f * a[i][j];
            }
            b[k] -= f * b[i];
        }
    }
    [
        b[0].clamp(-0.5, 0.5),
        b[1].clamp(-0.5, 0.5),
        b[2].clamp(-0.5, 2.),
    ]
}
#[derive(Serialize)]
pub struct Comparison {
    pub articulated: Scene,
    pub rigid: Scene,
    pub frames: Vec<super::Value>,
    pub alternative_orders: Vec<super::Value>,
}
pub fn fit(
    input: &[Picture],
    fine: &[Picture],
    times: &[f64],
    d: &engine::Discovery,
) -> Comparison {
    let images = input.iter().map(motion::normalized).collect::<Vec<_>>();
    let finer = fine.iter().map(motion::normalized).collect::<Vec<_>>();
    let mut candidates = Vec::new();
    for parent_id in 0..2 {
        for depth in [0.85, 1.15] {
            let parent = d.groups[parent_id].poses.clone();
            let child_models = &d.groups[1 - parent_id].poses;
            let relative = parent
                .iter()
                .zip(child_models)
                .map(|(a, b)| log(mm(tr(motion::rotation(a.omega)), motion::rotation(b.omega))))
                .collect::<Vec<_>>();
            let pivot = pivot_seed(&parent, child_models, &relative, &d.train);
            let mut scene = Scene {
                parent,
                relative_rotation: relative,
                child_pivot: pivot,
                child_depth_scale: depth,
                shapes: [d.groups[parent_id].shape, d.groups[1 - parent_id].shape],
                labels: d
                    .labels
                    .iter()
                    .map(|&g| usize::from(g != parent_id))
                    .collect(),
                parent_seed_group: parent_id,
            };
            for cycle in 0..2 {
                for &t in d.train.iter().skip(1) {
                    fit_frame(&mut scene, t, &images[0], &images[t], &d.points, true);
                }
                let objective = |s: &Scene| {
                    d.train
                        .iter()
                        .skip(1)
                        .map(|&t| {
                            loss(
                                &images[0],
                                &images[t],
                                &d.points,
                                &s.labels,
                                s.models(t),
                                1.,
                            )
                            .0
                        })
                        .sum::<f64>()
                };
                for dim in 0..4 {
                    let old = scene.clone();
                    let mut best = objective(&scene);
                    for sign in [-1., 1.] {
                        let mut c = old.clone();
                        if dim < 3 {
                            c.child_pivot[dim] +=
                                sign * if dim == 2 { 0.2 } else { 0.05 } * 0.5f64.powi(cycle);
                        } else {
                            c.child_depth_scale += sign * 0.05;
                        }
                        if c.child_depth_scale < 0.65
                            || c.child_depth_scale > 1.35
                            || c.child_pivot[0].abs() > 0.6
                            || c.child_pivot[1].abs() > 0.6
                            || c.child_pivot[2] < -0.5
                            || c.child_pivot[2] > 2.
                        {
                            continue;
                        }
                        let v = objective(&c);
                        if v < best {
                            best = v;
                            scene = c;
                        }
                    }
                }
                // Reattribute reference cells from later training evidence. This boundary
                // change affects the complete earlier reconstruction, including occlusion.
                let ps = super::pixels::patches(&images[0], 1);
                let old = scene.labels.clone();
                for (i, p) in ps.iter().enumerate() {
                    let mut votes = [0.; 2];
                    let mut n = 0.;
                    for &t in d.train.iter().skip(1) {
                        let ms = scene.models(t);
                        if let Some(e) = super::pixels::paired_error(p, &images[t], ms[0], ms[1]) {
                            for g in 0..2 {
                                votes[g] += e[g];
                            }
                            n += 1.;
                        }
                    }
                    if n < 3. {
                        continue;
                    }
                    let neighbors = d
                        .points
                        .iter()
                        .enumerate()
                        .filter(|(j, q)| {
                            *j != i && (q[0] - p.p[0]).abs() + (q[1] - p.p[1]).abs() < 1.01
                        })
                        .map(|(j, _)| j)
                        .collect::<Vec<_>>();
                    for g in 0..2 {
                        votes[g] = votes[g] / n
                            + 0.02 * neighbors.iter().filter(|&&j| old[j] != g).count() as f64
                                / neighbors.len().max(1) as f64;
                    }
                    scene.labels[i] = usize::from(votes[1] < votes[0]);
                }
                if (0..2).any(|g| scene.labels.iter().filter(|&&x| x == g).count() < 12) {
                    scene.labels = old;
                }
            }
            // Boundary updates change both the visible texture and depth ordering.
            // Refit against that final partition before ranking hypotheses; otherwise
            // its poses describe the previous mask while the rigid baseline below
            // receives three fitting passes against the final one.
            for _ in 0..3 {
                for &t in d.train.iter().skip(1) {
                    fit_frame(&mut scene, t, &images[0], &images[t], &d.points, true);
                }
            }
            let training = d
                .train
                .iter()
                .skip(1)
                .map(|&t| {
                    loss(
                        &images[0],
                        &images[t],
                        &d.points,
                        &scene.labels,
                        scene.models(t),
                        1.,
                    )
                    .0
                })
                .sum::<f64>();
            candidates.push((training, scene));
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    let alternatives=candidates.iter().map(|(s,c)|super::json!({"training_loss":s,"parent_seed_group":c.parent_seed_group,"child_depth_scale":c.child_depth_scale,"pivot":c.child_pivot})).collect();
    let mut scene = candidates.remove(0).1;
    let mut rigid = scene.clone();
    rigid.relative_rotation.fill([0.; 3]);
    for &t in &d.train {
        rigid.parent[t] = d.shared[0].poses[t];
    }
    for _ in 0..3 {
        for &t in d.train.iter().skip(1) {
            fit_frame(&mut rigid, t, &images[0], &images[t], &d.points, false);
        }
    }
    // Interpolate the parent and child-relative rotations separately, then compose.
    // This preserves the same attachment pivot on withheld frames too.
    for t in (1..input.len()).step_by(2) {
        let a = t - 1;
        let b = t + 1;
        let u = (times[t] - times[a]) / (times[b] - times[a]);
        for s in [&mut scene, &mut rigid] {
            s.parent[t] =
                engine::interpolate(s.parent[a], s.parent[b], u, d.frames[t - 1].single.crop);
            let aa = Model {
                omega: s.relative_rotation[a],
                ..s.parent[a]
            };
            let bb = Model {
                omega: s.relative_rotation[b],
                ..s.parent[b]
            };
            s.relative_rotation[t] = engine::interpolate(aa, bb, u, [0.; 2]).omega;
        }
    }
    let frames=(1..input.len()).map(|t|{let ms=scene.models(t);let rs=rigid.models(t);let l=loss(&images[0],&images[t],&d.points,&scene.labels,ms,1.);let r=loss(&images[0],&images[t],&d.points,&rigid.labels,rs,1.);let f=loss(&finer[0],&finer[t],&d.points,&scene.labels,ms,3.);let rf=loss(&finer[0],&finer[t],&d.points,&rigid.labels,rs,3.);
  let swapped_labels=scene.labels.iter().map(|&g|1-g).collect::<Vec<_>>();
  let swapped=loss(&images[0],&images[t],&d.points,&swapped_labels,ms,1.).0;
  let z=render(&images[0],&d.points,&scene.labels,ms,1.);
  super::json!({"index":t,"held":t%2==1,"models":ms,"rigid_models":rs,"loss":l.0,"rigid_loss":r.0,"coverage":l.1,"collisions":l.2,"fine_loss":f.0,"fine_rigid_loss":rf.0,"swapped_loss":swapped,"owners":z.owner,"depth":z.depth})}).collect();
    Comparison {
        articulated: scene,
        rigid,
        frames,
        alternative_orders: alternatives,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_child_inherits_all_parent_rotation_and_translation() {
        let mut p = Model::new(18, 12, [0., 0.]);
        p.omega = [0.2, -0.1, 0.15];
        p.translation = [0.4, -0.2, 0.03];
        let c = child(p, [0.; 3], [0.1, -0.2, 0.8], p.surface, 1.);
        for q in [[3., 4.], [12., 7.]] {
            let a = p.prepare().map_depth(q).unwrap();
            let b = c.prepare().map_depth(q).unwrap();
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn child_pivot_remains_attached_during_relative_rotation() {
        let mut p = Model::new(18, 12, [0., 0.]);
        p.omega = [0.13, -0.2, 0.1];
        p.translation = [0.4, -0.3, 0.02];
        let pivot = [0.1, -0.08, 0.7];
        let c = child(p, [-0.2, 0.1, 0.2], pivot, [0.; 3], 1.);
        let rp = motion::rotation(p.omega);
        let rc = motion::rotation(c.omega);
        let a = mv(rp, pivot);
        let b = mv(rc, pivot);
        for i in 0..3 {
            let scale = if i < 2 { p.focal } else { 1. };
            assert!(
                (a[i] + p.translation[i] / scale - b[i] - c.translation[i] / scale).abs() < 1e-10
            );
        }
    }
    #[test]
    fn z_buffer_prefers_nearer_surface_independent_of_submission_order() {
        let im: Picture = std::array::from_fn(|_| super::super::super::Image {
            w: 18,
            h: 12,
            v: (0..216).map(|k| (k % 18) as f64 / 18.).collect(),
        });
        let a = Model::new(18, 12, [0.; 2]);
        let mut b = a;
        b.depth_scale = 0.8;
        b.crop = [-4., 0.];
        let r = render(&im, &[[5., 5.], [9., 5.]], &[0, 1], [a, b], 1.);
        assert_eq!(r.owner[5 * 18 + 5], Some(1));
        assert!(r.collisions > 0);
        let rr = render(&im, &[[9., 5.], [5., 5.]], &[1, 0], [a, b], 1.);
        assert_eq!(rr.owner[5 * 18 + 5], Some(1));
    }
}

/// Independent ray-intersection control: a textured rigid aperture occludes a
/// textured sphere attached behind it. Both inherit full XYZ rotation and XYZ
/// translation from frame one. Only the positive control later adds child
/// rotation; there is never a translation-only fitting or generation phase.
pub fn synthetic_attached(out: &std::path::Path, kind: &str) -> super::Result<()> {
    let (w, h) = (159usize, 105usize);
    let focal = 318.;
    let center = [79., 52.];
    let mut ultra = Vec::new();
    let mut coarse = Vec::new();
    let mut frames = Vec::new();
    let mut truth = Vec::new();
    let euler = |v: V| {
        let (sx, cx) = v[0].sin_cos();
        let (sy, cy) = v[1].sin_cos();
        let (sz, cz) = v[2].sin_cos();
        mm(
            [[cz, -sz, 0.], [sz, cz, 0.], [0., 0., 1.]],
            mm(
                [[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]],
                [[1., 0., 0.], [0., cx, -sx], [0., sx, cx]],
            ),
        )
    };
    for t in 0..51 {
        let u = t as f64 / 50.;
        let v = if kind.ends_with("null") {
            0.
        } else {
            ((t as f64 - 25.) / 25.).clamp(0., 1.)
        };
        let rotation = euler([0.028 * u, -0.045 * u, 0.018 * u]);
        let translation = [0.007 * u, -0.004 * u, 0.002 * u];
        let relative = euler([0.035 * v, -0.11 * v, 0.018 * v]);
        let inverse = tr(rotation);
        let mut origin = mv(
            inverse,
            [-translation[0], -translation[1], -1. - translation[2]],
        );
        origin[2] += 1.;
        let pivot = [0., 0., 1.26];
        let radius = 0.24;
        let mut channels: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.; w * h]);
        let mut owners = vec![None; w * h];
        for y in 0..h {
            for x in 0..w {
                let direction = mv(
                    inverse,
                    [
                        (x as f64 - center[0]) / focal,
                        (y as f64 - center[1]) / focal,
                        1.,
                    ],
                );
                let plane = (1. - origin[2]) / direction[2];
                let hit: V = std::array::from_fn(|i| origin[i] + plane * direction[i]);
                let aperture = (hit[0] / 0.18).powi(2) + (hit[1] / 0.105).powi(2) < 1.;
                let offset: V = std::array::from_fn(|i| origin[i] - pivot[i]);
                let a = direction.iter().map(|v| v * v).sum::<f64>();
                let b = 2.
                    * offset
                        .iter()
                        .zip(direction)
                        .map(|(a, b)| a * b)
                        .sum::<f64>();
                let c = offset.iter().map(|v| v * v).sum::<f64>() - radius * radius;
                let disc = b * b - 4. * a * c;
                let sphere = if disc >= 0. {
                    (-b - disc.sqrt()) / (2. * a)
                } else {
                    f64::INFINITY
                };
                let body = if sphere.is_finite() && sphere > 0. && (aperture || sphere < plane) {
                    1
                } else {
                    0
                };
                owners[y * w + x] = Some(body);
                let material = if body == 0 {
                    hit
                } else {
                    let q = std::array::from_fn(|i| origin[i] + sphere * direction[i] - pivot[i]);
                    mv(tr(relative), q)
                };
                let pixel = y * w + x;
                for c in 0..3 {
                    let (x, y) = (material[0], material[1]);
                    let phase = c as f64 * 0.7;
                    let value = if body == 0 {
                        0.38 + 0.08 * (39. * x + 17. * y + phase).sin()
                            + 0.06 * (83. * y - phase).cos()
                            + 0.05 * (117. * x - 47. * y).sin()
                    } else {
                        let r = x.hypot(y);
                        let base = if r < 0.033 {
                            0.055
                        } else if r < 0.11 {
                            0.18 + 0.045 * (17. * y.atan2(x) + phase).sin()
                        } else {
                            0.65
                        };
                        base + 0.04 * (95. * x + 19. * y + phase).sin()
                            + 0.03 * (67. * y - 23. * x).cos()
                    };
                    channels[c][pixel] = value.clamp(0., 1.);
                }
            }
        }
        let image: Picture = std::array::from_fn(|c| super::super::Image {
            w,
            h,
            v: channels[c].clone(),
        });
        let low = super::reduced(&image, 9);
        let fine = super::reduced(&image, 3);
        frames.push(super::frame(
            out,
            t,
            &low,
            &fine,
            t as f64 / 30.,
            "independent-attached-ray-scene",
        )?);
        super::png(out, &format!("raw-{t:03}.png"), &image)?;
        ultra.push(low);
        coarse.push(fine);
        truth.push(super::json!({"index":t,"rotation":rotation,"translation":translation,"child_relative_rotation":relative,"pivot":pivot,"owners":owners}));
    }
    std::fs::write(
        out.join("truth.json"),
        serde_json::to_vec(
            &super::json!({"kind":kind,"first_relative_motion_frame":if kind.ends_with("null"){None}else{Some(26)},"frames":truth}),
        )?,
    )?;
    super::run_history(
        out,
        &ultra,
        &coarse,
        frames,
        &vec![[0.; 2]; 51],
        super::json!({"synthetic":true,"kind":kind,"truth":"rigid plane with aperture occluding attached textured sphere; parent rotates and translates in all three dimensions from frame one; child starts relative rotation after frame 25 in positive control"}),
    )
}
