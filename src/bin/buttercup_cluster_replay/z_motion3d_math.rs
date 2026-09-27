//! Bounded perspective rigid motion over an uncertain curved depth surface.
//! Depth and focal length are assumptions, not calibrated anatomical measurements.
use super::{Image, P};
use serde::{Deserialize, Serialize};
pub type Picture = [Image; 3];

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Model {
    #[serde(default = "unit_depth")]
    pub depth_scale: f64,
    // Axis-angle radians; translation x/y in image pixels at unit reference depth.
    pub omega: [f64; 3],
    pub translation: [f64; 3],
    // z = 1 + sx*u + sy*v + curvature*(u*u+v*v), u,v in image-width units.
    pub surface: [f64; 3],
    pub focal: f64,
    pub center: P,
    pub width: f64,
    pub crop: P,
}
fn unit_depth() -> f64 {
    1.
}
impl Model {
    pub fn new(w: usize, h: usize, crop: P) -> Self {
        Self {
            depth_scale: 1.,
            omega: [0.; 3],
            translation: [0.; 3],
            surface: [0.; 3],
            focal: 2. * w as f64,
            center: [(w - 1) as f64 / 2., (h - 1) as f64 / 2.],
            width: w as f64,
            crop,
        }
    }
    pub fn scaled(self, s: f64) -> Self {
        Self {
            translation: [
                self.translation[0] * s,
                self.translation[1] * s,
                self.translation[2],
            ],
            focal: self.focal * s,
            center: self.center.map(|x| x * s),
            width: self.width * s,
            crop: self.crop.map(|x| x * s),
            ..self
        }
    }
    pub fn prepare(self) -> Prepared {
        Prepared {
            model: self,
            r: rotation(self.omega),
        }
    }
    pub(super) fn get(self, i: usize) -> f64 {
        match i {
            0..=2 => self.omega[i],
            3..=5 => self.translation[i - 3],
            _ => self.surface[i - 6],
        }
    }
    pub(super) fn set(&mut self, i: usize, x: f64) {
        match i {
            0..=2 => self.omega[i] = x,
            3..=5 => self.translation[i - 3] = x,
            _ => self.surface[i - 6] = x,
        }
    }
    pub(super) fn valid(self) -> bool {
        self.omega.iter().all(|v| v.abs() <= 0.52)
            && self.translation[..2]
                .iter()
                .all(|v| v.abs() <= self.width * 0.4)
            && self.translation[2].abs() <= 0.2
            && self.surface[..2].iter().all(|v| v.abs() <= 0.7)
            && self.surface[2].abs() <= 1.2
    }
}
#[derive(Clone, Copy)]
pub struct Prepared {
    model: Model,
    r: [[f64; 3]; 3],
}
pub(super) fn rotation(w: [f64; 3]) -> [[f64; 3]; 3] {
    let a = w.iter().map(|v| v * v).sum::<f64>().sqrt();
    if a < 1e-12 {
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    }
    let u = w.map(|v| v / a);
    let (s, c) = a.sin_cos();
    let k = [[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0.]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| c * f64::from(i == j) + (1. - c) * u[i] * u[j] + s * k[i][j])
    })
}
impl Prepared {
    pub fn map(self, p: P) -> Option<P> {
        self.map_depth(p).map(|q| [q[0], q[1]])
    }
    pub fn map_depth(self, p: P) -> Option<[f64; 3]> {
        let m = self.model;
        let u = (p[0] - m.center[0]) / m.width;
        let v = (p[1] - m.center[1]) / m.width;
        let z = m.depth_scale
            * (1. + m.surface[0] * u + m.surface[1] * v + m.surface[2] * (u * u + v * v));
        if !(0.35..=2.).contains(&z) {
            return None;
        }
        let x = [
            (p[0] - m.center[0]) * z / m.focal,
            (p[1] - m.center[1]) * z / m.focal,
            z - 1.,
        ];
        // Fixed origin is a gauge choice. A free pivot cannot be distinguished from translation.
        let q: [f64; 3] = std::array::from_fn(|i| {
            self.r[i].iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
                + if i == 2 {
                    1. + m.translation[2]
                } else {
                    m.translation[i] / m.focal
                }
        });
        if q[2] < 0.2 {
            return None;
        }
        let out = [
            m.focal * q[0] / q[2] + m.center[0] + m.crop[0],
            m.focal * q[1] / q[2] + m.center[1] + m.crop[1],
            q[2],
        ];
        out.iter().all(|v| v.is_finite()).then_some(out)
    }
}
#[derive(Clone)]
pub struct Patch {
    pub p: P,
    values: [[f64; 9]; 3],
    var: [f64; 3],
    mean: [f64; 3],
    pub held: bool,
}
pub fn patches(a: &Picture, stride: usize) -> Vec<Patch> {
    let mut out = Vec::new();
    for y in (2..a[0].h - 2).step_by(stride) {
        for x in (2..a[0].w - 2).step_by(stride) {
            let mean = std::array::from_fn(|c| {
                (0..9)
                    .map(|k| a[c].v[(y + k / 3 - 1) * a[c].w + x + k % 3 - 1])
                    .sum::<f64>()
                    / 9.
            });
            let values = std::array::from_fn(|c| {
                let mut vs = [0.; 9];
                for (k, v) in vs.iter_mut().enumerate() {
                    *v = a[c].v[(y + k / 3 - 1) * a[c].w + x + k % 3 - 1];
                }
                let mean = vs.iter().sum::<f64>() / 9.;
                vs.map(|v| v - mean)
            });
            let var: [f64; 3] = std::array::from_fn(|c| values[c].iter().map(|v| v * v).sum());
            if var.iter().filter(|&&v| v > 1e-6).count() >= 2 {
                out.push(Patch {
                    p: [x as f64, y as f64],
                    values,
                    var,
                    mean,
                    held: (x / (2 * stride) + y / (2 * stride)) % 2 == 1,
                });
            }
        }
    }
    out
}
pub fn cost(p: &Patch, b: &Picture, m: Prepared) -> f64 {
    let coords: Option<Vec<P>> = (0..9)
        .map(|k| m.map([p.p[0] + (k % 3) as f64 - 1., p.p[1] + (k / 3) as f64 - 1.]))
        .collect();
    let Some(coords) = coords else {
        return 1.;
    };
    let mut sum = 0.;
    let mut n = 0;
    for c in 0..3 {
        if p.var[c] <= 1e-6 {
            continue;
        }
        let mut y = [0.; 9];
        for (k, q) in coords.iter().enumerate() {
            let Some(v) = b[c].sample(*q) else {
                return 1.;
            };
            y[k] = v;
        }
        let mean = y.iter().sum::<f64>() / 9.;
        let y = y.map(|v| v - mean);
        let vy = y.iter().map(|v| v * v).sum::<f64>();
        if vy <= 1e-6 {
            continue;
        }
        let xy = y.iter().zip(p.values[c]).map(|(a, b)| a * b).sum::<f64>();
        sum += (1. - (xy / (vy * p.var[c]).sqrt()).clamp(-1., 1.)).min(1.);
        n += 1;
    }
    if n >= 2 {
        sum / n as f64
    } else {
        1.
    }
}
// Whole-image channel normalization preserves spatial brightness relationships;
// unlike per-patch NCC, it cannot explain every local ramp with a different gain.
pub(super) fn normalized(a: &Picture) -> Picture {
    std::array::from_fn(|c| {
        let im = &a[c];
        let mean = im.v.iter().sum::<f64>() / im.v.len() as f64;
        let scale = (im.v.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / im.v.len() as f64)
            .sqrt()
            .max(0.01);
        Image {
            w: im.w,
            h: im.h,
            v: im.v.iter().map(|v| (v - mean) / scale).collect(),
        }
    })
}
pub(super) fn relative_cost(p: &Patch, b: &Picture, m: Prepared) -> Option<f64> {
    let mut coords = [[0.; 2]; 9];
    for (k, q) in coords.iter_mut().enumerate() {
        *q = m.map([p.p[0] + (k % 3) as f64 - 1., p.p[1] + (k / 3) as f64 - 1.])?;
    }
    let mut sum = 0.;
    for c in 0..3 {
        for (k, q) in coords.iter().enumerate() {
            let y = b[c].sample(*q)?;
            let x = p.values[c][k] + p.mean[c];
            // Huber-like bounded error on spatially relative color, not absolute RAW brightness.
            let d = (x - y).abs();
            sum += if d < 0.5 { d * d } else { d - 0.25 };
        }
    }
    Some(sum / 27.)
}
fn loss(ps: &[&Patch], b: &Picture, m: Model) -> f64 {
    if ps.len() < 6 {
        return 10.;
    }
    let prepared = m.prepare();
    let mut values = ps
        .iter()
        .filter_map(|p| relative_cost(p, b, prepared).map(|v| 0.6 * v + 0.4 * cost(p, b, prepared)))
        .collect::<Vec<_>>();
    let coverage = values.len() as f64 / ps.len() as f64;
    if coverage < 0.5 {
        return 10. + 1. - coverage;
    }
    values.sort_by(f64::total_cmp);
    let n = (values.len() * 9 / 10).max(1);
    // Charge for cropped-away observations instead of accepting tiny perfect fragments.
    values[..n].iter().sum::<f64>() / n as f64 + 0.35 * (1. - coverage).powi(2)
}
#[derive(Clone, Serialize)]
pub struct Hypothesis {
    pub model: Model,
    pub fit_loss: f64,
}
fn optimize(ps: &[&Patch], b: &Picture, mut m: Model, three: bool, rounds: usize) -> Hypothesis {
    let dims: &[usize] = if three {
        &[0, 1, 2, 3, 4, 5, 6, 7, 8]
    } else {
        &[2, 3, 4]
    };
    let mut best = loss(ps, b, m);
    for round in 0..rounds {
        let s = 0.5f64.powi(round as i32);
        for _ in 0..2 {
            for &i in dims {
                let step = match i {
                    0..=2 => 0.18,
                    3..=4 => m.width * 0.075,
                    5 => 0.07,
                    _ => 0.35,
                } * s;
                let original = m;
                for sign in [-1., 1.] {
                    let mut candidate = original;
                    candidate.set(i, original.get(i) + sign * step);
                    if !candidate.valid() {
                        continue;
                    }
                    let value = loss(ps, b, candidate);
                    if value + 1e-9 < best {
                        best = value;
                        m = candidate;
                    }
                }
            }
        }
    }
    Hypothesis {
        model: m,
        fit_loss: best,
    }
}
pub fn search(ps: &[&Patch], b: &Picture, seed: Model, three: bool) -> Vec<Hypothesis> {
    if !three {
        let radius = (seed.width * 0.4).floor() as i32;
        let mut seeds = Vec::new();
        for y in -radius..=radius {
            for x in -radius..=radius {
                let mut m = seed;
                m.translation[0] = x as f64;
                m.translation[1] = y as f64;
                seeds.push(Hypothesis {
                    model: m,
                    fit_loss: loss(ps, b, m),
                });
            }
        }
        seeds.sort_by(|a, b| a.fit_loss.total_cmp(&b.fit_loss));
        seeds.truncate(8);
        let mut refined = seeds
            .iter()
            .map(|h| optimize(ps, b, h.model, false, 6))
            .collect::<Vec<_>>();
        refined.sort_by(|a, b| a.fit_loss.total_cmp(&b.fit_loss));
        refined.truncate(1);
        return refined;
    }
    let mut found = Vec::new();
    // Both tilt signs and multiple surface/focal hypotheses; no anatomical region masks.
    for focal in [1., 2., 4.] {
        for shape in [-0.65, 0., 0.65] {
            for tilt in [-0.2, 0.2] {
                let mut m = seed;
                m.focal = m.width * focal;
                m.surface = [shape * 0.4, -shape * 0.2, shape];
                m.omega[0] = tilt;
                m.omega[1] = -tilt;
                found.push(optimize(ps, b, m, true, 6));
            }
        }
    }
    found.push(optimize(ps, b, seed, true, 6));
    found.sort_by(|a, b| a.fit_loss.total_cmp(&b.fit_loss));
    let best = found[0].fit_loss;
    found.retain(|h| h.fit_loss <= best + 0.025);
    found.truncate(8);
    found
}
#[derive(Serialize)]
pub struct Metrics {
    pub loss: f64,
    pub support: f64,
    pub samples: usize,
}
pub fn metrics(ps: &[Patch], b: &Picture, m: Model, held: bool) -> Metrics {
    let values = ps
        .iter()
        .filter(|p| p.held == held)
        .map(|p| cost(p, b, m.prepare()))
        .collect::<Vec<_>>();
    Metrics {
        loss: if values.is_empty() {
            1.
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        },
        support: values.iter().filter(|&&x| x < 0.36).count() as f64 / values.len().max(1) as f64,
        samples: values.len(),
    }
}
#[derive(Serialize)]
pub struct Fidelity {
    pub relative_error: f64,
    pub overlap: f64,
    pub samples: usize,
}
pub fn fidelity(a: &Picture, b: &Picture, m: Model) -> Fidelity {
    let a = normalized(a);
    let b = normalized(b);
    let ps = patches(&a, if a[0].w > 18 { 3 } else { 1 });
    let held = ps.iter().filter(|p| p.held).collect::<Vec<_>>();
    let values = held
        .iter()
        .filter_map(|p| relative_cost(p, &b, m.prepare()))
        .collect::<Vec<_>>();
    Fidelity {
        relative_error: if values.is_empty() {
            1.
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        },
        overlap: values.len() as f64 / held.len().max(1) as f64,
        samples: values.len(),
    }
}
#[derive(Serialize)]
pub struct Level {
    pub width: usize,
    pub height: usize,
    pub relative_held: Fidelity,
    pub baseline: Hypothesis,
    pub retained: Vec<Hypothesis>,
    pub alternative: Option<Hypothesis>,
    pub baseline_held: Metrics,
    pub candidate_held: Metrics,
    pub layered_held: Metrics,
    pub observations: Vec<Observation>,
    pub baseline_map: Vec<Option<P>>,
    pub candidate_map: Vec<Option<P>>,
}
#[derive(Serialize)]
pub struct Observation {
    pub p: P,
    pub q: P,
    pub loss_2d: f64,
    pub loss_3d: f64,
    pub alternative_loss: Option<f64>,
    pub group: i8,
    pub held: bool,
}
pub fn evaluate(a: &Picture, b: &Picture, crop: P, previous: Option<&Level>) -> Level {
    let original_a = a;
    let original_b = b;
    let normalized_a = normalized(a);
    let normalized_b = normalized(b);
    let a = &normalized_a;
    let b = &normalized_b;
    let ps = patches(a, if previous.is_some() { 3 } else { 1 });
    let fit = ps.iter().filter(|p| !p.held).collect::<Vec<_>>();
    let seed = Model::new(a[0].w, a[0].h, crop);
    let baseline = if let Some(old) = previous {
        optimize(&fit, b, old.baseline.model.scaled(3.), false, 4)
    } else {
        search(&fit, b, seed, false).remove(0)
    };
    let retained = if let Some(old) = previous {
        let mut hs = old
            .retained
            .iter()
            .map(|h| optimize(&fit, b, h.model.scaled(3.), true, 4))
            .collect::<Vec<_>>();
        hs.sort_by(|a, b| a.fit_loss.total_cmp(&b.fit_loss));
        let best = hs[0].fit_loss;
        hs.retain(|h| h.fit_loss <= best + 0.025);
        hs
    } else {
        search(&fit, b, baseline.model, true)
    };
    let m = retained[0].model;
    let residual = fit
        .iter()
        .copied()
        .filter(|p| cost(p, b, m.prepare()) > 0.3)
        .collect::<Vec<_>>();
    let alternative = if residual.len() >= 12 {
        Some(search(&residual, b, m, true).remove(0))
    } else {
        None
    };
    let mut observations = Vec::new();
    let mut layered = Vec::new();
    for p in &ps {
        let c = cost(p, b, m.prepare());
        let ac = alternative.as_ref().map(|h| cost(p, b, h.model.prepare()));
        let other = ac.is_some_and(|x| x < 0.36 && x + 0.1 < c);
        let group = if other {
            1
        } else if c < 0.36 {
            0
        } else {
            -1
        };
        if p.held {
            layered.push(if other { ac.unwrap() } else { c });
        }
        observations.push(Observation {
            p: p.p,
            q: m.prepare().map(p.p).unwrap_or(p.p),
            loss_2d: cost(p, b, baseline.model.prepare()),
            loss_3d: c,
            alternative_loss: ac,
            group,
            held: p.held,
        });
    }
    let dense = |m: Model| {
        (0..a[0].w * a[0].h)
            .map(|k| m.prepare().map([(k % a[0].w) as f64, (k / a[0].w) as f64]))
            .collect()
    };
    Level {
        width: a[0].w,
        height: a[0].h,
        relative_held: fidelity(original_a, original_b, m),
        baseline_held: metrics(&ps, b, baseline.model, true),
        candidate_held: metrics(&ps, b, m, true),
        layered_held: Metrics {
            loss: if layered.is_empty() {
                1.
            } else {
                layered.iter().sum::<f64>() / layered.len() as f64
            },
            support: layered.iter().filter(|&&x| x < 0.36).count() as f64
                / layered.len().max(1) as f64,
            samples: layered.len(),
        },
        baseline_map: dense(baseline.model),
        candidate_map: dense(m),
        baseline,
        retained,
        alternative,
        observations,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn textureless_input_has_no_fresh_support() {
        let im: Picture = std::array::from_fn(|_| Image {
            w: 18,
            h: 12,
            v: vec![0.5; 216],
        });
        let result = evaluate(&im, &im, [0.; 2], None);
        assert_eq!(result.candidate_held.samples, 0);
        assert_eq!(result.candidate_held.support, 0.);
        assert_eq!(result.candidate_held.loss, 1.);
        assert!(result.observations.is_empty());
    }
    #[test]
    fn identity_and_crop_are_exact() {
        let m = Model::new(53, 35, [2., -1.]);
        for p in [[4., 7.], [34., 23.]] {
            let q = m.prepare().map(p).unwrap();
            assert!((q[0] - p[0] - 2.).abs() < 1e-10);
            assert!((q[1] - p[1] + 1.).abs() < 1e-10);
        }
    }
    #[test]
    fn yaw_has_depth_dependent_parallax() {
        let mut a = Model::new(53, 35, [0.; 2]);
        a.omega[1] = 0.2;
        let mut b = a;
        b.surface[2] = 1.;
        let p = [4., 7.];
        let qa = a.prepare().map(p).unwrap();
        let qb = b.prepare().map(p).unwrap();
        assert!((qa[0] - qb[0]).abs() > 0.2);
    }
    #[test]
    fn level_scaling_preserves_rays() {
        let mut m = Model::new(18, 12, [0.3, -0.2]);
        m.omega = [0.12, -0.2, 0.05];
        m.surface = [0.3, -0.2, 0.4];
        m.translation = [0.5, -0.2, 0.08];
        let p = [6., 7.];
        let a = m.prepare().map(p).unwrap();
        let b = m.scaled(3.).prepare().map(p.map(|v| v * 3.)).unwrap();
        for i in 0..2 {
            assert!((a[i] * 3. - b[i]).abs() < 1e-10);
        }
    }
    #[test]
    fn rotation_is_proper() {
        let r = rotation([0.1, -0.3, 0.2]);
        for i in 0..3 {
            for j in 0..3 {
                let d = (0..3).map(|k| r[i][k] * r[j][k]).sum::<f64>();
                assert!((d - f64::from(i == j)).abs() < 1e-12);
            }
        }
    }
}
