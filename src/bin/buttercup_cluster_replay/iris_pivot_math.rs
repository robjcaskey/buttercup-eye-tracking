//! Direct iris-texture search. A 3D rotation of a reference plane projects to
//! an affine warp under orthographic projection. Pivot coordinates have the
//! image's scale; they are not metric anatomical estimates.
use super::super::math::{distance, Image, P};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Affine {
    pub a: [[f64; 2]; 2],
    pub t: P,
}
impl Affine {
    pub fn translation(t: P) -> Self {
        Self {
            a: [[1., 0.], [0., 1.]],
            t,
        }
    }
    pub fn map(self, p: P) -> P {
        [
            self.a[0][0] * p[0] + self.a[0][1] * p[1] + self.t[0],
            self.a[1][0] * p[0] + self.a[1][1] * p[1] + self.t[1],
        ]
    }
    pub fn inverse(self) -> Option<Self> {
        let d = self.a[0][0] * self.a[1][1] - self.a[0][1] * self.a[1][0];
        if d.abs() < 0.1 {
            return None;
        }
        let a = [
            [self.a[1][1] / d, -self.a[0][1] / d],
            [-self.a[1][0] / d, self.a[0][0] / d],
        ];
        Some(Self {
            a,
            t: [
                -a[0][0] * self.t[0] - a[0][1] * self.t[1],
                -a[1][0] * self.t[0] - a[1][1] * self.t[1],
            ],
        })
    }
}
pub fn rotation(w: [f64; 3]) -> [[f64; 3]; 3] {
    let angle = w.iter().map(|v| v * v).sum::<f64>().sqrt();
    if angle < 1e-12 {
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    }
    let u = w.map(|v| v / angle);
    let (s, c) = angle.sin_cos();
    let k = [[0., -u[2], u[1]], [u[2], 0., -u[0]], [-u[1], u[0], 0.]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| c * f64::from(i == j) + (1. - c) * u[i] * u[j] + s * k[i][j])
    })
}
pub fn project(pivot: [f64; 3], omega: [f64; 3], origin_shift: P) -> Affine {
    let r = rotation(omega);
    Affine {
        a: [[r[0][0], r[0][1]], [r[1][0], r[1][1]]],
        t: std::array::from_fn(|i| {
            pivot[i] - (0..3).map(|j| r[i][j] * pivot[j]).sum::<f64>() + origin_shift[i]
        }),
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Region {
    pub center: P,
    pub radii: P,
    pub inner: f64,
    pub outer: f64,
}
impl Region {
    pub fn contains(self, p: P) -> bool {
        let x = (p[0] - self.center[0]) / self.radii[0];
        let y = (p[1] - self.center[1]) / self.radii[1];
        let r = x.hypot(y);
        r >= self.inner && r <= self.outer && y > -0.68
    }
    pub fn patch(self, p: P, r: f64) -> bool {
        [-r, 0., r].into_iter().all(|y| {
            [-r, 0., r]
                .into_iter()
                .all(|x| self.contains([p[0] + x, p[1] + y]))
        })
    }
    pub fn iris_context(self, p: P, r: f64) -> bool {
        let dx = (p[0] - self.center[0]).abs();
        let dy = (p[1] - self.center[1]).abs();
        ((dx + r) / self.radii[0]).hypot((dy + r) / self.radii[1]) < 0.96
            && ((dx - r).max(0.) / self.radii[0]).hypot((dy - r).max(0.) / self.radii[1]) > 0.34
            && p[1] - r > self.center[1] - 0.84 * self.radii[1]
    }
}

pub struct Features {
    pub image: Image,
    pub valid: Vec<bool>,
    pub center_hint: Option<P>,
}
pub fn smooth(im: &Image, radius: usize) -> Image {
    if radius == 0 {
        return im.clone();
    }
    let (w, h) = (im.w, im.h);
    let mut integral = vec![0.; (w + 1) * (h + 1)];
    for y in 0..h {
        for x in 0..w {
            let k = (y + 1) * (w + 1) + x + 1;
            integral[k] =
                im.v[y * w + x] + integral[k - 1] + integral[k - w - 1] - integral[k - w - 2];
        }
    }
    let v = (0..w * h)
        .map(|k| {
            let (x, y) = (k % w, k / w);
            let (x0, y0) = (x.saturating_sub(radius), y.saturating_sub(radius));
            let (x1, y1) = ((x + radius + 1).min(w), (y + radius + 1).min(h));
            rectangle(&integral, w, x0, y0, x1, y1) / ((x1 - x0) * (y1 - y0)) as f64
        })
        .collect();
    Image { w, h, v }
}
fn rectangle(sum: &[f64], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    sum[y1 * (w + 1) + x1] + sum[y0 * (w + 1) + x0]
        - sum[y0 * (w + 1) + x1]
        - sum[y1 * (w + 1) + x0]
}
impl Features {
    pub fn new(raw: &Image, bright_limit: f64) -> Self {
        Self::with_mask(raw, raw, bright_limit)
    }
    pub fn with_mask(raw: &Image, bright_source: &Image, bright_limit: f64) -> Self {
        let (w, h) = (raw.w, raw.h);
        let mut integral = vec![0.; (w + 1) * (h + 1)];
        let mut square = integral.clone();
        let mut bright = integral.clone();
        for y in 0..h {
            for x in 0..w {
                let k = (y + 1) * (w + 1) + x + 1;
                let v = raw.v[y * w + x];
                for (a, s) in [
                    (&mut integral, v),
                    (&mut square, v * v),
                    (
                        &mut bright,
                        f64::from(bright_source.v[y * w + x] > bright_limit),
                    ),
                ] {
                    a[k] = s + a[k - 1] + a[k - w - 1] - a[k - w - 2];
                }
            }
        }
        let mut v = vec![0.; w * h];
        let mut valid = vec![false; w * h];
        for y in 9..h - 9 {
            for x in 9..w - 9 {
                let mean = rectangle(&integral, w, x - 7, y - 7, x + 8, y + 8) / 225.;
                let var = (rectangle(&square, w, x - 7, y - 7, x + 8, y + 8) / 225. - mean * mean)
                    .max(0.);
                // Bounded local contrast normalization; full-period RAW averaging
                // happened upstream and is identical for every comparison method.
                v[y * w + x] = ((raw.v[y * w + x] - mean) / (var + 0.000025).sqrt()).clamp(-3., 3.);
                valid[y * w + x] = x >= 13
                    && y >= 13
                    && x + 14 <= w
                    && y + 14 <= h
                    && rectangle(&bright, w, x - 13, y - 13, x + 14, y + 14) < 0.5;
            }
        }
        Self {
            image: Image { w, h, v },
            valid,
            center_hint: None,
        }
    }
    pub fn sample(&self, p: P) -> Option<f64> {
        if p[0] < 1.
            || p[1] < 1.
            || p[0] >= self.image.w as f64 - 2.
            || p[1] >= self.image.h as f64 - 2.
        {
            return None;
        }
        let (x, y) = (p[0] as usize, p[1] as usize);
        let w = self.image.w;
        if ![
            y * w + x,
            y * w + x + 1,
            (y + 1) * w + x,
            (y + 1) * w + x + 1,
        ]
        .into_iter()
        .all(|k| self.valid[k])
        {
            return None;
        }
        self.image.sample(p)
    }
}
#[derive(Clone)]
pub struct Samples {
    pub points: Vec<P>,
    pub values: Vec<f64>,
    pub center: P,
}
impl Samples {
    pub fn new(f: &Features, roi: Region, fold: usize) -> Self {
        let mut points = Vec::new();
        let mut values = Vec::new();
        for y in (12..f.image.h - 12).step_by(3) {
            for x in (12..f.image.w - 12).step_by(3) {
                // Split image regions, with a guard at the tile boundary. This is
                // withheld texture, not independent biological identity evidence.
                if (x / 16 + y / 16) % 2 != fold
                    || !(3..=12).contains(&(x % 16))
                    || !(3..=12).contains(&(y % 16))
                {
                    continue;
                }
                let p = [x as f64, y as f64];
                if roi.contains(p) && roi.iris_context(p, 15.) {
                    if let Some(v) = f.sample(p) {
                        points.push(p);
                        values.push(v)
                    }
                }
            }
        }
        Self {
            points,
            values,
            center: roi.center,
        }
    }
    pub fn coarse(&self, max: usize) -> Self {
        let n = max.min(self.points.len());
        let ids: Vec<_> = (0..n).map(|i| i * self.points.len() / n).collect();
        Self {
            points: ids.iter().map(|&i| self.points[i]).collect(),
            values: ids.iter().map(|&i| self.values[i]).collect(),
            center: self.center,
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Score {
    pub loss: f64,
    pub ncc: f64,
    pub coverage: f64,
}
pub fn score(samples: &Samples, target: &Features, m: Affine) -> Score {
    if target
        .center_hint
        .is_some_and(|c| distance(m.map(samples.center), c) > 16.)
    {
        return Score {
            loss: 5.,
            ncc: -1.,
            coverage: 0.,
        };
    }
    let (mut sx, mut sy, mut xx, mut yy, mut xy, mut n) = (0., 0., 0., 0., 0., 0usize);
    for (&p, &x) in samples.points.iter().zip(&samples.values) {
        if let Some(y) = target.sample(m.map(p)) {
            sx += x;
            sy += y;
            xx += x * x;
            yy += y * y;
            xy += x * y;
            n += 1;
        }
    }
    let coverage = n as f64 / samples.points.len().max(1) as f64;
    let nf = n.max(1) as f64;
    let var = (xx - sx * sx / nf) * (yy - sy * sy / nf);
    let ncc = if n >= 12 && var > 1e-8 {
        (xy - sx * sy / nf) / var.sqrt()
    } else {
        -1.
    };
    // The bright-reflection mask denotes occluded texture. Penalizing each
    // missing point heavily made an incorrect unoccluded background beat the
    // correct iris warp. Require bounded visible support and report coverage.
    let loss = 1. - ncc + 0.25 * (1. - coverage) + if coverage < 0.60 { 3. } else { 0. };
    Score {
        loss,
        ncc,
        coverage,
    }
}
pub fn grid_angles() -> Vec<[f64; 3]> {
    let mut a = Vec::new();
    for x in -12..=12 {
        for y in -12..=12 {
            for z in -6..=6 {
                a.push([
                    x as f64 * 2f64.to_radians(),
                    y as f64 * 2f64.to_radians(),
                    z as f64 * 2f64.to_radians(),
                ])
            }
        }
    }
    a
}
pub fn grid_pivots(c: P) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for z in 0..9 {
        for y in -4..=4 {
            for x in -4..=4 {
                out.push([
                    c[0] + x as f64 * 20.,
                    c[1] + y as f64 * 20.,
                    60. + z as f64 * 30.,
                ])
            }
        }
    }
    out
}
pub fn refine_rotation(
    s: &Samples,
    f: &Features,
    pivot: [f64; 3],
    delta: P,
    mut w: [f64; 3],
) -> ([f64; 3], Score) {
    let mut best = score(s, f, project(pivot, w, delta));
    for step in [1.0f64, 0.5, 0.25, 0.125] {
        for _ in 0..6 {
            let previous = best.loss;
            for axis in 0..3 {
                for sign in [-1., 1.] {
                    let mut q = w;
                    q[axis] += sign * step.to_radians();
                    if q[axis].abs()
                        > if axis == 2 {
                            18f64.to_radians()
                        } else {
                            30f64.to_radians()
                        }
                    {
                        continue;
                    }
                    let v = score(s, f, project(pivot, q, delta));
                    if v.loss + 1e-9 < best.loss {
                        w = q;
                        best = v
                    }
                }
            }
            if (previous - best.loss).abs() < 1e-9 {
                break;
            }
        }
    }
    (w, best)
}
pub fn brute_rotation(
    s: &Samples,
    f: &Features,
    pivot: [f64; 3],
    delta: P,
    angles: &[[f64; 3]],
) -> ([f64; 3], Score) {
    let coarse = s.coarse(128);
    let mut basins: Vec<(f64, [f64; 3])> = Vec::new();
    for &a in angles {
        let v = score(&coarse, f, project(pivot, a, delta));
        if basins.len() < 6 || v.loss < basins.last().unwrap().0 {
            basins.push((v.loss, a));
            basins.sort_by(|a, b| a.0.total_cmp(&b.0));
            basins.truncate(6);
        }
    }
    basins
        .into_iter()
        .map(|(_, w)| refine_rotation(s, f, pivot, delta, w))
        .min_by(|a, b| a.1.loss.total_cmp(&b.1.loss))
        .unwrap()
}
pub fn brute_translation(s: &Samples, f: &Features, delta: P) -> (Affine, Score) {
    let coarse = s.coarse(64);
    let mut m = Affine::translation(delta);
    let mut best = score(&coarse, f, m);
    for y in -28..=28 {
        for x in -28..=28 {
            let q = Affine::translation([delta[0] + x as f64 * 2., delta[1] + y as f64 * 2.]);
            let v = score(&coarse, f, q);
            if v.loss < best.loss {
                m = q;
                best = v
            }
        }
    }
    refine_affine(s, f, m, [0., 0.], false)
}
pub fn iris_region_center(raw: &Image, roi: Region, delta: P) -> (P, f64) {
    // Coarse lateral dark-to-light contrast bounds the target iris region.
    // Radius is a fixed support prior; this is not a limbus geometry solve.
    let mut best = (
        [roi.center[0] + delta[0], roi.center[1] + delta[1]],
        f64::NEG_INFINITY,
    );
    for dy in -20..=20 {
        for dx in -28..=28 {
            let c = [
                roi.center[0] + delta[0] + dx as f64 * 2.,
                roi.center[1] + delta[1] + dy as f64 * 2.,
            ];
            let mut sides = [0.; 2];
            let mut valid = true;
            for side in 0..2 {
                for j in -3..=3 {
                    let t = j as f64 * 0.2 + side as f64 * std::f64::consts::PI;
                    let point = |r: f64| {
                        [
                            c[0] + r * roi.radii[0] * t.cos(),
                            c[1] + r * roi.radii[1] * t.sin(),
                        ]
                    };
                    match (raw.sample(point(0.8)), raw.sample(point(1.16))) {
                        (Some(a), Some(b)) => sides[side] += (b.min(0.65) - a.min(0.65)) / 7.,
                        _ => valid = false,
                    }
                }
            }
            let score = sides[0].min(sides[1]) + 0.15 * (sides[0] + sides[1]);
            if valid && score > best.1 {
                best = (c, score)
            }
        }
    }
    best
}
pub fn refine_affine(
    s: &Samples,
    f: &Features,
    mut m: Affine,
    c: P,
    deform: bool,
) -> (Affine, Score) {
    let mut best = score(s, f, m);
    for scale in [2., 1., 0.5, 0.25] {
        for _ in 0..10 {
            let prev = best.loss;
            for axis in 0..if deform { 6 } else { 2 } {
                for sign in [-1., 1.] {
                    let mut q = m;
                    if axis < 2 {
                        q.t[axis] += sign * scale;
                    } else {
                        let (i, j) = ((axis - 2) / 2, (axis - 2) % 2);
                        let v = sign * scale * 0.015;
                        q.a[i][j] += v;
                        q.t[i] -= v * c[j];
                    }
                    let det = q.a[0][0] * q.a[1][1] - q.a[0][1] * q.a[1][0];
                    if !(0.45..1.65).contains(&det)
                        || q.a[0][0] < 0.55
                        || q.a[1][1] < 0.55
                        || q.a[0][1].abs() > 0.5
                        || q.a[1][0].abs() > 0.5
                    {
                        continue;
                    }
                    let v = score(s, f, q);
                    if v.loss + 1e-9 < best.loss {
                        m = q;
                        best = v
                    }
                }
            }
            if prev - best.loss < 1e-9 {
                break;
            }
        }
    }
    (m, best)
}
pub fn iris_seeds(raw: &Image, f: &Features, roi: Region) -> Vec<P> {
    let mut ranked = Vec::new();
    for y in (20..raw.h - 20).step_by(3) {
        for x in (20..raw.w - 20).step_by(3) {
            let p = [x as f64, y as f64];
            if !roi.patch(p, 9.) || !roi.iris_context(p, 15.) || f.sample(p).is_none() {
                continue;
            }
            let (mut xx, mut xy, mut yy) = (0., 0., 0.);
            for dy in [-3., 0., 3.] {
                for dx in [-3., 0., 3.] {
                    let x = p[0] + dx;
                    let y = p[1] + dy;
                    let gx = raw.sample([x + 1., y]).unwrap() - raw.sample([x - 1., y]).unwrap();
                    let gy = raw.sample([x, y + 1.]).unwrap() - raw.sample([x, y - 1.]).unwrap();
                    xx += gx * gx;
                    xy += gx * gy;
                    yy += gy * gy;
                }
            }
            ranked.push((xx + yy - ((xx - yy).powi(2) + 4. * xy * xy).sqrt(), p));
        }
    }
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = Vec::new();
    for (_, p) in ranked {
        if out.iter().all(|&q| distance(p, q) >= 12.) {
            out.push(p);
            if out.len() == 48 {
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_projection_is_exact_for_plane_rotation_about_3d_pivot() {
        let c = [210., 130., 180.];
        let w = [0.13, -0.2, 0.07];
        let p = [155., 111.];
        let shift = [2., -3.];
        let r = rotation(w);
        let q = project(c, w, shift).map(p);
        for i in 0..2 {
            let exact = c[i] + r[i][0] * (p[0] - c[0]) + r[i][1] * (p[1] - c[1]) - r[i][2] * c[2]
                + shift[i];
            assert!((exact - q[i]).abs() < 1e-10)
        }
        assert!(distance(project(c, w, shift).inverse().unwrap().map(q), p) < 1e-10);
    }
    #[test]
    fn free_translation_makes_the_pivot_unidentifiable() {
        let a = project([190., 140., 180.], [0.1, -0.15, 0.03], [0., 0.]);
        let b = project([240., 95., 250.], [0.1, -0.15, 0.03], [0., 0.]);
        let corrected = Affine { t: a.t, ..b };
        for p in [[125., 135.], [200., 170.], [230., 110.]] {
            assert!(distance(a.map(p), corrected.map(p)) < 1e-10)
        }
    }
    #[test]
    fn iris_support_excludes_void_and_outer_boundary() {
        let r = Region {
            center: [190., 130.],
            radii: [90., 80.],
            inner: 0.48,
            outer: 0.84,
        };
        assert!(!r.contains(r.center));
        assert!(!r.contains([280., 130.]));
        assert!(r.contains([246., 130.]));
        assert!(!r.patch([235., 130.], 9.));
    }
}
