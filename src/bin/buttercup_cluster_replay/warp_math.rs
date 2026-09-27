//! Bounded image-only alignment experiment. No inferred 3D point is counted as
//! a fresh observation: every returned match must pass independent image checks.
use serde::Serialize;
pub type P = [f64; 2];
pub const IDENTITY: [[f64; 2]; 2] = [[1., 0.], [0., 1.]];
pub fn distance(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn mul(a: [[f64; 2]; 2], p: P) -> P {
    [
        a[0][0] * p[0] + a[0][1] * p[1],
        a[1][0] * p[0] + a[1][1] * p[1],
    ]
}
fn inverse(a: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let d = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    (d > 0.15 && d < 5.).then_some([[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]])
}
pub fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N], n: usize) -> Option<[f64; N]> {
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if !a[p][k].is_finite() || a[p][k].abs() < 1e-10 {
            return None;
        }
        a.swap(k, p);
        b.swap(k, p);
        let scale = a[k][k];
        for j in k..n {
            a[k][j] /= scale;
        }
        b[k] /= scale;
        for i in 0..n {
            if i != k {
                let s = a[i][k];
                for j in k..n {
                    a[i][j] -= s * a[k][j];
                }
                b[i] -= s * b[k];
            }
        }
    }
    b.iter().take(n).all(|v| v.is_finite()).then_some(b)
}
#[derive(Clone)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f64>,
}
impl Image {
    pub fn raw(raw: &[u16], w: usize, h: usize) -> Self {
        // Same complete 4x4 Quad-Bayer carrier period as the generic tracker.
        // Floating means avoid an additional quantization; both variants share it.
        let mut v = vec![0.; raw.len()];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.;
                for dy in -1isize..=2 {
                    for dx in -1isize..=2 {
                        s += raw[y.saturating_add_signed(dy).min(h - 1) * w
                            + x.saturating_add_signed(dx).min(w - 1)]
                            as f64;
                    }
                }
                v[y * w + x] = s / (16. * 1023.);
            }
        }
        Self { w, h, v }
    }
    pub fn sample(&self, p: P) -> Option<f64> {
        if !p[0].is_finite()
            || !p[1].is_finite()
            || p[0] < 0.
            || p[1] < 0.
            || p[0] >= self.w as f64 - 1.
            || p[1] >= self.h as f64 - 1.
        {
            return None;
        }
        let x = p[0] as usize;
        let y = p[1] as usize;
        let u = p[0] - x as f64;
        let v = p[1] - y as f64;
        Some(
            (1. - v) * ((1. - u) * self.v[y * self.w + x] + u * self.v[y * self.w + x + 1])
                + v * ((1. - u) * self.v[(y + 1) * self.w + x]
                    + u * self.v[(y + 1) * self.w + x + 1]),
        )
    }
    fn gradient(&self, p: P) -> Option<P> {
        Some([
            (self.sample([p[0] + 0.7, p[1]])? - self.sample([p[0] - 0.7, p[1]])?) / 1.4,
            (self.sample([p[0], p[1] + 0.7])? - self.sample([p[0], p[1] - 0.7])?) / 1.4,
        ])
    }
}
pub fn seeds(im: &Image, limit: usize) -> Vec<P> {
    // Spatially balanced Shi-Tomasi corners, no iris/face/reflection mask.
    let mut cells = Vec::new();
    for ty in (20..im.h - 20).step_by(20) {
        for tx in (20..im.w - 20).step_by(20) {
            let mut best = (0., [0., 0.]);
            for y in (ty..(ty + 20).min(im.h - 20)).step_by(2) {
                for x in (tx..(tx + 20).min(im.w - 20)).step_by(2) {
                    let (mut xx, mut xy, mut yy) = (0., 0., 0.);
                    for dy in [-3., 0., 3.] {
                        for dx in [-3., 0., 3.] {
                            let g = im.gradient([x as f64 + dx, y as f64 + dy]).unwrap();
                            xx += g[0] * g[0];
                            xy += g[0] * g[1];
                            yy += g[1] * g[1];
                        }
                    }
                    let score = 0.5 * (xx + yy - ((xx - yy).powi(2) + 4. * xy * xy).sqrt());
                    if score > best.0 {
                        best = (score, [x as f64, y as f64]);
                    }
                }
            }
            if best.0 > 1e-6 {
                cells.push(best);
            }
        }
    }
    cells.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = Vec::new();
    for (_, p) in cells {
        if out.iter().all(|&q| distance(p, q) > 9.) {
            out.push(p);
            if out.len() == limit {
                break;
            }
        }
    }
    out
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Sphere {
    pub center: P,
    pub radius: f64,
    pub shift: P,
    pub omega: [f64; 3],
    pub support: usize,
    pub residual: f64,
}
fn rotate(p: [f64; 3], w: [f64; 3]) -> [f64; 3] {
    let t = w.iter().map(|v| v * v).sum::<f64>().sqrt();
    if t < 1e-12 {
        return p;
    }
    let u = w.map(|v| v / t);
    let d = (0..3).map(|k| u[k] * p[k]).sum::<f64>();
    let (s, c) = t.sin_cos();
    let cross = [
        u[1] * p[2] - u[2] * p[1],
        u[2] * p[0] - u[0] * p[2],
        u[0] * p[1] - u[1] * p[0],
    ];
    std::array::from_fn(|k| c * p[k] + s * cross[k] + (1. - c) * d * u[k])
}
impl Sphere {
    pub fn map(&self, p: P) -> Option<P> {
        let x = p[0] - self.center[0];
        let y = p[1] - self.center[1];
        let z2 = self.radius * self.radius - x * x - y * y;
        if z2 < self.radius * self.radius * 0.06 {
            return None;
        }
        // Camera x-right/y-down/z-away, front hemisphere has negative z.
        let q = rotate([x, y, -z2.sqrt()], self.omega);
        if q[2] > -self.radius * 0.12 {
            return None;
        }
        Some([
            self.center[0] + self.shift[0] + q[0],
            self.center[1] + self.shift[1] + q[1],
        ])
    }
    fn reversed(self, extra: P) -> Self {
        Self {
            center: [
                self.center[0] + self.shift[0] + extra[0],
                self.center[1] + self.shift[1] + extra[1],
            ],
            shift: [-self.shift[0] - extra[0], -self.shift[1] - extra[1]],
            omega: self.omega.map(|v| -v),
            ..self
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Warp {
    pub p: P,
    pub q: P,
    pub a: [[f64; 2]; 2],
    pub sphere: Option<Sphere>,
}
impl Warp {
    pub fn plain(p: P, q: P) -> Self {
        Self {
            p,
            q,
            a: IDENTITY,
            sphere: None,
        }
    }
    fn at(&self, d: P) -> Option<P> {
        let v = if let Some(s) = self.sphere {
            let a = s.map(self.p)?;
            let b = s.map([self.p[0] + d[0], self.p[1] + d[1]])?;
            [b[0] - a[0], b[1] - a[1]]
        } else {
            mul(self.a, d)
        };
        Some([self.q[0] + v[0], self.q[1] + v[1]])
    }
    fn reversed(self) -> Option<Self> {
        let sphere = if let Some(s) = self.sphere {
            let q = s.map(self.p)?;
            Some(s.reversed([self.q[0] - q[0], self.q[1] - q[1]]))
        } else {
            None
        };
        Some(Self {
            p: self.q,
            q: self.p,
            a: inverse(self.a)?,
            sphere,
        })
    }
    fn plausible(&self) -> bool {
        let a = self.a;
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        det > 0.55
            && det < 1.8
            && a[0][0] > 0.55
            && a[1][1] > 0.55
            && a[0][1].abs() < 0.45
            && a[1][0].abs() < 0.45
    }
}
const R: f64 = 9.;
fn samples() -> impl Iterator<Item = (P, bool)> {
    (-6..=6)
        .flat_map(|y| (-6..=6).map(move |x| ([x as f64 * 1.5, y as f64 * 1.5], (x + y) % 2 == 0)))
}
fn ncc(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 10 {
        return None;
    }
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut xx, mut xy, mut yy) = (0., 0., 0.);
    for (&x, &y) in a.iter().zip(b) {
        xx += (x - ma).powi(2);
        xy += (x - ma) * (y - mb);
        yy += (y - mb).powi(2);
    }
    (xx > 1e-5 && yy > 1e-5).then_some(xy / (xx * yy).sqrt())
}
pub fn correlation(im: &Image, now: &Image, w: Warp, held: bool) -> Option<f64> {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for (d, train) in samples() {
        if train == held {
            continue;
        }
        a.push(im.sample([w.p[0] + d[0], w.p[1] + d[1]])?);
        b.push(now.sample(w.at(d)?)?);
    }
    ncc(&a, &b)
}
fn optimize(im: &Image, now: &Image, mut w: Warp, deform: bool) -> Result<Warp, &'static str> {
    let initial = w.q;
    let n = if deform { 8 } else { 4 };
    let photo = if deform { 6 } else { 2 };
    let mut gain = 1.;
    let mut bias = 0.;
    let template = samples()
        .filter(|(_, train)| *train)
        .map(|(d, _)| im.sample([w.p[0] + d[0], w.p[1] + d[1]]).map(|v| (d, v)))
        .collect::<Option<Vec<_>>>()
        .ok_or("patch_bounds")?;
    for _ in 0..16 {
        let mut h = [[0.; 8]; 8];
        let mut b = [0.; 8];
        for &(d, t) in &template {
            let p = w.at(d).ok_or("sphere_visibility")?;
            let v = now.sample(p).ok_or("patch_bounds")?;
            let g = now.gradient(p).ok_or("patch_bounds")?;
            let residual = v - gain * t - bias;
            let mut j = [0.; 8];
            j[0] = g[0];
            j[1] = g[1];
            if deform {
                j[2] = g[0] * d[0] / R;
                j[3] = g[0] * d[1] / R;
                j[4] = g[1] * d[0] / R;
                j[5] = g[1] * d[1] / R;
            }
            j[photo] = -t;
            j[photo + 1] = -1.;
            // Same robust photometric loss for all variants.
            let weight = (0.045 / residual.abs().max(0.045)).min(1.);
            for k in 0..n {
                b[k] -= weight * j[k] * residual;
                for l in 0..n {
                    h[k][l] += weight * j[k] * j[l];
                }
            }
        }
        for k in 0..n {
            h[k][k] += 1e-6;
        }
        // Shape regularization is centered on identity, identical for every
        // freely affine patch; it is not a motion-direction constraint.
        if deform {
            for k in 2..6 {
                let q = (k - 2) / 2;
                let r = (k - 2) % 2;
                let prior = R * (w.a[q][r] - IDENTITY[q][r]);
                h[k][k] += 0.00003;
                b[k] -= 0.00003 * prior;
            }
        }
        let step = solve(h, b, n).ok_or("singular_patch")?;
        let f = (2.0 / step[0].hypot(step[1]).max(2.)).min(1.);
        w.q[0] += step[0] * f;
        w.q[1] += step[1] * f;
        if deform {
            for k in 2..6 {
                w.a[(k - 2) / 2][(k - 2) % 2] += (step[k] * f / R).clamp(-0.08, 0.08);
            }
        }
        gain = (gain + step[photo] * f).clamp(0.35, 2.8);
        bias = (bias + step[photo + 1] * f).clamp(-0.35, 0.35);
        if !w.plausible() {
            return Err("warp_limit");
        }
        if distance(initial, w.q) > 24. {
            return Err("search_limit");
        }
        if step[0].hypot(step[1]) < 0.012 && (!deform || step[2..6].iter().all(|v| v.abs() < 0.012))
        {
            break;
        }
    }
    Ok(w)
}
#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    pub q: P,
    pub a: [[f64; 2]; 2],
    pub ncc: f64,
    pub fb: f64,
    pub source: &'static str,
    pub witnesses: usize,
    pub sphere_model: Option<Sphere>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Audit {
    pub accepted: bool,
    pub reason: String,
    pub best_ncc: Option<f64>,
    pub fb: Option<f64>,
    pub candidate: Option<P>,
}
impl Audit {
    fn fail(reason: &str) -> Self {
        Self {
            accepted: false,
            reason: reason.into(),
            best_ncc: None,
            fb: None,
            candidate: None,
        }
    }
}
pub fn checked(
    im: &Image,
    now: &Image,
    w: Warp,
    deform: bool,
    source: &'static str,
    min_ncc: f64,
) -> (Option<Hit>, Audit) {
    let w = match optimize(im, now, w, deform) {
        Ok(w) => w,
        Err(e) => return (None, Audit::fail(e)),
    };
    let score = correlation(im, now, w, true);
    let mut audit = Audit {
        accepted: false,
        reason: "photometric".into(),
        best_ncc: score,
        fb: None,
        candidate: Some(w.q),
    };
    if !score.is_some_and(|v| v >= min_ncc) {
        return (None, audit);
    }
    if let Some(s) = w.sphere {
        if !s
            .map(w.p)
            .is_some_and(|q| distance(q, w.q) <= (0.75 + 2. * s.residual).min(2.))
        {
            audit.reason = "shared_prediction_mismatch".into();
            return (None, audit);
        }
    }
    let Some(reverse) = w.reversed() else {
        audit.reason = "inverse_warp".into();
        return (None, audit);
    };
    let backward = match optimize(now, im, reverse, deform) {
        Ok(w) => w,
        Err(e) => {
            audit.reason = format!("reverse_{e}");
            return (None, audit);
        }
    };
    let fb = distance(backward.q, w.p);
    audit.fb = Some(fb);
    if fb > 0.9 {
        audit.reason = "forward_backward".into();
        return (None, audit);
    }
    if !correlation(now, im, backward, true).is_some_and(|v| v >= min_ncc) {
        audit.reason = "reverse_photometric".into();
        return (None, audit);
    }
    audit.accepted = true;
    audit.reason = "accepted".into();
    (
        Some(Hit {
            q: w.q,
            a: w.a,
            ncc: score.unwrap(),
            fb,
            source,
            witnesses: w.sphere.map_or(0, |s| s.support),
            sphere_model: w.sphere,
        }),
        audit,
    )
}
pub fn independent(
    im: &Image,
    now: &Image,
    p: P,
    predicted: P,
    deform: bool,
    min_ncc: f64,
) -> (Option<Hit>, Audit) {
    // Identical translation search for both ablations; affine degrees of
    // freedom are introduced only after selecting distinct translation basins.
    let mut candidates = Vec::new();
    let template: Vec<_> = (-3..=3)
        .flat_map(|y| (-3..=3).map(move |x| [x as f64 * 3., y as f64 * 3.]))
        .filter_map(|d| im.sample([p[0] + d[0], p[1] + d[1]]).map(|v| (d, v)))
        .collect();
    if template.len() != 49 {
        return (None, Audit::fail("patch_bounds"));
    }
    let a: Vec<_> = template.iter().map(|p| p.1).collect();
    for dy in -4..=4 {
        for dx in -4..=4 {
            let q = [predicted[0] + dx as f64 * 3., predicted[1] + dy as f64 * 3.];
            let b = template
                .iter()
                .map(|(d, _)| now.sample([q[0] + d[0], q[1] + d[1]]))
                .collect::<Option<Vec<_>>>();
            if let Some(score) = b.and_then(|b| ncc(&a, &b)) {
                candidates.push((score, q));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut basins = Vec::new();
    for (_, q) in candidates {
        if basins.iter().all(|&p| distance(p, q) >= 4.) {
            basins.push(q);
            if basins.len() == 2 {
                break;
            }
        }
    }
    let mut best = None;
    let mut audit = Audit::fail("no_candidate");
    for &q in &basins {
        let (h, a) = checked(im, now, Warp::plain(p, q), false, "translation", min_ncc);
        if let Some(h) = h {
            if best.as_ref().is_none_or(|b: &Hit| h.ncc > b.ncc) {
                best = Some(h);
                audit = a;
            }
        } else if best.is_none() && a.best_ncc.unwrap_or(-1.) > audit.best_ncc.unwrap_or(-2.) {
            audit = a;
        }
    }
    // The extra degrees of freedom are a rescue, never a compulsory refit of
    // an already validated simple patch. Free affine fitting on weak RAW
    // texture regressed sharply in the matched exploratory runs.
    if best.is_none() && deform {
        for q in basins {
            let (h, a) = checked(im, now, Warp::plain(p, q), true, "affine_rescue", min_ncc);
            if let Some(h) = h {
                if best.as_ref().is_none_or(|b: &Hit| h.ncc > b.ncc) {
                    best = Some(h);
                    audit = a;
                }
            } else if best.is_none() && a.best_ncc.unwrap_or(-1.) > audit.best_ncc.unwrap_or(-2.) {
                audit = a;
            }
        }
    }
    (best, audit)
}
#[derive(Clone, Copy)]
pub struct Pair {
    pub id: usize,
    pub p: P,
    pub q: P,
}
fn random(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}
fn fit_sphere(v: &[Pair], center: P, radius: f64) -> Option<Sphere> {
    let mut h = [[0.; 5]; 5];
    let mut b = [0.; 5];
    for m in v {
        let x = (m.p[0] - center[0]) / radius;
        let y = (m.p[1] - center[1]) / radius;
        let z2 = 1. - x * x - y * y;
        if z2 < 0.08 {
            return None;
        }
        let z = -z2.sqrt();
        let rows = [[1., 0., 0., z, -y], [0., 1., -z, 0., x]];
        for k in 0..2 {
            let target = m.q[k] - m.p[k];
            for i in 0..5 {
                b[i] += rows[k][i] * target;
                for j in 0..5 {
                    h[i][j] += rows[k][i] * rows[k][j];
                }
            }
        }
    }
    let solution = solve(h, b, 5)?;
    let mut s = Sphere {
        center,
        radius,
        shift: [solution[0], solution[1]],
        omega: [
            solution[2] / radius,
            solution[3] / radius,
            solution[4] / radius,
        ],
        support: 0,
        residual: 0.,
    };
    if s.shift[0].hypot(s.shift[1]) > 35.
        || s.omega.iter().map(|x| x * x).sum::<f64>().sqrt() > 0.18
    {
        return None;
    }
    // Refine exact Rodrigues projection, not the infinitesimal approximation.
    for _ in 0..4 {
        let mut h = [[0.; 5]; 5];
        let mut b = [0.; 5];
        for m in v {
            let q = s.map(m.p)?;
            let mut j = [[0.; 5]; 2];
            j[0][0] = 1.;
            j[1][1] = 1.;
            for k in 2..5 {
                let mut next = s;
                next.omega[k - 2] += 0.01 / radius;
                let r = next.map(m.p)?;
                for d in 0..2 {
                    j[d][k] = (r[d] - q[d]) / 0.01;
                }
            }
            for d in 0..2 {
                for k in 0..5 {
                    b[k] += j[d][k] * (m.q[d] - q[d]);
                    for l in 0..5 {
                        h[k][l] += j[d][k] * j[d][l];
                    }
                }
            }
        }
        let d = solve(h, b, 5)?;
        s.shift[0] += d[0];
        s.shift[1] += d[1];
        for k in 0..3 {
            s.omega[k] += d[k + 2] / radius;
        }
    }
    (s.shift[0].hypot(s.shift[1]) < 35. && s.omega.iter().map(|x| x * x).sum::<f64>().sqrt() < 0.18)
        .then_some(s)
}
fn spread(v: &[Pair]) -> bool {
    if v.len() < 8 {
        return false;
    }
    let n = v.len() as f64;
    let c: P = std::array::from_fn(|k| v.iter().map(|p| p.p[k]).sum::<f64>() / n);
    let (mut xx, mut xy, mut yy) = (0., 0., 0.);
    for p in v {
        let x = p.p[0] - c[0];
        let y = p.p[1] - c[1];
        xx += x * x / n;
        xy += x * y / n;
        yy += y * y / n;
    }
    0.5 * (xx + yy - ((xx - yy).powi(2) + 4. * xy * xy).sqrt()) > 144. && xx + yy > 1600.
}
pub fn sphere_models(pairs: &[Pair], w: usize, h: usize, target_parity: usize) -> Vec<Sphere> {
    // Whole tested identities are excluded, including their candidate positions.
    let witnesses: Vec<_> = pairs
        .iter()
        .filter(|p| p.id % 2 != target_parity)
        .copied()
        .collect();
    if witnesses.len() < 10 {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let mut rng = 0xd431ea5u64 + target_parity as u64;
    for cx in [0.3, 0.5, 0.7] {
        for cy in [0.35, 0.55] {
            for rr in [0.36, 0.55, 0.9] {
                let center = [w as f64 * cx, h as f64 * cy];
                let radius = w.max(h) as f64 * rr;
                let inside: Vec<_> = witnesses
                    .iter()
                    .filter(|p| distance(p.p, center) < radius * 0.9)
                    .copied()
                    .collect();
                if !spread(&inside) {
                    continue;
                }
                for _ in 0..18 {
                    let mut seed = Vec::new();
                    for _ in 0..40 {
                        let p = inside[random(&mut rng) as usize % inside.len()];
                        if !seed.iter().any(|a: &Pair| a.id == p.id) {
                            seed.push(p);
                        }
                        if seed.len() == 6 {
                            break;
                        }
                    }
                    let Some(s) = fit_sphere(&seed, center, radius) else {
                        continue;
                    };
                    let support: Vec<_> = inside
                        .iter()
                        .filter(|p| s.map(p.p).is_some_and(|q| distance(q, p.q) < 0.85))
                        .copied()
                        .collect();
                    if !spread(&support) || support.len() < 10 {
                        continue;
                    }
                    let Some(mut s) = fit_sphere(&support, center, radius) else {
                        continue;
                    };
                    let errors = support
                        .iter()
                        .map(|p| s.map(p.p).map(|q| distance(q, p.q)))
                        .collect::<Option<Vec<_>>>();
                    let Some(errors) = errors else { continue };
                    s.residual = errors.iter().sum::<f64>() / errors.len() as f64;
                    s.support = support.len();
                    if s.residual > 0.65 {
                        continue;
                    }
                    candidates.push(s);
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        (b.support as f64 - b.residual * 5.).total_cmp(&(a.support as f64 - a.residual * 5.))
    });
    let mut out = Vec::new();
    for s in candidates {
        if out.iter().all(|t: &Sphere| {
            let ds: Vec<_> = witnesses
                .iter()
                .filter_map(|p| Some(distance(s.map(p.p)?, t.map(p.p)?)))
                .collect();
            ds.is_empty() || ds.iter().sum::<f64>() / ds.len() as f64 > 0.8
        }) {
            out.push(s);
            if out.len() == 3 {
                break;
            }
        }
    }
    out
}
pub fn guided(
    im: &Image,
    now: &Image,
    p: P,
    independent: Option<Hit>,
    models: &[Sphere],
    min_ncc: f64,
) -> (Option<Hit>, Audit) {
    let mut best = independent;
    let mut audit = Audit::fail("no_independent_shared_support");
    for &s in models {
        let Some(q) = s.map(p) else { continue };
        let (h, a) = checked(
            im,
            now,
            Warp {
                p,
                q,
                a: IDENTITY,
                sphere: Some(s),
            },
            false,
            "shared_3d",
            min_ncc,
        );
        // The extra geometric prior may rescue a failed match, or replace a
        // match only with a material photometric improvement.
        if let Some(h) = h {
            if best.as_ref().is_none_or(|b| h.ncc > b.ncc + 0.015) {
                best = Some(h);
                audit = a;
            }
        } else if best.is_none() && a.best_ncc.unwrap_or(-1.) > audit.best_ncc.unwrap_or(-2.) {
            audit = a;
        }
    }
    if let Some(h) = &best {
        audit.accepted = true;
        audit.reason = "accepted".into();
        audit.best_ncc = Some(h.ncc);
        audit.fb = Some(h.fb);
        audit.candidate = Some(h.q);
    }
    (best, audit)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn texture(w: usize, h: usize) -> Image {
        Image {
            w,
            h,
            v: (0..w * h)
                .map(|i| {
                    let x = (i % w) as f64;
                    let y = (i / w) as f64;
                    0.5 + 0.11 * (x * 0.19 + y * 0.11).sin()
                        + 0.09 * (x * 0.37 - y * 0.21).cos()
                        + 0.07 * (x * 0.07 + y * 0.43).sin()
                })
                .collect(),
        }
    }
    #[test]
    fn affine_alignment_handles_rotation_shear_and_gain() {
        let im = texture(180, 150);
        let p = [85., 72.];
        let q = [90., 69.];
        let a = [[1.04, -0.19], [0.13, 0.95]];
        let inv = inverse(a).unwrap();
        let now = Image {
            w: im.w,
            h: im.h,
            v: (0..im.w * im.h)
                .map(|i| {
                    let d = mul(
                        inv,
                        [i as f64 % im.w as f64 - q[0], (i / im.w) as f64 - q[1]],
                    );
                    im.sample([p[0] + d[0], p[1] + d[1]]).unwrap_or(0.5) * 1.15 + 0.02
                })
                .collect(),
        };
        let (hit, audit) = checked(&im, &now, Warp::plain(p, q), true, "affine", 0.86);
        let h = hit.unwrap_or_else(|| panic!("{audit:?}"));
        assert!(distance(h.q, q) < 0.45, "{h:?}");
        assert!((h.a[0][1] - a[0][1]).abs() < 0.07, "{h:?}");
    }
    #[test]
    fn blank_and_disappearing_patch_are_not_matches() {
        let a = texture(120, 120);
        let b = Image {
            w: 120,
            h: 120,
            v: vec![0.5; 14400],
        };
        assert!(independent(&a, &b, [60., 60.], [60., 60.], true, 0.55)
            .0
            .is_none());
    }
    #[test]
    fn exact_spherical_rotation_and_inverse_agree() {
        let s = Sphere {
            center: [180., 130.],
            radius: 180.,
            shift: [2., -3.],
            omega: [0.03, -0.045, 0.02],
            support: 0,
            residual: 0.,
        };
        let p = [240., 165.];
        let q = s.map(p).unwrap();
        assert!(distance(s.reversed([0., 0.]).map(q).unwrap(), p) < 1e-8);
    }
    #[test]
    fn shared_pivot_predicts_held_points_under_out_of_plane_rotation() {
        let truth = Sphere {
            center: [210., 154.],
            radius: 231.,
            shift: [1.4, -0.8],
            omega: [0.025, -0.035, 0.012],
            support: 0,
            residual: 0.,
        };
        let mut pairs = Vec::new();
        for y in 0..5 {
            for x in 0..7 {
                let p = [85. + x as f64 * 35., 65. + y as f64 * 35.];
                pairs.push(Pair {
                    id: pairs.len(),
                    p,
                    q: truth.map(p).unwrap(),
                });
            }
        }
        let models = sphere_models(&pairs, 420, 280, 1);
        assert!(!models.is_empty());
        let error = pairs
            .iter()
            .filter(|p| p.id % 2 == 1)
            .map(|p| {
                models
                    .iter()
                    .filter_map(|s| s.map(p.p))
                    .map(|q| distance(q, p.q))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum::<f64>()
            / 17.;
        assert!(error < 0.2, "{error}");
        for p in &mut pairs {
            if p.id % 2 == 1 {
                p.q = [-1000., -1000.];
            }
        }
        let repeat = sphere_models(&pairs, 420, 280, 1);
        assert_eq!(
            serde_json::to_string(&models).unwrap(),
            serde_json::to_string(&repeat).unwrap()
        );
    }
    #[test]
    fn compact_reflection_cannot_supply_a_pivot() {
        let v: Vec<_> = (0..30)
            .map(|id| Pair {
                id,
                p: [190. + (id % 6) as f64 * 4., 120. + (id / 6) as f64 * 3.],
                q: [192. + (id % 6) as f64 * 4., 121. + (id / 6) as f64 * 3.],
            })
            .collect();
        assert!(sphere_models(&v, 420, 280, 1).is_empty());
    }
    #[test]
    fn image_match_must_still_agree_with_the_shared_rotation() {
        let im = texture(180, 150);
        let now = Image {
            w: 180,
            h: 150,
            v: (0..180 * 150)
                .map(|i| {
                    im.sample([(i % 180) as f64 - 5., (i / 180) as f64])
                        .unwrap_or(0.5)
                })
                .collect(),
        };
        let p = [85., 72.];
        let s = Sphere {
            center: [90., 75.],
            radius: 150.,
            shift: [0., 0.],
            omega: [0.; 3],
            support: 20,
            residual: 0.1,
        };
        let (hit, audit) = checked(
            &im,
            &now,
            Warp {
                sphere: Some(s),
                ..Warp::plain(p, p)
            },
            false,
            "shared_3d",
            0.65,
        );
        assert!(hit.is_none());
        assert_eq!(audit.reason, "shared_prediction_mismatch");
    }
}
