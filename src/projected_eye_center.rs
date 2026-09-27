//! Robust intersection of projected normal lines over source time.
//! Independent implementation of line least squares, with an optional linear
//! centre motion model. Support radii are engineering estimates, not probabilities.
//!
//! Experiment context: Jason Orlosky's public eye-tracking tutorial and the
//! projected-normal construction in Swirski and Dodgson (PETMEI 2013).
//! No external tracker implementation is incorporated. The corpus adapter uses
//! Buttercup's native circle unprojection and excludes each current ellipse
//! from its own centre fit. This module is an offline candidate, not a live
//! sign authority; image-centre motion and poor conics can invalidate its model.
#[derive(Clone, Copy, Debug)]
pub struct NormalLine {
    pub point: [f64; 2],
    pub direction: [f64; 2],
    pub seconds: f64,
    pub weight: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct CenterFit {
    pub center: [f64; 2],
    pub velocity: [f64; 2],
    pub support_radius_px: f64,
    pub rms_px: f64,
    pub median_px: f64,
    pub condition: f64,
    pub observations: usize,
}
fn inverse(mut a: [[f64; 4]; 4], n: usize) -> Option<[[f64; 4]; 4]> {
    let mut b = [[0.; 4]; 4];
    for i in 0..n {
        b[i][i] = 1.;
    }
    for k in 0..n {
        let pivot = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[pivot][k].abs() < 1e-10 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let v = a[k][k];
        for j in 0..n {
            a[k][j] /= v;
            b[k][j] /= v;
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let v = a[i][k];
            for j in 0..n {
                a[i][j] -= v * a[k][j];
                b[i][j] -= v * b[k][j];
            }
        }
    }
    Some(b)
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
/// `at` is the prediction time. Callers must exclude the scored observation.
pub fn fit(lines: &[NormalLine], at: f64, moving: bool) -> Option<CenterFit> {
    let n = if moving { 4 } else { 2 };
    if lines.len() < if moving { 12 } else { 8 } {
        return None;
    }
    let first = lines
        .iter()
        .map(|l| l.seconds)
        .fold(f64::INFINITY, f64::min);
    let last = lines
        .iter()
        .map(|l| l.seconds)
        .fold(f64::NEG_INFINITY, f64::max);
    if last - first < 0.5 || at - last > 0.5 || at < last {
        return None;
    }
    let origin = lines.last()?.point;
    let rows = lines
        .iter()
        .map(|l| {
            let len = l.direction[0].hypot(l.direction[1]);
            let u = [-l.direction[1] / len, l.direction[0] / len];
            let t = l.seconds - at;
            (
                [u[0], u[1], u[0] * t, u[1] * t],
                u[0] * (l.point[0] - origin[0]) + u[1] * (l.point[1] - origin[1]),
            )
        })
        .collect::<Vec<_>>();
    if rows
        .iter()
        .any(|(a, b)| !b.is_finite() || a.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    let mut weights = lines
        .iter()
        .map(|l| l.weight.clamp(0.01, 1.))
        .collect::<Vec<_>>();
    let mut x = [0.; 4];
    let mut cov = [[0.; 4]; 4];
    let mut condition = 0.;
    for _ in 0..7 {
        let mut a = [[0.; 4]; 4];
        let mut b = [0.; 4];
        for ((row, y), &w) in rows.iter().zip(&weights) {
            for i in 0..n {
                b[i] += w * row[i] * y;
                for j in 0..n {
                    a[i][j] += w * row[i] * row[j];
                }
            }
        }
        cov = inverse(a, n)?;
        let infnorm = |m: [[f64; 4]; 4]| {
            (0..n)
                .map(|i| m[i][..n].iter().map(|v| v.abs()).sum::<f64>())
                .fold(0., f64::max)
        };
        condition = infnorm(a) * infnorm(cov);
        if !condition.is_finite() || condition > 1e6 {
            return None;
        }
        x = std::array::from_fn(|i| (0..n).map(|j| cov[i][j] * b[j]).sum());
        let errors = rows
            .iter()
            .map(|(a, b)| ((0..n).map(|i| a[i] * x[i]).sum::<f64>() - b).abs())
            .collect::<Vec<_>>();
        let threshold = (2.5 * median(errors.clone())).max(2.);
        for i in 0..weights.len() {
            weights[i] = lines[i].weight.clamp(0.01, 1.) * (threshold / errors[i].max(threshold));
        }
    }
    let errors = rows
        .iter()
        .map(|(a, b)| ((0..n).map(|i| a[i] * x[i]).sum::<f64>() - b).abs())
        .collect::<Vec<_>>();
    let rms = (errors
        .iter()
        .zip(&weights)
        .map(|(e, w)| w * e * e)
        .sum::<f64>()
        / weights.iter().sum::<f64>())
    .sqrt();
    let eigen_max = (cov[0][0] + cov[1][1] + (cov[0][0] - cov[1][1]).hypot(2. * cov[0][1])) / 2.;
    let support = 2. + 2.5 * rms.max(1.) * eigen_max.max(0.).sqrt();
    Some(CenterFit {
        center: [origin[0] + x[0], origin[1] + x[1]],
        velocity: if moving { [x[2], x[3]] } else { [0., 0.] },
        support_radius_px: support,
        rms_px: rms,
        median_px: median(errors),
        condition,
        observations: lines.len(),
    })
}
