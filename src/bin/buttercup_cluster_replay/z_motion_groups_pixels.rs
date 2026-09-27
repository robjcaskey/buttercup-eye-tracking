//! Full-frame patch support. Boundary neighborhoods are one-sided and contain
//! real source pixels only; no padding, extrapolation or anatomical ROI.
use super::{
    motion::{Model, Picture},
    P,
};
#[derive(Clone)]
pub struct Patch {
    pub p: P,
    coords: [P; 9],
    values: [[f64; 9]; 3],
}
pub fn patches(a: &Picture, stride: usize) -> Vec<Patch> {
    let (w, h) = (a[0].w, a[0].h);
    assert!(w >= 3 && h >= 3 && stride > 0);
    let axis = |n: usize| {
        let mut v = (0..n).step_by(stride).collect::<Vec<_>>();
        if *v.last().unwrap() != n - 1 {
            v.push(n - 1);
        }
        v
    };
    let mut out = Vec::new();
    for y in axis(h) {
        for x in axis(w) {
            let sx = x.saturating_sub(1).min(w - 3);
            let sy = y.saturating_sub(1).min(h - 3);
            let coords = std::array::from_fn(|k| [(sx + k % 3) as f64, (sy + k / 3) as f64]);
            let values: [[f64; 9]; 3] = std::array::from_fn(|c| {
                std::array::from_fn(|k| a[c].v[(sy + k / 3) * w + sx + k % 3])
            });
            let textured = values
                .iter()
                .filter(|vs| {
                    let m = vs.iter().sum::<f64>() / 9.;
                    vs.iter().map(|v| (v - m).powi(2)).sum::<f64>() > 1e-6
                })
                .count()
                >= 2;
            if textured {
                out.push(Patch {
                    p: [x as f64, y as f64],
                    coords,
                    values,
                });
            }
        }
    }
    out
}
fn sample(im: &super::super::Image, p: P) -> Option<f64> {
    let (w, h) = (im.w, im.h);
    if !p.iter().all(|v| v.is_finite())
        || p[0] < -1e-9
        || p[1] < -1e-9
        || p[0] > (w - 1) as f64 + 1e-9
        || p[1] > (h - 1) as f64 + 1e-9
    {
        return None;
    }
    let px = p[0].clamp(0., (w - 1) as f64);
    let py = p[1].clamp(0., (h - 1) as f64);
    let x = (px as usize).min(w - 2);
    let y = (py as usize).min(h - 2);
    let u = px - x as f64;
    let v = py - y as f64;
    Some(
        (1. - v) * ((1. - u) * im.v[y * w + x] + u * im.v[y * w + x + 1])
            + v * ((1. - u) * im.v[(y + 1) * w + x] + u * im.v[(y + 1) * w + x + 1]),
    )
}
pub fn error(p: &Patch, b: &Picture, m: Model) -> f64 {
    error_mask(p, b, m, [true; 9])
}
// Compare models on the same observed source samples. Losing image overlap
// cannot itself create a confident motion-group preference.
pub fn paired_error(p: &Patch, b: &Picture, a: Model, c: Model) -> Option<[f64; 2]> {
    let models = [a.prepare(), c.prepare()];
    let mask = std::array::from_fn(|k| {
        models
            .iter()
            .all(|m| m.map(p.coords[k]).and_then(|q| sample(&b[0], q)).is_some())
    });
    if mask.iter().filter(|&&x| x).count() < 5 {
        return None;
    }
    Some([error_mask(p, b, a, mask), error_mask(p, b, c, mask)])
}
fn error_mask(p: &Patch, b: &Picture, m: Model, mask: [bool; 9]) -> f64 {
    let m = m.prepare();
    let mut xs = [[0.; 9]; 3];
    let mut ys = [[0.; 9]; 3];
    let mut n = 0;
    for k in 0..9 {
        if !mask[k] {
            continue;
        }
        let Some(q) = m.map(p.coords[k]) else {
            continue;
        };
        let Some(y) = (0..3).map(|c| sample(&b[c], q)).collect::<Option<Vec<_>>>() else {
            continue;
        };
        for c in 0..3 {
            xs[c][n] = p.values[c][k];
            ys[c][n] = y[c];
        }
        n += 1;
    }
    if n < 5 {
        return 1.;
    }
    let mut relative = 0.;
    let mut ncc = 0.;
    let mut channels = 0;
    for c in 0..3 {
        let mx = xs[c][..n].iter().sum::<f64>() / n as f64;
        let my = ys[c][..n].iter().sum::<f64>() / n as f64;
        let (mut vx, mut vy, mut xy) = (0., 0., 0.);
        for k in 0..n {
            let x = xs[c][k] - mx;
            let y = ys[c][k] - my;
            vx += x * x;
            vy += y * y;
            xy += x * y;
            let delta = (xs[c][k] - ys[c][k]).abs();
            relative += if delta < 0.5 {
                delta * delta
            } else {
                delta - 0.25
            };
        }
        if vx > 1e-6 && vy > 1e-6 {
            ncc += (1. - (xy / (vx * vy).sqrt()).clamp(-1., 1.)).min(1.);
            channels += 1;
        }
    }
    if channels < 2 {
        return 1.;
    }
    0.6 * (relative / (3 * n) as f64).min(1.)
        + 0.4 * ncc / channels as f64
        + 0.1 * (9 - n) as f64 / 9.
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn competing_models_use_identical_visible_samples() {
        let im: Picture = std::array::from_fn(|c| super::super::super::Image {
            w: 18,
            h: 12,
            v: (0..216)
                .map(|k| ((k * 17 + c * 7) % 43) as f64 / 43.)
                .collect(),
        });
        let ps = patches(&im, 1);
        let p = ps.iter().find(|p| p.p == [17., 5.]).unwrap();
        let a = Model::new(18, 12, [0., 0.]);
        let mut b = a;
        b.translation[0] = 0.3;
        assert!(error(p, &im, a) < 1e-12);
        let scores = paired_error(p, &im, a, b).unwrap();
        assert!((scores[0] - 0.1 / 3.).abs() < 1e-12);
        b.translation[0] = 10.;
        assert!(paired_error(p, &im, a, b).is_none());
    }
    #[test]
    fn full_image_includes_corners_and_identity_matches() {
        let im: Picture = std::array::from_fn(|c| super::super::super::Image {
            w: 18,
            h: 12,
            v: (0..216)
                .map(|k| ((k * 17 + c * 7) % 43) as f64 / 43.)
                .collect(),
        });
        let ps = patches(&im, 1);
        assert_eq!(ps.len(), 216);
        for q in [[0., 0.], [17., 0.], [0., 11.], [17., 11.]] {
            let p = ps.iter().find(|p| p.p == q).unwrap();
            assert!(error(p, &im, Model::new(18, 12, [0., 0.])) < 1e-12);
        }
        assert!(sample(&im[0], [-0.01, 0.]).is_none());
        assert!(sample(&im[0], [18., 11.]).is_none());
    }
}
