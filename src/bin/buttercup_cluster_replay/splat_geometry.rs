//! Conditional orthographic sphere and tangent Gaussian arithmetic.
//! The sphere is a declared prior, not a recovered anatomical measurement.
use serde::{Deserialize, Serialize};
pub type V3 = [f64; 3];
pub const CENTER: [f64; 2] = [210., 140.];
pub const RADIUS: f64 = 240.;
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Pose {
    pub angle: V3,
    pub translation: [f64; 2],
}
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub a: [f64; 2],
    pub b: [f64; 2],
}
pub fn dot(a: V3, b: V3) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn normalized(a: V3) -> V3 {
    let n = dot(a, a).sqrt();
    a.map(|x| x / n)
}
pub fn rotate(v: V3, angle: V3) -> V3 {
    let theta = dot(angle, angle).sqrt();
    if theta < 1e-12 {
        return v;
    }
    let n = angle.map(|a| a / theta);
    let d = dot(n, v);
    let c = cross(n, v);
    std::array::from_fn(|i| v[i] * theta.cos() + c[i] * theta.sin() + n[i] * d * (1. - theta.cos()))
}
pub fn lift(p: [f64; 2], center: [f64; 2], radius: f64) -> Option<V3> {
    let x = (p[0] - center[0]) / radius;
    let y = (p[1] - center[1]) / radius;
    let z = 1. - x * x - y * y;
    (z > 0.09).then(|| [x, y, z.sqrt()])
}
pub fn project(v: V3, pose: Pose, radius: f64) -> Option<[f64; 2]> {
    let v = rotate(v, pose.angle);
    (v[2] > 0.15).then(|| {
        [
            CENTER[0] + pose.translation[0] + radius * v[0],
            CENTER[1] + pose.translation[1] + radius * v[1],
        ]
    })
}
pub fn unproject(p: [f64; 2], pose: Pose, radius: f64) -> Option<V3> {
    let v = lift(
        p,
        [
            CENTER[0] + pose.translation[0],
            CENTER[1] + pose.translation[1],
        ],
        radius,
    )?;
    let v = rotate(v, pose.angle.map(|a| -a));
    (v[2] > 0.15).then_some(v)
}
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for k in 0..N {
        let pivot = (k..N).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[pivot][k].abs() < 1e-10 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let d = a[k][k];
        for j in k..N {
            a[k][j] /= d;
        }
        b[k] /= d;
        for i in 0..N {
            if i != k {
                let f = a[i][k];
                for j in k..N {
                    a[i][j] -= f * a[k][j];
                }
                b[i] -= f * b[k];
            }
        }
    }
    b.iter().all(|v| v.is_finite()).then_some(b)
}
pub fn fit(pairs: &[Pair], radius: f64) -> Option<Pose> {
    if pairs.len() < 8 {
        return None;
    }
    let mut pose = Pose::default();
    for k in 0..2 {
        let mut t = pairs.iter().map(|p| p.b[k] - p.a[k]).collect::<Vec<_>>();
        t.sort_by(f64::total_cmp);
        pose.translation[k] = t[t.len() / 2];
    }
    for _ in 0..25 {
        let mut normal = [[0.; 5]; 5];
        let mut rhs = [0.; 5];
        let mut usable = 0;
        for p in pairs {
            let Some(v) = lift(p.a, CENTER, radius) else {
                continue;
            };
            let Some(q) = project(v, pose, radius) else {
                continue;
            };
            let residual = [p.b[0] - q[0], p.b[1] - q[1]];
            let weight = (2. / residual[0].hypot(residual[1]).max(2.)).powi(2);
            let mut j = [[0.; 5]; 2];
            for k in 0..3 {
                let mut changed = pose;
                changed.angle[k] += 1e-5;
                let z = project(v, changed, radius)?;
                for axis in 0..2 {
                    j[axis][k] = (z[axis] - q[axis]) / 1e-5;
                }
            }
            j[0][3] = 1.;
            j[1][4] = 1.;
            for axis in 0..2 {
                for k in 0..5 {
                    rhs[k] += weight * j[axis][k] * residual[axis];
                    for l in 0..5 {
                        normal[k][l] += weight * j[axis][k] * j[axis][l];
                    }
                }
            }
            usable += 1;
        }
        if usable < 8 {
            return None;
        }
        // Small numerical damping only; no anatomical angle target or gaze input.
        for (k, row) in normal.iter_mut().enumerate() {
            row[k] += 1e-5;
        }
        let step = solve(normal, rhs)?;
        for k in 0..3 {
            pose.angle[k] += step[k].clamp(-0.025, 0.025);
        }
        for k in 0..2 {
            pose.translation[k] += step[k + 3].clamp(-5., 5.);
        }
        if dot(pose.angle, pose.angle) > 0.7 * 0.7
            || pose.translation.iter().any(|x| x.abs() > 120.)
        {
            return None;
        }
        if step.iter().map(|x| x * x).sum::<f64>() < 1e-10 {
            break;
        }
    }
    Some(pose)
}
pub fn error(pair: Pair, pose: Pose, radius: f64) -> Option<f64> {
    let q = project(lift(pair.a, CENTER, radius)?, pose, radius)?;
    Some((q[0] - pair.b[0]).hypot(q[1] - pair.b[1]))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Splat {
    pub mean: V3,
    pub tangent_u: V3,
    pub tangent_v: V3,
    pub sigma: f64,
    pub rgb: [f64; 3],
    pub observations: usize,
}
impl Splat {
    pub fn new(mean: V3, rgb: [f64; 3], sigma: f64) -> Self {
        let u = normalized(cross([0., 1., 0.], mean));
        let v = cross(mean, u);
        Self {
            mean,
            tangent_u: u,
            tangent_v: v,
            sigma,
            rgb,
            observations: 1,
        }
    }
}
/// Project the full tangent covariance, preserving foreshortening. This is
/// Gaussian surface splatting on a supplied sphere, not unconstrained 3DGS.
pub fn footprint(s: &Splat, pose: Pose, radius: f64) -> Option<([f64; 2], [f64; 3])> {
    let p = project(s.mean, pose, radius)?;
    let u = rotate(s.tangent_u, pose.angle);
    let v = rotate(s.tangent_v, pose.angle);
    let k = (s.sigma * radius).powi(2);
    Some((
        p,
        [
            k * (u[0] * u[0] + v[0] * v[0]) + 0.16,
            k * (u[0] * u[1] + v[0] * v[1]),
            k * (u[1] * u[1] + v[1] * v[1]) + 0.16,
        ],
    ))
}
pub fn render(
    splats: &[Splat],
    pose: Pose,
    radius: f64,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<f64>) {
    let mut sum = vec![[0.; 3]; w * h];
    let mut mass = vec![0.; w * h];
    // All admitted elements lie on one visible convex sheet, so normalized
    // Gaussian surface fusion suffices. This is not volumetric alpha tracing.
    for s in splats {
        let Some((p, cov)) = footprint(s, pose, radius) else {
            continue;
        };
        let det = cov[0] * cov[2] - cov[1] * cov[1];
        if det <= 0. {
            continue;
        }
        let rx = 3. * cov[0].sqrt();
        let ry = 3. * cov[2].sqrt();
        for y in ((p[1] - ry).floor() as i32).max(0)..=((p[1] + ry).ceil() as i32).min(h as i32 - 1)
        {
            for x in
                ((p[0] - rx).floor() as i32).max(0)..=((p[0] + rx).ceil() as i32).min(w as i32 - 1)
            {
                let dx = x as f64 - p[0];
                let dy = y as f64 - p[1];
                let d = (cov[2] * dx * dx - 2. * cov[1] * dx * dy + cov[0] * dy * dy) / det;
                if d > 9. {
                    continue;
                }
                let weight = (-0.5 * d).exp();
                let j = y as usize * w + x as usize;
                mass[j] += weight;
                for k in 0..3 {
                    sum[j][k] += weight * s.rgb[k];
                }
            }
        }
    }
    let mut bgra = vec![0; w * h * 4];
    for j in 0..w * h {
        let background = if ((j % w) / 12 + (j / w) / 12) % 2 == 0 {
            23
        } else {
            31
        };
        for k in 0..3 {
            bgra[4 * j + 2 - k] = if mass[j] >= 0.3 {
                (sum[j][k] / mass[j]).round().clamp(0., 255.) as u8
            } else {
                background
            };
        }
        bgra[4 * j + 3] = 255;
    }
    (bgra, mass)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_sphere_motion_and_inverse() {
        let pose = Pose {
            angle: [0.035, -0.045, 0.012],
            translation: [6., -3.],
        };
        let pairs = (0..7)
            .flat_map(|y| (0..9).map(move |x| [55. + x as f64 * 38., 50. + y as f64 * 30.]))
            .map(|a| Pair {
                a,
                b: project(lift(a, CENTER, RADIUS).unwrap(), pose, RADIUS).unwrap(),
            })
            .collect::<Vec<_>>();
        let fitted = fit(&pairs, RADIUS).unwrap();
        assert!(pairs
            .iter()
            .all(|&p| error(p, fitted, RADIUS).unwrap() < 1e-4));
        for p in pairs {
            let a = unproject(p.b, pose, RADIUS).unwrap();
            let b = lift(p.a, CENTER, RADIUS).unwrap();
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-10));
        }
    }
    #[test]
    fn sphere_fit_resists_outliers_and_refuses_empty() {
        assert!(fit(&[], RADIUS).is_none());
        let pose = Pose {
            angle: [0.02, -0.035, 0.007],
            translation: [4., 2.],
        };
        let mut pairs = (0..6)
            .flat_map(|y| (0..8).map(move |x| [65. + x as f64 * 40., 45. + y as f64 * 36.]))
            .map(|a| Pair {
                a,
                b: project(lift(a, CENTER, RADIUS).unwrap(), pose, RADIUS).unwrap(),
            })
            .collect::<Vec<_>>();
        let clean = pairs.clone();
        for p in pairs.iter_mut().step_by(7) {
            p.b[0] += 30.;
            p.b[1] -= 20.;
        }
        let fitted = fit(&pairs, RADIUS).unwrap();
        assert!(
            clean
                .iter()
                .map(|&p| error(p, fitted, RADIUS).unwrap())
                .sum::<f64>()
                / (clean.len() as f64)
                < 0.3
        );
    }
    #[test]
    fn gaussian_covariance_rotates_and_unobserved_surface_stays_empty() {
        let s = Splat::new([0., 0., 1.], [180., 120., 110.], 1.5 / RADIUS);
        let (_, c) = footprint(&s, Pose::default(), RADIUS).unwrap();
        let (_, d) = footprint(
            &s,
            Pose {
                angle: [0., 0.6, 0.],
                ..Pose::default()
            },
            RADIUS,
        )
        .unwrap();
        assert!(d[0] < c[0]);
        assert!((d[2] - c[2]).abs() < 1e-10);
        let (image, mass) = render(&[s], Pose::default(), RADIUS, 420, 280);
        assert!(mass[140 * 420 + 210] > 0.9);
        assert_eq!(mass[0], 0.);
        assert_eq!(image[4 * (140 * 420 + 210) + 2], 180);
        assert!(project([0., 0., -1.], Pose::default(), RADIUS).is_none());
    }
}
