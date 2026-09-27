//! Conditional nested-circle sign comparison. Diagnostic only, not gaze authority.
//! The observed fitted pupil conic is a single correlated observation; sampling
//! it more densely never increases its information weight. No generated art.
use super::joint::{circle_pose_hypotheses, CirclePoseSeed, PinholeCamera, ProjectedCircle};
use crate::geometry::{dot3, normalized3, Ellipse};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Fit {
    pub rms_px: f64,
    pub center: [f64; 3],
    pub normal: [f64; 3],
    pub radius: f64,
    pub offset: [f64; 2],
    pub ellipse: Ellipse,
}
impl Fit {
    fn json(self) -> Value {
        json!({"rms_px":self.rms_px,"center_per_outer_radius":self.center,
            "normal":self.normal,"radius_per_outer_radius":self.radius,"decentration_per_outer_radius":self.offset,
            "predicted_pupil":{"center":self.ellipse.center,"major_radius":self.ellipse.major_radius,
                "minor_radius":self.ellipse.minor_radius,"angle":self.ellipse.angle}})
    }
}
pub(crate) struct Comparison {
    pub fits: [Fit; 2],
    pub preference: Option<usize>,
    pub report: Value,
}

fn fit(
    camera: PinholeCamera,
    pose: CirclePoseSeed,
    pupil: Ellipse,
    origin: [u32; 2],
    bound: f64,
    depth: f64,
) -> Option<Fit> {
    let n = pose.normal;
    let t = normalized3([n[2], 0., -n[0]])?;
    let b = [
        n[1] * t[2] - n[2] * t[1],
        n[2] * t[0] - n[0] * t[2],
        n[0] * t[1] - n[1] * t[0],
    ];
    let base: [f64; 3] = std::array::from_fn(|j| pose.center_per_radius[j] - depth * n[j]);
    let points = pupil.dense_points(48);
    let mut plane = Vec::with_capacity(points.len());
    for q in &points {
        let ray = camera.unproject([q.0 + origin[0] as f64, q.1 + origin[1] as f64], 1.);
        let denom = dot3(n, ray);
        if denom.abs() < 1e-10 {
            return None;
        }
        let distance = dot3(n, base) / denom;
        if distance <= 0. {
            return None;
        }
        let delta = std::array::from_fn(|j| distance * ray[j] - base[j]);
        plane.push([dot3(delta, t), dot3(delta, b)]);
    }
    // Algebraic circle initialization in the hypothesized iris plane.
    let mean: [f64; 2] =
        std::array::from_fn(|j| plane.iter().map(|p| p[j]).sum::<f64>() / plane.len() as f64);
    let (mut xx, mut xy, mut yy, mut xz, mut yz) = (0., 0., 0., 0., 0.);
    for p in &plane {
        let x = p[0] - mean[0];
        let y = p[1] - mean[1];
        let z = x * x + y * y;
        xx += x * x;
        xy += x * y;
        yy += y * y;
        xz += x * z;
        yz += y * z;
    }
    let det = xx * yy - xy * xy;
    if det <= 1e-12 {
        return None;
    }
    let mut pars = [
        mean[0] + (yy * xz - xy * yz) / (2. * det),
        mean[1] + (xx * yz - xy * xz) / (2. * det),
        0.,
    ];
    pars[2] = plane
        .iter()
        .map(|p| (p[0] - pars[0]).hypot(p[1] - pars[1]))
        .sum::<f64>()
        / plane.len() as f64;
    let constrain = |mut v: [f64; 3]| {
        let len = v[0].hypot(v[1]);
        if len > bound {
            v[0] *= bound / len;
            v[1] *= bound / len;
        }
        v[2] = v[2].clamp(0.08, 0.8_f64.min(1. - v[0].hypot(v[1])));
        v
    };
    let evaluate = |v: [f64; 3]| -> Option<Fit> {
        let center = std::array::from_fn(|j| base[j] + v[0] * t[j] + v[1] * b[j]);
        let conic = ProjectedCircle::project(camera, center, n, v[2], origin)?;
        let cost = (points
            .iter()
            .map(|&q| conic.residual_px(q).powi(2))
            .sum::<f64>()
            / points.len() as f64)
            .sqrt();
        cost.is_finite().then(|| Fit {
            rms_px: cost,
            center,
            normal: n,
            radius: v[2],
            offset: [v[0], v[1]],
            ellipse: conic.ellipse().unwrap(),
        })
    };
    pars = constrain(pars);
    let mut best = evaluate(pars)?;
    // Same bounded image-space refinement budget for both hypotheses.
    let mut step = 0.04;
    for _ in 0..36 {
        let mut improved = false;
        for axis in 0..3 {
            for sign in [-1., 1.] {
                let mut next = pars;
                next[axis] += sign * step;
                next = constrain(next);
                if let Some(f) = evaluate(next) {
                    if f.rms_px < best.rms_px {
                        pars = next;
                        best = f;
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            step *= 0.5;
            if step < 1e-5 {
                break;
            }
        }
    }
    Some(best)
}

pub(crate) fn compare(
    camera: PinholeCamera,
    outer: Ellipse,
    pupil: Ellipse,
    origin: [u32; 2],
) -> Option<Comparison> {
    if ![
        pupil.center.0,
        pupil.center.1,
        pupil.major_radius,
        pupil.minor_radius,
        pupil.angle,
    ]
    .iter()
    .all(|v| v.is_finite())
        || pupil.minor_radius <= 0.
        || pupil.major_radius < pupil.minor_radius
    {
        return None;
    }
    let poses = circle_pose_hypotheses(camera, outer, origin)?;
    // Engineering sensitivity envelope, not a measured anatomical prior.
    let mut variants = vec![];
    let mut nominal = None;
    let mut unanimous = None;
    let mut robust = true;
    for bound in [0., 0.05, 0.10, 0.20] {
        for depth in [0., 0.10, 0.20] {
            let Some(a) = fit(camera, poses[0], pupil, origin, bound, depth) else {
                return None;
            };
            let Some(b) = fit(camera, poses[1], pupil, origin, bound, depth) else {
                return None;
            };
            let fits = [a, b];
            let winner = usize::from(b.rms_px < a.rms_px);
            let margin = fits[1 - winner].rms_px - fits[winner].rms_px;
            let eligible = fits[winner].rms_px <= 2. && margin >= 1.;
            // Exact concentricity is a deliberately brittle baseline, not a veto
            // on legitimate decentration. Require all nonzero bounds/depths to agree.
            if bound > 0. {
                if !eligible || unanimous.is_some_and(|u| u != winner) {
                    robust = false;
                }
                unanimous = Some(winner);
            }
            if bound == 0.10 && depth == 0. {
                nominal = Some(fits);
            }
            variants.push(json!({"max_decentration_outer_radius":bound,"inward_depth_outer_radius":depth,
            "fits":fits.map(Fit::json),"lower_cost_branch":winner,"margin_px":margin,"eligible":eligible}));
        }
    }
    let fits = nominal?;
    let preference = if robust { unanimous } else { None };
    Some(Comparison {
        fits,
        preference,
        report: json!({"status":if preference.is_some(){"conditional-preference"}else{"ambiguous"},
        "preferred_branch":preference,"nominal_fits":fits.map(Fit::json),"variants":variants,
        "thresholds":{"maximum_winner_rms_px":2.,"minimum_margin_px":1.},
        "evidence":"same-exposure fitted pupil conic, not independent raw samples; one shape observation",
        "authority":"diagnostic only; no live sign or calibration authority", "probability":null}),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(distance: f64, tilt: f64) -> (PinholeCamera, Ellipse, Ellipse, [f64; 3]) {
        let c = PinholeCamera {
            focal_px: [1200.; 2],
            principal_px: [400., 300.],
        };
        let n = [tilt.sin(), 0., tilt.cos()];
        let center = [5., 3., -distance];
        let e = |r| {
            ProjectedCircle::project(c, center, n, r, [0, 0])
                .unwrap()
                .ellipse()
                .unwrap()
        };
        (c, e(6.), e(2.5), n)
    }
    #[test]
    fn exact_concentric_perspective_prefers_true_branch_for_both_signs() {
        for tilt in [-0.65, 0.65] {
            let (c, o, p, n) = fixture(70., tilt);
            let poses = circle_pose_hypotheses(c, o, [0, 0]).unwrap();
            let fits = poses.map(|s| fit(c, s, p, [0, 0], 0., 0.).unwrap());
            let truth = usize::from(dot3(poses[1].normal, n) > dot3(poses[0].normal, n));
            assert!(fits[truth].rms_px < 1e-5, "{:?}", fits);
            assert!(fits[1 - truth].rms_px > 0.1, "{:?}", fits);
        }
    }
    #[test]
    fn frontal_and_weak_perspective_abstain() {
        for (distance, tilt) in [(100., 0.), (10000., 0.6)] {
            let (c, o, p, _) = fixture(distance, tilt);
            assert!(compare(c, o, p, [0, 0]).unwrap().preference.is_none());
        }
    }
    #[test]
    fn bounded_decentration_is_profiled_equally_and_does_not_move_outer() {
        let (c, o, mut p, _) = fixture(80., 0.5);
        p.center.0 += 2.;
        p.center.1 -= 1.;
        let poses = circle_pose_hypotheses(c, o, [0, 0]).unwrap();
        for pose in poses {
            let fixed = fit(c, pose, p, [0, 0], 0., 0.).unwrap();
            let free = fit(c, pose, p, [0, 0], 0.2, 0.).unwrap();
            assert!(free.rms_px <= fixed.rms_px + 0.01);
            assert!(free.offset[0].hypot(free.offset[1]) <= 0.2 + 1e-10);
        }
    }
}
