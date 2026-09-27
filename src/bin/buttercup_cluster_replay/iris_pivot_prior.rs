//! Explicit, defeasible priors for a small-motion iris experiment. These are
//! search bounds in reference-image units, not measured anatomy or confidence.
use super::geometry::{project, rotation, score, Affine, Features, Region, Samples, Score};
use super::P;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MotionPrior {
    pub angle_limit_deg: [f64; 3],
    pub angle_step_deg: f64,
    pub pivot_depth_radii: [f64; 2],
    pub pivot_offset_radii: f64,
    pub translation_radii: f64,
    pub radial_reference_plane: bool,
    pub weight: f64,
}
impl Default for MotionPrior {
    fn default() -> Self {
        Self {
            angle_limit_deg: [12., 12., 3.],
            angle_step_deg: 0.5,
            pivot_depth_radii: [1.3, 2.5],
            pivot_offset_radii: 0.6,
            translation_radii: 0.04,
            radial_reference_plane: true,
            weight: 0.02,
        }
    }
}
impl MotionPrior {
    pub fn valid(&self) -> bool {
        self.angle_limit_deg
            .iter()
            .all(|x| x.is_finite() && (1. ..=20.).contains(x))
            && self.angle_limit_deg[2] <= 6.
            && (0.25..=2.).contains(&self.angle_step_deg)
            && self
                .pivot_depth_radii
                .iter()
                .all(|x| x.is_finite() && (0.8..=3.).contains(x))
            && self.pivot_depth_radii[1] - self.pivot_depth_radii[0] >= 0.2
            && (0.1..=0.9).contains(&self.pivot_offset_radii)
            && (0. ..=0.08).contains(&self.translation_radii)
            && (0. ..=0.1).contains(&self.weight)
            && self.angle_counts().iter().product::<usize>() <= 100_000
    }
    fn angle_counts(&self) -> [usize; 3] {
        self.angle_limit_deg
            .map(|x| 2 * (x / self.angle_step_deg).floor() as usize + 1)
    }
    pub fn angles(&self) -> Vec<[f64; 3]> {
        let counts = self.angle_counts();
        let mut out = Vec::new();
        for x in 0..counts[0] {
            for y in 0..counts[1] {
                for z in 0..counts[2] {
                    let ids = [x, y, z];
                    out.push(std::array::from_fn(|k| {
                        (ids[k] as f64 - (counts[k] / 2) as f64) * self.angle_step_deg.to_radians()
                    }));
                }
            }
        }
        out
    }
}

pub struct Search<'a> {
    pub prior: &'a MotionPrior,
    pub roi: Region,
}
#[derive(Clone, Copy)]
pub struct Fit {
    pub omega: [f64; 3],
    pub translation: P,
    pub score: Score,
    pub objective: f64,
}
impl Search<'_> {
    pub fn radius(&self) -> f64 {
        self.roi.radii[0].max(self.roi.radii[1])
    }
    pub fn pivots(&self) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        let d = self.prior.pivot_depth_radii;
        let r = self.radius();
        for z in 0..5 {
            for y in -2..=2 {
                for x in -2..=2 {
                    out.push([
                        self.roi.center[0] + x as f64 * self.prior.pivot_offset_radii * r / 2.,
                        self.roi.center[1] + y as f64 * self.prior.pivot_offset_radii * r / 2.,
                        (d[0] + z as f64 * (d[1] - d[0]) / 4.) * r,
                    ]);
                }
            }
        }
        out
    }
    pub fn contains_pivot(&self, c: [f64; 3]) -> bool {
        let r = self.radius();
        (0..2)
            .all(|k| (c[k] - self.roi.center[k]).abs() <= self.prior.pivot_offset_radii * r + 1e-9)
            && c[2] >= self.prior.pivot_depth_radii[0] * r - 1e-9
            && c[2] <= self.prior.pivot_depth_radii[1] * r + 1e-9
    }
    pub fn gradient(&self, c: [f64; 3]) -> P {
        if self.prior.radial_reference_plane {
            // Reference iris plane perpendicular to the pivot-to-iris-center
            // direction. Each image point is lifted onto that tilted plane.
            std::array::from_fn(|k| (self.roi.center[k] - c[k]) / c[2])
        } else {
            [0., 0.]
        }
    }
    pub fn warp(&self, c: [f64; 3], w: [f64; 3], delta: P, t: P) -> Affine {
        let mut m = project(c, w, delta);
        let r = rotation(w);
        let gradient = self.gradient(c);
        for i in 0..2 {
            for (j, slope) in gradient.iter().enumerate() {
                m.a[i][j] += r[i][2] * slope;
                m.t[i] -= r[i][2] * slope * self.roi.center[j];
            }
            m.t[i] += t[i];
        }
        m
    }
    pub(super) fn motion_penalty(&self, w: [f64; 3], t: P) -> f64 {
        let a: [f64; 3] =
            std::array::from_fn(|k| w[k] / self.prior.angle_limit_deg[k].to_radians());
        let bound = self.prior.translation_radii * self.radius();
        self.prior.weight
            * (0.2 * (a[0] * a[0] + a[1] * a[1])
                + 0.5 * a[2] * a[2]
                + if bound > 0. {
                    0.5 * (t[0] * t[0] + t[1] * t[1]) / bound.powi(2)
                } else {
                    0.
                })
    }
    pub fn pivot_penalty(&self, c: [f64; 3]) -> f64 {
        let r = self.radius();
        let d = self.prior.pivot_depth_radii;
        let z = (c[2] / r - (d[0] + d[1]) / 2.) / ((d[1] - d[0]) / 2.);
        let xy = (0..2)
            .map(|k| ((c[k] - self.roi.center[k]) / (self.prior.pivot_offset_radii * r)).powi(2))
            .sum::<f64>();
        self.prior.weight * 0.2 * (xy + z * z)
    }
    pub(super) fn feasible(&self, w: [f64; 3], t: P) -> bool {
        (0..3).all(|k| w[k].abs() <= self.prior.angle_limit_deg[k].to_radians() + 1e-12)
            && t[0].hypot(t[1]) <= self.prior.translation_radii * self.radius() + 1e-12
    }
    pub fn at_limit(&self, w: [f64; 3], t: P) -> bool {
        (0..3).any(|k| {
            w[k].abs() >= self.prior.angle_limit_deg[k].to_radians() - 0.05f64.to_radians()
        }) || (self.prior.translation_radii > 0.
            && t[0].hypot(t[1]) >= self.prior.translation_radii * self.radius() - 0.1)
    }
    pub(super) fn evaluate(
        &self,
        s: &Samples,
        f: &Features,
        c: [f64; 3],
        delta: P,
        w: [f64; 3],
        t: P,
    ) -> Fit {
        let score = score(s, f, self.warp(c, w, delta, t));
        Fit {
            omega: w,
            translation: t,
            score,
            objective: score.loss + self.motion_penalty(w, t),
        }
    }
    pub(super) fn refine(
        &self,
        s: &Samples,
        f: &Features,
        c: [f64; 3],
        delta: P,
        w: [f64; 3],
        t: P,
    ) -> Fit {
        self.refine_scored(c, delta, w, t, |m| score(s, f, m))
    }
    pub(super) fn refine_scored(
        &self,
        c: [f64; 3],
        delta: P,
        w: [f64; 3],
        t: P,
        score: impl Fn(Affine) -> Score,
    ) -> Fit {
        let evaluate = |w, t| {
            let score = score(self.warp(c, w, delta, t));
            Fit {
                omega: w,
                translation: t,
                objective: score.loss + self.motion_penalty(w, t),
                score,
            }
        };
        let mut best = evaluate(w, t);
        for step in [0.5f64, 0.25, 0.125, 0.0625, 0.03125] {
            for _ in 0..10 {
                let old = best.objective;
                for axis in 0..if self.prior.translation_radii > 0. {
                    5
                } else {
                    3
                } {
                    for sign in [-1., 1.] {
                        let mut w = best.omega;
                        let mut t = best.translation;
                        if axis < 3 {
                            w[axis] += sign * step.to_radians();
                        } else {
                            t[axis - 3] += sign * step * 2.;
                        }
                        if !self.feasible(w, t) {
                            continue;
                        }
                        let q = evaluate(w, t);
                        if q.objective + 1e-9 < best.objective {
                            best = q;
                        }
                    }
                }
                if old - best.objective < 1e-9 {
                    break;
                }
            }
        }
        best
    }
    pub fn fit(
        &self,
        s: &Samples,
        f: &Features,
        c: [f64; 3],
        delta: P,
        angles: &[[f64; 3]],
        start: Option<([f64; 3], P)>,
    ) -> Fit {
        if let Some((w, t)) = start {
            return self.refine(s, f, c, delta, w, t);
        }
        let coarse = s.coarse(128);
        let mut starts: Vec<Fit> = Vec::new();
        for &w in angles {
            let q = self.evaluate(&coarse, f, c, delta, w, [0., 0.]);
            if starts.len() < 8 || q.objective < starts.last().unwrap().objective {
                starts.push(q);
                starts.sort_by(|a, b| a.objective.total_cmp(&b.objective));
                starts.truncate(8);
            }
        }
        starts
            .into_iter()
            .map(|p| self.refine(s, f, c, delta, p.omega, p.translation))
            .min_by(|a, b| a.objective.total_cmp(&b.objective))
            .unwrap()
    }
}

pub fn rotation_difference_degrees(a: [f64; 3], b: [f64; 3]) -> f64 {
    let a = rotation(a);
    let b = rotation(b);
    let trace = (0..3)
        .flat_map(|i| (0..3).map(move |j| a[i][j] * b[i][j]))
        .sum::<f64>();
    ((trace - 1.) / 2.).clamp(-1., 1.).acos().to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tilted_plane_projection_matches_explicit_3d_rotation() {
        let prior = MotionPrior::default();
        let search = Search {
            prior: &prior,
            roi: Region {
                center: [190., 130.],
                radii: [90., 80.],
                inner: 0.48,
                outer: 0.84,
            },
        };
        let c = [220., 100., 180.];
        let w = [0.08, -0.11, 0.02];
        let p = [240., 150.];
        let t = [1.2, -0.6];
        let delta = [24., 24.];
        let g = search.gradient(c);
        let q = [p[0], p[1], g[0] * (p[0] - 190.) + g[1] * (p[1] - 130.)];
        let r = rotation(w);
        let got = search.warp(c, w, delta, t).map(p);
        for i in 0..2 {
            let exact =
                c[i] + (0..3).map(|j| r[i][j] * (q[j] - c[j])).sum::<f64>() + delta[i] + t[i];
            assert!((got[i] - exact).abs() < 1e-10);
        }
        assert!(
            super::super::distance(search.warp(c, [0.; 3], [0.; 2], [0.; 2]).map(p), p) < 1e-10
        );
    }
    #[test]
    fn hard_search_bounds_and_translation_ball_are_enforced() {
        let prior = MotionPrior::default();
        assert!(prior.valid());
        let search = Search {
            prior: &prior,
            roi: Region {
                center: [190., 130.],
                radii: [90., 80.],
                inner: 0.48,
                outer: 0.84,
            },
        };
        assert!(search.pivots().iter().all(|&c| search.contains_pivot(c)));
        assert!(!search.contains_pivot([190., 130., 45.]));
        assert!(!search.feasible([13f64.to_radians(), 0., 0.], [0.; 2]));
        assert!(!search.feasible([0.; 3], [3., 3.]));
        assert!(search.feasible([0.; 3], [2., 2.]));
        assert!(
            (rotation_difference_degrees([0.; 3], [0., 0., 0.1]) - 0.1f64.to_degrees()).abs()
                < 1e-9
        );
    }
}
