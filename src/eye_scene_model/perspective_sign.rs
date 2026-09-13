//! Fixed common-source-window dynamic programming; no learned dependencies.
//! Scores and margins are engineering diagnostics, never probabilities.
use std::collections::VecDeque;
pub const DEPTH_RATIOS: [f64; 5] = [1., 1.25, 1.5, 1.75, 2.];
pub const SHAPE_STATES: usize = 7;
const NUISANCE_STATES: usize = SHAPE_STATES * 5;
pub const MAX_STATES: usize = 40 * SHAPE_STATES;
pub const MAX_TRANSITIONS: usize = 860 * SHAPE_STATES;
const MAX_INTERVALS: usize = 12;
const HORIZON_NS: u64 = 1_250_000_000;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    pub stream: u64,
    pub frame: u64,
    pub ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observation(frame: u64, ns: u64, theta: f64) -> Observation {
        let p = Pose {
            normal: [theta.sin(), 0., theta.cos()],
            pivot_px: [0.; 2],
            pivot_grid_px: [[0.; 2]; 5],
            allowance_px: 2.,
            uncertainty: None,
        };
        Observation {
            source: Source {
                stream: 1,
                frame,
                ns,
            },
            fresh: true,
            poses: [p; 2],
            seed: 0,
            transport: None,
            head_rotation: None,
        }
    }
    #[test]
    fn derivative_order_changes_actual_provisional_readout() {
        let mut model = Tracker::default();
        let mut ablated = Tracker::default();
        for frame in 1..=3 {
            let o = observation(
                frame,
                1_000_000_000 + frame * 8_000_000,
                (frame - 1) as f64 * 0.1,
            );
            model.observe(o);
            ablated.observe_order(o, false);
        }
        let mut o = observation(4, 1_032_000_000, 0.19);
        o.poses[1].normal = [0.3_f64.sin(), 0., 0.3_f64.cos()];
        let a = ablated.observe_order(o, false);
        let b = model.observe(o);
        assert_eq!(a.preferred, Some(0), "nearest-normal ablation");
        assert_eq!(
            b.preferred,
            Some(1),
            "constant velocity continuation rather than reversing velocity"
        );
        assert!(a.identified.is_none() && b.identified.is_none());
        assert!(
            b.emitted_kinematics
                .unwrap()
                .acceleration_rad_s2
                .map(norm)
                .unwrap()
                < 1e-5
        );
    }
    #[test]
    fn derivatives_use_irregular_source_midpoints_and_reset_reference_changes() {
        let mut w = VecDeque::new();
        for (i, t) in [0., 0.008, 0.028, 0.128].into_iter().enumerate() {
            // theta=2*t+3*t² gives interval omega=2+6*midpoint.
            w.push_back(observation(
                i as u64 + 1,
                1_000_000_000 + (t * 1e9) as u64,
                2. * t + 3. * t * t,
            ));
        }
        let k = kinematics(&w, 3, 0, None);
        assert!((k.velocity_rad_s[1] - (2. + 6. * 0.078)).abs() < 1e-8);
        assert!((k.acceleration_rad_s2.unwrap()[1] - 6.).abs() < 1e-7);
        assert!(norm(k.jerk_rad_s3.unwrap()) < 1e-5);
        w[3].head_rotation = Some(HeadRotation {
            from: w[2].source,
            to: w[3].source,
            vector_rad: [0.; 3],
        });
        let changed = kinematics(&w, 3, 0, None);
        assert!(changed.head_compensated);
        assert!(changed.acceleration_rad_s2.is_none() && changed.jerk_rad_s3.is_none());
    }
    #[test]
    fn independent_head_rotation_removes_known_normal_motion_and_obeys_reflected_axial_basis() {
        let source = Source {
            stream: 1,
            frame: 1,
            ns: 1,
        };
        let rotation = |v| {
            Some(HeadRotation {
                from: source,
                to: source,
                vector_rad: v,
            })
        };
        let z = rotate(
            [1., 0., 0.],
            rotation([0., 0., std::f64::consts::FRAC_PI_2]),
        );
        assert!((z[1] - 1.).abs() < 1e-12);
        let v = [0.2, 0.4, 0.7];
        let r = [0.1, -0.3, 0.2];
        let original = rotate(v, rotation(r));
        let reflected = rotate([v[0], v[1], -v[2]], rotation([-r[0], -r[1], r[2]]));
        assert!(
            (reflected[0] - original[0]).abs() < 1e-12
                && (reflected[1] - original[1]).abs() < 1e-12
                && (reflected[2] + original[2]).abs() < 1e-12
        );
        let mut w = VecDeque::new();
        for frame in 1..=4 {
            let mut o = observation(
                frame,
                1_000_000_000 + frame * 20_000_000,
                frame as f64 * 0.1,
            );
            if let Some(previous) = w.back() {
                let previous: &Observation = previous;
                o.head_rotation = Some(HeadRotation {
                    from: previous.source,
                    to: o.source,
                    vector_rad: [0., 0.1, 0.],
                });
            }
            w.push_back(o);
        }
        let k = kinematics(&w, 3, 0, None);
        assert!(
            k.head_compensated
                && norm(k.velocity_rad_s) < 1e-9
                && norm(k.acceleration_rad_s2.unwrap()) < 1e-9
        );
    }
    #[test]
    fn shared_state_derivatives_follow_latent_conic_not_nominal_output() {
        let mut w = VecDeque::new();
        for (i, t) in [0., 0.008, 0.028, 0.128].into_iter().enumerate() {
            let mut o = observation(i as u64 + 1, 1_000_000_000 + (t * 1e9) as u64, 0.);
            for pose in &mut o.poses {
                pose.uncertainty = Some(ConicUncertainty {
                    normals: std::array::from_fn(|s| {
                        let theta = s as f64 * (2. * t + 3. * t * t);
                        [theta.sin(), 0., theta.cos()]
                    }),
                    pivots: [[[0.; 2]; 5]; SHAPE_STATES],
                    innovation_px: [[2.; 5]; SHAPE_STATES],
                    angular_radius_rad: 1.,
                    invalid_samples: 0,
                    full_budget_pivot_radius_px: [0.; 5],
                });
            }
            w.push_back(o);
        }
        let nominal = kinematics(&w, 3, 0, None);
        assert_eq!(norm(nominal.velocity_rad_s), 0.);
        for depth in 5..10 {
            let k = kinematics(&w, 3, 0, Some(depth));
            assert!((k.velocity_rad_s[1] - (2. + 6. * 0.078)).abs() < 1e-8);
            assert!((k.acceleration_rad_s2.unwrap()[1] - 6.).abs() < 1e-7);
            assert!(norm(k.jerk_rad_s3.unwrap()) < 1e-5);
        }
    }
    #[test]
    fn common_window_dp_equals_exhaustive_both_objectives() {
        let mut tracker = Tracker::default();
        let mut final_out = None;
        for frame in 1..=8 {
            let mut o = observation(
                frame,
                1_000_000_000 + frame * 20_000_000,
                0.15 + 0.04 * frame as f64,
            );
            o.poses[1].normal = [-o.poses[0].normal[0], 0., o.poses[0].normal[2]];
            for d in 0..5 {
                for i in 0..2 {
                    o.poses[i].pivot_grid_px[d] = [
                        (frame as f64 * 1.3 + i as f64 * 1.1 + d as f64 * 0.2).sin() * 8.,
                        i as f64 * 5.,
                    ];
                }
            }
            for branch in 0..2 {
                let base = o.poses[branch].pivot_grid_px;
                o.poses[branch].uncertainty = Some(ConicUncertainty {
                    normals: std::array::from_fn(|s| {
                        let theta = (if branch == 0 { 1. } else { -1. })
                            * (0.15 + 0.04 * frame as f64)
                            + 0.003 * s as f64 * (frame as f64 * 0.7).sin();
                        [theta.sin(), 0., theta.cos()]
                    }),
                    pivots: std::array::from_fn(|s| {
                        std::array::from_fn(|d| {
                            [
                                base[d][0] + (frame as f64 * 0.3 + s as f64).sin() * s as f64,
                                base[d][1] + (frame as f64 * 0.7 - s as f64).cos() * s as f64,
                            ]
                        })
                    }),
                    innovation_px: [[2.; 5]; SHAPE_STATES],
                    angular_radius_rad: 0.,
                    invalid_samples: 0,
                    full_budget_pivot_radius_px: [0.; 5],
                });
            }
            if let Some(p) = tracker.window.back() {
                if frame != 4 {
                    o.transport = Some(Transport {
                        from: p.source,
                        to: o.source,
                        a: 1.,
                        b: 0.,
                        translation: [0.; 2],
                        allowance_px: 1.,
                    });
                }
            }
            final_out = Some(tracker.observe(o));
        }
        let w = &tracker.window;
        let mut independent = [f64::INFINITY; 2];
        let mut joint = independent;
        for depth in 0..NUISANCE_STATES {
            for bits in 0_u16..(1 << w.len()) {
                let mut a = 0.;
                let mut b = 0.;
                for t in 1..w.len() {
                    let i = ((bits >> t) & 1) as usize;
                    let j = ((bits >> (t - 1)) & 1) as usize;
                    let r = w[t]
                        .transport
                        .map_or(0., |tr| residuals(w[t - 1], w[t], tr).0[depth][j][i]);
                    a += robust(r);
                    let mut k = kinematics(w, t, bits, Some(depth));
                    b += robust(r) + motion_cost(&mut k) + arc_cost(w, t, bits, depth);
                }
                let branch = ((bits >> (w.len() - 1)) & 1) as usize;
                independent[branch] = independent[branch].min(a / 6.);
                joint[branch] = joint[branch].min(b / 6.);
            }
        }
        let out = final_out.unwrap();
        for i in 0..2 {
            assert!((out.history_scores.unwrap()[i] - independent[i]).abs() < 1e-12);
            assert!((out.costs[i] - joint[i]).abs() < 1e-12);
        }
        assert_eq!(out.window_intervals, 7);
        assert_eq!(out.support, 6);
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub normal: [f64; 3],
    pub pivot_px: [f64; 2],
    pub pivot_grid_px: [[f64; 2]; 5],
    pub allowance_px: f64,
    pub uncertainty: Option<ConicUncertainty>,
}
/// Finite exact-forward samples, not a certified continuous enclosure.
/// Shared state has the same sensor-basis shape perturbation for every source.
#[derive(Clone, Copy, Debug)]
pub struct ConicUncertainty {
    pub normals: [[f64; 3]; SHAPE_STATES],
    pub pivots: [[[f64; 2]; 5]; SHAPE_STATES],
    pub innovation_px: [[f64; 5]; SHAPE_STATES],
    pub angular_radius_rad: f64,
    pub invalid_samples: usize,
    pub full_budget_pivot_radius_px: [f64; 5],
}
impl Pose {
    pub fn nuisance_normal(self, state: usize) -> [f64; 3] {
        self.uncertainty
            .map_or(self.normal, |u| u.normals[state / 5])
    }
    fn nuisance_pivot(self, state: usize) -> [f64; 2] {
        self.uncertainty.map_or(self.pivot_grid_px[state % 5], |u| {
            u.pivots[state / 5][state % 5]
        })
    }
    fn nuisance_allowance(self, state: usize) -> f64 {
        self.uncertainty
            .map_or(self.allowance_px, |u| u.innovation_px[state / 5][state % 5])
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Transport {
    pub from: Source,
    pub to: Source,
    pub a: f64,
    pub b: f64,
    pub translation: [f64; 2],
    pub allowance_px: f64,
}
/// Independent camera-basis rotation carrying the previous head into this one.
/// A 2-D image similarity is never converted to this measurement.
/// Component cross-product convention: positive Z rotates +right toward +down.
/// For a reflected basis M, axial rotation vectors transform as det(M)*M,
/// unlike polar positions/normals. Providers must convert their own convention.
#[derive(Clone, Copy, Debug)]
pub struct HeadRotation {
    pub from: Source,
    pub to: Source,
    pub vector_rad: [f64; 3],
}
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub source: Source,
    pub fresh: bool,
    pub poses: [Pose; 2],
    pub seed: usize,
    pub transport: Option<Transport>,
    pub head_rotation: Option<HeadRotation>,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Kinematics {
    pub velocity_rad_s: [f64; 3],
    pub acceleration_rad_s2: Option<[f64; 3]>,
    pub jerk_rad_s3: Option<[f64; 3]>,
    pub head_compensated: bool,
    pub mode: &'static str,
}
#[derive(Clone, Copy, Debug)]
struct Path {
    score: f64,
    max_residual: f64,
    bits: u16,
    nuisance: usize,
}
impl Path {
    const EMPTY: Self = Self {
        score: f64::INFINITY,
        max_residual: 0.,
        bits: 0,
        nuisance: 0,
    };
}
/// Each suffix cell runs two additive recurrences over exactly the same edges:
/// residual-only sign evidence, and residual+kinematics provisional ranking.
#[derive(Clone, Copy, Debug)]
struct Cell {
    independent: Path,
    joint: Path,
}
impl Cell {
    const EMPTY: Self = Self {
        independent: Path::EMPTY,
        joint: Path::EMPTY,
    };
}
#[derive(Clone, Debug, Default)]
pub struct Tracker {
    window: VecDeque<Observation>,
    emitted: VecDeque<Observation>,
    watermark: Option<Source>,
    preferred_normal: Option<[f64; 3]>,
}
#[derive(Clone, Debug)]
pub struct Outcome {
    pub preferred: Option<usize>,
    pub identified: Option<usize>,
    pub costs: [f64; 2],
    pub independent: Option<[f64; 2]>,
    pub history_bounds: Option<[f64; 2]>,
    pub history_scores: Option<[f64; 2]>,
    pub margin: Option<f64>,
    pub support: usize,
    pub states: usize,
    pub transitions: usize,
    pub residual_px: Option<[[f64; 2]; 2]>,
    pub bounds_px: Option<[[f64; 2]; 2]>,
    pub window_start: Option<Source>,
    pub window_intervals: usize,
    pub kinematics: Option<Kinematics>,
    pub emitted_kinematics: Option<Kinematics>,
    pub reason: &'static str,
    pub independent_nuisance: [usize; 2],
    pub joint_nuisance: [usize; 2],
}
pub fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter()
        .zip(b)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-1., 1.)
        .acos()
}
fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn rotate(v: [f64; 3], rotation: Option<HeadRotation>) -> [f64; 3] {
    let Some(r) = rotation else {
        return v;
    };
    let theta = norm(r.vector_rad);
    if theta < 1e-12 {
        return v;
    }
    let axis = r.vector_rad.map(|x| x / theta);
    let c = theta.cos();
    let s = theta.sin();
    let k = cross(axis, v);
    let dot = axis.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
    std::array::from_fn(|j| v[j] * c + k[j] * s + axis[j] * dot * (1. - c))
}
fn robust(r: f64) -> f64 {
    if r <= 1. {
        r * r
    } else {
        2. * r - 1.
    }
}
fn kinematics(
    w: &VecDeque<Observation>,
    end: usize,
    bits: u16,
    state: Option<usize>,
) -> Kinematics {
    let mut start = end.saturating_sub(3);
    // Reset derivative order at a change of reference semantics. An unknown
    // head interval cannot be differentiated against a head-removed interval.
    for t in (start + 1)..end {
        if w[t].head_rotation.is_some() != w[end].head_rotation.is_some() {
            start = t;
        }
    }
    let mut velocities = Vec::with_capacity(3);
    let mut accelerations = Vec::with_capacity(2);
    let mut result = Kinematics::default();
    let compensated = end > start && ((start + 1)..=end).all(|i| w[i].head_rotation.is_some());
    for t in (start + 1)..=end {
        let previous = w[t - 1];
        let current = w[t];
        let normal = |pose: Pose| state.map_or(pose.normal, |s| pose.nuisance_normal(s));
        let a = normal(previous.poses[((bits >> (t - 1)) & 1) as usize]);
        let b = normal(current.poses[((bits >> t) & 1) as usize]);
        let a = rotate(
            a,
            if compensated {
                current.head_rotation
            } else {
                None
            },
        );
        let axis = cross(a, b);
        let length = norm(axis);
        let dt = (current.source.ns - previous.source.ns) as f64 * 1e-9;
        let omega = if length > 1e-12 {
            axis.map(|x| x / length * angle(a, b) / dt)
        } else {
            [0.; 3]
        };
        // Express all derivative vectors in the final camera basis.
        let mut omega = omega;
        if compensated {
            for future in (t + 1)..=end {
                omega = rotate(omega, w[future].head_rotation);
            }
        }
        let midpoint = (previous.source.ns - w[start].source.ns) as f64 * 1e-9 + dt * 0.5;
        if let Some((last_t, last_v)) = velocities.last().copied() {
            let last_v: [f64; 3] = last_v;
            let acceleration =
                std::array::from_fn(|j| (omega[j] - last_v[j]) / (midpoint - last_t));
            let acceleration_time = (midpoint + last_t) * 0.5;
            if let Some((last_at, last_a)) = accelerations.last().copied() {
                let last_a: [f64; 3] = last_a;
                result.jerk_rad_s3 = Some(std::array::from_fn(|j| {
                    (acceleration[j] - last_a[j]) / (acceleration_time - last_at)
                }));
            }
            accelerations.push((acceleration_time, acceleration));
            result.acceleration_rad_s2 = Some(acceleration);
        }
        velocities.push((midpoint, omega));
        result.velocity_rad_s = omega;
    }
    result.head_compensated = compensated && end > 0;
    result
}
fn motion_cost(k: &mut Kinematics) -> f64 {
    // Interval estimates, not measured instantaneous physiological peaks.
    // A finite low-cost free-saccade mode prevents minimum-jerk enforcement.
    let speed = (norm(k.velocity_rad_s) / 1200_f64.to_radians() - 1.).max(0.);
    let acceleration = k
        .acceleration_rad_s2
        .map_or(0., |v| norm(v) / 40000_f64.to_radians());
    let jerk = k
        .jerk_rad_s3
        .map_or(0., |v| norm(v) / 8_000_000_f64.to_radians());
    let smooth = 0.01 * robust(acceleration) + 0.005 * robust(jerk);
    k.mode = if smooth <= 0.02 {
        "low-order"
    } else {
        "free-saccade"
    };
    0.002 * robust(speed).min(4.) + smooth.min(0.02)
}
/// Arc length, not a physiological rate limit. Every competing latent path pays
/// this same edge cost; an earlier emitted branch gets no privileged penalty.
fn arc_cost(w: &VecDeque<Observation>, end: usize, bits: u16, state: usize) -> f64 {
    let a = w[end - 1].poses[((bits >> (end - 1)) & 1) as usize].nuisance_normal(state);
    let b = w[end].poses[((bits >> end) & 1) as usize].nuisance_normal(state);
    0.03 * angle(rotate(a, w[end].head_rotation), b)
}
fn residuals(
    p: Observation,
    o: Observation,
    t: Transport,
) -> (
    [[[f64; 2]; 2]; NUISANCE_STATES],
    [[f64; 2]; 2],
    [[f64; 2]; 2],
) {
    let mut residual = [[[0.; 2]; 2]; NUISANCE_STATES];
    let mut pixels = [[0.; 2]; 2];
    let mut bounds = [[0.; 2]; 2];
    let dt = (o.source.ns - p.source.ns) as f64 * 1e-9;
    for depth in 0..NUISANCE_STATES {
        for j in 0..2 {
            for i in 0..2 {
                let error = |d: usize| {
                    let x = p.poses[j].nuisance_pivot(d);
                    let c = o.poses[i].nuisance_pivot(d);
                    [
                        t.a * x[0] - t.b * x[1] + t.translation[0] - c[0],
                        t.b * x[0] + t.a * x[1] + t.translation[1] - c[1],
                    ]
                };
                let e = error(depth);
                let adjacent = error(if depth % 5 == 4 { depth - 1 } else { depth + 1 });
                let subcell = 0.5 * (adjacent[0] - e[0]).hypot(adjacent[1] - e[1]);
                let bound = (p.poses[j].nuisance_allowance(depth) * t.a.hypot(t.b)
                    + o.poses[i].nuisance_allowance(depth)
                    + t.allowance_px
                    + subcell
                    + 0.5
                    + 2. * dt)
                    .max(0.1);
                let raw = e[0].hypot(e[1]);
                residual[depth][j][i] = raw / bound;
                if depth == 2 {
                    pixels[j][i] = raw;
                    bounds[j][i] = bound;
                }
            }
        }
    }
    (residual, pixels, bounds)
}
impl Tracker {
    pub fn observe(&mut self, o: Observation) -> Outcome {
        self.observe_order(o, true)
    }
    #[cfg(test)]
    pub fn observe_without_motion_order(&mut self, o: Observation) -> Outcome {
        self.observe_order(o, false)
    }
    fn observe_order(&mut self, mut o: Observation, motion_order: bool) -> Outcome {
        let cost = |k: &mut Kinematics| {
            if !motion_order {
                k.acceleration_rad_s2 = None;
                k.jerk_rad_s3 = None;
            }
            motion_cost(k)
        };
        let mut out = Outcome {
            preferred: None,
            identified: None,
            costs: [0.; 2],
            independent: None,
            history_bounds: None,
            history_scores: None,
            margin: None,
            support: 0,
            states: 0,
            transitions: 0,
            residual_px: None,
            bounds_px: None,
            window_start: None,
            window_intervals: 0,
            kinematics: None,
            emitted_kinematics: None,
            reason: "rejected",
            independent_nuisance: [0; 2],
            joint_nuisance: [0; 2],
        };
        if !o.fresh || o.source.ns == 0 {
            return out;
        }
        if self.watermark.is_some_and(|p| {
            p.stream == o.source.stream && (o.source.ns <= p.ns || o.source.frame <= p.frame)
        }) {
            return out;
        }
        self.watermark = Some(o.source);
        if o.seed > 1
            || o.poses.iter().any(|p| {
                p.normal
                    .iter()
                    .chain(p.pivot_grid_px.iter().flatten())
                    .any(|x| !x.is_finite())
                    || (norm(p.normal) - 1.).abs() > 1e-5
                    || p.normal[2] <= 0.
                    || !p.allowance_px.is_finite()
                    || p.allowance_px < 0.
                    || p.uncertainty.is_some_and(|u| {
                        u.normals.iter().any(|n| {
                            n.iter().any(|x| !x.is_finite())
                                || (norm(*n) - 1.).abs() > 1e-5
                                || n[2] <= 0.
                        }) || u.pivots.iter().flatten().flatten().any(|x| !x.is_finite())
                            || u.innovation_px
                                .iter()
                                .flatten()
                                .any(|x| !x.is_finite() || *x < 0.)
                            || !u.angular_radius_rad.is_finite()
                            || u.angular_radius_rad < 0.
                    })
            })
        {
            self.window.clear();
            self.emitted.clear();
            self.preferred_normal = None;
            out.reason = "invalid-geometry";
            return out;
        }
        if self.window.back().is_some_and(|p| {
            p.source.stream != o.source.stream || o.source.ns - p.source.ns > 750_000_000
        }) {
            self.window.clear();
            self.emitted.clear();
            self.preferred_normal = None;
        }
        let previous = self.window.back().copied();
        o.transport = o.transport.filter(|t| {
            previous.is_some_and(|p| t.from == p.source)
                && t.to == o.source
                && [t.a, t.b, t.translation[0], t.translation[1], t.allowance_px]
                    .into_iter()
                    .all(f64::is_finite)
                && t.a.hypot(t.b) > 0.
                && t.allowance_px >= 0.
        });
        o.head_rotation = o.head_rotation.filter(|h| {
            previous.is_some_and(|p| h.from == p.source)
                && h.to == o.source
                && h.vector_rad.into_iter().all(f64::is_finite)
                && norm(h.vector_rad).is_finite()
        });
        self.window.push_back(o);
        while self
            .emitted
            .front()
            .is_some_and(|p| o.source.ns - p.source.ns > HORIZON_NS)
        {
            self.emitted.pop_front();
        }
        while self.window.len() > MAX_INTERVALS + 1
            || self
                .window
                .front()
                .is_some_and(|p| o.source.ns - p.source.ns > HORIZON_NS)
        {
            self.window.pop_front();
        }
        let w = &self.window;
        let end = w.len() - 1;
        out.window_start = Some(w[0].source);
        out.window_intervals = end;
        let support = (1..w.len()).filter(|&i| w[i].transport.is_some()).count();
        out.support = support;
        let mut cells = [[Cell::EMPTY; 8]; NUISANCE_STATES];
        for (d, depth) in cells.iter_mut().enumerate() {
            for (branch, cell) in depth[..2].iter_mut().enumerate() {
                let p = Path {
                    score: 0.,
                    max_residual: 0.,
                    bits: branch as u16,
                    nuisance: d,
                };
                *cell = Cell {
                    independent: p,
                    joint: p,
                };
            }
        }
        for t in 1..w.len() {
            let r = w[t].transport.map(|tr| residuals(w[t - 1], w[t], tr));
            if t == end {
                if let Some((res, pixels, bounds)) = r {
                    out.residual_px = Some(pixels);
                    out.bounds_px = Some(bounds);
                    out.independent = Some(std::array::from_fn(|i| {
                        (0..NUISANCE_STATES)
                            .flat_map(|d| [res[d][0][i], res[d][1][i]])
                            .fold(f64::INFINITY, f64::min)
                    }));
                }
            }
            let mut next = [[Cell::EMPTY; 8]; NUISANCE_STATES];
            for d in 0..NUISANCE_STATES {
                for suffix in 0..8 {
                    let old = cells[d][suffix];
                    if !old.joint.score.is_finite() {
                        continue;
                    }
                    for i in 0..2 {
                        out.transitions += 1;
                        let branch = suffix & 1;
                        let target = ((suffix << 1) | i) & 7;
                        let rr = r.map_or(0., |(res, _, _)| res[d][branch][i]);
                        let evidence = robust(rr);
                        let bits = old.joint.bits | ((i as u16) << t);
                        let mut k = kinematics(w, t, bits, Some(d));
                        let motion = cost(&mut k) + arc_cost(w, t, bits, d);
                        let joint = Path {
                            score: old.joint.score + evidence + motion,
                            max_residual: old.joint.max_residual.max(rr),
                            bits,
                            nuisance: d,
                        };
                        let independent = Path {
                            score: old.independent.score + evidence,
                            max_residual: old.independent.max_residual.max(rr),
                            bits: old.independent.bits | ((i as u16) << t),
                            nuisance: d,
                        };
                        if joint.score < next[d][target].joint.score {
                            next[d][target].joint = joint;
                        }
                        if independent.score < next[d][target].independent.score {
                            next[d][target].independent = independent;
                        }
                    }
                }
            }
            cells = next;
        }
        out.states = cells
            .iter()
            .flatten()
            .filter(|c| c.joint.score.is_finite())
            .count();
        let mut best = [Cell::EMPTY; 2];
        for depth in cells {
            for (suffix, cell) in depth.into_iter().enumerate() {
                let i = suffix & 1;
                if cell.joint.score < best[i].joint.score {
                    best[i].joint = cell.joint;
                }
                if cell.independent.score < best[i].independent.score {
                    best[i].independent = cell.independent;
                }
            }
        }
        out.costs = best.map(|c| c.joint.score / support.max(1) as f64);
        out.independent_nuisance = best.map(|c| c.independent.nuisance);
        out.joint_nuisance = best.map(|c| c.joint.nuisance);
        let scores = best.map(|c| c.independent.score / support.max(1) as f64);
        let winner = usize::from(scores[1] < scores[0]);
        let margin = scores[1 - winner] - scores[winner];
        if support > 0 {
            out.history_scores = Some(scores);
            out.history_bounds = Some(best.map(|c| c.independent.max_residual));
            out.margin = Some(margin);
        }
        let meaningful = o.transport.is_some()
            && margin >= 0.15
            && scores[1 - winner] >= 2. * scores[winner] + 0.15
            && scores[winner] <= 1.;
        let mut selected = if (out.costs[0] - out.costs[1]).abs() < 1e-10 {
            self.preferred_normal.map_or(o.seed, |n| {
                usize::from(angle(n, o.poses[1].normal) < angle(n, o.poses[0].normal))
            })
        } else {
            usize::from(out.costs[1] < out.costs[0])
        };
        if meaningful {
            selected = winner;
        }
        let angular_overlap = o
            .poses
            .iter()
            .map(|p| p.uncertainty.map_or(0., |u| u.angular_radius_rad))
            .sum::<f64>();
        let valid_samples = o
            .poses
            .iter()
            .all(|p| p.uncertainty.is_none_or(|u| u.invalid_samples == 0));
        if meaningful
            && support >= 3
            && valid_samples
            && angle(o.poses[0].normal, o.poses[1].normal) > 0.05_f64.max(angular_overlap)
        {
            out.identified = Some(selected);
        }
        if end > 0 {
            let mut k = kinematics(
                w,
                end,
                best[selected].joint.bits,
                Some(best[selected].joint.nuisance),
            );
            motion_cost(&mut k);
            out.kinematics = Some(k);
        }
        out.preferred = Some(selected);
        let mut emitted = o;
        emitted.poses = [o.poses[selected]; 2];
        self.emitted.push_back(emitted);
        if self.emitted.len() > 1 {
            let mut k = kinematics(&self.emitted, self.emitted.len() - 1, 0, None);
            motion_cost(&mut k);
            out.emitted_kinematics = Some(k);
        }
        while self.emitted.len() > 3 {
            self.emitted.pop_front();
        }
        self.preferred_normal = Some(o.poses[selected].normal);
        out.reason = if out.identified.is_some() {
            "common-window-robust-margin"
        } else if o.transport.is_none() {
            "provisional-no-transport"
        } else {
            "provisional-ambiguous"
        };
        out
    }
}
