//! Experimental conventional CPU sign tracker, independent of learned runtimes.
//! Coordinates: sensor pixels; unit normals: camera x right, y down, z toward camera.
//! Costs/support are heuristic, never calibrated probabilities. The pivot is an
//! uncertain effective pivot, not a rigid anatomical hinge. No torque/inertia or
//! mass-derived acceleration model is used. One instance belongs to one ROI.
use std::collections::VecDeque;

pub const STATES_PER_SIGN: usize = 4;
pub const MAX_EVIDENCE: usize = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceIdentity {
    pub stream: u64,
    pub frame: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Hypothesis {
    pub normal_camera: [f64; 3],
    pub effective_pivot_sensor_px: [f64; 2],
    pub pivot_sigma_px: f64,
}

/// Explicit independent interval evidence. A missing transform is NOT identity.
#[derive(Clone, Copy, Debug)]
pub struct HeadTransport {
    pub from: SourceIdentity,
    pub to: SourceIdentity,
    pub from_source_ns: u64,
    pub to_source_ns: u64,
    /// p' = scale * R(angle) * (p-center) + center + translation.
    pub center_sensor_px: [f64; 2],
    pub translation_px: [f64; 2],
    pub scale: f64,
    pub angle_rad: f64,
    pub sigma_px: f64,
    /// Axis-angle vector, independently measured in camera coordinates.
    /// Image-plane similarity angle alone cannot provide this 3-D rotation.
    /// Operational convention: Rodrigues' component cross-product; positive Z
    /// rotates +right toward +down. Adapters must convert their handedness.
    pub rotation_vector_camera_rad: Option<[f64; 3]>,
}

#[derive(Clone, Copy, Debug)]
pub struct IndependentAnchor {
    pub source: SourceIdentity,
    pub normal_camera: [f64; 3],
    pub angular_sigma_rad: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub source: SourceIdentity,
    pub source_ns: Option<u64>,
    pub fresh: bool,
    pub visible: bool,
    pub hypotheses: [Hypothesis; 2],
    pub head: Option<HeadTransport>,
    pub anchor: Option<IndependentAnchor>,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub max_interval_s: f64,
    pub evidence_horizon_s: f64,
    pub min_support: usize,
    pub nonrigid_allowance_px: f64,
    pub frontal_separation_rad: f64,
    /// Loose engineering onset (1200 degrees/s); soft and saturated, not a bound.
    pub velocity_onset_rad_s: f64,
    /// Disabled by default; engineering penalties, not biomechanical inference.
    pub acceleration_onset_rad_s2: Option<f64>,
    pub jerk_onset_rad_s3: Option<f64>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            max_interval_s: 0.75,
            evidence_horizon_s: 1.25,
            min_support: 3,
            nonrigid_allowance_px: 0.5,
            frontal_separation_rad: 0.05,
            velocity_onset_rad_s: 1200_f64.to_radians(),
            acceleration_onset_rad_s2: None,
            jerk_onset_rad_s3: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Fixation,
    Saccade,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Rejected,
    MissingGeometry,
    Seeded,
    Ambiguous,
    Resolved,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    MissingClock,
    Held,
    DuplicateOrLate,
}
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub status: Status,
    pub reject_reason: Option<RejectReason>,
    pub selected: Option<usize>,
    pub support: usize,
    pub independent_costs: Option<[f64; 2]>,
    pub beam_costs: [f64; 2],
    /// Minimal normal-transport angular velocity; torsion about the normal is
    /// unobservable from a conic and is not claimed here.
    pub angular_velocity_rad_s: Option<[f64; 3]>,
    pub head_rotation_removed: bool,
    pub mode: Option<Mode>,
    pub states: usize,
    pub transitions: usize,
}
#[derive(Clone, Debug)]
struct State {
    sign: usize,
    parent: usize,
    h: Hypothesis,
    cost: f64,
    mode: Mode,
    velocity: Option<[f64; 3]>,
    acceleration: Option<[f64; 3]>,
    derivative_dt: Option<f64>,
    acceleration_dt: Option<f64>,
    head_removed: bool,
}
#[derive(Debug)]
pub struct SignKinematicBeam {
    config: Config,
    previous: Option<(SourceIdentity, u64)>,
    states: Vec<State>,
    evidence: VecDeque<(u64, [f64; 3])>,
}
impl Default for SignKinematicBeam {
    fn default() -> Self {
        Self::new(Config::default())
    }
}
impl SignKinematicBeam {
    pub fn new(config: Config) -> Self {
        assert!(config.max_interval_s.is_finite() && config.max_interval_s > 0.0);
        assert!(config.evidence_horizon_s.is_finite() && config.evidence_horizon_s > 0.0);
        assert!((1..=MAX_EVIDENCE).contains(&config.min_support));
        assert!(nonnegative(config.nonrigid_allowance_px));
        assert!(nonnegative(config.frontal_separation_rad));
        assert!(positive(config.velocity_onset_rad_s));
        assert!(config.acceleration_onset_rad_s2.is_none_or(positive));
        assert!(config.jerk_onset_rad_s3.is_none_or(positive));
        Self {
            config,
            previous: None,
            states: Vec::new(),
            evidence: VecDeque::new(),
        }
    }
    pub fn clear(&mut self) {
        self.previous = None;
        self.states.clear();
        self.evidence.clear();
    }
    pub fn state_count(&self) -> usize {
        self.states.len()
    }
    pub fn evidence_count(&self) -> usize {
        self.evidence.len()
    }
    fn empty(&self, status: Status) -> Outcome {
        Outcome {
            status,
            reject_reason: None,
            selected: None,
            support: 0,
            independent_costs: None,
            beam_costs: [0.0; 2],
            angular_velocity_rad_s: None,
            head_rotation_removed: false,
            mode: None,
            states: self.states.len(),
            transitions: 0,
        }
    }
    pub fn observe(&mut self, o: Observation) -> Outcome {
        let reject_reason = if o.source_ns.is_none() {
            Some(RejectReason::MissingClock)
        } else if !o.fresh {
            Some(RejectReason::Held)
        } else if self.previous.is_some_and(|(id, t)| {
            id.stream == o.source.stream
                && (o.source.frame <= id.frame || o.source_ns.unwrap() <= t)
        }) {
            Some(RejectReason::DuplicateOrLate)
        } else {
            None
        };
        if reject_reason.is_some() {
            let mut out = self.empty(Status::Rejected);
            out.reject_reason = reject_reason;
            return out;
        }
        let ns = o.source_ns.unwrap();
        if self
            .previous
            .is_some_and(|(id, _)| id.stream != o.source.stream)
        {
            self.clear();
        }
        let previous = self.previous;
        self.previous = Some((o.source, ns));
        if !o.visible || !o.hypotheses.iter().all(valid_hypothesis) {
            self.states.clear();
            self.evidence.clear();
            return self.empty(Status::MissingGeometry);
        }
        let dt = previous.map(|(_, t)| (ns - t) as f64 * 1e-9);
        if dt.is_none_or(|d| d < 0.001 || d > self.config.max_interval_s) {
            self.states.clear();
            self.evidence.clear();
        }
        let head = o.head.filter(|h| {
            previous.is_some_and(|(id, t)| h.from == id && h.from_source_ns == t)
                && h.to == o.source
                && h.to_source_ns == ns
                && valid_head(h)
        });
        let anchor = o.anchor.filter(|a| {
            a.source == o.source && unit(a.normal_camera) && positive(a.angular_sigma_rad)
        });
        let mut independent = [f64::INFINITY; 2];
        let mut expanded = Vec::with_capacity(36);
        for sign in 0..2 {
            let h = o.hypotheses[sign];
            let anchor_cost = anchor.map(|a| {
                saturated(angle(a.normal_camera, h.normal_camera) / a.angular_sigma_rad.max(0.01))
            });
            for (parent, p) in self.states.iter().enumerate() {
                let pivot_cost = head.map(|head| {
                    let predicted = transport(head, p.h.effective_pivot_sensor_px);
                    let allowance = (head.sigma_px.powi(2)
                        + h.pivot_sigma_px.powi(2)
                        + (head.scale * p.h.pivot_sigma_px).powi(2))
                    .sqrt()
                        + self.config.nonrigid_allowance_px;
                    saturated(distance(predicted, h.effective_pivot_sensor_px) / allowance.max(0.1))
                });
                let independent_cost = match (pivot_cost, anchor_cost) {
                    (Some(p), Some(a)) => Some((p + a) * 0.5),
                    (p, a) => p.or(a),
                };
                if let Some(c) = independent_cost {
                    independent[sign] = independent[sign].min(c);
                }
                let d = dt.unwrap();
                let rotation = head.and_then(|h| h.rotation_vector_camera_rad);
                let prior = rotation.map_or(p.h.normal_camera, |r| rotate(p.h.normal_camera, r));
                let velocity = scale(rotation_between(prior, h.normal_camera), 1.0 / d);
                // Derivative intervals are centered at their actual source-time midpoints.
                // Rotate previous vectors into the current camera basis when known.
                let comparable = p.head_removed == rotation.is_some();
                let old_v = p
                    .velocity
                    .filter(|_| comparable)
                    .map(|v| rotation.map_or(v, |r| rotate(v, r)));
                let midpoint_dt = (d + p.derivative_dt.unwrap_or(d)) * 0.5;
                let acceleration = old_v.map(|v| scale(sub(velocity, v), 1.0 / midpoint_dt));
                let jerk = acceleration
                    .zip(p.acceleration.filter(|_| comparable))
                    .map(|(a, b)| {
                        let b = rotation.map_or(b, |r| rotate(b, r));
                        norm(sub(a, b))
                            / ((midpoint_dt + p.acceleration_dt.unwrap_or(midpoint_dt)) * 0.5)
                    });
                for mode in [Mode::Fixation, Mode::Saccade] {
                    let onset = self.config.velocity_onset_rad_s;
                    let motion = match mode {
                        Mode::Fixation => 0.2 * saturated(norm(velocity) / (onset * 0.1)),
                        Mode::Saccade => 0.05 + 0.2 * soft_excess(norm(velocity), onset),
                    } + self
                        .config
                        .acceleration_onset_rad_s2
                        .zip(acceleration)
                        .map_or(0.0, |(onset, a)| 0.1 * soft_excess(norm(a), onset))
                        + self
                            .config
                            .jerk_onset_rad_s3
                            .zip(jerk)
                            .map_or(0.0, |(onset, j)| 0.05 * soft_excess(j, onset));
                    expanded.push(State {
                        sign,
                        parent,
                        h,
                        cost: p.cost * 0.65 + independent_cost.unwrap_or(0.0) + motion,
                        mode,
                        velocity: Some(velocity),
                        acceleration,
                        derivative_dt: Some(d),
                        acceleration_dt: acceleration.map(|_| midpoint_dt),
                        head_removed: rotation.is_some(),
                    });
                }
            }
            if self.states.is_empty() {
                if let Some(c) = anchor_cost {
                    independent[sign] = c;
                }
            }
        }
        let transitions = expanded.len();
        let seeded = self.states.is_empty();
        let mut retained = Vec::with_capacity(STATES_PER_SIGN * 2);
        for sign in 0..2 {
            let mut candidates: Vec<_> = expanded
                .iter()
                .filter(|s| s.sign == sign)
                .cloned()
                .collect();
            candidates.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            // Preserve distinct predecessor lineages before taking a second mode.
            let mut parents = Vec::new();
            for c in &candidates {
                if !parents.contains(&c.parent) && parents.len() < 2 {
                    parents.push(c.parent);
                    retained.push(c.clone());
                }
            }
            for c in &candidates {
                if retained.iter().filter(|s| s.sign == sign).count() >= STATES_PER_SIGN - 1 {
                    break;
                }
                if !retained
                    .iter()
                    .any(|s| s.sign == sign && s.parent == c.parent && s.mode == c.mode)
                {
                    retained.push(c.clone());
                }
            }
            // Fresh zero-derivative lineage on EVERY observation prevents initial lock-in.
            retained.push(State {
                sign,
                parent: usize::MAX,
                h: o.hypotheses[sign],
                cost: independent[sign]
                    .is_finite()
                    .then_some(independent[sign])
                    .unwrap_or(0.0)
                    + 0.75,
                mode: Mode::Fixation,
                velocity: None,
                acceleration: None,
                derivative_dt: None,
                acceleration_dt: None,
                head_removed: false,
            });
            if seeded {
                let mut alternate = retained.last().unwrap().clone();
                alternate.mode = Mode::Saccade;
                retained.push(alternate);
            }
        }
        self.states = retained;
        while self
            .evidence
            .front()
            .is_some_and(|(t, _)| (ns - *t) as f64 * 1e-9 > self.config.evidence_horizon_s)
            || self.evidence.len() >= MAX_EVIDENCE
        {
            self.evidence.pop_front();
        }
        let costs = independent
            .iter()
            .all(|x| x.is_finite())
            .then_some(independent);
        let separation = angle(o.hypotheses[0].normal_camera, o.hypotheses[1].normal_camera);
        let winner = costs.and_then(|c| {
            let best = usize::from(c[1] < c[0]);
            (c[best] <= 1.5
                && c[1 - best] - c[best] >= 1.0
                && separation > self.config.frontal_separation_rad)
                .then_some(best)
        });
        let mut support = 0;
        if let Some(w) = winner {
            self.evidence.push_back((ns, o.hypotheses[w].normal_camera));
            support = self
                .evidence
                .iter()
                .filter(|(_, n)| {
                    angle(*n, o.hypotheses[w].normal_camera)
                        < angle(*n, o.hypotheses[1 - w].normal_camera)
                })
                .count();
        }
        let best: [State; 2] = [0, 1].map(|i| {
            self.states
                .iter()
                .filter(|s| s.sign == i)
                .min_by(|a, b| a.cost.total_cmp(&b.cost))
                .unwrap()
                .clone()
        });
        // Independent observability authorizes a sign; the temporal objective
        // must also favor it. Smoothness can withhold, never manufacture support.
        // Reseeds and decaying historical cost bound disagreement/recovery delay.
        let selected = winner.filter(|&w| {
            support >= self.config.min_support && best[1 - w].cost - best[w].cost >= 0.5
        });
        let chosen = selected.map(|s| &best[s]);
        Outcome {
            status: if selected.is_some() {
                Status::Resolved
            } else if seeded {
                Status::Seeded
            } else {
                Status::Ambiguous
            },
            reject_reason: None,
            selected,
            support,
            independent_costs: costs,
            beam_costs: [best[0].cost, best[1].cost],
            angular_velocity_rad_s: chosen.and_then(|s| s.velocity),
            head_rotation_removed: chosen.is_some_and(|s| s.head_removed),
            mode: chosen.map(|s| s.mode),
            states: self.states.len(),
            transitions,
        }
    }
}

pub fn sensor_from_roi(local: [f64; 2], origin: [f64; 2]) -> [f64; 2] {
    [local[0] + origin[0], local[1] + origin[1]]
}
fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}
fn nonnegative(v: f64) -> bool {
    v.is_finite() && v >= 0.0
}
fn unit(v: [f64; 3]) -> bool {
    v.iter().all(|x| x.is_finite()) && (norm(v) - 1.0).abs() < 1e-6 && v[2] > 0.0
}
fn valid_hypothesis(h: &Hypothesis) -> bool {
    unit(h.normal_camera)
        && h.effective_pivot_sensor_px.iter().all(|x| x.is_finite())
        && nonnegative(h.pivot_sigma_px)
}
fn valid_head(h: &HeadTransport) -> bool {
    h.center_sensor_px
        .iter()
        .chain(h.translation_px.iter())
        .all(|x| x.is_finite())
        && positive(h.scale)
        && h.angle_rad.is_finite()
        && nonnegative(h.sigma_px)
        && h.rotation_vector_camera_rad
            .is_none_or(|r| r.iter().all(|x| x.is_finite()) && norm(r).is_finite())
}
fn transport(h: HeadTransport, p: [f64; 2]) -> [f64; 2] {
    let (s, c) = h.angle_rad.sin_cos();
    let x = p[0] - h.center_sensor_px[0];
    let y = p[1] - h.center_sensor_px[1];
    [
        h.center_sensor_px[0] + h.translation_px[0] + h.scale * (c * x - s * y),
        h.center_sensor_px[1] + h.translation_px[1] + h.scale * (s * x + c * y),
    ]
}
fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|x| x * s)
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}
fn rotation_between(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let axis = cross(a, b);
    let n = norm(axis);
    if n < 1e-12 {
        [0.0; 3]
    } else {
        scale(axis, angle(a, b) / n)
    }
}
fn rotate(v: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let t = norm(r);
    if t < 1e-12 {
        return v;
    }
    let k = scale(r, 1.0 / t);
    let (s, c) = t.sin_cos();
    let kv = cross(k, v);
    [0, 1, 2].map(|i| v[i] * c + kv[i] * s + k[i] * dot(k, v) * (1.0 - c))
}
fn saturated(x: f64) -> f64 {
    x.min(4.0)
}
fn soft_excess(value: f64, onset: f64) -> f64 {
    saturated((value / onset - 1.0).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source(frame: u64) -> SourceIdentity {
        SourceIdentity { stream: 1, frame }
    }
    fn input(frame: u64, axis: usize, tilt: f64, sign: usize) -> Observation {
        let mut n = [0.0, 0.0, tilt.cos()];
        n[axis] = tilt.sin();
        let normals = [n, [-n[0], -n[1], n[2]]];
        // Eye rotates about an independently translated approximate pivot.
        let pivot = [1000.0 + 2.0 * frame as f64, 2000.0 - frame as f64];
        let center = [
            pivot[0] + 100.0 * normals[sign][0],
            pivot[1] + 100.0 * normals[sign][1],
        ];
        let ns = 1 + frame * 20_000_000;
        Observation {
            source: source(frame),
            source_ns: Some(ns),
            fresh: true,
            visible: true,
            hypotheses: normals.map(|normal_camera| Hypothesis {
                normal_camera,
                effective_pivot_sensor_px: [
                    center[0] - 100.0 * normal_camera[0],
                    center[1] - 100.0 * normal_camera[1],
                ],
                pivot_sigma_px: 0.1,
            }),
            head: (frame > 0).then(|| HeadTransport {
                from: source(frame - 1),
                to: source(frame),
                from_source_ns: ns - 20_000_000,
                to_source_ns: ns,
                center_sensor_px: [0.0; 2],
                translation_px: [2.0, -1.0],
                scale: 1.0,
                angle_rad: 0.0,
                sigma_px: 0.1,
                rotation_vector_camera_rad: None,
            }),
            anchor: None,
        }
    }
    fn anchored(mut o: Observation, sign: usize) -> Observation {
        o.anchor = Some(IndependentAnchor {
            source: o.source,
            normal_camera: o.hypotheses[sign].normal_camera,
            angular_sigma_rad: 0.01,
        });
        o
    }
    #[test]
    fn both_x_and_y_signs_resolve_with_independent_transport() {
        for axis in 0..2 {
            for sign in 0..2 {
                let mut tracker = SignKinematicBeam::default();
                for f in 0..16 {
                    let result = tracker.observe(input(f, axis, 0.2 + 0.02 * f as f64, sign));
                    if f >= 4 {
                        assert_eq!(
                            result.selected,
                            Some(sign),
                            "axis {axis} sign {sign} frame {f}: {result:?}"
                        );
                    }
                }
            }
        }
    }
    #[test]
    fn wrong_initial_preference_recovers_and_reseeds_both_signs() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..5 {
            let mut o = anchored(input(f, 0, 0.3, 0), 1);
            o.head = None;
            tracker.observe(o);
        }
        for f in 5..25 {
            let result = tracker.observe(input(f, 0, 0.3 + 0.02 * (f - 5) as f64, 0));
            if f >= 9 {
                assert_eq!(result.selected, Some(0));
            }
            for sign in 0..2 {
                assert!(tracker.states.iter().filter(|s| s.sign == sign).count() >= 2);
                assert!(tracker
                    .states
                    .iter()
                    .any(|s| s.sign == sign && s.velocity.is_none()));
            }
        }
    }
    #[test]
    fn reflected_smooth_motion_and_missing_head_abstain() {
        for independent_identity in [false, true] {
            let mut tracker = SignKinematicBeam::default();
            for f in 0..40 {
                let mut o = input(f, 0, 0.3 + 0.01 * f as f64, 0);
                if independent_identity {
                    o.hypotheses[1].effective_pivot_sensor_px =
                        o.hypotheses[0].effective_pivot_sensor_px;
                } else {
                    o.head = None;
                }
                assert_eq!(tracker.observe(o).selected, None);
            }
        }
    }
    #[test]
    fn nuisance_uncertainty_prevents_manufactured_support() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..40 {
            let mut o = input(f, 1, 0.2 + 0.02 * f as f64, 1);
            for h in &mut o.hypotheses {
                h.pivot_sigma_px = 1000.0;
            }
            assert_eq!(tracker.observe(o).selected, None);
        }
    }
    #[test]
    fn frontal_abstains_even_with_anchor() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..20 {
            assert_eq!(
                tracker.observe(anchored(input(f, 0, 0.001, 0), 0)).selected,
                None
            );
        }
    }
    #[test]
    fn crossing_and_saccade_do_not_clamp_current_normal() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..24 {
            let tilt = 0.3 - 0.03 * f as f64;
            // Canonical positive-axis conic ordering reverses at the meridian.
            let sign = usize::from(tilt < 0.0);
            let result = tracker.observe(anchored(input(f, 0, tilt.abs(), sign), sign));
            if f >= 4 && (tilt.abs() > 0.1) {
                assert_eq!(result.selected, Some(sign), "{f}: {result:?}");
            }
        }
        // Fresh anchored large displacement remains selectable despite engineering onset.
        let mut config = Config::default();
        config.velocity_onset_rad_s = 0.01;
        let mut tracker = SignKinematicBeam::new(config);
        for f in 0..8 {
            let mut o = anchored(input(f, 1, if f < 4 { 0.2 } else { 0.9 }, 0), 0);
            o.head = None;
            let result = tracker.observe(o);
            if f >= 3 {
                assert_eq!(result.selected, Some(0));
            }
        }
    }
    #[test]
    fn duplicates_held_missing_clocks_and_late_frames_do_not_refresh() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..8 {
            tracker.observe(anchored(input(f, 0, 0.3, 0), 0));
        }
        let before = (
            tracker.previous,
            tracker.evidence.len(),
            tracker.state_count(),
        );
        for kind in 0..5 {
            let mut o = input(8, 0, 0.4, 0);
            match kind {
                0 => o.source_ns = None,
                1 => o.fresh = false,
                2 => o.source = source(7),
                3 => o.source_ns = input(7, 0, 0.3, 0).source_ns,
                _ => o.source_ns = Some(1),
            }
            assert_eq!(tracker.observe(o).status, Status::Rejected);
            assert_eq!(
                (
                    tracker.previous,
                    tracker.evidence.len(),
                    tracker.state_count()
                ),
                before
            );
        }
        let mut blink = input(8, 0, 0.3, 0);
        blink.visible = false;
        assert_eq!(tracker.observe(blink).status, Status::MissingGeometry);
        assert_eq!(tracker.evidence_count(), 0);
        assert_eq!(
            tracker.observe(input(7, 0, 0.3, 0)).status,
            Status::Rejected
        );
        assert_eq!(tracker.observe(input(9, 0, 0.3, 0)).status, Status::Seeded);
    }
    #[test]
    fn exact_transport_interval_and_stream_are_required() {
        for wrong in 0..4 {
            let mut tracker = SignKinematicBeam::default();
            for f in 0..12 {
                let mut o = input(f, 0, 0.2 + 0.03 * f as f64, 0);
                if let Some(h) = &mut o.head {
                    match wrong {
                        0 => h.from.frame += 1,
                        1 => h.to.stream += 1,
                        2 => h.from_source_ns += 1,
                        _ => h.to_source_ns += 1,
                    }
                }
                assert_eq!(tracker.observe(o).selected, None);
            }
        }
        let mut tracker = SignKinematicBeam::default();
        tracker.observe(input(100, 0, 0.3, 0));
        let mut new_stream = input(101, 0, 0.3, 0);
        new_stream.source.stream = 2;
        assert_eq!(tracker.observe(new_stream).status, Status::Seeded);
        assert_eq!(tracker.evidence_count(), 0);
    }
    #[test]
    fn geodesic_velocity_and_independent_head_rotation() {
        let n = [0.3_f64.sin(), 0.0, 0.3_f64.cos()];
        let r = [0.0, 0.04, 0.0];
        let moved = rotate(n, r);
        let velocity = scale(rotation_between(n, moved), 50.0);
        assert!((velocity[1] - 2.0).abs() < 1e-10);
        for known in [false, true] {
            let mut tracker = SignKinematicBeam::default();
            tracker.observe(anchored(input(0, 0, 0.3, 0), 0));
            let mut o = anchored(input(1, 0, 0.34, 0), 0);
            if known {
                o.head.as_mut().unwrap().rotation_vector_camera_rad = Some(r);
            }
            tracker.observe(o);
            let state = tracker
                .states
                .iter()
                .find(|s| s.sign == 0 && s.parent == 0)
                .unwrap();
            assert_eq!(state.head_removed, known);
            assert!((norm(state.velocity.unwrap()) - if known { 0.0 } else { 2.0 }).abs() < 1e-9);
        }
    }
    #[test]
    fn irregular_source_time_uses_midpoint_acceleration() {
        let mut tracker = SignKinematicBeam::default();
        for (frame, ns, tilt) in [(0, 1, 0.2), (1, 20_000_001, 0.22), (2, 60_000_001, 0.30)] {
            let mut o = anchored(input(frame, 0, tilt, 0), 0);
            o.source_ns = Some(ns);
            o.head = None;
            tracker.observe(o);
        }
        let acceleration = tracker
            .states
            .iter()
            .filter(|s| s.sign == 0)
            .filter_map(|s| s.acceleration)
            .map(|a| a[1])
            .collect::<Vec<_>>();
        assert!(
            acceleration.iter().any(|a| (*a - 1.0 / 0.03).abs() < 1e-8),
            "{acceleration:?}"
        );
    }
    #[test]
    fn roi_reframe_and_similarity_are_sensor_invariant() {
        assert_eq!(
            sensor_from_roi([10.0, 20.0], [100.0, 200.0]),
            sensor_from_roi([-10.0, 40.0], [120.0, 180.0])
        );
        let mut tracker = SignKinematicBeam::default();
        for f in 0..12 {
            let mut o = input(f, 0, 0.2 + 0.03 * f as f64, 0);
            let origin = if f < 6 {
                [900.0, 1900.0]
            } else {
                [970.0, 1860.0]
            };
            for h in &mut o.hypotheses {
                let p = h.effective_pivot_sensor_px;
                h.effective_pivot_sensor_px =
                    sensor_from_roi([p[0] - origin[0], p[1] - origin[1]], origin);
            }
            let result = tracker.observe(o);
            if f >= 4 {
                assert_eq!(result.selected, Some(0));
            }
        }
        let mut h = input(1, 0, 0.3, 0).head.unwrap();
        h.center_sensor_px = [10.0, 20.0];
        h.angle_rad = std::f64::consts::FRAC_PI_2;
        h.scale = 2.0;
        assert!(distance(transport(h, [11.0, 20.0]), [12.0, 21.0]) < 1e-9);
    }
    #[test]
    fn capacities_and_work_remain_bounded() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..10_000 {
            let result = tracker.observe(anchored(input(f, 0, 0.3, 0), 0));
            assert!(result.states <= STATES_PER_SIGN * 2);
            assert!(result.transitions <= 32);
            assert!(tracker.evidence_count() <= MAX_EVIDENCE);
        }
    }
    #[test]
    fn kinematic_disagreement_can_withhold_independently_supported_switch() {
        let mut results = Vec::new();
        for onset in [0.001, 1e9] {
            let mut config = Config::default();
            config.min_support = 1;
            config.velocity_onset_rad_s = onset;
            let mut tracker = SignKinematicBeam::new(config);
            for f in 0..10 {
                let mut o = anchored(input(f, 0, 0.3, 0), 1);
                o.head = None;
                tracker.observe(o);
            }
            let mut o = anchored(input(10, 0, 0.3, 0), 0);
            o.head = None;
            o.anchor.as_mut().unwrap().angular_sigma_rad = 0.6 / 1.01;
            let result = tracker.observe(o);
            assert!((result.independent_costs.unwrap()[1] - 1.01).abs() < 1e-9);
            results.push(result.selected);
        }
        assert_eq!(results, vec![None, Some(0)]);
    }
    #[test]
    fn old_support_cannot_authorize_missing_current_evidence_or_long_gap() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..10 {
            tracker.observe(anchored(input(f, 0, 0.3, 0), 0));
        }
        let mut neutral = input(10, 0, 0.3, 0);
        neutral.head = None;
        assert_eq!(tracker.observe(neutral).selected, None);
        let gap = anchored(input(100, 0, 0.3, 0), 0);
        let result = tracker.observe(gap);
        assert_eq!(result.selected, None);
        assert_eq!(result.support, 1);
    }
    #[test]
    fn malformed_geometry_and_wrong_source_anchor_cannot_vote() {
        let mut tracker = SignKinematicBeam::default();
        let mut o = input(0, 0, 0.3, 0);
        o.hypotheses[0].normal_camera = [0.0, 0.0, -1.0];
        assert_eq!(tracker.observe(o).status, Status::MissingGeometry);
        for f in 1..10 {
            let mut o = anchored(input(f, 0, 0.3, 0), 0);
            o.head = None;
            o.anchor.as_mut().unwrap().source.frame -= 1;
            assert_eq!(tracker.observe(o).selected, None);
        }
    }

    #[test]
    fn supported_fast_eye_rotation_uses_saccade_mode() {
        let mut tracker = SignKinematicBeam::default();
        for f in 0..10 {
            let mut o = anchored(input(f, 0, 0.3, 0), 0);
            o.head = None;
            tracker.observe(o);
        }
        let mut o = anchored(input(10, 0, 0.6, 0), 0);
        o.head = None;
        let result = tracker.observe(o);
        assert_eq!(result.selected, Some(0));
        assert_eq!(result.mode, Some(Mode::Saccade));
        assert!((norm(result.angular_velocity_rad_s.unwrap()) - 15.0).abs() < 1e-9);
    }

    fn retime(o: &mut Observation, interval_ns: u64) {
        let ns = 1 + o.source.frame * interval_ns;
        o.source_ns = Some(ns);
        if let Some(h) = &mut o.head {
            h.from_source_ns = ns - interval_ns;
            h.to_source_ns = ns;
        }
    }

    #[test]
    fn deterministic_parameter_sweep_recovers_physical_sign() {
        let mut cases = 0;
        let mut observations = 0;
        let mut verified_recovery_outputs = 0;
        let mut max_transitions = 0;
        for axis in 0..2 {
            for true_sign in 0..2 {
                for interval_ns in [8_000_000, 20_000_000, 100_000_000, 300_000_000] {
                    for step in [0.012, 0.025, 0.04] {
                        for wrong_initial in [false, true] {
                            for permute in [false, true] {
                                for episode in 0..3 {
                                    for offset in [[0.0, 0.0], [10_000.0, -7000.0]] {
                                        cases += 1;
                                        let mut tracker = SignKinematicBeam::default();
                                        for frame in 0..24 {
                                            let mut o = input(
                                                frame,
                                                axis,
                                                0.15 + step * frame as f64,
                                                true_sign,
                                            );
                                            retime(&mut o, interval_ns);
                                            for h in &mut o.hypotheses {
                                                h.effective_pivot_sensor_px = sensor_from_roi(
                                                    h.effective_pivot_sensor_px,
                                                    offset,
                                                );
                                            }
                                            let physical = o.hypotheses[true_sign].normal_camera;
                                            if wrong_initial && frame < 5 {
                                                o = anchored(o, 1 - true_sign);
                                                o.head = None;
                                            }
                                            let unavailable =
                                                (10..=12).contains(&frame) && episode != 0;
                                            if unavailable {
                                                if episode == 1 {
                                                    o.head = None;
                                                } else {
                                                    o.head.as_mut().unwrap().sigma_px = 1000.0;
                                                }
                                            }
                                            // The same physical candidates arrive in changing slot order.
                                            if permute && frame % 3 != 0 {
                                                o.hypotheses.swap(0, 1);
                                            }
                                            let result = tracker.observe(o);
                                            observations += 1;
                                            max_transitions =
                                                max_transitions.max(result.transitions);
                                            assert!(result.states <= 8 && result.transitions <= 32);
                                            assert!(tracker.evidence_count() <= MAX_EVIDENCE);
                                            if unavailable {
                                                assert_eq!(
                                                    result.selected, None,
                                                    "missing/nuisance case {cases} frame {frame}"
                                                );
                                            }
                                            // Six restored independent intervals after the episode must
                                            // defeat any initial wrong anchor and reproduce the true ray.
                                            if frame >= 19 {
                                                let selected = result.selected.unwrap_or_else(||panic!(
                                                    "recovery case={cases} axis={axis} sign={true_sign} dt={interval_ns} step={step} wrong={wrong_initial} permutation={permute} episode={episode} frame={frame}: {result:?}"));
                                                assert!(angle(o.hypotheses[selected].normal_camera, physical) < 1e-10,
                                                    "wrong physical normal case {cases} frame {frame}: {result:?}");
                                                verified_recovery_outputs += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 1152);
        assert_eq!(observations, 27_648);
        assert_eq!(verified_recovery_outputs, 5760);
        assert_eq!(max_transitions, 32);
        println!("recovery sweep: {cases} cases, {observations} observations, {verified_recovery_outputs} explicitly correct recovered normals, max {max_transitions} transitions");
    }

    #[test]
    fn deterministic_crossing_sweep_tracks_physical_normal_after_meridian() {
        let mut cases = 0;
        let mut checked = 0;
        for axis in 0..2 {
            for direction in [-1.0, 1.0] {
                for interval in [8_000_000, 20_000_000, 100_000_000, 300_000_000] {
                    for permute in [false, true] {
                        let mut tracker = SignKinematicBeam::default();
                        cases += 1;
                        for frame in 0..31 {
                            let tilt = direction * (0.45 - frame as f64 * 0.03);
                            let sign = usize::from(tilt < 0.0);
                            let mut o = input(frame, axis, tilt.abs(), sign);
                            retime(&mut o, interval);
                            let expected = o.hypotheses[sign].normal_camera;
                            if permute && frame % 2 == 1 {
                                o.hypotheses.swap(0, 1);
                            }
                            let result = tracker.observe(o);
                            if frame == 15 {
                                assert_eq!(result.selected, None);
                            }
                            // Motion-only independent pivot evidence must resolve on
                            // both sides; no anchor supplies the answer at a crossing.
                            if (5..=11).contains(&frame) || frame >= 22 {
                                let selected = result.selected.unwrap_or_else(|| {
                                    panic!(
                                    "crossing case={cases} frame={frame} dt={interval}: {result:?}")
                                });
                                assert!(
                                    angle(o.hypotheses[selected].normal_camera, expected) < 1e-10,
                                    "wrong crossing normal case={cases} frame={frame}"
                                );
                                checked += 1;
                            }
                            assert!(result.transitions <= 32 && result.states <= 8);
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 32);
        assert_eq!(checked, 512);
        println!("crossing sweep: {cases} cases, 992 observations, {checked} explicitly correct normals on both sides");
    }

    #[test]
    fn stream_change_resets_with_restarted_clock_and_frame_counter() {
        let mut tracker = SignKinematicBeam::default();
        for frame in 100..110 {
            tracker.observe(anchored(input(frame, 0, 0.3, 0), 0));
        }
        for frame in 0..10 {
            let mut o = anchored(input(frame, 0, 0.3, 1), 1);
            o.source.stream = 2;
            o.anchor.as_mut().unwrap().source = o.source;
            o.head = None;
            let result = tracker.observe(o);
            if frame == 0 {
                assert_eq!(result.status, Status::Seeded);
                assert_eq!(result.support, 1);
            }
            if frame >= 3 {
                assert_eq!(result.selected, Some(1));
            }
        }
        let watermark = tracker.previous;
        let mut duplicate = input(9, 0, 0.3, 1);
        duplicate.source.stream = 2;
        duplicate.source_ns = Some(999_999_999);
        assert_eq!(
            tracker.observe(duplicate).reject_reason,
            Some(RejectReason::DuplicateOrLate)
        );
        assert_eq!(tracker.previous, watermark);
        let mut duplicate_time = duplicate;
        duplicate_time.source.frame = 20;
        duplicate_time.source_ns = watermark.map(|(_, t)| t);
        assert_eq!(
            tracker.observe(duplicate_time).reject_reason,
            Some(RejectReason::DuplicateOrLate)
        );
        assert_eq!(tracker.previous, watermark);
    }

    #[test]
    fn longer_direct_source_intervals_resolve_slow_eye_motion_but_not_nuisance() {
        // Sensor cadence is 20 ms in every arm. The caller admits every 1st,
        // 4th, or 8th native frame and supplies a DIRECT independent head match
        // over that exact interval, not a stale adjacent transform reused as if
        // it covered 80/160 ms. Four-pixel pivot uncertainty is unchanged.
        let mut cases = 0;
        let mut observations = 0;
        for axis in 0..2 {
            for sign in 0..2 {
                for scenario in 0..4 {
                    for stride in [1_u64, 4, 8] {
                        cases += 1;
                        let mut tracker = SignKinematicBeam::default();
                        let mut first_resolution = None;
                        for frame in (0..=32).step_by(stride as usize) {
                            // Scenario 0: rotating eye and independently moving head.
                            // Scenario 1: equally slow head translation, fixed eye tilt.
                            // Scenario 2: fixed eye tilt plus uncertain nonrigid pivot drift.
                            // Scenario 3: eye rotates, but no independent head evidence.
                            let tilt = if scenario == 0 || scenario == 3 {
                                0.15 + 0.010 * frame as f64
                            } else {
                                0.35
                            };
                            let mut o = input(frame, axis, tilt, sign);
                            let physical = o.hypotheses[sign].normal_camera;
                            for h in &mut o.hypotheses {
                                h.pivot_sigma_px = 4.0;
                                if scenario == 2 {
                                    h.effective_pivot_sensor_px[axis] += 0.15 * frame as f64;
                                }
                            }
                            if let Some(h) = &mut o.head {
                                h.from = source(frame - stride);
                                h.from_source_ns = 1 + (frame - stride) * 20_000_000;
                                h.translation_px = [2.0 * stride as f64, -(stride as f64)];
                                // Explicitly broad nuisance support; no sign-specific
                                // drift measurement is smuggled into the head match.
                                if scenario == 2 {
                                    h.sigma_px = 4.0;
                                }
                            }
                            if scenario == 3 {
                                o.head = None;
                            }
                            let result = tracker.observe(o);
                            observations += 1;
                            if let Some(selected) = result.selected {
                                assert!(scenario == 0 && stride > 1,
                                    "unsupported resolution scenario={scenario} stride={stride} frame={frame}");
                                assert!(
                                    angle(o.hypotheses[selected].normal_camera, physical) < 1e-10
                                );
                                first_resolution.get_or_insert(frame);
                            }
                            if scenario == 0 && stride > 1 && frame >= 3 * stride {
                                assert!(result.selected.is_some(),
                                    "slow coherent motion failed axis={axis} sign={sign} stride={stride} frame={frame}: {result:?}");
                            } else if scenario != 0 || stride == 1 {
                                assert_eq!(result.selected, None);
                            }
                            assert!(result.transitions <= 32 && result.states <= 8);
                        }
                        let expected = (scenario == 0 && stride > 1).then_some(3 * stride);
                        assert_eq!(first_resolution, expected);
                        if axis == 0 && sign == 0 && scenario == 0 {
                            println!("slow-motion stride={stride}: sensor dt=20ms, admitted dt={}ms, first resolved source latency={:?}ms, pivot sigma=4px",
                                20 * stride, first_resolution.map(|f| f * 20));
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 48);
        assert_eq!(observations, 752);
        println!("slow-motion direct-interval sweep: {cases} cases, {observations} admitted observations; head-only, nonrigid-drift and missing-head negatives all abstain");
    }
}
