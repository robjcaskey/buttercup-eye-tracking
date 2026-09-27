//! Experimental shared continuous-sign diagnostics for monocular and joint paths.
//!
//! Dot products between the two normals and the inter-limbus baseline survive
//! any common rigid camera/head transform. This avoids pretending a normal in
//! yesterday's camera coordinates is a normal in today's coordinates. It does
//! NOT observe the missing rotation about that baseline. Smooth reflected gaze
//! histories can be indistinguishable, even far from camera-facing gaze.
//!
//! Scores are bounded engineering preferences, never calibrated probabilities.
//! This module runs beside the native solve; it cannot replace its posterior or
//! authorize calibration. The joint path needs both fitted eyes; the monocular
//! adapter consumes local image-motion support with explicit 3D motion limits.
//! Unique source time and alternative whole poses are required. No candidate
//! radius supplies scale.

use super::joint::JointConicSolution;
use crate::geometry::{cross3, dot3, normalized3, sub3};
use std::collections::VecDeque;

const MAX_GAP_NS: u64 = 1_500_000_000;
const WINDOW_NS: u64 = 3_000_000_000;
const MAX_SOURCES: usize = 64;
const MAX_CANDIDATES: usize = 16;
const MARGIN: f64 = 3.0;
const COALESCENCE_DEGREES: f64 = 5.0;

/// Single-eye adapter over the existing source-matched, motion-compensated
/// contact hypotheses. A 2D similarity is not an attested 3D head/camera pose.
/// These diagnostics never resolve sign or authorize calibration by themselves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MonocularReport {
    pub(crate) status: &'static str,
    pub(crate) preferred_candidate: Option<usize>,
    pub(crate) angular_spread_degrees: f64,
    pub(crate) source_span_ns: u64,
    pub(crate) unique_sources: usize,
}
impl MonocularReport {
    pub(crate) fn json(self) -> serde_json::Value {
        serde_json::json!({"status":self.status,"preferred_candidate":self.preferred_candidate,
            "angular_spread_degrees":self.angular_spread_degrees,"source_span_ns":self.source_span_ns.to_string(),
            "unique_sources":self.unique_sources,"motion_model":"local image similarity, not measured 3D camera pose",
            "probability":null,"independent_sign_evidence":false,"authorizes_calibration":false})
    }
}
#[derive(Debug, Default)]
pub(crate) struct MonocularContinuity {
    last_source: Option<u64>,
    epoch: Option<u64>,
    votes: VecDeque<(u64, usize)>,
    latest: Option<MonocularReport>,
}
impl MonocularContinuity {
    pub(crate) fn observe(
        &mut self,
        source: Option<u64>,
        epoch: u64,
        normals: [[f64; 3]; 2],
        residuals: [f64; 2],
        motion_sigma: Option<f64>,
    ) -> MonocularReport {
        let valid = normals
            .iter()
            .all(|n| dot3(*n, *n).is_finite() && (dot3(*n, *n) - 1.0).abs() < 1e-4);
        let mut r = MonocularReport {
            status: "motion-evidence-unavailable",
            preferred_candidate: None,
            angular_spread_degrees: if valid {
                dot3(normals[0], normals[1])
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            } else {
                0.0
            },
            source_span_ns: 0,
            unique_sources: 0,
        };
        let Some(source) = source else {
            r.status = "source-time-unavailable";
            return r;
        };
        if self.epoch.is_some_and(|old| old != epoch) {
            *self = Self::default();
        }
        self.epoch = Some(epoch);
        if self.last_source.is_some_and(|old| source < old) {
            r.status = "out-of-order-source";
            return r;
        }
        if self.last_source == Some(source) {
            return self.latest.unwrap_or(r);
        }
        if self
            .last_source
            .is_some_and(|old| source - old > MAX_GAP_NS)
        {
            self.votes.clear();
        }
        self.last_source = Some(source);
        while self
            .votes
            .front()
            .is_some_and(|(t, _)| source - *t > WINDOW_NS)
        {
            self.votes.pop_front();
        }
        if !valid {
            r.status = "invalid-directions";
        } else if r.angular_spread_degrees <= COALESCENCE_DEGREES {
            r.status = "coalesced-directions";
            self.votes.clear();
        } else if let Some(sigma) = motion_sigma.filter(|s| s.is_finite() && *s >= 0.0) {
            let margin = (4.0 * sigma).max(0.35);
            if residuals.iter().all(|v| v.is_finite() && *v >= 0.0)
                && (residuals[0] - residuals[1]).abs() > margin
            {
                let winner = usize::from(residuals[1] < residuals[0]);
                self.votes.push_back((source, winner));
                if self.votes.len() > MAX_SOURCES {
                    self.votes.pop_front();
                }
                let support = self.votes.iter().filter(|(_, v)| *v == winner).count();
                r.unique_sources = self.votes.len();
                r.source_span_ns = source - self.votes.front().unwrap().0;
                r.status = "collecting-local-motion";
                if support >= 6
                    && support >= self.votes.len() - support + 2
                    && r.source_span_ns >= 1_000_000_000
                {
                    r.status = "preferred-local-motion-only";
                    r.preferred_candidate = Some(winner);
                }
            } else {
                r.status = "ambiguous-trajectories";
            }
        }
        self.latest = Some(r);
        r
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) centers: [[f64; 3]; 2],
    pub(crate) normals: [[f64; 3]; 2],
    pub(crate) image_cost: f64,
}

impl Candidate {
    pub(crate) fn from_solution(s: &JointConicSolution) -> Option<Self> {
        if s.contributing_eyes != [true; 2] {
            return None;
        }
        Some(Self {
            centers: [s.eye_centers_camera_mm[0]?, s.eye_centers_camera_mm[1]?],
            normals: [s.eye_normals[0]?, s.eye_normals[1]?],
            image_cost: s.robust_cost,
        })
    }

    fn invariant(&self) -> Option<[f64; 4]> {
        if !self.image_cost.is_finite()
            || self.image_cost < 0.0
            || !self.centers.iter().flatten().all(|x| x.is_finite())
        {
            return None;
        }
        let b = normalized3(sub3(self.centers[1], self.centers[0]))?;
        let [l, r] = self.normals;
        // Reject invalid unit normals instead of silently legitimizing them.
        if [l, r]
            .into_iter()
            .any(|n| !dot3(n, n).is_finite() || (dot3(n, n) - 1.0).abs() > 1e-4)
        {
            return None;
        }
        Some([dot3(l, b), dot3(r, b), dot3(l, r), dot3(b, cross3(l, r))])
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Report {
    pub(crate) source_ns: u64,
    pub(crate) candidate_normals: Vec<[[f64; 3]; 2]>,
    pub(crate) status: &'static str,
    pub(crate) preferred_candidate: Option<usize>,
    pub(crate) candidate_scores: Vec<f64>,
    pub(crate) angular_spread_degrees: Option<f64>,
    pub(crate) unique_sources: usize,
    pub(crate) source_span_ns: u64,
    pub(crate) candidate_scope: &'static str,
}

impl Report {
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "source_ns":self.source_ns.to_string(),"candidate_normals":self.candidate_normals,
            "status":self.status, "preferred_candidate":self.preferred_candidate,
            "candidate_scores":self.candidate_scores,
            "angular_spread_degrees":self.angular_spread_degrees,
            "unique_sources":self.unique_sources, "source_span_ns":self.source_span_ns.to_string(),
            "candidate_scope":self.candidate_scope,
            "score_kind":"bounded image objective plus camera-invariant temporal preference",
            "probability":null, "independent_sign_evidence":false,
            "authorizes_calibration":false, "experimental":true,
        })
    }
}

struct Frame {
    source: u64,
    candidates: Vec<Candidate>,
    invariants: Vec<[f64; 4]>,
}

#[derive(Default)]
pub(crate) struct ContinuousGazeSign {
    frames: VecDeque<Frame>,
    scope: Option<&'static str>,
}

impl ContinuousGazeSign {
    pub(crate) fn observe(
        &mut self,
        source: u64,
        candidates: Vec<Candidate>,
        candidate_scope: &'static str,
    ) -> Report {
        let mut report = Report {
            source_ns: source,
            candidate_normals: Vec::new(),
            status: "collecting",
            preferred_candidate: None,
            candidate_scores: Vec::new(),
            angular_spread_degrees: None,
            unique_sources: 0,
            source_span_ns: 0,
            candidate_scope,
        };
        if self.frames.back().is_some_and(|f| source < f.source) {
            report.status = "out-of-order-source";
            return report;
        }
        if self.scope.is_some_and(|old| old != candidate_scope) {
            self.frames.clear();
        }
        self.scope = Some(candidate_scope);
        if self
            .frames
            .back()
            .is_some_and(|f| source - f.source > MAX_GAP_NS)
        {
            self.frames.clear();
        }
        let invariants = candidates
            .iter()
            .map(Candidate::invariant)
            .collect::<Option<Vec<_>>>();
        let Some(invariants) =
            invariants.filter(|_| !candidates.is_empty() && candidates.len() <= MAX_CANDIDATES)
        else {
            report.status = "missing-binocular-evidence";
            return report;
        };
        // Replace same-exposure revisions; never count a partner as another vote.
        if self.frames.back().is_some_and(|f| source == f.source) {
            self.frames.pop_back();
        }
        self.frames.push_back(Frame {
            source,
            candidates,
            invariants,
        });
        while self.frames.len() > MAX_SOURCES
            || self
                .frames
                .front()
                .is_some_and(|f| source - f.source > WINDOW_NS)
        {
            self.frames.pop_front();
        }
        report.unique_sources = self.frames.len();
        report.source_span_ns = source - self.frames.front().unwrap().source;

        // Recompute the bounded window so an ancient winner cannot become an
        // irreversible branch lock. Candidate array positions are not identities.
        let mut scores = Vec::<f64>::new();
        let mut previous: Option<&Frame> = None;
        for frame in &self.frames {
            let dt = previous.map_or(0.1, |p| (frame.source - p.source) as f64 / 1e9);
            let weight = (dt / 0.1).min(1.0);
            let best_image = frame
                .candidates
                .iter()
                .map(|c| c.image_cost)
                .fold(f64::INFINITY, f64::min);
            let next = frame
                .candidates
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let transition = previous.map_or(0.0, |p| {
                        p.invariants
                            .iter()
                            .enumerate()
                            .map(|(j, prior)| {
                                scores[j]
                                    + weight * transition_cost(*prior, frame.invariants[i], dt)
                            })
                            .fold(f64::INFINITY, f64::min)
                    });
                    transition + weight * (c.image_cost - best_image).min(25.0)
                })
                .collect::<Vec<_>>();
            let minimum = next.iter().copied().fold(f64::INFINITY, f64::min);
            scores = next.into_iter().map(|v| v - minimum).collect();
            previous = Some(frame);
        }
        let current = self.frames.back().unwrap();
        report.candidate_normals = current.candidates.iter().map(|c| c.normals).collect();
        let best = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let spread = current
            .candidates
            .iter()
            .map(|c| angular_distance(c, &current.candidates[best]))
            .fold(0.0, f64::max);
        report.angular_spread_degrees = Some(spread);
        let rival = current
            .candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| angular_distance(c, &current.candidates[best]) > COALESCENCE_DEGREES)
            .map(|(i, _)| scores[i])
            .fold(f64::INFINITY, f64::min);
        report.candidate_scores = scores;
        report.status = if current.candidates.len() < 2 {
            "alternatives-missing"
        } else if spread <= COALESCENCE_DEGREES {
            "coalesced-directions"
        } else if report.unique_sources < 6 || report.source_span_ns < 1_000_000_000 {
            "collecting"
        } else if rival < MARGIN {
            "ambiguous-trajectories"
        } else {
            report.preferred_candidate = Some(best);
            "preferred-conditional-trajectory"
        };
        report
    }
}

fn transition_cost(a: [f64; 4], b: [f64; 4], dt: f64) -> f64 {
    // Soft, bounded preference; rapid genuine gaze changes stay possible.
    // Camera motion itself has zero cost when fitted geometry is transported
    // rigidly. Real re-estimation errors and occlusion do not enjoy that theorem.
    let allowance = (8.0 + 120.0 * dt).to_radians();
    let angles = (0..3)
        .map(|i| (a[i].clamp(-1.0, 1.0).acos() - b[i].clamp(-1.0, 1.0).acos()).powi(2))
        .sum::<f64>();
    ((angles + (a[3] - b[3]).powi(2)) / (allowance * allowance)).min(4.0)
}

fn angular_distance(a: &Candidate, b: &Candidate) -> f64 {
    (0..2)
        .map(|eye| {
            dot3(a.normals[eye], b.normals[eye])
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        })
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn single_eye_requires_source_motion_and_never_counts_redraws() {
        let mut mono = MonocularContinuity::default();
        let normals = [
            normalized3([0.0, 0.4, 1.0]).unwrap(),
            normalized3([0.0, -0.4, 1.0]).unwrap(),
        ];
        for i in 0..10 {
            let at = i * 200_000_000;
            let a = mono.observe(Some(at), 1, normals, [0.1, 3.0], Some(0.1));
            let b = mono.observe(Some(at), 1, normals, [3.0, 0.1], Some(0.1));
            assert_eq!(a, b, "redraws cannot revise motion votes");
            if i >= 5 {
                assert_eq!(a.preferred_candidate, Some(0));
            }
        }
        let unavailable = mono.observe(Some(2_000_000_000), 1, normals, [0.1, 3.0], None);
        assert_eq!(unavailable.status, "motion-evidence-unavailable");
        assert_eq!(unavailable.preferred_candidate, None);
        let reset = mono.observe(Some(2_200_000_000), 2, normals, [0.1, 3.0], Some(0.1));
        assert_eq!(reset.unique_sources, 1);
        assert_eq!(reset.preferred_candidate, None);
        let coalesced = mono.observe(
            Some(2_400_000_000),
            2,
            [[0.0, 0.0, 1.0]; 2],
            [0.1, 3.0],
            Some(0.1),
        );
        assert_eq!(coalesced.status, "coalesced-directions");
        assert_eq!(coalesced.preferred_candidate, None);
        let gap = mono.observe(Some(5_000_000_000), 2, normals, [0.1, 3.0], Some(0.1));
        assert_eq!(gap.unique_sources, 1);
    }
    fn candidate(y: f64, cost: f64) -> Candidate {
        Candidate {
            centers: [[-32.0, 0.0, -350.0], [32.0, 0.0, -350.0]],
            normals: [
                normalized3([0.08, y, 1.0]).unwrap(),
                normalized3([-0.08, y, 1.0]).unwrap(),
            ],
            image_cost: cost,
        }
    }
    fn transform(mut c: Candidate, angle: f64, translation: [f64; 3]) -> Candidate {
        let rotate = |v: [f64; 3]| {
            let [x, y, z] = v;
            let p = angle * 0.37;
            let q = angle * 0.83;
            let [x, y, z] = [x, p.cos() * y - p.sin() * z, p.sin() * y + p.cos() * z];
            let [x, y, z] = [q.cos() * x + q.sin() * z, y, -q.sin() * x + q.cos() * z];
            [
                angle.cos() * x - angle.sin() * y,
                angle.sin() * x + angle.cos() * y,
                z,
            ]
        };
        c.normals = c.normals.map(rotate);
        c.centers = c
            .centers
            .map(|v| std::array::from_fn(|i| rotate(v)[i] + translation[i]));
        c
    }
    #[test]
    fn arbitrary_rigid_camera_motion_does_not_change_trajectory_scores() {
        let mut fixed = ContinuousGazeSign::default();
        let mut moving = ContinuousGazeSign::default();
        for i in 0..20 {
            let candidates = vec![
                candidate(0.2 + i as f64 * 0.015, 0.0),
                candidate(-0.2 - i as f64 * 0.015, 8.0),
            ];
            let rotated = candidates
                .iter()
                .cloned()
                .map(|c| {
                    transform(
                        c,
                        i as f64 * 0.7,
                        [i as f64 * 80.0, -100.0 * i as f64, 50.0],
                    )
                })
                .collect();
            let a = fixed.observe(i * 200_000_000, candidates, "test");
            let b = moving.observe(i * 200_000_000, rotated, "test");
            assert_eq!(a.status, b.status);
            assert_eq!(a.preferred_candidate, b.preferred_candidate);
            for (a, b) in a.candidate_scores.iter().zip(&b.candidate_scores) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    }
    #[test]
    fn smooth_mirror_histories_remain_ambiguous_far_from_camera_facing() {
        let mut tracker = ContinuousGazeSign::default();
        for i in 0..20 {
            let y = 0.4 + i as f64 * 0.01;
            let r = tracker.observe(
                i * 200_000_000,
                vec![candidate(y, 0.0), candidate(-y, 0.0)],
                "test",
            );
            assert_eq!(r.preferred_candidate, None);
            if i >= 5 {
                assert_eq!(r.status, "ambiguous-trajectories");
                assert!(r.angular_spread_degrees.unwrap() > 40.0);
            }
        }
    }
    #[test]
    fn coalescence_and_reseparation_do_not_invent_a_branch_identity() {
        let mut tracker = ContinuousGazeSign::default();
        for i in 0..8 {
            tracker.observe(
                i * 200_000_000,
                vec![candidate(0.4, 0.0), candidate(-0.4, 0.0)],
                "test",
            );
        }
        let at = 1_600_000_000;
        let near = tracker.observe(
            at,
            vec![candidate(0.01, 0.0), candidate(-0.01, 0.0)],
            "test",
        );
        assert_eq!(near.status, "coalesced-directions");
        assert_eq!(near.preferred_candidate, None);
        let split = tracker.observe(
            at + 200_000_000,
            vec![candidate(-0.4, 0.0), candidate(0.4, 0.0)],
            "test",
        );
        assert_eq!(split.status, "ambiguous-trajectories");
    }
    #[test]
    fn duplicates_bursts_missing_and_old_sources_never_create_support() {
        let mut tracker = ContinuousGazeSign::default();
        for _ in 0..20 {
            let r = tracker.observe(
                1_000,
                vec![candidate(0.4, 0.0), candidate(-0.4, 9.0)],
                "test",
            );
            assert_eq!(r.unique_sources, 1);
            assert_eq!(r.preferred_candidate, None);
        }
        for i in 1..20 {
            let r = tracker.observe(
                1_000 + i,
                vec![candidate(0.4, 0.0), candidate(-0.4, 9.0)],
                "test",
            );
            assert_eq!(r.preferred_candidate, None);
        }
        assert_eq!(
            tracker.observe(0, vec![], "test").status,
            "out-of-order-source"
        );
        assert_eq!(
            tracker.observe(2_000, vec![], "test").status,
            "missing-binocular-evidence"
        );
        let mut bad = candidate(0.4, 0.0);
        bad.normals[1] = [f64::NAN; 3];
        assert_eq!(
            tracker.observe(3_000, vec![bad], "test").status,
            "missing-binocular-evidence"
        );
        let after = tracker.observe(
            2_000_000_000,
            vec![candidate(0.4, 0.0), candidate(-0.4, 9.0)],
            "test",
        );
        assert_eq!(after.unique_sources, 1);
        assert_eq!(after.preferred_candidate, None);
    }
    #[test]
    fn new_evidence_can_change_preference_without_a_permanent_sign_lock() {
        let mut tracker = ContinuousGazeSign::default();
        for i in 0..40 {
            let (a, b) = if i < 15 { (0.0, 9.0) } else { (9.0, 0.0) };
            let r = tracker.observe(
                i * 200_000_000,
                vec![candidate(0.4, a), candidate(-0.4, b)],
                "test",
            );
            if i == 14 {
                assert_eq!(r.preferred_candidate, Some(0));
            }
            if i == 39 {
                assert_eq!(r.preferred_candidate, Some(1));
            }
        }
    }

    #[test]
    fn hypothesis_order_is_not_a_branch_identity_and_changed_priors_reset_history() {
        let mut tracker = ContinuousGazeSign::default();
        for i in 0..15 {
            let mut c = vec![candidate(0.4, 0.0), candidate(-0.4, 9.0)];
            if i % 2 == 1 {
                c.reverse();
            }
            let r = tracker.observe(i * 200_000_000, c, "unconditioned");
            if i >= 5 {
                assert_eq!(r.preferred_candidate, Some(i as usize % 2));
            }
        }
        let r = tracker.observe(
            3_000_000_000,
            vec![candidate(0.4, 0.0), candidate(-0.4, 9.0)],
            "screen-reference",
        );
        assert_eq!(r.unique_sources, 1);
        assert_eq!(r.preferred_candidate, None);
    }
}
