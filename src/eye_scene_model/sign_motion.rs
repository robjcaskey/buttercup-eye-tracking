//! Bounded single-eye evidence for revising a previously selected tilt branch.
//!
//! Each residual compares one branch's CURRENT implied pivot to that SAME
//! branch's PREVIOUS pivot, transported by independently observed RAW motion.
//! Never compare the alternative only to the selected branch's signed history:
//! that manufactures a saccade when correcting a bad initial choice.
//! Thresholds are engineering support margins, not calibrated probabilities
//! or hard anatomical pivot constraints. Missing transport cannot vote identity.

use std::collections::VecDeque;
use std::time::Instant;

const MAX_STEPS: usize = 12;
const HORIZON_SECONDS: f64 = 1.25;
const MAX_INTERVAL_SECONDS: f64 = 0.75;
const MIN_SUPPORT: usize = 4;

#[derive(Clone, Copy, Debug)]
struct Stamp {
    source_ns: Option<u64>,
    now: Instant,
}

impl Stamp {
    fn elapsed(self, previous: Self) -> Option<f64> {
        match (previous.source_ns, self.source_ns) {
            (Some(a), Some(b)) => b.checked_sub(a).map(|dt| dt as f64 * 1e-9),
            (None, None) => self
                .now
                .checked_duration_since(previous.now)
                .map(|dt| dt.as_secs_f64()),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Step {
    stamp: Stamp,
    costs: [f64; 2],
    winner: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MotionSignDecision {
    pub(super) candidate: usize,
    pub(super) support: usize,
    pub(super) mean_costs: [f64; 2],
}

#[derive(Debug, Default)]
pub(crate) struct MotionSignWindow {
    previous: Option<Stamp>,
    steps: VecDeque<Step>,
}

impl MotionSignWindow {
    pub(super) fn clear(&mut self) {
        *self = Self::default();
    }

    /// Residuals compare persistent branch identities, never selected/alternative
    /// ordering. They must be absent for missing/untrusted transport or
    /// near-frontal/invalid geometry. Missing transport does NOT mean identity.
    pub(super) fn observe(
        &mut self,
        source_ns: Option<u64>,
        now: Instant,
        residuals_px: Option<[f64; 2]>,
        radius_px: f64,
        transport_residual_px: f64,
    ) -> Option<MotionSignDecision> {
        let stamp = Stamp { source_ns, now };
        let Some(previous) = self.previous else {
            self.previous = Some(stamp);
            return None;
        };
        let dt = stamp.elapsed(previous);
        // Duplicate/late results neither accumulate nor refresh votes.
        if dt.is_some_and(|dt| dt == 0.0)
            || matches!((previous.source_ns, source_ns), (Some(a), Some(b)) if b < a)
        {
            return None;
        }
        self.previous = Some(stamp);
        if !dt.is_some_and(|dt| (0.001..=MAX_INTERVAL_SECONDS).contains(&dt)) {
            self.steps.clear();
            return None;
        }
        let Some(residuals) = residuals_px.filter(|r| {
            r.iter().all(|v| v.is_finite() && *v >= 0.0)
                && radius_px.is_finite()
                && radius_px > 0.0
                && transport_residual_px.is_finite()
                && transport_residual_px >= 0.0
        }) else {
            self.steps.clear();
            return None;
        };
        // Permit small effective-pivot drift and noisy fit/transport. These
        // margins are defeasible, not a rigid anatomical hinge constraint.
        let allowance = (2.0 * transport_residual_px).max(0.5) + 0.015 * radius_px;
        let costs = residuals.map(|r| (r / allowance).min(4.0));
        let best = usize::from(costs[1] < costs[0]);
        let winner = (costs[best] <= 2.0 && costs[1 - best] - costs[best] >= 1.0).then_some(best);
        self.steps.push_back(Step {
            stamp,
            costs,
            winner,
        });
        while self.steps.len() > MAX_STEPS
            || self.steps.front().is_some_and(|step| {
                !stamp
                    .elapsed(step.stamp)
                    .is_some_and(|age| age <= HORIZON_SECONDS)
            })
        {
            self.steps.pop_front();
        }
        // Require four actually discriminating source intervals, not four
        // re-scorings of one noisy fit. A neutral current frame cannot reissue
        // old evidence. Overlapping windows are not independent observations.
        let candidate = winner?;
        let support = self
            .steps
            .iter()
            .filter(|s| s.winner == Some(candidate))
            .count();
        let against = self
            .steps
            .iter()
            .filter(|s| s.winner == Some(1 - candidate))
            .count();
        if support < MIN_SUPPORT || against * 3 > support {
            return None;
        }
        let mean_costs = [0, 1]
            .map(|i| self.steps.iter().map(|s| s.costs[i]).sum::<f64>() / self.steps.len() as f64);
        (mean_costs[1 - candidate] - mean_costs[candidate] >= 0.75).then_some(MotionSignDecision {
            candidate,
            support,
            mean_costs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eye_scene_model::{quantize_frontal_disk_area, SurfaceGazeTracker};
    use crate::raw_iris_focus::{OuterIrisBoundary, OuterIrisPoint};
    use crate::roi_evidence::{NativeGlobalSimilarityEvidence, SimilarityMotion};
    use std::time::Duration;

    #[test]
    fn single_eye_motion_corrects_a_resolved_wrong_sign_in_each_direction_and_scale() {
        for major_angle in [0.0, std::f64::consts::FRAC_PI_2, 0.6] {
            for sign in [-1.0, 1.0] {
                for radius in [30.0, 60.0, 100.0] {
                    for slope in [-0.025, 0.025] {
                        for wrong_seed in [false, true] {
                            let axis = (-major_angle.sin() * sign, major_angle.cos() * sign);
                            let area = std::f64::consts::PI * radius * radius;
                            let (_, bucket) = quantize_frontal_disk_area(area).unwrap();
                            let face_radius = (bucket / std::f64::consts::PI).sqrt();
                            let depth = face_radius * (1.83_f64.powi(2) - 1.0).sqrt();
                            let mut tracker = SurfaceGazeTracker::default();
                            let started = Instant::now();
                            let mut initial_epoch = 0;
                            let mut correction_count = 0;
                            for frame in 0..24u64 {
                                let tilt = if slope > 0.0 { 0.30 } else { 0.80 }
                                    + frame.saturating_sub(3) as f64 * slope;
                                // Whole head motion and an in-sensor ROI nudge must
                                // cancel; a small extra pivot drift remains allowed.
                                let head = (frame as f64 * 2.0, frame as f64 * -1.5);
                                let origin = if frame < 9 {
                                    (3000, 1500)
                                } else {
                                    (3030, 1480)
                                };
                                let sensor = (
                                    3200.0 + head.0 + depth * axis.0 * tilt.sin(),
                                    1700.0
                                        + head.1
                                        + depth * axis.1 * tilt.sin()
                                        + 0.04 * frame as f64,
                                );
                                let outer = OuterIrisBoundary {
                                    center: (
                                        sensor.0 - f64::from(origin.0),
                                        sensor.1 - f64::from(origin.1),
                                    ),
                                    major_radius: radius,
                                    minor_radius: radius * tilt.cos(),
                                    angle: major_angle
                                        + if frame % 2 == 0 {
                                            0.0
                                        } else {
                                            std::f64::consts::PI
                                        },
                                    points: vec![OuterIrisPoint::default(); 8],
                                    ..OuterIrisBoundary::default()
                                };
                                let global = NativeGlobalSimilarityEvidence {
                                    reliable: true,
                                    motion: SimilarityMotion {
                                        translation: [2.0, -1.5],
                                        residual: 0.1,
                                        support: 16,
                                        ..SimilarityMotion::default()
                                    },
                                    motion_center_sensor: [3200.0, 1700.0],
                                    ..NativeGlobalSimilarityEvidence::default()
                                };
                                // Exercise both correct and persistently misleading
                                // seeds, moving toward and away from camera-normal.
                                let seed = if wrong_seed { -0.1 } else { 0.1 };
                                let sample = tracker
                                    .observe_keyed_with_global_similarity(
                                        1_000_000_000 + frame * 100_000_000,
                                        started + Duration::from_millis(frame * 100),
                                        origin,
                                        Some((seed * axis.0, seed * axis.1)),
                                        &outer,
                                        (frame > 0).then_some(global),
                                    )
                                    .unwrap();
                                let dot = sample.relative_gaze.right * axis.0
                                    + sample.relative_gaze.down * axis.1;
                                if frame == 3 {
                                    assert!(sample.sign_resolved && (dot < 0.0) == wrong_seed);
                                    initial_epoch = sample.sign_epoch;
                                }
                                correction_count +=
                                    usize::from(sample.kinematic_sign_correction[0]);
                                if frame >= 12 {
                                    assert!(dot > 0.0, "angle={major_angle} sign={sign} radius={radius} frame={frame} {tracker:?}");
                                    assert_eq!(
                                        sample.sign_epoch,
                                        initial_epoch + u64::from(wrong_seed)
                                    );
                                }
                                assert!((sample.frontal_equivalent_disk_area_px2 - area).abs() < 1e-7);
                                assert!(sample.relative_gaze.is_camera_facing());
                            }
                            assert_eq!(correction_count, usize::from(wrong_seed));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn motion_sign_window_needs_four_fresh_consistent_intervals() {
        for candidate in [0, 1] {
            let mut window = MotionSignWindow::default();
            let now = Instant::now();
            let r = if candidate == 0 {
                [0.3, 8.0]
            } else {
                [8.0, 0.3]
            };
            for frame in 0..=4 {
                let stamp = 1_000_000_000 + frame * 100_000_000;
                let decision = window.observe(Some(stamp), now, Some(r), 80.0, 0.2);
                assert_eq!(
                    decision.map(|d| d.candidate),
                    (frame == 4).then_some(candidate)
                );
                assert!(window
                    .observe(Some(stamp), now, Some(r), 80.0, 0.2)
                    .is_none());
                assert!(window
                    .observe(Some(stamp - 1), now, Some(r), 80.0, 0.2)
                    .is_none());
            }
        }
    }

    #[test]
    fn motion_sign_window_missing_motion_stale_votes_and_outliers_abstain() {
        for invalid in [
            None,
            Some([0.0, 0.0]),
            Some([20.0, 40.0]),
            Some([f64::NAN, 0.0]),
        ] {
            let mut window = MotionSignWindow::default();
            let now = Instant::now();
            for frame in 0..30 {
                assert!(window
                    .observe(Some(1 + frame * 100_000_000), now, invalid, 80.0, 0.2)
                    .is_none());
            }
        }
        let mut window = MotionSignWindow::default();
        let now = Instant::now();
        for frame in 0..4 {
            assert!(window
                .observe(
                    Some(1 + frame * 100_000_000),
                    now,
                    Some([0.0, 8.0]),
                    80.0,
                    0.2
                )
                .is_none());
        }
        assert!(window
            .observe(
                Some(2_000_000_000),
                now + Duration::from_secs(2),
                Some([0.0, 8.0]),
                80.0,
                0.2
            )
            .is_none());
        assert_eq!(window.steps.len(), 0);
    }

    #[test]
    fn motion_sign_window_fit_jitter_and_uncertain_transport_do_not_resolve() {
        for uncertain in [false, true] {
            let mut window = MotionSignWindow::default();
            let now = Instant::now();
            for frame in 0..30 {
                let r = if uncertain || frame % 2 == 0 {
                    [0.0, 8.0]
                } else {
                    [8.0, 0.0]
                };
                assert!(window
                    .observe(
                        Some(1 + frame * 100_000_000),
                        now,
                        Some(r),
                        80.0,
                        if uncertain { 8.0 } else { 0.2 }
                    )
                    .is_none());
            }
        }
    }
}
