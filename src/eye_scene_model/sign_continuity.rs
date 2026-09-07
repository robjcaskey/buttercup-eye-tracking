//! Local branch correspondence at a near-frontal crossing, not sign acquisition.
//!
//! Comparing only unit transverse directions makes an eye approaching camera
//! normal bounce off it if the actual crossing falls between exposures. An
//! already signed trajectory may continue through that degeneracy when
//! source-time velocity and an independently transported effective pivot agree.
//! When the prior branches coincide within the transport support allowance,
//! their chosen direction is not a reliable velocity veto: stronger current
//! pivot transport distinguishes continuing from turning back. An arbitrary
//! near-circle ellipse-axis direction must not dictate that choice.
//! The pivot remains approximate. All support margins below are heuristic.
use super::{ContactSignHypothesis, GazeKinematicFrame};
use std::collections::VecDeque;

const MAX_TRANSVERSE_NORMAL: f64 = 0.12;
const MAX_INTERVAL_SECONDS: f64 = 0.75;

pub(super) fn near_frontal_assignment(
    history: &VecDeque<GazeKinematicFrame>,
    source_ns: Option<u64>,
    center: (f64, f64),
    radius: f64,
    candidates: [ContactSignHypothesis; 2],
    predicted_pivots: [(f64, f64); 2],
    transport_residual: f64,
    selected: usize,
    direction_assignment: [usize; 2],
) -> Option<[usize; 2]> {
    let transported_branch_separation_px = (predicted_pivots[0].0 - predicted_pivots[1].0)
        .hypot(predicted_pivots[0].1 - predicted_pivots[1].1);
    if history.len() < 2
        || !radius.is_finite()
        || radius <= 0.0
        || !transport_residual.is_finite()
        || transport_residual < 0.0
        || !transported_branch_separation_px.is_finite()
        || transported_branch_separation_px < 0.0
    {
        return None;
    }
    let last = history[history.len() - 1];
    let previous = history[history.len() - 2];
    let dt = source_ns?.checked_sub(last.source_timestamp_ns?)? as f64 * 1e-9;
    let history_dt = last
        .source_timestamp_ns?
        .checked_sub(previous.source_timestamp_ns?)? as f64
        * 1e-9;
    let last_magnitude = last.projected_gaze.0.hypot(last.projected_gaze.1);
    if !(0.001..=MAX_INTERVAL_SECONDS).contains(&dt)
        || !(0.001..=MAX_INTERVAL_SECONDS).contains(&history_dt)
        || dt / history_dt > 2.5
        || last_magnitude > MAX_TRANSVERSE_NORMAL
    {
        return None;
    }
    let normals = candidates.map(|c| {
        (
            (c.near_surface_sensor_px.0 - center.0) / radius,
            (c.near_surface_sensor_px.1 - center.1) / radius,
        )
    });
    let magnitude = normals[0].0.hypot(normals[0].1);
    if !magnitude.is_finite() || magnitude <= 1e-9 || magnitude > MAX_TRANSVERSE_NORMAL {
        return None;
    }
    let support_margin_px = (2.0 * transport_residual).max(0.5);
    let previous_branches_distinguishable = transported_branch_separation_px > support_margin_px;
    // If the old pivots coalesce within that allowance, selecting one endpoint
    // gives an unsupported near-circle axis label leverage over the next frame.
    // Transport the center of their bounded support instead. Its half-width is
    // at most half the support margin; do not average the published gaze rays.
    let pivot_reference = if previous_branches_distinguishable {
        predicted_pivots[selected]
    } else {
        (
            (predicted_pivots[0].0 + predicted_pivots[1].0) * 0.5,
            (predicted_pivots[0].1 + predicted_pivots[1].1) * 0.5,
        )
    };
    let residuals = candidates.map(|c| {
        (c.effective_pivot_sensor_px.0 - pivot_reference.0)
            .hypot(c.effective_pivot_sensor_px.1 - pivot_reference.1)
    });
    if !residuals.iter().all(|r| r.is_finite()) {
        return None;
    }
    let best = usize::from(residuals[1] < residuals[0]);
    let allowance = support_margin_px + 0.015 * radius;
    if best == direction_assignment[selected]
        || residuals[best] > allowance
        || residuals[1 - best] - residuals[best] < support_margin_px
    {
        return None;
    }
    let predicted = (
        last.projected_gaze.0
            + (last.projected_gaze.0 - previous.projected_gaze.0) * dt / history_dt,
        last.projected_gaze.1
            + (last.projected_gaze.1 - previous.projected_gaze.1) * dt / history_dt,
    );
    let errors = normals.map(|n| (n.0 - predicted.0).hypot(n.1 - predicted.1));
    // Require both cues when the preceding branches are distinguishable at
    // the available RAW transport precision. Inside that allowance the previous
    // direction can be arbitrary; extrapolating it must not veto a supported
    // continuation or turnaround. Current pivots must still be separated by
    // MORE than that same margin, with a small absolute residual for the winner.
    if !errors.iter().all(|e| e.is_finite())
        || (previous_branches_distinguishable && errors[best] + 1e-9 >= errors[1 - best])
        || last.projected_gaze.0 * normals[best].0 + last.projected_gaze.1 * normals[best].1 > 0.0
    {
        return None;
    }
    Some(if selected == 0 {
        [best, 1 - best]
    } else {
        [1 - best, best]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn history() -> VecDeque<GazeKinematicFrame> {
        [-0.03, -0.01]
            .into_iter()
            .enumerate()
            .map(|(i, x)| GazeKinematicFrame {
                observed_at: Instant::now(),
                source_timestamp_ns: Some(1_000_000_000 + i as u64 * 100_000_000),
                ellipse_center_sensor: (1000.0 + 123.0 * x, 2000.0),
                implied_pivot_sensor_px: (1000.0, 2000.0),
                projected_gaze: (x, 0.0),
            })
            .collect()
    }

    fn candidates() -> [ContactSignHypothesis; 2] {
        [0.01, -0.01].map(|x| ContactSignHypothesis {
            effective_pivot_sensor_px: (1001.23 - 123.0 * x, 2000.0),
            near_surface_sensor_px: (1001.23 + 80.0 * x, 2000.0),
            residual_ema: 0.0,
            observations: 1,
        })
    }

    fn decide(
        history: &VecDeque<GazeKinematicFrame>,
        source: Option<u64>,
        pivot: (f64, f64),
        residual: f64,
    ) -> Option<[usize; 2]> {
        near_frontal_assignment(
            history,
            source,
            (1001.23, 2000.0),
            80.0,
            candidates(),
            [pivot, (pivot.0 - 2.46, pivot.1)],
            residual,
            0,
            [1, 0],
        )
    }

    #[test]
    fn continuity_needs_fresh_source_time_and_agreement_of_independent_cues() {
        let h = history();
        assert_eq!(
            decide(&h, Some(1_200_000_000), (1000.0, 2000.0), 0.1),
            Some([0, 1])
        );
        for source in [
            None,
            Some(1_100_000_000),
            Some(1_000_000_000),
            Some(3_000_000_000),
        ] {
            assert!(decide(&h, source, (1000.0, 2000.0), 0.1).is_none());
        }
        let mut unknown_clock = h.clone();
        unknown_clock[0].source_timestamp_ns = None;
        assert!(decide(&unknown_clock, Some(1_200_000_000), (1000.0, 2000.0), 0.1).is_none());
        let mut opposing_velocity = h.clone();
        opposing_velocity[0].projected_gaze = (0.01, 0.0);
        assert!(
            decide(
                &opposing_velocity,
                Some(1_200_000_000),
                (1000.0, 2000.0),
                0.1
            )
            .is_none()
        );
        for residual in [8.0, f64::NAN, -1.0] {
            assert!(decide(&h, Some(1_200_000_000), (1000.0, 2000.0), residual).is_none());
        }
        assert!(decide(&h, Some(1_200_000_000), (1100.0, 2000.0), 0.1).is_none());
    }

    #[test]
    fn continuity_preserves_either_slot_and_is_not_a_general_pivot_proximity_rematch() {
        let h = history();
        assert_eq!(
            near_frontal_assignment(
                &h,
                Some(1_200_000_000),
                (1001.23, 2000.0),
                80.0,
                candidates(),
                [(997.54, 2000.0), (1000.0, 2000.0)],
                0.1,
                1,
                [0, 1]
            ),
            Some([1, 0])
        );
        let mut distant = h.clone();
        distant[1].projected_gaze = (-0.3, 0.0);
        assert!(decide(&distant, Some(1_200_000_000), (1000.0, 2000.0), 0.1).is_none());
        let mut jitter = h.clone();
        jitter[0].projected_gaze = jitter[1].projected_gaze;
        assert!(decide(&jitter, Some(1_200_000_000), (1000.0, 2000.0), 0.1).is_none());
    }

    #[test]
    fn coalescent_prior_support_does_not_privilege_an_arbitrary_endpoint() {
        let mut h = history();
        h[0].projected_gaze = (0.015, 0.008);
        h[1].projected_gaze = (0.005, -0.008);
        let candidates_at = |normal: (f64, f64)| {
            let center = (1000.0 + 123.0 * normal.0, 2000.0 + 123.0 * normal.1);
            let candidates = [1.0, -1.0].map(|sign| ContactSignHypothesis {
                effective_pivot_sensor_px: (
                    center.0 - 123.0 * sign * normal.0,
                    center.1 - 123.0 * sign * normal.1,
                ),
                near_surface_sensor_px: (
                    center.0 + 80.0 * sign * normal.0,
                    center.1 + 80.0 * sign * normal.1,
                ),
                residual_ema: 0.0,
                observations: 1,
            });
            (center, candidates)
        };
        let predicted = [(998.77, 2001.968), (1000.0, 2000.0)];
        let (center, candidates) = candidates_at((-0.025, 0.008));
        for pivots in [predicted, [predicted[1], predicted[0]]] {
            assert_eq!(
                near_frontal_assignment(
                    &h,
                    Some(1_200_000_000),
                    center,
                    80.0,
                    candidates,
                    pivots,
                    1.5,
                    0,
                    [1, 0]
                ),
                Some([0, 1])
            );
        }
        // No transfer if the current branches are still indistinguishable:
        // neither a velocity prediction nor a prior endpoint may invent support.
        let (center, candidates) = candidates_at((-0.005, 0.008));
        assert!(
            near_frontal_assignment(
                &h,
                Some(1_200_000_000),
                center,
                80.0,
                candidates,
                predicted,
                1.5,
                0,
                [1, 0]
            )
            .is_none()
        );
        assert!(
            near_frontal_assignment(
                &h,
                Some(1_200_000_000),
                center,
                80.0,
                candidates,
                [(f64::NAN, 2000.0), predicted[1]],
                1.5,
                0,
                [1, 0]
            )
            .is_none()
        );
    }
}
