//! Lossless, production-code replay for comparing the three native RAW iris
//! segmenters on exactly the same recorded frames.  This is a child module of
//! `buttercup_wayland_raw_eyes`, so Driving uses the viewer's real private scorer
//! and admission state instead of a Python approximation.

use super::*;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::env;
use std::io::{Read as IoRead, Seek as IoSeek, SeekFrom};

#[derive(Default)]
struct ModelAggregate {
    candidates: usize,
    admitted: usize,
    radii: Vec<f64>,
    relative_radius_steps: Vec<f64>,
    center_steps_sensor_px: Vec<f64>,
    elapsed_ms: Vec<f64>,
    previous: Option<((f64, f64), f64)>,
}

impl ModelAggregate {
    fn observe(
        &mut self,
        candidate: bool,
        admitted: bool,
        center_local: Option<(f64, f64)>,
        radius: Option<f64>,
        sensor_origin: (u32, u32),
        elapsed_ms: f64,
    ) {
        self.candidates += usize::from(candidate);
        self.admitted += usize::from(admitted);
        self.elapsed_ms.push(elapsed_ms);
        let current = center_local.zip(radius).map(|(center, radius)| {
            (
                (
                    center.0 + f64::from(sensor_origin.0),
                    center.1 + f64::from(sensor_origin.1),
                ),
                radius,
            )
        });
        if admitted {
            if let Some((center, radius)) = current {
                self.radii.push(radius);
                if let Some((previous_center, previous_radius)) = self.previous {
                    if previous_radius > 1.0e-9 {
                        self.relative_radius_steps
                            .push(((radius - previous_radius) / previous_radius).abs());
                    }
                    self.center_steps_sensor_px
                        .push((center.0 - previous_center.0).hypot(center.1 - previous_center.1));
                }
                self.previous = Some((center, radius));
            } else {
                self.previous = None;
            }
        } else {
            self.previous = None;
        }
    }

    fn json(&self, frames: usize) -> Value {
        json!({
            "candidate_frames": self.candidates,
            "candidate_fraction": self.candidates as f64 / frames.max(1) as f64,
            "admitted_frames": self.admitted,
            "admitted_fraction": self.admitted as f64 / frames.max(1) as f64,
            "frontal_parallel_radius_px": distribution(&self.radii),
            "consecutive_admitted_radius_relative_step": {
                "samples": self.relative_radius_steps.len(),
                "median": percentile(&self.relative_radius_steps, 0.50),
                "p95": percentile(&self.relative_radius_steps, 0.95),
                "maximum": percentile(&self.relative_radius_steps, 1.0),
                "over_5_percent": self.relative_radius_steps.iter().filter(|step| **step > 0.05).count(),
            },
            "consecutive_admitted_center_step_sensor_px": distribution(&self.center_steps_sensor_px),
            "elapsed_ms": distribution(&self.elapsed_ms),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct PupilAffineTemporalObservation {
    timestamp_ns: u64,
    roi_center_sensor: (f64, f64),
    pupil_center_sensor: (f64, f64),
    pupil_center_canonical: (f64, f64),
    /// Image-axis pupil offset normalized by the de-affined limbus radius.
    /// This remains well-defined when a nearly circular ellipse swaps its
    /// arbitrary major-axis angle by approximately ninety degrees.
    pupil_center_outer_relative_image_axes: (f64, f64),
    limbus_projected_area_radius_px: f64,
    pupil_projected_area_radius_px: f64,
    pupil_to_limbus_area_fraction: f64,
}

fn pupil_affine_temporal_observation(
    timestamp_ns: u64,
    sensor_origin: (u32, u32),
    roi_size: (usize, usize),
    limbus_pose: DrivingAffinePose,
    pupil_center: (f64, f64),
    pupil_projected_area_radius_px: f64,
) -> Option<PupilAffineTemporalObservation> {
    let limbus_projected_area_radius_px =
        (limbus_pose.major_radius * limbus_pose.minor_radius).sqrt();
    let pupil_to_limbus_area_fraction =
        (pupil_projected_area_radius_px / limbus_projected_area_radius_px.max(1.0e-12)).powi(2);
    let pupil_center_canonical = driving_canonical_point(limbus_pose, pupil_center)?;
    let pupil_center_outer_relative_image_axes =
        driving_outer_relative_pupil_offset(limbus_pose, pupil_center)?;
    (limbus_projected_area_radius_px.is_finite()
        && limbus_projected_area_radius_px > 0.0
        && pupil_projected_area_radius_px.is_finite()
        && pupil_projected_area_radius_px > 0.0
        && pupil_to_limbus_area_fraction.is_finite()
        && pupil_to_limbus_area_fraction > 0.0)
        .then_some(PupilAffineTemporalObservation {
            timestamp_ns,
            roi_center_sensor: (
                f64::from(sensor_origin.0) + roi_size.0 as f64 * 0.5,
                f64::from(sensor_origin.1) + roi_size.1 as f64 * 0.5,
            ),
            pupil_center_sensor: (
                f64::from(sensor_origin.0) + pupil_center.0,
                f64::from(sensor_origin.1) + pupil_center.1,
            ),
            pupil_center_canonical,
            pupil_center_outer_relative_image_axes,
            limbus_projected_area_radius_px,
            pupil_projected_area_radius_px,
            pupil_to_limbus_area_fraction,
        })
}

#[derive(Clone, Copy, Debug)]
struct PupilAffineTemporalTransition {
    elapsed_seconds: f64,
    equivalent_radius_step_fraction: f64,
    area_fraction_log_step: f64,
    canonical_center_step: f64,
    image_axis_outer_relative_center_step: f64,
    first_log_area_derivative_per_second: f64,
    second_log_area_derivative_per_second2: Option<f64>,
    third_log_area_derivative_per_second3: Option<f64>,
    gross_reliable: bool,
    gross_translation_px: f64,
    gross_rotation_degrees: f64,
    gross_scale_delta: f64,
    gross_support: usize,
    gross_residual_px: f64,
    gross_compensated_pupil_radius_log_residual: Option<f64>,
    gross_transport_pupil_center_residual_px: Option<f64>,
}

fn pupil_affine_temporal_subset_json(transitions: Vec<&PupilAffineTemporalTransition>) -> Value {
    let equivalent_radius_steps = transitions
        .iter()
        .map(|transition| transition.equivalent_radius_step_fraction)
        .collect::<Vec<_>>();
    let area_log_steps = transitions
        .iter()
        .map(|transition| transition.area_fraction_log_step)
        .collect::<Vec<_>>();
    let canonical_center_steps = transitions
        .iter()
        .map(|transition| transition.canonical_center_step)
        .collect::<Vec<_>>();
    let image_axis_outer_relative_center_steps = transitions
        .iter()
        .map(|transition| transition.image_axis_outer_relative_center_step)
        .collect::<Vec<_>>();
    let gross_translation = transitions
        .iter()
        .map(|transition| transition.gross_translation_px)
        .collect::<Vec<_>>();
    let gross_rotation = transitions
        .iter()
        .map(|transition| transition.gross_rotation_degrees.abs())
        .collect::<Vec<_>>();
    let gross_scale = transitions
        .iter()
        .map(|transition| transition.gross_scale_delta.abs())
        .collect::<Vec<_>>();
    json!({
        "transitions": transitions.len(),
        "equivalent_radius_step_fraction": distribution(&equivalent_radius_steps),
        "equivalent_radius_step_over_2_percent_fraction": equivalent_radius_steps
            .iter()
            .filter(|step| **step > 0.02)
            .count() as f64
            / equivalent_radius_steps.len().max(1) as f64,
        "area_fraction_log_step": distribution(&area_log_steps),
        "canonical_pupil_center_step": distribution(&canonical_center_steps),
        "image_axis_outer_relative_pupil_center_step": distribution(
            &image_axis_outer_relative_center_steps,
        ),
        "gross_translation_px": distribution(&gross_translation),
        "gross_absolute_rotation_degrees": distribution(&gross_rotation),
        "gross_absolute_scale_delta": distribution(&gross_scale),
    })
}

#[derive(Default)]
struct PupilAffineTemporalAggregate {
    raw_diameter_frames: usize,
    observed_frames: usize,
    gap_resets: usize,
    previous: Option<PupilAffineTemporalObservation>,
    previous_first_derivative: Option<(f64, f64)>,
    previous_second_derivative: Option<(f64, f64)>,
    transitions: Vec<PupilAffineTemporalTransition>,
}

impl PupilAffineTemporalAggregate {
    fn observe(
        &mut self,
        raw_diameter_qualified: bool,
        observation: Option<PupilAffineTemporalObservation>,
        gross: raw_motion_octrees::NativeGlobalSimilarityEvidence,
    ) -> Value {
        self.raw_diameter_frames += usize::from(raw_diameter_qualified);
        let Some(current) = observation else {
            self.previous = None;
            self.previous_first_derivative = None;
            self.previous_second_derivative = None;
            return Value::Null;
        };
        self.observed_frames += 1;
        let mut transition_json = Value::Null;
        if let Some(previous) = self.previous {
            let elapsed_seconds =
                current.timestamp_ns.saturating_sub(previous.timestamp_ns) as f64 * 1.0e-9;
            if current.timestamp_ns > previous.timestamp_ns && elapsed_seconds <= 0.25 {
                let previous_log_area = previous.pupil_to_limbus_area_fraction.ln();
                let current_log_area = current.pupil_to_limbus_area_fraction.ln();
                let signed_log_step = current_log_area - previous_log_area;
                let first_derivative = signed_log_step / elapsed_seconds.max(1.0e-6);
                let second_derivative = self.previous_first_derivative.map(
                    |(previous_derivative, previous_elapsed)| {
                        (first_derivative - previous_derivative)
                            / (0.5 * (elapsed_seconds + previous_elapsed)).max(1.0e-6)
                    },
                );
                let third_derivative = second_derivative.zip(self.previous_second_derivative).map(
                    |(second_derivative, (previous_derivative, previous_elapsed))| {
                        (second_derivative - previous_derivative)
                            / (0.5 * (elapsed_seconds + previous_elapsed)).max(1.0e-6)
                    },
                );
                self.previous_first_derivative = Some((first_derivative, elapsed_seconds));
                self.previous_second_derivative =
                    second_derivative.map(|derivative| (derivative, elapsed_seconds));

                let gross_translation =
                    f64::from(gross.motion.translation[0].hypot(gross.motion.translation[1]));
                let gross_scale = 1.0 + f64::from(gross.motion.scale_delta);
                let gross_compensated_pupil_radius_log_residual =
                    (gross.reliable && gross_scale > 0.0).then(|| {
                        (current.pupil_projected_area_radius_px
                            / (previous.pupil_projected_area_radius_px * gross_scale).max(1.0e-12))
                        .ln()
                        .abs()
                    });
                let gross_transport_pupil_center_residual_px = gross.reliable.then(|| {
                    let x = previous.pupil_center_sensor.0 - previous.roi_center_sensor.0;
                    let y = previous.pupil_center_sensor.1 - previous.roi_center_sensor.1;
                    let predicted = (
                        previous.pupil_center_sensor.0
                            + f64::from(gross.motion.translation[0])
                            + f64::from(gross.motion.scale_delta) * x
                            - f64::from(gross.motion.rotation) * y,
                        previous.pupil_center_sensor.1
                            + f64::from(gross.motion.translation[1])
                            + f64::from(gross.motion.rotation) * x
                            + f64::from(gross.motion.scale_delta) * y,
                    );
                    (predicted.0 - current.pupil_center_sensor.0)
                        .hypot(predicted.1 - current.pupil_center_sensor.1)
                });
                let transition = PupilAffineTemporalTransition {
                    elapsed_seconds,
                    equivalent_radius_step_fraction: (0.5 * signed_log_step.abs()).exp() - 1.0,
                    area_fraction_log_step: signed_log_step.abs(),
                    canonical_center_step: (current.pupil_center_canonical.0
                        - previous.pupil_center_canonical.0)
                        .hypot(
                            current.pupil_center_canonical.1 - previous.pupil_center_canonical.1,
                        ),
                    image_axis_outer_relative_center_step: (current
                        .pupil_center_outer_relative_image_axes
                        .0
                        - previous.pupil_center_outer_relative_image_axes.0)
                        .hypot(
                            current.pupil_center_outer_relative_image_axes.1
                                - previous.pupil_center_outer_relative_image_axes.1,
                        ),
                    first_log_area_derivative_per_second: first_derivative,
                    second_log_area_derivative_per_second2: second_derivative,
                    third_log_area_derivative_per_second3: third_derivative,
                    gross_reliable: gross.reliable,
                    gross_translation_px: gross_translation,
                    gross_rotation_degrees: f64::from(gross.motion.rotation).to_degrees(),
                    gross_scale_delta: f64::from(gross.motion.scale_delta),
                    gross_support: gross.motion.support,
                    gross_residual_px: f64::from(gross.motion.residual),
                    gross_compensated_pupil_radius_log_residual,
                    gross_transport_pupil_center_residual_px,
                };
                transition_json = json!({
                    "elapsed_seconds": transition.elapsed_seconds,
                    "equivalent_radius_step_fraction": transition.equivalent_radius_step_fraction,
                    "area_fraction_log_step": transition.area_fraction_log_step,
                    "canonical_center_step": transition.canonical_center_step,
                    "image_axis_outer_relative_center_step": transition.image_axis_outer_relative_center_step,
                    "first_log_area_derivative_per_second": transition.first_log_area_derivative_per_second,
                    "second_log_area_derivative_per_second2": transition.second_log_area_derivative_per_second2,
                    "third_log_area_derivative_per_second3": transition.third_log_area_derivative_per_second3,
                    "gross_anatomy": {
                        "reliable": transition.gross_reliable,
                        "translation_px": transition.gross_translation_px,
                        "rotation_degrees": transition.gross_rotation_degrees,
                        "scale_delta": transition.gross_scale_delta,
                        "support": transition.gross_support,
                        "residual_px": transition.gross_residual_px,
                        "compensated_pupil_radius_log_residual": transition.gross_compensated_pupil_radius_log_residual,
                        "transport_pupil_center_residual_px": transition.gross_transport_pupil_center_residual_px,
                    },
                });
                self.transitions.push(transition);
            } else {
                self.gap_resets += 1;
                self.previous_first_derivative = None;
                self.previous_second_derivative = None;
            }
        }
        self.previous = Some(current);
        json!({
            "post_affine_definition": "pupil-to-limbus area fraction; common affine determinant cancels",
            "limbus_projected_area_radius_px": current.limbus_projected_area_radius_px,
            "pupil_projected_area_radius_px": current.pupil_projected_area_radius_px,
            "pupil_to_limbus_area_fraction": current.pupil_to_limbus_area_fraction,
            "equivalent_radius_ratio": current.pupil_to_limbus_area_fraction.sqrt(),
            "pupil_center_canonical": current.pupil_center_canonical,
            "pupil_center_outer_relative_image_axes": current.pupil_center_outer_relative_image_axes,
            "transition": transition_json,
        })
    }

    fn json(&self, frames: usize, observation_scope: &str) -> Value {
        let values = |map: fn(&PupilAffineTemporalTransition) -> Option<f64>| {
            self.transitions.iter().filter_map(map).collect::<Vec<_>>()
        };
        let equivalent_radius_steps =
            values(|transition| Some(transition.equivalent_radius_step_fraction));
        let area_log_steps = values(|transition| Some(transition.area_fraction_log_step));
        let canonical_center_steps = values(|transition| Some(transition.canonical_center_step));
        let image_axis_outer_relative_center_steps =
            values(|transition| Some(transition.image_axis_outer_relative_center_step));
        let first_derivatives =
            values(|transition| Some(transition.first_log_area_derivative_per_second.abs()));
        let second_derivatives = values(|transition| {
            transition
                .second_log_area_derivative_per_second2
                .map(f64::abs)
        });
        let third_derivatives = values(|transition| {
            transition
                .third_log_area_derivative_per_second3
                .map(f64::abs)
        });
        let compensated_radius =
            values(|transition| transition.gross_compensated_pupil_radius_log_residual);
        let transported_centers =
            values(|transition| transition.gross_transport_pupil_center_residual_px);
        let reliable = self
            .transitions
            .iter()
            .filter(|transition| transition.gross_reliable)
            .collect::<Vec<_>>();
        let unreliable = self
            .transitions
            .iter()
            .filter(|transition| !transition.gross_reliable)
            .collect::<Vec<_>>();
        let quasi_static = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable
                    && transition.gross_translation_px <= 1.0
                    && transition.gross_rotation_degrees.abs() <= 0.25
                    && transition.gross_scale_delta.abs() <= 0.005
            })
            .collect::<Vec<_>>();
        let moving = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable
                    && (transition.gross_translation_px > 1.0
                        || transition.gross_rotation_degrees.abs() > 0.25
                        || transition.gross_scale_delta.abs() > 0.005)
            })
            .collect::<Vec<_>>();
        let translation_below_one = self
            .transitions
            .iter()
            .filter(|transition| transition.gross_reliable && transition.gross_translation_px < 1.0)
            .collect::<Vec<_>>();
        let translation_one_to_three = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable && (1.0..3.0).contains(&transition.gross_translation_px)
            })
            .collect::<Vec<_>>();
        let translation_three_or_more = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable && transition.gross_translation_px >= 3.0
            })
            .collect::<Vec<_>>();
        let rotation_below_quarter = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable && transition.gross_rotation_degrees.abs() < 0.25
            })
            .collect::<Vec<_>>();
        let rotation_quarter_to_one = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable
                    && (0.25..1.0).contains(&transition.gross_rotation_degrees.abs())
            })
            .collect::<Vec<_>>();
        let rotation_one_or_more = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable && transition.gross_rotation_degrees.abs() >= 1.0
            })
            .collect::<Vec<_>>();
        let scale_below_half_percent = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable && transition.gross_scale_delta.abs() < 0.005
            })
            .collect::<Vec<_>>();
        let scale_half_to_two_percent = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable
                    && (0.005..0.02).contains(&transition.gross_scale_delta.abs())
            })
            .collect::<Vec<_>>();
        let scale_two_percent_or_more = self
            .transitions
            .iter()
            .filter(|transition| {
                transition.gross_reliable && transition.gross_scale_delta.abs() >= 0.02
            })
            .collect::<Vec<_>>();
        json!({
            "interpretation": "temporal self-consistency diagnostic only; stable wrong pupils do not establish accuracy",
            "observation_scope": observation_scope,
            "post_affine_area_definition": "q=(pupil_major*pupil_minor)/(limbus_major*limbus_minor)",
            "frames": frames,
            "raw_diameter_qualified_frames": self.raw_diameter_frames,
            "observed_curve_frames": self.observed_frames,
            "consecutive_observed_transitions": self.transitions.len(),
            "gap_resets": self.gap_resets,
            "equivalent_radius_step_fraction": {
                "distribution": distribution(&equivalent_radius_steps),
                "over_2_percent": equivalent_radius_steps.iter().filter(|step| **step > 0.02).count(),
            },
            "area_fraction_log_step": distribution(&area_log_steps),
            "canonical_pupil_center_step": distribution(&canonical_center_steps),
            "image_axis_outer_relative_pupil_center_step": distribution(
                &image_axis_outer_relative_center_steps,
            ),
            "absolute_first_log_area_derivative_per_second": distribution(&first_derivatives),
            "absolute_second_log_area_derivative_per_second2": distribution(&second_derivatives),
            "absolute_third_log_area_derivative_per_second3": distribution(&third_derivatives),
            "gross_reliable_transitions": self.transitions.iter().filter(|transition| transition.gross_reliable).count(),
            "gross_compensated_projected_pupil_radius_log_residual": distribution(&compensated_radius),
            "gross_transport_pupil_center_residual_sensor_px": distribution(&transported_centers),
            "conditional_on_gross_anatomy": {
                "reliable": pupil_affine_temporal_subset_json(reliable),
                "unreliable": pupil_affine_temporal_subset_json(unreliable),
                "joint_motion": {
                    "quasi_static_translation_le_1px_rotation_le_0_25deg_scale_le_0_005": pupil_affine_temporal_subset_json(quasi_static),
                    "above_quasi_static_envelope": pupil_affine_temporal_subset_json(moving),
                },
                "translation_px": {
                    "below_1": pupil_affine_temporal_subset_json(translation_below_one),
                    "1_to_below_3": pupil_affine_temporal_subset_json(translation_one_to_three),
                    "3_or_more": pupil_affine_temporal_subset_json(translation_three_or_more),
                },
                "absolute_rotation_degrees": {
                    "below_0_25": pupil_affine_temporal_subset_json(rotation_below_quarter),
                    "0_25_to_below_1": pupil_affine_temporal_subset_json(rotation_quarter_to_one),
                    "1_or_more": pupil_affine_temporal_subset_json(rotation_one_or_more),
                },
                "absolute_scale_delta": {
                    "below_0_005": pupil_affine_temporal_subset_json(scale_below_half_percent),
                    "0_005_to_below_0_02": pupil_affine_temporal_subset_json(scale_half_to_two_percent),
                    "0_02_or_more": pupil_affine_temporal_subset_json(scale_two_percent_or_more),
                },
            },
        })
    }
}

#[cfg(test)]
mod pupil_affine_temporal_metric_tests {
    use super::*;

    fn observation(
        timestamp_ns: u64,
        roi_center_sensor: (f64, f64),
        pupil_center_sensor: (f64, f64),
        limbus_radius: f64,
        pupil_radius: f64,
    ) -> PupilAffineTemporalObservation {
        PupilAffineTemporalObservation {
            timestamp_ns,
            roi_center_sensor,
            pupil_center_sensor,
            pupil_center_canonical: (0.12, -0.08),
            pupil_center_outer_relative_image_axes: (0.12, -0.08),
            limbus_projected_area_radius_px: limbus_radius,
            pupil_projected_area_radius_px: pupil_radius,
            pupil_to_limbus_area_fraction: (pupil_radius / limbus_radius).powi(2),
        }
    }

    #[test]
    fn common_affine_scale_cancels_from_pupil_area_fraction() {
        let mut aggregate = PupilAffineTemporalAggregate::default();
        let previous = observation(1_000_000_000, (100.0, 80.0), (112.0, 73.0), 60.0, 18.0);
        assert!(
            aggregate.observe(true, Some(previous), Default::default())["transition"].is_null()
        );

        let motion = raw_motion_octrees::SimilarityMotion {
            translation: [3.0, -2.0],
            rotation: 0.01,
            scale_delta: 0.021,
            residual: 0.25,
            support: 24,
        };
        let x = previous.pupil_center_sensor.0 - previous.roi_center_sensor.0;
        let y = previous.pupil_center_sensor.1 - previous.roi_center_sensor.1;
        let transported = (
            previous.pupil_center_sensor.0
                + f64::from(motion.translation[0])
                + f64::from(motion.scale_delta) * x
                - f64::from(motion.rotation) * y,
            previous.pupil_center_sensor.1
                + f64::from(motion.translation[1])
                + f64::from(motion.rotation) * x
                + f64::from(motion.scale_delta) * y,
        );
        let current = observation(
            1_100_000_000,
            (103.0, 78.0),
            transported,
            60.0 * 1.021,
            18.0 * 1.021,
        );
        let diagnostics = aggregate.observe(
            true,
            Some(current),
            raw_motion_octrees::NativeGlobalSimilarityEvidence {
                motion,
                reliable: true,
                ..Default::default()
            },
        );
        let transition = &diagnostics["transition"];
        assert!(
            transition["equivalent_radius_step_fraction"]
                .as_f64()
                .unwrap()
                < 1.0e-12
        );
        assert!(
            transition["gross_anatomy"]["compensated_pupil_radius_log_residual"]
                .as_f64()
                .unwrap()
                < 1.0e-7
        );
        assert!(
            transition["gross_anatomy"]["transport_pupil_center_residual_px"]
                .as_f64()
                .unwrap()
                < 1.0e-7
        );
        let summary = aggregate.json(2, "synthetic test");
        assert_eq!(
            summary["conditional_on_gross_anatomy"]["translation_px"]["3_or_more"]["transitions"],
            1
        );
        assert_eq!(
            summary["conditional_on_gross_anatomy"]["absolute_rotation_degrees"]["0_25_to_below_1"]
                ["transitions"],
            1
        );
        assert_eq!(
            summary["conditional_on_gross_anatomy"]["absolute_scale_delta"]["0_02_or_more"]
                ["transitions"],
            1
        );
    }

    #[test]
    fn unsupported_frame_breaks_the_derivative_chain() {
        let mut aggregate = PupilAffineTemporalAggregate::default();
        let first = observation(1_000_000_000, (100.0, 80.0), (110.0, 75.0), 60.0, 18.0);
        let second = observation(1_100_000_000, (100.0, 80.0), (110.0, 75.0), 60.0, 18.0);
        aggregate.observe(true, Some(first), Default::default());
        aggregate.observe(false, None, Default::default());
        let diagnostics = aggregate.observe(true, Some(second), Default::default());
        assert!(diagnostics["transition"].is_null());
        assert!(aggregate.transitions.is_empty());
    }

    #[test]
    fn image_axis_metric_rejects_a_near_circle_axis_angle_flip_artifact() {
        let first_pose = DrivingAffinePose {
            center: (240.0, 120.0),
            major_radius: 95.0,
            minor_radius: 94.5,
            angle: -2.95,
        };
        let flipped_pose = DrivingAffinePose {
            angle: first_pose.angle + std::f64::consts::FRAC_PI_2,
            ..first_pose
        };
        let pupil = (226.25, 103.5);
        let first = pupil_affine_temporal_observation(
            1_000_000_000,
            (4_000, 3_000),
            (480, 240),
            first_pose,
            pupil,
            30.0,
        )
        .unwrap();
        let flipped = pupil_affine_temporal_observation(
            1_100_000_000,
            (4_000, 3_000),
            (480, 240),
            flipped_pose,
            pupil,
            30.0,
        )
        .unwrap();
        let mut aggregate = PupilAffineTemporalAggregate::default();
        aggregate.observe(true, Some(first), Default::default());
        let diagnostics = aggregate.observe(true, Some(flipped), Default::default());
        let transition = &diagnostics["transition"];
        assert!(transition["canonical_center_step"].as_f64().unwrap() > 0.10);
        assert!(
            transition["image_axis_outer_relative_center_step"]
                .as_f64()
                .unwrap()
                < 1.0e-12
        );
    }
}

const CLOCK_PUPIL_CENTER_LABELS: [&str; 7] = [
    "rough_focus",
    "temporal_prediction",
    "raw_orbital_measurement",
    "published_center",
    "published_boundary",
    "rough_boundary_consensus",
    "rough_published_consensus",
];

#[derive(Clone, Copy, Debug, Default)]
struct ClockPupilSample {
    timestamp_ns: u64,
    target_gaze_tangent: Option<[f64; 2]>,
    target_moving: bool,
    sensor_origin: (u32, u32),
    centers: [Option<(f64, f64)>; CLOCK_PUPIL_CENTER_LABELS.len()],
    /// The same center hypotheses expressed in the finalized limbus' affine
    /// circle coordinates.  This removes sensor ROI steering, head
    /// translation, and the current projected limbus scale from the optical
    /// target comparison.  It is replay telemetry only and never enters the
    /// pupil solver.
    canonical_centers: [Option<(f64, f64)>; CLOCK_PUPIL_CENTER_LABELS.len()],
}

#[derive(Clone, Copy, Debug, Default)]
struct TargetMotionModel {
    x: [f64; 3],
    y: [f64; 3],
}

#[derive(Clone, Copy, Debug)]
struct TargetMotionRow {
    end_index: usize,
    input: [f64; 3],
    pupil_displacement: (f64, f64),
}

fn solve_three_by_three(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> Option<[f64; 3]> {
    for column in 0..3 {
        let pivot = (column..3).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() <= 1.0e-10 {
            return None;
        }
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let inverse = 1.0 / matrix[column][column];
        for entry in column..3 {
            matrix[column][entry] *= inverse;
        }
        rhs[column] *= inverse;
        for row in 0..3 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for entry in column..3 {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    rhs.iter().all(|value| value.is_finite()).then_some(rhs)
}

fn fit_target_motion_axis(rows: &[([f64; 3], f64)], weights: &[f64]) -> Option<[f64; 3]> {
    let mut normal = [[0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for ((input, output), weight) in rows.iter().zip(weights) {
        for row in 0..3 {
            rhs[row] += weight * input[row] * output;
            for column in 0..3 {
                normal[row][column] += weight * input[row] * input[column];
            }
        }
    }
    // A tiny ridge protects warm-up or path fragments without materially
    // influencing a full Lissajous capture.
    for (index, row) in normal.iter_mut().enumerate() {
        row[index] += 1.0e-8;
    }
    solve_three_by_three(normal, rhs)
}

fn target_motion_predict(model: TargetMotionModel, input: [f64; 3]) -> (f64, f64) {
    (
        model.x.iter().zip(input).map(|(a, b)| a * b).sum::<f64>(),
        model.y.iter().zip(input).map(|(a, b)| a * b).sum::<f64>(),
    )
}

fn target_motion_rows(
    samples: &[ClockPupilSample],
    center_index: usize,
    interval_frames: usize,
    target_lag_frames: usize,
) -> Vec<TargetMotionRow> {
    let first_end = interval_frames + target_lag_frames;
    (first_end..samples.len())
        .filter_map(|end_index| {
            let pupil_start_index = end_index - interval_frames;
            let target_end_index = end_index - target_lag_frames;
            let target_start_index = target_end_index - interval_frames;
            let pupil_start = samples[pupil_start_index];
            let pupil_end = samples[end_index];
            let target_start = samples[target_start_index];
            let target_end = samples[target_end_index];
            if !target_start.target_moving || !target_end.target_moving {
                return None;
            }
            let elapsed_ns = pupil_end
                .timestamp_ns
                .checked_sub(pupil_start.timestamp_ns)?;
            if elapsed_ns == 0 || elapsed_ns > interval_frames as u64 * 180_000_000 {
                return None;
            }
            let target_start = target_start.target_gaze_tangent?;
            let target_end = target_end.target_gaze_tangent?;
            let pupil_start_center = pupil_start.centers[center_index]?;
            let pupil_end_center = pupil_end.centers[center_index]?;
            let pupil_displacement = (
                pupil_end_center.0 + f64::from(pupil_end.sensor_origin.0)
                    - pupil_start_center.0
                    - f64::from(pupil_start.sensor_origin.0),
                pupil_end_center.1 + f64::from(pupil_end.sensor_origin.1)
                    - pupil_start_center.1
                    - f64::from(pupil_start.sensor_origin.1),
            );
            let target_displacement = (
                target_end[0] - target_start[0],
                target_end[1] - target_start[1],
            );
            (pupil_displacement.0.is_finite()
                && pupil_displacement.1.is_finite()
                && pupil_displacement.0.hypot(pupil_displacement.1) <= 100.0
                && target_displacement.0.is_finite()
                && target_displacement.1.is_finite()
                && target_displacement.0.hypot(target_displacement.1) > 1.0e-6)
                .then_some(TargetMotionRow {
                    end_index,
                    input: [1.0, target_displacement.0, target_displacement.1],
                    pupil_displacement,
                })
        })
        .collect()
}

fn robust_target_motion_model(rows: &[TargetMotionRow]) -> Option<TargetMotionModel> {
    if rows.len() < 24 {
        return None;
    }
    let x_rows = rows
        .iter()
        .map(|row| (row.input, row.pupil_displacement.0))
        .collect::<Vec<_>>();
    let y_rows = rows
        .iter()
        .map(|row| (row.input, row.pupil_displacement.1))
        .collect::<Vec<_>>();
    let mut weights = vec![1.0; rows.len()];
    let mut model = TargetMotionModel::default();
    for _ in 0..6 {
        model.x = fit_target_motion_axis(&x_rows, &weights)?;
        model.y = fit_target_motion_axis(&y_rows, &weights)?;
        let mut residuals = rows
            .iter()
            .map(|row| {
                let predicted = target_motion_predict(model, row.input);
                (predicted.0 - row.pupil_displacement.0)
                    .hypot(predicted.1 - row.pupil_displacement.1)
            })
            .collect::<Vec<_>>();
        let huber = (1.75 * percentile(&residuals, 0.50)).clamp(2.0, 24.0);
        for (weight, residual) in weights.iter_mut().zip(residuals.drain(..)) {
            *weight = if residual <= huber {
                1.0
            } else {
                huber / residual.max(1.0e-9)
            };
        }
    }
    Some(model)
}

fn target_motion_supervision_summary(samples: &[ClockPupilSample]) -> Value {
    let training_end = samples.len() * 3 / 5;
    let interval_frames = 2;
    // Estimate physiology/pipeline lag once from the dense frame-local rough
    // focus track, then hold it fixed for every candidate source.  Rough focus
    // is upstream of the pupil tracker being compared, so a detector variant
    // cannot improve its score by changing the evaluation lag.  Allowing each
    // sparse source to choose its own lag likewise lets missing-frame patterns
    // game the comparison.
    let shared_target_lag_frames = (0..=6)
        .filter_map(|target_lag_frames| {
            let training = target_motion_rows(
                samples,
                0, // rough_focus: independent of the pupil tracker under test
                interval_frames,
                target_lag_frames,
            )
            .into_iter()
            .filter(|row| row.end_index < training_end)
            .collect::<Vec<_>>();
            let model = robust_target_motion_model(&training)?;
            let residuals = training
                .iter()
                .map(|row| {
                    let predicted = target_motion_predict(model, row.input);
                    (predicted.0 - row.pupil_displacement.0)
                        .hypot(predicted.1 - row.pupil_displacement.1)
                })
                .collect::<Vec<_>>();
            Some((percentile(&residuals, 0.50), target_lag_frames))
        })
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .map_or(0, |(_, lag)| lag);
    let sources = CLOCK_PUPIL_CENTER_LABELS
        .iter()
        .enumerate()
        .map(|(center_index, label)| {
            let rows = target_motion_rows(
                samples,
                center_index,
                interval_frames,
                shared_target_lag_frames,
            );
            let training = rows
                .iter()
                .copied()
                .filter(|row| row.end_index < training_end)
                .collect::<Vec<_>>();
            let Some(model) = robust_target_motion_model(&training) else {
                return json!({"source": label, "model": null});
            };
            let residuals = rows
                .iter()
                .filter(|row| row.end_index >= training_end)
                .map(|row| {
                    let predicted = target_motion_predict(model, row.input);
                    (predicted.0 - row.pupil_displacement.0)
                        .hypot(predicted.1 - row.pupil_displacement.1)
                })
                .filter(|residual| residual.is_finite())
                .collect::<Vec<_>>();
            let within_8 = residuals
                .iter()
                .filter(|residual| **residual <= 8.0)
                .count();
            let within_16 = residuals
                .iter()
                .filter(|residual| **residual <= 16.0)
                .count();
            json!({
                "source": label,
                "training_samples": training.len(),
                "validation_samples": residuals.len(),
                "validation_residual_px": distribution(&residuals),
                "within_8px_fraction": within_8 as f64 / residuals.len().max(1) as f64,
                "within_16px_fraction": within_16 as f64 / residuals.len().max(1) as f64,
                "shared_target_lag_frames": shared_target_lag_frames,
                "model": {
                    "absolute_sensor_pupil_delta_x": model.x,
                    "absolute_sensor_pupil_delta_y": model.y,
                },
            })
        })
        .collect::<Vec<_>>();
    json!({
        "contract": "evaluation-only target-motion fit: lag and coefficients selected on the first 60%, final 40% held out; no host timestamps and no feedback into pupil inference",
        "model": "robust affine absolute-sensor pupil displacement from optically decoded fixation-target displacement",
        "caveat": "absolute sensor displacement includes residual head motion; this is a temporal consistency metric, not a PCCR or pixel ground-truth label",
        "interval_frames": interval_frames,
        "shared_target_lag_frames_selected_on_training": shared_target_lag_frames,
        "shared_target_lag_source": "rough_focus (upstream of pupil tracker)",
        "training_end_frame": training_end,
        "sources": sources,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct ClockPupilLagMetric {
    lag_frames: usize,
    samples: usize,
    x_correlation: f64,
    y_correlation: f64,
    joint_score: f64,
}

fn clock_pupil_lag_metric(
    samples: &[ClockPupilSample],
    center_index: usize,
    lag_frames: usize,
    canonical: bool,
) -> ClockPupilLagMetric {
    let mut count = 0usize;
    let mut target_x_squared = 0.0;
    let mut target_y_squared = 0.0;
    let mut pupil_x_squared = 0.0;
    let mut pupil_y_squared = 0.0;
    let mut x_cross = 0.0;
    let mut y_cross = 0.0;
    for index in lag_frames + 1..samples.len() {
        let target_current = samples[index - lag_frames];
        let target_previous = samples[index - lag_frames - 1];
        if !target_current.target_moving || !target_previous.target_moving {
            continue;
        }
        let Some(target_current) = target_current.target_gaze_tangent else {
            continue;
        };
        let Some(target_previous) = target_previous.target_gaze_tangent else {
            continue;
        };
        let centers = |sample: &ClockPupilSample| {
            if canonical {
                sample.canonical_centers[center_index]
            } else {
                sample.centers[center_index]
            }
        };
        let Some(pupil_current) = centers(&samples[index]) else {
            continue;
        };
        let Some(pupil_previous) = centers(&samples[index - 1]) else {
            continue;
        };
        let pupil_delta = if canonical {
            (
                pupil_current.0 - pupil_previous.0,
                pupil_current.1 - pupil_previous.1,
            )
        } else {
            (
                pupil_current.0 + f64::from(samples[index].sensor_origin.0)
                    - pupil_previous.0
                    - f64::from(samples[index - 1].sensor_origin.0),
                pupil_current.1 + f64::from(samples[index].sensor_origin.1)
                    - pupil_previous.1
                    - f64::from(samples[index - 1].sensor_origin.1),
            )
        };
        if !pupil_delta.0.is_finite()
            || !pupil_delta.1.is_finite()
            || pupil_delta.0.hypot(pupil_delta.1) > 50.0
        {
            continue;
        }
        let target_delta = (
            target_current[0] - target_previous[0],
            target_current[1] - target_previous[1],
        );
        target_x_squared += target_delta.0 * target_delta.0;
        target_y_squared += target_delta.1 * target_delta.1;
        pupil_x_squared += pupil_delta.0 * pupil_delta.0;
        pupil_y_squared += pupil_delta.1 * pupil_delta.1;
        x_cross += target_delta.0 * pupil_delta.0;
        y_cross += target_delta.1 * pupil_delta.1;
        count += 1;
    }
    let correlation = |cross: f64, target_squared: f64, pupil_squared: f64| {
        let denominator = (target_squared * pupil_squared).sqrt();
        (denominator > 1.0e-12)
            .then_some((cross / denominator).clamp(-1.0, 1.0))
            .unwrap_or(0.0)
    };
    let x_correlation = correlation(x_cross, target_x_squared, pupil_x_squared);
    let y_correlation = correlation(y_cross, target_y_squared, pupil_y_squared);
    ClockPupilLagMetric {
        lag_frames,
        samples: count,
        x_correlation,
        y_correlation,
        joint_score: (x_correlation.abs() * y_correlation.abs()).sqrt(),
    }
}

fn clock_pupil_supervision_summary(samples: &[ClockPupilSample]) -> Value {
    let frame_deltas = samples
        .windows(2)
        .filter_map(|pair| pair[1].timestamp_ns.checked_sub(pair[0].timestamp_ns))
        .filter(|delta| *delta > 0)
        .map(|delta| delta as f64 * 1.0e-6)
        .collect::<Vec<_>>();
    let frame_period_ms = percentile(&frame_deltas, 0.50);
    let source_metrics = |canonical: bool| {
        CLOCK_PUPIL_CENTER_LABELS
            .iter()
            .enumerate()
            .map(|(center_index, label)| {
                let center_frames = samples
                    .iter()
                    .filter(|sample| {
                        if canonical {
                            sample.canonical_centers[center_index].is_some()
                        } else {
                            sample.centers[center_index].is_some()
                        }
                    })
                    .count();
                let best = (0..=6)
                    .map(|lag| clock_pupil_lag_metric(samples, center_index, lag, canonical))
                    .filter(|metric| metric.samples >= 20)
                    .max_by(|left, right| left.joint_score.total_cmp(&right.joint_score));
                let zero = clock_pupil_lag_metric(samples, center_index, 0, canonical);
                json!({
                    "source": label,
                    "center_frames": center_frames,
                    "zero_lag": {
                        "samples": zero.samples,
                        "x_correlation": zero.x_correlation,
                        "y_correlation": zero.y_correlation,
                        "joint_score": zero.joint_score,
                    },
                    "best_lag": best.map(|metric| json!({
                        "lag_frames": metric.lag_frames,
                        "lag_ms": metric.lag_frames as f64 * frame_period_ms,
                        "samples": metric.samples,
                        "x_correlation": metric.x_correlation,
                        "y_correlation": metric.y_correlation,
                        "joint_score": metric.joint_score,
                    })),
                })
            })
            .collect::<Vec<_>>()
    };
    json!({
        "contract": "post-inference evaluation only; optical target never enters pupil search",
        "host_timestamp_used": false,
        "median_sensor_frame_period_ms": frame_period_ms,
        "joined_frames": samples.iter().filter(|sample| sample.target_gaze_tangent.is_some()).count(),
        "moving_frames": samples.iter().filter(|sample| sample.target_moving).count(),
        "absolute_sensor": {
            "metric": "per-axis correlation between target and absolute-sensor pupil displacement; this includes head motion",
            "sources": source_metrics(false),
        },
        "limbus_affine_circle": {
            "metric": "per-axis correlation between target and pupil displacement in finalized limbus affine-circle coordinates; head translation, ROI steering, and projected limbus scale are removed",
            "sources": source_metrics(true),
        },
    })
}

const PUPIL_POLAR_HISTORY: Duration = Duration::from_millis(900);
const PUPIL_POLAR_MAX_FRAMES: usize = 10;
const PUPIL_POLAR_SECTORS: usize = 21;
const PUPIL_POLAR_COARSE_RADIUS_STEP: f64 = 0.01;
const PUPIL_POLAR_FINE_RADIUS_STEP: f64 = 0.0025;
const PUPIL_POLAR_MATCH_SIGMA: f64 = 0.014;
const PUPIL_POLAR_MATCH_LIMIT: f64 = 0.035;
const PUPIL_POLAR_POSITIVE_MATCH_THRESHOLD: f64 = 0.12;
const PUPIL_POLAR_CENTER_SEARCH_LIMIT: f64 = 0.05;
const PUPIL_POLAR_CENTER_HALF_STEPS: isize = 5;
const PUPIL_POLAR_CENTER_WIDTH: usize = 11;
const PUPIL_POLAR_CENTER_STATES: usize = PUPIL_POLAR_CENTER_WIDTH * PUPIL_POLAR_CENTER_WIDTH;

#[derive(Clone, Copy, Debug)]
struct PupilPolarCandidateObservation {
    sector: usize,
    canonical: (f64, f64),
    radius_ratio_from_rough_center: f64,
    quality: f64,
}

#[derive(Clone, Debug)]
struct PupilPolarFrameObservation {
    at: Instant,
    candidates: Vec<PupilPolarCandidateObservation>,
    // Candidate radii for every small center-correction state are independent
    // of the pupil-radius hypothesis.  Precomputing them once per RAW frame
    // avoids repeating the square root for every temporal/radius trial.
    radial_distances: Vec<f64>,
    // Every center-state slice indexes the same candidates in increasing
    // radius order. Radius trials binary-search this order and never inspect
    // fragments outside the compact Gaussian support window.
    radial_order: Vec<usize>,
    adjacent_support_intervals: [Vec<(f64, f64)>; PUPIL_POLAR_SECTORS],
}

impl PupilPolarFrameObservation {
    fn new(at: Instant, candidates: Vec<PupilPolarCandidateObservation>) -> Self {
        let mut radial_distances = Vec::with_capacity(PUPIL_POLAR_CENTER_STATES * candidates.len());
        let mut radial_order = Vec::with_capacity(PUPIL_POLAR_CENTER_STATES * candidates.len());
        for state in 0..PUPIL_POLAR_CENTER_STATES {
            let center_offset = pupil_polar_center_offset(state);
            radial_distances.extend(candidates.iter().map(|candidate| {
                (candidate.canonical.0 - center_offset.0)
                    .hypot(candidate.canonical.1 - center_offset.1)
            }));
            let state_start = state * candidates.len();
            let mut order = (0..candidates.len()).collect::<Vec<_>>();
            order.sort_unstable_by(|left, right| {
                radial_distances[state_start + *left]
                    .total_cmp(&radial_distances[state_start + *right])
            });
            radial_order.extend(order);
        }
        let candidates_by_sector: [Vec<usize>; PUPIL_POLAR_SECTORS] =
            std::array::from_fn(|sector| {
                candidates
                    .iter()
                    .enumerate()
                    .filter_map(|(index, candidate)| (candidate.sector == sector).then_some(index))
                    .collect()
            });
        let support_limits = candidates
            .iter()
            .map(|candidate| pupil_polar_positive_support_limit(candidate.quality))
            .collect::<Vec<_>>();
        let mut adjacent_support_intervals: [Vec<(f64, f64)>; PUPIL_POLAR_SECTORS] =
            std::array::from_fn(|_| Vec::new());
        for state in 0..PUPIL_POLAR_CENTER_STATES {
            let state_start = state * candidates.len();
            for sector in 0..PUPIL_POLAR_SECTORS {
                let next_sector = (sector + 1) % PUPIL_POLAR_SECTORS;
                for left in candidates_by_sector[sector].iter().copied() {
                    let Some(left_limit) = support_limits[left] else {
                        continue;
                    };
                    let left_radius = radial_distances[state_start + left];
                    for right in candidates_by_sector[next_sector].iter().copied() {
                        let Some(right_limit) = support_limits[right] else {
                            continue;
                        };
                        let right_radius = radial_distances[state_start + right];
                        let minimum = (left_radius - left_limit).max(right_radius - right_limit);
                        let maximum = (left_radius + left_limit).min(right_radius + right_limit);
                        if minimum <= maximum {
                            adjacent_support_intervals[sector]
                                .push((minimum - 1.0e-12, maximum + 1.0e-12));
                        }
                    }
                }
            }
        }
        for intervals in &mut adjacent_support_intervals {
            intervals.sort_unstable_by(|left, right| {
                left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1))
            });
            let mut merged = Vec::<(f64, f64)>::with_capacity(intervals.len());
            for interval in intervals.drain(..) {
                if let Some(previous) = merged
                    .last_mut()
                    .filter(|previous| interval.0 <= previous.1)
                {
                    previous.1 = previous.1.max(interval.1);
                } else {
                    merged.push(interval);
                }
            }
            *intervals = merged;
        }
        Self {
            at,
            candidates,
            radial_distances,
            radial_order,
            adjacent_support_intervals,
        }
    }

    fn radial_distance(&self, state: usize, candidate: usize) -> f64 {
        self.radial_distances[state * self.candidates.len() + candidate]
    }

    fn ordered_candidates(&self, state: usize) -> &[usize] {
        let start = state * self.candidates.len();
        &self.radial_order[start..start + self.candidates.len()]
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PupilPolarSliceMatch {
    sector: usize,
    source_radius_ratio: f64,
    residual: f64,
    quality: f64,
    match_score: f64,
}

#[derive(Clone, Debug, Default)]
struct PupilPolarFrameFit {
    center_offset: (f64, f64),
    score: f64,
    supported_mask: u32,
    longest_arc_sectors: usize,
    matches: Vec<PupilPolarSliceMatch>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PupilPolarCoSolveDiagnostics {
    pub(super) elapsed_us: u64,
    pub(super) evaluated_radius_hypotheses: usize,
    pub(super) ratio: Option<f64>,
    pub(super) evidence_selected_ratio: Option<f64>,
    pub(super) score: f64,
    pub(super) raw_best_ratio: Option<f64>,
    pub(super) raw_best_score: f64,
    pub(super) incumbent_ratio: Option<f64>,
    pub(super) incumbent_score: f64,
    pub(super) incumbent_retained_as_supported_tie: bool,
    pub(super) kinematically_limited: bool,
    pub(super) supporting_frames: usize,
    pub(super) unique_sectors: usize,
    pub(super) current_defined_candidates: usize,
    pub(super) current_center_offset: Option<(f64, f64)>,
    pub(super) current_longest_arc_sectors: usize,
    current_matches: Vec<PupilPolarSliceMatch>,
    pub(super) provisional: bool,
    pub(super) qualified: bool,
    pub(super) legacy_radius_ratio: Option<f64>,
    pub(super) legacy_radius_vetoed: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct PupilPolarProjectedEllipse {
    pub(super) center: (f64, f64),
    pub(super) major_radius: f64,
    pub(super) minor_radius: f64,
    pub(super) angle: f64,
}

impl PupilPolarProjectedEllipse {
    pub(super) fn perimeter(self, samples: usize) -> Vec<(f64, f64)> {
        let samples = samples.max(8);
        let (sine, cosine) = self.angle.sin_cos();
        (0..samples)
            .map(|index| {
                let phase = std::f64::consts::TAU * index as f64 / samples as f64;
                let local = (
                    self.major_radius * phase.cos(),
                    self.minor_radius * phase.sin(),
                );
                (
                    self.center.0 + cosine * local.0 - sine * local.1,
                    self.center.1 + sine * local.0 + cosine * local.1,
                )
            })
            .collect()
    }
}

impl PupilPolarCoSolveDiagnostics {
    /// Project the shared fronto-parallel nested-circle state back into the
    /// untouched ROI. Provisional two-frame slices remain diagnostic only;
    /// the live overlay appears after three frames and four unique sectors.
    pub(super) fn projected_ellipse(
        &self,
        pose: DrivingAffinePose,
    ) -> Option<PupilPolarProjectedEllipse> {
        let ratio = self.ratio?;
        let offset = self.current_center_offset.unwrap_or((0.0, 0.0));
        if !self.qualified
            || !ratio.is_finite()
            || ratio <= 0.0
            || !offset.0.is_finite()
            || !offset.1.is_finite()
        {
            return None;
        }
        let local = (offset.0 * pose.major_radius, offset.1 * pose.minor_radius);
        let (sine, cosine) = pose.angle.sin_cos();
        Some(PupilPolarProjectedEllipse {
            center: (
                pose.center.0 + cosine * local.0 - sine * local.1,
                pose.center.1 + sine * local.0 + cosine * local.1,
            ),
            major_radius: pose.major_radius * ratio,
            minor_radius: pose.minor_radius * ratio,
            angle: pose.angle,
        })
    }

    pub(super) fn current_slice_points(
        &self,
        boundary: &raw_iris_focus::InnerIrisBoundary,
        pose: DrivingAffinePose,
    ) -> Vec<(f64, f64)> {
        self.current_matches
            .iter()
            .filter_map(|matched| {
                boundary
                    .radial_candidates
                    .iter()
                    .filter(|candidate| usize::from(candidate.sector_index) == matched.sector)
                    .min_by(|left, right| {
                        let ratio = |candidate: &raw_iris_focus::InnerIrisRadialCandidate| {
                            let dx = candidate.x - boundary.center.0;
                            let dy = candidate.y - boundary.center.1;
                            let (sine, cosine) = pose.angle.sin_cos();
                            let canonical_x =
                                (cosine * dx + sine * dy) / pose.major_radius.max(1.0);
                            let canonical_y =
                                (-sine * dx + cosine * dy) / pose.minor_radius.max(1.0);
                            canonical_x.hypot(canonical_y)
                        };
                        (ratio(left) - matched.source_radius_ratio)
                            .abs()
                            .total_cmp(&(ratio(right) - matched.source_radius_ratio).abs())
                    })
                    .map(|candidate| (candidate.x, candidate.y))
            })
            .collect()
    }
}

#[derive(Default)]
pub(super) struct PupilPolarCoSolver {
    frames: VecDeque<PupilPolarFrameObservation>,
    last_qualified_ratio: Option<(Instant, f64)>,
    published_ratio: Option<(Instant, f64)>,
    last_observed_at: Option<Instant>,
    last_sensor_pose: Option<DrivingAffinePose>,
}

fn pupil_polar_candidate_quality(
    candidate: raw_iris_focus::InnerIrisRadialCandidate,
) -> Option<f64> {
    if !candidate.raw_score.is_finite()
        || !candidate.peak_prominence.is_finite()
        || !candidate.luma_transition.is_finite()
        || !candidate.void_drop.is_finite()
        || !candidate.inside_void.is_finite()
        || candidate.raw_score < 0.30
        || candidate.peak_prominence < 0.055
        || candidate.inside_void < 0.52
        || (candidate.void_drop < 0.08 && candidate.luma_transition < 0.34)
        || candidate.broad_dark_step.is_finite() && candidate.broad_dark_step < 0.0
    {
        return None;
    }
    let raw = ((candidate.raw_score - 0.20) / 0.48).clamp(0.0, 1.0);
    let prominence = (candidate.peak_prominence / 0.20).clamp(0.0, 1.0);
    let luma = (candidate.luma_transition / 0.65).clamp(0.0, 1.0);
    let void = (candidate.void_drop / 0.32).clamp(0.0, 1.0);
    let interior = ((candidate.inside_void - 0.40) / 0.50).clamp(0.0, 1.0);
    let broad = if candidate.broad_dark_step.is_finite() {
        ((candidate.broad_dark_step + 0.02) / 0.40).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some(
        (0.24 * raw
            + 0.22 * prominence
            + 0.18 * luma
            + 0.18 * void
            + 0.10 * interior
            + 0.08 * broad)
            .clamp(0.0, 1.0),
    )
}

fn pupil_polar_positive_support_limit(quality: f64) -> Option<f64> {
    (quality.is_finite() && quality >= PUPIL_POLAR_POSITIVE_MATCH_THRESHOLD).then(|| {
        let normalized = (PUPIL_POLAR_POSITIVE_MATCH_THRESHOLD / quality).clamp(0.0, 1.0);
        (PUPIL_POLAR_MATCH_SIGMA * (-2.0 * normalized.ln()).sqrt()).min(PUPIL_POLAR_MATCH_LIMIT)
    })
}

fn pupil_polar_supported_mask(mask: u32) -> u32 {
    let all = (1u32 << PUPIL_POLAR_SECTORS) - 1;
    let left = ((mask << 1) | (mask >> (PUPIL_POLAR_SECTORS - 1))) & all;
    let right = (mask >> 1) | ((mask & 1) << (PUPIL_POLAR_SECTORS - 1));
    mask & (left | right)
}

fn pupil_polar_ratio_can_possibly_qualify(
    frames: &VecDeque<PupilPolarFrameObservation>,
    ratio: f64,
) -> bool {
    let mut supporting_frames = 0usize;
    let mut sector_union = 0u32;
    for frame in frames {
        let possible_support = frame.adjacent_support_intervals.iter().enumerate().fold(
            0u32,
            |mask, (sector, intervals)| {
                if intervals
                    .iter()
                    .any(|(minimum, maximum)| ratio >= *minimum && ratio <= *maximum)
                {
                    mask | (1u32 << sector) | (1u32 << ((sector + 1) % PUPIL_POLAR_SECTORS))
                } else {
                    mask
                }
            },
        );
        if possible_support.count_ones() >= 2 {
            supporting_frames += 1;
            sector_union |= possible_support;
        }
    }
    // This is only an upper-bound rejection. Each interval is the exact union
    // over all 121 allowed center states where an adjacent pair can clear the
    // positive-evidence threshold. Passing still requires the trajectory
    // solve; failing means no trajectory can meet the provisional gate.
    supporting_frames >= 2 && sector_union.count_ones() >= 3
}

fn pupil_polar_longest_arc(mask: u32) -> usize {
    let mut longest = 0usize;
    for start in 0..PUPIL_POLAR_SECTORS {
        let mut run = 0usize;
        for offset in 0..PUPIL_POLAR_SECTORS {
            let sector = (start + offset) % PUPIL_POLAR_SECTORS;
            if mask & (1u32 << sector) == 0 {
                break;
            }
            run += 1;
        }
        longest = longest.max(run);
    }
    longest
}

fn pupil_polar_center_offset(state: usize) -> (f64, f64) {
    debug_assert!(state < PUPIL_POLAR_CENTER_STATES);
    let x_step = state % PUPIL_POLAR_CENTER_WIDTH;
    let y_step = state / PUPIL_POLAR_CENTER_WIDTH;
    let step = PUPIL_POLAR_CENTER_SEARCH_LIMIT / PUPIL_POLAR_CENTER_HALF_STEPS as f64;
    (
        (x_step as isize - PUPIL_POLAR_CENTER_HALF_STEPS) as f64 * step,
        (y_step as isize - PUPIL_POLAR_CENTER_HALF_STEPS) as f64 * step,
    )
}

fn pupil_polar_gaussian_weight(residual: f64) -> f64 {
    (-0.5 * (residual / PUPIL_POLAR_MATCH_SIGMA).powi(2)).exp()
}

fn pupil_polar_select_matches(
    frame: &PupilPolarFrameObservation,
    ratio: f64,
    mut radial_distance: impl FnMut(usize) -> f64,
) -> [Option<PupilPolarSliceMatch>; PUPIL_POLAR_SECTORS] {
    let mut selected = [None::<PupilPolarSliceMatch>; PUPIL_POLAR_SECTORS];
    for (candidate_index, candidate) in frame.candidates.iter().enumerate() {
        pupil_polar_consider_match(
            &mut selected,
            *candidate,
            radial_distance(candidate_index),
            ratio,
        );
    }
    selected
}

fn pupil_polar_consider_match(
    selected: &mut [Option<PupilPolarSliceMatch>; PUPIL_POLAR_SECTORS],
    candidate: PupilPolarCandidateObservation,
    radial_distance: f64,
    ratio: f64,
) {
    let residual = (radial_distance - ratio).abs();
    if !residual.is_finite() || residual > PUPIL_POLAR_MATCH_LIMIT {
        return;
    }
    let match_score = candidate.quality * pupil_polar_gaussian_weight(residual);
    let matched = PupilPolarSliceMatch {
        sector: candidate.sector,
        source_radius_ratio: candidate.radius_ratio_from_rough_center,
        residual,
        quality: candidate.quality,
        match_score,
    };
    if selected[candidate.sector].is_none_or(|previous| match_score > previous.match_score) {
        selected[candidate.sector] = Some(matched);
    }
}

fn pupil_polar_select_matches_at_state(
    frame: &PupilPolarFrameObservation,
    ratio: f64,
    state: usize,
) -> [Option<PupilPolarSliceMatch>; PUPIL_POLAR_SECTORS] {
    let mut selected = [None::<PupilPolarSliceMatch>; PUPIL_POLAR_SECTORS];
    let ordered = frame.ordered_candidates(state);
    let minimum = ratio - PUPIL_POLAR_MATCH_LIMIT;
    let maximum = ratio + PUPIL_POLAR_MATCH_LIMIT;
    let start =
        ordered.partition_point(|candidate| frame.radial_distance(state, *candidate) < minimum);
    let end =
        ordered.partition_point(|candidate| frame.radial_distance(state, *candidate) <= maximum);
    for candidate_index in ordered[start..end].iter().copied() {
        pupil_polar_consider_match(
            &mut selected,
            frame.candidates[candidate_index],
            frame.radial_distance(state, candidate_index),
            ratio,
        );
    }
    selected
}

#[derive(Clone, Copy, Debug, Default)]
struct PupilPolarFrameEmission {
    score: f64,
    supported_mask: u32,
    longest_arc_sectors: usize,
}

fn pupil_polar_emission_from_matches(
    selected: &[Option<PupilPolarSliceMatch>; PUPIL_POLAR_SECTORS],
    center_offset: (f64, f64),
) -> Option<PupilPolarFrameEmission> {
    let raw_mask = selected
        .iter()
        .enumerate()
        .fold(0u32, |mask, (sector, matched)| {
            if matched
                .is_some_and(|matched| matched.match_score >= PUPIL_POLAR_POSITIVE_MATCH_THRESHOLD)
            {
                mask | (1u32 << sector)
            } else {
                mask
            }
        });
    // An isolated radial maximum is useful veto evidence but cannot nominate
    // a circle. Only adjacent polar slices contribute positive fit score.
    let supported_mask = pupil_polar_supported_mask(raw_mask);
    if supported_mask.count_ones() < 2 {
        return None;
    }
    let longest_arc_sectors = pupil_polar_longest_arc(supported_mask);
    let center_distance = center_offset.0.hypot(center_offset.1);
    let score = (selected
        .iter()
        .flatten()
        .filter(|matched| supported_mask & (1u32 << matched.sector) != 0)
        .map(|matched| matched.match_score)
        .sum::<f64>()
        + 0.08 * longest_arc_sectors.saturating_sub(1) as f64
        - 0.10 * (center_distance / PUPIL_POLAR_CENTER_SEARCH_LIMIT).powi(2))
    .max(0.0);
    Some(PupilPolarFrameEmission {
        score,
        supported_mask,
        longest_arc_sectors,
    })
}

fn pupil_polar_fit_from_matches(
    center_offset: (f64, f64),
    selected: [Option<PupilPolarSliceMatch>; PUPIL_POLAR_SECTORS],
) -> Option<PupilPolarFrameFit> {
    let emission = pupil_polar_emission_from_matches(&selected, center_offset)?;
    let matches = selected
        .into_iter()
        .flatten()
        .filter(|matched| emission.supported_mask & (1u32 << matched.sector) != 0)
        .collect();
    Some(PupilPolarFrameFit {
        center_offset,
        score: emission.score,
        supported_mask: emission.supported_mask,
        longest_arc_sectors: emission.longest_arc_sectors,
        matches,
    })
}

fn pupil_polar_fit_at_offset(
    frame: &PupilPolarFrameObservation,
    ratio: f64,
    center_offset: (f64, f64),
) -> Option<PupilPolarFrameFit> {
    let selected = pupil_polar_select_matches(frame, ratio, |candidate_index| {
        let candidate = frame.candidates[candidate_index];
        (candidate.canonical.0 - center_offset.0).hypot(candidate.canonical.1 - center_offset.1)
    });
    pupil_polar_fit_from_matches(center_offset, selected)
}

fn pupil_polar_emission_at_state(
    frame: &PupilPolarFrameObservation,
    ratio: f64,
    state: usize,
) -> Option<PupilPolarFrameEmission> {
    let selected = pupil_polar_select_matches_at_state(frame, ratio, state);
    pupil_polar_emission_from_matches(&selected, pupil_polar_center_offset(state))
}

fn pupil_polar_fit_at_state(
    frame: &PupilPolarFrameObservation,
    ratio: f64,
    state: usize,
) -> Option<PupilPolarFrameFit> {
    let selected = pupil_polar_select_matches_at_state(frame, ratio, state);
    pupil_polar_fit_from_matches(pupil_polar_center_offset(state), selected)
}

#[derive(Clone, Debug, Default)]
struct PupilPolarRatioFit {
    ratio: f64,
    objective: f64,
    supporting_frames: usize,
    unique_sectors: usize,
    current_state: Option<usize>,
}

fn pupil_polar_anchor_penalty(offset: (f64, f64)) -> f64 {
    0.06 * (offset.0.hypot(offset.1) / PUPIL_POLAR_CENTER_SEARCH_LIMIT).powi(2)
}

fn pupil_polar_max_quadratic_transform(
    input: &[f64; PUPIL_POLAR_CENTER_WIDTH],
    penalty_per_step_squared: f64,
) -> (
    [f64; PUPIL_POLAR_CENTER_WIDTH],
    [usize; PUPIL_POLAR_CENTER_WIDTH],
) {
    // Exact one-dimensional max-plus squared-distance transform. Each input
    // state defines a downward parabola; the envelope changes at the pairwise
    // intersections recorded in `boundaries`.
    let mut sites = [0usize; PUPIL_POLAR_CENTER_WIDTH];
    let mut boundaries = [f64::INFINITY; PUPIL_POLAR_CENTER_WIDTH + 1];
    boundaries[0] = f64::NEG_INFINITY;
    let mut envelope = 0usize;
    for candidate in 1..PUPIL_POLAR_CENTER_WIDTH {
        let mut intersection;
        loop {
            let incumbent = sites[envelope];
            let candidate_coordinate = candidate as f64;
            let incumbent_coordinate = incumbent as f64;
            // Negating the scores converts max(f_i-a(x-i)^2) into the
            // conventional lower envelope of upward parabolas.
            intersection = ((-input[candidate]
                + penalty_per_step_squared * candidate_coordinate.powi(2))
                - (-input[incumbent] + penalty_per_step_squared * incumbent_coordinate.powi(2)))
                / (2.0 * penalty_per_step_squared * (candidate_coordinate - incumbent_coordinate));
            if envelope == 0 || intersection > boundaries[envelope] {
                break;
            }
            envelope -= 1;
        }
        envelope += 1;
        sites[envelope] = candidate;
        boundaries[envelope] = intersection;
        boundaries[envelope + 1] = f64::INFINITY;
    }
    let mut values = [f64::NEG_INFINITY; PUPIL_POLAR_CENTER_WIDTH];
    let mut predecessors = [0usize; PUPIL_POLAR_CENTER_WIDTH];
    let mut segment = 0usize;
    for destination in 0..PUPIL_POLAR_CENTER_WIDTH {
        while boundaries[segment + 1] < destination as f64 {
            segment += 1;
        }
        let source = sites[segment];
        let delta = destination as f64 - source as f64;
        values[destination] = input[source] - penalty_per_step_squared * delta * delta;
        predecessors[destination] = source;
    }
    (values, predecessors)
}

fn pupil_polar_advance_center_path(
    scores: &[f64],
    emissions: &[Option<PupilPolarFrameEmission>],
) -> (
    [f64; PUPIL_POLAR_CENTER_STATES],
    [usize; PUPIL_POLAR_CENTER_STATES],
) {
    debug_assert_eq!(scores.len(), PUPIL_POLAR_CENTER_STATES);
    debug_assert_eq!(emissions.len(), PUPIL_POLAR_CENTER_STATES);
    let mut next_scores = [f64::NEG_INFINITY; PUPIL_POLAR_CENTER_STATES];
    let mut previous_states = [0usize; PUPIL_POLAR_CENTER_STATES];
    // The quadratic center-transition cost is separable in x and y. Apply an
    // exact squared-distance transform to every row and then every column.
    // This is the same all-to-all Viterbi transition without an artificial
    // center-motion cutoff, but it is linear rather than quadratic per axis.
    let center_step = PUPIL_POLAR_CENTER_SEARCH_LIMIT / PUPIL_POLAR_CENTER_HALF_STEPS as f64;
    let penalty_per_step_squared = 0.12 * (center_step / 0.025).powi(2);
    let mut x_scores = [f64::NEG_INFINITY; PUPIL_POLAR_CENTER_STATES];
    let mut x_predecessors = [0usize; PUPIL_POLAR_CENTER_STATES];
    for previous_y in 0..PUPIL_POLAR_CENTER_WIDTH {
        let row_start = previous_y * PUPIL_POLAR_CENTER_WIDTH;
        let row = std::array::from_fn(|x| scores[row_start + x]);
        let (row_scores, row_predecessors) =
            pupil_polar_max_quadratic_transform(&row, penalty_per_step_squared);
        for current_x in 0..PUPIL_POLAR_CENTER_WIDTH {
            let target = row_start + current_x;
            x_scores[target] = row_scores[current_x];
            x_predecessors[target] = row_predecessors[current_x];
        }
    }
    for current_x in 0..PUPIL_POLAR_CENTER_WIDTH {
        let column = std::array::from_fn(|y| x_scores[y * PUPIL_POLAR_CENTER_WIDTH + current_x]);
        let (column_scores, y_predecessors) =
            pupil_polar_max_quadratic_transform(&column, penalty_per_step_squared);
        for current_y in 0..PUPIL_POLAR_CENTER_WIDTH {
            let state = current_y * PUPIL_POLAR_CENTER_WIDTH + current_x;
            let previous_y = y_predecessors[current_y];
            let offset = pupil_polar_center_offset(state);
            next_scores[state] = column_scores[current_y]
                + emissions[state].map_or(0.0, |fit| fit.score)
                - pupil_polar_anchor_penalty(offset);
            previous_states[state] = previous_y * PUPIL_POLAR_CENTER_WIDTH
                + x_predecessors[previous_y * PUPIL_POLAR_CENTER_WIDTH + current_x];
        }
    }
    (next_scores, previous_states)
}

fn pupil_polar_ratio_fit(
    frames: &VecDeque<PupilPolarFrameObservation>,
    ratio: f64,
) -> PupilPolarRatioFit {
    if frames.is_empty() {
        return PupilPolarRatioFit::default();
    }
    // Radius is shared across the window, while the center correction is a
    // small trajectory. A real saccade can change the correction as the
    // transported rough center catches up; a smoothness term prevents each
    // frame from independently stitching unrelated fibres into a circle.
    let emissions: Vec<[Option<PupilPolarFrameEmission>; PUPIL_POLAR_CENTER_STATES]> = frames
        .iter()
        .map(|frame| {
            std::array::from_fn(|state| pupil_polar_emission_at_state(frame, ratio, state))
        })
        .collect::<Vec<_>>();
    let mut scores = std::array::from_fn(|state| {
        let offset = pupil_polar_center_offset(state);
        emissions[0][state].map_or(0.0, |fit| fit.score) - pupil_polar_anchor_penalty(offset)
    });
    let mut backpointers =
        Vec::<[usize; PUPIL_POLAR_CENTER_STATES]>::with_capacity(frames.len().saturating_sub(1));
    for frame_index in 1..frames.len() {
        let (next_scores, previous_states) =
            pupil_polar_advance_center_path(&scores, &emissions[frame_index]);
        scores = next_scores;
        backpointers.push(previous_states);
    }
    let mut state = scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map_or(0, |(state, _)| state);
    let mut selected_states = vec![0usize; frames.len()];
    selected_states[frames.len() - 1] = state;
    for frame_index in (1..frames.len()).rev() {
        state = backpointers[frame_index - 1][state];
        selected_states[frame_index - 1] = state;
    }
    let (supporting_frames, sector_mask) = selected_states
        .iter()
        .enumerate()
        .filter_map(|(frame, state)| emissions[frame][*state])
        .filter(|fit| fit.score >= 0.35)
        .fold((0usize, 0u32), |(frames, mask), fit| {
            (frames + 1, mask | fit.supported_mask)
        });
    let unique_sectors = sector_mask.count_ones() as usize;
    let path_score = scores.into_iter().max_by(f64::total_cmp).unwrap_or(0.0);
    let objective = path_score + 0.30 * supporting_frames as f64 + 0.025 * unique_sectors as f64;
    PupilPolarRatioFit {
        ratio,
        objective,
        supporting_frames,
        unique_sectors,
        current_state: selected_states.last().copied(),
    }
}

impl PupilPolarCoSolver {
    pub(super) fn clear_current_evidence(&mut self) {
        self.frames.clear();
        self.last_sensor_pose = None;
    }

    pub(super) fn observe(
        &mut self,
        now: Instant,
        sensor_origin: (u32, u32),
        boundary: &raw_iris_focus::InnerIrisBoundary,
        pose: DrivingAffinePose,
        hard_ratio_bounds: (f64, f64),
    ) -> PupilPolarCoSolveDiagnostics {
        let solve_started = Instant::now();
        if self
            .last_observed_at
            .is_some_and(|last| now.saturating_duration_since(last) > PUPIL_POLAR_HISTORY)
        {
            self.clear_current_evidence();
            // After a gap this long the iris may have genuinely dilated. Keep
            // the old qualified ratio as a soft tie candidate, but do not
            // rate-limit a newly supported physical output against a stale
            // publication timestamp.
            self.published_ratio = None;
        }
        let sensor_pose = DrivingAffinePose {
            center: (
                pose.center.0 + f64::from(sensor_origin.0),
                pose.center.1 + f64::from(sensor_origin.1),
            ),
            ..pose
        };
        if self.last_sensor_pose.is_some_and(|previous| {
            let reference = previous.major_radius.max(sensor_pose.major_radius).max(1.0);
            let center_jump = (previous.center.0 - sensor_pose.center.0)
                .hypot(previous.center.1 - sensor_pose.center.1)
                / reference;
            let scale_jump = (sensor_pose.major_radius / previous.major_radius.max(1.0))
                .ln()
                .abs();
            let previous_eccentricity =
                1.0 - previous.minor_radius / previous.major_radius.max(1.0);
            let current_eccentricity =
                1.0 - sensor_pose.minor_radius / sensor_pose.major_radius.max(1.0);
            let angle_jump = (sensor_pose.angle - previous.angle + std::f64::consts::PI)
                .rem_euclid(std::f64::consts::TAU)
                - std::f64::consts::PI;
            center_jump > 0.55
                || scale_jump > 0.30
                || (previous_eccentricity.min(current_eccentricity) > 0.15
                    && angle_jump.abs() > 0.70)
        }) {
            // A different sclera/limbus hypothesis must earn its own nested
            // slices; otherwise two unrelated polar coordinate systems can
            // accidentally complete one another's missing sectors.
            self.clear_current_evidence();
        }
        self.last_sensor_pose = Some(sensor_pose);
        self.last_observed_at = Some(now);
        let (axis_sine, axis_cosine) = pose.angle.sin_cos();
        let candidates = boundary
            .radial_candidates
            .iter()
            .copied()
            .filter_map(|source| {
                let quality = pupil_polar_candidate_quality(source)?;
                let dx = source.x - boundary.center.0;
                let dy = source.y - boundary.center.1;
                let local_x = axis_cosine * dx + axis_sine * dy;
                let local_y = -axis_sine * dx + axis_cosine * dy;
                let canonical = (
                    local_x / pose.major_radius.max(1.0),
                    local_y / pose.minor_radius.max(1.0),
                );
                let radius_ratio_from_rough_center = canonical.0.hypot(canonical.1);
                (canonical.0.is_finite()
                    && canonical.1.is_finite()
                    && radius_ratio_from_rough_center.is_finite())
                .then_some(PupilPolarCandidateObservation {
                    sector: usize::from(source.sector_index).min(PUPIL_POLAR_SECTORS - 1),
                    canonical,
                    radius_ratio_from_rough_center,
                    quality,
                })
            })
            .collect::<Vec<_>>();
        let current_defined_candidates = candidates.len();
        self.frames
            .push_back(PupilPolarFrameObservation::new(now, candidates));
        while self.frames.len() > PUPIL_POLAR_MAX_FRAMES
            || self
                .frames
                .front()
                .is_some_and(|frame| now.saturating_duration_since(frame.at) > PUPIL_POLAR_HISTORY)
        {
            self.frames.pop_front();
        }
        let minimum_ratio = hard_ratio_bounds.0.clamp(0.05, 0.85);
        let maximum_ratio = hard_ratio_bounds.1.clamp(minimum_ratio + 0.02, 0.90);
        // Preserve the original 0.01 full-range search, then polish its
        // winning basin at 0.0025. The per-state radial ordering makes this
        // exact search cheap without risking a multimodal alias.
        let steps =
            ((maximum_ratio - minimum_ratio) / PUPIL_POLAR_COARSE_RADIUS_STEP).floor() as usize;
        let mut evaluated_radius_hypotheses = 0usize;
        let mut best = (0..=steps)
            .map(|step| minimum_ratio + step as f64 * PUPIL_POLAR_COARSE_RADIUS_STEP)
            .filter(|ratio| pupil_polar_ratio_can_possibly_qualify(&self.frames, *ratio))
            .map(|ratio| {
                evaluated_radius_hypotheses += 1;
                pupil_polar_ratio_fit(&self.frames, ratio)
            })
            .filter(|fit| fit.supporting_frames >= 2 && fit.unique_sectors >= 3)
            .max_by(|left, right| left.objective.total_cmp(&right.objective));
        if let Some(coarse) = best.clone() {
            for offset in -4..=4 {
                let ratio = (coarse.ratio + f64::from(offset) * PUPIL_POLAR_FINE_RADIUS_STEP)
                    .clamp(minimum_ratio, maximum_ratio);
                evaluated_radius_hypotheses += 1;
                let candidate = pupil_polar_ratio_fit(&self.frames, ratio);
                if candidate.supporting_frames >= 2
                    && candidate.unique_sectors >= 3
                    && best
                        .as_ref()
                        .is_none_or(|best| candidate.objective > best.objective)
                {
                    best = Some(candidate);
                }
            }
        }
        let outer_equivalent_radius = (pose.major_radius * pose.minor_radius).sqrt();
        let legacy_radius_ratio = (boundary.radius.is_finite()
            && boundary.radius > 0.0
            && outer_equivalent_radius.is_finite()
            && outer_equivalent_radius > 0.0)
            .then_some(boundary.radius / outer_equivalent_radius);
        let Some(raw_best) = best else {
            return PupilPolarCoSolveDiagnostics {
                elapsed_us: solve_started.elapsed().as_micros() as u64,
                evaluated_radius_hypotheses,
                current_defined_candidates,
                legacy_radius_ratio,
                incumbent_ratio: self.last_qualified_ratio.map(|(_, ratio)| ratio),
                ..PupilPolarCoSolveDiagnostics::default()
            };
        };
        let raw_best_ratio = raw_best.ratio;
        let raw_best_score = raw_best.objective;
        let mut selected = raw_best;
        let mut incumbent_ratio = None;
        let mut incumbent_score = 0.0;
        let mut incumbent_retained_as_supported_tie = false;
        if let Some((_, ratio)) = self.last_qualified_ratio {
            evaluated_radius_hypotheses += 1;
            let incumbent = pupil_polar_ratio_fit(&self.frames, ratio);
            incumbent_ratio = Some(ratio);
            incumbent_score = incumbent.objective;
            let incumbent_still_supported =
                incumbent.supporting_frames >= 3 && incumbent.unique_sectors >= 4;
            // History cannot change what the RAW scan found. It may only win
            // a close global tie after the independent sparse-sector solve.
            // An eight-percent objective advantage (at least 0.75 absolute)
            // is considered decisive current temporal evidence and replaces
            // the incumbent immediately.
            let decisive_margin = (raw_best_score * 0.08).max(0.75);
            if incumbent_still_supported && raw_best_score - incumbent.objective <= decisive_margin
            {
                selected = incumbent;
                incumbent_retained_as_supported_tie = true;
            }
        }
        let best = selected;
        let current_fit = best.current_state.and_then(|state| {
            self.frames
                .back()
                .and_then(|frame| pupil_polar_fit_at_state(frame, best.ratio, state))
        });
        let provisional = best.supporting_frames >= 2 && best.unique_sectors >= 3;
        let qualified = best.supporting_frames >= 3 && best.unique_sectors >= 4;
        let evidence_selected_ratio = best.ratio;
        if qualified {
            self.last_qualified_ratio = Some((now, best.ratio));
        }
        let (published_ratio, kinematically_limited) = if qualified {
            let published = self
                .published_ratio
                .map_or(best.ratio, |(previous_at, previous)| {
                    let elapsed = now
                        .saturating_duration_since(previous_at)
                        .as_secs_f64()
                        .clamp(0.0, 0.50);
                    // The current RAW optimum is still reported independently.
                    // This limit belongs only to the physical-size hypothesis and
                    // therefore cannot hide or suppress a candidate arc.
                    let fractional_step = (0.20 * elapsed).clamp(0.0, 0.10);
                    best.ratio.clamp(
                        previous * (1.0 - fractional_step),
                        previous * (1.0 + fractional_step),
                    )
                });
            self.published_ratio = Some((now, published));
            (published, (published - best.ratio).abs() > 1.0e-9)
        } else {
            (best.ratio, false)
        };
        let legacy_radius_vetoed = legacy_radius_ratio.is_some_and(|legacy| {
            current_fit
                .as_ref()
                .is_some_and(|fit| fit.longest_arc_sectors >= 2)
                && (legacy - best.ratio).abs() >= 0.04
        });
        PupilPolarCoSolveDiagnostics {
            elapsed_us: solve_started.elapsed().as_micros() as u64,
            evaluated_radius_hypotheses,
            ratio: Some(published_ratio),
            evidence_selected_ratio: Some(evidence_selected_ratio),
            score: best.objective,
            raw_best_ratio: Some(raw_best_ratio),
            raw_best_score,
            incumbent_ratio,
            incumbent_score,
            incumbent_retained_as_supported_tie,
            kinematically_limited,
            supporting_frames: best.supporting_frames,
            unique_sectors: best.unique_sectors,
            current_defined_candidates,
            current_center_offset: current_fit.as_ref().map(|fit| fit.center_offset),
            current_longest_arc_sectors: current_fit
                .as_ref()
                .map_or(0, |fit| fit.longest_arc_sectors),
            current_matches: current_fit.map_or_else(Vec::new, |fit| fit.matches),
            provisional,
            qualified,
            legacy_radius_ratio,
            legacy_radius_vetoed,
        }
    }
}

fn pupil_polar_cosolve_json(diagnostics: &PupilPolarCoSolveDiagnostics) -> Value {
    json!({
        "diagnostic_only": true,
        "elapsed_us": diagnostics.elapsed_us,
        "evaluated_radius_hypotheses": diagnostics.evaluated_radius_hypotheses,
        "missing_sectors_are_negative_evidence": false,
        "ratio": diagnostics.ratio,
        "evidence_selected_ratio": diagnostics.evidence_selected_ratio,
        "score": diagnostics.score,
        "raw_best_ratio": diagnostics.raw_best_ratio,
        "raw_best_score": diagnostics.raw_best_score,
        "incumbent_ratio": diagnostics.incumbent_ratio,
        "incumbent_score": diagnostics.incumbent_score,
        "incumbent_retained_as_supported_tie": diagnostics.incumbent_retained_as_supported_tie,
        "kinematically_limited": diagnostics.kinematically_limited,
        "supporting_frames": diagnostics.supporting_frames,
        "unique_sectors": diagnostics.unique_sectors,
        "current_defined_candidates": diagnostics.current_defined_candidates,
        "current_center_offset_in_limbus_coordinates": diagnostics.current_center_offset,
        "current_longest_arc_sectors": diagnostics.current_longest_arc_sectors,
        "current_longest_arc_degrees": diagnostics.current_longest_arc_sectors as f64 * 360.0 / PUPIL_POLAR_SECTORS as f64,
        "current_matches": diagnostics.current_matches.iter().map(|matched| json!({
            "sector": matched.sector,
            "source_radius_ratio": matched.source_radius_ratio,
            "residual": matched.residual,
            "quality": matched.quality,
            "match_score": matched.match_score,
        })).collect::<Vec<_>>(),
        "provisional": diagnostics.provisional,
        "qualified": diagnostics.qualified,
        "legacy_radius_ratio": diagnostics.legacy_radius_ratio,
        "legacy_radius_vetoed_by_defined_arc": diagnostics.legacy_radius_vetoed,
    })
}

#[cfg(test)]
mod pupil_polar_tests {
    use super::*;

    fn replay_test_outer(center: (f64, f64)) -> raw_iris_focus::OuterIrisBoundary {
        let mut boundary = raw_iris_focus::OuterIrisBoundary::default();
        boundary.center = center;
        boundary.major_radius = 40.0;
        boundary.minor_radius = 36.0;
        boundary.points = vec![raw_iris_focus::OuterIrisPoint {
            x: center.0 + 40.0,
            y: center.1,
            contrast: 1.0,
        }];
        boundary
    }

    #[test]
    fn replay_pupil_never_uses_a_rejected_diagnostic_conic() {
        let glasses_junction_candidate = replay_test_outer((177.99, 110.47));
        assert!(
            select_common_pupil_outer_for_replay(None, &glasses_junction_candidate, false,)
                .is_none(),
            "a rejected native diagnostic must not create pupil state"
        );

        let (native, source) =
            select_common_pupil_outer_for_replay(None, &glasses_junction_candidate, true)
                .expect("an independently admitted native limbus is usable");
        assert_eq!(source, "native-published");
        assert_eq!(native.center, glasses_junction_candidate.center);

        let driving = replay_test_outer((210.0, 120.0));
        let (selected, source) = select_common_pupil_outer_for_replay(
            Some(driving.clone()),
            &glasses_junction_candidate,
            true,
        )
        .expect("a published Driving limbus is usable");
        assert_eq!(source, "driving-published");
        assert_eq!(selected.center, driving.center);
    }

    fn polar_candidate(sector: usize, radius_ratio: f64) -> PupilPolarCandidateObservation {
        let angle = std::f64::consts::TAU * sector as f64 / PUPIL_POLAR_SECTORS as f64;
        PupilPolarCandidateObservation {
            sector,
            canonical: (radius_ratio * angle.cos(), radius_ratio * angle.sin()),
            radius_ratio_from_rough_center: radius_ratio,
            quality: 1.0,
        }
    }

    fn polar_frame(
        at: Instant,
        sectors: &[usize],
        radius_ratio: f64,
    ) -> PupilPolarFrameObservation {
        PupilPolarFrameObservation::new(
            at,
            sectors
                .iter()
                .copied()
                .map(|sector| polar_candidate(sector, radius_ratio))
                .collect(),
        )
    }

    fn synthetic_sparse_boundary(
        pose: DrivingAffinePose,
        ratio: f64,
    ) -> raw_iris_focus::InnerIrisBoundary {
        let (axis_sine, axis_cosine) = pose.angle.sin_cos();
        let radial_candidates = [0usize, 1]
            .into_iter()
            .map(|sector| {
                let angle = std::f64::consts::TAU * sector as f64 / PUPIL_POLAR_SECTORS as f64;
                let local = (
                    ratio * pose.major_radius * angle.cos(),
                    ratio * pose.minor_radius * angle.sin(),
                );
                raw_iris_focus::InnerIrisRadialCandidate {
                    sector_index: sector as u8,
                    angle,
                    equivalent_radius_px: ratio * (pose.major_radius * pose.minor_radius).sqrt(),
                    x: pose.center.0 + axis_cosine * local.0 - axis_sine * local.1,
                    y: pose.center.1 + axis_sine * local.0 + axis_cosine * local.1,
                    raw_score: 0.80,
                    peak_prominence: 0.20,
                    luma_transition: 0.55,
                    chroma_transition: 0.25,
                    void_drop: 0.24,
                    inside_void: 0.82,
                    broad_dark_step: 0.20,
                }
            })
            .collect();
        raw_iris_focus::InnerIrisBoundary {
            center: pose.center,
            radius: ratio * (pose.major_radius * pose.minor_radius).sqrt(),
            major_radius: pose.major_radius * ratio,
            minor_radius: pose.minor_radius * ratio,
            angle: pose.angle,
            points: Vec::new(),
            radial_candidates,
        }
    }

    #[test]
    fn isolated_sharp_radial_fragments_cannot_nominate_a_circle() {
        let at = Instant::now();
        let isolated = polar_frame(at, &[4], 0.28);
        assert!(pupil_polar_fit_at_offset(&isolated, 0.28, (0.0, 0.0)).is_none());

        let separated = polar_frame(at, &[4, 11], 0.28);
        assert!(pupil_polar_fit_at_offset(&separated, 0.28, (0.0, 0.0)).is_none());

        let adjacent = polar_frame(at, &[4, 5], 0.28);
        let fit = pupil_polar_fit_at_offset(&adjacent, 0.28, (0.0, 0.0))
            .expect("an adjacent pair defines a short polar slice");
        assert_eq!(fit.supported_mask.count_ones(), 2);
        assert_eq!(fit.longest_arc_sectors, 2);
    }

    #[test]
    fn separable_center_transition_matches_the_brute_force_viterbi_step() {
        let scores = (0..PUPIL_POLAR_CENTER_STATES)
            .map(|state| {
                let scrambled = (state * 37 + 11) % 127;
                scrambled as f64 * 0.013 - state as f64 * 0.000_071
            })
            .collect::<Vec<_>>();
        let emissions = (0..PUPIL_POLAR_CENTER_STATES)
            .map(|state| {
                (state % 4 != 0).then_some(PupilPolarFrameEmission {
                    score: ((state * 19 + 3) % 41) as f64 * 0.007,
                    ..PupilPolarFrameEmission::default()
                })
            })
            .collect::<Vec<_>>();
        let (fast_scores, fast_predecessors) = pupil_polar_advance_center_path(&scores, &emissions);
        let axis_transition_penalty = |delta: f64| 0.12 * (delta / 0.025).powi(2);
        for state in 0..PUPIL_POLAR_CENTER_STATES {
            let offset = pupil_polar_center_offset(state);
            let emission = emissions[state].map_or(0.0, |fit| fit.score);
            let mut reference_score = f64::NEG_INFINITY;
            let mut reference_predecessor = 0usize;
            for previous in 0..PUPIL_POLAR_CENTER_STATES {
                let previous_offset = pupil_polar_center_offset(previous);
                let candidate = scores[previous]
                    - axis_transition_penalty(offset.0 - previous_offset.0)
                    - axis_transition_penalty(offset.1 - previous_offset.1)
                    + emission
                    - pupil_polar_anchor_penalty(offset);
                if candidate > reference_score {
                    reference_score = candidate;
                    reference_predecessor = previous;
                }
            }
            assert!(
                (fast_scores[state] - reference_score).abs() < 1.0e-12,
                "state={state} fast={} reference={reference_score}",
                fast_scores[state]
            );
            assert_eq!(fast_predecessors[state], reference_predecessor);
        }
    }

    #[test]
    fn sorted_precomputed_radius_window_matches_the_direct_fragment_scan() {
        let at = Instant::now();
        let frame = PupilPolarFrameObservation::new(
            at,
            (0..PUPIL_POLAR_SECTORS)
                .flat_map(|sector| {
                    [0.22, 0.285, 0.39]
                        .into_iter()
                        .map(move |ratio| polar_candidate(sector, ratio))
                })
                .collect(),
        );
        for state in 0..PUPIL_POLAR_CENTER_STATES {
            let center_offset = pupil_polar_center_offset(state);
            for ratio in [0.18, 0.24, 0.275, 0.31, 0.405] {
                let direct = pupil_polar_fit_at_offset(&frame, ratio, center_offset);
                let prepared = pupil_polar_fit_at_state(&frame, ratio, state);
                assert_eq!(direct.is_some(), prepared.is_some());
                if let (Some(direct), Some(prepared)) = (direct, prepared) {
                    assert!((direct.score - prepared.score).abs() < 1.0e-12);
                    assert_eq!(direct.supported_mask, prepared.supported_mask);
                    assert_eq!(direct.longest_arc_sectors, prepared.longest_arc_sectors);
                    assert_eq!(direct.matches.len(), prepared.matches.len());
                    for (direct, prepared) in direct.matches.iter().zip(&prepared.matches) {
                        assert_eq!(direct.sector, prepared.sector);
                        assert!((direct.residual - prepared.residual).abs() < 1.0e-12);
                        assert!((direct.match_score - prepared.match_score).abs() < 1.0e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn adjacent_interval_gate_never_rejects_an_exactly_provisional_ratio() {
        let started = Instant::now();
        let frame = |at, observations: &[(usize, f64)]| {
            PupilPolarFrameObservation::new(
                at,
                observations
                    .iter()
                    .map(|(sector, ratio)| polar_candidate(*sector, *ratio))
                    .collect(),
            )
        };
        let frames = VecDeque::from([
            frame(started, &[(0, 0.25), (1, 0.27), (8, 0.34), (9, 0.35)]),
            frame(
                started + Duration::from_millis(80),
                &[(1, 0.26), (2, 0.28), (13, 0.31), (14, 0.33)],
            ),
            frame(
                started + Duration::from_millis(160),
                &[(6, 0.24), (7, 0.265), (19, 0.36), (20, 0.37)],
            ),
        ]);
        for step in 0..=140 {
            let ratio = 0.10 + step as f64 * 0.0025;
            let exact = pupil_polar_ratio_fit(&frames, ratio);
            if exact.supporting_frames >= 2 && exact.unique_sectors >= 3 {
                assert!(
                    pupil_polar_ratio_can_possibly_qualify(&frames, ratio),
                    "gate rejected exact provisional fit at ratio {ratio}: {exact:?}"
                );
            }
        }
    }

    #[test]
    fn temporally_complementary_slices_cosolve_without_penalizing_missing_sectors() {
        let started = Instant::now();
        let informative = VecDeque::from([
            polar_frame(started, &[1, 2], 0.28),
            polar_frame(started + Duration::from_millis(100), &[7, 8], 0.28),
            polar_frame(started + Duration::from_millis(200), &[13, 14], 0.28),
        ]);
        assert!(pupil_polar_ratio_can_possibly_qualify(&informative, 0.28));
        let correct = pupil_polar_ratio_fit(&informative, 0.28);
        assert_eq!(correct.supporting_frames, 3);
        assert_eq!(correct.unique_sectors, 6);
        assert!(correct.objective > 1.0, "fit={correct:?}");

        let wrong = pupil_polar_ratio_fit(&informative, 0.38);
        assert_eq!(wrong.supporting_frames, 0, "fit={wrong:?}");
        assert!(wrong.objective < correct.objective);

        let mut with_unknown_frame = informative.clone();
        with_unknown_frame.insert(
            1,
            PupilPolarFrameObservation::new(started + Duration::from_millis(50), Vec::new()),
        );
        let with_unknown = pupil_polar_ratio_fit(&with_unknown_frame, 0.28);
        assert_eq!(with_unknown.supporting_frames, correct.supporting_frames);
        assert_eq!(with_unknown.unique_sectors, correct.unique_sectors);
        assert!(
            (with_unknown.objective - correct.objective).abs() < 1.0e-12,
            "a missing/reflected sector became negative evidence: correct={correct:?} with_unknown={with_unknown:?}"
        );
    }

    #[test]
    fn sensor_registered_pose_continuity_survives_roi_motion_and_resets_on_a_new_limbus() {
        let started = Instant::now();
        let first_pose = DrivingAffinePose {
            center: (100.0, 80.0),
            major_radius: 80.0,
            minor_radius: 60.0,
            angle: 0.15,
        };
        let mut solver = PupilPolarCoSolver::default();
        solver.observe(
            started,
            (1_000, 2_000),
            &synthetic_sparse_boundary(first_pose, 0.28),
            first_pose,
            (0.08, 0.72),
        );
        assert_eq!(solver.frames.len(), 1);

        // The client-local center moved because the ROI origin moved by the
        // opposite amount; its sensor-space limbus is identical.
        let steered_pose = DrivingAffinePose {
            center: (80.0, 70.0),
            ..first_pose
        };
        solver.observe(
            started + Duration::from_millis(100),
            (1_020, 2_010),
            &synthetic_sparse_boundary(steered_pose, 0.28),
            steered_pose,
            (0.08, 0.72),
        );
        assert_eq!(
            solver.frames.len(),
            2,
            "ROI steering erased physical continuity"
        );

        let unrelated_pose = DrivingAffinePose {
            center: (180.0, 70.0),
            ..steered_pose
        };
        solver.observe(
            started + Duration::from_millis(200),
            (1_020, 2_010),
            &synthetic_sparse_boundary(unrelated_pose, 0.28),
            unrelated_pose,
            (0.08, 0.72),
        );
        assert_eq!(
            solver.frames.len(),
            1,
            "unrelated limbus coordinate systems completed each other's missing slices"
        );
    }

    #[test]
    fn stale_sparse_history_expires_without_rate_limiting_a_new_physical_lock() {
        let started = Instant::now();
        let pose = DrivingAffinePose {
            center: (100.0, 80.0),
            major_radius: 80.0,
            minor_radius: 60.0,
            angle: 0.0,
        };
        let mut solver = PupilPolarCoSolver::default();
        solver.published_ratio = Some((started, 0.22));
        solver.last_qualified_ratio = Some((started, 0.22));
        solver.observe(
            started,
            (1_000, 2_000),
            &synthetic_sparse_boundary(pose, 0.28),
            pose,
            (0.08, 0.72),
        );
        solver.observe(
            started + PUPIL_POLAR_HISTORY + Duration::from_millis(1),
            (1_000, 2_000),
            &synthetic_sparse_boundary(pose, 0.28),
            pose,
            (0.08, 0.72),
        );
        assert_eq!(solver.frames.len(), 1);
        assert_eq!(solver.published_ratio, None);
        assert_eq!(
            solver.last_qualified_ratio,
            Some((started, 0.22)),
            "expired evidence should not erase the soft physical hypothesis"
        );
    }
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize]
}

fn distribution(values: &[f64]) -> Value {
    json!({
        "samples": values.len(),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "maximum": percentile(values, 1.0),
    })
}

fn integer(record: &Value, field: &str) -> Result<u64, String> {
    record
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing integer field {field}"))
}

fn parse_usize(value: Option<String>, default: usize, label: &str) -> Result<usize, String> {
    value.map_or(Ok(default), |value| {
        value
            .parse::<usize>()
            .map_err(|error| format!("invalid {label} {value}: {error}"))
    })
}

fn ellipse_json(center: (f64, f64), major_radius: f64, minor_radius: f64, angle: f64) -> Value {
    let projection =
        raw_iris_focus::assess_projected_circular_limbus_axes(major_radius, minor_radius);
    json!({
        "center": center,
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "minor_to_major": minor_radius.min(major_radius) / major_radius.max(minor_radius).max(1.0e-9),
        "angle": angle,
        "frontal_parallel_radius_px": major_radius.max(minor_radius),
        "projected_area_equivalent_radius_px": (major_radius * minor_radius).max(0.0).sqrt(),
        "projected_area_px2": std::f64::consts::PI * major_radius * minor_radius,
        "central_camera_projection": projection.map(|assessment| json!({
            "admissible": assessment.minor_to_major + 1.0e-12 >= assessment.minimum_minor_to_major,
            "minimum_minor_to_major": assessment.minimum_minor_to_major,
            "uncorrected_image_implied_tilt_degrees": assessment.uncorrected_image_implied_tilt_radians.to_degrees(),
            "maximum_supported_surface_tilt_degrees": assessment.maximum_supported_surface_tilt_radians.to_degrees(),
            "maximum_supported_image_tilt_degrees": assessment.maximum_supported_image_tilt_radians.to_degrees(),
            "provisional_focal_length_px": [
                raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.minimum_focal_length_px,
                raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.maximum_focal_length_px,
            ],
        })),
    })
}

fn boundary_json(boundary: &raw_iris_focus::OuterIrisBoundary) -> Value {
    if boundary.points.is_empty() {
        return Value::Null;
    }
    let mut value = ellipse_json(
        boundary.center,
        boundary.major_radius,
        boundary.minor_radius,
        boundary.angle,
    );
    let Value::Object(ref mut object) = value else {
        unreachable!();
    };
    object.insert(
        "evidence_points".to_string(),
        json!(boundary
            .evidence_points
            .iter()
            .map(|point| [point.x, point.y])
            .collect::<Vec<_>>()),
    );
    object.insert(
        "occluded_points".to_string(),
        json!(boundary
            .occluded_points
            .iter()
            .map(|point| [point.x, point.y])
            .collect::<Vec<_>>()),
    );
    value
}

fn partial_json(partial: Option<raw_iris_focus::RoiTruncatedLimbusObservation>) -> Value {
    partial.map_or(Value::Null, |partial| {
        let mut value = ellipse_json(
            partial.center,
            partial.major_radius,
            partial.minor_radius,
            partial.angle,
        );
        let Value::Object(ref mut object) = value else {
            unreachable!();
        };
        object.insert(
            "visible_arc_fraction".to_string(),
            json!(partial.visible_arc_fraction),
        );
        object.insert(
            "supported_probe_fraction".to_string(),
            json!(partial.supported_probe_fraction),
        );
        object.insert("confidence".to_string(), json!(partial.confidence));
        object.insert("censored_edges".to_string(), json!(partial.censored_edges));
        value
    })
}

fn diagnostics_json(diagnostics: raw_iris_focus::OuterIrisDiagnostics) -> Value {
    let outward_topology_detail = json!({
        "observable_left": diagnostics.outward_topology_observable_left,
        "supported_left": diagnostics.outward_topology_supported_left,
        "observable_right": diagnostics.outward_topology_observable_right,
        "supported_right": diagnostics.outward_topology_supported_right,
        "mean_score_left": diagnostics.outward_topology_mean_score_left,
        "mean_score_right": diagnostics.outward_topology_mean_score_right,
        "mean_limbus_order_left": diagnostics.outward_topology_mean_limbus_order_left,
        "mean_limbus_order_right": diagnostics.outward_topology_mean_limbus_order_right,
        "mean_ridge_distance_left_px": diagnostics.outward_topology_mean_ridge_distance_left_px,
        "mean_ridge_distance_right_px": diagnostics.outward_topology_mean_ridge_distance_right_px,
        "longest_coherent_run_left": diagnostics.outward_topology_longest_coherent_run_left,
        "longest_coherent_run_right": diagnostics.outward_topology_longest_coherent_run_right,
    });
    json!({
        "seed_usable": diagnostics.seed_usable,
        "accepted": diagnostics.accepted,
        "work_stride": diagnostics.work_stride,
        "sample_stride": diagnostics.sample_stride,
        "elapsed_us": diagnostics.elapsed_us,
        "ray_batch_elapsed_us": diagnostics.ray_batch_elapsed_us,
        "max_ray_elapsed_us": diagnostics.max_ray_elapsed_us,
        "active_rays": diagnostics.active_rays,
        "candidate_rays": diagnostics.candidate_rays,
        "candidate_count": diagnostics.candidate_count,
        "ray_overruns": diagnostics.ray_overruns,
        "ray_batch_timeouts": diagnostics.ray_batch_timeouts,
        "refinement_elapsed_us": diagnostics.refinement_elapsed_us,
        "refinement_iterations": diagnostics.refinement_iterations,
        "sector_overruns": diagnostics.sector_overruns,
        "opposing_supported": diagnostics.opposing_supported,
        "selected_right": diagnostics.selected_right,
        "selected_left": diagnostics.selected_left,
        "selected_lower": diagnostics.selected_lower,
        "outward_topology_observable": diagnostics.outward_topology_observable,
        "outward_topology_supported": diagnostics.outward_topology_supported,
        "outward_topology_detail": outward_topology_detail,
        "flat_rejected": diagnostics.flat_rejected,
        "occlusion_recovered": diagnostics.occlusion_recovered,
        "analog_force_samples": diagnostics.analog_force_samples,
        "analog_force_outward": diagnostics.analog_force_outward,
        "analog_force_inward": diagnostics.analog_force_inward,
        "analog_mean_signed_offset_px": diagnostics.analog_mean_signed_offset_px,
        "analog_mean_power": diagnostics.analog_mean_power,
        "analog_mean_certainty": diagnostics.analog_mean_certainty,
        "analog_refinement_elapsed_us": diagnostics.analog_refinement_elapsed_us,
        "analog_fit_applied": diagnostics.analog_fit_applied,
    })
}

fn feature_cluster_diagnostics_json(
    diagnostics: raw_motion_octrees::FeatureClusterIrisDiagnostics,
) -> Value {
    json!({
        "rejection": diagnostics.rejection,
        "semantic_split": diagnostics.semantic_split,
        "seed_available": diagnostics.seed_available,
        "eligible_layers": diagnostics.eligible_layers,
        "associated_edges": diagnostics.associated_edges,
        "fitted_layers": diagnostics.fitted_layers,
        "best_edge_confidence": diagnostics.best_edge_confidence,
        "best_seed_confidence": diagnostics.best_seed_confidence,
        "best_edge_score_gain": diagnostics.best_edge_score_gain,
        "best_angular_coverage": diagnostics.best_angular_coverage,
        "best_opposing_meridians": diagnostics.best_opposing_meridians,
    })
}

fn native_pupil_horizon_json(
    raw: &[u16],
    width: usize,
    height: usize,
    boundary: &raw_iris_focus::OuterIrisBoundary,
    focus: &raw_iris_focus::BorderFocus,
) -> Value {
    let evaluation = (!boundary.points.is_empty())
        .then_some(())
        .and(focus.pupil_hint)
        .and_then(|pupil| {
            evaluate_driving_pupil_horizon(
                raw,
                width,
                height,
                DrivingAffinePose {
                    center: boundary.center,
                    major_radius: boundary.major_radius,
                    minor_radius: boundary.minor_radius,
                    angle: boundary.angle,
                },
                pupil,
                1.0,
            )
        });
    evaluation.map_or(Value::Null, |evaluation| {
        json!({
            "score": evaluation.score,
            "left_transition": evaluation.left_transition,
            "right_transition": evaluation.right_transition,
            "pupil_canonical": evaluation.pupil_canonical,
        })
    })
}

fn motion_layers_json(overlay: &raw_motion_octrees::MotionOctreeOverlay) -> Value {
    Value::Array(
        (0..raw_motion_octrees::OBJECTS)
            .map(|object| {
                let layer = overlay.layers[object];
                let motion = overlay.motions[object];
                json!({
                    "object": object,
                    "label": raw_motion_octrees::motion_layer_label(object),
                    "persistent_tracks": layer.persistent_tracks,
                    "stable_frames": layer.stable_frames,
                    "coherence": layer.coherence,
                    "separation": layer.separation,
                    "trajectory_error": layer.trajectory_error,
                    "signature_samples": layer.signature_samples,
                    // ROI-local centroid of the exact persistent tracks that
                    // trained this layer.  In particular, object 2 is the
                    // compact neutral/specular cluster already separated by
                    // native RAW photometry; exporting it lets replay audit
                    // whether a proposed iris conic actually contains its
                    // corneal reflection instead of merely sharing its scale.
                    "centroid": layer.centroid,
                    "motion_support": motion.support,
                    "motion_residual": motion.residual,
                    "translation": motion.translation,
                    "rotation": motion.rotation,
                    "scale_delta": motion.scale_delta,
                })
            })
            .collect(),
    )
}

fn pupil_center_motion_gate_json(overlay: &raw_motion_octrees::MotionOctreeOverlay) -> Value {
    let coupled = overlay.coupled_motion;
    let relative = coupled.green_relative_to_cyan;
    let acceleration = relative.acceleration_px_s2[0].hypot(relative.acceleration_px_s2[1]);
    let jerk = relative.jerk_px_s3[0].hypot(relative.jerk_px_s3[1]);
    let pupil_layer = overlay.layers[raw_motion_octrees::PUPIL_LAYER];
    let pupil_motion = overlay.motions[raw_motion_octrees::PUPIL_LAYER];
    let general_layer = overlay.layers[raw_motion_octrees::GENERAL_LAYER];
    let general_motion = overlay.motions[raw_motion_octrees::GENERAL_LAYER];
    json!({
        "immediate_saccade_supported": pupil_center_saccade_motion_supported(overlay),
        "broad_search_warranted": pupil_center_saccade_search_warranted(overlay),
        "timestamp_ns": coupled.timestamp_ns,
        "dt_ms": coupled.dt_ms,
        "saccade_likelihood": coupled.saccade_likelihood,
        "micro_motion_likelihood": coupled.micro_motion_likelihood,
        "cyan": {
            "samples": coupled.cyan.samples,
            "confidence": coupled.cyan.confidence,
            "reference_point": coupled.cyan.reference_point,
            "speed_px_s": coupled.cyan.speed_px_s,
            "fit_residual_px": coupled.cyan.fit_residual_px,
        },
        "green": {
            "samples": coupled.green.samples,
            "confidence": coupled.green.confidence,
            "reference_point": coupled.green.reference_point,
            "speed_px_s": coupled.green.speed_px_s,
            "fit_residual_px": coupled.green.fit_residual_px,
        },
        "relative_reference_point": relative.reference_point,
        "relative_samples": relative.samples,
        "relative_confidence": relative.confidence,
        "relative_speed_px_s": relative.speed_px_s,
        "relative_acceleration_px_s2": acceleration,
        "relative_jerk_px_s3": jerk,
        "relative_angular_velocity_rad_s": relative.angular_velocity_rad_s,
        "general_layer": {
            "persistent_tracks": general_layer.persistent_tracks,
            "stable_frames": general_layer.stable_frames,
            "coherence": general_layer.coherence,
            "trajectory_error": general_layer.trajectory_error,
            "motion_support": general_motion.support,
            "motion_residual": general_motion.residual,
        },
        "pupil_layer": {
            "persistent_tracks": pupil_layer.persistent_tracks,
            "stable_frames": pupil_layer.stable_frames,
            "coherence": pupil_layer.coherence,
            "trajectory_error": pupil_layer.trajectory_error,
            "motion_support": pupil_motion.support,
            "motion_residual": pupil_motion.residual,
        },
        "specular_layer_excluded": true,
    })
}

fn cluster_json(hypothesis: Option<&raw_motion_octrees::FeatureClusterIrisHypothesis>) -> Value {
    hypothesis.map_or(Value::Null, |hypothesis| {
        let mut value = ellipse_json(
            hypothesis.center,
            hypothesis.major_radius,
            hypothesis.minor_radius,
            hypothesis.angle,
        );
        let Value::Object(ref mut object) = value else {
            unreachable!();
        };
        object.insert("score".to_string(), json!(hypothesis.score));
        object.insert("motion_layer".to_string(), json!(hypothesis.motion_layer));
        object.insert(
            "layer_coherence".to_string(),
            json!(hypothesis.layer_coherence),
        );
        object.insert(
            "layer_separation".to_string(),
            json!(hypothesis.layer_separation),
        );
        object.insert(
            "layer_parallax".to_string(),
            json!(hypothesis.layer_parallax),
        );
        object.insert(
            "layer_stable_frames".to_string(),
            json!(hypothesis.layer_stable_frames),
        );
        object.insert(
            "seed_edge_score".to_string(),
            json!(hypothesis.seed_edge_score),
        );
        object.insert(
            "edge_score_gain".to_string(),
            json!(hypothesis.edge_score_gain),
        );
        object.insert("edge_support".to_string(), json!(hypothesis.edge_support));
        object.insert(
            "angular_coverage".to_string(),
            json!(hypothesis.angular_coverage),
        );
        object.insert(
            "opposing_meridians".to_string(),
            json!(hypothesis.opposing_meridians),
        );
        object.insert(
            "bridged_current_frame_edges".to_string(),
            json!(hypothesis.bridged_current_frame_edges),
        );
        object.insert("evidence_points".to_string(), json!(hypothesis.features));
        value
    })
}

fn driving_json(hypothesis: Option<DrivingHypothesis>) -> Value {
    hypothesis.map_or(Value::Null, |hypothesis| {
        let mut value = ellipse_json(
            hypothesis.pose.center,
            hypothesis.pose.major_radius,
            hypothesis.pose.minor_radius,
            hypothesis.pose.angle,
        );
        let Value::Object(ref mut object) = value else {
            unreachable!();
        };
        object.insert("score".to_string(), json!(hypothesis.score));
        object.insert("normal_score".to_string(), json!(hypothesis.normal_score));
        object.insert("white_score".to_string(), json!(hypothesis.white_score));
        object.insert(
            "far_sclera_score".to_string(),
            json!(hypothesis.far_sclera_score),
        );
        object.insert("limbus_score".to_string(), json!(hypothesis.limbus_score));
        object.insert("pupil_score".to_string(), json!(hypothesis.pupil_score));
        object.insert(
            "pupil_enclosure".to_string(),
            json!(hypothesis.pupil_enclosure),
        );
        object.insert("pupil_margin".to_string(), json!(hypothesis.pupil_margin));
        let pupil_boundary = hypothesis
            .pupil_projected_area_radius_px
            .filter(|radius| radius.is_finite() && *radius > 0.0)
            .and_then(|projected_radius| {
                let major = hypothesis
                    .pose
                    .major_radius
                    .max(hypothesis.pose.minor_radius);
                let minor = hypothesis
                    .pose
                    .major_radius
                    .min(hypothesis.pose.minor_radius);
                let axis_ratio = minor / major.max(1.0e-9);
                let outer_projected_radius = (major * minor).sqrt();
                if !(raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE
                    .absolute_minimum_minor_to_major..=1.0)
                    .contains(&axis_ratio)
                    || !outer_projected_radius.is_finite()
                    || outer_projected_radius <= 0.0
                {
                    return None;
                }
                let fronto_parallel_radius = projected_radius / axis_ratio.sqrt();
                let radius_ratio = projected_radius / outer_projected_radius;
                let center = hypothesis.pupil_boundary_center();
                let mut value = ellipse_json(
                    center,
                    fronto_parallel_radius,
                    fronto_parallel_radius * axis_ratio,
                    hypothesis.pose.angle,
                );
                let Value::Object(ref mut pupil) = value else {
                    unreachable!();
                };
                pupil.insert(
                    "projected_area_equivalent_radius_px".to_string(),
                    json!(projected_radius),
                );
                pupil.insert(
                    "frontal_parallel_radius_px".to_string(),
                    json!(fronto_parallel_radius),
                );
                pupil.insert(
                    "pupil_to_limbus_radius_ratio".to_string(),
                    json!(radius_ratio),
                );
                pupil.insert(
                    "pupil_to_limbus_area_fraction".to_string(),
                    json!(radius_ratio * radius_ratio),
                );
                Some(value)
            })
            .unwrap_or(Value::Null);
        object.insert("pupil_boundary".to_string(), pupil_boundary);
        object.insert("pupil_horizon".to_string(), json!(hypothesis.pupil_horizon));
        object.insert(
            "through_eye_score".to_string(),
            json!(hypothesis.through_eye_score),
        );
        object.insert(
            "bilateral_limbus_order".to_string(),
            json!(hypothesis.bilateral_limbus_order),
        );
        object.insert(
            "lower_limbus_direct_visibility".to_string(),
            json!(hypothesis.lower_limbus_direct_visibility),
        );
        object.insert(
            "light_cohesion".to_string(),
            json!(hypothesis.light_cohesion),
        );
        object.insert(
            "affine_departure_fraction".to_string(),
            json!(hypothesis.affine_departure_fraction),
        );
        object.insert(
            "affine_repair_fraction".to_string(),
            json!(hypothesis.affine_repair_fraction),
        );
        object.insert(
            "affine_reinforced".to_string(),
            json!(hypothesis.affine_reinforced),
        );
        object.insert(
            "pupil_canonical".to_string(),
            json!(hypothesis.pupil_boundary_canonical()),
        );
        object.insert(
            "pupil_center".to_string(),
            json!(hypothesis.pupil_boundary_center()),
        );
        object.insert(
            "pupil_topology_canonical".to_string(),
            json!(hypothesis.pupil_canonical),
        );
        object.insert(
            "pupil_topology_center".to_string(),
            json!(driving_pose_point(
                hypothesis.pose,
                hypothesis.pupil_canonical,
            )),
        );
        object.insert("iterations".to_string(), json!(hypothesis.iterations));
        object.insert(
            "refinement_laps".to_string(),
            json!(hypothesis.refinement_laps),
        );
        value
    })
}

fn driving_inner_boundary_json(
    boundary: raw_iris_focus::InnerIrisBoundary,
    pose: DrivingAffinePose,
) -> Value {
    if boundary.points.len() < 8 || !boundary.radius.is_finite() || boundary.radius <= 0.0 {
        return Value::Null;
    }
    let outer_projected_radius = (pose.major_radius * pose.minor_radius).sqrt();
    if !outer_projected_radius.is_finite() || outer_projected_radius <= 0.0 {
        return Value::Null;
    }
    let radius_ratio = boundary.radius / outer_projected_radius;
    let mut value = ellipse_json(
        boundary.center,
        boundary.major_radius,
        boundary.minor_radius,
        boundary.angle,
    );
    let Value::Object(ref mut object) = value else {
        unreachable!();
    };
    object.insert(
        "projected_area_equivalent_radius_px".to_string(),
        json!(boundary.radius),
    );
    object.insert(
        "pupil_to_limbus_radius_ratio".to_string(),
        json!(radius_ratio),
    );
    object.insert(
        "pupil_to_limbus_area_fraction".to_string(),
        json!(radius_ratio * radius_ratio),
    );
    object.insert("point_count".to_string(), json!(boundary.points.len()));
    let diameter_evidence = pupil_diameter_arc_evidence(&boundary);
    object.insert(
        "diameter_arc_evidence".to_string(),
        json!({
            "qualified": diameter_evidence.qualified(),
            "total_points": diameter_evidence.total_points,
            "coherent_points": diameter_evidence.coherent_points,
            "strong_alternative_points": diameter_evidence.strong_alternative_points,
            "opposed_coherent_points": diameter_evidence.opposed_coherent_points,
            "longest_contiguous_arc_points": diameter_evidence.longest_contiguous_arc_points,
            "longest_contiguous_arc_degrees": diameter_evidence.longest_contiguous_arc_points as f64 * 360.0 / 21.0,
            "radius_tolerance_px": diameter_evidence.radius_tolerance_px,
            "median_absolute_radius_residual_px": diameter_evidence.median_absolute_radius_residual_px,
            "median_coherent_transition": diameter_evidence.median_coherent_transition,
        }),
    );
    let (pupil_sine, pupil_cosine) = boundary.angle.sin_cos();
    let evidence_points = boundary
        .points
        .iter()
        .map(|point| {
            let dx = point.x - boundary.center.0;
            let dy = point.y - boundary.center.1;
            let local_x = pupil_cosine * dx + pupil_sine * dy;
            let local_y = -pupil_sine * dx + pupil_cosine * dy;
            let normalized_radius = ((local_x / boundary.major_radius).powi(2)
                + (local_y / boundary.minor_radius).powi(2))
            .sqrt();
            let equivalent_radius = boundary.radius * normalized_radius;
            let image_angle = dy.atan2(dx).rem_euclid(std::f64::consts::TAU);
            let ray_index = ((image_angle / std::f64::consts::TAU * 21.0).round() as usize) % 21;
            json!({
                "x": point.x,
                "y": point.y,
                "ray_index": ray_index,
                "image_angle_rad": image_angle,
                // This is deliberately the current-frame RAW evidence at the
                // selected radius, excluding the mass and temporal ranking
                // priors. It lets the review distinguish a measured margin
                // from a merely plausible affine-circle completion.
                "raw_transition": point.score,
                "equivalent_radius_px": equivalent_radius,
                "radius_residual_px": equivalent_radius - boundary.radius,
            })
        })
        .collect::<Vec<_>>();
    object.insert("evidence_points".to_string(), Value::Array(evidence_points));
    object.insert(
        "prior_free_radial_candidates".to_string(),
        Value::Array(
            boundary
                .radial_candidates
                .iter()
                .map(|candidate| {
                    json!({
                        "sector_index": candidate.sector_index,
                        "angle_rad": candidate.angle,
                        "equivalent_radius_px": candidate.equivalent_radius_px,
                        "x": candidate.x,
                        "y": candidate.y,
                        "raw_score": candidate.raw_score,
                        "peak_prominence": candidate.peak_prominence,
                        "luma_transition": candidate.luma_transition,
                        "chroma_transition": candidate.chroma_transition,
                        "void_drop": candidate.void_drop,
                        "inside_void": candidate.inside_void,
                        "broad_dark_step": if candidate.broad_dark_step.is_finite() {
                            Some(candidate.broad_dark_step)
                        } else {
                            None
                        },
                    })
                })
                .collect(),
        ),
    );
    let raw_transitions = boundary
        .points
        .iter()
        .map(|point| point.score)
        .filter(|score| score.is_finite())
        .collect::<Vec<_>>();
    object.insert(
        "median_raw_transition".to_string(),
        json!(percentile(&raw_transitions, 0.50)),
    );
    object.insert(
        "raw_transition_coverage_022".to_string(),
        json!(
            raw_transitions
                .iter()
                .filter(|score| **score >= 0.22)
                .count() as f64
                / raw_transitions.len().max(1) as f64
        ),
    );
    let radius_residuals = boundary
        .points
        .iter()
        .filter_map(|point| {
            let dx = point.x - boundary.center.0;
            let dy = point.y - boundary.center.1;
            let local_x = pupil_cosine * dx + pupil_sine * dy;
            let local_y = -pupil_sine * dx + pupil_cosine * dy;
            let normalized_radius = ((local_x / boundary.major_radius).powi(2)
                + (local_y / boundary.minor_radius).powi(2))
            .sqrt();
            let residual = boundary.radius * (normalized_radius - 1.0);
            residual.is_finite().then_some(residual.abs())
        })
        .collect::<Vec<_>>();
    object.insert(
        "median_absolute_radius_residual_px".to_string(),
        json!(percentile(&radius_residuals, 0.50)),
    );
    value
}

fn pupil_center_track_json(diagnostics: PupilCenterTrackDiagnostics) -> Value {
    json!({
        "regime": diagnostics.regime.label(),
        "transport_source": diagnostics.transport_source.label(),
        "predicted_center": diagnostics.predicted_center,
        "measured_center": diagnostics.measured_center,
        "published_center": diagnostics.published_center,
        "limbus_transport_disagreement_px": diagnostics.limbus_transport_disagreement_px,
        "innovation_px": diagnostics.innovation_px,
        "fixation_gate_px": diagnostics.fixation_gate_px,
        "measurement_score": diagnostics.measurement_score,
        "measurement_admissible": diagnostics.measurement_admissible,
        "transported_hold": diagnostics.transported_hold,
        "pending_relocation_frames": diagnostics.pending_relocation_frames,
        "saccade_likelihood": diagnostics.saccade_likelihood,
        "relative_motion_confidence": diagnostics.relative_motion_confidence,
        "relative_speed_px_s": diagnostics.relative_speed_px_s,
        "relative_acceleration_px_s2": diagnostics.relative_acceleration_px_s2,
        "relative_jerk_px_s3": diagnostics.relative_jerk_px_s3,
        "specular_layer_excluded": diagnostics.specular_layer_excluded,
        "pursuit_predicted": diagnostics.pursuit_predicted,
        "pursuit_velocity_canonical_per_second": diagnostics.pursuit_velocity_canonical_per_second,
    })
}

fn driving_pupil_selection_variants_json(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_origin: (u32, u32),
    hypothesis: Option<DrivingHypothesis>,
) -> Value {
    hypothesis.map_or(Value::Null, |hypothesis| {
        let pose = hypothesis.pose;
        let pupil_center = driving_pose_point(pose, hypothesis.pupil_canonical);
        let coarse = raw_iris_focus::BorderFocus {
            eye_basin_valid: true,
            center: pose.center,
            radius: (pose.major_radius * pose.minor_radius).sqrt(),
            axis_ratio: pose.major_radius / pose.minor_radius.max(1.0),
            axis_angle: pose.angle,
            pupil_hint: Some(pupil_center),
            pupil_hint_score: 1.0,
            ..raw_iris_focus::BorderFocus::default()
        };
        let evaluate = |mass_prior_scale: f64, normalized_radius_penalty: f64| {
            driving_inner_boundary_json(
                raw_iris_focus::debug_inner_iris_boundary_at_center_tuned(
                    raw,
                    width,
                    height,
                    sensor_origin.0,
                    sensor_origin.1,
                    &coarse,
                    pupil_center,
                    mass_prior_scale,
                    normalized_radius_penalty,
                ),
                pose,
            )
        };
        let evaluate_ratio_prior = |ratio: f64, fractional_half_width: f64, hard_envelope: bool| {
            let outer_projected_radius = (pose.major_radius * pose.minor_radius).sqrt();
            let estimate = outer_projected_radius * ratio;
            let prior = raw_iris_focus::InnerIrisRadiusPrior::new(
                estimate,
                estimate * (1.0 - fractional_half_width),
                estimate * (1.0 + fractional_half_width),
                1.0,
            );
            let envelope_half_width = (estimate * fractional_half_width).max(1.25);
            let envelope = hard_envelope
                .then(|| {
                    raw_iris_focus::InnerIrisRadiusEnvelope::new(
                        estimate - envelope_half_width,
                        estimate + envelope_half_width,
                    )
                })
                .flatten();
            driving_inner_boundary_json(
                raw_iris_focus::debug_inner_iris_boundary_at_center_tuned_with_prior(
                    raw,
                    width,
                    height,
                    sensor_origin.0,
                    sensor_origin.1,
                    &coarse,
                    pupil_center,
                    envelope,
                    prior,
                    0.25,
                    0.08,
                ),
                pose,
            )
        };
        let bounded_ratio_grid = (18..=44)
            .step_by(2)
            .map(|percent| {
                let ratio = f64::from(percent) / 100.0;
                (
                    format!("{ratio:.2}"),
                    evaluate_ratio_prior(ratio, 0.07, true),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        json!({
            "stored_production": hypothesis.pupil_projected_area_radius_px.map_or(Value::Null, |_| {
                driving_json(Some(hypothesis))["pupil_boundary"].clone()
            }),
            "production": evaluate(1.0, 0.0),
            "half_mass": evaluate(0.50, 0.0),
            "quarter_mass": evaluate(0.25, 0.0),
            "edge_only": evaluate(0.0, 0.0),
            "quarter_mass_inner_tiebreak": evaluate(0.25, 0.08),
            "edge_only_inner_tiebreak": evaluate(0.0, 0.08),
            "quarter_mass_inner_prior_025": evaluate_ratio_prior(0.25, 0.06, false),
            "quarter_mass_inner_prior_030": evaluate_ratio_prior(0.30, 0.06, false),
            "quarter_mass_inner_prior_035": evaluate_ratio_prior(0.35, 0.06, false),
            "quarter_mass_inner_bounded_025": evaluate_ratio_prior(0.25, 0.07, true),
            "quarter_mass_inner_bounded_030": evaluate_ratio_prior(0.30, 0.07, true),
            "quarter_mass_inner_bounded_035": evaluate_ratio_prior(0.35, 0.07, true),
            "bounded_ratio_grid": bounded_ratio_grid,
        })
    })
}

fn temporal_feature_orbital_probe_json(
    probe: Option<TemporalFeatureOrbitalTopologyProbe>,
) -> Value {
    probe.map_or(Value::Null, |probe| {
        json!({
            "left_lateral_score": probe.left_lateral_score,
            "right_lateral_score": probe.right_lateral_score,
            "perimeter_limbus_score": probe.perimeter_limbus_score,
            "broad_through_eye_score": probe.broad_through_eye_score,
            "double_sclera_10_deg_score": probe.double_sclera_10_deg_score,
            "double_sclera_10_deg_support": probe.double_sclera_10_deg_support,
            "measured_fraction": probe.measured_fraction,
            "mean_absolute_offset_fraction": probe.mean_absolute_offset_fraction,
            "maximum_absolute_offset_fraction": probe.maximum_absolute_offset_fraction,
            "pupil_canonical_radius": probe.pupil_canonical_radius,
            "driven_fit": probe.driven_fit.map(|fit| json!({
                "center": fit.center,
                "major_radius": fit.major_radius,
                "minor_radius": fit.minor_radius,
                "angle": fit.angle,
                "frontal_parallel_radius_px": fit.major_radius.max(fit.minor_radius),
                "projected_area_equivalent_radius_px": (fit.major_radius * fit.minor_radius).max(1.0).sqrt(),
            })),
            "driven_fit_center_fraction": probe.driven_fit_center_fraction,
            "driven_fit_radius_log_error": probe.driven_fit_radius_log_error,
            "driven_fit_area_radius_log_error": probe.driven_fit_area_radius_log_error,
        })
    })
}

fn fit_assessment_json(assessment: Option<DrivingTemporalFitAssessment>) -> Value {
    assessment.map_or(Value::Null, |assessment| {
        json!({
            "strong": assessment.strong,
            "confidence": assessment.confidence,
            "boundary_saturated": assessment.boundary_saturated,
            "radius_innovation_px": assessment.radius_innovation_px,
            "radius_half_width_px": assessment.radius_half_width_px,
            "pose_distance": assessment.pose_distance,
        })
    })
}

fn temporal_feature_center_assessment_json(
    assessment: Option<TemporalFeatureLimbusCenterAssessment>,
) -> Value {
    assessment.map_or(Value::Null, |assessment| {
        json!({
            "checked": assessment.checked,
            "admissible": assessment.admissible,
            "innovation_px": assessment.innovation_px,
            "maximum_innovation_px": assessment.maximum_innovation_px,
            "predicted_center_sensor": assessment.predicted_center_sensor,
            "layer_support": assessment.layer_support,
            "layer_residual_px": assessment.layer_residual_px,
        })
    })
}

fn temporal_feature_semantic_assessment_json(
    assessment: TemporalFeatureSemanticAssessment,
) -> Value {
    json!({
        "admissible": assessment.admissible,
        "reason": assessment.reason,
        "layer_tracks": assessment.layer_tracks,
        "annular_tracks": assessment.annular_tracks,
        "seed_center_fraction": assessment.seed_center_fraction,
        "seed_radius_log_error": assessment.seed_radius_log_error,
        "seed_area_radius_log_error": assessment.seed_area_radius_log_error,
        "seed_normalized_shape_disagreement": assessment.seed_normalized_shape_disagreement,
        "reflection_normalized_radius": assessment.reflection_normalized_radius,
        "reflection_normalized_limit": assessment.reflection_normalized_limit,
        "reflection_tracks": assessment.reflection_tracks,
        "reflection_contained": assessment.reflection_contained,
        "current_raw_geometry_authoritative": assessment.current_raw_geometry_authoritative,
        "projection_geometry_admissible": assessment.projection_geometry_admissible,
        "scale_innovation_log": assessment.scale_innovation_log,
        "scale_kinematically_supported": assessment.scale_kinematically_supported,
        "complete_material": assessment.complete_material,
        "censored_material": assessment.censored_material,
    })
}

fn texture_json(texture: Option<RoiTruncatedIrisTextureEvidence>) -> Value {
    texture.map_or(Value::Null, |texture| {
        json!({
            "interior_edges": texture.interior_edges,
            "motion_shadow_edges": texture.motion_shadow_edges,
            "radial_edges": texture.radial_edges,
            "radial_sectors": texture.radial_sectors,
            "transverse_edges": texture.transverse_edges,
            "transverse_sectors": texture.transverse_sectors,
            "mean_radial_alignment": texture.mean_radial_alignment,
            "radial_orientation_bias": texture.radial_orientation_bias,
            "polar_orientation_anisotropy": texture.polar_orientation_anisotropy,
            "provisional_features": texture.provisional_features,
            "persistent_features": texture.persistent_features,
            "authorizes_probe": texture.authorizes_bounded_topology_probe(),
        })
    })
}

fn driving_limbus_material_json(evidence: Option<DrivingLimbusMaterialEvidence>) -> Value {
    evidence.map_or(Value::Null, |evidence| {
        json!({
            "score": evidence.score,
            "signed_double_canny_mean": evidence.signed_double_canny_mean,
            "signed_double_canny_fraction": evidence.signed_double_canny_fraction,
            "signed_double_canny_sectors": evidence.signed_double_canny_sectors,
            "tangent_persistent_mean": evidence.tangent_persistent_mean,
            "tangent_persistent_fraction": evidence.tangent_persistent_fraction,
            "plateau_edge_dominance_mean": evidence.plateau_edge_dominance_mean,
            "max_supported_arc_fraction": evidence.max_supported_arc_fraction,
            "lateral_signed_double_canny_mean": evidence.lateral_signed_double_canny_mean,
            "left_signed_double_canny_mean": evidence.left_signed_double_canny_mean,
            "right_signed_double_canny_mean": evidence.right_signed_double_canny_mean,
            "lateral_samples": evidence.lateral_samples,
            "left_samples": evidence.left_samples,
            "right_samples": evidence.right_samples,
            "material_cohort_samples": evidence.material_cohort_samples,
            "chroma_separation": evidence.chroma_separation,
            "iris_chroma_cohesion": evidence.iris_chroma_cohesion,
            "sclera_chroma_cohesion": evidence.sclera_chroma_cohesion,
            "iris_tangential_texture": evidence.iris_tangential_texture,
            "sclera_tangential_texture": evidence.sclera_tangential_texture,
            "visible_samples": evidence.visible_samples,
        })
    })
}

fn driving_multibank_limbus_json(evidence: Option<DrivingMultibankLimbusEvidence>) -> Value {
    evidence.map_or(Value::Null, |evidence| {
        json!({
            "score": evidence.score,
            "left_quantile": evidence.left_quantile,
            "right_quantile": evidence.right_quantile,
            "left_mean_support": evidence.left_mean_support,
            "right_mean_support": evidence.right_mean_support,
            "left_supported_fraction": evidence.left_supported_fraction,
            "right_supported_fraction": evidence.right_supported_fraction,
            "weakest_lateral_quantile": evidence.weakest_lateral_quantile,
            "mean_lateral_support": evidence.mean_lateral_support,
            "supported_fraction": evidence.supported_fraction,
            "narrow_mean": evidence.narrow_mean,
            "medium_mean": evidence.medium_mean,
            "broad_mean": evidence.broad_mean,
            "edge_centroid_signed_mean_px": evidence.edge_centroid_signed_mean_px,
            "edge_centroid_absolute_mean_px": evidence.edge_centroid_absolute_mean_px,
            "edge_centroid_coherence": evidence.edge_centroid_coherence,
            "far_sclera_step_mean": evidence.far_sclera_step_mean,
            "outside_plateau_mean": evidence.outside_plateau_mean,
            "outside_secondary_edge_mean": evidence.outside_secondary_edge_mean,
            "upper_vertical_mean_support": evidence.upper_vertical_mean_support,
            "lower_vertical_mean_support": evidence.lower_vertical_mean_support,
            "upper_vertical_quantile": evidence.upper_vertical_quantile,
            "lower_vertical_quantile": evidence.lower_vertical_quantile,
            "upper_vertical_supported_fraction": evidence.upper_vertical_supported_fraction,
            "lower_vertical_supported_fraction": evidence.lower_vertical_supported_fraction,
            "inside_texture": evidence.inside_texture,
            "outside_texture": evidence.outside_texture,
            "samples": evidence.samples,
            "bilateral": evidence.bilateral,
        })
    })
}

fn driving_semantic_eye_json(evidence: Option<DrivingSemanticEyeEvidence>) -> Value {
    let lid = |path: DrivingSemanticLidPathEvidence| {
        json!({
            "points": path.points,
            "correct_side_fraction": path.correct_side_fraction,
            "canonical_horizontal_span": path.canonical_horizontal_span,
            "signed_canonical_median": path.signed_canonical_median,
            "canonical_radius_median": path.canonical_radius_median,
            "quality_median": path.quality_median,
            "plausible": path.plausible,
        })
    };
    evidence.map_or(Value::Null, |evidence| {
        json!({
            "upper_lid": lid(evidence.upper_lid),
            "lower_lid": lid(evidence.lower_lid),
            "annular": {
                "samples": evidence.annular.samples,
                "active_sectors": evidence.annular.active_sectors,
                "median_gradient": evidence.annular.median_gradient,
                "upper_quartile_gradient": evidence.annular.upper_quartile_gradient,
                "global_orientation_anisotropy": evidence.annular.global_orientation_anisotropy,
                "global_gradient_normal_angle": evidence.annular.global_gradient_normal_angle,
                "polar_orientation_anisotropy": evidence.annular.polar_orientation_anisotropy,
                "polar_orientation_bias": evidence.annular.polar_orientation_bias,
                "pupil_darkness_log": evidence.annular.pupil_darkness_log,
                "pupil_samples": evidence.annular.pupil_samples,
                "iris_samples": evidence.annular.iris_samples,
            },
            "straight_band": {
                "samples": evidence.straight_band.samples,
                "exterior_samples": evidence.straight_band.exterior_samples,
                "dark_fraction": evidence.straight_band.dark_fraction,
                "exterior_dark_fraction": evidence.straight_band.exterior_dark_fraction,
                "median_darkness_log": evidence.straight_band.median_darkness_log,
                "tangent_angle": evidence.straight_band.tangent_angle,
                "half_width_px": evidence.straight_band.half_width_px,
                "center_normal_offset_px": evidence.straight_band.center_normal_offset_px,
                "veto": evidence.straight_band.veto,
            },
            "pupil_to_limbus_radius_ratio": evidence.pupil_to_limbus_radius_ratio,
            "pupil_canonical_offset": evidence.pupil_canonical_offset,
            "plausible_lids": evidence.plausible_lids,
            "score": evidence.score,
            "authorizes_cold_identity": evidence.authorizes_cold_identity,
        })
    })
}

fn focus_seed(focus: &raw_iris_focus::BorderFocus) -> Option<raw_motion_octrees::IrisEllipseSeed> {
    if !focus.eye_basin_valid {
        return None;
    }
    let ratio = if focus.axis_ratio.is_finite() && focus.axis_ratio > 0.0 {
        focus
            .axis_ratio
            .max(1.0 / focus.axis_ratio)
            .clamp(1.0, 2.08)
    } else {
        1.0
    };
    Some(raw_motion_octrees::IrisEllipseSeed {
        center: focus.center,
        major_radius: focus.radius * ratio.sqrt(),
        minor_radius: focus.radius / ratio.sqrt(),
        angle: focus.axis_angle,
    })
}

fn boundary_seed(
    boundary: &raw_iris_focus::OuterIrisBoundary,
) -> Option<raw_motion_octrees::IrisEllipseSeed> {
    (!boundary.points.is_empty()).then_some(raw_motion_octrees::IrisEllipseSeed {
        center: boundary.center,
        major_radius: boundary.major_radius,
        minor_radius: boundary.minor_radius,
        angle: boundary.angle,
    })
}

fn partial_seed(
    partial: raw_iris_focus::RoiTruncatedLimbusObservation,
) -> raw_motion_octrees::IrisEllipseSeed {
    raw_motion_octrees::IrisEllipseSeed {
        center: partial.center,
        major_radius: partial.major_radius,
        minor_radius: partial.minor_radius,
        angle: partial.angle,
    }
}

fn select_common_pupil_outer_for_replay(
    driving_admitted: Option<raw_iris_focus::OuterIrisBoundary>,
    native_candidate: &raw_iris_focus::OuterIrisBoundary,
    native_admitted: bool,
) -> Option<(raw_iris_focus::OuterIrisBoundary, &'static str)> {
    driving_admitted
        .filter(|boundary| !boundary.points.is_empty())
        .map(|boundary| (boundary, "driving-published"))
        .or_else(|| {
            (native_admitted && !native_candidate.points.is_empty())
                .then(|| (native_candidate.clone(), "native-published"))
        })
}

pub(super) fn probe_two_d_material<I>(mut args: I) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    let output_path = PathBuf::from(args.next().ok_or_else(|| {
        "usage: buttercup_wayland_raw_eyes --offline-two-d-material-probe OUTPUT.json REPORT.json STREAM.raw10".to_string()
    })?);
    let report_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "missing segmentation replay report".to_string())?,
    );
    let stream_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "missing RAW stream".to_string())?,
    );
    if output_path.exists() {
        return Err(format!("output already exists: {}", output_path.display()));
    }
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }

    let report_file = File::open(&report_path)
        .map_err(|error| format!("open {}: {error}", report_path.display()))?;
    let report: Value = serde_json::from_reader(report_file)
        .map_err(|error| format!("parse {}: {error}", report_path.display()))?;
    let frames = report
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{} has no frames array", report_path.display()))?;
    let mut stream = File::open(&stream_path)
        .map_err(|error| format!("open {}: {error}", stream_path.display()))?;
    let mut probes = Vec::new();
    for frame in frames {
        let Some(candidate) = frame
            .pointer("/two_d_features/candidate")
            .filter(|value| !value.is_null())
        else {
            continue;
        };
        let number = |value: &Value, field: &str| {
            value
                .get(field)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("2D candidate missing numeric {field}"))
        };
        let pair = |value: &Value, field: &str| -> Result<(f64, f64), String> {
            let values = value
                .get(field)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("2D candidate missing {field}"))?;
            if values.len() != 2 {
                return Err(format!("2D candidate {field} is not a pair"));
            }
            Ok((
                values[0]
                    .as_f64()
                    .ok_or_else(|| format!("2D candidate {field}[0] is not numeric"))?,
                values[1]
                    .as_f64()
                    .ok_or_else(|| format!("2D candidate {field}[1] is not numeric"))?,
            ))
        };
        let width = integer(frame, "width")? as usize;
        let height = integer(frame, "height")? as usize;
        let stride = integer(frame, "stride")? as usize;
        let offset = integer(frame, "source_offset")?;
        let length = integer(frame, "source_length")? as usize;
        if length != stride.saturating_mul(height) {
            return Err(format!(
                "frame {} RAW length {length} != stride {stride} * height {height}",
                integer(frame, "index")?
            ));
        }
        let mut packed = vec![0u8; length];
        stream
            .seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek {} to {offset}: {error}", stream_path.display()))?;
        stream
            .read_exact(&mut packed)
            .map_err(|error| format!("read {} at {offset}: {error}", stream_path.display()))?;
        let raw = unpack_raw10(&packed, width, height, stride);
        let pose = DrivingAffinePose {
            center: pair(candidate, "center")?,
            major_radius: number(candidate, "major_radius")?,
            minor_radius: number(candidate, "minor_radius")?,
            angle: number(candidate, "angle")?,
        };
        let probe = score_driving_pose(&raw, width, height, pose);
        let cluster = raw_motion_octrees::FeatureClusterIrisHypothesis {
            center: pose.center,
            radius: (pose.major_radius * pose.minor_radius).sqrt(),
            major_radius: pose.major_radius,
            minor_radius: pose.minor_radius,
            angle: pose.angle,
            score: number(candidate, "score")?,
            motion_layer: integer(candidate, "motion_layer")? as usize,
            layer_coherence: number(candidate, "layer_coherence")? as f32,
            layer_separation: number(candidate, "layer_separation")? as f32,
            layer_parallax: number(candidate, "layer_parallax")? as f32,
            layer_stable_frames: integer(candidate, "layer_stable_frames")? as u16,
            seed_edge_score: number(candidate, "seed_edge_score")?,
            edge_score_gain: number(candidate, "edge_score_gain")?,
            seed_angular_coverage: 0,
            seed_opposing_meridians: 0,
            features: candidate
                .get("evidence_points")
                .and_then(Value::as_array)
                .map(|points| {
                    points
                        .iter()
                        .filter_map(|point| {
                            let pair = point.as_array()?;
                            Some((pair.first()?.as_f64()?, pair.get(1)?.as_f64()?))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            object_support: [0; raw_motion_octrees::OBJECTS],
            edge_support: integer(candidate, "edge_support")? as usize,
            angular_coverage: integer(candidate, "angular_coverage")? as usize,
            opposing_meridians: integer(candidate, "opposing_meridians")? as usize,
            bridged_current_frame_edges: candidate
                .get("bridged_current_frame_edges")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            iterations: 0,
        };
        let pupil_center = frame
            .pointer("/focus/pupil_hint")
            .and_then(Value::as_array)
            .filter(|pair| pair.len() == 2)
            .and_then(|pair| Some((pair[0].as_f64()?, pair[1].as_f64()?)));
        let orbital_probe = pupil_center.and_then(|pupil_center| {
            temporal_feature_orbital_topology_probe(&raw, width, height, &cluster, pupil_center)
        });
        probes.push(json!({
            "index": integer(frame, "index")?,
            "sequence": integer(frame, "sequence")?,
            "source_record_index": integer(frame, "source_record_index")?,
            "published": frame.pointer("/two_d_features/published").is_some_and(|value| !value.is_null()),
            "candidate": candidate,
            "material_topology_probe": driving_json(probe),
            "orbital_material_topology_probe": temporal_feature_orbital_probe_json(orbital_probe),
        }));
    }
    let output = json!({
        "schema": 1,
        "algorithm": "Rust compact material scorer plus one-lap native-resolution orbital sclera/limbus topology probe evaluated at recorded 2D temporal-feature conic geometry; diagnostic only",
        "report": report_path,
        "stream": stream_path,
        "probes": probes,
    });
    let output_file = File::create(&output_path)
        .map_err(|error| format!("create {}: {error}", output_path.display()))?;
    serde_json::to_writer_pretty(output_file, &output)
        .map_err(|error| format!("write {}: {error}", output_path.display()))?;
    println!("{}", output_path.display());
    Ok(())
}

fn labeled_raw_metadata_path(raw: &std::path::Path) -> Result<PathBuf, String> {
    let replaced = raw.with_extension("json");
    if replaced.is_file() {
        return Ok(replaced);
    }
    let appended = PathBuf::from(format!("{}.json", raw.display()));
    if appended.is_file() {
        return Ok(appended);
    }
    Err(format!(
        "no metadata sidecar at {} or {}",
        replaced.display(),
        appended.display()
    ))
}

fn labeled_limbus_points(document: &Value, visibility: &str) -> Vec<(f64, f64)> {
    document
        .get("annotation_points")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|point| {
            point.get("kind").and_then(Value::as_str) == Some("iris_edge")
                && point.get("visibility").and_then(Value::as_str) == Some(visibility)
        })
        .filter_map(|point| Some((point.get("x")?.as_f64()?, point.get("y")?.as_f64()?)))
        .collect()
}

const LABELED_ELLIPSE_DISTANCE_PHASE_SAMPLES: usize = 1024;

fn labeled_ellipse_point_at_phase(pose: DrivingAffinePose, phase: f64) -> (f64, f64) {
    let (phase_sine, phase_cosine) = phase.sin_cos();
    let (axis_sine, axis_cosine) = pose.angle.sin_cos();
    let local_x = pose.major_radius * phase_cosine;
    let local_y = pose.minor_radius * phase_sine;
    (
        pose.center.0 + axis_cosine * local_x - axis_sine * local_y,
        pose.center.1 + axis_sine * local_x + axis_cosine * local_y,
    )
}

fn labeled_point_to_ellipse_distance(point: (f64, f64), pose: DrivingAffinePose) -> f64 {
    if !pose.center.0.is_finite()
        || !pose.center.1.is_finite()
        || !pose.major_radius.is_finite()
        || !pose.minor_radius.is_finite()
        || pose.major_radius <= 0.0
        || pose.minor_radius <= 0.0
    {
        return f64::NAN;
    }

    let squared_distance = |phase: f64| {
        let ellipse = labeled_ellipse_point_at_phase(pose, phase);
        (ellipse.0 - point.0).powi(2) + (ellipse.1 - point.1).powi(2)
    };
    let step = std::f64::consts::TAU / LABELED_ELLIPSE_DISTANCE_PHASE_SAMPLES as f64;
    let samples = (0..LABELED_ELLIPSE_DISTANCE_PHASE_SAMPLES)
        .map(|index| squared_distance(index as f64 * step))
        .collect::<Vec<_>>();

    // Distance-to-ellipse is not generally radial. Refine every sampled local
    // minimum, rather than only the phase implied by the point's polar angle;
    // this remains correct for points inside an eccentric ellipse where more
    // than one stationary closest-point candidate can exist.
    let inverse_phi = (5.0f64.sqrt() - 1.0) * 0.5;
    let mut best = samples.iter().copied().fold(f64::INFINITY, f64::min);
    for index in 0..LABELED_ELLIPSE_DISTANCE_PHASE_SAMPLES {
        let previous = samples[(index + LABELED_ELLIPSE_DISTANCE_PHASE_SAMPLES - 1)
            % LABELED_ELLIPSE_DISTANCE_PHASE_SAMPLES];
        let current = samples[index];
        let next = samples[(index + 1) % LABELED_ELLIPSE_DISTANCE_PHASE_SAMPLES];
        if current > previous || current > next {
            continue;
        }
        let mut left = index as f64 * step - step;
        let mut right = index as f64 * step + step;
        let mut interior_left = right - inverse_phi * (right - left);
        let mut interior_right = left + inverse_phi * (right - left);
        let mut value_left = squared_distance(interior_left);
        let mut value_right = squared_distance(interior_right);
        for _ in 0..64 {
            if value_left <= value_right {
                right = interior_right;
                interior_right = interior_left;
                value_right = value_left;
                interior_left = right - inverse_phi * (right - left);
                value_left = squared_distance(interior_left);
            } else {
                left = interior_left;
                interior_left = interior_right;
                value_left = value_right;
                interior_right = left + inverse_phi * (right - left);
                value_right = squared_distance(interior_right);
            }
        }
        best = best.min(value_left).min(value_right);
    }
    best.sqrt()
}

/// True shortest screen-space Euclidean distances from the human points to
/// the proposed ellipse. The previous benchmark used a normalized radial
/// proxy; that proxy can substantially mis-rank tilted or eccentric conics.
fn labeled_ellipse_residuals(points: &[(f64, f64)], pose: Option<DrivingAffinePose>) -> Vec<f64> {
    let Some(pose) = pose else {
        return Vec::new();
    };
    points
        .iter()
        .filter_map(|point| {
            let residual = labeled_point_to_ellipse_distance(*point, pose);
            residual.is_finite().then_some(residual)
        })
        .collect()
}

fn labeled_reference_pose(document: &Value) -> Option<DrivingAffinePose> {
    let fit = document.get("ellipse_fit")?;
    let center = fit.get("center")?.as_array()?;
    let center = (center.first()?.as_f64()?, center.get(1)?.as_f64()?);
    let (axis_a, axis_b, mut angle) =
        if let Some(radii) = fit.get("radii").and_then(Value::as_array) {
            (
                radii.first()?.as_f64()?,
                radii.get(1)?.as_f64()?,
                fit.get("angle_degrees")?.as_f64()?.to_radians(),
            )
        } else {
            // The earlier eye-tagging labeler stored already-normalized axes and
            // radians. This is the reviewed frame Rob explicitly relabeled; it is
            // ground truth, not a legacy prediction, and must remain in the full-
            // ellipse denominator.
            (
                fit.get("major_radius")?.as_f64()?,
                fit.get("minor_radius")?.as_f64()?,
                fit.get("angle")?.as_f64()?,
            )
        };
    let (major_radius, minor_radius) = if axis_a >= axis_b {
        (axis_a, axis_b)
    } else {
        angle += std::f64::consts::FRAC_PI_2;
        (axis_b, axis_a)
    };
    let pose = DrivingAffinePose {
        center,
        major_radius,
        minor_radius,
        angle: angle.rem_euclid(std::f64::consts::PI),
    };
    // Ground truth must never be censored by the algorithm under test. In
    // particular, do not call `driving_pose_has_plausible_limbus_projection`
    // here: doing so circularly hid a reviewed full-ellipse failure from the
    // benchmark. Only reject malformed numeric annotation data.
    (pose.center.0.is_finite()
        && pose.center.1.is_finite()
        && pose.major_radius.is_finite()
        && pose.minor_radius.is_finite()
        && pose.major_radius > 0.0
        && pose.minor_radius > 0.0)
        .then_some(pose)
}

fn labeled_rms(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| {
        (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
    })
}

fn labeled_distance_summary_json(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |fraction: f64| {
        sorted[((sorted.len().saturating_sub(1) as f64 * fraction).round() as usize)
            .min(sorted.len().saturating_sub(1))]
    };
    json!({
        "samples": sorted.len(),
        "rms_px": labeled_rms(&sorted),
        "p50_px": at(0.50),
        "p95_px": at(0.95),
        "max_px": at(1.0),
        "within_2px_fraction": sorted.iter().filter(|distance| **distance <= 2.0).count() as f64 / sorted.len() as f64,
        "within_3px_fraction": sorted.iter().filter(|distance| **distance <= 3.0).count() as f64 / sorted.len() as f64,
        "within_5px_fraction": sorted.iter().filter(|distance| **distance <= 5.0).count() as f64 / sorted.len() as f64,
    })
}

fn labeled_ellipse_reference_comparison_json(
    reference: Option<DrivingAffinePose>,
    prediction: Option<DrivingAffinePose>,
) -> Value {
    let (Some(reference), Some(prediction)) = (reference, prediction) else {
        return Value::Null;
    };
    const CONTOUR_SAMPLES: usize = 720;
    let reference_points = (0..CONTOUR_SAMPLES)
        .map(|index| {
            labeled_ellipse_point_at_phase(
                reference,
                std::f64::consts::TAU * index as f64 / CONTOUR_SAMPLES as f64,
            )
        })
        .collect::<Vec<_>>();
    let prediction_points = (0..CONTOUR_SAMPLES)
        .map(|index| {
            labeled_ellipse_point_at_phase(
                prediction,
                std::f64::consts::TAU * index as f64 / CONTOUR_SAMPLES as f64,
            )
        })
        .collect::<Vec<_>>();
    let reference_to_prediction = labeled_ellipse_residuals(&reference_points, Some(prediction));
    let prediction_to_reference = labeled_ellipse_residuals(&prediction_points, Some(reference));
    let mut symmetric = reference_to_prediction.clone();
    symmetric.extend_from_slice(&prediction_to_reference);
    let reference_area_radius = (reference.major_radius * reference.minor_radius).sqrt();
    let prediction_area_radius = (prediction.major_radius * prediction.minor_radius).sqrt();
    let reference_axis_ratio = reference.minor_radius / reference.major_radius.max(1.0e-12);
    let prediction_axis_ratio = prediction.minor_radius / prediction.major_radius.max(1.0e-12);
    let raw_angle_error = (prediction.angle - reference.angle)
        .rem_euclid(std::f64::consts::PI)
        .abs();
    let angle_error = raw_angle_error.min(std::f64::consts::PI - raw_angle_error);
    json!({
        "reference_to_prediction": labeled_distance_summary_json(&reference_to_prediction),
        "prediction_to_reference": labeled_distance_summary_json(&prediction_to_reference),
        "symmetric": labeled_distance_summary_json(&symmetric),
        "center_error_px": (prediction.center.0 - reference.center.0).hypot(prediction.center.1 - reference.center.1),
        "area_equivalent_radius_error_fraction": (prediction_area_radius / reference_area_radius.max(1.0e-12) - 1.0).abs(),
        "axis_ratio_error": (prediction_axis_ratio - reference_axis_ratio).abs(),
        "orientation_error_degrees": angle_error.to_degrees(),
        "orientation_reference_eccentricity": 1.0 - reference_axis_ratio,
    })
}

#[cfg(test)]
mod labeled_ellipse_metric_tests {
    use super::*;

    fn pose(center: (f64, f64), major: f64, minor: f64, angle: f64) -> DrivingAffinePose {
        DrivingAffinePose {
            center,
            major_radius: major,
            minor_radius: minor,
            angle,
        }
    }

    #[test]
    fn true_ellipse_distance_reduces_to_exact_circle_distance() {
        let circle = pose((7.0, -3.0), 12.0, 12.0, 0.73);
        for point in [(7.0, -3.0), (7.0, 9.0), (32.0, -3.0), (1.5, 4.25)] {
            let expected = ((point.0 - 7.0f64).hypot(point.1 + 3.0) - 12.0).abs();
            let measured = labeled_point_to_ellipse_distance(point, circle);
            assert!(
                (measured - expected).abs() < 1.0e-8,
                "{point:?}: {measured} != {expected}"
            );
        }
    }

    #[test]
    fn true_ellipse_distance_is_zero_on_a_rotated_eccentric_perimeter() {
        let ellipse = pose((93.25, 44.75), 84.0, 43.0, 1.17);
        for index in 0..37 {
            let point = labeled_ellipse_point_at_phase(
                ellipse,
                std::f64::consts::TAU * index as f64 / 37.0,
            );
            assert!(
                labeled_point_to_ellipse_distance(point, ellipse) < 1.0e-8,
                "phase {index}"
            );
        }
    }

    #[test]
    fn true_ellipse_distance_is_rotation_and_translation_invariant() {
        let base = pose((0.0, 0.0), 95.0, 51.0, 0.0);
        let point = (62.0, 47.0);
        let expected = labeled_point_to_ellipse_distance(point, base);
        let angle = 0.91f64;
        let (sine, cosine) = angle.sin_cos();
        let transform = |input: (f64, f64)| {
            (
                31.0 + cosine * input.0 - sine * input.1,
                -17.0 + sine * input.0 + cosine * input.1,
            )
        };
        let transformed = pose(transform(base.center), 95.0, 51.0, angle);
        let measured = labeled_point_to_ellipse_distance(transform(point), transformed);
        assert!((measured - expected).abs() < 1.0e-8);
    }

    #[test]
    fn symmetric_reference_metric_detects_an_oversized_conic() {
        let reference = pose((100.0, 80.0), 70.0, 55.0, 0.2);
        let oversized = pose((100.0, 80.0), 91.0, 71.5, 0.2);
        let comparison =
            labeled_ellipse_reference_comparison_json(Some(reference), Some(oversized));
        assert!(
            comparison["symmetric"]["p95_px"].as_f64().unwrap() > 14.0,
            "{comparison}"
        );
        assert!(
            (comparison["area_equivalent_radius_error_fraction"]
                .as_f64()
                .unwrap()
                - 0.30)
                .abs()
                < 1.0e-9
        );
    }

    #[test]
    fn reviewed_reference_is_not_censored_by_production_plausibility() {
        let document = json!({
            "ellipse_fit": {
                "center": [120.0, 80.0],
                "radii": [100.0, 20.0],
                "angle_degrees": 90.0,
            }
        });
        let reference = labeled_reference_pose(&document)
            .expect("well-formed human geometry must remain benchmark ground truth");
        assert_eq!(reference.major_radius, 100.0);
        assert_eq!(reference.minor_radius, 20.0);
        assert!(!driving_pose_has_plausible_limbus_projection(reference));
    }

    #[test]
    fn reviewed_eye_tagging_reference_schema_is_retained() {
        let document = json!({
            "ellipse_fit": {
                "kind": "eye-tagging-robust-ellipse",
                "center": [313.5, 183.25],
                "major_radius": 61.9,
                "minor_radius": 55.8,
                "angle": -0.0035,
            }
        });
        let reference = labeled_reference_pose(&document).expect("eye-tagging ellipse schema");
        assert_eq!(reference.center, (313.5, 183.25));
        assert_eq!(reference.major_radius, 61.9);
        assert_eq!(reference.minor_radius, 55.8);
        assert!((reference.angle - (-0.0035f64).rem_euclid(std::f64::consts::PI)).abs() < 1.0e-12);
    }
}

fn labeled_pose_json(pose: Option<DrivingAffinePose>) -> Value {
    pose.map_or(Value::Null, |pose| {
        json!({
            "center": [pose.center.0, pose.center.1],
            "major_radius": pose.major_radius,
            "minor_radius": pose.minor_radius,
            "fronto_parallel_radius_px": pose.major_radius.max(pose.minor_radius),
            "area_equivalent_radius_px": (pose.major_radius * pose.minor_radius).sqrt(),
            "angle": pose.angle,
        })
    })
}

fn labeled_eyelid_scene_json(
    raw: &[u16],
    width: usize,
    height: usize,
    pose: Option<DrivingAffinePose>,
    pupil: Option<(f64, f64)>,
) -> Value {
    let Some(pose) = pose else {
        return Value::Null;
    };
    let outer = driving_boundary_from_pose(pose, 255.0);
    let scene = raw_iris_focus::discover_eyelid_scene_nautilus(raw, width, height, &outer, pupil);
    json!({
        "upper_status": scene.upper_status.label(),
        "lower_status": scene.lower_status.label(),
        "upper_margin": scene.upper_margin.iter().map(|point| json!([point.x, point.y, point.quality])).collect::<Vec<_>>(),
        "lower_margin": scene.lower_margin.iter().map(|point| json!([point.x, point.y, point.quality])).collect::<Vec<_>>(),
        "upper_fold_points": scene.upper_fold.len(),
        "lower_fold_points": scene.lower_fold.len(),
        "upper_lash_points": scene.upper_lashes.len(),
        "lower_lash_points": scene.lower_lashes.len(),
        "upper_limbus_clearance_px": scene.upper_limbus_clearance_px,
        "lower_limbus_clearance_px": scene.lower_limbus_clearance_px,
        "elapsed_us": scene.elapsed_us,
    })
}

/// One label-blind member of the experimental wide current-frame beam.  The
/// human annotations are deliberately absent: they are consulted only after
/// selection by `labeled_wide64_selection_json` below.
#[derive(Clone, Copy, Debug)]
struct LabeledWide64RoadAudit {
    index: usize,
    proposal: DrivingMultibankLimbusProposal,
    anatomy: DrivingHypothesis,
    semantic: DrivingSemanticEyeEvidence,
    rank: f64,
    full_roi_context: bool,
    anatomy_admissible: bool,
    semantic_geometry: bool,
    complete_pupil_headed_geometry: bool,
}

fn labeled_wide64_selection_json(
    selection: Option<(LabeledWide64RoadAudit, DrivingHypothesis)>,
    visible: &[(f64, f64)],
    reference_pose: Option<DrivingAffinePose>,
) -> Value {
    let Some((road, final_hypothesis)) = selection else {
        return Value::Null;
    };
    let pose = final_hypothesis.pose;
    let distances = labeled_ellipse_residuals(visible, Some(pose));
    let visible_p95 = (!distances.is_empty()).then(|| percentile(&distances, 0.95));
    let visible_max = (!distances.is_empty()).then(|| percentile(&distances, 1.0));
    let point_geometry_pass = visible_p95.is_some_and(|distance| distance <= 5.0)
        && visible_max.is_some_and(|distance| distance <= 8.0);
    let reference_comparison =
        labeled_ellipse_reference_comparison_json(reference_pose, Some(pose));
    let full_reference_geometry_pass = reference_pose.map(|_| {
        reference_comparison
            .get("symmetric")
            .and_then(|summary| summary.get("p95_px"))
            .and_then(Value::as_f64)
            .is_some_and(|distance| distance <= 8.0)
    });
    json!({
        "selected_index": road.index,
        "pose": labeled_pose_json(Some(pose)),
        "pre_refinement_pose": labeled_pose_json(Some(road.anatomy.pose)),
        "proposal_pose": labeled_pose_json(Some(road.proposal.pose)),
        "rank": road.rank,
        "full_roi_context": road.full_roi_context,
        "anatomy_admissible": road.anatomy_admissible,
        "semantic_geometry": road.semantic_geometry,
        "complete_pupil_headed_geometry": road.complete_pupil_headed_geometry,
        "visible_point_distance": labeled_distance_summary_json(&distances),
        "point_geometry_pass": point_geometry_pass,
        "full_reference_geometry_pass": full_reference_geometry_pass,
        "combined_geometry_pass": point_geometry_pass
            && full_reference_geometry_pass.unwrap_or(true),
        "reference_ellipse_comparison": reference_comparison,
    })
}

/// Score hand-labelled still RAWs with Driving's exact production proposal
/// machinery.  Publication is intentionally not simulated: a still cannot
/// establish the three-frame physical-radius/identity consensus.  This audit
/// answers the orthogonal question of whether the current-frame road geometry
/// is near the visible human limbus points before temporal admission acts.
pub(super) fn labeled_raw_eval<I>(mut args: I) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    let output_path = PathBuf::from(args.next().ok_or_else(|| {
        "usage: buttercup_wayland_raw_eyes --offline-labeled-driving-eval OUTPUT.json LABEL.json [LABEL.json ...]".to_string()
    })?);
    let label_paths = args.map(PathBuf::from).collect::<Vec<_>>();
    if label_paths.is_empty() {
        return Err("at least one label JSON is required".to_string());
    }
    if output_path.exists() {
        return Err(format!("output already exists: {}", output_path.display()));
    }
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let cwd = env::current_dir().map_err(|error| error.to_string())?;
    let workspace = if cwd.join("rust-helpers").is_dir() {
        cwd
    } else if cwd.file_name().and_then(|name| name.to_str()) == Some("rust-helpers") {
        cwd.parent().unwrap_or(&cwd).to_path_buf()
    } else {
        cwd
    };

    let mut cases = Vec::with_capacity(label_paths.len());
    let mut visible_residuals = Vec::new();
    let mut fitted = 0usize;
    let mut production_visible_distances = Vec::new();
    let mut production_guessed_distances = Vec::new();
    let mut production_generated_cases = 0usize;
    let mut production_point_geometry_pass_cases = 0usize;
    let mut production_full_reference_cases = 0usize;
    let mut production_full_reference_pass_cases = 0usize;
    let mut production_combined_geometry_pass_cases = 0usize;
    let mut production_failed_cases = Vec::new();
    let mut bounded_wide_visible_distances = Vec::new();
    let mut bounded_wide_generated_cases = 0usize;
    let mut bounded_wide_point_geometry_pass_cases = 0usize;
    let mut bounded_wide_full_reference_cases = 0usize;
    let mut bounded_wide_full_reference_pass_cases = 0usize;
    let mut bounded_wide_combined_geometry_pass_cases = 0usize;
    let mut bounded_wide_failed_cases = Vec::new();
    let mut two_d_visible_distances = Vec::new();
    let mut two_d_generated_cases = 0usize;
    let mut two_d_point_geometry_pass_cases = 0usize;
    let mut two_d_full_reference_cases = 0usize;
    let mut two_d_full_reference_pass_cases = 0usize;
    let mut two_d_combined_geometry_pass_cases = 0usize;
    let mut two_d_failed_cases = Vec::new();
    let mut multibank_wide_beam_search_cases = 0usize;
    let mut multibank_wide_beam_point_pass_cases = 0usize;
    let mut multibank_wide_beam_combined_pass_cases = 0usize;
    let mut production_missed_available_point_pass_cases = 0usize;
    let mut production_missed_available_combined_pass_cases = 0usize;
    let analog_edge_target_px = env::var("BUTTERCUP_OFFLINE_EDGE_CENTROID_TARGET_PX")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && value.abs() <= 4.0);
    for (case_index, label_path) in label_paths.iter().enumerate() {
        let label: Value = serde_json::from_slice(
            &fs::read(label_path)
                .map_err(|error| format!("read {}: {error}", label_path.display()))?,
        )
        .map_err(|error| format!("parse {}: {error}", label_path.display()))?;
        let source = label
            .get("source_raw")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{} has no source_raw", label_path.display()))?;
        let source_raw = {
            let path = PathBuf::from(source);
            if path.is_absolute() {
                path
            } else {
                workspace.join(path)
            }
        };
        let metadata_path = labeled_raw_metadata_path(&source_raw)?;
        let metadata: Value = serde_json::from_slice(
            &fs::read(&metadata_path)
                .map_err(|error| format!("read {}: {error}", metadata_path.display()))?,
        )
        .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?;
        let width = integer(&metadata, "width")? as usize;
        let height = integer(&metadata, "height")? as usize;
        let stride = integer(&metadata, "stride")? as usize;
        let sensor_origin = (
            integer(&metadata, "sensor_x")? as u32,
            integer(&metadata, "sensor_y")? as u32,
        );
        let packed = fs::read(&source_raw)
            .map_err(|error| format!("read {}: {error}", source_raw.display()))?;
        let raw = raw10::try_unpack_raw10(&packed, width, height, stride)?;
        let visible = labeled_limbus_points(&label, "visible");
        let guessed = labeled_limbus_points(&label, "guessed");
        let reference_pose = labeled_reference_pose(&label);

        let focus = raw_iris_focus::score_stream_eye(&raw, width, height);
        let upper = raw_iris_focus::detect_upper_eyelid_points(
            &raw,
            width,
            height,
            sensor_origin.0,
            sensor_origin.1,
            &focus,
        );
        let lower = raw_iris_focus::detect_lower_eyelid_points(
            &raw,
            width,
            height,
            sensor_origin.0,
            sensor_origin.1,
            &focus,
        );
        let mut native_tracker = raw_iris_focus::OuterIrisTracker::default();
        let native = raw_iris_focus::detect_outer_iris_boundary_between_eyelids_tracked_for_driving(
            &raw,
            width,
            height,
            sensor_origin.0,
            sensor_origin.1,
            &focus,
            &upper,
            &lower,
            &mut native_tracker,
        );
        let partial = focus
            .roi_truncated_limbus
            .filter(|partial| roi_truncated_limbus_recovery_ready(*partial));
        let fallback = if !native.points.is_empty() {
            Some((
                native.center,
                (native.major_radius * native.minor_radius).sqrt(),
            ))
        } else if let Some(partial) = partial {
            Some((
                partial.center,
                (partial.major_radius * partial.minor_radius).sqrt(),
            ))
        } else if focus.eye_basin_valid {
            Some((focus.center, focus.radius))
        } else {
            None
        };
        // A censored observation supplies fallback geometry but is not
        // promoted into fabricated perimeter points for this still audit.
        let driving_input = native.clone();
        let seed_pose = driving_seed_pose(&driving_input, fallback);
        let radius_prior = std::env::var("BUTTERCUP_OFFLINE_IRIS_RADIUS_PRIOR_PX")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .and_then(|radius| {
                raw_iris_focus::FrontoParallelLimbusRadiusPrior::from_fractional_support(
                    radius,
                    0.05,
                    raw_iris_focus::FrontoParallelLimbusRadiusPriorSource::FixedReference,
                )
            })
            .or_else(|| driving_cold_start_radius_prior(&driving_input, fallback, Some(&focus)));
        // Reuse one exact current-frame native-resolution Canny field for the
        // labelled 2D audit. These proposals remain geometry-only; no still
        // image is allowed to fabricate the temporal motion-layer identity
        // required by the live Clusters publisher.
        let two_d_canny_overlay = raw_motion_octrees::canny_proposal_overlay(&raw, width, height);
        let driving_material_view = DrivingRawMaterialView::new(&raw, width, height, sensor_origin);
        let native_two_d_canny = seed_pose.and_then(|pose| {
            raw_motion_octrees::current_frame_canny_ellipse_proposal(
                &two_d_canny_overlay,
                width,
                height,
                iris_seed_from_driving_pose(pose),
            )
        });
        // Preserve the seed-local production road separately from Driving's
        // recovery-bank winner.  A recovery regression can otherwise look
        // like a failure to find the limbus even when the ordinary road was
        // already geometrically correct and was replaced later.
        let ordinary = seed_pose.and_then(|pose| {
            score_driving_native_anatomy_from_working_pose(
                &raw,
                width,
                height,
                sensor_origin,
                pose,
                pose,
                None,
                1,
                radius_prior,
            )
        });
        let multibank_started = Instant::now();
        let multibank_proposals = seed_pose.map_or_else(Vec::new, |pose| {
            driving_multibank_limbus_pose_shortlist(
                &raw,
                width,
                height,
                pose,
                Some(&focus),
                radius_prior,
            )
        });
        let multibank_search_ms = multibank_started.elapsed().as_secs_f64() * 1_000.0;
        // Exercise the exact topology-validated seed handoff used by live 2D
        // temporal Canny.  The seed and its Canny refinement are reported
        // separately: a label audit must not mistake a poor native starting
        // ellipse for a regression in the measured full-ROI recovery path.
        let two_d_measured_multibank_seed = seed_pose.and_then(|pose| {
            measured_multibank_temporal_canny_seed(
                &raw,
                width,
                height,
                sensor_origin,
                iris_seed_from_driving_pose(pose),
                Some(&focus),
                radius_prior,
            )
        });
        let two_d_measured_multibank_pose = two_d_measured_multibank_seed
            .map(driving_pose_from_iris_seed)
            .flatten();
        let two_d_measured_analog_polishes = two_d_measured_multibank_pose
            .map(|pose| {
                [-1.5, -1.0, -0.5, 0.0]
                    .into_iter()
                    .filter_map(|target_offset_px| {
                        driving_multibank_analog_polish_pose(
                            &raw,
                            width,
                            height,
                            pose,
                            radius_prior,
                            target_offset_px,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let two_d_measured_multibank_canny = two_d_measured_multibank_seed.and_then(|seed| {
            raw_motion_octrees::measured_seed_canny_support_proposal(
                &two_d_canny_overlay,
                width,
                height,
                seed,
            )
        });
        let two_d_measured_multibank_canny_pose =
            two_d_measured_multibank_canny
                .as_ref()
                .map(|candidate| DrivingAffinePose {
                    center: candidate.center,
                    major_radius: candidate.major_radius,
                    minor_radius: candidate.minor_radius,
                    angle: candidate.angle,
                });
        // Audit the native-pixel center-only closure independently of the
        // ordinary exact-Canny result. Production invokes this bounded basin
        // only when the exact measurement is absent or weak; keeping the
        // diagnostic separate makes the label report expose whether closure
        // improves geometry without letting labels participate in selection.
        let two_d_measured_center_closure = two_d_measured_multibank_seed.and_then(|seed| {
            temporal_canny_measured_center_geometry_candidates(
                &raw,
                width,
                height,
                sensor_origin,
                &two_d_canny_overlay,
                seed,
                radius_prior,
            )
            .into_iter()
            .next()
            .map(|(_, proposal)| proposal)
        });
        let two_d_measured_center_closure_pose =
            two_d_measured_center_closure
                .as_ref()
                .map(|candidate| DrivingAffinePose {
                    center: candidate.center,
                    major_radius: candidate.major_radius,
                    minor_radius: candidate.minor_radius,
                    angle: candidate.angle,
                });
        let two_d_measured_selected_canny = match (
            two_d_measured_multibank_canny.as_ref(),
            two_d_measured_center_closure.as_ref(),
        ) {
            (Some(base), Some(closure))
                if temporal_canny_center_closure_is_preferred(
                    two_d_measured_multibank_seed
                        .expect("selected Canny has a measured seed")
                        .center,
                    base.confidence,
                    base.opposing_meridians,
                    closure.center,
                    closure.confidence,
                    closure.opposing_meridians,
                ) =>
            {
                Some(closure.clone())
            }
            (Some(base), _) => Some(base.clone()),
            (None, _) => None,
        }
        .map(|mut selected| {
            let initial_pose = DrivingAffinePose {
                center: selected.center,
                major_radius: selected.major_radius,
                minor_radius: selected.minor_radius,
                angle: selected.angle,
            };
            if let Some((pose, iterations)) = temporal_canny_decisive_analog_polished_pose(
                &raw,
                width,
                height,
                sensor_origin,
                radius_prior,
                initial_pose,
            ) {
                selected.center = pose.center;
                selected.major_radius = pose.major_radius;
                selected.minor_radius = pose.minor_radius;
                selected.angle = pose.angle;
                selected.iterations = selected.iterations.saturating_add(iterations);
            }
            selected
        });
        let two_d_measured_selected_canny_pose =
            two_d_measured_selected_canny
                .as_ref()
                .map(|candidate| DrivingAffinePose {
                    center: candidate.center,
                    major_radius: candidate.major_radius,
                    minor_radius: candidate.minor_radius,
                    angle: candidate.angle,
                });
        // Keep a wider geometry-only beam in the labelled audit. Production
        // consumes four proposals; the wider report lets leave-one-case-out
        // tuning prove that an apparent ranking improvement did not merely
        // remove the correct conic from another labelled frame's beam.
        let wide_shortlists =
            seed_pose.map_or_else(DrivingMultibankLimbusShortlists::default, |pose| {
                driving_multibank_limbus_pose_shortlists_with_beam(
                    &raw,
                    width,
                    height,
                    pose,
                    Some(&focus),
                    radius_prior,
                    64,
                )
            });
        if wide_shortlists.direct_top_four != multibank_proposals {
            return Err(format!(
                "case-{:02}: wide proposal bank did not preserve the exact legacy four-candidate incumbent",
                case_index + 1,
            ));
        }
        let multibank_partial_audit = wide_shortlists.selected;
        // A candidate-independent semantic scene is anchored once at the
        // measured seed.  The partial ranker may censor sectors with this
        // scene, but no candidate is allowed to discover a bespoke lid and
        // then use that circular result to promote itself.
        let seed_semantic_pose = seed_pose.map(|pose| {
            let mut semantic_pose = pose;
            if let Some(prior) = radius_prior {
                let ratio = (pose.minor_radius / pose.major_radius.max(1.0)).clamp(0.48, 1.0);
                semantic_pose.major_radius = prior.estimate_px;
                semantic_pose.minor_radius = prior.estimate_px * ratio;
            }
            semantic_pose
        });
        let seed_eyelid_scene = seed_semantic_pose.map(|semantic_pose| {
            raw_iris_focus::discover_eyelid_scene_nautilus(
                &raw,
                width,
                height,
                &driving_boundary_from_pose(semantic_pose, 255.0),
                focus.pupil_hint,
            )
        });
        let partial_ranked = seed_pose.map_or_else(Vec::new, |pose| {
            driving_rank_partial_limbus_proposals(
                &raw,
                width,
                height,
                sensor_origin,
                pose,
                focus.pupil_hint,
                seed_eyelid_scene
                    .as_ref()
                    .map(|scene| (scene, sensor_origin)),
                &multibank_partial_audit,
            )
        });
        let bounded_wide_started = Instant::now();
        let bounded_wide_recovery = seed_pose.and_then(|pose| {
            driving_bounded_wide_recovery_from_proposals(
                &raw,
                width,
                height,
                sensor_origin,
                pose,
                Some(&focus),
                radius_prior,
                &multibank_partial_audit,
            )
        });
        let bounded_wide_elapsed_ms = bounded_wide_started.elapsed().as_secs_f64() * 1_000.0;
        // Actually exercise the 64-member beam rather than using labels only
        // to ask whether a good proposal happened to exist.  Every proposal
        // pays for the same complete native-RAW pupil/lid/anatomy road and is
        // ranked with the production outer-geometry score.  Labels remain
        // entirely downstream and are used only when serializing accuracy.
        // This exhaustive pass is an offline experiment; a live version must
        // retain the wide cheap proposal stage but bound the expensive final
        // anatomy beam after this report establishes a safe label-blind rank.
        let wide64_started = Instant::now();
        let wide64_roads = multibank_partial_audit
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, proposal)| {
                let full_roi_context = driving_multibank_proposal_has_full_roi_context(
                    &raw,
                    width,
                    height,
                    proposal,
                    focus.pupil_hint,
                );
                let anatomy = score_driving_native_anatomy_from_working_pose(
                    &raw,
                    width,
                    height,
                    sensor_origin,
                    proposal.pose,
                    proposal.pose,
                    None,
                    2,
                    radius_prior,
                )?;
                let semantic =
                    driving_semantic_eye_evidence(&raw, width, height, sensor_origin, anatomy)?;
                let semantic_geometry = semantic.plausible_lids >= 1
                    && semantic.annular.samples >= 64
                    && semantic.annular.active_sectors >= 14
                    && semantic.annular.upper_quartile_gradient >= 0.030
                    && semantic.annular.global_orientation_anisotropy <= 0.82
                    && semantic.score >= 0.66
                    && !semantic.straight_band.veto;
                let complete_pupil_headed_geometry = anatomy.score >= 0.44
                    && anatomy.limbus_score >= 0.24
                    && anatomy.white_score >= 0.24
                    && anatomy.far_sclera_score >= 0.34
                    && anatomy.pupil_score >= 0.48
                    && anatomy.pupil_margin >= 0.38
                    && anatomy.through_eye_score >= 0.44
                    && anatomy.bilateral_limbus_order >= 0.40
                    && anatomy.pupil_projected_area_radius_px.is_some()
                    && anatomy
                        .pupil_boundary_canonical()
                        .0
                        .hypot(anatomy.pupil_boundary_canonical().1)
                        <= 0.78;
                Some(LabeledWide64RoadAudit {
                    index,
                    proposal,
                    anatomy,
                    semantic,
                    rank: temporal_canny_outer_geometry_rank(anatomy, semantic, proposal),
                    full_roi_context,
                    anatomy_admissible: driving_hypothesis_admissible(anatomy),
                    semantic_geometry,
                    complete_pupil_headed_geometry,
                })
            })
            .collect::<Vec<_>>();
        let by_rank = |left: &&LabeledWide64RoadAudit, right: &&LabeledWide64RoadAudit| {
            left.rank.total_cmp(&right.rank)
        };
        let wide64_rank_only_road = wide64_roads.iter().max_by(by_rank).copied();
        let wide64_admissible_road = wide64_roads
            .iter()
            .filter(|road| road.full_roi_context && road.anatomy_admissible)
            .max_by(by_rank)
            .copied();
        let wide64_semantic_road = wide64_roads
            .iter()
            .filter(|road| {
                road.full_roi_context
                    && road.semantic_geometry
                    && road.complete_pupil_headed_geometry
            })
            .max_by(by_rank)
            .copied();
        let finish_wide64 = |road: Option<LabeledWide64RoadAudit>| {
            let road = road?;
            // The complete two-lap road is itself the measured geometry. Do
            // not snap it back to the integer proposal lattice after ranking:
            // that outer-refinement handoff is useful when only one proposal
            // was completed, but would discard the sub-pixel closure which
            // just distinguished members of this exhaustive beam.
            let candidate = road.anatomy;
            let mut tracker = DrivingSegmentationTracker::default();
            tracker
                .constrain_pupil_margin_to_active_trajectory(
                    &raw,
                    width,
                    height,
                    sensor_origin,
                    candidate,
                    Some(&focus),
                )
                .map(|candidate| (road, candidate))
        };
        let wide64_rank_only_candidate = finish_wide64(wide64_rank_only_road);
        let wide64_admissible_candidate = finish_wide64(wide64_admissible_road);
        let wide64_semantic_candidate = finish_wide64(wide64_semantic_road);
        let wide64_elapsed_ms = wide64_started.elapsed().as_secs_f64() * 1_000.0;
        let multibank_roads = seed_pose.map_or_else(Vec::new, |original_pose| {
            multibank_proposals
                .iter()
                .copied()
                .take(4)
                .map(|proposal| {
                    let road = score_driving_native_anatomy_from_working_pose(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        original_pose,
                        proposal.pose,
                        None,
                        2,
                        radius_prior,
                    );
                    (proposal, road)
                })
                .collect::<Vec<_>>()
        });
        let started = Instant::now();
        let mut audited_roads = Vec::new();
        let candidate = seed_pose.and_then(|pose| {
            score_driving_native_anatomy_with_audit(
                &raw,
                width,
                height,
                sensor_origin,
                pose,
                Some(&focus),
                radius_prior,
                Some(&mut audited_roads),
            )
        });
        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
        let multibank_refined_candidate = candidate
            .zip(multibank_proposals.first().copied())
            .and_then(|(candidate, proposal)| {
                driving_apply_multibank_outer_refinement(
                    &raw,
                    width,
                    height,
                    sensor_origin,
                    candidate,
                    proposal,
                    Some(&focus),
                    false,
                    false,
                )
            });
        // Mirror the production frame-local branch explicitly.  The older
        // `driving_candidate` diagnostic above intentionally begins at the
        // native seed so recovery failures remain visible, but live Driving
        // begins a fresh anatomy road at a strong multibank limbus and then
        // performs the fixed-geometry pupil-margin/prior pass.  Reporting the
        // two under one name made a good measured limbus look regressed.
        let production_multibank = multibank_proposals
            .first()
            .copied()
            .filter(|proposal| {
                driving_multibank_proposal_has_full_roi_context(
                    &raw,
                    width,
                    height,
                    *proposal,
                    focus.pupil_hint,
                )
            })
            .map(|proposal| {
                driving_analog_polish_multibank_proposal(
                    &raw,
                    width,
                    height,
                    proposal,
                    radius_prior,
                )
            });
        let production_censored_strong = production_multibank.is_none()
            && multibank_proposals.first().is_some_and(|proposal| {
                driving_multibank_proposal_is_strong_censored_geometry(*proposal, width, height)
            });
        let mut production_frame_local_tracker = DrivingSegmentationTracker::default();
        let mut production_independent_outer_selected = false;
        let production_frame_local_unconstrained_candidate = (!production_censored_strong)
            .then_some(())
            .and(seed_pose)
            .and_then(|frame_local_seed| {
                let measured = production_multibank.and_then(|proposal| {
                    score_driving_native_anatomy_from_working_pose(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        proposal.pose,
                        proposal.pose,
                        None,
                        2,
                        radius_prior,
                    )
                });
                let independently_verified = production_multibank.is_some_and(|proposal| {
                    let center_departure = (proposal.pose.center.0 - frame_local_seed.center.0)
                        .hypot(proposal.pose.center.1 - frame_local_seed.center.1)
                        / proposal.pose.major_radius.max(1.0);
                    let scale_departure = (proposal.pose.major_radius
                        / frame_local_seed.major_radius.max(1.0))
                    .ln()
                    .abs();
                    center_departure >= 0.10 || scale_departure >= 0.08
                });
                let (independent, seed_local_owns_outer) =
                    if measured.is_none() || independently_verified {
                        // Match live Driving: first ask whether one bounded
                        // seed-local lap already re-proves a complete outer
                        // road. Continuing to lap from such a result can walk
                        // inward onto a pupil/lid edge. Only when that strict
                        // self-support gate fails do we pay for the broader
                        // native recovery used previously.
                        let seed_local = score_driving_native_anatomy_from_working_pose(
                            &raw,
                            width,
                            height,
                            sensor_origin,
                            frame_local_seed,
                            frame_local_seed,
                            None,
                            1,
                            radius_prior,
                        )
                        .filter(|hypothesis| {
                            driving_seed_local_native_outer_road_self_supporting(
                                &raw,
                                width,
                                height,
                                sensor_origin,
                                *hypothesis,
                            )
                        });
                        if let Some(seed_local) = seed_local {
                            (Some(seed_local), true)
                        } else {
                            (
                                score_driving_native_anatomy(
                                    &raw,
                                    width,
                                    height,
                                    sensor_origin,
                                    frame_local_seed,
                                    Some(&focus),
                                    radius_prior,
                                ),
                                false,
                            )
                        }
                    } else {
                        (None, false)
                    };
                if measured.is_none() && seed_local_owns_outer {
                    production_independent_outer_selected = true;
                }
                if let (Some(measured), Some(independent), Some(proposal)) =
                    (measured, independent, production_multibank)
                {
                    if driving_independent_native_outer_road_preferred(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        independent,
                        measured,
                        proposal,
                    ) {
                        production_independent_outer_selected = true;
                        return Some(independent);
                    }
                }
                measured.or(independent)
            })
            .and_then(|candidate| {
                if let Some(proposal) =
                    production_multibank.filter(|_| !production_independent_outer_selected)
                {
                    driving_apply_multibank_outer_refinement(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        candidate,
                        proposal,
                        Some(&focus),
                        false,
                        false,
                    )
                } else {
                    Some(candidate)
                }
            });
        let production_frame_local_candidate = production_frame_local_unconstrained_candidate
            .and_then(|candidate| {
                production_frame_local_tracker.constrain_pupil_margin_to_active_trajectory(
                    &raw,
                    width,
                    height,
                    sensor_origin,
                    candidate,
                    Some(&focus),
                )
            });
        // Put the established current-frame result and every wide-beam
        // finalist onto exactly the same label-blind score scale.  Human
        // points remain entirely downstream of this arbitration.
        let production_frame_local_outer_audit =
            production_frame_local_candidate.and_then(|candidate| {
                driving_outer_geometry_audit(&raw, width, height, sensor_origin, candidate)
            });
        let wide64_admissible_outer_audit = wide64_admissible_candidate
            .map(|(_, candidate)| candidate)
            .and_then(|candidate| {
                driving_outer_geometry_audit(&raw, width, height, sensor_origin, candidate)
            });
        let wide64_semantic_outer_audit = wide64_semantic_candidate
            .map(|(_, candidate)| candidate)
            .and_then(|candidate| {
                driving_outer_geometry_audit(&raw, width, height, sensor_origin, candidate)
            });
        let bounded_wide_selected = bounded_wide_recovery.is_some_and(|recovery| {
            driving_wide_recovery_should_replace(
                production_frame_local_unconstrained_candidate.and_then(|candidate| {
                    driving_outer_geometry_audit(&raw, width, height, sensor_origin, candidate)
                }),
                recovery,
            )
        });
        let bounded_wide_unconstrained_candidate = if bounded_wide_selected {
            bounded_wide_recovery.map(|recovery| recovery.hypothesis)
        } else {
            production_frame_local_unconstrained_candidate
        };
        let mut bounded_wide_tracker = DrivingSegmentationTracker::default();
        let bounded_wide_candidate = bounded_wide_unconstrained_candidate.and_then(|candidate| {
            bounded_wide_tracker.constrain_pupil_margin_to_active_trajectory(
                &raw,
                width,
                height,
                sensor_origin,
                candidate,
                Some(&focus),
            )
        });
        let candidate_pose = candidate.map(|candidate| candidate.pose);
        let ordinary_pose = ordinary.map(|candidate| candidate.pose);
        fitted += usize::from(candidate_pose.is_some());
        let case_visible_residuals = labeled_ellipse_residuals(&visible, candidate_pose);
        let case_guessed_residuals = labeled_ellipse_residuals(&guessed, candidate_pose);
        visible_residuals.extend_from_slice(&case_visible_residuals);
        let production_pose = production_frame_local_candidate.map(|candidate| candidate.pose);
        let production_case_visible_distances =
            labeled_ellipse_residuals(&visible, production_pose);
        let production_case_guessed_distances =
            labeled_ellipse_residuals(&guessed, production_pose);
        let production_point_geometry_pass = production_pose.is_some()
            && !production_case_visible_distances.is_empty()
            && percentile(&production_case_visible_distances, 0.95) <= 5.0
            && percentile(&production_case_visible_distances, 1.0) <= 8.0;
        let production_reference_comparison =
            labeled_ellipse_reference_comparison_json(reference_pose, production_pose);
        let production_full_reference_pass = reference_pose.is_some()
            && production_reference_comparison
                .get("symmetric")
                .and_then(|summary| summary.get("p95_px"))
                .and_then(Value::as_f64)
                .is_some_and(|distance| distance <= 8.0);
        let production_combined_geometry_pass = production_point_geometry_pass
            && (reference_pose.is_none() || production_full_reference_pass);
        production_generated_cases += usize::from(production_pose.is_some());
        production_point_geometry_pass_cases += usize::from(production_point_geometry_pass);
        production_full_reference_cases += usize::from(reference_pose.is_some());
        production_full_reference_pass_cases += usize::from(production_full_reference_pass);
        production_combined_geometry_pass_cases += usize::from(production_combined_geometry_pass);
        production_visible_distances.extend_from_slice(&production_case_visible_distances);
        production_guessed_distances.extend_from_slice(&production_case_guessed_distances);
        if !production_point_geometry_pass
            || (reference_pose.is_some() && !production_full_reference_pass)
        {
            production_failed_cases.push(format!("case-{:02}", case_index + 1));
        }

        let bounded_wide_pose = bounded_wide_candidate.map(|candidate| candidate.pose);
        let bounded_wide_case_visible_distances =
            labeled_ellipse_residuals(&visible, bounded_wide_pose);
        let bounded_wide_point_geometry_pass = bounded_wide_pose.is_some()
            && !bounded_wide_case_visible_distances.is_empty()
            && percentile(&bounded_wide_case_visible_distances, 0.95) <= 5.0
            && percentile(&bounded_wide_case_visible_distances, 1.0) <= 8.0;
        let bounded_wide_reference_comparison =
            labeled_ellipse_reference_comparison_json(reference_pose, bounded_wide_pose);
        let bounded_wide_full_reference_pass = reference_pose.is_some()
            && bounded_wide_reference_comparison
                .get("symmetric")
                .and_then(|summary| summary.get("p95_px"))
                .and_then(Value::as_f64)
                .is_some_and(|distance| distance <= 8.0);
        let bounded_wide_combined_geometry_pass = bounded_wide_point_geometry_pass
            && (reference_pose.is_none() || bounded_wide_full_reference_pass);
        bounded_wide_generated_cases += usize::from(bounded_wide_pose.is_some());
        bounded_wide_point_geometry_pass_cases += usize::from(bounded_wide_point_geometry_pass);
        bounded_wide_full_reference_cases += usize::from(reference_pose.is_some());
        bounded_wide_full_reference_pass_cases += usize::from(bounded_wide_full_reference_pass);
        bounded_wide_combined_geometry_pass_cases +=
            usize::from(bounded_wide_combined_geometry_pass);
        bounded_wide_visible_distances.extend_from_slice(&bounded_wide_case_visible_distances);
        if !bounded_wide_point_geometry_pass
            || (reference_pose.is_some() && !bounded_wide_full_reference_pass)
        {
            bounded_wide_failed_cases.push(format!("case-{:02}", case_index + 1));
        }

        let two_d_case_visible_distances =
            labeled_ellipse_residuals(&visible, two_d_measured_selected_canny_pose);
        let two_d_point_geometry_pass = two_d_measured_selected_canny_pose.is_some()
            && !two_d_case_visible_distances.is_empty()
            && percentile(&two_d_case_visible_distances, 0.95) <= 5.0
            && percentile(&two_d_case_visible_distances, 1.0) <= 8.0;
        let two_d_reference_comparison = labeled_ellipse_reference_comparison_json(
            reference_pose,
            two_d_measured_selected_canny_pose,
        );
        let two_d_full_reference_pass = reference_pose.is_some()
            && two_d_reference_comparison
                .get("symmetric")
                .and_then(|summary| summary.get("p95_px"))
                .and_then(Value::as_f64)
                .is_some_and(|distance| distance <= 8.0);
        let two_d_combined_geometry_pass =
            two_d_point_geometry_pass && (reference_pose.is_none() || two_d_full_reference_pass);
        two_d_generated_cases += usize::from(two_d_measured_selected_canny_pose.is_some());
        two_d_point_geometry_pass_cases += usize::from(two_d_point_geometry_pass);
        two_d_full_reference_cases += usize::from(reference_pose.is_some());
        two_d_full_reference_pass_cases += usize::from(two_d_full_reference_pass);
        two_d_combined_geometry_pass_cases += usize::from(two_d_combined_geometry_pass);
        two_d_visible_distances.extend_from_slice(&two_d_case_visible_distances);
        if !two_d_point_geometry_pass || (reference_pose.is_some() && !two_d_full_reference_pass) {
            two_d_failed_cases.push(format!("case-{:02}", case_index + 1));
        }
        // Labels are used here only as a post-hoc search-coverage oracle. The
        // winning index is never fed back into production selection. This
        // separates "the candidate bank never found the limbus" from "the
        // evidence ranker preferred the wrong candidate" without hiding
        // either failure behind the other's aggregate.
        let multibank_wide_beam_point_oracle = multibank_partial_audit
            .iter()
            .enumerate()
            .filter_map(|(index, proposal)| {
                let distances = labeled_ellipse_residuals(&visible, Some(proposal.pose));
                (!distances.is_empty()).then(|| {
                    (
                        index,
                        proposal.pose,
                        distances.clone(),
                        percentile(&distances, 0.95),
                        percentile(&distances, 1.0),
                    )
                })
            })
            .min_by(|left, right| {
                left.3
                    .total_cmp(&right.3)
                    .then_with(|| left.4.total_cmp(&right.4))
            });
        let multibank_wide_beam_point_pass = multibank_wide_beam_point_oracle
            .as_ref()
            .is_some_and(|oracle| oracle.3 <= 5.0 && oracle.4 <= 8.0);
        let multibank_wide_beam_combined_oracle = multibank_partial_audit
            .iter()
            .enumerate()
            .filter_map(|(index, proposal)| {
                let distances = labeled_ellipse_residuals(&visible, Some(proposal.pose));
                if distances.is_empty() {
                    return None;
                }
                let visible_p95 = percentile(&distances, 0.95);
                let visible_max = percentile(&distances, 1.0);
                let reference_comparison =
                    labeled_ellipse_reference_comparison_json(reference_pose, Some(proposal.pose));
                let reference_p95 = reference_pose.and_then(|_| {
                    reference_comparison
                        .get("symmetric")
                        .and_then(|summary| summary.get("p95_px"))
                        .and_then(Value::as_f64)
                });
                let point_pass = visible_p95 <= 5.0 && visible_max <= 8.0;
                let full_reference_pass = reference_p95.map_or(true, |distance| distance <= 8.0);
                let combined_pass = point_pass && full_reference_pass;
                Some((
                    index,
                    proposal.pose,
                    distances,
                    visible_p95,
                    visible_max,
                    reference_comparison,
                    reference_p95,
                    combined_pass,
                ))
            })
            .min_by(|left, right| {
                right
                    .7
                    .cmp(&left.7)
                    .then_with(|| left.6.unwrap_or(0.0).total_cmp(&right.6.unwrap_or(0.0)))
                    .then_with(|| left.3.total_cmp(&right.3))
                    .then_with(|| left.4.total_cmp(&right.4))
            });
        let multibank_wide_beam_combined_pass = multibank_wide_beam_combined_oracle
            .as_ref()
            .is_some_and(|oracle| oracle.7);
        multibank_wide_beam_search_cases += usize::from(multibank_wide_beam_point_oracle.is_some());
        multibank_wide_beam_point_pass_cases += usize::from(multibank_wide_beam_point_pass);
        multibank_wide_beam_combined_pass_cases += usize::from(multibank_wide_beam_combined_pass);
        production_missed_available_point_pass_cases +=
            usize::from(multibank_wide_beam_point_pass && !production_point_geometry_pass);
        production_missed_available_combined_pass_cases +=
            usize::from(multibank_wide_beam_combined_pass && !production_combined_geometry_pass);
        let native_pose = (!native.points.is_empty()).then_some(DrivingAffinePose {
            center: native.center,
            major_radius: native.major_radius.max(native.minor_radius),
            minor_radius: native.major_radius.min(native.minor_radius),
            angle: if native.major_radius >= native.minor_radius {
                native.angle
            } else {
                native.angle + std::f64::consts::FRAC_PI_2
            },
        });
        cases.push(json!({
            "case": format!("case-{:02}", case_index + 1),
            "label": label_path,
            "source_raw": source_raw,
            "metadata": metadata_path,
            "width": width,
            "height": height,
            "stride": stride,
            "sensor_origin": [sensor_origin.0, sensor_origin.1],
            "visible_points": visible.iter().map(|point| [point.0, point.1]).collect::<Vec<_>>(),
            "guessed_points": guessed.iter().map(|point| [point.0, point.1]).collect::<Vec<_>>(),
            "label_reference": {
                "annotation_scope": if reference_pose.is_some() { "fitted_full_ellipse" } else { "visible_arc_points_only" },
                "pose": labeled_pose_json(reference_pose),
                "visible_point_fit_baseline": labeled_distance_summary_json(
                    &labeled_ellipse_residuals(&visible, reference_pose),
                ),
                "guessed_point_fit_baseline": labeled_distance_summary_json(
                    &labeled_ellipse_residuals(&guessed, reference_pose),
                ),
                "eyelid_scene": labeled_eyelid_scene_json(
                    &raw,
                    width,
                    height,
                    reference_pose,
                    focus.pupil_hint,
                ),
                "multibank_limbus": reference_pose.and_then(|pose| {
                    driving_multibank_limbus_evidence(&raw, width, height, pose)
                }).map(|evidence| json!({
                    "score": evidence.score,
                    "selection_score": DrivingMultibankLimbusProposal {
                        pose: reference_pose.unwrap(),
                        evidence,
                    }.selection_score(),
                    "left_quantile": evidence.left_quantile,
                    "right_quantile": evidence.right_quantile,
                    "left_mean_support": evidence.left_mean_support,
                    "right_mean_support": evidence.right_mean_support,
                    "left_supported_fraction": evidence.left_supported_fraction,
                    "right_supported_fraction": evidence.right_supported_fraction,
                    "weakest_lateral_quantile": evidence.weakest_lateral_quantile,
                    "mean_lateral_support": evidence.mean_lateral_support,
                    "supported_fraction": evidence.supported_fraction,
                    "narrow_mean": evidence.narrow_mean,
                    "medium_mean": evidence.medium_mean,
                    "broad_mean": evidence.broad_mean,
                    "edge_centroid_signed_mean_px": evidence.edge_centroid_signed_mean_px,
                    "edge_centroid_absolute_mean_px": evidence.edge_centroid_absolute_mean_px,
                    "edge_centroid_coherence": evidence.edge_centroid_coherence,
                    "edge_centroid_samples": evidence.edge_centroid_samples,
                    "far_sclera_step_mean": evidence.far_sclera_step_mean,
                    "outside_plateau_mean": evidence.outside_plateau_mean,
                    "outside_secondary_edge_mean": evidence.outside_secondary_edge_mean,
                    "far_lane_samples": evidence.far_lane_samples,
                    "upper_vertical_mean_support": evidence.upper_vertical_mean_support,
                    "lower_vertical_mean_support": evidence.lower_vertical_mean_support,
                    "upper_vertical_quantile": evidence.upper_vertical_quantile,
                    "lower_vertical_quantile": evidence.lower_vertical_quantile,
                    "upper_vertical_supported_fraction": evidence.upper_vertical_supported_fraction,
                    "lower_vertical_supported_fraction": evidence.lower_vertical_supported_fraction,
                    "vertical_samples": evidence.vertical_samples,
                    "inside_texture": evidence.inside_texture,
                    "outside_texture": evidence.outside_texture,
                    "samples": evidence.samples,
                    "left_samples": evidence.left_samples,
                    "right_samples": evidence.right_samples,
                    "bilateral": evidence.bilateral,
                })),
            },
            "focus": {
                "eye_basin_valid": focus.eye_basin_valid,
                "center": [focus.center.0, focus.center.1],
                "radius": focus.radius,
                "axis_ratio": focus.axis_ratio,
                "axis_angle": focus.axis_angle,
                "pupil_hint": focus.pupil_hint.map(|point| [point.0, point.1]),
                "pupil_hint_score": focus.pupil_hint_score,
                "partial_frame": partial.is_some(),
            },
            "upper_eyelid": upper.iter().map(|point| [point.x as f64, point.y as f64, point.quality]).collect::<Vec<_>>(),
            "lower_eyelid": lower.iter().map(|point| [point.x as f64, point.y as f64, point.quality]).collect::<Vec<_>>(),
            "native_seed": labeled_pose_json(native_pose),
            "two_d_current_canny_from_native_seed": native_two_d_canny.as_ref().map(|proposal| {
                let pose = DrivingAffinePose {
                    center: proposal.center,
                    major_radius: proposal.major_radius,
                    minor_radius: proposal.minor_radius,
                    angle: proposal.angle,
                };
                let residuals = labeled_ellipse_residuals(&visible, Some(pose));
                json!({
                    "pose": labeled_pose_json(Some(pose)),
                    "confidence": proposal.confidence,
                    "seed_confidence": proposal.seed_confidence,
                    "edge_support": proposal.edge_support,
                    "angular_coverage": proposal.angular_coverage,
                    "opposing_meridians": proposal.opposing_meridians,
                    "visible_label_rms_px": labeled_rms(&residuals),
                })
            }),
            "two_d_measured_multibank_seed": two_d_measured_multibank_pose.map(|pose| {
                let residuals = labeled_ellipse_residuals(&visible, Some(pose));
                json!({
                    "pose": labeled_pose_json(Some(pose)),
                    "visible_label_rms_px": labeled_rms(&residuals),
                    "canny": two_d_measured_multibank_canny.as_ref().map(|candidate| {
                        let residuals = labeled_ellipse_residuals(
                            &visible,
                            two_d_measured_multibank_canny_pose,
                        );
                        json!({
                            "pose": labeled_pose_json(two_d_measured_multibank_canny_pose),
                            "confidence": candidate.confidence,
                            "edge_support": candidate.edge_support,
                            "angular_coverage": candidate.angular_coverage,
                            "opposing_meridians": candidate.opposing_meridians,
                            "visible_label_rms_px": labeled_rms(&residuals),
                        })
                    }),
                })
            }),
            "two_d_measured_center_closure": two_d_measured_center_closure.as_ref().map(|candidate| {
                let residuals = labeled_ellipse_residuals(
                    &visible,
                    two_d_measured_center_closure_pose,
                );
                json!({
                    "pose": labeled_pose_json(two_d_measured_center_closure_pose),
                    "confidence": candidate.confidence,
                    "edge_support": candidate.edge_support,
                    "angular_coverage": candidate.angular_coverage,
                    "opposing_meridians": candidate.opposing_meridians,
                    "visible_label_rms_px": labeled_rms(&residuals),
                })
            }),
            "two_d_measured_selected_canny": two_d_measured_selected_canny.as_ref().map(|candidate| {
                let residuals = labeled_ellipse_residuals(
                    &visible,
                    two_d_measured_selected_canny_pose,
                );
                json!({
                    "pose": labeled_pose_json(two_d_measured_selected_canny_pose),
                    "confidence": candidate.confidence,
                    "edge_support": candidate.edge_support,
                    "angular_coverage": candidate.angular_coverage,
                    "opposing_meridians": candidate.opposing_meridians,
                    "visible_label_rms_px": labeled_rms(&residuals),
                    "visible_point_distance": labeled_distance_summary_json(&residuals),
                    "point_geometry_pass": two_d_point_geometry_pass,
                    "full_reference_geometry_pass": if reference_pose.is_some() {
                        Some(two_d_full_reference_pass)
                    } else {
                        None
                    },
                    "combined_geometry_pass": two_d_combined_geometry_pass,
                    "reference_ellipse_comparison": two_d_reference_comparison,
                })
            }),
            "two_d_measured_analog_polishes": two_d_measured_analog_polishes.iter().map(|polish| {
                let residuals = labeled_ellipse_residuals(&visible, Some(polish.pose));
                json!({
                    "pose": labeled_pose_json(Some(polish.pose)),
                    "iterations": polish.iterations,
                    "samples": polish.samples,
                    "initial_offset_rms_px": polish.initial_offset_rms_px,
                    "final_offset_rms_px": polish.final_offset_rms_px,
                    "target_offset_px": polish.target_offset_px,
                    "visible_label_rms_px": labeled_rms(&residuals),
                })
            }).collect::<Vec<_>>(),
            "ordinary_candidate": ordinary.map(|candidate| {
                let residuals = labeled_ellipse_residuals(&visible, ordinary_pose);
                let outer_multibank = driving_multibank_limbus_evidence(
                    &raw,
                    width,
                    height,
                    candidate.pose,
                );
                let semantic_eye = driving_semantic_eye_evidence(
                    &raw,
                    width,
                    height,
                    sensor_origin,
                    candidate,
                );
                json!({
                    "pose": labeled_pose_json(Some(candidate.pose)),
                    "score": candidate.score,
                    "white_score": candidate.white_score,
                    "far_sclera_score": candidate.far_sclera_score,
                    "limbus_score": candidate.limbus_score,
                    "pupil_score": candidate.pupil_score,
                    "pupil_enclosure": candidate.pupil_enclosure,
                    "pupil_margin": candidate.pupil_margin,
                    "pupil_horizon": candidate.pupil_horizon,
                    "through_eye_score": candidate.through_eye_score,
                    "bilateral_limbus_order": candidate.bilateral_limbus_order,
                    "light_cohesion": candidate.light_cohesion,
                    "affine_departure_fraction": candidate.affine_departure_fraction,
                    "affine_departure_ratio": candidate.affine_departure_ratio,
                    "pupil_boundary_center": candidate.pupil_boundary_center(),
                    "pupil_projected_area_radius_px": candidate.pupil_projected_area_radius_px,
                    "pupil_to_limbus_radius_ratio": driving_pupil_to_limbus_radius_ratio(candidate),
                    "outer_multibank": outer_multibank.map(|evidence| json!({
                        "score": evidence.score,
                        "selection_score": DrivingMultibankLimbusProposal {
                            pose: candidate.pose,
                            evidence,
                        }.selection_score(),
                        "weakest_lateral_quantile": evidence.weakest_lateral_quantile,
                        "supported_fraction": evidence.supported_fraction,
                        "samples": evidence.samples,
                    })),
                    "semantic_eye": semantic_eye.map(|evidence| json!({
                        "score": evidence.score,
                        "authorizes_cold_identity": evidence.authorizes_cold_identity,
                        "plausible_lids": evidence.plausible_lids,
                        "pupil_to_limbus_radius_ratio": evidence.pupil_to_limbus_radius_ratio,
                        "pupil_canonical_offset": evidence.pupil_canonical_offset,
                        "annular_active_sectors": evidence.annular.active_sectors,
                        "annular_upper_quartile_gradient": evidence.annular.upper_quartile_gradient,
                        "annular_global_orientation_anisotropy": evidence.annular.global_orientation_anisotropy,
                        "pupil_darkness_log": evidence.annular.pupil_darkness_log,
                        "straight_band_veto": evidence.straight_band.veto,
                        "upper_lid_plausible": evidence.upper_lid.plausible,
                        "lower_lid_plausible": evidence.lower_lid.plausible,
                    })),
                    "admissible": driving_hypothesis_admissible(candidate),
                    "visible_label_rms_px": labeled_rms(&residuals),
                    "visible_label_p95_px": if residuals.is_empty() { None } else { Some(percentile(&residuals, 0.95)) },
                })
            }),
            "multibank_search_ms": multibank_search_ms,
            "bounded_wide64_recovery": {
                "policy": "64 full-resolution RAW proposals; union of direct top-4 and independent material/lid top-4 receives complete anatomy; label-blind final-rank plus semantic hysteresis arbitrates against incumbent",
                "elapsed_ms": bounded_wide_elapsed_ms,
                "recovery_selected": bounded_wide_selected,
                "recovery": bounded_wide_recovery.map(|recovery| json!({
                    "proposal_count": recovery.proposal_count,
                    "finalist_count": recovery.finalist_count,
                    "proposal_pose": labeled_pose_json(Some(recovery.proposal.pose)),
                    "completed_pose": labeled_pose_json(Some(recovery.hypothesis.pose)),
                    "completion_rank": recovery.completion_rank,
                    "final_outer_rank": recovery.final_audit.rank,
                    "semantic_score": recovery.semantic.score,
                    "semantic_plausible_lids": recovery.semantic.plausible_lids,
                    "semantic_authorizes_cold_identity": recovery.semantic.authorizes_cold_identity,
                    "dual_edge_score": recovery.final_audit.evidence.score,
                })),
                "selected_candidate": bounded_wide_candidate.map(|candidate| json!({
                    "pose": labeled_pose_json(Some(candidate.pose)),
                    "source": if bounded_wide_selected { "wide64-bounded-recovery" } else { "incumbent" },
                    "visible_point_distance": labeled_distance_summary_json(&bounded_wide_case_visible_distances),
                    "point_geometry_pass": bounded_wide_point_geometry_pass,
                    "full_reference_geometry_pass": if reference_pose.is_some() {
                        Some(bounded_wide_full_reference_pass)
                    } else {
                        None
                    },
                    "combined_geometry_pass": bounded_wide_combined_geometry_pass,
                    "reference_ellipse_comparison": bounded_wide_reference_comparison,
                })),
            },
            "wide64_full_anatomy_search": {
                "policy": "all 64 label-blind multi-bank proposals receive complete native-RAW anatomy and semantic scoring; labels are evaluation-only",
                "elapsed_ms": wide64_elapsed_ms,
                "proposals": multibank_partial_audit.len(),
                "completed_roads": wide64_roads.len(),
                "full_roi_context_roads": wide64_roads.iter().filter(|road| road.full_roi_context).count(),
                "admissible_full_context_roads": wide64_roads.iter().filter(|road| {
                    road.full_roi_context && road.anatomy_admissible
                }).count(),
                "semantic_complete_full_context_roads": wide64_roads.iter().filter(|road| {
                    road.full_roi_context
                        && road.semantic_geometry
                        && road.complete_pupil_headed_geometry
                }).count(),
                "rank_only_selected": labeled_wide64_selection_json(
                    wide64_rank_only_candidate,
                    &visible,
                    reference_pose,
                ),
                "admissible_selected": labeled_wide64_selection_json(
                    wide64_admissible_candidate,
                    &visible,
                    reference_pose,
                ),
                "admissible_final_outer_audit": wide64_admissible_outer_audit.as_ref().map(
                    |audit| json!({
                        "label_blind_outer_rank": audit.rank,
                        "semantic_score": audit.semantic.score,
                        "semantic_plausible_lids": audit.semantic.plausible_lids,
                        "semantic_authorizes_cold_identity": audit.semantic.authorizes_cold_identity,
                        "semantic_straight_band_veto": audit.semantic.straight_band.veto,
                        "dual_edge_score": audit.evidence.score,
                        "dual_edge_selection_score": DrivingMultibankLimbusProposal {
                            pose: wide64_admissible_candidate
                                .map(|(_, candidate)| candidate.pose)
                                .expect("wide64 audit requires its selected candidate"),
                            evidence: audit.evidence,
                        }.selection_score(),
                    })
                ),
                "semantic_complete_selected": labeled_wide64_selection_json(
                    wide64_semantic_candidate,
                    &visible,
                    reference_pose,
                ),
                "semantic_complete_final_outer_audit": wide64_semantic_outer_audit.as_ref().map(
                    |audit| json!({
                        "label_blind_outer_rank": audit.rank,
                        "semantic_score": audit.semantic.score,
                        "semantic_plausible_lids": audit.semantic.plausible_lids,
                        "semantic_authorizes_cold_identity": audit.semantic.authorizes_cold_identity,
                        "semantic_straight_band_veto": audit.semantic.straight_band.veto,
                        "dual_edge_score": audit.evidence.score,
                        "dual_edge_selection_score": DrivingMultibankLimbusProposal {
                            pose: wide64_semantic_candidate
                                .map(|(_, candidate)| candidate.pose)
                                .expect("wide64 audit requires its selected candidate"),
                            evidence: audit.evidence,
                        }.selection_score(),
                    })
                ),
                "roads": wide64_roads.iter().map(|road| {
                    let distances = labeled_ellipse_residuals(&visible, Some(road.anatomy.pose));
                    let visible_p95 = (!distances.is_empty()).then(|| percentile(&distances, 0.95));
                    let visible_max = (!distances.is_empty()).then(|| percentile(&distances, 1.0));
                    let point_geometry_pass = visible_p95.is_some_and(|distance| distance <= 5.0)
                        && visible_max.is_some_and(|distance| distance <= 8.0);
                    let reference_comparison = labeled_ellipse_reference_comparison_json(
                        reference_pose,
                        Some(road.anatomy.pose),
                    );
                    let full_reference_geometry_pass = reference_pose.map(|_| {
                        reference_comparison
                            .get("symmetric")
                            .and_then(|summary| summary.get("p95_px"))
                            .and_then(Value::as_f64)
                            .is_some_and(|distance| distance <= 8.0)
                    });
                    // Optional label-blind localization probe. Selection and
                    // every anatomy gate above remain frozen; labels are used
                    // only after the completed road has taken the requested
                    // native analog-edge step. This isolates whether a miss is
                    // ranking/identity or merely subpixel ridge closure.
                    let completed_analog_polish = analog_edge_target_px.and_then(|target| {
                        driving_multibank_analog_polish_pose(
                            &raw,
                            width,
                            height,
                            road.anatomy.pose,
                            radius_prior,
                            target,
                        )
                    });
                    let completed_analog_distances = labeled_ellipse_residuals(
                        &visible,
                        completed_analog_polish.map(|polish| polish.pose),
                    );
                    json!({
                        "index": road.index,
                        "rank": road.rank,
                        "proposal_pose": labeled_pose_json(Some(road.proposal.pose)),
                        "completed_pose": labeled_pose_json(Some(road.anatomy.pose)),
                        "full_roi_context": road.full_roi_context,
                        "anatomy_admissible": road.anatomy_admissible,
                        "semantic_geometry": road.semantic_geometry,
                        "complete_pupil_headed_geometry": road.complete_pupil_headed_geometry,
                        "proposal_primary_score": road.proposal.evidence.score,
                        "proposal_selection_score": road.proposal.selection_score(),
                        "anatomy": {
                            "score": road.anatomy.score,
                            "limbus_score": road.anatomy.limbus_score,
                            "white_score": road.anatomy.white_score,
                            "far_sclera_score": road.anatomy.far_sclera_score,
                            "pupil_score": road.anatomy.pupil_score,
                            "pupil_enclosure": road.anatomy.pupil_enclosure,
                            "pupil_margin": road.anatomy.pupil_margin,
                            "pupil_horizon": road.anatomy.pupil_horizon,
                            "through_eye_score": road.anatomy.through_eye_score,
                            "bilateral_limbus_order": road.anatomy.bilateral_limbus_order,
                            "pupil_projected_area_radius_px": road.anatomy.pupil_projected_area_radius_px,
                            "pupil_canonical_radius": road.anatomy
                                .pupil_boundary_canonical()
                                .0
                                .hypot(road.anatomy.pupil_boundary_canonical().1),
                        },
                        "semantic": {
                            "score": road.semantic.score,
                            "authorizes_cold_identity": road.semantic.authorizes_cold_identity,
                            "plausible_lids": road.semantic.plausible_lids,
                            "annular_active_sectors": road.semantic.annular.active_sectors,
                            "annular_samples": road.semantic.annular.samples,
                            "annular_upper_quartile_gradient": road.semantic.annular.upper_quartile_gradient,
                            "annular_global_orientation_anisotropy": road.semantic.annular.global_orientation_anisotropy,
                            "straight_band_veto": road.semantic.straight_band.veto,
                        },
                        "visible_point_distance": labeled_distance_summary_json(&distances),
                        "point_geometry_pass": point_geometry_pass,
                        "full_reference_geometry_pass": full_reference_geometry_pass,
                        "combined_geometry_pass": point_geometry_pass
                            && full_reference_geometry_pass.unwrap_or(true),
                        "reference_ellipse_comparison": reference_comparison,
                        "completed_analog_polish": completed_analog_polish.map(|polish| json!({
                            "pose": labeled_pose_json(Some(polish.pose)),
                            "iterations": polish.iterations,
                            "samples": polish.samples,
                            "target_offset_px": polish.target_offset_px,
                            "initial_offset_rms_px": polish.initial_offset_rms_px,
                            "final_offset_rms_px": polish.final_offset_rms_px,
                            "visible_point_distance": labeled_distance_summary_json(
                                &completed_analog_distances,
                            ),
                            "reference_ellipse_comparison": labeled_ellipse_reference_comparison_json(
                                reference_pose,
                                Some(polish.pose),
                            ),
                        })),
                    })
                }).collect::<Vec<_>>(),
            },
            "multibank_partial_audit": multibank_partial_audit.iter().enumerate().map(|(index, proposal)| {
                let visible_residuals = labeled_ellipse_residuals(&visible, Some(proposal.pose));
                let guessed_residuals = labeled_ellipse_residuals(&guessed, Some(proposal.pose));
                let visible_p95 = (!visible_residuals.is_empty())
                    .then(|| percentile(&visible_residuals, 0.95));
                let visible_max = (!visible_residuals.is_empty())
                    .then(|| percentile(&visible_residuals, 1.0));
                let point_geometry_pass = visible_p95.is_some_and(|distance| distance <= 5.0)
                    && visible_max.is_some_and(|distance| distance <= 8.0);
                let reference_comparison = labeled_ellipse_reference_comparison_json(
                    reference_pose,
                    Some(proposal.pose),
                );
                let full_reference_geometry_pass = reference_pose.map(|_| {
                    reference_comparison
                        .get("symmetric")
                        .and_then(|summary| summary.get("p95_px"))
                        .and_then(Value::as_f64)
                        .is_some_and(|distance| distance <= 8.0)
                });
                let combined_geometry_pass = point_geometry_pass
                    && full_reference_geometry_pass.unwrap_or(true);
                let semantic_probe = score_driving_pose(&raw, width, height, proposal.pose);
                let material_evidence = driving_material_view
                    .as_ref()
                    .and_then(|view| driving_limbus_material_evidence(view, proposal.pose));
                let partial_rank = partial_ranked
                    .iter()
                    .position(|ranked| ranked.proposal.pose == proposal.pose);
                let partial_rank_score = partial_rank.and_then(|rank| {
                    partial_ranked.get(rank).map(|ranked| ranked.score)
                });
                let partial_rank_features = partial_rank.and_then(|rank| {
                    partial_ranked
                        .get(rank)
                        .map(|ranked| ranked.features.values().to_vec())
                });
                // Keep the candidate-conditioned semantic scene in the audit.
                // This remains diagnostic here: an unobserved or ROI-clipped
                // lid is censored evidence and must never manufacture a
                // penalty that changes the physical conic solve.
                let eyelid_scene = labeled_eyelid_scene_json(
                    &raw,
                    width,
                    height,
                    Some(proposal.pose),
                    focus.pupil_hint,
                );
                let analog_polish = analog_edge_target_px.and_then(|target| {
                    driving_multibank_analog_polish_pose(
                        &raw,
                        width,
                        height,
                        proposal.pose,
                        radius_prior,
                        target,
                    )
                });
                let analog_visible_residuals = labeled_ellipse_residuals(
                    &visible,
                    analog_polish.map(|polish| polish.pose),
                );
                let analog_guessed_residuals = labeled_ellipse_residuals(
                    &guessed,
                    analog_polish.map(|polish| polish.pose),
                );
                json!({
                    "index": index,
                    "pose": labeled_pose_json(Some(proposal.pose)),
                    "primary_score": proposal.evidence.score,
                    "selection_score": proposal.selection_score(),
                    "partial_rank": partial_rank,
                    "partial_rank_score": partial_rank_score,
                    "partial_rank_features": partial_rank_features,
                    "seed_upper_lid_status": seed_eyelid_scene.as_ref().map(|scene| scene.upper_status.label()),
                    "seed_lower_lid_status": seed_eyelid_scene.as_ref().map(|scene| scene.lower_status.label()),
                    "seed_upper_lid_points": seed_eyelid_scene.as_ref().map(|scene| scene.upper_margin.len()),
                    "seed_lower_lid_points": seed_eyelid_scene.as_ref().map(|scene| scene.lower_margin.len()),
                    "seed_upper_clipped_occluder_points": seed_eyelid_scene.as_ref().map(|scene| scene.upper_clipped_occluder.len()),
                    "seed_lower_clipped_occluder_points": seed_eyelid_scene.as_ref().map(|scene| scene.lower_clipped_occluder.len()),
                    "seed_upper_limbus_clearance_px": seed_eyelid_scene.as_ref().map(|scene| scene.upper_limbus_clearance_px),
                    "seed_lower_limbus_clearance_px": seed_eyelid_scene.as_ref().map(|scene| scene.lower_limbus_clearance_px),
                    "visible_label_rms_px": labeled_rms(&visible_residuals),
                    "guessed_label_rms_px": labeled_rms(&guessed_residuals),
                    "visible_point_distance": labeled_distance_summary_json(&visible_residuals),
                    "point_geometry_pass": point_geometry_pass,
                    "full_reference_geometry_pass": full_reference_geometry_pass,
                    "combined_geometry_pass": combined_geometry_pass,
                    "reference_ellipse_comparison": reference_comparison,
                    "semantic_probe": semantic_probe.map(|probe| json!({
                        "score": probe.score,
                        "white_score": probe.white_score,
                        "far_sclera_score": probe.far_sclera_score,
                        "limbus_score": probe.limbus_score,
                        "pupil_score": probe.pupil_score,
                        "pupil_enclosure": probe.pupil_enclosure,
                        "pupil_horizon": probe.pupil_horizon,
                        "pupil_margin": probe.pupil_margin,
                        "through_eye_score": probe.through_eye_score,
                        "light_cohesion": probe.light_cohesion,
                        "pupil_canonical": [probe.pupil_canonical.0, probe.pupil_canonical.1],
                    })),
                    "material_evidence": material_evidence.map(|evidence| json!({
                        "score": evidence.score,
                        "signed_double_canny_mean": evidence.signed_double_canny_mean,
                        "signed_double_canny_fraction": evidence.signed_double_canny_fraction,
                        "signed_double_canny_sectors": evidence.signed_double_canny_sectors,
                        "tangent_persistent_mean": evidence.tangent_persistent_mean,
                        "tangent_persistent_fraction": evidence.tangent_persistent_fraction,
                        "plateau_edge_dominance_mean": evidence.plateau_edge_dominance_mean,
                        "max_supported_arc_fraction": evidence.max_supported_arc_fraction,
                        "lateral_signed_double_canny_mean": evidence.lateral_signed_double_canny_mean,
                        "left_signed_double_canny_mean": evidence.left_signed_double_canny_mean,
                        "right_signed_double_canny_mean": evidence.right_signed_double_canny_mean,
                        "lateral_samples": evidence.lateral_samples,
                        "left_samples": evidence.left_samples,
                        "right_samples": evidence.right_samples,
                        "material_cohort_samples": evidence.material_cohort_samples,
                        "sector_persistent_mean": evidence.sector_persistent_mean,
                        "sector_signed_rise": evidence.sector_signed_rise,
                        "sector_inner_log_intensity": evidence.sector_inner_log_intensity,
                        "sector_outer_log_intensity": evidence.sector_outer_log_intensity,
                        "sector_visible_samples": evidence.sector_visible_samples,
                        "chroma_separation": evidence.chroma_separation,
                        "iris_chroma_cohesion": evidence.iris_chroma_cohesion,
                        "sclera_chroma_cohesion": evidence.sclera_chroma_cohesion,
                        "iris_tangential_texture": evidence.iris_tangential_texture,
                        "sclera_tangential_texture": evidence.sclera_tangential_texture,
                        "visible_samples": evidence.visible_samples,
                    })),
                    "eyelid_scene": eyelid_scene,
                    "analog_polish": analog_polish.map(|polish| json!({
                        "pose": labeled_pose_json(Some(polish.pose)),
                        "iterations": polish.iterations,
                        "samples": polish.samples,
                        "target_offset_px": polish.target_offset_px,
                        "initial_offset_rms_px": polish.initial_offset_rms_px,
                        "final_offset_rms_px": polish.final_offset_rms_px,
                        "visible_label_rms_px": labeled_rms(&analog_visible_residuals),
                        "guessed_label_rms_px": labeled_rms(&analog_guessed_residuals),
                    })),
                    "evidence": {
                        "bilateral": proposal.evidence.bilateral,
                        "left_quantile": proposal.evidence.left_quantile,
                        "right_quantile": proposal.evidence.right_quantile,
                        "left_mean_support": proposal.evidence.left_mean_support,
                        "right_mean_support": proposal.evidence.right_mean_support,
                        "left_supported_fraction": proposal.evidence.left_supported_fraction,
                        "right_supported_fraction": proposal.evidence.right_supported_fraction,
                        "weakest_lateral_quantile": proposal.evidence.weakest_lateral_quantile,
                        "mean_lateral_support": proposal.evidence.mean_lateral_support,
                        "supported_fraction": proposal.evidence.supported_fraction,
                        "narrow_mean": proposal.evidence.narrow_mean,
                        "medium_mean": proposal.evidence.medium_mean,
                        "broad_mean": proposal.evidence.broad_mean,
                        "edge_centroid_signed_mean_px": proposal.evidence.edge_centroid_signed_mean_px,
                        "edge_centroid_absolute_mean_px": proposal.evidence.edge_centroid_absolute_mean_px,
                        "edge_centroid_coherence": proposal.evidence.edge_centroid_coherence,
                        "edge_centroid_samples": proposal.evidence.edge_centroid_samples,
                        "far_sclera_step_mean": proposal.evidence.far_sclera_step_mean,
                        "outside_plateau_mean": proposal.evidence.outside_plateau_mean,
                        "outside_secondary_edge_mean": proposal.evidence.outside_secondary_edge_mean,
                        "upper_vertical_mean_support": proposal.evidence.upper_vertical_mean_support,
                        "lower_vertical_mean_support": proposal.evidence.lower_vertical_mean_support,
                        "upper_vertical_quantile": proposal.evidence.upper_vertical_quantile,
                        "lower_vertical_quantile": proposal.evidence.lower_vertical_quantile,
                        "upper_vertical_supported_fraction": proposal.evidence.upper_vertical_supported_fraction,
                        "lower_vertical_supported_fraction": proposal.evidence.lower_vertical_supported_fraction,
                        "vertical_samples": proposal.evidence.vertical_samples,
                        "inside_texture": proposal.evidence.inside_texture,
                        "outside_texture": proposal.evidence.outside_texture,
                        "far_lane_samples": proposal.evidence.far_lane_samples,
                        "samples": proposal.evidence.samples,
                    },
                })
            }).collect::<Vec<_>>(),
            "multibank_proposals": multibank_roads.iter().enumerate().map(|(proposal_index, (proposal, road))| {
                let proposal_residuals = labeled_ellipse_residuals(&visible, Some(proposal.pose));
                let proposal_guessed_residuals = labeled_ellipse_residuals(&guessed, Some(proposal.pose));
                let completed_residuals = labeled_ellipse_residuals(&visible, road.map(|road| road.pose));
                let two_d_canny = raw_motion_octrees::current_frame_canny_ellipse_proposal(
                    &two_d_canny_overlay,
                    width,
                    height,
                    iris_seed_from_driving_pose(proposal.pose),
                );
                let two_d_canny_pose = two_d_canny.as_ref().map(|candidate| DrivingAffinePose {
                    center: candidate.center,
                    major_radius: candidate.major_radius,
                    minor_radius: candidate.minor_radius,
                    angle: candidate.angle,
                });
                let two_d_canny_residuals =
                    labeled_ellipse_residuals(&visible, two_d_canny_pose);
                let measured_two_d_canny =
                    raw_motion_octrees::measured_seed_canny_support_proposal(
                        &two_d_canny_overlay,
                        width,
                        height,
                        iris_seed_from_driving_pose(proposal.pose),
                    );
                let measured_two_d_canny_pose = measured_two_d_canny.as_ref().map(|candidate| {
                    DrivingAffinePose {
                        center: candidate.center,
                        major_radius: candidate.major_radius,
                        minor_radius: candidate.minor_radius,
                        angle: candidate.angle,
                    }
                });
                let measured_two_d_canny_residuals =
                    labeled_ellipse_residuals(&visible, measured_two_d_canny_pose);
                json!({
                    "index": proposal_index,
                    "pose": labeled_pose_json(Some(proposal.pose)),
                    "evidence": {
                        "score": proposal.evidence.score,
                        "selection_score": proposal.selection_score(),
                        "left_quantile": proposal.evidence.left_quantile,
                        "right_quantile": proposal.evidence.right_quantile,
                        "left_mean_support": proposal.evidence.left_mean_support,
                        "right_mean_support": proposal.evidence.right_mean_support,
                        "left_supported_fraction": proposal.evidence.left_supported_fraction,
                        "right_supported_fraction": proposal.evidence.right_supported_fraction,
                        "weakest_lateral_quantile": proposal.evidence.weakest_lateral_quantile,
                        "mean_lateral_support": proposal.evidence.mean_lateral_support,
                        "supported_fraction": proposal.evidence.supported_fraction,
                        "narrow_mean": proposal.evidence.narrow_mean,
                        "medium_mean": proposal.evidence.medium_mean,
                        "broad_mean": proposal.evidence.broad_mean,
                        "edge_centroid_signed_mean_px": proposal.evidence.edge_centroid_signed_mean_px,
                        "edge_centroid_absolute_mean_px": proposal.evidence.edge_centroid_absolute_mean_px,
                        "edge_centroid_coherence": proposal.evidence.edge_centroid_coherence,
                        "edge_centroid_samples": proposal.evidence.edge_centroid_samples,
                        "far_sclera_step_mean": proposal.evidence.far_sclera_step_mean,
                        "outside_plateau_mean": proposal.evidence.outside_plateau_mean,
                        "outside_secondary_edge_mean": proposal.evidence.outside_secondary_edge_mean,
                        "far_lane_samples": proposal.evidence.far_lane_samples,
                        "upper_vertical_mean_support": proposal.evidence.upper_vertical_mean_support,
                        "lower_vertical_mean_support": proposal.evidence.lower_vertical_mean_support,
                        "upper_vertical_quantile": proposal.evidence.upper_vertical_quantile,
                        "lower_vertical_quantile": proposal.evidence.lower_vertical_quantile,
                        "upper_vertical_supported_fraction": proposal.evidence.upper_vertical_supported_fraction,
                        "lower_vertical_supported_fraction": proposal.evidence.lower_vertical_supported_fraction,
                        "vertical_samples": proposal.evidence.vertical_samples,
                        "inside_texture": proposal.evidence.inside_texture,
                        "outside_texture": proposal.evidence.outside_texture,
                        "samples": proposal.evidence.samples,
                        "left_samples": proposal.evidence.left_samples,
                        "right_samples": proposal.evidence.right_samples,
                        "bilateral": proposal.evidence.bilateral,
                    },
                    "visible_label_rms_px": labeled_rms(&proposal_residuals),
                    "guessed_label_rms_px": labeled_rms(&proposal_guessed_residuals),
                    "two_d_current_canny": two_d_canny.as_ref().map(|candidate| json!({
                        "pose": labeled_pose_json(two_d_canny_pose),
                        "confidence": candidate.confidence,
                        "seed_confidence": candidate.seed_confidence,
                        "edge_support": candidate.edge_support,
                        "angular_coverage": candidate.angular_coverage,
                        "opposing_meridians": candidate.opposing_meridians,
                        "visible_label_rms_px": labeled_rms(&two_d_canny_residuals),
                    })),
                    "two_d_measured_seed_canny": measured_two_d_canny.as_ref().map(|candidate| json!({
                        "pose": labeled_pose_json(measured_two_d_canny_pose),
                        "confidence": candidate.confidence,
                        "edge_support": candidate.edge_support,
                        "angular_coverage": candidate.angular_coverage,
                        "opposing_meridians": candidate.opposing_meridians,
                        "visible_label_rms_px": labeled_rms(&measured_two_d_canny_residuals),
                    })),
                    "eyelid_scene": (proposal_index == 0).then(|| {
                        labeled_eyelid_scene_json(
                            &raw,
                            width,
                            height,
                            Some(proposal.pose),
                            focus.pupil_hint,
                        )
                    }),
                    "completed_road": road.map(|road| json!({
                        "pose": labeled_pose_json(Some(road.pose)),
                        "score": road.score,
                        "admissible": driving_hypothesis_admissible(road),
                        "visible_label_rms_px": labeled_rms(&completed_residuals),
                        "white_score": road.white_score,
                        "far_sclera_score": road.far_sclera_score,
                        "limbus_score": road.limbus_score,
                        "pupil_enclosure": road.pupil_enclosure,
                        "pupil_margin": road.pupil_margin,
                        "pupil_horizon": road.pupil_horizon,
                        "through_eye_score": road.through_eye_score,
                        "bilateral_limbus_order": road.bilateral_limbus_order,
                        "double_sclera_10_deg_score": road.double_sclera_10_deg_score,
                        "double_sclera_10_deg_support": road.double_sclera_10_deg_support,
                        "pupil_boundary_center": road.pupil_boundary_center(),
                        "pupil_projected_area_radius_px": road.pupil_projected_area_radius_px,
                        "pupil_to_limbus_radius_ratio": driving_pupil_to_limbus_radius_ratio(road),
                        "outer_multibank": driving_multibank_limbus_evidence(
                            &raw,
                            width,
                            height,
                            road.pose,
                        ).map(|evidence| json!({
                            "score": evidence.score,
                            "selection_score": DrivingMultibankLimbusProposal {
                                pose: road.pose,
                                evidence,
                            }.selection_score(),
                            "weakest_lateral_quantile": evidence.weakest_lateral_quantile,
                            "supported_fraction": evidence.supported_fraction,
                            "samples": evidence.samples,
                        })),
                        "semantic_eye": driving_semantic_eye_evidence(
                            &raw,
                            width,
                            height,
                            sensor_origin,
                            road,
                        ).map(|evidence| json!({
                            "score": evidence.score,
                            "authorizes_cold_identity": evidence.authorizes_cold_identity,
                            "plausible_lids": evidence.plausible_lids,
                            "pupil_to_limbus_radius_ratio": evidence.pupil_to_limbus_radius_ratio,
                            "pupil_canonical_offset": evidence.pupil_canonical_offset,
                            "annular_active_sectors": evidence.annular.active_sectors,
                            "annular_upper_quartile_gradient": evidence.annular.upper_quartile_gradient,
                            "annular_global_orientation_anisotropy": evidence.annular.global_orientation_anisotropy,
                            "pupil_darkness_log": evidence.annular.pupil_darkness_log,
                            "straight_band_veto": evidence.straight_band.veto,
                            "upper_lid_plausible": evidence.upper_lid.plausible,
                            "lower_lid_plausible": evidence.lower_lid.plausible,
                        })),
                    })),
                })
            }).collect::<Vec<_>>(),
            "current_frame_roads": audited_roads.iter().enumerate().map(|(road_index, road)| {
                let residuals = labeled_ellipse_residuals(&visible, Some(road.pose));
                let multibank = driving_multibank_limbus_evidence(
                    &raw,
                    width,
                    height,
                    road.pose,
                );
                json!({
                    "index": road_index,
                    "selected": candidate == Some(*road),
                    "pose": labeled_pose_json(Some(road.pose)),
                    "score": road.score,
                    "normal_score": road.normal_score,
                    "white_score": road.white_score,
                    "far_sclera_score": road.far_sclera_score,
                    "limbus_score": road.limbus_score,
                    "pupil_score": road.pupil_score,
                    "pupil_enclosure": road.pupil_enclosure,
                    "pupil_margin": road.pupil_margin,
                    "pupil_horizon": road.pupil_horizon,
                    "through_eye_score": road.through_eye_score,
                    "bilateral_limbus_order": road.bilateral_limbus_order,
                    "double_sclera_10_deg_score": road.double_sclera_10_deg_score,
                    "double_sclera_10_deg_support": road.double_sclera_10_deg_support,
                    "double_sclera_10_deg_phase": road.double_sclera_10_deg_phase,
                    "light_cohesion": road.light_cohesion,
                    "affine_departure_fraction": road.affine_departure_fraction,
                    "affine_departure_ratio": road.affine_departure_ratio,
                    "affine_repair_fraction": road.affine_repair_fraction,
                    "lower_limbus_direct_visibility": road.lower_limbus_direct_visibility,
                    "pupil_center": driving_pose_point(road.pose, road.pupil_canonical),
                    "pupil_projected_area_radius_px": road.pupil_projected_area_radius_px,
                    "admissible": driving_hypothesis_admissible(*road),
                    "multibank_limbus": multibank.map(|evidence| json!({
                        "score": evidence.score,
                        "selection_score": DrivingMultibankLimbusProposal {
                            pose: road.pose,
                            evidence,
                        }.selection_score(),
                        "left_quantile": evidence.left_quantile,
                        "right_quantile": evidence.right_quantile,
                        "left_mean_support": evidence.left_mean_support,
                        "right_mean_support": evidence.right_mean_support,
                        "left_supported_fraction": evidence.left_supported_fraction,
                        "right_supported_fraction": evidence.right_supported_fraction,
                        "weakest_lateral_quantile": evidence.weakest_lateral_quantile,
                        "mean_lateral_support": evidence.mean_lateral_support,
                        "supported_fraction": evidence.supported_fraction,
                        "narrow_mean": evidence.narrow_mean,
                        "medium_mean": evidence.medium_mean,
                        "broad_mean": evidence.broad_mean,
                        "edge_centroid_signed_mean_px": evidence.edge_centroid_signed_mean_px,
                        "edge_centroid_absolute_mean_px": evidence.edge_centroid_absolute_mean_px,
                        "edge_centroid_coherence": evidence.edge_centroid_coherence,
                        "edge_centroid_samples": evidence.edge_centroid_samples,
                        "far_sclera_step_mean": evidence.far_sclera_step_mean,
                        "outside_plateau_mean": evidence.outside_plateau_mean,
                        "outside_secondary_edge_mean": evidence.outside_secondary_edge_mean,
                        "far_lane_samples": evidence.far_lane_samples,
                        "upper_vertical_mean_support": evidence.upper_vertical_mean_support,
                        "lower_vertical_mean_support": evidence.lower_vertical_mean_support,
                        "upper_vertical_quantile": evidence.upper_vertical_quantile,
                        "lower_vertical_quantile": evidence.lower_vertical_quantile,
                        "upper_vertical_supported_fraction": evidence.upper_vertical_supported_fraction,
                        "lower_vertical_supported_fraction": evidence.lower_vertical_supported_fraction,
                        "vertical_samples": evidence.vertical_samples,
                        "inside_texture": evidence.inside_texture,
                        "outside_texture": evidence.outside_texture,
                        "samples": evidence.samples,
                        "left_samples": evidence.left_samples,
                        "right_samples": evidence.right_samples,
                        "bilateral": evidence.bilateral,
                    })),
                    "visible_label_rms_px": labeled_rms(&residuals),
                    "visible_label_p95_px": if residuals.is_empty() { None } else { Some(percentile(&residuals, 0.95)) },
                })
            }).collect::<Vec<_>>(),
            "driving_candidate": candidate.map(|candidate| json!({
                "pose": labeled_pose_json(Some(candidate.pose)),
                "score": candidate.score,
                "white_score": candidate.white_score,
                "far_sclera_score": candidate.far_sclera_score,
                "limbus_score": candidate.limbus_score,
                "pupil_score": candidate.pupil_score,
                "pupil_enclosure": candidate.pupil_enclosure,
                "pupil_margin": candidate.pupil_margin,
                "pupil_horizon": candidate.pupil_horizon,
                "through_eye_score": candidate.through_eye_score,
                "broad_through_eye_score": candidate.broad_through_eye_score,
                "bilateral_limbus_order": candidate.bilateral_limbus_order,
                "double_sclera_10_deg_score": candidate.double_sclera_10_deg_score,
                "double_sclera_10_deg_support": candidate.double_sclera_10_deg_support,
                "light_cohesion": candidate.light_cohesion,
                "affine_departure_fraction": candidate.affine_departure_fraction,
                "affine_departure_ratio": candidate.affine_departure_ratio,
                "affine_repair_fraction": candidate.affine_repair_fraction,
                "pupil_center": driving_pose_point(candidate.pose, candidate.pupil_canonical),
                "pupil_projected_area_radius_px": candidate.pupil_projected_area_radius_px,
                "refinement_laps": candidate.refinement_laps,
                "admissible": driving_hypothesis_admissible(candidate),
                "visible_label_rms_px": labeled_rms(&case_visible_residuals),
                "visible_label_p95_px": if case_visible_residuals.is_empty() { None } else { Some(percentile(&case_visible_residuals, 0.95)) },
                "guessed_label_rms_px": labeled_rms(&case_guessed_residuals),
            })),
            "multibank_refined_candidate": multibank_refined_candidate.map(|candidate| {
                let visible_residuals = labeled_ellipse_residuals(&visible, Some(candidate.pose));
                let guessed_residuals = labeled_ellipse_residuals(&guessed, Some(candidate.pose));
                json!({
                    "pose": labeled_pose_json(Some(candidate.pose)),
                    "pupil_center": driving_pose_point(candidate.pose, candidate.pupil_canonical),
                    "visible_label_rms_px": labeled_rms(&visible_residuals),
                    "guessed_label_rms_px": labeled_rms(&guessed_residuals),
                })
            }),
            "production_frame_local_candidate": production_frame_local_candidate.map(|candidate| {
                let visible_residuals = labeled_ellipse_residuals(&visible, Some(candidate.pose));
                let guessed_residuals = labeled_ellipse_residuals(&guessed, Some(candidate.pose));
                let limbus_evidence = driving_multibank_limbus_evidence(
                    &raw,
                    width,
                    height,
                    candidate.pose,
                );
                json!({
                    "pose": labeled_pose_json(Some(candidate.pose)),
                    "topology_pupil_center": driving_pose_point(candidate.pose, candidate.pupil_canonical),
                    "boundary_pupil_center": candidate.pupil_boundary_center(),
                    "pupil_projected_area_radius_px": candidate.pupil_projected_area_radius_px,
                    "pupil_to_limbus_area_radius_ratio": candidate.pupil_projected_area_radius_px
                        .map(|radius| radius / (candidate.pose.major_radius * candidate.pose.minor_radius).sqrt().max(1.0)),
                    "pupil_void_curve_support": {
                        "pupil_score": candidate.pupil_score,
                        "pupil_enclosure": candidate.pupil_enclosure,
                        "pupil_horizon": candidate.pupil_horizon,
                        "pupil_margin": candidate.pupil_margin,
                        "through_eye_score": candidate.through_eye_score,
                    },
                    "dual_edge_limbus_support": limbus_evidence.map(|evidence| json!({
                        "score": evidence.score,
                        "selection_score": DrivingMultibankLimbusProposal {
                            pose: candidate.pose,
                            evidence,
                        }.selection_score(),
                        "bilateral": evidence.bilateral,
                        "left_mean_support": evidence.left_mean_support,
                        "right_mean_support": evidence.right_mean_support,
                        "left_supported_fraction": evidence.left_supported_fraction,
                        "right_supported_fraction": evidence.right_supported_fraction,
                        "weakest_lateral_quantile": evidence.weakest_lateral_quantile,
                        "supported_fraction": evidence.supported_fraction,
                        "narrow_mean": evidence.narrow_mean,
                        "medium_mean": evidence.medium_mean,
                        "broad_mean": evidence.broad_mean,
                        "edge_centroid_signed_mean_px": evidence.edge_centroid_signed_mean_px,
                        "edge_centroid_absolute_mean_px": evidence.edge_centroid_absolute_mean_px,
                        "edge_centroid_coherence": evidence.edge_centroid_coherence,
                        "far_sclera_step_mean": evidence.far_sclera_step_mean,
                        "outside_plateau_mean": evidence.outside_plateau_mean,
                        "outside_secondary_edge_mean": evidence.outside_secondary_edge_mean,
                        "samples": evidence.samples,
                    })),
                    "label_blind_outer_audit": production_frame_local_outer_audit.as_ref().map(
                        |audit| json!({
                            "rank": audit.rank,
                            "semantic_score": audit.semantic.score,
                            "semantic_plausible_lids": audit.semantic.plausible_lids,
                            "semantic_authorizes_cold_identity": audit.semantic.authorizes_cold_identity,
                            "semantic_straight_band_veto": audit.semantic.straight_band.veto,
                            "dual_edge_score": audit.evidence.score,
                            "dual_edge_selection_score": DrivingMultibankLimbusProposal {
                                pose: candidate.pose,
                                evidence: audit.evidence,
                            }.selection_score(),
                        })
                    ),
                    "used_strong_multibank_seed": production_multibank.is_some(),
                    "admissible": driving_hypothesis_admissible(candidate),
                    "visible_label_rms_px": labeled_rms(&visible_residuals),
                    "guessed_label_rms_px": labeled_rms(&guessed_residuals),
                    "visible_point_distance": labeled_distance_summary_json(&visible_residuals),
                    "guessed_point_distance": labeled_distance_summary_json(&guessed_residuals),
                    "point_geometry_pass": production_point_geometry_pass,
                    "full_reference_geometry_pass": if reference_pose.is_some() {
                        Some(production_full_reference_pass)
                    } else {
                        None
                    },
                    "combined_geometry_pass": production_combined_geometry_pass,
                    "reference_ellipse_comparison": production_reference_comparison,
                })
            }),
            "benchmark_judgement": {
                "production_generated": production_pose.is_some(),
                "production_point_geometry_pass": production_point_geometry_pass,
                "production_full_reference_geometry_pass": if reference_pose.is_some() {
                    Some(production_full_reference_pass)
                } else {
                    None
                },
                "production_combined_geometry_pass": production_combined_geometry_pass,
                "bounded_wide64_generated": bounded_wide_pose.is_some(),
                "bounded_wide64_point_geometry_pass": bounded_wide_point_geometry_pass,
                "bounded_wide64_full_reference_geometry_pass": if reference_pose.is_some() {
                    Some(bounded_wide_full_reference_pass)
                } else {
                    None
                },
                "bounded_wide64_combined_geometry_pass": bounded_wide_combined_geometry_pass,
                "two_d_generated": two_d_measured_selected_canny_pose.is_some(),
                "two_d_point_geometry_pass": two_d_point_geometry_pass,
                "two_d_full_reference_geometry_pass": if reference_pose.is_some() {
                    Some(two_d_full_reference_pass)
                } else {
                    None
                },
                "two_d_combined_geometry_pass": two_d_combined_geometry_pass,
            },
            "label_posthoc_search_oracle": {
                "inference_input": false,
                "purpose": "separate candidate-search coverage from evidence-ranking failure",
                "wide_multibank_candidates": multibank_partial_audit.len(),
                "point_geometry_pass_available": multibank_wide_beam_point_pass,
                "combined_geometry_pass_available": multibank_wide_beam_combined_pass,
                "best_visible_point_candidate": multibank_wide_beam_point_oracle.as_ref().map(|oracle| json!({
                    "index": oracle.0,
                    "pose": labeled_pose_json(Some(oracle.1)),
                    "visible_point_distance": labeled_distance_summary_json(&oracle.2),
                    "point_geometry_pass": oracle.3 <= 5.0 && oracle.4 <= 8.0,
                })),
                "best_combined_geometry_candidate": multibank_wide_beam_combined_oracle.as_ref().map(|oracle| json!({
                    "index": oracle.0,
                    "pose": labeled_pose_json(Some(oracle.1)),
                    "visible_point_distance": labeled_distance_summary_json(&oracle.2),
                    "point_geometry_pass": oracle.3 <= 5.0 && oracle.4 <= 8.0,
                    "full_reference_geometry_pass": oracle.6.map(|distance| distance <= 8.0),
                    "combined_geometry_pass": oracle.7,
                    "reference_ellipse_comparison": oracle.5,
                })),
                "production_missed_available_point_pass": multibank_wide_beam_point_pass
                    && !production_point_geometry_pass,
                "production_missed_available_combined_pass": multibank_wide_beam_combined_pass
                    && !production_combined_geometry_pass,
            },
            "elapsed_ms": elapsed_ms,
        }));
    }
    let report = json!({
        "schema": "buttercup-offline-labeled-driving-eval-v2",
        "algorithm": "exact native-resolution production Driving current-frame proposal; temporal publication deliberately excluded",
        "ground_truth_metric": {
            "name": "shortest screen-space Euclidean point-to-ellipse distance",
            "coordinate_space": "native lossless RAW ROI pixels",
            "numeric_method": "1024-phase global local-minimum enumeration plus 64-step golden-section refinement",
            "visible_point_case_pass": "p95 <= 5 px and max <= 8 px",
            "full_reference_case_pass": "symmetric 720-sample contour p95 <= 8 px",
            "combined_case_pass": "visible-point gate, plus full-reference gate whenever a fitted reference ellipse is available",
            "missing_prediction_policy": "hard case failure; never omitted from coverage or pass rate",
            "supporting_evidence_policy": "dual-edge limbus and pupil-void evidence are reported separately and cannot erase point geometry error",
        },
        "analog_edge_target_px": analog_edge_target_px,
        "cases_requested": label_paths.len(),
        "cases_fitted": fitted,
        "aggregate": {
            "visible_label_rms_px": labeled_rms(&visible_residuals),
            "visible_label_p50_px": if visible_residuals.is_empty() { None } else { Some(percentile(&visible_residuals, 0.50)) },
            "visible_label_p95_px": if visible_residuals.is_empty() { None } else { Some(percentile(&visible_residuals, 0.95)) },
        },
        "production_frame_local_benchmark": {
            "cases_requested": label_paths.len(),
            "generated_cases": production_generated_cases,
            "coverage_fraction": production_generated_cases as f64 / label_paths.len().max(1) as f64,
            "point_geometry_pass_cases": production_point_geometry_pass_cases,
            "point_geometry_pass_fraction_all_cases": production_point_geometry_pass_cases as f64 / label_paths.len().max(1) as f64,
            "full_reference_cases": production_full_reference_cases,
            "full_reference_pass_cases": production_full_reference_pass_cases,
            "full_reference_pass_fraction": production_full_reference_pass_cases as f64 / production_full_reference_cases.max(1) as f64,
            "combined_geometry_pass_cases": production_combined_geometry_pass_cases,
            "combined_geometry_pass_fraction_all_cases": production_combined_geometry_pass_cases as f64 / label_paths.len().max(1) as f64,
            "visible_point_distance_on_generated": labeled_distance_summary_json(&production_visible_distances),
            "guessed_point_distance_on_generated": labeled_distance_summary_json(&production_guessed_distances),
            "failed_cases": production_failed_cases,
        },
        "bounded_wide64_frame_local_benchmark": {
            "cases_requested": label_paths.len(),
            "generated_cases": bounded_wide_generated_cases,
            "coverage_fraction": bounded_wide_generated_cases as f64 / label_paths.len().max(1) as f64,
            "point_geometry_pass_cases": bounded_wide_point_geometry_pass_cases,
            "point_geometry_pass_fraction_all_cases": bounded_wide_point_geometry_pass_cases as f64 / label_paths.len().max(1) as f64,
            "full_reference_cases": bounded_wide_full_reference_cases,
            "full_reference_pass_cases": bounded_wide_full_reference_pass_cases,
            "full_reference_pass_fraction": bounded_wide_full_reference_pass_cases as f64 / bounded_wide_full_reference_cases.max(1) as f64,
            "combined_geometry_pass_cases": bounded_wide_combined_geometry_pass_cases,
            "combined_geometry_pass_fraction_all_cases": bounded_wide_combined_geometry_pass_cases as f64 / label_paths.len().max(1) as f64,
            "visible_point_distance_on_generated": labeled_distance_summary_json(&bounded_wide_visible_distances),
            "failed_cases": bounded_wide_failed_cases,
        },
        "two_d_current_frame_benchmark": {
            "cases_requested": label_paths.len(),
            "generated_cases": two_d_generated_cases,
            "coverage_fraction": two_d_generated_cases as f64 / label_paths.len().max(1) as f64,
            "point_geometry_pass_cases": two_d_point_geometry_pass_cases,
            "point_geometry_pass_fraction_all_cases": two_d_point_geometry_pass_cases as f64 / label_paths.len().max(1) as f64,
            "full_reference_cases": two_d_full_reference_cases,
            "full_reference_pass_cases": two_d_full_reference_pass_cases,
            "full_reference_pass_fraction": two_d_full_reference_pass_cases as f64 / two_d_full_reference_cases.max(1) as f64,
            "combined_geometry_pass_cases": two_d_combined_geometry_pass_cases,
            "combined_geometry_pass_fraction_all_cases": two_d_combined_geometry_pass_cases as f64 / label_paths.len().max(1) as f64,
            "visible_point_distance_on_generated": labeled_distance_summary_json(&two_d_visible_distances),
            "failed_cases": two_d_failed_cases,
        },
        "wide_multibank_search_coverage": {
            "policy": "post-hoc label oracle only; never an inference input",
            "cases_requested": label_paths.len(),
            "cases_with_candidates": multibank_wide_beam_search_cases,
            "cases_with_a_point-passing_candidate": multibank_wide_beam_point_pass_cases,
            "point-passing_candidate_fraction_all_cases": multibank_wide_beam_point_pass_cases as f64 / label_paths.len().max(1) as f64,
            "cases_with_a_combined_geometry_passing_candidate": multibank_wide_beam_combined_pass_cases,
            "combined_geometry_passing_candidate_fraction_all_cases": multibank_wide_beam_combined_pass_cases as f64 / label_paths.len().max(1) as f64,
            "production_missed_available_point_pass_cases": production_missed_available_point_pass_cases,
            "production_missed_available_combined_pass_cases": production_missed_available_combined_pass_cases,
        },
        "cases": cases,
    });
    let output = File::create(&output_path)
        .map_err(|error| format!("create {}: {error}", output_path.display()))?;
    serde_json::to_writer_pretty(output, &report)
        .map_err(|error| format!("write {}: {error}", output_path.display()))?;
    println!("{}", output_path.display());
    Ok(())
}

pub(super) fn run<I>(mut args: I) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    let output_path = PathBuf::from(args.next().ok_or_else(|| {
        "usage: buttercup_wayland_raw_eyes --offline-segmentation-replay OUTPUT.json FRAMES.jsonl STREAM.raw10 [START] [COUNT] [LABEL] [all|2d-only] [CLOCK.jsonl SCREEN-PRESENTATIONS.jsonl]".to_string()
    })?);
    let index_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "missing frames.jsonl".to_string())?,
    );
    let stream_path = PathBuf::from(
        args.next()
            .ok_or_else(|| "missing RAW stream".to_string())?,
    );
    let start = parse_usize(args.next(), 0, "start")?;
    let count = parse_usize(args.next(), usize::MAX, "count")?;
    let label = args.next().unwrap_or_else(|| "subject-right".to_string());
    let replay_scope = args.next().unwrap_or_else(|| "all".to_string());
    if replay_scope != "all" && replay_scope != "2d-only" {
        return Err(format!(
            "offline replay scope must be all or 2d-only, got {replay_scope}"
        ));
    }
    let optical_clock_path = args.next().map(PathBuf::from);
    let screen_manifest_path = args.next().map(PathBuf::from);
    if optical_clock_path.is_some() != screen_manifest_path.is_some() {
        return Err(
            "optical pupil supervision requires both CLOCK.jsonl and SCREEN-PRESENTATIONS.jsonl"
                .to_string(),
        );
    }
    if args.next().is_some() {
        return Err("unexpected argument after screen presentation manifest".to_string());
    }
    let optical_stimulus = optical_clock_path
        .as_ref()
        .zip(screen_manifest_path.as_ref())
        .map(|(clock, manifest)| {
            pupil_clock_supervision::OpticalStimulusTrack::from_files(manifest, clock, &label)
        })
        .transpose()?;
    let run_driving = replay_scope == "all";
    let pupil_selection_audit = env::var_os("BUTTERCUP_OFFLINE_PUPIL_SELECTION_AUDIT").is_some();
    if output_path.exists() {
        return Err(format!("output already exists: {}", output_path.display()));
    }
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }

    let all_records = fs::read_to_string(&index_path)
        .map_err(|error| format!("read {}: {error}", index_path.display()))?
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let records = all_records
        .into_iter()
        .filter(|record| record.get("label").and_then(Value::as_str) == Some(label.as_str()))
        .skip(start)
        .take(count)
        .collect::<Vec<_>>();
    if records.is_empty() {
        return Err(format!("no {label} records in requested range"));
    }

    let first_timestamp_ns = integer(&records[0], "timestamp_ns")?;
    let clock_start = Instant::now();
    let mut previous_timestamp_ns = first_timestamp_ns;
    let mut previous_now = clock_start;
    let mut stream = File::open(&stream_path)
        .map_err(|error| format!("open {}: {error}", stream_path.display()))?;

    let mut native_tracker = raw_iris_focus::OuterIrisTracker::default();
    let mut cluster_native_tracker = raw_iris_focus::OuterIrisTracker::default();
    let mut driving_native_tracker = raw_iris_focus::OuterIrisTracker::default();
    let mut native_radius = raw_iris_focus::FrontoParallelLimbusRadiusTracker::default();
    let mut cluster_radius = raw_iris_focus::FrontoParallelLimbusRadiusTracker::default();
    let mut shared_global_scale = raw_motion_octrees::NativeGlobalSimilarityTracker::default();
    let mut cluster_motion = raw_motion_octrees::FourMotionOctrees::default();
    let mut cluster_center_gate = TemporalFeatureLimbusCenterGate::default();
    let mut driving_canny = raw_motion_octrees::BoundedIrisCannyTracker::default();
    let mut driving = DrivingSegmentationTracker::default();
    let mut common_pupil_center = PupilCenterStateTracker::default();
    let mut common_pupil_size = PupilSizeTracker::default();
    let mut common_pupil_radius_limiter = RadiusRateLimiter::default();
    let mut common_pupil_polar_cosolver = PupilPolarCoSolver::default();
    let mut driving_candidate_pupil_affine_temporal = PupilAffineTemporalAggregate::default();
    let mut driving_published_pupil_affine_temporal = PupilAffineTemporalAggregate::default();
    let mut pupil_affine_temporal = PupilAffineTemporalAggregate::default();
    driving.set_submode(DrivingSubmode::Normal);
    DRIVING_SCORING_SUBMODE.with(|active| active.set(DrivingSubmode::Normal));

    let mut native_summary = ModelAggregate::default();
    let mut cluster_summary = ModelAggregate::default();
    let mut driving_summary = ModelAggregate::default();
    let mut frames = Vec::with_capacity(records.len());
    let mut optical_stimulus_frames = 0usize;
    let mut clock_pupil_samples = Vec::with_capacity(records.len());

    for (local_index, record) in records.iter().enumerate() {
        let width = integer(record, "width")? as usize;
        let height = integer(record, "height")? as usize;
        let stride = integer(record, "stride")? as usize;
        let offset = integer(record, "offset")?;
        let length = integer(record, "length")? as usize;
        let sensor_origin = (
            integer(record, "sensor_x")? as u32,
            integer(record, "sensor_y")? as u32,
        );
        let timestamp_ns = integer(record, "timestamp_ns")?;
        let stimulus_pose = optical_stimulus
            .as_ref()
            .and_then(|track| track.pose_at_sensor_timestamp(timestamp_ns));
        optical_stimulus_frames += usize::from(stimulus_pose.is_some());
        let now = if timestamp_ns >= previous_timestamp_ns && timestamp_ns >= first_timestamp_ns {
            clock_start + Duration::from_nanos(timestamp_ns - first_timestamp_ns)
        } else {
            previous_now + Duration::from_millis(100)
        };
        previous_timestamp_ns = timestamp_ns;
        previous_now = now;

        stream
            .seek(SeekFrom::Start(offset))
            .map_err(|error| format!("seek {} at {offset}: {error}", stream_path.display()))?;
        let mut packed = vec![0u8; length];
        stream
            .read_exact(&mut packed)
            .map_err(|error| format!("read {} at {offset}: {error}", stream_path.display()))?;
        let raw = Arc::new(raw10::try_unpack_raw10(&packed, width, height, stride)?);
        let global_similarity = shared_global_scale.observe(
            Arc::clone(&raw),
            width,
            height,
            sensor_origin.0,
            sensor_origin.1,
        );
        let scale_prediction = shared_global_limbus_scale_prediction(global_similarity);

        let focus = raw_iris_focus::score_stream_eye(&raw, width, height);
        let partial = focus
            .roi_truncated_limbus
            .filter(|partial| roi_truncated_limbus_recovery_ready(*partial));
        let partial_frame = partial.is_some();
        let focus_anatomy = has_eye_border_structure(focus.score, focus.points.len());
        let seed_usable = focus.radius >= 20.0
            && focus.radius <= width.min(height) as f64 * 0.45
            && focus.center.0.is_finite()
            && focus.center.1.is_finite()
            && (0.0..width as f64).contains(&focus.center.0)
            && (0.0..height as f64).contains(&focus.center.1);
        let upper = raw_iris_focus::detect_upper_eyelid_points(
            &raw,
            width,
            height,
            sensor_origin.0,
            sensor_origin.1,
            &focus,
        );
        let lower = raw_iris_focus::detect_lower_eyelid_points(
            &raw,
            width,
            height,
            sensor_origin.0,
            sensor_origin.1,
            &focus,
        );

        let native_started = Instant::now();
        let native_prior = native_radius.begin_frame_controlled(now, scale_prediction, true, None);
        let native_unbounded = if seed_usable {
            raw_iris_focus::detect_outer_iris_boundary_between_eyelids_tracked(
                &raw,
                width,
                height,
                sensor_origin.0,
                sensor_origin.1,
                &focus,
                &upper,
                &lower,
                &mut native_tracker,
            )
        } else {
            raw_iris_focus::OuterIrisBoundary::default()
        };
        let native_diagnostics = native_tracker.diagnostics();
        let native_candidate = !native_unbounded.points.is_empty();
        let native_boundary = if native_prior.is_none_or(|prior| {
            prior.admits_ellipse(native_unbounded.major_radius, native_unbounded.minor_radius)
        }) {
            native_unbounded.clone()
        } else {
            raw_iris_focus::OuterIrisBoundary::default()
        };
        // Ask the compact current-frame RAW scorer whether this already-
        // fitted conic contains an observable bilateral
        // sclera -> iris -> pupil -> iris -> sclera road. A cropped or
        // one-sided road remains useful proposal evidence, but cannot publish
        // complete anatomy or teach the shared physical-radius posterior.
        let native_topology_probe = (!native_boundary.points.is_empty())
            .then(|| {
                score_driving_pose(
                    &raw,
                    width,
                    height,
                    DrivingAffinePose {
                        center: native_boundary.center,
                        major_radius: native_boundary.major_radius,
                        minor_radius: native_boundary.minor_radius,
                        angle: native_boundary.angle,
                    },
                )
            })
            .flatten();
        let native_pupil_horizon =
            native_pupil_horizon_evaluation(&raw, width, height, &native_boundary, &focus);
        let native_material_admissible =
            native_material_topology_admissible(native_topology_probe, native_pupil_horizon);
        let native_material_veto = native_topology_probe.is_some() && !native_material_admissible;
        let native_specular_containment = native_material_admissible
            .then(|| {
                native_specular_containment_evidence(
                    &raw,
                    width,
                    height,
                    DrivingAffinePose {
                        center: native_boundary.center,
                        major_radius: native_boundary.major_radius,
                        minor_radius: native_boundary.minor_radius,
                        angle: native_boundary.angle,
                    },
                )
            })
            .flatten();
        let native_specular_admissible = native_specular_containment_admissible(
            native_specular_containment,
            native_pupil_horizon.is_some(),
        );
        let native_scale_kinematically_supported = native_prior.is_none_or(|prior| {
            prior.admits_kinematically_supported_ellipse(
                native_boundary.major_radius,
                native_boundary.minor_radius,
            )
        });
        let native_strong_measurement = !partial_frame
            && native_material_admissible
            && native_specular_admissible
            && native_scale_kinematically_supported
            && native_meridian_strong_limbus_measurement(&native_boundary, native_diagnostics);
        // Match the live Native path: on a cold start, strong measurements
        // vote into the de-affined circular-radius posterior but remain
        // proposal-only until independent temporal consensus exists. Once a
        // prior exists, its final latest-strong kinematic decision is also the
        // publication decision; merely calling the shared gate and ignoring a
        // false result would let Native bypass the common size invariant.
        let native_radius_admitted = if native_strong_measurement {
            let confidence =
                (0.45 + 0.55 * native_diagnostics.analog_mean_certainty).clamp(0.0, 1.0);
            native_radius.observe_strong_ellipse_for_active_frame(
                now,
                native_boundary.major_radius,
                native_boundary.minor_radius,
                confidence,
            )
        } else {
            false
        };
        let native_admitted = native_prior.is_some()
            && native_strong_measurement
            && native_radius_admitted
            && !native_boundary.points.is_empty();
        let native_elapsed_ms = native_started.elapsed().as_secs_f64() * 1_000.0;
        native_summary.observe(
            native_candidate,
            native_admitted,
            native_admitted.then_some(native_boundary.center),
            native_admitted.then_some(
                native_boundary
                    .major_radius
                    .max(native_boundary.minor_radius),
            ),
            sensor_origin,
            native_elapsed_ms,
        );

        let cluster_started = Instant::now();
        let cluster_prior =
            cluster_radius.begin_frame_controlled(now, scale_prediction, true, None);
        let cluster_native_unbounded = if seed_usable {
            raw_iris_focus::detect_outer_iris_boundary_between_eyelids_tracked(
                &raw,
                width,
                height,
                sensor_origin.0,
                sensor_origin.1,
                &focus,
                &upper,
                &lower,
                &mut cluster_native_tracker,
            )
        } else {
            raw_iris_focus::OuterIrisBoundary::default()
        };
        let cluster_native_diagnostics = cluster_native_tracker.diagnostics();
        let cluster_native_topology_probe = (!cluster_native_unbounded.points.is_empty())
            .then(|| {
                score_driving_pose(
                    &raw,
                    width,
                    height,
                    DrivingAffinePose {
                        center: cluster_native_unbounded.center,
                        major_radius: cluster_native_unbounded.major_radius,
                        minor_radius: cluster_native_unbounded.minor_radius,
                        angle: cluster_native_unbounded.angle,
                    },
                )
            })
            .flatten();
        let cluster_native_material_veto =
            native_material_topology_decisive_veto(cluster_native_topology_probe);
        let cluster_native_cold_vote = cluster_prior.is_none()
            && !partial_frame
            && !cluster_native_material_veto
            && native_meridian_strong_limbus_measurement(
                &cluster_native_unbounded,
                cluster_native_diagnostics,
            );
        let cluster_native_cold_vote_recorded = if cluster_native_cold_vote {
            let confidence =
                (0.45 + 0.55 * cluster_native_diagnostics.analog_mean_certainty).clamp(0.0, 1.0);
            cluster_radius.observe_strong_ellipse_for_active_frame(
                now,
                cluster_native_unbounded.major_radius,
                cluster_native_unbounded.minor_radius,
                confidence,
            )
        } else {
            false
        };
        let cluster_native = if cluster_prior.is_none_or(|prior| {
            prior.admits_ellipse(
                cluster_native_unbounded.major_radius,
                cluster_native_unbounded.minor_radius,
            )
        }) {
            cluster_native_unbounded.clone()
        } else {
            raw_iris_focus::OuterIrisBoundary::default()
        };
        let cluster_partial = partial.filter(|partial| {
            cluster_prior.is_none_or(|prior| {
                prior.admits_ellipse(partial.major_radius, partial.minor_radius)
            })
        });
        let cluster_native_ready = !cluster_native.points.is_empty() && cluster_partial.is_none();
        let mut cluster_seed = boundary_seed(&cluster_native)
            .or_else(|| cluster_partial.map(partial_seed))
            .or_else(|| {
                cluster_prior.and_then(|prior| {
                    (!cluster_native_unbounded.points.is_empty()).then(|| {
                        let axis_ratio = (cluster_native_unbounded.minor_radius
                            / cluster_native_unbounded.major_radius.max(1.0))
                        .clamp(0.45, 1.0);
                        raw_motion_octrees::IrisEllipseSeed {
                            center: cluster_native_unbounded.center,
                            major_radius: prior.estimate_px,
                            minor_radius: prior.estimate_px * axis_ratio,
                            angle: cluster_native_unbounded.angle,
                        }
                    })
                })
            })
            .or_else(|| focus_seed(&focus));
        let cluster_overlay = cluster_motion.observe_with_iris_seed_at(
            &raw,
            width,
            height,
            sensor_origin.0,
            sensor_origin.1,
            timestamp_ns,
            None,
            true,
            cluster_seed,
        );
        // Promote motion identity before paying for full pupil-headed
        // multi-bank geometry. A rough current seed is sufficient to collect
        // native patches; only a cohesive current-frame layer may request the
        // expensive RAW solve. This prevents a stable glasses/skin cohort
        // from forcing a near-second cold search on every replay frame.
        let mut cluster_seed_geometry_measured = false;
        if temporal_feature_layer_ready_for_geometry(&cluster_overlay) {
            if let Some(seed) = cluster_seed {
                if let Some(measured) = measured_multibank_temporal_canny_seed(
                    &raw,
                    width,
                    height,
                    sensor_origin,
                    seed,
                    Some(&focus),
                    cluster_prior,
                ) {
                    cluster_seed = Some(measured);
                    cluster_seed_geometry_measured = true;
                }
            }
        }
        let (cluster_unbounded_hypothesis, cluster_diagnostics) = if cluster_seed_geometry_measured
        {
            let seed = cluster_seed.expect("measured cluster seed");
            let exact = raw_motion_octrees::feature_cluster_iris_hypothesis_from_measured_seed_with_diagnostics(
                    &cluster_overlay,
                    width,
                    height,
                    seed,
                );
            if let Some(exact_hypothesis) = exact.0 {
                (
                    Some(temporal_canny_select_center_closure(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        &cluster_overlay,
                        seed,
                        cluster_prior,
                        exact_hypothesis,
                    )),
                    exact.1,
                )
            } else {
                let (refined, diagnostics) =
                    raw_motion_octrees::feature_cluster_iris_hypothesis_with_diagnostics(
                        &cluster_overlay,
                        width,
                        height,
                        Some(seed),
                        0.0,
                    );
                let refined = refined.filter(|hypothesis| {
                    temporal_canny_refined_outer_geometry_admissible(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        seed,
                        hypothesis,
                        cluster_prior,
                    )
                });
                let refined = refined.or_else(|| {
                    temporal_canny_measured_center_closure(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        &cluster_overlay,
                        seed,
                        cluster_prior,
                    )
                });
                (refined, diagnostics)
            }
        } else {
            raw_motion_octrees::feature_cluster_iris_hypothesis_with_diagnostics(
                &cluster_overlay,
                width,
                height,
                cluster_seed,
                if cluster_native_ready { 0.06 } else { 0.0 },
            )
        };
        let cluster_candidate = cluster_unbounded_hypothesis.clone().filter(|hypothesis| {
            cluster_prior.is_none_or(|prior| {
                prior.admits_ellipse(hypothesis.major_radius, hypothesis.minor_radius)
            })
        });
        let cluster_texture = cluster_candidate.as_ref().map(|hypothesis| {
            iris_texture_evidence_for_motion_layer(
                raw_motion_octrees::IrisEllipseSeed {
                    center: hypothesis.center,
                    major_radius: hypothesis.major_radius,
                    minor_radius: hypothesis.minor_radius,
                    angle: hypothesis.angle,
                },
                &cluster_overlay,
                hypothesis.motion_layer,
            )
        });
        // Diagnose the semantic identity of the conic at its own geometry.
        // Temporal layer coherence can make a lid, brow, or glint conic very
        // stable, so motion and radius feasibility alone are not evidence that
        // the tracked material is the sclera/limbus/pupil road.
        let cluster_topology_probe = cluster_candidate.as_ref().and_then(|hypothesis| {
            score_driving_pose(
                &raw,
                width,
                height,
                DrivingAffinePose {
                    center: hypothesis.center,
                    major_radius: hypothesis.major_radius,
                    minor_radius: hypothesis.minor_radius,
                    angle: hypothesis.angle,
                },
            )
        });
        let cluster_semantic_eye = cluster_topology_probe.and_then(|hypothesis| {
            driving_semantic_eye_evidence(&raw, width, height, sensor_origin, hypothesis)
        });
        let cluster_semantic_eye_authorized =
            cluster_semantic_eye.is_some_and(|evidence| evidence.authorizes_cold_identity);
        let cluster_reflection_disk_evidence = cluster_candidate.as_ref().and_then(|hypothesis| {
            temporal_feature_reflection_disk_evidence(hypothesis, &cluster_overlay)
        });
        let cluster_layer_tracks = cluster_candidate
            .as_ref()
            .and_then(|hypothesis| cluster_overlay.layers.get(hypothesis.motion_layer))
            .map_or(0, |layer| layer.persistent_tracks);
        let cluster_semantic_assessment = temporal_feature_limbus_semantic_assessment(
            cluster_candidate.as_ref(),
            cluster_seed,
            cluster_prior,
            cluster_texture,
            cluster_topology_probe,
            cluster_partial,
            cluster_native_material_veto
                && !cluster_seed_geometry_measured
                && !cluster_semantic_eye_authorized,
            cluster_seed_geometry_measured || native_admitted || cluster_semantic_eye_authorized,
            cluster_reflection_disk_evidence,
            cluster_layer_tracks,
        );
        let cluster_layer_motion = cluster_candidate.as_ref().and_then(|hypothesis| {
            cluster_overlay
                .motions
                .get(hypothesis.motion_layer)
                .copied()
        });
        let cluster_center_assessment = cluster_center_gate.observe_frame(
            timestamp_ns,
            sensor_origin,
            (width, height),
            cluster_candidate.as_ref().map(|hypothesis| {
                (
                    hypothesis.center,
                    hypothesis.major_radius.max(hypothesis.minor_radius),
                )
            }),
            cluster_layer_motion,
            cluster_prior.is_some() && cluster_semantic_assessment.admissible,
        );
        let cluster_center_admissible =
            cluster_center_assessment.is_none_or(|assessment| assessment.admissible);
        // A coherent temporal layer is strong evidence, but its first conic
        // cannot unilaterally define physical iris scale.  Cold candidates
        // vote into the same three-observation radius consensus as Native;
        // publication starts only on a later frame carrying that frozen prior.
        let cluster_radius_admitted = if !cluster_native_cold_vote
            && cluster_center_admissible
            && cluster_semantic_assessment.admissible
        {
            cluster_candidate.as_ref().is_some_and(|hypothesis| {
                cluster_radius.observe_strong_ellipse_for_active_frame(
                    now,
                    hypothesis.major_radius,
                    hypothesis.minor_radius,
                    hypothesis.score,
                )
            })
        } else {
            cluster_native_cold_vote_recorded
                && cluster_radius.established_dynamic_radius_px().is_some()
        };
        let cluster_radius_authoritative =
            cluster_prior.is_some() || cluster_radius.established_dynamic_radius_px().is_some();
        let cluster_hypothesis = (cluster_radius_authoritative
            && cluster_center_admissible
            && cluster_semantic_assessment.admissible
            && cluster_radius_admitted)
            .then(|| cluster_candidate.clone())
            .flatten();
        let cluster_elapsed_ms = cluster_started.elapsed().as_secs_f64() * 1_000.0;
        cluster_summary.observe(
            cluster_candidate.is_some(),
            cluster_hypothesis.is_some(),
            cluster_hypothesis
                .as_ref()
                .map(|hypothesis| hypothesis.center),
            cluster_hypothesis
                .as_ref()
                .map(|hypothesis| hypothesis.major_radius.max(hypothesis.minor_radius)),
            sensor_origin,
            cluster_elapsed_ms,
        );

        let driving_started = Instant::now();
        let driving_native = if run_driving && seed_usable {
            raw_iris_focus::detect_outer_iris_boundary_between_eyelids_tracked_for_driving(
                &raw,
                width,
                height,
                sensor_origin.0,
                sensor_origin.1,
                &focus,
                &upper,
                &lower,
                &mut driving_native_tracker,
            )
        } else {
            raw_iris_focus::OuterIrisBoundary::default()
        };
        let driving_native_diagnostics = driving_native_tracker.diagnostics();
        let driving_native_ready = !driving_native.points.is_empty() && !partial_frame;
        let trusted = driving
            .pose
            .zip(driving.last_scored)
            .and_then(|(pose, mut hypothesis)| {
                (driving.admission_streak >= DRIVING_ADMISSION_FRAMES).then(|| {
                    hypothesis.pose = pose;
                    hypothesis
                })
            });
        let trusted_seed = trusted.map(DrivingHypothesis::iris_seed);
        let driving_seed = boundary_seed(&driving_native)
            .or_else(|| partial.map(partial_seed))
            .or(trusted_seed)
            .or_else(|| focus_seed(&focus));
        let driving_overlay = if run_driving {
            let mut overlay = driving_canny.observe(
                &raw,
                width,
                height,
                sensor_origin.0,
                sensor_origin.1,
                driving_seed,
            );
            driving_canny.fuse_global_similarity_at(
                &mut overlay,
                global_similarity,
                timestamp_ns,
                width,
                height,
                sensor_origin.0,
                sensor_origin.1,
            );
            overlay
        } else {
            raw_motion_octrees::MotionOctreeOverlay::default()
        };
        let partial_texture =
            partial.map(|partial| roi_truncated_iris_texture_evidence(partial, &driving_overlay));
        let complete_texture = driving_native_ready
            .then(|| driving_seed.map(|seed| iris_texture_evidence(seed, &driving_overlay)))
            .flatten();
        let trusted_texture =
            trusted_seed.map(|seed| iris_texture_evidence(seed, &driving_overlay));
        let temporal_canny_seed =
            driving_trusted_canny_continuation_seed(trusted_seed, trusted_texture);
        let complete_ready = driving_complete_frame_seed_ready(
            driving_native_ready,
            driving_native_diagnostics.accepted,
            focus_anatomy,
            complete_texture,
        );
        let driving_partial = partial.zip(partial_texture).and_then(|(partial, texture)| {
            driving_partial_frame_seed(partial, texture, width, height)
        });
        let direct_ready = complete_ready || driving_partial.is_some();
        let cold_identity_authorized = complete_ready
            || partial_texture
                .is_some_and(|texture| texture.authorizes_cold_partial_topology_probe());
        let fallback = if !driving_native.points.is_empty() {
            Some((
                driving_native.center,
                (driving_native.major_radius * driving_native.minor_radius).sqrt(),
            ))
        } else if let Some(partial) = partial {
            Some((
                partial.center,
                (partial.major_radius * partial.minor_radius).sqrt(),
            ))
        } else if focus.eye_basin_valid {
            Some((focus.center, focus.radius))
        } else {
            None
        };
        let driving_input = driving_partial
            .clone()
            .unwrap_or_else(|| driving_native.clone());
        let driving_hypothesis = if run_driving {
            driving.observe_with_temporal_canny_seed_and_scale_prediction(
                now,
                sensor_origin,
                &raw,
                width,
                height,
                direct_ready,
                cold_identity_authorized,
                &driving_input,
                fallback,
                Some(&focus),
                temporal_canny_seed,
                scale_prediction,
                pupil_center_saccade_motion_supported(&driving_overlay),
                true,
                None,
            )
        } else {
            None
        };
        // A provisional strong vote may survive two missing frames solely for
        // cold-start consensus. Do not draw that stale conic as a measurement
        // on the intervening RAW frames; only a proven pose may be shown held.
        let driving_candidate = driving.last_candidate.or_else(|| {
            driving
                .pose
                .is_some()
                .then_some(driving.last_scored)
                .flatten()
        });
        let driving_fit = driving.last_fit_assessment;
        let driving_radius_prior = driving
            .limbus_radius
            .lock()
            .ok()
            .and_then(|tracker| tracker.active_frame_prior());
        let driving_elapsed_ms = driving_started.elapsed().as_secs_f64() * 1_000.0;
        driving_summary.observe(
            driving_candidate.is_some(),
            driving_hypothesis.is_some(),
            driving_hypothesis.map(|hypothesis| hypothesis.pose.center),
            driving_hypothesis.map(|hypothesis| hypothesis.pose.major_radius),
            sensor_origin,
            driving_elapsed_ms,
        );
        let driving_candidate_pupil_affine_observation = driving_candidate.and_then(|hypothesis| {
            pupil_affine_temporal_observation(
                timestamp_ns,
                sensor_origin,
                (width, height),
                hypothesis.pose,
                hypothesis.pupil_boundary_center(),
                hypothesis.pupil_projected_area_radius_px?,
            )
        });
        let driving_candidate_pupil_affine_temporal_json = driving_candidate_pupil_affine_temporal
            .observe(
                false,
                driving_candidate_pupil_affine_observation,
                global_similarity,
            );
        let driving_published_pupil_affine_observation =
            driving_hypothesis.and_then(|hypothesis| {
                pupil_affine_temporal_observation(
                    timestamp_ns,
                    sensor_origin,
                    (width, height),
                    hypothesis.pose,
                    hypothesis.pupil_boundary_center(),
                    hypothesis.pupil_projected_area_radius_px?,
                )
            });
        let driving_published_pupil_affine_temporal_json = driving_published_pupil_affine_temporal
            .observe(
                driving_hypothesis.is_some(),
                driving_published_pupil_affine_observation,
                global_similarity,
            );

        // Exercise the shared live pupil path against consecutive RAW frames.
        // A pupil search is downstream of *published eye anatomy*: a tempting
        // Driving/native diagnostic conic is not an iris and must never create
        // pupil state.  This distinction is essential for smooth circular
        // hard negatives such as the glasses temple/front-frame junction in
        // the optically clocked capture.  That crop had no eye basin, no focus
        // anatomy, and only one persistent iris-texture feature; Driving
        // correctly withheld it, while the old replay silently substituted
        // `native_unbounded` and drew a very confident fictitious pupil.
        //
        // SAM31 remains absent here: it is an offline oracle, never a
        // Racer/runtime dependency.
        let common_pupil_outer = select_common_pupil_outer_for_replay(
            driving_hypothesis.map(|hypothesis| hypothesis.outer_boundary()),
            &native_boundary,
            native_admitted,
        );
        let common_pupil_outer_source = common_pupil_outer.as_ref().map(|(_, source)| *source);
        let common_pupil_projection = common_pupil_outer.as_ref().and_then(|(boundary, _)| {
            PupilProjectionReference::from_outer(boundary, PupilProjectionSource::SelectedIris)
        });
        let common_pupil_proposal = focus.pupil_hint;
        let common_pupil_prediction = common_pupil_center.begin_frame(
            now,
            sensor_origin,
            (width, height),
            common_pupil_projection,
            common_pupil_proposal,
            &cluster_overlay,
        );
        let common_pupil_support = common_pupil_size.begin_frame(
            now,
            common_pupil_prediction.map(|prediction| prediction.center),
            common_pupil_projection,
            true,
            DEFAULT_PUPIL_RADIUS_LOWER_FRACTION,
            DEFAULT_PUPIL_RADIUS_UPPER_FRACTION,
        );
        let common_pupil_condition = common_pupil_projection.map_or_default(|projection| {
            PupilEvidenceCondition::fully_reliable(
                projection.fronto_parallel_limbus_radius_px.value(),
            )
        });
        let common_inner_radius_prior = inner_radius_prior_from_support_conditioned(
            common_pupil_support,
            common_pupil_condition,
        );
        let mut common_pupil_boundary = common_pupil_prediction
            .zip(common_pupil_projection)
            .map(|(prediction, projection)| {
                solve_pupil_boundary_from_temporal_state(
                    &mut common_pupil_center,
                    now,
                    sensor_origin,
                    &raw,
                    width,
                    height,
                    &focus,
                    projection,
                    prediction,
                    common_pupil_proposal,
                    &cluster_overlay,
                    common_pupil_support,
                    common_inner_radius_prior,
                    common_pupil_condition.raw_solver_condition(),
                )
            })
            .unwrap_or_default();
        let common_pupil_confidence =
            pupil_boundary_confidence(&common_pupil_boundary, common_pupil_support);
        let common_pupil_admission = stabilize_and_observe_pupil_size(
            &mut common_pupil_size,
            &mut common_pupil_radius_limiter,
            now,
            &mut common_pupil_boundary,
            common_pupil_support,
            common_pupil_confidence,
            true,
            common_pupil_condition,
        );
        let common_pupil_pose = common_pupil_projection.map(|projection| {
            let major_radius = projection.fronto_parallel_limbus_radius_px.value();
            DrivingAffinePose {
                center: projection.center,
                major_radius,
                minor_radius: major_radius * projection.minor_to_major,
                angle: projection.angle,
            }
        });
        let common_pupil_polar_diagnostics =
            common_pupil_pose.map_or_else(PupilPolarCoSolveDiagnostics::default, |pose| {
                let hard_ratio_bounds = common_pupil_support.map_or(
                    (
                        DEFAULT_PUPIL_RADIUS_LOWER_FRACTION,
                        DEFAULT_PUPIL_RADIUS_UPPER_FRACTION,
                    ),
                    |support| {
                        let reference = support
                            .reference_limbus_fronto_parallel_radius_px
                            .value()
                            .max(1.0);
                        (
                            support.lower_fronto_parallel_radius_px / reference,
                            support.upper_fronto_parallel_radius_px / reference,
                        )
                    },
                );
                common_pupil_polar_cosolver.observe(
                    now,
                    sensor_origin,
                    &common_pupil_boundary,
                    pose,
                    hard_ratio_bounds,
                )
            });
        let common_pupil_candidate_json = common_pupil_pose.map_or(Value::Null, |pose| {
            driving_inner_boundary_json(common_pupil_boundary.clone(), pose)
        });
        let common_pupil_json = common_pupil_admission
            .current_boundary_publishable()
            .then(|| common_pupil_candidate_json.clone())
            .unwrap_or(Value::Null);
        let common_pupil_center_diagnostics = common_pupil_center.diagnostics();
        let center_consensus_gate = common_pupil_projection.map_or(12.0, |projection| {
            (projection.fronto_parallel_limbus_radius_px.value() * 0.10).clamp(9.0, 17.0)
        });
        let agree = |left: Option<(f64, f64)>, right: Option<(f64, f64)>| {
            left.zip(right).and_then(|(left, right)| {
                ((left.0 - right.0).hypot(left.1 - right.1) <= center_consensus_gate).then_some((
                    0.58 * left.0 + 0.42 * right.0,
                    0.58 * left.1 + 0.42 * right.1,
                ))
            })
        };
        let published_boundary =
            (!common_pupil_boundary.points.is_empty()).then_some(common_pupil_boundary.center);
        let rough_boundary_consensus = common_pupil_admission
            .raw_diameter_qualified
            .then(|| agree(common_pupil_proposal, published_boundary))
            .flatten();
        let rough_published_consensus = agree(
            common_pupil_proposal,
            common_pupil_center_diagnostics.published_center,
        );
        let clock_centers = [
            common_pupil_proposal,
            common_pupil_center_diagnostics.predicted_center,
            common_pupil_center_diagnostics.measured_center,
            common_pupil_center_diagnostics.published_center,
            published_boundary,
            rough_boundary_consensus,
            rough_published_consensus,
        ];
        let clock_canonical_centers = clock_centers.map(|center| {
            common_pupil_projection
                .zip(center)
                .and_then(|(projection, center)| {
                    pupil_projection_canonical_point(projection, center)
                })
        });
        clock_pupil_samples.push(ClockPupilSample {
            timestamp_ns,
            target_gaze_tangent: stimulus_pose.map(|pose| pose.gaze_tangent),
            target_moving: stimulus_pose.is_some_and(|pose| pose.moving),
            sensor_origin,
            centers: clock_centers,
            canonical_centers: clock_canonical_centers,
        });
        let pupil_affine_observation = common_pupil_admission
            .current_boundary_publishable()
            .then_some(common_pupil_pose)
            .flatten()
            .and_then(|pose| {
                let pupil_projected_area_radius_px = (common_pupil_boundary.major_radius
                    * common_pupil_boundary.minor_radius)
                    .sqrt();
                pupil_affine_temporal_observation(
                    timestamp_ns,
                    sensor_origin,
                    (width, height),
                    pose,
                    common_pupil_boundary.center,
                    pupil_projected_area_radius_px,
                )
            });
        let pupil_affine_temporal_json = pupil_affine_temporal.observe(
            common_pupil_admission.raw_diameter_qualified,
            pupil_affine_observation,
            global_similarity,
        );

        frames.push(json!({
            "index": local_index,
            "source_record_index": start + local_index,
            "sequence": integer(record, "sequence")?,
            "timestamp_ns": timestamp_ns,
            "sensor_origin": sensor_origin,
            "width": width,
            "height": height,
            "stride": stride,
            "source_offset": offset,
            "source_length": length,
            "optical_screen_supervision": stimulus_pose.map(|pose| json!({
                "inference_input": false,
                "identity_source": "reflected spatial code plus sensor timestamp",
                "host_timestamp_used": false,
                "code_index": pose.code_index,
                "elapsed_seconds": pose.elapsed_seconds,
                "target_center_normalized": pose.center_normalized,
                "target_velocity_normalized_per_second": pose.velocity_normalized_per_second,
                "target_gaze_tangent": pose.gaze_tangent,
                "target_gaze_tangent_velocity_per_second": pose.gaze_tangent_velocity_per_second,
                "moving": pose.moving,
                "clock_carrier": pose.clock_carrier_center.map(|center| json!({
                    "center": center,
                    "extent": pose.clock_carrier_extent,
                    "score": pose.clock_carrier_score,
                    "semantic_role": "whole-image optical clock carrier; temporal identity only, not a corneal glint or anatomical coordinate",
                })),
            })),
            "focus": {
                "score": focus.score,
                "eye_basin_valid": focus.eye_basin_valid,
                "anatomy_valid": focus_anatomy,
                "center": focus.center,
                "radius": focus.radius,
                "points": focus.points.len(),
                "pupil_hint": focus.pupil_hint,
                "pupil_hint_radius": focus.pupil_hint_radius,
                "pupil_hint_score": focus.pupil_hint_score,
                "roi_truncated_limbus": partial_json(partial),
            },
            "shared_global_scale": {
                "reliable": global_similarity.reliable,
                "stable_frames": global_similarity.stable_frames,
                "motion_center_sensor": global_similarity.motion_center_sensor,
                "spatial_span": global_similarity.spatial_span,
                "occupied_quadrants": global_similarity.occupied_quadrants,
                "motion_support": global_similarity.motion.support,
                "motion_residual": global_similarity.motion.residual,
                "translation": global_similarity.motion.translation,
                "rotation": global_similarity.motion.rotation,
                "scale_delta": global_similarity.motion.scale_delta,
                "candidate_motion_support": global_similarity.candidate_motion.support,
                "candidate_matches": global_similarity.candidate_matches,
                "candidate_motion_residual": global_similarity.candidate_motion.residual,
                "candidate_translation": global_similarity.candidate_motion.translation,
                "candidate_rotation": global_similarity.candidate_motion.rotation,
                "candidate_scale_delta": global_similarity.candidate_motion.scale_delta,
                "prediction": scale_prediction.map(|prediction| json!({
                    "scale_ratio": prediction.scale_ratio,
                    "fractional_uncertainty": prediction.fractional_uncertainty,
                    "source": format!("{:?}", prediction.source),
                })),
            },
            "common_pupil": {
                "outer_source": common_pupil_outer_source,
                "proposal_center": common_pupil_proposal,
                "boundary": common_pupil_json,
                "boundary_candidate": common_pupil_candidate_json,
                "boundary_publishable": common_pupil_admission.current_boundary_publishable(),
                "sparse_polar_cosolve": pupil_polar_cosolve_json(&common_pupil_polar_diagnostics),
                "center_track": pupil_center_track_json(common_pupil_center_diagnostics),
                "confidence": common_pupil_confidence,
                "size_rate_limited": common_pupil_admission.rate_limited,
                "raw_diameter_qualified": common_pupil_admission.raw_diameter_qualified,
                "size_trajectory_updated": common_pupil_admission.trajectory_updated,
                "size_trained_posterior": common_pupil_admission.trained_posterior,
                "affine_temporal_consistency": pupil_affine_temporal_json,
            },
            "native_meridian": {
                "candidate": boundary_json(&native_unbounded),
                "published": native_admitted.then(|| boundary_json(&native_boundary)),
                "driving_material_probe": driving_json(native_topology_probe),
                "driving_material_admissible": native_material_admissible,
                "driving_material_veto": native_material_veto,
                "specular_containment": native_specular_containment.map(|evidence| json!({
                    "contained_z": evidence.contained_z,
                    "contained_contrast": evidence.contained_contrast,
                    "external_z": evidence.external_z,
                    "external_contrast": evidence.external_contrast,
                    "externally_dominant": evidence.externally_dominant(),
                    "contained_cohesive_peak": evidence.has_contained_cohesive_peak(),
                    "admissible": native_specular_admissible,
                })),
                "scale_kinematically_supported": native_scale_kinematically_supported,
                "pupil_horizon": native_pupil_horizon_json(&raw, width, height, &native_unbounded, &focus),
                "diagnostics": diagnostics_json(native_diagnostics),
                "radius_prior": native_prior.map(|prior| json!({
                    "estimate_px": prior.estimate_px,
                    "minimum_px": prior.minimum_px,
                    "maximum_px": prior.maximum_px,
                    "source": format!("{:?}", prior.source),
                })),
                "elapsed_ms": native_elapsed_ms,
            },
            "two_d_features": {
                "unbounded_candidate": cluster_json(cluster_unbounded_hypothesis.as_ref()),
                "candidate": cluster_json(cluster_candidate.as_ref()),
                "published": cluster_json(cluster_hypothesis.as_ref()),
                "material_topology_probe": driving_json(cluster_topology_probe),
                "iris_texture": texture_json(cluster_texture),
                "center_kinematics": temporal_feature_center_assessment_json(cluster_center_assessment),
                "semantic_admission": temporal_feature_semantic_assessment_json(cluster_semantic_assessment),
                "semantic_eye": driving_semantic_eye_json(cluster_semantic_eye),
                "diagnostics": feature_cluster_diagnostics_json(cluster_diagnostics),
                "radius_prior": cluster_prior.map(|prior| json!({
                    "estimate_px": prior.estimate_px,
                    "minimum_px": prior.minimum_px,
                    "maximum_px": prior.maximum_px,
                    "source": format!("{:?}", prior.source),
                })),
                "native_seed": cluster_seed.map(|seed| ellipse_json(seed.center, seed.major_radius, seed.minor_radius, seed.angle)),
                "native_seed_material_probe": driving_json(cluster_native_topology_probe),
                "native_seed_material_veto": cluster_native_material_veto,
                "stable_layers": cluster_overlay.active_objects,
                "coherent_temporal_hold": raw_motion_octrees::coherent_temporal_feature_hold(&cluster_overlay),
                "matched_features": cluster_overlay.matched_features,
                "canny_edges": cluster_overlay.edges.len(),
                "temporal_trails": cluster_overlay.trails.len(),
                "layers": motion_layers_json(&cluster_overlay),
                "semantic_iris": cluster_overlay.semantic_iris.map(|ellipse| ellipse_json(
                    ellipse.center,
                    f64::from(ellipse.major_radius),
                    f64::from(ellipse.minor_radius),
                    f64::from(ellipse.angle),
                )),
                "elapsed_ms": cluster_elapsed_ms,
            },
            "driving": {
                "published": driving_json(driving_hypothesis),
                "candidate": driving_json(driving_candidate),
                "published_pupil_affine_temporal_consistency": driving_published_pupil_affine_temporal_json,
                "candidate_pupil_affine_temporal_consistency": driving_candidate_pupil_affine_temporal_json,
                "pupil_center_motion_gate": pupil_center_motion_gate_json(&driving_overlay),
                "candidate_limbus_material": driving_limbus_material_json(
                    DrivingRawMaterialView::new(&raw, width, height, sensor_origin)
                        .and_then(|view| driving_candidate.and_then(|hypothesis| {
                            driving_limbus_material_evidence(&view, hypothesis.pose)
                        })),
                ),
                "candidate_multibank_limbus": driving_multibank_limbus_json(
                    driving_candidate.and_then(|hypothesis| {
                        driving_multibank_limbus_evidence(
                            &raw,
                            width,
                            height,
                            hypothesis.pose,
                        )
                    }),
                ),
                "candidate_semantic_eye": driving_semantic_eye_json(
                    driving_candidate.and_then(|hypothesis| {
                        driving_semantic_eye_evidence(
                            &raw,
                            width,
                            height,
                            sensor_origin,
                            hypothesis,
                        )
                    }),
                ),
                "eyelid_scene": labeled_eyelid_scene_json(
                    &raw,
                    width,
                    height,
                    driving_candidate.map(|hypothesis| hypothesis.pose),
                    driving_candidate.map(DrivingHypothesis::pupil_boundary_center),
                ),
                "pupil_selection_variants": pupil_selection_audit.then(|| {
                    driving_pupil_selection_variants_json(
                        &raw,
                        width,
                        height,
                        sensor_origin,
                        driving_candidate,
                    )
                }).unwrap_or(Value::Null),
                "native_seed": boundary_json(&driving_native),
                "native_diagnostics": diagnostics_json(driving_native_diagnostics),
                "direct_proposal_ready": direct_ready,
                "cold_identity_authorized": cold_identity_authorized,
                "complete_proposal_ready": complete_ready,
                "temporal_canny_continuation_ready": temporal_canny_seed.is_some(),
                "used_temporal_continuation": driving.last_used_temporal_continuation,
                "admission_streak": driving.admission_streak,
                "fit_assessment": fit_assessment_json(driving_fit),
                "pupil_ratio_trajectory": {
                    "proven": driving.pupil_ratio_proven,
                    "published_history": driving.pupil_ratio_published_log_history.len(),
                    "temporally_constrained": driving.last_pupil_margin_temporally_constrained,
                    "native_measurement_ratio": driving.last_pupil_margin_native_measurement_ratio,
                    "active_support": driving.active_pupil_ratio_support.map(|support| json!({
                        "estimate": support.estimate,
                        "minimum": support.minimum,
                        "maximum": support.maximum,
                    })),
                },
                "pupil_center_orbital_fit": driving.last_pupil_center_orbital_fit.map(|fit| json!({
                    "center": fit.center,
                    "score": fit.score,
                    "ring_transition": fit.ring_transition,
                    "ring_coverage": fit.ring_coverage,
                    "opposing_support": fit.opposing_support,
                    "interior_void": fit.interior_void,
                    "broad_dark_step": fit.broad_dark_step,
                    "broad_dark_support": fit.broad_dark_support,
                    "canonical_radius": fit.canonical_radius,
                    "evaluated_centers": fit.evaluated_centers,
                })),
                "pupil_center_prior_assessment": driving.last_pupil_center_prior_assessment.map(|assessment| json!({
                    "center_disagreement_limbus_fraction": assessment.center_disagreement_limbus_fraction,
                    "upward_disagreement_limbus_fraction": assessment.upward_disagreement_limbus_fraction,
                    "lower_lid_visibility_contradiction": assessment.lower_lid_visibility_contradiction,
                    "incumbent_outvote_margin": assessment.incumbent_outvote_margin,
                })),
                "pupil_center_temporally_imputed": driving.last_pupil_center_temporally_imputed,
                "radius_prior": driving_radius_prior.map(|prior| json!({
                    "estimate_px": prior.estimate_px,
                    "minimum_px": prior.minimum_px,
                    "maximum_px": prior.maximum_px,
                    "source": format!("{:?}", prior.source),
                })),
                "partial_texture": texture_json(partial_texture),
                "complete_texture": texture_json(complete_texture),
                "trusted_texture": texture_json(trusted_texture),
                "bounded_canny_edges": driving_overlay.edges.len(),
                "bounded_canny_tracks": driving_overlay.trails.len(),
                "elapsed_ms": driving_elapsed_ms,
            },
        }));

        if (local_index + 1) % 250 == 0 || local_index + 1 == records.len() {
            eprintln!(
                "offline segmentation replay {}/{}",
                local_index + 1,
                records.len()
            );
        }
    }

    let report = json!({
        "schema": "buttercup-production-native-segmentation-replay-v1",
        "source": {
            "index": fs::canonicalize(&index_path).unwrap_or(index_path),
            "stream": fs::canonicalize(&stream_path).unwrap_or(stream_path),
            "label": label,
            "scope": replay_scope,
            "start": start,
            "count": records.len(),
            "pixel_format": "RAW10_LE40_1X1",
            "lossless_raw_source_by_offset": true,
            "downsampled_for_inference": false,
            "optical_screen_supervision": optical_stimulus.as_ref().map(|track| json!({
                "session_id": track.session_id,
                "stream": track.stream,
                "recovered_clock_frames": track.recovered_frames(),
                "joined_replay_frames": optical_stimulus_frames,
                "host_timestamp_used": false,
                "used_as_inference_input": false,
                "screen_manifest": screen_manifest_path,
                "clock_decode": optical_clock_path,
            })),
        },
        "algorithm": {
            "native_meridian": "production stateful native meridian detector + analog illumination-independent edge-force refinement + shared central-camera projected-limbus envelope + common fronto-parallel radius support",
            "two_d_features": "production full-resolution temporal RAW patch/Canny layers, seeded by the independently fitted native meridian conic + shared central-camera projected-limbus envelope + common fronto-parallel radius support",
            "driving": "production bounded native RAW iris-Canny bank + affine-polar texture gate + synchronous stateful 3D Driving topology laps/admission + shared central-camera projected-limbus envelope + common fronto-parallel radius support",
            "projection_envelope": {
                "model": "weak-perspective projection of a physical circular limbus",
                "focal_length_px": [
                    raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.minimum_focal_length_px,
                    raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.maximum_focal_length_px,
                ],
                "maximum_pixel_aspect_error": raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.maximum_pixel_aspect_error,
                "maximum_local_metric_anisotropy": raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.maximum_local_metric_anisotropy,
                "maximum_anatomical_surface_tilt_degrees": raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.maximum_anatomical_surface_tilt_radians.to_degrees(),
                "uncalibrated_central_ray_slack_degrees": raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.uncalibrated_central_ray_slack_radians.to_degrees(),
                "maximum_limbus_half_angle_degrees": raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.maximum_limbus_half_angle_radians.to_degrees(),
                "absolute_minimum_minor_to_major": raw_iris_focus::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.absolute_minimum_minor_to_major,
                "formal_calibration": false,
            },
        },
        "summary": {
            "frames": records.len(),
            "native_meridian": native_summary.json(records.len()),
            "two_d_features": cluster_summary.json(records.len()),
            "driving": driving_summary.json(records.len()),
            "driving_candidate_post_affine_pupil_temporal_consistency": driving_candidate_pupil_affine_temporal.json(
                records.len(),
                "provisional current-frame Driving candidates carrying measured pupil-area geometry; never publication",
            ),
            "driving_published_post_affine_pupil_temporal_consistency": driving_published_pupil_affine_temporal.json(
                records.len(),
                "published Driving pupil presentation; imputed centers are presentation-only and never detector evidence",
            ),
            "post_affine_pupil_temporal_consistency": pupil_affine_temporal.json(
                records.len(),
                "publishable common pupil boundaries only",
            ),
            "optically_supervised_frames": optical_stimulus_frames,
            "optical_pupil_motion": optical_stimulus
                .as_ref()
                .map(|_| clock_pupil_supervision_summary(&clock_pupil_samples)),
            "optical_target_motion_validation": optical_stimulus
                .as_ref()
                .map(|_| target_motion_supervision_summary(&clock_pupil_samples)),
        },
        "frames": frames,
    });
    fs::write(
        &output_path,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", output_path.display()))?;
    println!("{}", output_path.display());
    Ok(())
}
