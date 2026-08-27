//! Coupled temporal kinematics for the general/cyan and pupil/green layers.
//!
//! The cyan layer defines a material reference frame. Green-after-cyan motion
//! still supplies a *2-D relative-motion fixed point*, but that algebraic
//! point is no longer treated as the anatomical globe pivot by itself. A
//! separate projected-globe state keeps a slowly varying pivot, a bounded
//! translation nuisance term, and fixation/saccade/occlusion regimes. Direct
//! current-frame iris geometry must corroborate the fixed point before the
//! projected globe state becomes publishable.

use super::{MotionLayerStatus, SimilarityMotion, GENERAL_LAYER, PUPIL_LAYER};
use std::collections::VecDeque;

const MIN_LAYER_SUPPORT: usize = 3;
const MAX_CONTIGUOUS_DT_NS: u64 = 750_000_000;
const MIN_CONTIGUOUS_DT_NS: u64 = 2_000_000;
const CENTER_INFORMATION_DECAY: f64 = 0.997;
const CYAN_CENTER_INFORMATION_DECAY: f64 = 0.86;
const MIN_CENTER_EXCITATION: f64 = 2.5e-5;
const CENTER_LOCK_MIN_SAMPLES: u32 = 10;
const GLOBE_TRANSLATION_LIMIT_RADII: f64 = 0.12;
const GLOBE_FIXED_POINT_GATE_RADII: f64 = 0.65;
// Current limbus centers retain a few pixels of signed-Canny quantization and
// partial-lid jitter. Treat sub-6%-of-radius innovations as fixation noise;
// a saccade must clear twice that bound or be left explicitly uncertain.
const GLOBE_FIXATION_STEP_RADII: f64 = 0.060;
const GLOBE_SACCADE_STEP_RADII: f64 = 0.120;
// A directly anatomy- and motion-corroborated keyframe may be transported
// across a short interval in which the current conic is present but its edge
// support falls below the anatomy-authority gate.  This is deliberately both
// time- and frame-bounded: a replay at 10 Hz and a live stream at 100 Hz must
// receive comparable protection without permitting an indefinitely carried
// semantic ellipse to become evidence for itself.
const GLOBE_TRUSTED_BRIDGE_SECONDS: f64 = 0.75;
const GLOBE_TRUSTED_BRIDGE_FRAMES: u16 = 12;
const GLOBE_BRIDGE_CENTER_GATE_RADII: f64 = 0.16;
const GLOBE_BRIDGE_RADIUS_GATE_FRACTION: f64 = 0.08;

/// Current-frame projected iris geometry supplied independently of the
/// green-after-cyan fixed-point equation. Coordinates are absolute sensor
/// pixels so a moving ROI cannot masquerade as eye motion.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedIrisGeometry {
    pub center: [f64; 2],
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle_rad: f64,
    pub confidence: f64,
    /// True only when current RAW anatomy (a measured seed or sufficient
    /// signed-Canny limbus support) authorizes this conic. A carried semantic
    /// ellipse may still be exported, but it cannot update the globe state.
    pub anatomy_authorized: bool,
}

impl ProjectedIrisGeometry {
    fn area_radius(self) -> f64 {
        (self.major_radius * self.minor_radius).max(0.0).sqrt()
    }

    fn valid(self) -> bool {
        self.center.into_iter().all(f64::is_finite)
            && self.major_radius.is_finite()
            && self.minor_radius.is_finite()
            && self.angle_rad.is_finite()
            && self.confidence.is_finite()
            && self.major_radius >= self.minor_radius
            && self.minor_radius >= 8.0
            && self.major_radius <= self.minor_radius * 2.10
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlobeMotionRegime {
    #[default]
    Unobserved,
    Fixation,
    Saccade,
    Uncertain,
    Occluded,
}

impl GlobeMotionRegime {
    pub fn label(self) -> &'static str {
        match self {
            Self::Unobserved => "unobserved",
            Self::Fixation => "fixation",
            Self::Saccade => "saccade",
            Self::Uncertain => "uncertain",
            Self::Occluded => "occluded",
        }
    }
}

/// A head-material-frame globe state. `projected_pivot` is a filtered latent
/// anatomical hypothesis; `projected_pole` is the current iris-center vector
/// from that pivot. Neither is publishable until current anatomy and the
/// independent relative-motion equation agree.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProjectedGlobePoseStatus {
    pub projected_pivot: Option<[f32; 2]>,
    pub projected_pole: Option<[f32; 2]>,
    pub relative_translation_px: [f32; 2],
    /// Correction required because the nominal cyan material frame transported
    /// the latent pivot implausibly far from current eye anatomy. This is head-
    /// frame uncertainty, not claimed globe translation.
    pub material_frame_correction_px: [f32; 2],
    pub sigma_px: f32,
    pub confidence: f32,
    pub anatomy_confidence: f32,
    pub anatomy_authorized: bool,
    pub fixed_point_corroborated: bool,
    pub anatomy_disagreement_radii: f32,
    pub fixed_point_disagreement_radii: f32,
    pub regime: GlobeMotionRegime,
    pub fixation_frames: u16,
    pub corroborated_frames: u16,
    pub occluded_frames: u16,
    /// True only while a previously direct, corroborated keyframe is being
    /// transported through a weak-current-anatomy interval. It is never set
    /// by the relative-motion fixed point alone.
    pub temporally_bridged: bool,
    pub bridge_age_ms: f32,
    pub bridge_prediction_disagreement_radii: f32,
    pub accepted_updates: u32,
    pub rejected_updates: u32,
    pub publishable: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KinematicDerivatives {
    /// A virtual material point transported by this layer's fitted transform.
    /// Unlike a raw feature centroid, it does not jump when tracks enter or
    /// leave the population.
    pub reference_point: [f32; 2],
    pub velocity_px_s: [f32; 2],
    pub acceleration_px_s2: [f32; 2],
    pub jerk_px_s3: [f32; 2],
    pub angle_rad: f32,
    pub angular_velocity_rad_s: f32,
    pub angular_acceleration_rad_s2: f32,
    pub angular_jerk_rad_s3: f32,
    pub speed_px_s: f32,
    pub fit_residual_px: f32,
    pub angular_fit_residual_rad: f32,
    pub confidence: f32,
    pub samples: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RotationCenterStatus {
    /// Current absolute sensor-space projection.  `MotionOctreeOverlay`
    /// translates this into ROI-local coordinates for display and telemetry.
    pub projected_center: Option<[f32; 2]>,
    pub transport_velocity_px_s: [f32; 2],
    pub sigma_px: f32,
    pub confidence: f32,
    pub constraint_residual_px: f32,
    /// Change caused by admitting the newest constraint, excluding transport
    /// by the cyan material frame.
    pub estimate_revision_px: f32,
    /// Velocity of estimator revision relative to the cyan-transported
    /// center. A mature rigid attachment should keep this near zero.
    pub relative_slip_velocity_px_s: [f32; 2],
    pub accepted_constraints: u32,
    pub rejected_constraints: u32,
    pub locked: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CoupledMotionStatus {
    pub timestamp_ns: u64,
    pub dt_ms: f32,
    pub reference_generation: u64,
    pub cyan: KinematicDerivatives,
    pub green: KinematicDerivatives,
    pub green_relative_to_cyan: KinematicDerivatives,
    /// Algebraic fixed point of the green-after-cyan 2-D similarity. This is
    /// useful corroboration, but is not an anatomical assertion.
    pub relative_motion_fixed_point: RotationCenterStatus,
    /// Backward-compatible telemetry alias for the relative-motion fixed
    /// point. New anatomy consumers must use `projected_globe`.
    pub green_rotation_center: RotationCenterStatus,
    pub projected_globe: ProjectedGlobePoseStatus,
    /// A short-horizon image-space center of the cyan transform.  Translation
    /// makes this correctly unobservable (a center at infinity).
    pub cyan_rotation_center: RotationCenterStatus,
    pub saccade_likelihood: f32,
    pub micro_motion_likelihood: f32,
}

impl CoupledMotionStatus {
    pub fn translated(mut self, dx: f32, dy: f32) -> Self {
        for derivatives in [&mut self.cyan, &mut self.green] {
            // A zero-sample derivative is an explicit "unobserved" value,
            // not a sensor-space point at (0, 0).  Preserve that sentinel so
            // an ROI offset cannot turn missing evidence into a plausible-
            // looking (and usually very negative) coordinate.
            if derivatives.samples != 0 {
                derivatives.reference_point[0] -= dx;
                derivatives.reference_point[1] -= dy;
            }
        }
        if let Some(center) = self.green_rotation_center.projected_center.as_mut() {
            center[0] -= dx;
            center[1] -= dy;
        }
        if let Some(center) = self.relative_motion_fixed_point.projected_center.as_mut() {
            center[0] -= dx;
            center[1] -= dy;
        }
        if let Some(pivot) = self.projected_globe.projected_pivot.as_mut() {
            pivot[0] -= dx;
            pivot[1] -= dy;
        }
        if let Some(center) = self.cyan_rotation_center.projected_center.as_mut() {
            center[0] -= dx;
            center[1] -= dy;
        }
        self
    }
}

#[derive(Clone, Copy, Debug)]
struct Affine2 {
    linear: [[f64; 2]; 2],
    translation: [f64; 2],
}

impl Default for Affine2 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Affine2 {
    fn identity() -> Self {
        Self {
            linear: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0; 2],
        }
    }

    fn from_motion(motion: SimilarityMotion, center: [f32; 2]) -> Self {
        let scale = 1.0 + motion.scale_delta as f64;
        let rotation = motion.rotation as f64;
        let linear = [[scale, -rotation], [rotation, scale]];
        let center = [center[0] as f64, center[1] as f64];
        let transformed_center = matrix_vector(linear, center);
        Self {
            linear,
            translation: [
                motion.translation[0] as f64 + center[0] - transformed_center[0],
                motion.translation[1] as f64 + center[1] - transformed_center[1],
            ],
        }
    }

    /// `self(inner(point))`.
    fn compose(self, inner: Self) -> Self {
        let linear = matrix_multiply(self.linear, inner.linear);
        let translated = matrix_vector(self.linear, inner.translation);
        Self {
            linear,
            translation: [
                translated[0] + self.translation[0],
                translated[1] + self.translation[1],
            ],
        }
    }

    fn inverse(self) -> Option<Self> {
        let determinant = determinant(self.linear);
        if !determinant.is_finite() || determinant.abs() < 1.0e-8 {
            return None;
        }
        let inverse = [
            [
                self.linear[1][1] / determinant,
                -self.linear[0][1] / determinant,
            ],
            [
                -self.linear[1][0] / determinant,
                self.linear[0][0] / determinant,
            ],
        ];
        let translated = matrix_vector(inverse, self.translation);
        Some(Self {
            linear: inverse,
            translation: [-translated[0], -translated[1]],
        })
    }

    fn apply(self, point: [f64; 2]) -> [f64; 2] {
        let transformed = matrix_vector(self.linear, point);
        [
            transformed[0] + self.translation[0],
            transformed[1] + self.translation[1],
        ]
    }

    fn apply_vector(self, vector: [f64; 2]) -> [f64; 2] {
        matrix_vector(self.linear, vector)
    }

    fn angle(self) -> f64 {
        self.linear[1][0].atan2(self.linear[0][0])
    }

    fn scale(self) -> f64 {
        determinant(self.linear).abs().sqrt()
    }
}

fn matrix_vector(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn matrix_multiply(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

fn determinant(matrix: [[f64; 2]; 2]) -> f64 {
    matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
}

fn fixed_point_equation(transform: Affine2) -> ([[f64; 2]; 2], [f64; 2]) {
    (
        [
            [1.0 - transform.linear[0][0], -transform.linear[0][1]],
            [-transform.linear[1][0], 1.0 - transform.linear[1][1]],
        ],
        transform.translation,
    )
}

fn equation_residual(matrix: [[f64; 2]; 2], rhs: [f64; 2], point: [f64; 2]) -> f64 {
    let predicted = matrix_vector(matrix, point);
    (predicted[0] - rhs[0]).hypot(predicted[1] - rhs[1])
}

#[derive(Clone, Debug, Default)]
struct CenterInformation {
    matrix: [[f64; 2]; 2],
    vector: [f64; 2],
    estimate: Option<[f64; 2]>,
    residual_ema: f64,
    accepted: u32,
    rejected: u32,
}

impl CenterInformation {
    fn add_prior(&mut self, center: [f64; 2], sigma: f64) {
        let weight = 1.0 / sigma.max(1.0).powi(2);
        self.matrix[0][0] += weight;
        self.matrix[1][1] += weight;
        self.vector[0] += weight * center[0];
        self.vector[1] += weight * center[1];
        self.estimate = solve_two(self.matrix, self.vector);
    }

    fn observe(
        &mut self,
        equation: [[f64; 2]; 2],
        rhs: [f64; 2],
        quality: f64,
        decay: f64,
        soft_residual_px: f64,
        hard_residual_px: f64,
    ) -> bool {
        for row in 0..2 {
            self.vector[row] *= decay;
            for column in 0..2 {
                self.matrix[row][column] *= decay;
            }
        }
        let excitation = equation
            .iter()
            .flat_map(|row| row.iter())
            .map(|value| value * value)
            .sum::<f64>();
        if !excitation.is_finite() || excitation < MIN_CENTER_EXCITATION {
            self.rejected = self.rejected.saturating_add(1);
            return false;
        }
        let residual = self
            .estimate
            .map(|estimate| equation_residual(equation, rhs, estimate))
            .unwrap_or(0.0);
        if !residual.is_finite() || residual > hard_residual_px {
            self.rejected = self.rejected.saturating_add(1);
            return false;
        }
        let robust = if residual <= soft_residual_px {
            1.0
        } else {
            soft_residual_px / residual.max(soft_residual_px)
        };
        let weight = quality.clamp(0.02, 1.0) * robust;
        for column in 0..2 {
            self.vector[column] +=
                weight * (equation[0][column] * rhs[0] + equation[1][column] * rhs[1]);
            for other in 0..2 {
                self.matrix[column][other] += weight
                    * (equation[0][column] * equation[0][other]
                        + equation[1][column] * equation[1][other]);
            }
        }
        let Some(estimate) = solve_two(self.matrix, self.vector) else {
            self.rejected = self.rejected.saturating_add(1);
            return false;
        };
        self.estimate = Some(estimate);
        self.residual_ema = if self.accepted == 0 {
            residual
        } else {
            0.85 * self.residual_ema + 0.15 * residual
        };
        self.accepted = self.accepted.saturating_add(1);
        true
    }

    fn sigma(&self) -> f64 {
        let Some(inverse) = inverse_two(self.matrix) else {
            return f64::INFINITY;
        };
        let trace = inverse[0][0] + inverse[1][1];
        let discriminant = ((inverse[0][0] - inverse[1][1]).powi(2)
            + 4.0 * inverse[0][1] * inverse[1][0])
            .max(0.0)
            .sqrt();
        ((trace + discriminant) * 0.5).max(0.0).sqrt()
    }

    fn confidence(&self, scale_px: f64) -> f64 {
        let sigma = self.sigma();
        if !sigma.is_finite() {
            return 0.0;
        }
        let sample_confidence = (self.accepted as f64 / 12.0).clamp(0.0, 1.0);
        let precision = (-sigma / scale_px.max(1.0)).exp();
        let residual = (-self.residual_ema / 4.0).exp();
        sample_confidence * precision * residual
    }
}

fn solve_two(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> Option<[f64; 2]> {
    let determinant = determinant(matrix);
    if !determinant.is_finite() || determinant.abs() < 1.0e-10 {
        return None;
    }
    Some([
        (vector[0] * matrix[1][1] - matrix[0][1] * vector[1]) / determinant,
        (matrix[0][0] * vector[1] - vector[0] * matrix[1][0]) / determinant,
    ])
}

fn inverse_two(matrix: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let determinant = determinant(matrix);
    if !determinant.is_finite() || determinant.abs() < 1.0e-10 {
        return None;
    }
    Some([
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ])
}

#[derive(Clone, Copy, Debug)]
struct PoseSample {
    timestamp_ns: u64,
    position: [f64; 2],
    angle: f64,
    quality: f64,
}

#[derive(Clone, Debug, Default)]
struct PoseHistory {
    samples: VecDeque<PoseSample>,
}

#[derive(Clone, Copy)]
enum MotionProfile {
    Cyan,
    Green,
    Relative,
}

impl PoseHistory {
    fn clear(&mut self) {
        self.samples.clear();
    }

    fn observe_delta(
        &mut self,
        previous_timestamp_ns: u64,
        timestamp_ns: u64,
        current_reference_point: [f64; 2],
        delta: [f64; 2],
        angle_delta: f64,
        quality: f64,
    ) {
        let contiguous = self
            .samples
            .back()
            .is_some_and(|sample| sample.timestamp_ns == previous_timestamp_ns)
            && timestamp_ns > previous_timestamp_ns
            && timestamp_ns - previous_timestamp_ns <= MAX_CONTIGUOUS_DT_NS;
        if !contiguous {
            self.samples.clear();
            self.samples.push_back(PoseSample {
                timestamp_ns: previous_timestamp_ns,
                position: [
                    current_reference_point[0] - delta[0],
                    current_reference_point[1] - delta[1],
                ],
                angle: 0.0,
                quality,
            });
        }
        let previous = self.samples.back().copied().unwrap();
        self.samples.push_back(PoseSample {
            timestamp_ns,
            position: [
                previous.position[0] + delta[0],
                previous.position[1] + delta[1],
            ],
            angle: previous.angle + angle_delta,
            quality,
        });
        while self.samples.len() > 16 {
            self.samples.pop_front();
        }
        while self
            .samples
            .front()
            .is_some_and(|sample| timestamp_ns.saturating_sub(sample.timestamp_ns) > 1_200_000_000)
        {
            self.samples.pop_front();
        }
    }

    fn derivatives(&self, profile: MotionProfile) -> KinematicDerivatives {
        let Some(latest) = self.samples.back().copied() else {
            return KinematicDerivatives::default();
        };
        let instantaneous = self
            .samples
            .iter()
            .rev()
            .take(2)
            .copied()
            .collect::<Vec<_>>();
        let (instantaneous_speed, instantaneous_angular_speed) = if instantaneous.len() == 2 {
            let dt = (instantaneous[0].timestamp_ns - instantaneous[1].timestamp_ns) as f64 * 1e-9;
            if dt > 1.0e-6 {
                (
                    (instantaneous[0].position[0] - instantaneous[1].position[0])
                        .hypot(instantaneous[0].position[1] - instantaneous[1].position[1])
                        / dt,
                    (instantaneous[0].angle - instantaneous[1].angle).abs() / dt,
                )
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };
        let saccadic = matches!(profile, MotionProfile::Relative)
            && (instantaneous_speed >= 24.0 || instantaneous_angular_speed >= 0.18);
        let (horizon_ns, maximum_samples) = match profile {
            // Cyan represents head/sclera structure and receives the longest
            // inertial window.
            MotionProfile::Cyan => (1_000_000_000u64, 8usize),
            // Green remains materially snappier, but spans four samples even
            // in the recorded 6-7 fps long-exposure corpus so jerk is
            // observable instead of silently zero.
            MotionProfile::Green => (650_000_000u64, 6usize),
            MotionProfile::Relative if saccadic => (500_000_000u64, 4usize),
            MotionProfile::Relative => (700_000_000u64, 6usize),
        };
        let mut samples = self
            .samples
            .iter()
            .rev()
            .take(maximum_samples)
            .take_while(|sample| {
                latest.timestamp_ns.saturating_sub(sample.timestamp_ns) <= horizon_ns
            })
            .copied()
            .collect::<Vec<_>>();
        samples.reverse();
        let x = fit_pose_component(&samples, |sample| sample.position[0]);
        let y = fit_pose_component(&samples, |sample| sample.position[1]);
        let angle = fit_pose_component(&samples, |sample| sample.angle);
        let confidence_scale = match profile {
            MotionProfile::Cyan => 1.7,
            MotionProfile::Green => 2.4,
            MotionProfile::Relative => 2.0,
        };
        let sample_confidence = ((samples.len().saturating_sub(1)) as f64 / 5.0).clamp(0.0, 1.0);
        let mean_quality =
            samples.iter().map(|sample| sample.quality).sum::<f64>() / samples.len().max(1) as f64;
        let residual_px = x.residual.hypot(y.residual);
        let confidence = sample_confidence
            * mean_quality.clamp(0.0, 1.0)
            * (-residual_px / confidence_scale).exp()
            * (-angle.residual / 0.035).exp();
        KinematicDerivatives {
            reference_point: [latest.position[0] as f32, latest.position[1] as f32],
            velocity_px_s: [x.velocity as f32, y.velocity as f32],
            acceleration_px_s2: [x.acceleration as f32, y.acceleration as f32],
            jerk_px_s3: [x.jerk as f32, y.jerk as f32],
            angle_rad: latest.angle as f32,
            angular_velocity_rad_s: angle.velocity as f32,
            angular_acceleration_rad_s2: angle.acceleration as f32,
            angular_jerk_rad_s3: angle.jerk as f32,
            speed_px_s: x.velocity.hypot(y.velocity) as f32,
            fit_residual_px: residual_px as f32,
            angular_fit_residual_rad: angle.residual as f32,
            confidence: confidence as f32,
            samples: samples.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PolynomialDerivatives {
    velocity: f64,
    acceleration: f64,
    jerk: f64,
    residual: f64,
}

fn fit_pose_component(
    samples: &[PoseSample],
    value: impl Fn(&PoseSample) -> f64,
) -> PolynomialDerivatives {
    if samples.len() < 2 {
        return PolynomialDerivatives::default();
    }
    let latest = samples.last().unwrap().timestamp_ns;
    let horizon = (latest - samples.first().unwrap().timestamp_ns) as f64 * 1e-9;
    if horizon <= 1.0e-6 {
        return PolynomialDerivatives::default();
    }
    let order = samples.len().saturating_sub(1).min(3);
    let mut robust_weights = vec![1.0f64; samples.len()];
    let mut coefficients = [0.0f64; 4];
    for iteration in 0..2 {
        let mut matrix = [[0.0f64; 4]; 4];
        let mut vector = [0.0f64; 4];
        for (index, sample) in samples.iter().enumerate() {
            let normalized_time = -((latest - sample.timestamp_ns) as f64 * 1e-9) / horizon;
            let basis = [
                1.0,
                normalized_time,
                normalized_time * normalized_time,
                normalized_time * normalized_time * normalized_time,
            ];
            let recency = (2.2 * normalized_time).exp();
            let weight = sample.quality.clamp(0.05, 1.0) * recency * robust_weights[index];
            for column in 0..=order {
                vector[column] += weight * basis[column] * value(sample);
                for other in 0..=order {
                    matrix[column][other] += weight * basis[column] * basis[other];
                }
            }
        }
        for diagonal in 0..=order {
            matrix[diagonal][diagonal] += 1.0e-9;
        }
        let Some(solution) = solve_polynomial(matrix, vector, order + 1) else {
            return PolynomialDerivatives::default();
        };
        coefficients = solution;
        if iteration == 0 && samples.len() >= 4 {
            let residuals = samples
                .iter()
                .map(|sample| {
                    let time = -((latest - sample.timestamp_ns) as f64 * 1e-9) / horizon;
                    (evaluate_polynomial(coefficients, time, order) - value(sample)).abs()
                })
                .collect::<Vec<_>>();
            let mut ranked_residuals = residuals.clone();
            ranked_residuals.sort_by(f64::total_cmp);
            let scale = ranked_residuals[ranked_residuals.len() / 2].max(1.0e-4) * 2.5;
            for (weight, residual) in robust_weights.iter_mut().zip(residuals.into_iter()) {
                *weight = if residual <= scale {
                    1.0
                } else {
                    scale / residual
                };
            }
        }
    }
    let residual = (samples
        .iter()
        .map(|sample| {
            let time = -((latest - sample.timestamp_ns) as f64 * 1e-9) / horizon;
            let error = evaluate_polynomial(coefficients, time, order) - value(sample);
            error * error
        })
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    PolynomialDerivatives {
        velocity: coefficients[1] / horizon,
        acceleration: if order >= 2 {
            2.0 * coefficients[2] / (horizon * horizon)
        } else {
            0.0
        },
        jerk: if order >= 3 {
            6.0 * coefficients[3] / (horizon * horizon * horizon)
        } else {
            0.0
        },
        residual,
    }
}

fn evaluate_polynomial(coefficients: [f64; 4], time: f64, order: usize) -> f64 {
    (0..=order).rev().fold(0.0, |accumulator, index| {
        accumulator * time + coefficients[index]
    })
}

fn solve_polynomial(
    mut matrix: [[f64; 4]; 4],
    mut vector: [f64; 4],
    dimensions: usize,
) -> Option<[f64; 4]> {
    for pivot in 0..dimensions {
        let best = (pivot..dimensions).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right][pivot].abs())
        })?;
        if matrix[best][pivot].abs() < 1.0e-12 {
            return None;
        }
        if best != pivot {
            matrix.swap(best, pivot);
            vector.swap(best, pivot);
        }
        let divisor = matrix[pivot][pivot];
        for column in pivot..dimensions {
            matrix[pivot][column] /= divisor;
        }
        vector[pivot] /= divisor;
        for row in 0..dimensions {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..dimensions {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            vector[row] -= factor * vector[pivot];
        }
    }
    Some(vector)
}

fn layer_quality(motion: SimilarityMotion, layer: MotionLayerStatus) -> f64 {
    if motion.support < MIN_LAYER_SUPPORT
        || layer.persistent_tracks < MIN_LAYER_SUPPORT
        || !motion.residual.is_finite()
        || motion.residual > 5.0
    {
        return 0.0;
    }
    let support = (motion.support.min(layer.persistent_tracks) as f64 / 8.0).clamp(0.0, 1.0);
    let residual = (-(motion.residual as f64) / 3.2).exp();
    let coherence = layer.coherence.max(0.08) as f64;
    let maturity = (layer.stable_frames as f64 / 4.0).clamp(0.20, 1.0);
    support * residual * coherence * maturity
}

#[derive(Default)]
struct ProjectedGlobeFilter {
    pivot_reference: Option<[f64; 2]>,
    pole_reference: Option<[f64; 2]>,
    variance_reference: f64,
    translation_reference: [f64; 2],
    material_frame_correction_reference: [f64; 2],
    last_geometry_sensor: Option<[f64; 2]>,
    last_radius_reference: f64,
    fixation_frames: u16,
    corroborated_frames: u16,
    occluded_frames: u16,
    trusted_keyframe: bool,
    bridge_age_s: f64,
    bridge_age_frames: u16,
    accepted_updates: u32,
    rejected_updates: u32,
    was_publishable: bool,
    status: ProjectedGlobePoseStatus,
}

impl ProjectedGlobeFilter {
    fn rebase(
        &mut self,
        retained_pivot: Option<[f64; 2]>,
        retained_pole: Option<[f64; 2]>,
        radius: f64,
    ) {
        self.pivot_reference = retained_pivot;
        self.pole_reference = retained_pole;
        self.variance_reference = self
            .variance_reference
            .max((radius.max(8.0) * 0.18).powi(2));
        self.translation_reference = [0.0; 2];
        self.material_frame_correction_reference = [0.0; 2];
        self.last_geometry_sensor = None;
        self.last_radius_reference = radius.max(8.0);
        self.fixation_frames = 0;
        self.corroborated_frames = self.corroborated_frames.saturating_sub(2);
        self.occluded_frames = 0;
        self.trusted_keyframe = false;
        self.bridge_age_s = 0.0;
        self.bridge_age_frames = 0;
        self.was_publishable = false;
        self.status.publishable = false;
        self.status.temporally_bridged = false;
        self.status.bridge_age_ms = 0.0;
        self.status.bridge_prediction_disagreement_radii = f32::MAX;
    }

    fn mark_occluded(
        &mut self,
        current_reference_pose: Option<Affine2>,
    ) -> ProjectedGlobePoseStatus {
        // No valid material-frame transport exists on this path. A trusted
        // keyframe cannot be carried through an unknown sensor/head motion.
        self.trusted_keyframe = false;
        self.bridge_age_s = 0.0;
        self.bridge_age_frames = 0;
        self.occluded_frames = self.occluded_frames.saturating_add(1);
        self.fixation_frames = self.fixation_frames.saturating_sub(1);
        self.corroborated_frames = self.corroborated_frames.saturating_sub(1);
        self.status.regime = GlobeMotionRegime::Occluded;
        self.status.anatomy_authorized = false;
        self.status.fixed_point_corroborated = false;
        self.status.fixation_frames = self.fixation_frames;
        self.status.corroborated_frames = self.corroborated_frames;
        self.status.occluded_frames = self.occluded_frames;
        self.status.temporally_bridged = false;
        self.status.bridge_age_ms = 0.0;
        self.status.bridge_prediction_disagreement_radii = f32::MAX;
        self.status.confidence *= 0.82;
        if let Some(pose) = current_reference_pose {
            self.status.projected_pivot = self.pivot_reference.map(|pivot| {
                let projected = pose.apply(pivot);
                [projected[0] as f32, projected[1] as f32]
            });
            self.status.projected_pole = self.pole_reference.map(|pole| {
                let projected = pose.apply_vector(pole);
                [projected[0] as f32, projected[1] as f32]
            });
        }
        // The old generic occlusion hold was unsafe because it could also run
        // when the cyan material transform itself was unavailable. Only
        // `bridge_weak_geometry` may carry publication, and only after checking
        // current-frame conic consistency under a valid material transform.
        self.status.publishable = false;
        self.was_publishable = false;
        self.status
    }

    fn bridge_weak_geometry(
        &mut self,
        dt_s: f64,
        current_reference_pose: Affine2,
        geometry_sensor: ProjectedIrisGeometry,
        geometry_reference: ProjectedIrisGeometry,
    ) -> ProjectedGlobePoseStatus {
        let radius_reference = geometry_reference.area_radius().max(8.0);
        let radius_sensor = geometry_sensor.area_radius().max(8.0);
        let predicted_center =
            self.pivot_reference
                .zip(self.pole_reference)
                .map(|(pivot, pole)| {
                    current_reference_pose.apply([pivot[0] + pole[0], pivot[1] + pole[1]])
                });
        let center_disagreement = predicted_center
            .map(|predicted| {
                (predicted[0] - geometry_sensor.center[0])
                    .hypot(predicted[1] - geometry_sensor.center[1])
                    / radius_sensor
            })
            .unwrap_or(f64::INFINITY);
        let radius_disagreement = if self.last_radius_reference >= 8.0 {
            (radius_reference - self.last_radius_reference).abs()
                / self.last_radius_reference.max(radius_reference)
        } else {
            f64::INFINITY
        };
        let next_age_s = self.bridge_age_s + dt_s.max(0.0);
        let next_age_frames = self.bridge_age_frames.saturating_add(1);
        // Confidence here describes *new edge authority*. A carried semantic
        // conic may correctly report zero while still serving as a bounded,
        // non-authoritative cross-check against the independently transported
        // keyframe. The hard time limit prevents that check from voting for
        // itself indefinitely.
        let prediction_consistent = center_disagreement <= GLOBE_BRIDGE_CENTER_GATE_RADII
            && radius_disagreement <= GLOBE_BRIDGE_RADIUS_GATE_FRACTION;
        let bridge_publishable = self.trusted_keyframe
            && prediction_consistent
            && next_age_s <= GLOBE_TRUSTED_BRIDGE_SECONDS
            && next_age_frames <= GLOBE_TRUSTED_BRIDGE_FRAMES
            && self.status.confidence >= 0.04;

        self.occluded_frames = self.occluded_frames.saturating_add(1);
        self.status.regime = GlobeMotionRegime::Occluded;
        self.status.anatomy_authorized = false;
        self.status.fixed_point_corroborated = false;
        self.status.occluded_frames = self.occluded_frames;
        self.status.anatomy_confidence = geometry_sensor.confidence.clamp(0.0, 1.0) as f32;
        self.status.bridge_prediction_disagreement_radii =
            center_disagreement.min(f32::MAX as f64) as f32;
        if bridge_publishable {
            self.bridge_age_s = next_age_s;
            self.bridge_age_frames = next_age_frames;
            // Preserve the last direct counters. They describe the keyframe,
            // while bridge_age explicitly describes how stale that authority
            // is. Do not manufacture current corroboration or fixation votes.
            self.status.temporally_bridged = true;
            self.status.bridge_age_ms = (self.bridge_age_s * 1_000.0) as f32;
            self.status.confidence *= (-dt_s.max(0.0) / 1.25).exp() as f32;
            self.status.projected_pivot = self.pivot_reference.map(|pivot| {
                let projected = current_reference_pose.apply(pivot);
                [projected[0] as f32, projected[1] as f32]
            });
            self.status.projected_pole = self.pole_reference.map(|pole| {
                let projected = current_reference_pose.apply_vector(pole);
                [projected[0] as f32, projected[1] as f32]
            });
            self.status.publishable = true;
            self.was_publishable = true;
            return self.status;
        }

        self.trusted_keyframe = false;
        self.bridge_age_s = 0.0;
        self.bridge_age_frames = 0;
        self.status.temporally_bridged = false;
        self.status.bridge_age_ms = 0.0;
        self.status.publishable = false;
        self.was_publishable = false;
        self.status.confidence *= 0.82;
        self.status
    }

    fn bridge_sensor_geometry(
        &mut self,
        dt_s: f64,
        current_reference_pose: Affine2,
        geometry_sensor: ProjectedIrisGeometry,
    ) -> ProjectedGlobePoseStatus {
        let Some(reference_inverse) = current_reference_pose.inverse() else {
            return self.mark_occluded(Some(current_reference_pose));
        };
        if !geometry_sensor.valid() {
            return self.mark_occluded(Some(current_reference_pose));
        }
        let geometry_reference = ProjectedIrisGeometry {
            center: reference_inverse.apply(geometry_sensor.center),
            major_radius: geometry_sensor.major_radius / current_reference_pose.scale().max(1.0e-6),
            minor_radius: geometry_sensor.minor_radius / current_reference_pose.scale().max(1.0e-6),
            angle_rad: (geometry_sensor.angle_rad + reference_inverse.angle())
                .rem_euclid(std::f64::consts::PI),
            confidence: geometry_sensor.confidence,
            anatomy_authorized: false,
        };
        self.bridge_weak_geometry(
            dt_s,
            current_reference_pose,
            geometry_sensor,
            geometry_reference,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        dt_s: f64,
        cyan_step: Affine2,
        current_reference_pose: Affine2,
        geometry_sensor: Option<ProjectedIrisGeometry>,
        fixed_point_reference: Option<[f64; 2]>,
        fixed_point_sigma_reference: f64,
        fixed_point_accepted: u32,
        latest_constraint: Option<([[f64; 2]; 2], [f64; 2])>,
        relative_confidence: f64,
    ) -> ProjectedGlobePoseStatus {
        let Some(reference_inverse) = current_reference_pose.inverse() else {
            return self.mark_occluded(Some(current_reference_pose));
        };
        let geometry_reference =
            geometry_sensor
                .filter(|geometry| geometry.valid())
                .map(|geometry| ProjectedIrisGeometry {
                    center: reference_inverse.apply(geometry.center),
                    major_radius: geometry.major_radius
                        / current_reference_pose.scale().max(1.0e-6),
                    minor_radius: geometry.minor_radius
                        / current_reference_pose.scale().max(1.0e-6),
                    angle_rad: (geometry.angle_rad + reference_inverse.angle())
                        .rem_euclid(std::f64::consts::PI),
                    confidence: geometry.confidence,
                    anatomy_authorized: geometry.anatomy_authorized,
                });
        let Some(geometry) = geometry_reference else {
            return self.mark_occluded(Some(current_reference_pose));
        };
        let radius = geometry.area_radius().max(8.0);
        let authorized = geometry.anatomy_authorized && geometry.confidence >= 0.12;
        if !authorized {
            return self.bridge_weak_geometry(
                dt_s,
                current_reference_pose,
                geometry_sensor.unwrap(),
                geometry,
            );
        }

        let geometry_sensor = geometry_sensor.unwrap();
        let step_radii = self
            .last_geometry_sensor
            .map(|previous| {
                let predicted = cyan_step.apply(previous);
                let material_residual = (geometry_sensor.center[0] - predicted[0])
                    .hypot(geometry_sensor.center[1] - predicted[1])
                    / geometry_sensor
                        .area_radius()
                        .max(self.last_radius_reference)
                        .max(8.0);
                let sensor_hold_residual = (geometry_sensor.center[0] - previous[0])
                    .hypot(geometry_sensor.center[1] - previous[1])
                    / geometry_sensor
                        .area_radius()
                        .max(self.last_radius_reference)
                        .max(8.0);
                material_residual.min(sensor_hold_residual)
            })
            .unwrap_or(0.0);
        let regime = if self.occluded_frames > 0 || self.last_geometry_sensor.is_none() {
            GlobeMotionRegime::Uncertain
        } else if step_radii <= GLOBE_FIXATION_STEP_RADII {
            GlobeMotionRegime::Fixation
        } else if step_radii < GLOBE_SACCADE_STEP_RADII
            && self.status.regime == GlobeMotionRegime::Fixation
            && self.fixation_frames >= 2
        {
            // Hysteresis keeps one marginal, quantized limbus-center step from
            // ending a fixation; a true saccade still clears the outer bound.
            GlobeMotionRegime::Fixation
        } else if step_radii >= GLOBE_SACCADE_STEP_RADII && relative_confidence >= 0.08 {
            GlobeMotionRegime::Saccade
        } else {
            GlobeMotionRegime::Uncertain
        };
        self.last_geometry_sensor = Some(geometry_sensor.center);
        self.last_radius_reference = radius;
        self.occluded_frames = 0;
        self.fixation_frames = match regime {
            GlobeMotionRegime::Fixation => self.fixation_frames.saturating_add(1),
            GlobeMotionRegime::Saccade => 0,
            _ => self.fixation_frames.saturating_sub(1),
        };

        if self.pivot_reference.is_none() {
            self.pivot_reference = Some(geometry.center);
            self.variance_reference = (radius * 0.45).powi(2);
            self.accepted_updates = self.accepted_updates.saturating_add(1);
        } else {
            // The iris center is a biased observation of the globe pivot once
            // gaze is eccentric. During a firmly observed fixation it may
            // correct slow orbital/head-frame drift, but it must not chase a
            // saccade. The time-based cap is cadence independent.
            if regime == GlobeMotionRegime::Fixation {
                let anatomy_alpha = (1.0 - (-dt_s.max(0.0) / 20.0).exp()).clamp(0.0, 0.012);
                self.bounded_update(
                    geometry.center,
                    (radius * 0.55).powi(2),
                    anatomy_alpha,
                    radius * 0.006,
                );
            }
            self.variance_reference += (radius * 0.0015 * dt_s.max(0.001).sqrt()).powi(2);
        }

        // The tight eye ROI does not always contain a rigid facial material
        // frame. Lids and brow skin can pull the nominal cyan transform while
        // the iris remains correctly localized. A true projected globe pivot
        // cannot end up several iris radii from the current iris center, so
        // absorb only the impossible excess as material-frame transport slip.
        // The remaining 0.90-radius allowance still permits eccentric gaze.
        self.material_frame_correction_reference = [0.0; 2];
        if let Some(pivot) = self.pivot_reference {
            let delta = [geometry.center[0] - pivot[0], geometry.center[1] - pivot[1]];
            let distance = delta[0].hypot(delta[1]);
            let permitted = radius * 0.90;
            if distance > permitted {
                let correction_length = (distance - permitted).min(radius * 0.35);
                let correction = [
                    delta[0] * correction_length / distance,
                    delta[1] * correction_length / distance,
                ];
                self.pivot_reference = Some([pivot[0] + correction[0], pivot[1] + correction[1]]);
                self.material_frame_correction_reference = correction;
                self.variance_reference += (correction_length * 0.20).powi(2);
                self.rejected_updates = self.rejected_updates.saturating_add(1);
            }
        }

        let pivot_before_fixed = self.pivot_reference.unwrap_or(geometry.center);
        let anatomy_disagreement = (pivot_before_fixed[0] - geometry.center[0])
            .hypot(pivot_before_fixed[1] - geometry.center[1])
            / radius;
        let fixed_point_disagreement = fixed_point_reference
            .map(|candidate| {
                (candidate[0] - pivot_before_fixed[0]).hypot(candidate[1] - pivot_before_fixed[1])
                    / radius
            })
            .unwrap_or(f64::INFINITY);
        let geometry_fixed_disagreement = fixed_point_reference
            .map(|candidate| {
                (candidate[0] - geometry.center[0]).hypot(candidate[1] - geometry.center[1])
                    / radius
            })
            .unwrap_or(f64::INFINITY);
        let translation = latest_constraint.map_or([0.0; 2], |(equation, rhs)| {
            let predicted = matrix_vector(equation, pivot_before_fixed);
            [rhs[0] - predicted[0], rhs[1] - predicted[1]]
        });
        let translation_radii = translation[0].hypot(translation[1]) / radius;
        let fixed_point_corroborated = fixed_point_accepted >= CENTER_LOCK_MIN_SAMPLES
            && fixed_point_sigma_reference.is_finite()
            && fixed_point_sigma_reference <= radius * 0.75
            && fixed_point_disagreement <= GLOBE_FIXED_POINT_GATE_RADII
            && geometry_fixed_disagreement <= GLOBE_FIXED_POINT_GATE_RADII
            && translation_radii <= GLOBE_TRANSLATION_LIMIT_RADII;
        if fixed_point_corroborated {
            self.corroborated_frames = self.corroborated_frames.saturating_add(1);
            if let Some(candidate) = fixed_point_reference {
                self.bounded_update(
                    candidate,
                    fixed_point_sigma_reference.max(radius * 0.18).powi(2),
                    0.055,
                    radius * 0.025,
                );
            }
            let translation_alpha = (1.0 - (-dt_s.max(0.0) / 0.35).exp()).clamp(0.02, 0.35);
            self.translation_reference = [
                (1.0 - translation_alpha) * self.translation_reference[0]
                    + translation_alpha * translation[0],
                (1.0 - translation_alpha) * self.translation_reference[1]
                    + translation_alpha * translation[1],
            ];
        } else {
            self.corroborated_frames = self.corroborated_frames.saturating_sub(1);
            self.rejected_updates = self.rejected_updates.saturating_add(1);
            self.translation_reference = [
                self.translation_reference[0] * 0.75,
                self.translation_reference[1] * 0.75,
            ];
        }

        let pivot_reference = self.pivot_reference.unwrap_or(geometry.center);
        self.pole_reference = Some([
            geometry.center[0] - pivot_reference[0],
            geometry.center[1] - pivot_reference[1],
        ]);
        let projected_pivot = current_reference_pose.apply(pivot_reference);
        let projected_pole = current_reference_pose.apply_vector(self.pole_reference.unwrap());
        let projected_translation = current_reference_pose.apply_vector(self.translation_reference);
        let projected_material_correction =
            current_reference_pose.apply_vector(self.material_frame_correction_reference);
        let sigma = self.variance_reference.max(0.0).sqrt() * current_reference_pose.scale();
        let trusted_direct_handoff = self.trusted_keyframe
            && fixed_point_corroborated
            && self.bridge_age_s <= GLOBE_TRUSTED_BRIDGE_SECONDS
            && self.bridge_age_frames <= GLOBE_TRUSTED_BRIDGE_FRAMES
            && matches!(
                regime,
                GlobeMotionRegime::Uncertain | GlobeMotionRegime::Saccade
            );
        let anatomy_factor = match regime {
            GlobeMotionRegime::Fixation => (self.fixation_frames as f64 / 4.0).clamp(0.0, 1.0),
            GlobeMotionRegime::Uncertain | GlobeMotionRegime::Saccade if trusted_direct_handoff => {
                0.75
            }
            _ => 0.0,
        };
        let corroboration_factor = (self.corroborated_frames as f64 / 4.0).clamp(0.0, 1.0);
        let precision = (-(sigma / (radius * current_reference_pose.scale()).max(1.0))).exp();
        let translation_factor = (-(translation_radii / 0.10).powi(2)).exp();
        let confidence = geometry.confidence.clamp(0.0, 1.0)
            * anatomy_factor
            * corroboration_factor
            * precision
            * translation_factor;
        let directly_stable = regime == GlobeMotionRegime::Fixation && self.fixation_frames >= 2;
        let publishable = (directly_stable || trusted_direct_handoff)
            && self.corroborated_frames >= 3
            && anatomy_disagreement <= 0.85
            && sigma <= radius * current_reference_pose.scale() * 0.55
            && confidence >= 0.06;
        if publishable {
            self.trusted_keyframe = true;
            self.bridge_age_s = 0.0;
            self.bridge_age_frames = 0;
        } else if self.trusted_keyframe {
            self.bridge_age_s += dt_s.max(0.0);
            self.bridge_age_frames = self.bridge_age_frames.saturating_add(1);
            if self.bridge_age_s > GLOBE_TRUSTED_BRIDGE_SECONDS
                || self.bridge_age_frames > GLOBE_TRUSTED_BRIDGE_FRAMES
            {
                self.trusted_keyframe = false;
            }
        }
        self.was_publishable = publishable;
        self.status = ProjectedGlobePoseStatus {
            projected_pivot: Some([projected_pivot[0] as f32, projected_pivot[1] as f32]),
            projected_pole: Some([projected_pole[0] as f32, projected_pole[1] as f32]),
            relative_translation_px: [
                projected_translation[0] as f32,
                projected_translation[1] as f32,
            ],
            material_frame_correction_px: [
                projected_material_correction[0] as f32,
                projected_material_correction[1] as f32,
            ],
            sigma_px: sigma.min(f32::MAX as f64) as f32,
            confidence: confidence as f32,
            anatomy_confidence: geometry.confidence.clamp(0.0, 1.0) as f32,
            anatomy_authorized: true,
            fixed_point_corroborated,
            anatomy_disagreement_radii: anatomy_disagreement as f32,
            fixed_point_disagreement_radii: fixed_point_disagreement.min(f32::MAX as f64) as f32,
            regime,
            fixation_frames: self.fixation_frames,
            corroborated_frames: self.corroborated_frames,
            occluded_frames: 0,
            temporally_bridged: false,
            bridge_age_ms: 0.0,
            bridge_prediction_disagreement_radii: 0.0,
            accepted_updates: self.accepted_updates,
            rejected_updates: self.rejected_updates,
            publishable,
        };
        self.status
    }

    fn bounded_update(
        &mut self,
        measurement: [f64; 2],
        measurement_variance: f64,
        maximum_gain: f64,
        maximum_step: f64,
    ) {
        let Some(previous) = self.pivot_reference else {
            self.pivot_reference = Some(measurement);
            self.variance_reference = measurement_variance.max(1.0);
            self.accepted_updates = self.accepted_updates.saturating_add(1);
            return;
        };
        let gain = (self.variance_reference
            / (self.variance_reference + measurement_variance.max(1.0)))
        .clamp(0.0, maximum_gain.max(0.0));
        let delta = [measurement[0] - previous[0], measurement[1] - previous[1]];
        let delta_length = delta[0].hypot(delta[1]);
        let step_scale = if delta_length > maximum_step.max(1.0e-6) {
            maximum_step / delta_length
        } else {
            1.0
        };
        self.pivot_reference = Some([
            previous[0] + gain * delta[0] * step_scale,
            previous[1] + gain * delta[1] * step_scale,
        ]);
        self.variance_reference =
            ((1.0 - gain) * self.variance_reference).max((maximum_step * 0.5).powi(2));
        self.accepted_updates = self.accepted_updates.saturating_add(1);
    }
}

#[derive(Default)]
pub struct CoupledEyeKinematics {
    last_timestamp_ns: Option<u64>,
    cyan_from_reference: Affine2,
    reference_generation: u64,
    chain_valid: bool,
    green_center: CenterInformation,
    cyan_center: CenterInformation,
    cyan_history: PoseHistory,
    green_history: PoseHistory,
    relative_history: PoseHistory,
    last_projected_green_center: Option<[f64; 2]>,
    projected_globe: ProjectedGlobeFilter,
    status: CoupledMotionStatus,
}

impl CoupledEyeKinematics {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn status(&self) -> CoupledMotionStatus {
        self.status
    }

    fn clear_current_dynamics(&mut self) {
        self.status.cyan = KinematicDerivatives::default();
        self.status.green = KinematicDerivatives::default();
        self.status.green_relative_to_cyan = KinematicDerivatives::default();
        self.status.saccade_likelihood = 0.0;
        self.status.micro_motion_likelihood = 0.0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        timestamp_ns: u64,
        analysis_center: [f32; 2],
        frame_extent_px: f64,
        motions: [SimilarityMotion; super::OBJECTS],
        layers: [MotionLayerStatus; super::OBJECTS],
        semantic_layers: bool,
        iris_geometry: Option<ProjectedIrisGeometry>,
    ) -> CoupledMotionStatus {
        let iris_center = iris_geometry.map(|geometry| geometry.center);
        let iris_radius = iris_geometry.map(ProjectedIrisGeometry::area_radius);
        let previous_timestamp_ns = self.last_timestamp_ns.replace(timestamp_ns);
        self.status.timestamp_ns = timestamp_ns;
        let Some(previous_timestamp_ns) = previous_timestamp_ns else {
            return self.status;
        };
        if timestamp_ns <= previous_timestamp_ns {
            self.clear_current_dynamics();
            return self.status;
        }
        let delta_ns = timestamp_ns - previous_timestamp_ns;
        self.status.dt_ms = delta_ns as f32 * 1.0e-6;
        let contiguous = (MIN_CONTIGUOUS_DT_NS..=MAX_CONTIGUOUS_DT_NS).contains(&delta_ns);
        let cyan_quality = layer_quality(motions[GENERAL_LAYER], layers[GENERAL_LAYER]);
        let green_quality = layer_quality(motions[PUPIL_LAYER], layers[PUPIL_LAYER]);
        let cyan_valid = contiguous && cyan_quality >= 0.015;
        let green_valid = cyan_valid && semantic_layers && green_quality >= 0.015;
        if !cyan_valid {
            // The full material-layer gate is intentionally conservative, but
            // once a direct globe keyframe exists a minimally supported cyan
            // transform is still more informative than either freezing sensor
            // coordinates or globally reacquiring. Use it only for the short
            // prediction-consistency bridge; it cannot update anatomy, the
            // fixed-point posterior, or any derivatives.
            let bridge_transport_valid = contiguous
                && self.chain_valid
                && self.projected_globe.trusted_keyframe
                && cyan_quality > 0.0;
            if bridge_transport_valid {
                let cyan = Affine2::from_motion(motions[GENERAL_LAYER], analysis_center);
                let current_reference_pose = cyan.compose(self.cyan_from_reference);
                if current_reference_pose.inverse().is_some() {
                    if let Some(geometry) = iris_geometry {
                        self.clear_current_dynamics();
                        self.status.green_rotation_center.confidence *= 0.82;
                        self.status.relative_motion_fixed_point.confidence *= 0.82;
                        self.status.cyan_rotation_center.confidence *= 0.75;
                        self.cyan_from_reference = current_reference_pose;
                        self.status.projected_globe = self.projected_globe.bridge_sensor_geometry(
                            delta_ns as f64 * 1.0e-9,
                            current_reference_pose,
                            ProjectedIrisGeometry {
                                anatomy_authorized: false,
                                ..geometry
                            },
                        );
                        if self.status.projected_globe.publishable {
                            return self.status;
                        }
                    }
                }
            }
            self.chain_valid = false;
            self.cyan_history.clear();
            self.green_history.clear();
            self.relative_history.clear();
            // The center posterior intentionally survives a temporary tissue
            // dropout, but derivatives describe current-frame evidence.  A
            // stale nonzero sample count must not authorize motion, saccade,
            // or observability decisions for a frame that failed the gate.
            self.clear_current_dynamics();
            self.status.green_rotation_center.confidence *= 0.82;
            self.status.relative_motion_fixed_point.confidence *= 0.82;
            self.status.cyan_rotation_center.confidence *= 0.75;
            self.status.projected_globe = self
                .projected_globe
                .mark_occluded(Some(self.cyan_from_reference));
            return self.status;
        }

        let cyan = Affine2::from_motion(motions[GENERAL_LAYER], analysis_center);
        let Some(cyan_inverse) = cyan.inverse() else {
            self.rebase(None, iris_radius.unwrap_or(frame_extent_px * 0.20));
            return self.status;
        };
        if !self.chain_valid {
            self.rebase(
                self.last_projected_green_center,
                iris_radius.unwrap_or(24.0),
            );
        }
        let previous_reference_pose = self.cyan_from_reference;
        let current_reference_pose = cyan.compose(previous_reference_pose);
        let cyan_transported_green_center = self
            .green_center
            .estimate
            .map(|center| current_reference_pose.apply(center));

        let cyan_current = [
            layers[GENERAL_LAYER].centroid[0] as f64,
            layers[GENERAL_LAYER].centroid[1] as f64,
        ];
        let cyan_previous = cyan_inverse.apply(cyan_current);
        self.cyan_history.observe_delta(
            previous_timestamp_ns,
            timestamp_ns,
            cyan_current,
            [
                cyan_current[0] - cyan_previous[0],
                cyan_current[1] - cyan_previous[1],
            ],
            cyan.angle(),
            cyan_quality,
        );

        let (cyan_equation, cyan_rhs) = fixed_point_equation(cyan);
        let previous_cyan_center = self.cyan_center.clone();
        if self.cyan_center.observe(
            cyan_equation,
            cyan_rhs,
            cyan_quality,
            CYAN_CENTER_INFORMATION_DECAY,
            5.0,
            18.0,
        ) {
            if self.cyan_center.estimate.is_some_and(|candidate| {
                (candidate[0] - analysis_center[0] as f64)
                    .hypot(candidate[1] - analysis_center[1] as f64)
                    > frame_extent_px * 18.0
            }) {
                self.cyan_center = previous_cyan_center;
                self.cyan_center.rejected = self.cyan_center.rejected.saturating_add(1);
            }
        }

        let mut latest_green_constraint = None;
        let mut green_center_revision = 0.0;
        if green_valid {
            let green = Affine2::from_motion(motions[PUPIL_LAYER], analysis_center);
            if let (Some(green_inverse), Some(reference_inverse)) =
                (green.inverse(), previous_reference_pose.inverse())
            {
                let green_current = [
                    layers[PUPIL_LAYER].centroid[0] as f64,
                    layers[PUPIL_LAYER].centroid[1] as f64,
                ];
                let green_previous = green_inverse.apply(green_current);
                self.green_history.observe_delta(
                    previous_timestamp_ns,
                    timestamp_ns,
                    green_current,
                    [
                        green_current[0] - green_previous[0],
                        green_current[1] - green_previous[1],
                    ],
                    green.angle(),
                    green_quality,
                );
                let relative = cyan_inverse.compose(green);
                let relative_current = relative.apply(green_previous);
                let relative_quality = cyan_quality.min(green_quality);
                self.relative_history.observe_delta(
                    previous_timestamp_ns,
                    timestamp_ns,
                    [0.0, 0.0],
                    [
                        relative_current[0] - green_previous[0],
                        relative_current[1] - green_previous[1],
                    ],
                    relative.angle(),
                    relative_quality,
                );

                // Conjugate the green-after-cyan residual into the persistent
                // cyan material frame before accumulating its fixed point.
                let relative_in_reference =
                    reference_inverse.compose(relative.compose(previous_reference_pose));
                let (equation, rhs) = fixed_point_equation(relative_in_reference);
                latest_green_constraint = Some((equation, rhs));
                if self.green_center.estimate.is_none() {
                    if let Some(iris_center) = iris_center {
                        let prior = reference_inverse.apply(iris_center);
                        self.green_center
                            .add_prior(prior, iris_radius.unwrap_or(24.0).mul_add(1.5, 4.0));
                    }
                }
                let previous_information = self.green_center.clone();
                let previous_projected = previous_information
                    .estimate
                    .map(|center| current_reference_pose.apply(center));
                let accepted = self.green_center.observe(
                    equation,
                    rhs,
                    relative_quality,
                    CENTER_INFORMATION_DECAY,
                    3.0,
                    (iris_radius.unwrap_or(24.0).max(12.0) * 0.18).clamp(7.0, 14.0),
                );
                if accepted {
                    let candidate_projected = self
                        .green_center
                        .estimate
                        .map(|candidate| current_reference_pose.apply(candidate));
                    green_center_revision = previous_projected
                        .zip(candidate_projected)
                        .map(|(previous, candidate)| {
                            (candidate[0] - previous[0]).hypot(candidate[1] - previous[1])
                        })
                        .unwrap_or(0.0);
                    let radius = iris_radius.unwrap_or(24.0).max(12.0);
                    let implausible = candidate_projected.is_some_and(|projected| {
                        iris_center.is_some_and(|iris| {
                            (projected[0] - iris[0]).hypot(projected[1] - iris[1]) > radius * 1.60
                        })
                    });
                    let broke_rigid_attachment =
                        previous_information.accepted >= 8 && green_center_revision > radius * 0.12;
                    if implausible || broke_rigid_attachment {
                        self.green_center = previous_information;
                        self.green_center.rejected = self.green_center.rejected.saturating_add(1);
                        green_center_revision = 0.0;
                    }
                }
            }
        } else {
            self.green_history.clear();
            self.relative_history.clear();
        }

        self.cyan_from_reference = current_reference_pose;
        let cyan_kinematics = self.cyan_history.derivatives(MotionProfile::Cyan);
        let green_kinematics = self.green_history.derivatives(MotionProfile::Green);
        let relative_kinematics = self.relative_history.derivatives(MotionProfile::Relative);
        let dt = delta_ns as f64 * 1.0e-9;
        let projected_green_center = self
            .green_center
            .estimate
            .map(|center| current_reference_pose.apply(center));
        let center_velocity = match (
            self.last_projected_green_center,
            cyan_transported_green_center,
        ) {
            (Some(previous), Some(transported)) if dt > 1.0e-6 => [
                ((transported[0] - previous[0]) / dt) as f32,
                ((transported[1] - previous[1]) / dt) as f32,
            ],
            _ => [0.0; 2],
        };
        let relative_slip_velocity = match (cyan_transported_green_center, projected_green_center) {
            (Some(transported), Some(current)) if dt > 1.0e-6 => [
                ((current[0] - transported[0]) / dt) as f32,
                ((current[1] - transported[1]) / dt) as f32,
            ],
            _ => [0.0; 2],
        };
        self.last_projected_green_center = projected_green_center;
        let green_sigma = self.green_center.sigma() * current_reference_pose.scale();
        let anatomical_offset = projected_green_center
            .zip(iris_center)
            .map(|(center, iris)| {
                (center[0] - iris[0]).hypot(center[1] - iris[1])
                    / iris_radius.unwrap_or(24.0).max(8.0)
            })
            .unwrap_or(0.0);
        let anatomical_confidence = (-0.5 * (anatomical_offset / 1.15).powi(4)).exp();
        let green_confidence = self
            .green_center
            .confidence(iris_radius.unwrap_or(frame_extent_px * 0.20).max(8.0))
            * anatomical_confidence;
        let green_residual = latest_green_constraint
            .zip(self.green_center.estimate)
            .map(|((equation, rhs), center)| equation_residual(equation, rhs, center))
            .unwrap_or(self.green_center.residual_ema);
        let cyan_sigma = self.cyan_center.sigma();
        let cyan_confidence = self.cyan_center.confidence(frame_extent_px.max(32.0) * 2.0);
        let relative_speed = relative_kinematics.speed_px_s as f64;
        let relative_angular_speed = relative_kinematics.angular_velocity_rad_s.abs() as f64;
        let cyan_speed = cyan_kinematics.speed_px_s as f64;
        // Likelihood describes the measured dynamics. Its independently
        // exported confidence says how strongly to trust the classification.
        let saccade_likelihood = sigmoid((relative_speed - 52.0 - cyan_speed * 0.10) / 13.0)
            .max(sigmoid((relative_angular_speed - 0.30) / 0.09));
        let micro_motion_likelihood = if relative_speed < 24.0 && relative_angular_speed < 0.18 {
            let reliability =
                ((relative_kinematics.confidence as f64 - 0.06) / 0.12).clamp(0.0, 1.0);
            sigmoid((relative_speed - 6.0 - cyan_speed * 0.05) / 3.0) * reliability
        } else {
            0.0
        };
        let relative_motion_fixed_point = RotationCenterStatus {
            projected_center: projected_green_center
                .map(|center| [center[0] as f32, center[1] as f32]),
            transport_velocity_px_s: center_velocity,
            sigma_px: green_sigma.min(f32::MAX as f64) as f32,
            confidence: green_confidence as f32,
            constraint_residual_px: green_residual as f32,
            estimate_revision_px: green_center_revision as f32,
            relative_slip_velocity_px_s: relative_slip_velocity,
            accepted_constraints: self.green_center.accepted,
            rejected_constraints: self.green_center.rejected,
            locked: self.green_center.accepted >= CENTER_LOCK_MIN_SAMPLES
                && green_confidence >= 0.20
                && green_sigma <= iris_radius.unwrap_or(frame_extent_px * 0.20).max(8.0) * 0.65,
        };
        let projected_globe = self.projected_globe.observe(
            dt,
            cyan,
            current_reference_pose,
            iris_geometry,
            self.green_center.estimate,
            self.green_center.sigma(),
            self.green_center.accepted,
            latest_green_constraint,
            relative_kinematics.confidence as f64,
        );
        self.status = CoupledMotionStatus {
            timestamp_ns,
            dt_ms: delta_ns as f32 * 1.0e-6,
            reference_generation: self.reference_generation,
            cyan: cyan_kinematics,
            green: green_kinematics,
            green_relative_to_cyan: relative_kinematics,
            relative_motion_fixed_point,
            green_rotation_center: relative_motion_fixed_point,
            projected_globe,
            cyan_rotation_center: RotationCenterStatus {
                projected_center: self
                    .cyan_center
                    .estimate
                    .map(|center| [center[0] as f32, center[1] as f32]),
                sigma_px: cyan_sigma.min(f32::MAX as f64) as f32,
                confidence: cyan_confidence as f32,
                constraint_residual_px: self.cyan_center.residual_ema as f32,
                accepted_constraints: self.cyan_center.accepted,
                rejected_constraints: self.cyan_center.rejected,
                locked: self.cyan_center.accepted >= CENTER_LOCK_MIN_SAMPLES
                    && cyan_confidence >= 0.10
                    && cyan_sigma <= frame_extent_px * 8.0,
                ..RotationCenterStatus::default()
            },
            saccade_likelihood: saccade_likelihood.clamp(0.0, 1.0) as f32,
            micro_motion_likelihood: micro_motion_likelihood.clamp(0.0, 1.0) as f32,
        };
        self.status
    }

    fn rebase(&mut self, retained_center: Option<[f64; 2]>, radius: f64) {
        let retained_globe_pivot = self
            .projected_globe
            .pivot_reference
            .map(|pivot| self.cyan_from_reference.apply(pivot));
        let retained_globe_pole = self
            .projected_globe
            .pole_reference
            .map(|pole| self.cyan_from_reference.apply_vector(pole));
        self.cyan_from_reference = Affine2::identity();
        self.reference_generation = self.reference_generation.saturating_add(1);
        self.chain_valid = true;
        self.green_center = CenterInformation::default();
        if let Some(center) = retained_center {
            self.green_center.add_prior(center, radius.max(12.0) * 2.0);
        }
        self.projected_globe
            .rebase(retained_globe_pivot, retained_globe_pole, radius);
        self.cyan_center = CenterInformation::default();
        self.cyan_history.clear();
        self.green_history.clear();
        self.relative_history.clear();
        self.status.reference_generation = self.reference_generation;
    }
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn motion_about(center: [f64; 2], angle: f64, translation: [f64; 2]) -> Affine2 {
        let cosine = angle.cos();
        let sine = angle.sin();
        let linear = [[cosine, -sine], [sine, cosine]];
        let rotated = matrix_vector(linear, center);
        Affine2 {
            linear,
            translation: [
                center[0] - rotated[0] + translation[0],
                center[1] - rotated[1] + translation[1],
            ],
        }
    }

    #[test]
    fn stacked_relative_transforms_recover_a_cyan_attached_center() {
        let expected = [122.5, 86.25];
        let mut information = CenterInformation::default();
        information.add_prior([120.0, 88.0], 30.0);
        let mut accepted = 0;
        for index in 0..32 {
            let angle = -0.08 + index as f64 * 0.005;
            let transform = motion_about(expected, angle, [0.0, 0.0]);
            let (equation, rhs) = fixed_point_equation(transform);
            accepted += usize::from(information.observe(equation, rhs, 1.0, 1.0, 2.0, 8.0));
        }
        assert!(accepted >= 30, "accepted={accepted}");
        let actual = information.estimate.unwrap();
        assert!((actual[0] - expected[0]).abs() < 0.05, "x={}", actual[0]);
        assert!((actual[1] - expected[1]).abs() < 0.05, "y={}", actual[1]);
    }

    #[test]
    fn cyan_conjugation_keeps_the_green_center_in_its_material_frame() {
        let reference_center = [108.0, 77.0];
        let cyan_pose = motion_about([30.0, 20.0], 0.08, [7.0, -4.0]);
        let green_relative = motion_about(cyan_pose.apply(reference_center), 0.045, [0.0, 0.0]);
        let reference_inverse = cyan_pose.inverse().unwrap();
        let relative_in_reference = reference_inverse.compose(green_relative.compose(cyan_pose));
        let (equation, rhs) = fixed_point_equation(relative_in_reference);
        let actual = solve_two(
            [
                [
                    equation[0][0] * equation[0][0] + equation[1][0] * equation[1][0],
                    equation[0][0] * equation[0][1] + equation[1][0] * equation[1][1],
                ],
                [
                    equation[0][1] * equation[0][0] + equation[1][1] * equation[1][0],
                    equation[0][1] * equation[0][1] + equation[1][1] * equation[1][1],
                ],
            ],
            [
                equation[0][0] * rhs[0] + equation[1][0] * rhs[1],
                equation[0][1] * rhs[0] + equation[1][1] * rhs[1],
            ],
        )
        .unwrap();
        assert!((actual[0] - reference_center[0]).abs() < 1.0e-6);
        assert!((actual[1] - reference_center[1]).abs() < 1.0e-6);
    }

    #[test]
    fn pure_translation_does_not_invent_a_finite_rotation_center() {
        let transform = Affine2 {
            linear: [[1.0, 0.0], [0.0, 1.0]],
            translation: [6.0, -3.0],
        };
        let (equation, rhs) = fixed_point_equation(transform);
        let mut information = CenterInformation::default();
        assert!(!information.observe(equation, rhs, 1.0, 1.0, 2.0, 8.0));
        assert!(information.estimate.is_none());
    }

    #[test]
    fn cubic_history_reports_first_second_and_third_derivatives() {
        let mut history = PoseHistory::default();
        let start = 1_000_000_000u64;
        let position = |time: f64| 4.0 + 3.0 * time + 2.0 * time * time + time.powi(3);
        for index in 0..8 {
            let timestamp = start + index * 40_000_000;
            let time = index as f64 * 0.04;
            let previous_time = (index.saturating_sub(1)) as f64 * 0.04;
            let delta = if index == 0 {
                0.0
            } else {
                position(time) - position(previous_time)
            };
            if index == 0 {
                history.samples.push_back(PoseSample {
                    timestamp_ns: timestamp,
                    position: [position(time), 0.0],
                    angle: 0.0,
                    quality: 1.0,
                });
            } else {
                history.observe_delta(
                    timestamp - 40_000_000,
                    timestamp,
                    [position(time), 0.0],
                    [delta, 0.0],
                    0.0,
                    1.0,
                );
            }
        }
        let derivatives = history.derivatives(MotionProfile::Cyan);
        let time = 7.0 * 0.04;
        let expected_velocity = 3.0 + 4.0 * time + 3.0 * time * time;
        let expected_acceleration = 4.0 + 6.0 * time;
        assert!((derivatives.velocity_px_s[0] as f64 - expected_velocity).abs() < 0.02);
        assert!((derivatives.acceleration_px_s2[0] as f64 - expected_acceleration).abs() < 0.2);
        assert!((derivatives.jerk_px_s3[0] as f64 - 6.0).abs() < 0.5);
    }

    #[test]
    fn roi_translation_preserves_unobserved_derivative_sentinel() {
        let mut status = CoupledMotionStatus::default();
        status.cyan.reference_point = [4200.0, 3100.0];
        status.cyan.samples = 4;
        status.green.reference_point = [0.0, 0.0];
        status.green.samples = 0;
        status.projected_globe.projected_pivot = Some([4210.0, 3110.0]);
        status.projected_globe.projected_pole = Some([8.0, -3.0]);

        let translated = status.translated(4000.0, 3000.0);
        assert_eq!(translated.cyan.reference_point, [200.0, 100.0]);
        assert_eq!(translated.green.reference_point, [0.0, 0.0]);
        assert_eq!(
            translated.projected_globe.projected_pivot,
            Some([210.0, 110.0])
        );
        assert_eq!(translated.projected_globe.projected_pole, Some([8.0, -3.0]));
    }

    fn authorized_geometry(center: [f64; 2]) -> ProjectedIrisGeometry {
        ProjectedIrisGeometry {
            center,
            major_radius: 52.0,
            minor_radius: 48.0,
            angle_rad: 0.12,
            confidence: 0.90,
            anatomy_authorized: true,
        }
    }

    #[test]
    fn stable_anatomy_and_relative_fixed_point_publish_a_globe_pose() {
        let expected = [120.0, 80.0];
        let transform = motion_about(expected, 0.08, [0.0, 0.0]);
        let constraint = fixed_point_equation(transform);
        let mut filter = ProjectedGlobeFilter::default();
        let mut status = ProjectedGlobePoseStatus::default();
        for _ in 0..8 {
            status = filter.observe(
                0.02,
                Affine2::identity(),
                Affine2::identity(),
                Some(authorized_geometry(expected)),
                Some(expected),
                4.0,
                12,
                Some(constraint),
                0.9,
            );
        }
        assert!(status.publishable, "status={status:?}");
        assert_eq!(status.regime, GlobeMotionRegime::Fixation);
        let pivot = status.projected_pivot.unwrap();
        assert!((pivot[0] as f64 - expected[0]).abs() < 0.1);
        assert!((pivot[1] as f64 - expected[1]).abs() < 0.1);
    }

    #[test]
    fn a_similarity_fixed_point_without_current_anatomy_never_publishes() {
        let expected = [120.0, 80.0];
        let transform = motion_about(expected, 0.08, [0.0, 0.0]);
        let constraint = fixed_point_equation(transform);
        let mut filter = ProjectedGlobeFilter::default();
        let mut geometry = authorized_geometry(expected);
        geometry.anatomy_authorized = false;
        let mut status = ProjectedGlobePoseStatus::default();
        for _ in 0..16 {
            status = filter.observe(
                0.02,
                Affine2::identity(),
                Affine2::identity(),
                Some(geometry),
                Some(expected),
                2.0,
                30,
                Some(constraint),
                1.0,
            );
        }
        assert!(!status.publishable);
        assert!(!status.anatomy_authorized);
        assert_eq!(status.regime, GlobeMotionRegime::Occluded);
    }

    #[test]
    fn corroborated_saccade_moves_the_pole_without_dragging_the_pivot() {
        let expected = [120.0, 80.0];
        let transform = motion_about(expected, 0.08, [0.0, 0.0]);
        let constraint = fixed_point_equation(transform);
        let mut filter = ProjectedGlobeFilter::default();
        for _ in 0..8 {
            filter.observe(
                0.02,
                Affine2::identity(),
                Affine2::identity(),
                Some(authorized_geometry(expected)),
                Some(expected),
                4.0,
                12,
                Some(constraint),
                0.9,
            );
        }
        let before = filter.status.projected_pivot.unwrap();
        let shifted_iris = [138.0, 80.0];
        let after = filter.observe(
            0.02,
            Affine2::identity(),
            Affine2::identity(),
            Some(authorized_geometry(shifted_iris)),
            Some(expected),
            4.0,
            12,
            Some(constraint),
            0.9,
        );
        assert_eq!(after.regime, GlobeMotionRegime::Saccade);
        assert!(after.publishable, "status={after:?}");
        assert!(!after.temporally_bridged);
        let pivot = after.projected_pivot.unwrap();
        assert!(
            (pivot[0] - before[0]).abs() < 0.2,
            "before={before:?} after={after:?}"
        );
        assert!(after.projected_pole.unwrap()[0] > 17.0);
    }

    #[test]
    fn trusted_keyframe_bridges_prediction_consistent_weak_geometry() {
        let expected = [120.0, 80.0];
        let transform = motion_about(expected, 0.08, [0.0, 0.0]);
        let constraint = fixed_point_equation(transform);
        let mut filter = ProjectedGlobeFilter::default();
        for _ in 0..8 {
            filter.observe(
                0.02,
                Affine2::identity(),
                Affine2::identity(),
                Some(authorized_geometry(expected)),
                Some(expected),
                4.0,
                12,
                Some(constraint),
                0.9,
            );
        }
        assert!(filter.status.publishable);

        let mut weak = authorized_geometry([122.0, 80.0]);
        weak.anatomy_authorized = false;
        weak.confidence = 0.25;
        let bridged = filter.observe(
            0.10,
            Affine2::identity(),
            Affine2::identity(),
            Some(weak),
            Some(expected),
            4.0,
            12,
            Some(constraint),
            0.9,
        );
        assert!(bridged.publishable, "status={bridged:?}");
        assert!(bridged.temporally_bridged);
        assert!(!bridged.anatomy_authorized);
        assert!(!bridged.fixed_point_corroborated);
        assert!(bridged.bridge_prediction_disagreement_radii < 0.05);
    }

    #[test]
    fn contradictory_weak_geometry_cuts_the_temporal_bridge_immediately() {
        let expected = [120.0, 80.0];
        let transform = motion_about(expected, 0.08, [0.0, 0.0]);
        let constraint = fixed_point_equation(transform);
        let mut filter = ProjectedGlobeFilter::default();
        for _ in 0..8 {
            filter.observe(
                0.02,
                Affine2::identity(),
                Affine2::identity(),
                Some(authorized_geometry(expected)),
                Some(expected),
                4.0,
                12,
                Some(constraint),
                0.9,
            );
        }
        let mut contradictory = authorized_geometry([142.0, 80.0]);
        contradictory.anatomy_authorized = false;
        let rejected = filter.observe(
            0.02,
            Affine2::identity(),
            Affine2::identity(),
            Some(contradictory),
            Some(expected),
            4.0,
            12,
            Some(constraint),
            0.9,
        );
        assert!(!rejected.publishable, "status={rejected:?}");
        assert!(!rejected.temporally_bridged);
        assert!(rejected.bridge_prediction_disagreement_radii > 0.40);
    }

    #[test]
    fn prediction_consistent_bridge_still_expires_by_elapsed_time() {
        let expected = [120.0, 80.0];
        let transform = motion_about(expected, 0.08, [0.0, 0.0]);
        let constraint = fixed_point_equation(transform);
        let mut filter = ProjectedGlobeFilter::default();
        for _ in 0..8 {
            filter.observe(
                0.02,
                Affine2::identity(),
                Affine2::identity(),
                Some(authorized_geometry(expected)),
                Some(expected),
                4.0,
                12,
                Some(constraint),
                0.9,
            );
        }
        let mut weak = authorized_geometry(expected);
        weak.anatomy_authorized = false;
        let mut status = ProjectedGlobePoseStatus::default();
        for _ in 0..4 {
            status = filter.observe(
                0.20,
                Affine2::identity(),
                Affine2::identity(),
                Some(weak),
                Some(expected),
                4.0,
                12,
                Some(constraint),
                0.9,
            );
        }
        assert!(!status.publishable, "status={status:?}");
        assert!(!status.temporally_bridged);
    }

    #[test]
    fn impossible_material_transport_is_exposed_and_bounded_by_anatomy() {
        let expected = [120.0, 80.0];
        let mut filter = ProjectedGlobeFilter {
            pivot_reference: Some([320.0, 80.0]),
            variance_reference: 20.0f64.powi(2),
            ..ProjectedGlobeFilter::default()
        };
        let mut status = ProjectedGlobePoseStatus::default();
        for _ in 0..16 {
            status = filter.observe(
                0.02,
                Affine2::identity(),
                Affine2::identity(),
                Some(authorized_geometry(expected)),
                None,
                f64::INFINITY,
                0,
                None,
                0.0,
            );
        }
        assert!(
            status.anatomy_disagreement_radii <= 0.91,
            "status={status:?}"
        );
        assert!(filter.rejected_updates > 0);
        assert!(!status.publishable);
    }

    #[test]
    fn invalid_current_frame_does_not_republish_stale_dynamics() {
        let mut tracker = CoupledEyeKinematics::default();
        let mut motions = [SimilarityMotion::default(); super::super::OBJECTS];
        let mut layers = [MotionLayerStatus::default(); super::super::OBJECTS];
        for object in [GENERAL_LAYER, PUPIL_LAYER] {
            motions[object] = SimilarityMotion {
                translation: [1.0, 0.25],
                residual: 0.5,
                support: 8,
                ..SimilarityMotion::default()
            };
            layers[object] = MotionLayerStatus {
                centroid: [120.0, 80.0],
                coherence: 0.9,
                persistent_tracks: 8,
                stable_frames: 4,
                ..MotionLayerStatus::default()
            };
        }

        let start = 1_000_000_000u64;
        for index in 0..5 {
            tracker.observe(
                start + index * 20_000_000,
                [120.0, 80.0],
                384.0,
                motions,
                layers,
                true,
                Some(ProjectedIrisGeometry {
                    center: [120.0, 80.0],
                    major_radius: 52.0,
                    minor_radius: 48.0,
                    angle_rad: 0.0,
                    confidence: 0.9,
                    anatomy_authorized: true,
                }),
            );
        }
        let valid = tracker.status();
        assert!(valid.cyan.samples > 0);
        assert!(valid.green.samples > 0);

        let invalid = tracker.observe(
            start + 5 * 20_000_000,
            [120.0, 80.0],
            384.0,
            [SimilarityMotion::default(); super::super::OBJECTS],
            [MotionLayerStatus::default(); super::super::OBJECTS],
            false,
            None,
        );
        assert_eq!(invalid.cyan.samples, 0);
        assert_eq!(invalid.green.samples, 0);
        assert_eq!(invalid.green_relative_to_cyan.samples, 0);
        assert_eq!(invalid.saccade_likelihood, 0.0);
        assert_eq!(invalid.micro_motion_likelihood, 0.0);
    }
}
