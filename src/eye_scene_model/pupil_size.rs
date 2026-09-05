//! Per-eye pupil/limbus ratio support, optical conditioning and radius continuity.
//!
//! Strong admitted RAW measurements alone train the bounded histories. Soft
//! preferences never widen hard operator support. Frozen automatic recomputation
//! still respects explicit operator edits. Legacy host-Instant timing is kept.

use super::pupil_projection::{PupilProjectionReference, PupilProjectionSource};
use super::pupil_radius_units::{
    AffineForeshortening, FrontoParallelCircleRadiusPx, ProjectedAreaEquivalentRadiusPx,
};
use crate::raw_iris_focus;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

const RADIUS_RATE_LIMITER_MAX_PUBLICATION_STEP: Duration = Duration::from_millis(100);
pub(crate) const PUPIL_SIZE_RATIO_WINDOW: usize = 9;
pub(crate) const PUPIL_SIZE_TEMPORAL_MIN_OBSERVATIONS: usize = 3;
pub(crate) const PUPIL_SIZE_MIN_TEMPORAL_HALF_WIDTH: f64 = 0.05;
pub(crate) const PUPIL_SIZE_MAX_TEMPORAL_HALF_WIDTH: f64 = 0.35;
pub(crate) const PUPIL_SIZE_STALE_EXPANSION_PER_SECOND: f64 = 0.035;
// Apparent fronto-parallel limbus scale is quantized at quarter octaves. This
// is fine enough to keep fixed-pixel pupil cues in one spatial regime while
// avoiding bucket churn from subpixel ellipse jitter.
const PUPIL_EVIDENCE_SCALE_BUCKETS: usize = 16;
pub(crate) const PUPIL_EVIDENCE_SCALE_BASE_RADIUS_PX: f64 = 24.0;
const PUPIL_EVIDENCE_SCALE_BUCKETS_PER_OCTAVE: f64 = 4.0;
const PUPIL_EVIDENCE_MIN_FOCUS_SUPPORT: usize = 6;
pub(crate) const PUPIL_EVIDENCE_MIN_TRAINING_FOCUS_RATIO: f64 = 0.55;
// A relative reference cannot prove that its first observation was sharp.
// Keep a deliberately low absolute floor so a cold start on obvious defocus
// may guide/display a pupil but cannot teach the physical size posterior.
const PUPIL_EVIDENCE_MIN_TRAINING_ABSOLUTE_SHARPNESS: f64 = 0.055;
// The native log/color plane has four-sensor-pixel spacing. A projected pupil
// radius below ten pixels supplies fewer than five samples across its diameter
// and must not teach a size posterior even when an ellipse can be displayed.
const PUPIL_EVIDENCE_MIN_TRAINING_PROJECTED_RADIUS_PX: f64 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RadiusKinematicSupport {
    pub(crate) estimate: f64,
    pub(crate) minimum: f64,
    pub(crate) maximum: f64,
}

#[derive(Debug, Default)]
pub(crate) struct RadiusRateLimiter {
    state: Option<(Instant, f64)>,
}

impl RadiusRateLimiter {
    #[cfg(test)]
    pub(crate) fn admitted_state(&self) -> Option<(Instant, f64)> {
        self.state
    }

    /// Freeze the same continuous-time physiological support used by the
    /// publication limiter so a native-coordinate search can look for an
    /// admissible edge instead of first choosing an impossible edge and then
    /// merely resizing or discarding it. After a one-second evidence gap the
    /// continuous trajectory deliberately expires; robust size history owns
    /// longer reacquisition intervals.
    pub(crate) fn kinematic_support(
        &self,
        now: Instant,
        max_contraction_per_second: f64,
        max_expansion_per_second: f64,
    ) -> Option<RadiusKinematicSupport> {
        let (previous_at, previous_radius) = self.state?;
        if !previous_radius.is_finite() || previous_radius <= 0.0 {
            return None;
        }
        let elapsed_since_observation = now.saturating_duration_since(previous_at);
        if elapsed_since_observation > Duration::from_secs(1) {
            return None;
        }
        // A delayed worker result or one missing segmentation frame is not
        // evidence that an anatomical boundary changed discontinuously.
        // Bound each publication by one ordinary 10 Hz interval; repeated
        // current-RAW edge pressure can still advance the trajectory on
        // subsequent frames. This keeps the pupil's reconstructed circular
        // area below a two-percent change between consecutive publications.
        let elapsed = elapsed_since_observation
            .min(RADIUS_RATE_LIMITER_MAX_PUBLICATION_STEP)
            .as_secs_f64();
        let minimum =
            previous_radius * (1.0 - max_contraction_per_second.max(0.0) * elapsed).max(0.0);
        let maximum = if max_expansion_per_second.is_finite() {
            previous_radius * (1.0 + max_expansion_per_second.max(0.0) * elapsed)
        } else {
            f64::INFINITY
        };
        Some(RadiusKinematicSupport {
            estimate: previous_radius,
            minimum,
            maximum,
        })
    }

    pub(crate) fn constrain(
        &self,
        now: Instant,
        measured_radius: f64,
        max_contraction_per_second: f64,
        max_expansion_per_second: f64,
    ) -> f64 {
        if !measured_radius.is_finite() || measured_radius <= 0.0 {
            return measured_radius;
        }
        let Some(support) =
            self.kinematic_support(now, max_contraction_per_second, max_expansion_per_second)
        else {
            return measured_radius;
        };
        measured_radius.clamp(support.minimum, support.maximum)
    }

    pub(crate) fn observe(
        &mut self,
        now: Instant,
        measured_radius: f64,
        max_contraction_per_second: f64,
        max_expansion_per_second: f64,
    ) -> f64 {
        if !measured_radius.is_finite() || measured_radius <= 0.0 {
            return measured_radius;
        }
        let Some((previous_at, _)) = self.state else {
            self.state = Some((now, measured_radius));
            return measured_radius;
        };
        let elapsed_since_observation = now.saturating_duration_since(previous_at);
        if elapsed_since_observation > Duration::from_secs(1) {
            // This limiter bounds a continuous physiological trajectory. An
            // extended missing-boundary interval is not continuous evidence;
            // the robust hard/soft size supports must validate a fresh fit
            // instead of dragging it toward an indefinitely old ratio.
            self.state = Some((now, measured_radius));
            return measured_radius;
        }
        let stabilized = self.constrain(
            now,
            measured_radius,
            max_contraction_per_second,
            max_expansion_per_second,
        );
        self.state = Some((now, stabilized));
        stabilized
    }

    pub(crate) fn constrain_with_hard_bounds(
        &self,
        now: Instant,
        measured_radius: f64,
        max_contraction_per_second: f64,
        max_expansion_per_second: f64,
        hard_lower: f64,
        hard_upper: f64,
    ) -> (f64, bool) {
        let stabilized = self.constrain(
            now,
            measured_radius,
            max_contraction_per_second,
            max_expansion_per_second,
        );
        let tolerance = 1.0e-9 * measured_radius.abs().max(stabilized.abs()).max(1.0);
        let trajectory_limited = (stabilized - measured_radius).abs() > tolerance;
        let bounded = if hard_lower.is_finite()
            && hard_upper.is_finite()
            && hard_lower > 0.0
            && hard_upper >= hard_lower
        {
            stabilized.clamp(hard_lower, hard_upper)
        } else {
            stabilized
        };
        (bounded, trajectory_limited)
    }

    pub(crate) fn enforce_state_hard_bounds(
        &mut self,
        now: Instant,
        hard_lower: f64,
        hard_upper: f64,
    ) -> bool {
        if !hard_lower.is_finite()
            || !hard_upper.is_finite()
            || hard_lower <= 0.0
            || hard_upper < hard_lower
        {
            return false;
        }
        let Some((_, previous)) = self.state else {
            return false;
        };
        let bounded = previous.clamp(hard_lower, hard_upper);
        let tolerance = 1.0e-9 * previous.abs().max(bounded.abs()).max(1.0);
        if (bounded - previous).abs() <= tolerance {
            return false;
        }
        // This mutation is authorized by an explicit operator guide, not by
        // the current image candidate. It remains valid while R is frozen and
        // cannot let a blurry measurement choose the new trajectory value.
        self.state = Some((now, bounded));
        true
    }

    pub(crate) fn observe_with_hard_bounds(
        &mut self,
        now: Instant,
        measured_radius: f64,
        max_contraction_per_second: f64,
        max_expansion_per_second: f64,
        hard_lower: f64,
        hard_upper: f64,
    ) -> (f64, bool) {
        let (bounded, trajectory_limited) = self.constrain_with_hard_bounds(
            now,
            measured_radius,
            max_contraction_per_second,
            max_expansion_per_second,
            hard_lower,
            hard_upper,
        );
        if bounded.is_finite() && bounded > 0.0 {
            // This is the admitted/mutating path. Explicit operator hard
            // guides therefore restart the trajectory at the published
            // boundary, while the read-only path above never changes state.
            self.state = Some((now, bounded));
        }
        (bounded, trajectory_limited)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PupilSizeSupportSource {
    #[default]
    LimbusFractionThresholds,
    TemporalPupilRatio,
}

impl PupilSizeSupportSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::LimbusFractionThresholds => "limbus-fraction",
            Self::TemporalPupilRatio => "temporal-pupil-ratio",
        }
    }
}

/// Quarter-octave bucket of the fronto-parallel limbus radius. This is an
/// image-resolution coordinate, not an anatomical pupil-size estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrontoParallelScaleBucket(u8);

impl FrontoParallelScaleBucket {
    pub(crate) fn from_radius(radius_px: f64) -> Option<Self> {
        if !radius_px.is_finite() || radius_px <= 0.0 {
            return None;
        }
        let logarithmic = (radius_px / PUPIL_EVIDENCE_SCALE_BASE_RADIUS_PX).log2()
            * PUPIL_EVIDENCE_SCALE_BUCKETS_PER_OCTAVE;
        let index = logarithmic
            .floor()
            .clamp(0.0, (PUPIL_EVIDENCE_SCALE_BUCKETS - 1) as f64) as u8;
        Some(Self(index))
    }

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }

    pub(crate) fn lower_radius_px(self) -> f64 {
        PUPIL_EVIDENCE_SCALE_BASE_RADIUS_PX
            * 2.0f64.powf(self.0 as f64 / PUPIL_EVIDENCE_SCALE_BUCKETS_PER_OCTAVE)
    }

    pub(crate) fn upper_radius_px(self) -> f64 {
        PUPIL_EVIDENCE_SCALE_BASE_RADIUS_PX
            * 2.0f64.powf((self.0 as f64 + 1.0) / PUPIL_EVIDENCE_SCALE_BUCKETS_PER_OCTAVE)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum OpticalFocusClass {
    #[default]
    Unknown,
    Soft,
    Usable,
    Sharp,
}

impl OpticalFocusClass {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Soft => "soft",
            Self::Usable => "usable",
            Self::Sharp => "sharp",
        }
    }
}

/// Independent coordinates which determine how much authority current-frame
/// pupil cues receive. Apparent scale controls spatial resolvability; optical
/// focus controls edge/color reliability. Neither one changes the physical
/// pupil/limbus ratio being estimated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PupilEvidenceCondition {
    pub(crate) scale_bucket: Option<FrontoParallelScaleBucket>,
    pub(crate) limbus_fronto_parallel_radius_px: Option<f64>,
    pub(crate) expected_projected_pupil_radius_px: Option<f64>,
    pub(crate) limbus_optical_sharpness: Option<f64>,
    pub(crate) limbus_focus_reference: Option<f64>,
    pub(crate) relative_focus: Option<f64>,
    pub(crate) focus_measurement_confidence: f64,
    pub(crate) spatial_resolution: f64,
    pub(crate) cue_reliability: f64,
    pub(crate) focus_verified: bool,
    pub(crate) focus_class: OpticalFocusClass,
}

impl PupilEvidenceCondition {
    pub(crate) fn fully_reliable(scale_radius_px: f64) -> Self {
        let scale_bucket = FrontoParallelScaleBucket::from_radius(scale_radius_px);
        Self {
            scale_bucket,
            limbus_fronto_parallel_radius_px: Some(scale_radius_px),
            expected_projected_pupil_radius_px: Some(scale_radius_px * 0.40),
            limbus_optical_sharpness: Some(1.0),
            limbus_focus_reference: Some(1.0),
            relative_focus: Some(1.0),
            focus_measurement_confidence: 1.0,
            spatial_resolution: 1.0,
            cue_reliability: 1.0,
            focus_verified: true,
            focus_class: OpticalFocusClass::Sharp,
        }
    }

    pub(crate) fn raw_solver_condition(self) -> raw_iris_focus::InnerIrisEvidenceCondition {
        raw_iris_focus::InnerIrisEvidenceCondition::new(self.cue_reliability)
    }

    pub(crate) fn projected_boundary_radius_px(
        boundary: &raw_iris_focus::InnerIrisBoundary,
    ) -> Option<f64> {
        let major = boundary.major_radius;
        let minor = boundary.minor_radius;
        if major.is_finite() && minor.is_finite() && major > 0.0 && minor > 0.0 {
            Some((major * minor).sqrt())
        } else if boundary.radius.is_finite() && boundary.radius > 0.0 {
            Some(boundary.radius)
        } else {
            None
        }
    }

    pub(crate) fn permits_posterior_training(
        self,
        boundary: &raw_iris_focus::InnerIrisBoundary,
    ) -> bool {
        self.focus_verified
            && self.scale_bucket.is_some()
            && self.limbus_optical_sharpness.is_some_and(|sharpness| {
                sharpness.is_finite() && sharpness >= PUPIL_EVIDENCE_MIN_TRAINING_ABSOLUTE_SHARPNESS
            })
            && self.relative_focus.is_some_and(|ratio| {
                ratio.is_finite() && ratio >= PUPIL_EVIDENCE_MIN_TRAINING_FOCUS_RATIO
            })
            && self.focus_measurement_confidence >= 0.25
            && Self::projected_boundary_radius_px(boundary)
                .is_some_and(|radius| radius >= PUPIL_EVIDENCE_MIN_TRAINING_PROJECTED_RADIUS_PX)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ScaleConditionedFocusReference {
    smoothed_sharpness: f64,
    sharp_reference: f64,
    observations: u32,
    last_observation: Option<Instant>,
}

#[derive(Default)]
pub(crate) struct PupilEvidenceConditionTracker {
    focus_by_scale: [ScaleConditionedFocusReference; PUPIL_EVIDENCE_SCALE_BUCKETS],
}

fn smooth_unit_interval(value: f64, lower: f64, upper: f64) -> f64 {
    if !value.is_finite() || upper <= lower {
        return 0.0;
    }
    let unit = ((value - lower) / (upper - lower)).clamp(0.0, 1.0);
    unit * unit * (3.0 - 2.0 * unit)
}

impl PupilEvidenceConditionTracker {
    pub(crate) fn observe(
        &mut self,
        now: Instant,
        geometry: Option<PupilSizeGeometry>,
        optical_focus: raw_iris_focus::LimbusOpticalFocus,
        focus_verified: bool,
    ) -> PupilEvidenceCondition {
        let scale_radius =
            geometry.map(|value| value.reference_limbus_fronto_parallel_radius_px.value());
        let scale_bucket = scale_radius.and_then(FrontoParallelScaleBucket::from_radius);
        let expected_projected_pupil_radius = geometry.map(|value| {
            // Radius uncertainty is multiplicative, and the learned posterior
            // itself lives in log-ratio space. Before temporal evidence is
            // warm, use the geometric (log-space) center of the operator's
            // broad hard interval. An arithmetic midpoint would assume a
            // conspicuously large pupil and over-trust fixed-pixel detail on
            // a genuinely small pupil.
            let fronto_parallel = value
                .estimated_fronto_parallel_radius_px
                .unwrap_or_else(|| {
                    (value.lower_fronto_parallel_radius_px * value.upper_fronto_parallel_radius_px)
                        .sqrt()
                });
            fronto_parallel
                * value
                    .projection_minor_to_major
                    .clamp(
                        crate::conic_solver::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE
                            .absolute_minimum_minor_to_major,
                        1.0,
                    )
                    .sqrt()
        });
        let spatial_resolution = expected_projected_pupil_radius
            .map_or(0.0, |radius| smooth_unit_interval(radius, 8.0, 20.0));
        let measured_sharpness = (optical_focus.support >= PUPIL_EVIDENCE_MIN_FOCUS_SUPPORT
            && optical_focus.sharpness.is_finite()
            && optical_focus.sharpness > 0.0)
            .then_some(optical_focus.sharpness);
        let support_confidence = smooth_unit_interval(optical_focus.support as f64, 5.0, 20.0);
        let contrast_confidence = smooth_unit_interval(optical_focus.contrast_raw10, 4.0, 36.0);
        let focus_measurement_confidence = (support_confidence * contrast_confidence).sqrt();

        let mut reference = None;
        let mut relative_focus = None;
        if let (Some(bucket), Some(sharpness)) = (scale_bucket, measured_sharpness) {
            let state = &mut self.focus_by_scale[bucket.index()];
            if focus_verified && focus_measurement_confidence >= 0.20 {
                if state.observations == 0 {
                    state.smoothed_sharpness = sharpness;
                    state.sharp_reference = sharpness;
                } else {
                    state.smoothed_sharpness = 0.72 * state.smoothed_sharpness + 0.28 * sharpness;
                    if state.smoothed_sharpness > state.sharp_reference {
                        // A newly resolved edge raises the reference
                        // deliberately, but one anomalous frame cannot replace
                        // it outright.
                        state.sharp_reference =
                            0.75 * state.sharp_reference + 0.25 * state.smoothed_sharpness;
                    } else {
                        // Subject/lighting changes may lower the attainable
                        // concentration over minutes. Defocus over a few
                        // frames must not teach itself as the new "sharp".
                        let elapsed = state.last_observation.map_or(0.0, |last| {
                            now.saturating_duration_since(last).as_secs_f64()
                        });
                        let downward_alpha = (elapsed * 0.0005).clamp(0.0, 0.01);
                        state.sharp_reference = state.sharp_reference * (1.0 - downward_alpha)
                            + state.smoothed_sharpness * downward_alpha;
                    }
                }
                state.observations = state.observations.saturating_add(1);
                state.last_observation = Some(now);
            }
            if state.observations > 0 && state.sharp_reference > f64::EPSILON {
                reference = Some(state.sharp_reference);
                // Condition this frame from this frame's edge concentration.
                // The EMA exists only to evolve the long-lived sharp
                // reference; using it here would leave several blurry frames
                // at full detail authority after a sudden focus loss.
                relative_focus = Some((sharpness / state.sharp_reference).clamp(0.0, 1.25));
            }
        }

        let absolute_focus =
            measured_sharpness.map_or(0.0, |sharpness| smooth_unit_interval(sharpness, 0.04, 0.28));
        let focus_reliability = relative_focus.map_or(0.55 * absolute_focus, |relative| {
            (0.72 * relative.clamp(0.0, 1.0) + 0.28 * absolute_focus).clamp(0.0, 1.0)
        });
        let cue_reliability =
            (spatial_resolution * focus_measurement_confidence * focus_reliability)
                .sqrt()
                .clamp(0.0, 1.0);
        let focus_class = if measured_sharpness.is_none() || relative_focus.is_none() {
            OpticalFocusClass::Unknown
        } else if measured_sharpness.unwrap() < PUPIL_EVIDENCE_MIN_TRAINING_ABSOLUTE_SHARPNESS {
            OpticalFocusClass::Soft
        } else if relative_focus.unwrap() < PUPIL_EVIDENCE_MIN_TRAINING_FOCUS_RATIO {
            OpticalFocusClass::Soft
        } else if relative_focus.unwrap() < 0.82 {
            OpticalFocusClass::Usable
        } else {
            OpticalFocusClass::Sharp
        };

        PupilEvidenceCondition {
            scale_bucket,
            limbus_fronto_parallel_radius_px: scale_radius,
            expected_projected_pupil_radius_px: expected_projected_pupil_radius,
            limbus_optical_sharpness: measured_sharpness,
            limbus_focus_reference: reference,
            relative_focus,
            focus_measurement_confidence,
            spatial_resolution,
            cue_reliability,
            focus_verified,
            focus_class,
        }
    }
}

/// Frozen per-frame solver support, also consumed by presentation adapters.
/// Hard operator bounds, soft temporal preferences and admission metadata are
/// deliberately distinct. This is not an independently observed pupil or a
/// calibrated confidence interval; rendering it must not train the posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PupilSizeSupport {
    pub(crate) center: (f64, f64),
    /// Center of the current affine limbus projection selected independently
    /// of Y. This remains distinct from the Y-owned rough/search center above.
    pub(crate) projection_center: (f64, f64),
    pub(crate) lower_fronto_parallel_radius_px: f64,
    pub(crate) upper_fronto_parallel_radius_px: f64,
    pub(crate) preferred_lower_fronto_parallel_radius_px: Option<f64>,
    pub(crate) preferred_upper_fronto_parallel_radius_px: Option<f64>,
    pub(crate) estimated_fronto_parallel_radius_px: Option<f64>,
    pub(crate) reference_limbus_fronto_parallel_radius_px: FrontoParallelCircleRadiusPx,
    pub(crate) projection_minor_to_major: f64,
    pub(crate) projection_angle: f64,
    pub(crate) projection_source: PupilProjectionSource,
    pub(crate) support_source: PupilSizeSupportSource,
    pub(crate) scale_bucket: FrontoParallelScaleBucket,
    pub(crate) temporal_scale_matched: bool,
    pub(crate) temporal_observations: usize,
    pub(crate) temporal_fractional_half_width: Option<f64>,
    pub(crate) recomputed_this_frame: bool,
    pub(crate) frozen: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PupilSizeObservationAdmission {
    pub(crate) rate_limited: bool,
    pub(crate) focus_size_qualified: bool,
    pub(crate) limbus_geometry_qualified: bool,
    pub(crate) raw_diameter_qualified: bool,
    pub(crate) trajectory_updated: bool,
    pub(crate) trained_posterior: bool,
}

impl PupilSizeObservationAdmission {
    pub(crate) fn current_boundary_publishable(self) -> bool {
        self.focus_size_qualified && self.limbus_geometry_qualified && self.raw_diameter_qualified
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PupilSizeGeometry {
    pub(crate) projection_center: (f64, f64),
    pub(crate) lower_fronto_parallel_radius_px: f64,
    pub(crate) upper_fronto_parallel_radius_px: f64,
    pub(crate) preferred_lower_fronto_parallel_radius_px: Option<f64>,
    pub(crate) preferred_upper_fronto_parallel_radius_px: Option<f64>,
    pub(crate) estimated_fronto_parallel_radius_px: Option<f64>,
    pub(crate) reference_limbus_fronto_parallel_radius_px: FrontoParallelCircleRadiusPx,
    pub(crate) projection_minor_to_major: f64,
    pub(crate) projection_angle: f64,
    pub(crate) projection_source: PupilProjectionSource,
    pub(crate) support_source: PupilSizeSupportSource,
    pub(crate) scale_bucket: FrontoParallelScaleBucket,
    pub(crate) temporal_scale_matched: bool,
    pub(crate) temporal_observations: usize,
    pub(crate) temporal_fractional_half_width: Option<f64>,
}

pub(crate) struct PupilSizeTracker {
    /// Robust history of the physical pupil/limbus radius ratio. Both radii
    /// are measured after fronto-parallel rectification, so gross image-scale
    /// changes cancel without coupling this state to MediaPipe or a specific
    /// outer-iris detector.
    strong_log_ratios: VecDeque<f64>,
    last_strong_observation: Option<Instant>,
    scale_log_ratios: [VecDeque<f64>; PUPIL_EVIDENCE_SCALE_BUCKETS],
    scale_last_strong_observation: [Option<Instant>; PUPIL_EVIDENCE_SCALE_BUCKETS],
    cached_geometry: Option<PupilSizeGeometry>,
}

impl Default for PupilSizeTracker {
    fn default() -> Self {
        Self {
            strong_log_ratios: VecDeque::new(),
            last_strong_observation: None,
            scale_log_ratios: std::array::from_fn(|_| VecDeque::new()),
            scale_last_strong_observation: [None; PUPIL_EVIDENCE_SCALE_BUCKETS],
            cached_geometry: None,
        }
    }
}

pub(crate) fn pupil_size_median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

fn pupil_temporal_geometry(
    reference_radius: f64,
    hard_lower: f64,
    hard_upper: f64,
    temporal: Option<(f64, f64)>,
) -> (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    PupilSizeSupportSource,
    Option<f64>,
) {
    let Some((ratio, half_width)) = temporal else {
        return (
            None,
            None,
            None,
            PupilSizeSupportSource::LimbusFractionThresholds,
            None,
        );
    };
    let estimate = reference_radius * ratio;
    if !estimate.is_finite() || !half_width.is_finite() {
        return (
            None,
            None,
            None,
            PupilSizeSupportSource::LimbusFractionThresholds,
            None,
        );
    }
    let preferred_lower = hard_lower.max(estimate * (1.0 - half_width));
    let preferred_upper = hard_upper.min(estimate * (1.0 + half_width));
    if preferred_lower.is_finite()
        && preferred_upper.is_finite()
        && preferred_upper > preferred_lower + 0.5
    {
        (
            Some(preferred_lower),
            Some(preferred_upper),
            Some(estimate),
            PupilSizeSupportSource::TemporalPupilRatio,
            Some(half_width),
        )
    } else {
        // Keep drawing/reporting the posterior estimate when the operator's
        // hard interval excludes it, but do not let it steer the search.
        (
            None,
            None,
            Some(estimate),
            PupilSizeSupportSource::LimbusFractionThresholds,
            Some(half_width),
        )
    }
}

impl PupilSizeTracker {
    #[cfg(test)]
    pub(crate) fn admitted_log_ratios(&self) -> &VecDeque<f64> {
        &self.strong_log_ratios
    }

    pub(crate) fn active_geometry(&self) -> Option<PupilSizeGeometry> {
        self.cached_geometry
    }

    pub(crate) fn robust_temporal_ratio_support(
        samples: &VecDeque<f64>,
        last_observation: Option<Instant>,
        now: Instant,
    ) -> Option<(f64, f64)> {
        if samples.len() < PUPIL_SIZE_TEMPORAL_MIN_OBSERVATIONS {
            return None;
        }
        let center = pupil_size_median(samples.iter().copied().collect())?;
        let mad = pupil_size_median(
            samples
                .iter()
                .map(|sample| (sample - center).abs())
                .collect(),
        )?;
        let observations = samples.len() as f64;
        // Three agreeing native boundaries are enough to prune a grossly
        // different radial branch, but not enough to claim sub-pixel size.
        // Converge from about six percent at N=3 to the five-percent floor;
        // the previous 8-11% interval still admitted conspicuous alternate
        // lid/glasses solutions before the physiological limiter could act.
        let sampling_half_width = 0.035 + 0.045 / observations.sqrt();
        let residual_half_width = (2.8 * mad).exp() - 1.0;
        let stale_half_width = last_observation.map_or(0.0, |last| {
            now.saturating_duration_since(last).as_secs_f64()
                * PUPIL_SIZE_STALE_EXPANSION_PER_SECOND
        });
        let fractional_half_width = sampling_half_width
            .max(residual_half_width)
            .max(stale_half_width)
            .clamp(
                PUPIL_SIZE_MIN_TEMPORAL_HALF_WIDTH,
                PUPIL_SIZE_MAX_TEMPORAL_HALF_WIDTH,
            );
        Some((center.exp(), fractional_half_width))
    }

    fn temporal_ratio_support(
        &self,
        now: Instant,
        scale_bucket: FrontoParallelScaleBucket,
    ) -> (Option<(f64, f64)>, usize, bool) {
        let index = scale_bucket.index();
        let scale_samples = &self.scale_log_ratios[index];
        if let Some(support) = Self::robust_temporal_ratio_support(
            scale_samples,
            self.scale_last_strong_observation[index],
            now,
        ) {
            return (Some(support), scale_samples.len(), true);
        }
        (
            Self::robust_temporal_ratio_support(
                &self.strong_log_ratios,
                self.last_strong_observation,
                now,
            ),
            self.strong_log_ratios.len(),
            false,
        )
    }

    pub(crate) fn begin_frame(
        &mut self,
        now: Instant,
        center: Option<(f64, f64)>,
        projection: Option<PupilProjectionReference>,
        recompute: bool,
        lower_fraction: f64,
        upper_fraction: f64,
    ) -> Option<PupilSizeSupport> {
        // Affine limbus projection geometry remains current even while the
        // selected Y mechanism temporarily has no center. Center availability
        // controls only whether a reticle/search support can be emitted below.
        let center = center.filter(|value| value.0.is_finite() && value.1.is_finite());
        let mut recomputed_this_frame = false;
        if recompute {
            if let Some(projection) = projection {
                let reference = projection.fronto_parallel_limbus_radius_px;
                let reference_px = reference.value();
                if let Some(scale_bucket) = FrontoParallelScaleBucket::from_radius(reference_px) {
                    let hard_lower = reference_px * lower_fraction.clamp(0.01, 0.90);
                    let hard_upper = reference_px * upper_fraction.clamp(0.02, 0.95);
                    let (temporal_support, temporal_observations, temporal_scale_matched) =
                        self.temporal_ratio_support(now, scale_bucket);
                    let (
                        preferred_lower,
                        preferred_upper,
                        estimate,
                        support_source,
                        temporal_half_width,
                    ) = pupil_temporal_geometry(
                        reference_px,
                        hard_lower,
                        hard_upper,
                        temporal_support,
                    );
                    if hard_lower.is_finite() && hard_upper.is_finite() && hard_upper > hard_lower {
                        self.cached_geometry = Some(PupilSizeGeometry {
                            projection_center: projection.center,
                            lower_fronto_parallel_radius_px: hard_lower,
                            upper_fronto_parallel_radius_px: hard_upper,
                            preferred_lower_fronto_parallel_radius_px: preferred_lower,
                            preferred_upper_fronto_parallel_radius_px: preferred_upper,
                            estimated_fronto_parallel_radius_px: estimate,
                            reference_limbus_fronto_parallel_radius_px: reference,
                            projection_minor_to_major: projection.minor_to_major,
                            projection_angle: projection.angle,
                            projection_source: projection.source,
                            support_source,
                            scale_bucket,
                            temporal_scale_matched,
                            temporal_observations,
                            temporal_fractional_half_width: temporal_half_width,
                        });
                        recomputed_this_frame = true;
                    }
                }
            }
        } else if let Some(mut geometry) = self.cached_geometry {
            // R freezes automatic transport from new limbus fits, not an
            // explicit operator adjustment. Move the hard guides immediately
            // against the last frozen reference when a threshold key changes.
            let reference = geometry.reference_limbus_fronto_parallel_radius_px;
            let reference_px = reference.value();
            let hard_lower = reference_px * lower_fraction.clamp(0.01, 0.90);
            let hard_upper = reference_px * upper_fraction.clamp(0.02, 0.95);
            if hard_lower.is_finite()
                && hard_upper.is_finite()
                && hard_upper > hard_lower
                && ((hard_lower - geometry.lower_fronto_parallel_radius_px).abs() > f64::EPSILON
                    || (hard_upper - geometry.upper_fronto_parallel_radius_px).abs() > f64::EPSILON)
            {
                let frozen_temporal = geometry
                    .estimated_fronto_parallel_radius_px
                    .zip(geometry.temporal_fractional_half_width)
                    .map(|(estimate, half_width)| (estimate / reference_px, half_width));
                let (
                    preferred_lower,
                    preferred_upper,
                    estimate,
                    support_source,
                    temporal_half_width,
                ) = pupil_temporal_geometry(reference_px, hard_lower, hard_upper, frozen_temporal);
                geometry.lower_fronto_parallel_radius_px = hard_lower;
                geometry.upper_fronto_parallel_radius_px = hard_upper;
                geometry.preferred_lower_fronto_parallel_radius_px = preferred_lower;
                geometry.preferred_upper_fronto_parallel_radius_px = preferred_upper;
                geometry.estimated_fronto_parallel_radius_px = estimate;
                geometry.support_source = support_source;
                geometry.temporal_fractional_half_width = temporal_half_width;
                self.cached_geometry = Some(geometry);
            }
        }
        let geometry = self.cached_geometry?;
        let center = center?;
        Some(PupilSizeSupport {
            center,
            projection_center: geometry.projection_center,
            lower_fronto_parallel_radius_px: geometry.lower_fronto_parallel_radius_px,
            upper_fronto_parallel_radius_px: geometry.upper_fronto_parallel_radius_px,
            preferred_lower_fronto_parallel_radius_px: geometry
                .preferred_lower_fronto_parallel_radius_px,
            preferred_upper_fronto_parallel_radius_px: geometry
                .preferred_upper_fronto_parallel_radius_px,
            estimated_fronto_parallel_radius_px: geometry.estimated_fronto_parallel_radius_px,
            reference_limbus_fronto_parallel_radius_px: geometry
                .reference_limbus_fronto_parallel_radius_px,
            projection_minor_to_major: geometry.projection_minor_to_major,
            projection_angle: geometry.projection_angle,
            projection_source: geometry.projection_source,
            support_source: geometry.support_source,
            scale_bucket: geometry.scale_bucket,
            temporal_scale_matched: geometry.temporal_scale_matched,
            temporal_observations: geometry.temporal_observations,
            temporal_fractional_half_width: geometry.temporal_fractional_half_width,
            recomputed_this_frame,
            frozen: !recompute,
        })
    }

    pub(crate) fn observe_strong_boundary(
        &mut self,
        now: Instant,
        boundary: &raw_iris_focus::InnerIrisBoundary,
        support: Option<PupilSizeSupport>,
        confidence: f64,
        recompute: bool,
        persistent_state_qualified: bool,
    ) -> bool {
        if !recompute || confidence < 0.25 || !persistent_state_qualified {
            return false;
        }
        let Some(support) = support else {
            return false;
        };
        let Some(radius) = fronto_parallel_area_equivalent_pupil_radius_px(boundary, Some(support))
        else {
            return false;
        };
        if !pupil_radius_within_bounds(radius.value(), Some(support)) {
            return false;
        }
        let Some(ratio) = radius
            .ratio_to(support.reference_limbus_fronto_parallel_radius_px)
            .map(|ratio| ratio.value())
        else {
            return false;
        };
        if !ratio.is_finite() || !(0.01..=0.95).contains(&ratio) {
            return false;
        }
        self.strong_log_ratios.push_back(ratio.ln());
        while self.strong_log_ratios.len() > PUPIL_SIZE_RATIO_WINDOW {
            self.strong_log_ratios.pop_front();
        }
        self.last_strong_observation = Some(now);
        let scale_index = support.scale_bucket.index();
        self.scale_log_ratios[scale_index].push_back(ratio.ln());
        while self.scale_log_ratios[scale_index].len() > PUPIL_SIZE_RATIO_WINDOW {
            self.scale_log_ratios[scale_index].pop_front();
        }
        self.scale_last_strong_observation[scale_index] = Some(now);
        true
    }
}

pub(crate) fn projected_pupil_major_radius_px(
    boundary: &raw_iris_focus::InnerIrisBoundary,
) -> Option<f64> {
    let ellipse_radius = boundary.major_radius.max(boundary.minor_radius);
    if ellipse_radius.is_finite() && ellipse_radius > 0.0 {
        Some(ellipse_radius)
    } else if boundary.radius.is_finite() && boundary.radius > 0.0 {
        Some(boundary.radius)
    } else {
        None
    }
}

/// Radius of the equal-area pupil circle after undoing the affine
/// foreshortening measured from the selected limbus.
///
/// If the projected pupil ellipse has semi-axes `a_p`, `b_p` and the limbus
/// projection has `q = b_l / a_l`, inverse rectification multiplies area by
/// `1 / q`. Therefore `r = sqrt(a_p * b_p / q)`. This remains meaningful when
/// partial occlusion makes the fitted pupil axes depart slightly from the
/// limbus aspect ratio, whereas taking only the pupil major axis does not.
pub(crate) fn fronto_parallel_area_equivalent_pupil_radius_px(
    boundary: &raw_iris_focus::InnerIrisBoundary,
    support: Option<PupilSizeSupport>,
) -> Option<FrontoParallelCircleRadiusPx> {
    let support = support?;
    let major = boundary.major_radius.max(boundary.minor_radius);
    let minor = boundary.major_radius.min(boundary.minor_radius);
    let projected = ProjectedAreaEquivalentRadiusPx::from_ellipse_axes(major, minor)?;
    let foreshortening =
        AffineForeshortening::from_minor_to_major(support.projection_minor_to_major)?;
    FrontoParallelCircleRadiusPx::from_projected_area(projected, foreshortening)
}

pub(crate) fn pupil_radius_within_bounds(radius: f64, bounds: Option<PupilSizeSupport>) -> bool {
    radius.is_finite()
        && radius > 0.0
        && bounds.is_some_and(|bounds| {
            radius >= bounds.lower_fronto_parallel_radius_px
                && radius <= bounds.upper_fronto_parallel_radius_px
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soft_temporal_preference_cannot_widen_hard_support() {
        let inside = pupil_temporal_geometry(100.0, 8.0, 72.0, Some((0.40, 0.10)));
        assert_eq!(inside.0, Some(36.0));
        assert_eq!(inside.1, Some(44.0));
        assert_eq!(inside.3, PupilSizeSupportSource::TemporalPupilRatio);
        let excluded = pupil_temporal_geometry(100.0, 8.0, 72.0, Some((0.90, 0.10)));
        assert_eq!((excluded.0, excluded.1), (None, None));
        assert_eq!(
            excluded.2,
            Some(90.0),
            "an excluded estimate remains diagnostic only"
        );
        assert_eq!(excluded.3, PupilSizeSupportSource::LimbusFractionThresholds);
    }

    #[test]
    fn rejected_size_candidates_leave_every_temporal_history_unchanged() {
        let now = Instant::now();
        let projection = PupilProjectionReference::from_axes(
            (100.0, 80.0),
            100.0,
            80.0,
            0.0,
            PupilProjectionSource::SelectedIris,
        )
        .unwrap();
        let mut tracker = PupilSizeTracker::default();
        let support = tracker.begin_frame(
            now,
            Some(projection.center),
            Some(projection),
            true,
            0.08,
            0.72,
        );
        let boundary = raw_iris_focus::InnerIrisBoundary {
            center: projection.center,
            major_radius: 30.0,
            minor_radius: 24.0,
            radius: (30.0f64 * 24.0).sqrt(),
            ..raw_iris_focus::InnerIrisBoundary::default()
        };
        assert!(tracker.observe_strong_boundary(now, &boundary, support, 0.9, true, true));
        let histories = (
            tracker.strong_log_ratios.clone(),
            tracker.scale_log_ratios.clone(),
            tracker.last_strong_observation,
            tracker.scale_last_strong_observation,
            tracker.cached_geometry,
        );
        let next = now + Duration::from_millis(100);
        for (confidence, recompute, qualified) in
            [(0.2, true, true), (0.9, false, true), (0.9, true, false)]
        {
            assert!(!tracker.observe_strong_boundary(
                next, &boundary, support, confidence, recompute, qualified
            ));
        }
        assert!(!tracker.observe_strong_boundary(next, &boundary, None, 0.9, true, true));
        let outside = raw_iris_focus::InnerIrisBoundary {
            major_radius: 90.0,
            minor_radius: 72.0,
            ..boundary
        };
        assert!(!tracker.observe_strong_boundary(next, &outside, support, 0.9, true, true));
        assert_eq!(
            histories,
            (
                tracker.strong_log_ratios,
                tracker.scale_log_ratios,
                tracker.last_strong_observation,
                tracker.scale_last_strong_observation,
                tracker.cached_geometry,
            )
        );
    }
}
