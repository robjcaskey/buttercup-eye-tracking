//! Per-eye center transport, ring admission, bounded relocation and pursuit.
//!
//! This preserves the legacy host-Instant state machine and every admission
//! threshold. A transported hold is not a fresh measurement. Effective limbus
//! coordinates are a local approximation, not a rigid anatomical pivot.

use super::pupil_projection::{
    pupil_projection_canonical_point, pupil_projection_image_point, PupilProjectionReference,
};
use super::CoupledMotionStatus;
use crate::raw_iris_focus;
use crate::roi_evidence::{MotionLayerStatus, SimilarityMotion};
use std::time::{Duration, Instant};

/// Read-only motion inputs for a single current ROI/exposure. The caller owns
/// frame alignment and sensor origin; this adapter does not invent a clock.
/// Borrow only the consumed state, never images, display nodes or trails.
/// The reflection layer cannot independently authorize a pupil relocation.
#[derive(Clone, Copy)]
pub(crate) struct PupilCenterMotionEvidence<'a> {
    pub(crate) global_motion: &'a SimilarityMotion,
    pub(crate) global_layer: &'a MotionLayerStatus,
    pub(crate) pupil_layer: &'a MotionLayerStatus,
    pub(crate) coupled_motion: &'a CoupledMotionStatus,
}

const PUPIL_CENTER_TRACK_STALE_AFTER: Duration = Duration::from_millis(1_250);
pub(crate) const PUPIL_CENTER_PENDING_RELOCATION_MAX_AGE: Duration = Duration::from_millis(450);
// A tracked fixation ball produces smooth pursuit rather than a sequence of
// stationary fixations.  At the native 10 Hz eye cadence, requiring the
// second remote pupil ring to remain at the first ring's sensor coordinate
// turns real pursuit into a repeated relocation-pending hold.  Two decisive
// RAW rings may instead establish a short-lived canonical velocity when both
// increments are aligned and remain a small fraction of the measured limbus.
// The prediction expires quickly, so a lid/glint edge cannot coast through an
// evidence gap or become a new acquisition mechanism.
const PUPIL_CENTER_PURSUIT_MAX_AGE: Duration = Duration::from_millis(350);
const PUPIL_CENTER_PURSUIT_MIN_CANONICAL_STEP: f64 = 0.004;
const PUPIL_CENTER_PURSUIT_MAX_CANONICAL_STEP: f64 = 0.105;
const PUPIL_CENTER_PURSUIT_MIN_DIRECTION_COSINE: f64 = 0.35;
const PUPIL_CENTER_PURSUIT_MIN_STEP_RATIO: f64 = 0.22;
const PUPIL_CENTER_PURSUIT_MAX_STEP_RATIO: f64 = 4.50;
const PUPIL_CENTER_PURSUIT_MAX_CANONICAL_SPEED_PER_SECOND: f64 = 1.10;
// Without independently classified saccade motion, rough-center/RAW flow may
// repair a lagging pursuit but may not teleport the pupil across half of the
// limbus.  Large genuine saccades retain the separate high-jerk path.
pub(crate) const PUPIL_CENTER_UNSUPPORTED_MAX_CANONICAL_RELOCATION: f64 = 0.24;

/// Motion regime for the shared pupil-center state.  This is deliberately not
/// a constant-velocity filter: a real saccade has large acceleration and jerk,
/// then stops abruptly.  Fixations are therefore transported in the limbus
/// coordinate frame, while a remote current-frame solution is either admitted
/// by independent pupil/iris motion or verified on a second frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PupilCenterMotionRegime {
    #[default]
    Uninitialized,
    Fixation,
    RelocationPending,
    Saccade,
    SmoothPursuit,
}

impl PupilCenterMotionRegime {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Uninitialized => "uninitialized",
            Self::Fixation => "fixation",
            Self::RelocationPending => "relocation-pending",
            Self::Saccade => "saccade",
            Self::SmoothPursuit => "smooth-pursuit",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PupilCenterPrediction {
    pub(crate) center: (f64, f64),
    pub(crate) transported_track: bool,
    pub(crate) transport_source: PupilCenterTransportSource,
    pub(crate) limbus_transport_disagreement_px: f64,
    pub(crate) pursuit_predicted: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PupilCenterTransportSource {
    #[default]
    Uninitialized,
    FrameProposal,
    SensorHold,
    GlobalSimilarity,
    LimbusConsensus,
    LimbusOnly,
    LimbusReacquisition,
}

impl PupilCenterTransportSource {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Uninitialized => "uninitialized",
            Self::FrameProposal => "frame-proposal",
            Self::SensorHold => "sensor-hold",
            Self::GlobalSimilarity => "global-similarity",
            Self::LimbusConsensus => "limbus-consensus",
            Self::LimbusOnly => "limbus-only",
            Self::LimbusReacquisition => "limbus-proposal-reacquisition",
        }
    }

    pub(crate) fn short_label(self) -> &'static str {
        match self {
            Self::Uninitialized => "WAIT",
            Self::FrameProposal => "PROPOSAL",
            Self::SensorHold => "SENSOR",
            Self::GlobalSimilarity => "GLOBAL",
            Self::LimbusConsensus => "GLOBAL+LIMBUS",
            Self::LimbusOnly => "LIMBUS",
            Self::LimbusReacquisition => "LIMBUS+PROPOSAL",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PendingPupilCenterRelocation {
    canonical_center: Option<(f64, f64)>,
    reference_canonical_center: Option<(f64, f64)>,
    sensor_center: (f64, f64),
    agreeing_frames: u8,
    last_seen: Option<Instant>,
    decisive: bool,
    measurement_score: f64,
    proposal_agrees: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PupilCenterTrackDiagnostics {
    pub(crate) regime: PupilCenterMotionRegime,
    pub(crate) transport_source: PupilCenterTransportSource,
    pub(crate) predicted_center: Option<(f64, f64)>,
    pub(crate) measured_center: Option<(f64, f64)>,
    pub(crate) published_center: Option<(f64, f64)>,
    pub(crate) limbus_transport_disagreement_px: f64,
    pub(crate) innovation_px: f64,
    pub(crate) fixation_gate_px: f64,
    pub(crate) measurement_score: f64,
    pub(crate) measurement_admissible: bool,
    pub(crate) transported_hold: bool,
    pub(crate) pending_relocation_frames: u8,
    pub(crate) saccade_score: f32,
    pub(crate) relative_motion_confidence: f32,
    pub(crate) relative_speed_px_s: f32,
    pub(crate) relative_acceleration_px_s2: f32,
    pub(crate) relative_jerk_px_s3: f32,
    pub(crate) specular_layer_excluded: bool,
    pub(crate) pursuit_predicted: bool,
    pub(crate) pursuit_velocity_canonical_per_second: Option<(f64, f64)>,
}

#[derive(Default)]
pub(crate) struct PupilCenterStateTracker {
    /// Pupil location in the current physical limbus' fronto-parallel affine
    /// coordinates.  Holding this fixed transports ordinary head/ROI motion
    /// without pretending that the specular pupil interior is rigid.
    canonical_center: Option<(f64, f64)>,
    /// Absolute sensor coordinate is the fallback when the selected limbus is
    /// temporarily withheld.  ROI-buffer and client-crop offsets therefore do
    /// not silently become apparent pupil motion.
    sensor_center: Option<(f64, f64)>,
    last_supported: Option<Instant>,
    fixation_streak: u8,
    pending_relocation: Option<PendingPupilCenterRelocation>,
    pursuit_velocity_canonical_per_second: Option<(f64, f64)>,
    pursuit_supported_at: Option<Instant>,
    last_frame_at: Option<Instant>,
    diagnostics: PupilCenterTrackDiagnostics,
}

pub(crate) fn pupil_center_orbital_measurement_admissible(
    measurement: raw_iris_focus::PupilCenterOrbitalFit,
) -> bool {
    measurement.score >= 0.54
        && measurement.ring_transition >= 0.16
        && measurement.ring_coverage >= 0.42
        && measurement.opposing_support >= 0.42
        && measurement.broad_dark_step >= 0.012
        && measurement.broad_dark_support >= 0.58
}

pub(crate) fn pupil_center_orbital_measurement_decisive(
    measurement: raw_iris_focus::PupilCenterOrbitalFit,
) -> bool {
    pupil_center_orbital_measurement_admissible(measurement)
        && measurement.score >= 0.62
        && measurement.ring_coverage >= 0.58
        && measurement.opposing_support >= 0.58
        && measurement.broad_dark_support >= 0.72
}

pub(crate) fn pupil_center_saccade_motion_supported<'a>(
    overlay: impl Into<PupilCenterMotionEvidence<'a>>,
) -> bool {
    let overlay = overlay.into();
    let coupled = overlay.coupled_motion;
    let relative = coupled.pupil_relative_to_general;
    let pupil_layer = overlay.pupil_layer;
    let jerk = relative.jerk_px_s3[0].hypot(relative.jerk_px_s3[1]);
    let classified_saccade = coupled.saccade_score >= 0.68;
    let high_jerk_transition = coupled.saccade_score >= 0.34 && jerk >= 650.0;
    relative.samples >= 4
        && relative.confidence >= 0.12
        && pupil_layer.persistent_tracks >= 3
        && pupil_layer.stable_frames >= 2
        && pupil_layer.coherence >= 0.18
        && (classified_saccade || high_jerk_transition)
}

/// A short-history acceleration burst may be real before the cubic motion
/// fit has accumulated enough confidence to authorize an immediate state
/// transition. It may broaden the bounded RAW search, but publication remains
/// behind `pupil_center_saccade_motion_supported` or multi-frame ring
/// confirmation. The reflection layer is absent from every term here.
pub(crate) fn pupil_center_saccade_search_warranted<'a>(
    overlay: impl Into<PupilCenterMotionEvidence<'a>>,
) -> bool {
    let overlay = overlay.into();
    let relative = overlay.coupled_motion.pupil_relative_to_general;
    let pupil_layer = overlay.pupil_layer;
    let acceleration = relative.acceleration_px_s2[0].hypot(relative.acceleration_px_s2[1]);
    let jerk = relative.jerk_px_s3[0].hypot(relative.jerk_px_s3[1]);
    relative.samples >= 4
        && relative.confidence >= 0.045
        && pupil_layer.persistent_tracks >= 3
        && pupil_layer.stable_frames >= 2
        && pupil_layer.coherence >= 0.18
        && (overlay.coupled_motion.saccade_score >= 0.22
            || acceleration >= 420.0
            || jerk >= 1_100.0)
}

fn bounded_pupil_pursuit_velocity(
    previous: (f64, f64),
    current: (f64, f64),
    elapsed: Duration,
) -> Option<(f64, f64)> {
    let seconds = elapsed.as_secs_f64();
    if !(0.035..=0.30).contains(&seconds) {
        return None;
    }
    let velocity = (
        (current.0 - previous.0) / seconds,
        (current.1 - previous.1) / seconds,
    );
    let speed = velocity.0.hypot(velocity.1);
    (speed.is_finite()
        && speed > 0.0
        && speed <= PUPIL_CENTER_PURSUIT_MAX_CANONICAL_SPEED_PER_SECOND)
        .then_some(velocity)
}

fn pupil_center_progressive_pursuit_velocity(
    pending: PendingPupilCenterRelocation,
    current_canonical: Option<(f64, f64)>,
    now: Instant,
    current_decisive: bool,
    current_measurement_score: f64,
    current_proposal_agrees: bool,
) -> Option<(f64, f64)> {
    let trace = std::env::var_os("BUTTERCUP_PUPIL_PURSUIT_TRACE").is_some();
    let raw_pair_strong = (pending.decisive && current_measurement_score >= 0.58)
        || (current_decisive && pending.measurement_score >= 0.58)
        || (pending.measurement_score >= 0.62 && current_measurement_score >= 0.62);
    if !raw_pair_strong || !pending.proposal_agrees || !current_proposal_agrees {
        if trace {
            eprintln!(
                "pupil-pursuit reject=raw/proposal prior={:.3}/{} current={:.3}/{} proposal={}/{}",
                pending.measurement_score,
                pending.decisive,
                current_measurement_score,
                current_decisive,
                pending.proposal_agrees,
                current_proposal_agrees,
            );
        }
        return None;
    }
    let reference = pending.reference_canonical_center?;
    let previous = pending.canonical_center?;
    let current = current_canonical?;
    let previous_seen = pending.last_seen?;
    let elapsed = now.saturating_duration_since(previous_seen);
    if elapsed > PUPIL_CENTER_PURSUIT_MAX_AGE {
        return None;
    }
    let first_step = (previous.0 - reference.0, previous.1 - reference.1);
    let second_step = (current.0 - previous.0, current.1 - previous.1);
    let first_length = first_step.0.hypot(first_step.1);
    let second_length = second_step.0.hypot(second_step.1);
    if !(PUPIL_CENTER_PURSUIT_MIN_CANONICAL_STEP..=PUPIL_CENTER_PURSUIT_MAX_CANONICAL_STEP)
        .contains(&first_length)
        || !(PUPIL_CENTER_PURSUIT_MIN_CANONICAL_STEP..=PUPIL_CENTER_PURSUIT_MAX_CANONICAL_STEP)
            .contains(&second_length)
    {
        if trace {
            eprintln!(
                "pupil-pursuit reject=step first={first_length:.5} second={second_length:.5}"
            );
        }
        return None;
    }
    let direction_cosine = (first_step.0 * second_step.0 + first_step.1 * second_step.1)
        / (first_length * second_length);
    let step_ratio = second_length / first_length;
    if direction_cosine < PUPIL_CENTER_PURSUIT_MIN_DIRECTION_COSINE
        || !(PUPIL_CENTER_PURSUIT_MIN_STEP_RATIO..=PUPIL_CENTER_PURSUIT_MAX_STEP_RATIO)
            .contains(&step_ratio)
    {
        if trace {
            eprintln!(
                "pupil-pursuit reject=direction cosine={direction_cosine:.4} ratio={step_ratio:.4} first={first_length:.5} second={second_length:.5}"
            );
        }
        return None;
    }
    let velocity = bounded_pupil_pursuit_velocity(previous, current, elapsed);
    if trace {
        eprintln!(
            "pupil-pursuit {} cosine={direction_cosine:.4} ratio={step_ratio:.4} first={first_length:.5} second={second_length:.5} velocity={velocity:?}",
            if velocity.is_some() {
                "accept"
            } else {
                "reject=speed"
            }
        );
    }
    velocity
}

impl PupilCenterStateTracker {
    #[cfg(test)]
    pub(crate) fn canonical_center(&self) -> Option<(f64, f64)> {
        self.canonical_center
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn diagnostics(&self) -> PupilCenterTrackDiagnostics {
        self.diagnostics
    }

    pub(crate) fn begin_frame<'a>(
        &mut self,
        now: Instant,
        sensor_origin: (u32, u32),
        frame_extent: (usize, usize),
        projection: Option<PupilProjectionReference>,
        frame_proposal: Option<(f64, f64)>,
        motion: impl Into<PupilCenterMotionEvidence<'a>>,
    ) -> Option<PupilCenterPrediction> {
        let motion = motion.into();
        if self.last_supported.is_some_and(|last| {
            now.saturating_duration_since(last) > PUPIL_CENTER_TRACK_STALE_AFTER
        }) {
            self.clear();
        }
        let frame_dt = self
            .last_frame_at
            .map(|previous| now.saturating_duration_since(previous).as_secs_f64())
            .unwrap_or(0.0)
            .clamp(0.0, 0.20);
        self.last_frame_at = Some(now);
        let pursuit_velocity = self.pursuit_velocity_canonical_per_second.filter(|_| {
            self.pursuit_supported_at.is_some_and(|supported| {
                now.saturating_duration_since(supported) <= PUPIL_CENTER_PURSUIT_MAX_AGE
            })
        });
        if pursuit_velocity.is_none() {
            self.pursuit_velocity_canonical_per_second = None;
            self.pursuit_supported_at = None;
        }
        let limbus_hold =
            projection
                .zip(self.canonical_center)
                .and_then(|(projection, canonical)| {
                    pupil_projection_image_point(projection, canonical)
                });
        let limbus_transport = projection
            .zip(self.canonical_center)
            .zip(pursuit_velocity)
            .filter(|_| frame_dt > 0.0)
            .and_then(|((projection, canonical), velocity)| {
                pupil_projection_image_point(
                    projection,
                    (
                        canonical.0 + velocity.0 * frame_dt,
                        canonical.1 + velocity.1 * frame_dt,
                    ),
                )
            })
            .or(limbus_hold);
        let pursuit_delta = limbus_hold
            .zip(limbus_transport)
            .map(|(held, pursued)| (pursued.0 - held.0, pursued.1 - held.1))
            .filter(|delta| delta.0.hypot(delta.1) > 1.0e-6);
        let sensor_hold = self.sensor_center.map(|sensor| {
            (
                sensor.0 - f64::from(sensor_origin.0),
                sensor.1 - f64::from(sensor_origin.1),
            )
        });
        let global_motion = motion.global_motion;
        let global_layer = motion.global_layer;
        let global_reliable = global_motion.support >= 4
            && global_layer.persistent_tracks >= 3
            && global_layer.stable_frames >= 2
            && global_layer.coherence >= 0.10
            && global_motion.residual.is_finite()
            && global_motion.residual <= 4.0;
        let global_transport = self
            .sensor_center
            .filter(|_| global_reliable)
            .map(|sensor| {
                // `SimilarityMotion` is fitted in absolute sensor coordinates
                // about the current RAW slice center. Applying the same bounded
                // transform to the persistent pupil state transports head motion
                // without treating iris texture or a reflection as rigid.
                let analysis_center = (
                    f64::from(sensor_origin.0) + frame_extent.0 as f64 * 0.5,
                    f64::from(sensor_origin.1) + frame_extent.1 as f64 * 0.5,
                );
                let x = sensor.0 - analysis_center.0;
                let y = sensor.1 - analysis_center.1;
                (
                    sensor.0
                        + f64::from(global_motion.translation[0])
                        + f64::from(global_motion.diagonal_coefficient_delta) * x
                        - f64::from(global_motion.rotation_coefficient) * y
                        - f64::from(sensor_origin.0),
                    sensor.1
                        + f64::from(global_motion.translation[1])
                        + f64::from(global_motion.rotation_coefficient) * x
                        + f64::from(global_motion.diagonal_coefficient_delta) * y
                        - f64::from(sensor_origin.1),
                )
            });
        let (material_transport, material_source) = if let Some(center) = global_transport {
            (Some(center), PupilCenterTransportSource::GlobalSimilarity)
        } else if let Some(center) = sensor_hold {
            (Some(center), PupilCenterTransportSource::SensorHold)
        } else {
            (None, PupilCenterTransportSource::Uninitialized)
        };
        // Global/sensor transport owns head and ROI motion.  Add only the
        // short-lived limbus-canonical pursuit displacement, keeping those
        // coordinate systems separate instead of treating iris texture as a
        // rigid pupil feature.
        let material_transport = material_transport.map(|center| {
            pursuit_delta.map_or(center, |delta| (center.0 + delta.0, center.1 + delta.1))
        });
        let limbus_disagreement = material_transport
            .zip(limbus_transport)
            .map_or(0.0, |(material, limbus)| {
                (material.0 - limbus.0).hypot(material.1 - limbus.1)
            });
        let limbus_consensus_gate = projection.map_or(3.0, |projection| {
            (projection.fronto_parallel_limbus_radius_px.value() * 0.035).clamp(2.5, 5.0)
        });
        let (transported, transport_source) = match (material_transport, limbus_transport) {
            (Some(material), Some(limbus)) if limbus_disagreement <= limbus_consensus_gate => {
                // Material motion is authoritative. The independently fitted
                // limbus gets a small subpixel refinement only in consensus.
                (
                    Some((
                        0.78 * material.0 + 0.22 * limbus.0,
                        0.78 * material.1 + 0.22 * limbus.1,
                    )),
                    PupilCenterTransportSource::LimbusConsensus,
                )
            }
            (Some(material), _) => (Some(material), material_source),
            (None, Some(limbus)) => (Some(limbus), PupilCenterTransportSource::LimbusOnly),
            (None, None) => (None, PupilCenterTransportSource::Uninitialized),
        };
        let (center, transport_source) = transported
            .map(|center| (center, transport_source))
            .or_else(|| {
                frame_proposal.map(|center| (center, PupilCenterTransportSource::FrameProposal))
            })?;
        if !center.0.is_finite() || !center.1.is_finite() {
            return None;
        }
        // A held sensor coordinate is not evidence that the eye stayed put.
        // If the independently acquired current pupil agrees with transport
        // by the current limbus, test that location as a NEW acquisition.
        // In particular, do not publish it through the transported-hold path:
        // assimilation must first verify a fresh native-RAW pupil ring.
        let coherent_reacquisition = limbus_transport
            .zip(frame_proposal)
            .zip(projection)
            .is_some_and(|((limbus, proposal), projection)| {
                let radius = projection.fronto_parallel_limbus_radius_px.value();
                // Small contour jitter belongs to normal fixation
                // refinement, not repeated cold acquisition.
                limbus_disagreement > (0.10 * radius).clamp(8.0, 16.0)
                    && (limbus.0 - proposal.0).hypot(limbus.1 - proposal.1)
                        <= (0.18 * radius).clamp(8.0, 20.0)
            });
        let prediction = PupilCenterPrediction {
            center: if coherent_reacquisition {
                frame_proposal.unwrap_or(center)
            } else {
                center
            },
            transported_track: transported.is_some() && !coherent_reacquisition,
            transport_source: if coherent_reacquisition {
                PupilCenterTransportSource::LimbusReacquisition
            } else {
                transport_source
            },
            limbus_transport_disagreement_px: limbus_disagreement,
            pursuit_predicted: pursuit_delta.is_some(),
        };
        self.diagnostics = PupilCenterTrackDiagnostics {
            regime: if prediction.transported_track {
                self.diagnostics.regime
            } else {
                PupilCenterMotionRegime::Uninitialized
            },
            transport_source: prediction.transport_source,
            predicted_center: Some(prediction.center),
            limbus_transport_disagreement_px: limbus_disagreement,
            specular_layer_excluded: true,
            pursuit_predicted: prediction.pursuit_predicted,
            pursuit_velocity_canonical_per_second: pursuit_velocity,
            ..PupilCenterTrackDiagnostics::default()
        };
        Some(prediction)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assimilate<'a>(
        &mut self,
        now: Instant,
        sensor_origin: (u32, u32),
        projection: Option<PupilProjectionReference>,
        prediction: PupilCenterPrediction,
        frame_proposal: Option<(f64, f64)>,
        measurement: Option<raw_iris_focus::PupilCenterOrbitalFit>,
        overlay: impl Into<PupilCenterMotionEvidence<'a>>,
    ) -> Option<(f64, f64)> {
        let overlay = overlay.into();
        let previous_canonical_center = self.canonical_center;
        let previous_supported_at = self.last_supported;
        let measured = measurement
            .filter(|measurement| pupil_center_orbital_measurement_admissible(*measurement));
        let limbus_radius = projection.map_or(64.0, |projection| {
            projection.fronto_parallel_limbus_radius_px.value()
        });
        let fixation_gate = (limbus_radius * 0.014).clamp(1.25, 2.40);
        let micro_gate = if overlay.coupled_motion.micro_motion_score >= 0.65 {
            fixation_gate * 1.35
        } else {
            fixation_gate
        };
        let relative = overlay.coupled_motion.pupil_relative_to_general;
        let relative_acceleration =
            relative.acceleration_px_s2[0].hypot(relative.acceleration_px_s2[1]);
        let relative_jerk = relative.jerk_px_s3[0].hypot(relative.jerk_px_s3[1]);
        let mut published = prediction.transported_track.then_some(prediction.center);
        let mut regime = if prediction.transported_track {
            if prediction.pursuit_predicted {
                PupilCenterMotionRegime::SmoothPursuit
            } else {
                PupilCenterMotionRegime::Fixation
            }
        } else {
            PupilCenterMotionRegime::Uninitialized
        };
        let mut innovation = 0.0;
        let mut transported_hold = prediction.transported_track;
        let mut confirmed_pursuit_velocity = None;

        if let Some(measurement) = measured {
            innovation = (measurement.center.0 - prediction.center.0)
                .hypot(measurement.center.1 - prediction.center.1);
            if !prediction.transported_track {
                published = Some(measurement.center);
                self.pending_relocation = None;
                self.fixation_streak = 1;
                regime = PupilCenterMotionRegime::Fixation;
                transported_hold = false;
            } else if innovation <= micro_gate {
                // A fixation measurement may converge below the integer pixel
                // grid, but one noisy meridian cannot move the state by more
                // than the physical fixation gate in a single frame.
                let gain = (0.42 + 0.38 * measurement.score).clamp(0.50, 0.78);
                let requested = (
                    (measurement.center.0 - prediction.center.0) * gain,
                    (measurement.center.1 - prediction.center.1) * gain,
                );
                let requested_length = requested.0.hypot(requested.1);
                let scale = if requested_length > fixation_gate {
                    fixation_gate / requested_length
                } else {
                    1.0
                };
                published = Some((
                    prediction.center.0 + requested.0 * scale,
                    prediction.center.1 + requested.1 * scale,
                ));
                self.pending_relocation = None;
                self.fixation_streak = self.fixation_streak.saturating_add(1);
                regime = if prediction.pursuit_predicted {
                    PupilCenterMotionRegime::SmoothPursuit
                } else if self.diagnostics.regime == PupilCenterMotionRegime::Saccade
                    && self.fixation_streak < 2
                {
                    PupilCenterMotionRegime::Saccade
                } else {
                    PupilCenterMotionRegime::Fixation
                };
                transported_hold = false;
            } else {
                let saccade_motion = pupil_center_saccade_motion_supported(overlay);
                // The independently produced rough center is usually close
                // even though it is not accurate enough to publish directly.
                // A remote ring beyond this anatomical acquisition corridor
                // is more likely an iris fibre/lid alias than a pupil.
                let proposal_agreement_gate = (limbus_radius * 0.11).clamp(6.0, 13.0);
                let proposal_agrees = frame_proposal.is_some_and(|proposal| {
                    (proposal.0 - measurement.center.0).hypot(proposal.1 - measurement.center.1)
                        <= proposal_agreement_gate
                });
                let immediate_saccade = saccade_motion
                    && proposal_agrees
                    && pupil_center_orbital_measurement_decisive(measurement);
                let measured_canonical = projection.and_then(|projection| {
                    pupil_projection_canonical_point(projection, measurement.center)
                });
                let progressive_pursuit_velocity = self.pending_relocation.and_then(|pending| {
                    pupil_center_progressive_pursuit_velocity(
                        pending,
                        measured_canonical,
                        now,
                        pupil_center_orbital_measurement_decisive(measurement),
                        measurement.score,
                        proposal_agrees,
                    )
                });
                if immediate_saccade {
                    published = Some(measurement.center);
                    self.pending_relocation = None;
                    self.pursuit_velocity_canonical_per_second = None;
                    self.pursuit_supported_at = None;
                    self.fixation_streak = 0;
                    regime = PupilCenterMotionRegime::Saccade;
                    transported_hold = false;
                } else if let Some(velocity) = progressive_pursuit_velocity {
                    published = Some(measurement.center);
                    self.pending_relocation = None;
                    self.fixation_streak = 0;
                    confirmed_pursuit_velocity = Some(velocity);
                    regime = PupilCenterMotionRegime::SmoothPursuit;
                    transported_hold = false;
                } else {
                    let measured_sensor = (
                        measurement.center.0 + f64::from(sensor_origin.0),
                        measurement.center.1 + f64::from(sensor_origin.1),
                    );
                    // Without already-supported saccade kinematics, temporal
                    // persistence means the same physical ring, not merely
                    // two vaguely nearby dark structures. A real continuing
                    // saccade takes the independent motion-authorized path.
                    let pending_agreement_gate = (limbus_radius * 0.042).clamp(3.0, 5.5);
                    let pending_same_ring = self.pending_relocation.is_some_and(|pending| {
                        if pending.last_seen.is_some_and(|last| {
                            now.saturating_duration_since(last)
                                > PUPIL_CENTER_PENDING_RELOCATION_MAX_AGE
                        }) {
                            return false;
                        }
                        match (pending.canonical_center, measured_canonical, projection) {
                            (Some(previous), Some(_), Some(projection)) => {
                                pupil_projection_image_point(projection, previous).is_some_and(
                                    |previous_image| {
                                        (previous_image.0 - measurement.center.0)
                                            .hypot(previous_image.1 - measurement.center.1)
                                            <= pending_agreement_gate
                                    },
                                )
                            }
                            _ => {
                                (pending.sensor_center.0 - measured_sensor.0)
                                    .hypot(pending.sensor_center.1 - measured_sensor.1)
                                    <= pending_agreement_gate
                            }
                        }
                    });
                    let bounded_canonical_relocation =
                        self.pending_relocation.is_some_and(|pending| {
                            pending
                                .reference_canonical_center
                                .zip(measured_canonical)
                                .is_some_and(|(reference, current)| {
                                    (current.0 - reference.0).hypot(current.1 - reference.1)
                                        <= PUPIL_CENTER_UNSUPPORTED_MAX_CANONICAL_RELOCATION
                                })
                        });
                    // The decoded fixation capture showed that agreement
                    // between two proposal-local paths is not independent:
                    // both can ride the same coherent iris texture.  Only
                    // persistence of the actual RAW ring may advance this
                    // no-saccade relocation counter.  Smooth pursuit retains
                    // its separate three-position, direction/step-ratio test
                    // above; classified high-jerk motion retains the immediate
                    // saccade path.
                    let pending_agrees = pending_same_ring;
                    let agreeing_frames = if pending_agrees {
                        self.pending_relocation
                            .map_or(1, |pending| pending.agreeing_frames.saturating_add(1))
                    } else {
                        1
                    };
                    self.pending_relocation = Some(PendingPupilCenterRelocation {
                        canonical_center: measured_canonical,
                        reference_canonical_center: self.canonical_center,
                        sensor_center: measured_sensor,
                        agreeing_frames,
                        last_seen: Some(now),
                        decisive: pupil_center_orbital_measurement_decisive(measurement),
                        measurement_score: measurement.score,
                        proposal_agrees,
                    });
                    let decisive_confirmation = agreeing_frames >= 2
                        && proposal_agrees
                        && bounded_canonical_relocation
                        && pupil_center_orbital_measurement_decisive(measurement);
                    if decisive_confirmation {
                        published = Some(measurement.center);
                        self.pending_relocation = None;
                        self.pursuit_velocity_canonical_per_second = None;
                        self.pursuit_supported_at = None;
                        self.fixation_streak = 0;
                        regime = PupilCenterMotionRegime::Saccade;
                        transported_hold = false;
                    } else {
                        regime = PupilCenterMotionRegime::RelocationPending;
                    }
                }
            }
        } else if !prediction.transported_track {
            // A proposal is an acquisition coordinate, not state. Do not let
            // an unverified dark basin become the temporal center merely
            // because no better measurement was available this frame.
            published = None;
        }

        if let Some(center) = published {
            self.sensor_center = Some((
                center.0 + f64::from(sensor_origin.0),
                center.1 + f64::from(sensor_origin.1),
            ));
            let next_canonical = projection
                .and_then(|projection| pupil_projection_canonical_point(projection, center));
            if regime == PupilCenterMotionRegime::SmoothPursuit {
                let observed_velocity = confirmed_pursuit_velocity.or_else(|| {
                    previous_canonical_center
                        .zip(next_canonical)
                        .zip(previous_supported_at)
                        .and_then(|((previous, current), previous_at)| {
                            bounded_pupil_pursuit_velocity(
                                previous,
                                current,
                                now.saturating_duration_since(previous_at),
                            )
                        })
                });
                if let Some(observed) = observed_velocity {
                    let filtered =
                        self.pursuit_velocity_canonical_per_second
                            .map_or(observed, |previous| {
                                (
                                    0.35 * previous.0 + 0.65 * observed.0,
                                    0.35 * previous.1 + 0.65 * observed.1,
                                )
                            });
                    self.pursuit_velocity_canonical_per_second = Some(filtered);
                    self.pursuit_supported_at = Some(now);
                }
            } else if matches!(
                regime,
                PupilCenterMotionRegime::Fixation
                    | PupilCenterMotionRegime::Saccade
                    | PupilCenterMotionRegime::Uninitialized
            ) {
                self.pursuit_velocity_canonical_per_second = None;
                self.pursuit_supported_at = None;
            }
            if let Some(projection) = projection {
                // A disagreeing material hold is not a new pupil/iris
                // observation. Rewriting the relative coordinate here made
                // one missed head-motion frame permanently move the pupil
                // within the iris, poisoning subsequent transport.
                let contradicted_hold = transported_hold
                    && prediction.limbus_transport_disagreement_px
                        > (0.035 * projection.fronto_parallel_limbus_radius_px.value())
                            .clamp(2.5, 5.0);
                if !contradicted_hold {
                    self.canonical_center = pupil_projection_canonical_point(projection, center);
                }
            }
            if !transported_hold {
                self.last_supported = Some(now);
            }
        }
        self.diagnostics = PupilCenterTrackDiagnostics {
            regime,
            transport_source: prediction.transport_source,
            predicted_center: Some(prediction.center),
            measured_center: measured.map(|measurement| measurement.center),
            published_center: published,
            limbus_transport_disagreement_px: prediction.limbus_transport_disagreement_px,
            innovation_px: innovation,
            fixation_gate_px: fixation_gate,
            measurement_score: measurement.map_or(0.0, |measurement| measurement.score),
            measurement_admissible: measured.is_some(),
            transported_hold,
            pending_relocation_frames: self
                .pending_relocation
                .map_or(0, |pending| pending.agreeing_frames),
            saccade_score: overlay.coupled_motion.saccade_score,
            relative_motion_confidence: relative.confidence,
            relative_speed_px_s: relative.speed_px_s,
            relative_acceleration_px_s2: relative_acceleration,
            relative_jerk_px_s3: relative_jerk,
            specular_layer_excluded: true,
            pursuit_predicted: prediction.pursuit_predicted,
            pursuit_velocity_canonical_per_second: self.pursuit_velocity_canonical_per_second,
        };
        published
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eye_scene_model::pupil_projection::PupilProjectionSource;

    fn projection_at(center: (f64, f64)) -> PupilProjectionReference {
        PupilProjectionReference::from_axes(
            center,
            100.0,
            80.0,
            0.15,
            PupilProjectionSource::SelectedIris,
        )
        .unwrap()
    }

    fn admitted_tracker(
        now: Instant,
        motion: PupilCenterMotionEvidence<'_>,
    ) -> PupilCenterStateTracker {
        let projection = projection_at((190.0, 128.0));
        let mut tracker = PupilCenterStateTracker::default();
        let center = (202.0, 150.0);
        let prediction = tracker
            .begin_frame(
                now,
                (3800, 3500),
                (384, 256),
                Some(projection),
                Some(center),
                motion,
            )
            .unwrap();
        let measurement = raw_iris_focus::PupilCenterOrbitalFit {
            center,
            score: 0.74,
            ring_transition: 0.26,
            ring_coverage: 0.82,
            opposing_support: 0.75,
            interior_void: 0.80,
            broad_dark_step: 0.18,
            broad_dark_support: 0.88,
            canonical_radius: 0.1,
            evaluated_centers: 32,
        };
        assert_eq!(
            tracker.assimilate(
                now,
                (3800, 3500),
                Some(projection),
                prediction,
                Some(center),
                Some(measurement),
                motion,
            ),
            Some(center)
        );
        tracker
    }

    #[test]
    fn narrow_motion_evidence_preserves_sensor_transport_across_roi_reframing() {
        let now = Instant::now();
        let global_motion = SimilarityMotion::default();
        let layer = MotionLayerStatus::default();
        let coupled = CoupledMotionStatus::default();
        let neutral = PupilCenterMotionEvidence {
            global_motion: &global_motion,
            global_layer: &layer,
            pupil_layer: &layer,
            coupled_motion: &coupled,
        };
        let mut tracker = admitted_tracker(now, neutral);
        let moving = SimilarityMotion {
            translation: [3.0, -2.0],
            support: 12,
            residual: 0.5,
            ..SimilarityMotion::default()
        };
        let stable_layer = MotionLayerStatus {
            persistent_tracks: 12,
            stable_frames: 5,
            coherence: 0.9,
            ..MotionLayerStatus::default()
        };
        let motion = PupilCenterMotionEvidence {
            global_motion: &moving,
            global_layer: &stable_layer,
            ..neutral
        };
        let next = now + Duration::from_millis(100);
        let projection = projection_at((193.0, 126.0));
        let predicted = tracker
            .begin_frame(
                next,
                (3800, 3500),
                (384, 256),
                Some(projection),
                None,
                motion,
            )
            .unwrap();
        assert!((predicted.center.0 - 205.0).hypot(predicted.center.1 - 148.0) < 1.0e-10);
        assert_eq!(
            predicted.transport_source,
            PupilCenterTransportSource::LimbusConsensus
        );
        tracker.assimilate(
            next,
            (3800, 3500),
            Some(projection),
            predicted,
            None,
            None,
            motion,
        );

        // Reframing the ROI is not physical eye motion. Preserve the sensor point.
        let reframed = tracker
            .begin_frame(
                now + Duration::from_millis(200),
                (3812, 3492),
                (384, 256),
                Some(projection_at((181.0, 134.0))),
                None,
                neutral,
            )
            .unwrap();
        assert!((reframed.center.0 - 193.0).hypot(reframed.center.1 - 156.0) < 1.0e-10);
        assert_eq!(tracker.last_supported, Some(now));
    }

    #[test]
    fn unmeasured_transport_does_not_renew_the_observation_clock() {
        let now = Instant::now();
        let global_motion = SimilarityMotion::default();
        let layer = MotionLayerStatus::default();
        let coupled = CoupledMotionStatus::default();
        let motion = PupilCenterMotionEvidence {
            global_motion: &global_motion,
            global_layer: &layer,
            pupil_layer: &layer,
            coupled_motion: &coupled,
        };
        let mut tracker = admitted_tracker(now, motion);
        let projection = projection_at((190.0, 128.0));
        for tick in 1..=12 {
            let at = now + Duration::from_millis(tick * 100);
            let predicted = tracker
                .begin_frame(at, (3800, 3500), (384, 256), Some(projection), None, motion)
                .unwrap();
            assert!(predicted.transported_track);
            assert!(tracker
                .assimilate(
                    at,
                    (3800, 3500),
                    Some(projection),
                    predicted,
                    None,
                    None,
                    motion,
                )
                .is_some());
            assert_eq!(tracker.last_supported, Some(now));
        }
        assert!(tracker
            .begin_frame(
                now + PUPIL_CENTER_TRACK_STALE_AFTER + Duration::from_millis(1),
                (3800, 3500),
                (384, 256),
                Some(projection),
                None,
                motion,
            )
            .is_none());
        assert_eq!(tracker.last_supported, None);
    }
}
