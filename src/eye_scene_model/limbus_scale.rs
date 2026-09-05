//! Candidate-independent limbus radius support and its legacy temporal posterior.
//!
//! Pixel radii are fronto-parallel apparent sizes, not anatomical millimeters.
//! Hard search support, soft preferred scale and observation provenance remain
//! distinct. Host `Instant` bounds posterior aging; independent observations
//! carry source exposure keys so asynchronous redraws cannot manufacture votes.

use crate::conic_solver::projected_circular_limbus_axes_plausible;
use crate::geometry::Ellipse;
use crate::roi_evidence::ExposureKey;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Origin of the admissible apparent-size interval for a circular limbus
/// after fronto-parallel rectification.  These names describe geometry, not
/// detector confidence: a coarse semantic pose and fine visual odometry may
/// eventually predict image scale without being allowed to assert iris
/// anatomy by themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontoParallelLimbusRadiusPriorSource {
    FixedReference,
    OperatorHardLimits,
    /// Broad, current-frame geometric support used only before a temporal
    /// Driving radius posterior exists and independently verified by the
    /// coarse eye-basin geometry. It may constrain a cold-start search but
    /// never counts as an independently observed temporal size sample.
    CurrentFrameGeometry,
    /// Broad proposal-only support from an unverified native conic or compact
    /// carrier. Its midpoint is not a scale measurement: it may preserve
    /// deterministic ordering inside the legacy carrier cohort, but expanded
    /// outer-limbus hypotheses use only the stated hard interval.
    CurrentFrameUnverifiedGeometry,
    TemporalRobustMedian,
    CoarseSemanticPose,
    FineVisualOdometry,
}

impl FrontoParallelLimbusRadiusPriorSource {
    pub fn short_label(self) -> &'static str {
        match self {
            Self::FixedReference => "FIXED",
            Self::OperatorHardLimits => "MANUAL",
            Self::CurrentFrameGeometry => "FRAME",
            Self::CurrentFrameUnverifiedGeometry => "FRAME-WIDE",
            Self::TemporalRobustMedian => "TEMP",
            Self::CoarseSemanticPose => "COARSE",
            Self::FineVisualOdometry => "FINE",
        }
    }
}

/// Hard support of the prior distribution for the apparent radius of the
/// physical circular limbus in a fronto-parallel image plane.
///
/// Under the weak-perspective tilted-circle model, the fronto-parallel radius
/// is the ellipse's larger semi-axis.  Calling this a support interval rather
/// than a confidence interval is intentional: candidates outside it are
/// geometrically inadmissible, not merely lower-scoring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontoParallelLimbusRadiusPrior {
    pub estimate_px: f64,
    pub minimum_px: f64,
    pub maximum_px: f64,
    pub source: FrontoParallelLimbusRadiusPriorSource,
}

impl FrontoParallelLimbusRadiusPrior {
    pub fn from_hard_support(
        estimate_px: f64,
        minimum_px: f64,
        maximum_px: f64,
        source: FrontoParallelLimbusRadiusPriorSource,
    ) -> Option<Self> {
        if !estimate_px.is_finite()
            || !minimum_px.is_finite()
            || !maximum_px.is_finite()
            || minimum_px < 4.0
            || maximum_px <= minimum_px
        {
            return None;
        }
        Some(Self {
            estimate_px: estimate_px.clamp(minimum_px, maximum_px),
            minimum_px,
            maximum_px,
            source,
        })
    }

    pub fn from_fractional_support(
        estimate_px: f64,
        fractional_half_width: f64,
        source: FrontoParallelLimbusRadiusPriorSource,
    ) -> Option<Self> {
        if !estimate_px.is_finite()
            || estimate_px < 4.0
            || !fractional_half_width.is_finite()
            || !(0.0..=0.75).contains(&fractional_half_width)
        {
            return None;
        }
        let minimum_px = estimate_px * (1.0 - fractional_half_width);
        let maximum_px = estimate_px * (1.0 + fractional_half_width);
        Some(Self {
            estimate_px,
            minimum_px,
            maximum_px,
            source,
        })
    }

    pub fn fronto_parallel_radius_px(major_radius: f64, minor_radius: f64) -> Option<f64> {
        if !major_radius.is_finite()
            || !minor_radius.is_finite()
            || major_radius <= 0.0
            || minor_radius <= 0.0
            || !projected_circular_limbus_axes_plausible(major_radius, minor_radius)
        {
            return None;
        }
        Some(major_radius.max(minor_radius))
    }

    pub fn admits_radius(self, radius_px: f64) -> bool {
        radius_px.is_finite() && radius_px >= self.minimum_px && radius_px <= self.maximum_px
    }

    pub fn admits_ellipse(self, major_radius: f64, minor_radius: f64) -> bool {
        Self::fronto_parallel_radius_px(major_radius, minor_radius)
            .is_some_and(|radius| self.admits_radius(radius))
    }

    /// Decide whether a semantically strong limbus observation may publish
    /// and update the physical-size posterior. `minimum_px..maximum_px` is a
    /// search envelope: after a long detector outage it intentionally becomes
    /// wide enough to *look* for a physically reachable eye. A candidate near
    /// an extreme of that envelope still cannot claim that the eye changed
    /// scale unless whole-ROI motion has transported `estimate_px` there.
    ///
    /// Fixed/operator/current-frame supports are explicit authorities or
    /// cold-start geometry and retain their stated hard bounds. Temporal,
    /// coarse-pose, and fine-odometry priors all carry a motion-registered
    /// estimate, so their publication corridor remains compact around it.
    pub fn admits_kinematically_supported_radius(self, radius_px: f64) -> bool {
        if !self.admits_radius(radius_px) {
            return false;
        }
        match self.source {
            FrontoParallelLimbusRadiusPriorSource::FixedReference
            | FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits
            | FrontoParallelLimbusRadiusPriorSource::CurrentFrameGeometry
            | FrontoParallelLimbusRadiusPriorSource::CurrentFrameUnverifiedGeometry => true,
            FrontoParallelLimbusRadiusPriorSource::TemporalRobustMedian
            | FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose
            | FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry => {
                (radius_px / self.estimate_px.max(1.0)).ln().abs()
                    <= LIMBUS_STRONG_OBSERVATION_MAX_LOG_INNOVATION
            }
        }
    }

    pub fn admits_kinematically_supported_ellipse(
        self,
        major_radius: f64,
        minor_radius: f64,
    ) -> bool {
        Self::fronto_parallel_radius_px(major_radius, minor_radius)
            .is_some_and(|radius| self.admits_kinematically_supported_radius(radius))
    }
}

/// Optional image-scale prediction used to transport the temporal radius
/// posterior into a new frame.  The scale ratio is relative to the last
/// admitted frame; its uncertainty expands the hard support for this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontoParallelLimbusScalePrediction {
    pub scale_ratio: f64,
    pub fractional_uncertainty: f64,
    pub source: FrontoParallelLimbusRadiusPriorSource,
}

impl FrontoParallelLimbusScalePrediction {
    pub fn coarse_semantic_pose(scale_ratio: f64, fractional_uncertainty: f64) -> Self {
        Self {
            scale_ratio,
            fractional_uncertainty,
            source: FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose,
        }
    }

    pub fn fine_visual_odometry(scale_ratio: f64, fractional_uncertainty: f64) -> Self {
        Self {
            scale_ratio,
            fractional_uncertainty,
            source: FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry,
        }
    }

    fn valid(self) -> bool {
        self.scale_ratio.is_finite()
            && (0.40..=2.50).contains(&self.scale_ratio)
            && self.fractional_uncertainty.is_finite()
            && (0.0..=0.50).contains(&self.fractional_uncertainty)
            && matches!(
                self.source,
                FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose
                    | FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry
            )
    }

    /// Compose chronologically adjacent image-scale transports. This is used
    /// when a bounded worker drops obsolete frames: every skipped native
    /// visual-odometry delta still contributes to the next processed frame.
    pub fn composed_with(self, next: Self) -> Option<Self> {
        if !self.valid() || !next.valid() {
            return None;
        }
        let source = if self.source == FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose
            || next.source == FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose
        {
            FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose
        } else {
            FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry
        };
        let composed = Self {
            scale_ratio: self.scale_ratio * next.scale_ratio,
            fractional_uncertainty: (self.fractional_uncertainty
                + next.fractional_uncertainty
                + self.fractional_uncertainty * next.fractional_uncertainty)
                .min(0.50),
            source,
        };
        composed.valid().then_some(composed)
    }
}

/// Stateful prior over apparent fronto-parallel limbus radius. `begin_frame`
/// freezes one support interval for every candidate and refit in that frame;
/// only independently strong observations change the between-frame robust
/// posterior.
#[derive(Clone, Debug, Default)]
pub struct FrontoParallelLimbusRadiusTracker {
    fixed_reference: Option<FrontoParallelLimbusRadiusPrior>,
    /// Robust posterior center in the coordinate system of the most recently
    /// begun frame. Valid image-scale motion transports this state and every
    /// retained strong sample together before the frame support is frozen.
    mean_log_radius: Option<f64>,
    /// Median window of strong log-radius measurements. The first strong
    /// mutually consistent cold-start observations establish this window;
    /// weak fits never enter it.
    strong_log_radii: VecDeque<f64>,
    /// Strong but not-yet-consistent cold-start measurements.  A single
    /// eyelid, aperture, or glasses curve must not become the physical-size
    /// authority merely because it was the first completed conic after
    /// startup.  Entries are carried into the current image-scale coordinate
    /// system by the same independent whole-ROI prediction as the posterior.
    cold_start_log_radii: VecDeque<(Instant, f64, f64)>,
    mean_absolute_log_residual: f64,
    effective_observations: f64,
    /// Slowly adapted robust physical-radius anchor, expressed in the
    /// coordinate system of the most recently begun frame. Independent
    /// whole-ROI scale evidence transports this anchor immediately alongside
    /// the robust history. Individual detector curves only nudge it through a
    /// bounded robust-median update: disagreement between candidate curves is
    /// not evidence that the physical eye changed size.
    transported_strong_anchor_log_radius: Option<f64>,
    /// Most recently published strong radius in the current image-scale
    /// coordinate system. The robust anchor defines the search corridor;
    /// this separate value is only the final adjacent-publication authority,
    /// so two opposite ends of that corridor cannot alternate. Cold-start
    /// votes are proposal-only and therefore initialize this value from their
    /// robust consensus rather than whichever vote happened to arrive last.
    /// Independent whole-ROI scale evidence transports both quantities.
    latest_strong_log_radius: Option<f64>,
    last_observed: Option<Instant>,
    active_frame_at: Option<Instant>,
    active_frame_prior: Option<FrontoParallelLimbusRadiusPrior>,
    active_frame_recompute: bool,
    fine_log_transport_since_observation: f64,
    last_independent_observation: Option<(ExposureKey, u64, LimbusRadiusAdmission)>,
    recovery_samples: VecDeque<LimbusRecoverySample>,
    pending_recovery: Option<(Instant, f64)>,
}

/// Caller has independently verified the anatomical conic and RAW material
/// evidence. Neither a rendered/held ellipse nor a weak outline is a sample.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LimbusRadiusObservation {
    pub(crate) exposure: ExposureKey,
    /// Detector/prompt session, separate from the actual sensor clock epoch.
    pub(crate) lineage: u64,
    pub(crate) ellipse_sensor_px: Ellipse,
    pub(crate) confidence: f64,
    /// False for an ROI-clipped contour: useful diagnostically, never a vote
    /// for a discontinuous change in physical apparent size.
    pub(crate) complete_in_source_roi: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LimbusRadiusAdmission {
    Published,
    ColdStart { votes: usize },
    RecoveryVerifying { votes: usize },
    RecoveryReady,
    HardSupportRejected,
    ContinuityRejected,
    InvalidObservation,
    StaleObservation,
}

impl LimbusRadiusAdmission {
    pub(crate) fn published(self) -> bool {
        self == Self::Published
    }

    pub(crate) fn label(self) -> String {
        match self {
            Self::Published => "ADMITTED".into(),
            Self::ColdStart { votes } => {
                format!("COLD CONSENSUS {votes}/{LIMBUS_COLD_START_CONSENSUS}")
            }
            Self::RecoveryVerifying { votes } => {
                format!("RECOVERY VERIFY {votes}/{LIMBUS_RECOVERY_CONSENSUS}")
            }
            Self::RecoveryReady => "RECOVERY VERIFIED; NEXT FRAME".into(),
            Self::HardSupportRejected => "HARD SUPPORT REJECT".into(),
            Self::ContinuityRejected => "CONTINUITY REJECT".into(),
            Self::InvalidObservation => "INVALID OBSERVATION".into(),
            Self::StaleObservation => "STALE OBSERVATION".into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LimbusRecoverySample {
    timestamp_ns: u64,
    ellipse_sensor_px: Ellipse,
    log_radius: f64,
}

// Without an independently measured image-scale transport, apparent limbus
// radius is nearly rigid from one native eye frame to the next. Keep enough
// room for conic jitter, then widen only at a bounded human-motion rate while
// observations are missing. Faster scale changes must arrive through coarse
// interocular scale or corroborated full-resolution visual odometry.
const LIMBUS_UNCORROBORATED_BASE_HALF_WIDTH: f64 = 0.035;
const LIMBUS_UNCORROBORATED_HALF_WIDTH_PER_SECOND: f64 = 0.040;
// A broad stale search envelope is not permission to publish a discontinuous
// apparent-size jump. The de-affined fronto-parallel limbus radius is a
// physical scale: without an independent whole-ROI scale transport, adjacent
// publications may change reconstructed circular area by at most two percent.
// Since that area is proportional to radius squared, the symmetric log-radius
// bound is half ln(1.02). This is deliberately a publication rule rather than
// a candidate-bank width: the wider bank can still find a transported or
// reacquired eye, but two iris/lid/glasses aliases cannot alternate inside it.
const LIMBUS_STRONG_OBSERVATION_MAX_LOG_INNOVATION: f64 = 0.009_901_313_648_089_865;
// `latest_strong_log_radius` protects frame-to-frame continuity, not long-gap
// reacquisition. The live stream can run near 10 Hz under the bounded Driving
// budget, so 350 ms covers one or two missing frames plus scheduling jitter.
// Letting the guard expire after the first missed frame allowed the same
// unsupported large/small branch flip to recur every other publication. The
// robust transported posterior remains the authority after this horizon.
pub(crate) const LIMBUS_LATEST_STRONG_CONTINUITY_HORIZON: Duration = Duration::from_millis(350);
// Even mutually accepted detector fits may share the same eyelid or glasses
// alias. Let the rolling robust median calibrate the physical anchor slowly;
// genuine rapid distance change belongs to the explicit scale-transport path.
const LIMBUS_STRONG_ANCHOR_ROBUST_ADAPTATION: f64 = 0.05;
// A near-adjacent frame remains tightly constrained, but a person can move
// materially toward or away from the camera during a long run with no strong
// limbus observation.  Capping this forever at 30% made a valid 2x iris
// physically unreachable after a two-minute capture interval.  The search
// support now expands at the same explicit human-motion rate up to 2.5x the
// last strong apparent radius; full-ROI scale evidence still transports the
// center immediately when it is available.
const LIMBUS_UNCORROBORATED_MAX_HALF_WIDTH: f64 = 1.50;
const LIMBUS_COLD_START_CONSENSUS: usize = 3;
const LIMBUS_COLD_START_MAX_RATIO: f64 = 1.12;
const LIMBUS_COLD_START_HISTORY: Duration = Duration::from_millis(900);
const LIMBUS_RECOVERY_CONSENSUS: usize = 3;
const LIMBUS_RECOVERY_MIN_SPAN_NS: u64 = 200_000_000;
// Three independent SAM answers can span several seconds. This cohort uses
// much tighter agreement than the legacy native cold-start alias bank.
const LIMBUS_RECOVERY_WINDOW_NS: u64 = 3_000_000_000;
const LIMBUS_RECOVERY_AFTER: Duration = Duration::from_secs(1);
const LIMBUS_FINE_TRANSPORT_MAX_AGE: Duration = Duration::from_secs(2);
const LIMBUS_FINE_UNCONFIRMED_MAX_RATIO: f64 = 1.15;

fn limbus_radius_prior_from_expanding_support(
    estimate_px: f64,
    fractional_half_width: f64,
    source: FrontoParallelLimbusRadiusPriorSource,
) -> Option<FrontoParallelLimbusRadiusPrior> {
    let fractional_half_width = fractional_half_width.clamp(
        LIMBUS_UNCORROBORATED_BASE_HALF_WIDTH,
        LIMBUS_UNCORROBORATED_MAX_HALF_WIDTH,
    );
    if fractional_half_width <= 0.75 {
        return FrontoParallelLimbusRadiusPrior::from_fractional_support(
            estimate_px,
            fractional_half_width,
            source,
        );
    }
    // `from_fractional_support` intentionally represents ordinary compact
    // uncertainty and stops at 75%. Long unsupported intervals are a
    // physically reachable search envelope rather than a confidence band, so
    // express them directly as hard support. The lower edge bottoms out at
    // the tracker-wide geometrical minimum while the upper edge grows
    // continuously to 2.5x.
    FrontoParallelLimbusRadiusPrior::from_hard_support(
        estimate_px,
        (estimate_px * (1.0 - fractional_half_width)).max(4.0),
        estimate_px * (1.0 + fractional_half_width),
        source,
    )
}

impl FrontoParallelLimbusRadiusTracker {
    pub fn with_fixed_reference(reference_px: f64, fractional_half_width: f64) -> Option<Self> {
        let fixed_reference = FrontoParallelLimbusRadiusPrior::from_fractional_support(
            reference_px,
            fractional_half_width,
            FrontoParallelLimbusRadiusPriorSource::FixedReference,
        )?;
        Some(Self {
            fixed_reference: Some(fixed_reference),
            active_frame_prior: Some(fixed_reference),
            ..Self::default()
        })
    }

    pub fn active_frame_prior(&self) -> Option<FrontoParallelLimbusRadiusPrior> {
        self.active_frame_prior
    }

    /// Return the robust dynamic radius only after cold-start observations
    /// have formed a genuine common scale bucket. Individual proposal votes
    /// intentionally leave this as `None`, even though recording each vote is
    /// not itself an error.
    pub fn established_dynamic_radius_px(&self) -> Option<f64> {
        self.mean_log_radius.map(f64::exp)
    }

    /// Discard observations expressed in a previous Driving session's image
    /// coordinates while retaining an explicitly configured fixed physical
    /// reference. A top-level segmentation round trip has no intervening RAW
    /// scale transport, so carrying its dynamic posterior into the next
    /// Driving session would constrain the search with stale pixel geometry.
    pub fn reset_dynamic_observations(&mut self) {
        let fixed_reference = self.fixed_reference;
        *self = Self {
            fixed_reference,
            active_frame_prior: fixed_reference,
            ..Self::default()
        };
    }

    /// Freeze the support used throughout one frame.  A future MediaPipe
    /// gross-pose update or feature/VSLAM fine-scale update enters only via
    /// `prediction`; neither source is treated as anatomical evidence.
    pub fn begin_frame(
        &mut self,
        now: Instant,
        prediction: Option<FrontoParallelLimbusScalePrediction>,
    ) -> Option<FrontoParallelLimbusRadiusPrior> {
        self.begin_frame_controlled(now, prediction, true, None)
    }

    /// Freeze one frame's automatic support, optionally replacing it with
    /// explicit operator hard limits. `recompute=false` retains the prior
    /// support in its current image coordinates; a manual limit change still
    /// takes effect immediately so the stable-ROI switch cannot make the UI
    /// controls inert.
    pub fn begin_frame_controlled(
        &mut self,
        now: Instant,
        prediction: Option<FrontoParallelLimbusScalePrediction>,
        recompute: bool,
        operator_limits_px: Option<(f64, f64)>,
    ) -> Option<FrontoParallelLimbusRadiusPrior> {
        // A candidate bank may ask for the frozen support more than once.
        // Never apply a per-frame scale delta twice.
        if self.active_frame_at == Some(now) {
            return self.active_frame_prior;
        }
        // A bounded asynchronous detector may finish an older native frame
        // after the receive loop has already frozen support for a newer one.
        // Never move the shared physical-scale coordinate system backwards.
        // The stale detector may still use the newest support diagnostically,
        // but `observe_strong_ellipse_for_active_frame` below prevents it from
        // teaching that support with an out-of-order observation.
        if self.active_frame_at.is_some_and(|active| now < active) {
            return self.active_frame_prior;
        }
        self.active_frame_at = Some(now);
        self.active_frame_recompute = recompute;
        let operator_prior = |estimate_px: Option<f64>| {
            let (minimum_px, maximum_px) = operator_limits_px?;
            FrontoParallelLimbusRadiusPrior::from_hard_support(
                estimate_px.unwrap_or(0.5 * (minimum_px + maximum_px)),
                minimum_px,
                maximum_px,
                FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits,
            )
        };
        if !recompute {
            self.recovery_samples.clear();
            self.pending_recovery = None;
            if let Some(manual) = operator_prior(
                self.active_frame_prior
                    .map(|prior| prior.estimate_px)
                    .or_else(|| self.mean_log_radius.map(f64::exp)),
            ) {
                self.active_frame_prior = Some(manual);
            }
            return self.active_frame_prior;
        }
        if let Some(fixed) = self.fixed_reference {
            let prior = operator_prior(Some(fixed.estimate_px)).unwrap_or(fixed);
            self.active_frame_prior = Some(prior);
            return Some(prior);
        }
        // Recovery never changes support in the candidate's own frame. Only
        // a following automatic frame may install the verified consensus.
        if let Some((at, recovered_log_radius)) = self.pending_recovery.take() {
            if operator_limits_px.is_none()
                && now.saturating_duration_since(at) <= LIMBUS_COLD_START_HISTORY
            {
                self.mean_log_radius = Some(recovered_log_radius);
                self.transported_strong_anchor_log_radius = Some(recovered_log_radius);
                self.latest_strong_log_radius = Some(recovered_log_radius);
                self.strong_log_radii =
                    VecDeque::from([recovered_log_radius; LIMBUS_RECOVERY_CONSENSUS]);
                self.cold_start_log_radii.clear();
                self.mean_absolute_log_residual = 0.0;
                self.effective_observations = LIMBUS_RECOVERY_CONSENSUS as f64;
                self.last_observed = Some(at);
                self.fine_log_transport_since_observation = 0.0;
            }
            self.recovery_samples.clear();
        }
        let prediction =
            prediction
                .filter(|prediction| prediction.valid())
                .and_then(|mut prediction| {
                    if prediction.source
                        == FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry
                    {
                        if self.last_observed.is_some_and(|last| {
                            now.saturating_duration_since(last) > LIMBUS_FINE_TRANSPORT_MAX_AGE
                        }) {
                            return None;
                        }
                        // A biased similarity fit must not integrate indefinitely
                        // while every actual anatomical measurement is rejected.
                        let limit = LIMBUS_FINE_UNCONFIRMED_MAX_RATIO.ln();
                        let next = (self.fine_log_transport_since_observation
                            + prediction.scale_ratio.ln())
                        .clamp(-limit, limit);
                        let delta = next - self.fine_log_transport_since_observation;
                        if delta.abs() < 1e-12 {
                            return None;
                        }
                        self.fine_log_transport_since_observation = next;
                        prediction.scale_ratio = delta.exp();
                    } else {
                        // A fresh coarse pose is an independent relocation authority.
                        self.fine_log_transport_since_observation = 0.0;
                    }
                    Some(prediction)
                });
        if let Some(prediction) = prediction {
            // These are log radii in image coordinates, not a fixed physical
            // unit. A visual scale transition changes the coordinate system
            // of the complete posterior. Shifting only its displayed mean
            // would let the unshifted robust history pull the next admitted
            // frame back toward an obsolete apparent size.
            let log_scale = prediction.scale_ratio.ln();
            self.mean_log_radius = self.mean_log_radius.map(|radius| radius + log_scale);
            for sample in &mut self.strong_log_radii {
                *sample += log_scale;
            }
            for (_, sample, _) in &mut self.cold_start_log_radii {
                *sample += log_scale;
            }
            self.transported_strong_anchor_log_radius = self
                .transported_strong_anchor_log_radius
                .map(|radius| radius + log_scale);
            self.latest_strong_log_radius = self
                .latest_strong_log_radius
                .map(|radius| radius + log_scale);
        }
        let Some(estimate_px) = self.mean_log_radius.map(f64::exp) else {
            self.active_frame_prior = operator_prior(None);
            return self.active_frame_prior;
        };
        let sampling_half_width =
            (0.05 + 0.07 / self.effective_observations.max(1.0).sqrt()).clamp(0.05, 0.12);
        let residual_half_width = self.mean_absolute_log_residual.exp() - 1.0;
        let stale_half_width = self.last_observed.map_or(0.0, |last| {
            now.saturating_duration_since(last)
                .as_secs_f64()
                .mul_add(0.035, 0.0)
        });
        let mut fractional_half_width = sampling_half_width
            .max(residual_half_width * 2.5)
            .max(stale_half_width.min(LIMBUS_UNCORROBORATED_MAX_HALF_WIDTH));
        let mut source = FrontoParallelLimbusRadiusPriorSource::TemporalRobustMedian;
        if let Some(prediction) = prediction {
            fractional_half_width = fractional_half_width.max(prediction.fractional_uncertainty);
            source = prediction.source;
        } else {
            // Residuals describe detector disagreement, not proof that the
            // eye changed scale. Do not let a run of inconsistent lid or
            // glasses edges widen its own future candidate bank.
            let elapsed = self.last_observed.map_or(0.0, |last| {
                now.saturating_duration_since(last).as_secs_f64()
            });
            let uncorroborated_half_width = (LIMBUS_UNCORROBORATED_BASE_HALF_WIDTH
                + LIMBUS_UNCORROBORATED_HALF_WIDTH_PER_SECOND * elapsed)
                .clamp(
                    LIMBUS_UNCORROBORATED_BASE_HALF_WIDTH,
                    LIMBUS_UNCORROBORATED_MAX_HALF_WIDTH,
                );
            fractional_half_width = fractional_half_width.min(uncorroborated_half_width);
        }
        let detector_prior =
            limbus_radius_prior_from_expanding_support(estimate_px, fractional_half_width, source);
        // Even with a valid scale transport, detector residual is only a
        // search-quality statistic.  Intersect that proposal with a physical
        // corridor around the latest independently strong radius.  The
        // corridor follows measured whole-ROI scale and may expand by that
        // measurement's uncertainty, but an iris/lid/glasses disagreement
        // can never widen it.  This prevents an adjacent frame from jumping
        // from one end of a residual-inflated candidate bank to the other.
        let automatic_prior = self
            .transported_strong_anchor_log_radius
            .map(f64::exp)
            .and_then(|anchor_px| {
                let elapsed = self.last_observed.map_or(0.0, |last| {
                    now.saturating_duration_since(last).as_secs_f64()
                });
                let uncorroborated_half_width = (LIMBUS_UNCORROBORATED_BASE_HALF_WIDTH
                    + LIMBUS_UNCORROBORATED_HALF_WIDTH_PER_SECOND * elapsed)
                    .clamp(
                        LIMBUS_UNCORROBORATED_BASE_HALF_WIDTH,
                        LIMBUS_UNCORROBORATED_MAX_HALF_WIDTH,
                    );
                let transport_uncertainty =
                    prediction.map_or(0.0, |prediction| prediction.fractional_uncertainty);
                let physical_half_width = uncorroborated_half_width + transport_uncertainty;
                let physical = limbus_radius_prior_from_expanding_support(
                    anchor_px,
                    physical_half_width,
                    source,
                )?;
                let Some(detector) = detector_prior else {
                    return Some(physical);
                };
                let minimum_px = detector.minimum_px.max(physical.minimum_px);
                let maximum_px = detector.maximum_px.min(physical.maximum_px);
                if minimum_px >= 4.0 && maximum_px > minimum_px {
                    FrontoParallelLimbusRadiusPrior::from_hard_support(
                        physical.estimate_px.clamp(minimum_px, maximum_px),
                        minimum_px,
                        maximum_px,
                        source,
                    )
                } else {
                    // A disjoint robust median and last-strong corridor
                    // signals stale detector history. Preserve the recent
                    // physical anchor rather than dropping all size
                    // constraints.
                    Some(physical)
                }
            })
            .or(detector_prior);
        let prior = operator_prior(Some(estimate_px)).or(automatic_prior);
        self.active_frame_prior = prior;
        prior
    }

    /// Update the temporal posterior only from an independently strong limbus
    /// road. Returns false when the observation violates the support frozen
    /// for the current frame, so a rejected lid/forehead curve cannot train
    /// the prior.
    pub fn observe_strong_ellipse(
        &mut self,
        now: Instant,
        major_radius: f64,
        minor_radius: f64,
        confidence: f64,
    ) -> bool {
        let Some(radius_px) =
            FrontoParallelLimbusRadiusPrior::fronto_parallel_radius_px(major_radius, minor_radius)
        else {
            return false;
        };
        if self
            .active_frame_prior
            .is_some_and(|prior| !prior.admits_kinematically_supported_radius(radius_px))
        {
            return false;
        }
        if self.fixed_reference.is_some() {
            self.last_observed = Some(now);
            self.fine_log_transport_since_observation = 0.0;
            return true;
        }
        let measurement = radius_px.ln();
        // The robust median defines the shared candidate bank, but two
        // opposite ends of that bank must not alternate between adjacent
        // publications. Compare the completed strong road with the latest
        // admitted strong radius after both have been transported by any
        // independent whole-ROI scale prediction. Operator/fixed supports are
        // explicit authorities, and cold start still establishes identity
        // from its three-road consensus below.
        let latest_strong_kinematically_supported =
            self.active_frame_prior
                .map_or(true, |prior| match prior.source {
                    FrontoParallelLimbusRadiusPriorSource::FixedReference
                    | FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits
                    | FrontoParallelLimbusRadiusPriorSource::CurrentFrameGeometry
                    | FrontoParallelLimbusRadiusPriorSource::CurrentFrameUnverifiedGeometry => true,
                    FrontoParallelLimbusRadiusPriorSource::TemporalRobustMedian
                    | FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose
                    | FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry => {
                        self.last_observed.is_none_or(|last| {
                            now.saturating_duration_since(last)
                                > LIMBUS_LATEST_STRONG_CONTINUITY_HORIZON
                                || self.latest_strong_log_radius.is_none_or(|latest| {
                                    (measurement - latest).abs()
                                        <= LIMBUS_STRONG_OBSERVATION_MAX_LOG_INNOVATION
                                })
                        })
                    }
                });
        if !latest_strong_kinematically_supported {
            return false;
        }
        let confidence = confidence.clamp(0.0, 1.0);
        let previous = self.mean_log_radius;
        if self.mean_log_radius.is_none()
            && !self.active_frame_prior.is_some_and(|prior| {
                prior.source == FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits
            })
        {
            self.cold_start_log_radii
                .push_back((now, measurement, confidence));
            while self.cold_start_log_radii.front().is_some_and(|(at, _, _)| {
                now.saturating_duration_since(*at) > LIMBUS_COLD_START_HISTORY
            }) {
                self.cold_start_log_radii.pop_front();
            }
            while self.cold_start_log_radii.len() > 12 {
                self.cold_start_log_radii.pop_front();
            }
            let mut ordered = self
                .cold_start_log_radii
                .iter()
                .copied()
                .collect::<Vec<_>>();
            ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
            let maximum_span = LIMBUS_COLD_START_MAX_RATIO.ln();
            let mut best = (0usize, 0usize, 0usize, 0.0f64, None::<Instant>);
            for begin in 0..ordered.len() {
                let mut end = begin;
                while end < ordered.len() && ordered[end].1 - ordered[begin].1 <= maximum_span {
                    end += 1;
                }
                let count = end - begin;
                let confidence_sum = ordered[begin..end].iter().map(|entry| entry.2).sum::<f64>();
                let newest = ordered[begin..end].iter().map(|entry| entry.0).max();
                if count > best.0
                    || (count == best.0 && confidence_sum > best.3)
                    || (count == best.0 && confidence_sum == best.3 && newest > best.4)
                {
                    best = (count, begin, end, confidence_sum, newest);
                }
            }
            if best.0 < LIMBUS_COLD_START_CONSENSUS {
                // The measurement is individually admissible but does not yet
                // own the between-frame physical-size authority.
                return true;
            }
            let consensus = &ordered[best.1..best.2];
            let middle = consensus.len() / 2;
            let robust_center = if consensus.len() % 2 == 0 {
                0.5 * (consensus[middle - 1].1 + consensus[middle].1)
            } else {
                consensus[middle].1
            };
            self.strong_log_radii = consensus
                .iter()
                .rev()
                .take(7)
                .map(|entry| entry.1)
                .collect();
            self.mean_log_radius = Some(robust_center);
            self.transported_strong_anchor_log_radius = Some(robust_center);
            // None of these cold votes was published: the consensus is the
            // first physical-size authority. Seeding the adjacent-publication
            // gate from the newest raw vote can immediately reject the
            // consensus-centered next frame whenever that vote sits near the
            // allowed 12% cold-start edge.
            self.latest_strong_log_radius = Some(robust_center);
            self.effective_observations = consensus
                .iter()
                .map(|entry| entry.2.max(0.20))
                .sum::<f64>()
                .min(64.0);
            self.last_observed = Some(now);
            self.fine_log_transport_since_observation = 0.0;
            self.cold_start_log_radii.clear();
            return true;
        }
        if self.strong_log_radii.is_empty() {
            // Explicit operator limits are already an independent physical
            // support, so their first in-band observation may initialize the
            // automatic posterior immediately.
            self.strong_log_radii.push_back(measurement);
        } else {
            self.strong_log_radii.push_back(measurement);
            while self.strong_log_radii.len() > 7 {
                self.strong_log_radii.pop_front();
            }
        }
        let mut ordered = self.strong_log_radii.iter().copied().collect::<Vec<_>>();
        ordered.sort_by(f64::total_cmp);
        let middle = ordered.len() / 2;
        let robust_center = if ordered.len() % 2 == 0 {
            0.5 * (ordered[middle - 1] + ordered[middle])
        } else {
            ordered[middle]
        };
        let residual = previous.map_or(0.0, |prior| (measurement - prior).abs());
        let residual_alpha = (0.10 + 0.15 * confidence).clamp(0.10, 0.25);
        self.mean_log_radius = Some(robust_center);
        let anchor = self
            .transported_strong_anchor_log_radius
            .unwrap_or(robust_center);
        self.transported_strong_anchor_log_radius = Some(
            anchor
                + (robust_center - anchor)
                    * LIMBUS_STRONG_ANCHOR_ROBUST_ADAPTATION
                    * confidence.max(0.20),
        );
        self.mean_absolute_log_residual =
            self.mean_absolute_log_residual * (1.0 - residual_alpha) + residual * residual_alpha;
        self.effective_observations =
            (self.effective_observations * 0.96 + confidence.max(0.20)).min(64.0);
        self.latest_strong_log_radius = Some(measurement);
        self.last_observed = Some(now);
        self.fine_log_transport_since_observation = 0.0;
        true
    }

    /// Use for real source observations, including asynchronous SAM results.
    /// Redraws may reuse admission, but cannot refresh anatomical timestamps,
    /// train the posterior or add cold-start/recovery votes.
    pub(crate) fn observe_independent_ellipse_for_active_frame(
        &mut self,
        now: Instant,
        observation: LimbusRadiusObservation,
    ) -> LimbusRadiusAdmission {
        use LimbusRadiusAdmission as Admission;
        if self.active_frame_at != Some(now) {
            return Admission::StaleObservation;
        }
        let ellipse = observation.ellipse_sensor_px;
        let Some(radius) = FrontoParallelLimbusRadiusPrior::fronto_parallel_radius_px(
            ellipse.major_radius,
            ellipse.minor_radius,
        ) else {
            return Admission::InvalidObservation;
        };
        if !ellipse.center.0.is_finite()
            || !ellipse.center.1.is_finite()
            || !ellipse.angle.is_finite()
            || !observation.confidence.is_finite()
            || observation.confidence <= 0.0
        {
            return Admission::InvalidObservation;
        }
        if let Some((previous, lineage, admission)) = self.last_independent_observation {
            if previous.roi != observation.exposure.roi {
                return Admission::InvalidObservation;
            }
            if previous.clock.domain == observation.exposure.clock.domain
                && (observation.exposure.clock.epoch < previous.clock.epoch
                    || (observation.exposure.clock == previous.clock
                        && observation.lineage < lineage))
            {
                return Admission::StaleObservation;
            }
            if previous.roi == observation.exposure.roi
                && previous.clock == observation.exposure.clock
                && lineage == observation.lineage
            {
                if observation.exposure.timestamp_ns < previous.timestamp_ns {
                    return Admission::StaleObservation;
                }
                if observation.exposure.timestamp_ns == previous.timestamp_ns {
                    return if matches!(
                        admission,
                        Admission::Published
                            | Admission::RecoveryReady
                            | Admission::ColdStart { .. }
                    ) && self
                        .active_frame_prior
                        .is_some_and(|prior| prior.admits_kinematically_supported_radius(radius))
                    {
                        Admission::Published
                    } else if admission == Admission::Published {
                        Admission::ContinuityRejected
                    } else {
                        admission
                    };
                }
            } else {
                self.recovery_samples.clear();
                self.pending_recovery = None;
                self.cold_start_log_radii.clear();
            }
        }
        let admission = if self.mean_log_radius.is_none() && self.active_frame_prior.is_none() {
            if self.active_frame_recompute
                && observation.complete_in_source_roi
                && observation.confidence >= 0.75
            {
                match self.collect_independent_consensus(now, observation, radius) {
                    Admission::RecoveryVerifying { votes } => Admission::ColdStart { votes },
                    Admission::RecoveryReady => Admission::ColdStart {
                        votes: LIMBUS_RECOVERY_CONSENSUS,
                    },
                    other => other,
                }
            } else {
                self.recovery_samples.clear();
                Admission::InvalidObservation
            }
        } else if self.observe_strong_ellipse_for_active_frame(
            now,
            ellipse.major_radius,
            ellipse.minor_radius,
            observation.confidence,
        ) {
            self.recovery_samples.clear();
            self.pending_recovery = None;
            if self.active_frame_prior.is_some() {
                Admission::Published
            } else {
                Admission::ColdStart {
                    votes: if self.mean_log_radius.is_some() {
                        LIMBUS_COLD_START_CONSENSUS
                    } else {
                        self.cold_start_log_radii
                            .len()
                            .min(LIMBUS_COLD_START_CONSENSUS)
                    },
                }
            }
        } else {
            self.observe_recovery_candidate(now, observation, radius)
        };
        self.last_independent_observation =
            Some((observation.exposure, observation.lineage, admission));
        admission
    }

    fn observe_recovery_candidate(
        &mut self,
        now: Instant,
        observation: LimbusRadiusObservation,
        radius: f64,
    ) -> LimbusRadiusAdmission {
        use LimbusRadiusAdmission as Admission;
        let Some(prior) = self.active_frame_prior else {
            return Admission::InvalidObservation;
        };
        if !prior.admits_radius(radius) {
            self.recovery_samples.clear();
            return Admission::HardSupportRejected;
        }
        if !self.active_frame_recompute
            || self.fixed_reference.is_some()
            || prior.source == FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits
            || !observation.complete_in_source_roi
            || observation.confidence < 0.75
            || self
                .last_observed
                .is_none_or(|last| now.saturating_duration_since(last) < LIMBUS_RECOVERY_AFTER)
        {
            self.recovery_samples.clear();
            return Admission::ContinuityRejected;
        }
        self.collect_independent_consensus(now, observation, radius)
    }

    fn collect_independent_consensus(
        &mut self,
        now: Instant,
        observation: LimbusRadiusObservation,
        radius: f64,
    ) -> LimbusRadiusAdmission {
        use LimbusRadiusAdmission as Admission;
        let timestamp_ns = observation.exposure.timestamp_ns;
        self.recovery_samples.retain(|sample| {
            timestamp_ns.saturating_sub(sample.timestamp_ns) <= LIMBUS_RECOVERY_WINDOW_NS
        });
        let ellipse = observation.ellipse_sensor_px;
        let log_radius = radius.ln();
        if self.recovery_samples.iter().any(|sample| {
            (sample.log_radius - log_radius).abs() > 1.018f64.ln()
                || (ellipse.center.0 - sample.ellipse_sensor_px.center.0)
                    .hypot(ellipse.center.1 - sample.ellipse_sensor_px.center.1)
                    > radius * 0.40
                || (ellipse.minor_radius / radius
                    - sample.ellipse_sensor_px.minor_radius / sample.log_radius.exp())
                .abs()
                    > 0.15
        }) {
            self.recovery_samples.clear();
        }
        self.recovery_samples.push_back(LimbusRecoverySample {
            timestamp_ns,
            ellipse_sensor_px: ellipse,
            log_radius,
        });
        while self.recovery_samples.len() > 12 {
            self.recovery_samples.pop_front();
        }
        if self.recovery_samples.len() < LIMBUS_RECOVERY_CONSENSUS
            || self.recovery_samples.front().is_none_or(|sample| {
                timestamp_ns.saturating_sub(sample.timestamp_ns) < LIMBUS_RECOVERY_MIN_SPAN_NS
            })
        {
            return Admission::RecoveryVerifying {
                votes: self.recovery_samples.len().min(LIMBUS_RECOVERY_CONSENSUS),
            };
        }
        let mut radii = self
            .recovery_samples
            .iter()
            .map(|sample| sample.log_radius)
            .collect::<Vec<_>>();
        radii.sort_by(f64::total_cmp);
        let middle = radii.len() / 2;
        let median = if radii.len() % 2 == 0 {
            0.5 * (radii[middle - 1] + radii[middle])
        } else {
            radii[middle]
        };
        self.pending_recovery = Some((now, median));
        Admission::RecoveryReady
    }

    /// Admit a strong measurement only when it belongs to the frame whose
    /// support is currently frozen. This is the mutation path for shared
    /// synchronous/asynchronous segmentation state: a late worker result can
    /// be rendered as a diagnostic, but cannot rewind or train the common
    /// apparent-size posterior.
    pub fn observe_strong_ellipse_for_active_frame(
        &mut self,
        now: Instant,
        major_radius: f64,
        minor_radius: f64,
        confidence: f64,
    ) -> bool {
        self.active_frame_at == Some(now)
            && self.observe_strong_ellipse(now, major_radius, minor_radius, confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi_evidence::{RoiId, SourceClock};

    fn established_tracker(started: Instant, radius: f64) -> FrontoParallelLimbusRadiusTracker {
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for index in 0..3 {
            let now = started + Duration::from_millis(index * 10);
            tracker.begin_frame(now, None);
            assert!(tracker.observe_strong_ellipse(now, radius, radius * 0.8, 1.0));
        }
        tracker.begin_frame(started + Duration::from_millis(30), None);
        tracker
    }

    fn observation(ms: u64, radius: f64) -> LimbusRadiusObservation {
        LimbusRadiusObservation {
            exposure: ExposureKey {
                roi: RoiId(0),
                clock: SourceClock {
                    domain: 0,
                    epoch: 0,
                },
                sequence: ms,
                timestamp_ns: 1 + ms * 1_000_000,
            },
            lineage: 1,
            ellipse_sensor_px: Ellipse {
                center: (4_200.0, 3_200.0),
                major_radius: radius,
                minor_radius: radius * 0.8,
                angle: 0.1,
            },
            confidence: 0.95,
            complete_in_source_roi: true,
        }
    }

    #[test]
    fn limbus_recovery_stops_unconfirmed_fine_odometry_integrating_forever() {
        let started = Instant::now();
        let mut tracker = established_tracker(started, 87.0);
        for index in 2..400 {
            let prior = tracker
                .begin_frame(
                    started + Duration::from_millis(index * 20),
                    Some(FrontoParallelLimbusScalePrediction::fine_visual_odometry(
                        0.98, 0.02,
                    )),
                )
                .unwrap();
            assert!(prior.estimate_px >= 87.0 / 1.15 - 1e-8, "{prior:?}");
        }
        // Unlike whole-ROI drift, a new coarse relocation remains useful.
        let before = tracker.established_dynamic_radius_px().unwrap();
        let coarse = tracker
            .begin_frame(
                started + Duration::from_secs(40),
                Some(FrontoParallelLimbusScalePrediction::coarse_semantic_pose(
                    1.5, 0.05,
                )),
            )
            .unwrap();
        assert!((coarse.estimate_px / before - 1.5).abs() < 1e-9);
    }

    #[test]
    fn limbus_recovery_reestablishes_68_to_87_only_after_fresh_consensus_and_next_frame() {
        let started = Instant::now();
        let mut tracker = established_tracker(started, 68.0);
        for (index, ms) in [10_000, 10_150, 10_300].into_iter().enumerate() {
            let now = started + Duration::from_millis(ms);
            let prior = tracker.begin_frame(now, None).unwrap();
            assert!(prior.admits_radius(87.0));
            assert!(!prior.admits_kinematically_supported_radius(87.0));
            let admission =
                tracker.observe_independent_ellipse_for_active_frame(now, observation(ms, 87.0));
            assert_eq!(
                admission,
                if index < 2 {
                    LimbusRadiusAdmission::RecoveryVerifying { votes: index + 1 }
                } else {
                    LimbusRadiusAdmission::RecoveryReady
                }
            );
            assert_eq!(tracker.active_frame_prior(), Some(prior));
            assert!((tracker.established_dynamic_radius_px().unwrap() - 68.0).abs() < 1e-8);
        }
        let now = started + Duration::from_millis(10_310);
        let recovered = tracker.begin_frame(now, None).unwrap();
        assert!(recovered.admits_kinematically_supported_radius(87.0));
        assert_eq!(
            tracker.observe_independent_ellipse_for_active_frame(now, observation(10_300, 87.0)),
            LimbusRadiusAdmission::Published
        );
        assert_eq!(
            tracker.last_observed,
            Some(started + Duration::from_millis(10_300))
        );
    }

    #[test]
    fn limbus_recovery_cached_and_out_of_order_sam_answers_never_add_votes() {
        let started = Instant::now();
        let mut tracker = established_tracker(started, 68.0);
        for ms in [10_000, 10_150, 10_300, 10_600] {
            let now = started + Duration::from_millis(ms);
            tracker.begin_frame(now, None);
            let mut cached = observation(10_000, 87.0);
            // Even a different sequence tag cannot invent a new exposure.
            cached.exposure.sequence = ms;
            assert_eq!(
                tracker.observe_independent_ellipse_for_active_frame(now, cached),
                LimbusRadiusAdmission::RecoveryVerifying { votes: 1 }
            );
        }
        assert!(tracker.pending_recovery.is_none());
        let now = started + Duration::from_millis(10_700);
        tracker.begin_frame(now, None);
        assert_eq!(
            tracker.observe_independent_ellipse_for_active_frame(now, observation(9_000, 87.0)),
            LimbusRadiusAdmission::StaleObservation
        );
        assert_eq!(tracker.recovery_samples.len(), 1);
    }

    #[test]
    fn limbus_recovery_cached_admission_does_not_refresh_anatomy_or_motion_budget() {
        let started = Instant::now();
        let mut tracker = established_tracker(started, 68.0);
        let source = observation(100, 68.0);
        let now = started + Duration::from_millis(100);
        tracker.begin_frame(now, None);
        assert!(
            tracker
                .observe_independent_ellipse_for_active_frame(now, source)
                .published()
        );
        let observations = tracker.effective_observations;
        for ms in [200, 300, 500] {
            let now = started + Duration::from_millis(ms);
            tracker.begin_frame(
                now,
                Some(FrontoParallelLimbusScalePrediction::fine_visual_odometry(
                    1.001, 0.02,
                )),
            );
            assert!(
                tracker
                    .observe_independent_ellipse_for_active_frame(now, source)
                    .published()
            );
        }
        assert_eq!(tracker.effective_observations, observations);
        assert_eq!(
            tracker.last_observed,
            Some(started + Duration::from_millis(100))
        );
        assert!(tracker.fine_log_transport_since_observation > 0.002);
    }

    #[test]
    fn limbus_recovery_sparse_cold_start_uses_source_frames_not_redraws() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for (index, ms) in [0, 900, 1_800].into_iter().enumerate() {
            let now = started + Duration::from_millis(ms);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert_eq!(
                tracker.observe_independent_ellipse_for_active_frame(now, observation(ms, 87.0)),
                LimbusRadiusAdmission::ColdStart { votes: index + 1 }
            );
            assert_eq!(tracker.active_frame_prior(), None);
        }
        let now = started + Duration::from_millis(1_810);
        let prior = tracker.begin_frame(now, None).unwrap();
        assert!(prior.admits_kinematically_supported_radius(87.0));
        assert!(
            tracker
                .observe_independent_ellipse_for_active_frame(now, observation(1_800, 87.0))
                .published()
        );
    }

    #[test]
    fn limbus_recovery_rejects_weak_clipped_incoherent_and_out_of_bounds_votes() {
        let started = Instant::now();
        for kind in 0..6 {
            let mut tracker = established_tracker(started, 68.0);
            for index in 0..6 {
                let ms = 10_000 + index * 150;
                let now = started + Duration::from_millis(ms);
                tracker.begin_frame(now, None);
                let mut sample = observation(ms, 87.0);
                match kind {
                    0 => sample.confidence = 0.5,
                    1 => sample.complete_in_source_roi = false,
                    2 => {
                        sample.ellipse_sensor_px.major_radius =
                            if index % 2 == 0 { 84.0 } else { 87.0 }
                    }
                    3 => sample.ellipse_sensor_px.major_radius = 200.0,
                    4 => sample.ellipse_sensor_px.center.0 += (index % 2) as f64 * 100.0,
                    5 => sample.ellipse_sensor_px.minor_radius = 1.0,
                    _ => unreachable!(),
                }
                let admission = tracker.observe_independent_ellipse_for_active_frame(now, sample);
                assert!(!admission.published(), "kind {kind}: {admission:?}");
                assert!(tracker.pending_recovery.is_none(), "kind {kind}");
                assert_eq!(
                    tracker.last_observed,
                    Some(started + Duration::from_millis(20))
                );
            }
        }
    }

    #[test]
    fn limbus_recovery_does_not_pool_detector_lineages_or_clock_epochs() {
        let started = Instant::now();
        for new_clock in [false, true] {
            let mut tracker = established_tracker(started, 68.0);
            for ms in [10_000, 10_150] {
                let now = started + Duration::from_millis(ms);
                tracker.begin_frame(now, None);
                tracker.observe_independent_ellipse_for_active_frame(now, observation(ms, 87.0));
            }
            let now = started + Duration::from_millis(10_300);
            tracker.begin_frame(now, None);
            let mut sample = observation(10_300, 87.0);
            if new_clock {
                sample.exposure.clock.epoch = 1;
            } else {
                sample.lineage = 2;
            }
            assert_eq!(
                tracker.observe_independent_ellipse_for_active_frame(now, sample),
                LimbusRadiusAdmission::RecoveryVerifying { votes: 1 }
            );
            assert_eq!(
                tracker
                    .observe_independent_ellipse_for_active_frame(now, observation(10_300, 87.0)),
                LimbusRadiusAdmission::StaleObservation
            );
            assert!(tracker.pending_recovery.is_none());
        }
    }

    #[test]
    fn limbus_recovery_frozen_or_operator_support_cancels_pending_promotion() {
        let started = Instant::now();
        for manual in [false, true] {
            let mut tracker = established_tracker(started, 68.0);
            for ms in [10_000, 10_150, 10_300] {
                let now = started + Duration::from_millis(ms);
                tracker.begin_frame(now, None);
                tracker.observe_independent_ellipse_for_active_frame(now, observation(ms, 87.0));
            }
            assert!(tracker.pending_recovery.is_some());
            let now = started + Duration::from_millis(10_350);
            let prior = tracker
                .begin_frame_controlled(now, None, manual, manual.then_some((50.0, 75.0)))
                .unwrap();
            assert!((prior.estimate_px - 68.0).abs() < 1e-8);
            assert!(tracker.pending_recovery.is_none());
            if manual {
                assert_eq!(
                    prior.source,
                    FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits
                );
            }
            assert!(
                !tracker
                    .observe_independent_ellipse_for_active_frame(now, observation(10_350, 87.0))
                    .published()
            );
        }
    }

    #[test]
    fn limbus_recovery_requires_dwell_and_recent_source_evidence() {
        let started = Instant::now();
        let mut tracker = established_tracker(started, 68.0);
        for ms in [10_000, 10_010, 10_020] {
            let now = started + Duration::from_millis(ms);
            tracker.begin_frame(now, None);
            tracker.observe_independent_ellipse_for_active_frame(now, observation(ms, 87.0));
            assert!(tracker.pending_recovery.is_none());
        }
        let now = started + Duration::from_millis(14_000);
        tracker.begin_frame(now, None);
        assert_eq!(
            tracker.observe_independent_ellipse_for_active_frame(now, observation(14_000, 87.0)),
            LimbusRadiusAdmission::RecoveryVerifying { votes: 1 }
        );
        assert!(tracker.pending_recovery.is_none());
    }

    #[test]
    fn limbus_recovery_never_overrides_adjacent_publication_continuity() {
        let started = Instant::now();
        let mut tracker = established_tracker(started, 68.0);
        for ms in [100, 250, 400, 550] {
            let now = started + Duration::from_millis(ms);
            let prior = tracker.begin_frame(now, None).unwrap();
            assert!(prior.admits_radius(70.0));
            assert_eq!(
                tracker.observe_independent_ellipse_for_active_frame(now, observation(ms, 70.0)),
                LimbusRadiusAdmission::ContinuityRejected
            );
            assert!(tracker.pending_recovery.is_none());
        }
    }

    #[test]
    fn fine_scale_transport_cannot_turn_detector_residual_into_motion_support() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for index in 0..3 {
            let now = started + Duration::from_millis(index * 10);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse(now, 100.0, 78.0, 1.0));
        }

        // Model a run in which otherwise admissible detector curves disagreed
        // strongly. This statistic may rank searches, but it cannot prove a
        // physical move toward or away from the camera.
        tracker.mean_absolute_log_residual = 0.18;
        let prior = tracker
            .begin_frame(
                started + Duration::from_millis(120),
                Some(FrontoParallelLimbusScalePrediction::fine_visual_odometry(
                    0.985, 0.028,
                )),
            )
            .expect("fine RAW odometry should transport the established radius");
        assert_eq!(
            prior.source,
            FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry
        );
        assert!((prior.estimate_px - 98.5).abs() < 1.0e-9, "{prior:?}");
        let half_width = (prior.maximum_px - prior.minimum_px) / (2.0 * prior.estimate_px);
        assert!(half_width <= 0.0671, "{prior:?}");
        assert!(prior.admits_radius(98.5));
        assert!(!prior.admits_radius(80.0), "{prior:?}");
        assert!(!prior.admits_radius(120.0), "{prior:?}");
    }
}
