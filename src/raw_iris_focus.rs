use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const MIN_IRIS_SCLERA_RADIAL_CONTRAST: usize = 24;
const OUTER_IRIS_SWEEP_BRANCH_SAMPLES: usize = 13;
// Keep the original 43-ray density over the difficult top/bottom arcs, but
// sample the usually cleaner lateral limbus three times as densely.  The
// weighted clock sectors are 2-4 and 7-10, yielding 79 cyclically ordered rays.
const OUTER_IRIS_DENSE_EVIDENCE_SAMPLES: usize = 79;
const OUTER_IRIS_MIN_WIDE_LUMA_CONTRAST: f64 = 24.0;
const OUTER_IRIS_MIN_LUMA_SUPPORT: f64 = 16.0;
const OUTER_IRIS_TRACK_MIN_CONFIDENCE: f64 = 0.12;
const OUTER_IRIS_MIN_SEARCH_SCALE: f64 = 0.66;
// Driving starts from the whole visible eye aperture, so its native limbus
// seed must also be allowed to contract to the iris before the pupil-headed
// road takes over. Keep this recovery range local to Driving: applying it to
// the general outer-boundary detector admits inner pupil and eyelid aliases in
// its stricter synthetic geometry contracts.
const DRIVING_OUTER_IRIS_MIN_SEARCH_SCALE: f64 = 0.50;
const OUTER_IRIS_MAX_SEARCH_SCALE: f64 = 1.50;
const OUTER_IRIS_SYSTEM_BUDGET: Duration = Duration::from_millis(14);
// Driving is the high-fidelity RAW path. It always evaluates the complete ray
// and radial lattices at native ROI coordinates; unlike the ordinary bounded
// preview detector it never responds to a prior overrun by skipping every
// second or fourth sample on a later frame.
const DRIVING_OUTER_IRIS_SYSTEM_BUDGET: Duration = Duration::from_millis(32);
// The pupil-centered fork recovery runs only after an ordinary complete eye
// road has already established identity. Give that rare cold/reacquisition
// lap enough time to finish the same ray lattice at all three seed scales;
// the normal per-frame outer tracker retains its stricter 14 ms budget.
const DRIVING_OUTER_IRIS_RECOVERY_BUDGET: Duration = Duration::from_millis(24);
const OUTER_IRIS_RAY_BATCH_BUDGET: Duration = Duration::from_millis(8);
const OUTER_IRIS_RAY_BUDGET: Duration = Duration::from_micros(1_250);
// Host scheduling jitter must not permanently lower the geometric search
// density. Require sustained pressure before degrading, then cautiously
// recover one density level after a long clean run. This keeps live work
// bounded without making the remainder of a replay depend on one slow frame.
const OUTER_IRIS_BUDGET_PRESSURE_FRAMES: u8 = 3;
const OUTER_IRIS_BUDGET_RECOVERY_FRAMES: u16 = 120;
// Refinement is bounded by deterministic operation counts, not microsecond
// cut-offs. Accepting whichever sectors happened to finish inside a 425 us
// scheduling slice made an identical RAW frame produce different conics under
// ordinary host load; that small seed variation could then send Driving down
// a completely different road. Five complete 79-ray passes are cheaper than
// the outer detector's wall budget on the native ROI and make the result a
// pure function of the candidate lattice.
const OUTER_IRIS_REFINEMENT_ITERATIONS: usize = 5;
// Once the discrete meridian road has been accepted, measure every accepted
// contact on the fixed 79-ray lattice. This stage never broadens the search;
// its operation count is statically bounded and it only supplies a signed
// force and fit weight for already accepted contacts.
const OUTER_IRIS_ANALOG_PROFILE_OFFSETS: [f64; 9] =
    [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
const OUTER_IRIS_ANALOG_APERTURES: [f64; 3] = [0.85, 1.70, 3.10];
// A profile below either floor may still be useful telemetry, but a whole
// conic must not move on a collection of weak or poorly localized edges.
const OUTER_IRIS_ANALOG_MIN_MEAN_POWER: f64 = 0.60;
const OUTER_IRIS_ANALOG_MIN_MEAN_CERTAINTY: f64 = 0.28;
const OUTER_IRIS_REFLECTANCE_FUSED_STRENGTH: f64 = 0.35;
const OUTER_IRIS_REFLECTANCE_LIGHT_STRENGTH: f64 = 0.50;
/// One-way intrinsic-image corroboration. It never subtracts established
/// evidence or replaces a clean candidate's score; it can only rescue an edge
/// where pure reflectance remains scleral while the brightness-aware posterior
/// has collapsed under an achromatic shadow.
const OUTER_IRIS_INTRINSIC_SHADOW_BONUS_STRENGTH: f64 = 0.90;

/// Conservative camera-and-anatomy envelope for treating an image conic as
/// the projection of the physical limbus.
///
/// This is intentionally not called a calibration.  The lower focal bound
/// sits below both the current relative solve (~3250 px) and the checkerboard
/// preview value (4737.6 px), while the upper bound sits above both.  Focal
/// uncertainty changes angular scale; it cannot by itself explain arbitrary
/// anisotropy on square sensor pixels.  `maximum_local_metric_anisotropy`
/// reserves a further six percent for residual central lens distortion,
/// pixel-aspect uncertainty, and use before a full-aperture calibration has
/// been accepted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentralCameraLimbusProjectionEnvelope {
    pub minimum_focal_length_px: f64,
    pub maximum_focal_length_px: f64,
    pub maximum_pixel_aspect_error: f64,
    pub maximum_local_metric_anisotropy: f64,
    pub maximum_anatomical_surface_tilt_radians: f64,
    pub uncalibrated_central_ray_slack_radians: f64,
    pub maximum_limbus_half_angle_radians: f64,
    pub absolute_minimum_minor_to_major: f64,
}

pub const PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE: CentralCameraLimbusProjectionEnvelope =
    CentralCameraLimbusProjectionEnvelope {
        minimum_focal_length_px: 3_000.0,
        maximum_focal_length_px: 5_200.0,
        maximum_pixel_aspect_error: 0.01,
        maximum_local_metric_anisotropy: 1.06,
        maximum_anatomical_surface_tilt_radians: 55.0 * PI / 180.0,
        uncalibrated_central_ray_slack_radians: 1.5 * PI / 180.0,
        maximum_limbus_half_angle_radians: 4.0 * PI / 180.0,
        absolute_minimum_minor_to_major: 0.47,
    };

/// Typed interpretation of a canonical projected-limbus ellipse.  The
/// implied tilt is the weak-perspective circle-plane tilt, `acos(b/a)`.
/// `minimum_minor_to_major` is more permissive than the nominal anatomical
/// bound because it already includes the worst focal, aperture, and local
/// camera-metric allowances above.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimbusProjectionAssessment {
    pub major_radius_px: f64,
    pub minor_radius_px: f64,
    pub minor_to_major: f64,
    /// `acos(b/a)` before undoing the bounded local camera-metric anisotropy.
    /// This is an image-derived tilt proxy, not a calibrated gaze angle.
    pub uncorrected_image_implied_tilt_radians: f64,
    /// Maximum physical plane tilt before the camera-metric allowance.
    pub maximum_supported_surface_tilt_radians: f64,
    /// Equivalent image-conic tilt after applying every conservative camera
    /// allowance. This is the value directly comparable to the uncorrected
    /// image proxy above.
    pub maximum_supported_image_tilt_radians: f64,
    pub minimum_minor_to_major: f64,
}

impl CentralCameraLimbusProjectionEnvelope {
    pub fn assess_axes(
        self,
        axis_a_radius_px: f64,
        axis_b_radius_px: f64,
    ) -> Option<LimbusProjectionAssessment> {
        if !axis_a_radius_px.is_finite()
            || !axis_b_radius_px.is_finite()
            || axis_a_radius_px <= 0.0
            || axis_b_radius_px <= 0.0
            || !self.minimum_focal_length_px.is_finite()
            || !self.maximum_focal_length_px.is_finite()
            || self.minimum_focal_length_px <= 0.0
            || self.maximum_focal_length_px < self.minimum_focal_length_px
            || !self.maximum_local_metric_anisotropy.is_finite()
            || self.maximum_local_metric_anisotropy < 1.0
        {
            return None;
        }
        let major_radius_px = axis_a_radius_px.max(axis_b_radius_px);
        let minor_radius_px = axis_a_radius_px.min(axis_b_radius_px);
        let minor_to_major = minor_radius_px / major_radius_px;
        let limbus_half_angle = (major_radius_px / self.minimum_focal_length_px)
            .atan()
            .min(self.maximum_limbus_half_angle_radians.max(0.0));
        let maximum_supported_tilt_radians = (self.maximum_anatomical_surface_tilt_radians
            + self.uncalibrated_central_ray_slack_radians
            + limbus_half_angle)
            .clamp(0.0, PI * 0.5 - 1.0e-6);
        let minimum_minor_to_major = (maximum_supported_tilt_radians.cos()
            / self.maximum_local_metric_anisotropy)
            .max(self.absolute_minimum_minor_to_major)
            .clamp(0.0, 1.0);
        Some(LimbusProjectionAssessment {
            major_radius_px,
            minor_radius_px,
            minor_to_major,
            uncorrected_image_implied_tilt_radians: minor_to_major.clamp(0.0, 1.0).acos(),
            maximum_supported_surface_tilt_radians: maximum_supported_tilt_radians,
            maximum_supported_image_tilt_radians: minimum_minor_to_major.acos(),
            minimum_minor_to_major,
        })
    }

    pub fn admits_axes(self, axis_a_radius_px: f64, axis_b_radius_px: f64) -> bool {
        self.assess_axes(axis_a_radius_px, axis_b_radius_px)
            .is_some_and(|assessment| {
                assessment.minor_to_major + 1.0e-12 >= assessment.minimum_minor_to_major
            })
    }

    pub fn maximum_major_to_minor(self, major_radius_px: f64) -> Option<f64> {
        let assessment = self.assess_axes(major_radius_px, major_radius_px)?;
        Some(1.0 / assessment.minimum_minor_to_major.max(1.0e-9))
    }
}

pub fn assess_projected_circular_limbus_axes(
    axis_a_radius_px: f64,
    axis_b_radius_px: f64,
) -> Option<LimbusProjectionAssessment> {
    PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.assess_axes(axis_a_radius_px, axis_b_radius_px)
}

pub fn projected_circular_limbus_axes_plausible(
    axis_a_radius_px: f64,
    axis_b_radius_px: f64,
) -> bool {
    PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.admits_axes(axis_a_radius_px, axis_b_radius_px)
}

fn outer_iris_evidence_angle(index: usize) -> f64 {
    // Angles increase clockwise in image coordinates: 3 o'clock is zero,
    // 6 is PI/2, 9 is PI, and 12 is 3*PI/2.  Each segment stores its ending
    // angle in degrees and its relative ray density.
    const SEGMENTS: [(f64, f64); 5] = [
        (30.0, 3.0),
        (120.0, 1.0),
        (210.0, 3.0),
        (330.0, 1.0),
        (360.0, 3.0),
    ];
    const WEIGHTED_DEGREES: f64 = 660.0;
    let mut weighted_offset = index.min(OUTER_IRIS_DENSE_EVIDENCE_SAMPLES - 1) as f64
        * WEIGHTED_DEGREES
        / OUTER_IRIS_DENSE_EVIDENCE_SAMPLES as f64;
    let mut segment_start = 0.0;
    for (segment_end, density) in SEGMENTS {
        let weighted_width = (segment_end - segment_start) * density;
        if weighted_offset < weighted_width {
            return (segment_start + weighted_offset / density).to_radians();
        }
        weighted_offset -= weighted_width;
        segment_start = segment_end;
    }
    0.0
}

/// Maximum angular reach of each lateral outer-iris evidence arm. The prior
/// guide arm was `hypot(0.80, 0.36)` radii long; this converts a chord that is
/// exactly ten percent longer back into its central sweep angle.
pub fn outer_iris_lateral_sweep_angle() -> f64 {
    let old_arm_length = 0.80f64.hypot(0.36);
    2.0 * (old_arm_length * 1.10 * 0.5).asin()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FocusFilter {
    #[default]
    LumaHighPass,
    GreenHighPass,
    RedHighPass,
    BlueHighPass,
    RedGreenHighPass,
    LumaBandPass,
}

impl FocusFilter {
    pub const ALL: [Self; 6] = [
        Self::LumaHighPass,
        Self::GreenHighPass,
        Self::RedHighPass,
        Self::BlueHighPass,
        Self::RedGreenHighPass,
        Self::LumaBandPass,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::LumaHighPass => "LUMA HP",
            Self::GreenHighPass => "GREEN HP",
            Self::RedHighPass => "RED HP",
            Self::BlueHighPass => "BLUE HP",
            Self::RedGreenHighPass => "R-G HP",
            Self::LumaBandPass => "LUMA DOG",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BorderPoint {
    pub x: usize,
    pub y: usize,
    pub quality: f64,
}

/// Whether an eyelid margin was directly measured in the current RAW ROI.
/// `RoiClipped` is intentionally distinct from a failed walk: the projected
/// limbus itself proves that this side of the eyelid cannot be observed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EyelidObservationStatus {
    #[default]
    NotObserved,
    Observed,
    RoiClipped,
}

impl EyelidObservationStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::NotObserved => "NO-ROAD",
            Self::Observed => "OBSERVED",
            Self::RoiClipped => "ROI-CLIPPED",
        }
    }
}

/// Direct, frame-local eyelid geometry discovered around an already proven
/// Driving limbus. The two margins are the only primary surfaces. Fold and
/// lash samples are deliberately optional scene cues: an empty vector means
/// that the corresponding structure was not directly observable, and none of
/// these points is allowed to vote in the pupil or limbus fit.
#[derive(Clone, Debug, Default)]
pub struct EyelidNautilusScene {
    pub upper_margin: Vec<BorderPoint>,
    pub lower_margin: Vec<BorderPoint>,
    /// A coherent edge found on a side whose projected limbus itself leaves
    /// the ROI.  It is deliberately not promoted to an anatomical lid: it
    /// may be an occluding lid, the visible tail of the limbus, or both.  A
    /// conic ranker may use it only as soft censoring/contradiction evidence.
    pub upper_clipped_occluder: Vec<BorderPoint>,
    pub lower_clipped_occluder: Vec<BorderPoint>,
    pub upper_fold: Vec<BorderPoint>,
    pub lower_fold: Vec<BorderPoint>,
    pub upper_lashes: Vec<BorderPoint>,
    pub lower_lashes: Vec<BorderPoint>,
    pub upper_status: EyelidObservationStatus,
    pub lower_status: EyelidObservationStatus,
    pub upper_limbus_clearance_px: Option<f64>,
    pub lower_limbus_clearance_px: Option<f64>,
    pub elapsed_us: u64,
}

#[derive(Clone, Debug, Default)]
pub struct BorderFocus {
    pub score: f64,
    /// Median narrow-to-broad rise ratio measured on the accepted native-RAW
    /// limbus rays. Unlike `score`, this is deliberately contrast-normalized
    /// so optical sharpness can be conditioned separately from illumination
    /// and anatomical support. Zero means unavailable, not necessarily blur.
    pub optical_sharpness: f64,
    /// Median full-resolution RAW luma rise across the accepted limbus rays.
    /// Kept separate from `optical_sharpness` so a dim but crisp edge and a
    /// bright but soft edge do not become the same focus observation.
    pub border_contrast: f64,
    pub eye_basin_valid: bool,
    pub center: (f64, f64),
    pub focus_center: Option<(f64, f64)>,
    pub radius: f64,
    pub axis_ratio: f64,
    pub axis_angle: f64,
    pub acquisition_score: f64,
    pub pupil_hint: Option<(f64, f64)>,
    pub pupil_hint_radius: f64,
    pub pupil_hint_score: f64,
    /// A geometrically bounded limbus proposal whose missing perimeter is
    /// explained by the RAW ROI boundary. This is a censored observation, not
    /// a completed ellipse and not eye-identity evidence. It may seed an
    /// in-frame arc search and a temporally confirmed camera-side reframe.
    pub roi_truncated_limbus: Option<RoiTruncatedLimbusObservation>,
    pub points: Vec<BorderPoint>,
}

pub const ROI_TRUNCATED_LEFT: u8 = 1 << 0;
pub const ROI_TRUNCATED_RIGHT: u8 = 1 << 1;
pub const ROI_TRUNCATED_TOP: u8 = 1 << 2;
pub const ROI_TRUNCATED_BOTTOM: u8 = 1 << 3;

/// A directly supported but ROI-censored outer-limbus conic. Geometry outside
/// the crop is prediction only; `visible_arc_fraction` describes how much of
/// the fitted perimeter is actually in frame, while `supported_probe_fraction`
/// describes material ordering only on probes that were observable.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoiTruncatedLimbusObservation {
    pub center: (f64, f64),
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
    pub visible_arc_fraction: f64,
    pub supported_probe_fraction: f64,
    pub confidence: f64,
    pub censored_edges: u8,
    /// Desired movement of the camera-side ROI origin in sensor pixels. The
    /// viewer applies its own temporal agreement and per-command step bound.
    pub reframe_delta_px: (f64, f64),
}

impl RoiTruncatedLimbusObservation {
    pub fn ellipse_seed(self) -> (f64, f64, f64, f64, f64) {
        (
            self.center.0,
            self.center.1,
            self.major_radius,
            self.minor_radius,
            self.angle,
        )
    }

    pub fn laterally_censored(self) -> bool {
        self.censored_edges & (ROI_TRUNCATED_LEFT | ROI_TRUNCATED_RIGHT) != 0
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OuterIrisPoint {
    pub x: f64,
    pub y: f64,
    pub contrast: f64,
}

#[derive(Clone, Debug, Default)]
pub struct OuterIrisBoundary {
    pub center: (f64, f64),
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
    pub evidence_points: Vec<OuterIrisPoint>,
    /// Direct meridian hits rejected as eyelid/flat-tire occlusions. These are
    /// diagnostics only and never participate in the published ellipse fit.
    pub occluded_points: Vec<OuterIrisPoint>,
    pub veto_sweep_endpoints: Vec<OuterIrisPoint>,
    pub points: Vec<OuterIrisPoint>,
}

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
    /// Driving radius posterior exists. It may constrain a cold-start search
    /// but never counts as an independently observed temporal size sample.
    CurrentFrameGeometry,
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
            | FrontoParallelLimbusRadiusPriorSource::CurrentFrameGeometry => true,
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
const LIMBUS_LATEST_STRONG_CONTINUITY_HORIZON: Duration = Duration::from_millis(350);
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
        let prediction = prediction.filter(|prediction| prediction.valid());
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
                    | FrontoParallelLimbusRadiusPriorSource::CurrentFrameGeometry => true,
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
        true
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

#[derive(Clone, Debug)]
pub struct GrainIrisRefinement {
    pub boundary: OuterIrisBoundary,
    pub score: f64,
    pub support: usize,
    pub iterations: usize,
    pub used_native_seed: bool,
    pub used_pupil_bootstrap: bool,
}

#[derive(Clone, Copy, Debug)]
struct TrackedOuterIrisEllipse {
    sensor_center: (f64, f64),
    major_radius: f64,
    minor_radius: f64,
    angle: f64,
    confidence: f64,
    curve_confidence: [f64; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
}

/// Bounded-work diagnostics for one outer-limbus attempt.  These counters are
/// deliberately independent of the autofocus validity gate: a replay or live
/// overlay can distinguish a missing coarse seed from an empty ray lattice, a
/// sector-consistency rejection, and a completed affine fit.
#[derive(Clone, Copy, Debug, Default)]
pub struct OuterIrisDiagnostics {
    pub seed_usable: bool,
    pub accepted: bool,
    pub work_stride: usize,
    pub sample_stride: usize,
    pub elapsed_us: u64,
    pub ray_batch_elapsed_us: u64,
    pub max_ray_elapsed_us: u64,
    pub active_rays: usize,
    pub candidate_rays: usize,
    pub candidate_count: usize,
    pub ray_overruns: usize,
    pub ray_batch_timeouts: usize,
    pub refinement_elapsed_us: u64,
    pub refinement_iterations: usize,
    pub sector_overruns: usize,
    pub opposing_supported: usize,
    pub selected_right: usize,
    pub selected_left: usize,
    pub selected_lower: usize,
    /// Selected lateral contacts for which the full outward topology was
    /// observable: limbus, sustained sclera, then a second outer-eye ridge.
    pub outward_topology_observable: usize,
    /// Observable lateral contacts whose farther ridge was supported by a
    /// tangentially cohesive, illumination-corrected transition.
    pub outward_topology_supported: usize,
    pub outward_topology_observable_left: usize,
    pub outward_topology_supported_left: usize,
    pub outward_topology_observable_right: usize,
    pub outward_topology_supported_right: usize,
    pub outward_topology_mean_score_left: f64,
    pub outward_topology_mean_score_right: f64,
    pub outward_topology_mean_limbus_order_left: f64,
    pub outward_topology_mean_limbus_order_right: f64,
    pub outward_topology_mean_ridge_distance_left_px: f64,
    pub outward_topology_mean_ridge_distance_right_px: f64,
    pub outward_topology_longest_coherent_run_left: usize,
    pub outward_topology_longest_coherent_run_right: usize,
    pub flat_rejected: usize,
    pub occlusion_recovered: bool,
    /// Accepted meridian contacts with a completed continuous edge profile.
    pub analog_force_samples: usize,
    pub analog_force_outward: usize,
    pub analog_force_inward: usize,
    /// Mean signed measurement from the provisional conic to the local edge;
    /// positive is outward (toward sclera), negative is inward (toward iris).
    pub analog_mean_signed_offset_px: f64,
    /// Mean normalized edge amplitude after low-frequency light removal.
    pub analog_mean_power: f64,
    /// Mean confidence in polarity, scale agreement, and peak localization.
    pub analog_mean_certainty: f64,
    pub analog_refinement_elapsed_us: u64,
    pub analog_fit_applied: bool,
}

/// Per-eye history for the fast outer-limbus search. The stored center is in
/// full-sensor coordinates, so moving the live ROI does not move the prior.
#[derive(Clone, Debug, Default)]
pub struct OuterIrisTracker {
    ellipse: Option<TrackedOuterIrisEllipse>,
    // The lighting surface is stateful but geometry-free.  It can stabilize a
    // smooth slope/curvature between frames, while a material edge or a prior
    // ellipse can never be written into this history.
    material_illumination: MaterialIlluminationTracker,
    // Zero is the default spelling of full density. Sustained system-budget
    // pressure halves work on later frames; isolated scheduling spikes do not.
    // The reliable lateral/lower sector anchors are retained at every stride.
    work_stride: usize,
    // Sustained ray-level pressure halves radial samples independently from
    // the system-level ray-count stride above.
    sample_stride: usize,
    system_budget_pressure: u8,
    ray_budget_pressure: u8,
    system_budget_clean_frames: u16,
    ray_budget_clean_frames: u16,
    diagnostics: OuterIrisDiagnostics,
}

impl OuterIrisTracker {
    fn work_stride(&self) -> usize {
        self.work_stride.max(1).min(4)
    }

    fn sample_stride(&self) -> usize {
        self.sample_stride.max(1).min(4)
    }

    fn finish_attempt(&mut self, mut diagnostics: OuterIrisDiagnostics, elapsed: Duration) {
        diagnostics.elapsed_us = elapsed.as_micros().min(u64::MAX as u128) as u64;
        if elapsed > OUTER_IRIS_SYSTEM_BUDGET {
            self.system_budget_pressure = self.system_budget_pressure.saturating_add(1);
            self.system_budget_clean_frames = 0;
        } else {
            self.system_budget_pressure = 0;
            self.system_budget_clean_frames = self.system_budget_clean_frames.saturating_add(1);
        }
        if self.system_budget_pressure >= OUTER_IRIS_BUDGET_PRESSURE_FRAMES {
            self.work_stride = (self.work_stride() * 2).min(4);
            self.system_budget_pressure = 0;
            self.system_budget_clean_frames = 0;
        } else if self.system_budget_clean_frames >= OUTER_IRIS_BUDGET_RECOVERY_FRAMES {
            self.work_stride = (self.work_stride() / 2).max(1);
            self.system_budget_clean_frames = 0;
        }

        let ray_budget_overrun = diagnostics.ray_overruns > 0 || diagnostics.ray_batch_timeouts > 0;
        if ray_budget_overrun {
            self.ray_budget_pressure = self.ray_budget_pressure.saturating_add(1);
            self.ray_budget_clean_frames = 0;
        } else {
            self.ray_budget_pressure = 0;
            self.ray_budget_clean_frames = self.ray_budget_clean_frames.saturating_add(1);
        }
        if self.ray_budget_pressure >= OUTER_IRIS_BUDGET_PRESSURE_FRAMES {
            self.sample_stride = (self.sample_stride() * 2).min(4);
            self.ray_budget_pressure = 0;
            self.ray_budget_clean_frames = 0;
        } else if self.ray_budget_clean_frames >= OUTER_IRIS_BUDGET_RECOVERY_FRAMES {
            self.sample_stride = (self.sample_stride() / 2).max(1);
            self.ray_budget_clean_frames = 0;
        }
        self.diagnostics = diagnostics;
    }

    pub fn diagnostics(&self) -> OuterIrisDiagnostics {
        self.diagnostics
    }

    fn local_prior(
        &self,
        sensor_x: u32,
        sensor_y: u32,
        seed: [f64; 3],
    ) -> Option<TrackedOuterIrisEllipse> {
        let mut prior = self.ellipse?;
        if prior.confidence < OUTER_IRIS_TRACK_MIN_CONFIDENCE {
            return None;
        }
        prior.sensor_center.0 -= sensor_x as f64;
        prior.sensor_center.1 -= sensor_y as f64;
        let center_shift = (prior.sensor_center.0 - seed[0]).hypot(prior.sensor_center.1 - seed[1]);
        if center_shift > seed[2] * 0.55
            || prior.major_radius < seed[2] * 0.55
            || prior.major_radius > seed[2] * 1.55
            || !projected_circular_limbus_axes_plausible(prior.major_radius, prior.minor_radius)
        {
            return None;
        }
        Some(prior)
    }

    fn decay(&mut self) {
        if let Some(prior) = &mut self.ellipse {
            prior.confidence *= 0.55;
            for confidence in &mut prior.curve_confidence {
                *confidence *= 0.72;
            }
            if prior.confidence < 0.04 {
                self.ellipse = None;
            }
        }
    }

    fn observe(
        &mut self,
        sensor_x: u32,
        sensor_y: u32,
        ellipse: [f64; 5],
        curve_confidence: [f64; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
        evidence_count: usize,
    ) {
        if !projected_circular_limbus_axes_plausible(ellipse[2], ellipse[3]) {
            self.decay();
            return;
        }
        let grouped_count = curve_confidence
            .iter()
            .filter(|confidence| **confidence > 0.0)
            .count();
        let evidence_coverage = evidence_count as f64 / OUTER_IRIS_DENSE_EVIDENCE_SAMPLES as f64;
        let grouped_coverage = grouped_count as f64 / OUTER_IRIS_DENSE_EVIDENCE_SAMPLES as f64;
        let quality = (0.25 * evidence_coverage + 0.75 * grouped_coverage).clamp(0.0, 1.0);
        if quality < OUTER_IRIS_TRACK_MIN_CONFIDENCE {
            self.decay();
            return;
        }
        let measured = TrackedOuterIrisEllipse {
            sensor_center: (ellipse[0] + sensor_x as f64, ellipse[1] + sensor_y as f64),
            major_radius: ellipse[2],
            minor_radius: ellipse[3],
            angle: ellipse[4],
            confidence: quality,
            curve_confidence,
        };
        let Some(previous) = self.ellipse else {
            self.ellipse = Some(measured);
            return;
        };
        let center_jump = (previous.sensor_center.0 - measured.sensor_center.0)
            .hypot(previous.sensor_center.1 - measured.sensor_center.1);
        let radius_jump = (previous.major_radius - measured.major_radius).abs()
            / previous.major_radius.max(measured.major_radius).max(1.0);
        if center_jump > measured.major_radius * 0.38 || radius_jump > 0.28 {
            self.decay();
            if quality >= 0.72 {
                self.ellipse = Some(measured);
            }
            return;
        }
        let alpha = (0.20 + 0.35 * quality).clamp(0.20, 0.55);
        let blend = |old: f64, new: f64| old * (1.0 - alpha) + new * alpha;
        let mut blended_curve = [0.0; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
        for index in 0..blended_curve.len() {
            blended_curve[index] = (previous.curve_confidence[index] * 0.68
                + curve_confidence[index] * 0.32)
                .clamp(0.0, 1.0);
        }
        self.ellipse = Some(TrackedOuterIrisEllipse {
            sensor_center: (
                blend(previous.sensor_center.0, measured.sensor_center.0),
                blend(previous.sensor_center.1, measured.sensor_center.1),
            ),
            major_radius: blend(previous.major_radius, measured.major_radius),
            minor_radius: blend(previous.minor_radius, measured.minor_radius),
            angle: blend_outer_ellipse_angle(previous.angle, measured.angle, alpha),
            confidence: (previous.confidence * 0.72 + measured.confidence * 0.28).clamp(0.0, 1.0),
            curve_confidence: blended_curve,
        });
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InnerIrisPoint {
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

/// One untouched-RAW local maximum along a pupil-centered polar sector.
///
/// This is deliberately separate from [`InnerIrisPoint`].  The latter is the
/// single radius which survived the legacy per-ray selection/regularization
/// path.  A reflective or partly occluded pupil does not provide one reliable
/// winner on every ray, so the sparse temporal co-solver needs the small bank
/// of alternatives without inheriting the mass or temporal radius reward.
#[derive(Clone, Copy, Debug, Default)]
pub struct InnerIrisRadialCandidate {
    pub sector_index: u8,
    pub angle: f64,
    pub equivalent_radius_px: f64,
    pub x: f64,
    pub y: f64,
    /// Current-frame RAW evidence only.  No mass, size-history, or neighbor
    /// regularization term is included.
    pub raw_score: f64,
    pub peak_prominence: f64,
    pub luma_transition: f64,
    pub chroma_transition: f64,
    pub void_drop: f64,
    pub inside_void: f64,
    /// Exposure-normalized broad luma rise from well inside the proposed rim
    /// to just outside it. Positive values support pupil-to-iris ordering.
    pub broad_dark_step: f64,
}

#[derive(Clone, Debug, Default)]
pub struct InnerIrisBoundary {
    pub center: (f64, f64),
    pub radius: f64,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
    pub points: Vec<InnerIrisPoint>,
    /// Up to three prior-free local maxima per polar sector.  These remain
    /// sparse evidence: missing sectors are unknown, not failed pupil arcs.
    pub radial_candidates: Vec<InnerIrisRadialCandidate>,
}

/// Soft temporal guidance for the native 21-ray pupil-margin solver. These
/// radii live in projected area-equivalent space because individual image
/// rays do not all measure the fronto-parallel semi-major radius. The interval
/// is deliberately preferred rather than admissible: current strong RAW edge
/// evidence may leave it and must never be discarded solely by history.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InnerIrisRadiusPrior {
    pub estimated_equivalent_radius_px: f64,
    pub preferred_minimum_equivalent_radius_px: f64,
    pub preferred_maximum_equivalent_radius_px: f64,
    pub confidence: f64,
}

impl InnerIrisRadiusPrior {
    pub fn new(
        estimated_equivalent_radius_px: f64,
        preferred_minimum_equivalent_radius_px: f64,
        preferred_maximum_equivalent_radius_px: f64,
        confidence: f64,
    ) -> Option<Self> {
        if !estimated_equivalent_radius_px.is_finite()
            || !preferred_minimum_equivalent_radius_px.is_finite()
            || !preferred_maximum_equivalent_radius_px.is_finite()
            || !confidence.is_finite()
            || estimated_equivalent_radius_px <= 2.0
            || preferred_minimum_equivalent_radius_px <= 1.0
            || preferred_maximum_equivalent_radius_px
                <= preferred_minimum_equivalent_radius_px + 0.5
        {
            return None;
        }
        Some(Self {
            estimated_equivalent_radius_px,
            preferred_minimum_equivalent_radius_px,
            preferred_maximum_equivalent_radius_px,
            confidence: confidence.clamp(0.0, 1.0),
        })
    }
}

/// Hard current-frame search envelope for the native pupil-margin solver.
///
/// Unlike [`InnerIrisRadiusPrior`], this is an admissibility constraint. The
/// values are projected area-equivalent radii, so a fronto-parallel pupil-size
/// guide must be multiplied by the square root of the limbus projection ratio
/// before constructing it. This lets the operator/physiological bounds limit
/// which RAW edge is selected instead of merely discarding an out-of-range
/// winner after the search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InnerIrisRadiusEnvelope {
    pub minimum_equivalent_radius_px: f64,
    pub maximum_equivalent_radius_px: f64,
}

impl InnerIrisRadiusEnvelope {
    pub fn new(
        minimum_equivalent_radius_px: f64,
        maximum_equivalent_radius_px: f64,
    ) -> Option<Self> {
        if !minimum_equivalent_radius_px.is_finite()
            || !maximum_equivalent_radius_px.is_finite()
            || minimum_equivalent_radius_px <= 0.0
            || maximum_equivalent_radius_px <= minimum_equivalent_radius_px + 0.5
        {
            return None;
        }
        Some(Self {
            minimum_equivalent_radius_px,
            maximum_equivalent_radius_px,
        })
    }
}

/// Reliability of pixel-scale pupil detail for the current apparent size and
/// optical focus. One preserves the original sharp/resolved cue weights;
/// values toward zero shift ranking toward broad luma, dark-void mass, and a
/// soft temporal radius hint. This never changes the hard radius envelope.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InnerIrisEvidenceCondition {
    pub detail_reliability: f64,
}

impl Default for InnerIrisEvidenceCondition {
    fn default() -> Self {
        Self {
            detail_reliability: 1.0,
        }
    }
}

impl InnerIrisEvidenceCondition {
    pub fn new(detail_reliability: f64) -> Self {
        Self {
            detail_reliability: if detail_reliability.is_finite() {
                detail_reliability.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }

    fn cue_weights(self) -> (f64, f64, f64, f64, f64) {
        let detail = self.detail_reliability;
        let luma = 0.48 + 0.10 * detail;
        let chroma = 0.08 + 0.14 * detail;
        let void = 1.0 - luma - chroma;
        let mass_prior = 0.22 - 0.06 * detail;
        let temporal_prior = 0.14 - 0.04 * detail;
        (luma, chroma, void, mass_prior, temporal_prior)
    }
}

/// A bounded, inspectable pupil acquisition run on the carrier-neutral native
/// log plane. `trace` begins at the rough limbus center, records the winning
/// coarse dark basin, and then records each sub-cell steering update.
#[derive(Clone, Debug, Default)]
pub struct PupilDriveDiagnostics {
    pub start: (f64, f64),
    pub center: (f64, f64),
    pub trace: Vec<(f64, f64)>,
    pub acquisition_score: f64,
    pub enclosure_score: f64,
    pub travel_px: f64,
    /// Number of spatially distinct native-log dark lobes supporting this
    /// center. One is an ordinary basin; two or three identify the bounded
    /// shared-center proposals used when a corneal glint divides the pupil.
    /// This is derived from the pupil map itself, never from the specular map.
    pub consensus_members: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LimbusPerimeterDriveSample {
    pub phase: f64,
    pub base_point: (f64, f64),
    pub driven_point: (f64, f64),
    pub outward_normal: (f64, f64),
    pub offset_px: f64,
    pub transition_score: f64,
    pub outside_luma: f64,
    pub inside_luma: f64,
    /// Evidence encountered by looking from this candidate toward the
    /// acquired pupil. A real limbus enters cohesive iris/pupil material;
    /// an eyelid fork usually enters another patch of sclera first.
    pub pupil_heading_score: f64,
    /// Evidence that the same sightline continues through the pupil/iris and
    /// exits into sclera again on the opposite side.  This is the missing
    /// "head of the road" cue at a lid/limbus fork: both branches can point at
    /// a dark pupil eventually, but only the limbus normally gives the full
    /// `sclera | iris/pupil | sclera` chord.
    pub opposite_sclera_score: f64,
    /// First-lap departure from the robust low-order normal-offset model of
    /// one projected affine ellipse, expressed in nominal millimetres.  An
    /// eyelid branch can be smooth and can enclose the pupil, but it cannot
    /// normally remain on the same projected circular limbus as both lateral
    /// anchors.
    pub affine_ellipse_residual_mm: f64,
    /// Residual of the repaired row against that same first-lap affine model.
    /// Comparing this with `affine_ellipse_residual_mm` measures the repair
    /// without letting a refitted model move toward the lid afterward.
    pub affine_ellipse_final_residual_mm: f64,
    /// True when this perimeter row was reconsidered during a partial second
    /// lap after the initial route looked inconsistent with the pupil heading.
    pub revisited: bool,
    /// True when the limbus is still unconfirmed after an eyelid bridge and
    /// this point remains only the projected ellipse between two directly
    /// measured sclera/iris rejoins. A projected point that subsequently lands
    /// on sustained direct image evidence is promoted back to measured here;
    /// the strip-level `conic_projected_rows` retains its selection history.
    pub inferred_occlusion: bool,
}

/// An ellipse-normal strip followed once around a limbus. Rows advance around
/// the perimeter and columns run from sclera (left/outward) to iris and pupil
/// (right/inward). The metric scale is explicitly nominal until the camera is
/// calibrated: a 12 mm physical limbus diameter is assumed.
#[derive(Clone, Debug, Default)]
pub struct LimbusPerimeterStrip {
    pub width: usize,
    pub height: usize,
    pub luma: Vec<u16>,
    pub nominal_mm_per_side: f64,
    pub pixels_per_mm: f64,
    pub guide_left_column: usize,
    pub guide_right_column: usize,
    pub affine_ellipse_residual_threshold_mm: Option<f64>,
    pub affine_ellipse_first_rms_mm: Option<f64>,
    pub affine_ellipse_final_rms_mm: Option<f64>,
    pub affine_ellipse_departure_first_rms_mm: Option<f64>,
    pub affine_ellipse_departure_final_rms_mm: Option<f64>,
    /// Third-lap geometrical closure error against the conic fitted with the
    /// projected/occluded rows assigned exactly zero weight.
    pub conic_closure_before_rms_mm: Option<f64>,
    pub conic_closure_after_rms_mm: Option<f64>,
    pub affine_reinforced: bool,
    pub lap_count: usize,
    /// Rows explicitly reconsidered by the bounded pupil-headed second lap,
    /// before the measured-only conic-closure lap marks projected occlusions.
    pub second_lap_revisited_rows: Vec<bool>,
    pub revisited_rows: Vec<bool>,
    /// Rows added when a suspect arc reaches the arbitrary last/first sample
    /// seam without encountering independently proven full-chord anatomy.
    pub cyclic_reentry_rows: Vec<bool>,
    /// Rows for which at least three neighboring native-resolution orbital
    /// fly-bys agree on a materially different complete limbus chord. These
    /// rows may steer the bounded second lap but do not bypass its continuity
    /// or projected-circle constraints.
    pub full_chord_flyby_rows: Vec<bool>,
    /// Rows that the bilateral lower-lid topology initially carried through
    /// the measured-only conic closure.  A row can subsequently be promoted
    /// back to measured when the projected conic lands on a sustained,
    /// directly ordered sclera-to-iris transition. These rows remain excluded
    /// from hypothesis ranking even after confirmation, because the conic
    /// selected their location before the pixels confirmed it.
    pub conic_projected_rows: Vec<bool>,
    /// Initially projected rows whose final conic location is independently
    /// confirmed by a coherent run of native-resolution image evidence.
    pub reconfirmed_rows: Vec<bool>,
    /// Rows carried across an eyelid by the ellipse inferred from visible
    /// limbus arcs and still unconfirmed after the closure lap. They are
    /// deliberately excluded from evidence scores.
    pub inferred_rows: Vec<bool>,
    /// `Some(true)` means a doubtful lower arc was bracketed by clean,
    /// pupil-headed sclera rejoins on both sides. `Some(false)` means the
    /// topology failed and no completed eye road may be published. `None`
    /// means no lower occlusion was detected.
    pub lower_occlusion_rejoined: Option<bool>,
    /// Directly measured rows immediately before and after the projected
    /// lower-lid bridge, in drive order.
    pub lower_occlusion_anchor_rows: Option<(usize, usize)>,
    pub first_lap_boundary_columns: Vec<f64>,
    /// Pupil-headed bounded-repair path before the measured-only conic
    /// closure lap. This makes the third circuit inspectable rather than
    /// presenting its final projected bridge as though lap two found it.
    pub second_lap_boundary_columns: Vec<f64>,
    pub boundary_columns: Vec<f64>,
    pub samples: Vec<LimbusPerimeterDriveSample>,
}

/// Ordered evidence on one of the two usually visible lateral limbus arcs.
/// A true iris boundary must provide both a meaningful outward-to-inward
/// sclera/iris transition and a sightline that crosses iris/pupil before
/// returning to sclera on the far side.  An eyelid or the outer eye aperture
/// can provide either cue by itself, but is much less likely to provide both
/// on both lateral sides.
#[derive(Clone, Copy, Debug, Default)]
pub struct LimbusLateralOrderEvidence {
    pub sample_count: usize,
    pub transition_fraction: f64,
    pub opposite_sclera_fraction: f64,
    pub ordered_score: f64,
}

pub fn limbus_lateral_order_evidence(
    strip: &LimbusPerimeterStrip,
    positive_camera_x_side: bool,
) -> LimbusLateralOrderEvidence {
    let mut sample_count = 0usize;
    let mut transitions = 0usize;
    let mut opposite_sclera = 0usize;
    for (_, sample) in strip.samples.iter().enumerate().filter(|(row, sample)| {
        let model_selected = strip
            .conic_projected_rows
            .get(*row)
            .copied()
            .unwrap_or(sample.inferred_occlusion);
        if model_selected {
            false
        } else if positive_camera_x_side {
            sample.outward_normal.0 >= 0.55
        } else {
            sample.outward_normal.0 <= -0.55
        }
    }) {
        sample_count += 1;
        // These thresholds correspond to 7.2 and 9 RAW luma codes after the
        // same multi-distance averaging used by the drive.  Requiring a
        // magnitude, rather than merely a positive sign, is essential: the
        // rejected pair-01 aperture road had 79% positive rows whose mean
        // transition was nevertheless only 0.029.
        transitions += usize::from(sample.transition_score > 0.04);
        opposite_sclera += usize::from(sample.opposite_sclera_score > 0.05);
    }
    if sample_count == 0 {
        return LimbusLateralOrderEvidence::default();
    }
    let transition_fraction = transitions as f64 / sample_count as f64;
    let opposite_sclera_fraction = opposite_sclera as f64 / sample_count as f64;
    LimbusLateralOrderEvidence {
        sample_count,
        transition_fraction,
        opposite_sclera_fraction,
        // Geometric combination prevents either a clean unrelated edge or a
        // fortuitous far bright patch from compensating for the missing cue.
        ordered_score: (transition_fraction * opposite_sclera_fraction).sqrt(),
    }
}

pub fn limbus_bilateral_order_score(strip: &LimbusPerimeterStrip) -> f64 {
    limbus_lateral_order_evidence(strip, false)
        .ordered_score
        .min(limbus_lateral_order_evidence(strip, true).ordered_score)
}

/// Select one radially steered sample per perimeter row only after evaluating
/// the complete ring. This is a cyclic Viterbi pass: a locally strong eyelid
/// edge cannot commit the drive at a fork unless its future continuation and
/// return to the starting sclera/iris boundary also pay for the departure.
fn closed_limbus_lookahead_path(
    local_scores: &[f64],
    row_count: usize,
    offsets: &[f64],
    pixels_per_mm: f64,
) -> Option<Vec<usize>> {
    let state_count = offsets.len();
    if row_count < 3
        || state_count == 0
        || local_scores.len() != row_count.saturating_mul(state_count)
        || !pixels_per_mm.is_finite()
        || pixels_per_mm <= 0.0
    {
        return None;
    }
    const MAX_STATE_JUMP: usize = 3;
    const CONTINUITY_WEIGHT: f64 = 0.85;
    let mut best_total = f64::NEG_INFINITY;
    let mut best_path = None;
    for start in 0..state_count {
        let mut previous = vec![f64::NEG_INFINITY; state_count];
        previous[start] = local_scores[start];
        let mut back = vec![0usize; row_count * state_count];
        for row in 1..row_count {
            let mut current = vec![f64::NEG_INFINITY; state_count];
            for state in 0..state_count {
                let first = state.saturating_sub(MAX_STATE_JUMP);
                let last = (state + MAX_STATE_JUMP).min(state_count - 1);
                let mut winning_prior = first;
                let mut winning_score = f64::NEG_INFINITY;
                for prior in first..=last {
                    let normalized_step = (offsets[state] - offsets[prior]) / pixels_per_mm;
                    let candidate = previous[prior] - CONTINUITY_WEIGHT * normalized_step.powi(2);
                    if candidate > winning_score {
                        winning_score = candidate;
                        winning_prior = prior;
                    }
                }
                current[state] = local_scores[row * state_count + state] + winning_score;
                back[row * state_count + state] = winning_prior;
            }
            previous = current;
        }
        for last in 0..state_count {
            if last.abs_diff(start) > MAX_STATE_JUMP {
                continue;
            }
            let normalized_closure = (offsets[last] - offsets[start]) / pixels_per_mm;
            let total = previous[last] - CONTINUITY_WEIGHT * normalized_closure.powi(2);
            if total <= best_total {
                continue;
            }
            let mut path = vec![0usize; row_count];
            path[row_count - 1] = last;
            for row in (1..row_count).rev() {
                path[row - 1] = back[row * state_count + path[row]];
            }
            best_total = total;
            best_path = Some(path);
        }
    }
    best_path
}

/// Make an independent closed-loop challenger after a pupil has been
/// acquired.  The ordinary lap is intentionally edge-only, but that also
/// means a smooth lower lid can look self-consistent all the way through a
/// fork.  This challenger scores every future state for whether travelling
/// inward from it immediately enters cohesive iris on the way to the pupil.
/// Its disagreements identify suspect arcs; it does not directly replace the
/// first path, so the eventual correction can remain a bounded partial lap.
fn closed_limbus_pupil_lookahead_path(
    local_scores: &[f64],
    pupil_heading_scores: &[f64],
    opposite_sclera_scores: &[f64],
    row_count: usize,
    offsets: &[f64],
    pixels_per_mm: f64,
) -> Option<Vec<usize>> {
    if local_scores.len() != pupil_heading_scores.len()
        || local_scores.len() != opposite_sclera_scores.len()
    {
        return None;
    }
    const PUPIL_LOOKAHEAD_WEIGHT: f64 = 0.92;
    const OPPOSITE_SCLERA_CHORD_WEIGHT: f64 = 0.78;
    let guided_scores = local_scores
        .iter()
        .zip(pupil_heading_scores)
        .zip(opposite_sclera_scores)
        .map(|((local, heading), opposite_sclera)| {
            local
                + PUPIL_LOOKAHEAD_WEIGHT * heading
                + OPPOSITE_SCLERA_CHORD_WEIGHT * opposite_sclera
        })
        .collect::<Vec<_>>();
    closed_limbus_lookahead_path(&guided_scores, row_count, offsets, pixels_per_mm)
}

/// Extend each end of a bounded suspect arc until the drive reaches a row
/// whose full pupilward chord independently proves limbus anatomy.  The mask
/// is circular: an arc ending near the last sample continues through row zero
/// rather than treating the storage seam as a valid rejoin.  Only the mask
/// present on entry supplies arc endpoints, so the bounded search cannot
/// recursively flood around an eye with no trustworthy re-entry.
fn extend_cyclic_revisit_to_direct_reentry(
    revisited_rows: &mut [bool],
    direct_reentry_rows: &[bool],
    maximum_extension: usize,
) -> Vec<bool> {
    let row_count = revisited_rows.len();
    if row_count < 3 || direct_reentry_rows.len() != row_count || maximum_extension == 0 {
        return vec![false; row_count];
    }
    let original = revisited_rows.to_vec();
    let mut extended_rows = vec![false; row_count];
    for row in 0..row_count {
        if !original[row] {
            continue;
        }
        let forward_neighbor = (row + 1) % row_count;
        if !original[forward_neighbor] {
            for step in 1..=maximum_extension.min(row_count - 1) {
                let candidate = (row + step) % row_count;
                if direct_reentry_rows[candidate] {
                    break;
                }
                revisited_rows[candidate] = true;
                extended_rows[candidate] = true;
            }
        }
        let backward_neighbor = (row + row_count - 1) % row_count;
        if !original[backward_neighbor] {
            for step in 1..=maximum_extension.min(row_count - 1) {
                let candidate = (row + row_count - step) % row_count;
                if direct_reentry_rows[candidate] {
                    break;
                }
                revisited_rows[candidate] = true;
                extended_rows[candidate] = true;
            }
        }
    }
    extended_rows
}

#[derive(Clone, Debug)]
struct LimbusAffineOffsetModel {
    predicted_mm: Vec<f64>,
    residual_mm: Vec<f64>,
    outlier_threshold_mm: f64,
}

/// Fit the normal displacement of a nearby projected ellipse.  Translation,
/// scale, eccentricity, and orientation live in the constant/first/second
/// angular harmonics; higher-frequency departures are therefore evidence that
/// one arc left the common limbus and followed a lid.  Robust reweighting keeps
/// that suspect arc from defining the model it is being tested against.
fn fit_limbus_affine_offset_model(
    offsets_mm: &[f64],
    base_weights: &[f64],
) -> Option<LimbusAffineOffsetModel> {
    if offsets_mm.len() < 12 || offsets_mm.len() != base_weights.len() {
        return None;
    }
    let active = |weight: f64| weight.is_finite() && weight > 0.0;
    if base_weights
        .iter()
        .filter(|weight| active(**weight))
        .count()
        < 8
    {
        return None;
    }
    let count = offsets_mm.len();
    let basis = (0..count)
        .map(|row| {
            let phase = 2.0 * PI * row as f64 / count as f64;
            [
                1.0,
                phase.cos(),
                phase.sin(),
                (2.0 * phase).cos(),
                (2.0 * phase).sin(),
            ]
        })
        .collect::<Vec<_>>();
    let mut weights = base_weights
        .iter()
        .map(|weight| {
            if active(*weight) {
                weight.clamp(0.04, 1.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mut coefficients = [0.0f64; 5];
    for _ in 0..5 {
        let mut matrix = [[0.0f64; 6]; 5];
        for row in 0..count {
            let weight = weights[row];
            for output in 0..5 {
                for input in 0..5 {
                    matrix[output][input] += weight * basis[row][output] * basis[row][input];
                }
                matrix[output][5] += weight * basis[row][output] * offsets_mm[row];
            }
        }
        for diagonal in 0..5 {
            matrix[diagonal][diagonal] += 1.0e-5;
        }
        coefficients = solve_outer_ellipse_normal_equations(matrix)?;
        let centered = (0..count)
            .map(|row| {
                offsets_mm[row]
                    - coefficients
                        .iter()
                        .zip(basis[row])
                        .map(|(coefficient, term)| coefficient * term)
                        .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let mut active_centered = centered
            .iter()
            .zip(base_weights)
            .filter_map(|(residual, weight)| active(*weight).then_some(*residual))
            .collect::<Vec<_>>();
        let center = percentile_f64(&mut active_centered, 0.50);
        let mut deviations = centered
            .iter()
            .zip(base_weights)
            .filter_map(|(residual, weight)| active(*weight).then_some((*residual - center).abs()))
            .collect::<Vec<_>>();
        let robust_scale = (1.4826 * percentile_f64(&mut deviations, 0.50)).max(0.025);
        for row in 0..count {
            if !active(base_weights[row]) {
                weights[row] = 0.0;
                continue;
            }
            let residual = (centered[row] - center).abs();
            let huber = (1.5 * robust_scale / residual.max(1.0e-9)).min(1.0);
            weights[row] = base_weights[row].clamp(0.04, 1.0) * huber;
        }
    }
    let predicted_mm = basis
        .iter()
        .map(|terms| {
            coefficients
                .iter()
                .zip(terms)
                .map(|(coefficient, term)| coefficient * term)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let residual_mm = offsets_mm
        .iter()
        .zip(&predicted_mm)
        .map(|(observed, predicted)| (observed - predicted).abs())
        .collect::<Vec<_>>();
    let mut absolute = residual_mm
        .iter()
        .zip(base_weights)
        .filter_map(|(residual, weight)| active(*weight).then_some(*residual))
        .collect::<Vec<_>>();
    let median_absolute = percentile_f64(&mut absolute, 0.50);
    let mut absolute_deviation = residual_mm
        .iter()
        .zip(base_weights)
        .filter_map(|(residual, weight)| {
            active(*weight).then_some((residual - median_absolute).abs())
        })
        .collect::<Vec<_>>();
    let outlier_threshold_mm = (median_absolute
        + 2.5 * 1.4826 * percentile_f64(&mut absolute_deviation, 0.50))
    .clamp(0.18, 0.65);
    Some(LimbusAffineOffsetModel {
        predicted_mm,
        residual_mm,
        outlier_threshold_mm,
    })
}

#[derive(Clone, Debug)]
struct LimbusLowerOcclusionBridge {
    inferred_rows: Vec<bool>,
    rejoined: Option<bool>,
    anchor_rows: Option<(usize, usize)>,
}

fn cyclic_forward_row_distance(from: usize, to: usize, count: usize) -> usize {
    (to + count - from) % count
}

/// Find the lower-lid interval by its topology, not by the strength of the
/// lid edge.  A directly measured limbus row must enter iris on its inward
/// side and, looking through the pupil, exit into sclera again.  Once a
/// coherent lower run fails that test, search both ways along the projected
/// ellipse for multi-row, opposing lateral rejoins.  Only the interval
/// bracketed by those rejoins may be carried as inferred geometry.
fn lower_limbus_occlusion_bridge(
    search: OuterSearchEllipse,
    path: &[usize],
    offsets: &[f64],
    measurements: &[(f64, f64)],
    pupil_heading_scores: &[f64],
    opposite_sclera_scores: &[f64],
    heading_reference: f64,
) -> LimbusLowerOcclusionBridge {
    let row_count = path.len();
    let state_count = offsets.len();
    let empty = || LimbusLowerOcclusionBridge {
        inferred_rows: vec![false; row_count],
        rejoined: None,
        anchor_rows: None,
    };
    if row_count < 24
        || state_count == 0
        || measurements.len() != row_count * state_count
        || pupil_heading_scores.len() != measurements.len()
        || opposite_sclera_scores.len() != measurements.len()
    {
        return empty();
    }
    let geometry = (0..row_count)
        .map(|row| {
            let phase = 2.0 * PI * row as f64 / row_count as f64;
            let (base, normal) = search.point_and_normal(phase, 1.0);
            let offset = offsets[path[row]];
            (
                (base.0 + normal.0 * offset, base.1 + normal.1 * offset),
                normal,
            )
        })
        .collect::<Vec<_>>();
    let selected = |row: usize| {
        let state = path[row];
        let flat = row * state_count + state;
        let (outside, inside) = measurements[flat];
        (
            (outside - inside) / 180.0,
            outside,
            inside,
            pupil_heading_scores[flat],
            opposite_sclera_scores[flat],
        )
    };
    let heading_floor = (heading_reference - 0.11).max(0.11);
    let lower_zone = geometry
        .iter()
        .map(|(point, _)| point.1 >= search.center.1 - search.minor_radius * 0.10)
        .collect::<Vec<_>>();
    let bad = (0..row_count)
        .map(|row| {
            if !lower_zone[row] {
                return false;
            }
            let (transition, outside, inside, heading, opposite) = selected(row);
            outside <= inside + 8.0
                || transition < 0.015
                || heading < heading_floor
                || opposite < 0.045
        })
        .collect::<Vec<_>>();
    // Isolated weak samples are normal RAW noise.  An eyelid fork is a
    // coherent road: require at least four failures in a seven-row window.
    let bad_core = (0..row_count).any(|center| {
        lower_zone[center]
            && (0..7)
                .filter(|offset| bad[(center + row_count + *offset - 3) % row_count])
                .count()
                >= 4
    });
    if !bad_core {
        return empty();
    }

    // The through-pupil ray is deliberately a weak cue: a glint or the
    // narrow nasal sclera can erase it for several adjacent RAW samples even
    // when the local limbus transition is excellent. Use a seven-row rejoin
    // window and require two independently positive far exits at the same
    // threshold as the bilateral-order gate. The wider window still demands
    // a four-row majority for direct material order and pupilward heading, so
    // a single bright flesh patch cannot become an anchor.
    const HALF_WINDOW: usize = 3;
    let anchor = |side: f64| {
        (0..row_count)
            .filter_map(|center| {
                let (point, normal) = geometry[center];
                if normal.0 * side < 0.42 || point.1 < search.center.1 - search.minor_radius * 0.24
                {
                    return None;
                }
                let rows = (0..=HALF_WINDOW * 2)
                    .map(|offset| (center + row_count + offset - HALF_WINDOW) % row_count)
                    .collect::<Vec<_>>();
                let ordered = rows
                    .iter()
                    .filter(|row| {
                        let (transition, outside, inside, _, _) = selected(**row);
                        transition >= 0.040 && outside > inside + 8.0
                    })
                    .count();
                let pupilward = rows
                    .iter()
                    .filter(|row| selected(**row).3 >= heading_floor)
                    .count();
                let far_rejoins = rows.iter().filter(|row| selected(**row).4 >= 0.05).count();
                let material_majority = ordered >= 4 && pupilward >= 4 && far_rejoins >= 2;
                // Near the nasal canthus the directly visible sclera band can
                // be only two or three RAW samples wide.  Do not force that
                // real rejoin to masquerade as a four-row transition: accept
                // the narrow material chord only when essentially the whole
                // window still points pupilward and at least three separate
                // through-eye rays exit into sclera.  A lid edge supplies a
                // clean local transition, but not that joint directional and
                // opposite-side evidence.
                let narrow_nasal_rejoin = ordered >= 2 && pupilward >= 6 && far_rejoins >= 3;
                if !material_majority && !narrow_nasal_rejoin {
                    return None;
                }
                let quality = rows
                    .iter()
                    .map(|row| {
                        let (transition, _, _, heading, opposite) = selected(*row);
                        transition.max(0.0) + 0.55 * heading.max(0.0) + 0.75 * opposite.max(0.0)
                    })
                    .sum::<f64>();
                Some((quality, center))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, row)| row)
    };
    let right_anchor = anchor(1.0);
    let left_anchor = anchor(-1.0);
    let Some((right_anchor, left_anchor)) = right_anchor.zip(left_anchor) else {
        #[cfg(test)]
        if std::env::var_os("BUTTERCUP_DRIVING_TRACE").is_some() {
            let best_counts = |side: f64| {
                (0..row_count)
                    .filter_map(|center| {
                        let (point, normal) = geometry[center];
                        if normal.0 * side < 0.42
                            || point.1 < search.center.1 - search.minor_radius * 0.24
                        {
                            return None;
                        }
                        let rows = (0..=HALF_WINDOW * 2)
                            .map(|offset| (center + row_count + offset - HALF_WINDOW) % row_count)
                            .collect::<Vec<_>>();
                        let ordered = rows
                            .iter()
                            .filter(|row| {
                                let (transition, outside, inside, _, _) = selected(**row);
                                transition >= 0.040 && outside > inside + 8.0
                            })
                            .count();
                        let pupilward = rows
                            .iter()
                            .filter(|row| selected(**row).3 >= heading_floor)
                            .count();
                        let far_rejoins =
                            rows.iter().filter(|row| selected(**row).4 >= 0.05).count();
                        Some((
                            ordered + pupilward + far_rejoins,
                            center,
                            ordered,
                            pupilward,
                            far_rejoins,
                        ))
                    })
                    .max_by_key(|counts| counts.0)
            };
            eprintln!(
                "LOWER_REJOIN_MISS heading-floor={heading_floor:.3} right={:?} left={:?}",
                best_counts(1.0),
                best_counts(-1.0),
            );
        }
        return LimbusLowerOcclusionBridge {
            inferred_rows: vec![false; row_count],
            rejoined: Some(false),
            anchor_rows: None,
        };
    };
    let bottom_row = geometry
        .iter()
        .enumerate()
        .max_by(|left, right| left.1 .0 .1.total_cmp(&right.1 .0 .1))
        .map(|(row, _)| row)
        .unwrap_or(0);
    let right_to_left = cyclic_forward_row_distance(right_anchor, left_anchor, row_count);
    let right_to_bottom = cyclic_forward_row_distance(right_anchor, bottom_row, row_count);
    let (start, end, span) = if right_to_bottom <= right_to_left {
        (right_anchor, left_anchor, right_to_left)
    } else {
        let span = cyclic_forward_row_distance(left_anchor, right_anchor, row_count);
        (left_anchor, right_anchor, span)
    };
    if span < row_count / 8 || span > row_count * 5 / 8 {
        return LimbusLowerOcclusionBridge {
            inferred_rows: vec![false; row_count],
            rejoined: Some(false),
            anchor_rows: Some((start, end)),
        };
    }
    let interior = (HALF_WINDOW + 1)..span.saturating_sub(HALF_WINDOW);
    let bad_count = interior
        .clone()
        .filter(|step| bad[(start + *step) % row_count])
        .count();
    if bad_count < 3 {
        return empty();
    }

    let mut inferred_rows = vec![false; row_count];
    for step in interior {
        inferred_rows[(start + step) % row_count] = true;
    }
    LimbusLowerOcclusionBridge {
        inferred_rows,
        rejoined: Some(true),
        anchor_rows: Some((start, end)),
    }
}

/// Re-confirm projected bridge rows only after the conic-closure lap has put
/// them on the common ellipse.  Testing the pre-closure road here is unsafe:
/// a long, clean lid edge can look exactly like bright-outside/dark-inside
/// limbus evidence.  At the projected location, a sustained native-resolution
/// run with the same material order and pupilward heading is independent image
/// confirmation, so it should be measured/cyan rather than inferred/magenta.
fn conic_projected_limbus_reconfirmations(
    conic_projected_rows: &[bool],
    path: &[usize],
    offsets: &[f64],
    measurements: &[(f64, f64)],
    pupil_heading_scores: &[f64],
    heading_reference: f64,
) -> Vec<bool> {
    let row_count = path.len();
    let state_count = offsets.len();
    if row_count < 9
        || conic_projected_rows.len() != row_count
        || state_count == 0
        || measurements.len() != row_count * state_count
        || pupil_heading_scores.len() != measurements.len()
    {
        return vec![false; row_count];
    }

    const HALF_WINDOW: usize = 4;
    let heading_floor = (heading_reference - 0.21).max(0.10);
    let directly_ordered = (0..row_count)
        .map(|row| {
            let flat = row * state_count + path[row];
            let (outside, inside) = measurements[flat];
            let transition = (outside - inside) / 180.0;
            transition >= 0.18
                && outside > inside + 32.0
                && pupil_heading_scores[flat] >= heading_floor
        })
        .collect::<Vec<_>>();

    (0..row_count)
        .map(|center| {
            conic_projected_rows[center]
                && (0..=HALF_WINDOW * 2)
                    .filter(|offset| {
                        directly_ordered[(center + row_count + *offset - HALF_WINDOW) % row_count]
                    })
                    .count()
                    >= 6
        })
        .collect()
}

pub const IRIS_LIGHT_RADIAL_BANDS: usize = 3;
pub const IRIS_LIGHT_ANGULAR_SECTORS: usize = 8;
pub const IRIS_LIGHT_MAP_CELLS: usize = IRIS_LIGHT_RADIAL_BANDS * IRIS_LIGHT_ANGULAR_SECTORS;

#[derive(Clone, Copy, Debug, Default)]
pub struct IrisLightMap {
    pub valid: bool,
    pub mean: f64,
    pub span: f64,
    pub gradient_x: f64,
    pub gradient_y: f64,
    pub cells: [f64; IRIS_LIGHT_MAP_CELLS],
}

#[derive(Clone, Copy)]
struct Candidate {
    angle: f64,
    radius: f64,
    x: usize,
    y: usize,
    contrast: f64,
    sharpness: f64,
    quality: f64,
}

fn sample(raw: &[u16], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let x = x.clamp(1.0, width.saturating_sub(2) as f64);
    let y = y.clamp(1.0, height.saturating_sub(2) as f64);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let p00 = raw[y0 * width + x0] as f64;
    let p10 = raw[y0 * width + x0 + 1] as f64;
    let p01 = raw[(y0 + 1) * width + x0] as f64;
    let p11 = raw[(y0 + 1) * width + x0 + 1] as f64;
    (p00 * (1.0 - fx) + p10 * fx) * (1.0 - fy) + (p01 * (1.0 - fx) + p11 * fx) * fy
}

// Average one complete Bayer cell so CFA contrast cannot masquerade as focus.
fn cfa_luma(raw: &[u16], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let x = (x.floor() as usize & !1).clamp(0, width.saturating_sub(2));
    let y = (y.floor() as usize & !1).clamp(0, height.saturating_sub(2));
    let i = y * width + x;
    (raw[i] as f64 + raw[i + 1] as f64 + raw[i + width] as f64 + raw[i + width + 1] as f64) * 0.25
}

struct BoxLuma5 {
    width: usize,
    height: usize,
    stride: usize,
    integral: Vec<u64>,
}

impl BoxLuma5 {
    fn new(raw: &[u16], width: usize, height: usize) -> Self {
        let stride = width + 1;
        let mut integral = vec![0u64; stride * (height + 1)];
        for y in 0..height {
            let mut row_sum = 0u64;
            for x in 0..width {
                row_sum += raw[y * width + x] as u64;
                integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row_sum;
            }
        }
        Self {
            width,
            height,
            stride,
            integral,
        }
    }

    fn integer_sample(&self, x: usize, y: usize) -> f64 {
        let x = x.clamp(2, self.width.saturating_sub(3));
        let y = y.clamp(2, self.height.saturating_sub(3));
        let x0 = x - 2;
        let y0 = y - 2;
        let x1 = x + 3;
        let y1 = y + 3;
        let sum = self.integral[y1 * self.stride + x1] + self.integral[y0 * self.stride + x0]
            - self.integral[y0 * self.stride + x1]
            - self.integral[y1 * self.stride + x0];
        sum as f64 / 25.0
    }

    fn sample(&self, x: f64, y: f64) -> f64 {
        let x = x.clamp(2.0, self.width.saturating_sub(3) as f64);
        let y = y.clamp(2.0, self.height.saturating_sub(3) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let top = self.integer_sample(x0, y0) * (1.0 - fx) + self.integer_sample(x0 + 1, y0) * fx;
        let bottom =
            self.integer_sample(x0, y0 + 1) * (1.0 - fx) + self.integer_sample(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }

    fn integer_sample3(&self, x: usize, y: usize) -> f64 {
        let x = x.clamp(1, self.width.saturating_sub(2));
        let y = y.clamp(1, self.height.saturating_sub(2));
        let x0 = x - 1;
        let y0 = y - 1;
        let x1 = x + 2;
        let y1 = y + 2;
        let sum = self.integral[y1 * self.stride + x1] + self.integral[y0 * self.stride + x0]
            - self.integral[y0 * self.stride + x1]
            - self.integral[y1 * self.stride + x0];
        sum as f64 / 9.0
    }

    fn sample3(&self, x: f64, y: f64) -> f64 {
        let x = x.clamp(1.0, self.width.saturating_sub(2) as f64);
        let y = y.clamp(1.0, self.height.saturating_sub(2) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let top = self.integer_sample3(x0, y0) * (1.0 - fx) + self.integer_sample3(x0 + 1, y0) * fx;
        let bottom = self.integer_sample3(x0, y0 + 1) * (1.0 - fx)
            + self.integer_sample3(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

fn outer_circle_cost(points: &[OuterIrisPoint], circle: [f64; 3]) -> f64 {
    let [center_x, center_y, radius] = circle;
    let mut residuals = points
        .iter()
        .map(|point| {
            let dx = point.x - center_x;
            let dy = point.y - center_y;
            ((dx.hypot(dy) - radius).abs(), point.contrast)
        })
        .collect::<Vec<_>>();
    residuals.sort_by(|left, right| left.0.total_cmp(&right.0));
    let scale = residuals[residuals.len() / 2].0 + 1.0;
    let mut weighted_error = 0.0;
    let mut weight = 0.0;
    for (residual, contrast) in residuals {
        let point_weight = contrast.min(80.0).sqrt();
        weighted_error += residual.min(2.5 * scale).powi(2) * point_weight;
        weight += point_weight;
    }
    weighted_error / weight.max(f64::EPSILON)
}

fn fit_outer_circle(points: &[OuterIrisPoint], seed: [f64; 3]) -> [f64; 3] {
    let mut circle = seed;
    let mut steps = [4.0, 4.0, 5.0];
    for _ in 0..24 {
        for parameter in 0..circle.len() {
            for direction in [-1.0, 1.0] {
                let mut candidate = circle;
                candidate[parameter] += direction * steps[parameter];
                let center_shift = (candidate[0] - seed[0]).hypot(candidate[1] - seed[1]);
                if candidate[2] >= seed[2] * 0.65
                    && candidate[2] <= seed[2] * 1.35
                    && center_shift <= seed[2] * 0.30
                    && outer_circle_cost(points, candidate) < outer_circle_cost(points, circle)
                {
                    circle = candidate;
                }
            }
        }
        for step in &mut steps {
            *step *= 0.68;
        }
    }
    circle
}

fn solve_outer_ellipse_normal_equations(mut matrix: [[f64; 6]; 5]) -> Option<[f64; 5]> {
    for column in 0..5 {
        let pivot = (column..5).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() < 1.0e-9 {
            return None;
        }
        matrix.swap(column, pivot);
        let divisor = matrix[column][column];
        for value in column..=5 {
            matrix[column][value] /= divisor;
        }
        for row in 0..5 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for value in column..=5 {
                matrix[row][value] -= factor * matrix[column][value];
            }
        }
    }
    Some(std::array::from_fn(|index| matrix[index][5]))
}

/// Direct per-frame rotated conic fit to the same radial candidates displayed
/// in yellow. Coordinates are normalized around the coarse eye seed, with a
/// light circular ridge prior so a partially occluded arc remains bounded.
fn fit_outer_ellipse(points: &[OuterIrisPoint], seed: [f64; 3]) -> [f64; 5] {
    fit_outer_ellipse_with_weights(points, None, seed)
}

/// Weighted form used only after a contact has produced an analog edge
/// measurement.  Weights are normalized to unit mean so they change the
/// relative authority of contacts without silently changing the circular
/// ridge strength or the admissible geometry.
fn fit_outer_ellipse_with_weights(
    points: &[OuterIrisPoint],
    weights: Option<&[f64]>,
    seed: [f64; 3],
) -> [f64; 5] {
    let fallback = || {
        let circle = fit_outer_circle(points, seed);
        [circle[0], circle[1], circle[2], circle[2], 0.0]
    };
    if points.len() < 5
        || seed[2] <= 1.0
        || weights.is_some_and(|weights| weights.len() != points.len())
    {
        return fallback();
    }
    let raw_weight = |index: usize| {
        weights
            .map_or(1.0, |weights| weights[index])
            .clamp(0.02, 4.0)
    };
    let weight_sum = (0..points.len()).map(raw_weight).sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 1.0e-6 {
        return fallback();
    }
    let weight_scale = points.len() as f64 / weight_sum;
    let mut matrix = [[0.0; 6]; 5];
    for (point_index, point) in points.iter().enumerate() {
        let weight = raw_weight(point_index) * weight_scale;
        let x = (point.x - seed[0]) / seed[2];
        let y = (point.y - seed[1]) / seed[2];
        let terms = [x * x, x * y, y * y, x, y];
        for row in 0..5 {
            for column in 0..5 {
                matrix[row][column] += weight * terms[row] * terms[column];
            }
            matrix[row][5] += weight * terms[row];
        }
    }
    let ridge = 0.04;
    let circular_prior = [1.0, 0.0, 1.0, 0.0, 0.0];
    for index in 0..5 {
        matrix[index][index] += ridge;
        matrix[index][5] += ridge * circular_prior[index];
    }
    let Some([quadratic_x, cross, quadratic_y, linear_x, linear_y]) =
        solve_outer_ellipse_normal_equations(matrix)
    else {
        return fallback();
    };
    let off_diagonal = cross * 0.5;
    let determinant = quadratic_x * quadratic_y - off_diagonal * off_diagonal;
    if determinant <= 1.0e-6 {
        return fallback();
    }
    let center_x = -0.5 * (quadratic_y * linear_x - off_diagonal * linear_y) / determinant;
    let center_y = -0.5 * (-off_diagonal * linear_x + quadratic_x * linear_y) / determinant;
    let level = 1.0
        + quadratic_x * center_x * center_x
        + 2.0 * off_diagonal * center_x * center_y
        + quadratic_y * center_y * center_y;
    let trace = quadratic_x + quadratic_y;
    let discriminant = ((quadratic_x - quadratic_y).powi(2) + 4.0 * off_diagonal.powi(2)).sqrt();
    let eigen_minimum = (trace - discriminant) * 0.5;
    let eigen_maximum = (trace + discriminant) * 0.5;
    if level <= 0.0 || eigen_minimum <= 1.0e-6 || eigen_maximum <= eigen_minimum {
        return fallback();
    }
    let major_radius = seed[2] * (level / eigen_minimum).sqrt();
    let minor_radius = seed[2] * (level / eigen_maximum).sqrt();
    let major_vector = if off_diagonal.abs() > 1.0e-9 {
        (off_diagonal, eigen_minimum - quadratic_x)
    } else if quadratic_x <= quadratic_y {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let angle = major_vector.1.atan2(major_vector.0);
    let fitted_center = (seed[0] + center_x * seed[2], seed[1] + center_y * seed[2]);
    let center_shift = (fitted_center.0 - seed[0]).hypot(fitted_center.1 - seed[1]);
    if !major_radius.is_finite()
        || !minor_radius.is_finite()
        || major_radius < seed[2] * 0.55
        || major_radius > seed[2] * 1.45
        || !projected_circular_limbus_axes_plausible(major_radius, minor_radius)
        || center_shift > seed[2] * 0.35
    {
        return fallback();
    }
    [
        fitted_center.0,
        fitted_center.1,
        major_radius,
        minor_radius,
        angle,
    ]
}

fn outer_ellipse_point_residual(point: OuterIrisPoint, ellipse: [f64; 5]) -> f64 {
    let (sin, cos) = ellipse[4].sin_cos();
    let dx = point.x - ellipse[0];
    let dy = point.y - ellipse[1];
    let local_x = dx * cos + dy * sin;
    let local_y = -dx * sin + dy * cos;
    let rho =
        ((local_x / ellipse[2].max(1.0)).powi(2) + (local_y / ellipse[3].max(1.0)).powi(2)).sqrt();
    (rho - 1.0).abs() * (ellipse[2] * ellipse[3]).sqrt()
}

/// Fit the authoritative outer ellipse to the boundary points produced by a
/// completed perimeter drive.  The drive is allowed to move every sampled
/// boundary location (and a partial second lap moves a selected subset), so
/// returning the input seed's stored center afterward is internally
/// inconsistent.  Start from the driven-point centroid, then robustly refit
/// after rejecting only geometric outliers; labels and the seed pose do not
/// participate in this result.
pub fn fit_driven_limbus_ellipse(strip: &LimbusPerimeterStrip) -> Option<OuterIrisBoundary> {
    let points = strip
        .samples
        .iter()
        .enumerate()
        .filter_map(|(row, sample)| {
            (sample.driven_point.0.is_finite() && sample.driven_point.1.is_finite()).then_some(
                OuterIrisPoint {
                    x: sample.driven_point.0,
                    y: sample.driven_point.1,
                    // A projected lower-lid bridge is geometrical support for
                    // the single conic, never image evidence.  Keep its
                    // contrast at zero so downstream diagnostics cannot
                    // mistake it for a measured sclera/iris transition.
                    contrast: if strip
                        .conic_projected_rows
                        .get(row)
                        .copied()
                        .unwrap_or(sample.inferred_occlusion)
                    {
                        0.0
                    } else {
                        sample.transition_score.max(0.0)
                    },
                },
            )
        })
        .collect::<Vec<_>>();
    fit_direct_limbus_points(points, strip.samples.len())
}

/// Robustly close a directly sampled native-RAW limbus road into a projected
/// circle. This is intentionally downstream of road selection: it describes
/// the geometry that was actually driven and is suitable as the affine
/// projection reference for a later, independent pupil-boundary solve.
pub fn fit_sampled_limbus_route_ellipse(route: &[(f64, f64)]) -> Option<OuterIrisBoundary> {
    let points = route
        .iter()
        .filter_map(|&(x, y)| {
            (x.is_finite() && y.is_finite()).then_some(OuterIrisPoint {
                x,
                y,
                // The route has already survived the material scorer. Keep a
                // neutral direct-support marker here; no label or completed
                // seed geometry is injected into the refit.
                contrast: 1.0,
            })
        })
        .collect::<Vec<_>>();
    fit_direct_limbus_points(points, route.len())
}

fn fit_direct_limbus_points(
    mut points: Vec<OuterIrisPoint>,
    output_samples: usize,
) -> Option<OuterIrisBoundary> {
    if points.len() < 8 {
        return None;
    }
    let centroid = (
        points.iter().map(|point| point.x).sum::<f64>() / points.len() as f64,
        points.iter().map(|point| point.y).sum::<f64>() / points.len() as f64,
    );
    let mut centroid_radii = points
        .iter()
        .map(|point| (point.x - centroid.0).hypot(point.y - centroid.1))
        .collect::<Vec<_>>();
    let seed_radius = median(&mut centroid_radii);
    if !seed_radius.is_finite() || seed_radius <= 4.0 {
        return None;
    }
    let mut ellipse = fit_outer_ellipse(&points, [centroid.0, centroid.1, seed_radius]);
    for _ in 0..3 {
        if !ellipse.iter().all(|value| value.is_finite()) || ellipse[2] <= 4.0 || ellipse[3] <= 4.0
        {
            return None;
        }
        let mut residuals = points
            .iter()
            .map(|point| outer_ellipse_point_residual(*point, ellipse))
            .collect::<Vec<_>>();
        let residual_median = median(&mut residuals);
        let mut deviations = residuals
            .iter()
            .map(|residual| (residual - residual_median).abs())
            .collect::<Vec<_>>();
        let residual_mad = median(&mut deviations);
        let threshold = residual_median
            + (3.5 * 1.4826 * residual_mad)
                .max((ellipse[2] * ellipse[3]).sqrt() * 0.018)
                .max(0.75);
        let retained = points
            .iter()
            .copied()
            .filter(|point| outer_ellipse_point_residual(*point, ellipse) <= threshold)
            .collect::<Vec<_>>();
        if retained.len() < 8 || retained.len() == points.len() {
            break;
        }
        points = retained;
        ellipse = fit_outer_ellipse(
            &points,
            [ellipse[0], ellipse[1], (ellipse[2] * ellipse[3]).sqrt()],
        );
    }
    if !ellipse.iter().all(|value| value.is_finite())
        || ellipse[2] <= 4.0
        || ellipse[3] <= 4.0
        || !projected_circular_limbus_axes_plausible(ellipse[2], ellipse[3])
    {
        return None;
    }
    Some(OuterIrisBoundary {
        center: (ellipse[0], ellipse[1]),
        major_radius: ellipse[2],
        minor_radius: ellipse[3],
        angle: ellipse[4],
        evidence_points: points.clone(),
        points: stable_screen_space_ellipse_points(ellipse, output_samples.max(8), 1.0),
        ..OuterIrisBoundary::default()
    })
}

fn blend_outer_ellipse_angle(previous: f64, measured: f64, alpha: f64) -> f64 {
    let alpha = alpha.clamp(0.0, 1.0);
    let previous_double = previous * 2.0;
    let measured_double = measured * 2.0;
    0.5 * ((1.0 - alpha) * previous_double.sin() + alpha * measured_double.sin())
        .atan2((1.0 - alpha) * previous_double.cos() + alpha * measured_double.cos())
}

fn tracked_outer_ellipse_local(prior: TrackedOuterIrisEllipse) -> [f64; 5] {
    [
        prior.sensor_center.0,
        prior.sensor_center.1,
        prior.major_radius,
        prior.minor_radius,
        prior.angle,
    ]
}

/// Positive intersection distance between a seed-centered radial ray and a
/// rotated ellipse. The seed is normally inside the limbus; if numerical noise
/// produces two positive roots, use the one nearest the expected iris radius.
fn outer_ellipse_radius_on_ray(ellipse: [f64; 5], seed: [f64; 3], angle: f64) -> Option<f64> {
    if ellipse[2] <= 1.0 || ellipse[3] <= 1.0 {
        return None;
    }
    let (ray_y, ray_x) = angle.sin_cos();
    let (ellipse_sin, ellipse_cos) = ellipse[4].sin_cos();
    let offset_x = seed[0] - ellipse[0];
    let offset_y = seed[1] - ellipse[1];
    let local_x = offset_x * ellipse_cos + offset_y * ellipse_sin;
    let local_y = -offset_x * ellipse_sin + offset_y * ellipse_cos;
    let local_dx = ray_x * ellipse_cos + ray_y * ellipse_sin;
    let local_dy = -ray_x * ellipse_sin + ray_y * ellipse_cos;
    let major_sq = ellipse[2] * ellipse[2];
    let minor_sq = ellipse[3] * ellipse[3];
    let quadratic = local_dx * local_dx / major_sq + local_dy * local_dy / minor_sq;
    let linear = 2.0 * (local_x * local_dx / major_sq + local_y * local_dy / minor_sq);
    let constant = local_x * local_x / major_sq + local_y * local_y / minor_sq - 1.0;
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if quadratic <= 1.0e-12 || discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let roots = [
        (-linear - root) / (2.0 * quadratic),
        (-linear + root) / (2.0 * quadratic),
    ];
    roots
        .into_iter()
        .filter(|radius| radius.is_finite() && *radius > seed[2] * 0.45)
        .min_by(|left, right| (left - seed[2]).abs().total_cmp(&(right - seed[2]).abs()))
}

fn stable_screen_space_ellipse_points(
    ellipse: [f64; 5],
    count: usize,
    contrast: f64,
) -> Vec<OuterIrisPoint> {
    if count == 0 || ellipse[2] <= 0.0 || ellipse[3] <= 0.0 {
        return Vec::new();
    }
    let (rotation_sin, rotation_cos) = ellipse[4].sin_cos();
    // Anchor phase at the screen-right (3-o'clock) intersection rather than
    // at the fitted ellipse's major axis, whose equivalent angle can flip or
    // rotate from tiny frame-to-frame fit changes.
    let start_parameter = (-rotation_sin / ellipse[3]).atan2(rotation_cos / ellipse[2]);
    let point_at = |parameter: f64| {
        let major_offset = ellipse[2] * parameter.cos();
        let minor_offset = ellipse[3] * parameter.sin();
        (
            ellipse[0] + major_offset * rotation_cos - minor_offset * rotation_sin,
            ellipse[1] + major_offset * rotation_sin + minor_offset * rotation_cos,
        )
    };
    const ARC_TABLE_STEPS: usize = 512;
    let mut table = Vec::with_capacity(ARC_TABLE_STEPS + 1);
    let mut cumulative = Vec::with_capacity(ARC_TABLE_STEPS + 1);
    let mut previous = point_at(start_parameter);
    table.push(previous);
    cumulative.push(0.0);
    for step in 1..=ARC_TABLE_STEPS {
        let parameter = start_parameter + 2.0 * PI * step as f64 / ARC_TABLE_STEPS as f64;
        let point = point_at(parameter);
        let length = cumulative[step - 1] + (point.0 - previous.0).hypot(point.1 - previous.1);
        table.push(point);
        cumulative.push(length);
        previous = point;
    }
    let perimeter = cumulative[ARC_TABLE_STEPS];
    if !perimeter.is_finite() || perimeter <= 1.0e-6 {
        return Vec::new();
    }
    let mut segment = 1usize;
    (0..count)
        .map(|index| {
            let target = perimeter * index as f64 / count as f64;
            while segment < ARC_TABLE_STEPS && cumulative[segment] < target {
                segment += 1;
            }
            let before = cumulative[segment - 1];
            let after = cumulative[segment];
            let alpha = ((target - before) / (after - before).max(1.0e-9)).clamp(0.0, 1.0);
            let left = table[segment - 1];
            let right = table[segment];
            OuterIrisPoint {
                x: left.0 + (right.0 - left.0) * alpha,
                y: left.1 + (right.1 - left.1) * alpha,
                contrast,
            }
        })
        .collect()
}

/// Reject isolated angular hits, keeping only points that agree with the
/// ellipse and have neighboring support around the ring. Two iterations let a
/// strong partial arc pull the fit away from scattered lid or skin edges.
fn fit_cohesive_outer_groups(
    rays: &[Option<OuterIrisPoint>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    analog_weights: Option<&[f64; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES]>,
    seed: [f64; 3],
    initial: [f64; 5],
) -> (
    [f64; 5],
    Vec<OuterIrisPoint>,
    [f64; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
) {
    let mut ellipse = initial;
    let mut selected = rays.iter().flatten().copied().collect::<Vec<_>>();
    let mut confidences = [0.0; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
    for _ in 0..2 {
        let residuals = std::array::from_fn::<_, OUTER_IRIS_DENSE_EVIDENCE_SAMPLES, _>(|index| {
            let point = rays[index]?;
            let angle = outer_iris_evidence_angle(index);
            let expected = outer_ellipse_radius_on_ray(ellipse, seed, angle)?;
            Some(((point.x - seed[0]).hypot(point.y - seed[1]) - expected).abs())
        });
        let mut finite = residuals.iter().flatten().copied().collect::<Vec<_>>();
        if finite.len() < 5 {
            break;
        }
        let median_residual = median(&mut finite);
        let threshold = (median_residual * 2.4 + 0.8)
            .max(seed[2] * 0.045)
            .min(seed[2] * 0.13);
        let inlier = std::array::from_fn::<_, OUTER_IRIS_DENSE_EVIDENCE_SAMPLES, _>(|index| {
            residuals[index].is_some_and(|residual| residual <= threshold)
        });
        confidences = std::array::from_fn(|index| {
            let Some(residual) = residuals[index] else {
                return 0.0;
            };
            if !inlier[index] {
                return 0.0;
            }
            let neighbor_count = [1usize, 2]
                .into_iter()
                .flat_map(|distance| {
                    [
                        (index + distance) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES,
                        (index + OUTER_IRIS_DENSE_EVIDENCE_SAMPLES - distance)
                            % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES,
                    ]
                })
                .filter(|neighbor| inlier[*neighbor])
                .count();
            if neighbor_count < 2 {
                0.0
            } else {
                let geometry = (1.0 - residual / threshold).clamp(0.05, 1.0)
                    * (neighbor_count as f64 / 4.0).clamp(0.5, 1.0);
                geometry * analog_weights.map_or(1.0, |weights| weights[index].clamp(0.14, 1.0))
            }
        });
        let cohesive_indices = rays
            .iter()
            .enumerate()
            .filter_map(|(index, point)| {
                (confidences[index] > 0.0 && point.is_some()).then_some(index)
            })
            .collect::<Vec<_>>();
        if cohesive_indices.len() < 8 {
            break;
        }
        let cohesive = cohesive_indices
            .iter()
            .filter_map(|index| rays[*index])
            .collect::<Vec<_>>();
        let fit_weights = analog_weights.map(|weights| {
            cohesive_indices
                .iter()
                .map(|index| weights[*index])
                .collect::<Vec<_>>()
        });
        selected = cohesive;
        ellipse = fit_outer_ellipse_with_weights(&selected, fit_weights.as_deref(), seed);
    }
    if selected.len() < 5 {
        selected = rays.iter().flatten().copied().collect();
        confidences = std::array::from_fn(|index| rays[index].is_some() as u8 as f64);
    }
    (ellipse, selected, confidences)
}

fn fit_and_reject_outer_circle_outliers(
    evidence: &mut Vec<OuterIrisPoint>,
    seed: [f64; 3],
) -> [f64; 3] {
    let mut circle = fit_outer_circle(evidence, seed);
    for _ in 0..2 {
        let radial_residual = |point: &OuterIrisPoint| {
            ((point.x - circle[0]).hypot(point.y - circle[1]) - circle[2]).abs()
        };
        let mut residuals = evidence.iter().map(radial_residual).collect::<Vec<_>>();
        let residual_median = median(&mut residuals);
        let mut deviations = residuals
            .iter()
            .map(|residual| (residual - residual_median).abs())
            .collect::<Vec<_>>();
        let residual_mad = median(&mut deviations);
        let consensus_width = (3.5 * 1.4826 * residual_mad)
            .max(circle[2] * 0.035)
            .max(2.0);
        let threshold = residual_median + consensus_width;
        let retained = evidence
            .iter()
            .copied()
            .filter(|point| radial_residual(point) <= threshold)
            .collect::<Vec<_>>();
        if retained.len() < 8 || retained.len() == evidence.len() {
            break;
        }
        *evidence = retained;
        circle = fit_outer_circle(evidence, seed);
    }
    circle
}

#[derive(Clone)]
struct NativeLogPlane {
    width: usize,
    height: usize,
    origin_x: f64,
    origin_y: f64,
    log_rg: Vec<f64>,
    log_bg: Vec<f64>,
    reflectance_log_rg: Vec<f64>,
    reflectance_log_bg: Vec<f64>,
    intensity: Vec<f64>,
    void: Vec<f64>,
}

impl NativeLogPlane {
    fn sample_map(&self, values: &[f64], x: f64, y: f64) -> f64 {
        let grid_x = ((x - self.origin_x) * 0.25).clamp(0.0, self.width.saturating_sub(1) as f64);
        let grid_y = ((y - self.origin_y) * 0.25).clamp(0.0, self.height.saturating_sub(1) as f64);
        let x0 = grid_x.floor() as usize;
        let y0 = grid_y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = grid_x - x0 as f64;
        let fy = grid_y - y0 as f64;
        let top = values[y0 * self.width + x0] * (1.0 - fx) + values[y0 * self.width + x1] * fx;
        let bottom = values[y1 * self.width + x0] * (1.0 - fx) + values[y1 * self.width + x1] * fx;
        top * (1.0 - fy) + bottom * fy
    }

    fn sample_void(&self, x: f64, y: f64) -> f64 {
        self.sample_map(&self.void, x, y)
    }

    fn sample_chroma(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.sample_map(&self.log_rg, x, y),
            self.sample_map(&self.log_bg, x, y),
        )
    }

    fn sample_intensity(&self, x: f64, y: f64) -> f64 {
        self.sample_map(&self.intensity, x, y)
    }

    fn sample_log_intensity(&self, x: f64, y: f64) -> f64 {
        (self.sample_intensity(x, y) + 8.0).ln()
    }
}

fn blur_native_plane(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut blurred = vec![0.0; values.len()];
    for y in 0..height {
        for x in 0..width {
            let mut weighted = 0.0;
            let mut weight = 0.0;
            for offset_y in -1isize..=1 {
                let sample_y = y.saturating_add_signed(offset_y).min(height - 1);
                let weight_y = if offset_y == 0 { 2.0 } else { 1.0 };
                for offset_x in -1isize..=1 {
                    let sample_x = x.saturating_add_signed(offset_x).min(width - 1);
                    let weight_x = if offset_x == 0 { 2.0 } else { 1.0 };
                    let sample_weight = weight_x * weight_y;
                    weighted += values[sample_y * width + sample_x] * sample_weight;
                    weight += sample_weight;
                }
            }
            blurred[y * width + x] = weighted / weight;
        }
    }
    blurred
}

fn percentile_f64(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    let (_, value, _) = values.select_nth_unstable_by(index, f64::total_cmp);
    *value
}

/// Two-dimensional centered log-chromaticity for one white-balanced RAW CFA
/// cell. Multiplying every channel by the same positive illumination factor
/// also multiplies `stabilizer`, so the returned material coordinate is
/// unchanged. This differs from a fixed additive epsilon, which pulls a dark
/// eyelid-shadow sample toward neutral and can make sclera look like iris.
fn illumination_invariant_log_chroma(red: f64, green: f64, blue: f64) -> (f64, f64) {
    let mean = (red + green + blue) / 3.0;
    if !mean.is_finite() || mean <= 1.0e-9 {
        return (0.0, 0.0);
    }
    // A scale-relative floor bounds shot/read-noise excursions without
    // reintroducing intensity into the chromatic coordinate.
    let stabilizer = mean * 0.02;
    (
        ((red + stabilizer) / (green + stabilizer))
            .ln()
            .clamp(-3.0, 3.0),
        ((blue + stabilizer) / (green + stabilizer))
            .ln()
            .clamp(-3.0, 3.0),
    )
}

/// Signed local Weber contrast in log space. The stabilizer is proportional
/// to the pair's own mean, so multiplying both samples by any positive
/// achromatic illumination factor leaves the result exactly unchanged.
fn illumination_invariant_log_intensity_step(inside: f64, outside: f64) -> f64 {
    let mean = (inside + outside) * 0.5;
    if !mean.is_finite() || mean <= 1.0e-9 {
        return 0.0;
    }
    let stabilizer = mean * 0.02;
    ((outside + stabilizer) / (inside + stabilizer))
        .ln()
        .clamp(-3.0, 3.0)
}

fn native_log_plane(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
) -> Option<NativeLogPlane> {
    let start_x = ((4 - sensor_x as usize % 4) % 4) as usize;
    let start_y = ((4 - sensor_y as usize % 4) % 4) as usize;
    let plane_width = width.saturating_sub(start_x) / 4;
    let plane_height = height.saturating_sub(start_y) / 4;
    if plane_width < 8 || plane_height < 8 {
        return None;
    }
    let mut channels = Vec::with_capacity(plane_width * plane_height);
    for cell_y in 0..plane_height {
        let y = start_y + cell_y * 4;
        for cell_x in 0..plane_width {
            let x = start_x + cell_x * 4;
            let average = |offset_x: usize, offset_y: usize| {
                let i = (y + offset_y) * width + x + offset_x;
                (raw[i] as f64
                    + raw[i + 1] as f64
                    + raw[i + width] as f64
                    + raw[i + width + 1] as f64)
                    * 0.25
            };
            channels.push([average(0, 0), average(2, 0), average(0, 2), average(2, 2)]);
        }
    }
    let mut all = channels.iter().flatten().copied().collect::<Vec<_>>();
    let black = percentile_f64(&mut all, 0.001);
    let mut responses = [1.0f64; 4];
    for channel in 0..4 {
        let mut values = channels
            .iter()
            .map(|sample| (sample[channel] - black).max(0.0))
            .collect::<Vec<_>>();
        responses[channel] = percentile_f64(&mut values, 0.65).max(1.0);
    }
    let green_reference = (responses[1] + responses[2]) * 0.5;
    let gains = responses.map(|response| (green_reference / response).clamp(0.25, 4.0));
    let mut intensity = Vec::with_capacity(channels.len());
    let mut log_rg = Vec::with_capacity(channels.len());
    let mut log_bg = Vec::with_capacity(channels.len());
    let mut reflectance_log_rg = Vec::with_capacity(channels.len());
    let mut reflectance_log_bg = Vec::with_capacity(channels.len());
    for sample in channels {
        let balanced = sample.map(|value| (value - black).max(0.0));
        let red = balanced[0] * gains[0];
        let green = (balanced[1] * gains[1] + balanced[2] * gains[2]) * 0.5;
        let blue = balanced[3] * gains[3];
        intensity.push((red + green + blue) / 3.0);
        let chroma = illumination_invariant_log_chroma(red, green, blue);
        reflectance_log_rg.push(chroma.0);
        reflectance_log_bg.push(chroma.1);
        // Preserve the established noise-stabilized chroma plane for all
        // existing cues. The new reflectance plane above is intentionally
        // separate so shadow robustness cannot silently retune legacy gates.
        log_rg.push(((red + 8.0) / (green + 8.0)).ln());
        log_bg.push(((blue + 8.0) / (green + 8.0)).ln());
    }
    let origin_x = start_x as f64 + 1.5;
    let origin_y = start_y as f64 + 1.5;
    let mut iris_rg = Vec::new();
    let mut iris_bg = Vec::new();
    for y in 0..plane_height {
        for x in 0..plane_width {
            let px = origin_x + x as f64 * 4.0;
            let py = origin_y + y as f64 * 4.0;
            let rho = (px - coarse.center.0).hypot(py - coarse.center.1) / coarse.radius;
            if (0.48..=0.84).contains(&rho) {
                let index = y * plane_width + x;
                iris_rg.push(log_rg[index]);
                iris_bg.push(log_bg[index]);
            }
        }
    }
    if iris_rg.len() < 24 {
        return None;
    }
    let iris_rg_median = percentile_f64(&mut iris_rg, 0.5);
    let iris_bg_median = percentile_f64(&mut iris_bg, 0.5);
    let mut rg_deviation = log_rg
        .iter()
        .map(|value| (value - iris_rg_median).abs())
        .collect::<Vec<_>>();
    let mut bg_deviation = log_bg
        .iter()
        .map(|value| (value - iris_bg_median).abs())
        .collect::<Vec<_>>();
    let rg_mad = (1.4826 * percentile_f64(&mut rg_deviation, 0.5)).max(0.025);
    let bg_mad = (1.4826 * percentile_f64(&mut bg_deviation, 0.5)).max(0.025);
    let mut intensity_range = intensity.clone();
    let low = percentile_f64(&mut intensity_range, 0.01);
    let high = percentile_f64(&mut intensity_range, 0.99).max(low + 1.0);
    let mut void = intensity
        .iter()
        .zip(log_rg.iter().zip(&log_bg))
        .map(|(intensity, (rg, bg))| {
            let darkness = (1.0 - (intensity - low) / (high - low)).clamp(0.0, 1.0);
            let iris_z =
                ((rg - iris_rg_median) / rg_mad).powi(2) + ((bg - iris_bg_median) / bg_mad).powi(2);
            let non_iris = 1.0 - (-0.25 * iris_z).exp();
            darkness.powf(0.65) * (0.28 + 0.72 * non_iris)
        })
        .collect::<Vec<_>>();
    let mut void_range = void.clone();
    let void_low = percentile_f64(&mut void_range, 0.01);
    let void_high = percentile_f64(&mut void_range, 0.99).max(void_low + 1.0e-6);
    for value in &mut void {
        *value = ((*value - void_low) / (void_high - void_low)).clamp(0.0, 1.0);
    }
    Some(NativeLogPlane {
        width: plane_width,
        height: plane_height,
        origin_x,
        origin_y,
        log_rg,
        log_bg,
        reflectance_log_rg,
        reflectance_log_bg,
        intensity,
        void,
    })
}

/// The offline outer-limbus fitter uses a lightly blurred, CFA-neutral plane.
/// Keep that smoothing local to the outer detector: eyelid and inner-iris
/// passes deliberately consume the native-resolution maps and call this
/// constructor independently on the live path.
fn blur_outer_appearance(mut native: NativeLogPlane) -> NativeLogPlane {
    native.log_rg = blur_native_plane(&native.log_rg, native.width, native.height);
    native.log_bg = blur_native_plane(&native.log_bg, native.width, native.height);
    native.reflectance_log_rg =
        blur_native_plane(&native.reflectance_log_rg, native.width, native.height);
    native.reflectance_log_bg =
        blur_native_plane(&native.reflectance_log_bg, native.width, native.height);
    native.intensity = blur_native_plane(&native.intensity, native.width, native.height);
    native
}

/// Finds the outer iris/sclera transition without allowing the stronger upper
/// and lower eyelid edges to define the result. Native lateral evidence fits a
/// circle; the returned forty points sample that inferred outline, including
/// portions hidden by an eyelid.
pub fn detect_outer_iris_boundary(
    raw: &[u16],
    width: usize,
    height: usize,
    coarse: &BorderFocus,
) -> OuterIrisBoundary {
    detect_outer_iris_boundary_between_eyelids(raw, width, height, coarse, &[], &[])
}

pub fn detect_outer_iris_boundary_between_eyelids(
    raw: &[u16],
    width: usize,
    height: usize,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
) -> OuterIrisBoundary {
    detect_outer_iris_boundary_between_eyelids_at_sensor(
        raw,
        width,
        height,
        0,
        0,
        coarse,
        upper_eyelid,
        lower_eyelid,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn detect_outer_iris_boundary_between_eyelids_at_sensor(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
) -> OuterIrisBoundary {
    let mut tracker = OuterIrisTracker::default();
    detect_outer_iris_boundary_between_eyelids_tracked(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        &mut tracker,
    )
}

/// One-shot native limbus seed for Driving mode. Driving validates this broad
/// proposal again with its pupil-headed bilateral road and perimeter lap, so
/// it can safely search farther inside a whole-eye aperture than the ordinary
/// standalone outer-boundary detector.
#[allow(clippy::too_many_arguments)]
pub fn detect_outer_iris_boundary_between_eyelids_at_sensor_for_driving(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
) -> OuterIrisBoundary {
    let mut tracker = OuterIrisTracker::default();
    detect_outer_iris_boundary_between_eyelids_tracked_for_driving(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        &mut tracker,
    )
}

/// One-shot Driving proposal for the bounded pupil-centered second lap.  This
/// never participates in the normal native/SAM outer presentation and cannot
/// establish eye identity by itself.
#[allow(clippy::too_many_arguments)]
pub fn detect_outer_iris_boundary_between_eyelids_at_sensor_for_driving_recovery(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
) -> OuterIrisBoundary {
    let mut tracker = OuterIrisTracker::default();
    detect_outer_iris_boundary_between_eyelids_tracked_with_minimum_scale(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        &mut tracker,
        DRIVING_OUTER_IRIS_MIN_SEARCH_SCALE,
        DRIVING_OUTER_IRIS_RECOVERY_BUDGET,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn detect_outer_iris_boundary_between_eyelids_tracked(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    tracker: &mut OuterIrisTracker,
) -> OuterIrisBoundary {
    detect_outer_iris_boundary_between_eyelids_tracked_with_minimum_scale(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        tracker,
        OUTER_IRIS_MIN_SEARCH_SCALE,
        OUTER_IRIS_SYSTEM_BUDGET,
    )
}

/// Persistent native limbus seed for Driving mode. The wider contraction is
/// intentionally unavailable to Native/SAM outer-boundary presentation.
#[allow(clippy::too_many_arguments)]
pub fn detect_outer_iris_boundary_between_eyelids_tracked_for_driving(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    tracker: &mut OuterIrisTracker,
) -> OuterIrisBoundary {
    // `finish_attempt` may reduce ordinary detector density after sustained
    // deadline pressure. Reset that adaptive preview state before every
    // Driving frame so its diagnostic contract remains
    // work_stride=1/sample_stride=1.
    tracker.work_stride = 1;
    tracker.sample_stride = 1;
    detect_outer_iris_boundary_between_eyelids_tracked_with_minimum_scale(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        tracker,
        DRIVING_OUTER_IRIS_MIN_SEARCH_SCALE,
        DRIVING_OUTER_IRIS_SYSTEM_BUDGET,
    )
}

#[allow(clippy::too_many_arguments)]
fn detect_outer_iris_boundary_between_eyelids_tracked_with_minimum_scale(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    tracker: &mut OuterIrisTracker,
    minimum_search_scale: f64,
    system_budget: Duration,
) -> OuterIrisBoundary {
    let detection_started = Instant::now();
    let work_stride = tracker.work_stride();
    let sample_stride = tracker.sample_stride();
    let mut diagnostics = OuterIrisDiagnostics {
        work_stride,
        sample_stride,
        ..OuterIrisDiagnostics::default()
    };
    if width < 16
        || height < 16
        || raw.len() < width * height
        || coarse.radius < 20.0
        || coarse.radius > width.min(height) as f64 * 0.45
        || !coarse.center.0.is_finite()
        || !coarse.center.1.is_finite()
        || coarse.center.0 < 0.0
        || coarse.center.1 < 0.0
        || coarse.center.0 >= width as f64
        || coarse.center.1 >= height as f64
    {
        tracker.decay();
        tracker.finish_attempt(diagnostics, detection_started.elapsed());
        return OuterIrisBoundary::default();
    }

    diagnostics.seed_usable = true;
    let system_deadline = detection_started + system_budget;
    let seed = [coarse.center.0, coarse.center.1, coarse.radius];
    let rough_search = OuterSearchEllipse::from_coarse(coarse);
    let luma = Arc::new(BoxLuma5::new(raw, width, height));
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)
        .map(blur_outer_appearance)
        .map(Arc::new);
    let appearance = native
        .as_deref()
        .and_then(|plane| estimate_iris_sclera_appearance(plane, rough_search));
    let reflectance_appearance = native
        .as_deref()
        .and_then(|plane| estimate_iris_sclera_reflectance_appearance(plane, rough_search));
    let material_illumination = native
        .as_deref()
        .zip(appearance)
        .and_then(|(plane, appearance)| fit_material_illumination(plane, appearance, rough_search))
        .filter(|model| material_illumination_is_reliable(*model))
        .map(|current| {
            let stabilized = tracker.material_illumination.blend_prior(current);
            tracker.material_illumination.observe(current);
            stabilized
        });
    let sclera_probability = native
        .as_deref()
        .zip(appearance)
        .map(|(plane, appearance)| Arc::new(iris_sclera_probability_map(plane, appearance)));
    let reflectance_sclera_probability =
        native
            .as_deref()
            .zip(reflectance_appearance)
            .map(|(plane, appearance)| {
                Arc::new(iris_sclera_reflectance_probability_map(plane, appearance))
            });
    let dense_upper_eyelid = Arc::new(upper_eyelid.to_vec());
    let dense_lower_eyelid = Arc::new(lower_eyelid.to_vec());
    let sclera_color = native
        .as_deref()
        .and_then(|plane| estimate_lateral_sclera_color(plane, seed));
    // The fused branch is the normal path. It searches along the projected
    // ellipse's surface normals and scores both the local luma transition and
    // a sustained sclera-like plateau outside the candidate. A luma-only
    // branch is evaluated only when the fused fit violates rough 3D geometry;
    // this rejects the two observed aliases: the inner pupil rim and an
    // oversized eyelid/skin contour.
    let Some(fused) = run_outer_iris_branch(
        Arc::clone(&luma),
        native.clone(),
        sclera_probability.clone(),
        reflectance_sclera_probability.clone(),
        material_illumination,
        Arc::clone(&dense_upper_eyelid),
        Arc::clone(&dense_lower_eyelid),
        width,
        height,
        seed,
        rough_search,
        OuterScoreMode::Fused,
        minimum_search_scale,
        work_stride,
        sample_stride,
        system_deadline,
        &mut diagnostics,
    ) else {
        tracker.decay();
        tracker.finish_attempt(diagnostics, detection_started.elapsed());
        return OuterIrisBoundary::default();
    };
    let branch = if outer_geometry_plausible(fused.ellipse, rough_search) {
        fused
    } else {
        let luma_branch = run_outer_iris_branch(
            Arc::clone(&luma),
            native.clone(),
            sclera_probability.clone(),
            reflectance_sclera_probability.clone(),
            material_illumination,
            Arc::clone(&dense_upper_eyelid),
            Arc::clone(&dense_lower_eyelid),
            width,
            height,
            seed,
            rough_search,
            OuterScoreMode::Luma,
            minimum_search_scale,
            // The optional fallback receives a fixed half-density operation
            // budget. Host load must not decide whether identical RAW sees
            // 79 or 40 luma rays.
            (work_stride * 2).min(4),
            sample_stride,
            system_deadline,
            &mut diagnostics,
        );
        let Some(luma_branch) = luma_branch
            .filter(|candidate| outer_geometry_plausible(candidate.ellipse, rough_search))
        else {
            // The old path selected whichever invalid branch had the smaller
            // area/center penalty.  That allowed a stable lid or brow conic to
            // publish at 3:1 or worse.  A camera-inadmissible projected circle
            // is now a failed observation, never a lower-scored iris.
            tracker.decay();
            tracker.finish_attempt(diagnostics, detection_started.elapsed());
            return OuterIrisBoundary::default();
        };
        luma_branch
    };
    let ellipse = branch.ellipse;
    let dense_evidence = branch.evidence;
    let occluded_points = branch.occluded_points;
    let curve_confidence = branch.curve_confidence;
    let sweep_angle = outer_iris_lateral_sweep_angle();
    let curve_paths: [Vec<Option<OuterIrisPoint>>; 4] = std::array::from_fn(|branch| {
        let side_angle = if branch < 2 { 0.0 } else { PI };
        let sweep_direction = if branch % 2 == 0 { -1.0 } else { 1.0 };
        (0..OUTER_IRIS_SWEEP_BRANCH_SAMPLES)
            .map(|step| {
                let phase = step as f64 / (OUTER_IRIS_SWEEP_BRANCH_SAMPLES - 1) as f64;
                let angle = side_angle + sweep_direction * sweep_angle * phase;
                best_outer_iris_candidate(luma.as_ref(), width, height, seed, angle).filter(
                    |point| {
                        let below_upper =
                            eyelid_y_at_x(upper_eyelid, point.x).is_none_or(|y| point.y > y);
                        let above_lower =
                            eyelid_y_at_x(lower_eyelid, point.x).is_none_or(|y| point.y < y);
                        below_upper && above_lower
                    },
                )
            })
            .collect()
    });
    let cue_paths: [Vec<Option<OuterIrisPoint>>; 4] =
        std::array::from_fn(|branch| {
            let side_angle = if branch < 2 { 0.0 } else { PI };
            let sweep_direction = if branch % 2 == 0 { -1.0 } else { 1.0 };
            curve_paths[branch]
                .iter()
                .enumerate()
                .map(|(step, candidate)| {
                    let phase = step as f64 / (OUTER_IRIS_SWEEP_BRANCH_SAMPLES - 1) as f64;
                    let angle = side_angle + sweep_direction * sweep_angle * phase;
                    candidate.filter(|point| {
                        // The exact 3/9-o'clock anchors establish each pair of
                        // hands. Tip vetoes begin only once an arm is sweeping.
                        step == 0
                            || (native.as_deref().zip(sclera_color).is_none_or(
                                |(plane, reference)| {
                                    outer_tip_matches_sclera_color(plane, reference, *point, angle)
                                },
                            ) && !outer_tip_has_transverse_edge(luma.as_ref(), *point, angle))
                    })
                })
                .collect()
        });
    let (_sweep_evidence, cue_evidence) =
        select_lateral_outer_curve_sweeps_with_fallback(&cue_paths, &curve_paths, seed);

    tracker.observe(
        sensor_x,
        sensor_y,
        ellipse,
        curve_confidence,
        dense_evidence.len(),
    );
    diagnostics.accepted = true;
    tracker.finish_attempt(diagnostics, detection_started.elapsed());
    let mut contrasts = dense_evidence
        .iter()
        .map(|point| point.contrast)
        .collect::<Vec<_>>();
    let contrast = median(&mut contrasts).max(1.0);
    let points = stable_screen_space_ellipse_points(ellipse, 64, contrast);
    OuterIrisBoundary {
        center: (ellipse[0], ellipse[1]),
        major_radius: ellipse[2],
        minor_radius: ellipse[3],
        angle: ellipse[4],
        evidence_points: dense_evidence,
        occluded_points,
        veto_sweep_endpoints: lateral_sweep_endpoints(&cue_evidence, seed),
        points,
    }
}

fn grain_great_circle_objective(luma: &BoxLuma5, center: (f64, f64), radius: f64) -> (f64, usize) {
    if radius < 8.0 {
        return (f64::NEG_INFINITY, 0);
    }
    // A gnomonic lift gives every image radius a true angular distance from
    // the candidate iris pole. Grain following then compares like distances
    // even when perspective spreads the outer samples more than inner ones.
    let focal = radius * 2.5;
    let boundary_angle = (radius / focal).atan();
    let sample_radius = |fraction: f64| focal * (boundary_angle * fraction).tan();
    let grain_energy = |angle: f64, fraction: f64| -> Option<f64> {
        let (direction_y, direction_x) = angle.sin_cos();
        let distance = sample_radius(fraction);
        let x = center.0 + direction_x * distance;
        let y = center.1 + direction_y * distance;
        if x <= 4.0
            || y <= 4.0
            || x >= luma.width.saturating_sub(5) as f64
            || y >= luma.height.saturating_sub(5) as f64
        {
            return None;
        }
        let gradient_x = luma.sample3(x + 1.5, y) - luma.sample3(x - 1.5, y);
        let gradient_y = luma.sample3(x, y + 1.5) - luma.sample3(x, y - 1.5);
        let magnitude = gradient_x.hypot(gradient_y);
        if magnitude < 1.0 {
            return Some(0.0);
        }
        // Iris fibres run approximately along the pole-to-limbus great
        // circles. Their image tangent is perpendicular to the local luma
        // gradient. Squared alignment is polarity-independent.
        let tangent_x = -gradient_y / magnitude;
        let tangent_y = gradient_x / magnitude;
        let alignment = (tangent_x * direction_x + tangent_y * direction_y).powi(2);
        Some(alignment * (magnitude / 56.0).clamp(0.0, 1.0))
    };

    let mut ray_scores = Vec::with_capacity(33);
    let mut support = 0usize;
    let mut left_support = 0usize;
    let mut right_support = 0usize;
    // Use the exposed lower semicircle, including both 3/9-o'clock anchors.
    // This matches the occlusion policy of the material limbus detector.
    for ray in 0..=32 {
        let angle = PI * ray as f64 / 32.0;
        let inner = [0.52, 0.66, 0.78, 0.88, 0.96]
            .into_iter()
            .filter_map(|fraction| grain_energy(angle, fraction))
            .collect::<Vec<_>>();
        let outer = [1.06, 1.15, 1.24]
            .into_iter()
            .filter_map(|fraction| grain_energy(angle, fraction))
            .collect::<Vec<_>>();
        if inner.len() < 4 || outer.len() < 2 {
            continue;
        }
        let inner_mean = inner.iter().sum::<f64>() / inner.len() as f64;
        let inner_floor = inner.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0);
        let outer_mean = outer.iter().sum::<f64>() / outer.len() as f64;
        // Reward a coherent radial tract up to the hypothesized limbus and a
        // loss of that tract beyond it. A fibre still visible below a proposed
        // bottom edge therefore pushes the radius farther outward.
        let ray_score = 0.72 * inner_mean + 0.28 * inner_floor - 0.82 * outer_mean;
        ray_scores.push(ray_score);
        if inner_mean >= 0.045 && inner_mean >= outer_mean * 1.08 {
            support += 1;
            if ray < 16 {
                left_support += 1;
            } else if ray > 16 {
                right_support += 1;
            }
        }
    }
    // Reflected lashes on the iris are strong but form a sparse, nearly
    // parallel family. Genuine iris grain is a converging fan with support on
    // both sides of the pupil. Require that angular coverage, then use the
    // median and lower quartile so a handful of intense reflection lines are
    // trimmed out of the center/radius objective.
    if ray_scores.len() < 20 || support < 12 || left_support < 5 || right_support < 5 {
        return (f64::NEG_INFINITY, support);
    }
    ray_scores.sort_by(f64::total_cmp);
    let median_score = ray_scores[ray_scores.len() / 2];
    let lower_quartile = ray_scores[ray_scores.len() / 4];
    (0.68 * median_score + 0.32 * lower_quartile, support)
}

/// Refine a material-derived limbus hypothesis by following coherent iris
/// grain along pole-centered great circles. This is a bounded live refinement:
/// it cannot invent an eye without a valid native seed and falls back cleanly
/// when grain support is weak.
pub fn refine_outer_iris_boundary_by_grain(
    raw: &[u16],
    width: usize,
    height: usize,
    initial: &OuterIrisBoundary,
) -> Option<GrainIrisRefinement> {
    if raw.len() < width * height
        || initial.points.len() < 8
        || initial.major_radius <= 4.0
        || initial.minor_radius <= 4.0
    {
        return None;
    }
    let luma = BoxLuma5::new(raw, width, height);
    let initial_radius = (initial.major_radius * initial.minor_radius).sqrt();
    let mut center = initial.center;
    let mut radius = initial_radius;
    let mut step_center = (initial_radius * 0.055).clamp(1.5, 5.0);
    let mut step_radius = (initial_radius * 0.075).clamp(2.0, 7.0);
    let mut best = grain_great_circle_objective(&luma, center, radius);
    let mut iterations = 0usize;
    for _ in 0..6 {
        iterations += 1;
        let proposals = [
            (center.0 - step_center, center.1, radius),
            (center.0 + step_center, center.1, radius),
            (center.0, center.1 - step_center, radius),
            (center.0, center.1 + step_center, radius),
            (center.0, center.1, radius - step_radius),
            (center.0, center.1, radius + step_radius),
        ];
        let mut improved = false;
        for (candidate_x, candidate_y, candidate_radius) in proposals {
            if !(initial_radius * 0.78..=initial_radius * 1.28).contains(&candidate_radius) {
                continue;
            }
            let candidate =
                grain_great_circle_objective(&luma, (candidate_x, candidate_y), candidate_radius);
            if candidate.1 >= 12 && candidate.0 > best.0 + 0.002 {
                center = (candidate_x, candidate_y);
                radius = candidate_radius;
                best = candidate;
                improved = true;
            }
        }
        step_center *= if improved { 0.78 } else { 0.52 };
        step_radius *= if improved { 0.78 } else { 0.52 };
    }
    if best.1 < 12 || !best.0.is_finite() || best.0 < 0.015 {
        return None;
    }
    let scale = radius / initial_radius.max(1.0);
    let ellipse = [
        center.0,
        center.1,
        initial.major_radius * scale,
        initial.minor_radius * scale,
        initial.angle,
    ];
    let contrast = (best.0 * 256.0).clamp(1.0, 255.0);
    let evidence_points = (0..=20)
        .map(|index| {
            let angle = PI * index as f64 / 20.0;
            OuterIrisPoint {
                x: center.0 + radius * angle.cos(),
                y: center.1 + radius * angle.sin(),
                contrast,
            }
        })
        .collect();
    Some(GrainIrisRefinement {
        boundary: OuterIrisBoundary {
            center,
            major_radius: ellipse[2],
            minor_radius: ellipse[3],
            angle: ellipse[4],
            evidence_points,
            occluded_points: initial.occluded_points.clone(),
            veto_sweep_endpoints: initial.veto_sweep_endpoints.clone(),
            points: stable_screen_space_ellipse_points(ellipse, 64, contrast),
        },
        score: best.0,
        support: best.1,
        iterations,
        used_native_seed: true,
        used_pupil_bootstrap: false,
    })
}

/// Locate the limbus from iris grain when the material detector has not yet
/// completed. A valid native boundary is preferred, but the rough pupillary
/// center/ellipse is a sufficient bounded starting hypothesis.
pub fn locate_outer_iris_boundary_by_grain(
    raw: &[u16],
    width: usize,
    height: usize,
    coarse: &BorderFocus,
    native_seed: Option<&OuterIrisBoundary>,
) -> Option<GrainIrisRefinement> {
    if let Some(native_seed) = native_seed.filter(|seed| seed.points.len() >= 8) {
        return refine_outer_iris_boundary_by_grain(raw, width, height, native_seed);
    }
    if coarse.radius < 12.0
        || !coarse.center.0.is_finite()
        || !coarse.center.1.is_finite()
        || coarse.center.0 < 0.0
        || coarse.center.1 < 0.0
        || coarse.center.0 >= width as f64
        || coarse.center.1 >= height as f64
    {
        let center = darkest_center(raw, width, height)?;
        let luma = BoxLuma5::new(raw, width, height);
        let minimum_radius = (width.min(height) as f64 * 0.15).max(24.0);
        let maximum_radius = (width.min(height) as f64 * 0.42).min(112.0);
        let mut best_radius: Option<(f64, (f64, usize))> = None;
        let mut radius = minimum_radius;
        while radius <= maximum_radius {
            let objective = grain_great_circle_objective(&luma, center, radius);
            if objective.1 >= 12
                && best_radius
                    .as_ref()
                    .is_none_or(|(_, best)| objective.0 > best.0)
            {
                best_radius = Some((radius, objective));
            }
            radius += 4.0;
        }
        let (radius, _) = best_radius?;
        let ellipse = [center.0, center.1, radius, radius, 0.0];
        let bootstrap_seed = OuterIrisBoundary {
            center,
            major_radius: radius,
            minor_radius: radius,
            angle: 0.0,
            evidence_points: Vec::new(),
            occluded_points: Vec::new(),
            veto_sweep_endpoints: Vec::new(),
            points: stable_screen_space_ellipse_points(ellipse, 64, 1.0),
        };
        let mut refinement =
            refine_outer_iris_boundary_by_grain(raw, width, height, &bootstrap_seed)?;
        refinement.used_native_seed = false;
        refinement.used_pupil_bootstrap = true;
        return Some(refinement);
    }
    let search = OuterSearchEllipse::from_coarse(coarse);
    let ellipse = [
        search.center.0,
        search.center.1,
        search.major_radius,
        search.minor_radius,
        search.angle,
    ];
    let rough_seed = OuterIrisBoundary {
        center: search.center,
        major_radius: search.major_radius,
        minor_radius: search.minor_radius,
        angle: search.angle,
        evidence_points: Vec::new(),
        occluded_points: Vec::new(),
        veto_sweep_endpoints: Vec::new(),
        points: stable_screen_space_ellipse_points(ellipse, 64, 1.0),
    };
    let mut refinement = refine_outer_iris_boundary_by_grain(raw, width, height, &rough_seed)?;
    refinement.used_native_seed = false;
    refinement.used_pupil_bootstrap = false;
    Some(refinement)
}

#[derive(Clone, Copy, Debug)]
struct ScleraColorReference {
    log_rg: f64,
    log_bg: f64,
    tolerance: f64,
}

#[derive(Clone, Copy, Debug)]
struct OuterSearchEllipse {
    center: (f64, f64),
    major_radius: f64,
    minor_radius: f64,
    angle: f64,
}

impl OuterSearchEllipse {
    fn from_coarse(coarse: &BorderFocus) -> Self {
        // This is an eye-basin search window, not a published limbus.  A lid-
        // bounded aperture can be much flatter than the iris inside it, so do
        // not apply the projected-circle gate until a candidate conic has
        // actually been fitted.  The historical 2.4 cap remains only as a
        // finite work-window bound.
        let mut ratio = coarse.axis_ratio.abs().clamp(0.45, 2.40);
        let mut angle = coarse.axis_angle;
        if ratio < 1.0 {
            ratio = 1.0 / ratio;
            angle += PI * 0.5;
        }
        let ratio_root = ratio.sqrt();
        Self {
            center: coarse.center,
            major_radius: coarse.radius * ratio_root,
            minor_radius: coarse.radius / ratio_root,
            angle,
        }
    }

    fn from_fit(ellipse: [f64; 5]) -> Self {
        let mut search = Self {
            center: (ellipse[0], ellipse[1]),
            major_radius: ellipse[2],
            minor_radius: ellipse[3],
            angle: ellipse[4],
        };
        if search.major_radius < search.minor_radius {
            std::mem::swap(&mut search.major_radius, &mut search.minor_radius);
            search.angle += PI * 0.5;
        }
        search
    }

    fn equivalent_radius(self) -> f64 {
        (self.major_radius * self.minor_radius).sqrt()
    }

    fn point_and_normal(self, angle: f64, scale: f64) -> ((f64, f64), (f64, f64)) {
        let (sin, cos) = angle.sin_cos();
        let (ellipse_sin, ellipse_cos) = self.angle.sin_cos();
        let local_x = self.major_radius * cos * scale;
        let local_y = self.minor_radius * sin * scale;
        let x = self.center.0 + ellipse_cos * local_x - ellipse_sin * local_y;
        let y = self.center.1 + ellipse_sin * local_x + ellipse_cos * local_y;
        let normal_local_x = cos / self.major_radius.max(1.0);
        let normal_local_y = sin / self.minor_radius.max(1.0);
        let normal_x = ellipse_cos * normal_local_x - ellipse_sin * normal_local_y;
        let normal_y = ellipse_sin * normal_local_x + ellipse_cos * normal_local_y;
        let length = normal_x.hypot(normal_y).max(1.0e-9);
        ((x, y), (normal_x / length, normal_y / length))
    }

    fn normalized_coordinates(self, x: f64, y: f64) -> (f64, f64) {
        let (sin, cos) = self.angle.sin_cos();
        let dx = x - self.center.0;
        let dy = y - self.center.1;
        (
            (dx * cos + dy * sin) / self.major_radius.max(1.0),
            (-dx * sin + dy * cos) / self.minor_radius.max(1.0),
        )
    }

    /// Return the farther intersection of a sensor-space ray with this
    /// ellipse. `direction` need not be normalized; the returned scale is in
    /// multiples of that vector. Driving calls this with pupil-point, so a
    /// value greater than one is the opposite limbus beyond the pupil.
    fn far_ray_intersection_scale(self, point: (f64, f64), direction: (f64, f64)) -> Option<f64> {
        let start = self.normalized_coordinates(point.0, point.1);
        let end = self.normalized_coordinates(point.0 + direction.0, point.1 + direction.1);
        let delta = (end.0 - start.0, end.1 - start.1);
        let a = delta.0 * delta.0 + delta.1 * delta.1;
        let b = 2.0 * (start.0 * delta.0 + start.1 * delta.1);
        let c = start.0 * start.0 + start.1 * start.1 - 1.0;
        if !a.is_finite() || a <= 1.0e-12 {
            return None;
        }
        let discriminant = b * b - 4.0 * a * c;
        if !discriminant.is_finite() || discriminant < 0.0 {
            return None;
        }
        let root = discriminant.sqrt();
        let first = (-b - root) / (2.0 * a);
        let second = (-b + root) / (2.0 * a);
        [first, second]
            .into_iter()
            .filter(|scale| scale.is_finite() && *scale > 1.02)
            .max_by(f64::total_cmp)
    }
}

/// Follow the supplied limbus estimate in ellipse-normal coordinates and
/// locally steer toward the strongest ordered iris-to-sclera transition. The
/// returned strip retains a nominal five-millimetre view on either side by
/// default; positive normal distance is placed on the left, so the visual
/// topology is `sclera | limbus | iris/pupil`.
pub fn debug_drive_limbus_perimeter_strip(
    raw: &[u16],
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
    nominal_mm_per_side: f64,
    sample_count: usize,
) -> Option<LimbusPerimeterStrip> {
    debug_drive_limbus_perimeter_strip_internal(
        raw,
        width,
        height,
        outer,
        None,
        nominal_mm_per_side,
        sample_count,
        1,
    )
}

/// Pupil-anchored variant used by Driving. The first lap is still a purely
/// photometric closed loop. If that lap points away from the acquired pupil
/// or makes a conspicuous turn at a lid/limbus fork, only the doubtful arc is
/// driven again with the pupil-heading evidence enabled.
pub fn debug_drive_limbus_perimeter_strip_with_pupil(
    raw: &[u16],
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
    pupil_center: (f64, f64),
    nominal_mm_per_side: f64,
    sample_count: usize,
) -> Option<LimbusPerimeterStrip> {
    debug_drive_limbus_perimeter_strip_with_pupil_lap_limit(
        raw,
        width,
        height,
        outer,
        pupil_center,
        nominal_mm_per_side,
        sample_count,
        3,
    )
}

/// Diagnostic variant with an explicit lap ceiling. Production uses three:
/// the edge-only circuit, a pupil-headed bounded repair, and a measured-only
/// conic-closure circuit. Keeping the ceiling explicit lets RAW regressions
/// compare a new lap against the exact same earlier path.
pub fn debug_drive_limbus_perimeter_strip_with_pupil_lap_limit(
    raw: &[u16],
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
    pupil_center: (f64, f64),
    nominal_mm_per_side: f64,
    sample_count: usize,
    maximum_laps: usize,
) -> Option<LimbusPerimeterStrip> {
    debug_drive_limbus_perimeter_strip_internal(
        raw,
        width,
        height,
        outer,
        Some(pupil_center),
        nominal_mm_per_side,
        sample_count,
        maximum_laps.clamp(1, 3),
    )
}

fn debug_drive_limbus_perimeter_strip_internal(
    raw: &[u16],
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
    pupil_center: Option<(f64, f64)>,
    nominal_mm_per_side: f64,
    sample_count: usize,
    maximum_laps: usize,
) -> Option<LimbusPerimeterStrip> {
    if raw.len() < width.saturating_mul(height)
        || outer.major_radius < 12.0
        || outer.minor_radius < 8.0
        || !outer.center.0.is_finite()
        || !outer.center.1.is_finite()
    {
        return None;
    }
    let search = OuterSearchEllipse::from_fit([
        outer.center.0,
        outer.center.1,
        outer.major_radius,
        outer.minor_radius,
        outer.angle,
    ]);
    let equivalent_radius = search.equivalent_radius();
    let pixels_per_mm = equivalent_radius * 2.0 / 12.0;
    let nominal_mm_per_side = nominal_mm_per_side.clamp(1.0, 6.0);
    let outside_extent = (nominal_mm_per_side * pixels_per_mm).max(8.0);
    let guide_half_width = (0.65 * pixels_per_mm).clamp(3.0, 10.0);
    let maximum_steer = guide_half_width * 1.75;
    let steer_step = (pixels_per_mm * 0.13).clamp(0.65, 1.75);
    let probe_distances = [
        (pixels_per_mm * 0.25).clamp(2.0, 4.0),
        (pixels_per_mm * 0.55).clamp(4.0, 7.0),
        (pixels_per_mm * 0.90).clamp(7.0, 11.0),
    ];
    let luma = BoxLuma5::new(raw, width, height);
    let sample_count = sample_count.clamp(48, 360);
    // "Lateral" means image left/right, not the fitted ellipse's local major
    // axis. A nearly circular conic has an arbitrary major-axis angle; using
    // phase.cos() made that arbitrary angle rotate trusted anchors onto an
    // eyelid. Keep the camera-space normal components for every later lap.
    let camera_normals = (0..sample_count)
        .map(|index| {
            let phase = 2.0 * PI * index as f64 / sample_count as f64;
            search.point_and_normal(phase, 1.0).1
        })
        .collect::<Vec<_>>();
    let start = (-maximum_steer / steer_step).ceil() as i32;
    let end = (maximum_steer / steer_step).floor() as i32;
    let offsets = (start..=end)
        .map(|step| f64::from(step) * steer_step)
        .collect::<Vec<_>>();
    let state_count = offsets.len();
    let mut local_scores = vec![f64::NEG_INFINITY; sample_count * state_count];
    let mut measurements = vec![(0.0f64, 0.0f64); sample_count * state_count];
    let mut pupil_heading_scores = vec![0.0f64; sample_count * state_count];
    let mut opposite_sclera_scores = vec![0.0f64; sample_count * state_count];
    let pupil_luma = pupil_center.map(|center| luma.sample3(center.0, center.1));
    for index in 0..sample_count {
        let phase = 2.0 * PI * index as f64 / sample_count as f64;
        let (base_point, normal) = search.point_and_normal(phase, 1.0);
        let visibility = 0.34 + 0.66 * normal.0.abs();
        for (state, offset) in offsets.iter().copied().enumerate() {
            let point = (
                base_point.0 + normal.0 * offset,
                base_point.1 + normal.1 * offset,
            );
            let outside_luma = probe_distances
                .iter()
                .map(|distance| {
                    luma.sample3(point.0 + normal.0 * distance, point.1 + normal.1 * distance)
                })
                .sum::<f64>()
                / probe_distances.len() as f64;
            let inside_luma = probe_distances
                .iter()
                .map(|distance| {
                    luma.sample3(point.0 - normal.0 * distance, point.1 - normal.1 * distance)
                })
                .sum::<f64>()
                / probe_distances.len() as f64;
            // Normalize photometric evidence before combining it with spatial
            // costs. The former RAW-code objective made a one-row lid spike
            // hundreds of times stronger than continuity, effectively
            // reducing the drive to independent greedy row choices.
            let transition = ((outside_luma - inside_luma) / 180.0).clamp(-1.25, 1.25);
            let (pupil_heading_score, opposite_sclera_score) =
                pupil_center
                    .zip(pupil_luma)
                    .map_or((0.0, 0.0), |(pupil, pupil_luma)| {
                        let dx = pupil.0 - point.0;
                        let dy = pupil.1 - point.1;
                        let distance = dx.hypot(dy);
                        if distance < pixels_per_mm * 1.2 {
                            return (-1.0, -1.0);
                        }
                        let heading = (dx / distance, dy / distance);
                        let at_fraction = |fraction: f64| {
                            luma.sample3(
                                point.0 + heading.0 * distance * fraction,
                                point.1 + heading.1 * distance * fraction,
                            )
                        };
                        // The first pair is deliberately close to the candidate.
                        // Correct limbus enters iris immediately. A lid branch
                        // often heads through bright sclera before it reaches the
                        // same dark pupil later, which the old edge-only drive
                        // could not distinguish.
                        let head_near = 0.5 * (at_fraction(0.10) + at_fraction(0.20));
                        let head_middle = 0.5 * (at_fraction(0.38) + at_fraction(0.58));
                        let immediate_entry = ((outside_luma - head_near) / 180.0).clamp(-1.0, 1.0);
                        let sustained_iris =
                            ((outside_luma - head_middle) / 220.0).clamp(-1.0, 1.0);
                        let enclosed_pupil = ((outside_luma - pupil_luma) / 260.0).clamp(-1.0, 1.0);
                        // Look through the pupil to the far side of the same
                        // fitted eye, not merely *at* the pupil.  At a true
                        // limbus, the chord leaves iris/pupil and becomes
                        // sclera again. A lower-lid branch can share the near
                        // dark target but usually cannot supply this opposite
                        // iris-to-sclera exit.
                        let opposite_sclera = search
                            .far_ray_intersection_scale(point, (dx, dy))
                            .map_or(-0.35, |far_scale| {
                                let probe_scale =
                                    (probe_distances[1] / distance).clamp(0.025, 0.22);
                                let far_inside_scale = (far_scale - probe_scale).max(1.04);
                                let far_outside_scale = far_scale + probe_scale;
                                let far_iris_scale =
                                    (1.0 + (far_scale - 1.0) * 0.58).min(far_inside_scale);
                                let sightline_sample = |scale: f64| {
                                    luma.sample3(point.0 + dx * scale, point.1 + dy * scale)
                                };
                                let far_inside = 0.5
                                    * (sightline_sample(far_inside_scale)
                                        + sightline_sample(far_iris_scale));
                                let far_outside = sightline_sample(far_outside_scale);
                                ((far_outside - far_inside) / 180.0).clamp(-1.0, 1.0)
                            });
                        let normal_agreement = ((pupil.0 - point.0) * -normal.0
                            + (pupil.1 - point.1) * -normal.1)
                            / distance;
                        let inward_heading = ((normal_agreement + 0.10) / 0.85).clamp(0.0, 1.0);
                        let score = (0.34 * immediate_entry
                            + 0.18 * sustained_iris
                            + 0.11 * enclosed_pupil
                            + 0.27 * opposite_sclera
                            + 0.10 * inward_heading)
                            .clamp(-1.0, 1.0);
                        (score, opposite_sclera)
                    });
            let seed_tether = 0.16 * (offset / pixels_per_mm).powi(2);
            let flat_index = index * state_count + state;
            local_scores[flat_index] = transition * visibility - seed_tether;
            measurements[flat_index] = (outside_luma, inside_luma);
            pupil_heading_scores[flat_index] = pupil_heading_score;
            opposite_sclera_scores[flat_index] = opposite_sclera_score;
        }
    }
    let first_path =
        closed_limbus_lookahead_path(&local_scores, sample_count, &offsets, pixels_per_mm)?;
    let mut path = first_path.clone();
    let mut affine_predicted_offset_mm = first_path
        .iter()
        .map(|state| offsets[*state] / pixels_per_mm)
        .collect::<Vec<_>>();
    let mut affine_ellipse_residual_mm = vec![0.0f64; sample_count];
    let mut affine_ellipse_residual_threshold_mm = None;
    let mut affine_ellipse_first_rms_mm = None;
    let mut revisited_rows = vec![false; sample_count];
    let mut cyclic_reentry_rows = vec![false; sample_count];
    let mut full_chord_flyby_rows = vec![false; sample_count];
    let mut lap_count = 1usize;
    let mut affine_reinforced = false;
    if pupil_center.is_some() && maximum_laps >= 2 {
        let selected_heading = (0..sample_count)
            .map(|row| pupil_heading_scores[row * state_count + first_path[row]])
            .collect::<Vec<_>>();
        let mut lateral_heading = (0..sample_count)
            .filter(|row| {
                camera_normals[*row].0.abs() >= 0.58
                    && measurements[*row * state_count + first_path[*row]].0
                        > measurements[*row * state_count + first_path[*row]].1
            })
            .map(|row| selected_heading[row])
            .collect::<Vec<_>>();
        lateral_heading.sort_by(f64::total_cmp);
        let heading_reference = lateral_heading
            .get(lateral_heading.len() / 2)
            .copied()
            .unwrap_or_else(|| {
                let mut values = selected_heading.clone();
                values.sort_by(f64::total_cmp);
                values[values.len() / 2]
            });
        let mut lateral_offsets = (0..sample_count)
            .filter(|row| camera_normals[*row].0.abs() >= 0.58)
            .map(|row| offsets[first_path[row]])
            .collect::<Vec<_>>();
        lateral_offsets.sort_by(f64::total_cmp);
        let lateral_offset_reference = lateral_offsets[lateral_offsets.len() / 2];
        // Do not wait for the edge-only route to declare itself suspicious.
        // A lid can be smooth, closed, and photometrically strong, so every
        // state gets one independent closed-loop lookahead that asks what lies
        // ahead on the way to the already acquired pupil.
        let pupil_lookahead_path = closed_limbus_pupil_lookahead_path(
            &local_scores,
            &pupil_heading_scores,
            &opposite_sclera_scores,
            sample_count,
            &offsets,
            pixels_per_mm,
        )
        .unwrap_or_else(|| first_path.clone());
        let full_chord_challenger_rows = (0..sample_count)
            .map(|row| {
                let current = first_path[row];
                let challenger = pupil_lookahead_path[row];
                let current_opposite = opposite_sclera_scores[row * state_count + current];
                let challenger_opposite = opposite_sclera_scores[row * state_count + challenger];
                challenger != current
                    && challenger_opposite >= 0.10
                    && challenger_opposite - current_opposite >= 0.08
            })
            .collect::<Vec<_>>();
        // A closed Viterbi challenger can still be captured wholesale by a
        // long lid/outer-eye edge.  Make a second, bounded orbital fly-by:
        // each meridian nominates the state with the strongest complete
        // near-limbus/pupil/far-sclera chord.  A five-meridian cyclic median
        // and three-hit consensus reject isolated bright far-side patches.
        // These fly-bys never publish pixels directly; they only provide a
        // target to the subsequent closed, continuity-constrained repair.
        let full_chord_flyby_path = (0..sample_count)
            .map(|row| {
                (0..state_count)
                    .max_by(|left, right| {
                        let score = |state: usize| {
                            let flat = row * state_count + state;
                            local_scores[flat]
                                + 0.45 * pupil_heading_scores[flat]
                                + 1.55 * opposite_sclera_scores[flat]
                        };
                        score(*left).total_cmp(&score(*right))
                    })
                    .unwrap_or(first_path[row])
            })
            .collect::<Vec<_>>();
        let full_chord_flyby_smoothed_offset = (0..sample_count)
            .map(|row| {
                let mut neighborhood = (0..5)
                    .map(|delta| {
                        let neighbor = (row + sample_count + delta - 2) % sample_count;
                        offsets[full_chord_flyby_path[neighbor]]
                    })
                    .collect::<Vec<_>>();
                neighborhood.sort_by(f64::total_cmp);
                neighborhood[neighborhood.len() / 2]
            })
            .collect::<Vec<_>>();
        full_chord_flyby_rows = (0..sample_count)
            .map(|row| {
                let target = full_chord_flyby_smoothed_offset[row];
                let coherent = (0..5)
                    .filter(|delta| {
                        let neighbor = (row + sample_count + *delta - 2) % sample_count;
                        let state = full_chord_flyby_path[neighbor];
                        let flat = neighbor * state_count + state;
                        let (outside, inside) = measurements[flat];
                        (offsets[state] - target).abs() <= 2.1 * steer_step
                            && outside > inside + 14.0
                            && opposite_sclera_scores[flat] >= 0.10
                    })
                    .count();
                coherent >= 3 && (target - offsets[first_path[row]]).abs() >= 2.0 * steer_step
            })
            .collect::<Vec<_>>();
        let first_offsets_mm = first_path
            .iter()
            .map(|state| offsets[*state] / pixels_per_mm)
            .collect::<Vec<_>>();
        let affine_weights = (0..sample_count)
            .map(|row| {
                let state = first_path[row];
                let lookahead = pupil_lookahead_path[row];
                let (outside, inside) = measurements[row * state_count + state];
                let transition_support = ((outside - inside) / 180.0).clamp(0.0, 1.0);
                let heading_support = ((selected_heading[row] + 0.05) / 0.55).clamp(0.0, 1.0);
                let agreement = if lookahead.abs_diff(state) <= 1 {
                    1.0
                } else {
                    0.20
                };
                let lateral_reliability = 0.45 + 0.55 * camera_normals[row].0.abs();
                lateral_reliability
                    * (0.12 + 0.43 * transition_support + 0.30 * heading_support + 0.15 * agreement)
            })
            .collect::<Vec<_>>();
        if let Some(model) = fit_limbus_affine_offset_model(&first_offsets_mm, &affine_weights) {
            affine_ellipse_first_rms_mm = Some(
                (model
                    .residual_mm
                    .iter()
                    .map(|residual| residual * residual)
                    .sum::<f64>()
                    / sample_count as f64)
                    .sqrt(),
            );
            affine_predicted_offset_mm = model.predicted_mm;
            affine_ellipse_residual_mm = model.residual_mm;
            affine_ellipse_residual_threshold_mm = Some(model.outlier_threshold_mm);
        }
        // A partial lap needs immovable pieces of road.  Without explicit
        // anchors, scattered one-row disagreements plus the approach padding
        // below can join up around nearly the entire ring; the nominally
        // partial correction then becomes a second free lap and is able to
        // choose the same lid again.  The lateral limbus is the least occluded
        // part of an eye, so retain rows where edge-only and pupil-lookahead
        // agree, the material order is correct, and the road points toward
        // the acquired pupil.  A lateral row that fails any of those checks
        // remains revisitable rather than being protected by clock angle.
        // A lateral start/end sample is not a stable anchor merely because
        // it has a bright-to-dark near edge.  The upper/lower lid can supply
        // that exact cue at the arbitrary phase-zero seam.  Require the full
        // pupilward chord to return to sclera on the far side before allowing
        // a row to stop the cyclic suspect-arc padding.  This makes rows near
        // `sample_count - 1` and row zero one continuous re-entry decision
        // instead of giving the first row a privileged, immutable status.
        const MINIMUM_DIRECT_OPPOSITE_SCLERA: f64 = 0.08;
        let stable_lateral_anchors = (0..sample_count)
            .map(|row| {
                let state = first_path[row];
                let lookahead = pupil_lookahead_path[row];
                let (outside, inside) = measurements[row * state_count + state];
                let opposite_sclera = opposite_sclera_scores[row * state_count + state];
                let heading_gain = pupil_heading_scores[row * state_count + lookahead]
                    - pupil_heading_scores[row * state_count + state];
                camera_normals[row].0.abs() >= 0.58
                    && outside > inside + 8.0
                    && opposite_sclera >= MINIMUM_DIRECT_OPPOSITE_SCLERA
                    && selected_heading[row] >= (heading_reference - 0.12).max(0.08)
                    && (lookahead.abs_diff(state) <= 1 || heading_gain < 0.025)
                    && affine_ellipse_residual_threshold_mm
                        .is_none_or(|threshold| affine_ellipse_residual_mm[row] <= threshold)
            })
            .collect::<Vec<_>>();
        for row in 0..sample_count {
            let previous = first_path[(row + sample_count - 1) % sample_count];
            let current = first_path[row];
            let next = first_path[(row + 1) % sample_count];
            let curvature =
                (offsets[previous] - 2.0 * offsets[current] + offsets[next]).abs() / pixels_per_mm;
            let (outside, inside) = measurements[row * state_count + current];
            let vertical_fork_zone = camera_normals[row].1.abs() >= 0.42;
            let bad_heading = selected_heading[row] < (heading_reference - 0.16).max(0.12);
            let wrong_material_order = outside <= inside + 8.0;
            let conspicuous_turn = curvature >= 0.22;
            let route_slope = (offsets[current] - offsets[previous]).abs() / pixels_per_mm;
            let corridor_edge = current <= 1 || current + 2 >= state_count;
            let departed_from_lateral_curve = vertical_fork_zone
                && (offsets[current] - lateral_offset_reference).abs() / pixels_per_mm >= 0.48;
            let lookahead = pupil_lookahead_path[row];
            let heading_gain = pupil_heading_scores[row * state_count + lookahead]
                - pupil_heading_scores[row * state_count + current];
            let combined_gain = local_scores[row * state_count + lookahead]
                + 0.92 * pupil_heading_scores[row * state_count + lookahead]
                - local_scores[row * state_count + current]
                - 0.92 * pupil_heading_scores[row * state_count + current];
            let pupil_branch_disagreement =
                lookahead != current && heading_gain >= 0.035 && combined_gain >= -0.10;
            let affine_ellipse_departure = affine_ellipse_residual_threshold_mm
                .is_some_and(|threshold| affine_ellipse_residual_mm[row] > threshold);
            if !stable_lateral_anchors[row]
                && ((bad_heading && (wrong_material_order || corridor_edge || conspicuous_turn))
                    || (conspicuous_turn && selected_heading[row] < heading_reference - 0.07)
                    || (departed_from_lateral_curve
                        && (route_slope >= 0.09 || wrong_material_order || bad_heading))
                    || pupil_branch_disagreement
                    || affine_ellipse_departure)
            {
                revisited_rows[row] = true;
            }
        }
        // Include enough approach on both sides of a suspect fork for the
        // second drive to choose a different branch without a discontinuity.
        let suspect = revisited_rows.clone();
        for (row, revisit) in suspect.into_iter().enumerate() {
            if !revisit {
                continue;
            }
            for delta in 0..=3 {
                let forward = (row + delta) % sample_count;
                let backward = (row + sample_count - delta) % sample_count;
                if !stable_lateral_anchors[forward] {
                    revisited_rows[forward] = true;
                }
                if !stable_lateral_anchors[backward] {
                    revisited_rows[backward] = true;
                }
            }
        }
        // The fixed three-row approach above is enough for most forks, but a
        // storage seam can land just beyond it.  Continue from the resulting
        // arc endpoints through any still-unproven rows until direct anatomy
        // supplies a real re-entry.  At 96 perimeter samples the eight-row
        // ceiling is 30 degrees: enough to cross the seam without turning a
        // partial correction into an unrestricted second lap.
        let maximum_reentry_extension = (sample_count / 12).clamp(3, 12);
        cyclic_reentry_rows = extend_cyclic_revisit_to_direct_reentry(
            &mut revisited_rows,
            &stable_lateral_anchors,
            maximum_reentry_extension,
        );
        // Keep the anchors locked even when two padded suspect arcs approach
        // them from opposite directions.  This is what makes the operation a
        // bounded arc repair rather than a second global contour search.
        for (revisit, anchored) in revisited_rows.iter_mut().zip(&stable_lateral_anchors) {
            if *anchored {
                *revisit = false;
            }
        }
        // Once the complete circuit identifies a doubtful approach/re-entry,
        // refit the projected-circle displacement without allowing that arc
        // to vote.  The original robust fit can still be captured by a long,
        // smooth lid edge; zero-weighting the now-suspect rows gives the
        // independently measured remainder of the ring a chance to predict
        // where the limbus should cross the arbitrary phase-zero seam.
        let measured_only_reentry_weights = affine_weights
            .iter()
            .enumerate()
            .map(|(row, weight)| if revisited_rows[row] { 0.0 } else { *weight })
            .collect::<Vec<_>>();
        let reentry_predicted_offset_mm =
            fit_limbus_affine_offset_model(&first_offsets_mm, &measured_only_reentry_weights)
                .map_or_else(
                    || affine_predicted_offset_mm.clone(),
                    |model| model.predicted_mm,
                );
        let revisit_count = revisited_rows.iter().filter(|revisit| **revisit).count();
        if revisit_count >= 3 {
            let mut second_scores = local_scores.clone();
            for row in 0..sample_count {
                let smooth_target = if revisited_rows[row] {
                    let mut backward = 1usize;
                    while backward < sample_count
                        && revisited_rows[(row + sample_count - backward) % sample_count]
                    {
                        backward += 1;
                    }
                    let mut forward = 1usize;
                    while forward < sample_count && revisited_rows[(row + forward) % sample_count] {
                        forward += 1;
                    }
                    if backward < sample_count && forward < sample_count {
                        let before =
                            offsets[first_path[(row + sample_count - backward) % sample_count]];
                        let after = offsets[first_path[(row + forward) % sample_count]];
                        let phase = backward as f64 / (backward + forward) as f64;
                        let endpoint_curve = before * (1.0 - phase) + after * phase;
                        let affine_target = reentry_predicted_offset_mm[row] * pixels_per_mm;
                        let affine_departure = affine_ellipse_residual_threshold_mm
                            .is_some_and(|threshold| affine_ellipse_residual_mm[row] > threshold)
                            || cyclic_reentry_rows[row];
                        if full_chord_flyby_rows[row] {
                            // The original conic and endpoint interpolation
                            // can both inherit the off-eye seam. Once three
                            // neighboring orbital fly-bys agree on a complete
                            // chord, use that consensus even when the more
                            // conservative global challenger stayed on the
                            // original road.
                            0.68 * full_chord_flyby_smoothed_offset[row]
                                + 0.15 * offsets[pupil_lookahead_path[row]]
                                + 0.10 * endpoint_curve
                                + 0.07 * affine_target
                        } else if pupil_lookahead_path[row] != first_path[row] {
                            // At an actual fork, steer primarily toward the
                            // future path that enters iris/pupil. The retained
                            // endpoint and lateral anchors still make this a
                            // partial correction rather than a new free loop.
                            if affine_departure {
                                0.45 * offsets[pupil_lookahead_path[row]]
                                    + 0.10 * endpoint_curve
                                    + 0.15 * lateral_offset_reference
                                    + 0.30 * affine_target
                            } else {
                                0.60 * offsets[pupil_lookahead_path[row]]
                                    + 0.15 * endpoint_curve
                                    + 0.25 * lateral_offset_reference
                            }
                        } else {
                            // A lid branch may remain smooth for many rows.
                            // Keep the second lap tied to the directly visible
                            // lateral limbus anchors as well as the suspect arc
                            // endpoints.
                            if affine_departure {
                                0.20 * endpoint_curve
                                    + 0.30 * lateral_offset_reference
                                    + 0.50 * affine_target
                            } else {
                                0.35 * endpoint_curve + 0.65 * lateral_offset_reference
                            }
                        }
                    } else {
                        let affine_target = reentry_predicted_offset_mm[row] * pixels_per_mm;
                        if full_chord_flyby_rows[row] {
                            0.72 * full_chord_flyby_smoothed_offset[row]
                                + 0.18 * offsets[pupil_lookahead_path[row]]
                                + 0.10 * affine_target
                        } else if pupil_lookahead_path[row] != first_path[row] {
                            let affine_departure =
                                affine_ellipse_residual_threshold_mm.is_some_and(|threshold| {
                                    affine_ellipse_residual_mm[row] > threshold
                                }) || cyclic_reentry_rows[row];
                            if affine_departure {
                                0.55 * offsets[pupil_lookahead_path[row]]
                                    + 0.15 * lateral_offset_reference
                                    + 0.30 * affine_target
                            } else {
                                0.70 * offsets[pupil_lookahead_path[row]]
                                    + 0.30 * lateral_offset_reference
                            }
                        } else {
                            if affine_ellipse_residual_threshold_mm.is_some_and(|threshold| {
                                affine_ellipse_residual_mm[row] > threshold
                            }) {
                                0.45 * lateral_offset_reference
                                    + 0.55 * reentry_predicted_offset_mm[row] * pixels_per_mm
                            } else {
                                lateral_offset_reference
                            }
                        }
                    }
                } else {
                    offsets[first_path[row]]
                };
                for state in 0..state_count {
                    let flat = row * state_count + state;
                    if revisited_rows[row] {
                        let geometric_departure = (offsets[state] - smooth_target) / pixels_per_mm;
                        let lookahead_disagrees = pupil_lookahead_path[row] != first_path[row];
                        let affine_departure = affine_ellipse_residual_threshold_mm
                            .is_some_and(|threshold| affine_ellipse_residual_mm[row] > threshold)
                            || cyclic_reentry_rows[row]
                            || full_chord_challenger_rows[row]
                            || full_chord_flyby_rows[row];
                        let heading_weight = if lookahead_disagrees { 0.92 } else { 0.72 };
                        let geometric_weight = if affine_departure {
                            1.85
                        } else if lookahead_disagrees {
                            0.90
                        } else {
                            1.35
                        };
                        second_scores[flat] += heading_weight * pupil_heading_scores[flat]
                            - geometric_weight * geometric_departure.powi(2);
                    } else if state != first_path[row] {
                        second_scores[flat] = f64::NEG_INFINITY;
                    }
                }
            }
            if let Some(second_path) =
                closed_limbus_lookahead_path(&second_scores, sample_count, &offsets, pixels_per_mm)
            {
                // Do not let a nominal repair turn farther onto a lid.  The
                // rows classified above as departures are the portion of the
                // first road that cannot belong to the single ellipse formed
                // by a projected circular limbus.  Accept the partial second
                // lap only when it preserves or reduces their error against
                // that same, retrospectively fitted ellipse.  Rows outside
                // the suspect arc are deliberately excluded: they may move
                // slightly to maintain a closed, pupil-heading path.
                let mut best_second_path = second_path;
                let mut best_second_was_reinforced = false;
                let mut affine_repair_is_non_worsening = true;
                if let Some(threshold) = affine_ellipse_residual_threshold_mm {
                    let departure_rows = affine_ellipse_residual_mm
                        .iter()
                        .enumerate()
                        .filter_map(|(row, residual)| (*residual > threshold).then_some(row))
                        .collect::<Vec<_>>();
                    let departure_squared_error = |candidate: &[usize]| {
                        departure_rows
                            .iter()
                            .map(|row| {
                                let residual = offsets[candidate[*row]] / pixels_per_mm
                                    - affine_predicted_offset_mm[*row];
                                residual * residual
                            })
                            .sum::<f64>()
                    };
                    let first_squared_error = departure_rows
                        .iter()
                        .map(|row| affine_ellipse_residual_mm[*row].powi(2))
                        .sum::<f64>();
                    let mut best_squared_error = departure_squared_error(&best_second_path);

                    // A smooth lid can survive the ordinary pupil-headed pass
                    // when its local edge is exceptionally clean.  If the
                    // resulting suspect arc still lies outside the model's
                    // own outlier band, make one reinforced solve over that
                    // exact same bounded arc.  This is not another free lap:
                    // all trusted rows remain locked and the acquired pupil
                    // is unchanged.  The extra term only asks each original
                    // affine-departure row to return to the common projected
                    // ellipse instead of following the lid around again.
                    if !departure_rows.is_empty()
                        && best_squared_error > departure_rows.len() as f64 * threshold.powi(2)
                    {
                        let mut reinforced_scores = second_scores.clone();
                        for row in &departure_rows {
                            for state in 0..state_count {
                                let affine_residual = offsets[state] / pixels_per_mm
                                    - affine_predicted_offset_mm[*row];
                                reinforced_scores[*row * state_count + state] -=
                                    2.80 * affine_residual.powi(2);
                            }
                        }
                        if let Some(reinforced_path) = closed_limbus_lookahead_path(
                            &reinforced_scores,
                            sample_count,
                            &offsets,
                            pixels_per_mm,
                        ) {
                            let reinforced_squared_error =
                                departure_squared_error(&reinforced_path);
                            if reinforced_squared_error < best_squared_error {
                                best_second_path = reinforced_path;
                                best_squared_error = reinforced_squared_error;
                                best_second_was_reinforced = true;
                            }
                        }
                    }
                    affine_repair_is_non_worsening = departure_rows.is_empty()
                        || best_squared_error <= first_squared_error + 1.0e-12;
                }
                if affine_repair_is_non_worsening {
                    path = best_second_path;
                    affine_reinforced = best_second_was_reinforced;
                    lap_count = 2;
                }
            }
        }
    }
    let second_lap_path = path.clone();
    let second_lap_revisited_rows = revisited_rows.clone();
    // Recompute the same lateral reference used by the partial lap from the
    // final path so a corrected rejoin is allowed to prove itself. A median
    // over the lateral rows remains independent of the lower occluder being
    // classified. Keep it for the post-closure image-confirmation pass too.
    let lower_heading_reference = {
        let mut headings = (0..sample_count)
            .filter(|row| camera_normals[*row].0.abs() >= 0.58)
            .map(|row| pupil_heading_scores[row * state_count + path[row]])
            .collect::<Vec<_>>();
        headings.sort_by(f64::total_cmp);
        headings.get(headings.len() / 2).copied().unwrap_or(0.16)
    };
    let lower_occlusion = if pupil_center.is_some() && maximum_laps >= 2 {
        lower_limbus_occlusion_bridge(
            search,
            &path,
            &offsets,
            &measurements,
            &pupil_heading_scores,
            &opposite_sclera_scores,
            lower_heading_reference,
        )
    } else {
        LimbusLowerOcclusionBridge {
            inferred_rows: vec![false; sample_count],
            rejoined: None,
            anchor_rows: None,
        }
    };
    let mut conic_closure_before_rms_mm = None;
    let mut conic_closure_after_rms_mm = None;
    if lower_occlusion.rejoined == Some(true) {
        // Once both lateral rejoins have proved which lower rows are hidden,
        // make a third, conic-closure lap using only directly measured rows.
        // The first affine model necessarily predates the occlusion mask and
        // therefore gives even a robustly downweighted lid some leverage.
        // Zero-weighting the projected interval here lets the visible arcs
        // determine the continuation behind the lid without pretending that
        // another pass over flesh supplied new limbus evidence.
        let measured_offsets_mm = path
            .iter()
            .map(|state| offsets[*state] / pixels_per_mm)
            .collect::<Vec<_>>();
        let measured_weights = (0..sample_count)
            .map(|row| {
                if lower_occlusion.inferred_rows[row] {
                    return 0.0;
                }
                let state = path[row];
                let flat = row * state_count + state;
                let (outside, inside) = measurements[flat];
                let transition = ((outside - inside) / 180.0).clamp(0.0, 1.0);
                let heading = ((pupil_heading_scores[flat] + 0.05) / 0.55).clamp(0.0, 1.0);
                let opposite = ((opposite_sclera_scores[flat] + 0.02) / 0.25).clamp(0.0, 1.0);
                let lateral_reliability = 0.55 + 0.45 * camera_normals[row].0.abs();
                lateral_reliability * (0.10 + 0.38 * transition + 0.27 * heading + 0.25 * opposite)
            })
            .collect::<Vec<_>>();
        let closure_model = (maximum_laps >= 3)
            .then(|| fit_limbus_affine_offset_model(&measured_offsets_mm, &measured_weights))
            .flatten();
        for row in 0..sample_count {
            if !lower_occlusion.inferred_rows[row] {
                continue;
            }
            let projected_mm =
                closure_model
                    .as_ref()
                    .map_or(affine_predicted_offset_mm[row], |model| {
                        // This is still a local refinement of the already
                        // admitted conic, not an unbounded extrapolation from a
                        // partial arc. About 0.4 nominal mm is four RAW pixels at
                        // the present scale and is enough to remove residual lid
                        // leverage without inventing a different-sized eye.
                        affine_predicted_offset_mm[row]
                            + (model.predicted_mm[row] - affine_predicted_offset_mm[row])
                                .clamp(-0.40, 0.40)
                    });
            let projected_offset = projected_mm * pixels_per_mm;
            path[row] = offsets
                .iter()
                .enumerate()
                .min_by(|left, right| {
                    (left.1 - projected_offset)
                        .abs()
                        .total_cmp(&(right.1 - projected_offset).abs())
                })
                .map(|(state, _)| state)
                .unwrap_or(path[row]);
            revisited_rows[row] = true;
        }
        if let Some(model) = closure_model.as_ref() {
            conic_closure_before_rms_mm = Some(
                (measured_offsets_mm
                    .iter()
                    .zip(&model.predicted_mm)
                    .map(|(observed, predicted)| (observed - predicted).powi(2))
                    .sum::<f64>()
                    / sample_count as f64)
                    .sqrt(),
            );
            conic_closure_after_rms_mm = Some(
                (path
                    .iter()
                    .enumerate()
                    .map(|(row, state)| {
                        (offsets[*state] / pixels_per_mm - model.predicted_mm[row]).powi(2)
                    })
                    .sum::<f64>()
                    / sample_count as f64)
                    .sqrt(),
            );
        }
        lap_count = lap_count.max(if closure_model.is_some() { 3 } else { 2 });
    }
    let conic_projected_rows = lower_occlusion.inferred_rows.clone();
    let reconfirmed_rows = if lower_occlusion.rejoined == Some(true) {
        conic_projected_limbus_reconfirmations(
            &conic_projected_rows,
            &path,
            &offsets,
            &measurements,
            &pupil_heading_scores,
            lower_heading_reference,
        )
    } else {
        vec![false; sample_count]
    };
    let inferred_rows = conic_projected_rows
        .iter()
        .zip(&reconfirmed_rows)
        .map(|(projected, reconfirmed)| *projected && !*reconfirmed)
        .collect::<Vec<_>>();
    let affine_ellipse_final_residual_mm = path
        .iter()
        .enumerate()
        .map(|(row, state)| {
            (offsets[*state] / pixels_per_mm - affine_predicted_offset_mm[row]).abs()
        })
        .collect::<Vec<_>>();
    let affine_ellipse_final_rms_mm = affine_ellipse_residual_threshold_mm.map(|_| {
        (affine_ellipse_final_residual_mm
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>()
            / sample_count as f64)
            .sqrt()
    });
    let departure_indices =
        affine_ellipse_residual_threshold_mm.map_or_else(Vec::new, |threshold| {
            affine_ellipse_residual_mm
                .iter()
                .enumerate()
                .filter_map(|(row, residual)| (*residual > threshold).then_some(row))
                .collect::<Vec<_>>()
        });
    let departure_rms = |residuals: &[f64]| {
        (!departure_indices.is_empty()).then(|| {
            (departure_indices
                .iter()
                .map(|row| residuals[*row] * residuals[*row])
                .sum::<f64>()
                / departure_indices.len() as f64)
                .sqrt()
        })
    };
    let affine_ellipse_departure_first_rms_mm = departure_rms(&affine_ellipse_residual_mm);
    let affine_ellipse_departure_final_rms_mm = departure_rms(&affine_ellipse_final_residual_mm);
    let mut samples = Vec::with_capacity(sample_count);
    for (index, state) in path.iter().copied().enumerate() {
        let phase = 2.0 * PI * index as f64 / sample_count as f64;
        let (base_point, normal) = search.point_and_normal(phase, 1.0);
        let offset = offsets[state];
        let (outside_luma, inside_luma) = measurements[index * state_count + state];
        let driven_point = (
            base_point.0 + normal.0 * offset,
            base_point.1 + normal.1 * offset,
        );
        samples.push(LimbusPerimeterDriveSample {
            phase,
            base_point,
            driven_point,
            outward_normal: normal,
            offset_px: offset,
            transition_score: ((outside_luma - inside_luma) / 180.0).clamp(-1.0, 1.0),
            outside_luma,
            inside_luma,
            pupil_heading_score: pupil_heading_scores[index * state_count + state],
            opposite_sclera_score: opposite_sclera_scores[index * state_count + state],
            affine_ellipse_residual_mm: affine_ellipse_residual_mm[index],
            affine_ellipse_final_residual_mm: affine_ellipse_final_residual_mm[index],
            revisited: revisited_rows[index],
            inferred_occlusion: inferred_rows[index],
        });
    }

    let strip_width = (outside_extent.ceil() as usize * 2 + 1).max(33);
    let center_column = (strip_width - 1) as f64 * 0.5;
    let column_scale = outside_extent * 2.0 / (strip_width - 1) as f64;
    let mut strip_luma = Vec::with_capacity(strip_width * sample_count);
    let first_lap_boundary_columns = first_path
        .iter()
        .map(|state| center_column - offsets[*state] / column_scale)
        .collect::<Vec<_>>();
    let second_lap_boundary_columns = second_lap_path
        .iter()
        .map(|state| center_column - offsets[*state] / column_scale)
        .collect::<Vec<_>>();
    let mut boundary_columns = Vec::with_capacity(sample_count);
    for sample in &samples {
        for column in 0..strip_width {
            let signed_distance = outside_extent - column as f64 * column_scale;
            let value = luma.sample3(
                sample.base_point.0 + sample.outward_normal.0 * signed_distance,
                sample.base_point.1 + sample.outward_normal.1 * signed_distance,
            );
            strip_luma.push(value.round().clamp(0.0, 1023.0) as u16);
        }
        boundary_columns.push(center_column - sample.offset_px / column_scale);
    }
    let guide_left_column = (center_column - guide_half_width / column_scale)
        .round()
        .clamp(0.0, (strip_width - 1) as f64) as usize;
    let guide_right_column = (center_column + guide_half_width / column_scale)
        .round()
        .clamp(0.0, (strip_width - 1) as f64) as usize;
    Some(LimbusPerimeterStrip {
        width: strip_width,
        height: sample_count,
        luma: strip_luma,
        nominal_mm_per_side,
        pixels_per_mm,
        guide_left_column,
        guide_right_column,
        affine_ellipse_residual_threshold_mm,
        affine_ellipse_first_rms_mm,
        affine_ellipse_final_rms_mm,
        affine_ellipse_departure_first_rms_mm,
        affine_ellipse_departure_final_rms_mm,
        conic_closure_before_rms_mm,
        conic_closure_after_rms_mm,
        affine_reinforced,
        lap_count,
        second_lap_revisited_rows,
        revisited_rows,
        cyclic_reentry_rows,
        full_chord_flyby_rows,
        conic_projected_rows,
        reconfirmed_rows,
        inferred_rows,
        lower_occlusion_rejoined: lower_occlusion.rejoined,
        lower_occlusion_anchor_rows: lower_occlusion.anchor_rows,
        first_lap_boundary_columns,
        second_lap_boundary_columns,
        boundary_columns,
        samples,
    })
}

#[derive(Clone, Copy, Debug)]
struct IrisScleraAppearance {
    iris_log_rg: f64,
    iris_log_bg: f64,
    sclera_log_rg: f64,
    sclera_log_bg: f64,
    color_scale: f64,
    luma_midpoint: f64,
    luma_scale: f64,
}

impl IrisScleraAppearance {
    /// A material-only posterior.  It deliberately excludes brightness so a
    /// cast eyelid shadow does not turn sclera into iris (or skin into sclera)
    /// before the illumination model has a chance to explain that shadow.
    fn color_sclera_probability(self, log_rg: f64, log_bg: f64) -> f64 {
        let iris_distance = (log_rg - self.iris_log_rg).hypot(log_bg - self.iris_log_bg);
        let sclera_distance = (log_rg - self.sclera_log_rg).hypot(log_bg - self.sclera_log_bg);
        let color_margin =
            ((iris_distance - sclera_distance) / self.color_scale).clamp(-12.0, 12.0);
        1.0 / (1.0 + (-color_margin).exp())
    }

    fn sclera_probability(self, log_rg: f64, log_bg: f64, intensity: f64) -> f64 {
        let iris_distance = (log_rg - self.iris_log_rg).hypot(log_bg - self.iris_log_bg);
        let sclera_distance = (log_rg - self.sclera_log_rg).hypot(log_bg - self.sclera_log_bg);
        let color_margin = (iris_distance - sclera_distance) / self.color_scale;
        let luma_margin = 0.72 * (intensity - self.luma_midpoint) / self.luma_scale;
        let logit = (color_margin + luma_margin).clamp(-12.0, 12.0);
        1.0 / (1.0 + (-logit).exp())
    }
}

/// A deliberately low-frequency illumination reconstruction.  The model is
/// fitted only from material-confident iris and *lateral* sclera cells.  It
/// jointly solves a single smooth log-light surface and an iris/sclera
/// reflectance offset, so the material edge itself cannot be blurred into the
/// lighting field as it would be by an ordinary image blur.
#[derive(Clone, Copy, Debug)]
struct MaterialIlluminationModel {
    center: (f64, f64),
    radius: f64,
    coefficients: [f64; 6],
    sample_count: usize,
    iris_sample_count: usize,
    sclera_sample_count: usize,
    lateral_sclera_balance: f64,
    residual_median: f64,
    inlier_fraction: f64,
    light_span: f64,
}

/// Offline diagnostics for the deliberately vague material/light fit.  These
/// values describe only the fit's own support—they are not hand-label-derived
/// and are intended to gate experimental use of illumination evidence.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialIlluminationDiagnostics {
    pub sample_count: usize,
    pub iris_sample_count: usize,
    pub sclera_sample_count: usize,
    /// `1` means balanced lateral sclera support; `0` means one side was absent.
    pub lateral_sclera_balance: f64,
    pub residual_median: f64,
    pub inlier_fraction: f64,
    /// Log-light range across the eye-sized normalized support.
    pub light_span: f64,
}

impl From<MaterialIlluminationModel> for MaterialIlluminationDiagnostics {
    fn from(model: MaterialIlluminationModel) -> Self {
        Self {
            sample_count: model.sample_count,
            iris_sample_count: model.iris_sample_count,
            sclera_sample_count: model.sclera_sample_count,
            lateral_sclera_balance: model.lateral_sclera_balance,
            residual_median: model.residual_median,
            inlier_fraction: model.inlier_fraction,
            light_span: model.light_span,
        }
    }
}

/// Short-horizon history for the vague lighting surface. Coefficients are
/// represented in the current eye's normalized coordinates, so ROI movement
/// and modest pupil/iris translation do not make the prior unusable. Only the
/// low-frequency light coefficients are retained—never a material class,
/// contact location, or ellipse—so this history cannot reinforce a prior edge.
#[derive(Clone, Debug, Default)]
pub struct MaterialIlluminationTracker {
    coefficient_history: Vec<[f64; 6]>,
}

impl MaterialIlluminationTracker {
    fn blend_prior(&self, current: MaterialIlluminationModel) -> MaterialIlluminationModel {
        if self.coefficient_history.is_empty() {
            return current;
        }
        let mut coefficients = current.coefficients;
        for index in 0..coefficients.len() {
            let mut history = self
                .coefficient_history
                .iter()
                .map(|coefficients| coefficients[index])
                .collect::<Vec<_>>();
            let prior = median(&mut history);
            // Constant exposure can change between frames and does not affect
            // a local edge, so it receives very little temporal pull.  The
            // slope/curvature is the actual vague scene-light hypothesis.
            let pull = if index == 0 {
                0.08
            } else if (coefficients[index] - prior).abs() <= 0.28 {
                0.38
            } else {
                0.12
            };
            coefficients[index] = coefficients[index] * (1.0 - pull) + prior * pull;
        }
        MaterialIlluminationModel {
            coefficients,
            ..current
        }
    }

    fn observe(&mut self, model: MaterialIlluminationModel) {
        self.coefficient_history.push(model.coefficients);
        const MAX_HISTORY: usize = 8;
        if self.coefficient_history.len() > MAX_HISTORY {
            self.coefficient_history.remove(0);
        }
    }
}

impl MaterialIlluminationModel {
    fn light_at(self, x: f64, y: f64) -> f64 {
        let radius = self.radius.max(1.0);
        let u = ((x - self.center.0) / radius).clamp(-2.0, 2.0);
        let v = ((y - self.center.1) / radius).clamp(-2.0, 2.0);
        let terms = [1.0, u, v, u * u, u * v, v * v];
        self.coefficients
            .into_iter()
            .zip(terms)
            .map(|(coefficient, term)| coefficient * term)
            .sum()
    }

    fn corrected_intensity(self, native: &NativeLogPlane, x: f64, y: f64) -> f64 {
        native.sample_log_intensity(x, y) - self.light_at(x, y)
    }
}

#[derive(Clone, Copy)]
struct MaterialLightSample {
    u: f64,
    v: f64,
    log_intensity: f64,
    sclera: f64,
    weight: f64,
}

fn solve_material_light_normal_equations(mut matrix: [[f64; 8]; 7]) -> Option<[f64; 7]> {
    for column in 0..7 {
        let pivot = (column..7).max_by(|left, right| {
            matrix[*left][column]
                .abs()
                .total_cmp(&matrix[*right][column].abs())
        })?;
        if matrix[pivot][column].abs() < 1.0e-9 {
            return None;
        }
        matrix.swap(column, pivot);
        let divisor = matrix[column][column];
        for value in column..=7 {
            matrix[column][value] /= divisor;
        }
        for row in 0..7 {
            if row == column {
                continue;
            }
            let factor = matrix[row][column];
            for value in column..=7 {
                matrix[row][value] -= factor * matrix[column][value];
            }
        }
    }
    Some(std::array::from_fn(|index| matrix[index][7]))
}

fn fit_material_illumination(
    native: &NativeLogPlane,
    appearance: IrisScleraAppearance,
    search: OuterSearchEllipse,
) -> Option<MaterialIlluminationModel> {
    let radius = search.equivalent_radius();
    if radius < 12.0 {
        return None;
    }
    let mut samples = Vec::new();
    let mut iris_count = 0usize;
    let mut sclera_count = 0usize;
    let mut sclera_left = 0usize;
    let mut sclera_right = 0usize;
    for grid_y in 0..native.height {
        for grid_x in 0..native.width {
            let index = grid_y * native.width + grid_x;
            let x = native.origin_x + grid_x as f64 * 4.0;
            let y = native.origin_y + grid_y as f64 * 4.0;
            let (local_x, local_y) = search.normalized_coordinates(x, y);
            let rho = local_x.hypot(local_y);
            let color_sclera =
                appearance.color_sclera_probability(native.log_rg[index], native.log_bg[index]);
            let (sclera, weight) = if (0.48..=0.84).contains(&rho) && color_sclera <= 0.32 {
                iris_count += 1;
                (0.0, (1.0 - color_sclera).powi(2))
            } else if (1.04..=1.38).contains(&rho)
                && local_x.abs() >= local_y.abs() * 0.72
                && color_sclera >= 0.68
            {
                sclera_count += 1;
                if local_x < 0.0 {
                    sclera_left += 1;
                } else {
                    sclera_right += 1;
                }
                (1.0, color_sclera.powi(2))
            } else {
                continue;
            };
            samples.push(MaterialLightSample {
                u: (x - search.center.0) / radius,
                v: (y - search.center.1) / radius,
                log_intensity: (native.intensity[index] + 8.0).ln(),
                sclera,
                weight,
            });
        }
    }
    if iris_count < 12 || sclera_count < 12 || samples.len() < 32 {
        return None;
    }
    let mut robust = vec![1.0; samples.len()];
    let mut solution = None;
    for _ in 0..3 {
        let mut matrix = [[0.0; 8]; 7];
        for (sample, robust_weight) in samples.iter().zip(&robust) {
            let terms = [
                1.0,
                sample.u,
                sample.v,
                sample.u * sample.u,
                sample.u * sample.v,
                sample.v * sample.v,
                sample.sclera,
            ];
            let weight = sample.weight * robust_weight;
            for row in 0..7 {
                for column in 0..7 {
                    matrix[row][column] += weight * terms[row] * terms[column];
                }
                matrix[row][7] += weight * terms[row] * sample.log_intensity;
            }
        }
        // Keep the reconstruction truly vague.  Linear falloff is allowed;
        // quadratic bend is a weak option rather than a way to fit texture,
        // glints, or a limbus itself.
        for (index, ridge) in [0.001, 0.030, 0.030, 0.120, 0.120, 0.120, 0.008]
            .into_iter()
            .enumerate()
        {
            matrix[index][index] += ridge;
        }
        let fitted = solve_material_light_normal_equations(matrix)?;
        for (robust_weight, sample) in robust.iter_mut().zip(&samples) {
            let prediction = fitted[0]
                + fitted[1] * sample.u
                + fitted[2] * sample.v
                + fitted[3] * sample.u * sample.u
                + fitted[4] * sample.u * sample.v
                + fitted[5] * sample.v * sample.v
                + fitted[6] * sample.sclera;
            let residual = (sample.log_intensity - prediction).abs();
            // Huber-like rejection of glints, eyelashes, and mislabeled skin.
            *robust_weight = (0.16 / residual.max(0.16)).clamp(0.04, 1.0);
        }
        solution = Some(fitted);
    }
    let solution = solution?;
    let residuals = samples
        .iter()
        .map(|sample| {
            (sample.log_intensity
                - (solution[0]
                    + solution[1] * sample.u
                    + solution[2] * sample.v
                    + solution[3] * sample.u * sample.u
                    + solution[4] * sample.u * sample.v
                    + solution[5] * sample.v * sample.v
                    + solution[6] * sample.sclera))
                .abs()
        })
        .collect::<Vec<_>>();
    let mut residual_median_values = residuals.clone();
    let residual_median = median(&mut residual_median_values);
    let inlier_fraction = residuals
        .iter()
        .filter(|residual| **residual <= 0.16)
        .count() as f64
        / residuals.len().max(1) as f64;
    let coefficients: [f64; 6] = solution[..6].try_into().ok()?;
    let mut light_values = [-1.0, 0.0, 1.0]
        .into_iter()
        .flat_map(|u| {
            [-1.0, 0.0, 1.0].into_iter().map(move |v| {
                coefficients[0]
                    + coefficients[1] * u
                    + coefficients[2] * v
                    + coefficients[3] * u * u
                    + coefficients[4] * u * v
                    + coefficients[5] * v * v
            })
        })
        .collect::<Vec<_>>();
    let light_low = percentile_f64(&mut light_values, 0.0);
    let light_high = percentile_f64(&mut light_values, 1.0);
    Some(MaterialIlluminationModel {
        center: search.center,
        radius,
        coefficients,
        sample_count: samples.len(),
        iris_sample_count: iris_count,
        sclera_sample_count: sclera_count,
        lateral_sclera_balance: sclera_left.min(sclera_right) as f64
            / sclera_left.max(sclera_right).max(1) as f64,
        residual_median,
        inlier_fraction,
        light_span: light_high - light_low,
    })
}

/// Inspect illumination-model support without tracing an iris boundary.  This
/// is intentionally offline/debug-only so live tracking behavior remains
/// unchanged while its reliability gate is being validated.
pub fn debug_material_illumination_diagnostics(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
) -> Option<MaterialIlluminationDiagnostics> {
    if raw.len() < width * height || coarse.radius <= 1.0 {
        return None;
    }
    let search = OuterSearchEllipse::from_coarse(coarse);
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)?;
    let native = blur_outer_appearance(native);
    let appearance = estimate_iris_sclera_appearance(&native, search)?;
    let model = fit_material_illumination(&native, appearance, search)?;
    Some(model.into())
}

/// Whether a material/light reconstruction has enough bilateral support to
/// safely influence an outer-limbus fit.  A smooth light field can legitimately
/// contain a shadow, but it must not infer an extreme whole-eye gradient from
/// almost entirely one-sided sclera samples.  This gate is deliberately
/// conservative and gates both the live analog force and offline ablations.
pub fn debug_material_illumination_is_reliable(
    diagnostics: MaterialIlluminationDiagnostics,
) -> bool {
    diagnostics.sample_count >= 64
        && diagnostics.iris_sample_count >= 24
        && diagnostics.sclera_sample_count >= 24
        && diagnostics.residual_median.is_finite()
        && diagnostics.inlier_fraction >= 0.45
        // A log-light span of 0.90 is a 2.46x predicted exposure change
        // across the eye.  Permit it only when both lateral sclera regions
        // actually constrain the slope instead of letting one side
        // extrapolate through the entire iris.
        && !(diagnostics.light_span > 0.90 && diagnostics.lateral_sclera_balance < 0.45)
}

fn material_illumination_is_reliable(model: MaterialIlluminationModel) -> bool {
    debug_material_illumination_is_reliable(model.into())
}

/// A continuous reliability rather than a second accept/reject decision. The
/// hard gate above decides whether a lighting surface may be used at all;
/// this value controls how certain a local analog edge may become.
fn material_illumination_confidence(model: MaterialIlluminationModel) -> f64 {
    let support = (model.sample_count as f64 / 160.0).clamp(0.0, 1.0);
    let class_support =
        (model.iris_sample_count.min(model.sclera_sample_count) as f64 / 48.0).clamp(0.0, 1.0);
    let balance = model.lateral_sclera_balance.clamp(0.0, 1.0);
    let residual = (1.0 - model.residual_median / 0.24).clamp(0.0, 1.0);
    let inliers = ((model.inlier_fraction - 0.35) / 0.50).clamp(0.0, 1.0);
    (0.16
        + 0.20 * support
        + 0.20 * class_support
        + 0.16 * balance
        + 0.14 * residual
        + 0.14 * inliers)
        .clamp(0.0, 1.0)
}

fn iris_sclera_probability_map(
    native: &NativeLogPlane,
    appearance: IrisScleraAppearance,
) -> Vec<f64> {
    native
        .log_rg
        .iter()
        .zip(&native.log_bg)
        .zip(&native.intensity)
        .map(|((log_rg, log_bg), intensity)| {
            appearance.sclera_probability(*log_rg, *log_bg, *intensity)
        })
        .collect()
}

/// Sclera-vs-iris material posterior with the intensity term deliberately
/// omitted. `log_rg` and `log_bg` are centered log-chromatic coordinates, so
/// this map survives an achromatic multiplicative shadow cast by an eyelid.
/// The existing mixed probability remains useful independent evidence; this
/// map is an additional bounded vote rather than a replacement.
fn iris_sclera_reflectance_probability_map(
    native: &NativeLogPlane,
    appearance: IrisScleraAppearance,
) -> Vec<f64> {
    native
        .reflectance_log_rg
        .iter()
        .zip(&native.reflectance_log_bg)
        .map(|(log_rg, log_bg)| appearance.color_sclera_probability(*log_rg, *log_bg))
        .collect()
}

fn estimate_iris_sclera_appearance(
    native: &NativeLogPlane,
    search: OuterSearchEllipse,
) -> Option<IrisScleraAppearance> {
    estimate_iris_sclera_appearance_from_chroma(
        native,
        search,
        &native.log_rg,
        &native.log_bg,
        false,
    )
}

fn estimate_iris_sclera_reflectance_appearance(
    native: &NativeLogPlane,
    search: OuterSearchEllipse,
) -> Option<IrisScleraAppearance> {
    estimate_iris_sclera_appearance_from_chroma(
        native,
        search,
        &native.reflectance_log_rg,
        &native.reflectance_log_bg,
        true,
    )
}

fn estimate_iris_sclera_appearance_from_chroma(
    native: &NativeLogPlane,
    search: OuterSearchEllipse,
    log_rg: &[f64],
    log_bg: &[f64],
    intensity_independent: bool,
) -> Option<IrisScleraAppearance> {
    if log_rg.len() != native.width * native.height || log_bg.len() != log_rg.len() {
        return None;
    }
    let mut iris = Vec::new();
    let mut sclera = Vec::new();
    for grid_y in 0..native.height {
        for grid_x in 0..native.width {
            let index = grid_y * native.width + grid_x;
            let x = native.origin_x + grid_x as f64 * 4.0;
            let y = native.origin_y + grid_y as f64 * 4.0;
            let (local_x, local_y) = search.normalized_coordinates(x, y);
            let rho = local_x.hypot(local_y);
            let sample = (log_rg[index], log_bg[index], native.intensity[index]);
            if (0.54..=0.88).contains(&rho) {
                iris.push(sample);
            } else if (1.02..=1.48).contains(&rho) && local_x.abs() >= local_y.abs() * 0.75 {
                sclera.push(sample);
            }
        }
    }
    if iris.len() < 24 || sclera.len() < 24 {
        return None;
    }
    // The established mixed posterior deliberately keeps only the brighter
    // half of its lateral sclera candidates. The intrinsic posterior must not
    // do that: changing an achromatic light field may not change which color
    // samples define iris and sclera reflectance. Lateral sampling plus the
    // coordinate-wise median already gives robust material references.
    if !intensity_independent {
        let mut sclera_luma = sclera.iter().map(|sample| sample.2).collect::<Vec<_>>();
        let lateral_luma_median = median(&mut sclera_luma);
        sclera.retain(|sample| sample.2 >= lateral_luma_median);
    }
    if sclera.len() < 12 {
        return None;
    }
    let reference = |samples: &[(f64, f64, f64)]| {
        let mut rg = samples.iter().map(|sample| sample.0).collect::<Vec<_>>();
        let mut bg = samples.iter().map(|sample| sample.1).collect::<Vec<_>>();
        let mut luma = samples.iter().map(|sample| sample.2).collect::<Vec<_>>();
        (median(&mut rg), median(&mut bg), median(&mut luma))
    };
    let (iris_log_rg, iris_log_bg, iris_luma) = reference(&iris);
    let (sclera_log_rg, sclera_log_bg, sclera_luma) = reference(&sclera);
    let reference_separation = (sclera_log_rg - iris_log_rg).hypot(sclera_log_bg - iris_log_bg);
    Some(IrisScleraAppearance {
        iris_log_rg,
        iris_log_bg,
        sclera_log_rg,
        sclera_log_bg,
        color_scale: (reference_separation * 0.32).max(0.018),
        luma_midpoint: (iris_luma + sclera_luma) * 0.5,
        luma_scale: ((sclera_luma - iris_luma).abs() * 0.5).max(18.0),
    })
}

struct OuterRayContext {
    luma: Arc<BoxLuma5>,
    native: Option<Arc<NativeLogPlane>>,
    sclera_probability: Option<Arc<Vec<f64>>>,
    /// Intensity-independent log-chromatic material probability. This stays
    /// separate from `sclera_probability`, whose luma term remains a useful
    /// but shadow-sensitive cue.
    reflectance_sclera_probability: Option<Arc<Vec<f64>>>,
    material_illumination: Option<MaterialIlluminationModel>,
    upper_eyelid: Arc<Vec<BorderPoint>>,
    lower_eyelid: Arc<Vec<BorderPoint>>,
    luma_gate: LumaTransitionGate,
    width: usize,
    height: usize,
    search: OuterSearchEllipse,
    rough_search: OuterSearchEllipse,
    scale_range: (f64, f64),
}

#[derive(Clone, Copy, Debug)]
struct LumaTransitionGate {
    dark_maximum: f64,
    sliver_bright_minimum: f64,
    sliver_minimum_step: f64,
}

fn estimate_luma_transition_gate(luma: &BoxLuma5) -> LumaTransitionGate {
    let mut values = Vec::with_capacity((luma.width / 4) * (luma.height / 4));
    for y in (4..luma.height.saturating_sub(4)).step_by(4) {
        for x in (4..luma.width.saturating_sub(4)).step_by(4) {
            values.push(luma.integer_sample(x, y));
        }
    }
    let mut low_values = values.clone();
    let low = percentile_f64(&mut low_values, 0.08);
    let high = percentile_f64(&mut values, 0.92).max(low + 8.0);
    let span = high - low;
    LumaTransitionGate {
        dark_maximum: low + 0.40 * span,
        // This is an ordering veto, not an evidence requirement. A narrow
        // exposed-sclera strip can be too small to win the relative boundary
        // score, but it still proves that a farther transition is eyelid/skin
        // rather than limbus.
        sliver_bright_minimum: low + 0.53 * span,
        sliver_minimum_step: (0.12 * span).max(2.5),
    }
}

#[derive(Clone, Copy, Debug)]
struct OuterRayCandidate {
    score: f64,
    luma_score: f64,
    material_light_score: f64,
    reflectance_score: f64,
    intrinsic_shadow_bonus: f64,
    shadow_disagreement: f64,
    reflectance_plateau: f64,
    intrinsic_intensity_step: f64,
    meridian_profile_score: f64,
    margin_clarity: f64,
    pupil_void: f64,
    inner_limbus_step: f64,
    iris_band: f64,
    sclera_out: f64,
    far_sclera: f64,
    radius: f64,
    rough_rho: f64,
    point: OuterIrisPoint,
}

struct OuterRayJob {
    context: Arc<OuterRayContext>,
    index: usize,
    sample_stride: usize,
    reply: mpsc::Sender<OuterRayReply>,
}

struct OuterRayReply {
    index: usize,
    candidates: Vec<OuterRayCandidate>,
    elapsed: Duration,
    overrun: bool,
}

struct OuterRayBatch {
    candidates: [Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    elapsed: Duration,
    max_ray_elapsed: Duration,
    active_rays: usize,
    candidate_rays: usize,
    candidate_count: usize,
    ray_overruns: usize,
    batch_budget_overruns: usize,
}

struct OuterRayPool {
    jobs: mpsc::Sender<OuterRayJob>,
}

impl OuterRayPool {
    fn new() -> Self {
        let (jobs, receiver) = mpsc::channel::<OuterRayJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        let logical_cpus = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(2);
        let default_workers = logical_cpus.saturating_sub(4).clamp(1, 24);
        let worker_count = std::env::var("BUTTERCUP_OUTER_IRIS_WORKERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(|value| value.clamp(1, 24))
            .unwrap_or(default_workers);
        for worker in 0..worker_count {
            let receiver = Arc::clone(&receiver);
            thread::Builder::new()
                .name(format!("outer-iris-ray-{worker}"))
                .spawn(move || loop {
                    let job = receiver
                        .lock()
                        .ok()
                        .and_then(|receiver| receiver.recv().ok());
                    let Some(job) = job else {
                        return;
                    };
                    let started = Instant::now();
                    let candidates =
                        evaluate_outer_iris_ray(&job.context, job.index, job.sample_stride);
                    let elapsed = started.elapsed();
                    let _ = job.reply.send(OuterRayReply {
                        index: job.index,
                        candidates,
                        elapsed,
                        overrun: elapsed > OUTER_IRIS_RAY_BUDGET,
                    });
                })
                .expect("spawn outer-iris ray worker");
        }
        Self { jobs }
    }
}

static OUTER_RAY_POOL: OnceLock<OuterRayPool> = OnceLock::new();

fn evaluate_outer_iris_ray(
    context: &OuterRayContext,
    index: usize,
    sample_stride: usize,
) -> Vec<OuterRayCandidate> {
    let angle = outer_iris_evidence_angle(index);
    // Upper candidates are retained as weak opposing-meridian and occlusion
    // evidence. They are never authoritative fit points; throwing them away
    // here made it impossible to ask whether both halves of a projected 3D
    // meridian tell a compatible radial story.
    outer_iris_ray_candidates(context, angle, sample_stride)
}

fn evaluate_outer_iris_rays(
    context: Arc<OuterRayContext>,
    work_stride: usize,
    sample_stride: usize,
) -> OuterRayBatch {
    let started = Instant::now();
    let pool = OUTER_RAY_POOL.get_or_init(OuterRayPool::new);
    let (reply, results) = mpsc::channel();
    let work_stride = work_stride.max(1).min(4);
    let sample_stride = sample_stride.max(1).min(4);
    let active = (0..OUTER_IRIS_DENSE_EVIDENCE_SAMPLES)
        .filter(|index| {
            let angle = outer_iris_evidence_angle(*index);
            index % work_stride == 0
                || angle.sin().abs() < 0.18
                || (angle.sin() > 0.45 && angle.cos().abs() < 0.42)
                // Always keep the shadow-prone opposite of each authoritative
                // lower meridian. It is comparator/veto work, never a fit vote.
                || (angle.sin() < -0.45 && angle.cos().abs() < 0.42)
        })
        .collect::<Vec<_>>();
    let mut send_failed = false;
    for &index in &active {
        if pool
            .jobs
            .send(OuterRayJob {
                context: Arc::clone(&context),
                index,
                sample_stride,
                reply: reply.clone(),
            })
            .is_err()
        {
            send_failed = true;
            break;
        }
    }
    drop(reply);
    let mut candidates = std::array::from_fn(|_| Vec::new());
    let mut max_ray_elapsed = Duration::ZERO;
    let mut ray_overruns = 0usize;
    let mut received = 0usize;
    if send_failed {
        for &index in &active {
            let ray_started = Instant::now();
            let ray_candidates = evaluate_outer_iris_ray(&context, index, sample_stride);
            let elapsed = ray_started.elapsed();
            max_ray_elapsed = max_ray_elapsed.max(elapsed);
            ray_overruns += usize::from(elapsed > OUTER_IRIS_RAY_BUDGET);
            candidates[index] = ray_candidates;
            received += 1;
        }
    } else {
        while received < active.len() {
            let Ok(result) = results.recv() else {
                break;
            };
            max_ray_elapsed = max_ray_elapsed.max(result.elapsed);
            ray_overruns += usize::from(result.overrun);
            candidates[result.index] = result.candidates;
            received += 1;
        }
    }
    let candidate_rays = candidates.iter().filter(|ray| !ray.is_empty()).count();
    let candidate_count = candidates.iter().map(Vec::len).sum();
    let elapsed = started.elapsed();
    OuterRayBatch {
        candidates,
        elapsed,
        max_ray_elapsed,
        active_rays: active.len(),
        candidate_rays,
        candidate_count,
        ray_overruns,
        // Compatibility telemetry: all deterministic rays are complete. The
        // old "timeout" counter now records only whether the completed batch
        // exceeded its presentation budget, never missing anatomical input.
        batch_budget_overruns: usize::from(elapsed > OUTER_IRIS_RAY_BATCH_BUDGET)
            + active.len().saturating_sub(received),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OuterIrisCandidateDebug {
    pub ray_index: usize,
    pub x: f64,
    pub y: f64,
    pub fused_score: f64,
    pub luma_score: f64,
    pub material_light_score: f64,
    pub reflectance_score: f64,
    pub intrinsic_shadow_bonus: f64,
    pub shadow_disagreement: f64,
    pub reflectance_plateau: f64,
    pub intrinsic_intensity_step: f64,
    pub meridian_profile_score: f64,
    pub margin_clarity: f64,
    pub pupil_void: f64,
    pub inner_limbus_step: f64,
    pub iris_band: f64,
    pub sclera_out: f64,
    pub far_sclera: f64,
    pub rough_rho: f64,
}

/// Offline diagnostics for validating the complete limbus candidate lattice
/// against hand labels. This performs the normal broad fused search but does
/// not fit, track, or mutate detector state.
#[allow(clippy::too_many_arguments)]
pub fn debug_outer_iris_candidate_lattice(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
) -> Vec<OuterIrisCandidateDebug> {
    if raw.len() < width * height || coarse.radius <= 1.0 {
        return Vec::new();
    }
    let rough_search = OuterSearchEllipse::from_coarse(coarse);
    let luma = Arc::new(BoxLuma5::new(raw, width, height));
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)
        .map(blur_outer_appearance)
        .map(Arc::new);
    let appearance = native
        .as_deref()
        .and_then(|plane| estimate_iris_sclera_appearance(plane, rough_search));
    let reflectance_appearance = native
        .as_deref()
        .and_then(|plane| estimate_iris_sclera_reflectance_appearance(plane, rough_search));
    let material_illumination = native
        .as_deref()
        .zip(appearance)
        .and_then(|(plane, appearance)| fit_material_illumination(plane, appearance, rough_search));
    let sclera_probability = native
        .as_deref()
        .zip(appearance)
        .map(|(plane, appearance)| Arc::new(iris_sclera_probability_map(plane, appearance)));
    let reflectance_sclera_probability =
        native
            .as_deref()
            .zip(reflectance_appearance)
            .map(|(plane, appearance)| {
                Arc::new(iris_sclera_reflectance_probability_map(plane, appearance))
            });
    let context = Arc::new(OuterRayContext {
        luma: Arc::clone(&luma),
        native,
        sclera_probability,
        reflectance_sclera_probability,
        material_illumination,
        upper_eyelid: Arc::new(upper_eyelid.to_vec()),
        lower_eyelid: Arc::new(lower_eyelid.to_vec()),
        luma_gate: estimate_luma_transition_gate(luma.as_ref()),
        width,
        height,
        search: rough_search,
        rough_search,
        scale_range: (OUTER_IRIS_MIN_SEARCH_SCALE, OUTER_IRIS_MAX_SEARCH_SCALE),
    });
    evaluate_outer_iris_rays(context, 1, 1)
        .candidates
        .into_iter()
        .enumerate()
        .flat_map(|(ray_index, candidates)| {
            candidates
                .into_iter()
                .map(move |candidate| OuterIrisCandidateDebug {
                    ray_index,
                    x: candidate.point.x,
                    y: candidate.point.y,
                    fused_score: candidate.score,
                    luma_score: candidate.luma_score,
                    material_light_score: candidate.material_light_score,
                    reflectance_score: candidate.reflectance_score,
                    intrinsic_shadow_bonus: candidate.intrinsic_shadow_bonus,
                    shadow_disagreement: candidate.shadow_disagreement,
                    reflectance_plateau: candidate.reflectance_plateau,
                    intrinsic_intensity_step: candidate.intrinsic_intensity_step,
                    meridian_profile_score: candidate.meridian_profile_score,
                    margin_clarity: candidate.margin_clarity,
                    pupil_void: candidate.pupil_void,
                    inner_limbus_step: candidate.inner_limbus_step,
                    iris_band: candidate.iris_band,
                    sclera_out: candidate.sclera_out,
                    far_sclera: candidate.far_sclera,
                    rough_rho: candidate.rough_rho,
                })
        })
        .collect()
}

/// Offline-only ablation: use the same candidate/ellipse machinery as the
/// live outer-limbus detector, but rank candidates by the material-aware
/// illumination-corrected score.  Keeping this separate lets hand labels
/// decide whether the reconstruction is genuinely useful before it touches a
/// live frame.
#[allow(clippy::too_many_arguments)]
pub fn debug_outer_iris_boundary_with_material_lighting(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
) -> OuterIrisBoundary {
    debug_outer_iris_boundary_with_material_lighting_strength(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        1.0,
    )
}

/// Offline-only material-light ablation with a bounded blend between the
/// established fused score (`0.0`) and the illumination-corrected score
/// (`1.0`).  This lets the evaluator measure whether lighting reconstruction
/// should act as a conservative extra cue instead of a wholesale replacement.
#[allow(clippy::too_many_arguments)]
pub fn debug_outer_iris_boundary_with_material_lighting_strength(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    light_strength: f64,
) -> OuterIrisBoundary {
    debug_outer_iris_boundary_with_material_lighting_internal(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        None,
        light_strength,
    )
}

/// Same offline ablation, with a bounded history of prior material-light
/// surfaces.  The current frame still supplies an independent estimate; the
/// history only stabilizes a plausible smooth lighting slope/curvature.
#[allow(clippy::too_many_arguments)]
pub fn debug_outer_iris_boundary_with_temporal_material_lighting(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    tracker: &mut MaterialIlluminationTracker,
) -> OuterIrisBoundary {
    debug_outer_iris_boundary_with_temporal_material_lighting_strength(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        tracker,
        1.0,
    )
}

/// Causal version of [`debug_outer_iris_boundary_with_material_lighting_strength`].
/// The temporal prior only stabilizes the smooth illumination surface; it does
/// not carry geometry or label information between frames.
#[allow(clippy::too_many_arguments)]
pub fn debug_outer_iris_boundary_with_temporal_material_lighting_strength(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    tracker: &mut MaterialIlluminationTracker,
    light_strength: f64,
) -> OuterIrisBoundary {
    debug_outer_iris_boundary_with_material_lighting_internal(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        Some(tracker),
        light_strength,
    )
}

/// Offline combined ablation: retain the established production-style outer
/// boundary unless the current material/light surface has bilateral, bounded
/// support.  When reliable, apply the causal material-light branch; when not,
/// the rejected light fit does not enter the temporal history.
#[allow(clippy::too_many_arguments)]
pub fn debug_outer_iris_boundary_with_gated_temporal_material_lighting(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    tracker: &mut MaterialIlluminationTracker,
) -> OuterIrisBoundary {
    let diagnostics =
        debug_material_illumination_diagnostics(raw, width, height, sensor_x, sensor_y, coarse);
    if !diagnostics.is_some_and(debug_material_illumination_is_reliable) {
        return detect_outer_iris_boundary_between_eyelids_at_sensor(
            raw,
            width,
            height,
            sensor_x,
            sensor_y,
            coarse,
            upper_eyelid,
            lower_eyelid,
        );
    }
    debug_outer_iris_boundary_with_temporal_material_lighting(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        upper_eyelid,
        lower_eyelid,
        tracker,
    )
}

#[allow(clippy::too_many_arguments)]
fn debug_outer_iris_boundary_with_material_lighting_internal(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    mut temporal_tracker: Option<&mut MaterialIlluminationTracker>,
    light_strength: f64,
) -> OuterIrisBoundary {
    if width < 16
        || height < 16
        || raw.len() < width * height
        || coarse.radius < 20.0
        || coarse.radius > width.min(height) as f64 * 0.45
    {
        return OuterIrisBoundary::default();
    }
    let seed = [coarse.center.0, coarse.center.1, coarse.radius];
    let rough_search = OuterSearchEllipse::from_coarse(coarse);
    let luma = Arc::new(BoxLuma5::new(raw, width, height));
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)
        .map(blur_outer_appearance)
        .map(Arc::new);
    let appearance = native
        .as_deref()
        .and_then(|plane| estimate_iris_sclera_appearance(plane, rough_search));
    let reflectance_appearance = native
        .as_deref()
        .and_then(|plane| estimate_iris_sclera_reflectance_appearance(plane, rough_search));
    let material_illumination = native
        .as_deref()
        .zip(appearance)
        .and_then(|(plane, appearance)| fit_material_illumination(plane, appearance, rough_search));
    let Some(current_illumination) = material_illumination else {
        return OuterIrisBoundary::default();
    };
    let material_illumination = temporal_tracker
        .as_deref_mut()
        .map(|tracker| {
            let blended = tracker.blend_prior(current_illumination);
            tracker.observe(current_illumination);
            blended
        })
        .unwrap_or(current_illumination);
    let sclera_probability = native
        .as_deref()
        .zip(appearance)
        .map(|(plane, appearance)| Arc::new(iris_sclera_probability_map(plane, appearance)));
    let reflectance_sclera_probability =
        native
            .as_deref()
            .zip(reflectance_appearance)
            .map(|(plane, appearance)| {
                Arc::new(iris_sclera_reflectance_probability_map(plane, appearance))
            });
    let branch_started = Instant::now();
    let mut diagnostics = OuterIrisDiagnostics {
        seed_usable: true,
        work_stride: 1,
        sample_stride: 1,
        ..OuterIrisDiagnostics::default()
    };
    let Some(branch) = run_outer_iris_branch(
        Arc::clone(&luma),
        native,
        sclera_probability,
        reflectance_sclera_probability,
        Some(material_illumination),
        Arc::new(upper_eyelid.to_vec()),
        Arc::new(lower_eyelid.to_vec()),
        width,
        height,
        seed,
        rough_search,
        OuterScoreMode::MaterialLighting(light_strength.clamp(0.0, 1.0)),
        OUTER_IRIS_MIN_SEARCH_SCALE,
        1,
        1,
        branch_started + OUTER_IRIS_SYSTEM_BUDGET,
        &mut diagnostics,
    ) else {
        return OuterIrisBoundary::default();
    };
    let mut contrasts = branch
        .evidence
        .iter()
        .map(|point| point.contrast)
        .collect::<Vec<_>>();
    let contrast = median(&mut contrasts).max(1.0);
    OuterIrisBoundary {
        center: (branch.ellipse[0], branch.ellipse[1]),
        major_radius: branch.ellipse[2],
        minor_radius: branch.ellipse[3],
        angle: branch.ellipse[4],
        evidence_points: branch.evidence,
        occluded_points: branch.occluded_points,
        veto_sweep_endpoints: Vec::new(),
        points: stable_screen_space_ellipse_points(branch.ellipse, 64, contrast),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OuterFeatureSample {
    scale: f64,
    point: OuterIrisPoint,
    linear_step: f64,
    normalized_step: f64,
    support: f64,
    chroma_step: f64,
    /// Pure log-chromatic sclera probability change. Unlike `chroma_step`,
    /// this excludes the legacy mixed posterior's luma term.
    reflectance_step: f64,
    reflectance_support: f64,
    reflectance_sclera_out: f64,
    reflectance_far_sclera: f64,
    reflectance_inner_iris: f64,
    /// Local log-intensity contrast with a scale-relative stabilizer. Under a
    /// slowly varying achromatic shadow, multiplying both sides of the edge
    /// by the same light factor leaves this coordinate unchanged.
    intrinsic_intensity_step: f64,
    intrinsic_intensity_support: f64,
    sclera_out: f64,
    far_sclera: f64,
    far_luma: f64,
    inner_iris: f64,
    pupil_void: f64,
    inner_limbus_step: f64,
    iris_band: f64,
    corrected_step: f64,
    corrected_support: f64,
}

impl OuterFeatureSample {
    /// Bounded, one-way support for a sclera transition whose brightness was
    /// suppressed by a lid-facing shadow. It becomes nonzero only when pure
    /// reflectance says "sclera" more strongly than the legacy intensity-
    /// aware posterior, which is the signature of multiplicative dimming.
    fn shadow_reflectance_score(
        self,
        angle: f64,
        visibility: f64,
        z_step: f64,
        z_support: f64,
        z_plateau: f64,
    ) -> f64 {
        let ramp = |value: f64, low: f64, high: f64| {
            ((value - low) / (high - low).max(1.0e-9)).clamp(0.0, 1.0)
        };
        let reflectance_plateau =
            0.55 * self.reflectance_sclera_out + 0.45 * self.reflectance_far_sclera;
        let mixed_plateau = 0.55 * self.sclera_out + 0.45 * self.far_sclera;
        let relative_prominence = (0.5
            + (0.18 * (z_step / 1.5).tanh()
                + 0.09 * (z_support / 1.5).tanh()
                + 0.07 * (z_plateau / 1.5).tanh()))
        .clamp(0.0, 1.0);
        let lid_shadow_weight = ramp(-angle.sin(), 0.05, 0.40);
        let brightness_disagreement = ramp(reflectance_plateau - mixed_plateau, 0.10, 0.35);
        visibility.clamp(0.0, 1.0)
            * lid_shadow_weight
            * brightness_disagreement
            * ramp(self.reflectance_step, 0.015, 0.14)
            * (0.24
                + 0.24 * ramp(self.reflectance_support, -0.01, 0.08)
                + 0.22 * ramp(reflectance_plateau, 0.48, 0.72)
                + 0.30 * relative_prominence)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OutwardEyeEdgeTopology {
    observable: bool,
    score: f64,
    limbus_order_score: f64,
    ridge_distance_px: f64,
}

/// Test the topology outside one lateral limbus proposal without using a
/// resized image or a learned material label. A genuine lateral limbus should
/// leave a short, sustained sclera band before the sightline reaches the
/// outer eye/lid margin. The farther margin is required to be a coherent
/// transition at three tangent offsets; a single lash, hot pixel, or skin
/// texture ridge therefore cannot satisfy the check by itself.
fn sample_outward_eye_edge_topology(
    context: &OuterRayContext,
    point: OuterIrisPoint,
    outward_normal: (f64, f64),
    equivalent_radius: f64,
) -> OutwardEyeEdgeTopology {
    let normal_length = outward_normal.0.hypot(outward_normal.1);
    if normal_length <= 1.0e-9 || !normal_length.is_finite() {
        return OutwardEyeEdgeTopology::default();
    }
    let normal = (
        outward_normal.0 / normal_length,
        outward_normal.1 / normal_length,
    );
    let tangent = (-normal.1, normal.0);
    // Leave a material-sized runway after the proposed limbus. Starting at
    // eight pixels let the +/-6 px shoulders re-measure the very edge under
    // test and call its broad halo a second ridge. At the current native eye
    // scale, fourteen-to-twenty pixels is the minimum resolvable sclera band.
    let minimum_distance = (equivalent_radius * 0.15).clamp(14.0, 20.0) as usize;
    let maximum_distance = (equivalent_radius * 0.46).clamp(34.0, 52.0) as usize;
    let inside = |x: f64, y: f64| {
        x >= 4.0
            && y >= 4.0
            && x < context.width.saturating_sub(5) as f64
            && y < context.height.saturating_sub(5) as f64
    };
    let ramp = |value: f64, low: f64, high: f64| {
        ((value - low) / (high - low).max(1.0e-9)).clamp(0.0, 1.0)
    };
    let mixed_material = context
        .native
        .as_deref()
        .zip(context.sclera_probability.as_deref())
        .map(|(native, probability)| (native, probability.as_slice()));
    let reflectance_material = context
        .native
        .as_deref()
        .zip(context.reflectance_sclera_probability.as_deref())
        .map(|(native, probability)| (native, probability.as_slice()));
    let sample_material = |material: Option<(&NativeLogPlane, &[f64])>, distance: f64| {
        material.map(|(native, probability)| {
            native.sample_map(
                probability,
                point.x + normal.0 * distance,
                point.y + normal.1 * distance,
            )
        })
    };

    // A farther ridge is meaningful only after independently re-establishing
    // the material order at the proposed point. A false outer-eye contour has
    // bright sclera on its inward side and darker lid/skin outward; it must
    // not earn topology credit merely because another wrinkle exists later.
    let mut corrected_order_votes = [0.0; 3];
    let mut local_order_inside = true;
    for (index, tangent_offset) in [-4.0, 0.0, 4.0].into_iter().enumerate() {
        let mut iris_side = 0.0;
        let mut sclera_side = 0.0;
        for distance in [3.0, 6.0, 9.0] {
            let inner = (
                point.x - normal.0 * distance + tangent.0 * tangent_offset,
                point.y - normal.1 * distance + tangent.1 * tangent_offset,
            );
            let outer = (
                point.x + normal.0 * distance + tangent.0 * tangent_offset,
                point.y + normal.1 * distance + tangent.1 * tangent_offset,
            );
            if !inside(inner.0, inner.1) || !inside(outer.0, outer.1) {
                local_order_inside = false;
                break;
            }
            iris_side += corrected_outer_profile_log_luma(context, inner.0, inner.1);
            sclera_side += corrected_outer_profile_log_luma(context, outer.0, outer.1);
        }
        if !local_order_inside {
            break;
        }
        corrected_order_votes[index] = (sclera_side - iris_side) / 3.0;
    }
    if !local_order_inside {
        return OutwardEyeEdgeTopology::default();
    }
    let coherent_corrected_order = corrected_order_votes
        .into_iter()
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    let corrected_luma_order = ramp(coherent_corrected_order, 0.012, 0.105);
    let material_order = |material: Option<(&NativeLogPlane, &[f64])>| {
        let Some(_) = material else {
            return corrected_luma_order;
        };
        let inner = [3.0, 6.0, 9.0]
            .into_iter()
            .filter_map(|distance| sample_material(material, -distance))
            .sum::<f64>()
            / 3.0;
        let outer = [3.0, 6.0, 9.0]
            .into_iter()
            .filter_map(|distance| sample_material(material, distance))
            .sum::<f64>()
            / 3.0;
        ramp(outer - inner, 0.035, 0.24)
    };
    let mixed_order = material_order(mixed_material);
    let reflectance_order = material_order(reflectance_material);
    let limbus_order_score = if mixed_material.is_some() || reflectance_material.is_some() {
        0.55 * corrected_luma_order + 0.25 * mixed_order + 0.20 * reflectance_order
    } else {
        corrected_luma_order
    };

    let mut eligible_distances = 0usize;
    let mut best_score = 0.0f64;
    let mut best_distance = 0.0f64;
    for distance in (minimum_distance..=maximum_distance).step_by(2) {
        let distance = distance as f64;
        let mut tangent_drops = [0.0; 3];
        let mut samples_inside = true;
        for (index, tangent_offset) in [-4.0, 0.0, 4.0].into_iter().enumerate() {
            let mut before = 0.0;
            let mut after = 0.0;
            for shoulder in [2.0, 4.0, 6.0] {
                let before_at = (
                    point.x + normal.0 * (distance - shoulder) + tangent.0 * tangent_offset,
                    point.y + normal.1 * (distance - shoulder) + tangent.1 * tangent_offset,
                );
                let after_at = (
                    point.x + normal.0 * (distance + shoulder) + tangent.0 * tangent_offset,
                    point.y + normal.1 * (distance + shoulder) + tangent.1 * tangent_offset,
                );
                if !inside(before_at.0, before_at.1) || !inside(after_at.0, after_at.1) {
                    samples_inside = false;
                    break;
                }
                before += corrected_outer_profile_log_luma(context, before_at.0, before_at.1);
                after += corrected_outer_profile_log_luma(context, after_at.0, after_at.1);
            }
            if !samples_inside {
                break;
            }
            tangent_drops[index] = (before - after) / 3.0;
        }
        if !samples_inside {
            continue;
        }
        eligible_distances += 1;
        // The outer eye margin normally darkens from sclera toward lid/skin.
        // Retain a small polarity-independent component for pale skin under
        // uneven illumination, but require the transition to survive all
        // three tangent offsets.
        let coherent_signed_drop = tangent_drops
            .into_iter()
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let coherent_magnitude = tangent_drops
            .into_iter()
            .map(f64::abs)
            .min_by(f64::total_cmp)
            .unwrap_or(0.0);
        let luma_ridge = 0.82 * ramp(coherent_signed_drop, 0.018, 0.090)
            + 0.18 * ramp(coherent_magnitude, 0.025, 0.110);

        let material_transition = |material: Option<(&NativeLogPlane, &[f64])>| {
            let Some(_) = material else {
                return (0.35, 0.45);
            };
            let before = [2.0, 4.0, 6.0]
                .into_iter()
                .filter_map(|shoulder| sample_material(material, distance - shoulder))
                .sum::<f64>()
                / 3.0;
            let after = [2.0, 4.0, 6.0]
                .into_iter()
                .filter_map(|shoulder| sample_material(material, distance + shoulder))
                .sum::<f64>()
                / 3.0;
            let band = [3.0, 6.0, (distance - 5.0).max(7.0)]
                .into_iter()
                .filter_map(|sample_distance| sample_material(material, sample_distance))
                .sum::<f64>()
                / 3.0;
            (
                ramp(before - after, 0.045, 0.24) * ramp(before, 0.40, 0.68),
                ramp(band, 0.36, 0.68),
            )
        };
        let (mixed_ridge, mixed_band) = material_transition(mixed_material);
        let (reflectance_ridge, reflectance_band) = material_transition(reflectance_material);
        let material_ridge = mixed_ridge.max(reflectance_ridge);
        let sclera_band = mixed_band.max(reflectance_band);
        let score = luma_ridge
            * (0.44 + 0.34 * material_ridge + 0.22 * sclera_band).clamp(0.0, 1.0)
            * (0.08 + 0.92 * limbus_order_score).clamp(0.0, 1.0);
        if score > best_score {
            best_score = score;
            best_distance = distance;
        }
    }
    OutwardEyeEdgeTopology {
        // Six 2-pixel-spaced hypotheses cover a twelve-pixel terminal search
        // interval. Less runway is a clipped/occluded observation, not proof
        // that skin is sclera.
        observable: eligible_distances >= 6,
        score: best_score.clamp(0.0, 1.0),
        limbus_order_score: limbus_order_score.clamp(0.0, 1.0),
        ridge_distance_px: best_distance,
    }
}

fn outer_feature_stats(
    samples: &[OuterFeatureSample],
    feature: impl Fn(&OuterFeatureSample) -> f64,
    minimum_scale: f64,
) -> (f64, f64) {
    let mut values = samples.iter().map(&feature).collect::<Vec<_>>();
    let center = percentile_f64(&mut values, 0.5);
    let mut deviations = samples
        .iter()
        .map(|sample| (feature(sample) - center).abs())
        .collect::<Vec<_>>();
    (
        center,
        (1.4826 * percentile_f64(&mut deviations, 0.5)).max(minimum_scale),
    )
}

fn outer_iris_ray_candidates(
    context: &OuterRayContext,
    angle: f64,
    sample_stride: usize,
) -> Vec<OuterRayCandidate> {
    let (first_scale, last_scale) = context.scale_range;
    let equivalent_radius = context.search.equivalent_radius();
    let range_width = (last_scale - first_scale).max(0.01);
    // Preserve sub-pixel refinement without blindly taking 101 samples on a
    // seven-pixel-wide final pass. The broad pass remains dense; progressively
    // narrower passes retain at least 33 samples and at most 0.45 px spacing.
    let base_sample_count =
        ((equivalent_radius * range_width / 0.45).ceil() as usize + 1).clamp(33, 101);
    let sample_stride = sample_stride.max(1).min(4);
    // Round the lattice interval count up so both radial endpoints survive
    // every exact 2x work reduction.
    let sample_count = (base_sample_count - 1).div_ceil(sample_stride) * sample_stride + 1;
    let mut samples = Vec::with_capacity(sample_count);
    let mut first_near_boundary = None;
    for step in (0..sample_count).step_by(sample_stride) {
        let scale =
            first_scale + step as f64 * (last_scale - first_scale) / (sample_count - 1) as f64;
        if first_near_boundary.is_some_and(|first| scale > first + 1.0 / equivalent_radius.max(1.0))
        {
            break;
        }
        let ((x, y), (normal_x, normal_y)) = context.search.point_and_normal(angle, scale);
        if x <= 25.0
            || y <= 25.0
            || x >= context.width.saturating_sub(26) as f64
            || y >= context.height.saturating_sub(26) as f64
        {
            continue;
        }
        // The dense fitter previously ignored the eyelid anatomy already
        // measured for this frame. Once an outward ray leaves the visible eye
        // opening, every farther edge is lid, skin, or eyebrow—not limbus.
        // Stop the ray at that boundary instead of letting a strong eyebrow
        // candidate recenter the subsequent refinement passes.
        let outside_eye_opening = eyelid_y_at_x(context.upper_eyelid.as_ref(), x)
            .is_some_and(|upper_y| y <= upper_y)
            || eyelid_y_at_x(context.lower_eyelid.as_ref(), x).is_some_and(|lower_y| y >= lower_y);
        if outside_eye_opening {
            break;
        }
        let inside_samples = [2.5, 5.5, 9.0].map(|distance| {
            context
                .luma
                .sample(x - normal_x * distance, y - normal_y * distance)
        });
        let outside_samples = [2.5, 5.5, 9.0].map(|distance| {
            context
                .luma
                .sample(x + normal_x * distance, y + normal_y * distance)
        });
        let inside = inside_samples.into_iter().sum::<f64>() / 3.0;
        let outside = outside_samples.into_iter().sum::<f64>() / 3.0;
        if first_near_boundary.is_none() {
            let near_inside = context.luma.sample(x - normal_x * 3.5, y - normal_y * 3.5);
            let sliver_luma = (context.luma.sample(x, y)
                + context.luma.sample(x + normal_x * 1.5, y + normal_y * 1.5))
                * 0.5;
            let luma_sliver = near_inside <= context.luma_gate.dark_maximum
                && sliver_luma >= context.luma_gate.sliver_bright_minimum
                && sliver_luma - near_inside >= context.luma_gate.sliver_minimum_step;
            if luma_sliver {
                first_near_boundary = Some(scale);
            }
        }
        if first_near_boundary.is_none() && normal_y < -0.20 {
            if let Some((native, sclera_probability)) = context
                .native
                .as_deref()
                .zip(context.sclera_probability.as_deref())
            {
                let transitions = [1.5, 3.0, 5.0].map(|distance| {
                    let iris_side = native.sample_map(
                        sclera_probability,
                        x - normal_x * distance,
                        y - normal_y * distance,
                    );
                    let sclera_side = native.sample_map(
                        sclera_probability,
                        x + normal_x * distance,
                        y + normal_y * distance,
                    );
                    (iris_side, sclera_side, sclera_side - iris_side)
                });
                let coherent_widths = transitions
                    .iter()
                    .filter(|transition| transition.2 >= 0.055)
                    .count();
                let iris_side = transitions
                    .iter()
                    .map(|transition| transition.0)
                    .sum::<f64>()
                    / transitions.len() as f64;
                let sclera_side = transitions
                    .iter()
                    .map(|transition| transition.1)
                    .sum::<f64>()
                    / transitions.len() as f64;
                // Top sclera can be deeply shadowed and therefore never pass
                // the absolute-bright sliver gate above.  Its chromatic and
                // local-texture class still changes coherently on the side
                // away from the pupil.  Preserve that first relative change
                // so a later sclera-to-lid shadow cannot win the same ray.
                if coherent_widths >= 2 && sclera_side - iris_side >= 0.075 && sclera_side >= 0.30 {
                    first_near_boundary = Some(scale);
                }
            }
        }
        if first_near_boundary.is_none() && normal_y < -0.20 {
            if let Some((native, reflectance_probability)) = context
                .native
                .as_deref()
                .zip(context.reflectance_sclera_probability.as_deref())
            {
                let transitions = [1.5, 3.0, 5.0].map(|distance| {
                    let iris_side = native.sample_map(
                        reflectance_probability,
                        x - normal_x * distance,
                        y - normal_y * distance,
                    );
                    let sclera_side = native.sample_map(
                        reflectance_probability,
                        x + normal_x * distance,
                        y + normal_y * distance,
                    );
                    (iris_side, sclera_side, sclera_side - iris_side)
                });
                let coherent_widths = transitions
                    .iter()
                    .filter(|transition| transition.2 >= 0.070)
                    .count();
                let iris_side = transitions
                    .iter()
                    .map(|transition| transition.0)
                    .sum::<f64>()
                    / transitions.len() as f64;
                let sclera_side = transitions
                    .iter()
                    .map(|transition| transition.1)
                    .sum::<f64>()
                    / transitions.len() as f64;
                // This fallback has no intensity term. It is deliberately
                // stricter than the established mixed-material stop and runs
                // only when both that stop and absolute luma failed.
                if coherent_widths == transitions.len()
                    && sclera_side - iris_side >= 0.10
                    && sclera_side >= 0.38
                {
                    first_near_boundary = Some(scale);
                }
            }
        }
        let linear_step = outside - inside;
        let support = inside_samples
            .into_iter()
            .zip(outside_samples)
            .map(|(inside, outside)| outside - inside)
            .min_by(f64::total_cmp)
            .unwrap_or(linear_step);
        if first_near_boundary.is_none() && normal_y < -0.20 {
            let micro_differences = [1.0, 2.0, 3.0].map(|distance| {
                let inside = context
                    .luma
                    .sample3(x - normal_x * distance, y - normal_y * distance);
                let outside = context
                    .luma
                    .sample3(x + normal_x * distance, y + normal_y * distance);
                outside - inside
            });
            let coherent_widths = micro_differences
                .into_iter()
                .filter(|difference| *difference >= context.luma_gate.sliver_minimum_step * 0.30)
                .count();
            let micro_step = micro_differences.into_iter().sum::<f64>() / 3.0;
            // A partially occluded top limbus is often only a modest positive
            // transition, not an absolute-white sclera plateau. Two agreeing
            // micro-widths make it a real local boundary. The 3x3 aperture
            // preserves a one- or two-pixel sclera sliver that the normal 5x5,
            // 2.5-9px scoring samples intentionally smooth away.
            if coherent_widths >= 2 && micro_step >= context.luma_gate.sliver_minimum_step * 0.55 {
                first_near_boundary = Some(scale);
            }
        }
        let local_low = inside_samples
            .into_iter()
            .chain(outside_samples)
            .min_by(f64::total_cmp)
            .unwrap_or(inside.min(outside));
        let local_high = inside_samples
            .into_iter()
            .chain(outside_samples)
            .max_by(f64::total_cmp)
            .unwrap_or(inside.max(outside));
        let normalized_step = linear_step / (local_high - local_low).max(24.0);
        let far_luma = [11.0, 17.0, 23.0]
            .into_iter()
            .map(|distance| {
                context
                    .luma
                    .sample(x + normal_x * distance, y + normal_y * distance)
            })
            .sum::<f64>()
            / 3.0;
        let (
            chroma_step,
            sclera_out,
            far_sclera,
            inner_iris,
            pupil_void,
            inner_limbus_step,
            iris_band,
        ) = context
            .native
            .as_deref()
            .zip(context.sclera_probability.as_deref())
            .map_or(
                (0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5),
                |(native, appearance)| {
                    let probability = |direction: f64, distance: f64| {
                        native.sample_map(
                            appearance,
                            x + normal_x * direction * distance,
                            y + normal_y * direction * distance,
                        )
                    };
                    let sclera_in = [2.5, 5.5, 9.0]
                        .into_iter()
                        .map(|distance| probability(-1.0, distance))
                        .sum::<f64>()
                        / 3.0;
                    let sclera_out = [2.5, 5.5, 9.0]
                        .into_iter()
                        .map(|distance| probability(1.0, distance))
                        .sum::<f64>()
                        / 3.0;
                    let far_sclera = [11.0, 17.0, 23.0]
                        .into_iter()
                        .map(|distance| probability(1.0, distance))
                        .sum::<f64>()
                        / 3.0;
                    let inner_iris = [7.0, 12.0]
                        .into_iter()
                        .map(|distance| 1.0 - probability(-1.0, distance))
                        .sum::<f64>()
                        / 2.0;
                    let iris_in = 1.0 - sclera_in;
                    let iris_out = 1.0 - sclera_out;
                    let radial_distance = (x - context.rough_search.center.0)
                        .hypot(y - context.rough_search.center.1)
                        .max(1.0);
                    let radial_x = (x - context.rough_search.center.0) / radial_distance;
                    let radial_y = (y - context.rough_search.center.1) / radial_distance;
                    let radial_at = |fraction: f64| {
                        (
                            context.rough_search.center.0 + radial_x * radial_distance * fraction,
                            context.rough_search.center.1 + radial_y * radial_distance * fraction,
                        )
                    };
                    let pupil_void = [0.12, 0.24, 0.36]
                        .into_iter()
                        .map(|fraction| {
                            let at = radial_at(fraction);
                            native.sample_void(at.0, at.1)
                        })
                        .sum::<f64>()
                        / 3.0;
                    let pupil_side = radial_at(0.28);
                    let iris_side = radial_at(0.50);
                    let inner_limbus_step = native.sample_void(pupil_side.0, pupil_side.1)
                        - native.sample_void(iris_side.0, iris_side.1);
                    let iris_band = [0.52, 0.68, 0.84]
                        .into_iter()
                        .map(|fraction| {
                            let at = radial_at(fraction);
                            1.0 - native.sample_map(appearance, at.0, at.1)
                        })
                        .sum::<f64>()
                        / 3.0;
                    (
                        (sclera_out - sclera_in) + 0.65 * (iris_in - iris_out),
                        sclera_out,
                        far_sclera,
                        inner_iris,
                        pupil_void,
                        inner_limbus_step,
                        iris_band,
                    )
                },
            );
        let (
            reflectance_step,
            reflectance_support,
            reflectance_sclera_out,
            reflectance_far_sclera,
            reflectance_inner_iris,
        ) = context
            .native
            .as_deref()
            .zip(context.reflectance_sclera_probability.as_deref())
            .map_or((0.0, 0.0, 0.5, 0.5, 0.5), |(native, reflectance)| {
                let probability = |direction: f64, distance: f64| {
                    native.sample_map(
                        reflectance,
                        x + normal_x * direction * distance,
                        y + normal_y * direction * distance,
                    )
                };
                let differences = [2.5, 5.5, 9.0]
                    .map(|distance| probability(1.0, distance) - probability(-1.0, distance));
                let sclera_out = [2.5, 5.5, 9.0]
                    .into_iter()
                    .map(|distance| probability(1.0, distance))
                    .sum::<f64>()
                    / 3.0;
                let far_sclera = [11.0, 17.0, 23.0]
                    .into_iter()
                    .map(|distance| probability(1.0, distance))
                    .sum::<f64>()
                    / 3.0;
                let inner_iris = [7.0, 12.0]
                    .into_iter()
                    .map(|distance| 1.0 - probability(-1.0, distance))
                    .sum::<f64>()
                    / 2.0;
                (
                    differences.into_iter().sum::<f64>() / differences.len() as f64,
                    differences
                        .into_iter()
                        .min_by(f64::total_cmp)
                        .unwrap_or(0.0),
                    sclera_out,
                    far_sclera,
                    inner_iris,
                )
            });
        let (intrinsic_intensity_step, intrinsic_intensity_support) =
            context.native.as_deref().map_or((0.0, 0.0), |native| {
                let differences = [2.5, 5.5, 9.0].map(|distance| {
                    let inside =
                        native.sample_intensity(x - normal_x * distance, y - normal_y * distance);
                    let outside =
                        native.sample_intensity(x + normal_x * distance, y + normal_y * distance);
                    illumination_invariant_log_intensity_step(inside, outside)
                });
                (
                    differences.into_iter().sum::<f64>() / differences.len() as f64,
                    differences
                        .into_iter()
                        .min_by(f64::total_cmp)
                        .unwrap_or(0.0),
                )
            });
        let (corrected_step, corrected_support) = context
            .native
            .as_deref()
            .zip(context.material_illumination)
            .map_or((0.0, 0.0), |(native, illumination)| {
                let differences = [2.5, 5.5, 9.0].map(|distance| {
                    illumination.corrected_intensity(
                        native,
                        x + normal_x * distance,
                        y + normal_y * distance,
                    ) - illumination.corrected_intensity(
                        native,
                        x - normal_x * distance,
                        y - normal_y * distance,
                    )
                });
                (
                    differences.into_iter().sum::<f64>() / differences.len() as f64,
                    differences
                        .into_iter()
                        .min_by(f64::total_cmp)
                        .unwrap_or(0.0),
                )
            });
        samples.push(OuterFeatureSample {
            scale,
            point: OuterIrisPoint {
                x,
                y,
                contrast: linear_step.max(1.0),
            },
            linear_step,
            normalized_step,
            support,
            chroma_step,
            reflectance_step,
            reflectance_support,
            reflectance_sclera_out,
            reflectance_far_sclera,
            reflectance_inner_iris,
            intrinsic_intensity_step,
            intrinsic_intensity_support,
            sclera_out,
            far_sclera,
            far_luma,
            inner_iris,
            pupil_void,
            inner_limbus_step,
            iris_band,
            corrected_step,
            corrected_support,
        });
    }
    if samples.is_empty() {
        return Vec::new();
    }
    if angle.sin() > 0.15 && context.sclera_probability.is_some() {
        // On the exposed lower arc, a real limbus must leave iris material.
        // A strong fibre/shadow edge can have excellent luma contrast while
        // the near and far samples beyond it are still plainly iris. Reject
        // that material-continuous edge before relative per-ray ranking can
        // promote it. This is deliberately a bottom-only veto: the upper arc
        // is occluded and already excluded, while lateral sclera can be a very
        // narrow sliver at oblique gaze.
        samples.retain(|sample| {
            let sustained_sclera = 0.55 * sample.sclera_out + 0.45 * sample.far_sclera;
            let iris_continues_outward =
                sample.sclera_out < 0.42 && sample.far_sclera < 0.42 && sustained_sclera < 0.38;
            !iris_continues_outward
        });
        if samples.is_empty() {
            return Vec::new();
        }
    }
    // The luma scan above already clipped this ray just beyond the first
    // sclera-colored sliver. Preserve the complete relative-score lattice on
    // its iris side: the real limbus can be a narrow, low-amplitude transition
    // and must not pass a stronger sustained-plateau gate merely to
    // participate in the closed-curve fit.
    let linear_stats = outer_feature_stats(&samples, |sample| sample.linear_step, 12.0);
    let normalized_stats = outer_feature_stats(&samples, |sample| sample.normalized_step, 0.035);
    let support_stats = outer_feature_stats(&samples, |sample| sample.support, 12.0);
    let chroma_stats = outer_feature_stats(&samples, |sample| sample.chroma_step, 0.035);
    let reflectance_step_stats =
        outer_feature_stats(&samples, |sample| sample.reflectance_step, 0.035);
    let reflectance_support_stats =
        outer_feature_stats(&samples, |sample| sample.reflectance_support, 0.035);
    let reflectance_plateau_stats =
        outer_feature_stats(&samples, |sample| sample.reflectance_far_sclera, 0.035);
    let intrinsic_intensity_step_stats =
        outer_feature_stats(&samples, |sample| sample.intrinsic_intensity_step, 0.035);
    let intrinsic_intensity_support_stats =
        outer_feature_stats(&samples, |sample| sample.intrinsic_intensity_support, 0.035);
    let sclera_stats = outer_feature_stats(&samples, |sample| sample.sclera_out, 0.035);
    let plateau_stats = outer_feature_stats(&samples, |sample| sample.far_sclera, 0.035);
    let far_luma_stats = outer_feature_stats(&samples, |sample| sample.far_luma, 12.0);
    let corrected_step_stats = outer_feature_stats(&samples, |sample| sample.corrected_step, 0.025);
    let corrected_support_stats =
        outer_feature_stats(&samples, |sample| sample.corrected_support, 0.025);
    let z = |value: f64, stats: (f64, f64)| ((value - stats.0) / stats.1).clamp(-5.0, 7.0);
    let visibility = if angle.sin() < -0.35 {
        0.10
    } else if angle.sin() < -0.02 {
        0.28
    } else {
        0.62 + 0.38 * angle.cos().abs()
    };
    let scored = samples
        .into_iter()
        .map(|sample| {
            let z_linear = z(sample.linear_step, linear_stats);
            let z_normalized = z(sample.normalized_step, normalized_stats);
            let z_support = z(sample.support, support_stats);
            let z_chroma = z(sample.chroma_step, chroma_stats);
            let z_reflectance_step = z(sample.reflectance_step, reflectance_step_stats);
            let z_reflectance_support = z(sample.reflectance_support, reflectance_support_stats);
            let z_reflectance_plateau = z(sample.reflectance_far_sclera, reflectance_plateau_stats);
            let z_intrinsic_intensity_step = z(
                sample.intrinsic_intensity_step,
                intrinsic_intensity_step_stats,
            );
            let z_intrinsic_intensity_support = z(
                sample.intrinsic_intensity_support,
                intrinsic_intensity_support_stats,
            );
            let z_sclera = z(sample.sclera_out, sclera_stats);
            let z_plateau = z(sample.far_sclera, plateau_stats);
            let z_far_luma = z(sample.far_luma, far_luma_stats);
            let z_corrected_step = z(sample.corrected_step, corrected_step_stats);
            let z_corrected_support = z(sample.corrected_support, corrected_support_stats);
            let meridian_profile_score = visibility
                * (0.55 * (sample.pupil_void - 0.5)
                    + 0.85 * sample.inner_limbus_step
                    + 1.10 * (sample.iris_band - 0.5)
                    + 0.80 * (sample.inner_iris - 0.5)
                    + 1.30 * (0.55 * sample.sclera_out + 0.45 * sample.far_sclera - 0.5));
            // The lower and lateral limbus already have stronger ordinary
            // evidence and define the authoritative fit. Spend this new cue
            // only on lid-facing meridians, where it can improve an opposing
            // comparator without pulling the conic away from the exposed arc.
            let reflectance_score = sample.shadow_reflectance_score(
                angle,
                visibility,
                z_reflectance_step,
                z_reflectance_support,
                z_reflectance_plateau,
            );
            let margin_clarity = (0.50
                + 0.12 * z_normalized
                + 0.10 * z_support
                + 0.10 * z_chroma
                + 0.08 * z_plateau
                + 0.08 * reflectance_score)
                .clamp(0.0, 1.0);
            let rough_coordinates = context
                .rough_search
                .normalized_coordinates(sample.point.x, sample.point.y);
            let rough_rho = rough_coordinates.0.hypot(rough_coordinates.1);
            let rough_delta = rough_rho - 1.0;
            // Stay anchored to the original rough iris hypothesis even after
            // a refinement ellipse moves. This is an independent per-point
            // prior, not a spring between neighboring yellow measurements.
            let lateral = angle.cos().abs().powi(4);
            let rough_penalty = if rough_delta >= 0.0 {
                // Eyelid/skin aliases justify a strong outward prior on the
                // vertical arcs, but that prior was too conservative at 3/9
                // o'clock: it preferred an iris texture edge before reaching
                // visible sclera.  Permit the material transition to win over
                // the rough seed on the lateral limbus.
                let outward_weight = 9.0 * (1.0 - lateral) + 2.25 * lateral;
                let free_margin = 0.04 * (1.0 - lateral) + 0.10 * lateral;
                outward_weight * ((rough_delta - free_margin).max(0.0) / 0.22).powi(2)
            } else {
                // Preserve the broad inward recovery away from the lateral
                // anchors, while mildly disfavoring the side-of-iris texture
                // edges that caused the observed undersized fit.
                let inward_weight = 3.0 + 2.0 * lateral;
                let free_margin = 0.18 * (1.0 - lateral) + 0.10 * lateral;
                inward_weight * ((-rough_delta - free_margin).max(0.0) / 0.24).powi(2)
            };
            let scale_penalty = 0.10 * ((sample.scale - 1.0) / range_width).powi(2) + rough_penalty;
            let luma_score = visibility
                * (1.50 * z_normalized + 0.85 * z_support + 0.35 * z_linear)
                - scale_penalty;
            let legacy_core = 1.25 * z_normalized
                + 0.55 * z_linear
                + 0.72 * z_support
                + 1.10 * z_chroma
                + 0.48 * z_sclera
                + 1.35 * z_plateau
                + 0.55 * z_far_luma
                + 1.75 * (sample.far_sclera - 0.5)
                + 0.75 * (sample.inner_iris - 0.5);
            let reflectance_plateau =
                0.55 * sample.reflectance_sclera_out + 0.45 * sample.reflectance_far_sclera;
            let mixed_plateau = 0.55 * sample.sclera_out + 0.45 * sample.far_sclera;
            let shadow_disagreement =
                ((reflectance_plateau - mixed_plateau - 0.04) / 0.22).clamp(0.0, 1.0);
            let reflectance_material_support =
                ((reflectance_plateau - 0.42) / 0.28).clamp(0.0, 1.0);
            let reflectance_inner_support =
                ((sample.reflectance_inner_iris - 0.42) / 0.28).clamp(0.0, 1.0);
            let intrinsic_rank = (0.34 * z_intrinsic_intensity_step
                + 0.20 * z_intrinsic_intensity_support
                + 0.28 * z_reflectance_step
                + 0.12 * z_reflectance_support
                + 0.24 * z_reflectance_plateau)
                .clamp(0.0, 3.0);
            let intrinsic_shadow_bonus = visibility
                * shadow_disagreement
                * reflectance_material_support
                * reflectance_inner_support
                * intrinsic_rank;
            let score = visibility * legacy_core
                + meridian_profile_score
                + OUTER_IRIS_INTRINSIC_SHADOW_BONUS_STRENGTH * intrinsic_shadow_bonus
                + OUTER_IRIS_REFLECTANCE_FUSED_STRENGTH * reflectance_score
                - scale_penalty;
            // This branch keeps the ordinary material evidence but replaces
            // most raw-luma preference with a transition measured after a
            // material-aware low-frequency lighting reconstruction.  It is
            // experimental/offline until labels prove that it helps.
            let material_light_score = visibility
                * (1.38 * z_corrected_step
                    + 0.92 * z_corrected_support
                    + 1.18 * z_chroma
                    + 0.58 * z_sclera
                    + 0.92 * z_plateau
                    + 0.42 * z_far_luma
                    + 1.25 * (sample.far_sclera - 0.5)
                    + 0.62 * (sample.inner_iris - 0.5))
                + OUTER_IRIS_REFLECTANCE_LIGHT_STRENGTH * reflectance_score
                - scale_penalty;
            OuterRayCandidate {
                score,
                luma_score,
                material_light_score,
                reflectance_score,
                intrinsic_shadow_bonus,
                shadow_disagreement,
                reflectance_plateau,
                intrinsic_intensity_step: sample.intrinsic_intensity_step,
                meridian_profile_score,
                margin_clarity,
                pupil_void: sample.pupil_void,
                inner_limbus_step: sample.inner_limbus_step,
                iris_band: sample.iris_band,
                sclera_out: sample.sclera_out,
                far_sclera: sample.far_sclera,
                radius: sample.scale * equivalent_radius,
                rough_rho,
                point: sample.point,
            }
        })
        .collect::<Vec<_>>();
    let candidate_limit = if range_width > 0.40 { 14 } else { 12 };
    let minimum_separation = (equivalent_radius * range_width / 80.0).clamp(0.28, 1.20);
    let mut selected = Vec::with_capacity(candidate_limit * 2);
    // Preserve the best match in each radial band before adding the global
    // score leaders. Otherwise many samples around one very strong eyebrow
    // edge can evict the weaker mid-distance limbus arc from the shortlist.
    for bin in 0..10 {
        let low = 0.62 + bin as f64 * 0.09;
        let high = low + 0.09;
        let candidate = scored
            .iter()
            .copied()
            .filter(|candidate| candidate.rough_rho >= low && candidate.rough_rho < high)
            .max_by(|left, right| left.score.total_cmp(&right.score));
        if let Some(candidate) = candidate {
            if !selected.iter().any(|existing: &OuterRayCandidate| {
                (existing.radius - candidate.radius).abs() < minimum_separation
            }) {
                selected.push(candidate);
            }
        }
    }
    for score_mode in [OuterScoreMode::Fused, OuterScoreMode::Luma] {
        let mut ranked = (0..scored.len()).collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            score_mode
                .score(scored[*right])
                .total_cmp(&score_mode.score(scored[*left]))
        });
        for index in ranked.into_iter().take(candidate_limit * 2) {
            let candidate = scored[index];
            if selected.iter().any(|existing: &OuterRayCandidate| {
                (existing.radius - candidate.radius).abs() < minimum_separation
            }) {
                continue;
            }
            selected.push(candidate);
            if selected.len() >= candidate_limit * 2 {
                break;
            }
        }
    }
    selected
}

#[derive(Clone, Copy, Debug)]
enum OuterScoreMode {
    Fused,
    Luma,
    MaterialLighting(f64),
}

impl OuterScoreMode {
    fn score(self, candidate: OuterRayCandidate) -> f64 {
        match self {
            Self::Fused => candidate.score,
            Self::Luma => candidate.luma_score,
            // Both component scores share the same geometry/visibility
            // penalties.  Keeping the blend in score space retains the
            // current candidate lattice and only changes its ranking.
            Self::MaterialLighting(strength) => {
                let strength = strength.clamp(0.0, 1.0);
                candidate.score * (1.0 - strength) + candidate.material_light_score * strength
            }
        }
    }
}

fn trace_outer_iris_curve(
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    expected_radius: f64,
    score_mode: OuterScoreMode,
    refined: bool,
) -> [Option<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES] {
    let cohesion_tolerance = if refined { 0.060 } else { 0.090 };
    let supported_score = |ray: usize, candidate: OuterRayCandidate| {
        let base = score_mode.score(candidate);
        let mut support = 0.0;
        for distance in [1usize, 2] {
            for neighbor in [
                (ray + distance) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES,
                (ray + OUTER_IRIS_DENSE_EVIDENCE_SAMPLES - distance)
                    % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES,
            ] {
                let neighbor_max = candidates[neighbor]
                    .iter()
                    .map(|candidate| score_mode.score(*candidate))
                    .max_by(f64::total_cmp);
                let matching = candidates[neighbor]
                    .iter()
                    .filter(|other| {
                        (other.rough_rho - candidate.rough_rho).abs() <= cohesion_tolerance
                    })
                    .map(|other| score_mode.score(*other))
                    .max_by(f64::total_cmp);
                if let (Some(neighbor_max), Some(matching)) = (neighbor_max, matching) {
                    support += (1.0 + (matching - neighbor_max) / 2.0).clamp(0.0, 1.0);
                }
            }
        }
        // A discrete arc vote: nearby rays either contain a competitive match
        // in this radial band or they do not. Unlike the removed spring, this
        // never penalizes or pulls a point by a distance-squared force.
        base + 0.70 * support
    };
    let fallback = || {
        std::array::from_fn(|index| {
            candidates[index]
                .iter()
                .max_by(|left, right| {
                    supported_score(index, **left).total_cmp(&supported_score(index, **right))
                })
                .copied()
        })
    };
    // The former dynamic program had a zero inter-ray spring, so its nested
    // candidate walk reduced mathematically to this same independent maximum.
    // Neighboring support is already included in supported_score; selecting
    // it directly preserves cohesion without spending O(rays*candidates^2)
    // or allowing one strong eyelid hit to move adjacent measurements.
    let _ = expected_radius;
    fallback()
}

fn outer_meridian_reliability(angle: f64) -> f64 {
    let vertical = angle.sin();
    if vertical < -0.35 {
        0.08
    } else if vertical < -0.08 {
        0.30
    } else if vertical > 0.30 {
        0.82 + 0.18 * angle.cos().abs()
    } else {
        1.0
    }
}

fn angular_distance(left: f64, right: f64) -> f64 {
    let difference = (left - right).rem_euclid(2.0 * PI);
    difference.min(2.0 * PI - difference)
}

fn opposing_outer_ray_indices(index: usize) -> [usize; 2] {
    let target = (outer_iris_evidence_angle(index) + PI).rem_euclid(2.0 * PI);
    let mut best = [(usize::MAX, f64::INFINITY); 2];
    for candidate in 0..OUTER_IRIS_DENSE_EVIDENCE_SAMPLES {
        let distance = angular_distance(outer_iris_evidence_angle(candidate), target);
        if distance < best[0].1 {
            best[1] = best[0];
            best[0] = (candidate, distance);
        } else if distance < best[1].1 {
            best[1] = (candidate, distance);
        }
    }
    [best[0].0, best[1].0]
}

fn opposing_meridian_support(
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    index: usize,
    candidate: OuterRayCandidate,
    score_mode: OuterScoreMode,
) -> f64 {
    opposing_outer_ray_indices(index)
        .into_iter()
        .filter(|opposite| *opposite < OUTER_IRIS_DENSE_EVIDENCE_SAMPLES)
        .filter_map(|opposite| {
            let maximum = candidates[opposite]
                .iter()
                .map(|other| score_mode.score(*other))
                .max_by(f64::total_cmp)?;
            candidates[opposite]
                .iter()
                .filter(|other| (other.rough_rho - candidate.rough_rho).abs() <= 0.085)
                .map(|other| {
                    let competitiveness =
                        (1.0 + (score_mode.score(*other) - maximum) / 2.5).clamp(0.0, 1.0);
                    // Opposite image rays represent the two halves of one 3D
                    // iris meridian. Strong support therefore requires more
                    // than a coincident edge: pupil void, inner-limbus exit,
                    // iris tissue, outer transition, and sustained sclera must
                    // tell a compatible radial material story.
                    let profile_disagreement = 0.12
                        * (other.pupil_void - candidate.pupil_void).abs()
                        + 0.16 * (other.inner_limbus_step - candidate.inner_limbus_step).abs()
                        + 0.24 * (other.iris_band - candidate.iris_band).abs()
                        + 0.24 * (other.sclera_out - candidate.sclera_out).abs()
                        + 0.24 * (other.far_sclera - candidate.far_sclera).abs();
                    let profile_agreement = (1.0 - profile_disagreement).clamp(0.0, 1.0);
                    let theory_strength = (0.5
                        + 0.18
                            * candidate
                                .meridian_profile_score
                                .min(other.meridian_profile_score))
                    .clamp(0.0, 1.0);
                    let opposite_reliability =
                        outer_meridian_reliability(outer_iris_evidence_angle(opposite));
                    let margin_agreement = if opposite_reliability < 0.25 {
                        // Missing high-detail margin structure is expected on
                        // the eyelash/shadowed upper third. It may veto a bad
                        // material ordering, but cannot erase a clear lower hit.
                        0.62 + 0.38 * candidate.margin_clarity.max(other.margin_clarity)
                    } else {
                        (1.0 - (candidate.margin_clarity - other.margin_clarity).abs())
                            * (0.45 + 0.55 * candidate.margin_clarity.min(other.margin_clarity))
                    };
                    competitiveness
                        * (0.18
                            + 0.38 * profile_agreement
                            + 0.24 * theory_strength
                            + 0.20 * margin_agreement)
                })
                .max_by(f64::total_cmp)
        })
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug, Default)]
struct AnalogOuterEdgeForce {
    /// Sub-pixel location of the positive iris-to-sclera derivative relative
    /// to the accepted discrete contact. Positive is outward.
    edge_offset_px: f64,
    /// Absolute transition amplitude after log-light correction, normalized
    /// to `[0, 1]` independently of the force direction.
    power: f64,
    /// Agreement of polarity, derivative scale, peak localization, material
    /// ordering, and the illumination reconstruction.
    certainty: f64,
}

fn corrected_outer_profile_log_luma(context: &OuterRayContext, x: f64, y: f64) -> f64 {
    let raw_log = (context.luma.sample3(x, y) + 8.0).ln();
    raw_log
        - context
            .material_illumination
            .map_or(0.0, |illumination| illumination.light_at(x, y))
}

/// Measure a continuous, signed edge around one already accepted meridian
/// contact. The narrow response localizes the edge; the wider apertures ask
/// whether the same iris-to-sclera polarity survives scale. All samples stay
/// in native ROI coordinates and reuse the existing RAW integral/material
/// planes—there is no resized image or learned inference path here.
fn sample_analog_outer_edge_force(
    context: &OuterRayContext,
    anchor: (f64, f64),
    outward_normal: (f64, f64),
) -> Option<AnalogOuterEdgeForce> {
    let normal_length = outward_normal.0.hypot(outward_normal.1);
    if !anchor.0.is_finite()
        || !anchor.1.is_finite()
        || !normal_length.is_finite()
        || normal_length <= 1.0e-9
    {
        return None;
    }
    let normal = (
        outward_normal.0 / normal_length,
        outward_normal.1 / normal_length,
    );
    let material = context
        .native
        .as_deref()
        .zip(context.sclera_probability.as_deref());
    let mut responses = [0.0; OUTER_IRIS_ANALOG_PROFILE_OFFSETS.len()];
    let mut coherences = [0.0; OUTER_IRIS_ANALOG_PROFILE_OFFSETS.len()];
    for (offset_index, offset) in OUTER_IRIS_ANALOG_PROFILE_OFFSETS.into_iter().enumerate() {
        let center = (anchor.0 + normal.0 * offset, anchor.1 + normal.1 * offset);
        let mut luma_vote = 0.0;
        let mut material_vote = 0.0;
        let mut agreeing = 0usize;
        let mut vote_count = 0usize;
        for aperture in OUTER_IRIS_ANALOG_APERTURES {
            let inside = (
                center.0 - normal.0 * aperture,
                center.1 - normal.1 * aperture,
            );
            let outside = (
                center.0 + normal.0 * aperture,
                center.1 + normal.1 * aperture,
            );
            let luma_difference = corrected_outer_profile_log_luma(context, outside.0, outside.1)
                - corrected_outer_profile_log_luma(context, inside.0, inside.1);
            luma_vote += (luma_difference / 0.085).tanh();
            agreeing += usize::from(luma_difference >= 0.018);
            vote_count += 1;
            if let Some((native, sclera_probability)) = material {
                let material_difference =
                    native.sample_map(sclera_probability, outside.0, outside.1)
                        - native.sample_map(sclera_probability, inside.0, inside.1);
                material_vote += (material_difference / 0.11).tanh();
                agreeing += usize::from(material_difference >= 0.025);
                vote_count += 1;
            }
        }
        luma_vote /= OUTER_IRIS_ANALOG_APERTURES.len() as f64;
        let combined = if material.is_some() {
            material_vote /= OUTER_IRIS_ANALOG_APERTURES.len() as f64;
            0.62 * luma_vote + 0.38 * material_vote
        } else {
            luma_vote
        };
        responses[offset_index] = combined.max(0.0);
        coherences[offset_index] = agreeing as f64 / vote_count.max(1) as f64;
    }

    let mut ordered_response = responses;
    let baseline = percentile_f64(&mut ordered_response, 0.18);
    let peak = responses.into_iter().max_by(f64::total_cmp).unwrap_or(0.0);
    if !peak.is_finite() || peak < 0.055 {
        return None;
    }
    let masses = responses.map(|response| (response - 0.72 * baseline).max(0.0).powi(2));
    let mass_sum = masses.into_iter().sum::<f64>();
    if !mass_sum.is_finite() || mass_sum <= 1.0e-8 {
        return None;
    }
    let centroid = OUTER_IRIS_ANALOG_PROFILE_OFFSETS
        .into_iter()
        .zip(masses)
        .map(|(offset, mass)| offset * mass)
        .sum::<f64>()
        / mass_sum;
    let concentration = OUTER_IRIS_ANALOG_PROFILE_OFFSETS
        .into_iter()
        .zip(masses)
        .filter(|(offset, _)| (*offset - centroid).abs() <= 2.15)
        .map(|(_, mass)| mass)
        .sum::<f64>()
        / mass_sum;
    let scale_coherence = coherences
        .into_iter()
        .zip(masses)
        .map(|(coherence, mass)| coherence * mass)
        .sum::<f64>()
        / mass_sum;
    let prominence = ((peak - baseline) / (peak + 0.08)).clamp(0.0, 1.0);

    let refined = (
        anchor.0 + normal.0 * centroid,
        anchor.1 + normal.1 * centroid,
    );
    let order_apertures = [2.5, 4.75];
    let luma_order = order_apertures
        .into_iter()
        .map(|distance| {
            corrected_outer_profile_log_luma(
                context,
                refined.0 + normal.0 * distance,
                refined.1 + normal.1 * distance,
            ) - corrected_outer_profile_log_luma(
                context,
                refined.0 - normal.0 * distance,
                refined.1 - normal.1 * distance,
            )
        })
        .sum::<f64>()
        / order_apertures.len() as f64;
    let luma_order = (luma_order / 0.11).tanh().max(0.0);
    let material_order = material.map_or(luma_order, |(native, sclera_probability)| {
        order_apertures
            .into_iter()
            .map(|distance| {
                native.sample_map(
                    sclera_probability,
                    refined.0 + normal.0 * distance,
                    refined.1 + normal.1 * distance,
                ) - native.sample_map(
                    sclera_probability,
                    refined.0 - normal.0 * distance,
                    refined.1 - normal.1 * distance,
                )
            })
            .sum::<f64>()
            / order_apertures.len() as f64
    });
    let material_order = (material_order / 0.12).tanh().max(0.0);
    let side_order = if material.is_some() {
        0.58 * luma_order + 0.42 * material_order
    } else {
        luma_order
    };
    let lighting_certainty = context
        .material_illumination
        .map_or(0.58, material_illumination_confidence);
    let power = (peak / 0.62).tanh().clamp(0.0, 1.0);
    let certainty = lighting_certainty
        * (0.22 + 0.78 * scale_coherence.clamp(0.0, 1.0))
        * (0.34 + 0.66 * concentration.clamp(0.0, 1.0))
        * (0.24 + 0.76 * prominence)
        * (0.28 + 0.72 * side_order.clamp(0.0, 1.0));
    Some(AnalogOuterEdgeForce {
        edge_offset_px: centroid.clamp(-4.0, 4.0),
        power,
        certainty: certainty.clamp(0.0, 1.0),
    })
}

#[derive(Clone)]
struct OuterAnalogRefinement {
    rays: [Option<OuterIrisPoint>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    fit_weights: [f64; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    elapsed: Duration,
    samples: usize,
    outward: usize,
    inward: usize,
    mean_signed_offset_px: f64,
    mean_power: f64,
    mean_certainty: f64,
    fit_applied: bool,
}

fn analog_measurement_cost(
    measurements: &[Option<OuterIrisPoint>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    weights: &[f64; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    ellipse: [f64; 5],
) -> f64 {
    let mut weighted_error = 0.0;
    let mut weight_sum = 0.0;
    for (index, point) in measurements.iter().enumerate() {
        let Some(point) = point else {
            continue;
        };
        let weight = weights[index].clamp(0.0, 1.0);
        let residual = outer_ellipse_point_residual(*point, ellipse).min(8.0);
        weighted_error += weight * residual * residual;
        weight_sum += weight;
    }
    weighted_error / weight_sum.max(1.0e-9)
}

fn fitted_outer_phase_for_contact(search: OuterSearchEllipse, point: OuterIrisPoint) -> f64 {
    let fitted_coordinates = search.normalized_coordinates(point.x, point.y);
    fitted_coordinates.1.atan2(fitted_coordinates.0)
}

fn refine_outer_analog_edge_forces(
    context: &OuterRayContext,
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    rays: &[Option<OuterIrisPoint>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    seed: [f64; 3],
    score_mode: OuterScoreMode,
    _system_deadline: Instant,
) -> Option<OuterAnalogRefinement> {
    let started = Instant::now();
    if rays.iter().flatten().count() < 8 {
        return None;
    }
    let initial_points = rays.iter().flatten().copied().collect::<Vec<_>>();
    let initial_ellipse = fit_outer_ellipse(&initial_points, seed);
    let fitted_search = OuterSearchEllipse::from_fit(initial_ellipse);
    let mut adjusted = *rays;
    let mut measurements = [None; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
    let mut fit_weights = [0.18; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
    let mut order = (0..OUTER_IRIS_DENSE_EVIDENCE_SAMPLES)
        .filter(|index| rays[*index].is_some())
        .collect::<Vec<_>>();
    order.sort_by(|left, right| {
        let priority = |index: usize| {
            let angle = outer_iris_evidence_angle(index);
            outer_meridian_reliability(angle)
                + 0.22 * angle.cos().abs()
                + 0.18 * (angle.sin() > 0.30) as u8 as f64
        };
        priority(*right).total_cmp(&priority(*left))
    });

    let maximum_pull = (seed[2] * 0.065).clamp(2.5, 6.0);
    let mut samples = 0usize;
    let mut outward = 0usize;
    let mut inward = 0usize;
    let mut signed_sum = 0.0;
    let mut signed_weight = 0.0;
    let mut power_sum = 0.0;
    let mut certainty_sum = 0.0;
    let mut left = 0usize;
    let mut right = 0usize;
    let mut lower = 0usize;
    for index in order {
        let accepted = rays[index]?;
        let candidate = candidates[index].iter().copied().min_by(|left, right| {
            (left.point.x - accepted.x)
                .hypot(left.point.y - accepted.y)
                .total_cmp(&(right.point.x - accepted.x).hypot(right.point.y - accepted.y))
        })?;
        let angle = outer_iris_evidence_angle(index);
        // `angle` parameterizes the rough search ellipse. A rotated conic fit
        // may use an axis-swapped but geometrically equivalent parameterization,
        // so feeding that phase directly to the fitted ellipse can land on the
        // opposite side of the eye. Project this accepted contact into the
        // fitted conic's own coordinates and recover its local phase there.
        let fitted_phase = fitted_outer_phase_for_contact(fitted_search, accepted);
        let (model_point, model_normal) = fitted_search.point_and_normal(fitted_phase, 1.0);
        let Some(profile) =
            sample_analog_outer_edge_force(context, (accepted.x, accepted.y), model_normal)
        else {
            continue;
        };
        let measured = OuterIrisPoint {
            x: accepted.x + model_normal.0 * profile.edge_offset_px,
            y: accepted.y + model_normal.1 * profile.edge_offset_px,
            contrast: accepted.contrast,
        };
        let signed_offset = (measured.x - model_point.0) * model_normal.0
            + (measured.y - model_point.1) * model_normal.1;
        let opposite = opposing_meridian_support(candidates, index, candidate, score_mode);
        let discrete_certainty =
            (0.24 + 0.46 * candidate.margin_clarity + 0.30 * opposite).clamp(0.0, 1.0);
        let reliability = outer_meridian_reliability(angle);
        let certainty =
            (profile.certainty * (0.58 + 0.42 * discrete_certainty) * reliability).clamp(0.0, 1.0);
        let power = (profile.power * (0.64 + 0.36 * candidate.margin_clarity)).clamp(0.0, 1.0);
        let authority = (power * (0.34 + 0.66 * certainty)).clamp(0.0, 1.0);
        let applied_offset = signed_offset.clamp(-maximum_pull, maximum_pull) * authority;
        adjusted[index] = Some(OuterIrisPoint {
            x: model_point.0 + model_normal.0 * applied_offset,
            y: model_point.1 + model_normal.1 * applied_offset,
            contrast: accepted.contrast,
        });
        measurements[index] = Some(measured);
        fit_weights[index] = (0.14 + 0.86 * power * certainty).clamp(0.14, 1.0);
        let evidence_weight = (power * certainty).max(0.02);
        signed_sum += signed_offset * evidence_weight;
        signed_weight += evidence_weight;
        power_sum += power;
        certainty_sum += certainty;
        samples += 1;
        outward += usize::from(signed_offset > 0.15);
        inward += usize::from(signed_offset < -0.15);
        left += usize::from(angle.cos() < -0.35);
        right += usize::from(angle.cos() > 0.35);
        lower += usize::from(angle.sin() > 0.30);
    }
    if samples < 8 || left < 2 || right < 2 || lower < 2 {
        return None;
    }

    let adjusted_indices = adjusted
        .iter()
        .enumerate()
        .filter_map(|(index, point)| point.map(|point| (index, point)))
        .collect::<Vec<_>>();
    let adjusted_points = adjusted_indices
        .iter()
        .map(|(_, point)| *point)
        .collect::<Vec<_>>();
    let adjusted_weights = adjusted_indices
        .iter()
        .map(|(index, _)| fit_weights[*index])
        .collect::<Vec<_>>();
    let proposed = fit_outer_ellipse_with_weights(&adjusted_points, Some(&adjusted_weights), seed);
    let initial_cost = analog_measurement_cost(&measurements, &fit_weights, initial_ellipse);
    let proposed_cost = analog_measurement_cost(&measurements, &fit_weights, proposed);
    let center_shift = (proposed[0] - initial_ellipse[0]).hypot(proposed[1] - initial_ellipse[1]);
    let axis_shift = (proposed[2] - initial_ellipse[2])
        .abs()
        .max((proposed[3] - initial_ellipse[3]).abs());
    let area_ratio = proposed[2] * proposed[3] / (initial_ellipse[2] * initial_ellipse[3]).max(1.0);
    let mean_power = power_sum / samples as f64;
    let mean_certainty = certainty_sum / samples as f64;
    let fit_applied = proposed.iter().all(|value| value.is_finite())
        && mean_power >= OUTER_IRIS_ANALOG_MIN_MEAN_POWER
        && mean_certainty >= OUTER_IRIS_ANALOG_MIN_MEAN_CERTAINTY
        && center_shift <= (seed[2] * 0.075).clamp(2.5, 7.0)
        && axis_shift <= (seed[2] * 0.090).clamp(3.0, 8.0)
        && (0.86..=1.16).contains(&area_ratio)
        && proposed_cost + 0.0025 < initial_cost;
    if !fit_applied {
        adjusted = *rays;
        fit_weights = [1.0; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
    }
    Some(OuterAnalogRefinement {
        rays: adjusted,
        fit_weights,
        elapsed: started.elapsed(),
        samples,
        outward,
        inward,
        mean_signed_offset_px: signed_sum / signed_weight.max(1.0e-9),
        mean_power,
        mean_certainty,
        fit_applied,
    })
}

fn outer_meridian_sector(index: usize) -> usize {
    ((outer_iris_evidence_angle(index).rem_euclid(2.0 * PI) / (PI * 0.25)).floor() as usize).min(7)
}

struct OuterRefinementResult {
    selected: [Option<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    elapsed: Duration,
    iterations: usize,
    sector_overruns: usize,
    opposing_supported: usize,
    selected_right: usize,
    selected_left: usize,
    selected_lower: usize,
}

fn finish_outer_refinement(
    selected: [Option<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    score_mode: OuterScoreMode,
    started: Instant,
    iterations: usize,
    sector_overruns: usize,
) -> OuterRefinementResult {
    let selected_right = selected
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            candidate.is_some() && outer_iris_evidence_angle(*index).cos() > 0.35
        })
        .count();
    let selected_left = selected
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            candidate.is_some() && outer_iris_evidence_angle(*index).cos() < -0.35
        })
        .count();
    let selected_lower = selected
        .iter()
        .enumerate()
        .filter(|(index, candidate)| {
            candidate.is_some() && outer_iris_evidence_angle(*index).sin() > 0.30
        })
        .count();
    let opposing_supported = selected
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| candidate.map(|candidate| (index, candidate)))
        .filter(|(index, candidate)| {
            opposing_meridian_support(candidates, *index, *candidate, score_mode) >= 0.20
        })
        .count();
    OuterRefinementResult {
        selected,
        elapsed: started.elapsed(),
        iterations,
        sector_overruns,
        opposing_supported,
        selected_right,
        selected_left,
        selected_lower,
    }
}

#[derive(Clone)]
struct OuterAffineHypothesis {
    ellipse: [f64; 5],
    selected: [Option<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    score: f64,
    support: usize,
    selected_left: usize,
    selected_right: usize,
    opposing_supported: usize,
}

fn outer_candidate_ellipse_residual(candidate: OuterRayCandidate, ellipse: [f64; 5]) -> f64 {
    let search = OuterSearchEllipse::from_fit(ellipse);
    let normalized = search.normalized_coordinates(candidate.point.x, candidate.point.y);
    (normalized.0.hypot(normalized.1) - 1.0).abs() * (ellipse[2] * ellipse[3]).sqrt()
}

fn opposing_meridian_support_for_ellipse(
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    index: usize,
    candidate: OuterRayCandidate,
    ellipse: [f64; 5],
    score_mode: OuterScoreMode,
) -> f64 {
    let residual_limit = ((ellipse[2] * ellipse[3]).sqrt() * 0.075).clamp(3.0, 10.0);
    opposing_outer_ray_indices(index)
        .into_iter()
        .filter(|opposite| *opposite < OUTER_IRIS_DENSE_EVIDENCE_SAMPLES)
        .filter_map(|opposite| {
            let maximum = candidates[opposite]
                .iter()
                .map(|other| score_mode.score(*other))
                .max_by(f64::total_cmp)?;
            candidates[opposite]
                .iter()
                .filter(|other| {
                    outer_candidate_ellipse_residual(**other, ellipse) <= residual_limit
                })
                .map(|other| {
                    let competitiveness =
                        (1.0 + (score_mode.score(*other) - maximum) / 2.5).clamp(0.0, 1.0);
                    let profile_disagreement = 0.12
                        * (other.pupil_void - candidate.pupil_void).abs()
                        + 0.16 * (other.inner_limbus_step - candidate.inner_limbus_step).abs()
                        + 0.24 * (other.iris_band - candidate.iris_band).abs()
                        + 0.24 * (other.sclera_out - candidate.sclera_out).abs()
                        + 0.24 * (other.far_sclera - candidate.far_sclera).abs();
                    let profile_agreement = (1.0 - profile_disagreement).clamp(0.0, 1.0);
                    let theory_strength = (0.5
                        + 0.18
                            * candidate
                                .meridian_profile_score
                                .min(other.meridian_profile_score))
                    .clamp(0.0, 1.0);
                    let opposite_reliability =
                        outer_meridian_reliability(outer_iris_evidence_angle(opposite));
                    let margin_agreement = if opposite_reliability < 0.25 {
                        0.62 + 0.38 * candidate.margin_clarity.max(other.margin_clarity)
                    } else {
                        (1.0 - (candidate.margin_clarity - other.margin_clarity).abs())
                            * (0.45 + 0.55 * candidate.margin_clarity.min(other.margin_clarity))
                    };
                    competitiveness
                        * (0.18
                            + 0.38 * profile_agreement
                            + 0.24 * theory_strength
                            + 0.20 * margin_agreement)
                })
                .max_by(f64::total_cmp)
        })
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

fn evaluate_outer_affine_hypothesis(
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    ellipse: [f64; 5],
    seed: [f64; 3],
    score_mode: OuterScoreMode,
    require_opposing: bool,
) -> Option<OuterAffineHypothesis> {
    let center_shift = (ellipse[0] - seed[0]).hypot(ellipse[1] - seed[1]);
    if !ellipse.iter().all(|value| value.is_finite())
        || ellipse[2] < seed[2] * 0.46
        || ellipse[2] > seed[2] * 1.58
        || ellipse[3] > ellipse[2]
        || !projected_circular_limbus_axes_plausible(ellipse[2], ellipse[3])
        || center_shift > seed[2] * 0.82
    {
        return None;
    }
    let mut selected = [None; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
    let mut score = 0.0;
    let mut support = 0usize;
    let mut selected_left = 0usize;
    let mut selected_right = 0usize;
    let mut opposing_supported = 0usize;
    for index in 0..OUTER_IRIS_DENSE_EVIDENCE_SAMPLES {
        let angle = outer_iris_evidence_angle(index);
        let reliability = outer_meridian_reliability(angle);
        if angle.sin() < -0.20 || reliability < 0.25 || candidates[index].is_empty() {
            continue;
        }
        let best = candidates[index]
            .iter()
            .copied()
            .filter_map(|candidate| {
                let residual = outer_candidate_ellipse_residual(candidate, ellipse).min(12.0);
                let opposite = require_opposing.then(|| {
                    opposing_meridian_support_for_ellipse(
                        candidates, index, candidate, ellipse, score_mode,
                    )
                });
                let value = reliability
                    * (score_mode.score(candidate)
                        + 0.70 * candidate.meridian_profile_score
                        + 0.55 * candidate.margin_clarity
                        + 1.10 * opposite.unwrap_or(0.0))
                    - 0.075 * residual * residual;
                (value > 0.0).then_some((candidate, value, opposite.unwrap_or(0.0)))
            })
            .max_by(|left, right| left.1.total_cmp(&right.1));
        let Some((candidate, value, opposite)) = best else {
            continue;
        };
        selected[index] = Some(candidate);
        score += value;
        support += 1;
        selected_left += usize::from(angle.cos() < -0.35);
        selected_right += usize::from(angle.cos() > 0.35);
        opposing_supported += usize::from(opposite >= 0.20);
    }
    if support < 8
        || selected_left < 2
        || selected_right < 2
        || (require_opposing && opposing_supported < 4)
    {
        return None;
    }
    let area_ratio = ellipse[2] * ellipse[3] / (seed[2] * seed[2]).max(1.0);
    score -= 0.80 * (center_shift / seed[2].max(1.0)).powi(2)
        + 0.30 * area_ratio.max(1.0e-3).ln().powi(2);
    Some(OuterAffineHypothesis {
        ellipse,
        selected,
        score,
        support,
        selected_left,
        selected_right,
        opposing_supported,
    })
}

fn recover_clipped_outer_affine_hypothesis(
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    seed: [f64; 3],
    score_mode: OuterScoreMode,
    _deadline: Instant,
) -> Option<OuterAffineHypothesis> {
    let mut best: Option<OuterAffineHypothesis> = None;
    for center_x_scale in [-0.32, 0.0, 0.32] {
        for center_y_scale in [0.28, 0.50, 0.70] {
            for major_scale in [1.0, 1.18, 1.34] {
                for axis_ratio in [0.74, 0.90] {
                    for angle in [-0.32, 0.0, 0.32] {
                        let ellipse = [
                            seed[0] + center_x_scale * seed[2],
                            seed[1] + center_y_scale * seed[2],
                            major_scale * seed[2],
                            major_scale * axis_ratio * seed[2],
                            angle,
                        ];
                        let Some(hypothesis) = evaluate_outer_affine_hypothesis(
                            candidates, ellipse, seed, score_mode, false,
                        ) else {
                            continue;
                        };
                        if best.as_ref().is_none_or(|old| hypothesis.score > old.score) {
                            best = Some(hypothesis);
                        }
                    }
                }
            }
        }
    }
    let mut best = best?;
    let mut steps = [
        seed[2] * 0.08,
        seed[2] * 0.08,
        seed[2] * 0.07,
        seed[2] * 0.07,
        0.10,
    ];
    for _ in 0..10 {
        for parameter in 0..5 {
            for direction in [-1.0, 1.0] {
                let mut ellipse = best.ellipse;
                ellipse[parameter] += direction * steps[parameter];
                let Some(hypothesis) =
                    evaluate_outer_affine_hypothesis(candidates, ellipse, seed, score_mode, false)
                else {
                    continue;
                };
                if hypothesis.score > best.score {
                    best = hypothesis;
                }
            }
        }
        for step in &mut steps {
            *step *= 0.62;
        }
    }
    evaluate_outer_affine_hypothesis(candidates, best.ellipse, seed, score_mode, true)
}

/// Alternates fuzzy ray intervals with a single projected ellipse. Each
/// selected point must explain the radial pupil/iris/limbus/sclera ordering,
/// agree with local arc neighbors, and (when visible) have a compatible hit
/// on the opposite half of its 3D meridian. The upper third can support or
/// contradict a hypothesis but can never become an authoritative point.
fn refine_outer_meridian_hypothesis(
    candidates: &[Vec<OuterRayCandidate>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    seed: [f64; 3],
    score_mode: OuterScoreMode,
    _system_deadline: Instant,
) -> OuterRefinementResult {
    let started = Instant::now();
    let initial = trace_outer_iris_curve(candidates, seed[2], score_mode, false);
    let initial_points = initial
        .iter()
        .enumerate()
        .filter(|(index, _)| outer_iris_evidence_angle(*index).sin() >= -0.20)
        .filter_map(|(_, candidate)| candidate.map(|candidate| candidate.point))
        .collect::<Vec<_>>();
    if initial_points.len() < 8 {
        let selected = initial.map(|candidate| {
            candidate.filter(|candidate| {
                candidate.margin_clarity >= 0.42 && candidate.point.y >= seed[1] - seed[2] * 0.20
            })
        });
        return finish_outer_refinement(selected, candidates, score_mode, started, 0, 0);
    }
    let mut ellipse = fit_outer_ellipse(&initial_points, seed);
    let mut selected = initial.map(|candidate| {
        candidate.filter(|candidate| {
            candidate.margin_clarity >= 0.32 && candidate.point.y >= seed[1] - seed[2] * 0.20
        })
    });
    // Lateral and lower sectors are evaluated before the two shadow-prone
    // upper sectors, but every sector completes. Wall-clock sector slices used
    // to leave a partially mutated iteration behind and made the fitted conic
    // depend on host scheduling rather than the RAW samples.
    const SECTOR_PRIORITY: [usize; 8] = [0, 3, 1, 2, 7, 4, 5, 6];
    let mut completed_iterations = 0usize;
    for _iteration in 0..OUTER_IRIS_REFINEMENT_ITERATIONS {
        let previous_ellipse = ellipse;
        let fitted_search = OuterSearchEllipse::from_fit(ellipse);
        let mut iteration_selected = selected;
        for sector in SECTOR_PRIORITY {
            for index in 0..OUTER_IRIS_DENSE_EVIDENCE_SAMPLES {
                if outer_meridian_sector(index) != sector {
                    continue;
                }
                let angle = outer_iris_evidence_angle(index);
                let reliability = outer_meridian_reliability(angle);
                if reliability < 0.25 || candidates[index].is_empty() {
                    iteration_selected[index] = None;
                    continue;
                }
                let mut scored = Vec::with_capacity(candidates[index].len());
                for candidate in candidates[index].iter().copied() {
                    let normalized =
                        fitted_search.normalized_coordinates(candidate.point.x, candidate.point.y);
                    let radial_residual = normalized.0.hypot(normalized.1) - 1.0;
                    let opposite =
                        opposing_meridian_support(candidates, index, candidate, score_mode);
                    let score = reliability
                        * (score_mode.score(candidate)
                            + 0.70 * candidate.meridian_profile_score
                            + 0.55 * candidate.margin_clarity
                            + 1.10 * opposite)
                        - 3.2 * (radial_residual.abs() / 0.10).powi(2).min(3.0);
                    scored.push((candidate, score, opposite));
                }
                let Some((best, best_score, opposite)) = scored
                    .iter()
                    .copied()
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                else {
                    iteration_selected[index] = None;
                    continue;
                };
                let fuzzy = scored
                    .iter()
                    .copied()
                    .filter_map(|(candidate, score, _)| {
                        (score >= best_score - 0.80
                            && (candidate.radius - best.radius).abs() <= seed[2] * 0.075)
                            .then_some((candidate, (score - best_score + 0.85).max(0.05)))
                    })
                    .collect::<Vec<_>>();
                let total_weight = fuzzy
                    .iter()
                    .map(|(_, weight)| weight)
                    .sum::<f64>()
                    .max(1.0e-6);
                let mean_radius = fuzzy
                    .iter()
                    .map(|(candidate, weight)| candidate.radius * weight)
                    .sum::<f64>()
                    / total_weight;
                let interval_width = fuzzy
                    .iter()
                    .map(|(candidate, _)| (candidate.radius - mean_radius).abs())
                    .max_by(f64::total_cmp)
                    .unwrap_or(0.0);
                let confidence = (0.22 + 0.38 * best.margin_clarity + 0.25 * opposite
                    - 0.22 * (interval_width / (seed[2] * 0.075).max(1.0)))
                .clamp(0.0, 1.0);
                let lateral_requires_opposite = angle.sin().abs() < 0.35;
                let meridian_supported = !lateral_requires_opposite
                    || opposite >= 0.14
                    || (best.margin_clarity >= 0.74 && best.meridian_profile_score >= 0.18);
                // The entire upper third remains a hypothesis/veto source only.
                iteration_selected[index] = (angle.sin() >= -0.20
                    && confidence >= 0.32
                    && meridian_supported)
                    .then_some(OuterRayCandidate {
                        point: OuterIrisPoint {
                            contrast: best.point.contrast * confidence,
                            ..best.point
                        },
                        ..best
                    });
            }
        }
        let points = iteration_selected
            .iter()
            .flatten()
            .map(|candidate| candidate.point)
            .collect::<Vec<_>>();
        let right = iteration_selected
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                candidate.is_some() && outer_iris_evidence_angle(*index).cos() > 0.35
            })
            .count();
        let left = iteration_selected
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                candidate.is_some() && outer_iris_evidence_angle(*index).cos() < -0.35
            })
            .count();
        let lower = iteration_selected
            .iter()
            .enumerate()
            .filter(|(index, candidate)| {
                candidate.is_some() && outer_iris_evidence_angle(*index).sin() > 0.30
            })
            .count();
        if points.len() < 8 || right < 2 || left < 2 || lower < 2 {
            break;
        }
        selected = iteration_selected;
        ellipse = fit_outer_ellipse(&points, seed);
        completed_iterations += 1;
        let movement = (ellipse[0] - previous_ellipse[0]).hypot(ellipse[1] - previous_ellipse[1])
            + (ellipse[2] - previous_ellipse[2]).abs()
            + (ellipse[3] - previous_ellipse[3]).abs();
        if movement < 0.35 {
            break;
        }
    }
    finish_outer_refinement(
        selected,
        candidates,
        score_mode,
        started,
        completed_iterations,
        0,
    )
}

fn mark_coherent_lower_lid_occlusion_runs(
    rays: &[Option<OuterIrisPoint>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    seed: [f64; 3],
    lower_eyelid: &[BorderPoint],
    reject: &mut [bool; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
) {
    // A detected lid path is already a long, ordered, direct edge measurement.
    // Do not extrapolate it past its measured horizontal span: the clamped
    // eyelid interpolation is useful for stopping rays, but would manufacture
    // a horizontal occluder outside the actual lid observations here.
    if lower_eyelid.len() < 7 || seed[2] <= 1.0 {
        return;
    }
    let first_x = lower_eyelid.first().map_or(0.0, |point| point.x as f64);
    let last_x = lower_eyelid.last().map_or(0.0, |point| point.x as f64);
    if last_x - first_x < seed[2] * 0.32 {
        return;
    }

    let maximum_gap = (seed[2] * 0.22).clamp(6.0, 18.0);
    let mut near_lid = Vec::<(usize, f64, OuterIrisPoint, f64)>::new();
    for (index, point) in rays.iter().copied().enumerate() {
        let Some(point) = point else {
            continue;
        };
        let angle = outer_iris_evidence_angle(index);
        if angle.sin() <= 0.12 || point.x < first_x || point.x > last_x {
            continue;
        }
        let Some(lid_y) = eyelid_y_at_x(lower_eyelid, point.x) else {
            continue;
        };
        let gap = lid_y - point.y;
        if (-1.5..=maximum_gap).contains(&gap) {
            near_lid.push((index, angle, point, gap));
        }
    }
    if near_lid.len() < 3 {
        return;
    }

    let point_line_distance = |point: (f64, f64), start: (f64, f64), end: (f64, f64)| {
        let chord = (end.0 - start.0, end.1 - start.1);
        let length = chord.0.hypot(chord.1);
        if length < 1.0e-6 {
            f64::INFINITY
        } else {
            ((point.0 - start.0) * chord.1 - (point.1 - start.1) * chord.0).abs() / length
        }
    };
    let qualifies = |run: &[(usize, f64, OuterIrisPoint, f64)]| {
        if run.len() < 3 {
            return false;
        }
        let first = run.first().unwrap();
        let last = run.last().unwrap();
        let x_span = (last.2.x - first.2.x).abs();
        let angle_span = angular_distance(first.1, last.1);
        if x_span < (seed[2] * 0.28).clamp(8.0, 30.0) || angle_span < 0.30 {
            return false;
        }

        // A real visible limbus and the lid have different curvature. An
        // occluding chord instead keeps an almost constant offset from the
        // independently measured lid path over several meridians.
        let mut gaps = run.iter().map(|sample| sample.3).collect::<Vec<_>>();
        let gap_center = median(&mut gaps);
        let mut gap_deviations = run
            .iter()
            .map(|sample| (sample.3 - gap_center).abs())
            .collect::<Vec<_>>();
        let gap_mad = 1.4826 * median(&mut gap_deviations);
        if gap_mad > (seed[2] * 0.045).clamp(1.5, 4.0) {
            return false;
        }

        let Some(first_lid_y) = eyelid_y_at_x(lower_eyelid, first.2.x) else {
            return false;
        };
        let Some(last_lid_y) = eyelid_y_at_x(lower_eyelid, last.2.x) else {
            return false;
        };
        let point_tangent = (last.2.x - first.2.x, last.2.y - first.2.y);
        let lid_tangent = (last.2.x - first.2.x, last_lid_y - first_lid_y);
        let tangent_denominator =
            point_tangent.0.hypot(point_tangent.1) * lid_tangent.0.hypot(lid_tangent.1);
        if tangent_denominator <= 1.0e-6
            || (point_tangent.0 * lid_tangent.0 + point_tangent.1 * lid_tangent.1).abs()
                / tangent_denominator
                < 0.94
        {
            return false;
        }

        // Require the measured run to have substantially less sagitta than a
        // circular limbus would have across the same meridian angles. This is
        // the explicit flat-tire test and protects a close but still-visible
        // curved limbus from the near-lid proximity rule above.
        let measured_start = (first.2.x, first.2.y);
        let measured_end = (last.2.x, last.2.y);
        let expected_start = (
            seed[0] + seed[2] * first.1.cos(),
            seed[1] + seed[2] * first.1.sin(),
        );
        let expected_end = (
            seed[0] + seed[2] * last.1.cos(),
            seed[1] + seed[2] * last.1.sin(),
        );
        let measured_sagitta = run
            .iter()
            .map(|sample| {
                point_line_distance((sample.2.x, sample.2.y), measured_start, measured_end)
            })
            .max_by(f64::total_cmp)
            .unwrap_or(f64::INFINITY);
        let expected_sagitta = run
            .iter()
            .map(|sample| {
                point_line_distance(
                    (
                        seed[0] + seed[2] * sample.1.cos(),
                        seed[1] + seed[2] * sample.1.sin(),
                    ),
                    expected_start,
                    expected_end,
                )
            })
            .max_by(f64::total_cmp)
            .unwrap_or(0.0);
        expected_sagitta >= 1.0 && measured_sagitta <= (expected_sagitta * 0.62).max(1.5)
    };

    // Work-budget decimation can leave every second or fourth ray absent.
    // Group by angular/geometric continuity instead of array adjacency so a
    // stable lid chord remains detectable at every bounded-work stride.
    let mut run_start = 0usize;
    for boundary in 1..=near_lid.len() {
        let split = boundary == near_lid.len()
            || angular_distance(near_lid[boundary - 1].1, near_lid[boundary].1) > 0.55
            || (near_lid[boundary - 1].2.x - near_lid[boundary].2.x)
                .hypot(near_lid[boundary - 1].2.y - near_lid[boundary].2.y)
                > seed[2] * 0.60;
        if !split {
            continue;
        }
        let run = &near_lid[run_start..boundary];
        if qualifies(run) {
            for &(index, _, _, _) in run {
                reject[index] = true;
            }
        }
        run_start = boundary;
    }
}

fn reject_outer_flat_tire_runs(
    rays: &mut [Option<OuterIrisPoint>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    preliminary_ellipse: [f64; 5],
    seed: [f64; 3],
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
) -> Vec<OuterIrisPoint> {
    let seed_radius = seed[2];
    let normalized = OuterSearchEllipse::from_fit(preliminary_ellipse);
    let mut flat = [false; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
    for index in 0..OUTER_IRIS_DENSE_EVIDENCE_SAMPLES {
        let angle = outer_iris_evidence_angle(index);
        // Test curvature in every projected dimension. Reliable lateral rays
        // are still protected by the adjacent-run requirement below, while a
        // flat chord can no longer survive merely by rotating toward them.
        let previous_index =
            (index + OUTER_IRIS_DENSE_EVIDENCE_SAMPLES - 2) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES;
        let next_index = (index + 2) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES;
        let (Some(previous_point), Some(current_point), Some(next_point)) =
            (rays[previous_index], rays[index], rays[next_index])
        else {
            continue;
        };
        let previous = normalized.normalized_coordinates(previous_point.x, previous_point.y);
        let current = normalized.normalized_coordinates(current_point.x, current_point.y);
        let next = normalized.normalized_coordinates(next_point.x, next_point.y);
        let incoming = (current.0 - previous.0, current.1 - previous.1);
        let outgoing = (next.0 - current.0, next.1 - current.1);
        let incoming_length = incoming.0.hypot(incoming.1);
        let outgoing_length = outgoing.0.hypot(outgoing.1);
        if incoming_length < 1.0e-5 || outgoing_length < 1.0e-5 {
            continue;
        }
        let measured_turn = (incoming.0 * outgoing.1 - incoming.1 * outgoing.0).abs()
            / (incoming_length * outgoing_length);
        let previous_angle = outer_iris_evidence_angle(previous_index);
        let next_angle = outer_iris_evidence_angle(next_index);
        let previous_span = (angle - previous_angle).rem_euclid(2.0 * PI);
        let next_span = (next_angle - angle).rem_euclid(2.0 * PI);
        let expected_turn = ((previous_span + next_span) * 0.5).sin().abs();
        let curvature_flat = expected_turn > 0.04 && measured_turn < expected_turn * 0.34;

        let point_line_distance = |point: (f64, f64), start: (f64, f64), end: (f64, f64)| {
            let chord = (end.0 - start.0, end.1 - start.1);
            let length = chord.0.hypot(chord.1);
            if length < 1.0e-6 {
                0.0
            } else {
                ((point.0 - start.0) * chord.1 - (point.1 - start.1) * chord.0).abs() / length
            }
        };
        let expected_current = (angle.cos(), angle.sin());
        let expected_previous = (previous_angle.cos(), previous_angle.sin());
        let expected_next = (next_angle.cos(), next_angle.sin());
        let measured_sagitta = point_line_distance(current, previous, next);
        let expected_sagitta =
            point_line_distance(expected_current, expected_previous, expected_next);
        let narrow_sagitta_flat =
            expected_sagitta > 0.004 && measured_sagitta < expected_sagitta * 0.38;

        // A lid/reflection chord can bend slightly and pass the local turn
        // test while remaining far too flat over a larger arc. Compare a
        // second normalized sagitta four rays to either side, so this catches
        // the "roundish flat tire" in every affine-projected dimension.
        let wide_previous_index =
            (index + OUTER_IRIS_DENSE_EVIDENCE_SAMPLES - 4) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES;
        let wide_next_index = (index + 4) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES;
        let wide_sagitta_flat = rays[wide_previous_index]
            .zip(rays[wide_next_index])
            .is_some_and(|(wide_previous_point, wide_next_point)| {
                let wide_previous =
                    normalized.normalized_coordinates(wide_previous_point.x, wide_previous_point.y);
                let wide_next =
                    normalized.normalized_coordinates(wide_next_point.x, wide_next_point.y);
                let wide_previous_angle = outer_iris_evidence_angle(wide_previous_index);
                let wide_next_angle = outer_iris_evidence_angle(wide_next_index);
                let measured = point_line_distance(current, wide_previous, wide_next);
                let expected = point_line_distance(
                    expected_current,
                    (wide_previous_angle.cos(), wide_previous_angle.sin()),
                    (wide_next_angle.cos(), wide_next_angle.sin()),
                );
                expected > 0.012 && measured < expected * 0.45
            });

        let chain_tangent = (
            next_point.x - previous_point.x,
            next_point.y - previous_point.y,
        );
        let lid_overlap_alias = |lid: &[BorderPoint], vertical_distance: f64| {
            if lid.len() < 3
                || !(0.0..=(seed_radius * 0.11).clamp(3.0, 9.0)).contains(&vertical_distance)
            {
                return false;
            }
            let tangent_span = (seed_radius * 0.10).clamp(5.0, 10.0);
            let Some(left_y) = eyelid_y_at_x(lid, current_point.x - tangent_span) else {
                return false;
            };
            let Some(right_y) = eyelid_y_at_x(lid, current_point.x + tangent_span) else {
                return false;
            };
            let lid_tangent = (2.0 * tangent_span, right_y - left_y);
            let denominator =
                chain_tangent.0.hypot(chain_tangent.1) * lid_tangent.0.hypot(lid_tangent.1);
            denominator > 1.0e-6
                && (chain_tangent.0 * lid_tangent.0 + chain_tangent.1 * lid_tangent.1).abs()
                    / denominator
                    >= 0.90
        };
        let overlap_alias = if angle.sin() > 0.0 {
            eyelid_y_at_x(lower_eyelid, current_point.x)
                .is_some_and(|lid_y| lid_overlap_alias(lower_eyelid, lid_y - current_point.y))
        } else {
            eyelid_y_at_x(upper_eyelid, current_point.x)
                .is_some_and(|lid_y| lid_overlap_alias(upper_eyelid, current_point.y - lid_y))
        };
        flat[index] = curvature_flat || narrow_sagitta_flat || wide_sagitta_flat || overlap_alias;
    }

    // A flat tire is a run, not one noisy sample.  Require an adjacent flat
    // vote and expand by one ray so the straight segment's endpoints do not
    // continue pulling the second fit toward the lid.
    let mut reject = [false; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES];
    for index in 0..OUTER_IRIS_DENSE_EVIDENCE_SAMPLES {
        if !flat[index] {
            continue;
        }
        let previous =
            (index + OUTER_IRIS_DENSE_EVIDENCE_SAMPLES - 1) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES;
        let next = (index + 1) % OUTER_IRIS_DENSE_EVIDENCE_SAMPLES;
        if flat[previous] || flat[next] {
            reject[previous] = true;
            reject[index] = true;
            reject[next] = true;
        }
    }
    mark_coherent_lower_lid_occlusion_runs(rays, seed, lower_eyelid, &mut reject);
    let rejected_count = rays
        .iter()
        .zip(reject)
        .filter(|(point, reject)| point.is_some() && *reject)
        .count();
    let retained = rays.iter().filter(|point| point.is_some()).count() - rejected_count;
    if rejected_count == 0 || retained < 8 {
        return Vec::new();
    }
    let mut rejected_points = Vec::with_capacity(rejected_count);
    for (point, reject) in rays.iter_mut().zip(reject) {
        if reject {
            if let Some(point) = point.take() {
                rejected_points.push(point);
            }
        }
    }
    rejected_points
}

struct OuterBranchResult {
    ellipse: [f64; 5],
    rays: [Option<OuterIrisPoint>; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
    evidence: Vec<OuterIrisPoint>,
    occluded_points: Vec<OuterIrisPoint>,
    curve_confidence: [f64; OUTER_IRIS_DENSE_EVIDENCE_SAMPLES],
}

#[allow(clippy::too_many_arguments)]
fn run_outer_iris_branch(
    luma: Arc<BoxLuma5>,
    native: Option<Arc<NativeLogPlane>>,
    sclera_probability: Option<Arc<Vec<f64>>>,
    reflectance_sclera_probability: Option<Arc<Vec<f64>>>,
    material_illumination: Option<MaterialIlluminationModel>,
    upper_eyelid: Arc<Vec<BorderPoint>>,
    lower_eyelid: Arc<Vec<BorderPoint>>,
    width: usize,
    height: usize,
    seed: [f64; 3],
    initial_search: OuterSearchEllipse,
    score_mode: OuterScoreMode,
    minimum_search_scale: f64,
    work_stride: usize,
    sample_stride: usize,
    system_deadline: Instant,
    diagnostics: &mut OuterIrisDiagnostics,
) -> Option<OuterBranchResult> {
    let rough_search = initial_search;
    let luma_gate = estimate_luma_transition_gate(luma.as_ref());
    let context = Arc::new(OuterRayContext {
        luma: Arc::clone(&luma),
        native,
        sclera_probability,
        reflectance_sclera_probability,
        material_illumination,
        upper_eyelid: Arc::clone(&upper_eyelid),
        lower_eyelid: Arc::clone(&lower_eyelid),
        luma_gate,
        width,
        height,
        search: initial_search,
        rough_search,
        scale_range: (
            minimum_search_scale.clamp(0.35, OUTER_IRIS_MAX_SEARCH_SCALE),
            OUTER_IRIS_MAX_SEARCH_SCALE,
        ),
    });
    let ray_batch = evaluate_outer_iris_rays(Arc::clone(&context), work_stride, sample_stride);
    diagnostics.ray_batch_elapsed_us += ray_batch.elapsed.as_micros().min(u64::MAX as u128) as u64;
    diagnostics.max_ray_elapsed_us = diagnostics
        .max_ray_elapsed_us
        .max(ray_batch.max_ray_elapsed.as_micros().min(u64::MAX as u128) as u64);
    diagnostics.active_rays += ray_batch.active_rays;
    diagnostics.candidate_rays += ray_batch.candidate_rays;
    diagnostics.candidate_count += ray_batch.candidate_count;
    diagnostics.ray_overruns += ray_batch.ray_overruns;
    diagnostics.ray_batch_timeouts += ray_batch.batch_budget_overruns;
    let candidates = ray_batch.candidates;
    let recovery = || {
        trace_outer_iris_curve(
            &candidates,
            initial_search.equivalent_radius(),
            score_mode,
            false,
        )
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            candidate
                .filter(|candidate| {
                    outer_iris_evidence_angle(index).sin() >= -0.20
                        && candidate.margin_clarity >= 0.25
                        && candidate.meridian_profile_score >= -0.80
                })
                .map(|candidate| candidate.point)
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap_or_else(|_| std::array::from_fn(|_| None))
    };
    let refinement =
        refine_outer_meridian_hypothesis(&candidates, seed, score_mode, system_deadline);
    diagnostics.refinement_elapsed_us +=
        refinement.elapsed.as_micros().min(u64::MAX as u128) as u64;
    diagnostics.refinement_iterations += refinement.iterations;
    diagnostics.sector_overruns += refinement.sector_overruns;
    diagnostics.opposing_supported = diagnostics
        .opposing_supported
        .max(refinement.opposing_supported);
    diagnostics.selected_right = diagnostics.selected_right.max(refinement.selected_right);
    diagnostics.selected_left = diagnostics.selected_left.max(refinement.selected_left);
    diagnostics.selected_lower = diagnostics.selected_lower.max(refinement.selected_lower);
    let mut rays = refinement
        .selected
        .map(|candidate| candidate.map(|candidate| candidate.point));
    if rays.iter().flatten().count() < 8 {
        rays = recovery();
    }
    let initial_evidence = rays.iter().flatten().copied().collect::<Vec<_>>();
    if initial_evidence.len() < 8 {
        return None;
    }
    let preliminary = fit_outer_ellipse(&initial_evidence, seed);
    let mut occluded_points = reject_outer_flat_tire_runs(
        &mut rays,
        preliminary,
        seed,
        upper_eyelid.as_ref(),
        lower_eyelid.as_ref(),
    );
    diagnostics.flat_rejected += occluded_points.len();
    if rays.iter().flatten().count() < 8 {
        rays = recovery();
        let recovery_preliminary =
            fit_outer_ellipse(&rays.iter().flatten().copied().collect::<Vec<_>>(), seed);
        occluded_points = reject_outer_flat_tire_runs(
            &mut rays,
            recovery_preliminary,
            seed,
            upper_eyelid.as_ref(),
            lower_eyelid.as_ref(),
        );
        diagnostics.flat_rejected += occluded_points.len();
    }
    let preanalog_evidence = rays.iter().flatten().copied().collect::<Vec<_>>();
    if preanalog_evidence.len() < 8 {
        return None;
    }
    let right = rays
        .iter()
        .enumerate()
        .filter(|(index, point)| point.is_some() && outer_iris_evidence_angle(*index).cos() > 0.35)
        .count();
    let left = rays
        .iter()
        .enumerate()
        .filter(|(index, point)| point.is_some() && outer_iris_evidence_angle(*index).cos() < -0.35)
        .count();
    let lower = rays
        .iter()
        .enumerate()
        .filter(|(index, point)| point.is_some() && outer_iris_evidence_angle(*index).sin() > 0.30)
        .count();
    // Test the terminal eye-edge topology only after the meridian road has
    // converged. Sampling every radial proposal made this cue both expensive
    // and strong enough to steer a good conic before it had any global
    // context. Here it is a bounded second-order consistency measurement on
    // at most the 79 accepted full-resolution RAW contacts.
    let topology_ellipse = fit_outer_ellipse(&preanalog_evidence, seed);
    let topology_equivalent_radius = (topology_ellipse[2] * topology_ellipse[3]).sqrt().max(1.0);
    let selected_lateral_topology = rays
        .iter()
        .enumerate()
        .filter_map(|(index, point)| point.map(|point| (index, point)))
        .filter(|(index, _)| outer_iris_evidence_angle(*index).cos().abs() > 0.35)
        .map(|(index, point)| {
            // The terminal eye margin is an independent scene cue. Deriving
            // this direction from the candidate conic lets a wrong, rotated
            // lid-sized ellipse point its own probe into an eyelid and
            // self-justify. On authoritative lateral contacts, "outward" is
            // the horizontal route from the eye center toward the canthus.
            let outward_normal = if point.x < topology_ellipse[0] {
                (-1.0, 0.0)
            } else {
                (1.0, 0.0)
            };
            (
                index,
                point,
                sample_outward_eye_edge_topology(
                    context.as_ref(),
                    point,
                    outward_normal,
                    topology_equivalent_radius,
                ),
            )
        })
        .filter(|(_, _, topology)| topology.observable)
        .collect::<Vec<_>>();
    diagnostics.outward_topology_observable = selected_lateral_topology.len();
    diagnostics.outward_topology_supported = selected_lateral_topology
        .iter()
        .filter(|(_, _, topology)| topology.score >= 0.28)
        .count();
    let side_topology = |left: bool| {
        let side = selected_lateral_topology
            .iter()
            .filter(|(index, _, _)| (outer_iris_evidence_angle(*index).cos() < 0.0) == left)
            .collect::<Vec<_>>();
        let supported = side
            .iter()
            .filter(|(_, _, topology)| topology.score >= 0.28)
            .count();
        let mean_score = side
            .iter()
            .map(|(_, _, topology)| topology.score)
            .sum::<f64>()
            / side.len().max(1) as f64;
        let mean_limbus_order = side
            .iter()
            .map(|(_, _, topology)| topology.limbus_order_score)
            .sum::<f64>()
            / side.len().max(1) as f64;
        let supported_ridges = side
            .iter()
            .filter(|(_, _, topology)| topology.score >= 0.28)
            .collect::<Vec<_>>();
        let mean_ridge_distance = supported_ridges
            .iter()
            .map(|(_, _, topology)| topology.ridge_distance_px)
            .sum::<f64>()
            / supported_ridges.len().max(1) as f64;
        let mut longest_coherent_run = 0usize;
        let mut current_run = 0usize;
        let mut previous: Option<(usize, f64)> = None;
        for (index, _, topology) in &side {
            if topology.score < 0.28 {
                current_run = 0;
                previous = None;
                continue;
            }
            let coherent = previous.is_some_and(|(previous_index, previous_distance)| {
                index.saturating_sub(previous_index) <= 2
                    && (topology.ridge_distance_px - previous_distance).abs() <= 6.0
            });
            current_run = if coherent { current_run + 1 } else { 1 };
            longest_coherent_run = longest_coherent_run.max(current_run);
            previous = Some((*index, topology.ridge_distance_px));
        }
        (
            side.len(),
            supported,
            mean_score,
            mean_limbus_order,
            mean_ridge_distance,
            longest_coherent_run,
        )
    };
    let (
        observable_left,
        supported_left,
        mean_left,
        mean_order_left,
        mean_ridge_left,
        longest_left,
    ) = side_topology(true);
    let (
        observable_right,
        supported_right,
        mean_right,
        mean_order_right,
        mean_ridge_right,
        longest_right,
    ) = side_topology(false);
    diagnostics.outward_topology_observable_left = observable_left;
    diagnostics.outward_topology_supported_left = supported_left;
    diagnostics.outward_topology_observable_right = observable_right;
    diagnostics.outward_topology_supported_right = supported_right;
    diagnostics.outward_topology_mean_score_left = mean_left;
    diagnostics.outward_topology_mean_score_right = mean_right;
    diagnostics.outward_topology_mean_limbus_order_left = mean_order_left;
    diagnostics.outward_topology_mean_limbus_order_right = mean_order_right;
    diagnostics.outward_topology_mean_ridge_distance_left_px = mean_ridge_left;
    diagnostics.outward_topology_mean_ridge_distance_right_px = mean_ridge_right;
    diagnostics.outward_topology_longest_coherent_run_left = longest_left;
    diagnostics.outward_topology_longest_coherent_run_right = longest_right;
    let final_opposing_supported = rays
        .iter()
        .enumerate()
        .filter_map(|(index, point)| point.map(|point| (index, point)))
        .filter(|(index, point)| {
            candidates[*index]
                .iter()
                .copied()
                .min_by(|left, right| {
                    (left.point.x - point.x)
                        .hypot(left.point.y - point.y)
                        .total_cmp(&(right.point.x - point.x).hypot(right.point.y - point.y))
                })
                .is_some_and(|candidate| {
                    opposing_meridian_support(&candidates, *index, candidate, score_mode) >= 0.20
                })
        })
        .count();
    diagnostics.selected_right = right;
    diagnostics.selected_left = left;
    diagnostics.selected_lower = lower;
    diagnostics.opposing_supported = final_opposing_supported;
    let bilateral = right >= 2 && left >= 2 && lower >= 2 && final_opposing_supported >= 2;
    // Occlusion theory permits one missing lateral side only when the other
    // lateral arc and lower-middle form a long, strongly sampled curve. This
    // is the expected geometry for a glint/eyelash reflection erasing one
    // half-meridian; a short lid chord cannot meet the 24-ray coverage
    // threshold. A second recovery covers a clipped lower edge when many
    // genuinely opposing lateral meridians remain.
    let unilateral_occlusion =
        (right >= 10 || left >= 10) && lower >= 10 && preanalog_evidence.len() >= 24;
    let lower_occlusion =
        right >= 6 && left >= 6 && final_opposing_supported >= 6 && preanalog_evidence.len() >= 16;
    diagnostics.occlusion_recovered = !bilateral && (unilateral_occlusion || lower_occlusion);
    if !bilateral && !diagnostics.occlusion_recovered {
        return None;
    }
    let analog = refine_outer_analog_edge_forces(
        context.as_ref(),
        &candidates,
        &rays,
        seed,
        score_mode,
        system_deadline,
    );
    let mut analog_weights = None;
    if let Some(analog) = analog {
        diagnostics.analog_force_samples += analog.samples;
        diagnostics.analog_force_outward += analog.outward;
        diagnostics.analog_force_inward += analog.inward;
        diagnostics.analog_mean_signed_offset_px = analog.mean_signed_offset_px;
        diagnostics.analog_mean_power = analog.mean_power;
        diagnostics.analog_mean_certainty = analog.mean_certainty;
        diagnostics.analog_refinement_elapsed_us +=
            analog.elapsed.as_micros().min(u64::MAX as u128) as u64;
        diagnostics.analog_fit_applied |= analog.fit_applied;
        if analog.fit_applied {
            rays = analog.rays;
            analog_weights = Some(analog.fit_weights);
        }
    }
    let evidence = rays.iter().flatten().copied().collect::<Vec<_>>();
    let evidence_weights = analog_weights.as_ref().map(|weights| {
        rays.iter()
            .enumerate()
            .filter_map(|(index, point)| point.map(|_| weights[index]))
            .collect::<Vec<_>>()
    });
    let fitted = fit_outer_ellipse_with_weights(&evidence, evidence_weights.as_deref(), seed);
    let (ellipse, cohesive, curve_confidence) =
        fit_cohesive_outer_groups(&rays, analog_weights.as_ref(), seed, fitted);
    // When the predicted lower limbus is outside the crop, the normal local
    // refinement can lock onto a one-sided inner texture arc: the correct
    // translated ellipse has deliberately non-constant radius around the bad
    // coarse center. Run the wider five-parameter beam only if the completed
    // normal fit has no high-confidence bilateral support. This keeps both its
    // cost and its authority out of ordinary well-acquired frames.
    let rough_lower_y = initial_search.point_and_normal(PI * 0.5, 1.0).0 .1;
    let normal_bilateral =
        evaluate_outer_affine_hypothesis(&candidates, ellipse, seed, score_mode, false).is_some();
    if !normal_bilateral && rough_lower_y >= height.saturating_sub(12) as f64 {
        if let Some(recovery) =
            recover_clipped_outer_affine_hypothesis(&candidates, seed, score_mode, system_deadline)
        {
            let mut recovery_rays = recovery
                .selected
                .map(|candidate| candidate.map(|c| c.point));
            let recovery_occluded_points = reject_outer_flat_tire_runs(
                &mut recovery_rays,
                recovery.ellipse,
                seed,
                upper_eyelid.as_ref(),
                lower_eyelid.as_ref(),
            );
            diagnostics.flat_rejected += recovery_occluded_points.len();
            let recovery_evidence = recovery_rays.iter().flatten().copied().collect::<Vec<_>>();
            if recovery_evidence.len() >= 8
                && recovery.score / recovery.support.max(1) as f64 >= 1.35
            {
                diagnostics.selected_left = recovery.selected_left;
                diagnostics.selected_right = recovery.selected_right;
                diagnostics.selected_lower = recovery
                    .selected
                    .iter()
                    .enumerate()
                    .filter(|(index, candidate)| {
                        candidate.is_some() && outer_iris_evidence_angle(*index).sin() > 0.30
                    })
                    .count();
                diagnostics.opposing_supported = recovery.opposing_supported;
                diagnostics.occlusion_recovered = true;
                let recovery_confidence = std::array::from_fn(|index| {
                    recovery_rays[index]
                        .and(recovery.selected[index])
                        .map_or(0.0, |candidate| candidate.margin_clarity)
                });
                return Some(OuterBranchResult {
                    ellipse: recovery.ellipse,
                    rays: recovery_rays,
                    evidence: recovery_evidence,
                    occluded_points: recovery_occluded_points,
                    curve_confidence: recovery_confidence,
                });
            }
        }
    }
    (cohesive.len() >= 8).then_some(OuterBranchResult {
        ellipse,
        rays,
        evidence,
        occluded_points,
        curve_confidence,
    })
}

fn outer_geometry_plausible(ellipse: [f64; 5], rough: OuterSearchEllipse) -> bool {
    let area_ratio = ellipse[2] * ellipse[3] / (rough.major_radius * rough.minor_radius).max(1.0);
    let center_shift = (ellipse[0] - rough.center.0).hypot(ellipse[1] - rough.center.1);
    projected_circular_limbus_axes_plausible(ellipse[2], ellipse[3])
        && (0.72..=1.32).contains(&area_ratio)
        && center_shift <= rough.major_radius * 0.42
}

fn estimate_lateral_sclera_color(
    native: &NativeLogPlane,
    seed: [f64; 3],
) -> Option<ScleraColorReference> {
    let mut samples = Vec::with_capacity(18);
    for side in [0.0, PI] {
        for angle_offset in [-0.24, 0.0, 0.24] {
            let angle: f64 = side + angle_offset;
            let (sin, cos) = angle.sin_cos();
            for radius_scale in [1.08, 1.16, 1.24] {
                samples.push(native.sample_chroma(
                    seed[0] + seed[2] * radius_scale * cos,
                    seed[1] + seed[2] * radius_scale * sin,
                ));
            }
        }
    }
    if samples.len() < 8 {
        return None;
    }
    let mut rg = samples.iter().map(|sample| sample.0).collect::<Vec<_>>();
    let mut bg = samples.iter().map(|sample| sample.1).collect::<Vec<_>>();
    let log_rg = median(&mut rg);
    let log_bg = median(&mut bg);
    let mut deviations = samples
        .iter()
        .map(|sample| (sample.0 - log_rg).hypot(sample.1 - log_bg))
        .collect::<Vec<_>>();
    let tolerance = (3.5 * 1.4826 * median(&mut deviations)).clamp(0.10, 0.42);
    Some(ScleraColorReference {
        log_rg,
        log_bg,
        tolerance,
    })
}

fn outer_tip_matches_sclera_color(
    native: &NativeLogPlane,
    reference: ScleraColorReference,
    point: OuterIrisPoint,
    angle: f64,
) -> bool {
    let (direction_y, direction_x) = angle.sin_cos();
    let sample = native.sample_chroma(point.x + direction_x * 8.0, point.y + direction_y * 8.0);
    (sample.0 - reference.log_rg).hypot(sample.1 - reference.log_bg) <= reference.tolerance
}

fn outer_tip_has_transverse_edge(luma: &BoxLuma5, point: OuterIrisPoint, angle: f64) -> bool {
    let (direction_y, direction_x) = angle.sin_cos();
    let tangent = (-direction_y, direction_x);
    [0.0, 7.0].into_iter().any(|radial_offset| {
        let center_x = point.x + direction_x * radial_offset;
        let center_y = point.y + direction_y * radial_offset;
        let center = luma.sample(center_x, center_y);
        let before = luma.sample(center_x - tangent.0 * 4.0, center_y - tangent.1 * 4.0);
        let after = luma.sample(center_x + tangent.0 * 4.0, center_y + tangent.1 * 4.0);
        let tangent_gradient = (after - before).abs();
        let tangent_laplacian = (before + after - 2.0 * center).abs();
        let relative_threshold = point.contrast * 0.72;
        (tangent_gradient >= 42.0 && tangent_gradient >= relative_threshold)
            || (tangent_laplacian >= 52.0 && tangent_laplacian >= relative_threshold)
    })
}

fn best_outer_iris_candidate(
    luma: &BoxLuma5,
    width: usize,
    height: usize,
    seed: [f64; 3],
    angle: f64,
) -> Option<OuterIrisPoint> {
    best_outer_iris_candidate_scored(luma, width, height, seed, angle)
        .filter(|(score, _)| *score > 1.0)
        .map(|(_, point)| point)
}

fn best_outer_iris_candidate_scored(
    luma: &BoxLuma5,
    width: usize,
    height: usize,
    seed: [f64; 3],
    angle: f64,
) -> Option<(f64, OuterIrisPoint)> {
    best_outer_iris_candidate_scored_with_prior(luma, width, height, seed, angle, None)
}

#[derive(Clone, Copy, Debug)]
struct OuterRadialPrior {
    radius: f64,
    half_width: f64,
    strength: f64,
}

fn outer_radial_search_range(seed_radius: f64, prior: Option<OuterRadialPrior>) -> (f64, f64) {
    prior.map_or((0.72, 1.30), |prior| {
        (
            ((prior.radius - prior.half_width) / seed_radius).clamp(0.62, 1.38),
            ((prior.radius + prior.half_width) / seed_radius).clamp(0.62, 1.38),
        )
    })
}

fn best_outer_iris_candidate_scored_with_prior(
    luma: &BoxLuma5,
    width: usize,
    height: usize,
    seed: [f64; 3],
    angle: f64,
    prior: Option<OuterRadialPrior>,
) -> Option<(f64, OuterIrisPoint)> {
    let (direction_y, direction_x) = angle.sin_cos();
    let mut best: Option<(f64, OuterIrisPoint)> = None;
    let (first_scale, last_scale) = outer_radial_search_range(seed[2], prior);
    for step in 0..=100 {
        let scale = first_scale + step as f64 * (last_scale - first_scale) / 100.0;
        let x = seed[0] + scale * seed[2] * direction_x;
        let y = seed[1] + scale * seed[2] * direction_y;
        if x <= 5.0
            || y <= 5.0
            || x >= width.saturating_sub(6) as f64
            || y >= height.saturating_sub(6) as f64
        {
            continue;
        }
        let inside = [2.0, 4.0, 6.0]
            .iter()
            .map(|distance| luma.sample(x - direction_x * distance, y - direction_y * distance))
            .sum::<f64>()
            / 3.0;
        let outside = [2.0, 4.0, 6.0]
            .iter()
            .map(|distance| luma.sample(x + direction_x * distance, y + direction_y * distance))
            .sum::<f64>()
            / 3.0;
        let contrast = outside - inside;
        let prior_penalty = prior.map_or(0.0, |prior| {
            let normalized = (scale * seed[2] - prior.radius) / prior.half_width.max(1.0);
            prior.strength * normalized * normalized
        });
        let score = contrast - 0.06 * (scale - 1.0).abs() * seed[2] - prior_penalty;
        let point = OuterIrisPoint {
            x,
            y,
            contrast: contrast.max(1.0),
        };
        if best.as_ref().is_none_or(|candidate| score > candidate.0) {
            best = Some((score, point));
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn best_outer_iris_candidate_color_weighted(
    luma: &BoxLuma5,
    native: &NativeLogPlane,
    sclera: ScleraColorReference,
    width: usize,
    height: usize,
    seed: [f64; 3],
    angle: f64,
) -> Option<(f64, OuterIrisPoint)> {
    best_outer_iris_candidate_color_weighted_with_prior(
        luma, native, sclera, width, height, seed, angle, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn best_outer_iris_candidate_color_weighted_with_prior(
    luma: &BoxLuma5,
    native: &NativeLogPlane,
    sclera: ScleraColorReference,
    width: usize,
    height: usize,
    seed: [f64; 3],
    angle: f64,
    prior: Option<OuterRadialPrior>,
) -> Option<(f64, OuterIrisPoint)> {
    let shortlist_limit = if prior.is_some() { 28 } else { 12 };
    let (direction_y, direction_x) = angle.sin_cos();
    let mut shortlist = Vec::<(f64, OuterIrisPoint)>::with_capacity(shortlist_limit);
    let (first_scale, last_scale) = outer_radial_search_range(seed[2], prior);
    for step in 0..=100 {
        let scale = first_scale + step as f64 * (last_scale - first_scale) / 100.0;
        let x = seed[0] + scale * seed[2] * direction_x;
        let y = seed[1] + scale * seed[2] * direction_y;
        if x <= 5.0
            || y <= 5.0
            || x >= width.saturating_sub(6) as f64
            || y >= height.saturating_sub(6) as f64
        {
            continue;
        }
        let inside = [2.0, 4.0, 6.0]
            .iter()
            .map(|distance| luma.sample(x - direction_x * distance, y - direction_y * distance))
            .sum::<f64>()
            / 3.0;
        let outside = [2.0, 4.0, 6.0]
            .iter()
            .map(|distance| luma.sample(x + direction_x * distance, y + direction_y * distance))
            .sum::<f64>()
            / 3.0;
        let contrast = outside - inside;
        let prior_penalty = prior.map_or(0.0, |prior| {
            let normalized = (scale * seed[2] - prior.radius) / prior.half_width.max(1.0);
            prior.strength * normalized * normalized
        });
        let luma_score = contrast - 0.06 * (scale - 1.0).abs() * seed[2] - prior_penalty;
        let point = OuterIrisPoint {
            x,
            y,
            contrast: contrast.max(1.0),
        };
        if shortlist.len() < shortlist_limit {
            shortlist.push((luma_score, point));
        } else if let Some((weakest_index, weakest)) = shortlist
            .iter()
            .enumerate()
            .min_by(|left, right| left.1 .0.total_cmp(&right.1 .0))
        {
            if luma_score > weakest.0 {
                shortlist[weakest_index] = (luma_score, point);
            }
        }
    }

    let tolerance = sclera.tolerance.max(0.05);
    shortlist
        .into_iter()
        .filter_map(|(luma_score, point)| {
            let radial_luma = [6.0, 10.0, 14.0].map(|distance| {
                let inside = luma.sample(
                    point.x - direction_x * distance,
                    point.y - direction_y * distance,
                );
                let outside = luma.sample(
                    point.x + direction_x * distance,
                    point.y + direction_y * distance,
                );
                (inside, outside, outside - inside)
            });
            let inside_luma = radial_luma.iter().map(|sample| sample.0).sum::<f64>() / 3.0;
            let outside_luma = radial_luma.iter().map(|sample| sample.1).sum::<f64>() / 3.0;
            let wide_contrast = outside_luma - inside_luma;
            let supported_distances = radial_luma
                .iter()
                .filter(|sample| sample.2 >= OUTER_IRIS_MIN_LUMA_SUPPORT)
                .count();
            // Sclera must form a sustained brighter plateau outside the iris,
            // not merely a narrow pale-skin or eyelash edge.
            if wide_contrast < OUTER_IRIS_MIN_WIDE_LUMA_CONTRAST
                || supported_distances < 1
                || outside_luma < inside_luma * 1.06 + 8.0
            {
                return None;
            }
            let inside =
                native.sample_chroma(point.x - direction_x * 8.0, point.y - direction_y * 8.0);
            let outside =
                native.sample_chroma(point.x + direction_x * 8.0, point.y + direction_y * 8.0);
            let inside_distance = (inside.0 - sclera.log_rg).hypot(inside.1 - sclera.log_bg);
            let outside_distance = (outside.0 - sclera.log_rg).hypot(outside.1 - sclera.log_bg);
            let outside_normalized = outside_distance / tolerance;
            let improvement = (inside_distance - outside_distance) / tolerance;
            // A skin-colored outside is not a limbus hit even when its luma
            // edge is stronger. Retain a little tolerance for sensor noise.
            if outside_normalized > 1.45 || improvement < -0.20 {
                return None;
            }
            let chroma_jump = (inside.0 - outside.0).hypot(inside.1 - outside.1) / tolerance;
            let color_score = 300.0 * (1.0 - outside_normalized).clamp(-0.45, 1.0)
                + 220.0 * improvement.clamp(-0.20, 2.0)
                + 90.0 * chroma_jump.clamp(0.0, 2.0);
            Some((luma_score + color_score, point))
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

fn outer_curve_agreement_limit(radius: f64) -> f64 {
    (radius * 0.055).clamp(2.5, 6.0)
}

fn point_follows_outer_curve(point: OuterIrisPoint, circle: [f64; 3]) -> bool {
    let residual = ((point.x - circle[0]).hypot(point.y - circle[1]) - circle[2]).abs();
    residual <= outer_curve_agreement_limit(circle[2])
}

fn select_lateral_outer_curve_sweeps(
    paths: &[Vec<Option<OuterIrisPoint>>; 4],
    seed: [f64; 3],
) -> Vec<OuterIrisPoint> {
    let mut evidence = Vec::with_capacity(2 + 4 * (OUTER_IRIS_SWEEP_BRANCH_SAMPLES - 1));
    let mut active = [false; 4];

    for (anchor_branch, arms) in [(0usize, [0usize, 1usize]), (2, [2, 3])] {
        if let Some(anchor) = paths[anchor_branch]
            .first()
            .copied()
            .flatten()
            .filter(|point| point_follows_outer_curve(*point, seed))
        {
            evidence.push(anchor);
            for arm in arms {
                active[arm] = true;
            }
        }
    }

    let mut curve = seed;
    for step in 1..OUTER_IRIS_SWEEP_BRANCH_SAMPLES {
        let mut accepted_at_step = Vec::with_capacity(4);
        for branch in 0..paths.len() {
            if !active[branch] {
                continue;
            }
            match paths[branch]
                .get(step)
                .copied()
                .flatten()
                .filter(|point| point_follows_outer_curve(*point, curve))
            {
                Some(point) => accepted_at_step.push(point),
                None => active[branch] = false,
            }
        }
        evidence.extend(accepted_at_step);
        if evidence.len() >= 6 {
            curve = fit_outer_circle(&evidence, seed);
        }
    }
    evidence
}

fn select_lateral_outer_curve_sweeps_with_fallback(
    cue_paths: &[Vec<Option<OuterIrisPoint>>; 4],
    curve_paths: &[Vec<Option<OuterIrisPoint>>; 4],
    seed: [f64; 3],
) -> (Vec<OuterIrisPoint>, Vec<OuterIrisPoint>) {
    let cue_evidence = select_lateral_outer_curve_sweeps(cue_paths, seed);
    if cue_evidence.len() >= 8 {
        (cue_evidence.clone(), cue_evidence)
    } else {
        // Auxiliary appearance cues may shorten a hand, but they must never
        // erase a geometrically coherent ring by collectively starving it.
        (
            select_lateral_outer_curve_sweeps(curve_paths, seed),
            cue_evidence,
        )
    }
}

fn lateral_sweep_endpoints(evidence: &[OuterIrisPoint], seed: [f64; 3]) -> Vec<OuterIrisPoint> {
    let sweep_limit = outer_iris_lateral_sweep_angle();
    let anchor_contrast = |side_angle: f64| {
        evidence
            .iter()
            .filter_map(|point| {
                let angle = (point.y - seed[1]).atan2(point.x - seed[0]);
                let mut delta = (angle - side_angle).rem_euclid(2.0 * PI);
                if delta > PI {
                    delta -= 2.0 * PI;
                }
                (delta.abs() <= 0.04).then_some(point.contrast)
            })
            .max_by(f64::total_cmp)
            .unwrap_or(f64::NEG_INFINITY)
    };
    // One lateral anchor launches exactly two hands: clockwise and
    // counter-clockwise. Prefer the stronger 3/9-o'clock anchor each frame,
    // but keep 3 o'clock as the deterministic tie-breaker.
    let side_angle = if anchor_contrast(0.0) >= anchor_contrast(PI) {
        0.0
    } else {
        PI
    };
    let mut endpoints = Vec::with_capacity(2);
    for sweep_direction in [-1.0, 1.0] {
        let endpoint = evidence
            .iter()
            .copied()
            .filter_map(|point| {
                let angle = (point.y - seed[1]).atan2(point.x - seed[0]);
                let mut delta = (angle - side_angle).rem_euclid(2.0 * PI);
                if delta > PI {
                    delta -= 2.0 * PI;
                }
                let directed = delta * sweep_direction;
                (directed >= -0.02 && directed <= sweep_limit + 0.04)
                    .then_some((directed.max(0.0), point))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .map(|(_, point)| point);
        if let Some(endpoint) = endpoint {
            endpoints.push(endpoint);
        }
    }
    endpoints
}

fn eyelid_y_at_x(points: &[BorderPoint], x: f64) -> Option<f64> {
    let first = points.first()?;
    if x <= first.x as f64 {
        return Some(first.y as f64);
    }
    for pair in points.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if x <= right.x as f64 {
            let span = right.x.saturating_sub(left.x) as f64;
            if span <= f64::EPSILON {
                return Some((left.y as f64 + right.y as f64) * 0.5);
            }
            let phase = ((x - left.x as f64) / span).clamp(0.0, 1.0);
            return Some(left.y as f64 * (1.0 - phase) + right.y as f64 * phase);
        }
    }
    points.last().map(|point| point.y as f64)
}

/// Detects direct upper-eyelid edge samples in the anatomically plausible
/// arch above the iris. The constrained search keeps stronger eyebrow edges
/// outside the admission band and returns no inferred or completed points.
pub fn detect_upper_eyelid_points(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
) -> Vec<BorderPoint> {
    detect_eyelid_points(raw, width, height, sensor_x, sensor_y, coarse, true)
}

/// Detects direct lower-eyelid edge samples below the iris. Visible lower
/// sclera leaves this arch outside the limbus; an occluding lower lid can then
/// be distinguished from direct iris/sclera evidence by position.
pub fn detect_lower_eyelid_points(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
) -> Vec<BorderPoint> {
    detect_eyelid_points(raw, width, height, sensor_x, sensor_y, coarse, false)
}

fn detect_eyelid_points(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper: bool,
) -> Vec<BorderPoint> {
    if width < 16
        || height < 16
        || raw.len() < width * height
        || coarse.radius < 20.0
        || coarse.radius > width.min(height) as f64 * 0.45
        || !coarse.center.0.is_finite()
        || !coarse.center.1.is_finite()
    {
        return Vec::new();
    }
    const SAMPLE_COUNT: usize = 25;
    const MINIMUM_CONTIGUOUS_POINTS: usize = 11;
    #[derive(Clone)]
    struct LidPath {
        points: Vec<BorderPoint>,
        score: f64,
    }

    let luma = BoxLuma5::new(raw, width, height);
    let Some(native) = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse) else {
        return Vec::new();
    };
    let radius = coarse.radius;
    let maximum_step = (radius * 0.09).clamp(4.0, 9.0);
    let mut active_paths: Vec<LidPath> = Vec::new();
    let mut best_path: Option<LidPath> = None;
    for index in 0..SAMPLE_COUNT {
        let normalized_x = index as f64 * 2.0 / (SAMPLE_COUNT - 1) as f64 - 1.0;
        let x = coarse.center.0 + normalized_x * radius;
        if x <= 6.0 || x >= width.saturating_sub(7) as f64 {
            active_paths.clear();
            continue;
        }
        let expected_offset = if upper {
            -(0.62 - 0.16 * normalized_x.powi(2))
        } else {
            0.62 - 0.11 * normalized_x.powi(2)
        };
        let expected_y = coarse.center.1 + radius * expected_offset;
        let search_half_height = (radius * 0.48).clamp(12.0, 48.0);
        let first_y = (expected_y - search_half_height).ceil().max(6.0) as usize;
        let last_y = (expected_y + search_half_height)
            .floor()
            .min(height.saturating_sub(7) as f64) as usize;
        let mut candidates = Vec::new();
        for sample_y in first_y..=last_y {
            let y = sample_y as f64;
            let above = (luma.sample(x, y - 4.0) + luma.sample(x, y - 7.0)) * 0.5;
            let on_edge = luma.sample(x, y);
            let below = (luma.sample(x, y + 4.0) + luma.sample(x, y + 7.0)) * 0.5;
            let edge_strength = (below - above).abs();
            if edge_strength <= 12.0 {
                continue;
            }
            let mut thickness_support = 0usize;
            let mut void_separation = 0.0;
            let mut chroma_separation = 0.0;
            for distance in [6.0, 10.0, 14.0] {
                let (inside_y, outside_y) = if upper {
                    (y + distance, y - distance)
                } else {
                    (y - distance, y + distance)
                };
                let inside_void = native.sample_void(x, inside_y);
                let outside_void = native.sample_void(x, outside_y);
                let inside_chroma = native.sample_chroma(x, inside_y);
                let outside_chroma = native.sample_chroma(x, outside_y);
                let chroma =
                    (inside_chroma.0 - outside_chroma.0).hypot(inside_chroma.1 - outside_chroma.1);
                let void = outside_void - inside_void;
                let luma_separation = (luma.sample(x, inside_y) - luma.sample(x, outside_y)).abs();
                if inside_void <= 0.62
                    && luma_separation >= 10.0
                    && (void >= 0.04 || chroma >= 0.06)
                {
                    thickness_support += 1;
                }
                void_separation += void;
                chroma_separation += chroma;
            }
            if thickness_support < 3 {
                continue;
            }
            void_separation /= 3.0;
            chroma_separation /= 3.0;
            let dark_line = ((above + below) * 0.5 - on_edge).max(0.0);
            let proximity_penalty = (y - expected_y).abs() * 0.035;
            let quality = edge_strength
                + 0.55 * dark_line
                + 42.0 * void_separation.max(0.0)
                + 30.0 * chroma_separation.min(1.0)
                - proximity_penalty;
            let edge_at = |sample_y: f64| {
                let above = (luma.sample(x, sample_y - 4.0) + luma.sample(x, sample_y - 7.0)) * 0.5;
                let below = (luma.sample(x, sample_y + 4.0) + luma.sample(x, sample_y + 7.0)) * 0.5;
                (below - above).abs()
            };
            if edge_strength + 1.0e-6 < edge_at(y - 1.0).max(edge_at(y + 1.0)) {
                continue;
            }
            candidates.push(BorderPoint {
                x: x.round() as usize,
                y: sample_y,
                quality,
            });
        }
        candidates.sort_by(|left, right| right.quality.total_cmp(&left.quality));
        candidates.truncate(6);
        let mut next_paths = Vec::with_capacity(candidates.len());
        for point in candidates {
            let predecessor = active_paths
                .iter()
                .filter(|path| {
                    path.points
                        .last()
                        .is_some_and(|last| last.y.abs_diff(point.y) as f64 <= maximum_step)
                })
                .max_by(|left, right| {
                    let left_last = left.points.last().unwrap();
                    let right_last = right.points.last().unwrap();
                    let left_value = left.score - left_last.y.abs_diff(point.y) as f64 * 0.35;
                    let right_value = right.score - right_last.y.abs_diff(point.y) as f64 * 0.35;
                    left_value.total_cmp(&right_value)
                });
            let mut path = predecessor.cloned().unwrap_or(LidPath {
                points: Vec::new(),
                score: 0.0,
            });
            if let Some(last) = path.points.last() {
                path.score -= last.y.abs_diff(point.y) as f64 * 0.35;
            }
            path.score += point.quality;
            path.points.push(point);
            if best_path.as_ref().is_none_or(|best| {
                path.points.len() > best.points.len()
                    || (path.points.len() == best.points.len() && path.score > best.score)
            }) {
                best_path = Some(path.clone());
            }
            next_paths.push(path);
        }
        active_paths = next_paths;
    }
    best_path
        .filter(|path| {
            path.points.len() >= MINIMUM_CONTIGUOUS_POINTS
                && path.score / path.points.len() as f64 >= 10.0
        })
        .map(|path| path.points)
        .unwrap_or_default()
}

#[derive(Clone)]
struct NautilusLidColumn {
    x: usize,
    expected_y: f64,
    candidates: Vec<BorderPoint>,
}

/// A zero-copy local-luma view over the native RAW buffer.  The nautilus pass
/// touches only the small neighborhoods it evaluates; it neither clones the
/// ROI nor constructs a downsampled or full-frame integral image.
#[derive(Clone, Copy)]
struct NautilusRawLuma<'a> {
    raw: &'a [u16],
    width: usize,
    height: usize,
}

impl<'a> NautilusRawLuma<'a> {
    fn new(raw: &'a [u16], width: usize, height: usize) -> Self {
        Self { raw, width, height }
    }

    #[inline]
    fn integer_sample3(self, x: usize, y: usize) -> f64 {
        let x = x.clamp(1, self.width.saturating_sub(2));
        let y = y.clamp(1, self.height.saturating_sub(2));
        let mut sum = 0u32;
        for sample_y in y - 1..=y + 1 {
            let row = sample_y * self.width;
            sum += self.raw[row + x - 1] as u32;
            sum += self.raw[row + x] as u32;
            sum += self.raw[row + x + 1] as u32;
        }
        sum as f64 / 9.0
    }

    #[inline]
    fn sample3(self, x: f64, y: f64) -> f64 {
        let x = x.clamp(1.0, self.width.saturating_sub(2) as f64);
        let y = y.clamp(1.0, self.height.saturating_sub(2) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let top = self.integer_sample3(x0, y0) * (1.0 - fx) + self.integer_sample3(x0 + 1, y0) * fx;
        let bottom = self.integer_sample3(x0, y0 + 1) * (1.0 - fx)
            + self.integer_sample3(x0 + 1, y0 + 1) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// Optical edge concentration measured directly on the selected limbus in
/// the untouched native RAW ROI. `sharpness` is contrast-normalized into
/// 0..=1; `contrast_raw10` remains separate so illumination cannot masquerade
/// as focus. An empty/default result means that too few lateral iris/sclera
/// transitions were directly observable.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LimbusOpticalFocus {
    pub sharpness: f64,
    pub contrast_raw10: f64,
    pub support: usize,
}

/// Measure how concentrated the iris-to-sclera rise is along lateral normals
/// of a directly selected ellipse. This is a bounded zero-copy pass: it reads
/// 26 small neighborhoods from `raw`, constructs no image pyramid/integral
/// plane, and never resamples or downsizes the ROI.
pub fn measure_limbus_optical_focus(
    raw: &[u16],
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
) -> LimbusOpticalFocus {
    if raw.len() < width.saturating_mul(height)
        || width < 32
        || height < 24
        || outer.points.len() < 8
        || !outer.center.0.is_finite()
        || !outer.center.1.is_finite()
        || !outer.major_radius.is_finite()
        || !outer.minor_radius.is_finite()
        || outer.major_radius < 12.0
        || outer.minor_radius < 8.0
    {
        return LimbusOpticalFocus::default();
    }

    let luma = NautilusRawLuma::new(raw, width, height);
    let (ellipse_sin, ellipse_cos) = outer.angle.sin_cos();
    let offsets = [-8.0f64, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0];
    let mut sharpnesses = [0.0f64; 26];
    let mut contrasts = [0.0f64; 26];
    let mut support = 0usize;

    for sector_center in [0.0, PI] {
        for step in 0..13 {
            let phase = sector_center - PI / 3.0 + step as f64 * PI / 18.0;
            let (phase_sin, phase_cos) = phase.sin_cos();
            let local_x = outer.major_radius * phase_cos;
            let local_y = outer.minor_radius * phase_sin;
            let point = (
                outer.center.0 + ellipse_cos * local_x - ellipse_sin * local_y,
                outer.center.1 + ellipse_sin * local_x + ellipse_cos * local_y,
            );
            // The gradient of x²/a² + y²/b² supplies the actual projected
            // conic normal; a radial ray is wrong on an oblique ellipse.
            let local_normal_x = phase_cos / outer.major_radius;
            let local_normal_y = phase_sin / outer.minor_radius;
            let normal_length = local_normal_x.hypot(local_normal_y);
            if !normal_length.is_finite() || normal_length <= f64::EPSILON {
                continue;
            }
            let normal = (
                (ellipse_cos * local_normal_x - ellipse_sin * local_normal_y) / normal_length,
                (ellipse_sin * local_normal_x + ellipse_cos * local_normal_y) / normal_length,
            );
            let first = (
                point.0 + normal.0 * offsets[0],
                point.1 + normal.1 * offsets[0],
            );
            let last = (
                point.0 + normal.0 * offsets[offsets.len() - 1],
                point.1 + normal.1 * offsets[offsets.len() - 1],
            );
            if first.0 < 2.0
                || first.1 < 2.0
                || last.0 < 2.0
                || last.1 < 2.0
                || first.0 >= width.saturating_sub(3) as f64
                || first.1 >= height.saturating_sub(3) as f64
                || last.0 >= width.saturating_sub(3) as f64
                || last.1 >= height.saturating_sub(3) as f64
            {
                continue;
            }

            let mut profile = [0.0f64; 9];
            for (sample, offset) in profile.iter_mut().zip(offsets) {
                *sample = luma.sample3(point.0 + normal.0 * offset, point.1 + normal.1 * offset);
            }
            let net_rise = profile[8] - profile[0];
            if net_rise < 4.0 {
                continue;
            }
            let mut positive_rise = [0.0f64; 8];
            let mut positive_total = 0.0;
            let mut total_variation = 0.0;
            for index in 0..8 {
                let delta = profile[index + 1] - profile[index];
                positive_rise[index] = delta.max(0.0);
                positive_total += positive_rise[index];
                total_variation += delta.abs();
            }
            if positive_total < 4.0 || total_variation <= f64::EPSILON {
                continue;
            }
            let concentrated_rise = positive_rise
                .windows(2)
                .map(|window| window[0] + window[1])
                .fold(0.0, f64::max);
            let concentration = (concentrated_rise / positive_total).clamp(0.0, 1.0);
            let monotonicity = (net_rise / total_variation).clamp(0.0, 1.0);
            sharpnesses[support] = concentration * monotonicity;
            contrasts[support] = net_rise;
            support += 1;
        }
    }

    if support < 6 {
        return LimbusOpticalFocus::default();
    }
    LimbusOpticalFocus {
        sharpness: median(&mut sharpnesses[..support]).clamp(0.0, 1.0),
        contrast_raw10: median(&mut contrasts[..support]).max(0.0),
        support,
    }
}

#[derive(Clone, Default)]
struct NautilusLidPath {
    points: Vec<(usize, BorderPoint)>,
    score: f64,
    gaps: usize,
}

fn nautilus_vertical_edge(luma: NautilusRawLuma<'_>, x: f64, y: f64) -> (f64, f64, f64) {
    let above = 0.5 * (luma.sample3(x, y - 3.0) + luma.sample3(x, y - 7.0));
    let below = 0.5 * (luma.sample3(x, y + 3.0) + luma.sample3(x, y + 7.0));
    let center = luma.sample3(x, y);
    let wide_above = luma.sample3(x, y - 12.0);
    let wide_below = luma.sample3(x, y + 12.0);
    (
        (below - above).abs(),
        (0.5 * (above + below) - center).max(0.0),
        (wide_below - wide_above).abs(),
    )
}

/// Follow one bank of candidate edge samples from a canthus toward the other
/// side.  The predicted heading comes from the last two accepted samples.  A
/// missed column increases the next fan radius, producing the slowly opening
/// shell which gives the walk its nautilus name; the shell is capped after two
/// misses, so it cannot jump from an eyelid to an eyebrow or cheek crease.
fn nautilus_lid_walk(
    columns: &[NautilusLidColumn],
    reverse: bool,
    base_fan_px: f64,
) -> Vec<(usize, BorderPoint)> {
    if columns.is_empty() {
        return Vec::new();
    }
    let order = if reverse {
        (0..columns.len()).rev().collect::<Vec<_>>()
    } else {
        (0..columns.len()).collect::<Vec<_>>()
    };
    let mut active = Vec::<NautilusLidPath>::new();
    let mut best = NautilusLidPath::default();
    for (order_index, &column_index) in order.iter().enumerate() {
        let column = &columns[column_index];
        let mut next = Vec::<NautilusLidPath>::new();
        for path in &active {
            for &candidate in &column.candidates {
                let Some((_, last)) = path.points.last() else {
                    continue;
                };
                let dx = candidate.x.abs_diff(last.x).max(1) as f64;
                let previous_slope = path.points.iter().rev().nth(1).map_or(0.0, |(_, before)| {
                    (last.y as f64 - before.y as f64) / last.x.abs_diff(before.x).max(1) as f64
                });
                let direction = if candidate.x >= last.x { 1.0 } else { -1.0 };
                let predicted_y = last.y as f64 + previous_slope * dx * direction;
                let prediction_error = (candidate.y as f64 - predicted_y).abs();
                let shell = base_fan_px * (1.0 + 0.55 * path.gaps.min(2) as f64)
                    + 0.10 * previous_slope.abs() * dx;
                if prediction_error > shell {
                    continue;
                }
                let current_slope = (candidate.y as f64 - last.y as f64) / (dx * direction);
                let curvature = (current_slope - previous_slope).abs();
                let mut extended = path.clone();
                extended.points.push((column_index, candidate));
                extended.gaps = 0;
                extended.score += candidate.quality.min(220.0)
                    - 1.80 * prediction_error
                    - 4.0 * curvature.min(4.0);
                next.push(extended);
            }
            if path.gaps < 2 {
                let mut carried = path.clone();
                carried.gaps += 1;
                carried.score -= 12.0 * carried.gaps as f64;
                next.push(carried);
            }
        }
        // Only the first few lateral columns may freely launch a hand. Later
        // starts are admitted solely when all earlier hands have died.
        if order_index < 4 || next.is_empty() {
            for &candidate in &column.candidates {
                next.push(NautilusLidPath {
                    points: vec![(column_index, candidate)],
                    score: candidate.quality.min(220.0),
                    gaps: 0,
                });
            }
        }
        next.sort_by(|left, right| {
            let rank = |path: &NautilusLidPath| {
                path.score + 28.0 * path.points.len() as f64 - 8.0 * path.gaps as f64
            };
            rank(right).total_cmp(&rank(left))
        });
        next.truncate(10);
        for path in &next {
            let path_rank =
                path.points.len() as f64 * 1000.0 + path.score / path.points.len().max(1) as f64;
            let best_rank =
                best.points.len() as f64 * 1000.0 + best.score / best.points.len().max(1) as f64;
            if path_rank > best_rank {
                best = path.clone();
            }
        }
        active = next;
    }
    best.points.sort_by_key(|(column, _)| *column);
    best.points
}

fn fuse_nautilus_lid_walks(
    columns: &[NautilusLidColumn],
    from_left: &[(usize, BorderPoint)],
    from_right: &[(usize, BorderPoint)],
    agreement_px: f64,
    minimum_fraction: f64,
) -> Vec<BorderPoint> {
    let mut left = vec![None; columns.len()];
    let mut right = vec![None; columns.len()];
    for &(column, point) in from_left {
        if column < left.len() {
            left[column] = Some(point);
        }
    }
    for &(column, point) in from_right {
        if column < right.len() {
            right[column] = Some(point);
        }
    }
    let mut fused = Vec::new();
    for (column_index, column) in columns.iter().enumerate() {
        let point = match (left[column_index], right[column_index]) {
            (Some(first), Some(second)) if first.y.abs_diff(second.y) as f64 <= agreement_px => {
                let first_weight = first.quality.max(1.0);
                let second_weight = second.quality.max(1.0);
                Some(BorderPoint {
                    x: column.x,
                    y: ((first.y as f64 * first_weight + second.y as f64 * second_weight)
                        / (first_weight + second_weight))
                        .round() as usize,
                    quality: 0.5 * (first.quality + second.quality),
                })
            }
            (Some(first), Some(second)) => {
                let expected = column.expected_y;
                let first_error = (first.y as f64 - expected).abs();
                let second_error = (second.y as f64 - expected).abs();
                let preferred = if first_error + 1.0 < second_error
                    && first.quality >= second.quality * 0.80
                {
                    Some(first)
                } else if second_error + 1.0 < first_error && second.quality >= first.quality * 0.80
                {
                    Some(second)
                } else {
                    None
                };
                preferred
            }
            (Some(point), None) | (None, Some(point)) => Some(point),
            (None, None) => None,
        };
        if let Some(point) = point {
            fused.push(point);
        }
    }
    let minimum_points = ((columns.len() as f64 * minimum_fraction).ceil() as usize).max(7);
    if fused.len() < minimum_points {
        let fallback = if from_left.len() >= from_right.len() {
            from_left
        } else {
            from_right
        };
        if fallback.len() < minimum_points {
            return Vec::new();
        }
        fused = fallback.iter().map(|(_, point)| *point).collect();
        fused.sort_by_key(|point| point.x);
    }

    // A hand may die on the true margin and relaunch later on a lid fold or
    // skin edge. Never draw a straight bridge across that discontinuity. Keep
    // the longest coherent run that reaches the limbus center and reject the
    // whole margin if that run no longer supplies the required support.
    let center_x = columns[columns.len() / 2].x;
    let mut column_steps = columns
        .windows(2)
        .map(|pair| pair[0].x.abs_diff(pair[1].x))
        .collect::<Vec<_>>();
    column_steps.sort_unstable();
    let nominal_column_step = column_steps
        .get(column_steps.len() / 2)
        .copied()
        .unwrap_or(1)
        .max(1);
    let center_reach_px = nominal_column_step * 2;
    let mut runs = Vec::new();
    let mut run_start = 0usize;
    for index in 1..fused.len() {
        let previous = fused[index - 1];
        let current = fused[index];
        let dx = previous.x.abs_diff(current.x).max(1) as f64;
        let dy = previous.y.abs_diff(current.y) as f64;
        let maximum_continuous_dy = agreement_px * 1.5 + 0.65 * dx;
        if dy > maximum_continuous_dy {
            runs.push((run_start, index));
            run_start = index;
        }
    }
    runs.push((run_start, fused.len()));
    let &(best_start, best_end) = runs
        .iter()
        .max_by_key(|&&(start, end)| {
            let run = &fused[start..end];
            let reaches_center = run
                .iter()
                .any(|point| point.x.abs_diff(center_x) <= center_reach_px);
            let span = run
                .first()
                .zip(run.last())
                .map_or(0, |(first, last)| last.x.saturating_sub(first.x));
            (reaches_center, run.len(), span)
        })
        .expect("at least one fused lid run");
    fused = fused[best_start..best_end].to_vec();
    if fused.len() < minimum_points {
        return Vec::new();
    }
    let reaches_center = fused
        .iter()
        .any(|point| point.x.abs_diff(center_x) <= center_reach_px);
    let span = fused
        .first()
        .zip(fused.last())
        .map_or(0, |(first, last)| last.x.saturating_sub(first.x));
    let required_span = columns
        .first()
        .zip(columns.last())
        .map_or(0, |(first, last)| last.x.saturating_sub(first.x) / 2);
    if !reaches_center || span < required_span {
        return Vec::new();
    }
    // A short median polish removes single-CFA stair steps without inventing
    // samples in columns where neither hand saw an edge.
    let original = fused.clone();
    for index in 1..fused.len().saturating_sub(1) {
        let mut ys = [
            original[index - 1].y,
            original[index].y,
            original[index + 1].y,
        ];
        ys.sort_unstable();
        fused[index].y = ys[1];
    }
    fused
}

fn nautilus_margin_columns(
    luma: NautilusRawLuma<'_>,
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
    pupil: (f64, f64),
    upper: bool,
    deadline: Instant,
) -> Vec<NautilusLidColumn> {
    const COLUMN_COUNT: usize = 41;
    let horizontal_radius =
        (outer.major_radius * 1.22).clamp(28.0, width.saturating_sub(14) as f64 * 0.5);
    let first_x = (outer.center.0 - horizontal_radius).max(6.0);
    let last_x = (outer.center.0 + horizontal_radius).min(width.saturating_sub(7) as f64);
    if last_x <= first_x + 12.0 {
        return Vec::new();
    }
    let anchor_y = 0.82 * outer.center.1 + 0.18 * pupil.1;
    let mut columns = Vec::with_capacity(COLUMN_COUNT);
    let mut previous_x = usize::MAX;
    for index in 0..COLUMN_COUNT {
        if Instant::now() >= deadline {
            break;
        }
        let phase = index as f64 / (COLUMN_COUNT - 1) as f64;
        let x = (first_x * (1.0 - phase) + last_x * phase).round() as usize;
        if x == previous_x {
            continue;
        }
        previous_x = x;
        let normalized_x =
            ((x as f64 - outer.center.0) / (outer.major_radius * 1.14).max(1.0)).clamp(-1.0, 1.0);
        let opening = (1.0 - normalized_x * normalized_x).max(0.0).sqrt();
        let expected_y = if upper {
            anchor_y - outer.minor_radius * 0.72 * opening
        } else {
            anchor_y + outer.minor_radius * 0.66 * opening
        };
        let half_search = (outer.minor_radius * 0.40).clamp(11.0, 42.0);
        let anatomical_limit = if upper {
            pupil.1 + outer.minor_radius * 0.18
        } else {
            pupil.1 - outer.minor_radius * 0.18
        };
        let (first_y, last_y) = if upper {
            (
                (expected_y - half_search).ceil().max(13.0) as usize,
                (expected_y + half_search)
                    .floor()
                    .min(anatomical_limit)
                    .min(height.saturating_sub(14) as f64) as usize,
            )
        } else {
            (
                (expected_y - half_search)
                    .ceil()
                    .max(anatomical_limit)
                    .max(13.0) as usize,
                (expected_y + half_search)
                    .floor()
                    .min(height.saturating_sub(14) as f64) as usize,
            )
        };
        let mut candidates = Vec::new();
        if first_y <= last_y {
            for y in first_y..=last_y {
                let (edge, dark_line, wide_edge) = nautilus_vertical_edge(luma, x as f64, y as f64);
                if edge < 7.0 || edge + wide_edge < 16.0 {
                    continue;
                }
                let neighbor_edge = nautilus_vertical_edge(luma, x as f64, y as f64 - 1.0)
                    .0
                    .max(nautilus_vertical_edge(luma, x as f64, y as f64 + 1.0).0);
                if edge + 1.0e-6 < neighbor_edge {
                    continue;
                }
                let inward_sign = if upper { 1.0 } else { -1.0 };
                let inside = 0.5
                    * (luma.sample3(x as f64, y as f64 + inward_sign * 5.0)
                        + luma.sample3(x as f64, y as f64 + inward_sign * 10.0));
                let outside = 0.5
                    * (luma.sample3(x as f64, y as f64 - inward_sign * 5.0)
                        + luma.sample3(x as f64, y as f64 - inward_sign * 10.0));
                let pupil_dx = pupil.0 - x as f64;
                let pupil_dy = pupil.1 - y as f64;
                let pupil_distance = pupil_dx.hypot(pupil_dy).max(1.0);
                let pupilward = luma.sample3(
                    x as f64 + pupil_dx / pupil_distance * 10.0,
                    y as f64 + pupil_dy / pupil_distance * 10.0,
                );
                let geometric_error = (y as f64 - expected_y).abs() / half_search.max(1.0);
                let quality = 0.78 * edge
                    + 0.28 * wide_edge
                    + 0.52 * dark_line
                    + 0.20 * (inside - outside).abs()
                    + 0.12 * (pupilward - outside).abs()
                    - 24.0 * geometric_error.powi(2);
                if quality >= 8.0 {
                    candidates.push(BorderPoint { x, y, quality });
                }
            }
        }
        candidates.sort_by(|left, right| right.quality.total_cmp(&left.quality));
        candidates.truncate(8);
        columns.push(NautilusLidColumn {
            x,
            expected_y,
            candidates,
        });
    }
    columns
}

fn nautilus_margin(
    luma: NautilusRawLuma<'_>,
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
    pupil: (f64, f64),
    upper: bool,
    deadline: Instant,
) -> Vec<BorderPoint> {
    let columns = nautilus_margin_columns(luma, width, height, outer, pupil, upper, deadline);
    if columns.len() < 12 {
        return Vec::new();
    }
    let base_fan = (outer.minor_radius * 0.10).clamp(4.0, 10.0);
    let from_left = nautilus_lid_walk(&columns, false, base_fan);
    let from_right = nautilus_lid_walk(&columns, true, base_fan);
    fuse_nautilus_lid_walks(
        &columns,
        &from_left,
        &from_right,
        (outer.minor_radius * 0.11).clamp(4.0, 9.0),
        0.46,
    )
}

fn nautilus_parallel_fold(
    luma: NautilusRawLuma<'_>,
    margin: &[BorderPoint],
    outer: &OuterIrisBoundary,
    upper: bool,
    deadline: Instant,
) -> Vec<BorderPoint> {
    if margin.len() < 9 || Instant::now() >= deadline {
        return Vec::new();
    }
    let outward = if upper { -1.0 } else { 1.0 };
    let (minimum_separation, expected_separation, maximum_separation) = if upper {
        (
            (outer.minor_radius * 0.10).clamp(5.0, 11.0),
            (outer.minor_radius * 0.23).clamp(9.0, 24.0),
            (outer.minor_radius * 0.39).clamp(15.0, 38.0),
        )
    } else {
        (
            (outer.minor_radius * 0.08).clamp(4.0, 9.0),
            (outer.minor_radius * 0.16).clamp(7.0, 18.0),
            (outer.minor_radius * 0.29).clamp(12.0, 28.0),
        )
    };
    let mut columns = Vec::with_capacity(margin.len());
    for &margin_point in margin {
        if Instant::now() >= deadline {
            break;
        }
        let expected_y = margin_point.y as f64 + outward * expected_separation;
        let first = margin_point.y as f64 + outward * minimum_separation;
        let last = margin_point.y as f64 + outward * maximum_separation;
        let minimum_y = first.min(last).ceil().max(13.0) as usize;
        let maximum_y = first
            .max(last)
            .floor()
            .min(luma.height.saturating_sub(14) as f64) as usize;
        let mut candidates = Vec::new();
        if minimum_y <= maximum_y {
            for y in minimum_y..=maximum_y {
                let (edge, dark_line, wide_edge) =
                    nautilus_vertical_edge(luma, margin_point.x as f64, y as f64);
                if edge < 5.0 || edge + wide_edge < 12.0 {
                    continue;
                }
                let separation = (y as f64 - margin_point.y as f64).abs();
                let separation_error =
                    (separation - expected_separation).abs() / maximum_separation.max(1.0);
                let quality = 0.72 * edge + 0.34 * wide_edge + 0.34 * dark_line
                    - 22.0 * separation_error.powi(2);
                if quality >= 6.0 {
                    candidates.push(BorderPoint {
                        x: margin_point.x,
                        y,
                        quality,
                    });
                }
            }
        }
        candidates.sort_by(|left, right| right.quality.total_cmp(&left.quality));
        candidates.truncate(5);
        columns.push(NautilusLidColumn {
            x: margin_point.x,
            expected_y,
            candidates,
        });
    }
    if columns.len() < 8 {
        return Vec::new();
    }
    let base_fan = (outer.minor_radius * 0.09).clamp(4.0, 9.0);
    let from_left = nautilus_lid_walk(&columns, false, base_fan);
    let from_right = nautilus_lid_walk(&columns, true, base_fan);
    let fold = fuse_nautilus_lid_walks(
        &columns,
        &from_left,
        &from_right,
        (outer.minor_radius * 0.09).clamp(4.0, 8.0),
        0.42,
    );
    (fold.len() * 5 >= margin.len() * 2)
        .then_some(fold)
        .unwrap_or_default()
}

fn nautilus_lashes(
    raw: &[u16],
    width: usize,
    height: usize,
    margin: &[BorderPoint],
    outer: &OuterIrisBoundary,
    upper: bool,
    deadline: Instant,
) -> Vec<BorderPoint> {
    if margin.len() < 7 || Instant::now() >= deadline {
        return Vec::new();
    }
    let outward = if upper { -1.0 } else { 1.0 };
    let maximum_length = (outer.minor_radius * 0.18).clamp(5.0, 15.0) as usize;
    let mut candidates = Vec::<Option<BorderPoint>>::with_capacity(margin.len());
    for &root in margin {
        if Instant::now() >= deadline {
            break;
        }
        let mut best: Option<BorderPoint> = None;
        for distance in 2..=maximum_length {
            let y = root.y as f64 + outward * distance as f64;
            if y < 3.0 || y >= height.saturating_sub(3) as f64 || root.x < 4 || root.x + 4 >= width
            {
                continue;
            }
            let center = cfa_luma(raw, width, height, root.x as f64, y);
            let lateral = 0.5
                * (cfa_luma(raw, width, height, root.x as f64 - 4.0, y)
                    + cfa_luma(raw, width, height, root.x as f64 + 4.0, y));
            let farther = cfa_luma(raw, width, height, root.x as f64, y + outward * 3.0);
            let needle_score = (lateral - center).max(0.0) + 0.35 * (farther - center).max(0.0)
                - 0.35 * distance as f64;
            if needle_score >= 13.0
                && best
                    .as_ref()
                    .is_none_or(|point| needle_score > point.quality)
            {
                best = Some(BorderPoint {
                    x: root.x,
                    y: y.round() as usize,
                    quality: needle_score,
                });
            }
        }
        candidates.push(best);
    }
    let mut lashes = Vec::new();
    for index in 0..candidates.len() {
        let Some(point) = candidates[index] else {
            continue;
        };
        let previous = index
            .checked_sub(1)
            .and_then(|neighbor| candidates[neighbor])
            .map_or(0.0, |neighbor| neighbor.quality);
        let next = candidates
            .get(index + 1)
            .copied()
            .flatten()
            .map_or(0.0, |neighbor| neighbor.quality);
        if point.quality + 2.0 >= previous.max(next) {
            lashes.push(point);
        }
    }
    lashes
}

fn eyelid_roi_clearance(
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
) -> Option<(f64, f64, f64)> {
    if width == 0
        || height == 0
        || !outer.center.0.is_finite()
        || !outer.center.1.is_finite()
        || !outer.major_radius.is_finite()
        || !outer.minor_radius.is_finite()
        || !outer.angle.is_finite()
    {
        return None;
    }
    let (sin_angle, cos_angle) = outer.angle.sin_cos();
    let vertical_half_extent = ((outer.major_radius * sin_angle).powi(2)
        + (outer.minor_radius * cos_angle).powi(2))
    .sqrt();
    let upper_clearance = outer.center.1 - vertical_half_extent;
    let lower_clearance = height.saturating_sub(1) as f64 - (outer.center.1 + vertical_half_extent);
    let edge_guard = (outer.minor_radius * 0.08).clamp(6.0, 12.0);
    Some((upper_clearance, lower_clearance, edge_guard))
}

/// Build optional eyelid scene geometry around a completed Driving limbus.
///
/// Two bounded hands launch from opposite canthi for each lid. Their search
/// fan opens only after a missed sample and closes as soon as the other hand
/// confirms the same road. Folds are separately walked parallel candidates;
/// lashes are sparse native-CFA needle responses rooted on a confirmed margin.
/// Every output is presentation/model context only and has zero weight in the
/// physical limbus or pupil solve.
pub fn discover_eyelid_scene_nautilus(
    raw: &[u16],
    width: usize,
    height: usize,
    outer: &OuterIrisBoundary,
    pupil: Option<(f64, f64)>,
) -> EyelidNautilusScene {
    let started = Instant::now();
    let mut scene = EyelidNautilusScene::default();
    if raw.len() < width.saturating_mul(height)
        || width < 48
        || height < 40
        || !outer.center.0.is_finite()
        || !outer.center.1.is_finite()
        || outer.major_radius < 18.0
        || outer.minor_radius < 12.0
        || outer.major_radius > width as f64 * 0.62
        || outer.minor_radius > height as f64 * 0.62
    {
        return scene;
    }
    let pupil = pupil
        .filter(|point| {
            point.0.is_finite()
                && point.1.is_finite()
                && (0.0..width as f64).contains(&point.0)
                && (0.0..height as f64).contains(&point.1)
        })
        .unwrap_or(outer.center);
    let Some((upper_clearance, lower_clearance, edge_guard)) =
        eyelid_roi_clearance(width, height, outer)
    else {
        return scene;
    };
    scene.upper_limbus_clearance_px = Some(upper_clearance);
    scene.lower_limbus_clearance_px = Some(lower_clearance);
    scene.upper_status = if upper_clearance < edge_guard {
        EyelidObservationStatus::RoiClipped
    } else {
        EyelidObservationStatus::NotObserved
    };
    scene.lower_status = if lower_clearance < edge_guard {
        EyelidObservationStatus::RoiClipped
    } else {
        EyelidObservationStatus::NotObserved
    };
    // Debug builds need a relaxed ceiling so fixture assertions do not depend
    // on host load. The optimized live path remains confined to eight ms.
    let budget = if cfg!(test) {
        Duration::from_millis(60)
    } else {
        Duration::from_millis(8)
    };
    let deadline = started + budget;
    let luma = NautilusRawLuma::new(raw, width, height);
    // A projected limbus crossing the ROI edge censors the *unoccluded*
    // continuation of that conic; it does not prove that an occluding edge is
    // absent. In a tight crop the lid can cross inside the projected limbus,
    // but the same walk can also land on the remaining limbus tail. Preserve
    // that distinction: clipped-side roads are censor-only and never become
    // anatomical lid margins, folds, or lashes.
    let upper_walk = nautilus_margin(luma, width, height, outer, pupil, true, deadline);
    if scene.upper_status == EyelidObservationStatus::RoiClipped {
        scene.upper_clipped_occluder = upper_walk;
    } else if !upper_walk.is_empty() {
        scene.upper_margin = upper_walk;
        scene.upper_status = EyelidObservationStatus::Observed;
    }
    if Instant::now() < deadline {
        let lower_walk = nautilus_margin(luma, width, height, outer, pupil, false, deadline);
        if scene.lower_status == EyelidObservationStatus::RoiClipped {
            scene.lower_clipped_occluder = lower_walk;
        } else if !lower_walk.is_empty() {
            scene.lower_margin = lower_walk;
            scene.lower_status = EyelidObservationStatus::Observed;
        }
    }
    if scene.upper_status == EyelidObservationStatus::Observed && Instant::now() < deadline {
        scene.upper_fold = nautilus_parallel_fold(luma, &scene.upper_margin, outer, true, deadline);
    }
    if scene.lower_status == EyelidObservationStatus::Observed && Instant::now() < deadline {
        scene.lower_fold =
            nautilus_parallel_fold(luma, &scene.lower_margin, outer, false, deadline);
    }
    if scene.upper_status == EyelidObservationStatus::Observed && Instant::now() < deadline {
        scene.upper_lashes = nautilus_lashes(
            raw,
            width,
            height,
            &scene.upper_margin,
            outer,
            true,
            deadline,
        );
    }
    if scene.lower_status == EyelidObservationStatus::Observed && Instant::now() < deadline {
        scene.lower_lashes = nautilus_lashes(
            raw,
            width,
            height,
            &scene.lower_margin,
            outer,
            false,
            deadline,
        );
    }
    scene.elapsed_us = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
    scene
}

/// Frame-local subset of the Python 21-point native-log detector. It retains
/// RAW10 log(R/G), log(B/G), dark-void gating, radial luma transition, the
/// 58/42 baseline-to-dark-mass radius estimate, and circular neighbor
/// regularization. Temporal, grain, glint, and 3D-plane terms remain owned by
/// the online model and are deliberately not fabricated here.
fn inner_margin_transition_cues(
    luma: &BoxLuma5,
    native: &NativeLogPlane,
    full_resolution_void: Option<&BorrowedFullResolutionPupilView<'_>>,
    x: f64,
    y: f64,
    direction_x: f64,
    direction_y: f64,
) -> (f64, f64, f64) {
    // A tightly focused pupil margin is not necessarily a single sharp edge:
    // oblique corneal imaging can reveal iris fibres through a several-pixel
    // transition band. Integrate the positive radial gradient across that
    // band and favor the location at its energy centroid. A lone fibre can
    // contribute, but cannot move the boundary without neighboring gradients.
    let mut gradient_sum = 0.0;
    let mut weighted_offset = 0.0;
    let mut coherent_gradients = 0usize;
    for offset in [-6.0f64, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0] {
        let center_x = x + direction_x * offset;
        let center_y = y + direction_y * offset;
        let inward = luma.sample3(center_x - direction_x, center_y - direction_y);
        let outward = luma.sample3(center_x + direction_x, center_y + direction_y);
        let gradient = (outward - inward).max(0.0);
        if gradient >= 2.0 {
            coherent_gradients += 1;
        }
        gradient_sum += gradient;
        weighted_offset += offset * gradient;
    }
    let centroid_offset = weighted_offset / gradient_sum.max(1.0e-6);
    let centered = (-0.5 * (centroid_offset / 2.75).powi(2)).exp();
    let deep_inward = luma.sample3(x - direction_x * 8.0, y - direction_y * 8.0);
    let deep_outward = luma.sample3(x + direction_x * 8.0, y + direction_y * 8.0);
    let broad_rise = ((deep_outward - deep_inward) / 128.0).clamp(0.0, 1.0);
    let coherence = (coherent_gradients as f64 / 4.0).clamp(0.35, 1.0);
    let luma_transition = broad_rise * centered * coherence;

    let mut chroma_transition = 0.0;
    let mut void_drop = 0.0;
    for distance in [3.0, 6.0] {
        let (inside_rg, inside_bg) =
            native.sample_chroma(x - direction_x * distance, y - direction_y * distance);
        let (outside_rg, outside_bg) =
            native.sample_chroma(x + direction_x * distance, y + direction_y * distance);
        chroma_transition +=
            ((outside_rg - inside_rg).hypot(outside_bg - inside_bg) / 0.25).clamp(0.0, 1.0);
        let inside_point = (x - direction_x * distance, y - direction_y * distance);
        let outside_point = (x + direction_x * distance, y + direction_y * distance);
        let inside_void = full_resolution_void.map_or_else(
            || native.sample_void(inside_point.0, inside_point.1),
            |view| view.sample_void(inside_point.0, inside_point.1),
        );
        let outside_void = full_resolution_void.map_or_else(
            || native.sample_void(outside_point.0, outside_point.1),
            |view| view.sample_void(outside_point.0, outside_point.1),
        );
        void_drop += (inside_void - outside_void).clamp(0.0, 1.0);
    }
    (luma_transition, chroma_transition * 0.5, void_drop * 0.5)
}

/// Borrowed, full-coordinate pupil-material view for the ordinary live pupil
/// solver. The RAW frame remains owned by the caller and no pupil image,
/// pyramid, or decimated geometry plane is constructed. Quad-Bayer chroma is
/// naturally measured per 4x4 color cell, but it is interpolated only as a
/// material cue; the darkness term and every candidate coordinate retain the
/// native RAW lattice.
struct BorrowedFullResolutionPupilView<'a> {
    width: usize,
    height: usize,
    luma: &'a BoxLuma5,
    native_material: &'a NativeLogPlane,
    intensity_low: f64,
    intensity_high: f64,
}

impl<'a> BorrowedFullResolutionPupilView<'a> {
    fn new(
        raw: &[u16],
        width: usize,
        height: usize,
        luma: &'a BoxLuma5,
        native_material: &'a NativeLogPlane,
    ) -> Option<Self> {
        let population = width.saturating_mul(height);
        if width < 8 || height < 8 || raw.len() < population {
            return None;
        }
        let mut histogram = [0u32; 1024];
        for &value in raw.iter().take(population) {
            histogram[usize::from(value.min(1023))] += 1;
        }
        let intensity_low = histogram_percentile(&histogram, population, 1) as f64;
        let intensity_high =
            (histogram_percentile(&histogram, population, 99) as f64).max(intensity_low + 1.0);
        Some(Self {
            width,
            height,
            luma,
            native_material,
            intensity_low,
            intensity_high,
        })
    }

    #[inline]
    fn combine_void(&self, intensity: f64, x: f64, y: f64) -> f64 {
        let darkness = (1.0
            - (intensity - self.intensity_low) / (self.intensity_high - self.intensity_low))
            .clamp(0.0, 1.0)
            .powf(0.72);
        let material_void = self.native_material.sample_void(x, y);
        (0.46 * darkness + 0.54 * material_void).clamp(0.0, 1.0)
    }

    /// Native-coordinate 3x3 RAW luma plus physical Quad-Bayer material.
    #[inline]
    fn sample_void(&self, x: f64, y: f64) -> f64 {
        self.combine_void(self.luma.sample3(x, y), x, y)
    }

    /// The center finder deliberately uses a slightly broader 5x5 RAW box;
    /// this suppresses CFA/glint noise without changing the image lattice.
    #[inline]
    fn sample_center_void(&self, x: f64, y: f64) -> f64 {
        self.combine_void(self.luma.sample(x, y), x, y)
    }
}

trait FullResolutionPupilView {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn center_void(&self, x: f64, y: f64) -> f64;
}

impl FullResolutionPupilView for BorrowedFullResolutionPupilView<'_> {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn center_void(&self, x: f64, y: f64) -> f64 {
        self.sample_center_void(x, y)
    }
}

fn full_resolution_pupil_mass_radius(
    view: &BorrowedFullResolutionPupilView<'_>,
    center: (f64, f64),
    coarse: &BorderFocus,
) -> Option<f64> {
    let coarse_limit = coarse.radius * 0.92;
    let minimum_x = (coarse.center.0 - coarse_limit).floor().max(1.0) as usize;
    let maximum_x = (coarse.center.0 + coarse_limit)
        .ceil()
        .min(view.width.saturating_sub(2) as f64) as usize;
    let minimum_y = (coarse.center.1 - coarse_limit).floor().max(1.0) as usize;
    let maximum_y = (coarse.center.1 + coarse_limit)
        .ceil()
        .min(view.height.saturating_sub(2) as f64) as usize;
    if minimum_x > maximum_x || minimum_y > maximum_y {
        return None;
    }

    let mut histogram = [0u32; 256];
    let mut population = 0usize;
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            let point = (x as f64, y as f64);
            if (point.0 - coarse.center.0).hypot(point.1 - coarse.center.1) > coarse_limit {
                continue;
            }
            let bin = (view.sample_void(point.0, point.1) * 255.0)
                .round()
                .clamp(0.0, 255.0) as usize;
            histogram[bin] += 1;
            population += 1;
        }
    }
    if population < 20 {
        return None;
    }
    let threshold = histogram_percentile(&histogram, population, 70);
    let center_limit = coarse.radius * 0.65;
    let mut mass_samples = 0usize;
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            let point = (x as f64, y as f64);
            if (point.0 - center.0).hypot(point.1 - center.1) > center_limit
                || (point.0 - coarse.center.0).hypot(point.1 - coarse.center.1) > coarse_limit
            {
                continue;
            }
            let bin = (view.sample_void(point.0, point.1) * 255.0)
                .round()
                .clamp(0.0, 255.0) as usize;
            mass_samples += usize::from(bin >= threshold);
        }
    }
    Some((mass_samples.max(1) as f64 / PI).sqrt())
}

/// Borrowed full-coordinate pupil view used only by Driving.
///
/// The Quad-Bayer log plane contributes material identity, but never defines
/// the pupil's spatial lattice. Every candidate coordinate and every luma
/// sample below is evaluated against the native unpacked RAW10 dimensions.
/// The box-luma integral is a derived statistic over every RAW sample; it is
/// not a resized image and the source pixels remain borrowed by the caller.
struct DrivingFullResolutionPupilView {
    width: usize,
    height: usize,
    // This is a full-resolution derived field, not a copy or resize of the
    // RAW frame. Arc lets broad recovery reuse the native-coordinate luma
    // evidence while rebuilding only its alternate material hypothesis.
    darkness: Arc<Vec<f64>>,
    blurred_void: Vec<f64>,
}

impl DrivingFullResolutionPupilView {
    fn new(
        raw: &[u16],
        width: usize,
        height: usize,
        luma: &BoxLuma5,
        native_material: &NativeLogPlane,
    ) -> Self {
        let mut histogram = [0usize; 1024];
        for &value in raw.iter().take(width.saturating_mul(height)) {
            histogram[usize::from(value.min(1023))] += 1;
        }
        let percentile = |fraction: f64| {
            let population = histogram.iter().sum::<usize>().max(1);
            let target = ((population - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
            let mut cumulative = 0usize;
            for (code, count) in histogram.iter().copied().enumerate() {
                cumulative += count;
                if cumulative > target {
                    return code as f64;
                }
            }
            1023.0
        };
        let low = percentile(0.01);
        let high = percentile(0.99).max(low + 1.0);
        let darkness = Arc::new(
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let intensity = luma.integer_sample3(x, y);
                        (1.0 - (intensity - low) / (high - low))
                            .clamp(0.0, 1.0)
                            .powf(0.72)
                    })
                })
                .collect::<Vec<_>>(),
        );
        let blurred_void =
            Self::build_blurred_void(width, height, darkness.as_slice(), native_material);
        Self {
            width,
            height,
            darkness,
            blurred_void,
        }
    }

    fn with_material(&self, native_material: &NativeLogPlane) -> Self {
        DrivingFullResolutionPupilView {
            width: self.width,
            height: self.height,
            darkness: Arc::clone(&self.darkness),
            blurred_void: Self::build_blurred_void(
                self.width,
                self.height,
                self.darkness.as_slice(),
                native_material,
            ),
        }
    }

    fn build_blurred_void(
        width: usize,
        height: usize,
        darkness: &[f64],
        native_material: &NativeLogPlane,
    ) -> Vec<f64> {
        if width == 0 || height == 0 || darkness.len() < width.saturating_mul(height) {
            return Vec::new();
        }
        const OFFSETS: [isize; 5] = [-4, -2, 0, 2, 4];
        const WEIGHTS: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
        let mut combined = vec![0.0; width * height];
        for y in 0..height {
            for x in 0..width {
                let material_void = native_material.sample_void(x as f64, y as f64);
                // Geometry and darkness remain one value per RAW sample. The
                // physical 4x4 Quad-Bayer color cell is only a material cue;
                // it never changes this field's dimensions or coordinates.
                combined[y * width + x] =
                    (0.34 * darkness[y * width + x] + 0.66 * material_void).clamp(0.0, 1.0);
            }
        }

        // Precompute the exact separable 1-4-6-4-1 pupil-scale blur once per
        // full-size field. Candidate evaluation then remains O(1) while still
        // addressing every native RAW coordinate; no preview or resize is
        // created anywhere in this path.
        let mut horizontal = vec![0.0; combined.len()];
        for y in 0..height {
            for x in 0..width {
                horizontal[y * width + x] = OFFSETS
                    .into_iter()
                    .zip(WEIGHTS)
                    .map(|(offset, weight)| {
                        let sample_x = x.saturating_add_signed(offset).min(width - 1);
                        combined[y * width + sample_x] * weight
                    })
                    .sum::<f64>()
                    / 16.0;
            }
        }
        let mut blurred = vec![0.0; combined.len()];
        for y in 0..height {
            for x in 0..width {
                blurred[y * width + x] = OFFSETS
                    .into_iter()
                    .zip(WEIGHTS)
                    .map(|(offset, weight)| {
                        let sample_y = y.saturating_add_signed(offset).min(height - 1);
                        horizontal[sample_y * width + x] * weight
                    })
                    .sum::<f64>()
                    / 16.0;
            }
        }
        blurred
    }

    #[inline]
    fn blurred_sample(&self, x: f64, y: f64) -> f64 {
        if self.width == 0 || self.height == 0 || self.blurred_void.is_empty() {
            return 0.0;
        }
        let x = x.clamp(0.0, self.width.saturating_sub(1) as f64);
        let y = y.clamp(0.0, self.height.saturating_sub(1) as f64);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let top = self.blurred_void[y0 * self.width + x0] * (1.0 - fx)
            + self.blurred_void[y0 * self.width + x1] * fx;
        let bottom = self.blurred_void[y1 * self.width + x0] * (1.0 - fx)
            + self.blurred_void[y1 * self.width + x1] * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

impl FullResolutionPupilView for DrivingFullResolutionPupilView {
    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn center_void(&self, x: f64, y: f64) -> f64 {
        self.blurred_sample(x, y)
    }
}

fn full_resolution_pupil_local_threshold(
    view: &impl FullResolutionPupilView,
    center: (f64, f64),
    coarse: &BorderFocus,
    window_radius: f64,
    maximum_center_offset: f64,
) -> Option<f64> {
    let mut histogram = [0usize; 256];
    let minimum_x = (center.0 - window_radius).floor().max(2.0) as usize;
    let maximum_x = (center.0 + window_radius)
        .ceil()
        .min(view.width().saturating_sub(3) as f64) as usize;
    let minimum_y = (center.1 - window_radius).floor().max(2.0) as usize;
    let maximum_y = (center.1 + window_radius)
        .ceil()
        .min(view.height().saturating_sub(3) as f64) as usize;
    let mut population = 0usize;
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            let point = (x as f64, y as f64);
            if (point.0 - center.0).hypot(point.1 - center.1) > window_radius
                || (point.0 - coarse.center.0).hypot(point.1 - coarse.center.1)
                    > maximum_center_offset.max(coarse.radius * 0.92)
            {
                continue;
            }
            let bin = (view.center_void(point.0, point.1) * 255.0)
                .round()
                .clamp(0.0, 255.0) as usize;
            histogram[bin] += 1;
            population += 1;
        }
    }
    if population < 12 {
        return None;
    }
    let target = ((population - 1) as f64 * 0.35).round() as usize;
    let mut cumulative = 0usize;
    for (bin, count) in histogram.into_iter().enumerate() {
        cumulative += count;
        if cumulative > target {
            return Some(bin as f64 / 255.0);
        }
    }
    Some(1.0)
}

fn locate_pupil_center_full_resolution_with_limit(
    view: &impl FullResolutionPupilView,
    coarse: &BorderFocus,
    maximum_center_offset_factor: f64,
) -> Option<PupilDriveDiagnostics> {
    let radius = coarse.radius;
    let core_offset = (radius * 0.10).clamp(4.0, 11.0);
    let maximum_center_offset = radius * maximum_center_offset_factor.clamp(0.20, 1.15);
    let hint = coarse.pupil_hint.filter(|hint| {
        coarse.pupil_hint_score > 0.0
            && (hint.0 - coarse.center.0).hypot(hint.1 - coarse.center.1) <= maximum_center_offset
    });
    let minimum_x = (coarse.center.0 - maximum_center_offset).floor().max(6.0) as usize;
    let maximum_x = (coarse.center.0 + maximum_center_offset)
        .ceil()
        .min(view.width().saturating_sub(7) as f64) as usize;
    let minimum_y = (coarse.center.1 - maximum_center_offset).floor().max(6.0) as usize;
    let maximum_y = (coarse.center.1 + maximum_center_offset)
        .ceil()
        .min(view.height().saturating_sub(7) as f64) as usize;
    if minimum_x > maximum_x || minimum_y > maximum_y {
        return None;
    }
    let mut best: Option<(f64, (f64, f64))> = None;
    for y in minimum_y..=maximum_y {
        for x in minimum_x..=maximum_x {
            let point = (x as f64, y as f64);
            let center_offset = (point.0 - coarse.center.0).hypot(point.1 - coarse.center.1);
            if center_offset > maximum_center_offset {
                continue;
            }
            let mut core = [0.0; 9];
            core[0] = view.center_void(point.0, point.1);
            for step in 0..8 {
                let angle = 2.0 * PI * step as f64 / 8.0;
                core[step + 1] = view.center_void(
                    point.0 + core_offset * angle.cos(),
                    point.1 + core_offset * angle.sin(),
                );
            }
            core.sort_by(f64::total_cmp);
            let basin_support = 0.58 * core[2] + 0.42 * core[4];
            let center_penalty = 0.025 * (center_offset / radius.max(1.0)).powi(2);
            let hint_reward = hint.map_or(0.0, |hint| {
                let distance = (point.0 - hint.0).hypot(point.1 - hint.1);
                0.045 * (-0.5 * (distance / (radius * 0.20).max(4.0)).powi(2)).exp()
            });
            let score = basin_support + hint_reward - center_penalty;
            if best.as_ref().is_none_or(|old| score > old.0) {
                best = Some((score, point));
            }
        }
    }
    let (acquisition_score, mut center) = best?;
    let start = coarse.center;
    let mut trace = vec![start];
    if (center.0 - start.0).hypot(center.1 - start.1) > 0.05 {
        trace.push(center);
    }

    // Full-coordinate mean shift. No four-pixel grid or resized image enters
    // this refinement; every integer RAW location in the local basin votes.
    let window_radius = radius * 0.62;
    for _ in 0..3 {
        let Some(threshold) = full_resolution_pupil_local_threshold(
            view,
            center,
            coarse,
            window_radius,
            maximum_center_offset,
        ) else {
            break;
        };
        let minimum_x = (center.0 - window_radius).floor().max(2.0) as usize;
        let maximum_x = (center.0 + window_radius)
            .ceil()
            .min(view.width().saturating_sub(3) as f64) as usize;
        let minimum_y = (center.1 - window_radius).floor().max(2.0) as usize;
        let maximum_y = (center.1 + window_radius)
            .ceil()
            .min(view.height().saturating_sub(3) as f64) as usize;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut total_weight = 0.0;
        for y in minimum_y..=maximum_y {
            for x in minimum_x..=maximum_x {
                let point = (x as f64, y as f64);
                let local_distance = (point.0 - center.0).hypot(point.1 - center.1);
                if local_distance > window_radius
                    || (point.0 - coarse.center.0).hypot(point.1 - coarse.center.1)
                        > maximum_center_offset.max(radius * 0.92)
                {
                    continue;
                }
                let excess = (view.center_void(point.0, point.1) - threshold).max(0.0);
                let spatial = (-0.5 * (local_distance / (radius * 0.45).max(6.0)).powi(2)).exp();
                let weight = excess.powi(3) * spatial;
                weighted_x += point.0 * weight;
                weighted_y += point.1 * weight;
                total_weight += weight;
            }
        }
        if total_weight <= 1.0e-8 {
            break;
        }
        let next = (weighted_x / total_weight, weighted_y / total_weight);
        if (next.0 - coarse.center.0).hypot(next.1 - coarse.center.1) > maximum_center_offset {
            break;
        }
        let movement = (next.0 - center.0).hypot(next.1 - center.1);
        center = next;
        trace.push(center);
        if movement < 0.20 {
            break;
        }
    }

    let enclosure_radius = (coarse.radius * 0.40).clamp(14.0, 34.0);
    let center_void = view.center_void(center.0, center.1);
    let mut contrast_sum = 0.0;
    let mut supported = 0usize;
    for step in 0..16 {
        let angle = 2.0 * PI * step as f64 / 16.0;
        let ring_void = view.center_void(
            center.0 + enclosure_radius * angle.cos(),
            center.1 + enclosure_radius * angle.sin(),
        );
        let contrast = center_void - ring_void;
        contrast_sum += contrast.max(0.0);
        if contrast >= 0.018 {
            supported += 1;
        }
    }
    let contrast = (contrast_sum / 16.0 / 0.12).clamp(0.0, 1.0);
    let coverage = supported as f64 / 16.0;
    let enclosure_score = (0.62 * contrast + 0.38 * coverage).clamp(0.0, 1.0);
    Some(PupilDriveDiagnostics {
        start,
        center,
        trace,
        acquisition_score,
        enclosure_score,
        travel_px: (center.0 - start.0).hypot(center.1 - start.1),
        consensus_members: 1,
    })
}

/// Run the same native-log pupil acquisition used by the live inner-boundary
/// detector while retaining its short coarse-to-fine steering trace.
pub fn debug_drive_pupil_center(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
) -> Option<PupilDriveDiagnostics> {
    if raw.len() < width.saturating_mul(height)
        || coarse.radius < 20.0
        || !coarse.center.0.is_finite()
        || !coarse.center.1.is_finite()
    {
        return None;
    }
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)?;
    let luma = BoxLuma5::new(raw, width, height);
    let view = DrivingFullResolutionPupilView::new(raw, width, height, &luma, &native);
    locate_pupil_center_full_resolution_with_limit(&view, coarse, 0.68)
}

/// Return several spatially distinct pupil basins for Driving's off-iris
/// recovery. This is intentionally more expensive than the ordinary pupil
/// path and should only run after that path fails. Each broad anchor is
/// re-estimated with its own local iris chroma reference before being exposed
/// to the caller; the caller remains responsible for full anatomy admission.
pub fn debug_drive_pupil_candidates_broad(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
) -> Vec<PupilDriveDiagnostics> {
    if raw.len() < width.saturating_mul(height)
        || coarse.radius < 20.0
        || !coarse.center.0.is_finite()
        || !coarse.center.1.is_finite()
    {
        return Vec::new();
    }
    let Some(native) = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse) else {
        return Vec::new();
    };
    let luma = BoxLuma5::new(raw, width, height);
    let view = DrivingFullResolutionPupilView::new(raw, width, height, &luma, &native);
    let radius = coarse.radius;
    let core_offset = (radius * 0.10).clamp(4.0, 11.0);
    // A first lap can be a small local lid/hair contour as well as an
    // oversized eye-aperture contour. Preserve the scale-relative corridor,
    // but guarantee enough frame-relative reach for the former so the true
    // pupil can seed a larger partial second lap.
    let maximum_center_offset = (radius * 1.12).max(width.min(height) as f64 * 0.28);
    let mut anchors = Vec::new();
    let minimum_x = (coarse.center.0 - maximum_center_offset).floor().max(6.0) as usize;
    let maximum_x = (coarse.center.0 + maximum_center_offset)
        .ceil()
        .min(width.saturating_sub(7) as f64) as usize;
    let minimum_y = (coarse.center.1 - maximum_center_offset).floor().max(6.0) as usize;
    let maximum_y = (coarse.center.1 + maximum_center_offset)
        .ceil()
        .min(height.saturating_sub(7) as f64) as usize;
    if minimum_x > maximum_x || minimum_y > maximum_y {
        return Vec::new();
    }
    for sample_y in minimum_y..=maximum_y {
        for sample_x in minimum_x..=maximum_x {
            let x = sample_x as f64;
            let y = sample_y as f64;
            let center_offset = (x - coarse.center.0).hypot(y - coarse.center.1);
            if center_offset > maximum_center_offset {
                continue;
            }
            let mut core = [0.0; 9];
            core[0] = view.blurred_sample(x, y);
            for step in 0..8 {
                let angle = 2.0 * PI * step as f64 / 8.0;
                core[step + 1] = view
                    .blurred_sample(x + core_offset * angle.cos(), y + core_offset * angle.sin());
            }
            core.sort_by(f64::total_cmp);
            let basin_support = 0.58 * core[2] + 0.42 * core[4];
            let center_penalty = 0.008 * (center_offset / radius.max(1.0)).powi(2);
            anchors.push((basin_support - center_penalty, (x, y)));
        }
    }
    anchors.sort_by(|left, right| right.0.total_cmp(&left.0));
    let suppression_radius = (radius * 0.19).clamp(16.0, 30.0);
    let mut selected = Vec::new();
    for anchor in anchors {
        if selected.iter().any(|(_, point): &(f64, (f64, f64))| {
            (point.0 - anchor.1 .0).hypot(point.1 - anchor.1 .1) < suppression_radius
        }) {
            continue;
        }
        selected.push(anchor);
        if selected.len() >= 8 {
            break;
        }
    }

    let mut candidates = Vec::<PupilDriveDiagnostics>::new();
    // The independent coarse detector often has a compact pupil basin within
    // a few native pixels even when the broad material map is split by a
    // corneal glint. Preserve that as one *measured* proposal: rebuild the
    // material hypothesis around it and permit only a small full-resolution
    // refinement. It remains merely a candidate and must later prove the
    // complete sclera|iris|pupil|iris|sclera road.
    let trusted_hint = coarse.pupil_hint.filter(|hint| {
        coarse.pupil_hint_score > 0.0
            && hint.0.is_finite()
            && hint.1.is_finite()
            && (hint.0 - coarse.center.0).hypot(hint.1 - coarse.center.1) <= maximum_center_offset
    });
    if let Some(hint) = trusted_hint {
        let mut local = coarse.clone();
        local.center = hint;
        local.radius = (radius * 0.52).clamp(24.0, radius);
        local.pupil_hint = Some(hint);
        local.pupil_hint_score = coarse.pupil_hint_score;
        if let Some(local_native) = native_log_plane(raw, width, height, sensor_x, sensor_y, &local)
        {
            let local_view = view.with_material(&local_native);
            if let Some(mut candidate) =
                locate_pupil_center_full_resolution_with_limit(&local_view, &local, 0.25)
            {
                candidate.start = coarse.center;
                if candidate.trace.first().copied() != Some(coarse.center) {
                    candidate.trace.insert(0, coarse.center);
                }
                candidate.travel_px = (candidate.center.0 - coarse.center.0)
                    .hypot(candidate.center.1 - coarse.center.1);
                candidates.push(candidate);
            }
        }
    }
    for (broad_score, anchor) in selected {
        let mut local = coarse.clone();
        local.center = anchor;
        local.radius = (radius * 0.62).clamp(28.0, radius);
        local.pupil_hint = Some(anchor);
        local.pupil_hint_score = 1.0;
        let Some(local_native) = native_log_plane(raw, width, height, sensor_x, sensor_y, &local)
        else {
            continue;
        };
        let local_view = view.with_material(&local_native);
        let Some(mut candidate) =
            locate_pupil_center_full_resolution_with_limit(&local_view, &local, 0.68)
        else {
            continue;
        };
        candidate.acquisition_score = candidate.acquisition_score.max(broad_score);
        candidate.start = coarse.center;
        if candidate.trace.first().copied() != Some(coarse.center) {
            candidate.trace.insert(0, coarse.center);
        }
        candidate.travel_px =
            (candidate.center.0 - coarse.center.0).hypot(candidate.center.1 - coarse.center.1);
        if let Some(existing) = candidates.iter_mut().find(|existing| {
            (existing.center.0 - candidate.center.0).hypot(existing.center.1 - candidate.center.1)
                < 10.0
        }) {
            let existing_quality = existing.enclosure_score + 0.20 * existing.acquisition_score;
            let candidate_quality = candidate.enclosure_score + 0.20 * candidate.acquisition_score;
            if candidate_quality > existing_quality {
                *existing = candidate;
            }
        } else {
            candidates.push(candidate);
        }
    }
    candidates.sort_by(|left, right| {
        let left_quality = left.enclosure_score + 0.20 * left.acquisition_score;
        let right_quality = right.enclosure_score + 0.20 * right.acquisition_score;
        right_quality.total_cmp(&left_quality)
    });

    // A large compact corneal glint can divide the pupil void into two or
    // three dark lobes that are individually well enclosed but displaced in
    // opposite directions. Add their shared center as a bounded recovery
    // proposal. This does not inspect or weight a specular map: two native-log
    // basins must independently exist, and the shared center still competes
    // through the complete sclera|iris|pupil|iris|sclera anatomy scorer.
    let lobes = candidates
        .iter()
        .take(6)
        .filter(|candidate| {
            candidate.enclosure_score >= 0.28 && candidate.acquisition_score >= 0.50
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut shared_centers = Vec::<PupilDriveDiagnostics>::new();
    for first in 0..lobes.len() {
        for second in first + 1..lobes.len() {
            let distance = (lobes[first].center.0 - lobes[second].center.0)
                .hypot(lobes[first].center.1 - lobes[second].center.1);
            if !(radius * 0.48..=radius * 1.10).contains(&distance) {
                continue;
            }
            let center = (
                (lobes[first].center.0 + lobes[second].center.0) * 0.5,
                (lobes[first].center.1 + lobes[second].center.1) * 0.5,
            );
            if (center.0 - coarse.center.0).hypot(center.1 - coarse.center.1) > radius * 0.72 {
                continue;
            }
            shared_centers.push(PupilDriveDiagnostics {
                start: coarse.center,
                center,
                trace: vec![
                    coarse.center,
                    lobes[first].center,
                    lobes[second].center,
                    center,
                ],
                acquisition_score: ((lobes[first].acquisition_score
                    + lobes[second].acquisition_score)
                    * 0.5
                    - 0.08)
                    .max(0.0),
                enclosure_score: (lobes[first].enclosure_score + lobes[second].enclosure_score)
                    * 0.5,
                travel_px: (center.0 - coarse.center.0).hypot(center.1 - coarse.center.1),
                consensus_members: 2,
            });
        }
    }
    for first in 0..lobes.len() {
        for second in first + 1..lobes.len() {
            for third in second + 1..lobes.len() {
                let members = [&lobes[first], &lobes[second], &lobes[third]];
                let distances = [
                    (members[0].center.0 - members[1].center.0)
                        .hypot(members[0].center.1 - members[1].center.1),
                    (members[0].center.0 - members[2].center.0)
                        .hypot(members[0].center.1 - members[2].center.1),
                    (members[1].center.0 - members[2].center.0)
                        .hypot(members[1].center.1 - members[2].center.1),
                ];
                if distances
                    .iter()
                    .any(|distance| !(radius * 0.38..=radius * 1.15).contains(distance))
                {
                    continue;
                }
                let center = (
                    members.iter().map(|member| member.center.0).sum::<f64>() / 3.0,
                    members.iter().map(|member| member.center.1).sum::<f64>() / 3.0,
                );
                if (center.0 - coarse.center.0).hypot(center.1 - coarse.center.1) > radius * 0.72 {
                    continue;
                }
                shared_centers.push(PupilDriveDiagnostics {
                    start: coarse.center,
                    center,
                    trace: vec![
                        coarse.center,
                        members[0].center,
                        members[1].center,
                        members[2].center,
                        center,
                    ],
                    acquisition_score: (members
                        .iter()
                        .map(|member| member.acquisition_score)
                        .sum::<f64>()
                        / 3.0
                        - 0.08)
                        .max(0.0),
                    enclosure_score: members
                        .iter()
                        .map(|member| member.enclosure_score)
                        .sum::<f64>()
                        / 3.0,
                    travel_px: (center.0 - coarse.center.0).hypot(center.1 - coarse.center.1),
                    consensus_members: 3,
                });
            }
        }
    }
    for candidate in shared_centers {
        if candidates.iter().any(|existing| {
            (existing.center.0 - candidate.center.0).hypot(existing.center.1 - candidate.center.1)
                < 10.0
        }) {
            continue;
        }
        candidates.push(candidate);
    }
    candidates.sort_by(|left, right| {
        let left_quality = left.enclosure_score + 0.20 * left.acquisition_score;
        let right_quality = right.enclosure_score + 0.20 * right.acquisition_score;
        right_quality.total_cmp(&left_quality)
    });
    candidates
}

#[derive(Clone, Copy, Debug)]
struct ProjectedAffinePupilRay {
    /// Image-space radial distance divided by projected area-equivalent
    /// radius for this polar ray.
    image_radius_scale: f64,
    /// Unit outward normal of the projected ellipse at the ray intersection.
    outward_normal: (f64, f64),
}

/// Normalize the coarse limbus projection to a major/minor ratio and its
/// image-space major-axis angle. A rough eye-basin fit is search guidance, not
/// authority to invent an extreme projected limbus, so the search affine is
/// clipped to the same provisional camera/anatomy envelope used at
/// publication.
fn normalized_affine_projection(coarse: &BorderFocus) -> (f64, f64) {
    let mut major_to_minor = if coarse.axis_ratio.is_finite() && coarse.axis_ratio.abs() > 1.0e-6 {
        coarse.axis_ratio.abs()
    } else {
        1.0
    };
    let mut angle = if coarse.axis_angle.is_finite() {
        coarse.axis_angle
    } else {
        0.0
    };
    if major_to_minor < 1.0 {
        major_to_minor = 1.0 / major_to_minor;
        angle += PI * 0.5;
    }
    let preliminary_major = coarse.radius.max(1.0) * major_to_minor.sqrt();
    let maximum_ratio = PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE
        .maximum_major_to_minor(preliminary_major)
        .unwrap_or(1.0);
    (major_to_minor.clamp(1.0, maximum_ratio), angle)
}

/// Map one projected area-equivalent pupil radius onto an image-space polar
/// ray under the limbus affine-circle projection. If q is minor/major, an
/// equivalent radius R has axes R/sqrt(q) and R*sqrt(q). The edge cues are
/// sampled along the corresponding conic normal rather than along the polar
/// ray, which matters increasingly as the eye turns away from the camera.
fn projected_affine_pupil_ray(coarse: &BorderFocus, ray_angle: f64) -> ProjectedAffinePupilRay {
    let (major_to_minor, major_axis_angle) = normalized_affine_projection(coarse);
    let projection_ratio = 1.0 / major_to_minor;
    let relative_angle = ray_angle - major_axis_angle;
    let (relative_sin, relative_cos) = relative_angle.sin_cos();
    let image_radius_scale = 1.0
        / (projection_ratio * relative_cos * relative_cos
            + relative_sin * relative_sin / projection_ratio)
            .sqrt();

    // The equivalent-radius ellipse has local squared axes proportional to
    // major_to_minor and 1/major_to_minor. Its implicit-gradient normal is
    // therefore (x/major_to_minor, y*major_to_minor).
    let normal_local_x = relative_cos / major_to_minor;
    let normal_local_y = relative_sin * major_to_minor;
    let (axis_sin, axis_cos) = major_axis_angle.sin_cos();
    let mut normal_x = axis_cos * normal_local_x - axis_sin * normal_local_y;
    let mut normal_y = axis_sin * normal_local_x + axis_cos * normal_local_y;
    let normal_length = normal_x.hypot(normal_y).max(1.0e-9);
    normal_x /= normal_length;
    normal_y /= normal_length;
    ProjectedAffinePupilRay {
        image_radius_scale,
        outward_normal: (normal_x, normal_y),
    }
}

/// Transition cues look eight pixels to either side and the luma sampler has
/// its own small support. Do not turn clamped edge pixels into fabricated
/// pupil evidence. Missing rays remain censored and are excluded from the fit.
fn inner_margin_candidate_is_observable(x: f64, y: f64, width: usize, height: usize) -> bool {
    const SAMPLE_MARGIN: f64 = 10.0;
    let maximum_x = width.saturating_sub(1) as f64 - SAMPLE_MARGIN;
    let maximum_y = height.saturating_sub(1) as f64 - SAMPLE_MARGIN;
    maximum_x >= SAMPLE_MARGIN
        && maximum_y >= SAMPLE_MARGIN
        && (SAMPLE_MARGIN..=maximum_x).contains(&x)
        && (SAMPLE_MARGIN..=maximum_y).contains(&y)
}

/// Result of a bounded native-RAW search which translates a fixed projected
/// pupil circle inside an already finalized limbus. The pupil radius and the
/// outer affine projection are inputs, never search degrees of freedom.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PupilCenterOrbitalFit {
    pub center: (f64, f64),
    pub score: f64,
    pub ring_transition: f64,
    pub ring_coverage: f64,
    pub opposing_support: f64,
    pub interior_void: f64,
    /// Median exposure-normalized luma increase from the broad pupil interior
    /// to the iris immediately beyond the proposed fixed-radius rim.  A thin
    /// dark eyelash, iris texture ring, or reflection moat can have a strong
    /// local edge while remaining bright on its inside; those aliases have a
    /// non-positive value here.
    pub broad_dark_step: f64,
    /// Fraction of independently sampled meridians whose broad interior-to-
    /// iris ordering is positive.  This is kept separate from the local edge
    /// coverage so one cohesive shadow gradient cannot impersonate a pupil.
    pub broad_dark_support: f64,
    pub canonical_radius: f64,
    pub evaluated_centers: usize,
}

/// One bounded cold-start comparison between the independently measured
/// compact-pupil geometry and a more distant/radially larger topology edge.
/// The radius is the projected area-equivalent radius in native RAW pixels;
/// it is deliberately not a fronto-parallel limbus radius.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PupilGeometryPriorFit {
    pub measurement: PupilCenterOrbitalFit,
    pub equivalent_radius: f64,
    pub incumbent_score: Option<f64>,
    pub used_prior_geometry: bool,
}

/// Compare an existing pupil center with a high-confidence, independently
/// acquired center prior while holding the de-affined pupil radius and limbus
/// projection fixed.
///
/// The prior gets only a small native-pixel neighborhood, so this cannot turn
/// into another open-ended iris/lid search. A remote incumbent survives only
/// when its current-frame fixed-ring evidence exceeds the best prior-local
/// fit by `incumbent_outvote_margin`. This is the intended Bayesian ordering
/// for the ordinary frontal case: the rough center is the dominant prior, but
/// decisive untouched RAW evidence can still disprove it.
#[allow(clippy::too_many_arguments)]
pub fn select_pupil_center_near_strong_prior_at_fixed_radius(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    incumbent_center: (f64, f64),
    prior_center: (f64, f64),
    equivalent_radius: f64,
    maximum_prior_refinement_px: f64,
    incumbent_outvote_margin: f64,
) -> Option<PupilCenterOrbitalFit> {
    if raw.len() < width.saturating_mul(height)
        || coarse.radius < 20.0
        || !equivalent_radius.is_finite()
        || equivalent_radius <= 4.0
        || equivalent_radius >= coarse.radius * 0.72
        || !incumbent_center.0.is_finite()
        || !incumbent_center.1.is_finite()
        || !prior_center.0.is_finite()
        || !prior_center.1.is_finite()
    {
        return None;
    }
    let maximum_prior_refinement_px = if maximum_prior_refinement_px.is_finite() {
        maximum_prior_refinement_px.clamp(1.0, 18.0)
    } else {
        return None;
    };
    let incumbent_outvote_margin = if incumbent_outvote_margin.is_finite() {
        incumbent_outvote_margin.clamp(0.0, 0.50)
    } else {
        return None;
    };
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)?;
    let luma = BoxLuma5::new(raw, width, height);
    let pupil_view = BorrowedFullResolutionPupilView::new(raw, width, height, &luma, &native)?;
    let prior_sigma = (maximum_prior_refinement_px * 0.55).max(2.0);
    let temporal_prior = Some((prior_center, prior_sigma));
    let mut evaluated = 0usize;
    let mut evaluate = |center| {
        evaluated += 1;
        fixed_radius_pupil_center_evidence(
            &luma,
            &native,
            &pupil_view,
            width,
            height,
            coarse,
            center,
            equivalent_radius,
            temporal_prior,
        )
    };

    let incumbent = evaluate(incumbent_center);
    let mut prior_best = evaluate(prior_center)?;
    let mut center = prior_best.center;
    for step in [
        maximum_prior_refinement_px,
        maximum_prior_refinement_px * 0.50,
        maximum_prior_refinement_px * 0.25,
        maximum_prior_refinement_px * 0.125,
    ] {
        let mut next = prior_best;
        for direction in 0..8 {
            let phase = 2.0 * PI * direction as f64 / 8.0;
            let candidate_center = (center.0 + step * phase.cos(), center.1 + step * phase.sin());
            if (candidate_center.0 - prior_center.0).hypot(candidate_center.1 - prior_center.1)
                > maximum_prior_refinement_px + 1.0e-9
            {
                continue;
            }
            if let Some(candidate) = evaluate(candidate_center) {
                if candidate.score > next.score {
                    next = candidate;
                }
            }
        }
        prior_best = next;
        center = prior_best.center;
    }

    let incumbent_distance =
        (incumbent_center.0 - prior_center.0).hypot(incumbent_center.1 - prior_center.1);
    let mut selected = match incumbent {
        Some(incumbent)
            if incumbent_distance > maximum_prior_refinement_px
                && incumbent.score > prior_best.score + incumbent_outvote_margin =>
        {
            incumbent
        }
        Some(incumbent)
            if incumbent_distance <= maximum_prior_refinement_px
                && incumbent.score > prior_best.score =>
        {
            incumbent
        }
        _ => prior_best,
    };
    selected.evaluated_centers = evaluated;
    Some(selected)
}

/// Test a small native-resolution center/radius bank around an independent
/// compact-pupil estimate before a cold tracker teaches itself the radius of
/// an iris texture band.  The incumbent may still win, but only by materially
/// outscoring the best compact-prior road in untouched RAW evidence.
#[allow(clippy::too_many_arguments)]
pub fn select_pupil_geometry_near_strong_prior(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    incumbent_center: (f64, f64),
    incumbent_equivalent_radius: f64,
    prior_center: (f64, f64),
    prior_equivalent_radius: f64,
    maximum_prior_refinement_px: f64,
    incumbent_outvote_margin: f64,
) -> Option<PupilGeometryPriorFit> {
    if raw.len() < width.saturating_mul(height)
        || coarse.radius < 20.0
        || !incumbent_equivalent_radius.is_finite()
        || incumbent_equivalent_radius <= 4.0
        || incumbent_equivalent_radius >= coarse.radius * 0.72
        || !prior_equivalent_radius.is_finite()
        || prior_equivalent_radius <= 4.0
        || prior_equivalent_radius >= coarse.radius * 0.65
        || !incumbent_center.0.is_finite()
        || !incumbent_center.1.is_finite()
        || !prior_center.0.is_finite()
        || !prior_center.1.is_finite()
    {
        return None;
    }
    let maximum_prior_refinement_px = maximum_prior_refinement_px.clamp(1.0, 18.0);
    let incumbent_outvote_margin = incumbent_outvote_margin.clamp(0.0, 0.50);
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)?;
    let luma = BoxLuma5::new(raw, width, height);
    let pupil_view = BorrowedFullResolutionPupilView::new(raw, width, height, &luma, &native)?;
    let incumbent = fixed_radius_pupil_center_evidence(
        &luma,
        &native,
        &pupil_view,
        width,
        height,
        coarse,
        incumbent_center,
        incumbent_equivalent_radius,
        None,
    );

    let mut radii = [0.75, 0.875, 1.0, 1.125, 1.25]
        .into_iter()
        .map(|scale| prior_equivalent_radius * scale)
        .filter(|radius| (5.0..coarse.radius * 0.65).contains(radius))
        .collect::<Vec<_>>();
    radii.push(prior_equivalent_radius);
    radii.sort_by(f64::total_cmp);
    radii.dedup_by(|left, right| (*left - *right).abs() < 0.50);
    let mut evaluated = usize::from(incumbent.is_some());
    // Radius evidence at the independent center is cheap and discriminative.
    // Probe the whole compact bank once, then spend the native-center walk on
    // only its two strongest radii. This preserves the 0.75..1.25 scale reach
    // while keeping the final-candidate correction inside a fixed small
    // budget rather than multiplying every radius by every orbital point.
    let mut radius_seeds = radii
        .into_iter()
        .filter_map(|radius| {
            evaluated += 1;
            fixed_radius_pupil_center_evidence(
                &luma,
                &native,
                &pupil_view,
                width,
                height,
                coarse,
                prior_center,
                radius,
                None,
            )
            .map(|fit| (radius, fit))
        })
        .collect::<Vec<_>>();
    radius_seeds.sort_by(|left, right| right.1.score.total_cmp(&left.1.score));
    radius_seeds.truncate(2);
    let mut prior_best = None::<(f64, PupilCenterOrbitalFit)>;
    for (radius, seed_fit) in radius_seeds {
        let mut candidate = Some(seed_fit);
        let mut center = seed_fit.center;
        for step in [
            maximum_prior_refinement_px,
            maximum_prior_refinement_px * 0.50,
            maximum_prior_refinement_px * 0.25,
        ] {
            for direction in 0..8 {
                let phase = 2.0 * PI * direction as f64 / 8.0;
                let proposed = (center.0 + step * phase.cos(), center.1 + step * phase.sin());
                if (proposed.0 - prior_center.0).hypot(proposed.1 - prior_center.1)
                    > maximum_prior_refinement_px + 1.0e-9
                {
                    continue;
                }
                evaluated += 1;
                if let Some(fit) = fixed_radius_pupil_center_evidence(
                    &luma,
                    &native,
                    &pupil_view,
                    width,
                    height,
                    coarse,
                    proposed,
                    radius,
                    None,
                ) {
                    if candidate.is_none_or(|old| fit.score > old.score) {
                        candidate = Some(fit);
                        center = fit.center;
                    }
                }
            }
        }
        if let Some(fit) = candidate {
            if prior_best.is_none_or(|(_, old)| fit.score > old.score) {
                prior_best = Some((radius, fit));
            }
        }
    }
    let (prior_radius, prior_fit) = prior_best?;
    let incumbent_score = incumbent.map(|fit| fit.score);
    let (equivalent_radius, mut measurement, used_prior_geometry) = match incumbent {
        Some(incumbent) if incumbent.score > prior_fit.score + incumbent_outvote_margin => {
            (incumbent_equivalent_radius, incumbent, false)
        }
        _ => (prior_radius, prior_fit, true),
    };
    measurement.evaluated_centers = evaluated;
    Some(PupilGeometryPriorFit {
        measurement,
        equivalent_radius,
        incumbent_score,
        used_prior_geometry,
    })
}

#[allow(clippy::too_many_arguments)]
fn fixed_radius_pupil_center_evidence(
    luma: &BoxLuma5,
    native: &NativeLogPlane,
    pupil_view: &BorrowedFullResolutionPupilView<'_>,
    width: usize,
    height: usize,
    coarse: &BorderFocus,
    center: (f64, f64),
    equivalent_radius: f64,
    temporal_prior: Option<((f64, f64), f64)>,
) -> Option<PupilCenterOrbitalFit> {
    const RAYS: usize = 24;
    let (major_to_minor, major_axis_angle) = normalized_affine_projection(coarse);
    let ratio_root = major_to_minor.sqrt();
    let outer_major = coarse.radius * ratio_root;
    let outer_minor = coarse.radius / ratio_root;
    let (axis_sin, axis_cos) = major_axis_angle.sin_cos();
    let delta_x = center.0 - coarse.center.0;
    let delta_y = center.1 - coarse.center.1;
    let canonical_x = (delta_x * axis_cos + delta_y * axis_sin) / outer_major.max(1.0);
    let canonical_y = (-delta_x * axis_sin + delta_y * axis_cos) / outer_minor.max(1.0);
    let canonical_radius = canonical_x.hypot(canonical_y);
    let pupil_ratio = equivalent_radius / coarse.radius.max(1.0);
    // The projected pupil must leave a real iris annulus. This single affine-
    // invariant containment rule rejects a dark lid/glint lobe whose circle
    // would overlap an otherwise excellent limbus fit.
    let maximum_center_radius = (0.92 - pupil_ratio).clamp(0.10, 0.72);
    if canonical_radius > maximum_center_radius {
        return None;
    }

    let (luma_weight, chroma_weight, void_weight, _, _) =
        InnerIrisEvidenceCondition::default().cue_weights();
    let mut scores = [f64::NAN; RAYS];
    let mut broad_dark_steps = [f64::NAN; RAYS];
    let mut observed = 0usize;
    let mut raw_transition_sum = 0.0;
    for (index, score) in scores.iter_mut().enumerate() {
        let angle = 2.0 * PI * index as f64 / RAYS as f64;
        let affine_ray = projected_affine_pupil_ray(coarse, angle);
        let image_radius = equivalent_radius * affine_ray.image_radius_scale;
        let x = center.0 + image_radius * angle.cos();
        let y = center.1 + image_radius * angle.sin();
        if !inner_margin_candidate_is_observable(x, y, width, height) {
            continue;
        }
        let (luma_transition, chroma_transition, void_drop) = inner_margin_transition_cues(
            luma,
            native,
            Some(pupil_view),
            x,
            y,
            affine_ray.outward_normal.0,
            affine_ray.outward_normal.1,
        );
        let inside_void = pupil_view.sample_void(
            x - affine_ray.outward_normal.0 * 3.0,
            y - affine_ray.outward_normal.1 * 3.0,
        );
        let gate = 0.24 + 0.76 * inside_void;
        *score = (luma_weight * luma_transition
            + chroma_weight * chroma_transition
            + void_weight * void_drop)
            * gate;
        // The local edge above can be fooled by a dark iris fibre or the
        // moat surrounding a corneal reflection.  Look farther down the same
        // projected meridian: a real pupil remains broadly darker well inside
        // the rim, then becomes brighter on the iris side.  The paired ratio
        // removes exposure scale and the median across meridians rejects the
        // few pairs crossed by a specular highlight.
        let broad_inside_x = center.0 + image_radius * 0.55 * angle.cos();
        let broad_inside_y = center.1 + image_radius * 0.55 * angle.sin();
        let broad_outside_x = x + affine_ray.outward_normal.0 * 6.0;
        let broad_outside_y = y + affine_ray.outward_normal.1 * 6.0;
        if inner_margin_candidate_is_observable(broad_inside_x, broad_inside_y, width, height)
            && inner_margin_candidate_is_observable(broad_outside_x, broad_outside_y, width, height)
        {
            let inside_luma = luma.sample(broad_inside_x, broad_inside_y);
            let outside_luma = luma.sample(broad_outside_x, broad_outside_y);
            let exposure_scale = (0.5 * (inside_luma.abs() + outside_luma.abs())).max(8.0);
            broad_dark_steps[index] = (outside_luma - inside_luma) / exposure_scale;
        }
        raw_transition_sum += *score;
        observed += 1;
    }
    if observed < 16 {
        return None;
    }
    let mut finite_scores = scores
        .iter()
        .copied()
        .filter(|score| score.is_finite())
        .collect::<Vec<_>>();
    finite_scores.sort_by(f64::total_cmp);
    let ring_transition = finite_scores[finite_scores.len() / 2].max(0.0);
    let ring_coverage = finite_scores.iter().filter(|score| **score >= 0.20).count() as f64
        / finite_scores.len() as f64;
    let mut opposing_pairs = Vec::with_capacity(RAYS / 2);
    for index in 0..RAYS / 2 {
        let opposite = scores[index + RAYS / 2];
        if scores[index].is_finite() && opposite.is_finite() {
            opposing_pairs.push(scores[index].min(opposite));
        }
    }
    if opposing_pairs.len() < 7 {
        return None;
    }
    let opposing_support = opposing_pairs
        .iter()
        .filter(|score| **score >= 0.17)
        .count() as f64
        / opposing_pairs.len() as f64;

    let mut finite_broad_steps = broad_dark_steps
        .iter()
        .copied()
        .filter(|step| step.is_finite())
        .collect::<Vec<_>>();
    // Eight widely separated meridians still provide a useful robust order
    // statistic in a clipped ROI.  With less support, leave this cue neutral
    // instead of turning clamped pixels into evidence against a partial eye.
    let (broad_dark_step, broad_dark_support, broad_dark_score, broad_support_score) =
        if finite_broad_steps.len() >= 8 {
            let step = percentile_f64(&mut finite_broad_steps, 0.50);
            let support = finite_broad_steps
                .iter()
                .filter(|sample| **sample >= 0.012)
                .count() as f64
                / finite_broad_steps.len() as f64;
            (
                step,
                support,
                ((step - 0.005) / 0.075).clamp(0.0, 1.0),
                ((support - 0.30) / 0.55).clamp(0.0, 1.0),
            )
        } else {
            (0.0, 0.0, 0.35, 0.0)
        };

    // A cohesive pupil may contain one or two intense corneal reflections.
    // Use the lower-third and median of nine full-resolution material/void
    // samples so those highlights cannot disqualify the correct dark basin.
    let mut core = [0.0; 9];
    core[0] = pupil_view.sample_void(center.0, center.1);
    for index in 0..8 {
        let angle = 2.0 * PI * index as f64 / 8.0;
        let affine_ray = projected_affine_pupil_ray(coarse, angle);
        let radius = equivalent_radius * 0.34 * affine_ray.image_radius_scale;
        core[index + 1] = pupil_view.sample_void(
            center.0 + radius * angle.cos(),
            center.1 + radius * angle.sin(),
        );
    }
    core.sort_by(f64::total_cmp);
    let interior_void = 0.56 * core[2] + 0.44 * core[4];
    let transition_score = ((ring_transition - 0.10) / 0.34).clamp(0.0, 1.0);
    let mean_transition_score =
        ((raw_transition_sum / observed as f64 - 0.10) / 0.34).clamp(0.0, 1.0);
    let clearance = ((maximum_center_radius - canonical_radius)
        / maximum_center_radius.max(1.0e-6))
    .clamp(0.0, 1.0);
    let temporal = temporal_prior.map_or(0.0, |(prior, sigma)| {
        let distance = (center.0 - prior.0).hypot(center.1 - prior.1);
        (-0.5 * (distance / sigma.max(2.0)).powi(2)).exp()
    });
    let current_frame_score = (0.21 * transition_score
        + 0.12 * mean_transition_score
        + 0.17 * ring_coverage
        + 0.15 * opposing_support
        + 0.10 * interior_void
        + 0.14 * broad_dark_score
        + 0.05 * broad_support_score
        + 0.06 * clearance)
        .clamp(0.0, 1.0);
    // Current RAW remains authoritative, but a transported, already admitted
    // pupil center is meaningful evidence when several similarly strong iris
    // texture rings are present.  Blend rather than add so absence of a prior
    // cannot lower a cold candidate's attainable score.
    let score = if temporal_prior.is_some() {
        (0.92 * current_frame_score + 0.08 * temporal).clamp(0.0, 1.0)
    } else {
        current_frame_score
    };
    Some(PupilCenterOrbitalFit {
        center,
        score,
        ring_transition,
        ring_coverage,
        opposing_support,
        interior_void,
        broad_dark_step,
        broad_dark_support,
        canonical_radius,
        evaluated_centers: 1,
    })
}

/// Translate a fixed-size pupil ellipse inside a finalized limbus by making
/// a small coarse orbital sweep followed by four native-pixel hill-climb
/// passes. All candidates read the original RAW slice through one shared
/// luma/material preparation; no image pyramid, resize, or pupil bitmap is
/// created. The fixed 24-ray x <=80-center budget keeps this final-candidate-
/// only correction independent of the much larger outer-iris proposal bank.
pub fn optimize_pupil_center_orbitally_at_fixed_radius(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    seed: (f64, f64),
    equivalent_radius: f64,
    temporal_prior: Option<((f64, f64), f64)>,
) -> Option<PupilCenterOrbitalFit> {
    if raw.len() < width.saturating_mul(height)
        || coarse.radius < 20.0
        || !equivalent_radius.is_finite()
        || equivalent_radius <= 4.0
        || equivalent_radius >= coarse.radius * 0.72
    {
        return None;
    }
    let native = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse)?;
    let luma = BoxLuma5::new(raw, width, height);
    let pupil_view = BorrowedFullResolutionPupilView::new(raw, width, height, &luma, &native)?;
    let (major_to_minor, angle) = normalized_affine_projection(coarse);
    let ratio_root = major_to_minor.sqrt();
    let outer_major = coarse.radius * ratio_root;
    let outer_minor = coarse.radius / ratio_root;
    let maximum_center_radius =
        (0.92 - equivalent_radius / coarse.radius.max(1.0)).clamp(0.10, 0.72);
    let (axis_sin, axis_cos) = angle.sin_cos();
    let from_canonical = |canonical: (f64, f64)| {
        let local_x = canonical.0 * outer_major;
        let local_y = canonical.1 * outer_minor;
        (
            coarse.center.0 + local_x * axis_cos - local_y * axis_sin,
            coarse.center.1 + local_x * axis_sin + local_y * axis_cos,
        )
    };
    let mut evaluated = 0usize;
    let mut best = None::<PupilCenterOrbitalFit>;
    macro_rules! consider {
        ($center:expr) => {{
            evaluated += 1;
            if let Some(candidate) = fixed_radius_pupil_center_evidence(
                &luma,
                &native,
                &pupil_view,
                width,
                height,
                coarse,
                $center,
                equivalent_radius,
                temporal_prior,
            ) {
                if best.is_none_or(|incumbent| candidate.score > incumbent.score) {
                    best = Some(candidate);
                }
            }
        }};
    }

    consider!(seed);
    consider!(coarse.center);
    if let Some((prior, _)) = temporal_prior {
        consider!(prior);
    }
    for grid_y in -3i32..=3 {
        for grid_x in -3i32..=3 {
            let canonical = (
                grid_x as f64 * maximum_center_radius / 3.0,
                grid_y as f64 * maximum_center_radius / 3.0,
            );
            if canonical.0.hypot(canonical.1) <= maximum_center_radius + 1.0e-9 {
                consider!(from_canonical(canonical));
            }
        }
    }
    let mut center = best?.center;
    for step in [10.0, 5.0, 2.5, 1.25] {
        let incumbent = best?;
        for direction in 0..8 {
            let phase = 2.0 * PI * direction as f64 / 8.0;
            consider!((center.0 + step * phase.cos(), center.1 + step * phase.sin()));
        }
        let next = best?;
        if next.score > incumbent.score + 1.0e-9 {
            center = next.center;
        }
    }
    let mut best = best?;
    best.evaluated_centers = evaluated;
    Some(best)
}

/// Return the evidence attached to the radius which survived cyclic
/// regularization.  A different, stronger transition elsewhere on the same
/// ray may have been a lid, glint, or limbus edge; borrowing its score would
/// make the fitted pupil look better supported than the boundary we actually
/// selected.
fn selected_inner_margin_score(radii: &[f64], cues: &[f64], selected_radius: f64) -> f64 {
    radii
        .iter()
        .zip(cues)
        .filter(|(_, score)| score.is_finite())
        .min_by(|(left, _), (right, _)| {
            (*left - selected_radius)
                .abs()
                .total_cmp(&(*right - selected_radius).abs())
        })
        .map(|(_, score)| *score)
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug, Default)]
struct InnerMarginRawCues {
    raw_score: f64,
    luma_transition: f64,
    chroma_transition: f64,
    void_drop: f64,
    inside_void: f64,
    broad_dark_step: f64,
}

fn inner_margin_peak_prominence(samples: &[Option<InnerMarginRawCues>], index: usize) -> f64 {
    let Some(center) = samples.get(index).copied().flatten() else {
        return 0.0;
    };
    // The integrated margin cue is intentionally several pixels wide. Use a
    // four-pixel shoulder on each side rather than comparing adjacent
    // half-pixel samples, which would label the whole plateau non-prominent.
    const SHOULDER_STEPS: usize = 8;
    let left = index
        .checked_sub(1)
        .and_then(|end| {
            let start = index.saturating_sub(SHOULDER_STEPS);
            samples.get(start..=end)
        })
        .and_then(|values| {
            values
                .iter()
                .flatten()
                .map(|sample| sample.raw_score)
                .filter(|score| score.is_finite())
                .min_by(f64::total_cmp)
        });
    let right = samples
        .get(
            index.saturating_add(1)..=(index + SHOULDER_STEPS).min(samples.len().saturating_sub(1)),
        )
        .and_then(|values| {
            values
                .iter()
                .flatten()
                .map(|sample| sample.raw_score)
                .filter(|score| score.is_finite())
                .min_by(f64::total_cmp)
        });
    match (left, right) {
        (Some(left), Some(right)) => (center.raw_score - left.max(right)).max(0.0),
        // A maximum at the hard search edge cannot establish its own radial
        // location; retain it for audit but give it no veto prominence.
        _ => 0.0,
    }
}

fn sparse_inner_margin_candidates(
    center: (f64, f64),
    coarse: &BorderFocus,
    radii: &[f64],
    sampled: &[Vec<Option<InnerMarginRawCues>>],
) -> Vec<InnerIrisRadialCandidate> {
    const LOCAL_MAXIMUM_HALF_WINDOW_STEPS: usize = 4;
    const MINIMUM_RETAINED_RAW_SCORE: f64 = 0.12;
    const MINIMUM_SEPARATION_PX: f64 = 3.0;
    const MAXIMUM_PER_SECTOR: usize = 3;
    let mut retained = Vec::with_capacity(sampled.len() * MAXIMUM_PER_SECTOR);
    for (sector_index, ray) in sampled.iter().enumerate() {
        let angle = 2.0 * PI * sector_index as f64 / sampled.len().max(1) as f64;
        let affine_ray = projected_affine_pupil_ray(coarse, angle);
        let mut local_maxima = ray
            .iter()
            .enumerate()
            .filter_map(|(radius_index, sample)| {
                let sample = (*sample)?;
                if !sample.raw_score.is_finite() || sample.raw_score < MINIMUM_RETAINED_RAW_SCORE {
                    return None;
                }
                let start = radius_index.saturating_sub(LOCAL_MAXIMUM_HALF_WINDOW_STEPS);
                let end = (radius_index + LOCAL_MAXIMUM_HALF_WINDOW_STEPS)
                    .min(ray.len().saturating_sub(1));
                if ray[start..=end].iter().flatten().any(|neighbor| {
                    neighbor.raw_score.is_finite()
                        && neighbor.raw_score > sample.raw_score + 1.0e-12
                }) {
                    return None;
                }
                let equivalent_radius_px = radii[radius_index];
                let image_radius = equivalent_radius_px * affine_ray.image_radius_scale;
                Some(InnerIrisRadialCandidate {
                    sector_index: sector_index as u8,
                    angle,
                    equivalent_radius_px,
                    x: center.0 + image_radius * angle.cos(),
                    y: center.1 + image_radius * angle.sin(),
                    raw_score: sample.raw_score,
                    peak_prominence: inner_margin_peak_prominence(ray, radius_index),
                    luma_transition: sample.luma_transition,
                    chroma_transition: sample.chroma_transition,
                    void_drop: sample.void_drop,
                    inside_void: sample.inside_void,
                    broad_dark_step: sample.broad_dark_step,
                })
            })
            .collect::<Vec<_>>();
        local_maxima.sort_by(|left, right| right.raw_score.total_cmp(&left.raw_score));
        let mut sector_retained =
            Vec::<InnerIrisRadialCandidate>::with_capacity(MAXIMUM_PER_SECTOR);
        for candidate in local_maxima {
            if sector_retained.iter().any(|existing| {
                (existing.equivalent_radius_px - candidate.equivalent_radius_px).abs()
                    < MINIMUM_SEPARATION_PX
            }) {
                continue;
            }
            sector_retained.push(candidate);
            if sector_retained.len() == MAXIMUM_PER_SECTOR {
                break;
            }
        }
        retained.extend(sector_retained);
    }
    retained
}

fn detect_inner_iris_boundary_with_center(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    center_override: Option<(f64, f64)>,
    radius_envelope: Option<InnerIrisRadiusEnvelope>,
    radius_prior: Option<InnerIrisRadiusPrior>,
    evidence_condition: InnerIrisEvidenceCondition,
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center_tuned(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        center_override,
        radius_envelope,
        radius_prior,
        evidence_condition,
        1.0,
        0.0,
    )
}

#[allow(clippy::too_many_arguments)]
fn detect_inner_iris_boundary_with_center_tuned(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    center_override: Option<(f64, f64)>,
    radius_envelope: Option<InnerIrisRadiusEnvelope>,
    radius_prior: Option<InnerIrisRadiusPrior>,
    evidence_condition: InnerIrisEvidenceCondition,
    mass_prior_scale: f64,
    normalized_radius_penalty: f64,
) -> InnerIrisBoundary {
    if raw.len() < width * height
        || coarse.radius < 20.0
        || !coarse.center.0.is_finite()
        || !coarse.center.1.is_finite()
    {
        return InnerIrisBoundary::default();
    }
    let Some(native) = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse) else {
        return InnerIrisBoundary::default();
    };
    let luma = BoxLuma5::new(raw, width, height);
    let Some(full_resolution_void) =
        BorrowedFullResolutionPupilView::new(raw, width, height, &luma, &native)
    else {
        return InnerIrisBoundary::default();
    };
    let center = match center_override {
        Some(center) if center.0.is_finite() && center.1.is_finite() => center,
        Some(_) => return InnerIrisBoundary::default(),
        None => {
            let Some(center) =
                locate_pupil_center_full_resolution_with_limit(&full_resolution_void, coarse, 0.68)
                    .map(|diagnostics| diagnostics.center)
            else {
                return InnerIrisBoundary::default();
            };
            center
        }
    };
    let Some(mass_radius) =
        full_resolution_pupil_mass_radius(&full_resolution_void, center, coarse)
    else {
        return InnerIrisBoundary::default();
    };
    // The dark-mass estimate now counts every native RAW coordinate. Keep the
    // explicit blend so a later temporal method can replace this frame-local
    // estimate without changing the boundary contract.
    let baseline_radius = mass_radius.clamp(coarse.radius * 0.18, coarse.radius * 0.55);
    let estimated_radius = (0.58 * baseline_radius + 0.42 * mass_radius)
        .clamp(coarse.radius * 0.16, coarse.radius * 0.60);
    let fallback_minimum_radius = (coarse.radius * 0.14).max(8.0);
    let fallback_maximum_radius = coarse.radius * 0.65;
    let minimum_radius = radius_envelope.map_or(fallback_minimum_radius, |envelope| {
        envelope.minimum_equivalent_radius_px.max(4.0)
    });
    // The old fixed 40-pixel ceiling silently clipped a resolved pupil when
    // the eye occupied more of the RAW ROI. The envelope is expressed in
    // projected area-equivalent space; each ray below maps that value through
    // the limbus affine projection before sampling the native RAW plane.
    let maximum_radius = radius_envelope.map_or(fallback_maximum_radius, |envelope| {
        envelope.maximum_equivalent_radius_px
    });
    if maximum_radius <= minimum_radius + 2.0 {
        return InnerIrisBoundary::default();
    }
    let radius_steps = ((maximum_radius - minimum_radius) * 2.0).floor() as usize + 1;
    let radii = (0..radius_steps)
        .map(|index| minimum_radius + index as f64 * 0.5)
        .collect::<Vec<_>>();
    let (luma_weight, chroma_weight, void_weight, mass_prior_weight, temporal_prior_weight) =
        evidence_condition.cue_weights();
    let mass_prior_scale = if mass_prior_scale.is_finite() {
        mass_prior_scale.clamp(0.0, 2.0)
    } else {
        1.0
    };
    let normalized_radius_penalty = if normalized_radius_penalty.is_finite() {
        normalized_radius_penalty.clamp(0.0, 0.50)
    } else {
        0.0
    };
    // `selection_cues` may include frame-local mass and temporal priors to
    // rank ambiguous radii. `observed_cues` is deliberately current-frame RAW
    // evidence only; it is what downstream confidence and posterior admission
    // are allowed to consume.
    let mut selection_cues = Vec::with_capacity(21);
    let mut observed_cues = Vec::with_capacity(21);
    let mut raw_candidate_cues = Vec::with_capacity(21);
    for index in 0..21 {
        let angle = 2.0 * PI * index as f64 / 21.0;
        let (direction_y, direction_x) = angle.sin_cos();
        let affine_ray = projected_affine_pupil_ray(coarse, angle);
        let mut selection_ray = Vec::with_capacity(radii.len());
        let mut observed_ray = Vec::with_capacity(radii.len());
        let mut raw_candidate_ray = Vec::with_capacity(radii.len());
        for &equivalent_radius in &radii {
            let image_radius = equivalent_radius * affine_ray.image_radius_scale;
            let x = center.0 + image_radius * direction_x;
            let y = center.1 + image_radius * direction_y;
            if !inner_margin_candidate_is_observable(x, y, width, height) {
                selection_ray.push(f64::NEG_INFINITY);
                observed_ray.push(f64::NEG_INFINITY);
                raw_candidate_ray.push(None);
                continue;
            }
            let (luma_transition, chroma_transition, void_drop) = inner_margin_transition_cues(
                &luma,
                &native,
                Some(&full_resolution_void),
                x,
                y,
                affine_ray.outward_normal.0,
                affine_ray.outward_normal.1,
            );
            let inside_void = full_resolution_void.sample_void(
                x - affine_ray.outward_normal.0 * 3.0,
                y - affine_ray.outward_normal.1 * 3.0,
            );
            let native_raw_gate = 0.28 + 0.72 * inside_void;
            let baseline_reward = mass_prior_scale
                * mass_prior_weight
                * (-0.5 * ((equivalent_radius - estimated_radius) / 5.0).powi(2)).exp();
            // History may break a tie between similarly plausible margins,
            // but it must not narrow the current-frame search or overpower a
            // strong RAW transition.  In particular, a real pupil dilation
            // is allowed to leave the preferred interval immediately.
            let temporal_reward = radius_prior.map_or(0.0, |prior| {
                let prior_sigma = (prior.estimated_equivalent_radius_px
                    - prior.preferred_minimum_equivalent_radius_px)
                    .abs()
                    .max(
                        (prior.preferred_maximum_equivalent_radius_px
                            - prior.estimated_equivalent_radius_px)
                            .abs(),
                    )
                    .max(
                        0.5 * (prior.preferred_maximum_equivalent_radius_px
                            - prior.preferred_minimum_equivalent_radius_px),
                    )
                    .max(2.0);
                temporal_prior_weight
                    * prior.confidence
                    * (-0.5
                        * ((equivalent_radius - prior.estimated_equivalent_radius_px)
                            / prior_sigma)
                            .powi(2))
                    .exp()
            });
            let observed_score = (luma_weight * luma_transition
                + chroma_weight * chroma_transition
                + void_weight * void_drop)
                * native_raw_gate;
            let broad_inside_x = center.0 + image_radius * 0.55 * direction_x;
            let broad_inside_y = center.1 + image_radius * 0.55 * direction_y;
            let broad_outside_x = x + affine_ray.outward_normal.0 * 6.0;
            let broad_outside_y = y + affine_ray.outward_normal.1 * 6.0;
            let broad_dark_step = if inner_margin_candidate_is_observable(
                broad_inside_x,
                broad_inside_y,
                width,
                height,
            ) && inner_margin_candidate_is_observable(
                broad_outside_x,
                broad_outside_y,
                width,
                height,
            ) {
                let inside_luma = luma.sample(broad_inside_x, broad_inside_y);
                let outside_luma = luma.sample(broad_outside_x, broad_outside_y);
                let exposure_scale = (0.5 * (inside_luma.abs() + outside_luma.abs())).max(8.0);
                (outside_luma - inside_luma) / exposure_scale
            } else {
                f64::NAN
            };
            let radius_penalty =
                normalized_radius_penalty * equivalent_radius / coarse.radius.max(1.0);
            selection_ray.push(observed_score + baseline_reward + temporal_reward - radius_penalty);
            observed_ray.push(observed_score);
            raw_candidate_ray.push(Some(InnerMarginRawCues {
                raw_score: observed_score,
                luma_transition,
                chroma_transition,
                void_drop,
                inside_void,
                broad_dark_step,
            }));
        }
        selection_cues.push(selection_ray);
        observed_cues.push(observed_ray);
        raw_candidate_cues.push(raw_candidate_ray);
    }
    // Retain a small, prior-free local-maximum bank before any per-ray winner
    // or cyclic neighbor regularization is applied.  The temporal polar
    // co-solver consumes this sparse evidence; the legacy boundary below is
    // left intact during the diagnostic phase.
    let radial_candidates =
        sparse_inner_margin_candidates(center, coarse, &radii, &raw_candidate_cues);
    let mut selected = selection_cues
        .iter()
        .map(|ray| {
            ray.iter()
                .enumerate()
                .filter(|(_, score)| score.is_finite())
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| radii[index])
        })
        .collect::<Vec<_>>();
    for _ in 0..2 {
        let previous = selected.clone();
        for index in 0..21 {
            let mut neighborhood = [
                previous[(index + 20) % 21],
                previous[index],
                previous[(index + 1) % 21],
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
            if neighborhood.is_empty() {
                continue;
            }
            let neighbor = median(&mut neighborhood);
            selected[index] = radii
                .iter()
                .zip(&selection_cues[index])
                .filter(|(_, score)| score.is_finite())
                .max_by(|left, right| {
                    (left.1 - 0.020 * (left.0 - neighbor).abs())
                        .total_cmp(&(right.1 - 0.020 * (right.0 - neighbor).abs()))
                })
                .map(|(radius, _)| *radius);
        }
    }
    let mut radius_population = selected.iter().flatten().copied().collect::<Vec<_>>();
    if radius_population.len() < 8 {
        return InnerIrisBoundary::default();
    }
    let radius = median(&mut radius_population);
    let points = selected
        .iter()
        .enumerate()
        .filter_map(|(index, radius)| {
            let equivalent_radius = (*radius)?;
            let angle = 2.0 * PI * index as f64 / 21.0;
            let affine_ray = projected_affine_pupil_ray(coarse, angle);
            let image_radius = equivalent_radius * affine_ray.image_radius_scale;
            Some(InnerIrisPoint {
                x: center.0 + image_radius * angle.cos(),
                y: center.1 + image_radius * angle.sin(),
                score: selected_inner_margin_score(
                    &radii,
                    &observed_cues[index],
                    equivalent_radius,
                ),
            })
        })
        .collect::<Vec<_>>();
    // The 21 samples are independent radial edge measurements around the
    // acquired pupil center. Their selected radii are deliberately allowed to
    // differ when a lid, glint, or weak iris sector makes one ray ambiguous.
    // Consequently their arithmetic centroid is *not* an ellipse-center
    // estimate: one coherent lower-lid run can move it by many pixels even
    // though the ray origin was correct. Keep the independently acquired
    // center as the center degree of freedom for both complete and censored
    // rings. The samples are also uniform in image polar angle rather than
    // ellipse parameter angle, so their covariance cannot recover the axes of
    // a projected circle either. Preserve the selected limbus projection and
    // use the robust area-equivalent radius as the single pupil-size degree of
    // freedom; this is the fixed-center affine-circle model the search itself
    // enforced.
    let (major_to_minor, angle) = normalized_affine_projection(coarse);
    let ratio_root = major_to_minor.sqrt();
    let major_radius = radius * ratio_root;
    let minor_radius = radius / ratio_root;
    InnerIrisBoundary {
        center,
        radius,
        major_radius,
        minor_radius,
        angle,
        points,
        radial_candidates,
    }
}

pub fn detect_inner_iris_boundary(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        None,
        None,
        None,
        InnerIrisEvidenceCondition::default(),
    )
}

/// Run the production pupil-margin solver with an optional temporal size
/// preference. The prior is deliberately a ranking hint rather than an
/// admissibility bound; strong current RAW evidence remains authoritative.
pub fn detect_inner_iris_boundary_with_radius_prior(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    radius_prior: Option<InnerIrisRadiusPrior>,
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        None,
        None,
        radius_prior,
        InnerIrisEvidenceCondition::default(),
    )
}

/// Production pupil-margin solver with a hard current-frame radius envelope,
/// a separate soft temporal-size preference, and focus/apparent-scale cue
/// conditioning. All underlying samples remain native RAW; the condition
/// changes only the analog cue weights used to rank the bounded radii.
pub fn detect_inner_iris_boundary_conditioned(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    radius_envelope: Option<InnerIrisRadiusEnvelope>,
    radius_prior: Option<InnerIrisRadiusPrior>,
    evidence_condition: InnerIrisEvidenceCondition,
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        None,
        radius_envelope,
        radius_prior,
        evidence_condition,
    )
}

/// Production pupil-margin solver around a rough center selected by an
/// independent acquisition mechanism.
///
/// The supplied center establishes the ray origin only: it contributes no
/// pupil radius or boundary pixels. The returned size and projected-circle
/// shape are still solved exclusively from the untouched full-resolution RAW
/// samples, under the finalized limbus projection carried by `coarse`.
pub fn detect_inner_iris_boundary_conditioned_at_center(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    rough_center: (f64, f64),
    radius_envelope: Option<InnerIrisRadiusEnvelope>,
    radius_prior: Option<InnerIrisRadiusPrior>,
    evidence_condition: InnerIrisEvidenceCondition,
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        Some(rough_center),
        radius_envelope,
        radius_prior,
        evidence_condition,
    )
}

/// Run the production 21-ray native-log pupil-margin solver around a supplied
/// center. Driving uses this only after broad dark-basin acquisition, so a
/// candidate must demonstrate a real closed inner boundary rather than merely
/// winning the darkness/enclosure map.
pub fn debug_inner_iris_boundary_at_center(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    center: (f64, f64),
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        Some(center),
        None,
        None,
        InnerIrisEvidenceCondition::default(),
    )
}

/// Offline-only selection probe for the same native-coordinate 21-ray pupil
/// solver. `mass_prior_scale=1` and `normalized_radius_penalty=0` reproduce
/// production exactly. Lower mass weight reveals whether a dark iris/shadow
/// population pulled the selected margin outward; the optional normalized
/// penalty is a deliberately weak tie-break toward the first cohesive inner
/// transition, not a hard pupil-size constraint.
pub fn debug_inner_iris_boundary_at_center_tuned(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    center: (f64, f64),
    mass_prior_scale: f64,
    normalized_radius_penalty: f64,
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center_tuned(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        Some(center),
        None,
        None,
        InnerIrisEvidenceCondition::default(),
        mass_prior_scale,
        normalized_radius_penalty,
    )
}

/// Offline/Driving selection probe with the same soft temporal radius prior
/// accepted by the production pupil solver.  Keeping the prior in projected
/// area-equivalent pixels lets callers form it from a dimensionless
/// pupil/limbus radius ratio without confusing a foreshortened image ellipse
/// with its fronto-parallel circle.
#[allow(clippy::too_many_arguments)]
pub fn debug_inner_iris_boundary_at_center_tuned_with_prior(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    center: (f64, f64),
    radius_envelope: Option<InnerIrisRadiusEnvelope>,
    radius_prior: Option<InnerIrisRadiusPrior>,
    mass_prior_scale: f64,
    normalized_radius_penalty: f64,
) -> InnerIrisBoundary {
    detect_inner_iris_boundary_with_center_tuned(
        raw,
        width,
        height,
        sensor_x,
        sensor_y,
        coarse,
        Some(center),
        radius_envelope,
        radius_prior,
        InnerIrisEvidenceCondition::default(),
        mass_prior_scale,
        normalized_radius_penalty,
    )
}

pub fn co_solve_concentric_iris_boundaries(
    outer: &mut OuterIrisBoundary,
    inner: &mut InnerIrisBoundary,
) -> bool {
    if outer.evidence_points.len() < 8
        || inner.points.len() < 8
        || outer.major_radius <= 1.0
        || outer.minor_radius <= 1.0
        || inner.radius <= 1.0
    {
        return false;
    }
    let initial_midpoint = (
        inner.center.0 * 0.65 + outer.center.0 * 0.35,
        inner.center.1 * 0.65 + outer.center.1 * 0.35,
    );
    let center_separation =
        (inner.center.0 - outer.center.0).hypot(inner.center.1 - outer.center.1);
    let robust_median = |mut values: Vec<f64>| {
        if values.is_empty() {
            f64::NAN
        } else {
            median(&mut values)
        }
    };
    let score_center = |center: (f64, f64)| {
        let inner_radii = inner
            .points
            .iter()
            .map(|point| (point.x - center.0).hypot(point.y - center.1))
            .collect::<Vec<_>>();
        let inner_radius = robust_median(inner_radii.clone());
        if !inner_radius.is_finite() || inner_radius <= 2.0 {
            return f64::INFINITY;
        }
        let inner_error = robust_median(
            inner_radii
                .iter()
                .map(|radius| (radius - inner_radius).abs())
                .collect(),
        );
        let (angle_sin, angle_cos) = outer.angle.sin_cos();
        let outer_rho = outer
            .evidence_points
            .iter()
            .filter_map(|point| {
                let dx = point.x - center.0;
                let dy = point.y - center.1;
                let distance = dx.hypot(dy);
                if distance < inner_radius * 1.18 {
                    return None;
                }
                let local_x = dx * angle_cos + dy * angle_sin;
                let local_y = -dx * angle_sin + dy * angle_cos;
                Some(
                    ((local_x / outer.major_radius).powi(2)
                        + (local_y / outer.minor_radius).powi(2))
                    .sqrt(),
                )
            })
            .collect::<Vec<_>>();
        if outer_rho.len() < 8 {
            return f64::INFINITY;
        }
        let rho = robust_median(outer_rho.clone());
        let outer_error =
            robust_median(outer_rho.iter().map(|value| (value - rho).abs()).collect())
                * (outer.major_radius * outer.minor_radius).sqrt();
        1.45 * inner_error
            + outer_error
            + 0.015 * (center.0 - initial_midpoint.0).hypot(center.1 - initial_midpoint.1)
    };

    // A small pattern-search walk co-solves the shared center without another
    // RAW radial scan. Each orbit tests the eight neighboring centers and
    // halves its step, giving sub-pixel refinement in a bounded number of
    // evaluations.
    let mut center = initial_midpoint;
    let mut best_score = score_center(center);
    let mut step = (center_separation * 0.35).clamp(2.0, 8.0);
    for _ in 0..6 {
        let mut best = center;
        for (dx, dy) in [
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ] {
            let candidate = (center.0 + dx * step, center.1 + dy * step);
            let score = score_center(candidate);
            if score < best_score {
                best_score = score;
                best = candidate;
            }
        }
        center = best;
        step *= 0.5;
    }
    if !best_score.is_finite() {
        return false;
    }

    let inner_radius = robust_median(
        inner
            .points
            .iter()
            .map(|point| (point.x - center.0).hypot(point.y - center.1))
            .collect(),
    );
    outer
        .evidence_points
        .retain(|point| (point.x - center.0).hypot(point.y - center.1) >= inner_radius * 1.18);
    if outer.evidence_points.len() < 8 {
        return false;
    }
    let (angle_sin, angle_cos) = outer.angle.sin_cos();
    let outer_scale = robust_median(
        outer
            .evidence_points
            .iter()
            .map(|point| {
                let dx = point.x - center.0;
                let dy = point.y - center.1;
                let local_x = dx * angle_cos + dy * angle_sin;
                let local_y = -dx * angle_sin + dy * angle_cos;
                ((local_x / outer.major_radius).powi(2) + (local_y / outer.minor_radius).powi(2))
                    .sqrt()
            })
            .collect(),
    );
    let minimum_outer_radius = inner_radius * 1.22;
    let scaled_equivalent_radius = (outer.major_radius * outer.minor_radius).sqrt() * outer_scale;
    let ordering_scale = (minimum_outer_radius / scaled_equivalent_radius.max(1.0)).max(1.0);
    outer.center = center;
    outer.major_radius *= outer_scale * ordering_scale;
    outer.minor_radius *= outer_scale * ordering_scale;
    let mut contrasts = outer
        .evidence_points
        .iter()
        .map(|point| point.contrast)
        .collect::<Vec<_>>();
    let contrast = median(&mut contrasts).max(1.0);
    outer.points = stable_screen_space_ellipse_points(
        [
            center.0,
            center.1,
            outer.major_radius,
            outer.minor_radius,
            outer.angle,
        ],
        64,
        contrast,
    );

    let inner_score = inner
        .points
        .iter()
        .map(|point| point.score)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    inner.center = center;
    inner.radius = inner_radius;
    inner.major_radius = inner_radius;
    inner.minor_radius = inner_radius;
    inner.angle = outer.angle;
    inner.points = (0..21)
        .map(|index| {
            let angle = 2.0 * PI * index as f64 / 21.0;
            InnerIrisPoint {
                x: center.0 + inner_radius * angle.cos(),
                y: center.1 + inner_radius * angle.sin(),
                score: inner_score,
            }
        })
        .collect();
    true
}

/// One deterministic member of the offline joint-evidence tournament.
///
/// `outer.evidence_points` and `inner_evidence` contain only measurements that
/// survived the variant's ambiguity and cohort tests.  `outer.points` and
/// `inner.points` are complete equation-generated rings, so diagnostics never
/// imply that unmeasured portions of either boundary were directly observed.
#[derive(Clone, Debug)]
pub struct JointIrisEvidenceVariant {
    pub name: String,
    pub valid: bool,
    pub objective: f64,
    pub outer: OuterIrisBoundary,
    pub inner: InnerIrisBoundary,
    pub inner_evidence: Vec<InnerIrisPoint>,
    pub outer_candidate_count: usize,
    pub inner_candidate_count: usize,
}

fn trimmed_mean(mut values: Vec<f64>, keep_fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    values.sort_by(f64::total_cmp);
    let count = ((values.len() as f64 * keep_fraction).ceil() as usize).clamp(1, values.len());
    values[..count].iter().sum::<f64>() / count as f64
}

fn joint_evidence_center_score(
    center: (f64, f64),
    outer: &[OuterIrisPoint],
    inner: &[InnerIrisPoint],
    ellipse: [f64; 5],
    keep_fraction: f64,
    squared_loss: bool,
    prior: (f64, f64),
) -> f64 {
    if outer.len() < 5 || inner.len() < 5 || ellipse[2] <= 1.0 || ellipse[3] <= 1.0 {
        return f64::INFINITY;
    }
    let inner_radii = inner
        .iter()
        .map(|point| (point.x - center.0).hypot(point.y - center.1))
        .collect::<Vec<_>>();
    let mut inner_population = inner_radii.clone();
    let inner_radius = median(&mut inner_population);
    if !inner_radius.is_finite() || inner_radius <= 2.0 {
        return f64::INFINITY;
    }
    let loss = |residual: f64, scale: f64| {
        let normalized = residual / scale.max(1.0);
        if squared_loss {
            normalized * normalized
        } else {
            // A bounded redescending-like loss keeps one eyelid or pupil-void
            // alias from purchasing movement of the common center.
            let clipped = normalized.abs().min(2.5);
            clipped * (1.0 - 0.12 * clipped).max(0.45)
        }
    };
    let inner_error = trimmed_mean(
        inner_radii
            .into_iter()
            .map(|radius| loss(radius - inner_radius, inner_radius * 0.08))
            .collect(),
        keep_fraction,
    );
    let (rotation_sin, rotation_cos) = ellipse[4].sin_cos();
    let outer_rhos = outer
        .iter()
        .filter_map(|point| {
            let dx = point.x - center.0;
            let dy = point.y - center.1;
            let distance = dx.hypot(dy);
            (distance >= inner_radius * 1.28).then(|| {
                let local_x = dx * rotation_cos + dy * rotation_sin;
                let local_y = -dx * rotation_sin + dy * rotation_cos;
                ((local_x / ellipse[2]).powi(2) + (local_y / ellipse[3]).powi(2)).sqrt()
            })
        })
        .collect::<Vec<_>>();
    if outer_rhos.len() < 5 {
        return f64::INFINITY;
    }
    let mut rho_population = outer_rhos.clone();
    let rho = median(&mut rho_population);
    let outer_error = trimmed_mean(
        outer_rhos
            .into_iter()
            .map(|value| loss(value - rho, 0.065))
            .collect(),
        keep_fraction,
    );
    let equivalent_outer = (ellipse[2] * ellipse[3]).sqrt() * rho;
    let ordering_penalty =
        ((1.36 * inner_radius - equivalent_outer).max(0.0) / inner_radius.max(1.0)).powi(2) * 18.0;
    inner_error * 1.25
        + outer_error
        + ordering_penalty
        + 0.0008 * (center.0 - prior.0).hypot(center.1 - prior.1)
}

fn walk_joint_evidence_center(
    initial: (f64, f64),
    outer: &[OuterIrisPoint],
    inner: &[InnerIrisPoint],
    ellipse: [f64; 5],
    keep_fraction: f64,
    squared_loss: bool,
    seed: u64,
) -> ((f64, f64), f64) {
    let score = |center| {
        joint_evidence_center_score(
            center,
            outer,
            inner,
            ellipse,
            keep_fraction,
            squared_loss,
            initial,
        )
    };
    let mut state = seed | 1;
    let mut random_unit = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as u32 as f64 / u32::MAX as f64) * 2.0 - 1.0
    };
    let mut global = initial;
    let mut global_score = score(global);
    // Multiple short deterministic walks are cheaper and much less brittle
    // than one long descent from a pupil-void-biased seed.
    for restart in 0..12 {
        let jitter = if restart == 0 { 0.0 } else { 10.0 };
        let mut center = (
            initial.0 + random_unit() * jitter,
            initial.1 + random_unit() * jitter,
        );
        let mut best = score(center);
        let mut step = 7.0;
        for _ in 0..8 {
            let mut next = center;
            for (dx, dy) in [
                (-1.0, -1.0),
                (0.0, -1.0),
                (1.0, -1.0),
                (-1.0, 0.0),
                (1.0, 0.0),
                (-1.0, 1.0),
                (0.0, 1.0),
                (1.0, 1.0),
            ] {
                let candidate = (center.0 + dx * step, center.1 + dy * step);
                let candidate_score = score(candidate);
                if candidate_score < best {
                    best = candidate_score;
                    next = candidate;
                }
            }
            center = next;
            step *= 0.52;
        }
        if best < global_score {
            global = center;
            global_score = best;
        }
    }
    (global, global_score)
}

fn circular_cohort_mask(mask: &[bool]) -> Vec<bool> {
    if mask.len() < 5 {
        return vec![false; mask.len()];
    }
    (0..mask.len())
        .map(|index| {
            if !mask[index] {
                return false;
            }
            [1usize, 2]
                .into_iter()
                .flat_map(|distance| {
                    [
                        (index + distance) % mask.len(),
                        (index + mask.len() - distance) % mask.len(),
                    ]
                })
                .filter(|neighbor| mask[*neighbor])
                .count()
                >= 2
        })
        .collect()
}

/// Produce forty deterministic policies for selecting a small, unambiguous
/// joint inner/outer evidence set.  This is intentionally offline-only: the
/// tournament can establish which policy survives hard gaze before a single
/// policy is moved into the live tracker.
pub fn joint_iris_evidence_variants(
    coarse: &BorderFocus,
    outer_input: &OuterIrisBoundary,
    inner_input: &InnerIrisBoundary,
) -> Vec<JointIrisEvidenceVariant> {
    let outer_candidate_count = outer_input.evidence_points.len();
    let inner_candidate_count = inner_input.points.len();
    let center_blends: [f64; 5] = [0.15, 0.325, 0.50, 0.675, 0.85];
    let keep_fractions: [f64; 4] = [0.38, 0.52, 0.66, 0.80];
    let mut variants = Vec::with_capacity(40);
    for (blend_index, blend) in center_blends.into_iter().enumerate() {
        for (keep_index, keep_fraction) in keep_fractions.into_iter().enumerate() {
            for squared_loss in [false, true] {
                let name = format!(
                    "{}-b{:02}-k{:02}",
                    if squared_loss { "huber2" } else { "bounded" },
                    (blend * 100.0).round() as usize,
                    (keep_fraction * 100.0).round() as usize,
                );
                let initial = (
                    outer_input.center.0 * (1.0 - blend) + inner_input.center.0 * blend,
                    outer_input.center.1 * (1.0 - blend) + inner_input.center.1 * blend,
                );
                let mut ellipse = [
                    outer_input.center.0,
                    outer_input.center.1,
                    outer_input.major_radius,
                    outer_input.minor_radius,
                    outer_input.angle,
                ];
                let mut selected_outer = outer_input.evidence_points.clone();
                let mut selected_inner = inner_input.points.clone();
                let mut center = initial;
                let mut objective = f64::INFINITY;
                for iteration in 0..4 {
                    (center, objective) = walk_joint_evidence_center(
                        center,
                        &selected_outer,
                        &selected_inner,
                        ellipse,
                        keep_fraction,
                        squared_loss,
                        ((blend_index * 8 + keep_index * 2 + squared_loss as usize) as u64 + 1)
                            * 0x9e3779b97f4a7c15u64
                            + iteration as u64,
                    );
                    let mut inner_radii = inner_input
                        .points
                        .iter()
                        .map(|point| (point.x - center.0).hypot(point.y - center.1))
                        .collect::<Vec<_>>();
                    let inner_radius = median(&mut inner_radii);
                    let mut inner_residuals = inner_input
                        .points
                        .iter()
                        .map(|point| {
                            ((point.x - center.0).hypot(point.y - center.1) - inner_radius).abs()
                        })
                        .collect::<Vec<_>>();
                    let inner_limit = percentile_f64(&mut inner_residuals, keep_fraction)
                        .max(inner_radius * 0.025)
                        .min(inner_radius * 0.22);
                    let mut inner_scores = inner_input
                        .points
                        .iter()
                        .map(|point| point.score)
                        .collect::<Vec<_>>();
                    let inner_score_floor =
                        percentile_f64(&mut inner_scores, (1.0 - keep_fraction) * 0.55);
                    let inner_pre_mask = inner_input
                        .points
                        .iter()
                        .map(|point| {
                            ((point.x - center.0).hypot(point.y - center.1) - inner_radius).abs()
                                <= inner_limit
                                && point.score >= inner_score_floor
                        })
                        .collect::<Vec<_>>();
                    let inner_mask = circular_cohort_mask(&inner_pre_mask);
                    selected_inner = inner_input
                        .points
                        .iter()
                        .zip(inner_mask)
                        .filter_map(|(point, retain)| retain.then_some(*point))
                        .collect();

                    let (rotation_sin, rotation_cos) = ellipse[4].sin_cos();
                    let outer_residual = |point: &OuterIrisPoint| {
                        let dx = point.x - center.0;
                        let dy = point.y - center.1;
                        let local_x = dx * rotation_cos + dy * rotation_sin;
                        let local_y = -dx * rotation_sin + dy * rotation_cos;
                        (((local_x / ellipse[2]).powi(2) + (local_y / ellipse[3]).powi(2)).sqrt()
                            - 1.0)
                            .abs()
                    };
                    let mut outer_residuals = outer_input
                        .evidence_points
                        .iter()
                        .map(outer_residual)
                        .collect::<Vec<_>>();
                    let outer_limit = percentile_f64(&mut outer_residuals, keep_fraction)
                        .max(0.018)
                        .min(0.18);
                    let mut contrasts = outer_input
                        .evidence_points
                        .iter()
                        .map(|point| point.contrast)
                        .collect::<Vec<_>>();
                    let contrast_floor =
                        percentile_f64(&mut contrasts, (1.0 - keep_fraction) * 0.65);
                    let outer_pre_mask = outer_input
                        .evidence_points
                        .iter()
                        .map(|point| {
                            outer_residual(point) <= outer_limit
                                && point.contrast >= contrast_floor
                                && (point.x - center.0).hypot(point.y - center.1)
                                    >= inner_radius * 1.30
                        })
                        .collect::<Vec<_>>();
                    let outer_mask = circular_cohort_mask(&outer_pre_mask);
                    selected_outer = outer_input
                        .evidence_points
                        .iter()
                        .zip(outer_mask)
                        .filter_map(|(point, retain)| retain.then_some(*point))
                        .collect();
                    if selected_outer.len() >= 8 {
                        let fitted = fit_outer_ellipse(
                            &selected_outer,
                            [
                                center.0,
                                center.1,
                                coarse.radius.max(outer_input.minor_radius),
                            ],
                        );
                        ellipse = [center.0, center.1, fitted[2], fitted[3], fitted[4]];
                    }
                    if selected_outer.len() < 8 || selected_inner.len() < 6 {
                        break;
                    }
                }
                let valid =
                    selected_outer.len() >= 8 && selected_inner.len() >= 6 && objective.is_finite();
                let mut outer = outer_input.clone();
                let mut inner = inner_input.clone();
                if valid {
                    let mut inner_radii = selected_inner
                        .iter()
                        .map(|point| (point.x - center.0).hypot(point.y - center.1))
                        .collect::<Vec<_>>();
                    let inner_radius = median(&mut inner_radii);
                    let fitted = fit_outer_ellipse(
                        &selected_outer,
                        [
                            center.0,
                            center.1,
                            coarse.radius.max(outer_input.minor_radius),
                        ],
                    );
                    let minimum_equivalent = inner_radius * 1.36;
                    let equivalent = (fitted[2] * fitted[3]).sqrt();
                    let scale = (minimum_equivalent / equivalent.max(1.0)).max(1.0);
                    outer.center = center;
                    outer.major_radius = fitted[2] * scale;
                    outer.minor_radius = fitted[3] * scale;
                    outer.angle = fitted[4];
                    outer.evidence_points = selected_outer.clone();
                    let contrast = selected_outer
                        .iter()
                        .map(|point| point.contrast)
                        .sum::<f64>()
                        / selected_outer.len() as f64;
                    outer.points = stable_screen_space_ellipse_points(
                        [
                            center.0,
                            center.1,
                            outer.major_radius,
                            outer.minor_radius,
                            outer.angle,
                        ],
                        64,
                        contrast.max(1.0),
                    );
                    inner.center = center;
                    inner.radius = inner_radius;
                    inner.major_radius = inner_radius;
                    inner.minor_radius = inner_radius;
                    inner.angle = outer.angle;
                    let score = selected_inner.iter().map(|point| point.score).sum::<f64>()
                        / selected_inner.len() as f64;
                    inner.points = (0..64)
                        .map(|index| {
                            let angle = 2.0 * PI * index as f64 / 64.0;
                            InnerIrisPoint {
                                x: center.0 + inner_radius * angle.cos(),
                                y: center.1 + inner_radius * angle.sin(),
                                score,
                            }
                        })
                        .collect();
                    objective += 0.22 / (selected_outer.len() as f64).sqrt()
                        + 0.22 / (selected_inner.len() as f64).sqrt();
                } else {
                    outer.evidence_points.clear();
                    outer.points.clear();
                    inner.points.clear();
                    objective = f64::INFINITY;
                }
                variants.push(JointIrisEvidenceVariant {
                    name,
                    valid,
                    objective,
                    outer,
                    inner,
                    inner_evidence: selected_inner,
                    outer_candidate_count,
                    inner_candidate_count,
                });
            }
        }
    }
    variants
}

#[derive(Clone, Copy)]
struct JointInnerCandidate {
    point: InnerIrisPoint,
    raw_score: f64,
}

fn joint_inner_candidate_rays(
    raw: &[u16],
    width: usize,
    height: usize,
    native: &NativeLogPlane,
    coarse: &BorderFocus,
    center_seeds: [(f64, f64); 3],
    expected_radius: f64,
) -> [Vec<JointInnerCandidate>; 41] {
    let luma = BoxLuma5::new(raw, width, height);
    let minimum_radius = (coarse.radius * 0.12).max(7.0);
    let maximum_radius = (coarse.radius * 0.68).min(44.0);
    std::array::from_fn(|ray_index| {
        let angle = 2.0 * PI * ray_index as f64 / 41.0;
        let (direction_y, direction_x) = angle.sin_cos();
        let mut sampled = Vec::new();
        for center in center_seeds {
            let steps = ((maximum_radius - minimum_radius) * 2.0).max(1.0) as usize;
            for radius_index in 0..=steps {
                let radius = minimum_radius + 0.5 * radius_index as f64;
                let x = center.0 + radius * direction_x;
                let y = center.1 + radius * direction_y;
                if x <= 6.0
                    || y <= 6.0
                    || x >= width.saturating_sub(7) as f64
                    || y >= height.saturating_sub(7) as f64
                {
                    continue;
                }
                let (luma_transition, chroma_transition, void_drop) = inner_margin_transition_cues(
                    &luma,
                    native,
                    None,
                    x,
                    y,
                    direction_x,
                    direction_y,
                );
                let inside_void = native.sample_void(x - direction_x * 3.0, y - direction_y * 3.0);
                let expected_reward =
                    0.10 * (-0.5 * ((radius - expected_radius) / 8.0).powi(2)).exp();
                let score = (0.58 * luma_transition + 0.22 * chroma_transition + 0.20 * void_drop)
                    * (0.22 + 0.78 * inside_void)
                    + expected_reward;
                sampled.push(JointInnerCandidate {
                    point: InnerIrisPoint { x, y, score },
                    raw_score: score,
                });
            }
        }
        sampled.sort_by(|left, right| right.raw_score.total_cmp(&left.raw_score));
        let mut retained = Vec::<JointInnerCandidate>::with_capacity(14);
        for candidate in sampled {
            if retained.iter().any(|existing| {
                (existing.point.x - candidate.point.x).hypot(existing.point.y - candidate.point.y)
                    < 1.75
            }) {
                continue;
            }
            retained.push(candidate);
            if retained.len() >= 14 {
                break;
            }
        }
        retained
    })
}

#[derive(Clone, Copy)]
struct JointWinner<T> {
    point: T,
    margin: f64,
    raw_score: f64,
}

fn anchor_joint_outer_fit(
    _fitted: [f64; 5],
    baseline: &OuterIrisBoundary,
    center: (f64, f64),
) -> [f64; 5] {
    [
        center.0,
        center.1,
        baseline.major_radius,
        baseline.minor_radius,
        baseline.angle,
    ]
}

fn joint_inner_has_bilateral_support(
    points: &[InnerIrisPoint],
    center: (f64, f64),
    radius: f64,
) -> bool {
    if points.len() < 8 || radius <= 2.0 {
        return false;
    }
    let mut sectors = [0usize; 8];
    let mut left = 0usize;
    let mut right = 0usize;
    let mut upper = 0usize;
    let mut lower = 0usize;
    for point in points {
        let dx = point.x - center.0;
        let dy = point.y - center.1;
        let angle = dy.atan2(dx).rem_euclid(2.0 * PI);
        sectors[((angle / (2.0 * PI) * 8.0).floor() as usize).min(7)] += 1;
        if dx <= -0.22 * radius {
            left += 1;
        }
        if dx >= 0.22 * radius {
            right += 1;
        }
        if dy <= -0.22 * radius {
            upper += 1;
        }
        if dy >= 0.22 * radius {
            lower += 1;
        }
    }
    let occupied = sectors.iter().filter(|count| **count > 0).count();
    let maximum_empty_run = (0..16)
        .fold((0usize, 0usize), |(current, maximum), index| {
            if sectors[index % 8] == 0 {
                (current + 1, maximum.max(current + 1))
            } else {
                (0, maximum)
            }
        })
        .1
        .min(8);
    occupied >= 6 && maximum_empty_run <= 2 && left >= 2 && right >= 2 && upper >= 2 && lower >= 2
}

/// Second-generation offline tournament. Unlike
/// [`joint_iris_evidence_variants`], this keeps multiple radial candidates per
/// ray until the joint inner/outer model has scored them. Thus a strong pupil
/// edge or eyelid hit cannot erase the weaker true limbus before the cohesive
/// geometry is considered.
#[allow(clippy::too_many_arguments)]
pub fn joint_iris_raw_evidence_variants(
    raw: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    coarse: &BorderFocus,
    upper_eyelid: &[BorderPoint],
    lower_eyelid: &[BorderPoint],
    outer_input: &OuterIrisBoundary,
    inner_input: &InnerIrisBoundary,
) -> Vec<JointIrisEvidenceVariant> {
    if raw.len() < width * height || outer_input.major_radius <= 1.0 || inner_input.radius <= 1.0 {
        return joint_iris_evidence_variants(coarse, outer_input, inner_input);
    }
    let rough_search = OuterSearchEllipse::from_coarse(coarse);
    let luma = Arc::new(BoxLuma5::new(raw, width, height));
    let native_unblurred = native_log_plane(raw, width, height, sensor_x, sensor_y, coarse);
    let native_outer = native_unblurred
        .clone()
        .map(blur_outer_appearance)
        .map(Arc::new);
    let appearance = native_outer
        .as_deref()
        .and_then(|native| estimate_iris_sclera_appearance(native, rough_search));
    let reflectance_appearance = native_outer
        .as_deref()
        .and_then(|native| estimate_iris_sclera_reflectance_appearance(native, rough_search));
    let sclera_probability = native_outer
        .as_deref()
        .zip(appearance)
        .map(|(native, appearance)| Arc::new(iris_sclera_probability_map(native, appearance)));
    let reflectance_sclera_probability =
        native_outer
            .as_deref()
            .zip(reflectance_appearance)
            .map(|(native, appearance)| {
                Arc::new(iris_sclera_reflectance_probability_map(native, appearance))
            });
    let outer_context = Arc::new(OuterRayContext {
        luma: Arc::clone(&luma),
        native: native_outer,
        sclera_probability,
        reflectance_sclera_probability,
        material_illumination: None,
        upper_eyelid: Arc::new(upper_eyelid.to_vec()),
        lower_eyelid: Arc::new(lower_eyelid.to_vec()),
        luma_gate: estimate_luma_transition_gate(luma.as_ref()),
        width,
        height,
        search: rough_search,
        rough_search,
        scale_range: (0.62, OUTER_IRIS_MAX_SEARCH_SCALE),
    });
    let outer_rays = evaluate_outer_iris_rays(outer_context, 1, 1).candidates;
    let Some(native_inner) = native_unblurred.as_ref() else {
        return joint_iris_evidence_variants(coarse, outer_input, inner_input);
    };
    let inner_rays = joint_inner_candidate_rays(
        raw,
        width,
        height,
        native_inner,
        coarse,
        [inner_input.center, coarse.center, outer_input.center],
        inner_input.radius,
    );
    let geometry_weights: [f64; 5] = [0.55, 0.90, 1.45, 2.35, 3.80];
    let keep_fractions: [f64; 4] = [0.34, 0.48, 0.62, 0.76];
    let outer_candidate_count = outer_rays.iter().map(Vec::len).sum();
    let inner_candidate_count = inner_rays.iter().map(Vec::len).sum();
    let mut variants = Vec::with_capacity(40);
    for (weight_index, geometry_weight) in geometry_weights.into_iter().enumerate() {
        for (keep_index, keep_fraction) in keep_fractions.into_iter().enumerate() {
            for squared_loss in [false, true] {
                let name = format!(
                    "lattice-{}-g{:03}-k{:02}",
                    if squared_loss { "h2" } else { "bounded" },
                    (geometry_weight * 100.0).round() as usize,
                    (keep_fraction * 100.0).round() as usize,
                );
                let mut center = (
                    inner_input.center.0 * 0.62 + outer_input.center.0 * 0.38,
                    inner_input.center.1 * 0.62 + outer_input.center.1 * 0.38,
                );
                let mut ellipse = [
                    center.0,
                    center.1,
                    outer_input.major_radius,
                    outer_input.minor_radius,
                    outer_input.angle,
                ];
                let mut inner_radius = inner_input.radius;
                let mut selected_outer = Vec::new();
                let mut selected_inner = Vec::new();
                let mut objective = f64::INFINITY;
                for iteration in 0..5 {
                    let (rotation_sin, rotation_cos) = ellipse[4].sin_cos();
                    let mut outer_winners = outer_rays
                        .iter()
                        .map(|ray| {
                            let mut ranked = ray
                                .iter()
                                .filter_map(|candidate| {
                                    let point = candidate.point;
                                    let distance = (point.x - center.0).hypot(point.y - center.1);
                                    if distance < inner_radius * 1.28 {
                                        return None;
                                    }
                                    let dx = point.x - center.0;
                                    let dy = point.y - center.1;
                                    let local_x = dx * rotation_cos + dy * rotation_sin;
                                    let local_y = -dx * rotation_sin + dy * rotation_cos;
                                    let residual = (((local_x / ellipse[2]).powi(2)
                                        + (local_y / ellipse[3]).powi(2))
                                    .sqrt()
                                        - 1.0)
                                        .abs();
                                    Some((
                                        candidate.score - geometry_weight * 12.0 * residual,
                                        candidate.score,
                                        point,
                                    ))
                                })
                                .collect::<Vec<_>>();
                            ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
                            ranked.first().map(|winner| JointWinner {
                                point: winner.2,
                                margin: winner.0
                                    - ranked
                                        .get(1)
                                        .map(|runner| runner.0)
                                        .unwrap_or(winner.0 - 8.0),
                                raw_score: winner.1,
                            })
                        })
                        .collect::<Vec<_>>();
                    let mut outer_margins = outer_winners
                        .iter()
                        .flatten()
                        .map(|winner| winner.margin)
                        .collect::<Vec<_>>();
                    let outer_margin_floor =
                        percentile_f64(&mut outer_margins, 1.0 - keep_fraction);
                    let mut outer_scores = outer_winners
                        .iter()
                        .flatten()
                        .map(|winner| winner.raw_score)
                        .collect::<Vec<_>>();
                    let outer_score_floor = percentile_f64(&mut outer_scores, 0.18);
                    let outer_pre_mask = outer_winners
                        .iter()
                        .map(|winner| {
                            winner.is_some_and(|winner| {
                                winner.margin >= outer_margin_floor
                                    && winner.raw_score >= outer_score_floor
                            })
                        })
                        .collect::<Vec<_>>();
                    let outer_mask = circular_cohort_mask(&outer_pre_mask);
                    selected_outer = outer_winners
                        .drain(..)
                        .zip(outer_mask)
                        .filter_map(|(winner, retain)| retain.then_some(winner?.point))
                        .collect();

                    let mut inner_winners = inner_rays
                        .iter()
                        .map(|ray| {
                            let mut ranked = ray
                                .iter()
                                .map(|candidate| {
                                    let radius = (candidate.point.x - center.0)
                                        .hypot(candidate.point.y - center.1);
                                    let residual =
                                        (radius - inner_radius).abs() / inner_radius.max(1.0);
                                    (
                                        candidate.raw_score - geometry_weight * 1.8 * residual,
                                        candidate.raw_score,
                                        candidate.point,
                                    )
                                })
                                .collect::<Vec<_>>();
                            ranked.sort_by(|left, right| right.0.total_cmp(&left.0));
                            ranked.first().map(|winner| JointWinner {
                                point: winner.2,
                                margin: winner.0
                                    - ranked
                                        .get(1)
                                        .map(|runner| runner.0)
                                        .unwrap_or(winner.0 - 1.0),
                                raw_score: winner.1,
                            })
                        })
                        .collect::<Vec<_>>();
                    let mut inner_margins = inner_winners
                        .iter()
                        .flatten()
                        .map(|winner| winner.margin)
                        .collect::<Vec<_>>();
                    let inner_margin_floor =
                        percentile_f64(&mut inner_margins, 1.0 - keep_fraction);
                    let mut inner_scores = inner_winners
                        .iter()
                        .flatten()
                        .map(|winner| winner.raw_score)
                        .collect::<Vec<_>>();
                    let inner_score_floor = percentile_f64(&mut inner_scores, 0.18);
                    let inner_pre_mask = inner_winners
                        .iter()
                        .map(|winner| {
                            winner.is_some_and(|winner| {
                                winner.margin >= inner_margin_floor
                                    && winner.raw_score >= inner_score_floor
                            })
                        })
                        .collect::<Vec<_>>();
                    let inner_mask = circular_cohort_mask(&inner_pre_mask);
                    selected_inner = inner_winners
                        .drain(..)
                        .zip(inner_mask)
                        .filter_map(|(winner, retain)| retain.then_some(winner?.point))
                        .collect();
                    if selected_outer.len() < 8 || selected_inner.len() < 6 {
                        break;
                    }
                    let fitted = anchor_joint_outer_fit(
                        fit_outer_ellipse(
                            &selected_outer,
                            [
                                center.0,
                                center.1,
                                coarse.radius.max(outer_input.minor_radius),
                            ],
                        ),
                        outer_input,
                        center,
                    );
                    ellipse = fitted;
                    (center, objective) = walk_joint_evidence_center(
                        center,
                        &selected_outer,
                        &selected_inner,
                        ellipse,
                        keep_fraction,
                        squared_loss,
                        ((weight_index * 8 + keep_index * 2 + squared_loss as usize + 1) as u64)
                            * 0xd1b54a32d192ed03u64
                            + iteration as u64,
                    );
                    let center_limit = (outer_input.minor_radius * 0.055).clamp(1.5, 3.0);
                    let center_dx = center.0 - outer_input.center.0;
                    let center_dy = center.1 - outer_input.center.1;
                    let center_distance = center_dx.hypot(center_dy);
                    if center_distance > center_limit {
                        let scale = center_limit / center_distance.max(1.0e-9);
                        center = (
                            outer_input.center.0 + center_dx * scale,
                            outer_input.center.1 + center_dy * scale,
                        );
                    }
                    let mut radii = selected_inner
                        .iter()
                        .map(|point| (point.x - center.0).hypot(point.y - center.1))
                        .collect::<Vec<_>>();
                    inner_radius = median(&mut radii);
                    ellipse[0] = center.0;
                    ellipse[1] = center.1;
                }
                // The last random walk follows the last ray selection. Remove
                // any pre-walk winner that no longer supports the displayed
                // model, then fit the complete equation from only that final
                // extreme-confidence cohort.
                if selected_outer.len() >= 8 && selected_inner.len() >= 6 {
                    let provisional = anchor_joint_outer_fit(
                        fit_outer_ellipse(
                            &selected_outer,
                            [
                                center.0,
                                center.1,
                                coarse.radius.max(outer_input.minor_radius),
                            ],
                        ),
                        outer_input,
                        center,
                    );
                    let (sin, cos) = provisional[4].sin_cos();
                    let residual = |point: &OuterIrisPoint| {
                        let dx = point.x - center.0;
                        let dy = point.y - center.1;
                        let local_x = dx * cos + dy * sin;
                        let local_y = -dx * sin + dy * cos;
                        (((local_x / provisional[2]).powi(2) + (local_y / provisional[3]).powi(2))
                            .sqrt()
                            - 1.0)
                            .abs()
                    };
                    let mut outer_final_residuals =
                        selected_outer.iter().map(residual).collect::<Vec<_>>();
                    let outer_final_limit =
                        (percentile_f64(&mut outer_final_residuals, 0.50) * 2.8 + 0.012)
                            .clamp(0.035, 0.115);
                    selected_outer.retain(|point| residual(point) <= outer_final_limit);

                    let mut inner_radii = selected_inner
                        .iter()
                        .map(|point| (point.x - center.0).hypot(point.y - center.1))
                        .collect::<Vec<_>>();
                    inner_radius = median(&mut inner_radii);
                    let mut inner_final_residuals = inner_radii
                        .iter()
                        .map(|radius| (radius - inner_radius).abs())
                        .collect::<Vec<_>>();
                    let inner_final_limit =
                        (percentile_f64(&mut inner_final_residuals, 0.50) * 2.8 + 0.6)
                            .clamp(1.0, inner_radius * 0.16);
                    selected_inner.retain(|point| {
                        ((point.x - center.0).hypot(point.y - center.1) - inner_radius).abs()
                            <= inner_final_limit
                    });
                }
                let valid = selected_outer.len() >= 8
                    && selected_inner.len() >= 8
                    && joint_inner_has_bilateral_support(&selected_inner, center, inner_radius)
                    && objective.is_finite();
                let mut outer = outer_input.clone();
                let mut inner = inner_input.clone();
                if valid {
                    let fitted = anchor_joint_outer_fit(
                        fit_outer_ellipse(
                            &selected_outer,
                            [
                                center.0,
                                center.1,
                                coarse.radius.max(outer_input.minor_radius),
                            ],
                        ),
                        outer_input,
                        center,
                    );
                    let equivalent = (fitted[2] * fitted[3]).sqrt();
                    inner_radius = inner_radius.min(equivalent / 1.36);
                    outer.center = center;
                    outer.major_radius = fitted[2];
                    outer.minor_radius = fitted[3];
                    outer.angle = fitted[4];
                    outer.evidence_points = selected_outer.clone();
                    let contrast = selected_outer
                        .iter()
                        .map(|point| point.contrast)
                        .sum::<f64>()
                        / selected_outer.len() as f64;
                    outer.points = stable_screen_space_ellipse_points(
                        [
                            center.0,
                            center.1,
                            outer.major_radius,
                            outer.minor_radius,
                            outer.angle,
                        ],
                        64,
                        contrast.max(1.0),
                    );
                    inner.center = center;
                    inner.radius = inner_radius;
                    inner.major_radius = inner_radius;
                    inner.minor_radius = inner_radius;
                    inner.angle = outer.angle;
                    let score = selected_inner.iter().map(|point| point.score).sum::<f64>()
                        / selected_inner.len() as f64;
                    inner.points = (0..64)
                        .map(|index| {
                            let angle = 2.0 * PI * index as f64 / 64.0;
                            InnerIrisPoint {
                                x: center.0 + inner_radius * angle.cos(),
                                y: center.1 + inner_radius * angle.sin(),
                                score,
                            }
                        })
                        .collect();
                    objective += 0.16 / (selected_outer.len() as f64).sqrt()
                        + 0.16 / (selected_inner.len() as f64).sqrt();
                } else {
                    outer.evidence_points.clear();
                    outer.points.clear();
                    inner.points.clear();
                    objective = f64::INFINITY;
                }
                variants.push(JointIrisEvidenceVariant {
                    name,
                    valid,
                    objective,
                    outer,
                    inner,
                    inner_evidence: selected_inner,
                    outer_candidate_count,
                    inner_candidate_count,
                });
            }
        }
    }
    variants
}

fn quad_bayer_luma(raw: &[u16], width: usize, height: usize, x: f64, y: f64) -> f64 {
    let x = x.round() as isize;
    let y = y.round() as isize;
    let x0 = (x - 1).clamp(0, width.saturating_sub(4) as isize) as usize;
    let y0 = (y - 1).clamp(0, height.saturating_sub(4) as isize) as usize;
    let mut sum = 0u32;
    for dy in 0..4 {
        for dx in 0..4 {
            sum += raw[(y0 + dy) * width + x0 + dx] as u32;
        }
    }
    sum as f64 / 16.0
}

/// Samples a small canonical iris/sclera illumination map after eye geometry
/// has been established. The map is exposure-normalized, fixed-size, and uses
/// complete Bayer cells so it can condition a focus model without introducing
/// CFA phase or an unbounded per-frame image model.
pub fn iris_light_map(
    raw: &[u16],
    width: usize,
    height: usize,
    focus: &BorderFocus,
) -> IrisLightMap {
    if width < 16
        || height < 16
        || raw.len() < width * height
        || focus.radius < 6.0
        || !focus.center.0.is_finite()
        || !focus.center.1.is_finite()
    {
        return IrisLightMap::default();
    }

    const RADII: [f64; IRIS_LIGHT_RADIAL_BANDS] = [0.35, 0.78, 1.18];
    const RADIAL_OFFSETS: [f64; 3] = [-0.07, 0.0, 0.07];
    const ANGULAR_OFFSETS: [f64; 3] = [-PI / 48.0, 0.0, PI / 48.0];
    let mut raw_cells = [0.0; IRIS_LIGHT_MAP_CELLS];
    let mut valid_cells = 0usize;

    for (band, radius_scale) in RADII.iter().enumerate() {
        for sector in 0..IRIS_LIGHT_ANGULAR_SECTORS {
            let base_angle = (sector as f64 + 0.5) * 2.0 * PI / IRIS_LIGHT_ANGULAR_SECTORS as f64;
            let mut sum = 0.0;
            let mut count = 0usize;
            for radial_offset in RADIAL_OFFSETS {
                let radius = focus.radius * (radius_scale + radial_offset);
                for angular_offset in ANGULAR_OFFSETS {
                    let angle = base_angle + angular_offset;
                    let x = focus.center.0 + radius * angle.cos();
                    let y = focus.center.1 + radius * angle.sin();
                    if x < 2.0
                        || y < 2.0
                        || x >= width.saturating_sub(3) as f64
                        || y >= height.saturating_sub(3) as f64
                    {
                        continue;
                    }
                    sum += quad_bayer_luma(raw, width, height, x, y);
                    count += 1;
                }
            }
            if count > 0 {
                raw_cells[band * IRIS_LIGHT_ANGULAR_SECTORS + sector] = sum / count as f64;
                valid_cells += 1;
            }
        }
    }
    if valid_cells < IRIS_LIGHT_MAP_CELLS * 3 / 4 {
        return IrisLightMap::default();
    }

    let mean = raw_cells.iter().sum::<f64>() / valid_cells as f64;
    if !mean.is_finite() || mean < 1.0 {
        return IrisLightMap::default();
    }
    let mut populated = raw_cells;
    for value in &mut populated {
        if *value == 0.0 {
            *value = mean;
        }
    }
    let mut sorted = populated;
    sorted.sort_by(f64::total_cmp);
    let low = sorted[sorted.len() / 10];
    let high = sorted[sorted.len() * 9 / 10];
    let span = (high - low).max(0.0) / mean;
    let mut gradient_x = 0.0;
    let mut gradient_y = 0.0;
    let mut cells = [0.0; IRIS_LIGHT_MAP_CELLS];
    for band in 0..IRIS_LIGHT_RADIAL_BANDS {
        for sector in 0..IRIS_LIGHT_ANGULAR_SECTORS {
            let index = band * IRIS_LIGHT_ANGULAR_SECTORS + sector;
            let normalized = (populated[index] / mean).clamp(0.0, 4.0);
            cells[index] = normalized;
            let angle = (sector as f64 + 0.5) * 2.0 * PI / IRIS_LIGHT_ANGULAR_SECTORS as f64;
            gradient_x += normalized * angle.cos();
            gradient_y += normalized * angle.sin();
        }
    }
    let gradient_scale = 2.0 / IRIS_LIGHT_MAP_CELLS as f64;
    IrisLightMap {
        valid: true,
        mean,
        span,
        gradient_x: gradient_x * gradient_scale,
        gradient_y: gradient_y * gradient_scale,
        cells,
    }
}

fn cfa_detail(
    raw: &[u16],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    filter: FocusFilter,
) -> f64 {
    let x = (x.floor() as usize & !1).clamp(0, width.saturating_sub(2));
    let y = (y.floor() as usize & !1).clamp(0, height.saturating_sub(2));
    let i = y * width + x;
    let red = raw[i] as f64;
    let green = (raw[i + 1] as f64 + raw[i + width] as f64) * 0.5;
    let blue = raw[i + width + 1] as f64;
    match filter {
        FocusFilter::LumaHighPass | FocusFilter::LumaBandPass => (red + green * 2.0 + blue) * 0.25,
        FocusFilter::GreenHighPass => green,
        FocusFilter::RedHighPass => red,
        FocusFilter::BlueHighPass => blue,
        FocusFilter::RedGreenHighPass => red - green,
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn histogram_percentile(histogram: &[u32], count: usize, percentile: usize) -> usize {
    if count == 0 {
        return 0;
    }
    let target = (count.saturating_sub(1) * percentile.min(100)) / 100;
    let mut accumulated = 0usize;
    for (value, samples) in histogram.iter().enumerate() {
        accumulated = accumulated.saturating_add(*samples as usize);
        if accumulated > target {
            return value;
        }
    }
    histogram.len().saturating_sub(1)
}

fn neutral_quad_plane(raw: &[u16], width: usize, height: usize) -> Option<(Vec<u8>, u16, u16)> {
    let plane_width = width / 4;
    let plane_height = height / 4;
    if plane_width < 16 || plane_height < 12 {
        return None;
    }
    let mut neutral = vec![0u16; plane_width * plane_height];
    let mut histogram = [0u32; 1024];
    for y in 0..plane_height {
        for x in 0..plane_width {
            let mut sum = 0u32;
            for dy in 0..4 {
                let row = (y * 4 + dy) * width + x * 4;
                for dx in 0..4 {
                    sum += raw[row + dx] as u32;
                }
            }
            let value = ((sum + 8) / 16).min(1023) as u16;
            neutral[y * plane_width + x] = value;
            histogram[value as usize] += 1;
        }
    }
    let low = histogram_percentile(&histogram, neutral.len(), 2) as u16;
    let high = histogram_percentile(&histogram, neutral.len(), 98) as u16;
    if high <= low.saturating_add(12) {
        return None;
    }
    let range = (high - low) as u32;
    let normalized = neutral
        .into_iter()
        .map(|value| ((value.saturating_sub(low) as u32 * 255 / range).min(255)) as u8)
        .collect();
    Some((normalized, low, high))
}

/// Ranks optical sharpness inside a semantically proposed eye crop without
/// claiming that the crop contains valid eye anatomy. Quad-Bayer cells are
/// averaged before measuring detail, so CFA phase and chroma do not masquerade
/// as focus. This is only an acquisition metric; model admission continues to
/// require the stricter iris/sclera evidence returned by `score_stream_eye`.
pub fn provisional_focus_score(raw: &[u16], width: usize, height: usize) -> f64 {
    let Some((neutral, _, _)) = neutral_quad_plane(raw, width, height) else {
        return 0.0;
    };
    let plane_width = width / 4;
    let plane_height = height / 4;
    // The per-frame percentile normalization needed for exposure tolerance
    // also stretches read noise when the optical image disappears. Measure
    // only detail that survives three low-pass stages (roughly a 24-sensor-pixel
    // support) so CFA/read noise cannot win a VCM sweep over an iris edge.
    let neutral = blur_neutral_plane(&neutral, plane_width, plane_height);
    let neutral = blur_neutral_plane(&neutral, plane_width, plane_height);
    let neutral = blur_neutral_plane(&neutral, plane_width, plane_height);
    let margin_x = (plane_width / 8).max(2);
    let margin_y = (plane_height / 8).max(2);
    if plane_width <= margin_x * 2 + 2 || plane_height <= margin_y * 2 + 2 {
        return 0.0;
    }

    let mut detail =
        Vec::with_capacity((plane_width - margin_x * 2) * (plane_height - margin_y * 2));
    for y in margin_y.max(1)..plane_height.saturating_sub(margin_y.max(1)) {
        for x in margin_x.max(1)..plane_width.saturating_sub(margin_x.max(1)) {
            let index = y * plane_width + x;
            let center = neutral[index] as i32;
            let left = neutral[index - 1] as i32;
            let right = neutral[index + 1] as i32;
            let above = neutral[index - plane_width] as i32;
            let below = neutral[index + plane_width] as i32;
            let gradient = (right - left).abs() + (below - above).abs();
            let laplacian = (center * 4 - left - right - above - below).abs();
            detail.push(gradient as f64 * 0.35 + laplacian as f64 * 0.65);
        }
    }
    if detail.len() < 64 {
        return 0.0;
    }
    detail.sort_by(f64::total_cmp);
    let percentile = |numerator: usize| {
        detail[(detail.len().saturating_sub(1) * numerator / 100).min(detail.len() - 1)]
    };
    // Strong detail is sparse in an eye crop. Blending the 90th and 98th
    // percentiles preserves iris/eyelid edges without allowing one hot pixel
    // or one lens glint to control the VCM decision.
    percentile(90) * 0.65 + percentile(98) * 0.35
}

fn blur_neutral_plane(input: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = input.to_vec();
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let mut sum = 0u32;
            for (dy, wy) in [(-1isize, 1u32), (0, 2), (1, 1)] {
                for (dx, wx) in [(-1isize, 1u32), (0, 2), (1, 1)] {
                    let index = (y as isize + dy) as usize * width + (x as isize + dx) as usize;
                    sum += input[index] as u32 * wx * wy;
                }
            }
            output[y * width + x] = ((sum + 8) / 16) as u8;
        }
    }
    output
}

fn opened_dark_mask(input: &[u8], width: usize, height: usize) -> Vec<bool> {
    let mut histogram = [0u32; 256];
    for value in input {
        histogram[*value as usize] += 1;
    }
    let threshold = histogram_percentile(&histogram, input.len(), 15) as u8;
    let dark = input
        .iter()
        .map(|value| *value <= threshold)
        .collect::<Vec<_>>();
    let mut eroded = vec![false; input.len()];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let index = y * width + x;
            eroded[index] = dark[index]
                && dark[index - 1]
                && dark[index + 1]
                && dark[index - width]
                && dark[index + width];
        }
    }
    let mut opened = vec![false; input.len()];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let index = y * width + x;
            opened[index] = eroded[index]
                || eroded[index - 1]
                || eroded[index + 1]
                || eroded[index - width]
                || eroded[index + width];
        }
    }
    opened
}

#[derive(Clone, Copy, Debug)]
struct DarkBasin {
    center: (f64, f64),
    radius: f64,
    axis_ratio: f64,
    axis_angle: f64,
}

#[derive(Clone, Copy, Debug)]
struct CoarseLimbusSeed {
    center: (f64, f64),
    radius: f64,
    axis_ratio: f64,
    axis_angle: f64,
    score: f64,
    pupil: Option<ReducedPupilCandidate>,
    visible_arc_fraction: f64,
    supported_probe_fraction: f64,
    censored_edges: u8,
    confidence: f64,
    reframe_delta_px: (f64, f64),
}

#[derive(Clone, Copy, Debug)]
struct ReducedLimbusCandidate {
    center: (f64, f64),
    radius: f64,
    score: f64,
    material_score: f64,
    pupil: Option<ReducedPupilCandidate>,
    visible_probes: u8,
    supported_probes: u8,
    censored_probes: u8,
    censored_edges: u8,
    interior_dark_fraction: f64,
    interior_level: f64,
    iris_level: f64,
}

#[derive(Clone, Copy, Debug)]
struct ReducedPupilCandidate {
    center: (f64, f64),
    radius: f64,
    score: f64,
}

fn reduced_plane_sample(plane: &[u8], width: usize, height: usize, x: f64, y: f64) -> Option<f64> {
    if x < 0.0
        || y < 0.0
        || x > width.saturating_sub(1) as f64
        || y > height.saturating_sub(1) as f64
    {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let top = plane[y0 * width + x0] as f64 * (1.0 - fx) + plane[y0 * width + x1] as f64 * fx;
    let bottom = plane[y1 * width + x0] as f64 * (1.0 - fx) + plane[y1 * width + x1] as f64 * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

fn reduced_limbus_ray_material_direction(
    plane: &[u8],
    width: usize,
    height: usize,
    center: (f64, f64),
    radius: f64,
    direction: (f64, f64),
) -> Option<(f64, f64, bool)> {
    let (cos, sin) = direction;
    let at = |fraction: f64| {
        reduced_plane_sample(
            plane,
            width,
            height,
            center.0 + cos * radius * fraction,
            center.1 + sin * radius * fraction,
        )
    };
    let mut iris = [at(0.52)?, at(0.70)?, at(0.86)?];
    let boundary_inside = at(0.94)?;
    let boundary_outside = at(1.06)?;
    let mut outside = [0.0; 3];
    outside[0] = boundary_outside;
    let mut outside_count = 1usize;
    for fraction in [1.18, 1.30] {
        if let Some(sample) = at(fraction) {
            outside[outside_count] = sample;
            outside_count += 1;
        }
    }
    iris.sort_by(f64::total_cmp);
    outside[..outside_count].sort_by(f64::total_cmp);
    let iris_level = iris[1];
    let outside_level = outside[outside_count / 2];
    let narrow_step = boundary_outside - boundary_inside;
    let sustained_step = outside_level - iris_level;
    let far_step = outside[0] - iris_level;
    let negative_penalty = (-narrow_step).max(0.0) * 0.70 + (-far_step).max(0.0) * 0.45;
    let quality = 0.55 * narrow_step.max(0.0) + sustained_step.max(0.0) + 0.30 * far_step.max(0.0)
        - negative_penalty;
    let supported = narrow_step >= 1.5 && sustained_step >= 6.0 && far_step >= 2.5;
    Some((quality, iris_level, supported))
}

fn reduced_limbus_ray_material(
    plane: &[u8],
    width: usize,
    height: usize,
    center: (f64, f64),
    radius: f64,
    angle: f64,
) -> Option<(f64, f64, bool)> {
    let (sin, cos) = angle.sin_cos();
    reduced_limbus_ray_material_direction(plane, width, height, center, radius, (cos, sin))
}

fn reduced_sector_score(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let median = values[values.len() / 2];
    let upper = values[values.len() * 3 / 4];
    0.72 * median + 0.28 * upper
}

fn reduced_circle_censored_edges(
    center: (f64, f64),
    radius: f64,
    width: usize,
    height: usize,
) -> u8 {
    let mut edges = 0u8;
    if center.0 - radius < 0.0 {
        edges |= ROI_TRUNCATED_LEFT;
    }
    if center.0 + radius > width.saturating_sub(1) as f64 {
        edges |= ROI_TRUNCATED_RIGHT;
    }
    if center.1 - radius < 0.0 {
        edges |= ROI_TRUNCATED_TOP;
    }
    if center.1 + radius > height.saturating_sub(1) as f64 {
        edges |= ROI_TRUNCATED_BOTTOM;
    }
    edges
}

fn native_ellipse_censored_edges(ellipse: [f64; 5], width: usize, height: usize) -> u8 {
    let (sin, cos) = ellipse[4].sin_cos();
    let extent_x = (ellipse[2] * cos).hypot(ellipse[3] * sin);
    let extent_y = (ellipse[2] * sin).hypot(ellipse[3] * cos);
    let maximum_x = width.saturating_sub(1) as f64;
    let maximum_y = height.saturating_sub(1) as f64;
    let mut edges = 0u8;
    if ellipse[0] - extent_x < 0.0 {
        edges |= ROI_TRUNCATED_LEFT;
    }
    if ellipse[0] + extent_x > maximum_x {
        edges |= ROI_TRUNCATED_RIGHT;
    }
    if ellipse[1] - extent_y < 0.0 {
        edges |= ROI_TRUNCATED_TOP;
    }
    if ellipse[1] + extent_y > maximum_y {
        edges |= ROI_TRUNCATED_BOTTOM;
    }
    edges
}

fn reduced_probe_is_roi_censored(
    center: (f64, f64),
    radius: f64,
    direction: (f64, f64),
    width: usize,
    height: usize,
) -> bool {
    let x = center.0 + direction.0 * radius * 1.06;
    let y = center.1 + direction.1 * radius * 1.06;
    x < 0.0 || y < 0.0 || x > width.saturating_sub(1) as f64 || y > height.saturating_sub(1) as f64
}

fn score_reduced_pupil_candidate(
    plane: &[u8],
    width: usize,
    height: usize,
    center: (f64, f64),
    radius: f64,
    dark_center_limit: f64,
) -> Option<ReducedPupilCandidate> {
    const DIRECTIONS: [(f64, f64); 16] = [
        (1.0, 0.0),
        (0.9238795325, 0.3826834324),
        (0.7071067812, 0.7071067812),
        (0.3826834324, 0.9238795325),
        (0.0, 1.0),
        (-0.3826834324, 0.9238795325),
        (-0.7071067812, 0.7071067812),
        (-0.9238795325, 0.3826834324),
        (-1.0, 0.0),
        (-0.9238795325, -0.3826834324),
        (-0.7071067812, -0.7071067812),
        (-0.3826834324, -0.9238795325),
        (0.0, -1.0),
        (0.3826834324, -0.9238795325),
        (0.7071067812, -0.7071067812),
        (0.9238795325, -0.3826834324),
    ];
    let mut core = [0.0; 9];
    core[0] = reduced_plane_sample(plane, width, height, center.0, center.1)?;
    for (index, (cos, sin)) in DIRECTIONS.into_iter().step_by(2).enumerate() {
        core[index + 1] = reduced_plane_sample(
            plane,
            width,
            height,
            center.0 + cos * radius * 0.42,
            center.1 + sin * radius * 0.42,
        )?;
    }
    core.sort_by(f64::total_cmp);
    let core_level = core[3];
    if core_level > dark_center_limit {
        return None;
    }
    let mut ring = [0.0; DIRECTIONS.len()];
    let mut bright = [false; DIRECTIONS.len()];
    let mut quadrant_support = [0usize; 4];
    for (index, (cos, sin)) in DIRECTIONS.into_iter().enumerate() {
        ring[index] = reduced_plane_sample(
            plane,
            width,
            height,
            center.0 + cos * radius * 1.30,
            center.1 + sin * radius * 1.30,
        )?;
        bright[index] = ring[index] >= core_level + 3.0;
        quadrant_support[index / 4] += usize::from(bright[index]);
    }
    // A pupil is a compact void. Exact opposing horizontal samples must both
    // leave it; this is specifically what a long glasses rim or lash cannot
    // explain, even when its diagonal samples happen to be bright.
    if !bright[0]
        || !bright[8]
        || bright.iter().filter(|value| **value).count() < 10
        || quadrant_support.iter().any(|count| *count < 2)
    {
        return None;
    }
    ring.sort_by(f64::total_cmp);
    let lower_quartile = ring[4];
    let ring_median = ring[8];
    if lower_quartile < core_level + 2.0 {
        return None;
    }
    let score = ring_median - core_level
        + 0.62 * (lower_quartile - core_level)
        + bright.iter().filter(|value| **value).count() as f64 * 0.75
        + radius * 4.0;
    Some(ReducedPupilCandidate {
        center,
        radius,
        score,
    })
}

fn reduced_pupil_centers(
    plane: &[u8],
    width: usize,
    height: usize,
    dark_center_limit: f64,
) -> Vec<ReducedPupilCandidate> {
    let minimum_y = (height * 16 / 100).max(4);
    let mut leaders = Vec::with_capacity(24);
    for y in (minimum_y..height.saturating_sub(4)).step_by(2) {
        for x in (4..width.saturating_sub(4)).step_by(2) {
            for radius in [3.5, 5.5, 7.5, 9.0, 11.5, 14.0] {
                if let Some(candidate) = score_reduced_pupil_candidate(
                    plane,
                    width,
                    height,
                    (x as f64, y as f64),
                    radius,
                    dark_center_limit,
                ) {
                    leaders.push(candidate);
                }
            }
        }
    }
    leaders.sort_by(|left, right| right.score.total_cmp(&left.score));
    // Preserve spatially distinct voids. A glasses reflection may be the
    // strongest compact blob, but it must not delete the real pupil from the
    // beam before their surrounding meridians are compared.
    let mut diverse = Vec::with_capacity(10);
    for candidate in leaders {
        let separated = diverse.iter().all(|old: &ReducedPupilCandidate| {
            (old.center.0 - candidate.center.0).hypot(old.center.1 - candidate.center.1)
                >= (old.radius + candidate.radius) * 0.72
        });
        if separated {
            diverse.push(candidate);
            if diverse.len() == 10 {
                break;
            }
        }
    }
    let mut refined = Vec::with_capacity(diverse.len());
    for leader in diverse {
        let mut best = leader;
        for dy in -1..=1 {
            for dx in -1..=1 {
                for dr in [-1.0, 0.0, 1.0] {
                    let center = (leader.center.0 + dx as f64, leader.center.1 + dy as f64);
                    let radius = leader.radius + dr;
                    if radius < 3.0 || radius > 15.0 {
                        continue;
                    }
                    if let Some(candidate) = score_reduced_pupil_candidate(
                        plane,
                        width,
                        height,
                        center,
                        radius,
                        dark_center_limit,
                    ) {
                        if candidate.score > best.score {
                            best = candidate;
                        }
                    }
                }
            }
        }
        refined.push(best);
    }
    refined.sort_by(|left, right| right.score.total_cmp(&left.score));
    refined
}

fn reduced_pupil_near_limbus(
    plane: &[u8],
    width: usize,
    height: usize,
    limbus: ReducedLimbusCandidate,
    dark_center_limit: f64,
) -> Option<ReducedPupilCandidate> {
    // Do not let an unrelated dark blob elsewhere in the ROI nominate an
    // eye.  A pupil hypothesis belongs to one limbus hypothesis: search only
    // the small projective displacement allowed inside that candidate and
    // require a compact dark void with opposing exits into iris material.
    let offset = (limbus.radius * 0.18).clamp(1.0, 4.5);
    let mut best = None;
    for dy in [-offset, 0.0, offset] {
        for dx in [-offset, 0.0, offset] {
            let center = (limbus.center.0 + dx, limbus.center.1 + dy);
            for fraction in [0.20, 0.27, 0.34, 0.42, 0.50] {
                let radius = limbus.radius * fraction;
                if !(3.0..=11.5).contains(&radius) {
                    continue;
                }
                let Some(mut pupil) = score_reduced_pupil_candidate(
                    plane,
                    width,
                    height,
                    center,
                    radius,
                    dark_center_limit,
                ) else {
                    continue;
                };
                let displacement = (pupil.center.0 - limbus.center.0)
                    .hypot(pupil.center.1 - limbus.center.1)
                    / limbus.radius.max(1.0);
                pupil.score += 12.0 * (0.30 - displacement).max(0.0);
                if best.is_none_or(|old: ReducedPupilCandidate| pupil.score > old.score) {
                    best = Some(pupil);
                }
            }
        }
    }
    best
}

fn add_reduced_pupil_topology(
    plane: &[u8],
    width: usize,
    height: usize,
    dark_center_limit: f64,
    mut candidate: ReducedLimbusCandidate,
) -> ReducedLimbusCandidate {
    let local = reduced_pupil_near_limbus(plane, width, height, candidate, dark_center_limit);
    let plausible_pupil = |pupil: ReducedPupilCandidate| {
        let ratio = pupil.radius / candidate.radius.max(1.0);
        (0.14..=0.60).contains(&ratio).then_some(pupil)
    };
    candidate.pupil = match (
        candidate.pupil.and_then(plausible_pupil),
        local.and_then(plausible_pupil),
    ) {
        (Some(guided), Some(local)) => Some(if local.score > guided.score {
            local
        } else {
            guided
        }),
        (guided, local) => guided.or(local),
    };
    if let Some(pupil) = candidate.pupil {
        let displacement = (pupil.center.0 - candidate.center.0)
            .hypot(pupil.center.1 - candidate.center.1)
            / candidate.radius.max(1.0);
        let pupil_ratio = pupil.radius / candidate.radius.max(1.0);
        let compactness = ((0.60 - pupil_ratio) / 0.30).clamp(0.0, 1.0);
        // The pupil nominates a limbus neighborhood; it must not outvote the
        // opposing lateral/lower meridians that actually establish the outer
        // iris.  In particular, do not count the same guided compact void both
        // before and after the limbus shortlist.  Cap its contribution so a
        // spectacle reflection with an excellent pupil-like score cannot beat
        // a much larger, bilaterally supported eye. A compact pupil remains a
        // strong cue; its influence tapers continuously toward the anatomical
        // ratio limit instead of changing discontinuously at that limit.
        let pupil_weight = 0.16 + 0.46 * compactness;
        candidate.score +=
            pupil_weight * pupil.score.min(160.0) + 18.0 * (0.30 - displacement).max(0.0);
    }
    candidate
}

fn score_reduced_limbus_candidate_mode(
    plane: &[u8],
    width: usize,
    height: usize,
    center: (f64, f64),
    radius: f64,
    dark_center_limit: f64,
    pupil_guided: bool,
) -> Option<ReducedLimbusCandidate> {
    const CORE_DIRECTIONS: [(f64, f64); 8] = [
        (1.0, 0.0),
        (0.7071067812, 0.7071067812),
        (0.0, 1.0),
        (-0.7071067812, 0.7071067812),
        (-1.0, 0.0),
        (-0.7071067812, -0.7071067812),
        (0.0, -1.0),
        (0.7071067812, -0.7071067812),
    ];
    let mut core = [0.0; 9];
    core[0] = reduced_plane_sample(plane, width, height, center.0, center.1)?;
    for (index, (cos, sin)) in CORE_DIRECTIONS.into_iter().enumerate() {
        core[index + 1] = reduced_plane_sample(
            plane,
            width,
            height,
            center.0 + radius * 0.20 * cos,
            center.1 + radius * 0.20 * sin,
        )?;
    }
    core.sort_by(f64::total_cmp);
    // A compact corneal glint may occupy one or two samples, but most of this
    // two-dimensional core must remain pupil/iris-dark. A long glasses rim has
    // dark horizontal samples and bright vertical samples and therefore fails
    // the upper-core check below.
    let core_level = core[4];
    let core_upper = core[6];
    if core_level > dark_center_limit {
        return None;
    }

    // The acquisition objective deliberately excludes the upper central
    // third. Five rays establish each lateral side and seven more establish
    // the exposed lower arc. A horizontal lid/lash band cannot satisfy both
    // opposing lateral material transitions at one common radius.
    const DIRECTIONS: [(f64, f64); 17] = [
        (0.8660254038, -0.5),
        (0.9659258263, -0.2588190451),
        (1.0, 0.0),
        (0.9659258263, 0.2588190451),
        (0.8660254038, 0.5),
        (0.7071067812, 0.7071067812),
        (0.5, 0.8660254038),
        (0.2588190451, 0.9659258263),
        (0.0, 1.0),
        (-0.2588190451, 0.9659258263),
        (-0.5, 0.8660254038),
        (-0.7071067812, 0.7071067812),
        (-0.8660254038, 0.5),
        (-0.9659258263, 0.2588190451),
        (-1.0, 0.0),
        (-0.9659258263, -0.2588190451),
        (-0.8660254038, -0.5),
    ];
    let mut right = [0.0; 5];
    let mut lower = [0.0; 7];
    let mut left = [0.0; 5];
    let mut all = [0.0; DIRECTIONS.len()];
    let mut iris_levels = [0.0; DIRECTIONS.len()];
    let mut right_count = 0usize;
    let mut lower_count = 0usize;
    let mut left_count = 0usize;
    let mut all_count = 0usize;
    let mut right_support = 0usize;
    let mut lower_support = 0usize;
    let mut left_support = 0usize;
    let mut support_flags = [false; DIRECTIONS.len()];
    let mut censored_probes = 0usize;
    for (index, direction) in DIRECTIONS.into_iter().enumerate() {
        let Some((quality, iris_level, supported)) =
            reduced_limbus_ray_material_direction(plane, width, height, center, radius, direction)
        else {
            censored_probes += usize::from(reduced_probe_is_roi_censored(
                center, radius, direction, width, height,
            ));
            continue;
        };
        all[all_count] = quality;
        iris_levels[all_count] = iris_level;
        all_count += 1;
        support_flags[index] = supported;
        if index < 5 {
            right[right_count] = quality;
            right_count += 1;
            right_support += usize::from(supported);
        } else if index < 12 {
            lower[lower_count] = quality;
            lower_count += 1;
            lower_support += usize::from(supported);
        } else {
            left[left_count] = quality;
            left_count += 1;
            left_support += usize::from(supported);
        }
    }
    let right_clipped = center.0 + radius * 1.06 >= width.saturating_sub(1) as f64;
    let left_clipped = center.0 - radius * 1.06 <= 0.0;
    let lower_clipped = center.1 + radius * 1.06 >= height.saturating_sub(1) as f64;
    if ((!pupil_guided || !right_clipped) && right_count < 3)
        || ((!pupil_guided || !left_clipped) && left_count < 3)
    {
        return None;
    }
    // Do not let the two diagonal exits from a long spectacle rim or eyelid
    // masquerade as a bilateral circle. At least two of the three nearly
    // horizontal rays on *each* side must independently leave iris material.
    // This is the coarse equivalent of the native solver's opposing-meridian
    // comparator and eliminates the observed top-border micro-circle alias.
    let opposing_pairs = (0..5)
        .filter(|right_index| support_flags[*right_index] && support_flags[12 + *right_index])
        .count();
    let opposing_lateral = (support_flags[2] && support_flags[14])
        || (pupil_guided && opposing_pairs >= 1 && right_support >= 2 && left_support >= 2);
    let bilateral = opposing_lateral
        && right_support >= 2
        && left_support >= 2
        && lower_count >= 4
        && lower_support >= 1;
    let lower_occlusion = opposing_lateral
        && ((right_support >= 3 && left_support >= 3)
            || (lower_clipped && right_support >= 2 && left_support >= 2));
    let lateral_clipping_occlusion = pupil_guided
        && ((right_clipped && left_support >= 3) || (left_clipped && right_support >= 3))
        && (lower_support >= 1 || lower_clipped);
    if !bilateral && !lower_occlusion && !lateral_clipping_occlusion {
        return None;
    }
    let right_score = reduced_sector_score(&mut right[..right_count]);
    let left_score = reduced_sector_score(&mut left[..left_count]);
    let lower_score = reduced_sector_score(&mut lower[..lower_count]);
    let all_score = reduced_sector_score(&mut all[..all_count]);
    iris_levels[..all_count].sort_by(f64::total_cmp);
    let iris_level = iris_levels[all_count / 2];
    if core_upper > iris_level + 12.0 {
        return None;
    }
    let pupil_void_reward = (iris_level - core_level).clamp(0.0, 36.0);
    if pupil_void_reward < 2.0 {
        return None;
    }
    let weaker_lateral = right_score.min(left_score);
    let stronger_lateral = right_score.max(left_score);
    let support = right_support + lower_support + left_support;
    let material_score = 0.92 * weaker_lateral
        + 0.38 * stronger_lateral
        + if lower_count >= 4 {
            0.58 * lower_score
        } else {
            0.0
        }
        + 0.25 * all_score
        + 0.28 * pupil_void_reward
        + support as f64 * 1.25
        // Apply the weak whole-iris preference before shortlisting, so a
        // large partially clipped eye is not crowded out by many tiny aliases
        // from one high-contrast glasses edge.
        + 1.10 * radius;
    let mut interior = [0.0; 48];
    let mut interior_count = 0usize;
    for ray in 0..24 {
        let angle = 2.0 * PI * ray as f64 / 24.0;
        for fraction in [0.48, 0.74] {
            if let Some(value) = reduced_plane_sample(
                plane,
                width,
                height,
                center.0 + angle.cos() * radius * fraction,
                center.1 + angle.sin() * radius * fraction,
            ) {
                interior[interior_count] = value;
                interior_count += 1;
            }
        }
    }
    if interior_count < 12 {
        return None;
    }
    let interior_dark = interior[..interior_count]
        .iter()
        .filter(|value| **value <= iris_level + 6.0)
        .count() as f64
        / interior_count as f64;
    interior[..interior_count].sort_by(f64::total_cmp);
    let interior_level = interior[interior_count / 2];
    (material_score >= 20.0).then_some(ReducedLimbusCandidate {
        center,
        radius,
        score: material_score,
        material_score,
        pupil: None,
        visible_probes: all_count.min(u8::MAX as usize) as u8,
        supported_probes: support.min(u8::MAX as usize) as u8,
        censored_probes: censored_probes.min(u8::MAX as usize) as u8,
        censored_edges: reduced_circle_censored_edges(center, radius, width, height),
        interior_dark_fraction: interior_dark,
        interior_level,
        iris_level,
    })
}

fn score_reduced_limbus_candidate(
    plane: &[u8],
    width: usize,
    height: usize,
    center: (f64, f64),
    radius: f64,
    dark_center_limit: f64,
) -> Option<ReducedLimbusCandidate> {
    score_reduced_limbus_candidate_mode(
        plane,
        width,
        height,
        center,
        radius,
        dark_center_limit,
        false,
    )
}

fn coarse_limbus_seed_from_plane(
    plane: &[u8],
    width: usize,
    height: usize,
) -> Option<CoarseLimbusSeed> {
    if width < 24 || height < 16 || plane.len() < width * height {
        return None;
    }
    let mut histogram = [0u32; 256];
    for value in plane {
        histogram[*value as usize] += 1;
    }
    let dark_center_limit =
        (histogram_percentile(&histogram, plane.len(), 58) as f64 + 18.0).min(220.0);
    let global_pupils = reduced_pupil_centers(plane, width, height, dark_center_limit);
    let minimum_radius = (width.min(height) as f64 * 0.16).max(10.0);
    let maximum_radius = (width.min(height) as f64 * 0.44).min(29.0);
    if maximum_radius < minimum_radius {
        return None;
    }

    // Fixed coarse-to-fine work replaces an unbounded Hough search. At this
    // 4x-reduced resolution the first pass has at most about 3,500 hypotheses;
    // only the six leaders receive one-cell center/radius refinement.
    let mut leaders = Vec::with_capacity(32);
    let mut truncated_leaders = Vec::with_capacity(32);
    let minimum_center_y = (height * 16 / 100).max(4);
    let minimum_center_x = 4;
    let maximum_center_x = width.saturating_sub(3);
    let maximum_center_y = height.saturating_sub(3);
    let center_step = 4;
    for center_y in (minimum_center_y..maximum_center_y).step_by(center_step) {
        for center_x in (minimum_center_x..maximum_center_x).step_by(center_step) {
            let mut radius = minimum_radius;
            while radius <= maximum_radius {
                if let Some(candidate) = score_reduced_limbus_candidate(
                    plane,
                    width,
                    height,
                    (center_x as f64, center_y as f64),
                    radius,
                    dark_center_limit,
                ) {
                    if candidate.censored_edges == 0 {
                        leaders.push(candidate);
                    } else {
                        truncated_leaders.push(candidate);
                    }
                }
                radius += 2.0;
            }
        }
    }
    // A crop edge censors evidence; it is not negative evidence. Run a
    // separate, still fixed-size beam only for circles that intersect an ROI
    // boundary. These hypotheses must later win on their observable material
    // ordering, rather than receiving a generic edge bonus.
    for center_y in (minimum_center_y..maximum_center_y).step_by(center_step) {
        for center_x in (minimum_center_x..maximum_center_x).step_by(center_step) {
            let center = (center_x as f64, center_y as f64);
            let mut radius = minimum_radius;
            while radius <= maximum_radius {
                if reduced_circle_censored_edges(center, radius, width, height) != 0 {
                    if let Some(candidate) = score_reduced_limbus_candidate_mode(
                        plane,
                        width,
                        height,
                        center,
                        radius,
                        dark_center_limit,
                        true,
                    ) {
                        truncated_leaders.push(candidate);
                    }
                }
                radius += 2.0;
            }
        }
    }
    // A 9x6-style spatial beam keeps distinct compact voids alive. Around
    // each void, test only a small family of 3D-meridian-compatible limbus
    // scales. This recovers a large, clipped eye even when a smaller glasses
    // reflection has the single strongest neutral-intensity circle.
    for pupil in global_pupils.iter().copied().take(8) {
        let center_offset = (pupil.radius * 0.32).clamp(1.0, 4.0);
        for (dx, dy) in [
            (0.0, 0.0),
            (-center_offset, 0.0),
            (center_offset, 0.0),
            (0.0, -center_offset),
            (0.0, center_offset),
        ] {
            let center = (pupil.center.0 + dx, pupil.center.1 + dy);
            let mut radius = minimum_radius.max(pupil.radius * 1.45);
            // A very dark pupil with several glints may be represented by a
            // compact subfeature rather than its whole void. Keep radius
            // bounded by the eye-sized acquisition range, not by that feature
            // alone; the meridian checks still have to explain the limbus.
            let pupil_maximum = maximum_radius;
            while radius <= pupil_maximum {
                if let Some(mut candidate) = score_reduced_limbus_candidate_mode(
                    plane,
                    width,
                    height,
                    center,
                    radius,
                    dark_center_limit,
                    true,
                ) {
                    let displacement = (pupil.center.0 - center.0).hypot(pupil.center.1 - center.1)
                        / radius.max(1.0);
                    let pupil_ratio = pupil.radius / radius.max(1.0);
                    if displacement <= 0.34 && (0.14..=0.60).contains(&pupil_ratio) {
                        candidate.pupil = Some(pupil);
                        // Pupil topology is scored once after the bounded
                        // limbus shortlist.  Here it only defines a spatial
                        // beam and rewards projective co-location.
                        candidate.score += 18.0 * (0.34 - displacement).max(0.0);
                        if candidate.censored_edges == 0 {
                            leaders.push(candidate);
                        } else {
                            truncated_leaders.push(candidate);
                        }
                    }
                }
                radius += 2.0;
            }
        }
    }
    leaders.sort_by(|left, right| right.score.total_cmp(&left.score));
    // Pupil topology is more expensive than a 17-ray material check.  Apply
    // it only to a bounded shortlist, then rerank before fine refinement.
    leaders.truncate(64);
    for candidate in &mut leaders {
        *candidate =
            add_reduced_pupil_topology(plane, width, height, dark_center_limit, *candidate);
    }
    leaders.sort_by(|left, right| right.score.total_cmp(&left.score));
    leaders.truncate(4);
    let mut best = leaders.first().copied();
    for leader in leaders {
        for dy in -2..=2 {
            for dx in -2..=2 {
                let center = (leader.center.0 + dx as f64, leader.center.1 + dy as f64);
                if center.0 < 2.0
                    || center.1 < 2.0
                    || center.0 > width as f64 - 3.0
                    || center.1 > height as f64 - 3.0
                {
                    continue;
                }
                for dr in -1..=1 {
                    let radius = leader.radius + dr as f64;
                    if radius < minimum_radius || radius > maximum_radius + 1.0 {
                        continue;
                    }
                    if let Some(candidate) = score_reduced_limbus_candidate(
                        plane,
                        width,
                        height,
                        center,
                        radius,
                        dark_center_limit,
                    ) {
                        let candidate = add_reduced_pupil_topology(
                            plane,
                            width,
                            height,
                            dark_center_limit,
                            candidate,
                        );
                        if best.is_none_or(|old| candidate.score > old.score) {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }
    }

    // Preserve a second shortlist ordered primarily by direct material
    // evidence. A compact false pupil elsewhere in the frame must not erase a
    // larger visible limbus arc merely because its missing side lies outside
    // the crop and therefore cannot produce a pupil-like closed void.
    let truncated_rank = |candidate: &ReducedLimbusCandidate| {
        let visible = candidate.visible_probes as f64 / 17.0;
        let support =
            candidate.supported_probes as f64 / (candidate.visible_probes as f64).max(1.0);
        // This is a projected-surface support term, not a generic large-circle
        // reward: only the fraction with observable probes contributes. It
        // prevents a tiny, high-contrast lid/skin corner from outranking a
        // substantially larger coherent iris arc at another crop boundary.
        candidate.material_score
            * candidate.radius
            * visible.sqrt()
            * (0.60 + 0.40 * support.sqrt())
            + 0.12 * (candidate.score - candidate.material_score)
    };
    truncated_leaders.retain(|candidate| {
        candidate.interior_dark_fraction >= 0.72
            && candidate.iris_level >= candidate.interior_level + 8.0
    });
    truncated_leaders.sort_by(|left, right| {
        truncated_rank(right)
            .total_cmp(&truncated_rank(left))
            .then_with(|| right.material_score.total_cmp(&left.material_score))
    });
    truncated_leaders.truncate(48);
    for candidate in &mut truncated_leaders {
        *candidate =
            add_reduced_pupil_topology(plane, width, height, dark_center_limit, *candidate);
    }
    truncated_leaders.sort_by(|left, right| truncated_rank(right).total_cmp(&truncated_rank(left)));
    truncated_leaders.truncate(6);
    let mut best_truncated = truncated_leaders.first().copied();
    for leader in truncated_leaders {
        for dy in -2..=2 {
            for dx in -2..=2 {
                let center = (leader.center.0 + dx as f64, leader.center.1 + dy as f64);
                if center.0 < 2.0
                    || center.1 < 2.0
                    || center.0 > width as f64 - 3.0
                    || center.1 > height as f64 - 3.0
                {
                    continue;
                }
                for dr in -1..=1 {
                    let radius = leader.radius + dr as f64;
                    if radius < minimum_radius || radius > maximum_radius + 1.0 {
                        continue;
                    }
                    let Some(candidate) = score_reduced_limbus_candidate_mode(
                        plane,
                        width,
                        height,
                        center,
                        radius,
                        dark_center_limit,
                        true,
                    ) else {
                        continue;
                    };
                    if candidate.censored_edges == 0 || candidate.censored_probes == 0 {
                        continue;
                    }
                    let candidate = add_reduced_pupil_topology(
                        plane,
                        width,
                        height,
                        dark_center_limit,
                        candidate,
                    );
                    if best_truncated
                        .is_none_or(|old| truncated_rank(&candidate) > truncated_rank(&old))
                    {
                        best_truncated = Some(candidate);
                    }
                }
            }
        }
    }
    let credible_truncated = best_truncated.filter(|candidate| {
        let visible = candidate.visible_probes as usize;
        let supported = candidate.supported_probes as usize;
        visible >= 8
            && supported >= 5
            && candidate.censored_probes >= 2
            && candidate.censored_edges.count_ones() <= 2
            && supported as f64 / visible as f64 >= 0.45
    });
    let best = match (best, credible_truncated) {
        (None, partial) => partial?,
        (Some(complete), None) => complete,
        (Some(complete), Some(partial)) => {
            let material_win = partial.material_score >= complete.material_score + 7.0;
            let larger_coherent_arc = partial.material_score >= complete.material_score * 0.96
                && partial.radius >= complete.radius * 1.10
                && partial.score >= complete.score * 0.78;
            if material_win || larger_coherent_arc {
                partial
            } else {
                complete
            }
        }
    };

    // Turn the circle-like acquisition maximum into an affine seed by finding
    // the best material transition independently on each reliable meridian.
    // The final native-resolution solver still owns the authoritative points;
    // this fit only keeps its search normals close to an oblique limbus.
    let native_seed = [
        best.center.0 * 4.0 + 1.5,
        best.center.1 * 4.0 + 1.5,
        best.radius * 4.0,
    ];
    let mut edge_points = Vec::with_capacity(40);
    for ray in 0..48 {
        let angle = 2.0 * PI * ray as f64 / 48.0;
        if angle.sin() < -0.58 {
            continue;
        }
        let mut best_ray: Option<(f64, f64)> = None;
        let mut radius = best.radius * 0.68;
        while radius <= best.radius * 1.34 {
            if let Some((quality, _, supported)) =
                reduced_limbus_ray_material(plane, width, height, best.center, radius, angle)
            {
                if supported && best_ray.is_none_or(|(_, old_quality)| quality > old_quality) {
                    best_ray = Some((radius, quality));
                }
            }
            radius += 0.5;
        }
        if let Some((radius, quality)) = best_ray {
            edge_points.push(OuterIrisPoint {
                x: (best.center.0 + angle.cos() * radius) * 4.0 + 1.5,
                y: (best.center.1 + angle.sin() * radius) * 4.0 + 1.5,
                contrast: quality.max(1.0),
            });
        }
    }
    let measured = if edge_points.len() >= 10 {
        fit_outer_ellipse(&edge_points, native_seed)
    } else {
        [
            native_seed[0],
            native_seed[1],
            native_seed[2],
            native_seed[2],
            0.0,
        ]
    };
    let measured_radius = (measured[2] * measured[3]).sqrt();
    let measured_shift = (measured[0] - native_seed[0]).hypot(measured[1] - native_seed[1]);
    let native_seed_ellipse = [
        native_seed[0],
        native_seed[1],
        native_seed[2],
        native_seed[2],
        0.0,
    ];
    let native_seed_censored_edges =
        native_ellipse_censored_edges(native_seed_ellipse, width * 4, height * 4);
    let fitted = if native_seed_censored_edges != 0 {
        // A one-sided arc cannot identify the missing half of a conic without
        // a temporal/pose prior. Keep the bounded circle-like acquisition
        // support instead of letting an unconstrained least-squares ellipse
        // move inward and falsely make itself fully visible. The viewer marks
        // this as censored and recenters before publishing complete anatomy.
        native_seed_ellipse
    } else if (native_seed[2] * 0.78..=native_seed[2] * 1.22).contains(&measured_radius)
        && measured_radius >= width.min(height) as f64 * 4.0 * 0.16
        && measured_radius <= width.min(height) as f64 * 4.0 * 0.44
        && measured_shift <= native_seed[2] * 0.28
    {
        measured
    } else {
        native_seed_ellipse
    };
    let censored_edges = native_ellipse_censored_edges(fitted, width * 4, height * 4);
    let visible_arc_fraction = if censored_edges == 0 {
        1.0
    } else {
        let maximum_x = width as f64 * 4.0 - 1.0;
        let maximum_y = height as f64 * 4.0 - 1.0;
        (0..96)
            .filter(|sample| {
                let angle = 2.0 * PI * *sample as f64 / 96.0;
                let local_x = fitted[2] * angle.cos();
                let local_y = fitted[3] * angle.sin();
                let (sin, cos) = fitted[4].sin_cos();
                let x = fitted[0] + cos * local_x - sin * local_y;
                let y = fitted[1] + sin * local_x + cos * local_y;
                (0.0..=maximum_x).contains(&x) && (0.0..=maximum_y).contains(&y)
            })
            .count() as f64
            / 96.0
    };
    let (major_radius, minor_radius, axis_angle) = if fitted[2] >= fitted[3] {
        (fitted[2], fitted[3], fitted[4])
    } else {
        (fitted[3], fitted[2], fitted[4] + PI * 0.5)
    };
    if !projected_circular_limbus_axes_plausible(major_radius, minor_radius) {
        return None;
    }
    Some(CoarseLimbusSeed {
        center: (fitted[0], fitted[1]),
        radius: (major_radius * minor_radius).sqrt(),
        axis_ratio: major_radius / minor_radius,
        axis_angle,
        score: best.score,
        pupil: best.pupil,
        visible_arc_fraction,
        supported_probe_fraction: best.supported_probes as f64
            / (best.visible_probes as f64).max(1.0),
        censored_edges,
        confidence: (0.45 * (best.supported_probes as f64 / (best.visible_probes as f64).max(1.0))
            + 0.25 * (best.visible_probes as f64 / 12.0).clamp(0.0, 1.0)
            + 0.30 * (best.material_score / 180.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0),
        reframe_delta_px: (
            fitted[0] - width as f64 * 2.0,
            fitted[1] - height as f64 * 2.0,
        ),
    })
}

fn truncated_limbus_observation(seed: CoarseLimbusSeed) -> Option<RoiTruncatedLimbusObservation> {
    (seed.censored_edges != 0).then(|| {
        let ratio = seed.axis_ratio.max(1.0).sqrt();
        RoiTruncatedLimbusObservation {
            center: seed.center,
            major_radius: seed.radius * ratio,
            minor_radius: seed.radius / ratio,
            angle: seed.axis_angle,
            visible_arc_fraction: seed.visible_arc_fraction,
            supported_probe_fraction: seed.supported_probe_fraction,
            confidence: seed.confidence,
            censored_edges: seed.censored_edges,
            reframe_delta_px: seed.reframe_delta_px,
        }
    })
}

fn geometric_seed_focus(seed: CoarseLimbusSeed) -> BorderFocus {
    let pupil_hint = seed
        .pupil
        .map(|pupil| (pupil.center.0 * 4.0 + 1.5, pupil.center.1 * 4.0 + 1.5));
    BorderFocus {
        center: seed.center,
        focus_center: Some(seed.center),
        radius: seed.radius,
        axis_ratio: seed.axis_ratio,
        axis_angle: seed.axis_angle,
        acquisition_score: seed.score,
        pupil_hint,
        pupil_hint_radius: seed.pupil.map_or(0.0, |pupil| pupil.radius * 4.0),
        pupil_hint_score: seed.pupil.map_or(0.0, |pupil| pupil.score),
        roi_truncated_limbus: truncated_limbus_observation(seed),
        ..BorderFocus::default()
    }
}

fn apply_geometric_seed(
    mut focus: BorderFocus,
    seed: Option<CoarseLimbusSeed>,
    minimum_radius: f64,
) -> BorderFocus {
    if let Some(seed) = seed {
        let pupil_hint = seed
            .pupil
            .map(|pupil| (pupil.center.0 * 4.0 + 1.5, pupil.center.1 * 4.0 + 1.5));
        focus.center = seed.center;
        focus.focus_center = Some(seed.center);
        focus.radius = seed.radius;
        focus.axis_ratio = seed.axis_ratio;
        focus.axis_angle = seed.axis_angle;
        focus.acquisition_score = seed.score;
        focus.pupil_hint = pupil_hint;
        focus.pupil_hint_radius = seed.pupil.map_or(0.0, |pupil| pupil.radius * 4.0);
        focus.pupil_hint_score = seed.pupil.map_or(0.0, |pupil| pupil.score);
        focus.roi_truncated_limbus = truncated_limbus_observation(seed);
    } else if focus.radius < minimum_radius {
        return BorderFocus::default();
    }
    focus
}

fn largest_eye_basin(mask: &[bool], width: usize, height: usize) -> Option<DarkBasin> {
    let mut visited = vec![false; mask.len()];
    let mut queue = Vec::with_capacity(mask.len() / 4);
    let minimum_area = (mask.len() * 65 / 1000).max(12);
    let maximum_area = mask.len() * 3 / 10;
    let mut best: Option<(usize, f64, f64, f64, f64)> = None;
    for start in 0..mask.len() {
        if !mask[start] || visited[start] {
            continue;
        }
        queue.clear();
        queue.push(start);
        visited[start] = true;
        let mut cursor = 0usize;
        let mut area = 0usize;
        let mut sum_x = 0usize;
        let mut sum_y = 0usize;
        let mut minimum_x = width;
        let mut maximum_x = 0usize;
        let mut minimum_y = height;
        let mut maximum_y = 0usize;
        while cursor < queue.len() {
            let index = queue[cursor];
            cursor += 1;
            let x = index % width;
            let y = index / width;
            area += 1;
            sum_x += x;
            sum_y += y;
            minimum_x = minimum_x.min(x);
            maximum_x = maximum_x.max(x);
            minimum_y = minimum_y.min(y);
            maximum_y = maximum_y.max(y);
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx >= width as isize || ny >= height as isize {
                        continue;
                    }
                    let neighbor = ny as usize * width + nx as usize;
                    if mask[neighbor] && !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push(neighbor);
                    }
                }
            }
        }
        if area < minimum_area || area > maximum_area {
            continue;
        }
        let component_width = maximum_x - minimum_x + 1;
        let component_height = maximum_y - minimum_y + 1;
        let aspect = component_width as f64 / component_height as f64;
        let fill = area as f64 / (component_width * component_height) as f64;
        if !(0.42..=2.40).contains(&aspect) || fill < 0.34 {
            continue;
        }
        let center_x = sum_x as f64 / area as f64;
        let center_y = sum_y as f64 / area as f64;
        if center_x < width as f64 * 0.08
            || center_x > width as f64 * 0.92
            || center_y < height as f64 * 0.28
            || center_y > height as f64 * 0.94
        {
            continue;
        }
        let mut covariance_xx = 0.0;
        let mut covariance_xy = 0.0;
        let mut covariance_yy = 0.0;
        for index in &queue {
            let dx = (*index % width) as f64 - center_x;
            let dy = (*index / width) as f64 - center_y;
            covariance_xx += dx * dx;
            covariance_xy += dx * dy;
            covariance_yy += dy * dy;
        }
        covariance_xx /= area as f64;
        covariance_xy /= area as f64;
        covariance_yy /= area as f64;
        let discriminant =
            ((covariance_xx - covariance_yy).powi(2) + 4.0 * covariance_xy * covariance_xy).sqrt();
        let major = ((covariance_xx + covariance_yy + discriminant) * 0.5).max(1.0e-6);
        let minor = ((covariance_xx + covariance_yy - discriminant) * 0.5).max(1.0e-6);
        let axis_ratio = (major / minor).sqrt().clamp(1.0, 2.4);
        let axis_angle = if axis_ratio < 1.05 {
            0.0
        } else {
            0.5 * (2.0 * covariance_xy).atan2(covariance_xx - covariance_yy)
        };
        if best.map(|candidate| area > candidate.0).unwrap_or(true) {
            best = Some((area, center_x, center_y, axis_ratio, axis_angle));
        }
    }
    best.map(
        |(area, center_x, center_y, axis_ratio, axis_angle)| DarkBasin {
            center: (center_x, center_y),
            radius: (area as f64 / PI).sqrt(),
            axis_ratio,
            axis_angle,
        },
    )
}

fn eye_basin_has_lateral_sclera(
    plane: &[u8],
    width: usize,
    height: usize,
    basin: DarkBasin,
) -> bool {
    let iris_inner2 = (basin.radius * 0.60).powi(2);
    let iris_outer2 = (basin.radius * 0.90).powi(2);
    let inner2 = (basin.radius * 1.05).powi(2);
    let outer2 = (basin.radius * 1.75).powi(2);
    let sclera_inner2 = (basin.radius * 1.10).powi(2);
    let sclera_outer2 = (basin.radius * 1.50).powi(2);
    let mut iris = [0u32; 256];
    let mut sclera = [0u32; 256];
    let mut annulus = [0u32; 256];
    let mut left = [0u32; 256];
    let mut right = [0u32; 256];
    let mut iris_count = 0usize;
    let mut sclera_count = 0usize;
    let mut annulus_count = 0usize;
    let mut left_count = 0usize;
    let mut right_count = 0usize;
    for y in 0..height {
        for x in 0..width {
            let dx = x as f64 - basin.center.0;
            let dy = y as f64 - basin.center.1;
            let distance2 = dx * dx + dy * dy;
            let value = plane[y * width + x] as usize;
            if (iris_inner2..=iris_outer2).contains(&distance2) {
                iris[value] += 1;
                iris_count += 1;
            }
            if (sclera_inner2..=sclera_outer2).contains(&distance2) {
                sclera[value] += 1;
                sclera_count += 1;
            }
            if distance2 < inner2 || distance2 > outer2 {
                continue;
            }
            annulus[value] += 1;
            annulus_count += 1;
            if dy.abs() <= dx.abs() * 0.727 {
                if dx < 0.0 {
                    left[value] += 1;
                    left_count += 1;
                } else {
                    right[value] += 1;
                    right_count += 1;
                }
            }
        }
    }
    iris_count >= 12
        && sclera_count >= 24
        && histogram_percentile(&sclera, sclera_count, 50)
            >= histogram_percentile(&iris, iris_count, 50)
                .saturating_add(MIN_IRIS_SCLERA_RADIAL_CONTRAST)
        && annulus_count >= 24
        && left_count >= 8
        && right_count >= 8
        && histogram_percentile(&annulus, annulus_count, 75) >= 75
        && histogram_percentile(&left, left_count, 50) >= 40
        && histogram_percentile(&right, right_count, 50) >= 40
}

fn neutral_raw_sample(
    raw: &[u16],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    low: u16,
    high: u16,
) -> f64 {
    let x = x.round() as isize;
    let y = y.round() as isize;
    let x0 = (x - 1).clamp(0, width.saturating_sub(4) as isize) as usize;
    let y0 = (y - 1).clamp(0, height.saturating_sub(4) as isize) as usize;
    let mut sum = 0u32;
    for dy in 0..4 {
        for dx in 0..4 {
            sum += raw[(y0 + dy) * width + x0 + dx] as u32;
        }
    }
    let value = (sum as f64 / 16.0 - low as f64).max(0.0);
    (value * 255.0 / (high.saturating_sub(low).max(1) as f64)).clamp(0.0, 255.0)
}

#[derive(Clone, Copy, Debug, Default)]
struct EyeArcSample {
    quality: f64,
    sharpness: f64,
    radius: f64,
    contrast: f64,
    point: (usize, usize),
}

/// Scores only a verified lateral iris/sclera transition in a native Quad
/// Bayer eye slice. The coarse neutral plane rejects eyelids, skin, glasses,
/// and reflections before the bounded native-resolution radial probe runs.
pub fn score_stream_eye(raw: &[u16], width: usize, height: usize) -> BorderFocus {
    if width < 64 || height < 48 || raw.len() < width * height {
        return BorderFocus::default();
    }
    let Some((neutral, low, high)) = neutral_quad_plane(raw, width, height) else {
        return BorderFocus::default();
    };
    let plane_width = width / 4;
    let plane_height = height / 4;
    let minimum_geometric_radius = width.min(height) as f64 * 0.16;
    let blurred = blur_neutral_plane(&neutral, plane_width, plane_height);
    let geometric_seed = coarse_limbus_seed_from_plane(&blurred, plane_width, plane_height);
    let mask = opened_dark_mask(&blurred, plane_width, plane_height);
    let Some(basin) = largest_eye_basin(&mask, plane_width, plane_height) else {
        return geometric_seed.map(geometric_seed_focus).unwrap_or_default();
    };
    let native_center = (basin.center.0 * 4.0 + 1.5, basin.center.1 * 4.0 + 1.5);
    let native_radius = basin.radius * 4.0;
    if !eye_basin_has_lateral_sclera(&blurred, plane_width, plane_height, basin) {
        return geometric_seed.map(geometric_seed_focus).unwrap_or_default();
    }

    let mut rays = [EyeArcSample::default(); 26];
    let mut ray_count = 0usize;
    for sector_center in [0.0, PI] {
        for step in 0..13 {
            let angle = sector_center - PI / 3.0 + step as f64 * PI / 18.0;
            let (sin, cos) = angle.sin_cos();
            let minimum_radius = (native_radius * 0.68).max(12.0);
            let maximum_radius = (native_radius * 1.42).min(width.min(height) as f64 * 0.5);
            let mut radius = minimum_radius;
            let mut best: Option<EyeArcSample> = None;
            while radius <= maximum_radius {
                let at = |offset: f64| {
                    neutral_raw_sample(
                        raw,
                        width,
                        height,
                        native_center.0 + (radius + offset) * cos,
                        native_center.1 + (radius + offset) * sin,
                        low,
                        high,
                    )
                };
                let far_inside = at(-8.0);
                let near_inside = at(-2.0);
                let near_outside = at(2.0);
                let far_outside = at(8.0);
                let broad = far_outside - far_inside;
                let narrow = near_outside - near_inside;
                if broad >= 12.0 && narrow >= 1.0 && far_outside <= 254.0 {
                    let sharpness = narrow / (broad.abs() + 1.0);
                    let quality = sharpness * broad.sqrt();
                    let sample = EyeArcSample {
                        quality,
                        sharpness,
                        radius,
                        contrast: broad,
                        point: (
                            (native_center.0 + radius * cos)
                                .round()
                                .clamp(0.0, width.saturating_sub(1) as f64)
                                as usize,
                            (native_center.1 + radius * sin)
                                .round()
                                .clamp(0.0, height.saturating_sub(1) as f64)
                                as usize,
                        ),
                    };
                    if best.map(|old| quality > old.quality).unwrap_or(true) {
                        best = Some(sample);
                    }
                }
                radius += 1.0;
            }
            if let Some(best) = best {
                rays[ray_count] = best;
                ray_count += 1;
            }
        }
    }
    if ray_count < 10 {
        return apply_geometric_seed(
            BorderFocus {
                eye_basin_valid: true,
                center: native_center,
                focus_center: Some(native_center),
                radius: native_radius,
                axis_ratio: basin.axis_ratio,
                axis_angle: basin.axis_angle,
                ..BorderFocus::default()
            },
            geometric_seed,
            minimum_geometric_radius,
        );
    }
    let mut radii = [0.0; 26];
    for (destination, sample) in radii.iter_mut().zip(&rays[..ray_count]) {
        *destination = sample.radius;
    }
    let median_radius = median(&mut radii[..ray_count]);
    let mut deviations = [0.0; 26];
    for (destination, sample) in deviations.iter_mut().zip(&rays[..ray_count]) {
        *destination = (sample.radius - median_radius).abs();
    }
    let mad = median(&mut deviations[..ray_count]);
    let radius_limit = (mad * 2.5).max(4.0);
    let mut qualities = [0.0; 26];
    let mut sharpnesses = [0.0; 26];
    let mut contrasts = [0.0; 26];
    let mut quality_count = 0usize;
    let mut points = Vec::with_capacity(ray_count);
    for sample in &rays[..ray_count] {
        if (sample.radius - median_radius).abs() > radius_limit {
            continue;
        }
        qualities[quality_count] = sample.quality;
        sharpnesses[quality_count] = sample.sharpness;
        contrasts[quality_count] = sample.contrast;
        quality_count += 1;
        points.push(BorderPoint {
            x: sample.point.0,
            y: sample.point.1,
            quality: sample.quality * sample.contrast.sqrt(),
        });
    }
    if quality_count < 10 {
        return apply_geometric_seed(
            BorderFocus {
                eye_basin_valid: true,
                center: native_center,
                focus_center: Some(native_center),
                radius: median_radius,
                axis_ratio: basin.axis_ratio,
                axis_angle: basin.axis_angle,
                ..BorderFocus::default()
            },
            geometric_seed,
            minimum_geometric_radius,
        );
    }
    let score = median(&mut qualities[..quality_count]) * 20.0;
    let optical_sharpness = median(&mut sharpnesses[..quality_count]);
    let border_contrast = median(&mut contrasts[..quality_count]);
    apply_geometric_seed(
        BorderFocus {
            score,
            optical_sharpness,
            border_contrast,
            eye_basin_valid: true,
            center: native_center,
            focus_center: Some(native_center),
            radius: median_radius,
            axis_ratio: basin.axis_ratio,
            axis_angle: basin.axis_angle,
            points,
            ..BorderFocus::default()
        },
        geometric_seed,
        minimum_geometric_radius,
    )
}

#[cfg(test)]
mod stream_eye_tests {
    use super::*;
    use std::path::PathBuf;

    const WIDTH: usize = 384;
    const HEIGHT: usize = 256;

    #[test]
    fn provisional_camera_envelope_rejects_extreme_projected_limbus_conics() {
        let envelope = PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE;
        assert!(envelope.minimum_focal_length_px <= 3_250.0);
        assert!(envelope.maximum_focal_length_px >= 4_737.6);
        assert!(envelope.maximum_pixel_aspect_error <= 0.01);

        let moderate = envelope.assess_axes(100.0, 72.0).unwrap();
        assert!(moderate.minor_to_major >= moderate.minimum_minor_to_major);
        assert!(envelope.admits_axes(100.0, 72.0));
        assert!(envelope.admits_axes(72.0, 100.0));

        let extreme = envelope.assess_axes(100.0, 34.0).unwrap();
        assert!(
            extreme.uncorrected_image_implied_tilt_radians
                > extreme.maximum_supported_image_tilt_radians
        );
        assert!(extreme.minor_to_major < extreme.minimum_minor_to_major);
        assert!(!envelope.admits_axes(100.0, 34.0));
        assert!(!envelope.admits_axes(34.0, 100.0));
    }

    #[test]
    fn impossible_projected_limbus_cannot_teach_shared_physical_scale() {
        assert_eq!(
            FrontoParallelLimbusRadiusPrior::fronto_parallel_radius_px(100.0, 30.0),
            None
        );
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        assert!(!tracker.observe_strong_ellipse(started, 100.0, 30.0, 1.0));
        assert_eq!(
            tracker.begin_frame(started + Duration::from_millis(10), None),
            None
        );
    }

    #[test]
    fn native_geometry_gate_does_not_choose_a_less_bad_extreme_conic() {
        let rough = OuterSearchEllipse {
            center: (190.0, 128.0),
            major_radius: 100.0,
            minor_radius: 72.0,
            angle: 0.0,
        };
        assert!(outer_geometry_plausible(
            [190.0, 128.0, 100.0, 72.0, PI * 0.5],
            rough,
        ));
        assert!(!outer_geometry_plausible(
            [190.0, 128.0, 145.0, 49.655_172_413_8, PI * 0.5],
            rough,
        ));
    }

    #[test]
    fn outer_iris_budget_hysteresis_ignores_spikes_and_recovers_density() {
        let mut tracker = OuterIrisTracker::default();
        let pressured = OuterIrisDiagnostics {
            ray_overruns: 1,
            ..OuterIrisDiagnostics::default()
        };
        let slow = OUTER_IRIS_SYSTEM_BUDGET + Duration::from_millis(1);

        for _ in 0..(OUTER_IRIS_BUDGET_PRESSURE_FRAMES - 1) {
            tracker.finish_attempt(pressured, slow);
        }
        assert_eq!(tracker.work_stride(), 1);
        assert_eq!(tracker.sample_stride(), 1);

        tracker.finish_attempt(pressured, slow);
        assert_eq!(tracker.work_stride(), 2);
        assert_eq!(tracker.sample_stride(), 2);

        tracker.finish_attempt(OuterIrisDiagnostics::default(), Duration::ZERO);
        tracker.finish_attempt(pressured, slow);
        assert_eq!(tracker.work_stride(), 2);
        assert_eq!(tracker.sample_stride(), 2);

        for _ in 0..OUTER_IRIS_BUDGET_RECOVERY_FRAMES {
            tracker.finish_attempt(OuterIrisDiagnostics::default(), Duration::ZERO);
        }
        assert_eq!(tracker.work_stride(), 1);
        assert_eq!(tracker.sample_stride(), 1);
    }

    #[test]
    fn log_chromatic_material_coordinate_is_exactly_illumination_invariant() {
        let reference = illumination_invariant_log_chroma(310.0, 470.0, 185.0);
        for illumination in [0.08, 0.23, 0.61, 1.0, 2.7] {
            let shadowed = illumination_invariant_log_chroma(
                310.0 * illumination,
                470.0 * illumination,
                185.0 * illumination,
            );
            assert!((shadowed.0 - reference.0).abs() < 1.0e-12);
            assert!((shadowed.1 - reference.1).abs() < 1.0e-12);
        }
    }

    #[test]
    fn local_log_intensity_step_is_exactly_illumination_invariant() {
        let reference = illumination_invariant_log_intensity_step(185.0, 610.0);
        assert!(reference > 1.0);
        for illumination in [0.04, 0.11, 0.37, 1.0, 4.8] {
            let shadowed = illumination_invariant_log_intensity_step(
                185.0 * illumination,
                610.0 * illumination,
            );
            assert!((shadowed - reference).abs() < 1.0e-12);
        }
        assert_eq!(illumination_invariant_log_intensity_step(0.0, 0.0), 0.0);
    }

    #[test]
    fn reflectance_sclera_map_does_not_consume_intensity() {
        let appearance = IrisScleraAppearance {
            iris_log_rg: -0.22,
            iris_log_bg: -0.31,
            sclera_log_rg: 0.11,
            sclera_log_bg: -0.04,
            color_scale: 0.09,
            luma_midpoint: 320.0,
            luma_scale: 80.0,
        };
        let plane = |intensity| NativeLogPlane {
            width: 2,
            height: 2,
            origin_x: 1.5,
            origin_y: 1.5,
            log_rg: vec![0.08, -0.18, 0.09, -0.20],
            log_bg: vec![-0.03, -0.27, -0.02, -0.29],
            reflectance_log_rg: vec![0.08, -0.18, 0.09, -0.20],
            reflectance_log_bg: vec![-0.03, -0.27, -0.02, -0.29],
            intensity: vec![intensity; 4],
            void: vec![0.0; 4],
        };
        let lit = iris_sclera_reflectance_probability_map(&plane(720.0), appearance);
        let shadowed = iris_sclera_reflectance_probability_map(&plane(72.0), appearance);
        assert_eq!(lit, shadowed);

        let mixed_lit = iris_sclera_probability_map(&plane(720.0), appearance);
        let mixed_shadowed = iris_sclera_probability_map(&plane(72.0), appearance);
        assert!(mixed_lit
            .iter()
            .zip(mixed_shadowed)
            .any(|(lit, shadowed)| (lit - shadowed).abs() > 0.10));
    }

    #[test]
    fn reflectance_bonus_is_lid_facing_one_way_shadow_corroboration() {
        let shadowed_sclera = OuterFeatureSample {
            reflectance_step: 0.20,
            reflectance_support: 0.12,
            reflectance_sclera_out: 0.86,
            reflectance_far_sclera: 0.82,
            // The intensity-aware posterior was pulled toward iris by the
            // synthetic lid shadow.
            sclera_out: 0.24,
            far_sclera: 0.28,
            ..OuterFeatureSample::default()
        };
        let upper = shadowed_sclera.shadow_reflectance_score(-PI * 0.5, 0.30, 2.0, 1.5, 1.5);
        assert!(upper > 0.12 && upper <= 0.30, "upper={upper}");
        assert_eq!(
            shadowed_sclera.shadow_reflectance_score(PI * 0.5, 1.0, 2.0, 1.5, 1.5),
            0.0,
            "the exposed lower limbus must keep its established grader"
        );

        let normally_lit = OuterFeatureSample {
            sclera_out: 0.82,
            far_sclera: 0.80,
            ..shadowed_sclera
        };
        assert_eq!(
            normally_lit.shadow_reflectance_score(-PI * 0.5, 0.30, 2.0, 1.5, 1.5),
            0.0,
            "agreement with ordinary evidence needs no shadow bonus"
        );
        let wrong_polarity = OuterFeatureSample {
            reflectance_step: -0.20,
            ..shadowed_sclera
        };
        assert_eq!(
            wrong_polarity.shadow_reflectance_score(-PI * 0.5, 0.30, 2.0, 1.5, 1.5),
            0.0,
            "an iris-facing or arbitrary color edge cannot be promoted"
        );
    }

    #[test]
    fn native_raw_log_chromaticity_survives_a_deep_achromatic_shadow() {
        const SIDE: usize = 96;
        const BLACK: u16 = 64;
        let mut raw = vec![BLACK; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                if x < 4 && y < 4 {
                    continue;
                }
                let radius = (x as f64 - 48.0).hypot(y as f64 - 48.0);
                let channels = if radius <= 28.0 {
                    [300u16, 430, 430, 235]
                } else {
                    [670u16, 625, 625, 565]
                };
                let channel = match (x % 4 < 2, y % 4 < 2) {
                    (true, true) => 0,
                    (false, true) => 1,
                    (true, false) => 2,
                    (false, false) => 3,
                };
                raw[y * SIDE + x] = BLACK + channels[channel];
            }
        }
        let shadowed = raw
            .iter()
            .map(|value| {
                (f64::from(BLACK) + f64::from(value.saturating_sub(BLACK)) * 0.22).round() as u16
            })
            .collect::<Vec<_>>();
        let coarse = BorderFocus {
            center: (48.0, 48.0),
            radius: 28.0,
            ..BorderFocus::default()
        };
        let lit = native_log_plane(&raw, SIDE, SIDE, 0, 0, &coarse).unwrap();
        let dark = native_log_plane(&shadowed, SIDE, SIDE, 0, 0, &coarse).unwrap();
        let maximum_difference = lit
            .reflectance_log_rg
            .iter()
            .zip(&lit.reflectance_log_bg)
            .zip(dark.reflectance_log_rg.iter().zip(&dark.reflectance_log_bg))
            .map(|((lit_rg, lit_bg), (dark_rg, dark_bg))| {
                (lit_rg - dark_rg).abs().max((lit_bg - dark_bg).abs())
            })
            .fold(0.0, f64::max);
        assert!(
            maximum_difference < 0.012,
            "difference={maximum_difference}"
        );
    }

    #[test]
    fn inner_pupil_cue_weights_shift_from_detail_to_robust_priors() {
        let sharp = InnerIrisEvidenceCondition::default().cue_weights();
        let soft = InnerIrisEvidenceCondition::new(0.0).cue_weights();

        assert!((sharp.0 - 0.58).abs() < 1.0e-12);
        assert!((sharp.1 - 0.22).abs() < 1.0e-12);
        assert!((sharp.2 - 0.20).abs() < 1.0e-12);
        assert!((sharp.3 - 0.16).abs() < 1.0e-12);
        assert!((sharp.4 - 0.10).abs() < 1.0e-12);
        assert!(soft.0 < sharp.0, "soft={soft:?} sharp={sharp:?}");
        assert!(soft.1 < sharp.1, "soft={soft:?} sharp={sharp:?}");
        assert!(soft.2 > sharp.2, "soft={soft:?} sharp={sharp:?}");
        assert!(soft.3 > sharp.3, "soft={soft:?} sharp={sharp:?}");
        assert!(soft.4 > sharp.4, "soft={soft:?} sharp={sharp:?}");
    }

    #[test]
    fn labeled_frame_uses_a_five_percent_fronto_parallel_radius_support() {
        let prior = FrontoParallelLimbusRadiusPrior::from_fractional_support(
            61.93933,
            0.05,
            FrontoParallelLimbusRadiusPriorSource::FixedReference,
        )
        .expect("valid labeled-frame radius prior");
        assert!((prior.minimum_px - 58.8423635).abs() < 1.0e-7);
        assert!((prior.maximum_px - 65.0362965).abs() < 1.0e-7);
        assert!(prior.admits_ellipse(61.93933, 55.90055));
        assert!(!prior.admits_ellipse(83.8, 69.7));
        assert!(!prior.admits_ellipse(55.0, 51.0));
    }

    #[test]
    fn rectified_radius_tracker_freezes_each_frame_and_updates_only_after_admission() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        assert_eq!(tracker.begin_frame(started, None), None);
        assert!(tracker.observe_strong_ellipse(started, 62.0, 48.0, 0.90));
        assert_eq!(
            tracker.begin_frame(started + Duration::from_millis(10), None),
            None,
            "one completed conic must not own physical scale"
        );
        assert!(tracker.observe_strong_ellipse(
            started + Duration::from_millis(10),
            62.0,
            48.0,
            0.90,
        ));
        assert_eq!(
            tracker.begin_frame(started + Duration::from_millis(20), None),
            None,
            "two completed conics must not own physical scale"
        );
        assert!(tracker.observe_strong_ellipse(
            started + Duration::from_millis(20),
            62.0,
            48.0,
            0.90,
        ));

        let temporal = tracker
            .begin_frame(started + Duration::from_millis(120), None)
            .expect("temporal prior after three consistent admitted roads");
        assert_eq!(
            temporal.source,
            FrontoParallelLimbusRadiusPriorSource::TemporalRobustMedian
        );
        assert!(temporal.admits_radius(62.0));
        let temporal_half_width =
            (temporal.maximum_px - temporal.minimum_px) / (2.0 * temporal.estimate_px);
        assert!(
            (temporal_half_width - 0.039).abs() < 1.0e-9,
            "uncorroborated one-frame support widened too far: {temporal:?}"
        );
        assert!(!temporal.admits_radius(59.0));
        assert!(!temporal.admits_radius(83.8));
        assert!(!tracker.observe_strong_ellipse(
            started + Duration::from_millis(120),
            83.8,
            69.7,
            1.0,
        ));
        assert_eq!(tracker.active_frame_prior(), Some(temporal));

        let predicted = tracker
            .begin_frame(
                started + Duration::from_millis(200),
                Some(FrontoParallelLimbusScalePrediction::coarse_semantic_pose(
                    1.20, 0.08,
                )),
            )
            .expect("coarse-pose transported prior");
        assert_eq!(
            predicted.source,
            FrontoParallelLimbusRadiusPriorSource::CoarseSemanticPose
        );
        assert!((predicted.estimate_px - 74.4).abs() < 1.0e-9);
        assert!(predicted.admits_radius(74.4));
        assert!(tracker.observe_strong_ellipse(
            started + Duration::from_millis(200),
            74.4,
            54.0,
            0.9,
        ));

        let fine = tracker
            .begin_frame(
                started + Duration::from_millis(300),
                Some(FrontoParallelLimbusScalePrediction::fine_visual_odometry(
                    1.01, 0.03,
                )),
            )
            .expect("fine visual-odometry transported prior");
        assert_eq!(
            fine.source,
            FrontoParallelLimbusRadiusPriorSource::FineVisualOdometry
        );
        assert!((fine.estimate_px - 75.144).abs() < 1.0e-9, "{fine:?}");

        // Candidate/refinement stages share one frozen frame support. Asking
        // for it again with the same timestamp must not apply 1.01 twice.
        let repeated = tracker
            .begin_frame(
                started + Duration::from_millis(300),
                Some(FrontoParallelLimbusScalePrediction::fine_visual_odometry(
                    1.01, 0.03,
                )),
            )
            .unwrap();
        assert_eq!(repeated, fine);
        let carried = tracker
            .begin_frame(started + Duration::from_millis(400), None)
            .unwrap();
        assert!((carried.estimate_px - 75.144).abs() < 1.0e-9, "{carried:?}");

        // A long detector miss widens only at the explicit elapsed-time rate;
        // it neither discards physical-size history nor accepts an arbitrary
        // much smaller lid/reflection curve.
        let after_gap = tracker
            .begin_frame(started + Duration::from_millis(1_500), None)
            .expect("size posterior survives a tracking gap");
        let gap_half_width =
            (after_gap.maximum_px - after_gap.minimum_px) / (2.0 * after_gap.estimate_px);
        assert!((gap_half_width - 0.087).abs() < 1.0e-9, "{after_gap:?}");
        assert!(after_gap.admits_radius(75.144));
        assert!(!after_gap.admits_radius(60.0), "{after_gap:?}");
    }

    #[test]
    fn rectified_radius_long_miss_reopens_physically_reachable_scale_support() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for offset_ms in [0, 10, 20] {
            let now = started + Duration::from_millis(offset_ms);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse_for_active_frame(now, 50.0, 40.0, 0.95,));
        }

        let adjacent = tracker
            .begin_frame(started + Duration::from_millis(120), None)
            .expect("three strong roads establish a tight adjacent-frame prior");
        assert!(!adjacent.admits_radius(100.0), "{adjacent:?}");

        let long_gap = tracker
            .begin_frame(started + Duration::from_secs(40), None)
            .expect("physical size remains an anchor after a long miss");
        assert!(
            long_gap.admits_radius(100.0),
            "a physically reachable 2x move must become searchable after 40 seconds: {long_gap:?}"
        );
        assert!(long_gap.admits_kinematically_supported_radius(50.0));
        assert!(
            !long_gap.admits_kinematically_supported_radius(100.0),
            "stale search support alone must not publish a 2x scale jump: {long_gap:?}"
        );
        assert!(!tracker.observe_strong_ellipse_for_active_frame(
            started + Duration::from_secs(40),
            100.0,
            80.0,
            1.0,
        ));
        assert!(long_gap.maximum_px <= 125.0 + 1.0e-9, "{long_gap:?}");
        assert!(!long_gap.admits_radius(130.0), "{long_gap:?}");

        let transported = tracker
            .begin_frame(
                started + Duration::from_millis(40_100),
                Some(FrontoParallelLimbusScalePrediction::coarse_semantic_pose(
                    2.0, 0.08,
                )),
            )
            .expect("independent whole-ROI scale should transport the publication corridor");
        assert!((transported.estimate_px - 100.0).abs() < 1.0e-9);
        assert!(transported.admits_kinematically_supported_radius(100.0));
    }

    #[test]
    fn rectified_radius_latest_strong_fit_prunes_an_unsupported_adjacent_branch_flip() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for offset_ms in [0, 10, 20] {
            let now = started + Duration::from_millis(offset_ms);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse_for_active_frame(now, 133.0, 108.0, 0.95,));
        }

        // A road whose de-affined area changes by less than two percent may
        // become the next publication reference without moving the
        // median-centered search.
        let first_at = started + Duration::from_millis(30);
        let first_prior = tracker.begin_frame(first_at, None).unwrap();
        let first_radius = 134.2;
        assert!(first_prior.admits_radius(first_radius));
        assert!(tracker.observe_strong_ellipse_for_active_frame(
            first_at,
            first_radius,
            first_radius * 0.824,
            0.95,
        ));

        // A substantially smaller road about 98 ms later remains inside the
        // median-centered candidate envelope. The final shared publication
        // gate rejects its unsupported equal-area discontinuity without
        // requiring every segmenter to duplicate the temporal rule.
        let next_at = started + Duration::from_millis(128);
        let next_prior = tracker.begin_frame(next_at, None).unwrap();
        assert!(
            (next_prior.estimate_px - 133.0).abs() < 1.0e-9,
            "{next_prior:?}"
        );
        assert!(next_prior.admits_radius(129.769_611_607_013_5));
        assert!(!tracker.observe_strong_ellipse_for_active_frame(
            next_at,
            129.769_611_607_013_5,
            106.987_691_052_812_86,
            0.95,
        ));

        // The same apparent-size change is legal once full-resolution visual
        // odometry explicitly transports the complete posterior, including
        // both its latest and non-ratcheting robust centers.
        let scale_ratio = 129.769_611_607_013_5 / first_radius;
        let transported_at = started + Duration::from_millis(226);
        let transported = tracker
            .begin_frame(
                transported_at,
                Some(FrontoParallelLimbusScalePrediction::fine_visual_odometry(
                    scale_ratio,
                    0.02,
                )),
            )
            .unwrap();
        assert!(transported.admits_radius(129.769_611_607_013_5));
        assert!(tracker.observe_strong_ellipse_for_active_frame(
            transported_at,
            129.769_611_607_013_5,
            106.987_691_052_812_86,
            0.95,
        ));
    }

    #[test]
    fn rectified_radius_latest_fit_expires_before_long_gap_reacquisition() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for offset_ms in [0, 10, 20] {
            let now = started + Duration::from_millis(offset_ms);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse_for_active_frame(now, 133.0, 108.0, 0.95,));
        }

        let latest_at = started + Duration::from_millis(30);
        let latest_radius = 134.2;
        assert!(tracker
            .begin_frame(latest_at, None)
            .unwrap()
            .admits_radius(latest_radius));
        assert!(tracker.observe_strong_ellipse_for_active_frame(
            latest_at,
            latest_radius,
            latest_radius * 0.824,
            0.95,
        ));

        // After a missing-frame gap the latest raw fit is no longer an
        // adjacent-frame authority. The robust posterior still admits this
        // central road, so reacquisition can resume without pretending that
        // detector silence was independent visual scale evidence.
        let reacquired_at =
            latest_at + LIMBUS_LATEST_STRONG_CONTINUITY_HORIZON + Duration::from_millis(1);
        let reacquisition_prior = tracker.begin_frame(reacquired_at, None).unwrap();
        // This road remains central to the robust, non-ratcheting posterior,
        // but is more than one reconstructed-area step from the last raw fit.
        let reacquired_radius = 132.0;
        assert!(reacquisition_prior.admits_kinematically_supported_radius(reacquired_radius));
        assert!(tracker.observe_strong_ellipse_for_active_frame(
            reacquired_at,
            reacquired_radius,
            reacquired_radius * 0.824,
            0.95,
        ));
    }

    #[test]
    fn shared_rectified_radius_tracker_rejects_late_async_frame_mutation() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        assert_eq!(tracker.begin_frame(started, None), None);
        assert!(tracker.observe_strong_ellipse_for_active_frame(started, 62.0, 48.0, 1.0));
        let second_at = started + Duration::from_millis(10);
        assert_eq!(tracker.begin_frame(second_at, None), None);
        assert!(tracker.observe_strong_ellipse_for_active_frame(second_at, 62.0, 48.0, 1.0));
        let third_at = started + Duration::from_millis(20);
        assert_eq!(tracker.begin_frame(third_at, None), None);
        assert!(tracker.observe_strong_ellipse_for_active_frame(third_at, 62.0, 48.0, 1.0));

        let current_at = started + Duration::from_millis(200);
        let current = tracker.begin_frame(current_at, None).unwrap();
        let stale_at = started + Duration::from_millis(100);
        assert_eq!(tracker.begin_frame(stale_at, None), Some(current));
        assert!(!tracker.observe_strong_ellipse_for_active_frame(stale_at, 61.0, 47.0, 1.0));
        assert_eq!(tracker.active_frame_prior(), Some(current));

        assert!(tracker.observe_strong_ellipse_for_active_frame(current_at, 62.0, 48.0, 1.0));
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

    #[test]
    fn operator_limbus_limits_override_auto_and_remain_adjustable_while_r_frozen() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        let manual = tracker
            .begin_frame_controlled(started, None, true, Some((55.0, 65.0)))
            .expect("manual limits can cold-start a hard support");
        assert_eq!(
            manual.source,
            FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits
        );
        assert_eq!((manual.minimum_px, manual.maximum_px), (55.0, 65.0));
        assert!(!tracker.observe_strong_ellipse(started, 70.0, 50.0, 1.0));
        assert!(tracker.observe_strong_ellipse(started, 60.0, 50.0, 1.0));

        let automatic = tracker
            .begin_frame_controlled(started + Duration::from_millis(10), None, true, None)
            .expect("admitted manual observation trains the automatic posterior");
        assert_eq!(
            automatic.source,
            FrontoParallelLimbusRadiusPriorSource::TemporalRobustMedian
        );

        let frozen = tracker
            .begin_frame_controlled(started + Duration::from_millis(20), None, false, None)
            .expect("R freeze retains the current automatic support");
        assert_eq!(frozen, automatic);
        let adjusted_while_frozen = tracker
            .begin_frame_controlled(
                started + Duration::from_millis(30),
                None,
                false,
                Some((57.0, 63.0)),
            )
            .expect("explicit controls remain live under R freeze");
        assert_eq!(
            adjusted_while_frozen.source,
            FrontoParallelLimbusRadiusPriorSource::OperatorHardLimits
        );
        assert_eq!(
            (
                adjusted_while_frozen.minimum_px,
                adjusted_while_frozen.maximum_px,
            ),
            (57.0, 63.0)
        );
    }

    #[test]
    fn rectified_radius_scale_motion_cannot_create_anatomy_from_no_prior() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        assert_eq!(
            tracker.begin_frame(
                started,
                Some(FrontoParallelLimbusScalePrediction::coarse_semantic_pose(
                    1.20, 0.10,
                )),
            ),
            None,
        );
        assert_eq!(tracker.active_frame_prior(), None);
    }

    #[test]
    fn rectified_radius_session_reset_drops_dynamic_pixels_but_keeps_fixed_reference() {
        let started = Instant::now();
        let mut dynamic = FrontoParallelLimbusRadiusTracker::default();
        assert!(dynamic.observe_strong_ellipse(started, 62.0, 48.0, 1.0));
        assert!(dynamic.observe_strong_ellipse(
            started + Duration::from_millis(5),
            62.0,
            48.0,
            1.0,
        ));
        assert!(dynamic.observe_strong_ellipse(
            started + Duration::from_millis(10),
            62.0,
            48.0,
            1.0,
        ));
        assert!(dynamic
            .begin_frame(started + Duration::from_millis(20), None)
            .is_some());
        dynamic.reset_dynamic_observations();
        assert_eq!(
            dynamic.begin_frame(started + Duration::from_millis(30), None),
            None,
        );

        let mut fixed = FrontoParallelLimbusRadiusTracker::with_fixed_reference(62.0, 0.05)
            .expect("valid fixed reference");
        fixed.reset_dynamic_observations();
        let prior = fixed
            .begin_frame(started + Duration::from_millis(20), None)
            .expect("fixed reference survives a Driving session boundary");
        assert_eq!(
            prior.source,
            FrontoParallelLimbusRadiusPriorSource::FixedReference
        );
        assert!((prior.estimate_px - 62.0).abs() < 1.0e-9);
    }

    #[test]
    fn rectified_radius_tracker_uses_a_robust_strong_only_median() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        assert!(tracker.observe_strong_ellipse(started, 62.0, 48.0, 1.0));
        for (index, radius) in [63.0, 64.0].into_iter().enumerate() {
            tracker.begin_frame(started + Duration::from_millis(10 + index as u64), None);
            assert!(tracker.observe_strong_ellipse(
                started + Duration::from_millis(10 + index as u64),
                radius,
                48.0,
                1.0,
            ));
        }
        let prior = tracker
            .begin_frame(started + Duration::from_millis(30), None)
            .unwrap();
        // Cold start uses the median of three mutually consistent strong
        // roads instead of triplicating whichever curve happened to arrive
        // first.
        assert!((prior.estimate_px - 63.0).abs() < 1.0e-9, "{prior:?}");
    }

    #[test]
    fn rectified_radius_cold_consensus_not_latest_vote_owns_first_publication() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        // Reproduce the right-capture startup shape: all three roads are
        // coherent enough to establish identity, but the newest vote lies
        // more than 5% above both the robust center and the next valid road.
        for (index, radius) in [81.072_565, 83.0, 88.568_164].into_iter().enumerate() {
            let now = started + Duration::from_millis(index as u64 * 10);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse_for_active_frame(
                now,
                radius,
                radius * 0.82,
                0.95,
            ));
        }

        let publish_at = started + Duration::from_millis(30);
        let prior = tracker
            .begin_frame(publish_at, None)
            .expect("three coherent cold roads establish a robust radius");
        assert!((prior.estimate_px - 83.0).abs() < 1.0e-9, "{prior:?}");
        // Stay inside the adjacent-publication invariant as measured on the
        // reconstructed circular area: (82.25 / 83.0)^2 differs by less than
        // two percent.  The newest cold proposal at 88.568 px is deliberately
        // much farther away, so this still proves that proposal-only votes do
        // not own the first publication gate.
        let next_radius = 82.25;
        assert!(prior.admits_radius(next_radius), "{prior:?}");
        assert!(
            tracker.observe_strong_ellipse_for_active_frame(
                publish_at,
                next_radius,
                next_radius * 0.82,
                0.95,
            ),
            "the first publication must be compared with the cold consensus, not its newest proposal-only vote",
        );
    }

    #[test]
    fn rectified_radius_search_center_resists_untransported_measurement_ratchets() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for (index, radius) in [62.0, 63.0, 64.0].into_iter().enumerate() {
            let now = started + Duration::from_millis(index as u64 * 10);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse(now, radius, radius * 0.80, 1.0));
        }
        let fourth_at = started + Duration::from_millis(30);
        let fourth_prior = tracker.begin_frame(fourth_at, None).unwrap();
        assert!(fourth_prior.admits_radius(65.0), "{fourth_prior:?}");
        assert!(
            !tracker.observe_strong_ellipse(fourth_at, 65.0, 52.0, 1.0),
            "the wider search envelope may find 65 px, but an untransported 6.5% area jump must not publish",
        );

        let next = tracker
            .begin_frame(started + Duration::from_millis(40), None)
            .unwrap();
        // The rejected 65 px curve remains searchable but cannot train or
        // ratchet the 63 px physical authority. A valid whole-ROI scale
        // transport is the only path for an adjacent jump this large.
        assert!((next.estimate_px - 63.0).abs() < 1.0e-9, "{next:?}");
        assert!(next.admits_radius(65.0));
        assert!(!next.admits_kinematically_supported_radius(65.0));
    }

    #[test]
    fn rectified_radius_cold_start_rejects_incoherent_size_aliases() {
        let started = Instant::now();
        let mut tracker = FrontoParallelLimbusRadiusTracker::default();
        for (index, radius) in [87.0, 53.0, 83.0].into_iter().enumerate() {
            let now = started + Duration::from_millis(index as u64 * 20);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse(now, radius, radius * 0.82, 0.95));
        }
        assert_eq!(
            tracker.begin_frame(started + Duration::from_millis(80), None),
            None,
            "an aperture/iris/aperture sequence must not establish one physical scale"
        );

        // Once the old aliases age out, three mutually consistent roads may
        // establish the de-affined circular-radius posterior.
        for (index, radius) in [52.0, 53.0, 51.0].into_iter().enumerate() {
            let now = started + Duration::from_millis(1_100 + index as u64 * 20);
            assert_eq!(tracker.begin_frame(now, None), None);
            assert!(tracker.observe_strong_ellipse(now, radius, radius * 0.82, 0.95));
        }
        let prior = tracker
            .begin_frame(started + Duration::from_millis(1_180), None)
            .expect("three coherent roads should establish physical scale");
        assert!((prior.estimate_px - 52.0).abs() < 1.0e-9, "{prior:?}");
    }

    #[test]
    fn lateral_order_is_partitioned_in_camera_coordinates() {
        let mut strip = LimbusPerimeterStrip::default();
        for index in 0..6 {
            strip.samples.push(LimbusPerimeterDriveSample {
                // Deliberately claim the opposite local ellipse side. A
                // phase-based implementation would swap these two groups.
                phase: 0.05 * index as f64,
                outward_normal: (-1.0, 0.0),
                transition_score: 0.20,
                opposite_sclera_score: 0.20,
                ..LimbusPerimeterDriveSample::default()
            });
            strip.samples.push(LimbusPerimeterDriveSample {
                phase: PI + 0.05 * index as f64,
                outward_normal: (1.0, 0.0),
                transition_score: 0.0,
                opposite_sclera_score: 0.0,
                ..LimbusPerimeterDriveSample::default()
            });
        }
        let camera_left = limbus_lateral_order_evidence(&strip, false);
        let camera_right = limbus_lateral_order_evidence(&strip, true);
        assert_eq!(camera_left.sample_count, 6);
        assert_eq!(camera_left.ordered_score, 1.0);
        assert_eq!(camera_right.sample_count, 6);
        assert_eq!(camera_right.ordered_score, 0.0);
    }

    #[derive(Clone, Copy)]
    struct DrivingRawFixture {
        pair: usize,
        center: (f64, f64),
        radius: f64,
        axis_ratio: f64,
        axis_angle: f64,
    }

    const DRIVING_RAW_FIXTURES: [DrivingRawFixture; 10] = [
        DrivingRawFixture {
            pair: 0,
            center: (134.208535, 163.026570),
            radius: 63.241927,
            axis_ratio: 1.301448,
            axis_angle: 0.163038,
        },
        DrivingRawFixture {
            pair: 2,
            center: (122.896104, 164.123377),
            radius: 63.087663,
            axis_ratio: 1.336285,
            axis_angle: 0.157670,
        },
        DrivingRawFixture {
            pair: 4,
            center: (122.572100, 154.791536),
            radius: 56.761833,
            axis_ratio: 1.392734,
            axis_angle: 0.060169,
        },
        DrivingRawFixture {
            pair: 6,
            center: (118.470332, 160.718150),
            radius: 60.734259,
            axis_ratio: 1.202800,
            axis_angle: 0.546040,
        },
        DrivingRawFixture {
            pair: 8,
            center: (115.439189, 153.756757),
            radius: 65.338324,
            axis_ratio: 1.175964,
            axis_angle: 0.199916,
        },
        DrivingRawFixture {
            pair: 10,
            center: (150.900324, 155.094814),
            radius: 64.118566,
            axis_ratio: 1.384891,
            axis_angle: 0.197324,
        },
        DrivingRawFixture {
            pair: 12,
            center: (248.509464, 146.427445),
            radius: 60.640132,
            axis_ratio: 1.269435,
            axis_angle: 0.449774,
        },
        DrivingRawFixture {
            pair: 14,
            center: (144.347858, 150.380355),
            radius: 61.928988,
            axis_ratio: 1.394972,
            axis_angle: 0.121195,
        },
        DrivingRawFixture {
            pair: 16,
            center: (204.143799, 139.558047),
            radius: 67.250181,
            axis_ratio: 1.538323,
            axis_angle: 0.025870,
        },
        DrivingRawFixture {
            pair: 18,
            center: (162.901752, 128.473717),
            radius: 71.377784,
            axis_ratio: 1.586367,
            axis_angle: -0.009487,
        },
    ];

    // Independently reviewed on the CFA-neutral RAW renders. These deliberately
    // have a broad tolerance: the regression is for the dark pupil basin, not
    // sub-pixel agreement with one particular rim fit.
    const DRIVING_EXPECTED_PUPILS: [(f64, f64); 10] = [
        (130.0, 189.0),
        (103.0, 174.0),
        (102.0, 134.0),
        (109.0, 142.0),
        (122.0, 172.0),
        (148.0, 169.0),
        (263.0, 118.0),
        (142.0, 178.0),
        (189.0, 152.0),
        (158.0, 131.0),
    ];

    // Robust ellipse-normal fits from the five-method RAW ablation. They are
    // kept separate from the rough anatomy sidecar so the perimeter test starts
    // from a limbus, then tests only the requested local driving behavior.
    const DRIVING_REVIEW_LIMBUS: [[f64; 5]; 10] = [
        [136.684845, 161.166946, 64.212143, 58.993172, -2.678605],
        [127.073410, 162.751678, 68.347038, 59.798210, -2.486625],
        [120.802162, 156.344070, 73.416222, 56.726463, -2.927695],
        [123.919388, 162.938797, 62.755074, 53.452354, -1.949538],
        [107.968025, 152.103210, 82.623291, 67.501976, -2.658873],
        [155.514999, 154.804153, 65.862488, 56.158932, -2.064117],
        [252.085312, 146.244949, 66.757652, 54.529606, -2.197375],
        [147.805740, 148.994324, 67.914551, 60.101936, -2.476918],
        [178.263611, 136.274414, 96.203461, 73.408890, -0.008566],
        [140.268066, 131.377121, 95.341309, 69.712936, -0.070251],
    ];

    fn unpack_test_raw10(payload: &[u8]) -> Vec<u16> {
        assert_eq!(payload.len(), WIDTH * HEIGHT * 5 / 4);
        let mut raw = Vec::with_capacity(WIDTH * HEIGHT);
        for group in payload.chunks_exact(5) {
            let word = u64::from(group[0])
                | (u64::from(group[1]) << 8)
                | (u64::from(group[2]) << 16)
                | (u64::from(group[3]) << 24)
                | (u64::from(group[4]) << 32);
            for lane in 0..4 {
                raw.push(((word >> (lane * 10)) & 0x3ff) as u16);
            }
        }
        raw
    }

    fn driving_fixture_raw(pair: usize) -> Vec<u16> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../outputs/paired-eye-reverse-stereo-20260809T113752Z/raw-pairs")
            .join(format!("pair-{pair:02}.subject-right.raw10"));
        let payload = std::fs::read(&path).unwrap_or_else(|error| {
            panic!("read lossless Driving fixture {}: {error}", path.display())
        });
        unpack_test_raw10(&payload)
    }

    #[test]
    fn reviewed_right_clipped_raw_recovers_a_censored_limbus_instead_of_the_left_alias() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/partial-frame-right-clipped.raw10");
        let payload = std::fs::read(&path).unwrap_or_else(|error| {
            panic!(
                "read reviewed partial-frame RAW {}: {error}",
                path.display()
            )
        });
        let raw = unpack_test_raw10(&payload);
        let focus = score_stream_eye(&raw, WIDTH, HEIGHT);
        let partial = focus
            .roi_truncated_limbus
            .expect("reviewed frame must retain its directly visible censored arc");

        assert!(
            !focus.eye_basin_valid,
            "partial arc is not complete anatomy"
        );
        assert_eq!(focus.score, 0.0, "partial arc is not autofocus identity");
        assert!(
            partial.censored_edges & ROI_TRUNCATED_RIGHT != 0,
            "partial={partial:?}"
        );
        assert!(
            (270.0..=335.0).contains(&partial.center.0),
            "partial={partial:?}"
        );
        assert!(
            (125.0..=195.0).contains(&partial.center.1),
            "partial={partial:?}"
        );
        assert!(partial.reframe_delta_px.0 >= 80.0, "partial={partial:?}");
        assert!(partial.visible_arc_fraction >= 0.68, "partial={partial:?}");
        assert!(
            partial.supported_probe_fraction >= 0.70,
            "partial={partial:?}"
        );
        assert!(partial.confidence >= 0.78, "partial={partial:?}");
    }

    #[test]
    fn synthetic_roi_censoring_is_recovered_on_all_four_frame_edges() {
        let cases = [
            ((330.0, 148.0), ROI_TRUNCATED_RIGHT, (1.0, 0.0)),
            ((54.0, 148.0), ROI_TRUNCATED_LEFT, (-1.0, 0.0)),
            ((194.0, 210.0), ROI_TRUNCATED_BOTTOM, (0.0, 1.0)),
            ((194.0, 24.0), ROI_TRUNCATED_TOP, (0.0, -1.0)),
        ];
        for (expected_center, edge, direction) in cases {
            let focus = score_stream_eye(
                &synthetic_eye(expected_center.0, expected_center.1, 2.0),
                WIDTH,
                HEIGHT,
            );
            let partial = focus.roi_truncated_limbus.unwrap_or_else(|| {
                panic!("missing partial observation at {expected_center:?}: {focus:?}")
            });
            assert!(partial.censored_edges & edge != 0, "partial={partial:?}");
            assert!(
                (partial.center.0 - expected_center.0).abs() <= 32.0
                    && (partial.center.1 - expected_center.1).abs() <= 32.0,
                "expected={expected_center:?} partial={partial:?}",
            );
            assert!(
                partial.reframe_delta_px.0 * direction.0 + partial.reframe_delta_px.1 * direction.1
                    > 32.0,
                "partial={partial:?}",
            );
        }
    }

    #[test]
    fn fully_visible_limbus_near_an_roi_edge_is_not_mislabeled_as_truncated() {
        // The outer sclera probe band extends beyond the right edge, but the
        // limbus itself ends on the final valid pixel. That is reduced support
        // context, not a censored conic, and must never trigger an ROI move or
        // suppress an independently complete native fit.
        let focus = score_stream_eye(
            &synthetic_eye_with_radius(310.0, 148.0, 2.0, 68.0),
            WIDTH,
            HEIGHT,
        );
        assert!(focus.center.0 >= 285.0, "focus={focus:?}");
        assert!(focus.radius >= 55.0, "focus={focus:?}");
        assert!(focus.roi_truncated_limbus.is_none(), "focus={focus:?}");
    }

    fn synthetic_eye_with_radius(
        center_x: f64,
        center_y: f64,
        transition: f64,
        iris_radius: f64,
    ) -> Vec<u16> {
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let radius = (dx * dx + dy * dy).sqrt();
                let iris = 155.0 + radius * 1.1;
                let sclera = 790.0 + ((x * 17 + y * 11) % 31) as f64;
                let blend = ((radius - iris_radius) / transition + 0.5).clamp(0.0, 1.0);
                raw[y * WIDTH + x] = (iris * (1.0 - blend) + sclera * blend)
                    .round()
                    .clamp(0.0, 1023.0) as u16;
            }
        }
        raw
    }

    fn synthetic_eye(center_x: f64, center_y: f64, transition: f64) -> Vec<u16> {
        synthetic_eye_with_radius(center_x, center_y, transition, 70.0)
    }

    fn analog_step_context(boundary_x: f64, reverse_polarity: bool) -> (Vec<u16>, OuterRayContext) {
        const PROFILE_WIDTH: usize = 176;
        const PROFILE_HEIGHT: usize = 96;
        let mut raw = vec![0u16; PROFILE_WIDTH * PROFILE_HEIGHT];
        for y in 0..PROFILE_HEIGHT {
            for x in 0..PROFILE_WIDTH {
                let blend = ((x as f64 - boundary_x) / 2.2 + 0.5).clamp(0.0, 1.0);
                let (inside, outside) = if reverse_polarity {
                    (680.0, 185.0)
                } else {
                    (185.0, 680.0)
                };
                // Deliberately strong multiplicative left-to-right lighting.
                // The analog profile must localize the material transition,
                // not drift toward this smooth slope.
                let illumination = (0.52 * (x as f64 - 88.0) / 44.0).exp();
                let texture = ((x * 13 + y * 7) % 9) as f64 - 4.0;
                raw[y * PROFILE_WIDTH + x] =
                    ((inside * (1.0 - blend) + outside * blend) * illumination + texture)
                        .round()
                        .clamp(0.0, 1023.0) as u16;
            }
        }
        let luma = Arc::new(BoxLuma5::new(&raw, PROFILE_WIDTH, PROFILE_HEIGHT));
        let search = OuterSearchEllipse {
            center: (88.0, 48.0),
            major_radius: 40.0,
            minor_radius: 34.0,
            angle: 0.0,
        };
        let context = OuterRayContext {
            luma: Arc::clone(&luma),
            native: None,
            sclera_probability: None,
            reflectance_sclera_probability: None,
            material_illumination: Some(MaterialIlluminationModel {
                center: (88.0, 48.0),
                radius: 44.0,
                // The constant is irrelevant to a derivative. The x slope
                // exactly describes the synthetic multiplicative light.
                coefficients: [0.0, 0.52, 0.0, 0.0, 0.0, 0.0],
                sample_count: 180,
                iris_sample_count: 72,
                sclera_sample_count: 72,
                lateral_sclera_balance: 0.92,
                residual_median: 0.035,
                inlier_fraction: 0.91,
                light_span: 1.04,
            }),
            upper_eyelid: Arc::new(Vec::new()),
            lower_eyelid: Arc::new(Vec::new()),
            luma_gate: estimate_luma_transition_gate(luma.as_ref()),
            width: PROFILE_WIDTH,
            height: PROFILE_HEIGHT,
            search,
            rough_search: search,
            scale_range: (0.66, 1.50),
        };
        (raw, context)
    }

    #[test]
    fn analog_edge_force_reports_direction_power_and_certainty_after_light_correction() {
        let (_raw, context) = analog_step_context(91.25, false);
        let from_inside = sample_analog_outer_edge_force(&context, (88.0, 48.0), (1.0, 0.0))
            .expect("positive iris-to-sclera edge");
        let from_outside = sample_analog_outer_edge_force(&context, (94.0, 48.0), (1.0, 0.0))
            .expect("same edge viewed from its outer side");
        assert!(
            from_inside.edge_offset_px > 1.0 && from_inside.edge_offset_px <= 4.0,
            "force={from_inside:?}"
        );
        assert!(
            from_outside.edge_offset_px < -0.8 && from_outside.edge_offset_px >= -4.0,
            "force={from_outside:?}"
        );
        assert!(from_inside.power > 0.60, "force={from_inside:?}");
        assert!(from_inside.certainty > 0.35, "force={from_inside:?}");

        let (_raw, reversed) = analog_step_context(91.25, true);
        let reverse = sample_analog_outer_edge_force(&reversed, (91.0, 48.0), (1.0, 0.0));
        assert!(
            reverse.is_none_or(|force| force.power < 0.35 || force.certainty < 0.20),
            "reverse-polarity edge gained outward authority: {reverse:?}"
        );
    }

    #[test]
    fn analog_fit_weights_prevent_a_weak_contact_from_steering_the_conic() {
        let seed = [100.0, 90.0, 50.0];
        let mut points = (0..24)
            .map(|index| {
                let angle = 2.0 * PI * index as f64 / 24.0;
                OuterIrisPoint {
                    x: seed[0] + seed[2] * angle.cos(),
                    y: seed[1] + seed[2] * angle.sin(),
                    contrast: 100.0,
                }
            })
            .collect::<Vec<_>>();
        points[0].x += 16.0;
        let unweighted = fit_outer_ellipse(&points, seed);
        let mut weights = vec![1.0; points.len()];
        weights[0] = 0.02;
        let weighted = fit_outer_ellipse_with_weights(&points, Some(&weights), seed);
        let parameter_error = |ellipse: [f64; 5]| {
            (ellipse[0] - seed[0]).hypot(ellipse[1] - seed[1])
                + (ellipse[2] - seed[2]).abs()
                + (ellipse[3] - seed[2]).abs()
        };
        assert!(
            parameter_error(weighted) < parameter_error(unweighted) * 0.35,
            "unweighted={unweighted:?} weighted={weighted:?}"
        );
    }

    #[test]
    fn analog_force_recovers_phase_in_the_fitted_conic_frame() {
        // This spelling deliberately has its nominal first axis shorter than
        // the second. `from_fit` swaps the axes and rotates the local frame;
        // reusing a rough-search meridian phase would therefore be capable of
        // sampling the opposite side of the eye.
        let search = OuterSearchEllipse::from_fit([104.0, 83.0, 34.0, 62.0, -0.35]);
        let contact_phase = 2.46;
        let (contact, _) = search.point_and_normal(contact_phase, 1.035);
        let accepted = OuterIrisPoint {
            x: contact.0,
            y: contact.1,
            contrast: 100.0,
        };
        let recovered_phase = fitted_outer_phase_for_contact(search, accepted);
        let (recovered, _) = search.point_and_normal(recovered_phase, 1.0);
        let (wrong_side, _) = search.point_and_normal(recovered_phase + PI, 1.0);
        let distance = |point: (f64, f64)| (point.0 - contact.0).hypot(point.1 - contact.1);
        assert!(
            distance(recovered) < 2.5,
            "contact={contact:?} recovered={recovered:?}"
        );
        assert!(
            distance(recovered) * 20.0 < distance(wrong_side),
            "contact={contact:?} recovered={recovered:?} wrong_side={wrong_side:?}"
        );
    }

    #[test]
    fn outer_boundary_recovers_python_style_radial_points() {
        let raw = synthetic_eye(194.0, 148.0, 2.0);
        let coarse = BorderFocus {
            eye_basin_valid: true,
            center: (194.0, 148.0),
            radius: 70.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            ..BorderFocus::default()
        };
        let boundary = detect_outer_iris_boundary(&raw, WIDTH, HEIGHT, &coarse);
        assert_eq!(boundary.points.len(), 64, "boundary={boundary:?}");
        assert!(
            (boundary.center.0 - 194.0).abs() < 2.0,
            "boundary={boundary:?}"
        );
        assert!(
            (boundary.center.1 - 148.0).abs() < 4.0,
            "boundary={boundary:?}"
        );
        assert!(
            (boundary.major_radius - 70.0).abs() < 3.0,
            "boundary={boundary:?}"
        );
        assert!(
            (boundary.minor_radius - 70.0).abs() < 5.0,
            "boundary={boundary:?}"
        );
        assert!(
            boundary.points.iter().all(|point| point.contrast > 100.0),
            "boundary={boundary:?}"
        );
        assert!(
            boundary.evidence_points.iter().any(|point| {
                point.y > coarse.center.1 + coarse.radius * 0.70
                    && (point.x - coarse.center.0).abs() > coarse.radius * 0.35
            }),
            "lower lateral sweep evidence missing: boundary={boundary:?}"
        );
    }

    #[test]
    fn expired_presentation_budget_does_not_truncate_current_outer_lattice() {
        let raw = synthetic_eye(194.0, 148.0, 2.0);
        let coarse = BorderFocus {
            eye_basin_valid: true,
            center: (194.0, 148.0),
            radius: 70.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            ..BorderFocus::default()
        };
        let detect = |budget| {
            let mut tracker = OuterIrisTracker::default();
            detect_outer_iris_boundary_between_eyelids_tracked_with_minimum_scale(
                &raw,
                WIDTH,
                HEIGHT,
                0,
                0,
                &coarse,
                &[],
                &[],
                &mut tracker,
                OUTER_IRIS_MIN_SEARCH_SCALE,
                budget,
            )
        };
        let ordinary = detect(OUTER_IRIS_SYSTEM_BUDGET);
        let already_expired = detect(Duration::ZERO);

        assert_eq!(already_expired.points.len(), 64);
        assert_eq!(
            ordinary.evidence_points.len(),
            already_expired.evidence_points.len()
        );
        assert!((ordinary.center.0 - already_expired.center.0).abs() < 1.0e-12);
        assert!((ordinary.center.1 - already_expired.center.1).abs() < 1.0e-12);
        assert!((ordinary.major_radius - already_expired.major_radius).abs() < 1.0e-12);
        assert!((ordinary.minor_radius - already_expired.minor_radius).abs() < 1.0e-12);
        assert!((ordinary.angle - already_expired.angle).abs() < 1.0e-12);
    }

    #[test]
    fn lateral_sweep_is_ten_percent_longer_and_rolls_back_before_mismatch() {
        let old_arm_length = 0.80f64.hypot(0.36);
        let sweep = outer_iris_lateral_sweep_angle();
        let new_arm_length = 2.0 * (sweep * 0.5).sin();
        assert!((new_arm_length / old_arm_length - 1.10).abs() < 1.0e-12);

        let seed = [100.0, 80.0, 40.0];
        let mut paths: [Vec<Option<OuterIrisPoint>>; 4] = std::array::from_fn(|branch| {
            let side = if branch < 2 { 0.0 } else { PI };
            let direction = if branch % 2 == 0 { -1.0 } else { 1.0 };
            (0..OUTER_IRIS_SWEEP_BRANCH_SAMPLES)
                .map(|step| {
                    let angle = side
                        + direction * sweep * step as f64
                            / (OUTER_IRIS_SWEEP_BRANCH_SAMPLES - 1) as f64;
                    Some(OuterIrisPoint {
                        x: seed[0] + seed[2] * angle.cos(),
                        y: seed[1] + seed[2] * angle.sin(),
                        contrast: 120.0,
                    })
                })
                .collect()
        });
        let starving_cues: [Vec<Option<OuterIrisPoint>>; 4] = std::array::from_fn(|branch| {
            paths[branch]
                .iter()
                .enumerate()
                .map(|(step, point)| if step == 0 { *point } else { None })
                .collect()
        });
        let (fallback, cue_evidence) =
            select_lateral_outer_curve_sweeps_with_fallback(&starving_cues, &paths, seed);
        assert!(fallback.len() > 8, "fallback={fallback:?}");
        assert_eq!(cue_evidence.len(), 2);
        assert_eq!(lateral_sweep_endpoints(&cue_evidence, seed).len(), 2);

        let mismatch_step = 5;
        let mismatch_angle =
            sweep * mismatch_step as f64 / (OUTER_IRIS_SWEEP_BRANCH_SAMPLES - 1) as f64;
        paths[1][mismatch_step] = Some(OuterIrisPoint {
            x: seed[0] + 58.0 * mismatch_angle.cos(),
            y: seed[1] + 58.0 * mismatch_angle.sin(),
            contrast: 900.0,
        });
        let later_angle =
            sweep * (mismatch_step + 1) as f64 / (OUTER_IRIS_SWEEP_BRANCH_SAMPLES - 1) as f64;
        let later_point = (
            seed[0] + seed[2] * later_angle.cos(),
            seed[1] + seed[2] * later_angle.sin(),
        );

        let evidence = select_lateral_outer_curve_sweeps(&paths, seed);

        assert!(!evidence
            .iter()
            .any(|point| { (point.x - later_point.0).hypot(point.y - later_point.1) < 0.01 }));
        let last_good_angle =
            sweep * (mismatch_step - 1) as f64 / (OUTER_IRIS_SWEEP_BRANCH_SAMPLES - 1) as f64;
        let last_good = (
            seed[0] + seed[2] * last_good_angle.cos(),
            seed[1] + seed[2] * last_good_angle.sin(),
        );
        assert!(evidence
            .iter()
            .any(|point| (point.x - last_good.0).hypot(point.y - last_good.1) < 0.01));
    }

    #[test]
    fn non_sclera_chroma_and_transverse_edges_are_hard_tip_stops() {
        let neutral_plane = NativeLogPlane {
            width: 8,
            height: 8,
            origin_x: 0.0,
            origin_y: 0.0,
            log_rg: vec![0.0; 64],
            log_bg: vec![0.0; 64],
            reflectance_log_rg: vec![0.0; 64],
            reflectance_log_bg: vec![0.0; 64],
            intensity: vec![0.0; 64],
            void: vec![0.0; 64],
        };
        let colored_plane = NativeLogPlane {
            log_rg: vec![0.5; 64],
            ..neutral_plane
        };
        let reference = ScleraColorReference {
            log_rg: 0.0,
            log_bg: 0.0,
            tolerance: 0.10,
        };
        let tip = OuterIrisPoint {
            x: 3.0,
            y: 3.0,
            contrast: 100.0,
        };
        assert!(!outer_tip_matches_sclera_color(
            &colored_plane,
            reference,
            tip,
            0.0,
        ));

        let width = 32;
        let height = 32;
        let crossed = (0..height)
            .flat_map(|y| (0..width).map(move |_| if y < height / 2 { 80u16 } else { 900u16 }))
            .collect::<Vec<_>>();
        let crossed_luma = BoxLuma5::new(&crossed, width, height);
        let uniform_luma = BoxLuma5::new(&vec![400u16; width * height], width, height);
        let tip = OuterIrisPoint {
            x: 16.0,
            y: 16.0,
            contrast: 100.0,
        };
        assert!(outer_tip_has_transverse_edge(&crossed_luma, tip, 0.0));
        assert!(!outer_tip_has_transverse_edge(&uniform_luma, tip, 0.0));
    }

    #[test]
    fn outer_boundary_requires_plausible_coarse_geometry() {
        let raw = synthetic_eye(194.0, 148.0, 2.0);
        let boundary = detect_outer_iris_boundary(&raw, WIDTH, HEIGHT, &BorderFocus::default());
        assert!(boundary.points.is_empty());
    }

    #[test]
    fn outer_boundary_ignores_stronger_occluding_eyelids() {
        let mut raw = synthetic_eye(194.0, 148.0, 3.0);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let upper_lid = 105.0 + 0.0018 * (x as f64 - 194.0).powi(2);
                let lower_lid = 191.0 - 0.0015 * (x as f64 - 194.0).powi(2);
                if y as f64 <= upper_lid || y as f64 >= lower_lid {
                    raw[y * WIDTH + x] = if y as f64 <= upper_lid { 900 } else { 80 };
                }
            }
        }
        let coarse = BorderFocus {
            // A soft selected eye may supply geometry before satisfying the
            // stricter autofocus evidence gate.
            eye_basin_valid: false,
            center: (194.0, 148.0),
            radius: 70.0,
            axis_ratio: 3.0,
            axis_angle: 0.0,
            ..BorderFocus::default()
        };
        let boundary = detect_outer_iris_boundary(&raw, WIDTH, HEIGHT, &coarse);
        assert_eq!(boundary.points.len(), 64, "boundary={boundary:?}");
        assert!(
            (boundary.center.0 - 194.0).abs() < 4.0,
            "boundary={boundary:?}"
        );
        assert!(
            (boundary.center.1 - 148.0).abs() < 4.0,
            "boundary={boundary:?}"
        );
        assert!(
            (boundary.major_radius - 70.0).abs() < 5.0,
            "boundary={boundary:?}"
        );
        let (angle_sine, angle_cosine) = boundary.angle.sin_cos();
        assert!(
            boundary.points.iter().all(|point| {
                let delta_x = point.x - boundary.center.0;
                let delta_y = point.y - boundary.center.1;
                let local_x = angle_cosine * delta_x + angle_sine * delta_y;
                let local_y = -angle_sine * delta_x + angle_cosine * delta_y;
                ((local_x / boundary.major_radius).powi(2)
                    + (local_y / boundary.minor_radius).powi(2)
                    - 1.0)
                    .abs()
                    < 0.001
            }),
            "boundary={boundary:?}"
        );
    }

    #[test]
    fn outer_prediction_excludes_evidence_outside_measured_eyelids() {
        let raw = synthetic_eye(194.0, 148.0, 2.0);
        let coarse = BorderFocus {
            center: (194.0, 148.0),
            radius: 70.0,
            ..BorderFocus::default()
        };
        let upper = [
            BorderPoint {
                x: 110,
                y: 120,
                quality: 10.0,
            },
            BorderPoint {
                x: 278,
                y: 120,
                quality: 10.0,
            },
        ];
        let lower = [
            BorderPoint {
                x: 110,
                y: 200,
                quality: 10.0,
            },
            BorderPoint {
                x: 278,
                y: 200,
                quality: 10.0,
            },
        ];

        let boundary = detect_outer_iris_boundary_between_eyelids(
            &raw, WIDTH, HEIGHT, &coarse, &upper, &lower,
        );

        assert_eq!(boundary.points.len(), 64, "boundary={boundary:?}");
        assert!(
            boundary
                .evidence_points
                .iter()
                .all(|point| point.y > 120.0 && point.y < 200.0),
            "forbidden evidence reached fit: boundary={boundary:?}"
        );
        assert!(
            (boundary.major_radius - 70.0).abs() < 4.0,
            "boundary={boundary:?}"
        );
    }

    #[test]
    fn outer_prediction_drops_isolated_eyelash_radial_outliers_and_refits() {
        let seed = [194.0, 148.0, 70.0];
        let mut evidence = (0..18)
            .map(|index| {
                let angle = -0.65 + index as f64 * 1.30 / 17.0;
                OuterIrisPoint {
                    x: seed[0] + seed[2] * angle.cos(),
                    y: seed[1] + seed[2] * angle.sin(),
                    contrast: 120.0,
                }
            })
            .collect::<Vec<_>>();
        evidence.extend([
            OuterIrisPoint {
                x: seed[0] + 112.0,
                y: seed[1] - 8.0,
                contrast: 900.0,
            },
            OuterIrisPoint {
                x: seed[0] - 104.0,
                y: seed[1] + 18.0,
                contrast: 850.0,
            },
            OuterIrisPoint {
                x: seed[0] + 12.0,
                y: seed[1] + 108.0,
                contrast: 920.0,
            },
        ]);

        let circle = fit_and_reject_outer_circle_outliers(&mut evidence, seed);

        assert_eq!(evidence.len(), 18, "evidence={evidence:?}");
        assert!((circle[0] - seed[0]).abs() < 1.0, "circle={circle:?}");
        assert!((circle[1] - seed[1]).abs() < 1.0, "circle={circle:?}");
        assert!((circle[2] - seed[2]).abs() < 1.0, "circle={circle:?}");
    }

    #[test]
    fn upper_eyeliner_tracks_the_lid_and_rejects_a_stronger_eyebrow() {
        let mut raw = synthetic_eye_with_radius(194.0, 148.0, 3.0, 30.0);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let offset = x as f64 - 194.0;
                let upper_lid = 105.0 + 0.0018 * offset.powi(2);
                let eyebrow = 70.0 + 0.0012 * offset.powi(2);
                if y as f64 <= upper_lid {
                    raw[y * WIDTH + x] = 90;
                }
                if (y as f64 - eyebrow).abs() <= 2.0 {
                    raw[y * WIDTH + x] = 20;
                }
            }
        }
        let coarse = BorderFocus {
            center: (194.0, 148.0),
            radius: 70.0,
            ..BorderFocus::default()
        };

        let points = detect_upper_eyelid_points(&raw, WIDTH, HEIGHT, 0, 0, &coarse);

        assert!(points.len() >= 11, "points={points:?}");
        let center = points
            .iter()
            .min_by_key(|point| point.x.abs_diff(194))
            .unwrap();
        assert!(center.y.abs_diff(105) <= 5, "center={center:?}");
        assert!(points.iter().all(|point| point.y > 88), "points={points:?}");
    }

    #[test]
    fn lower_eyeliner_tracks_the_lid_below_visible_sclera() {
        let mut raw = synthetic_eye_with_radius(194.0, 148.0, 3.0, 30.0);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let offset = x as f64 - 194.0;
                let lower_lid = 191.0 - 0.0015 * offset.powi(2);
                let stronger_cheek_crease = 225.0 - 0.0008 * offset.powi(2);
                if y as f64 >= lower_lid {
                    raw[y * WIDTH + x] = 90;
                }
                if (y as f64 - stronger_cheek_crease).abs() <= 2.0 {
                    raw[y * WIDTH + x] = 980;
                }
            }
        }
        let coarse = BorderFocus {
            center: (194.0, 148.0),
            radius: 70.0,
            ..BorderFocus::default()
        };

        let points = detect_lower_eyelid_points(&raw, WIDTH, HEIGHT, 0, 0, &coarse);

        assert!(points.len() >= 11, "points={points:?}");
        let center = points
            .iter()
            .min_by_key(|point| point.x.abs_diff(194))
            .unwrap();
        assert!(center.y.abs_diff(191) <= 5, "center={center:?}");
        assert!(
            points.iter().all(|point| point.y < 210),
            "points={points:?}"
        );
    }

    #[test]
    fn nautilus_scene_walks_both_lids_and_keeps_folds_and_lashes_secondary() {
        let center = (194.0, 148.0);
        let mut raw = vec![360u16; WIDTH * HEIGHT];
        let lash_columns = [135usize, 152, 169, 186, 203, 220, 237, 254];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let normalized = ((x as f64 - center.0) / 84.0).clamp(-1.0, 1.0);
                let opening = (1.0 - normalized * normalized).max(0.0).sqrt();
                let upper = center.1 - 44.0 * opening;
                let lower = center.1 + 42.0 * opening;
                let radius = (x as f64 - center.0).hypot(y as f64 - center.1);
                let mut value = if y as f64 >= upper && y as f64 <= lower {
                    if radius < 25.0 {
                        65
                    } else if radius < 70.0 {
                        270
                    } else {
                        790
                    }
                } else {
                    360
                };
                let upper_fold = upper - 14.0;
                let lower_fold = lower + 10.0;
                if (y as f64 - upper_fold).abs() <= 1.0 || (y as f64 - lower_fold).abs() <= 1.0 {
                    value = 145;
                }
                // A stronger eyebrow is outside the limbus-anchored shell.
                if (y as f64 - (64.0 + 0.0007 * (x as f64 - center.0).powi(2))).abs() <= 2.0 {
                    value = 20;
                }
                if lash_columns.iter().any(|column| x.abs_diff(*column) <= 1)
                    && y as f64 <= upper
                    && y as f64 >= upper - 10.0
                {
                    value = 28;
                }
                raw[y * WIDTH + x] = value;
            }
        }
        let outer = OuterIrisBoundary {
            center,
            major_radius: 70.0,
            minor_radius: 62.0,
            angle: 0.0,
            points: vec![OuterIrisPoint {
                x: center.0 + 70.0,
                y: center.1,
                contrast: 1.0,
            }],
            ..OuterIrisBoundary::default()
        };

        let scene =
            discover_eyelid_scene_nautilus(&raw, WIDTH, HEIGHT, &outer, Some((194.0, 150.0)));

        assert!(scene.upper_margin.len() >= 18, "scene={scene:?}");
        assert!(scene.lower_margin.len() >= 18, "scene={scene:?}");
        let upper_center = scene
            .upper_margin
            .iter()
            .min_by_key(|point| point.x.abs_diff(194))
            .unwrap();
        let lower_center = scene
            .lower_margin
            .iter()
            .min_by_key(|point| point.x.abs_diff(194))
            .unwrap();
        assert!(upper_center.y.abs_diff(104) <= 5, "scene={scene:?}");
        assert!(lower_center.y.abs_diff(190) <= 5, "scene={scene:?}");
        assert!(scene.upper_fold.len() >= 7, "scene={scene:?}");
        assert!(scene.lower_fold.len() >= 7, "scene={scene:?}");
        assert!(scene.upper_lashes.len() >= 2, "scene={scene:?}");
        assert!(scene.upper_fold.iter().all(|fold| {
            scene
                .upper_margin
                .iter()
                .min_by_key(|margin| margin.x.abs_diff(fold.x))
                .is_some_and(|margin| fold.y < margin.y)
        }));
        assert!(scene.lower_fold.iter().all(|fold| {
            scene
                .lower_margin
                .iter()
                .min_by_key(|margin| margin.x.abs_diff(fold.x))
                .is_some_and(|margin| fold.y > margin.y)
        }));
    }

    #[test]
    fn nautilus_scene_finds_ordered_lids_on_the_reviewed_lossless_raw() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../outputs/raw-focus-hotkey/subject-right-focus-0562-1786917874-518201484.raw10",
        );
        let payload = std::fs::read(&path).unwrap_or_else(|error| {
            panic!("read reviewed lossless RAW {}: {error}", path.display())
        });
        let raw = unpack_test_raw10(&payload);
        let outer = OuterIrisBoundary {
            center: (215.49966430664062, 120.35222625732422),
            major_radius: 103.78759002685547,
            minor_radius: 96.19979858398438,
            angle: 3.417328097057218,
            points: vec![OuterIrisPoint {
                x: 313.84424951876974,
                y: 120.23248794824086,
                contrast: 1.0,
            }],
            ..OuterIrisBoundary::default()
        };

        let scene = discover_eyelid_scene_nautilus(
            &raw,
            WIDTH,
            HEIGHT,
            &outer,
            Some((201.334677765, 144.769831329)),
        );

        assert!(scene.upper_margin.len() >= 12, "scene={scene:?}");
        assert!(scene.lower_margin.len() >= 12, "scene={scene:?}");
        assert_eq!(scene.upper_status, EyelidObservationStatus::Observed);
        assert_eq!(scene.lower_status, EyelidObservationStatus::Observed);
        let upper_median = scene.upper_margin[scene.upper_margin.len() / 2].y;
        let lower_median = scene.lower_margin[scene.lower_margin.len() / 2].y;
        assert!(upper_median < lower_median, "scene={scene:?}");
    }

    #[test]
    fn nautilus_scene_marks_reviewed_lower_lids_outside_the_raw_roi_as_clipped() {
        let fixtures = [
            (
                "subject-right-focus-0562-1786917878-520879320.raw10",
                (177.18551635742188, 148.80807495117188),
                117.38675689697266,
                101.78601837158203,
                3.069374249571515,
            ),
            (
                "subject-right-focus-0562-1786917880-100512714.raw10",
                (213.4573211669922, 154.85443115234375),
                123.00695037841797,
                118.80276489257812,
                3.532584382713668,
            ),
        ];
        for (name, center, major_radius, minor_radius, angle) in fixtures {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../outputs/raw-focus-hotkey")
                .join(name);
            let payload = std::fs::read(&path).unwrap_or_else(|error| {
                panic!("read reviewed lossless RAW {}: {error}", path.display())
            });
            let raw = unpack_test_raw10(&payload);
            let outer = OuterIrisBoundary {
                center,
                major_radius,
                minor_radius,
                angle,
                points: vec![OuterIrisPoint {
                    x: center.0 + major_radius,
                    y: center.1,
                    contrast: 1.0,
                }],
                ..OuterIrisBoundary::default()
            };

            let scene = discover_eyelid_scene_nautilus(&raw, WIDTH, HEIGHT, &outer, Some(center));

            assert_eq!(
                scene.lower_status,
                EyelidObservationStatus::RoiClipped,
                "{name}: {scene:?}"
            );
            assert!(scene.lower_margin.is_empty(), "{name}: {scene:?}");
            assert!(
                scene.lower_clipped_occluder.len() >= 12,
                "{name}: clipped-side ambiguity should remain available as censor-only evidence: {scene:?}"
            );
            assert!(scene.lower_fold.is_empty(), "{name}: {scene:?}");
            assert!(scene.lower_lashes.is_empty(), "{name}: {scene:?}");
            assert_ne!(
                scene.upper_status,
                EyelidObservationStatus::RoiClipped,
                "{name}: {scene:?}"
            );
            assert!(
                scene
                    .upper_margin
                    .windows(2)
                    .all(|pair| pair[0].y.abs_diff(pair[1].y) <= 20),
                "{name}: upper margin bridged unrelated edges: {scene:?}"
            );
        }
    }

    #[test]
    fn eyeliner_rejects_a_thin_contiguous_scleral_distractor() {
        let mut raw = synthetic_eye_with_radius(194.0, 148.0, 3.0, 30.0);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let offset = x as f64 - 194.0;
                let thin_arc = 191.0 - 0.0015 * offset.powi(2);
                if (y as f64 - thin_arc).abs() <= 2.0 {
                    raw[y * WIDTH + x] = 40;
                }
            }
        }
        let coarse = BorderFocus {
            center: (194.0, 148.0),
            radius: 70.0,
            ..BorderFocus::default()
        };

        let points = detect_lower_eyelid_points(&raw, WIDTH, HEIGHT, 0, 0, &coarse);

        assert!(points.is_empty(), "thin distractor admitted: {points:?}");
    }

    #[test]
    fn inner_boundary_uses_native_log_chromaticity_at_pupil_rim() {
        let center = (194.0, 132.0);
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let radius = (x as f64 - center.0).hypot(y as f64 - center.1);
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                raw[y * WIDTH + x] = if radius < 28.0 {
                    72
                } else if radius < 70.0 {
                    match phase {
                        (0, 0) => 280,
                        (1, 1) => 170,
                        _ => 390,
                    }
                } else {
                    match phase {
                        (0, 0) => 820,
                        (1, 1) => 720,
                        _ => 790,
                    }
                };
            }
        }
        let coarse = BorderFocus {
            center,
            radius: 70.0,
            axis_ratio: 1.0,
            ..BorderFocus::default()
        };
        let boundary = detect_inner_iris_boundary(&raw, WIDTH, HEIGHT, 0, 0, &coarse);
        assert_eq!(boundary.points.len(), 21, "boundary={boundary:?}");
        assert!(
            (boundary.center.0 - center.0).abs() < 4.0,
            "boundary={boundary:?}"
        );
        assert!(
            (boundary.center.1 - center.1).abs() < 4.0,
            "boundary={boundary:?}"
        );
        assert!(
            (boundary.radius - 28.0).abs() < 5.0,
            "boundary={boundary:?}"
        );
        assert!(
            boundary.radial_candidates.len() >= 18,
            "the clean pupil rim did not populate the sparse polar bank: boundary={boundary:?}"
        );
        assert!(
            boundary.radial_candidates.iter().all(|candidate| {
                candidate.raw_score.is_finite()
                    && candidate.peak_prominence.is_finite()
                    && candidate.equivalent_radius_px.is_finite()
            }),
            "the sparse polar bank contains non-finite evidence: boundary={boundary:?}"
        );

        // A stale/confident temporal estimate may rank an ambiguous edge,
        // but it must not clip a decisive current-frame pupil transition.
        let stale_prior = InnerIrisRadiusPrior::new(12.0, 10.0, 14.0, 1.0).unwrap();
        let guided = detect_inner_iris_boundary_with_radius_prior(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(stale_prior),
        );
        assert_eq!(guided.points.len(), 21, "guided={guided:?}");
        assert!((guided.radius - 28.0).abs() < 5.0, "guided={guided:?}");
        assert!(
            guided.radius > stale_prior.preferred_maximum_equivalent_radius_px + 5.0,
            "a temporal preference became a hard boundary: guided={guided:?} prior={stale_prior:?}"
        );
        assert_eq!(
            guided.radial_candidates.len(),
            boundary.radial_candidates.len(),
            "a soft radius preference changed the prior-free candidate population"
        );
        for (unguided, guided) in boundary
            .radial_candidates
            .iter()
            .zip(&guided.radial_candidates)
        {
            assert_eq!(unguided.sector_index, guided.sector_index);
            assert!((unguided.equivalent_radius_px - guided.equivalent_radius_px).abs() < 1.0e-12);
            assert!((unguided.raw_score - guided.raw_score).abs() < 1.0e-12);
            assert!((unguided.peak_prominence - guided.peak_prominence).abs() < 1.0e-12);
        }
    }

    #[test]
    fn inner_boundary_scales_past_legacy_cap_and_obeys_hard_radius_envelope() {
        let center = (194.0, 128.0);
        let pupil_radius = 56.0;
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let radius = (x as f64 - center.0).hypot(y as f64 - center.1);
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                raw[y * WIDTH + x] = if radius < pupil_radius {
                    72
                } else if radius < 100.0 {
                    match phase {
                        (0, 0) => 280,
                        (1, 1) => 170,
                        _ => 390,
                    }
                } else {
                    match phase {
                        (0, 0) => 820,
                        (1, 1) => 720,
                        _ => 790,
                    }
                };
            }
        }
        let coarse = BorderFocus {
            center,
            radius: 100.0,
            axis_ratio: 1.0,
            ..BorderFocus::default()
        };
        let broad_envelope = InnerIrisRadiusEnvelope::new(8.0, 72.0).unwrap();
        let boundary = detect_inner_iris_boundary_conditioned(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(broad_envelope),
            None,
            InnerIrisEvidenceCondition::default(),
        );
        assert_eq!(boundary.points.len(), 21, "boundary={boundary:?}");
        assert!(
            (boundary.radius - pupil_radius).abs() < 5.0,
            "large native pupil was not recovered: boundary={boundary:?}"
        );
        assert!(
            boundary.radius > 45.0,
            "the legacy absolute 40px ceiling is still active: boundary={boundary:?}"
        );

        let tight_envelope = InnerIrisRadiusEnvelope::new(8.0, 44.0).unwrap();
        let constrained = detect_inner_iris_boundary_conditioned(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(tight_envelope),
            None,
            InnerIrisEvidenceCondition::default(),
        );
        assert_eq!(constrained.points.len(), 21, "constrained={constrained:?}");
        assert!(
            constrained.major_radius <= tight_envelope.maximum_equivalent_radius_px + 0.5,
            "hard radius envelope did not constrain the selected RAW edge: constrained={constrained:?} envelope={tight_envelope:?}"
        );
    }

    #[test]
    fn inner_boundary_point_confidence_uses_its_selected_margin_not_another_edge() {
        let radii = [10.0, 10.5, 11.0, 11.5];
        // Cyclic regularization may retain the 10.5 px pupil road even when a
        // distracting outer transition is locally stronger on this ray.
        let cues = [0.12, 0.24, f64::NEG_INFINITY, 0.93];
        assert_eq!(selected_inner_margin_score(&radii, &cues, 10.5), 0.24);
        assert_ne!(
            selected_inner_margin_score(&radii, &cues, 10.5),
            cues.iter().copied().reduce(f64::max).unwrap(),
        );
    }

    #[test]
    fn pupil_mass_and_temporal_priors_cannot_masquerade_as_raw_margin_evidence() {
        let center = (194.0, 128.0);
        let raw = vec![512u16; WIDTH * HEIGHT];
        let coarse = BorderFocus {
            center,
            radius: 80.0,
            axis_ratio: 1.0,
            ..BorderFocus::default()
        };
        let boundary = detect_inner_iris_boundary_with_center(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(center),
            InnerIrisRadiusEnvelope::new(10.0, 40.0),
            InnerIrisRadiusPrior::new(20.0, 18.0, 22.0, 1.0),
            InnerIrisEvidenceCondition::default(),
        );

        // The ranking priors may still produce an inspectable proposal on a
        // flat frame, but they supply no current-frame edge confidence.
        assert_eq!(boundary.points.len(), 21, "boundary={boundary:?}");
        assert!(
            boundary
                .points
                .iter()
                .all(|point| point.score.abs() < 1.0e-12),
            "a selection prior leaked into observation confidence: boundary={boundary:?}"
        );
    }

    #[test]
    fn complete_asymmetric_radial_hits_cannot_move_the_fixed_pupil_center() {
        let center = (194.0, 128.0);
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let dx = x as f64 - center.0;
                let dy = y as f64 - center.1;
                let radius = dx.hypot(dy);
                // Model the exact failure caused by a coherent lower-lid
                // transition: lower-facing rays select a farther boundary
                // than upper-facing rays even though the acquired pupil
                // center remains known.
                let pupil_radius = if dy > 0.0 { 39.0 } else { 24.0 };
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                raw[y * WIDTH + x] = if radius < pupil_radius {
                    68
                } else if radius < 82.0 {
                    match phase {
                        (0, 0) => 300,
                        (1, 1) => 175,
                        _ => 405,
                    }
                } else {
                    790
                };
            }
        }
        let coarse = BorderFocus {
            center,
            radius: 82.0,
            axis_ratio: 1.0,
            ..BorderFocus::default()
        };
        let boundary = detect_inner_iris_boundary_with_center(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(center),
            InnerIrisRadiusEnvelope::new(12.0, 48.0),
            None,
            InnerIrisEvidenceCondition::default(),
        );

        assert_eq!(boundary.points.len(), 21, "boundary={boundary:?}");
        let radial_hit_centroid = (
            boundary.points.iter().map(|point| point.x).sum::<f64>() / boundary.points.len() as f64,
            boundary.points.iter().map(|point| point.y).sum::<f64>() / boundary.points.len() as f64,
        );
        assert!(
            (radial_hit_centroid.1 - center.1).abs() > 2.0,
            "fixture did not create a lid-biased radial centroid: boundary={boundary:?}"
        );
        assert_eq!(
            boundary.center, center,
            "unequal ray radii were misrepresented as a pupil-center fit"
        );
    }

    #[test]
    fn tilted_inner_boundary_applies_equivalent_radius_envelope_in_affine_space() {
        let center = (194.0, 128.0);
        let projection_ratio = 0.50_f64;
        let angle = 0.35_f64;
        let pupil_major_radius = 30.0_f64;
        let pupil_minor_radius = pupil_major_radius * projection_ratio;
        let limbus_major_radius = 100.0_f64;
        let limbus_minor_radius = limbus_major_radius * projection_ratio;
        let pupil_equivalent_radius = (pupil_major_radius * pupil_minor_radius).sqrt();
        let limbus_equivalent_radius = (limbus_major_radius * limbus_minor_radius).sqrt();
        let (angle_sin, angle_cos) = angle.sin_cos();
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let dx = x as f64 - center.0;
                let dy = y as f64 - center.1;
                let local_x = angle_cos * dx + angle_sin * dy;
                let local_y = -angle_sin * dx + angle_cos * dy;
                let in_pupil = (local_x / pupil_major_radius).powi(2)
                    + (local_y / pupil_minor_radius).powi(2)
                    < 1.0;
                let in_iris = (local_x / limbus_major_radius).powi(2)
                    + (local_y / limbus_minor_radius).powi(2)
                    < 1.0;
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                raw[y * WIDTH + x] = if in_pupil {
                    72
                } else if in_iris {
                    match phase {
                        (0, 0) => 280,
                        (1, 1) => 170,
                        _ => 390,
                    }
                } else {
                    match phase {
                        (0, 0) => 820,
                        (1, 1) => 720,
                        _ => 790,
                    }
                };
            }
        }
        let coarse = BorderFocus {
            center,
            radius: limbus_equivalent_radius,
            axis_ratio: 1.0 / projection_ratio,
            axis_angle: angle,
            ..BorderFocus::default()
        };
        let envelope = InnerIrisRadiusEnvelope::new(
            pupil_equivalent_radius - 2.0,
            pupil_equivalent_radius + 2.0,
        )
        .unwrap();
        let boundary = detect_inner_iris_boundary_conditioned(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(envelope),
            None,
            InnerIrisEvidenceCondition::default(),
        );
        assert_eq!(boundary.points.len(), 21, "boundary={boundary:?}");
        assert!(
            (boundary.radius - pupil_equivalent_radius).abs() < 2.5,
            "the selected scalar is not the projected area-equivalent radius: boundary={boundary:?}"
        );
        assert!(
            (boundary.major_radius - pupil_major_radius).abs() < 3.0,
            "the tilted pupil's image-space major axis was clipped by the equivalent-radius envelope: boundary={boundary:?}"
        );
        assert!(
            (boundary.minor_radius - pupil_minor_radius).abs() < 3.0,
            "the tilted pupil's image-space minor axis was not affine-scaled: boundary={boundary:?}"
        );
        assert!(
            boundary.major_radius > envelope.maximum_equivalent_radius_px + 4.0,
            "an area-equivalent upper bound was incorrectly applied to every image-space axis: boundary={boundary:?} envelope={envelope:?}"
        );

        let tight_envelope = InnerIrisRadiusEnvelope::new(15.0, 18.0).unwrap();
        let constrained = detect_inner_iris_boundary_conditioned(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(tight_envelope),
            None,
            InnerIrisEvidenceCondition::default(),
        );
        let major_scale = (1.0 / projection_ratio).sqrt();
        let minor_scale = projection_ratio.sqrt();
        assert_eq!(constrained.points.len(), 21, "constrained={constrained:?}");
        assert!(
            constrained.major_radius
                <= tight_envelope.maximum_equivalent_radius_px * major_scale + 1.0,
            "the hard equivalent-radius maximum did not scale the major axis: constrained={constrained:?} envelope={tight_envelope:?}"
        );
        assert!(
            constrained.minor_radius
                <= tight_envelope.maximum_equivalent_radius_px * minor_scale + 1.0,
            "the hard equivalent-radius maximum did not scale the minor axis: constrained={constrained:?} envelope={tight_envelope:?}"
        );
    }

    #[test]
    fn clipped_tilted_inner_boundary_censors_missing_rays_without_moving_its_center() {
        let center = (365.0_f64, 128.0_f64);
        let projection_ratio = 0.50_f64;
        let angle = 0.22_f64;
        let pupil_major_radius = 30.0_f64;
        let pupil_minor_radius = pupil_major_radius * projection_ratio;
        let limbus_major_radius = 80.0_f64;
        let limbus_minor_radius = limbus_major_radius * projection_ratio;
        let pupil_equivalent_radius = (pupil_major_radius * pupil_minor_radius).sqrt();
        let (angle_sin, angle_cos) = angle.sin_cos();
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let dx = x as f64 - center.0;
                let dy = y as f64 - center.1;
                let local_x = angle_cos * dx + angle_sin * dy;
                let local_y = -angle_sin * dx + angle_cos * dy;
                let in_pupil = (local_x / pupil_major_radius).powi(2)
                    + (local_y / pupil_minor_radius).powi(2)
                    < 1.0;
                let in_iris = (local_x / limbus_major_radius).powi(2)
                    + (local_y / limbus_minor_radius).powi(2)
                    < 1.0;
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                raw[y * WIDTH + x] = if in_pupil {
                    72
                } else if in_iris {
                    match phase {
                        (0, 0) => 280,
                        (1, 1) => 170,
                        _ => 390,
                    }
                } else {
                    790
                };
            }
        }
        let coarse = BorderFocus {
            center,
            radius: (limbus_major_radius * limbus_minor_radius).sqrt(),
            axis_ratio: 1.0 / projection_ratio,
            axis_angle: angle,
            ..BorderFocus::default()
        };
        let envelope = InnerIrisRadiusEnvelope::new(
            pupil_equivalent_radius - 2.0,
            pupil_equivalent_radius + 2.0,
        )
        .unwrap();
        let boundary = detect_inner_iris_boundary_with_center(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            Some(center),
            Some(envelope),
            None,
            InnerIrisEvidenceCondition::default(),
        );

        assert!(
            (8..21).contains(&boundary.points.len()),
            "the clipped pupil should retain only observable RAW rays: boundary={boundary:?}"
        );
        assert_eq!(boundary.center, center, "boundary={boundary:?}");
        assert!(
            boundary.points.iter().all(|point| {
                inner_margin_candidate_is_observable(point.x, point.y, WIDTH, HEIGHT)
            }),
            "a clamped frame edge became fabricated pupil evidence: boundary={boundary:?}"
        );
        assert!(
            (boundary.radius - pupil_equivalent_radius).abs() < 3.0,
            "visible rays did not retain the area-equivalent pupil scale: boundary={boundary:?}"
        );
        assert!(
            (boundary.major_radius / boundary.minor_radius - 1.0 / projection_ratio).abs() < 1.0e-9,
            "the censored boundary lost its affine-circle projection: boundary={boundary:?}"
        );
    }

    #[test]
    fn ordinary_inner_solver_refines_center_on_native_raw_coordinates() {
        let render = |center: (f64, f64)| {
            let mut raw = vec![0u16; WIDTH * HEIGHT];
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let radius = (x as f64 - center.0).hypot(y as f64 - center.1);
                    let phase = (((y / 2) & 1), ((x / 2) & 1));
                    raw[y * WIDTH + x] = if radius < 27.0 {
                        68
                    } else if radius < 82.0 {
                        match phase {
                            (0, 0) => 300,
                            (1, 1) => 175,
                            _ => 405,
                        }
                    } else {
                        790
                    };
                }
            }
            raw
        };
        let solve = |true_center: (f64, f64)| {
            let raw = render(true_center);
            let coarse = BorderFocus {
                center: (true_center.0 + 9.0, true_center.1 - 7.0),
                radius: 82.0,
                axis_ratio: 1.0,
                ..BorderFocus::default()
            };
            detect_inner_iris_boundary_conditioned(
                &raw,
                WIDTH,
                HEIGHT,
                0,
                0,
                &coarse,
                InnerIrisRadiusEnvelope::new(10.0, 42.0),
                None,
                InnerIrisEvidenceCondition::default(),
            )
        };

        let first_true = (193.25, 127.50);
        let second_true = (194.25, 128.50);
        let first = solve(first_true);
        let second = solve(second_true);
        assert_eq!(first.points.len(), 21, "first={first:?}");
        assert_eq!(second.points.len(), 21, "second={second:?}");
        assert!(
            (first.center.0 - first_true.0).hypot(first.center.1 - first_true.1) < 2.5,
            "native-coordinate center refinement missed the first pupil: first={first:?}"
        );
        assert!(
            (second.center.0 - second_true.0).hypot(second.center.1 - second_true.1) < 2.5,
            "native-coordinate center refinement missed the shifted pupil: second={second:?}"
        );
        let measured_shift = (
            second.center.0 - first.center.0,
            second.center.1 - first.center.1,
        );
        assert!(
            (measured_shift.0 - 1.0).abs() < 0.75 && (measured_shift.1 - 1.0).abs() < 0.75,
            "a one-pixel RAW shift was quantized by a lower-resolution geometry lattice: shift={measured_shift:?} first={first:?} second={second:?}"
        );
    }

    #[test]
    fn fixed_radius_orbital_search_recovers_a_contained_pupil_center() {
        let true_center = (190.0, 130.0);
        let pupil_radius = 27.0;
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let radius = (x as f64 - true_center.0).hypot(y as f64 - true_center.1);
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                raw[y * WIDTH + x] = if radius < pupil_radius {
                    68
                } else if radius < 82.0 {
                    match phase {
                        (0, 0) => 300,
                        (1, 1) => 175,
                        _ => 405,
                    }
                } else {
                    790
                };
            }
        }
        let coarse = BorderFocus {
            eye_basin_valid: true,
            center: true_center,
            radius: 82.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            ..BorderFocus::default()
        };
        let bad_seed = (238.0, 96.0);
        let fit = optimize_pupil_center_orbitally_at_fixed_radius(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            bad_seed,
            pupil_radius,
            None,
        )
        .unwrap();
        assert!(
            (fit.center.0 - true_center.0).hypot(fit.center.1 - true_center.1) < 4.0,
            "fit={fit:?}"
        );
        assert!(fit.ring_coverage >= 0.75, "fit={fit:?}");
        assert!(fit.opposing_support >= 0.75, "fit={fit:?}");
        assert!(fit.evaluated_centers <= 80, "fit={fit:?}");
    }

    #[test]
    fn strong_center_prior_beats_a_remote_iris_texture_ring_at_fixed_radius() {
        let limbus_center = (192.0, 128.0);
        let true_center = (184.0, 132.0);
        let remote_ring = (230.0, 94.0);
        let pupil_radius = 27.0;
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let point = (x as f64, y as f64);
                let limbus_radius = (point.0 - limbus_center.0).hypot(point.1 - limbus_center.1);
                let pupil_distance = (point.0 - true_center.0).hypot(point.1 - true_center.1);
                let remote_distance = (point.0 - remote_ring.0).hypot(point.1 - remote_ring.1);
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                let iris = match phase {
                    (0, 0) => 300,
                    (1, 1) => 175,
                    _ => 405,
                };
                raw[y * WIDTH + x] = if pupil_distance < pupil_radius {
                    68
                } else if (remote_distance - pupil_radius).abs() < 3.0 {
                    // A locally sharp but non-pupil iris texture ring.
                    95
                } else if limbus_radius < 82.0 {
                    iris
                } else {
                    790
                };
            }
        }
        let coarse = BorderFocus {
            eye_basin_valid: true,
            center: limbus_center,
            radius: 82.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            ..BorderFocus::default()
        };
        let fit = select_pupil_center_near_strong_prior_at_fixed_radius(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            remote_ring,
            true_center,
            pupil_radius,
            8.0,
            0.18,
        )
        .unwrap();
        assert!(
            (fit.center.0 - true_center.0).hypot(fit.center.1 - true_center.1) <= 8.1,
            "remote texture outvoted the strong pupil-center prior: fit={fit:?}"
        );
        assert!(fit.broad_dark_step > 0.03, "fit={fit:?}");
        assert!(fit.evaluated_centers <= 34, "fit={fit:?}");
    }

    #[test]
    fn compact_geometry_prior_beats_a_larger_concentric_iris_texture_ring() {
        let center = (192.0, 128.0);
        let pupil_radius = 22.0;
        let false_ring_radius = 42.0;
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let radius = (x as f64 - center.0).hypot(y as f64 - center.1);
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                let iris = match phase {
                    (0, 0) => 300,
                    (1, 1) => 175,
                    _ => 405,
                };
                raw[y * WIDTH + x] = if radius < pupil_radius {
                    68
                } else if (radius - false_ring_radius).abs() < 2.5 {
                    110
                } else if radius < 82.0 {
                    iris
                } else {
                    790
                };
            }
        }
        let coarse = BorderFocus {
            eye_basin_valid: true,
            center,
            radius: 82.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            ..BorderFocus::default()
        };
        let fit = select_pupil_geometry_near_strong_prior(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            center,
            false_ring_radius,
            center,
            pupil_radius,
            7.0,
            0.18,
        )
        .unwrap();
        assert!(fit.used_prior_geometry, "fit={fit:?}");
        assert!(
            (fit.equivalent_radius - pupil_radius).abs() <= pupil_radius * 0.25 + 0.5,
            "the concentric iris band taught itself as pupil size: fit={fit:?}"
        );
        assert!(fit.measurement.broad_dark_step > 0.03, "fit={fit:?}");
    }

    #[test]
    fn fixed_radius_orbital_search_rejects_a_bright_core_dark_moat() {
        let limbus_center = (192.0, 128.0);
        let true_center = (184.0, 132.0);
        let distractor_center = (230.0, 96.0);
        let pupil_radius = 27.0;
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let point = (x as f64, y as f64);
                let limbus_radius = (point.0 - limbus_center.0).hypot(point.1 - limbus_center.1);
                let pupil_distance = (point.0 - true_center.0).hypot(point.1 - true_center.1);
                let distractor_distance =
                    (point.0 - distractor_center.0).hypot(point.1 - distractor_center.1);
                let phase = (((y / 2) & 1), ((x / 2) & 1));
                let iris = match phase {
                    (0, 0) => 300,
                    (1, 1) => 175,
                    _ => 405,
                };
                raw[y * WIDTH + x] = if pupil_distance < pupil_radius {
                    68
                } else if distractor_distance < 18.0 {
                    // A reflection island surrounded by a dark moat has the
                    // same attractive local outer edge as a pupil, but the
                    // region broadly inside that edge is bright.
                    360
                } else if distractor_distance < pupil_radius {
                    62
                } else if limbus_radius < 82.0 {
                    iris
                } else {
                    790
                };
            }
        }
        let coarse = BorderFocus {
            eye_basin_valid: true,
            center: limbus_center,
            radius: 82.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            ..BorderFocus::default()
        };
        let fit = optimize_pupil_center_orbitally_at_fixed_radius(
            &raw,
            WIDTH,
            HEIGHT,
            0,
            0,
            &coarse,
            distractor_center,
            pupil_radius,
            None,
        )
        .unwrap();
        assert!(
            (fit.center.0 - true_center.0).hypot(fit.center.1 - true_center.1) < 5.0,
            "bright-core moat won the pupil search: fit={fit:?}"
        );
        assert!(fit.broad_dark_step > 0.05, "fit={fit:?}");
        assert!(fit.broad_dark_support >= 0.70, "fit={fit:?}");
        assert!(fit.evaluated_centers <= 80, "fit={fit:?}");
    }

    #[test]
    fn closed_limbus_lookahead_ignores_a_locally_tempting_lid_branch() {
        let row_count = 24usize;
        let offsets: [f64; 5] = [-12.0, -6.0, 0.0, 6.0, 12.0];
        let pixels_per_mm = 10.0f64;
        let mut scores = vec![-3.0; row_count * offsets.len()];
        for row in 0..row_count {
            scores[row * offsets.len() + 2] = 0.50;
        }
        // At the fork, the lid is locally stronger. A greedy walker changes
        // branch and then stays there because the immediate cost of returning
        // is fractionally larger than one weak lid sample. Looking through the
        // rest of the closed ring shows that this branch is a dead end.
        scores[6 * offsets.len() + 3] = 1.10;
        for row in 7..18 {
            scores[row * offsets.len() + 3] = 0.20;
        }
        let mut greedy = 2usize;
        let mut greedy_path = vec![greedy];
        for row in 1..row_count {
            greedy = (0..offsets.len())
                .filter(|state| state.abs_diff(greedy) <= 3)
                .max_by(|left, right| {
                    let objective = |state: usize| {
                        let step = (offsets[state] - offsets[greedy]) / pixels_per_mm;
                        scores[row * offsets.len() + state] - 0.85 * step * step
                    };
                    objective(*left).total_cmp(&objective(*right))
                })
                .unwrap();
            greedy_path.push(greedy);
        }
        assert_eq!(greedy_path[6], 3, "fixture must tempt the greedy branch");
        assert!(greedy_path[7..18].contains(&3));

        let path = closed_limbus_lookahead_path(&scores, row_count, &offsets, pixels_per_mm)
            .expect("closed lookahead path");
        assert!(path.iter().all(|state| *state == 2), "path={path:?}");
    }

    #[test]
    fn affine_offset_model_assigns_zero_leverage_to_a_projected_lid_arc() {
        let count = 96usize;
        let truth = (0..count)
            .map(|row| {
                let phase = 2.0 * PI * row as f64 / count as f64;
                0.08 + 0.11 * phase.cos() - 0.07 * phase.sin()
                    + 0.05 * (2.0 * phase).cos()
                    + 0.03 * (2.0 * phase).sin()
            })
            .collect::<Vec<_>>();
        let mut observed = truth.clone();
        let mut weights = vec![1.0; count];
        for row in 44..=81 {
            observed[row] += 1.4 + 0.2 * (row as f64 * 0.7).sin();
            weights[row] = 0.0;
        }

        let model = fit_limbus_affine_offset_model(&observed, &weights)
            .expect("measured lateral arcs determine a closure model");
        let maximum_prediction_error = model
            .predicted_mm
            .iter()
            .zip(&truth)
            .map(|(predicted, expected)| (predicted - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(
            maximum_prediction_error < 1.0e-4,
            "projected arc leaked into model: max={maximum_prediction_error} model={model:?}",
        );
        assert!(model.residual_mm[60] > 1.0, "model={model:?}");
    }

    #[test]
    fn cyclic_reentry_extension_crosses_the_storage_seam_until_direct_anatomy() {
        let mut revisited = vec![false; 16];
        revisited[13] = true;
        revisited[14] = true;
        let mut direct_reentry = vec![false; 16];
        direct_reentry[12] = true;
        direct_reentry[1] = true;

        let extended = extend_cyclic_revisit_to_direct_reentry(&mut revisited, &direct_reentry, 5);

        assert!(extended[15] && extended[0], "extended={extended:?}");
        assert!(revisited[15] && revisited[0], "revisited={revisited:?}");
        assert!(
            !extended[1],
            "the direct re-entry itself must stay anchored"
        );
        assert!(!revisited[12] && !revisited[1], "direct anchors moved");
    }

    #[test]
    fn pupil_lookahead_takes_the_iris_when_a_smooth_lid_wins_the_edge_loop() {
        let row_count = 32usize;
        let offsets: [f64; 5] = [-12.0, -6.0, 0.0, 6.0, 12.0];
        let pixels_per_mm = 10.0f64;
        let mut local = vec![-3.0; row_count * offsets.len()];
        let mut heading = vec![-0.8; row_count * offsets.len()];
        for row in 0..row_count {
            // The true limbus enters cohesive iris and then pupil.
            local[row * offsets.len() + 2] = 0.45;
            heading[row * offsets.len() + 2] = 0.70;
            // The neighboring state is a plausible approach to the lid fork.
            local[row * offsets.len() + 3] = 0.05;
            heading[row * offsets.len() + 3] = 0.10;
        }
        // Unlike the short dead end in the ordinary lookahead test, this lid
        // remains a stronger, smooth edge for over half of the closed lap. An
        // edge-only global solver therefore takes it. Looking ahead toward the
        // pupil sees bright sclera beyond that branch and keeps the iris road.
        for row in 7..=25 {
            local[row * offsets.len() + 3] = 0.82;
            heading[row * offsets.len() + 3] = -0.25;
        }
        let edge_only =
            closed_limbus_lookahead_path(&local, row_count, &offsets, pixels_per_mm).unwrap();
        assert!(
            edge_only[9..24].iter().filter(|state| **state == 3).count() >= 12,
            "edge-only path must take the smooth lid: {edge_only:?}"
        );

        let pupil_guided = closed_limbus_pupil_lookahead_path(
            &local,
            &heading,
            &heading,
            row_count,
            &offsets,
            pixels_per_mm,
        )
        .unwrap();
        assert!(
            pupil_guided.iter().all(|state| *state == 2),
            "pupil-guided path must take the iris at the fork: {pupil_guided:?}"
        );
    }

    #[test]
    fn completed_drive_refits_center_from_its_outer_perimeter_points() {
        let expected_center = (123.75, 81.25);
        let expected_major = 58.0;
        let expected_minor = 39.0;
        let expected_angle = 0.43f64;
        let (rotation_sin, rotation_cos) = expected_angle.sin_cos();
        let mut samples = Vec::with_capacity(96);
        for index in 0..96 {
            let phase = 2.0 * PI * index as f64 / 96.0;
            let major_offset = expected_major * phase.cos();
            let minor_offset = expected_minor * phase.sin();
            let mut point = (
                expected_center.0 + major_offset * rotation_cos - minor_offset * rotation_sin,
                expected_center.1 + major_offset * rotation_sin + minor_offset * rotation_cos,
            );
            // Deterministic sub-pixel boundary noise plus two gross fork
            // excursions exercise the robust refit without smuggling the
            // expected center into the fitter as a seed.
            let noise = 0.22 * (phase * 7.0).sin();
            point.0 += noise * phase.cos();
            point.1 += noise * phase.sin();
            if index == 17 || index == 71 {
                point.0 += 12.0;
                point.1 -= 9.0;
            }
            samples.push(LimbusPerimeterDriveSample {
                phase,
                driven_point: point,
                transition_score: 0.8,
                ..LimbusPerimeterDriveSample::default()
            });
        }
        let strip = LimbusPerimeterStrip {
            samples,
            ..LimbusPerimeterStrip::default()
        };

        let fitted = fit_driven_limbus_ellipse(&strip).expect("fit driven perimeter ellipse");
        let angle_error = {
            let mut difference = (fitted.angle - expected_angle).rem_euclid(PI);
            if difference > PI * 0.5 {
                difference -= PI;
            }
            difference.abs()
        };
        assert!(
            (fitted.center.0 - expected_center.0).hypot(fitted.center.1 - expected_center.1) < 0.55,
            "fitted={fitted:?}"
        );
        assert!(
            (fitted.major_radius - expected_major).abs() < 0.75,
            "fitted={fitted:?}"
        );
        assert!(
            (fitted.minor_radius - expected_minor).abs() < 0.75,
            "fitted={fitted:?}"
        );
        assert!(
            angle_error < 0.02,
            "angle_error={angle_error} fitted={fitted:?}"
        );
    }

    #[test]
    fn lower_lid_bridge_requires_bilateral_measured_rejoins_and_marks_non_voting_rows() {
        let row_count = 96usize;
        let search = OuterSearchEllipse {
            center: (104.0, 76.0),
            major_radius: 52.0,
            minor_radius: 44.0,
            angle: 0.0,
        };
        let path = vec![0usize; row_count];
        let offsets = [0.0];
        let mut measurements = vec![(850.0, 285.0); row_count];
        let mut headings = vec![1.0; row_count];
        let mut opposite_sclera = vec![1.0; row_count];

        // The bottom road is hidden. Strong, multi-row lateral measurements
        // at rows 10 and 38 bracket it on opposing sides of the ellipse.
        for row in 20..=28 {
            measurements[row] = (290.0, 285.0);
            headings[row] = 0.0;
            opposite_sclera[row] = 0.0;
        }
        for row in (8..=12).chain(36..=40) {
            measurements[row] = (1_000.0, 200.0);
            headings[row] = 1.5;
            opposite_sclera[row] = 1.5;
        }

        let bridge = lower_limbus_occlusion_bridge(
            search,
            &path,
            &offsets,
            &measurements,
            &headings,
            &opposite_sclera,
            0.75,
        );
        let (start, end) = bridge.anchor_rows.expect("bilateral measured rejoins");
        assert_eq!(bridge.rejoined, Some(true));
        assert!(bridge.inferred_rows[24], "bottom row must be projected");
        assert!(
            !bridge.inferred_rows[start],
            "first rejoin remains measured"
        );
        assert!(!bridge.inferred_rows[end], "second rejoin remains measured");
        assert!(
            bridge.inferred_rows[16] && bridge.inferred_rows[32],
            "the closure lap must initially carry the full bracketed interval: {bridge:?}",
        );
        let reconfirmed = conic_projected_limbus_reconfirmations(
            &bridge.inferred_rows,
            &path,
            &offsets,
            &measurements,
            &headings,
            0.75,
        );
        assert!(
            reconfirmed[16] && reconfirmed[32],
            "sustained direct road on both sides must be re-confirmed",
        );
        assert!(
            !reconfirmed[24],
            "the genuinely hidden center must remain projected/non-voting",
        );
        assert!(
            bridge.inferred_rows.iter().filter(|row| **row).count() >= 8,
            "bridge={bridge:?}",
        );
    }

    #[test]
    fn lower_lid_bridge_keeps_reviewed_five_oclock_sclera_directly_measured() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../outputs/raw-focus-hotkey/subject-right-focus-0562-1786917874-518201484.raw10",
        );
        let payload = std::fs::read(&path).unwrap_or_else(|error| {
            panic!("read reviewed lossless RAW {}: {error}", path.display())
        });
        let raw = unpack_test_raw10(&payload);
        // Exact full-resolution Rust result exercised by the reviewed-series
        // integration test.  The 5-o'clock arc has clear outside sclera and a
        // long ordered transition even though several far-side sightlines are
        // weak.  Those rows are observations, not an eyelid extrapolation.
        let outer = OuterIrisBoundary {
            center: (216.72718945505565, 104.04217275438349),
            major_radius: 105.01283051854729,
            minor_radius: 88.34502325219559,
            angle: -2.6337313718609257,
            ..OuterIrisBoundary::default()
        };
        let pupil = (201.334677765, 144.769831329);
        let strip = debug_drive_limbus_perimeter_strip_with_pupil(
            &raw, WIDTH, HEIGHT, &outer, pupil, 5.0, 96,
        )
        .expect("reviewed pupil-anchored perimeter drive");
        assert_eq!(
            strip.lower_occlusion_rejoined,
            Some(true),
            "the lower-lid candidate must complete a bounded conic-closure lap",
        );
        let five_oclock = strip
            .samples
            .iter()
            .filter(|sample| (3.59..=4.13).contains(&sample.phase))
            .collect::<Vec<_>>();
        assert!(
            five_oclock.len() >= 8,
            "expected at least eight native samples around 5 o'clock; got {}",
            five_oclock.len(),
        );
        assert!(
            five_oclock.iter().all(|sample| {
                sample.transition_score >= 0.18 && sample.outside_luma > sample.inside_luma + 32.0
            }),
            "fixture stopped exposing the reviewed clear-sclera road: {five_oclock:?}",
        );
        assert!(
            five_oclock.iter().all(|sample| !sample.inferred_occlusion),
            "the visible 5-o'clock road was painted projected/non-voting: {five_oclock:?}",
        );
        assert!(
            (55..=63).all(|row| { strip.conic_projected_rows[row] && strip.reconfirmed_rows[row] }),
            "the initially projected 5-o'clock road was not independently re-confirmed",
        );
        let inferred = strip.inferred_rows.iter().filter(|row| **row).count();
        assert!(
            (3..=20).contains(&inferred),
            "at most a short genuinely broken subsection may be projected; inferred={inferred}",
        );
        assert!(
            strip.second_lap_revisited_rows[0],
            "the arbitrary first row retained privileged anchor status",
        );
        assert!(
            strip.full_chord_flyby_rows[0],
            "the coherent full-chord fly-bys did not challenge the off-eye start",
        );
        let first = &strip.samples[0];
        let center_column = (strip.width - 1) as f64 * 0.5;
        let outside_extent = strip.nominal_mm_per_side * strip.pixels_per_mm;
        let column_scale = outside_extent * 2.0 / (strip.width - 1) as f64;
        let first_offset = (center_column - strip.first_lap_boundary_columns[0]) * column_scale;
        let first_point = OuterIrisPoint {
            x: first.base_point.0 + first.outward_normal.0 * first_offset,
            y: first.base_point.1 + first.outward_normal.1 * first_offset,
            contrast: 1.0,
        };
        let final_point = OuterIrisPoint {
            x: first.driven_point.0,
            y: first.driven_point.1,
            contrast: 1.0,
        };
        let reviewed_reference = [
            215.49966430664062,
            120.35222625732422,
            103.78759002685547,
            96.19979858398438,
            3.417328097057218,
        ];
        let first_residual = outer_ellipse_point_residual(first_point, reviewed_reference);
        let final_residual = outer_ellipse_point_residual(final_point, reviewed_reference);
        assert!(
            final_residual + 5.0 < first_residual && final_residual < 5.5,
            "cyclic re-entry did not leave the wrong starting road: first={first_residual:.3}px final={final_residual:.3}px sample={first:?}",
        );
        assert!(
            first.opposite_sclera_score >= 0.10,
            "the repaired start still lacks a complete pupilward chord: {first:?}",
        );
    }

    #[test]
    fn pupil_heading_partial_second_lap_reconsiders_a_lower_lid_fork() {
        const TEST_WIDTH: usize = 208;
        const TEST_HEIGHT: usize = 168;
        let center = (104.0, 76.0);
        let major = 52.0;
        let minor = 44.0;
        let pupil = (112.0, 76.0);
        let mut raw = vec![850u16; TEST_WIDTH * TEST_HEIGHT];
        for y in 0..TEST_HEIGHT {
            for x in 0..TEST_WIDTH {
                let normalized = ((x as f64 - center.0) / major).powi(2)
                    + ((y as f64 - center.1) / minor).powi(2);
                if normalized <= 1.0 {
                    raw[y * TEST_WIDTH + x] = 285;
                }
                if (x as f64 - pupil.0).hypot(y as f64 - pupil.1) <= 15.0 {
                    raw[y * TEST_WIDTH + x] = 65;
                }
            }
        }
        // A bright lower lid makes a strong local iris-to-skin edge eight
        // pixels inside the true limbus. This ideal fixture leaves the true
        // rim measurable, so the pupil-heading lap should return to that
        // direct road. Projected/non-voting occlusion bridging is tested
        // independently above with explicit bilateral rejoin evidence.
        for y in 111..TEST_HEIGHT {
            for x in 62..=158 {
                raw[y * TEST_WIDTH + x] = 900;
            }
        }
        let outer = OuterIrisBoundary {
            center,
            major_radius: major,
            minor_radius: minor,
            angle: 0.0,
            ..OuterIrisBoundary::default()
        };
        let strip = debug_drive_limbus_perimeter_strip_with_pupil(
            &raw,
            TEST_WIDTH,
            TEST_HEIGHT,
            &outer,
            pupil,
            5.0,
            96,
        )
        .expect("pupil-anchored perimeter drive");
        let changed = strip
            .first_lap_boundary_columns
            .iter()
            .zip(&strip.boundary_columns)
            .filter(|(first, second)| (*first - *second).abs() > 0.25)
            .count();
        let revisited = strip.revisited_rows.iter().filter(|row| **row).count();
        let inferred = strip.inferred_rows.iter().filter(|row| **row).count();
        let center_column = (strip.width - 1) as f64 * 0.5;
        let first_departure = strip
            .first_lap_boundary_columns
            .iter()
            .zip(&strip.revisited_rows)
            .filter(|(_, revisit)| **revisit)
            .map(|(column, _)| (column - center_column).abs())
            .sum::<f64>();
        let second_departure = strip
            .boundary_columns
            .iter()
            .zip(&strip.revisited_rows)
            .filter(|(_, revisit)| **revisit)
            .map(|(column, _)| (column - center_column).abs())
            .sum::<f64>();
        let offset_span = strip
            .samples
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |span, sample| {
                (span.0.min(sample.offset_px), span.1.max(sample.offset_px))
            });
        let heading_span =
            strip
                .samples
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |span, sample| {
                    (
                        span.0.min(sample.pupil_heading_score),
                        span.1.max(sample.pupil_heading_score),
                    )
                });
        let opposite_sclera = strip
            .samples
            .iter()
            .filter(|sample| sample.opposite_sclera_score >= 0.50)
            .count();
        let affine_departures = strip
            .affine_ellipse_residual_threshold_mm
            .map(|threshold| {
                strip
                    .samples
                    .iter()
                    .filter(|sample| sample.affine_ellipse_residual_mm > threshold)
                    .count()
            })
            .unwrap_or(0);
        eprintln!(
            "synthetic lower-lid fork: laps={} revisited={revisited} inferred={inferred} rejoined={:?} anchors={:?} changed={changed} offsets={offset_span:?} heading={heading_span:?} opposite-sclera={opposite_sclera}/{} affine-departures={affine_departures} rms={:?}->{:?}",
            strip.lap_count,
            strip.lower_occlusion_rejoined,
            strip.lower_occlusion_anchor_rows,
            strip.samples.len(),
            strip.affine_ellipse_first_rms_mm,
            strip.affine_ellipse_final_rms_mm,
        );
        assert_eq!(strip.lap_count, 2, "strip={strip:?}");
        assert!(revisited >= 6, "strip={strip:?}");
        assert_eq!(strip.lower_occlusion_rejoined, None, "strip={strip:?}");
        assert_eq!(
            inferred, 0,
            "directly measured fixture should not be projected"
        );
        assert!(changed >= 1, "strip={strip:?}");
        assert!(
            opposite_sclera * 2 >= strip.samples.len(),
            "the forward road did not exit into opposite sclera: {strip:?}"
        );
        assert!(
            second_departure < first_departure,
            "first={first_departure} second={second_departure} strip={strip:?}"
        );
        assert!(
            strip
                .affine_ellipse_departure_first_rms_mm
                .zip(strip.affine_ellipse_departure_final_rms_mm)
                .is_some_and(|(first, final_value)| final_value <= first + 1.0e-6),
            "the repair did not pull the suspect arc toward the projected circle: {strip:?}"
        );
    }

    #[test]
    fn driving_pupil_converges_on_ten_lossless_raw_rois() {
        for (fixture_index, fixture) in DRIVING_RAW_FIXTURES.into_iter().enumerate() {
            let raw = driving_fixture_raw(fixture.pair);
            let coarse = BorderFocus {
                eye_basin_valid: true,
                center: fixture.center,
                radius: fixture.radius,
                axis_ratio: fixture.axis_ratio,
                axis_angle: fixture.axis_angle,
                ..BorderFocus::default()
            };
            let drive = debug_drive_pupil_center(&raw, WIDTH, HEIGHT, 4072, 4730, &coarse)
                .unwrap_or_else(|| panic!("pair-{:02}: no pupil drive", fixture.pair));
            let boundary = detect_inner_iris_boundary(&raw, WIDTH, HEIGHT, 4072, 4730, &coarse);
            let normalized_offset = (drive.center.0 - coarse.center.0)
                .hypot(drive.center.1 - coarse.center.1)
                / coarse.radius;
            eprintln!(
                "pair-{:02} start=({:.1},{:.1}) pupil=({:.1},{:.1}) steps={} travel={:.1}px rho={:.3} enclosure={:.3} inner_points={} inner_radius={:.1}",
                fixture.pair,
                drive.start.0,
                drive.start.1,
                drive.center.0,
                drive.center.1,
                drive.trace.len().saturating_sub(1),
                drive.travel_px,
                normalized_offset,
                drive.enclosure_score,
                boundary.points.len(),
                boundary.radius,
            );
            assert!(
                drive.trace.len() <= 5,
                "pair-{:02}: drive took too many steps: {drive:?}",
                fixture.pair
            );
            assert!(
                normalized_offset <= 0.72,
                "pair-{:02}: pupil escaped limbus search: {drive:?}",
                fixture.pair
            );
            assert!(
                drive.enclosure_score >= 0.08,
                "pair-{:02}: dark basin is not enclosed: {drive:?}",
                fixture.pair
            );
            assert!(
                (drive.center.0 - DRIVING_EXPECTED_PUPILS[fixture_index].0)
                    .hypot(drive.center.1 - DRIVING_EXPECTED_PUPILS[fixture_index].1)
                    <= 12.0,
                "pair-{:02}: drove away from reviewed pupil {:?}: {drive:?}",
                fixture.pair,
                DRIVING_EXPECTED_PUPILS[fixture_index],
            );
            assert_eq!(
                boundary.points.len(),
                21,
                "pair-{:02}: pupil rim did not close: {boundary:?}",
                fixture.pair
            );
        }
    }

    #[test]
    fn driving_pupil_geometry_resolves_one_native_raw_pixel() {
        const TEST_WIDTH: usize = 128;
        const TEST_HEIGHT: usize = 96;
        let acquire = |pupil_x: f64| {
            let pupil_center = (pupil_x, 53.0);
            let mut raw = vec![0u16; TEST_WIDTH * TEST_HEIGHT];
            for y in 0..TEST_HEIGHT {
                for x in 0..TEST_WIDTH {
                    let iris_radius = (x as f64 - 64.0).hypot(y as f64 - 48.0);
                    let pupil_radius = (x as f64 - pupil_center.0).hypot(y as f64 - pupil_center.1);
                    let base = if pupil_radius <= 10.0 {
                        90.0
                    } else if iris_radius <= 32.0 {
                        410.0
                    } else {
                        780.0
                    };
                    raw[y * TEST_WIDTH + x] =
                        (base + ((x * 13 + y * 7) % 11) as f64).round() as u16;
                }
            }
            let coarse = BorderFocus {
                eye_basin_valid: true,
                center: (64.0, 48.0),
                radius: 32.0,
                axis_ratio: 1.0,
                axis_angle: 0.0,
                ..BorderFocus::default()
            };
            debug_drive_pupil_center(&raw, TEST_WIDTH, TEST_HEIGHT, 0, 0, &coarse)
                .expect("full-resolution pupil drive")
                .center
        };

        let first = acquire(61.0);
        let second = acquire(62.0);
        assert!((first.0 - 61.0).abs() <= 1.5, "first={first:?}");
        assert!((second.0 - 62.0).abs() <= 1.5, "second={second:?}");
        assert!(
            second.0 - first.0 >= 0.60,
            "one native RAW pixel was quantized away: first={first:?} second={second:?}"
        );
    }

    #[test]
    fn driving_limbus_perimeter_stays_between_guides_on_ten_lossless_raw_rois() {
        for (fixture_index, fixture) in DRIVING_RAW_FIXTURES.into_iter().enumerate() {
            let raw = driving_fixture_raw(fixture.pair);
            let [center_x, center_y, major_radius, minor_radius, angle] =
                DRIVING_REVIEW_LIMBUS[fixture_index];
            let outer = OuterIrisBoundary {
                center: (center_x, center_y),
                major_radius,
                minor_radius,
                angle,
                ..OuterIrisBoundary::default()
            };
            let strip = debug_drive_limbus_perimeter_strip(&raw, WIDTH, HEIGHT, &outer, 5.0, 96)
                .unwrap_or_else(|| panic!("pair-{:02}: no perimeter strip", fixture.pair));
            let inside_guides = strip
                .boundary_columns
                .iter()
                .filter(|column| {
                    **column >= strip.guide_left_column as f64
                        && **column <= strip.guide_right_column as f64
                })
                .count();
            let positive_transitions = strip
                .samples
                .iter()
                .filter(|sample| sample.transition_score > 0.0)
                .count();
            let mean_transition = strip
                .samples
                .iter()
                .map(|sample| sample.transition_score)
                .sum::<f64>()
                / strip.samples.len() as f64;
            eprintln!(
                "pair-{:02} strip={}x{} px/mm={:.2} guides={}-{} in={}/{} positive={}/{} mean_step={:.3}",
                fixture.pair,
                strip.width,
                strip.height,
                strip.pixels_per_mm,
                strip.guide_left_column,
                strip.guide_right_column,
                inside_guides,
                strip.height,
                positive_transitions,
                strip.height,
                mean_transition,
            );
            assert_eq!(strip.luma.len(), strip.width * strip.height);
            assert_eq!(strip.height, 96);
            assert!(strip.guide_left_column < strip.guide_right_column);
            assert!(
                inside_guides * 4 >= strip.height * 3,
                "pair-{:02}: limbus escaped guide corridor: {strip:?}",
                fixture.pair
            );
            assert!(
                positive_transitions * 5 >= strip.height * 3,
                "pair-{:02}: ordered sclera/iris transition missing: {strip:?}",
                fixture.pair
            );
            assert!(
                mean_transition > 0.02,
                "pair-{:02}: mean sclera-to-iris step too weak: {mean_transition:.3}",
                fixture.pair
            );
        }
    }

    fn synthetic_eyelid() -> Vec<u16> {
        let mut raw = vec![760u16; WIDTH * HEIGHT];
        for y in 128..174 {
            for x in 30..354 {
                raw[y * WIDTH + x] = 180 + ((x + y) % 23) as u16;
            }
        }
        raw
    }

    #[test]
    fn sharp_eye_scores_above_blurred_eye() {
        let sharp = score_stream_eye(&synthetic_eye(194.0, 148.0, 2.0), WIDTH, HEIGHT);
        let blurred = score_stream_eye(&synthetic_eye(194.0, 148.0, 16.0), WIDTH, HEIGHT);
        assert!(sharp.eye_basin_valid, "sharp result: {sharp:?}");
        assert!(blurred.eye_basin_valid, "blurred result: {blurred:?}");
        assert!(sharp.points.len() >= 10, "sharp result: {sharp:?}");
        assert!(blurred.points.len() >= 10, "blurred result: {blurred:?}");
        assert!(
            sharp.optical_sharpness > blurred.optical_sharpness,
            "sharp ratio={:.4} blurred ratio={:.4}",
            sharp.optical_sharpness,
            blurred.optical_sharpness,
        );
        assert!(
            sharp.score > blurred.score * 1.20,
            "sharp={:.3} blurred={:.3}",
            sharp.score,
            blurred.score
        );
    }

    #[test]
    fn selected_limbus_optical_focus_ranks_sharp_and_blurred_native_raw() {
        let outer = OuterIrisBoundary {
            center: (194.0, 148.0),
            major_radius: 70.0,
            minor_radius: 70.0,
            angle: 0.0,
            points: vec![OuterIrisPoint::default(); 8],
            ..OuterIrisBoundary::default()
        };
        let sharp =
            measure_limbus_optical_focus(&synthetic_eye(194.0, 148.0, 2.0), WIDTH, HEIGHT, &outer);
        let blurred =
            measure_limbus_optical_focus(&synthetic_eye(194.0, 148.0, 16.0), WIDTH, HEIGHT, &outer);
        assert!(sharp.support >= 12, "sharp={sharp:?}");
        assert!(blurred.support >= 12, "blurred={blurred:?}");
        assert!(
            sharp.sharpness > blurred.sharpness * 1.15,
            "sharp={sharp:?} blurred={blurred:?}",
        );
    }

    #[test]
    fn provisional_focus_ranks_blur_without_admitting_anatomy() {
        let sharp = synthetic_eye(194.0, 148.0, 2.0);
        let blurred = synthetic_eye(194.0, 148.0, 16.0);
        let sharp_score = provisional_focus_score(&sharp, WIDTH, HEIGHT);
        let blurred_score = provisional_focus_score(&blurred, WIDTH, HEIGHT);
        assert!(
            sharp_score > blurred_score * 1.15,
            "sharp={sharp_score:.3} blurred={blurred_score:.3}"
        );
    }

    #[test]
    fn provisional_focus_rejects_a_flat_crop() {
        assert_eq!(
            provisional_focus_score(&vec![416u16; WIDTH * HEIGHT], WIDTH, HEIGHT),
            0.0
        );
    }

    #[test]
    fn provisional_focus_does_not_rank_sensor_noise_above_an_eye() {
        let sharp = synthetic_eye(194.0, 148.0, 2.0);
        let mut state = 0x9e37_79b9u32;
        let noise = (0..WIDTH * HEIGHT)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                480u16.saturating_add((state % 97) as u16)
            })
            .collect::<Vec<_>>();
        let sharp_score = provisional_focus_score(&sharp, WIDTH, HEIGHT);
        let noise_score = provisional_focus_score(&noise, WIDTH, HEIGHT);
        assert!(
            noise_score < sharp_score * 0.5,
            "sharp={sharp_score:.3} noise={noise_score:.3}"
        );
    }

    #[test]
    fn long_dark_eyelid_is_not_focus_evidence() {
        let result = score_stream_eye(&synthetic_eyelid(), WIDTH, HEIGHT);
        assert!(!result.eye_basin_valid, "eyelid result: {result:?}");
        assert_eq!(result.score, 0.0, "eyelid result: {result:?}");
        assert!(result.points.is_empty(), "eyelid result: {result:?}");
    }

    #[test]
    fn translated_eye_remains_valid() {
        let center = score_stream_eye(&synthetic_eye(194.0, 148.0, 3.0), WIDTH, HEIGHT);
        let translated = score_stream_eye(&synthetic_eye(150.0, 176.0, 3.0), WIDTH, HEIGHT);
        assert!(center.points.len() >= 10, "center result: {center:?}");
        assert!(
            translated.points.len() >= 10,
            "translated result: {translated:?}"
        );
        let ratio = translated.score / center.score;
        assert!(
            (0.80..=1.20).contains(&ratio),
            "translation score ratio={ratio:.3}"
        );
    }

    #[test]
    fn smaller_supported_eye_remains_valid() {
        let result = score_stream_eye(
            &synthetic_eye_with_radius(194.0, 148.0, 2.0, 64.0),
            WIDTH,
            HEIGHT,
        );
        assert!(result.eye_basin_valid, "scaled result: {result:?}");
        assert!(result.points.len() >= 10, "scaled result: {result:?}");
    }

    #[test]
    fn iris_light_map_normalizes_uniform_exposure() {
        let raw = vec![416u16; WIDTH * HEIGHT];
        let focus = BorderFocus {
            center: (WIDTH as f64 * 0.5, HEIGHT as f64 * 0.5),
            radius: 64.0,
            ..BorderFocus::default()
        };
        let map = iris_light_map(&raw, WIDTH, HEIGHT, &focus);
        assert!(map.valid, "map={map:?}");
        assert!((map.mean - 416.0).abs() < 0.01, "map={map:?}");
        assert!(map.span < 1.0e-9, "map={map:?}");
        assert!(map.gradient_x.abs() < 1.0e-9, "map={map:?}");
        assert!(map.gradient_y.abs() < 1.0e-9, "map={map:?}");
        assert!(map.cells.iter().all(|value| (*value - 1.0).abs() < 1.0e-9));
    }

    #[test]
    fn iris_light_map_retains_directional_light_field() {
        let mut raw = vec![0u16; WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                raw[y * WIDTH + x] = (180 + x) as u16;
            }
        }
        let focus = BorderFocus {
            center: (WIDTH as f64 * 0.5, HEIGHT as f64 * 0.5),
            radius: 64.0,
            ..BorderFocus::default()
        };
        let map = iris_light_map(&raw, WIDTH, HEIGHT, &focus);
        assert!(map.valid, "map={map:?}");
        assert!(map.gradient_x > 0.05, "map={map:?}");
        assert!(map.gradient_y.abs() < 0.01, "map={map:?}");
    }

    #[test]
    fn dark_basin_recovers_projected_iris_axis_ratio() {
        let width = 96usize;
        let height = 64usize;
        let mut mask = vec![false; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = (x as f64 - 48.0) / 18.0;
                let dy = (y as f64 - 36.0) / 10.0;
                mask[y * width + x] = dx * dx + dy * dy <= 1.0;
            }
        }
        let basin = largest_eye_basin(&mask, width, height).expect("ellipse basin");
        assert!((1.6..=2.0).contains(&basin.axis_ratio), "basin={basin:?}");
        assert!(basin.axis_angle.abs() < 0.05, "basin={basin:?}");
    }

    #[test]
    fn radial_sclera_gate_rejects_a_weak_cavity() {
        let width = 96usize;
        let height = 64usize;
        let basin = DarkBasin {
            center: (48.0, 36.0),
            radius: 14.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
        };
        let mut plane = vec![90u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - basin.center.0;
                let dy = y as f64 - basin.center.1;
                let radius = (dx * dx + dy * dy).sqrt() / basin.radius;
                if (0.60..=0.90).contains(&radius) {
                    plane[y * width + x] = 45;
                } else if (1.10..=1.50).contains(&radius) {
                    plane[y * width + x] = 55;
                }
            }
        }
        assert!(!eye_basin_has_lateral_sclera(&plane, width, height, basin));

        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - basin.center.0;
                let dy = y as f64 - basin.center.1;
                let radius = (dx * dx + dy * dy).sqrt() / basin.radius;
                if (1.10..=1.50).contains(&radius) {
                    plane[y * width + x] = 90;
                }
            }
        }
        assert!(eye_basin_has_lateral_sclera(&plane, width, height, basin));
    }
}

fn darkest_center(raw: &[u16], width: usize, height: usize) -> Option<(f64, f64)> {
    // Reduce first so Bayer alternation, eyelashes, and single-pixel glints do
    // not win the pupil search. Four sensor pixels per output dimension keeps
    // the pupil several samples wide while making the search very cheap.
    let scale = 4usize;
    let reduced_width = width / scale;
    let reduced_height = height / scale;
    if reduced_width < 16 || reduced_height < 12 {
        return None;
    }
    let mut reduced = vec![0.0; reduced_width * reduced_height];
    let mut reduced_chroma = vec![0.0; reduced_width * reduced_height];
    for ry in 0..reduced_height {
        for rx in 0..reduced_width {
            let mut sum = 0u32;
            let mut phases = [0u32; 4];
            let mut phase_counts = [0u32; 4];
            for dy in 0..scale {
                for dx in 0..scale {
                    let value = raw[(ry * scale + dy) * width + rx * scale + dx] as u32;
                    sum += value;
                    let phase = (dy & 1) * 2 + (dx & 1);
                    phases[phase] += value;
                    phase_counts[phase] += 1;
                }
            }
            let index = ry * reduced_width + rx;
            reduced[index] = sum as f64 / (scale * scale) as f64;
            let phase_means = phases
                .iter()
                .zip(phase_counts)
                .map(|(value, count)| *value as f64 / count.max(1) as f64)
                .collect::<Vec<_>>();
            let minimum = phase_means.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = phase_means.iter().copied().fold(0.0, f64::max);
            reduced_chroma[index] = (maximum - minimum) * 100.0 / (reduced[index] + 1.0);
        }
    }
    // Search the 4x Bayer-cell reduction directly. There is deliberately no
    // Gaussian or other spatial blur in the pupil-localization path.
    let blurred = reduced.clone();
    let mut best = (width as f64 * 0.5, height as f64 * 0.5);
    let mut best_cost = f64::INFINITY;
    let mut sorted_luma = blurred.clone();
    sorted_luma.sort_by(f64::total_cmp);
    let dark_limit = sorted_luma[sorted_luma.len() / 4];
    let mut sorted_chroma = reduced_chroma.clone();
    sorted_chroma.sort_by(f64::total_cmp);
    let iris_chroma_floor = sorted_chroma[sorted_chroma.len() * 2 / 5].max(2.0);
    // Two reduced pixels are eight physical pixels: still an extremely sparse
    // search in the original RAW plane.
    for y in (reduced_height * 2 / 10..reduced_height * 8 / 10).step_by(2) {
        for x in (reduced_width * 2 / 10..reduced_width * 8 / 10).step_by(2) {
            let dark = blurred[y * reduced_width + x];
            if dark > dark_limit {
                continue;
            }
            // A pupil is a compact dark void surrounded by brighter iris.
            // Reward that local topology so a broad eyelid/glasses shadow is
            // not selected merely because its absolute RAW value is lower.
            let mut ring = Vec::with_capacity(16);
            let mut chromatic_directions = 0usize;
            for step in 0..16 {
                let angle = step as f64 * 2.0 * PI / 16.0;
                let sx = (x as f64 + angle.cos() * 8.0)
                    .round()
                    .clamp(0.0, reduced_width as f64 - 1.0) as usize;
                let sy = (y as f64 + angle.sin() * 8.0)
                    .round()
                    .clamp(0.0, reduced_height as f64 - 1.0) as usize;
                ring.push(blurred[sy * reduced_width + sx]);
                if reduced_chroma[sy * reduced_width + sx] >= iris_chroma_floor {
                    chromatic_directions += 1;
                }
            }
            ring.sort_by(f64::total_cmp);
            let surround = ring[ring.len() / 2];
            let ring_lower_quartile = ring[ring.len() / 4];
            let bright_directions = ring.iter().filter(|value| **value >= dark + 6.0).count();
            // Require an enclosed dark basin. An eyelash is line-shaped: its
            // surrounding ring remains dark in the two directions following
            // the lash and therefore fails this majority/all-quadrant test.
            if bright_directions < 10
                || chromatic_directions < 8
                || ring_lower_quartile < dark + 5.0
            {
                continue;
            }
            let compact_void_reward = (surround - dark).max(0.0);
            let nx = (x as f64 - reduced_width as f64 * 0.5) / (reduced_width as f64 * 0.5);
            let ny = (y as f64 - reduced_height as f64 * 0.5) / (reduced_height as f64 * 0.5);
            let cost = dark - compact_void_reward * 0.75 + 40.0 * (nx * nx + ny * ny);
            if cost < best_cost {
                best_cost = cost;
                best = (
                    (x * scale + scale / 2) as f64,
                    (y * scale + scale / 2) as f64,
                );
            }
        }
    }
    best_cost.is_finite().then_some(best)
}

fn radial_candidate(
    raw: &[u16],
    width: usize,
    height: usize,
    center: (f64, f64),
    angle: f64,
    radius: f64,
) -> Option<Candidate> {
    let (sin, cos) = angle.sin_cos();
    let at = |offset: f64| {
        cfa_luma(
            raw,
            width,
            height,
            center.0 + (radius + offset) * cos,
            center.1 + (radius + offset) * sin,
        )
    };
    let inside = (at(-8.0) + at(-6.0) + at(-4.0)) / 3.0;
    let outside = (at(4.0) + at(6.0) + at(8.0)) / 3.0;
    let contrast = outside - inside;
    if contrast < 18.0 || outside > 1005.0 || at(0.0) > 1008.0 {
        return None;
    }
    let narrow = (at(2.0) - at(-2.0)).abs();
    let wide = contrast.abs().max(1.0);
    let sharpness = (narrow / wide).clamp(0.0, 1.5);
    let px = center.0 + radius * cos;
    let py = center.1 + radius * sin;
    let tangent = 3.0;
    let left = sample(raw, width, height, px - sin * tangent, py + cos * tangent);
    let right = sample(raw, width, height, px + sin * tangent, py - cos * tangent);
    let tangent_penalty = 1.0 / (1.0 + (left - right).abs() / wide);
    let quality = contrast * sharpness * tangent_penalty;
    Some(Candidate {
        angle,
        radius,
        x: px.round().clamp(0.0, width.saturating_sub(1) as f64) as usize,
        y: py.round().clamp(0.0, height.saturating_sub(1) as f64) as usize,
        contrast,
        sharpness,
        quality,
    })
}

// Score fine, vessel-sized structure in bright sclera immediately outside the
// iris.  All samples are complete Bayer-cell averages and all derivatives use
// even offsets, so CFA alternation cannot masquerade as a sharp blood vessel.
// A narrow/broad Laplacian ratio rejects the broad iris boundary, eyelid
// shadows, and illumination gradients while retaining thin vessel edges.
fn sclera_vessel_focus(
    raw: &[u16],
    width: usize,
    height: usize,
    center: (f64, f64),
    iris_radius: f64,
    nasal_direction: i32,
    filter: FocusFilter,
    filter_level: f64,
) -> (f64, Vec<BorderPoint>, Option<(f64, f64)>) {
    let mut luma = Vec::with_capacity((width / 2) * (height / 2));
    for y in (2..height.saturating_sub(2)).step_by(2) {
        for x in (2..width.saturating_sub(2)).step_by(2) {
            luma.push(cfa_luma(raw, width, height, x as f64, y as f64));
        }
    }
    if luma.len() < 32 {
        return (0.0, Vec::new(), None);
    }
    luma.sort_by(f64::total_cmp);
    // Sclera need not be the absolute brightest material in the crop (a lens
    // glint often is), but it should belong to the bright third of the eye.
    let bright_floor = luma[luma.len() * 2 / 3];
    let mut features = Vec::new();
    let inner = (iris_radius * 1.15).max(18.0);
    let outer = (iris_radius * 3.2).min(width as f64 * 0.42);
    let vertical = (iris_radius * 0.95).max(18.0);
    let near_offset = 2isize;
    let far_offset = 8isize;
    let filter_level = filter_level.clamp(0.25, 4.0);
    for y in (10..height.saturating_sub(10)).step_by(2) {
        let dy = y as f64 - center.1;
        if dy.abs() > vertical {
            continue;
        }
        for x in (10..width.saturating_sub(10)).step_by(2) {
            let dx = x as f64 - center.0;
            // Search only the nasal side: toward image center and the nose.
            // The caller supplies +1 for the image-left/subject-right eye and
            // -1 for the image-right/subject-left eye.
            if dx * (nasal_direction as f64) < inner || dx.abs() > outer {
                continue;
            }
            let luma_at = |ox: isize, oy: isize| {
                cfa_luma(
                    raw,
                    width,
                    height,
                    (x as isize + ox) as f64,
                    (y as isize + oy) as f64,
                )
            };
            let at = |ox: isize, oy: isize| {
                cfa_detail(
                    raw,
                    width,
                    height,
                    (x as isize + ox) as f64,
                    (y as isize + oy) as f64,
                    filter,
                )
            };
            let c = at(0, 0);
            let near_samples = [
                at(-near_offset, 0),
                at(near_offset, 0),
                at(0, -near_offset),
                at(0, near_offset),
            ];
            let far_samples = [
                at(-far_offset, 0),
                at(far_offset, 0),
                at(0, -far_offset),
                at(0, far_offset),
            ];
            let near_mean = near_samples.iter().sum::<f64>() * 0.25;
            let far_mean = far_samples.iter().sum::<f64>() * 0.25;
            // Use the surrounding white background for eligibility so the
            // darker vessel pixel itself is not rejected as "not sclera".
            let background = [luma_at(-8, 0), luma_at(8, 0), luma_at(0, -8), luma_at(0, 8)]
                .iter()
                .sum::<f64>()
                * 0.25;
            if background < bright_floor || background > 1008.0 {
                continue;
            }
            // Require a broad bright field around the candidate. A lash,
            // eyebrow, spectacle rim, or nose boundary has dark samples on
            // one or both sides; a thin vessel is embedded in bright sclera.
            let bright_surround = [luma_at(-8, 0), luma_at(8, 0), luma_at(0, -8), luma_at(0, 8)]
                .iter()
                .filter(|value| **value >= bright_floor * 0.94)
                .count();
            if bright_surround < 3 {
                continue;
            }
            let fine = if filter == FocusFilter::LumaBandPass {
                (near_mean - far_mean).abs()
            } else {
                (c - near_mean).abs()
            };
            let broad = (c - far_mean).abs();
            let vessel_darkness = if filter == FocusFilter::RedGreenHighPass {
                fine
            } else {
                (near_mean.max(far_mean) - c).max(0.0)
            };
            // Match what remains visually meaningful in the useful 15%
            // contrast diagnostic view: discard weak RAW texture/noise and
            // rank only definite dark canthus/vessel detail.
            if fine < 4.0 * filter_level || vessel_darkness < 2.0 * filter_level {
                continue;
            }
            let scale_selectivity = fine / (fine + broad + 1.0);
            let normalized =
                100.0 * (fine + vessel_darkness * 0.5) * scale_selectivity / (background + 16.0);
            if normalized >= 0.25 * filter_level {
                features.push(BorderPoint {
                    x,
                    y,
                    quality: normalized,
                });
            }
        }
    }
    if features.len() < 10 {
        return (0.0, Vec::new(), None);
    }
    // Keep a bounded set of the strongest spatial samples. The median of this
    // upper population is stable against an occasional glint or hot pixel.
    features.sort_by(|a, b| b.quality.total_cmp(&a.quality));
    features.truncate(48);
    let mut qualities = features
        .iter()
        .map(|point| point.quality)
        .collect::<Vec<_>>();
    let vessel_score = median(&mut qualities) * 20.0;
    let mut xs = features
        .iter()
        .map(|point| point.x as f64)
        .collect::<Vec<_>>();
    let mut ys = features
        .iter()
        .map(|point| point.y as f64)
        .collect::<Vec<_>>();
    let focus_center = Some((median(&mut xs), median(&mut ys)));
    (vessel_score, features, focus_center)
}

pub fn score(
    raw: &[u16],
    width: usize,
    height: usize,
    nasal_direction: i32,
    filter: FocusFilter,
    filter_level: f64,
) -> BorderFocus {
    if width < 64 || height < 48 || raw.len() < width * height {
        return BorderFocus::default();
    }
    let Some(center) = darkest_center(raw, width, height) else {
        return BorderFocus::default();
    };
    let min_dim = width.min(height) as f64;
    // The pupil void must remain small relative to the RAW eye slice. The old
    // 12..42% interval commonly reached the outer iris, eyelid, or glasses.
    // This 5..20% interval targets the pupil-to-iris transition and makes a
    // large dark region ineligible by construction.
    let min_radius = (min_dim * 0.05).max(10.0);
    let max_radius = (min_dim * 0.20).min(52.0);
    let mut candidates = Vec::new();
    // Restrict to lateral arcs; upper/lower arcs are dominated by eyelids.
    for sector_center in [0.0, PI] {
        for step in -14..=14 {
            let angle = sector_center + step as f64 * (2.5 * PI / 180.0);
            let mut best: Option<Candidate> = None;
            let mut radius = min_radius;
            while radius <= max_radius {
                if let Some(candidate) = radial_candidate(raw, width, height, center, angle, radius)
                {
                    if best
                        .map(|old| candidate.quality > old.quality)
                        .unwrap_or(true)
                    {
                        best = Some(candidate);
                    }
                }
                radius += 2.0;
            }
            if let Some(candidate) = best {
                candidates.push(candidate);
            }
        }
    }
    if candidates.len() < 10 {
        return BorderFocus {
            center,
            ..BorderFocus::default()
        };
    }
    let mut radii = candidates
        .iter()
        .map(|candidate| candidate.radius)
        .collect::<Vec<_>>();
    let radius = median(&mut radii);
    let mut deviations = candidates
        .iter()
        .map(|candidate| (candidate.radius - radius).abs())
        .collect::<Vec<_>>();
    let mad = median(&mut deviations).max(2.0);
    candidates.retain(|candidate| {
        (candidate.radius - radius).abs() <= (2.5 * mad).min(min_dim * 0.07)
            && candidate.sharpness >= 0.20
            && candidate.contrast >= 18.0
    });
    candidates.sort_by(|a, b| a.angle.total_cmp(&b.angle));
    if candidates.len() < 10 {
        return BorderFocus {
            center,
            radius,
            ..BorderFocus::default()
        };
    }
    let (score, vessel_points, focus_center) = sclera_vessel_focus(
        raw,
        width,
        height,
        center,
        radius,
        nasal_direction.signum(),
        filter,
        filter_level,
    );
    // Vessel-bearing sclera is mandatory autofocus evidence. Never fall back
    // to the pupil/iris boundary: when the crop contains only nose, eyebrow,
    // lashes, or glasses, an empty point set deliberately forces reacquisition.
    let points = if vessel_points.len() >= 10 {
        vessel_points
    } else {
        Vec::new()
    };
    BorderFocus {
        score,
        eye_basin_valid: points.len() >= 10,
        center,
        focus_center,
        radius,
        axis_ratio: 1.0,
        axis_angle: 0.0,
        points,
        ..BorderFocus::default()
    }
}
