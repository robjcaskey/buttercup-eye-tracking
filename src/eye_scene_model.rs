//! Per-eye temporal surface/sign state and approximate eye/scene geometry.
//!
//! Migrated live algorithms, not a calibrated 3D biomechanical model. Projected
//! pivots may drift and carry translation; do not reinterpret them as rigid
//! anatomical hinges. Camera/metric uncertainty and cooperative pivot feedback
//! remain future work. Convex camera-facing output is still a hard invariant.

use crate::geometry::{
    blend_point, dot3, upper_median_or_zero as median_focus, wrapped_angle_distance,
};
use crate::raw_iris_focus;
use crate::roi_evidence::NativeGlobalSimilarityEvidence;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub(crate) mod limbus_scale;
pub(crate) mod binocular_pose;
pub(crate) mod pupil_center;
pub(crate) mod pupil_projection;
pub(crate) mod pupil_size;
pub(crate) mod sign_motion;
mod sign_continuity;

/// Unit-safe radius coordinates for the physical pupil-size posterior.
///
/// These constructors deliberately live behind a private module boundary:
/// image-plane ellipse radii cannot be passed to temporal physical-size state
/// as bare `f64`s. A fronto-parallel radius is produced either by the
/// projected-circle limbus model or by explicitly undoing the limbus affine
/// foreshortening from a projected equal-area pupil radius.
pub(crate) mod pupil_radius_units {
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct ProjectedAreaEquivalentRadiusPx(f64);

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct AffineForeshortening(f64);

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct FrontoParallelCircleRadiusPx(f64);

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct PhysicalRadiusRatio(f64);

    impl ProjectedAreaEquivalentRadiusPx {
        pub(crate) fn from_ellipse_axes(axis_a: f64, axis_b: f64) -> Option<Self> {
            if !axis_a.is_finite() || !axis_b.is_finite() || axis_a <= 0.0 || axis_b <= 0.0 {
                return None;
            }
            let radius = (axis_a * axis_b).sqrt();
            (radius.is_finite() && radius > 0.0).then_some(Self(radius))
        }
    }

    impl AffineForeshortening {
        pub(crate) fn from_minor_to_major(minor_to_major: f64) -> Option<Self> {
            (minor_to_major.is_finite()
                && (crate::conic_solver::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE
                    .absolute_minimum_minor_to_major..=1.0)
                    .contains(&minor_to_major))
            .then_some(Self(minor_to_major))
        }
    }

    impl FrontoParallelCircleRadiusPx {
        /// Under weak perspective, a projected physical circle retains its
        /// true radius along the ellipse's unforeshortened major axis.
        pub(crate) fn from_projected_circular_limbus_axes(
            axis_a: f64,
            axis_b: f64,
        ) -> Option<Self> {
            ProjectedAreaEquivalentRadiusPx::from_ellipse_axes(axis_a, axis_b)?;
            let major = axis_a.max(axis_b);
            let minor = axis_a.min(axis_b);
            AffineForeshortening::from_minor_to_major(minor / major)?;
            // This is the closed-form rectification for a projected circle:
            // sqrt(a*b) / sqrt(b/a) = a. Return the measured major axis
            // directly to avoid introducing roundoff into operator guides.
            Some(Self(major))
        }

        pub(crate) fn from_projected_area(
            projected: ProjectedAreaEquivalentRadiusPx,
            foreshortening: AffineForeshortening,
        ) -> Option<Self> {
            let radius = projected.0 / foreshortening.0.sqrt();
            (radius.is_finite() && radius > 0.0).then_some(Self(radius))
        }

        pub(crate) fn value(self) -> f64 {
            self.0
        }

        pub(crate) fn ratio_to(self, reference: Self) -> Option<PhysicalRadiusRatio> {
            let ratio = self.0 / reference.0;
            (ratio.is_finite() && ratio > 0.0).then_some(PhysicalRadiusRatio(ratio))
        }
    }

    impl PhysicalRadiusRatio {
        pub(crate) fn value(self) -> f64 {
            self.0
        }
    }
}

#[path = "coupled_eye_kinematics.rs"]
mod coupled_eye_kinematics;
pub use coupled_eye_kinematics::{
    CoupledEyeKinematics, CoupledMotionStatus, GlobeMotionRegime, KinematicDerivatives,
    ProjectedGlobePoseStatus, ProjectedIrisGeometry, RotationCenterStatus,
};

pub(crate) const FRONTAL_DISK_AREA_BIN_RATIO: f64 = 1.04;
// Continuity evidence only; never apply this averaging to published gaze.
pub(crate) const GAZE_SURFACE_AVERAGE_ALPHA: f64 = 0.35;
pub(crate) const GAZE_SURFACE_RESET_AFTER: Duration = Duration::from_millis(1_250);
pub(crate) const FRONTAL_DISK_AREA_MAX_BIN_JUMP: i32 = 8;
/// A physical limbus cannot change scale discontinuously in one asynchronous
/// SAM answer. Require several mutually consistent out-of-family fits before
/// replacing the established scale/sign lineage; isolated whole-eye or lid
/// masks are withheld without resetting the physical normal.
pub(crate) const FRONTAL_DISK_SCALE_RELOCK_UPDATES: u8 = 3;
pub(crate) const FRONTAL_DISK_AREA_RELOCK_BIN_TOLERANCE: i32 = 2;
pub(crate) const CONTACT_SIGN_CONFIRMATION_UPDATES: u8 = 4;
// A pupil/limbus displacement which is effectively camera-normal, or nearly
// perpendicular to the fitted ellipse normal, contains no trustworthy
// information about which antipodal normal is physical.
pub(crate) const GAZE_SURFACE_MIN_SIGN_ANCHOR_MAGNITUDE: f64 = 0.025;
pub(crate) const GAZE_SURFACE_MIN_SIGN_ANCHOR_ALIGNMENT: f64 = 0.20;
pub(crate) const GAZE_SURFACE_MIN_SIGN_PROJECTION: f64 = 0.025;
// The iris/limbus can only be observed on the camera-facing half of the eye.
// Keep this as a hard half-space boundary rather than a score: a zero or
// negative Z surface normal describes an edge-on/away-facing (concave from
// the camera) solution which the head would occlude.
pub(crate) const CAMERA_FACING_NORMAL_MIN_Z: f64 = 1.0e-6;
pub(crate) const RELATIVE_GAZE_UNIT_TOLERANCE: f64 = 1.0e-6;
pub(crate) const GAZE_KINEMATIC_HISTORY_FRAMES: usize = 6;
pub(crate) const ROTATION_CENTER_HISTORY: Duration = Duration::from_secs(3);
pub(crate) const ROTATION_CENTER_MIN_OBSERVATIONS: usize = 12;
pub(crate) const ROTATION_POSE_STILL_MOTION_MAX: f64 = 35.0;
pub(crate) const ROTATION_POSE_FIRM_START_FRAMES: u32 = 4;
pub(crate) const ROTATION_POSE_FIRM_FULL_FRAMES: u32 = 24;
/// Unit gaze direction in the camera coordinate system. `right` and `down`
/// follow sensor coordinates; positive `toward_camera` points out through the
/// visible corneal surface. This is deliberately independent of ROI origin,
/// display scale, and the authority which supplied the eye surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RelativeGazeVector {
    pub(crate) right: f64,
    pub(crate) down: f64,
    pub(crate) toward_camera: f64,
}

impl RelativeGazeVector {
    pub(crate) fn from_projected(right: f64, down: f64) -> Option<Self> {
        if !right.is_finite() || !down.is_finite() {
            return None;
        }
        let projected_squared = right * right + down * down;
        if !projected_squared.is_finite() || projected_squared > 1.0 + 1.0e-9 {
            return None;
        }
        let gaze = Self {
            right,
            down,
            toward_camera: (1.0 - projected_squared.min(1.0)).sqrt(),
        };
        gaze.is_camera_facing().then_some(gaze)
    }

    pub(crate) fn projected(self) -> (f64, f64) {
        (self.right, self.down)
    }

    pub(crate) fn projected_direction(self) -> Option<(f64, f64)> {
        let length = self.right.hypot(self.down);
        (length.is_finite() && length >= 1.0e-9)
            .then_some((self.right / length, self.down / length))
    }

    pub(crate) fn as_array(self) -> [f64; 3] {
        [self.right, self.down, self.toward_camera]
    }

    pub(crate) fn is_camera_facing(self) -> bool {
        let norm_squared =
            self.right * self.right + self.down * self.down + self.toward_camera.powi(2);
        self.right.is_finite()
            && self.down.is_finite()
            && self.toward_camera.is_finite()
            && self.toward_camera > CAMERA_FACING_NORMAL_MIN_Z
            && norm_squared.is_finite()
            && (norm_squared - 1.0).abs() <= RELATIVE_GAZE_UNIT_TOLERANCE
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SurfaceGazeSample {
    /// Source exposure which supplied the fitted surface. SAM results are
    /// asynchronous and may be presented over several newer camera frames;
    /// retaining this key prevents those repeats from masquerading as fresh
    /// temporal or calibration evidence.
    pub(crate) source_timestamp_ns: Option<u64>,
    pub(crate) frontal_equivalent_disk_area_px2: f64,
    pub(crate) area_bucket: i32,
    pub(crate) quantized_frontal_disk_radius_px: f64,
    pub(crate) near_surface_point_sensor_px: (f64, f64),
    pub(crate) relative_gaze: RelativeGazeVector,
    /// False while an anchorless ellipse still has two equally plausible
    /// projected normal branches. Such a sample may be rendered provisionally,
    /// but it must never train an eye-to-screen calibration.
    pub(crate) sign_resolved: bool,
    /// Changes whenever the persistent sign state is reset or changes branch.
    /// Consumers can distinguish training and current sign lineage; completed
    /// cursor mappings deliberately continue across a sign-only epoch change.
    pub(crate) sign_epoch: u64,
    pub(crate) kinematic_sign_correction: [bool; 2],
    /// Immutable source-time diagnostics; presentation duplicates do not refresh
    /// this evidence. None is reserved for old imported/test samples.
    pub(crate) sign_diagnostics: Option<SurfaceSignDiagnostics>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SurfaceSignEvidence {
    #[default]
    Unresolved,
    MotionInterval,
    PupilAnchor,
    MotionWindow,
    KinematicCorrection,
}

impl SurfaceSignEvidence {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Unresolved => "unresolved",
            Self::MotionInterval => "motion-interval",
            Self::PupilAnchor => "pupil-anchor",
            Self::MotionWindow => "motion-window",
            Self::KinematicCorrection => "kinematic-correction",
        }
    }
    pub(crate) fn sustained_acquisition_support(self) -> bool {
        matches!(self, Self::PupilAnchor | Self::MotionWindow)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SurfaceSignDiagnostics {
    pub(crate) evidence: SurfaceSignEvidence,
    pub(crate) selected_branch: usize,
    pub(crate) branch_residual_ema_px: [f64; 2],
    pub(crate) source_motion_residual_px: Option<f64>,
    pub(crate) temporal_margin_px: Option<f64>,
    pub(crate) pending_anchor_votes: u8,
    /// Source-time branch correspondence through the frontal degeneracy;
    /// not an antipodal sign correction or a new training epoch.
    pub(crate) near_frontal_continuation: bool,
}

/// The established rule remains an explicit offline baseline. The live fallback
/// additionally permits the same sustained motion vote to acquire an unknown
/// sign; it does not lower its support or physical-geometry requirements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SignAcquisitionPolicy {
    Established,
    #[default]
    MotionWindowFallback,
    MotionWindowOnly,
    ReliableMotionSeed,
}

#[derive(Debug, Default)]
pub(crate) struct SurfaceGazeTracker {
    pub(crate) acquisition_policy: SignAcquisitionPolicy,
    pub(crate) reliable_motion_observations: u16,
    pub(crate) sign_evidence: SurfaceSignEvidence,
    pub(crate) same_sign_anchor_support: u8,
    pub(crate) floating_center_sensor: Option<(f64, f64)>,
    pub(crate) floating_near_point_sensor: Option<(f64, f64)>,
    pub(crate) area_bucket: Option<i32>,
    pub(crate) pending_area_bucket: Option<i32>,
    pub(crate) pending_area_bucket_frames: u8,
    pub(crate) last_observed: Option<Instant>,
    pub(crate) contact_sign_hypotheses: Option<[ContactSignHypothesis; 2]>,
    pub(crate) selected_sign_hypothesis: usize,
    pub(crate) pending_sign_hypothesis: Option<usize>,
    pub(crate) pending_sign_frames: u8,
    pub(crate) pending_kinematic_sign_hypothesis: Option<usize>,
    pub(crate) pending_kinematic_sign_frames: u8,
    pub(crate) sign_resolved: bool,
    pub(crate) sign_epoch: u64,
    pub(crate) kinematic_history: VecDeque<GazeKinematicFrame>,
    pub(crate) motion_sign_window: sign_motion::MotionSignWindow,
    /// Most recent source which actually produced an admitted surface. The
    /// caller composes whole-ROI motion from this exposure to the next SAM
    /// result, so a rejected intermediate proposal must not advance it.
    pub(crate) last_keyed_source_timestamp_ns: Option<u64>,
    /// Duplicate/out-of-order guard is separate from the successful source:
    /// a rejected proposal must not be evaluated repeatedly on redraw.
    pub(crate) last_keyed_attempted_source_timestamp_ns: Option<u64>,
    pub(crate) last_keyed_sample: Option<SurfaceGazeSample>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ContactSignHypothesis {
    pub(crate) effective_pivot_sensor_px: (f64, f64),
    pub(crate) near_surface_sensor_px: (f64, f64),
    pub(crate) residual_ema: f64,
    pub(crate) observations: u16,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GazeKinematicFrame {
    pub(crate) observed_at: Instant,
    pub(crate) source_timestamp_ns: Option<u64>,
    pub(crate) ellipse_center_sensor: (f64, f64),
    pub(crate) projected_gaze: (f64, f64),
    pub(crate) implied_pivot_sensor_px: (f64, f64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GazeSignCorrection {
    pub(crate) projected_gaze: (f64, f64),
    pub(crate) flipped_x: bool,
    pub(crate) flipped_y: bool,
    pub(crate) resolved: bool,
}

pub(crate) fn projected_gaze_angle_between(first: (f64, f64), second: (f64, f64)) -> f64 {
    let first_z = (1.0 - first.0 * first.0 - first.1 * first.1)
        .max(0.0)
        .sqrt();
    let second_z = (1.0 - second.0 * second.0 - second.1 * second.1)
        .max(0.0)
        .sqrt();
    (first.0 * second.0 + first.1 * second.1 + first_z * second_z)
        .clamp(-1.0, 1.0)
        .acos()
}

/// Resolve the antipodal normal ambiguity of a projected circular surface
/// from a short biomechanical motion history. A measured ellipse has exactly
/// two possible camera-facing projected normals: `(x, y)` and `(-x, -y)`.
/// Independent one-axis reflections describe a different ellipse and must
/// not be manufactured here. The score compares constant-angular-velocity
/// gaze, angular acceleration, and the translated globe center implied by the
/// moving limbus ellipse.
pub(crate) fn kinematic_gaze_sign_correction(
    history: &VecDeque<GazeKinematicFrame>,
    observed_at: Instant,
    source_timestamp_ns: Option<u64>,
    ellipse_center_sensor: (f64, f64),
    limbus_plane_offset_px: f64,
    observed_gaze: (f64, f64),
    global_similarity: Option<NativeGlobalSimilarityEvidence>,
) -> GazeSignCorrection {
    let unchanged = GazeSignCorrection {
        projected_gaze: observed_gaze,
        flipped_x: false,
        flipped_y: false,
        resolved: false,
    };
    if history.is_empty()
        || !limbus_plane_offset_px.is_finite()
        || limbus_plane_offset_px <= 1.0e-6
        || !observed_gaze.0.is_finite()
        || !observed_gaze.1.is_finite()
    {
        return unchanged;
    }
    let last = history[history.len() - 1];
    let elapsed_seconds =
        |earlier: GazeKinematicFrame, later_at: Instant, later_source_timestamp_ns: Option<u64>| {
            earlier
                .source_timestamp_ns
                .zip(later_source_timestamp_ns)
                .and_then(|(earlier, later)| {
                    later
                        .checked_sub(earlier)
                        .map(|nanoseconds| nanoseconds as f64 / 1_000_000_000.0)
                })
                .unwrap_or_else(|| {
                    later_at
                        .saturating_duration_since(earlier.observed_at)
                        .as_secs_f64()
                })
        };
    let current_interval_s = elapsed_seconds(last, observed_at, source_timestamp_ns);
    if current_interval_s < 1.0e-4 || current_interval_s > 0.75 {
        return unchanged;
    }
    let (predicted_gaze, extrapolated_center_sensor_px, extrapolated_pivot_sensor_px, previous_angular_speed_rad_s) =
        if history.len() >= 2 {
            let previous = history[history.len() - 2];
            let history_interval_s = elapsed_seconds(previous, last.observed_at, last.source_timestamp_ns);
            if history_interval_s < 1.0e-4 {
                return unchanged;
            }
            let extrapolation = current_interval_s / history_interval_s;
            (
                (
                    last.projected_gaze.0
                        + (last.projected_gaze.0 - previous.projected_gaze.0) * extrapolation,
                    last.projected_gaze.1
                        + (last.projected_gaze.1 - previous.projected_gaze.1) * extrapolation,
                ),
                (
                    last.ellipse_center_sensor.0
                        + (last.ellipse_center_sensor.0 - previous.ellipse_center_sensor.0)
                            * extrapolation,
                    last.ellipse_center_sensor.1
                        + (last.ellipse_center_sensor.1 - previous.ellipse_center_sensor.1)
                            * extrapolation,
                ),
                (
                    last.implied_pivot_sensor_px.0
                        + (last.implied_pivot_sensor_px.0
                            - previous.implied_pivot_sensor_px.0)
                            * extrapolation,
                    last.implied_pivot_sensor_px.1
                        + (last.implied_pivot_sensor_px.1
                            - previous.implied_pivot_sensor_px.1)
                            * extrapolation,
                ),
                projected_gaze_angle_between(previous.projected_gaze, last.projected_gaze)
                    / history_interval_s,
            )
        } else {
            // With one prior frame there is no velocity estimate yet, but the
            // implied fixed globe center already makes an ellipse translation
            // an immediate component-sign cue away from a camera-normal pose.
            (
                last.projected_gaze,
                last.ellipse_center_sensor,
                last.implied_pivot_sensor_px,
                0.0,
            )
        };
    let predict_global = |point: (f64, f64)| {
        let Some(global) = global_similarity.filter(|evidence| evidence.reliable) else {
            return point;
        };
        let x = point.0 - f64::from(global.motion_center_sensor[0]);
        let y = point.1 - f64::from(global.motion_center_sensor[1]);
        (
            point.0
                + f64::from(global.motion.translation[0])
                + f64::from(global.motion.diagonal_coefficient_delta) * x
                - f64::from(global.motion.rotation_coefficient) * y,
            point.1
                + f64::from(global.motion.translation[1])
                + f64::from(global.motion.rotation_coefficient) * x
                + f64::from(global.motion.diagonal_coefficient_delta) * y,
        )
    };
    // A reliable whole-ROI transform removes head/camera translation from
    // the biomechanical cue. Without it, retain the short velocity predictor
    // as a presentation fallback, but do not let that fallback establish an
    // otherwise unresolved physical sign in SurfaceGazeTracker.
    let predicted_center = global_similarity
        .filter(|evidence| evidence.reliable)
        .map_or(extrapolated_center_sensor_px, |_| {
            predict_global(last.ellipse_center_sensor)
        });
    let predicted_pivot_sensor_px = global_similarity
        .filter(|evidence| evidence.reliable)
        .map_or(extrapolated_pivot_sensor_px, |_| {
            predict_global(last.implied_pivot_sensor_px)
        });
    let normalized_center_position_residual = (ellipse_center_sensor.0 - predicted_center.0)
        .hypot(ellipse_center_sensor.1 - predicted_center.1)
        / limbus_plane_offset_px;

    let candidates = [
        (observed_gaze, false),
        ((-observed_gaze.0, -observed_gaze.1), true),
    ];
    let score = |candidate: (f64, f64), flipped: bool| {
        let transverse_direction_residual =
            (candidate.0 - predicted_gaze.0).hypot(candidate.1 - predicted_gaze.1);
        let angular_speed_rad_s =
            projected_gaze_angle_between(last.projected_gaze, candidate) / current_interval_s;
        // This legacy regularizer has units of angle: |delta angular speed|
        // times this interval. It is NOT angular acceleration (rad/s^2), nor
        // angular jerk. Renaming documents the existing math without changing it.
        let angular_speed_change_step_rad = (angular_speed_rad_s - previous_angular_speed_rad_s).abs() * current_interval_s;
        let implied_pivot_sensor_px = (
            ellipse_center_sensor.0 - limbus_plane_offset_px * candidate.0,
            ellipse_center_sensor.1 - limbus_plane_offset_px * candidate.1,
        );
        let normalized_pivot_position_residual = (implied_pivot_sensor_px.0 - predicted_pivot_sensor_px.0)
            .hypot(implied_pivot_sensor_px.1 - predicted_pivot_sensor_px.1)
            / limbus_plane_offset_px;
        3.0 * transverse_direction_residual
            + 1.25 * angular_speed_change_step_rad
            + normalized_pivot_position_residual / (1.0 + normalized_center_position_residual)
            + 0.05 * f64::from(flipped)
    };
    let mut scored =
        candidates.map(|(candidate, flipped)| (score(candidate, flipped), candidate, flipped));
    scored.sort_by(|left, right| left.0.total_cmp(&right.0));
    let required_improvement = 0.10 + 0.40 * normalized_center_position_residual.min(1.0);
    if scored[1].0 - scored[0].0 < required_improvement {
        unchanged
    } else {
        GazeSignCorrection {
            projected_gaze: scored[0].1,
            flipped_x: scored[0].2,
            flipped_y: scored[0].2,
            resolved: true,
        }
    }
}

pub(crate) fn frontal_equivalent_iris_disk_area_px2(major_radius: f64, minor_radius: f64) -> Option<f64> {
    let (major_radius, minor_radius) = if major_radius >= minor_radius {
        (major_radius, minor_radius)
    } else {
        (minor_radius, major_radius)
    };
    if !major_radius.is_finite()
        || !minor_radius.is_finite()
        || major_radius < 4.0
        || minor_radius < 1.0
        || !raw_iris_focus::projected_circular_limbus_axes_plausible(major_radius, minor_radius)
    {
        return None;
    }
    // Orthographic projection turns a circular disk into an ellipse whose
    // minor/major ratio is the camera-normal cosine. Undo that foreshortening
    // so tilted and front-facing observations of the same iris have the same
    // area estimate: (pi*a*b)/(b/a) = pi*a^2.
    let projected_area = std::f64::consts::PI * major_radius * minor_radius;
    let camera_normal_cosine = minor_radius / major_radius;
    let frontal_equivalent_disk_area = projected_area / camera_normal_cosine;
    frontal_equivalent_disk_area.is_finite().then_some(frontal_equivalent_disk_area)
}

pub(crate) fn quantize_frontal_disk_area(frontal_equivalent_disk_area_px2: f64) -> Option<(i32, f64)> {
    if !frontal_equivalent_disk_area_px2.is_finite() || frontal_equivalent_disk_area_px2 <= 0.0 {
        return None;
    }
    let logarithmic_bucket =
        (frontal_equivalent_disk_area_px2.ln() / FRONTAL_DISK_AREA_BIN_RATIO.ln()).round();
    if logarithmic_bucket < i32::MIN as f64 || logarithmic_bucket > i32::MAX as f64 {
        return None;
    }
    let bucket = logarithmic_bucket as i32;
    let representative_area = FRONTAL_DISK_AREA_BIN_RATIO.powi(bucket);
    representative_area
        .is_finite()
        .then_some((bucket, representative_area))
}

impl SurfaceGazeTracker {
    pub(crate) fn clear_floating_point(&mut self) {
        self.motion_sign_window.clear();
        self.sign_evidence = SurfaceSignEvidence::Unresolved;
        self.same_sign_anchor_support = 0;
        self.reliable_motion_observations = 0;
        self.floating_center_sensor = None;
        self.floating_near_point_sensor = None;
        self.area_bucket = None;
        self.contact_sign_hypotheses = None;
        self.pending_area_bucket = None;
        self.pending_area_bucket_frames = 0;
        self.selected_sign_hypothesis = 0;
        self.pending_sign_hypothesis = None;
        self.pending_sign_frames = 0;
        self.pending_kinematic_sign_hypothesis = None;
        self.pending_kinematic_sign_frames = 0;
        self.sign_resolved = false;
        self.sign_epoch = self.sign_epoch.wrapping_add(1);
    }

    /// A missing asynchronous result is not evidence that the physical eye
    /// changed antipodes. Drop only velocity and presentation interpolation;
    /// retain the two transported contact identities, selected sign, and
    /// accepted scale lineage. Explicit ROI/provider resets still replace the
    /// whole tracker, while a sustained scale-family change still calls the
    /// full reset above and advances the epoch.
    pub(crate) fn clear_stale_motion_preserving_sign(&mut self) {
        self.motion_sign_window.clear();
        self.same_sign_anchor_support = 0;
        self.reliable_motion_observations = 0;
        self.floating_center_sensor = None;
        self.floating_near_point_sensor = None;
        self.pending_area_bucket = None;
        self.pending_area_bucket_frames = 0;
        self.pending_sign_hypothesis = None;
        self.pending_sign_frames = 0;
        self.pending_kinematic_sign_hypothesis = None;
        self.pending_kinematic_sign_frames = 0;
        self.kinematic_history.clear();
        if let Some(hypotheses) = self.contact_sign_hypotheses.as_mut() {
            for hypothesis in hypotheses {
                // Preserve spatial identity but do not compare a new motion
                // interval with an EMA accumulated before the missing span.
                hypothesis.residual_ema = 0.0;
                hypothesis.observations = 1;
            }
        }
    }

    /// Return true only when a new, discontinuous scale family has persisted
    /// long enough to replace the established physical surface. SAM can
    /// occasionally fit the eye opening or an eyelid chord for one answer;
    /// such a proposal remains visible in its diagnostic view but must not
    /// reset gaze sign or enter calibration.
    pub(crate) fn consider_area_bucket_jump(&mut self, candidate: i32) -> bool {
        let continues_pending = self.pending_area_bucket.is_some_and(|pending| {
            (pending - candidate).abs() <= FRONTAL_DISK_AREA_RELOCK_BIN_TOLERANCE
        });
        if continues_pending {
            self.pending_area_bucket_frames = self.pending_area_bucket_frames.saturating_add(1);
            // Follow gradual noise within the candidate family without
            // allowing that family to drift arbitrarily far before commit.
            self.pending_area_bucket = Some(candidate);
        } else {
            self.pending_area_bucket = Some(candidate);
            self.pending_area_bucket_frames = 1;
        }
        self.pending_area_bucket_frames >= FRONTAL_DISK_SCALE_RELOCK_UPDATES
    }

    pub(crate) fn consider_sign_hypothesis(&mut self, candidate: usize) {
        if self.sign_resolved && candidate == self.selected_sign_hypothesis {
            // A later pupil cue can validate a weak motion-only seed without
            // changing the chosen branch. Four fresh matching anchors are
            // still required; redraws are rejected by the source-key guard.
            self.same_sign_anchor_support = self.same_sign_anchor_support.saturating_add(1);
            if self.same_sign_anchor_support >= CONTACT_SIGN_CONFIRMATION_UPDATES {
                self.sign_evidence = SurfaceSignEvidence::PupilAnchor;
            }
            self.pending_sign_hypothesis = None;
            self.pending_sign_frames = 0;
            return;
        }
        self.same_sign_anchor_support = 0;
        if self.pending_sign_hypothesis == Some(candidate) {
            self.pending_sign_frames = self.pending_sign_frames.saturating_add(1);
        } else {
            self.pending_sign_hypothesis = Some(candidate);
            self.pending_sign_frames = 1;
        }
        if self.pending_sign_frames >= CONTACT_SIGN_CONFIRMATION_UPDATES {
            if candidate != self.selected_sign_hypothesis {
                self.selected_sign_hypothesis = candidate;
                self.sign_epoch = self.sign_epoch.wrapping_add(1);
            }
            self.pending_sign_hypothesis = None;
            self.pending_sign_frames = 0;
            self.sign_resolved = true;
            self.sign_evidence = SurfaceSignEvidence::PupilAnchor;
        }
    }

    /// Kinematics may challenge an already resolved branch, but a single
    /// discontinuity must never invert the physical surface. Keep this vote
    /// separate from the pupil/contact vote so an anchorless frame cannot
    /// erase a motion challenge before it has accumulated evidence.
    pub(crate) fn consider_kinematic_sign_hypothesis(&mut self, candidate: usize) -> bool {
        if candidate == self.selected_sign_hypothesis {
            self.pending_kinematic_sign_hypothesis = None;
            self.pending_kinematic_sign_frames = 0;
            return false;
        }
        if self.pending_kinematic_sign_hypothesis == Some(candidate) {
            self.pending_kinematic_sign_frames =
                self.pending_kinematic_sign_frames.saturating_add(1);
        } else {
            self.pending_kinematic_sign_hypothesis = Some(candidate);
            self.pending_kinematic_sign_frames = 1;
        }
        if self.pending_kinematic_sign_frames < CONTACT_SIGN_CONFIRMATION_UPDATES {
            return false;
        }
        self.selected_sign_hypothesis = candidate;
        self.pending_kinematic_sign_hypothesis = None;
        self.pending_kinematic_sign_frames = 0;
        self.pending_sign_hypothesis = None;
        self.pending_sign_frames = 0;
        self.sign_resolved = true;
        self.sign_evidence = SurfaceSignEvidence::KinematicCorrection;
        self.sign_epoch = self.sign_epoch.wrapping_add(1);
        self.kinematic_history.clear();
        self.floating_center_sensor = None;
        self.floating_near_point_sensor = None;
        true
    }

    pub(crate) fn observe(
        &mut self,
        now: Instant,
        sensor_origin: (u32, u32),
        pupil_limbus_gaze_anchor: Option<(f64, f64)>,
        outer: &raw_iris_focus::OuterIrisBoundary,
    ) -> Option<SurfaceGazeSample> {
        self.observe_with_global_similarity_at_source(
            now,
            None,
            sensor_origin,
            pupil_limbus_gaze_anchor,
            outer,
            None,
        )
    }

    pub(crate) fn observe_with_global_similarity(
        &mut self,
        now: Instant,
        sensor_origin: (u32, u32),
        pupil_limbus_gaze_anchor: Option<(f64, f64)>,
        outer: &raw_iris_focus::OuterIrisBoundary,
        global_similarity: Option<NativeGlobalSimilarityEvidence>,
    ) -> Option<SurfaceGazeSample> {
        self.observe_with_global_similarity_at_source(
            now,
            None,
            sensor_origin,
            pupil_limbus_gaze_anchor,
            outer,
            global_similarity,
        )
    }

    pub(crate) fn observe_with_global_similarity_at_source(
        &mut self,
        now: Instant,
        source_timestamp_ns: Option<u64>,
        sensor_origin: (u32, u32),
        pupil_limbus_gaze_anchor: Option<(f64, f64)>,
        outer: &raw_iris_focus::OuterIrisBoundary,
        global_similarity: Option<NativeGlobalSimilarityEvidence>,
    ) -> Option<SurfaceGazeSample> {
        if outer.points.len() < 8 || !outer.angle.is_finite() {
            return None;
        }
        let (major_radius, minor_radius, major_angle) = if outer.major_radius >= outer.minor_radius
        {
            (outer.major_radius, outer.minor_radius, outer.angle)
        } else {
            (
                outer.minor_radius,
                outer.major_radius,
                outer.angle + std::f64::consts::FRAC_PI_2,
            )
        };
        let frontal_equivalent_disk_area_px2 = frontal_equivalent_iris_disk_area_px2(major_radius, minor_radius)?;
        let (area_bucket, bucketed_area_px2) = quantize_frontal_disk_area(frontal_equivalent_disk_area_px2)?;
        let quantized_frontal_disk_radius_px = (bucketed_area_px2 / std::f64::consts::PI).sqrt();
        let axis_ratio = (minor_radius / major_radius).clamp(0.0, 1.0);
        let projected_normal_length = (1.0 - axis_ratio * axis_ratio).sqrt();
        if !quantized_frontal_disk_radius_px.is_finite() || !projected_normal_length.is_finite() {
            return None;
        }

        let stale = self
            .last_observed
            .is_some_and(|last| now.saturating_duration_since(last) > GAZE_SURFACE_RESET_AFTER);
        let bucket_jump = self
            .area_bucket
            .is_some_and(|previous| (previous - area_bucket).abs() > FRONTAL_DISK_AREA_MAX_BIN_JUMP);
        if stale {
            self.clear_stale_motion_preserving_sign();
        } else if bucket_jump {
            if !self.consider_area_bucket_jump(area_bucket) {
                // This source frame was processed and therefore keeps the
                // established lineage alive, but its incompatible geometry
                // is neither published nor allowed into motion history.
                self.last_observed = Some(now);
                self.kinematic_history.clear();
                self.motion_sign_window.clear();
                return None;
            }
            // A sustained new scale estimate invalidates both the floating
            // contact and its direction/velocity arc. Commit one explicit
            // epoch transition rather than resetting on every outlier.
            self.clear_floating_point();
            self.kinematic_history.clear();
        } else {
            self.pending_area_bucket = None;
            self.pending_area_bucket_frames = 0;
        }

        let center_sensor = (
            sensor_origin.0 as f64 + outer.center.0,
            sensor_origin.1 as f64 + outer.center.1,
        );
        if !center_sensor.0.is_finite() || !center_sensor.1.is_finite() {
            return None;
        }
        // The ellipse minor axis is the projected disk-normal axis. Its sign
        // is ambiguous in one frame, yielding two possible camera-near
        // surface points. Select the candidate conspicuously closer to the
        // floating point; the independent pupil/limbus displacement seeds the
        // sign on the first valid frame.
        let projected_axis = (-major_angle.sin(), major_angle.cos());
        let positive_offset = (
            projected_axis.0 * quantized_frontal_disk_radius_px * projected_normal_length,
            projected_axis.1 * quantized_frontal_disk_radius_px * projected_normal_length,
        );
        let negative_offset = (-positive_offset.0, -positive_offset.1);
        let sign_anchor = pupil_limbus_gaze_anchor.filter(|anchor| {
            let magnitude = anchor.0.hypot(anchor.1);
            magnitude.is_finite()
                && magnitude >= GAZE_SURFACE_MIN_SIGN_ANCHOR_MAGNITUDE
                && projected_normal_length >= GAZE_SURFACE_MIN_SIGN_PROJECTION
                && ((projected_axis.0 * anchor.0 + projected_axis.1 * anchor.1) / magnitude).abs()
                    >= GAZE_SURFACE_MIN_SIGN_ANCHOR_ALIGNMENT
        });
        let previous_offset = self
            .floating_center_sensor
            .zip(self.floating_near_point_sensor)
            .map(|(center, near)| (near.0 - center.0, near.1 - center.1));
        let continuity_offset = if let Some(previous) = previous_offset {
            let positive_distance =
                (positive_offset.0 - previous.0).hypot(positive_offset.1 - previous.1);
            let negative_distance =
                (negative_offset.0 - previous.0).hypot(negative_offset.1 - previous.1);
            if positive_distance <= negative_distance {
                positive_offset
            } else {
                negative_offset
            }
        } else if let Some(anchor) = sign_anchor {
            if positive_offset.0 * anchor.0 + positive_offset.1 * anchor.1 >= 0.0 {
                positive_offset
            } else {
                negative_offset
            }
        } else if positive_offset.0 > 0.0
            || (positive_offset.0.abs() <= 1.0e-12 && positive_offset.1 >= 0.0)
        {
            positive_offset
        } else {
            negative_offset
        };
        let sphere_radius = quantized_frontal_disk_radius_px * 1.83;
        let limbus_plane_offset_px = (sphere_radius * sphere_radius
            - quantized_frontal_disk_radius_px * quantized_frontal_disk_radius_px)
            .max(0.0)
            .sqrt();
        let offsets = [positive_offset, negative_offset];
        let candidates = offsets.map(|offset| {
            let near = (center_sensor.0 + offset.0, center_sensor.1 + offset.1);
            let normal = (
                offset.0 / quantized_frontal_disk_radius_px,
                offset.1 / quantized_frontal_disk_radius_px,
            );
            ContactSignHypothesis {
                effective_pivot_sensor_px: (
                    center_sensor.0 - limbus_plane_offset_px * normal.0,
                    center_sensor.1 - limbus_plane_offset_px * normal.1,
                ),
                near_surface_sensor_px: near,
                residual_ema: 0.0,
                observations: 1,
            }
        });
        let sign_state_before_contact = (
            self.selected_sign_hypothesis,
            self.sign_resolved,
            self.sign_epoch,
        );
        let mut motion_window_switch = false;
        let mut motion_window_supported = false;
        let mut near_frontal_continuation = false;
        if let Some(previous) = self.contact_sign_hypotheses {
            let temporal_motion_reliable = global_similarity.is_some_and(|global| global.reliable);
            let predict = |point: (f64, f64)| {
                let Some(global) = global_similarity.filter(|global| global.reliable) else {
                    return point;
                };
                let x = point.0 - f64::from(global.motion_center_sensor[0]);
                let y = point.1 - f64::from(global.motion_center_sensor[1]);
                (
                    point.0
                        + f64::from(global.motion.translation[0])
                        + f64::from(global.motion.diagonal_coefficient_delta) * x
                        - f64::from(global.motion.rotation_coefficient) * y,
                    point.1
                        + f64::from(global.motion.translation[1])
                        + f64::from(global.motion.rotation_coefficient) * x
                        + f64::from(global.motion.diagonal_coefficient_delta) * y,
                )
            };
            let predicted = previous.map(|hypothesis| predict(hypothesis.effective_pivot_sensor_px));
            let error = |prediction: (f64, f64), candidate: ContactSignHypothesis| {
                (prediction.0 - candidate.effective_pivot_sensor_px.0)
                    .hypot(prediction.1 - candidate.effective_pivot_sensor_px.1)
            };
            let direction = |hypothesis: ContactSignHypothesis| {
                let vector = (
                    hypothesis.near_surface_sensor_px.0 - hypothesis.effective_pivot_sensor_px.0,
                    hypothesis.near_surface_sensor_px.1 - hypothesis.effective_pivot_sensor_px.1,
                );
                let length = vector.0.hypot(vector.1);
                (length.is_finite() && length > 1.0e-9)
                    .then_some((vector.0 / length, vector.1 / length))
            };
            let direction_error = |left: ContactSignHypothesis, right: ContactSignHypothesis| {
                direction(left)
                    .zip(direction(right))
                    .map(|(left, right)| {
                        let transported = global_similarity
                            .filter(|global| global.reliable)
                            .map_or(left, |global| {
                                let a = 1.0 + f64::from(global.motion.diagonal_coefficient_delta);
                                let b = f64::from(global.motion.rotation_coefficient);
                                let length = a.hypot(b).max(1.0e-12);
                                (
                                    (a * left.0 - b * left.1) / length,
                                    (b * left.0 + a * left.1) / length,
                                )
                            });
                        (transported.0 - right.0).hypot(transported.1 - right.1)
                    })
                    .unwrap_or(f64::INFINITY)
            };
            // Match physical identities by transported direction, including
            // equivalent ellipse angles separated by PI. Absolute pivot
            // proximity can exchange the two identities after a translation
            // or fit jump, silently reversing gaze without a sign vote/epoch.
            // Pivot residuals remain evidence for explicit sign decisions.
            let (direct, crossed) = (
                direction_error(previous[0], candidates[0])
                    + direction_error(previous[1], candidates[1]),
                direction_error(previous[0], candidates[1])
                    + direction_error(previous[1], candidates[0]),
            );
            let mut assignment = if direct <= crossed { [0, 1] } else { [1, 0] };
            if self.sign_resolved {
                if let Some(global) = global_similarity.filter(|g| g.reliable) {
                    if let Some(continuing) = sign_continuity::near_frontal_assignment(
                        &self.kinematic_history, source_timestamp_ns, center_sensor,
                        quantized_frontal_disk_radius_px, candidates,
                        predicted,
                        f64::from(global.motion.residual),
                        self.selected_sign_hypothesis, assignment,
                    ) {
                        assignment = continuing;
                        near_frontal_continuation = true;
                    }
                }
            }
            let updated = std::array::from_fn(|index| {
                let mut candidate = candidates[assignment[index]];
                let residual = error(predicted[index], candidate);
                candidate.residual_ema = if !temporal_motion_reliable {
                    // Directional assignment is dimensionless. Do not mix it
                    // into the pixel residual used to resolve a later frame
                    // with independently measured whole-ROI motion.
                    previous[index].residual_ema
                } else if previous[index].observations <= 1
                    || (self.acquisition_policy == SignAcquisitionPolicy::ReliableMotionSeed
                        && self.reliable_motion_observations == 0)
                {
                    residual
                } else {
                    0.75 * previous[index].residual_ema + 0.25 * residual
                };
                candidate.observations = previous[index].observations.saturating_add(1);
                candidate
            });
            self.contact_sign_hypotheses = Some(updated);
            if temporal_motion_reliable {
                self.reliable_motion_observations = self.reliable_motion_observations.saturating_add(1);
            }
            // Both histories use their OWN transported pivots. Sustained
            // current-source evidence may challenge a resolved sign without
            // penalizing that retrospective correction as a new saccade.
            let residuals = (temporal_motion_reliable && projected_normal_length >= 0.08)
                .then(|| [0, 1].map(|i| error(predicted[i], updated[i])));
            let motion_decision = self.motion_sign_window.observe(
                source_timestamp_ns, now, residuals, quantized_frontal_disk_radius_px,
                global_similarity.map_or(0.0, |g| f64::from(g.motion.residual)),
            );
            if sign_anchor.is_none() { self.same_sign_anchor_support = 0; }
            let anchor_candidate = sign_anchor.map(|anchor| {
                let score = |hypothesis: ContactSignHypothesis| {
                    let offset = (
                        hypothesis.near_surface_sensor_px.0 - center_sensor.0,
                        hypothesis.near_surface_sensor_px.1 - center_sensor.1,
                    );
                    offset.0 * anchor.0 + offset.1 * anchor.1
                };
                usize::from(score(updated[1]) > score(updated[0]))
            });
            if anchor_candidate != Some(self.selected_sign_hypothesis) {
                self.same_sign_anchor_support = 0;
            }
            let temporal_residual_gap = (updated[0].residual_ema - updated[1].residual_ema).abs();
            let temporal_resolution_threshold = global_similarity
                .filter(|global| global.reliable)
                .map_or(0.35, |global| {
                    (4.0 * f64::from(global.motion.residual)).max(0.35)
                });
            let temporal_candidate = (sign_anchor.is_none()
                && temporal_motion_reliable
                && updated[0].observations >= 2
                && temporal_residual_gap >= temporal_resolution_threshold)
                .then(|| usize::from(updated[1].residual_ema < updated[0].residual_ema));
            let window_can_acquire = matches!(self.acquisition_policy,
                SignAcquisitionPolicy::MotionWindowFallback | SignAcquisitionPolicy::MotionWindowOnly);
            if let Some(decision) = motion_decision.filter(|_| self.sign_resolved || window_can_acquire) {
                self.sign_evidence = SurfaceSignEvidence::MotionWindow;
                motion_window_supported = true;
                self.pending_sign_hypothesis = None;
                self.pending_sign_frames = 0;
                self.pending_kinematic_sign_hypothesis = None;
                self.pending_kinematic_sign_frames = 0;
                if decision.candidate != self.selected_sign_hypothesis {
                    self.selected_sign_hypothesis = decision.candidate;
                    self.sign_epoch = self.sign_epoch.wrapping_add(1);
                    motion_window_switch = true;
                    eprintln!("SURFACE_SIGN_MOTION source_ns={source_timestamp_ns:?} branch={} support={} normalized_pivot_costs={:?} epoch={}",
                        decision.candidate, decision.support, decision.mean_costs, self.sign_epoch);
                    self.motion_sign_window.clear();
                }
                self.sign_resolved = true;
            } else if let Some(candidate) = anchor_candidate {
                // A RAW dark component may be an eyelid shadow. It can seed
                // an unresolved branch, but overturning an established one
                // also requires independent current-interval motion evidence.
                let selected = self.selected_sign_hypothesis;
                let corroborated = temporal_motion_reliable
                    && error(predicted[selected], updated[selected])
                        - error(predicted[candidate], updated[candidate])
                        >= temporal_resolution_threshold;
                if !self.sign_resolved || candidate == selected || corroborated {
                    self.consider_sign_hypothesis(candidate);
                } else {
                    self.pending_sign_hypothesis = None;
                    self.pending_sign_frames = 0;
                }
            } else if let Some(candidate) = temporal_candidate.filter(|_| !self.sign_resolved
                && self.acquisition_policy != SignAcquisitionPolicy::MotionWindowOnly) {
                self.sign_evidence = SurfaceSignEvidence::MotionInterval;
                // Whole-ROI motion can break the initial two-way tie, but the
                // hypotheses above are persistent physical identities. Once a
                // sign is resolved, re-ranking their historical residual EMAs
                // must not hop from one identity to the other. Only a sustained
                // pupil/contact anchor or the separate biomechanical validator
                // below may overturn an established branch.
                if candidate != self.selected_sign_hypothesis {
                    self.selected_sign_hypothesis = candidate;
                    self.sign_epoch = self.sign_epoch.wrapping_add(1);
                }
                self.pending_sign_hypothesis = None;
                self.pending_sign_frames = 0;
                self.sign_resolved = true;
            } else {
                self.pending_sign_hypothesis = None;
                self.pending_sign_frames = 0;
            }
        } else {
            self.motion_sign_window.observe(source_timestamp_ns, now, None, quantized_frontal_disk_radius_px, 0.0);
            self.contact_sign_hypotheses = Some(candidates);
            self.selected_sign_hypothesis = usize::from(
                (continuity_offset.0 - negative_offset.0)
                    .hypot(continuity_offset.1 - negative_offset.1)
                    < (continuity_offset.0 - positive_offset.0)
                        .hypot(continuity_offset.1 - positive_offset.1),
            );
            self.pending_sign_hypothesis = None;
            self.pending_sign_frames = 0;
            self.sign_resolved = false;
            self.sign_evidence = SurfaceSignEvidence::Unresolved;
            if sign_anchor.is_some() {
                // A single dark component can be a reflection or eyelid
                // shadow.  Seed, but do not resolve, the selected antipode;
                // only independent source frames can complete the vote.
                self.consider_sign_hypothesis(self.selected_sign_hypothesis);
            }
        }
        let mut reset_projection_smoothing = false;
        if sign_state_before_contact
            != (
                self.selected_sign_hypothesis,
                self.sign_resolved,
                self.sign_epoch,
            )
        {
            // The velocity history is expressed in the previously selected
            // branch. Once independent pivot/anchor evidence establishes or
            // changes the physical branch, that old history is not admissible
            // evidence against the newly resolved sign.
            self.kinematic_history.clear();
            if sign_state_before_contact.0 != self.selected_sign_hypothesis {
                self.floating_center_sensor = None;
                self.floating_near_point_sensor = None;
                reset_projection_smoothing = true;
            }
        }
        // Always publish the persistent temporal identity. The pupil/limbus
        // anchor seeds it and may challenge it over several frames, but must
        // never bypass it for a one-frame sign decision.
        let selected = self.contact_sign_hypotheses?[self.selected_sign_hypothesis];
        let selected_offset = (
            selected.near_surface_sensor_px.0 - center_sensor.0,
            selected.near_surface_sensor_px.1 - center_sensor.1,
        );
        let raw_projected_gaze = (
            selected_offset.0 / quantized_frontal_disk_radius_px,
            selected_offset.1 / quantized_frontal_disk_radius_px,
        );
        let sign_correction = kinematic_gaze_sign_correction(
            &self.kinematic_history,
            now,
            source_timestamp_ns,
            center_sensor,
            limbus_plane_offset_px,
            raw_projected_gaze,
            global_similarity,
        );
        let kinematic_candidate = if sign_correction.flipped_x {
            1usize.saturating_sub(self.selected_sign_hypothesis)
        } else {
            self.selected_sign_hypothesis
        };
        let kinematic_switch_committed = sign_correction.resolved
            && !motion_window_supported
            && self.sign_resolved
            && global_similarity.is_some_and(|global| global.reliable)
            && self.consider_kinematic_sign_hypothesis(kinematic_candidate);
        if kinematic_switch_committed {
            // Never average opposite normals together. The first sample in a
            // new sign epoch seeds both presentation smoothing and motion.
            reset_projection_smoothing = true;
        } else if motion_window_supported || !sign_correction.resolved
            || !self.sign_resolved
            || !global_similarity.is_some_and(|global| global.reliable)
        {
            self.pending_kinematic_sign_hypothesis = None;
            self.pending_kinematic_sign_frames = 0;
        }
        // Publish only the persistent branch. A provisional kinematic vote is
        // diagnostics, not an output sign; on the commit frame this lookup
        // immediately moves to the newly selected antipode.
        let selected = self.contact_sign_hypotheses?[self.selected_sign_hypothesis];
        let selected_projected_gaze = (
            (selected.near_surface_sensor_px.0 - center_sensor.0) / quantized_frontal_disk_radius_px,
            (selected.near_surface_sensor_px.1 - center_sensor.1) / quantized_frontal_disk_radius_px,
        );
        let selected_offset = (
            selected_projected_gaze.0 * quantized_frontal_disk_radius_px,
            selected_projected_gaze.1 * quantized_frontal_disk_radius_px,
        );
        let implied_pivot_sensor_px = (
            center_sensor.0 - limbus_plane_offset_px * selected_projected_gaze.0,
            center_sensor.1 - limbus_plane_offset_px * selected_projected_gaze.1,
        );
        self.kinematic_history.push_back(GazeKinematicFrame {
            observed_at: now,
            source_timestamp_ns,
            ellipse_center_sensor: center_sensor,
            projected_gaze: selected_projected_gaze,
            implied_pivot_sensor_px,
        });
        while self.kinematic_history.len() > GAZE_KINEMATIC_HISTORY_FRAMES {
            self.kinematic_history.pop_front();
        }
        let near_surface_point_sensor_px = (
            center_sensor.0 + selected_offset.0,
            center_sensor.1 + selected_offset.1,
        );
        let alpha = if previous_offset.is_some() && !reset_projection_smoothing {
            GAZE_SURFACE_AVERAGE_ALPHA
        } else {
            1.0
        };
        let floating_center_sensor = self
            .floating_center_sensor
            .map_or(center_sensor, |previous| {
                blend_point(previous, center_sensor, alpha)
            });
        let floating_near_point_sensor = self
            .floating_near_point_sensor
            .map_or(near_surface_point_sensor_px, |previous| {
                blend_point(previous, near_surface_point_sensor_px, alpha)
            });
        // The temporal state above stabilizes *which sign* is physical. Once
        // selected, publish this exposure's direction immediately. Averaging
        // the output made absolute gaze cursors creep for several SAM periods.
        let relative_gaze = RelativeGazeVector::from_projected(
            selected_projected_gaze.0, selected_projected_gaze.1,
        )?;

        self.floating_center_sensor = Some(floating_center_sensor);
        self.floating_near_point_sensor = Some(floating_near_point_sensor);
        // Follow real perspective scale gradually. Comparing future fits to
        // this slow physical anchor prevents a series of unrelated masks,
        // each only moderately larger than the last, from ratcheting the
        // accepted iris radius across the eye opening. A separately sustained
        // discontinuity clears the anchor above and seeds its new family here.
        self.area_bucket = Some(self.area_bucket.map_or(area_bucket, |previous| {
            previous + (area_bucket - previous).clamp(-1, 1)
        }));
        self.last_observed = Some(now);
        Some(SurfaceGazeSample {
            source_timestamp_ns,
            frontal_equivalent_disk_area_px2,
            area_bucket,
            quantized_frontal_disk_radius_px,
            near_surface_point_sensor_px,
            relative_gaze,
            sign_resolved: self.sign_resolved,
            sign_epoch: self.sign_epoch,
            kinematic_sign_correction: [kinematic_switch_committed || motion_window_switch; 2],
            sign_diagnostics: Some(SurfaceSignDiagnostics {
                evidence: self.sign_evidence,
                selected_branch: self.selected_sign_hypothesis,
                branch_residual_ema_px: self.contact_sign_hypotheses?.map(|h| h.residual_ema),
                source_motion_residual_px: global_similarity.filter(|g| g.reliable).map(|g| f64::from(g.motion.residual)),
                temporal_margin_px: global_similarity.filter(|g| g.reliable).map(|g| (4.0 * f64::from(g.motion.residual)).max(0.35)),
                pending_anchor_votes: self.pending_sign_frames.max(self.same_sign_anchor_support.min(CONTACT_SIGN_CONFIRMATION_UPDATES)),
                near_frontal_continuation,
            }),
        })
    }

    pub(crate) fn observe_keyed_with_global_similarity(
        &mut self,
        source_timestamp_ns: u64,
        now: Instant,
        sensor_origin: (u32, u32),
        pupil_limbus_gaze_anchor: Option<(f64, f64)>,
        outer: &raw_iris_focus::OuterIrisBoundary,
        global_similarity: Option<NativeGlobalSimilarityEvidence>,
    ) -> Option<SurfaceGazeSample> {
        if self
            .last_keyed_attempted_source_timestamp_ns
            .is_some_and(|previous| source_timestamp_ns <= previous)
        {
            return self.last_keyed_sample;
        }
        let sample = self
            .observe_with_global_similarity_at_source(
                now,
                Some(source_timestamp_ns),
                sensor_origin,
                pupil_limbus_gaze_anchor,
                outer,
                global_similarity,
            )
            .map(|mut sample| {
                sample.source_timestamp_ns = Some(source_timestamp_ns);
                sample
            });
        self.last_keyed_attempted_source_timestamp_ns = Some(source_timestamp_ns);
        if sample.is_some() {
            self.last_keyed_source_timestamp_ns = Some(source_timestamp_ns);
        }
        self.last_keyed_sample = sample;
        sample
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RingPoseObservation {
    pub(crate) captured_at: Instant,
    pub(crate) center: (f64, f64),
    pub(crate) major_radius: f64,
    pub(crate) minor_radius: f64,
    pub(crate) angle: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProjectedVisionPose {
    pub(crate) center: (f64, f64),
    // Projected vector from the approximate effective rotation center toward the
    // camera-visible iris pole. Retaining the vector, rather than only its
    // endpoint, lets temporal scoring distinguish translation from rotation.
    pub(crate) pole: (f64, f64),
}

#[derive(Default)]
pub(crate) struct RotationCenterHistory {
    pub(crate) observations: VecDeque<RingPoseObservation>,
    pub(crate) estimate: Option<ProjectedVisionPose>,
    pub(crate) previous_ring: Option<RingPoseObservation>,
    pub(crate) stable_frames: u32,
}

impl RotationCenterHistory {
    pub(crate) fn observe(
        &mut self,
        now: Instant,
        center: (f64, f64),
        major_radius: f64,
        minor_radius: f64,
        angle: f64,
        motion_score: f64,
        comparable_to_previous: bool,
    ) -> Option<ProjectedVisionPose> {
        let mut accepted_observation = None;
        if center.0.is_finite()
            && center.1.is_finite()
            && major_radius.is_finite()
            && minor_radius.is_finite()
            && angle.is_finite()
            && major_radius >= 8.0
            && minor_radius <= major_radius
            && raw_iris_focus::projected_circular_limbus_axes_plausible(major_radius, minor_radius)
        {
            let observation = RingPoseObservation {
                captured_at: now,
                center,
                major_radius,
                minor_radius,
                angle,
            };
            let geometrically_still = self.previous_ring.is_some_and(|previous| {
                let center_limit = (major_radius * 0.10).clamp(1.5, 3.0);
                let center_shift =
                    (previous.center.0 - center.0).hypot(previous.center.1 - center.1);
                let radius_shift = (previous.major_radius - major_radius).abs()
                    / previous.major_radius.max(major_radius).max(1.0);
                let previous_ratio = previous.minor_radius / previous.major_radius.max(1.0);
                let ratio = minor_radius / major_radius.max(1.0);
                let ellipse_is_directional = previous_ratio.min(ratio) < 0.96;
                let angle_shift = wrapped_angle_distance(previous.angle, angle);
                center_shift <= center_limit
                    && radius_shift <= 0.08
                    && (previous_ratio - ratio).abs() <= 0.06
                    && (!ellipse_is_directional || angle_shift <= 0.16)
            });
            let image_still = comparable_to_previous
                && motion_score.is_finite()
                && motion_score <= ROTATION_POSE_STILL_MOTION_MAX;
            self.stable_frames = if geometrically_still && image_still {
                self.stable_frames.saturating_add(1)
            } else {
                0
            };
            self.previous_ring = Some(observation);
            self.observations.push_back(observation);
            accepted_observation = Some(observation);
        } else {
            self.stable_frames = 0;
            self.previous_ring = None;
        }
        while self.observations.front().is_some_and(|observation| {
            now.saturating_duration_since(observation.captured_at) > ROTATION_CENTER_HISTORY
        }) {
            self.observations.pop_front();
        }
        if self.observations.is_empty() {
            self.estimate = None;
            return None;
        }
        let firmness = rotation_pose_firmness(self.stable_frames);
        let next = infer_projected_vision_pose(&self.observations, self.estimate, firmness);
        let Some(next) = next else {
            if firmness > 0.0 {
                if let (Some(mut held), Some(observation)) = (self.estimate, accepted_observation) {
                    // A static ring cannot triangulate a new globe center, but
                    // it is excellent evidence that an already-established
                    // center and pole should stay put. Permit only a very slow
                    // pole correction for detector quantization.
                    let observed_pole = (
                        observation.center.0 - held.center.0,
                        observation.center.1 - held.center.1,
                    );
                    let pole_alpha = 0.02 * (1.0 - firmness);
                    held.pole = blend_point(held.pole, observed_pole, pole_alpha);
                    self.estimate = Some(held);
                    return self.estimate;
                }
            }
            self.estimate = None;
            return None;
        };
        self.estimate = Some(match self.estimate {
            Some(previous) => {
                // React promptly to genuine motion, then progressively harden
                // the solution as image and ring geometry remain still.
                let alpha = 0.45 - firmness * 0.40;
                ProjectedVisionPose {
                    center: blend_point(previous.center, next.center, alpha),
                    pole: blend_point(previous.pole, next.pole, alpha),
                }
            }
            None => next,
        });
        self.estimate
    }
}

pub(crate) fn rotation_pose_firmness(stable_frames: u32) -> f64 {
    if stable_frames <= ROTATION_POSE_FIRM_START_FRAMES {
        return 0.0;
    }
    ((stable_frames - ROTATION_POSE_FIRM_START_FRAMES) as f64
        / (ROTATION_POSE_FIRM_FULL_FRAMES - ROTATION_POSE_FIRM_START_FRAMES) as f64)
        .clamp(0.0, 1.0)
}

pub(crate) fn infer_projected_rotation_center(
    observations: &VecDeque<RingPoseObservation>,
) -> Option<(f64, f64)> {
    infer_projected_vision_pose(observations, None, 0.0).map(|pose| pose.center)
}

pub(crate) fn infer_projected_vision_pose(
    observations: &VecDeque<RingPoseObservation>,
    prior: Option<ProjectedVisionPose>,
    firmness: f64,
) -> Option<ProjectedVisionPose> {
    let informative = observations
        .iter()
        .filter_map(|observation| {
            let ratio = (observation.minor_radius / observation.major_radius).clamp(0.0, 1.0);
            let tilt = (1.0 - ratio * ratio).sqrt();
            (tilt >= 0.08).then(|| {
                let normal = (-observation.angle.sin(), observation.angle.cos());
                (*observation, tilt, normal)
            })
        })
        .collect::<Vec<_>>();
    if informative.len() < ROTATION_CENTER_MIN_OBSERVATIONS {
        return None;
    }
    let mut angular_diversity = 0.0f64;
    let mut center_span = 0.0f64;
    for left in &informative {
        for right in &informative {
            angular_diversity =
                angular_diversity.max((left.2 .0 * right.2 .1 - left.2 .1 * right.2 .0).abs());
            center_span = center_span.max(
                (left.0.center.0 - right.0.center.0).hypot(left.0.center.1 - right.0.center.1),
            );
        }
    }
    if angular_diversity < 0.10 || center_span < 1.5 {
        return None;
    }

    let mut best: Option<(f64, f64, ProjectedVisionPose)> = None;
    for scale_step in 0..=20 {
        let depth_to_ring_radius = 1.15 + scale_step as f64 * 0.085;
        let latest = informative.last()?;
        for initial_sign in [-1.0, 1.0] {
            let latest_offset = depth_to_ring_radius * latest.0.major_radius * latest.1;
            let mut center = (
                latest.0.center.0 + initial_sign * latest_offset * latest.2 .0,
                latest.0.center.1 + initial_sign * latest_offset * latest.2 .1,
            );
            let mut selected = Vec::with_capacity(informative.len());
            for _ in 0..4 {
                selected.clear();
                for (observation, tilt, normal) in &informative {
                    let offset = depth_to_ring_radius * observation.major_radius * tilt;
                    let plus = (
                        observation.center.0 + offset * normal.0,
                        observation.center.1 + offset * normal.1,
                    );
                    let minus = (
                        observation.center.0 - offset * normal.0,
                        observation.center.1 - offset * normal.1,
                    );
                    selected.push(
                        if (plus.0 - center.0).hypot(plus.1 - center.1)
                            <= (minus.0 - center.0).hypot(minus.1 - center.1)
                        {
                            plus
                        } else {
                            minus
                        },
                    );
                }
                let mut xs = selected.iter().map(|point| point.0).collect::<Vec<_>>();
                let mut ys = selected.iter().map(|point| point.1).collect::<Vec<_>>();
                center = (median_focus(&mut xs), median_focus(&mut ys));
            }
            let mut residuals = selected
                .iter()
                .map(|point| (point.0 - center.0).hypot(point.1 - center.1))
                .collect::<Vec<_>>();
            let residual = median_focus(&mut residuals);
            let pole = (latest.0.center.0 - center.0, latest.0.center.1 - center.1);
            let pose = ProjectedVisionPose { center, pole };
            let temporal_penalty = prior
                .map(|prior| projected_pose_temporal_penalty(prior, pose, latest.0.major_radius))
                .unwrap_or(0.0)
                * firmness.clamp(0.0, 1.0);
            let score = residual + temporal_penalty;
            if best.as_ref().is_none_or(|candidate| score < candidate.0) {
                best = Some((score, residual, pose));
            }
        }
    }
    best.filter(|(_, residual, _)| *residual <= 5.0)
        .map(|(_, _, pose)| pose)
}

pub(crate) fn projected_pose_temporal_penalty(
    prior: ProjectedVisionPose,
    candidate: ProjectedVisionPose,
    ring_radius: f64,
) -> f64 {
    let center_translation =
        (candidate.center.0 - prior.center.0).hypot(candidate.center.1 - prior.center.1);
    let pole_translation = (candidate.pole.0 - prior.pole.0).hypot(candidate.pole.1 - prior.pole.1);
    let prior_length = prior.pole.0.hypot(prior.pole.1);
    let candidate_length = candidate.pole.0.hypot(candidate.pole.1);
    let pole_angle_change_rad = if prior_length > 1.0e-6 && candidate_length > 1.0e-6 {
        ((prior.pole.0 * candidate.pole.0 + prior.pole.1 * candidate.pole.1)
            / (prior_length * candidate_length))
            .clamp(-1.0, 1.0)
            .acos()
    } else {
        0.0
    };
    center_translation * 0.55 + pole_translation * 0.30 + pole_angle_change_rad * ring_radius.max(1.0) * 0.75
}

pub(crate) fn resolve_projected_surface_normal(
    projected_gaze_pole: Option<(f64, f64)>,
    boundary_center: (f64, f64),
    rotation_center: (f64, f64),
    limbus_plane_offset_px: f64,
) -> Option<[f64; 3]> {
    let projected_gaze_pole = projected_gaze_pole.filter(|pole| {
        pole.0.is_finite() && pole.1.is_finite() && pole.0.hypot(pole.1) >= limbus_plane_offset_px * 0.03
    });
    let (mut normal_x, mut normal_y) = projected_gaze_pole
        .map(|pole| (pole.0 / limbus_plane_offset_px, pole.1 / limbus_plane_offset_px))
        .unwrap_or_else(|| {
            (
                (boundary_center.0 - rotation_center.0) / limbus_plane_offset_px,
                (boundary_center.1 - rotation_center.1) / limbus_plane_offset_px,
            )
        });
    let projected_length = normal_x.hypot(normal_y);
    if !projected_length.is_finite() {
        return None;
    }
    if projected_length < 0.03 {
        return Some([0.0, 0.0, 1.0]);
    }
    if projected_length > 0.85 {
        normal_x *= 0.85 / projected_length;
        normal_y *= 0.85 / projected_length;
    }
    let normal_z_squared = 1.0 - normal_x * normal_x - normal_y * normal_y;
    if !normal_z_squared.is_finite() || normal_z_squared <= 0.0 {
        return None;
    }
    Some([normal_x, normal_y, normal_z_squared.sqrt()])
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RotationRenderGeometry {
    pub(crate) boundary_center: (f64, f64),
    pub(crate) boundary_radius: f64,
    pub(crate) sphere_radius: f64,
    pub(crate) limbus_plane_offset_px: f64,
    pub(crate) rotation_center: (f64, f64),
}

/// Proof that a proposed eye sphere exposes a convex surface to the camera.
/// The limbus slice is the local Z=0 plane. A valid globe center is behind
/// that plane, its outward pole is in the +Z half-space, and the supplied
/// normal agrees with the radius from the globe center toward the visible
/// limbus. Invalid higher-authority candidates must be discarded before
/// scoring or presentation so a lower-authority valid branch can be tried.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraFacingConvexContact {
    pub(crate) rotation_center_z: f64,
}

/// Invalid solver candidates remain rejectable. An invalid contact submitted
/// to the shared contact/laser renderer is a fatal invariant violation, even
/// in an optimized build or on a worker thread.
pub(crate) fn require_convex_contact_for_output(
    geometry: RotationRenderGeometry,
    relative_gaze: RelativeGazeVector,
    rotation_center_z: Option<f64>,
) -> CameraFacingConvexContact {
    if let Some(proof) = camera_facing_convex_contact(geometry, relative_gaze, rotation_center_z) {
        return proof;
    }
    let report = format!(
        "FATAL CONTACT INVARIANT: inward/concave or invalid contact reached output\ngeometry={geometry:?}\ngaze={relative_gaze:?}\nrotation_center_z={rotation_center_z:?}\nbacktrace={}\n",
        std::backtrace::Backtrace::force_capture(),
    );
    eprintln!("{report}");
    #[cfg(not(test))]
    {
        // Runtime evidence stays outside the source checkout. Preserve each
        // incident independently rather than overwriting the previous crash.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let directory = concat!(env!("CARGO_MANIFEST_DIR"), "/outputs/contact-invariant");
        let path = format!("{directory}/{}-{stamp}.txt", std::process::id());
        let saved =
            std::fs::create_dir_all(directory).and_then(|()| std::fs::write(&path, &report));
        match saved {
            Ok(()) => eprintln!("fatal contact evidence saved: {path}"),
            Err(error) => eprintln!("failed to save fatal contact evidence: {error}"),
        }
        // Abort is deliberate: a thread panic could otherwise leave the
        // viewer running with a dead prediction or presentation worker.
        std::process::abort();
    }
    #[cfg(test)]
    panic!("{report}");
}

pub(crate) fn camera_facing_convex_contact(
    geometry: RotationRenderGeometry,
    relative_gaze: RelativeGazeVector,
    proposed_rotation_center_z: Option<f64>,
) -> Option<CameraFacingConvexContact> {
    if !relative_gaze.is_camera_facing() {
        return None;
    }
    let normal = relative_gaze.as_array();
    let rotation_center_z =
        proposed_rotation_center_z.unwrap_or(-geometry.limbus_plane_offset_px * relative_gaze.toward_camera);
    if !rotation_center_z.is_finite() || rotation_center_z >= -CAMERA_FACING_NORMAL_MIN_Z {
        return None;
    }

    // The outward pole itself must emerge in front of the observed limbus
    // plane; otherwise this is the rear/concave sphere intersection.
    let apex_z = rotation_center_z + geometry.sphere_radius * relative_gaze.toward_camera;
    if !apex_z.is_finite() || apex_z <= CAMERA_FACING_NORMAL_MIN_Z {
        return None;
    }

    let center_to_limbus = [
        geometry.boundary_center.0 - geometry.rotation_center.0,
        geometry.boundary_center.1 - geometry.rotation_center.1,
        -rotation_center_z,
    ];
    let center_to_limbus_length =
        (center_to_limbus[0].powi(2) + center_to_limbus[1].powi(2) + center_to_limbus[2].powi(2))
            .sqrt();
    if !center_to_limbus_length.is_finite()
        || center_to_limbus_length <= CAMERA_FACING_NORMAL_MIN_Z
    {
        return None;
    }
    let outward_alignment = dot3(normal, center_to_limbus) / center_to_limbus_length;
    if !outward_alignment.is_finite() || outward_alignment <= CAMERA_FACING_NORMAL_MIN_Z {
        return None;
    }
    Some(CameraFacingConvexContact { rotation_center_z })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProvisionalSurfacePose {
    pub(crate) rotation_center: (f64, f64),
    pub(crate) relative_gaze: RelativeGazeVector,
    pub(crate) sphere_radius: f64,
}

pub(crate) fn provisional_surface_pose(
    surface_gaze: Option<SurfaceGazeSample>,
    outer_prediction_points: &[(f64, f64)],
) -> Option<ProvisionalSurfacePose> {
    let surface_gaze = surface_gaze?;
    if !surface_gaze.relative_gaze.is_camera_facing() {
        return None;
    }
    if outer_prediction_points.len() < 8 {
        return None;
    }
    let boundary_center = outer_prediction_points
        .iter()
        .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
    let boundary_center = (
        boundary_center.0 / outer_prediction_points.len() as f64,
        boundary_center.1 / outer_prediction_points.len() as f64,
    );
    let projected_boundary_radius = outer_prediction_points
        .iter()
        .map(|point| (point.0 - boundary_center.0).hypot(point.1 - boundary_center.1))
        .sum::<f64>()
        / outer_prediction_points.len() as f64;
    let sphere_radius = surface_gaze.quantized_frontal_disk_radius_px * 1.83;
    let limbus_plane_offset_squared_px2 =
        sphere_radius * sphere_radius - projected_boundary_radius * projected_boundary_radius;
    if !sphere_radius.is_finite() || !limbus_plane_offset_squared_px2.is_finite() || limbus_plane_offset_squared_px2 <= 0.0
    {
        return None;
    }
    let mut normal_x = surface_gaze.relative_gaze.right;
    let mut normal_y = surface_gaze.relative_gaze.down;
    let projected_length = normal_x.hypot(normal_y);
    if !projected_length.is_finite() {
        return None;
    }
    if projected_length > 0.85 {
        normal_x *= 0.85 / projected_length;
        normal_y *= 0.85 / projected_length;
    }
    let limbus_plane_offset_px = limbus_plane_offset_squared_px2.sqrt();
    Some(ProvisionalSurfacePose {
        rotation_center: (
            boundary_center.0 - limbus_plane_offset_px * normal_x,
            boundary_center.1 - limbus_plane_offset_px * normal_y,
        ),
        relative_gaze: RelativeGazeVector::from_projected(normal_x, normal_y)?,
        sphere_radius,
    })
}

pub(crate) fn relative_gaze_for_contact(
    rotation_center: (f64, f64),
    projected_gaze_pole: Option<(f64, f64)>,
    sphere_radius: Option<f64>,
    boundary: &[(f64, f64)],
    frame_width: usize,
    frame_height: usize,
    surface_fallback: Option<SurfaceGazeSample>,
) -> Option<RelativeGazeVector> {
    let geometry = resolve_rotation_render_geometry(
        Some(rotation_center),
        sphere_radius,
        &[],
        boundary,
        frame_width,
        frame_height,
    );
    if let Some(geometry) = geometry {
        if let Some(normal) = resolve_projected_surface_normal(
            projected_gaze_pole,
            geometry.boundary_center,
            geometry.rotation_center,
            geometry.limbus_plane_offset_px,
        ) {
            return RelativeGazeVector::from_projected(normal[0], normal[1])
                .filter(|gaze| gaze.is_camera_facing());
        }
    }
    surface_fallback
        .map(|surface| surface.relative_gaze)
        .filter(|gaze| gaze.is_camera_facing())
}

pub(crate) fn resolve_rotation_render_geometry(
    rotation_center: Option<(f64, f64)>,
    sphere_radius_override: Option<f64>,
    inner_ring_points: &[(f64, f64)],
    outer_prediction_points: &[(f64, f64)],
    frame_width: usize,
    frame_height: usize,
) -> Option<RotationRenderGeometry> {
    if outer_prediction_points.len() < 8 {
        return None;
    }
    let boundary_center = outer_prediction_points
        .iter()
        .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
    let boundary_center = (
        boundary_center.0 / outer_prediction_points.len() as f64,
        boundary_center.1 / outer_prediction_points.len() as f64,
    );
    let boundary_radius = outer_prediction_points
        .iter()
        .map(|point| (point.0 - boundary_center.0).hypot(point.1 - boundary_center.1))
        .sum::<f64>()
        / outer_prediction_points.len() as f64;
    if !boundary_radius.is_finite() || boundary_radius < 4.0 {
        return None;
    }
    let boundary_radius = boundary_radius.min(frame_width.min(frame_height) as f64 * 0.48);

    const SPHERE_TO_RING_RADIUS: f64 = 1.83;
    let sphere_radius = sphere_radius_override
        .filter(|radius| radius.is_finite() && *radius > boundary_radius)
        .unwrap_or(boundary_radius * SPHERE_TO_RING_RADIUS);
    let limbus_plane_offset_squared_px2 = sphere_radius * sphere_radius - boundary_radius * boundary_radius;
    if !limbus_plane_offset_squared_px2.is_finite() || limbus_plane_offset_squared_px2 <= 0.0 {
        return None;
    }
    let limbus_plane_offset_px = limbus_plane_offset_squared_px2.sqrt();
    let inferred_rotation_center = infer_rotation_center_from_inner_ring(
        inner_ring_points,
        boundary_center,
        sphere_radius,
        limbus_plane_offset_px,
    );
    // Callers only supply a projected center after an independently admitted
    // motion lock, a manual XYZ lock, or the explicitly provisional limbus
    // surface solve above. Coincidence with the ring center is the legitimate
    // straight-at-camera case, not missing geometry.
    let rotation_center = rotation_center
        .filter(|center| center.0.is_finite() && center.1.is_finite())
        .or(inferred_rotation_center)?;
    if !rotation_center.0.is_finite() || !rotation_center.1.is_finite() {
        return None;
    }

    Some(RotationRenderGeometry {
        boundary_center,
        boundary_radius,
        sphere_radius,
        limbus_plane_offset_px,
        rotation_center,
    })
}

pub(crate) fn infer_rotation_center_from_inner_ring(
    inner_ring_points: &[(f64, f64)],
    boundary_center: (f64, f64),
    sphere_radius: f64,
    limbus_plane_offset_px: f64,
) -> Option<(f64, f64)> {
    if inner_ring_points.len() < 8 {
        return None;
    }
    let inner_center = inner_ring_points
        .iter()
        .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
    let apex = (
        inner_center.0 / inner_ring_points.len() as f64,
        inner_center.1 / inner_ring_points.len() as f64,
    );
    let apex_above_slice = sphere_radius - limbus_plane_offset_px;
    if apex_above_slice <= 1.0e-6 {
        return None;
    }
    let mut normal_x = (apex.0 - boundary_center.0) / apex_above_slice;
    let mut normal_y = (apex.1 - boundary_center.1) / apex_above_slice;
    let projected_length = normal_x.hypot(normal_y);
    if !projected_length.is_finite() || projected_length < 0.03 {
        return None;
    }
    if projected_length > 0.75 {
        normal_x *= 0.75 / projected_length;
        normal_y *= 0.75 / projected_length;
    }
    Some((
        boundary_center.0 - limbus_plane_offset_px * normal_x,
        boundary_center.1 - limbus_plane_offset_px * normal_y,
    ))
}

/// Presentation-only physical scale. MediaPipe reports an image-space iris
/// radius but not a subject-specific physical diameter, so the initial
/// projection assumes a 12 mm limbus. The interval is intentionally broad and
/// changes only at semantic reacquisition boundaries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CentimeterScaleEstimate {
    pub(crate) estimate_px: f64,
    pub(crate) minimum_px: f64,
    pub(crate) maximum_px: f64,
    pub(crate) movement_fraction: f64,
    pub(crate) reacquisition_count: u32,
    pub(crate) semantic_center_sensor: [f64; 2],
}

pub(crate) const ROUGH_LIMBUS_DIAMETER_MM: f64 = 12.0;

pub(crate) fn centimeter_scale_half_width(
    previous: Option<CentimeterScaleEstimate>,
    movement_fraction: f64,
) -> f64 {
    if let Some(old) = previous {
        let old_half_width = (old.maximum_px - old.minimum_px) / (2.0 * old.estimate_px.max(1.0));
        if movement_fraction <= 0.08 {
            (old_half_width * 0.82).clamp(0.12, 0.25)
        } else {
            (0.14 + movement_fraction * 0.55)
                .max(old_half_width)
                .clamp(0.14, 0.65)
        }
    } else {
        0.25
    }
}

/// Coarse image measurement, deliberately independent of the MediaPipe adapter.
/// Full-sensor pixel coordinates and apparent radius, not a metric eye pose.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoarseEyeScaleSeed {
    pub(crate) center_sensor: (f64, f64),
    pub(crate) iris_radius_px: f64,
}

pub(crate) fn coarse_centimeter_scales(
    previous_scales: [Option<CentimeterScaleEstimate>; 2],
    old_centers: [(f64, f64); 2],
    eyes: [CoarseEyeScaleSeed; 2],
) -> [Option<CentimeterScaleEstimate>; 2] {
    let sensor_centers = eyes.map(|eye| eye.center_sensor);
    let interocular_span = (sensor_centers[0].0 - sensor_centers[1].0)
        .hypot(sensor_centers[0].1 - sensor_centers[1].1)
        .max(1.0);
    std::array::from_fn(|index| {
        let radius_px = eyes[index].iris_radius_px;
        let estimate_px = radius_px * 20.0 / ROUGH_LIMBUS_DIAMETER_MM;
        if !estimate_px.is_finite() || !(4.0..=2_000.0).contains(&estimate_px) {
            return None;
        }
        let previous_scale = previous_scales[index];
        let prior_center = previous_scale
            .map(|old| (old.semantic_center_sensor[0], old.semantic_center_sensor[1]))
            .unwrap_or(old_centers[index]);
        let center_motion = (sensor_centers[index].0 - prior_center.0)
            .hypot(sensor_centers[index].1 - prior_center.1)
            / interocular_span;
        let scale_motion = previous_scale
            .map(|old| (estimate_px / old.estimate_px).ln().abs())
            .unwrap_or(0.0);
        let movement_fraction = (center_motion + scale_motion).clamp(0.0, 2.0);
        // A first anthropometric projection starts at +/-25%. Consistent
        // reacquisitions narrow it toward +/-12%; head translation, apparent
        // scale change, or a new pose broadens it as far as +/-65%.
        let fractional_half_width = centimeter_scale_half_width(previous_scale, movement_fraction);
        Some(CentimeterScaleEstimate {
            estimate_px,
            minimum_px: estimate_px * (1.0 - fractional_half_width),
            maximum_px: estimate_px * (1.0 + fractional_half_width),
            movement_fraction,
            reacquisition_count: previous_scale
                .map(|old| old.reacquisition_count.saturating_add(1))
                .unwrap_or(1),
            semantic_center_sensor: [sensor_centers[index].0, sensor_centers[index].1],
        })
    })
}
