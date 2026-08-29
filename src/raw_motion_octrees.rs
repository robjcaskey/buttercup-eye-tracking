//! Four persistent RAW10-luma motion octrees for the live eye viewer.
//!
//! This is deliberately an image-measurement stage.  It uses native 1x1 RAW
//! samples, keeps bounded feature trails in absolute sensor coordinates, fits
//! a similarity motion (translation, rotation, and scale) per object, and uses
//! differential/parallax motion as the third octree coordinate.  It does not
//! claim metric depth or semantic object identity.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

#[path = "coupled_eye_kinematics.rs"]
mod coupled_eye_kinematics;
pub use coupled_eye_kinematics::{
    CoupledEyeKinematics, CoupledMotionStatus, GlobeMotionRegime, KinematicDerivatives,
    ProjectedGlobePoseStatus, ProjectedIrisGeometry, RotationCenterStatus,
};

pub const OBJECTS: usize = 4;
pub const GENERAL_LAYER: usize = 0;
pub const PUPIL_LAYER: usize = 1;
pub const REFLECTION_LAYER: usize = 2;
pub const RESIDUAL_LAYER: usize = 3;

/// Runtime-selectable native-RAW edge definition for the 2D temporal learning
/// layers.  This is deliberately a calculation profile, not a renderer mode:
/// changing it changes which current-frame Canny samples may seed and sustain
/// feature tracks.  Other segmentation modes retain their established
/// balanced Canny calculation unless they explicitly opt into this type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LearningCannyProfile {
    #[default]
    CannyBalanced,
    CannySensitive,
    CannyStrict,
    SobelSharp,
    SobelSmooth,
    ScharrCanny,
    Laplacian,
    DifferenceOfGaussians,
    GradientCompass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CannyGradientKernel {
    Sobel,
    Scharr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CannyInputFilter {
    Raw,
    Gaussian(u8),
    Laplacian,
    DifferenceOfGaussians,
}

#[derive(Clone, Copy, Debug)]
struct LearningCannySettings {
    input: CannyInputFilter,
    kernel: CannyGradientKernel,
    high_quantile: f64,
    low_ratio: f32,
    high_floor: f32,
}

impl LearningCannyProfile {
    fn settings(self) -> LearningCannySettings {
        match self {
            // Balanced is the exact historical learning-layer calculation.
            Self::CannyBalanced => LearningCannySettings {
                input: CannyInputFilter::Gaussian(1),
                kernel: CannyGradientKernel::Scharr,
                high_quantile: 0.82,
                low_ratio: 0.36,
                high_floor: 48.0,
            },
            Self::CannySensitive => LearningCannySettings {
                input: CannyInputFilter::Gaussian(1),
                kernel: CannyGradientKernel::Scharr,
                high_quantile: 0.68,
                low_ratio: 0.30,
                high_floor: 28.0,
            },
            Self::CannyStrict => LearningCannySettings {
                input: CannyInputFilter::Gaussian(1),
                kernel: CannyGradientKernel::Scharr,
                high_quantile: 0.92,
                low_ratio: 0.58,
                high_floor: 72.0,
            },
            Self::SobelSharp => LearningCannySettings {
                input: CannyInputFilter::Raw,
                kernel: CannyGradientKernel::Sobel,
                high_quantile: 0.80,
                low_ratio: 0.38,
                // A Sobel response is approximately one quarter of Scharr's
                // response for the same step on this RAW10 scale.
                high_floor: 12.0,
            },
            Self::SobelSmooth => LearningCannySettings {
                input: CannyInputFilter::Gaussian(2),
                kernel: CannyGradientKernel::Sobel,
                high_quantile: 0.80,
                low_ratio: 0.40,
                high_floor: 10.0,
            },
            Self::ScharrCanny => LearningCannySettings {
                input: CannyInputFilter::Gaussian(1),
                kernel: CannyGradientKernel::Scharr,
                high_quantile: 0.86,
                low_ratio: 0.42,
                high_floor: 60.0,
            },
            Self::Laplacian => LearningCannySettings {
                input: CannyInputFilter::Laplacian,
                kernel: CannyGradientKernel::Scharr,
                high_quantile: 0.84,
                low_ratio: 0.42,
                high_floor: 24.0,
            },
            Self::DifferenceOfGaussians => LearningCannySettings {
                input: CannyInputFilter::DifferenceOfGaussians,
                kernel: CannyGradientKernel::Scharr,
                high_quantile: 0.82,
                low_ratio: 0.35,
                high_floor: 18.0,
            },
            Self::GradientCompass => LearningCannySettings {
                input: CannyInputFilter::Raw,
                kernel: CannyGradientKernel::Scharr,
                high_quantile: 0.90,
                low_ratio: 0.18,
                high_floor: 80.0,
            },
        }
    }
}
// Eighty spatially tiled native-RAW tracks leave ample support for four
// independent motion layers while bounding the quadratic patch-search work.
// Offline 112-track replays regularly saturated the budget without improving
// the number of coherent layers.
const MAX_FEATURES: usize = 80;
// A dormant identity needs enough pre-occlusion history for nearby surviving
// material to transport it across an ordinary blink.  This is coordinate
// state only; no additional RAW frames are retained.
const MAX_TRAIL: usize = 32;
const MAX_AGE: u8 = 3;
const MAX_FEATURE_DORMANT_AGE: u8 = 24;
const MAX_FEATURE_DORMANT_NS: u64 = 750_000_000;
// Keep only the strongest pre-occlusion identities beyond the ordinary
// three-frame gap. Dormant tracks are cheap (they skip adjacent patch search),
// but letting all eighty occupy the detector budget would prevent fresh
// current-frame features from being seeded during unrelated local losses.
const MAX_EXTENDED_DORMANT_FEATURES: usize = 24;
const PATCH_RADIUS: i32 = 7;
const PYRAMID_PATCH_RADIUS: i32 = 9;
const PREDICTED_PATCH_RADIUS: i32 = 5;
const PREDICTED_PYRAMID_PATCH_RADIUS: i32 = 6;
const SEARCH_RADIUS: i32 = 16;
const MIN_FEATURE_SEPARATION: f32 = 8.0;
// Forward/backward consistency is the ambiguity test. A conventional
// second-best margin rejects exactly the long, locally one-dimensional Canny
// contours whose motion we need to watch over time.
const MIN_MATCH_MARGIN: f32 = 0.0;
const MAX_MATCH_COST: f32 = 0.72;
const MATCH_EXCLUSION_RADIUS: f32 = 2.5;
// The integer ZNCC search finds the correct native-RAW correlation basin. A
// positive-definite quadratic fitted to its 3x3 native cost neighborhood then
// estimates the peak below the sensor-pixel grid. The eigenvalue-ratio gate is
// deliberately strict: a long edge may provide accurate normal flow, but it
// must not invent a two-dimensional fractional feature position along its
// unconstrained tangent.
const SUBPIXEL_MIN_CURVATURE: f32 = 1.0e-4;
const SUBPIXEL_MIN_CURVATURE_RATIO: f32 = 0.025;
const SUBPIXEL_MAX_OFFSET: f32 = 0.75;
// Seeds begin at least eight pixels apart. If independently searched tracks
// land this close together, they have converged onto the same Canny ridge and
// must not both contribute a duplicate motion signature.
const MIN_MATCH_DESTINATION_SEPARATION: f32 = 3.5;
const MIN_NEW_TRACK_CANNY_SUPPORT: f32 = 0.20;
const MIN_PERSISTENT_CANNY_SUPPORT: f32 = 0.055;
const EDGE_TILE_SIZE: usize = 24;
const EDGES_PER_TILE: usize = 18;
const FEATURE_SEED_TILE_SIZE: usize = 48;
// The general graph remains capped at eighty tracks, but an anatomically
// plausible iris-to-sclera transition receives a lower *selection* cost so a
// dense lid, brow, or glasses ridge cannot crowd every limbus observation out
// of that fixed budget.  The discount is intentionally bounded: these are
// still proposals, and an inaccurate iris seed must not manufacture anatomy.
const LIMBUS_FEATURE_RESERVE: usize = 24;
const LIMBUS_FEATURE_SECTORS: usize = 24;
const LIMBUS_FEATURE_MAX_PER_TILE: u8 = 3;
const LIMBUS_FEATURE_MAX_PER_SECTOR: u8 = 2;
const LIMBUS_FEATURE_MIN_AFFINITY: f32 = 0.30;
const LIMBUS_FEATURE_MAX_COST_DISCOUNT: f32 = 0.45;
// A nautilus fingerprint follows four slowly rotating, opposed branches over
// four outward shells.  Pair means retain the regional material while signed
// pair differences retain direction; z-normalization makes the 32-byte result
// insensitive to affine exposure/gain changes.  A short per-ID observation
// bank permits honest re-identification after a missed exposure without
// pretending the old point belongs to the immediately previous RAW frame.
const NAUTILUS_SHELLS: usize = 6;
const NAUTILUS_BRANCH_PAIRS: usize = 8;
const NAUTILUS_COMPONENTS: usize = NAUTILUS_SHELLS * NAUTILUS_BRANCH_PAIRS * 2;
const NAUTILUS_BANK_CAPACITY: usize = 5;
const NAUTILUS_TREE_LEAF_CAPACITY: usize = 16;
const NAUTILUS_MAX_CANDIDATES: usize = 2048;
const NAUTILUS_CANDIDATE_SEPARATION: f32 = 1.5;
const NAUTILUS_MIN_BANK_OBSERVATIONS: usize = 2;
const NAUTILUS_MAX_DISTANCE: f32 = 0.78;
const NAUTILUS_MIN_ABSOLUTE_MARGIN: f32 = 0.08;
const NAUTILUS_MAX_DISTANCE_RATIO: f32 = 0.84;
const NAUTILUS_MIN_REVERSE_MARGIN: f32 = 0.04;
const NAUTILUS_MIN_ACCEPTANCE_CONFIDENCE: f32 = 0.85;
const NAUTILUS_IDENTITY_RADIUS_BASE: f32 = 2.0;
const NAUTILUS_IDENTITY_RADIUS_PER_GAP: f32 = 0.5;
const NAUTILUS_EXTENDED_GAP_IDENTITY_RADIUS_MAX: f32 = 6.5;
const NAUTILUS_ANCHOR_PRIOR_DISAGREEMENT_BASE: f32 = 6.0;
const NAUTILUS_ANCHOR_PRIOR_DISAGREEMENT_PER_GAP: f32 = 4.0;
const NAUTILUS_ANCHOR_RESIDUAL_BASE: f32 = 2.5;
const NAUTILUS_ANCHOR_RESIDUAL_PER_GAP: f32 = 0.75;
// Horizontal light-field walks stay on the native RAW lattice.  A broad
// linear box kernel is evaluated first; only textured, well-correlated
// regions descend to narrower kernels.  The tree is deliberately small
// enough to act as a gross search prior without competing with identity-
// bearing point patches for the frame budget.
const LIGHT_FIELD_MAX_DEPTH: u8 = 3;
const LIGHT_FIELD_MAX_NODES: usize = 64;
const LIGHT_FIELD_MIN_WIDTH: f32 = 56.0;
const LIGHT_FIELD_MIN_HEIGHT: f32 = 36.0;
const LIGHT_FIELD_BLUR_RADIUS: [i32; 4] = [12, 7, 3, 1];
const LIGHT_FIELD_SAMPLE_STEP: [usize; 4] = [8, 6, 4, 3];
const LIGHT_FIELD_TRANSLATION_RADIUS: [f32; 4] = [24.0, 6.0, 3.0, 1.5];
const LIGHT_FIELD_TRANSLATION_STEP: [f32; 4] = [4.0, 2.0, 1.0, 0.5];
const LIGHT_FIELD_SCALE_RADIUS: [f32; 4] = [0.08, 0.030, 0.015, 0.0075];
const LIGHT_FIELD_SCALE_STEP: [f32; 4] = [0.020, 0.010, 0.005, 0.0025];
const LIGHT_FIELD_VERTICAL_RADIUS: [f32; 4] = [8.0, 4.0, 2.0, 1.0];
const LIGHT_FIELD_VERTICAL_STEP: [f32; 4] = [4.0, 2.0, 1.0, 1.0];
const LIGHT_FIELD_MIN_CORRELATION: f32 = 0.58;
const LIGHT_FIELD_MIN_TEXTURE: f32 = 0.010;
const LIGHT_FIELD_BRANCH_CONFIDENCE: f32 = 0.24;
// Twenty-four angular lanes provide opposing-meridian leverage without
// turning the limbus into a dense contour tracer. Every lane samples the
// native RAW field at the boundary plus five points inward and five outward.
// The descriptor compares normalized gradients, so exposure/gain changes do
// not erase a broad translucent iris-to-sclera transition.
const RADIAL_LIMBUS_SECTORS: usize = 24;
pub const RADIAL_LIMBUS_HALF_SAMPLES: usize = 5;
const RADIAL_LIMBUS_SAMPLES: usize = RADIAL_LIMBUS_HALF_SAMPLES * 2 + 1;
const RADIAL_LIMBUS_GRADIENTS: usize = RADIAL_LIMBUS_SAMPLES - 1;
pub const RADIAL_LIMBUS_SAMPLE_SPACING_PX: f32 = 1.5;
const RADIAL_LIMBUS_REFERENCE_SEARCH_PX: i32 = 4;
const RADIAL_LIMBUS_CURRENT_SEARCH_PX: i32 = 8;
const RADIAL_LIMBUS_MIN_QUALITY: f32 = 0.10;
const RADIAL_LIMBUS_MAX_PROFILE_COST: f32 = 0.46;
const RADIAL_LIMBUS_MAX_PRIOR_NORMAL_ERROR_PX: f32 = 5.0;
const RADIAL_LIMBUS_MIN_JOINT_SUPPORT: usize = 5;
const RADIAL_LIMBUS_MAX_FUSED: usize = 12;
const ELLIPSE_ANGLE_BINS: usize = 24;
const MOTION_SIGNATURE_LEN: usize = 8;
const MIN_MOTION_SIGNATURE: usize = 4;
const MIN_LAYER_SUPPORT: usize = 3;
const MIN_LAYER_PERSISTENT_TRACKS: usize = 3;
const MIN_LAYER_STABLE_FRAMES: u16 = 2;
const MAX_LAYER_RESIDUAL: f32 = 2.8;
const MIN_LAYER_SEPARATION: f32 = 0.30;
const MIN_SIGNATURE_SEED_SEPARATION: f32 = 0.55;
const MAX_SIGNATURE_MEMBER_ERROR: f32 = 4.5;
const SEMANTIC_MOTION_CORE_RADIUS: f32 = 3.6;
// A relation edge is inferred from two native/subpixel feature
// correspondences, then cross-examined against every other current
// correspondence.  The complete graph is still bounded: MAX_FEATURES=80
// gives at most 3,160 pair hypotheses and 252,800 point predictions.
const RELATION_MIN_BASELINE: f32 = 7.0;
const RELATION_MAX_BASELINE: f32 = 220.0;
const RELATION_INLIER_RADIUS: f32 = 1.65;
const RELATION_COMPONENT_INLIER_RADIUS: f32 = 0.36;
const RELATION_MIN_COMPONENT_SUPPORT: usize = 4;
const RELATION_STRONG_EDGE_COHERENCE: f32 = 0.36;
const RELATION_STRONG_EDGE_RESIDUAL: f32 = 1.35;
const RELATION_MAX_EDGE_AGE: u8 = 2;
const RELATION_MIN_TRACK_STREAK: u8 = 2;
const RELATION_ORIGIN_MIN_RATE: f32 = 0.0025;
// Anatomical naming is stateful even though every tensor is measured from the
// current full-resolution RAW pair.  A component may disappear briefly when
// its texture is specular or a lid crosses it, but a different material cohort
// must not take over the iris name on the next exposure merely because it also
// forms a precise transform.
const RELATION_IRIS_IDENTITY_MAX_AGE: u8 = 12;
const RELATION_IRIS_IDENTITY_MIN_CONFIRMATIONS: u16 = 2;
const RELATION_IRIS_IDENTITY_MIN_OVERLAP: f32 = 0.10;
const RELATION_IRIS_IDENTITY_STRONG_OVERLAP: f32 = 0.30;
const RELATION_IRIS_IDENTITY_MAX_CENTROID_STEP_RADII: f32 = 0.55;
const RELATION_IRIS_IDENTITY_MAX_ORIGIN_STEP_RADII: f32 = 0.55;
const RELATION_IRIS_INITIAL_MAX_ORIGIN_OFFSET_RADII: f32 = 0.80;
const RELATION_IRIS_MIN_DIFFERENTIAL_PX: f32 = 0.40;
const RELATION_IRIS_FULL_DIFFERENTIAL_PX: f32 = 0.55;
const RELATION_IRIS_MIN_COMPONENT_PURITY: f32 = 0.50;
const RELATION_IRIS_UNOBSERVABLE_ORIGIN_MIN_PURITY: f32 = 0.95;
const RELATION_IRIS_MIN_OBSERVATION_EVIDENCE: f32 = 0.55;
const RELATION_IRIS_IDENTITY_MIN_EVIDENCE: f32 = 2.0;
const SPECULAR_HIGH_SCORE: f32 = 2.40;
const SPECULAR_HOLD_SCORE: f32 = 1.55;
const REFLECTION_SPATIAL_RADIUS: f32 = 32.0;
const MIN_REFLECTION_SUPPORT: usize = 2;
const LIMBUS_INNER_NORMALIZED_RADIUS: f32 = 0.64;
const LIMBUS_OUTER_NORMALIZED_RADIUS: f32 = 1.20;
const MIN_LIMBUS_RADIAL_ALIGNMENT: f32 = 0.52;
const MAX_LIMBUS_NORMAL_FLOW_ERROR: f32 = 2.8;
const MAX_LIMBUS_NORMAL_SIGNATURE_ERROR: f32 = 4.0;
const LAYER_EDGE_ASSOCIATION_RADIUS: f32 = 9.0;
const FOCUS_MIN_POSITIONS: usize = 3;
const FOCUS_MIN_POSITION_SPAN: u16 = 12;
const FOCUS_MIN_FEATURES: usize = 8;
const SFM_TRAIN_FRAMES: usize = 3;
const SFM_TEST_FRAMES: usize = 5;
const SFM_MIN_TEST_SAMPLES: usize = 24;
const SFM_ACCEPT_RATIO: f32 = 0.85;

pub fn motion_layer_label(object: usize) -> &'static str {
    match object {
        GENERAL_LAYER => "general",
        PUPIL_LAYER => "pupil/iris",
        REFLECTION_LAYER => "reflection",
        _ => "residual",
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FocusDepthProbe {
    pub position: u16,
    pub sweeping: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FocusSfmPhase {
    #[default]
    Idle,
    Collecting,
    Validating,
    Accepted,
    Rejected,
}

impl FocusSfmPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Collecting => "collecting",
            Self::Validating => "validating",
            Self::Accepted => "accepted",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FocusSfmStatus {
    pub generation: u64,
    pub phase: FocusSfmPhase,
    pub calibrated_features: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub planar_error: f32,
    pub depth_error: f32,
    pub improvement: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SimilarityMotion {
    pub translation: [f32; 2],
    pub rotation: f32,
    pub scale_delta: f32,
    pub residual: f32,
    pub support: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MotionLayerStatus {
    /// Mean image-space position of the currently associated tracks.
    pub centroid: [f32; 2],
    /// Translation relative to the robust whole-frame motion.
    pub differential: [f32; 2],
    /// Signed coordinate on the learned dominant parallax axis. This is not
    /// metric depth, but remains directionally consistent across frames.
    pub parallax: f32,
    /// Temporal agreement of member tracks with this layer's motion model.
    pub coherence: f32,
    /// Mean RMS distance between member motion histories and the layer's
    /// multi-frame signature, in full-resolution pixels.
    pub trajectory_error: f32,
    pub signature_samples: usize,
    /// Distance in motion space to the nearest other supported layer.
    pub separation: f32,
    pub persistent_tracks: usize,
    pub stable_frames: u16,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RelationIrisCandidateDiagnostics {
    pub selector_calls: usize,
    pub components_examined: usize,
    pub rejected_component_quality: usize,
    pub rejected_spatial_support: usize,
    pub rejected_purity: usize,
    pub rejected_differential: usize,
    pub rejected_rank_one: usize,
    pub rejected_identity: usize,
    pub rejected_initial_origin: usize,
    pub rejected_untrusted_origin_seed: usize,
    pub withheld_untrusted_origin: usize,
    pub rejected_score: usize,
    pub ranked_components: usize,
    pub provisional_observations: usize,
    pub authorized_observations: usize,
    pub maximum_selected_support: usize,
    pub maximum_purity: f32,
    pub maximum_differential_px: f32,
    pub maximum_two_dimensionality: f32,
    pub maximum_score: f32,
    pub initial_origin_candidates: usize,
    pub invalid_initial_origins: usize,
    pub minimum_initial_origin_offset_radii: f32,
    pub maximum_initial_origin_offset_radii: f32,
    pub finite_origin_candidates: usize,
    pub minimum_origin_offset_radii: f32,
    pub maximum_origin_offset_radii: f32,
}

impl RelationIrisCandidateDiagnostics {
    fn accumulate(&mut self, other: Self) {
        self.selector_calls = self.selector_calls.saturating_add(other.selector_calls);
        self.components_examined = self
            .components_examined
            .saturating_add(other.components_examined);
        self.rejected_component_quality = self
            .rejected_component_quality
            .saturating_add(other.rejected_component_quality);
        self.rejected_spatial_support = self
            .rejected_spatial_support
            .saturating_add(other.rejected_spatial_support);
        self.rejected_purity = self.rejected_purity.saturating_add(other.rejected_purity);
        self.rejected_differential = self
            .rejected_differential
            .saturating_add(other.rejected_differential);
        self.rejected_rank_one = self
            .rejected_rank_one
            .saturating_add(other.rejected_rank_one);
        self.rejected_identity = self
            .rejected_identity
            .saturating_add(other.rejected_identity);
        self.rejected_initial_origin = self
            .rejected_initial_origin
            .saturating_add(other.rejected_initial_origin);
        self.rejected_untrusted_origin_seed = self
            .rejected_untrusted_origin_seed
            .saturating_add(other.rejected_untrusted_origin_seed);
        self.withheld_untrusted_origin = self
            .withheld_untrusted_origin
            .saturating_add(other.withheld_untrusted_origin);
        self.rejected_score = self.rejected_score.saturating_add(other.rejected_score);
        self.ranked_components = self
            .ranked_components
            .saturating_add(other.ranked_components);
        self.provisional_observations = self
            .provisional_observations
            .saturating_add(other.provisional_observations);
        self.authorized_observations = self
            .authorized_observations
            .saturating_add(other.authorized_observations);
        self.maximum_selected_support = self
            .maximum_selected_support
            .max(other.maximum_selected_support);
        self.maximum_purity = self.maximum_purity.max(other.maximum_purity);
        self.maximum_differential_px = self
            .maximum_differential_px
            .max(other.maximum_differential_px);
        self.maximum_two_dimensionality = self
            .maximum_two_dimensionality
            .max(other.maximum_two_dimensionality);
        self.maximum_score = self.maximum_score.max(other.maximum_score);
        if other.initial_origin_candidates > 0 {
            self.minimum_initial_origin_offset_radii = if self.initial_origin_candidates == 0 {
                other.minimum_initial_origin_offset_radii
            } else {
                self.minimum_initial_origin_offset_radii
                    .min(other.minimum_initial_origin_offset_radii)
            };
            self.maximum_initial_origin_offset_radii = self
                .maximum_initial_origin_offset_radii
                .max(other.maximum_initial_origin_offset_radii);
        }
        self.initial_origin_candidates = self
            .initial_origin_candidates
            .saturating_add(other.initial_origin_candidates);
        self.invalid_initial_origins = self
            .invalid_initial_origins
            .saturating_add(other.invalid_initial_origins);
        if other.finite_origin_candidates > 0 {
            self.minimum_origin_offset_radii = if self.finite_origin_candidates == 0 {
                other.minimum_origin_offset_radii
            } else {
                self.minimum_origin_offset_radii
                    .min(other.minimum_origin_offset_radii)
            };
            self.maximum_origin_offset_radii = self
                .maximum_origin_offset_radii
                .max(other.maximum_origin_offset_radii);
        }
        self.finite_origin_candidates = self
            .finite_origin_candidates
            .saturating_add(other.finite_origin_candidates);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MatchDiagnostics {
    pub considered: usize,
    /// Build the native-resolution carrier-neutral RAW plane.
    pub neutral_micros: u64,
    /// Construct the multi-scale signed Canny field, including hysteresis.
    pub canny_micros: u64,
    /// Compact the accepted Canny field into the spatially bounded edge bank.
    pub edge_micros: u64,
    pub canny_primary_blur_micros: u64,
    pub canny_gradient_micros: u64,
    pub canny_hysteresis_micros: u64,
    pub canny_nms_micros: u64,
    pub canny_quantile_micros: u64,
    pub canny_flood_micros: u64,
    pub canny_broad_blur_micros: u64,
    pub canny_attribute_micros: u64,
    pub canny_attribute_candidates: usize,
    pub canny_attribute_evaluated: usize,
    pub canny_texture_evaluated: usize,
    pub canny_texture_micros: u64,
    pub canny_texture_simd_evaluated: usize,
    pub preprocess_micros: u64,
    pub pyramid_micros: u64,
    pub matching_micros: u64,
    /// Build and SIMD-score the sparse pairwise motion-relation graph. This
    /// excludes the semantic naming pass, which remains in `layering_micros`.
    pub relation_micros: u64,
    pub layering_micros: u64,
    pub maintenance_micros: u64,
    /// ZNCC patch evaluations performed at the half-resolution search level.
    /// The input and refinement remain native RAW; this counter makes the
    /// prediction-bounded search auditable in offline temporal replay.
    pub coarse_patch_evaluations: usize,
    /// Native-resolution ZNCC refinement evaluations.
    pub native_patch_evaluations: usize,
    /// Accepted integer-grid matches presented to the native 3x3 quadratic
    /// ZNCC peak estimator.
    pub subpixel_attempted: usize,
    /// Well-conditioned two-dimensional peaks which produced a fractional
    /// native-RAW position.
    pub subpixel_accepted: usize,
    /// Flat, saddle-shaped, or off-center peaks retained at their safe integer
    /// location instead of being assigned false fractional precision.
    pub subpixel_rejected: usize,
    /// Sum of accepted fractional correction magnitudes, for bounded replay
    /// diagnostics without retaining another per-track allocation.
    pub subpixel_offset_sum: f32,
    pub no_candidate: usize,
    pub cost_rejected: usize,
    pub margin_rejected: usize,
    pub backward_rejected: usize,
    pub temporal_rejected: usize,
    pub destination_collision_rejected: usize,
    pub accepted: usize,
    /// Multi-shell descriptor hierarchy built from current-frame native RAW
    /// edge points. These counters deliberately distinguish a withheld
    /// ambiguous identity from a confident but incorrect reconnection.
    pub nautilus_build_micros: u64,
    pub nautilus_candidates: usize,
    pub nautilus_tree_nodes: usize,
    pub nautilus_queries: usize,
    pub nautilus_nodes_visited: usize,
    pub nautilus_descriptor_evaluations: usize,
    pub nautilus_refinement_evaluations: usize,
    pub nautilus_distance_rejected: usize,
    pub nautilus_spatial_rejected: usize,
    pub nautilus_ambiguous: usize,
    pub nautilus_reverse_ambiguous: usize,
    pub nautilus_collision_rejected: usize,
    pub nautilus_relocated: usize,
    pub nautilus_gap_relocated: usize,
    pub nautilus_margin_sum: f32,
    /// Sparse native-RAW horizontal light-field hierarchy.  `evaluations`
    /// counts complete affine walk hypotheses, not individual RAW reads.
    pub light_field_micros: u64,
    pub light_field_nodes: usize,
    pub light_field_leaves: usize,
    pub light_field_evaluations: usize,
    pub light_field_reliable: bool,
    /// Established tracks whose ordinary prediction did not cover the
    /// independently measured horizontal field basin.
    pub light_field_rescue_tracks: usize,
    /// Half-resolution patch evaluations spent only in those rescue basins.
    pub light_field_rescue_evaluations: usize,
    /// Exact current-frame matches selected from a field-guided basin after
    /// the ordinary basin lost the feature.
    pub light_field_rescue_accepted: usize,
    /// Native five-in/five-out radial profile lanes evaluated around the
    /// projected iris/sclera boundary.
    pub radial_limbus_evaluations: usize,
    /// Illumination-normalized adjacent-frame radial transitions retained as
    /// current evidence before joint-motion consistency is applied.
    pub radial_limbus_accepted: usize,
    /// Accepted radial transitions actually admitted as one-dimensional
    /// normal-flow constraints to the iris similarity solve.
    pub radial_limbus_fused: usize,
    pub radial_limbus_micros: u64,
    /// Historical IDs retained without pretending their last observation is
    /// present in the current exposure.
    pub dormant_tracks: usize,
    /// Dormant IDs recovered after more than the ordinary three-frame gap.
    pub extended_gap_relocated: usize,
    /// Mature subpixel tracks admitted as nodes to the pairwise motion graph.
    pub relation_nodes: usize,
    /// Pairwise similarity tensors retained after baseline/finite checks.
    pub relation_edges: usize,
    /// Relations observed for the same feature-ID pair on at least two
    /// consecutive exposures, before independent-support quality gates.
    pub relation_recurrent_edges: usize,
    /// Recurrent relations that predict at least four current feature nodes.
    pub relation_supported_edges: usize,
    /// Recurrent, independently supported relations below the strong-edge
    /// residual limit, before temporal-coherence weighting.
    pub relation_precise_edges: usize,
    /// Temporally persistent graph edges supported by at least one third
    /// track.  A two-point transform alone is an exact construction, not
    /// independent evidence of a material component.
    pub relation_coherent_edges: usize,
    /// Strong connected material-motion components in the current frame.
    pub relation_components: usize,
    /// Components whose strong recurrent edges cover at least four members.
    /// Only these components may receive an anatomical semantic name.
    pub relation_persistent_components: usize,
    pub relation_max_component_persistent_edges: usize,
    pub relation_max_component_persistent_nodes: usize,
    pub relation_max_persistent_differential_px: f32,
    /// Exact stable-track overlap with the previously named iris component.
    /// This is not the hashed SIMD support sketch used to score tensors.
    pub relation_iris_identity_overlap: f32,
    pub relation_iris_identity_age: u8,
    pub relation_iris_identity_confirmations: u16,
    pub relation_iris_identity_evidence: f32,
    pub relation_iris_identity_switch_rejections: usize,
    pub relation_iris_initial_origin_rejections: usize,
    pub relation_iris_identity_carried: bool,
    pub relation_iris_provisional_support: usize,
    pub relation_iris_identity_confirmed: bool,
    pub relation_iris_candidates: RelationIrisCandidateDiagnostics,
    pub relation_origin_outlier_rejected: bool,
    /// Support of the component selected as iris material by spatial and
    /// temporal semantic priors. Zero means the graph did not authorize one.
    pub relation_iris_support: usize,
    /// Candidate common center of rotation/scale in ROI sensor coordinates.
    /// It remains diagnostic unless `relation_origin_valid` is true.
    pub relation_origin: [f32; 2],
    pub relation_origin_spread: f32,
    pub relation_origin_valid: bool,
    pub relation_max_shared_frames: u16,
    pub relation_max_coherence: f32,
    pub relation_mean_recurrent_coherence: f32,
    pub relation_mean_recurrent_residual: f32,
    pub relation_mean_support_continuity: f32,
}

impl SimilarityMotion {
    fn predict(self, point: [f32; 2], center: [f32; 2]) -> [f32; 2] {
        let x = point[0] - center[0];
        let y = point[1] - center[1];
        [
            point[0] + self.translation[0] + self.scale_delta * x - self.rotation * y,
            point[1] + self.translation[1] + self.rotation * x + self.scale_delta * y,
        ]
    }
}

#[derive(Clone, Debug)]
struct FeatureTrack {
    id: u64,
    points: VecDeque<[f32; 3]>, // absolute sensor x/y and relative parallax z
    object: usize,
    age: u8,
    last_seen_timestamp_ns: u64,
    score: f32,
    motion_ema: [f32; 2],
    motion_variance: f32,
    matched_streak: u8,
    layer_evidence: bool,
    /// This track constrains only motion perpendicular to a smooth edge. A
    /// limbus edge has no stable point identity along its tangent (the aperture
    /// problem), but its normal flow is still valid pupil/iris evidence.
    normal_flow_evidence: bool,
    specularity: f32,
    assignment_confidence: f32,
    edge_normal: [f32; 2],
    /// Consecutive image motion after subtracting the robust whole-frame
    /// affine motion. Clusters are learned from this trajectory, not from a
    /// single frame's Canny geometry or displacement.
    residual_history: VecDeque<[f32; 2]>,
    focus_bins: Vec<FocusBin>,
    focus_peak: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
struct NautilusFingerprint {
    values: [i8; NAUTILUS_COMPONENTS],
    quality: f32,
}

#[derive(Clone, Debug, Default)]
struct NautilusFingerprintBank {
    observations: VecDeque<NautilusFingerprint>,
}

impl NautilusFingerprintBank {
    fn observe(&mut self, fingerprint: NautilusFingerprint) {
        // Near-identical consecutive observations add no appearance coverage;
        // replace the newest one so the bounded bank spans a longer interval.
        if self
            .observations
            .back()
            .is_some_and(|previous| nautilus_fingerprint_distance(previous, &fingerprint) <= 0.035)
        {
            self.observations.pop_back();
        }
        self.observations.push_back(fingerprint);
        while self.observations.len() > NAUTILUS_BANK_CAPACITY {
            self.observations.pop_front();
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NautilusCandidate {
    point: [f32; 2],
    normal: [f32; 2],
    fingerprint: NautilusFingerprint,
    evidence: f32,
}

#[derive(Clone, Copy, Debug)]
struct NautilusRefinedCandidate {
    candidate_index: usize,
    point: [f32; 2],
    normal: [f32; 2],
    fingerprint: NautilusFingerprint,
    score: f32,
}

#[derive(Clone, Debug)]
struct NautilusTreeNode {
    spatial_bounds: [f32; 4],
    descriptor_min: [i8; NAUTILUS_COMPONENTS],
    descriptor_max: [i8; NAUTILUS_COMPONENTS],
    split_dimension: usize,
    split_value: i8,
    left: Option<usize>,
    right: Option<usize>,
    candidates: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
struct NautilusFingerprintTree {
    candidates: Vec<NautilusCandidate>,
    nodes: Vec<NautilusTreeNode>,
    root: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct NautilusRelocation {
    track_index: usize,
    candidate_index: usize,
    current: [f32; 2],
    fingerprint: NautilusFingerprint,
    distance: f32,
    margin: f32,
    distance_ratio: f32,
    prediction_distance: f32,
    identity_radius: f32,
    reverse_margin: f32,
    normal_alignment: f32,
    anchor_conditioned: bool,
    confidence: f32,
    crossed_gap: bool,
}

#[derive(Clone, Debug, Default)]
struct LayerMotionSignature {
    /// Per-frame centroid of the member tracks' residual motion, oldest first.
    samples: VecDeque<[f32; 2]>,
    support: usize,
    age: u8,
}

#[derive(Clone, Copy, Debug)]
struct FocusBin {
    position: u16,
    sharpness_sum: f32,
    samples: u16,
}

#[derive(Clone, Copy, Debug)]
pub struct TrailPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug)]
pub struct OverlayTrail {
    pub id: u64,
    pub object: usize,
    pub match_score: f32,
    pub matched_streak: u8,
    pub layer_evidence: bool,
    pub normal_flow_evidence: bool,
    pub specularity: f32,
    pub assignment_confidence: f32,
    pub motion_ema: [f32; 2],
    pub motion_variance: f32,
    pub residual_history: Vec<[f32; 2]>,
    pub points: Vec<TrailPoint>,
}

#[derive(Clone, Copy, Debug)]
pub struct OverlayNode {
    pub object: usize,
    pub bounds: [f32; 6],
    pub depth: u8,
    pub count: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeEvidence {
    pub x: f32,
    pub y: f32,
    /// Unit image-gradient direction. Positive points from dark to bright.
    pub gradient_x: f32,
    pub gradient_y: f32,
    /// Scharr magnitude normalized by the adaptive high Canny threshold.
    pub strength: f32,
    /// Agreement of signed Scharr normals at the narrow and broad Gaussian
    /// scales. A persistent step approaches one; a fine lash/iris fibre or
    /// one side of a thin glasses ridge approaches zero.
    pub multiscale_consistency: f32,
    /// Persistence of the dark-to-bright step between the +/-2 px and +/-5 px
    /// normal lanes. Unlike gradient magnitude, this rejects a thin ridge
    /// whose two far samples return to the same material.
    pub signed_step_persistence: f32,
    /// Illumination-normalized tangential texture on the dark/bright sides of
    /// the signed edge. These are reliability weights, never color classes.
    pub dark_side_texture: f32,
    pub bright_side_texture: f32,
    /// Analog agreement with the robust native-RAW iris motion. `1.0` means
    /// the edge is unconditioned or moves with the iris; values near zero
    /// mean an upper-eye patch was substantially better explained by a
    /// different motion (for example an eyebrow/lid shadow during eye
    /// rotation). This only changes the edge's vote, never the RAW samples.
    pub iris_motion_consistency: f32,
}

impl Default for EdgeEvidence {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            gradient_x: 0.0,
            gradient_y: 0.0,
            strength: 0.0,
            multiscale_consistency: 1.0,
            signed_step_persistence: 1.0,
            dark_side_texture: 0.0,
            bright_side_texture: 0.0,
            iris_motion_consistency: 1.0,
        }
    }
}

/// One accepted region in the coarse-to-fine horizontal RAW walk tree.
/// Bounds and local translation are expressed on the previous/current ROI
/// lattices; `sensor_translation_px` removes any camera-side ROI relocation
/// and therefore describes material motion in absolute sensor coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct HorizontalLightFieldNode {
    pub parent: Option<u16>,
    pub depth: u8,
    pub leaf: bool,
    pub source_bounds: [f32; 4],
    pub source_center: [f32; 2],
    pub destination_center: [f32; 2],
    pub local_translation_px: f32,
    pub sensor_translation_px: f32,
    pub vertical_nuisance_px: f32,
    pub horizontal_scale_delta: f32,
    pub blur_radius_px: u8,
    pub correlation: f32,
    pub ambiguity_margin: f32,
    pub texture: f32,
    pub confidence: f32,
    pub samples: u16,
}

/// Regional horizontal deformation evidence gathered independently of Canny
/// feature identity.  It may center a broad point search, but it never labels
/// a feature or directly publishes eye anatomy.
#[derive(Clone, Debug, Default)]
pub struct HorizontalLightFieldStatus {
    pub nodes: Vec<HorizontalLightFieldNode>,
    pub horizontal_translation_px: f32,
    pub horizontal_scale_delta: f32,
    pub residual_px: f32,
    pub confidence: f32,
    pub horizontal_coverage: f32,
    pub leaf_nodes: usize,
    pub evaluations: usize,
    pub elapsed_micros: u64,
    pub reliable: bool,
}

/// Exact current-frame radial limbus observation. `point` and `normal` are
/// ROI-local native sensor coordinates; the five inward/five outward samples
/// remain implicit so rendering does not retain or duplicate RAW pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct RadialLimbusProbe {
    pub point: [f32; 2],
    pub normal: [f32; 2],
    pub phase_rad: f32,
    pub radial_shift_px: f32,
    pub profile_cost: f32,
    pub confidence: f32,
    pub fused: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MotionOctreeOverlay {
    pub generation: u64,
    pub trails: Vec<OverlayTrail>,
    pub nodes: Vec<OverlayNode>,
    pub motions: [SimilarityMotion; OBJECTS],
    pub layers: [MotionLayerStatus; OBJECTS],
    pub parallax_axis: [f32; 2],
    pub active_objects: usize,
    pub matched_features: usize,
    pub match_diagnostics: MatchDiagnostics,
    /// Current Canny-backed seed patches which have not yet survived a second
    /// frame. They are diagnostic proposals only and never vote in a motion
    /// layer or ellipse until promoted to a multi-point trail.
    pub provisional_features: Vec<(f32, f32)>,
    /// Exact native-resolution hysteresis result used by this frame's 2D
    /// temporal feature learner. One byte per sensor sample keeps the view
    /// lossless while avoiding a second, renderer-only Canny calculation.
    /// Empty for modes which did not run the temporal learning field.
    pub learning_canny_mask: Vec<u8>,
    pub learning_canny_profile: LearningCannyProfile,
    pub edges: Vec<EdgeEvidence>,
    pub edge_high_threshold: f32,
    /// Canny samples whose vote was reduced by the bounded upper-eye motion
    /// comparison. Kept as telemetry so the live overlay can make the mask
    /// auditable instead of silently deleting evidence.
    pub motion_shadow_edges_downweighted: usize,
    /// Coarse-to-fine native-RAW horizontal translation/scale field for the
    /// exact displayed exposure pair.
    pub horizontal_light_field: HorizontalLightFieldStatus,
    /// Illumination-normalized five-in/five-out radial transitions measured
    /// on this exact exposure pair. These are normal-flow constraints, not
    /// claims of tangential point identity along a smooth limbus.
    pub radial_limbus_probes: Vec<RadialLimbusProbe>,
    /// Current label-free/seeded outer-iris motion region in ROI-local pixels.
    /// It is exported so lossless reviews can audit exactly which Canny tracks
    /// were considered possible limbus normal-flow constraints.
    pub semantic_iris: Option<IrisEllipseSeed>,
    pub focus_sfm: FocusSfmStatus,
    pub coupled_motion: CoupledMotionStatus,
}

/// Weak texture cannot publish anatomy, but two independently coherent
/// full-resolution temporal layers can retain a fine ROI while semantic eye
/// evidence is temporarily occluded. Kept beside the layer implementation so
/// live and offline replay apply identical evidence thresholds.
pub fn coherent_temporal_feature_hold(overlay: &MotionOctreeOverlay) -> bool {
    if overlay
        .layers
        .iter()
        .map(|layer| layer.persistent_tracks)
        .sum::<usize>()
        < 16
    {
        return false;
    }
    overlay
        .motions
        .iter()
        .zip(overlay.layers.iter())
        .filter(|(motion, layer)| {
            motion.support >= 4
                && motion.residual.is_finite()
                && motion.residual <= 3.0
                && layer.persistent_tracks >= 4
                && layer.stable_frames >= 2
                && layer.signature_samples >= 2
                && layer.coherence.is_finite()
                && layer.coherence >= 0.28
                && layer.trajectory_error.is_finite()
                && layer.trajectory_error <= 3.0
        })
        .count()
        >= 2
}

#[derive(Clone, Debug)]
pub struct FeatureClusterIrisHypothesis {
    pub center: (f64, f64),
    pub radius: f64,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
    pub score: f64,
    pub motion_layer: usize,
    pub layer_coherence: f32,
    pub layer_separation: f32,
    pub layer_parallax: f32,
    pub layer_stable_frames: u16,
    pub seed_edge_score: f64,
    pub edge_score_gain: f64,
    pub seed_angular_coverage: usize,
    pub seed_opposing_meridians: usize,
    pub features: Vec<(f64, f64)>,
    pub object_support: [usize; OBJECTS],
    pub edge_support: usize,
    pub angular_coverage: usize,
    pub opposing_meridians: usize,
    pub iterations: usize,
    pub bridged_current_frame_edges: bool,
}

/// Stage-by-stage observability for the temporal-feature limbus solve.  This
/// remains small enough to emit for every lossless RAW replay frame, making a
/// sparse publish rate explainable without changing the production decision.
#[derive(Clone, Copy, Debug)]
pub struct FeatureClusterIrisDiagnostics {
    pub rejection: &'static str,
    pub semantic_split: bool,
    pub seed_available: bool,
    pub eligible_layers: usize,
    pub associated_edges: [usize; OBJECTS],
    pub fitted_layers: usize,
    pub best_edge_confidence: f64,
    pub best_seed_confidence: f64,
    pub best_edge_score_gain: f64,
    pub best_angular_coverage: usize,
    pub best_opposing_meridians: usize,
}

impl Default for FeatureClusterIrisDiagnostics {
    fn default() -> Self {
        Self {
            rejection: "uninitialized",
            semantic_split: false,
            seed_available: false,
            eligible_layers: 0,
            associated_edges: [0; OBJECTS],
            fitted_layers: 0,
            best_edge_confidence: 0.0,
            best_seed_confidence: 0.0,
            best_edge_score_gain: 0.0,
            best_angular_coverage: 0,
            best_opposing_meridians: 0,
        }
    }
}

/// A current-frame, native-resolution Canny conic fitted around an already
/// bounded anatomical seed.  This is proposal evidence only: unlike
/// [`FeatureClusterIrisHypothesis`], it has no temporal-layer identity and
/// must never be published without an independent eye-topology gate.
#[derive(Clone, Debug)]
pub struct CannyEllipseProposal {
    pub center: (f64, f64),
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
    pub confidence: f64,
    pub seed_confidence: f64,
    pub seed_angular_coverage: usize,
    pub seed_opposing_meridians: usize,
    pub edge_support: usize,
    pub angular_coverage: usize,
    pub opposing_meridians: usize,
    pub iterations: usize,
    pub features: Vec<(f64, f64)>,
}

#[derive(Clone, Copy, Debug)]
pub struct IrisEllipseSeed {
    pub center: (f64, f64),
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
}

impl IrisEllipseSeed {
    pub fn circle(center: (f64, f64), radius: f64) -> Self {
        Self {
            center,
            major_radius: radius,
            minor_radius: radius,
            angle: 0.0,
        }
    }

    fn area_seed(self) -> ((f64, f64), f64) {
        (self.center, (self.major_radius * self.minor_radius).sqrt())
    }

    fn ellipse(self) -> EdgeEllipse {
        canonical_ellipse(EdgeEllipse {
            center: self.center,
            major: self.major_radius,
            minor: self.minor_radius,
            angle: self.angle,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct EdgeEllipse {
    center: (f64, f64),
    major: f64,
    minor: f64,
    angle: f64,
}

#[derive(Clone, Debug)]
struct EllipseEvidence {
    objective: f64,
    confidence: f64,
    inliers: Vec<usize>,
    angular_coverage: usize,
    opposing_meridians: usize,
}

fn canonical_ellipse(mut ellipse: EdgeEllipse) -> EdgeEllipse {
    if ellipse.minor > ellipse.major {
        std::mem::swap(&mut ellipse.major, &mut ellipse.minor);
        ellipse.angle += std::f64::consts::FRAC_PI_2;
    }
    ellipse.angle = ellipse.angle.rem_euclid(std::f64::consts::PI);
    ellipse
}

fn ellipse_is_bounded(
    ellipse: EdgeEllipse,
    seed: ((f64, f64), f64),
    width: usize,
    height: usize,
) -> bool {
    let seed_radius = seed.1;
    let area_radius = (ellipse.major * ellipse.minor).sqrt();
    let center_offset = (ellipse.center.0 - seed.0 .0).hypot(ellipse.center.1 - seed.0 .1);
    ellipse.center.0 >= 2.0
        && ellipse.center.0 < width as f64 - 2.0
        && ellipse.center.1 >= 2.0
        && ellipse.center.1 < height as f64 - 2.0
        && ellipse.minor >= 12.0
        && ellipse.major <= width.min(height) as f64 * 0.49
        && ellipse.minor / ellipse.major >= 0.48
        && (seed_radius * 0.70..=seed_radius * 1.32).contains(&area_radius)
        && center_offset <= seed_radius * 0.24
}

fn score_edge_ellipse(
    ellipse: EdgeEllipse,
    edges: &[EdgeEvidence],
    seed: ((f64, f64), f64),
    collect_inliers: bool,
) -> EllipseEvidence {
    let cosine = ellipse.angle.cos();
    let sine = ellipse.angle.sin();
    let residual_limit = (ellipse.minor * 0.035).clamp(1.8, 3.4);
    let mut bins = [false; ELLIPSE_ANGLE_BINS];
    let mut inliers = Vec::new();
    let mut quality_sum = 0.0f64;
    let mut inlier_count = 0usize;
    for (index, edge) in edges.iter().enumerate() {
        let dx = edge.x as f64 - ellipse.center.0;
        let dy = edge.y as f64 - ellipse.center.1;
        let local_x = cosine * dx + sine * dy;
        let local_y = -sine * dx + cosine * dy;
        let normalized_x = local_x / ellipse.major;
        let normalized_y = local_y / ellipse.minor;
        let normalized_radius = normalized_x.hypot(normalized_y);
        if !normalized_radius.is_finite() || normalized_radius < 0.5 {
            continue;
        }
        let residual = (normalized_radius - 1.0).abs() * ellipse.minor;
        if residual > residual_limit {
            continue;
        }
        let normal_local_x = local_x / (ellipse.major * ellipse.major);
        let normal_local_y = local_y / (ellipse.minor * ellipse.minor);
        let normal_x = cosine * normal_local_x - sine * normal_local_y;
        let normal_y = sine * normal_local_x + cosine * normal_local_y;
        let normal_length = normal_x.hypot(normal_y).max(1.0e-9);
        let alignment =
            (normal_x * edge.gradient_x as f64 + normal_y * edge.gradient_y as f64) / normal_length;
        // The visible lateral/lower limbus normally runs from darker iris to
        // brighter sclera in the outward direction. Unsigned edges admit the
        // pupil, lashes, and both sides of every specular ridge.
        if alignment < 0.35 {
            continue;
        }
        let parameter_angle = normalized_y
            .atan2(normalized_x)
            .rem_euclid(std::f64::consts::TAU);
        let bin = ((parameter_angle / std::f64::consts::TAU * ELLIPSE_ANGLE_BINS as f64).floor()
            as usize)
            .min(ELLIPSE_ANGLE_BINS - 1);
        bins[bin] = true;
        let residual_quality = 1.0 - residual / residual_limit;
        let strength = (edge.strength as f64).clamp(0.35, 2.5).sqrt()
            * (edge.iris_motion_consistency as f64)
                .clamp(0.05, 1.0)
                .sqrt()
            * (0.25 + 0.75 * f64::from(edge.multiscale_consistency).clamp(0.0, 1.0)).sqrt()
            * (0.30 + 0.70 * f64::from(edge.signed_step_persistence).clamp(0.0, 1.0)).sqrt()
            * (0.82
                + 0.18
                    * (0.68 * f64::from(edge.dark_side_texture)
                        + 0.32 * f64::from(edge.bright_side_texture))
                    .clamp(0.0, 1.0));
        quality_sum += alignment * residual_quality * strength;
        inlier_count += 1;
        if collect_inliers {
            inliers.push(index);
        }
    }
    let angular_coverage = bins.iter().filter(|present| **present).count();
    let opposing_meridians = (0..ELLIPSE_ANGLE_BINS / 2)
        .filter(|bin| bins[*bin] && bins[*bin + ELLIPSE_ANGLE_BINS / 2])
        .count();
    let mean_quality = quality_sum / inlier_count.max(1) as f64;
    let coverage_score = angular_coverage as f64 / ELLIPSE_ANGLE_BINS as f64;
    let support_score = (inlier_count as f64 / 36.0).clamp(0.0, 1.0);
    let opposition_score = (opposing_meridians as f64 / 6.0).clamp(0.0, 1.0);
    let confidence = coverage_score
        * support_score
        * (0.55 + 0.45 * opposition_score)
        * (0.55 + 0.45 * mean_quality.clamp(0.0, 1.0));
    let center_offset = (ellipse.center.0 - seed.0 .0).hypot(ellipse.center.1 - seed.0 .1) / seed.1;
    let area_radius = (ellipse.major * ellipse.minor).sqrt();
    let scale_offset = (area_radius / seed.1).ln().abs();
    let objective = confidence - 0.10 * center_offset * center_offset - 0.08 * scale_offset;
    EllipseEvidence {
        objective,
        confidence,
        inliers,
        angular_coverage,
        opposing_meridians,
    }
}

fn fit_edge_ellipse(
    edges: &[EdgeEvidence],
    ellipse_seed: IrisEllipseSeed,
    width: usize,
    height: usize,
) -> Option<(EdgeEllipse, EllipseEvidence, EllipseEvidence, usize)> {
    let seed = ellipse_seed.area_seed();
    let seed_radius = seed.1.clamp(20.0, width.min(height) as f64 * 0.48);
    let radial_edge_indices = edges
        .iter()
        .enumerate()
        .filter(|edge| {
            let edge = edge.1;
            let dx = edge.x as f64 - seed.0 .0;
            let dy = edge.y as f64 - seed.0 .1;
            let distance = dx.hypot(dy);
            if !(seed_radius * 0.55..=seed_radius * 1.48).contains(&distance) {
                return false;
            }
            let radial_alignment =
                (dx * edge.gradient_x as f64 + dy * edge.gradient_y as f64) / distance.max(1.0e-9);
            radial_alignment >= 0.08
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let radial_edges = radial_edge_indices
        .iter()
        .map(|index| edges[*index])
        .collect::<Vec<_>>();
    if radial_edges.len() < 20 {
        return None;
    }

    let circular_seed = EdgeEllipse {
        center: seed.0,
        major: seed_radius,
        minor: seed_radius,
        angle: 0.0,
    };
    let shaped_seed = ellipse_seed.ellipse();
    let seed_evidence = score_edge_ellipse(shaped_seed, &radial_edges, seed, false);
    let mut best = if ellipse_is_bounded(shaped_seed, seed, width, height) {
        shaped_seed
    } else {
        circular_seed
    };
    let mut best_evidence = score_edge_ellipse(best, &radial_edges, seed, false);
    let mut evaluations = 1usize;
    for offset_y in [-0.07, 0.0, 0.07] {
        for offset_x in [-0.07, 0.0, 0.07] {
            for scale in [0.88, 1.0, 1.12] {
                for ratio in [0.58f64, 0.72, 0.86, 1.0] {
                    let orientations = if ratio > 0.99 { 1 } else { 6 };
                    for orientation in 0..orientations {
                        let root_ratio = ratio.sqrt();
                        let candidate = EdgeEllipse {
                            center: (
                                seed.0 .0 + offset_x * seed_radius,
                                seed.0 .1 + offset_y * seed_radius,
                            ),
                            major: seed_radius * scale / root_ratio,
                            minor: seed_radius * scale * root_ratio,
                            angle: orientation as f64 * std::f64::consts::PI / orientations as f64,
                        };
                        if !ellipse_is_bounded(candidate, seed, width, height) {
                            continue;
                        }
                        let evidence = score_edge_ellipse(candidate, &radial_edges, seed, false);
                        evaluations += 1;
                        if evidence.objective > best_evidence.objective {
                            best = candidate;
                            best_evidence = evidence;
                        }
                    }
                }
            }
        }
    }

    let mut steps = [
        seed_radius * 0.045,
        seed_radius * 0.045,
        seed_radius * 0.055,
        seed_radius * 0.055,
        std::f64::consts::PI / 18.0,
    ];
    for _ in 0..6 {
        for parameter in 0..steps.len() {
            for direction in [-1.0, 1.0] {
                let mut candidate = best;
                match parameter {
                    0 => candidate.center.0 += direction * steps[parameter],
                    1 => candidate.center.1 += direction * steps[parameter],
                    2 => candidate.major += direction * steps[parameter],
                    3 => candidate.minor += direction * steps[parameter],
                    _ => candidate.angle += direction * steps[parameter],
                }
                candidate = canonical_ellipse(candidate);
                if !ellipse_is_bounded(candidate, seed, width, height) {
                    continue;
                }
                let evidence = score_edge_ellipse(candidate, &radial_edges, seed, false);
                evaluations += 1;
                if evidence.objective > best_evidence.objective {
                    best = candidate;
                    best_evidence = evidence;
                }
            }
        }
        for step in &mut steps {
            *step *= 0.56;
        }
    }
    let mut evidence = score_edge_ellipse(best, &radial_edges, seed, true);
    if evidence.confidence < 0.18
        || evidence.inliers.len() < 20
        || evidence.angular_coverage < 10
        || evidence.opposing_meridians < 3
    {
        return None;
    }
    for index in &mut evidence.inliers {
        *index = radial_edge_indices[*index];
    }
    Some((best, evidence, seed_evidence, evaluations))
}

/// Refine a bounded anatomical ellipse with current-frame signed Canny
/// normals.  The search remains tied to `seed` by `ellipse_is_bounded`, and
/// the returned conic is deliberately not an eye-identity result.  Driving
/// uses it only as an alternate road entrance and still requires the complete
/// `sclera | limbus | pupil | limbus | sclera` topology before admission.
pub fn current_frame_canny_ellipse_proposal(
    overlay: &MotionOctreeOverlay,
    width: usize,
    height: usize,
    seed: IrisEllipseSeed,
) -> Option<CannyEllipseProposal> {
    let (ellipse, evidence, seed_evidence, iterations) =
        fit_edge_ellipse(&overlay.edges, seed, width, height)?;
    Some(CannyEllipseProposal {
        center: ellipse.center,
        major_radius: ellipse.major,
        minor_radius: ellipse.minor,
        angle: ellipse.angle,
        confidence: evidence.confidence,
        seed_confidence: seed_evidence.confidence,
        seed_angular_coverage: seed_evidence.angular_coverage,
        seed_opposing_meridians: seed_evidence.opposing_meridians,
        edge_support: evidence.inliers.len(),
        angular_coverage: evidence.angular_coverage,
        opposing_meridians: evidence.opposing_meridians,
        iterations,
        features: evidence
            .inliers
            .iter()
            .map(|index| {
                let edge = overlay.edges[*index];
                (edge.x as f64, edge.y as f64)
            })
            .collect(),
    })
}

/// Measure native-resolution signed-Canny support for an independently
/// measured limbus conic without allowing the Canny optimizer to deform that
/// conic.  The returned points are current-frame measurements; the geometry
/// remains exactly `seed`.
///
/// This is the correct contract when a multi-bank sclera/limbus road (or an
/// equally strong physical source) already supplied the geometry.  Canny can
/// establish direct edge support and temporal identity, but a lid, lash, or
/// pupil ridge cannot use a higher raw edge count to replace the measured
/// outer boundary.
pub fn measured_seed_canny_support_proposal(
    overlay: &MotionOctreeOverlay,
    width: usize,
    height: usize,
    seed: IrisEllipseSeed,
) -> Option<CannyEllipseProposal> {
    let ellipse = seed.ellipse();
    if !ellipse_is_bounded(ellipse, seed.area_seed(), width, height) {
        return None;
    }
    let evidence = score_edge_ellipse(ellipse, &overlay.edges, seed.area_seed(), true);
    if evidence.confidence < 0.18
        || evidence.inliers.len() < 20
        || evidence.angular_coverage < 10
        || evidence.opposing_meridians < 3
    {
        return None;
    }
    Some(CannyEllipseProposal {
        center: ellipse.center,
        major_radius: ellipse.major,
        minor_radius: ellipse.minor,
        angle: ellipse.angle,
        confidence: evidence.confidence,
        seed_confidence: evidence.confidence,
        seed_angular_coverage: evidence.angular_coverage,
        seed_opposing_meridians: evidence.opposing_meridians,
        edge_support: evidence.inliers.len(),
        angular_coverage: evidence.angular_coverage,
        opposing_meridians: evidence.opposing_meridians,
        iterations: 1,
        features: evidence
            .inliers
            .iter()
            .map(|index| {
                let edge = overlay.edges[*index];
                (edge.x as f64, edge.y as f64)
            })
            .collect(),
    })
}

fn normalized_ellipse_radius(ellipse: EdgeEllipse, point: (f64, f64)) -> f64 {
    let cosine = ellipse.angle.cos();
    let sine = ellipse.angle.sin();
    let dx = point.0 - ellipse.center.0;
    let dy = point.1 - ellipse.center.1;
    let local_x = cosine * dx + sine * dy;
    let local_y = -sine * dx + cosine * dy;
    (local_x / ellipse.major).hypot(local_y / ellipse.minor)
}

fn edges_for_motion_layer(
    overlay: &MotionOctreeOverlay,
    object: usize,
    seed: IrisEllipseSeed,
) -> Vec<EdgeEvidence> {
    let seed_ellipse = seed.ellipse();
    let anchors = overlay
        .trails
        .iter()
        .filter(|trail| {
            trail.object == object
                && trail.layer_evidence
                && trail.residual_history.len() >= MIN_MOTION_SIGNATURE
        })
        .filter_map(|trail| trail.points.last())
        .filter(|point| {
            let radius = normalized_ellipse_radius(seed_ellipse, (point.x as f64, point.y as f64));
            (0.62..=1.30).contains(&radius)
        })
        .map(|point| [point.x, point.y])
        .collect::<Vec<_>>();
    if anchors.len() < MIN_LAYER_PERSISTENT_TRACKS {
        return Vec::new();
    }
    overlay
        .edges
        .iter()
        .copied()
        .filter(|edge| {
            let radius = normalized_ellipse_radius(seed_ellipse, (edge.x as f64, edge.y as f64));
            (0.55..=1.45).contains(&radius)
                && anchors.iter().any(|anchor| {
                    (edge.x - anchor[0]).hypot(edge.y - anchor[1]) <= LAYER_EDGE_ASSOCIATION_RADIUS
                })
        })
        .collect()
}

/// Infer an outer-iris ellipse only after Canny-adjacent features have formed
/// a persistent, separated temporal motion layer. Current-frame edges are
/// admitted only near tracks belonging to that layer; untracked brow, eyelid,
/// and nose contours cannot vote merely because they are close to the seed.
pub fn feature_cluster_iris_hypothesis(
    overlay: &MotionOctreeOverlay,
    width: usize,
    height: usize,
    seed: Option<IrisEllipseSeed>,
    minimum_edge_score_gain: f64,
) -> Option<FeatureClusterIrisHypothesis> {
    feature_cluster_iris_hypothesis_with_diagnostics(
        overlay,
        width,
        height,
        seed,
        minimum_edge_score_gain,
    )
    .0
}

pub fn feature_cluster_iris_hypothesis_with_diagnostics(
    overlay: &MotionOctreeOverlay,
    width: usize,
    height: usize,
    seed: Option<IrisEllipseSeed>,
    minimum_edge_score_gain: f64,
) -> (
    Option<FeatureClusterIrisHypothesis>,
    FeatureClusterIrisDiagnostics,
) {
    feature_cluster_iris_hypothesis_internal(
        overlay,
        width,
        height,
        seed,
        minimum_edge_score_gain,
        false,
    )
}

/// Establish temporal motion-layer identity and current-frame Canny support
/// around geometry supplied by an independently measured limbus road.  The
/// temporal/Canny evidence may accept or reject the seed, but may not deform
/// it.  This keeps geometric measurement separate from semantic identity.
pub fn feature_cluster_iris_hypothesis_from_measured_seed(
    overlay: &MotionOctreeOverlay,
    width: usize,
    height: usize,
    seed: IrisEllipseSeed,
) -> Option<FeatureClusterIrisHypothesis> {
    feature_cluster_iris_hypothesis_internal(overlay, width, height, Some(seed), 0.0, true).0
}

pub fn feature_cluster_iris_hypothesis_from_measured_seed_with_diagnostics(
    overlay: &MotionOctreeOverlay,
    width: usize,
    height: usize,
    seed: IrisEllipseSeed,
) -> (
    Option<FeatureClusterIrisHypothesis>,
    FeatureClusterIrisDiagnostics,
) {
    feature_cluster_iris_hypothesis_internal(overlay, width, height, Some(seed), 0.0, true)
}

fn feature_cluster_iris_hypothesis_internal(
    overlay: &MotionOctreeOverlay,
    width: usize,
    height: usize,
    seed: Option<IrisEllipseSeed>,
    minimum_edge_score_gain: f64,
    measured_seed_geometry: bool,
) -> (
    Option<FeatureClusterIrisHypothesis>,
    FeatureClusterIrisDiagnostics,
) {
    let mut diagnostics = FeatureClusterIrisDiagnostics::default();
    if width < 32 || height < 24 {
        diagnostics.rejection = "frame-too-small";
        return (None, diagnostics);
    }
    let mut object_points: [Vec<(f64, f64)>; OBJECTS] = std::array::from_fn(|_| Vec::new());
    for trail in &overlay.trails {
        if trail.object >= OBJECTS || trail.points.len() < 3 {
            continue;
        }
        let Some(point) = trail.points.last() else {
            continue;
        };
        if !(2.0..width as f32 - 2.0).contains(&point.x)
            || !(2.0..height as f32 - 2.0).contains(&point.y)
        {
            continue;
        }
        object_points[trail.object].push((point.x as f64, point.y as f64));
    }
    let semantic_split = overlay.layers[GENERAL_LAYER].persistent_tracks >= MIN_LAYER_SUPPORT
        && overlay.layers[PUPIL_LAYER].persistent_tracks >= MIN_LAYER_SUPPORT
        && overlay.layers[REFLECTION_LAYER].persistent_tracks >= MIN_REFLECTION_SUPPORT;
    diagnostics.semantic_split = semantic_split;
    let track_points = if semantic_split {
        object_points[PUPIL_LAYER].clone()
    } else {
        object_points.iter().flatten().copied().collect::<Vec<_>>()
    };
    let seed = seed.or_else(|| {
        if track_points.len() < 10 {
            return None;
        }
        let mut xs = track_points.iter().map(|point| point.0).collect::<Vec<_>>();
        let mut ys = track_points.iter().map(|point| point.1).collect::<Vec<_>>();
        xs.sort_by(f64::total_cmp);
        ys.sort_by(f64::total_cmp);
        Some(IrisEllipseSeed::circle(
            (xs[xs.len() / 2], ys[ys.len() / 2]),
            width.min(height) as f64 * 0.30,
        ))
    });
    let Some(seed) = seed else {
        diagnostics.rejection = "no-seed";
        return (None, diagnostics);
    };
    diagnostics.seed_available = true;
    let mut layer_candidates = Vec::new();
    let mut bridged_full_frame_fit = None;
    for object in 0..OBJECTS {
        if semantic_split && object != PUPIL_LAYER {
            continue;
        }
        let layer = overlay.layers[object];
        let motion = overlay.motions[object];
        let ordinary_separated_layer = layer.stable_frames >= MIN_LAYER_STABLE_FRAMES
            && layer.persistent_tracks >= MIN_LAYER_PERSISTENT_TRACKS
            && layer.coherence >= 0.20
            && layer.separation >= MIN_LAYER_SEPARATION
            && motion.support >= MIN_LAYER_SUPPORT
            && motion.residual <= MAX_LAYER_RESIDUAL;
        // A same-frame semantic/pupil-headed RAW solve has already supplied
        // the measured conic in this branch. During rigid head motion the iris
        // can legitimately share the general layer's translation, leaving no
        // differential separation even while dozens of native patches remain
        // coherent on the eye. Requiring parallax separation here repeatedly
        // reset a correct 2D lock after one frame. Mature, low-residual support
        // may establish temporal *identity* for measured geometry without
        // claiming a separately moving object; freely fitted conics retain the
        // original separated-layer gate above.
        let measured_rigid_identity_layer = measured_seed_geometry
            && layer.persistent_tracks >= 8
            && layer.signature_samples >= MIN_MOTION_SIGNATURE
            && layer.coherence >= 0.24
            && motion.support >= 8
            && motion.residual <= 3.20;
        if !ordinary_separated_layer && !measured_rigid_identity_layer {
            continue;
        }
        diagnostics.eligible_layers += 1;
        let edges = edges_for_motion_layer(overlay, object, seed);
        diagnostics.associated_edges[object] = edges.len();
        // When another exact-frame RAW algorithm has already measured the
        // conic, the temporal layer owns *identity*, not geometry.  Requiring
        // the sparse track-adjacent edges to refit that conic first made the
        // two contracts circular: a perfectly supported measured limbus was
        // discarded whenever the promoted tracks occupied only a few iris
        // sectors.  Keep a small signed-Canny coincidence requirement around
        // those tracks, then corroborate the unchanged measured conic against
        // the complete current-frame Canny field.  No untracked edge is
        // allowed to move the seed.
        let measured_fit = measured_seed_geometry.then(|| {
            let ellipse = seed.ellipse();
            let temporal_evidence = score_edge_ellipse(ellipse, &edges, seed.area_seed(), true);
            let full_evidence = score_edge_ellipse(ellipse, &overlay.edges, seed.area_seed(), true);
            diagnostics.best_edge_confidence = diagnostics
                .best_edge_confidence
                .max(full_evidence.confidence);
            diagnostics.best_seed_confidence = diagnostics
                .best_seed_confidence
                .max(full_evidence.confidence);
            diagnostics.best_angular_coverage = diagnostics
                .best_angular_coverage
                .max(full_evidence.angular_coverage);
            diagnostics.best_opposing_meridians = diagnostics
                .best_opposing_meridians
                .max(full_evidence.opposing_meridians);
            (temporal_evidence.inliers.len() >= MIN_LAYER_PERSISTENT_TRACKS
                && temporal_evidence.angular_coverage >= 2
                // This conic was already measured by an independent
                // pupil-headed, semantic-eye-qualified native RAW road. The
                // temporal layer is establishing identity, not asking sparse
                // Canny sectors to rediscover geometry. Two opposed
                // meridians and nine sectors are sufficient corroboration;
                // the stricter 3/10/0.18 solve remains unchanged for every
                // freely fitted edge ellipse.
                && full_evidence.confidence >= 0.15
                && full_evidence.inliers.len() >= 18
                && full_evidence.angular_coverage >= 9
                && full_evidence.opposing_meridians >= 2)
                .then_some((ellipse, full_evidence.clone(), full_evidence, 1usize))
        });
        let direct_fit = if measured_seed_geometry {
            measured_fit.flatten()
        } else {
            fit_edge_ellipse(&edges, seed, width, height)
        };
        let (fit_edges, ellipse, evidence, seed_evidence, evaluations, bridged) =
            if let Some((ellipse, evidence, seed_evidence, evaluations)) = direct_fit {
                (
                    if measured_seed_geometry {
                        overlay.edges.clone()
                    } else {
                        edges.clone()
                    },
                    ellipse,
                    evidence,
                    seed_evidence,
                    evaluations,
                    measured_seed_geometry,
                )
            } else {
                if measured_seed_geometry {
                    continue;
                }
                // Persistent layer anchors often occupy only the textured
                // quadrants of the iris, so requiring every Canny sample to
                // sit beside an anchor leaves an otherwise obvious conic with
                // too little angular coverage.  Let those promoted anchors
                // identify the anatomical layer, then bridge between them
                // using the current frame's signed, native-resolution Canny
                // normals.  The completed conic still has to be seed-bounded
                // and independently corroborated by layer-associated inliers.
                if bridged_full_frame_fit.is_none() {
                    bridged_full_frame_fit = fit_edge_ellipse(&overlay.edges, seed, width, height);
                }
                let Some((ellipse, evidence, seed_evidence, evaluations)) =
                    bridged_full_frame_fit.clone()
                else {
                    continue;
                };
                let temporal_evidence = score_edge_ellipse(ellipse, &edges, seed.area_seed(), true);
                if temporal_evidence.inliers.len() < 7
                    || temporal_evidence.angular_coverage < 4
                    || (temporal_evidence.opposing_meridians == 0
                        && temporal_evidence.angular_coverage < 7)
                {
                    continue;
                }
                (
                    overlay.edges.clone(),
                    ellipse,
                    evidence,
                    seed_evidence,
                    evaluations,
                    true,
                )
            };
        diagnostics.fitted_layers += 1;
        let layer_quality = layer.coherence as f64
            * (layer.separation as f64 / 1.5).clamp(0.0, 1.0)
            * (layer.persistent_tracks as f64 / 8.0).clamp(0.0, 1.0);
        let selection_score = evidence.confidence * (0.72 + 0.28 * layer_quality);
        layer_candidates.push((
            selection_score,
            object,
            fit_edges,
            ellipse,
            evidence,
            seed_evidence,
            evaluations,
            bridged,
        ));
    }
    let Some((
        _,
        motion_layer,
        layer_edges,
        ellipse,
        evidence,
        seed_evidence,
        evaluations,
        bridged_current_frame_edges,
    )) = layer_candidates
        .into_iter()
        .max_by(|left, right| left.0.total_cmp(&right.0))
    else {
        diagnostics.rejection = if diagnostics.eligible_layers == 0 {
            "no-eligible-temporal-layer"
        } else if diagnostics.associated_edges.iter().all(|count| *count == 0) {
            "no-layer-associated-edges"
        } else {
            "no-supported-conic-fit"
        };
        return (None, diagnostics);
    };
    let (ellipse, evidence, seed_evidence) = if measured_seed_geometry {
        let measured_ellipse = seed.ellipse();
        let measured_evidence =
            score_edge_ellipse(measured_ellipse, &layer_edges, seed.area_seed(), true);
        if measured_evidence.confidence < 0.15
            || measured_evidence.inliers.len() < 18
            || measured_evidence.angular_coverage < 9
            || measured_evidence.opposing_meridians < 2
        {
            diagnostics.rejection = "measured-seed-lacks-direct-canny-support";
            return (None, diagnostics);
        }
        (
            measured_ellipse,
            measured_evidence.clone(),
            measured_evidence,
        )
    } else {
        (ellipse, evidence, seed_evidence)
    };
    let edge_score_gain = evidence.confidence - seed_evidence.confidence;
    diagnostics.best_edge_confidence = evidence.confidence;
    diagnostics.best_seed_confidence = seed_evidence.confidence;
    diagnostics.best_edge_score_gain = edge_score_gain;
    diagnostics.best_angular_coverage = evidence.angular_coverage;
    diagnostics.best_opposing_meridians = evidence.opposing_meridians;
    if minimum_edge_score_gain > 0.0
        && seed_evidence.confidence >= 0.18
        && (edge_score_gain < minimum_edge_score_gain
            || evidence.angular_coverage < seed_evidence.angular_coverage
            || evidence.opposing_meridians < seed_evidence.opposing_meridians)
    {
        diagnostics.rejection = "insufficient-improvement-over-seed";
        return (None, diagnostics);
    }

    let mut object_support = [0usize; OBJECTS];
    for object in 0..OBJECTS {
        let motion = overlay.motions[object];
        if motion.support < 3 || motion.residual > 6.0 {
            continue;
        }
        object_support[object] = object_points[object]
            .iter()
            .filter(|point| {
                let radius = normalized_ellipse_radius(ellipse, **point);
                (0.24..=1.12).contains(&radius)
            })
            .count();
    }
    let layer = overlay.layers[motion_layer];
    let layer_quality = layer.coherence as f64
        * (layer.separation as f64 / 1.5).clamp(0.0, 1.0)
        * (layer.persistent_tracks as f64 / 8.0).clamp(0.0, 1.0);
    let score = evidence.confidence * (0.72 + 0.28 * layer_quality);
    diagnostics.rejection = "accepted";
    (
        Some(FeatureClusterIrisHypothesis {
            center: ellipse.center,
            radius: (ellipse.major * ellipse.minor).sqrt(),
            major_radius: ellipse.major,
            minor_radius: ellipse.minor,
            angle: ellipse.angle,
            score,
            motion_layer,
            layer_coherence: layer.coherence,
            layer_separation: layer.separation,
            layer_parallax: layer.parallax,
            layer_stable_frames: layer.stable_frames,
            seed_edge_score: seed_evidence.confidence,
            edge_score_gain,
            seed_angular_coverage: seed_evidence.angular_coverage,
            seed_opposing_meridians: seed_evidence.opposing_meridians,
            features: evidence
                .inliers
                .iter()
                .map(|index| {
                    let edge = layer_edges[*index];
                    (edge.x as f64, edge.y as f64)
                })
                .collect(),
            object_support,
            edge_support: evidence.inliers.len(),
            angular_coverage: evidence.angular_coverage,
            opposing_meridians: evidence.opposing_meridians,
            iterations: evaluations,
            bridged_current_frame_edges,
        }),
        diagnostics,
    )
}

#[derive(Clone)]
struct RawFrame {
    sensor_x: u32,
    sensor_y: u32,
    width: usize,
    height: usize,
    pixels: Vec<u16>,
}

/// Native-resolution row integrals used only to evaluate sparse horizontal
/// box-filtered walks.  No image is resized, demosaiced, or warped: every
/// queried value is the linear mean of an interval on one original RAW row.
struct NativeHorizontalRowIntegrals {
    width: usize,
    height: usize,
    rows: Vec<u64>,
}

impl NativeHorizontalRowIntegrals {
    fn new(frame: &RawFrame) -> Self {
        let stride = frame.width + 1;
        let mut rows = vec![0u64; stride.saturating_mul(frame.height)];
        for y in 0..frame.height {
            let mut sum = 0u64;
            let output = y * stride;
            let input = y * frame.width;
            for x in 0..frame.width {
                sum = sum.saturating_add(u64::from(frame.pixels[input + x]));
                rows[output + x + 1] = sum;
            }
        }
        Self {
            width: frame.width,
            height: frame.height,
            rows,
        }
    }

    fn integer_box_mean(&self, x: i32, y: i32, radius: i32) -> Option<f32> {
        let left = x - radius;
        let right = x + radius;
        if y < 0 || y >= self.height as i32 || left < 0 || right >= self.width as i32 || radius < 0
        {
            return None;
        }
        let stride = self.width + 1;
        let row = y as usize * stride;
        let sum = self.rows[row + right as usize + 1] - self.rows[row + left as usize];
        Some(sum as f32 / (2 * radius + 1) as f32)
    }

    /// Linearly interpolate between neighboring native box centers.  This
    /// gives the affine scale search sub-pixel continuity without creating an
    /// interpolated image plane.
    fn sample_horizontal_blur(&self, x: f32, y: f32, radius: i32) -> Option<f32> {
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        let x0 = x.floor() as i32;
        let x_phase = x - x0 as f32;
        let y0 = y.round() as i32;
        let vertical = (radius / 3).clamp(1, 4);
        let mut weighted = 0.0f32;
        let mut weight_sum = 0.0f32;
        for (offset_y, weight) in [(-vertical, 1.0f32), (0, 2.0), (vertical, 1.0)] {
            let left = self.integer_box_mean(x0, y0 + offset_y, radius)?;
            let right = self.integer_box_mean(x0 + 1, y0 + offset_y, radius)?;
            weighted += (left * (1.0 - x_phase) + right * x_phase) * weight;
            weight_sum += weight;
        }
        Some(weighted / weight_sum)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct HorizontalWalkTransform {
    translation: f32,
    scale_delta: f32,
    vertical_nuisance: f32,
}

#[derive(Clone, Copy, Debug)]
struct HorizontalWalkSample {
    x: f32,
    y: f32,
    value: f32,
}

#[derive(Clone, Debug)]
struct HorizontalWalkSource {
    samples: Vec<HorizontalWalkSample>,
    sum: f64,
    squared_sum: f64,
    texture: f32,
}

#[derive(Clone, Copy, Debug)]
struct HorizontalWalkCandidate {
    transform: HorizontalWalkTransform,
    cost: f32,
    correlation: f32,
}

fn inclusive_search_values(center: f32, radius: f32, step: f32) -> Vec<f32> {
    let half_steps = (radius / step).round().max(0.0) as i32;
    (-half_steps..=half_steps)
        .map(|offset| center + offset as f32 * step)
        .collect()
}

fn horizontal_walk_source(
    source: &NativeHorizontalRowIntegrals,
    bounds: [f32; 4],
    depth: u8,
) -> Option<HorizontalWalkSource> {
    let level = usize::from(depth.min(LIGHT_FIELD_MAX_DEPTH));
    let radius = LIGHT_FIELD_BLUR_RADIUS[level];
    let step = LIGHT_FIELD_SAMPLE_STEP[level];
    let lane_count = if depth == 0 { 5usize } else { 3usize };
    let left = bounds[0].ceil() as usize;
    let right = bounds[2].floor().max(0.0) as usize;
    if right <= left || bounds[3] <= bounds[1] {
        return None;
    }
    let mut samples = Vec::new();
    for lane in 0..lane_count {
        let phase = (lane + 1) as f32 / (lane_count + 1) as f32;
        let y = bounds[1] * (1.0 - phase) + bounds[3] * phase;
        for x in (left..=right).step_by(step) {
            let x = x as f32;
            if let Some(value) = source.sample_horizontal_blur(x, y, radius) {
                samples.push(HorizontalWalkSample { x, y, value });
            }
        }
    }
    let minimum_samples = if depth >= 3 { 12 } else { 18 };
    if samples.len() < minimum_samples {
        return None;
    }
    let sum = samples
        .iter()
        .map(|sample| f64::from(sample.value))
        .sum::<f64>();
    let squared_sum = samples
        .iter()
        .map(|sample| f64::from(sample.value).powi(2))
        .sum::<f64>();
    let count = samples.len() as f64;
    let energy = (squared_sum - sum * sum / count).max(0.0);
    let mean = (sum / count).abs().max(16.0);
    let texture = (energy / count).sqrt() as f32 / mean as f32;
    Some(HorizontalWalkSource {
        samples,
        sum,
        squared_sum,
        texture,
    })
}

fn score_horizontal_walk(
    source: &HorizontalWalkSource,
    current: &NativeHorizontalRowIntegrals,
    center_x: f32,
    radius: i32,
    transform: HorizontalWalkTransform,
) -> Option<HorizontalWalkCandidate> {
    let count = source.samples.len() as f64;
    let mut current_sum = 0.0f64;
    let mut current_squared_sum = 0.0f64;
    let mut cross = 0.0f64;
    for sample in &source.samples {
        let current_x =
            sample.x + transform.translation + transform.scale_delta * (sample.x - center_x);
        let current_y = sample.y + transform.vertical_nuisance;
        let current_value = current.sample_horizontal_blur(current_x, current_y, radius)?;
        let current_value = f64::from(current_value);
        current_sum += current_value;
        current_squared_sum += current_value * current_value;
        cross += f64::from(sample.value) * current_value;
    }
    let source_energy = source.squared_sum - source.sum * source.sum / count;
    let current_energy = current_squared_sum - current_sum * current_sum / count;
    if source_energy <= 1.0 || current_energy <= 1.0 {
        return None;
    }
    let covariance = cross - source.sum * current_sum / count;
    let correlation =
        (covariance / (source_energy * current_energy).sqrt()).clamp(-1.0, 1.0) as f32;
    Some(HorizontalWalkCandidate {
        transform,
        cost: 1.0 - correlation,
        correlation,
    })
}

#[allow(clippy::too_many_arguments)]
fn append_horizontal_light_field_node(
    previous: &RawFrame,
    current: &RawFrame,
    previous_rows: &NativeHorizontalRowIntegrals,
    current_rows: &NativeHorizontalRowIntegrals,
    bounds: [f32; 4],
    depth: u8,
    parent: Option<usize>,
    prior: HorizontalWalkTransform,
    nodes: &mut Vec<HorizontalLightFieldNode>,
    evaluations: &mut usize,
) -> Option<usize> {
    if nodes.len() >= LIGHT_FIELD_MAX_NODES
        || bounds[2] - bounds[0] < LIGHT_FIELD_MIN_WIDTH
        || bounds[3] - bounds[1] < LIGHT_FIELD_MIN_HEIGHT
    {
        return None;
    }
    let level = usize::from(depth.min(LIGHT_FIELD_MAX_DEPTH));
    let source = horizontal_walk_source(previous_rows, bounds, depth)?;
    if source.texture < LIGHT_FIELD_MIN_TEXTURE {
        return None;
    }
    let center = [0.5 * (bounds[0] + bounds[2]), 0.5 * (bounds[1] + bounds[3])];
    let translations = inclusive_search_values(
        prior.translation,
        LIGHT_FIELD_TRANSLATION_RADIUS[level],
        LIGHT_FIELD_TRANSLATION_STEP[level],
    );
    let scales = inclusive_search_values(
        prior.scale_delta,
        LIGHT_FIELD_SCALE_RADIUS[level],
        LIGHT_FIELD_SCALE_STEP[level],
    );
    let vertical = inclusive_search_values(
        prior.vertical_nuisance,
        LIGHT_FIELD_VERTICAL_RADIUS[level],
        LIGHT_FIELD_VERTICAL_STEP[level],
    );
    let mut candidates = Vec::with_capacity(
        translations
            .len()
            .saturating_mul(scales.len())
            .saturating_mul(vertical.len()),
    );
    for translation in translations {
        for scale_delta in scales.iter().copied() {
            if !scale_delta.is_finite() || scale_delta.abs() > 0.12 {
                continue;
            }
            for vertical_nuisance in vertical.iter().copied() {
                *evaluations = evaluations.saturating_add(1);
                if let Some(candidate) = score_horizontal_walk(
                    &source,
                    current_rows,
                    center[0],
                    LIGHT_FIELD_BLUR_RADIUS[level],
                    HorizontalWalkTransform {
                        translation,
                        scale_delta,
                        vertical_nuisance,
                    },
                ) {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates.sort_by(|left, right| left.cost.total_cmp(&right.cost));
    let best = candidates.first().copied()?;
    let span = (bounds[2] - bounds[0]).max(1.0);
    let second_cost = candidates
        .iter()
        .find(|candidate| {
            (candidate.transform.translation - best.transform.translation).abs()
                >= LIGHT_FIELD_TRANSLATION_STEP[level] * 1.5
                || (candidate.transform.scale_delta - best.transform.scale_delta).abs() * span
                    >= 3.0
                || (candidate.transform.vertical_nuisance - best.transform.vertical_nuisance).abs()
                    >= LIGHT_FIELD_VERTICAL_STEP[level] * 1.5
        })
        .map_or(1.0, |candidate| candidate.cost);
    let ambiguity_margin = (second_cost - best.cost).max(0.0);
    let correlation_quality = ((best.correlation - 0.45) / 0.48).clamp(0.0, 1.0);
    let texture_quality = ((source.texture - 0.008) / 0.075).clamp(0.0, 1.0);
    let margin_quality = (ambiguity_margin / 0.018).clamp(0.0, 1.0);
    let confidence = correlation_quality * texture_quality * (0.35 + 0.65 * margin_quality);
    let minimum_correlation = if depth == 0 {
        LIGHT_FIELD_MIN_CORRELATION
    } else {
        LIGHT_FIELD_MIN_CORRELATION - 0.04
    };
    if best.correlation < minimum_correlation || confidence < 0.06 {
        return None;
    }
    let index = nodes.len();
    let sensor_translation =
        best.transform.translation + current.sensor_x as f32 - previous.sensor_x as f32;
    nodes.push(HorizontalLightFieldNode {
        parent: parent.and_then(|value| u16::try_from(value).ok()),
        depth,
        leaf: true,
        source_bounds: bounds,
        source_center: center,
        destination_center: [
            center[0] + best.transform.translation,
            center[1] + best.transform.vertical_nuisance,
        ],
        local_translation_px: best.transform.translation,
        sensor_translation_px: sensor_translation,
        vertical_nuisance_px: best.transform.vertical_nuisance,
        horizontal_scale_delta: best.transform.scale_delta,
        blur_radius_px: LIGHT_FIELD_BLUR_RADIUS[level] as u8,
        correlation: best.correlation,
        ambiguity_margin,
        texture: source.texture,
        confidence,
        samples: source.samples.len().min(usize::from(u16::MAX)) as u16,
    });

    let child_width = 0.5 * (bounds[2] - bounds[0]);
    let child_height = 0.5 * (bounds[3] - bounds[1]);
    if depth >= LIGHT_FIELD_MAX_DEPTH
        || confidence < LIGHT_FIELD_BRANCH_CONFIDENCE
        || best.correlation < 0.64
        || child_width < LIGHT_FIELD_MIN_WIDTH
        || child_height < LIGHT_FIELD_MIN_HEIGHT
    {
        return Some(index);
    }
    let midpoint = [center[0], center[1]];
    let mut children = 0usize;
    for child in 0..4usize {
        if nodes.len() >= LIGHT_FIELD_MAX_NODES {
            break;
        }
        let child_bounds = [
            if child & 1 == 0 {
                bounds[0]
            } else {
                midpoint[0]
            },
            if child & 2 == 0 {
                bounds[1]
            } else {
                midpoint[1]
            },
            if child & 1 == 0 {
                midpoint[0]
            } else {
                bounds[2]
            },
            if child & 2 == 0 {
                midpoint[1]
            } else {
                bounds[3]
            },
        ];
        let child_center_x = 0.5 * (child_bounds[0] + child_bounds[2]);
        let child_prior = HorizontalWalkTransform {
            translation: best.transform.translation
                + best.transform.scale_delta * (child_center_x - center[0]),
            scale_delta: best.transform.scale_delta,
            vertical_nuisance: best.transform.vertical_nuisance,
        };
        children += usize::from(
            append_horizontal_light_field_node(
                previous,
                current,
                previous_rows,
                current_rows,
                child_bounds,
                depth + 1,
                Some(index),
                child_prior,
                nodes,
                evaluations,
            )
            .is_some(),
        );
    }
    if children > 0 {
        nodes[index].leaf = false;
    }
    Some(index)
}

fn summarize_horizontal_light_field(
    previous: &RawFrame,
    nodes: Vec<HorizontalLightFieldNode>,
    evaluations: usize,
    elapsed_micros: u64,
) -> HorizontalLightFieldStatus {
    let leaves = nodes
        .iter()
        .filter(|node| node.leaf && node.confidence >= 0.08)
        .collect::<Vec<_>>();
    if leaves.is_empty() {
        return HorizontalLightFieldStatus {
            nodes,
            evaluations,
            elapsed_micros,
            ..HorizontalLightFieldStatus::default()
        };
    }
    let mut weight_sum = 0.0f64;
    let mut mean_x = 0.0f64;
    let mut mean_displacement = 0.0f64;
    let mut weighted_node_scale = 0.0f64;
    let mut minimum_x = f32::INFINITY;
    let mut maximum_x = f32::NEG_INFINITY;
    for node in &leaves {
        let weight = f64::from(node.confidence.max(0.03))
            * f64::from(node.confidence.max(0.03))
            * f64::from(node.samples.max(1));
        let x = previous.sensor_x as f64 + f64::from(node.source_center[0]);
        weight_sum += weight;
        mean_x += weight * x;
        mean_displacement += weight * f64::from(node.sensor_translation_px);
        weighted_node_scale += weight * f64::from(node.horizontal_scale_delta);
        minimum_x = minimum_x.min(node.source_center[0]);
        maximum_x = maximum_x.max(node.source_center[0]);
    }
    mean_x /= weight_sum.max(1.0e-9);
    mean_displacement /= weight_sum.max(1.0e-9);
    weighted_node_scale /= weight_sum.max(1.0e-9);
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for node in &leaves {
        let weight = f64::from(node.confidence.max(0.03)).powi(2) * f64::from(node.samples.max(1));
        let x = previous.sensor_x as f64 + f64::from(node.source_center[0]) - mean_x;
        let displacement = f64::from(node.sensor_translation_px) - mean_displacement;
        numerator += weight * x * displacement;
        denominator += weight * x * x;
    }
    let regression_scale = if denominator > 1.0 {
        numerator / denominator
    } else {
        weighted_node_scale
    };
    let scale_delta =
        (0.72 * regression_scale + 0.28 * weighted_node_scale).clamp(-0.12, 0.12) as f32;
    let center_sensor_x = previous.sensor_x as f64 + previous.width as f64 * 0.5;
    let translation =
        (mean_displacement + f64::from(scale_delta) * (center_sensor_x - mean_x)) as f32;
    let mut residual_sum = 0.0f64;
    let mut confidence_sum = 0.0f64;
    for node in &leaves {
        let weight = f64::from(node.confidence.max(0.03)).powi(2) * f64::from(node.samples.max(1));
        let x = previous.sensor_x as f64 + f64::from(node.source_center[0]) - center_sensor_x;
        let predicted = f64::from(translation) + f64::from(scale_delta) * x;
        residual_sum += weight * (predicted - f64::from(node.sensor_translation_px)).abs();
        confidence_sum += weight * f64::from(node.confidence);
    }
    let residual_px = (residual_sum / weight_sum.max(1.0e-9)) as f32;
    let horizontal_coverage =
        ((maximum_x - minimum_x) / previous.width.max(1) as f32).clamp(0.0, 1.0);
    let mean_confidence = (confidence_sum / weight_sum.max(1.0e-9)) as f32;
    let coverage_quality = (horizontal_coverage / 0.52).clamp(0.0, 1.0);
    let residual_quality = (1.0 - residual_px / 6.0).clamp(0.0, 1.0);
    let confidence = mean_confidence * (0.40 + 0.60 * coverage_quality) * residual_quality;
    let leaf_nodes = leaves.len();
    let reliable = leaf_nodes >= 2
        && horizontal_coverage >= 0.30
        && confidence >= 0.16
        && residual_px <= 4.0
        && translation.is_finite()
        && translation.abs() <= 48.0
        && scale_delta.is_finite()
        && scale_delta.abs() <= 0.10;
    HorizontalLightFieldStatus {
        nodes,
        horizontal_translation_px: translation,
        horizontal_scale_delta: scale_delta,
        residual_px,
        confidence,
        horizontal_coverage,
        leaf_nodes,
        evaluations,
        elapsed_micros,
        reliable,
    }
}

fn horizontal_light_field_tree(
    previous: &RawFrame,
    current: &RawFrame,
    prior_motion: SimilarityMotion,
) -> HorizontalLightFieldStatus {
    let started = Instant::now();
    if previous.width != current.width
        || previous.height != current.height
        || previous.width < 128
        || previous.height < 96
    {
        return HorizontalLightFieldStatus::default();
    }
    let previous_rows = NativeHorizontalRowIntegrals::new(previous);
    let current_rows = NativeHorizontalRowIntegrals::new(current);
    let prior_translation = if prior_motion.support >= 3
        && prior_motion.residual.is_finite()
        && prior_motion.residual <= 4.0
        && prior_motion.translation[0].abs() <= 24.0
    {
        prior_motion.translation
    } else {
        [0.0; 2]
    };
    let prior_scale = if prior_motion.support >= 4
        && prior_motion.scale_delta.is_finite()
        && prior_motion.scale_delta.abs() <= 0.06
    {
        prior_motion.scale_delta
    } else {
        0.0
    };
    let root_prior = HorizontalWalkTransform {
        translation: previous.sensor_x as f32 - current.sensor_x as f32 + prior_translation[0],
        scale_delta: prior_scale,
        vertical_nuisance: previous.sensor_y as f32 - current.sensor_y as f32
            + prior_translation[1],
    };
    let root_level = 0usize;
    let horizontal_guard = LIGHT_FIELD_BLUR_RADIUS[root_level] as f32
        + LIGHT_FIELD_TRANSLATION_RADIUS[root_level]
        + LIGHT_FIELD_SCALE_RADIUS[root_level] * previous.width as f32 * 0.5
        + 3.0;
    let vertical_guard = LIGHT_FIELD_BLUR_RADIUS[root_level] as f32 / 3.0
        + LIGHT_FIELD_VERTICAL_RADIUS[root_level]
        + 3.0;
    let bounds = [
        horizontal_guard + (-root_prior.translation).max(0.0),
        vertical_guard + (-root_prior.vertical_nuisance).max(0.0),
        previous.width as f32 - horizontal_guard - root_prior.translation.max(0.0),
        previous.height as f32 - vertical_guard - root_prior.vertical_nuisance.max(0.0),
    ];
    let mut nodes = Vec::new();
    let mut evaluations = 0usize;
    let _ = append_horizontal_light_field_node(
        previous,
        current,
        &previous_rows,
        &current_rows,
        bounds,
        0,
        None,
        root_prior,
        &mut nodes,
        &mut evaluations,
    );
    summarize_horizontal_light_field(
        previous,
        nodes,
        evaluations,
        started.elapsed().as_micros() as u64,
    )
}

fn horizontal_light_field_sensor_prediction(
    field: &HorizontalLightFieldStatus,
    previous: &RawFrame,
    current: &RawFrame,
    previous_sensor: [f32; 2],
) -> Option<f32> {
    if !field.reliable {
        return None;
    }
    let local = [
        previous_sensor[0] - previous.sensor_x as f32,
        previous_sensor[1] - previous.sensor_y as f32,
    ];
    let regional = field
        .nodes
        .iter()
        .filter(|node| {
            node.confidence >= 0.10
                && local[0] >= node.source_bounds[0]
                && local[0] <= node.source_bounds[2]
                && local[1] >= node.source_bounds[1]
                && local[1] <= node.source_bounds[3]
        })
        .max_by(|left, right| {
            left.depth
                .cmp(&right.depth)
                .then_with(|| left.confidence.total_cmp(&right.confidence))
        });
    let sensor_x = if let Some(node) = regional {
        current.sensor_x as f32
            + local[0]
            + node.local_translation_px
            + node.horizontal_scale_delta * (local[0] - node.source_center[0])
    } else {
        let center = previous.sensor_x as f32 + previous.width as f32 * 0.5;
        previous_sensor[0]
            + field.horizontal_translation_px
            + field.horizontal_scale_delta * (previous_sensor[0] - center)
    };
    sensor_x.is_finite().then_some(sensor_x)
}

/// Zero-copy backing frame for the segmentation-independent whole-ROI scale
/// gate. Unlike the Clusters matcher below, this path never constructs a
/// neutral image or pyramid: native RAW samples are carrier-neutralized only
/// at the sparse patch coordinates that are actually compared.
#[derive(Clone)]
struct SharedNativeRawFrame {
    sensor_x: u32,
    sensor_y: u32,
    width: usize,
    height: usize,
    pixels: Arc<Vec<u16>>,
}

/// Independent full-ROI evidence that an apparent radius change is supported
/// by coherent image scale elsewhere in the native eye frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeGlobalSimilarityEvidence {
    /// Motion authorized as a physical image-scale transport. This remains
    /// zero when the broadly distributed native matches fail any reliability
    /// gate, so downstream anatomy code cannot accidentally consume a merely
    /// diagnostic fit.
    pub motion: SimilarityMotion,
    /// Best robust fit before the whole-ROI reliability gates. Keeping this
    /// diagnostic separate is important: a zero authorized motion otherwise
    /// hides whether support, spatial coverage, residual, or excessive gross
    /// movement rejected an otherwise informative candidate.
    pub candidate_motion: SimilarityMotion,
    pub candidate_matches: usize,
    pub reliable: bool,
    pub stable_frames: u16,
    pub spatial_span: [f32; 2],
    pub occupied_quadrants: usize,
    /// Absolute sensor-space point about which `motion.translation` is
    /// defined. Keeping the center beside the fitted transform is essential
    /// when the sensor ROI moves: applying a scale/rotation about the current
    /// crop center would otherwise manufacture relative pupil motion.
    pub motion_center_sensor: [f32; 2],
}

/// Bounded native-resolution feature matcher used solely by the shared
/// physical-size feasibility gate. It does not identify an iris and cannot
/// publish anatomy; it only measures a broadly supported whole-ROI similarity
/// transform between adjacent lossless RAW frames.
#[derive(Default)]
pub struct NativeGlobalSimilarityTracker {
    previous: Option<SharedNativeRawFrame>,
    stable_frames: u16,
}

#[derive(Clone, Debug)]
struct CannyField {
    gradient_x: Vec<f32>,
    gradient_y: Vec<f32>,
    magnitude: Vec<f32>,
    accepted: Vec<bool>,
    high_threshold: f32,
    blurred: Option<Vec<f32>>,
    broad_blurred: Option<Vec<f32>>,
    primary_blur_micros: u64,
    gradient_micros: u64,
    hysteresis_micros: u64,
    nms_micros: u64,
    quantile_micros: u64,
    flood_micros: u64,
    broad_blur_micros: u64,
}

fn cfa_neutral_raw_scalar(pixels: &[u16], width: usize, height: usize) -> Vec<u16> {
    if width == 0 || height == 0 || pixels.len() != width * height {
        return Vec::new();
    }
    // A sliding 4x4 box spans one complete IMX582 Quad-Bayer carrier period.
    // Unlike a block reduction it remains translationally smooth, so neither
    // the CFA phase nor artificial 4-pixel block boundaries become corners.
    let mut horizontal = vec![0u32; pixels.len()];
    for y in 0..height {
        for x in 0..width {
            horizontal[y * width + x] = (-1isize..=2)
                .map(|offset| {
                    let sample_x = x.saturating_add_signed(offset).min(width - 1);
                    pixels[y * width + sample_x] as u32
                })
                .sum();
        }
    }
    let mut neutral = vec![0u16; pixels.len()];
    for y in 0..height {
        for x in 0..width {
            let sum = (-1isize..=2)
                .map(|offset| {
                    let sample_y = y.saturating_add_signed(offset).min(height - 1);
                    horizontal[sample_y * width + x]
                })
                .sum::<u32>();
            neutral[y * width + x] = ((sum + 8) / 16) as u16;
        }
    }
    neutral
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cfa_neutral_raw_avx2(pixels: &[u16], width: usize, height: usize) -> Vec<u16> {
    use std::arch::x86_64::*;

    let mut horizontal = vec![0u32; pixels.len()];
    let mut neutral = vec![0u16; pixels.len()];
    unsafe {
        let load_u16 =
            |pointer: *const u16| _mm256_cvtepu16_epi32(_mm_loadu_si128(pointer.cast::<__m128i>()));
        for y in 0..height {
            let row = y * width;
            let mut x = 0usize;
            while x < width.min(1) {
                horizontal[row + x] = (-1isize..=2)
                    .map(|offset| {
                        let sample_x = x.saturating_add_signed(offset).min(width - 1);
                        pixels[row + sample_x] as u32
                    })
                    .sum();
                x += 1;
            }
            while x + 9 < width {
                let index = row + x;
                let mut sum = load_u16(pixels.as_ptr().add(index - 1));
                sum = _mm256_add_epi32(sum, load_u16(pixels.as_ptr().add(index)));
                sum = _mm256_add_epi32(sum, load_u16(pixels.as_ptr().add(index + 1)));
                sum = _mm256_add_epi32(sum, load_u16(pixels.as_ptr().add(index + 2)));
                _mm256_storeu_si256(horizontal.as_mut_ptr().add(index).cast::<__m256i>(), sum);
                x += 8;
            }
            while x < width {
                horizontal[row + x] = (-1isize..=2)
                    .map(|offset| {
                        let sample_x = x.saturating_add_signed(offset).min(width - 1);
                        pixels[row + sample_x] as u32
                    })
                    .sum();
                x += 1;
            }
        }
        let rounding = _mm256_set1_epi32(8);
        for y in 0..height {
            let rows = [
                y.saturating_sub(1),
                y,
                (y + 1).min(height - 1),
                (y + 2).min(height - 1),
            ];
            let mut x = 0usize;
            while x + 7 < width {
                let mut sum = _mm256_loadu_si256(
                    horizontal
                        .as_ptr()
                        .add(rows[0] * width + x)
                        .cast::<__m256i>(),
                );
                for sample_y in &rows[1..] {
                    sum = _mm256_add_epi32(
                        sum,
                        _mm256_loadu_si256(
                            horizontal
                                .as_ptr()
                                .add(*sample_y * width + x)
                                .cast::<__m256i>(),
                        ),
                    );
                }
                sum = _mm256_srli_epi32(_mm256_add_epi32(sum, rounding), 4);
                let low = _mm256_castsi256_si128(sum);
                let high = _mm256_extracti128_si256::<1>(sum);
                let packed = _mm_packus_epi32(low, high);
                _mm_storeu_si128(
                    neutral.as_mut_ptr().add(y * width + x).cast::<__m128i>(),
                    packed,
                );
                x += 8;
            }
            while x < width {
                let sum = rows
                    .into_iter()
                    .map(|sample_y| horizontal[sample_y * width + x])
                    .sum::<u32>();
                neutral[y * width + x] = ((sum + 8) / 16) as u16;
                x += 1;
            }
        }
    }
    neutral
}

fn cfa_neutral_raw(pixels: &[u16], width: usize, height: usize) -> Vec<u16> {
    if width == 0 || height == 0 || pixels.len() != width * height {
        return Vec::new();
    }
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is runtime-verified; horizontal batches preserve the
        // carrier halo and vertical batches only read complete valid rows.
        return unsafe { cfa_neutral_raw_avx2(pixels, width, height) };
    }
    cfa_neutral_raw_scalar(pixels, width, height)
}

fn gaussian5_scalar(source: &[u16], width: usize, height: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0f32; source.len()];
    let mut output = vec![0.0f32; source.len()];
    const KERNEL: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    for y in 0..height {
        for x in 0..width {
            horizontal[y * width + x] = (-2isize..=2)
                .zip(KERNEL)
                .map(|(offset, weight)| {
                    let sample_x = x.saturating_add_signed(offset).min(width - 1);
                    source[y * width + sample_x] as f32 * weight
                })
                .sum::<f32>()
                / 16.0;
        }
    }
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = (-2isize..=2)
                .zip(KERNEL)
                .map(|(offset, weight)| {
                    let sample_y = y.saturating_add_signed(offset).min(height - 1);
                    horizontal[sample_y * width + x] * weight
                })
                .sum::<f32>()
                / 16.0;
        }
    }
    output
}

fn gaussian5_f32_scalar(source: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut horizontal = vec![0.0f32; source.len()];
    let mut output = vec![0.0f32; source.len()];
    const KERNEL: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    for y in 0..height {
        for x in 0..width {
            horizontal[y * width + x] = (-2isize..=2)
                .zip(KERNEL)
                .map(|(offset, weight)| {
                    let sample_x = x.saturating_add_signed(offset).min(width - 1);
                    source[y * width + sample_x] * weight
                })
                .sum::<f32>()
                / 16.0;
        }
    }
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = (-2isize..=2)
                .zip(KERNEL)
                .map(|(offset, weight)| {
                    let sample_y = y.saturating_add_signed(offset).min(height - 1);
                    horizontal[sample_y * width + x] * weight
                })
                .sum::<f32>()
                / 16.0;
        }
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gaussian5_avx2(source: &[u16], width: usize, height: usize) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut horizontal = vec![0.0f32; source.len()];
    let mut output = vec![0.0f32; source.len()];
    unsafe {
        let four = _mm256_set1_ps(4.0);
        let six = _mm256_set1_ps(6.0);
        let inverse_sixteen = _mm256_set1_ps(1.0 / 16.0);
        let load_u16 = |pointer: *const u16| {
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                pointer.cast::<__m128i>(),
            )))
        };
        for y in 0..height {
            let row = y * width;
            let mut x = 0usize;
            while x < width.min(2) {
                horizontal[row + x] = (-2isize..=2)
                    .zip([1.0f32, 4.0, 6.0, 4.0, 1.0])
                    .map(|(offset, weight)| {
                        let sample_x = x.saturating_add_signed(offset).min(width - 1);
                        source[row + sample_x] as f32 * weight
                    })
                    .sum::<f32>()
                    / 16.0;
                x += 1;
            }
            while x + 7 < width.saturating_sub(2) {
                let index = row + x;
                let left_two = load_u16(source.as_ptr().add(index - 2));
                let left_one = load_u16(source.as_ptr().add(index - 1));
                let center = load_u16(source.as_ptr().add(index));
                let right_one = load_u16(source.as_ptr().add(index + 1));
                let right_two = load_u16(source.as_ptr().add(index + 2));
                let mut sum = _mm256_add_ps(_mm256_setzero_ps(), left_two);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(left_one, four));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(center, six));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(right_one, four));
                sum = _mm256_add_ps(sum, right_two);
                _mm256_storeu_ps(
                    horizontal.as_mut_ptr().add(index),
                    _mm256_mul_ps(sum, inverse_sixteen),
                );
                x += 8;
            }
            while x < width {
                horizontal[row + x] = (-2isize..=2)
                    .zip([1.0f32, 4.0, 6.0, 4.0, 1.0])
                    .map(|(offset, weight)| {
                        let sample_x = x.saturating_add_signed(offset).min(width - 1);
                        source[row + sample_x] as f32 * weight
                    })
                    .sum::<f32>()
                    / 16.0;
                x += 1;
            }
        }
        for y in 0..height {
            let rows = [
                y.saturating_sub(2),
                y.saturating_sub(1),
                y,
                (y + 1).min(height - 1),
                (y + 2).min(height - 1),
            ];
            let mut x = 0usize;
            while x + 7 < width {
                let mut sum = _mm256_add_ps(
                    _mm256_setzero_ps(),
                    _mm256_loadu_ps(horizontal.as_ptr().add(rows[0] * width + x)),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(horizontal.as_ptr().add(rows[1] * width + x)),
                        four,
                    ),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(horizontal.as_ptr().add(rows[2] * width + x)),
                        six,
                    ),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(horizontal.as_ptr().add(rows[3] * width + x)),
                        four,
                    ),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_loadu_ps(horizontal.as_ptr().add(rows[4] * width + x)),
                );
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(y * width + x),
                    _mm256_mul_ps(sum, inverse_sixteen),
                );
                x += 8;
            }
            while x < width {
                output[y * width + x] = rows
                    .into_iter()
                    .zip([1.0f32, 4.0, 6.0, 4.0, 1.0])
                    .map(|(sample_y, weight)| horizontal[sample_y * width + x] * weight)
                    .sum::<f32>()
                    / 16.0;
                x += 1;
            }
        }
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gaussian5_f32_avx2(source: &[f32], width: usize, height: usize) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut horizontal = vec![0.0f32; source.len()];
    let mut output = vec![0.0f32; source.len()];
    unsafe {
        let four = _mm256_set1_ps(4.0);
        let six = _mm256_set1_ps(6.0);
        let inverse_sixteen = _mm256_set1_ps(1.0 / 16.0);
        for y in 0..height {
            let row = y * width;
            let mut x = 0usize;
            while x < width.min(2) {
                horizontal[row + x] = (-2isize..=2)
                    .zip([1.0f32, 4.0, 6.0, 4.0, 1.0])
                    .map(|(offset, weight)| {
                        let sample_x = x.saturating_add_signed(offset).min(width - 1);
                        source[row + sample_x] * weight
                    })
                    .sum::<f32>()
                    / 16.0;
                x += 1;
            }
            while x + 7 < width.saturating_sub(2) {
                let index = row + x;
                let mut sum = _mm256_add_ps(
                    _mm256_setzero_ps(),
                    _mm256_loadu_ps(source.as_ptr().add(index - 2)),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(_mm256_loadu_ps(source.as_ptr().add(index - 1)), four),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(_mm256_loadu_ps(source.as_ptr().add(index)), six),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(_mm256_loadu_ps(source.as_ptr().add(index + 1)), four),
                );
                sum = _mm256_add_ps(sum, _mm256_loadu_ps(source.as_ptr().add(index + 2)));
                _mm256_storeu_ps(
                    horizontal.as_mut_ptr().add(index),
                    _mm256_mul_ps(sum, inverse_sixteen),
                );
                x += 8;
            }
            while x < width {
                horizontal[row + x] = (-2isize..=2)
                    .zip([1.0f32, 4.0, 6.0, 4.0, 1.0])
                    .map(|(offset, weight)| {
                        let sample_x = x.saturating_add_signed(offset).min(width - 1);
                        source[row + sample_x] * weight
                    })
                    .sum::<f32>()
                    / 16.0;
                x += 1;
            }
        }
        for y in 0..height {
            let rows = [
                y.saturating_sub(2),
                y.saturating_sub(1),
                y,
                (y + 1).min(height - 1),
                (y + 2).min(height - 1),
            ];
            let mut x = 0usize;
            while x + 7 < width {
                let mut sum = _mm256_add_ps(
                    _mm256_setzero_ps(),
                    _mm256_loadu_ps(horizontal.as_ptr().add(rows[0] * width + x)),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(horizontal.as_ptr().add(rows[1] * width + x)),
                        four,
                    ),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(horizontal.as_ptr().add(rows[2] * width + x)),
                        six,
                    ),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(horizontal.as_ptr().add(rows[3] * width + x)),
                        four,
                    ),
                );
                sum = _mm256_add_ps(
                    sum,
                    _mm256_loadu_ps(horizontal.as_ptr().add(rows[4] * width + x)),
                );
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(y * width + x),
                    _mm256_mul_ps(sum, inverse_sixteen),
                );
                x += 8;
            }
            while x < width {
                output[y * width + x] = rows
                    .into_iter()
                    .zip([1.0f32, 4.0, 6.0, 4.0, 1.0])
                    .map(|(sample_y, weight)| horizontal[sample_y * width + x] * weight)
                    .sum::<f32>()
                    / 16.0;
                x += 1;
            }
        }
    }
    output
}

fn gaussian5(source: &[u16], width: usize, height: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is runtime-verified. Interior horizontal vectors leave
        // a two-pixel halo, and vertical vectors load complete valid rows.
        return unsafe { gaussian5_avx2(source, width, height) };
    }
    gaussian5_scalar(source, width, height)
}

fn gaussian5_f32(source: &[f32], width: usize, height: usize) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the same bounds proof as gaussian5_avx2 applies.
        return unsafe { gaussian5_f32_avx2(source, width, height) };
    }
    gaussian5_f32_scalar(source, width, height)
}

fn sample_f32_bilinear(image: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let top = image[y0 * width + x0] * (1.0 - fx) + image[y0 * width + x1] * fx;
    let bottom = image[y1 * width + x0] * (1.0 - fx) + image[y1 * width + x1] * fx;
    top * (1.0 - fy) + bottom * fy
}

fn canny_simd_disabled() -> bool {
    static SIMD_DISABLED: OnceLock<bool> = OnceLock::new();
    *SIMD_DISABLED.get_or_init(|| {
        std::env::var_os("BUTTERCUP_DISABLE_CANNY_SIMD").is_some_and(|value| value != "0")
    })
}

fn gradients_scalar(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
    kernel: CannyGradientKernel,
) {
    let (corner, middle) = match kernel {
        CannyGradientKernel::Sobel => (1.0, 2.0),
        CannyGradientKernel::Scharr => (3.0, 10.0),
    };
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let at = |dx: isize, dy: isize| {
                source[y.saturating_add_signed(dy) * width + x.saturating_add_signed(dx)]
            };
            let gx = -corner * at(-1, -1) + corner * at(1, -1) - middle * at(-1, 0)
                + middle * at(1, 0)
                - corner * at(-1, 1)
                + corner * at(1, 1);
            let gy = -corner * at(-1, -1) - middle * at(0, -1) - corner * at(1, -1)
                + corner * at(-1, 1)
                + middle * at(0, 1)
                + corner * at(1, 1);
            let index = y * width + x;
            let power = gx.hypot(gy);
            gradient_x[index] = gx;
            gradient_y[index] = gy;
            magnitude[index] = power;
            let (absolute_x, absolute_y) = (gx.abs(), gy.abs());
            direction[index] = if absolute_x >= absolute_y * 2.414 {
                0
            } else if absolute_y >= absolute_x * 2.414 {
                2
            } else if gx * gy >= 0.0 {
                1
            } else {
                3
            };
        }
    }
}

fn scharr_gradients_scalar(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
) {
    gradients_scalar(
        source,
        width,
        height,
        gradient_x,
        gradient_y,
        magnitude,
        direction,
        CannyGradientKernel::Scharr,
    );
}

fn sobel_gradients_scalar(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
) {
    gradients_scalar(
        source,
        width,
        height,
        gradient_x,
        gradient_y,
        magnitude,
        direction,
        CannyGradientKernel::Sobel,
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gradients_avx2(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
    kernel: CannyGradientKernel,
) {
    use std::arch::x86_64::*;

    unsafe {
        let (corner, middle) = match kernel {
            CannyGradientKernel::Sobel => (1.0, 2.0),
            CannyGradientKernel::Scharr => (3.0, 10.0),
        };
        let negative_corner = _mm256_set1_ps(-corner);
        let positive_corner = _mm256_set1_ps(corner);
        let negative_middle = _mm256_set1_ps(-middle);
        let positive_middle = _mm256_set1_ps(middle);
        for y in 1..height - 1 {
            let mut x = 1usize;
            while x + 7 < width - 1 {
                let index = y * width + x;
                let load = |offset: isize| {
                    _mm256_loadu_ps(source.as_ptr().offset(index as isize + offset))
                };
                let top_left = load(-(width as isize) - 1);
                let top_center = load(-(width as isize));
                let top_right = load(-(width as isize) + 1);
                let center_left = load(-1);
                let center_right = load(1);
                let bottom_left = load(width as isize - 1);
                let bottom_center = load(width as isize);
                let bottom_right = load(width as isize + 1);

                let mut gx = _mm256_mul_ps(negative_corner, top_left);
                gx = _mm256_add_ps(gx, _mm256_mul_ps(positive_corner, top_right));
                gx = _mm256_add_ps(gx, _mm256_mul_ps(negative_middle, center_left));
                gx = _mm256_add_ps(gx, _mm256_mul_ps(positive_middle, center_right));
                gx = _mm256_add_ps(gx, _mm256_mul_ps(negative_corner, bottom_left));
                gx = _mm256_add_ps(gx, _mm256_mul_ps(positive_corner, bottom_right));

                let mut gy = _mm256_mul_ps(negative_corner, top_left);
                gy = _mm256_add_ps(gy, _mm256_mul_ps(negative_middle, top_center));
                gy = _mm256_add_ps(gy, _mm256_mul_ps(negative_corner, top_right));
                gy = _mm256_add_ps(gy, _mm256_mul_ps(positive_corner, bottom_left));
                gy = _mm256_add_ps(gy, _mm256_mul_ps(positive_middle, bottom_center));
                gy = _mm256_add_ps(gy, _mm256_mul_ps(positive_corner, bottom_right));

                _mm256_storeu_ps(gradient_x.as_mut_ptr().add(index), gx);
                _mm256_storeu_ps(gradient_y.as_mut_ptr().add(index), gy);
                let mut gx_lanes = [0.0f32; 8];
                let mut gy_lanes = [0.0f32; 8];
                _mm256_storeu_ps(gx_lanes.as_mut_ptr(), gx);
                _mm256_storeu_ps(gy_lanes.as_mut_ptr(), gy);
                for lane in 0..8 {
                    let gx = gx_lanes[lane];
                    let gy = gy_lanes[lane];
                    magnitude[index + lane] = gx.hypot(gy);
                    let (absolute_x, absolute_y) = (gx.abs(), gy.abs());
                    direction[index + lane] = if absolute_x >= absolute_y * 2.414 {
                        0
                    } else if absolute_y >= absolute_x * 2.414 {
                        2
                    } else if gx * gy >= 0.0 {
                        1
                    } else {
                        3
                    };
                }
                x += 8;
            }
            for x in x..width - 1 {
                let at = |dx: isize, dy: isize| {
                    source[y.saturating_add_signed(dy) * width + x.saturating_add_signed(dx)]
                };
                let gx = -corner * at(-1, -1) + corner * at(1, -1) - middle * at(-1, 0)
                    + middle * at(1, 0)
                    - corner * at(-1, 1)
                    + corner * at(1, 1);
                let gy = -corner * at(-1, -1) - middle * at(0, -1) - corner * at(1, -1)
                    + corner * at(-1, 1)
                    + middle * at(0, 1)
                    + corner * at(1, 1);
                let index = y * width + x;
                magnitude[index] = gx.hypot(gy);
                gradient_x[index] = gx;
                gradient_y[index] = gy;
                let (absolute_x, absolute_y) = (gx.abs(), gy.abs());
                direction[index] = if absolute_x >= absolute_y * 2.414 {
                    0
                } else if absolute_y >= absolute_x * 2.414 {
                    2
                } else if gx * gy >= 0.0 {
                    1
                } else {
                    3
                };
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scharr_gradients_avx2(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
) {
    unsafe {
        gradients_avx2(
            source,
            width,
            height,
            gradient_x,
            gradient_y,
            magnitude,
            direction,
            CannyGradientKernel::Scharr,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sobel_gradients_avx2(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
) {
    unsafe {
        gradients_avx2(
            source,
            width,
            height,
            gradient_x,
            gradient_y,
            magnitude,
            direction,
            CannyGradientKernel::Sobel,
        )
    }
}

fn scharr_gradients(
    blurred: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is runtime-verified and vector loads remain within the
        // interior row plus its one-pixel Scharr halo.
        unsafe {
            scharr_gradients_avx2(
                blurred, width, height, gradient_x, gradient_y, magnitude, direction,
            )
        };
        return;
    }
    scharr_gradients_scalar(
        blurred, width, height, gradient_x, gradient_y, magnitude, direction,
    );
}

fn sobel_gradients(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
) {
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is runtime-verified and uses the same one-pixel halo
        // and vector bounds as the Scharr path above.
        unsafe {
            sobel_gradients_avx2(
                source, width, height, gradient_x, gradient_y, magnitude, direction,
            )
        };
        return;
    }
    sobel_gradients_scalar(
        source, width, height, gradient_x, gradient_y, magnitude, direction,
    );
}

fn canny_gradients(
    source: &[f32],
    width: usize,
    height: usize,
    gradient_x: &mut [f32],
    gradient_y: &mut [f32],
    magnitude: &mut [f32],
    direction: &mut [u8],
    kernel: CannyGradientKernel,
) {
    match kernel {
        CannyGradientKernel::Sobel => sobel_gradients(
            source, width, height, gradient_x, gradient_y, magnitude, direction,
        ),
        CannyGradientKernel::Scharr => scharr_gradients(
            source, width, height, gradient_x, gradient_y, magnitude, direction,
        ),
    }
}

fn nonmaximum_suppression_scalar(
    magnitude: &[f32],
    direction: &[u8],
    width: usize,
    height: usize,
) -> Vec<f32> {
    let mut suppressed = vec![0.0f32; width.saturating_mul(height)];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let index = y * width + x;
            let neighbors = match direction[index] {
                0 => (index - 1, index + 1),
                1 => (index - width - 1, index + width + 1),
                2 => (index - width, index + width),
                _ => (index - width + 1, index + width - 1),
            };
            let value = magnitude[index];
            if value >= magnitude[neighbors.0] && value >= magnitude[neighbors.1] {
                suppressed[index] = value;
            }
        }
    }
    suppressed
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nonmaximum_suppression_avx2(
    gradient_x: &[f32],
    gradient_y: &[f32],
    magnitude: &[f32],
    direction: &[u8],
    width: usize,
    height: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut suppressed = vec![0.0f32; width.saturating_mul(height)];
    unsafe {
        let zero = _mm256_setzero_ps();
        let all = _mm256_cmp_ps(zero, zero, _CMP_EQ_OQ);
        let ratio = _mm256_set1_ps(2.414);
        let absolute_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
        for y in 1..height - 1 {
            let mut x = 1usize;
            while x + 7 < width - 1 {
                let index = y * width + x;
                let gx = _mm256_loadu_ps(gradient_x.as_ptr().add(index));
                let gy = _mm256_loadu_ps(gradient_y.as_ptr().add(index));
                let value = _mm256_loadu_ps(magnitude.as_ptr().add(index));
                let absolute_x = _mm256_and_ps(gx, absolute_mask);
                let absolute_y = _mm256_and_ps(gy, absolute_mask);
                let horizontal =
                    _mm256_cmp_ps(absolute_x, _mm256_mul_ps(absolute_y, ratio), _CMP_GE_OQ);
                let vertical_candidate =
                    _mm256_cmp_ps(absolute_y, _mm256_mul_ps(absolute_x, ratio), _CMP_GE_OQ);
                let vertical = _mm256_andnot_ps(horizontal, vertical_candidate);
                let diagonal = _mm256_andnot_ps(_mm256_or_ps(horizontal, vertical), all);
                let same_sign = _mm256_cmp_ps(_mm256_mul_ps(gx, gy), zero, _CMP_GE_OQ);
                let diagonal_same = _mm256_and_ps(diagonal, same_sign);
                let diagonal_opposite = _mm256_andnot_ps(same_sign, diagonal);

                let pair_mask = |first_offset: isize, second_offset: isize| {
                    let first =
                        _mm256_loadu_ps(magnitude.as_ptr().offset(index as isize + first_offset));
                    let second =
                        _mm256_loadu_ps(magnitude.as_ptr().offset(index as isize + second_offset));
                    _mm256_and_ps(
                        _mm256_cmp_ps(value, first, _CMP_GE_OQ),
                        _mm256_cmp_ps(value, second, _CMP_GE_OQ),
                    )
                };
                let horizontal_keep = _mm256_and_ps(horizontal, pair_mask(-1, 1));
                let same_keep = _mm256_and_ps(
                    diagonal_same,
                    pair_mask(-(width as isize) - 1, width as isize + 1),
                );
                let vertical_keep =
                    _mm256_and_ps(vertical, pair_mask(-(width as isize), width as isize));
                let opposite_keep = _mm256_and_ps(
                    diagonal_opposite,
                    pair_mask(-(width as isize) + 1, width as isize - 1),
                );
                let keep = _mm256_or_ps(
                    _mm256_or_ps(horizontal_keep, same_keep),
                    _mm256_or_ps(vertical_keep, opposite_keep),
                );
                _mm256_storeu_ps(
                    suppressed.as_mut_ptr().add(index),
                    _mm256_and_ps(value, keep),
                );
                x += 8;
            }
            for x in x..width - 1 {
                let index = y * width + x;
                let neighbors = match direction[index] {
                    0 => (index - 1, index + 1),
                    1 => (index - width - 1, index + width + 1),
                    2 => (index - width, index + width),
                    _ => (index - width + 1, index + width - 1),
                };
                let value = magnitude[index];
                if value >= magnitude[neighbors.0] && value >= magnitude[neighbors.1] {
                    suppressed[index] = value;
                }
            }
        }
    }
    suppressed
}

fn nonmaximum_suppression(
    gradient_x: &[f32],
    gradient_y: &[f32],
    magnitude: &[f32],
    direction: &[u8],
    width: usize,
    height: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is runtime-verified, and each vector row excludes the
        // one-pixel border required by every neighbour load.
        return unsafe {
            nonmaximum_suppression_avx2(gradient_x, gradient_y, magnitude, direction, width, height)
        };
    }
    nonmaximum_suppression_scalar(magnitude, direction, width, height)
}

fn raw_f32(source: &[u16]) -> Vec<f32> {
    source.iter().map(|value| *value as f32).collect()
}

fn laplacian_response(source: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut response = vec![0.0; source.len()];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let index = y * width + x;
            response[index] = 4.0 * source[index]
                - source[index - 1]
                - source[index + 1]
                - source[index - width]
                - source[index + width];
        }
    }
    response
}

/// Return (physical intensity plane used for signed-side attributes,
/// optional alternate calculation plane differentiated by Canny).  `None`
/// means the physical narrow-Gaussian plane is also the calculation plane;
/// the default profile therefore retains the historical single allocation
/// and memory-bandwidth cost. Keeping these separate lets band-pass profiles
/// alter edge discovery without redefining which side of a discovered edge
/// is physically dark, bright, textured, or scleral.
fn learning_canny_inputs(
    frame: &RawFrame,
    filter: CannyInputFilter,
) -> (Vec<f32>, Option<Vec<f32>>) {
    let narrow = gaussian5(&frame.pixels, frame.width, frame.height);
    let calculation = match filter {
        CannyInputFilter::Raw => Some(raw_f32(&frame.pixels)),
        CannyInputFilter::Gaussian(passes) => {
            if passes == 0 {
                Some(raw_f32(&frame.pixels))
            } else if passes == 1 {
                None
            } else {
                let mut filtered = gaussian5_f32(&narrow, frame.width, frame.height);
                for _ in 2..passes {
                    filtered = gaussian5_f32(&filtered, frame.width, frame.height);
                }
                Some(filtered)
            }
        }
        CannyInputFilter::Laplacian => Some(laplacian_response(&narrow, frame.width, frame.height)),
        CannyInputFilter::DifferenceOfGaussians => {
            let medium = gaussian5_f32(&narrow, frame.width, frame.height);
            let broad = gaussian5_f32(&medium, frame.width, frame.height);
            Some(
                narrow
                    .iter()
                    .zip(broad.iter())
                    .map(|(near, far)| near - far)
                    .collect(),
            )
        }
    };
    (narrow, calculation)
}

fn canny_field(frame: &RawFrame) -> CannyField {
    canny_field_with_profile(frame, LearningCannyProfile::default())
}

fn canny_field_with_profile(frame: &RawFrame, profile: LearningCannyProfile) -> CannyField {
    let pixel_count = frame.width.saturating_mul(frame.height);
    let mut field = CannyField {
        gradient_x: vec![0.0; pixel_count],
        gradient_y: vec![0.0; pixel_count],
        magnitude: vec![0.0; pixel_count],
        accepted: vec![false; pixel_count],
        high_threshold: 0.0,
        blurred: None,
        broad_blurred: None,
        primary_blur_micros: 0,
        gradient_micros: 0,
        hysteresis_micros: 0,
        nms_micros: 0,
        quantile_micros: 0,
        flood_micros: 0,
        broad_blur_micros: 0,
    };
    if frame.width < 7 || frame.height < 7 || frame.pixels.len() != pixel_count {
        return field;
    }
    let settings = profile.settings();
    let primary_blur_started = Instant::now();
    let (blurred, alternate_calculation) = learning_canny_inputs(frame, settings.input);
    field.primary_blur_micros = primary_blur_started.elapsed().as_micros() as u64;
    let gradient_started = Instant::now();
    let mut direction = vec![0u8; pixel_count];
    let calculation = alternate_calculation.as_deref().unwrap_or(&blurred);
    canny_gradients(
        calculation,
        frame.width,
        frame.height,
        &mut field.gradient_x,
        &mut field.gradient_y,
        &mut field.magnitude,
        &mut direction,
        settings.kernel,
    );
    field.gradient_micros = gradient_started.elapsed().as_micros() as u64;
    let hysteresis_started = Instant::now();
    let nms_started = Instant::now();
    let suppressed = nonmaximum_suppression(
        &field.gradient_x,
        &field.gradient_y,
        &field.magnitude,
        &direction,
        frame.width,
        frame.height,
    );
    let mut positive = suppressed
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect::<Vec<_>>();
    field.nms_micros = nms_started.elapsed().as_micros() as u64;
    if positive.is_empty() {
        field.hysteresis_micros = hysteresis_started.elapsed().as_micros() as u64;
        return field;
    }
    let quantile_started = Instant::now();
    let quantile_index = ((positive.len() - 1) as f64 * settings.high_quantile).round() as usize;
    positive.select_nth_unstable_by(quantile_index, f32::total_cmp);
    let high = positive[quantile_index].max(settings.high_floor);
    let low = high * settings.low_ratio;
    field.high_threshold = high;
    let mut stack = suppressed
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value >= high).then_some(index))
        .collect::<Vec<_>>();
    field.quantile_micros = quantile_started.elapsed().as_micros() as u64;
    let flood_started = Instant::now();
    for &index in &stack {
        field.accepted[index] = true;
    }
    while let Some(index) = stack.pop() {
        // Only non-maxima from the interior can enter this stack: the
        // suppressed border is identically zero while `low` is positive.
        // Preserve the old row-major eight-neighbour visit order without
        // paying for saturating coordinate arithmetic and division at every
        // propagated edge pixel.
        let neighbors = [
            index - frame.width - 1,
            index - frame.width,
            index - frame.width + 1,
            index - 1,
            index + 1,
            index + frame.width - 1,
            index + frame.width,
            index + frame.width + 1,
        ];
        for neighbor in neighbors {
            if !field.accepted[neighbor] && suppressed[neighbor] >= low {
                field.accepted[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    field.flood_micros = flood_started.elapsed().as_micros() as u64;
    field.hysteresis_micros = hysteresis_started.elapsed().as_micros() as u64;

    // Retain the two native-resolution blur planes only until the bounded
    // edge bank consumes them immediately below this stage.
    let broad_blur_started = Instant::now();
    let broad_blurred = gaussian5_f32(&blurred, frame.width, frame.height);
    field.broad_blur_micros = broad_blur_started.elapsed().as_micros() as u64;
    field.blurred = Some(blurred);
    field.broad_blurred = Some(broad_blurred);
    field
}

#[derive(Default)]
struct EdgeEvidenceOutput {
    edges: Vec<EdgeEvidence>,
    attribute_micros: u64,
    texture_micros: u64,
    attribute_candidates: usize,
    attribute_evaluated: usize,
    texture_evaluated: usize,
    texture_simd_evaluated: usize,
}

#[derive(Clone, Copy)]
struct EdgeAttributeCandidate {
    index: usize,
    upper_strength: f32,
}

struct ScoredEdgeEvidence {
    edge: EdgeEvidence,
    far_step: f32,
}

fn measured_edge_score_attributes(
    field: &CannyField,
    blurred: &[f32],
    broad_blurred: &[f32],
    width: usize,
    height: usize,
    candidate: EdgeAttributeCandidate,
) -> ScoredEdgeEvidence {
    let index = candidate.index;
    let x = index % width;
    let y = index / width;
    let gx = field.gradient_x[index];
    let gy = field.gradient_y[index];
    let magnitude = field.magnitude[index];
    let broad_at = |dx: isize, dy: isize| {
        broad_blurred[y.saturating_add_signed(dy) * width + x.saturating_add_signed(dx)]
    };
    let broad_gx = -3.0 * broad_at(-1, -1) + 3.0 * broad_at(1, -1) - 10.0 * broad_at(-1, 0)
        + 10.0 * broad_at(1, 0)
        - 3.0 * broad_at(-1, 1)
        + 3.0 * broad_at(1, 1);
    let broad_gy = -3.0 * broad_at(-1, -1) - 10.0 * broad_at(0, -1) - 3.0 * broad_at(1, -1)
        + 3.0 * broad_at(-1, 1)
        + 10.0 * broad_at(0, 1)
        + 3.0 * broad_at(1, 1);
    let broad_magnitude = broad_gx.hypot(broad_gy);
    let signed_alignment =
        (gx * broad_gx + gy * broad_gy) / (magnitude * broad_magnitude).max(1.0e-6);
    let broad_power = (broad_magnitude / (magnitude * 0.34).max(1.0e-6)).clamp(0.0, 1.0);
    let multiscale_consistency = signed_alignment.clamp(0.0, 1.0) * broad_power;

    let normal_x = gx / magnitude;
    let normal_y = gy / magnitude;
    let sample_normal = |offset: f32| {
        sample_f32_bilinear(
            blurred,
            width,
            height,
            x as f32 + normal_x * offset,
            y as f32 + normal_y * offset,
        )
    };
    let near_step = sample_normal(2.0) - sample_normal(-2.0);
    let far_step = sample_normal(5.0) - sample_normal(-5.0);
    let signed_step_persistence = if near_step > 0.0 && far_step > 0.0 {
        near_step.min(far_step) / near_step.max(far_step).max(1.0e-6)
    } else {
        0.0
    };
    let reliability =
        (0.35 + 0.65 * multiscale_consistency) * (0.40 + 0.60 * signed_step_persistence);
    ScoredEdgeEvidence {
        edge: EdgeEvidence {
            x: x as f32,
            y: y as f32,
            gradient_x: normal_x,
            gradient_y: normal_y,
            strength: candidate.upper_strength * reliability,
            multiscale_consistency,
            signed_step_persistence,
            dark_side_texture: 0.0,
            bright_side_texture: 0.0,
            iris_motion_consistency: 1.0,
        },
        far_step,
    }
}

fn measure_edge_side_textures(
    blurred: &[f32],
    width: usize,
    height: usize,
    scored: &mut ScoredEdgeEvidence,
) {
    let x = scored.edge.x as usize;
    let y = scored.edge.y as usize;
    let normal_x = scored.edge.gradient_x;
    let normal_y = scored.edge.gradient_y;
    let tangent_x = -normal_y;
    let tangent_y = normal_x;
    let side_texture = |normal_offset: f32| {
        let sample = |tangent_offset: f32| {
            sample_f32_bilinear(
                blurred,
                width,
                height,
                x as f32 + normal_x * normal_offset + tangent_x * tangent_offset,
                y as f32 + normal_y * normal_offset + tangent_y * tangent_offset,
            )
        };
        let left = sample(-3.0);
        let center = sample(0.0);
        let right = sample(3.0);
        ((2.0 * center - left - right).abs() / scored.far_step.abs().max(8.0)).clamp(0.0, 1.0)
    };
    scored.edge.dark_side_texture = side_texture(-4.0);
    scored.edge.bright_side_texture = side_texture(4.0);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bilinear_sample_8_avx2(
    image: &[f32],
    width: usize,
    height: usize,
    x: [f32; 8],
    y: [f32; 8],
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let mut top_left = [0i32; 8];
    let mut top_right = [0i32; 8];
    let mut bottom_left = [0i32; 8];
    let mut bottom_right = [0i32; 8];
    let mut fraction_x = [0.0f32; 8];
    let mut fraction_y = [0.0f32; 8];
    for lane in 0..8 {
        let sample_x = x[lane].clamp(0.0, width.saturating_sub(1) as f32);
        let sample_y = y[lane].clamp(0.0, height.saturating_sub(1) as f32);
        let x0 = sample_x.floor() as usize;
        let y0 = sample_y.floor() as usize;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);
        top_left[lane] = (y0 * width + x0) as i32;
        top_right[lane] = (y0 * width + x1) as i32;
        bottom_left[lane] = (y1 * width + x0) as i32;
        bottom_right[lane] = (y1 * width + x1) as i32;
        fraction_x[lane] = sample_x - x0 as f32;
        fraction_y[lane] = sample_y - y0 as f32;
    }
    unsafe {
        let indices = |values: &[i32; 8]| _mm256_loadu_si256(values.as_ptr().cast::<__m256i>());
        let tl = _mm256_i32gather_ps(image.as_ptr(), indices(&top_left), 4);
        let tr = _mm256_i32gather_ps(image.as_ptr(), indices(&top_right), 4);
        let bl = _mm256_i32gather_ps(image.as_ptr(), indices(&bottom_left), 4);
        let br = _mm256_i32gather_ps(image.as_ptr(), indices(&bottom_right), 4);
        let fx = _mm256_loadu_ps(fraction_x.as_ptr());
        let fy = _mm256_loadu_ps(fraction_y.as_ptr());
        let one = _mm256_set1_ps(1.0);
        let top = _mm256_add_ps(
            _mm256_mul_ps(tl, _mm256_sub_ps(one, fx)),
            _mm256_mul_ps(tr, fx),
        );
        let bottom = _mm256_add_ps(
            _mm256_mul_ps(bl, _mm256_sub_ps(one, fx)),
            _mm256_mul_ps(br, fx),
        );
        _mm256_add_ps(
            _mm256_mul_ps(top, _mm256_sub_ps(one, fy)),
            _mm256_mul_ps(bottom, fy),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn measure_edge_side_textures_avx2(
    blurred: &[f32],
    width: usize,
    height: usize,
    scored: &mut [ScoredEdgeEvidence],
) -> usize {
    use std::arch::x86_64::*;

    let simd_count = scored.len() / 8 * 8;
    for batch in scored[..simd_count].chunks_exact_mut(8) {
        let mut center_x = [0.0f32; 8];
        let mut center_y = [0.0f32; 8];
        let mut normal_x = [0.0f32; 8];
        let mut normal_y = [0.0f32; 8];
        let mut tangent_x = [0.0f32; 8];
        let mut tangent_y = [0.0f32; 8];
        let mut denominator = [0.0f32; 8];
        for lane in 0..8 {
            center_x[lane] = batch[lane].edge.x;
            center_y[lane] = batch[lane].edge.y;
            normal_x[lane] = batch[lane].edge.gradient_x;
            normal_y[lane] = batch[lane].edge.gradient_y;
            tangent_x[lane] = -normal_y[lane];
            tangent_y[lane] = normal_x[lane];
            denominator[lane] = batch[lane].far_step.abs().max(8.0);
        }
        for (normal_offset, dark_side) in [(-4.0f32, true), (4.0f32, false)] {
            let sample_at = |tangent_offset: f32| {
                let mut x = [0.0f32; 8];
                let mut y = [0.0f32; 8];
                for lane in 0..8 {
                    x[lane] = center_x[lane]
                        + normal_x[lane] * normal_offset
                        + tangent_x[lane] * tangent_offset;
                    y[lane] = center_y[lane]
                        + normal_y[lane] * normal_offset
                        + tangent_y[lane] * tangent_offset;
                }
                unsafe { bilinear_sample_8_avx2(blurred, width, height, x, y) }
            };
            unsafe {
                let left = sample_at(-3.0);
                let center = sample_at(0.0);
                let right = sample_at(3.0);
                let numerator = _mm256_andnot_ps(
                    _mm256_set1_ps(-0.0),
                    _mm256_sub_ps(_mm256_sub_ps(_mm256_add_ps(center, center), left), right),
                );
                let value = _mm256_min_ps(
                    _mm256_set1_ps(1.0),
                    _mm256_max_ps(
                        _mm256_setzero_ps(),
                        _mm256_div_ps(numerator, _mm256_loadu_ps(denominator.as_ptr())),
                    ),
                );
                let mut output = [0.0f32; 8];
                _mm256_storeu_ps(output.as_mut_ptr(), value);
                for lane in 0..8 {
                    if dark_side {
                        batch[lane].edge.dark_side_texture = output[lane];
                    } else {
                        batch[lane].edge.bright_side_texture = output[lane];
                    }
                }
            }
        }
    }
    for edge in &mut scored[simd_count..] {
        measure_edge_side_textures(blurred, width, height, edge);
    }
    simd_count
}

fn measure_edge_side_textures_batched(
    blurred: &[f32],
    width: usize,
    height: usize,
    scored: &mut [ScoredEdgeEvidence],
) -> usize {
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection above proves AVX2 availability;
        // every gathered index is clamped to the supplied image dimensions.
        return unsafe { measure_edge_side_textures_avx2(blurred, width, height, scored) };
    }
    for edge in scored {
        measure_edge_side_textures(blurred, width, height, edge);
    }
    0
}

fn edge_evidence(field: &mut CannyField, width: usize, height: usize) -> EdgeEvidenceOutput {
    if field.high_threshold <= 0.0 {
        field.blurred = None;
        field.broad_blurred = None;
        return EdgeEvidenceOutput::default();
    }
    let Some(blurred) = field.blurred.take() else {
        return EdgeEvidenceOutput::default();
    };
    let Some(broad_blurred) = field.broad_blurred.take() else {
        return EdgeEvidenceOutput::default();
    };
    let tile_columns = width.div_ceil(EDGE_TILE_SIZE);
    let tile_rows = height.div_ceil(EDGE_TILE_SIZE);
    let mut tiles = vec![Vec::<EdgeAttributeCandidate>::new(); tile_columns * tile_rows];
    for y in 2..height.saturating_sub(2) {
        for x in 2..width.saturating_sub(2) {
            let index = y * width + x;
            if !field.accepted[index] {
                continue;
            }
            let magnitude = field.magnitude[index];
            if magnitude <= 1.0e-6 {
                continue;
            }
            let tile = (y / EDGE_TILE_SIZE) * tile_columns + x / EDGE_TILE_SIZE;
            tiles[tile].push(EdgeAttributeCandidate {
                index,
                // Both multiplicative reliability terms are in [0, 1], so
                // this is a strict upper bound on final edge strength.
                upper_strength: (magnitude / field.high_threshold).clamp(0.20, 4.0),
            });
        }
    }
    let attribute_candidates = tiles.iter().map(Vec::len).sum();
    let attribute_started = Instant::now();
    let mut attribute_evaluated = 0usize;
    let mut selected = Vec::<ScoredEdgeEvidence>::new();
    for tile in &mut tiles {
        tile.sort_by(|left, right| {
            right
                .upper_strength
                .total_cmp(&left.upper_strength)
                .then_with(|| left.index.cmp(&right.index))
        });
        let mut evaluated = Vec::<(usize, ScoredEdgeEvidence)>::new();
        let mut top_strengths = [f32::NEG_INFINITY; EDGES_PER_TILE];
        for (candidate_index, candidate) in tile.iter().copied().enumerate() {
            attribute_evaluated += 1;
            let edge = measured_edge_score_attributes(
                field,
                &blurred,
                &broad_blurred,
                width,
                height,
                candidate,
            );
            let mut insertion = EDGES_PER_TILE;
            for (rank, strength) in top_strengths.iter().enumerate() {
                if edge.edge.strength > *strength {
                    insertion = rank;
                    break;
                }
            }
            if insertion < EDGES_PER_TILE {
                top_strengths.copy_within(insertion..EDGES_PER_TILE - 1, insertion + 1);
                top_strengths[insertion] = edge.edge.strength;
            }
            evaluated.push((candidate.index, edge));
            if evaluated.len() >= EDGES_PER_TILE {
                let next_upper = tile
                    .get(candidate_index + 1)
                    .map_or(f32::NEG_INFINITY, |next| next.upper_strength);
                // Strict inequality deliberately evaluates every candidate
                // tied at the cutoff, preserving the former stable row-major
                // sort exactly.
                if next_upper < top_strengths[EDGES_PER_TILE - 1] {
                    break;
                }
            }
        }
        evaluated.sort_by(|(left_index, left), (right_index, right)| {
            right
                .edge
                .strength
                .total_cmp(&left.edge.strength)
                .then_with(|| left_index.cmp(right_index))
        });
        evaluated.truncate(EDGES_PER_TILE);
        selected.extend(evaluated.into_iter().map(|(_, edge)| edge));
    }
    let attribute_micros = attribute_started.elapsed().as_micros() as u64;
    let texture_evaluated = selected.len();
    let texture_started = Instant::now();
    let texture_simd_evaluated =
        measure_edge_side_textures_batched(&blurred, width, height, &mut selected);
    let texture_micros = texture_started.elapsed().as_micros() as u64;
    EdgeEvidenceOutput {
        edges: selected.into_iter().map(|edge| edge.edge).collect(),
        attribute_micros,
        texture_micros,
        attribute_candidates,
        attribute_evaluated,
        texture_evaluated,
        texture_simd_evaluated,
    }
}

/// Extract native-resolution Canny evidence and spatially bounded feature
/// proposals without running the multi-frame patch matcher. This is the
/// cadence-safe diagnostic/probe path for Driving and ROI-censored Native
/// frames; Clusters mode uses `FourMotionOctrees::observe` for full temporal
/// promotion instead.
pub fn canny_proposal_overlay(pixels: &[u16], width: usize, height: usize) -> MotionOctreeOverlay {
    let current = RawFrame {
        sensor_x: 0,
        sensor_y: 0,
        width,
        height,
        pixels: cfa_neutral_raw(pixels, width, height),
    };
    let mut field = canny_field(&current);
    let edges = edge_evidence(&mut field, width, height).edges;
    let provisional_features = seed_points(&current, Some(&field), &edges, None, &[], MAX_FEATURES)
        .into_iter()
        .map(|(point, _)| (point[0], point[1]))
        .collect();
    MotionOctreeOverlay {
        edges,
        edge_high_threshold: field.high_threshold,
        provisional_features,
        ..MotionOctreeOverlay::default()
    }
}

const BOUNDED_IRIS_FEATURES: usize = 24;
const BOUNDED_IRIS_PATCH_RADIUS: i32 = 4;
const BOUNDED_IRIS_SEARCH_RADIUS: i32 = 7;
const BOUNDED_IRIS_MAX_AGE: u8 = 1;
const MOTION_SHADOW_CELL_SIZE: usize = 12;
const MOTION_SHADOW_PATCH_RADIUS: i32 = 2;
const MOTION_SHADOW_SEARCH_RADIUS: i32 = 6;

#[derive(Clone, Debug)]
struct BoundedIrisTrack {
    id: u64,
    points: VecDeque<[f32; 2]>, // absolute sensor coordinates
    score: f32,
    age: u8,
    matched_streak: u8,
}

#[derive(Clone, Copy, Debug)]
struct BoundedIrisMatch {
    track: usize,
    current: [f32; 2], // absolute sensor coordinates
    cost: f32,
}

fn bounded_iris_similarity_motion(
    pairs: &[([f32; 2], [f32; 2])],
    center: [f32; 2],
) -> SimilarityMotion {
    let matches = pairs
        .iter()
        .enumerate()
        .map(|(track_index, (previous, current))| Match {
            track_index,
            previous: *previous,
            current: *current,
            score: 1.0,
            object: PUPIL_LAYER,
            z: 0.0,
            assignment_margin: 1.0,
            layer_evidence: true,
            normal_flow_evidence: false,
            specularity: 0.0,
        })
        .collect::<Vec<_>>();
    robust_global_similarity(&matches, center)
}

/// Express the same similarity transform about a different image point.
/// `translation` is the displacement at the transform center, so rotation or
/// scale makes a naïve center substitution change the physical transform.
fn reanchor_similarity_motion(
    mut motion: SimilarityMotion,
    old_center: [f32; 2],
    new_center: [f32; 2],
) -> SimilarityMotion {
    let x = new_center[0] - old_center[0];
    let y = new_center[1] - old_center[1];
    motion.translation = [
        motion.translation[0] + motion.scale_delta * x - motion.rotation * y,
        motion.translation[1] + motion.rotation * x + motion.scale_delta * y,
    ];
    motion
}

/// Reduce upper-eye Canny votes which do not follow the robust iris motion.
///
/// The comparison stays on the native CFA-neutral RAW grid. One strongest
/// edge patch per 12x12 cell is checked against the iris-predicted location in
/// the previous frame and a small alternative-motion neighborhood. The
/// resulting analog weight is shared by upper-annular edges in that cell,
/// bounding the extra work independently of the length of a shadow contour.
/// With too little iris motion or weak temporal support the function leaves
/// every edge at `1.0`; a static image cannot honestly distinguish a cast
/// shadow from anatomy by motion.
fn condition_upper_edges_by_iris_motion(
    edges: &mut [EdgeEvidence],
    previous: &RawFrame,
    current: &RawFrame,
    seed: IrisEllipseSeed,
    iris_motion: SimilarityMotion,
    iris_center_absolute: [f32; 2],
) -> usize {
    if previous.width != current.width
        || previous.height != current.height
        || iris_motion.support < 4
        || !iris_motion.residual.is_finite()
        || iris_motion.residual > 2.5
        || !iris_motion.translation[0].is_finite()
        || !iris_motion.translation[1].is_finite()
        || !iris_motion.rotation.is_finite()
        || !iris_motion.scale_delta.is_finite()
        || !iris_center_absolute[0].is_finite()
        || !iris_center_absolute[1].is_finite()
    {
        return 0;
    }

    let ellipse = seed.ellipse();
    if ellipse.major < 12.0 || ellipse.minor < 8.0 {
        return 0;
    }
    let cosine = ellipse.angle.cos();
    let sine = ellipse.angle.sin();
    let columns = current.width.div_ceil(MOTION_SHADOW_CELL_SIZE);
    let rows = current.height.div_ceil(MOTION_SHADOW_CELL_SIZE);
    let mut representatives = vec![None::<usize>; columns.saturating_mul(rows)];
    let upper_annular_cell = |edge: &EdgeEvidence| {
        let dx = edge.x as f64 - ellipse.center.0;
        let dy = edge.y as f64 - ellipse.center.1;
        let local_x = cosine * dx + sine * dy;
        let local_y = -sine * dx + cosine * dy;
        let canonical_x = local_x / ellipse.major.max(1.0);
        let canonical_y = local_y / ellipse.minor.max(1.0);
        canonical_y <= -0.05 && (0.30..=1.55).contains(&canonical_x.hypot(canonical_y))
    };
    for (index, edge) in edges.iter().enumerate() {
        if !upper_annular_cell(edge) {
            continue;
        }
        let cell_x = (edge.x.max(0.0) as usize / MOTION_SHADOW_CELL_SIZE).min(columns - 1);
        let cell_y = (edge.y.max(0.0) as usize / MOTION_SHADOW_CELL_SIZE).min(rows - 1);
        let cell = cell_y * columns + cell_x;
        if representatives[cell].is_none_or(|prior| edges[prior].strength < edge.strength) {
            representatives[cell] = Some(index);
        }
    }

    let patch_inside = |point: [f32; 2], frame: &RawFrame| {
        let x = point[0].round() as i32;
        let y = point[1].round() as i32;
        x >= MOTION_SHADOW_PATCH_RADIUS
            && y >= MOTION_SHADOW_PATCH_RADIUS
            && x + MOTION_SHADOW_PATCH_RADIUS < frame.width as i32
            && y + MOTION_SHADOW_PATCH_RADIUS < frame.height as i32
    };
    let mut cell_consistency = vec![1.0f32; representatives.len()];
    for (cell, representative) in representatives.into_iter().enumerate() {
        let Some(edge_index) = representative else {
            continue;
        };
        let edge = edges[edge_index];
        let current_point = [edge.x, edge.y];
        let current_absolute = [
            edge.x + current.sensor_x as f32,
            edge.y + current.sensor_y as f32,
        ];
        // Invert the same linearized similarity used by the native tracker,
        // rather than treating a rotating/scaling iris as translation-only.
        // This avoids labelling a real upper iris striation as a shadow merely
        // because it is far from the similarity center.
        let affine_scale = 1.0 + iris_motion.scale_delta;
        let determinant = affine_scale * affine_scale + iris_motion.rotation * iris_motion.rotation;
        if !determinant.is_finite() || determinant < 0.25 {
            continue;
        }
        let current_relative = [
            current_absolute[0] - iris_center_absolute[0] - iris_motion.translation[0],
            current_absolute[1] - iris_center_absolute[1] - iris_motion.translation[1],
        ];
        let predicted_previous_absolute = [
            iris_center_absolute[0]
                + (affine_scale * current_relative[0] + iris_motion.rotation * current_relative[1])
                    / determinant,
            iris_center_absolute[1]
                + (-iris_motion.rotation * current_relative[0]
                    + affine_scale * current_relative[1])
                    / determinant,
        ];
        let expected_translation = [
            current_absolute[0] - predicted_previous_absolute[0],
            current_absolute[1] - predicted_previous_absolute[1],
        ];
        if expected_translation[0].hypot(expected_translation[1]) < 0.75 {
            continue;
        }
        let iris_previous = [
            predicted_previous_absolute[0] - previous.sensor_x as f32,
            predicted_previous_absolute[1] - previous.sensor_y as f32,
        ];
        if !patch_inside(current_point, current) || !patch_inside(iris_previous, previous) {
            continue;
        }
        let iris_cost = patch_cost_with_radius(
            previous,
            current,
            iris_previous,
            current_point,
            MOTION_SHADOW_PATCH_RADIUS,
        );
        if !iris_cost.is_finite() {
            continue;
        }
        let mut best = (iris_cost, iris_previous);
        for dy in -MOTION_SHADOW_SEARCH_RADIUS..=MOTION_SHADOW_SEARCH_RADIUS {
            for dx in -MOTION_SHADOW_SEARCH_RADIUS..=MOTION_SHADOW_SEARCH_RADIUS {
                let candidate = [iris_previous[0] + dx as f32, iris_previous[1] + dy as f32];
                if !patch_inside(candidate, previous) {
                    continue;
                }
                let cost = patch_cost_with_radius(
                    previous,
                    current,
                    candidate,
                    current_point,
                    MOTION_SHADOW_PATCH_RADIUS,
                );
                if cost < best.0 {
                    best = (cost, candidate);
                }
            }
        }
        let observed_translation = [
            current_absolute[0] - (best.1[0] + previous.sensor_x as f32),
            current_absolute[1] - (best.1[1] + previous.sensor_y as f32),
        ];
        let motion_disagreement = (observed_translation[0] - expected_translation[0])
            .hypot(observed_translation[1] - expected_translation[1]);
        let improvement = iris_cost - best.0;
        let relative_improvement = improvement / iris_cost.max(0.10);
        if best.0 <= 0.65
            && improvement >= 0.045
            && relative_improvement >= 0.08
            && motion_disagreement >= 1.25
        {
            let preference = ((improvement - 0.035) / 0.20).clamp(0.0, 1.0);
            let separation = ((motion_disagreement - 1.0) / 3.0).clamp(0.0, 1.0);
            cell_consistency[cell] = (1.0 - 0.88 * preference * separation).clamp(0.12, 1.0);
        }
    }

    let mut downweighted = 0usize;
    for edge in edges {
        if !upper_annular_cell(edge) {
            continue;
        }
        let cell_x = (edge.x.max(0.0) as usize / MOTION_SHADOW_CELL_SIZE).min(columns - 1);
        let cell_y = (edge.y.max(0.0) as usize / MOTION_SHADOW_CELL_SIZE).min(rows - 1);
        let consistency = cell_consistency[cell_y * columns + cell_x];
        if consistency < 0.999 {
            edge.iris_motion_consistency = consistency;
            downweighted += 1;
        }
    }
    downweighted
}

/// Small native-resolution matcher restricted to an anatomically proposed
/// iris annulus.  It deliberately has no pyramid/downsample stage: at most 24
/// CFA-neutral RAW patches are searched over a 15x15 full-resolution window.
/// The result can corroborate a censored proposal across frames, but it is not
/// an eye detector and cannot publish anatomy by itself.
#[derive(Default)]
pub struct BoundedIrisCannyTracker {
    previous: Option<RawFrame>,
    tracks: Vec<BoundedIrisTrack>,
    generation: u64,
    next_id: u64,
    stable_frames: u16,
    /// State-only cyan/green kinematics. The cyan transform is supplied by
    /// the already-running sparse whole-ROI native matcher; this tracker never
    /// starts the expensive general feature bank in Driving mode.
    coupled_kinematics: CoupledEyeKinematics,
}

fn bounded_iris_contains(seed: IrisEllipseSeed, point: [f32; 2], range: (f64, f64)) -> bool {
    let ellipse = seed.ellipse();
    if ellipse.major <= 1.0 || ellipse.minor <= 1.0 {
        return false;
    }
    let radius = normalized_ellipse_radius(ellipse, (point[0] as f64, point[1] as f64));
    (range.0..=range.1).contains(&radius)
}

fn bounded_iris_seed_points(
    frame: &RawFrame,
    canny: &CannyField,
    seed: IrisEllipseSeed,
    existing: &[[f32; 2]],
    wanted: usize,
) -> Vec<([f32; 2], f32)> {
    if wanted == 0 {
        return Vec::new();
    }
    // Build this bank *inside* the affine iris annulus. Filtering a completed
    // global bank still lets a detailed lid, lash, or brow consume most of the
    // global candidates before the annular filter runs. These are native RAW
    // samples on the original pixel grid; there is no pyramid or resized
    // intermediate.
    const ANGULAR_SECTORS: usize = 16;
    const MAX_PER_SECTOR: u8 = 3;
    let ellipse = seed.ellipse();
    let cosine = ellipse.angle.cos();
    let sine = ellipse.angle.sin();
    let tile_columns = frame.width.div_ceil(FEATURE_SEED_TILE_SIZE);
    let tile_rows = frame.height.div_ceil(FEATURE_SEED_TILE_SIZE);
    let mut candidates = Vec::<([f32; 2], f32, usize, usize)>::new();
    for y in (5..frame.height.saturating_sub(5)).step_by(3) {
        for x in (5..frame.width.saturating_sub(5)).step_by(3) {
            let point = [x as f32, y as f32];
            if !bounded_iris_contains(seed, point, (0.30, 0.92)) {
                continue;
            }
            let edge_support = local_canny_support(canny, frame.width, x, y);
            if edge_support < 0.35 {
                continue;
            }
            let corner = corner_score(frame, x, y) * edge_support.clamp(0.0, 3.0);
            let line = 180.0 * edge_support.clamp(0.0, 2.0);
            let score = corner.max(line);
            if score < 64.0 {
                continue;
            }
            let dx = x as f64 - ellipse.center.0;
            let dy = y as f64 - ellipse.center.1;
            let local_x = cosine * dx + sine * dy;
            let local_y = -sine * dx + cosine * dy;
            let phase = (local_y / ellipse.minor.max(1.0))
                .atan2(local_x / ellipse.major.max(1.0))
                .rem_euclid(std::f64::consts::TAU);
            let sector = ((phase / std::f64::consts::TAU * ANGULAR_SECTORS as f64).floor()
                as usize)
                .min(ANGULAR_SECTORS - 1);
            let tile = (y / FEATURE_SEED_TILE_SIZE) * tile_columns + x / FEATURE_SEED_TILE_SIZE;
            candidates.push((point, score, tile, sector));
        }
    }
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    let mut tile_counts = vec![0u8; tile_columns * tile_rows];
    let mut sector_counts = [0u8; ANGULAR_SECTORS];
    let mut selected = Vec::<([f32; 2], f32)>::new();
    for (point, score, tile, sector) in candidates {
        if selected.len() >= wanted {
            break;
        }
        if tile_counts[tile] >= 3 || sector_counts[sector] >= MAX_PER_SECTOR {
            continue;
        }
        if existing
            .iter()
            .chain(selected.iter().map(|item| &item.0))
            .all(|other| (other[0] - point[0]).hypot(other[1] - point[1]) >= MIN_FEATURE_SEPARATION)
        {
            tile_counts[tile] = tile_counts[tile].saturating_add(1);
            sector_counts[sector] = sector_counts[sector].saturating_add(1);
            selected.push((point, score));
        }
    }
    selected
}

impl BoundedIrisCannyTracker {
    fn reset(&mut self) {
        self.previous = None;
        self.tracks.clear();
        self.stable_frames = 0;
        self.coupled_kinematics.clear();
    }

    pub fn clear(&mut self) {
        self.reset();
    }

    pub fn observe(
        &mut self,
        pixels: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        seed: Option<IrisEllipseSeed>,
    ) -> MotionOctreeOverlay {
        if width < 24 || height < 20 || pixels.len() < width.saturating_mul(height) {
            self.reset();
            return MotionOctreeOverlay::default();
        }
        let current = RawFrame {
            sensor_x,
            sensor_y,
            width,
            height,
            pixels: cfa_neutral_raw(pixels, width, height),
        };
        let mut canny = canny_field(&current);
        let mut edges = edge_evidence(&mut canny, width, height).edges;
        let Some(seed) = seed else {
            self.reset();
            return MotionOctreeOverlay {
                edges,
                edge_high_threshold: canny.high_threshold,
                ..MotionOctreeOverlay::default()
            };
        };
        if !seed.center.0.is_finite()
            || !seed.center.1.is_finite()
            || !seed.major_radius.is_finite()
            || !seed.minor_radius.is_finite()
            || seed.major_radius < 12.0
            || seed.minor_radius < 8.0
        {
            self.reset();
            return MotionOctreeOverlay {
                edges,
                edge_high_threshold: canny.high_threshold,
                ..MotionOctreeOverlay::default()
            };
        }

        let incompatible = self
            .previous
            .as_ref()
            .is_some_and(|previous| previous.width != width || previous.height != height);
        if incompatible {
            self.reset();
        }

        let mut accepted = Vec::<BoundedIrisMatch>::new();
        if let Some(previous) = self.previous.as_ref() {
            for (track_index, track) in self.tracks.iter().enumerate() {
                let Some(last) = track.points.back().copied() else {
                    continue;
                };
                let previous_local = [
                    last[0] - previous.sensor_x as f32,
                    last[1] - previous.sensor_y as f32,
                ];
                let predicted = [last[0] - sensor_x as f32, last[1] - sensor_y as f32];
                let patch_inside = |point: [f32; 2], frame_width: usize, frame_height: usize| {
                    let x = point[0].round() as i32;
                    let y = point[1].round() as i32;
                    x >= BOUNDED_IRIS_PATCH_RADIUS
                        && y >= BOUNDED_IRIS_PATCH_RADIUS
                        && x + BOUNDED_IRIS_PATCH_RADIUS < frame_width as i32
                        && y + BOUNDED_IRIS_PATCH_RADIUS < frame_height as i32
                };
                if !patch_inside(previous_local, previous.width, previous.height)
                    || !bounded_iris_contains(seed, predicted, (0.24, 0.98))
                {
                    continue;
                }
                let mut candidates = Vec::with_capacity(
                    ((2 * BOUNDED_IRIS_SEARCH_RADIUS + 1) * (2 * BOUNDED_IRIS_SEARCH_RADIUS + 1))
                        as usize,
                );
                for dy in -BOUNDED_IRIS_SEARCH_RADIUS..=BOUNDED_IRIS_SEARCH_RADIUS {
                    for dx in -BOUNDED_IRIS_SEARCH_RADIUS..=BOUNDED_IRIS_SEARCH_RADIUS {
                        let candidate = [predicted[0] + dx as f32, predicted[1] + dy as f32];
                        if !patch_inside(candidate, width, height)
                            || !bounded_iris_contains(seed, candidate, (0.27, 0.95))
                        {
                            continue;
                        }
                        let candidate_x = candidate[0].round() as usize;
                        let candidate_y = candidate[1].round() as usize;
                        if local_canny_support(&canny, width, candidate_x, candidate_y) < 0.08 {
                            continue;
                        }
                        let cost = patch_cost_with_radius(
                            previous,
                            &current,
                            previous_local,
                            candidate,
                            BOUNDED_IRIS_PATCH_RADIUS,
                        );
                        if cost.is_finite() {
                            candidates.push((cost, candidate));
                        }
                    }
                }
                candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
                let Some((best_cost, best_point)) = candidates.first().copied() else {
                    continue;
                };
                if best_cost > 0.62 {
                    continue;
                }
                let second_cost = candidates
                    .iter()
                    .find(|(_, point)| {
                        (point[0] - best_point[0]).hypot(point[1] - best_point[1]) >= 3.0
                    })
                    .map_or(f32::INFINITY, |candidate| candidate.0);
                if second_cost.is_finite()
                    && (second_cost - best_cost) / second_cost.max(1.0e-5) < 0.015
                {
                    continue;
                }
                // Native-resolution forward/backward consistency rejects a
                // repeated skin ridge that merely has a similar local patch.
                let backward = (-2..=2)
                    .flat_map(|back_y| (-2..=2).map(move |back_x| (back_x, back_y)))
                    .filter_map(|(back_x, back_y)| {
                        let candidate = [
                            previous_local[0] + back_x as f32,
                            previous_local[1] + back_y as f32,
                        ];
                        let cost = patch_cost_with_radius(
                            &current,
                            previous,
                            best_point,
                            candidate,
                            BOUNDED_IRIS_PATCH_RADIUS,
                        );
                        cost.is_finite().then_some((cost, candidate))
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0));
                if !backward.is_some_and(|(cost, point)| {
                    cost <= 0.62
                        && (point[0] - previous_local[0]).hypot(point[1] - previous_local[1]) <= 1.5
                }) {
                    continue;
                }
                accepted.push(BoundedIrisMatch {
                    track: track_index,
                    current: [
                        best_point[0] + sensor_x as f32,
                        best_point[1] + sensor_y as f32,
                    ],
                    cost: best_cost,
                });
            }
        }

        // One physical patch may be the best destination for adjacent edge
        // seeds. Keep the strongest unique assignments before estimating the
        // common iris motion.
        accepted.sort_by(|left, right| left.cost.total_cmp(&right.cost));
        let mut unique = Vec::<BoundedIrisMatch>::with_capacity(accepted.len());
        for candidate in accepted {
            if unique.iter().all(|kept| {
                (kept.current[0] - candidate.current[0])
                    .hypot(kept.current[1] - candidate.current[1])
                    >= 4.0
            }) {
                unique.push(candidate);
            }
        }
        unique.sort_by_key(|item| item.track);

        let mut translations_x = unique
            .iter()
            .filter_map(|item| {
                let last = self.tracks.get(item.track)?.points.back()?;
                Some(item.current[0] - last[0])
            })
            .collect::<Vec<_>>();
        let mut translations_y = unique
            .iter()
            .filter_map(|item| {
                let last = self.tracks.get(item.track)?.points.back()?;
                Some(item.current[1] - last[1])
            })
            .collect::<Vec<_>>();
        let common_motion = [median(&mut translations_x), median(&mut translations_y)];
        if unique.len() >= 3 {
            unique.retain(|item| {
                let Some(last) = self
                    .tracks
                    .get(item.track)
                    .and_then(|track| track.points.back())
                else {
                    return false;
                };
                (item.current[0] - last[0] - common_motion[0])
                    .hypot(item.current[1] - last[1] - common_motion[1])
                    <= 3.5
            });
        }
        let similarity_pairs = unique
            .iter()
            .filter_map(|item| {
                let previous = *self.tracks.get(item.track)?.points.back()?;
                Some((previous, item.current))
            })
            .collect::<Vec<_>>();
        let similarity = bounded_iris_similarity_motion(
            &similarity_pairs,
            [
                sensor_x as f32 + seed.center.0 as f32,
                sensor_y as f32 + seed.center.1 as f32,
            ],
        );
        // Scale needs wider angular leverage than translation. Keep it as
        // telemetry unless at least six independently matched native patches
        // support a low-residual, physically small inter-frame similarity.
        let similarity_scale_reliable = similarity.support >= 6
            && similarity.residual.is_finite()
            && similarity.residual <= 2.0
            && similarity.rotation.is_finite()
            && similarity.rotation.abs() <= 0.08
            && similarity.scale_delta.is_finite()
            && similarity.scale_delta.abs() <= 0.08;
        let motion_residual = if unique.is_empty() {
            0.0
        } else {
            unique
                .iter()
                .filter_map(|item| {
                    let last = self.tracks.get(item.track)?.points.back()?;
                    Some(
                        (item.current[0] - last[0] - common_motion[0])
                            .hypot(item.current[1] - last[1] - common_motion[1]),
                    )
                })
                .sum::<f32>()
                / unique.len() as f32
        };
        let iris_motion_for_edges = SimilarityMotion {
            translation: common_motion,
            rotation: if similarity_scale_reliable {
                similarity.rotation
            } else {
                0.0
            },
            scale_delta: if similarity_scale_reliable {
                similarity.scale_delta
            } else {
                0.0
            },
            residual: if similarity_scale_reliable {
                similarity.residual
            } else {
                motion_residual
            },
            support: similarity_pairs.len(),
        };
        let motion_shadow_edges_downweighted = self.previous.as_ref().map_or(0, |previous| {
            condition_upper_edges_by_iris_motion(
                &mut edges,
                previous,
                &current,
                seed,
                iris_motion_for_edges,
                [
                    sensor_x as f32 + seed.center.0 as f32,
                    sensor_y as f32 + seed.center.1 as f32,
                ],
            )
        });

        let mut seen = vec![false; self.tracks.len()];
        for item in &unique {
            let track = &mut self.tracks[item.track];
            track.points.push_back(item.current);
            while track.points.len() > MAX_TRAIL {
                track.points.pop_front();
            }
            track.score = (1.0 - item.cost).clamp(0.0, 1.0);
            track.age = 0;
            track.matched_streak = track.matched_streak.saturating_add(1);
            seen[item.track] = true;
        }
        for (index, track) in self.tracks.iter_mut().enumerate() {
            if !seen.get(index).copied().unwrap_or(false) {
                track.age = track.age.saturating_add(1);
                track.matched_streak = 0;
            }
        }
        self.tracks
            .retain(|track| track.age <= BOUNDED_IRIS_MAX_AGE);

        let existing = self
            .tracks
            .iter()
            .filter_map(|track| track.points.back())
            .map(|point| [point[0] - sensor_x as f32, point[1] - sensor_y as f32])
            .collect::<Vec<_>>();
        let wanted = BOUNDED_IRIS_FEATURES.saturating_sub(self.tracks.len());
        for (point, score) in bounded_iris_seed_points(&current, &canny, seed, &existing, wanted) {
            self.tracks.push(BoundedIrisTrack {
                id: self.next_id,
                points: VecDeque::from([[point[0] + sensor_x as f32, point[1] + sensor_y as f32]]),
                score,
                age: 0,
                matched_streak: 0,
            });
            self.next_id = self.next_id.saturating_add(1);
        }

        let persistent = self
            .tracks
            .iter()
            .filter(|track| track.points.len() >= 2 && track.matched_streak >= 1)
            .count();
        self.stable_frames = if persistent >= 3 {
            self.stable_frames.saturating_add(1)
        } else {
            0
        };
        let trails = self
            .tracks
            .iter()
            .filter(|track| track.points.len() >= 2 && track.matched_streak >= 1)
            .map(|track| OverlayTrail {
                id: track.id,
                object: PUPIL_LAYER,
                match_score: track.score,
                matched_streak: track.matched_streak,
                layer_evidence: true,
                normal_flow_evidence: false,
                specularity: 0.0,
                assignment_confidence: track.score,
                motion_ema: common_motion,
                motion_variance: 0.0,
                residual_history: Vec::new(),
                points: track
                    .points
                    .iter()
                    .map(|point| TrailPoint {
                        x: point[0] - sensor_x as f32,
                        y: point[1] - sensor_y as f32,
                        z: 0.0,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let provisional_features = self
            .tracks
            .iter()
            .filter(|track| track.points.len() == 1)
            .filter_map(|track| track.points.back())
            .map(|point| (point[0] - sensor_x as f32, point[1] - sensor_y as f32))
            .collect::<Vec<_>>();
        let (centroid_sum, centroid_count, signature_samples) = self
            .tracks
            .iter()
            .filter(|track| track.points.len() >= 2 && track.matched_streak >= 1)
            .filter_map(|track| {
                track.points.back().map(|point| {
                    (
                        [point[0] - sensor_x as f32, point[1] - sensor_y as f32],
                        track.points.len(),
                    )
                })
            })
            .fold(
                ([0.0f32; 2], 0usize, usize::MAX),
                |(mut sum, count, minimum_samples), (point, samples)| {
                    sum[0] += point[0];
                    sum[1] += point[1];
                    (sum, count + 1, minimum_samples.min(samples))
                },
            );
        let pupil_centroid = if centroid_count == 0 {
            [seed.center.0 as f32, seed.center.1 as f32]
        } else {
            [
                centroid_sum[0] / centroid_count as f32,
                centroid_sum[1] / centroid_count as f32,
            ]
        };
        let mut motions = [SimilarityMotion::default(); OBJECTS];
        motions[PUPIL_LAYER] = iris_motion_for_edges;
        let mut layers = [MotionLayerStatus::default(); OBJECTS];
        layers[PUPIL_LAYER] = MotionLayerStatus {
            centroid: pupil_centroid,
            coherence: (-motion_residual / 2.5).exp(),
            trajectory_error: motion_residual,
            signature_samples: if centroid_count == 0 {
                0
            } else {
                signature_samples
            },
            persistent_tracks: persistent,
            stable_frames: self.stable_frames,
            ..MotionLayerStatus::default()
        };
        self.generation = self.generation.saturating_add(1);
        self.previous = Some(current);
        MotionOctreeOverlay {
            generation: self.generation,
            trails,
            motions,
            layers,
            active_objects: usize::from(persistent >= 3),
            matched_features: persistent,
            provisional_features,
            edges,
            edge_high_threshold: canny.high_threshold,
            motion_shadow_edges_downweighted,
            semantic_iris: Some(seed),
            ..MotionOctreeOverlay::default()
        }
    }

    /// Attach the bounded pupil/iris motion to the independent whole-ROI
    /// material frame and derive relative velocity, acceleration, and jerk.
    /// This is a state-only operation over two sparse similarity fits; it
    /// performs no additional image sampling and never changes Canny/limbus
    /// geometry or admits anatomy.
    #[allow(clippy::too_many_arguments)]
    pub fn fuse_global_similarity_at(
        &mut self,
        overlay: &mut MotionOctreeOverlay,
        global: NativeGlobalSimilarityEvidence,
        timestamp_ns: u64,
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
    ) {
        let current_frame_center = [
            sensor_x as f32 + width as f32 * 0.5,
            sensor_y as f32 + height as f32 * 0.5,
        ];
        let analysis_center = if global.motion_center_sensor[0].is_finite()
            && global.motion_center_sensor[1].is_finite()
            && global.motion_center_sensor != [0.0; 2]
        {
            global.motion_center_sensor
        } else {
            current_frame_center
        };
        let mut motions = overlay.motions;
        let mut layers = overlay.layers;
        motions[GENERAL_LAYER] = if global.reliable {
            global.motion
        } else {
            SimilarityMotion::default()
        };
        layers[GENERAL_LAYER] = if global.reliable {
            MotionLayerStatus {
                // A virtual material point transported by the authorized
                // global transform. It is absolute here because CoupledEye-
                // Kinematics operates in sensor coordinates.
                centroid: [
                    analysis_center[0] + global.motion.translation[0],
                    analysis_center[1] + global.motion.translation[1],
                ],
                coherence: (-global.motion.residual.max(0.0) / 3.2).exp(),
                trajectory_error: global.motion.residual.max(0.0),
                signature_samples: usize::from(global.stable_frames).saturating_add(1),
                persistent_tracks: global.motion.support,
                stable_frames: global.stable_frames,
                ..MotionLayerStatus::default()
            }
        } else {
            MotionLayerStatus::default()
        };

        let semantic_iris = overlay.semantic_iris;
        if let Some(seed) = semantic_iris {
            let pupil_motion_center = [
                sensor_x as f32 + seed.center.0 as f32,
                sensor_y as f32 + seed.center.1 as f32,
            ];
            motions[PUPIL_LAYER] = reanchor_similarity_motion(
                motions[PUPIL_LAYER],
                pupil_motion_center,
                analysis_center,
            );
        }
        if layers[PUPIL_LAYER].persistent_tracks != 0 {
            layers[PUPIL_LAYER].centroid[0] += sensor_x as f32;
            layers[PUPIL_LAYER].centroid[1] += sensor_y as f32;
        }
        let semantic_layers = global.reliable
            && semantic_iris.is_some()
            && layers[PUPIL_LAYER].persistent_tracks >= 3
            && motions[PUPIL_LAYER].support >= 3;
        let iris_geometry = semantic_iris.map(|seed| ProjectedIrisGeometry {
            center: [
                sensor_x as f64 + seed.center.0,
                sensor_y as f64 + seed.center.1,
            ],
            major_radius: seed.major_radius,
            minor_radius: seed.minor_radius,
            angle_rad: seed.angle,
            confidence: (layers[PUPIL_LAYER].coherence as f64
                * (layers[PUPIL_LAYER].persistent_tracks as f64 / 6.0).clamp(0.0, 1.0))
            .clamp(0.0, 1.0),
            anatomy_authorized: semantic_layers && layers[PUPIL_LAYER].stable_frames >= 2,
        });
        let mut coupled = self.coupled_kinematics.observe(
            timestamp_ns,
            analysis_center,
            width.max(height) as f64,
            motions,
            layers,
            semantic_layers,
            iris_geometry,
        );
        // CoupledEyeKinematics retains its rotation-center posterior across a
        // temporary cyan dropout. Dynamic derivatives, however, are not
        // current-frame evidence and must never authorize a saccade then.
        if !global.reliable {
            coupled.cyan = KinematicDerivatives::default();
            coupled.green = KinematicDerivatives::default();
            coupled.green_relative_to_cyan = KinematicDerivatives::default();
            coupled.saccade_likelihood = 0.0;
            coupled.micro_motion_likelihood = 0.0;
        }
        overlay.coupled_motion = coupled.translated(sensor_x as f32, sensor_y as f32);
        overlay.motions[GENERAL_LAYER] = motions[GENERAL_LAYER];
        let mut local_general = layers[GENERAL_LAYER];
        if local_general.persistent_tracks != 0 {
            local_general.centroid[0] -= sensor_x as f32;
            local_general.centroid[1] -= sensor_y as f32;
        }
        overlay.layers[GENERAL_LAYER] = local_general;
        overlay.active_objects = usize::from(
            overlay.layers[PUPIL_LAYER].persistent_tracks >= 3
                && overlay.layers[PUPIL_LAYER].stable_frames >= 1,
        ) + usize::from(
            global.reliable
                && overlay.layers[GENERAL_LAYER].persistent_tracks >= 3
                && overlay.layers[GENERAL_LAYER].stable_frames >= 1,
        );
    }
}

#[derive(Default)]
pub struct FourMotionOctrees {
    previous: Option<RawFrame>,
    canny_features: bool,
    learning_canny_profile: LearningCannyProfile,
    tracks: Vec<FeatureTrack>,
    /// Several native-RAW observations per persistent feature ID. The bank is
    /// separate from `previous`: it is expressly for identity recovery across
    /// a missed frame, while ordinary motion remains adjacent-frame evidence.
    nautilus_banks: BTreeMap<u64, NautilusFingerprintBank>,
    /// Current adjacent-frame regional horizontal light-field evidence.  It
    /// is recomputed from the exact RAW pair and never survives a cadence or
    /// geometry reset as if it were a new observation.
    horizontal_light_field: HorizontalLightFieldStatus,
    /// Previous projected iris region and current native-RAW radial
    /// transitions. They remain separate from point-ID tracks because a
    /// smooth limbus constrains only motion along its normal.
    radial_limbus_region: Option<EyeMotionRegion>,
    radial_limbus_flows: Vec<RadialLimbusFlow>,
    motions: [SimilarityMotion; OBJECTS],
    layers: [MotionLayerStatus; OBJECTS],
    layer_signatures: [LayerMotionSignature; OBJECTS],
    /// Persistent pairwise feature-relation graph. This is deliberately kept
    /// beside, but independent from, the four semantic motion slots: graph
    /// components are physical evidence used to authorize those names.
    motion_relations: PersistentMotionRelationGraph,
    /// Exact component identity retained across brief visibility gaps so an
    /// eyelash/lid cohort cannot inherit the iris semantic slot.
    relation_iris_identity: PersistentRelationIrisIdentity,
    parallax_axis: [f32; 2],
    /// Absolute sensor-space center of the persistent specular cluster. This
    /// anchors the label-free pupil/iris region when no anatomical seed is
    /// supplied by the live viewer.
    semantic_eye_center: Option<[f32; 2]>,
    /// Absolute sensor-space outer-iris region. Unlike the glint center above,
    /// this retains ellipse aspect and orientation so eyelid edges above and
    /// below the iris are not folded into the pupil motion thesis.
    semantic_eye_region: Option<EyeMotionRegion>,
    generation: u64,
    next_id: u64,
    match_diagnostics: MatchDiagnostics,
    focus_sfm: ProspectiveFocusSfm,
    focus_sweep_seen: bool,
    last_stable_focus: Option<u16>,
    // Offline A/B control. Live/default operation always uses the bounded
    // predictor; replay can force the legacy exhaustive corridor to quantify
    // speed and stability against identical RAW frames.
    exhaustive_search_for_replay: bool,
    /// Offline oracle control. When set, strict adjacent-frame ZNCC tracks can
    /// be harvested as an independent correspondence reference for evaluating
    /// nautilus retrieval without letting retrieval label its own answers.
    disable_nautilus_for_replay: bool,
    coupled_kinematics: CoupledEyeKinematics,
    fallback_timestamp_ns: u64,
    /// Timestamp of the RAW allocation backing `previous`. A motion learned
    /// over one exposure interval is never extrapolated across a capture gap;
    /// the matcher falls back to its complete native search corridor instead.
    previous_timestamp_ns: Option<u64>,
}

#[derive(Default)]
struct ProspectiveFocusSfm {
    status: FocusSfmStatus,
    train_frames: usize,
    test_frames: usize,
    training: Vec<(f32, f32)>,
    slope: f32,
    intercept: f32,
    baseline: f32,
    planar_error_sum: f32,
    depth_error_sum: f32,
}

#[derive(Clone, Copy)]
struct Match {
    track_index: usize,
    previous: [f32; 2],
    current: [f32; 2],
    score: f32,
    object: usize,
    z: f32,
    assignment_margin: f32,
    layer_evidence: bool,
    normal_flow_evidence: bool,
    specularity: f32,
}

/// The pairwise differential of a 2-D similarity transform.  For two
/// correspondences `(p_i -> q_i, p_j -> q_j)`, subtracting their image
/// displacements cancels translation and leaves the 2x2 tensor
///
///     [ scale  -rotation ]
///     [ rotation  scale  ]
///
/// in native sensor coordinates.  Translation is recovered only after this
/// relative tensor is known.  This is the graph edge Rob was asking for: an
/// edge describes how two feature nodes are connected by translation,
/// rotation, and scale rather than merely comparing their velocity vectors.
#[derive(Clone, Copy, Debug, Default)]
struct PairwiseMotionTensor {
    motion: SimilarityMotion,
    support: usize,
    residual: f32,
    /// Hashed set of independently predicted track identities. Temporal
    /// coherence compares this set rather than demanding an implausibly
    /// constant per-frame angular velocity through a saccade.
    support_fingerprint: [u64; 4],
    /// Fixed point of the motion after robust whole-frame similarity is
    /// removed. It is meaningful only when local rotation/scale is large
    /// enough to condition the inverse.
    shared_origin: [f32; 2],
    origin_valid: bool,
}

#[derive(Clone, Copy, Debug)]
struct PersistentMotionRelationEdge {
    track_ids: (u64, u64),
    tensor: PairwiseMotionTensor,
    coherence: f32,
    shared_frames: u16,
    age: u8,
}

#[derive(Clone, Copy, Debug)]
struct FrameMotionRelationEdge {
    left_node: usize,
    right_node: usize,
    tensor: PairwiseMotionTensor,
    coherence: f32,
    support_continuity: f32,
    shared_frames: u16,
}

#[derive(Clone, Debug, Default)]
struct MotionRelationComponent {
    /// Indices into the current `matches` slice.
    members: Vec<usize>,
    /// Exact stable feature identities represented by `members`, sorted for a
    /// bounded allocation-free intersection with the previous iris cohort.
    track_ids: Vec<u64>,
    centroid: [f32; 2],
    coherence: f32,
    shared_origin: [f32; 2],
    origin_spread: f32,
    origin_valid: bool,
    persistent_edges: usize,
    persistent_nodes: usize,
}

#[derive(Clone, Debug, Default)]
struct MotionRelationFrame {
    node_match_indices: Vec<usize>,
    edges: Vec<FrameMotionRelationEdge>,
    components: Vec<MotionRelationComponent>,
    observed_iris_component: Option<usize>,
    selected_iris_component: Option<usize>,
    selected_identity_overlap: f32,
    selected_origin_consistent: bool,
    observed_motion_evidence: f32,
    selected_by_identity_carry: bool,
    identity_switch_rejections: usize,
    initial_origin_rejections: usize,
    iris_candidate_diagnostics: RelationIrisCandidateDiagnostics,
}

impl MotionRelationFrame {
    fn recurrent_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| edge.shared_frames >= 2)
            .count()
    }

    fn supported_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| edge.shared_frames >= 2 && edge.tensor.support >= MIN_LAYER_SUPPORT)
            .count()
    }

    fn precise_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| {
                edge.shared_frames >= 2
                    && edge.tensor.support >= MIN_LAYER_SUPPORT
                    && edge.tensor.residual <= RELATION_STRONG_EDGE_RESIDUAL
            })
            .count()
    }

    fn coherent_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| relation_edge_is_strong(edge))
            .count()
    }

    fn persistent_component_count(&self) -> usize {
        self.components
            .iter()
            .filter(|component| relation_component_is_persistent(component))
            .count()
    }

    fn maximum_component_persistence(&self) -> (usize, usize) {
        self.components.iter().fold((0, 0), |maximum, component| {
            (
                maximum.0.max(component.persistent_edges),
                maximum.1.max(component.persistent_nodes),
            )
        })
    }

    fn selected_iris(&self) -> Option<&MotionRelationComponent> {
        self.selected_iris_component
            .and_then(|index| self.components.get(index))
    }

    fn observed_iris(&self) -> Option<&MotionRelationComponent> {
        self.observed_iris_component
            .and_then(|index| self.components.get(index))
    }

    fn maximum_shared_frames(&self) -> u16 {
        self.edges
            .iter()
            .map(|edge| edge.shared_frames)
            .max()
            .unwrap_or(0)
    }

    fn maximum_coherence(&self) -> f32 {
        self.edges
            .iter()
            .map(|edge| edge.coherence)
            .max_by(f32::total_cmp)
            .unwrap_or(0.0)
    }

    fn mean_recurrent_quality(&self) -> (f32, f32, f32) {
        let mut count = 0usize;
        let mut coherence = 0.0f32;
        let mut residual = 0.0f32;
        let mut support_continuity = 0.0f32;
        for edge in self.edges.iter().filter(|edge| edge.shared_frames >= 2) {
            count += 1;
            coherence += edge.coherence;
            residual += edge.tensor.residual;
            support_continuity += edge.support_continuity;
        }
        if count == 0 {
            (0.0, 0.0, 0.0)
        } else {
            (
                coherence / count as f32,
                residual / count as f32,
                support_continuity / count as f32,
            )
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RelationIrisIdentityContinuity {
    compatible: bool,
    track_overlap: f32,
    centroid_step_radii: f32,
    origin_step_radii: Option<f32>,
    origin_consistent: bool,
}

impl Default for RelationIrisIdentityContinuity {
    fn default() -> Self {
        Self {
            compatible: true,
            track_overlap: 0.0,
            centroid_step_radii: 0.0,
            origin_step_radii: None,
            origin_consistent: true,
        }
    }
}

/// Identity posterior for the component authorized as iris material.  It
/// retains only bounded graph metadata; no image is copied or downsampled.
#[derive(Clone, Debug, Default)]
struct PersistentRelationIrisIdentity {
    track_ids: Vec<u64>,
    centroid: [f32; 2],
    shared_origin: [f32; 2],
    origin_valid: bool,
    age: u8,
    confirmations: u16,
    evidence: f32,
}

fn relation_iris_observation_evidence(differential: f32) -> f32 {
    let span =
        (RELATION_IRIS_FULL_DIFFERENTIAL_PX - RELATION_IRIS_MIN_DIFFERENTIAL_PX).max(f32::EPSILON);
    let strength = ((differential - RELATION_IRIS_MIN_DIFFERENTIAL_PX) / span).clamp(0.0, 1.0);
    RELATION_IRIS_MIN_OBSERVATION_EVIDENCE
        + (1.0 - RELATION_IRIS_MIN_OBSERVATION_EVIDENCE) * strength
}

fn relation_sorted_track_id_jaccard(left: &[u64], right: &[u64]) -> f32 {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut intersection = 0usize;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => {
                intersection += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    let union = left.len() + right.len() - intersection;
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

impl PersistentRelationIrisIdentity {
    fn active(&self) -> bool {
        !self.track_ids.is_empty() && self.age <= RELATION_IRIS_IDENTITY_MAX_AGE
    }

    fn confirmed(&self) -> bool {
        self.active()
            && self.confirmations >= RELATION_IRIS_IDENTITY_MIN_CONFIRMATIONS
            && self.evidence >= RELATION_IRIS_IDENTITY_MIN_EVIDENCE
    }

    fn continuity(
        &self,
        component: &MotionRelationComponent,
        eye_region: EyeMotionRegion,
    ) -> RelationIrisIdentityContinuity {
        if !self.active() {
            return RelationIrisIdentityContinuity::default();
        }
        let radius = eye_region.major.max(eye_region.minor).max(8.0);
        let track_overlap = relation_sorted_track_id_jaccard(&self.track_ids, &component.track_ids);
        let centroid_step_radii = (self.centroid[0] - component.centroid[0])
            .hypot(self.centroid[1] - component.centroid[1])
            / radius;
        let origin_step_radii = (self.origin_valid && component.origin_valid).then(|| {
            (self.shared_origin[0] - component.shared_origin[0])
                .hypot(self.shared_origin[1] - component.shared_origin[1])
                / radius
        });
        let origin_consistent = origin_step_radii
            .is_none_or(|step| step <= RELATION_IRIS_IDENTITY_MAX_ORIGIN_STEP_RADII);

        // Consecutive frames must share at least one exact material track.
        // Across a short dropout, a new track cohort may re-enter, but it must
        // agree spatially with both the iris region and the retained pivot.
        let compatible = if self.age <= 1 {
            track_overlap >= RELATION_IRIS_IDENTITY_MIN_OVERLAP
                && (origin_consistent || track_overlap >= RELATION_IRIS_IDENTITY_STRONG_OVERLAP)
        } else if track_overlap >= RELATION_IRIS_IDENTITY_MIN_OVERLAP {
            origin_consistent || track_overlap >= RELATION_IRIS_IDENTITY_STRONG_OVERLAP
        } else {
            centroid_step_radii <= RELATION_IRIS_IDENTITY_MAX_CENTROID_STEP_RADII
                && origin_consistent
        };
        RelationIrisIdentityContinuity {
            compatible,
            track_overlap,
            centroid_step_radii,
            origin_step_radii,
            origin_consistent,
        }
    }

    fn observe(
        &mut self,
        component: Option<&MotionRelationComponent>,
        continuity: RelationIrisIdentityContinuity,
        observation_evidence: f32,
    ) {
        let Some(component) = component else {
            if self.track_ids.is_empty() {
                return;
            }
            self.age = self.age.saturating_add(1);
            if self.age > RELATION_IRIS_IDENTITY_MAX_AGE {
                *self = Self::default();
            }
            return;
        };
        let continuing =
            self.active() && continuity.track_overlap >= RELATION_IRIS_IDENTITY_MIN_OVERLAP;
        self.confirmations = if continuing {
            self.confirmations.saturating_add(1)
        } else {
            1
        };
        self.evidence = if continuing {
            (self.evidence + observation_evidence).min(16.0)
        } else {
            observation_evidence
        };
        self.track_ids.clone_from(&component.track_ids);
        self.centroid = component.centroid;
        if component.origin_valid && continuity.origin_consistent {
            if self.origin_valid && continuing {
                self.shared_origin = [
                    0.72 * self.shared_origin[0] + 0.28 * component.shared_origin[0],
                    0.72 * self.shared_origin[1] + 0.28 * component.shared_origin[1],
                ];
            } else {
                self.shared_origin = component.shared_origin;
            }
            self.origin_valid = true;
        } else if !continuing {
            self.origin_valid = false;
        }
        self.age = 0;
    }
}

fn relation_component_is_persistent(component: &MotionRelationComponent) -> bool {
    component.members.len() >= RELATION_MIN_COMPONENT_SUPPORT
        && component.persistent_edges >= 2
        && component.persistent_nodes >= RELATION_MIN_COMPONENT_SUPPORT
}

fn relation_graph_is_informative(relations: &MotionRelationFrame) -> bool {
    relations.node_match_indices.len() >= MIN_LAYER_SUPPORT * 2
        && relations.persistent_component_count() >= 2
        && relations.coherent_edge_count() >= 4
}

fn relation_graph_has_persistent_component(relations: &MotionRelationFrame) -> bool {
    relations.node_match_indices.len() >= MIN_LAYER_SUPPORT
        && relations.persistent_component_count() >= 1
        && relations.coherent_edge_count() >= 2
}

fn maximum_persistent_component_differential(
    relations: &MotionRelationFrame,
    matches: &[Match],
    global: SimilarityMotion,
    center: [f32; 2],
) -> f32 {
    relations
        .components
        .iter()
        .filter(|component| relation_component_is_persistent(component))
        .map(|component| {
            component
                .members
                .iter()
                .map(|index| {
                    let residual = residual_motion(&matches[*index], global, center);
                    residual[0].hypot(residual[1])
                })
                .sum::<f32>()
                / component.members.len().max(1) as f32
        })
        .max_by(f32::total_cmp)
        .unwrap_or(0.0)
}

#[derive(Clone, Debug, Default)]
struct PersistentMotionRelationGraph {
    /// Sorted by `(low_track_id, high_track_id)` for allocation-light binary
    /// lookup on the next exposure.
    edges: Vec<PersistentMotionRelationEdge>,
}

/// Structure-of-arrays packing for SIMD tensor evaluation. The source
/// correspondences remain the native/full-resolution `Match` records; this is
/// only a bounded 4x80-float arithmetic view, not an image copy or resize.
#[derive(Clone, Debug, Default)]
struct MotionRelationNodes {
    match_indices: Vec<usize>,
    track_ids: Vec<u64>,
    previous_x: Vec<f32>,
    previous_y: Vec<f32>,
    current_x: Vec<f32>,
    current_y: Vec<f32>,
}

impl MotionRelationNodes {
    fn from_matches(match_indices: Vec<usize>, matches: &[Match]) -> Self {
        Self::from_matches_with_ids(match_indices, matches, |item| item.track_index as u64)
    }

    fn from_matches_and_tracks(
        match_indices: Vec<usize>,
        matches: &[Match],
        tracks: &[FeatureTrack],
    ) -> Self {
        Self::from_matches_with_ids(match_indices, matches, |item| tracks[item.track_index].id)
    }

    fn from_matches_with_ids(
        match_indices: Vec<usize>,
        matches: &[Match],
        mut track_id: impl FnMut(&Match) -> u64,
    ) -> Self {
        let mut nodes = Self {
            match_indices,
            ..Self::default()
        };
        nodes.track_ids.reserve(nodes.match_indices.len());
        nodes.previous_x.reserve(nodes.match_indices.len());
        nodes.previous_y.reserve(nodes.match_indices.len());
        nodes.current_x.reserve(nodes.match_indices.len());
        nodes.current_y.reserve(nodes.match_indices.len());
        for match_index in &nodes.match_indices {
            let item = &matches[*match_index];
            nodes.track_ids.push(track_id(item));
            nodes.previous_x.push(item.previous[0]);
            nodes.previous_y.push(item.previous[1]);
            nodes.current_x.push(item.current[0]);
            nodes.current_y.push(item.current[1]);
        }
        nodes
    }

    fn len(&self) -> usize {
        self.match_indices.len()
    }
}

#[derive(Clone, Copy, Debug)]
struct RelationTensorScore {
    support: usize,
    residual: f32,
    support_fingerprint: [u64; 4],
}

fn relation_fingerprint_insert(fingerprint: &mut [u64; 4], track_id: u64) {
    // SplitMix64 avalanche keeps monotonically allocated track IDs from
    // clustering in one word. One bit per ID leaves this 256-bit sketch sparse
    // enough for a useful approximate Jaccard score at <=80 graph nodes.
    let mut hash = track_id.wrapping_add(0x9e37_79b9_7f4a_7c15);
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= hash >> 31;
    let bit = (hash & 255) as usize;
    fingerprint[bit >> 6] |= 1u64 << (bit & 63);
}

fn relation_fingerprint_jaccard(left: [u64; 4], right: [u64; 4]) -> f32 {
    let intersection = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| (left & right).count_ones())
        .sum::<u32>();
    let union = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| (left | right).count_ones())
        .sum::<u32>();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn score_relation_tensor_scalar(
    motion: SimilarityMotion,
    center: [f32; 2],
    nodes: &MotionRelationNodes,
    radius: f32,
) -> RelationTensorScore {
    let mut support = 0usize;
    let mut squared_error = 0.0f32;
    let mut support_fingerprint = [0u64; 4];
    let radius_sq = radius * radius;
    for index in 0..nodes.len() {
        let x = nodes.previous_x[index] - center[0];
        let y = nodes.previous_y[index] - center[1];
        let predicted_x = nodes.previous_x[index] + motion.translation[0] + motion.scale_delta * x
            - motion.rotation * y;
        let predicted_y = nodes.previous_y[index]
            + motion.translation[1]
            + motion.rotation * x
            + motion.scale_delta * y;
        let dx = predicted_x - nodes.current_x[index];
        let dy = predicted_y - nodes.current_y[index];
        let error_sq = dx * dx + dy * dy;
        if error_sq <= radius_sq {
            support += 1;
            squared_error += error_sq;
            relation_fingerprint_insert(&mut support_fingerprint, nodes.track_ids[index]);
        }
    }
    let residual = if support == 0 {
        f32::INFINITY
    } else {
        (squared_error / support as f32).sqrt()
    };
    RelationTensorScore {
        support,
        residual,
        support_fingerprint,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn score_relation_tensor_avx2(
    motion: SimilarityMotion,
    center: [f32; 2],
    nodes: &MotionRelationNodes,
    radius: f32,
) -> RelationTensorScore {
    use std::arch::x86_64::*;

    let center_x = _mm256_set1_ps(center[0]);
    let center_y = _mm256_set1_ps(center[1]);
    let translation_x = _mm256_set1_ps(motion.translation[0]);
    let translation_y = _mm256_set1_ps(motion.translation[1]);
    let scale = _mm256_set1_ps(motion.scale_delta);
    let rotation = _mm256_set1_ps(motion.rotation);
    let radius_sq = _mm256_set1_ps(radius * radius);
    let mut squared_sum = _mm256_setzero_ps();
    let mut support = 0usize;
    let mut support_fingerprint = [0u64; 4];
    let simd_end = nodes.len() / 8 * 8;
    for index in (0..simd_end).step_by(8) {
        let previous_x = _mm256_loadu_ps(nodes.previous_x.as_ptr().add(index));
        let previous_y = _mm256_loadu_ps(nodes.previous_y.as_ptr().add(index));
        let current_x = _mm256_loadu_ps(nodes.current_x.as_ptr().add(index));
        let current_y = _mm256_loadu_ps(nodes.current_y.as_ptr().add(index));
        let x = _mm256_sub_ps(previous_x, center_x);
        let y = _mm256_sub_ps(previous_y, center_y);
        let predicted_x = _mm256_sub_ps(
            _mm256_add_ps(
                _mm256_add_ps(previous_x, translation_x),
                _mm256_mul_ps(scale, x),
            ),
            _mm256_mul_ps(rotation, y),
        );
        let predicted_y = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(previous_y, translation_y),
                _mm256_mul_ps(rotation, x),
            ),
            _mm256_mul_ps(scale, y),
        );
        let dx = _mm256_sub_ps(predicted_x, current_x);
        let dy = _mm256_sub_ps(predicted_y, current_y);
        let error_sq = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
        let inlier = _mm256_cmp_ps(error_sq, radius_sq, _CMP_LE_OQ);
        let inlier_mask = _mm256_movemask_ps(inlier) as u32;
        support += inlier_mask.count_ones() as usize;
        for lane in 0..8 {
            if inlier_mask & (1 << lane) != 0 {
                relation_fingerprint_insert(
                    &mut support_fingerprint,
                    nodes.track_ids[index + lane],
                );
            }
        }
        squared_sum = _mm256_add_ps(squared_sum, _mm256_and_ps(error_sq, inlier));
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), squared_sum);
    let mut squared_error = lanes.iter().sum::<f32>();
    let radius_sq_scalar = radius * radius;
    for index in simd_end..nodes.len() {
        let x = nodes.previous_x[index] - center[0];
        let y = nodes.previous_y[index] - center[1];
        let predicted_x = nodes.previous_x[index] + motion.translation[0] + motion.scale_delta * x
            - motion.rotation * y;
        let predicted_y = nodes.previous_y[index]
            + motion.translation[1]
            + motion.rotation * x
            + motion.scale_delta * y;
        let dx = predicted_x - nodes.current_x[index];
        let dy = predicted_y - nodes.current_y[index];
        let error_sq = dx * dx + dy * dy;
        if error_sq <= radius_sq_scalar {
            support += 1;
            squared_error += error_sq;
            relation_fingerprint_insert(&mut support_fingerprint, nodes.track_ids[index]);
        }
    }
    let residual = if support == 0 {
        f32::INFINITY
    } else {
        (squared_error / support as f32).sqrt()
    };
    RelationTensorScore {
        support,
        residual,
        support_fingerprint,
    }
}

fn score_relation_tensor(
    motion: SimilarityMotion,
    center: [f32; 2],
    nodes: &MotionRelationNodes,
    radius: f32,
) -> RelationTensorScore {
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime AVX2 detection guards the target-feature body; all
        // loads stay inside equal-length owned f32 arrays.
        return unsafe { score_relation_tensor_avx2(motion, center, nodes, radius) };
    }
    score_relation_tensor_scalar(motion, center, nodes, radius)
}

fn relation_tensor_squared_errors_scalar(
    motion: SimilarityMotion,
    center: [f32; 2],
    nodes: &MotionRelationNodes,
    squared_errors: &mut [f32],
) {
    for index in 0..nodes.len() {
        let x = nodes.previous_x[index] - center[0];
        let y = nodes.previous_y[index] - center[1];
        let predicted_x = nodes.previous_x[index] + motion.translation[0] + motion.scale_delta * x
            - motion.rotation * y;
        let predicted_y = nodes.previous_y[index]
            + motion.translation[1]
            + motion.rotation * x
            + motion.scale_delta * y;
        let dx = predicted_x - nodes.current_x[index];
        let dy = predicted_y - nodes.current_y[index];
        squared_errors[index] = dx * dx + dy * dy;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relation_tensor_squared_errors_avx2(
    motion: SimilarityMotion,
    center: [f32; 2],
    nodes: &MotionRelationNodes,
    squared_errors: &mut [f32],
) {
    use std::arch::x86_64::*;

    let center_x = _mm256_set1_ps(center[0]);
    let center_y = _mm256_set1_ps(center[1]);
    let translation_x = _mm256_set1_ps(motion.translation[0]);
    let translation_y = _mm256_set1_ps(motion.translation[1]);
    let scale = _mm256_set1_ps(motion.scale_delta);
    let rotation = _mm256_set1_ps(motion.rotation);
    let simd_end = nodes.len() / 8 * 8;
    for index in (0..simd_end).step_by(8) {
        let previous_x = _mm256_loadu_ps(nodes.previous_x.as_ptr().add(index));
        let previous_y = _mm256_loadu_ps(nodes.previous_y.as_ptr().add(index));
        let current_x = _mm256_loadu_ps(nodes.current_x.as_ptr().add(index));
        let current_y = _mm256_loadu_ps(nodes.current_y.as_ptr().add(index));
        let x = _mm256_sub_ps(previous_x, center_x);
        let y = _mm256_sub_ps(previous_y, center_y);
        let predicted_x = _mm256_sub_ps(
            _mm256_add_ps(
                _mm256_add_ps(previous_x, translation_x),
                _mm256_mul_ps(scale, x),
            ),
            _mm256_mul_ps(rotation, y),
        );
        let predicted_y = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(previous_y, translation_y),
                _mm256_mul_ps(rotation, x),
            ),
            _mm256_mul_ps(scale, y),
        );
        let dx = _mm256_sub_ps(predicted_x, current_x);
        let dy = _mm256_sub_ps(predicted_y, current_y);
        let error_sq = _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy));
        _mm256_storeu_ps(squared_errors.as_mut_ptr().add(index), error_sq);
    }
    for index in simd_end..nodes.len() {
        let x = nodes.previous_x[index] - center[0];
        let y = nodes.previous_y[index] - center[1];
        let predicted_x = nodes.previous_x[index] + motion.translation[0] + motion.scale_delta * x
            - motion.rotation * y;
        let predicted_y = nodes.previous_y[index]
            + motion.translation[1]
            + motion.rotation * x
            + motion.scale_delta * y;
        let dx = predicted_x - nodes.current_x[index];
        let dy = predicted_y - nodes.current_y[index];
        squared_errors[index] = dx * dx + dy * dy;
    }
}

fn relation_tensor_squared_errors(
    motion: SimilarityMotion,
    center: [f32; 2],
    nodes: &MotionRelationNodes,
    squared_errors: &mut [f32],
) {
    debug_assert_eq!(squared_errors.len(), nodes.len());
    #[cfg(target_arch = "x86_64")]
    if !canny_simd_disabled() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime AVX2 detection and equal-length arrays as above.
        unsafe { relation_tensor_squared_errors_avx2(motion, center, nodes, squared_errors) };
        return;
    }
    relation_tensor_squared_errors_scalar(motion, center, nodes, squared_errors);
}

fn relation_tensor_inlier_statistics(
    motion: SimilarityMotion,
    center: [f32; 2],
    nodes: &MotionRelationNodes,
    enabled: &[bool],
    radius: f32,
    squared_errors: &mut [f32],
) -> (usize, f32) {
    relation_tensor_squared_errors(motion, center, nodes, squared_errors);
    let radius_sq = radius * radius;
    let mut support = 0usize;
    let mut squared_error = 0.0f32;
    for (index, error_sq) in squared_errors.iter().copied().enumerate() {
        if enabled.get(index).copied().unwrap_or(false) && error_sq <= radius_sq {
            support += 1;
            squared_error += error_sq;
        }
    }
    let residual = if support == 0 {
        f32::INFINITY
    } else {
        (squared_error / support as f32).sqrt()
    };
    (support, residual)
}

fn pairwise_motion_tensor(
    left: &Match,
    right: &Match,
    nodes: &MotionRelationNodes,
    center: [f32; 2],
    global: SimilarityMotion,
) -> Option<PairwiseMotionTensor> {
    let baseline = [
        right.previous[0] - left.previous[0],
        right.previous[1] - left.previous[1],
    ];
    let baseline_sq = baseline[0] * baseline[0] + baseline[1] * baseline[1];
    let baseline_length = baseline_sq.sqrt();
    if !(RELATION_MIN_BASELINE..=RELATION_MAX_BASELINE).contains(&baseline_length) {
        return None;
    }
    let left_displacement = [
        left.current[0] - left.previous[0],
        left.current[1] - left.previous[1],
    ];
    let right_displacement = [
        right.current[0] - right.previous[0],
        right.current[1] - right.previous[1],
    ];
    let relative_displacement = [
        right_displacement[0] - left_displacement[0],
        right_displacement[1] - left_displacement[1],
    ];
    let inverse_baseline = 1.0 / baseline_sq.max(1.0e-6);
    let scale_delta = (baseline[0] * relative_displacement[0]
        + baseline[1] * relative_displacement[1])
        * inverse_baseline;
    let rotation = (baseline[0] * relative_displacement[1]
        - baseline[1] * relative_displacement[0])
        * inverse_baseline;
    if !scale_delta.is_finite() || !rotation.is_finite() {
        return None;
    }
    let left_about_center = [left.previous[0] - center[0], left.previous[1] - center[1]];
    let left_tensor_delta = [
        scale_delta * left_about_center[0] - rotation * left_about_center[1],
        rotation * left_about_center[0] + scale_delta * left_about_center[1],
    ];
    let motion = SimilarityMotion {
        translation: [
            left_displacement[0] - left_tensor_delta[0],
            left_displacement[1] - left_tensor_delta[1],
        ],
        rotation,
        scale_delta,
        residual: 0.0,
        support: 2,
    };

    // A pair hypothesis is an exact two-point construction. It earns graph
    // weight only by predicting independent nodes from the same frame.
    let relation_score = score_relation_tensor(motion, center, nodes, RELATION_INLIER_RADIUS);

    // Remove the robust carrier motion before asking whether this local
    // tensor has a finite common origin. Translation-only motion correctly
    // produces no origin instead of an arbitrary point at infinity.
    let left_residual = residual_motion(left, global, center);
    let right_residual = residual_motion(right, global, center);
    let relative_residual = [
        right_residual[0] - left_residual[0],
        right_residual[1] - left_residual[1],
    ];
    let local_scale = (baseline[0] * relative_residual[0] + baseline[1] * relative_residual[1])
        * inverse_baseline;
    let local_rotation = (baseline[0] * relative_residual[1] - baseline[1] * relative_residual[0])
        * inverse_baseline;
    let local_rate_sq = local_scale * local_scale + local_rotation * local_rotation;
    let mut shared_origin = [0.0f32; 2];
    let mut origin_valid = local_rate_sq.sqrt() >= RELATION_ORIGIN_MIN_RATE;
    if origin_valid {
        let local_tensor_delta = [
            local_scale * left_about_center[0] - local_rotation * left_about_center[1],
            local_rotation * left_about_center[0] + local_scale * left_about_center[1],
        ];
        let local_translation = [
            left_residual[0] - local_tensor_delta[0],
            left_residual[1] - local_tensor_delta[1],
        ];
        let inverse = 1.0 / local_rate_sq.max(1.0e-9);
        let origin_offset = [
            -(local_scale * local_translation[0] + local_rotation * local_translation[1]) * inverse,
            (local_rotation * local_translation[0] - local_scale * local_translation[1]) * inverse,
        ];
        shared_origin = [center[0] + origin_offset[0], center[1] + origin_offset[1]];
        origin_valid = shared_origin[0].is_finite()
            && shared_origin[1].is_finite()
            && origin_offset[0].hypot(origin_offset[1]) <= RELATION_MAX_BASELINE * 2.5;
    }
    Some(PairwiseMotionTensor {
        motion,
        support: relation_score.support,
        residual: relation_score.residual,
        support_fingerprint: relation_score.support_fingerprint,
        shared_origin,
        origin_valid,
    })
}

fn relation_edge_is_strong(edge: &FrameMotionRelationEdge) -> bool {
    edge.shared_frames >= 2
        && edge.tensor.support >= MIN_LAYER_SUPPORT
        && edge.tensor.residual <= RELATION_STRONG_EDGE_RESIDUAL
        && edge.coherence >= RELATION_STRONG_EDGE_COHERENCE
}

fn median_copy(mut values: Vec<f32>) -> f32 {
    median(&mut values)
}

impl PersistentMotionRelationGraph {
    fn clear(&mut self) {
        self.edges.clear();
    }

    fn observe(
        &mut self,
        matches: &[Match],
        tracks: &[FeatureTrack],
        center: [f32; 2],
        global: SimilarityMotion,
    ) -> MotionRelationFrame {
        let mut node_match_indices = matches
            .iter()
            .enumerate()
            .filter_map(|(match_index, item)| {
                let track = &tracks[item.track_index];
                (track.matched_streak >= RELATION_MIN_TRACK_STREAK
                    && item.score >= 0.20
                    && item.previous[0].is_finite()
                    && item.previous[1].is_finite()
                    && item.current[0].is_finite()
                    && item.current[1].is_finite())
                .then_some(match_index)
            })
            .collect::<Vec<_>>();
        node_match_indices.sort_by_key(|match_index| tracks[matches[*match_index].track_index].id);
        let nodes =
            MotionRelationNodes::from_matches_and_tracks(node_match_indices, matches, tracks);

        let pair_capacity = nodes.len().saturating_mul(nodes.len().saturating_sub(1)) / 2;
        let mut current_edges = Vec::with_capacity(pair_capacity);
        let mut next_persistent = Vec::with_capacity(pair_capacity);
        for left_node in 0..nodes.len() {
            for right_node in left_node + 1..nodes.len() {
                let left_index = nodes.match_indices[left_node];
                let right_index = nodes.match_indices[right_node];
                let left = &matches[left_index];
                let right = &matches[right_index];
                let Some(tensor) = pairwise_motion_tensor(left, right, &nodes, center, global)
                else {
                    continue;
                };
                let left_id = tracks[left.track_index].id;
                let right_id = tracks[right.track_index].id;
                let track_ids = if left_id <= right_id {
                    (left_id, right_id)
                } else {
                    (right_id, left_id)
                };
                let support_quality =
                    ((tensor.support.saturating_sub(2)) as f32 / 2.0).clamp(0.0, 1.0);
                let residual_quality = (-tensor.residual / 1.15).exp();
                let current_coherence = support_quality * residual_quality;
                let prior = self
                    .edges
                    .binary_search_by_key(&track_ids, |edge| edge.track_ids)
                    .ok()
                    .map(|index| self.edges[index]);
                let mut support_continuity = 0.0f32;
                let coherence = prior.map_or(current_coherence, |edge| {
                    // Acceleration is allowed, but a discontinuous pair
                    // tensor should not inherit full historical authority.
                    // Cohort identity is stronger evidence than constant
                    // angular velocity: a saccade can change the tensor
                    // sharply while the same iris texture points remain its
                    // independent inliers.
                    support_continuity = relation_fingerprint_jaccard(
                        edge.tensor.support_fingerprint,
                        tensor.support_fingerprint,
                    );
                    let parameter_jump = (edge.tensor.motion.translation[0]
                        - tensor.motion.translation[0])
                        .hypot(edge.tensor.motion.translation[1] - tensor.motion.translation[1])
                        + 36.0
                            * ((edge.tensor.motion.rotation - tensor.motion.rotation).abs()
                                + (edge.tensor.motion.scale_delta - tensor.motion.scale_delta)
                                    .abs());
                    let parameter_continuity = (-parameter_jump / 8.0).exp().clamp(0.20, 1.0);
                    let continuity = 0.85 * support_continuity + 0.15 * parameter_continuity;
                    let historical = edge.coherence * (0.65 + 0.35 * continuity);
                    let current = current_coherence * (0.50 + 0.50 * support_continuity);
                    0.30 * historical + 0.70 * current
                });
                let shared_frames = prior.map_or(1, |edge| {
                    if tensor.support >= MIN_LAYER_SUPPORT
                        && tensor.residual <= RELATION_INLIER_RADIUS
                    {
                        edge.shared_frames.saturating_add(1)
                    } else {
                        1
                    }
                });
                next_persistent.push(PersistentMotionRelationEdge {
                    track_ids,
                    tensor,
                    coherence,
                    shared_frames,
                    age: 0,
                });
                current_edges.push(FrameMotionRelationEdge {
                    left_node,
                    right_node,
                    tensor,
                    coherence,
                    support_continuity,
                    shared_frames,
                });
            }
        }

        // Keep a missing relation only while both underlying feature tracks
        // could still survive normal MAX_AGE aging. It cannot connect a
        // current component until observed again.
        for prior in &self.edges {
            if next_persistent
                .binary_search_by_key(&prior.track_ids, |edge| edge.track_ids)
                .is_err()
                && prior.age < RELATION_MAX_EDGE_AGE
            {
                let mut held = *prior;
                held.age = held.age.saturating_add(1);
                held.coherence *= 0.78;
                next_persistent.push(held);
            }
        }
        next_persistent.sort_by_key(|edge| edge.track_ids);
        next_persistent.dedup_by_key(|edge| edge.track_ids);
        self.edges = next_persistent;

        // Extract disjoint consensus components from the pair-hypothesis
        // graph. A single accidental cross-material edge cannot merge two
        // populations: it must beat the internally precise tensor for every
        // still-unassigned node it claims. This is robust-model selection on
        // the graph, not transitive thresholded union-find.
        let mut unassigned = vec![true; nodes.len()];
        let mut squared_errors = vec![0.0f32; nodes.len()];
        let mut components = Vec::new();
        while components.len() < OBJECTS {
            let mut best: Option<(bool, f32, f32, usize, usize)> = None;
            for (edge_index, edge) in current_edges.iter().enumerate() {
                if edge.tensor.support < MIN_LAYER_SUPPORT
                    || !unassigned.get(edge.left_node).copied().unwrap_or(false)
                    || !unassigned.get(edge.right_node).copied().unwrap_or(false)
                {
                    continue;
                }
                let (support, residual) = relation_tensor_inlier_statistics(
                    edge.tensor.motion,
                    center,
                    &nodes,
                    &unassigned,
                    RELATION_COMPONENT_INLIER_RADIUS,
                    &mut squared_errors,
                );
                let radius_sq = RELATION_COMPONENT_INLIER_RADIUS * RELATION_COMPONENT_INLIER_RADIUS;
                if support < RELATION_MIN_COMPONENT_SUPPORT
                    || squared_errors[edge.left_node] > radius_sq
                    || squared_errors[edge.right_node] > radius_sq
                {
                    continue;
                }
                // Residual precision intentionally dominates raw support.
                // Cross-object pairs often interpolate through several points
                // at ~1 px, while a real rigid cohort predicts fewer points
                // to a small fraction of a pixel.
                let precision = (-residual / 0.46).exp();
                let persistence = 0.78
                    + 0.14 * edge.coherence.clamp(0.0, 1.0)
                    + 0.08 * (edge.shared_frames as f32 / 4.0).clamp(0.0, 1.0);
                // A conditioned finite fixed point is information that a
                // translation-only velocity cluster cannot supply. Prefer a
                // slightly smaller rigid rotational cohort when its pair
                // tensor converges on such a point; otherwise one iris point
                // whose instantaneous velocity happens to equal the lashes
                // lets the larger translation cohort steal it.
                let shared_origin_information = if edge.tensor.origin_valid { 1.32 } else { 1.0 };
                let score = support as f32 * precision * persistence * shared_origin_information;
                let persistent_model = relation_edge_is_strong(edge);
                let replace = best.as_ref().is_none_or(|current| {
                    persistent_model > current.0
                        || (persistent_model == current.0
                            && (score > current.1
                                || (score == current.1
                                    && (residual < current.2
                                        || (residual == current.2 && support > current.4)))))
                });
                if replace {
                    best = Some((persistent_model, score, residual, edge_index, support));
                }
            }
            let Some((_, _, selected_residual, selected_edge_index, _)) = best else {
                break;
            };
            let selected_edge = &current_edges[selected_edge_index];
            relation_tensor_squared_errors(
                selected_edge.tensor.motion,
                center,
                &nodes,
                &mut squared_errors,
            );
            let radius_sq = RELATION_COMPONENT_INLIER_RADIUS * RELATION_COMPONENT_INLIER_RADIUS;
            let node_members = squared_errors
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, error_sq)| {
                    (unassigned[index] && error_sq <= radius_sq).then_some(index)
                })
                .collect::<Vec<_>>();
            let mut member_mask = vec![false; nodes.len()];
            for node in &node_members {
                member_mask[*node] = true;
            }
            let mut internal_edges = Vec::new();
            for edge in &current_edges {
                if !member_mask[edge.left_node] || !member_mask[edge.right_node] {
                    continue;
                }
                // `tensor.residual` is deliberately measured against every
                // nearby node, so another material moving differently can
                // make a perfectly rigid iris edge look globally noisy.
                // Re-evaluate the hypothesis against this component only.
                relation_tensor_squared_errors(
                    edge.tensor.motion,
                    center,
                    &nodes,
                    &mut squared_errors,
                );
                let explains_component = node_members.iter().all(|node| {
                    squared_errors[*node]
                        <= RELATION_COMPONENT_INLIER_RADIUS * RELATION_COMPONENT_INLIER_RADIUS
                });
                if explains_component {
                    internal_edges.push(edge);
                }
            }
            internal_edges.sort_by(|left, right| right.coherence.total_cmp(&left.coherence));
            let internal_coherence = if internal_edges.is_empty() {
                selected_edge.coherence
            } else {
                internal_edges
                    .iter()
                    .map(|edge| edge.coherence)
                    .sum::<f32>()
                    / internal_edges.len() as f32
            };
            let coherence = (0.62 * internal_coherence + 0.38 * (-selected_residual / 0.70).exp())
                .clamp(0.0, 1.0);
            let origin_edges = internal_edges
                .iter()
                .filter(|edge| edge.tensor.origin_valid)
                .collect::<Vec<_>>();
            let shared_origin = if origin_edges.is_empty() {
                [0.0; 2]
            } else {
                [
                    median_copy(
                        origin_edges
                            .iter()
                            .map(|edge| edge.tensor.shared_origin[0])
                            .collect(),
                    ),
                    median_copy(
                        origin_edges
                            .iter()
                            .map(|edge| edge.tensor.shared_origin[1])
                            .collect(),
                    ),
                ]
            };
            let origin_spread = if origin_edges.is_empty() {
                f32::INFINITY
            } else {
                median_copy(
                    origin_edges
                        .iter()
                        .map(|edge| {
                            (edge.tensor.shared_origin[0] - shared_origin[0])
                                .hypot(edge.tensor.shared_origin[1] - shared_origin[1])
                        })
                        .collect(),
                )
            };
            let minimum_origin_edges = node_members.len().saturating_sub(1).max(2);
            let origin_valid = origin_edges.len() >= minimum_origin_edges
                && origin_spread.is_finite()
                && origin_spread <= 28.0;
            let mut persistent_node_mask = vec![false; nodes.len()];
            let persistent_edges = internal_edges
                .iter()
                .filter(|edge| relation_edge_is_strong(edge))
                .inspect(|edge| {
                    persistent_node_mask[edge.left_node] = true;
                    persistent_node_mask[edge.right_node] = true;
                })
                .count();
            let persistent_nodes = node_members
                .iter()
                .filter(|node| persistent_node_mask[**node])
                .count();
            let mut track_ids = node_members
                .iter()
                .map(|node| nodes.track_ids[*node])
                .collect::<Vec<_>>();
            track_ids.sort_unstable();
            track_ids.dedup();
            let centroid_inverse = 1.0 / node_members.len().max(1) as f32;
            let centroid = [
                node_members
                    .iter()
                    .map(|node| nodes.current_x[*node])
                    .sum::<f32>()
                    * centroid_inverse,
                node_members
                    .iter()
                    .map(|node| nodes.current_y[*node])
                    .sum::<f32>()
                    * centroid_inverse,
            ];
            components.push(MotionRelationComponent {
                members: node_members
                    .iter()
                    .map(|node| nodes.match_indices[*node])
                    .collect(),
                track_ids,
                centroid,
                coherence,
                shared_origin,
                origin_spread,
                origin_valid,
                persistent_edges,
                persistent_nodes,
            });
            for node in node_members {
                unassigned[node] = false;
            }
        }
        components.sort_by(|left, right| {
            right
                .members
                .len()
                .cmp(&left.members.len())
                .then_with(|| right.coherence.total_cmp(&left.coherence))
        });
        MotionRelationFrame {
            node_match_indices: nodes.match_indices,
            edges: current_edges,
            components,
            observed_iris_component: None,
            selected_iris_component: None,
            selected_identity_overlap: 0.0,
            selected_origin_consistent: true,
            observed_motion_evidence: 0.0,
            selected_by_identity_carry: false,
            identity_switch_rejections: 0,
            initial_origin_rejections: 0,
            iris_candidate_diagnostics: RelationIrisCandidateDiagnostics::default(),
        }
    }
}

fn enforce_unique_match_destinations(matches: &mut Vec<Match>, track_priorities: &[f32]) -> usize {
    let candidate_count = matches.len();
    // Patch score remains the primary criterion. A small persistence bonus
    // makes a mature trajectory win a near-tie against a newly seeded track
    // without allowing track age to rescue a visibly poorer patch match.
    matches.sort_by(|left, right| {
        let left_priority = left.score
            + track_priorities
                .get(left.track_index)
                .copied()
                .unwrap_or(0.0);
        let right_priority = right.score
            + track_priorities
                .get(right.track_index)
                .copied()
                .unwrap_or(0.0);
        right_priority
            .total_cmp(&left_priority)
            .then_with(|| left.track_index.cmp(&right.track_index))
    });
    let mut unique = Vec::with_capacity(matches.len());
    for candidate in matches.drain(..) {
        if unique.iter().all(|accepted: &Match| {
            (accepted.current[0] - candidate.current[0])
                .hypot(accepted.current[1] - candidate.current[1])
                >= MIN_MATCH_DESTINATION_SEPARATION
        }) {
            unique.push(candidate);
        }
    }
    unique.sort_by_key(|item| item.track_index);
    let rejected = candidate_count - unique.len();
    *matches = unique;
    rejected
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

// Keep this sparse relative to the native ROI, but dense enough that a
// broadly rigid skin/eye cohort still has nine members after independently
// moving lid, iris, lash, and glasses features are rejected.  One corner per
// cell also prevents any single high-texture strip from monopolizing the
// evidence.  These are native-coordinate samples; no image is resized or
// materialized for this tracker.
const NATIVE_GLOBAL_FEATURE_COLUMNS: usize = 10;
const NATIVE_GLOBAL_FEATURE_ROWS: usize = 8;
const NATIVE_GLOBAL_SEARCH_RADIUS: i32 = 12;
const NATIVE_GLOBAL_PATCH_RADIUS: i32 = 4;
const NATIVE_GLOBAL_MIN_SUPPORT: usize = 9;

fn shared_native_neutral_sample(frame: &SharedNativeRawFrame, x: i32, y: i32) -> Option<f32> {
    if x < 2 || y < 2 || x + 2 >= frame.width as i32 || y + 2 >= frame.height as i32 {
        return None;
    }
    // One complete 4x4 Quad-Bayer carrier period, evaluated on demand at the
    // original pixel coordinate. This is a local statistic, not a resized or
    // materialized image.
    let mut sum = 0u32;
    for offset_y in -1..=2 {
        for offset_x in -1..=2 {
            sum += u32::from(
                frame.pixels[(y + offset_y) as usize * frame.width + (x + offset_x) as usize],
            );
        }
    }
    Some(sum as f32 / 16.0)
}

fn shared_native_corner_score(frame: &SharedNativeRawFrame, x: i32, y: i32) -> f32 {
    let mut xx = 0.0f32;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;
    for offset_y in -1..=1 {
        for offset_x in -1..=1 {
            let sample_x = x + offset_x;
            let sample_y = y + offset_y;
            let Some(gx) = shared_native_neutral_sample(frame, sample_x + 1, sample_y)
                .zip(shared_native_neutral_sample(frame, sample_x - 1, sample_y))
                .map(|(right, left)| right - left)
            else {
                return 0.0;
            };
            let Some(gy) = shared_native_neutral_sample(frame, sample_x, sample_y + 1)
                .zip(shared_native_neutral_sample(frame, sample_x, sample_y - 1))
                .map(|(bottom, top)| bottom - top)
            else {
                return 0.0;
            };
            xx += gx * gx;
            yy += gy * gy;
            xy += gx * gy;
        }
    }
    let trace = xx + yy;
    let determinant = (xx * yy - xy * xy).max(0.0);
    if trace <= 1.0 {
        0.0
    } else {
        determinant / trace
    }
}

fn shared_native_global_features(frame: &SharedNativeRawFrame) -> Vec<[f32; 2]> {
    if frame.width < 64 || frame.height < 48 || frame.pixels.len() < frame.width * frame.height {
        return Vec::new();
    }
    let margin = 10usize;
    let usable_width = frame.width.saturating_sub(2 * margin);
    let usable_height = frame.height.saturating_sub(2 * margin);
    let mut features =
        Vec::with_capacity(NATIVE_GLOBAL_FEATURE_COLUMNS * NATIVE_GLOBAL_FEATURE_ROWS);
    for row in 0..NATIVE_GLOBAL_FEATURE_ROWS {
        let top = margin + row * usable_height / NATIVE_GLOBAL_FEATURE_ROWS;
        let bottom = margin + (row + 1) * usable_height / NATIVE_GLOBAL_FEATURE_ROWS;
        for column in 0..NATIVE_GLOBAL_FEATURE_COLUMNS {
            let left = margin + column * usable_width / NATIVE_GLOBAL_FEATURE_COLUMNS;
            let right = margin + (column + 1) * usable_width / NATIVE_GLOBAL_FEATURE_COLUMNS;
            let mut best = None::<(f32, i32, i32)>;
            for y in (top..bottom).step_by(4) {
                for x in (left..right).step_by(4) {
                    let score = shared_native_corner_score(frame, x as i32, y as i32);
                    if score.is_finite() && best.is_none_or(|candidate| score > candidate.0) {
                        best = Some((score, x as i32, y as i32));
                    }
                }
            }
            if let Some((score, x, y)) = best.filter(|candidate| candidate.0 >= 18.0) {
                let _ = score;
                features.push([x as f32, y as f32]);
            }
        }
    }
    features
}

fn shared_native_patch_cost(
    previous: &SharedNativeRawFrame,
    current: &SharedNativeRawFrame,
    previous_point: [f32; 2],
    current_point: [f32; 2],
) -> f32 {
    let previous_x = previous_point[0].round() as i32;
    let previous_y = previous_point[1].round() as i32;
    let current_x = current_point[0].round() as i32;
    let current_y = current_point[1].round() as i32;
    let mut left_sum = 0.0f32;
    let mut right_sum = 0.0f32;
    let mut left_squared = 0.0f32;
    let mut right_squared = 0.0f32;
    let mut cross = 0.0f32;
    let mut count = 0.0f32;
    for offset_y in (-NATIVE_GLOBAL_PATCH_RADIUS..=NATIVE_GLOBAL_PATCH_RADIUS).step_by(2) {
        for offset_x in (-NATIVE_GLOBAL_PATCH_RADIUS..=NATIVE_GLOBAL_PATCH_RADIUS).step_by(2) {
            let (Some(left), Some(right)) = (
                shared_native_neutral_sample(
                    previous,
                    previous_x + offset_x,
                    previous_y + offset_y,
                ),
                shared_native_neutral_sample(current, current_x + offset_x, current_y + offset_y),
            ) else {
                return f32::INFINITY;
            };
            left_sum += left;
            right_sum += right;
            left_squared += left * left;
            right_squared += right * right;
            cross += left * right;
            count += 1.0;
        }
    }
    let left_energy = left_squared - left_sum * left_sum / count;
    let right_energy = right_squared - right_sum * right_sum / count;
    if left_energy < 48.0 || right_energy < 48.0 {
        return f32::INFINITY;
    }
    let covariance = cross - left_sum * right_sum / count;
    let correlation = (covariance / (left_energy * right_energy).sqrt().max(48.0)).clamp(-1.0, 1.0);
    (1.0 - correlation).max(0.0).sqrt()
}

fn shared_native_parabolic_patch_offset(negative: f32, center: f32, positive: f32) -> f32 {
    let curvature = negative - 2.0 * center + positive;
    if !negative.is_finite() || !center.is_finite() || !positive.is_finite() || curvature <= 1.0e-5
    {
        return 0.0;
    }
    (0.5 * (negative - positive) / curvature).clamp(-0.75, 0.75)
}

impl NativeGlobalSimilarityTracker {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn observe(
        &mut self,
        pixels: Arc<Vec<u16>>,
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
    ) -> NativeGlobalSimilarityEvidence {
        let current = SharedNativeRawFrame {
            sensor_x,
            sensor_y,
            width,
            height,
            pixels,
        };
        if width < 64 || height < 48 || current.pixels.len() < width.saturating_mul(height) {
            self.clear();
            return NativeGlobalSimilarityEvidence::default();
        }
        let Some(previous) = self.previous.replace(current.clone()) else {
            self.stable_frames = 0;
            return NativeGlobalSimilarityEvidence::default();
        };
        if previous.width != width || previous.height != height {
            self.stable_frames = 0;
            return NativeGlobalSimilarityEvidence::default();
        }

        let mut matches = Vec::<Match>::new();
        for (track_index, previous_local) in shared_native_global_features(&previous)
            .into_iter()
            .enumerate()
        {
            let previous_sensor = [
                previous_local[0] + previous.sensor_x as f32,
                previous_local[1] + previous.sensor_y as f32,
            ];
            let predicted = [
                previous_sensor[0] - current.sensor_x as f32,
                previous_sensor[1] - current.sensor_y as f32,
            ];
            let mut candidates = Vec::<(f32, [f32; 2])>::new();
            for delta_y in -NATIVE_GLOBAL_SEARCH_RADIUS..=NATIVE_GLOBAL_SEARCH_RADIUS {
                for delta_x in -NATIVE_GLOBAL_SEARCH_RADIUS..=NATIVE_GLOBAL_SEARCH_RADIUS {
                    let candidate = [predicted[0] + delta_x as f32, predicted[1] + delta_y as f32];
                    let cost =
                        shared_native_patch_cost(&previous, &current, previous_local, candidate);
                    if cost.is_finite() {
                        candidates.push((cost, candidate));
                    }
                }
            }
            candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
            let Some((best_cost, best_local)) = candidates.first().copied() else {
                continue;
            };
            let distinct_second = candidates
                .iter()
                .find(|candidate| {
                    (candidate.1[0] - best_local[0]).hypot(candidate.1[1] - best_local[1]) >= 3.0
                })
                .map_or(f32::INFINITY, |candidate| candidate.0);
            let margin = if distinct_second.is_finite() {
                (distinct_second - best_cost) / distinct_second.max(1.0e-5)
            } else {
                1.0
            };
            if best_cost > 0.58 || margin < 0.012 {
                continue;
            }
            // The RAW search itself stays on the sensor pixel lattice.  Fit a
            // separable parabola to the native patch-cost basin so the global
            // scale/rotation estimate is not quantized independently at every
            // corner.  No interpolated or resized image is constructed.
            let cost_at = |delta_x: f32, delta_y: f32| {
                shared_native_patch_cost(
                    &previous,
                    &current,
                    previous_local,
                    [best_local[0] + delta_x, best_local[1] + delta_y],
                )
            };
            let refined_local = [
                best_local[0]
                    + shared_native_parabolic_patch_offset(
                        cost_at(-1.0, 0.0),
                        best_cost,
                        cost_at(1.0, 0.0),
                    ),
                best_local[1]
                    + shared_native_parabolic_patch_offset(
                        cost_at(0.0, -1.0),
                        best_cost,
                        cost_at(0.0, 1.0),
                    ),
            ];
            // Native-resolution forward/backward identity check. Search only
            // the immediate source neighborhood: a repeated lid/glasses edge
            // may win forward matching, but should not return to this corner.
            let mut backward = (f32::INFINITY, [0.0f32; 2]);
            for delta_y in -2..=2 {
                for delta_x in -2..=2 {
                    let candidate = [
                        previous_local[0] + delta_x as f32,
                        previous_local[1] + delta_y as f32,
                    ];
                    let cost = shared_native_patch_cost(&current, &previous, best_local, candidate);
                    if cost < backward.0 {
                        backward = (cost, candidate);
                    }
                }
            }
            if backward.0 > 0.58
                || (backward.1[0] - previous_local[0]).hypot(backward.1[1] - previous_local[1])
                    > 1.5
            {
                continue;
            }
            matches.push(Match {
                track_index,
                previous: previous_sensor,
                current: [
                    refined_local[0] + current.sensor_x as f32,
                    refined_local[1] + current.sensor_y as f32,
                ],
                score: (1.0 - best_cost).clamp(0.0, 1.0),
                object: GENERAL_LAYER,
                z: 0.0,
                assignment_margin: margin,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            });
        }
        enforce_unique_match_destinations(&mut matches, &[]);
        let center = [
            previous.sensor_x as f32 + width as f32 * 0.5,
            previous.sensor_y as f32 + height as f32 * 0.5,
        ];
        let candidate_matches = matches.len();
        let (motion, inlier_indices) = shared_native_robust_global_similarity(&matches, center);
        let range = |axis: usize| {
            inlier_indices
                .iter()
                .map(|index| matches[*index].previous[axis])
                .fold((f32::INFINITY, f32::NEG_INFINITY), |range, value| {
                    (range.0.min(value), range.1.max(value))
                })
        };
        let x_range = range(0);
        let y_range = range(1);
        let spatial_span = [
            (x_range.1 - x_range.0).max(0.0),
            (y_range.1 - y_range.0).max(0.0),
        ];
        let mut quadrants = [false; 4];
        for index in &inlier_indices {
            let item = &matches[*index];
            let right = usize::from(item.previous[0] >= center[0]);
            let bottom = usize::from(item.previous[1] >= center[1]);
            quadrants[right | (bottom << 1)] = true;
        }
        let occupied_quadrants = quadrants.into_iter().filter(|occupied| *occupied).count();
        let reliable = motion.support >= NATIVE_GLOBAL_MIN_SUPPORT
            && spatial_span[0] >= width as f32 * 0.52
            && spatial_span[1] >= height as f32 * 0.42
            && occupied_quadrants >= 3
            && motion.residual.is_finite()
            && motion.residual <= 2.0
            && motion.translation[0].hypot(motion.translation[1]) <= 24.0
            && motion.rotation.is_finite()
            && motion.rotation.abs() <= 0.10
            && motion.scale_delta.is_finite()
            && motion.scale_delta.abs() <= 0.08;
        self.stable_frames = if reliable {
            self.stable_frames.saturating_add(1)
        } else {
            0
        };
        NativeGlobalSimilarityEvidence {
            motion: if reliable {
                motion
            } else {
                SimilarityMotion::default()
            },
            candidate_motion: motion,
            candidate_matches,
            reliable,
            stable_frames: self.stable_frames,
            spatial_span,
            occupied_quadrants,
            motion_center_sensor: center,
        }
    }
}

fn normalized_vector(vector: [f32; 2]) -> [f32; 2] {
    let length = vector[0].hypot(vector[1]);
    if length > 1.0e-6 {
        [vector[0] / length, vector[1] / length]
    } else {
        [0.0; 2]
    }
}

fn sample(frame: &RawFrame, x: i32, y: i32) -> Option<f32> {
    if x < 0 || y < 0 || x >= frame.width as i32 || y >= frame.height as i32 {
        None
    } else {
        Some(frame.pixels[y as usize * frame.width + x as usize] as f32)
    }
}

struct IntegralPatchMoments {
    stride: usize,
    width: usize,
    height: usize,
    sum: Vec<u64>,
    squared_sum: Vec<u64>,
}

impl IntegralPatchMoments {
    fn new(frame: &RawFrame) -> Self {
        let stride = frame.width + 1;
        let mut sum = vec![0u64; stride * (frame.height + 1)];
        let mut squared_sum = vec![0u64; stride * (frame.height + 1)];
        for y in 0..frame.height {
            let mut row_sum = 0u64;
            let mut row_squared_sum = 0u64;
            for x in 0..frame.width {
                let value = frame.pixels[y * frame.width + x] as u64;
                row_sum += value;
                row_squared_sum += value * value;
                let destination = (y + 1) * stride + x + 1;
                sum[destination] = sum[y * stride + x + 1] + row_sum;
                squared_sum[destination] = squared_sum[y * stride + x + 1] + row_squared_sum;
            }
        }
        Self {
            stride,
            width: frame.width,
            height: frame.height,
            sum,
            squared_sum,
        }
    }

    fn patch(&self, center: [f32; 2], radius: i32) -> Option<(u64, u64)> {
        let center_x = center[0].round() as i32;
        let center_y = center[1].round() as i32;
        let x0 = center_x - radius;
        let y0 = center_y - radius;
        let x1 = center_x + radius + 1;
        let y1 = center_y + radius + 1;
        if x0 < 0 || y0 < 0 || x1 > self.width as i32 || y1 > self.height as i32 {
            return None;
        }
        let (x0, y0, x1, y1) = (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
        let rectangle = |integral: &[u64]| {
            integral[y1 * self.stride + x1] + integral[y0 * self.stride + x0]
                - integral[y0 * self.stride + x1]
                - integral[y1 * self.stride + x0]
        };
        Some((rectangle(&self.sum), rectangle(&self.squared_sum)))
    }
}

fn patch_cost_with_integral_moments(
    previous: &RawFrame,
    current: &RawFrame,
    previous_moments: &IntegralPatchMoments,
    current_moments: &IntegralPatchMoments,
    a: [f32; 2],
    b: [f32; 2],
    radius: i32,
) -> f32 {
    let (Some((left_sum, left_squared)), Some((right_sum, right_squared))) = (
        previous_moments.patch(a, radius),
        current_moments.patch(b, radius),
    ) else {
        return f32::INFINITY;
    };
    let ax = a[0].round() as i32;
    let ay = a[1].round() as i32;
    let bx = b[0].round() as i32;
    let by = b[1].round() as i32;
    let mut cross = 0u64;
    let diameter = (2 * radius + 1) as usize;
    for dy in -radius..=radius {
        let previous_row = (ay + dy) as usize * previous.width;
        let current_row = (by + dy) as usize * current.width;
        let previous_start = previous_row + (ax - radius) as usize;
        let current_start = current_row + (bx - radius) as usize;
        let previous_patch = &previous.pixels[previous_start..previous_start + diameter];
        let current_patch = &current.pixels[current_start..current_start + diameter];
        cross += previous_patch
            .iter()
            .zip(current_patch)
            .map(|(left, right)| u64::from(*left) * u64::from(*right))
            .sum::<u64>();
    }
    let count = (diameter * diameter) as f64;
    let left_sum = left_sum as f64;
    let right_sum = right_sum as f64;
    let left_energy = left_squared as f64 - left_sum * left_sum / count;
    let right_energy = right_squared as f64 - right_sum * right_sum / count;
    let covariance = cross as f64 - left_sum * right_sum / count;
    if left_energy < 32.0 || right_energy < 32.0 {
        return f32::INFINITY;
    }
    let correlation = (covariance / (left_energy * right_energy).sqrt().max(32.0)).clamp(-1.0, 1.0);
    (1.0 - correlation).max(0.0).sqrt() as f32
}

#[derive(Clone, Copy, Debug)]
struct SubpixelPatchMatch {
    current: [f32; 2],
    correction: [f32; 2],
}

/// Refine one already-selected native ZNCC basin below the integer sensor
/// grid. Squared ZNCC distance (`1-correlation`) has a smoother local bowl
/// than its square root, so the finite-difference Hessian is fitted to that
/// objective. This is an interpolation of nine exact native-RAW patch costs;
/// it does not resize or copy the source ROI.
fn refine_native_zncc_subpixel(
    previous: &RawFrame,
    current: &RawFrame,
    previous_moments: &IntegralPatchMoments,
    current_moments: &IntegralPatchMoments,
    previous_point: [f32; 2],
    integer_match: [f32; 2],
    radius: i32,
    native_patch_evaluations: &mut usize,
) -> Option<SubpixelPatchMatch> {
    let anchor = [integer_match[0].round(), integer_match[1].round()];
    let mut objective = [[f32::INFINITY; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let candidate = [
                anchor[0] + column as f32 - 1.0,
                anchor[1] + row as f32 - 1.0,
            ];
            let cost = patch_cost_with_integral_moments(
                previous,
                current,
                previous_moments,
                current_moments,
                previous_point,
                candidate,
                radius,
            );
            *native_patch_evaluations += 1;
            if !cost.is_finite() {
                return None;
            }
            objective[row][column] = cost * cost;
        }
    }

    let center = objective[1][1];
    let gradient_x = 0.5 * (objective[1][2] - objective[1][0]);
    let gradient_y = 0.5 * (objective[2][1] - objective[0][1]);
    let hessian_xx = objective[1][2] - 2.0 * center + objective[1][0];
    let hessian_yy = objective[2][1] - 2.0 * center + objective[0][1];
    let hessian_xy = 0.25 * (objective[2][2] - objective[2][0] - objective[0][2] + objective[0][0]);
    let trace = hessian_xx + hessian_yy;
    let discriminant = ((hessian_xx - hessian_yy).powi(2) + 4.0 * hessian_xy.powi(2)).sqrt();
    let minimum_eigenvalue = 0.5 * (trace - discriminant);
    let maximum_eigenvalue = 0.5 * (trace + discriminant);
    if !minimum_eigenvalue.is_finite()
        || !maximum_eigenvalue.is_finite()
        || minimum_eigenvalue < SUBPIXEL_MIN_CURVATURE
        || maximum_eigenvalue <= 0.0
        || minimum_eigenvalue / maximum_eigenvalue < SUBPIXEL_MIN_CURVATURE_RATIO
    {
        return None;
    }
    let determinant = hessian_xx * hessian_yy - hessian_xy * hessian_xy;
    if !determinant.is_finite() || determinant <= SUBPIXEL_MIN_CURVATURE.powi(2) {
        return None;
    }
    let correction = [
        (-hessian_yy * gradient_x + hessian_xy * gradient_y) / determinant,
        (hessian_xy * gradient_x - hessian_xx * gradient_y) / determinant,
    ];
    if !correction[0].is_finite()
        || !correction[1].is_finite()
        || correction[0].abs() > SUBPIXEL_MAX_OFFSET
        || correction[1].abs() > SUBPIXEL_MAX_OFFSET
    {
        return None;
    }

    // The correlation patch is centered on the rounded previous coordinate.
    // Preserve the tracked feature's prior fractional phase when transporting
    // the refined displacement into the current sensor frame; otherwise every
    // exposure would silently quantize yesterday's subpixel estimate away.
    let previous_phase = [
        previous_point[0] - previous_point[0].round(),
        previous_point[1] - previous_point[1].round(),
    ];
    Some(SubpixelPatchMatch {
        current: [
            anchor[0] + correction[0] + previous_phase[0],
            anchor[1] + correction[1] + previous_phase[1],
        ],
        correction,
    })
}

fn patch_cost_with_radius(
    previous: &RawFrame,
    current: &RawFrame,
    a: [f32; 2],
    b: [f32; 2],
    radius: i32,
) -> f32 {
    let ax = a[0].round() as i32;
    let ay = a[1].round() as i32;
    let bx = b[0].round() as i32;
    let by = b[1].round() as i32;
    let mut left_sum = 0.0f32;
    let mut right_sum = 0.0f32;
    let mut left_squared = 0.0f32;
    let mut right_squared = 0.0f32;
    let mut cross = 0.0f32;
    let mut count = 0.0f32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (Some(p), Some(q)) = (
                sample(previous, ax + dx, ay + dy),
                sample(current, bx + dx, by + dy),
            ) else {
                return f32::INFINITY;
            };
            left_sum += p;
            right_sum += q;
            left_squared += p * p;
            right_squared += q * q;
            cross += p * q;
            count += 1.0;
        }
    }
    let left_energy = left_squared - left_sum * left_sum / count;
    let right_energy = right_squared - right_sum * right_sum / count;
    let covariance = cross - left_sum * right_sum / count;
    if left_energy < 32.0 || right_energy < 32.0 {
        return f32::INFINITY;
    }
    // Zero-mean normalized correlation is invariant to the large exposure and
    // local contrast changes in the RAW burst. The previous normalized SSD
    // still penalized gain changes and killed otherwise valid long tracks.
    let correlation = (covariance / (left_energy * right_energy).sqrt().max(32.0)).clamp(-1.0, 1.0);
    (1.0 - correlation).max(0.0).sqrt()
}

fn patch_cost(previous: &RawFrame, current: &RawFrame, a: [f32; 2], b: [f32; 2]) -> f32 {
    patch_cost_with_radius(previous, current, a, b, PATCH_RADIUS)
}

fn downsample_two(frame: &RawFrame) -> RawFrame {
    let width = frame.width / 2;
    let height = frame.height / 2;
    let mut pixels = vec![0u16; width * height];
    for y in 0..height {
        for x in 0..width {
            let source = 2 * y * frame.width + 2 * x;
            let sum = frame.pixels[source] as u32
                + frame.pixels[source + 1] as u32
                + frame.pixels[source + frame.width] as u32
                + frame.pixels[source + frame.width + 1] as u32;
            pixels[y * width + x] = ((sum + 2) / 4) as u16;
        }
    }
    RawFrame {
        sensor_x: frame.sensor_x / 2,
        sensor_y: frame.sensor_y / 2,
        width,
        height,
        pixels,
    }
}

fn corner_score(frame: &RawFrame, x: usize, y: usize) -> f32 {
    if x < 2 || y < 2 || x + 2 >= frame.width || y + 2 >= frame.height {
        return 0.0;
    }
    let mut xx = 0.0f32;
    let mut yy = 0.0f32;
    let mut xy = 0.0f32;
    for oy in -1..=1 {
        for ox in -1..=1 {
            let px = x as i32 + ox;
            let py = y as i32 + oy;
            let gx =
                sample(frame, px + 1, py).unwrap_or(0.0) - sample(frame, px - 1, py).unwrap_or(0.0);
            let gy =
                sample(frame, px, py + 1).unwrap_or(0.0) - sample(frame, px, py - 1).unwrap_or(0.0);
            xx += gx * gx;
            yy += gy * gy;
            xy += gx * gy;
        }
    }
    let trace = xx + yy;
    let determinant = (xx * yy - xy * xy).max(0.0);
    0.5 * (trace - (trace * trace - 4.0 * determinant).max(0.0).sqrt())
}

/// Exposure-invariant bright-peak score for separating corneal/glint motion
/// from the darker pupil/iris surface. The center peak is compared with the
/// median and RMS contrast of a larger local patch, so a merely bright sclera
/// or exposure change does not become reflection evidence.
fn feature_specularity(frame: &RawFrame, point: [f32; 2]) -> f32 {
    let cx = point[0].round() as i32;
    let cy = point[1].round() as i32;
    if cx < 9 || cy < 9 || cx + 9 >= frame.width as i32 || cy + 9 >= frame.height as i32 {
        return 0.0;
    }
    let mut patch = Vec::with_capacity(19 * 19);
    let mut peak = f32::NEG_INFINITY;
    for dy in -9..=9 {
        for dx in -9..=9 {
            let value = sample(frame, cx + dx, cy + dy).unwrap_or(0.0);
            patch.push(value);
            if dx.abs() <= 2 && dy.abs() <= 2 {
                peak = peak.max(value);
            }
        }
    }
    let mut median_samples = patch.clone();
    let baseline = median(&mut median_samples);
    let mean = patch.iter().sum::<f32>() / patch.len() as f32;
    let deviation = (patch
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / patch.len() as f32)
        .sqrt()
        .max(1.0);
    ((peak - baseline) / deviation).max(0.0)
}

// CFA-neutral local Tenengrad energy, normalized by patch contrast.  This is
// sampled from the same native RAW10 plane as matching; no demosaic or ISP
// sharpening is allowed to influence the prospective depth curve.
fn feature_sharpness(frame: &RawFrame, point: [f32; 2]) -> f32 {
    let cx = point[0].round() as i32;
    let cy = point[1].round() as i32;
    let mut gradient = 0.0f32;
    let mut sum = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut count = 0.0f32;
    for dy in -3..=3 {
        for dx in -3..=3 {
            let (Some(value), Some(left), Some(right), Some(up), Some(down)) = (
                sample(frame, cx + dx, cy + dy),
                sample(frame, cx + dx - 1, cy + dy),
                sample(frame, cx + dx + 1, cy + dy),
                sample(frame, cx + dx, cy + dy - 1),
                sample(frame, cx + dx, cy + dy + 1),
            ) else {
                continue;
            };
            let gx = right - left;
            let gy = down - up;
            gradient += gx * gx + gy * gy;
            sum += value;
            sum2 += value * value;
            count += 1.0;
        }
    }
    if count < 16.0 {
        return 0.0;
    }
    let variance = (sum2 - sum * sum / count).max(64.0);
    gradient / variance
}

fn add_focus_sample(track: &mut FeatureTrack, position: u16, sharpness: f32) {
    if !sharpness.is_finite() || sharpness <= 0.0 {
        return;
    }
    if let Some(bin) = track
        .focus_bins
        .iter_mut()
        .find(|bin| bin.position == position)
    {
        bin.sharpness_sum += sharpness;
        bin.samples = bin.samples.saturating_add(1);
    } else {
        track.focus_bins.push(FocusBin {
            position,
            sharpness_sum: sharpness,
            samples: 1,
        });
    }
}

fn estimate_focus_peak(bins: &[FocusBin]) -> Option<f32> {
    if bins.len() < FOCUS_MIN_POSITIONS {
        return None;
    }
    let mut curve = bins
        .iter()
        .filter(|bin| bin.samples > 0)
        .map(|bin| (bin.position, bin.sharpness_sum / bin.samples.max(1) as f32))
        .collect::<Vec<_>>();
    curve.sort_by_key(|sample| sample.0);
    let span = curve.last()?.0.saturating_sub(curve.first()?.0);
    if curve.len() < FOCUS_MIN_POSITIONS || span < FOCUS_MIN_POSITION_SPAN {
        return None;
    }
    let minimum = curve
        .iter()
        .map(|sample| sample.1)
        .fold(f32::INFINITY, f32::min);
    let maximum = curve
        .iter()
        .map(|sample| sample.1)
        .fold(f32::NEG_INFINITY, f32::max);
    if !minimum.is_finite() || maximum <= minimum * 1.01 {
        return None;
    }
    // A positive, contrast-relative centroid is deliberately bounded by the
    // measured sweep.  It is less brittle than extrapolating a three-sample
    // parabola when the eye moves slightly between focus positions.
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (position, sharpness) in curve {
        let weight = (sharpness - minimum).max(0.0);
        numerator += position as f32 * weight;
        denominator += weight;
    }
    (denominator > 1.0e-6).then_some(numerator / denominator)
}

fn fit_depth_motion(samples: &[(f32, f32)]) -> Option<(f32, f32, f32)> {
    if samples.len() < FOCUS_MIN_FEATURES {
        return None;
    }
    let inverse = 1.0 / samples.len() as f32;
    let mean_x = samples.iter().map(|sample| sample.0).sum::<f32>() * inverse;
    let mean_y = samples.iter().map(|sample| sample.1).sum::<f32>() * inverse;
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (x, y) in samples {
        numerator += (x - mean_x) * (y - mean_y);
        denominator += (x - mean_x) * (x - mean_x);
    }
    if denominator < 1.0 {
        return None;
    }
    let slope = numerator / denominator;
    Some((slope, mean_y - slope * mean_x, mean_y))
}

impl ProspectiveFocusSfm {
    fn begin(&mut self) {
        self.status.generation = self.status.generation.saturating_add(1);
        self.status.phase = FocusSfmPhase::Collecting;
        self.status.calibrated_features = 0;
        self.status.train_samples = 0;
        self.status.test_samples = 0;
        self.status.planar_error = 0.0;
        self.status.depth_error = 0.0;
        self.status.improvement = 0.0;
        self.train_frames = 0;
        self.test_frames = 0;
        self.training.clear();
        self.planar_error_sum = 0.0;
        self.depth_error_sum = 0.0;
    }

    fn finish_collection(&mut self, calibrated_features: usize) {
        self.status.calibrated_features = calibrated_features;
        self.status.phase = if calibrated_features >= FOCUS_MIN_FEATURES {
            FocusSfmPhase::Validating
        } else {
            FocusSfmPhase::Rejected
        };
    }

    fn observe_motion(&mut self, samples: &[(f32, f32)]) {
        if self.status.phase != FocusSfmPhase::Validating || samples.is_empty() {
            return;
        }
        if self.train_frames < SFM_TRAIN_FRAMES {
            self.training.extend_from_slice(samples);
            self.train_frames += 1;
            self.status.train_samples = self.training.len();
            if self.train_frames == SFM_TRAIN_FRAMES {
                if let Some((slope, intercept, baseline)) = fit_depth_motion(&self.training) {
                    self.slope = slope;
                    self.intercept = intercept;
                    self.baseline = baseline;
                } else {
                    self.status.phase = FocusSfmPhase::Rejected;
                }
            }
            return;
        }
        for (depth, motion) in samples {
            self.planar_error_sum += (motion - self.baseline).abs();
            self.depth_error_sum += (motion - (self.slope * depth + self.intercept)).abs();
        }
        self.test_frames += 1;
        self.status.test_samples += samples.len();
        if self.test_frames < SFM_TEST_FRAMES {
            return;
        }
        let count = self.status.test_samples.max(1) as f32;
        self.status.planar_error = self.planar_error_sum / count;
        self.status.depth_error = self.depth_error_sum / count;
        self.status.improvement = if self.status.planar_error > 1.0e-5 {
            1.0 - self.status.depth_error / self.status.planar_error
        } else {
            0.0
        };
        self.status.phase = if self.status.test_samples >= SFM_MIN_TEST_SAMPLES
            && self.status.depth_error <= self.status.planar_error * SFM_ACCEPT_RATIO
        {
            FocusSfmPhase::Accepted
        } else {
            FocusSfmPhase::Rejected
        };
    }
}

fn local_canny_peak(edges: &CannyField, width: usize, x: usize, y: usize) -> (f32, [f32; 2]) {
    let height = edges.accepted.len() / width.max(1);
    let mut strongest = (0.0f32, [0.0f32; 2]);
    for dy in -2isize..=2 {
        for dx in -2isize..=2 {
            let sample_x = x.saturating_add_signed(dx).min(width - 1);
            let sample_y = y.saturating_add_signed(dy).min(height - 1);
            let index = sample_y * width + sample_x;
            let magnitude = edges.magnitude[index];
            if edges.accepted[index] && magnitude > strongest.0 {
                strongest = (
                    magnitude,
                    [
                        edges.gradient_x[index] / magnitude.max(1.0e-6),
                        edges.gradient_y[index] / magnitude.max(1.0e-6),
                    ],
                );
            }
        }
    }
    (strongest.0 / edges.high_threshold.max(1.0), strongest.1)
}

fn local_canny_support(edges: &CannyField, width: usize, x: usize, y: usize) -> f32 {
    local_canny_peak(edges, width, x, y).0
}

fn local_canny_normal(edges: &CannyField, width: usize, x: usize, y: usize) -> [f32; 2] {
    local_canny_peak(edges, width, x, y).1
}

fn sample_native_raw_bilinear(frame: &RawFrame, x: f32, y: f32) -> Option<f32> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.0
        || y < 0.0
        || x + 1.0 >= frame.width as f32
        || y + 1.0 >= frame.height as f32
    {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let row0 = y0 * frame.width;
    let row1 = (y0 + 1) * frame.width;
    let top = frame.pixels[row0 + x0] as f32 * (1.0 - tx) + frame.pixels[row0 + x0 + 1] as f32 * tx;
    let bottom =
        frame.pixels[row1 + x0] as f32 * (1.0 - tx) + frame.pixels[row1 + x0 + 1] as f32 * tx;
    Some(top * (1.0 - ty) + bottom * ty)
}

fn nautilus_fingerprint(
    frame: &RawFrame,
    point: [f32; 2],
    _normal: [f32; 2],
) -> Option<NautilusFingerprint> {
    const RADII: [f32; NAUTILUS_SHELLS] = [1.5, 3.0, 5.5, 9.0, 14.0, 21.0];
    // Keep the sparse texture walk in sensor axes. Canny normals are useful
    // geometric evidence, but rotating the outer 21 px shell by a noisy
    // per-frame normal estimate moves samples several pixels and destroys the
    // very temporal identity this descriptor is meant to preserve. Normal
    // agreement is therefore evaluated separately by the candidate gate.
    let base_angle = 0.0f32;
    let mut opposed_samples = [(0.0f32, 0.0f32); NAUTILUS_SHELLS * NAUTILUS_BRANCH_PAIRS];
    let mut raw_samples = [0.0f32; NAUTILUS_SHELLS * NAUTILUS_BRANCH_PAIRS * 2];
    let mut raw_index = 0usize;
    for (shell, radius) in RADII.into_iter().enumerate() {
        // The outward phase drift is the nautilus walk: adjacent shells do not
        // repeatedly sample one straight ridge, so a long eyelid edge cannot
        // look unique merely because every radius hit the same line.
        let shell_twist = shell as f32 * 0.17;
        for branch in 0..NAUTILUS_BRANCH_PAIRS {
            let angle = base_angle
                + branch as f32 * std::f32::consts::PI / NAUTILUS_BRANCH_PAIRS as f32
                + shell_twist;
            let offset = [radius * angle.cos(), radius * angle.sin()];
            let positive =
                sample_native_raw_bilinear(frame, point[0] + offset[0], point[1] + offset[1])?;
            let negative =
                sample_native_raw_bilinear(frame, point[0] - offset[0], point[1] - offset[1])?;
            opposed_samples[shell * NAUTILUS_BRANCH_PAIRS + branch] = (positive, negative);
            raw_samples[raw_index] = positive;
            raw_samples[raw_index + 1] = negative;
            raw_index += 2;
        }
    }
    let mean = raw_samples.iter().sum::<f32>() / raw_samples.len() as f32;
    let variance = raw_samples
        .iter()
        .map(|sample| (sample - mean) * (sample - mean))
        .sum::<f32>()
        / raw_samples.len() as f32;
    let deviation = variance.sqrt();
    if !deviation.is_finite() || deviation < 4.0 {
        return None;
    }
    let mut values = [0i8; NAUTILUS_COMPONENTS];
    for (pair_index, (positive, negative)) in opposed_samples.into_iter().enumerate() {
        let pair_mean = 0.5 * (positive + negative);
        let signed_difference = 0.5 * (positive - negative);
        values[pair_index * 2] = (((pair_mean - mean) / deviation) * 42.0)
            .round()
            .clamp(-127.0, 127.0) as i8;
        values[pair_index * 2 + 1] = ((signed_difference / deviation) * 34.0)
            .round()
            .clamp(-127.0, 127.0) as i8;
    }
    Some(NautilusFingerprint {
        values,
        quality: (deviation / 72.0).clamp(0.0, 1.0),
    })
}

fn nautilus_fingerprint_distance(left: &NautilusFingerprint, right: &NautilusFingerprint) -> f32 {
    // Fixed-width arithmetic is intentionally simple so LLVM can vectorize
    // the embarrassingly parallel descriptor comparison on the host CPU.
    let squared = left
        .values
        .iter()
        .zip(right.values.iter())
        .map(|(left, right)| {
            let difference = *left as f32 - *right as f32;
            difference * difference
        })
        .sum::<f32>();
    let rms = (squared / NAUTILUS_COMPONENTS as f32).sqrt() / 64.0;
    rms + 0.035 * (1.0 - left.quality.min(right.quality))
}

fn nautilus_descriptor_box_distance(
    fingerprint: &NautilusFingerprint,
    minimum: &[i8; NAUTILUS_COMPONENTS],
    maximum: &[i8; NAUTILUS_COMPONENTS],
) -> f32 {
    let squared = fingerprint
        .values
        .iter()
        .zip(minimum.iter().zip(maximum.iter()))
        .map(|(value, (minimum, maximum))| {
            let difference = if value < minimum {
                *minimum as f32 - *value as f32
            } else if value > maximum {
                *value as f32 - *maximum as f32
            } else {
                0.0
            };
            difference * difference
        })
        .sum::<f32>();
    (squared / NAUTILUS_COMPONENTS as f32).sqrt() / 64.0
}

fn point_to_bounds_distance(point: [f32; 2], bounds: [f32; 4]) -> f32 {
    let dx = if point[0] < bounds[0] {
        bounds[0] - point[0]
    } else if point[0] > bounds[2] {
        point[0] - bounds[2]
    } else {
        0.0
    };
    let dy = if point[1] < bounds[1] {
        bounds[1] - point[1]
    } else if point[1] > bounds[3] {
        point[1] - bounds[3]
    } else {
        0.0
    };
    dx.hypot(dy)
}

impl NautilusFingerprintTree {
    fn from_edges(frame: &RawFrame, edges: &[EdgeEvidence]) -> Self {
        let mut ranked = edges
            .iter()
            .filter_map(|edge| {
                let point = [edge.x, edge.y];
                let border = 24.0;
                if point[0] < border
                    || point[1] < border
                    || point[0] + border >= frame.width as f32
                    || point[1] + border >= frame.height as f32
                {
                    return None;
                }
                let normal = normalized_vector([edge.gradient_x, edge.gradient_y]);
                let fingerprint = nautilus_fingerprint(frame, point, normal)?;
                let material = 0.35
                    + 0.35 * edge.multiscale_consistency.clamp(0.0, 1.0)
                    + 0.30 * edge.signed_step_persistence.clamp(0.0, 1.0);
                let texture = (0.55
                    + 0.12 * (edge.dark_side_texture + edge.bright_side_texture).clamp(0.0, 3.0))
                .clamp(0.55, 1.0);
                let evidence = edge.strength.max(0.0)
                    * material
                    * texture
                    * (0.55 + 0.45 * fingerprint.quality);
                Some(NautilusCandidate {
                    point,
                    normal,
                    fingerprint,
                    evidence,
                })
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.evidence.total_cmp(&left.evidence));

        // Keep the search global but spatially balanced. A dense brow or lid
        // may be strong, yet it must not consume every descriptor leaf.
        let tile_columns = frame.width.div_ceil(EDGE_TILE_SIZE);
        let tile_rows = frame.height.div_ceil(EDGE_TILE_SIZE);
        let mut tile_counts = vec![0u8; tile_columns * tile_rows];
        let mut candidates = Vec::<NautilusCandidate>::new();
        for candidate in ranked {
            let tile_x = (candidate.point[0] as usize / EDGE_TILE_SIZE).min(tile_columns - 1);
            let tile_y = (candidate.point[1] as usize / EDGE_TILE_SIZE).min(tile_rows - 1);
            let tile = tile_y * tile_columns + tile_x;
            if tile_counts[tile] >= 18
                || candidates.iter().any(|selected| {
                    (selected.point[0] - candidate.point[0])
                        .hypot(selected.point[1] - candidate.point[1])
                        < NAUTILUS_CANDIDATE_SEPARATION
                })
            {
                continue;
            }
            tile_counts[tile] = tile_counts[tile].saturating_add(1);
            candidates.push(candidate);
            if candidates.len() >= NAUTILUS_MAX_CANDIDATES {
                break;
            }
        }
        let mut tree = Self {
            candidates,
            ..Self::default()
        };
        if !tree.candidates.is_empty() {
            let indices = (0..tree.candidates.len()).collect::<Vec<_>>();
            tree.root = Some(tree.append_node(indices));
        }
        tree
    }

    fn append_node(&mut self, mut candidates: Vec<usize>) -> usize {
        let mut spatial_bounds = [
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ];
        let mut descriptor_min = [i8::MAX; NAUTILUS_COMPONENTS];
        let mut descriptor_max = [i8::MIN; NAUTILUS_COMPONENTS];
        for candidate_index in &candidates {
            let candidate = self.candidates[*candidate_index];
            spatial_bounds[0] = spatial_bounds[0].min(candidate.point[0]);
            spatial_bounds[1] = spatial_bounds[1].min(candidate.point[1]);
            spatial_bounds[2] = spatial_bounds[2].max(candidate.point[0]);
            spatial_bounds[3] = spatial_bounds[3].max(candidate.point[1]);
            for dimension in 0..NAUTILUS_COMPONENTS {
                descriptor_min[dimension] =
                    descriptor_min[dimension].min(candidate.fingerprint.values[dimension]);
                descriptor_max[dimension] =
                    descriptor_max[dimension].max(candidate.fingerprint.values[dimension]);
            }
        }
        let split_dimension = (0..NAUTILUS_COMPONENTS)
            .max_by_key(|dimension| {
                descriptor_max[*dimension] as i16 - descriptor_min[*dimension] as i16
            })
            .unwrap_or(0);
        let node_index = self.nodes.len();
        self.nodes.push(NautilusTreeNode {
            spatial_bounds,
            descriptor_min,
            descriptor_max,
            split_dimension,
            split_value: 0,
            left: None,
            right: None,
            candidates: Vec::new(),
        });
        let descriptor_range =
            descriptor_max[split_dimension] as i16 - descriptor_min[split_dimension] as i16;
        if candidates.len() <= NAUTILUS_TREE_LEAF_CAPACITY || descriptor_range == 0 {
            self.nodes[node_index].candidates = candidates;
            return node_index;
        }
        candidates.sort_by_key(|candidate_index| {
            self.candidates[*candidate_index].fingerprint.values[split_dimension]
        });
        let right_candidates = candidates.split_off(candidates.len() / 2);
        let split_value = self.candidates[right_candidates[0]].fingerprint.values[split_dimension];
        let left = self.append_node(candidates);
        let right = self.append_node(right_candidates);
        self.nodes[node_index].split_value = split_value;
        self.nodes[node_index].left = Some(left);
        self.nodes[node_index].right = Some(right);
        node_index
    }

    #[allow(clippy::too_many_arguments)]
    fn query_nearest(
        &self,
        fingerprint: &NautilusFingerprint,
        predicted: [f32; 2],
        radius: f32,
        prior_normal: [f32; 2],
        excluded: &[[f32; 2]],
        maximum_results: usize,
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<(f32, usize)> {
        let mut results = Vec::<(f32, usize)>::new();
        if let Some(root) = self.root {
            self.query_node(
                root,
                fingerprint,
                predicted,
                radius,
                prior_normal,
                excluded,
                maximum_results,
                diagnostics,
                &mut results,
            );
        }
        results.sort_by(|left, right| left.0.total_cmp(&right.0));
        results.truncate(maximum_results);
        results
    }

    #[allow(clippy::too_many_arguments)]
    fn query_node(
        &self,
        node_index: usize,
        fingerprint: &NautilusFingerprint,
        predicted: [f32; 2],
        radius: f32,
        prior_normal: [f32; 2],
        excluded: &[[f32; 2]],
        maximum_results: usize,
        diagnostics: &mut MatchDiagnostics,
        results: &mut Vec<(f32, usize)>,
    ) {
        let node = &self.nodes[node_index];
        diagnostics.nautilus_nodes_visited += 1;
        if point_to_bounds_distance(predicted, node.spatial_bounds) > radius {
            return;
        }
        if results.len() >= maximum_results {
            let worst = results
                .iter()
                .map(|result| result.0)
                .max_by(f32::total_cmp)
                .unwrap_or(f32::INFINITY);
            if nautilus_descriptor_box_distance(
                fingerprint,
                &node.descriptor_min,
                &node.descriptor_max,
            ) > worst
            {
                return;
            }
        }
        if node.left.is_none() {
            for candidate_index in &node.candidates {
                let candidate = self.candidates[*candidate_index];
                if (candidate.point[0] - predicted[0]).hypot(candidate.point[1] - predicted[1])
                    > radius
                    || excluded.iter().any(|point| {
                        (candidate.point[0] - point[0]).hypot(candidate.point[1] - point[1])
                            < MIN_MATCH_DESTINATION_SEPARATION
                    })
                {
                    continue;
                }
                if prior_normal[0].hypot(prior_normal[1]) > 0.5
                    && (prior_normal[0] * candidate.normal[0]
                        + prior_normal[1] * candidate.normal[1])
                        .abs()
                        < 0.08
                {
                    continue;
                }
                diagnostics.nautilus_descriptor_evaluations += 1;
                let distance = nautilus_fingerprint_distance(fingerprint, &candidate.fingerprint);
                results.push((distance, *candidate_index));
                results.sort_by(|left, right| left.0.total_cmp(&right.0));
                results.truncate(maximum_results);
            }
            return;
        }
        let (first, second) = if fingerprint.values[node.split_dimension] < node.split_value {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        if let Some(first) = first {
            self.query_node(
                first,
                fingerprint,
                predicted,
                radius,
                prior_normal,
                excluded,
                maximum_results,
                diagnostics,
                results,
            );
        }
        if let Some(second) = second {
            self.query_node(
                second,
                fingerprint,
                predicted,
                radius,
                prior_normal,
                excluded,
                maximum_results,
                diagnostics,
                results,
            );
        }
    }
}

fn nautilus_track_prediction(
    track: &FeatureTrack,
    motions: &[SimilarityMotion; OBJECTS],
    layers: &[MotionLayerStatus; OBJECTS],
    center: [f32; 2],
    cadence_contiguous: bool,
) -> ([f32; 2], f32) {
    let last = track
        .points
        .back()
        .copied()
        .unwrap_or([center[0], center[1], 0.0]);
    let elapsed_steps = track.age as f32 + 1.0;
    // Constant-velocity extrapolation through an entire blink is generally
    // worse than holding the last material coordinate. Surviving current
    // anchors, when available, still estimate the complete cross-gap motion
    // in `nautilus_anchor_conditioned_prediction` below.
    let prediction_steps = if track.age > MAX_AGE {
        1.0
    } else {
        elapsed_steps
    };
    let model = motions[track.object];
    let layer = layers[track.object];
    let model_usable = cadence_contiguous
        && model.support >= 3
        && model.residual <= 3.0
        && layer.stable_frames >= MIN_LAYER_STABLE_FRAMES
        && layer.coherence >= 0.20;
    let track_usable = cadence_contiguous
        && track.motion_ema[0].hypot(track.motion_ema[1]) <= SEARCH_RADIUS as f32 * 1.5;
    let track_displacement = track.motion_ema;
    let modeled = model.predict([last[0], last[1]], center);
    let model_displacement = [modeled[0] - last[0], modeled[1] - last[1]];
    let displacement = if model_usable && track_usable {
        [
            0.5 * (track_displacement[0] + model_displacement[0]),
            0.5 * (track_displacement[1] + model_displacement[1]),
        ]
    } else if model_usable {
        model_displacement
    } else if track_usable {
        track_displacement
    } else {
        [0.0; 2]
    };
    let uncertainty = 20.0
        + track.age as f32 * 11.0
        + 1.5 * track.motion_variance.max(0.0).sqrt()
        + if model_usable || track_usable {
            0.0
        } else {
            10.0
        };
    (
        [
            last[0] + displacement[0] * prediction_steps,
            last[1] + displacement[1] * prediction_steps,
        ],
        uncertainty.clamp(20.0, 58.0),
    )
}

fn nautilus_identity_acceptance_radius(track: &FeatureTrack) -> f32 {
    (7.0 + track.age as f32 * 4.0 + 1.5 * track.motion_variance.max(0.0).sqrt()).clamp(7.0, 20.0)
}

#[derive(Clone, Copy, Debug)]
struct NautilusAnchorPrediction {
    point: [f32; 2],
    search_radius: f32,
    identity_radius: f32,
    fallback_disagreement: f32,
    residual: f32,
    same_object_support: usize,
}

/// Predict a temporarily missing point from the ordinary tracks which did
/// survive into the current frame. For a two-frame gap, an anchor's point one
/// observation behind its latest point is paired with its current match; this
/// keeps every correspondence on the same temporal baseline as the missing
/// point instead of comparing a stale location with a one-frame velocity.
fn nautilus_anchor_conditioned_prediction(
    track_index: usize,
    track: &FeatureTrack,
    tracks: &[FeatureTrack],
    matches: &[Match],
    fallback: [f32; 2],
) -> Option<NautilusAnchorPrediction> {
    let source = track.points.back().copied()?;
    let history_index = track.age as usize;
    let mut anchors = matches
        .iter()
        .filter_map(|item| {
            if item.track_index == track_index {
                return None;
            }
            let anchor_track = tracks.get(item.track_index)?;
            let historical = anchor_track.points.iter().rev().nth(history_index)?;
            let previous = [historical[0], historical[1]];
            let distance = (previous[0] - source[0]).hypot(previous[1] - source[1]);
            let same_object = item.object == track.object;
            let inverse_steps = 1.0 / (track.age as f32 + 1.0);
            let anchor_velocity = [
                (item.current[0] - previous[0]) * inverse_steps,
                (item.current[1] - previous[1]) * inverse_steps,
            ];
            let motion_error = (anchor_velocity[0] - track.motion_ema[0])
                .hypot(anchor_velocity[1] - track.motion_ema[1]);
            Some((
                same_object,
                distance,
                motion_error,
                Match { previous, ..*item },
            ))
        })
        .collect::<Vec<_>>();
    if anchors.len() < 2 {
        return None;
    }
    let same_object_support = anchors.iter().filter(|anchor| anchor.0).count();
    if same_object_support >= 4 {
        anchors.retain(|anchor| anchor.0);
    }
    let motion_tolerance = (2.0 + 1.5 * track.motion_variance.max(0.0).sqrt()).clamp(2.0, 5.0);
    if anchors
        .iter()
        .filter(|anchor| anchor.2 <= motion_tolerance)
        .count()
        >= 4
    {
        anchors.retain(|anchor| anchor.2 <= motion_tolerance);
    }
    anchors.sort_by(|left, right| left.1.total_cmp(&right.1));
    let local = anchors
        .into_iter()
        .take(12)
        .map(|anchor| anchor.3)
        .collect::<Vec<_>>();
    if local.len() < 2 {
        return None;
    }
    let source_xy = [source[0], source[1]];
    let model = robust_global_similarity(&local, source_xy);
    let point = model.predict(source_xy, source_xy);
    if !point[0].is_finite()
        || !point[1].is_finite()
        || (point[0] - fallback[0]).hypot(point[1] - fallback[1]) > 18.0 + 8.0 * track.age as f32
    {
        return None;
    }
    let mut residuals = local
        .iter()
        .map(|item| {
            let predicted = model.predict(item.previous, source_xy);
            (predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1])
        })
        .collect::<Vec<_>>();
    let residual = median(&mut residuals).max(0.20);
    Some(NautilusAnchorPrediction {
        point,
        search_radius: (5.0 + 2.0 * residual + 1.5 * track.age as f32).clamp(6.0, 16.0),
        identity_radius: (2.0 + 1.5 * residual + 0.75 * track.age as f32).clamp(2.5, 7.0),
        fallback_disagreement: (point[0] - fallback[0]).hypot(point[1] - fallback[1]),
        residual,
        same_object_support,
    })
}

fn nautilus_bank_descriptor_score(
    bank: &NautilusFingerprintBank,
    candidate: &NautilusFingerprint,
) -> f32 {
    let Some(newest) = bank.observations.back() else {
        return f32::INFINITY;
    };
    let newest_distance = nautilus_fingerprint_distance(newest, candidate);
    let historical_distance = bank
        .observations
        .iter()
        .map(|observation| nautilus_fingerprint_distance(observation, candidate))
        .min_by(f32::total_cmp)
        .unwrap_or(newest_distance);
    // Most recent appearance is the causal reference. A historical exemplar
    // can soften a transient glint, but cannot win an identity on its own.
    0.72 * newest_distance + 0.28 * historical_distance
}

#[allow(clippy::too_many_arguments)]
fn nautilus_rank_bank_candidates(
    tree: &NautilusFingerprintTree,
    bank: &NautilusFingerprintBank,
    predicted: [f32; 2],
    radius: f32,
    prior_normal: [f32; 2],
    excluded: &[[f32; 2]],
    diagnostics: &mut MatchDiagnostics,
) -> Vec<(usize, f32)> {
    let mut candidate_indices = BTreeMap::<usize, ()>::new();
    for observation in &bank.observations {
        for (_, candidate_index) in tree.query_nearest(
            observation,
            predicted,
            radius,
            prior_normal,
            excluded,
            24,
            diagnostics,
        ) {
            candidate_indices.insert(candidate_index, ());
        }
    }
    let mut ranked = candidate_indices
        .into_keys()
        .map(|candidate_index| {
            let candidate = tree.candidates[candidate_index];
            let descriptor = nautilus_bank_descriptor_score(bank, &candidate.fingerprint);
            let spatial =
                (candidate.point[0] - predicted[0]).hypot(candidate.point[1] - predicted[1]);
            let spatial_penalty = 0.80 * (spatial / radius.max(1.0)).powi(2);
            (candidate_index, descriptor + spatial_penalty)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1));
    ranked
}

#[allow(clippy::too_many_arguments)]
fn nautilus_refine_ranked_candidates(
    frame: &RawFrame,
    tree: &NautilusFingerprintTree,
    bank: &NautilusFingerprintBank,
    predicted: [f32; 2],
    radius: f32,
    prior_normal: [f32; 2],
    excluded: &[[f32; 2]],
    diagnostics: &mut MatchDiagnostics,
) -> Vec<NautilusRefinedCandidate> {
    let coarse = nautilus_rank_bank_candidates(
        tree,
        bank,
        predicted,
        radius,
        prior_normal,
        excluded,
        diagnostics,
    );
    let mut refined = Vec::<NautilusRefinedCandidate>::new();
    for (candidate_index, _) in coarse.into_iter().take(10) {
        let candidate = tree.candidates[candidate_index];
        let mut best = None::<NautilusRefinedCandidate>;
        for dy in -2..=2 {
            for dx in -2..=2 {
                let point = [
                    candidate.point[0] + dx as f32,
                    candidate.point[1] + dy as f32,
                ];
                let spatial = (point[0] - predicted[0]).hypot(point[1] - predicted[1]);
                if spatial > radius
                    || excluded.iter().any(|excluded_point| {
                        (point[0] - excluded_point[0]).hypot(point[1] - excluded_point[1])
                            < MIN_MATCH_DESTINATION_SEPARATION
                    })
                {
                    continue;
                }
                let Some(fingerprint) = nautilus_fingerprint(frame, point, candidate.normal) else {
                    continue;
                };
                diagnostics.nautilus_refinement_evaluations += 1;
                let score = nautilus_bank_descriptor_score(bank, &fingerprint)
                    + 0.80 * (spatial / radius.max(1.0)).powi(2);
                let proposal = NautilusRefinedCandidate {
                    candidate_index,
                    point,
                    normal: candidate.normal,
                    fingerprint,
                    score,
                };
                if best.is_none_or(|previous| proposal.score < previous.score) {
                    best = Some(proposal);
                }
            }
        }
        if let Some(best) = best {
            refined.push(best);
        }
    }
    refined.sort_by(|left, right| left.score.total_cmp(&right.score));
    let mut distinct = Vec::<NautilusRefinedCandidate>::new();
    for candidate in refined {
        if distinct.iter().all(|previous| {
            (candidate.point[0] - previous.point[0]).hypot(candidate.point[1] - previous.point[1])
                >= MIN_MATCH_DESTINATION_SEPARATION
        }) {
            distinct.push(candidate);
        }
    }
    distinct
}

#[allow(clippy::too_many_arguments)]
fn propose_nautilus_relocations(
    current: &RawFrame,
    tree: &NautilusFingerprintTree,
    tracks: &[FeatureTrack],
    banks: &BTreeMap<u64, NautilusFingerprintBank>,
    matches: &[Match],
    motions: &[SimilarityMotion; OBJECTS],
    layers: &[MotionLayerStatus; OBJECTS],
    center: [f32; 2],
    sensor_origin: [f32; 2],
    cadence_contiguous: bool,
    extended_gap_recovery_ready: bool,
    diagnostics: &mut MatchDiagnostics,
) -> Vec<NautilusRelocation> {
    if tree.candidates.is_empty() {
        return Vec::new();
    }
    let mut matched = vec![false; tracks.len()];
    let mut regular_destinations = Vec::new();
    for item in matches {
        matched[item.track_index] = true;
        let track = &tracks[item.track_index];
        // A one-observation seed is not an established identity. After a
        // blink it may have appeared exactly where a dormant, fingerprinted
        // feature is returning; let the established identity compete for that
        // destination, then retire the duplicate if re-identification wins.
        if track.points.len() > 1 || track.matched_streak > 0 {
            regular_destinations.push([
                item.current[0] - sensor_origin[0],
                item.current[1] - sensor_origin[1],
            ]);
        }
    }
    let mut proposals = Vec::<NautilusRelocation>::new();
    for (track_index, track) in tracks.iter().enumerate() {
        if matched[track_index] {
            continue;
        }
        // Longer-lived IDs are intentionally dormant during the occlusion.
        // They may re-enter only after independently measured eye geometry
        // and a coherent open-frame light field have both returned.
        if track.age > MAX_AGE && !extended_gap_recovery_ready {
            continue;
        }
        let Some(bank) = banks.get(&track.id) else {
            continue;
        };
        if bank.observations.len() < NAUTILUS_MIN_BANK_OBSERVATIONS {
            continue;
        }
        diagnostics.nautilus_queries += 1;
        let (fallback_prediction, fallback_radius) =
            nautilus_track_prediction(track, motions, layers, center, cadence_contiguous);
        let anchor_prediction = nautilus_anchor_conditioned_prediction(
            track_index,
            track,
            tracks,
            matches,
            fallback_prediction,
        );
        if anchor_prediction.is_some_and(|prediction| {
            prediction.fallback_disagreement
                > NAUTILUS_ANCHOR_PRIOR_DISAGREEMENT_BASE
                    + NAUTILUS_ANCHOR_PRIOR_DISAGREEMENT_PER_GAP * track.age as f32
                || prediction.residual
                    > NAUTILUS_ANCHOR_RESIDUAL_BASE
                        + NAUTILUS_ANCHOR_RESIDUAL_PER_GAP * track.age as f32
        }) {
            diagnostics.nautilus_spatial_rejected += 1;
            continue;
        }
        let predicted_sensor = anchor_prediction
            .map(|prediction| prediction.point)
            .unwrap_or(fallback_prediction);
        let radius = anchor_prediction
            .map(|prediction| prediction.search_radius)
            .unwrap_or(fallback_radius);
        let mut physical_identity_radius =
            NAUTILUS_IDENTITY_RADIUS_BASE + NAUTILUS_IDENTITY_RADIUS_PER_GAP * track.age as f32;
        if track.age > MAX_AGE {
            physical_identity_radius =
                physical_identity_radius.min(NAUTILUS_EXTENDED_GAP_IDENTITY_RADIUS_MAX);
        }
        let identity_radius = anchor_prediction
            .map(|prediction| prediction.identity_radius)
            .unwrap_or_else(|| nautilus_identity_acceptance_radius(track))
            .min(physical_identity_radius);
        let predicted = [
            predicted_sensor[0] - sensor_origin[0],
            predicted_sensor[1] - sensor_origin[1],
        ];
        let ranked = nautilus_refine_ranked_candidates(
            current,
            tree,
            bank,
            predicted,
            radius,
            track.edge_normal,
            &regular_destinations,
            diagnostics,
        );
        let Some(&candidate) = ranked.first() else {
            diagnostics.nautilus_distance_rejected += 1;
            continue;
        };
        let candidate_index = candidate.candidate_index;
        let best_distance = candidate.score;
        let best_spatial_distance =
            (candidate.point[0] - predicted[0]).hypot(candidate.point[1] - predicted[1]);
        if best_spatial_distance > identity_radius {
            diagnostics.nautilus_spatial_rejected += 1;
            continue;
        }
        if best_distance > NAUTILUS_MAX_DISTANCE {
            diagnostics.nautilus_distance_rejected += 1;
            continue;
        }
        let second_distance = ranked
            .get(1)
            .map_or(f32::INFINITY, |candidate| candidate.score);
        let margin = if second_distance.is_finite() {
            second_distance - best_distance
        } else {
            1.0
        };
        let ratio = if second_distance.is_finite() {
            best_distance / second_distance.max(1.0e-5)
        } else {
            0.0
        };
        if margin < NAUTILUS_MIN_ABSOLUTE_MARGIN || ratio > NAUTILUS_MAX_DISTANCE_RATIO {
            diagnostics.nautilus_ambiguous += 1;
            continue;
        }

        // Reverse identity check: the selected current fingerprint must also
        // prefer this historical ID over every other live track bank,
        // including IDs which the ordinary adjacent matcher already found.
        // Motion may explain where to search, but it cannot make a repeated
        // eyelid/limbus texture globally unique.
        let own_identity_distance = nautilus_bank_descriptor_score(bank, &candidate.fingerprint);
        let mut competing_identity_distance = f32::INFINITY;
        for (other_index, other_track) in tracks.iter().enumerate() {
            if other_index == track_index {
                continue;
            }
            if other_track.points.len() <= 1 && other_track.matched_streak == 0 {
                continue;
            }
            let Some(other_bank) = banks.get(&other_track.id) else {
                continue;
            };
            competing_identity_distance = competing_identity_distance.min(
                nautilus_bank_descriptor_score(other_bank, &candidate.fingerprint),
            );
        }
        let reverse_margin = competing_identity_distance - own_identity_distance;
        if competing_identity_distance.is_finite() && reverse_margin < NAUTILUS_MIN_REVERSE_MARGIN {
            diagnostics.nautilus_reverse_ambiguous += 1;
            continue;
        }
        let confidence = (0.45 * (1.0 - best_distance / NAUTILUS_MAX_DISTANCE)
            + 0.35 * (margin / 0.30).clamp(0.0, 1.0)
            + 0.20 * (reverse_margin / 0.25).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        if confidence < NAUTILUS_MIN_ACCEPTANCE_CONFIDENCE {
            diagnostics.nautilus_ambiguous += 1;
            continue;
        }
        proposals.push(NautilusRelocation {
            track_index,
            candidate_index,
            current: [
                candidate.point[0] + sensor_origin[0],
                candidate.point[1] + sensor_origin[1],
            ],
            fingerprint: candidate.fingerprint,
            distance: best_distance,
            margin,
            distance_ratio: ratio,
            prediction_distance: best_spatial_distance,
            identity_radius,
            reverse_margin,
            normal_alignment: (track.edge_normal[0] * candidate.normal[0]
                + track.edge_normal[1] * candidate.normal[1])
                .abs(),
            anchor_conditioned: anchor_prediction.is_some(),
            confidence,
            crossed_gap: track.age > 0,
        });
    }
    proposals.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    let mut accepted = Vec::<NautilusRelocation>::new();
    let mut used_candidates = vec![false; tree.candidates.len()];
    for proposal in proposals {
        if used_candidates[proposal.candidate_index]
            || accepted.iter().any(|previous| {
                (previous.current[0] - proposal.current[0])
                    .hypot(previous.current[1] - proposal.current[1])
                    < MIN_MATCH_DESTINATION_SEPARATION
            })
        {
            diagnostics.nautilus_collision_rejected += 1;
            continue;
        }
        used_candidates[proposal.candidate_index] = true;
        diagnostics.nautilus_relocated += 1;
        diagnostics.nautilus_gap_relocated += usize::from(proposal.crossed_gap);
        diagnostics.nautilus_margin_sum += proposal.margin;
        accepted.push(proposal);
    }
    accepted
}

fn seed_points(
    frame: &RawFrame,
    canny: Option<&CannyField>,
    edge_evidence: &[EdgeEvidence],
    iris_seed: Option<IrisEllipseSeed>,
    existing: &[[f32; 2]],
    wanted: usize,
) -> Vec<([f32; 2], f32)> {
    #[derive(Clone, Copy)]
    struct Candidate {
        point: [f32; 2],
        score: f32,
        tile: usize,
        limbus_affinity: f32,
        limbus_sector: usize,
    }

    let mut candidates = Vec::<Candidate>::new();
    let tile_columns = frame.width.div_ceil(FEATURE_SEED_TILE_SIZE);
    for y in (5..frame.height.saturating_sub(5)).step_by(3) {
        for x in (5..frame.width.saturating_sub(5)).step_by(3) {
            let edge_support = canny
                .map(|edges| local_canny_support(edges, frame.width, x, y))
                .unwrap_or(1.0);
            let corner = corner_score(frame, x, y) * edge_support.clamp(0.0, 3.0);
            // Iris striations and a partially visible limbus are often clean
            // Canny lines rather than Harris-like corners. Admit those patches
            // as temporal proposals; forward/backward matching and motion
            // coherence still decide whether they survive. The tile id below
            // prevents one high-contrast lid or hair edge from consuming the
            // whole bounded feature budget.
            let line = canny.map_or(0.0, |_| 180.0 * edge_support.clamp(0.0, 2.0));
            let score = corner.max(line);
            if canny.is_none_or(|_| edge_support >= 0.35) && score >= 64.0 {
                candidates.push(Candidate {
                    point: [x as f32, y as f32],
                    score,
                    tile: (y / FEATURE_SEED_TILE_SIZE) * tile_columns + x / FEATURE_SEED_TILE_SIZE,
                    limbus_affinity: 0.0,
                    limbus_sector: 0,
                });
            }
        }
    }

    // The compact edge bank has already paid for full-resolution,
    // illumination-normalized side-texture attributes. Reuse those exact RAW
    // measurements rather than allocating another image or sampling a
    // rendered color representation. Exact edge centers also avoid moving a
    // smooth limbus observation onto the nearest three-pixel seed lattice.
    if let Some(seed) = iris_seed {
        for edge in edge_evidence {
            let point = [edge.x, edge.y];
            if point[0] < 5.0
                || point[1] < 5.0
                || point[0] + 5.0 >= frame.width as f32
                || point[1] + 5.0 >= frame.height as f32
            {
                continue;
            }
            let Some((affinity, sector)) = limbus_feature_budget_affinity(*edge, seed) else {
                continue;
            };
            if affinity < 0.08 {
                continue;
            }
            let x = point[0].round() as usize;
            let y = point[1].round() as usize;
            let edge_support = edge.strength.clamp(0.20, 3.0);
            let base_score = (corner_score(frame, x, y) * edge_support)
                .max(180.0 * edge_support.clamp(0.0, 2.0));
            if base_score < 64.0 {
                continue;
            }
            // This is the requested budget discount. It changes ordering and
            // the reserved limbus share, never the hard MAX_FEATURES cap.
            let cost = (1.0 - LIMBUS_FEATURE_MAX_COST_DISCOUNT * affinity).clamp(0.55, 1.0);
            candidates.push(Candidate {
                point,
                score: base_score / cost,
                tile: (y / FEATURE_SEED_TILE_SIZE) * tile_columns + x / FEATURE_SEED_TILE_SIZE,
                limbus_affinity: affinity,
                limbus_sector: sector,
            });
        }
    }

    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| right.limbus_affinity.total_cmp(&left.limbus_affinity))
            .then_with(|| left.tile.cmp(&right.tile))
    });
    let mut selected = Vec::new();
    let mut tile_counts = vec![0u8; tile_columns * frame.height.div_ceil(FEATURE_SEED_TILE_SIZE)];
    let mut sector_counts = [0u8; LIMBUS_FEATURE_SECTORS];
    let mut covered_sectors = [false; LIMBUS_FEATURE_SECTORS];
    for candidate in &candidates {
        if candidate.limbus_affinity >= LIMBUS_FEATURE_MIN_AFFINITY {
            covered_sectors[candidate.limbus_sector] = true;
        }
    }
    let angular_coverage = covered_sectors.iter().filter(|covered| **covered).count();
    let opposing_sectors = (0..LIMBUS_FEATURE_SECTORS / 2)
        .filter(|sector| {
            covered_sectors[*sector] && covered_sectors[*sector + LIMBUS_FEATURE_SECTORS / 2]
        })
        .count();
    // A one-sided high-contrast arc is much more likely to be a lid, brow, or
    // glasses rim. Only activate the reserved share after the current RAW
    // frame contains distributed, opposing support around the proposed conic.
    let reserve_enabled = angular_coverage >= 6 && opposing_sectors >= 2;

    macro_rules! admit {
        ($candidate:expr, $limbus_pass:expr) => {{
            let candidate = $candidate;
            let limbus_pass = $limbus_pass;
            let rejected = selected.len() >= wanted
                || (canny.is_some()
                    && tile_counts[candidate.tile]
                        >= if limbus_pass {
                            LIMBUS_FEATURE_MAX_PER_TILE
                        } else {
                            2
                        })
                || (limbus_pass
                    && sector_counts[candidate.limbus_sector] >= LIMBUS_FEATURE_MAX_PER_SECTOR)
                || existing
                    .iter()
                    .chain(selected.iter().map(|item: &([f32; 2], f32)| &item.0))
                    .any(|point| {
                        (point[0] - candidate.point[0]).hypot(point[1] - candidate.point[1])
                            < MIN_FEATURE_SEPARATION
                    });
            if !rejected {
                tile_counts[candidate.tile] = tile_counts[candidate.tile].saturating_add(1);
                if limbus_pass {
                    sector_counts[candidate.limbus_sector] =
                        sector_counts[candidate.limbus_sector].saturating_add(1);
                }
                selected.push((candidate.point, candidate.score));
            }
            !rejected
        }};
    }

    if reserve_enabled {
        for candidate in candidates
            .iter()
            .copied()
            .filter(|candidate| candidate.limbus_affinity >= LIMBUS_FEATURE_MIN_AFFINITY)
        {
            if selected.len() >= wanted.min(LIMBUS_FEATURE_RESERVE) {
                break;
            }
            let _ = admit!(candidate, true);
        }
    }
    for candidate in candidates {
        if selected.len() >= wanted {
            break;
        }
        let limbus_pass =
            reserve_enabled && candidate.limbus_affinity >= LIMBUS_FEATURE_MIN_AFFINITY;
        let _ = admit!(candidate, limbus_pass);
    }
    selected
}

/// Score how cheaply an exact RAW edge should compete for a slot in the
/// general feature bank. The score is deliberately not an anatomy claim: it
/// needs a supplied affine iris seed, a persistent dark-to-bright step, an
/// outward conic normal, and the characteristic contrast/texture difference
/// between the two materials. A unilateral edge cannot unlock the reserved
/// budget by itself; `seed_points` checks opposing angular support separately.
fn limbus_feature_budget_affinity(
    edge: EdgeEvidence,
    seed: IrisEllipseSeed,
) -> Option<(f32, usize)> {
    let ellipse = seed.ellipse();
    if !ellipse.center.0.is_finite()
        || !ellipse.center.1.is_finite()
        || !ellipse.major.is_finite()
        || !ellipse.minor.is_finite()
        || ellipse.major < 8.0
        || ellipse.minor < 8.0
    {
        return None;
    }
    let (sine, cosine) = ellipse.angle.sin_cos();
    let dx = edge.x as f64 - ellipse.center.0;
    let dy = edge.y as f64 - ellipse.center.1;
    let local_x = cosine * dx + sine * dy;
    let local_y = -sine * dx + cosine * dy;
    let normalized_x = local_x / ellipse.major;
    let normalized_y = local_y / ellipse.minor;
    let radius = normalized_x.hypot(normalized_y);
    if !radius.is_finite() {
        return None;
    }
    let radial_support = (1.0 - ((radius - 1.0).abs() / 0.20)).clamp(0.0, 1.0) as f32;
    if radial_support <= 0.0 {
        return None;
    }
    let normal_local_x = local_x / (ellipse.major * ellipse.major);
    let normal_local_y = local_y / (ellipse.minor * ellipse.minor);
    let normal_x = cosine * normal_local_x - sine * normal_local_y;
    let normal_y = sine * normal_local_x + cosine * normal_local_y;
    let normal_length = normal_x.hypot(normal_y).max(1.0e-9);
    let alignment = ((normal_x * edge.gradient_x as f64 + normal_y * edge.gradient_y as f64)
        / normal_length)
        .clamp(0.0, 1.0) as f32;
    let alignment_support = ((alignment - 0.25) / 0.75).clamp(0.0, 1.0);
    if alignment_support <= 0.0 {
        return None;
    }
    let material = ((0.35 + 0.65 * edge.multiscale_consistency.clamp(0.0, 1.0))
        * (0.40 + 0.60 * edge.signed_step_persistence.clamp(0.0, 1.0)))
    .sqrt();
    let texture_split = (edge.dark_side_texture - edge.bright_side_texture)
        .abs()
        .clamp(0.0, 1.0);
    let iris_texture_advantage = (edge.dark_side_texture - edge.bright_side_texture)
        .max(0.0)
        .clamp(0.0, 1.0);
    let side_contrast =
        (0.55 + 0.25 * texture_split + 0.20 * iris_texture_advantage).clamp(0.55, 1.0);
    let affinity = (radial_support * alignment_support * material * side_contrast).clamp(0.0, 1.0);
    let phase = normalized_y
        .atan2(normalized_x)
        .rem_euclid(std::f64::consts::TAU);
    let sector = ((phase / std::f64::consts::TAU * LIMBUS_FEATURE_SECTORS as f64).floor() as usize)
        .min(LIMBUS_FEATURE_SECTORS - 1);
    Some((affinity, sector))
}

fn fit_similarity_selected(selected: &[&Match], center: [f32; 2]) -> SimilarityMotion {
    if selected.len() < 2 {
        return SimilarityMotion::default();
    }
    // Least squares for dx=tx+s*x-w*y, dy=ty+w*x+s*y.  Two small normal
    // systems are avoided by eliminating translation around the centroid.
    let mut pc = [0.0f32; 2];
    let mut dc = [0.0f32; 2];
    for item in selected {
        pc[0] += item.previous[0] - center[0];
        pc[1] += item.previous[1] - center[1];
        dc[0] += item.current[0] - item.previous[0];
        dc[1] += item.current[1] - item.previous[1];
    }
    let inverse = 1.0 / selected.len() as f32;
    pc[0] *= inverse;
    pc[1] *= inverse;
    dc[0] *= inverse;
    dc[1] *= inverse;
    let mut denominator = 0.0f32;
    let mut scale_numerator = 0.0f32;
    let mut rotation_numerator = 0.0f32;
    for item in selected {
        let x = item.previous[0] - center[0] - pc[0];
        let y = item.previous[1] - center[1] - pc[1];
        let dx = item.current[0] - item.previous[0] - dc[0];
        let dy = item.current[1] - item.previous[1] - dc[1];
        denominator += x * x + y * y;
        scale_numerator += x * dx + y * dy;
        rotation_numerator += x * dy - y * dx;
    }
    let scale_delta = scale_numerator / denominator.max(1.0);
    let rotation = rotation_numerator / denominator.max(1.0);
    let translation = [
        dc[0] - scale_delta * pc[0] + rotation * pc[1],
        dc[1] - rotation * pc[0] - scale_delta * pc[1],
    ];
    let mut residual = 0.0f32;
    for item in selected {
        let predicted = SimilarityMotion {
            translation,
            rotation,
            scale_delta,
            ..SimilarityMotion::default()
        }
        .predict(item.previous, center);
        residual += (predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1]);
    }
    SimilarityMotion {
        translation,
        rotation,
        scale_delta,
        residual: residual * inverse,
        support: selected.len(),
    }
}

fn stable_similarity_prior(selected: &[&Match], center: [f32; 2]) -> SimilarityMotion {
    if selected.is_empty() {
        return SimilarityMotion::default();
    }
    let mut dx = selected
        .iter()
        .map(|item| item.current[0] - item.previous[0])
        .collect::<Vec<_>>();
    let mut dy = selected
        .iter()
        .map(|item| item.current[1] - item.previous[1])
        .collect::<Vec<_>>();
    let translation = [median(&mut dx), median(&mut dy)];
    let mut fitted = fit_similarity_selected(selected, center);
    let x_span = selected
        .iter()
        .map(|item| item.previous[0])
        .fold((f32::INFINITY, f32::NEG_INFINITY), |range, value| {
            (range.0.min(value), range.1.max(value))
        });
    let y_span = selected
        .iter()
        .map(|item| item.previous[1])
        .fold((f32::INFINITY, f32::NEG_INFINITY), |range, value| {
            (range.0.min(value), range.1.max(value))
        });
    // A compact patch of four edge tracks cannot independently determine
    // scale and rotation. Extrapolating that ill-conditioned fit to the far
    // side of the limbus was the direct reason those edges failed the layer
    // gate. Retain only translation until normal-flow constraints span it.
    if x_span.1 - x_span.0 < 28.0
        || y_span.1 - y_span.0 < 20.0
        || fitted.rotation.abs() > 0.08
        || fitted.scale_delta.abs() > 0.08
    {
        fitted = SimilarityMotion {
            translation,
            support: selected.len(),
            ..SimilarityMotion::default()
        };
        fitted.residual = selected
            .iter()
            .map(|item| {
                (item.current[0] - item.previous[0] - translation[0])
                    .hypot(item.current[1] - item.previous[1] - translation[1])
            })
            .sum::<f32>()
            / selected.len() as f32;
    }
    fitted
}

fn accumulate_normal_equation(
    matrix: &mut [[f64; 4]; 4],
    vector: &mut [f64; 4],
    row: [f64; 4],
    value: f64,
    weight: f64,
) {
    for column in 0..4 {
        vector[column] += weight * row[column] * value;
        for other in 0..4 {
            matrix[column][other] += weight * row[column] * row[other];
        }
    }
}

fn solve_four(mut matrix: [[f64; 4]; 4], mut vector: [f64; 4]) -> Option<[f64; 4]> {
    for pivot in 0..4 {
        let best = (pivot..4).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right][pivot].abs())
        })?;
        if matrix[best][pivot].abs() < 1.0e-8 {
            return None;
        }
        if best != pivot {
            matrix.swap(best, pivot);
            vector.swap(best, pivot);
        }
        let divisor = matrix[pivot][pivot];
        for column in pivot..4 {
            matrix[pivot][column] /= divisor;
        }
        vector[pivot] /= divisor;
        for row in 0..4 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..4 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            vector[row] -= factor * vector[pivot];
        }
    }
    Some(vector)
}

/// Fit a similarity transform from ordinary point tracks plus one-dimensional
/// normal-flow constraints. A smooth limbus edge contributes only n dot flow;
/// its arbitrary tangential patch match is deliberately excluded.
fn fit_similarity_with_normal_constraints(
    point_indices: &[usize],
    normal_indices: &[usize],
    radial_constraints: &[RadialLimbusFlow],
    matches: &[Match],
    tracks: &[FeatureTrack],
    center: [f32; 2],
) -> SimilarityMotion {
    let points = point_indices
        .iter()
        .map(|index| &matches[*index])
        .collect::<Vec<_>>();
    let prior = stable_similarity_prior(&points, center);
    if normal_indices.is_empty() && radial_constraints.is_empty() {
        return prior;
    }
    let mut matrix = [[0.0f64; 4]; 4];
    let mut vector = [0.0f64; 4];
    for item in &points {
        let x = (item.previous[0] - center[0]) as f64;
        let y = (item.previous[1] - center[1]) as f64;
        let dx = (item.current[0] - item.previous[0]) as f64;
        let dy = (item.current[1] - item.previous[1]) as f64;
        accumulate_normal_equation(&mut matrix, &mut vector, [1.0, 0.0, x, -y], dx, 1.0);
        accumulate_normal_equation(&mut matrix, &mut vector, [0.0, 1.0, y, x], dy, 1.0);
    }
    for index in normal_indices {
        let item = &matches[*index];
        let normal = normalized_vector(tracks[item.track_index].edge_normal);
        if normal[0].hypot(normal[1]) < 0.5 {
            continue;
        }
        let x = (item.previous[0] - center[0]) as f64;
        let y = (item.previous[1] - center[1]) as f64;
        let nx = normal[0] as f64;
        let ny = normal[1] as f64;
        let dx = (item.current[0] - item.previous[0]) as f64;
        let dy = (item.current[1] - item.previous[1]) as f64;
        accumulate_normal_equation(
            &mut matrix,
            &mut vector,
            [nx, ny, nx * x + ny * y, -nx * y + ny * x],
            nx * dx + ny * dy,
            0.80,
        );
    }
    for flow in radial_constraints {
        let normal = normalized_vector(flow.normal);
        if normal[0].hypot(normal[1]) < 0.5 {
            continue;
        }
        let x = (flow.previous[0] - center[0]) as f64;
        let y = (flow.previous[1] - center[1]) as f64;
        let nx = normal[0] as f64;
        let ny = normal[1] as f64;
        let dx = (flow.current[0] - flow.previous[0]) as f64;
        let dy = (flow.current[1] - flow.previous[1]) as f64;
        accumulate_normal_equation(
            &mut matrix,
            &mut vector,
            [nx, ny, nx * x + ny * y, -nx * y + ny * x],
            nx * dx + ny * dy,
            (0.30 + 0.35 * flow.confidence) as f64,
        );
    }
    let prior_values = [
        prior.translation[0] as f64,
        prior.translation[1] as f64,
        prior.scale_delta as f64,
        prior.rotation as f64,
    ];
    for parameter in 0..4 {
        let weight = if parameter < 2 { 0.35 } else { 900.0 };
        matrix[parameter][parameter] += weight;
        vector[parameter] += weight * prior_values[parameter];
    }
    let Some(solution) = solve_four(matrix, vector) else {
        return prior;
    };
    let mut fitted = SimilarityMotion {
        translation: [solution[0] as f32, solution[1] as f32],
        scale_delta: solution[2] as f32,
        rotation: solution[3] as f32,
        support: point_indices.len() + normal_indices.len() + radial_constraints.len(),
        ..SimilarityMotion::default()
    };
    if fitted.translation[0].hypot(fitted.translation[1]) > SEARCH_RADIUS as f32 * 1.75
        || fitted.rotation.abs() > 0.12
        || fitted.scale_delta.abs() > 0.12
    {
        return SimilarityMotion {
            support: fitted.support,
            ..prior
        };
    }
    let point_error = points
        .iter()
        .map(|item| {
            let predicted = fitted.predict(item.previous, center);
            (predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1])
        })
        .sum::<f32>();
    let normal_error = normal_indices
        .iter()
        .map(|index| {
            let item = &matches[*index];
            let normal = normalized_vector(tracks[item.track_index].edge_normal);
            let predicted = fitted.predict(item.previous, center);
            ((predicted[0] - item.current[0]) * normal[0]
                + (predicted[1] - item.current[1]) * normal[1])
                .abs()
        })
        .sum::<f32>();
    let radial_error = radial_constraints
        .iter()
        .map(|flow| {
            let predicted = fitted.predict(flow.previous, center);
            ((predicted[0] - flow.current[0]) * flow.normal[0]
                + (predicted[1] - flow.current[1]) * flow.normal[1])
                .abs()
        })
        .sum::<f32>();
    fitted.residual = (point_error + normal_error + radial_error) / fitted.support.max(1) as f32;
    fitted
}

fn fit_similarity(matches: &[Match], object: usize, center: [f32; 2]) -> SimilarityMotion {
    let selected = matches
        .iter()
        .filter(|item| item.object == object)
        .collect::<Vec<_>>();
    fit_similarity_selected(&selected, center)
}

fn robust_global_similarity(matches: &[Match], center: [f32; 2]) -> SimilarityMotion {
    if matches.is_empty() {
        return SimilarityMotion::default();
    }
    let mut dx = matches
        .iter()
        .map(|item| item.current[0] - item.previous[0])
        .collect::<Vec<_>>();
    let mut dy = matches
        .iter()
        .map(|item| item.current[1] - item.previous[1])
        .collect::<Vec<_>>();
    let translation = [median(&mut dx), median(&mut dy)];
    if matches.len() < 4 {
        return SimilarityMotion {
            translation,
            support: matches.len(),
            ..SimilarityMotion::default()
        };
    }

    let mut selected = matches.iter().collect::<Vec<_>>();
    let mut fitted = fit_similarity_selected(&selected, center);
    for _ in 0..2 {
        let mut residuals = selected
            .iter()
            .map(|item| {
                let predicted = fitted.predict(item.previous, center);
                (predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1])
            })
            .collect::<Vec<_>>();
        let robust_scale = median(&mut residuals).max(0.35);
        let cutoff = (robust_scale * 2.8).clamp(1.25, 6.0);
        selected = matches
            .iter()
            .filter(|item| {
                let predicted = fitted.predict(item.previous, center);
                (predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1]) <= cutoff
            })
            .collect();
        if selected.len() < 4 {
            break;
        }
        fitted = fit_similarity_selected(&selected, center);
    }
    if fitted.translation[0].hypot(fitted.translation[1]) > SEARCH_RADIUS as f32 * 1.75
        || fitted.rotation.abs() > 0.12
        || fitted.scale_delta.abs() > 0.12
    {
        SimilarityMotion {
            translation,
            support: matches.len(),
            ..SimilarityMotion::default()
        }
    } else {
        fitted
    }
}

const SHARED_NATIVE_SIMILARITY_INLIER_PX: f32 = 2.25;

fn shared_native_similarity_inliers(
    matches: &[Match],
    motion: SimilarityMotion,
    center: [f32; 2],
) -> Vec<usize> {
    matches
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let predicted = motion.predict(item.previous, center);
            ((predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1])
                <= SHARED_NATIVE_SIMILARITY_INLIER_PX)
                .then_some(index)
        })
        .collect()
}

fn shared_native_similarity_quality(
    matches: &[Match],
    inliers: &[usize],
    motion: SimilarityMotion,
    center: [f32; 2],
) -> f32 {
    if inliers.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut quadrants = [false; 4];
    let mut x_range = (f32::INFINITY, f32::NEG_INFINITY);
    let mut y_range = (f32::INFINITY, f32::NEG_INFINITY);
    let mut residual = 0.0f32;
    for index in inliers {
        let item = &matches[*index];
        let right = usize::from(item.previous[0] >= center[0]);
        let bottom = usize::from(item.previous[1] >= center[1]);
        quadrants[right | (bottom << 1)] = true;
        x_range.0 = x_range.0.min(item.previous[0]);
        x_range.1 = x_range.1.max(item.previous[0]);
        y_range.0 = y_range.0.min(item.previous[1]);
        y_range.1 = y_range.1.max(item.previous[1]);
        let predicted = motion.predict(item.previous, center);
        residual += (predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1]);
    }
    let occupied_quadrants = quadrants.into_iter().filter(|occupied| *occupied).count();
    let x_span = (x_range.1 - x_range.0).max(0.0);
    let y_span = (y_range.1 - y_range.0).max(0.0);
    let mean_residual = residual / inliers.len() as f32;
    // A coherent eyelid or glasses rim may have more corners than quiet skin,
    // but it occupies a compact strip. Reward true two-dimensional coverage
    // before a small support advantage, then prefer more and tighter inliers.
    let broad_bonus = if occupied_quadrants >= 3 { 6.0 } else { 0.0 };
    broad_bonus
        + occupied_quadrants as f32 * 1.25
        + inliers.len() as f32
        + (x_span / 128.0).min(4.0)
        + (y_span / 96.0).min(4.0)
        - mean_residual * 0.35
}

/// Deterministic bounded RANSAC for the segmentation-independent whole-ROI
/// scale authority. The ordinary least-squares initialization can be pulled
/// between skin, lid, iris, and glasses motions and leave no coherent inlier
/// cohort. Exhaustively testing the at-most-80 sparse feature pairs is cheap,
/// uses the borrowed native RAW matches directly, and lets the broadly
/// distributed rigid cohort win without treating a moving lid as head scale.
fn shared_native_robust_global_similarity(
    matches: &[Match],
    center: [f32; 2],
) -> (SimilarityMotion, Vec<usize>) {
    if matches.is_empty() {
        return (SimilarityMotion::default(), Vec::new());
    }
    let mut dx = matches
        .iter()
        .map(|item| item.current[0] - item.previous[0])
        .collect::<Vec<_>>();
    let mut dy = matches
        .iter()
        .map(|item| item.current[1] - item.previous[1])
        .collect::<Vec<_>>();
    let translation_only = SimilarityMotion {
        translation: [median(&mut dx), median(&mut dy)],
        ..SimilarityMotion::default()
    };
    let mut best_motion = translation_only;
    let mut best_inliers = shared_native_similarity_inliers(matches, best_motion, center);
    let mut best_quality =
        shared_native_similarity_quality(matches, &best_inliers, best_motion, center);

    for first_index in 0..matches.len() {
        for second_index in first_index + 1..matches.len() {
            let first = &matches[first_index];
            let second = &matches[second_index];
            let previous_delta = [
                second.previous[0] - first.previous[0],
                second.previous[1] - first.previous[1],
            ];
            let denominator =
                previous_delta[0] * previous_delta[0] + previous_delta[1] * previous_delta[1];
            if denominator < 48.0 * 48.0 {
                continue;
            }
            let current_delta = [
                second.current[0] - first.current[0],
                second.current[1] - first.current[1],
            ];
            let scale_delta = (previous_delta[0] * current_delta[0]
                + previous_delta[1] * current_delta[1])
                / denominator
                - 1.0;
            let rotation = (previous_delta[0] * current_delta[1]
                - previous_delta[1] * current_delta[0])
                / denominator;
            if !scale_delta.is_finite()
                || !rotation.is_finite()
                || scale_delta.abs() > 0.10
                || rotation.abs() > 0.12
            {
                continue;
            }
            let translation_for = |item: &Match| {
                let x = item.previous[0] - center[0];
                let y = item.previous[1] - center[1];
                [
                    item.current[0] - item.previous[0] - scale_delta * x + rotation * y,
                    item.current[1] - item.previous[1] - rotation * x - scale_delta * y,
                ]
            };
            let first_translation = translation_for(first);
            let second_translation = translation_for(second);
            let motion = SimilarityMotion {
                translation: [
                    0.5 * (first_translation[0] + second_translation[0]),
                    0.5 * (first_translation[1] + second_translation[1]),
                ],
                scale_delta,
                rotation,
                ..SimilarityMotion::default()
            };
            if motion.translation[0].hypot(motion.translation[1]) > 32.0 {
                continue;
            }
            let inliers = shared_native_similarity_inliers(matches, motion, center);
            if inliers.len() < 4 {
                continue;
            }
            let quality = shared_native_similarity_quality(matches, &inliers, motion, center);
            if quality > best_quality {
                best_motion = motion;
                best_inliers = inliers;
                best_quality = quality;
            }
        }
    }

    // Refit all members of the winning cohort, but retain the pair hypothesis
    // if least squares pulls toward newly admitted boundary outliers.
    for _ in 0..2 {
        if best_inliers.len() < 4 {
            break;
        }
        let selected = best_inliers
            .iter()
            .map(|index| &matches[*index])
            .collect::<Vec<_>>();
        let fitted = fit_similarity_selected(&selected, center);
        let fitted_inliers = shared_native_similarity_inliers(matches, fitted, center);
        let fitted_quality =
            shared_native_similarity_quality(matches, &fitted_inliers, fitted, center);
        if fitted_quality + 1.0e-4 < best_quality {
            break;
        }
        best_motion = fitted;
        best_inliers = fitted_inliers;
        best_quality = fitted_quality;
    }
    if !best_inliers.is_empty() {
        best_motion.support = best_inliers.len();
        best_motion.residual = best_inliers
            .iter()
            .map(|index| {
                let item = &matches[*index];
                let predicted = best_motion.predict(item.previous, center);
                (predicted[0] - item.current[0]).hypot(predicted[1] - item.current[1])
            })
            .sum::<f32>()
            / best_inliers.len() as f32;
    }
    (best_motion, best_inliers)
}

fn motion_delta(motion: SimilarityMotion, point: [f32; 2], center: [f32; 2]) -> [f32; 2] {
    let predicted = motion.predict(point, center);
    [predicted[0] - point[0], predicted[1] - point[1]]
}

fn residual_motion(item: &Match, global: SimilarityMotion, center: [f32; 2]) -> [f32; 2] {
    let observed = [
        item.current[0] - item.previous[0],
        item.current[1] - item.previous[1],
    ];
    let background = motion_delta(global, item.previous, center);
    [observed[0] - background[0], observed[1] - background[1]]
}

fn motion_signature(track: &FeatureTrack, current: [f32; 2]) -> Vec<[f32; 2]> {
    let history_to_keep = MOTION_SIGNATURE_LEN.saturating_sub(1);
    let skip = track.residual_history.len().saturating_sub(history_to_keep);
    track
        .residual_history
        .iter()
        .skip(skip)
        .copied()
        .chain(std::iter::once(current))
        .collect()
}

/// RMS trajectory distance, aligning signatures at their newest sample. This
/// measures whether two edge tracks have moved together over several frames;
/// two unrelated edges that happen to share one displacement do not collapse
/// into the same layer.
fn signature_distance(left: &[[f32; 2]], right: &[[f32; 2]]) -> f32 {
    let overlap = left.len().min(right.len());
    if overlap == 0 {
        return f32::INFINITY;
    }
    let left_start = left.len() - overlap;
    let right_start = right.len() - overlap;
    let mut weighted_error = 0.0f32;
    let mut weight_sum = 0.0f32;
    for index in 0..overlap {
        let weight = 0.55 + 0.45 * (index + 1) as f32 / overlap as f32;
        let dx = left[left_start + index][0] - right[right_start + index][0];
        let dy = left[left_start + index][1] - right[right_start + index][1];
        weighted_error += weight * (dx * dx + dy * dy);
        weight_sum += weight;
    }
    (weighted_error / weight_sum.max(1.0e-6)).sqrt()
}

fn normal_signature_distance(left: &[[f32; 2]], right: &[[f32; 2]], normal: [f32; 2]) -> f32 {
    let normal = normalized_vector(normal);
    let overlap = left.len().min(right.len());
    if overlap == 0 || normal[0].hypot(normal[1]) < 0.5 {
        return f32::INFINITY;
    }
    let left_start = left.len() - overlap;
    let right_start = right.len() - overlap;
    let mut weighted_error = 0.0f32;
    let mut weight_sum = 0.0f32;
    for index in 0..overlap {
        let weight = 0.55 + 0.45 * (index + 1) as f32 / overlap as f32;
        let difference = [
            left[left_start + index][0] - right[right_start + index][0],
            left[left_start + index][1] - right[right_start + index][1],
        ];
        let error = difference[0] * normal[0] + difference[1] * normal[1];
        weighted_error += weight * error * error;
        weight_sum += weight;
    }
    (weighted_error / weight_sum.max(1.0e-6)).sqrt()
}

fn signature_centroid<'a>(signatures: impl Iterator<Item = &'a Vec<[f32; 2]>>) -> Vec<[f32; 2]> {
    let signatures = signatures.collect::<Vec<_>>();
    let length = signatures
        .iter()
        .map(|signature| signature.len())
        .max()
        .unwrap_or(0)
        .min(MOTION_SIGNATURE_LEN);
    let mut newest_first = Vec::with_capacity(length);
    for recency in 0..length {
        let mut sum = [0.0f32; 2];
        let mut count = 0usize;
        for signature in &signatures {
            if recency < signature.len() {
                let sample = signature[signature.len() - 1 - recency];
                sum[0] += sample[0];
                sum[1] += sample[1];
                count += 1;
            }
        }
        if count > 0 {
            newest_first.push([sum[0] / count as f32, sum[1] / count as f32]);
        }
    }
    newest_first.reverse();
    newest_first
}

#[derive(Clone, Debug)]
struct SignatureCandidate {
    match_index: usize,
    samples: Vec<[f32; 2]>,
}

#[derive(Clone, Debug)]
struct SignatureCluster {
    members: Vec<usize>, // indices into SignatureCandidate
    centroid: Vec<[f32; 2]>,
    within_error: f32,
}

fn recompute_signature_cluster(cluster: &mut SignatureCluster, candidates: &[SignatureCandidate]) {
    cluster.centroid = signature_centroid(
        cluster
            .members
            .iter()
            .map(|index| &candidates[*index].samples),
    );
    cluster.within_error = if cluster.members.is_empty() {
        f32::INFINITY
    } else {
        cluster
            .members
            .iter()
            .map(|index| signature_distance(&candidates[*index].samples, &cluster.centroid))
            .sum::<f32>()
            / cluster.members.len() as f32
    };
}

fn retain_cohesive_signature_core(
    cluster: &mut SignatureCluster,
    candidates: &[SignatureCandidate],
) {
    if cluster.members.len() < MIN_LAYER_SUPPORT {
        cluster.members.clear();
        return;
    }
    // K-means must not turn every tracked edge into physical-layer evidence.
    // Find the densest trajectory core first, then trim against its centroid;
    // unrelated aperture-problem tracks remain provisional instead of
    // inflating a layer's error and suppressing every coherent member.
    let anchor = cluster.members.iter().copied().max_by(|left, right| {
        let density = |candidate_index: usize| {
            let distances = cluster
                .members
                .iter()
                .map(|other| {
                    signature_distance(
                        &candidates[candidate_index].samples,
                        &candidates[*other].samples,
                    )
                })
                .collect::<Vec<_>>();
            let neighbors = distances
                .iter()
                .filter(|distance| **distance <= MAX_SIGNATURE_MEMBER_ERROR)
                .count();
            let mean = distances.iter().sum::<f32>() / distances.len().max(1) as f32;
            (neighbors, -mean)
        };
        let left_density = density(*left);
        let right_density = density(*right);
        left_density
            .0
            .cmp(&right_density.0)
            .then_with(|| left_density.1.total_cmp(&right_density.1))
    });
    let Some(anchor) = anchor else {
        cluster.members.clear();
        return;
    };
    cluster.members.retain(|member| {
        signature_distance(&candidates[*member].samples, &candidates[anchor].samples)
            <= MAX_SIGNATURE_MEMBER_ERROR
    });
    if cluster.members.len() < MIN_LAYER_SUPPORT {
        cluster.members.clear();
        return;
    }
    for _ in 0..2 {
        recompute_signature_cluster(cluster, candidates);
        let centroid = cluster.centroid.clone();
        cluster.members.retain(|member| {
            signature_distance(&candidates[*member].samples, &centroid)
                <= MAX_SIGNATURE_MEMBER_ERROR
        });
        if cluster.members.len() < MIN_LAYER_SUPPORT {
            cluster.members.clear();
            return;
        }
    }
    recompute_signature_cluster(cluster, candidates);
}

fn cluster_signatures(candidates: &[SignatureCandidate]) -> Vec<SignatureCluster> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let maximum_clusters = (candidates.len() / MIN_LAYER_SUPPORT).max(1).min(OBJECTS);
    let energy = |signature: &[[f32; 2]]| {
        (signature
            .iter()
            .map(|sample| sample[0] * sample[0] + sample[1] * sample[1])
            .sum::<f32>()
            / signature.len().max(1) as f32)
            .sqrt()
    };
    // The least residual motion is the robust background seed. Additional
    // layers must be separated over the whole trajectory, not just today.
    let first = candidates
        .iter()
        .enumerate()
        .min_by(|left, right| energy(&left.1.samples).total_cmp(&energy(&right.1.samples)))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let mut centroids = vec![candidates[first].samples.clone()];
    while centroids.len() < maximum_clusters {
        let Some((separation, candidate)) = candidates
            .iter()
            .map(|candidate| {
                let nearest = centroids
                    .iter()
                    .map(|centroid| signature_distance(&candidate.samples, centroid))
                    .fold(f32::INFINITY, f32::min);
                (nearest, candidate.samples.clone())
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
        else {
            break;
        };
        if separation < MIN_SIGNATURE_SEED_SEPARATION {
            break;
        }
        centroids.push(candidate);
    }

    let mut labels = vec![0usize; candidates.len()];
    for _ in 0..7 {
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            labels[candidate_index] = centroids
                .iter()
                .enumerate()
                .min_by(|left, right| {
                    signature_distance(&candidate.samples, left.1)
                        .total_cmp(&signature_distance(&candidate.samples, right.1))
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
        }
        for (cluster_index, centroid) in centroids.iter_mut().enumerate() {
            let replacement = signature_centroid(
                candidates
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| labels[*index] == cluster_index)
                    .map(|(_, candidate)| &candidate.samples),
            );
            if !replacement.is_empty() {
                *centroid = replacement;
            }
        }
    }

    let mut clusters = centroids
        .into_iter()
        .enumerate()
        .map(|(cluster_index, centroid)| SignatureCluster {
            members: labels
                .iter()
                .enumerate()
                .filter_map(|(candidate_index, label)| {
                    (*label == cluster_index).then_some(candidate_index)
                })
                .collect(),
            centroid,
            within_error: 0.0,
        })
        .filter(|cluster| !cluster.members.is_empty())
        .collect::<Vec<_>>();

    for cluster in &mut clusters {
        retain_cohesive_signature_core(cluster, candidates);
    }
    clusters.retain(|cluster| cluster.members.len() >= MIN_LAYER_SUPPORT);
    clusters
}

#[derive(Clone, Copy, Debug)]
struct EyeMotionRegion {
    center: [f32; 2],
    major: f32,
    minor: f32,
    angle: f32,
}

impl EyeMotionRegion {
    fn contains(self, point: [f32; 2]) -> bool {
        self.normalized_radius(point) <= 1.0
    }

    fn contains_scaled(self, point: [f32; 2], scale: f32) -> bool {
        self.normalized_radius(point) <= scale
    }

    fn normalized_radius(self, point: [f32; 2]) -> f32 {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let (sine, cosine) = self.angle.sin_cos();
        let x = cosine * dx + sine * dy;
        let y = -sine * dx + cosine * dy;
        (x / self.major.max(1.0)).hypot(y / self.minor.max(1.0))
    }

    fn outward_normal(self, point: [f32; 2]) -> [f32; 2] {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let (sine, cosine) = self.angle.sin_cos();
        let local_x = cosine * dx + sine * dy;
        let local_y = -sine * dx + cosine * dy;
        normalized_vector([
            cosine * local_x / self.major.max(1.0).powi(2)
                - sine * local_y / self.minor.max(1.0).powi(2),
            sine * local_x / self.major.max(1.0).powi(2)
                + cosine * local_y / self.minor.max(1.0).powi(2),
        ])
    }

    fn local_seed(self, frame: &RawFrame) -> IrisEllipseSeed {
        IrisEllipseSeed {
            center: (
                (self.center[0] - frame.sensor_x as f32) as f64,
                (self.center[1] - frame.sensor_y as f32) as f64,
            ),
            major_radius: self.major as f64,
            minor_radius: self.minor as f64,
            angle: self.angle as f64,
        }
    }

    fn from_local_seed(seed: IrisEllipseSeed, frame: &RawFrame) -> Self {
        Self {
            center: [
                seed.center.0 as f32 + frame.sensor_x as f32,
                seed.center.1 as f32 + frame.sensor_y as f32,
            ],
            major: seed.major_radius as f32,
            minor: seed.minor_radius as f32,
            angle: seed.angle as f32,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RadialLimbusProfile {
    normalized: [f32; RADIAL_LIMBUS_SAMPLES],
    gradients: [f32; RADIAL_LIMBUS_GRADIENTS],
    signed_step: f32,
    quality: f32,
}

#[derive(Clone, Copy, Debug)]
struct RadialLimbusFlow {
    previous: [f32; 2],
    current: [f32; 2],
    normal: [f32; 2],
    phase_rad: f32,
    radial_shift_px: f32,
    profile_cost: f32,
    confidence: f32,
    fused: bool,
}

fn eye_region_boundary(region: EyeMotionRegion, phase: f32) -> [f32; 2] {
    let (phase_sine, phase_cosine) = phase.sin_cos();
    let (axis_sine, axis_cosine) = region.angle.sin_cos();
    let local_x = region.major * phase_cosine;
    let local_y = region.minor * phase_sine;
    [
        region.center[0] + axis_cosine * local_x - axis_sine * local_y,
        region.center[1] + axis_sine * local_x + axis_cosine * local_y,
    ]
}

fn radial_limbus_profile(
    frame: &RawFrame,
    point_sensor: [f32; 2],
    normal: [f32; 2],
) -> Option<RadialLimbusProfile> {
    let normal = normalized_vector(normal);
    if normal[0].hypot(normal[1]) < 0.5 {
        return None;
    }
    let tangent = [-normal[1], normal[0]];
    let local = [
        point_sensor[0] - frame.sensor_x as f32,
        point_sensor[1] - frame.sensor_y as f32,
    ];
    let mut samples = [0.0f32; RADIAL_LIMBUS_SAMPLES];
    for (sample_index, sample) in samples.iter_mut().enumerate() {
        let radial_offset = (sample_index as f32 - RADIAL_LIMBUS_HALF_SAMPLES as f32)
            * RADIAL_LIMBUS_SAMPLE_SPACING_PX;
        let radial_point = [
            local[0] + normal[0] * radial_offset,
            local[1] + normal[1] * radial_offset,
        ];
        // A narrow three-lane average suppresses individual iris fibres and
        // residual CFA phase while retaining the native radial transition.
        // It reads the existing RAW allocation directly; no band image or
        // resized representation is materialized.
        let mut sum = 0.0f32;
        for lateral in [-1.25f32, 0.0, 1.25] {
            sum += sample_native_raw_bilinear(
                frame,
                radial_point[0] + tangent[0] * lateral,
                radial_point[1] + tangent[1] * lateral,
            )?;
        }
        *sample = sum / 3.0;
    }
    let mean = samples.iter().sum::<f32>() / RADIAL_LIMBUS_SAMPLES as f32;
    let variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f32>()
        / RADIAL_LIMBUS_SAMPLES as f32;
    let deviation = variance.sqrt();
    if !deviation.is_finite() || deviation < 2.0 {
        return None;
    }
    let mut normalized = [0.0f32; RADIAL_LIMBUS_SAMPLES];
    for (destination, sample) in normalized.iter_mut().zip(samples) {
        *destination = (sample - mean) / deviation;
    }
    let inner_mean = normalized[..RADIAL_LIMBUS_HALF_SAMPLES].iter().sum::<f32>()
        / RADIAL_LIMBUS_HALF_SAMPLES as f32;
    let outer_mean = normalized[RADIAL_LIMBUS_HALF_SAMPLES + 1..]
        .iter()
        .sum::<f32>()
        / RADIAL_LIMBUS_HALF_SAMPLES as f32;
    let signed_step = outer_mean - inner_mean;
    // Iris-to-sclera evidence is a dark-to-bright outward transition. A very
    // weak positive step is retained because oblique semi-opaque tissue can
    // make it broad and gray, but a polarity reversal is usually lid/lash.
    if !signed_step.is_finite() || signed_step < 0.05 {
        return None;
    }
    let mut gradients = [0.0f32; RADIAL_LIMBUS_GRADIENTS];
    for index in 0..RADIAL_LIMBUS_GRADIENTS {
        gradients[index] = normalized[index + 1] - normalized[index];
    }
    let gradient_energy = gradients.iter().map(|value| value * value).sum::<f32>();
    if gradient_energy < 0.025 {
        return None;
    }
    let inverse_gradient_norm = gradient_energy.sqrt().recip();
    for gradient in &mut gradients {
        *gradient *= inverse_gradient_norm;
    }
    let positive_peak = normalized
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(0.0f32, f32::max);
    let contrast_quality = ((deviation - 2.0) / 24.0).clamp(0.0, 1.0);
    let step_quality = ((signed_step - 0.05) / 0.75).clamp(0.0, 1.0);
    let peak_quality = ((positive_peak - 0.05) / 0.70).clamp(0.0, 1.0);
    let quality = contrast_quality.sqrt() * (0.20 + 0.50 * step_quality + 0.30 * peak_quality);
    Some(RadialLimbusProfile {
        normalized,
        gradients,
        signed_step,
        quality: quality.clamp(0.0, 1.0),
    })
}

fn radial_limbus_profile_cost(
    reference: RadialLimbusProfile,
    candidate: RadialLimbusProfile,
) -> f32 {
    let gradient_correlation = reference
        .gradients
        .iter()
        .zip(candidate.gradients)
        .map(|(left, right)| left * right)
        .sum::<f32>()
        .clamp(-1.0, 1.0);
    let gradient_cost = 0.5 * (1.0 - gradient_correlation);
    let shape_rms = (reference
        .normalized
        .iter()
        .zip(candidate.normalized)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f32>()
        / RADIAL_LIMBUS_SAMPLES as f32)
        .sqrt();
    let step_cost = (reference.signed_step - candidate.signed_step).abs() / 2.5;
    (0.68 * gradient_cost + 0.22 * (shape_rms / 2.0).min(1.0) + 0.10 * step_cost.min(1.0))
        .clamp(0.0, 1.0)
}

fn radial_limbus_flows(
    previous: &RawFrame,
    current: &RawFrame,
    previous_region: EyeMotionRegion,
    current_region: EyeMotionRegion,
) -> (Vec<RadialLimbusFlow>, usize, u64) {
    let started = Instant::now();
    let valid_region = |region: EyeMotionRegion| {
        region.center[0].is_finite()
            && region.center[1].is_finite()
            && region.major.is_finite()
            && region.minor.is_finite()
            && region.major >= 12.0
            && region.minor >= 8.0
    };
    if !valid_region(previous_region) || !valid_region(current_region) {
        return (Vec::new(), 0, started.elapsed().as_micros() as u64);
    }
    let mut flows = Vec::with_capacity(RADIAL_LIMBUS_SECTORS);
    let mut evaluations = 0usize;
    for sector in 0..RADIAL_LIMBUS_SECTORS {
        let phase = std::f32::consts::TAU * sector as f32 / RADIAL_LIMBUS_SECTORS as f32;
        let previous_base = eye_region_boundary(previous_region, phase);
        let previous_normal = previous_region.outward_normal(previous_base);
        let mut reference = None::<([f32; 2], RadialLimbusProfile, f32)>;
        for shift in -RADIAL_LIMBUS_REFERENCE_SEARCH_PX..=RADIAL_LIMBUS_REFERENCE_SEARCH_PX {
            let point = [
                previous_base[0] + previous_normal[0] * shift as f32,
                previous_base[1] + previous_normal[1] * shift as f32,
            ];
            let Some(profile) = radial_limbus_profile(previous, point, previous_normal) else {
                continue;
            };
            let rank = profile.quality - 0.012 * shift.abs() as f32;
            if reference.is_none_or(|prior| rank > prior.2) {
                reference = Some((point, profile, rank));
            }
        }
        let Some((previous_point, reference_profile, _)) = reference else {
            continue;
        };
        if reference_profile.quality < RADIAL_LIMBUS_MIN_QUALITY {
            continue;
        }
        let current_base = eye_region_boundary(current_region, phase);
        let current_normal = current_region.outward_normal(current_base);
        let current_tangent = [-current_normal[1], current_normal[0]];
        let mut candidates = Vec::<(f32, f32, f32, [f32; 2], RadialLimbusProfile)>::new();
        for tangent_shift in [-2.0f32, 0.0, 2.0] {
            for radial_shift in -RADIAL_LIMBUS_CURRENT_SEARCH_PX..=RADIAL_LIMBUS_CURRENT_SEARCH_PX {
                let point = [
                    current_base[0]
                        + current_normal[0] * radial_shift as f32
                        + current_tangent[0] * tangent_shift,
                    current_base[1]
                        + current_normal[1] * radial_shift as f32
                        + current_tangent[1] * tangent_shift,
                ];
                let Some(profile) = radial_limbus_profile(current, point, current_normal) else {
                    continue;
                };
                evaluations += 1;
                if profile.quality < RADIAL_LIMBUS_MIN_QUALITY * 0.70 {
                    continue;
                }
                let appearance = radial_limbus_profile_cost(reference_profile, profile);
                let search_penalty = 0.012
                    * (radial_shift as f32 / RADIAL_LIMBUS_CURRENT_SEARCH_PX as f32).powi(2)
                    + 0.006 * (tangent_shift / 2.0).powi(2);
                candidates.push((
                    appearance + search_penalty,
                    radial_shift as f32,
                    tangent_shift,
                    point,
                    profile,
                ));
            }
        }
        candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
        let Some(best) = candidates.first().copied() else {
            continue;
        };
        if best.0 > RADIAL_LIMBUS_MAX_PROFILE_COST {
            continue;
        }
        let second = candidates
            .iter()
            .find(|candidate| {
                (candidate.1 - best.1).abs() >= 2.0 || (candidate.2 - best.2).abs() >= 2.0
            })
            .map_or(1.0, |candidate| candidate.0);
        let margin = (second - best.0).max(0.0);
        let appearance_confidence = (1.0 - best.0 / RADIAL_LIMBUS_MAX_PROFILE_COST).clamp(0.0, 1.0);
        let quality = (reference_profile.quality * best.4.quality).sqrt();
        let confidence = (0.58 * appearance_confidence
            + 0.16 * (margin / 0.10).clamp(0.0, 1.0)
            + 0.26 * quality)
            .clamp(0.0, 1.0);
        if confidence < 0.26 {
            continue;
        }
        let mut combined_normal = current_normal;
        if previous_normal[0] * current_normal[0] + previous_normal[1] * current_normal[1] > 0.0 {
            combined_normal = normalized_vector([
                previous_normal[0] + current_normal[0],
                previous_normal[1] + current_normal[1],
            ]);
        }
        flows.push(RadialLimbusFlow {
            previous: previous_point,
            current: best.3,
            normal: combined_normal,
            phase_rad: phase,
            radial_shift_px: best.1,
            profile_cost: best.0,
            confidence,
            fused: false,
        });
    }
    if flows.len() < RADIAL_LIMBUS_MIN_JOINT_SUPPORT {
        flows.clear();
    }
    (flows, evaluations, started.elapsed().as_micros() as u64)
}

fn blend_ellipse_angle(previous: f32, current: f32, alpha: f32) -> f32 {
    let previous_vector = [(2.0 * previous).cos(), (2.0 * previous).sin()];
    let current_vector = [(2.0 * current).cos(), (2.0 * current).sin()];
    (0.5 * ((1.0 - alpha) * previous_vector[1] + alpha * current_vector[1])
        .atan2((1.0 - alpha) * previous_vector[0] + alpha * current_vector[0]))
    .rem_euclid(std::f32::consts::PI)
}

/// Recover a label-free elliptical iris region from the compact glint center
/// and signed Canny evidence. The glint supplies only a bounded center prior;
/// ellipse shape and orientation come from the current RAW edge field. A
/// previous region is used as the next bounded seed, not as current evidence.
fn semantic_eye_region(
    frame: &RawFrame,
    edges: &[EdgeEvidence],
    anatomical_seed: Option<IrisEllipseSeed>,
    observed_center: [f32; 2],
    previous: Option<EyeMotionRegion>,
) -> EyeMotionRegion {
    if let Some(seed) = anatomical_seed {
        return EyeMotionRegion::from_local_seed(seed, frame);
    }
    let extent = frame.width.min(frame.height) as f32;
    let rough = EyeMotionRegion {
        center: observed_center,
        major: previous
            .map_or(extent * 0.35, |region| region.major)
            .clamp(extent * 0.27, extent * 0.40),
        minor: previous
            .map_or(extent * 0.25, |region| region.minor)
            .clamp(extent * 0.18, extent * 0.31),
        angle: previous.map_or(0.0, |region| region.angle),
    };
    let seed = rough.local_seed(frame);
    let fitted =
        fit_edge_ellipse(edges, seed, frame.width, frame.height).map(|(ellipse, _, _, _)| {
            EyeMotionRegion {
                center: [
                    ellipse.center.0 as f32 + frame.sensor_x as f32,
                    ellipse.center.1 as f32 + frame.sensor_y as f32,
                ],
                major: ellipse.major as f32,
                minor: ellipse.minor as f32,
                angle: ellipse.angle as f32,
            }
        });
    let Some(mut fitted) = fitted else {
        return rough;
    };
    fitted.center = [
        0.72 * fitted.center[0] + 0.28 * observed_center[0],
        0.72 * fitted.center[1] + 0.28 * observed_center[1],
    ];
    fitted.major = fitted.major.clamp(extent * 0.27, extent * 0.40);
    fitted.minor = fitted.minor.clamp(extent * 0.18, extent * 0.31);
    previous.map_or(fitted, |prior| EyeMotionRegion {
        center: fitted.center,
        major: 0.62 * rough.major + 0.38 * fitted.major,
        minor: 0.62 * rough.minor + 0.38 * fitted.minor,
        angle: blend_ellipse_angle(prior.angle, fitted.angle, 0.38),
    })
}

fn semantic_motion_core(
    candidates: &[usize],
    matches: &[Match],
    global: SimilarityMotion,
    center: [f32; 2],
    minimum_support: usize,
) -> Vec<usize> {
    let residuals = candidates
        .iter()
        .map(|index| (*index, residual_motion(&matches[*index], global, center)))
        .collect::<Vec<_>>();
    let Some(anchor) = residuals.iter().max_by(|left, right| {
        let rank = |candidate: &(usize, [f32; 2])| {
            let neighbors = residuals
                .iter()
                .filter(|other| {
                    (candidate.1[0] - other.1[0]).hypot(candidate.1[1] - other.1[1])
                        <= SEMANTIC_MOTION_CORE_RADIUS
                })
                .count();
            (neighbors, matches[candidate.0].score)
        };
        let left_rank = rank(left);
        let right_rank = rank(right);
        left_rank
            .0
            .cmp(&right_rank.0)
            .then_with(|| left_rank.1.total_cmp(&right_rank.1))
    }) else {
        return Vec::new();
    };
    let selected = residuals
        .iter()
        .filter_map(|(index, residual)| {
            ((residual[0] - anchor.1[0]).hypot(residual[1] - anchor.1[1])
                <= SEMANTIC_MOTION_CORE_RADIUS)
                .then_some(*index)
        })
        .collect::<Vec<_>>();
    if selected.len() >= minimum_support {
        selected
    } else {
        Vec::new()
    }
}

/// Select an iris-material component from the pairwise relation graph.  The
/// graph supplies kinematics; the elliptical region supplies topology.  A
/// global/lash component that happens to cross the iris is penalized because
/// most of its nodes live outside the candidate region, while a genuinely
/// rotating iris component remains spatially compact and two-dimensional.
fn relation_graph_iris_core(
    candidates: &[usize],
    matches: &[Match],
    tracks: &[FeatureTrack],
    global: SimilarityMotion,
    center: [f32; 2],
    eye_region: EyeMotionRegion,
    relations: &mut MotionRelationFrame,
    iris_identity: Option<&PersistentRelationIrisIdentity>,
    minimum_support: usize,
) -> Vec<usize> {
    let candidate_mask = candidates
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut ranked = Vec::new();
    let mut identity_switch_rejections = 0usize;
    let mut initial_origin_rejections = 0usize;
    let mut candidate_diagnostics = RelationIrisCandidateDiagnostics {
        selector_calls: 1,
        ..RelationIrisCandidateDiagnostics::default()
    };
    let identity_active = iris_identity.is_some_and(PersistentRelationIrisIdentity::active);
    for (component_index, component) in relations.components.iter().enumerate() {
        candidate_diagnostics.components_examined += 1;
        let mut continuity = iris_identity
            .map(|identity| identity.continuity(component, eye_region))
            .unwrap_or_default();
        let persistent_component = relation_component_is_persistent(component);
        let identity_carried = identity_active
            && continuity.compatible
            && continuity.track_overlap >= RELATION_IRIS_IDENTITY_MIN_OVERLAP;
        if component.coherence < 0.32 || (!persistent_component && !identity_carried) {
            candidate_diagnostics.rejected_component_quality += 1;
            continue;
        }
        let selected = component
            .members
            .iter()
            .copied()
            .filter(|index| candidate_mask.contains(index))
            .collect::<Vec<_>>();
        if selected.len() < minimum_support {
            candidate_diagnostics.rejected_spatial_support += 1;
            continue;
        }
        let purity = selected.len() as f32 / component.members.len().max(1) as f32;
        candidate_diagnostics.maximum_selected_support = candidate_diagnostics
            .maximum_selected_support
            .max(selected.len());
        candidate_diagnostics.maximum_purity = candidate_diagnostics.maximum_purity.max(purity);
        // A component that is mostly lid/skin with a few tracks crossing the
        // projected iris is exactly the false cyan bridge this gate rejects.
        if purity < RELATION_IRIS_MIN_COMPONENT_PURITY {
            candidate_diagnostics.rejected_purity += 1;
            continue;
        }
        let prior_pupil = selected
            .iter()
            .filter(|index| tracks[matches[**index].track_index].object == PUPIL_LAYER)
            .count() as f32
            / selected.len() as f32;
        let interior = selected
            .iter()
            .map(|index| {
                (1.18 - eye_region.normalized_radius(matches[*index].current)).clamp(0.0, 1.0)
            })
            .sum::<f32>()
            / selected.len() as f32;
        let differential = selected
            .iter()
            .map(|index| {
                let value = residual_motion(&matches[*index], global, center);
                value[0].hypot(value[1])
            })
            .sum::<f32>()
            / selected.len() as f32;
        candidate_diagnostics.maximum_differential_px = candidate_diagnostics
            .maximum_differential_px
            .max(differential);

        // Normalize into the proposed ellipse before measuring whether the
        // material has two-dimensional support. A lash row is nearly rank-1;
        // iris texture/limbus points normally span both normalized axes.
        let (sine, cosine) = eye_region.angle.sin_cos();
        let normalized = selected
            .iter()
            .map(|index| {
                let point = matches[*index].current;
                let dx = point[0] - eye_region.center[0];
                let dy = point[1] - eye_region.center[1];
                [
                    (cosine * dx + sine * dy) / eye_region.major.max(1.0),
                    (-sine * dx + cosine * dy) / eye_region.minor.max(1.0),
                ]
            })
            .collect::<Vec<_>>();
        let inverse = 1.0 / normalized.len() as f32;
        let mean = [
            normalized.iter().map(|point| point[0]).sum::<f32>() * inverse,
            normalized.iter().map(|point| point[1]).sum::<f32>() * inverse,
        ];
        let covariance = normalized.iter().fold([0.0f32; 3], |mut sum, point| {
            let x = point[0] - mean[0];
            let y = point[1] - mean[1];
            sum[0] += x * x;
            sum[1] += x * y;
            sum[2] += y * y;
            sum
        });
        let xx = covariance[0] * inverse;
        let xy = covariance[1] * inverse;
        let yy = covariance[2] * inverse;
        let trace = xx + yy;
        let determinant = (xx * yy - xy * xy).max(0.0);
        let two_dimensionality = if trace > 1.0e-5 {
            (4.0 * determinant / (trace * trace)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        candidate_diagnostics.maximum_two_dimensionality = candidate_diagnostics
            .maximum_two_dimensionality
            .max(two_dimensionality);
        // A coherent eyelash row can cross the projected iris and can move
        // rapidly during a blink, but it remains spatially rank-1. Likewise a
        // carrier-motion component should not become iris material merely
        // because its tracks persist. Require both genuine differential motion
        // and a small but observable two-dimensional material footprint.
        if differential < RELATION_IRIS_MIN_DIFFERENTIAL_PX {
            candidate_diagnostics.rejected_differential += 1;
            continue;
        }
        if two_dimensionality < 0.045 {
            candidate_diagnostics.rejected_rank_one += 1;
            continue;
        }
        let radius = eye_region.major.max(eye_region.minor).max(8.0);
        let origin_offset_radii = component.origin_valid.then(|| {
            (component.shared_origin[0] - eye_region.center[0])
                .hypot(component.shared_origin[1] - eye_region.center[1])
                / radius
        });
        let current_origin_plausible = origin_offset_radii
            .is_some_and(|offset| offset <= RELATION_IRIS_INITIAL_MAX_ORIGIN_OFFSET_RADII);
        if let Some(offset) = origin_offset_radii {
            if candidate_diagnostics.finite_origin_candidates == 0 {
                candidate_diagnostics.minimum_origin_offset_radii = offset;
            } else {
                candidate_diagnostics.minimum_origin_offset_radii = candidate_diagnostics
                    .minimum_origin_offset_radii
                    .min(offset);
            }
            candidate_diagnostics.maximum_origin_offset_radii = candidate_diagnostics
                .maximum_origin_offset_radii
                .max(offset);
            candidate_diagnostics.finite_origin_candidates += 1;
        }
        // An identity first learned while its pivot was unobservable must not
        // later acquire a finite but anatomically remote pivot by default.
        // Exact material overlap may continue accumulating, but `observe`
        // sees this as an origin inconsistency and leaves the pivot unset.
        if identity_active
            && iris_identity.is_some_and(|identity| !identity.origin_valid)
            && component.origin_valid
            && !current_origin_plausible
        {
            continuity.origin_consistent = false;
        }
        if !continuity.compatible {
            identity_switch_rejections += 1;
            candidate_diagnostics.rejected_identity += 1;
            continue;
        }
        if !identity_active {
            if component.origin_valid {
                if candidate_diagnostics.initial_origin_candidates == 0 {
                    candidate_diagnostics.minimum_initial_origin_offset_radii =
                        origin_offset_radii.unwrap_or_default();
                } else {
                    candidate_diagnostics.minimum_initial_origin_offset_radii =
                        candidate_diagnostics
                            .minimum_initial_origin_offset_radii
                            .min(origin_offset_radii.unwrap_or_default());
                }
                candidate_diagnostics.maximum_initial_origin_offset_radii = candidate_diagnostics
                    .maximum_initial_origin_offset_radii
                    .max(origin_offset_radii.unwrap_or_default());
                candidate_diagnostics.initial_origin_candidates += 1;
            } else {
                candidate_diagnostics.invalid_initial_origins += 1;
            }
            // With no conditioned pivot, spatial containment is the only
            // independent anatomical check on a new material identity.  Make
            // that check deliberately strict; a finite central pivot retains
            // the ordinary component-purity threshold above.
            if !component.origin_valid && purity < RELATION_IRIS_UNOBSERVABLE_ORIGIN_MIN_PURITY {
                candidate_diagnostics.rejected_untrusted_origin_seed += 1;
                continue;
            }
            // A finite pivot far outside the proposed eye contradicts the
            // anatomy and must not seed a new identity.  No finite pivot is a
            // different condition: near-translation motion cannot condition
            // the fixed point, but its exact material tracks may still serve
            // as provisional identity evidence.  Such an observation cannot
            // publish an origin because `component.origin_valid` remains false.
            if component.origin_valid && !current_origin_plausible {
                initial_origin_rejections += 1;
                candidate_diagnostics.rejected_initial_origin += 1;
                continue;
            }
        }
        let identity_score = iris_identity
            .filter(|identity| identity.active())
            .map_or(0.0, |_| {
                let centroid_agreement = (-continuity.centroid_step_radii / 0.32)
                    .exp()
                    .clamp(0.0, 1.0);
                let origin_agreement = continuity
                    .origin_step_radii
                    .map_or(0.5, |step| (-step / 0.28).exp().clamp(0.0, 1.0));
                1.10 * continuity.track_overlap
                    + 0.35 * centroid_agreement
                    + 0.25 * origin_agreement
            });
        let score = 2.20 * purity
            + 1.10 * interior
            + 0.85 * prior_pupil
            + 0.60 * two_dimensionality
            + 0.50 * component.coherence
            + 0.25 * (differential / 2.5).clamp(0.0, 1.0)
            + 0.15 * (selected.len() as f32 / 6.0).clamp(0.0, 1.0)
            + identity_score;
        candidate_diagnostics.maximum_score = candidate_diagnostics.maximum_score.max(score);
        candidate_diagnostics.ranked_components += 1;
        ranked.push((
            score,
            selected.len(),
            component_index,
            selected,
            continuity,
            identity_carried && !persistent_component,
            relation_iris_observation_evidence(differential),
            current_origin_plausible,
        ));
    }
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| right.1.cmp(&left.1))
    });
    relations.identity_switch_rejections = relations
        .identity_switch_rejections
        .saturating_add(identity_switch_rejections);
    relations.initial_origin_rejections = relations
        .initial_origin_rejections
        .saturating_add(initial_origin_rejections);
    relations
        .iris_candidate_diagnostics
        .accumulate(candidate_diagnostics);
    let Some((
        score,
        _,
        component_index,
        selected,
        continuity,
        identity_carried,
        observation_evidence,
        current_origin_plausible,
    )) = ranked.into_iter().next()
    else {
        return Vec::new();
    };
    if score < 2.35 {
        relations.iris_candidate_diagnostics.rejected_score += 1;
        return Vec::new();
    }
    relations.observed_iris_component = Some(component_index);
    relations.selected_identity_overlap = continuity.track_overlap;
    relations.selected_origin_consistent = continuity.origin_consistent;
    relations.observed_motion_evidence = observation_evidence;
    let identity_authorized = iris_identity.is_some_and(|identity| {
        let continuing =
            identity.active() && continuity.track_overlap >= RELATION_IRIS_IDENTITY_MIN_OVERLAP;
        continuing
            && identity.evidence + observation_evidence >= RELATION_IRIS_IDENTITY_MIN_EVIDENCE
            && identity.confirmations.saturating_add(1) >= RELATION_IRIS_IDENTITY_MIN_CONFIRMATIONS
    });
    let origin_authorized =
        iris_identity.is_some_and(|identity| identity.origin_valid) || current_origin_plausible;
    if identity_authorized && !origin_authorized {
        relations
            .iris_candidate_diagnostics
            .withheld_untrusted_origin += 1;
    }
    let authorized = identity_authorized && origin_authorized;
    if !authorized {
        relations
            .iris_candidate_diagnostics
            .provisional_observations += 1;
        return Vec::new();
    }
    relations.iris_candidate_diagnostics.authorized_observations += 1;
    relations.selected_iris_component = Some(component_index);
    relations.selected_by_identity_carry = identity_carried;
    selected
}

fn update_semantic_layer(
    object: usize,
    point_indices: &[usize],
    normal_indices: &[usize],
    radial_constraints: &[RadialLimbusFlow],
    matches: &[Match],
    tracks: &[FeatureTrack],
    motions: &mut [SimilarityMotion; OBJECTS],
    layers: &mut [MotionLayerStatus; OBJECTS],
    signatures: &mut [LayerMotionSignature; OBJECTS],
    previous_layers: &[MotionLayerStatus; OBJECTS],
    center: [f32; 2],
    global: SimilarityMotion,
) {
    let selected = point_indices
        .iter()
        .map(|index| &matches[*index])
        .collect::<Vec<_>>();
    motions[object] = fit_similarity_with_normal_constraints(
        point_indices,
        normal_indices,
        radial_constraints,
        matches,
        tracks,
        center,
    );
    let evidence_count = point_indices.len() + normal_indices.len() + radial_constraints.len();
    let inverse_evidence = 1.0 / evidence_count.max(1) as f32;
    layers[object].centroid = [
        (point_indices
            .iter()
            .chain(normal_indices)
            .map(|index| matches[*index].current[0])
            .sum::<f32>()
            + radial_constraints
                .iter()
                .map(|flow| flow.current[0])
                .sum::<f32>())
            * inverse_evidence,
        (point_indices
            .iter()
            .chain(normal_indices)
            .map(|index| matches[*index].current[1])
            .sum::<f32>()
            + radial_constraints
                .iter()
                .map(|flow| flow.current[1])
                .sum::<f32>())
            * inverse_evidence,
    ];
    let residuals = selected
        .iter()
        .map(|item| residual_motion(item, global, center))
        .collect::<Vec<_>>();
    let inverse = 1.0 / selected.len().max(1) as f32;
    let differential = if object == GENERAL_LAYER {
        [0.0; 2]
    } else {
        [
            residuals.iter().map(|value| value[0]).sum::<f32>() * inverse,
            residuals.iter().map(|value| value[1]).sum::<f32>() * inverse,
        ]
    };
    layers[object].differential =
        if signatures[object].age == 0 && !signatures[object].samples.is_empty() {
            [
                0.62 * previous_layers[object].differential[0] + 0.38 * differential[0],
                0.62 * previous_layers[object].differential[1] + 0.38 * differential[1],
            ]
        } else {
            differential
        };
    layers[object].trajectory_error = residuals
        .iter()
        .map(|value| (value[0] - differential[0]).hypot(value[1] - differential[1]))
        .sum::<f32>()
        * inverse;
    layers[object].persistent_tracks = evidence_count;
    signatures[object].samples.push_back(differential);
    while signatures[object].samples.len() > MOTION_SIGNATURE_LEN {
        signatures[object].samples.pop_front();
    }
    signatures[object].support = evidence_count;
    signatures[object].age = 0;
    layers[object].signature_samples = signatures[object].samples.len();
}

/// Split an eye view into fixed physical hypotheses before the anonymous
/// fallback clusterer runs:
///   L0 general/global affine motion,
///   L1 non-specular pupil/iris motion,
///   L2 compact non-Lambertian reflection motion.
///
/// Motion consensus still determines which tracks train each layer. RAW
/// photometry only prevents a compact bright glint from being treated as a
/// material point on the pupil when their instantaneous velocities coincide.
fn cluster_semantic_eye_layers(
    matches: &mut [Match],
    tracks: &[FeatureTrack],
    frame: &RawFrame,
    edges: &[EdgeEvidence],
    iris_seed: Option<IrisEllipseSeed>,
    motions: &mut [SimilarityMotion; OBJECTS],
    layers: &mut [MotionLayerStatus; OBJECTS],
    signatures: &mut [LayerMotionSignature; OBJECTS],
    parallax_axis: &mut [f32; 2],
    semantic_eye_center: &mut Option<[f32; 2]>,
    semantic_eye_region_state: &mut Option<EyeMotionRegion>,
    radial_limbus_flows: &mut [RadialLimbusFlow],
    radial_limbus_fused: &mut usize,
    center: [f32; 2],
    global: SimilarityMotion,
    relations: &mut MotionRelationFrame,
    iris_identity: Option<&PersistentRelationIrisIdentity>,
) -> bool {
    if matches.len() < MIN_LAYER_SUPPORT * 2 + MIN_REFLECTION_SUPPORT {
        return false;
    }
    let previous_center = *semantic_eye_center;
    let previous_region = *semantic_eye_region_state;
    let seeded_region = iris_seed.map(|seed| EyeMotionRegion::from_local_seed(seed, frame));
    let specular_search_region = seeded_region.or(previous_region);
    let specular_candidates = matches
        .iter()
        .enumerate()
        .filter_map(|(match_index, item)| {
            let track = &tracks[item.track_index];
            if track.matched_streak < 1 {
                return None;
            }
            let score = 0.58 * track.specularity + 0.42 * item.specularity;
            let held = previous_center.is_some_and(|prior| {
                (item.current[0] - prior[0]).hypot(item.current[1] - prior[1])
                    <= REFLECTION_SPATIAL_RADIUS
                    && score >= SPECULAR_HOLD_SCORE
            }) || (track.object == REFLECTION_LAYER && score >= SPECULAR_HOLD_SCORE);
            let inside_seed = specular_search_region
                .is_none_or(|region| region.contains_scaled(item.current, 1.15));
            (inside_seed && (score >= SPECULAR_HIGH_SCORE || held)).then_some((match_index, score))
        })
        .collect::<Vec<_>>();
    let Some((reflection_anchor, _)) = specular_candidates.iter().copied().max_by(
        |(left_index, left_score), (right_index, right_score)| {
            let rank = |index: usize, score: f32| {
                let point = matches[index].current;
                let neighbors = specular_candidates
                    .iter()
                    .filter(|(other, _)| {
                        (point[0] - matches[*other].current[0])
                            .hypot(point[1] - matches[*other].current[1])
                            <= REFLECTION_SPATIAL_RADIUS
                    })
                    .count();
                let prior_distance = previous_center.map_or(0.0, |prior| {
                    (point[0] - prior[0]).hypot(point[1] - prior[1])
                });
                (neighbors, score, -prior_distance)
            };
            let left = rank(*left_index, *left_score);
            let right = rank(*right_index, *right_score);
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.total_cmp(&right.1))
                .then_with(|| left.2.total_cmp(&right.2))
        },
    ) else {
        return false;
    };
    let reflection_spatial = specular_candidates
        .iter()
        .filter_map(|(index, _)| {
            ((matches[*index].current[0] - matches[reflection_anchor].current[0])
                .hypot(matches[*index].current[1] - matches[reflection_anchor].current[1])
                <= REFLECTION_SPATIAL_RADIUS)
                .then_some(*index)
        })
        .collect::<Vec<_>>();
    let reflection = semantic_motion_core(
        &reflection_spatial,
        matches,
        global,
        center,
        MIN_REFLECTION_SUPPORT,
    );
    if reflection.len() < MIN_REFLECTION_SUPPORT {
        return false;
    }
    let mut reflection_x = reflection
        .iter()
        .map(|index| matches[*index].current[0])
        .collect::<Vec<_>>();
    let mut reflection_y = reflection
        .iter()
        .map(|index| matches[*index].current[1])
        .collect::<Vec<_>>();
    let observed_center = [median(&mut reflection_x), median(&mut reflection_y)];
    let tracked_center = previous_center.map_or(observed_center, |prior| {
        if (observed_center[0] - prior[0]).hypot(observed_center[1] - prior[1]) <= 80.0 {
            [
                0.70 * prior[0] + 0.30 * observed_center[0],
                0.70 * prior[1] + 0.30 * observed_center[1],
            ]
        } else {
            observed_center
        }
    });
    *semantic_eye_center = Some(tracked_center);
    let eye_region = semantic_eye_region(frame, edges, iris_seed, tracked_center, previous_region);
    *semantic_eye_region_state = Some(eye_region);
    let reflection_mask = reflection
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut pupil_candidates = Vec::new();
    let mut general_candidates = Vec::new();
    let mut limbus_geometry = Vec::new();
    for (match_index, item) in matches.iter().enumerate() {
        if reflection_mask.contains(&match_index) {
            continue;
        }
        if tracks[item.track_index].matched_streak < 1 {
            continue;
        }
        let track = &tracks[item.track_index];
        let normalized_radius = eye_region.normalized_radius(item.current);
        let outward = eye_region.outward_normal(item.current);
        let edge_normal = normalized_vector(track.edge_normal);
        let radial_alignment = (outward[0] * edge_normal[0] + outward[1] * edge_normal[1]).abs();
        let possible_limbus = (LIMBUS_INNER_NORMALIZED_RADIUS..=LIMBUS_OUTER_NORMALIZED_RADIUS)
            .contains(&normalized_radius)
            && radial_alignment >= MIN_LIMBUS_RADIAL_ALIGNMENT;
        if possible_limbus {
            limbus_geometry.push(match_index);
        }
        if eye_region.contains_scaled(item.current, 1.10) || possible_limbus {
            pupil_candidates.push(match_index);
        } else {
            let residual = residual_motion(item, global, center);
            if residual[0].hypot(residual[1]) <= SEMANTIC_MOTION_CORE_RADIUS {
                general_candidates.push(match_index);
            }
        }
    }
    let relation_graph_persistent = relation_graph_has_persistent_component(relations)
        || iris_identity.is_some_and(PersistentRelationIrisIdentity::active);
    let relation_graph_informative = relation_graph_is_informative(relations);
    let relation_pupil = if relation_graph_persistent {
        relation_graph_iris_core(
            &pupil_candidates,
            matches,
            tracks,
            global,
            center,
            eye_region,
            relations,
            iris_identity,
            MIN_LAYER_SUPPORT,
        )
    } else {
        Vec::new()
    };
    let pupil = if relation_pupil.len() >= MIN_LAYER_SUPPORT {
        relation_pupil
    } else if relation_graph_informative {
        // Once the graph has enough independent relational evidence to expose
        // multiple material populations, never collapse them back into one
        // velocity-only cyan layer. Ambiguity is safer than false anatomy.
        return false;
    } else {
        semantic_motion_core(
            &pupil_candidates,
            matches,
            global,
            center,
            MIN_LAYER_SUPPORT,
        )
    };
    if pupil.len() < MIN_LAYER_SUPPORT || general_candidates.len() < MIN_LAYER_SUPPORT {
        return false;
    }

    // Grow the pupil layer across the visible limbus using only observable
    // edge-normal motion. Tangential drift on a smooth contour is the aperture
    // problem, not evidence for another physical layer.
    let pupil_points = pupil
        .iter()
        .map(|index| &matches[*index])
        .collect::<Vec<_>>();
    let pupil_prior = stable_similarity_prior(&pupil_points, center);
    let pupil_residuals = pupil
        .iter()
        .map(|index| residual_motion(&matches[*index], global, center))
        .collect::<Vec<_>>();
    let pupil_inverse = 1.0 / pupil_residuals.len().max(1) as f32;
    let current_pupil_differential = [
        pupil_residuals.iter().map(|value| value[0]).sum::<f32>() * pupil_inverse,
        pupil_residuals.iter().map(|value| value[1]).sum::<f32>() * pupil_inverse,
    ];
    let mut expected_pupil_signature = signatures[PUPIL_LAYER]
        .samples
        .iter()
        .copied()
        .collect::<Vec<_>>();
    expected_pupil_signature.push(current_pupil_differential);
    if expected_pupil_signature.len() > MOTION_SIGNATURE_LEN {
        expected_pupil_signature.remove(0);
    }
    let pupil_mask = pupil
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let pupil_normal_flow = limbus_geometry
        .iter()
        .copied()
        .filter(|index| !pupil_mask.contains(index))
        .filter(|index| {
            let item = &matches[*index];
            let track = &tracks[item.track_index];
            if track.matched_streak < MIN_MOTION_SIGNATURE as u8 - 1 {
                return false;
            }
            let normal = normalized_vector(track.edge_normal);
            let predicted = pupil_prior.predict(item.previous, center);
            let current_error = ((item.current[0] - predicted[0]) * normal[0]
                + (item.current[1] - predicted[1]) * normal[1])
                .abs();
            if current_error > MAX_LIMBUS_NORMAL_FLOW_ERROR {
                return false;
            }
            let residual = residual_motion(item, global, center);
            let signature = motion_signature(track, residual);
            signature.len() >= MIN_MOTION_SIGNATURE
                && normal_signature_distance(&signature, &expected_pupil_signature, normal)
                    <= MAX_LIMBUS_NORMAL_SIGNATURE_ERROR
        })
        .collect::<Vec<_>>();
    for flow in radial_limbus_flows.iter_mut() {
        flow.fused = false;
    }
    let mut radial_ranked = radial_limbus_flows
        .iter()
        .enumerate()
        .filter_map(|(index, flow)| {
            let predicted = pupil_prior.predict(flow.previous, center);
            let normal_error = ((flow.current[0] - predicted[0]) * flow.normal[0]
                + (flow.current[1] - predicted[1]) * flow.normal[1])
                .abs();
            (normal_error <= RADIAL_LIMBUS_MAX_PRIOR_NORMAL_ERROR_PX)
                .then_some((index, flow.confidence * (-normal_error / 3.5).exp()))
        })
        .collect::<Vec<_>>();
    radial_ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    radial_ranked.truncate(RADIAL_LIMBUS_MAX_FUSED);
    if radial_ranked.len() < RADIAL_LIMBUS_MIN_JOINT_SUPPORT {
        radial_ranked.clear();
    }
    for (index, _) in &radial_ranked {
        radial_limbus_flows[*index].fused = true;
    }
    let radial_normal_flow = radial_ranked
        .iter()
        .map(|(index, _)| radial_limbus_flows[*index])
        .collect::<Vec<_>>();
    *radial_limbus_fused = radial_normal_flow.len();

    for (match_index, item) in matches.iter_mut().enumerate() {
        item.layer_evidence = false;
        item.normal_flow_evidence = false;
        item.assignment_margin = 0.0;
        item.object = if reflection_spatial.contains(&match_index) {
            REFLECTION_LAYER
        } else if pupil_candidates.contains(&match_index) {
            PUPIL_LAYER
        } else {
            GENERAL_LAYER
        };
    }
    for (object, selected) in [
        (GENERAL_LAYER, general_candidates.as_slice()),
        (PUPIL_LAYER, pupil.as_slice()),
        (REFLECTION_LAYER, reflection.as_slice()),
    ] {
        for match_index in selected {
            matches[*match_index].object = object;
            matches[*match_index].layer_evidence = true;
            matches[*match_index].assignment_margin = 1.0;
        }
    }
    for match_index in &pupil_normal_flow {
        matches[*match_index].object = PUPIL_LAYER;
        matches[*match_index].layer_evidence = true;
        matches[*match_index].normal_flow_evidence = true;
        matches[*match_index].assignment_margin = 1.0;
    }

    let previous_layers = *layers;
    update_semantic_layer(
        GENERAL_LAYER,
        &general_candidates,
        &[],
        &[],
        matches,
        tracks,
        motions,
        layers,
        signatures,
        &previous_layers,
        center,
        global,
    );
    update_semantic_layer(
        PUPIL_LAYER,
        &pupil,
        &pupil_normal_flow,
        &radial_normal_flow,
        matches,
        tracks,
        motions,
        layers,
        signatures,
        &previous_layers,
        center,
        global,
    );
    update_semantic_layer(
        REFLECTION_LAYER,
        &reflection,
        &[],
        &[],
        matches,
        tracks,
        motions,
        layers,
        signatures,
        &previous_layers,
        center,
        global,
    );
    for object in [GENERAL_LAYER, PUPIL_LAYER, REFLECTION_LAYER] {
        let current_signature = signatures[object]
            .samples
            .iter()
            .copied()
            .collect::<Vec<_>>();
        layers[object].separation = [GENERAL_LAYER, PUPIL_LAYER, REFLECTION_LAYER]
            .into_iter()
            .filter(|other| *other != object && signatures[*other].support > 0)
            .map(|other| {
                let other_signature = signatures[other]
                    .samples
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                signature_distance(&current_signature, &other_signature)
            })
            .fold(f32::INFINITY, f32::min);
        if !layers[object].separation.is_finite() {
            layers[object].separation = 0.0;
        }
        let trajectory_coherence = (-layers[object].trajectory_error / 3.2).exp();
        let affine_coherence = (-motions[object].residual / 4.0).exp();
        let support_target = if object == REFLECTION_LAYER { 2.0 } else { 6.0 };
        let support_coherence =
            (layers[object].persistent_tracks as f32 / support_target).clamp(0.0, 1.0);
        let frame_coherence = trajectory_coherence * affine_coherence * support_coherence;
        layers[object].coherence = if previous_layers[object].stable_frames > 0 {
            0.66 * previous_layers[object].coherence + 0.34 * frame_coherence
        } else {
            frame_coherence
        };
        let minimum_support = if object == REFLECTION_LAYER {
            MIN_REFLECTION_SUPPORT
        } else {
            MIN_LAYER_SUPPORT
        };
        let coherent = layers[object].persistent_tracks >= minimum_support
            && layers[object].trajectory_error <= SEMANTIC_MOTION_CORE_RADIUS
            && motions[object].residual <= 4.0
            && layers[object].coherence >= 0.14;
        layers[object].stable_frames = if coherent {
            previous_layers[object].stable_frames.saturating_add(1)
        } else {
            previous_layers[object].stable_frames.saturating_sub(1)
        };
    }
    motions[RESIDUAL_LAYER].support = 0;
    layers[RESIDUAL_LAYER].persistent_tracks = 0;
    layers[RESIDUAL_LAYER].stable_frames = layers[RESIDUAL_LAYER].stable_frames.saturating_sub(1);
    layers[RESIDUAL_LAYER].coherence *= 0.8;
    signatures[RESIDUAL_LAYER].support = 0;
    signatures[RESIDUAL_LAYER].age = signatures[RESIDUAL_LAYER].age.saturating_add(1);

    let axis_candidate =
        if layers[PUPIL_LAYER].differential[0].hypot(layers[PUPIL_LAYER].differential[1]) > 0.05 {
            normalized_vector(layers[PUPIL_LAYER].differential)
        } else {
            normalized_vector([
                layers[REFLECTION_LAYER].differential[0] - layers[PUPIL_LAYER].differential[0],
                layers[REFLECTION_LAYER].differential[1] - layers[PUPIL_LAYER].differential[1],
            ])
        };
    if axis_candidate[0].hypot(axis_candidate[1]) > 0.5 {
        let mut candidate = axis_candidate;
        if parallax_axis[0] * candidate[0] + parallax_axis[1] * candidate[1] < 0.0 {
            candidate = [-candidate[0], -candidate[1]];
        }
        *parallax_axis = if parallax_axis[0].hypot(parallax_axis[1]) > 0.5 {
            normalized_vector([
                0.78 * parallax_axis[0] + 0.22 * candidate[0],
                0.78 * parallax_axis[1] + 0.22 * candidate[1],
            ])
        } else {
            candidate
        };
    }
    layers[GENERAL_LAYER].parallax = 0.0;
    for object in [PUPIL_LAYER, REFLECTION_LAYER] {
        layers[object].parallax = layers[object].differential[0] * parallax_axis[0]
            + layers[object].differential[1] * parallax_axis[1];
    }
    for item in matches {
        item.z = layers[item.object].parallax;
    }
    true
}

/// Use graph components to keep physically distinct material populations in
/// separate display/tracking layers even when the full semantic eye split is
/// unavailable (most commonly, no stable glint/reflection layer). This does
/// not authorize anatomy: the caller still passes `semantic_layers=false` to
/// coupled kinematics. It merely prevents iris and lash tracks from being
/// painted and predicted as one cyan object.
#[allow(clippy::too_many_arguments)]
fn cluster_relation_motion_layers(
    matches: &mut [Match],
    tracks: &[FeatureTrack],
    relations: &mut MotionRelationFrame,
    eye_region: Option<EyeMotionRegion>,
    motions: &mut [SimilarityMotion; OBJECTS],
    layers: &mut [MotionLayerStatus; OBJECTS],
    signatures: &mut [LayerMotionSignature; OBJECTS],
    parallax_axis: &mut [f32; 2],
    center: [f32; 2],
    global: SimilarityMotion,
    iris_identity: Option<&PersistentRelationIrisIdentity>,
) -> bool {
    let supported_components = relations
        .components
        .iter()
        .enumerate()
        .filter(|(_, component)| {
            component.members.len() >= MIN_LAYER_SUPPORT
                && component.coherence >= 0.32
                && relation_component_is_persistent(component)
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if supported_components.len() < 2 {
        return false;
    }

    if relations.selected_iris_component.is_none() {
        if let Some(region) = eye_region {
            let candidates = matches
                .iter()
                .enumerate()
                .filter_map(|(index, item)| {
                    region.contains_scaled(item.current, 1.10).then_some(index)
                })
                .collect::<Vec<_>>();
            let _ = relation_graph_iris_core(
                &candidates,
                matches,
                tracks,
                global,
                center,
                region,
                relations,
                iris_identity,
                MIN_LAYER_SUPPORT,
            );
        }
    }
    let Some(iris_component) = relations.selected_iris_component else {
        return false;
    };
    if !supported_components.contains(&iris_component) {
        return false;
    }

    let residual_energy = |component_index: usize| {
        let component = &relations.components[component_index];
        component
            .members
            .iter()
            .map(|index| {
                let residual = residual_motion(&matches[*index], global, center);
                residual[0] * residual[0] + residual[1] * residual[1]
            })
            .sum::<f32>()
            / component.members.len().max(1) as f32
    };
    let Some(general_component) = supported_components
        .iter()
        .copied()
        .filter(|index| *index != iris_component)
        .min_by(|left, right| residual_energy(*left).total_cmp(&residual_energy(*right)))
    else {
        return false;
    };

    let mut component_objects = vec![
        (general_component, GENERAL_LAYER),
        (iris_component, PUPIL_LAYER),
    ];
    let mut reflection_used = false;
    for component_index in supported_components
        .iter()
        .copied()
        .filter(|index| *index != general_component && *index != iris_component)
    {
        let component = &relations.components[component_index];
        let specularity = component
            .members
            .iter()
            .map(|index| {
                let item = &matches[*index];
                0.55 * tracks[item.track_index].specularity + 0.45 * item.specularity
            })
            .sum::<f32>()
            / component.members.len().max(1) as f32;
        let object = if !reflection_used && specularity >= SPECULAR_HOLD_SCORE {
            reflection_used = true;
            REFLECTION_LAYER
        } else {
            RESIDUAL_LAYER
        };
        if !component_objects.iter().any(|(_, used)| *used == object) {
            component_objects.push((component_index, object));
        }
    }

    for item in matches.iter_mut() {
        item.layer_evidence = false;
        item.normal_flow_evidence = false;
        item.assignment_margin = 0.0;
    }
    for (component_index, object) in &component_objects {
        let component = &relations.components[*component_index];
        for match_index in &component.members {
            let item = &mut matches[*match_index];
            item.object = *object;
            item.layer_evidence = true;
            item.assignment_margin = component.coherence;
        }
    }

    let previous_layers = *layers;
    let mut object_supported = [false; OBJECTS];
    for (component_index, object) in &component_objects {
        let component = &relations.components[*component_index];
        update_semantic_layer(
            *object,
            &component.members,
            &[],
            &[],
            matches,
            tracks,
            motions,
            layers,
            signatures,
            &previous_layers,
            center,
            global,
        );
        layers[*object].trajectory_error = motions[*object].residual;
        let frame_coherence = (component.coherence
            * (-motions[*object].residual / 2.8).exp()
            * (component.members.len() as f32 / 6.0).clamp(0.0, 1.0))
        .clamp(0.0, 1.0);
        layers[*object].coherence = if previous_layers[*object].stable_frames > 0 {
            0.66 * previous_layers[*object].coherence + 0.34 * frame_coherence
        } else {
            frame_coherence
        };
        object_supported[*object] = true;
    }
    for object in 0..OBJECTS {
        if object_supported[object] {
            let nearest = component_objects
                .iter()
                .filter(|(_, other)| *other != object)
                .map(|(_, other)| {
                    let translation = (motions[object].translation[0]
                        - motions[*other].translation[0])
                        .hypot(motions[object].translation[1] - motions[*other].translation[1]);
                    translation
                        + 36.0
                            * ((motions[object].rotation - motions[*other].rotation).abs()
                                + (motions[object].scale_delta - motions[*other].scale_delta).abs())
                })
                .fold(f32::INFINITY, f32::min);
            layers[object].separation = if nearest.is_finite() { nearest } else { 0.0 };
            let coherent = layers[object].persistent_tracks >= MIN_LAYER_SUPPORT
                && motions[object].residual <= MAX_LAYER_RESIDUAL
                && layers[object].separation >= MIN_LAYER_SEPARATION
                && layers[object].coherence >= 0.14;
            layers[object].stable_frames = if coherent {
                previous_layers[object].stable_frames.saturating_add(1)
            } else {
                previous_layers[object].stable_frames.saturating_sub(1)
            };
        } else {
            motions[object].support = 0;
            layers[object].persistent_tracks = 0;
            layers[object].coherence *= 0.8;
            layers[object].trajectory_error = 0.0;
            layers[object].signature_samples = 0;
            layers[object].stable_frames = layers[object].stable_frames.saturating_sub(1);
            layers[object].separation = 0.0;
            signatures[object].support = 0;
            signatures[object].age = signatures[object].age.saturating_add(1);
        }
    }

    let axis_candidate = normalized_vector(layers[PUPIL_LAYER].differential);
    if axis_candidate[0].hypot(axis_candidate[1]) > 0.5 {
        let mut candidate = axis_candidate;
        if parallax_axis[0] * candidate[0] + parallax_axis[1] * candidate[1] < 0.0 {
            candidate = [-candidate[0], -candidate[1]];
        }
        *parallax_axis = if parallax_axis[0].hypot(parallax_axis[1]) > 0.5 {
            normalized_vector([
                0.78 * parallax_axis[0] + 0.22 * candidate[0],
                0.78 * parallax_axis[1] + 0.22 * candidate[1],
            ])
        } else {
            candidate
        };
    }
    layers[GENERAL_LAYER].parallax = 0.0;
    for object in [PUPIL_LAYER, REFLECTION_LAYER, RESIDUAL_LAYER] {
        layers[object].parallax = layers[object].differential[0] * parallax_axis[0]
            + layers[object].differential[1] * parallax_axis[1];
    }
    for item in matches {
        item.z = layers[item.object].parallax;
    }
    true
}

fn cluster_motion_layers(
    matches: &mut [Match],
    tracks: &[FeatureTrack],
    motions: &mut [SimilarityMotion; OBJECTS],
    layers: &mut [MotionLayerStatus; OBJECTS],
    layer_signatures: &mut [LayerMotionSignature; OBJECTS],
    parallax_axis: &mut [f32; 2],
    center: [f32; 2],
    global: SimilarityMotion,
) {
    let previous_layers = *layers;
    if matches.is_empty() {
        for object in 0..OBJECTS {
            motions[object].support = 0;
            layers[object].persistent_tracks = 0;
            layers[object].coherence *= 0.8;
            layers[object].trajectory_error = 0.0;
            layers[object].signature_samples = 0;
            layers[object].stable_frames = layers[object].stable_frames.saturating_sub(1);
            layers[object].separation = 0.0;
            layer_signatures[object].support = 0;
            layer_signatures[object].age = layer_signatures[object].age.saturating_add(1);
        }
        return;
    }

    let candidates = matches
        .iter()
        .enumerate()
        .filter_map(|(match_index, item)| {
            let track = &tracks[item.track_index];
            let residual = residual_motion(item, global, center);
            let samples = motion_signature(track, residual);
            (samples.len() >= MIN_MOTION_SIGNATURE).then_some(SignatureCandidate {
                match_index,
                samples,
            })
        })
        .collect::<Vec<_>>();
    let clusters = cluster_signatures(&candidates);

    // Match anonymous current-frame clusters back to persistent layer slots by
    // comparing the overlapping (pre-current-frame) trajectory. This keeps a
    // brow/skin layer from changing names with the iris when their latest
    // velocities cross.
    let mut cluster_objects = vec![usize::MAX; clusters.len()];
    let mut object_used = [false; OBJECTS];
    let mut identity_pairs = Vec::new();
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let previous_current = &cluster.centroid[..cluster.centroid.len().saturating_sub(1)];
        for object in 0..OBJECTS {
            let prior = &layer_signatures[object];
            if prior.samples.len() < MIN_MOTION_SIGNATURE.saturating_sub(1) || prior.age > MAX_AGE {
                continue;
            }
            let prior_samples = prior.samples.iter().copied().collect::<Vec<_>>();
            identity_pairs.push((
                signature_distance(previous_current, &prior_samples),
                cluster_index,
                object,
            ));
        }
    }
    identity_pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    for (cost, cluster_index, object) in identity_pairs {
        if cost > 2.4 || cluster_objects[cluster_index] != usize::MAX || object_used[object] {
            continue;
        }
        cluster_objects[cluster_index] = object;
        object_used[object] = true;
    }
    for object in 0..OBJECTS {
        if object_used[object] {
            continue;
        }
        if let Some(cluster_index) = cluster_objects
            .iter()
            .position(|assigned| *assigned == usize::MAX)
        {
            cluster_objects[cluster_index] = object;
            object_used[object] = true;
        }
    }

    let mut mature_match = vec![false; matches.len()];
    for (cluster_index, cluster) in clusters.iter().enumerate() {
        let object = cluster_objects[cluster_index];
        for candidate_index in &cluster.members {
            let match_index = candidates[*candidate_index].match_index;
            matches[match_index].object = object;
            matches[match_index].layer_evidence = true;
            mature_match[match_index] = true;
            let nearest_other = clusters
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != cluster_index)
                .map(|(_, other)| {
                    signature_distance(&candidates[*candidate_index].samples, &other.centroid)
                })
                .fold(f32::INFINITY, f32::min);
            let own = signature_distance(&candidates[*candidate_index].samples, &cluster.centroid);
            matches[match_index].assignment_margin = if nearest_other.is_finite() {
                ((nearest_other - own) / (1.0 + nearest_other)).clamp(0.0, 1.0)
            } else {
                1.0
            };
        }
    }
    // Provisional tracks do not train a layer. They receive a display/search
    // identity from the nearest current residual and only become evidence once
    // a multi-frame signature has accumulated.
    for (match_index, item) in matches.iter_mut().enumerate() {
        if mature_match[match_index] || clusters.is_empty() {
            continue;
        }
        let residual = residual_motion(item, global, center);
        let previous_object = tracks[item.track_index].object;
        let mut ranked = clusters
            .iter()
            .enumerate()
            .map(|(cluster_index, cluster)| {
                let latest = cluster.centroid.last().copied().unwrap_or([0.0; 2]);
                let mut cost = (residual[0] - latest[0]).hypot(residual[1] - latest[1]);
                if cluster_objects[cluster_index] != previous_object {
                    cost += 0.25 * tracks[item.track_index].assignment_confidence;
                }
                (cost, cluster_index)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| left.0.total_cmp(&right.0));
        if let Some((best, cluster_index)) = ranked.first().copied() {
            item.object = cluster_objects[cluster_index];
            item.assignment_margin = ranked.get(1).map_or(1.0, |second| {
                ((second.0 - best) / (1.0 + second.0)).clamp(0.0, 1.0)
            });
        }
    }

    for object in 0..OBJECTS {
        let cluster_for_object = cluster_objects
            .iter()
            .position(|assigned| *assigned == object);
        let Some(cluster_index) = cluster_for_object else {
            motions[object].support = 0;
            layers[object].persistent_tracks = 0;
            layers[object].coherence *= 0.8;
            layers[object].trajectory_error = 0.0;
            layers[object].signature_samples = 0;
            layers[object].stable_frames = layers[object].stable_frames.saturating_sub(1);
            layers[object].separation = 0.0;
            layer_signatures[object].support = 0;
            layer_signatures[object].age = layer_signatures[object].age.saturating_add(1);
            continue;
        };
        let cluster = &clusters[cluster_index];
        let selected = cluster
            .members
            .iter()
            .map(|candidate_index| &matches[candidates[*candidate_index].match_index])
            .collect::<Vec<_>>();
        motions[object] = fit_similarity_selected(&selected, center);
        let inverse = 1.0 / selected.len().max(1) as f32;
        layers[object].centroid = [
            selected.iter().map(|item| item.current[0]).sum::<f32>() * inverse,
            selected.iter().map(|item| item.current[1]).sum::<f32>() * inverse,
        ];
        let current_differential = cluster.centroid.last().copied().unwrap_or([0.0; 2]);
        layers[object].differential =
            if layer_signatures[object].age == 0 && !layer_signatures[object].samples.is_empty() {
                [
                    0.62 * previous_layers[object].differential[0] + 0.38 * current_differential[0],
                    0.62 * previous_layers[object].differential[1] + 0.38 * current_differential[1],
                ]
            } else {
                current_differential
            };
        let separation = clusters
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != cluster_index)
            .map(|(_, other)| signature_distance(&cluster.centroid, &other.centroid))
            .fold(f32::INFINITY, f32::min);
        layers[object].separation = if separation.is_finite() {
            separation
        } else {
            // The robust whole-frame affine is an implicit background layer.
            // A single surviving cluster can therefore still be physically
            // separated from the background; compare its residual trajectory
            // against the all-zero global-motion signature.
            signature_distance(&cluster.centroid, &vec![[0.0; 2]; cluster.centroid.len()])
        };
        layers[object].persistent_tracks = selected.len();
        layers[object].trajectory_error = cluster.within_error;
        layers[object].signature_samples = cluster.centroid.len();
        // Motion signatures are expressed at native/full resolution. Real LK
        // replay establishes that coherent non-background layers commonly
        // span 2-5 full pixels RMS over eight frames, so the old 1.35-pixel
        // falloff mislabeled genuine parallax as incoherent.
        let trajectory_coherence = (-cluster.within_error / 4.5).exp();
        let affine_coherence = (-motions[object].residual / 4.0).exp();
        let support_coherence = (selected.len() as f32 / 8.0).clamp(0.0, 1.0);
        let frame_coherence = trajectory_coherence * affine_coherence * support_coherence;
        layers[object].coherence = if previous_layers[object].stable_frames > 0 {
            0.66 * previous_layers[object].coherence + 0.34 * frame_coherence
        } else {
            frame_coherence
        };
        let coherent = selected.len() >= MIN_LAYER_SUPPORT
            && layers[object].persistent_tracks >= MIN_LAYER_PERSISTENT_TRACKS
            && motions[object].residual <= MAX_LAYER_RESIDUAL
            && layers[object].separation >= MIN_LAYER_SEPARATION
            && layers[object].coherence >= 0.16;
        layers[object].stable_frames = if coherent {
            previous_layers[object].stable_frames.saturating_add(1)
        } else {
            previous_layers[object].stable_frames.saturating_sub(1)
        };
        layer_signatures[object].samples = cluster.centroid.iter().copied().collect();
        layer_signatures[object].support = selected.len();
        layer_signatures[object].age = 0;
    }

    let axis_candidate = layers
        .iter()
        .enumerate()
        .filter(|(object, layer)| {
            layer_signatures[*object].support >= MIN_LAYER_SUPPORT
                && layer.separation >= MIN_LAYER_SEPARATION
        })
        .max_by(|left, right| {
            let magnitude =
                |layer: &MotionLayerStatus| layer.differential[0].hypot(layer.differential[1]);
            magnitude(left.1).total_cmp(&magnitude(right.1))
        })
        .map(|(_, layer)| normalized_vector(layer.differential));
    if let Some(mut candidate) = axis_candidate {
        if parallax_axis[0] * candidate[0] + parallax_axis[1] * candidate[1] < 0.0 {
            candidate = [-candidate[0], -candidate[1]];
        }
        *parallax_axis = if parallax_axis[0].hypot(parallax_axis[1]) > 0.5 {
            normalized_vector([
                0.78 * parallax_axis[0] + 0.22 * candidate[0],
                0.78 * parallax_axis[1] + 0.22 * candidate[1],
            ])
        } else {
            candidate
        };
    }
    for layer in layers.iter_mut() {
        layer.parallax =
            layer.differential[0] * parallax_axis[0] + layer.differential[1] * parallax_axis[1];
    }
    for item in matches {
        item.z = layers[item.object].parallax;
    }
}

fn rebuild_nodes(points: &[(usize, [f32; 3])], output: &mut Vec<OverlayNode>) {
    for object in 0..OBJECTS {
        let object_points = points
            .iter()
            .filter(|item| item.0 == object)
            .map(|item| item.1)
            .collect::<Vec<_>>();
        if object_points.is_empty() {
            continue;
        }
        let mut bounds = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for point in &object_points {
            bounds[0] = bounds[0].min(point[0]);
            bounds[1] = bounds[1].max(point[0]);
            bounds[2] = bounds[2].min(point[1]);
            bounds[3] = bounds[3].max(point[1]);
            bounds[4] = bounds[4].min(point[2]);
            bounds[5] = bounds[5].max(point[2]);
        }
        append_node(object, &object_points, bounds, 0, output);
    }
}

fn append_node(
    object: usize,
    points: &[[f32; 3]],
    bounds: [f32; 6],
    depth: u8,
    output: &mut Vec<OverlayNode>,
) {
    output.push(OverlayNode {
        object,
        bounds,
        depth,
        count: points.len(),
    });
    if depth >= 4 || points.len() < 7 {
        return;
    }
    let midpoint = [
        (bounds[0] + bounds[1]) * 0.5,
        (bounds[2] + bounds[3]) * 0.5,
        (bounds[4] + bounds[5]) * 0.5,
    ];
    for child in 0..8 {
        let subset = points
            .iter()
            .copied()
            .filter(|point| {
                ((point[0] >= midpoint[0]) as usize
                    | (((point[1] >= midpoint[1]) as usize) << 1)
                    | (((point[2] >= midpoint[2]) as usize) << 2))
                    == child
            })
            .collect::<Vec<_>>();
        if subset.is_empty() {
            continue;
        }
        let child_bounds = [
            if child & 1 == 0 {
                bounds[0]
            } else {
                midpoint[0]
            },
            if child & 1 == 0 {
                midpoint[0]
            } else {
                bounds[1]
            },
            if child & 2 == 0 {
                bounds[2]
            } else {
                midpoint[1]
            },
            if child & 2 == 0 {
                midpoint[1]
            } else {
                bounds[3]
            },
            if child & 4 == 0 {
                bounds[4]
            } else {
                midpoint[2]
            },
            if child & 4 == 0 {
                midpoint[2]
            } else {
                bounds[5]
            },
        ];
        append_node(object, &subset, child_bounds, depth + 1, output);
    }
}

impl FourMotionOctrees {
    pub fn set_exhaustive_search_for_replay(&mut self, exhaustive: bool) {
        self.exhaustive_search_for_replay = exhaustive;
    }

    #[cfg(test)]
    fn set_nautilus_disabled_for_replay(&mut self, disabled: bool) {
        self.disable_nautilus_for_replay = disabled;
    }

    /// Drop all mode-specific temporal state when the UI leaves Clusters.
    /// This prevents the global patch matcher and its full RAW backing frame
    /// from remaining resident while another segmentation submode is active.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn observe(
        &mut self,
        pixels: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        focus_probe: Option<FocusDepthProbe>,
        use_canny_features: bool,
    ) -> MotionOctreeOverlay {
        self.observe_with_iris_seed(
            pixels,
            width,
            height,
            sensor_x,
            sensor_y,
            focus_probe,
            use_canny_features,
            None,
        )
    }

    pub fn observe_with_iris_seed(
        &mut self,
        pixels: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        focus_probe: Option<FocusDepthProbe>,
        use_canny_features: bool,
        iris_seed: Option<IrisEllipseSeed>,
    ) -> MotionOctreeOverlay {
        self.observe_with_iris_seed_timestamp(
            pixels,
            width,
            height,
            sensor_x,
            sensor_y,
            focus_probe,
            use_canny_features,
            LearningCannyProfile::default(),
            iris_seed,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn observe_with_iris_seed_at(
        &mut self,
        pixels: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        timestamp_ns: u64,
        focus_probe: Option<FocusDepthProbe>,
        use_canny_features: bool,
        iris_seed: Option<IrisEllipseSeed>,
    ) -> MotionOctreeOverlay {
        self.observe_with_iris_seed_timestamp(
            pixels,
            width,
            height,
            sensor_x,
            sensor_y,
            focus_probe,
            use_canny_features,
            LearningCannyProfile::default(),
            iris_seed,
            Some(timestamp_ns),
        )
    }

    /// Observe one native RAW frame with an explicitly selected calculation
    /// profile for the temporal Canny learning layers.  The ordinary entry
    /// points above retain the historical balanced profile for callers which
    /// are not controlled by the live K selector.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_with_iris_seed_at_with_canny_profile(
        &mut self,
        pixels: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        timestamp_ns: u64,
        focus_probe: Option<FocusDepthProbe>,
        profile: LearningCannyProfile,
        iris_seed: Option<IrisEllipseSeed>,
    ) -> MotionOctreeOverlay {
        self.observe_with_iris_seed_timestamp(
            pixels,
            width,
            height,
            sensor_x,
            sensor_y,
            focus_probe,
            true,
            profile,
            iris_seed,
            Some(timestamp_ns),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn observe_with_iris_seed_timestamp(
        &mut self,
        pixels: &[u16],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        focus_probe: Option<FocusDepthProbe>,
        use_canny_features: bool,
        learning_canny_profile: LearningCannyProfile,
        iris_seed: Option<IrisEllipseSeed>,
        timestamp_ns: Option<u64>,
    ) -> MotionOctreeOverlay {
        let timestamp_ns = timestamp_ns.unwrap_or_else(|| {
            self.fallback_timestamp_ns = self
                .fallback_timestamp_ns
                .saturating_add(25_000_000)
                .max(25_000_000);
            self.fallback_timestamp_ns
        });
        const MAX_CAUSAL_PREDICTION_GAP_NS: u64 = 250_000_000;
        let mut prediction_cadence_contiguous =
            self.previous_timestamp_ns.is_some_and(|previous| {
                timestamp_ns > previous && timestamp_ns - previous <= MAX_CAUSAL_PREDICTION_GAP_NS
            });
        self.previous_timestamp_ns = Some(timestamp_ns);
        let preprocess_started = Instant::now();
        let canny_definition_changed =
            use_canny_features && self.learning_canny_profile != learning_canny_profile;
        if self.previous.is_some()
            && (self.canny_features != use_canny_features || canny_definition_changed)
        {
            prediction_cadence_contiguous = false;
            self.previous = None;
            self.tracks.clear();
            self.nautilus_banks.clear();
            self.horizontal_light_field = HorizontalLightFieldStatus::default();
            self.radial_limbus_region = None;
            self.radial_limbus_flows.clear();
            self.motions = [SimilarityMotion::default(); OBJECTS];
            self.layers = [MotionLayerStatus::default(); OBJECTS];
            self.layer_signatures = Default::default();
            self.motion_relations.clear();
            self.relation_iris_identity = Default::default();
            self.parallax_axis = [0.0; 2];
            self.semantic_eye_center = None;
            self.semantic_eye_region = None;
            self.focus_sfm = Default::default();
            self.focus_sweep_seen = false;
            self.last_stable_focus = None;
            self.coupled_kinematics.clear();
        }
        self.canny_features = use_canny_features;
        self.learning_canny_profile = learning_canny_profile;
        let neutral_started = Instant::now();
        let current_pixels = if use_canny_features {
            cfa_neutral_raw(pixels, width, height)
        } else {
            pixels.to_vec()
        };
        let neutral_micros = neutral_started.elapsed().as_micros() as u64;
        let current = RawFrame {
            sensor_x,
            sensor_y,
            width,
            height,
            pixels: current_pixels,
        };
        let canny_started = Instant::now();
        let mut canny =
            use_canny_features.then(|| canny_field_with_profile(&current, learning_canny_profile));
        let canny_micros = canny_started.elapsed().as_micros() as u64;
        let (
            canny_primary_blur_micros,
            canny_gradient_micros,
            canny_hysteresis_micros,
            canny_nms_micros,
            canny_quantile_micros,
            canny_flood_micros,
            canny_broad_blur_micros,
        ) = canny.as_ref().map_or((0, 0, 0, 0, 0, 0, 0), |field| {
            (
                field.primary_blur_micros,
                field.gradient_micros,
                field.hysteresis_micros,
                field.nms_micros,
                field.quantile_micros,
                field.flood_micros,
                field.broad_blur_micros,
            )
        });
        let edge_started = Instant::now();
        let edge_output = canny
            .as_mut()
            .map(|field| edge_evidence(field, width, height))
            .unwrap_or_default();
        let learning_canny_mask = canny
            .as_ref()
            .map(|field| {
                field
                    .accepted
                    .iter()
                    .map(|accepted| u8::from(*accepted))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mut edges = edge_output.edges;
        let edge_micros = edge_started.elapsed().as_micros() as u64;
        let canny_attribute_micros = edge_output.attribute_micros;
        let canny_attribute_candidates = edge_output.attribute_candidates;
        let canny_attribute_evaluated = edge_output.attribute_evaluated;
        let canny_texture_evaluated = edge_output.texture_evaluated;
        let canny_texture_micros = edge_output.texture_micros;
        let canny_texture_simd_evaluated = edge_output.texture_simd_evaluated;
        let edge_high_threshold = canny.as_ref().map_or(0.0, |field| field.high_threshold);
        let preprocess_micros = preprocess_started.elapsed().as_micros() as u64;
        let Some(previous) = self.previous.as_ref() else {
            self.previous = Some(current.clone());
            self.horizontal_light_field = HorizontalLightFieldStatus::default();
            self.radial_limbus_region =
                iris_seed.map(|seed| EyeMotionRegion::from_local_seed(seed, &current));
            self.radial_limbus_flows.clear();
            let feature_iris_seed = iris_seed.or_else(|| {
                self.semantic_eye_region
                    .map(|region| region.local_seed(&current))
            });
            let seeds = seed_points(
                &current,
                canny.as_ref(),
                &edges,
                feature_iris_seed,
                &[],
                MAX_FEATURES,
            );
            for (point, score) in seeds {
                let edge_normal = canny.as_ref().map_or([0.0; 2], |field| {
                    local_canny_normal(
                        field,
                        width,
                        point[0].round() as usize,
                        point[1].round() as usize,
                    )
                });
                let mut track = FeatureTrack {
                    id: self.next_id,
                    points: VecDeque::from([[
                        point[0] + sensor_x as f32,
                        point[1] + sensor_y as f32,
                        0.0,
                    ]]),
                    object: 0,
                    age: 0,
                    last_seen_timestamp_ns: timestamp_ns,
                    score,
                    motion_ema: [0.0; 2],
                    motion_variance: 0.0,
                    matched_streak: 0,
                    layer_evidence: false,
                    normal_flow_evidence: false,
                    specularity: feature_specularity(&current, point),
                    assignment_confidence: 0.0,
                    edge_normal,
                    residual_history: VecDeque::new(),
                    focus_bins: Vec::new(),
                    focus_peak: None,
                };
                if let Some(probe) = focus_probe {
                    add_focus_sample(
                        &mut track,
                        probe.position,
                        feature_sharpness(&current, point),
                    );
                    if !probe.sweeping {
                        self.last_stable_focus = Some(probe.position);
                    }
                }
                if let Some(fingerprint) = nautilus_fingerprint(&current, point, edge_normal) {
                    self.nautilus_banks
                        .entry(track.id)
                        .or_default()
                        .observe(fingerprint);
                }
                self.tracks.push(track);
                self.next_id += 1;
            }
            self.match_diagnostics = MatchDiagnostics {
                neutral_micros,
                canny_micros,
                edge_micros,
                canny_primary_blur_micros,
                canny_gradient_micros,
                canny_hysteresis_micros,
                canny_nms_micros,
                canny_quantile_micros,
                canny_flood_micros,
                canny_broad_blur_micros,
                canny_attribute_micros,
                canny_attribute_candidates,
                canny_attribute_evaluated,
                canny_texture_evaluated,
                canny_texture_micros,
                canny_texture_simd_evaluated,
                preprocess_micros,
                ..MatchDiagnostics::default()
            };
            self.coupled_kinematics.observe(
                timestamp_ns,
                [
                    sensor_x as f32 + width as f32 * 0.5,
                    sensor_y as f32 + height as f32 * 0.5,
                ],
                width.max(height) as f64,
                self.motions,
                self.layers,
                false,
                None,
            );
            return self.overlay(
                sensor_x,
                sensor_y,
                edges,
                edge_high_threshold,
                0,
                learning_canny_mask,
            );
        };
        if previous.width != width || previous.height != height {
            self.tracks.clear();
            self.nautilus_banks.clear();
            self.horizontal_light_field = HorizontalLightFieldStatus::default();
            self.radial_limbus_region =
                iris_seed.map(|seed| EyeMotionRegion::from_local_seed(seed, &current));
            self.radial_limbus_flows.clear();
            self.motions = [SimilarityMotion::default(); OBJECTS];
            self.layers = [MotionLayerStatus::default(); OBJECTS];
            self.layer_signatures = Default::default();
            self.motion_relations.clear();
            self.parallax_axis = [0.0; 2];
            self.semantic_eye_center = None;
            self.semantic_eye_region = None;
            self.coupled_kinematics.clear();
            self.previous = Some(current);
            self.match_diagnostics = MatchDiagnostics {
                neutral_micros,
                canny_micros,
                edge_micros,
                canny_primary_blur_micros,
                canny_gradient_micros,
                canny_hysteresis_micros,
                canny_nms_micros,
                canny_quantile_micros,
                canny_flood_micros,
                canny_broad_blur_micros,
                canny_attribute_micros,
                canny_attribute_candidates,
                canny_attribute_evaluated,
                canny_texture_evaluated,
                canny_texture_micros,
                canny_texture_simd_evaluated,
                preprocess_micros,
                ..MatchDiagnostics::default()
            };
            self.coupled_kinematics.observe(
                timestamp_ns,
                [
                    sensor_x as f32 + width as f32 * 0.5,
                    sensor_y as f32 + height as f32 * 0.5,
                ],
                width.max(height) as f64,
                self.motions,
                self.layers,
                false,
                None,
            );
            return self.overlay(
                sensor_x,
                sensor_y,
                edges,
                edge_high_threshold,
                0,
                learning_canny_mask,
            );
        }
        let center = [
            sensor_x as f32 + width as f32 * 0.5,
            sensor_y as f32 + height as f32 * 0.5,
        ];
        let horizontal_light_field = if use_canny_features && prediction_cadence_contiguous {
            horizontal_light_field_tree(previous, &current, self.motions[GENERAL_LAYER])
        } else {
            HorizontalLightFieldStatus::default()
        };
        let previous_radial_region = self.radial_limbus_region.or(self.semantic_eye_region);
        let current_radial_region = iris_seed
            .map(|seed| EyeMotionRegion::from_local_seed(seed, &current))
            .or_else(|| {
                previous_radial_region.map(|region| {
                    let prior = if self.motions[PUPIL_LAYER].support >= 3
                        && self.motions[PUPIL_LAYER].residual <= 3.0
                    {
                        self.motions[PUPIL_LAYER]
                    } else if self.motions[GENERAL_LAYER].support >= 3
                        && self.motions[GENERAL_LAYER].residual <= 3.0
                    {
                        self.motions[GENERAL_LAYER]
                    } else if horizontal_light_field.reliable {
                        SimilarityMotion {
                            translation: [horizontal_light_field.horizontal_translation_px, 0.0],
                            scale_delta: horizontal_light_field.horizontal_scale_delta,
                            support: horizontal_light_field.leaf_nodes,
                            residual: horizontal_light_field.residual_px,
                            ..SimilarityMotion::default()
                        }
                    } else {
                        SimilarityMotion::default()
                    };
                    let scale = (1.0 + prior.scale_delta).clamp(0.88, 1.12);
                    EyeMotionRegion {
                        center: prior.predict(region.center, center),
                        major: region.major * scale,
                        minor: region.minor * scale,
                        angle: (region.angle + prior.rotation).rem_euclid(std::f32::consts::PI),
                    }
                })
            });
        let (mut radial_limbus_flows, radial_limbus_evaluations, radial_limbus_micros) =
            if use_canny_features && prediction_cadence_contiguous {
                previous_radial_region.zip(current_radial_region).map_or(
                    (Vec::new(), 0, 0),
                    |(previous_region, current_region)| {
                        radial_limbus_flows(previous, &current, previous_region, current_region)
                    },
                )
            } else {
                (Vec::new(), 0, 0)
            };
        if let Some(probe) = focus_probe {
            if probe.sweeping && !self.focus_sweep_seen {
                self.focus_sfm.begin();
                for track in &mut self.tracks {
                    track.focus_peak = None;
                    track.focus_bins.retain(|bin| {
                        self.last_stable_focus
                            .is_some_and(|position| bin.position == position)
                    });
                }
                self.focus_sweep_seen = true;
            } else if !probe.sweeping && self.focus_sweep_seen {
                let mut calibrated = 0usize;
                for track in &mut self.tracks {
                    track.focus_peak = estimate_focus_peak(&track.focus_bins);
                    calibrated += usize::from(track.focus_peak.is_some());
                }
                self.focus_sfm.finish_collection(calibrated);
                self.focus_sweep_seen = false;
            }
            if !probe.sweeping {
                self.last_stable_focus = Some(probe.position);
            }
        }
        // The half-resolution level suppresses residual CFA/noise and gives
        // each match a 38x38-pixel full-resolution context, comparable to a
        // pyramidal LK window. Full-resolution refinement keeps native sensor
        // coordinates and the signed Canny association exact.
        let pyramid_started = Instant::now();
        let previous_pyramid = downsample_two(previous);
        let current_pyramid = downsample_two(&current);
        let previous_moments = IntegralPatchMoments::new(previous);
        let current_moments = IntegralPatchMoments::new(&current);
        let previous_pyramid_moments = IntegralPatchMoments::new(&previous_pyramid);
        let current_pyramid_moments = IntegralPatchMoments::new(&current_pyramid);
        let pyramid_micros = pyramid_started.elapsed().as_micros() as u64;
        let matching_started = Instant::now();
        let mut matches = Vec::new();
        let mut match_diagnostics = MatchDiagnostics {
            neutral_micros,
            canny_micros,
            edge_micros,
            canny_primary_blur_micros,
            canny_gradient_micros,
            canny_hysteresis_micros,
            canny_nms_micros,
            canny_quantile_micros,
            canny_flood_micros,
            canny_broad_blur_micros,
            canny_attribute_micros,
            canny_attribute_candidates,
            canny_attribute_evaluated,
            canny_texture_evaluated,
            canny_texture_micros,
            canny_texture_simd_evaluated,
            preprocess_micros,
            pyramid_micros,
            light_field_micros: horizontal_light_field.elapsed_micros,
            light_field_nodes: horizontal_light_field.nodes.len(),
            light_field_leaves: horizontal_light_field.leaf_nodes,
            light_field_evaluations: horizontal_light_field.evaluations,
            light_field_reliable: horizontal_light_field.reliable,
            radial_limbus_evaluations,
            radial_limbus_accepted: radial_limbus_flows.len(),
            radial_limbus_micros,
            ..MatchDiagnostics::default()
        };
        for (track_index, track) in self.tracks.iter().enumerate() {
            match_diagnostics.considered += 1;
            // An aged track's last observation predates `previous`; comparing
            // that point against the adjacent-frame buffer would manufacture
            // a false source patch. Only its historical nautilus bank may
            // reconnect the ID across that gap.
            if track.age > 0 {
                continue;
            }
            let Some(last) = track.points.back() else {
                continue;
            };
            let previous_local = [
                last[0] - previous.sensor_x as f32,
                last[1] - previous.sensor_y as f32,
            ];
            let model = self.motions[track.object];
            // An under-supported or high-residual model must never move the
            // next search window away from the sensor-registered feature.
            // This is especially important immediately after an ROI move.
            let model_usable = prediction_cadence_contiguous
                && model.support >= 3
                && model.residual <= 3.0
                && model.translation[0].hypot(model.translation[1]) <= SEARCH_RADIUS as f32 * 1.5;
            let track_prediction_usable = prediction_cadence_contiguous
                && track.matched_streak >= 1
                && track.motion_ema[0].hypot(track.motion_ema[1]) <= SEARCH_RADIUS as f32 * 1.5;
            let layer = self.layers[track.object];
            let layer_prediction_usable = model_usable
                && layer.stable_frames >= MIN_LAYER_STABLE_FRAMES
                && layer.coherence >= 0.20
                && track.assignment_confidence >= 0.12;
            let established_prediction =
                layer_prediction_usable || track_prediction_usable || model_usable;
            let mut predicted_sensor = if layer_prediction_usable && track_prediction_usable {
                // Cross-validated on both lossless eye streams: the robust
                // layer similarity has the best tail error, while a damped
                // same-ID feature EMA has the best median during snappy local
                // motion.  Their equal blend beats either source alone for
                // general material and remains between the left/right iris
                // optima.  Both predictions are causal and sensor-absolute.
                let layer_prediction = model.predict([last[0], last[1]], center);
                let track_prediction = [
                    last[0] + 0.65 * track.motion_ema[0],
                    last[1] + 0.65 * track.motion_ema[1],
                ];
                [
                    0.5 * (layer_prediction[0] + track_prediction[0]),
                    0.5 * (layer_prediction[1] + track_prediction[1]),
                ]
            } else if layer_prediction_usable {
                model.predict([last[0], last[1]], center)
            } else if track_prediction_usable {
                [
                    last[0] + 0.65 * track.motion_ema[0],
                    last[1] + 0.65 * track.motion_ema[1],
                ]
            } else if model_usable {
                model.predict([last[0], last[1]], center)
            } else {
                [last[0], last[1]]
            };
            // The walk field has already compared this exact adjacent RAW
            // pair, so it may center the gross horizontal search even before
            // a point has been assigned an identity in `current`.  When an
            // point/model prediction exists, keep its identity-bearing
            // estimate. The broad walk is deliberately a gross fallback for
            // an otherwise unpredicted feature, not a force that can pull a
            // mature track away from its patch basin.
            let field_sensor_x = horizontal_light_field_sensor_prediction(
                &horizontal_light_field,
                previous,
                &current,
                [last[0], last[1]],
            );
            if let Some(field_x) = field_sensor_x {
                if !established_prediction {
                    predicted_sensor[0] = field_x;
                }
            }
            let predicted = [
                predicted_sensor[0] - sensor_x as f32,
                predicted_sensor[1] - sensor_y as f32,
            ];
            let candidate_is_valid = |candidate: [f32; 2]| {
                let candidate_x = candidate[0].round() as i32;
                let candidate_y = candidate[1].round() as i32;
                if candidate_x < PATCH_RADIUS
                    || candidate_y < PATCH_RADIUS
                    || candidate_x + PATCH_RADIUS >= width as i32
                    || candidate_y + PATCH_RADIUS >= height as i32
                {
                    return false;
                }
                if let Some(field) = canny.as_ref() {
                    let candidate_x = candidate_x as usize;
                    let candidate_y = candidate_y as usize;
                    // Canny is the detector, not a demand to re-detect the
                    // same physical ridge at full hysteresis strength every
                    // frame. Iris striations cross the adaptive threshold as
                    // illumination changes, so established RAW tracks use a
                    // lower continuation threshold.
                    let minimum_support = if track.matched_streak > 0 {
                        MIN_PERSISTENT_CANNY_SUPPORT
                    } else {
                        MIN_NEW_TRACK_CANNY_SUPPORT
                    };
                    let (candidate_support, candidate_normal) =
                        local_canny_peak(field, width, candidate_x, candidate_y);
                    if candidate_support < minimum_support {
                        return false;
                    }
                    let prior_normal = track.edge_normal;
                    // Patch identity needs edge orientation, not photometric
                    // polarity. Signed polarity is still enforced later when
                    // a current edge votes for the limbus ellipse.
                    if prior_normal[0].hypot(prior_normal[1]) > 0.5
                        && (candidate_normal[0] * prior_normal[0]
                            + candidate_normal[1] * prior_normal[1])
                            .abs()
                            < 0.20
                    {
                        return false;
                    }
                }
                true
            };
            let previous_half = [previous_local[0] * 0.5, previous_local[1] * 0.5];
            let predicted_half = [predicted[0] * 0.5, predicted[1] * 0.5];
            // Once a zero-mean-normalized patch track or its learned temporal
            // layer predicts the next location, do not continue paying for
            // the cold-start 17x17 search. Bound uncertainty by the previous
            // affine residual, per-track velocity variance, and model/track
            // disagreement. Unproven tracks retain the complete corridor.
            // A track does not need a semantic layer name before its own
            // z-normalized patch history can predict the next exposure. This
            // lets provisional texture stay cheap while the longer layered
            // motion signature is still forming. If prediction fails, normal
            // aging/reseeding gives that location a fresh exhaustive search.
            let bounded_prediction_usable = !self.exhaustive_search_for_replay
                && track_prediction_usable
                && track.matched_streak >= 2
                && track.motion_variance.is_finite()
                && track.motion_variance <= 64.0;
            let track_predicted_sensor =
                [last[0] + track.motion_ema[0], last[1] + track.motion_ema[1]];
            let prediction_disagreement = if layer_prediction_usable && track_prediction_usable {
                let model_prediction = model.predict([last[0], last[1]], center);
                (model_prediction[0] - track_predicted_sensor[0])
                    .hypot(model_prediction[1] - track_predicted_sensor[1])
            } else {
                0.0
            };
            let pyramid_search = if bounded_prediction_usable {
                let prediction_floor = if layer_prediction_usable { 2.5 } else { 4.0 };
                let full_resolution_uncertainty = prediction_floor
                    + if layer_prediction_usable {
                        model.residual.max(0.0)
                    } else {
                        0.0
                    }
                    + track.motion_variance.max(0.0).sqrt()
                    + prediction_disagreement;
                ((0.5 * full_resolution_uncertainty).ceil() as i32)
                    .clamp(3, (SEARCH_RADIUS + 1) / 2)
            } else {
                (SEARCH_RADIUS + 1) / 2
            };
            let maximum_basins = if bounded_prediction_usable { 2 } else { 4 };
            let refinement_radius = if bounded_prediction_usable { 2 } else { 3 };
            let backward_radius = if bounded_prediction_usable { 1 } else { 2 };
            let native_patch_radius = if bounded_prediction_usable {
                PREDICTED_PATCH_RADIUS
            } else {
                PATCH_RADIUS
            };
            let pyramid_patch_radius = if bounded_prediction_usable {
                PREDICTED_PYRAMID_PATCH_RADIUS
            } else {
                PYRAMID_PATCH_RADIUS
            };
            let search_diameter = (2 * pyramid_search + 1) as usize;
            let mut coarse_searches = Vec::<([f32; 2], i32, bool)>::with_capacity(2);
            coarse_searches.push((predicted_half, pyramid_search, false));
            if established_prediction
                && horizontal_light_field.reliable
                && horizontal_light_field.confidence >= 0.32
                && horizontal_light_field.residual_px <= 2.5
                && horizontal_light_field.horizontal_coverage >= 0.42
                && horizontal_light_field.leaf_nodes >= 4
            {
                if let Some(field_x) = field_sensor_x {
                    let field_local_x = field_x - sensor_x as f32;
                    let primary_reach_px = 2.0 * pyramid_search as f32;
                    let disagreement = (field_local_x - predicted[0]).abs();
                    if field_local_x.is_finite()
                        && disagreement > primary_reach_px + 1.0
                        && disagreement <= 56.0
                    {
                        let rescue_radius = ((horizontal_light_field.residual_px + 2.0) * 0.5)
                            .ceil()
                            .clamp(2.0, 4.0) as i32;
                        coarse_searches.push((
                            [field_local_x * 0.5, predicted_half[1]],
                            rescue_radius,
                            true,
                        ));
                        match_diagnostics.light_field_rescue_tracks += 1;
                    }
                }
            }
            let rescue_capacity = coarse_searches
                .get(1)
                .map_or(0, |(_, radius, _)| (2 * *radius + 1).pow(2) as usize);
            let mut coarse_candidates =
                Vec::with_capacity(search_diameter * search_diameter + rescue_capacity);
            for (search_center, search_radius, field_guided) in coarse_searches {
                for dy in -search_radius..=search_radius {
                    for dx in -search_radius..=search_radius {
                        let candidate_half =
                            [search_center[0] + dx as f32, search_center[1] + dy as f32];
                        let candidate = [candidate_half[0] * 2.0, candidate_half[1] * 2.0];
                        if !candidate_is_valid(candidate) {
                            continue;
                        }
                        let cost = patch_cost_with_integral_moments(
                            &previous_pyramid,
                            &current_pyramid,
                            &previous_pyramid_moments,
                            &current_pyramid_moments,
                            previous_half,
                            candidate_half,
                            pyramid_patch_radius,
                        );
                        match_diagnostics.coarse_patch_evaluations += 1;
                        match_diagnostics.light_field_rescue_evaluations +=
                            usize::from(field_guided);
                        if cost.is_finite() {
                            coarse_candidates.push((cost, candidate, field_guided));
                        }
                    }
                }
            }
            coarse_candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
            let mut basins = Vec::<([f32; 2], bool)>::new();
            for (_, candidate, field_guided) in coarse_candidates {
                if basins.iter().all(|(basin, _)| {
                    (basin[0] - candidate[0]).hypot(basin[1] - candidate[1]) >= 5.0
                }) {
                    basins.push((candidate, field_guided));
                }
                if basins.len() >= maximum_basins {
                    break;
                }
            }
            if basins.is_empty() {
                match_diagnostics.no_candidate += 1;
                continue;
            }
            let mut candidates = Vec::with_capacity(196);
            for (basin, field_guided) in basins {
                for dy in -refinement_radius..=refinement_radius {
                    for dx in -refinement_radius..=refinement_radius {
                        let candidate = [basin[0] + dx as f32, basin[1] + dy as f32];
                        if !candidate_is_valid(candidate) {
                            continue;
                        }
                        let full_cost = patch_cost_with_integral_moments(
                            previous,
                            &current,
                            &previous_moments,
                            &current_moments,
                            previous_local,
                            candidate,
                            native_patch_radius,
                        );
                        match_diagnostics.native_patch_evaluations += 1;
                        let half_cost = patch_cost_with_integral_moments(
                            &previous_pyramid,
                            &current_pyramid,
                            &previous_pyramid_moments,
                            &current_pyramid_moments,
                            previous_half,
                            [candidate[0] * 0.5, candidate[1] * 0.5],
                            pyramid_patch_radius,
                        );
                        if full_cost.is_finite() && half_cost.is_finite() {
                            candidates.push((
                                0.22 * full_cost + 0.78 * half_cost,
                                candidate,
                                field_guided,
                            ));
                        }
                    }
                }
            }
            let Some(best) = candidates
                .iter()
                .copied()
                .min_by(|left, right| left.0.total_cmp(&right.0))
            else {
                match_diagnostics.no_candidate += 1;
                continue;
            };
            // Adjacent samples on the same sub-pixel cost basin are not an
            // independent alternative. Comparing against them rejects good
            // Canny corners merely because the edge is smooth. Use the best
            // spatially distinct alternative for the ambiguity margin.
            let second = candidates
                .iter()
                .filter(|(_, candidate, _)| {
                    (candidate[0] - best.1[0]).hypot(candidate[1] - best.1[1])
                        >= MATCH_EXCLUSION_RADIUS
                })
                .map(|candidate| candidate.0)
                .min_by(f32::total_cmp)
                .unwrap_or(f32::INFINITY);
            let margin = if second.is_finite() {
                (second - best.0) / second.max(1.0e-5)
            } else {
                1.0
            };
            let current_half = [best.1[0] * 0.5, best.1[1] * 0.5];
            let backward = (-backward_radius..=backward_radius)
                .flat_map(|dy| (-backward_radius..=backward_radius).map(move |dx| (dx, dy)))
                .filter_map(|(dx, dy)| {
                    let candidate = [previous_half[0] + dx as f32, previous_half[1] + dy as f32];
                    let cost = patch_cost_with_integral_moments(
                        &current_pyramid,
                        &previous_pyramid,
                        &current_pyramid_moments,
                        &previous_pyramid_moments,
                        current_half,
                        candidate,
                        pyramid_patch_radius,
                    );
                    cost.is_finite().then_some((cost, candidate))
                })
                .min_by(|left, right| left.0.total_cmp(&right.0));
            let backward_consistent = backward.is_some_and(|(cost, point)| {
                cost <= MAX_MATCH_COST
                    && (point[0] - previous_half[0]).hypot(point[1] - previous_half[1]) <= 1.5
            });
            // Do not pre-reject a valid forward/backward track because its
            // velocity changed: those changes are precisely the signal used
            // below to learn separate temporal motion/parallax layers.
            let temporally_coherent = true;
            let cost_accepted = best.0 <= MAX_MATCH_COST;
            let margin_accepted = margin >= MIN_MATCH_MARGIN;
            if !cost_accepted {
                match_diagnostics.cost_rejected += 1;
            }
            if !margin_accepted {
                match_diagnostics.margin_rejected += 1;
            }
            if !backward_consistent {
                match_diagnostics.backward_rejected += 1;
            }
            if !temporally_coherent {
                match_diagnostics.temporal_rejected += 1;
            }
            if cost_accepted && margin_accepted && backward_consistent && temporally_coherent {
                match_diagnostics.light_field_rescue_accepted += usize::from(best.2);
                match_diagnostics.subpixel_attempted += 1;
                let subpixel = refine_native_zncc_subpixel(
                    previous,
                    &current,
                    &previous_moments,
                    &current_moments,
                    previous_local,
                    best.1,
                    native_patch_radius,
                    &mut match_diagnostics.native_patch_evaluations,
                );
                let current_local = if let Some(refined) = subpixel {
                    match_diagnostics.subpixel_accepted += 1;
                    match_diagnostics.subpixel_offset_sum +=
                        refined.correction[0].hypot(refined.correction[1]);
                    refined.current
                } else {
                    match_diagnostics.subpixel_rejected += 1;
                    best.1
                };
                matches.push(Match {
                    track_index,
                    previous: [last[0], last[1]],
                    current: [
                        current_local[0] + sensor_x as f32,
                        current_local[1] + sensor_y as f32,
                    ],
                    score: (1.0 - best.0).clamp(0.0, 1.0),
                    object: track.object,
                    z: last[2],
                    assignment_margin: 0.0,
                    layer_evidence: false,
                    normal_flow_evidence: false,
                    specularity: feature_specularity(&current, best.1),
                });
            }
        }
        let track_priorities = self
            .tracks
            .iter()
            .map(|track| {
                0.025 * (track.matched_streak.min(8) as f32 / 8.0)
                    + 0.010 * track.assignment_confidence.clamp(0.0, 1.0)
            })
            .collect::<Vec<_>>();
        match_diagnostics.destination_collision_rejected =
            enforce_unique_match_destinations(&mut matches, &track_priorities);
        match_diagnostics.accepted = matches.len();
        let extended_gap_recovery_ready = iris_seed.is_some()
            && horizontal_light_field.reliable
            && horizontal_light_field.confidence >= 0.28
            && horizontal_light_field.residual_px <= 3.5
            && horizontal_light_field.horizontal_coverage >= 0.38
            && horizontal_light_field.leaf_nodes >= 4;
        let nautilus_needed = !self.disable_nautilus_for_replay
            && use_canny_features
            && self.tracks.iter().enumerate().any(|(track_index, track)| {
                !matches.iter().any(|item| item.track_index == track_index)
                    && (track.age <= MAX_AGE || extended_gap_recovery_ready)
                    && self.nautilus_banks.get(&track.id).is_some_and(|bank| {
                        bank.observations.len() >= NAUTILUS_MIN_BANK_OBSERVATIONS
                    })
            });
        let nautilus_started = Instant::now();
        let nautilus_tree = if nautilus_needed {
            NautilusFingerprintTree::from_edges(&current, &edges)
        } else {
            NautilusFingerprintTree::default()
        };
        match_diagnostics.nautilus_build_micros = nautilus_started.elapsed().as_micros() as u64;
        match_diagnostics.nautilus_candidates = nautilus_tree.candidates.len();
        match_diagnostics.nautilus_tree_nodes = nautilus_tree.nodes.len();
        let mut nautilus_relocations = propose_nautilus_relocations(
            &current,
            &nautilus_tree,
            &self.tracks,
            &self.nautilus_banks,
            &matches,
            &self.motions,
            &self.layers,
            center,
            [sensor_x as f32, sensor_y as f32],
            prediction_cadence_contiguous,
            extended_gap_recovery_ready,
            &mut match_diagnostics,
        );
        // The extra dormant cache must not turn the 80-point live graph into
        // a 104-point graph on the re-entry frame. Established adjacent
        // matches and ordinary short-gap relocations keep their slots; only
        // the highest-confidence extended relocations fill the remainder.
        let established_regular = matches
            .iter()
            .filter(|item| {
                let track = &self.tracks[item.track_index];
                track.points.len() > 1 || track.matched_streak > 0
            })
            .count();
        let ordinary_relocations = nautilus_relocations
            .iter()
            .filter(|relocation| self.tracks[relocation.track_index].age <= MAX_AGE)
            .count();
        let maximum_extended = MAX_FEATURES
            .saturating_sub(established_regular)
            .saturating_sub(ordinary_relocations);
        let mut retained_extended = 0usize;
        let mut dropped_extended = Vec::new();
        nautilus_relocations.retain(|relocation| {
            if self.tracks[relocation.track_index].age <= MAX_AGE {
                return true;
            }
            if retained_extended < maximum_extended {
                retained_extended += 1;
                true
            } else {
                dropped_extended.push(*relocation);
                false
            }
        });
        for relocation in dropped_extended {
            match_diagnostics.nautilus_relocated =
                match_diagnostics.nautilus_relocated.saturating_sub(1);
            match_diagnostics.nautilus_gap_relocated =
                match_diagnostics.nautilus_gap_relocated.saturating_sub(1);
            match_diagnostics.nautilus_margin_sum =
                (match_diagnostics.nautilus_margin_sum - relocation.margin).max(0.0);
        }
        let extended_reentry_points = nautilus_relocations
            .iter()
            .filter(|relocation| self.tracks[relocation.track_index].age > MAX_AGE)
            .map(|relocation| relocation.current)
            .collect::<Vec<_>>();
        if !extended_reentry_points.is_empty() {
            let before = matches.len();
            matches.retain(|item| {
                let track = &self.tracks[item.track_index];
                let provisional = track.points.len() <= 1 && track.matched_streak == 0;
                !provisional
                    || extended_reentry_points.iter().all(|point| {
                        (point[0] - item.current[0]).hypot(point[1] - item.current[1])
                            >= MIN_MATCH_DESTINATION_SEPARATION
                    })
            });
            match_diagnostics.destination_collision_rejected += before - matches.len();
        }
        // If a returning established ID did not spatially overlap its
        // one-frame duplicate, retire the weakest other provisional seeds so
        // matches plus relocations remain within the invariant live budget.
        let regular_budget = MAX_FEATURES.saturating_sub(nautilus_relocations.len());
        if matches.len() > regular_budget {
            let fixed = matches
                .iter()
                .filter(|item| {
                    let track = &self.tracks[item.track_index];
                    track.points.len() > 1 || track.matched_streak > 0
                })
                .count();
            let provisional_budget = regular_budget.saturating_sub(fixed);
            let mut provisional = matches
                .iter()
                .filter(|item| {
                    let track = &self.tracks[item.track_index];
                    track.points.len() <= 1 && track.matched_streak == 0
                })
                .map(|item| (item.track_index, item.score))
                .collect::<Vec<_>>();
            provisional.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            provisional.truncate(provisional_budget);
            let retained_provisional = provisional
                .into_iter()
                .map(|candidate| candidate.0)
                .collect::<BTreeSet<_>>();
            let before = matches.len();
            matches.retain(|item| {
                let track = &self.tracks[item.track_index];
                track.points.len() > 1
                    || track.matched_streak > 0
                    || retained_provisional.contains(&item.track_index)
            });
            match_diagnostics.destination_collision_rejected += before - matches.len();
        }
        debug_assert!(matches.len() + nautilus_relocations.len() <= MAX_FEATURES);
        match_diagnostics.accepted = matches.len();
        match_diagnostics.matching_micros = matching_started.elapsed().as_micros() as u64;
        let layering_started = Instant::now();
        let global = robust_global_similarity(&matches, center);
        let relation_started = Instant::now();
        let mut motion_relations =
            self.motion_relations
                .observe(&matches, &self.tracks, center, global);
        match_diagnostics.relation_micros = relation_started.elapsed().as_micros() as u64;
        let relation_eye_region = iris_seed
            .map(|seed| EyeMotionRegion::from_local_seed(seed, &current))
            .or(self.semantic_eye_region);
        if relation_graph_has_persistent_component(&motion_relations)
            || self.relation_iris_identity.active()
        {
            if let Some(region) = relation_eye_region {
                let candidates = matches
                    .iter()
                    .enumerate()
                    .filter_map(|(index, item)| {
                        region.contains_scaled(item.current, 1.10).then_some(index)
                    })
                    .collect::<Vec<_>>();
                let _ = relation_graph_iris_core(
                    &candidates,
                    &matches,
                    &self.tracks,
                    global,
                    center,
                    region,
                    &mut motion_relations,
                    Some(&self.relation_iris_identity),
                    MIN_LAYER_SUPPORT,
                );
            }
        }
        let mut radial_limbus_fused = 0usize;
        let semantic_layers = cluster_semantic_eye_layers(
            &mut matches,
            &self.tracks,
            &current,
            &edges,
            iris_seed,
            &mut self.motions,
            &mut self.layers,
            &mut self.layer_signatures,
            &mut self.parallax_axis,
            &mut self.semantic_eye_center,
            &mut self.semantic_eye_region,
            &mut radial_limbus_flows,
            &mut radial_limbus_fused,
            center,
            global,
            &mut motion_relations,
            Some(&self.relation_iris_identity),
        );
        match_diagnostics.radial_limbus_fused = radial_limbus_fused;
        if !semantic_layers {
            let relation_layers = cluster_relation_motion_layers(
                &mut matches,
                &self.tracks,
                &mut motion_relations,
                relation_eye_region,
                &mut self.motions,
                &mut self.layers,
                &mut self.layer_signatures,
                &mut self.parallax_axis,
                center,
                global,
                Some(&self.relation_iris_identity),
            );
            if !relation_layers {
                cluster_motion_layers(
                    &mut matches,
                    &self.tracks,
                    &mut self.motions,
                    &mut self.layers,
                    &mut self.layer_signatures,
                    &mut self.parallax_axis,
                    center,
                    global,
                );
            }
        }
        let identity_continuity = RelationIrisIdentityContinuity {
            track_overlap: motion_relations.selected_identity_overlap,
            origin_consistent: motion_relations.selected_origin_consistent,
            ..RelationIrisIdentityContinuity::default()
        };
        // The first geometrically plausible component is deliberately only an
        // observation.  Feed that provisional cohort to the identity state so
        // a second exact material observation can confirm it; anatomical
        // selection remains gated by `selected_iris_component`.
        self.relation_iris_identity.observe(
            motion_relations.observed_iris(),
            identity_continuity,
            motion_relations.observed_motion_evidence,
        );
        match_diagnostics.relation_nodes = motion_relations.node_match_indices.len();
        match_diagnostics.relation_edges = motion_relations.edges.len();
        match_diagnostics.relation_recurrent_edges = motion_relations.recurrent_edge_count();
        match_diagnostics.relation_supported_edges = motion_relations.supported_edge_count();
        match_diagnostics.relation_precise_edges = motion_relations.precise_edge_count();
        match_diagnostics.relation_coherent_edges = motion_relations.coherent_edge_count();
        match_diagnostics.relation_components = motion_relations.components.len();
        match_diagnostics.relation_persistent_components =
            motion_relations.persistent_component_count();
        (
            match_diagnostics.relation_max_component_persistent_edges,
            match_diagnostics.relation_max_component_persistent_nodes,
        ) = motion_relations.maximum_component_persistence();
        match_diagnostics.relation_max_persistent_differential_px =
            maximum_persistent_component_differential(&motion_relations, &matches, global, center);
        match_diagnostics.relation_iris_identity_overlap =
            motion_relations.selected_identity_overlap;
        match_diagnostics.relation_iris_identity_age = self.relation_iris_identity.age;
        match_diagnostics.relation_iris_identity_confirmations =
            self.relation_iris_identity.confirmations;
        match_diagnostics.relation_iris_identity_evidence = self.relation_iris_identity.evidence;
        match_diagnostics.relation_iris_identity_confirmed =
            self.relation_iris_identity.confirmed();
        match_diagnostics.relation_iris_candidates = motion_relations.iris_candidate_diagnostics;
        match_diagnostics.relation_iris_identity_switch_rejections =
            motion_relations.identity_switch_rejections;
        match_diagnostics.relation_iris_initial_origin_rejections =
            motion_relations.initial_origin_rejections;
        match_diagnostics.relation_iris_identity_carried =
            motion_relations.selected_by_identity_carry;
        match_diagnostics.relation_origin_outlier_rejected =
            motion_relations.selected_iris_component.is_some()
                && !motion_relations.selected_origin_consistent;
        if motion_relations.selected_iris_component.is_none() {
            match_diagnostics.relation_iris_provisional_support = motion_relations
                .observed_iris()
                .map_or(0, |component| component.members.len());
        }
        match_diagnostics.relation_max_shared_frames = motion_relations.maximum_shared_frames();
        match_diagnostics.relation_max_coherence = motion_relations.maximum_coherence();
        (
            match_diagnostics.relation_mean_recurrent_coherence,
            match_diagnostics.relation_mean_recurrent_residual,
            match_diagnostics.relation_mean_support_continuity,
        ) = motion_relations.mean_recurrent_quality();
        if let Some(component) = motion_relations.selected_iris() {
            match_diagnostics.relation_iris_support = component.members.len();
            match_diagnostics.relation_origin = component.shared_origin;
            match_diagnostics.relation_origin_spread = component.origin_spread;
            match_diagnostics.relation_origin_valid =
                component.origin_valid && motion_relations.selected_origin_consistent;
        }
        let limbus_normal_flow_support = matches
            .iter()
            .filter(|item| item.normal_flow_evidence)
            .count()
            + radial_limbus_fused;
        let iris_geometry = self.semantic_eye_region.map(|region| {
            let pupil_layer = self.layers[PUPIL_LAYER];
            let track_support = (pupil_layer.persistent_tracks as f64 / 8.0).clamp(0.0, 1.0);
            let edge_independence = if iris_seed.is_some() {
                1.0
            } else {
                0.35 + 0.65 * (limbus_normal_flow_support as f64 / 4.0).clamp(0.0, 1.0)
            };
            ProjectedIrisGeometry {
                center: [region.center[0] as f64, region.center[1] as f64],
                major_radius: region.major as f64,
                minor_radius: region.minor as f64,
                angle_rad: region.angle as f64,
                confidence: (pupil_layer.coherence as f64
                    * track_support.sqrt()
                    * edge_independence)
                    .clamp(0.0, 1.0),
                anatomy_authorized: semantic_layers
                    && pupil_layer.stable_frames >= 2
                    && (iris_seed.is_some() || limbus_normal_flow_support >= 2),
            }
        });
        self.coupled_kinematics.observe(
            timestamp_ns,
            center,
            width.max(height) as f64,
            self.motions,
            self.layers,
            semantic_layers,
            iris_geometry,
        );
        match_diagnostics.layering_micros = layering_started.elapsed().as_micros() as u64;
        let maintenance_started = Instant::now();
        if let Some(probe) = focus_probe {
            for item in &matches {
                let point = [
                    item.current[0] - sensor_x as f32,
                    item.current[1] - sensor_y as f32,
                ];
                add_focus_sample(
                    &mut self.tracks[item.track_index],
                    probe.position,
                    feature_sharpness(&current, point),
                );
            }
        }
        let depth_motion = matches
            .iter()
            .filter_map(|item| {
                self.tracks[item.track_index]
                    .focus_peak
                    .map(|depth| (depth, item.z.abs()))
            })
            .collect::<Vec<_>>();
        self.focus_sfm.observe_motion(&depth_motion);
        let mut seen = vec![false; self.tracks.len()];
        for item in matches {
            let residual = residual_motion(&item, global, center);
            let track = &mut self.tracks[item.track_index];
            let instantaneous = [
                item.current[0] - item.previous[0],
                item.current[1] - item.previous[1],
            ];
            let motion_error = (instantaneous[0] - track.motion_ema[0])
                .hypot(instantaneous[1] - track.motion_ema[1]);
            let alpha = if track.matched_streak == 0 { 1.0 } else { 0.28 };
            track.motion_ema = [
                track.motion_ema[0] * (1.0 - alpha) + instantaneous[0] * alpha,
                track.motion_ema[1] * (1.0 - alpha) + instantaneous[1] * alpha,
            ];
            track.motion_variance = if track.matched_streak == 0 {
                0.0
            } else {
                0.78 * track.motion_variance + 0.22 * motion_error * motion_error
            };
            track.object = item.object;
            track.age = 0;
            track.last_seen_timestamp_ns = timestamp_ns;
            track.score = item.score;
            let specularity_alpha = if track.matched_streak == 0 { 1.0 } else { 0.35 };
            track.specularity = track.specularity * (1.0 - specularity_alpha)
                + item.specularity * specularity_alpha;
            track.matched_streak = track.matched_streak.saturating_add(1);
            track.layer_evidence = item.layer_evidence;
            track.normal_flow_evidence = item.normal_flow_evidence;
            track.assignment_confidence =
                0.72 * track.assignment_confidence + 0.28 * item.assignment_margin.clamp(0.0, 1.0);
            track.residual_history.push_back(residual);
            while track.residual_history.len() > MOTION_SIGNATURE_LEN {
                track.residual_history.pop_front();
            }
            if let Some(field) = canny.as_ref() {
                let local_x = (item.current[0] - sensor_x as f32)
                    .round()
                    .clamp(0.0, width.saturating_sub(1) as f32)
                    as usize;
                let local_y = (item.current[1] - sensor_y as f32)
                    .round()
                    .clamp(0.0, height.saturating_sub(1) as f32)
                    as usize;
                let mut observed = local_canny_normal(field, width, local_x, local_y);
                if observed[0].hypot(observed[1]) > 0.5 {
                    // Edge orientation is polarity-insensitive for tracking.
                    // Keep the stored normal in a consistent hemisphere so
                    // alternating signed Canny polarity cannot cancel it.
                    if observed[0] * track.edge_normal[0] + observed[1] * track.edge_normal[1] < 0.0
                    {
                        observed = [-observed[0], -observed[1]];
                    }
                    track.edge_normal = normalized_vector([
                        0.72 * track.edge_normal[0] + 0.28 * observed[0],
                        0.72 * track.edge_normal[1] + 0.28 * observed[1],
                    ]);
                }
            }
            track
                .points
                .push_back([item.current[0], item.current[1], item.z]);
            while track.points.len() > MAX_TRAIL {
                track.points.pop_front();
            }
            let fingerprint_point = [
                item.current[0] - sensor_x as f32,
                item.current[1] - sensor_y as f32,
            ];
            let fingerprint = nautilus_fingerprint(&current, fingerprint_point, track.edge_normal);
            let track_id = track.id;
            seen[item.track_index] = true;
            if let Some(fingerprint) = fingerprint {
                self.nautilus_banks
                    .entry(track_id)
                    .or_default()
                    .observe(fingerprint);
            }
        }
        // Re-identified observations preserve the feature ID but are kept out
        // of this frame's global/layer fit: a displacement spanning two or
        // more exposures is not a one-frame velocity tensor. The next normal
        // adjacent match can again contribute motion evidence.
        for relocation in nautilus_relocations {
            if seen.get(relocation.track_index).copied().unwrap_or(true) {
                continue;
            }
            let candidate = nautilus_tree.candidates[relocation.candidate_index];
            let track = &mut self.tracks[relocation.track_index];
            let crossed_extended_gap = track.age > MAX_AGE;
            let previous = track.points.back().copied().unwrap_or([
                relocation.current[0],
                relocation.current[1],
                0.0,
            ]);
            let elapsed_frames = track.age as f32 + 1.0;
            let instantaneous = [
                (relocation.current[0] - previous[0]) / elapsed_frames,
                (relocation.current[1] - previous[1]) / elapsed_frames,
            ];
            let motion_error = (instantaneous[0] - track.motion_ema[0])
                .hypot(instantaneous[1] - track.motion_ema[1]);
            track.motion_ema = [
                0.82 * track.motion_ema[0] + 0.18 * instantaneous[0],
                0.82 * track.motion_ema[1] + 0.18 * instantaneous[1],
            ];
            track.motion_variance =
                0.88 * track.motion_variance + 0.12 * motion_error * motion_error;
            track.age = 0;
            track.last_seen_timestamp_ns = timestamp_ns;
            track.score = (1.0 - relocation.distance / NAUTILUS_MAX_DISTANCE).clamp(0.0, 1.0);
            // This is an exact current observation of the same identity, but
            // it is not adjacent motion. Start a fresh visible trail segment;
            // the next ordinary match establishes continuity again.
            track.matched_streak = 0;
            track.layer_evidence = false;
            track.normal_flow_evidence = false;
            track.assignment_confidence =
                0.60 * track.assignment_confidence + 0.40 * relocation.confidence;
            track.residual_history.clear();
            let mut observed_normal = candidate.normal;
            if observed_normal[0] * track.edge_normal[0] + observed_normal[1] * track.edge_normal[1]
                < 0.0
            {
                observed_normal = [-observed_normal[0], -observed_normal[1]];
            }
            track.edge_normal = normalized_vector([
                0.72 * track.edge_normal[0] + 0.28 * observed_normal[0],
                0.72 * track.edge_normal[1] + 0.28 * observed_normal[1],
            ]);
            let local = [
                relocation.current[0] - sensor_x as f32,
                relocation.current[1] - sensor_y as f32,
            ];
            track.specularity =
                0.65 * track.specularity + 0.35 * feature_specularity(&current, local);
            track
                .points
                .push_back([relocation.current[0], relocation.current[1], previous[2]]);
            while track.points.len() > MAX_TRAIL {
                track.points.pop_front();
            }
            let track_id = track.id;
            seen[relocation.track_index] = true;
            self.nautilus_banks
                .entry(track_id)
                .or_default()
                .observe(relocation.fingerprint);
            match_diagnostics.extended_gap_relocated += usize::from(crossed_extended_gap);
        }
        for (index, track) in self.tracks.iter_mut().enumerate() {
            if !seen.get(index).copied().unwrap_or(false) {
                track.age = track.age.saturating_add(1);
                track.matched_streak = 0;
                track.layer_evidence = false;
                track.normal_flow_evidence = false;
                track.assignment_confidence *= 0.8;
                track.residual_history.clear();
            }
        }
        self.tracks.retain(|track| {
            track.age <= MAX_FEATURE_DORMANT_AGE
                && timestamp_ns.saturating_sub(track.last_seen_timestamp_ns)
                    <= MAX_FEATURE_DORMANT_NS
        });
        // Extended dormancy is an identity cache, not part of the live
        // detector population. Only fingerprints with enough independent
        // observations can ever pass the Nautilus identity gate, so discard
        // weaker long-gap tracks and bound the cache by maturity/quality.
        let mut extended_dormant = self
            .tracks
            .iter()
            .filter(|track| track.age > MAX_AGE)
            .filter_map(|track| {
                let bank_observations = self
                    .nautilus_banks
                    .get(&track.id)
                    .map_or(0, |bank| bank.observations.len());
                (bank_observations >= NAUTILUS_MIN_BANK_OBSERVATIONS).then_some((
                    track.id,
                    bank_observations,
                    track.points.len(),
                    track.assignment_confidence,
                    track.score,
                    track.age,
                ))
            })
            .collect::<Vec<_>>();
        extended_dormant.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| right.2.cmp(&left.2))
                .then_with(|| right.3.total_cmp(&left.3))
                .then_with(|| right.4.total_cmp(&left.4))
                .then_with(|| left.5.cmp(&right.5))
                .then_with(|| left.0.cmp(&right.0))
        });
        extended_dormant.truncate(MAX_EXTENDED_DORMANT_FEATURES);
        let retained_extended_ids = extended_dormant
            .into_iter()
            .map(|candidate| candidate.0)
            .collect::<BTreeSet<_>>();
        self.tracks
            .retain(|track| track.age <= MAX_AGE || retained_extended_ids.contains(&track.id));
        match_diagnostics.dormant_tracks = self.tracks.iter().filter(|track| track.age > 0).count();
        let retained_ids = self.tracks.iter().map(|track| track.id).collect::<Vec<_>>();
        self.nautilus_banks
            .retain(|id, _| retained_ids.contains(id));
        let existing = self
            .tracks
            .iter()
            .filter_map(|track| track.points.back())
            .map(|point| [point[0] - sensor_x as f32, point[1] - sensor_y as f32])
            .collect::<Vec<_>>();
        // Long-gap identities live in a separate bounded cache. They must not
        // consume all eighty current/recent detector slots after an unrelated
        // patch loss; otherwise no fresh evidence can enter the motion graph.
        let current_or_recent = self
            .tracks
            .iter()
            .filter(|track| track.age <= MAX_AGE)
            .count();
        let wanted = MAX_FEATURES.saturating_sub(current_or_recent);
        let feature_iris_seed = iris_seed.or_else(|| {
            self.semantic_eye_region
                .map(|region| region.local_seed(&current))
        });
        for (point, score) in seed_points(
            &current,
            canny.as_ref(),
            &edges,
            feature_iris_seed,
            &existing,
            wanted,
        ) {
            let edge_normal = canny.as_ref().map_or([0.0; 2], |field| {
                local_canny_normal(
                    field,
                    width,
                    point[0].round() as usize,
                    point[1].round() as usize,
                )
            });
            let mut track = FeatureTrack {
                id: self.next_id,
                points: VecDeque::from([[
                    point[0] + sensor_x as f32,
                    point[1] + sensor_y as f32,
                    0.0,
                ]]),
                object: 0,
                age: 0,
                last_seen_timestamp_ns: timestamp_ns,
                score,
                motion_ema: [0.0; 2],
                motion_variance: 0.0,
                matched_streak: 0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: feature_specularity(&current, point),
                assignment_confidence: 0.0,
                edge_normal,
                residual_history: VecDeque::new(),
                focus_bins: Vec::new(),
                focus_peak: None,
            };
            if let Some(probe) = focus_probe {
                add_focus_sample(
                    &mut track,
                    probe.position,
                    feature_sharpness(&current, point),
                );
            }
            if let Some(fingerprint) = nautilus_fingerprint(&current, point, edge_normal) {
                self.nautilus_banks
                    .entry(track.id)
                    .or_default()
                    .observe(fingerprint);
            }
            self.tracks.push(track);
            self.next_id += 1;
        }
        // Apply the same motion-conditioned upper-edge reliability used by
        // the bounded Driving bank to the full 2D temporal-Canny overlay.
        // This happens only after a persistent, coherent pupil/iris motion
        // layer exists; cold/static frames retain unit weights because
        // motion cannot honestly distinguish an eyebrow shadow from anatomy.
        // The current RAW allocation and its native coordinates are reused
        // directly -- no resized image or extra inference representation is
        // introduced here.
        let motion_shadow_edges_downweighted = self
            .previous
            .as_ref()
            .zip(self.semantic_eye_region)
            .filter(|_| {
                let layer = self.layers[PUPIL_LAYER];
                layer.stable_frames >= 2 && layer.persistent_tracks >= 4 && layer.coherence >= 0.45
            })
            .map_or(0, |(previous, region)| {
                condition_upper_edges_by_iris_motion(
                    &mut edges,
                    previous,
                    &current,
                    region.local_seed(&current),
                    self.motions[PUPIL_LAYER],
                    region.center,
                )
            });
        self.generation = self.generation.saturating_add(1);
        self.horizontal_light_field = horizontal_light_field;
        self.radial_limbus_region = iris_seed
            .map(|seed| EyeMotionRegion::from_local_seed(seed, &current))
            .or(self.semantic_eye_region)
            .or(current_radial_region);
        self.radial_limbus_flows = radial_limbus_flows;
        self.previous = Some(current);
        match_diagnostics.maintenance_micros = maintenance_started.elapsed().as_micros() as u64;
        self.match_diagnostics = match_diagnostics;
        self.overlay(
            sensor_x,
            sensor_y,
            edges,
            edge_high_threshold,
            motion_shadow_edges_downweighted,
            learning_canny_mask,
        )
    }

    fn overlay(
        &self,
        sensor_x: u32,
        sensor_y: u32,
        edges: Vec<EdgeEvidence>,
        edge_high_threshold: f32,
        motion_shadow_edges_downweighted: usize,
        learning_canny_mask: Vec<u8>,
    ) -> MotionOctreeOverlay {
        let trails = self
            .tracks
            .iter()
            .filter(|track| track.age == 0 && track.points.len() >= 2)
            .map(|track| {
                // Never render or semantically consume a stale endpoint as
                // though it were measured in this exposure. After a Nautilus
                // cross-gap re-identification `matched_streak == 0`, so only
                // its exact current point is exported and no fictitious line
                // is drawn across the occluded interval.
                let visible_points = usize::from(track.matched_streak)
                    .saturating_add(1)
                    .min(track.points.len());
                OverlayTrail {
                    id: track.id,
                    object: track.object,
                    match_score: track.score,
                    matched_streak: track.matched_streak,
                    layer_evidence: track.layer_evidence,
                    normal_flow_evidence: track.normal_flow_evidence,
                    specularity: track.specularity,
                    assignment_confidence: track.assignment_confidence,
                    motion_ema: track.motion_ema,
                    motion_variance: track.motion_variance,
                    residual_history: track.residual_history.iter().copied().collect(),
                    points: track
                        .points
                        .iter()
                        .skip(track.points.len() - visible_points)
                        .map(|point| TrailPoint {
                            x: point[0] - sensor_x as f32,
                            y: point[1] - sensor_y as f32,
                            z: point[2],
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let latest = self
            .tracks
            .iter()
            .filter(|track| track.age == 0)
            .filter_map(|track| {
                track.points.back().map(|point| {
                    (
                        track.object,
                        [
                            point[0] - sensor_x as f32,
                            point[1] - sensor_y as f32,
                            point[2],
                        ],
                    )
                })
            })
            .collect::<Vec<_>>();
        let provisional_features = self
            .tracks
            .iter()
            .filter(|track| track.age == 0 && track.points.len() == 1)
            .filter_map(|track| track.points.back())
            .map(|point| (point[0] - sensor_x as f32, point[1] - sensor_y as f32))
            .collect::<Vec<_>>();
        let mut nodes = Vec::new();
        rebuild_nodes(&latest, &mut nodes);
        let mut layers = self.layers;
        for layer in &mut layers {
            layer.centroid[0] -= sensor_x as f32;
            layer.centroid[1] -= sensor_y as f32;
        }
        let semantic_iris = self.semantic_eye_region.map(|region| IrisEllipseSeed {
            center: (
                (region.center[0] - sensor_x as f32) as f64,
                (region.center[1] - sensor_y as f32) as f64,
            ),
            major_radius: region.major as f64,
            minor_radius: region.minor as f64,
            angle: region.angle as f64,
        });
        let radial_limbus_probes = self
            .radial_limbus_flows
            .iter()
            .map(|flow| RadialLimbusProbe {
                point: [
                    flow.current[0] - sensor_x as f32,
                    flow.current[1] - sensor_y as f32,
                ],
                normal: flow.normal,
                phase_rad: flow.phase_rad,
                radial_shift_px: flow.radial_shift_px,
                profile_cost: flow.profile_cost,
                confidence: flow.confidence,
                fused: flow.fused,
            })
            .collect::<Vec<_>>();
        MotionOctreeOverlay {
            generation: self.generation,
            matched_features: trails.len(),
            match_diagnostics: self.match_diagnostics,
            provisional_features,
            active_objects: self
                .layers
                .iter()
                .enumerate()
                .filter(|(object, layer)| {
                    let minimum_support = if *object == REFLECTION_LAYER {
                        MIN_REFLECTION_SUPPORT
                    } else {
                        MIN_LAYER_SUPPORT
                    };
                    layer.stable_frames >= MIN_LAYER_STABLE_FRAMES
                        && layer.persistent_tracks >= minimum_support
                        && self.motions[*object].support >= minimum_support
                })
                .count(),
            trails,
            nodes,
            motions: self.motions,
            layers,
            parallax_axis: self.parallax_axis,
            learning_canny_mask,
            learning_canny_profile: self.learning_canny_profile,
            edges,
            edge_high_threshold,
            motion_shadow_edges_downweighted,
            horizontal_light_field: self.horizontal_light_field.clone(),
            radial_limbus_probes,
            semantic_iris,
            focus_sfm: self.focus_sfm.status,
            coupled_motion: self
                .coupled_kinematics
                .status()
                .translated(sensor_x as f32, sensor_y as f32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_translucent_limbus_frame(
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        region: EyeMotionRegion,
        gain: f32,
        offset: f32,
    ) -> RawFrame {
        let (axis_sine, axis_cosine) = region.angle.sin_cos();
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let sensor_point = [x as f32 + sensor_x as f32, y as f32 + sensor_y as f32];
                    let dx = sensor_point[0] - region.center[0];
                    let dy = sensor_point[1] - region.center[1];
                    let local_x = axis_cosine * dx + axis_sine * dy;
                    let local_y = -axis_sine * dx + axis_cosine * dy;
                    let normalized_radius = (local_x / region.major).hypot(local_y / region.minor);
                    let phase = (local_y / region.minor).atan2(local_x / region.major);
                    // A broad sigmoid models the gray, semi-opaque transition;
                    // two angular/radial carriers model iris material that
                    // changes visibility under an oblique projection.
                    let transition =
                        (1.0 + (-(normalized_radius - 1.0) * region.minor / 3.2).exp()).recip();
                    let iris_texture = (1.0 - transition)
                        * (18.0 * (7.0 * phase + 0.11 * local_x).sin()
                            + 10.0 * (0.19 * local_y - 3.0 * phase).cos());
                    let layered_haze = (1.0 - transition)
                        * 20.0
                        * (0.5 + 0.5 * phase.cos()).powi(2)
                        * (0.08 * local_x).sin();
                    let value =
                        gain * (230.0 + 245.0 * transition + iris_texture + layered_haze) + offset;
                    value.round().clamp(0.0, 1023.0) as u16
                })
            })
            .collect::<Vec<_>>();
        RawFrame {
            sensor_x,
            sensor_y,
            width,
            height,
            pixels,
        }
    }

    #[test]
    fn radial_limbus_bands_survive_exposure_and_affine_perspective_change() {
        let width = 220;
        let height = 170;
        let sensor_x = 4_000;
        let sensor_y = 3_000;
        let previous_region = EyeMotionRegion {
            center: [sensor_x as f32 + 108.0, sensor_y as f32 + 84.0],
            major: 55.0,
            minor: 37.0,
            angle: -0.04,
        };
        let current_region = EyeMotionRegion {
            center: [sensor_x as f32 + 113.0, sensor_y as f32 + 81.5],
            major: 58.0,
            minor: 33.5,
            angle: 0.085,
        };
        let previous = synthetic_translucent_limbus_frame(
            width,
            height,
            sensor_x,
            sensor_y,
            previous_region,
            1.0,
            0.0,
        );
        let current = synthetic_translucent_limbus_frame(
            width,
            height,
            sensor_x,
            sensor_y,
            current_region,
            1.28,
            37.0,
        );
        let (flows, evaluations, micros) =
            radial_limbus_flows(&previous, &current, previous_region, current_region);
        assert!(flows.len() >= 16, "flows={} {flows:#?}", flows.len());
        assert!(evaluations >= flows.len() * 12, "evaluations={evaluations}");
        let mut boundary_errors = flows
            .iter()
            .map(|flow| {
                let expected = eye_region_boundary(current_region, flow.phase_rad);
                (flow.current[0] - expected[0]).hypot(flow.current[1] - expected[1])
            })
            .collect::<Vec<_>>();
        let boundary_error = median(&mut boundary_errors);
        assert!(
            boundary_error <= 2.5,
            "median={boundary_error} flows={flows:#?}"
        );
        assert!(
            flows
                .iter()
                .all(|flow| flow.profile_cost <= RADIAL_LIMBUS_MAX_PROFILE_COST),
            "{flows:#?}"
        );
        assert!(micros < 50_000, "radial matcher took {micros}us");
    }

    #[test]
    fn radial_normal_flows_add_scale_without_claiming_tangential_identity() {
        let center = [100.0f32, 80.0f32];
        let truth = SimilarityMotion {
            translation: [2.0, -1.5],
            scale_delta: 0.035,
            // A radial band contributes normal flow. It should sharpen
            // translation/scale, but must not invent tangential identity or
            // angular motion on an otherwise featureless circular contour.
            rotation: 0.0,
            ..SimilarityMotion::default()
        };
        let matches = [[-5.0f32, -4.0], [5.0, -4.0], [5.0, 4.0], [-5.0, 4.0]]
            .into_iter()
            .enumerate()
            .map(|(track_index, offset)| {
                let previous = [center[0] + offset[0], center[1] + offset[1]];
                Match {
                    track_index,
                    previous,
                    current: [previous[0] + 2.0, previous[1] - 1.5],
                    score: 1.0,
                    object: PUPIL_LAYER,
                    z: 0.0,
                    assignment_margin: 1.0,
                    layer_evidence: true,
                    normal_flow_evidence: false,
                    specularity: 0.0,
                }
            })
            .collect::<Vec<_>>();
        let radial = (0..12)
            .map(|index| {
                let phase = std::f32::consts::TAU * index as f32 / 12.0;
                let normal = [phase.cos(), phase.sin()];
                let previous = [center[0] + 50.0 * normal[0], center[1] + 38.0 * normal[1]];
                RadialLimbusFlow {
                    previous,
                    current: truth.predict(previous, center),
                    normal,
                    phase_rad: phase,
                    radial_shift_px: 0.0,
                    profile_cost: 0.08,
                    confidence: 0.9,
                    fused: true,
                }
            })
            .collect::<Vec<_>>();
        let fitted = fit_similarity_with_normal_constraints(
            &[0, 1, 2, 3],
            &[],
            &radial,
            &matches,
            &[],
            center,
        );
        assert!(
            (fitted.translation[0] - truth.translation[0]).abs() <= 0.25,
            "{fitted:#?}"
        );
        assert!(
            (fitted.translation[1] - truth.translation[1]).abs() <= 0.25,
            "{fitted:#?}"
        );
        assert!(
            (fitted.scale_delta - truth.scale_delta).abs() <= 0.010,
            "{fitted:#?}"
        );
        assert!(
            (fitted.rotation - truth.rotation).abs() <= 0.010,
            "{fitted:#?}"
        );
        assert_eq!(fitted.support, 16);
    }

    #[test]
    #[ignore = "set BUTTERCUP_RADIAL_LIMBUS_REPLAY to a lossless RAW replay directory containing report.json"]
    fn radial_limbus_bands_replay_consecutive_lossless_raw() {
        use std::fs;
        use std::path::PathBuf;

        let root = std::env::var_os("BUTTERCUP_RADIAL_LIMBUS_REPLAY")
            .map(PathBuf::from)
            .expect("BUTTERCUP_RADIAL_LIMBUS_REPLAY is required");
        let report: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(root.join("report.json")).expect("read report.json"),
        )
        .expect("parse report.json");
        let records = report["frames"].as_array().expect("report frames");
        let mut raw_paths = fs::read_dir(&root)
            .expect("read replay directory")
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "raw10")
            })
            .collect::<Vec<_>>();
        raw_paths.sort();
        let replay_frames = raw_paths
            .into_iter()
            .filter_map(|path| {
                let ordinal = path
                    .file_stem()?
                    .to_str()?
                    .strip_prefix("frame-")?
                    .parse::<usize>()
                    .ok()?;
                records.get(ordinal).map(|record| (ordinal, path, record))
            })
            .collect::<Vec<_>>();
        assert!(
            replay_frames.len() >= 8,
            "mapped replay frames={}",
            replay_frames.len()
        );

        let region_for = |record: &serde_json::Value| {
            let seed = record.get("semantic_iris")?;
            (!seed.is_null()).then(|| {
                let origin = record["sensor_origin"].as_array().expect("sensor origin");
                let sensor_x = origin[0].as_u64().expect("sensor x") as u32;
                let sensor_y = origin[1].as_u64().expect("sensor y") as u32;
                let center = seed["center"].as_array().expect("iris center");
                let ellipse = IrisEllipseSeed {
                    center: (
                        center[0].as_f64().expect("iris center x"),
                        center[1].as_f64().expect("iris center y"),
                    ),
                    major_radius: seed["major_radius"].as_f64().expect("iris major"),
                    minor_radius: seed["minor_radius"].as_f64().expect("iris minor"),
                    angle: seed["angle"].as_f64().expect("iris angle"),
                };
                (
                    sensor_x,
                    sensor_y,
                    EyeMotionRegion::from_local_seed(
                        ellipse,
                        &RawFrame {
                            sensor_x,
                            sensor_y,
                            width: 384,
                            height: 256,
                            pixels: Vec::new(),
                        },
                    ),
                )
            })
        };

        let mut attempted_pairs = 0usize;
        let mut accepted_pairs = 0usize;
        let mut accepted_spokes = Vec::<usize>::new();
        let mut boundary_errors = Vec::<f32>::new();
        let mut profile_costs = Vec::<f32>::new();
        let mut timings = Vec::<u64>::new();
        let mut search_limit_spokes = 0usize;
        for pair in replay_frames.windows(2) {
            let [(previous_ordinal, previous_path, previous_record), (current_ordinal, current_path, current_record)] =
                pair
            else {
                unreachable!();
            };
            if *current_ordinal != previous_ordinal + 1 {
                continue;
            }
            let Some((previous_x, previous_y, previous_region)) = region_for(previous_record)
            else {
                continue;
            };
            let Some((current_x, current_y, current_region)) = region_for(current_record) else {
                continue;
            };
            let previous_payload = fs::read(previous_path).expect("read previous RAW");
            let current_payload = fs::read(current_path).expect("read current RAW");
            let previous_pixels = crate::raw10::unpack_raw10(&previous_payload, 384, 256, 480);
            let current_pixels = crate::raw10::unpack_raw10(&current_payload, 384, 256, 480);
            let previous = RawFrame {
                sensor_x: previous_x,
                sensor_y: previous_y,
                width: 384,
                height: 256,
                pixels: cfa_neutral_raw(&previous_pixels, 384, 256),
            };
            let current = RawFrame {
                sensor_x: current_x,
                sensor_y: current_y,
                width: 384,
                height: 256,
                pixels: cfa_neutral_raw(&current_pixels, 384, 256),
            };
            attempted_pairs += 1;
            let (flows, _, micros) =
                radial_limbus_flows(&previous, &current, previous_region, current_region);
            timings.push(micros);
            if !flows.is_empty() {
                accepted_pairs += 1;
                accepted_spokes.push(flows.len());
            }
            for flow in flows {
                search_limit_spokes += usize::from(
                    flow.radial_shift_px.abs() >= RADIAL_LIMBUS_CURRENT_SEARCH_PX as f32 - 0.5,
                );
                let expected = eye_region_boundary(current_region, flow.phase_rad);
                boundary_errors
                    .push((flow.current[0] - expected[0]).hypot(flow.current[1] - expected[1]));
                profile_costs.push(flow.profile_cost);
            }
        }
        let quantile = |values: &mut Vec<f32>, fraction: f32| {
            values.sort_by(f32::total_cmp);
            values
                .get(((values.len().saturating_sub(1)) as f32 * fraction).round() as usize)
                .copied()
                .unwrap_or(0.0)
        };
        let timing_quantile = |values: &mut Vec<u64>, fraction: f32| {
            values.sort_unstable();
            values
                .get(((values.len().saturating_sub(1)) as f32 * fraction).round() as usize)
                .copied()
                .unwrap_or(0)
        };
        let spokes_mean =
            accepted_spokes.iter().sum::<usize>() as f32 / accepted_spokes.len().max(1) as f32;
        let boundary_p50 = quantile(&mut boundary_errors, 0.50);
        let boundary_p90 = quantile(&mut boundary_errors, 0.90);
        let cost_p50 = quantile(&mut profile_costs, 0.50);
        let cost_p90 = quantile(&mut profile_costs, 0.90);
        let timing_p50 = timing_quantile(&mut timings, 0.50);
        let timing_p99 = timing_quantile(&mut timings, 0.99);
        let search_limit_fraction =
            search_limit_spokes as f32 / boundary_errors.len().max(1) as f32;
        eprintln!(
            "radial limbus RAW replay pairs={attempted_pairs} accepted-pairs={accepted_pairs} mean-spokes={spokes_mean:.2} boundary-p50/p90={boundary_p50:.2}/{boundary_p90:.2}px search-limit={search_limit_fraction:.3} cost-p50/p90={cost_p50:.3}/{cost_p90:.3} timing-us-p50/p99={timing_p50}/{timing_p99}"
        );
        assert!(attempted_pairs >= 8, "attempted pairs={attempted_pairs}");
        assert!(
            accepted_pairs * 2 >= attempted_pairs,
            "accepted pairs={accepted_pairs}/{attempted_pairs}"
        );
        assert!(boundary_p50 <= 5.0, "boundary p50={boundary_p50}");
        assert!(boundary_p90 <= 8.5, "boundary p90={boundary_p90}");
    }

    fn synthetic_shared_similarity_frame(
        width: usize,
        height: usize,
        scale: f64,
        translation: (f64, f64),
    ) -> Arc<Vec<u16>> {
        let center = (width as f64 * 0.5, height as f64 * 0.5);
        Arc::new(
            (0..height)
                .flat_map(|y| {
                    (0..width).map(move |x| {
                        let source_x = (x as f64 - center.0 - translation.0) / scale + center.0;
                        let source_y = (y as f64 - center.1 - translation.1) / scale + center.1;
                        let carrier = 430.0
                            + 105.0 * (0.109 * source_x + 0.071 * source_y).sin()
                            + 82.0 * (0.061 * source_x - 0.137 * source_y).cos()
                            + 47.0 * (0.181 * source_x).sin() * (0.157 * source_y).cos();
                        let cell = (((source_x / 19.0).floor() as i32
                            + (source_y / 17.0).floor() as i32)
                            & 1) as f64;
                        (carrier + 42.0 * cell).round().clamp(0.0, 1023.0) as u16
                    })
                })
                .collect(),
        )
    }

    fn raw_frame_from_shared(pixels: Arc<Vec<u16>>, width: usize, height: usize) -> RawFrame {
        RawFrame {
            sensor_x: 4_000,
            sensor_y: 3_000,
            width,
            height,
            pixels: Arc::unwrap_or_clone(pixels),
        }
    }

    #[test]
    fn horizontal_light_field_walk_tree_recovers_translation_and_expansion() {
        let width = 384;
        let height = 256;
        let previous = raw_frame_from_shared(
            synthetic_shared_similarity_frame(width, height, 1.0, (0.0, 0.0)),
            width,
            height,
        );
        let mut current = raw_frame_from_shared(
            synthetic_shared_similarity_frame(width, height, 1.028, (5.0, -2.0)),
            width,
            height,
        );
        // Correlation, rather than absolute brightness, is the walk score.
        // Exercise that contract with a large global exposure change.
        for pixel in &mut current.pixels {
            *pixel = (f32::from(*pixel) * 1.17 + 19.0).round().clamp(0.0, 1023.0) as u16;
        }
        let field = horizontal_light_field_tree(&previous, &current, SimilarityMotion::default());
        assert!(field.reliable, "{field:#?}");
        assert!(field.nodes.len() >= 3, "{field:#?}");
        assert!(field.leaf_nodes >= 2, "{field:#?}");
        assert!(
            (field.horizontal_translation_px - 5.0).abs() <= 2.0,
            "{field:#?}"
        );
        assert!(
            (field.horizontal_scale_delta - 0.028).abs() <= 0.018,
            "{field:#?}"
        );
        assert!(field.evaluations < 8_000, "{field:#?}");
    }

    #[test]
    fn horizontal_light_field_walk_tree_keeps_static_exposure_change_static() {
        let width = 384;
        let height = 256;
        let previous = raw_frame_from_shared(
            synthetic_shared_similarity_frame(width, height, 1.0, (0.0, 0.0)),
            width,
            height,
        );
        let mut current = previous.clone();
        for pixel in &mut current.pixels {
            *pixel = (f32::from(*pixel) * 0.74 + 83.0).round().clamp(0.0, 1023.0) as u16;
        }
        let field = horizontal_light_field_tree(&previous, &current, SimilarityMotion::default());
        assert!(field.reliable, "{field:#?}");
        assert!(field.horizontal_translation_px.abs() <= 1.0, "{field:#?}");
        assert!(field.horizontal_scale_delta.abs() <= 0.008, "{field:#?}");
    }

    #[test]
    fn horizontal_light_field_rescue_keeps_ids_through_a_fast_reversal() {
        use std::collections::BTreeSet;

        let width = 384;
        let height = 256;
        let mut tracker = FourMotionOctrees::default();
        let seed_for = |translation_x: f64| IrisEllipseSeed {
            center: (192.0 + translation_x, 128.0),
            major_radius: 72.0,
            minor_radius: 58.0,
            angle: 0.0,
        };
        let mut before_reversal = BTreeSet::<u64>::new();
        let mut reversed = MotionOctreeOverlay::default();
        for (index, translation_x) in [0.0, 4.0, 8.0, 12.0, -8.0].into_iter().enumerate() {
            let raw = synthetic_shared_similarity_frame(width, height, 1.0, (translation_x, 0.0));
            let overlay = tracker.observe_with_iris_seed_at(
                raw.as_slice(),
                width,
                height,
                4_000,
                3_000,
                1_000_000_000 + index as u64 * 100_000_000,
                None,
                true,
                Some(seed_for(translation_x)),
            );
            if index == 3 {
                before_reversal = tracker
                    .tracks
                    .iter()
                    .filter(|track| track.age == 0 && track.matched_streak >= 2)
                    .map(|track| track.id)
                    .collect();
            }
            if index == 4 {
                reversed = overlay;
            }
        }
        let after_reversal = tracker
            .tracks
            .iter()
            .filter(|track| track.age == 0)
            .map(|track| track.id)
            .collect::<BTreeSet<_>>();
        let retained = before_reversal.intersection(&after_reversal).count();
        assert!(before_reversal.len() >= 12, "before={before_reversal:?}");
        assert!(
            reversed.match_diagnostics.light_field_rescue_tracks >= 4,
            "diagnostics={:#?}",
            reversed.match_diagnostics
        );
        assert!(
            reversed.match_diagnostics.light_field_rescue_accepted >= 3,
            "diagnostics={:#?}",
            reversed.match_diagnostics
        );
        assert!(
            retained * 2 >= before_reversal.len(),
            "retained={retained}/{} diagnostics={:#?}",
            before_reversal.len(),
            reversed.match_diagnostics,
        );
    }

    #[test]
    fn dormant_nautilus_ids_reenter_after_a_blink_without_stale_overlay_points() {
        use std::collections::BTreeSet;

        let width = 384;
        let height = 256;
        let seed = IrisEllipseSeed {
            center: (192.0, 128.0),
            major_radius: 72.0,
            minor_radius: 58.0,
            angle: 0.0,
        };
        let mut tracker = FourMotionOctrees::default();
        let mut timestamp_ns = 1_000_000_000u64;
        for frame_index in 0..6usize {
            let mut raw = Arc::unwrap_or_clone(synthetic_shared_similarity_frame(
                width,
                height,
                1.0 + frame_index as f64 * 0.0015,
                (frame_index as f64 * 0.35, 0.0),
            ));
            // A moving compact glint gives the bounded appearance bank honest
            // pre-blink diversity while leaving the underlying identity and
            // large-scale light field unchanged.
            let glint_x = 150 + frame_index * 3;
            for y in 108..124 {
                for x in glint_x..(glint_x + 12) {
                    raw[y * width + x] = (raw[y * width + x] + 130).min(1023);
                }
            }
            let _ = tracker.observe_with_iris_seed_at(
                &raw,
                width,
                height,
                4_000,
                3_000,
                timestamp_ns,
                None,
                true,
                Some(seed),
            );
            timestamp_ns += 100_000_000;
        }
        let before_blink = tracker
            .tracks
            .iter()
            .filter(|track| {
                track.age == 0
                    && track.matched_streak >= 2
                    && tracker
                        .nautilus_banks
                        .get(&track.id)
                        .is_some_and(|bank| bank.observations.len() >= 2)
            })
            .map(|track| track.id)
            .collect::<BTreeSet<_>>();
        assert!(
            before_blink.len() >= 4,
            "banks={:?}",
            tracker.nautilus_banks
        );

        let closed = vec![512u16; width * height];
        for _ in 0..5 {
            let overlay = tracker.observe_with_iris_seed_at(
                &closed,
                width,
                height,
                4_000,
                3_000,
                timestamp_ns,
                None,
                true,
                None,
            );
            timestamp_ns += 100_000_000;
            assert!(
                overlay.trails.is_empty(),
                "a dormant coordinate must not masquerade as current: {overlay:#?}"
            );
        }
        let reopened = Arc::unwrap_or_clone(synthetic_shared_similarity_frame(
            width,
            height,
            1.0075,
            (1.75, 0.0),
        ));
        let _ = tracker.observe_with_iris_seed_at(
            &reopened,
            width,
            height,
            4_000,
            3_000,
            timestamp_ns,
            None,
            true,
            Some(seed),
        );
        timestamp_ns += 100_000_000;
        let recovered = tracker.observe_with_iris_seed_at(
            &reopened,
            width,
            height,
            4_000,
            3_000,
            timestamp_ns,
            None,
            true,
            Some(seed),
        );
        let after_blink = tracker
            .tracks
            .iter()
            .filter(|track| track.age == 0)
            .map(|track| track.id)
            .collect::<BTreeSet<_>>();
        let retained = before_blink.intersection(&after_blink).count();
        assert!(
            recovered.match_diagnostics.extended_gap_relocated >= 1,
            "retained={retained} diagnostics={:#?}",
            recovered.match_diagnostics,
        );
        assert!(
            retained >= 1,
            "before={before_blink:?} after={after_blink:?}"
        );
        assert!(recovered
            .trails
            .iter()
            .all(|trail| trail.points.len() == 1 || trail.matched_streak > 0));
        assert!(
            recovered.trails.len() <= MAX_FEATURES,
            "re-entry exceeded live graph budget: {}",
            recovered.trails.len(),
        );
    }

    #[test]
    #[ignore = "lossless consecutive RAW corpus replay; optionally set BUTTERCUP_LIGHT_FIELD_CORPUS_FRAMES"]
    fn horizontal_light_field_walk_tree_tracks_consecutive_raw_corpus() {
        use std::fs;
        use std::path::PathBuf;

        let root = std::env::var_os("BUTTERCUP_NAUTILUS_CORPUS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    "/mnt/bulk_data/osbot-drv-data/outputs/native-limbus-session-20260813/corpus",
                )
            });
        let maximum_frames = std::env::var("BUTTERCUP_LIGHT_FIELD_CORPUS_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(160);
        let mut archives = fs::read_dir(&root)
            .unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        archives.sort();
        let mut processed = 0usize;
        let mut attempts = 0usize;
        let mut reliable = 0usize;
        let mut joint_reference = 0usize;
        let mut translation_agreement = 0usize;
        let mut scale_agreement = 0usize;
        let mut timings = Vec::<u64>::new();
        for archive in archives {
            if processed >= maximum_frames {
                break;
            }
            let records = fs::read_to_string(archive.join("frames.jsonl"))
                .unwrap_or_else(|error| panic!("read {} index: {error}", archive.display()));
            let stream = fs::read(archive.join("subject-right.raw10"))
                .unwrap_or_else(|error| panic!("read {} RAW: {error}", archive.display()));
            let mut previous = None::<RawFrame>;
            let mut prior = SimilarityMotion::default();
            let mut independent = NativeGlobalSimilarityTracker::default();
            for record in records
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .filter(|record| {
                    record.get("label").and_then(serde_json::Value::as_str) == Some("subject-right")
                })
            {
                if processed >= maximum_frames {
                    break;
                }
                let offset = record["offset"].as_u64().expect("offset") as usize;
                let length = record["length"].as_u64().expect("length") as usize;
                let raw =
                    crate::raw10::unpack_raw10(&stream[offset..offset + length], 384, 256, 480);
                let sensor_x = record["sensor_x"].as_u64().expect("sensor_x") as u32;
                let sensor_y = record["sensor_y"].as_u64().expect("sensor_y") as u32;
                let reference =
                    independent.observe(Arc::new(raw.clone()), 384, 256, sensor_x, sensor_y);
                let current = RawFrame {
                    sensor_x,
                    sensor_y,
                    width: 384,
                    height: 256,
                    pixels: cfa_neutral_raw(&raw, 384, 256),
                };
                if let Some(previous) = previous.as_ref() {
                    attempts += 1;
                    let field = horizontal_light_field_tree(previous, &current, prior);
                    timings.push(field.elapsed_micros);
                    if field.reliable {
                        reliable += 1;
                        prior = SimilarityMotion {
                            translation: [field.horizontal_translation_px, 0.0],
                            scale_delta: field.horizontal_scale_delta,
                            residual: field.residual_px,
                            support: field.leaf_nodes,
                            ..SimilarityMotion::default()
                        };
                        if reference.reliable {
                            joint_reference += 1;
                            translation_agreement += usize::from(
                                (field.horizontal_translation_px - reference.motion.translation[0])
                                    .abs()
                                    <= 4.0,
                            );
                            scale_agreement += usize::from(
                                (field.horizontal_scale_delta - reference.motion.scale_delta).abs()
                                    <= 0.030,
                            );
                        }
                    } else {
                        prior = SimilarityMotion::default();
                    }
                }
                previous = Some(current);
                processed += 1;
            }
        }
        timings.sort_unstable();
        let percentile = |fraction: f32| {
            timings
                .get(((timings.len().saturating_sub(1)) as f32 * fraction).round() as usize)
                .copied()
                .unwrap_or(0)
        };
        eprintln!(
            "horizontal light-field RAW corpus frames={processed} attempts={attempts} reliable={reliable} joint-reference={joint_reference} translation-agreement={translation_agreement} scale-agreement={scale_agreement} timing-us-p50/p90/p99={}/{}/{}",
            percentile(0.50),
            percentile(0.90),
            percentile(0.99),
        );
        assert!(processed >= maximum_frames.min(80), "processed={processed}");
        assert!(
            attempts >= processed.saturating_sub(16),
            "attempts={attempts}"
        );
        assert!(reliable * 20 >= attempts, "reliable={reliable}/{attempts}");
        assert!(percentile(0.99) <= 20_000, "timings={timings:?}");
        if joint_reference >= 10 {
            assert!(
                translation_agreement * 2 >= joint_reference,
                "translation agreement={translation_agreement}/{joint_reference}"
            );
            assert!(
                scale_agreement * 2 >= joint_reference,
                "scale agreement={scale_agreement}/{joint_reference}"
            );
        }
    }

    #[test]
    fn integral_patch_moments_preserve_zero_mean_normalized_cost() {
        let width = 72;
        let height = 56;
        let previous = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: (0..width * height)
                .map(|index| {
                    let x = index % width;
                    let y = index / width;
                    (90 + (x * 37 + y * 71 + x * y * 3) % 850) as u16
                })
                .collect(),
        };
        let current = RawFrame {
            pixels: (0..width * height)
                .map(|index| {
                    let x = index % width;
                    let y = index / width;
                    previous.pixels[y * width + x.saturating_sub(2)]
                })
                .collect(),
            ..previous.clone()
        };
        let previous_moments = IntegralPatchMoments::new(&previous);
        let current_moments = IntegralPatchMoments::new(&current);
        for radius in [5, 7, 9] {
            let legacy =
                patch_cost_with_radius(&previous, &current, [34.0, 28.0], [36.0, 28.0], radius);
            let bounded = patch_cost_with_integral_moments(
                &previous,
                &current,
                &previous_moments,
                &current_moments,
                [34.0, 28.0],
                [36.0, 28.0],
                radius,
            );
            assert!((legacy - bounded).abs() < 1.0e-5, "{legacy} != {bounded}");
        }
    }

    fn fractional_texture_frame(
        width: usize,
        height: usize,
        shift: (f64, f64),
        aperture_only: bool,
    ) -> RawFrame {
        RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: (0..width * height)
                .map(|index| {
                    let x = index % width;
                    let y = index / width;
                    let source_x = x as f64 - shift.0;
                    let source_y = y as f64 - shift.1;
                    let mut value =
                        480.0 + 106.0 * (0.187 * source_x).sin() + 74.0 * (0.311 * source_x).cos();
                    if !aperture_only {
                        value += 83.0 * (0.233 * source_y).sin()
                            + 61.0 * (0.139 * source_x + 0.271 * source_y).cos()
                            + 43.0 * (0.347 * source_x - 0.193 * source_y).sin();
                    }
                    // Exercise ZNCC's exposure invariance at the same time as
                    // the fractional position. The shifted frame has both a
                    // gain and offset, but no spatial resampling performed by
                    // the matcher itself.
                    if shift != (0.0, 0.0) {
                        value = 1.08 * value + 17.0;
                    }
                    value.round().clamp(0.0, 1023.0) as u16
                })
                .collect(),
        }
    }

    #[test]
    fn native_zncc_peak_refines_below_the_integer_pixel_grid() {
        let width = 96;
        let height = 80;
        let shift = (0.34f32, -0.29f32);
        let previous = fractional_texture_frame(width, height, (0.0, 0.0), false);
        let current =
            fractional_texture_frame(width, height, (shift.0 as f64, shift.1 as f64), false);
        let previous_moments = IntegralPatchMoments::new(&previous);
        let current_moments = IntegralPatchMoments::new(&current);
        let previous_point = [48.27, 40.18];
        let mut evaluations = 0;
        let refined = refine_native_zncc_subpixel(
            &previous,
            &current,
            &previous_moments,
            &current_moments,
            previous_point,
            [48.0, 40.0],
            PREDICTED_PATCH_RADIUS,
            &mut evaluations,
        )
        .expect("two-dimensional native texture should have a conditioned subpixel peak");
        let expected = [previous_point[0] + shift.0, previous_point[1] + shift.1];
        assert_eq!(evaluations, 9);
        assert!(
            (refined.current[0] - expected[0]).abs() < 0.12,
            "refined={refined:?} expected={expected:?}"
        );
        assert!(
            (refined.current[1] - expected[1]).abs() < 0.12,
            "refined={refined:?} expected={expected:?}"
        );
        assert!(refined.correction[0].abs() > 0.10, "refined={refined:?}");
        assert!(refined.correction[1].abs() > 0.10, "refined={refined:?}");
    }

    #[test]
    fn subpixel_refinement_rejects_an_aperture_flat_tangent() {
        let width = 96;
        let height = 80;
        let previous = fractional_texture_frame(width, height, (0.0, 0.0), true);
        let current = fractional_texture_frame(width, height, (0.31, 0.0), true);
        let previous_moments = IntegralPatchMoments::new(&previous);
        let current_moments = IntegralPatchMoments::new(&current);
        let mut evaluations = 0;
        assert!(
            refine_native_zncc_subpixel(
                &previous,
                &current,
                &previous_moments,
                &current_moments,
                [48.0, 40.0],
                [48.0, 40.0],
                PREDICTED_PATCH_RADIUS,
                &mut evaluations,
            )
            .is_none(),
            "a one-dimensional edge must not claim a 2D fractional position"
        );
        assert_eq!(evaluations, 9);
    }

    #[test]
    fn native_global_similarity_recovers_broad_scale_without_copy_or_pyramid() {
        let width = 192;
        let height = 128;
        let mut tracker = NativeGlobalSimilarityTracker::default();
        let first = tracker.observe(
            synthetic_shared_similarity_frame(width, height, 1.0, (0.0, 0.0)),
            width,
            height,
            4_000,
            3_000,
        );
        assert_eq!(first.motion.support, 0);

        let second = tracker.observe(
            synthetic_shared_similarity_frame(width, height, 1.02, (1.0, -1.0)),
            width,
            height,
            4_000,
            3_000,
        );
        assert!(
            second.motion.support >= NATIVE_GLOBAL_MIN_SUPPORT,
            "{second:?}"
        );
        assert!(second.reliable, "{second:?}");
        assert_eq!(second.candidate_motion.support, second.motion.support);
        assert!(
            (second.motion.scale_delta - 0.02).abs() <= 0.010,
            "{second:?}"
        );
        assert_eq!(second.stable_frames, 1, "{second:?}");

        let third = tracker.observe(
            synthetic_shared_similarity_frame(width, height, 1.02 * 1.02, (2.0, -2.0)),
            width,
            height,
            4_000,
            3_000,
        );
        assert!(
            third.motion.support >= NATIVE_GLOBAL_MIN_SUPPORT,
            "{third:?}"
        );
        assert!(
            (third.motion.scale_delta - 0.02).abs() <= 0.010,
            "{third:?}"
        );
        assert!(third.stable_frames >= 2, "{third:?}");
        assert!(third.reliable, "{third:?}");
        assert!(third.occupied_quadrants >= 3, "{third:?}");
    }

    #[test]
    fn similarity_fit_recovers_translation_and_rotation() {
        let center = [50.0, 40.0];
        let truth = SimilarityMotion {
            translation: [3.0, -2.0],
            rotation: 0.04,
            scale_delta: 0.015,
            ..SimilarityMotion::default()
        };
        let points = [
            [20.0, 20.0],
            [80.0, 20.0],
            [20.0, 60.0],
            [80.0, 60.0],
            [50.0, 25.0],
        ];
        let matches = points
            .iter()
            .enumerate()
            .map(|(index, point)| Match {
                track_index: index,
                previous: *point,
                current: truth.predict(*point, center),
                score: 1.0,
                object: 2,
                z: 0.0,
                assignment_margin: 0.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            })
            .collect::<Vec<_>>();
        let fit = fit_similarity(&matches, 2, center);
        assert!((fit.translation[0] - truth.translation[0]).abs() < 1.0e-4);
        assert!((fit.translation[1] - truth.translation[1]).abs() < 1.0e-4);
        assert!((fit.rotation - truth.rotation).abs() < 1.0e-4);
        assert!((fit.scale_delta - truth.scale_delta).abs() < 1.0e-4);
    }

    #[test]
    fn shared_native_patch_parabola_recovers_subpixel_minimum() {
        // Samples of (x - 0.30)^2 at x=-1,0,+1 have their vertex at +0.30.
        let negative = (-1.0f32 - 0.30).powi(2);
        let center = (0.0f32 - 0.30).powi(2);
        let positive = (1.0f32 - 0.30).powi(2);
        let offset = shared_native_parabolic_patch_offset(negative, center, positive);
        assert!((offset - 0.30).abs() < 1.0e-6, "{offset}");
        assert_eq!(shared_native_parabolic_patch_offset(1.0, 1.0, 1.0), 0.0);
    }

    #[test]
    fn shared_native_similarity_prefers_broad_rigid_motion_over_a_larger_lid_strip() {
        let center = [100.0, 70.0];
        let truth = SimilarityMotion {
            translation: [1.5, -0.75],
            rotation: 0.006,
            scale_delta: 0.018,
            ..SimilarityMotion::default()
        };
        let broad = [
            [20.0, 20.0],
            [180.0, 20.0],
            [20.0, 120.0],
            [180.0, 120.0],
            [50.0, 40.0],
            [150.0, 40.0],
            [50.0, 100.0],
            [150.0, 100.0],
        ];
        let mut matches = broad
            .iter()
            .enumerate()
            .map(|(index, point)| Match {
                track_index: index,
                previous: *point,
                current: truth.predict(*point, center),
                score: 1.0,
                object: GENERAL_LAYER,
                z: 0.0,
                assignment_margin: 1.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            })
            .collect::<Vec<_>>();
        // More matches occupy a compact moving-lid strip. Ordinary global
        // least squares is pulled between these incompatible motions; the
        // scale authority must instead choose the four-quadrant cohort.
        for index in 0..10 {
            let point = [64.0 + index as f32 * 8.0, 66.0 + (index % 2) as f32 * 3.0];
            matches.push(Match {
                track_index: broad.len() + index,
                previous: point,
                current: [point[0] + 7.0, point[1] + 5.0],
                score: 1.0,
                object: GENERAL_LAYER,
                z: 0.0,
                assignment_margin: 1.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            });
        }

        let (fit, inliers) = shared_native_robust_global_similarity(&matches, center);
        assert!(
            inliers.len() >= broad.len(),
            "fit={fit:?} inliers={inliers:?}"
        );
        assert!(
            (fit.translation[0] - truth.translation[0]).abs() < 0.05,
            "{fit:?}"
        );
        assert!(
            (fit.translation[1] - truth.translation[1]).abs() < 0.05,
            "{fit:?}"
        );
        assert!((fit.rotation - truth.rotation).abs() < 0.001, "{fit:?}");
        assert!(
            (fit.scale_delta - truth.scale_delta).abs() < 0.001,
            "{fit:?}"
        );
    }

    #[test]
    fn bounded_native_iris_pairs_recover_fine_image_scale() {
        let center = [4_200.0, 2_100.0];
        let truth = SimilarityMotion {
            translation: [2.5, -1.5],
            rotation: 0.018,
            scale_delta: 0.035,
            ..SimilarityMotion::default()
        };
        let pairs = [
            [-62.0, -8.0],
            [-38.0, 34.0],
            [0.0, 48.0],
            [42.0, 30.0],
            [64.0, -5.0],
            [35.0, -40.0],
            [-20.0, -45.0],
            [-55.0, -28.0],
        ]
        .map(|offset| {
            let previous = [center[0] + offset[0], center[1] + offset[1]];
            (previous, truth.predict(previous, center))
        });

        let measured = bounded_iris_similarity_motion(&pairs, center);

        assert_eq!(measured.support, pairs.len());
        assert!(measured.residual < 1.0e-3, "measured={measured:?}");
        assert!((measured.translation[0] - truth.translation[0]).abs() < 1.0e-3);
        assert!((measured.translation[1] - truth.translation[1]).abs() < 1.0e-3);
        assert!((measured.rotation - truth.rotation).abs() < 1.0e-3);
        assert!((measured.scale_delta - truth.scale_delta).abs() < 1.0e-3);
    }

    #[test]
    fn reanchored_similarity_is_the_same_sensor_transform() {
        let old_center = [4_210.0, 2_080.0];
        let new_center = [4_096.0, 2_160.0];
        let motion = SimilarityMotion {
            translation: [3.5, -2.25],
            rotation: 0.037,
            scale_delta: -0.021,
            residual: 0.4,
            support: 9,
        };
        let reanchored = reanchor_similarity_motion(motion, old_center, new_center);
        for point in [[4_000.0, 2_000.0], [4_210.0, 2_080.0], [4_340.0, 2_260.0]] {
            let expected = motion.predict(point, old_center);
            let actual = reanchored.predict(point, new_center);
            assert!(
                (actual[0] - expected[0]).abs() < 1.0e-3,
                "{actual:?} {expected:?}"
            );
            assert!(
                (actual[1] - expected[1]).abs() < 1.0e-3,
                "{actual:?} {expected:?}"
            );
        }
    }

    fn synthetic_bounded_coupling_overlay(
        pupil_translation: [f32; 2],
        stable_frames: u16,
    ) -> MotionOctreeOverlay {
        let mut overlay = MotionOctreeOverlay::default();
        overlay.motions[PUPIL_LAYER] = SimilarityMotion {
            translation: pupil_translation,
            residual: 0.40,
            support: 6,
            ..SimilarityMotion::default()
        };
        overlay.layers[PUPIL_LAYER] = MotionLayerStatus {
            centroid: [160.0, 120.0],
            coherence: 0.86,
            trajectory_error: 0.40,
            signature_samples: usize::from(stable_frames).saturating_add(1),
            persistent_tracks: 6,
            stable_frames,
            ..MotionLayerStatus::default()
        };
        overlay.semantic_iris = Some(IrisEllipseSeed {
            center: (160.0, 120.0),
            major_radius: 72.0,
            minor_radius: 64.0,
            angle: 0.12,
        });
        overlay
    }

    fn synthetic_global_coupling_evidence(
        reliable: bool,
        stable_frames: u16,
    ) -> NativeGlobalSimilarityEvidence {
        let motion = SimilarityMotion {
            translation: [2.0, -1.0],
            residual: 0.35,
            support: 12,
            ..SimilarityMotion::default()
        };
        NativeGlobalSimilarityEvidence {
            motion: reliable.then_some(motion).unwrap_or_default(),
            candidate_motion: motion,
            candidate_matches: 14,
            reliable,
            stable_frames,
            spatial_span: [260.0, 180.0],
            occupied_quadrants: 4,
            motion_center_sensor: [4_160.0, 3_120.0],
        }
    }

    #[test]
    fn bounded_global_coupling_rejects_rigid_head_motion_as_a_saccade() {
        let mut tracker = BoundedIrisCannyTracker::default();
        let mut last = MotionOctreeOverlay::default();
        for index in 0..7u64 {
            let mut overlay = synthetic_bounded_coupling_overlay([2.0, -1.0], index as u16 + 1);
            tracker.fuse_global_similarity_at(
                &mut overlay,
                synthetic_global_coupling_evidence(true, index as u16 + 1),
                1_000_000_000 + index * 40_000_000,
                320,
                240,
                4_000,
                3_000,
            );
            last = overlay;
        }
        let relative = last.coupled_motion.green_relative_to_cyan;
        assert!(relative.samples >= 4, "{relative:?}");
        assert!(relative.speed_px_s < 0.5, "{relative:?}");
        assert!(
            last.coupled_motion.saccade_likelihood < 0.10,
            "{:?}",
            last.coupled_motion
        );
    }

    #[test]
    fn bounded_global_coupling_authorizes_only_observed_relative_motion() {
        let mut tracker = BoundedIrisCannyTracker::default();
        let mut last = MotionOctreeOverlay::default();
        for index in 0..7u64 {
            let mut overlay = synthetic_bounded_coupling_overlay([6.0, -1.0], index as u16 + 1);
            tracker.fuse_global_similarity_at(
                &mut overlay,
                synthetic_global_coupling_evidence(true, index as u16 + 1),
                1_000_000_000 + index * 40_000_000,
                320,
                240,
                4_000,
                3_000,
            );
            last = overlay;
        }
        let relative = last.coupled_motion.green_relative_to_cyan;
        assert!(relative.samples >= 4, "{relative:?}");
        assert!(relative.confidence >= 0.12, "{relative:?}");
        assert!(relative.speed_px_s >= 70.0, "{relative:?}");
        assert!(
            last.coupled_motion.saccade_likelihood >= 0.68,
            "{:?}",
            last.coupled_motion
        );

        let mut dropout = synthetic_bounded_coupling_overlay([6.0, -1.0], 8);
        tracker.fuse_global_similarity_at(
            &mut dropout,
            synthetic_global_coupling_evidence(false, 0),
            1_280_000_000,
            320,
            240,
            4_000,
            3_000,
        );
        assert_eq!(dropout.coupled_motion.green_relative_to_cyan.samples, 0);
        assert_eq!(dropout.coupled_motion.saccade_likelihood, 0.0);
    }

    fn synthetic_native_similarity_texture(
        width: usize,
        height: usize,
        center: (f64, f64),
        absolute_scale: f64,
        translation: (f64, f64),
    ) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    // Evaluate one continuous, non-periodically symmetric RAW
                    // texture through the inverse camera similarity. This is
                    // a full-resolution resampling, not a resized test image.
                    let source_x =
                        center.0 + (x as f64 - center.0 - translation.0) / absolute_scale;
                    let source_y =
                        center.1 + (y as f64 - center.1 - translation.1) / absolute_scale;
                    let radial = (source_x - center.0).hypot(source_y - center.1);
                    let value = 510.0
                        + 105.0 * (0.113 * source_x + 0.071 * source_y).sin()
                        + 82.0 * (0.037 * source_x - 0.173 * source_y).cos()
                        + 63.0 * (0.241 * source_x + 0.199 * source_y).sin()
                        + 47.0 * (0.317 * source_x - 0.049 * source_y).cos()
                        + 38.0 * (0.29 * radial + 0.013 * source_x).sin();
                    value.round().clamp(0.0, 1023.0) as u16
                })
            })
            .collect()
    }

    #[test]
    fn bounded_native_raw_matcher_recovers_scale_without_a_pyramid() {
        let width = 192;
        let height = 144;
        let center = (96.0, 72.0);
        let mut tracker = BoundedIrisCannyTracker::default();
        let observations = [
            (1.0, (0.0, 0.0)),
            (1.025, (2.0, -1.0)),
            (1.050625, (4.0, -2.0)),
        ];
        let mut overlays = Vec::new();
        for (scale, translation) in observations {
            let raw =
                synthetic_native_similarity_texture(width, height, center, scale, translation);
            overlays.push(tracker.observe(
                &raw,
                width,
                height,
                4_000,
                2_000,
                Some(IrisEllipseSeed {
                    center: (center.0 + translation.0, center.1 + translation.1),
                    major_radius: 54.0 * scale,
                    minor_radius: 42.0 * scale,
                    angle: 0.0,
                }),
            ));
        }

        assert!(overlays[0].provisional_features.len() >= 12);
        let motion = overlays[2].motions[PUPIL_LAYER];
        assert!(motion.support >= 6, "motion={motion:?}");
        assert!(motion.residual <= 2.0, "motion={motion:?}");
        assert!(
            (motion.scale_delta - 0.025).abs() <= 0.012,
            "motion={motion:?}",
        );
        assert!(overlays[2].layers[PUPIL_LAYER].stable_frames >= 2);
    }

    #[test]
    fn temporal_patch_search_does_not_extrapolate_motion_across_a_capture_gap() {
        let width = 96usize;
        let height = 72usize;
        let pixels =
            synthetic_native_similarity_texture(width, height, (48.0, 36.0), 1.0, (0.0, 0.0));
        let make_tracker = || {
            let mut tracker = FourMotionOctrees::default();
            tracker.previous = Some(RawFrame {
                sensor_x: 0,
                sensor_y: 0,
                width,
                height,
                pixels: pixels.clone(),
            });
            tracker.previous_timestamp_ns = Some(1_000_000_000);
            tracker.tracks.push(FeatureTrack {
                id: 1,
                points: VecDeque::from([[48.0, 36.0, 0.0]]),
                object: GENERAL_LAYER,
                age: 0,
                last_seen_timestamp_ns: 1_000_000_000,
                score: 1.0,
                motion_ema: [1.0, 0.0],
                motion_variance: 0.0,
                matched_streak: 4,
                layer_evidence: true,
                normal_flow_evidence: false,
                specularity: 0.0,
                assignment_confidence: 0.9,
                edge_normal: [0.0; 2],
                residual_history: VecDeque::from([[0.0; 2]; MIN_MOTION_SIGNATURE]),
                focus_bins: Vec::new(),
                focus_peak: None,
            });
            tracker.motions[GENERAL_LAYER] = SimilarityMotion {
                translation: [1.0, 0.0],
                residual: 0.1,
                support: 8,
                ..SimilarityMotion::default()
            };
            tracker.layers[GENERAL_LAYER] = MotionLayerStatus {
                coherence: 0.9,
                persistent_tracks: 8,
                stable_frames: MIN_LAYER_STABLE_FRAMES,
                ..MotionLayerStatus::default()
            };
            tracker
        };

        let contiguous = make_tracker()
            .observe_with_iris_seed_at(
                &pixels,
                width,
                height,
                0,
                0,
                1_100_000_000,
                None,
                false,
                None,
            )
            .match_diagnostics
            .coarse_patch_evaluations;
        let after_gap = make_tracker()
            .observe_with_iris_seed_at(
                &pixels,
                width,
                height,
                0,
                0,
                1_500_000_001,
                None,
                false,
                None,
            )
            .match_diagnostics
            .coarse_patch_evaluations;

        assert!(
            contiguous > 0,
            "bounded predictor did not exercise its search"
        );
        assert!(
            after_gap > contiguous,
            "a stale motion vector still bounded the gap search: contiguous={contiguous} gap={after_gap}",
        );
    }

    #[test]
    fn upper_canny_shadow_motion_is_downweighted_on_the_native_raw_grid() {
        let width = 128usize;
        let height = 96usize;
        let previous_pixels = (0..height)
            .flat_map(|y| {
                (0..width)
                    .map(move |x| (140 + (x * 73 + y * 151 + x * y * 17 + x * x * 3) % 760) as u16)
            })
            .collect::<Vec<_>>();
        let mut current_pixels = vec![0u16; width * height];
        for y in 0..height {
            for x in 0..width {
                current_pixels[y * width + x] = previous_pixels[y * width + x.saturating_sub(3)];
            }
        }
        let shadow = (67usize, 28usize);
        for dy in -3isize..=3 {
            for dx in -3isize..=3 {
                let x = shadow.0.saturating_add_signed(dx);
                let y = shadow.1.saturating_add_signed(dy);
                current_pixels[y * width + x] = previous_pixels[y * width + x];
            }
        }
        let previous = RawFrame {
            sensor_x: 4_000,
            sensor_y: 2_000,
            width,
            height,
            pixels: previous_pixels,
        };
        let current = RawFrame {
            sensor_x: 4_000,
            sensor_y: 2_000,
            width,
            height,
            pixels: current_pixels,
        };
        let mut edges = vec![
            EdgeEvidence {
                x: shadow.0 as f32,
                y: shadow.1 as f32,
                gradient_x: 0.0,
                gradient_y: -1.0,
                strength: 2.0,
                iris_motion_consistency: 1.0,
                ..EdgeEvidence::default()
            },
            EdgeEvidence {
                x: 45.0,
                y: 36.0,
                gradient_x: -0.8,
                gradient_y: -0.6,
                strength: 2.0,
                iris_motion_consistency: 1.0,
                ..EdgeEvidence::default()
            },
        ];
        let seed = IrisEllipseSeed {
            center: (67.0, 52.0),
            major_radius: 36.0,
            minor_radius: 28.0,
            angle: 0.0,
        };

        let downweighted = condition_upper_edges_by_iris_motion(
            &mut edges,
            &previous,
            &current,
            seed,
            SimilarityMotion {
                translation: [3.0, 0.0],
                residual: 0.25,
                support: 8,
                ..SimilarityMotion::default()
            },
            [4_067.0, 2_052.0],
        );

        assert_eq!(downweighted, 1, "conditioned edges={edges:?}");
        assert!(
            edges[0].iris_motion_consistency < 0.45,
            "stationary upper shadow should lose its iris vote: {edges:?}"
        );
        assert_eq!(edges[1].iris_motion_consistency, 1.0);

        for edge in &mut edges {
            edge.iris_motion_consistency = 1.0;
        }
        assert_eq!(
            condition_upper_edges_by_iris_motion(
                &mut edges,
                &previous,
                &current,
                seed,
                SimilarityMotion {
                    translation: [0.2, 0.0],
                    residual: 0.25,
                    support: 8,
                    ..SimilarityMotion::default()
                },
                [4_067.0, 2_052.0],
            ),
            0,
            "uninformative motion must remain unclassified"
        );
        assert!(edges.iter().all(|edge| edge.iris_motion_consistency == 1.0));
    }

    #[test]
    fn analog_motion_consistency_reduces_an_edge_ellipse_vote() {
        let center = (80.0f64, 70.0f64);
        let radius = 42.0f64;
        let edges = (0..96)
            .map(|sample| {
                let phase = sample as f64 * std::f64::consts::TAU / 96.0;
                EdgeEvidence {
                    x: (center.0 + radius * phase.cos()) as f32,
                    y: (center.1 + radius * phase.sin()) as f32,
                    gradient_x: phase.cos() as f32,
                    gradient_y: phase.sin() as f32,
                    strength: 1.0,
                    iris_motion_consistency: 1.0,
                    ..EdgeEvidence::default()
                }
            })
            .collect::<Vec<_>>();
        let ellipse = EdgeEllipse {
            center,
            major: radius,
            minor: radius,
            angle: 0.0,
        };
        let baseline = score_edge_ellipse(ellipse, &edges, (center, radius), false);
        let mut shadowed = edges;
        for edge in &mut shadowed {
            if edge.y < center.1 as f32 {
                edge.iris_motion_consistency = 0.12;
            }
        }
        let conditioned = score_edge_ellipse(ellipse, &shadowed, (center, radius), false);
        assert!(conditioned.confidence < baseline.confidence);
        assert!(conditioned.objective < baseline.objective);
    }

    #[test]
    fn limbus_normal_flow_ignores_tangential_aperture_drift() {
        let center = [50.0, 50.0];
        let translation = [2.0, -1.0];
        let points = [
            ([45.0, 45.0], [1.0, 0.0], 0.0),
            ([50.0, 45.0], [0.0, 1.0], 0.0),
            ([45.0, 50.0], [1.0, 0.0], 0.0),
            ([10.0, 50.0], [-1.0, 0.0], 12.0),
            ([90.0, 50.0], [1.0, 0.0], -12.0),
            ([50.0, 10.0], [0.0, -1.0], 12.0),
            ([50.0, 90.0], [0.0, 1.0], -12.0),
        ];
        let tracks = points
            .iter()
            .enumerate()
            .map(|(index, (point, normal, _))| FeatureTrack {
                id: index as u64,
                points: VecDeque::from([[point[0], point[1], 0.0]]),
                object: PUPIL_LAYER,
                age: 0,
                last_seen_timestamp_ns: 0,
                score: 1.0,
                motion_ema: translation,
                motion_variance: 0.0,
                matched_streak: 4,
                layer_evidence: true,
                normal_flow_evidence: index >= 3,
                specularity: 0.0,
                assignment_confidence: 1.0,
                edge_normal: *normal,
                residual_history: VecDeque::from([translation; MIN_MOTION_SIGNATURE]),
                focus_bins: Vec::new(),
                focus_peak: None,
            })
            .collect::<Vec<_>>();
        let matches = points
            .iter()
            .enumerate()
            .map(|(index, (point, normal, tangential_drift))| {
                let tangent = [-normal[1], normal[0]];
                Match {
                    track_index: index,
                    previous: *point,
                    current: [
                        point[0] + translation[0] + tangent[0] * tangential_drift,
                        point[1] + translation[1] + tangent[1] * tangential_drift,
                    ],
                    score: 1.0,
                    object: PUPIL_LAYER,
                    z: 0.0,
                    assignment_margin: 1.0,
                    layer_evidence: true,
                    normal_flow_evidence: index >= 3,
                    specularity: 0.0,
                }
            })
            .collect::<Vec<_>>();
        let fitted = fit_similarity_with_normal_constraints(
            &[0, 1, 2],
            &[3, 4, 5, 6],
            &[],
            &matches,
            &tracks,
            center,
        );
        assert!((fitted.translation[0] - translation[0]).abs() < 0.05);
        assert!((fitted.translation[1] - translation[1]).abs() < 0.05);
        assert!(fitted.rotation.abs() < 0.01);
        assert!(fitted.scale_delta.abs() < 0.01);
        assert!(fitted.residual < 0.05);
    }

    #[test]
    fn match_destinations_are_one_to_one() {
        let mut matches = vec![
            Match {
                track_index: 0,
                previous: [8.0, 10.0],
                current: [10.0, 10.0],
                score: 0.82,
                object: 0,
                z: 0.0,
                assignment_margin: 0.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            },
            Match {
                track_index: 1,
                previous: [9.0, 10.0],
                current: [10.7, 10.2],
                score: 0.80,
                object: 0,
                z: 0.0,
                assignment_margin: 0.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            },
            Match {
                track_index: 2,
                previous: [19.0, 20.0],
                current: [20.0, 20.0],
                score: 0.74,
                object: 0,
                z: 0.0,
                assignment_margin: 0.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            },
        ];
        let rejected = enforce_unique_match_destinations(&mut matches, &[0.0, 0.0, 0.0]);
        assert_eq!(rejected, 1);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].track_index, 0);
        assert_eq!(matches[1].track_index, 2);
    }

    #[test]
    fn signature_clusters_leave_sparse_outliers_provisional() {
        let mut candidates = (0..6)
            .map(|index| SignatureCandidate {
                match_index: index,
                samples: (0..MIN_MOTION_SIGNATURE)
                    .map(|sample| [sample as f32 * 0.03 + index as f32 * 0.02, 0.0])
                    .collect(),
            })
            .collect::<Vec<_>>();
        for index in 0..3 {
            candidates.push(SignatureCandidate {
                match_index: 6 + index,
                samples: vec![[8.0 * (index + 1) as f32, -6.0]; MIN_MOTION_SIGNATURE],
            });
        }
        let clusters = cluster_signatures(&candidates);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].members.len(), 6);
        assert!(clusters[0].within_error < 0.2);
    }

    #[test]
    fn relation_graph_separates_iris_rotation_from_matching_lash_velocity() {
        let center = [50.0f32, 50.0f32];
        let global = SimilarityMotion {
            translation: [1.0, -0.2],
            support: 8,
            ..SimilarityMotion::default()
        };
        let iris = [[35.0, 50.0], [50.0, 35.0], [65.0, 50.0], [50.0, 65.0]];
        let lashes = [[30.0, 25.0], [43.0, 25.0], [57.0, 25.0], [70.0, 25.0]];
        let mut tracks = Vec::new();
        let mut matches = Vec::new();
        for (index, previous) in iris.into_iter().chain(lashes).enumerate() {
            let local = if index < 4 {
                let x = previous[0] - center[0];
                let y = previous[1] - center[1];
                [-0.04 * y, 0.04 * x]
            } else {
                // This equals the iris top point's instantaneous local
                // velocity, defeating a velocity-only cluster at that point.
                [0.60, 0.0]
            };
            tracks.push(FeatureTrack {
                id: index as u64,
                points: VecDeque::from([[previous[0], previous[1], 0.0]]),
                object: GENERAL_LAYER,
                age: 0,
                last_seen_timestamp_ns: 0,
                score: 1.0,
                motion_ema: [
                    global.translation[0] + local[0],
                    global.translation[1] + local[1],
                ],
                motion_variance: 0.0,
                matched_streak: 4,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.2,
                assignment_confidence: 0.8,
                edge_normal: [1.0, 0.0],
                residual_history: VecDeque::from([local; MIN_MOTION_SIGNATURE]),
                focus_bins: Vec::new(),
                focus_peak: None,
            });
            matches.push(Match {
                track_index: index,
                previous,
                current: [
                    previous[0] + global.translation[0] + local[0],
                    previous[1] + global.translation[1] + local[1],
                ],
                score: 1.0,
                object: GENERAL_LAYER,
                z: 0.0,
                assignment_margin: 0.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.2,
            });
        }
        let mut graph = PersistentMotionRelationGraph::default();
        let first_relations = graph.observe(&matches, &tracks, center, global);
        assert!(
            first_relations.components.len() >= 2,
            "{:#?}",
            first_relations.components
        );
        assert_eq!(first_relations.coherent_edge_count(), 0);
        // On a later exposure the iris reverses while the lash carrier keeps
        // translating. The persistent pair graph must resolve the one point
        // whose velocity was exactly ambiguous above.
        let second_global = SimilarityMotion {
            translation: [0.8, 0.1],
            support: 8,
            ..SimilarityMotion::default()
        };
        for (index, item) in matches.iter_mut().enumerate() {
            item.previous = item.current;
            let local = if index < 4 {
                let x = item.previous[0] - center[0];
                let y = item.previous[1] - center[1];
                [0.04 * y, -0.04 * x]
            } else {
                [0.60, 0.0]
            };
            item.current = [
                item.previous[0] + second_global.translation[0] + local[0],
                item.previous[1] + second_global.translation[1] + local[1],
            ];
        }
        let mut relations = graph.observe(&matches, &tracks, center, second_global);
        assert!(relations.coherent_edge_count() >= 2);
        assert_eq!(relations.components.len(), 2, "{:#?}", relations.components);
        let iris_component = relations
            .components
            .iter()
            .position(|component| component.members.iter().all(|index| *index < 4))
            .expect("independent iris tensor component");
        let lash_component = relations
            .components
            .iter()
            .position(|component| component.members.iter().all(|index| *index >= 4))
            .unwrap_or_else(|| {
                panic!(
                    "independent lash tensor component: {:#?}",
                    relations.components
                )
            });
        assert_ne!(iris_component, lash_component);
        assert_eq!(relations.components[iris_component].members.len(), 4);
        assert_eq!(relations.components[lash_component].members.len(), 4);
        let recovered = &relations.components[iris_component];
        assert!(recovered.origin_valid);
        assert!((recovered.shared_origin[0] - center[0]).abs() < 0.01);
        assert!((recovered.shared_origin[1] - center[1]).abs() < 0.01);
        assert!(!relations.components[lash_component].origin_valid);

        let eye_region = EyeMotionRegion {
            center,
            major: 30.0,
            minor: 30.0,
            angle: 0.0,
        };
        let candidates = (0..matches.len()).collect::<Vec<_>>();

        // Without a pivot, a component that leaks into external material is
        // not allowed to seed iris identity even when its internal transform
        // is persistent and its in-eye subset has sufficient support.
        let mut unconditioned_impure = relations.clone();
        let external_member = relations.components[lash_component].members[0];
        unconditioned_impure.components[iris_component]
            .members
            .push(external_member);
        unconditioned_impure.components[iris_component]
            .track_ids
            .push(tracks[matches[external_member].track_index].id);
        unconditioned_impure.components[iris_component]
            .track_ids
            .sort_unstable();
        unconditioned_impure.components[iris_component].origin_valid = false;
        let iris_only_candidates = (0..4).collect::<Vec<_>>();
        let rejected_impure = relation_graph_iris_core(
            &iris_only_candidates,
            &matches,
            &tracks,
            second_global,
            center,
            eye_region,
            &mut unconditioned_impure,
            Some(&PersistentRelationIrisIdentity::default()),
            MIN_LAYER_SUPPORT,
        );
        assert!(rejected_impure.is_empty());
        assert_eq!(unconditioned_impure.observed_iris_component, None);
        assert!(
            unconditioned_impure
                .iris_candidate_diagnostics
                .rejected_untrusted_origin_seed
                > 0
        );

        // When differential motion is too close to translation, the fixed
        // point is mathematically unobservable.  Preserve the exact material
        // cohort as provisional identity evidence without inventing a pivot.
        let mut unconditioned_relations = relations.clone();
        unconditioned_relations.components[iris_component].origin_valid = false;
        let mut unconditioned_identity = PersistentRelationIrisIdentity::default();
        let unconditioned_provisional = relation_graph_iris_core(
            &candidates,
            &matches,
            &tracks,
            second_global,
            center,
            eye_region,
            &mut unconditioned_relations,
            Some(&unconditioned_identity),
            MIN_LAYER_SUPPORT,
        );
        assert!(unconditioned_provisional.is_empty());
        assert_eq!(
            unconditioned_relations.observed_iris_component,
            Some(iris_component)
        );
        assert_eq!(unconditioned_relations.selected_iris_component, None);
        assert!(
            unconditioned_relations
                .iris_candidate_diagnostics
                .invalid_initial_origins
                > 0
        );
        assert_eq!(unconditioned_relations.initial_origin_rejections, 0);
        unconditioned_identity.observe(
            unconditioned_relations.observed_iris(),
            RelationIrisIdentityContinuity {
                track_overlap: unconditioned_relations.selected_identity_overlap,
                origin_consistent: unconditioned_relations.selected_origin_consistent,
                ..RelationIrisIdentityContinuity::default()
            },
            unconditioned_relations.observed_motion_evidence,
        );
        assert!(unconditioned_identity.active());
        assert!(!unconditioned_identity.origin_valid);

        let still_provisional = relation_graph_iris_core(
            &candidates,
            &matches,
            &tracks,
            second_global,
            center,
            eye_region,
            &mut unconditioned_relations,
            Some(&unconditioned_identity),
            MIN_LAYER_SUPPORT,
        );
        assert!(still_provisional.is_empty());
        assert_eq!(unconditioned_relations.selected_iris_component, None);
        assert!(
            unconditioned_relations
                .iris_candidate_diagnostics
                .withheld_untrusted_origin
                > 0
        );

        let mut identity = PersistentRelationIrisIdentity::default();
        let provisional = relation_graph_iris_core(
            &candidates,
            &matches,
            &tracks,
            second_global,
            center,
            eye_region,
            &mut relations,
            Some(&identity),
            MIN_LAYER_SUPPORT,
        );
        assert!(provisional.is_empty());
        assert_eq!(relations.observed_iris_component, Some(iris_component));
        assert_eq!(relations.selected_iris_component, None);
        identity.observe(
            relations.observed_iris(),
            RelationIrisIdentityContinuity {
                track_overlap: relations.selected_identity_overlap,
                origin_consistent: relations.selected_origin_consistent,
                ..RelationIrisIdentityContinuity::default()
            },
            relations.observed_motion_evidence,
        );
        assert!(identity.active());
        assert!(!identity.confirmed());

        let selected = relation_graph_iris_core(
            &candidates,
            &matches,
            &tracks,
            second_global,
            center,
            eye_region,
            &mut relations,
            Some(&identity),
            MIN_LAYER_SUPPORT,
        );
        assert_eq!(selected.len(), 4);
        assert!(selected.iter().all(|index| *index < 4));
        identity.observe(
            relations.observed_iris(),
            RelationIrisIdentityContinuity {
                track_overlap: relations.selected_identity_overlap,
                origin_consistent: relations.selected_origin_consistent,
                ..RelationIrisIdentityContinuity::default()
            },
            relations.observed_motion_evidence,
        );
        assert!(identity.confirmed());

        let mut motions = [SimilarityMotion::default(); OBJECTS];
        let mut layers = [MotionLayerStatus::default(); OBJECTS];
        let mut signatures: [LayerMotionSignature; OBJECTS] = Default::default();
        let mut axis = [0.0; 2];
        assert!(cluster_relation_motion_layers(
            &mut matches,
            &tracks,
            &mut relations,
            Some(eye_region),
            &mut motions,
            &mut layers,
            &mut signatures,
            &mut axis,
            center,
            second_global,
            Some(&identity),
        ));
        assert!(matches[..4]
            .iter()
            .all(|item| item.object == PUPIL_LAYER && item.layer_evidence));
        assert!(matches[4..]
            .iter()
            .all(|item| item.object == GENERAL_LAYER && item.layer_evidence));
    }

    #[test]
    fn relation_iris_identity_rejects_a_consecutive_material_switch() {
        let component =
            |track_ids: Vec<u64>, centroid: [f32; 2], origin: [f32; 2]| MotionRelationComponent {
                members: (0..track_ids.len()).collect(),
                track_ids,
                centroid,
                coherence: 0.8,
                shared_origin: origin,
                origin_spread: 1.0,
                origin_valid: true,
                persistent_edges: 3,
                persistent_nodes: 4,
            };
        assert!(
            (relation_sorted_track_id_jaccard(&[1, 2, 3, 4], &[3, 4, 5, 6]) - 1.0 / 3.0).abs()
                < 1.0e-6
        );

        let eye_region = EyeMotionRegion {
            center: [50.0, 50.0],
            major: 30.0,
            minor: 24.0,
            angle: 0.0,
        };
        let first = component(vec![1, 2, 3, 4], [50.0, 50.0], [50.0, 50.0]);
        let mut identity = PersistentRelationIrisIdentity::default();
        identity.observe(Some(&first), RelationIrisIdentityContinuity::default(), 1.0);
        assert!(identity.active());

        // Spatial proximity alone cannot let a disjoint lash cohort inherit
        // the iris name on the immediately following exposure.
        let impostor = component(vec![10, 11, 12, 13], [51.0, 50.0], [52.0, 50.0]);
        let switched = identity.continuity(&impostor, eye_region);
        assert!(!switched.compatible);
        assert_eq!(switched.track_overlap, 0.0);

        // A strongly overlapping material cohort remains the iris even if an
        // ill-conditioned instantaneous tensor puts its fixed point far away;
        // the component is usable, but that origin must not be published.
        let bad_origin = component(vec![1, 2, 3, 9], [51.0, 50.0], [150.0, 50.0]);
        let continued = identity.continuity(&bad_origin, eye_region);
        assert!(continued.compatible);
        assert!(continued.track_overlap >= RELATION_IRIS_IDENTITY_STRONG_OVERLAP);
        assert!(!continued.origin_consistent);

        // After a bounded dropout, fresh feature identities may re-enter only
        // if their material centroid and shared origin agree with the prior.
        identity.observe(None, RelationIrisIdentityContinuity::default(), 0.0);
        identity.observe(None, RelationIrisIdentityContinuity::default(), 0.0);
        let reentered = component(vec![20, 21, 22, 23], [52.0, 49.0], [53.0, 49.0]);
        let reentry = identity.continuity(&reentered, eye_region);
        assert!(reentry.compatible);
        identity.observe(Some(&reentered), reentry, 1.0);
        assert_eq!(identity.confirmations, 1);
        assert!(!identity.confirmed());
    }

    #[test]
    fn relation_iris_identity_accumulates_motion_weighted_confirmation() {
        let component = MotionRelationComponent {
            members: (0..4).collect(),
            track_ids: vec![1, 2, 3, 4],
            centroid: [50.0, 50.0],
            coherence: 0.8,
            shared_origin: [50.0, 50.0],
            origin_spread: 1.0,
            origin_valid: true,
            persistent_edges: 3,
            persistent_nodes: 4,
        };
        let continuing = RelationIrisIdentityContinuity {
            track_overlap: 1.0,
            ..RelationIrisIdentityContinuity::default()
        };

        let mut strong = PersistentRelationIrisIdentity::default();
        strong.observe(
            Some(&component),
            RelationIrisIdentityContinuity::default(),
            1.0,
        );
        assert!(!strong.confirmed());
        strong.observe(Some(&component), continuing, 1.0);
        assert!(strong.confirmed());

        let weak_evidence = relation_iris_observation_evidence(RELATION_IRIS_MIN_DIFFERENTIAL_PX);
        let mut weak = PersistentRelationIrisIdentity::default();
        weak.observe(
            Some(&component),
            RelationIrisIdentityContinuity::default(),
            weak_evidence,
        );
        weak.observe(Some(&component), continuing, weak_evidence);
        assert!(!weak.confirmed());
        weak.observe(Some(&component), continuing, weak_evidence);
        assert!(!weak.confirmed());
        weak.observe(Some(&component), continuing, weak_evidence);
        assert!(weak.confirmed());
    }

    #[test]
    fn avx2_relation_tensor_scoring_matches_scalar_with_a_tail() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let center = [64.0f32, 48.0f32];
        let motion = SimilarityMotion {
            translation: [1.7, -0.8],
            rotation: 0.018,
            scale_delta: -0.006,
            ..SimilarityMotion::default()
        };
        let matches = (0..19)
            .map(|index| {
                let previous = [
                    8.0 + (index * 17 % 103) as f32,
                    8.0 + (index * 29 % 79) as f32,
                ];
                let mut current = motion.predict(previous, center);
                if index < 15 {
                    current[0] += (index as f32 * 0.17).sin() * 0.08;
                    current[1] += (index as f32 * 0.23).cos() * 0.08;
                } else {
                    current[0] += 3.0 + index as f32 * 0.1;
                }
                Match {
                    track_index: index,
                    previous,
                    current,
                    score: 1.0,
                    object: 0,
                    z: 0.0,
                    assignment_margin: 0.0,
                    layer_evidence: false,
                    normal_flow_evidence: false,
                    specularity: 0.0,
                }
            })
            .collect::<Vec<_>>();
        let nodes = MotionRelationNodes::from_matches((0..matches.len()).collect(), &matches);
        let scalar = score_relation_tensor_scalar(motion, center, &nodes, 0.35);
        let simd = unsafe { score_relation_tensor_avx2(motion, center, &nodes, 0.35) };
        assert_eq!(simd.support, scalar.support);
        assert!((simd.residual - scalar.residual).abs() < 1.0e-5);
        assert_eq!(simd.support_fingerprint, scalar.support_fingerprint);

        let mut scalar_errors = vec![0.0f32; nodes.len()];
        let mut simd_errors = vec![0.0f32; nodes.len()];
        relation_tensor_squared_errors_scalar(motion, center, &nodes, &mut scalar_errors);
        unsafe { relation_tensor_squared_errors_avx2(motion, center, &nodes, &mut simd_errors) };
        assert!(scalar_errors
            .iter()
            .zip(simd_errors.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-5));
    }

    #[test]
    fn semantic_eye_split_keeps_general_pupil_and_reflection_distinct() {
        let global = SimilarityMotion {
            translation: [1.0, 0.0],
            support: 10,
            ..SimilarityMotion::default()
        };
        let groups = [
            (GENERAL_LAYER, [10.0, 10.0], [0.1, 0.0], 0.4),
            (GENERAL_LAYER, [90.0, 10.0], [-0.1, 0.1], 0.5),
            (GENERAL_LAYER, [10.0, 90.0], [0.0, -0.1], 0.3),
            (GENERAL_LAYER, [90.0, 90.0], [0.1, 0.1], 0.4),
            (PUPIL_LAYER, [35.0, 45.0], [2.0, 0.1], 0.8),
            (PUPIL_LAYER, [40.0, 55.0], [2.2, -0.1], 0.7),
            (PUPIL_LAYER, [60.0, 45.0], [1.8, 0.0], 0.9),
            (PUPIL_LAYER, [65.0, 55.0], [2.1, 0.1], 0.8),
            (REFLECTION_LAYER, [50.0, 48.0], [4.0, 0.0], 3.2),
            (REFLECTION_LAYER, [55.0, 51.0], [4.2, 0.1], 3.0),
        ];
        let mut tracks = Vec::new();
        let mut matches = Vec::new();
        for (index, (object, previous, residual, specularity)) in groups.into_iter().enumerate() {
            tracks.push(FeatureTrack {
                id: index as u64,
                points: VecDeque::from([[previous[0], previous[1], 0.0]]),
                object,
                age: 0,
                last_seen_timestamp_ns: 0,
                score: 1.0,
                motion_ema: [global.translation[0] + residual[0], residual[1]],
                motion_variance: 0.01,
                matched_streak: 3,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity,
                assignment_confidence: 0.8,
                edge_normal: [1.0, 0.0],
                residual_history: VecDeque::from([residual; MIN_MOTION_SIGNATURE]),
                focus_bins: Vec::new(),
                focus_peak: None,
            });
            matches.push(Match {
                track_index: index,
                previous,
                current: [
                    previous[0] + global.translation[0] + residual[0],
                    previous[1] + global.translation[1] + residual[1],
                ],
                score: 1.0,
                object,
                z: 0.0,
                assignment_margin: 0.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity,
            });
        }
        let frame = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width: 100,
            height: 100,
            pixels: vec![100; 100 * 100],
        };
        let mut motions = [SimilarityMotion::default(); OBJECTS];
        let mut layers = [MotionLayerStatus::default(); OBJECTS];
        let mut signatures: [LayerMotionSignature; OBJECTS] = Default::default();
        let mut axis = [0.0; 2];
        let mut eye_center = None;
        let mut eye_region = None;
        let mut radial_limbus_flows = Vec::new();
        let mut radial_limbus_fused = 0usize;
        let mut relation_graph = PersistentMotionRelationGraph::default();
        for _ in 0..2 {
            let mut relations = relation_graph.observe(&matches, &tracks, [50.0, 50.0], global);
            let split = cluster_semantic_eye_layers(
                &mut matches,
                &tracks,
                &frame,
                &[],
                Some(IrisEllipseSeed::circle((50.0, 50.0), 30.0)),
                &mut motions,
                &mut layers,
                &mut signatures,
                &mut axis,
                &mut eye_center,
                &mut eye_region,
                &mut radial_limbus_flows,
                &mut radial_limbus_fused,
                [50.0, 50.0],
                global,
                &mut relations,
                None,
            );
            assert!(split, "{relations:#?}");
        }
        assert_eq!(layers[GENERAL_LAYER].persistent_tracks, 4);
        assert_eq!(layers[PUPIL_LAYER].persistent_tracks, 4);
        assert_eq!(layers[REFLECTION_LAYER].persistent_tracks, 2);
        assert!(layers[..3].iter().all(|layer| layer.stable_frames >= 2));
        assert!(matches[..4]
            .iter()
            .all(|item| item.object == GENERAL_LAYER && item.layer_evidence));
        assert!(matches[4..8]
            .iter()
            .all(|item| item.object == PUPIL_LAYER && item.layer_evidence));
        assert!(matches[8..]
            .iter()
            .all(|item| item.object == REFLECTION_LAYER && item.layer_evidence));
    }

    #[test]
    fn temporal_motion_layers_keep_opposite_parallax_groups_separate() {
        let center = [60.0, 45.0];
        let mut tracks = Vec::new();
        let mut matches = Vec::new();
        for index in 0..12 {
            let slow = index < 6;
            let velocity = if slow { [0.15, 0.05] } else { [2.15, 0.05] };
            let object = if slow { 1 } else { 0 };
            let previous = [
                18.0 + (index % 6) as f32 * 15.0,
                24.0 + (index / 3) as f32 * 11.0,
            ];
            tracks.push(FeatureTrack {
                id: index as u64,
                points: VecDeque::from([
                    [previous[0] - 0.3, previous[1] - 0.1, 0.0],
                    [previous[0] - 0.15, previous[1] - 0.05, 0.0],
                    [previous[0], previous[1], 0.0],
                ]),
                object,
                age: 0,
                last_seen_timestamp_ns: 0,
                score: 1.0,
                motion_ema: velocity,
                motion_variance: 0.02,
                matched_streak: 4,
                layer_evidence: true,
                normal_flow_evidence: false,
                specularity: 0.0,
                assignment_confidence: 0.85,
                edge_normal: [1.0, 0.0],
                residual_history: VecDeque::from(if slow {
                    [[-2.0, 0.0]; MIN_MOTION_SIGNATURE - 1]
                } else {
                    [[0.0, 0.0]; MIN_MOTION_SIGNATURE - 1]
                }),
                focus_bins: Vec::new(),
                focus_peak: None,
            });
            matches.push(Match {
                track_index: index,
                previous,
                current: [previous[0] + velocity[0], previous[1] + velocity[1]],
                score: 1.0,
                object,
                z: 0.0,
                assignment_margin: 0.0,
                layer_evidence: false,
                normal_flow_evidence: false,
                specularity: 0.0,
            });
        }
        let mut motions = [SimilarityMotion::default(); OBJECTS];
        let mut layers = [MotionLayerStatus::default(); OBJECTS];
        let mut signatures: [LayerMotionSignature; OBJECTS] = Default::default();
        let mut axis = [0.0; 2];
        for _ in 0..2 {
            cluster_motion_layers(
                &mut matches,
                &tracks,
                &mut motions,
                &mut layers,
                &mut signatures,
                &mut axis,
                center,
                SimilarityMotion {
                    translation: [2.15, 0.05],
                    support: tracks.len(),
                    ..SimilarityMotion::default()
                },
            );
        }
        let slow_object = matches[0].object;
        let fast_object = matches[6].object;
        assert_ne!(slow_object, fast_object);
        assert!(matches[..6].iter().all(|item| item.object == slow_object));
        assert!(matches[6..].iter().all(|item| item.object == fast_object));
        assert!(layers[slow_object].stable_frames >= MIN_LAYER_STABLE_FRAMES);
        assert!(layers[fast_object].stable_frames >= MIN_LAYER_STABLE_FRAMES);
        assert!(layers[slow_object].separation > 1.5);
        assert!(layers[fast_object].separation > 1.5);
        assert!(axis[0].hypot(axis[1]) > 0.99);
        assert!((layers[slow_object].parallax - layers[fast_object].parallax).abs() > 1.5);
    }

    #[test]
    fn octree_keeps_four_object_namespaces_separate() {
        let mut nodes = Vec::new();
        let points = vec![
            (0, [1.0, 2.0, 0.0]),
            (0, [2.0, 3.0, 0.2]),
            (3, [30.0, 20.0, -2.0]),
            (3, [32.0, 22.0, -2.4]),
        ];
        rebuild_nodes(&points, &mut nodes);
        assert!(nodes.iter().any(|node| node.object == 0));
        assert!(nodes.iter().any(|node| node.object == 3));
        assert!(!nodes.iter().any(|node| node.object == 1));
    }

    #[test]
    fn cfa_neutral_luma_suppresses_the_quad_bayer_carrier() {
        let width = 40usize;
        let height = 32usize;
        let carrier = [
            72i32, 72, -31, -31, 72, 72, -31, -31, -18, -18, 45, 45, -18, -18, 45, 45,
        ];
        let raw = (0..height)
            .flat_map(|y| (0..width).map(move |x| (512 + carrier[(y % 4) * 4 + x % 4]) as u16))
            .collect::<Vec<_>>();
        let neutral = cfa_neutral_raw(&raw, width, height);
        let interior = (4..height - 4)
            .flat_map(|y| {
                let neutral = &neutral;
                (4..width - 4).map(move |x| neutral[y * width + x])
            })
            .collect::<Vec<_>>();
        assert!(interior.iter().all(|sample| *sample == interior[0]));
        assert_eq!(interior[0], 529);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_canny_planes_are_bit_exact_at_vector_tails() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for (width, height) in [(7usize, 7usize), (11, 13), (37, 29), (64, 48)] {
            let raw = (0..width * height)
                .map(|index| {
                    let x = index % width;
                    let y = index / width;
                    ((x * 37 + y * 71 + x * y * 11 + 19) % 1024) as u16
                })
                .collect::<Vec<_>>();
            let neutral_scalar = cfa_neutral_raw_scalar(&raw, width, height);
            // SAFETY: runtime feature detection above proves AVX2 support.
            let neutral_avx2 = unsafe { cfa_neutral_raw_avx2(&raw, width, height) };
            assert_eq!(neutral_avx2, neutral_scalar, "carrier {width}x{height}");

            let blur_scalar = gaussian5_scalar(&neutral_scalar, width, height);
            // SAFETY: runtime feature detection above proves AVX2 support.
            let blur_avx2 = unsafe { gaussian5_avx2(&neutral_scalar, width, height) };
            assert_eq!(blur_avx2, blur_scalar, "primary blur {width}x{height}");
            let broad_scalar = gaussian5_f32_scalar(&blur_scalar, width, height);
            // SAFETY: runtime feature detection above proves AVX2 support.
            let broad_avx2 = unsafe { gaussian5_f32_avx2(&blur_scalar, width, height) };
            assert_eq!(broad_avx2, broad_scalar, "broad blur {width}x{height}");

            let pixel_count = width * height;
            let mut scalar_x = vec![0.0f32; pixel_count];
            let mut scalar_y = vec![0.0f32; pixel_count];
            let mut scalar_magnitude = vec![0.0f32; pixel_count];
            let mut scalar_direction = vec![0u8; pixel_count];
            scharr_gradients_scalar(
                &blur_scalar,
                width,
                height,
                &mut scalar_x,
                &mut scalar_y,
                &mut scalar_magnitude,
                &mut scalar_direction,
            );
            let mut avx2_x = vec![0.0f32; pixel_count];
            let mut avx2_y = vec![0.0f32; pixel_count];
            let mut avx2_magnitude = vec![0.0f32; pixel_count];
            let mut avx2_direction = vec![0u8; pixel_count];
            // SAFETY: runtime feature detection above proves AVX2 support.
            unsafe {
                scharr_gradients_avx2(
                    &blur_avx2,
                    width,
                    height,
                    &mut avx2_x,
                    &mut avx2_y,
                    &mut avx2_magnitude,
                    &mut avx2_direction,
                );
            }
            assert_eq!(avx2_x, scalar_x, "gradient x {width}x{height}");
            assert_eq!(avx2_y, scalar_y, "gradient y {width}x{height}");
            assert_eq!(
                avx2_magnitude, scalar_magnitude,
                "magnitude {width}x{height}"
            );
            assert_eq!(
                avx2_direction, scalar_direction,
                "direction {width}x{height}"
            );
            let mut sobel_scalar_x = vec![0.0f32; pixel_count];
            let mut sobel_scalar_y = vec![0.0f32; pixel_count];
            let mut sobel_scalar_magnitude = vec![0.0f32; pixel_count];
            let mut sobel_scalar_direction = vec![0u8; pixel_count];
            sobel_gradients_scalar(
                &blur_scalar,
                width,
                height,
                &mut sobel_scalar_x,
                &mut sobel_scalar_y,
                &mut sobel_scalar_magnitude,
                &mut sobel_scalar_direction,
            );
            let mut sobel_avx2_x = vec![0.0f32; pixel_count];
            let mut sobel_avx2_y = vec![0.0f32; pixel_count];
            let mut sobel_avx2_magnitude = vec![0.0f32; pixel_count];
            let mut sobel_avx2_direction = vec![0u8; pixel_count];
            // SAFETY: runtime feature detection above proves AVX2 support.
            unsafe {
                sobel_gradients_avx2(
                    &blur_avx2,
                    width,
                    height,
                    &mut sobel_avx2_x,
                    &mut sobel_avx2_y,
                    &mut sobel_avx2_magnitude,
                    &mut sobel_avx2_direction,
                );
            }
            assert_eq!(sobel_avx2_x, sobel_scalar_x, "Sobel x {width}x{height}");
            assert_eq!(sobel_avx2_y, sobel_scalar_y, "Sobel y {width}x{height}");
            assert_eq!(
                sobel_avx2_magnitude, sobel_scalar_magnitude,
                "Sobel magnitude {width}x{height}"
            );
            assert_eq!(
                sobel_avx2_direction, sobel_scalar_direction,
                "Sobel direction {width}x{height}"
            );
            let suppressed_scalar =
                nonmaximum_suppression_scalar(&scalar_magnitude, &scalar_direction, width, height);
            // SAFETY: runtime feature detection above proves AVX2 support.
            let suppressed_avx2 = unsafe {
                nonmaximum_suppression_avx2(
                    &avx2_x,
                    &avx2_y,
                    &avx2_magnitude,
                    &avx2_direction,
                    width,
                    height,
                )
            };
            assert_eq!(
                suppressed_avx2, suppressed_scalar,
                "nonmaximum suppression {width}x{height}"
            );
        }
    }

    fn learning_profile_fixture(width: usize, height: usize) -> Vec<u16> {
        (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = x as f64 - width as f64 * 0.52;
                    let dy = y as f64 - height as f64 * 0.49;
                    let radius = (dx / 28.0).hypot(dy / 21.0);
                    let iris_fibres = (0.43 * dx.atan2(dy) + 0.19 * radius * 28.0).sin();
                    let base = if radius < 0.48 {
                        120.0
                    } else if radius < 1.0 {
                        360.0 + 54.0 * iris_fibres
                    } else {
                        735.0
                    };
                    let fine = ((x * 37 + y * 71 + x * y * 11) % 43) as f64 - 21.0;
                    (base + fine).round().clamp(0.0, 1023.0) as u16
                })
            })
            .collect()
    }

    #[test]
    fn learning_canny_profiles_change_the_native_hysteresis_field() {
        let width = 112usize;
        let height = 84usize;
        let frame = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: cfa_neutral_raw(&learning_profile_fixture(width, height), width, height),
        };
        let profiles = [
            LearningCannyProfile::CannyBalanced,
            LearningCannyProfile::CannySensitive,
            LearningCannyProfile::CannyStrict,
            LearningCannyProfile::SobelSharp,
            LearningCannyProfile::SobelSmooth,
            LearningCannyProfile::ScharrCanny,
            LearningCannyProfile::Laplacian,
            LearningCannyProfile::DifferenceOfGaussians,
            LearningCannyProfile::GradientCompass,
        ];
        let fields = profiles
            .into_iter()
            .map(|profile| canny_field_with_profile(&frame, profile).accepted)
            .collect::<Vec<_>>();
        assert!(fields
            .iter()
            .all(|field| field.len() == width * height && field.iter().any(|value| *value)));
        let distinct = (0..fields.len())
            .filter(|index| fields[*index] != fields[0])
            .count();
        assert!(
            distinct >= 7,
            "K profiles collapsed onto the balanced calculation: distinct={distinct}"
        );
    }

    #[test]
    fn changing_learning_canny_profile_resets_old_edge_tracks() {
        let width = 112usize;
        let height = 84usize;
        let raw = learning_profile_fixture(width, height);
        let mut tracker = FourMotionOctrees::default();
        let _ = tracker.observe_with_iris_seed_at_with_canny_profile(
            &raw,
            width,
            height,
            0,
            0,
            1_000_000_000,
            None,
            LearningCannyProfile::CannyBalanced,
            None,
        );
        let established = tracker.observe_with_iris_seed_at_with_canny_profile(
            &raw,
            width,
            height,
            0,
            0,
            1_020_000_000,
            None,
            LearningCannyProfile::CannyBalanced,
            None,
        );
        assert!(
            !established.trails.is_empty(),
            "fixture failed to establish temporal tracks"
        );
        let changed = tracker.observe_with_iris_seed_at_with_canny_profile(
            &raw,
            width,
            height,
            0,
            0,
            1_040_000_000,
            None,
            LearningCannyProfile::CannyStrict,
            None,
        );
        assert_eq!(
            changed.learning_canny_profile,
            LearningCannyProfile::CannyStrict
        );
        assert_eq!(changed.learning_canny_mask.len(), width * height);
        assert!(
            changed.trails.is_empty(),
            "tracks learned under the old Canny definition survived K"
        );
        assert!(!changed.provisional_features.is_empty());
    }

    #[test]
    fn signed_canny_edges_recover_a_seeded_occluded_ellipse() {
        let width = 220usize;
        let height = 180usize;
        let center = (111.0f64, 91.0f64);
        let major = 58.0f64;
        let minor = 43.0f64;
        let angle = 0.30f64;
        let axis_cosine = angle.cos();
        let axis_sine = angle.sin();
        let carrier = [
            28i32, 28, -15, -15, 28, 28, -15, -15, -11, -11, 18, 18, -11, -11, 18, 18,
        ];
        let mut raw = vec![0u16; width * height];
        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - center.0;
                let dy = y as f64 - center.1;
                let local_x = axis_cosine * dx + axis_sine * dy;
                let local_y = -axis_sine * dx + axis_cosine * dy;
                let radius = (local_x / major).hypot(local_y / minor);
                let pupil = (dx / 17.0).hypot(dy / 14.0);
                let texture = ((x * 17 + y * 29) % 13) as i32 - 6;
                let mut value = if pupil < 1.0 {
                    105
                } else if radius < 1.0 {
                    295 + texture * 2
                } else {
                    735 + texture
                };
                // A dark upper-lid strip removes a real section of the limbus
                // and contributes a strong non-elliptic edge of its own.
                if y as f64 + 0.12 * (x as f64) < 74.0 {
                    value = 175 + texture;
                }
                value += carrier[(y % 4) * 4 + x % 4];
                raw[y * width + x] = value.clamp(0, 1023) as u16;
            }
        }

        let mut tracker = FourMotionOctrees::default();
        let mut overlay = tracker.observe(&raw, width, height, 0, 0, None, true);
        let seed_radius = (major * minor).sqrt();
        assert!(feature_cluster_iris_hypothesis(
            &overlay,
            width,
            height,
            Some(IrisEllipseSeed::circle(center, seed_radius)),
            0.0,
        )
        .is_none());
        // Current-frame Canny is deliberately insufficient. Associate the
        // true limbus edges with a stable temporal layer; the strong synthetic
        // eyelid edge remains untracked and therefore cannot vote.
        overlay.trails = (0..32)
            .map(|step| {
                let phase = std::f64::consts::TAU * step as f64 / 32.0;
                let local_x = major * phase.cos();
                let local_y = minor * phase.sin();
                let x = (center.0 + axis_cosine * local_x - axis_sine * local_y) as f32;
                let y = (center.1 + axis_sine * local_x + axis_cosine * local_y) as f32;
                OverlayTrail {
                    id: step,
                    object: 1,
                    match_score: 1.0,
                    matched_streak: 4,
                    layer_evidence: true,
                    normal_flow_evidence: false,
                    specularity: 0.0,
                    assignment_confidence: 0.9,
                    motion_ema: [0.8, -0.2],
                    motion_variance: 0.02,
                    residual_history: vec![[0.8, -0.2]; MIN_MOTION_SIGNATURE],
                    points: vec![
                        TrailPoint {
                            x: x - 1.6,
                            y: y + 0.4,
                            z: 1.5,
                        },
                        TrailPoint {
                            x: x - 0.8,
                            y: y + 0.2,
                            z: 1.5,
                        },
                        TrailPoint { x, y, z: 1.5 },
                    ],
                }
            })
            .collect();
        overlay.motions[0] = SimilarityMotion {
            translation: [0.1, 0.0],
            residual: 0.25,
            support: 12,
            ..SimilarityMotion::default()
        };
        overlay.motions[1] = SimilarityMotion {
            translation: [1.6, -0.4],
            residual: 0.20,
            support: 32,
            ..SimilarityMotion::default()
        };
        overlay.layers[0] = MotionLayerStatus {
            differential: [0.0, 0.0],
            coherence: 0.8,
            separation: 1.5,
            persistent_tracks: 12,
            stable_frames: 4,
            ..MotionLayerStatus::default()
        };
        overlay.layers[1] = MotionLayerStatus {
            centroid: [center.0 as f32, center.1 as f32],
            differential: [1.5, -0.4],
            parallax: 1.55,
            coherence: 0.94,
            separation: 1.5,
            persistent_tracks: 32,
            stable_frames: 4,
            ..MotionLayerStatus::default()
        };
        let hypothesis = feature_cluster_iris_hypothesis(
            &overlay,
            width,
            height,
            Some(IrisEllipseSeed::circle(
                (center.0 + 2.0, center.1 - 1.0),
                seed_radius * 1.03,
            )),
            0.0,
        )
        .expect("motion-layer-associated signed limbus edges should fit an ellipse");
        assert_eq!(hypothesis.motion_layer, 1);

        let exact_seed = IrisEllipseSeed {
            center,
            major_radius: major,
            minor_radius: minor,
            angle,
        };
        let exact_seed_candidate =
            feature_cluster_iris_hypothesis(&overlay, width, height, Some(exact_seed), 0.0)
                .expect("the exact seed should have coherent signed edge evidence");
        assert!(exact_seed_candidate.seed_edge_score >= 0.18);
        assert!(
            feature_cluster_iris_hypothesis(&overlay, width, height, Some(exact_seed), 1.0,)
                .is_none()
        );

        assert!((hypothesis.center.0 - center.0).abs() < 3.5);
        assert!((hypothesis.center.1 - center.1).abs() < 3.5);
        assert!((hypothesis.major_radius - major).abs() < 5.0);
        assert!((hypothesis.minor_radius - minor).abs() < 5.0);
        let angle_error = (hypothesis.angle - angle)
            .rem_euclid(std::f64::consts::PI)
            .min((angle - hypothesis.angle).rem_euclid(std::f64::consts::PI));
        assert!(
            angle_error < 0.22,
            "angle error={angle_error} fit center={:?} axes={:.3}/{:.3} angle={:.3} support={} coverage={} opposition={}",
            hypothesis.center,
            hypothesis.major_radius,
            hypothesis.minor_radius,
            hypothesis.angle,
            hypothesis.edge_support,
            hypothesis.angular_coverage,
            hypothesis.opposing_meridians,
        );
        let fitted = EdgeEllipse {
            center: hypothesis.center,
            major: hypothesis.major_radius,
            minor: hypothesis.minor_radius,
            angle: hypothesis.angle,
        };
        let boundary_rms = ((0..180)
            .map(|step| {
                let phase = std::f64::consts::TAU * step as f64 / 180.0;
                let local_x = major * phase.cos();
                let local_y = minor * phase.sin();
                let point = (
                    center.0 + axis_cosine * local_x - axis_sine * local_y,
                    center.1 + axis_sine * local_x + axis_cosine * local_y,
                );
                let residual =
                    (normalized_ellipse_radius(fitted, point) - 1.0).abs() * fitted.minor;
                residual * residual
            })
            .sum::<f64>()
            / 180.0)
            .sqrt();
        assert!(boundary_rms < 3.25, "boundary RMS was {boundary_rms}");
        assert!(hypothesis.angular_coverage >= 10);
        assert!(hypothesis.opposing_meridians >= 3);
    }

    #[test]
    fn unilateral_edge_arc_is_rejected() {
        let center = (100.0f64, 80.0f64);
        let radius = 48.0f64;
        let edges = (0..80)
            .map(|step| {
                let phase = -1.25 + 2.50 * step as f64 / 79.0;
                EdgeEvidence {
                    x: (center.0 + radius * phase.cos()) as f32,
                    y: (center.1 + radius * phase.sin()) as f32,
                    gradient_x: phase.cos() as f32,
                    gradient_y: phase.sin() as f32,
                    strength: 1.0,
                    iris_motion_consistency: 1.0,
                    ..EdgeEvidence::default()
                }
            })
            .collect::<Vec<_>>();
        let exact = EdgeEllipse {
            center,
            major: radius,
            minor: radius,
            angle: 0.0,
        };
        let evidence = score_edge_ellipse(exact, &edges, (center, radius), false);
        assert_eq!(evidence.opposing_meridians, 0);
        let overlay = MotionOctreeOverlay {
            edges,
            ..MotionOctreeOverlay::default()
        };
        assert!(feature_cluster_iris_hypothesis(
            &overlay,
            200,
            160,
            Some(IrisEllipseSeed::circle(center, radius)),
            0.0,
        )
        .is_none());
    }

    #[test]
    fn limbus_feature_budget_affinity_prefers_a_radial_raw_material_transition() {
        let seed = IrisEllipseSeed {
            center: (96.0, 72.0),
            major_radius: 44.0,
            minor_radius: 31.0,
            angle: 0.23,
        };
        let ellipse = seed.ellipse();
        let phase = 0.71f64;
        let (angle_sine, angle_cosine) = ellipse.angle.sin_cos();
        let local_x = ellipse.major * phase.cos();
        let local_y = ellipse.minor * phase.sin();
        let point = [
            (ellipse.center.0 + angle_cosine * local_x - angle_sine * local_y) as f32,
            (ellipse.center.1 + angle_sine * local_x + angle_cosine * local_y) as f32,
        ];
        let normal = normalized_vector([
            (angle_cosine * local_x / ellipse.major.powi(2)
                - angle_sine * local_y / ellipse.minor.powi(2)) as f32,
            (angle_sine * local_x / ellipse.major.powi(2)
                + angle_cosine * local_y / ellipse.minor.powi(2)) as f32,
        ]);
        let good = EdgeEvidence {
            x: point[0],
            y: point[1],
            gradient_x: normal[0],
            gradient_y: normal[1],
            strength: 1.4,
            multiscale_consistency: 0.94,
            signed_step_persistence: 0.91,
            dark_side_texture: 0.92,
            bright_side_texture: 0.10,
            ..EdgeEvidence::default()
        };
        let (good_affinity, _) =
            limbus_feature_budget_affinity(good, seed).expect("radial limbus edge");
        assert!(good_affinity >= 0.72, "affinity={good_affinity}");

        let balanced = EdgeEvidence {
            dark_side_texture: 0.51,
            bright_side_texture: 0.49,
            ..good
        };
        let (balanced_affinity, _) =
            limbus_feature_budget_affinity(balanced, seed).expect("same geometric edge");
        assert!(
            good_affinity > balanced_affinity + 0.18,
            "split={good_affinity} balanced={balanced_affinity}"
        );

        let tangential = EdgeEvidence {
            gradient_x: -normal[1],
            gradient_y: normal[0],
            ..good
        };
        assert!(limbus_feature_budget_affinity(tangential, seed).is_none());
        let off_ring = EdgeEvidence {
            x: (seed.center.0 + 0.60 * (point[0] as f64 - seed.center.0)) as f32,
            y: (seed.center.1 + 0.60 * (point[1] as f64 - seed.center.1)) as f32,
            ..good
        };
        assert!(limbus_feature_budget_affinity(off_ring, seed).is_none());
    }

    #[test]
    fn limbus_feature_reserve_keeps_distributed_opposing_raw_edges() {
        let width = 192usize;
        let height = 144usize;
        let seed = IrisEllipseSeed {
            center: (96.0, 72.0),
            major_radius: 49.0,
            minor_radius: 35.0,
            angle: 0.16,
        };
        let ellipse = seed.ellipse();
        let (angle_sine, angle_cosine) = ellipse.angle.sin_cos();
        let mut raw = vec![0u16; width * height];
        for y in 0..height {
            for x in 0..width {
                let radius = normalized_ellipse_radius(ellipse, (x as f64, y as f64));
                let texture = ((x * 17 + y * 29) % 31) as u16;
                raw[y * width + x] = if radius < 1.0 {
                    260 + 3 * texture
                } else {
                    720 + texture
                };
            }
        }
        let frame = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: raw,
        };
        let canny = canny_field(&frame);
        let edges = (0..LIMBUS_FEATURE_SECTORS)
            .map(|sector| {
                let phase = std::f64::consts::TAU * sector as f64 / LIMBUS_FEATURE_SECTORS as f64;
                let local_x = ellipse.major * phase.cos();
                let local_y = ellipse.minor * phase.sin();
                let point = [
                    (ellipse.center.0 + angle_cosine * local_x - angle_sine * local_y) as f32,
                    (ellipse.center.1 + angle_sine * local_x + angle_cosine * local_y) as f32,
                ];
                let normal = normalized_vector([
                    (angle_cosine * local_x / ellipse.major.powi(2)
                        - angle_sine * local_y / ellipse.minor.powi(2)) as f32,
                    (angle_sine * local_x / ellipse.major.powi(2)
                        + angle_cosine * local_y / ellipse.minor.powi(2))
                        as f32,
                ]);
                EdgeEvidence {
                    x: point[0],
                    y: point[1],
                    gradient_x: normal[0],
                    gradient_y: normal[1],
                    strength: 1.5,
                    multiscale_consistency: 0.95,
                    signed_step_persistence: 0.93,
                    dark_side_texture: 0.88,
                    bright_side_texture: 0.12,
                    ..EdgeEvidence::default()
                }
            })
            .collect::<Vec<_>>();
        let selected = seed_points(&frame, Some(&canny), &edges, Some(seed), &[], 40);
        let retained_limbus = selected
            .iter()
            .filter(|(point, _)| {
                (normalized_ellipse_radius(ellipse, (point[0] as f64, point[1] as f64)) - 1.0).abs()
                    <= 0.025
            })
            .count();
        assert!(
            retained_limbus >= 12,
            "retained only {retained_limbus} of {} selected features: {selected:?}",
            selected.len()
        );
        assert!(selected.len() <= 40);
    }

    #[test]
    #[ignore = "set BUTTERCUP_LIMBUS_FEATURE_LABEL and BUTTERCUP_LIMBUS_FEATURE_RAW"]
    fn limbus_feature_budget_replays_a_human_labeled_lossless_raw() {
        let label_path = std::env::var("BUTTERCUP_LIMBUS_FEATURE_LABEL")
            .expect("set BUTTERCUP_LIMBUS_FEATURE_LABEL to a reviewed labels JSON");
        let raw_path = std::env::var("BUTTERCUP_LIMBUS_FEATURE_RAW")
            .expect("set BUTTERCUP_LIMBUS_FEATURE_RAW to the corresponding RAW10 export");
        let document: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&label_path).expect("read human limbus labels JSON"),
        )
        .expect("parse human limbus labels JSON");
        assert_eq!(
            document.get("reviewed").and_then(serde_json::Value::as_bool),
            Some(true),
            "the corpus replay must not treat an unreviewed algorithm proposal as truth"
        );
        let width = document
            .get("frame_width")
            .and_then(serde_json::Value::as_u64)
            .expect("frame_width") as usize;
        let height = document
            .get("frame_height")
            .and_then(serde_json::Value::as_u64)
            .expect("frame_height") as usize;
        let fit = document.get("ellipse_fit").expect("human ellipse fit");
        let center = fit
            .get("center")
            .and_then(serde_json::Value::as_array)
            .expect("ellipse center");
        let radii = fit
            .get("radii")
            .and_then(serde_json::Value::as_array)
            .expect("ellipse radii");
        let seed = IrisEllipseSeed {
            center: (
                center[0].as_f64().expect("center x"),
                center[1].as_f64().expect("center y"),
            ),
            major_radius: radii[0].as_f64().expect("radius 0"),
            minor_radius: radii[1].as_f64().expect("radius 1"),
            angle: fit
                .get("angle_degrees")
                .and_then(serde_json::Value::as_f64)
                .expect("ellipse angle")
                .to_radians(),
        };
        let packed = std::fs::read(&raw_path).expect("read lossless RAW10");
        let raw = crate::raw10::unpack_raw10(&packed, width, height, width * 5 / 4);
        assert_eq!(raw.len(), width * height);
        let frame = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: cfa_neutral_raw(&raw, width, height),
        };
        let mut canny = canny_field(&frame);
        let edges = edge_evidence(&mut canny, width, height).edges;
        let baseline = seed_points(&frame, Some(&canny), &edges, None, &[], MAX_FEATURES);
        let discounted = seed_points(
            &frame,
            Some(&canny),
            &edges,
            Some(seed),
            &[],
            MAX_FEATURES,
        );
        let ellipse = seed.ellipse();
        let near_limbus = |point: [f32; 2]| {
            (normalized_ellipse_radius(ellipse, (point[0] as f64, point[1] as f64)) - 1.0)
                .abs()
                * ellipse.minor
                <= 4.0
        };
        let baseline_limbus = baseline
            .iter()
            .filter(|(point, _)| near_limbus(*point))
            .count();
        let discounted_limbus = discounted
            .iter()
            .filter(|(point, _)| near_limbus(*point))
            .count();
        let discounted_affine_edges = edges
            .iter()
            .filter(|edge| {
                limbus_feature_budget_affinity(**edge, seed)
                    .is_some_and(|(affinity, _)| affinity >= LIMBUS_FEATURE_MIN_AFFINITY)
            })
            .count();
        eprintln!(
            "limbus-feature-budget label={label_path} raw={raw_path} edges={} eligible={} baseline={baseline_limbus}/{} discounted={discounted_limbus}/{}",
            edges.len(),
            discounted_affine_edges,
            baseline.len(),
            discounted.len(),
        );
        assert!(discounted.len() <= MAX_FEATURES);
        assert!(
            discounted_limbus >= baseline_limbus,
            "discount removed human-limbus features: baseline={baseline_limbus} discounted={discounted_limbus}"
        );
    }

    #[test]
    fn canny_work_is_scoped_to_the_cluster_feature_mode() {
        let width = 64usize;
        let height = 48usize;
        let raw = (0..height)
            .flat_map(|_| (0..width).map(|x| if x < width / 2 { 180 } else { 760 }))
            .collect::<Vec<u16>>();
        let mut tracker = FourMotionOctrees::default();
        let native = tracker.observe(&raw, width, height, 0, 0, None, false);
        assert!(native.edges.is_empty());
        assert_eq!(native.edge_high_threshold, 0.0);

        let clusters = tracker.observe(&raw, width, height, 0, 0, None, true);
        assert!(!clusters.edges.is_empty());
        assert!(clusters.edge_high_threshold > 0.0);
        // Changing representations starts a new trail generation rather than
        // ever matching native-mosaic patches against CFA-neutral patches.
        assert!(clusters.trails.is_empty());
    }

    #[test]
    fn nautilus_fingerprint_is_affine_exposure_invariant() {
        let width = 160usize;
        let height = 120usize;
        let pixels =
            synthetic_native_similarity_texture(width, height, (80.0, 60.0), 1.0, (0.0, 0.0));
        let exposed = pixels
            .iter()
            .map(|value| (0.68 * *value as f32 + 137.0).round().clamp(0.0, 1023.0) as u16)
            .collect::<Vec<_>>();
        let first = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels,
        };
        let second = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: exposed,
        };
        let point = [73.25, 54.75];
        let normal = normalized_vector([0.78, -0.63]);
        let left = nautilus_fingerprint(&first, point, normal).expect("textured fingerprint");
        let right = nautilus_fingerprint(&second, point, normal).expect("exposed fingerprint");
        assert!(
            nautilus_fingerprint_distance(&left, &right) <= 0.035,
            "left={left:?} right={right:?} distance={}",
            nautilus_fingerprint_distance(&left, &right),
        );
    }

    #[test]
    fn nautilus_tree_uniquely_relocates_a_shifted_native_raw_point() {
        let width = 160usize;
        let height = 120usize;
        let translation = [4.0f32, -3.0f32];
        let previous = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: synthetic_native_similarity_texture(
                width,
                height,
                (80.0, 60.0),
                1.0,
                (0.0, 0.0),
            ),
        };
        let current = RawFrame {
            sensor_x: 0,
            sensor_y: 0,
            width,
            height,
            pixels: synthetic_native_similarity_texture(
                width,
                height,
                (80.0, 60.0),
                1.0,
                (translation[0] as f64, translation[1] as f64),
            ),
        };
        let source = [71.0f32, 57.0f32];
        let target = [source[0] + translation[0], source[1] + translation[1]];
        let normal = normalized_vector([0.83, 0.56]);
        let query = nautilus_fingerprint(&previous, source, normal).expect("source fingerprint");
        let mut edges = vec![EdgeEvidence {
            x: target[0],
            y: target[1],
            gradient_x: normal[0],
            gradient_y: normal[1],
            strength: 4.0,
            ..EdgeEvidence::default()
        }];
        for y in [22.0f32, 40.0, 76.0, 96.0] {
            for x in [24.0f32, 48.0, 102.0, 132.0] {
                edges.push(EdgeEvidence {
                    x,
                    y,
                    gradient_x: normal[0],
                    gradient_y: normal[1],
                    strength: 2.0,
                    ..EdgeEvidence::default()
                });
            }
        }
        let tree = NautilusFingerprintTree::from_edges(&current, &edges);
        let mut diagnostics = MatchDiagnostics::default();
        let ranked = tree.query_nearest(&query, target, 58.0, normal, &[], 4, &mut diagnostics);
        assert!(ranked.len() >= 2, "ranked={ranked:?}");
        let best = tree.candidates[ranked[0].1];
        assert!(
            (best.point[0] - target[0]).hypot(best.point[1] - target[1]) < 0.1,
            "target={target:?} best={best:?} ranked={ranked:?}",
        );
        assert!(ranked[1].0 - ranked[0].0 >= NAUTILUS_MIN_ABSOLUTE_MARGIN);
        assert!(diagnostics.nautilus_descriptor_evaluations < tree.candidates.len());
    }

    #[test]
    fn nautilus_reidentification_withholds_duplicate_fingerprints() {
        let fingerprint = NautilusFingerprint {
            values: std::array::from_fn(|index| (index as i32 * 3 - 120).clamp(-127, 127) as i8),
            quality: 1.0,
        };
        let mut tree = NautilusFingerprintTree {
            candidates: vec![
                NautilusCandidate {
                    point: [30.0, 30.0],
                    normal: [1.0, 0.0],
                    fingerprint,
                    evidence: 1.0,
                },
                NautilusCandidate {
                    point: [42.0, 30.0],
                    normal: [1.0, 0.0],
                    fingerprint,
                    evidence: 1.0,
                },
            ],
            ..NautilusFingerprintTree::default()
        };
        tree.root = Some(tree.append_node(vec![0, 1]));
        let bank = NautilusFingerprintBank {
            observations: VecDeque::from([fingerprint, fingerprint]),
        };
        let mut diagnostics = MatchDiagnostics::default();
        let ranked = nautilus_rank_bank_candidates(
            &tree,
            &bank,
            [36.0, 30.0],
            20.0,
            [1.0, 0.0],
            &[],
            &mut diagnostics,
        );
        assert_eq!(ranked.len(), 2);
        let margin = ranked[1].1 - ranked[0].1;
        let ratio = ranked[0].1 / ranked[1].1.max(1.0e-5);
        assert!(
            margin < NAUTILUS_MIN_ABSOLUTE_MARGIN || ratio > NAUTILUS_MAX_DISTANCE_RATIO,
            "duplicate candidates unexpectedly passed the ambiguity gate: {ranked:?}",
        );
    }

    #[derive(Clone)]
    struct NautilusCorpusSnapshot {
        tracks: BTreeMap<u64, FeatureTrack>,
        banks: BTreeMap<u64, NautilusFingerprintBank>,
        motions: [SimilarityMotion; OBJECTS],
        layers: [MotionLayerStatus; OBJECTS],
        center: [f32; 2],
    }

    #[derive(Clone, Copy, Debug)]
    struct NautilusCorpusScore {
        correct: bool,
        best_distance: f32,
        margin: f32,
        ratio: f32,
        selected_prediction_distance: f32,
        target_prediction_distance: f32,
    }

    #[derive(Clone, Copy, Debug)]
    struct NautilusProductionScore {
        correct: bool,
        object: usize,
        distance: f32,
        margin: f32,
        ratio: f32,
        prediction_distance: f32,
        target_prediction_distance: f32,
        anchor_fallback_disagreement: f32,
        anchor_residual: f32,
        same_object_support: usize,
        identity_radius: f32,
        reverse_margin: f32,
        normal_alignment: f32,
        confidence: f32,
        anchor_conditioned: bool,
    }

    #[derive(Clone, Debug, Default)]
    struct NautilusCorpusStats {
        oracle_correspondences: usize,
        prediction_misses: usize,
        candidate_covered: usize,
        top_one_correct: usize,
        accepted_correct: usize,
        accepted_wrong: usize,
        withheld_ambiguous: usize,
        ranked_queries: usize,
        target_error_sum: f32,
        descriptor_evaluations: usize,
        production_accepted_correct: usize,
        production_accepted_wrong: usize,
        production_accepted_unverified: usize,
        production_relocations: usize,
        production_reverse_ambiguous: usize,
        production_collisions: usize,
        scores: Vec<NautilusCorpusScore>,
        production_scores: Vec<NautilusProductionScore>,
    }

    fn evaluate_nautilus_production_gap(
        gap: usize,
        source: &NautilusCorpusSnapshot,
        current_tracks: &BTreeMap<u64, FeatureTrack>,
        current_frame: &RawFrame,
        current_tree: &NautilusFingerprintTree,
        current_sensor_origin: [f32; 2],
        stats: &mut NautilusCorpusStats,
    ) {
        let common_ids = source
            .tracks
            .keys()
            .filter(|id| {
                current_tracks.contains_key(id)
                    && source.banks.get(id).is_some_and(|bank| {
                        bank.observations.len() >= NAUTILUS_MIN_BANK_OBSERVATIONS
                    })
            })
            .copied()
            .collect::<Vec<_>>();
        // Five-fold leave-out replay: four fifths of the independently
        // tracked points act as the ordinary adjacent-frame anchors while the
        // remaining fifth must be relocated solely by its historical bank.
        // Every oracle ID is queried exactly once, and no target point is
        // exposed as an anchor in the fold where that ID is scored.
        for fold in 0..5usize {
            let mut tracks = Vec::<FeatureTrack>::new();
            let mut matches = Vec::<Match>::new();
            for (ordinal, id) in common_ids.iter().copied().enumerate() {
                let query = ordinal % 5 == fold;
                let target_track = &current_tracks[&id];
                let Some(target) = target_track.points.back().copied() else {
                    continue;
                };
                let track_index = tracks.len();
                if query {
                    let mut track = source.tracks[&id].clone();
                    track.age = gap.saturating_sub(1) as u8;
                    tracks.push(track);
                } else {
                    // Reconstruct the state immediately before the current
                    // observation: the current endpoint belongs in `Match`,
                    // while the retained history ends on the previous frame.
                    let mut track = target_track.clone();
                    track.points.pop_back();
                    let Some(previous) = track.points.back().copied() else {
                        continue;
                    };
                    track.age = 0;
                    let object = track.object;
                    let z = previous[2];
                    tracks.push(track);
                    matches.push(Match {
                        track_index,
                        previous: [previous[0], previous[1]],
                        current: [target[0], target[1]],
                        score: 1.0,
                        object,
                        z,
                        assignment_margin: 1.0,
                        layer_evidence: true,
                        normal_flow_evidence: false,
                        specularity: 0.0,
                    });
                }
            }
            let mut diagnostics = MatchDiagnostics::default();
            let relocations = propose_nautilus_relocations(
                current_frame,
                current_tree,
                &tracks,
                &source.banks,
                &matches,
                &source.motions,
                &source.layers,
                source.center,
                current_sensor_origin,
                true,
                false,
                &mut diagnostics,
            );
            stats.production_relocations += relocations.len();
            stats.production_reverse_ambiguous += diagnostics.nautilus_reverse_ambiguous;
            stats.production_collisions += diagnostics.nautilus_collision_rejected;
            for relocation in relocations {
                let track = &tracks[relocation.track_index];
                let Some(target) = current_tracks
                    .get(&track.id)
                    .and_then(|target_track| target_track.points.back())
                else {
                    stats.production_accepted_unverified += 1;
                    continue;
                };
                let error =
                    (relocation.current[0] - target[0]).hypot(relocation.current[1] - target[1]);
                let (fallback, _) = nautilus_track_prediction(
                    track,
                    &source.motions,
                    &source.layers,
                    source.center,
                    true,
                );
                let prediction = nautilus_anchor_conditioned_prediction(
                    relocation.track_index,
                    track,
                    &tracks,
                    &matches,
                    fallback,
                )
                .unwrap_or(NautilusAnchorPrediction {
                    point: fallback,
                    search_radius: 0.0,
                    identity_radius: 0.0,
                    fallback_disagreement: 0.0,
                    residual: f32::INFINITY,
                    same_object_support: 0,
                });
                if error <= 4.0 {
                    stats.production_accepted_correct += 1;
                } else {
                    stats.production_accepted_wrong += 1;
                }
                stats.production_scores.push(NautilusProductionScore {
                    correct: error <= 4.0,
                    object: track.object,
                    distance: relocation.distance,
                    margin: relocation.margin,
                    ratio: relocation.distance_ratio,
                    prediction_distance: relocation.prediction_distance,
                    target_prediction_distance: (prediction.point[0] - target[0])
                        .hypot(prediction.point[1] - target[1]),
                    anchor_fallback_disagreement: prediction.fallback_disagreement,
                    anchor_residual: prediction.residual,
                    same_object_support: prediction.same_object_support,
                    identity_radius: relocation.identity_radius,
                    reverse_margin: relocation.reverse_margin,
                    normal_alignment: relocation.normal_alignment,
                    confidence: relocation.confidence,
                    anchor_conditioned: relocation.anchor_conditioned,
                });
            }
        }
    }

    fn evaluate_nautilus_corpus_gap(
        gap: usize,
        source: &NautilusCorpusSnapshot,
        current_tracks: &BTreeMap<u64, FeatureTrack>,
        current_frame: &RawFrame,
        current_tree: &NautilusFingerprintTree,
        current_sensor_origin: [f32; 2],
        stats: &mut NautilusCorpusStats,
    ) {
        for (id, source_track) in &source.tracks {
            let Some(target_track) = current_tracks.get(id) else {
                continue;
            };
            let Some(bank) = source.banks.get(id) else {
                continue;
            };
            if bank.observations.len() < NAUTILUS_MIN_BANK_OBSERVATIONS {
                continue;
            }
            let Some(target_sensor) = target_track.points.back() else {
                continue;
            };
            stats.oracle_correspondences += 1;
            let mut missing_track = source_track.clone();
            missing_track.age = gap.saturating_sub(1) as u8;
            let (predicted_sensor, radius) = nautilus_track_prediction(
                &missing_track,
                &source.motions,
                &source.layers,
                source.center,
                true,
            );
            let predicted = [
                predicted_sensor[0] - current_sensor_origin[0],
                predicted_sensor[1] - current_sensor_origin[1],
            ];
            let target = [
                target_sensor[0] - current_sensor_origin[0],
                target_sensor[1] - current_sensor_origin[1],
            ];
            if (target[0] - predicted[0]).hypot(target[1] - predicted[1]) > radius {
                stats.prediction_misses += 1;
                continue;
            }
            let target_covered = current_tree.candidates.iter().any(|candidate| {
                (candidate.point[0] - target[0]).hypot(candidate.point[1] - target[1]) <= 6.0
            });
            stats.candidate_covered += usize::from(target_covered);
            let mut diagnostics = MatchDiagnostics::default();
            let ranked = nautilus_refine_ranked_candidates(
                current_frame,
                current_tree,
                bank,
                predicted,
                radius,
                source_track.edge_normal,
                &[],
                &mut diagnostics,
            );
            stats.descriptor_evaluations += diagnostics.nautilus_descriptor_evaluations
                + diagnostics.nautilus_refinement_evaluations;
            let Some(best) = ranked.first() else {
                stats.withheld_ambiguous += 1;
                continue;
            };
            let best_distance = best.score;
            let best_point = best.point;
            let target_error = (best_point[0] - target[0]).hypot(best_point[1] - target[1]);
            stats.ranked_queries += 1;
            stats.target_error_sum += target_error;
            let correct = target_error <= 4.0;
            stats.top_one_correct += usize::from(correct);
            let second_distance = ranked
                .get(1)
                .map_or(f32::INFINITY, |candidate| candidate.score);
            let margin = if second_distance.is_finite() {
                second_distance - best_distance
            } else {
                1.0
            };
            let ratio = if second_distance.is_finite() {
                best_distance / second_distance.max(1.0e-5)
            } else {
                0.0
            };
            let accepted = best_distance <= NAUTILUS_MAX_DISTANCE
                && margin >= NAUTILUS_MIN_ABSOLUTE_MARGIN
                && ratio <= NAUTILUS_MAX_DISTANCE_RATIO
                && (best_point[0] - predicted[0]).hypot(best_point[1] - predicted[1])
                    <= nautilus_identity_acceptance_radius(&missing_track);
            stats.scores.push(NautilusCorpusScore {
                correct,
                best_distance,
                margin,
                ratio,
                selected_prediction_distance: (best_point[0] - predicted[0])
                    .hypot(best_point[1] - predicted[1]),
                target_prediction_distance: (target[0] - predicted[0])
                    .hypot(target[1] - predicted[1]),
            });
            if accepted && correct {
                stats.accepted_correct += 1;
            } else if accepted {
                stats.accepted_wrong += 1;
            } else {
                stats.withheld_ambiguous += 1;
            }
        }
    }

    #[test]
    #[ignore = "lossless RAW corpus replay; optionally set BUTTERCUP_NAUTILUS_CORPUS"]
    fn nautilus_fingerprint_tree_recovers_unique_points_in_raw_roi_corpus() {
        use std::fs;
        use std::path::PathBuf;

        let root = std::env::var_os("BUTTERCUP_NAUTILUS_CORPUS")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    "/mnt/bulk_data/osbot-drv-data/outputs/native-limbus-session-20260813/corpus",
                )
            });
        let mut archives = fs::read_dir(&root)
            .unwrap_or_else(|error| panic!("read {}: {error}", root.display()))
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect::<Vec<_>>();
        archives.sort();
        let mut stats: [NautilusCorpusStats; 3] = std::array::from_fn(|_| Default::default());
        let mut total_frames = 0usize;
        for archive in archives {
            let records = fs::read_to_string(archive.join("frames.jsonl"))
                .unwrap_or_else(|error| panic!("read {} index: {error}", archive.display()));
            let stream = fs::read(archive.join("subject-right.raw10"))
                .unwrap_or_else(|error| panic!("read {} RAW: {error}", archive.display()));
            let mut tracker = FourMotionOctrees::default();
            tracker.set_nautilus_disabled_for_replay(true);
            let mut history = VecDeque::<NautilusCorpusSnapshot>::new();
            for record in records
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .filter(|record| {
                    record.get("label").and_then(serde_json::Value::as_str) == Some("subject-right")
                })
            {
                let offset = record["offset"].as_u64().expect("offset") as usize;
                let length = record["length"].as_u64().expect("length") as usize;
                assert_eq!(length, 480 * 256);
                let raw =
                    crate::raw10::unpack_raw10(&stream[offset..offset + length], 384, 256, 480);
                let sensor_x = record["sensor_x"].as_u64().expect("sensor_x") as u32;
                let sensor_y = record["sensor_y"].as_u64().expect("sensor_y") as u32;
                let timestamp_ns = record["timestamp_ns"].as_u64().expect("timestamp_ns");
                let overlay = tracker.observe_with_iris_seed_at(
                    &raw,
                    384,
                    256,
                    sensor_x,
                    sensor_y,
                    timestamp_ns,
                    None,
                    true,
                    None,
                );
                total_frames += 1;
                let current_frame = tracker.previous.as_ref().expect("current frame");
                let current_tree =
                    NautilusFingerprintTree::from_edges(current_frame, &overlay.edges);
                let current_tracks = tracker
                    .tracks
                    .iter()
                    .filter(|track| track.age == 0 && track.matched_streak > 0)
                    .cloned()
                    .map(|track| (track.id, track))
                    .collect::<BTreeMap<_, _>>();
                let current_sensor_origin = [sensor_x as f32, sensor_y as f32];
                for gap in 1..=3 {
                    if let Some(source) = history.iter().rev().nth(gap - 1) {
                        evaluate_nautilus_corpus_gap(
                            gap,
                            source,
                            &current_tracks,
                            current_frame,
                            &current_tree,
                            current_sensor_origin,
                            &mut stats[gap - 1],
                        );
                        evaluate_nautilus_production_gap(
                            gap,
                            source,
                            &current_tracks,
                            current_frame,
                            &current_tree,
                            current_sensor_origin,
                            &mut stats[gap - 1],
                        );
                    }
                }
                let center = [sensor_x as f32 + 192.0, sensor_y as f32 + 128.0];
                history.push_back(NautilusCorpusSnapshot {
                    tracks: current_tracks,
                    banks: tracker.nautilus_banks.clone(),
                    motions: tracker.motions,
                    layers: tracker.layers,
                    center,
                });
                while history.len() > 3 {
                    history.pop_front();
                }
            }
        }
        eprintln!("nautilus RAW corpus frames={total_frames}");
        for (index, result) in stats.iter().enumerate() {
            let mean_error = if result.ranked_queries > 0 {
                result.target_error_sum / result.ranked_queries as f32
            } else {
                0.0
            };
            eprintln!(
                "gap={} oracle={} prediction-miss={} coverage={}/{} top1-correct={} ranked-gate-correct={} ranked-gate-wrong={} withheld={} production-correct={} production-wrong={} production-unverified={} production-total={} reverse-ambiguous={} collisions={} mean-target-error={:.3}px descriptor-evals={}",
                index + 1,
                result.oracle_correspondences,
                result.prediction_misses,
                result.candidate_covered,
                result.oracle_correspondences,
                result.top_one_correct,
                result.accepted_correct,
                result.accepted_wrong,
                result.withheld_ambiguous,
                result.production_accepted_correct,
                result.production_accepted_wrong,
                result.production_accepted_unverified,
                result.production_relocations,
                result.production_reverse_ambiguous,
                result.production_collisions,
                mean_error,
                result.descriptor_evaluations,
            );
            let quantiles = |correct: bool, field: fn(&NautilusCorpusScore) -> f32| {
                let mut values = result
                    .scores
                    .iter()
                    .filter(|score| score.correct == correct)
                    .map(field)
                    .collect::<Vec<_>>();
                values.sort_by(f32::total_cmp);
                let at = |fraction: f32| {
                    values
                        .get(((values.len().saturating_sub(1)) as f32 * fraction).round() as usize)
                        .copied()
                        .unwrap_or(0.0)
                };
                [at(0.10), at(0.50), at(0.90)]
            };
            for correct in [true, false] {
                eprintln!(
                    "  {} distance-p10/50/90={:?} margin={:?} ratio={:?} selected-prediction-px={:?} target-prediction-px={:?}",
                    if correct { "correct" } else { "wrong" },
                    quantiles(correct, |score| score.best_distance),
                    quantiles(correct, |score| score.margin),
                    quantiles(correct, |score| score.ratio),
                    quantiles(correct, |score| score.selected_prediction_distance),
                    quantiles(correct, |score| score.target_prediction_distance),
                );
            }
            let production_quantiles =
                |correct: bool, field: fn(&NautilusProductionScore) -> f32| {
                    let mut values = result
                        .production_scores
                        .iter()
                        .filter(|score| score.correct == correct)
                        .map(field)
                        .collect::<Vec<_>>();
                    values.sort_by(f32::total_cmp);
                    let at = |fraction: f32| {
                        values
                            .get(((values.len().saturating_sub(1)) as f32 * fraction).round()
                                as usize)
                            .copied()
                            .unwrap_or(0.0)
                    };
                    [at(0.10), at(0.50), at(0.90)]
                };
            for correct in [true, false] {
                eprintln!(
                    "  production-{} distance={:?} margin={:?} ratio={:?} selected-prediction-px={:?} target-prediction-px={:?} anchor-vs-prior-px={:?} anchor-residual={:?} same-object-support={:?} identity-radius={:?} reverse-margin={:?} normal-alignment={:?} confidence={:?} anchor={}/{}",
                    if correct { "correct" } else { "wrong" },
                    production_quantiles(correct, |score| score.distance),
                    production_quantiles(correct, |score| score.margin),
                    production_quantiles(correct, |score| score.ratio),
                    production_quantiles(correct, |score| score.prediction_distance),
                    production_quantiles(correct, |score| score.target_prediction_distance),
                    production_quantiles(correct, |score| score.anchor_fallback_disagreement),
                    production_quantiles(correct, |score| score.anchor_residual),
                    production_quantiles(correct, |score| score.same_object_support as f32),
                    production_quantiles(correct, |score| score.identity_radius),
                    production_quantiles(correct, |score| score.reverse_margin),
                    production_quantiles(correct, |score| score.normal_alignment),
                    production_quantiles(correct, |score| score.confidence),
                    result.production_scores.iter().filter(|score| score.correct == correct && score.anchor_conditioned).count(),
                    result.production_scores.iter().filter(|score| score.correct == correct).count(),
                );
            }
            for object in 0..OBJECTS {
                let correct = result
                    .production_scores
                    .iter()
                    .filter(|score| score.object == object && score.correct)
                    .count();
                let wrong = result
                    .production_scores
                    .iter()
                    .filter(|score| score.object == object && !score.correct)
                    .count();
                eprintln!("  production-object={object} correct={correct} wrong={wrong}");
            }
            let false_budget = result.oracle_correspondences / 100;
            let mut best_stricter = None::<(usize, usize, f32, f32, f32, f32, f32)>;
            for spatial in [1.0f32, 1.5, 2.0, 2.5, 3.0] {
                for margin in [0.08f32, 0.10, 0.12, 0.15, 0.20] {
                    for ratio in [0.84f32, 0.78, 0.72, 0.66] {
                        for confidence in [0.0f32, 0.55, 0.65, 0.75, 0.85] {
                            for prior_agreement in [1.0f32, 2.0, 3.0, 4.0, 6.0, 10.0] {
                                let mut correct_count = 0usize;
                                let mut wrong_count = 0usize;
                                for score in &result.production_scores {
                                    let accepted = score.prediction_distance <= spatial
                                        && score.margin >= margin
                                        && score.ratio <= ratio
                                        && score.confidence >= confidence
                                        && score.anchor_fallback_disagreement <= prior_agreement;
                                    correct_count += usize::from(accepted && score.correct);
                                    wrong_count += usize::from(accepted && !score.correct);
                                }
                                if wrong_count <= false_budget
                                    && best_stricter.is_none_or(|best| correct_count > best.0)
                                {
                                    best_stricter = Some((
                                        correct_count,
                                        wrong_count,
                                        spatial,
                                        margin,
                                        ratio,
                                        confidence,
                                        prior_agreement,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            eprintln!("  best-stricter-one-percent={best_stricter:?}");
            for spatial_limit in [2.0f32, 3.0, 4.0, 5.0, 6.0, 8.0] {
                let mut correct = 0usize;
                let mut wrong = 0usize;
                for score in &result.scores {
                    let accepted = score.best_distance <= NAUTILUS_MAX_DISTANCE
                        && score.margin >= NAUTILUS_MIN_ABSOLUTE_MARGIN
                        && score.ratio <= NAUTILUS_MAX_DISTANCE_RATIO
                        && score.selected_prediction_distance <= spatial_limit;
                    correct += usize::from(accepted && score.correct);
                    wrong += usize::from(accepted && !score.correct);
                }
                eprintln!(
                    "  spatial-limit={spatial_limit:.1}px accepted-correct={correct} accepted-wrong={wrong}"
                );
            }
        }
        assert_eq!(total_frames, 635, "lossless corpus frame count changed");
        assert!(stats[0].oracle_correspondences >= 1_000);
        // The production gate is deliberately precision-first. A wrong
        // accepted identity is materially worse than an ambiguous point that
        // remains withheld and is freshly seeded nearby.
        for result in &stats {
            assert!(
                result.production_accepted_wrong * 100 <= result.oracle_correspondences.max(1),
                "accepted false reconnections exceeded one percent: wrong={} oracle={}",
                result.production_accepted_wrong,
                result.oracle_correspondences,
            );
        }
    }

    #[test]
    fn clearing_cluster_tracker_releases_temporal_raw_state() {
        let width = 32usize;
        let height = 24usize;
        let raw = vec![512u16; width * height];
        let mut tracker = FourMotionOctrees::default();
        let _ = tracker.observe(&raw, width, height, 0, 0, None, true);
        assert!(tracker.previous.is_some());

        tracker.clear();

        assert!(tracker.previous.is_none());
        assert!(tracker.tracks.is_empty());
        assert!(tracker.nautilus_banks.is_empty());
        assert!(!tracker.canny_features);
        assert_eq!(tracker.generation, 0);
    }
}
