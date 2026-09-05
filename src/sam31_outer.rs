//! Native promptable SAM3.1 segmentation for the live RAW10 viewer.
//!
//! The live path presents each native-aspect RAW10 ROI independently to the
//! SAM3.1 backbone, then carries mask memory, object pointers, presence, and
//! occlusion state frame by frame. The temporally conditioned mask is the live
//! candidate; untouched RAW10 geometry and photometry remain its publication
//! gate. No Python, decoded image files, or repeated filmstrip adapter queries
//! exist in the live path. Offline diagnostics retain the legacy multi-adapter
//! filmstrip machinery for controlled comparisons.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
#[cfg(feature = "sam31")]
use std::thread;
#[cfg(feature = "sam31")]
use std::time::Instant;

pub use crate::geometry::Ellipse;
pub use crate::conic_solver::OuterContourScaleContext;
// Compatibility entry point for the existing offline tools.
#[allow(unused_imports)]
pub use crate::conic_solver::fit_trusted_arc_points;
pub use crate::outline_conic_segments::ContourFitEvidence as OuterMaskFitReview;
use crate::geometry::ellipse_coordinate;
use crate::conic_solver::{
    moments_ellipse, normalize_ellipse, plausible_ellipse, ellipse_support_summary, median,
};
#[cfg(test)]
use crate::conic_solver::{
    robust_contour_fit, NumpyPcg64, ConicArcConstraints, robust_ransac_ellipse,
};
use crate::outline_conic_segments::{
    native_component_contour, sample_closed_contour,
    deflattened_mask_fit_with_context, deflattened_mask_fit_with_noise,
};
#[cfg(test)]
use crate::outline_conic_segments::deflattened_mask_fit;

pub const HISTORY_FRAMES: usize = 5;
pub const FRAME_WIDTH: usize = 384;
pub const FRAME_HEIGHT: usize = 256;
const FILMSTRIP_WIDTH: usize = FRAME_WIDTH * HISTORY_FRAMES;
const FILMSTRIP_PIXELS: usize = FILMSTRIP_WIDTH * FRAME_HEIGHT;
// The corpus includes ordinary iris masks below the old 20,000-pixel floor
// (~20% of a model ROI). A 5% trial admitted tiny lid/pupil fragments; retain
// a conservative 12% cold-fit floor together with conic and RAW-edge checks.
const MIN_COMPONENT_AREA_FULL_RES: usize = FRAME_WIDTH * FRAME_HEIGHT * 12 / 100;
const MAX_COMPONENT_AREA_FULL_RES: usize = FRAME_WIDTH * FRAME_HEIGHT * 9 / 10;
const MAX_MASK_CANDIDATE_FITS: usize = 8;
const MIN_RAW_RING_SUPPORT_SCORE: f64 = 2.45;
const MIN_PUPIL_VOID_SUPPORT_SCORE: f64 = 2.05;
const MAX_UNPROMPTED_PUPIL_CENTER_OFFSET: f64 = 0.35;
const PUPIL_CONTOUR_MAX_GAP_NS: u64 = 1_000_000_000;
const PUPIL_CONTOUR_WINDOW_NS: u64 = 2_500_000_000;
pub const PUPIL_DISK_PROMPT: usize = 2;

fn enabled_env_flag(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |value| {
        !matches!(value.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off")
    })
}

pub fn default_model_path() -> PathBuf {
    let shared = PathBuf::from("data/models/sam31_semantic_video_shared_features_u8.pt");
    if shared.is_file() {
        shared
    } else {
        PathBuf::from("data/models/sam31_semantic_video_features_u8.pt")
    }
}

fn prompt_bundle_path(model: &Path) -> PathBuf {
    std::env::var_os("BUTTERCUP_SAM31_PROMPT_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| model.with_file_name("sam31_semantic_prompts_cuda_bf16.pt"))
}

fn tracker_bundle_path() -> PathBuf {
    std::env::var_os("BUTTERCUP_SAM31_TRACKER_BUNDLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/models/sam31_tracker_weights.pt"))
}

/// Cheap startup eligibility check, not a model/CUDA warmup. Explicit SAM
/// requests still report runtime errors instead of silently changing modes.
pub fn available_for_startup(model: &Path) -> bool {
    startup_assets_available(model, &prompt_bundle_path(model), &tracker_bundle_path())
}

fn startup_assets_available(model: &Path, prompts: &Path, tracker: &Path) -> bool {
    cfg!(feature = "sam31") && model.is_file() && prompts.is_file() && tracker.is_file()
}

/// Keep offline reports explicit about the otherwise process-local settings.
pub fn live_configuration() -> serde_json::Value {
    serde_json::json!({
        "preprocess": PreprocessRegime::configured_live().ok().map(PreprocessRegime::label),
        "legacy_outer_selection": legacy_outer_selection(),
        "minimum_outer_component_area_model_pixels": minimum_outer_component_area(),
        "semantic_pupil": enabled_env_flag("BUTTERCUP_SAM31_SEMANTIC_PUPIL", true),
        "pupil_first_qualified_query": enabled_env_flag("BUTTERCUP_SAM31_PUPIL_RANK_FIRST", false),
        "prefer_shared_image_features": enabled_env_flag("BUTTERCUP_SAM31_SHARED_FEATURE_PROMPT", true),
        "pupil_contour_history_window_ns": PUPIL_CONTOUR_WINDOW_NS,
        "pupil_contour_maximum_gap_ns": PUPIL_CONTOUR_MAX_GAP_NS,
        "prompt_bundle": std::env::var("BUTTERCUP_SAM31_PROMPT_BUNDLE").ok(),
        "pupil_diagnostics_contract": "stateless RAW component audit, not the selected semantic-pupil decision",
    })
}

// Reproduce the pre-corpus-tuning selector in offline A/B replays. This
// opt-in diagnostic also restores its first-RAW-valid detector selection.
fn legacy_outer_selection() -> bool {
    std::env::var_os("BUTTERCUP_SAM31_LEGACY_OUTER_SELECTION").is_some()
}

fn minimum_outer_component_area() -> usize {
    if legacy_outer_selection() { 20_000 } else { MIN_COMPONENT_AREA_FULL_RES }
}

pub const SEMANTIC_PROMPT_COUNT: usize = 6;
pub const OUTER_IRIS_PROMPT: usize = 0;
pub const DEFAULT_OUTER_IRIS_PROMPT_TEXT: &str =
    "the complete dark circular iris disk surrounding the pupil, excluding the eyelids and sclera";
pub const SEMANTIC_PROMPT_LABELS: [&str; SEMANTIC_PROMPT_COUNT] = [
    "OUTER IRIS DISK",
    "IRIS ANNULUS",
    "PUPIL DISK",
    "VISIBLE SCLERA",
    "UPPER EYELID",
    "LOWER EYELID",
];

pub fn semantic_prompt_label(index: usize) -> &'static str {
    SEMANTIC_PROMPT_LABELS
        .get(index)
        .copied()
        .unwrap_or("UNKNOWN SAM PROMPT")
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Target {
    #[default]
    OuterLimbus,
    InnerPupilVoid,
    /// Preserve the outer-limbus product even when the optional pupil prompt
    /// and RAW validation fail. The two text-conditioned products share the
    /// image encoder when the loaded graph supports it, but neither selector
    /// changes the other's availability.
    OuterLimbusAndInnerPupilVoid,
}

impl Target {
    pub fn label(self) -> &'static str {
        match self {
            Self::OuterLimbus => "outer-limbus",
            Self::InnerPupilVoid => "inner-pupil-void",
            Self::OuterLimbusAndInnerPupilVoid => "outer-limbus+inner-pupil-void",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RawFrame {
    pub eye_index: usize,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub width: usize,
    pub height: usize,
    /// Generic current-eye translation anchor used only to register a delayed
    /// asynchronous result into a newer frame. It is not a pupil prompt and
    /// must not choose among pupil components.
    pub registration_anchor: Option<(f64, f64)>,
    /// Optional target-owned seed for selecting an inner-pupil component.
    /// The live SAM path intentionally leaves this `None`, so its
    /// pupil comes from the pupil prompt, constrained by its own SAM outer
    /// ellipse and untouched source-exposure RAW evidence.
    pub pupil_component_seed: Option<(f64, f64)>,
    pub pixels: Arc<Vec<u16>>,
}

/// A sensor-fixed RAW10 outlier confirmed across multiple source frames.
/// Coordinates are expressed in the uncropped sensor space so ROI motion does
/// not make a defective photosite appear to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PersistentHotPixel {
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub detected_frames: usize,
    pub visible_frames: usize,
    pub peak_raw10: u16,
}

fn same_cfa_neighbor_median(frame: &RawFrame, x: usize, y: usize) -> Option<(u16, u16)> {
    // This sensor has a 4x4 Quad-Bayer period. Sampling at +/-4 preserves both
    // colour and position within each same-colour 2x2 photosite group.
    const OFFSETS: [(isize, isize); 8] = [
        (-4, -4),
        (0, -4),
        (4, -4),
        (-4, 0),
        (4, 0),
        (-4, 4),
        (0, 4),
        (4, 4),
    ];
    let mut values = [0u16; 8];
    for (index, (dx, dy)) in OFFSETS.iter().copied().enumerate() {
        let xx = x as isize + dx;
        let yy = y as isize + dy;
        if xx < 0 || yy < 0 || xx >= frame.width as isize || yy >= frame.height as isize {
            return None;
        }
        values[index] = frame.pixels[yy as usize * frame.width + xx as usize];
    }
    values.sort_unstable();
    let median = ((values[3] as u32 + values[4] as u32) / 2) as u16;
    let mut deviations = values.map(|value| value.abs_diff(median));
    deviations.sort_unstable();
    let mad = ((deviations[3] as u32 + deviations[4] as u32) / 2) as u16;
    Some((median, mad))
}

fn raw10_hot_pixel_candidates(frame: &RawFrame) -> Vec<(u32, u32, u16)> {
    let mut candidates = Vec::new();
    for y in 4..frame.height.saturating_sub(4) {
        for x in 4..frame.width.saturating_sub(4) {
            let value = frame.pixels[y * frame.width + x].min(1023);
            let Some((median, mad)) = same_cfa_neighbor_median(frame, x, y) else {
                continue;
            };
            let delta = value.saturating_sub(median);
            let robust_threshold = 72u16.saturating_add(mad.saturating_mul(10));
            if value >= 384 && delta >= robust_threshold {
                candidates.push((frame.sensor_x + x as u32, frame.sensor_y + y as u32, value));
            }
        }
    }
    candidates
}

/// Locate persistent isolated bright photosites without mistaking a moving
/// glint or anatomical edge for a sensor defect.
pub fn persistent_raw10_hot_pixels(frames: &[Arc<RawFrame>]) -> Vec<PersistentHotPixel> {
    let mut detections = HashMap::<(u32, u32), (usize, u16)>::new();
    for frame in frames {
        for (sensor_x, sensor_y, value) in raw10_hot_pixel_candidates(frame) {
            let entry = detections.entry((sensor_x, sensor_y)).or_default();
            entry.0 += 1;
            entry.1 = entry.1.max(value);
        }
    }
    let mut confirmed = detections
        .into_iter()
        .filter_map(|((sensor_x, sensor_y), (detected_frames, peak_raw10))| {
            let visible_frames = frames
                .iter()
                .filter(|frame| {
                    sensor_x >= frame.sensor_x + 4
                        && sensor_y >= frame.sensor_y + 4
                        && sensor_x + 4 < frame.sensor_x + frame.width as u32
                        && sensor_y + 4 < frame.sensor_y + frame.height as u32
                })
                .count();
            let minimum = if peak_raw10 >= 1008 {
                3.max((visible_frames + 3) / 4)
            } else {
                4.max((visible_frames * 3 + 3) / 4)
            };
            (visible_frames >= 3 && detected_frames >= minimum).then_some(PersistentHotPixel {
                sensor_x,
                sensor_y,
                detected_frames,
                visible_frames,
                peak_raw10,
            })
        })
        .collect::<Vec<_>>();
    confirmed.sort_unstable_by_key(|pixel| (pixel.sensor_y, pixel.sensor_x));
    confirmed
}

fn hot_pixel_check_enabled() -> bool {
    std::env::var_os("BUTTERCUP_RAW10_HOT_PIXEL_CHECK").is_some()
}

fn corrected_hot_pixel_frames(
    frames: &[Arc<RawFrame>],
    hot_pixels: &[PersistentHotPixel],
) -> Vec<Arc<RawFrame>> {
    if hot_pixels.is_empty() {
        return frames.to_vec();
    }
    let hot_pixels = hot_pixels
        .iter()
        .map(|pixel| (pixel.sensor_x, pixel.sensor_y))
        .collect::<HashSet<_>>();
    frames
        .iter()
        .map(|frame| {
            let mut corrected = (**frame).clone();
            let mut pixels = (*frame.pixels).clone();
            for &(sensor_x, sensor_y) in &hot_pixels {
                let Some(x) = sensor_x
                    .checked_sub(frame.sensor_x)
                    .map(|value| value as usize)
                else {
                    continue;
                };
                let Some(y) = sensor_y
                    .checked_sub(frame.sensor_y)
                    .map(|value| value as usize)
                else {
                    continue;
                };
                if x >= frame.width || y >= frame.height {
                    continue;
                }
                if let Some((median, _)) = same_cfa_neighbor_median(frame, x, y) {
                    pixels[y * frame.width + x] = median;
                }
            }
            corrected.pixels = Arc::new(pixels);
            Arc::new(corrected)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalAdapter {
    QuadRgb,
    RawLuma,
    LogChroma,
}

impl ProposalAdapter {
    pub fn label(self) -> &'static str {
        match self {
            Self::QuadRgb => "QUAD-RGB",
            Self::RawLuma => "RAW-LUMA",
            Self::LogChroma => "LOG-CHROMA",
        }
    }
}

/// Explicit RAW10-to-model preprocessing used by offline temporal trials.
///
/// These regimes never alter the retained native source frame. They only
/// alter the transient U8 tensor presented to the currently available SAM3.1
/// detector graph, so temporal comparisons can distinguish photometric input
/// conditioning from tracker-memory behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PreprocessRegime {
    BalancedQuadRgb,
    ShadowBoost,
    GentleShadowLift,
    /// Default for native temporal conditioning. Across the labeled low-light
    /// and well-lit sequences this suppressed Quad-Bayer/high-frequency
    /// distractors without the geometry-changing failures of stronger tone
    /// transforms.
    #[default]
    MildBlur,
    /// Diagnostic-only center exclusion: use the normal mild-blur adapter,
    /// locate the darkest compact central region, then cover it with a
    /// saturated pink box before SAM inference. The retained RAW frame is
    /// never modified.
    PinkCenterMask,
    Unsharp,
    GentleUnsharp,
    IlluminationNormalizedAlbedo,
    PartialAlbedo,
    RawLuma,
    LogChroma,
    StrongLowPass,
    HighPassLuma,
    CannyLumaOverlay,
    CannyEdgeOnly,
    NormalizedChromaticity,
    DarkFloor,
}

impl PreprocessRegime {
    fn configured_live() -> Result<Self, String> {
        match std::env::var("BUTTERCUP_SAM31_PREPROCESS") {
            Ok(label) => Self::ALL.into_iter().find(|regime| regime.label() == label)
                .ok_or_else(|| format!("unknown SAM31 preprocessing regime: {label}")),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(error) => Err(format!("SAM31 preprocessing setting: {error}")),
        }
    }

    pub const ALL: [Self; 17] = [
        Self::BalancedQuadRgb,
        Self::ShadowBoost,
        Self::GentleShadowLift,
        Self::MildBlur,
        Self::PinkCenterMask,
        Self::Unsharp,
        Self::GentleUnsharp,
        Self::IlluminationNormalizedAlbedo,
        Self::PartialAlbedo,
        Self::RawLuma,
        Self::LogChroma,
        Self::StrongLowPass,
        Self::HighPassLuma,
        Self::CannyLumaOverlay,
        Self::CannyEdgeOnly,
        Self::NormalizedChromaticity,
        Self::DarkFloor,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::BalancedQuadRgb => "balanced-quad-rgb",
            Self::ShadowBoost => "shadow-boost",
            Self::GentleShadowLift => "gentle-shadow-lift",
            Self::MildBlur => "mild-blur",
            Self::PinkCenterMask => "pink-center-mask",
            Self::Unsharp => "unsharp",
            Self::GentleUnsharp => "gentle-unsharp",
            Self::IlluminationNormalizedAlbedo => "illumination-normalized-albedo",
            Self::PartialAlbedo => "partial-albedo",
            Self::RawLuma => "raw-luma",
            Self::LogChroma => "log-chroma",
            Self::StrongLowPass => "strong-low-pass",
            Self::HighPassLuma => "high-pass-luma",
            Self::CannyLumaOverlay => "canny-luma-overlay",
            Self::CannyEdgeOnly => "canny-edge-only",
            Self::NormalizedChromaticity => "normalized-chromaticity",
            Self::DarkFloor => "dark-floor",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|regime| regime.label() == value.trim())
    }
}

#[derive(Clone, Debug)]
pub struct ProposalMask {
    pub query: usize,
    pub score: f32,
    /// One binary, low-resolution mask for the latest (rightmost) temporal
    /// tile. The renderer scales this back onto the exact native RAW source
    /// frame retained by `ProposalMasks`.
    pub pixels: Arc<Vec<u8>>,
    /// Linear indices of the four-connected boundary, computed once on the
    /// inference worker so a high-refresh diagnostic view never rescans every
    /// proposal pixel on every display frame.
    pub boundary_pixels: Arc<Vec<u32>>,
}

#[derive(Clone, Debug)]
pub struct AdapterProposalMasks {
    pub adapter: ProposalAdapter,
    pub width: usize,
    pub height: usize,
    pub selected_query: Option<usize>,
    pub masks: Vec<ProposalMask>,
}

#[derive(Clone, Debug)]
pub struct SemanticProposalMasks {
    pub prompt_index: usize,
    pub width: usize,
    pub height: usize,
    pub selected_query: Option<usize>,
    pub masks: Vec<ProposalMask>,
}

#[derive(Clone, Debug, Default)]
pub struct ProposalMasks {
    /// Caller-owned tracking session generation. Results from a previous
    /// replay/live session must not be rendered into the current one.
    pub tracking_epoch: u64,
    /// Prompt-bundle generation which produced this answer.  The renderer
    /// must never present a proposal from an older interactive prompt.
    pub prompt_generation: u64,
    pub eye_index: usize,
    pub source_sequence: u64,
    pub source_timestamp_ns: u64,
    pub source_sensor_origin: (u32, u32),
    pub source_width: usize,
    pub source_height: usize,
    pub source_raw: Arc<Vec<u16>>,
    /// Candidate objects returned for one explicitly named semantic question.
    /// Multiple candidates are answers to the same question, not additional
    /// questions. The prompt index follows `SEMANTIC_PROMPT_LABELS`.
    pub semantic: Option<SemanticProposalMasks>,
    /// De-flat-tired contour and fit from the selected Quad-RGB answer to the
    /// mandatory OUTER IRIS DISK question.  This remains available even when
    /// later adapter-consensus or RAW photometric gates reject the batch.
    pub outer_fit: Option<OuterMaskFitReview>,
    /// Dark pupil void fitted from the untouched RAW source inside the exact
    /// de-flat-tired review ellipse above. This is retained with the proposal
    /// so consumers can disambiguate the two projected normal branches
    /// without borrowing a pupil or limbus from another frame/algorithm.
    pub inner_pupil_fit: Option<PupilVoidFitReview>,
    pub adapters: Vec<AdapterProposalMasks>,
}

impl ProposalMasks {
    pub fn adapter(&self, adapter: ProposalAdapter) -> Option<&AdapterProposalMasks> {
        self.adapters
            .iter()
            .find(|candidate| candidate.adapter == adapter)
    }
}

/// One synchronous, offline prompt-engineering run over a fixed native RAW10
/// history.  Unlike the live mailbox API, this retains every requested
/// semantic answer so combinations of one to three follow-on questions can be
/// compared on the exact same target frame.
#[derive(Clone, Debug)]
pub struct OfflineSemanticSuite {
    pub source_width: usize,
    pub source_height: usize,
    pub source_raw: Arc<Vec<u16>>,
    pub outer_fit: Option<OuterMaskFitReview>,
    pub passes: Vec<SemanticProposalMasks>,
    pub video_feature_shapes: Option<VideoFeatureShapes>,
    pub elapsed_ms: u64,
}

#[derive(Clone, Debug)]
pub struct VideoFeatureShapes {
    pub pyramid: [Vec<i64>; 3],
    pub decoder_queries: Vec<i64>,
}

/// Adjacent-frame agreement from the feature tensors exported by the native
/// SAM3.1 graph. These values are measured before any mask post-processing.
#[derive(Clone, Debug)]
pub struct VideoFeatureTransition {
    pub from_frame: usize,
    pub to_frame: usize,
    pub pyramid_cosine: [f64; 3],
    pub pyramid_normalized_rms_change: [f64; 3],
    pub decoder_query_cosine: f64,
    pub decoder_query_normalized_rms_change: f64,
    pub source_selected_query: Option<usize>,
    pub target_detector_query: Option<usize>,
    pub matched_query: Option<usize>,
    pub matched_query_cosine: Option<f64>,
    pub matched_mask_iou_sensor: Option<f64>,
    pub mask_memory_cosine: Option<f64>,
    pub mask_memory_normalized_rms_change: Option<f64>,
    pub temporal_conditioned_cosine: Option<f64>,
    pub temporal_conditioned_normalized_rms_change: Option<f64>,
    pub temporal_update_cosine: Option<f64>,
    pub temporal_update_normalized_rms_change: Option<f64>,
    pub tracker_object_score_logit: Option<f64>,
    pub tracker_object_present: Option<bool>,
    pub tracker_reconditioned_from_detector: bool,
    pub detector_recondition_area_fraction: Option<f64>,
    pub tracker_selected_iou_score: Option<f64>,
    pub tracker_mask_iou_sensor: Option<f64>,
    pub tracker_vs_detector_mask_iou: Option<f64>,
    pub tracker_area_fraction: Option<f64>,
    pub tracker_equivalent_radius_px: Option<f64>,
    pub tracker_radius_ratio_from_prior: Option<f64>,
    pub tracker_centroid_motion_sensor_px: Option<f64>,
}

/// A true frame-by-frame feature run. Unlike the legacy detector filmstrip,
/// every source frame is presented to SAM3.1 at its native ROI aspect ratio.
#[derive(Clone, Debug)]
pub struct OfflineVideoFeatureSequence {
    pub frame_count: usize,
    pub feature_shapes: VideoFeatureShapes,
    pub mask_memory_shape: Option<Vec<i64>>,
    pub mask_memory_position_shape: Option<Vec<i64>>,
    pub temporal_conditioned_shape: Option<Vec<i64>>,
    pub transitions: Vec<VideoFeatureTransition>,
    pub review_frames: Vec<OfflineVideoReviewFrame>,
    pub elapsed_ms: u64,
}

/// Lossless source plus the exact mask admitted to temporal memory. This is
/// retained only by the explicit offline diagnostic so its animated affine
/// review cannot accidentally substitute a detector mask for tracker output.
#[derive(Clone, Debug)]
pub struct OfflineVideoReviewFrame {
    pub source: Arc<RawFrame>,
    pub mask: Option<Arc<Vec<u8>>>,
    pub mask_width: usize,
    pub mask_height: usize,
    pub fit: Option<OuterMaskFitReview>,
    /// Best frame-local contour interpretation before the distributed and
    /// temporal publication gates. Rejected frames retain this solely so the
    /// review can display which samples were tentatively usable or censored.
    pub contour_hypothesis: Option<OuterMaskFitReview>,
    /// Native-coordinate outline of the largest tracker-mask component. If no
    /// ellipse hypothesis survives, review renders these samples as ignored
    /// rather than presenting an unexplained empty panel.
    pub raw_contour_points: Arc<Vec<(f64, f64)>>,
    /// Confirmed sensor defects visible in this ROI, in native ROI pixels.
    /// The source remains untouched; only transient model input is corrected.
    pub hot_pixels: Arc<Vec<(usize, usize)>>,
    pub propagated: bool,
}

/// Offline-only inspection before any contour/ellipse rejection. This keeps
/// failed detector outlines available for testing alternate arc combiners.
/// It does not change live selection, tracking memory, or publication gates.
#[cfg(feature = "sam31")]
pub fn export_native_outline_sequence(
    model: &Path,
    frames: &[Arc<RawFrame>],
) -> Result<serde_json::Value, String> {
    runtime::export_native_outline_sequence(model, frames)
}

#[cfg(not(feature = "sam31"))]
pub fn export_native_outline_sequence(
    _model: &Path,
    _frames: &[Arc<RawFrame>],
) -> Result<serde_json::Value, String> {
    Err("SAM31 support is not compiled in; rebuild with --features sam31".to_string())
}

/// Run an arbitrary-size prompt bundle synchronously for an offline trial.
/// The live viewer deliberately retains its bounded asynchronous API; only the
/// temporary corpus experiment uses this entry point.
#[cfg(feature = "sam31")]
pub fn run_offline_semantic_suite(
    model: impl AsRef<Path>,
    outer_prompt_bundle: impl AsRef<Path>,
    prompt_bundle: impl AsRef<Path>,
    prompt_count: usize,
    frames: &[Arc<RawFrame>],
    prompt_indices: &[usize],
) -> Result<OfflineSemanticSuite, String> {
    run_offline_semantic_suite_with_regime(
        model,
        outer_prompt_bundle,
        prompt_bundle,
        prompt_count,
        frames,
        prompt_indices,
        PreprocessRegime::BalancedQuadRgb,
    )
}

#[cfg(feature = "sam31")]
pub fn run_offline_semantic_suite_with_regime(
    model: impl AsRef<Path>,
    outer_prompt_bundle: impl AsRef<Path>,
    prompt_bundle: impl AsRef<Path>,
    prompt_count: usize,
    frames: &[Arc<RawFrame>],
    prompt_indices: &[usize],
    regime: PreprocessRegime,
) -> Result<OfflineSemanticSuite, String> {
    runtime::offline_semantic_suite(
        model.as_ref(),
        outer_prompt_bundle.as_ref(),
        prompt_bundle.as_ref(),
        prompt_count,
        frames,
        prompt_indices,
        regime,
    )
}

#[cfg(feature = "sam31")]
pub fn run_offline_video_feature_sequence(
    model: impl AsRef<Path>,
    prompt_bundle: impl AsRef<Path>,
    frames: &[Arc<RawFrame>],
    regime: PreprocessRegime,
) -> Result<OfflineVideoFeatureSequence, String> {
    runtime::offline_video_feature_sequence(model.as_ref(), prompt_bundle.as_ref(), frames, regime)
}

#[cfg(not(feature = "sam31"))]
pub fn run_offline_video_feature_sequence(
    _model: impl AsRef<Path>,
    _prompt_bundle: impl AsRef<Path>,
    _frames: &[Arc<RawFrame>],
    _regime: PreprocessRegime,
) -> Result<OfflineVideoFeatureSequence, String> {
    Err("SAM31 support is not compiled in; rebuild with --features sam31".to_string())
}

#[cfg(not(feature = "sam31"))]
pub fn run_offline_semantic_suite(
    _model: impl AsRef<Path>,
    _outer_prompt_bundle: impl AsRef<Path>,
    _prompt_bundle: impl AsRef<Path>,
    _prompt_count: usize,
    _frames: &[Arc<RawFrame>],
    _prompt_indices: &[usize],
) -> Result<OfflineSemanticSuite, String> {
    Err("SAM31 support is not compiled in; rebuild with --features sam31".to_string())
}

#[cfg(not(feature = "sam31"))]
pub fn run_offline_semantic_suite_with_regime(
    _model: impl AsRef<Path>,
    _outer_prompt_bundle: impl AsRef<Path>,
    _prompt_bundle: impl AsRef<Path>,
    _prompt_count: usize,
    _frames: &[Arc<RawFrame>],
    _prompt_indices: &[usize],
    _regime: PreprocessRegime,
) -> Result<OfflineSemanticSuite, String> {
    Err("SAM31 support is not compiled in; rebuild with --features sam31".to_string())
}

#[derive(Clone, Debug)]
pub struct OuterResult {
    /// Caller-owned tracking session generation which produced this result.
    pub tracking_epoch: u64,
    /// Prompt-bundle generation which produced this accepted geometry.
    pub prompt_generation: u64,
    pub target: Target,
    pub eye_index: usize,
    pub source_sequence: u64,
    pub source_timestamp_ns: u64,
    pub source_sensor_origin: (u32, u32),
    pub source_registration_anchor_sensor: Option<(f64, f64)>,
    /// Target's primary ellipse in full-sensor coordinates; radii remain RAW
    /// pixels. Combined requests keep the outer limbus primary so a missing
    /// optional pupil cannot suppress the iris product.
    pub sensor_ellipse: Ellipse,
    /// The outer-iris prompt's independently validated limbus guide. This is
    /// identical to `sensor_ellipse` for outer and combined requests and
    /// remains available as the validation gate for the pupil-only target.
    pub sensor_outer_ellipse: Ellipse,
    /// Independently inferred inner-pupil ellipse, when requested and
    /// supported by the untouched RAW frame. It is a rough-center product;
    /// consumers must not publish it as their final pupil boundary.
    pub sensor_pupil_ellipse: Option<Ellipse>,
    pub agreeing_adapters: usize,
    pub quality: f64,
    /// Photometric agreement of the proposed limbus with untouched RAW10.
    /// This rejects mutually consistent SAM masks that merely cover skin or
    /// another broad dark region when the eye has left a stale ROI.
    pub raw_ring_support_score: f64,
    pub raw_ring_support_points: usize,
    pub raw_ring_positive_fraction: f64,
    pub raw_ring_strong_sectors: usize,
    /// Target-specific RAW support. For the outer mode this equals the ring
    /// support above; for pupil mode it scores the dark-to-iris transition at
    /// the returned inner ellipse.
    pub raw_target_support_score: f64,
    pub raw_target_support_points: usize,
    pub raw_target_positive_fraction: f64,
    pub raw_target_strong_sectors: usize,
    pub elapsed_ms: u64,
    /// True when the published candidate came from SAM3.1's temporally
    /// conditioned video-memory path rather than independent filmstrip
    /// adapter queries.
    pub video_tracked: bool,
    /// Every outer-iris candidate for each RAW adapter, retained strictly for
    /// the tracker diagnostics. The semantic answer selected by F is stored
    /// separately, so candidate objects are never mislabeled as questions.
    /// These masks never participate in tracking after
    /// the normal consensus/RAW-support gates have made their decision.
    pub proposal_masks: Arc<ProposalMasks>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawRingSupport {
    pub score: f64,
    pub points: usize,
    pub positive_fraction: f64,
    pub strong_sectors: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PupilVoidFitReview {
    pub ellipse: Ellipse,
    pub raw_support: RawRingSupport,
}

/// Resolve independently validated products after image and prompt inference.
/// The combined request always returns the independently valid outer limbus;
/// pupil failure is represented only by a missing optional center product.
fn select_target_products(
    target: Target,
    outer_ellipse: Ellipse,
    outer_support: RawRingSupport,
    pupil_fit: Option<(Ellipse, RawRingSupport)>,
) -> Result<(Ellipse, RawRingSupport, Option<Ellipse>), String> {
    match target {
        Target::OuterLimbus => Ok((outer_ellipse, outer_support, None)),
        Target::InnerPupilVoid => pupil_fit
            .map(|(ellipse, support)| (ellipse, support, Some(ellipse)))
            .ok_or_else(|| {
                "SAM31 inner pupil had no proposal with consistent RAW meridian support"
                    .to_string()
            }),
        Target::OuterLimbusAndInnerPupilVoid => Ok((
            outer_ellipse,
            // Combined requests remain outer-primary in both geometry and
            // diagnostics; the optional center cannot mutate G's metrics.
            outer_support,
            pupil_fit.map(|(ellipse, _)| ellipse),
        )),
    }
}

#[derive(Clone, Debug)]
pub struct StatusSnapshot {
    pub state: &'static str,
    pub detail: String,
    pub accepted_batches: u64,
    pub dropped_batches: u64,
    pub completed_batches: u64,
    pub last_elapsed_ms: Option<u64>,
}

impl Default for StatusSnapshot {
    fn default() -> Self {
        Self {
            state: "idle",
            detail: "waiting for the first RAW10 video frame".to_string(),
            accepted_batches: 0,
            dropped_batches: 0,
            completed_batches: 0,
            last_elapsed_ms: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitOutcome {
    Accepted,
    DroppedBusy,
    Invalid,
}

/// Select the only mask that may enter live temporal memory for this frame.
///
/// A healthy propagated mask keeps the normal video path. If propagation no
/// longer yields plausible limbus geometry, the current frame's detector mask
/// for the carried decoder-query identity becomes a new conditioning frame
/// immediately. When neither is plausible, retaining the last admitted
/// history is safer than encoding an artificial all-negative mask into it.
#[cfg(any(feature = "sam31", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveTemporalUpdate {
    Propagate,
    ConditionFromDetector(usize),
    HoldLastConditioning,
}

const LIVE_TRACKER_MIN_PRIOR_MASK_IOU: f64 = 0.68;
const LIVE_RECOVERY_MIN_QUERY_COSINE: f64 = 0.70;
const LIVE_TRACKER_MAX_HOLD_MISSES: u8 = 3;
const LIVE_TRACKER_MAX_TIMESTAMP_GAP_NS: u64 = 900_000_000;

#[cfg(any(feature = "sam31", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LiveTrackerInput {
    tracking_epoch: u64,
    prompt_generation: u64,
    sequence: u64,
    timestamp_ns: u64,
    sensor_origin: (u32, u32),
    width: usize,
    height: usize,
}

#[cfg(any(feature = "sam31", test))]
fn live_rois_overlap(first: LiveTrackerInput, second: LiveTrackerInput) -> bool {
    let first_right = u64::from(first.sensor_origin.0).saturating_add(first.width as u64);
    let first_bottom = u64::from(first.sensor_origin.1).saturating_add(first.height as u64);
    let second_right = u64::from(second.sensor_origin.0).saturating_add(second.width as u64);
    let second_bottom = u64::from(second.sensor_origin.1).saturating_add(second.height as u64);
    u64::from(first.sensor_origin.0) < second_right
        && u64::from(second.sensor_origin.0) < first_right
        && u64::from(first.sensor_origin.1) < second_bottom
        && u64::from(second.sensor_origin.1) < first_bottom
}

#[cfg(any(feature = "sam31", test))]
fn live_tracker_requires_reset(
    previous: Option<LiveTrackerInput>,
    current: LiveTrackerInput,
) -> bool {
    let Some(previous) = previous else {
        return false;
    };
    let roi_changed = previous.sensor_origin != current.sensor_origin
        || previous.width != current.width
        || previous.height != current.height;
    previous.tracking_epoch != current.tracking_epoch
        || previous.prompt_generation != current.prompt_generation
        || current.sequence <= previous.sequence
        || current.timestamp_ns < previous.timestamp_ns
        || current.timestamp_ns.saturating_sub(previous.timestamp_ns)
            > LIVE_TRACKER_MAX_TIMESTAMP_GAP_NS
        || roi_changed
        || !live_rois_overlap(previous, current)
}

#[cfg(any(feature = "sam31", test))]
fn pupil_history_survives_roi_relocation(
    previous: Option<LiveTrackerInput>,
    current: LiveTrackerInput,
) -> bool {
    previous.is_some_and(|previous| {
        previous.tracking_epoch == current.tracking_epoch
            && previous.prompt_generation == current.prompt_generation
            && current.sequence > previous.sequence
            && current.timestamp_ns > previous.timestamp_ns
            && current.timestamp_ns - previous.timestamp_ns <= LIVE_TRACKER_MAX_TIMESTAMP_GAP_NS
            && previous.width == current.width
            && previous.height == current.height
            && live_rois_overlap(previous, current)
    })
}

#[cfg(any(feature = "sam31", test))]
fn live_detector_candidate_is_plausible(
    score: f32,
    area_fraction: Option<f64>,
    has_plausible_fit: bool,
) -> bool {
    score.is_finite()
        && area_fraction.is_some_and(|area| area.is_finite() && (0.05..=0.50).contains(&area))
        && has_plausible_fit
}

#[cfg(any(feature = "sam31", test))]
fn live_detector_raw_gate_passes(support: RawRingSupport) -> bool {
    support.score.is_finite() && support.score >= MIN_RAW_RING_SUPPORT_SCORE
}

#[cfg(any(feature = "sam31", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LiveMemorySource {
    Propagation,
    Detector,
}

#[cfg(any(feature = "sam31", test))]
fn live_memory_commit_allowed(source: LiveMemorySource, support: RawRingSupport) -> bool {
    matches!(source, LiveMemorySource::Propagation) || live_detector_raw_gate_passes(support)
}

#[cfg(any(feature = "sam31", test))]
fn live_committed_frame_counts_as_miss(source: LiveMemorySource, support: RawRingSupport) -> bool {
    matches!(source, LiveMemorySource::Propagation) && !live_detector_raw_gate_passes(support)
}

#[cfg(any(feature = "sam31", test))]
fn next_live_hold_miss(consecutive_misses: u8) -> (u8, bool) {
    let next = consecutive_misses.saturating_add(1);
    (next, next >= LIVE_TRACKER_MAX_HOLD_MISSES)
}

#[cfg(any(feature = "sam31", test))]
fn ranked_finite_query_indices(scores: &[f32]) -> Vec<usize> {
    let mut ranked = scores
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.into_iter().map(|(query, _)| query).collect()
}

#[cfg(any(feature = "sam31", test))]
fn live_propagation_is_healthy(
    positive_object_score: bool,
    area_fraction: Option<f64>,
    has_plausible_fit: bool,
    history_exists: bool,
    prior_mask_iou: Option<f64>,
) -> bool {
    positive_object_score
        && area_fraction.is_some_and(|area| (0.05..=0.50).contains(&area))
        && has_plausible_fit
        && (!history_exists
            || prior_mask_iou.is_some_and(|iou| iou >= LIVE_TRACKER_MIN_PRIOR_MASK_IOU))
}

#[cfg(any(feature = "sam31", test))]
fn choose_live_recovery_queries(
    history_empty: bool,
    ranked_bootstrap_queries: &[usize],
    matched_identity_query: Option<(usize, f64)>,
) -> Vec<usize> {
    if history_empty {
        ranked_bootstrap_queries.to_vec()
    } else {
        matched_identity_query
            .filter(|matched| matched.1 >= LIVE_RECOVERY_MIN_QUERY_COSINE)
            .map(|matched| matched.0)
            .into_iter()
            .collect()
    }
}

#[cfg(any(feature = "sam31", test))]
fn choose_live_temporal_update(
    tracker_healthy: bool,
    plausible_current_detector_query: Option<usize>,
) -> LiveTemporalUpdate {
    if tracker_healthy {
        LiveTemporalUpdate::Propagate
    } else if let Some(query) = plausible_current_detector_query {
        LiveTemporalUpdate::ConditionFromDetector(query)
    } else {
        LiveTemporalUpdate::HoldLastConditioning
    }
}

struct Batch {
    target: Target,
    semantic_prompt: usize,
    prompt_generation: u64,
    tracking_epoch: u64,
    eye_index: usize,
    frames: Vec<Arc<RawFrame>>,
}

enum WorkerRequest {
    Batch(Batch),
    ReloadPrompts(PathBuf),
}

pub struct Client {
    request: Option<SyncSender<WorkerRequest>>,
    results: Receiver<OuterResult>,
    proposal_masks: Receiver<Arc<ProposalMasks>>,
    status: Arc<Mutex<StatusSnapshot>>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Client {
    pub fn start(model: impl AsRef<Path>) -> Result<Self, String> {
        Self::start_with_prompt_bundle(model, None::<&Path>)
    }

    pub fn start_with_prompt_bundle(
        model: impl AsRef<Path>,
        prompt_bundle_override: Option<impl AsRef<Path>>,
    ) -> Result<Self, String> {
        PreprocessRegime::configured_live()?;
        let model = model.as_ref().to_path_buf();
        if !model.is_file() {
            return Err(format!(
                "SAM31 promptable model not found: {}",
                model.display()
            ));
        }
        let prompt_bundle = prompt_bundle_override
            .map(|path| path.as_ref().to_path_buf())
            .unwrap_or_else(|| prompt_bundle_path(&model));
        if !prompt_bundle.is_file() {
            return Err(format!(
                "SAM31 semantic prompt bundle not found: {}",
                prompt_bundle.display()
            ));
        }
        // The live tracker currently submits only anatomical subject-right.
        // A rendezvous channel deliberately keeps no FIFO backlog: while the
        // model is busy, stale frames are dropped and the first current frame
        // offered after completion wins. A two-batch FIFO
        // at 42 Hz made every displayed fit roughly three inference periods
        // old even though each individual inference was healthy.
        let (request_tx, request_rx) = sync_channel(0);
        let (result_tx, result_rx) = sync_channel(4);
        // Diagnostic masks are independently published even when anatomical
        // consensus rejects the batch. A capacity of one plus non-blocking
        // publication always favors a recent diagnostic without allowing the
        // renderer to back-pressure CUDA inference.
        let (proposal_tx, proposal_rx) = sync_channel(1);
        let status = Arc::new(Mutex::new(StatusSnapshot::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = start_worker(
            model,
            prompt_bundle,
            request_rx,
            result_tx,
            proposal_tx,
            Arc::clone(&status),
            Arc::clone(&stop),
        )?;
        Ok(Self {
            request: Some(request_tx),
            results: result_rx,
            proposal_masks: proposal_rx,
            status,
            stop,
            worker: Some(worker),
        })
    }

    pub fn submit_history(
        &self,
        history: &VecDeque<Arc<RawFrame>>,
        target: Target,
        semantic_prompt: usize,
        prompt_generation: u64,
        tracking_epoch: u64,
    ) -> SubmitOutcome {
        if history.is_empty() {
            return SubmitOutcome::Invalid;
        }
        let frames = history
            .iter()
            .rev()
            .take(1)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>();
        let eye_index = frames
            .last()
            .map(|frame| frame.eye_index)
            .unwrap_or(usize::MAX);
        let valid = eye_index < 2
            && frames.len() == 1
            && frames.iter().all(|frame| {
                frame.eye_index == eye_index
                    && frame.width >= 4
                    && frame.height >= 4
                    && frame.width * FRAME_HEIGHT == frame.height * FRAME_WIDTH
                    && frame.pixels.len() == frame.width * frame.height
            });
        if !valid {
            if let Ok(mut status) = self.status.lock() {
                status.state = "error";
                status.detail = format!(
                    "SAM31 video tracking requires one current RAW frame with {}:{} aspect ratio from one eye",
                    FRAME_WIDTH, FRAME_HEIGHT
                );
            }
            return SubmitOutcome::Invalid;
        }
        let Some(request) = self.request.as_ref() else {
            return SubmitOutcome::Invalid;
        };
        match request.try_send(WorkerRequest::Batch(Batch {
            target,
            semantic_prompt: semantic_prompt.min(SEMANTIC_PROMPT_COUNT - 1),
            prompt_generation,
            tracking_epoch,
            eye_index,
            frames,
        })) {
            Ok(()) => {
                if let Ok(mut status) = self.status.lock() {
                    status.accepted_batches = status.accepted_batches.saturating_add(1);
                    if status.state == "idle" {
                        status.state = "queued";
                        status.detail = format!("first streaming {} query queued", target.label());
                    }
                }
                SubmitOutcome::Accepted
            }
            Err(TrySendError::Full(_)) => {
                if let Ok(mut status) = self.status.lock() {
                    status.dropped_batches = status.dropped_batches.saturating_add(1);
                }
                SubmitOutcome::DroppedBusy
            }
            Err(TrySendError::Disconnected(_)) => {
                if let Ok(mut status) = self.status.lock() {
                    status.state = "error";
                    status.detail = "SAM31 worker stopped".to_string();
                }
                SubmitOutcome::Invalid
            }
        }
    }

    /// Replace the six-row semantic bundle without unloading the resident
    /// image graph. The caller supplies an already encoded native prompt
    /// bundle; a rendezvous send prevents this control update from joining a
    /// stale frame FIFO.
    pub fn reload_prompt_bundle(&self, path: impl AsRef<Path>) -> SubmitOutcome {
        let Some(request) = self.request.as_ref() else {
            return SubmitOutcome::Invalid;
        };
        match request.try_send(WorkerRequest::ReloadPrompts(path.as_ref().to_path_buf())) {
            Ok(()) => SubmitOutcome::Accepted,
            Err(TrySendError::Full(_)) => SubmitOutcome::DroppedBusy,
            Err(TrySendError::Disconnected(_)) => SubmitOutcome::Invalid,
        }
    }

    pub fn drain_results(&self) -> Vec<OuterResult> {
        let mut results = Vec::new();
        loop {
            match self.results.try_recv() {
                Ok(result) => results.push(result),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        results
    }

    pub fn drain_proposal_masks(&self) -> Vec<Arc<ProposalMasks>> {
        let mut proposals = Vec::new();
        loop {
            match self.proposal_masks.try_recv() {
                Ok(candidate) => proposals.push(candidate),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => break,
            }
        }
        proposals
    }

    pub fn status(&self) -> StatusSnapshot {
        self.status
            .lock()
            .map(|status| status.clone())
            .unwrap_or_else(|_| StatusSnapshot {
                state: "error",
                detail: "SAM31 status lock poisoned".to_string(),
                ..StatusSnapshot::default()
            })
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        // CUDA and LibTorch process-global state must outlive every tensor.
        // Stop accepting queued work, wake the receiver, and join the worker
        // before the viewer can begin C++/CUDA runtime teardown.
        self.stop.store(true, AtomicOrdering::Release);
        self.request.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(not(feature = "sam31"))]
fn start_worker(
    _model: PathBuf,
    _prompt_bundle: PathBuf,
    _request: Receiver<WorkerRequest>,
    _results: SyncSender<OuterResult>,
    _proposal_masks: SyncSender<Arc<ProposalMasks>>,
    _status: Arc<Mutex<StatusSnapshot>>,
    _stop: Arc<AtomicBool>,
) -> Result<std::thread::JoinHandle<()>, String> {
    Err("SAM31 support is not compiled in; rebuild with --features sam31".to_string())
}

#[cfg(feature = "sam31")]
fn start_worker(
    model: PathBuf,
    prompt_bundle: PathBuf,
    request: Receiver<WorkerRequest>,
    results: SyncSender<OuterResult>,
    proposal_masks: SyncSender<Arc<ProposalMasks>>,
    status: Arc<Mutex<StatusSnapshot>>,
    stop: Arc<AtomicBool>,
) -> Result<std::thread::JoinHandle<()>, String> {
    thread::Builder::new()
        .name("sam31-iris-segmenter".to_string())
        .spawn(move || {
            runtime::worker(
                model,
                prompt_bundle,
                request,
                results,
                proposal_masks,
                status,
                stop,
            )
        })
        .map_err(|error| format!("spawn SAM31 iris worker: {error}"))
}

#[derive(Clone)]
struct FloatImage {
    width: usize,
    height: usize,
    data: Vec<[f32; 3]>,
}

impl FloatImage {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![[0.0; 3]; width * height],
        }
    }

    fn sample_clamped(&self, x: isize, y: isize) -> [f32; 3] {
        let x = x.clamp(0, self.width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, self.height.saturating_sub(1) as isize) as usize;
        self.data[y * self.width + x]
    }

    fn sample_reflect101(&self, x: isize, y: isize) -> [f32; 3] {
        let x = reflect101(x, self.width);
        let y = reflect101(y, self.height);
        self.data[y * self.width + x]
    }

    fn sample_bilinear_inside(&self, x: f64, y: f64) -> Option<f64> {
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x > self.width.saturating_sub(1) as f64
            || y > self.height.saturating_sub(1) as f64
        {
            return None;
        }
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let fx = x - x0 as f64;
        let fy = y - y0 as f64;
        let sample = |sample_x: usize, sample_y: usize| {
            self.data[sample_y * self.width + sample_x][0] as f64
        };
        Some(
            sample(x0, y0) * (1.0 - fx) * (1.0 - fy)
                + sample(x1, y0) * fx * (1.0 - fy)
                + sample(x0, y1) * (1.0 - fx) * fy
                + sample(x1, y1) * fx * fy,
        )
    }
}

fn raw_ring_support(image: &FloatImage, ellipse: Ellipse) -> RawRingSupport {
    raw_ring_support_below_ceiling(image, ellipse, None)
}

/// Pupil contrast must terminate in iris tissue, not in a screen reflection.
/// Use the surrounding iris to bound reflectance in the *original* RAW luma;
/// the center can be almost entirely covered by a specular highlight.
fn pupil_iris_luma_ceiling(image: &FloatImage, outer: Ellipse) -> f64 {
    let mut annulus = Vec::new();
    for y in (0..image.height).step_by(2) {
        for x in (0..image.width).step_by(2) {
            let radius = ellipse_coordinate((x as f64, y as f64), outer);
            let value = image.data[y * image.width + x][0] as f64;
            if (0.55..=0.85).contains(&radius) && value.is_finite() {
                annulus.push(value);
            }
        }
    }
    if annulus.len() < 64 {
        return 0.0;
    }
    let center = median(annulus.clone());
    let deviation = median(annulus.into_iter().map(|value| (value - center).abs()).collect());
    center + (4.0 * deviation).max(0.4 * center).max(12.0)
}

fn pupil_raw_support_is_sufficient(support: RawRingSupport) -> bool {
    support.score >= MIN_PUPIL_VOID_SUPPORT_SCORE
        && support.points >= 96
        && support.positive_fraction >= 0.42
        && support.strong_sectors >= 4
}

fn raw_ring_support_below_ceiling(
    image: &FloatImage,
    ellipse: Ellipse,
    ceiling: Option<f64>,
) -> RawRingSupport {
    const SAMPLES: usize = 192;
    const SECTORS: usize = 8;
    let scale = (ellipse.major_radius * ellipse.minor_radius).sqrt();
    if image.width == 0 || image.height == 0 || !scale.is_finite() || scale <= 1.0 {
        return RawRingSupport::default();
    }
    let offsets = [0.035, 0.060, 0.090].map(|factor| (scale * factor).clamp(2.5, 8.0));
    let (angle_sine, angle_cosine) = ellipse.angle.sin_cos();
    let mut contrasts = Vec::with_capacity(SAMPLES);
    let mut sectors: [Vec<f64>; SECTORS] = std::array::from_fn(|_| Vec::new());
    for index in 0..SAMPLES {
        let phase = std::f64::consts::TAU * index as f64 / SAMPLES as f64;
        let (phase_sine, phase_cosine) = phase.sin_cos();
        let x = ellipse.center.0 + angle_cosine * ellipse.major_radius * phase_cosine
            - angle_sine * ellipse.minor_radius * phase_sine;
        let y = ellipse.center.1
            + angle_sine * ellipse.major_radius * phase_cosine
            + angle_cosine * ellipse.minor_radius * phase_sine;
        let normal_local_x = phase_cosine / ellipse.major_radius.max(1e-6);
        let normal_local_y = phase_sine / ellipse.minor_radius.max(1e-6);
        let mut normal_x = angle_cosine * normal_local_x - angle_sine * normal_local_y;
        let mut normal_y = angle_sine * normal_local_x + angle_cosine * normal_local_y;
        let normal_length = normal_x.hypot(normal_y);
        if normal_length <= 1e-9 {
            continue;
        }
        normal_x /= normal_length;
        normal_y /= normal_length;
        let mut bands = Vec::with_capacity(offsets.len());
        for offset in offsets {
            let inside = image.sample_bilinear_inside(x - normal_x * offset, y - normal_y * offset);
            let outside =
                image.sample_bilinear_inside(x + normal_x * offset, y + normal_y * offset);
            if let (Some(inside), Some(outside)) = (inside, outside) {
                if ceiling.is_some_and(|limit| inside > limit || outside > limit) {
                    continue;
                }
                bands.push((outside - inside) / (0.5 * (outside + inside)).max(8.0));
            }
        }
        if bands.len() < 2 {
            continue;
        }
        let contrast = median(bands);
        if contrast.is_finite() {
            contrasts.push(contrast);
            sectors[index * SECTORS / SAMPLES].push(contrast);
        }
    }
    if contrasts.is_empty() {
        return RawRingSupport::default();
    }
    let points = contrasts.len();
    let contrast_median = median(contrasts.clone());
    let positive_fraction = contrasts
        .iter()
        .filter(|&&contrast| contrast > 0.015)
        .count() as f64
        / points as f64;
    let strong_sectors = sectors
        .into_iter()
        .filter(|sector| {
            sector.len() >= if ceiling.is_some() { 8 } else { 1 }
                && median(sector.clone()) > 0.025
        })
        .count();
    let support_fraction = points as f64 / SAMPLES as f64;
    let score = 4.0 * contrast_median
        + positive_fraction
        + 0.12 * strong_sectors as f64
        + (1.5 * support_fraction).min(1.0);
    RawRingSupport {
        score,
        points,
        positive_fraction,
        strong_sectors,
    }
}

/// Model pixels use center-aligned resampling. Same-aspect camera ROIs use
/// one uniform scale, preserving ellipse angle and physical foreshortening.
fn model_ellipse_in_source(mut ellipse: Ellipse, width: usize) -> Ellipse {
    let scale = width as f64 / FRAME_WIDTH as f64;
    ellipse.center = (
        (ellipse.center.0 + 0.5) * scale - 0.5,
        (ellipse.center.1 + 0.5) * scale - 0.5,
    );
    ellipse.major_radius *= scale;
    ellipse.minor_radius *= scale;
    ellipse
}

fn model_review_in_source(mut review: OuterMaskFitReview, width: usize) -> OuterMaskFitReview {
    let scale = width as f64 / FRAME_WIDTH as f64;
    review.ellipse = model_ellipse_in_source(review.ellipse, width);
    for points in [&mut review.retained_points, &mut review.flat_tire_points] {
        *points = Arc::new(
            points
                .iter()
                .map(|point| ((point.0 + 0.5) * scale - 0.5, (point.1 + 0.5) * scale - 0.5))
                .collect(),
        );
    }
    review.source_component_area_px *= scale * scale;
    review
}

fn outer_limbus_candidate_supersedes(
    candidate: Ellipse, candidate_raw: f64, candidate_semantic: f32,
    prior: Ellipse, prior_raw: f64, prior_semantic: f32,
) -> bool {
    // A nested dark region is often a pupil or a glint-fragmented inner iris.
    // Prefer its encompassing limbus only with stronger untouched-RAW edge
    // evidence; size alone must never promote a lid/whole-eye mask.
    let area_ratio = candidate.major_radius * candidate.minor_radius
        / (prior.major_radius * prior.minor_radius).max(1.0);
    area_ratio >= 1.25 && area_ratio <= 3.5
        && candidate_raw >= prior_raw + 0.15
        && candidate_semantic >= prior_semantic * 0.1
        && prior.dense_points(32).iter().filter(|&&p| ellipse_coordinate(p,candidate) <= 1.10).count() >= 28
}

fn pupil_ellipse_plausible(pupil: Ellipse, outer: Ellipse) -> bool {
    let radius_ratio = (pupil.major_radius * pupil.minor_radius
        / (outer.major_radius * outer.minor_radius).max(1.0))
    .sqrt();
    pupil.center.0.is_finite()
        && pupil.center.1.is_finite()
        && pupil.major_radius.is_finite()
        && pupil.minor_radius.is_finite()
        && pupil.major_radius >= 7.0
        && pupil.minor_radius >= 5.0
        && pupil.minor_radius / pupil.major_radius.max(1.0) >= 0.30
        && pupil_rectified_axis_ratio(pupil, outer).is_some_and(|ratio| ratio <= 1.65)
        && (0.09..=0.72).contains(&radius_ratio)
        && ellipse_coordinate(pupil.center, outer) <= 0.58
        && pupil
            .dense_points(32)
            .into_iter()
            .all(|point| ellipse_coordinate(point, outer) <= 0.92)
}

/// Remove the limbus-owned affine foreshortening before assessing pupil
/// shape. Testing only image-space aspect permits a crescent beside a glint
/// to masquerade as a strongly tilted pupil in an almost frontal iris.
/// The bound allows the optical/fit mismatch seen in clear corpus frames;
/// it does not force the two independently observed conics to be identical.
fn pupil_rectified_axis_ratio(pupil: Ellipse, outer: Ellipse) -> Option<f64> {
    if [pupil.major_radius,pupil.minor_radius,outer.major_radius,outer.minor_radius]
        .into_iter().any(|radius| !radius.is_finite() || radius<=0.0) { return None; }
    let (sine,cosine)=(pupil.angle-outer.angle).sin_cos();
    let p2=pupil.major_radius.powi(2);let q2=pupil.minor_radius.powi(2);
    let xx=(p2*cosine*cosine+q2*sine*sine)/outer.major_radius.powi(2);
    let yy=(p2*sine*sine+q2*cosine*cosine)/outer.minor_radius.powi(2);
    let xy=(p2-q2)*sine*cosine/(outer.major_radius*outer.minor_radius);
    let trace=xx+yy;let discriminant=(xx-yy).hypot(2.0*xy);
    let ratio=((trace+discriminant)/(trace-discriminant)).sqrt();
    (ratio.is_finite() && trace>discriminant).then_some(ratio)
}

#[derive(Clone, Copy, Debug)]
struct PupilFitPrior {
    radius_ratio: f64,
    center_offset: (f64, f64),
    log_radius_half_width: f64,
    center_half_width: f64,
}

impl PupilFitPrior {
    fn admits(self, pupil: Ellipse, outer: Ellipse) -> bool {
        let scale=(outer.major_radius*outer.minor_radius).sqrt();
        let ratio=(pupil.major_radius*pupil.minor_radius).sqrt()/scale;
        let offset=((pupil.center.0-outer.center.0)/scale,(pupil.center.1-outer.center.1)/scale);
        ((ratio/self.radius_ratio).ln()).abs()<=self.log_radius_half_width
            && (offset.0-self.center_offset.0).hypot(offset.1-self.center_offset.1)<=self.center_half_width
    }
}

#[derive(Clone, Copy)]
struct PupilContourObservation {
    timestamp_ns: u64,
    log_radius_ratio: f64,
    center_offset: (f64, f64),
}

#[derive(Default)]
struct PupilContourHistory {
    observations: VecDeque<PupilContourObservation>,
}

impl PupilContourHistory {
    fn prior(&self,timestamp_ns:u64) -> Option<PupilFitPrior> {
        let last=self.observations.back()?;
        if timestamp_ns<=last.timestamp_ns || timestamp_ns-last.timestamp_ns>PUPIL_CONTOUR_MAX_GAP_NS {return None;}
        // Two eyes share one inference worker. At 2–3 updates/second per eye,
        // a one-second *history* window never accumulated the three independent
        // observations required to activate the size prior. Keep that history
        // longer while still expiring a stale latest observation after 1 s.
        let recent=self.observations.iter().filter(|p|timestamp_ns-p.timestamp_ns<=PUPIL_CONTOUR_WINDOW_NS).collect::<Vec<_>>();
        if recent.len()<3 {return None;}
        let log_ratio=median(recent.iter().map(|p|p.log_radius_ratio).collect());
        let dispersion=median(recent.iter().map(|p|(p.log_radius_ratio-log_ratio).abs()).collect());
        let dt=(timestamp_ns-last.timestamp_ns) as f64*1e-9;
        Some(PupilFitPrior {
            radius_ratio:log_ratio.exp(),
            center_offset:(median(recent.iter().map(|p|p.center_offset.0).collect()),
                median(recent.iter().map(|p|p.center_offset.1).collect())),
            // Measurement dispersion and a small physical-size allowance are
            // separate. Neither raw sensor translation nor ROI relocation is
            // charged against this scale-invariant pupil/limbus trajectory.
            log_radius_half_width:(3.0*dispersion).clamp(0.12,0.20)+0.095*dt,
            center_half_width:(0.18+0.8*dt).min(0.40),
        })
    }

    fn observe(&mut self,timestamp_ns:u64,pupil:Ellipse,outer:Ellipse) {
        if !pupil_ellipse_plausible(pupil,outer) {return;}
        if self.observations.back().is_some_and(|p|timestamp_ns<=p.timestamp_ns
            || timestamp_ns-p.timestamp_ns>PUPIL_CONTOUR_MAX_GAP_NS) {self.observations.clear();}
        if self.prior(timestamp_ns).is_some_and(|prior|!prior.admits(pupil,outer)) {return;}
        let scale=(outer.major_radius*outer.minor_radius).sqrt();
        self.observations.push_back(PupilContourObservation {
            timestamp_ns,log_radius_ratio:((pupil.major_radius*pupil.minor_radius).sqrt()/scale).ln(),
            center_offset:((pupil.center.0-outer.center.0)/scale,(pupil.center.1-outer.center.1)/scale),
        });
        while self.observations.len()>7 {self.observations.pop_front();}
    }
}

/// A prior may guide a fresh RAW search, but never stand in for evidence.
/// All surviving candidates are independently checked on this exposure and
/// missing/occluded arcs cannot receive credit from the predicted ellipse.
fn refit_pupil_from_prior(image:&FloatImage,outer:Ellipse,prior:PupilFitPrior) -> Option<PupilVoidFitReview> {
    let scale=(outer.major_radius*outer.minor_radius).sqrt();
    let ceiling=pupil_iris_luma_ceiling(image,outer);
    let mut best=None::<(f64,PupilVoidFitReview)>;
    for dx in -2..=2 {for dy in -2..=2 {for dr in -2..=2 {
        let ratio=prior.radius_ratio*(0.05*dr as f64).exp();
        let pupil=Ellipse {
            center:(outer.center.0+scale*(prior.center_offset.0+0.025*dx as f64),
                outer.center.1+scale*(prior.center_offset.1+0.025*dy as f64)),
            major_radius:outer.major_radius*ratio,minor_radius:outer.minor_radius*ratio,angle:outer.angle,
        };
        if !pupil_ellipse_plausible(pupil,outer) || !prior.admits(pupil,outer) {continue;}
        let support=raw_ring_support_below_ceiling(image,pupil,Some(ceiling));
        if !pupil_raw_support_is_sufficient(support) {continue;}
        let objective=support.score-0.01*((dx*dx+dy*dy) as f64).sqrt();
        if best.is_none_or(|p|objective>p.0) {
            best=Some((objective,PupilVoidFitReview {ellipse:pupil,raw_support:support}));
        }
    }}}
    best.map(|(_,fit)|fit)
}

fn choose_current_pupil_recovery(
    component: Option<PupilVoidFitReview>,
    recovered: Option<PupilVoidFitReview>,
    outer: Ellipse,
    prior: Option<PupilFitPrior>,
) -> Option<(PupilVoidFitReview, bool)> {
    let rank = |fit: PupilVoidFitReview| {
        let penalty = prior.map_or(0.0, |prior| {
            let scale = (outer.major_radius * outer.minor_radius).sqrt();
            let offset = ((fit.ellipse.center.0 - outer.center.0) / scale,
                (fit.ellipse.center.1 - outer.center.1) / scale);
            0.75 * (offset.0 - prior.center_offset.0).hypot(offset.1 - prior.center_offset.1)
                / prior.center_half_width.max(0.05)
        });
        fit.raw_support.score - penalty
    };
    // Both alternatives have already passed the same current-exposure RAW
    // and geometry gates. A fragment's existence alone must not bypass a
    // better-supported, motion-consistent recovery of the established pupil.
    match (component, recovered) {
        (Some(component), Some(recovered)) if rank(recovered) > rank(component) => Some((recovered, false)),
        (Some(component), _) => Some((component, true)),
        (None, recovered) => recovered.map(|fit| (fit, false)),
    }
}

/// A semantic miss may briefly recover an established pupil using current
/// RAW edges, but cannot cold-acquire a different dark component. The boolean
/// explicitly marks independent observations allowed to refresh history.
fn choose_pupil_observation(
    semantic_requested: bool,
    semantic: Option<PupilVoidFitReview>,
    component: Option<PupilVoidFitReview>,
    recovered: Option<PupilVoidFitReview>,
    outer: Ellipse,
    prior: Option<PupilFitPrior>,
) -> Option<(PupilVoidFitReview, bool)> {
    if semantic_requested {
        semantic.map(|fit| (fit, true)).or_else(|| {
            prior.and(recovered).map(|fit| (fit, false))
        })
    } else {
        choose_current_pupil_recovery(component, recovered, outer, prior)
    }
}

fn fill_small_pupil_highlights(mask: &mut [bool], width: usize, height: usize) {
    if width < 3 || height < 3 {
        return;
    }
    // Specular pinholes and reflected eyelashes can split an otherwise solid
    // pupil component. Two conservative majority passes fill those holes
    // without the broad dilation that would merge dark iris fibers.
    for _ in 0..2 {
        let previous = mask.to_vec();
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let index = y * width + x;
                if previous[index] {
                    continue;
                }
                let neighbors = (-1isize..=1)
                    .flat_map(|dy| (-1isize..=1).map(move |dx| (dx, dy)))
                    .filter(|&(dx, dy)| dx != 0 || dy != 0)
                    .filter(|&(dx, dy)| {
                        previous[(y as isize + dy) as usize * width + (x as isize + dx) as usize]
                    })
                    .count();
                if neighbors >= 6 {
                    mask[index] = true;
                }
            }
        }
    }
}

/// Apply the same ordered-contour exclusion/refit as the SAM limbus at a
/// canonical scale, since its pixel thresholds are tuned for an iris-sized
/// contour. Preserve a two-source-pixel residual ceiling when magnifying the
/// pupil: four canonical pixels alone can be less than one sensor pixel.
/// Map back before applying any pupil geometry or RAW support gates.
fn deflattened_pupil_component(
    component: &[usize],
    width: usize,
    height: usize,
    reference: Ellipse,
) -> Option<Ellipse> {
    if reference.major_radius < 7.0 || reference.minor_radius < 5.0 {
        return None;
    }
    let scale = 80.0 / reference.major_radius;
    let origin = (FRAME_WIDTH as f64 * 0.5, FRAME_HEIGHT as f64 * 0.5);
    let contour = native_component_contour(component, width, height)
        .into_iter()
        .map(|point| {
            (
                origin.0 + (point.0 - reference.center.0) * scale,
                origin.1 + (point.1 - reference.center.1) * scale,
            )
        })
        .collect();
    let review = deflattened_mask_fit_with_noise(
        contour,
        Ellipse {
            center: origin,
            major_radius: reference.major_radius * scale,
            minor_radius: reference.minor_radius * scale,
            angle: reference.angle,
        },
        None,
        (0.5 * scale).max(1.0),
        // The joint-arc conditioning policy is validated on limbus labels.
        // Small glint-fragmented pupils keep their existing flat-tire gates
        // until pupil-specific evidence can validate equivalent constraints.
        false,
    )?;
    // A tiny surviving arc is insufficient to distinguish a pupil from a
    // shadow. Require substantial direct support after chord exclusion.
    if review.retained_points.len() < 48 {
        return None;
    }
    let fitted = review.ellipse;
    Some(Ellipse {
        center: (
            reference.center.0 + (fitted.center.0 - origin.0) / scale,
            reference.center.1 + (fitted.center.1 - origin.1) / scale,
        ),
        major_radius: fitted.major_radius / scale,
        minor_radius: fitted.minor_radius / scale,
        angle: fitted.angle,
    })
}

#[derive(Default, Debug)]
pub struct PupilFitDiagnostics {
    glare_ceiling: f64,
    components_in_area_range: usize,
    contour_fit_rejected: usize,
    geometry_rejected: usize,
    raw_support_rejected: usize,
    accepted: usize,
    competing_component_rejected: bool,
    detailed: bool,
    candidates: Vec<serde_json::Value>,
}

pub fn inspect_pupil_fit(frame: Arc<RawFrame>, outer: Ellipse) -> serde_json::Value {
    let image = raw_luma(&[frame]);
    let mut diagnostics = PupilFitDiagnostics::default();
    diagnostics.detailed = true;
    let fitted = fit_inner_pupil_void_diagnostic(&image[0], outer, None, &mut diagnostics);
    serde_json::json!({
        "glare_ceiling": diagnostics.glare_ceiling,
        "ellipse": fitted.map(|(ellipse, _)| serde_json::json!({
            "center":ellipse.center,"major_radius":ellipse.major_radius,
            "minor_radius":ellipse.minor_radius,"angle":ellipse.angle})),
        "support": fitted.map(|(_, support)| serde_json::json!({
            "score":support.score,"points":support.points,
            "positive_fraction":support.positive_fraction,"strong_sectors":support.strong_sectors})),
        "components_in_area_range": diagnostics.components_in_area_range,
        "contour_fit_rejected": diagnostics.contour_fit_rejected,
        "geometry_rejected": diagnostics.geometry_rejected,
        "raw_support_rejected": diagnostics.raw_support_rejected,
        "accepted": diagnostics.accepted,
        "competing_component_rejected": diagnostics.competing_component_rejected,
        "candidates": diagnostics.candidates,
    })
}

fn fit_inner_pupil_void(
    image: &FloatImage,
    outer: Ellipse,
    component_seed: Option<(f64, f64)>,
) -> Option<(Ellipse, RawRingSupport)> {
    fit_inner_pupil_void_diagnostic(
        image,
        outer,
        component_seed,
        &mut PupilFitDiagnostics::default(),
    )
}

fn fit_inner_pupil_void_diagnostic(
    image: &FloatImage,
    outer: Ellipse,
    component_seed: Option<(f64, f64)>,
    diagnostics: &mut PupilFitDiagnostics,
) -> Option<(Ellipse, RawRingSupport)> {
    fit_inner_pupil_void_conditioned(image,outer,component_seed,None,diagnostics)
}

fn fit_inner_pupil_void_conditioned(
    image: &FloatImage,
    outer: Ellipse,
    component_seed: Option<(f64, f64)>,
    prior: Option<PupilFitPrior>,
    diagnostics: &mut PupilFitDiagnostics,
) -> Option<(Ellipse, RawRingSupport)> {
    if image.width < 4 || image.height < 4 || !plausible_ellipse(outer) {
        return None;
    }
    let glare_ceiling = pupil_iris_luma_ceiling(image, outer);
    diagnostics.glare_ceiling = glare_ceiling;
    let mut population = Vec::<f32>::new();
    for y in 0..image.height {
        for x in 0..image.width {
            if ellipse_coordinate((x as f64, y as f64), outer) <= 0.74 {
                let value = image.data[y * image.width + x][0];
                if value.is_finite() {
                    population.push(value);
                }
            }
        }
    }
    if population.len() < 400 {
        return None;
    }
    population.sort_unstable_by(|left, right| left.total_cmp(right));
    let dark = percentile(&population, 10.0);
    let iris = percentile(&population, 44.0);
    let threshold = dark + 0.42 * (iris - dark).max(1.0);
    let mut mask = vec![false; image.width * image.height];
    for y in 0..image.height {
        for x in 0..image.width {
            let index = y * image.width + x;
            mask[index] = ellipse_coordinate((x as f64, y as f64), outer) <= 0.74
                && image.data[index][0] <= threshold;
        }
    }
    fill_small_pupil_highlights(&mut mask, image.width, image.height);

    let outer_area = std::f64::consts::PI * outer.major_radius * outer.minor_radius;
    let minimum_area = (outer_area * 0.012).max(140.0) as usize;
    let maximum_area = (outer_area * 0.42).max(minimum_area as f64 + 1.0) as usize;
    let seed = component_seed
        .filter(|&center| ellipse_coordinate(center, outer) <= 0.62)
        .unwrap_or(outer.center);
    let outer_scale = (outer.major_radius * outer.minor_radius).sqrt().max(1.0);
    let mut visited = vec![false; mask.len()];
    let mut queue = VecDeque::<usize>::new();
    let mut best: Option<(f64, Ellipse, RawRingSupport, usize)> = None;
    let mut largest_central_component = 0;
    for start in 0..mask.len() {
        if visited[start] || !mask[start] {
            continue;
        }
        visited[start] = true;
        queue.push_back(start);
        let mut component = Vec::<usize>::new();
        while let Some(index) = queue.pop_front() {
            component.push(index);
            let x = index % image.width;
            let y = index / image.width;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if (dx == 0 && dy == 0)
                        || !(0..image.width as isize).contains(&(x as isize + dx))
                        || !(0..image.height as isize).contains(&(y as isize + dy))
                    {
                        continue;
                    }
                    let neighbor =
                        (y as isize + dy) as usize * image.width + (x as isize + dx) as usize;
                    if !visited[neighbor] && mask[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        if component.len() < minimum_area || component.len() > maximum_area {
            continue;
        }
        diagnostics.components_in_area_range += 1;
        let points = component
            .iter()
            .map(|&index| ((index % image.width) as f64, (index / image.width) as f64))
            .collect::<Vec<_>>();
        let Some(mut ellipse) = moments_ellipse(&points) else {
            continue;
        };
        normalize_ellipse(&mut ellipse);
        if ellipse_coordinate(ellipse.center, outer) <= MAX_UNPROMPTED_PUPIL_CENTER_OFFSET {
            largest_central_component = largest_central_component.max(component.len());
        }
        if diagnostics.detailed {
            diagnostics
                .candidates
                .push(serde_json::json!({"area":component.len(),
                "center":ellipse.center,"radii":[ellipse.major_radius,ellipse.minor_radius]}));
        }
        let Some(ellipse) =
            deflattened_pupil_component(&component, image.width, image.height, ellipse)
        else {
            diagnostics.contour_fit_rejected += 1;
            continue;
        };
        // Unprompted acquisition must not pick a distant lid/shadow just
        // because its dark contour is strong. Explicit operator component
        // seeds retain their existing wider anatomical corridor.
        if !pupil_ellipse_plausible(ellipse, outer)
            || (component_seed.is_none()
                && ellipse_coordinate(ellipse.center, outer) > MAX_UNPROMPTED_PUPIL_CENTER_OFFSET)
        {
            diagnostics.geometry_rejected += 1;
            continue;
        }

        // Thresholding places the initial boundary close to the transition.
        // A bounded scale search then aligns it to untouched RAW10 instead of
        // trusting the exposure-dependent threshold as final geometry.
        let mut aligned = None::<(f64, Ellipse, RawRingSupport)>;
        for scale_step in -4i32..=4 {
            let scale = 1.0 + scale_step as f64 * 0.035;
            let candidate = Ellipse {
                major_radius: ellipse.major_radius * scale,
                minor_radius: ellipse.minor_radius * scale,
                ..ellipse
            };
            if !pupil_ellipse_plausible(candidate, outer)
                || prior.is_some_and(|prior|!prior.admits(candidate,outer)) {
                continue;
            }
            let support = raw_ring_support_below_ceiling(image, candidate, Some(glare_ceiling));
            let center_penalty =
                (candidate.center.0 - seed.0).hypot(candidate.center.1 - seed.1) / outer_scale;
            let objective = support.score - 0.55 * center_penalty;
            if aligned.as_ref().is_none_or(|current| objective > current.0) {
                aligned = Some((objective, candidate, support));
            }
        }
        let Some((objective, ellipse, support)) = aligned else {
            continue;
        };
        if !pupil_raw_support_is_sufficient(support) {
            diagnostics.raw_support_rejected += 1;
            continue;
        }
        diagnostics.accepted += 1;
        if best.as_ref().is_none_or(|current| objective > current.0) {
            best = Some((objective, ellipse, support, component.len()));
        }
    }
    let (_, ellipse, support, area) = best?;
    // Failure to fit the dominant dark region does not make a much smaller
    // iris fragment into a pupil. An explicit operator seed may intentionally
    // select another component; an unprompted fit must fail closed here.
    if component_seed.is_none() && area * 2 < largest_central_component {
        diagnostics.competing_component_rejected = true;
        return None;
    }
    Some((ellipse, support))
}

fn reflect101(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let length = length as isize;
    while index < 0 || index >= length {
        index = if index < 0 {
            -index
        } else {
            2 * length - index - 2
        };
    }
    index as usize
}

fn balanced_quad_rgb(frames: &[Arc<RawFrame>]) -> Vec<FloatImage> {
    let mut images = frames
        .iter()
        .map(|frame| demosaic_quad(frame))
        .collect::<Vec<_>>();
    temporal_white_balance(&mut images);
    images
}

fn demosaic_quad(frame: &RawFrame) -> FloatImage {
    let start_x = (2 - frame.sensor_x as usize % 2) % 2;
    let start_y = (2 - frame.sensor_y as usize % 2) % 2;
    let mosaic_width = (frame.width - start_x) / 2;
    let mosaic_height = (frame.height - start_y) / 2;
    let mut mosaic = vec![0.0f32; mosaic_width * mosaic_height];
    for y in 0..mosaic_height {
        for x in 0..mosaic_width {
            let source_x = start_x + x * 2;
            let source_y = start_y + y * 2;
            let mut sum = 0u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    sum += frame.pixels[(source_y + dy) * frame.width + source_x + dx] as u32;
                }
            }
            mosaic[y * mosaic_width + x] = sum as f32 * 0.25;
        }
    }
    let mosaic_sample = |x: isize, y: isize| {
        let x = x.clamp(0, mosaic_width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, mosaic_height.saturating_sub(1) as isize) as usize;
        mosaic[y * mosaic_width + x]
    };
    let origin_x = (frame.sensor_x as usize + start_x) / 2;
    let origin_y = (frame.sensor_y as usize + start_y) / 2;
    let mut half = FloatImage::new(mosaic_width, mosaic_height);
    for y in 0..mosaic_height {
        let even_y = (y + origin_y) & 1 == 0;
        for x in 0..mosaic_width {
            let even_x = (x + origin_x) & 1 == 0;
            let center = mosaic_sample(x as isize, y as isize);
            let horizontal = 0.5
                * (mosaic_sample(x as isize - 1, y as isize)
                    + mosaic_sample(x as isize + 1, y as isize));
            let vertical = 0.5
                * (mosaic_sample(x as isize, y as isize - 1)
                    + mosaic_sample(x as isize, y as isize + 1));
            let diagonal = 0.25
                * (mosaic_sample(x as isize - 1, y as isize - 1)
                    + mosaic_sample(x as isize + 1, y as isize - 1)
                    + mosaic_sample(x as isize - 1, y as isize + 1)
                    + mosaic_sample(x as isize + 1, y as isize + 1));
            half.data[y * mosaic_width + x] = match (even_y, even_x) {
                (true, true) => [center, 0.5 * (horizontal + vertical), diagonal],
                (true, false) => [horizontal, center, vertical],
                (false, true) => [vertical, center, horizontal],
                (false, false) => [diagonal, 0.5 * (horizontal + vertical), center],
            };
        }
    }
    resize_bilinear(&half, frame.width, frame.height)
}

fn resize_bilinear(source: &FloatImage, width: usize, height: usize) -> FloatImage {
    let mut output = FloatImage::new(width, height);
    for y in 0..height {
        let source_y = (y as f64 + 0.5) * source.height as f64 / height as f64 - 0.5;
        let y0 = source_y.floor() as isize;
        let fy = (source_y - y0 as f64) as f32;
        for x in 0..width {
            let source_x = (x as f64 + 0.5) * source.width as f64 / width as f64 - 0.5;
            let x0 = source_x.floor() as isize;
            let fx = (source_x - x0 as f64) as f32;
            let p00 = source.sample_clamped(x0, y0);
            let p10 = source.sample_clamped(x0 + 1, y0);
            let p01 = source.sample_clamped(x0, y0 + 1);
            let p11 = source.sample_clamped(x0 + 1, y0 + 1);
            let mut value = [0.0; 3];
            for channel in 0..3 {
                let top = p00[channel] * (1.0 - fx) + p10[channel] * fx;
                let bottom = p01[channel] * (1.0 - fx) + p11[channel] * fx;
                value[channel] = top * (1.0 - fy) + bottom * fy;
            }
            output.data[y * width + x] = value;
        }
    }
    output
}

fn temporal_white_balance(images: &mut [FloatImage]) {
    let mut sums = [0.0f64; 3];
    let mut count = 0usize;
    for image in images.iter() {
        for pixel in &image.data {
            for channel in 0..3 {
                sums[channel] += pixel[channel] as f64;
            }
            count += 1;
        }
    }
    if count == 0 {
        return;
    }
    let means = sums.map(|sum| sum / count as f64);
    let gains = means.map(|mean| (means[1] / mean.max(1.0)).clamp(0.25, 4.0) as f32);
    for image in images {
        for pixel in &mut image.data {
            for channel in 0..3 {
                pixel[channel] *= gains[channel];
            }
        }
    }
}

fn raw_luma(frames: &[Arc<RawFrame>]) -> Vec<FloatImage> {
    frames
        .iter()
        .map(|frame| {
            let mut image = FloatImage::new(frame.width, frame.height);
            for y in 0..frame.height {
                for x in 0..frame.width {
                    let mut sum = 0u32;
                    // OpenCV's even 4x4 default anchor is the lower/right
                    // middle sample, hence the [-2, +1] footprint.
                    for dy in -2isize..=1 {
                        let sy = reflect101(y as isize + dy, frame.height);
                        for dx in -2isize..=1 {
                            let sx = reflect101(x as isize + dx, frame.width);
                            sum += frame.pixels[sy * frame.width + sx] as u32;
                        }
                    }
                    let value = sum as f32 / 16.0;
                    image.data[y * frame.width + x] = [value; 3];
                }
            }
            image
        })
        .collect()
}

fn gaussian_blur(source: &FloatImage, sigma: f64) -> FloatImage {
    let radius = (sigma * 3.0).ceil() as isize;
    let mut kernel = (-radius..=radius)
        .map(|offset| (-0.5 * (offset as f64 / sigma).powi(2)).exp() as f32)
        .collect::<Vec<_>>();
    let sum = kernel.iter().copied().sum::<f32>().max(f32::EPSILON);
    for value in &mut kernel {
        *value /= sum;
    }
    let mut horizontal = FloatImage::new(source.width, source.height);
    for y in 0..source.height {
        for x in 0..source.width {
            let mut value = [0.0; 3];
            for (kernel_index, &weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                let sample = source.sample_reflect101(x as isize + offset, y as isize);
                for channel in 0..3 {
                    value[channel] += sample[channel] * weight;
                }
            }
            horizontal.data[y * source.width + x] = value;
        }
    }
    let mut output = FloatImage::new(source.width, source.height);
    for y in 0..source.height {
        for x in 0..source.width {
            let mut value = [0.0; 3];
            for (kernel_index, &weight) in kernel.iter().enumerate() {
                let offset = kernel_index as isize - radius;
                let sample = horizontal.sample_reflect101(x as isize, y as isize + offset);
                for channel in 0..3 {
                    value[channel] += sample[channel] * weight;
                }
            }
            output.data[y * source.width + x] = value;
        }
    }
    output
}

fn log_chroma(balanced: &[FloatImage]) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let smooth = gaussian_blur(image, 2.2);
            let mut output = FloatImage::new(image.width, image.height);
            for (destination, source) in output.data.iter_mut().zip(smooth.data.iter()) {
                let red = source[0];
                let green = source[1];
                let blue = source[2];
                *destination = [
                    ((red + 4.0) / (green + 4.0)).log2(),
                    0.25 * red + 0.50 * green + 0.25 * blue,
                    ((blue + 4.0) / (green + 4.0)).log2(),
                ];
            }
            output
        })
        .collect()
}

fn mildly_blurred(balanced: &[FloatImage]) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| gaussian_blur(image, 1.15))
        .collect()
}

fn unsharp(balanced: &[FloatImage], strength: f32) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let smooth = gaussian_blur(image, 1.35);
            let mut output = FloatImage::new(image.width, image.height);
            for ((destination, source), low_pass) in output
                .data
                .iter_mut()
                .zip(image.data.iter())
                .zip(smooth.data.iter())
            {
                for channel in 0..3 {
                    destination[channel] = (source[channel]
                        + strength * (source[channel] - low_pass[channel]))
                        .max(0.0);
                }
            }
            output
        })
        .collect()
}

fn partial_albedo(balanced: &[FloatImage]) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let illumination = gaussian_blur(image, 10.0);
            let mut output = FloatImage::new(image.width, image.height);
            for ((destination, source), field) in output
                .data
                .iter_mut()
                .zip(image.data.iter())
                .zip(illumination.data.iter())
            {
                for channel in 0..3 {
                    let correction = ((source[channel] + 8.0) / (field[channel] + 8.0))
                        .powf(0.28)
                        .clamp(0.55, 1.8);
                    destination[channel] = source[channel] * correction;
                }
            }
            output
        })
        .collect()
}

/// A deliberately conservative reflectance proxy. Dividing by a broad local
/// illumination field suppresses smooth eyelid/brow shadow ramps while
/// retaining vessel and limbus-scale structure. This is an input adapter, not
/// a claim of calibrated physical albedo.
fn illumination_normalized_albedo(balanced: &[FloatImage]) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let illumination = gaussian_blur(image, 10.0);
            let mut output = FloatImage::new(image.width, image.height);
            for ((destination, source), field) in output
                .data
                .iter_mut()
                .zip(image.data.iter())
                .zip(illumination.data.iter())
            {
                for channel in 0..3 {
                    destination[channel] =
                        ((source[channel] + 8.0) / (field[channel] + 8.0)).log2();
                }
            }
            output
        })
        .collect()
}

fn luma_pixel(pixel: [f32; 3]) -> f32 {
    0.25 * pixel[0] + 0.50 * pixel[1] + 0.25 * pixel[2]
}

fn high_pass_luma(balanced: &[FloatImage]) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let low = gaussian_blur(image, 3.2);
            let mut output = FloatImage::new(image.width, image.height);
            for ((destination, source), field) in output
                .data
                .iter_mut()
                .zip(image.data.iter())
                .zip(low.data.iter())
            {
                let detail = luma_pixel(*source) - luma_pixel(*field);
                *destination = [detail; 3];
            }
            output
        })
        .collect()
}

fn normalized_chromaticity(balanced: &[FloatImage]) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let smooth = gaussian_blur(image, 1.2);
            let mut output = FloatImage::new(image.width, image.height);
            for (destination, source) in output.data.iter_mut().zip(smooth.data.iter()) {
                let sum = source.iter().sum::<f32>() + 12.0;
                *destination = [
                    source[0] / sum,
                    luma_pixel(*source).ln_1p(),
                    source[2] / sum,
                ];
            }
            output
        })
        .collect()
}

fn dark_floor(balanced: &[FloatImage]) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let illumination = gaussian_blur(image, 8.0);
            let mut output = FloatImage::new(image.width, image.height);
            for ((destination, source), field) in output
                .data
                .iter_mut()
                .zip(image.data.iter())
                .zip(illumination.data.iter())
            {
                for channel in 0..3 {
                    destination[channel] = (source[channel] - 0.38 * field[channel]).max(0.0);
                }
            }
            output
        })
        .collect()
}

fn canny_luma(balanced: &[FloatImage], overlay: bool) -> Vec<FloatImage> {
    balanced
        .iter()
        .map(|image| {
            let mut gray = FloatImage::new(image.width, image.height);
            for (destination, source) in gray.data.iter_mut().zip(image.data.iter()) {
                let value = luma_pixel(*source).ln_1p();
                *destination = [value; 3];
            }
            let smooth = gaussian_blur(&gray, 1.05);
            let mut magnitude = vec![0.0f32; image.width * image.height];
            let mut direction = vec![0.0f32; magnitude.len()];
            for y in 1..image.height - 1 {
                for x in 1..image.width - 1 {
                    let sample = |dx: isize, dy: isize| {
                        smooth.data
                            [(y as isize + dy) as usize * image.width + (x as isize + dx) as usize]
                            [0]
                    };
                    let gx = -sample(-1, -1) + sample(1, -1) - 2.0 * sample(-1, 0)
                        + 2.0 * sample(1, 0)
                        - sample(-1, 1)
                        + sample(1, 1);
                    let gy = -sample(-1, -1) - 2.0 * sample(0, -1) - sample(1, -1)
                        + sample(-1, 1)
                        + 2.0 * sample(0, 1)
                        + sample(1, 1);
                    let index = y * image.width + x;
                    magnitude[index] = gx.hypot(gy);
                    direction[index] = gy.atan2(gx);
                }
            }
            let mut suppressed = vec![0.0f32; magnitude.len()];
            for y in 1..image.height - 1 {
                for x in 1..image.width - 1 {
                    let index = y * image.width + x;
                    let angle = direction[index].to_degrees().rem_euclid(180.0);
                    let ((dx1, dy1), (dx2, dy2)) = if !(22.5..157.5).contains(&angle) {
                        ((-1isize, 0isize), (1, 0))
                    } else if angle < 67.5 {
                        ((-1, -1), (1, 1))
                    } else if angle < 112.5 {
                        ((0, -1), (0, 1))
                    } else {
                        ((-1, 1), (1, -1))
                    };
                    let neighbor = |dx: isize, dy: isize| {
                        magnitude
                            [(y as isize + dy) as usize * image.width + (x as isize + dx) as usize]
                    };
                    if magnitude[index] >= neighbor(dx1, dy1)
                        && magnitude[index] >= neighbor(dx2, dy2)
                    {
                        suppressed[index] = magnitude[index];
                    }
                }
            }
            let mut population = suppressed
                .iter()
                .copied()
                .filter(|value| *value > 0.0 && value.is_finite())
                .collect::<Vec<_>>();
            population.sort_unstable_by(f32::total_cmp);
            let high = population
                .get(population.len().saturating_mul(82) / 100)
                .copied()
                .unwrap_or(1.0)
                .max(f32::EPSILON);
            let low = high * 0.42;
            let mut edge = vec![false; suppressed.len()];
            let mut queue = VecDeque::new();
            for (index, &value) in suppressed.iter().enumerate() {
                if value >= high {
                    edge[index] = true;
                    queue.push_back(index);
                }
            }
            while let Some(index) = queue.pop_front() {
                let x = index % image.width;
                let y = index / image.width;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let xx = x as isize + dx;
                        let yy = y as isize + dy;
                        if xx < 0
                            || yy < 0
                            || xx >= image.width as isize
                            || yy >= image.height as isize
                        {
                            continue;
                        }
                        let neighbor = yy as usize * image.width + xx as usize;
                        if !edge[neighbor] && suppressed[neighbor] >= low {
                            edge[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
            let mut output = FloatImage::new(image.width, image.height);
            for (index, destination) in output.data.iter_mut().enumerate() {
                let base = smooth.data[index][0];
                let line = if edge[index] { high * 2.5 } else { 0.0 };
                *destination = if overlay {
                    [base + line, base, base + 0.35 * line]
                } else {
                    [line; 3]
                };
            }
            output
        })
        .collect()
}

fn write_preprocessed_filmstrip(
    frames: &[Arc<RawFrame>],
    regime: PreprocessRegime,
    destination: &mut [u8],
) -> Result<(), String> {
    let hot_pixels = hot_pixel_check_enabled()
        .then(|| persistent_raw10_hot_pixels(frames))
        .unwrap_or_default();
    let corrected_frames = corrected_hot_pixel_frames(frames, &hot_pixels);
    let frames = corrected_frames.as_slice();
    let balanced = balanced_quad_rgb(frames);
    match regime {
        PreprocessRegime::BalancedQuadRgb => {
            write_quantized_filmstrip(&balanced, 0.35, 99.65, 0.82, destination)
        }
        PreprocessRegime::ShadowBoost => {
            write_quantized_filmstrip(&balanced, 0.10, 99.85, 0.62, destination)
        }
        PreprocessRegime::GentleShadowLift => {
            write_quantized_filmstrip(&balanced, 0.25, 99.75, 0.74, destination)
        }
        PreprocessRegime::MildBlur => {
            let images = mildly_blurred(&balanced);
            write_quantized_filmstrip(&images, 0.35, 99.65, 0.82, destination)
        }
        PreprocessRegime::PinkCenterMask => {
            let images = mildly_blurred(&balanced);
            write_quantized_filmstrip(&images, 0.35, 99.65, 0.82, destination)?;
            paint_pink_center_exclusions(destination, frames.len())?;
            Ok(())
        }
        PreprocessRegime::Unsharp => {
            let images = unsharp(&balanced, 0.85);
            write_quantized_filmstrip(&images, 0.35, 99.65, 0.88, destination)
        }
        PreprocessRegime::GentleUnsharp => {
            let images = unsharp(&balanced, 0.32);
            write_quantized_filmstrip(&images, 0.35, 99.65, 0.82, destination)
        }
        PreprocessRegime::IlluminationNormalizedAlbedo => {
            let images = illumination_normalized_albedo(&balanced);
            write_quantized_filmstrip(&images, 0.30, 99.70, 1.0, destination)
        }
        PreprocessRegime::PartialAlbedo => {
            let images = partial_albedo(&balanced);
            write_quantized_filmstrip(&images, 0.35, 99.65, 0.82, destination)
        }
        PreprocessRegime::RawLuma => {
            let images = raw_luma(frames);
            write_quantized_filmstrip(&images, 0.20, 99.75, 0.80, destination)
        }
        PreprocessRegime::LogChroma => {
            let images = log_chroma(&balanced);
            write_quantized_filmstrip(&images, 0.60, 99.40, 1.0, destination)
        }
        PreprocessRegime::StrongLowPass => {
            let images = balanced
                .iter()
                .map(|image| gaussian_blur(image, 2.4))
                .collect::<Vec<_>>();
            write_quantized_filmstrip(&images, 0.35, 99.65, 0.82, destination)
        }
        PreprocessRegime::HighPassLuma => {
            let images = high_pass_luma(&balanced);
            write_quantized_filmstrip(&images, 0.50, 99.50, 1.0, destination)
        }
        PreprocessRegime::CannyLumaOverlay => {
            let images = canny_luma(&balanced, true);
            write_quantized_filmstrip(&images, 0.25, 99.75, 0.88, destination)
        }
        PreprocessRegime::CannyEdgeOnly => {
            let images = canny_luma(&balanced, false);
            write_quantized_filmstrip(&images, 0.0, 100.0, 1.0, destination)
        }
        PreprocessRegime::NormalizedChromaticity => {
            let images = normalized_chromaticity(&balanced);
            write_quantized_filmstrip(&images, 0.40, 99.60, 1.0, destination)
        }
        PreprocessRegime::DarkFloor => {
            let images = dark_floor(&balanced);
            write_quantized_filmstrip(&images, 0.30, 99.70, 0.78, destination)
        }
    }
}

fn paint_pink_center_exclusions(destination: &mut [u8], frame_count: usize) -> Result<(), String> {
    let filmstrip_width = FRAME_WIDTH
        .checked_mul(frame_count)
        .ok_or("pink-center filmstrip width overflow")?;
    let plane_pixels = filmstrip_width
        .checked_mul(FRAME_HEIGHT)
        .ok_or("pink-center filmstrip area overflow")?;
    if frame_count == 0 || destination.len() != plane_pixels * 3 {
        return Err(format!(
            "pink-center adapter expected {} RGB bytes, got {}",
            plane_pixels * 3,
            destination.len()
        ));
    }

    // Search compact windows rather than individual minima so eyelashes and
    // hot/dead pixels cannot determine the exclusion center. The bounded
    // central search is deliberately crude: this is a nuisance-reflection
    // mask, never an anatomical pupil measurement.
    const SEARCH_HALF_W: usize = 25;
    const SEARCH_HALF_H: usize = 19;
    const BOX_HALF_W: usize = 34;
    const BOX_HALF_H: usize = 27;
    for frame in 0..frame_count {
        let tile_x = frame * FRAME_WIDTH;
        let mut best = None::<(u64, usize, usize)>;
        for center_y in (52..FRAME_HEIGHT.saturating_sub(40)).step_by(4) {
            for center_x in (52..FRAME_WIDTH.saturating_sub(40)).step_by(4) {
                let mut sum = 0u64;
                let mut samples = 0u64;
                for y in (center_y - SEARCH_HALF_H..=center_y + SEARCH_HALF_H).step_by(3) {
                    for x in (center_x - SEARCH_HALF_W..=center_x + SEARCH_HALF_W).step_by(3) {
                        let index = y * filmstrip_width + tile_x + x;
                        let red = destination[index] as u64;
                        let green = destination[plane_pixels + index] as u64;
                        let blue = destination[2 * plane_pixels + index] as u64;
                        sum += 2 * red + 5 * green + blue;
                        samples += 8;
                    }
                }
                let mean = sum / samples.max(1);
                if best.is_none_or(|candidate| mean < candidate.0) {
                    best = Some((mean, center_x, center_y));
                }
            }
        }
        let (_, center_x, center_y) = best.ok_or("pink-center search produced no window")?;
        let minimum_x = center_x.saturating_sub(BOX_HALF_W);
        let maximum_x = (center_x + BOX_HALF_W).min(FRAME_WIDTH - 1);
        let minimum_y = center_y.saturating_sub(BOX_HALF_H);
        let maximum_y = (center_y + BOX_HALF_H).min(FRAME_HEIGHT - 1);
        for y in minimum_y..=maximum_y {
            for x in minimum_x..=maximum_x {
                let index = y * filmstrip_width + tile_x + x;
                destination[index] = 255;
                destination[plane_pixels + index] = 0;
                destination[2 * plane_pixels + index] = 255;
            }
        }
    }
    Ok(())
}

fn extract_preprocessed_frame(
    filmstrip: &[u8],
    frame_index: usize,
    destination: &mut [u8],
) -> Result<(), String> {
    let frame_pixels = FRAME_WIDTH * FRAME_HEIGHT;
    let frame_count = filmstrip.len() / (frame_pixels * 3);
    let film_width = FRAME_WIDTH * frame_count;
    let film_pixels = film_width * FRAME_HEIGHT;
    if frame_count == 0
        || filmstrip.len() != frame_pixels * frame_count * 3
        || destination.len() != frame_pixels * 3
        || frame_index >= frame_count
    {
        return Err("invalid SAM31 frame extraction geometry".to_string());
    }
    for channel in 0..3 {
        for y in 0..FRAME_HEIGHT {
            let source_start = channel * film_pixels + y * film_width + frame_index * FRAME_WIDTH;
            let destination_start = channel * frame_pixels + y * FRAME_WIDTH;
            destination[destination_start..destination_start + FRAME_WIDTH]
                .copy_from_slice(&filmstrip[source_start..source_start + FRAME_WIDTH]);
        }
    }
    Ok(())
}

fn percentile(sorted: &[f32], quantile: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let position = (quantile / 100.0).clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let blend = (position - lower as f64) as f32;
    sorted[lower] * (1.0 - blend) + sorted[upper] * blend
}

fn adapter_bounds(images: &[FloatImage], low_q: f64, high_q: f64) -> ([f32; 3], [f32; 3]) {
    let mut samples: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    for image in images {
        // The six-query reference computes these bounds from every pixel in
        // the five-frame batch. Sampling here changes enough quad-RGB bytes to
        // alter SAM's instance ranking, so preserve that exact contract.
        for pixel in &image.data {
            for channel in 0..3 {
                if pixel[channel].is_finite() {
                    samples[channel].push(pixel[channel]);
                }
            }
        }
    }
    let mut low = [0.0; 3];
    let mut high = [1.0; 3];
    for channel in 0..3 {
        samples[channel].sort_unstable_by(|first, second| first.total_cmp(second));
        low[channel] = percentile(&samples[channel], low_q);
        high[channel] = percentile(&samples[channel], high_q).max(low[channel] + 1e-6);
    }
    (low, high)
}

fn write_quantized_filmstrip(
    images: &[FloatImage],
    low_q: f64,
    high_q: f64,
    gamma: f32,
    destination: &mut [u8],
) -> Result<(), String> {
    // Resize demosaiced/linear adapter pixels, never packed Bayer samples.
    // Native RAW and all published coordinates retain the camera ROI size.
    if images
        .iter()
        .any(|image| image.width != FRAME_WIDTH || image.height != FRAME_HEIGHT)
    {
        if images
            .iter()
            .any(|image| image.width == 0 || image.height == 0)
        {
            return Err("empty SAM31 adapter image".to_string());
        }
        let resized = images
            .iter()
            .map(|image| resize_bilinear(image, FRAME_WIDTH, FRAME_HEIGHT))
            .collect::<Vec<_>>();
        return write_quantized_filmstrip(&resized, low_q, high_q, gamma, destination);
    }
    let film_width = FRAME_WIDTH * images.len();
    let film_pixels = film_width * FRAME_HEIGHT;
    if images.is_empty()
        || images
            .iter()
            .any(|image| image.width != FRAME_WIDTH || image.height != FRAME_HEIGHT)
        || destination.len() != film_pixels * 3
    {
        return Err("invalid SAM31 adapter filmstrip geometry".to_string());
    }
    let (low, high) = adapter_bounds(images, low_q, high_q);
    for (frame_index, image) in images.iter().enumerate() {
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                let pixel = image.data[y * FRAME_WIDTH + x];
                let film_x = frame_index * FRAME_WIDTH + x;
                let output_index = y * film_width + film_x;
                for channel in 0..3 {
                    let normalized = ((pixel[channel] - low[channel])
                        / (high[channel] - low[channel]))
                        .clamp(0.0, 1.0);
                    let mapped = if (gamma - 1.0).abs() > f32::EPSILON {
                        normalized.powf(gamma)
                    } else {
                        normalized
                    };
                    destination[channel * film_pixels + output_index] =
                        (mapped * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "sam31")]
#[doc(hidden)]
pub fn diagnostic_quantized_adapters(
    frames: &[Arc<RawFrame>],
) -> Result<Vec<(&'static str, Vec<u8>)>, String> {
    if frames.len() != HISTORY_FRAMES {
        return Err(format!(
            "expected {HISTORY_FRAMES} frames, got {}",
            frames.len()
        ));
    }
    let balanced = balanced_quad_rgb(frames);
    let mut result = Vec::with_capacity(3);
    let mut bytes = vec![0u8; FILMSTRIP_PIXELS * 3];
    write_quantized_filmstrip(&balanced, 0.35, 99.65, 0.82, &mut bytes)?;
    result.push(("quad_rgb", bytes.clone()));
    let luma = raw_luma(frames);
    write_quantized_filmstrip(&luma, 0.20, 99.75, 0.80, &mut bytes)?;
    result.push(("raw_luma", bytes.clone()));
    let chroma = log_chroma(&balanced);
    write_quantized_filmstrip(&chroma, 0.60, 99.40, 1.0, &mut bytes)?;
    result.push(("log_chroma", bytes));
    Ok(result)
}

#[cfg(feature = "sam31")]
#[doc(hidden)]
pub fn diagnostic_fit_binary_filmstrip_mask(mask: &[u8], tile: usize) -> Option<Ellipse> {
    (mask.len() == FILMSTRIP_PIXELS && tile < HISTORY_FRAMES)
        .then(|| fit_mask_component(mask, FILMSTRIP_WIDTH, FRAME_HEIGHT, tile))
        .flatten()
}

#[cfg(feature = "sam31")]
#[doc(hidden)]
pub fn diagnostic_fit_single_frame_mask(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
) -> Option<OuterMaskFitReview> {
    if mask_width == 0 || mask_height == 0 || mask.len() != mask_width * mask_height {
        return None;
    }
    let wide_width = mask_width * HISTORY_FRAMES;
    let mut wide = vec![0u8; wide_width * mask_height];
    for y in 0..mask_height {
        let source = y * mask_width;
        let destination = y * wide_width + (HISTORY_FRAMES - 1) * mask_width;
        wide[destination..destination + mask_width]
            .copy_from_slice(&mask[source..source + mask_width]);
    }
    fit_mask_component_review(&wide, wide_width, mask_height, HISTORY_FRAMES - 1)
}

fn fit_single_frame_mask_with_context(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    scale_context: OuterContourScaleContext,
) -> Option<OuterMaskFitReview> {
    let fit =
        fit_single_frame_mask_candidate_with_context(mask, mask_width, mask_height, scale_context)?;
    distributed_contour_support(&fit).then_some(fit)
}

fn fit_single_frame_mask_candidate_with_context(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    scale_context: OuterContourScaleContext,
) -> Option<OuterMaskFitReview> {
    if mask_width == 0 || mask_height == 0 || mask.len() != mask_width * mask_height {
        return None;
    }
    // ProposalMasks stores only the latest temporal tile. Reconstruct the
    // filmstrip coordinate contract expected by the component/contour fitter
    // instead of accidentally interpreting this narrow mask as five tiles.
    let wide_width = mask_width * HISTORY_FRAMES;
    let mut wide = vec![0u8; wide_width * mask_height];
    for y in 0..mask_height {
        let source = y * mask_width;
        let destination = y * wide_width + (HISTORY_FRAMES - 1) * mask_width;
        wide[destination..destination + mask_width]
            .copy_from_slice(&mask[source..source + mask_width]);
    }
    fit_mask_component_review_with_context(
        &wide,
        wide_width,
        mask_height,
        HISTORY_FRAMES - 1,
        Some(scale_context),
    )
}

fn single_frame_mask_contour(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
) -> Vec<(f64, f64)> {
    if mask_width == 0 || mask_height == 0 || mask.len() != mask_width * mask_height {
        return Vec::new();
    }
    let wide_width = mask_width * HISTORY_FRAMES;
    let mut wide = vec![0u8; wide_width * mask_height];
    for y in 0..mask_height {
        let source = y * mask_width;
        let destination = y * wide_width + (HISTORY_FRAMES - 1) * mask_width;
        wide[destination..destination + mask_width]
            .copy_from_slice(&mask[source..source + mask_width]);
    }
    let component = largest_tile_component(&wide, wide_width, mask_height, HISTORY_FRAMES - 1);
    ordered_component_contour(&component, wide_width, mask_height, HISTORY_FRAMES - 1)
}

fn native_outline_points(
    mask: &[u8], mask_width: usize, mask_height: usize,
    source_width: usize, source_height: usize,
) -> Vec<(f64, f64)> {
    if source_width == 0 || source_height == 0 {
        return Vec::new();
    }
    let contour = single_frame_mask_contour(mask, mask_width, mask_height);
    sample_closed_contour(&contour, 256).into_iter().map(|(x, y)| (
        (x + 0.5) * source_width as f64 / FRAME_WIDTH as f64 - 0.5,
        (y + 0.5) * source_height as f64 / FRAME_HEIGHT as f64 - 0.5,
    )).collect()
}

#[derive(Clone, Copy)]
struct MaskCandidate {
    query: usize,
    objective: f64,
    model_score: f64,
}

fn mask_candidates_for_tile(
    masks: &[u8],
    scores: &[f32],
    query_count: usize,
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Vec<MaskCandidate> {
    let pixels = mask_width * mask_height;
    let scale =
        FILMSTRIP_WIDTH as f64 / mask_width as f64 * FRAME_HEIGHT as f64 / mask_height as f64;
    let mut candidates = Vec::new();
    for query in 0..query_count.min(scores.len()) {
        let score = scores[query] as f64;
        if !score.is_finite() || score <= 0.005 {
            continue;
        }
        let mask = &masks[query * pixels..(query + 1) * pixels];
        // The reference ranks individual external components, not the sum of
        // every disconnected object produced by one query. A stray eyelash in
        // the same instance must not make that query beat the iris component.
        let component = largest_tile_component(mask, mask_width, mask_height, tile);
        let full_area = component.len() as f64 * scale;
        if full_area < minimum_outer_component_area() as f64
            || full_area > MAX_COMPONENT_AREA_FULL_RES as f64
        {
            continue;
        }
        let objective = score + full_area / 1_000_000.0;
        candidates.push(MaskCandidate {
            query,
            objective,
            model_score: score,
        });
    }
    candidates.sort_unstable_by(|left, right| right.objective.total_cmp(&left.objective));
    candidates
}

fn tile_for_mask_x(x: usize, mask_width: usize) -> Option<usize> {
    let film_x = (x as f64 + 0.5) * FILMSTRIP_WIDTH as f64 / mask_width as f64 - 0.5;
    let tile = (film_x / FRAME_WIDTH as f64).floor() as isize;
    (0..HISTORY_FRAMES as isize)
        .contains(&tile)
        .then_some(tile as usize)
}

fn mask_x_range_for_tile(mask_width: usize, tile: usize) -> Option<(usize, usize)> {
    let start = (0..mask_width).find(|&x| tile_for_mask_x(x, mask_width) == Some(tile))?;
    let end = (start..mask_width)
        .find(|&x| tile_for_mask_x(x, mask_width) != Some(tile))
        .unwrap_or(mask_width);
    Some((start, end))
}

fn binary_mask_boundary_indices(mask: &[u8], width: usize, height: usize) -> Vec<u32> {
    if width == 0 || height == 0 || mask.len() != width.saturating_mul(height) {
        return Vec::new();
    }
    let mut boundary = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if mask[index] == 0 {
                continue;
            }
            if x == 0
                || y == 0
                || x + 1 == width
                || y + 1 == height
                || mask[index - 1] == 0
                || mask[index + 1] == 0
                || mask[index - width] == 0
                || mask[index + width] == 0
            {
                boundary.push(index as u32);
            }
        }
    }
    boundary
}

fn largest_tile_component(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Vec<usize> {
    let Some((start_x, end_x)) = mask_x_range_for_tile(mask_width, tile) else {
        return Vec::new();
    };
    let mut visited = vec![false; mask.len()];
    let mut best = Vec::new();
    let mut queue = VecDeque::new();
    for y in 0..mask_height {
        for x in start_x..end_x {
            let start = y * mask_width + x;
            if visited[start] || mask[start] == 0 {
                continue;
            }
            visited[start] = true;
            queue.push_back(start);
            let mut component = Vec::new();
            while let Some(index) = queue.pop_front() {
                component.push(index);
                let px = index % mask_width;
                let py = index / mask_width;
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = px as isize + dx;
                        let ny = py as isize + dy;
                        if nx < 0
                            || ny < 0
                            || nx >= mask_width as isize
                            || ny >= mask_height as isize
                        {
                            continue;
                        }
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if nx < start_x || nx >= end_x {
                            continue;
                        }
                        let neighbor = ny * mask_width + nx;
                        if !visited[neighbor] && mask[neighbor] != 0 {
                            visited[neighbor] = true;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
            if component.len() > best.len() {
                best = component;
            }
        }
    }
    best
}

fn ordered_component_contour_fallback(
    component: &[usize],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Vec<(f64, f64)> {
    native_component_contour(component, mask_width, mask_height)
        .into_iter()
        .map(|(x, y)| {
            (
                (x + 0.5) * FILMSTRIP_WIDTH as f64 / mask_width as f64
                    - 0.5
                    - tile as f64 * FRAME_WIDTH as f64,
                (y + 0.5) * FRAME_HEIGHT as f64 / mask_height as f64 - 0.5,
            )
        })
        .collect()
}

#[cfg(feature = "sam31")]
fn opencv_component_contour(
    component: &[usize],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Option<Vec<(f64, f64)>> {
    use opencv::core::{Mat, Point, Scalar, Vector, CV_8UC1};
    use opencv::prelude::*;

    let (start_x, end_x) = mask_x_range_for_tile(mask_width, tile)?;
    let tile_width = end_x.checked_sub(start_x)?;
    let mut image = Mat::new_rows_cols_with_default(
        mask_height.try_into().ok()?,
        tile_width.try_into().ok()?,
        CV_8UC1,
        Scalar::all(0.0),
    )
    .ok()?;
    let bytes = image.data_bytes_mut().ok()?;
    for &index in component {
        let x = index % mask_width;
        let y = index / mask_width;
        if (start_x..end_x).contains(&x) && y < mask_height {
            bytes[y * tile_width + x - start_x] = 1;
        }
    }
    let mut contours = Vector::<Vector<Point>>::new();
    opencv::imgproc::find_contours_def(
        &image,
        &mut contours,
        opencv::imgproc::RETR_EXTERNAL,
        opencv::imgproc::CHAIN_APPROX_NONE,
    )
    .ok()?;
    let contour = contours
        .iter()
        .filter_map(|contour| {
            let area = opencv::imgproc::contour_area(&contour, false).ok()?;
            Some((area, contour))
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))?
        .1;
    let points = contour
        .iter()
        .map(|point| {
            let low_x = start_x + point.x.max(0) as usize;
            (
                (low_x as f64 + 0.5) * FILMSTRIP_WIDTH as f64 / mask_width as f64
                    - 0.5
                    - tile as f64 * FRAME_WIDTH as f64,
                (point.y as f64 + 0.5) * FRAME_HEIGHT as f64 / mask_height as f64 - 0.5,
            )
        })
        .collect::<Vec<_>>();
    (points.len() >= 5).then_some(points)
}

fn ordered_component_contour(
    component: &[usize],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Vec<(f64, f64)> {
    #[cfg(feature = "sam31")]
    if let Some(contour) = opencv_component_contour(component, mask_width, mask_height, tile) {
        return contour;
    }
    ordered_component_contour_fallback(component, mask_width, mask_height, tile)
}

fn low_to_tile_point(
    index: usize,
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> (f64, f64) {
    let x = index % mask_width;
    let y = index / mask_width;
    (
        (x as f64 + 0.5) * FILMSTRIP_WIDTH as f64 / mask_width as f64
            - 0.5
            - tile as f64 * FRAME_WIDTH as f64,
        (y as f64 + 0.5) * FRAME_HEIGHT as f64 / mask_height as f64 - 0.5,
    )
}

fn fit_mask_component(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Option<Ellipse> {
    fit_mask_component_review(mask, mask_width, mask_height, tile).map(|fit| fit.ellipse)
}

fn fit_mask_component_review(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Option<OuterMaskFitReview> {
    fit_mask_component_review_with_context(mask, mask_width, mask_height, tile, None)
}

fn fit_mask_component_review_with_context(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
    scale_context: Option<OuterContourScaleContext>,
) -> Option<OuterMaskFitReview> {
    let component = largest_tile_component(mask, mask_width, mask_height, tile);
    let scale =
        FILMSTRIP_WIDTH as f64 / mask_width as f64 * FRAME_HEIGHT as f64 / mask_height as f64;
    let full_area = component.len() as f64 * scale;
    if full_area < minimum_outer_component_area() as f64 {
        if std::env::var_os("BUTTERCUP_SAM31_FIT_DEBUG").is_some() {
            eprintln!(
                "SAM31_FIT_DEBUG tile={tile} component_pixels={} full_area={full_area:.1} rejected=area",
                component.len(),
            );
        }
        return None;
    }
    let filled_points = component
        .iter()
        .map(|&index| low_to_tile_point(index, mask_width, mask_height, tile))
        .collect::<Vec<_>>();
    let initial = moments_ellipse(&filled_points)?;
    let contour = ordered_component_contour(&component, mask_width, mask_height, tile);
    let contour_points = contour.len();
    let mut de_flat_tired = deflattened_mask_fit_with_context(contour, initial, scale_context);
    if let Some(fit) = de_flat_tired.as_mut() {
        fit.source_component_area_px = full_area;
    }
    let initial_plausible = plausible_ellipse(initial);
    if std::env::var_os("BUTTERCUP_SAM31_FIT_DEBUG").is_some() {
        eprintln!(
            "SAM31_FIT_DEBUG tile={tile} component_pixels={} full_area={full_area:.1} contour_points={contour_points} initial={initial:?} initial_plausible={initial_plausible} de_flat_tired={:?}",
            component.len(),
            de_flat_tired.as_ref().map(|fit| (
                fit.ellipse,
                fit.retained_points.len(),
                fit.flat_tire_points.len(),
                fit.upper_flat_tire,
                fit.lower_flat_tire,
            )),
        );
    }
    // Moments describe the filled, possibly clipped component and therefore
    // reproduce its flat tire.  They are useful only as a crude orientation
    // and scale reference; never publish them when the boundary fit fails.
    de_flat_tired
}

/// Refit the selected canonical outer-iris answer under a frozen physical
/// scale context.  This is used by temporal/offline evaluators after an
/// independently estimated 2D affine scale has transported the prior; it
/// never changes SAM's mask or candidate selection.
pub fn contextual_outer_fit(
    proposal: &SemanticProposalMasks,
    scale_context: OuterContourScaleContext,
) -> Option<OuterMaskFitReview> {
    // The model-score winner may be an eyelid-shaped or whole-eye component
    // during a blink even when another returned object contains the usable
    // limbus arcs. Search every semantic answer under the independently
    // frozen scale interval; the interval constrains geometry but never
    // changes the SAM logits or invents points.
    let candidates = proposal
        .masks
        .iter()
        .filter_map(|mask| {
            let fit = fit_single_frame_mask_with_context(
                mask.pixels.as_slice(),
                proposal.width,
                proposal.height,
                scale_context,
            )?;
            let radius = fit.ellipse.major_radius.max(fit.ellipse.minor_radius);
            let support = ellipse_support_summary(&fit.retained_points, fit.ellipse);
            let ellipse_area = std::f64::consts::PI
                * fit.ellipse.major_radius
                * fit.ellipse.minor_radius;
            let component_fill = fit.source_component_area_px / ellipse_area.max(1.0);
            let scale_error =
                (radius / scale_context.estimated_fronto_parallel_radius_px.max(1.0))
                    .ln()
                    .abs();
            // Usable current-frame arc support owns the ranking. Model score
            // and closeness to the transported estimate only break otherwise
            // similar geometric explanations.
            let objective = fit.retained_points.len() as f64
                + 3.0 * support.occupied_sectors as f64
                - 20.0 * scale_error
                - 18.0 * (component_fill - 0.82).abs()
                + 2.0 * f64::from(mask.score.clamp(-1.0, 1.0));
            if std::env::var_os("BUTTERCUP_SAM31_FIT_DEBUG").is_some() {
                eprintln!(
                    "SAM31_CONTEXT_FIT query={} objective={:.3} score={:.4} center=({:.2},{:.2}) radii=({:.2},{:.2}) usable={} censored={} sectors={} largest_gap={} opposite_pairs={} component_fill={:.3}",
                    mask.query,
                    objective,
                    mask.score,
                    fit.ellipse.center.0,
                    fit.ellipse.center.1,
                    fit.ellipse.major_radius,
                    fit.ellipse.minor_radius,
                    fit.retained_points.len(),
                    fit.flat_tire_points.len(),
                    support.occupied_sectors,
                    support.largest_empty_run,
                    support.opposite_pairs,
                    component_fill,
                );
            }
            Some((objective, mask.query, fit, support, component_fill))
        })
        .collect::<Vec<_>>();
    candidates
        .into_iter()
        .max_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| right.1.cmp(&left.1))
        })
        .and_then(|(_, _, fit, _, _)| {
            // A closed lid often supplies one long, nearly collinear slit
            // whose RANSAC extrapolation has the expected major radius. It is
            // not observable limbus geometry. Require distributed phase
            // support and at least two diametrically opposed sector pairs;
            // genuine lid occlusion may remove top/bottom sectors while the
            // two lateral arcs still satisfy this test.
            distributed_contour_support(&fit).then_some(fit)
        })
}

fn distributed_contour_support(fit: &OuterMaskFitReview) -> bool {
    let support = ellipse_support_summary(&fit.retained_points, fit.ellipse);
    let ellipse_area = std::f64::consts::PI * fit.ellipse.major_radius * fit.ellipse.minor_radius;
    let component_fill = fit.source_component_area_px / ellipse_area.max(1.0);
    support.occupied_sectors >= 7
        && support.largest_empty_run <= 6
        && support.opposite_pairs >= 2
        && (0.38..=1.32).contains(&component_fill)
}

fn partial_temporal_contour_support(fit: &OuterMaskFitReview, bootstrap: bool) -> bool {
    let ellipse_area = std::f64::consts::PI * fit.ellipse.major_radius * fit.ellipse.minor_radius;
    let component_fill = fit.source_component_area_px / ellipse_area.max(1.0);
    let axis_ratio = fit.ellipse.minor_radius / fit.ellipse.major_radius.max(1.0);
    let minimum_points = if bootstrap { 40 } else { 32 };
    fit.retained_points.len() >= minimum_points
        && axis_ratio >= if bootstrap { 0.72 } else { 0.68 }
        && (0.38..=1.32).contains(&component_fill)
}

fn temporal_contour_support(
    fit: &OuterMaskFitReview,
    source: &RawFrame,
    trusted: Option<(&OuterMaskFitReview, (u32, u32))>,
) -> bool {
    let axis_ratio = fit.ellipse.minor_radius / fit.ellipse.major_radius.max(1.0);
    let strong = fit.retained_points.len() >= 70 && axis_ratio >= 0.64;
    let Some((prior, prior_origin)) = trusted else {
        return strong;
    };
    let current_sensor_center = (
        source.sensor_x as f64 + fit.ellipse.center.0,
        source.sensor_y as f64 + fit.ellipse.center.1,
    );
    let prior_sensor_center = (
        prior_origin.0 as f64 + prior.ellipse.center.0,
        prior_origin.1 as f64 + prior.ellipse.center.1,
    );
    let center_motion = (current_sensor_center.0 - prior_sensor_center.0)
        .hypot(current_sensor_center.1 - prior_sensor_center.1);
    let minor_ratio = fit.ellipse.minor_radius / prior.ellipse.minor_radius.max(1.0);
    let ordinary_motion =
        center_motion <= prior.ellipse.major_radius * 0.24 && (0.78..=1.28).contains(&minor_ratio);
    ordinary_motion
        || (strong && center_motion <= prior.ellipse.major_radius * 0.38 && axis_ratio >= 0.72)
}

fn consensus(ellipses: &[Ellipse]) -> Option<Ellipse> {
    if ellipses.len() < 2 {
        return None;
    }
    let center_x = median(ellipses.iter().map(|ellipse| ellipse.center.0).collect());
    let center_y = median(ellipses.iter().map(|ellipse| ellipse.center.1).collect());
    let major = median(
        ellipses
            .iter()
            .map(|ellipse| ellipse.major_radius)
            .collect(),
    );
    let minor = median(
        ellipses
            .iter()
            .map(|ellipse| ellipse.minor_radius)
            .collect(),
    );
    let doubled = ellipses.iter().fold((0.0, 0.0), |sum, ellipse| {
        (
            sum.0 + (2.0 * ellipse.angle).cos(),
            sum.1 + (2.0 * ellipse.angle).sin(),
        )
    });
    let mut result = Ellipse {
        center: (center_x, center_y),
        major_radius: major,
        minor_radius: minor,
        angle: 0.5 * doubled.1.atan2(doubled.0),
    };
    normalize_ellipse(&mut result);
    plausible_ellipse(result).then_some(result)
}

fn ellipse_axis_angle_distance(first: f64, second: f64) -> f64 {
    (first - second + std::f64::consts::FRAC_PI_2).rem_euclid(std::f64::consts::PI)
        - std::f64::consts::FRAC_PI_2
}

fn adapter_ellipse_pair_cost(first: Ellipse, second: Ellipse) -> Option<f64> {
    let first_scale = (first.major_radius * first.minor_radius).sqrt();
    let second_scale = (second.major_radius * second.minor_radius).sqrt();
    let reference_scale = first_scale.min(second_scale).max(1.0);
    let center_error = (first.center.0 - second.center.0).hypot(first.center.1 - second.center.1)
        / reference_scale;
    let major_error = (first.major_radius / second.major_radius).ln().abs();
    let minor_error = (first.minor_radius / second.minor_radius).ln().abs();
    let first_ratio = first.major_radius / first.minor_radius.max(1.0);
    let second_ratio = second.major_radius / second.minor_radius.max(1.0);
    let angle_error = ellipse_axis_angle_distance(first.angle, second.angle).abs();

    // Merely producing two plausible components is not adapter agreement.
    // The fixed prompts must describe the same projected limbus.  Angle is
    // intentionally ignored when either fit is nearly circular, where its
    // fitted axis direction is mathematically unstable.
    // Upper/lower lid occlusion can move the component-derived vertical
    // center in opposite directions between luma and color while both masks
    // still describe the same limbus.  Permit that bounded displacement; the
    // independent major/minor-axis checks below continue to reject unrelated
    // full-face or glasses components.
    if center_error > 0.34
        || major_error > 1.25f64.ln()
        || minor_error > 1.25f64.ln()
        || (first_ratio.min(second_ratio) > 1.12 && angle_error > 35.0f64.to_radians())
    {
        return None;
    }
    Some(center_error + major_error + minor_error + 0.20 * angle_error)
}

fn agreeing_consensus(candidates: &[(Ellipse, f64)]) -> Option<(Ellipse, Vec<usize>)> {
    if candidates.len() < 2 {
        return None;
    }
    let mut best: Option<(usize, f64, f64, Vec<usize>)> = None;
    for mask in 1usize..(1usize << candidates.len()) {
        let indices = (0..candidates.len())
            .filter(|index| mask & (1usize << index) != 0)
            .collect::<Vec<_>>();
        if indices.len() < 2 {
            continue;
        }
        let mut pair_cost = 0.0;
        let mut compatible = true;
        for left in 0..indices.len() {
            for right in (left + 1)..indices.len() {
                let Some(cost) = adapter_ellipse_pair_cost(
                    candidates[indices[left]].0,
                    candidates[indices[right]].0,
                ) else {
                    compatible = false;
                    break;
                };
                pair_cost += cost;
            }
            if !compatible {
                break;
            }
        }
        if !compatible {
            continue;
        }
        let model_score = indices
            .iter()
            .map(|&index| candidates[index].1)
            .sum::<f64>();
        let replace = best.as_ref().is_none_or(|best| {
            indices.len() > best.0
                || (indices.len() == best.0
                    && (pair_cost < best.1 - 1.0e-9
                        || ((pair_cost - best.1).abs() <= 1.0e-9 && model_score > best.2)))
        });
        if replace {
            best = Some((indices.len(), pair_cost, model_score, indices));
        }
    }
    let (_, _, _, indices) = best?;
    let ellipses = indices
        .iter()
        .map(|&index| candidates[index].0)
        .collect::<Vec<_>>();
    consensus(&ellipses).map(|ellipse| (ellipse, indices))
}

#[cfg(feature = "sam31")]
mod runtime {
    use super::*;
    use std::convert::TryInto;
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_void};
    use tch::{CModule, Device, IValue, Kind, Tensor};

    const RTLD_NOW: c_int = 0x0002;
    const RTLD_GLOBAL: c_int = 0x0100;

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlerror() -> *const c_char;
        #[link_name = "_ZN2at8autocast18set_autocast_dtypeEN3c1010DeviceTypeENS1_10ScalarTypeE"]
        fn torch_set_autocast_dtype(device_type: i8, scalar_type: i8);
    }

    fn load_cuda_dispatch_library() -> Result<(), String> {
        // GNU ld drops libtorch_cuda under --as-needed because tch reaches its
        // kernels through PyTorch's dispatcher rather than a direct symbol.
        // Loading it globally registers CUDA kernels before JIT parameters are
        // materialized. Keep the handle process-resident intentionally.
        let library = CString::new("libtorch_cuda.so").unwrap();
        let handle = unsafe { dlopen(library.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            let detail = unsafe {
                let error = dlerror();
                if error.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    CStr::from_ptr(error).to_string_lossy().into_owned()
                }
            };
            Err(format!("load libtorch CUDA dispatch kernels: {detail}"))
        } else {
            Ok(())
        }
    }

    fn configure_cuda_bfloat16_autocast() {
        // c10::DeviceType::CUDA = 1 and c10::ScalarType::BFloat16 = 15.
        // The accepted six-query baseline explicitly uses CUDA bfloat16;
        // LibTorch otherwise defaults CUDA autocast to float16, which moves
        // SAM's zero-logit contour enough to spoil the outer-iris refit.
        unsafe { torch_set_autocast_dtype(1, 15) };
    }

    struct InferenceOutput {
        logits: Tensor,
        masks: Vec<u8>,
        scores: Vec<f32>,
        query_count: usize,
        mask_width: usize,
        mask_height: usize,
        video_features: Option<NativeVideoFeatures>,
    }

    struct NativeVideoFeatures {
        pyramid: [Tensor; 3],
        decoder_queries: Tensor,
    }

    struct PriorFrameFeatures {
        frame_index: usize,
        sequence: u64,
        features: NativeVideoFeatures,
        tracked_query: Option<usize>,
        masks: Vec<u8>,
        mask_width: usize,
        mask_height: usize,
        sensor_origin: (u32, u32),
        mask_memory: Option<Tensor>,
        temporal_conditioned: Option<Tensor>,
        tracker_mask: Option<Vec<u8>>,
        object_pointer: Option<Tensor>,
    }

    struct NativeMaskMemoryEncoder {
        weights: HashMap<String, Tensor>,
        spatial_position: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        random_image_position: Tensor,
    }

    struct NativeTrackerDecode {
        mask_logits: Tensor,
        iou_scores: Tensor,
        object_score_logit: Tensor,
        object_pointers: Tensor,
    }

    impl NativeMaskMemoryEncoder {
        fn load(path: &Path, device: Device) -> Result<Self, String> {
            let weights = Tensor::load_multi_with_device(path, Device::Cpu)
                .map_err(|error| format!("load SAM31 tracker bundle {}: {error}", path.display()))?
                .into_iter()
                .map(|(name, tensor)| {
                    (
                        name,
                        tensor.to_device_(device, Kind::BFloat16, false, false),
                    )
                })
                .collect::<HashMap<_, _>>();
            for required in [
                "maskmem_backbone.mask_downsampler.encoder.0.weight",
                "maskmem_backbone.mask_downsampler.encoder.12.weight",
                "maskmem_backbone.pix_feat_proj.weight",
                "maskmem_backbone.fuser.layers.1.pwconv2.weight",
            ] {
                if !weights.contains_key(required) {
                    return Err(format!("SAM31 tracker bundle lacks {required}"));
                }
            }
            let mut position = vec![0f32; 256 * 72 * 72];
            for y in 0..72 {
                for x in 0..72 {
                    let normalized_y = (y + 1) as f64 / 72.0 * std::f64::consts::TAU;
                    let normalized_x = (x + 1) as f64 / 72.0 * std::f64::consts::TAU;
                    for dimension in 0..128 {
                        let scale = 10000f64.powf(2.0 * (dimension / 2) as f64 / 128.0);
                        let y_value = if dimension % 2 == 0 {
                            (normalized_y / scale).sin()
                        } else {
                            (normalized_y / scale).cos()
                        };
                        let x_value = if dimension % 2 == 0 {
                            (normalized_x / scale).sin()
                        } else {
                            (normalized_x / scale).cos()
                        };
                        position[(dimension * 72 + y) * 72 + x] = y_value as f32;
                        position[((128 + dimension) * 72 + y) * 72 + x] = x_value as f32;
                    }
                }
            }
            let spatial_position = Tensor::from_slice(&position)
                .view([1, 256, 72, 72])
                .to_device_(device, Kind::BFloat16, false, false);
            let mut rope_cos = vec![0f32; 72 * 72 * 16];
            let mut rope_sin = vec![0f32; 72 * 72 * 16];
            for y in 0..72 {
                for x in 0..72 {
                    let token = y * 72 + x;
                    for pair in 0..16 {
                        let axial_pair = pair % 8;
                        let coordinate = if pair < 8 { x } else { y } as f64;
                        let frequency = 10000f64.powf(-((4 * axial_pair) as f64) / 32.0);
                        let phase = coordinate * frequency;
                        rope_cos[token * 16 + pair] = phase.cos() as f32;
                        rope_sin[token * 16 + pair] = phase.sin() as f32;
                    }
                }
            }
            let rope_cos = Tensor::from_slice(&rope_cos)
                .view([1, 1, 72 * 72, 16])
                .to_device_(device, Kind::Float, false, false);
            let rope_sin = Tensor::from_slice(&rope_sin)
                .view([1, 1, 72 * 72, 16])
                .to_device_(device, Kind::Float, false, false);
            let mut coordinates = Vec::with_capacity(72 * 72 * 2);
            for y in 0..72 {
                for x in 0..72 {
                    coordinates.push((x as f32 + 0.5) / 72.0);
                    coordinates.push((y as f32 + 0.5) / 72.0);
                }
            }
            let coordinates = Tensor::from_slice(&coordinates)
                .view([72, 72, 2])
                .to_device_(device, Kind::Float, false, false);
            let gaussian = weights
                .get("image_pe_layer.positional_encoding_gaussian_matrix")
                .ok_or_else(|| "SAM31 tracker bundle lacks propagation image PE".to_string())?
                .to_kind(Kind::Float);
            let phase = ((coordinates * 2.0 - 1.0).matmul(&gaussian)) * std::f64::consts::TAU;
            let random_image_position = Tensor::cat(&[phase.sin(), phase.cos()], -1)
                .permute([2, 0, 1])
                .unsqueeze(0)
                .to_kind(Kind::BFloat16);
            Ok(Self {
                weights,
                spatial_position,
                rope_cos,
                rope_sin,
                random_image_position,
            })
        }

        fn weight(&self, name: &str) -> Result<&Tensor, String> {
            self.weights
                .get(name)
                .ok_or_else(|| format!("SAM31 tracker tensor is missing: {name}"))
        }

        fn spatial_position(&self) -> &Tensor {
            &self.spatial_position
        }

        fn linear(&self, input: &Tensor, prefix: &str) -> Result<Tensor, String> {
            Ok(input.linear(
                self.weight(&format!("{prefix}.weight"))?,
                Some(self.weight(&format!("{prefix}.bias"))?),
            ))
        }

        fn layer_norm_last(&self, input: &Tensor, prefix: &str) -> Result<Tensor, String> {
            Ok(input.layer_norm(
                [256],
                Some(self.weight(&format!("{prefix}.weight"))?),
                Some(self.weight(&format!("{prefix}.bias"))?),
                1e-5,
                false,
            ))
        }

        fn rotate_axial(&self, input: &Tensor, repeat: usize) -> Tensor {
            let kind = input.kind();
            let paired = input.to_kind(Kind::Float).view([
                input.size()[0],
                input.size()[1],
                input.size()[2],
                16,
                2,
            ]);
            let real = paired.select(-1, 0);
            let imaginary = paired.select(-1, 1);
            let cosine = if repeat == 1 {
                self.rope_cos.shallow_clone()
            } else {
                self.rope_cos.repeat([1, 1, repeat as i64, 1])
            };
            let sine = if repeat == 1 {
                self.rope_sin.shallow_clone()
            } else {
                self.rope_sin.repeat([1, 1, repeat as i64, 1])
            };
            Tensor::stack(
                &[
                    &real * &cosine - &imaginary * &sine,
                    &real * &sine + &imaginary * &cosine,
                ],
                -1,
            )
            .flatten(3, 4)
            .to_kind(kind)
        }

        fn rope_attention(
            &self,
            query: &Tensor,
            key: &Tensor,
            value: &Tensor,
            repeat_key_rope: bool,
            key_tokens_without_rope: usize,
        ) -> Result<Tensor, String> {
            let batch = query.size()[0];
            let query_tokens = query.size()[1];
            let key_tokens = key.size()[1];
            let spatial_key_tokens = key_tokens - key_tokens_without_rope as i64;
            if query_tokens != 72 * 72
                || spatial_key_tokens % query_tokens != 0
                || (!repeat_key_rope && spatial_key_tokens != query_tokens)
            {
                return Err(format!(
                    "SAM31 temporal RoPE geometry is unsupported: q={:?} k={:?}",
                    query.size(),
                    key.size()
                ));
            }
            let key_repeat = (spatial_key_tokens / query_tokens) as usize;
            let query = query.view([batch, query_tokens, 8, 32]).transpose(1, 2);
            let key = key.view([batch, key_tokens, 8, 32]).transpose(1, 2);
            let value = value.view([batch, key_tokens, 8, 32]).transpose(1, 2);
            let query = self.rotate_axial(&query, 1);
            let spatial_key = self.rotate_axial(&key.narrow(2, 0, spatial_key_tokens), key_repeat);
            let key = if key_tokens_without_rope == 0 {
                spatial_key
            } else {
                Tensor::cat(
                    &[
                        spatial_key,
                        key.narrow(2, spatial_key_tokens, key_tokens_without_rope as i64),
                    ],
                    2,
                )
            };
            Ok(Tensor::scaled_dot_product_attention(
                &query,
                &key,
                &value,
                None::<&Tensor>,
                0.0,
                false,
                None,
                false,
            )
            .transpose(1, 2)
            .contiguous()
            .view([batch, query_tokens, 256]))
        }

        fn plain_attention(
            &self,
            query: &Tensor,
            key: &Tensor,
            value: &Tensor,
            prefix: &str,
        ) -> Result<Tensor, String> {
            let query = self.linear(query, &format!("{prefix}.q_proj"))?;
            let key = self.linear(key, &format!("{prefix}.k_proj"))?;
            let value = self.linear(value, &format!("{prefix}.v_proj"))?;
            let batch = query.size()[0];
            let query_tokens = query.size()[1];
            let key_tokens = key.size()[1];
            let internal = query.size()[2];
            if internal % 8 != 0 || key.size()[2] != internal || value.size()[2] != internal {
                return Err(format!(
                    "SAM31 two-way attention projection mismatch at {prefix}"
                ));
            }
            let head_width = internal / 8;
            let query = query
                .view([batch, query_tokens, 8, head_width])
                .transpose(1, 2);
            let key = key.view([batch, key_tokens, 8, head_width]).transpose(1, 2);
            let value = value
                .view([batch, key_tokens, 8, head_width])
                .transpose(1, 2);
            let attended = Tensor::scaled_dot_product_attention(
                &query,
                &key,
                &value,
                None::<&Tensor>,
                0.0,
                false,
                None,
                false,
            )
            .transpose(1, 2)
            .contiguous()
            .view([batch, query_tokens, internal]);
            self.linear(&attended, &format!("{prefix}.out_proj"))
        }

        fn two_way_transformer(
            &self,
            image_embedding: &Tensor,
            point_embedding: &Tensor,
        ) -> Result<(Tensor, Tensor), String> {
            let image_position = &self.random_image_position;
            let mut queries = point_embedding.shallow_clone();
            let mut keys = image_embedding.flatten(2, 3).transpose(1, 2);
            let key_position = image_position.flatten(2, 3).transpose(1, 2);
            for layer in 0..2 {
                let prefix = format!("sam_mask_decoder.transformer.layers.{layer}");
                if layer == 0 {
                    queries = self.plain_attention(
                        &queries,
                        &queries,
                        &queries,
                        &format!("{prefix}.self_attn"),
                    )?;
                } else {
                    let positioned = &queries + point_embedding;
                    queries += self.plain_attention(
                        &positioned,
                        &positioned,
                        &queries,
                        &format!("{prefix}.self_attn"),
                    )?;
                }
                queries = self.layer_norm_last(&queries, &format!("{prefix}.norm1"))?;

                let positioned_queries = &queries + point_embedding;
                let positioned_keys = &keys + &key_position;
                queries += self.plain_attention(
                    &positioned_queries,
                    &positioned_keys,
                    &keys,
                    &format!("{prefix}.cross_attn_token_to_image"),
                )?;
                queries = self.layer_norm_last(&queries, &format!("{prefix}.norm2"))?;

                let mlp = self.linear(&queries, &format!("{prefix}.mlp.lin1"))?.relu();
                queries += self.linear(&mlp, &format!("{prefix}.mlp.lin2"))?;
                queries = self.layer_norm_last(&queries, &format!("{prefix}.norm3"))?;

                let positioned_queries = &queries + point_embedding;
                let positioned_keys = &keys + &key_position;
                keys += self.plain_attention(
                    &positioned_keys,
                    &positioned_queries,
                    &queries,
                    &format!("{prefix}.cross_attn_image_to_token"),
                )?;
                keys = self.layer_norm_last(&keys, &format!("{prefix}.norm4"))?;
            }
            let query = &queries + point_embedding;
            let key = &keys + &key_position;
            queries += self.plain_attention(
                &query,
                &key,
                &keys,
                "sam_mask_decoder.transformer.final_attn_token_to_image",
            )?;
            queries =
                self.layer_norm_last(&queries, "sam_mask_decoder.transformer.norm_final_attn")?;
            Ok((queries, keys))
        }

        fn mlp(&self, input: &Tensor, prefix: &str, layers: usize) -> Result<Tensor, String> {
            let mut output = input.shallow_clone();
            for layer in 0..layers {
                output = self.linear(&output, &format!("{prefix}.layers.{layer}"))?;
                if layer + 1 != layers {
                    output = output.relu();
                }
            }
            Ok(output)
        }

        fn propagation_mask_decode(
            &self,
            conditioned_features: &Tensor,
            high_resolution_features: [&Tensor; 2],
        ) -> Result<NativeTrackerDecode, String> {
            if conditioned_features.size() != [1, 256, 72, 72]
                || high_resolution_features[0].size() != [1, 256, 288, 288]
                || high_resolution_features[1].size() != [1, 256, 144, 144]
            {
                return Err("SAM31 propagation decoder received incompatible FPN geometry".into());
            }
            let object_tokens = self.weight("sam_mask_decoder.obj_score_token.weight")?;
            let iou_tokens = self.weight("sam_mask_decoder.iou_token.weight")?;
            let valid = self.weight("output_valid_embed")?.narrow(0, 0, 1);
            let invalid = self.weight("output_invalid_embed")?.narrow(0, 1, 15);
            let suppression = Tensor::cat(&[valid, invalid], 0);
            let mask_tokens = self
                .weight("sam_mask_decoder.mask_tokens.weight")?
                .view([16, 3, 256])
                + suppression.unsqueeze(1);
            let tokens = Tensor::cat(&[object_tokens, iou_tokens, &mask_tokens.flatten(0, 1)], 0)
                .unsqueeze(0);
            let (tokens_out, image_out) =
                self.two_way_transformer(conditioned_features, &tokens)?;
            let object_tokens_out = tokens_out.narrow(1, 0, 16);
            let iou_tokens_out = tokens_out.narrow(1, 16, 16);
            let mask_tokens_out = tokens_out.narrow(1, 32, 48).view([1, 16, 3, 256]);

            let image = image_out.transpose(1, 2).view([1, 256, 72, 72]);
            let high_1 = self.conv(
                high_resolution_features[1],
                "sam_mask_decoder.conv_s1",
                1,
                0,
                1,
            )?;
            let mut upscaled = image.conv_transpose2d(
                self.weight("sam_mask_decoder.output_upscaling.0.weight")?,
                Some(self.weight("sam_mask_decoder.output_upscaling.0.bias")?),
                2,
                0,
                0,
                1,
                1,
            ) + high_1;
            upscaled = self
                .layer_norm_2d(&upscaled, "sam_mask_decoder.output_upscaling.1")?
                .gelu("none");
            let high_0 = self.conv(
                high_resolution_features[0],
                "sam_mask_decoder.conv_s0",
                1,
                0,
                1,
            )?;
            upscaled = upscaled.conv_transpose2d(
                self.weight("sam_mask_decoder.output_upscaling.3.weight")?,
                Some(self.weight("sam_mask_decoder.output_upscaling.3.bias")?),
                2,
                0,
                0,
                1,
                1,
            ) + high_0;
            upscaled = upscaled.gelu("none");

            let hyper = (0..3)
                .map(|mask| {
                    self.mlp(
                        &mask_tokens_out.select(2, mask),
                        &format!("sam_mask_decoder.output_hypernetworks_mlps.{mask}"),
                        3,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let hyper = Tensor::stack(&hyper, 2);
            let masks = hyper
                .flatten(1, 2)
                .bmm(&upscaled.flatten(2, 3))
                .view([1, 16, 3, 288, 288]);
            let iou_scores = self
                .mlp(&iou_tokens_out, "sam_mask_decoder.iou_prediction_head", 3)?
                .view([1, 16, 3]);
            let object_scores = self
                .mlp(
                    &object_tokens_out,
                    "sam_mask_decoder.pred_obj_score_head",
                    3,
                )?
                .view([1, 16]);
            let object_pointers = self.mlp(&mask_tokens_out, "obj_ptr_proj", 3)?;
            Ok(NativeTrackerDecode {
                mask_logits: masks.select(1, 0),
                iou_scores: iou_scores.select(1, 0),
                object_score_logit: object_scores.select(1, 0),
                object_pointers: object_pointers.select(1, 0),
            })
        }

        fn selected_object_pointer(
            &self,
            decoded: &NativeTrackerDecode,
            selected: usize,
            object_is_present: bool,
        ) -> Result<Tensor, String> {
            let pointer = decoded
                .object_pointers
                .get(0)
                .get(selected as i64)
                .unsqueeze(0);
            if object_is_present {
                Ok(pointer)
            } else {
                self.linear(&pointer, "no_obj_ptr_linear")
            }
        }

        fn condition_with_memory_bank(
            &self,
            current_image: &Tensor,
            memories: &[(&Tensor, &Tensor, usize)],
            pointers: &[(&Tensor, usize)],
        ) -> Result<Tensor, String> {
            if memories.is_empty() || memories.len() > 7 {
                return Err(format!(
                    "SAM31 temporal attention requires 1..=7 memories, got {}",
                    memories.len()
                ));
            }
            for (label, tensor) in
                std::iter::once(("current image", current_image)).chain(memories.iter().flat_map(
                    |(image, memory, _)| [("previous image", *image), ("previous memory", *memory)],
                ))
            {
                if tensor.size() != [1, 256, 72, 72] {
                    return Err(format!(
                        "SAM31 {label} feature has unexpected shape {:?}",
                        tensor.size()
                    ));
                }
            }
            let flatten = |tensor: &Tensor| tensor.flatten(2, 3).transpose(1, 2);
            let image = flatten(current_image);
            let source_position = flatten(&self.spatial_position);
            let mut memory_images = Vec::with_capacity(memories.len());
            let mut memory_masks = Vec::with_capacity(memories.len());
            let mut memory_positions = Vec::with_capacity(memories.len());
            let mut object_pointers = Vec::with_capacity(pointers.len());
            let mut object_pointer_positions = Vec::with_capacity(pointers.len());
            // Official non-conditioning order is oldest to newest. Axial RoPE
            // repeats per 72x72 block; the learned v2 temporal embedding
            // disambiguates the distance of each block.
            for &(previous_image, previous_memory, temporal_distance) in memories.iter().rev() {
                memory_images.push(flatten(previous_image));
                memory_masks.push(flatten(previous_memory));
                let temporal_index = if (1..7).contains(&temporal_distance) {
                    temporal_distance - 1
                } else {
                    6
                };
                let temporal = self
                    .weight("maskmem_tpos_enc")?
                    .get(temporal_index as i64)
                    .to_kind(source_position.kind());
                memory_positions.push(&source_position + temporal);
            }
            for &(pointer, temporal_distance) in pointers.iter().rev() {
                object_pointers.push(pointer.shallow_clone());
                let normalized = temporal_distance.min(15) as f32 / 15.0;
                let mut sine = Vec::with_capacity(256);
                for dimension in 0..128 {
                    let scale = 10000f32.powf(2.0 * (dimension / 2) as f32 / 128.0);
                    sine.push((normalized / scale).sin());
                }
                for dimension in 0..128 {
                    let scale = 10000f32.powf(2.0 * (dimension / 2) as f32 / 128.0);
                    sine.push((normalized / scale).cos());
                }
                let position = Tensor::from_slice(&sine).view([1, 1, 256]).to_device_(
                    current_image.device(),
                    current_image.kind(),
                    false,
                    false,
                );
                object_pointer_positions
                    .push(self.linear(&position, "obj_ptr_tpos_proj")?.squeeze_dim(1));
            }
            let mut memory_image = Tensor::cat(&memory_images.iter().collect::<Vec<_>>(), 1);
            let mut memory = Tensor::cat(&memory_masks.iter().collect::<Vec<_>>(), 1);
            let mut memory_image_position =
                Tensor::cat(&memory_positions.iter().collect::<Vec<_>>(), 1);
            let pointer_count = object_pointers.len();
            if pointer_count > 0 {
                let pointers =
                    Tensor::cat(&object_pointers.iter().collect::<Vec<_>>(), 0).unsqueeze(0);
                let pointer_positions =
                    Tensor::cat(&object_pointer_positions.iter().collect::<Vec<_>>(), 0)
                        .unsqueeze(0);
                let zeros = Tensor::zeros(
                    [1, pointer_count as i64, 256],
                    (memory_image.kind(), memory_image.device()),
                );
                memory_image = Tensor::cat(&[memory_image, zeros], 1);
                memory = Tensor::cat(&[memory, pointers], 1);
                memory_image_position = Tensor::cat(&[memory_image_position, pointer_positions], 1);
            }
            let mut output = &image + 0.1 * &source_position;
            for layer in 0..4 {
                let prefix = format!("transformer.encoder.layers.{layer}");
                let normalized = self.layer_norm_last(&output, &format!("{prefix}.norm1"))?;
                let query = self.linear(&normalized, &format!("{prefix}.self_attn_q_proj"))?;
                let key = self.linear(&normalized, &format!("{prefix}.self_attn_k_proj"))?;
                let value = self.linear(&normalized, &format!("{prefix}.self_attn_v_proj"))?;
                let attended = self.rope_attention(&query, &key, &value, false, 0)?;
                output += self.linear(&attended, &format!("{prefix}.self_attn_out_proj"))?;

                let normalized = self.layer_norm_last(&output, &format!("{prefix}.norm2"))?;
                let query = self.linear(&image, &format!("{prefix}.image_cross_attn_q_proj"))?
                    + self.linear(&normalized, &format!("{prefix}.cross_attn_q_proj"))?;
                let key = self
                    .linear(&memory_image, &format!("{prefix}.image_cross_attn_k_proj"))?
                    + self.linear(&memory, &format!("{prefix}.cross_attn_k_proj"))?
                    + &memory_image_position;
                let value = self.linear(&memory, &format!("{prefix}.cross_attn_v_proj"))?;
                let attended = self.rope_attention(&query, &key, &value, true, pointer_count)?;
                output += self.linear(&attended, &format!("{prefix}.cross_attn_out_proj"))?;

                let normalized = self.layer_norm_last(&output, &format!("{prefix}.norm3"))?;
                let feed_forward = self
                    .linear(&normalized, &format!("{prefix}.linear1"))?
                    .gelu("none");
                output += self.linear(&feed_forward, &format!("{prefix}.linear2"))?;
            }
            let output = self.layer_norm_last(&output, "transformer.encoder.norm")?;
            Ok(output.transpose(1, 2).view([1, 256, 72, 72]))
        }

        fn conv(
            &self,
            input: &Tensor,
            prefix: &str,
            stride: i64,
            padding: i64,
            groups: i64,
        ) -> Result<Tensor, String> {
            Ok(input.conv2d(
                self.weight(&format!("{prefix}.weight"))?,
                Some(self.weight(&format!("{prefix}.bias"))?),
                stride,
                padding,
                1,
                groups,
            ))
        }

        fn layer_norm_2d(&self, input: &Tensor, prefix: &str) -> Result<Tensor, String> {
            let mean = input.mean_dim([1].as_slice(), true, Kind::Float);
            let centered = input - &mean;
            let variance =
                centered
                    .pow_tensor_scalar(2.0)
                    .mean_dim([1].as_slice(), true, Kind::Float);
            let normalized = centered / (variance + 1e-6).sqrt();
            Ok(self
                .weight(&format!("{prefix}.weight"))?
                .view([1, -1, 1, 1])
                * normalized
                + self.weight(&format!("{prefix}.bias"))?.view([1, -1, 1, 1]))
        }

        fn encode(&self, pixel_features: &Tensor, mask_logits: &Tensor) -> Result<Tensor, String> {
            let mask = (mask_logits.sigmoid() * 2.0 - 1.0).internal_upsample_bilinear2d_aa(
                [1152, 1152],
                false,
                None,
                None,
            );
            let zeros = Tensor::zeros([1, 15, 1152, 1152], (mask.kind(), mask.device()));
            let condition = Tensor::ones([1, 1, 1152, 1152], (mask.kind(), mask.device()));
            let mut encoded = Tensor::cat(&[&mask, &zeros, &condition, &zeros], 1);
            for (conv_index, norm_index) in [(0, 1), (3, 4), (6, 7), (9, 10)] {
                encoded = self.conv(
                    &encoded,
                    &format!("maskmem_backbone.mask_downsampler.encoder.{conv_index}"),
                    2,
                    1,
                    1,
                )?;
                encoded = self
                    .layer_norm_2d(
                        &encoded,
                        &format!("maskmem_backbone.mask_downsampler.encoder.{norm_index}"),
                    )?
                    .gelu("none");
            }
            encoded = self.conv(
                &encoded,
                "maskmem_backbone.mask_downsampler.encoder.12",
                1,
                0,
                1,
            )?;
            let projected = self.conv(pixel_features, "maskmem_backbone.pix_feat_proj", 1, 0, 1)?;
            let mut fused = projected + encoded;
            for layer in 0..2 {
                let residual = fused.shallow_clone();
                let prefix = format!("maskmem_backbone.fuser.layers.{layer}");
                let mut update = self.conv(&fused, &format!("{prefix}.dwconv"), 1, 3, 256)?;
                update = self.layer_norm_2d(&update, &format!("{prefix}.norm"))?;
                update = update
                    .permute([0, 2, 3, 1])
                    .linear(
                        self.weight(&format!("{prefix}.pwconv1.weight"))?,
                        Some(self.weight(&format!("{prefix}.pwconv1.bias"))?),
                    )
                    .gelu("none")
                    .linear(
                        self.weight(&format!("{prefix}.pwconv2.weight"))?,
                        Some(self.weight(&format!("{prefix}.pwconv2.bias"))?),
                    );
                update *= self.weight(&format!("{prefix}.gamma"))?;
                fused = residual + update.permute([0, 3, 1, 2]);
            }
            // Only slot zero is occupied in this single-eye trial. Match the
            // multiplex model's learned spatial embedding for the 15 empty
            // slots after mask/image fusion.
            if let Some(no_object) = self.weights.get("no_obj_embed_spatial") {
                fused += no_object
                    .narrow(0, 1, 15)
                    .sum_dim_intlist([0].as_slice(), false, fused.kind())
                    .view([1, 256, 1, 1]);
            }
            Ok(fused)
        }
    }

    struct RuntimePrompts {
        language_features: Tensor,
        language_mask: Tensor,
        img_ids: Tensor,
        text_ids: Vec<Tensor>,
    }

    fn load_runtime_prompts(
        path: &Path,
        device: Device,
        prompt_count: usize,
    ) -> Result<RuntimePrompts, String> {
        let mut tensors = Tensor::load_multi_with_device(path, Device::Cpu)
            .map_err(|error| format!("load SAM31 prompt bundle {}: {error}", path.display()))?
            .into_iter()
            .collect::<HashMap<_, _>>();
        let language_features = tensors
            .remove("language_features")
            .ok_or_else(|| "SAM31 prompt bundle lacks language_features".to_string())?;
        let language_mask = tensors
            .remove("language_mask")
            .ok_or_else(|| "SAM31 prompt bundle lacks language_mask".to_string())?;
        if prompt_count == 0
            || language_features.size() != [32, prompt_count as i64, 256]
            || language_mask.size() != [prompt_count as i64, 32]
        {
            return Err(format!(
                "SAM31 prompt bundle has incompatible shapes: features={:?} mask={:?}",
                language_features.size(),
                language_mask.size(),
            ));
        }
        let language_features = language_features.to_device_(device, Kind::BFloat16, false, false);
        let language_mask = language_mask.to_device_(device, Kind::Bool, false, false);
        let img_ids = Tensor::zeros([1], (Kind::Int64, device));
        let text_ids = (0..prompt_count)
            .map(|index| Tensor::from_slice(&[index as i64]).to_device(device))
            .collect();
        Ok(RuntimePrompts {
            language_features,
            language_mask,
            img_ids,
            text_ids,
        })
    }

    pub(super) fn export_native_outline_sequence(
        model_path: &Path,
        frames: &[Arc<RawFrame>],
    ) -> Result<serde_json::Value, String> {
        if frames.is_empty() || frames.iter().any(|f| f.width == 0 || f.height == 0
            || f.width.checked_mul(f.height) != Some(f.pixels.len())
            || f.width * FRAME_HEIGHT != f.height * FRAME_WIDTH) {
            return Err("outline export needs nonempty, same-aspect native RAW frames".into());
        }
        load_cuda_dispatch_library()?;
        configure_cuda_bfloat16_autocast();
        let _no_grad = tch::no_grad_guard();
        tch::autocast(true, || {
            let device = Device::Cuda(0);
            let mut module = CModule::load_on_device(model_path, device)
                .map_err(|e| format!("load outline detector: {e}"))?;
            module.set_eval();
            let bundle = std::env::var_os("BUTTERCUP_SAM31_PROMPT_BUNDLE")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data/models/sam31_semantic_prompts_cuda_bf16.pt"));
            let prompts = load_runtime_prompts(&bundle, device, SEMANTIC_PROMPT_COUNT)?;
            let regime = PreprocessRegime::configured_live()?;
            let staging = Tensor::zeros(
                [1, 3, FRAME_HEIGHT as i64, FRAME_WIDTH as i64],
                (Kind::Uint8, Device::Cpu),
            ).pin_memory(device);
            let ellipse_json = |e: Ellipse| serde_json::json!({
                "center": e.center, "major_radius": e.major_radius,
                "minor_radius": e.minor_radius, "angle": e.angle,
            });
            let mut cases = Vec::new();
            for (index, frame) in frames.iter().enumerate() {
                let started = Instant::now();
                write_preprocessed_filmstrip(std::slice::from_ref(frame), regime,
                    staging_bytes_len(&staging, FRAME_WIDTH * FRAME_HEIGHT * 3))?;
                let output = infer(&module, &staging, device, &prompts, OUTER_IRIS_PROMPT)?;
                let luma = raw_luma(std::slice::from_ref(frame)).into_iter().next()
                    .ok_or("outline export could not decode RAW luma")?;
                let mut candidates = Vec::new();
                for query in ranked_finite_query_indices(&output.scores).into_iter().take(12) {
                    let count = output.mask_width * output.mask_height;
                    let mask = &output.masks[query*count..(query+1)*count];
                    let area_fraction = mask.iter().filter(|&&v| v != 0).count() as f64 / count as f64;
                    if area_fraction == 0.0 { continue; }
                    let outline = native_outline_points(mask, output.mask_width, output.mask_height,
                        frame.width, frame.height);
                    let old = diagnostic_fit_single_frame_mask(mask, output.mask_width, output.mask_height)
                        .map(|r| model_review_in_source(r, frame.width));
                    let support = old.as_ref().map(|r| raw_ring_support(&luma, r.ellipse));
                    candidates.push(serde_json::json!({
                        "query": query, "semantic_score": output.scores[query],
                        "mask_area_fraction": area_fraction, "outline": outline,
                        "baseline_ellipse": old.as_ref().map(|r| ellipse_json(r.ellipse)),
                        "baseline_retained": old.as_ref().map(|r| r.retained_points.as_ref()),
                        "baseline_censored": old.as_ref().map(|r| r.flat_tire_points.as_ref()),
                        "baseline_raw_admitted": support.is_some_and(live_detector_raw_gate_passes),
                        "baseline_raw_score": support.map(|s| s.score),
                    }));
                }
                cases.push(serde_json::json!({
                    "sequence": frame.sequence, "timestamp_ns": frame.timestamp_ns,
                    "sensor_origin": [frame.sensor_x, frame.sensor_y],
                    "width": frame.width, "height": frame.height, "candidates": candidates,
                    "elapsed_ms": started.elapsed().as_millis() as u64,
                }));
                eprintln!("SAM outline export {}/{} sequence={}", index+1, frames.len(), frame.sequence);
            }
            Ok(serde_json::json!({
                "schema": "buttercup-native-sam-outlines-v1", "model": model_path,
                "configuration": live_configuration(),
                "contract": "Current-frame detector masks before contour rejection, using live native-ROI preprocessing and canonical outer prompt. No video-memory propagation, prediction seeds, human labels, or live state changes. Baseline is the production stateless mask fitter, not end-to-end live acceptance. Coordinates are native ROI pixels.",
                "cases": cases,
            }))
        })
    }

    pub(super) fn offline_semantic_suite(
        model_path: &Path,
        outer_prompt_bundle_path: &Path,
        prompt_bundle_path: &Path,
        prompt_count: usize,
        frames: &[Arc<RawFrame>],
        prompt_indices: &[usize],
        regime: PreprocessRegime,
    ) -> Result<OfflineSemanticSuite, String> {
        if frames.len() != HISTORY_FRAMES
            || frames.iter().any(|frame| {
                frame.width != FRAME_WIDTH
                    || frame.height != FRAME_HEIGHT
                    || frame.pixels.len() != FRAME_WIDTH * FRAME_HEIGHT
            })
        {
            return Err(format!(
                "offline SAM31 arc trial requires exactly {HISTORY_FRAMES} native {FRAME_WIDTH}x{FRAME_HEIGHT} RAW10 frames"
            ));
        }
        if prompt_indices.is_empty()
            || prompt_indices
                .iter()
                .any(|&prompt_index| prompt_index >= prompt_count)
        {
            return Err(format!(
                "offline SAM31 arc trial received invalid prompt indices {prompt_indices:?} for {prompt_count} prompts"
            ));
        }
        load_cuda_dispatch_library()?;
        configure_cuda_bfloat16_autocast();
        tch::autocast(true, || {
            let started = Instant::now();
            let device = Device::Cuda(0);
            let mut module = CModule::load_on_device(model_path, device)
                .map_err(|error| format!("load SAM31 graph {}: {error}", model_path.display()))?;
            module.set_eval();
            let outer_prompts =
                load_runtime_prompts(outer_prompt_bundle_path, device, SEMANTIC_PROMPT_COUNT)?;
            let prompts = load_runtime_prompts(prompt_bundle_path, device, prompt_count)?;
            // The detector is not perfectly invariant to the number of
            // language rows supplied alongside the selected text ID. Give
            // every experimental question the same six-row context as the
            // live path: preserve five canonical rows and replace only the
            // final row with that one trial feature. This makes a one-prompt
            // result truly independent of which other trials were requested.
            let isolated_prompts = (1..prompt_count)
                .map(|prompt_index| RuntimePrompts {
                    language_features: Tensor::cat(
                        &[
                            outer_prompts.language_features.narrow(1, 0, 5),
                            prompts.language_features.narrow(1, prompt_index as i64, 1),
                        ],
                        1,
                    ),
                    language_mask: Tensor::cat(
                        &[
                            outer_prompts.language_mask.narrow(0, 0, 5),
                            prompts.language_mask.narrow(0, prompt_index as i64, 1),
                        ],
                        0,
                    ),
                    img_ids: Tensor::zeros([1], (Kind::Int64, device)),
                    text_ids: (0..SEMANTIC_PROMPT_COUNT)
                        .map(|index| Tensor::from_slice(&[index as i64]).to_device(device))
                        .collect(),
                })
                .collect::<Vec<_>>();
            let staging = Tensor::zeros(
                [1, 3, FRAME_HEIGHT as i64, FILMSTRIP_WIDTH as i64],
                (Kind::Uint8, Device::Cpu),
            )
            .pin_memory(device);
            write_preprocessed_filmstrip(frames, regime, staging_bytes(&staging))?;

            let mut passes = Vec::with_capacity(prompt_indices.len());
            let mut outer_fit = None;
            let mut video_feature_shapes = None;
            let reflection_mask_first =
                std::env::var_os("BUTTERCUP_SAM31_REFLECTION_MASK_FIRST").is_some();
            let center_void_mask_first =
                std::env::var_os("BUTTERCUP_SAM31_CENTER_VOID_MASK_FIRST").is_some();
            let pupil_isolate_first =
                std::env::var_os("BUTTERCUP_SAM31_PUPIL_ISOLATE_FIRST").is_some();
            let pupil_inner_occlude_first =
                std::env::var_os("BUTTERCUP_SAM31_PUPIL_INNER_OCCLUDE_FIRST").is_some();
            // Control arm for pupil re-segmentation experiments: choose the
            // same dark compact proposal as the pink-mask arm, but leave the
            // detector input untouched between prompts.
            let pupil_resegment_control =
                std::env::var_os("BUTTERCUP_SAM31_PUPIL_RESEGMENT_CONTROL").is_some();
            for (step_position, &prompt_index) in prompt_indices.iter().enumerate() {
                let output = if prompt_index == OUTER_IRIS_PROMPT {
                    infer(&module, &staging, device, &outer_prompts, OUTER_IRIS_PROMPT)?
                } else {
                    infer(
                        &module,
                        &staging,
                        device,
                        &isolated_prompts[prompt_index - 1],
                        SEMANTIC_PROMPT_COUNT - 1,
                    )?
                };
                if video_feature_shapes.is_none() {
                    video_feature_shapes =
                        output
                            .video_features
                            .as_ref()
                            .map(|features| VideoFeatureShapes {
                                pyramid: features.pyramid.each_ref().map(Tensor::size),
                                decoder_queries: features.decoder_queries.size(),
                            });
                }
                let selected_query = if (center_void_mask_first
                    || pupil_isolate_first
                    || pupil_inner_occlude_first
                    || pupil_resegment_control)
                    && step_position == 0
                {
                    select_dark_center_void_query(&output, staging_bytes(&staging))
                } else if reflection_mask_first && step_position == 0 {
                    select_bright_small_reflection_query(&output, staging_bytes(&staging))
                } else if prompt_index == OUTER_IRIS_PROMPT {
                    let extracted = extract_latest_adapter_ellipse(&output);
                    if outer_fit.is_none() {
                        outer_fit = extracted.3;
                    }
                    extracted.2
                } else {
                    strongest_nonempty_latest_query(&output)
                };
                let adapter =
                    latest_tile_proposal_masks(ProposalAdapter::QuadRgb, &output, selected_query);
                if pupil_inner_occlude_first && step_position == 0 {
                    if let Some(query) = selected_query {
                        if let Some(mask) = adapter.masks.iter().find(|mask| mask.query == query) {
                            paint_latest_tile_mask_interior_pink(
                                staging_bytes(&staging),
                                &mask.pixels,
                                adapter.width,
                                adapter.height,
                                5,
                            )?;
                        }
                    }
                } else if pupil_isolate_first && step_position == 0 {
                    if let Some(query) = selected_query {
                        if let Some(mask) = adapter.masks.iter().find(|mask| mask.query == query) {
                            isolate_latest_tile_mask_on_pink(
                                staging_bytes(&staging),
                                &mask.pixels,
                                adapter.width,
                                adapter.height,
                            )?;
                        }
                    }
                } else if (reflection_mask_first || center_void_mask_first) && step_position == 0 {
                    if let Some(query) = selected_query {
                        if let Some(mask) = adapter.masks.iter().find(|mask| mask.query == query) {
                            paint_latest_tile_mask_pink(
                                staging_bytes(&staging),
                                &mask.pixels,
                                adapter.width,
                                adapter.height,
                                if center_void_mask_first { 5 } else { 1 },
                            )?;
                        }
                    }
                }
                passes.push(semantic_proposal_masks(prompt_index, adapter));
            }
            let source = frames
                .last()
                .ok_or_else(|| "offline SAM31 arc trial has no target frame".to_string())?;
            Ok(OfflineSemanticSuite {
                source_width: source.width,
                source_height: source.height,
                source_raw: Arc::clone(&source.pixels),
                outer_fit,
                passes,
                video_feature_shapes,
                elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            })
        })
    }

    pub(super) fn offline_video_feature_sequence(
        model_path: &Path,
        prompt_bundle_path: &Path,
        frames: &[Arc<RawFrame>],
        regime: PreprocessRegime,
    ) -> Result<OfflineVideoFeatureSequence, String> {
        if frames.len() < 2
            || frames.iter().any(|frame| {
                frame.width != FRAME_WIDTH
                    || frame.height != FRAME_HEIGHT
                    || frame.pixels.len() != FRAME_WIDTH * FRAME_HEIGHT
            })
        {
            return Err(format!(
                "SAM31 video feature trial requires at least two native {FRAME_WIDTH}x{FRAME_HEIGHT} RAW10 frames"
            ));
        }
        load_cuda_dispatch_library()?;
        configure_cuda_bfloat16_autocast();
        tch::autocast(true, || {
            let started = Instant::now();
            let device = Device::Cuda(0);
            let mut module = CModule::load_on_device(model_path, device).map_err(|error| {
                format!("load SAM31 feature graph {}: {error}", model_path.display())
            })?;
            module.set_eval();
            let video_prompt_count = std::env::var("BUTTERCUP_SAM31_VIDEO_PROMPT_COUNT")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(SEMANTIC_PROMPT_COUNT);
            let video_prompt_index = std::env::var("BUTTERCUP_SAM31_VIDEO_PROMPT_INDEX")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(OUTER_IRIS_PROMPT);
            if video_prompt_index >= video_prompt_count {
                return Err(format!(
                    "SAM31 video prompt index {video_prompt_index} is outside {video_prompt_count} prompts"
                ));
            }
            let prompts = load_runtime_prompts(prompt_bundle_path, device, video_prompt_count)?;
            let tracker_bundle = tracker_bundle_path();
            let mask_memory_encoder = NativeMaskMemoryEncoder::load(&tracker_bundle, device)?;

            // Keep the percentile/tone transform common across the sequence,
            // then split its exact planar bytes back into independent frames.
            // This prevents per-frame auto-levels from masquerading as motion.
            let mut filmstrip = vec![0u8; frames.len() * FRAME_WIDTH * FRAME_HEIGHT * 3];
            write_preprocessed_filmstrip(frames, regime, &mut filmstrip)?;
            let staging = Tensor::zeros(
                [1, 3, FRAME_HEIGHT as i64, FRAME_WIDTH as i64],
                (Kind::Uint8, Device::Cpu),
            )
            .pin_memory(device);
            let mut history = Vec::<PriorFrameFeatures>::with_capacity(7);
            let mut shapes = None;
            let mut mask_memory_shape = None;
            let mut temporal_conditioned_shape = None;
            let mut transitions = Vec::with_capacity(frames.len().saturating_sub(1));
            let mut review_frames = Vec::with_capacity(frames.len());
            let persistent_hot_pixels = hot_pixel_check_enabled()
                .then(|| persistent_raw10_hot_pixels(frames))
                .unwrap_or_default();
            let mut absent_streak = 0usize;
            let mut trusted_flat_tire_fit: Option<(OuterMaskFitReview, (u32, u32))> = None;
            let mut bootstrap_flat_tire_fit: Option<(OuterMaskFitReview, (u32, u32), usize)> = None;
            let flat_tire_scale_context = std::env::var("BUTTERCUP_SAM31_FLAT_TIRE_RADIUS_SUPPORT")
                .ok()
                .and_then(|support| {
                    let values = support
                        .split(',')
                        .map(str::trim)
                        .map(str::parse::<f64>)
                        .collect::<Result<Vec<_>, _>>()
                        .ok()?;
                    (values.len() == 3)
                        .then(|| OuterContourScaleContext::new(values[0], values[1], values[2]))
                        .flatten()
                });

            for frame_index in 0..frames.len() {
                extract_preprocessed_frame(
                    &filmstrip,
                    frame_index,
                    staging_bytes_len(&staging, FRAME_WIDTH * FRAME_HEIGHT * 3),
                )?;
                let output = infer(&module, &staging, device, &prompts, video_prompt_index)?;
                let selected_query = strongest_model_query(&output);
                let current = output.video_features.ok_or_else(|| {
                    "SAM31 model is detector-only; generate/use the native feature graph first"
                        .to_string()
                })?;
                if shapes.is_none() {
                    shapes = Some(VideoFeatureShapes {
                        pyramid: current.pyramid.each_ref().map(Tensor::size),
                        decoder_queries: current.decoder_queries.size(),
                    });
                }
                let mut carried_query = selected_query;
                let mut matched = None;
                if let Some(prior) = history.last() {
                    matched = prior.tracked_query.and_then(|query| {
                        best_decoder_query_match(
                            &prior.features.decoder_queries,
                            query,
                            &current.decoder_queries,
                        )
                    });
                    carried_query = matched.map(|matched| matched.0).or(selected_query);
                }
                // Preserve the first detector-conditioned frame and combine it
                // with the six nearest non-conditioning frames, matching the
                // official seven-memory video policy on long sequences.
                let mut selected_history = history.iter().rev().take(6).collect::<Vec<_>>();
                if let Some(conditioning) = history.first() {
                    if !selected_history
                        .iter()
                        .any(|prior| prior.frame_index == conditioning.frame_index)
                    {
                        selected_history.push(conditioning);
                    }
                }
                let memory_bank = selected_history
                    .into_iter()
                    .filter_map(|prior| {
                        prior.mask_memory.as_ref().map(|memory| {
                            (
                                &prior.features.pyramid[2],
                                memory,
                                frame_index - prior.frame_index,
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                let mut selected_pointer_history =
                    history.iter().rev().take(15).collect::<Vec<_>>();
                if let Some(conditioning) = history.first() {
                    if !selected_pointer_history
                        .iter()
                        .any(|prior| prior.frame_index == conditioning.frame_index)
                    {
                        selected_pointer_history.push(conditioning);
                    }
                }
                let pointer_bank = selected_pointer_history
                    .into_iter()
                    .filter_map(|prior| {
                        prior
                            .object_pointer
                            .as_ref()
                            .map(|pointer| (pointer, frame_index - prior.frame_index))
                    })
                    .collect::<Vec<_>>();
                let temporal_conditioned = (!memory_bank.is_empty())
                    .then(|| {
                        mask_memory_encoder.condition_with_memory_bank(
                            &current.pyramid[2],
                            &memory_bank,
                            &pointer_bank,
                        )
                    })
                    .transpose()?;
                if temporal_conditioned_shape.is_none() {
                    temporal_conditioned_shape = temporal_conditioned.as_ref().map(Tensor::size);
                }
                let tracker_decode = temporal_conditioned
                    .as_ref()
                    .map(|conditioned| {
                        mask_memory_encoder.propagation_mask_decode(
                            conditioned,
                            [&current.pyramid[0], &current.pyramid[1]],
                        )
                    })
                    .transpose()?;
                let tracker_selection = tracker_decode.as_ref().map(|decoded| {
                    let selected = decoded.iou_scores.argmax(-1, false).int64_value(&[0]) as usize;
                    (
                        selected,
                        decoded.object_score_logit.double_value(&[0]),
                        decoded.iou_scores.double_value(&[0, selected as i64]),
                    )
                });
                let detector_logits = carried_query.map(|query| {
                    output
                        .logits
                        .get(0)
                        .get(query as i64)
                        .unsqueeze(0)
                        .unsqueeze(0)
                });
                let detector_recondition_area_fraction = detector_logits.as_ref().map(|logits| {
                    let mask = binary_mask_bytes(logits);
                    mask.iter().filter(|&&value| value != 0).count() as f64
                        / mask.len().max(1) as f64
                });
                let tracker_raw_logits = tracker_selection.as_ref().map(|&(selected, _, _)| {
                    tracker_decode
                        .as_ref()
                        .unwrap()
                        .mask_logits
                        .get(0)
                        .get(selected as i64)
                        .unsqueeze(0)
                        .unsqueeze(0)
                });
                let tracker_candidate_area = tracker_raw_logits.as_ref().map(|logits| {
                    let mask = binary_mask_bytes(logits);
                    mask.iter().filter(|&&value| value != 0).count() as f64
                        / mask.len().max(1) as f64
                });
                let tracker_present = tracker_selection
                    .zip(tracker_candidate_area)
                    .map(|(selection, area)| selection.1 > 0.0 && (0.05..=0.50).contains(&area));
                match tracker_present {
                    Some(true) => absent_streak = 0,
                    Some(false) => absent_streak += 1,
                    None => {}
                }
                let reconditioned_from_detector = absent_streak >= 3
                    && detector_recondition_area_fraction
                        .is_some_and(|area| (0.05..=0.50).contains(&area));
                let tracker_logits = tracker_raw_logits.map(|logits| {
                    if tracker_present == Some(true) {
                        logits
                    } else {
                        Tensor::full_like(&logits, -1024.0)
                    }
                });
                let object_pointer = tracker_selection
                    .as_ref()
                    .map(|&(selected, score, _)| {
                        mask_memory_encoder.selected_object_pointer(
                            tracker_decode.as_ref().unwrap(),
                            selected,
                            score > 0.0 && tracker_present == Some(true),
                        )
                    })
                    .transpose()?;
                let memory_logits = if reconditioned_from_detector {
                    detector_logits.as_ref()
                } else {
                    tracker_logits.as_ref().or_else(|| {
                        detector_recondition_area_fraction
                            .is_some_and(|area| (0.05..=0.50).contains(&area))
                            .then_some(detector_logits.as_ref())
                            .flatten()
                    })
                };
                let mask_memory = memory_logits
                    .map(|logits| mask_memory_encoder.encode(&current.pyramid[2], logits))
                    .transpose()?;
                if mask_memory_shape.is_none() {
                    mask_memory_shape = mask_memory.as_ref().map(Tensor::size);
                }
                let admitted_mask_size = memory_logits.map(Tensor::size);
                let tracker_mask = memory_logits.map(binary_mask_bytes);
                let admitted_mask_width = admitted_mask_size
                    .as_ref()
                    .and_then(|shape| shape.get(3))
                    .copied()
                    .unwrap_or(0) as usize;
                let admitted_mask_height = admitted_mask_size
                    .as_ref()
                    .and_then(|shape| shape.get(2))
                    .copied()
                    .unwrap_or(0) as usize;
                if let Some(prior) = history.last() {
                    let pyramid_metrics = std::array::from_fn(|level| {
                        feature_agreement(&prior.features.pyramid[level], &current.pyramid[level])
                    });
                    let decoder = feature_agreement(
                        &prior.features.decoder_queries,
                        &current.decoder_queries,
                    );
                    let matched_mask_iou = prior.tracked_query.zip(carried_query).and_then(
                        |(first_query, second_query)| {
                            sensor_aligned_query_mask_iou(
                                &prior.masks,
                                prior.mask_width,
                                prior.mask_height,
                                first_query,
                                prior.sensor_origin,
                                &output.masks,
                                output.mask_width,
                                output.mask_height,
                                second_query,
                                (frames[frame_index].sensor_x, frames[frame_index].sensor_y),
                            )
                        },
                    );
                    let memory_metrics = prior
                        .mask_memory
                        .as_ref()
                        .zip(mask_memory.as_ref())
                        .map(|(first, second)| feature_agreement(first, second));
                    let temporal_metrics = prior
                        .temporal_conditioned
                        .as_ref()
                        .zip(temporal_conditioned.as_ref())
                        .map(|(first, second)| feature_agreement(first, second));
                    let temporal_update = temporal_conditioned
                        .as_ref()
                        .map(|conditioned| feature_agreement(&current.pyramid[2], conditioned));
                    let tracker_mask_iou = prior
                        .tracker_mask
                        .as_ref()
                        .zip(tracker_mask.as_ref())
                        .and_then(|(first, second)| {
                            sensor_aligned_query_mask_iou(
                                first,
                                288,
                                288,
                                0,
                                prior.sensor_origin,
                                second,
                                288,
                                288,
                                0,
                                (frames[frame_index].sensor_x, frames[frame_index].sensor_y),
                            )
                        });
                    let tracker_vs_detector = tracker_mask.as_ref().zip(carried_query).and_then(
                        |(tracker, detector_query)| {
                            sensor_aligned_query_mask_iou(
                                tracker,
                                288,
                                288,
                                0,
                                (frames[frame_index].sensor_x, frames[frame_index].sensor_y),
                                &output.masks,
                                output.mask_width,
                                output.mask_height,
                                detector_query,
                                (frames[frame_index].sensor_x, frames[frame_index].sensor_y),
                            )
                        },
                    );
                    let tracker_geometry = tracker_mask.as_ref().and_then(|mask| {
                        mask_geometry(
                            mask,
                            288,
                            288,
                            (frames[frame_index].sensor_x, frames[frame_index].sensor_y),
                        )
                    });
                    let prior_tracker_geometry = prior
                        .tracker_mask
                        .as_ref()
                        .and_then(|mask| mask_geometry(mask, 288, 288, prior.sensor_origin));
                    let tracker_radius_ratio = prior_tracker_geometry
                        .zip(tracker_geometry)
                        .and_then(|(first, second)| (first.1 > 0.0).then_some(second.1 / first.1));
                    let tracker_centroid_motion =
                        prior_tracker_geometry
                            .zip(tracker_geometry)
                            .map(|(first, second)| {
                                (second.2 .0 - first.2 .0).hypot(second.2 .1 - first.2 .1)
                            });
                    transitions.push(VideoFeatureTransition {
                        from_frame: frame_index - 1,
                        to_frame: frame_index,
                        pyramid_cosine: pyramid_metrics.map(|metric| metric.0),
                        pyramid_normalized_rms_change: pyramid_metrics.map(|metric| metric.1),
                        decoder_query_cosine: decoder.0,
                        decoder_query_normalized_rms_change: decoder.1,
                        source_selected_query: prior.tracked_query,
                        target_detector_query: selected_query,
                        matched_query: matched.map(|matched| matched.0),
                        matched_query_cosine: matched.map(|matched| matched.1),
                        matched_mask_iou_sensor: matched_mask_iou,
                        mask_memory_cosine: memory_metrics.map(|metrics| metrics.0),
                        mask_memory_normalized_rms_change: memory_metrics.map(|metrics| metrics.1),
                        temporal_conditioned_cosine: temporal_metrics.map(|metrics| metrics.0),
                        temporal_conditioned_normalized_rms_change: temporal_metrics
                            .map(|metrics| metrics.1),
                        temporal_update_cosine: temporal_update.map(|metrics| metrics.0),
                        temporal_update_normalized_rms_change: temporal_update
                            .map(|metrics| metrics.1),
                        tracker_object_score_logit: tracker_selection.map(|selection| selection.1),
                        tracker_object_present: tracker_present,
                        tracker_reconditioned_from_detector: reconditioned_from_detector,
                        detector_recondition_area_fraction,
                        tracker_selected_iou_score: tracker_selection.map(|selection| selection.2),
                        tracker_mask_iou_sensor: tracker_mask_iou,
                        tracker_vs_detector_mask_iou: tracker_vs_detector,
                        tracker_area_fraction: tracker_geometry.map(|geometry| geometry.0),
                        tracker_equivalent_radius_px: tracker_geometry.map(|geometry| geometry.1),
                        tracker_radius_ratio_from_prior: tracker_radius_ratio,
                        tracker_centroid_motion_sensor_px: tracker_centroid_motion,
                    });
                }
                if reconditioned_from_detector {
                    history.clear();
                    absent_streak = 0;
                }
                let candidate_fit =
                    tracker_mask
                        .as_ref()
                        .and_then(|mask| match flat_tire_scale_context {
                            Some(context) => fit_single_frame_mask_candidate_with_context(
                                mask,
                                admitted_mask_width,
                                admitted_mask_height,
                                context,
                            ),
                            None => {
                                tracker_fit_review(mask, admitted_mask_width, admitted_mask_height)
                            }
                        });
                let raw_contour_points = tracker_mask
                    .as_ref()
                    .map(|mask| {
                        single_frame_mask_contour(mask, admitted_mask_width, admitted_mask_height)
                    })
                    .unwrap_or_default();
                let source_origin = (frames[frame_index].sensor_x, frames[frame_index].sensor_y);
                let temporal_support = candidate_fit.as_ref().is_some_and(|fit| {
                    if let Some((prior, origin)) = trusted_flat_tire_fit.as_ref() {
                        return (distributed_contour_support(fit)
                            || partial_temporal_contour_support(fit, false))
                            && temporal_contour_support(
                                fit,
                                &frames[frame_index],
                                Some((prior, *origin)),
                            );
                    }
                    if !partial_temporal_contour_support(fit, true) {
                        bootstrap_flat_tire_fit = None;
                        return false;
                    }
                    let compatible =
                        bootstrap_flat_tire_fit
                            .as_ref()
                            .is_some_and(|(prior, prior_origin, _)| {
                                let current_center = (
                                    source_origin.0 as f64 + fit.ellipse.center.0,
                                    source_origin.1 as f64 + fit.ellipse.center.1,
                                );
                                let prior_center = (
                                    prior_origin.0 as f64 + prior.ellipse.center.0,
                                    prior_origin.1 as f64 + prior.ellipse.center.1,
                                );
                                let center_motion = (current_center.0 - prior_center.0)
                                    .hypot(current_center.1 - prior_center.1);
                                let major_ratio =
                                    fit.ellipse.major_radius / prior.ellipse.major_radius.max(1.0);
                                let minor_ratio =
                                    fit.ellipse.minor_radius / prior.ellipse.minor_radius.max(1.0);
                                center_motion <= prior.ellipse.major_radius * 0.18
                                    && (0.92..=1.08).contains(&major_ratio)
                                    && (0.72..=1.38).contains(&minor_ratio)
                            });
                    let streak = if compatible {
                        bootstrap_flat_tire_fit
                            .as_ref()
                            .map_or(1, |(_, _, streak)| streak.saturating_add(1))
                    } else {
                        1
                    };
                    bootstrap_flat_tire_fit = Some((fit.clone(), source_origin, streak));
                    streak >= 5
                });
                let accepted_fit = candidate_fit
                    .clone()
                    .filter(|_| flat_tire_scale_context.is_none() || temporal_support);
                if let Some(fit) = accepted_fit.as_ref() {
                    let ratio = fit.ellipse.minor_radius / fit.ellipse.major_radius.max(1.0);
                    if trusted_flat_tire_fit.is_none()
                        || (fit.retained_points.len() >= 70 && ratio >= 0.64)
                    {
                        trusted_flat_tire_fit = Some((fit.clone(), source_origin));
                        bootstrap_flat_tire_fit = None;
                    }
                }
                review_frames.push(OfflineVideoReviewFrame {
                    source: Arc::clone(&frames[frame_index]),
                    mask: tracker_mask.as_ref().map(|mask| Arc::new(mask.clone())),
                    mask_width: admitted_mask_width,
                    mask_height: admitted_mask_height,
                    fit: accepted_fit,
                    contour_hypothesis: candidate_fit,
                    raw_contour_points: Arc::new(raw_contour_points),
                    hot_pixels: Arc::new(
                        persistent_hot_pixels
                            .iter()
                            .filter_map(|pixel| {
                                let x = pixel.sensor_x.checked_sub(frames[frame_index].sensor_x)?
                                    as usize;
                                let y = pixel.sensor_y.checked_sub(frames[frame_index].sensor_y)?
                                    as usize;
                                (x < frames[frame_index].width && y < frames[frame_index].height)
                                    .then_some((x, y))
                            })
                            .collect(),
                    ),
                    propagated: frame_index != 0 && !reconditioned_from_detector,
                });
                history.push(PriorFrameFeatures {
                    frame_index,
                    sequence: frames[frame_index].sequence,
                    features: current,
                    tracked_query: carried_query,
                    masks: output.masks,
                    mask_width: output.mask_width,
                    mask_height: output.mask_height,
                    sensor_origin: (frames[frame_index].sensor_x, frames[frame_index].sensor_y),
                    mask_memory,
                    temporal_conditioned,
                    tracker_mask,
                    object_pointer: if reconditioned_from_detector {
                        None
                    } else {
                        object_pointer
                    },
                });
                if history.len() > 16 {
                    history.remove(1);
                }
            }

            Ok(OfflineVideoFeatureSequence {
                frame_count: frames.len(),
                feature_shapes: shapes
                    .ok_or_else(|| "SAM31 feature sequence was empty".to_string())?,
                mask_memory_shape,
                mask_memory_position_shape: Some(mask_memory_encoder.spatial_position().size()),
                temporal_conditioned_shape,
                transitions,
                review_frames,
                elapsed_ms: started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            })
        })
    }

    fn feature_agreement(first: &Tensor, second: &Tensor) -> (f64, f64) {
        let first = first.to_kind(Kind::Float);
        let second = second.to_kind(Kind::Float);
        let epsilon = 1e-12;
        let dot = (&first * &second).sum(Kind::Float).double_value(&[]);
        let first_energy = (&first * &first).mean(Kind::Float).double_value(&[]);
        let second_energy = (&second * &second).mean(Kind::Float).double_value(&[]);
        let cosine = dot
            / ((first_energy.sqrt() * second_energy.sqrt() * first.numel() as f64).max(epsilon));
        let delta_energy = (&first - &second)
            .pow_tensor_scalar(2.0)
            .mean(Kind::Float)
            .double_value(&[]);
        let normalized_rms_change =
            delta_energy.sqrt() / ((0.5 * (first_energy + second_energy)).sqrt().max(epsilon));
        (cosine.clamp(-1.0, 1.0), normalized_rms_change)
    }

    fn binary_mask_bytes(logits: &Tensor) -> Vec<u8> {
        let binary = logits
            .gt(0.0)
            .to_kind(Kind::Uint8)
            .to_device_(Device::Cpu, Kind::Uint8, false, false)
            .contiguous();
        let len = binary.numel();
        let mut bytes = vec![0u8; len];
        binary.copy_data_u8(&mut bytes, len);
        bytes
    }

    fn mask_geometry(
        mask: &[u8],
        width: usize,
        height: usize,
        sensor_origin: (u32, u32),
    ) -> Option<(f64, f64, (f64, f64))> {
        if mask.len() != width.checked_mul(height)? {
            return None;
        }
        let mut count = 0usize;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for (index, &value) in mask.iter().enumerate() {
            if value == 0 {
                continue;
            }
            count += 1;
            sum_x += (index % width) as f64 + 0.5;
            sum_y += (index / width) as f64 + 0.5;
        }
        if count == 0 {
            return Some((0.0, 0.0, (sensor_origin.0 as f64, sensor_origin.1 as f64)));
        }
        let area_fraction = count as f64 / (width * height) as f64;
        let native_area = area_fraction * (FRAME_WIDTH * FRAME_HEIGHT) as f64;
        let equivalent_radius = (native_area / std::f64::consts::PI).sqrt();
        let centroid = (
            sensor_origin.0 as f64 + sum_x / count as f64 * FRAME_WIDTH as f64 / width as f64,
            sensor_origin.1 as f64 + sum_y / count as f64 * FRAME_HEIGHT as f64 / height as f64,
        );
        Some((area_fraction, equivalent_radius, centroid))
    }

    fn best_decoder_query_match(
        first_layers: &Tensor,
        first_query: usize,
        second_layers: &Tensor,
    ) -> Option<(usize, f64)> {
        let first_shape = first_layers.size();
        if first_shape.len() != 4
            || first_shape[0] == 0
            || first_shape[1] != 1
            || first_query >= first_shape[2] as usize
        {
            return None;
        }
        let anchor = first_layers
            .get(first_shape[0] - 1)
            .get(0)
            .get(first_query as i64)
            .to_kind(Kind::Float);
        best_decoder_query_match_from_anchor(&anchor, second_layers)
    }

    fn decoder_query_anchor(layers: &Tensor, query: usize) -> Option<Tensor> {
        let shape = layers.size();
        if shape.len() != 4 || shape[0] == 0 || shape[1] != 1 || query >= shape[2] as usize {
            return None;
        }
        Some(
            layers
                .get(shape[0] - 1)
                .get(0)
                .get(query as i64)
                .to_kind(Kind::Float)
                .detach(),
        )
    }

    fn best_decoder_query_match_from_anchor(
        anchor: &Tensor,
        second_layers: &Tensor,
    ) -> Option<(usize, f64)> {
        let second_shape = second_layers.size();
        let anchor_shape = anchor.size();
        if anchor_shape.len() != 1
            || second_shape.len() != 4
            || second_shape[0] == 0
            || second_shape[1] != 1
            || anchor_shape[0] != second_shape[3]
        {
            return None;
        }
        let second = second_layers
            .get(second_shape[0] - 1)
            .get(0)
            .to_kind(Kind::Float);
        let anchor = anchor
            / anchor
                .pow_tensor_scalar(2.0)
                .sum(Kind::Float)
                .sqrt()
                .clamp_min(1e-12);
        let second_norm = second
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([1].as_slice(), true, Kind::Float)
            .sqrt()
            .clamp_min(1e-12);
        let similarities =
            second.matmul(&anchor.unsqueeze(1)).squeeze_dim(1) / second_norm.squeeze_dim(1);
        let (value, index) = similarities.max_dim(0, false);
        Some((index.int64_value(&[]) as usize, value.double_value(&[])))
    }

    #[allow(clippy::too_many_arguments)]
    fn sensor_aligned_query_mask_iou(
        first_masks: &[u8],
        first_width: usize,
        first_height: usize,
        first_query: usize,
        first_origin: (u32, u32),
        second_masks: &[u8],
        second_width: usize,
        second_height: usize,
        second_query: usize,
        second_origin: (u32, u32),
    ) -> Option<f64> {
        sensor_aligned_query_mask_iou_with_extent(
            first_masks,
            first_width,
            first_height,
            first_query,
            first_origin,
            second_masks,
            second_width,
            second_height,
            second_query,
            second_origin,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn sensor_aligned_query_mask_iou_with_extent(
        first_masks: &[u8],
        first_width: usize,
        first_height: usize,
        first_query: usize,
        first_origin: (u32, u32),
        second_masks: &[u8],
        second_width: usize,
        second_height: usize,
        second_query: usize,
        second_origin: (u32, u32),
        extent: (usize, usize),
    ) -> Option<f64> {
        let first_plane = first_width.checked_mul(first_height)?;
        let second_plane = second_width.checked_mul(second_height)?;
        let first_start = first_query.checked_mul(first_plane)?;
        let second_start = second_query.checked_mul(second_plane)?;
        let first = first_masks.get(first_start..first_start + first_plane)?;
        let second = second_masks.get(second_start..second_start + second_plane)?;
        let min_x = first_origin.0.min(second_origin.0);
        let min_y = first_origin.1.min(second_origin.1);
        let max_x = (first_origin.0 + extent.0 as u32).max(second_origin.0 + extent.0 as u32);
        let max_y = (first_origin.1 + extent.1 as u32).max(second_origin.1 + extent.1 as u32);
        let mut intersection = 0usize;
        let mut union = 0usize;
        for sensor_y in min_y..max_y {
            for sensor_x in min_x..max_x {
                let sample = |mask: &[u8], width: usize, height: usize, origin: (u32, u32)| {
                    if sensor_x < origin.0
                        || sensor_y < origin.1
                        || sensor_x >= origin.0 + extent.0 as u32
                        || sensor_y >= origin.1 + extent.1 as u32
                    {
                        return false;
                    }
                    let native_x = (sensor_x - origin.0) as usize;
                    let native_y = (sensor_y - origin.1) as usize;
                    let mask_x = (native_x * width / extent.0).min(width - 1);
                    let mask_y = (native_y * height / extent.1).min(height - 1);
                    mask[mask_y * width + mask_x] != 0
                };
                let first_on = sample(first, first_width, first_height, first_origin);
                let second_on = sample(second, second_width, second_height, second_origin);
                intersection += usize::from(first_on && second_on);
                union += usize::from(first_on || second_on);
            }
        }
        (union != 0).then_some(intersection as f64 / union as f64)
    }

    #[derive(Default)]
    struct LiveTrackerState {
        history: Vec<PriorFrameFeatures>,
        pupil_history: PupilContourHistory,
        frame_index: usize,
        last_input: Option<LiveTrackerInput>,
        consecutive_misses: u8,
        /// Frozen decoder-query embedding from the last detector mask that
        /// passed the untouched-RAW ring gate. Propagation never replaces it.
        decoder_query_anchor: Option<Tensor>,
    }

    impl LiveTrackerState {
        fn prepare(&mut self, input: LiveTrackerInput) {
            if live_tracker_requires_reset(self.last_input, input) {
                // Pixel-addressed features must reset on a crop move. Pupil
                // size and offset are iris-relative, so an overlapping crop
                // relocation of the same tracked eye must not erase them.
                // Identity/prompt/size/time changes still reset everything.
                let pupil_history = if pupil_history_survives_roi_relocation(self.last_input, input) {
                    std::mem::take(&mut self.pupil_history)
                } else {
                    PupilContourHistory::default()
                };
                *self = Self {
                    last_input: Some(input),
                    pupil_history,
                    ..Self::default()
                };
            } else {
                self.last_input = Some(input);
            }
        }

        fn clear_temporal_memory(&mut self) {
            self.history.clear();
            self.frame_index = 0;
            self.consecutive_misses = 0;
            self.decoder_query_anchor = None;
        }

        fn record_processed_miss(&mut self) {
            self.frame_index = self.frame_index.saturating_add(1);
            self.record_miss();
        }

        fn record_miss(&mut self) {
            let (misses, release) = next_live_hold_miss(self.consecutive_misses);
            self.consecutive_misses = misses;
            if release {
                self.clear_temporal_memory();
            }
        }
    }

    fn tracker_fit_review(mask: &[u8], width: usize, height: usize) -> Option<OuterMaskFitReview> {
        if mask.len() != width.checked_mul(height)? {
            return None;
        }
        let wide_width = width * HISTORY_FRAMES;
        let mut wide = vec![0u8; wide_width * height];
        for y in 0..height {
            let source = y * width;
            let destination = y * wide_width + (HISTORY_FRAMES - 1) * width;
            wide[destination..destination + width].copy_from_slice(&mask[source..source + width]);
        }
        fit_mask_component_review(&wide, wide_width, height, HISTORY_FRAMES - 1)
    }

    struct LiveTemporalOuterProposal {
        semantic: SemanticProposalMasks,
        outer_fit: Option<OuterMaskFitReview>,
        outer_support: RawRingSupport,
        pupil_fit: Option<PupilVoidFitReview>,
    }

    struct LiveSelectedMask {
        logits: Tensor,
        mask: Vec<u8>,
        fit: OuterMaskFitReview,
        query: Option<usize>,
        score: f32,
        outer_support: RawRingSupport,
    }

    fn live_temporal_outer_proposal(
        module: &CModule,
        staging: &Tensor,
        device: Device,
        prompts: &RuntimePrompts,
        encoder: &NativeMaskMemoryEncoder,
        state: &mut LiveTrackerState,
        source: &RawFrame,
        tracking_epoch: u64,
        prompt_generation: u64,
        current_luma: &FloatImage,
        request_pupil: bool,
        shared_pupil_prompt: bool,
    ) -> Result<LiveTemporalOuterProposal, String> {
        state.prepare(LiveTrackerInput {
            tracking_epoch,
            prompt_generation,
            sequence: source.sequence,
            timestamp_ns: source.timestamp_ns,
            sensor_origin: (source.sensor_x, source.sensor_y),
            width: source.width,
            height: source.height,
        });
        let output = infer(module, staging, device, prompts, OUTER_IRIS_PROMPT)?;
        let current = output.video_features.ok_or_else(|| {
            "SAM31 live video tracking requires the native feature graph".to_string()
        })?;
        let history_exists = !state.history.is_empty();
        let matched_query = state.decoder_query_anchor.as_ref().and_then(|anchor| {
            best_decoder_query_match_from_anchor(anchor, &current.decoder_queries)
        });
        let carried_query = matched_query.map(|matched| matched.0);
        // Once an object has entered video memory, recovery remains attached
        // to the last RAW-validated detector embedding. Propagated slots and
        // frame-local global maxima never advance that stable identity.
        let ranked_bootstrap = ranked_finite_query_indices(
            &output.scores[..output.query_count.min(output.scores.len())],
        );
        let recovery_queries =
            choose_live_recovery_queries(!history_exists, &ranked_bootstrap, matched_query);

        let mut selected_history = state.history.iter().rev().take(6).collect::<Vec<_>>();
        if let Some(conditioning) = state.history.first() {
            if !selected_history
                .iter()
                .any(|prior| prior.frame_index == conditioning.frame_index)
            {
                selected_history.push(conditioning);
            }
        }
        let memory_bank = selected_history
            .into_iter()
            .filter_map(|prior| {
                prior.mask_memory.as_ref().map(|memory| {
                    (
                        &prior.features.pyramid[2],
                        memory,
                        state.frame_index - prior.frame_index,
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut selected_pointer_history = state.history.iter().rev().take(15).collect::<Vec<_>>();
        if let Some(conditioning) = state.history.first() {
            if !selected_pointer_history
                .iter()
                .any(|prior| prior.frame_index == conditioning.frame_index)
            {
                selected_pointer_history.push(conditioning);
            }
        }
        let pointer_bank = selected_pointer_history
            .into_iter()
            .filter_map(|prior| {
                prior
                    .object_pointer
                    .as_ref()
                    .map(|pointer| (pointer, state.frame_index - prior.frame_index))
            })
            .collect::<Vec<_>>();
        let temporal_conditioned = (!memory_bank.is_empty())
            .then(|| {
                encoder.condition_with_memory_bank(&current.pyramid[2], &memory_bank, &pointer_bank)
            })
            .transpose()?;
        let tracker_decode = temporal_conditioned
            .as_ref()
            .map(|conditioned| {
                encoder.propagation_mask_decode(
                    conditioned,
                    [&current.pyramid[0], &current.pyramid[1]],
                )
            })
            .transpose()?;
        let tracker_selection = tracker_decode.as_ref().map(|decoded| {
            let selected = decoded.iou_scores.argmax(-1, false).int64_value(&[0]) as usize;
            (
                selected,
                decoded.object_score_logit.double_value(&[0]),
                decoded.iou_scores.double_value(&[0, selected as i64]),
            )
        });
        let tracker_raw_logits = tracker_selection.map(|(selected, _, _)| {
            tracker_decode
                .as_ref()
                .unwrap()
                .mask_logits
                .get(0)
                .get(selected as i64)
                .unsqueeze(0)
                .unsqueeze(0)
        });
        let tracker_mask = tracker_raw_logits.as_ref().map(binary_mask_bytes);
        let tracker_area = tracker_mask.as_ref().map(|mask| {
            mask.iter().filter(|&&value| value != 0).count() as f64 / mask.len().max(1) as f64
        });
        let tracker_dimensions = tracker_raw_logits.as_ref().map(|logits| {
            let shape = logits.size();
            (shape[3] as usize, shape[2] as usize)
        });
        let tracker_fit = tracker_mask
            .as_ref()
            .zip(tracker_dimensions)
            .and_then(|(mask, (width, height))| tracker_fit_review(mask, width, height));
        let tracker_prior_iou = state.history.last().and_then(|prior| {
            prior
                .tracker_mask
                .as_ref()
                .zip(tracker_mask.as_ref())
                .zip(tracker_dimensions)
                .and_then(|((prior_mask, current_mask), (width, height))| {
                    sensor_aligned_query_mask_iou_with_extent(
                        prior_mask,
                        width,
                        height,
                        0,
                        prior.sensor_origin,
                        current_mask,
                        width,
                        height,
                        0,
                        (source.sensor_x, source.sensor_y),
                        (source.width, source.height),
                    )
                })
        });
        let tracker_healthy = live_propagation_is_healthy(
            tracker_selection.is_some_and(|selection| selection.1 > 0.0),
            tracker_area,
            tracker_fit.is_some(),
            history_exists,
            tracker_prior_iou,
        );

        // A bootstrap searches every finite-scored slot in descending model
        // order. Anchored recovery deliberately supplies at most one slot.
        // Keep the strongest geometry-plausible RAW failure for diagnostics,
        // but only a RAW-passing detector candidate may condition memory.
        let mut diagnostic_detector = None::<LiveSelectedMask>;
        let mut raw_valid_detector = None::<LiveSelectedMask>;
        let audit_candidates = std::env::var_os("BUTTERCUP_SAM31_CANDIDATE_AUDIT").is_some();
        let mut raw_candidates_seen = 0;
        if !tracker_healthy {
            for query in recovery_queries {
                let Some(&score) = output.scores.get(query) else {
                    continue;
                };
                let logits = output
                    .logits
                    .get(0)
                    .get(query as i64)
                    .unsqueeze(0)
                    .unsqueeze(0);
                let mask = binary_mask_bytes(&logits);
                let area = Some(
                    mask.iter().filter(|&&value| value != 0).count() as f64
                        / mask.len().max(1) as f64,
                );
                let fit = tracker_fit_review(&mask, output.mask_width, output.mask_height);
                if !live_detector_candidate_is_plausible(score, area, fit.is_some()) {
                    continue;
                }
                let fit = fit.expect("a plausible detector candidate has a fit");
                let outer_support = raw_ring_support(
                    current_luma,
                    model_ellipse_in_source(fit.ellipse, source.width),
                );
                let candidate = LiveSelectedMask {
                    logits,
                    mask,
                    fit,
                    query: Some(query),
                    score,
                    outer_support,
                };
                if audit_candidates {
                    eprintln!("SAM31_DETECTOR_CANDIDATE {}", serde_json::json!({
                        "sequence":source.sequence,"query":query,"score":score,"area_fraction":area,
                        "center":candidate.fit.ellipse.center,"major_radius":candidate.fit.ellipse.major_radius,
                        "minor_radius":candidate.fit.ellipse.minor_radius,"angle":candidate.fit.ellipse.angle,
                        "raw_score":outer_support.score,"raw_positive_fraction":outer_support.positive_fraction,
                        "raw_strong_sectors":outer_support.strong_sectors,
                        "raw_admitted":live_memory_commit_allowed(LiveMemorySource::Detector,outer_support)}));
                }
                if live_memory_commit_allowed(LiveMemorySource::Detector, outer_support) {
                    raw_candidates_seen += 1;
                    let replace = raw_valid_detector.as_ref().is_none_or(|prior|
                        outer_limbus_candidate_supersedes(candidate.fit.ellipse,candidate.outer_support.score,candidate.score,
                            prior.fit.ellipse,prior.outer_support.score,prior.score));
                    if replace { raw_valid_detector = Some(candidate); }
                    // Recovery has one identity-anchored query. Only cold
                    // acquisition compares a bounded set of RAW-valid masks.
                    if legacy_outer_selection() || (!audit_candidates && raw_candidates_seen >= 8) { break; }
                    continue;
                }
                if diagnostic_detector.is_none() {
                    diagnostic_detector = Some(candidate);
                }
            }
        }
        let raw_valid_detector_query = raw_valid_detector
            .as_ref()
            .and_then(|candidate| candidate.query);
        let update = choose_live_temporal_update(tracker_healthy, raw_valid_detector_query);
        let object_pointer = tracker_selection
            .filter(|_| matches!(update, LiveTemporalUpdate::Propagate))
            .map(|(selected, score, _)| {
                encoder.selected_object_pointer(
                    tracker_decode.as_ref().unwrap(),
                    selected,
                    score > 0.0,
                )
            })
            .transpose()?;
        let recondition =
            matches!(update, LiveTemporalUpdate::ConditionFromDetector(_)) && history_exists;
        let (selected, commit_memory) = match update {
            LiveTemporalUpdate::Propagate => {
                let fit = tracker_fit.expect("a healthy tracker has plausible limbus geometry");
                let outer_support = raw_ring_support(
                    current_luma,
                    model_ellipse_in_source(fit.ellipse, source.width),
                );
                (
                    LiveSelectedMask {
                        logits: tracker_raw_logits.expect("a healthy tracker has mask logits"),
                        mask: tracker_mask.expect("a healthy tracker has a binary mask"),
                        fit,
                        query: carried_query,
                        score: tracker_selection
                            .map(|selection| selection.2 as f32)
                            .unwrap_or_default(),
                        outer_support,
                    },
                    live_memory_commit_allowed(LiveMemorySource::Propagation, outer_support),
                )
            }
            LiveTemporalUpdate::ConditionFromDetector(query) => {
                let candidate = raw_valid_detector
                    .take()
                    .expect("a RAW-valid detector query has a candidate");
                debug_assert_eq!(candidate.query, Some(query));
                (candidate, true)
            }
            LiveTemporalUpdate::HoldLastConditioning => {
                let Some(candidate) = diagnostic_detector.take() else {
                    state.record_processed_miss();
                    return Err(format!(
                        "SAM31 video tracker held prior memory; propagated area {:?}/fit={} and no RAW-plausible detector geometry was available",
                        tracker_area,
                        tracker_fit.is_some(),
                    ));
                };
                (candidate, false)
            }
        };
        let mask_width = selected.logits.size()[3] as usize;
        let mask_height = selected.logits.size()[2] as usize;
        let proposal = ProposalMask {
            query: 0,
            score: selected.score,
            boundary_pixels: Arc::new(binary_mask_boundary_indices(
                &selected.mask,
                mask_width,
                mask_height,
            )),
            pixels: Arc::new(selected.mask.clone()),
        };
        let trace_area = selected.mask.iter().filter(|&&value| value != 0).count() as f64
            / selected.mask.len().max(1) as f64;
        let outer = model_ellipse_in_source(selected.fit.ellipse, source.width);
        let pupil_prior = state.pupil_history.prior(source.timestamp_ns);
        let semantic_requested = request_pupil && enabled_env_flag("BUTTERCUP_SAM31_SEMANTIC_PUPIL", true);
        let pupil_inference = if semantic_requested {
            let shared = shared_pupil_prompt;
            let result = if shared {
                infer_from_features(module, &current, prompts, PUPIL_DISK_PROMPT)
            } else {
                infer(module, staging, device, prompts, PUPIL_DISK_PROMPT)
            };
            if shared && enabled_env_flag("BUTTERCUP_SAM31_VERIFY_SHARED_PROMPT", false) {
                if let Ok(ref cached) = result {
                    let full = infer(module, staging, device, prompts, PUPIL_DISK_PROMPT)?;
                    let score_error = cached.scores.iter().zip(&full.scores)
                        .map(|(left, right)| (left - right).abs()).fold(0.0f32, f32::max);
                    let mask_differences = cached.masks.iter().zip(&full.masks).filter(|(left, right)| left != right).count();
                    let logit_error = (&cached.logits - &full.logits).abs().max().double_value(&[]);
                    eprintln!("SAM31_SHARED_PROMPT_PARITY {}", serde_json::json!({
                        "sequence":source.sequence,"score_max_abs":score_error,
                        "logit_max_abs":logit_error,"mask_different_pixels":mask_differences,
                        "mask_bytes":cached.masks.len()}));
                }
            }
            // The pupil is optional. Its inference failure must never discard
            // a healthy independently admitted limbus or its video memory.
            match result {
                Ok(output) => Some(output),
                Err(error) => { eprintln!("SAM31 optional pupil inference unavailable: {error}"); None }
            }
        } else { None };
        if commit_memory {
            // Encoding can fail; do it before clearing a prior conditioning
            // transaction so a runtime error cannot partially replace state.
            let mask_memory = Some(encoder.encode(&current.pyramid[2], &selected.logits)?);
            let refreshed_anchor = if let LiveTemporalUpdate::ConditionFromDetector(query) = update
            {
                Some(
                    decoder_query_anchor(&current.decoder_queries, query).ok_or_else(|| {
                        "SAM31 video detector conditioning lacked a decoder-query anchor"
                            .to_string()
                    })?,
                )
            } else {
                None
            };
            if matches!(update, LiveTemporalUpdate::ConditionFromDetector(_)) {
                // The selected detector mask already passed the RAW ring gate.
                // Only this transaction may replace the stable identity.
                state.clear_temporal_memory();
                state.decoder_query_anchor = refreshed_anchor;
            }
            state.history.push(PriorFrameFeatures {
                frame_index: state.frame_index,
                sequence: source.sequence,
                features: current,
                tracked_query: selected.query,
                masks: output.masks,
                mask_width: output.mask_width,
                mask_height: output.mask_height,
                sensor_origin: (source.sensor_x, source.sensor_y),
                mask_memory,
                temporal_conditioned,
                tracker_mask: Some(selected.mask.clone()),
                object_pointer: if matches!(update, LiveTemporalUpdate::Propagate) {
                    object_pointer
                } else {
                    None
                },
            });
            if state.history.len() > 16 {
                state.history.remove(1);
            }
            state.frame_index = state.frame_index.saturating_add(1);
            if live_committed_frame_counts_as_miss(
                if matches!(update, LiveTemporalUpdate::Propagate) {
                    LiveMemorySource::Propagation
                } else {
                    LiveMemorySource::Detector
                },
                selected.outer_support,
            ) {
                // Keep short occlusion continuity as requested, but do not
                // let a geometrically healthy yet permanently RAW-invalid
                // propagation suppress strict reacquisition forever.
                state.record_miss();
            } else {
                state.consecutive_misses = 0;
            }
        } else {
            // The diagnostic detector mask failed the RAW gate and must not
            // disturb memory. Three such processed misses release the stale
            // identity so the next frame performs a strict ranked bootstrap.
            state.record_processed_miss();
        }
        if std::env::var_os("BUTTERCUP_SAM31_VIDEO_TRACE").is_some() {
            eprintln!(
                "SAM31_VIDEO sequence={} history={} present={:?} recondition={} committed={} misses={} area={:.4} fit={}",
                source.sequence,
                state.history.len(),
                Some(tracker_healthy),
                recondition,
                commit_memory,
                state.consecutive_misses,
                trace_area,
                true,
            );
        }
        let semantic_pupil = if let Some(pupil_output) = pupil_inference {
            let glare_ceiling = pupil_iris_luma_ceiling(current_luma, outer);
            let mut best = None::<PupilVoidFitReview>;
            for query in ranked_finite_query_indices(&pupil_output.scores).into_iter().take(12) {
                let start = query * pupil_output.mask_width * pupil_output.mask_height;
                let mask = &pupil_output.masks[start..start+pupil_output.mask_width*pupil_output.mask_height];
                let component = (0..source.width*source.height).filter(|&index| {
                    let x=index%source.width; let y=index/source.width;
                    let mx=x*pupil_output.mask_width/source.width;
                    let my=y*pupil_output.mask_height/source.height;
                    mask[my*pupil_output.mask_width+mx] != 0
                        && ellipse_coordinate((x as f64,y as f64),outer)<=0.92
                }).collect::<Vec<_>>();
                if component.len()<100 { continue; }
                let points=component.iter().map(|&i|((i%source.width) as f64,(i/source.width) as f64)).collect::<Vec<_>>();
                let Some(reference)=moments_ellipse(&points) else {continue;};
                let Some(ellipse)=deflattened_pupil_component(&component,source.width,source.height,reference) else {continue;};
                if std::env::var_os("BUTTERCUP_SAM31_PUPIL_CANDIDATE_AUDIT").is_some() {
                    eprintln!("SAM31_PUPIL_CANDIDATE {}",serde_json::json!({
                        "sequence":source.sequence,"query":query,"semantic_score":pupil_output.scores[query],
                        "ellipse":{"center":ellipse.center,"major_radius":ellipse.major_radius,
                            "minor_radius":ellipse.minor_radius,"angle":ellipse.angle},
                        "geometry_admitted":pupil_ellipse_plausible(ellipse,outer),
                        "raw_support":format!("{:?}",raw_ring_support_below_ceiling(current_luma,ellipse,Some(glare_ceiling)))}));
                }
                if !pupil_ellipse_plausible(ellipse,outer)
                    || (source.pupil_component_seed.is_none()
                        && ellipse_coordinate(ellipse.center,outer)>MAX_UNPROMPTED_PUPIL_CENTER_OFFSET)
                    || pupil_prior.is_some_and(|prior|!prior.admits(ellipse,outer)) {continue;}
                let support=raw_ring_support_below_ceiling(current_luma,ellipse,Some(glare_ceiling));
                if !pupil_raw_support_is_sufficient(support) {continue;}
                if best.is_none_or(|b|support.score>b.raw_support.score) {
                    best=Some(PupilVoidFitReview {ellipse,raw_support:support});
                }
                // First-qualified semantic ranking is useful for latency
                // experiments, but the corpus retained more stable pupils
                // when qualified queries compete on untouched RAW support.
                if enabled_env_flag("BUTTERCUP_SAM31_PUPIL_RANK_FIRST", false) { break; }
            }
            if enabled_env_flag("BUTTERCUP_SAM31_VIDEO_TRACE", false)
                || enabled_env_flag("BUTTERCUP_SAM31_PUPIL_CANDIDATE_AUDIT", false)
            {
                eprintln!("SAM31_SEMANTIC_PUPIL sequence={} fit={best:?}",source.sequence);
            }
            best
        } else {None};
        let component = if semantic_requested { None } else {
            fit_inner_pupil_void_conditioned(current_luma,outer,source.pupil_component_seed,
                pupil_prior,&mut PupilFitDiagnostics::default())
                .map(|(ellipse,raw_support)|PupilVoidFitReview {ellipse,raw_support})
        };
        let recovered = if semantic_pupil.is_some() { None } else {
            pupil_prior.and_then(|prior|refit_pupil_from_prior(current_luma,outer,prior))
        };
        let pupil_selection = choose_pupil_observation(
            semantic_requested, semantic_pupil, component, recovered, outer, pupil_prior,
        );
        let pupil_fit = pupil_selection.map(|(fit, _)| fit);
        if live_detector_raw_gate_passes(selected.outer_support) {
            // A guided RAW recovery may support this exposure, but cannot
            // teach its own size/offset back into the prior. Only independent
            // current SAM masks or RAW components refresh contour history.
            if let Some((pupil, true)) = pupil_selection {
                state.pupil_history.observe(source.timestamp_ns, pupil.ellipse, outer);
            }
        }
        Ok(LiveTemporalOuterProposal {
            pupil_fit,
            semantic: SemanticProposalMasks {
                prompt_index: OUTER_IRIS_PROMPT,
                width: mask_width,
                height: mask_height,
                selected_query: Some(0),
                masks: vec![proposal],
            },
            outer_fit: Some(selected.fit),
            outer_support: selected.outer_support,
        })
    }

    pub(super) fn worker(
        model_path: PathBuf,
        prompt_bundle_path: PathBuf,
        request: Receiver<WorkerRequest>,
        results: SyncSender<OuterResult>,
        proposal_masks: SyncSender<Arc<ProposalMasks>>,
        status: Arc<Mutex<StatusSnapshot>>,
        stop: Arc<AtomicBool>,
    ) {
        if let Err(error) = load_cuda_dispatch_library() {
            update_status(&status, "error", &error);
            return;
        }
        let regime = match PreprocessRegime::configured_live() {
            Ok(regime) => regime,
            Err(error) => { update_status(&status, "error", &error); return; }
        };
        eprintln!("SAM31 live preprocessing: {}", regime.label());
        configure_cuda_bfloat16_autocast();
        // Autocast is thread-local. Keep one guard around the worker lifetime
        // instead of toggling deprecated LibTorch state around every query.
        tch::autocast(true, || {
            let mut module: Option<CModule> = None;
            let mut prompts: Option<RuntimePrompts> = None;
            let mut tracker_encoder: Option<NativeMaskMemoryEncoder> = None;
            let mut tracker_states = HashMap::<usize, LiveTrackerState>::new();
            let device = Device::Cuda(0);
            let mut native_staging: Option<Tensor> = None;
            let mut warmed = false;
            let mut shared_pupil_prompt = false;
            while let Ok(request) = request.recv() {
                if stop.load(AtomicOrdering::Acquire) {
                    break;
                }
                let batch = match request {
                    WorkerRequest::Batch(batch) => batch,
                    WorkerRequest::ReloadPrompts(path) => {
                        update_status(&status, "loading", "loading custom SAM prompt embedding");
                        match load_runtime_prompts(&path, device, SEMANTIC_PROMPT_COUNT) {
                            Ok(loaded) => {
                                prompts = Some(loaded);
                                tracker_states.clear();
                                update_status(
                                    &status,
                                    "idle",
                                    "custom outer-iris prompt active; waiting for RAW10 history",
                                );
                            }
                            Err(error) => update_status(&status, "error", &error),
                        }
                        continue;
                    }
                };
                let started = Instant::now();
                if module.is_none() {
                    update_status(&status, "loading", "loading promptable SAM3.1 graph");
                    match CModule::load_on_device(&model_path, device) {
                        Ok(mut loaded) => {
                            loaded.set_eval();
                            module = Some(loaded);
                        }
                        Err(error) => {
                            update_status(&status, "error", &format!("load SAM31 graph: {error}"));
                            break;
                        }
                    }
                }
                if prompts.is_none() {
                    match load_runtime_prompts(&prompt_bundle_path, device, SEMANTIC_PROMPT_COUNT) {
                        Ok(loaded) => prompts = Some(loaded),
                        Err(error) => {
                            update_status(&status, "error", &error);
                            break;
                        }
                    }
                }
                if tracker_encoder.is_none() {
                    let tracker_bundle = tracker_bundle_path();
                    match NativeMaskMemoryEncoder::load(&tracker_bundle, device) {
                        Ok(loaded) => tracker_encoder = Some(loaded),
                        Err(error) => {
                            update_status(&status, "error", &error);
                            break;
                        }
                    }
                }
                if native_staging.is_none() {
                    native_staging = Some(
                        Tensor::zeros(
                            [1, 3, FRAME_HEIGHT as i64, FRAME_WIDTH as i64],
                            (Kind::Uint8, Device::Cpu),
                        )
                        .pin_memory(device),
                    );
                }
                if !warmed {
                    update_status(
                        &status,
                        "warming",
                        "warming promptable graph before publishing a boundary",
                    );
                    let warmup = write_preprocessed_filmstrip(
                        &batch.frames[batch.frames.len() - 1..],
                        regime,
                        staging_bytes_len(
                            native_staging.as_ref().unwrap(),
                            FRAME_WIDTH * FRAME_HEIGHT * 3,
                        ),
                    )
                    .and_then(|_| {
                        infer(
                            module.as_ref().unwrap(),
                            native_staging.as_ref().unwrap(),
                            device,
                            prompts.as_ref().unwrap(),
                            OUTER_IRIS_PROMPT,
                        )
                        .map(|output| {
                            if enabled_env_flag("BUTTERCUP_SAM31_SHARED_FEATURE_PROMPT", true) {
                                if let Some(features) = output.video_features.as_ref() {
                                    match infer_from_features(
                                        module.as_ref().unwrap(), features, prompts.as_ref().unwrap(), PUPIL_DISK_PROMPT,
                                    ) {
                                        Ok(_) => shared_pupil_prompt = true,
                                        Err(_) => eprintln!("SAM31 graph has no usable shared-prompt method; using independent prompt inference"),
                                    }
                                }
                            }
                        })
                    });
                    if let Err(error) = warmup {
                        update_status(&status, "error", &format!("SAM31 warmup: {error}"));
                        break;
                    }
                    warmed = true;
                }
                update_status(
                    &status,
                    "running",
                    &format!(
                        "streaming video memory solving {} from the latest RAW frame",
                        batch.target.label()
                    ),
                );
                let current_luma = raw_luma(&batch.frames[batch.frames.len() - 1..])
                    .into_iter()
                    .next();
                let video_outer = (|| {
                    write_preprocessed_filmstrip(
                        &batch.frames[batch.frames.len() - 1..],
                        regime,
                        staging_bytes_len(
                            native_staging.as_ref().unwrap(),
                            FRAME_WIDTH * FRAME_HEIGHT * 3,
                        ),
                    )?;
                    let source = batch
                        .frames
                        .last()
                        .ok_or_else(|| "SAM31 live batch has no target frame".to_string())?;
                    let luma = current_luma.as_ref().ok_or_else(|| {
                        "SAM31 live batch could not construct current RAW luma".to_string()
                    })?;
                    live_temporal_outer_proposal(
                        module.as_ref().unwrap(),
                        native_staging.as_ref().unwrap(),
                        device,
                        prompts.as_ref().unwrap(),
                        tracker_encoder.as_ref().unwrap(),
                        tracker_states.entry(batch.eye_index).or_default(),
                        source,
                        batch.tracking_epoch,
                        batch.prompt_generation,
                        luma,
                        matches!(batch.target, Target::InnerPupilVoid | Target::OuterLimbusAndInnerPupilVoid),
                        shared_pupil_prompt,
                    )
                })();
                let video_outer = match video_outer {
                    Ok(proposal) => Some(proposal),
                    Err(error) => {
                        eprintln!("SAM31 live video tracker unavailable: {error}");
                        None
                    }
                };
                let run = process_video_frame(
                    &batch,
                    &proposal_masks,
                    current_luma.as_ref(),
                    video_outer,
                );
                let elapsed_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                if std::env::var_os("BUTTERCUP_SAM31_VIDEO_TRACE").is_some() {
                    eprintln!(
                        "SAM31_VIDEO_QUERY sequence={} elapsed_ms={} result={}",
                        batch.frames.last().map_or(0, |frame| frame.sequence),
                        elapsed_ms,
                        if run.is_ok() { "accepted" } else { "rejected" },
                    );
                }
                match run {
                    Ok(mut result) => {
                        result.elapsed_ms = elapsed_ms;
                        if let Ok(mut snapshot) = status.lock() {
                            snapshot.state = "ready";
                            snapshot.detail = match result.target {
                                Target::OuterLimbusAndInnerPupilVoid
                                    if result.sensor_pupil_ellipse.is_none() =>
                                {
                                    format!(
                                        "video tracker and RAW support agree; outer limbus ready, optional pupil center unavailable",
                                    )
                                }
                                Target::OuterLimbusAndInnerPupilVoid => format!(
                                    "video tracker and RAW support agree; independent outer limbus and pupil center ready",
                                ),
                                _ => format!(
                                    "video tracker and RAW support agree; latest {} ellipse ready",
                                    result.target.label(),
                                ),
                            };
                        }
                        let _ = results.try_send(result);
                    }
                    Err(error) => {
                        // A frame that has no adapter consensus or no
                        // photometric limbus support is an expected negative
                        // detection, not a worker failure.  Keep true runtime
                        // and model failures visually distinct from ordinary
                        // rejection so the live UI does not imply that SAM31
                        // has crashed whenever no eye is present.
                        let state = if error.starts_with("SAM31 video")
                            || error.starts_with("SAM31 outer consensus")
                            || error.starts_with("SAM31 outer RAW ring support")
                            || error.starts_with("SAM31 inner pupil void")
                        {
                            "rejected"
                        } else {
                            "error"
                        };
                        update_status(&status, state, &error);
                    }
                }
                // Completion is a publication barrier: readers observing the
                // new count must also be able to drain this batch's result and
                // read its final rejection/ready status.
                if let Ok(mut snapshot) = status.lock() {
                    snapshot.completed_batches = snapshot.completed_batches.saturating_add(1);
                    snapshot.last_elapsed_ms = Some(elapsed_ms);
                }
            }
        });
    }

    fn update_status(status: &Arc<Mutex<StatusSnapshot>>, state: &'static str, detail: &str) {
        if let Ok(mut status) = status.lock() {
            status.state = state;
            status.detail.clear();
            status.detail.push_str(detail);
        }
    }

    fn staging_bytes(staging: &Tensor) -> &mut [u8] {
        // The worker is the sole owner/user of this pinned tensor. It waits for
        // CPU materialization of model output before reusing the storage, so the
        // prior non-blocking H2D transfer is complete by construction.
        staging_bytes_len(staging, FILMSTRIP_PIXELS * 3)
    }

    fn staging_bytes_len(staging: &Tensor, len: usize) -> &mut [u8] {
        // Callers validate the tensor geometry before selecting this length.
        unsafe { std::slice::from_raw_parts_mut(staging.data_ptr() as *mut u8, len) }
    }

    fn infer(
        module: &CModule,
        staging: &Tensor,
        device: Device,
        prompts: &RuntimePrompts,
        prompt_index: usize,
    ) -> Result<InferenceOutput, String> {
        let text_ids = prompts
            .text_ids
            .get(prompt_index)
            .ok_or_else(|| format!("invalid SAM31 semantic prompt index {prompt_index}"))?;
        // The staging tensor is deliberately reused for all adapters. Make
        // the one mandatory H2D copy synchronous before Rust mutates that
        // storage for the next adapter.
        let input = staging.to_device_(device, Kind::Uint8, false, true);
        let output = tch::no_grad(|| {
            module.forward_is(&[
                IValue::Tensor(input),
                IValue::Tensor(prompts.language_features.shallow_clone()),
                IValue::Tensor(prompts.language_mask.shallow_clone()),
                IValue::Tensor(prompts.img_ids.shallow_clone()),
                IValue::Tensor(text_ids.shallow_clone()),
            ])
        })
        .map_err(|error| format!("SAM31 forward: {error}"))?;
        decode_inference_output(output)
    }

    fn infer_from_features(
        module: &CModule, features: &NativeVideoFeatures,
        prompts: &RuntimePrompts, prompt_index: usize,
    ) -> Result<InferenceOutput, String> {
        let text_ids = prompts.text_ids.get(prompt_index)
            .ok_or_else(|| format!("invalid SAM31 semantic prompt index {prompt_index}"))?;
        let output = tch::no_grad(|| module.method_is("prompt_from_features", &[
            IValue::Tensor(features.pyramid[0].shallow_clone()),
            IValue::Tensor(features.pyramid[1].shallow_clone()),
            IValue::Tensor(features.pyramid[2].shallow_clone()),
            IValue::Tensor(prompts.language_features.shallow_clone()),
            IValue::Tensor(prompts.language_mask.shallow_clone()),
            IValue::Tensor(prompts.img_ids.shallow_clone()),
            IValue::Tensor(text_ids.shallow_clone()),
        ])).map_err(|error| format!("SAM31 shared-feature prompt: {error}"))?;
        decode_inference_output(output)
    }

    fn decode_inference_output(output: IValue) -> Result<InferenceOutput, String> {
        let IValue::Tuple(mut values) = output else {
            return Err("SAM31 forward did not return a tuple".to_string());
        };
        let video_features = match values.len() {
            2 => None,
            6 => {
                let decoder_queries = values
                    .pop()
                    .unwrap()
                    .try_into()
                    .map_err(|error| format!("SAM31 decoder-query output: {error}"))?;
                let fpn_level_2 = values
                    .pop()
                    .unwrap()
                    .try_into()
                    .map_err(|error| format!("SAM31 FPN level 2 output: {error}"))?;
                let fpn_level_1 = values
                    .pop()
                    .unwrap()
                    .try_into()
                    .map_err(|error| format!("SAM31 FPN level 1 output: {error}"))?;
                let fpn_level_0 = values
                    .pop()
                    .unwrap()
                    .try_into()
                    .map_err(|error| format!("SAM31 FPN level 0 output: {error}"))?;
                Some(NativeVideoFeatures {
                    pyramid: [fpn_level_0, fpn_level_1, fpn_level_2],
                    decoder_queries,
                })
            }
            count => {
                return Err(format!(
                    "SAM31 output tuple has {count} values; expected detector-only 2 or feature graph 6"
                ));
            }
        };
        let masks: Tensor = values
            .pop()
            .unwrap()
            .try_into()
            .map_err(|error| format!("SAM31 mask output: {error}"))?;
        let scores: Tensor = values
            .pop()
            .unwrap()
            .try_into()
            .map_err(|error| format!("SAM31 score output: {error}"))?;
        let mask_shape = masks.size();
        if mask_shape.len() != 4 || mask_shape[0] != 1 {
            return Err(format!("unexpected SAM31 mask shape {mask_shape:?}"));
        }
        let query_count = mask_shape[1] as usize;
        let mask_height = mask_shape[2] as usize;
        let mask_width = mask_shape[3] as usize;
        let mask_tensor = masks
            .gt(0.0)
            .to_kind(Kind::Uint8)
            .to_device_(Device::Cpu, Kind::Uint8, false, false)
            .contiguous();
        let mut mask_bytes = vec![0u8; query_count * mask_width * mask_height];
        let mask_len = mask_bytes.len();
        mask_tensor.copy_data_u8(&mut mask_bytes, mask_len);
        let score_tensor = scores
            .to_device_(Device::Cpu, Kind::Float, false, false)
            .contiguous();
        let mut score_values = vec![0f32; query_count];
        score_tensor.copy_data(&mut score_values, query_count);
        Ok(InferenceOutput {
            logits: masks,
            masks: mask_bytes,
            scores: score_values,
            query_count,
            mask_width,
            mask_height,
            video_features,
        })
    }

    fn extract_latest_adapter_ellipse(
        output: &InferenceOutput,
    ) -> (
        Option<Ellipse>,
        f64,
        Option<usize>,
        Option<OuterMaskFitReview>,
    ) {
        let tile = HISTORY_FRAMES - 1;
        let candidates = mask_candidates_for_tile(
            &output.masks,
            &output.scores,
            output.query_count,
            output.mask_width,
            output.mask_height,
            tile,
        );
        if candidates.is_empty() {
            return (None, 0.0, None, None);
        }
        let first = candidates[0];
        let debug_all = std::env::var_os("BUTTERCUP_SAM31_FIT_DEBUG").is_some();
        let mut first_fit = None;
        for candidate in candidates.into_iter().take(MAX_MASK_CANDIDATE_FITS) {
            let full = output
                .logits
                .get(0)
                .get(candidate.query as i64)
                .unsqueeze(0)
                .unsqueeze(0)
                .upsample_bilinear2d(
                    [FRAME_HEIGHT as i64, FILMSTRIP_WIDTH as i64],
                    false,
                    None,
                    None,
                )
                .gt(0.0)
                .to_kind(Kind::Uint8)
                .to_device_(Device::Cpu, Kind::Uint8, false, false)
                .contiguous();
            let mut mask = vec![0u8; FILMSTRIP_PIXELS];
            full.copy_data_u8(&mut mask, FILMSTRIP_PIXELS);
            let fit = fit_mask_component_review(&mask, FILMSTRIP_WIDTH, FRAME_HEIGHT, tile);
            if debug_all {
                eprintln!(
                    "SAM31_FIT_DEBUG query={} objective={:.6} model_score={:.6} fit={:?}",
                    candidate.query,
                    candidate.objective,
                    candidate.model_score,
                    fit.as_ref().map(|review| (
                        review.ellipse,
                        review.retained_points.len(),
                        review.flat_tire_points.len(),
                    )),
                );
            }
            if let Some(review) = fit {
                first_fit.get_or_insert((
                    review.ellipse,
                    candidate.model_score + candidate.objective * 0.05,
                    candidate.query,
                    review,
                ));
                if !debug_all {
                    break;
                }
            }
        }
        if let Some((ellipse, score, query, review)) = first_fit {
            return (Some(ellipse), score, Some(query), Some(review));
        }
        (
            None,
            first.model_score + first.objective * 0.05,
            Some(first.query),
            None,
        )
    }

    fn latest_tile_proposal_masks(
        adapter: ProposalAdapter,
        output: &InferenceOutput,
        selected_query: Option<usize>,
    ) -> AdapterProposalMasks {
        let tile = HISTORY_FRAMES - 1;
        let Some((start_x, end_x)) = mask_x_range_for_tile(output.mask_width, tile) else {
            return AdapterProposalMasks {
                adapter,
                width: 0,
                height: 0,
                selected_query,
                masks: Vec::new(),
            };
        };
        let width = end_x.saturating_sub(start_x);
        let height = output.mask_height;
        let source_stride = output.mask_width;
        let source_plane = source_stride * height;
        let mut masks = Vec::with_capacity(output.query_count);
        for query in 0..output.query_count {
            let mut pixels = vec![0u8; width * height];
            for row in 0..height {
                let source = query * source_plane + row * source_stride + start_x;
                let destination = row * width;
                pixels[destination..destination + width]
                    .copy_from_slice(&output.masks[source..source + width]);
            }
            masks.push(ProposalMask {
                query,
                score: output.scores.get(query).copied().unwrap_or_default(),
                boundary_pixels: Arc::new(binary_mask_boundary_indices(&pixels, width, height)),
                pixels: Arc::new(pixels),
            });
        }
        AdapterProposalMasks {
            adapter,
            width,
            height,
            selected_query,
            masks,
        }
    }

    fn semantic_proposal_masks(
        prompt_index: usize,
        adapter: AdapterProposalMasks,
    ) -> SemanticProposalMasks {
        SemanticProposalMasks {
            prompt_index,
            width: adapter.width,
            height: adapter.height,
            selected_query: adapter.selected_query,
            masks: adapter.masks,
        }
    }

    fn select_bright_small_reflection_query(
        output: &InferenceOutput,
        staging: &[u8],
    ) -> Option<usize> {
        let proposals = latest_tile_proposal_masks(ProposalAdapter::QuadRgb, output, None);
        if proposals.width == 0 || proposals.height == 0 || staging.len() != FILMSTRIP_PIXELS * 3 {
            return None;
        }
        proposals
            .masks
            .iter()
            .filter_map(|candidate| {
                let occupied = candidate.pixels.iter().filter(|&&value| value != 0).count();
                let area_fraction = occupied as f64 / candidate.pixels.len().max(1) as f64;
                if !(0.002..=0.08).contains(&area_fraction) {
                    return None;
                }
                let mut luminance = 0u64;
                let mut samples = 0u64;
                let mut minimum_x = FRAME_WIDTH;
                let mut maximum_x = 0usize;
                let mut minimum_y = FRAME_HEIGHT;
                let mut maximum_y = 0usize;
                for y in 0..FRAME_HEIGHT {
                    let mask_y = (y * proposals.height / FRAME_HEIGHT).min(proposals.height - 1);
                    for x in 0..FRAME_WIDTH {
                        let mask_x = (x * proposals.width / FRAME_WIDTH).min(proposals.width - 1);
                        if candidate.pixels[mask_y * proposals.width + mask_x] == 0 {
                            continue;
                        }
                        let index = y * FILMSTRIP_WIDTH + (HISTORY_FRAMES - 1) * FRAME_WIDTH + x;
                        luminance += 2 * staging[index] as u64
                            + 5 * staging[FILMSTRIP_PIXELS + index] as u64
                            + staging[2 * FILMSTRIP_PIXELS + index] as u64;
                        samples += 8;
                        minimum_x = minimum_x.min(x);
                        maximum_x = maximum_x.max(x);
                        minimum_y = minimum_y.min(y);
                        maximum_y = maximum_y.max(y);
                    }
                }
                if samples < 32 || minimum_x >= FRAME_WIDTH || minimum_y >= FRAME_HEIGHT {
                    return None;
                }
                let mut surround_luminance = 0u64;
                let mut surround_samples = 0u64;
                let ring_minimum_x = minimum_x.saturating_sub(12);
                let ring_maximum_x = (maximum_x + 12).min(FRAME_WIDTH - 1);
                let ring_minimum_y = minimum_y.saturating_sub(12);
                let ring_maximum_y = (maximum_y + 12).min(FRAME_HEIGHT - 1);
                for y in ring_minimum_y..=ring_maximum_y {
                    let mask_y = (y * proposals.height / FRAME_HEIGHT).min(proposals.height - 1);
                    for x in ring_minimum_x..=ring_maximum_x {
                        let mask_x = (x * proposals.width / FRAME_WIDTH).min(proposals.width - 1);
                        if candidate.pixels[mask_y * proposals.width + mask_x] != 0 {
                            continue;
                        }
                        let index = y * FILMSTRIP_WIDTH + (HISTORY_FRAMES - 1) * FRAME_WIDTH + x;
                        surround_luminance += 2 * staging[index] as u64
                            + 5 * staging[FILMSTRIP_PIXELS + index] as u64
                            + staging[2 * FILMSTRIP_PIXELS + index] as u64;
                        surround_samples += 8;
                    }
                }
                if surround_samples < 32 {
                    return None;
                }
                let mean_luminance = luminance as f64 / samples as f64;
                let surround_mean = surround_luminance as f64 / surround_samples as f64;
                let local_contrast = mean_luminance - surround_mean;
                (local_contrast >= 12.0).then(|| {
                    // A pupil/iris reflection is a bright compact island in a
                    // dark neighborhood. Bright sclera fails the local-ring
                    // contrast term even when its absolute luminance is high.
                    let objective = 2.2 * local_contrast + 0.15 * mean_luminance
                        - 40.0 * area_fraction.sqrt()
                        + 3.0 * candidate.score as f64;
                    (objective, candidate.query)
                })
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .map(|candidate| candidate.1)
    }

    fn select_dark_center_void_query(output: &InferenceOutput, staging: &[u8]) -> Option<usize> {
        let proposals = latest_tile_proposal_masks(ProposalAdapter::QuadRgb, output, None);
        if proposals.width == 0 || proposals.height == 0 || staging.len() != FILMSTRIP_PIXELS * 3 {
            return None;
        }
        proposals
            .masks
            .iter()
            .filter_map(|candidate| {
                let occupied = candidate.pixels.iter().filter(|&&value| value != 0).count();
                let area_fraction = occupied as f64 / candidate.pixels.len().max(1) as f64;
                if !(0.01..=0.30).contains(&area_fraction) {
                    return None;
                }
                let mut luminance = 0u64;
                let mut samples = 0u64;
                let mut minimum_x = FRAME_WIDTH;
                let mut maximum_x = 0usize;
                let mut minimum_y = FRAME_HEIGHT;
                let mut maximum_y = 0usize;
                for y in 0..FRAME_HEIGHT {
                    let mask_y = (y * proposals.height / FRAME_HEIGHT).min(proposals.height - 1);
                    for x in 0..FRAME_WIDTH {
                        let mask_x = (x * proposals.width / FRAME_WIDTH).min(proposals.width - 1);
                        if candidate.pixels[mask_y * proposals.width + mask_x] == 0 {
                            continue;
                        }
                        let index = y * FILMSTRIP_WIDTH + (HISTORY_FRAMES - 1) * FRAME_WIDTH + x;
                        luminance += 2 * staging[index] as u64
                            + 5 * staging[FILMSTRIP_PIXELS + index] as u64
                            + staging[2 * FILMSTRIP_PIXELS + index] as u64;
                        samples += 8;
                        minimum_x = minimum_x.min(x);
                        maximum_x = maximum_x.max(x);
                        minimum_y = minimum_y.min(y);
                        maximum_y = maximum_y.max(y);
                    }
                }
                if samples < 64 || minimum_x >= FRAME_WIDTH || minimum_y >= FRAME_HEIGHT {
                    return None;
                }
                let box_width = maximum_x - minimum_x + 1;
                let box_height = maximum_y - minimum_y + 1;
                let aspect =
                    box_width.max(box_height) as f64 / box_width.min(box_height).max(1) as f64;
                let compactness = samples as f64 / 8.0 / (box_width * box_height).max(1) as f64;
                if aspect > 3.0 || compactness < 0.22 {
                    return None;
                }
                let mean_luminance = luminance as f64 / samples as f64;
                // The text query defines the object. Photometry only breaks
                // ties among geometrically plausible semantic answers; if
                // darkness dominates, a half-iris shadow wins over the void.
                let objective = -0.08 * mean_luminance + 35.0 * compactness
                    - 12.0 * area_fraction.sqrt()
                    + 100.0 * candidate.score as f64;
                Some((objective, candidate.query))
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .map(|candidate| candidate.1)
    }

    fn paint_latest_tile_mask_pink(
        staging: &mut [u8],
        mask: &[u8],
        mask_width: usize,
        mask_height: usize,
        dilation: usize,
    ) -> Result<(), String> {
        if staging.len() != FILMSTRIP_PIXELS * 3
            || mask_width == 0
            || mask_height == 0
            || mask.len() != mask_width * mask_height
        {
            return Err("reflection exclusion received incompatible mask geometry".to_string());
        }
        for y in 0..FRAME_HEIGHT {
            let mask_y = (y * mask_height / FRAME_HEIGHT).min(mask_height - 1);
            for x in 0..FRAME_WIDTH {
                let mask_x = (x * mask_width / FRAME_WIDTH).min(mask_width - 1);
                let minimum_y = mask_y.saturating_sub(dilation);
                let maximum_y = (mask_y + dilation).min(mask_height - 1);
                let minimum_x = mask_x.saturating_sub(dilation);
                let maximum_x = (mask_x + dilation).min(mask_width - 1);
                let covered = (minimum_y..=maximum_y).any(|near_y| {
                    (minimum_x..=maximum_x).any(|near_x| mask[near_y * mask_width + near_x] != 0)
                });
                if !covered {
                    continue;
                }
                let index = y * FILMSTRIP_WIDTH + (HISTORY_FRAMES - 1) * FRAME_WIDTH + x;
                staging[index] = 255;
                staging[FILMSTRIP_PIXELS + index] = 0;
                staging[2 * FILMSTRIP_PIXELS + index] = 255;
            }
        }
        Ok(())
    }

    /// Replace the complete detector filmstrip with a saturated nuisance
    /// field, then restore only the selected latest-tile pupil pixels. This
    /// deliberately removes all surrounding eye anatomy for the follow-on
    /// semantic question; retained RAW10 and review imagery remain untouched.
    fn isolate_latest_tile_mask_on_pink(
        staging: &mut [u8],
        mask: &[u8],
        mask_width: usize,
        mask_height: usize,
    ) -> Result<(), String> {
        if staging.len() != FILMSTRIP_PIXELS * 3
            || mask_width == 0
            || mask_height == 0
            || mask.len() != mask_width * mask_height
        {
            return Err("pupil isolation received incompatible mask geometry".to_string());
        }
        let original = staging.to_vec();
        staging[..FILMSTRIP_PIXELS].fill(255);
        staging[FILMSTRIP_PIXELS..2 * FILMSTRIP_PIXELS].fill(0);
        staging[2 * FILMSTRIP_PIXELS..].fill(255);
        for y in 0..FRAME_HEIGHT {
            let mask_y = (y * mask_height / FRAME_HEIGHT).min(mask_height - 1);
            for x in 0..FRAME_WIDTH {
                let mask_x = (x * mask_width / FRAME_WIDTH).min(mask_width - 1);
                if mask[mask_y * mask_width + mask_x] == 0 {
                    continue;
                }
                let index = y * FILMSTRIP_WIDTH + (HISTORY_FRAMES - 1) * FRAME_WIDTH + x;
                staging[index] = original[index];
                staging[FILMSTRIP_PIXELS + index] = original[FILMSTRIP_PIXELS + index];
                staging[2 * FILMSTRIP_PIXELS + index] = original[2 * FILMSTRIP_PIXELS + index];
            }
        }
        Ok(())
    }

    fn smoothed_inset_native_mask(
        mask: &[u8],
        mask_width: usize,
        mask_height: usize,
        inset: usize,
    ) -> Vec<u8> {
        let mut native = vec![0u8; FRAME_WIDTH * FRAME_HEIGHT];
        for y in 0..FRAME_HEIGHT {
            let mask_y = (y * mask_height / FRAME_HEIGHT).min(mask_height - 1);
            for x in 0..FRAME_WIDTH {
                let mask_x = (x * mask_width / FRAME_WIDTH).min(mask_width - 1);
                native[y * FRAME_WIDTH + x] = mask[mask_y * mask_width + mask_x];
            }
        }
        // A small majority filter suppresses stair steps and isolated mask
        // hairs before measuring the requested five-pixel inward offset.
        let mut smooth = vec![0u8; native.len()];
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                let mut occupied = 0usize;
                let mut samples = 0usize;
                for dy in -2isize..=2 {
                    for dx in -2isize..=2 {
                        let xx = x as isize + dx;
                        let yy = y as isize + dy;
                        if xx < 0
                            || yy < 0
                            || xx >= FRAME_WIDTH as isize
                            || yy >= FRAME_HEIGHT as isize
                        {
                            continue;
                        }
                        samples += 1;
                        occupied += (native[yy as usize * FRAME_WIDTH + xx as usize] != 0) as usize;
                    }
                }
                smooth[y * FRAME_WIDTH + x] = (occupied * 2 >= samples) as u8;
            }
        }
        let mut eroded = vec![0u8; smooth.len()];
        let inset = inset as isize;
        for y in inset..FRAME_HEIGHT as isize - inset {
            for x in inset..FRAME_WIDTH as isize - inset {
                let covered = (-inset..=inset).all(|dy| {
                    (-inset..=inset).all(|dx| {
                        dx * dx + dy * dy > inset * inset
                            || smooth[(y + dy) as usize * FRAME_WIDTH + (x + dx) as usize] != 0
                    })
                });
                eroded[y as usize * FRAME_WIDTH + x as usize] = covered as u8;
            }
        }
        eroded
    }

    fn paint_latest_tile_mask_interior_pink(
        staging: &mut [u8],
        mask: &[u8],
        mask_width: usize,
        mask_height: usize,
        inset: usize,
    ) -> Result<(), String> {
        if staging.len() != FILMSTRIP_PIXELS * 3
            || mask_width == 0
            || mask_height == 0
            || mask.len() != mask_width * mask_height
        {
            return Err("pupil interior occlusion received incompatible mask geometry".to_string());
        }
        let occlusion = smoothed_inset_native_mask(mask, mask_width, mask_height, inset);
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                if occlusion[y * FRAME_WIDTH + x] == 0 {
                    continue;
                }
                let index = y * FILMSTRIP_WIDTH + (HISTORY_FRAMES - 1) * FRAME_WIDTH + x;
                staging[index] = 255;
                staging[FILMSTRIP_PIXELS + index] = 0;
                staging[2 * FILMSTRIP_PIXELS + index] = 255;
            }
        }
        Ok(())
    }

    fn strongest_model_query(output: &InferenceOutput) -> Option<usize> {
        output
            .scores
            .iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(query, _)| query)
    }

    fn strongest_nonempty_latest_query(output: &InferenceOutput) -> Option<usize> {
        latest_tile_proposal_masks(ProposalAdapter::QuadRgb, output, None)
            .masks
            .iter()
            .filter(|candidate| candidate.pixels.iter().any(|&value| value != 0))
            .filter(|candidate| candidate.score.is_finite())
            .max_by(|left, right| left.score.total_cmp(&right.score))
            .map(|candidate| candidate.query)
            .or_else(|| strongest_model_query(output))
    }

    fn process_video_frame(
        batch: &Batch,
        proposal_publisher: &SyncSender<Arc<ProposalMasks>>,
        current_luma: Option<&FloatImage>,
        video_outer: Option<LiveTemporalOuterProposal>,
    ) -> Result<OuterResult, String> {
        let source = batch
            .frames
            .last()
            .ok_or_else(|| "SAM31 video tracker received no source frame".to_string())?;
        let LiveTemporalOuterProposal {
            semantic,
            outer_fit,
            outer_support,
            pupil_fit,
        } = video_outer
            .ok_or_else(|| "SAM31 video tracker produced no current-frame mask".to_string())?;
        let outer_fit = outer_fit
            .ok_or_else(|| "SAM31 video tracker mask had no plausible limbus fit".to_string())?;
        let outer_fit = model_review_in_source(outer_fit, source.width);
        let quality = semantic
            .selected_query
            .and_then(|selected| semantic.masks.iter().find(|mask| mask.query == selected))
            .map(|mask| f64::from(mask.score))
            .unwrap_or_default();
        let outer_ellipse = outer_fit.ellipse;
        // Always retain the same-exposure RAW pupil void alongside an outer
        // proposal.  Virtual contact needs this private cue to choose between
        // the two antipodal surface normals even when the operator has not
        // selected SAM as the public rough-center provider.  Target selection
        // below still controls whether the pupil is published as a normal Y
        // product; this review-only fit cannot silently change that mode.
        let proposal_pupil_fit = pupil_fit;
        let proposal_masks = Arc::new(ProposalMasks {
            tracking_epoch: batch.tracking_epoch,
            prompt_generation: batch.prompt_generation,
            eye_index: batch.eye_index,
            source_sequence: source.sequence,
            source_timestamp_ns: source.timestamp_ns,
            source_sensor_origin: (source.sensor_x, source.sensor_y),
            source_width: source.width,
            source_height: source.height,
            source_raw: Arc::clone(&source.pixels),
            semantic: Some(semantic),
            outer_fit: Some(outer_fit),
            inner_pupil_fit: proposal_pupil_fit,
            adapters: Vec::new(),
        });
        // Publish the exact attempted mask even when the independent RAW gate
        // rejects it, so the operator can inspect the rejected proposal.
        let _ = proposal_publisher.try_send(Arc::clone(&proposal_masks));

        if current_luma.is_none() {
            return Err("SAM31 video tracker could not construct current RAW luma".to_string());
        }
        if !live_detector_raw_gate_passes(outer_support) {
            return Err(format!(
                "SAM31 video outer RAW ring support {:.3} from {} samples and {} strong sectors is below {:.3}",
                outer_support.score,
                outer_support.points,
                outer_support.strong_sectors,
                MIN_RAW_RING_SUPPORT_SCORE,
            ));
        }
        let pupil_fit = matches!(
            batch.target,
            Target::InnerPupilVoid | Target::OuterLimbusAndInnerPupilVoid
        )
        .then(|| proposal_pupil_fit.map(|review| (review.ellipse, review.raw_support)))
        .flatten();
        let (target_ellipse, target_support, pupil_ellipse) =
            select_target_products(batch.target, outer_ellipse, outer_support, pupil_fit)?;
        let to_sensor = |mut ellipse: Ellipse| {
            ellipse.center.0 += source.sensor_x as f64;
            ellipse.center.1 += source.sensor_y as f64;
            ellipse
        };
        let source_registration_anchor_sensor = source.registration_anchor.map(|center| {
            (
                center.0 + source.sensor_x as f64,
                center.1 + source.sensor_y as f64,
            )
        });
        Ok(OuterResult {
            tracking_epoch: batch.tracking_epoch,
            prompt_generation: batch.prompt_generation,
            target: batch.target,
            eye_index: batch.eye_index,
            source_sequence: source.sequence,
            source_timestamp_ns: source.timestamp_ns,
            source_sensor_origin: (source.sensor_x, source.sensor_y),
            source_registration_anchor_sensor,
            sensor_ellipse: to_sensor(target_ellipse),
            sensor_outer_ellipse: to_sensor(outer_ellipse),
            sensor_pupil_ellipse: pupil_ellipse.map(to_sensor),
            agreeing_adapters: 0,
            quality,
            raw_ring_support_score: outer_support.score,
            raw_ring_support_points: outer_support.points,
            raw_ring_positive_fraction: outer_support.positive_fraction,
            raw_ring_strong_sectors: outer_support.strong_sectors,
            raw_target_support_score: target_support.score,
            raw_target_support_points: target_support.points,
            raw_target_positive_fraction: target_support.positive_fraction,
            raw_target_strong_sectors: target_support.strong_sectors,
            elapsed_ms: 0,
            video_tracked: true,
            proposal_masks,
        })
    }

    fn process_batch(
        module: &CModule,
        staging: &Tensor,
        device: Device,
        batch: &Batch,
        proposal_publisher: &SyncSender<Arc<ProposalMasks>>,
        prompts: &RuntimePrompts,
        temporal_outer: Option<(SemanticProposalMasks, Option<OuterMaskFitReview>)>,
    ) -> Result<OuterResult, String> {
        let mut latest = Vec::<(Ellipse, f64)>::new();
        let mut proposal_adapters = Vec::with_capacity(3);

        let balanced = balanced_quad_rgb(&batch.frames);
        write_quantized_filmstrip(&balanced, 0.35, 99.65, 0.82, staging_bytes(staging))?;
        let output = infer(module, staging, device, prompts, OUTER_IRIS_PROMPT)?;
        let extracted = extract_latest_adapter_ellipse(&output);
        let temporal_fit = temporal_outer
            .as_ref()
            .and_then(|(_, fit)| fit.as_ref())
            .cloned();
        let outer_fit = temporal_fit.or_else(|| extracted.3.clone());
        let quad_proposals =
            latest_tile_proposal_masks(ProposalAdapter::QuadRgb, &output, extracted.2);
        let mut semantic = if batch.semantic_prompt == OUTER_IRIS_PROMPT {
            temporal_outer.map(|(proposal, _)| proposal).or_else(|| {
                Some(semantic_proposal_masks(
                    OUTER_IRIS_PROMPT,
                    quad_proposals.clone(),
                ))
            })
        } else {
            None
        };
        proposal_adapters.push(quad_proposals);
        debug_adapter("quad_rgb", &extracted);
        if let (Some(ellipse), score, _, _) = extracted {
            latest.push((ellipse, score));
        }

        let luma = raw_luma(&batch.frames);
        write_quantized_filmstrip(&luma, 0.20, 99.75, 0.80, staging_bytes(staging))?;
        let output = infer(module, staging, device, prompts, OUTER_IRIS_PROMPT)?;
        let extracted = extract_latest_adapter_ellipse(&output);
        proposal_adapters.push(latest_tile_proposal_masks(
            ProposalAdapter::RawLuma,
            &output,
            extracted.2,
        ));
        debug_adapter("raw_luma", &extracted);
        if let (Some(ellipse), score, _, _) = extracted {
            latest.push((ellipse, score));
        }

        let chroma = log_chroma(&balanced);
        write_quantized_filmstrip(&chroma, 0.60, 99.40, 1.0, staging_bytes(staging))?;
        let output = infer(module, staging, device, prompts, OUTER_IRIS_PROMPT)?;
        let extracted = extract_latest_adapter_ellipse(&output);
        proposal_adapters.push(latest_tile_proposal_masks(
            ProposalAdapter::LogChroma,
            &output,
            extracted.2,
        ));
        debug_adapter("log_chroma", &extracted);
        if let (Some(ellipse), score, _, _) = extracted {
            latest.push((ellipse, score));
        }

        if semantic.is_none() {
            // Semantic review uses the same lossless-source Quad-RGB adapter
            // as the accepted outer-iris query. It is deliberately one extra
            // prompt per completed batch, never six hidden graph passes.
            write_quantized_filmstrip(&balanced, 0.35, 99.65, 0.82, staging_bytes(staging))?;
            let output = infer(module, staging, device, prompts, batch.semantic_prompt)?;
            let adapter = latest_tile_proposal_masks(
                ProposalAdapter::QuadRgb,
                &output,
                strongest_model_query(&output),
            );
            semantic = Some(semantic_proposal_masks(batch.semantic_prompt, adapter));
        }

        let latest_index = HISTORY_FRAMES - 1;
        let source = &batch.frames[latest_index];
        // As in the video path, this is proposal-private sign evidence.  Keep
        // it available whenever an outer surface exists, independently of the
        // public rough-center mode selected by Y.
        let proposal_pupil_fit = outer_fit
            .as_ref()
            .and_then(|review| {
                fit_inner_pupil_void(
                    &luma[latest_index],
                    review.ellipse,
                    source.pupil_component_seed,
                )
            })
            .map(|(ellipse, raw_support)| PupilVoidFitReview {
                ellipse,
                raw_support,
            });
        let proposal_masks = Arc::new(ProposalMasks {
            tracking_epoch: batch.tracking_epoch,
            prompt_generation: batch.prompt_generation,
            eye_index: batch.eye_index,
            source_sequence: source.sequence,
            source_timestamp_ns: source.timestamp_ns,
            source_sensor_origin: (source.sensor_x, source.sensor_y),
            source_width: source.width,
            source_height: source.height,
            source_raw: Arc::clone(&source.pixels),
            semantic,
            outer_fit,
            inner_pupil_fit: proposal_pupil_fit,
            adapters: proposal_adapters,
        });
        // Publish the queries before anatomical consensus and RAW support are
        // evaluated: rejected batches are exactly the ones an operator needs
        // to inspect. Never stall inference when the one-slot diagnostic
        // mailbox still holds an older frame.
        let _ = proposal_publisher.try_send(Arc::clone(&proposal_masks));
        let (outer_ellipse, agreeing) = agreeing_consensus(&latest).ok_or_else(|| {
            format!(
                "SAM31 outer consensus had no agreeing pair among {} plausible adapter fit(s)",
                latest.len()
            )
        })?;
        let outer_support = raw_ring_support(&luma[latest_index], outer_ellipse);
        if outer_support.score < MIN_RAW_RING_SUPPORT_SCORE {
            return Err(format!(
                "SAM31 outer RAW ring support {:.3} from {} samples and {} strong sectors is below {:.3}",
                outer_support.score,
                outer_support.points,
                outer_support.strong_sectors,
                MIN_RAW_RING_SUPPORT_SCORE,
            ));
        }
        let pupil_fit = matches!(
            batch.target,
            Target::InnerPupilVoid | Target::OuterLimbusAndInnerPupilVoid
        )
        .then(|| {
            fit_inner_pupil_void(
                &luma[latest_index],
                outer_ellipse,
                source.pupil_component_seed,
            )
        })
        .flatten();
        let (target_ellipse, target_support, pupil_ellipse) =
            select_target_products(batch.target, outer_ellipse, outer_support, pupil_fit)?;
        let mut sensor_ellipse = target_ellipse;
        sensor_ellipse.center.0 += source.sensor_x as f64;
        sensor_ellipse.center.1 += source.sensor_y as f64;
        let mut sensor_outer_ellipse = outer_ellipse;
        sensor_outer_ellipse.center.0 += source.sensor_x as f64;
        sensor_outer_ellipse.center.1 += source.sensor_y as f64;
        let sensor_pupil_ellipse = pupil_ellipse.map(|mut ellipse| {
            ellipse.center.0 += source.sensor_x as f64;
            ellipse.center.1 += source.sensor_y as f64;
            ellipse
        });
        let source_registration_anchor_sensor = source.registration_anchor.map(|center| {
            (
                center.0 + source.sensor_x as f64,
                center.1 + source.sensor_y as f64,
            )
        });
        let quality =
            agreeing.iter().map(|&index| latest[index].1).sum::<f64>() / agreeing.len() as f64;
        Ok(OuterResult {
            tracking_epoch: batch.tracking_epoch,
            prompt_generation: batch.prompt_generation,
            target: batch.target,
            eye_index: batch.eye_index,
            source_sequence: source.sequence,
            source_timestamp_ns: source.timestamp_ns,
            source_sensor_origin: (source.sensor_x, source.sensor_y),
            source_registration_anchor_sensor,
            sensor_ellipse,
            sensor_outer_ellipse,
            sensor_pupil_ellipse,
            agreeing_adapters: agreeing.len(),
            quality,
            raw_ring_support_score: outer_support.score,
            raw_ring_support_points: outer_support.points,
            raw_ring_positive_fraction: outer_support.positive_fraction,
            raw_ring_strong_sectors: outer_support.strong_sectors,
            raw_target_support_score: target_support.score,
            raw_target_support_points: target_support.points,
            raw_target_positive_fraction: target_support.positive_fraction,
            raw_target_strong_sectors: target_support.strong_sectors,
            elapsed_ms: 0,
            video_tracked: false,
            proposal_masks,
        })
    }

    fn debug_adapter(
        name: &str,
        extracted: &(
            Option<Ellipse>,
            f64,
            Option<usize>,
            Option<OuterMaskFitReview>,
        ),
    ) {
        if std::env::var_os("BUTTERCUP_SAM31_DEBUG").is_none() {
            return;
        }
        eprintln!(
            "SAM31_DEBUG adapter={name} ellipse={:?} score={:.6} query={:?} retained={} censored={} upper={} lower={}",
            extracted.0,
            extracted.1,
            extracted.2,
            extracted
                .3
                .as_ref()
                .map_or(0, |review| review.retained_points.len()),
            extracted
                .3
                .as_ref()
                .map_or(0, |review| review.flat_tire_points.len()),
            extracted
                .3
                .as_ref()
                .is_some_and(|review| review.upper_flat_tire),
            extracted
                .3
                .as_ref()
                .is_some_and(|review| review.lower_flat_tire),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_requires_native_support_and_all_three_asset_files() {
        // This checks presence only; model loading remains the worker's job.
        let present = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
        let missing = Path::new("");
        assert_eq!(startup_assets_available(present, present, present), cfg!(feature = "sam31"));
        assert!(!startup_assets_available(missing, present, present));
        assert!(!startup_assets_available(present, missing, present));
        assert!(!startup_assets_available(present, present, missing));
        assert!(!startup_assets_available(present.parent().unwrap(), present, present));
    }

    #[test]
    fn exported_outlines_keep_native_pixel_center_coordinates() {
        let mut mask = vec![0; 64 * 64];
        for y in 12..48 {
            for x in 10..50 { mask[y * 64 + x] = 1; }
        }
        let model = native_outline_points(&mask, 64, 64, FRAME_WIDTH, FRAME_HEIGHT);
        let native = native_outline_points(&mask, 64, 64, 420, 280);
        assert_eq!(model.len(), 256);
        assert_eq!(native.len(), model.len());
        for (a, b) in model.iter().zip(&native) {
            assert!((b.0 - ((a.0 + 0.5) * 420.0 / FRAME_WIDTH as f64 - 0.5)).abs() < 1e-8);
            assert!((b.1 - ((a.1 + 0.5) * 280.0 / FRAME_HEIGHT as f64 - 0.5)).abs() < 1e-8);
        }
    }

    #[test]
    fn exported_outlines_reject_empty_and_malformed_masks() {
        assert!(native_outline_points(&[], 0, 0, 420, 280).is_empty());
        assert!(native_outline_points(&[1, 1], 64, 64, 420, 280).is_empty());
        assert!(native_outline_points(&[0; 64 * 64], 64, 64, 420, 280).is_empty());
        assert!(native_outline_points(&[1; 64 * 64], 64, 64, 0, 280).is_empty());
    }

    #[test]
    fn enlarged_roi_model_projection_preserves_gaze_shape_and_native_pixels() {
        let ellipse = Ellipse {
            center: (191.5, 127.5),
            major_radius: 80.0,
            minor_radius: 56.0,
            angle: 0.37,
        };
        let native = model_ellipse_in_source(ellipse, 420);
        assert_eq!(native.center, (209.5, 139.5));
        assert_eq!(native.angle, ellipse.angle);
        assert!((native.minor_radius / native.major_radius - 0.7).abs() < 1e-12);
        let mut image = FloatImage::new(420, 280);
        for y in 0..280 {
            for x in 0..420 {
                image.data[y * 420 + x] = [x as f32, y as f32, (x + y) as f32];
            }
        }
        let before = image.data.clone();
        let mut quantized = vec![0; FRAME_WIDTH * FRAME_HEIGHT * 3];
        write_quantized_filmstrip(&[image.clone()], 0.0, 100.0, 1.0, &mut quantized).unwrap();
        assert_eq!(image.data, before);
        assert_eq!(quantized[0], 0);
        assert_eq!(quantized[FRAME_WIDTH - 1], 255);
        assert_eq!(
            quantized[FRAME_WIDTH * FRAME_HEIGHT + (FRAME_HEIGHT - 1) * FRAME_WIDTH],
            255
        );
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3)
        })
    }

    #[test]
    fn live_temporal_update_preserves_a_healthy_propagated_mask() {
        assert_eq!(
            choose_live_temporal_update(true, Some(104)),
            LiveTemporalUpdate::Propagate,
        );
    }

    #[test]
    fn live_temporal_update_immediately_uses_the_current_plausible_detector_query() {
        assert_eq!(
            choose_live_temporal_update(false, Some(104)),
            LiveTemporalUpdate::ConditionFromDetector(104),
        );
    }

    #[test]
    fn live_temporal_update_holds_history_instead_of_admitting_an_absent_mask() {
        assert_eq!(
            choose_live_temporal_update(false, None),
            LiveTemporalUpdate::HoldLastConditioning,
        );
    }

    #[test]
    fn live_recovery_refuses_to_replace_a_carried_identity_with_global_search() {
        assert_eq!(
            choose_live_recovery_queries(false, &[28, 91], Some((144, 0.79))),
            vec![144],
        );
        assert_eq!(
            choose_live_recovery_queries(false, &[28, 91], Some((144, 0.42))),
            Vec::<usize>::new(),
        );
        assert_eq!(
            choose_live_recovery_queries(false, &[28, 91], None),
            Vec::<usize>::new(),
        );
        assert_eq!(
            choose_live_recovery_queries(true, &[28, 91], None),
            vec![28, 91],
        );
    }

    #[test]
    fn live_propagation_rejects_the_observed_identity_break_transition() {
        // The R-off calibration replay's first break was still a plausible
        // eye-sized ellipse, but overlap with the admitted identity fell to
        // 0.586. It must recover from query 144 rather than encode that
        // propagated mask or jump to the strongest global query 28.
        assert!(!live_propagation_is_healthy(
            true,
            Some(0.21),
            true,
            true,
            Some(0.586),
        ));
        assert!(live_propagation_is_healthy(
            true,
            Some(0.21),
            true,
            true,
            Some(0.846),
        ));
    }

    #[test]
    fn live_propagation_fails_closed_when_prior_iou_is_missing() {
        assert!(!live_propagation_is_healthy(
            true,
            Some(0.21),
            true,
            true,
            None,
        ));
    }

    fn live_input(sequence: u64, timestamp_ns: u64) -> LiveTrackerInput {
        LiveTrackerInput {
            tracking_epoch: 7,
            prompt_generation: 11,
            sequence,
            timestamp_ns,
            sensor_origin: (100, 200),
            width: FRAME_WIDTH,
            height: FRAME_HEIGHT,
        }
    }

    #[test]
    fn pupil_history_only_survives_same_identity_overlapping_roi_translation() {
        let previous = live_input(40, 2_000_000_000);
        let moved = LiveTrackerInput {
            sensor_origin: (120, 216), ..live_input(41, 2_400_000_000)
        };
        assert!(live_tracker_requires_reset(Some(previous), moved), "pixel-addressed SAM memory must reset");
        assert!(pupil_history_survives_roi_relocation(Some(previous), moved));
        for incompatible in [
            LiveTrackerInput { tracking_epoch: 8, ..moved },
            LiveTrackerInput { prompt_generation: 12, ..moved },
            LiveTrackerInput { width: FRAME_WIDTH + 4, ..moved },
            LiveTrackerInput { height: FRAME_HEIGHT + 4, ..moved },
            LiveTrackerInput { sequence: 40, ..moved },
            LiveTrackerInput { timestamp_ns: 1_999_999_999, ..moved },
            LiveTrackerInput { timestamp_ns: 3_000_000_001, ..moved },
            LiveTrackerInput { sensor_origin: (100 + FRAME_WIDTH as u32, 200), ..moved },
        ] {
            assert!(!pupil_history_survives_roi_relocation(Some(previous), incompatible));
        }
        assert!(!pupil_history_survives_roi_relocation(None, moved));
    }

    #[test]
    fn live_tracker_resets_on_session_time_and_roi_discontinuities() {
        let previous = live_input(40, 2_000_000_000);
        assert!(!live_tracker_requires_reset(
            Some(previous),
            live_input(41, 2_900_000_000),
        ));

        let mut changed = live_input(41, 2_000_000_001);
        changed.tracking_epoch += 1;
        assert!(live_tracker_requires_reset(Some(previous), changed));
        changed = live_input(41, 2_000_000_001);
        changed.prompt_generation += 1;
        assert!(live_tracker_requires_reset(Some(previous), changed));
        assert!(live_tracker_requires_reset(
            Some(previous),
            live_input(39, 2_000_000_001),
        ));
        assert!(live_tracker_requires_reset(
            Some(previous),
            live_input(41, 1_999_999_999),
        ));
        assert!(live_tracker_requires_reset(
            Some(previous),
            live_input(41, 2_900_000_001),
        ));
        changed = live_input(41, 2_000_000_001);
        changed.sensor_origin.0 += 1;
        assert!(live_tracker_requires_reset(Some(previous), changed));
        changed = live_input(41, 2_000_000_001);
        changed.sensor_origin.0 += FRAME_WIDTH as u32;
        assert!(!live_rois_overlap(previous, changed));
        assert!(live_tracker_requires_reset(Some(previous), changed));
    }

    #[test]
    fn live_tracker_holds_exactly_three_processed_misses() {
        assert_eq!(next_live_hold_miss(0), (1, false));
        assert_eq!(next_live_hold_miss(1), (2, false));
        assert_eq!(next_live_hold_miss(2), (3, true));
    }

    #[test]
    fn live_bootstrap_ranks_all_finite_candidates_and_applies_strict_shape_policy() {
        assert_eq!(
            ranked_finite_query_indices(&[0.4, f32::NAN, 0.9, 0.4, f32::INFINITY]),
            vec![2, 0, 3],
        );
        assert!(live_detector_candidate_is_plausible(0.1, Some(0.05), true,));
        assert!(live_detector_candidate_is_plausible(0.1, Some(0.50), true,));
        assert!(!live_detector_candidate_is_plausible(
            f32::NAN,
            Some(0.21),
            true,
        ));
        assert!(!live_detector_candidate_is_plausible(0.1, Some(0.51), true,));
        assert!(!live_detector_candidate_is_plausible(
            0.1,
            Some(0.21),
            false,
        ));
    }

    #[test]
    fn detector_conditioning_requires_the_outer_raw_gate() {
        let rejected = RawRingSupport {
            score: MIN_RAW_RING_SUPPORT_SCORE - 0.001,
            ..RawRingSupport::default()
        };
        let accepted = RawRingSupport {
            score: MIN_RAW_RING_SUPPORT_SCORE,
            ..RawRingSupport::default()
        };
        assert!(!live_memory_commit_allowed(
            LiveMemorySource::Detector,
            rejected,
        ));
        assert!(live_memory_commit_allowed(
            LiveMemorySource::Detector,
            accepted,
        ));
        assert!(live_memory_commit_allowed(
            LiveMemorySource::Propagation,
            rejected,
        ));
        assert!(live_committed_frame_counts_as_miss(
            LiveMemorySource::Propagation,
            rejected,
        ));
        assert!(!live_committed_frame_counts_as_miss(
            LiveMemorySource::Propagation,
            accepted,
        ));
        assert_eq!(
            choose_live_temporal_update(false, None),
            LiveTemporalUpdate::HoldLastConditioning,
        );
    }

    #[test]
    fn hot_pixel_check_requires_a_persistent_sensor_fixed_raw10_outlier() {
        let frames = (0..6usize)
            .map(|index| {
                let mut pixels = vec![120u16; 24 * 24];
                pixels[12 * 24 + 12] = 1023;
                // A bright scene point that moves does not recur at one sensor
                // coordinate and therefore must not enter the defect map.
                pixels[8 * 24 + 6 + index] = 1023;
                Arc::new(RawFrame {
                    eye_index: 0,
                    sequence: index as u64,
                    timestamp_ns: index as u64 * 1_000_000,
                    sensor_x: 100,
                    sensor_y: 200,
                    width: 24,
                    height: 24,
                    registration_anchor: None,
                    pupil_component_seed: None,
                    pixels: Arc::new(pixels),
                })
            })
            .collect::<Vec<_>>();
        let hot = persistent_raw10_hot_pixels(&frames);
        assert_eq!(hot.len(), 1, "{hot:?}");
        assert_eq!((hot[0].sensor_x, hot[0].sensor_y), (112, 212));
        assert_eq!(hot[0].detected_frames, 6);

        let corrected = corrected_hot_pixel_frames(&frames, &hot);
        assert_eq!(frames[0].pixels[12 * 24 + 12], 1023);
        assert_eq!(corrected[0].pixels[12 * 24 + 12], 120);
    }

    #[test]
    fn combined_target_never_suppresses_outer_when_pupil_center_is_missing() {
        let outer = Ellipse {
            center: (190.0, 125.0),
            major_radius: 92.0,
            minor_radius: 68.0,
            angle: -0.2,
        };
        let outer_support = RawRingSupport {
            score: 0.8,
            points: 64,
            positive_fraction: 0.9,
            strong_sectors: 8,
        };
        let (primary, support, pupil) = select_target_products(
            Target::OuterLimbusAndInnerPupilVoid,
            outer,
            outer_support,
            None,
        )
        .unwrap();
        assert_eq!(primary, outer);
        assert_eq!(support, outer_support);
        assert_eq!(pupil, None);
        assert!(
            select_target_products(Target::InnerPupilVoid, outer, outer_support, None,).is_err()
        );
    }

    #[test]
    fn combined_target_keeps_outer_primary_and_exposes_only_optional_pupil_product() {
        let outer = Ellipse {
            center: (190.0, 125.0),
            major_radius: 92.0,
            minor_radius: 68.0,
            angle: -0.2,
        };
        let pupil = Ellipse {
            center: (205.0, 128.0),
            major_radius: 24.0,
            minor_radius: 18.0,
            angle: -0.2,
        };
        let outer_support = RawRingSupport {
            score: 0.8,
            points: 64,
            positive_fraction: 0.9,
            strong_sectors: 8,
        };
        let pupil_support = RawRingSupport {
            score: 0.7,
            points: 48,
            positive_fraction: 0.8,
            strong_sectors: 7,
        };
        let (primary, support, optional_pupil) = select_target_products(
            Target::OuterLimbusAndInnerPupilVoid,
            outer,
            outer_support,
            Some((pupil, pupil_support)),
        )
        .unwrap();
        assert_eq!(primary, outer);
        assert_eq!(support, outer_support);
        assert_eq!(optional_pupil, Some(pupil));
    }

    #[test]
    fn raw_ring_support_accepts_dark_iris_to_bright_sclera_transition() {
        let ellipse = Ellipse {
            center: (192.0, 128.0),
            major_radius: 100.0,
            minor_radius: 70.0,
            angle: -0.18,
        };
        let (angle_sine, angle_cosine) = ellipse.angle.sin_cos();
        let mut image = FloatImage::new(FRAME_WIDTH, FRAME_HEIGHT);
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                let dx = x as f64 - ellipse.center.0;
                let dy = y as f64 - ellipse.center.1;
                let local_x = angle_cosine * dx + angle_sine * dy;
                let local_y = -angle_sine * dx + angle_cosine * dy;
                let inside = (local_x / ellipse.major_radius).powi(2)
                    + (local_y / ellipse.minor_radius).powi(2)
                    <= 1.0;
                image.data[y * FRAME_WIDTH + x] = [if inside { 60.0 } else { 220.0 }; 3];
            }
        }
        let support = raw_ring_support(&image, ellipse);
        assert!(support.score > MIN_RAW_RING_SUPPORT_SCORE, "{support:?}");
        assert!(support.positive_fraction > 0.95, "{support:?}");
        assert_eq!(support.strong_sectors, 8);
    }

    #[test]
    fn raw_ring_support_rejects_a_consistent_mask_on_flat_material() {
        let ellipse = Ellipse {
            center: (192.0, 128.0),
            major_radius: 100.0,
            minor_radius: 70.0,
            angle: 0.0,
        };
        let mut image = FloatImage::new(FRAME_WIDTH, FRAME_HEIGHT);
        image.data.fill([120.0; 3]);
        let support = raw_ring_support(&image, ellipse);
        assert!(support.score < MIN_RAW_RING_SUPPORT_SCORE, "{support:?}");
        assert_eq!(support.strong_sectors, 0);
    }

    #[test]
    fn pupil_support_rejects_a_dark_hole_inside_a_screen_reflection() {
        let outer = Ellipse { center: (192.0, 128.0), major_radius: 100.0,
            minor_radius: 78.0, angle: 0.0 };
        let hole = Ellipse { center: outer.center, major_radius: 14.0,
            minor_radius: 8.0, angle: 0.0 };
        let mut image = FloatImage::new(FRAME_WIDTH, FRAME_HEIGHT);
        image.data.fill([220.0; 3]);
        for y in 106..150 {
            for x in 164..220 {
                image.data[y * FRAME_WIDTH + x] = [if ellipse_coordinate((x as f64,y as f64),hole) <= 1.0 {
                    260.0
                } else { 750.0 }; 3];
            }
        }
        assert!(raw_ring_support(&image,hole).score > MIN_PUPIL_VOID_SUPPORT_SCORE);
        let support = raw_ring_support_below_ceiling(&image,hole,
            Some(pupil_iris_luma_ceiling(&image,outer)));
        assert!(!pupil_raw_support_is_sufficient(support), "{support:?}");
    }

    #[test]
    fn pupil_shape_is_assessed_after_limbus_foreshortening() {
        let outer = Ellipse {center:(192.0,128.0),major_radius:100.0,minor_radius:45.0,angle:0.6};
        let pupil = Ellipse {major_radius:30.0,minor_radius:13.5,..outer};
        assert!((pupil_rectified_axis_ratio(pupil,outer).unwrap()-1.0).abs()<1e-8);
        assert!(pupil_ellipse_plausible(pupil,outer));
        let fragment = Ellipse {minor_radius:7.0,..pupil};
        assert!(!pupil_ellipse_plausible(fragment,outer));
        let sideways = Ellipse {angle:outer.angle+std::f64::consts::FRAC_PI_2,..pupil};
        assert!(!pupil_ellipse_plausible(sideways,outer));
        let near_frontal = Ellipse {minor_radius:90.0,..outer};
        let tolerated = Ellipse {major_radius:30.0,minor_radius:20.0,..near_frontal};
        assert!(pupil_ellipse_plausible(tolerated,near_frontal));
    }

    #[test]
    fn pupil_continuity_rejects_size_flashes_but_not_head_translation_or_scale() {
        let outer=Ellipse {center:(192.0,128.0),major_radius:100.0,minor_radius:80.0,angle:0.1};
        let pupil=Ellipse {center:(197.0,130.0),major_radius:30.0,minor_radius:24.0,angle:0.1};
        let mut history=PupilContourHistory::default();
        for timestamp in [1,100_000_001,200_000_001] {history.observe(timestamp,pupil,outer);}
        let prior=history.prior(300_000_001).expect("three independent accepted exposures");
        assert!(prior.admits(pupil,outer));
        let flash=Ellipse {major_radius:15.0,minor_radius:12.0,..pupil};
        assert!(!prior.admits(flash,outer));
        history.observe(300_000_001,flash,outer);
        assert_eq!(history.observations.len(),3,"rejected fit cannot teach size history");
        let transport=|e:Ellipse|Ellipse {center:(e.center.0*1.4+160.0,e.center.1*1.4-45.0),
            major_radius:e.major_radius*1.4,minor_radius:e.minor_radius*1.4,..e};
        assert!(prior.admits(transport(pupil),transport(outer)));
        assert!(history.prior(1_300_000_001).is_none(),"old size cannot constrain a new acquisition");
        assert!(history.prior(1).is_none(),"reversed time cannot reuse future evidence");
    }

    #[test]
    fn pupil_continuity_allows_gradual_dilation_without_learning_a_fragment() {
        let outer=Ellipse {center:(192.0,128.0),major_radius:100.0,minor_radius:80.0,angle:0.0};
        let mut history=PupilContourHistory::default();
        for index in 0..25 {
            let ratio=0.30*(index as f64*0.008).exp();
            let pupil=Ellipse {major_radius:100.0*ratio,minor_radius:80.0*ratio,..outer};
            let time=1+index*100_000_000;
            if let Some(prior)=history.prior(time) {assert!(prior.admits(pupil,outer));}
            history.observe(time,pupil,outer);
        }
        assert_eq!(history.observations.len(),7);
        assert!(history.observations.back().unwrap().log_radius_ratio.exp()>0.35);
    }

    #[test]
    fn pupil_prior_activates_at_live_two_eye_inference_cadence_but_expires_when_stale() {
        let outer = Ellipse {
            center: (192.0, 128.0), major_radius: 100.0, minor_radius: 80.0, angle: 0.0,
        };
        let pupil = Ellipse { major_radius: 30.0, minor_radius: 24.0, ..outer };
        let mut history = PupilContourHistory::default();
        for time in [1, 700_000_001, 1_400_000_001] {
            history.observe(time, pupil, outer);
        }
        let prior = history.prior(2_100_000_001)
            .expect("three independent observations remain available at the real worker cadence");
        assert!(prior.admits(pupil, outer));
        assert!(!prior.admits(Ellipse { major_radius: 15.0, minor_radius: 12.0, ..pupil }, outer));
        assert!(history.prior(2_500_000_001).is_none(), "history never extends freshness");
    }

    #[test]
    fn semantic_pupil_miss_cannot_cold_acquire_a_raw_fragment() {
        let outer = Ellipse {
            center: (192.0, 128.0), major_radius: 100.0, minor_radius: 80.0, angle: 0.0,
        };
        let fit = PupilVoidFitReview {
            ellipse: Ellipse { major_radius: 30.0, minor_radius: 24.0, ..outer },
            raw_support: RawRingSupport {
                score: 2.8, points: 160, positive_fraction: 0.8, strong_sectors: 6,
            },
        };
        let prior = PupilFitPrior {
            radius_ratio: 0.3, center_offset: (0.0, 0.0),
            log_radius_half_width: 0.13, center_half_width: 0.2,
        };
        assert!(choose_pupil_observation(true, None, Some(fit), None, outer, None).is_none());
        assert!(choose_pupil_observation(true, None, None, Some(fit), outer, None).is_none());
        let (_, independent) = choose_pupil_observation(
            true, None, Some(fit), Some(fit), outer, Some(prior)).unwrap();
        assert!(!independent, "RAW recovery cannot teach the SAM pupil prior");
        let (_, independent) = choose_pupil_observation(
            true, Some(fit), None, None, outer, None).unwrap();
        assert!(independent);
        assert!(choose_pupil_observation(false, None, Some(fit), None, outer, None).is_some());
    }

    #[test]
    fn pupil_temporal_refit_requires_new_raw_edges() {
        let outer=Ellipse {center:(192.0,128.0),major_radius:100.0,minor_radius:80.0,angle:0.0};
        let prior=PupilFitPrior {radius_ratio:0.3,center_offset:(0.0,0.0),
            log_radius_half_width:0.13,center_half_width:0.2};
        let mut image=FloatImage::new(FRAME_WIDTH,FRAME_HEIGHT);image.data.fill([220.0;3]);
        assert!(refit_pupil_from_prior(&image,outer,prior).is_none());
        let pupil=Ellipse {major_radius:30.0,minor_radius:24.0,..outer};
        for y in 0..FRAME_HEIGHT {for x in 0..FRAME_WIDTH {
            if ellipse_coordinate((x as f64,y as f64),pupil)<=1.0 {image.data[y*FRAME_WIDTH+x]=[80.0;3];}
        }}
        let fitted=refit_pupil_from_prior(&image,outer,prior).expect("current visible pupil");
        assert!((fitted.ellipse.center.0-pupil.center.0).hypot(fitted.ellipse.center.1-pupil.center.1)<5.0);
        assert!((fitted.ellipse.major_radius-pupil.major_radius).abs()<4.0);
    }

    #[test]
    fn pupil_recovery_competes_with_fragments_without_teaching_its_own_prior() {
        let outer = Ellipse { center: (192.0, 128.0), major_radius: 100.0,
            minor_radius: 80.0, angle: 0.0 };
        let prior = PupilFitPrior { radius_ratio: 0.3, center_offset: (0.0, 0.0),
            log_radius_half_width: 0.13, center_half_width: 0.2 };
        let supported = PupilVoidFitReview {
            ellipse: Ellipse { major_radius: 30.0, minor_radius: 24.0, ..outer },
            raw_support: RawRingSupport { score: 2.8, points: 160,
                positive_fraction: 0.8, strong_sectors: 6 },
        };
        let fragment = PupilVoidFitReview { ellipse: Ellipse {
            center: (205.0, 138.0), ..supported.ellipse },
            raw_support: RawRingSupport { score: 2.9, ..supported.raw_support } };
        let (selected, independent) = choose_current_pupil_recovery(
            Some(fragment), Some(supported), outer, Some(prior)).unwrap();
        assert_eq!(selected.ellipse, supported.ellipse);
        assert!(!independent, "guided recovery must not refresh independent contour history");
        let (_, independent) = choose_current_pupil_recovery(
            Some(supported), None, outer, Some(prior)).unwrap();
        assert!(independent);
        assert!(choose_current_pupil_recovery(None, None, outer, Some(prior)).is_none());
    }

    #[test]
    fn pupil_support_keeps_visible_boundary_beside_a_large_glint() {
        let outer = Ellipse { center: (192.0, 128.0), major_radius: 100.0,
            minor_radius: 78.0, angle: 0.0 };
        let pupil = Ellipse { center: outer.center, major_radius: 30.0,
            minor_radius: 24.0, angle: 0.0 };
        let mut image = FloatImage::new(FRAME_WIDTH, FRAME_HEIGHT);
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                let p=(x as f64,y as f64);
                let value = if (185..215).contains(&x) && (114..144).contains(&y) { 750.0 }
                    else if ellipse_coordinate(p,pupil) <= 1.0 { 120.0 }
                    else if ellipse_coordinate(p,outer) <= 1.0 { 220.0 }
                    else { 480.0 };
                image.data[y * FRAME_WIDTH + x] = [value; 3];
            }
        }
        let support = raw_ring_support_below_ceiling(&image,pupil,
            Some(pupil_iris_luma_ceiling(&image,outer)));
        assert!(pupil_raw_support_is_sufficient(support), "{support:?}");
        assert!(support.points < 192, "reflection must actually be excluded: {support:?}");
    }

    #[test]
    fn unprompted_sam_pupil_uses_its_own_outer_center_not_a_registration_anchor() {
        let outer = Ellipse {
            center: (192.0, 128.0),
            major_radius: 100.0,
            minor_radius: 78.0,
            angle: 0.0,
        };
        let central = Ellipse {
            center: (192.0, 128.0),
            major_radius: 25.0,
            minor_radius: 22.0,
            angle: 0.0,
        };
        let distractor = Ellipse {
            center: (242.0, 128.0),
            major_radius: 18.0,
            minor_radius: 16.0,
            angle: 0.0,
        };
        let mut image = FloatImage::new(FRAME_WIDTH, FRAME_HEIGHT);
        image.data.fill([440.0; 3]);
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                let point = (x as f64, y as f64);
                let value = if ellipse_coordinate(point, central) <= 1.0
                    || ellipse_coordinate(point, distractor) <= 1.0
                {
                    45.0
                } else if ellipse_coordinate(point, outer) <= 1.0 {
                    280.0
                } else {
                    440.0
                };
                image.data[y * FRAME_WIDTH + x] = [value; 3];
            }
        }

        let (unprompted, _) =
            fit_inner_pupil_void(&image, outer, None).expect("unprompted central pupil");
        assert!(
            (unprompted.center.0 - central.center.0).hypot(unprompted.center.1 - central.center.1)
                < 3.0,
            "unprompted={unprompted:?}",
        );

        // This demonstrates why an asynchronous registration anchor cannot
        // share the component-seed field: a seed near the second dark object
        // deliberately changes the selected pupil.
        let (prompted, _) = fit_inner_pupil_void(&image, outer, Some(distractor.center))
            .expect("explicit distractor component seed");
        assert!(
            (prompted.center.0 - distractor.center.0)
                .hypot(prompted.center.1 - distractor.center.1)
                < 3.0,
            "prompted={prompted:?}",
        );
    }

    #[test]
    fn pupil_flat_tire_refits_the_visible_arc_without_the_upper_lid_chord() {
        let expected = Ellipse {
            center: (192.0, 128.0),
            major_radius: 27.0,
            minor_radius: 23.0,
            angle: 0.0,
        };
        let component: Vec<_> = (0..FRAME_WIDTH * FRAME_HEIGHT)
            .filter(|&index| {
                let point = ((index % FRAME_WIDTH) as f64, (index / FRAME_WIDTH) as f64);
                ellipse_coordinate(point, expected) <= 1.0 && point.1 >= 117.0
            })
            .collect();
        let points: Vec<_> = component
            .iter()
            .map(|&index| ((index % FRAME_WIDTH) as f64, (index / FRAME_WIDTH) as f64))
            .collect();
        let reference = moments_ellipse(&points).unwrap();
        let fit = deflattened_pupil_component(&component, FRAME_WIDTH, FRAME_HEIGHT, reference)
            .expect("surviving curved pupil boundary");
        assert!((fit.center.1 - expected.center.1).abs() < 2.0, "{fit:?}");
        assert!(
            (fit.minor_radius - expected.minor_radius).abs() < 2.5,
            "{fit:?}"
        );
        assert!((reference.center.1 - expected.center.1).abs() > 3.0);
    }

    #[test]
    fn pupil_flat_tire_rejects_a_polygonal_shadow_without_curved_support() {
        let component: Vec<_> = (110..140)
            .flat_map(|y| (170..220).map(move |x| y * FRAME_WIDTH + x))
            .collect();
        let points: Vec<_> = component
            .iter()
            .map(|&index| ((index % FRAME_WIDTH) as f64, (index / FRAME_WIDTH) as f64))
            .collect();
        assert!(deflattened_pupil_component(
            &component,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            moments_ellipse(&points).unwrap()
        )
        .is_none());
    }

    #[test]
    fn pupil_flat_tire_tolerates_source_pixel_noise_after_magnification() {
        let expected = Ellipse {
            center: (192.0, 128.0),
            major_radius: 25.0,
            minor_radius: 21.0,
            angle: 0.0,
        };
        let component: Vec<_> = (0..FRAME_WIDTH * FRAME_HEIGHT)
            .filter(|&index| {
                let p = ((index % FRAME_WIDTH) as f64, (index / FRAME_WIDTH) as f64);
                let phase = ((p.1 - expected.center.1) / expected.minor_radius)
                    .atan2((p.0 - expected.center.0) / expected.major_radius);
                let noise = 1.5 * (11.0 * phase).sin() + 0.5 * (23.0 * phase).cos();
                ellipse_coordinate(p, expected) <= 1.0 + noise / expected.major_radius
                    && p.1 >= 117.0
            })
            .collect();
        let points: Vec<_> = component
            .iter()
            .map(|&i| ((i % FRAME_WIDTH) as f64, (i / FRAME_WIDTH) as f64))
            .collect();
        let fit = deflattened_pupil_component(
            &component,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            moments_ellipse(&points).unwrap(),
        )
        .expect("noisy but curved pupil arc");
        assert!(
            (fit.center.0 - expected.center.0).hypot(fit.center.1 - expected.center.1) < 3.0,
            "{fit:?}"
        );
        assert!(
            (fit.major_radius - expected.major_radius).abs() < 3.0,
            "{fit:?}"
        );
        assert!(
            (fit.minor_radius - expected.minor_radius).abs() < 3.0,
            "{fit:?}"
        );
    }

    #[test]
    fn rejected_dominant_void_does_not_promote_a_small_iris_fragment() {
        let outer = Ellipse {
            center: (192.0, 128.0),
            major_radius: 100.0,
            minor_radius: 80.0,
            angle: 0.0,
        };
        let fragment = Ellipse {
            center: (248.0, 128.0),
            major_radius: 14.0,
            minor_radius: 12.0,
            angle: 0.0,
        };
        let mut image = FloatImage::new(FRAME_WIDTH, FRAME_HEIGHT);
        image.data.fill([280.0; 3]);
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                if ((167..217).contains(&x) && (108..148).contains(&y))
                    || ellipse_coordinate((x as f64, y as f64), fragment) <= 1.0
                {
                    image.data[y * FRAME_WIDTH + x] = [45.0; 3];
                }
            }
        }
        let mut diagnostics = PupilFitDiagnostics::default();
        assert!(fit_inner_pupil_void_diagnostic(&image, outer, None, &mut diagnostics).is_none());
        assert!(diagnostics.competing_component_rejected || diagnostics.geometry_rejected > 0,
            "the fragment must fail either the central-acquisition corridor or the competing-component veto: {diagnostics:?}");
        assert!(fit_inner_pupil_void(&image, outer, Some(fragment.center)).is_some());
    }

    #[test]
    fn ransac_outliers_cannot_expand_their_own_refit_admission() {
        let expected = Ellipse {
            center: (192.0, 128.0),
            major_radius: 95.0,
            minor_radius: 72.0,
            angle: 0.2,
        };
        let points = expected
            .dense_points(128)
            .into_iter()
            .enumerate()
            .map(|(i, p)| {
                let displacement = if i % 3 == 0 {
                    22.0
                } else {
                    3.0 * (i as f64 * 1.7).sin()
                };
                let dx = p.0 - expected.center.0;
                let dy = p.1 - expected.center.1;
                let length = dx.hypot(dy);
                (
                    p.0 + dx / length * displacement,
                    p.1 + dy / length * displacement,
                )
            })
            .collect::<Vec<_>>();
        let fit = robust_ransac_ellipse(&points, &mut NumpyPcg64::baseline_fit_stream()).unwrap();
        assert!(fit.cutoff <= 4.0, "cutoff={}", fit.cutoff);
        assert!(
            (fit.ellipse.center.0 - expected.center.0)
                .hypot(fit.ellipse.center.1 - expected.center.1)
                < 3.0
        );
    }

    fn unpack_reference_raw10(payload: &[u8]) -> Vec<u16> {
        let mut raw = Vec::with_capacity(FRAME_WIDTH * FRAME_HEIGHT);
        for group in payload.chunks_exact(5) {
            let word = group.iter().enumerate().fold(0u64, |word, (lane, byte)| {
                word | (*byte as u64) << (8 * lane)
            });
            for lane in 0..4 {
                raw.push(((word >> (10 * lane)) & 0x3ff) as u16);
            }
        }
        raw
    }

    #[test]
    #[ignore = "requires an external six-query RAW10 reference directory"]
    fn recorded_adapters_match_six_query_input_bytes() {
        let root = std::env::var_os("BUTTERCUP_SAM31_REFERENCE_DIR")
            .map(PathBuf::from)
            .expect("BUTTERCUP_SAM31_REFERENCE_DIR must name the external fixture directory");
        let frames = (1..=HISTORY_FRAMES)
            .map(|index| {
                let payload = std::fs::read(root.join(format!("frame-{index:02}.raw10"))).unwrap();
                Arc::new(RawFrame {
                    eye_index: 0,
                    sequence: index as u64,
                    timestamp_ns: index as u64,
                    sensor_x: 3964,
                    sensor_y: 3356,
                    width: FRAME_WIDTH,
                    height: FRAME_HEIGHT,
                    registration_anchor: None,
                    pupil_component_seed: None,
                    pixels: Arc::new(unpack_reference_raw10(&payload)),
                })
            })
            .collect::<Vec<_>>();
        let mut bytes = vec![0u8; FILMSTRIP_PIXELS * 3];
        let balanced = balanced_quad_rgb(&frames);
        write_quantized_filmstrip(&balanced, 0.35, 99.65, 0.82, &mut bytes).unwrap();
        assert_eq!(fnv1a(&bytes), 0xac60_b7e8_f392_4dc5);
        let luma = raw_luma(&frames);
        write_quantized_filmstrip(&luma, 0.20, 99.75, 0.80, &mut bytes).unwrap();
        assert_eq!(fnv1a(&bytes), 0x37d2_623e_2277_204a);
        let chroma = log_chroma(&balanced);
        write_quantized_filmstrip(&chroma, 0.60, 99.40, 1.0, &mut bytes).unwrap();
        // Rust/libm and OpenCV differ by one output level at 1.82% of the
        // Gaussian/log2 chroma pixels; every difference is exactly one LSB.
        // Pin the verified Rust result so that this bounded numerical parity
        // cannot silently regress.
        assert_eq!(fnv1a(&bytes), 0x8ec9_c694_5e16_53f9);
    }

    #[test]
    fn enclosing_limbus_requires_both_containment_and_better_raw_evidence() {
        let inner = Ellipse {center:(190.0,130.0),major_radius:60.0,minor_radius:45.0,angle:0.0};
        let outer = Ellipse {center:(192.0,128.0),major_radius:100.0,minor_radius:80.0,angle:0.0};
        assert!(outer_limbus_candidate_supersedes(outer,3.4,0.01,inner,2.6,0.02));
        assert!(!outer_limbus_candidate_supersedes(outer,2.5,0.01,inner,2.6,0.02));
        assert!(!outer_limbus_candidate_supersedes(inner,4.0,0.01,outer,2.6,0.02));
        assert!(!outer_limbus_candidate_supersedes(Ellipse {center:(50.0,40.0),..outer},4.0,0.01,inner,2.6,0.02));
        assert!(!outer_limbus_candidate_supersedes(outer,3.4,0.0001,inner,2.6,0.02));
    }

    #[test]
    fn ordinary_small_iris_mask_reaches_constrained_fit() {
        let expected = Ellipse { center: (192.0,128.0),major_radius: 72.0,minor_radius: 54.0,angle: -0.2 };
        let width = FILMSTRIP_WIDTH;
        let mut mask = vec![0; width * FRAME_HEIGHT];
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                if ellipse_coordinate((x as f64,y as f64),expected) <= 1.0 {
                    mask[y*width + (HISTORY_FRAMES-1)*FRAME_WIDTH+x] = 255;
                }
            }
        }
        let area = mask.iter().filter(|&&value|value != 0).count();
        assert!(area < 20_000 && area > MIN_COMPONENT_AREA_FULL_RES);
        let fit = fit_mask_component_review(&mask,width,FRAME_HEIGHT,HISTORY_FRAMES-1)
            .expect("ordinary smaller iris must reach the conic/RAW gates");
        assert!((fit.ellipse.center.0-expected.center.0).hypot(fit.ellipse.center.1-expected.center.1)<2.0);
        assert!((fit.ellipse.major_radius-expected.major_radius).abs()<2.0);
    }

    #[test]
    fn nonlinear_fit_recovers_synthetic_ellipse_after_lid_cull() {
        let expected = Ellipse {
            center: (188.4, 129.7),
            major_radius: 96.2,
            minor_radius: 67.8,
            angle: -0.21,
        };
        let points = expected.dense_points(180);
        let initial = Ellipse {
            center: (185.0, 132.0),
            major_radius: 91.0,
            minor_radius: 72.0,
            angle: -0.15,
        };
        let fitted = robust_contour_fit(&points, initial).unwrap();
        assert!((fitted.center.0 - expected.center.0).abs() < 0.02);
        assert!((fitted.center.1 - expected.center.1).abs() < 0.02);
        assert!((fitted.major_radius - expected.major_radius).abs() < 0.02);
        assert!((fitted.minor_radius - expected.minor_radius).abs() < 0.02);
    }

    fn ellipse_point_at(ellipse: Ellipse, phase: f64) -> (f64, f64) {
        let (phase_sine, phase_cosine) = phase.sin_cos();
        let (angle_sine, angle_cosine) = ellipse.angle.sin_cos();
        let local_x = ellipse.major_radius * phase_cosine;
        let local_y = ellipse.minor_radius * phase_sine;
        (
            ellipse.center.0 + angle_cosine * local_x - angle_sine * local_y,
            ellipse.center.1 + angle_sine * local_x + angle_cosine * local_y,
        )
    }

    fn line_samples(first: (f64, f64), second: (f64, f64), count: usize) -> Vec<(f64, f64)> {
        (0..count)
            .map(|index| {
                let blend = index as f64 / (count - 1).max(1) as f64;
                (
                    first.0 * (1.0 - blend) + second.0 * blend,
                    first.1 * (1.0 - blend) + second.1 * blend,
                )
            })
            .collect()
    }

    fn arc_samples(
        ellipse: Ellipse,
        first_phase: f64,
        last_phase: f64,
        count: usize,
    ) -> Vec<(f64, f64)> {
        (0..count)
            .map(|index| {
                let blend = index as f64 / (count - 1).max(1) as f64;
                ellipse_point_at(ellipse, first_phase * (1.0 - blend) + last_phase * blend)
            })
            .collect()
    }

    fn assert_ellipse_close(actual: Ellipse, expected: Ellipse, tolerance: f64) {
        assert!(
            (actual.center.0 - expected.center.0).abs() <= tolerance,
            "actual={actual:?} expected={expected:?}",
        );
        assert!(
            (actual.center.1 - expected.center.1).abs() <= tolerance,
            "actual={actual:?} expected={expected:?}",
        );
        assert!(
            (actual.major_radius - expected.major_radius).abs() <= tolerance,
            "actual={actual:?} expected={expected:?}",
        );
        assert!(
            (actual.minor_radius - expected.minor_radius).abs() <= tolerance,
            "actual={actual:?} expected={expected:?}",
        );
    }

    #[test]
    fn de_flat_tire_recovers_ellipse_from_upper_and_lower_occlusion_chords() {
        let expected = Ellipse {
            center: (191.0, 128.0),
            major_radius: 102.0,
            minor_radius: 75.0,
            angle: 0.08,
        };
        let upper_level = -0.56f64;
        let lower_level = 0.62f64;
        let upper_right = std::f64::consts::TAU + upper_level.asin();
        let upper_left = std::f64::consts::PI - upper_level.asin();
        let lower_right = std::f64::consts::TAU + lower_level.asin();
        let lower_left = std::f64::consts::PI - lower_level.asin();
        let mut contour = line_samples(
            ellipse_point_at(expected, upper_left),
            ellipse_point_at(expected, upper_right),
            70,
        );
        contour.extend(arc_samples(expected, upper_right, lower_right, 65));
        contour.extend(line_samples(
            ellipse_point_at(expected, lower_right),
            ellipse_point_at(expected, lower_left),
            70,
        ));
        contour.extend(arc_samples(expected, lower_left, upper_left, 65));

        let crude_reference = Ellipse {
            center: (190.0, 131.0),
            major_radius: 94.0,
            minor_radius: 48.0,
            angle: 0.07,
        };
        let fit = deflattened_mask_fit(contour, crude_reference).expect("de-flat-tired fit");
        assert!(fit.upper_flat_tire, "{fit:?}");
        assert!(fit.lower_flat_tire, "{fit:?}");
        assert!(fit.flat_tire_points.len() >= 45, "{fit:?}");
        assert_ellipse_close(fit.ellipse, expected, 2.0);
    }

    #[test]
    fn physical_curvature_veto_censors_an_oblique_foreground_chord() {
        let expected = Ellipse {
            center: (190.0, 126.0),
            major_radius: 104.0,
            minor_radius: 72.0,
            angle: -0.11,
        };
        // A steep chord on the camera-right side: neither an upper nor lower
        // eyelid heuristic can identify it, so only the general scale-aware
        // curvature veto may remove it.
        let first_phase = -0.72;
        let last_phase = 0.72;
        let mut contour = line_samples(
            ellipse_point_at(expected, first_phase),
            ellipse_point_at(expected, last_phase),
            65,
        );
        contour.extend(arc_samples(
            expected,
            last_phase,
            first_phase + std::f64::consts::TAU,
            190,
        ));
        let fit = deflattened_mask_fit(contour, expected).expect("oblique-chord fit");
        assert!(!fit.upper_flat_tire, "{fit:?}");
        assert!(!fit.lower_flat_tire, "{fit:?}");
        assert!(fit.flat_tire_points.len() >= 18, "{fit:?}");
        assert_ellipse_close(fit.ellipse, expected, 1.5);
    }

    #[test]
    fn an_unoccluded_outer_disk_keeps_its_real_curvature() {
        let expected = Ellipse {
            center: (193.0, 127.0),
            major_radius: 98.0,
            minor_radius: 69.0,
            angle: 0.16,
        };
        let fit = deflattened_mask_fit(expected.dense_points(240), expected)
            .expect("complete ellipse fit");
        assert!(fit.flat_tire_points.is_empty(), "{fit:?}");
        assert_ellipse_close(fit.ellipse, expected, 0.2);
    }

    #[test]
    fn conic_arc_search_recovers_disk_beneath_a_bowed_upper_lid() {
        let expected = Ellipse {
            center: (192.0, 128.0),
            major_radius: 100.0,
            minor_radius: 75.0,
            angle: 0.0,
        };
        for rotation in [0.0, 0.65, -0.8, std::f64::consts::PI] {
            let level = -0.35f64;
            let half_width = expected.major_radius * (1.0 - level * level).sqrt();
            let mut contour = (0..80)
                .map(|i| {
                    let u = -1.0 + 2.0 * i as f64 / 79.0;
                    (
                        expected.center.0 + half_width * u,
                        expected.center.1 + level * expected.minor_radius - 18.0 * (1.0 - u * u),
                    )
                })
                .collect::<Vec<_>>();
            contour.extend(arc_samples(
                expected,
                level.asin(),
                std::f64::consts::PI - level.asin(),
                160,
            ));
            let (s, c) = f64::sin_cos(rotation);
            for p in &mut contour {
                let dx = p.0 - expected.center.0;
                let dy = p.1 - expected.center.1;
                *p = (
                    expected.center.0 + c * dx - s * dy,
                    expected.center.1 + s * dx + c * dy,
                );
            }
            let expected = Ellipse {
                angle: rotation,
                ..expected
            };
            let fit =
                deflattened_mask_fit(contour, expected).expect("compatible side and lower arcs");
            assert_ellipse_close(fit.ellipse, expected, 3.0);
        }
    }

    #[test]
    fn conic_arc_constraints_reject_short_support_and_incompatible_tangents() {
        let ellipse = Ellipse {
            center: (192.0, 128.0),
            major_radius: 100.0,
            minor_radius: 75.0,
            angle: 0.0,
        };
        let points = arc_samples(ellipse, 0.0, 1.0, 40);
        let constraints = ConicArcConstraints {
            tangents: vec![Some((0.0, 1.0)); points.len()],
        };
        assert!(!constraints.admits(&points, &vec![true; points.len()], ellipse, 4.0));
        assert!(constraints.tangent_agrees(0, (292.0, 128.0), ellipse));
        assert!(!constraints.tangent_agrees(0, (192.0, 53.0), ellipse));
        let mut opposed = arc_samples(ellipse, -0.1, 0.1, 40);
        opposed.extend(arc_samples(
            ellipse,
            std::f64::consts::PI - 0.1,
            std::f64::consts::PI + 0.1,
            40,
        ));
        assert!(!constraints.admits(&opposed, &vec![true; 80], ellipse, 4.0));
    }

    #[test]
    fn consensus_uses_doubled_angle_and_component_medians() {
        let ellipses = [
            Ellipse {
                center: (100.0, 80.0),
                major_radius: 60.0,
                minor_radius: 45.0,
                angle: 1.55,
            },
            Ellipse {
                center: (102.0, 79.0),
                major_radius: 62.0,
                minor_radius: 44.0,
                angle: -1.56,
            },
            Ellipse {
                center: (101.0, 81.0),
                major_radius: 61.0,
                minor_radius: 46.0,
                angle: 1.56,
            },
        ];
        let result = consensus(&ellipses).unwrap();
        assert_eq!(result.center, (101.0, 80.0));
        assert_eq!(result.major_radius, 61.0);
        assert_eq!(result.minor_radius, 45.0);
        assert!(result.angle.abs() > 1.5);
    }

    #[test]
    fn adapter_agreement_accepts_recorded_eye_cluster() {
        let candidates = [
            (
                Ellipse {
                    center: (225.14, 112.13),
                    major_radius: 160.73,
                    minor_radius: 129.83,
                    angle: -0.302,
                },
                0.46,
            ),
            (
                Ellipse {
                    center: (225.41, 121.14),
                    major_radius: 155.60,
                    minor_radius: 143.64,
                    angle: -0.038,
                },
                0.32,
            ),
            (
                Ellipse {
                    center: (226.57, 112.52),
                    major_radius: 159.13,
                    minor_radius: 131.19,
                    angle: -0.252,
                },
                0.09,
            ),
        ];
        let (ellipse, agreeing) = agreeing_consensus(&candidates).unwrap();
        assert_eq!(agreeing.len(), 3);
        assert!((ellipse.center.0 - 225.41).abs() < 0.02);
        assert!((ellipse.center.1 - 112.52).abs() < 0.02);
    }

    #[test]
    fn adapter_agreement_rejects_unrelated_live_components() {
        let candidates = [
            (
                Ellipse {
                    center: (315.20, 89.27),
                    major_radius: 177.41,
                    minor_radius: 68.18,
                    angle: 1.546,
                },
                0.12,
            ),
            (
                Ellipse {
                    center: (307.36, 120.40),
                    major_radius: 98.43,
                    minor_radius: 85.87,
                    angle: 1.429,
                },
                0.08,
            ),
        ];
        assert!(agreeing_consensus(&candidates).is_none());
    }

    #[test]
    fn adapter_agreement_accepts_lid_occluded_live_iris() {
        let candidates = [
            (
                Ellipse {
                    center: (140.602, 168.841),
                    major_radius: 141.419,
                    minor_radius: 118.205,
                    angle: -0.146,
                },
                0.014,
            ),
            (
                Ellipse {
                    center: (133.792, 130.339),
                    major_radius: 149.771,
                    minor_radius: 135.558,
                    angle: 0.124,
                },
                0.028,
            ),
        ];
        let (ellipse, agreeing) = agreeing_consensus(&candidates).unwrap();
        assert_eq!(agreeing.len(), 2);
        assert!((ellipse.center.0 - 137.197).abs() < 0.01);
        assert!((ellipse.center.1 - 149.590).abs() < 0.01);
    }

    #[test]
    fn quantized_filmstrip_is_planar_chw_and_tile_ordered() {
        let images = (0..HISTORY_FRAMES)
            .map(|index| FloatImage {
                width: FRAME_WIDTH,
                height: FRAME_HEIGHT,
                data: vec![
                    [index as f32, index as f32 + 1.0, index as f32 + 2.0];
                    FRAME_WIDTH * FRAME_HEIGHT
                ],
            })
            .collect::<Vec<_>>();
        let mut destination = vec![0u8; FILMSTRIP_PIXELS * 3];
        write_quantized_filmstrip(&images, 0.0, 100.0, 1.0, &mut destination).unwrap();
        assert_eq!(destination[0], 0);
        assert!(destination[FRAME_WIDTH * 4] > destination[0]);
        assert!(destination[FILMSTRIP_PIXELS + FRAME_WIDTH * 4] > destination[FILMSTRIP_PIXELS]);
    }

    #[test]
    fn preprocessed_frame_split_preserves_each_planar_tile_exactly() {
        let frame_pixels = FRAME_WIDTH * FRAME_HEIGHT;
        let mut filmstrip = vec![0u8; FILMSTRIP_PIXELS * 3];
        for channel in 0..3 {
            for frame in 0..HISTORY_FRAMES {
                for y in 0..FRAME_HEIGHT {
                    let start =
                        channel * FILMSTRIP_PIXELS + y * FILMSTRIP_WIDTH + frame * FRAME_WIDTH;
                    filmstrip[start..start + FRAME_WIDTH]
                        .fill((channel * 50 + frame * 7 + y % 5) as u8);
                }
            }
        }
        for frame in 0..HISTORY_FRAMES {
            let mut split = vec![0u8; frame_pixels * 3];
            extract_preprocessed_frame(&filmstrip, frame, &mut split).unwrap();
            for channel in 0..3 {
                for y in 0..FRAME_HEIGHT {
                    let expected = (channel * 50 + frame * 7 + y % 5) as u8;
                    let start = channel * frame_pixels + y * FRAME_WIDTH;
                    assert!(split[start..start + FRAME_WIDTH]
                        .iter()
                        .all(|&value| value == expected));
                }
            }
        }
    }

    #[test]
    fn preprocessed_frame_split_supports_long_video_windows() {
        let frame_count = 9;
        let frame_pixels = FRAME_WIDTH * FRAME_HEIGHT;
        let film_width = FRAME_WIDTH * frame_count;
        let film_pixels = film_width * FRAME_HEIGHT;
        let mut filmstrip = vec![0u8; film_pixels * 3];
        for channel in 0..3 {
            for frame in 0..frame_count {
                for y in 0..FRAME_HEIGHT {
                    let start = channel * film_pixels + y * film_width + frame * FRAME_WIDTH;
                    filmstrip[start..start + FRAME_WIDTH]
                        .fill((channel * 60 + frame * 9 + y % 7) as u8);
                }
            }
        }
        for frame in 0..frame_count {
            let mut split = vec![0u8; frame_pixels * 3];
            extract_preprocessed_frame(&filmstrip, frame, &mut split).unwrap();
            for channel in 0..3 {
                for y in 0..FRAME_HEIGHT {
                    let expected = (channel * 60 + frame * 9 + y % 7) as u8;
                    let start = channel * frame_pixels + y * FRAME_WIDTH;
                    assert!(split[start..start + FRAME_WIDTH]
                        .iter()
                        .all(|&value| value == expected));
                }
            }
        }
    }

    #[test]
    fn proposal_boundary_indices_keep_only_the_four_connected_perimeter() {
        let width = 7;
        let height = 7;
        let mut mask = vec![0u8; width * height];
        for y in 2..=4 {
            for x in 2..=4 {
                mask[y * width + x] = 1;
            }
        }
        let boundary = binary_mask_boundary_indices(&mask, width, height);
        assert_eq!(boundary.len(), 8);
        assert!(!boundary.contains(&((3 * width + 3) as u32)));
        assert!(boundary.contains(&((2 * width + 2) as u32)));
        assert!(binary_mask_boundary_indices(&mask[..10], width, height).is_empty());
    }
}
