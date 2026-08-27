//! Native green/white checkerboard calibration for the live RAW10 viewer.
//!
//! The physical contract is deliberately narrow: seven interior intersections
//! describe six 17 mm intervals (102 mm total) in each direction.  A larger
//! detected lattice is cropped to its central 7x7 window, so rounded or warped
//! decorative squares at the outside of the target never enter the fit.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;

#[cfg(feature = "checkerboard")]
use std::fs;
#[cfg(feature = "checkerboard")]
use std::sync::mpsc::RecvTimeoutError;
#[cfg(feature = "checkerboard")]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const SQUARE_MM: f64 = 102.0 / 6.0;
pub const CENTRAL_SQUARES: usize = 6;
pub const INNER_CORNERS: usize = CENTRAL_SQUARES + 1;
pub const CENTRAL_SPAN_MM: f64 = SQUARE_MM * CENTRAL_SQUARES as f64;

const MIN_VIEWS: usize = 15;
const MAX_TRAINING_RMS_PX: f64 = 2.5;
const MAX_HOLDOUT_RMS_PX: f64 = 4.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleLayout {
    /// Exact 1x1 Quad Bayer sensor samples from a live ROI.
    QuadBayerRaw10,
    /// Exact little-endian samples from the camera service's full-aperture
    /// Bayer-aware linear reduction.  `sensor_extent` describes the physical
    /// aperture represented by the sample plane.
    LinearGray16,
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
    pub layout: SampleLayout,
    /// Physical sensor extent represented by this sample plane.  It equals
    /// `width x height` for a 1x1 ROI and 8000x6000 for the 512x384 full-frame
    /// linear calibration snapshot.
    pub sensor_extent: (u32, u32),
    pub focus_target: u16,
    pub focus_position: u16,
    pub focus_settled: bool,
    pub focus_generation: u32,
    pub pixels: Arc<Vec<u16>>,
}

#[derive(Clone, Debug, Default)]
pub struct Overlay {
    pub eye_index: usize,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub sensor_origin: (u32, u32),
    pub sample_size: (usize, usize),
    pub sensor_scale: (f64, f64),
    pub full_sensor: bool,
    /// Lossless sensor-derived preview used for live review.  This is never a
    /// desktop or application capture.
    pub preview: Option<Arc<Vec<u32>>>,
    /// Central 7x7 intersections in sample-plane coordinates.
    pub points: Vec<(f64, f64)>,
    /// Reprojection of the same intersections after a camera fit exists.
    pub reprojected: Vec<(f64, f64)>,
    /// Projected origin, +X, +Y, and +Z endpoints for live 3D review.
    pub pose_axes: Vec<(f64, f64)>,
    pub accepted_view: bool,
}

#[derive(Clone, Debug)]
pub struct StatusSnapshot {
    pub active: bool,
    pub generation: u64,
    pub state: String,
    pub detail: String,
    pub frames_seen: u64,
    pub detections: u64,
    pub accepted_views: usize,
    pub duplicate_views: u64,
    pub rejected_views: u64,
    pub dropped_frames: u64,
    pub focus_position: Option<u16>,
    pub training_rms_px: Option<f64>,
    pub holdout_rms_px: Option<f64>,
    pub calibrated: bool,
    pub output: Option<PathBuf>,
    pub overlay: Option<Overlay>,
}

impl Default for StatusSnapshot {
    fn default() -> Self {
        Self {
            active: false,
            generation: 0,
            state: "off".to_string(),
            detail: format!(
                "press C to collect the central {CENTRAL_SQUARES}x{CENTRAL_SQUARES} squares at {SQUARE_MM:.1} mm pitch"
            ),
            frames_seen: 0,
            detections: 0,
            accepted_views: 0,
            duplicate_views: 0,
            rejected_views: 0,
            dropped_frames: 0,
            focus_position: None,
            training_rms_px: None,
            holdout_rms_px: None,
            calibrated: false,
            output: None,
            overlay: None,
        }
    }
}

impl StatusSnapshot {
    pub fn one_line(&self) -> String {
        if !self.active && self.state == "off" {
            return "CHECKER C START  CENTRAL 6x6 = 102.0MM".to_string();
        }
        let rms = match (self.training_rms_px, self.holdout_rms_px) {
            (Some(train), Some(holdout)) => format!(" RMS {train:.2}/{holdout:.2}PX"),
            _ => String::new(),
        };
        format!(
            "CHECKER {} VIEWS {}/{}{}  {}",
            self.state.to_ascii_uppercase(),
            self.accepted_views,
            MIN_VIEWS,
            rms,
            self.detail,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitOutcome {
    Accepted,
    DroppedBusy,
    Inactive,
    Invalid,
}

pub struct Client {
    request: Option<SyncSender<Arc<RawFrame>>>,
    status: Arc<Mutex<StatusSnapshot>>,
    active: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
    eye_index: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Client {
    pub fn start(output: impl AsRef<Path>) -> Result<Self, String> {
        let status = Arc::new(Mutex::new(StatusSnapshot::default()));
        let active = Arc::new(AtomicBool::new(false));
        let generation = Arc::new(AtomicU64::new(0));
        let eye_index = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (request_tx, request_rx) = sync_channel(1);

        #[cfg(feature = "checkerboard")]
        let worker = Some(start_worker(
            output.as_ref().to_path_buf(),
            request_rx,
            Arc::clone(&status),
            Arc::clone(&active),
            Arc::clone(&generation),
            Arc::clone(&eye_index),
            Arc::clone(&stop),
        )?);

        #[cfg(not(feature = "checkerboard"))]
        let worker = {
            drop(request_rx);
            if let Ok(mut snapshot) = status.lock() {
                snapshot.state = "unavailable".to_string();
                snapshot.detail = "viewer was built without the checkerboard feature".to_string();
            }
            None
        };

        Ok(Self {
            request: Some(request_tx),
            status,
            active,
            generation,
            eye_index,
            stop,
            worker,
        })
    }

    pub fn configure(&self, active: bool, generation: u64, eye_index: usize) {
        self.eye_index.store(eye_index.min(1), Ordering::Release);
        self.generation.store(generation, Ordering::Release);
        self.active.store(active, Ordering::Release);
        if let Ok(mut status) = self.status.lock() {
            if status.generation != generation {
                *status = StatusSnapshot {
                    active,
                    generation,
                    state: if active { "starting" } else { "off" }.to_string(),
                    detail: if active {
                        "waiting for a sharp green/white central lattice".to_string()
                    } else {
                        StatusSnapshot::default().detail
                    },
                    ..StatusSnapshot::default()
                };
            } else if status.state != "unavailable" {
                status.active = active;
                if !active && status.state != "calibrated" && status.state != "off" {
                    status.state = "paused".to_string();
                    status.detail = "press C to resume as a fresh bounded session".to_string();
                }
            }
        }
    }

    pub fn submit(&self, frame: Arc<RawFrame>) -> SubmitOutcome {
        if !self.active.load(Ordering::Acquire) {
            return SubmitOutcome::Inactive;
        }
        if frame.eye_index != self.eye_index.load(Ordering::Acquire)
            || frame.width < 64
            || frame.height < 64
            || frame.width & 1 != 0
            || frame.height & 1 != 0
            || frame.pixels.len() != frame.width.saturating_mul(frame.height)
            || frame.sensor_extent.0 == 0
            || frame.sensor_extent.1 == 0
            || (frame.layout == SampleLayout::QuadBayerRaw10
                && frame.sensor_extent != (frame.width as u32, frame.height as u32))
        {
            return SubmitOutcome::Invalid;
        }
        let Some(request) = self.request.as_ref() else {
            return SubmitOutcome::Inactive;
        };
        match request.try_send(frame) {
            Ok(()) => SubmitOutcome::Accepted,
            Err(TrySendError::Full(_)) => {
                if let Ok(mut status) = self.status.lock() {
                    status.dropped_frames = status.dropped_frames.saturating_add(1);
                }
                SubmitOutcome::DroppedBusy
            }
            Err(TrySendError::Disconnected(_)) => SubmitOutcome::Invalid,
        }
    }

    pub fn status(&self) -> StatusSnapshot {
        self.status
            .lock()
            .map(|value| value.clone())
            .unwrap_or_default()
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.request.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Load the custom camera service's lossless full-aperture linear snapshot.
/// The service writes the payload atomically before publishing the metadata,
/// so a successful load is one coherent sensor observation.
#[cfg(feature = "checkerboard")]
pub fn load_linear_snapshot(
    metadata_path: &Path,
    eye_index: usize,
    focus_target: u16,
    focus_position: u16,
    focus_settled: bool,
    focus_generation: u32,
) -> Result<RawFrame, String> {
    let text = fs::read_to_string(metadata_path).map_err(|error| {
        format!(
            "read full-sensor linear metadata {}: {error}",
            metadata_path.display()
        )
    })?;
    let document: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("parse full-sensor linear metadata: {error}"))?;
    if document.get("schema").and_then(serde_json::Value::as_str)
        != Some("buttercup-linear-sensor-thumbnail-v1")
        || document
            .get("sample_format")
            .and_then(serde_json::Value::as_str)
            != Some("gray16le-linear-raw-average")
        || document
            .get("sample_max")
            .and_then(serde_json::Value::as_u64)
            != Some(1023)
    {
        return Err("linear snapshot metadata has an unsupported sensor contract".to_string());
    }
    let number = |name: &str| {
        document
            .get(name)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("linear snapshot metadata field {name} is missing"))
    };
    let sensor_x = u32::try_from(number("sensor_x")?)
        .map_err(|_| "linear snapshot sensor_x exceeds u32".to_string())?;
    let sensor_y = u32::try_from(number("sensor_y")?)
        .map_err(|_| "linear snapshot sensor_y exceeds u32".to_string())?;
    let width = usize::try_from(number("width")?)
        .map_err(|_| "linear snapshot width exceeds usize".to_string())?;
    let height = usize::try_from(number("height")?)
        .map_err(|_| "linear snapshot height exceeds usize".to_string())?;
    let sequence = number("sequence")?;
    if sensor_x != 0
        || sensor_y != 0
        || width < 128
        || height < 96
        || width.saturating_mul(3) != height.saturating_mul(4)
    {
        return Err(format!(
            "linear snapshot must represent the complete 4:3 aperture, got origin={sensor_x},{sensor_y} samples={width}x{height}"
        ));
    }
    let payload_value = document
        .get("payload")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "linear snapshot payload path is missing".to_string())?;
    let payload_path = {
        let candidate = PathBuf::from(payload_value);
        if candidate.is_absolute() {
            candidate
        } else {
            metadata_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(candidate)
        }
    };
    let bytes = fs::read(&payload_path)
        .map_err(|error| format!("read linear snapshot {}: {error}", payload_path.display()))?;
    let sample_count = width
        .checked_mul(height)
        .ok_or_else(|| "linear snapshot dimensions overflow".to_string())?;
    if bytes.len() != sample_count.saturating_mul(2) {
        return Err(format!(
            "linear snapshot payload has {} bytes, expected {}",
            bytes.len(),
            sample_count * 2
        ));
    }
    let mut pixels = Vec::with_capacity(sample_count);
    for sample in bytes.chunks_exact(2) {
        let value = u16::from_le_bytes([sample[0], sample[1]]);
        if value > 1023 {
            return Err(format!(
                "linear snapshot contains out-of-range sample {value}"
            ));
        }
        pixels.push(value);
    }
    let timestamp_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u64::MAX as u128) as u64;
    Ok(RawFrame {
        eye_index: eye_index.min(1),
        sequence,
        timestamp_ns,
        sensor_x,
        sensor_y,
        width,
        height,
        layout: SampleLayout::LinearGray16,
        sensor_extent: (8_000, 6_000),
        focus_target,
        focus_position,
        focus_settled,
        focus_generation,
        pixels: Arc::new(pixels),
    })
}

#[cfg(not(feature = "checkerboard"))]
pub fn load_linear_snapshot(
    _metadata_path: &Path,
    _eye_index: usize,
    _focus_target: u16,
    _focus_position: u16,
    _focus_settled: bool,
    _focus_generation: u32,
) -> Result<RawFrame, String> {
    Err("viewer was built without the checkerboard feature".to_string())
}

#[cfg(feature = "checkerboard")]
#[derive(Clone)]
struct CalibrationView {
    sequence: u64,
    timestamp_ns: u64,
    sensor_origin: (u32, u32),
    sample_size: (usize, usize),
    sensor_scale: (f64, f64),
    layout: SampleLayout,
    preview: Arc<Vec<u32>>,
    image_points: Vec<[f64; 2]>,
    local_points: Vec<(f64, f64)>,
    centroid: [f64; 2],
    scale: f64,
    orientation_degrees: f64,
    planar_rms_px: f64,
    sharpness_px: f64,
    contrast: f64,
}

#[cfg(feature = "checkerboard")]
#[derive(Clone, Debug)]
struct CalibrationFit {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    k1: f64,
    k2: f64,
    training_rms_px: f64,
    holdout_rms_px: f64,
    training_views: usize,
    holdout_views: usize,
}

#[cfg(feature = "checkerboard")]
struct WorkerSession {
    generation: u64,
    output: PathBuf,
    sensor_size: (u32, u32),
    focus_position: Option<u16>,
    views: Vec<CalibrationView>,
    frames_seen: u64,
    detections: u64,
    duplicate_views: u64,
    rejected_views: u64,
    last_processed: Option<Instant>,
    fit: Option<CalibrationFit>,
}

#[cfg(feature = "checkerboard")]
impl WorkerSession {
    fn new(base: &Path, generation: u64) -> Result<Self, String> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let output = base.join(format!(
            "session-{}-{:09}-g{generation}",
            stamp.as_secs(),
            stamp.subsec_nanos(),
        ));
        fs::create_dir_all(output.join("accepted-raw10")).map_err(|error| {
            format!("create checkerboard session {}: {error}", output.display())
        })?;
        Ok(Self {
            generation,
            output,
            sensor_size: (8_000, 6_000),
            focus_position: None,
            views: Vec::new(),
            frames_seen: 0,
            detections: 0,
            duplicate_views: 0,
            rejected_views: 0,
            last_processed: None,
            fit: None,
        })
    }

    fn process(&mut self, frame: &RawFrame) -> Result<StatusSnapshot, String> {
        self.frames_seen = self.frames_seen.saturating_add(1);
        if self
            .last_processed
            .is_some_and(|last| last.elapsed() < Duration::from_millis(120))
        {
            return Ok(self.snapshot(
                "collecting",
                "holding the live detector below the frame-time budget",
                None,
            ));
        }
        self.last_processed = Some(Instant::now());

        if frame.layout == SampleLayout::LinearGray16 {
            persist_latest_linear_frame(&self.output, frame)?;
        }

        if !frame.focus_settled
            || frame.focus_target == u16::MAX
            || frame.focus_position == u16::MAX
            || frame.focus_target != frame.focus_position
        {
            self.rejected_views = self.rejected_views.saturating_add(1);
            return Ok(self.snapshot("waiting", "focus must be settled and unchanged", None));
        }
        if let Some(focus) = self.focus_position {
            if focus != frame.focus_position {
                self.rejected_views = self.rejected_views.saturating_add(1);
                return Ok(self.snapshot(
                    "waiting",
                    "focus changed; press C twice to start a new calibration",
                    None,
                ));
            }
        }

        let detection = match detect_green_white_checkerboard(frame) {
            Ok(Some(detection)) => detection,
            Ok(None) => {
                let status = self.snapshot(
                    "searching",
                    "show seven straight central intersections; outer rounded cells are ignored",
                    preview_overlay(frame).ok(),
                );
                self.persist_status(&status)?;
                return Ok(status);
            }
            Err(error) => {
                self.rejected_views = self.rejected_views.saturating_add(1);
                let status = self.snapshot("rejected", &error, preview_overlay(frame).ok());
                self.persist_status(&status)?;
                return Ok(status);
            }
        };
        self.detections = self.detections.saturating_add(1);
        let mut overlay = Overlay {
            eye_index: frame.eye_index,
            sequence: frame.sequence,
            timestamp_ns: frame.timestamp_ns,
            sensor_origin: (frame.sensor_x, frame.sensor_y),
            sample_size: detection.sample_size,
            sensor_scale: detection.sensor_scale,
            full_sensor: detection.layout == SampleLayout::LinearGray16,
            preview: Some(Arc::clone(&detection.preview)),
            points: detection.local_points.clone(),
            accepted_view: false,
            ..Overlay::default()
        };

        let maximum_planar_rms = if detection.layout == SampleLayout::LinearGray16 {
            4.0
        } else {
            1.50
        };
        if detection.planar_rms_px > maximum_planar_rms {
            self.rejected_views = self.rejected_views.saturating_add(1);
            return Ok(self.snapshot(
                "rejected",
                &format!(
                    "central grid is not planar enough ({:.2}px RMS); flatten it or exclude more edge cells",
                    detection.planar_rms_px
                ),
                Some(overlay),
            ));
        }
        if detection.contrast < 9.0 {
            self.rejected_views = self.rejected_views.saturating_add(1);
            return Ok(self.snapshot(
                "rejected",
                &format!(
                    "green/white chroma contrast is too low ({:.1})",
                    detection.contrast
                ),
                Some(overlay),
            ));
        }
        let maximum_sharpness = if detection.layout == SampleLayout::LinearGray16 {
            75.0
        } else {
            6.0
        };
        if detection.sharpness_px > maximum_sharpness {
            self.rejected_views = self.rejected_views.saturating_add(1);
            return Ok(self.snapshot(
                "rejected",
                &format!(
                    "checker transition is too soft ({:.1}px)",
                    detection.sharpness_px
                ),
                Some(overlay),
            ));
        }
        if !self.pose_is_new(&detection) {
            self.duplicate_views = self.duplicate_views.saturating_add(1);
            return Ok(self.snapshot(
                "duplicate",
                "move or tilt the board farther before holding it still",
                Some(overlay),
            ));
        }

        self.focus_position = Some(frame.focus_position);
        overlay.accepted_view = true;
        persist_accepted_raw(&self.output, self.views.len(), frame, &detection)?;
        self.views.push(detection);

        // Animate an actual 3D board pose from the first accepted view.  Until
        // this session has solved its own camera, use only the guarded nominal
        // focal model for the pose axes and deliberately hide its reprojection
        // residuals.  Once fitted, both axes and magenta fitted points use the
        // trained camera model.
        let pose_fit = self.fit.clone().unwrap_or_else(nominal_pose_camera);
        if let Ok(mut posed) = project_fit_overlay(
            overlay.clone(),
            &self.views[self.views.len() - 1],
            &pose_fit,
        ) {
            if self.fit.is_none() {
                posed.reprojected.clear();
            }
            overlay = posed;
        }

        let readiness = self.readiness_reasons();
        if self.views.len() >= MIN_VIEWS {
            match fit_intrinsics(&self.views, self.sensor_size) {
                Ok(fit) => {
                    overlay = project_fit_overlay(
                        overlay.clone(),
                        &self.views[self.views.len() - 1],
                        &fit,
                    )
                    .unwrap_or_else(|_| overlay.clone());
                    self.fit = Some(fit.clone());
                    persist_fit(
                        &self.output,
                        self.focus_position.unwrap_or(u16::MAX),
                        &fit,
                        &readiness,
                    )?;
                    let validated = readiness.is_empty()
                        && fit.training_rms_px <= MAX_TRAINING_RMS_PX
                        && fit.holdout_rms_px <= MAX_HOLDOUT_RMS_PX
                        && plausible_fit(&fit, self.sensor_size);
                    if validated {
                        persist_compatible_artifact(
                            &self.output,
                            self.focus_position.unwrap_or(u16::MAX),
                            &fit,
                        )?;
                        let status = self.snapshot(
                            "calibrated",
                            "validated artifact written; C pauses without overwriting the installed camera model",
                            Some(overlay),
                        );
                        self.persist_status(&status)?;
                        return Ok(status);
                    }
                }
                Err(error) => {
                    let status = self.snapshot(
                        "fit-rejected",
                        &format!("provisional intrinsic fit failed: {error}"),
                        Some(overlay),
                    );
                    self.persist_status(&status)?;
                    return Ok(status);
                }
            }
        }

        let detail = if let Some(reason) = readiness.first() {
            reason.clone()
        } else if let Some(fit) = self.fit.as_ref() {
            format!(
                "fit needs lower residual (train {:.2}px, held {:.2}px)",
                fit.training_rms_px, fit.holdout_rms_px
            )
        } else {
            "accepted; move the board toward a different sensor edge and tilt".to_string()
        };
        let status = self.snapshot("collecting", &detail, Some(overlay));
        self.persist_status(&status)?;
        Ok(status)
    }

    fn pose_is_new(&self, candidate: &CalibrationView) -> bool {
        self.views.iter().all(|prior| {
            let centroid_delta = ((candidate.centroid[0] - prior.centroid[0])
                .hypot(candidate.centroid[1] - prior.centroid[1]))
                / f64::from(self.sensor_size.0.max(self.sensor_size.1));
            let scale_delta = (candidate.scale / prior.scale.max(1.0)).ln().abs();
            let angle_delta =
                angle_delta_degrees(candidate.orientation_degrees, prior.orientation_degrees);
            centroid_delta >= 0.035 || scale_delta >= 0.08 || angle_delta >= 4.0
        })
    }

    fn readiness_reasons(&self) -> Vec<String> {
        let mut reasons = Vec::new();
        if self.views.len() < MIN_VIEWS {
            reasons.push(format!(
                "need {} more distinct view(s)",
                MIN_VIEWS - self.views.len()
            ));
            return reasons;
        }
        let all_points = self
            .views
            .iter()
            .flat_map(|view| view.image_points.iter().copied())
            .collect::<Vec<_>>();
        let span = point_span(&all_points);
        let centroids = self
            .views
            .iter()
            .map(|view| view.centroid)
            .collect::<Vec<_>>();
        let centroid_span = point_span(&centroids);
        let min_scale = self
            .views
            .iter()
            .map(|view| view.scale)
            .fold(f64::INFINITY, f64::min);
        let max_scale = self.views.iter().map(|view| view.scale).fold(0.0, f64::max);
        let orientation_span = self
            .views
            .iter()
            .enumerate()
            .flat_map(|(index, left)| {
                self.views[index + 1..].iter().map(move |right| {
                    angle_delta_degrees(left.orientation_degrees, right.orientation_degrees)
                })
            })
            .fold(0.0, f64::max);
        if span[0] < f64::from(self.sensor_size.0) * 0.35 {
            reasons.push("move the checker farther left and right across the sensor".to_string());
        }
        if span[1] < f64::from(self.sensor_size.1) * 0.30 {
            reasons.push("move the checker farther up and down across the sensor".to_string());
        }
        if centroid_span[0] < f64::from(self.sensor_size.0) * 0.18
            || centroid_span[1] < f64::from(self.sensor_size.1) * 0.14
        {
            reasons.push("spread whole-board centers across more of the aperture".to_string());
        }
        if min_scale <= 0.0 || max_scale / min_scale < 1.40 {
            reasons.push("include both a nearer and a farther board size".to_string());
        }
        if orientation_span < 20.0 {
            reasons.push("add at least 20 degrees of board roll/tilt diversity".to_string());
        }
        reasons
    }

    fn snapshot(&self, state: &str, detail: &str, overlay: Option<Overlay>) -> StatusSnapshot {
        StatusSnapshot {
            active: true,
            generation: self.generation,
            state: state.to_string(),
            detail: detail.to_string(),
            frames_seen: self.frames_seen,
            detections: self.detections,
            accepted_views: self.views.len(),
            duplicate_views: self.duplicate_views,
            rejected_views: self.rejected_views,
            dropped_frames: 0,
            focus_position: self.focus_position,
            training_rms_px: self.fit.as_ref().map(|fit| fit.training_rms_px),
            holdout_rms_px: self.fit.as_ref().map(|fit| fit.holdout_rms_px),
            calibrated: state == "calibrated",
            output: Some(self.output.clone()),
            overlay,
        }
    }

    fn persist_status(&self, status: &StatusSnapshot) -> Result<(), String> {
        let document = serde_json::json!({
            "schema": "buttercup-green-white-checkerboard-session-v1",
            "state": status.state,
            "detail": status.detail,
            "generation": status.generation,
            "active": status.active,
            "frames_seen": status.frames_seen,
            "detections": status.detections,
            "accepted_views": status.accepted_views,
            "duplicate_views": status.duplicate_views,
            "rejected_views": status.rejected_views,
            "focus_position": status.focus_position,
            "target": {
                "colors": ["green", "white"],
                "square_pitch_mm": SQUARE_MM,
                "central_intersections": [INNER_CORNERS, INNER_CORNERS],
                "central_span_mm": CENTRAL_SPAN_MM,
                "outer_squares": "excluded",
            },
            "training_rms_px": status.training_rms_px,
            "holdout_rms_px": status.holdout_rms_px,
            "calibrated": status.calibrated,
            "output": status.output.as_ref().map(|path| path.to_string_lossy().to_string()),
        });
        write_atomic_json(&self.output.join("status.json"), &document)
    }
}

#[cfg(feature = "checkerboard")]
fn start_worker(
    output: PathBuf,
    request: std::sync::mpsc::Receiver<Arc<RawFrame>>,
    status: Arc<Mutex<StatusSnapshot>>,
    active: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
    eye_index: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
) -> Result<thread::JoinHandle<()>, String> {
    thread::Builder::new()
        .name("green-white-checkerboard-calibration".to_string())
        .spawn(move || {
            let _ = opencv::core::set_num_threads(1);
            let mut session: Option<WorkerSession> = None;
            while !stop.load(Ordering::Acquire) {
                let frame = match request.recv_timeout(Duration::from_millis(100)) {
                    Ok(frame) => frame,
                    Err(RecvTimeoutError::Timeout) => continue,
                    Err(RecvTimeoutError::Disconnected) => break,
                };
                if !active.load(Ordering::Acquire)
                    || frame.eye_index != eye_index.load(Ordering::Acquire)
                {
                    continue;
                }
                let requested_generation = generation.load(Ordering::Acquire);
                if session
                    .as_ref()
                    .is_none_or(|current| current.generation != requested_generation)
                {
                    match WorkerSession::new(&output, requested_generation) {
                        Ok(next) => {
                            eprintln!(
                                "checkerboard calibration generation={} output={} square_pitch_mm={SQUARE_MM:.6} central_span_mm={CENTRAL_SPAN_MM:.3}",
                                requested_generation,
                                next.output.display(),
                            );
                            session = Some(next);
                        }
                        Err(error) => {
                            if let Ok(mut snapshot) = status.lock() {
                                snapshot.state = "error".to_string();
                                snapshot.detail = error;
                            }
                            continue;
                        }
                    }
                }
                let Some(current) = session.as_mut() else {
                    continue;
                };
                let dropped = status
                    .lock()
                    .map(|snapshot| snapshot.dropped_frames)
                    .unwrap_or(0);
                let next = match current.process(&frame) {
                    Ok(mut next) => {
                        next.dropped_frames = dropped;
                        next
                    }
                    Err(error) => {
                        let mut next = current.snapshot("error", &error, None);
                        next.dropped_frames = dropped;
                        next
                    }
                };
                if let Ok(mut snapshot) = status.lock() {
                    *snapshot = next;
                }
            }
        })
        .map_err(|error| format!("spawn checkerboard calibration worker: {error}"))
}

#[cfg(feature = "checkerboard")]
fn angle_delta_degrees(left: f64, right: f64) -> f64 {
    ((left - right + 90.0).rem_euclid(180.0) - 90.0).abs()
}

#[cfg(feature = "checkerboard")]
fn point_span(points: &[[f64; 2]]) -> [f64; 2] {
    if points.is_empty() {
        return [0.0, 0.0];
    }
    let mut low = [f64::INFINITY; 2];
    let mut high = [f64::NEG_INFINITY; 2];
    for point in points {
        for axis in 0..2 {
            low[axis] = low[axis].min(point[axis]);
            high[axis] = high[axis].max(point[axis]);
        }
    }
    [high[0] - low[0], high[1] - low[1]]
}

#[cfg(feature = "checkerboard")]
fn write_atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("checkerboard output {} has no parent", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("create checkerboard output {}: {error}", parent.display()))?;
    let temporary = PathBuf::from(format!("{}.new", path.display()));
    fs::write(&temporary, bytes)
        .and_then(|()| fs::rename(&temporary, path))
        .map_err(|error| format!("write checkerboard output {}: {error}", path.display()))
}

#[cfg(feature = "checkerboard")]
fn write_atomic_json(path: &Path, document: &serde_json::Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(document)
        .map_err(|error| format!("encode checkerboard JSON: {error}"))?;
    bytes.push(b'\n');
    write_atomic_bytes(path, &bytes)
}

#[cfg(feature = "checkerboard")]
fn pack_raw10(values: &[u16], width: usize, height: usize) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 || width % 4 != 0 || values.len() != width * height {
        return Err("checkerboard RAW10 export geometry is invalid".to_string());
    }
    let mut output = Vec::with_capacity(width / 4 * 5 * height);
    for row in values.chunks_exact(width) {
        for group in row.chunks_exact(4) {
            let word = u64::from(group[0].min(1023))
                | (u64::from(group[1].min(1023)) << 10)
                | (u64::from(group[2].min(1023)) << 20)
                | (u64::from(group[3].min(1023)) << 30);
            output.extend_from_slice(&word.to_le_bytes()[..5]);
        }
    }
    Ok(output)
}

#[cfg(feature = "checkerboard")]
fn persist_accepted_raw(
    output: &Path,
    view_index: usize,
    frame: &RawFrame,
    view: &CalibrationView,
) -> Result<(), String> {
    let stem = format!("view-{view_index:03}-seq-{:010}", frame.sequence);
    let (payload_name, sample_format, stride) = match frame.layout {
        SampleLayout::QuadBayerRaw10 => {
            let name = format!("{stem}.raw10");
            write_atomic_bytes(
                &output.join("accepted-raw10").join(&name),
                &pack_raw10(&frame.pixels, frame.width, frame.height)?,
            )?;
            (name, "RAW10_LE40_1X1", frame.width / 4 * 5)
        }
        SampleLayout::LinearGray16 => {
            let name = format!("{stem}.gray16le");
            let bytes = gray16le_bytes(&frame.pixels);
            write_atomic_bytes(&output.join("accepted-raw10").join(&name), &bytes)?;
            persist_linear_excerpt(output, &stem, frame, view)?;
            (name, "gray16le-linear-raw-average", frame.width * 2)
        }
    };
    let sidecar = serde_json::json!({
        "schema": "buttercup-checkerboard-raw10-evidence-v1",
        "sequence": frame.sequence,
        "timestamp_ns": frame.timestamp_ns,
        "eye_index": frame.eye_index,
        "sensor_x": frame.sensor_x,
        "sensor_y": frame.sensor_y,
        "width": frame.width,
        "height": frame.height,
        "sensor_extent": [frame.sensor_extent.0, frame.sensor_extent.1],
        "sensor_scale": [view.sensor_scale.0, view.sensor_scale.1],
        "stride": stride,
        "pixel_format": sample_format,
        "payload": payload_name,
        "focus": {
            "target": frame.focus_target,
            "readback": frame.focus_position,
            "settled": frame.focus_settled,
            "generation": frame.focus_generation,
        },
        "target": {
            "colors": ["green", "white"],
            "square_pitch_mm": SQUARE_MM,
            "central_intersections": [INNER_CORNERS, INNER_CORNERS],
            "central_span_mm": CENTRAL_SPAN_MM,
            "outer_squares": "excluded-before-fit",
        },
        "corners_sensor_px": view.image_points,
        "planar_rms_px": view.planar_rms_px,
        "transition_sharpness_px": view.sharpness_px,
        "green_white_contrast": view.contrast,
    });
    write_atomic_json(
        &output.join("accepted-raw10").join(format!("{stem}.json")),
        &sidecar,
    )
}

#[cfg(feature = "checkerboard")]
fn gray16le_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(feature = "checkerboard")]
fn persist_latest_linear_frame(output: &Path, frame: &RawFrame) -> Result<(), String> {
    if frame.layout != SampleLayout::LinearGray16 {
        return Ok(());
    }
    let payload_name = "latest-full-sensor.gray16le";
    write_atomic_bytes(&output.join(payload_name), &gray16le_bytes(&frame.pixels))?;
    let document = serde_json::json!({
        "schema": "buttercup-checkerboard-lossless-linear-debug-v1",
        "sequence": frame.sequence,
        "timestamp_ns": frame.timestamp_ns,
        "sensor_origin": [frame.sensor_x, frame.sensor_y],
        "sensor_extent": [frame.sensor_extent.0, frame.sensor_extent.1],
        "width": frame.width,
        "height": frame.height,
        "stride": frame.width * 2,
        "sample_format": "gray16le-linear-raw-average",
        "sample_max": 1023,
        "focus": {
            "target": frame.focus_target,
            "readback": frame.focus_position,
            "settled": frame.focus_settled,
            "generation": frame.focus_generation,
        },
        "payload": payload_name,
        "desktop_capture": false,
    });
    write_atomic_json(&output.join("latest-full-sensor.json"), &document)
}

/// Save a lossless sensor-derived excerpt around the accepted lattice.  This
/// keeps review artifacts compact without ever substituting a desktop capture
/// for the exact calibration evidence above.
#[cfg(feature = "checkerboard")]
fn persist_linear_excerpt(
    output: &Path,
    stem: &str,
    frame: &RawFrame,
    view: &CalibrationView,
) -> Result<(), String> {
    if frame.layout != SampleLayout::LinearGray16 || view.local_points.is_empty() {
        return Ok(());
    }
    let pitch = view
        .local_points
        .windows(2)
        .take(INNER_CORNERS - 1)
        .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
        .sum::<f64>()
        / (INNER_CORNERS - 1) as f64;
    let margin = pitch.max(2.0).ceil() as usize;
    let minimum_x = view
        .local_points
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0) as usize;
    let maximum_x = view
        .local_points
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .max(0.0) as usize;
    let minimum_y = view
        .local_points
        .iter()
        .map(|point| point.1)
        .fold(f64::INFINITY, f64::min)
        .floor()
        .max(0.0) as usize;
    let maximum_y = view
        .local_points
        .iter()
        .map(|point| point.1)
        .fold(f64::NEG_INFINITY, f64::max)
        .ceil()
        .max(0.0) as usize;
    let x0 = minimum_x.saturating_sub(margin).min(frame.width - 1);
    let y0 = minimum_y.saturating_sub(margin).min(frame.height - 1);
    let x1 = maximum_x
        .saturating_add(margin + 1)
        .clamp(x0 + 1, frame.width);
    let y1 = maximum_y
        .saturating_add(margin + 1)
        .clamp(y0 + 1, frame.height);
    let excerpt_width = x1 - x0;
    let excerpt_height = y1 - y0;
    let mut excerpt = Vec::with_capacity(excerpt_width * excerpt_height);
    for row in y0..y1 {
        excerpt.extend_from_slice(&frame.pixels[row * frame.width + x0..row * frame.width + x1]);
    }
    let payload_name = format!("{stem}-lattice-excerpt.gray16le");
    write_atomic_bytes(
        &output.join("accepted-raw10").join(&payload_name),
        &gray16le_bytes(&excerpt),
    )?;
    let sidecar = serde_json::json!({
        "schema": "buttercup-checkerboard-lossless-linear-excerpt-v1",
        "source_sequence": frame.sequence,
        "sample_origin": [x0, y0],
        "width": excerpt_width,
        "height": excerpt_height,
        "stride": excerpt_width * 2,
        "sample_format": "gray16le-linear-raw-average",
        "sensor_origin": [
            f64::from(frame.sensor_x) + x0 as f64 * view.sensor_scale.0,
            f64::from(frame.sensor_y) + y0 as f64 * view.sensor_scale.1,
        ],
        "sensor_scale": [view.sensor_scale.0, view.sensor_scale.1],
        "payload": payload_name,
    });
    write_atomic_json(
        &output
            .join("accepted-raw10")
            .join(format!("{stem}-lattice-excerpt.json")),
        &sidecar,
    )
}

#[cfg(feature = "checkerboard")]
fn plausible_fit(fit: &CalibrationFit, sensor_size: (u32, u32)) -> bool {
    let maximum = f64::from(sensor_size.0.max(sensor_size.1));
    fit.fx.is_finite()
        && fit.fy.is_finite()
        && (maximum * 0.2..=maximum * 10.0).contains(&fit.fx)
        && (maximum * 0.2..=maximum * 10.0).contains(&fit.fy)
        && (fit.fx - fit.fy).abs() <= fit.fx.max(fit.fy) * 0.005
        && (0.0..=f64::from(sensor_size.0)).contains(&fit.cx)
        && (0.0..=f64::from(sensor_size.1)).contains(&fit.cy)
        && fit.k1.abs() <= 2.0
        && fit.k2.abs() <= 2.0
}

#[cfg(feature = "checkerboard")]
fn nominal_pose_camera() -> CalibrationFit {
    // This is used only to make the live X/Y/Z board pose visible before the
    // session has enough diverse views to train intrinsics.  It is never
    // persisted or accepted as calibration evidence.
    CalibrationFit {
        fx: 4_737.6,
        fy: 4_737.6,
        cx: 4_000.0,
        cy: 3_000.0,
        k1: 0.0,
        k2: 0.0,
        training_rms_px: f64::INFINITY,
        holdout_rms_px: f64::INFINITY,
        training_views: 0,
        holdout_views: 0,
    }
}

#[cfg(feature = "checkerboard")]
fn persist_fit(
    output: &Path,
    focus_position: u16,
    fit: &CalibrationFit,
    readiness: &[String],
) -> Result<(), String> {
    let validated = readiness.is_empty()
        && fit.training_rms_px <= MAX_TRAINING_RMS_PX
        && fit.holdout_rms_px <= MAX_HOLDOUT_RMS_PX
        && plausible_fit(fit, (8_000, 6_000));
    let document = serde_json::json!({
        "schema": "buttercup-green-white-checkerboard-intrinsics-v1",
        "calibrated": validated,
        "auto_applied": false,
        "sensor_width": 8000,
        "sensor_height": 6000,
        "focus_position": focus_position,
        "fx_px": fit.fx,
        "fy_px": fit.fy,
        "cx_px": fit.cx,
        "cy_px": fit.cy,
        "k1": fit.k1,
        "k2": fit.k2,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
        "training_rms_px": fit.training_rms_px,
        "holdout_rms_px": fit.holdout_rms_px,
        "training_view_count": fit.training_views,
        "holdout_view_count": fit.holdout_views,
        "readiness_reasons": readiness,
        "target": {
            "colors": ["green", "white"],
            "square_pitch_mm": SQUARE_MM,
            "central_span_mm": CENTRAL_SPAN_MM,
            "outer_squares": "excluded",
        },
        "method": "Zhang/Bouguet planar calibration with whole-view holdout",
    });
    write_atomic_json(
        &output.join("camera-intrinsics-checkerboard.json"),
        &document,
    )
}

#[cfg(feature = "checkerboard")]
fn persist_compatible_artifact(
    output: &Path,
    focus_position: u16,
    fit: &CalibrationFit,
) -> Result<(), String> {
    let document = serde_json::json!({
        "schema": "buttercup-mediapipe-relative-camera-intrinsics-v1",
        "calibrated": true,
        "relative_only": true,
        "full_aperture_validated": true,
        "image_width": 8000,
        "image_height": 6000,
        "sensor_active": [0, 0, 8000, 6000],
        "focus_position": focus_position,
        "fx_over_width": fit.fx / 8000.0,
        "fy_over_height": fit.fy / 6000.0,
        "cx_over_width": fit.cx / 8000.0,
        "cy_over_height": fit.cy / 6000.0,
        "k1": fit.k1,
        "k2": fit.k2,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
        "training_view_count": fit.training_views,
        "holdout_view_count": fit.holdout_views,
        "holdout_rms_basis_px": fit.holdout_rms_px,
        "source": "native green/white checkerboard central 6x6-square calibration",
        "square_pitch_mm": SQUARE_MM,
        "auto_applied": false,
    });
    write_atomic_json(&output.join("camera-intrinsics-relative.json"), &document)
}

#[cfg(feature = "checkerboard")]
fn fit_intrinsics(
    views: &[CalibrationView],
    sensor_size: (u32, u32),
) -> Result<CalibrationFit, String> {
    use opencv::calib3d;
    use opencv::core::{
        Mat, Point2f, Point3f, Size, TermCriteria, TermCriteria_Type, Vector, CV_64F,
    };
    use opencv::prelude::*;

    if views.len() < MIN_VIEWS {
        return Err(format!("need at least {MIN_VIEWS} views"));
    }
    let holdout_indices = (0..views.len())
        .filter(|index| index % 5 == 4)
        .collect::<Vec<_>>();
    let training_indices = (0..views.len())
        .filter(|index| !holdout_indices.contains(index))
        .collect::<Vec<_>>();
    if training_indices.len() < 8 || holdout_indices.len() < 3 {
        return Err("need at least eight training and three held-out views".to_string());
    }

    let object = checker_object_points();
    let mut objects = Vector::<Vector<Point3f>>::new();
    let mut images = Vector::<Vector<Point2f>>::new();
    for &index in &training_indices {
        objects.push(object.clone());
        images.push(point2f_vector(&views[index].image_points));
    }

    let mut camera = Mat::eye(3, 3, CV_64F)
        .map_err(cv_error)?
        .to_mat()
        .map_err(cv_error)?;
    let initial_focal = f64::from(sensor_size.0) * 0.60;
    *camera.at_2d_mut::<f64>(0, 0).map_err(cv_error)? = initial_focal;
    *camera.at_2d_mut::<f64>(1, 1).map_err(cv_error)? = initial_focal;
    *camera.at_2d_mut::<f64>(0, 2).map_err(cv_error)? = f64::from(sensor_size.0) * 0.5;
    *camera.at_2d_mut::<f64>(1, 2).map_err(cv_error)? = f64::from(sensor_size.1) * 0.5;
    let mut distortion = Mat::zeros(5, 1, CV_64F)
        .map_err(cv_error)?
        .to_mat()
        .map_err(cv_error)?;
    let mut rvecs = Vector::<Mat>::new();
    let mut tvecs = Vector::<Mat>::new();
    let mut intrinsic_std = Mat::default();
    let mut extrinsic_std = Mat::default();
    let mut per_view_errors = Mat::default();
    let criteria = TermCriteria::new(
        TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
        100,
        1.0e-12,
    )
    .map_err(cv_error)?;
    let flags = calib3d::CALIB_USE_INTRINSIC_GUESS
        | calib3d::CALIB_FIX_ASPECT_RATIO
        | calib3d::CALIB_ZERO_TANGENT_DIST
        | calib3d::CALIB_FIX_K3;
    let training_rms = calib3d::calibrate_camera_extended(
        &objects,
        &images,
        Size::new(sensor_size.0 as i32, sensor_size.1 as i32),
        &mut camera,
        &mut distortion,
        &mut rvecs,
        &mut tvecs,
        &mut intrinsic_std,
        &mut extrinsic_std,
        &mut per_view_errors,
        flags,
        criteria,
    )
    .map_err(cv_error)?;

    let mut holdout_squared = Vec::new();
    for &index in &holdout_indices {
        let image = point2f_vector(&views[index].image_points);
        let mut rvec = Mat::default();
        let mut tvec = Mat::default();
        let mut inliers = Mat::default();
        let solved = calib3d::solve_pnp_ransac(
            &object,
            &image,
            &camera,
            &distortion,
            &mut rvec,
            &mut tvec,
            false,
            200,
            if views[index].layout == SampleLayout::LinearGray16 {
                5.0
            } else {
                2.0
            },
            0.999,
            &mut inliers,
            calib3d::SOLVEPNP_ITERATIVE,
        )
        .map_err(cv_error)?;
        if !solved || inliers.total() < INNER_CORNERS * INNER_CORNERS * 4 / 5 {
            return Err("held-out checkerboard pose solve lacked 80% inliers".to_string());
        }
        let mut projected = Vector::<Point2f>::new();
        calib3d::project_points_def(&object, &rvec, &tvec, &camera, &distortion, &mut projected)
            .map_err(cv_error)?;
        for (actual, predicted) in views[index].image_points.iter().zip(projected.to_vec()) {
            let dx = actual[0] - f64::from(predicted.x);
            let dy = actual[1] - f64::from(predicted.y);
            holdout_squared.push(dx * dx + dy * dy);
        }
    }
    let holdout_rms =
        (holdout_squared.iter().sum::<f64>() / holdout_squared.len().max(1) as f64).sqrt();
    let coefficients = distortion.data_typed::<f64>().map_err(cv_error)?;
    let fit = CalibrationFit {
        fx: *camera.at_2d::<f64>(0, 0).map_err(cv_error)?,
        fy: *camera.at_2d::<f64>(1, 1).map_err(cv_error)?,
        cx: *camera.at_2d::<f64>(0, 2).map_err(cv_error)?,
        cy: *camera.at_2d::<f64>(1, 2).map_err(cv_error)?,
        k1: coefficients.first().copied().unwrap_or(0.0),
        k2: coefficients.get(1).copied().unwrap_or(0.0),
        training_rms_px: training_rms,
        holdout_rms_px: holdout_rms,
        training_views: training_indices.len(),
        holdout_views: holdout_indices.len(),
    };
    if !plausible_fit(&fit, sensor_size) {
        return Err("intrinsic solution escaped the guarded camera range".to_string());
    }
    Ok(fit)
}

#[cfg(feature = "checkerboard")]
fn checker_object_points() -> opencv::core::Vector<opencv::core::Point3f> {
    use opencv::core::{Point3f, Vector};
    let mut points = Vector::<Point3f>::new();
    for row in 0..INNER_CORNERS {
        for column in 0..INNER_CORNERS {
            points.push(Point3f::new(
                (column as f64 * SQUARE_MM) as f32,
                (row as f64 * SQUARE_MM) as f32,
                0.0,
            ));
        }
    }
    points
}

#[cfg(feature = "checkerboard")]
fn point2f_vector(points: &[[f64; 2]]) -> opencv::core::Vector<opencv::core::Point2f> {
    use opencv::core::{Point2f, Vector};
    points
        .iter()
        .map(|point| Point2f::new(point[0] as f32, point[1] as f32))
        .collect::<Vector<_>>()
}

#[cfg(feature = "checkerboard")]
fn camera_mats(fit: &CalibrationFit) -> Result<(opencv::core::Mat, opencv::core::Mat), String> {
    use opencv::core::{Mat, CV_64F};
    use opencv::prelude::*;
    let mut camera = Mat::eye(3, 3, CV_64F)
        .map_err(cv_error)?
        .to_mat()
        .map_err(cv_error)?;
    *camera.at_2d_mut::<f64>(0, 0).map_err(cv_error)? = fit.fx;
    *camera.at_2d_mut::<f64>(1, 1).map_err(cv_error)? = fit.fy;
    *camera.at_2d_mut::<f64>(0, 2).map_err(cv_error)? = fit.cx;
    *camera.at_2d_mut::<f64>(1, 2).map_err(cv_error)? = fit.cy;
    let mut distortion = Mat::zeros(5, 1, CV_64F)
        .map_err(cv_error)?
        .to_mat()
        .map_err(cv_error)?;
    *distortion.at_2d_mut::<f64>(0, 0).map_err(cv_error)? = fit.k1;
    *distortion.at_2d_mut::<f64>(1, 0).map_err(cv_error)? = fit.k2;
    Ok((camera, distortion))
}

#[cfg(feature = "checkerboard")]
fn project_fit_overlay(
    mut overlay: Overlay,
    view: &CalibrationView,
    fit: &CalibrationFit,
) -> Result<Overlay, String> {
    use opencv::calib3d;
    use opencv::core::{Mat, Point2f, Point3f, Vector};
    use opencv::prelude::*;
    let object = checker_object_points();
    let image = point2f_vector(&view.image_points);
    let (camera, distortion) = camera_mats(fit)?;
    let mut rvec = Mat::default();
    let mut tvec = Mat::default();
    let mut inliers = Mat::default();
    if !calib3d::solve_pnp_ransac(
        &object,
        &image,
        &camera,
        &distortion,
        &mut rvec,
        &mut tvec,
        false,
        200,
        if view.layout == SampleLayout::LinearGray16 {
            5.0
        } else {
            2.0
        },
        0.999,
        &mut inliers,
        calib3d::SOLVEPNP_ITERATIVE,
    )
    .map_err(cv_error)?
    {
        return Err("project current checkerboard pose".to_string());
    }
    let mut projected = Vector::<Point2f>::new();
    calib3d::project_points_def(&object, &rvec, &tvec, &camera, &distortion, &mut projected)
        .map_err(cv_error)?;
    overlay.reprojected = projected
        .to_vec()
        .into_iter()
        .map(|point| {
            (
                (f64::from(point.x) - f64::from(view.sensor_origin.0)) / view.sensor_scale.0,
                (f64::from(point.y) - f64::from(view.sensor_origin.1)) / view.sensor_scale.1,
            )
        })
        .collect();

    let axis_length = (CENTRAL_SPAN_MM * 0.45) as f32;
    let axes = [
        Point3f::new(0.0, 0.0, 0.0),
        Point3f::new(axis_length, 0.0, 0.0),
        Point3f::new(0.0, axis_length, 0.0),
        Point3f::new(0.0, 0.0, -axis_length),
    ]
    .into_iter()
    .collect::<Vector<Point3f>>();
    let mut projected_axes = Vector::<Point2f>::new();
    calib3d::project_points_def(
        &axes,
        &rvec,
        &tvec,
        &camera,
        &distortion,
        &mut projected_axes,
    )
    .map_err(cv_error)?;
    overlay.pose_axes = projected_axes
        .to_vec()
        .into_iter()
        .map(|point| {
            (
                (f64::from(point.x) - f64::from(view.sensor_origin.0)) / view.sensor_scale.0,
                (f64::from(point.y) - f64::from(view.sensor_origin.1)) / view.sensor_scale.1,
            )
        })
        .collect();
    Ok(overlay)
}

#[cfg(feature = "checkerboard")]
fn cv_error(error: opencv::Error) -> String {
    format!("OpenCV checkerboard: {error}")
}

#[cfg(feature = "checkerboard")]
struct PlaneCandidate {
    points: Vec<opencv::core::Point2f>,
    planar_rms_half_px: f64,
    sharpness_half_px: f64,
    contrast: f64,
    source: &'static str,
}

#[cfg(feature = "checkerboard")]
fn preview_overlay(frame: &RawFrame) -> Result<Overlay, String> {
    let (values, plane_width, plane_height) = match frame.layout {
        SampleLayout::QuadBayerRaw10 => {
            let (_, luma, width, height) = green_white_planes(frame)?;
            (luma, width, height)
        }
        SampleLayout::LinearGray16 => (linear_gray_plane(frame)?, frame.width, frame.height),
    };
    let sensor_scale = (
        f64::from(frame.sensor_extent.0) / frame.width as f64,
        f64::from(frame.sensor_extent.1) / frame.height as f64,
    );
    Ok(Overlay {
        eye_index: frame.eye_index,
        sequence: frame.sequence,
        timestamp_ns: frame.timestamp_ns,
        sensor_origin: (frame.sensor_x, frame.sensor_y),
        sample_size: (frame.width, frame.height),
        sensor_scale,
        full_sensor: frame.layout == SampleLayout::LinearGray16,
        preview: Some(Arc::new(expand_preview(
            &values,
            plane_width,
            plane_height,
            frame.width,
            frame.height,
        ))),
        ..Overlay::default()
    })
}

#[cfg(feature = "checkerboard")]
fn detect_green_white_checkerboard(frame: &RawFrame) -> Result<Option<CalibrationView>, String> {
    let (
        first_plane,
        second_plane,
        plane_width,
        plane_height,
        point_scale,
        point_offset,
        sensor_scale,
    ) = match frame.layout {
        SampleLayout::QuadBayerRaw10 => {
            let (chroma, luma, width, height) = green_white_planes(frame)?;
            (
                (chroma, "green-chroma"),
                Some((luma, "linear-luma")),
                width,
                height,
                (2.0, 2.0),
                (0.5, 0.5),
                (1.0, 1.0),
            )
        }
        SampleLayout::LinearGray16 => {
            let plane = linear_gray_plane(frame)?;
            (
                (plane, "full-sensor-linear"),
                None,
                frame.width,
                frame.height,
                (1.0, 1.0),
                (0.0, 0.0),
                (
                    f64::from(frame.sensor_extent.0) / frame.width as f64,
                    f64::from(frame.sensor_extent.1) / frame.height as f64,
                ),
            )
        }
    };
    let point_scale: (f64, f64) = point_scale;
    let point_offset: (f64, f64) = point_offset;
    let sensor_scale: (f64, f64) = sensor_scale;
    let mut candidates = Vec::new();
    if let Some(candidate) = detect_plane(&first_plane.0, plane_width, plane_height, first_plane.1)?
    {
        candidates.push(candidate);
    }
    if let Some((values, source)) = second_plane.as_ref() {
        if let Some(candidate) = detect_plane(values, plane_width, plane_height, source)? {
            candidates.push(candidate);
        }
    }
    let Some(candidate) = candidates.into_iter().min_by(|left, right| {
        let left_score = left.planar_rms_half_px + left.sharpness_half_px * 0.08
            - left.contrast.min(80.0) * 0.002;
        let right_score = right.planar_rms_half_px + right.sharpness_half_px * 0.08
            - right.contrast.min(80.0) * 0.002;
        left_score.total_cmp(&right_score)
    }) else {
        return Ok(None);
    };
    eprintln!(
        "checkerboard detected source={} planar={:.3}px sharpness={:.3}px contrast={:.1}",
        candidate.source,
        candidate.planar_rms_half_px * point_scale.0.hypot(point_scale.1) / 2.0_f64.sqrt()
            * sensor_scale.0.hypot(sensor_scale.1)
            / 2.0_f64.sqrt(),
        candidate.sharpness_half_px * point_scale.0.hypot(point_scale.1) / 2.0_f64.sqrt()
            * sensor_scale.0.hypot(sensor_scale.1)
            / 2.0_f64.sqrt(),
        candidate.contrast,
    );
    let local_points = candidate
        .points
        .iter()
        .map(|point| {
            (
                f64::from(point.x) * point_scale.0 + point_offset.0,
                f64::from(point.y) * point_scale.1 + point_offset.1,
            )
        })
        .collect::<Vec<_>>();
    let image_points = local_points
        .iter()
        .map(|point| {
            [
                point.0 * sensor_scale.0 + f64::from(frame.sensor_x),
                point.1 * sensor_scale.1 + f64::from(frame.sensor_y),
            ]
        })
        .collect::<Vec<_>>();
    let centroid = image_points.iter().fold([0.0, 0.0], |mut sum, point| {
        sum[0] += point[0];
        sum[1] += point[1];
        sum
    });
    let centroid = [
        centroid[0] / image_points.len() as f64,
        centroid[1] / image_points.len() as f64,
    ];
    let corners = [
        image_points[0],
        image_points[INNER_CORNERS - 1],
        image_points[INNER_CORNERS * INNER_CORNERS - 1],
        image_points[INNER_CORNERS * (INNER_CORNERS - 1)],
    ];
    let area = polygon_area(&corners).abs();
    let first = image_points[0];
    let last = image_points[INNER_CORNERS - 1];
    let orientation = (last[1] - first[1])
        .atan2(last[0] - first[0])
        .to_degrees()
        .rem_euclid(180.0);
    let preview_values = if candidate.source == first_plane.1 {
        &first_plane.0
    } else {
        second_plane
            .as_ref()
            .map(|plane| &plane.0)
            .unwrap_or(&first_plane.0)
    };
    let preview = Arc::new(expand_preview(
        preview_values,
        plane_width,
        plane_height,
        frame.width,
        frame.height,
    ));
    let detector_to_sample = point_scale.0.hypot(point_scale.1) / 2.0_f64.sqrt();
    let sample_to_sensor = sensor_scale.0.hypot(sensor_scale.1) / 2.0_f64.sqrt();
    Ok(Some(CalibrationView {
        sequence: frame.sequence,
        timestamp_ns: frame.timestamp_ns,
        sensor_origin: (frame.sensor_x, frame.sensor_y),
        sample_size: (frame.width, frame.height),
        sensor_scale,
        layout: frame.layout,
        preview,
        image_points,
        local_points,
        centroid,
        scale: area.sqrt(),
        orientation_degrees: orientation,
        planar_rms_px: candidate.planar_rms_half_px * detector_to_sample * sample_to_sensor,
        sharpness_px: candidate.sharpness_half_px * detector_to_sample * sample_to_sensor,
        contrast: candidate.contrast,
    }))
}

#[cfg(feature = "checkerboard")]
fn linear_gray_plane(frame: &RawFrame) -> Result<Vec<u8>, String> {
    if frame.layout != SampleLayout::LinearGray16
        || frame.pixels.len() != frame.width.saturating_mul(frame.height)
    {
        return Err("full-sensor linear checkerboard plane is incomplete".to_string());
    }
    let values = frame
        .pixels
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    Ok(normalize_plane(&box_blur(
        &values,
        frame.width,
        frame.height,
    )))
}

#[cfg(feature = "checkerboard")]
fn expand_preview(
    values: &[u8],
    source_width: usize,
    source_height: usize,
    output_width: usize,
    output_height: usize,
) -> Vec<u32> {
    let mut output = vec![0u32; output_width.saturating_mul(output_height)];
    if values.len() != source_width.saturating_mul(source_height)
        || source_width == 0
        || source_height == 0
    {
        return output;
    }
    for y in 0..output_height {
        let source_y = y * source_height / output_height.max(1);
        for x in 0..output_width {
            let source_x = x * source_width / output_width.max(1);
            let value = u32::from(values[source_y * source_width + source_x]);
            output[y * output_width + x] = (value << 16) | (value << 8) | value;
        }
    }
    output
}

#[cfg(feature = "checkerboard")]
fn polygon_area(points: &[[f64; 2]; 4]) -> f64 {
    (0..points.len())
        .map(|index| {
            let next = (index + 1) % points.len();
            points[index][0] * points[next][1] - points[next][0] * points[index][1]
        })
        .sum::<f64>()
        * 0.5
}

#[cfg(feature = "checkerboard")]
fn green_white_planes(frame: &RawFrame) -> Result<(Vec<u8>, Vec<u8>, usize, usize), String> {
    if frame.width & 1 != 0
        || frame.height & 1 != 0
        || frame.pixels.len() != frame.width * frame.height
    {
        return Err("checkerboard requires an even, complete RAW10 ROI".to_string());
    }
    let width = frame.width / 2;
    let height = frame.height / 2;
    let mut quad = vec![0.0f64; width * height];
    for y in 0..height {
        for x in 0..width {
            let raw_x = x * 2;
            let raw_y = y * 2;
            quad[y * width + x] = (f64::from(frame.pixels[raw_y * frame.width + raw_x])
                + f64::from(frame.pixels[raw_y * frame.width + raw_x + 1])
                + f64::from(frame.pixels[(raw_y + 1) * frame.width + raw_x])
                + f64::from(frame.pixels[(raw_y + 1) * frame.width + raw_x + 1]))
                * 0.25;
        }
    }
    let at = |x: isize, y: isize| {
        let x = x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, height.saturating_sub(1) as isize) as usize;
        quad[y * width + x]
    };
    let mut chroma = vec![0.0; quad.len()];
    let mut luma = vec![0.0; quad.len()];
    for y in 0..height {
        for x in 0..width {
            let x_even = ((x as u32 + frame.sensor_x / 2) & 1) == 0;
            let y_even = ((y as u32 + frame.sensor_y / 2) & 1) == 0;
            let center = at(x as isize, y as isize);
            let horizontal =
                (at(x as isize - 1, y as isize) + at(x as isize + 1, y as isize)) * 0.5;
            let vertical = (at(x as isize, y as isize - 1) + at(x as isize, y as isize + 1)) * 0.5;
            let diagonal = (at(x as isize - 1, y as isize - 1)
                + at(x as isize + 1, y as isize - 1)
                + at(x as isize - 1, y as isize + 1)
                + at(x as isize + 1, y as isize + 1))
                * 0.25;
            let [red, green, blue] = match (y_even, x_even) {
                (true, true) => [center, (horizontal + vertical) * 0.5, diagonal],
                (true, false) => [horizontal, center, vertical],
                (false, true) => [vertical, center, horizontal],
                (false, false) => [diagonal, (horizontal + vertical) * 0.5, center],
            };
            let index = y * width + x;
            chroma[index] = (red + blue) * 0.5 / green.max(1.0);
            luma[index] = (red + green * 2.0 + blue) * 0.25;
        }
    }
    let chroma = normalize_plane(&box_blur(&chroma, width, height));
    let luma = normalize_plane(&box_blur(&luma, width, height));
    Ok((chroma, luma, width, height))
}

#[cfg(feature = "checkerboard")]
fn box_blur(values: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut output = vec![0.0; values.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0.0;
            for dy in -1..=1 {
                let yy = y.saturating_add_signed(dy).min(height - 1);
                for dx in -1..=1 {
                    let xx = x.saturating_add_signed(dx).min(width - 1);
                    sum += values[yy * width + xx];
                    count += 1.0;
                }
            }
            output[y * width + x] = sum / count;
        }
    }
    output
}

#[cfg(feature = "checkerboard")]
fn normalize_plane(values: &[f64]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let low = sorted[((sorted.len() - 1) as f64 * 0.02).round() as usize];
    let high = sorted[((sorted.len() - 1) as f64 * 0.98).round() as usize].max(low + 1.0e-9);
    values
        .iter()
        .map(|value| {
            (((value - low) * 255.0 / (high - low))
                .round()
                .clamp(0.0, 255.0)) as u8
        })
        .collect()
}

#[cfg(feature = "checkerboard")]
fn detect_plane(
    values: &[u8],
    width: usize,
    height: usize,
    source: &'static str,
) -> Result<Option<PlaneCandidate>, String> {
    use opencv::calib3d;
    use opencv::core::{Mat, Point2f, Scalar, Size, Vector, CV_8UC1};
    use opencv::prelude::*;

    let mut image =
        Mat::new_rows_cols_with_default(height as i32, width as i32, CV_8UC1, Scalar::all(0.0))
            .map_err(cv_error)?;
    image
        .data_bytes_mut()
        .map_err(cv_error)?
        .copy_from_slice(values);
    let pattern = Size::new(INNER_CORNERS as i32, INNER_CORNERS as i32);
    let flags = calib3d::CALIB_CB_NORMALIZE_IMAGE
        | calib3d::CALIB_CB_EXHAUSTIVE
        | calib3d::CALIB_CB_ACCURACY
        | calib3d::CALIB_CB_LARGER;
    let mut detected = Vector::<Point2f>::new();
    let mut meta = Mat::default();
    let found = calib3d::find_chessboard_corners_sb_with_meta(
        &image,
        pattern,
        &mut detected,
        flags,
        &mut meta,
    )
    .map_err(cv_error)?;
    if !found {
        return Ok(None);
    }
    let points = central_complete_window(&detected.to_vec(), &meta)?;
    if points.len() != INNER_CORNERS * INNER_CORNERS {
        return Ok(None);
    }
    let point_vector = points.iter().copied().collect::<Vector<Point2f>>();
    let sharpness = calib3d::estimate_chessboard_sharpness_def(&image, pattern, &point_vector)
        .map_err(cv_error)?;
    let contrast = (sharpness[2] - sharpness[1]).abs();
    let planar_rms = homography_rms(&points)?;
    Ok(Some(PlaneCandidate {
        points,
        planar_rms_half_px: planar_rms,
        sharpness_half_px: sharpness[0].abs(),
        contrast,
        source,
    }))
}

#[cfg(feature = "checkerboard")]
fn central_complete_window(
    corners: &[opencv::core::Point2f],
    meta: &opencv::core::Mat,
) -> Result<Vec<opencv::core::Point2f>, String> {
    use opencv::prelude::*;
    let rows = meta.rows().max(0) as usize;
    let columns = meta.cols().max(0) as usize;
    if rows < INNER_CORNERS
        || columns < INNER_CORNERS
        || corners.len() != rows.saturating_mul(columns)
    {
        return if corners.len() == INNER_CORNERS * INNER_CORNERS {
            Ok(corners.to_vec())
        } else {
            Err(format!(
                "checkerboard detector returned {} corners with {}x{} metadata",
                corners.len(),
                columns,
                rows
            ))
        };
    }
    let metadata = meta.data_bytes().map_err(cv_error)?;
    let center_row = (rows.saturating_sub(INNER_CORNERS)) / 2;
    let center_column = (columns.saturating_sub(INNER_CORNERS)) / 2;
    let mut windows = Vec::new();
    for start_row in 0..=rows - INNER_CORNERS {
        for start_column in 0..=columns - INNER_CORNERS {
            let complete = (0..INNER_CORNERS).all(|row| {
                (0..INNER_CORNERS).all(|column| {
                    metadata
                        .get((start_row + row) * columns + start_column + column)
                        .copied()
                        .unwrap_or(0)
                        != 0
                })
            });
            if complete {
                let distance =
                    start_row.abs_diff(center_row) + start_column.abs_diff(center_column);
                windows.push((distance, start_row, start_column));
            }
        }
    }
    let Some((_, start_row, start_column)) = windows.into_iter().min() else {
        return Err(
            "no complete central 7x7 checkerboard window; rounded outer cells remain excluded"
                .to_string(),
        );
    };
    Ok((0..INNER_CORNERS)
        .flat_map(|row| {
            (0..INNER_CORNERS)
                .map(move |column| corners[(start_row + row) * columns + start_column + column])
        })
        .collect())
}

#[cfg(feature = "checkerboard")]
fn homography_rms(points: &[opencv::core::Point2f]) -> Result<f64, String> {
    use opencv::calib3d;
    use opencv::core::{Mat, Point2f, Vector};
    use opencv::prelude::*;
    let object = (0..INNER_CORNERS)
        .flat_map(|row| {
            (0..INNER_CORNERS).map(move |column| Point2f::new(column as f32, row as f32))
        })
        .collect::<Vector<Point2f>>();
    let image = points.iter().copied().collect::<Vector<Point2f>>();
    let mut mask = Mat::default();
    let homography = calib3d::find_homography_def(&object, &image, &mut mask).map_err(cv_error)?;
    if homography.rows() != 3 || homography.cols() != 3 {
        return Err("central checkerboard homography is degenerate".to_string());
    }
    let h = |row, column| {
        homography
            .at_2d::<f64>(row, column)
            .copied()
            .map_err(cv_error)
    };
    let h00 = h(0, 0)?;
    let h01 = h(0, 1)?;
    let h02 = h(0, 2)?;
    let h10 = h(1, 0)?;
    let h11 = h(1, 1)?;
    let h12 = h(1, 2)?;
    let h20 = h(2, 0)?;
    let h21 = h(2, 1)?;
    let h22 = h(2, 2)?;
    let mut squared = 0.0;
    for row in 0..INNER_CORNERS {
        for column in 0..INNER_CORNERS {
            let x = column as f64;
            let y = row as f64;
            let denominator = h20 * x + h21 * y + h22;
            if denominator.abs() < 1.0e-12 {
                return Err("central checkerboard homography crosses infinity".to_string());
            }
            let predicted_x = (h00 * x + h01 * y + h02) / denominator;
            let predicted_y = (h10 * x + h11 * y + h12) / denominator;
            let actual = points[row * INNER_CORNERS + column];
            let dx = predicted_x - f64::from(actual.x);
            let dy = predicted_y - f64::from(actual.y);
            squared += dx * dx + dy * dy;
        }
    }
    Ok((squared / points.len() as f64).sqrt())
}
