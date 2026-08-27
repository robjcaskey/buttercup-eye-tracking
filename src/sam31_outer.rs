//! Fixed-prompt SAM3.1 iris segmentation for the live RAW10 viewer.
//!
//! One five-frame filmstrip passes through each of `quad_rgb`, `raw_luma`, and
//! `log_chroma`. The target parameter either publishes their outer-limbus
//! consensus directly or uses that consensus as the anatomical support region
//! for a RAW10 dark pupil-void fit. No Python, image files, or tracker session
//! exists in the live path.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
#[cfg(feature = "sam31")]
use std::thread;
#[cfg(feature = "sam31")]
use std::time::Instant;

pub const HISTORY_FRAMES: usize = 5;
pub const FRAME_WIDTH: usize = 384;
pub const FRAME_HEIGHT: usize = 256;
const FILMSTRIP_WIDTH: usize = FRAME_WIDTH * HISTORY_FRAMES;
const FILMSTRIP_PIXELS: usize = FILMSTRIP_WIDTH * FRAME_HEIGHT;
const MIN_COMPONENT_AREA_FULL_RES: usize = 20_000;
const MAX_COMPONENT_AREA_FULL_RES: usize = FRAME_WIDTH * FRAME_HEIGHT * 9 / 10;
const MAX_MASK_CANDIDATE_FITS: usize = 8;
const MIN_RAW_RING_SUPPORT_SCORE: f64 = 2.45;
const MIN_PUPIL_VOID_SUPPORT_SCORE: f64 = 2.05;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Target {
    #[default]
    OuterLimbus,
    InnerPupilVoid,
    /// Preserve the outer-limbus product even when the optional pupil-void
    /// post-process cannot find a valid component. This lets independent
    /// outer-iris and rough-center selectors share one expensive fixed-graph
    /// pass without either selector changing the other's availability.
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
    /// The live fixed-prompt SAM path intentionally leaves this `None`, so its
    /// pupil is derived from the independently inferred SAM outer ellipse.
    pub pupil_component_seed: Option<(f64, f64)>,
    pub pixels: Arc<Vec<u16>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ellipse {
    pub center: (f64, f64),
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
}

impl Ellipse {
    pub fn dense_points(self, count: usize) -> Vec<(f64, f64)> {
        let count = count.max(8);
        let (sin_angle, cos_angle) = self.angle.sin_cos();
        (0..count)
            .map(|index| {
                let phase = std::f64::consts::TAU * index as f64 / count as f64;
                let (sin_phase, cos_phase) = phase.sin_cos();
                let x = self.major_radius * cos_phase;
                let y = self.minor_radius * sin_phase;
                (
                    self.center.0 + cos_angle * x - sin_angle * y,
                    self.center.1 + sin_angle * x + cos_angle * y,
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct OuterResult {
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
    /// The fixed graph's independently validated outer-limbus guide. This is
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
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RawRingSupport {
    pub score: f64,
    pub points: usize,
    pub positive_fraction: f64,
    pub strong_sectors: usize,
}

/// Resolve target-specific products after the single expensive graph pass.
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
                "SAM31 inner pupil void had no dark component with consistent RAW meridian support"
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
            detail: "waiting for five RAW10 frames".to_string(),
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

struct Batch {
    target: Target,
    eye_index: usize,
    frames: Vec<Arc<RawFrame>>,
}

pub struct Client {
    request: Option<SyncSender<Batch>>,
    results: Receiver<OuterResult>,
    status: Arc<Mutex<StatusSnapshot>>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Client {
    pub fn start(model: impl AsRef<Path>) -> Result<Self, String> {
        let model = model.as_ref().to_path_buf();
        if !model.is_file() {
            return Err(format!(
                "SAM31 fixed-prompt model not found: {}",
                model.display()
            ));
        }
        // The live tracker currently submits only anatomical subject-right.
        // A rendezvous channel deliberately keeps no FIFO backlog: while the
        // model is busy, stale batches are dropped and the first current
        // five-frame history offered after completion wins.  A two-batch FIFO
        // at 42 Hz made every displayed fit roughly three inference periods
        // old even though each individual inference was healthy.
        let (request_tx, request_rx) = sync_channel(0);
        let (result_tx, result_rx) = sync_channel(4);
        let status = Arc::new(Mutex::new(StatusSnapshot::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker = start_worker(
            model,
            request_rx,
            result_tx,
            Arc::clone(&status),
            Arc::clone(&stop),
        )?;
        Ok(Self {
            request: Some(request_tx),
            results: result_rx,
            status,
            stop,
            worker: Some(worker),
        })
    }

    pub fn submit_history(
        &self,
        history: &VecDeque<Arc<RawFrame>>,
        target: Target,
    ) -> SubmitOutcome {
        if history.len() < HISTORY_FRAMES {
            return SubmitOutcome::Invalid;
        }
        let frames = history
            .iter()
            .rev()
            .take(HISTORY_FRAMES)
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
            && frames.len() == HISTORY_FRAMES
            && frames.iter().all(|frame| {
                frame.eye_index == eye_index
                    && frame.width == FRAME_WIDTH
                    && frame.height == FRAME_HEIGHT
                    && frame.pixels.len() == FRAME_WIDTH * FRAME_HEIGHT
            });
        if !valid {
            if let Ok(mut status) = self.status.lock() {
                status.state = "error";
                status.detail = format!(
                    "SAM31 requires five {}x{} frames from one eye",
                    FRAME_WIDTH, FRAME_HEIGHT
                );
            }
            return SubmitOutcome::Invalid;
        }
        let Some(request) = self.request.as_ref() else {
            return SubmitOutcome::Invalid;
        };
        match request.try_send(Batch {
            target,
            eye_index,
            frames,
        }) {
            Ok(()) => {
                if let Ok(mut status) = self.status.lock() {
                    status.accepted_batches = status.accepted_batches.saturating_add(1);
                    if status.state == "idle" {
                        status.state = "queued";
                        status.detail =
                            format!("first three-adapter {} query queued", target.label());
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
    _request: Receiver<Batch>,
    _results: SyncSender<OuterResult>,
    _status: Arc<Mutex<StatusSnapshot>>,
    _stop: Arc<AtomicBool>,
) -> Result<std::thread::JoinHandle<()>, String> {
    Err("SAM31 support is not compiled in; rebuild with --features sam31".to_string())
}

#[cfg(feature = "sam31")]
fn start_worker(
    model: PathBuf,
    request: Receiver<Batch>,
    results: SyncSender<OuterResult>,
    status: Arc<Mutex<StatusSnapshot>>,
    stop: Arc<AtomicBool>,
) -> Result<std::thread::JoinHandle<()>, String> {
    thread::Builder::new()
        .name("sam31-iris-segmenter".to_string())
        .spawn(move || runtime::worker(model, request, results, status, stop))
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
        .filter(|sector| !sector.is_empty() && median(sector.clone()) > 0.025)
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

fn ellipse_coordinate(point: (f64, f64), ellipse: Ellipse) -> f64 {
    let (angle_sine, angle_cosine) = ellipse.angle.sin_cos();
    let dx = point.0 - ellipse.center.0;
    let dy = point.1 - ellipse.center.1;
    let local_x = angle_cosine * dx + angle_sine * dy;
    let local_y = -angle_sine * dx + angle_cosine * dy;
    ((local_x / ellipse.major_radius.max(1.0)).powi(2)
        + (local_y / ellipse.minor_radius.max(1.0)).powi(2))
    .sqrt()
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
        && (0.09..=0.72).contains(&radius_ratio)
        && ellipse_coordinate(pupil.center, outer) <= 0.58
        && pupil
            .dense_points(32)
            .into_iter()
            .all(|point| ellipse_coordinate(point, outer) <= 0.92)
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

fn fit_inner_pupil_void(
    image: &FloatImage,
    outer: Ellipse,
    component_seed: Option<(f64, f64)>,
) -> Option<(Ellipse, RawRingSupport)> {
    if image.width != FRAME_WIDTH || image.height != FRAME_HEIGHT || !plausible_ellipse(outer) {
        return None;
    }
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
    let mut best: Option<(f64, Ellipse, RawRingSupport)> = None;
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
        let points = component
            .iter()
            .map(|&index| ((index % image.width) as f64, (index / image.width) as f64))
            .collect::<Vec<_>>();
        let Some(mut ellipse) = moments_ellipse(&points) else {
            continue;
        };
        normalize_ellipse(&mut ellipse);
        if !pupil_ellipse_plausible(ellipse, outer) {
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
            if !pupil_ellipse_plausible(candidate, outer) {
                continue;
            }
            let support = raw_ring_support(image, candidate);
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
        if support.score < MIN_PUPIL_VOID_SUPPORT_SCORE
            || support.positive_fraction < 0.42
            || support.strong_sectors < 3
        {
            continue;
        }
        if best.as_ref().is_none_or(|current| objective > current.0) {
            best = Some((objective, ellipse, support));
        }
    }
    best.map(|(_, ellipse, support)| (ellipse, support))
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
    if images.len() != HISTORY_FRAMES
        || images
            .iter()
            .any(|image| image.width != FRAME_WIDTH || image.height != FRAME_HEIGHT)
        || destination.len() != FILMSTRIP_PIXELS * 3
    {
        return Err("invalid SAM31 adapter filmstrip geometry".to_string());
    }
    let (low, high) = adapter_bounds(images, low_q, high_q);
    for (frame_index, image) in images.iter().enumerate() {
        for y in 0..FRAME_HEIGHT {
            for x in 0..FRAME_WIDTH {
                let pixel = image.data[y * FRAME_WIDTH + x];
                let film_x = frame_index * FRAME_WIDTH + x;
                let output_index = y * FILMSTRIP_WIDTH + film_x;
                for channel in 0..3 {
                    let normalized = ((pixel[channel] - low[channel])
                        / (high[channel] - low[channel]))
                        .clamp(0.0, 1.0);
                    let mapped = if (gamma - 1.0).abs() > f32::EPSILON {
                        normalized.powf(gamma)
                    } else {
                        normalized
                    };
                    destination[channel * FILMSTRIP_PIXELS + output_index] =
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
        if full_area < MIN_COMPONENT_AREA_FULL_RES as f64
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

#[derive(Clone, Copy)]
struct BoundaryEdge {
    start: (i32, i32),
    end: (i32, i32),
    owner: usize,
}

fn ordered_component_contour_fallback(
    component: &[usize],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Vec<(f64, f64)> {
    if component.len() < 5 {
        return Vec::new();
    }
    let mut membership = vec![false; mask_width * mask_height];
    for &index in component {
        membership[index] = true;
    }
    let mut sorted_component = component.to_vec();
    sorted_component.sort_unstable();
    let mut edges = Vec::<BoundaryEdge>::new();
    for index in sorted_component {
        let x = index % mask_width;
        let y = index / mask_width;
        let x0 = x as i32;
        let y0 = y as i32;
        if y == 0 || !membership[index - mask_width] {
            edges.push(BoundaryEdge {
                start: (x0, y0),
                end: (x0 + 1, y0),
                owner: index,
            });
        }
        if x + 1 == mask_width || !membership[index + 1] {
            edges.push(BoundaryEdge {
                start: (x0 + 1, y0),
                end: (x0 + 1, y0 + 1),
                owner: index,
            });
        }
        if y + 1 == mask_height || !membership[index + mask_width] {
            edges.push(BoundaryEdge {
                start: (x0 + 1, y0 + 1),
                end: (x0, y0 + 1),
                owner: index,
            });
        }
        if x == 0 || !membership[index - 1] {
            edges.push(BoundaryEdge {
                start: (x0, y0 + 1),
                end: (x0, y0),
                owner: index,
            });
        }
    }
    let mut outgoing = HashMap::<(i32, i32), Vec<usize>>::new();
    for (index, edge) in edges.iter().enumerate() {
        outgoing.entry(edge.start).or_default().push(index);
    }
    let mut used = vec![false; edges.len()];
    let mut best_vertices = Vec::<(i32, i32)>::new();
    let mut best_owners = Vec::<usize>::new();
    let mut best_area = 0.0f64;
    for first in 0..edges.len() {
        if used[first] {
            continue;
        }
        let start = edges[first].start;
        let mut current = first;
        let mut vertices = Vec::new();
        let mut owners = Vec::new();
        for _ in 0..=edges.len() {
            if used[current] {
                break;
            }
            let edge = edges[current];
            used[current] = true;
            vertices.push(edge.start);
            owners.push(edge.owner);
            if edge.end == start {
                break;
            }
            let Some(candidates) = outgoing.get(&edge.end) else {
                break;
            };
            let incoming = (edge.end.0 - edge.start.0, edge.end.1 - edge.start.1);
            let next = candidates
                .iter()
                .copied()
                .filter(|&candidate| !used[candidate])
                .max_by_key(|&candidate| {
                    let candidate = edges[candidate];
                    let direction = (
                        candidate.end.0 - candidate.start.0,
                        candidate.end.1 - candidate.start.1,
                    );
                    let cross = incoming.0 * direction.1 - incoming.1 * direction.0;
                    let dot = incoming.0 * direction.0 + incoming.1 * direction.1;
                    match (cross.signum(), dot.signum()) {
                        (1, _) => 3,
                        (0, 1) => 2,
                        (-1, _) => 1,
                        _ => 0,
                    }
                });
            let Some(next) = next else {
                break;
            };
            current = next;
        }
        if vertices.len() < 5 || edges[current].end != start {
            continue;
        }
        let twice_area = vertices
            .iter()
            .zip(vertices.iter().cycle().skip(1))
            .take(vertices.len())
            .map(|(first, second)| {
                first.0 as f64 * second.1 as f64 - second.0 as f64 * first.1 as f64
            })
            .sum::<f64>();
        let area = 0.5 * twice_area.abs();
        if area > best_area {
            best_area = area;
            best_vertices = vertices;
            best_owners = owners;
        }
    }
    let _ = best_vertices;
    let mut contour = Vec::with_capacity(best_owners.len());
    for owner in best_owners {
        let point = low_to_tile_point(owner, mask_width, mask_height, tile);
        if contour.last().copied() != Some(point) {
            contour.push(point);
        }
    }
    if contour.len() >= 5 {
        if let Some(start) = (0..contour.len()).min_by(|&first, &second| {
            contour[first]
                .1
                .total_cmp(&contour[second].1)
                .then_with(|| contour[first].0.total_cmp(&contour[second].0))
        }) {
            contour.rotate_left(start);
        }
    }
    contour
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

fn moments_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 20 {
        return None;
    }
    let count = points.len() as f64;
    let center = points
        .iter()
        .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
    let center = (center.0 / count, center.1 / count);
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    for &(x, y) in points {
        let dx = x - center.0;
        let dy = y - center.1;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    xx /= count;
    xy /= count;
    yy /= count;
    let root = ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt();
    let lambda_major = ((xx + yy + root) * 0.5).max(1.0);
    let lambda_minor = ((xx + yy - root) * 0.5).max(1.0);
    Some(Ellipse {
        center,
        major_radius: 2.0 * lambda_major.sqrt(),
        minor_radius: 2.0 * lambda_minor.sqrt(),
        angle: 0.5 * (2.0 * xy).atan2(xx - yy),
    })
}

fn solve_five(mut matrix: [[f64; 5]; 5], mut rhs: [f64; 5]) -> Option<[f64; 5]> {
    for pivot in 0..5 {
        let row = (pivot..5).max_by(|&first, &second| {
            matrix[first][pivot]
                .abs()
                .partial_cmp(&matrix[second][pivot].abs())
                .unwrap_or(CmpOrdering::Equal)
        })?;
        if matrix[row][pivot].abs() < 1e-12 {
            return None;
        }
        if row != pivot {
            matrix.swap(row, pivot);
            rhs.swap(row, pivot);
        }
        let divisor = matrix[pivot][pivot];
        for column in pivot..5 {
            matrix[pivot][column] /= divisor;
        }
        rhs[pivot] /= divisor;
        for row in 0..5 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..5 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

fn robust_contour_fit(points: &[(f64, f64)], initial: Ellipse) -> Option<Ellipse> {
    if points.len() < 24 {
        return None;
    }
    let mut parameters = [
        initial.center.0,
        initial.center.1,
        initial.major_radius.max(8.0),
        initial.minor_radius.max(8.0),
        initial.angle,
    ];
    for _ in 0..14 {
        let (sin_angle, cos_angle) = parameters[4].sin_cos();
        let major = parameters[2].max(8.0);
        let minor = parameters[3].max(8.0);
        let major2 = major * major;
        let minor2 = minor * minor;
        let mut normal = [[0.0f64; 5]; 5];
        let mut gradient = [0.0f64; 5];
        for &(x, y) in points {
            let dx = x - parameters[0];
            let dy = y - parameters[1];
            let xp = cos_angle * dx + sin_angle * dy;
            let yp = -sin_angle * dx + cos_angle * dy;
            let residual = xp * xp / major2 + yp * yp / minor2 - 1.0;
            let weight = if residual.abs() <= 0.18 {
                1.0
            } else {
                0.18 / residual.abs()
            };
            let jacobian = [
                -2.0 * xp * cos_angle / major2 + 2.0 * yp * sin_angle / minor2,
                -2.0 * xp * sin_angle / major2 - 2.0 * yp * cos_angle / minor2,
                -2.0 * xp * xp / major.powi(3),
                -2.0 * yp * yp / minor.powi(3),
                2.0 * xp * yp * (1.0 / major2 - 1.0 / minor2),
            ];
            for row in 0..5 {
                gradient[row] += weight * jacobian[row] * residual;
                for column in 0..5 {
                    normal[row][column] += weight * jacobian[row] * jacobian[column];
                }
            }
        }
        for index in 0..5 {
            normal[index][index] += 1e-7 * normal[index][index].abs().max(1.0);
        }
        let mut step = solve_five(normal, gradient.map(|value| -value))?;
        step[0] = step[0].clamp(-5.0, 5.0);
        step[1] = step[1].clamp(-5.0, 5.0);
        step[2] = step[2].clamp(-8.0, 8.0);
        step[3] = step[3].clamp(-8.0, 8.0);
        step[4] = step[4].clamp(-0.15, 0.15);
        for index in 0..5 {
            parameters[index] += step[index];
        }
        parameters[2] = parameters[2].clamp(12.0, FRAME_WIDTH as f64 * 0.55);
        parameters[3] = parameters[3].clamp(12.0, FRAME_HEIGHT as f64 * 0.70);
        if step.iter().map(|value| value.abs()).sum::<f64>() < 1e-4 {
            break;
        }
    }
    let mut ellipse = Ellipse {
        center: (parameters[0], parameters[1]),
        major_radius: parameters[2],
        minor_radius: parameters[3],
        angle: parameters[4],
    };
    normalize_ellipse(&mut ellipse);
    plausible_ellipse(ellipse).then_some(ellipse)
}

fn normalize_ellipse(ellipse: &mut Ellipse) {
    if ellipse.minor_radius > ellipse.major_radius {
        std::mem::swap(&mut ellipse.major_radius, &mut ellipse.minor_radius);
        ellipse.angle += std::f64::consts::FRAC_PI_2;
    }
    ellipse.angle = (ellipse.angle + std::f64::consts::FRAC_PI_2).rem_euclid(std::f64::consts::PI)
        - std::f64::consts::FRAC_PI_2;
}

fn plausible_ellipse(ellipse: Ellipse) -> bool {
    ellipse.center.0.is_finite()
        && ellipse.center.1.is_finite()
        && ellipse.major_radius.is_finite()
        && ellipse.minor_radius.is_finite()
        && ellipse.center.0 > -(FRAME_WIDTH as f64) * 0.1
        && ellipse.center.0 < FRAME_WIDTH as f64 * 1.1
        && ellipse.center.1 > -(FRAME_HEIGHT as f64) * 0.1
        && ellipse.center.1 < FRAME_HEIGHT as f64 * 1.1
        && (30.0..=FRAME_WIDTH as f64 * 0.52).contains(&ellipse.major_radius)
        && (24.0..=FRAME_HEIGHT as f64 * 0.65).contains(&ellipse.minor_radius)
        && ellipse.minor_radius / ellipse.major_radius >= 0.28
}

fn direct_conic_fit(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 5 {
        return None;
    }
    let count = points.len() as f64;
    let center = points
        .iter()
        .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
    let center = (center.0 / count, center.1 / count);
    let scale = (points
        .iter()
        .map(|point| (point.0 - center.0).powi(2) + (point.1 - center.1).powi(2))
        .sum::<f64>()
        / count)
        .sqrt()
        .max(1.0);
    let mut normal = [[0.0f64; 5]; 5];
    let mut rhs = [0.0f64; 5];
    for &(point_x, point_y) in points {
        let x = (point_x - center.0) / scale;
        let y = (point_y - center.1) / scale;
        let terms = [x * x, x * y, y * y, x, y];
        for row in 0..5 {
            rhs[row] += terms[row];
            for column in 0..5 {
                normal[row][column] += terms[row] * terms[column];
            }
        }
    }
    for index in 0..5 {
        normal[index][index] += 1e-10;
    }
    let [quadratic_x, cross, quadratic_y, linear_x, linear_y] = solve_five(normal, rhs)?;
    let off_diagonal = cross * 0.5;
    let determinant = quadratic_x * quadratic_y - off_diagonal * off_diagonal;
    if determinant <= 1e-10 {
        return None;
    }
    let local_center_x = -0.5 * (quadratic_y * linear_x - off_diagonal * linear_y) / determinant;
    let local_center_y = -0.5 * (-off_diagonal * linear_x + quadratic_x * linear_y) / determinant;
    let level = 1.0
        + quadratic_x * local_center_x * local_center_x
        + 2.0 * off_diagonal * local_center_x * local_center_y
        + quadratic_y * local_center_y * local_center_y;
    let trace = quadratic_x + quadratic_y;
    let discriminant = ((quadratic_x - quadratic_y).powi(2) + 4.0 * off_diagonal.powi(2)).sqrt();
    let eigen_minimum = (trace - discriminant) * 0.5;
    let eigen_maximum = (trace + discriminant) * 0.5;
    if level <= 0.0 || eigen_minimum <= 1e-10 || eigen_maximum <= eigen_minimum {
        return None;
    }
    let major_vector = if off_diagonal.abs() > 1e-10 {
        (off_diagonal, eigen_minimum - quadratic_x)
    } else if quadratic_x <= quadratic_y {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let mut ellipse = Ellipse {
        center: (
            center.0 + local_center_x * scale,
            center.1 + local_center_y * scale,
        ),
        major_radius: scale * (level / eigen_minimum).sqrt(),
        minor_radius: scale * (level / eigen_maximum).sqrt(),
        angle: major_vector.1.atan2(major_vector.0),
    };
    normalize_ellipse(&mut ellipse);
    (ellipse.center.0.is_finite()
        && ellipse.center.1.is_finite()
        && ellipse.major_radius.is_finite()
        && ellipse.minor_radius.is_finite()
        && ellipse.major_radius < 2000.0
        && ellipse.minor_radius >= 2.0
        && ellipse.minor_radius / ellipse.major_radius >= 0.05)
        .then_some(ellipse)
}

#[cfg(feature = "sam31")]
fn least_squares_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    use opencv::core::{Point2f, Vector};

    if points.len() < 5 {
        return None;
    }
    let mut input = Vector::<Point2f>::with_capacity(points.len());
    for &(x, y) in points {
        input.push(Point2f::new(x as f32, y as f32));
    }
    let fitted = opencv::imgproc::fit_ellipse(&input).ok()?;
    let mut ellipse = Ellipse {
        center: (fitted.center.x as f64, fitted.center.y as f64),
        major_radius: fitted.size.width as f64 * 0.5,
        minor_radius: fitted.size.height as f64 * 0.5,
        angle: (fitted.angle as f64).to_radians(),
    };
    normalize_ellipse(&mut ellipse);
    (ellipse.center.0.is_finite()
        && ellipse.center.1.is_finite()
        && ellipse.major_radius.is_finite()
        && ellipse.minor_radius.is_finite()
        && ellipse.major_radius < 2000.0
        && ellipse.minor_radius >= 2.0
        && ellipse.minor_radius / ellipse.major_radius >= 0.05)
        .then_some(ellipse)
}

#[cfg(not(feature = "sam31"))]
fn least_squares_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    direct_conic_fit(points)
}

fn ellipse_residual(point: (f64, f64), ellipse: Ellipse) -> f64 {
    let (sin_angle, cos_angle) = ellipse.angle.sin_cos();
    let dx = point.0 - ellipse.center.0;
    let dy = point.1 - ellipse.center.1;
    let x = cos_angle * dx + sin_angle * dy;
    let y = -sin_angle * dx + cos_angle * dy;
    let radial = ((x / ellipse.major_radius.max(1e-6)).powi(2)
        + (y / ellipse.minor_radius.max(1e-6)).powi(2))
    .sqrt();
    (radial - 1.0).abs() * (ellipse.major_radius * ellipse.minor_radius).sqrt()
}

struct RobustFit {
    ellipse: Ellipse,
    residuals: Vec<f64>,
    inliers: Vec<bool>,
    cutoff: f64,
}

struct NumpyPcg64 {
    state: u128,
    increment: u128,
    buffered_upper: Option<u32>,
}

impl NumpyPcg64 {
    const MULTIPLIER: u128 =
        ((2_549_297_995_355_413_924u128) << 64) | 4_865_540_595_714_422_341u128;

    fn baseline_fit_stream() -> Self {
        // np.random.default_rng(20260812).bit_generator.state after NumPy's
        // SeedSequence expansion. Keeping this tiny PCG stream in Rust avoids
        // a Python dependency while preserving the accepted six-query fit.
        Self {
            state: 94_368_598_227_006_152_144_556_554_533_211_010_852u128,
            increment: 174_605_511_227_888_025_715_320_101_340_171_151_329u128,
            buffered_upper: None,
        }
    }

    fn vertical_cull_stream() -> Self {
        // np.random.default_rng(20260813).bit_generator.state.
        Self {
            state: 32_836_341_824_432_678_873_702_219_582_571_864_733u128,
            increment: 70_717_288_352_767_355_186_512_493_438_108_981_773u128,
            buffered_upper: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.increment);
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((self.state >> 122) as u32)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.buffered_upper.take() {
            return value;
        }
        let value = self.next_u64();
        self.buffered_upper = Some((value >> 32) as u32);
        value as u32
    }

    fn bounded_inclusive(&mut self, maximum: u32) -> u32 {
        if maximum == 0 {
            return 0;
        }
        let range = maximum + 1;
        let threshold = (u32::MAX - maximum) % range;
        loop {
            let product = self.next_u32() as u64 * range as u64;
            if product as u32 >= threshold {
                return (product >> 32) as u32;
            }
        }
    }

    fn choice_five(&mut self, population: usize) -> Option<[usize; 5]> {
        if population < 5 || population > u32::MAX as usize {
            return None;
        }
        // Generator.choice uses Floyd's algorithm here. For a five-element
        // draw its 1.2x hash table always rounds up to eight slots.
        let mut hash = [usize::MAX; 8];
        let mut choice = [0usize; 5];
        for (slot, candidate) in ((population - 5)..population).enumerate() {
            let value = self.bounded_inclusive(candidate as u32) as usize;
            let mut location = value & 7;
            while hash[location] != usize::MAX && hash[location] != value {
                location = (location + 1) & 7;
            }
            if hash[location] == usize::MAX {
                hash[location] = value;
                choice[slot] = value;
            } else {
                location = candidate & 7;
                while hash[location] != usize::MAX {
                    location = (location + 1) & 7;
                }
                hash[location] = candidate;
                choice[slot] = candidate;
            }
        }
        for index in (1..5).rev() {
            let other = self.bounded_inclusive(index as u32) as usize;
            choice.swap(index, other);
        }
        Some(choice)
    }
}

fn robust_ransac_ellipse(points: &[(f64, f64)], random: &mut NumpyPcg64) -> Option<RobustFit> {
    if points.len() < 5 {
        return None;
    }
    let mut best: Option<(usize, f64, Ellipse, Vec<bool>)> = None;
    for _ in 0..4000 {
        let indices = random.choice_five(points.len())?;
        let subset = indices.map(|index| points[index]);
        let Some(ellipse) = least_squares_ellipse(&subset) else {
            continue;
        };
        let residuals = points
            .iter()
            .map(|&point| ellipse_residual(point, ellipse))
            .collect::<Vec<_>>();
        let inliers = residuals
            .iter()
            .map(|residual| *residual <= 2.5)
            .collect::<Vec<_>>();
        let count = inliers.iter().filter(|&&inlier| inlier).count();
        if count < 5 {
            continue;
        }
        let sum = residuals
            .iter()
            .zip(inliers.iter())
            .filter_map(|(residual, inlier)| inlier.then_some(*residual))
            .sum::<f64>();
        if best.as_ref().is_none_or(|candidate| {
            count > candidate.0 || (count == candidate.0 && sum < candidate.1)
        }) {
            best = Some((count, sum, ellipse, inliers));
        }
    }
    let (_, _, _, first_inliers) = best?;
    let first_points = points
        .iter()
        .zip(first_inliers.iter())
        .filter_map(|(&point, &inlier)| inlier.then_some(point))
        .collect::<Vec<_>>();
    let mut ellipse = least_squares_ellipse(&first_points)?;
    let mut residuals = points
        .iter()
        .map(|&point| ellipse_residual(point, ellipse))
        .collect::<Vec<_>>();
    let residual_median = median(residuals.clone());
    let mad = median(
        residuals
            .iter()
            .map(|residual| (residual - residual_median).abs())
            .collect(),
    );
    let cutoff = 2.5f64.max(residual_median + 3.5 * (1.4826 * mad).max(0.5));
    let mut inliers = residuals
        .iter()
        .map(|residual| *residual <= cutoff)
        .collect::<Vec<_>>();
    let final_points = points
        .iter()
        .zip(inliers.iter())
        .filter_map(|(&point, &inlier)| inlier.then_some(point))
        .collect::<Vec<_>>();
    ellipse = least_squares_ellipse(&final_points)?;
    residuals = points
        .iter()
        .map(|&point| ellipse_residual(point, ellipse))
        .collect();
    inliers = residuals
        .iter()
        .map(|residual| *residual <= cutoff)
        .collect();
    Some(RobustFit {
        ellipse,
        residuals,
        inliers,
        cutoff,
    })
}

fn sample_closed_contour(points: &[(f64, f64)], count: usize) -> Vec<(f64, f64)> {
    if points.len() <= 1 || count == 0 {
        return points.to_vec();
    }
    let mut cumulative = Vec::with_capacity(points.len() + 1);
    cumulative.push(0.0);
    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        let length = (points[next].0 - points[index].0).hypot(points[next].1 - points[index].1);
        cumulative.push(cumulative.last().copied().unwrap() + length);
    }
    let total = *cumulative.last().unwrap();
    if total <= 1e-9 {
        return points.iter().copied().take(count).collect();
    }
    (0..count)
        .map(|sample| {
            let target = (sample as f64 + 0.5) * total / count as f64;
            let segment = cumulative
                .partition_point(|value| *value <= target)
                .saturating_sub(1)
                .min(points.len() - 1);
            let next = (segment + 1) % points.len();
            let span = (cumulative[segment + 1] - cumulative[segment]).max(1e-9);
            let blend = (target - cumulative[segment]) / span;
            (
                (points[segment].0 * (1.0 - blend) + points[next].0 * blend) as f32 as f64,
                (points[segment].1 * (1.0 - blend) + points[next].1 * blend) as f32 as f64,
            )
        })
        .collect()
}

fn baseline_mask_fit(contour: Vec<(f64, f64)>) -> Option<Ellipse> {
    // sample_contour in the six-query baseline walks cv2's external contour in
    // boundary order. Sorting by polar angle is not equivalent for an eyelid-
    // occluded iris: it can jump between unrelated arcs and manufacture a
    // pinched or undersized ellipse.
    let samples = sample_closed_contour(&contour, 47);
    let mut fit_random = NumpyPcg64::baseline_fit_stream();
    let first = robust_ransac_ellipse(&samples, &mut fit_random)?;
    let vertical_radius = (0..360)
        .map(|index| {
            let phase = index as f64 * std::f64::consts::TAU / 360.0;
            let (phase_sine, phase_cosine) = phase.sin_cos();
            let (angle_sine, angle_cosine) = first.ellipse.angle.sin_cos();
            let y = first.ellipse.center.1
                + first.ellipse.major_radius * phase_cosine * angle_sine
                + first.ellipse.minor_radius * phase_sine * angle_cosine;
            (y - first.ellipse.center.1).abs()
        })
        .fold(1.0f64, f64::max);
    let top = first.ellipse.center.1 - 0.84 * vertical_radius;
    let bottom = first.ellipse.center.1 + 0.78 * vertical_radius;
    let mut contacts = first
        .inliers
        .iter()
        .map(|inlier| !*inlier)
        .collect::<Vec<_>>();
    let lid_seeds = samples
        .iter()
        .map(|point| point.1 <= top || point.1 >= bottom)
        .collect::<Vec<_>>();
    for index in 0..samples.len() {
        if lid_seeds[index]
            || lid_seeds[(index + samples.len() - 1) % samples.len()]
            || lid_seeds[(index + 1) % samples.len()]
            || lid_seeds[(index + samples.len() - 2) % samples.len()]
            || lid_seeds[(index + 2) % samples.len()]
        {
            contacts[index] = true;
        }
    }
    let trusted_indices = contacts
        .iter()
        .enumerate()
        .filter_map(|(index, contact)| (!*contact).then_some(index))
        .collect::<Vec<_>>();
    let trusted = trusted_indices
        .iter()
        .map(|&index| samples[index])
        .collect::<Vec<_>>();
    let trusted_fit = robust_ransac_ellipse(&trusted, &mut fit_random)?;
    let mut final_inliers = vec![false; samples.len()];
    for (&sample_index, &inlier) in trusted_indices.iter().zip(trusted_fit.inliers.iter()) {
        final_inliers[sample_index] = inlier;
    }
    let retained_indices = final_inliers
        .iter()
        .enumerate()
        .filter_map(|(index, inlier)| inlier.then_some(index))
        .collect::<Vec<_>>();
    if retained_indices.len() < 5 {
        return None;
    }
    let minimum_y = retained_indices
        .iter()
        .map(|&index| samples[index].1)
        .fold(f64::INFINITY, f64::min);
    let maximum_y = retained_indices
        .iter()
        .map(|&index| samples[index].1)
        .fold(f64::NEG_INFINITY, f64::max);
    let margin = 0.08 * (maximum_y - minimum_y).max(1.0);
    let safe = retained_indices
        .iter()
        .filter_map(|&index| {
            let y = samples[index].1;
            (y > minimum_y + margin && y < maximum_y - margin).then_some(samples[index])
        })
        .collect::<Vec<_>>();
    let mut cull_random = NumpyPcg64::vertical_cull_stream();
    let result = robust_ransac_ellipse(&safe, &mut cull_random)?;
    let _diagnostic_use = (
        first.residuals.len(),
        trusted_fit.cutoff,
        result.residuals.len(),
    );
    plausible_ellipse(result.ellipse).then_some(result.ellipse)
}

fn fit_mask_component(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    tile: usize,
) -> Option<Ellipse> {
    let component = largest_tile_component(mask, mask_width, mask_height, tile);
    let scale =
        FILMSTRIP_WIDTH as f64 / mask_width as f64 * FRAME_HEIGHT as f64 / mask_height as f64;
    let full_area = component.len() as f64 * scale;
    if full_area < MIN_COMPONENT_AREA_FULL_RES as f64 {
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
    let baseline = baseline_mask_fit(contour);
    let initial_plausible = plausible_ellipse(initial);
    if std::env::var_os("BUTTERCUP_SAM31_FIT_DEBUG").is_some() {
        eprintln!(
            "SAM31_FIT_DEBUG tile={tile} component_pixels={} full_area={full_area:.1} contour_points={contour_points} initial={initial:?} initial_plausible={initial_plausible} baseline={baseline:?}",
            component.len(),
        );
    }
    baseline.or_else(|| initial_plausible.then_some(initial))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(|first, second| first.total_cmp(second));
    if values.len() & 1 == 1 {
        values[values.len() / 2]
    } else {
        0.5 * (values[values.len() / 2 - 1] + values[values.len() / 2])
    }
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
    }

    pub(super) fn worker(
        model_path: PathBuf,
        request: Receiver<Batch>,
        results: SyncSender<OuterResult>,
        status: Arc<Mutex<StatusSnapshot>>,
        stop: Arc<AtomicBool>,
    ) {
        if let Err(error) = load_cuda_dispatch_library() {
            update_status(&status, "error", &error);
            return;
        }
        configure_cuda_bfloat16_autocast();
        // Autocast is thread-local. Keep one guard around the worker lifetime
        // instead of toggling deprecated LibTorch state around every query.
        tch::autocast(true, || {
            let mut module: Option<CModule> = None;
            let device = Device::Cuda(0);
            let mut staging: Option<Tensor> = None;
            let mut warmed = false;
            while let Ok(batch) = request.recv() {
                if stop.load(AtomicOrdering::Acquire) {
                    break;
                }
                let started = Instant::now();
                if module.is_none() {
                    update_status(&status, "loading", "loading fixed-prompt SAM3.1 graph");
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
                if staging.is_none() {
                    staging = Some(
                        Tensor::zeros(
                            [1, 3, FRAME_HEIGHT as i64, FILMSTRIP_WIDTH as i64],
                            (Kind::Uint8, Device::Cpu),
                        )
                        .pin_memory(device),
                    );
                }
                if !warmed {
                    update_status(
                        &status,
                        "warming",
                        "warming fixed-prompt graph before publishing a boundary",
                    );
                    let balanced = balanced_quad_rgb(&batch.frames);
                    let warmup = write_quantized_filmstrip(
                        &balanced,
                        0.35,
                        99.65,
                        0.82,
                        staging_bytes(staging.as_ref().unwrap()),
                    )
                    .and_then(|_| {
                        infer(module.as_ref().unwrap(), staging.as_ref().unwrap(), device)
                            .map(|_| ())
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
                        "three adapters solving {}: quad RGB, RAW luma, log chroma",
                        batch.target.label()
                    ),
                );
                let run = process_batch(
                    module.as_ref().unwrap(),
                    staging.as_ref().unwrap(),
                    device,
                    &batch,
                );
                match run {
                    Ok(mut result) => {
                        result.elapsed_ms =
                            started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                        if let Ok(mut snapshot) = status.lock() {
                            snapshot.state = "ready";
                            snapshot.detail = match result.target {
                                Target::OuterLimbusAndInnerPupilVoid
                                    if result.sensor_pupil_ellipse.is_none() =>
                                {
                                    format!(
                                        "{} adapters agreed; outer limbus ready, optional pupil center unavailable",
                                        result.agreeing_adapters,
                                    )
                                }
                                Target::OuterLimbusAndInnerPupilVoid => format!(
                                    "{} adapters agreed; independent outer limbus and pupil center ready",
                                    result.agreeing_adapters,
                                ),
                                _ => format!(
                                    "{} adapters agreed; latest {} ellipse ready",
                                    result.agreeing_adapters,
                                    result.target.label(),
                                ),
                            };
                            snapshot.completed_batches =
                                snapshot.completed_batches.saturating_add(1);
                            snapshot.last_elapsed_ms = Some(result.elapsed_ms);
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
                        let state = if error.starts_with("SAM31 outer consensus")
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
        unsafe {
            std::slice::from_raw_parts_mut(staging.data_ptr() as *mut u8, FILMSTRIP_PIXELS * 3)
        }
    }

    fn infer(
        module: &CModule,
        staging: &Tensor,
        device: Device,
    ) -> Result<InferenceOutput, String> {
        // The staging tensor is deliberately reused for all adapters. Make
        // the one mandatory H2D copy synchronous before Rust mutates that
        // storage for the next adapter.
        let input = staging.to_device_(device, Kind::Uint8, false, true);
        let output = tch::no_grad(|| module.forward_is(&[IValue::Tensor(input)]))
            .map_err(|error| format!("SAM31 forward: {error}"))?;
        let (scores, masks): (Tensor, Tensor) = output
            .try_into()
            .map_err(|error| format!("SAM31 output tuple: {error}"))?;
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
        })
    }

    fn extract_latest_adapter_ellipse(
        output: &InferenceOutput,
    ) -> (Option<Ellipse>, f64, Option<usize>) {
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
            return (None, 0.0, None);
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
            let fit = fit_mask_component(&mask, FILMSTRIP_WIDTH, FRAME_HEIGHT, tile);
            if debug_all {
                eprintln!(
                    "SAM31_FIT_DEBUG query={} objective={:.6} model_score={:.6} fit={fit:?}",
                    candidate.query, candidate.objective, candidate.model_score,
                );
            }
            if let Some(ellipse) = fit {
                first_fit.get_or_insert((
                    ellipse,
                    candidate.model_score + candidate.objective * 0.05,
                    candidate.query,
                ));
                if !debug_all {
                    break;
                }
            }
        }
        if let Some((ellipse, score, query)) = first_fit {
            return (Some(ellipse), score, Some(query));
        }
        (
            None,
            first.model_score + first.objective * 0.05,
            Some(first.query),
        )
    }

    fn process_batch(
        module: &CModule,
        staging: &Tensor,
        device: Device,
        batch: &Batch,
    ) -> Result<OuterResult, String> {
        let mut latest = Vec::<(Ellipse, f64)>::new();

        let balanced = balanced_quad_rgb(&batch.frames);
        write_quantized_filmstrip(&balanced, 0.35, 99.65, 0.82, staging_bytes(staging))?;
        let output = infer(module, staging, device)?;
        let extracted = extract_latest_adapter_ellipse(&output);
        debug_adapter("quad_rgb", &extracted);
        if let (Some(ellipse), score, _) = extracted {
            latest.push((ellipse, score));
        }

        let luma = raw_luma(&batch.frames);
        write_quantized_filmstrip(&luma, 0.20, 99.75, 0.80, staging_bytes(staging))?;
        let output = infer(module, staging, device)?;
        let extracted = extract_latest_adapter_ellipse(&output);
        debug_adapter("raw_luma", &extracted);
        if let (Some(ellipse), score, _) = extracted {
            latest.push((ellipse, score));
        }

        let chroma = log_chroma(&balanced);
        write_quantized_filmstrip(&chroma, 0.60, 99.40, 1.0, staging_bytes(staging))?;
        let output = infer(module, staging, device)?;
        let extracted = extract_latest_adapter_ellipse(&output);
        debug_adapter("log_chroma", &extracted);
        if let (Some(ellipse), score, _) = extracted {
            latest.push((ellipse, score));
        }

        let latest_index = HISTORY_FRAMES - 1;
        let (outer_ellipse, agreeing) = agreeing_consensus(&latest).ok_or_else(|| {
            format!(
                "SAM31 outer consensus had no agreeing pair among {} plausible adapter fit(s)",
                latest.len()
            )
        })?;
        let source = &batch.frames[latest_index];
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
        })
    }

    fn debug_adapter(name: &str, extracted: &(Option<Ellipse>, f64, Option<usize>)) {
        if std::env::var_os("BUTTERCUP_SAM31_DEBUG").is_none() {
            return;
        }
        eprintln!("SAM31_DEBUG adapter={name} fits={extracted:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3)
        })
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
}
