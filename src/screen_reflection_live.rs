//! Live packed-RAW recovery of the full-screen optical clock.
//!
//! The camera receiver lends the newest native RAW10 allocation to this
//! bounded worker through an `Arc`; the render/tracking path never waits for
//! the scan and no preview, demosaic, or resized raster is constructed.  A
//! recovered code transition is sent back to the stimulus over its local Unix
//! socket so the stimulus can compare that camera packet's arrival with the
//! exact display commit that introduced the code.

use crate::screen_reflection_clock::{
    analyze_whole_raw_roi, detect_optical_activity_onset,
    solve_optical_clock_in_delta_range_with_scheme, ClockWitnessStream, WholeRoiClockWitness,
};
use crate::screen_reflection_code::{
    decode_soft_cells_constrained_with_scheme, DecodeGeometry, GridTransform, OpticalCodeScheme,
};
use crate::screen_reflection_raw::PackedRaw10;
use crate::screen_reflection_stimulus::{
    report_recovered_frame, report_recovery_progress, session_snapshot, ClockSessionSnapshot,
    RecoveryPhase, RecoveryProgress, RecoveryReport,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const QUEUE_FRAMES: usize = 3;
const TEMPORAL_WITNESSES_REQUIRED: usize = 24;
const OBSERVATION_FRAMES_MAXIMUM: usize = 192;
const SNAPSHOT_REFRESH: Duration = Duration::from_millis(120);
const INACTIVE_SNAPSHOT_REFRESH: Duration = Duration::from_millis(500);
const PROGRESS_REFRESH: Duration = Duration::from_millis(250);
const FIT_MINIMUM_INTERVAL: Duration = Duration::from_millis(250);
const MODEL_MAXIMUM_SENSOR_AGE: Duration = Duration::from_secs(4);
const MAXIMUM_PRIOR_DISTANCE_TICKS: u64 = 24;
const MINIMUM_OPTICAL_CONTROL_MARGIN: f64 = 0.002;
const MODEL_CONFIRMATIONS_REQUIRED: usize = 2;
const SINGLE_FRAME_MINIMUM_SCORE: f64 = 0.18;
const SINGLE_FRAME_MINIMUM_MARGIN: f64 = 0.08;

#[derive(Clone)]
pub struct LiveRawClockFrame {
    pub sequence: u64,
    pub sensor_timestamp_ns: u64,
    pub host_arrival_unix_ns: u64,
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub payload: Arc<Vec<u8>>,
}

pub struct LiveScreenClockAnalyzer {
    sender: Option<SyncSender<LiveRawClockFrame>>,
    submitted: Arc<AtomicU64>,
    dropped: Arc<AtomicU64>,
    worker: Option<thread::JoinHandle<()>>,
}

impl LiveScreenClockAnalyzer {
    pub fn start() -> Result<Self, String> {
        let (sender, receiver) = sync_channel(QUEUE_FRAMES);
        let submitted = Arc::new(AtomicU64::new(0));
        let dropped = Arc::new(AtomicU64::new(0));
        let worker = thread::Builder::new()
            .name("screen-clock-live-raw".to_string())
            .spawn(move || worker_loop(receiver))
            .map_err(|error| format!("spawn live screen-clock recovery: {error}"))?;
        Ok(Self {
            sender: Some(sender),
            submitted,
            dropped,
            worker: Some(worker),
        })
    }

    pub fn submit(&self, frame: LiveRawClockFrame) {
        let Some(sender) = self.sender.as_ref() else {
            return;
        };
        match sender.try_send(frame) {
            Ok(()) => {
                self.submitted.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for LiveScreenClockAnalyzer {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        let submitted = self.submitted.load(Ordering::Relaxed);
        let dropped = self.dropped.load(Ordering::Relaxed);
        if submitted > 0 || dropped > 0 {
            eprintln!("live optical-clock worker stopped submitted={submitted} dropped={dropped}");
        }
    }
}

#[derive(Clone)]
struct LiveClockObservation {
    sensor_timestamp_ns: u64,
    host_arrival_unix_ns: u64,
    witness: Option<WholeRoiClockWitness>,
}

#[derive(Clone, Copy, Debug)]
struct LiveClockModel {
    anchor_sensor_timestamp_ns: u64,
    rate_hz: f64,
    fractional_phase: f64,
    anchor_counter_delta: i32,
    score: f64,
    confidence_margin: f64,
    fitted_through_sensor_timestamp_ns: u64,
}

impl LiveClockModel {
    fn predicted_code(self, sensor_timestamp_ns: u64) -> Option<u64> {
        let seconds =
            (sensor_timestamp_ns as i128 - self.anchor_sensor_timestamp_ns as i128) as f64 / 1.0e9;
        let code = (self.rate_hz * seconds + self.fractional_phase).floor() as i64
            + i64::from(self.anchor_counter_delta);
        u64::try_from(code).ok()
    }

    fn fresh_for(self, sensor_timestamp_ns: u64) -> bool {
        sensor_timestamp_ns.saturating_sub(self.fitted_through_sensor_timestamp_ns)
            <= MODEL_MAXIMUM_SENSOR_AGE.as_nanos() as u64
    }
}

#[derive(Default)]
struct DecodeState {
    session_id: String,
    observations: VecDeque<LiveClockObservation>,
    valid_frames: usize,
    scanned_frames: usize,
    valid_since_fit: usize,
    model_candidate: Option<(LiveClockModel, usize)>,
    established_model: Option<LiveClockModel>,
    last_reported_code: Option<u64>,
    last_progress: Option<Instant>,
    last_fit_attempt: Option<Instant>,
    last_sensor_timestamp_ns: Option<u64>,
}

impl DecodeState {
    fn reset_for(&mut self, session_id: &str) {
        *self = Self {
            session_id: session_id.to_string(),
            observations: VecDeque::with_capacity(OBSERVATION_FRAMES_MAXIMUM),
            ..Self::default()
        };
    }

    fn send_progress(&mut self, phase: RecoveryPhase, force: bool) {
        if !force
            && self
                .last_progress
                .is_some_and(|last| last.elapsed() < PROGRESS_REFRESH)
        {
            return;
        }
        let _ = report_recovery_progress(RecoveryProgress {
            session_id: &self.session_id,
            phase,
            valid_frames: self.valid_frames.min(TEMPORAL_WITNESSES_REQUIRED),
            required_frames: TEMPORAL_WITNESSES_REQUIRED,
        });
        self.last_progress = Some(Instant::now());
    }

    fn observe_model(&mut self, candidate: LiveClockModel, reference_sensor_timestamp_ns: u64) {
        let agrees_with = |current: LiveClockModel| {
            current
                .predicted_code(reference_sensor_timestamp_ns)
                .zip(candidate.predicted_code(reference_sensor_timestamp_ns))
                .is_some_and(|(left, right)| left.abs_diff(right) <= 1)
                && (current.rate_hz - candidate.rate_hz).abs() <= 0.08
        };
        if let Some(established) = self.established_model {
            if agrees_with(established) {
                self.established_model = Some(candidate);
                self.model_candidate = None;
            } else {
                self.established_model = None;
                self.model_candidate = Some((candidate, 1));
            }
            return;
        }
        match self.model_candidate {
            Some((current, streak)) if agrees_with(current) => {
                let streak = streak.saturating_add(1);
                self.model_candidate = Some((candidate, streak));
                if streak >= MODEL_CONFIRMATIONS_REQUIRED {
                    self.established_model = Some(candidate);
                    self.model_candidate = None;
                }
            }
            _ => self.model_candidate = Some((candidate, 1)),
        }
    }
}

fn recovered_index_is_plausible(recovered: u64, current: u64) -> bool {
    // The snapshot is read after this camera packet arrived, so an optical
    // code from that packet cannot be newer than the display's current code.
    // One tick of slack covers snapshot/repeated-commit quantization; larger
    // future aliases would pin the monotonic decoder ahead of reality.
    recovered <= current.saturating_add(1)
        && current.saturating_add(1).saturating_sub(recovered) <= MAXIMUM_PRIOR_DISTANCE_TICKS
}

fn unwrap_counter_in_plausible_window(
    counter_mod: u16,
    current: u64,
    scheme: OpticalCodeScheme,
) -> Option<u64> {
    let modulus = u64::from(scheme.counter_modulus());
    let base = current / modulus * modulus + u64::from(counter_mod);
    [
        base.saturating_sub(modulus),
        base,
        base.saturating_add(modulus),
    ]
    .into_iter()
    .find(|candidate| recovered_index_is_plausible(*candidate, current))
}

#[derive(Clone, Copy, Debug)]
struct CheckedSingleFrameMatch {
    recovered_code_index: u64,
    score: f64,
    confidence_margin: f64,
}

fn checked_single_frame_match(
    witness: &WholeRoiClockWitness,
    snapshot: &ClockSessionSnapshot,
    host_arrival_unix_ns: u64,
) -> Option<CheckedSingleFrameMatch> {
    if snapshot.code_scheme != OpticalCodeScheme::ReedMullerV3
        || !witness.valid
        || witness.supported_cells < 27
    {
        return None;
    }
    let packet_time_code = display_code_near_host_time(snapshot, host_arrival_unix_ns);
    let expected_mod =
        (packet_time_code % u64::from(snapshot.code_scheme.counter_modulus())) as u16;
    // The reflected screen is an upright convex-mirror image and RAW CFA
    // opponent polarity is fixed. Holding that physical geometry is the
    // equivalent of a QR finder's orientation: allowing it to float would
    // spend the Reed-Muller distance on an artificial ambiguity.
    let decoded = decode_soft_cells_constrained_with_scheme(
        &witness.canonical_cells,
        snapshot.session_tag,
        Some(expected_mod),
        snapshot.code_scheme.counter_modulus() - 1,
        DecodeGeometry {
            transform: GridTransform::Identity,
            polarity: 1,
        },
        snapshot.code_scheme,
    )?;
    if decoded.hard_bit_distance > snapshot.code_scheme.correctable_logical_bit_errors()
        || decoded.score < SINGLE_FRAME_MINIMUM_SCORE
        || decoded.confidence_margin < SINGLE_FRAME_MINIMUM_MARGIN
    {
        return None;
    }
    let recovered_code_index = unwrap_counter_in_plausible_window(
        decoded.counter_mod,
        packet_time_code,
        snapshot.code_scheme,
    )?;
    Some(CheckedSingleFrameMatch {
        recovered_code_index,
        score: decoded.score,
        confidence_margin: decoded.confidence_margin,
    })
}

fn display_code_near_host_time(snapshot: &ClockSessionSnapshot, host_unix_ns: u64) -> u64 {
    if host_unix_ns >= snapshot.present_commit_unix_ns {
        let elapsed_ns = host_unix_ns - snapshot.present_commit_unix_ns;
        let ticks = (elapsed_ns as f64 * snapshot.code_hz / 1.0e9).ceil() as u64;
        snapshot.code_index.saturating_add(ticks)
    } else {
        let elapsed_ns = snapshot.present_commit_unix_ns - host_unix_ns;
        let ticks = (elapsed_ns as f64 * snapshot.code_hz / 1.0e9).floor() as u64;
        snapshot.code_index.saturating_sub(ticks)
    }
}

fn fit_temporal_model(
    observations: &VecDeque<LiveClockObservation>,
    snapshot: &ClockSessionSnapshot,
) -> Option<LiveClockModel> {
    // The legacy Gray/CRC sequence can retain several convincing absolute
    // phase aliases on causal prefixes. It remains decodable offline with the
    // entire manifest, but must never generate a live latency readout.
    if snapshot.code_scheme == OpticalCodeScheme::GrayCrcV1 {
        return None;
    }
    if observations.len() < 32
        || observations
            .iter()
            .filter(|observation| {
                observation
                    .witness
                    .as_ref()
                    .is_some_and(|witness| witness.valid)
            })
            .count()
            < TEMPORAL_WITNESSES_REQUIRED
    {
        return None;
    }
    let timestamps_ns = observations
        .iter()
        .map(|observation| observation.sensor_timestamp_ns)
        .collect::<Vec<_>>();
    let stream = ClockWitnessStream {
        name: "subject-right-live".to_string(),
        witnesses: observations
            .iter()
            .map(|observation| observation.witness.clone())
            .collect(),
    };
    let (onset_index, onset_score, onset_runner_up_score) =
        detect_optical_activity_onset(std::slice::from_ref(&stream))?;
    let onset = &observations[onset_index];
    let host_phase_center = display_code_near_host_time(snapshot, onset.host_arrival_unix_ns);
    let host_phase_center = i32::try_from(host_phase_center.min(2047)).ok()?;
    let minimum_delta = host_phase_center.saturating_sub(MAXIMUM_PRIOR_DISTANCE_TICKS as i32);
    // Snapshot cadence and repeated display commits can put the packet-time
    // estimate one code tick behind. This is timing uncertainty, not a license
    // for the optical exposure to occur after packet arrival.
    let maximum_delta = host_phase_center.saturating_add(2);
    let maximum_code_index = snapshot
        .code_index
        .saturating_add(128)
        .max(maximum_delta.max(0) as u64);
    let fit = solve_optical_clock_in_delta_range_with_scheme(
        &timestamps_ns,
        std::slice::from_ref(&stream),
        onset_index,
        onset_score,
        onset_runner_up_score,
        snapshot.code_hz,
        maximum_code_index,
        snapshot.session_tag,
        minimum_delta,
        maximum_delta,
        snapshot.code_scheme,
    )?;
    let strongest_control = fit
        .different_counter_delta_runner_up_score
        .max(fit.wrong_session_tag_score)
        .max(fit.reversed_time_score)
        .max(fit.spatial_scramble_score);
    if fit.score <= strongest_control
        || fit.onset_score <= fit.onset_runner_up_score
        || fit.confidence_margin <= MINIMUM_OPTICAL_CONTROL_MARGIN
    {
        return None;
    }
    Some(LiveClockModel {
        anchor_sensor_timestamp_ns: timestamps_ns[onset_index],
        rate_hz: fit.rate_hz,
        fractional_phase: fit.fractional_phase,
        anchor_counter_delta: fit.onset_counter_delta,
        score: fit.score,
        confidence_margin: fit.confidence_margin,
        fitted_through_sensor_timestamp_ns: *timestamps_ns.last()?,
    })
}

fn process_frame(
    state: &mut DecodeState,
    snapshot: &ClockSessionSnapshot,
    frame: LiveRawClockFrame,
) {
    state.scanned_frames = state.scanned_frames.saturating_add(1);
    state.last_sensor_timestamp_ns = Some(frame.sensor_timestamp_ns);
    let witness = PackedRaw10::new(
        frame.payload.as_slice(),
        frame.width,
        frame.height,
        frame.stride,
        frame.sensor_x,
        frame.sensor_y,
    )
    .ok()
    .and_then(analyze_whole_raw_roi);
    let direct_witness = witness
        .as_ref()
        .is_some_and(|witness| witness.valid && witness.supported_cells >= 27);
    let checked_match = witness.as_ref().and_then(|witness| {
        checked_single_frame_match(witness, snapshot, frame.host_arrival_unix_ns)
    });
    let single_reported = checked_match.is_some_and(|matched| {
        report_recovered_frame(RecoveryReport {
            session_id: &state.session_id,
            sequence: frame.sequence,
            host_arrival_unix_ns: frame.host_arrival_unix_ns,
            recovered_code_index: matched.recovered_code_index,
            score: matched.score,
            confidence_margin: matched.confidence_margin,
            verified: false,
        })
        .is_ok()
    });
    if state.observations.len() == OBSERVATION_FRAMES_MAXIMUM {
        state.observations.pop_front();
    }
    state.observations.push_back(LiveClockObservation {
        sensor_timestamp_ns: frame.sensor_timestamp_ns,
        host_arrival_unix_ns: frame.host_arrival_unix_ns,
        witness,
    });
    state.valid_frames = state
        .observations
        .iter()
        .filter(|observation| {
            observation
                .witness
                .as_ref()
                .is_some_and(|witness| witness.valid && witness.supported_cells >= 27)
        })
        .count();
    state.valid_since_fit = state
        .valid_since_fit
        .saturating_add(usize::from(direct_witness));
    if state.valid_frames < TEMPORAL_WITNESSES_REQUIRED || state.observations.len() < 32 {
        if single_reported {
            state.send_progress(RecoveryPhase::Noisy, false);
        } else {
            state.send_progress(RecoveryPhase::Warming, false);
        }
        return;
    }

    let new_witnesses_required = if state.established_model.is_some() {
        4
    } else {
        1
    };
    let fit_due = state.valid_since_fit >= new_witnesses_required
        && state
            .last_fit_attempt
            .is_none_or(|attempt| attempt.elapsed() >= FIT_MINIMUM_INTERVAL);
    if fit_due {
        state.last_fit_attempt = Some(Instant::now());
        if let Some(model) = fit_temporal_model(&state.observations, snapshot) {
            state.observe_model(model, frame.sensor_timestamp_ns);
            state.valid_since_fit = 0;
        }
    }
    let Some(model) = state
        .established_model
        .filter(|model| model.fresh_for(frame.sensor_timestamp_ns))
    else {
        if single_reported {
            state.send_progress(RecoveryPhase::Noisy, false);
        } else {
            state.send_progress(RecoveryPhase::Searching, false);
        }
        return;
    };
    state.send_progress(RecoveryPhase::Locked, false);
    if !direct_witness {
        return;
    }
    let Some(recovered) = model.predicted_code(frame.sensor_timestamp_ns) else {
        return;
    };
    let packet_time_code = display_code_near_host_time(snapshot, frame.host_arrival_unix_ns);
    if !recovered_index_is_plausible(recovered, packet_time_code)
        || state
            .last_reported_code
            .is_some_and(|previous| recovered < previous)
    {
        state.send_progress(RecoveryPhase::Searching, false);
        return;
    }
    if state.last_reported_code == Some(recovered) {
        return;
    }
    match report_recovered_frame(RecoveryReport {
        session_id: &state.session_id,
        sequence: frame.sequence,
        host_arrival_unix_ns: frame.host_arrival_unix_ns,
        recovered_code_index: recovered,
        score: model.score,
        confidence_margin: model.confidence_margin,
        verified: true,
    }) {
        Ok(()) => state.last_reported_code = Some(recovered),
        Err(error) if error.contains("transition-window") => {
            // The display ring deliberately expires old epochs. Do not let a
            // delayed worker report pin the monotonic decoder to that epoch.
        }
        Err(_) => {}
    }
}

fn worker_loop(receiver: Receiver<LiveRawClockFrame>) {
    let mut state = DecodeState::default();
    let mut snapshot: Option<ClockSessionSnapshot> = None;
    let mut last_snapshot_attempt = Instant::now() - INACTIVE_SNAPSHOT_REFRESH;
    loop {
        let frame = match receiver.recv_timeout(Duration::from_millis(250)) {
            Ok(frame) => frame,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        let refresh_due = snapshot.as_ref().map_or_else(
            || last_snapshot_attempt.elapsed() >= INACTIVE_SNAPSHOT_REFRESH,
            |_| last_snapshot_attempt.elapsed() >= SNAPSHOT_REFRESH,
        );
        if refresh_due {
            last_snapshot_attempt = Instant::now();
            match session_snapshot() {
                Ok(current) => {
                    if state.session_id != current.session_id {
                        state.reset_for(&current.session_id);
                        state.send_progress(RecoveryPhase::Warming, true);
                    }
                    snapshot = Some(current);
                }
                Err(_) => {
                    snapshot = None;
                    continue;
                }
            }
        }
        let Some(current) = snapshot.as_ref() else {
            continue;
        };
        process_frame(&mut state, current, frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_time_projection_is_conservative_across_snapshot_cadence() {
        let snapshot = ClockSessionSnapshot {
            session_id: "test".to_string(),
            session_tag: 3,
            code_hz: 30.0,
            display_refresh_hz: 60.0,
            presentation_index: 200,
            code_index: 100,
            present_commit_unix_ns: 2_000_000_000,
            code_scheme: crate::screen_reflection_code::OpticalCodeScheme::PermutedCounterV2,
        };
        assert_eq!(display_code_near_host_time(&snapshot, 2_000_000_000), 100);
        assert_eq!(display_code_near_host_time(&snapshot, 1_966_666_666), 99);
        assert_eq!(display_code_near_host_time(&snapshot, 2_000_000_001), 101);
        assert_eq!(display_code_near_host_time(&snapshot, 2_067_000_000), 103);
    }

    #[test]
    fn fitted_clock_projects_on_the_sensor_time_axis() {
        let model = LiveClockModel {
            anchor_sensor_timestamp_ns: 10_000_000_000,
            rate_hz: 30.0,
            fractional_phase: 0.25,
            anchor_counter_delta: 7,
            score: 0.1,
            confidence_margin: 0.01,
            fitted_through_sensor_timestamp_ns: 10_000_000_000,
        };
        assert_eq!(model.predicted_code(10_000_000_000), Some(7));
        assert_eq!(model.predicted_code(10_100_000_000), Some(10));
    }

    #[test]
    fn live_prior_rejects_future_and_remote_counter_aliases() {
        assert!(recovered_index_is_plausible(100, 100));
        assert!(recovered_index_is_plausible(77, 100));
        assert!(recovered_index_is_plausible(101, 100));
        assert!(!recovered_index_is_plausible(76, 100));
        assert!(!recovered_index_is_plausible(102, 100));
    }

    fn checked_snapshot(current_code: u64) -> ClockSessionSnapshot {
        ClockSessionSnapshot {
            session_id: "checked-test".to_string(),
            session_tag: 9,
            code_hz: 30.0,
            display_refresh_hz: 60.0,
            presentation_index: current_code * 2,
            code_index: current_code,
            present_commit_unix_ns: 2_000_000_000,
            code_scheme: OpticalCodeScheme::ReedMullerV3,
        }
    }

    fn checked_witness(code_index: u64) -> WholeRoiClockWitness {
        use crate::screen_reflection_code::FrameCode;
        let scheme = OpticalCodeScheme::ReedMullerV3;
        WholeRoiClockWitness {
            proposal_score: 2.0,
            valid: true,
            quad_roi: crate::screen_reflection_raw::ProjectiveQuad {
                corners: [(0.0, 0.0), (64.0, 0.0), (64.0, 32.0), (0.0, 32.0)],
            },
            canonical_cells: FrameCode::new(code_index, 9)
                .physical_signs_for(scheme)
                .map(f64::from),
            supported_cells: 32,
            repeat_agreement: 1.0,
            component_area_carriers: 128,
            luminance_median: 400.0,
        }
    }

    #[test]
    fn one_checked_frame_recovers_without_temporal_agreement() {
        let snapshot = checked_snapshot(100);
        let mut witness = checked_witness(90);
        for logical in [1usize, 6, 14] {
            witness.canonical_cells.swap(
                crate::screen_reflection_code::PAIR_POSITIVE_CELLS[logical],
                crate::screen_reflection_code::PAIR_NEGATIVE_CELLS[logical],
            );
        }
        let matched = checked_single_frame_match(&witness, &snapshot, 2_000_000_000).unwrap();
        assert_eq!(matched.recovered_code_index, 90);
        assert!(matched.score > SINGLE_FRAME_MINIMUM_SCORE);
        assert!(matched.confidence_margin > SINGLE_FRAME_MINIMUM_MARGIN);
    }

    #[test]
    fn ambiguous_four_symbol_read_is_not_published() {
        let snapshot = checked_snapshot(100);
        let mut witness = checked_witness(90);
        let scheme = OpticalCodeScheme::ReedMullerV3;
        let current = crate::screen_reflection_code::FrameCode::new(90, 9);
        let other = crate::screen_reflection_code::FrameCode::new(91, 9);
        let mut changed = 0;
        for logical in 0..crate::screen_reflection_code::LOGICAL_BIT_COUNT {
            if current.optical_bit(logical, scheme) != other.optical_bit(logical, scheme)
                && changed < 4
            {
                witness.canonical_cells.swap(
                    crate::screen_reflection_code::PAIR_POSITIVE_CELLS[logical],
                    crate::screen_reflection_code::PAIR_NEGATIVE_CELLS[logical],
                );
                changed += 1;
            }
        }
        assert!(checked_single_frame_match(&witness, &snapshot, 2_000_000_000).is_none());
    }
}
