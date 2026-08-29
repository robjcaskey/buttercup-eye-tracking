//! Full-screen fixation stimulus carrying an optically recoverable frame clock.
//!
//! This is a host-only Buttercup presentation mode. It neither opens the
//! camera nor relies on compositor or packet timestamps for frame identity.

use crate::screen_reflection_code::{
    FrameCode, OpticalCodeScheme, CHECKED_COUNTER_BITS, CHECKED_COUNTER_MODULUS,
    DISPLAY_GRID_COLUMNS, DISPLAY_GRID_ROWS, GRID_COLUMNS, GRID_ROWS, LOGICAL_BIT_COUNT,
    PAIR_NEGATIVE_CELLS, PAIR_POSITIVE_CELLS,
};
use serde_json::{json, Value};
use softbuffer::{Context, Surface};
use std::collections::VecDeque;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::num::NonZeroU32;
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::wayland::WindowAttributesExtWayland;
use winit::raw_window_handle::{DisplayHandle, HasDisplayHandle};
use winit::window::{Fullscreen, Window, WindowId};

pub const SUBCOMMAND: &str = "--screen-clock-stimulus";

const SOCKET_FILE: &str = "buttercup-screen-clock.sock";
const APP_ID: &str = "buttercup-screen-reflection-clock";
const MANIFEST_SCHEMA: &str = "buttercup-screen-reflection-calibration-v1";
const CONTROL_TIMEOUT: Duration = Duration::from_millis(180);
const START_TIMEOUT: Duration = Duration::from_secs(4);
const SURFACE_SETTLE: Duration = Duration::from_millis(220);
const FULLSCREEN_RETRY: Duration = Duration::from_millis(180);
const MAX_FULLSCREEN_ATTEMPTS: u8 = 8;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(20);
const RECOVERY_STALE_AFTER: Duration = Duration::from_millis(900);
const LATENCY_SAMPLE_LIMIT: usize = 31;
const CODE_TRANSITION_LIMIT: usize = 96;
const DISPLAY_CADENCE_SAMPLE_LIMIT: usize = 61;
const STIMULUS_CODE_SCHEME: OpticalCodeScheme = OpticalCodeScheme::ReedMullerV3;

// Approximately isoluminant opponent colors. Blue carries most code energy so
// the clock remains comparatively unobtrusive around the fixation target.
const BASE_RGB: [f64; 3] = [0.39, 0.405, 0.42];
const OPPONENT_AXIS_RGB: [f64; 3] = [0.55, -0.30, 1.00];

#[derive(Clone, Debug)]
struct Config {
    output: PathBuf,
    render_hz: f64,
    code_hz: f64,
    amplitude: f64,
    screen_diagonal_inches: f64,
    viewing_distance_inches: f64,
    ball_diameter_degrees: f64,
    horizontal_period_seconds: f64,
    vertical_period_seconds: f64,
    warmup_seconds: f64,
    duration_seconds: Option<f64>,
    fullscreen: bool,
    self_test_render: bool,
}

#[derive(Clone, Copy, Debug)]
struct BallPose {
    x_norm: f64,
    y_norm: f64,
    vx_norm_per_sec: f64,
    vy_norm_per_sec: f64,
    phase: &'static str,
}

struct ScreenWindow {
    window: Arc<Window>,
    surface: Surface<DisplayHandle<'static>, Arc<Window>>,
}

struct Manifest {
    path: PathBuf,
    writer: BufWriter<File>,
    records_since_flush: usize,
    last_flush: Instant,
}

impl Manifest {
    fn create(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|error| format!("create {}: {error}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            writer: BufWriter::new(file),
            records_since_flush: 0,
            last_flush: Instant::now(),
        })
    }

    fn write(&mut self, value: &Value) -> Result<(), String> {
        serde_json::to_writer(&mut self.writer, value)
            .map_err(|error| format!("serialize manifest: {error}"))?;
        self.writer
            .write_all(b"\n")
            .map_err(|error| format!("write {}: {error}", self.path.display()))?;
        self.records_since_flush = self.records_since_flush.saturating_add(1);
        if self.records_since_flush >= 16 || self.last_flush.elapsed() >= Duration::from_millis(120)
        {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), String> {
        self.writer
            .flush()
            .map_err(|error| format!("flush {}: {error}", self.path.display()))?;
        self.records_since_flush = 0;
        self.last_flush = Instant::now();
        Ok(())
    }
}

impl Drop for Manifest {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

struct SocketGuard(PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct StimulusApp {
    context: Context<DisplayHandle<'static>>,
    window_state: Option<ScreenWindow>,
    listener: UnixListener,
    _socket_guard: SocketGuard,
    config: Config,
    manifest: Manifest,
    session_id: String,
    session_tag: u8,
    session_unix_ns: u64,
    target_workspace: Option<String>,
    started: Instant,
    presentation_index: u64,
    display_refresh_hz: f64,
    display_commit_intervals_ns: VecDeque<u64>,
    last_commit: Option<(Instant, u64)>,
    current_code_index: Option<u64>,
    current_commit_unix_ns: Option<u64>,
    code_transitions: VecDeque<CodeTransition>,
    recovery: RecoveryReadout,
    ready_at: Option<Instant>,
    surface_ready: bool,
    header_written: bool,
    fullscreen_attempts: u8,
    paused: bool,
    paused_at: Option<Instant>,
    accumulated_pause: Duration,
    fatal_error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ClockSessionSnapshot {
    pub session_id: String,
    pub session_tag: u8,
    pub code_hz: f64,
    pub display_refresh_hz: f64,
    pub presentation_index: u64,
    pub code_index: u64,
    pub present_commit_unix_ns: u64,
    pub code_scheme: OpticalCodeScheme,
}

#[derive(Clone, Copy, Debug)]
pub enum RecoveryPhase {
    Warming,
    Searching,
    Noisy,
    Locked,
}

impl RecoveryPhase {
    fn wire(self) -> &'static str {
        match self {
            Self::Warming => "warming",
            Self::Searching => "searching",
            Self::Noisy => "noisy",
            Self::Locked => "locked",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Warming => "WARMING",
            Self::Searching => "SEARCHING",
            Self::Noisy => "CHECKED SINGLE FRAME",
            Self::Locked => "LOCKED",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecoveryProgress<'a> {
    pub session_id: &'a str,
    pub phase: RecoveryPhase,
    pub valid_frames: usize,
    pub required_frames: usize,
}

#[derive(Clone, Debug)]
pub struct RecoveryReport<'a> {
    pub session_id: &'a str,
    pub sequence: u64,
    pub host_arrival_unix_ns: u64,
    pub recovered_code_index: u64,
    pub score: f64,
    pub confidence_margin: f64,
    /// A checked single-frame code is immediately reportable. Multi-frame
    /// agreement only upgrades it to a verified temporal lock.
    pub verified: bool,
}

#[derive(Clone, Copy, Debug)]
struct CodeTransition {
    code_index: u64,
    presentation_index: u64,
    commit_unix_ns: u64,
}

#[derive(Clone, Debug)]
struct RecoveryReadout {
    phase: RecoveryPhase,
    valid_frames: usize,
    required_frames: usize,
    last_update: Option<Instant>,
    last_sequence: Option<u64>,
    last_sequence_verified: bool,
    last_sampled_code_index: Option<u64>,
    last_accepted_sequence: Option<u64>,
    last_accepted_code_index: Option<u64>,
    last_accepted_at: Option<Instant>,
    last_observed_ms: Option<f64>,
    samples_ms: VecDeque<f64>,
    estimate_ms: Option<f64>,
    last_score: f64,
    last_margin: f64,
    rejected_samples: u64,
}

impl Default for RecoveryReadout {
    fn default() -> Self {
        Self {
            phase: RecoveryPhase::Warming,
            valid_frames: 0,
            required_frames: 24,
            last_update: None,
            last_sequence: None,
            last_sequence_verified: false,
            last_sampled_code_index: None,
            last_accepted_sequence: None,
            last_accepted_code_index: None,
            last_accepted_at: None,
            last_observed_ms: None,
            samples_ms: VecDeque::with_capacity(LATENCY_SAMPLE_LIMIT),
            estimate_ms: None,
            last_score: 0.0,
            last_margin: 0.0,
            rejected_samples: 0,
        }
    }
}

fn unix_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

unsafe extern "C" {
    #[link_name = "getuid"]
    fn system_getuid() -> u32;
}

fn socket_path() -> PathBuf {
    if let Some(path) = env::var_os("BUTTERCUP_SCREEN_CLOCK_SOCKET") {
        return PathBuf::from(path);
    }
    if let Some(runtime) = env::var_os("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime).join(SOCKET_FILE);
    }
    PathBuf::from(format!(
        "/run/user/{}/{}",
        unsafe { system_getuid() },
        SOCKET_FILE
    ))
}

fn send_control(request: &str) -> Result<String, String> {
    let path = socket_path();
    let mut stream = UnixStream::connect(&path)
        .map_err(|error| format!("connect {}: {error}", path.display()))?;
    stream
        .set_read_timeout(Some(CONTROL_TIMEOUT))
        .and_then(|_| stream.set_write_timeout(Some(CONTROL_TIMEOUT)))
        .map_err(|error| format!("set screen-clock timeout: {error}"))?;
    stream
        .write_all(format!("{request}\n").as_bytes())
        .map_err(|error| format!("send screen-clock {request}: {error}"))?;
    let mut response = [0u8; 1_024];
    let count = stream
        .read(&mut response)
        .map_err(|error| format!("read screen-clock {request}: {error}"))?;
    let response = String::from_utf8_lossy(&response[..count])
        .trim()
        .to_string();
    if response.is_empty() || response.starts_with("error") {
        Err(format!("screen-clock {request} returned {response:?}"))
    } else {
        Ok(response)
    }
}

pub fn session_snapshot() -> Result<ClockSessionSnapshot, String> {
    let response = send_control("snapshot")?;
    let value: Value = serde_json::from_str(&response)
        .map_err(|error| format!("parse screen-clock snapshot: {error}"))?;
    let required_u64 = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("screen-clock snapshot lacks {name}"))
    };
    let required_f64 = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("screen-clock snapshot lacks {name}"))
    };
    Ok(ClockSessionSnapshot {
        session_id: value
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "screen-clock snapshot lacks session_id".to_string())?
            .to_string(),
        session_tag: u8::try_from(required_u64("session_tag")?)
            .map_err(|_| "screen-clock session_tag is out of range".to_string())?,
        code_hz: required_f64("code_hz")?,
        display_refresh_hz: required_f64("display_refresh_hz")?,
        presentation_index: required_u64("presentation_index")?,
        code_index: required_u64("code_index")?,
        present_commit_unix_ns: required_u64("present_commit_unix_ns")?,
        code_scheme: value
            .get("code_scheme")
            .and_then(Value::as_str)
            .and_then(OpticalCodeScheme::from_wire_name)
            .unwrap_or(OpticalCodeScheme::GrayCrcV1),
    })
}

pub fn report_recovery_progress(progress: RecoveryProgress<'_>) -> Result<(), String> {
    let response = send_control(&format!(
        "progress {} {} {} {}",
        progress.session_id,
        progress.phase.wire(),
        progress.valid_frames,
        progress.required_frames,
    ))?;
    (response == "ok")
        .then_some(())
        .ok_or_else(|| format!("screen-clock progress returned {response:?}"))
}

pub fn report_recovered_frame(report: RecoveryReport<'_>) -> Result<(), String> {
    let response = send_control(&format!(
        "recovery {} {} {} {} {:.8} {:.8} {}",
        report.session_id,
        report.sequence,
        report.host_arrival_unix_ns,
        report.recovered_code_index,
        report.score,
        report.confidence_margin,
        if report.verified {
            "verified"
        } else {
            "single"
        },
    ))?;
    (response == "ok")
        .then_some(())
        .ok_or_else(|| format!("screen-clock recovery returned {response:?}"))
}

pub fn toggle_or_start() -> Result<bool, String> {
    if send_control("status").as_deref() == Ok("on") {
        send_control("quit")?;
        return Ok(false);
    }
    let executable = env::current_exe().map_err(|error| format!("locate viewer: {error}"))?;
    let mut child = Command::new(&executable)
        .arg(SUBCOMMAND)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("start {} {SUBCOMMAND}: {error}", executable.display()))?;
    let deadline = Instant::now() + START_TIMEOUT;
    loop {
        if send_control("status").as_deref() == Ok("on") {
            return Ok(true);
        }
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("poll screen-clock child: {error}"))?
        {
            return Err(format!(
                "screen-clock child exited during startup with {status}"
            ));
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            return Err(format!(
                "screen-clock did not bind {} within {:.1}s",
                socket_path().display(),
                START_TIMEOUT.as_secs_f64()
            ));
        }
        thread::sleep(Duration::from_millis(40));
    }
}

pub fn toggle_or_start_async(origin: &'static str) {
    let _ = thread::Builder::new()
        .name("screen-reflection-clock-toggle".to_string())
        .spawn(move || match toggle_or_start() {
            Ok(active) => eprintln!(
                "screen-reflection clock {} by {origin}; Z/Esc exits the stimulus",
                if active { "enabled" } else { "disabled" }
            ),
            Err(error) => eprintln!("screen-reflection clock toggle by {origin} failed: {error}"),
        });
}

fn bind_control_socket() -> Result<(UnixListener, SocketGuard), String> {
    let path = socket_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    if let Ok(metadata) = fs::symlink_metadata(&path) {
        if !metadata.file_type().is_socket() {
            return Err(format!(
                "refusing to replace non-socket screen-clock path {}",
                path.display()
            ));
        }
        if UnixStream::connect(&path).is_ok() {
            return Err("another screen-reflection clock is already running".to_string());
        }
        fs::remove_file(&path)
            .map_err(|error| format!("remove stale {}: {error}", path.display()))?;
    }
    let listener =
        UnixListener::bind(&path).map_err(|error| format!("bind {}: {error}", path.display()))?;
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))
        .map_err(|error| format!("secure {}: {error}", path.display()))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("set screen-clock socket nonblocking: {error}"))?;
    Ok((listener, SocketGuard(path)))
}

fn output_root() -> PathBuf {
    env::var_os("BUTTERCUP_OUTPUT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("outputs"))
}

fn default_output(now_ns: u64) -> PathBuf {
    output_root()
        .join("screen-reflection-calibration")
        .join(format!("session-{now_ns}.jsonl"))
}

fn environment_f64(name: &str, default: f64) -> Result<f64, String> {
    env::var(name)
        .map(|value| {
            value
                .parse::<f64>()
                .map_err(|error| format!("invalid {name}: {error}"))
        })
        .unwrap_or(Ok(default))
}

fn option_f64(name: &str, value: Option<String>) -> Result<f64, String> {
    value
        .ok_or_else(|| format!("{name} requires a value"))?
        .parse::<f64>()
        .map_err(|error| format!("invalid {name}: {error}"))
}

fn usage() -> &'static str {
    "screen clock options:\n\
     --output PATH                 presentation manifest\n\
     --hz HZ                       fixation render cadence [100]\n\
     --code-hz HZ                  optical identity cadence [30]\n\
     --amplitude FRACTION          opponent-color amplitude [0.035]\n\
     --duration-seconds N          automatically stop after N seconds\n\
     --windowed                    1280x720 diagnostic window\n\
     --self-test-render            exercise encoder and 4K renderer without a window\n\
     Keys in stimulus: Z/Esc/Q exit, Space pause/resume"
}

fn parse_config<I>(arguments: I, now_ns: u64) -> Result<Config, String>
where
    I: IntoIterator<Item = String>,
{
    let mut config = Config {
        output: default_output(now_ns),
        render_hz: environment_f64("BUTTERCUP_SCREEN_CLOCK_HZ", 100.0)?,
        code_hz: environment_f64("BUTTERCUP_SCREEN_CLOCK_CODE_HZ", 30.0)?,
        amplitude: environment_f64("BUTTERCUP_SCREEN_CLOCK_AMPLITUDE", 0.035)?,
        screen_diagonal_inches: environment_f64("BUTTERCUP_SCREEN_DIAGONAL_INCHES", 27.0)?,
        viewing_distance_inches: environment_f64("BUTTERCUP_VIEWING_DISTANCE_INCHES", 22.0)?,
        ball_diameter_degrees: 1.10,
        horizontal_period_seconds: 9.0,
        vertical_period_seconds: 7.0,
        warmup_seconds: 2.0,
        duration_seconds: None,
        fullscreen: true,
        self_test_render: false,
    };
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--output" => {
                config.output = PathBuf::from(
                    arguments
                        .next()
                        .ok_or_else(|| "--output requires a value".to_string())?,
                )
            }
            "--hz" => config.render_hz = option_f64("--hz", arguments.next())?,
            "--code-hz" => config.code_hz = option_f64("--code-hz", arguments.next())?,
            "--amplitude" => config.amplitude = option_f64("--amplitude", arguments.next())?,
            "--duration-seconds" => {
                config.duration_seconds = Some(option_f64("--duration-seconds", arguments.next())?)
            }
            "--windowed" => config.fullscreen = false,
            "--self-test-render" => config.self_test_render = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => {
                return Err(format!(
                    "unknown screen-clock option {argument:?}\n{}",
                    usage()
                ))
            }
        }
    }
    if !(20.0..=360.0).contains(&config.render_hz) {
        return Err("--hz must be in 20..=360".to_string());
    }
    if !(1.0..=120.0).contains(&config.code_hz) || config.code_hz > config.render_hz {
        return Err("--code-hz must be in 1..=120 and no greater than --hz".to_string());
    }
    if !(0.004..=0.12).contains(&config.amplitude) {
        return Err("--amplitude must be in 0.004..=0.12".to_string());
    }
    if !(10.0..=100.0).contains(&config.screen_diagonal_inches)
        || !(8.0..=120.0).contains(&config.viewing_distance_inches)
    {
        return Err("screen geometry is implausible".to_string());
    }
    if config
        .duration_seconds
        .is_some_and(|duration| duration <= 0.0)
    {
        return Err("--duration-seconds must be positive".to_string());
    }
    Ok(config)
}

fn screen_size_inches(diagonal: f64) -> (f64, f64) {
    let denominator = (16.0f64 * 16.0 + 9.0 * 9.0).sqrt();
    (diagonal * 16.0 / denominator, diagonal * 9.0 / denominator)
}

fn screen_field_degrees(config: &Config) -> (f64, f64) {
    let (width, height) = screen_size_inches(config.screen_diagonal_inches);
    let field = |extent: f64| {
        2.0 * (extent / (2.0 * config.viewing_distance_inches))
            .atan()
            .to_degrees()
    };
    (field(width), field(height))
}

fn ball_pose(config: &Config, elapsed_seconds: f64) -> BallPose {
    if elapsed_seconds < config.warmup_seconds {
        return BallPose {
            x_norm: 0.5,
            y_norm: 0.5,
            vx_norm_per_sec: 0.0,
            vy_norm_per_sec: 0.0,
            phase: "warmup",
        };
    }
    let time = elapsed_seconds - config.warmup_seconds;
    let x_omega = std::f64::consts::TAU / config.horizontal_period_seconds;
    let y_omega = std::f64::consts::TAU / config.vertical_period_seconds;
    BallPose {
        x_norm: (0.5 + 0.43 * (x_omega * time).sin()).clamp(0.07, 0.93),
        y_norm: (0.5 + 0.40 * (y_omega * time).sin()).clamp(0.10, 0.90),
        vx_norm_per_sec: 0.43 * x_omega * (x_omega * time).cos(),
        vy_norm_per_sec: 0.40 * y_omega * (y_omega * time).cos(),
        phase: "motion",
    }
}

fn channel(value: f64) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn packed_rgb(rgb: [f64; 3]) -> u32 {
    (u32::from(channel(rgb[0])) << 16)
        | (u32::from(channel(rgb[1])) << 8)
        | u32::from(channel(rgb[2]))
}

fn code_rgb(sign: i8, amplitude: f64) -> [f64; 3] {
    let sign = f64::from(sign);
    [
        BASE_RGB[0] + sign * amplitude * OPPONENT_AXIS_RGB[0],
        BASE_RGB[1] + sign * amplitude * OPPONENT_AXIS_RGB[1],
        BASE_RGB[2] + sign * amplitude * OPPONENT_AXIS_RGB[2],
    ]
}

fn ball_radius_pixels(config: &Config, width: usize) -> f64 {
    width as f64 * config.ball_diameter_degrees / screen_field_degrees(config).0 / 2.0
}

fn draw_disc(
    output: &mut [u32],
    width: usize,
    height: usize,
    center: (f64, f64),
    radius: f64,
    color: u32,
) {
    if width == 0 || height == 0 || radius <= 0.0 {
        return;
    }
    let x0 = (center.0 - radius).floor().max(0.0) as usize;
    let x1 = (center.0 + radius)
        .ceil()
        .min(width.saturating_sub(1) as f64) as usize;
    let y0 = (center.1 - radius).floor().max(0.0) as usize;
    let y1 = (center.1 + radius)
        .ceil()
        .min(height.saturating_sub(1) as f64) as usize;
    let radius_squared = radius * radius;
    for y in y0..=y1 {
        let dy = y as f64 + 0.5 - center.1;
        for x in x0..=x1 {
            let dx = x as f64 + 0.5 - center.0;
            if dx * dx + dy * dy <= radius_squared {
                output[y * width + x] = color;
            }
        }
    }
}

fn draw_target(
    output: &mut [u32],
    width: usize,
    height: usize,
    config: &Config,
    pose: BallPose,
) -> (f64, (f64, f64)) {
    let radius = ball_radius_pixels(config, width).clamp(8.0, height as f64 * 0.055);
    let center = (pose.x_norm * width as f64, pose.y_norm * height as f64);
    draw_disc(output, width, height, center, radius * 1.42, 0x0020_2428);
    draw_disc(output, width, height, center, radius, 0x00f2_d486);
    draw_disc(output, width, height, center, radius * 0.34, 0x00ff_f3c4);
    draw_disc(output, width, height, center, radius * 0.13, 0x0010_1113);
    (radius, center)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

fn latency_display_frames(latency_ms: f64, display_refresh_hz: f64) -> f64 {
    latency_ms * display_refresh_hz / 1_000.0
}

impl RecoveryReadout {
    fn update_progress(&mut self, phase: RecoveryPhase, valid: usize, required: usize) {
        self.phase = phase;
        self.valid_frames = valid;
        self.required_frames = required.max(1);
        self.last_update = Some(Instant::now());
    }

    fn accept(&mut self, report: &RecoveryReport<'_>, transition: CodeTransition) -> Option<f64> {
        let verified_upgrade = self.last_sequence == Some(report.sequence)
            && report.verified
            && !self.last_sequence_verified;
        if self.last_sequence.is_some_and(|sequence| {
            report.sequence < sequence || (report.sequence == sequence && !verified_upgrade)
        }) {
            return None;
        }
        self.last_sequence = Some(report.sequence);
        self.last_sequence_verified = report.verified;
        self.last_update = Some(Instant::now());
        self.phase = if report.verified {
            RecoveryPhase::Locked
        } else {
            RecoveryPhase::Noisy
        };
        self.last_score = report.score;
        self.last_margin = report.confidence_margin;
        if self.last_sampled_code_index == Some(report.recovered_code_index) && !verified_upgrade {
            return None;
        }
        self.last_sampled_code_index = Some(report.recovered_code_index);
        let latency_ns = report
            .host_arrival_unix_ns
            .checked_sub(transition.commit_unix_ns)?;
        let latency_ms = latency_ns as f64 / 1.0e6;
        if !(0.0..=500.0).contains(&latency_ms) {
            self.rejected_samples = self.rejected_samples.saturating_add(1);
            return None;
        }
        if report.verified && self.samples_ms.len() >= 5 {
            let center = median(self.samples_ms.iter().copied().collect())?;
            let deviation = median(
                self.samples_ms
                    .iter()
                    .map(|sample| (sample - center).abs())
                    .collect(),
            )
            .unwrap_or(0.0);
            // First-observation latency is quantized by the camera cadence.
            // Keep that real spread, but reject a false counter lock that is
            // several camera frames away from the established transition.
            let limit = (4.0 * 1.4826 * deviation).max(45.0);
            if (latency_ms - center).abs() > limit {
                self.rejected_samples = self.rejected_samples.saturating_add(1);
                return None;
            }
        }
        self.last_accepted_sequence = Some(report.sequence);
        self.last_accepted_code_index = Some(report.recovered_code_index);
        self.last_accepted_at = Some(Instant::now());
        self.last_observed_ms = Some(latency_ms);
        if report.verified {
            if self.samples_ms.len() == LATENCY_SAMPLE_LIMIT {
                self.samples_ms.pop_front();
            }
            self.samples_ms.push_back(latency_ms);
            self.estimate_ms = median(self.samples_ms.iter().copied().collect());
        }
        Some(latency_ms)
    }

    fn fresh(&self, now: Instant) -> bool {
        self.last_update
            .is_some_and(|updated| now.saturating_duration_since(updated) <= RECOVERY_STALE_AFTER)
    }
}

fn fill_rectangle(
    output: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    rectangle_width: i32,
    rectangle_height: i32,
    color: u32,
) {
    let x0 = x.clamp(0, width as i32) as usize;
    let y0 = y.clamp(0, height as i32) as usize;
    let x1 = (x + rectangle_width).clamp(0, width as i32) as usize;
    let y1 = (y + rectangle_height).clamp(0, height as i32) as usize;
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    for row in y0..y1 {
        output[row * width + x0..row * width + x1].fill(color);
    }
}

fn overlay_glyph(character: char) -> [u8; 7] {
    match character {
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [15, 16, 16, 16, 16, 16, 15],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 14],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [31, 4, 4, 4, 4, 4, 31],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        ':' => [0, 4, 4, 0, 4, 4, 0],
        '.' => [0, 0, 0, 0, 0, 4, 4],
        '>' => [16, 8, 4, 2, 4, 8, 16],
        _ => [0; 7],
    }
}

fn draw_overlay_text(
    output: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    text: &str,
    color: u32,
    scale: i32,
) {
    let mut cursor = x;
    for character in text.chars() {
        for (row, bits) in overlay_glyph(character).iter().enumerate() {
            for column in 0..5 {
                if bits & (1 << (4 - column)) != 0 {
                    fill_rectangle(
                        output,
                        width,
                        height,
                        cursor + column * scale,
                        y + row as i32 * scale,
                        scale,
                        scale,
                        color,
                    );
                }
            }
        }
        cursor += 6 * scale;
    }
}

fn draw_recovery_overlay(
    output: &mut [u32],
    width: usize,
    height: usize,
    recovery: &RecoveryReadout,
    display_refresh_hz: f64,
) {
    // This is measurement feedback, not decoration. Keep it legible from the
    // normal calibration distance even on a 4K panel.
    let scale = ((height / 480).max(2) as i32).clamp(2, 5);
    let x = 18 * scale;
    let y = 16 * scale;
    let line_height = 10 * scale;
    let panel_width = 58 * 6 * scale;
    let panel_height = 34 * scale;
    fill_rectangle(
        output,
        width,
        height,
        x - 6 * scale,
        y - 5 * scale,
        panel_width,
        panel_height,
        packed_rgb(BASE_RGB),
    );
    let now = Instant::now();
    let fresh = recovery.fresh(now);
    let phase = if fresh {
        recovery.phase
    } else {
        RecoveryPhase::Searching
    };
    let locked = fresh && matches!(phase, RecoveryPhase::Locked);
    // Preserve the most recent accepted camera observation on screen even if
    // the live solver subsequently loses lock. Blanking it would make a real
    // measurement indistinguishable from one that never existed.
    let primary = recovery.last_observed_ms.map_or_else(
        || "LAST CLOCK LAG  --.- FRAMES / --.- MS".to_string(),
        |latency_ms| {
            let frames = latency_display_frames(latency_ms, display_refresh_hz);
            format!("LAST CLOCK LAG  {frames:.1} FRAMES / {latency_ms:.1} MS")
        },
    );
    let secondary = match (
        recovery.last_accepted_code_index,
        recovery.last_accepted_sequence,
    ) {
        (Some(code_index), Some(sequence)) => {
            format!("RECOVERED CODE {code_index}  CAMERA SEQUENCE {sequence}")
        }
        _ => "RECOVERED CODE NONE".to_string(),
    };
    let tertiary = if recovery.estimate_ms.is_some() && locked {
        let median_ms = recovery.estimate_ms.unwrap_or_default();
        let median_frames = latency_display_frames(median_ms, display_refresh_hz);
        format!(
            "OPTICAL CLOCK {}  MEDIAN {median_frames:.1} FRAMES / {median_ms:.1} MS",
            phase.label(),
        )
    } else if fresh && matches!(phase, RecoveryPhase::Noisy) {
        format!(
            "{}  SCORE {:.2}  MARGIN {:.2}",
            phase.label(),
            recovery.last_score,
            recovery.last_margin,
        )
    } else if let (Some(_), Some(accepted_at)) =
        (recovery.last_observed_ms, recovery.last_accepted_at)
    {
        let age_seconds = now.saturating_duration_since(accepted_at).as_secs_f64();
        format!("OPTICAL CLOCK STALE  LAST RECEIVED {age_seconds:.1} S AGO")
    } else {
        format!(
            "OPTICAL CLOCK {}  {}/{}",
            phase.label(),
            recovery.valid_frames,
            recovery.required_frames,
        )
    };
    draw_overlay_text(
        output,
        width,
        height,
        x + scale,
        y + scale,
        &primary,
        0x0015_1b20,
        scale,
    );
    draw_overlay_text(output, width, height, x, y, &primary, 0x00e8_f4f6, scale);
    draw_overlay_text(
        output,
        width,
        height,
        x,
        y + line_height,
        &secondary,
        if fresh { 0x0095_e3d3 } else { 0x00c0_c8ce },
        scale,
    );
    draw_overlay_text(
        output,
        width,
        height,
        x,
        y + 2 * line_height,
        &tertiary,
        if locked { 0x0095_e3d3 } else { 0x00c0_c8ce },
        scale,
    );
}

fn render_stimulus(
    output: &mut [u32],
    width: usize,
    height: usize,
    config: &Config,
    code: FrameCode,
    pose: BallPose,
    recovery: &RecoveryReadout,
    display_refresh_hz: f64,
) -> (f64, (f64, f64)) {
    if output.len() != width * height {
        return (0.0, (0.0, 0.0));
    }
    let signs = code.display_signs_for(STIMULUS_CODE_SCHEME);
    let colors = [
        packed_rgb(code_rgb(-1, config.amplitude)),
        packed_rgb(code_rgb(1, config.amplitude)),
    ];
    for row in 0..DISPLAY_GRID_ROWS {
        let top = row * height / DISPLAY_GRID_ROWS;
        let bottom = (row + 1) * height / DISPLAY_GRID_ROWS;
        for column in 0..DISPLAY_GRID_COLUMNS {
            let left = column * width / DISPLAY_GRID_COLUMNS;
            let right = (column + 1) * width / DISPLAY_GRID_COLUMNS;
            let color = colors[usize::from(signs[row * DISPLAY_GRID_COLUMNS + column] > 0)];
            for y in top..bottom {
                output[y * width + left..y * width + right].fill(color);
            }
        }
    }
    let target = draw_target(output, width, height, config, pose);
    draw_recovery_overlay(output, width, height, recovery, display_refresh_hz);
    target
}

fn resize_surface(
    surface: &mut Surface<DisplayHandle<'static>, Arc<Window>>,
    size: PhysicalSize<u32>,
) -> Result<(), String> {
    let (Some(width), Some(height)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
    else {
        return Ok(());
    };
    surface
        .resize(width, height)
        .map_err(|error| format!("resize screen-clock surface: {error}"))
}

fn present_bootstrap(config: &Config, state: &mut ScreenWindow) -> Result<(), String> {
    let size = state.window.inner_size();
    if size.width == 0 || size.height == 0 {
        return Ok(());
    }
    resize_surface(&mut state.surface, size)?;
    let mut buffer = state
        .surface
        .buffer_mut()
        .map_err(|error| format!("acquire screen-clock bootstrap buffer: {error}"))?;
    buffer.fill(packed_rgb(BASE_RGB));
    draw_target(
        &mut buffer,
        size.width as usize,
        size.height as usize,
        config,
        BallPose {
            x_norm: 0.5,
            y_norm: 0.5,
            vx_norm_per_sec: 0.0,
            vy_norm_per_sec: 0.0,
            phase: "bootstrap",
        },
    );
    state.window.pre_present_notify();
    buffer
        .present()
        .map_err(|error| format!("present screen-clock bootstrap: {error}"))
}

fn find_pid_node(node: &Value, pid: u32) -> Option<&Value> {
    if node.get("pid").and_then(Value::as_u64) == Some(u64::from(pid)) {
        return Some(node);
    }
    for name in ["nodes", "floating_nodes"] {
        if let Some(children) = node.get(name).and_then(Value::as_array) {
            for child in children {
                if let Some(found) = find_pid_node(child, pid) {
                    return Some(found);
                }
            }
        }
    }
    None
}

fn focused_sway_workspace() -> Result<(Option<String>, Option<f64>), String> {
    if env::var_os("SWAYSOCK").is_none() {
        return Ok((None, None));
    }
    let output = Command::new("swaymsg")
        .args(["-t", "get_workspaces", "-r"])
        .output()
        .map_err(|error| format!("query focused Sway workspace: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let value: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("parse Sway workspace list: {error}"))?;
    let focused = value.as_array().and_then(|workspaces| {
        workspaces.iter().find_map(|workspace| {
            (workspace.get("focused").and_then(Value::as_bool) == Some(true)).then_some(workspace)
        })
    });
    let workspace_name = focused
        .and_then(|workspace| workspace.get("name"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let output_name = focused
        .and_then(|workspace| workspace.get("output"))
        .and_then(Value::as_str);
    let Some(output_name) = output_name else {
        return Ok((workspace_name, None));
    };
    let outputs = Command::new("swaymsg")
        .args(["-t", "get_outputs", "-r"])
        .output()
        .map_err(|error| format!("query Sway output refresh: {error}"))?;
    if !outputs.status.success() {
        return Err(String::from_utf8_lossy(&outputs.stderr).trim().to_string());
    }
    let outputs: Value = serde_json::from_slice(&outputs.stdout)
        .map_err(|error| format!("parse Sway output list: {error}"))?;
    let refresh_hz = outputs
        .as_array()
        .and_then(|outputs| {
            outputs
                .iter()
                .find(|output| output.get("name").and_then(Value::as_str) == Some(output_name))
        })
        .and_then(|output| output.get("current_mode"))
        .and_then(|mode| mode.get("refresh"))
        .and_then(Value::as_u64)
        .filter(|refresh_millihertz| *refresh_millihertz > 0)
        .map(|refresh_millihertz| refresh_millihertz as f64 / 1_000.0);
    Ok((workspace_name, refresh_hz))
}

fn command_output_error(context: &str, output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    match (stderr.is_empty(), stdout.is_empty()) {
        (false, false) => format!("{context}: {stderr}; response={stdout}"),
        (false, true) => format!("{context}: {stderr}"),
        (true, false) => format!("{context}: {stdout}"),
        (true, true) => format!("{context}: command exited with {}", output.status),
    }
}

fn sway_window_is_visible_fullscreen(pid: u32) -> Result<bool, String> {
    let tree = Command::new("swaymsg")
        .args(["-t", "get_tree", "-r"])
        .output()
        .map_err(|error| format!("verify Sway fullscreen: {error}"))?;
    if !tree.status.success() {
        return Err(command_output_error("verify Sway fullscreen", &tree));
    }
    let tree: Value = serde_json::from_slice(&tree.stdout)
        .map_err(|error| format!("parse Sway tree: {error}"))?;
    Ok(find_pid_node(&tree, pid).is_some_and(|node| {
        node.get("fullscreen_mode")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            > 0
            && node.get("visible").and_then(Value::as_bool) == Some(true)
    }))
}

fn enforce_sway_fullscreen(pid: u32, workspace: Option<&str>) -> Result<bool, String> {
    if env::var_os("SWAYSOCK").is_none() {
        return Ok(true);
    }
    // `fullscreen enable` is not idempotent in every Sway release: asking an
    // already-fullscreen container to enable it again may return an error even
    // though the desired state is valid. Verify before issuing mutations.
    if sway_window_is_visible_fullscreen(pid)? {
        return Ok(true);
    }
    let criterion = format!("[pid=\"{pid}\"]");
    let command = match workspace {
        Some(workspace) => {
            let workspace = serde_json::to_string(workspace)
                .map_err(|error| format!("quote Sway workspace: {error}"))?;
            format!(
                "{criterion} move container to workspace {workspace}; {criterion} focus; {criterion} fullscreen enable"
            )
        }
        None => format!("{criterion} focus; {criterion} fullscreen enable"),
    };
    let response = Command::new("swaymsg")
        .args(["-r", &command])
        .output()
        .map_err(|error| format!("enforce Sway fullscreen: {error}"))?;
    let verified = sway_window_is_visible_fullscreen(pid)?;
    if verified {
        Ok(true)
    } else if !response.status.success() {
        Err(command_output_error("enforce Sway fullscreen", &response))
    } else {
        Ok(false)
    }
}

impl StimulusApp {
    fn active_elapsed(&self, now: Instant) -> Duration {
        let current_pause = self.paused_at.map_or(Duration::ZERO, |paused| {
            now.saturating_duration_since(paused)
        });
        now.saturating_duration_since(self.started)
            .saturating_sub(self.accumulated_pause + current_pause)
    }

    fn log_event(&mut self, event: &str) {
        let _ = self.manifest.write(&json!({
            "record_type": "event",
            "event": event,
            "active_elapsed_ns": self.active_elapsed(Instant::now()).as_nanos().min(u128::from(u64::MAX)) as u64,
            "unix_ns": unix_time_ns(),
            "presentation_index": self.presentation_index
        }));
        let _ = self.manifest.flush();
    }

    fn write_header(&mut self, size: PhysicalSize<u32>) -> Result<(), String> {
        if self.header_written {
            return Ok(());
        }
        let (screen_width_inches, screen_height_inches) =
            screen_size_inches(self.config.screen_diagonal_inches);
        let (horizontal_field, vertical_field) = screen_field_degrees(&self.config);
        let pairs = (0..LOGICAL_BIT_COUNT)
            .map(|index| json!([PAIR_POSITIVE_CELLS[index], PAIR_NEGATIVE_CELLS[index]]))
            .collect::<Vec<_>>();
        self.manifest.write(&json!({
            "record_type": "session",
            "schema": MANIFEST_SCHEMA,
            "session_id": self.session_id,
            "session_tag": self.session_tag,
            "session_unix_ns": self.session_unix_ns,
            "output": self.manifest.path,
            "window_px": [size.width, size.height],
            "fullscreen": self.config.fullscreen,
            "sway_target_workspace": self.target_workspace,
            "target_hz": self.config.render_hz,
            "presentation_hz": self.display_refresh_hz,
            "detected_display_refresh_hz": self.display_refresh_hz,
            "render_loop": "wayland-frame-callback",
            "code_hz": self.config.code_hz,
            "screen_diagonal_inches": self.config.screen_diagonal_inches,
            "screen_size_inches": [screen_width_inches, screen_height_inches],
            "viewing_distance_inches": self.config.viewing_distance_inches,
            "screen_field_degrees": [horizontal_field, vertical_field],
            "ball_diameter_degrees": self.config.ball_diameter_degrees,
            "stimulus_kind": "balanced_chromatic_reflection_clock",
            "motion": {
                "kind": "smooth_lissajous_bounce",
                "horizontal_period_seconds": self.config.horizontal_period_seconds,
                "vertical_period_seconds": self.config.vertical_period_seconds,
                "warmup_seconds": self.config.warmup_seconds,
                "x_range_normalized": [0.07, 0.93],
                "y_range_normalized": [0.10, 0.90]
            },
            "code": {
                "grid": [DISPLAY_GRID_COLUMNS, DISPLAY_GRID_ROWS],
                "logical_grid": [GRID_COLUMNS, GRID_ROWS],
                "spatial_repeats": [2, 2],
                "counter_bits": CHECKED_COUNTER_BITS,
                "counter_modulus": CHECKED_COUNTER_MODULUS,
                "payload": "5-bit optical epoch counter in a session-keyed nonlinear coset",
                "symbol_sequence": STIMULUS_CODE_SCHEME.wire_name(),
                "temporal_whitening": "invertible affine permutation modulo 32",
                "error_correction": "RM(1,4) [16,5,8]; corrects 3 logical-symbol errors per frame",
                "minimum_hamming_distance": 8,
                "correctable_logical_symbol_errors": 3,
                "uncorrectable_policy": "reject instead of publishing latency",
                "pair_cells": pairs,
                "base_srgb": BASE_RGB,
                "opponent_axis_srgb": OPPONENT_AXIS_RGB,
                "amplitude": self.config.amplitude,
                "screen_mean_invariant": true,
                "every_2x2_block_mean_invariant": true
            },
            "timing_note": "the reflected spatial identity is authoritative; host commit time is diagnostic"
        }))?;
        self.manifest.flush()?;
        self.header_written = true;
        Ok(())
    }

    fn poll_control(&mut self, event_loop: &ActiveEventLoop) {
        loop {
            let (mut stream, _) = match self.listener.accept() {
                Ok(connection) => connection,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) => {
                    eprintln!("screen-clock control accept failed: {error}");
                    break;
                }
            };
            let _ = stream.set_read_timeout(Some(CONTROL_TIMEOUT));
            let _ = stream.set_write_timeout(Some(CONTROL_TIMEOUT));
            let mut request = [0u8; 512];
            let count = stream.read(&mut request).unwrap_or(0);
            let request = String::from_utf8_lossy(&request[..count])
                .trim()
                .to_string();
            let fields = request.split_whitespace().collect::<Vec<_>>();
            match fields.as_slice() {
                ["status"] => {
                    let _ = stream.write_all(b"on\n");
                }
                ["snapshot"] => match (self.current_code_index, self.current_commit_unix_ns) {
                    (Some(code_index), Some(commit_unix_ns)) => {
                        let response = json!({
                            "session_id": self.session_id,
                            "session_tag": self.session_tag,
                            "code_hz": self.config.code_hz,
                            "code_scheme": STIMULUS_CODE_SCHEME.wire_name(),
                            "display_refresh_hz": self.display_refresh_hz,
                            "presentation_index": self.presentation_index.saturating_sub(1),
                            "code_index": code_index,
                            "present_commit_unix_ns": commit_unix_ns,
                        });
                        let _ = serde_json::to_writer(&mut stream, &response);
                        let _ = stream.write_all(b"\n");
                    }
                    _ => {
                        let _ = stream.write_all(b"error not-ready\n");
                    }
                },
                ["progress", session, phase, valid, required] => {
                    if *session != self.session_id {
                        let _ = stream.write_all(b"error wrong-session\n");
                        continue;
                    }
                    let phase = match *phase {
                        "warming" => Some(RecoveryPhase::Warming),
                        "searching" => Some(RecoveryPhase::Searching),
                        "noisy" => Some(RecoveryPhase::Noisy),
                        "locked" => Some(RecoveryPhase::Locked),
                        _ => None,
                    };
                    let parsed = phase
                        .zip(valid.parse::<usize>().ok())
                        .zip(required.parse::<usize>().ok());
                    if let Some(((phase, valid), required)) = parsed {
                        self.recovery.update_progress(phase, valid, required);
                        let _ = stream.write_all(b"ok\n");
                    } else {
                        let _ = stream.write_all(b"error malformed-progress\n");
                    }
                }
                ["recovery", session, sequence, arrival, code_index, score, margin, verification] =>
                {
                    if *session != self.session_id {
                        let _ = stream.write_all(b"error wrong-session\n");
                        continue;
                    }
                    let parsed = sequence
                        .parse::<u64>()
                        .ok()
                        .zip(arrival.parse::<u64>().ok())
                        .zip(code_index.parse::<u64>().ok())
                        .zip(score.parse::<f64>().ok())
                        .zip(margin.parse::<f64>().ok());
                    let Some((
                        (((sequence, host_arrival_unix_ns), recovered_code_index), score),
                        confidence_margin,
                    )) = parsed
                    else {
                        let _ = stream.write_all(b"error malformed-recovery\n");
                        continue;
                    };
                    let transition = self
                        .code_transitions
                        .iter()
                        .copied()
                        .find(|transition| transition.code_index == recovered_code_index);
                    let Some(transition) = transition else {
                        let _ = stream.write_all(b"error code-outside-transition-window\n");
                        continue;
                    };
                    let report = RecoveryReport {
                        session_id: session,
                        sequence,
                        host_arrival_unix_ns,
                        recovered_code_index,
                        score,
                        confidence_margin,
                        verified: *verification == "verified",
                    };
                    if let Some(sample_ms) = self.recovery.accept(&report, transition) {
                        let _ = self.manifest.write(&json!({
                            "record_type": "clock-recovery-latency",
                            "sequence": sequence,
                            "recovered_code_index": recovered_code_index,
                            "transition_presentation_index": transition.presentation_index,
                            "transition_commit_unix_ns": transition.commit_unix_ns,
                            "camera_packet_arrival_unix_ns": host_arrival_unix_ns,
                            "observed_latency_ms": sample_ms,
                            "median_latency_ms": self.recovery.estimate_ms,
                            "display_refresh_hz": self.display_refresh_hz,
                            "median_latency_display_frames": self.recovery.estimate_ms.map(|latency| latency_display_frames(latency, self.display_refresh_hz)),
                            "decode_score": score,
                            "decode_confidence_margin": confidence_margin,
                            "verified_temporal_lock": report.verified,
                            "measurement_note": if report.verified {
                                "display transition commit to temporally verified camera-packet recovery"
                            } else {
                                "display transition commit to independently error-checked single-frame camera recovery"
                            }
                        }));
                    }
                    let _ = stream.write_all(b"ok\n");
                }
                ["quit"] | ["off"] | ["toggle"] => {
                    let _ = stream.write_all(b"off\n");
                    self.log_event("control_exit");
                    event_loop.exit();
                }
                _ => {
                    let _ = stream.write_all(b"error unknown-command\n");
                }
            }
        }
    }

    fn toggle_pause(&mut self) {
        let now = Instant::now();
        if self.paused {
            if let Some(paused_at) = self.paused_at.take() {
                self.accumulated_pause += now.saturating_duration_since(paused_at);
            }
            self.paused = false;
            self.log_event("resumed");
            if let Some(state) = self.window_state.as_ref() {
                state.window.request_redraw();
            }
        } else {
            self.paused = true;
            self.paused_at = Some(now);
            self.log_event("paused");
        }
    }

    fn present_now(&mut self) -> Result<(), String> {
        let Some(mut state) = self.window_state.take() else {
            return Ok(());
        };
        let result = self.present_to(&mut state);
        self.window_state = Some(state);
        result
    }

    fn present_to(&mut self, state: &mut ScreenWindow) -> Result<(), String> {
        let size = state.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }
        self.write_header(size)?;
        resize_surface(&mut state.surface, size)?;
        let render_started = Instant::now();
        let elapsed = self.active_elapsed(render_started);
        let elapsed_seconds = elapsed.as_secs_f64();
        let pose = ball_pose(&self.config, elapsed_seconds);
        let code_index = (elapsed_seconds * self.config.code_hz).floor() as u64;
        let code = FrameCode::new(code_index, self.session_tag);
        let mut buffer = state
            .surface
            .buffer_mut()
            .map_err(|error| format!("acquire screen-clock buffer: {error}"))?;
        let (radius, center) = render_stimulus(
            &mut buffer,
            size.width as usize,
            size.height as usize,
            &self.config,
            code,
            pose,
            &self.recovery,
            self.display_refresh_hz,
        );
        state.window.pre_present_notify();
        buffer
            .present()
            .map_err(|error| format!("present screen clock: {error}"))?;
        let committed = Instant::now();
        let commit_unix_ns = unix_time_ns();
        let inter_commit_ns = self.last_commit.map(|(previous, _)| {
            committed
                .saturating_duration_since(previous)
                .as_nanos()
                .min(u128::from(u64::MAX)) as u64
        });
        let inter_commit_us = inter_commit_ns.map(|interval| interval / 1_000);
        if let Some(interval_ns) =
            inter_commit_ns.filter(|interval| (8_000_000..=50_000_000).contains(interval))
        {
            if self.display_commit_intervals_ns.len() == DISPLAY_CADENCE_SAMPLE_LIMIT {
                self.display_commit_intervals_ns.pop_front();
            }
            self.display_commit_intervals_ns.push_back(interval_ns);
            if self.display_commit_intervals_ns.len() >= 9 {
                if let Some(period_ns) = median(
                    self.display_commit_intervals_ns
                        .iter()
                        .map(|interval| *interval as f64)
                        .collect(),
                ) {
                    let measured_hz = 1.0e9 / period_ns;
                    if (20.0..=360.0).contains(&measured_hz) {
                        self.display_refresh_hz = measured_hz;
                    }
                }
            }
        }
        if self.current_code_index != Some(code_index) {
            if self.code_transitions.len() == CODE_TRANSITION_LIMIT {
                self.code_transitions.pop_front();
            }
            self.code_transitions.push_back(CodeTransition {
                code_index,
                presentation_index: self.presentation_index,
                commit_unix_ns,
            });
        }
        self.current_code_index = Some(code_index);
        self.current_commit_unix_ns = Some(commit_unix_ns);
        self.manifest.write(&json!({
            "record_type": "presentation",
            "presentation_index": self.presentation_index,
            "code_index": code_index,
            "counter_mod": code.counter_mod,
            "gray": code.gray,
            "crc4": code.crc4,
            "logical_word": code.logical_word,
            "logical_word_hex": format!("{:04x}", code.logical_word),
            "optical_word": code.optical_word(STIMULUS_CODE_SCHEME),
            "optical_word_hex": format!("{:04x}", code.optical_word(STIMULUS_CODE_SCHEME)),
            "code_scheme": STIMULUS_CODE_SCHEME.wire_name(),
            "active_elapsed_ns": elapsed.as_nanos().min(u128::from(u64::MAX)) as u64,
            "present_commit_unix_ns": commit_unix_ns,
            "inter_commit_us": inter_commit_us,
            "redraw_source": "wayland-frame-callback",
            "render_duration_us": committed.saturating_duration_since(render_started).as_micros().min(u128::from(u64::MAX)) as u64,
            "window_px": [size.width, size.height],
            "ball_center_px": [center.0, center.1],
            "ball_center_normalized": [pose.x_norm, pose.y_norm],
            "ball_velocity_normalized_per_sec": [pose.vx_norm_per_sec, pose.vy_norm_per_sec],
            "ball_radius_px": radius,
            "phase": pose.phase
        }))?;
        self.last_commit = Some((committed, commit_unix_ns));
        self.presentation_index = self.presentation_index.wrapping_add(1);
        Ok(())
    }
}

impl ApplicationHandler for StimulusApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window_state.is_some() {
            return;
        }
        let mut attributes = Window::default_attributes()
            .with_title("Buttercup optical screen clock")
            .with_name(APP_ID, APP_ID)
            .with_decorations(false);
        if self.config.fullscreen {
            attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
        } else {
            attributes = attributes.with_inner_size(LogicalSize::new(1280.0, 720.0));
        }
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("create screen-clock window"),
        );
        if self.config.fullscreen {
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        window.set_cursor_visible(false);
        if self.target_workspace.is_none() {
            if let Some(refresh_millihertz) = window
                .current_monitor()
                .and_then(|monitor| monitor.refresh_rate_millihertz())
                .filter(|refresh| *refresh > 0)
            {
                self.display_refresh_hz = f64::from(refresh_millihertz) / 1_000.0;
            }
        }
        let surface = Surface::new(&self.context, Arc::clone(&window))
            .expect("create screen-clock softbuffer surface");
        self.window_state = Some(ScreenWindow { window, surface });
        self.ready_at = Some(Instant::now() + SURFACE_SETTLE);
        self.surface_ready = false;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.log_event("window_closed");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = self.window_state.as_mut() {
                    if let Err(error) = resize_surface(&mut state.surface, size) {
                        self.fatal_error = Some(error);
                        event_loop.exit();
                        return;
                    }
                    if !self.header_written {
                        if let Err(error) = present_bootstrap(&self.config, state) {
                            self.fatal_error = Some(error);
                            event_loop.exit();
                            return;
                        }
                    }
                }
                if !self.header_written && size.width > 0 && size.height > 0 {
                    self.surface_ready = false;
                    self.ready_at = Some(Instant::now() + SURFACE_SETTLE);
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !event.repeat =>
            {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape)
                    | PhysicalKey::Code(KeyCode::KeyQ)
                    | PhysicalKey::Code(KeyCode::KeyZ) => {
                        self.log_event("operator_exit");
                        event_loop.exit();
                    }
                    PhysicalKey::Code(KeyCode::Space) => self.toggle_pause(),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested if self.surface_ready && !self.paused => {
                if let Err(error) = self.present_now() {
                    self.fatal_error = Some(error);
                    event_loop.exit();
                    return;
                }
                if let Some(state) = self.window_state.as_ref() {
                    // `pre_present_notify` requested the Wayland frame
                    // callback. Winit holds this redraw until the compositor
                    // acknowledges that callback, phase-locking motion to the
                    // real output instead of a drifting userspace timer.
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.poll_control(event_loop);
        let now = Instant::now();
        if !self.surface_ready {
            let ready_at = self.ready_at.unwrap_or(now + SURFACE_SETTLE);
            if now < ready_at {
                event_loop.set_control_flow(ControlFlow::WaitUntil(ready_at));
                return;
            }
            if self.config.fullscreen {
                let verified =
                    enforce_sway_fullscreen(std::process::id(), self.target_workspace.as_deref());
                if verified
                    .as_ref()
                    .is_err_and(|_| self.fullscreen_attempts == 0)
                {
                    eprintln!("screen-clock fullscreen request is waiting for the compositor");
                }
                if !matches!(verified, Ok(true)) {
                    self.fullscreen_attempts = self.fullscreen_attempts.saturating_add(1);
                    if self.fullscreen_attempts >= MAX_FULLSCREEN_ATTEMPTS {
                        let error = match verified {
                            Ok(false) => {
                                "screen-clock window never became visible fullscreen".to_string()
                            }
                            Err(error) => {
                                format!("screen-clock fullscreen verification failed: {error}")
                            }
                            Ok(true) => unreachable!(),
                        };
                        self.fatal_error = Some(error);
                        event_loop.exit();
                        return;
                    }
                    let retry_at = now + FULLSCREEN_RETRY;
                    self.ready_at = Some(retry_at);
                    event_loop.set_control_flow(ControlFlow::WaitUntil(retry_at));
                    return;
                }
            }
            self.surface_ready = true;
            self.started = now;
            eprintln!(
                "screen-reflection clock visible at detected {:.3} Hz (requested {:.0} Hz, optical code {:.0} Hz); Z/Esc exits",
                self.display_refresh_hz, self.config.render_hz, self.config.code_hz
            );
            // A newly fullscreen Wayland surface is not guaranteed to receive
            // a redraw notification until it has committed a real buffer.
            // Commit frame zero here to start the pre_present_notify/frame-
            // callback chain instead of waiting for the very callback that
            // this first presentation is meant to arm.
            if let Err(error) = self.present_now() {
                self.fatal_error = Some(error);
                event_loop.exit();
                return;
            }
            if let Some(state) = self.window_state.as_ref() {
                state.window.request_redraw();
            }
        }
        if self
            .config
            .duration_seconds
            .is_some_and(|duration| self.active_elapsed(now).as_secs_f64() >= duration)
        {
            self.log_event("duration_complete");
            event_loop.exit();
            return;
        }
        if self.paused {
            event_loop.set_control_flow(ControlFlow::WaitUntil(now + Duration::from_millis(50)));
            return;
        }
        // Frame callbacks drive rendering. This short wake only services the
        // local recovery/status socket if the compositor is temporarily idle.
        event_loop.set_control_flow(ControlFlow::WaitUntil(now + CONTROL_POLL_INTERVAL));
    }
}

fn run_self_test(config: &Config) -> Result<(), String> {
    let width = 3840usize;
    let height = 2160usize;
    let mut output = vec![0u32; width * height];
    let started = Instant::now();
    let mut checksum = 0u32;
    for index in 0..120u64 {
        let elapsed = index as f64 / config.render_hz;
        let code = FrameCode::new((elapsed * config.code_hz).floor() as u64, 7);
        let (_, center) = render_stimulus(
            &mut output,
            width,
            height,
            config,
            code,
            ball_pose(config, elapsed),
            &RecoveryReadout::default(),
            config.render_hz,
        );
        if !(0.0..width as f64).contains(&center.0) || !(0.0..height as f64).contains(&center.1) {
            return Err("fixation target escaped the render surface".to_string());
        }
        checksum = output
            .iter()
            .step_by(4093)
            .fold(checksum, |sum, value| sum.rotate_left(3) ^ value);
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "screen-clock 4K self-test: 120 frames in {elapsed:.3}s ({:.1} fps), checksum={checksum:08x}",
        120.0 / elapsed.max(1.0e-9)
    );
    if checksum == 0 {
        return Err("screen-clock renderer produced an empty checksum".to_string());
    }
    Ok(())
}

pub fn run<I>(arguments: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let now_ns = unix_time_ns();
    let config = parse_config(arguments, now_ns)?;
    if config.self_test_render {
        return run_self_test(&config);
    }
    let (listener, socket_guard) = bind_control_socket()?;
    let (target_workspace, sway_refresh_hz) = focused_sway_workspace()?;
    let manifest = Manifest::create(&config.output)?;
    let session_mix =
        now_ns ^ u64::from(std::process::id()).rotate_left(21) ^ now_ns.rotate_right(17);
    let session_tag = (session_mix & 0x0f) as u8;
    let session_id = format!("{now_ns:016x}-{:08x}", std::process::id());
    let event_loop = EventLoop::new().map_err(|error| error.to_string())?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let display = event_loop
        .display_handle()
        .map_err(|error| error.to_string())?;
    let context = Context::new(unsafe {
        std::mem::transmute::<DisplayHandle<'_>, DisplayHandle<'static>>(display)
    })
    .map_err(|error| error.to_string())?;
    let now = Instant::now();
    let requested_refresh_hz = sway_refresh_hz.unwrap_or(config.render_hz);
    let mut app = StimulusApp {
        context,
        window_state: None,
        listener,
        _socket_guard: socket_guard,
        config,
        manifest,
        session_id,
        session_tag,
        session_unix_ns: now_ns,
        target_workspace,
        started: now,
        presentation_index: 0,
        display_refresh_hz: requested_refresh_hz,
        display_commit_intervals_ns: VecDeque::with_capacity(DISPLAY_CADENCE_SAMPLE_LIMIT),
        last_commit: None,
        current_code_index: None,
        current_commit_unix_ns: None,
        code_transitions: VecDeque::with_capacity(CODE_TRANSITION_LIMIT),
        recovery: RecoveryReadout::default(),
        ready_at: None,
        surface_ready: false,
        header_written: false,
        fullscreen_attempts: 0,
        paused: false,
        paused_at: None,
        accumulated_pause: Duration::ZERO,
        fatal_error: None,
    };
    eprintln!(
        "screen-reflection session={} manifest={}",
        app.session_id,
        app.manifest.path.display()
    );
    event_loop
        .run_app(&mut app)
        .map_err(|error| error.to_string())?;
    if let Some(error) = app.fatal_error {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        parse_config(Vec::<String>::new(), 123).unwrap()
    }

    #[test]
    fn motion_starts_centered_and_remains_on_screen() {
        let config = test_config();
        assert_eq!(ball_pose(&config, 0.0).x_norm, 0.5);
        assert_eq!(ball_pose(&config, 1.9).y_norm, 0.5);
        for sample in 0..5000 {
            let pose = ball_pose(&config, sample as f64 / 100.0);
            assert!((0.07..=0.93).contains(&pose.x_norm));
            assert!((0.10..=0.90).contains(&pose.y_norm));
        }
    }

    #[test]
    fn stated_screen_geometry_matches_the_calibration_setup() {
        let config = test_config();
        let (horizontal, vertical) = screen_field_degrees(&config);
        assert!((45.0..65.0).contains(&horizontal));
        assert!((25.0..45.0).contains(&vertical));
        assert!((8.0..80.0).contains(&ball_radius_pixels(&config, 3840)));
    }

    #[test]
    fn small_render_contains_both_code_colors_and_fixation_ink() {
        let config = test_config();
        let mut pixels = vec![0; 320 * 180];
        render_stimulus(
            &mut pixels,
            320,
            180,
            &config,
            FrameCode::new(77, 3),
            ball_pose(&config, 5.0),
            &RecoveryReadout::default(),
            config.render_hz,
        );
        let mut colors = pixels;
        colors.sort_unstable();
        colors.dedup();
        assert!(colors.len() >= 5);
    }

    #[test]
    fn recovery_readout_uses_a_robust_latency_median() {
        let mut readout = RecoveryReadout::default();
        for (offset, latency_ms) in [40_u64, 44, 42, 41, 43].into_iter().enumerate() {
            let commit_unix_ns = 1_000_000_000 + offset as u64 * 100_000_000;
            let code_index = 70 + offset as u64;
            let report = RecoveryReport {
                session_id: "test",
                sequence: 100 + offset as u64,
                host_arrival_unix_ns: commit_unix_ns + latency_ms * 1_000_000,
                recovered_code_index: code_index,
                score: 0.8,
                confidence_margin: 0.2,
                verified: true,
            };
            let transition = CodeTransition {
                code_index,
                presentation_index: offset as u64,
                commit_unix_ns,
            };
            assert_eq!(readout.accept(&report, transition), Some(latency_ms as f64));
        }
        assert_eq!(readout.estimate_ms, Some(42.0));
        assert!((latency_display_frames(42.0, 100.0) - 4.2).abs() < f64::EPSILON);

        let outlier = RecoveryReport {
            session_id: "test",
            sequence: 105,
            host_arrival_unix_ns: 1_800_000_000,
            recovered_code_index: 75,
            score: 0.8,
            confidence_margin: 0.2,
            verified: true,
        };
        let transition = CodeTransition {
            code_index: 75,
            presentation_index: 5,
            commit_unix_ns: 1_500_000_000,
        };
        assert_eq!(readout.accept(&outlier, transition), None);
        assert_eq!(readout.estimate_ms, Some(42.0));
        assert_eq!(readout.samples_ms.len(), 5);
        assert_eq!(readout.rejected_samples, 1);
    }

    #[test]
    fn checked_single_frame_is_visible_without_claiming_temporal_lock() {
        let mut readout = RecoveryReadout::default();
        let transition = CodeTransition {
            code_index: 90,
            presentation_index: 180,
            commit_unix_ns: 1_000_000_000,
        };
        let report = RecoveryReport {
            session_id: "test",
            sequence: 41,
            host_arrival_unix_ns: 1_123_000_000,
            recovered_code_index: 90,
            score: 0.72,
            confidence_margin: 0.31,
            verified: false,
        };
        assert_eq!(readout.accept(&report, transition), Some(123.0));
        assert!(matches!(readout.phase, RecoveryPhase::Noisy));
        assert_eq!(readout.last_observed_ms, Some(123.0));
        assert_eq!(readout.last_accepted_code_index, Some(90));
        assert!(readout.samples_ms.is_empty());
        assert_eq!(readout.estimate_ms, None);

        let verified = RecoveryReport {
            verified: true,
            ..report
        };
        assert_eq!(readout.accept(&verified, transition), Some(123.0));
        assert!(matches!(readout.phase, RecoveryPhase::Locked));
        assert_eq!(
            readout.samples_ms.iter().copied().collect::<Vec<_>>(),
            vec![123.0]
        );
        assert_eq!(readout.estimate_ms, Some(123.0));
    }
}
