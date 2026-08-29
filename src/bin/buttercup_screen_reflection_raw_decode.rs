#![recursion_limit = "256"]

#[path = "../screen_reflection_clock.rs"]
mod screen_reflection_clock;
#[path = "../screen_reflection_code.rs"]
mod screen_reflection_code;
#[path = "../screen_reflection_raw.rs"]
mod screen_reflection_raw;

use screen_reflection_clock::{
    analyze_whole_raw_roi, detect_optical_activity_onset,
    solve_optical_clock_in_delta_range_with_scheme, solve_optical_clock_with_scheme,
    ClockWitnessStream, OpticalClockFit, WholeRoiClockWitness,
};
use screen_reflection_code::{
    decode_soft_cells_constrained_with_scheme, decode_soft_cells_temporal_constrained_with_scheme,
    unwrap_counter_near_with_scheme, DecodeGeometry, GridTransform, OpticalCodeScheme,
    SpatialCodeLayout, GRID_COLUMNS, GRID_ROWS, PHYSICAL_CELL_COUNT,
};
use screen_reflection_raw::{
    code_aware_temporal_log_baseline_with_scheme, estimate_native_frame_translations,
    opponent_residual_cells, refine_reflection_quad_with_layout,
    sample_cell_spectra_interpolated_with_layout, score_quad_with_layout,
    search_reflection_quad_with_layout, CellSpectra, PackedRaw10, ProjectiveQuad, QuadFit,
    CFA_BAND_NAMES,
};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const RAW_DECODE_SCHEMA: &str = "buttercup-screen-reflection-raw-decode-v1";

#[derive(Clone, Debug)]
struct Config {
    bundle: PathBuf,
    manifest: PathBuf,
    output: Option<PathBuf>,
    eye: String,
    seed_center: Option<(f64, f64)>,
    seed_sensor_center: Option<(f64, f64)>,
    seed_radius: Option<f64>,
    seed_quad: Option<ProjectiveQuad>,
    locator_frames: usize,
    locator_span_ms: u64,
    counter_offset_radius: i16,
    track_counter_radius: u16,
    minimum_locator_score: f64,
    minimum_decode_margin: f64,
    maximum_hard_bit_errors: usize,
    initial_presentation: Option<u64>,
    maximum_frames: Option<usize>,
    whole_roi_clock: bool,
    host_phase_prior: bool,
}

fn usage() -> &'static str {
    "usage: buttercup_screen_reflection_raw_decode --bundle BUNDLE --manifest SESSION.jsonl [options]\n\
     \n\
     BUNDLE may be an extracted buttercup RAW-eye bundle directory or its POSIX\n\
     ustar file. Packed RAW10 is sampled directly at native ROI coordinates.\n\
     No preview, demosaic, resized image, or desktop capture enters the solve.\n\
     \n\
     --output PATH                 decoded JSONL; defaults to stdout\n\
     --eye right|left              subject eye [right]\n\
     --seed-center X,Y             optional native-ROI locator center\n\
     --seed-sensor-center X,Y      optional absolute sensor-space eye center\n\
     --seed-radius PX              optional locator search scale\n\
     --seed-quad X0,Y0,...,X3,Y3   optional TL,TR,BR,BL native-ROI quad\n\
     --locator-frames N            warmup observations [20]\n\
     --locator-span-ms N           warmup temporal span [1400]\n\
     --counter-offset-radius N     host/code phase search +/−N [24]\n\
     --track-counter-radius N      per-frame decode search +/−N [12]\n\
     --minimum-locator-score N     fail below temporal locator score [0.10]\n\
     --minimum-decode-margin N     accepted identity margin [0.035]\n\
     --maximum-hard-bit-errors N   accepted logical-bit errors [3]\n\
     --initial-presentation N      fallback for old bundles without host clock\n\
     --maximum-frames N            bound sequential decoding\n\
     --whole-roi-clock             anatomy-independent capture-wide clock solve\n\
     --host-phase-prior            bound absolute phase near packet-time counter\n\
     -h, --help                    show this text"
}

fn parse_pair(option: &str, text: &str) -> Result<(f64, f64), String> {
    let values = text
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("invalid {option}: {error}"))?;
    if values.len() != 2 || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{option} requires two finite comma-separated numbers"
        ));
    }
    Ok((values[0], values[1]))
}

fn parse_quad(text: &str) -> Result<ProjectiveQuad, String> {
    let values = text
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("invalid --seed-quad: {error}"))?;
    if values.len() != 8 || values.iter().any(|value| !value.is_finite()) {
        return Err("--seed-quad requires eight finite comma-separated numbers".to_string());
    }
    Ok(ProjectiveQuad {
        corners: [
            (values[0], values[1]),
            (values[2], values[3]),
            (values[4], values[5]),
            (values[6], values[7]),
        ],
    })
}

fn parse_config_from<I>(arguments: I) -> Result<Config, String>
where
    I: IntoIterator<Item = String>,
{
    let mut bundle = None;
    let mut manifest = None;
    let mut output = None;
    let mut eye = "subject-right".to_string();
    let mut seed_center = None;
    let mut seed_sensor_center = None;
    let mut seed_radius = None;
    let mut seed_quad = None;
    let mut locator_frames = 20usize;
    let mut locator_span_ms = 1_400u64;
    let mut counter_offset_radius = 24i16;
    let mut track_counter_radius = 12u16;
    let mut minimum_locator_score = 0.10f64;
    let mut minimum_decode_margin = 0.035f64;
    let mut maximum_hard_bit_errors = 3usize;
    let mut initial_presentation = None;
    let mut maximum_frames = None;
    let mut whole_roi_clock = false;
    let mut host_phase_prior = false;
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let value = |arguments: &mut I::IntoIter| {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} requires a value"))
        };
        match argument.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(value(&mut arguments)?)),
            "--manifest" => manifest = Some(PathBuf::from(value(&mut arguments)?)),
            "--output" => output = Some(PathBuf::from(value(&mut arguments)?)),
            "--eye" => {
                eye = match value(&mut arguments)?.as_str() {
                    "right" | "subject-right" => "subject-right".to_string(),
                    "left" | "subject-left" => "subject-left".to_string(),
                    other => return Err(format!("invalid --eye {other:?}; use right or left")),
                }
            }
            "--seed-center" => seed_center = Some(parse_pair(&argument, &value(&mut arguments)?)?),
            "--seed-sensor-center" => {
                seed_sensor_center = Some(parse_pair(&argument, &value(&mut arguments)?)?)
            }
            "--seed-radius" => {
                seed_radius = Some(
                    value(&mut arguments)?
                        .parse::<f64>()
                        .map_err(|error| format!("invalid --seed-radius: {error}"))?,
                )
            }
            "--seed-quad" => seed_quad = Some(parse_quad(&value(&mut arguments)?)?),
            "--locator-frames" => {
                locator_frames = value(&mut arguments)?
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --locator-frames: {error}"))?
            }
            "--locator-span-ms" => {
                locator_span_ms = value(&mut arguments)?
                    .parse::<u64>()
                    .map_err(|error| format!("invalid --locator-span-ms: {error}"))?
            }
            "--counter-offset-radius" => {
                counter_offset_radius = value(&mut arguments)?
                    .parse::<i16>()
                    .map_err(|error| format!("invalid --counter-offset-radius: {error}"))?
            }
            "--track-counter-radius" => {
                track_counter_radius = value(&mut arguments)?
                    .parse::<u16>()
                    .map_err(|error| format!("invalid --track-counter-radius: {error}"))?
            }
            "--minimum-locator-score" => {
                minimum_locator_score = value(&mut arguments)?
                    .parse::<f64>()
                    .map_err(|error| format!("invalid --minimum-locator-score: {error}"))?
            }
            "--minimum-decode-margin" => {
                minimum_decode_margin = value(&mut arguments)?
                    .parse::<f64>()
                    .map_err(|error| format!("invalid --minimum-decode-margin: {error}"))?
            }
            "--maximum-hard-bit-errors" => {
                maximum_hard_bit_errors = value(&mut arguments)?
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --maximum-hard-bit-errors: {error}"))?
            }
            "--initial-presentation" => {
                initial_presentation = Some(
                    value(&mut arguments)?
                        .parse::<u64>()
                        .map_err(|error| format!("invalid --initial-presentation: {error}"))?,
                )
            }
            "--maximum-frames" => {
                maximum_frames = Some(
                    value(&mut arguments)?
                        .parse::<usize>()
                        .map_err(|error| format!("invalid --maximum-frames: {error}"))?,
                )
            }
            "--whole-roi-clock" => whole_roi_clock = true,
            "--host-phase-prior" => host_phase_prior = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            _ => return Err(format!("unknown option {argument:?}\n{}", usage())),
        }
    }
    let config = Config {
        bundle: bundle.ok_or_else(|| format!("--bundle is required\n{}", usage()))?,
        manifest: manifest.ok_or_else(|| format!("--manifest is required\n{}", usage()))?,
        output,
        eye,
        seed_center,
        seed_sensor_center,
        seed_radius,
        seed_quad,
        locator_frames,
        locator_span_ms,
        counter_offset_radius,
        track_counter_radius,
        minimum_locator_score,
        minimum_decode_margin,
        maximum_hard_bit_errors,
        initial_presentation,
        maximum_frames,
        whole_roi_clock,
        host_phase_prior,
    };
    if !(5..=64).contains(&config.locator_frames) {
        return Err("--locator-frames must be in 5..=64".to_string());
    }
    if !(100..=10_000).contains(&config.locator_span_ms) {
        return Err("--locator-span-ms must be in 100..=10000".to_string());
    }
    if !(0..=256).contains(&config.counter_offset_radius) {
        return Err("--counter-offset-radius must be in 0..=256".to_string());
    }
    if config.track_counter_radius > 256 {
        return Err("--track-counter-radius must be in 0..=256".to_string());
    }
    if !(-1.0..=1.0).contains(&config.minimum_locator_score)
        || !(0.0..=1.0).contains(&config.minimum_decode_margin)
    {
        return Err("score/margin thresholds are out of range".to_string());
    }
    if config.maximum_hard_bit_errors > 8 {
        return Err("--maximum-hard-bit-errors must be in 0..=8".to_string());
    }
    if config
        .seed_radius
        .is_some_and(|radius| !radius.is_finite() || radius < 8.0)
    {
        return Err("--seed-radius must be finite and at least 8 pixels".to_string());
    }
    if config.seed_center.is_some() && config.seed_sensor_center.is_some() {
        return Err("--seed-center and --seed-sensor-center are mutually exclusive".to_string());
    }
    if config.maximum_frames == Some(0) {
        return Err("--maximum-frames must be nonzero".to_string());
    }
    Ok(config)
}

#[derive(Clone, Debug)]
struct Presentation {
    render_index: u64,
    code_index: u64,
    counter_mod: u16,
    commit_unix_ns: Option<u64>,
    ball_center_px: Option<[f64; 2]>,
    ball_center_normalized: Option<[f64; 2]>,
}

#[derive(Clone, Debug)]
struct ScreenManifest {
    session_id: String,
    session_tag: u8,
    code_hz: f64,
    code_layout: SpatialCodeLayout,
    code_scheme: OpticalCodeScheme,
    presentations: Vec<Presentation>,
}

fn fixed_pair(value: Option<&Value>) -> Option<[f64; 2]> {
    let values = value?.as_array()?;
    if values.len() != 2 {
        return None;
    }
    Some([values[0].as_f64()?, values[1].as_f64()?])
}

fn fixed_usize_pair(value: Option<&Value>) -> Option<[usize; 2]> {
    let values = value?.as_array()?;
    if values.len() != 2 {
        return None;
    }
    Some([
        usize::try_from(values[0].as_u64()?).ok()?,
        usize::try_from(values[1].as_u64()?).ok()?,
    ])
}

fn code_layout_from_session(value: &Value) -> Result<SpatialCodeLayout, String> {
    let Some(code) = value.get("code").and_then(Value::as_object) else {
        return Ok(SpatialCodeLayout::LEGACY);
    };
    let logical_grid = fixed_usize_pair(code.get("logical_grid"));
    if logical_grid.is_some_and(|grid| grid != [GRID_COLUMNS, GRID_ROWS]) {
        return Err(format!(
            "unsupported logical frame-code grid {:?}; expected {}x{}",
            logical_grid.unwrap(),
            GRID_COLUMNS,
            GRID_ROWS
        ));
    }
    let display_grid = fixed_usize_pair(code.get("grid"));
    let explicit_repeats = fixed_usize_pair(code.get("spatial_repeats"));
    let repeats = if let Some(repeats) = explicit_repeats {
        repeats
    } else if let Some(grid) = display_grid {
        if !grid[0].is_multiple_of(GRID_COLUMNS) || !grid[1].is_multiple_of(GRID_ROWS) {
            return Err(format!(
                "frame-code display grid {}x{} is not a complete tiling of {}x{}",
                grid[0], grid[1], GRID_COLUMNS, GRID_ROWS
            ));
        }
        [grid[0] / GRID_COLUMNS, grid[1] / GRID_ROWS]
    } else {
        [1, 1]
    };
    let layout = SpatialCodeLayout::new(repeats[0], repeats[1]).ok_or_else(|| {
        format!(
            "unsupported frame-code spatial repetition {}x{} (valid range 1..=4)",
            repeats[0], repeats[1]
        )
    })?;
    if let Some(grid) = display_grid {
        let expected = [layout.display_columns(), layout.display_rows()];
        if grid != expected {
            return Err(format!(
                "frame-code grid {}x{} disagrees with spatial repetition {}x{} (expected {}x{})",
                grid[0], grid[1], repeats[0], repeats[1], expected[0], expected[1]
            ));
        }
    }
    Ok(layout)
}

fn load_screen_manifest(path: &Path) -> Result<ScreenManifest, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut session_id = None;
    let mut session_tag = None;
    let mut presentation_hz = 60.0;
    let mut code_hz = None;
    let mut code_layout = SpatialCodeLayout::LEGACY;
    let mut code_scheme = OpticalCodeScheme::GrayCrcV1;
    let mut presentations = Vec::new();
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("read {}: {error}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|error| {
            format!("parse {} line {}: {error}", path.display(), line_index + 1)
        })?;
        match value.get("record_type").and_then(Value::as_str) {
            Some("session") => {
                code_layout = code_layout_from_session(&value)?;
                if let Some(symbol_sequence) = value
                    .get("code")
                    .and_then(|code| code.get("symbol_sequence"))
                    .and_then(Value::as_str)
                {
                    code_scheme =
                        OpticalCodeScheme::from_wire_name(symbol_sequence).ok_or_else(|| {
                            format!("unsupported optical code scheme {symbol_sequence:?}")
                        })?;
                }
                session_id = value
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                session_tag = value
                    .get("session_tag")
                    .and_then(Value::as_u64)
                    .and_then(|tag| u8::try_from(tag).ok());
                presentation_hz = value
                    .get("presentation_hz")
                    .or_else(|| value.get("target_hz"))
                    .and_then(Value::as_f64)
                    .unwrap_or(60.0);
                code_hz = value.get("code_hz").and_then(Value::as_f64);
            }
            Some("presentation") => {
                let render_index = value
                    .get("presentation_index")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| {
                        format!("manifest line {} lacks presentation_index", line_index + 1)
                    })?;
                let code_index = value
                    .get("code_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(render_index);
                let counter_mod = value
                    .get("counter_mod")
                    .and_then(Value::as_u64)
                    .and_then(|counter| u16::try_from(counter).ok())
                    .ok_or_else(|| format!("manifest line {} lacks counter_mod", line_index + 1))?;
                presentations.push(Presentation {
                    render_index,
                    code_index,
                    counter_mod,
                    commit_unix_ns: value.get("present_commit_unix_ns").and_then(Value::as_u64),
                    ball_center_px: fixed_pair(value.get("ball_center_px")),
                    ball_center_normalized: fixed_pair(value.get("ball_center_normalized")),
                });
            }
            _ => {}
        }
    }
    presentations.sort_by_key(|presentation| presentation.render_index);
    if presentations.is_empty() {
        return Err(format!("{} contains no presentations", path.display()));
    }
    if !presentation_hz.is_finite() || presentation_hz <= 0.0 {
        return Err("manifest presentation_hz/target_hz is invalid".to_string());
    }
    let code_hz = code_hz.unwrap_or(presentation_hz);
    if !code_hz.is_finite() || code_hz <= 0.0 || code_hz > presentation_hz {
        return Err("manifest code_hz is invalid".to_string());
    }
    Ok(ScreenManifest {
        session_id: session_id.ok_or_else(|| "manifest lacks session_id".to_string())?,
        session_tag: session_tag.ok_or_else(|| "manifest lacks session_tag".to_string())? & 0x0f,
        code_hz,
        code_layout,
        code_scheme,
        presentations,
    })
}

impl ScreenManifest {
    fn nearest_by_time(&self, timestamp_ns: u64) -> Option<&Presentation> {
        let timed = self
            .presentations
            .iter()
            .filter(|presentation| presentation.commit_unix_ns.is_some())
            .collect::<Vec<_>>();
        if timed.is_empty() {
            return None;
        }
        let insertion = timed.partition_point(|presentation| {
            presentation.commit_unix_ns.unwrap_or(0) <= timestamp_ns
        });
        let mut candidates = Vec::with_capacity(2);
        if insertion > 0 {
            candidates.push(timed[insertion - 1]);
        }
        if insertion < timed.len() {
            candidates.push(timed[insertion]);
        }
        candidates.into_iter().min_by_key(|presentation| {
            presentation
                .commit_unix_ns
                .unwrap_or(0)
                .abs_diff(timestamp_ns)
        })
    }

    fn by_code_index(&self, index: u64) -> Option<&Presentation> {
        self.presentations
            .iter()
            .find(|presentation| presentation.code_index == index)
    }

    fn by_code_index_near_time(
        &self,
        index: u64,
        timestamp_ns: Option<u64>,
    ) -> Option<&Presentation> {
        let matching = self
            .presentations
            .iter()
            .filter(|presentation| presentation.code_index == index);
        if let Some(timestamp_ns) = timestamp_ns {
            matching.min_by_key(|presentation| {
                presentation
                    .commit_unix_ns
                    .unwrap_or(0)
                    .abs_diff(timestamp_ns)
            })
        } else {
            matching.min_by_key(|presentation| presentation.render_index)
        }
    }
}

#[derive(Clone, Debug)]
struct FrameRecord {
    sequence: u64,
    timestamp_ns: u64,
    host_arrival_unix_ns: Option<u64>,
    eye_id: u32,
    label: String,
    sensor_x: u32,
    sensor_y: u32,
    width: usize,
    height: usize,
    stride: usize,
    stream: String,
    offset: u64,
    length: usize,
}

fn required_u64(value: &Value, field: &str, line: usize) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("frames.jsonl line {line} lacks {field}"))
}

fn load_frame_index(bytes: &[u8]) -> Result<Vec<FrameRecord>, String> {
    let text =
        std::str::from_utf8(bytes).map_err(|error| format!("frames.jsonl UTF-8: {error}"))?;
    let mut frames = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_number = line_index + 1;
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("frames.jsonl line {line_number}: {error}"))?;
        let width = usize::try_from(required_u64(&value, "width", line_number)?)
            .map_err(|_| "frame width overflows usize".to_string())?;
        let height = usize::try_from(required_u64(&value, "height", line_number)?)
            .map_err(|_| "frame height overflows usize".to_string())?;
        let stride = usize::try_from(required_u64(&value, "stride", line_number)?)
            .map_err(|_| "frame stride overflows usize".to_string())?;
        let length = usize::try_from(required_u64(&value, "length", line_number)?)
            .map_err(|_| "frame length overflows usize".to_string())?;
        if length != stride.saturating_mul(height) {
            return Err(format!(
                "frames.jsonl line {line_number} length {length} != stride*height {}",
                stride.saturating_mul(height)
            ));
        }
        if value.get("pixel_format").and_then(Value::as_str) != Some("RAW10_LE40_1X1") {
            return Err(format!(
                "frames.jsonl line {line_number} is not RAW10_LE40_1X1"
            ));
        }
        frames.push(FrameRecord {
            sequence: required_u64(&value, "sequence", line_number)?,
            timestamp_ns: required_u64(&value, "timestamp_ns", line_number)?,
            host_arrival_unix_ns: value.get("host_arrival_unix_ns").and_then(Value::as_u64),
            eye_id: u32::try_from(required_u64(&value, "eye_id", line_number)?)
                .map_err(|_| "eye_id overflows u32".to_string())?,
            label: value
                .get("label")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("frames.jsonl line {line_number} lacks label"))?
                .to_string(),
            sensor_x: u32::try_from(required_u64(&value, "sensor_x", line_number)?)
                .map_err(|_| "sensor_x overflows u32".to_string())?,
            sensor_y: u32::try_from(required_u64(&value, "sensor_y", line_number)?)
                .map_err(|_| "sensor_y overflows u32".to_string())?,
            width,
            height,
            stride,
            stream: value
                .get("stream")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("frames.jsonl line {line_number} lacks stream"))?
                .to_string(),
            offset: required_u64(&value, "offset", line_number)?,
            length,
        });
    }
    Ok(frames)
}

#[derive(Clone, Copy, Debug)]
struct TarEntry {
    data_offset: u64,
    size: u64,
}

#[derive(Debug)]
enum BundleSource {
    Directory(PathBuf),
    Tar {
        path: PathBuf,
        entries: HashMap<String, TarEntry>,
    },
}

fn tar_text(field: &[u8]) -> String {
    let end = field
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).trim().to_string()
}

fn tar_octal(field: &[u8]) -> Result<u64, String> {
    let text = String::from_utf8_lossy(field)
        .trim_matches(|character: char| character == '\0' || character.is_ascii_whitespace())
        .to_string();
    if text.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(&text, 8).map_err(|error| format!("invalid tar size {text:?}: {error}"))
}

impl BundleSource {
    fn open(path: &Path) -> Result<Self, String> {
        if path.is_dir() {
            return Ok(Self::Directory(path.to_path_buf()));
        }
        let mut file = File::open(path)
            .map_err(|error| format!("open RAW bundle {}: {error}", path.display()))?;
        let file_size = file
            .metadata()
            .map_err(|error| format!("stat {}: {error}", path.display()))?
            .len();
        let mut entries = HashMap::new();
        let mut header_offset = 0u64;
        while header_offset + 512 <= file_size {
            file.seek(SeekFrom::Start(header_offset))
                .map_err(|error| format!("seek tar: {error}"))?;
            let mut header = [0u8; 512];
            file.read_exact(&mut header)
                .map_err(|error| format!("read tar header: {error}"))?;
            if header.iter().all(|byte| *byte == 0) {
                break;
            }
            let name = tar_text(&header[..100]);
            let prefix = tar_text(&header[345..500]);
            let name = if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            };
            let size = tar_octal(&header[124..136])?;
            let data_offset = header_offset + 512;
            if data_offset.saturating_add(size) > file_size {
                return Err(format!("tar entry {name:?} extends beyond bundle"));
            }
            entries.insert(name, TarEntry { data_offset, size });
            header_offset = data_offset + size.div_ceil(512) * 512;
        }
        if entries.is_empty() {
            return Err(format!(
                "{} is not a populated POSIX tar bundle",
                path.display()
            ));
        }
        Ok(Self::Tar {
            path: path.to_path_buf(),
            entries,
        })
    }

    fn read_range(&self, name: &str, offset: u64, length: usize) -> Result<Vec<u8>, String> {
        let length_u64 = length as u64;
        let (path, absolute_offset, entry_size) = match self {
            Self::Directory(root) => {
                let path = root.join(name);
                let size = fs::metadata(&path)
                    .map_err(|error| format!("stat {}: {error}", path.display()))?
                    .len();
                (path, offset, size)
            }
            Self::Tar { path, entries } => {
                let entry = entries
                    .get(name)
                    .ok_or_else(|| format!("tar bundle lacks entry {name:?}"))?;
                (path.clone(), entry.data_offset + offset, entry.size)
            }
        };
        if offset.saturating_add(length_u64) > entry_size {
            return Err(format!(
                "read {name:?} range {offset}+{length} exceeds {entry_size} bytes"
            ));
        }
        let mut file =
            File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
        file.seek(SeekFrom::Start(absolute_offset))
            .map_err(|error| format!("seek {}: {error}", path.display()))?;
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        Ok(bytes)
    }

    fn read_entry(&self, name: &str) -> Result<Vec<u8>, String> {
        let size = match self {
            Self::Directory(root) => fs::metadata(root.join(name))
                .map_err(|error| format!("stat bundle entry {name:?}: {error}"))?
                .len(),
            Self::Tar { entries, .. } => {
                entries
                    .get(name)
                    .ok_or_else(|| format!("tar bundle lacks entry {name:?}"))?
                    .size
            }
        };
        let length = usize::try_from(size)
            .map_err(|_| format!("bundle entry {name:?} is too large for this host"))?;
        self.read_range(name, 0, length)
    }
}

#[derive(Debug)]
struct OwnedFrame {
    record: FrameRecord,
    payload: Vec<u8>,
}

impl OwnedFrame {
    fn raw(&self) -> Result<PackedRaw10<'_>, String> {
        PackedRaw10::new(
            &self.payload,
            self.record.width,
            self.record.height,
            self.record.stride,
            self.record.sensor_x,
            self.record.sensor_y,
        )
    }
}

fn load_owned(source: &BundleSource, record: &FrameRecord) -> Result<OwnedFrame, String> {
    Ok(OwnedFrame {
        payload: source.read_range(&record.stream, record.offset, record.length)?,
        record: record.clone(),
    })
}

fn evenly_spaced<T: Clone>(values: &[T], maximum: usize) -> Vec<T> {
    if values.len() <= maximum {
        return values.to_vec();
    }
    (0..maximum)
        .map(|index| {
            let position = index * (values.len() - 1) / (maximum - 1);
            values[position].clone()
        })
        .collect()
}

fn offset_index(index: u64, offset: i16) -> u64 {
    if offset >= 0 {
        index.saturating_add(offset as u64)
    } else {
        index.saturating_sub(u64::from(offset.unsigned_abs()))
    }
}

fn expected_presentation<'a>(
    manifest: &'a ScreenManifest,
    record: &FrameRecord,
    first_sensor_timestamp: u64,
    initial_presentation: Option<u64>,
) -> Option<&'a Presentation> {
    if let Some(host_time) = record.host_arrival_unix_ns {
        return manifest.nearest_by_time(host_time);
    }
    let initial = initial_presentation?;
    let elapsed = record.timestamp_ns.saturating_sub(first_sensor_timestamp) as f64 / 1.0e9;
    let index = initial.saturating_add((elapsed * manifest.code_hz).floor().max(0.0) as u64);
    manifest.by_code_index(index)
}

fn select_capture_frames(
    frames: &[FrameRecord],
    manifest: &ScreenManifest,
    config: &Config,
) -> Result<Vec<FrameRecord>, String> {
    let mut eye_frames = frames
        .iter()
        .filter(|frame| frame.label == config.eye)
        .cloned()
        .collect::<Vec<_>>();
    eye_frames.sort_by_key(|frame| frame.host_arrival_unix_ns.unwrap_or(frame.timestamp_ns));
    if eye_frames.is_empty() {
        return Err(format!("RAW bundle contains no {} frames", config.eye));
    }
    let first_commit = manifest
        .presentations
        .iter()
        .find_map(|presentation| presentation.commit_unix_ns);
    let last_commit = manifest
        .presentations
        .iter()
        .rev()
        .find_map(|presentation| presentation.commit_unix_ns);
    let has_host_clock = eye_frames
        .iter()
        .any(|frame| frame.host_arrival_unix_ns.is_some());
    if let (true, Some(first_commit), Some(last_commit)) =
        (has_host_clock, first_commit, last_commit)
    {
        let lower = first_commit.saturating_sub(500_000_000);
        let upper = last_commit.saturating_add(500_000_000);
        eye_frames.retain(|frame| {
            frame
                .host_arrival_unix_ns
                .is_some_and(|timestamp| (lower..=upper).contains(&timestamp))
        });
    } else if config.initial_presentation.is_none() {
        return Err(
            "bundle frames lack host_arrival_unix_ns; pass --initial-presentation for this older bundle"
                .to_string(),
        );
    }
    if eye_frames.len() < 5 {
        return Err("fewer than five eye frames overlap the screen manifest".to_string());
    }
    if let Some(maximum) = config.maximum_frames {
        eye_frames.truncate(maximum);
    }
    Ok(eye_frames)
}

fn select_locator_frames(
    frames: &[FrameRecord],
    manifest: &ScreenManifest,
    config: &Config,
) -> Vec<FrameRecord> {
    let span_ns = config.locator_span_ms.saturating_mul(1_000_000);
    let candidates = if let Some(first_commit) = manifest
        .presentations
        .iter()
        .find_map(|presentation| presentation.commit_unix_ns)
    {
        let lower = first_commit.saturating_sub(150_000_000);
        let upper = first_commit
            .saturating_add(span_ns)
            .saturating_add(250_000_000);
        frames
            .iter()
            .filter(|frame| {
                frame
                    .host_arrival_unix_ns
                    .is_some_and(|timestamp| (lower..=upper).contains(&timestamp))
            })
            .cloned()
            .collect::<Vec<_>>()
    } else {
        let first = frames[0].timestamp_ns;
        frames
            .iter()
            .take_while(|frame| frame.timestamp_ns.saturating_sub(first) <= span_ns)
            .cloned()
            .collect::<Vec<_>>()
    };
    evenly_spaced(&candidates, config.locator_frames)
}

fn default_seed(frames: &[OwnedFrame]) -> ((f64, f64), f64, &'static str) {
    let width = frames[0].record.width as f64;
    let height = frames[0].record.height as f64;
    (
        (width * 0.5, height * 0.5),
        width.min(height) * 0.30,
        "typed-roi-center/native-search",
    )
}

fn transform_name(transform: GridTransform) -> &'static str {
    match transform {
        GridTransform::Identity => "identity",
        GridTransform::MirrorHorizontal => "mirror-horizontal",
        GridTransform::MirrorVertical => "mirror-vertical",
        GridTransform::Rotate180 => "rotate-180",
    }
}

fn quad_json(quad: ProjectiveQuad) -> Value {
    json!(quad.corners.map(|point| [point.0, point.1]))
}

fn sensor_quad(local: ProjectiveQuad, record: &FrameRecord) -> ProjectiveQuad {
    local.translated(record.sensor_x as f64, record.sensor_y as f64)
}

fn roi_quad(sensor: ProjectiveQuad, record: &FrameRecord) -> ProjectiveQuad {
    sensor.translated(-(record.sensor_x as f64), -(record.sensor_y as f64))
}

fn advance_code_index_by_sensor_time(
    previous_index: u64,
    previous_sensor_timestamp_ns: u64,
    sensor_timestamp_ns: u64,
    code_hz: f64,
) -> u64 {
    let elapsed_ns = sensor_timestamp_ns.saturating_sub(previous_sensor_timestamp_ns);
    let advance = (elapsed_ns as f64 * code_hz / 1.0e9).round().max(0.0) as u64;
    previous_index.saturating_add(advance)
}

fn fuse_code_time_priors(host_index: u64, sensor_index: u64, radius: u16) -> (u64, bool) {
    let maximum_disagreement = u64::from(radius.clamp(1, 3));
    if host_index.abs_diff(sensor_index) <= maximum_disagreement {
        (sensor_index, false)
    } else {
        // Host arrival is not the identity measurement, but it is a bounded
        // wall-clock guard against accumulating one bad optical decode for an
        // entire run. The reflected code still chooses within this radius.
        (host_index, true)
    }
}

#[derive(Clone, Debug)]
struct TrackedEvaluation {
    quad: ProjectiveQuad,
    spectra: CellSpectra,
    cells: [f64; PHYSICAL_CELL_COUNT],
    score: f64,
    objective: f64,
}

#[derive(Clone, Copy)]
struct TrackingContext<'a> {
    baseline: &'a [[f64; 4]; PHYSICAL_CELL_COUNT],
    manifest: &'a ScreenManifest,
    code_layout: SpatialCodeLayout,
    expected_mod: u16,
    counter_radius: u16,
    transform: GridTransform,
    polarity: i8,
    predicted: ProjectiveQuad,
    search_scale: f64,
}

fn evaluate_tracking_quad(
    raw: PackedRaw10<'_>,
    quad: ProjectiveQuad,
    context: &TrackingContext<'_>,
) -> Option<TrackedEvaluation> {
    if !quad.plausible_in(raw.width, raw.height) {
        return None;
    }
    let spectra = sample_cell_spectra_interpolated_with_layout(raw, quad, 5, context.code_layout);
    let cells = opponent_residual_cells(&spectra, context.baseline, 1);
    let decoded = decode_soft_cells_constrained_with_scheme(
        &cells,
        context.manifest.session_tag,
        Some(context.expected_mod),
        context.counter_radius,
        DecodeGeometry {
            transform: context.transform,
            polarity: context.polarity,
        },
        context.manifest.code_scheme,
    )?;
    let center_error_px = {
        let first = quad.center();
        let second = context.predicted.center();
        (first.0 - second.0).hypot(first.1 - second.1)
    };
    let area_error = (quad.area() / context.predicted.area().max(1.0e-9))
        .ln()
        .abs();
    let allowed_step_px = (0.025 * context.search_scale).clamp(1.25, 3.0);
    // A single codeword has enough degrees of freedom to reward a wrong
    // translation over static iris texture. The temporal locator is the
    // strong geometry measurement; one frame may move it only when the code
    // gain clearly pays a physical inter-frame motion cost.
    let objective =
        decoded.score - 0.080 * (center_error_px / allowed_step_px).powi(2) - 0.40 * area_error;
    Some(TrackedEvaluation {
        quad,
        spectra,
        cells,
        score: decoded.score,
        objective,
    })
}

fn track_frame_quad(
    raw: PackedRaw10<'_>,
    context: TrackingContext<'_>,
) -> Option<TrackedEvaluation> {
    let mut best = evaluate_tracking_quad(raw, context.predicted, &context)?;
    for translation_step in
        [0.030, 0.015, 0.0075].map(|fraction| (fraction * context.search_scale).clamp(0.75, 3.0))
    {
        for axis in 0..2 {
            let base = best.quad;
            for direction in [-1.0, 1.0] {
                let candidate = if axis == 0 {
                    base.translated(direction * translation_step, 0.0)
                } else {
                    base.translated(0.0, direction * translation_step)
                };
                let Some(evaluation) = evaluate_tracking_quad(raw, candidate, &context) else {
                    continue;
                };
                if evaluation.objective > best.objective {
                    best = evaluation;
                }
            }
        }
    }
    Some(best)
}

fn reacquire_frame_quad(
    raw: PackedRaw10<'_>,
    context: TrackingContext<'_>,
) -> Option<TrackedEvaluation> {
    // A semantic ROI jump means the feature moved with the crop, so the old
    // absolute sensor coordinate is no longer a useful one-frame prior. Keep
    // the locator's projective shape, but search a bounded native-pixel grid
    // around its ROI-relative position. This remains a direct packed-RAW10
    // solve; no preview, resize, pyramid, or demosaic is constructed.
    let step = (0.10 * context.search_scale).clamp(4.0, 12.0);
    let radius = (0.45 * context.search_scale).clamp(16.0, 60.0);
    let steps = (radius / step).ceil().clamp(2.0, 6.0) as i32;
    let mut best: Option<TrackedEvaluation> = None;
    for y_step in -steps..=steps {
        for x_step in -steps..=steps {
            let candidate = context
                .predicted
                .translated(x_step as f64 * step, y_step as f64 * step);
            let Some(evaluation) = evaluate_tracking_quad(raw, candidate, &context) else {
                continue;
            };
            if best
                .as_ref()
                .is_none_or(|current| evaluation.score > current.score)
            {
                best = Some(evaluation);
            }
        }
    }
    let coarse = best?;
    track_frame_quad(
        raw,
        TrackingContext {
            predicted: coarse.quad,
            ..context
        },
    )
}

fn prefer_tracking_candidate(
    current: Option<(TrackedEvaluation, &'static str)>,
    candidate: Option<TrackedEvaluation>,
    transport: &'static str,
) -> Option<(TrackedEvaluation, &'static str)> {
    let Some(candidate) = candidate else {
        return current;
    };
    if current
        .as_ref()
        .is_none_or(|(best, _)| candidate.objective > best.objective)
    {
        Some((candidate, transport))
    } else {
        current
    }
}

fn create_output(path: Option<&Path>) -> Result<Box<dyn Write>, String> {
    match path {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|error| format!("create {}: {error}", parent.display()))?;
            }
            let file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(path)
                .map_err(|error| format!("create {}: {error}", path.display()))?;
            Ok(Box::new(BufWriter::new(file)))
        }
        None => Ok(Box::new(BufWriter::new(io::stdout()))),
    }
}

fn write_json_line(writer: &mut dyn Write, value: &Value) -> Result<(), String> {
    serde_json::to_writer(&mut *writer, value).map_err(|error| error.to_string())?;
    writer.write_all(b"\n").map_err(|error| error.to_string())
}

#[derive(Clone, Debug)]
struct WholeRoiAnalyzedFrame {
    record: FrameRecord,
    witness: WholeRoiClockWitness,
}

fn integer_median(mut values: Vec<i64>) -> Option<i64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    Some(values[values.len() / 2])
}

fn floating_median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    let middle = values.len().checked_sub(1)? / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle] + values[middle + 1]) * 0.5
    } else {
        values[middle]
    })
}

fn clock_fit_json(fit: &OpticalClockFit) -> Value {
    json!({
        "onset_index": fit.onset_index,
        "onset_score": fit.onset_score,
        "onset_runner_up_score": fit.onset_runner_up_score,
        "rate_hz": fit.rate_hz,
        "fractional_phase": fit.fractional_phase,
        "onset_counter_delta": fit.onset_counter_delta,
        "score": fit.score,
        "distinct_runner_up_score": fit.distinct_runner_up_score,
        "different_counter_delta_runner_up_score": fit.different_counter_delta_runner_up_score,
        "counter_phase_family_margin": fit.counter_phase_family_margin,
        "confidence_margin": fit.confidence_margin,
        "schedule_ensemble_size": fit.schedule_ensemble_size,
        "schedule_consensus_frames_75pct": fit.schedule_consensus_frames_75pct,
        "fit_frames": fit.fit_frames,
        "direct_witness_timestamps": fit.direct_witness_frames,
        "stream_geometry": fit.stream_geometry.iter().map(|(name, geometry)| json!({
            "stream": name,
            "grid_transform": transform_name(geometry.transform),
            "opponent_polarity": geometry.polarity,
            "score": geometry.score,
            "support_frames": geometry.support_frames
        })).collect::<Vec<_>>(),
        "negative_controls": {
            "best_wrong_session_tag_score": fit.wrong_session_tag_score,
            "reversed_sensor_time_score": fit.reversed_time_score,
            "spatial_cell_scramble_score": fit.spatial_scramble_score
        }
    })
}

fn run_whole_roi_clock(
    config: &Config,
    manifest: &ScreenManifest,
    source: &BundleSource,
    frame_index: &[FrameRecord],
) -> Result<(), String> {
    // This selection is deliberately independent of the host and screen
    // clocks. Every captured native ROI from either field is eligible to vote.
    let labels = frame_index
        .iter()
        .filter(|frame| frame.label == "subject-right" || frame.label == "subject-left")
        .map(|frame| frame.label.clone())
        .collect::<BTreeSet<_>>();
    if labels.is_empty() {
        return Err("RAW bundle contains no native subject ROI streams".to_string());
    }
    let mut timestamps_ns = frame_index
        .iter()
        .filter(|frame| labels.contains(&frame.label))
        .map(|frame| frame.timestamp_ns)
        .collect::<Vec<_>>();
    timestamps_ns.sort_unstable();
    timestamps_ns.dedup();
    if let Some(maximum) = config.maximum_frames {
        timestamps_ns.truncate(maximum);
    }
    if timestamps_ns.len() < 32 {
        return Err("whole-ROI clock solve requires at least 32 sensor timestamps".to_string());
    }
    let allowed_timestamps = timestamps_ns.iter().copied().collect::<BTreeSet<_>>();
    let mut records = frame_index
        .iter()
        .filter(|frame| {
            labels.contains(&frame.label) && allowed_timestamps.contains(&frame.timestamp_ns)
        })
        .cloned()
        .collect::<Vec<_>>();
    records.sort_by(|left, right| {
        left.timestamp_ns
            .cmp(&right.timestamp_ns)
            .then_with(|| left.label.cmp(&right.label))
    });
    let mut analyzed = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let owned = load_owned(source, record)?;
        let raw = owned.raw()?;
        let witness = analyze_whole_raw_roi(raw).ok_or_else(|| {
            format!(
                "whole-ROI native scan failed for {} sequence {}",
                record.label, record.sequence
            )
        })?;
        analyzed.push(WholeRoiAnalyzedFrame {
            record: record.clone(),
            witness,
        });
        if (index + 1).is_multiple_of(100) {
            eprintln!(
                "whole-ROI optical scan {}/{} native frames",
                index + 1,
                records.len()
            );
        }
    }
    let timestamp_index = timestamps_ns
        .iter()
        .enumerate()
        .map(|(index, timestamp)| (*timestamp, index))
        .collect::<HashMap<_, _>>();
    let mut streams = labels
        .iter()
        .map(|label| ClockWitnessStream {
            name: label.clone(),
            witnesses: vec![None; timestamps_ns.len()],
        })
        .collect::<Vec<_>>();
    let stream_index = streams
        .iter()
        .enumerate()
        .map(|(index, stream)| (stream.name.clone(), index))
        .collect::<HashMap<_, _>>();
    for frame in &analyzed {
        let timeline = timestamp_index[&frame.record.timestamp_ns];
        let stream = stream_index[&frame.record.label];
        let slot = &mut streams[stream].witnesses[timeline];
        if slot.as_ref().is_none_or(|current: &WholeRoiClockWitness| {
            frame.witness.proposal_score > current.proposal_score
        }) {
            *slot = Some(frame.witness.clone());
        }
    }
    for stream in &streams {
        let valid = stream
            .witnesses
            .iter()
            .enumerate()
            .filter_map(|(index, witness)| {
                witness
                    .as_ref()
                    .is_some_and(|value| value.valid)
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        eprintln!(
            "whole-ROI stream={} valid_witnesses={} first={:?} last={:?}",
            stream.name,
            valid.len(),
            valid.first(),
            valid.last()
        );
    }
    let (onset_index, onset_score, onset_runner_up_score) = detect_optical_activity_onset(&streams)
        .ok_or_else(|| {
            "whole-ROI sustained optical activity onset could not be established".to_string()
        })?;
    eprintln!(
        "whole-ROI selected optical-activity onset index={} score={:.6} pre-onset={:.6}",
        onset_index, onset_score, onset_runner_up_score
    );
    let maximum_code_index = manifest
        .presentations
        .iter()
        .map(|presentation| presentation.code_index)
        .max()
        .ok_or_else(|| "screen manifest contains no code indexes".to_string())?;
    let host_phase_bounds = if config.host_phase_prior {
        let onset_timestamp = timestamps_ns[onset_index];
        let center = analyzed
            .iter()
            .filter(|frame| frame.record.timestamp_ns == onset_timestamp)
            .filter_map(|frame| frame.record.host_arrival_unix_ns)
            .filter_map(|arrival| manifest.nearest_by_time(arrival))
            .filter_map(|presentation| i32::try_from(presentation.code_index).ok())
            .min()
            .ok_or_else(|| "host phase prior lacks an onset packet/manifest join".to_string())?;
        let radius = i32::from(config.counter_offset_radius);
        let bounds = (center.saturating_sub(radius), center);
        eprintln!(
            "whole-ROI bounded host phase prior delta={}..={} (optical score chooses within range)",
            bounds.0, bounds.1
        );
        Some(bounds)
    } else {
        None
    };
    let fit = host_phase_bounds
        .map_or_else(
            || {
                solve_optical_clock_with_scheme(
                    &timestamps_ns,
                    &streams,
                    onset_index,
                    onset_score,
                    onset_runner_up_score,
                    manifest.code_hz,
                    maximum_code_index,
                    manifest.session_tag,
                    manifest.code_scheme,
                )
            },
            |(minimum_delta, maximum_delta)| {
                solve_optical_clock_in_delta_range_with_scheme(
                    &timestamps_ns,
                    &streams,
                    onset_index,
                    onset_score,
                    onset_runner_up_score,
                    manifest.code_hz,
                    maximum_code_index,
                    manifest.session_tag,
                    minimum_delta,
                    maximum_delta,
                    manifest.code_scheme,
                )
            },
        )
        .ok_or_else(|| {
            "whole-ROI optical clock solve had insufficient direct witnesses".to_string()
        })?;

    // Without --host-phase-prior, host timestamps enter only after the
    // optical fit is immutable. With it, they select a bounded absolute phase
    // family; the optical evidence still chooses within that family and alone
    // determines geometry, rate, fractional phase, and acceptance.
    let onset_sensor_timestamp = timestamps_ns[onset_index];
    let protocol_seconds = (maximum_code_index + 1) as f64 / manifest.code_hz;
    let active_schedule_mask = timestamps_ns
        .iter()
        .map(|timestamp| {
            let seconds = (*timestamp as i128 - onset_sensor_timestamp as i128) as f64 / 1.0e9;
            (-0.25..=protocol_seconds + 0.10).contains(&seconds)
        })
        .collect::<Vec<_>>();
    let mut host_residual_ticks = Vec::new();
    let mut host_residual_ticks_by_stream = labels
        .iter()
        .map(|label| (label.clone(), Vec::<i64>::new()))
        .collect::<HashMap<_, _>>();
    for frame in &analyzed {
        let timeline = timestamp_index[&frame.record.timestamp_ns];
        let predicted = fit.predicted_indices[timeline];
        if !active_schedule_mask[timeline] || predicted < 0 || predicted > maximum_code_index as i32
        {
            continue;
        }
        let Some(host) = frame.record.host_arrival_unix_ns else {
            continue;
        };
        let Some(presentation) = manifest.nearest_by_time(host) else {
            continue;
        };
        let residual = presentation.code_index as i64 - i64::from(predicted);
        host_residual_ticks.push(residual);
        host_residual_ticks_by_stream
            .entry(frame.record.label.clone())
            .or_default()
            .push(residual);
    }
    let coarse_host_latency_ticks = integer_median(host_residual_ticks.clone());
    let mut host_exact = 0usize;
    let mut host_within_one = 0usize;
    if let Some(latency) = coarse_host_latency_ticks {
        for residual in &host_residual_ticks {
            let error = residual - latency;
            host_exact += usize::from(error == 0);
            host_within_one += usize::from(error.abs() <= 1);
        }
    }
    let per_stream_host_evaluation = labels
        .iter()
        .map(|label| {
            let residuals = host_residual_ticks_by_stream
                .get(label)
                .cloned()
                .unwrap_or_default();
            let latency = integer_median(residuals.clone());
            let exact = latency.map_or(0, |latency| {
                residuals
                    .iter()
                    .filter(|value| **value - latency == 0)
                    .count()
            });
            let within_one = latency.map_or(0, |latency| {
                residuals
                    .iter()
                    .filter(|value| (**value - latency).abs() <= 1)
                    .count()
            });
            json!({
                "stream": label,
                "median_packet_latency_code_ticks": latency,
                "samples": residuals.len(),
                "exact_after_median_latency": exact,
                "within_one_tick_after_median_latency": within_one,
                "within_one_tick_fraction": within_one as f64 / residuals.len().max(1) as f64
            })
        })
        .collect::<Vec<_>>();

    let active_timestamps = active_schedule_mask
        .iter()
        .filter(|active| **active)
        .count();
    let recovered_mask = (0..timestamps_ns.len())
        .map(|timeline| {
            active_schedule_mask[timeline]
                && (0..=maximum_code_index as i32).contains(&fit.predicted_indices[timeline])
                // The schedule ensemble deliberately has an odd number of
                // near-equivalent sub-tick fits. A strict majority is the
                // unique maximum-likelihood code identity for this exposure;
                // lower confidence is retained as an explicit diagnostic.
                && fit.schedule_consensus[timeline] > 0.50
        })
        .collect::<Vec<_>>();
    let recovered_active_timestamps = recovered_mask.iter().filter(|value| **value).count();
    let direct_active_timestamps = (0..timestamps_ns.len())
        .filter(|timeline| {
            recovered_mask[*timeline]
                && streams.iter().any(|stream| {
                    stream.witnesses[*timeline]
                        .as_ref()
                        .is_some_and(|witness| witness.valid)
                })
        })
        .count();
    let interpolated_active_timestamps = recovered_active_timestamps - direct_active_timestamps;
    let recovery_fraction = recovered_active_timestamps as f64 / active_timestamps.max(1) as f64;
    let exceeds_95_percent = recovery_fraction > 0.95;
    let strongest_control = fit
        .different_counter_delta_runner_up_score
        .max(fit.wrong_session_tag_score)
        .max(fit.reversed_time_score)
        .max(fit.spatial_scramble_score);
    let optical_lock_supported = fit.score > strongest_control
        && fit.onset_score > fit.onset_runner_up_score
        && fit.confidence_margin > 0.002;

    let display_commit_intervals_ns = manifest
        .presentations
        .windows(2)
        .filter_map(|pair| pair[1].commit_unix_ns?.checked_sub(pair[0].commit_unix_ns?))
        .filter(|interval| (5_000_000..=50_000_000).contains(interval))
        .map(|interval| interval as f64)
        .collect::<Vec<_>>();
    let effective_display_hz =
        floating_median(display_commit_intervals_ns).map(|period_ns| 1.0e9 / period_ns);
    let mut latency_samples_by_stream = labels
        .iter()
        .map(|label| (label.clone(), Vec::<f64>::new()))
        .collect::<HashMap<_, _>>();
    let mut direct_latency_samples_by_stream = labels
        .iter()
        .map(|label| (label.clone(), Vec::<f64>::new()))
        .collect::<HashMap<_, _>>();
    for frame in &analyzed {
        let timeline = timestamp_index[&frame.record.timestamp_ns];
        if !recovered_mask[timeline] {
            continue;
        }
        let predicted = fit.predicted_indices[timeline];
        let Some(code_index) = u64::try_from(predicted).ok() else {
            continue;
        };
        let Some(arrival) = frame.record.host_arrival_unix_ns else {
            continue;
        };
        let Some(commit) = manifest
            .by_code_index(code_index)
            .and_then(|presentation| presentation.commit_unix_ns)
        else {
            continue;
        };
        let Some(latency_ns) = arrival.checked_sub(commit) else {
            continue;
        };
        let latency_ms = latency_ns as f64 / 1.0e6;
        if !(0.0..=1_000.0).contains(&latency_ms) {
            continue;
        }
        latency_samples_by_stream
            .entry(frame.record.label.clone())
            .or_default()
            .push(latency_ms);
        if frame.witness.valid {
            direct_latency_samples_by_stream
                .entry(frame.record.label.clone())
                .or_default()
                .push(latency_ms);
        }
    }
    let latency_by_stream = labels
        .iter()
        .map(|label| {
            let samples = latency_samples_by_stream
                .get(label)
                .cloned()
                .unwrap_or_default();
            let direct = direct_latency_samples_by_stream
                .get(label)
                .cloned()
                .unwrap_or_default();
            let median_ms = floating_median(samples.clone());
            let direct_median_ms = floating_median(direct.clone());
            json!({
                "stream": label,
                "samples": samples.len(),
                "median_ms": median_ms,
                "direct_witness_samples": direct.len(),
                "direct_witness_median_ms": direct_median_ms,
                "direct_witness_median_display_frames": direct_median_ms.zip(effective_display_hz).map(|(latency, hz)| latency * hz / 1_000.0)
            })
        })
        .collect::<Vec<_>>();
    let estimated_primary_latency_ms = direct_latency_samples_by_stream
        .get("subject-right")
        .cloned()
        .filter(|samples| samples.len() >= 5)
        .and_then(floating_median);
    let estimated_primary_latency_display_frames = estimated_primary_latency_ms
        .zip(effective_display_hz)
        .map(|(latency, hz)| latency * hz / 1_000.0);
    let primary_latency_ms = optical_lock_supported
        .then_some(estimated_primary_latency_ms)
        .flatten();
    let primary_latency_display_frames = optical_lock_supported
        .then_some(estimated_primary_latency_display_frames)
        .flatten();

    let mut output = create_output(config.output.as_deref())?;
    write_json_line(
        output.as_mut(),
        &json!({
            "record_type": "whole-roi-clock-session",
            "schema": "buttercup-screen-reflection-whole-roi-clock-v1",
            "source_bundle": config.bundle,
            "source_manifest": config.manifest,
            "session_id": manifest.session_id,
            "session_tag": manifest.session_tag,
            "sensor_timestamps": timestamps_ns.len(),
            "raw_frames": analyzed.len(),
            "streams": labels,
            "protocol": {
                "code_hz": manifest.code_hz,
                "code_scheme": manifest.code_scheme.wire_name(),
                "maximum_code_index": maximum_code_index,
                "logical_grid": [GRID_COLUMNS, GRID_ROWS],
                "display_grid": [manifest.code_layout.display_columns(), manifest.code_layout.display_rows()]
            },
            "inference_contract": {
                "search_domain": "complete native RAW ROI in every stream",
                "geometry_semantics": "screen-reflection witness; no eye anatomy",
                "raw_sampling": "direct packed RAW10 2x2 physical Quad-Bayer blocks",
                "host_timestamp_used_for_inference": config.host_phase_prior,
                "host_timestamp_scope": if config.host_phase_prior {
                    "bounded absolute counter phase only; optical evidence chooses within range"
                } else {
                    "post-fit evaluation only"
                },
                "host_phase_delta_bounds": host_phase_bounds,
                "preview_used": false,
                "demosaic_used": false,
                "resize_used": false,
                "pupil_or_iris_model_used": false
            }
        }),
    )?;
    write_json_line(
        output.as_mut(),
        &json!({
            "record_type": "whole-roi-clock-fit",
            "optical_lock_supported": optical_lock_supported,
            "fit": clock_fit_json(&fit),
            "onset_sensor_timestamp_ns": timestamps_ns[onset_index],
            "onset_kind": "sustained-direct-optical-witness-cohort",
            "display_camera_latency": {
                "primary_stream": "subject-right",
                "median_ms": primary_latency_ms,
                "median_display_frames": primary_latency_display_frames,
                "effective_display_hz_from_commit_cadence": effective_display_hz,
                "per_stream": latency_by_stream,
                "measurement_interval": "first display commit of recovered code to complete camera packet arrival",
                "phase_prior": host_phase_bounds,
                "valid_only_when_optical_lock_supported": true
            },
            "evaluation_only_host_latency": {
                "median_packet_latency_code_ticks": coarse_host_latency_ticks,
                "samples": host_residual_ticks.len(),
                "exact_after_median_latency": host_exact,
                "within_one_tick_after_median_latency": host_within_one,
                "per_stream": per_stream_host_evaluation,
                "note": "host packet arrival is a coarse post-fit diagnostic and is not exposure identity"
            }
        }),
    )?;
    for frame in &analyzed {
        let timeline = timestamp_index[&frame.record.timestamp_ns];
        let predicted = fit.predicted_indices[timeline];
        let active =
            active_schedule_mask[timeline] && (0..=maximum_code_index as i32).contains(&predicted);
        let recovered = recovered_mask[timeline];
        let source = if recovered && frame.witness.valid {
            "direct-whole-roi-witness+sensor-clock"
        } else if recovered {
            "sensor-clock-interpolation"
        } else if active {
            "schedule-boundary-ambiguous"
        } else {
            "optically-inferred-stimulus-inactive"
        };
        let host_evaluation = frame
            .record
            .host_arrival_unix_ns
            .and_then(|host| manifest.nearest_by_time(host))
            .map(|presentation| {
                json!({
                    "manifest_code_nearest_packet_arrival": presentation.code_index,
                    "packet_minus_optical_ticks": presentation.code_index as i64 - i64::from(predicted),
                    "used_only_for_bounded_phase_family": config.host_phase_prior,
                    "excluded_from_geometry_rate_fractional_phase_and_acceptance": true
                })
            });
        let display_camera_latency_ms = recovered
            .then(|| {
                let code_index = u64::try_from(predicted).ok()?;
                let arrival = frame.record.host_arrival_unix_ns?;
                let commit = manifest.by_code_index(code_index)?.commit_unix_ns?;
                arrival
                    .checked_sub(commit)
                    .map(|latency_ns| latency_ns as f64 / 1.0e6)
                    .filter(|latency_ms| (0.0..=1_000.0).contains(latency_ms))
            })
            .flatten();
        write_json_line(
            output.as_mut(),
            &json!({
                "record_type": "whole-roi-clock-frame",
                "timeline_index": timeline,
                "sequence": frame.record.sequence,
                "sensor_timestamp_ns": frame.record.timestamp_ns,
                "host_arrival_unix_ns": frame.record.host_arrival_unix_ns,
                "stream": frame.record.label,
                "eye_id": frame.record.eye_id,
                "sensor_origin": [frame.record.sensor_x, frame.record.sensor_y],
                "native_size": [frame.record.width, frame.record.height],
                "status": if recovered { "recovered" } else if active { "ambiguous" } else { "stimulus-inactive" },
                "recovery_source": source,
                "code_index": recovered.then_some(predicted),
                "counter_mod": recovered.then_some(predicted.rem_euclid(2048)),
                "display_camera_latency_ms": display_camera_latency_ms,
                "schedule_consensus": fit.schedule_consensus[timeline],
                "direct_witness": {
                    "valid": frame.witness.valid,
                    "proposal_score": frame.witness.proposal_score,
                    "quad_roi_corners": quad_json(frame.witness.quad_roi),
                    "supported_logical_cells": frame.witness.supported_cells,
                    "repeat_agreement": frame.witness.repeat_agreement,
                    "component_area_carriers": frame.witness.component_area_carriers,
                    "luminance_median": frame.witness.luminance_median
                },
                "evaluation_only": host_evaluation
            }),
        )?;
    }
    write_json_line(
        output.as_mut(),
        &json!({
            "record_type": "whole-roi-clock-summary",
            "optical_lock_supported": optical_lock_supported,
            "active_sensor_timestamps": active_timestamps,
            "direct_active_timestamps": direct_active_timestamps,
            "interpolated_active_timestamps": interpolated_active_timestamps,
            "recovered_active_timestamps": recovered_active_timestamps,
            "recovery_fraction": recovery_fraction,
            "recovery_percent": recovery_fraction * 100.0,
            "exceeds_95_percent": exceeds_95_percent,
            "inactive_sensor_timestamps_classified": timestamps_ns.len() - active_timestamps,
            "raw_frames": analyzed.len(),
            "sensor_timestamps": timestamps_ns.len()
        }),
    )?;
    output.flush().map_err(|error| error.to_string())?;
    eprintln!(
        "whole-ROI optical clock score={:.4} control={:.4} active={} direct={} interpolated={} recovery={:.2}% supported={}",
        fit.score,
        strongest_control,
        active_timestamps,
        direct_active_timestamps,
        interpolated_active_timestamps,
        recovery_fraction * 100.0,
        optical_lock_supported
    );
    Ok(())
}

fn run(config: Config) -> Result<(), String> {
    let manifest = load_screen_manifest(&config.manifest)?;
    let source = BundleSource::open(&config.bundle)?;
    let frame_index = load_frame_index(&source.read_entry("frames.jsonl")?)?;
    if config.whole_roi_clock {
        return run_whole_roi_clock(&config, &manifest, &source, &frame_index);
    }
    let capture_frames = select_capture_frames(&frame_index, &manifest, &config)?;
    let locator_records = select_locator_frames(&capture_frames, &manifest, &config);
    if locator_records.len() < 5 {
        return Err(format!(
            "only {} frames fall in the locator warmup window",
            locator_records.len()
        ));
    }
    let first_sensor_timestamp = capture_frames[0].timestamp_ns;
    let locator_owned = locator_records
        .iter()
        .map(|record| load_owned(&source, record))
        .collect::<Result<Vec<_>, _>>()?;
    let locator_views = locator_owned
        .iter()
        .map(OwnedFrame::raw)
        .collect::<Result<Vec<_>, _>>()?;
    let base_presentations = locator_records
        .iter()
        .map(|record| {
            expected_presentation(
                &manifest,
                record,
                first_sensor_timestamp,
                config.initial_presentation,
            )
            .map(|presentation| presentation.counter_mod)
            .ok_or_else(|| {
                format!(
                    "cannot establish a screen-time prior for RAW sequence {}",
                    record.sequence
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let reference_record = &locator_owned[0].record;
    let (default_center, default_radius, default_method) = default_seed(&locator_owned);
    let sensor_seed_center = config.seed_sensor_center.map(|center| {
        (
            center.0 - reference_record.sensor_x as f64,
            center.1 - reference_record.sensor_y as f64,
        )
    });
    let seed_center = config
        .seed_center
        .or(sensor_seed_center)
        .unwrap_or(default_center);
    let seed_method = if config.seed_quad.is_some() {
        "explicit-projective-quad"
    } else if config.seed_sensor_center.is_some() {
        "qualified-eye-center/sensor-space"
    } else if config.seed_center.is_some() {
        "explicit-native-roi-center"
    } else {
        default_method
    };
    let seed_radius = config.seed_radius.unwrap_or(default_radius);
    let locator_translations =
        estimate_native_frame_translations(&locator_views, seed_center, seed_radius * 0.72);
    let minimum_offset = -config.counter_offset_radius;
    // Sensor exposure precedes RAW packet arrival on the same host clock, so
    // the reflected screen identity cannot be materially newer than the
    // manifest presentation nearest that later arrival. Permit one tick for
    // compositor callback/display-scan uncertainty, but reject the positive
    // multi-tick aliases that previously won on static iris texture.
    let maximum_offset = config.counter_offset_radius.min(1);
    let locator_fit: QuadFit = if let Some(quad) = config.seed_quad {
        let fit = score_quad_with_layout(
            &locator_views,
            &base_presentations,
            manifest.session_tag,
            quad,
            minimum_offset,
            maximum_offset,
            3,
            manifest.code_layout,
        )
        .ok_or_else(|| "the supplied --seed-quad is outside the native ROI".to_string())?;
        refine_reflection_quad_with_layout(
            &locator_views,
            &base_presentations,
            manifest.session_tag,
            fit,
            seed_radius,
            minimum_offset,
            maximum_offset,
            manifest.code_layout,
        )
    } else {
        search_reflection_quad_with_layout(
            &locator_views,
            &base_presentations,
            manifest.session_tag,
            seed_center,
            seed_radius,
            minimum_offset,
            maximum_offset,
            manifest.code_layout,
        )
        .ok_or_else(|| {
            "native temporal search found no plausible reflected screen quad".to_string()
        })?
    };
    if locator_fit.temporal.score < config.minimum_locator_score {
        return Err(format!(
            "reflected screen locator score {:.4} is below {:.4}; refusing a false frame identity",
            locator_fit.temporal.score, config.minimum_locator_score
        ));
    }

    let reference_sensor_quad = sensor_quad(locator_fit.quad, reference_record);
    let locator_lane = locator_fit.lane;
    let locator_observations = locator_views
        .iter()
        .map(|raw| {
            let local =
                reference_sensor_quad.translated(-(raw.sensor_x as f64), -(raw.sensor_y as f64));
            sample_cell_spectra_interpolated_with_layout(*raw, local, 5, manifest.code_layout)
        })
        .collect::<Vec<_>>();
    let baseline = code_aware_temporal_log_baseline_with_scheme(
        &locator_observations,
        &base_presentations,
        manifest.session_tag,
        locator_fit.temporal.counter_offset,
        locator_fit.temporal.transform,
        manifest.code_scheme,
    );

    let mut output = create_output(config.output.as_deref())?;
    write_json_line(
        output.as_mut(),
        &json!({
            "record_type": "raw-decode-session",
            "schema": RAW_DECODE_SCHEMA,
            "source_bundle": config.bundle,
            "source_manifest": config.manifest,
            "session_id": manifest.session_id,
            "session_tag": manifest.session_tag,
            "code_layout": {
                "logical_grid": [GRID_COLUMNS, GRID_ROWS],
                "spatial_repeats": [
                    manifest.code_layout.repeat_columns,
                    manifest.code_layout.repeat_rows
                ],
                "display_grid": [
                    manifest.code_layout.display_columns(),
                    manifest.code_layout.display_rows()
                ],
                "aggregation": "native samples pooled by canonical logical cell"
            },
            "eye": config.eye,
            "native_contract": {
                "pixel_format": "RAW10_LE40_1X1",
                "sampling": "direct packed ten-bit lanes",
                "coordinate_space": "complete native ROI plus explicit absolute sensor origin",
                "cfa_bands": CFA_BAND_NAMES,
                "preview_used": false,
                "demosaic_used": false,
                "resize_used": false
            },
            "host_clock_note": "host_arrival_unix_ns is a coarse join only; reflected code is authoritative"
            ,"acceptance_gate": {
                "minimum_decode_margin": config.minimum_decode_margin,
                "maximum_hard_bit_errors": config.maximum_hard_bit_errors
            }
        }),
    )?;
    write_json_line(
        output.as_mut(),
        &json!({
            "record_type": "reflection-locator",
            "status": "accepted",
            "reference_sequence": reference_record.sequence,
            "reference_sensor_origin": [reference_record.sensor_x, reference_record.sensor_y],
            "locator_sequences": locator_records.iter().map(|frame| frame.sequence).collect::<Vec<_>>(),
            "seed": {
                "method": seed_method,
                "center_roi": [seed_center.0, seed_center.1],
                "center_sensor": config.seed_sensor_center.map(|point| [point.0, point.1]),
                "radius_px": seed_radius
            },
            "quad_roi_corners": quad_json(locator_fit.quad),
            "quad_sensor_corners": quad_json(reference_sensor_quad),
            "score": locator_fit.temporal.score,
            "runner_up_score": locator_fit.temporal.runner_up_score,
            "confidence_margin": locator_fit.temporal.confidence_margin,
            "counter_offset": locator_fit.temporal.counter_offset,
            "grid_transform": transform_name(locator_fit.temporal.transform),
            "opponent_polarity": locator_fit.temporal.polarity,
            "band_correlations": locator_fit.temporal.band_correlations,
            "support_fraction": locator_fit.temporal.support_fraction,
            "native_registration": locator_translations.iter().map(|transport| json!({
                "cumulative_px": [transport.cumulative.0, transport.cumulative.1],
                "step_px": [transport.step.0, transport.step.1],
                "support": transport.support,
                "residual_px": if transport.residual.is_finite() {
                    Some(transport.residual)
                } else {
                    None
                }
            })).collect::<Vec<_>>(),
            "lane_comb": locator_lane.map(|lane| json!({
                "score": lane.score,
                "best_proposal_score": locator_fit.best_lane_proposal_score,
                "grid_transform": transform_name(lane.transform),
                "opponent_activity": lane.opponent_activity,
                "complementary_agreement": lane.complementary_agreement,
                "repeat_agreement": lane.repeat_agreement,
                "common_mode_rejection": lane.common_mode_rejection,
                "transitions": lane.transitions
            }))
        }),
    )?;

    // Camera sequence numbers restart at zero after each coarse acquisition
    // and fine-ROI reprogram.  The locator boundary is one unique capture
    // identity, not a minimum counter value: filtering every later epoch by
    // `sequence >= first_locator_sequence` silently discarded the first 42
    // native RAW frames after both observed reacquisitions.
    let first_locator_capture_index = capture_frames
        .iter()
        .position(|record| {
            record.timestamp_ns == locator_records[0].timestamp_ns
                && record.eye_id == locator_records[0].eye_id
                && record.offset == locator_records[0].offset
        })
        .ok_or_else(|| "first locator frame is absent from the capture timeline".to_string())?;
    let mut prior_sensor_quad = reference_sensor_quad;
    let mut prior_roi_quad = locator_fit.quad;
    let reference_roi_quad = locator_fit.quad;
    let mut previous_accepted: Option<(u64, u64, u16, [f64; PHYSICAL_CELL_COUNT])> = None;
    let mut previous_record_geometry: Option<(u64, u32, u32)> = None;
    let mut decoder_segment_id = 0u64;
    let mut decoder_segment_start_reason = Some("initial-locator");
    let mut reacquisition_active = false;
    let mut consecutive_ambiguous = 0usize;
    let mut accepted = 0usize;
    let mut ambiguous = 0usize;
    let mut lost = 0usize;
    let mut decoded_count = 0usize;
    for record in capture_frames.iter().skip(first_locator_capture_index) {
        let owned = load_owned(&source, record)?;
        let raw = owned.raw()?;
        let Some(coarse_presentation) = expected_presentation(
            &manifest,
            record,
            first_sensor_timestamp,
            config.initial_presentation,
        ) else {
            continue;
        };
        let (missing_sequences, sequence_reset, sensor_origin_step_px) = previous_record_geometry
            .map(|(sequence, sensor_x, sensor_y)| {
                (
                    record.sequence.saturating_sub(sequence).saturating_sub(1),
                    record.sequence <= sequence,
                    f64::from(
                        record
                            .sensor_x
                            .abs_diff(sensor_x)
                            .max(record.sensor_y.abs_diff(sensor_y)),
                    ),
                )
            })
            .unwrap_or((0, false, 0.0));
        let hard_boundary_reason = match (
            sequence_reset,
            missing_sequences > 1,
            sensor_origin_step_px > 128.0,
        ) {
            (true, _, true) => Some("camera-sequence-reset-and-sensor-roi-reacquisition"),
            (true, _, false) => Some("camera-sequence-reset"),
            (false, true, true) => Some("camera-gap-and-sensor-roi-reacquisition"),
            (false, true, false) => Some("camera-sequence-gap"),
            (false, false, true) => Some("sensor-roi-reacquisition"),
            (false, false, false) => None,
        };
        if let Some(reason) = hard_boundary_reason {
            if !reacquisition_active {
                decoder_segment_id = decoder_segment_id.saturating_add(1);
            }
            decoder_segment_start_reason = Some(reason);
            reacquisition_active = true;
            consecutive_ambiguous = 0;
            previous_accepted = None;
        }
        previous_record_geometry = Some((record.sequence, record.sensor_x, record.sensor_y));
        let host_expected_index = offset_index(
            coarse_presentation.code_index,
            locator_fit.temporal.counter_offset,
        );
        let sensor_expected_index = previous_accepted
            .as_ref()
            .map(|(previous_index, previous_timestamp_ns, _, _)| {
                advance_code_index_by_sensor_time(
                    *previous_index,
                    *previous_timestamp_ns,
                    record.timestamp_ns,
                    manifest.code_hz,
                )
            })
            .unwrap_or(host_expected_index);
        let temporal_radius = if previous_accepted.is_some() {
            config.track_counter_radius.clamp(1, 3)
        } else {
            config.track_counter_radius
        };
        let (temporal_expected_index, temporal_prior_reset) =
            fuse_code_time_priors(host_expected_index, sensor_expected_index, temporal_radius);
        let expected_mod = (temporal_expected_index & 2047) as u16;
        let context = |predicted| TrackingContext {
            baseline: &baseline,
            manifest: &manifest,
            code_layout: manifest.code_layout,
            expected_mod,
            counter_radius: temporal_radius,
            transform: locator_fit.temporal.transform,
            polarity: locator_fit.temporal.polarity,
            predicted,
            search_scale: seed_radius,
        };
        let mut tracking_choice = if reacquisition_active {
            prefer_tracking_candidate(
                None,
                reacquire_frame_quad(raw, context(reference_roi_quad)),
                "bounded-roi-relocalization",
            )
        } else {
            let sensor_rigid = track_frame_quad(raw, context(roi_quad(prior_sensor_quad, record)));
            let mut choice = prefer_tracking_candidate(None, sensor_rigid, "sensor-rigid");
            if sensor_origin_step_px > 0.0 {
                let roi_rigid = track_frame_quad(raw, context(prior_roi_quad));
                choice = prefer_tracking_candidate(choice, roi_rigid, "roi-rigid/crop-follow");
            }
            choice
        };
        if tracking_choice.is_none() && !reacquisition_active {
            decoder_segment_id = decoder_segment_id.saturating_add(1);
            decoder_segment_start_reason = Some("projective-lock-left-roi");
            reacquisition_active = true;
            consecutive_ambiguous = 0;
            previous_accepted = None;
            tracking_choice = prefer_tracking_candidate(
                None,
                reacquire_frame_quad(raw, context(reference_roi_quad)),
                "bounded-roi-relocalization",
            );
        }
        decoded_count += 1;
        let Some((tracking, tracking_transport)) = tracking_choice else {
            lost += 1;
            write_json_line(
                output.as_mut(),
                &json!({
                    "record_type": "raw-decode",
                    "capture_index": decoded_count - 1,
                    "sequence": record.sequence,
                    "sensor_timestamp_ns": record.timestamp_ns,
                    "host_arrival_unix_ns": record.host_arrival_unix_ns,
                    "eye_id": record.eye_id,
                    "sensor_origin": [record.sensor_x, record.sensor_y],
                    "native_size": [record.width, record.height],
                    "status": "lost",
                    "loss_reason": "no plausible projective quad in the native ROI",
                    "decoder_segment_id": decoder_segment_id,
                    "decoder_segment_start_reason": decoder_segment_start_reason,
                    "reacquisition_active": true,
                    "missing_sequences_before": missing_sequences,
                    "sequence_reset_before": sequence_reset,
                    "sensor_origin_step_before_px": sensor_origin_step_px,
                    "expected_presentation": temporal_expected_index,
                    "expected_code_index": temporal_expected_index,
                    "host_expected_code_index": host_expected_index,
                    "sensor_expected_code_index": sensor_expected_index,
                    "temporal_prior_reset_to_host": temporal_prior_reset,
                    "temporal_prior_reset_to_segment": true,
                    "temporal_search_radius": temporal_radius,
                    "confidence_margin": 0.0,
                    "hard_bit_errors": Value::Null,
                    "grid_transform": transform_name(locator_fit.temporal.transform),
                    "opponent_polarity": locator_fit.temporal.polarity,
                }),
            )?;
            continue;
        };
        let row_decoder_segment_id = decoder_segment_id;
        let row_segment_start_reason = decoder_segment_start_reason;
        let row_reacquisition_active = reacquisition_active;
        let decoded = if let Some((_, _, previous_counter, previous_cells)) =
            previous_accepted.as_ref().filter(|_| !temporal_prior_reset)
        {
            decode_soft_cells_temporal_constrained_with_scheme(
                &tracking.cells,
                previous_cells,
                *previous_counter,
                manifest.session_tag,
                Some(expected_mod),
                temporal_radius,
                DecodeGeometry {
                    transform: locator_fit.temporal.transform,
                    polarity: locator_fit.temporal.polarity,
                },
                manifest.code_scheme,
            )
        } else {
            decode_soft_cells_constrained_with_scheme(
                &tracking.cells,
                manifest.session_tag,
                Some(expected_mod),
                temporal_radius,
                DecodeGeometry {
                    transform: locator_fit.temporal.transform,
                    polarity: locator_fit.temporal.polarity,
                },
                manifest.code_scheme,
            )
        }
        .ok_or_else(|| format!("no frame-code candidates at sequence {}", record.sequence))?;
        let presentation_index = unwrap_counter_near_with_scheme(
            decoded.counter_mod,
            temporal_expected_index,
            manifest.code_scheme,
        );
        let checked_symbol_errors_ok = if manifest.code_scheme == OpticalCodeScheme::ReedMullerV3 {
            decoded.hard_bit_distance <= manifest.code_scheme.correctable_logical_bit_errors()
        } else {
            decoded.hard_bit_errors <= config.maximum_hard_bit_errors
        };
        let status = if decoded.confidence_margin >= config.minimum_decode_margin
            && checked_symbol_errors_ok
        {
            accepted += 1;
            consecutive_ambiguous = 0;
            reacquisition_active = false;
            previous_accepted = Some((
                presentation_index,
                record.timestamp_ns,
                decoded.counter_mod,
                tracking.cells,
            ));
            prior_sensor_quad = sensor_quad(tracking.quad, record);
            prior_roi_quad = tracking.quad;
            decoder_segment_start_reason = None;
            "accepted"
        } else {
            ambiguous += 1;
            consecutive_ambiguous = consecutive_ambiguous.saturating_add(1);
            // A moderately supported spatial lock may carry geometry through
            // an LCD transition, but an ambiguous code never updates temporal
            // identity state.
            if tracking.score > 0.10 {
                prior_sensor_quad = sensor_quad(tracking.quad, record);
                prior_roi_quad = tracking.quad;
            }
            if consecutive_ambiguous >= 2 && !reacquisition_active {
                // LCD transition ambiguity is a temporal boundary, not proof
                // that the reflected quad moved. Reset identity history while
                // retaining the supported geometry; broad spatial search is
                // reserved for an actual crop/sequence discontinuity or a
                // quad that leaves the native ROI.
                consecutive_ambiguous = 0;
                previous_accepted = None;
            }
            "ambiguous"
        };
        let matched =
            manifest.by_code_index_near_time(presentation_index, record.host_arrival_unix_ns);
        write_json_line(
            output.as_mut(),
            &json!({
                "record_type": "raw-decode",
                "capture_index": decoded_count - 1,
                "sequence": record.sequence,
                "sensor_timestamp_ns": record.timestamp_ns,
                "host_arrival_unix_ns": record.host_arrival_unix_ns,
                "eye_id": record.eye_id,
                "sensor_origin": [record.sensor_x, record.sensor_y],
                "native_size": [record.width, record.height],
                "status": status,
                "decoder_segment_id": row_decoder_segment_id,
                "decoder_segment_start_reason": row_segment_start_reason,
                "reacquisition_active": row_reacquisition_active,
                "missing_sequences_before": missing_sequences,
                "sequence_reset_before": sequence_reset,
                "sensor_origin_step_before_px": sensor_origin_step_px,
                "expected_presentation": temporal_expected_index,
                "expected_code_index": temporal_expected_index,
                "host_expected_code_index": host_expected_index,
                "sensor_expected_code_index": sensor_expected_index,
                "temporal_prior_reset_to_host": temporal_prior_reset,
                "temporal_prior_reset_to_segment": row_decoder_segment_id > 0
                    && row_segment_start_reason.is_some(),
                "temporal_search_radius": temporal_radius,
                "presentation_index": presentation_index,
                "code_index": presentation_index,
                "counter_mod": decoded.counter_mod,
                "score": decoded.score,
                "runner_up_score": decoded.runner_up_score,
                "confidence_margin": decoded.confidence_margin,
                "hard_bit_errors": decoded.hard_bit_errors,
                "hard_bit_distance": decoded.hard_bit_distance,
                "grid_transform": transform_name(decoded.transform),
                "opponent_polarity": decoded.polarity,
                "tracking_score": tracking.score,
                "tracking_objective": tracking.objective,
                "tracking_transport": tracking_transport,
                "quad_roi_corners": quad_json(tracking.quad),
                "quad_sensor_corners": quad_json(sensor_quad(tracking.quad, record)),
                "cell_support_fraction": tracking.spectra.support_fraction(),
                // Leakage-free optical evidence retained for capture-wide
                // schedule fitting. These are direct per-cell log means for
                // R/G1/G2/B before the host/code-aware baseline is applied.
                "raw_cfa_log_cells": tracking.spectra.log_values(),
                "opponent_cells": tracking.cells,
                "manifest_match": matched.map(|presentation| json!({
                    "render_presentation_index": presentation.render_index,
                    "code_index": presentation.code_index,
                    "counter_mod": presentation.counter_mod,
                    "present_commit_unix_ns": presentation.commit_unix_ns,
                    "ball_center_px": presentation.ball_center_px,
                    "ball_center_normalized": presentation.ball_center_normalized
                }))
            }),
        )?;
    }
    output.flush().map_err(|error| error.to_string())?;
    eprintln!(
        "RAW reflection decode locator={:.4} frames={} accepted={} ambiguous={} lost={} segments={} session={}",
        locator_fit.temporal.score,
        decoded_count,
        accepted,
        ambiguous,
        lost,
        decoder_segment_id + 1,
        manifest.session_id
    );
    Ok(())
}

fn main() {
    if let Err(error) = parse_config_from(env::args().skip(1)).and_then(run) {
        eprintln!("buttercup_screen_reflection_raw_decode: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use screen_reflection_code::{FrameCode, GRID_COLUMNS, GRID_ROWS};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn sensor_clock_advances_a_confident_code_prior_without_frame_jumps() {
        assert_eq!(
            advance_code_index_by_sensor_time(100, 5_000_000_000, 5_100_000_000, 30.0),
            103
        );
        assert_eq!(
            advance_code_index_by_sensor_time(103, 5_100_000_000, 5_196_000_000, 30.0),
            106
        );
        assert_eq!(
            advance_code_index_by_sensor_time(106, 5_196_000_000, 5_196_000_000, 30.0),
            106
        );
        assert_eq!(fuse_code_time_priors(106, 108, 3), (108, false));
        assert_eq!(fuse_code_time_priors(106, 142, 3), (106, true));
    }

    #[test]
    fn manifest_keeps_fast_render_commits_on_their_slower_code_clock() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "buttercup-screen-clock-manifest-{}-{nonce}.jsonl",
            std::process::id()
        ));
        let base = 1_800_000_000_000_000_000u64;
        let mut text = format!(
            "{}\n",
            json!({
                "record_type": "session",
                "session_id": "decoupled-clock-test",
                "session_tag": 3,
                "presentation_hz": 240.0,
                "code_hz": 30.0
            })
        );
        for render_index in 0..24u64 {
            let code_index = render_index / 8;
            text.push_str(&format!(
                "{}\n",
                json!({
                    "record_type": "presentation",
                    "presentation_index": render_index,
                    "code_index": code_index,
                    "counter_mod": code_index,
                    "present_commit_unix_ns": base + render_index * 4_166_667,
                    "ball_center_px": [100.0 + render_index as f64, 200.0]
                })
            ));
        }
        fs::write(&path, text).unwrap();
        let manifest = load_screen_manifest(&path).unwrap();
        assert!((manifest.code_hz - 30.0).abs() < f64::EPSILON);
        assert_eq!(manifest.code_layout, SpatialCodeLayout::LEGACY);
        assert_eq!(
            manifest
                .nearest_by_time(base + 38_000_000)
                .unwrap()
                .code_index,
            1
        );
        let matched = manifest
            .by_code_index_near_time(1, Some(base + 58_000_000))
            .unwrap();
        assert_eq!(matched.render_index, 14);
        assert_eq!(matched.ball_center_px, Some([114.0, 200.0]));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn manifest_selects_dense_repetition_and_rejects_inconsistent_grid_metadata() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "buttercup-screen-dense-manifest-{}-{nonce}.jsonl",
            std::process::id()
        ));
        let session = |grid: [usize; 2], repeats: [usize; 2]| {
            format!(
                "{}\n{}\n",
                json!({
                    "record_type": "session",
                    "session_id": "dense-layout-test",
                    "session_tag": 4,
                    "target_hz": 30.0,
                    "code": {
                        "grid": grid,
                        "logical_grid": [GRID_COLUMNS, GRID_ROWS],
                        "spatial_repeats": repeats,
                        "repeat_layout": "complete-logical-grid-tiles"
                    }
                }),
                json!({
                    "record_type": "presentation",
                    "presentation_index": 0,
                    "counter_mod": 0
                })
            )
        };
        fs::write(&path, session([16, 8], [2, 2])).unwrap();
        let manifest = load_screen_manifest(&path).unwrap();
        assert_eq!(manifest.code_layout, SpatialCodeLayout::CURRENT);
        assert_eq!(manifest.code_layout.display_columns(), 16);
        assert_eq!(manifest.code_layout.display_rows(), 8);

        fs::write(&path, session([16, 8], [1, 1])).unwrap();
        let error = load_screen_manifest(&path).unwrap_err();
        assert!(
            error.contains("disagrees with spatial repetition"),
            "{error}"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn options_normalize_eye_and_validate_native_seed() {
        let config = parse_config_from([
            "--bundle".into(),
            "capture.tar".into(),
            "--manifest".into(),
            "screen.jsonl".into(),
            "--eye".into(),
            "left".into(),
            "--seed-center".into(),
            "190.5,127.25".into(),
            "--seed-radius".into(),
            "72".into(),
            "--whole-roi-clock".into(),
        ])
        .unwrap();
        assert_eq!(config.eye, "subject-left");
        assert_eq!(config.seed_center, Some((190.5, 127.25)));
        assert_eq!(config.seed_radius, Some(72.0));
        assert!(config.whole_roi_clock);
        assert!(parse_config_from(Vec::<String>::new()).is_err());
    }

    #[test]
    fn tar_fields_and_presentation_offsets_are_bounded() {
        assert_eq!(tar_octal(b"00000000123\0").unwrap(), 0o123);
        assert_eq!(offset_index(2, -7), 0);
        assert_eq!(offset_index(2, 7), 9);
    }

    #[test]
    fn evenly_spaced_keeps_endpoints() {
        let selected = evenly_spaced(&(0..101).collect::<Vec<_>>(), 6);
        assert_eq!(selected, vec![0, 20, 40, 60, 80, 100]);
    }

    fn pack_raw10(values: &[u16], width: usize, height: usize) -> Vec<u8> {
        let stride = width / 4 * 5;
        let mut packed = vec![0u8; stride * height];
        for y in 0..height {
            for group in 0..width / 4 {
                let mut word = 0u64;
                for lane in 0..4 {
                    word |= u64::from(values[y * width + group * 4 + lane] & 0x03ff) << (lane * 10);
                }
                for byte in 0..5 {
                    packed[y * stride + group * 5 + byte] = (word >> (byte * 8)) as u8;
                }
            }
        }
        packed
    }

    fn synthetic_raw_frame(
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        quad: ProjectiveQuad,
        counter: u16,
        session_tag: u8,
        frame: usize,
        layout: SpatialCodeLayout,
    ) -> Vec<u8> {
        let signs = FrameCode::from_counter_mod(counter, session_tag).physical_signs();
        let bases = [470.0, 590.0, 575.0, 420.0];
        let amplitudes = [27.0, -15.0, -14.0, 43.0];
        let exposure = 0.94 + frame as f64 * 0.006;
        let mut values = vec![0u16; width * height];
        for y in 0..height {
            for x in 0..width {
                let phase_x = (sensor_x as usize + x) & 3;
                let phase_y = (sensor_y as usize + y) & 3;
                let band = match (phase_y < 2, phase_x < 2) {
                    (true, true) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                };
                let mut value =
                    bases[band] + 13.0 * (x as f64 * 0.071).sin() + 9.0 * (y as f64 * 0.053).cos();
                if let Some((u, v)) = quad.inverse_map(x as f64 + 0.5, y as f64 + 0.5) {
                    if (0.0..1.0).contains(&u) && (0.0..1.0).contains(&v) {
                        let column = (u * layout.display_columns() as f64) as usize;
                        let row = (v * layout.display_rows() as f64) as usize;
                        let cell = layout
                            .canonical_cell(column, row)
                            .expect("synthetic coordinate is inside display lattice");
                        value += amplitudes[band] * f64::from(signs[cell]);
                    }
                }
                let noise = (((x * 17 + y * 31 + frame * 13) % 11) as f64 - 5.0) * 0.55;
                values[y * width + x] =
                    (exposure * value + noise).round().clamp(1.0, 1023.0) as u16;
            }
        }
        pack_raw10(&values, width, height)
    }

    #[test]
    fn bounded_native_relocalization_recovers_a_crop_following_head_move() {
        let width = 192usize;
        let height = 128usize;
        let stride = width / 4 * 5;
        let session_tag = 10u8;
        let layout = SpatialCodeLayout::CURRENT;
        let reference = ProjectiveQuad {
            corners: [(47.0, 35.0), (151.0, 28.0), (161.0, 101.0), (39.0, 108.0)],
        };
        let pre_payloads = (0..12usize)
            .map(|frame| {
                synthetic_raw_frame(
                    width,
                    height,
                    2,
                    1,
                    reference,
                    frame as u16,
                    session_tag,
                    frame,
                    layout,
                )
            })
            .collect::<Vec<_>>();
        let pre_raw = pre_payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let pre_spectra = pre_raw
            .iter()
            .map(|raw| sample_cell_spectra_interpolated_with_layout(*raw, reference, 5, layout))
            .collect::<Vec<_>>();
        let counters = (0..12u16).collect::<Vec<_>>();
        let baseline = code_aware_temporal_log_baseline_with_scheme(
            &pre_spectra,
            &counters,
            session_tag,
            0,
            GridTransform::Identity,
            OpticalCodeScheme::GrayCrcV1,
        );
        let moved = reference.translated(27.0, -20.0);
        let post_payload =
            synthetic_raw_frame(width, height, 202, 201, moved, 12, session_tag, 12, layout);
        let post_raw = PackedRaw10::new(&post_payload, width, height, stride, 202, 201).unwrap();
        let manifest = ScreenManifest {
            session_id: "crop-follow-relocalization".to_string(),
            session_tag,
            code_hz: 20.0,
            code_layout: layout,
            code_scheme: OpticalCodeScheme::GrayCrcV1,
            presentations: Vec::new(),
        };
        let recovered = reacquire_frame_quad(
            post_raw,
            TrackingContext {
                baseline: &baseline,
                manifest: &manifest,
                code_layout: layout,
                expected_mod: 12,
                counter_radius: 2,
                transform: GridTransform::Identity,
                polarity: 1,
                predicted: reference,
                search_scale: 68.0,
            },
        )
        .expect("bounded native crop-follow search should retain the reflected code");
        assert!(
            (recovered.quad.center().0 - moved.center().0).abs() < 5.0,
            "recovered={:?} expected={:?}",
            recovered.quad.center(),
            moved.center()
        );
        assert!(
            (recovered.quad.center().1 - moved.center().1).abs() < 5.0,
            "recovered={:?} expected={:?}",
            recovered.quad.center(),
            moved.center()
        );
    }

    #[test]
    fn dense_native_bundle_to_manifest_join_decodes_end_to_end() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!(
            "buttercup-screen-reflection-raw-decode-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let manifest_path = root.join("screen.jsonl");
        let frames_path = root.join("frames.jsonl");
        let stream_path = root.join("subject-right.raw10");
        let output_path = root.join("decoded.jsonl");
        let width = 192usize;
        let height = 128usize;
        let stride = width / 4 * 5;
        let length = stride * height;
        let session_tag = 10u8;
        let layout = SpatialCodeLayout::CURRENT;
        let base_unix_ns = 1_800_000_000_000_000_000u64;
        let quad = ProjectiveQuad {
            corners: [(47.0, 35.0), (151.0, 28.0), (161.0, 101.0), (39.0, 108.0)],
        };
        let mut manifest_text = format!(
            "{}\n",
            json!({
                "record_type": "session",
                "session_id": "raw-integration-test",
                "session_tag": session_tag,
                "target_hz": 20.0,
                "code": {
                    "grid": [layout.display_columns(), layout.display_rows()],
                    "logical_grid": [GRID_COLUMNS, GRID_ROWS],
                    "spatial_repeats": [layout.repeat_columns, layout.repeat_rows],
                    "repeat_layout": "complete-logical-grid-tiles"
                }
            })
        );
        let mut frame_text = String::new();
        let mut stream = Vec::new();
        let frame_count = 32u64;
        for frame in 0..frame_count {
            let commit = base_unix_ns + frame * 50_000_000;
            let sensor_x: u32 = if frame < 16 { 2 } else { 6 };
            let sensor_y: u32 = if frame < 24 { 1 } else { 5 };
            let motion_x = frame.saturating_sub(15) as f64 * 0.45;
            let motion_y = frame.saturating_sub(23) as f64 * 0.30;
            let frame_quad = quad.translated(
                2.0 + motion_x - f64::from(sensor_x),
                1.0 + motion_y - f64::from(sensor_y),
            );
            manifest_text.push_str(&format!(
                "{}\n",
                json!({
                    "record_type": "presentation",
                    "presentation_index": frame,
                    "counter_mod": frame,
                    "present_commit_unix_ns": commit,
                    "ball_center_px": [960.0 + frame as f64, 540.0],
                    "ball_center_normalized": [0.5, 0.5]
                })
            ));
            let payload = synthetic_raw_frame(
                width,
                height,
                sensor_x,
                sensor_y,
                frame_quad,
                frame as u16,
                session_tag,
                frame as usize,
                layout,
            );
            stream.extend_from_slice(&payload);
            frame_text.push_str(&format!(
                "{}\n",
                json!({
                    "sequence": 100 + frame,
                    "timestamp_ns": frame * 50_000_000,
                    "host_arrival_unix_ns": commit + 4_000_000,
                    "eye_id": 1,
                    "label": "subject-right",
                    "sensor_x": sensor_x,
                    "sensor_y": sensor_y,
                    "width": width,
                    "height": height,
                    "stride": stride,
                    "pixel_format": "RAW10_LE40_1X1",
                    "stream": "subject-right.raw10",
                    "offset": frame * length as u64,
                    "length": length
                })
            ));
        }
        fs::write(&manifest_path, manifest_text).unwrap();
        fs::write(&frames_path, frame_text).unwrap();
        fs::write(&stream_path, stream).unwrap();
        let config = Config {
            bundle: root.clone(),
            manifest: manifest_path,
            output: Some(output_path.clone()),
            eye: "subject-right".to_string(),
            seed_center: Some(quad.center()),
            seed_sensor_center: None,
            seed_radius: Some(68.0),
            seed_quad: Some(quad),
            locator_frames: 16,
            locator_span_ms: 500,
            counter_offset_radius: 3,
            track_counter_radius: 2,
            minimum_locator_score: 0.50,
            minimum_decode_margin: 0.005,
            maximum_hard_bit_errors: 3,
            initial_presentation: None,
            maximum_frames: None,
            whole_roi_clock: false,
            host_phase_prior: false,
        };
        run(config).unwrap();
        let decoded = fs::read_to_string(output_path).unwrap();
        let rows = decoded
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let locator = rows
            .iter()
            .find(|row| row["record_type"] == "reflection-locator")
            .unwrap();
        assert!(locator["score"].as_f64().unwrap() > 0.70, "{locator:#?}");
        let frame_rows = rows
            .iter()
            .filter(|row| row["record_type"] == "raw-decode")
            .collect::<Vec<_>>();
        assert_eq!(frame_rows.len(), frame_count as usize);
        let exact = frame_rows
            .iter()
            .filter(|row| {
                row["status"] == "accepted"
                    && row["presentation_index"].as_u64() == row["capture_index"].as_u64()
            })
            .count();
        let identity_exact = frame_rows
            .iter()
            .filter(|row| row["presentation_index"].as_u64() == row["capture_index"].as_u64())
            .count();
        let identities = frame_rows
            .iter()
            .map(|row| {
                (
                    row["capture_index"].as_u64(),
                    row["status"].as_str(),
                    row["presentation_index"].as_u64(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            identity_exact, frame_count as usize,
            "identities={identities:?}"
        );
        assert!(exact >= 24, "exact={exact} identities={identities:?}");
        let last_sensor_quad = frame_rows.last().unwrap()["quad_sensor_corners"]
            .as_array()
            .unwrap();
        let recovered_center_x = last_sensor_quad
            .iter()
            .map(|point| point[0].as_f64().unwrap())
            .sum::<f64>()
            * 0.25;
        let expected_center_x = quad.center().0 + 2.0 + (31 - 15) as f64 * 0.45;
        assert!(
            (recovered_center_x - expected_center_x).abs() < 2.5,
            "recovered={recovered_center_x} expected={expected_center_x}"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
