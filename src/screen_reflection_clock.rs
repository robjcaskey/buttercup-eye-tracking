//! Anatomy-independent recovery of the reflected screen clock.
//!
//! The detector scans the complete packed-RAW ROI for a low-distraction
//! screen witness.  A small carrier-neutral integral is used only to propose
//! bright landscape quadrilaterals.  Identity evidence is then accumulated
//! directly from every native 2x2 Quad-Bayer color block inside the proposal;
//! no preview, demosaic, resized raster, pupil estimate, or host timestamp is
//! accepted by the optical solver.

#![allow(dead_code)]

use crate::screen_reflection_code::{
    FrameCode, GridTransform, OpticalCodeScheme, PHYSICAL_CELL_COUNT,
};
use crate::screen_reflection_raw::{PackedRaw10, ProjectiveQuad, CFA_BANDS};

const DISPLAY_COLUMNS: usize = 16;
const DISPLAY_ROWS: usize = 8;
const DISPLAY_CELLS: usize = DISPLAY_COLUMNS * DISPLAY_ROWS;
const OPPONENT_SIGNS: [f64; CFA_BANDS] = [1.0, -1.0, -1.0, 1.0];

#[derive(Clone, Debug)]
struct CarrierGrid {
    width: usize,
    height: usize,
    sensor_x: i64,
    sensor_y: i64,
    bands: Vec<[f64; CFA_BANDS]>,
    neutral: Vec<f64>,
}

impl CarrierGrid {
    fn from_raw(raw: PackedRaw10<'_>) -> Option<Self> {
        let roi_left = i64::from(raw.sensor_x);
        let roi_top = i64::from(raw.sensor_y);
        let roi_right = roi_left + raw.width as i64;
        let roi_bottom = roi_top + raw.height as i64;
        let sensor_x = roi_left + (-roi_left).rem_euclid(4);
        let sensor_y = roi_top + (-roi_top).rem_euclid(4);
        let width = usize::try_from((roi_right - sensor_x).max(0) / 4).ok()?;
        let height = usize::try_from((roi_bottom - sensor_y).max(0) / 4).ok()?;
        if width < 32 || height < 20 {
            return None;
        }
        let mut bands = Vec::with_capacity(width * height);
        let mut neutral = Vec::with_capacity(width * height);
        for carrier_y in 0..height {
            for carrier_x in 0..width {
                let absolute_x = sensor_x + carrier_x as i64 * 4;
                let absolute_y = sensor_y + carrier_y as i64 * 4;
                let mut values = [0.0; CFA_BANDS];
                for (band, value) in values.iter_mut().enumerate() {
                    let offset_x = if band == 0 || band == 2 { 0 } else { 2 };
                    let offset_y = if band < 2 { 0 } else { 2 };
                    let local_x = absolute_x + offset_x - roi_left;
                    let local_y = absolute_y + offset_y - roi_top;
                    if local_x < 0
                        || local_y < 0
                        || local_x + 1 >= raw.width as i64
                        || local_y + 1 >= raw.height as i64
                    {
                        return None;
                    }
                    let x = local_x as usize;
                    let y = local_y as usize;
                    *value = [
                        raw.pixel(x, y),
                        raw.pixel(x + 1, y),
                        raw.pixel(x, y + 1),
                        raw.pixel(x + 1, y + 1),
                    ]
                    .into_iter()
                    .map(f64::from)
                    .sum::<f64>()
                        * 0.25;
                }
                neutral.push(values.iter().sum::<f64>() * 0.25);
                bands.push(values);
            }
        }
        Some(Self {
            width,
            height,
            sensor_x,
            sensor_y,
            bands,
            neutral,
        })
    }

    fn index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    fn luminance_median(&self) -> f64 {
        finite_median(self.neutral.iter().copied()).unwrap_or(1.0)
    }
}

#[derive(Clone, Copy, Debug)]
struct IntegralImage {
    sum: f64,
    squared_sum: f64,
}

#[derive(Clone, Debug)]
struct IntegralGrid {
    stride: usize,
    values: Vec<IntegralImage>,
}

impl IntegralGrid {
    fn new(grid: &CarrierGrid) -> Self {
        let stride = grid.width + 1;
        let mut values = vec![
            IntegralImage {
                sum: 0.0,
                squared_sum: 0.0,
            };
            stride * (grid.height + 1)
        ];
        for y in 0..grid.height {
            let mut row_sum = 0.0;
            let mut row_squared = 0.0;
            for x in 0..grid.width {
                let value = grid.neutral[grid.index(x, y)];
                row_sum += value;
                row_squared += value * value;
                let above = values[y * stride + x + 1];
                values[(y + 1) * stride + x + 1] = IntegralImage {
                    sum: above.sum + row_sum,
                    squared_sum: above.squared_sum + row_squared,
                };
            }
        }
        Self { stride, values }
    }

    fn moments(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Option<(f64, f64)> {
        if x0 >= x1 || y0 >= y1 {
            return None;
        }
        let at = |x: usize, y: usize| self.values[y * self.stride + x];
        let a = at(x0, y0);
        let b = at(x1, y0);
        let c = at(x0, y1);
        let d = at(x1, y1);
        let count = (x1 - x0) * (y1 - y0);
        let sum = d.sum + a.sum - b.sum - c.sum;
        let squared = d.squared_sum + a.squared_sum - b.squared_sum - c.squared_sum;
        let mean = sum / count as f64;
        Some((mean, (squared / count as f64 - mean * mean).max(0.0)))
    }
}

#[derive(Clone, Copy, Debug)]
struct RectangleProposal {
    score: f64,
    center_x: usize,
    center_y: usize,
    width: usize,
    height: usize,
    inner_mean: f64,
    outer_mean: f64,
}

fn centered_bounds(center: usize, extent: usize, limit: usize) -> Option<(usize, usize)> {
    let start = center.checked_sub(extent / 2)?;
    let end = start.checked_add(extent)?;
    (end <= limit).then_some((start, end))
}

fn scan_landscape_rectangle(grid: &CarrierGrid) -> Option<RectangleProposal> {
    let integral = IntegralGrid::new(grid);
    let mut best: Option<RectangleProposal> = None;
    for height in 3..=8 {
        for width in 8..=18 {
            let aspect = width as f64 / height as f64;
            if !(1.7..=4.0).contains(&aspect) {
                continue;
            }
            let outer_width = width + 10;
            let outer_height = height + 8;
            for center_y in 10..grid.height.saturating_sub(10) {
                let Some((inner_y0, inner_y1)) = centered_bounds(center_y, height, grid.height)
                else {
                    continue;
                };
                let Some((outer_y0, outer_y1)) =
                    centered_bounds(center_y, outer_height, grid.height)
                else {
                    continue;
                };
                for center_x in 12..grid.width.saturating_sub(12) {
                    let Some((inner_x0, inner_x1)) = centered_bounds(center_x, width, grid.width)
                    else {
                        continue;
                    };
                    let Some((outer_x0, outer_x1)) =
                        centered_bounds(center_x, outer_width, grid.width)
                    else {
                        continue;
                    };
                    let Some((inner_mean, _)) =
                        integral.moments(inner_x0, inner_y0, inner_x1, inner_y1)
                    else {
                        continue;
                    };
                    let Some((outer_mean, outer_variance)) =
                        integral.moments(outer_x0, outer_y0, outer_x1, outer_y1)
                    else {
                        continue;
                    };
                    let score = (inner_mean - outer_mean) / outer_variance.max(9.0).sqrt()
                        - 0.012 * width.abs_diff(11) as f64
                        - 0.010 * height.abs_diff(5) as f64;
                    let proposal = RectangleProposal {
                        score,
                        center_x,
                        center_y,
                        width,
                        height,
                        inner_mean,
                        outer_mean,
                    };
                    if best.as_ref().is_none_or(|current| score > current.score) {
                        best = Some(proposal);
                    }
                }
            }
        }
    }
    best
}

#[derive(Clone, Copy, Debug)]
struct CarrierBounds {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    area: usize,
}

fn threshold_component(grid: &CarrierGrid, proposal: RectangleProposal) -> Option<CarrierBounds> {
    let threshold = proposal.outer_mean + 0.25 * (proposal.inner_mean - proposal.outer_mean);
    let mut visited = vec![false; grid.width * grid.height];
    let mut stack = vec![(proposal.center_x, proposal.center_y)];
    let mut x0 = proposal.center_x;
    let mut x1 = proposal.center_x + 1;
    let mut y0 = proposal.center_y;
    let mut y1 = proposal.center_y + 1;
    let mut area = 0usize;
    while let Some((x, y)) = stack.pop() {
        if x >= grid.width || y >= grid.height {
            continue;
        }
        let index = grid.index(x, y);
        if visited[index] {
            continue;
        }
        visited[index] = true;
        if grid.neutral[index] <= threshold {
            continue;
        }
        area += 1;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x + 1);
        y1 = y1.max(y + 1);
        for offset_y in -1i32..=1 {
            for offset_x in -1i32..=1 {
                if offset_x == 0 && offset_y == 0 {
                    continue;
                }
                let next_x = x as i32 + offset_x;
                let next_y = y as i32 + offset_y;
                if next_x >= 0 && next_y >= 0 {
                    stack.push((next_x as usize, next_y as usize));
                }
            }
        }
    }
    (area > 0).then_some(CarrierBounds {
        x0,
        y0,
        x1,
        y1,
        area,
    })
}

#[derive(Clone, Debug)]
pub struct WholeRoiClockWitness {
    pub proposal_score: f64,
    pub valid: bool,
    pub quad_roi: ProjectiveQuad,
    pub canonical_cells: [f64; PHYSICAL_CELL_COUNT],
    pub supported_cells: usize,
    pub repeat_agreement: f64,
    pub component_area_carriers: usize,
    pub luminance_median: f64,
}

fn correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    let pairs = left
        .iter()
        .zip(right)
        .filter(|(left, right)| left.is_finite() && right.is_finite())
        .map(|(left, right)| (*left, *right))
        .collect::<Vec<_>>();
    if pairs.len() < 8 {
        return None;
    }
    let left_mean = pairs.iter().map(|pair| pair.0).sum::<f64>() / pairs.len() as f64;
    let right_mean = pairs.iter().map(|pair| pair.1).sum::<f64>() / pairs.len() as f64;
    let mut cross = 0.0;
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    for (left, right) in pairs {
        let left = left - left_mean;
        let right = right - right_mean;
        cross += left * right;
        left_energy += left * left;
        right_energy += right * right;
    }
    (left_energy > 1.0e-12 && right_energy > 1.0e-12)
        .then_some(cross / (left_energy * right_energy).sqrt())
}

pub fn analyze_whole_raw_roi(raw: PackedRaw10<'_>) -> Option<WholeRoiClockWitness> {
    let grid = CarrierGrid::from_raw(raw)?;
    let luminance_median = grid.luminance_median();
    let proposal = scan_landscape_rectangle(&grid)?;
    let fallback = CarrierBounds {
        x0: proposal.center_x.saturating_sub(proposal.width / 2),
        y0: proposal.center_y.saturating_sub(proposal.height / 2),
        x1: (proposal.center_x + proposal.width.div_ceil(2)).min(grid.width),
        y1: (proposal.center_y + proposal.height.div_ceil(2)).min(grid.height),
        area: proposal.width * proposal.height,
    };
    let component = threshold_component(&grid, proposal).unwrap_or(fallback);
    let width = component.x1.saturating_sub(component.x0);
    let height = component.y1.saturating_sub(component.y0);
    let aspect = width as f64 / height.max(1) as f64;
    let component_valid = (8..=22).contains(&width)
        && (3..=12).contains(&height)
        && (1.4..=5.0).contains(&aspect)
        && component.area >= 20;

    let absolute_x0 = grid.sensor_x + component.x0 as i64 * 4;
    let absolute_y0 = grid.sensor_y + component.y0 as i64 * 4;
    let absolute_x1 = grid.sensor_x + component.x1 as i64 * 4;
    let absolute_y1 = grid.sensor_y + component.y1 as i64 * 4;
    let local_x0 = absolute_x0 as f64 - f64::from(raw.sensor_x);
    let local_y0 = absolute_y0 as f64 - f64::from(raw.sensor_y);
    let local_x1 = absolute_x1 as f64 - f64::from(raw.sensor_x);
    let local_y1 = absolute_y1 as f64 - f64::from(raw.sensor_y);
    let quad_roi = ProjectiveQuad {
        corners: [
            (local_x0, local_y0),
            (local_x1, local_y0),
            (local_x1, local_y1),
            (local_x0, local_y1),
        ],
    };

    let mut band_samples: [Vec<f64>; CFA_BANDS] = std::array::from_fn(|_| Vec::new());
    for y in component.y0..component.y1 {
        for x in component.x0..component.x1 {
            let values = grid.bands[grid.index(x, y)];
            for band in 0..CFA_BANDS {
                band_samples[band].push(values[band].max(1.0).ln_1p());
            }
        }
    }
    let band_medians: [f64; CFA_BANDS] = std::array::from_fn(|band| {
        finite_median(band_samples[band].iter().copied()).unwrap_or(0.0)
    });
    let mut display_sum = [0.0; DISPLAY_CELLS];
    let mut display_count = [0u16; DISPLAY_CELLS];
    let span_x = (absolute_x1 - absolute_x0).max(1) as f64;
    let span_y = (absolute_y1 - absolute_y0).max(1) as f64;
    let band_offsets = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0)];
    for y in component.y0..component.y1 {
        for x in component.x0..component.x1 {
            let values = grid.bands[grid.index(x, y)];
            let carrier_x = grid.sensor_x + x as i64 * 4;
            let carrier_y = grid.sensor_y + y as i64 * 4;
            for band in 0..CFA_BANDS {
                let sample_x = carrier_x as f64 + band_offsets[band].0 + 1.0;
                let sample_y = carrier_y as f64 + band_offsets[band].1 + 1.0;
                let column = ((sample_x - absolute_x0 as f64) / span_x * DISPLAY_COLUMNS as f64)
                    .floor() as isize;
                let row = ((sample_y - absolute_y0 as f64) / span_y * DISPLAY_ROWS as f64).floor()
                    as isize;
                if !(0..DISPLAY_COLUMNS as isize).contains(&column)
                    || !(0..DISPLAY_ROWS as isize).contains(&row)
                {
                    continue;
                }
                let display = row as usize * DISPLAY_COLUMNS + column as usize;
                let value =
                    OPPONENT_SIGNS[band] * (values[band].max(1.0).ln_1p() - band_medians[band]);
                display_sum[display] += value;
                display_count[display] = display_count[display].saturating_add(1);
            }
        }
    }
    let display_values = std::array::from_fn::<_, DISPLAY_CELLS, _>(|cell| {
        if display_count[cell] == 0 {
            f64::NAN
        } else {
            display_sum[cell] / f64::from(display_count[cell])
        }
    });
    let mut canonical_sum = [0.0; PHYSICAL_CELL_COUNT];
    let mut canonical_count = [0u16; PHYSICAL_CELL_COUNT];
    for row in 0..DISPLAY_ROWS {
        for column in 0..DISPLAY_COLUMNS {
            let display = row * DISPLAY_COLUMNS + column;
            let canonical = (row % 4) * 8 + column % 8;
            if display_values[display].is_finite() {
                canonical_sum[canonical] += display_values[display];
                canonical_count[canonical] = canonical_count[canonical].saturating_add(1);
            }
        }
    }
    let canonical_cells = std::array::from_fn(|cell| {
        if canonical_count[cell] == 0 {
            f64::NAN
        } else {
            canonical_sum[cell] / f64::from(canonical_count[cell])
        }
    });
    let supported_cells = canonical_cells
        .iter()
        .filter(|value| value.is_finite())
        .count();
    let tile = |repeat_x: usize, repeat_y: usize| {
        (0..4)
            .flat_map(|row| {
                (0..8).map(move |column| {
                    display_values[(repeat_y * 4 + row) * DISPLAY_COLUMNS + repeat_x * 8 + column]
                })
            })
            .collect::<Vec<_>>()
    };
    let tiles = [tile(0, 0), tile(1, 0), tile(0, 1), tile(1, 1)];
    let mut repeat_correlations = Vec::new();
    for right in 0..4 {
        for left in 0..right {
            if let Some(value) = correlation(&tiles[left], &tiles[right]) {
                repeat_correlations.push(value);
            }
        }
    }
    let repeat_agreement = if repeat_correlations.is_empty() {
        0.0
    } else {
        repeat_correlations.iter().sum::<f64>() / repeat_correlations.len() as f64
    };
    Some(WholeRoiClockWitness {
        proposal_score: proposal.score,
        valid: proposal.score > 1.45 && component_valid && supported_cells >= 27,
        quad_roi,
        canonical_cells,
        supported_cells,
        repeat_agreement,
        component_area_carriers: component.area,
        luminance_median,
    })
}

fn finite_median(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

#[derive(Clone, Debug)]
pub struct ClockWitnessStream {
    pub name: String,
    pub witnesses: Vec<Option<WholeRoiClockWitness>>,
}

#[derive(Clone, Copy, Debug)]
pub struct StreamGeometryFit {
    pub transform: GridTransform,
    pub polarity: i8,
    pub score: f64,
    pub support_frames: usize,
}

#[derive(Clone, Debug)]
pub struct OpticalClockFit {
    pub onset_index: usize,
    pub onset_score: f64,
    pub onset_runner_up_score: f64,
    pub rate_hz: f64,
    pub fractional_phase: f64,
    pub onset_counter_delta: i32,
    pub score: f64,
    pub distinct_runner_up_score: f64,
    pub different_counter_delta_runner_up_score: f64,
    pub counter_phase_family_margin: f64,
    pub confidence_margin: f64,
    pub stream_geometry: Vec<(String, StreamGeometryFit)>,
    pub predicted_indices: Vec<i32>,
    pub winner_indices: Vec<i32>,
    pub schedule_ensemble_size: usize,
    pub schedule_consensus: Vec<f64>,
    pub schedule_consensus_frames_75pct: usize,
    pub fit_frames: usize,
    pub direct_witness_frames: usize,
    pub wrong_session_tag_score: f64,
    pub reversed_time_score: f64,
    pub spatial_scramble_score: f64,
}

#[derive(Clone, Debug)]
struct NormalizedStream {
    name: String,
    values: Vec<[f64; PHYSICAL_CELL_COUNT]>,
    counts: [usize; PHYSICAL_CELL_COUNT],
    support_frames: usize,
}

fn transform_signs(
    signs: [i8; PHYSICAL_CELL_COUNT],
    transform: GridTransform,
) -> [i8; PHYSICAL_CELL_COUNT] {
    let mut observed = [0i8; PHYSICAL_CELL_COUNT];
    for canonical in 0..PHYSICAL_CELL_COUNT {
        observed[transform.observed_cell(canonical)] = signs[canonical];
    }
    observed
}

fn codebook(session_tag: u8, scheme: OpticalCodeScheme) -> [[[i8; PHYSICAL_CELL_COUNT]; 4]; 2048] {
    std::array::from_fn(|counter| {
        let signs =
            FrameCode::from_counter_mod(counter as u16, session_tag).physical_signs_for(scheme);
        std::array::from_fn(|transform| transform_signs(signs, GridTransform::ALL[transform]))
    })
}

fn normalize_stream(stream: &ClockWitnessStream, fit_mask: &[bool]) -> Option<NormalizedStream> {
    if stream.witnesses.len() != fit_mask.len() {
        return None;
    }
    let support_frames = stream
        .witnesses
        .iter()
        .zip(fit_mask)
        .filter(|(witness, fit)| **fit && witness.as_ref().is_some_and(|witness| witness.valid))
        .count();
    if support_frames < 24 {
        return None;
    }
    let mut means = [0.0; PHYSICAL_CELL_COUNT];
    let mut counts = [0usize; PHYSICAL_CELL_COUNT];
    for (frame, fit) in stream.witnesses.iter().zip(fit_mask) {
        let Some(witness) = frame.as_ref().filter(|witness| *fit && witness.valid) else {
            continue;
        };
        for cell in 0..PHYSICAL_CELL_COUNT {
            if witness.canonical_cells[cell].is_finite() {
                means[cell] += witness.canonical_cells[cell];
                counts[cell] += 1;
            }
        }
    }
    for cell in 0..PHYSICAL_CELL_COUNT {
        if counts[cell] > 0 {
            means[cell] /= counts[cell] as f64;
        }
    }
    let mut energies = [0.0; PHYSICAL_CELL_COUNT];
    for (frame, fit) in stream.witnesses.iter().zip(fit_mask) {
        let Some(witness) = frame.as_ref().filter(|witness| *fit && witness.valid) else {
            continue;
        };
        for cell in 0..PHYSICAL_CELL_COUNT {
            let value = witness.canonical_cells[cell];
            if value.is_finite() {
                energies[cell] += (value - means[cell]).powi(2);
            }
        }
    }
    let values = stream
        .witnesses
        .iter()
        .zip(fit_mask)
        .map(|(frame, fit)| {
            let mut result = [f64::NAN; PHYSICAL_CELL_COUNT];
            if let Some(witness) = frame.as_ref().filter(|witness| *fit && witness.valid) {
                for cell in 0..PHYSICAL_CELL_COUNT {
                    let value = witness.canonical_cells[cell];
                    if value.is_finite() && energies[cell] > 1.0e-12 {
                        result[cell] = (value - means[cell]) / energies[cell].sqrt();
                    }
                }
            }
            result
        })
        .collect();
    Some(NormalizedStream {
        name: stream.name.clone(),
        values,
        counts,
        support_frames,
    })
}

fn geometry_score(
    stream: &NormalizedStream,
    indices: &[i32],
    codebook: &[[[i8; PHYSICAL_CELL_COUNT]; 4]; 2048],
    reverse_time: bool,
    spatial_rotation: usize,
) -> StreamGeometryFit {
    let mut best = StreamGeometryFit {
        transform: GridTransform::Identity,
        polarity: 1,
        score: f64::NEG_INFINITY,
        support_frames: stream.support_frames,
    };
    for (transform_index, transform) in GridTransform::ALL.into_iter().enumerate() {
        let mut cell_scores = [0.0; PHYSICAL_CELL_COUNT];
        for (frame_index, values) in stream.values.iter().enumerate() {
            let code_frame = if reverse_time {
                indices.len() - 1 - frame_index
            } else {
                frame_index
            };
            let counter = indices[code_frame].rem_euclid(2048) as usize;
            let signs = &codebook[counter][transform_index];
            for observed_cell in 0..PHYSICAL_CELL_COUNT {
                let value = values[observed_cell];
                if value.is_finite() {
                    let expected_cell = (observed_cell + spatial_rotation) % PHYSICAL_CELL_COUNT;
                    cell_scores[observed_cell] += value * f64::from(signs[expected_cell]);
                }
            }
        }
        let score = cell_scores
            .iter()
            .enumerate()
            .filter(|(cell, _)| stream.counts[*cell] >= 20)
            .map(|(cell, score)| score / (stream.counts[cell] as f64).sqrt())
            .sum::<f64>()
            / cell_scores
                .iter()
                .enumerate()
                .filter(|(cell, _)| stream.counts[*cell] >= 20)
                .count()
                .max(1) as f64;
        let candidate = StreamGeometryFit {
            transform,
            polarity: if score >= 0.0 { 1 } else { -1 },
            score: score.abs(),
            support_frames: stream.support_frames,
        };
        if candidate.score > best.score {
            best = candidate;
        }
    }
    best
}

fn schedule_indices(
    timestamps_ns: &[u64],
    onset_index: usize,
    rate_hz: f64,
    fractional_phase: f64,
    delta: i32,
) -> Vec<i32> {
    let onset = timestamps_ns[onset_index];
    timestamps_ns
        .iter()
        .map(|timestamp| {
            let seconds = (*timestamp as i128 - onset as i128) as f64 / 1.0e9;
            (rate_hz * seconds + fractional_phase).floor() as i32 + delta
        })
        .collect()
}

fn schedule_score(
    streams: &[NormalizedStream],
    indices: &[i32],
    codebook: &[[[i8; PHYSICAL_CELL_COUNT]; 4]; 2048],
) -> (f64, Vec<StreamGeometryFit>) {
    let geometry = streams
        .iter()
        .map(|stream| geometry_score(stream, indices, codebook, false, 0))
        .collect::<Vec<_>>();
    let weight = geometry
        .iter()
        .map(|fit| fit.support_frames as f64)
        .sum::<f64>()
        .max(1.0);
    let score = geometry
        .iter()
        .map(|fit| fit.score * fit.support_frames as f64)
        .sum::<f64>()
        / weight;
    (score, geometry)
}

#[derive(Clone, Debug)]
struct ScheduleCandidate {
    rate_hz: f64,
    fractional_phase: f64,
    delta: i32,
    score: f64,
    geometry: Vec<StreamGeometryFit>,
    indices: Vec<i32>,
}

fn median_window(values: &[f64]) -> f64 {
    finite_median(values.iter().copied()).unwrap_or(0.0)
}

pub fn detect_shared_photometric_onset(
    streams: &[ClockWitnessStream],
) -> Option<(usize, f64, f64)> {
    let frames = streams.first()?.witnesses.len();
    if frames < 12
        || streams
            .iter()
            .any(|stream| stream.witnesses.len() != frames)
    {
        return None;
    }
    let luminance = (0..frames)
        .map(|frame| {
            finite_median(streams.iter().filter_map(|stream| {
                stream.witnesses[frame]
                    .as_ref()
                    .map(|witness| witness.luminance_median.max(1.0).ln())
            }))
            .unwrap_or(f64::NAN)
        })
        .collect::<Vec<_>>();
    let mut scores = (4..frames.saturating_sub(4))
        .filter_map(|index| {
            let before = median_window(&luminance[index - 4..index]);
            let after = median_window(&luminance[index..index + 4]);
            let score = after - before;
            score.is_finite().then_some((score, index))
        })
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| right.0.total_cmp(&left.0));
    let (score, index) = *scores.first()?;
    let runner = scores
        .iter()
        .find(|(_, candidate)| candidate.abs_diff(index) > 6)
        .map(|value| value.0)
        .unwrap_or(0.0);
    Some((index, score, runner))
}

/// Locate the beginning of the first sustained, directly decoded optical
/// witness cohort. Auto-exposure can create large luminance steps before the
/// stimulus; the repeated Gray/CRC lattice itself only becomes valid when it
/// is optically present. The absolute counter is still solved from the RAW
/// code. This index only defines zero seconds for the rate/phase search.
pub fn detect_optical_activity_onset(streams: &[ClockWitnessStream]) -> Option<(usize, f64, f64)> {
    let frames = streams.first()?.witnesses.len();
    if frames < 32
        || streams
            .iter()
            .any(|stream| stream.witnesses.len() != frames)
    {
        return None;
    }
    let strongest = streams.iter().max_by_key(|stream| {
        stream
            .witnesses
            .iter()
            .filter(|witness| witness.as_ref().is_some_and(|value| value.valid))
            .count()
    })?;
    let valid = strongest
        .witnesses
        .iter()
        .map(|witness| witness.as_ref().is_some_and(|value| value.valid))
        .collect::<Vec<_>>();
    if valid.iter().filter(|value| **value).count() < 24 {
        return None;
    }
    const WINDOW: usize = 24;
    let mut ranked = (0..=frames - WINDOW)
        .map(|start| {
            let count = valid[start..start + WINDOW]
                .iter()
                .filter(|value| **value)
                .count();
            (count as f64 / WINDOW as f64, start)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
    });
    let (score, cohort_start) = *ranked.first()?;
    if score < 0.25 {
        return None;
    }
    let onset = (cohort_start..cohort_start + WINDOW).find(|index| valid[*index])?;
    let pre_onset_runner = ranked
        .iter()
        .filter(|(_, start)| start.saturating_add(WINDOW) <= onset)
        .map(|(score, _)| *score)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    (score > pre_onset_runner + 0.10).then_some((onset, score, pre_onset_runner))
}

#[allow(clippy::too_many_arguments)]
pub fn solve_optical_clock(
    timestamps_ns: &[u64],
    streams: &[ClockWitnessStream],
    onset_index: usize,
    onset_score: f64,
    onset_runner_up_score: f64,
    code_hz: f64,
    maximum_code_index: u64,
    session_tag: u8,
) -> Option<OpticalClockFit> {
    solve_optical_clock_with_scheme(
        timestamps_ns,
        streams,
        onset_index,
        onset_score,
        onset_runner_up_score,
        code_hz,
        maximum_code_index,
        session_tag,
        OpticalCodeScheme::GrayCrcV1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn solve_optical_clock_with_scheme(
    timestamps_ns: &[u64],
    streams: &[ClockWitnessStream],
    onset_index: usize,
    onset_score: f64,
    onset_runner_up_score: f64,
    code_hz: f64,
    maximum_code_index: u64,
    session_tag: u8,
    scheme: OpticalCodeScheme,
) -> Option<OpticalClockFit> {
    solve_optical_clock_in_delta_range_with_scheme(
        timestamps_ns,
        streams,
        onset_index,
        onset_score,
        onset_runner_up_score,
        code_hz,
        maximum_code_index,
        session_tag,
        -16,
        i32::try_from(maximum_code_index.min(2047)).ok()?,
        scheme,
    )
}

/// Solve the optical clock while using host timing only to bound the absolute
/// counter phase family. Geometry, rate, fractional phase, frame identities,
/// and acceptance scores remain optical. This mirrors the live decoder's
/// bounded current-counter prior and is appropriate for latency estimation.
#[allow(clippy::too_many_arguments)]
pub fn solve_optical_clock_in_delta_range(
    timestamps_ns: &[u64],
    streams: &[ClockWitnessStream],
    onset_index: usize,
    onset_score: f64,
    onset_runner_up_score: f64,
    code_hz: f64,
    maximum_code_index: u64,
    session_tag: u8,
    minimum_delta: i32,
    maximum_delta: i32,
) -> Option<OpticalClockFit> {
    solve_optical_clock_in_delta_range_with_scheme(
        timestamps_ns,
        streams,
        onset_index,
        onset_score,
        onset_runner_up_score,
        code_hz,
        maximum_code_index,
        session_tag,
        minimum_delta,
        maximum_delta,
        OpticalCodeScheme::GrayCrcV1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn solve_optical_clock_in_delta_range_with_scheme(
    timestamps_ns: &[u64],
    streams: &[ClockWitnessStream],
    onset_index: usize,
    onset_score: f64,
    onset_runner_up_score: f64,
    code_hz: f64,
    maximum_code_index: u64,
    session_tag: u8,
    minimum_delta: i32,
    maximum_delta: i32,
    scheme: OpticalCodeScheme,
) -> Option<OpticalClockFit> {
    if timestamps_ns.len() < 32
        || onset_index >= timestamps_ns.len()
        || streams
            .iter()
            .any(|stream| stream.witnesses.len() != timestamps_ns.len())
    {
        return None;
    }
    let onset_timestamp = timestamps_ns[onset_index];
    let protocol_seconds = (maximum_code_index + 1) as f64 / code_hz;
    let fit_mask = timestamps_ns
        .iter()
        .map(|timestamp| {
            let seconds = (*timestamp as i128 - onset_timestamp as i128) as f64 / 1.0e9;
            seconds > 0.20 && seconds < protocol_seconds + 0.05
        })
        .collect::<Vec<_>>();
    let normalized = streams
        .iter()
        .filter_map(|stream| normalize_stream(stream, &fit_mask))
        .collect::<Vec<_>>();
    if normalized.is_empty() {
        return None;
    }
    let table = codebook(session_tag, scheme);
    let mut coarse = Vec::new();
    // A capture can begin long after the stimulus started. In that case the
    // strongest shared luminance step may be an exposure/pose transition,
    // not code zero. Search every counter phase that can occur in this
    // manifest instead of silently assuming the detected step is within 16
    // ticks of stimulus onset. The optical word is modulo 2048, so one full
    // codebook is the largest independently observable search domain.
    let minimum_delta = minimum_delta.max(-16);
    let maximum_delta = maximum_delta.min(i32::try_from(maximum_code_index.min(2047)).ok()?);
    if minimum_delta > maximum_delta {
        return None;
    }
    for phase_step in 0..20 {
        let fractional_phase = phase_step as f64 * 0.05;
        for delta in minimum_delta..=maximum_delta {
            let indices =
                schedule_indices(timestamps_ns, onset_index, code_hz, fractional_phase, delta);
            let (score, geometry) = schedule_score(&normalized, &indices, &table);
            coarse.push(ScheduleCandidate {
                rate_hz: code_hz,
                fractional_phase,
                delta,
                score,
                geometry,
                indices,
            });
        }
    }
    coarse.sort_by(|left, right| right.score.total_cmp(&left.score));
    let best_delta = coarse.first()?.delta;
    let mut candidates = coarse;
    for rate_step in -8..=8 {
        let rate_hz = code_hz + rate_step as f64 * 0.005;
        if (rate_hz - code_hz).abs() < 1.0e-9 {
            continue;
        }
        for phase_step in 0..20 {
            let fractional_phase = phase_step as f64 * 0.05;
            for delta in best_delta - 1..=best_delta + 1 {
                let indices =
                    schedule_indices(timestamps_ns, onset_index, rate_hz, fractional_phase, delta);
                let (score, geometry) = schedule_score(&normalized, &indices, &table);
                candidates.push(ScheduleCandidate {
                    rate_hz,
                    fractional_phase,
                    delta,
                    score,
                    geometry,
                    indices,
                });
            }
        }
    }
    candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
    let winner = candidates.first()?.clone();
    let distinct_runner_up_score = candidates
        .iter()
        .skip(1)
        .find(|candidate| {
            let differences = candidate
                .indices
                .iter()
                .zip(&winner.indices)
                .zip(&fit_mask)
                .filter(|((left, right), fit)| **fit && left != right)
                .count();
            differences * 20 >= fit_mask.iter().filter(|fit| **fit).count()
        })
        .map(|candidate| candidate.score)
        .unwrap_or(0.0);
    let different_counter_delta_runner_up_score = candidates
        .iter()
        .find(|candidate| candidate.delta != winner.delta)
        .map(|candidate| candidate.score)
        .unwrap_or(0.0);

    // Rate and sub-tick phase variants inside one counter-offset family are
    // not competing global locks. They disagree only for exposures adjacent
    // to an LCD transition. Keep that uncertainty per frame instead of
    // rejecting the capture-wide phase. The tolerance is in mean correlation
    // units and is deliberately smaller than the observed counter-family gap.
    let ensemble_tolerance =
        (winner.score - different_counter_delta_runner_up_score).max(0.0) * 0.40;
    let ensemble = candidates
        .iter()
        .filter(|candidate| {
            candidate.delta == winner.delta
                && winner.score - candidate.score <= ensemble_tolerance.max(0.0015)
        })
        .collect::<Vec<_>>();
    let mut predicted_indices = Vec::with_capacity(winner.indices.len());
    let mut schedule_consensus = Vec::with_capacity(winner.indices.len());
    for frame in 0..winner.indices.len() {
        let mut counts = std::collections::BTreeMap::<i32, usize>::new();
        for candidate in &ensemble {
            *counts.entry(candidate.indices[frame]).or_default() += 1;
        }
        let (index, count) = counts
            .into_iter()
            .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
            .unwrap_or((winner.indices[frame], 1));
        predicted_indices.push(index);
        schedule_consensus.push(count as f64 / ensemble.len().max(1) as f64);
    }
    let schedule_consensus_frames_75pct = schedule_consensus
        .iter()
        .filter(|confidence| **confidence >= 0.75)
        .count();

    let wrong_session_tag_score = (0u8..16)
        .filter(|tag| *tag != (session_tag & 0x0f))
        .map(|tag| {
            let wrong = codebook(tag, scheme);
            schedule_score(&normalized, &winner.indices, &wrong).0
        })
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    let reversed_time_score = normalized
        .iter()
        .map(|stream| geometry_score(stream, &winner.indices, &table, true, 0).score)
        .sum::<f64>()
        / normalized.len() as f64;
    let spatial_scramble_score = normalized
        .iter()
        .map(|stream| geometry_score(stream, &winner.indices, &table, false, 5).score)
        .sum::<f64>()
        / normalized.len() as f64;
    let stream_geometry = normalized
        .iter()
        .zip(&winner.geometry)
        .map(|(stream, geometry)| (stream.name.clone(), *geometry))
        .collect::<Vec<_>>();
    let direct_witness_frames = (0..timestamps_ns.len())
        .filter(|frame| {
            streams.iter().any(|stream| {
                stream.witnesses[*frame]
                    .as_ref()
                    .is_some_and(|witness| witness.valid)
            })
        })
        .count();
    Some(OpticalClockFit {
        onset_index,
        onset_score,
        onset_runner_up_score,
        rate_hz: winner.rate_hz,
        fractional_phase: winner.fractional_phase,
        onset_counter_delta: winner.delta,
        score: winner.score,
        distinct_runner_up_score,
        different_counter_delta_runner_up_score,
        counter_phase_family_margin: winner.score - different_counter_delta_runner_up_score,
        confidence_margin: winner.score - different_counter_delta_runner_up_score,
        stream_geometry,
        predicted_indices,
        winner_indices: winner.indices,
        schedule_ensemble_size: ensemble.len(),
        schedule_consensus,
        schedule_consensus_frames_75pct,
        fit_frames: fit_mask.iter().filter(|fit| **fit).count(),
        direct_witness_frames,
        wrong_session_tag_score,
        reversed_time_score,
        spatial_scramble_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn whole_roi_scan_finds_a_screen_without_anatomy() {
        let width = 160usize;
        let height = 96usize;
        let screen = (52usize, 34usize, 108usize, 62usize);
        let code = FrameCode::new(123, 2);
        let signs = code.physical_signs();
        let mut values = vec![100u16; width * height];
        for y in screen.1..screen.3 {
            for x in screen.0..screen.2 {
                let column = (x - screen.0) * DISPLAY_COLUMNS / (screen.2 - screen.0);
                let row = (y - screen.1) * DISPLAY_ROWS / (screen.3 - screen.1);
                let canonical = (row % 4) * 8 + column % 8;
                let band = match (y % 4 < 2, x % 4 < 2) {
                    (true, true) => 0,
                    (true, false) => 1,
                    (false, true) => 2,
                    (false, false) => 3,
                };
                let modulation =
                    i32::from(signs[canonical]) * if OPPONENT_SIGNS[band] > 0.0 { 36 } else { -36 };
                values[y * width + x] = (300 + modulation) as u16;
            }
        }
        let packed = pack_raw10(&values, width, height);
        let raw = PackedRaw10::new(&packed, width, height, width / 4 * 5, 0, 0).unwrap();
        let witness = analyze_whole_raw_roi(raw).unwrap();
        assert!(witness.valid, "{witness:#?}");
        assert!((witness.quad_roi.center().0 - 80.0).abs() <= 4.0);
        assert!((witness.quad_roi.center().1 - 48.0).abs() <= 4.0);
        let expected = signs.map(f64::from);
        let code_correlation = correlation(&witness.canonical_cells, &expected).unwrap();
        assert!(code_correlation > 0.25, "{code_correlation} {witness:#?}");
    }

    fn synthetic_witness(
        counter: i32,
        transform: GridTransform,
        polarity: i8,
    ) -> WholeRoiClockWitness {
        synthetic_witness_for_scheme(counter, transform, polarity, OpticalCodeScheme::GrayCrcV1)
    }

    fn synthetic_witness_for_scheme(
        counter: i32,
        transform: GridTransform,
        polarity: i8,
        scheme: OpticalCodeScheme,
    ) -> WholeRoiClockWitness {
        let signs = transform_signs(
            FrameCode::new(counter as u64, 2).physical_signs_for(scheme),
            transform,
        );
        WholeRoiClockWitness {
            proposal_score: 2.0,
            valid: true,
            quad_roi: ProjectiveQuad::oriented_rectangle((80.0, 50.0), 52.0, 24.0, 0.0),
            canonical_cells: std::array::from_fn(|cell| {
                f64::from(signs[cell]) * f64::from(polarity)
                    + 0.02 * ((counter * 17 + cell as i32 * 11).rem_euclid(9) - 4) as f64
            }),
            supported_cells: PHYSICAL_CELL_COUNT,
            repeat_agreement: 0.9,
            component_area_carriers: 60,
            luminance_median: 220.0,
        }
    }

    #[test]
    fn whitened_clock_resolves_a_causal_prefix_inside_the_host_bound() {
        let timestamps = (0..64)
            .map(|frame| 3_000_000_000u64 + frame * 97_000_000)
            .collect::<Vec<_>>();
        let onset = 5usize;
        let truth = schedule_indices(&timestamps, onset, 30.0, 0.35, 4);
        let witnesses = truth
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                (frame % 3 != 0).then(|| {
                    synthetic_witness_for_scheme(
                        *counter,
                        GridTransform::Rotate180,
                        1,
                        OpticalCodeScheme::PermutedCounterV2,
                    )
                })
            })
            .collect();
        let stream = ClockWitnessStream {
            name: "whitened-causal-prefix".to_string(),
            witnesses,
        };
        let fit = solve_optical_clock_in_delta_range_with_scheme(
            &timestamps,
            &[stream],
            onset,
            0.8,
            0.0,
            30.0,
            220,
            2,
            -12,
            12,
            OpticalCodeScheme::PermutedCounterV2,
        )
        .unwrap();
        assert_eq!(fit.onset_counter_delta, 4);
        assert!(fit.confidence_margin > 0.01, "{fit:#?}");
        let matching = fit
            .predicted_indices
            .iter()
            .zip(&truth)
            .filter(|(left, right)| left == right)
            .count();
        assert!(matching as f64 / truth.len() as f64 > 0.95);
    }

    #[test]
    fn checked_clock_resolves_without_a_long_phase_history() {
        let timestamps = (0..40)
            .map(|frame| 4_000_000_000u64 + frame * 97_000_000)
            .collect::<Vec<_>>();
        let onset = 4usize;
        let truth = schedule_indices(&timestamps, onset, 30.0, 0.42, 7);
        let witnesses = truth
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                (frame % 4 != 0).then(|| {
                    synthetic_witness_for_scheme(
                        *counter,
                        GridTransform::Identity,
                        1,
                        OpticalCodeScheme::ReedMullerV3,
                    )
                })
            })
            .collect();
        let stream = ClockWitnessStream {
            name: "checked-short-prefix".to_string(),
            witnesses,
        };
        let fit = solve_optical_clock_in_delta_range_with_scheme(
            &timestamps,
            &[stream],
            onset,
            0.8,
            0.0,
            30.0,
            180,
            2,
            -5,
            12,
            OpticalCodeScheme::ReedMullerV3,
        )
        .unwrap();
        assert_eq!(fit.onset_counter_delta, 7, "{fit:#?}");
        assert!(fit.confidence_margin > 0.01, "{fit:#?}");
    }

    #[test]
    fn capture_clock_survives_sparse_direct_witnesses() {
        let timestamps = (0..160)
            .map(|frame| 1_000_000_000u64 + frame * 97_000_000)
            .collect::<Vec<_>>();
        let onset = 8usize;
        let truth = schedule_indices(&timestamps, onset, 30.0, 0.65, -3);
        let mut witnesses = Vec::new();
        for (frame, counter) in truth.iter().enumerate() {
            witnesses.push(
                (frame % 5 != 0)
                    .then(|| synthetic_witness(*counter, GridTransform::MirrorHorizontal, -1)),
            );
        }
        let stream = ClockWitnessStream {
            name: "synthetic-whole-roi".to_string(),
            witnesses,
        };
        let fit =
            solve_optical_clock(&timestamps, &[stream], onset, 0.5, 0.1, 30.0, 450, 2).unwrap();
        let matching = fit
            .predicted_indices
            .iter()
            .zip(&truth)
            .filter(|(left, right)| left == right)
            .count();
        assert!(matching as f64 / truth.len() as f64 > 0.95);
        assert_eq!(fit.onset_counter_delta, -3);
        assert_eq!(
            fit.stream_geometry[0].1.transform,
            GridTransform::MirrorHorizontal
        );
        assert_eq!(fit.stream_geometry[0].1.polarity, -1);
    }

    #[test]
    fn capture_clock_recovers_absolute_phase_when_recording_starts_mid_session() {
        let timestamps = (0..112)
            .map(|frame| 2_000_000_000u64 + frame * 98_000_000)
            .collect::<Vec<_>>();
        // This is deliberately a false mid-capture photometric "onset". The
        // optical counter is already around 510 when it occurs.
        let onset = 48usize;
        let truth = schedule_indices(&timestamps, onset, 30.0, 0.40, 510);
        let witnesses = truth
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                (frame % 4 != 0).then(|| synthetic_witness(*counter, GridTransform::Rotate180, -1))
            })
            .collect();
        let stream = ClockWitnessStream {
            name: "mid-session-capture".to_string(),
            witnesses,
        };
        let fit =
            solve_optical_clock(&timestamps, &[stream], onset, 0.02, 0.01, 30.0, 750, 2).unwrap();
        let matching = fit
            .predicted_indices
            .iter()
            .zip(&truth)
            .filter(|(left, right)| left == right)
            .count();
        assert!(matching as f64 / truth.len() as f64 > 0.95);
        assert_eq!(fit.onset_counter_delta, 510);
        assert_eq!(fit.stream_geometry[0].1.transform, GridTransform::Rotate180);
        assert_eq!(fit.stream_geometry[0].1.polarity, -1);
    }

    #[test]
    fn optical_activity_ignores_early_non_clock_brightness_changes() {
        let mut witnesses = vec![None; 180];
        for frame in 74..154 {
            if frame % 5 != 0 {
                witnesses[frame] =
                    Some(synthetic_witness(frame as i32, GridTransform::Identity, 1));
            }
        }
        let stream = ClockWitnessStream {
            name: "activity-onset".to_string(),
            witnesses,
        };
        let (onset, score, runner) = detect_optical_activity_onset(&[stream]).unwrap();
        assert!((74..=78).contains(&onset), "{onset}");
        assert!(score > 0.70, "{score}");
        assert!(runner < 0.10, "{runner}");
    }
}
