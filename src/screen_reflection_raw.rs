//! Native packed-RAW extraction for the screen-reflection frame code.
//!
//! Geometry is evaluated in the original ROI coordinate system. Samples are
//! read directly from packed RAW10 and accumulated separately in the four
//! physical IMX582 Quad-Bayer bands; no preview, demosaic, pyramid, or resized
//! image participates in localization or decoding.

#![allow(dead_code)]

use crate::screen_reflection_code::{
    FrameCode, GridTransform, OpticalCodeScheme, SpatialCodeLayout, GRID_COLUMNS, GRID_ROWS,
    LOGICAL_BIT_COUNT, PAIR_NEGATIVE_CELLS, PAIR_POSITIVE_CELLS, PHYSICAL_CELL_COUNT,
};

pub const CFA_BANDS: usize = 4;
pub const CFA_BAND_NAMES: [&str; CFA_BANDS] = ["R", "G1", "G2", "B"];
const MAX_SPATIAL_REPEATS: usize = 16;
const OPPONENT_SIGNS: [f64; CFA_BANDS] = [1.0, -1.0, -1.0, 1.0];
const OPPONENT_WEIGHTS: [f64; CFA_BANDS] = [0.55, 0.30, 0.30, 1.00];

#[derive(Clone, Copy, Debug)]
pub struct PackedRaw10<'a> {
    payload: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub sensor_x: u32,
    pub sensor_y: u32,
}

impl<'a> PackedRaw10<'a> {
    pub fn new(
        payload: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
        sensor_x: u32,
        sensor_y: u32,
    ) -> Result<Self, String> {
        if width == 0 || height == 0 || !width.is_multiple_of(4) {
            return Err(format!("RAW10 dimensions must be nonzero and width divisible by four, got {width}x{height}"));
        }
        let packed_row = width / 4 * 5;
        if stride < packed_row || payload.len() < stride.saturating_mul(height) {
            return Err(format!(
                "RAW10 payload/stride mismatch: bytes={} geometry={}x{} stride={} minimum-row={packed_row}",
                payload.len(), width, height, stride
            ));
        }
        Ok(Self {
            payload,
            width,
            height,
            stride,
            sensor_x,
            sensor_y,
        })
    }

    #[inline]
    pub fn pixel(self, x: usize, y: usize) -> u16 {
        debug_assert!(x < self.width && y < self.height);
        let group = y * self.stride + x / 4 * 5;
        let word = u64::from(self.payload[group])
            | (u64::from(self.payload[group + 1]) << 8)
            | (u64::from(self.payload[group + 2]) << 16)
            | (u64::from(self.payload[group + 3]) << 24)
            | (u64::from(self.payload[group + 4]) << 32);
        ((word >> ((x & 3) * 10)) & 0x03ff) as u16
    }

    #[inline]
    pub fn cfa_band(self, x: usize, y: usize) -> usize {
        let phase_x = (self.sensor_x as usize + x) & 3;
        let phase_y = (self.sensor_y as usize + y) & 3;
        match (phase_y < 2, phase_x < 2) {
            (true, true) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (false, false) => 3,
        }
    }

    fn carrier_band_mean(
        self,
        carrier_sensor_x: i64,
        carrier_sensor_y: i64,
        band: usize,
    ) -> Option<f64> {
        let offset_x = if band == 0 || band == 2 { 0 } else { 2 };
        let offset_y = if band < 2 { 0 } else { 2 };
        let local_x = carrier_sensor_x + offset_x - i64::from(self.sensor_x);
        let local_y = carrier_sensor_y + offset_y - i64::from(self.sensor_y);
        if local_x < 0
            || local_y < 0
            || local_x + 1 >= self.width as i64
            || local_y + 1 >= self.height as i64
        {
            return None;
        }
        let x = local_x as usize;
        let y = local_y as usize;
        Some(
            [
                self.pixel(x, y),
                self.pixel(x + 1, y),
                self.pixel(x, y + 1),
                self.pixel(x + 1, y + 1),
            ]
            .into_iter()
            .map(f64::from)
            .sum::<f64>()
                * 0.25,
        )
    }

    /// Read one complete native 4x4 Quad-Bayer carrier around a requested
    /// point and return its four independent 2x2 plane means. This is the
    /// inexpensive primitive used by broad line-comb proposal scans.
    fn nearest_carrier_bands(self, x: f64, y: f64) -> Option<[f64; CFA_BANDS]> {
        let absolute_x = i64::from(self.sensor_x) + x.round() as i64;
        let absolute_y = i64::from(self.sensor_y) + y.round() as i64;
        let carrier_x = absolute_x - absolute_x.rem_euclid(4);
        let carrier_y = absolute_y - absolute_y.rem_euclid(4);
        let mut bands = [0.0; CFA_BANDS];
        for (band, destination) in bands.iter_mut().enumerate() {
            *destination = self.carrier_band_mean(carrier_x, carrier_y, band)?;
        }
        Some(bands)
    }

    /// Carrier-neutral native statistic for sparse motion registration. The
    /// sliding 4x4 window spans one complete Quad-Bayer period, but remains at
    /// the requested native coordinate rather than constructing a reduced or
    /// demosaiced image.
    fn neutral_sample(self, x: i32, y: i32) -> Option<f64> {
        if x < 1 || y < 1 || x + 2 >= self.width as i32 || y + 2 >= self.height as i32 {
            return None;
        }
        let mut sum = 0u32;
        for offset_y in -1..=2 {
            for offset_x in -1..=2 {
                sum += u32::from(self.pixel((x + offset_x) as usize, (y + offset_y) as usize));
            }
        }
        Some(sum as f64 / 16.0)
    }

    /// Bilinear interpolation on one physical Quad-Bayer plane. The four
    /// source values are means of complete native 2x2 color blocks in four
    /// neighboring 4x4 carriers. This is a zero-copy RAW accessor, not a
    /// demosaic or a resized image.
    fn sample_band_bilinear(self, x: f64, y: f64, band: usize) -> Option<f64> {
        let band_center_x = if band == 0 || band == 2 { 0.5 } else { 2.5 };
        let band_center_y = if band < 2 { 0.5 } else { 2.5 };
        let sensor_x = f64::from(self.sensor_x) + x;
        let sensor_y = f64::from(self.sensor_y) + y;
        let grid_x = (sensor_x - band_center_x) * 0.25;
        let grid_y = (sensor_y - band_center_y) * 0.25;
        let base_x = grid_x.floor();
        let base_y = grid_y.floor();
        let fraction_x = grid_x - base_x;
        let fraction_y = grid_y - base_y;
        let carrier_x = base_x as i64 * 4;
        let carrier_y = base_y as i64 * 4;
        let top_left = self.carrier_band_mean(carrier_x, carrier_y, band)?;
        let top_right = self.carrier_band_mean(carrier_x + 4, carrier_y, band)?;
        let bottom_left = self.carrier_band_mean(carrier_x, carrier_y + 4, band)?;
        let bottom_right = self.carrier_band_mean(carrier_x + 4, carrier_y + 4, band)?;
        let top = top_left + fraction_x * (top_right - top_left);
        let bottom = bottom_left + fraction_x * (bottom_right - bottom_left);
        Some(top + fraction_y * (bottom - top))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeFrameTranslation {
    /// Cumulative motion from the first locator frame in absolute sensor
    /// pixels. Positive X means the registered eye material moved right.
    pub cumulative: (f64, f64),
    pub step: (f64, f64),
    pub support: usize,
    pub residual: f64,
}

fn sparse_corner_score(raw: PackedRaw10<'_>, x: i32, y: i32) -> f64 {
    let mut xx = 0.0;
    let mut yy = 0.0;
    let mut xy = 0.0;
    for offset_y in -1..=1 {
        for offset_x in -1..=1 {
            let sample_x = x + offset_x;
            let sample_y = y + offset_y;
            let Some(gx) = raw
                .neutral_sample(sample_x + 1, sample_y)
                .zip(raw.neutral_sample(sample_x - 1, sample_y))
                .map(|(right, left)| right - left)
            else {
                return 0.0;
            };
            let Some(gy) = raw
                .neutral_sample(sample_x, sample_y + 1)
                .zip(raw.neutral_sample(sample_x, sample_y - 1))
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
    if trace <= 1.0 {
        0.0
    } else {
        (xx * yy - xy * xy).max(0.0) / trace
    }
}

fn sparse_patch_cost(
    previous: PackedRaw10<'_>,
    current: PackedRaw10<'_>,
    previous_point: (f64, f64),
    current_point: (f64, f64),
) -> f64 {
    let previous_x = previous_point.0.round() as i32;
    let previous_y = previous_point.1.round() as i32;
    let current_x = current_point.0.round() as i32;
    let current_y = current_point.1.round() as i32;
    let mut left_sum = 0.0;
    let mut right_sum = 0.0;
    let mut left_squared = 0.0;
    let mut right_squared = 0.0;
    let mut cross = 0.0;
    let mut count = 0.0;
    for offset_y in (-6..=6).step_by(2) {
        for offset_x in (-6..=6).step_by(2) {
            let (Some(left), Some(right)) = (
                previous.neutral_sample(previous_x + offset_x, previous_y + offset_y),
                current.neutral_sample(current_x + offset_x, current_y + offset_y),
            ) else {
                return f64::INFINITY;
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
        return f64::INFINITY;
    }
    let covariance = cross - left_sum * right_sum / count;
    let correlation = (covariance / (left_energy * right_energy).sqrt().max(48.0)).clamp(-1.0, 1.0);
    (1.0 - correlation).max(0.0).sqrt()
}

fn sparse_registration_features(
    raw: PackedRaw10<'_>,
    center: (f64, f64),
    radius: f64,
) -> Vec<(f64, f64)> {
    let margin = 12.0;
    let left = (center.0 - radius).max(margin);
    let right = (center.0 + radius).min(raw.width as f64 - margin);
    let top = (center.1 - radius * 0.75).max(margin);
    let bottom = (center.1 + radius * 0.75).min(raw.height as f64 - margin);
    if right - left < 48.0 || bottom - top < 36.0 {
        return Vec::new();
    }
    let mut features = Vec::with_capacity(20);
    for row in 0..4 {
        let cell_top = top + (bottom - top) * row as f64 / 4.0;
        let cell_bottom = top + (bottom - top) * (row + 1) as f64 / 4.0;
        for column in 0..5 {
            let cell_left = left + (right - left) * column as f64 / 5.0;
            let cell_right = left + (right - left) * (column + 1) as f64 / 5.0;
            let mut best = None::<(f64, f64, f64)>;
            let mut y = cell_top.ceil() as i32;
            while (y as f64) < cell_bottom {
                let mut x = cell_left.ceil() as i32;
                while (x as f64) < cell_right {
                    let score = sparse_corner_score(raw, x, y);
                    if score.is_finite() && best.is_none_or(|candidate| score > candidate.0) {
                        best = Some((score, x as f64, y as f64));
                    }
                    x += 4;
                }
                y += 4;
            }
            if let Some((score, x, y)) = best.filter(|candidate| candidate.0 >= 12.0) {
                let _ = score;
                features.push((x, y));
            }
        }
    }
    features
}

/// Estimate consecutive eye/material translation directly from sparse native
/// packed-RAW patches. Only the matched sample coordinates are evaluated; no
/// unpacked frame, neutral raster, pyramid, preview, or demosaic is created.
pub fn estimate_native_frame_translations(
    frames: &[PackedRaw10<'_>],
    seed_center: (f64, f64),
    seed_radius: f64,
) -> Vec<NativeFrameTranslation> {
    if frames.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(frames.len());
    result.push(NativeFrameTranslation::default());
    let mut cumulative = (0.0, 0.0);
    for pair in frames.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        let previous_center = (
            seed_center.0 + cumulative.0 + frames[0].sensor_x as f64 - previous.sensor_x as f64,
            seed_center.1 + cumulative.1 + frames[0].sensor_y as f64 - previous.sensor_y as f64,
        );
        let features = sparse_registration_features(previous, previous_center, seed_radius);
        let mut matches = Vec::<(f64, f64, f64)>::new();
        for previous_local in features {
            let previous_sensor = (
                previous_local.0 + previous.sensor_x as f64,
                previous_local.1 + previous.sensor_y as f64,
            );
            let predicted = (
                previous_sensor.0 - current.sensor_x as f64,
                previous_sensor.1 - current.sensor_y as f64,
            );
            let mut candidates = Vec::<(f64, f64, f64)>::new();
            for delta_y in -12..=12 {
                for delta_x in -12..=12 {
                    let current_local =
                        (predicted.0 + delta_x as f64, predicted.1 + delta_y as f64);
                    let cost = sparse_patch_cost(previous, current, previous_local, current_local);
                    if cost.is_finite() {
                        candidates.push((cost, delta_x as f64, delta_y as f64));
                    }
                }
            }
            candidates.sort_by(|left, right| left.0.total_cmp(&right.0));
            let Some(best) = candidates.first().copied() else {
                continue;
            };
            let second = candidates
                .iter()
                .find(|candidate| (candidate.1 - best.1).hypot(candidate.2 - best.2) >= 3.0)
                .map_or(f64::INFINITY, |candidate| candidate.0);
            let margin = if second.is_finite() {
                (second - best.0) / second.max(1.0e-9)
            } else {
                1.0
            };
            if best.0 <= 0.62 && margin >= 0.010 {
                matches.push((best.1, best.2, best.0));
            }
        }
        // A tight eye ROI can contain iris, lid, lashes, and skin motions at
        // once. Select the largest displacement mode before taking medians;
        // averaging all accepted patches is exactly how a lid shadow can drag
        // a nominal eye registration between two physical layers.
        let cluster = matches
            .iter()
            .map(|seed| {
                matches
                    .iter()
                    .filter(|candidate| (candidate.0 - seed.0).hypot(candidate.1 - seed.1) <= 2.25)
                    .copied()
                    .collect::<Vec<_>>()
            })
            .max_by(|left, right| {
                left.len().cmp(&right.len()).then_with(|| {
                    let left_cost =
                        left.iter().map(|item| item.2).sum::<f64>() / left.len().max(1) as f64;
                    let right_cost =
                        right.iter().map(|item| item.2).sum::<f64>() / right.len().max(1) as f64;
                    right_cost.total_cmp(&left_cost)
                })
            })
            .unwrap_or_default();
        let step = if cluster.len() >= 5 {
            let mut x = cluster.iter().map(|item| item.0).collect::<Vec<_>>();
            let mut y = cluster.iter().map(|item| item.1).collect::<Vec<_>>();
            let median_x = finite_median(&mut x).unwrap_or(0.0);
            let median_y = finite_median(&mut y).unwrap_or(0.0);
            (median_x, median_y)
        } else {
            (0.0, 0.0)
        };
        let residual = if cluster.is_empty() {
            f64::INFINITY
        } else {
            let mut values = cluster
                .iter()
                .map(|item| (item.0 - step.0).hypot(item.1 - step.1))
                .collect::<Vec<_>>();
            finite_median(&mut values).unwrap_or(f64::INFINITY)
        };
        let reliable = cluster.len() >= 5 && residual <= 1.75 && step.0.hypot(step.1) <= 12.0;
        let step = if reliable { step } else { (0.0, 0.0) };
        cumulative.0 += step.0;
        cumulative.1 += step.1;
        result.push(NativeFrameTranslation {
            cumulative,
            step,
            support: cluster.len(),
            residual,
        });
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectiveQuad {
    /// Top-left, top-right, bottom-right, bottom-left in native ROI pixels.
    pub corners: [(f64, f64); 4],
}

impl ProjectiveQuad {
    pub fn oriented_rectangle(center: (f64, f64), width: f64, height: f64, angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        let rotate = |x: f64, y: f64| (center.0 + x * cos - y * sin, center.1 + x * sin + y * cos);
        Self {
            corners: [
                rotate(-width * 0.5, -height * 0.5),
                rotate(width * 0.5, -height * 0.5),
                rotate(width * 0.5, height * 0.5),
                rotate(-width * 0.5, height * 0.5),
            ],
        }
    }

    pub fn center(self) -> (f64, f64) {
        let sum = self
            .corners
            .iter()
            .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
        (sum.0 * 0.25, sum.1 * 0.25)
    }

    pub fn width(self) -> f64 {
        let top = distance(self.corners[0], self.corners[1]);
        let bottom = distance(self.corners[3], self.corners[2]);
        (top + bottom) * 0.5
    }

    pub fn height(self) -> f64 {
        let left = distance(self.corners[0], self.corners[3]);
        let right = distance(self.corners[1], self.corners[2]);
        (left + right) * 0.5
    }

    pub fn area(self) -> f64 {
        let mut twice = 0.0;
        for index in 0..4 {
            let current = self.corners[index];
            let next = self.corners[(index + 1) & 3];
            twice += current.0 * next.1 - current.1 * next.0;
        }
        twice.abs() * 0.5
    }

    pub fn plausible_in(self, width: usize, height: usize) -> bool {
        let mut winding_sign = 0.0;
        let convex = (0..4).all(|index| {
            let first = self.corners[index];
            let second = self.corners[(index + 1) & 3];
            let third = self.corners[(index + 2) & 3];
            let cross = (second.0 - first.0) * (third.1 - second.1)
                - (second.1 - first.1) * (third.0 - second.0);
            if cross.abs() < 1.0e-6 {
                return false;
            }
            if winding_sign == 0.0 {
                winding_sign = cross.signum();
            }
            cross.signum() == winding_sign
        });
        let quad_width = self.width();
        let quad_height = self.height();
        convex
            // A native Quad-Bayer carrier is four by four pixels. Anything
            // below 32x16 cannot independently resolve an 8x4 code and must
            // not win merely by reusing the same carrier for adjacent cells.
            && self.area() >= 384.0
            && quad_width >= 30.0
            && quad_height >= 14.0
            && quad_width / quad_height.max(1.0e-9) < 12.0
            && quad_height / quad_width.max(1.0e-9) < 4.0
            && self.corners.iter().all(|point| {
                point.0 >= 2.0
                    && point.1 >= 2.0
                    && point.0 < width.saturating_sub(2) as f64
                    && point.1 < height.saturating_sub(2) as f64
            })
    }

    pub fn transformed_about(
        self,
        origin: (f64, f64),
        destination: (f64, f64),
        scale: f64,
        angle: f64,
    ) -> Self {
        let (sin, cos) = angle.sin_cos();
        let corners = self.corners.map(|point| {
            let x = (point.0 - origin.0) * scale;
            let y = (point.1 - origin.1) * scale;
            (
                destination.0 + x * cos - y * sin,
                destination.1 + x * sin + y * cos,
            )
        });
        Self { corners }
    }

    pub fn translated(self, delta_x: f64, delta_y: f64) -> Self {
        Self {
            corners: self
                .corners
                .map(|point| (point.0 + delta_x, point.1 + delta_y)),
        }
    }

    fn matrix(self) -> Option<[f64; 9]> {
        let [(x0, y0), (x1, y1), (x2, y2), (x3, y3)] = self.corners;
        let dx1 = x1 - x2;
        let dx2 = x3 - x2;
        let dx3 = x0 - x1 + x2 - x3;
        let dy1 = y1 - y2;
        let dy2 = y3 - y2;
        let dy3 = y0 - y1 + y2 - y3;
        let (g, h) = if dx3.abs() + dy3.abs() < 1.0e-12 {
            (0.0, 0.0)
        } else {
            let denominator = dx1 * dy2 - dx2 * dy1;
            if denominator.abs() < 1.0e-12 {
                return None;
            }
            (
                (dx3 * dy2 - dx2 * dy3) / denominator,
                (dx1 * dy3 - dx3 * dy1) / denominator,
            )
        };
        Some([
            x1 - x0 + g * x1,
            x3 - x0 + h * x3,
            x0,
            y1 - y0 + g * y1,
            y3 - y0 + h * y3,
            y0,
            g,
            h,
            1.0,
        ])
    }

    pub fn map(self, u: f64, v: f64) -> Option<(f64, f64)> {
        let matrix = self.matrix()?;
        map_matrix(matrix, u, v)
    }

    pub fn inverse_map(self, x: f64, y: f64) -> Option<(f64, f64)> {
        let inverse = inverse_3x3(self.matrix()?)?;
        map_matrix(inverse, x, y)
    }
}

fn distance(first: (f64, f64), second: (f64, f64)) -> f64 {
    (first.0 - second.0).hypot(first.1 - second.1)
}

fn map_matrix(matrix: [f64; 9], x: f64, y: f64) -> Option<(f64, f64)> {
    let denominator = matrix[6] * x + matrix[7] * y + matrix[8];
    if denominator.abs() < 1.0e-12 {
        return None;
    }
    Some((
        (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
        (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator,
    ))
}

fn inverse_3x3(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let [a, b, c, d, e, f, g, h, i] = matrix;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if determinant.abs() < 1.0e-12 {
        return None;
    }
    let inverse = 1.0 / determinant;
    Some([
        (e * i - f * h) * inverse,
        (c * h - b * i) * inverse,
        (b * f - c * e) * inverse,
        (f * g - d * i) * inverse,
        (a * i - c * g) * inverse,
        (c * d - a * f) * inverse,
        (d * h - e * g) * inverse,
        (b * g - a * h) * inverse,
        (a * e - b * d) * inverse,
    ])
}

#[derive(Clone, Copy, Debug)]
struct TrimmedAccumulator {
    sum: f64,
    minimum: u16,
    maximum: u16,
    count: u16,
}

impl Default for TrimmedAccumulator {
    fn default() -> Self {
        Self {
            sum: 0.0,
            minimum: u16::MAX,
            maximum: 0,
            count: 0,
        }
    }
}

impl TrimmedAccumulator {
    fn push(&mut self, value: u16) {
        self.sum += f64::from(value);
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        self.count = self.count.saturating_add(1);
    }

    fn mean(self) -> f64 {
        if self.count >= 5 {
            (self.sum - f64::from(self.minimum) - f64::from(self.maximum))
                / f64::from(self.count - 2)
        } else if self.count != 0 {
            self.sum / f64::from(self.count)
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FloatAccumulator {
    sum: f64,
    minimum: f64,
    maximum: f64,
    count: u16,
}

impl Default for FloatAccumulator {
    fn default() -> Self {
        Self {
            sum: 0.0,
            minimum: f64::INFINITY,
            maximum: f64::NEG_INFINITY,
            count: 0,
        }
    }
}

impl FloatAccumulator {
    fn push(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.sum += value;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        self.count = self.count.saturating_add(1);
    }

    fn mean(self) -> f64 {
        if self.count >= 5 {
            (self.sum - self.minimum - self.maximum) / f64::from(self.count - 2)
        } else if self.count != 0 {
            self.sum / f64::from(self.count)
        } else {
            f64::NAN
        }
    }
}

#[derive(Clone, Debug)]
pub struct CellSpectra {
    pub values: [[f64; CFA_BANDS]; PHYSICAL_CELL_COUNT],
    pub support: [[u16; CFA_BANDS]; PHYSICAL_CELL_COUNT],
}

fn robust_spatial_repeat_mean(mut values: [f64; MAX_SPATIAL_REPEATS], count: usize) -> f64 {
    if count == 0 {
        return f64::NAN;
    }
    values[..count].sort_by(f64::total_cmp);
    let middle = count / 2;
    if count.is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

impl CellSpectra {
    pub fn support_fraction(&self) -> f64 {
        let valid = self
            .values
            .iter()
            .flat_map(|bands| bands.iter())
            .filter(|value| value.is_finite())
            .count();
        valid as f64 / (PHYSICAL_CELL_COUNT * CFA_BANDS) as f64
    }

    pub fn log_values(&self) -> [[f64; CFA_BANDS]; PHYSICAL_CELL_COUNT] {
        self.values
            .map(|bands| bands.map(|value| value.max(1.0).ln()))
    }
}

pub fn sample_cell_spectra(
    raw: PackedRaw10<'_>,
    quad: ProjectiveQuad,
    subsamples_per_axis: usize,
) -> CellSpectra {
    sample_cell_spectra_with_layout(raw, quad, subsamples_per_axis, SpatialCodeLayout::LEGACY)
}

/// Sample every spatial repeat directly from native packed RAW and accumulate
/// separated copies into the canonical 8 x 4 decoder cells. No repeated tile
/// is resampled or materialized; each contribution comes from its own native
/// corneal coordinates.
pub fn sample_cell_spectra_with_layout(
    raw: PackedRaw10<'_>,
    quad: ProjectiveQuad,
    subsamples_per_axis: usize,
    layout: SpatialCodeLayout,
) -> CellSpectra {
    let subsamples = subsamples_per_axis.clamp(1, 5);
    let repeat_count = layout.repeat_columns * layout.repeat_rows;
    debug_assert!(repeat_count <= MAX_SPATIAL_REPEATS);
    let mut accumulators =
        [[[TrimmedAccumulator::default(); CFA_BANDS]; MAX_SPATIAL_REPEATS]; PHYSICAL_CELL_COUNT];
    let display_columns = layout.display_columns();
    let display_rows = layout.display_rows();
    for row in 0..display_rows {
        for column in 0..display_columns {
            let cell = layout
                .canonical_cell(column, row)
                .expect("display iteration remains inside spatial code layout");
            let repeat = (row / GRID_ROWS) * layout.repeat_columns + column / GRID_COLUMNS;
            for sub_y in 0..subsamples {
                for sub_x in 0..subsamples {
                    let local_u = (sub_x as f64 + 0.5) / subsamples as f64;
                    let local_v = (sub_y as f64 + 0.5) / subsamples as f64;
                    let u = (column as f64 + 0.18 + 0.64 * local_u) / display_columns as f64;
                    let v = (row as f64 + 0.18 + 0.64 * local_v) / display_rows as f64;
                    let Some((x, y)) = quad.map(u, v) else {
                        continue;
                    };
                    let absolute_x = raw.sensor_x as i64 + x.round() as i64;
                    let absolute_y = raw.sensor_y as i64 + y.round() as i64;
                    let carrier_x = absolute_x - absolute_x.rem_euclid(4) - i64::from(raw.sensor_x);
                    let carrier_y = absolute_y - absolute_y.rem_euclid(4) - i64::from(raw.sensor_y);
                    if carrier_x < 0
                        || carrier_y < 0
                        || carrier_x + 3 >= raw.width as i64
                        || carrier_y + 3 >= raw.height as i64
                    {
                        continue;
                    }
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let sample_x = carrier_x as usize + dx;
                            let sample_y = carrier_y as usize + dy;
                            let band = raw.cfa_band(sample_x, sample_y);
                            accumulators[cell][repeat][band].push(raw.pixel(sample_x, sample_y));
                        }
                    }
                }
            }
        }
    }
    let values = std::array::from_fn(|cell| {
        std::array::from_fn(|band| {
            let mut repeat_values = [f64::NAN; MAX_SPATIAL_REPEATS];
            let mut valid = 0usize;
            for repeat in accumulators[cell].iter().take(repeat_count) {
                let value = repeat[band].mean();
                if value.is_finite() {
                    repeat_values[valid] = value;
                    valid += 1;
                }
            }
            robust_spatial_repeat_mean(repeat_values, valid)
        })
    });
    let support = std::array::from_fn(|cell| {
        std::array::from_fn(|band| {
            accumulators[cell]
                .iter()
                .take(repeat_count)
                .fold(0u16, |sum, repeat| sum.saturating_add(repeat[band].count))
        })
    });
    CellSpectra { values, support }
}

/// Sample the same native CFA planes with sub-carrier interpolation and code-
/// cell edge support. This is used only for bounded geometry tracking: samples
/// stay in packed RAW coordinates and no intermediate raster is allocated.
pub fn sample_cell_spectra_interpolated(
    raw: PackedRaw10<'_>,
    quad: ProjectiveQuad,
    subsamples_per_axis: usize,
) -> CellSpectra {
    sample_cell_spectra_interpolated_with_layout(
        raw,
        quad,
        subsamples_per_axis,
        SpatialCodeLayout::LEGACY,
    )
}

pub fn sample_cell_spectra_interpolated_with_layout(
    raw: PackedRaw10<'_>,
    quad: ProjectiveQuad,
    subsamples_per_axis: usize,
    layout: SpatialCodeLayout,
) -> CellSpectra {
    let subsamples = subsamples_per_axis.clamp(2, 7);
    let repeat_count = layout.repeat_columns * layout.repeat_rows;
    debug_assert!(repeat_count <= MAX_SPATIAL_REPEATS);
    let mut accumulators =
        [[[FloatAccumulator::default(); CFA_BANDS]; MAX_SPATIAL_REPEATS]; PHYSICAL_CELL_COUNT];
    let display_columns = layout.display_columns();
    let display_rows = layout.display_rows();
    for row in 0..display_rows {
        for column in 0..display_columns {
            let cell = layout
                .canonical_cell(column, row)
                .expect("display iteration remains inside spatial code layout");
            let repeat = (row / GRID_ROWS) * layout.repeat_columns + column / GRID_COLUMNS;
            for sub_y in 0..subsamples {
                for sub_x in 0..subsamples {
                    let span = (subsamples - 1) as f64;
                    let local_u = 0.04 + 0.92 * sub_x as f64 / span;
                    let local_v = 0.04 + 0.92 * sub_y as f64 / span;
                    let u = (column as f64 + local_u) / display_columns as f64;
                    let v = (row as f64 + local_v) / display_rows as f64;
                    let Some((x, y)) = quad.map(u, v) else {
                        continue;
                    };
                    for (band, accumulator) in accumulators[cell][repeat].iter_mut().enumerate() {
                        if let Some(value) = raw.sample_band_bilinear(x, y, band) {
                            accumulator.push(value);
                        }
                    }
                }
            }
        }
    }
    let values = std::array::from_fn(|cell| {
        std::array::from_fn(|band| {
            let mut repeat_values = [f64::NAN; MAX_SPATIAL_REPEATS];
            let mut valid = 0usize;
            for repeat in accumulators[cell].iter().take(repeat_count) {
                let value = repeat[band].mean();
                if value.is_finite() {
                    repeat_values[valid] = value;
                    valid += 1;
                }
            }
            robust_spatial_repeat_mean(repeat_values, valid)
        })
    });
    let support = std::array::from_fn(|cell| {
        std::array::from_fn(|band| {
            accumulators[cell]
                .iter()
                .take(repeat_count)
                .fold(0u16, |sum, repeat| sum.saturating_add(repeat[band].count))
        })
    });
    CellSpectra { values, support }
}

#[derive(Clone, Copy, Debug)]
pub struct TemporalCodeFit {
    pub score: f64,
    pub runner_up_score: f64,
    pub confidence_margin: f64,
    pub counter_offset: i16,
    pub transform: GridTransform,
    pub polarity: i8,
    pub band_correlations: [f64; CFA_BANDS],
    pub support_fraction: f64,
}

fn offset_counter(counter: u16, offset: i16) -> u16 {
    (i32::from(counter) + i32::from(offset)).rem_euclid(2048) as u16
}

fn band_correlations(
    observations: &[CellSpectra],
    base_counters: &[u16],
    session_tag: u8,
    counter_offset: i16,
    transform: GridTransform,
) -> [f64; CFA_BANDS] {
    let frames = observations.len().min(base_counters.len());
    let mut covariance = [0.0; CFA_BANDS];
    let mut observed_energy = [0.0; CFA_BANDS];
    let mut expected_energy = [0.0; CFA_BANDS];
    for logical in 0..LOGICAL_BIT_COUNT {
        let observed_positive = transform.observed_cell(PAIR_POSITIVE_CELLS[logical]);
        let observed_negative = transform.observed_cell(PAIR_NEGATIVE_CELLS[logical]);
        let mut observed = [[f64::NAN; CFA_BANDS]; 64];
        let mut expected = [0.0; 64];
        let used = frames.min(observed.len());
        for frame in 0..used {
            let code = FrameCode::from_counter_mod(
                offset_counter(base_counters[frame], counter_offset),
                session_tag,
            );
            expected[frame] = if code.logical_bit(logical) { 1.0 } else { -1.0 };
            for ((destination, positive), negative) in observed[frame]
                .iter_mut()
                .zip(observations[frame].values[observed_positive])
                .zip(observations[frame].values[observed_negative])
            {
                if positive.is_finite() && negative.is_finite() {
                    // Log ratios turn frame exposure/gain into an additive
                    // term which disappears in this complementary difference.
                    *destination = positive.max(1.0).ln() - negative.max(1.0).ln();
                }
            }
        }
        for band in 0..CFA_BANDS {
            let valid = (0..used)
                .filter(|frame| observed[*frame][band].is_finite())
                .collect::<Vec<_>>();
            if valid.len() < 4 {
                continue;
            }
            let expected_mean =
                valid.iter().map(|frame| expected[*frame]).sum::<f64>() / valid.len() as f64;
            let observed_mean = valid
                .iter()
                .map(|frame| observed[*frame][band])
                .sum::<f64>()
                / valid.len() as f64;
            for frame in valid {
                let x = observed[frame][band] - observed_mean;
                let y = expected[frame] - expected_mean;
                covariance[band] += x * y;
                observed_energy[band] += x * x;
                expected_energy[band] += y * y;
            }
        }
    }
    std::array::from_fn(|band| {
        let denominator = (observed_energy[band] * expected_energy[band]).sqrt();
        if denominator > 1.0e-12 {
            covariance[band] / denominator
        } else {
            0.0
        }
    })
}

fn spectral_code_score(correlations: [f64; CFA_BANDS], polarity: i8, support: f64) -> f64 {
    let signed = std::array::from_fn::<_, CFA_BANDS, _>(|band| {
        correlations[band] * OPPONENT_SIGNS[band] * f64::from(polarity)
    });
    let mut signed_sum = 0.0;
    let mut wrong_sign = 0.0;
    let mut weight_sum = 0.0;
    for band in 0..CFA_BANDS {
        signed_sum += OPPONENT_WEIGHTS[band] * signed[band];
        wrong_sign += OPPONENT_WEIGHTS[band] * (-signed[band]).max(0.0);
        weight_sum += OPPONENT_WEIGHTS[band];
    }
    let spectral_agreement = signed_sum / weight_sum;
    // Broadband shadow, lid, and exposure changes move all four CFA bands
    // together.  They used to win because the positive red/blue lobe has more
    // total weight than the two green planes, even though both greens had the
    // wrong sign.  The emitted clock is chromatic: red/blue and green are
    // opposing lobes. Require both lobes to support an identity instead of
    // letting one strong common-mode lobe conceal the other.
    let red_blue_agreement = (OPPONENT_WEIGHTS[0] * signed[0] + OPPONENT_WEIGHTS[3] * signed[3])
        / (OPPONENT_WEIGHTS[0] + OPPONENT_WEIGHTS[3]);
    let green_agreement = (signed[1] + signed[2]) * 0.5;
    let weakest_lobe = red_blue_agreement.min(green_agreement);
    (0.55 * spectral_agreement + 0.45 * weakest_lobe - 0.75 * wrong_sign / weight_sum)
        * support.sqrt()
}

pub fn fit_temporal_code(
    observations: &[CellSpectra],
    base_counters: &[u16],
    session_tag: u8,
    minimum_offset: i16,
    maximum_offset: i16,
) -> Option<TemporalCodeFit> {
    if observations.len().min(base_counters.len()) < 5 || minimum_offset > maximum_offset {
        return None;
    }
    let support_fraction = observations
        .iter()
        .map(CellSpectra::support_fraction)
        .sum::<f64>()
        / observations.len() as f64;
    let mut best: Option<TemporalCodeFit> = None;
    let mut runner_up_score = f64::NEG_INFINITY;
    for counter_offset in minimum_offset..=maximum_offset {
        for transform in GridTransform::ALL {
            let correlations = band_correlations(
                observations,
                base_counters,
                session_tag,
                counter_offset,
                transform,
            );
            for polarity in [-1i8, 1i8] {
                let score = spectral_code_score(correlations, polarity, support_fraction);
                let candidate = TemporalCodeFit {
                    score,
                    runner_up_score: f64::NEG_INFINITY,
                    confidence_margin: 0.0,
                    counter_offset,
                    transform,
                    polarity,
                    band_correlations: correlations,
                    support_fraction,
                };
                if best.map(|current| score > current.score).unwrap_or(true) {
                    if let Some(current) = best {
                        runner_up_score = runner_up_score.max(current.score);
                    }
                    best = Some(candidate);
                } else {
                    runner_up_score = runner_up_score.max(score);
                }
            }
        }
    }
    best.map(|mut fit| {
        fit.runner_up_score = runner_up_score;
        fit.confidence_margin = fit.score - runner_up_score;
        fit
    })
}

pub fn temporal_log_baseline(
    observations: &[CellSpectra],
) -> [[f64; CFA_BANDS]; PHYSICAL_CELL_COUNT] {
    std::array::from_fn(|cell| {
        std::array::from_fn(|band| {
            let mut values = observations
                .iter()
                .map(|observation| observation.values[cell][band])
                .filter(|value| value.is_finite())
                .map(|value| value.max(1.0).ln())
                .collect::<Vec<_>>();
            if values.is_empty() {
                return 0.0;
            }
            finite_median(&mut values).unwrap_or(0.0)
        })
    })
}

fn finite_median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Some((values[middle - 1] + values[middle]) * 0.5)
    } else {
        Some(values[middle])
    }
}

fn upper_median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[values.len() / 2])
}

/// Estimate the static cell/chromatic baseline without baking an unbalanced
/// short Gray-code warmup into it. For every observed cell and CFA band, the
/// midpoint of the expected positive/negative cohorts removes the screen
/// modulation. A frame-band median first removes exposure drift. Cells whose
/// symbol never toggled remain neutral/erased rather than becoming confidently
/// wrong; temporal decoding and CRC can still use the symbols that did toggle.
pub fn code_aware_temporal_log_baseline(
    observations: &[CellSpectra],
    base_counters: &[u16],
    session_tag: u8,
    counter_offset: i16,
    transform: GridTransform,
) -> [[f64; CFA_BANDS]; PHYSICAL_CELL_COUNT] {
    code_aware_temporal_log_baseline_with_scheme(
        observations,
        base_counters,
        session_tag,
        counter_offset,
        transform,
        OpticalCodeScheme::GrayCrcV1,
    )
}

pub fn code_aware_temporal_log_baseline_with_scheme(
    observations: &[CellSpectra],
    base_counters: &[u16],
    session_tag: u8,
    counter_offset: i16,
    transform: GridTransform,
    scheme: OpticalCodeScheme,
) -> [[f64; CFA_BANDS]; PHYSICAL_CELL_COUNT] {
    let frames = observations.len().min(base_counters.len());
    if frames == 0 {
        return [[0.0; CFA_BANDS]; PHYSICAL_CELL_COUNT];
    }
    let logs = observations
        .iter()
        .take(frames)
        .map(CellSpectra::log_values)
        .collect::<Vec<_>>();
    let frame_band_medians: Vec<[f64; CFA_BANDS]> = logs
        .iter()
        .map(|frame| {
            std::array::from_fn(|band| {
                let mut values = frame
                    .iter()
                    .map(|cell| cell[band])
                    .filter(|value| value.is_finite())
                    .collect::<Vec<_>>();
                if values.is_empty() {
                    return 0.0;
                }
                // Locator windows are intentionally short. Retaining an
                // observed order statistic is more robust to an LCD/global-
                // shutter transition than averaging samples from different
                // presentation phases.
                upper_median(&mut values).unwrap_or(0.0)
            })
        })
        .collect::<Vec<_>>();
    let expected = base_counters
        .iter()
        .take(frames)
        .map(|counter| {
            let canonical =
                FrameCode::from_counter_mod(offset_counter(*counter, counter_offset), session_tag)
                    .physical_signs_for(scheme);
            let mut observed = [0i8; PHYSICAL_CELL_COUNT];
            for canonical_cell in 0..PHYSICAL_CELL_COUNT {
                observed[transform.observed_cell(canonical_cell)] = canonical[canonical_cell];
            }
            observed
        })
        .collect::<Vec<_>>();
    std::array::from_fn(|cell| {
        std::array::from_fn(|band| {
            let mut positive = Vec::new();
            let mut negative = Vec::new();
            for frame in 0..frames {
                let normalized = logs[frame][cell][band] - frame_band_medians[frame][band];
                if !normalized.is_finite() {
                    continue;
                }
                if expected[frame][cell] > 0 {
                    positive.push(normalized);
                } else {
                    negative.push(normalized);
                }
            }
            match (upper_median(&mut positive), upper_median(&mut negative)) {
                (Some(high), Some(low)) => (high + low) * 0.5,
                (Some(value), None) | (None, Some(value)) => value,
                (None, None) => 0.0,
            }
        })
    })
}

pub fn opponent_residual_cells(
    observation: &CellSpectra,
    baseline: &[[f64; CFA_BANDS]; PHYSICAL_CELL_COUNT],
    polarity: i8,
) -> [f64; PHYSICAL_CELL_COUNT] {
    let logs = observation.log_values();
    let mut frame_band_medians = [0.0; CFA_BANDS];
    for band in 0..CFA_BANDS {
        let mut residuals = (0..PHYSICAL_CELL_COUNT)
            .map(|cell| logs[cell][band] - baseline[cell][band])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        residuals.sort_by(f64::total_cmp);
        if !residuals.is_empty() {
            frame_band_medians[band] = residuals[residuals.len() / 2];
        }
    }
    std::array::from_fn(|cell| {
        let mut value = 0.0;
        let mut weight = 0.0;
        for band in 0..CFA_BANDS {
            let residual = logs[cell][band] - baseline[cell][band] - frame_band_medians[band];
            if residual.is_finite() {
                value += OPPONENT_WEIGHTS[band] * OPPONENT_SIGNS[band] * residual;
                weight += OPPONENT_WEIGHTS[band];
            }
        }
        f64::from(polarity) * value / weight.max(1.0e-9)
    })
}

#[derive(Clone, Copy, Debug)]
pub struct QuadFit {
    pub quad: ProjectiveQuad,
    pub temporal: TemporalCodeFit,
    pub lane: Option<TemporalLaneFit>,
    pub best_lane_proposal_score: Option<f64>,
}

/// Code-identity-independent evidence that a projective patch contains the
/// displayed lattice.  The score is built from consecutive native-RAW CFA
/// observations before any Gray counter, CRC, or host-time prior is consulted.
/// It is therefore suitable as a spatial proposal stage rather than as proof
/// of a particular frame identity.
#[derive(Clone, Copy, Debug)]
pub struct TemporalLaneFit {
    pub score: f64,
    pub transform: GridTransform,
    pub opponent_activity: f64,
    pub complementary_agreement: f64,
    pub repeat_agreement: f64,
    pub common_mode_rejection: f64,
    pub transitions: usize,
}

fn display_cell_log_spectra(
    raw: PackedRaw10<'_>,
    quad: ProjectiveQuad,
    layout: SpatialCodeLayout,
) -> Option<Vec<[f64; CFA_BANDS]>> {
    if !quad.plausible_in(raw.width, raw.height) {
        return None;
    }
    let columns = layout.display_columns();
    let rows = layout.display_rows();
    let mut cells = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for column in 0..columns {
            // One complete native carrier is enough for the broad proposal
            // stage. It costs sixteen packed samples, preserves all four CFA
            // planes, and never creates an intermediate image. Fine scoring
            // subsequently uses the sub-carrier interpolated sampler.
            let u = (column as f64 + 0.50) / columns as f64;
            let v = (row as f64 + 0.50) / rows as f64;
            let (x, y) = quad.map(u, v)?;
            cells.push(
                raw.nearest_carrier_bands(x, y)?
                    .map(|value| value.max(1.0).ln()),
            );
        }
    }
    Some(cells)
}

#[derive(Clone)]
struct LaneObservation {
    opponent: [f64; PHYSICAL_CELL_COUNT],
    common: [f64; PHYSICAL_CELL_COUNT],
    scatter: [f64; PHYSICAL_CELL_COUNT],
    weight: f64,
}

fn collapse_display_lanes(
    display: &[[f64; CFA_BANDS]],
    layout: SpatialCodeLayout,
    weight: f64,
) -> Option<LaneObservation> {
    let repeat_count = layout.repeat_columns * layout.repeat_rows;
    let display_columns = layout.display_columns();
    if display.len() != display_columns * layout.display_rows() || repeat_count < 2 {
        return None;
    }
    // Exposure and broad illumination drift are additive in log RAW. A
    // per-band spatial median removes them without mixing CFA planes.
    let band_medians: [f64; CFA_BANDS] = std::array::from_fn(|band| {
        let mut values = display
            .iter()
            .map(|cell| cell[band])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        finite_median(&mut values).unwrap_or(0.0)
    });
    let mut repeat_opponent = [[f64::NAN; MAX_SPATIAL_REPEATS]; PHYSICAL_CELL_COUNT];
    let mut repeat_common = [[f64::NAN; MAX_SPATIAL_REPEATS]; PHYSICAL_CELL_COUNT];
    for display_row in 0..layout.display_rows() {
        for display_column in 0..display_columns {
            let display_cell = display_row * display_columns + display_column;
            let canonical = layout.canonical_cell(display_column, display_row)?;
            let repeat =
                (display_row / GRID_ROWS) * layout.repeat_columns + display_column / GRID_COLUMNS;
            let normalized = std::array::from_fn::<_, CFA_BANDS, _>(|band| {
                display[display_cell][band] - band_medians[band]
            });
            if normalized.iter().any(|value| !value.is_finite()) {
                continue;
            }
            let red_blue = (OPPONENT_WEIGHTS[0] * normalized[0]
                + OPPONENT_WEIGHTS[3] * normalized[3])
                / (OPPONENT_WEIGHTS[0] + OPPONENT_WEIGHTS[3]);
            let green = (normalized[1] + normalized[2]) * 0.5;
            repeat_opponent[canonical][repeat] = (red_blue - green) * 0.5;
            repeat_common[canonical][repeat] = (red_blue + green) * 0.5;
        }
    }
    let mut opponent = [f64::NAN; PHYSICAL_CELL_COUNT];
    let mut common = [f64::NAN; PHYSICAL_CELL_COUNT];
    let mut scatter = [f64::NAN; PHYSICAL_CELL_COUNT];
    for cell in 0..PHYSICAL_CELL_COUNT {
        let mut opponent_values = repeat_opponent[cell][..repeat_count]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let mut common_values = repeat_common[cell][..repeat_count]
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if opponent_values.len() != repeat_count || common_values.len() != repeat_count {
            continue;
        }
        opponent[cell] = finite_median(&mut opponent_values).unwrap_or(f64::NAN);
        common[cell] = finite_median(&mut common_values).unwrap_or(f64::NAN);
        let mut deviations = opponent_values
            .iter()
            .map(|value| (value - opponent[cell]).abs())
            .collect::<Vec<_>>();
        scatter[cell] = finite_median(&mut deviations).unwrap_or(f64::NAN);
    }
    Some(LaneObservation {
        opponent,
        common,
        scatter,
        weight,
    })
}

/// Sweep a rotated, regularly spaced line comb over a candidate quad and ask
/// whether its temporal RAW-color changes have the invariants of the screen
/// lattice:
///
/// * red/blue and the two green planes form opposing chromatic lobes;
/// * every complete tiled copy reports the same canonical-cell change; and
/// * the two cells in each balanced logical pair change oppositely.
///
/// Static iris texture has no temporal activity after registration.  Motion
/// through an iris/lid edge can create activity, but it normally fails both
/// the separated-tile repetition and complementary-pair tests.  A broadband
/// shadow moves the two chromatic lobes together and is explicitly rejected.
pub fn score_temporal_lattice_lanes_with_layout(
    frames: &[PackedRaw10<'_>],
    quad: ProjectiveQuad,
    layout: SpatialCodeLayout,
) -> Option<TemporalLaneFit> {
    score_temporal_lattice_lanes_registered(frames, quad, layout, &[])
}

pub fn score_temporal_lattice_lanes_registered(
    frames: &[PackedRaw10<'_>],
    quad: ProjectiveQuad,
    layout: SpatialCodeLayout,
    translations: &[NativeFrameTranslation],
) -> Option<TemporalLaneFit> {
    if frames.len() < 3 {
        return None;
    }
    let reference_sensor_x = frames[0].sensor_x as f64;
    let reference_sensor_y = frames[0].sensor_y as f64;
    let sampled = frames
        .iter()
        .enumerate()
        .map(|(index, frame)| {
            let material_motion = translations
                .get(index)
                .map_or((0.0, 0.0), |transport| transport.cumulative);
            let frame_quad = quad.translated(
                reference_sensor_x - frame.sensor_x as f64 + material_motion.0,
                reference_sensor_y - frame.sensor_y as f64 + material_motion.1,
            );
            display_cell_log_spectra(*frame, frame_quad, layout)
        })
        .collect::<Option<Vec<_>>>()?;
    let repeat_count = layout.repeat_columns * layout.repeat_rows;
    if repeat_count < 2 {
        return None;
    }

    let mut observations = Vec::with_capacity(sampled.len() * 2 - 1);
    // A spatial snapshot is the rotated-line/Canny analogue the proposal
    // stage was missing: the four separated tiles must contain the same
    // signed opponent-color edge arrangement, and every balanced pair must be
    // complementary. Weight it below the inter-frame change measurement so
    // static iris chroma can propose a patch but cannot dominate it.
    for display in &sampled {
        observations.push(collapse_display_lanes(display, layout, 0.35)?);
    }
    for pair in sampled.windows(2) {
        let mut deltas = vec![[f64::NAN; CFA_BANDS]; pair[0].len()];
        for (cell, destination) in deltas.iter_mut().enumerate() {
            for band in 0..CFA_BANDS {
                let previous = pair[0][cell][band];
                let current = pair[1][cell][band];
                if previous.is_finite() && current.is_finite() {
                    destination[band] = current - previous;
                }
            }
        }
        observations.push(collapse_display_lanes(&deltas, layout, 1.0)?);
    }

    let mut best: Option<TemporalLaneFit> = None;
    for transform in GridTransform::ALL {
        let mut signal = 0.0;
        let mut complement_leakage = 0.0;
        let mut repeat_scatter = 0.0;
        let mut common_leakage = 0.0;
        let mut terms = 0.0;
        for observation in &observations {
            for logical in 0..LOGICAL_BIT_COUNT {
                let positive = transform.observed_cell(PAIR_POSITIVE_CELLS[logical]);
                let negative = transform.observed_cell(PAIR_NEGATIVE_CELLS[logical]);
                let first = observation.opponent[positive];
                let second = observation.opponent[negative];
                if !first.is_finite() || !second.is_finite() {
                    continue;
                }
                signal += observation.weight * (first - second).abs() * 0.5;
                complement_leakage += observation.weight * (first + second).abs() * 0.5;
                repeat_scatter += observation.weight
                    * (observation.scatter[positive] + observation.scatter[negative]);
                common_leakage += observation.weight
                    * (observation.common[positive].abs() + observation.common[negative].abs());
                terms += observation.weight;
            }
        }
        if terms <= 0.0 {
            continue;
        }
        let epsilon = 1.0e-9;
        let complementary_agreement = ((signal - complement_leakage)
            / (signal + complement_leakage + epsilon))
            .clamp(-1.0, 1.0);
        let repeat_agreement =
            (1.0 - repeat_scatter / (2.0 * signal + repeat_scatter + epsilon)).clamp(0.0, 1.0);
        let common_mode_rejection =
            (1.0 - common_leakage / (2.0 * signal + common_leakage + epsilon)).clamp(0.0, 1.0);
        let opponent_activity = signal / terms;
        // Around 0.4% differential log-RAW activity is already meaningful at
        // this stimulus amplitude.  The saturation prevents a moving hard
        // edge from winning solely by being brighter than the screen code.
        let activity_gate = (opponent_activity / 0.004).tanh();
        let score =
            complementary_agreement * repeat_agreement * common_mode_rejection * activity_gate;
        let candidate = TemporalLaneFit {
            score,
            transform,
            opponent_activity,
            complementary_agreement,
            repeat_agreement,
            common_mode_rejection,
            transitions: sampled.len().saturating_sub(1),
        };
        if best.is_none_or(|current| candidate.score > current.score) {
            best = Some(candidate);
        }
    }
    best
}

pub fn score_quad(
    frames: &[PackedRaw10<'_>],
    base_counters: &[u16],
    session_tag: u8,
    quad: ProjectiveQuad,
    minimum_offset: i16,
    maximum_offset: i16,
    subsamples_per_axis: usize,
) -> Option<QuadFit> {
    score_quad_with_layout(
        frames,
        base_counters,
        session_tag,
        quad,
        minimum_offset,
        maximum_offset,
        subsamples_per_axis,
        SpatialCodeLayout::LEGACY,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn score_quad_with_layout(
    frames: &[PackedRaw10<'_>],
    base_counters: &[u16],
    session_tag: u8,
    quad: ProjectiveQuad,
    minimum_offset: i16,
    maximum_offset: i16,
    subsamples_per_axis: usize,
    layout: SpatialCodeLayout,
) -> Option<QuadFit> {
    if frames.is_empty() || !quad.plausible_in(frames[0].width, frames[0].height) {
        return None;
    }
    let reference_sensor_x = frames[0].sensor_x as f64;
    let reference_sensor_y = frames[0].sensor_y as f64;
    let observations = frames
        .iter()
        .map(|frame| {
            let frame_quad = quad.translated(
                reference_sensor_x - frame.sensor_x as f64,
                reference_sensor_y - frame.sensor_y as f64,
            );
            sample_cell_spectra_with_layout(*frame, frame_quad, subsamples_per_axis, layout)
        })
        .collect::<Vec<_>>();
    let temporal = fit_temporal_code(
        &observations,
        base_counters,
        session_tag,
        minimum_offset,
        maximum_offset,
    )?;
    let lane = if layout.repeat_columns * layout.repeat_rows >= 2 {
        score_temporal_lattice_lanes_with_layout(frames, quad, layout)
    } else {
        None
    };
    Some(QuadFit {
        quad,
        temporal,
        lane,
        best_lane_proposal_score: None,
    })
}

pub fn search_reflection_quad(
    frames: &[PackedRaw10<'_>],
    base_counters: &[u16],
    session_tag: u8,
    seed_center: (f64, f64),
    seed_radius: f64,
    minimum_offset: i16,
    maximum_offset: i16,
) -> Option<QuadFit> {
    search_reflection_quad_with_layout(
        frames,
        base_counters,
        session_tag,
        seed_center,
        seed_radius,
        minimum_offset,
        maximum_offset,
        SpatialCodeLayout::LEGACY,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn search_reflection_quad_with_layout(
    frames: &[PackedRaw10<'_>],
    base_counters: &[u16],
    session_tag: u8,
    seed_center: (f64, f64),
    seed_radius: f64,
    minimum_offset: i16,
    maximum_offset: i16,
    layout: SpatialCodeLayout,
) -> Option<QuadFit> {
    if frames.len().min(base_counters.len()) < 5 || seed_radius < 8.0 {
        return None;
    }
    // The current display contains separated complete repetitions of the
    // logical lattice.  Sweep a rotated temporal line comb first and reserve
    // the expensive code/CRC evaluation for the strongest spatial proposals.
    // Legacy one-tile recordings cannot provide this invariant and retain the
    // exhaustive historical path.
    let use_lane_proposals = layout.repeat_columns * layout.repeat_rows >= 2;
    let frame_translations = if use_lane_proposals {
        estimate_native_frame_translations(frames, seed_center, seed_radius * 0.72)
    } else {
        Vec::new()
    };
    let mut proposals = Vec::<(ProjectiveQuad, f64)>::new();
    for center_y in [-0.30, -0.15, 0.0, 0.15, 0.30] {
        for center_x in [-0.36, -0.18, 0.0, 0.18, 0.36] {
            let center = (
                seed_center.0 + center_x * seed_radius,
                seed_center.1 + center_y * seed_radius,
            );
            for width_factor in [0.55, 0.80, 1.05, 1.30, 1.55, 1.80] {
                let width = width_factor * seed_radius;
                for aspect in [0.38, 0.55, 0.72] {
                    for angle in [-0.22, 0.0, 0.22] {
                        let quad = ProjectiveQuad::oriented_rectangle(
                            center,
                            width,
                            width * aspect,
                            angle,
                        );
                        if !quad.plausible_in(frames[0].width, frames[0].height) {
                            continue;
                        }
                        let lane_score = if use_lane_proposals {
                            let Some(lane) = score_temporal_lattice_lanes_registered(
                                frames,
                                quad,
                                layout,
                                &frame_translations,
                            ) else {
                                continue;
                            };
                            lane.score
                        } else {
                            0.0
                        };
                        proposals.push((quad, lane_score));
                    }
                }
            }
        }
    }
    if use_lane_proposals {
        proposals.sort_by(|left, right| right.1.total_cmp(&left.1));
    }
    let best_lane_proposal_score = use_lane_proposals
        .then(|| proposals.first().map(|proposal| proposal.1))
        .flatten();
    if use_lane_proposals {
        proposals.truncate(64);
    }
    let mut coarse = proposals
        .into_iter()
        .filter_map(|(quad, lane_score)| {
            score_quad_with_layout(
                frames,
                base_counters,
                session_tag,
                quad,
                minimum_offset,
                maximum_offset,
                2,
                layout,
            )
            .map(|fit| (fit, lane_score))
        })
        .collect::<Vec<_>>();
    coarse.sort_by(|left, right| {
        let left_score = left.0.temporal.score + 0.30 * left.1;
        let right_score = right.0.temporal.score + 0.30 * right.1;
        right_score.total_cmp(&left_score)
    });
    coarse.truncate(6);
    let mut refined = coarse.clone();
    for (seed, _) in coarse {
        let center = seed.quad.center();
        let width = seed.quad.width();
        let height = seed.quad.height();
        let edge = (
            seed.quad.corners[1].0 - seed.quad.corners[0].0,
            seed.quad.corners[1].1 - seed.quad.corners[0].1,
        );
        let base_angle = edge.1.atan2(edge.0);
        for dy in [-0.07, 0.0, 0.07] {
            for dx in [-0.07, 0.0, 0.07] {
                for width_scale in [0.92, 1.0, 1.08] {
                    for height_scale in [0.92, 1.0, 1.08] {
                        for angle_delta in [-0.06, 0.0, 0.06] {
                            let quad = ProjectiveQuad::oriented_rectangle(
                                (center.0 + dx * seed_radius, center.1 + dy * seed_radius),
                                width * width_scale,
                                height * height_scale,
                                base_angle + angle_delta,
                            );
                            if let Some(fit) = score_quad_with_layout(
                                frames,
                                base_counters,
                                session_tag,
                                quad,
                                minimum_offset,
                                maximum_offset,
                                3,
                                layout,
                            ) {
                                let lane_score = if use_lane_proposals {
                                    score_temporal_lattice_lanes_registered(
                                        frames,
                                        quad,
                                        layout,
                                        &frame_translations,
                                    )
                                    .map_or(f64::NEG_INFINITY, |lane| lane.score)
                                } else {
                                    0.0
                                };
                                refined.push((fit, lane_score));
                            }
                        }
                    }
                }
            }
        }
    }
    let best = refined
        .into_iter()
        .max_by(|left, right| {
            let left_score = left.0.temporal.score + 0.30 * left.1;
            let right_score = right.0.temporal.score + 0.30 * right.1;
            left_score.total_cmp(&right_score)
        })?
        .0;
    let mut result = refine_reflection_quad_with_layout(
        frames,
        base_counters,
        session_tag,
        best,
        seed_radius,
        minimum_offset,
        maximum_offset,
        layout,
    );
    result.best_lane_proposal_score = best_lane_proposal_score;
    if use_lane_proposals {
        result.lane = score_temporal_lattice_lanes_registered(
            frames,
            result.quad,
            layout,
            &frame_translations,
        );
    }
    Some(result)
}

/// Let the four corners depart from an oriented rectangle after the coarse
/// search. A corneal screen image is locally projective but rarely a perfect
/// rectangle; bounded coordinate descent recovers keystone and shear without
/// opening an unconstrained eight-dimensional search.
pub fn refine_reflection_quad(
    frames: &[PackedRaw10<'_>],
    base_counters: &[u16],
    session_tag: u8,
    seed: QuadFit,
    seed_radius: f64,
    minimum_offset: i16,
    maximum_offset: i16,
) -> QuadFit {
    refine_reflection_quad_with_layout(
        frames,
        base_counters,
        session_tag,
        seed,
        seed_radius,
        minimum_offset,
        maximum_offset,
        SpatialCodeLayout::LEGACY,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn refine_reflection_quad_with_layout(
    frames: &[PackedRaw10<'_>],
    base_counters: &[u16],
    session_tag: u8,
    seed: QuadFit,
    seed_radius: f64,
    minimum_offset: i16,
    maximum_offset: i16,
    layout: SpatialCodeLayout,
) -> QuadFit {
    let anchor = seed.quad;
    let mut best = seed;
    let regularized_score = |candidate: &QuadFit| {
        let mean_corner_displacement = candidate
            .quad
            .corners
            .iter()
            .zip(anchor.corners)
            .map(|(point, origin)| (point.0 - origin.0).hypot(point.1 - origin.1))
            .sum::<f64>()
            * 0.25;
        let lane_score = candidate.lane.map_or(0.0, |lane| lane.score);
        // Temporal identity still has authority to recover real keystone and
        // shear, but corner refinement must retain the independently observed
        // repeated-line structure. Score-equivalent solutions may not walk
        // over fixed iris texture after the lane proposal stage.
        candidate.temporal.score + 0.30 * lane_score
            - 0.018 * mean_corner_displacement / seed_radius.max(1.0)
    };
    for step in [0.060, 0.030, 0.015].map(|fraction| fraction * seed_radius) {
        for _ in 0..2 {
            let mut improved = false;
            for corner in 0..4 {
                for axis in 0..2 {
                    let base = best;
                    for direction in [-1.0, 1.0] {
                        let mut quad = base.quad;
                        if axis == 0 {
                            quad.corners[corner].0 += direction * step;
                        } else {
                            quad.corners[corner].1 += direction * step;
                        }
                        let Some(candidate) = score_quad_with_layout(
                            frames,
                            base_counters,
                            session_tag,
                            quad,
                            minimum_offset,
                            maximum_offset,
                            3,
                            layout,
                        ) else {
                            continue;
                        };
                        if regularized_score(&candidate) > regularized_score(&best) {
                            best = candidate;
                            improved = true;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
        }
    }
    best
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

    fn synthetic_frame(
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        quad: ProjectiveQuad,
        counter: u16,
        session_tag: u8,
        frame: usize,
    ) -> Vec<u8> {
        synthetic_frame_with_layout(
            width,
            height,
            sensor_x,
            sensor_y,
            quad,
            counter,
            session_tag,
            frame,
            SpatialCodeLayout::LEGACY,
        )
    }

    fn synthetic_frame_with_layout(
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
        synthetic_frame_with_layout_and_corruption(
            width,
            height,
            sensor_x,
            sensor_y,
            quad,
            counter,
            session_tag,
            frame,
            layout,
            false,
        )
    }

    fn synthetic_frame_with_layout_and_corruption(
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        quad: ProjectiveQuad,
        counter: u16,
        session_tag: u8,
        frame: usize,
        layout: SpatialCodeLayout,
        corrupt_first_repeat: bool,
    ) -> Vec<u8> {
        let code = FrameCode::from_counter_mod(counter, session_tag);
        let signs = code.physical_signs();
        let inverse = inverse_3x3(quad.matrix().unwrap()).unwrap();
        let bases = [470.0, 590.0, 575.0, 420.0];
        let amplitudes = [27.0, -15.0, -14.0, 43.0];
        let exposure = 0.91 + frame as f64 * 0.012;
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
                let static_texture = 18.0 * (x as f64 * 0.071).sin()
                    + 13.0 * (y as f64 * 0.053).cos()
                    + 9.0 * ((x + y) as f64 * 0.021).sin();
                let mut value = bases[band] + static_texture;
                if let Some((u, v)) = map_matrix(inverse, x as f64 + 0.5, y as f64 + 0.5) {
                    if (0.0..1.0).contains(&u) && (0.0..1.0).contains(&v) {
                        let column = (u * layout.display_columns() as f64) as usize;
                        let row = (v * layout.display_rows() as f64) as usize;
                        let cell = layout
                            .canonical_cell(column, row)
                            .expect("synthetic coordinate is inside display lattice");
                        let repeat =
                            (row / GRID_ROWS) * layout.repeat_columns + column / GRID_COLUMNS;
                        if corrupt_first_repeat && repeat == 0 {
                            // Model a moving eyelid/glasses/glint contaminant
                            // that covers one complete spatial observation.
                            value += if frame.is_multiple_of(2) {
                                165.0
                            } else {
                                -165.0
                            };
                        } else {
                            value += amplitudes[band] * f64::from(signs[cell]);
                        }
                    }
                }
                let noise = (((x * 17 + y * 31 + frame * 13) % 11) as f64 - 5.0) * 0.7;
                values[y * width + x] =
                    (exposure * value + noise).round().clamp(1.0, 1023.0) as u16;
            }
        }
        pack_raw10(&values, width, height)
    }

    #[test]
    fn packed_accessor_preserves_every_ten_bit_lane_and_quad_band() {
        let width = 12;
        let height = 8;
        let values = (0..width * height)
            .map(|index| ((index * 37 + 11) & 1023) as u16)
            .collect::<Vec<_>>();
        let payload = pack_raw10(&values, width, height);
        let raw = PackedRaw10::new(&payload, width, height, width / 4 * 5, 2, 1).unwrap();
        for y in 0..height {
            for x in 0..width {
                assert_eq!(raw.pixel(x, y), values[y * width + x]);
            }
        }
        assert_eq!(raw.cfa_band(2, 3), 0);
        assert_eq!(raw.cfa_band(0, 0), 1);
    }

    #[test]
    fn sparse_native_registration_recovers_eye_translation_without_a_raster() {
        let width = 192;
        let height = 128;
        let sensor_x = 2u32;
        let sensor_y = 1u32;
        let shifts = [(0i32, 0i32), (2, -1), (4, -2), (6, -3), (8, -4)];
        let payloads = shifts
            .iter()
            .map(|(shift_x, shift_y)| {
                let values = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let source_x = x as i32 - shift_x;
                            let source_y = y as i32 - shift_y;
                            let checker =
                                ((source_x.div_euclid(13) + source_y.div_euclid(11)) & 1) as f64;
                            let texture = 120.0 * checker
                                + 62.0 * (source_x as f64 * 0.083).sin()
                                + 48.0 * (source_y as f64 * 0.067).cos();
                            (440.0 + texture).round().clamp(1.0, 1023.0) as u16
                        })
                    })
                    .collect::<Vec<_>>();
                pack_raw10(&values, width, height)
            })
            .collect::<Vec<_>>();
        let views = payloads
            .iter()
            .map(|payload| {
                PackedRaw10::new(payload, width, height, width / 4 * 5, sensor_x, sensor_y).unwrap()
            })
            .collect::<Vec<_>>();
        let transports = estimate_native_frame_translations(&views, (96.0, 64.0), 70.0);
        assert_eq!(transports.len(), shifts.len());
        for (transport, expected) in transports.iter().zip(shifts) {
            assert!(
                (transport.cumulative.0 - expected.0 as f64).abs() <= 1.0
                    && (transport.cumulative.1 - expected.1 as f64).abs() <= 1.0,
                "transport={transport:?} expected={expected:?} all={transports:?}"
            );
        }
    }

    #[test]
    fn interpolated_accessor_never_mixes_physical_cfa_planes() {
        let width = 24;
        let height = 24;
        let sensor_x = 2u32;
        let sensor_y = 1u32;
        let levels = [103u16, 277, 509, 881];
        let values = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let phase_x = (sensor_x as usize + x) & 3;
                    let phase_y = (sensor_y as usize + y) & 3;
                    let band = match (phase_y < 2, phase_x < 2) {
                        (true, true) => 0,
                        (true, false) => 1,
                        (false, true) => 2,
                        (false, false) => 3,
                    };
                    levels[band]
                })
            })
            .collect::<Vec<_>>();
        let payload = pack_raw10(&values, width, height);
        let raw =
            PackedRaw10::new(&payload, width, height, width / 4 * 5, sensor_x, sensor_y).unwrap();
        for (band, expected) in levels.into_iter().enumerate() {
            let sampled = raw.sample_band_bilinear(10.37, 11.62, band).unwrap();
            assert!((sampled - f64::from(expected)).abs() < 1.0e-9);
        }
    }

    #[test]
    fn projective_mapping_round_trips_unit_square() {
        let quad = ProjectiveQuad {
            corners: [(31.0, 28.0), (141.0, 21.0), (151.0, 104.0), (24.0, 111.0)],
        };
        for point in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.31, 0.77)] {
            let mapped = quad.map(point.0, point.1).unwrap();
            let restored = quad.inverse_map(mapped.0, mapped.1).unwrap();
            assert!((restored.0 - point.0).abs() < 1.0e-9);
            assert!((restored.1 - point.1).abs() < 1.0e-9);
        }
    }

    #[test]
    fn chromatic_clock_score_rejects_broadband_shadow_correlation() {
        let chromatic = spectral_code_score([0.30, -0.24, -0.22, 0.34], 1, 1.0);
        let common_mode_positive = spectral_code_score([0.30, 0.30, 0.30, 0.30], 1, 1.0)
            .max(spectral_code_score([0.30, 0.30, 0.30, 0.30], -1, 1.0));
        let common_mode_negative = spectral_code_score([-0.30, -0.30, -0.30, -0.30], 1, 1.0)
            .max(spectral_code_score([-0.30, -0.30, -0.30, -0.30], -1, 1.0));
        assert!(chromatic > 0.24, "chromatic={chromatic}");
        assert!(
            common_mode_positive < 0.0,
            "positive shadow={common_mode_positive}"
        );
        assert!(
            common_mode_negative < 0.0,
            "negative shadow={common_mode_negative}"
        );
    }

    #[test]
    fn rotated_lane_comb_prefers_the_repeated_chromatic_lattice() {
        let width = 224;
        let height = 160;
        let stride = width / 4 * 5;
        let session_tag = 6;
        let layout = SpatialCodeLayout::CURRENT;
        let quad = ProjectiveQuad {
            corners: [(45.0, 42.0), (185.0, 31.0), (192.0, 119.0), (36.0, 130.0)],
        };
        let payloads = (0..14)
            .map(|frame| {
                synthetic_frame_with_layout(
                    width,
                    height,
                    2,
                    1,
                    quad,
                    430 + frame as u16 * 3,
                    session_tag,
                    frame,
                    layout,
                )
            })
            .collect::<Vec<_>>();
        let views = payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let actual = score_temporal_lattice_lanes_with_layout(&views, quad, layout).unwrap();
        let displaced =
            score_temporal_lattice_lanes_with_layout(&views, quad.translated(-28.0, 18.0), layout)
                .unwrap();
        assert!(actual.score > 0.18, "actual={actual:?}");
        assert!(
            actual.score > displaced.score + 0.08,
            "actual={actual:?} displaced={displaced:?}"
        );
        assert!(actual.repeat_agreement > displaced.repeat_agreement);
    }

    #[test]
    fn temporal_native_cfa_fit_recovers_phase_and_rejects_wrong_geometry() {
        let width = 192;
        let height = 128;
        let stride = width / 4 * 5;
        let session_tag = 9;
        let quad = ProjectiveQuad {
            corners: [(47.0, 35.0), (151.0, 28.0), (161.0, 101.0), (39.0, 108.0)],
        };
        let actual = (0..16)
            .map(|frame| 610u16.wrapping_add(frame * 5))
            .collect::<Vec<_>>();
        let approximate = actual
            .iter()
            .map(|counter| (counter + 7) & 2047)
            .collect::<Vec<_>>();
        let payloads = actual
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                synthetic_frame(width, height, 2, 1, quad, *counter, session_tag, frame)
            })
            .collect::<Vec<_>>();
        let views = payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let observations = views
            .iter()
            .map(|view| sample_cell_spectra(*view, quad, 3))
            .collect::<Vec<_>>();
        let fit = fit_temporal_code(&observations, &approximate, session_tag, -12, 4).unwrap();
        assert_eq!(fit.counter_offset, -7, "{fit:#?}");
        assert!(fit.score > 0.72, "{fit:#?}");
        assert!(fit.band_correlations[0] > 0.75, "{fit:#?}");
        assert!(fit.band_correlations[1] < -0.65, "{fit:#?}");
        assert!(fit.band_correlations[2] < -0.65, "{fit:#?}");
        assert!(fit.band_correlations[3] > 0.80, "{fit:#?}");

        let wrong_quad = ProjectiveQuad::oriented_rectangle((96.0, 67.0), 54.0, 34.0, 0.0);
        let wrong = score_quad(&views, &approximate, session_tag, wrong_quad, -12, 4, 3).unwrap();
        assert!(
            fit.score > wrong.temporal.score + 0.20,
            "true={fit:#?} wrong={wrong:#?}"
        );
    }

    #[test]
    fn dense_spatial_repeats_collapse_to_the_canonical_native_decoder_lattice() {
        let width = 256;
        let height = 176;
        let stride = width / 4 * 5;
        let session_tag = 12;
        let layout = SpatialCodeLayout::CURRENT;
        let quad = ProjectiveQuad {
            corners: [(43.0, 29.0), (211.0, 24.0), (222.0, 148.0), (34.0, 154.0)],
        };
        let counters = (0..16)
            .map(|frame| 380u16.wrapping_add(frame * 5))
            .collect::<Vec<_>>();
        let payloads = counters
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                synthetic_frame_with_layout(
                    width,
                    height,
                    2,
                    1,
                    quad,
                    *counter,
                    session_tag,
                    frame,
                    layout,
                )
            })
            .collect::<Vec<_>>();
        let views = payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let observations = views
            .iter()
            .map(|view| sample_cell_spectra_with_layout(*view, quad, 3, layout))
            .collect::<Vec<_>>();
        assert!(observations
            .iter()
            .all(|observation| observation.support_fraction() > 0.99));
        let fit = fit_temporal_code(&observations, &counters, session_tag, 0, 0).unwrap();
        assert_eq!(fit.counter_offset, 0, "{fit:#?}");
        assert!(fit.score > 0.70, "{fit:#?}");

        let first_support = observations[0].support[0]
            .iter()
            .copied()
            .map(usize::from)
            .sum::<usize>();
        let legacy = sample_cell_spectra(views[0], quad, 3);
        let legacy_support = legacy.support[0]
            .iter()
            .copied()
            .map(usize::from)
            .sum::<usize>();
        assert!(
            first_support >= legacy_support * 3,
            "dense support={first_support} legacy support={legacy_support}"
        );

        let corrupted_payloads = counters
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                synthetic_frame_with_layout_and_corruption(
                    width,
                    height,
                    2,
                    1,
                    quad,
                    *counter,
                    session_tag,
                    frame,
                    layout,
                    true,
                )
            })
            .collect::<Vec<_>>();
        let corrupted_views = corrupted_payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let corrupted_observations = corrupted_views
            .iter()
            .map(|view| sample_cell_spectra_with_layout(*view, quad, 3, layout))
            .collect::<Vec<_>>();
        let corrupted_fit =
            fit_temporal_code(&corrupted_observations, &counters, session_tag, 0, 0).unwrap();
        assert!(corrupted_fit.score > 0.65, "{corrupted_fit:#?}");
    }

    #[test]
    fn dense_lattice_decodes_at_the_observed_ninety_by_thirty_five_pixel_scale() {
        let width = 192;
        let height = 128;
        let stride = width / 4 * 5;
        let session_tag = 7;
        let layout = SpatialCodeLayout::CURRENT;
        let quad = ProjectiveQuad {
            corners: [(49.0, 47.0), (141.0, 43.0), (145.0, 79.0), (46.0, 83.0)],
        };
        assert!((88.0..=102.0).contains(&quad.width()));
        assert!((34.0..=40.0).contains(&quad.height()));
        let counters = (0..20)
            .map(|frame| 1_240u16.wrapping_add(frame * 3))
            .collect::<Vec<_>>();
        let payloads = counters
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                synthetic_frame_with_layout(
                    width,
                    height,
                    2,
                    1,
                    quad,
                    *counter,
                    session_tag,
                    frame,
                    layout,
                )
            })
            .collect::<Vec<_>>();
        let views = payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let observations = views
            .iter()
            .map(|view| sample_cell_spectra_interpolated_with_layout(*view, quad, 5, layout))
            .collect::<Vec<_>>();
        let fit = fit_temporal_code(&observations, &counters, session_tag, 0, 0).unwrap();
        assert!(fit.support_fraction > 0.99, "{fit:#?}");
        assert!(fit.score > 0.60, "{fit:#?}");
        assert!(fit.band_correlations[3] > 0.62, "{fit:#?}");

        let corrupted_payloads = counters
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                synthetic_frame_with_layout_and_corruption(
                    width,
                    height,
                    2,
                    1,
                    quad,
                    *counter,
                    session_tag,
                    frame,
                    layout,
                    true,
                )
            })
            .collect::<Vec<_>>();
        let corrupted_views = corrupted_payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let corrupted_observations = corrupted_views
            .iter()
            .map(|view| sample_cell_spectra_interpolated_with_layout(*view, quad, 5, layout))
            .collect::<Vec<_>>();
        let corrupted_fit =
            fit_temporal_code(&corrupted_observations, &counters, session_tag, 0, 0).unwrap();
        assert!(corrupted_fit.score > 0.55, "{corrupted_fit:#?}");
    }

    #[test]
    fn bounded_native_search_recovers_projective_screen_from_a_rough_roi_seed() {
        let width = 192;
        let height = 128;
        let stride = width / 4 * 5;
        let session_tag = 6;
        let truth = ProjectiveQuad {
            corners: [(47.0, 35.0), (151.0, 28.0), (161.0, 101.0), (39.0, 108.0)],
        };
        let actual = (0..10)
            .map(|frame| 920u16.wrapping_add(frame * 7))
            .collect::<Vec<_>>();
        let approximate = actual
            .iter()
            .map(|counter| (counter + 5) & 2047)
            .collect::<Vec<_>>();
        let payloads = actual
            .iter()
            .enumerate()
            .map(|(frame, counter)| {
                synthetic_frame(width, height, 2, 1, truth, *counter, session_tag, frame)
            })
            .collect::<Vec<_>>();
        let views = payloads
            .iter()
            .map(|payload| PackedRaw10::new(payload, width, height, stride, 2, 1).unwrap())
            .collect::<Vec<_>>();
        let fit =
            search_reflection_quad(&views, &approximate, session_tag, (96.0, 64.0), 68.0, -9, 2)
                .unwrap();
        assert_eq!(fit.temporal.counter_offset, -5, "{fit:#?}");
        assert!(fit.temporal.score > 0.72, "{fit:#?}");
        assert!(
            distance(fit.quad.center(), truth.center()) < 7.0,
            "{fit:#?}"
        );
        assert!(
            (fit.quad.width() / truth.width() - 1.0).abs() < 0.18,
            "{fit:#?}"
        );
        assert!(
            (fit.quad.height() / truth.height() - 1.0).abs() < 0.22,
            "{fit:#?}"
        );
    }

    #[test]
    fn temporal_fit_preserves_sensor_coordinates_across_roi_reacquisition() {
        let width = 192;
        let height = 128;
        let stride = width / 4 * 5;
        let session_tag = 3;
        let reference_origin = (2u32, 1u32);
        let reference_quad = ProjectiveQuad {
            corners: [(47.0, 35.0), (151.0, 28.0), (161.0, 101.0), (39.0, 108.0)],
        };
        let origins = [
            (2, 1),
            (6, 1),
            (6, 5),
            (10, 5),
            (10, 9),
            (6, 9),
            (2, 9),
            (2, 5),
        ];
        let counters = (0..origins.len())
            .map(|frame| 1_120u16.wrapping_add(frame as u16 * 9))
            .collect::<Vec<_>>();
        let payloads = origins
            .iter()
            .zip(&counters)
            .enumerate()
            .map(|(frame, ((sensor_x, sensor_y), counter))| {
                let local_quad = reference_quad.translated(
                    f64::from(reference_origin.0) - f64::from(*sensor_x),
                    f64::from(reference_origin.1) - f64::from(*sensor_y),
                );
                synthetic_frame(
                    width,
                    height,
                    *sensor_x,
                    *sensor_y,
                    local_quad,
                    *counter,
                    session_tag,
                    frame,
                )
            })
            .collect::<Vec<_>>();
        let views = payloads
            .iter()
            .zip(origins)
            .map(|(payload, (sensor_x, sensor_y))| {
                PackedRaw10::new(payload, width, height, stride, sensor_x, sensor_y).unwrap()
            })
            .collect::<Vec<_>>();
        let fit = score_quad(&views, &counters, session_tag, reference_quad, 0, 0, 3).unwrap();
        assert!(fit.temporal.score > 0.72, "{fit:#?}");
        assert_eq!(fit.temporal.counter_offset, 0);
    }
}
