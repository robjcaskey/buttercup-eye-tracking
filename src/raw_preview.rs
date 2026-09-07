//! Shared native/Wasm RAW presentation. Callers validate RAW dimensions first.
pub fn percentile_range(values: &[u16]) -> (u16, u16) {
    let mut histogram = [0u32; 1024];
    for &value in values {
        histogram[value.min(1023) as usize] += 1;
    }
    let count = values.len() as u32;
    let low_target = count / 100;
    let high_target = count * 99 / 100;
    let mut cumulative = 0u32;
    let mut low = 0u16;
    let mut high = 1023u16;
    for (index, &hits) in histogram.iter().enumerate() {
        cumulative += hits;
        if cumulative >= low_target {
            low = index as u16;
            break;
        }
    }
    cumulative = 0;
    for (index, &hits) in histogram.iter().enumerate() {
        cumulative += hits;
        if cumulative >= high_target {
            high = index as u16;
            break;
        }
    }
    (low, high.max(low + 1))
}

pub fn quad_luma_preview(
    values: &[u16],
    width: usize,
    height: usize,
    contrast_percent: u16,
) -> Vec<u32> {
    let luma = (0..values.len())
        .map(|index| {
            let x = index % width;
            let y = index / width;
            let quad_x = x & !3;
            let quad_y = y & !3;
            let mut sum = 0u32;
            let mut count = 0u32;
            for yy in quad_y..(quad_y + 4).min(height) {
                for xx in quad_x..(quad_x + 4).min(width) {
                    sum += values[yy * width + xx] as u32;
                    count += 1;
                }
            }
            (sum / count.max(1)) as u16
        })
        .collect::<Vec<_>>();
    let (low, high) = percentile_range(&luma);
    luma.into_iter()
        .map(|value| {
            let baseline =
                value.saturating_sub(low) as f64 * 255.0 / high.saturating_sub(low).max(1) as f64;
            let gray = (128.0 + (baseline - 128.0) * contrast_percent as f64 / 100.0)
                .round()
                .clamp(0.0, 255.0) as u32;
            (gray << 16) | (gray << 8) | gray
        })
        .collect()
}

pub fn raw10_luma_preview(values: &[u16], contrast_percent: u16) -> Vec<u32> {
    let (low, high) = percentile_range(values);
    values
        .iter()
        .map(|&value| {
            let baseline =
                value.saturating_sub(low) as f64 * 255.0 / high.saturating_sub(low).max(1) as f64;
            let gray = (128.0 + (baseline - 128.0) * contrast_percent as f64 / 100.0)
                .round()
                .clamp(0.0, 255.0) as u32;
            (gray << 16) | (gray << 8) | gray
        })
        .collect()
}

pub fn raw10_color_preview(
    values: &[u16],
    width: usize,
    sensor_x: u32,
    sensor_y: u32,
    contrast_percent: u16,
) -> Vec<u32> {
    let (low, high) = percentile_range(values);
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let x = index % width;
            let y = index / width;
            let baseline =
                value.saturating_sub(low) as f64 * 255.0 / high.saturating_sub(low).max(1) as f64;
            let level = (128.0 + (baseline - 128.0) * contrast_percent as f64 / 100.0)
                .round()
                .clamp(0.0, 255.0) as u32;
            let quad_x_even = (((x as u32 + sensor_x) / 2) & 1) == 0;
            let quad_y_even = (((y as u32 + sensor_y) / 2) & 1) == 0;
            match (quad_y_even, quad_x_even) {
                (true, true) => level << 16,
                (false, false) => level,
                _ => level << 8,
            }
        })
        .collect()
}

#[derive(Default)]
pub struct DisplayColorBalance {
    white_balance: Option<[f64; 3]>,
    luma_range: Option<[f64; 2]>,
}

impl DisplayColorBalance {
    fn smooth_white_balance(&mut self, candidate: [f64; 3]) -> [f64; 3] {
        // CFA channel means are noisy in a small eye ROI. A per-frame white
        // balance makes that noise look like an illuminant toggling on and off,
        // so retain a display-only estimate across frames and reconnects.
        const ALPHA: f64 = 0.02;
        let mut balance = self.white_balance.unwrap_or(candidate);
        for channel in 0..3 {
            balance[channel] += (candidate[channel] - balance[channel]) * ALPHA;
        }
        self.white_balance = Some(balance);
        balance
    }

    fn smooth_luma_range(&mut self, low: u16, high: u16) -> (f64, f64) {
        // Percentile endpoints also move as the eyelid and pupil cross the ROI.
        // Follow real illumination/exposure changes over several frames instead
        // of renormalizing the entire preview for every image.
        const ALPHA: f64 = 0.05;
        let candidate = [low as f64, high as f64];
        let mut range = self.luma_range.unwrap_or(candidate);
        for endpoint in 0..2 {
            range[endpoint] += (candidate[endpoint] - range[endpoint]) * ALPHA;
        }
        if range[1] < range[0] + 1.0 {
            range[1] = range[0] + 1.0;
        }
        self.luma_range = Some(range);
        (range[0], range[1])
    }
}

pub fn color_preview(
    values: &[u16],
    width: usize,
    height: usize,
    sensor_x: u32,
    sensor_y: u32,
    contrast_percent: u16,
    mut display_balance: Option<&mut DisplayColorBalance>,
) -> Vec<u32> {
    if width < 4 || height < 4 || width & 1 != 0 || height & 1 != 0 {
        return quad_luma_preview(values, width, height, contrast_percent);
    }

    // IMX582 full-resolution 1x1 readout is Quad Bayer: each conventional
    // RGGB sample is a physical 2x2 same-color group. Treating adjacent sensor
    // pixels as ordinary RGGB creates the visible every-other-pixel lattice.
    // Average each native color group, demosaic the resulting half-resolution
    // RGGB plane, then bilinearly enlarge only this display copy. Tracking,
    // autofocus, recording, and transport continue to use untouched RAW10.
    let quad_width = width / 2;
    let quad_height = height / 2;
    let mut quad = vec![0.0; quad_width * quad_height];
    for quad_y in 0..quad_height {
        for quad_x in 0..quad_width {
            let x = quad_x * 2;
            let y = quad_y * 2;
            quad[quad_y * quad_width + quad_x] = (values[y * width + x] as f64
                + values[y * width + x + 1] as f64
                + values[(y + 1) * width + x] as f64
                + values[(y + 1) * width + x + 1] as f64)
                * 0.25;
        }
    }

    let at = |x: isize, y: isize| -> f64 {
        let x = x.clamp(0, quad_width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, quad_height.saturating_sub(1) as isize) as usize;
        quad[y * quad_width + x]
    };
    let mut quad_rgb = Vec::with_capacity(quad.len());
    for y in 0..quad_height {
        for x in 0..quad_width {
            let xe = ((x as u32 + sensor_x / 2) & 1) == 0;
            let ye = ((y as u32 + sensor_y / 2) & 1) == 0;
            let c = at(x as isize, y as isize);
            let horizontal =
                (at(x as isize - 1, y as isize) + at(x as isize + 1, y as isize)) * 0.5;
            let vertical = (at(x as isize, y as isize - 1) + at(x as isize, y as isize + 1)) * 0.5;
            let diagonal = (at(x as isize - 1, y as isize - 1)
                + at(x as isize + 1, y as isize - 1)
                + at(x as isize - 1, y as isize + 1)
                + at(x as isize + 1, y as isize + 1))
                * 0.25;
            quad_rgb.push(match (ye, xe) {
                (true, true) => [c, (horizontal + vertical) * 0.5, diagonal],
                (true, false) => [horizontal, c, vertical],
                (false, true) => [vertical, c, horizontal],
                (false, false) => [diagonal, (horizontal + vertical) * 0.5, c],
            });
        }
    }

    let mut channel_means = [0.0; 3];
    for pixel in &quad_rgb {
        for channel in 0..3 {
            channel_means[channel] += pixel[channel];
        }
    }
    for mean in &mut channel_means {
        *mean /= quad_rgb.len().max(1) as f64;
    }
    let white_balance_candidate = [
        (channel_means[1] / channel_means[0].max(1.0)).clamp(0.25, 4.0),
        1.0,
        (channel_means[1] / channel_means[2].max(1.0)).clamp(0.25, 4.0),
    ];
    let white_balance = display_balance
        .as_deref_mut()
        .map(|balance| balance.smooth_white_balance(white_balance_candidate))
        .unwrap_or(white_balance_candidate);

    // Keep chroma deliberately lower-bandwidth than luma. Raw sensor noise in
    // each same-color group otherwise survives as a larger Quad Bayer lattice
    // even after the CFA phase itself is decoded correctly.
    let mut quad_chroma = vec![[0.0; 3]; quad_rgb.len()];
    const CHROMA_RADIUS: isize = 2;
    for y in 0..quad_height {
        for x in 0..quad_width {
            let mut sum = [0.0; 3];
            let mut samples = 0.0;
            for dy in -CHROMA_RADIUS..=CHROMA_RADIUS {
                let yy = (y as isize + dy).clamp(0, quad_height as isize - 1) as usize;
                for dx in -CHROMA_RADIUS..=CHROMA_RADIUS {
                    let xx = (x as isize + dx).clamp(0, quad_width as isize - 1) as usize;
                    let pixel = quad_rgb[yy * quad_width + xx];
                    for channel in 0..3 {
                        sum[channel] += pixel[channel] * white_balance[channel];
                    }
                    samples += 1.0;
                }
            }
            let local_luma = ((sum[0] + sum[1] * 2.0 + sum[2]) * 0.25 / samples).max(1.0);
            quad_chroma[y * quad_width + x] = [
                sum[0] / samples / local_luma,
                sum[1] / samples / local_luma,
                sum[2] / samples / local_luma,
            ];
        }
    }

    let quad_pixel = |x: isize, y: isize| -> [f64; 3] {
        let x = x.clamp(0, quad_width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, quad_height.saturating_sub(1) as isize) as usize;
        quad_chroma[y * quad_width + x]
    };

    let integral_stride = width + 1;
    let mut integral = vec![0u32; integral_stride * (height + 1)];
    for y in 0..height {
        let mut row_sum = 0u32;
        for x in 0..width {
            row_sum += values[y * width + x] as u32;
            integral[(y + 1) * integral_stride + x + 1] =
                integral[y * integral_stride + x + 1] + row_sum;
        }
    }
    let neutral_luma = |x: usize, y: usize| -> f64 {
        let x0 = x.saturating_sub(1).min(width - 4);
        let y0 = y.saturating_sub(1).min(height - 4);
        let x1 = x0 + 4;
        let y1 = y0 + 4;
        let sum = integral[y1 * integral_stride + x1] + integral[y0 * integral_stride + x0]
            - integral[y0 * integral_stride + x1]
            - integral[y1 * integral_stride + x0];
        sum as f64 / 16.0
    };
    let mut rgb = Vec::with_capacity(values.len());
    let mut luma = Vec::with_capacity(values.len());
    for y in 0..height {
        let source_y = (y as f64 + 0.5) * 0.5 - 0.5;
        let y0 = source_y.floor() as isize;
        let fy = source_y - y0 as f64;
        for x in 0..width {
            let source_x = (x as f64 + 0.5) * 0.5 - 0.5;
            let x0 = source_x.floor() as isize;
            let fx = source_x - x0 as f64;
            let p00 = quad_pixel(x0, y0);
            let p10 = quad_pixel(x0 + 1, y0);
            let p01 = quad_pixel(x0, y0 + 1);
            let p11 = quad_pixel(x0 + 1, y0 + 1);
            let mut pixel = [0.0; 3];
            let local_luma = neutral_luma(x, y);
            for channel in 0..3 {
                pixel[channel] = ((p00[channel] * (1.0 - fx) + p10[channel] * fx) * (1.0 - fy)
                    + (p01[channel] * (1.0 - fx) + p11[channel] * fx) * fy)
                    * local_luma;
            }
            luma.push(local_luma as u16);
            rgb.push(pixel);
        }
    }
    let (candidate_low, candidate_high) = percentile_range(&luma);
    let (low, high) = display_balance
        .as_deref_mut()
        .map(|balance| balance.smooth_luma_range(candidate_low, candidate_high))
        .unwrap_or((candidate_low as f64, candidate_high as f64));
    let span = (high - low).max(1.0);
    rgb.into_iter()
        .map(|pixel| {
            let channel = |value: f64| {
                let baseline = (value - low) * 255.0 / span;
                (128.0 + (baseline - 128.0) * contrast_percent as f64 / 100.0)
                    .round()
                    .clamp(0.0, 255.0) as u32
            };
            (channel(pixel[0]) << 16) | (channel(pixel[1]) << 8) | channel(pixel[2])
        })
        .collect()
}

