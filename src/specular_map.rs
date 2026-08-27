//! Motion-compensated diagnostic decomposition of a live color eye crop.
//!
//! The camera does not provide a measured polarization channel.  These views
//! are therefore estimates: a neutral, locally bright component is treated as
//! specular; the cross-polarized estimate subtracts that neutral component;
//! and the diffuse estimate fills it from the motion-registered prior surface
//! (falling back to a local low-pass color on the first/unregistered frame).

#[derive(Clone, Copy, Debug, Default)]
pub struct SimilarityMotion {
    pub translation: [f32; 2],
    pub rotation: f32,
    pub scale_delta: f32,
}

#[derive(Clone, Debug)]
pub struct SpecularViews {
    pub specular_map: Vec<u32>,
    pub cross_polarized: Vec<u32>,
    pub diffuse: Vec<u32>,
    pub motion_compensated: bool,
}

impl SpecularViews {
    fn blank(pixel_count: usize) -> Self {
        Self {
            specular_map: vec![0; pixel_count],
            cross_polarized: vec![0; pixel_count],
            diffuse: vec![0; pixel_count],
            motion_compensated: false,
        }
    }
}

#[derive(Clone)]
struct PreviousDiffuse {
    sensor_x: u32,
    sensor_y: u32,
    width: usize,
    height: usize,
    pixels: Vec<[f32; 3]>,
}

#[derive(Default)]
pub struct SpecularMapTracker {
    previous: Option<PreviousDiffuse>,
}

fn smoothstep(low: f32, high: f32, value: f32) -> f32 {
    let t = ((value - low) / (high - low).max(1.0e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn luma(pixel: [f32; 3]) -> f32 {
    pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722
}

fn unpack(pixel: u32) -> [f32; 3] {
    [
        ((pixel >> 16) & 0xff) as f32,
        ((pixel >> 8) & 0xff) as f32,
        (pixel & 0xff) as f32,
    ]
}

fn pack(pixel: [f32; 3]) -> u32 {
    let channel = |value: f32| value.round().clamp(0.0, 255.0) as u32;
    (channel(pixel[0]) << 16) | (channel(pixel[1]) << 8) | channel(pixel[2])
}

fn percentile(histogram: &[u32; 256], count: usize, quantile: f32) -> f32 {
    if count == 0 {
        return 0.0;
    }
    let target = ((count - 1) as f32 * quantile.clamp(0.0, 1.0)).round() as u32;
    let mut cumulative = 0u32;
    for (value, hits) in histogram.iter().enumerate() {
        cumulative = cumulative.saturating_add(*hits);
        if cumulative > target {
            return value as f32;
        }
    }
    255.0
}

fn build_integral(pixels: &[[f32; 3]], width: usize, height: usize) -> Vec<[f32; 4]> {
    let stride = width + 1;
    let mut integral = vec![[0.0; 4]; stride * (height + 1)];
    for y in 0..height {
        let mut row = [0.0f32; 4];
        for x in 0..width {
            let pixel = pixels[y * width + x];
            row[0] += luma(pixel);
            row[1] += pixel[0];
            row[2] += pixel[1];
            row[3] += pixel[2];
            let above = integral[y * stride + x + 1];
            integral[(y + 1) * stride + x + 1] = std::array::from_fn(|i| above[i] + row[i]);
        }
    }
    integral
}

fn box_average(
    integral: &[[f32; 4]],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
) -> [f32; 4] {
    let stride = width + 1;
    let x0 = x.saturating_sub(radius);
    let y0 = y.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let y1 = (y + radius + 1).min(height);
    let count = ((x1 - x0) * (y1 - y0)).max(1) as f32;
    std::array::from_fn(|channel| {
        (integral[y1 * stride + x1][channel] + integral[y0 * stride + x0][channel]
            - integral[y0 * stride + x1][channel]
            - integral[y1 * stride + x0][channel])
            / count
    })
}

fn inverse_motion_point(
    current_sensor: [f32; 2],
    current_center: [f32; 2],
    motion: SimilarityMotion,
) -> Option<[f32; 2]> {
    let a = 1.0 + motion.scale_delta;
    let b = motion.rotation;
    let determinant = a * a + b * b;
    if !determinant.is_finite() || determinant < 0.25 {
        return None;
    }
    let x = current_sensor[0] - current_center[0] - motion.translation[0];
    let y = current_sensor[1] - current_center[1] - motion.translation[1];
    Some([
        current_center[0] + (a * x + b * y) / determinant,
        current_center[1] + (-b * x + a * y) / determinant,
    ])
}

fn bilinear_sample(frame: &PreviousDiffuse, sensor_point: [f32; 2]) -> Option<[f32; 3]> {
    let x = sensor_point[0] - frame.sensor_x as f32;
    let y = sensor_point[1] - frame.sensor_y as f32;
    if x < 0.0
        || y < 0.0
        || x > frame.width.saturating_sub(1) as f32
        || y > frame.height.saturating_sub(1) as f32
    {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(frame.width - 1);
    let y1 = (y0 + 1).min(frame.height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let p00 = frame.pixels[y0 * frame.width + x0];
    let p10 = frame.pixels[y0 * frame.width + x1];
    let p01 = frame.pixels[y1 * frame.width + x0];
    let p11 = frame.pixels[y1 * frame.width + x1];
    Some(std::array::from_fn(|channel| {
        (p00[channel] * (1.0 - fx) + p10[channel] * fx) * (1.0 - fy)
            + (p01[channel] * (1.0 - fx) + p11[channel] * fx) * fy
    }))
}

impl SpecularMapTracker {
    pub fn observe(
        &mut self,
        color: &[u32],
        width: usize,
        height: usize,
        sensor_x: u32,
        sensor_y: u32,
        motion: Option<SimilarityMotion>,
    ) -> SpecularViews {
        let pixel_count = width.saturating_mul(height);
        if width < 3 || height < 3 || color.len() < pixel_count {
            self.previous = None;
            return SpecularViews::blank(pixel_count);
        }

        let pixels = color
            .iter()
            .take(pixel_count)
            .map(|pixel| unpack(*pixel))
            .collect::<Vec<_>>();
        let mut histogram = [0u32; 256];
        for pixel in &pixels {
            histogram[luma(*pixel).round().clamp(0.0, 255.0) as usize] += 1;
        }
        let baseline = percentile(&histogram, pixel_count, 0.62);
        let peak = percentile(&histogram, pixel_count, 0.985);
        let amplitude = (peak - baseline).max(18.0);
        let threshold = baseline + amplitude * 0.48;
        let full = (threshold + amplitude * 0.38).max(threshold + 8.0);
        let integral = build_integral(&pixels, width, height);
        let previous = self.previous.as_ref();
        let current_center = [
            sensor_x as f32 + width as f32 * 0.5,
            sensor_y as f32 + height as f32 * 0.5,
        ];

        let mut specular_map = Vec::with_capacity(pixel_count);
        let mut cross_polarized = Vec::with_capacity(pixel_count);
        let mut diffuse = Vec::with_capacity(pixel_count);
        let mut diffuse_linear = Vec::with_capacity(pixel_count);
        let mut used_motion = false;
        for y in 0..height {
            for x in 0..width {
                let index = y * width + x;
                let pixel = pixels[index];
                let pixel_luma = luma(pixel);
                let local = box_average(&integral, width, height, x, y, 4);
                let local_rgb = [local[1], local[2], local[3]];
                let maximum = pixel[0].max(pixel[1]).max(pixel[2]);
                let minimum = pixel[0].min(pixel[1]).min(pixel[2]);
                let chroma = maximum - minimum;
                let neutral = 1.0 - smoothstep(24.0, 96.0, chroma);
                let globally_bright = smoothstep(threshold, full, pixel_luma);
                let locally_bright = smoothstep(3.0, 28.0, pixel_luma - local[0]);
                let mut score = globally_bright * neutral * (0.20 + 0.80 * locally_bright);

                let prior = motion.and_then(|motion| {
                    let previous = previous?;
                    let current_sensor = [sensor_x as f32 + x as f32, sensor_y as f32 + y as f32];
                    let previous_sensor =
                        inverse_motion_point(current_sensor, current_center, motion)?;
                    bilinear_sample(previous, previous_sensor)
                });
                if let Some(prior) = prior {
                    used_motion = true;
                    let temporal_outlier = smoothstep(6.0, 42.0, pixel_luma - luma(prior));
                    score = score.max(globally_bright * neutral * (0.25 + 0.75 * temporal_outlier));
                }
                score = score.clamp(0.0, 1.0);

                let local_neutral = local_rgb[0].min(local_rgb[1]).min(local_rgb[2]);
                let neutral_excess = (minimum - local_neutral).max(0.0);
                let global_excess = (pixel_luma - threshold).max(0.0) * 0.55;
                let removed = score * neutral_excess.max(global_excess).min(minimum);
                let cross = std::array::from_fn(|channel| (pixel[channel] - removed).max(0.0));

                let replacement = prior.map_or(local_rgb, |prior| {
                    std::array::from_fn(|channel| prior[channel] * 0.82 + local_rgb[channel] * 0.18)
                });
                let diffuse_pixel = std::array::from_fn(|channel| {
                    pixel[channel] * (1.0 - score) + replacement[channel] * score
                });
                let map_level = (score.sqrt() * 255.0).round() as u32;
                specular_map.push((map_level << 16) | (map_level << 8) | map_level);
                cross_polarized.push(pack(cross));
                diffuse.push(pack(diffuse_pixel));
                diffuse_linear.push(diffuse_pixel);
            }
        }

        self.previous = Some(PreviousDiffuse {
            sensor_x,
            sensor_y,
            width,
            height,
            pixels: diffuse_linear,
        });
        SpecularViews {
            specular_map,
            cross_polarized,
            diffuse,
            motion_compensated: used_motion,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(pixel: u32) -> u32 {
        pixel & 0xff
    }

    #[test]
    fn compact_neutral_highlight_isolated_from_a_flat_diffuse_field() {
        let width = 32;
        let height = 24;
        let mut color = vec![0x003c_3c3c; width * height];
        let highlight = 12 * width + 16;
        color[highlight] = 0x00ff_ffff;
        let views = SpecularMapTracker::default().observe(&color, width, height, 0, 0, None);

        assert!(gray(views.specular_map[highlight]) > 220);
        assert_eq!(gray(views.specular_map[0]), 0);
        assert!(gray(views.cross_polarized[highlight]) < 100);
        assert!(gray(views.diffuse[highlight]) < 100);
    }

    #[test]
    fn motion_registered_prior_fills_a_shifted_highlight_with_surface_color() {
        let width = 32;
        let height = 24;
        let mut first = vec![0x0028_3038; width * height];
        first[12 * width + 10] = 0x0078_4020;
        let mut tracker = SpecularMapTracker::default();
        tracker.observe(&first, width, height, 100, 200, None);

        let mut second = vec![0x0028_3038; width * height];
        second[12 * width + 12] = 0x00ff_ffff;
        let views = tracker.observe(
            &second,
            width,
            height,
            100,
            200,
            Some(SimilarityMotion {
                translation: [2.0, 0.0],
                ..SimilarityMotion::default()
            }),
        );

        assert!(views.motion_compensated);
        let reconstructed = views.diffuse[12 * width + 12];
        let red = (reconstructed >> 16) & 0xff;
        let blue = reconstructed & 0xff;
        assert!(red > blue + 35, "reconstructed={reconstructed:#08x}");
    }
}
