//! Experimental, image-only chromatic ridge proposals. These are not sclera
//! labels, persistent identities, or gaze measurements. Kept out of live use
//! until native-source overlays and landmark correspondences support it.

#[derive(Clone, Copy, Debug)]
pub struct Ridge {
    pub point: [f32; 2],
    pub tangent: [f32; 2],
    pub contrast: f32,
}

fn shifted(input: &[f32], w: usize, x: usize, y: usize, dx: isize, dy: isize) -> f32 {
    let i = y * w + x;
    let ix = y * w + x.checked_add_signed(dx).unwrap();
    let iy = y.checked_add_signed(dy).unwrap() * w + x;
    let ixy = y.checked_add_signed(dy).unwrap() * w + x.checked_add_signed(dx).unwrap();
    // All four physical CFA blocks are interpolated by the same quarter-cell
    // distance to a common sensor location; raw colour phases are not treated
    // as co-located measurements.
    0.5625 * input[i] + 0.1875 * (input[ix] + input[iy]) + 0.0625 * input[ixy]
}

pub fn detect(raw: &[u16], width: usize, height: usize, sx: u32, sy: u32) -> Vec<Ridge> {
    let ox = (4 - sx as usize % 4) % 4;
    let oy = (4 - sy as usize % 4) % 4;
    let w = width.saturating_sub(ox) / 4;
    let h = height.saturating_sub(oy) / 4;
    if w < 14 || h < 14 || raw.len() < width.saturating_mul(height) {
        return Vec::new();
    }
    let mut physical = [
        vec![0.; w * h],
        vec![0.; w * h],
        vec![0.; w * h],
        vec![0.; w * h],
    ];
    for y in 0..h {
        for x in 0..w {
            for (channel, (dx, dy)) in [(0, 0), (2, 0), (0, 2), (2, 2)].into_iter().enumerate() {
                let i = (oy + 4 * y + dy) * width + ox + 4 * x + dx;
                physical[channel][y * w + x] = (raw[i] as f32
                    + raw[i + 1] as f32
                    + raw[i + width] as f32
                    + raw[i + width + 1] as f32)
                    * 0.25;
            }
        }
    }
    let mut luma = vec![0.; w * h];
    let mut green = vec![0.; w * h];
    let mut pigment = vec![0.; w * h];
    let mut rg = vec![0.; w * h];
    let mut brightness = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let r = shifted(&physical[0], w, x, y, 1, 1);
            let g = (shifted(&physical[1], w, x, y, -1, 1) + shifted(&physical[2], w, x, y, 1, -1))
                * 0.5;
            let b = shifted(&physical[3], w, x, y, -1, -1);
            let i = y * w + x;
            luma[i] = (r + 2. * g + b) * 0.25;
            green[i] = g;
            let eps = (luma[i] * 0.02).max(1.);
            rg[i] = ((r + eps) / (g + eps)).ln();
            pigment[i] = 0.5 * (rg[i] + ((r + eps) / (b + eps)).ln());
            brightness.push(luma[i]);
        }
    }
    let q = brightness.len() * 65 / 100;
    let floor = *brightness.select_nth_unstable_by(q, f32::total_cmp).1;
    // Reject alternating colour fringes introduced by staggered sampling of
    // an achromatic narrow ridge. Chromatic evidence must also survive local
    // integration; the unsmoothed green samples still localize its dark core.
    let smooth = |input: &[f32]| {
        let mut output = vec![0.; w * h];
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        let weight = if dx == 0 { 2. } else { 1. } * if dy == 0 { 2. } else { 1. };
                        output[y * w + x] += input[y.checked_add_signed(dy).unwrap() * w
                            + x.checked_add_signed(dx).unwrap()]
                            * weight
                            / 16.;
                    }
                }
            }
        }
        output
    };
    let pigment = smooth(&pigment);
    let rg = smooth(&rg);
    let normals = [(1isize, 0isize), (1, 1), (0, 1), (-1, 1)];
    let mut response = vec![0f32; w * h];
    let mut direction = vec![0usize; w * h];
    for y in 4..h - 4 {
        for x in 4..w - 4 {
            let i = y * w + x;
            for (k, &(dx, dy)) in normals.iter().enumerate() {
                for radius in 1..=2 {
                    let a = y.checked_add_signed(radius * dy).unwrap() * w
                        + x.checked_add_signed(radius * dx).unwrap();
                    let b = y.checked_add_signed(-radius * dy).unwrap() * w
                        + x.checked_add_signed(-radius * dx).unwrap();
                    // A one-sided colour/lid edge is not a vessel ridge. Both
                    // flanks must support a brighter background and less pigment.
                    if luma[a].min(luma[b]) < floor * 0.94 || luma[a].max(luma[b]) >= 1010. {
                        continue;
                    }
                    if green[a].min(green[b]) < green[i] + 0.003 * green[i] {
                        continue;
                    }
                    let contrast = (pigment[i] - pigment[a]).min(pigment[i] - pigment[b]);
                    let red_green = (rg[i] - rg[a]).min(rg[i] - rg[b]);
                    if contrast >= 0.008 && red_green >= 0.003 && contrast > response[i] {
                        response[i] = contrast;
                        direction[i] = k;
                    }
                }
            }
        }
    }
    let mut ranked = Vec::new();
    for y in 5..h - 5 {
        for x in 5..w - 5 {
            let i = y * w + x;
            if response[i] == 0. {
                continue;
            }
            let (dx, dy) = normals[direction[i]];
            let index = |dx: isize, dy: isize| {
                y.checked_add_signed(dy).unwrap() * w + x.checked_add_signed(dx).unwrap()
            };
            if response[i] < response[index(dx, dy)] || response[i] < response[index(-dx, -dy)] {
                continue;
            }
            let supported = [index(-dy, dx), index(dy, -dx)]
                .into_iter()
                .filter(|&j| response[j] >= 0.008 && direction[i].abs_diff(direction[j]) != 2)
                .count();
            if supported == 0 {
                continue;
            }
            ranked.push((i, response[i] * (1. + supported as f32 * 0.25)));
        }
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let columns = width.div_ceil(32);
    let mut tiles = vec![0u8; columns * height.div_ceil(32)];
    let mut points: Vec<Ridge> = Vec::new();
    // Spatial balance keeps many strong skin features from consuming all of
    // the proposal budget. This does not itself classify the surviving tissue.
    for limit in 1..=2 {
        for &(i, _) in &ranked {
            let point = [
                (ox + 4 * (i % w)) as f32 + 1.5,
                (oy + 4 * (i / w)) as f32 + 1.5,
            ];
            let tile = (point[1] as usize / 32) * columns + point[0] as usize / 32;
            if tiles[tile] >= limit
                || points
                    .iter()
                    .any(|p| (p.point[0] - point[0]).hypot(p.point[1] - point[1]) < 7.)
            {
                continue;
            }
            let (nx, ny) = normals[direction[i]];
            let length = (nx as f32).hypot(ny as f32);
            points.push(Ridge {
                point,
                tangent: [-ny as f32 / length, nx as f32 / length],
                contrast: response[i],
            });
            tiles[tile] += 1;
            if points.len() >= 80 {
                return points;
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    fn raw_pattern(sx: u32, sy: u32, dx: f32, colored: bool, step: bool) -> Vec<u16> {
        (0..128 * 96)
            .map(|i| {
                let x = (i % 128) as f32;
                let position = x - dx;
                let line = if step {
                    if position > 60. {
                        1.
                    } else {
                        0.
                    }
                } else {
                    (-0.5 * ((position - 60.) / 3.).powi(2)).exp()
                };
                let red = ((i % 128) as u32 + sx) % 4 < 2 && ((i / 128) as u32 + sy) % 4 < 2;
                (700.
                    - if red && colored {
                        10. * line
                    } else {
                        140. * line
                    }) as u16
            })
            .collect()
    }
    #[test]
    fn narrow_colored_ridge_survives_translation_and_roi_cfa_phase() {
        for (sx, sy) in [(0, 0), (1, 3), (3, 1)] {
            for shift in [0., 12.] {
                let p = detect(&raw_pattern(sx, sy, shift, true, false), 128, 96, sx, sy);
                assert!(!p.is_empty(), "phase {sx},{sy}, shift {shift}");
                assert!(
                    p.iter().all(|p| (p.point[0] - 60. - shift).abs() < 3.),
                    "{p:?}"
                );
            }
        }
    }
    #[test]
    fn single_color_step_and_neutral_line_do_not_masquerade_as_vessels() {
        assert!(detect(&raw_pattern(0, 0, 0., true, true), 128, 96, 0, 0).is_empty());
        assert!(detect(&raw_pattern(0, 0, 0., false, false), 128, 96, 0, 0).is_empty());
    }
}
