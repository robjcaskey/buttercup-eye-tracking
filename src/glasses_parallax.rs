//! Non-learned RAW corner-copy and temporal-motion evidence. Layers are anonymous.
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<f64>,
    pub green: Vec<f64>,
}
pub type Point = [f64; 2];
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
fn norm(a: Point) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}
pub fn median(mut a: Vec<f64>) -> f64 {
    if a.is_empty() {
        return 0.;
    }
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}
fn median_point(a: &[Point]) -> Point {
    [
        median(a.iter().map(|x| x[0]).collect()),
        median(a.iter().map(|x| x[1]).collect()),
    ]
}

/// LE40 packs four successive 10-bit words, unlike MIPI RAW10. Average each
/// complete CFA cell; green-only replicate provides an independent mosaic check.
pub fn decode(
    bytes: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    sensor_x: usize,
    sensor_y: usize,
) -> Result<Image, String> {
    if w % 4 != 0
        || h % 2 != 0
        || stride < w * 5 / 4
        || stride.checked_mul(h) != Some(bytes.len())
        || w < 32
        || h < 32
    {
        return Err("invalid LE40 shape/stride/length".into());
    }
    let mut raw = vec![0.; w * h];
    for y in 0..h {
        for x in (0..w).step_by(4) {
            let mut word = 0u64;
            for k in 0..5 {
                word |= (bytes[y * stride + x / 4 * 5 + k] as u64) << (8 * k);
            }
            for k in 0..4 {
                raw[y * w + x + k] = ((word >> (10 * k)) & 1023) as f64;
            }
        }
    }
    let start_x = (4 - sensor_x % 4) % 4;
    let start_y = (4 - sensor_y % 4) % 4;
    let ww = (w - start_x) / 4;
    let hh = (h - start_y) / 4;
    let mut pixels = Vec::with_capacity(ww * hh);
    let mut green = Vec::with_capacity(ww * hh);
    for cy in 0..hh {
        for cx in 0..ww {
            let x = start_x + cx * 4;
            let y = start_y + cy * 4;
            let block = |dx: usize, dy: usize| {
                (raw[(y + dy) * w + x + dx]
                    + raw[(y + dy) * w + x + dx + 1]
                    + raw[(y + dy + 1) * w + x + dx]
                    + raw[(y + dy + 1) * w + x + dx + 1])
                    / 4.
            };
            let g = (block(2, 0) + block(0, 2)) / 2.;
            pixels.push((block(0, 0) + 2. * g + block(2, 2)) / 4.);
            green.push(g);
        }
    }
    // Symmetric smoothing after CFA reduction, never match native CFA microtexture.
    let smooth = |input: Vec<f64>| {
        let mut out = input.clone();
        for y in 1..hh - 1 {
            for x in 1..ww - 1 {
                out[y * ww + x] = (input[y * ww + x] * 4.
                    + (input[y * ww + x - 1]
                        + input[y * ww + x + 1]
                        + input[(y - 1) * ww + x]
                        + input[(y + 1) * ww + x])
                        * 2.
                    + input[(y - 1) * ww + x - 1]
                    + input[(y - 1) * ww + x + 1]
                    + input[(y + 1) * ww + x - 1]
                    + input[(y + 1) * ww + x + 1])
                    / 16.;
            }
        }
        out
    };
    Ok(Image {
        w: ww,
        h: hh,
        pixels: smooth(pixels),
        green: smooth(green),
    })
}
fn patch(image: &Image, p: Point, green: bool) -> Option<Vec<f64>> {
    let x = p[0].round() as isize;
    let y = p[1].round() as isize;
    if x < 5 || y < 5 || x + 5 >= image.w as isize || y + 5 >= image.h as isize {
        return None;
    }
    let pix = if green { &image.green } else { &image.pixels };
    let mut out = Vec::with_capacity(81);
    for dy in -4..=4 {
        for dx in -4..=4 {
            out.push(pix[(y + dy) as usize * image.w + (x + dx) as usize]);
        }
    }
    let mean = out.iter().sum::<f64>() / out.len() as f64;
    let energy = out.iter().map(|v| (v - mean).powi(2)).sum::<f64>().sqrt();
    if energy < 20. {
        return None;
    }
    for v in &mut out {
        *v = (*v - mean) / energy;
    }
    Some(out)
}
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn corners(image: &Image) -> Vec<Point> {
    let mut values = Vec::new();
    for y in (6..image.h - 6).step_by(2) {
        for x in (6..image.w - 6).step_by(2) {
            let mut xx = 0.;
            let mut yy = 0.;
            let mut xy = 0.;
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    let p = (y as isize + dy) as usize * image.w + (x as isize + dx) as usize;
                    let gx = image.pixels[p + 1] - image.pixels[p - 1];
                    let gy = image.pixels[p + image.w] - image.pixels[p - image.w];
                    xx += gx * gx;
                    yy += gy * gy;
                    xy += gx * gy;
                }
            }
            let score = (xx + yy - ((xx - yy).powi(2) + 4. * xy * xy).sqrt()) / 2.;
            if score > 80. {
                values.push((score, [x as f64, y as f64]));
            }
        }
    }
    values.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut out = Vec::new();
    for (_, p) in values {
        if out.iter().all(|&q| norm(sub(p, q)) >= 6.) {
            out.push(p);
            if out.len() == 96 {
                break;
            }
        }
    }
    out
}
fn locate(template: &[f64], image: &Image, center: Point) -> Option<(Point, f64)> {
    let mut best = None;
    for dy in -5..=5 {
        for dx in -5..=5 {
            let p = [center[0] + dx as f64, center[1] + dy as f64];
            if let Some(v) = patch(image, p, false) {
                let c = correlation(template, &v);
                if best.is_none_or(|(_, old)| c > old) {
                    best = Some((p, c));
                }
            }
        }
    }
    best.filter(|(_, c)| *c >= 0.88)
}
pub fn track(images: &[Image], start: Point) -> (Vec<Point>, Option<&'static str>) {
    let mut path = vec![start];
    for t in 1..images.len() {
        let previous = *path.last().unwrap();
        let Some(a) = patch(&images[t - 1], previous, false) else {
            return (path, Some("previous_patch_missing_or_flat"));
        };
        let Some((next, _)) = locate(&a, &images[t], previous) else {
            return (path, Some("forward_NCC_below_0.88_or_outside_ROI"));
        };
        let Some(b) = patch(&images[t], next, false) else {
            return (path, Some("next_patch_missing_or_flat"));
        };
        let Some((back, _)) = locate(&b, &images[t - 1], next) else {
            return (path, Some("reverse_NCC_below_0.88_or_outside_ROI"));
        };
        if norm(sub(back, previous)) > 1. {
            return (path, Some("forward_backward_error_above4native_px"));
        }
        path.push(next);
    }
    (path, None)
}

fn transform(m: [f64; 4], p: Point) -> Point {
    [
        m[0] * p[0] - m[1] * p[1] + m[2],
        m[1] * p[0] + m[0] * p[1] + m[3],
    ]
}
/// Robust shared image-plane translation/rotation/isotropic-scale fit. Its
/// source is anonymous texture, not an asserted anatomical/material surface.
fn common_similarity(seeds: &[Point], positions: &[Point]) -> ([f64; 4], usize) {
    if seeds.len() < 3 {
        return ([1., 0., 0., 0.], 0);
    }
    let mut ids: Vec<_> = (0..seeds.len()).collect();
    let mut m = [1., 0., 0., 0.];
    for _ in 0..4 {
        let mean = |points: &[Point]| {
            let mut s = [0., 0.];
            for &i in &ids {
                s[0] += points[i][0];
                s[1] += points[i][1];
            }
            [s[0] / ids.len() as f64, s[1] / ids.len() as f64]
        };
        let a = mean(seeds);
        let b = mean(positions);
        let mut den = 0.;
        let mut dot = 0.;
        let mut cross = 0.;
        for &i in &ids {
            let p = sub(seeds[i], a);
            let q = sub(positions[i], b);
            den += p[0] * p[0] + p[1] * p[1];
            dot += p[0] * q[0] + p[1] * q[1];
            cross += p[0] * q[1] - p[1] * q[0];
        }
        if den < 1e-6 {
            return ([1., 0., 0., 0.], 0);
        }
        m = [dot / den, cross / den, 0., 0.];
        let center = transform(m, a);
        m[2] = b[0] - center[0];
        m[3] = b[1] - center[1];
        ids = (0..seeds.len()).collect();
        ids.sort_by(|&a, &b| {
            norm(sub(transform(m, seeds[a]), positions[a]))
                .total_cmp(&norm(sub(transform(m, seeds[b]), positions[b])))
        });
        ids.truncate((seeds.len() * 3 / 4).max(3));
    }
    let inliers = (0..seeds.len())
        .filter(|&i| norm(sub(transform(m, seeds[i]), positions[i])) <= 1.)
        .count();
    (m, inliers)
}

/// Motion support uses chronological holdout, with an independently reordered
/// endpoint as a temporal null. All distances here are physical Quad-Bayer cell pixels.
pub fn motion_metrics(a: &[Point], b: &[Point], common: &[Point], times: &[u64]) -> Value {
    if a.len() < 12 || a.len() != b.len() || a.len() != common.len() || a.len() != times.len() {
        return json!({"differential_motion_supported":false,"reason":"short_or_missing_tracks"});
    }
    if times
        .windows(2)
        .any(|t| t[1] <= t[0] || t[1] - t[0] > 250_000_000)
    {
        return json!({"differential_motion_supported":false,"reason":"invalid_or_gapped_source_timestamps"});
    }
    let ar: Vec<_> = a
        .iter()
        .zip(common)
        .map(|(&p, &c)| sub(sub(p, a[0]), c))
        .collect();
    let br: Vec<_> = b
        .iter()
        .zip(common)
        .map(|(&p, &c)| sub(sub(p, b[0]), c))
        .collect();
    let velocities = |p: &[Point]| {
        p.windows(2)
            .zip(times.windows(2))
            .map(|(w, t)| {
                let d = sub(w[1], w[0]);
                let dt = (t[1] - t[0]) as f64 / 1e9;
                [d[0] / dt, d[1] / dt]
            })
            .collect::<Vec<_>>()
    };
    let av = velocities(&ar);
    let bv = velocities(&br);
    let split = av.len() / 2;
    let fit = |x: &[Point], y: &[Point]| {
        let den = x.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum::<f64>();
        x.iter()
            .zip(y)
            .map(|(a, b)| a[0] * b[0] + a[1] * b[1])
            .sum::<f64>()
            / (den + 1e-6)
    };
    let gain = fit(&av[..split], &bv[..split]);
    let score = |x: &[Point], y: &[Point]| {
        let den = y.iter().map(|a| a[0] * a[0] + a[1] * a[1]).sum::<f64>();
        1. - x
            .iter()
            .zip(y)
            .map(|(a, b)| (gain * a[0] - b[0]).powi(2) + (gain * a[1] - b[1]).powi(2))
            .sum::<f64>()
            / (den + 1e-6)
    };
    let held = score(&av[split..], &bv[split..]);
    let n = bv.len() - split;
    let nulls: Vec<_> = (1..n)
        .map(|shift| {
            let y: Vec<_> = (0..n).map(|i| bv[split + (i + shift) % n]).collect();
            score(&av[split..], &y)
        })
        .collect();
    let null_best = nulls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let diff: Vec<_> = ar.iter().zip(&br).map(|(&a, &b)| sub(a, b)).collect();
    let mean = median_point(&diff);
    let diff_rms = (diff
        .iter()
        .map(|&p| norm(sub(p, mean)).powi(2))
        .sum::<f64>()
        / diff.len() as f64)
        .sqrt();
    let residual_rms =
        |p: &[Point]| (p.iter().map(|&p| norm(p).powi(2)).sum::<f64>() / p.len() as f64).sqrt();
    let first = median(diff[..diff.len() / 2].iter().map(|&p| norm(p)).collect());
    let last = median(diff[diff.len() / 2..].iter().map(|&p| norm(p)).collect());
    let supported = diff_rms >= 0.75
        && residual_rms(&ar) >= 0.75
        && residual_rms(&br) >= 0.75
        && held >= 0.5
        && held - null_best >= 0.15
        && first >= 0.5
        && last >= 0.5;
    json!({"differential_motion_supported":supported,"differential_centered_rms_native_px":4.*diff_rms,"a_residual_rms_native_px":4.*residual_rms(&ar),"b_residual_rms_native_px":4.*residual_rms(&br),"chronological_train_scalar_gain":gain,"heldout_velocity_skill_vs_zero":held,"cyclic_time_null_best_skill":null_best,"null_count":nulls.len(),"null_margin":held-null_best,"first_half_differential_median_native_px":4.*first,"second_half_differential_median_native_px":4.*last,"velocity_units":"Quad-Bayer cells per source second","null_note":"cyclic heldout endpoint velocity shifts; diagnostic null rank, not a calibrated probability"})
}
pub fn analyze(images: &[Image], times: &[u64]) -> Value {
    if images.is_empty() {
        return json!({"state":"unknown","reason":"no_frames"});
    }
    let points = corners(&images[0]);
    let initial_descriptors: Vec<_> = points
        .iter()
        .map(|&p| patch(&images[0], p, false))
        .collect();
    let mut initial_proposals = Vec::new();
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            let a = points[i];
            let b = points[j];
            let d = norm(sub(a, b));
            if !(9. ..60.).contains(&d) || ((a[0] - b[0]).abs() < 9. && (a[1] - b[1]).abs() < 9.) {
                continue;
            }
            if let (Some(x), Some(y)) = (&initial_descriptors[i], &initial_descriptors[j]) {
                let ncc = correlation(x, y);
                if ncc >= 0.90 {
                    initial_proposals
                        .push(json!({"a_quad_cell":a,"b_quad_cell":b,"initial_ncc":ncc}));
                }
            }
        }
    }
    let traces: Vec<_> = points
        .iter()
        .map(|&p| {
            let (path, reason) = track(images, p);
            (p, path, reason)
        })
        .collect();
    let tracks: Vec<_> = traces
        .iter()
        .filter(|(_, _, reason)| reason.is_none())
        .map(|(p, path, _)| (*p, path.clone()))
        .collect();
    for proposal in &mut initial_proposals {
        for (key, outkey) in [("a_quad_cell", "a_tracking"), ("b_quad_cell", "b_tracking")] {
            let p = [
                proposal[key][0].as_f64().unwrap(),
                proposal[key][1].as_f64().unwrap(),
            ];
            if let Some((_, path, reason)) = traces.iter().find(|(seed, _, _)| *seed == p) {
                proposal[outkey] = json!({"fresh_prefix_quad_cell":path,"observed_frames":path.len(),"first_failed_frame":reason.map(|_|path.len()),"failure":reason});
            }
        }
        proposal["both_endpoints_survive_window"] =
            json!(["a_quad_cell", "b_quad_cell"].iter().all(|&key| {
                let p = [
                    proposal[key][0].as_f64().unwrap(),
                    proposal[key][1].as_f64().unwrap(),
                ];
                tracks.iter().any(|(seed, _)| *seed == p)
            }));
    }
    let common: Vec<_> = (0..images.len())
        .map(|t| {
            median_point(
                &tracks
                    .iter()
                    .map(|(_, p)| sub(p[t], p[0]))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let seeds: Vec<_> = tracks.iter().map(|(p, _)| *p).collect();
    let fits: Vec<_> = (0..images.len())
        .map(|t| {
            common_similarity(
                &seeds,
                &tracks.iter().map(|(_, p)| p[t]).collect::<Vec<_>>(),
            )
        })
        .collect();
    let stabilized: Vec<Vec<_>> = tracks
        .iter()
        .map(|(seed, path)| {
            path.iter()
                .enumerate()
                .map(|(t, &p)| {
                    let residual = sub(p, transform(fits[t].0, *seed));
                    [seed[0] + residual[0], seed[1] + residual[1]]
                })
                .collect()
        })
        .collect();
    let zeros = vec![[0., 0.]; images.len()];
    let mut pairs = Vec::new();
    for i in 0..tracks.len() {
        for j in i + 1..tracks.len() {
            let a = &tracks[i].1;
            let b = &tracks[j].1;
            let distance = norm(sub(a[0], b[0]));
            // Radius-four square patches must never overlap, including diagonally.
            if !(9. ..60.).contains(&distance)
                || ((a[0][0] - b[0][0]).abs() < 9. && (a[0][1] - b[0][1]).abs() < 9.)
            {
                continue;
            }
            let initial = correlation(
                &patch(&images[0], a[0], false).unwrap(),
                &patch(&images[0], b[0], false).unwrap(),
            );
            if initial < 0.90 {
                continue;
            }
            let mut ncc = Vec::new();
            let mut green = Vec::new();
            for t in 0..images.len() {
                ncc.push(correlation(
                    &patch(&images[t], a[t], false).unwrap(),
                    &patch(&images[t], b[t], false).unwrap(),
                ));
                green.push(
                    match (patch(&images[t], a[t], true), patch(&images[t], b[t], true)) {
                        (Some(x), Some(y)) => correlation(&x, &y),
                        _ => -1.,
                    },
                );
            }
            let persistence =
                ncc.iter().filter(|&&v| v >= 0.90).count() as f64 / images.len() as f64;
            let cfa = green.iter().filter(|&&v| v >= 0.85).count() as f64 / images.len() as f64;
            let motion = motion_metrics(&stabilized[i], &stabilized[j], &zeros, times);
            let differential = persistence >= 0.75
                && cfa >= 0.75
                && tracks.len() >= 12
                && fits.iter().all(|(_, n)| *n >= 12)
                && motion["differential_motion_supported"] == true;
            pairs.push(json!({"a_quad_cell":a,"b_quad_cell":b,"initial_patch_ncc":initial,"temporal_copy_support_fraction":persistence,"green_only_copy_support_fraction":cfa,"patch_ncc_by_frame":ncc,"green_only_ncc_by_frame":green,"motion":motion,"anonymous_differential_candidate":differential,"semantic_layer":"anonymous; anatomy/corneal/lens provenance unestablished","glasses_evidence":"unknown","rejection":if differential{"material_provenance_unestablished"}else if persistence<0.75{"copy_not_persistent"}else if cfa<0.75{"cfa_replicate_failed"}else if tracks.len()<12{"insufficient_common_motion_support"}else{"differential_or_temporal_null_gate_failed"}}));
        }
    }
    pairs.sort_by(|a, b| {
        b["temporal_copy_support_fraction"]
            .as_f64()
            .unwrap()
            .total_cmp(&a["temporal_copy_support_fraction"].as_f64().unwrap())
    });
    json!({"state":"unknown","initial_corners":points.len(),"initial_copy_proposals":initial_proposals,"photometry_by_frame":images.iter().map(|im|{let mut p=im.pixels.clone();p.sort_by(f64::total_cmp);json!({"raw_mean":p.iter().sum::<f64>()/p.len()as f64,"raw_p05":p[p.len()/20],"raw_median":p[p.len()/2],"raw_p95":p[p.len()*19/20],"raw_p99":p[p.len()*99/100],"fraction_above_950":p.iter().filter(|&&v|v>950.).count()as f64/p.len()as f64})}).collect::<Vec<_>>(),"complete_tracks":tracks.len(),"track_survival_fraction":tracks.len() as f64/points.len().max(1) as f64,"common_translation_quad_cell":common,"common_similarity_by_frame":fits.iter().map(|(m,n)|json!({"scale_cos_scale_sin_tx_ty":m,"inliers_within_one_cell":n})).collect::<Vec<_>>(),"common_motion_model":"robust image-plane similarity fit; anonymous texture support, not full projective head motion or established eye/material layer","pairs":pairs,"strong_glasses_interval":false,"material_provenance":"not_established","surface_assignment":null})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn times(n: usize) -> Vec<u64> {
        (0..n).map(|i| i as u64 * 33_000_000).collect()
    }
    fn pack(raw: &[u16]) -> Vec<u8> {
        raw.chunks_exact(4)
            .flat_map(|g| {
                let v = g
                    .iter()
                    .enumerate()
                    .fold(0u64, |v, (k, &x)| v | ((x as u64) << (10 * k)));
                (0..5).map(move |k| (v >> (8 * k)) as u8)
            })
            .collect()
    }
    #[test]
    fn quad_bayer_phase_and_color_do_not_create_texture() {
        for sy in 0..4 {
            for sx in 0..4 {
                let raw: Vec<_> = (0..64 * 64)
                    .map(|i| {
                        let x = (i % 64 + sx) % 4;
                        let y = (i / 64 + sy) % 4;
                        if x < 2 && y < 2 {
                            900
                        } else if x >= 2 && y >= 2 {
                            100
                        } else {
                            300
                        }
                    })
                    .collect();
                let image = decode(&pack(&raw), 64, 64, 80, sx, sy).unwrap();
                assert!(image.pixels.iter().all(|v| (*v - 400.).abs() < 1e-9));
                assert!(image.green.iter().all(|v| (*v - 300.).abs() < 1e-9));
                assert!(corners(&image).is_empty());
            }
        }
    }
    #[test]
    fn source_time_gaps_and_reversal_abstain() {
        let (a, b, c) = tracks(3);
        let mut ts = times(a.len());
        ts[5] = ts[4];
        assert_eq!(
            motion_metrics(&a, &b, &c, &ts)["reason"],
            "invalid_or_gapped_source_timestamps"
        );
    }
    fn rendered_layers(mode: usize) -> Vec<Image> {
        let mut images = Vec::new();
        let signal = [0, 1, 0, 1, 2, 1, 0, -1, 0, 1, 0, -1, -2, -1, 0, 1, 0];
        for (t, &s) in signal.iter().enumerate() {
            let global = if mode == 0 { 0 } else { (t % 3) as i32 - 1 };
            let mut cells = vec![100u16; 128 * 96];
            // A distributed static random material texture, translated by the camera.
            for y in 0..96i32 {
                for x in 0..128i32 {
                    let xx = x - global;
                    let yy = y;
                    let mut seed = (xx.wrapping_mul(73856093) ^ yy.wrapping_mul(19349663)) as u32;
                    seed ^= seed >> 13;
                    cells[y as usize * 128 + x as usize] = 100 + (seed % 180) as u16;
                }
            }
            for (cx, gain) in [(32, 1), (88, if mode == 3 { 2 } else { 1 })] {
                let movement = if mode >= 2 { s * gain } else { 0 };
                for dy in -7i32..=7 {
                    for dx in -7i32..=7 {
                        let seed = ((dx + 8) * 43 + (dy + 8) * 59 + (dx + 8) * (dy + 8) * 17) % 101;
                        let value = if dx >= 0 && dy >= 0 {
                            500 + seed * 3
                        } else {
                            80 + seed
                        };
                        let x = cx + dx + movement + global;
                        let y = 50 + dy;
                        cells[y as usize * 128 + x as usize] = value as u16;
                    }
                }
            }
            let raw: Vec<_> = (0..512 * 384)
                .map(|i| cells[(i / 512 / 4) * 128 + (i % 512 / 4)])
                .collect();
            images.push(decode(&pack(&raw), 512, 384, 640, 0, 0).unwrap());
        }
        images
    }
    #[test]
    fn native_raw_image_three_layers_and_common_motion_controls() {
        for mode in 0..4 {
            let images = rendered_layers(mode);
            let report = analyze(&images, &times(images.len()));
            let candidates = report["pairs"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|p| p["anonymous_differential_candidate"] == true)
                .count();
            eprintln!(
                "synthetic mode {mode}: corners {} tracks {} pairs {} differential {candidates}",
                report["initial_corners"],
                report["complete_tracks"],
                report["pairs"].as_array().unwrap().len()
            );
            assert_eq!(report["strong_glasses_interval"], false);
            if mode < 3 {
                assert_eq!(candidates, 0);
            } else {
                assert!(candidates > 0, "{report}");
            }
        }
    }
    #[test]
    fn common_similarity_rotation_scale_is_removed() {
        let base = rendered_layers(0).remove(0);
        let mut frames = Vec::new();
        for t in 0..17 {
            let angle = 0.025 * ((t * 7 % 9) as f64 - 4.);
            let scale = 1. + 0.006 * ((t * 5 % 7) as f64 - 3.);
            let mut pixels = vec![100.; base.pixels.len()];
            for y in 0..base.h {
                for x in 0..base.w {
                    let px = (x as f64 - 64.) / scale;
                    let py = (y as f64 - 48.) / scale;
                    let xx = (angle.cos() * px + angle.sin() * py + 64.).round() as isize;
                    let yy = (-angle.sin() * px + angle.cos() * py + 48.).round() as isize;
                    if xx >= 0 && yy >= 0 && xx < base.w as isize && yy < base.h as isize {
                        pixels[y * base.w + x] = base.pixels[yy as usize * base.w + xx as usize];
                    }
                }
            }
            frames.push(Image {
                w: base.w,
                h: base.h,
                green: pixels.clone(),
                pixels,
            });
        }
        let report = analyze(&frames, &times(frames.len()));
        assert!(
            report["pairs"]
                .as_array()
                .unwrap()
                .iter()
                .all(|p| p["anonymous_differential_candidate"] == false),
            "{report}"
        );
    }
    fn tracks(mode: usize) -> (Vec<Point>, Vec<Point>, Vec<Point>) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        let mut s = 0.;
        for t in 0..25 {
            let v = ((t * 7 % 11) as f64 - 5.) * 0.6;
            s += v;
            let global = [t as f64 * 0.3, 0.];
            c.push(global);
            a.push([global[0] + if mode > 1 { s } else { 0. }, 0.]);
            b.push([
                20. + global[0]
                    + if mode > 2 {
                        2. * s
                    } else if mode == 2 {
                        s
                    } else {
                        0.
                    },
                0.,
            ]);
        }
        (a, b, c)
    }
    #[test]
    fn no_motion_and_common_motion_are_unknown() {
        for mode in 0..3 {
            let (a, b, c) = tracks(mode);
            assert_eq!(
                motion_metrics(&a, &b, &c, &times(a.len()))["differential_motion_supported"],
                false
            );
        }
    }
    #[test]
    fn differential_repeatable_motion_passes_and_shuffle_fails() {
        let (a, b, c) = tracks(3);
        let m = motion_metrics(&a, &b, &c, &times(a.len()));
        assert_eq!(m["differential_motion_supported"], true, "{m}");
        let shuffled: Vec<_> = (0..b.len()).map(|i| b[(i * 7) % b.len()]).collect();
        assert_eq!(
            motion_metrics(&a, &shuffled, &c, &times(a.len()))["differential_motion_supported"],
            false
        );
    }
    #[test]
    fn missing_short_is_explicit() {
        assert_eq!(
            motion_metrics(&[], &[], &[], &[])["reason"],
            "short_or_missing_tracks"
        );
    }
    #[test]
    fn le40_is_not_mipi_and_rejects_bad_shape() {
        let raw: Vec<u8> = (0..32 * 16)
            .flat_map(|_| {
                let v = 100u64 | (200 << 10) | (300 << 20) | (400 << 30);
                (0..5).map(move |k| (v >> (k * 8)) as u8)
            })
            .collect();
        let image = decode(&raw, 64, 32, 80, 0, 0).unwrap();
        assert!(image.pixels.iter().all(|x| (*x - 250.).abs() < 0.01));
        assert!(decode(&raw, 63, 32, 80, 0, 0).is_err());
    }
    #[test]
    fn duplicate_anatomy_remains_unknown() {
        let mut pixels = vec![100.; 96 * 48];
        for x in [20, 60] {
            for dy in 0..10 {
                for dx in 0..10 {
                    pixels[(16 + dy) * 96 + x + dx] = if dx < 5 && dy < 5 { 240. } else { 40. };
                }
            }
        }
        let im = Image {
            w: 96,
            h: 48,
            green: pixels.clone(),
            pixels,
        };
        let out = analyze(&vec![im; 17], &times(17));
        assert_eq!(out["strong_glasses_interval"], false);
        assert!(out["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["anonymous_differential_candidate"] == false));
    }
}
