// Validate an illustrative pink-stroke edit; never treat it as a physical gaze solve.
use serde_json::{json, Value};
#[path = "buttercup_waterfall_validate/composite.rs"]
mod composite;
use sha2::{Digest, Sha256};
use std::{fs, path::Path, process::Command};
#[allow(dead_code)]
mod png {
    include!("buttercup_raw10_preview.rs");
    pub fn save(p: &Path, w: usize, h: usize, rgb: &[u8]) {
        write_png(p, w, h, rgb).unwrap();
    }
}
type Res<T> = Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn read(p: &Path) -> Res<(usize, usize, Vec<u8>)> {
    let bytes = fs::read(p)?;
    if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("PNG input required".into());
    }
    let w = u32::from_be_bytes(bytes[16..20].try_into()?) as usize;
    let h = u32::from_be_bytes(bytes[20..24].try_into()?) as usize;
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|n| n > 20_000_000) {
        return Err("invalid/oversized PNG".into());
    }
    let o = Command::new("ffmpeg")
        .args(["-v", "error", "-threads", "1", "-i"])
        .arg(p)
        .args([
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "pipe:1",
        ])
        .output()?;
    if !o.status.success() || o.stdout.len() != w * h * 3 {
        return Err(format!("PNG decode failed: {}", String::from_utf8_lossy(&o.stderr)).into());
    }
    Ok((w, h, o.stdout))
}
fn pink(p: &[u8]) -> i32 {
    (p[0] as i32 - p[1] as i32).min(p[2] as i32 - p[1] as i32)
}
// Zhang-Suen thinning: recover stroke paths, retaining endpoints and connectivity.
fn skeleton(mut m: Vec<bool>, w: usize, h: usize) -> Vec<bool> {
    for _ in 0..100 {
        let mut changed = false;
        for step in 0..2 {
            let mut remove = Vec::new();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let i = y * w + x;
                    if !m[i] {
                        continue;
                    }
                    let p = [
                        m[i - w],
                        m[i - w + 1],
                        m[i + 1],
                        m[i + w + 1],
                        m[i + w],
                        m[i + w - 1],
                        m[i - 1],
                        m[i - w - 1],
                    ];
                    let n = p.iter().filter(|&&v| v).count();
                    let trans = (0..8).filter(|&k| !p[k] && p[(k + 1) % 8]).count();
                    let keep = if step == 0 {
                        p[0] && p[2] && p[4] || p[2] && p[4] && p[6]
                    } else {
                        p[0] && p[2] && p[6] || p[0] && p[4] && p[6]
                    };
                    if (2..=6).contains(&n) && trans == 1 && !keep {
                        remove.push(i);
                    }
                }
            }
            changed |= !remove.is_empty();
            for i in remove {
                m[i] = false;
            }
        }
        if !changed {
            break;
        }
    }
    m
}
#[derive(Clone, Copy)]
struct Ray {
    p: [f64; 2],
    d: [f64; 2],
}
fn neighbors(i: usize, m: &[bool], w: usize, h: usize) -> Vec<usize> {
    let (x, y) = (i % w, i / w);
    let mut r = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (xx, yy) = (x as isize + dx, y as isize + dy);
            if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                let j = yy as usize * w + xx as usize;
                // A diagonal beside a cardinal connection is a redundant
                // shortcut, not a real skeleton branch.
                let shortcut =
                    dx != 0 && dy != 0 && (m[y * w + xx as usize] || m[yy as usize * w + x]);
                if m[j] && !shortcut {
                    r.push(j);
                }
            }
        }
    }
    r
}
fn fit(r: &[Ray], skip: Option<usize>) -> Option<([f64; 2], f64)> {
    let mut center = [0.; 2];
    let mut condition = 0.;
    for iter in 0..8 {
        let (mut a, mut b, mut c, mut u, mut v) = (0., 0., 0., 0., 0.);
        for (i, r) in r.iter().enumerate() {
            if skip == Some(i) {
                continue;
            }
            let n = [-r.d[1], r.d[0]];
            let residual = (n[0] * (center[0] - r.p[0]) + n[1] * (center[1] - r.p[1])).abs();
            let weight = if iter == 0 {
                1.
            } else {
                (2. / residual.max(2.)).powi(2)
            };
            let q = n[0] * r.p[0] + n[1] * r.p[1];
            a += weight * n[0] * n[0];
            b += weight * n[0] * n[1];
            c += weight * n[1] * n[1];
            u += weight * n[0] * q;
            v += weight * n[1] * q;
        }
        let det = a * c - b * b;
        if det < 1e-8 {
            return None;
        }
        condition = det / (a + c).powi(2);
        center = [(c * u - b * v) / det, (a * v - b * u) / det];
    }
    Some((center, condition))
}
fn estimate(mask: Vec<bool>, w: usize, h: usize) -> Value {
    let points: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect();
    if points.len() < 30 {
        return json!({"status":"insufficient_strokes","pixel_xy":null});
    }
    let centroid = [
        points.iter().map(|i| (i % w) as f64).sum::<f64>() / points.len() as f64,
        points.iter().map(|i| (i / w) as f64).sum::<f64>() / points.len() as f64,
    ];
    let m = skeleton(mask, w, h);
    let mut rays = Vec::new();
    for i in 0..m.len() {
        if !m[i] || neighbors(i, &m, w, h).len() != 1 {
            continue;
        }
        let mut path = vec![i];
        for _ in 0..8 {
            let ns: Vec<_> = neighbors(*path.last().unwrap(), &m, w, h)
                .into_iter()
                .filter(|j| !path.contains(j))
                .collect();
            if ns.len() != 1 {
                break;
            }
            path.push(ns[0]);
        }
        if path.len() < 6 {
            continue;
        }
        let last = *path.last().unwrap();
        let p = [(i % w) as f64, (i / w) as f64];
        let d = [p[0] - (last % w) as f64, p[1] - (last / w) as f64];
        let len = d[0].hypot(d[1]);
        if len < 4. {
            continue;
        }
        let d = [d[0] / len, d[1] / len];
        // Keep ends directed inward, not the outer tips of the strokes.
        if d[0] * (centroid[0] - p[0]) + d[1] * (centroid[1] - p[1]) > 0. {
            rays.push(Ray { p, d });
        }
    }
    let Some((center, condition)) = fit(&rays, None) else {
        return json!({"status":"no_unique_convergence","rays":rays.len(),"pixel_xy":null});
    };
    let mut errors: Vec<f64> = rays
        .iter()
        .map(|r| (-r.d[1] * (center[0] - r.p[0]) + r.d[0] * (center[1] - r.p[1])).abs())
        .collect();
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    let forward = rays
        .iter()
        .filter(|r| r.d[0] * (center[0] - r.p[0]) + r.d[1] * (center[1] - r.p[1]) >= 0.)
        .count();
    let sensitivity = (0..rays.len())
        .filter_map(|i| fit(&rays, Some(i)))
        .map(|(c, _)| (c[0] - center[0]).hypot(c[1] - center[1]))
        .fold(0., f64::max);
    let stable = rays.len() >= 6
        && condition > 0.025
        && median <= 3.
        && sensitivity <= 1.
        && forward * 5 >= rays.len() * 4
        && center[0] >= 0.
        && center[0] < w as f64
        && center[1] >= 0.
        && center[1] < h as f64;
    json!({"status":if stable{"stable_illustration_estimate"}else{"unstable_convergence"},"pixel_xy":if stable{Some([center[0].round() as i64,center[1].round() as i64])}else{None},"diagnostic_float_xy":center,"rays":rays.iter().map(|r|json!({"endpoint":r.p,"direction":r.d})).collect::<Vec<_>>(),"median_line_residual_px":median,"leave_one_ray_out_max_shift_px":sensitivity,"conditioning":condition,"forward_rays":forward,"subpixel_precision_supported":false,"precision_note":"Floating fit is numerical only. No human/source-grounded subpixel accuracy established; sensitivity is not a calibrated confidence interval."})
}
fn run() -> Res<bool> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        return Err("usage: buttercup-waterfall-validate ORIGINAL.png EDITED.png PREPARATION_PROVENANCE.json NEW_OUTPUT_DIR [AUTHORIZED_MASK.png]".into());
    }
    let out = Path::new(&args[4]);
    if out.exists() {
        return Err("output must be new".into());
    }
    if !fs::canonicalize(out.parent().ok_or("missing output parent")?)?
        .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("output must be beneath checked outputs".into());
    }
    let orig_path = Path::new(&args[1]);
    let edit_path = Path::new(&args[2]);
    let receipt: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    if receipt["sheet"]["sha256"].as_str() != Some(&hash(&fs::read(orig_path)?)) {
        return Err("original hash does not match preparation provenance".into());
    }
    let (w, h, a) = read(orig_path)?;
    let (ew, eh, b) = read(edit_path)?;
    fs::create_dir(out)?;
    if (w, h) != (ew, eh) {
        fs::write(
            out.join("report.json"),
            serde_json::to_vec_pretty(
                &json!({"status":"rejected_dimensions","original":[w,h],"edited":[ew,eh],"centers":null,"note":"No resizing or registration is allowed to conceal changed source geometry."}),
            )?,
        )?;
        return Ok(false);
    }
    let explicit = if args.len() == 6 {
        let (mw, mh, m) = read(Path::new(&args[5]))?;
        if (mw, mh) != (w, h) {
            return Err("mask size mismatch".into());
        }
        Some(m)
    } else {
        None
    };
    let mut mask = vec![false; w * h];
    let mut stroke = vec![false; w * h];
    for i in 0..w * h {
        let aa = &a[i * 3..i * 3 + 3];
        let bb = &b[i * 3..i * 3 + 3];
        stroke[i] = pink(bb) > 25 && pink(bb) - pink(aa) > 15 && bb[0] > 100;
        mask[i] = explicit.as_ref().map_or(stroke[i], |m| m[i * 3] > 127);
    }
    // Account for antialiasing only in inferred mode, bounded to a one-pixel halo.
    if explicit.is_none() {
        let base = mask.clone();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                if base[y * w + x] {
                    for yy in y - 1..=y + 1 {
                        for xx in x - 1..=x + 1 {
                            mask[yy * w + xx] = true;
                        }
                    }
                }
            }
        }
    }
    let (mut total_changed, mut outside_changed) = (0, 0);
    let (mut absolute_error, mut outside_pixels, mut above8, mut above32) =
        (0u64, 0usize, 0usize, 0usize);
    let mut diff = vec![0u8; w * h * 3];
    let mut mask_rgb = vec![0u8; w * h * 3];
    for i in 0..w * h {
        if !mask[i] {
            outside_pixels += 1;
            let delta: Vec<u8> = (0..3)
                .map(|c| a[i * 3 + c].abs_diff(b[i * 3 + c]))
                .collect();
            absolute_error += delta.iter().map(|&d| d as u64).sum::<u64>();
            above8 += usize::from(*delta.iter().max().unwrap() > 8);
            above32 += usize::from(*delta.iter().max().unwrap() > 32);
        }
        let changed = a[i * 3..i * 3 + 3] != b[i * 3..i * 3 + 3];
        total_changed += usize::from(changed);
        if mask[i] {
            mask_rgb[i * 3..i * 3 + 3].fill(255);
        } else if changed {
            outside_changed += 1;
            diff[i * 3] = 255;
        }
    }
    png::save(&out.join("excluded-mask.png"), w, h, &mask_rgb);
    png::save(&out.join("outside-mask-changes.png"), w, h, &diff);
    let mut tiles = Vec::new();
    let mut csv = String::from(
        "tile,status,x_roi_px,y_roi_px,x_sheet_px,y_sheet_px,x_sensor_px,y_sensor_px\n",
    );
    for row in receipt["frames"]
        .as_array()
        .ok_or("missing frame records")?
    {
        let r = row["tile_xywh"]
            .as_array()
            .ok_or("missing tile rectangle")?;
        let vals: Vec<_> = r
            .iter()
            .map(|v| v.as_u64().unwrap_or(usize::MAX as u64) as usize)
            .collect();
        if vals.len() != 4 {
            return Err("invalid tile rectangle".into());
        }
        let (x, y, tw, th) = (vals[0], vals[1], vals[2], vals[3]);
        if tw < 3
            || th < 3
            || x.checked_add(tw).is_none_or(|v| v > w)
            || y.checked_add(th).is_none_or(|v| v > h)
        {
            return Err("tile out of bounds".into());
        }
        let (mut changed, mut excluded) = (0, 0);
        let mut tile_strokes = Vec::new();
        for yy in y..y + th {
            for xx in x..x + tw {
                let i = yy * w + xx;
                excluded += usize::from(mask[i]);
                changed += usize::from(!mask[i] && a[i * 3..i * 3 + 3] != b[i * 3..i * 3 + 3]);
                tile_strokes.push(stroke[i] && mask[i]);
            }
        }
        let estimate = estimate(tile_strokes, tw, th);
        let admissible = changed == 0
            && excluded > 0
            && excluded * 4 < tw * th
            && estimate["status"] == "stable_illustration_estimate";
        let center = if admissible {
            estimate["pixel_xy"].clone()
        } else {
            Value::Null
        };
        let status = if changed > 0 {
            "rejected_source_changed"
        } else if excluded * 4 >= tw * th {
            "rejected_excessive_mask"
        } else if admissible {
            "accepted_illustrated_center"
        } else {
            "rejected_unstable_or_missing_strokes"
        };
        let mut sheet = Value::Null;
        let mut sensor = Value::Null;
        if let Some(c) = center.as_array() {
            let (cx, cy) = (c[0].as_i64().unwrap(), c[1].as_i64().unwrap());
            sheet = json!([cx + x as i64, cy + y as i64]);
            let f = &row["source"]["frame"];
            if let (Some(sx), Some(sy)) = (f["sensor_x"].as_i64(), f["sensor_y"].as_i64()) {
                sensor = json!([cx + sx, cy + sy]);
            }
            csv.push_str(&format!(
                "{},{status},{cx},{cy},{},{},{},{}\n",
                row["tile"], sheet[0], sheet[1], sensor[0], sensor[1]
            ));
        } else {
            csv.push_str(&format!("{},{status},,,,,,\n", row["tile"]));
        }
        tiles.push(json!({"tile":row["tile"],"source":row["source"],"status":status,"changed_pixels_outside_mask":changed,"excluded_pixels":excluded,"pixel_xy_roi":center,"pixel_xy_sheet":sheet,"pixel_xy_sensor":sensor,"illustration_diagnostics":estimate}));
    }
    let report = json!({"schema":"buttercup-waterfall-validation-v1","original_sha256":hash(&fs::read(orig_path)?),"edited_sha256":hash(&fs::read(edit_path)?),"dimensions":[w,h],"pixel_coordinate_convention":"zero-based; integer coordinates denote pixel centers; ROI coordinates map by translation only","comparison":"exact decoded RGB bytes outside excluded mask; no alignment, resizing or color normalization","mask_source":if explicit.is_some(){"user_supplied_authorized_mask"}else{"inferred_new_pink_plus_one_pixel_halo"},"mask_caveat":"An inferred mask is not independent authorization; natural pink changes can be excluded. Use a predeclared mask for independent preservation verification. Changes inside any mask remain unchecked.","total_changed_pixels":total_changed,"outside_mask_pixels":outside_pixels,"outside_mask_mean_absolute_channel_error":absolute_error as f64 / (outside_pixels.max(1)*3) as f64,"outside_mask_pixels_max_channel_error_above_8":above8,"outside_mask_pixels_max_channel_error_above_32":above32,"changed_pixels_outside_mask":outside_changed,"outside_mask_exact_match":outside_changed==0,"physical_gaze_origin":null,"tiles":tiles});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    fs::write(out.join("centers.csv"), csv)?;
    println!(
        "{} pixels changed outside mask; {}/{} centers admitted",
        outside_changed,
        tiles
            .iter()
            .filter(|t| t["status"] == "accepted_illustrated_center")
            .count(),
        tiles.len()
    );
    Ok(outside_changed == 0
        && tiles
            .iter()
            .all(|t| t["status"] == "accepted_illustrated_center"))
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = if args.get(1).is_some_and(|v| v == "--composite") {
        composite::run(&args)
    } else if args.get(1).is_some_and(|v| v == "--landmarks") {
        composite::landmarks::run(&args)
    } else {
        run()
    };
    match result {
        Ok(true) => (),
        Ok(false) => std::process::exit(2),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
