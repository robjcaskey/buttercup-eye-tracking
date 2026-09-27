//! Presentation-only global alignment from outer material tracks.
use opencv::{
    calib3d,
    core::{self, Mat, Point2d, Vector},
    prelude::*,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fmt::Write as _,
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
type E = Box<dyn std::error::Error>;
type Affine = [[f64; 3]; 2];
fn number(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn apply(a: Affine, p: [f64; 2]) -> [f64; 2] {
    a.map(|r| r[0] * p[0] + r[1] * p[1] + r[2])
}
fn inverse(a: Affine) -> Affine {
    let d = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    assert!(d.abs() > 1e-8);
    let b = [[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]];
    std::array::from_fn(|i| [b[i][0], b[i][1], -b[i][0] * a[0][2] - b[i][1] * a[1][2]])
}
struct Image {
    w: usize,
    h: usize,
    origin: [f64; 2],
    rgb: Vec<u32>,
}
fn image(f: &Value) -> Result<Image, E> {
    let input = &f["input"];
    let meta = &input["frame"];
    let w = number(&meta["width"]) as usize;
    let h = number(&meta["height"]) as usize;
    let origin = [number(&meta["sensor_x"]), number(&meta["sensor_y"])];
    let mut file = fs::File::open(input["raw_file"].as_str().unwrap())?;
    file.seek(SeekFrom::Start(input["raw_offset"].as_u64().unwrap()))?;
    let mut bytes = vec![0; input["raw_length"].as_u64().unwrap() as usize];
    file.read_exact(&mut bytes)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        f["raw_sha256"].as_str().unwrap()
    );
    let raw = crate::raw10::try_unpack_raw10(&bytes, w, h, number(&meta["stride"]) as usize)?;
    let rgb = crate::raw_preview::color_preview(
        &raw,
        w,
        h,
        origin[0] as u32,
        origin[1] as u32,
        100,
        None,
    );
    Ok(Image { w, h, origin, rgb })
}
fn sample(im: &Image, p: [f64; 2]) -> u32 {
    let x = p[0] - im.origin[0];
    let y = p[1] - im.origin[1];
    let ix = x.floor() as usize;
    let iy = y.floor() as usize;
    assert!(ix + 1 < im.w && iy + 1 < im.h);
    let ax = x - ix as f64;
    let ay = y - iy as f64;
    let mut color = 0;
    for shift in [0, 8, 16] {
        let channel = |dx, dy| ((im.rgb[(iy + dy) * im.w + ix + dx] >> shift) & 255u32) as f64;
        let v = (1. - ay) * ((1. - ax) * channel(0, 0) + ax * channel(1, 0))
            + ay * ((1. - ax) * channel(0, 1) + ax * channel(1, 1));
        color |= (v.round() as u32) << shift;
    }
    color
}
fn rect(valid: &[bool], w: usize, h: usize) -> [usize; 4] {
    let mut heights = vec![0; w];
    let mut best = [0; 4];
    let mut area = 0;
    for y in 0..h {
        for x in 0..w {
            heights[x] = if valid[y * w + x] { heights[x] + 1 } else { 0 };
        }
        let mut stack: Vec<(usize, usize)> = vec![];
        for x in 0..=w {
            let height = if x < w { heights[x] } else { 0 };
            let mut left = x;
            while stack.last().is_some_and(|&(_, v)| v > height) {
                let (start, v) = stack.pop().unwrap();
                if (x - start) * v > area {
                    area = (x - start) * v;
                    best = [start, y + 1 - v, x, y + 1];
                }
                left = start;
            }
            if height > 0 && stack.last().is_none_or(|&(_, v)| v < height) {
                stack.push((left, height));
            }
        }
    }
    best
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn project(v: [f64; 3]) -> [f64; 2] {
    [4000. + 4000. * v[0] / v[2], 3000. + 4000. * v[1] / v[2]]
}
fn pstr(points: &[[f64; 2]]) -> String {
    points
        .iter()
        .map(|p| format!("{:.3},{:.3}", p[0], p[1]))
        .collect::<Vec<_>>()
        .join(" ")
}
pub fn render(previous: &Value, current: &Value, tracks: &Value, out: &Path) -> Result<(), E> {
    let tracks = tracks.as_array().ok_or("missing outer tracks")?;
    if tracks.len() < 8 {
        return Err("insufficient outer points for visualization warp".into());
    }
    let a: Vector<Point2d> = tracks
        .iter()
        .map(|p| Point2d::new(number(&p["current"][0]), number(&p["current"][1])))
        .collect();
    let b: Vector<Point2d> = tracks
        .iter()
        .map(|p| Point2d::new(number(&p["previous"][0]), number(&p["previous"][1])))
        .collect();
    core::set_rng_seed(19)?;
    let mut inliers = Mat::default();
    let matrix =
        calib3d::estimate_affine_2d(&a, &b, &mut inliers, calib3d::RANSAC, 1.5, 4000, 0.999, 20)?;
    if matrix.rows() != 2 || matrix.cols() != 3 {
        return Err("affine fit failed".into());
    }
    let mut transform = [[0.; 3]; 2];
    for r in 0..2 {
        for c in 0..3 {
            transform[r][c] = *matrix.at_2d::<f64>(r as i32, c as i32)?;
        }
    }
    let inv = inverse(transform);
    let first = image(previous)?;
    let second = image(current)?;
    let valid: Vec<_> = (0..first.w * first.h)
        .map(|i| {
            let x = i % first.w;
            let y = i / first.w;
            let target = apply(
                inv,
                [first.origin[0] + x as f64, first.origin[1] + y as f64],
            );
            let sx = target[0] - second.origin[0];
            let sy = target[1] - second.origin[1];
            x > 0
                && y > 0
                && x + 1 < first.w
                && y + 1 < first.h
                && sx >= 1.
                && sy >= 1.
                && sx < (second.w - 2) as f64
                && sy < (second.h - 2) as f64
        })
        .collect();
    let crop = rect(&valid, first.w, first.h);
    let w = crop[2] - crop[0];
    let h = crop[3] - crop[1];
    if w < 32 || h < 32 {
        return Err("shared visible crop too small".into());
    }
    let origin = [
        first.origin[0] + crop[0] as f64,
        first.origin[1] + crop[1] as f64,
    ];
    let mut rgbs = [Vec::with_capacity(w * h), Vec::with_capacity(w * h)];
    for y in 0..h {
        for x in 0..w {
            let pos = [origin[0] + x as f64, origin[1] + y as f64];
            rgbs[0].push(sample(&first, pos));
            rgbs[1].push(sample(&second, apply(inv, pos)));
        }
    }
    let mut before = vec![];
    let mut after = vec![];
    let mut accepted = vec![];
    for (i, p) in tracks.iter().enumerate() {
        let a = [number(&p["previous"][0]), number(&p["previous"][1])];
        let b = [number(&p["current"][0]), number(&p["current"][1])];
        let c = apply(transform, b);
        before.push((b[0] - a[0]).hypot(b[1] - a[1]));
        after.push((c[0] - a[0]).hypot(c[1] - a[1]));
        if *inliers.at_2d::<u8>(i as i32, 0)? != 0 {
            accepted.push(p["id"].clone());
        }
    }
    let mut svg=String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1500' height='1220'><rect width='100%' height='100%' fill='#161922'/><g font-family='sans-serif' fill='white'><text x='25' y='36' font-size='25'>Globally stabilized comparison — outer-region affine registration</text><text x='25' y='65' font-size='17'>First frame is the reference; second frame and its overlays share the same inverse-motion warp and common crop.</text>");
    let scale = (670. / w as f64).min(420. / h as f64);
    for row in 0..2 {
        let f = if row == 0 { previous } else { current };
        for col in 0..2 {
            let x0 = 25 + col * 745;
            let y0 = 125 + row * 495;
            write!(svg,"<text x='{x0}' y='{}' font-size='20'>Source {} · candidate {} · {}</text><g transform='translate({x0},{y0}) scale({scale})'>",y0-15,f["sequence"],['A','B'][col],if row==0{"reference"}else{"stabilized"})?;
            svg.push_str("<g shape-rendering='crispEdges'>");
            for y in 0..h {
                for x in 0..w {
                    write!(
                        svg,
                        "<rect x='{x}' y='{y}' width='1' height='1' fill='#{:06x}'/>",
                        rgbs[row][y * w + x]
                    )?;
                }
            }
            svg.push_str("</g>");
            let warp = |p: [f64; 2]| {
                let p = if row == 0 { p } else { apply(transform, p) };
                [p[0] - origin[0], p[1] - origin[1]]
            };
            let e = &f["ellipse"];
            let (ca, sa) = (number(&e["angle"]).cos(), number(&e["angle"]).sin());
            let curve: Vec<_> = (0..=180)
                .map(|i| {
                    let t = i as f64 * std::f64::consts::TAU / 180.;
                    let x = number(&e["a"]) * t.cos();
                    let y = number(&e["b"]) * t.sin();
                    warp([
                        number(&e["center_sensor_px"][0]) + ca * x - sa * y,
                        number(&e["center_sensor_px"][1]) + sa * x + ca * y,
                    ])
                })
                .collect();
            write!(
                svg,
                "<polyline points='{}' fill='none' stroke='white' stroke-width='1'/>",
                pstr(&curve)
            )?;
            let p = &f["poses"][col];
            let c = std::array::from_fn(|i| number(&p[i]));
            let tip = std::array::from_fn(|i| c[i] + 7. * number(&p[i + 3]));
            let line = [warp(project(c)), warp(project(tip))];
            write!(svg,"<polyline points='{}' fill='none' stroke='#ffdc55' stroke-width='1.5'/><circle cx='{}' cy='{}' r='2' fill='#ffdc55'/>",pstr(&line),line[1][0],line[1][1])?;
            for p in tracks {
                let coords = if row == 0 {
                    &p["previous"]
                } else {
                    &p["current"]
                };
                let pos = warp([number(&coords[0]), number(&coords[1])]);
                if pos[0] >= 0. && pos[0] < w as f64 && pos[1] >= 0. && pos[1] < h as f64 {
                    let color = if accepted.contains(&p["id"]) {
                        "#52e8dd"
                    } else {
                        "#ef8894"
                    };
                    write!(svg,"<circle cx='{}' cy='{}' r='1.6' stroke='{color}' stroke-width='.7' fill='none'/>",pos[0],pos[1])?;
                }
            }
            svg.push_str("</g>");
        }
    }
    write!(svg,"<text x='25' y='1090' font-size='18'>Shared crop: {w} × {h} pixels. Outer tracks: {}, RANSAC inliers: {}. Median displacement {:.2} → {:.2} px.</text>",tracks.len(),accepted.len(),median(before.clone()),median(after.clone()))?;
    svg.push_str("<text x='25' y='1120' font-size='17'>Cyan rings: outer fit inliers; pink rings: rejected tracks. White: fitted iris; yellow: hypothetical normal.</text><text x='25' y='1150' font-size='17'>2D affine stabilization approximates global motion; it is NOT a recovered 3D affine/depth warp.</text><text x='25' y='1180' font-size='17'>Existing color preview; bilinear resampling for display only. No per-iris alignment, refit, or sign selection.</text></g></svg>");
    fs::write(out.join("stabilized-contact.svg"), svg)?;
    for row in 0..2 {
        let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
        for c in &rgbs[row] {
            ppm.extend_from_slice(&[(c >> 16) as u8, (c >> 8) as u8, *c as u8]);
        }
        fs::write(out.join(format!("stabilized-{row}.ppm")), ppm)?;
    }
    let info = json!({"method":"robust image-plane affine from second absolute sensor coordinates to first; not known 3D motion","affine_second_to_first":transform,"shared_crop_first_roi_xyxy":crop,"shared_crop_sensor_origin":origin,"size":[w,h],"tracks":tracks.len(),"inlier_ids":accepted,"threshold_px":1.5,"median_before_px":median(before),"median_after_px":median(after),"raw_sha256":[previous["raw_sha256"],current["raw_sha256"]],"sampling":"existing color preview; bilinear presentation sampling only; no geometry refit"});
    fs::write(
        out.join("stabilization.json"),
        serde_json::to_vec_pretty(&info)?,
    )?;
    println!("stabilization: {info}");
    Ok(())
}
