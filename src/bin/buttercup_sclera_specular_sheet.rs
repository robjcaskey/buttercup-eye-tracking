//! Evaluation-only projected globe masks over the existing live specular estimate.
//! Neither the masks nor brightness summaries constitute anatomical/sign truth.
#![allow(dead_code)]
#[path = "../raw10.rs"]
mod raw10;
#[path = "../raw_preview.rs"]
mod raw_preview;
#[path = "../specular_map.rs"]
mod specular_map;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fmt::Write as _,
    fs,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
type E = Box<dyn std::error::Error>;
type V = [f64; 3];
fn dot(a: V, b: V) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn ui(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}
fn project(c: V) -> [f64; 2] {
    [4000. + 4000. * c[0] / c[2], 3000. + 4000. * c[1] / c[2]]
}
struct Globe {
    center: V,
    normal: V,
    a: f64,
    b: f64,
}
impl Globe {
    fn new(p: &Value, a: f64, b: f64) -> Self {
        let c: V = std::array::from_fn(|i| num(&p[i]));
        let normal = std::array::from_fn(|i| num(&p[i + 3]));
        let d = b * (1. - 36. / (a * a)).sqrt();
        Self {
            center: std::array::from_fn(|i| c[i] - d * normal[i]),
            normal,
            a,
            b,
        }
    }
    fn q(&self, x: V, y: V) -> f64 {
        dot(x, y) / (self.a * self.a)
            + (1. / (self.b * self.b) - 1. / (self.a * self.a))
                * dot(x, self.normal)
                * dot(y, self.normal)
    }
    fn hits(&self, x: f64, y: f64) -> bool {
        let ray = [(x - 4000.) / 4000., (y - 3000.) / 4000., 1.];
        let aa = self.q(ray, ray);
        let bb = -2. * self.q(ray, self.center);
        let cc = self.q(self.center, self.center) - 1.;
        let disc = bb * bb - 4. * aa * cc;
        disc >= 0. && (-bb - disc.sqrt()) / (2. * aa) > 0.
    }
}
fn iris(f: &Value, x: f64, y: f64) -> bool {
    let e = &f["ellipse"];
    let dx = x - num(&e["center_sensor_px"][0]);
    let dy = y - num(&e["center_sensor_px"][1]);
    let t = num(&e["angle"]);
    let u = dx * t.cos() + dy * t.sin();
    let v = -dx * t.sin() + dy * t.cos();
    (u / num(&e["a"])).powi(2) + (v / num(&e["b"])).powi(2) <= 1.
}
struct Frame {
    f: Value,
    w: usize,
    h: usize,
    sx: usize,
    sy: usize,
    color: Vec<u32>,
    map: Vec<u32>,
    masks: [Vec<bool>; 2],
    globes: [Globe; 2],
}
fn text(s: &mut String, x: usize, y: usize, t: &str) {
    write!(s, "<text x='{x}' y='{y}'>{t}</text>").unwrap();
}
/// Illustrative wide-open lid edges on the camera-facing hemisphere.
/// These assume camera-horizontal corners; they are not fitted lid anatomy
/// and do not inherit the iris normal as a supposed head orientation.
fn illustrative_lid(globe: &Globe, upper: bool, sx: usize, sy: usize) -> String {
    let length = dot(globe.center, globe.center).sqrt();
    let view = globe.center.map(|v| v / length);
    let right_length = view[2].hypot(view[0]);
    let right = [view[2] / right_length, 0., -view[0] / right_length];
    let down = [
        view[1] * right[2],
        view[2] * right[0] - view[0] * right[2],
        -view[1] * right[0],
    ];
    let mut path = String::new();
    for i in 0..=160 {
        let t = -1. + 2. * i as f64 / 160.;
        let x = 0.92 * t;
        let y = if upper { -0.68 } else { 0.50 } * (1. - t * t).sqrt();
        let z = -(1. - x * x - y * y).max(0.).sqrt();
        let point = std::array::from_fn(|k| {
            globe.center[k] + globe.a * (x * right[k] + y * down[k] + z * view[k])
        });
        let p = project(point);
        write!(
            path,
            "{} {:.3},{:.3} ",
            if i == 0 { "M" } else { "L" },
            p[0] - sx as f64,
            p[1] - sy as f64
        )
        .unwrap();
    }
    path
}
fn sheet(frames: &[Frame], out: &Path, lids: bool) -> Result<(), E> {
    let pw = 470;
    let ph = 365;
    let mut s=String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1900' height='950'><rect width='100%' height='100%' fill='#141820'/><g font-family='sans-serif' font-size='18' fill='white'>");
    text(
        &mut s,
        20,
        30,
        &format!("Two candidate free-globe silhouettes on consecutive RAW exposures (sphere radius {:.1} mm)", frames[0].globes[0].a),
    );
    text(
        &mut s,
        20,
        57,
        if lids {
            "Illustrative wide-open lids: pink upper / lime lower. Cyan/orange: hypothetical globe; white: iris; +: globe center."
        } else {
            "Color and estimated specular map per branch. Cyan/orange: hypothetical globe; white: shared iris fit; +: projected globe center."
        },
    );
    for (row, f) in frames.iter().enumerate() {
        for col in 0..4 {
            let branch = col / 2;
            let map = col % 2 == 1;
            let ox = 20 + col * pw;
            let oy = 105 + row * ph;
            let title = format!(
                "{} eye {} seq {} / {} {}",
                f.f["selection"]["capture"].as_str().unwrap(),
                ui(&f.f["eye"]),
                ui(&f.f["sequence"]),
                if branch == 0 { "A" } else { "B" },
                if map { "specular" } else { "color" }
            );
            text(&mut s, ox, oy - 12, &title);
            write!(s, "<g transform='translate({ox},{oy})'>")?;
            let pixels = if map { &f.map } else { &f.color };
            let mask = &f.masks[branch];
            for y in (0..f.h).step_by(2) {
                for x in (0..f.w).step_by(2) {
                    let i = y * f.w + x;
                    let c = pixels[i];
                    let scale = if mask[i] { 1. } else { 0.22 };
                    let r = (((c >> 16) & 255) as f64 * scale) as u32;
                    let g = (((c >> 8) & 255) as f64 * scale) as u32;
                    let b = ((c & 255) as f64 * scale) as u32;
                    write!(s,"<rect x='{x}' y='{y}' width='2' height='2' fill='#{r:02x}{g:02x}{b:02x}'/>")?;
                }
            }
            let color = if branch == 0 { "#33e6f2" } else { "#ffac48" };
            for y in 1..f.h - 1 {
                for x in 1..f.w - 1 {
                    let i = y * f.w + x;
                    if mask[i] && (!mask[i - 1] || !mask[i + 1] || !mask[i - f.w] || !mask[i + f.w])
                    {
                        write!(
                            s,
                            "<rect x='{x}' y='{y}' width='1' height='1' fill='{color}'/>"
                        )?;
                    }
                }
            }
            let e = &f.f["ellipse"];
            let cx = num(&e["center_sensor_px"][0]) - f.sx as f64;
            let cy = num(&e["center_sensor_px"][1]) - f.sy as f64;
            write!(s,"<ellipse cx='{cx}' cy='{cy}' rx='{}' ry='{}' transform='rotate({} {cx} {cy})' fill='none' stroke='white' stroke-width='1'/>",num(&e["a"]),num(&e["b"]),num(&e["angle"]).to_degrees())?;
            let p = project(f.globes[branch].center);
            let x = p[0] - f.sx as f64;
            let y = p[1] - f.sy as f64;
            if lids {
                write!(s, "<defs><clipPath id='lids-{row}-{col}'><rect width='{}' height='{}'/></clipPath></defs><g clip-path='url(#lids-{row}-{col})'>", f.w, f.h)?;
                for upper in [true, false] {
                    let path = illustrative_lid(&f.globes[branch], upper, f.sx, f.sy);
                    let stroke = if upper { "#ff65d5" } else { "#baff65" };
                    write!(s, "<path d='{path}' fill='none' stroke='#10131a' stroke-width='4' stroke-opacity='.75'/><path d='{path}' fill='none' stroke='{stroke}' stroke-width='2' stroke-dasharray='7 3'/>")?;
                }
                s.push_str("</g>");
            }
            write!(
                s,
                "<path d='M {},{y} h 12 M {x},{} v 12' stroke='{color}' stroke-width='2'/></g>",
                x - 6.,
                y - 6.
            )?;
            text(
                &mut s,
                ox,
                oy + f.h + 23,
                &format!(
                    "Mask within ROI: {} pixels",
                    mask.iter().filter(|m| **m).count()
                ),
            );
        }
    }
    text(
        &mut s,
        20,
        870,
        if lids {
            "Lids are invented camera-facing guides, NOT anatomical fits. They do not affect scores; changing globe size DOES change masks."
        } else {
            "Mask = projected free globe minus iris disk. Eyelids, skin, glasses and cornea are NOT segmented; missing ROI is unknown."
        },
    );
    text(&mut s,20,897,"Map reuses live neutral-bright estimator and live demosaiced color preview; it is NOT measured polarization or a reflection model.");
    text(&mut s,20,924,"No light positions, independent sign labels or anatomical globe measurements: more highlights inside a mask is not a sign solution.");
    s.push_str("</g></svg>");
    fs::write(out, s)?;
    Ok(())
}
/// Unmasked context is essential: a free-globe silhouette is not an eyelid edge.
fn projection_audit(frames: &[Frame], out: &Path) -> Result<Value, E> {
    let mut s = String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1900' height='950'><rect width='100%' height='100%' fill='#141820'/><g font-family='sans-serif' font-size='18' fill='white'>");
    text(&mut s,20,30,"Projection audit: full RAW context; 12 mm vs 13.2 mm globe radius, same 6 mm iris and cached poses");
    text(&mut s,20,57,"White: input ellipse. Pink: reprojected 3D iris circle. Cyan/orange: free-globe silhouette; +: globe center.");
    text(&mut s,20,82,"No invented eyelids and no darkening of skin: the outer sphere is mostly hidden by real eyelids, not fitted to their edges.");
    let mut reports = vec![];
    for (row, f) in frames.iter().enumerate() {
        for col in 0..4 {
            let branch = col % 2;
            let radius = if col < 2 { 12. } else { 13.2 };
            let globe = Globe::new(&f.f["poses"][branch], radius, radius);
            let ox = 20 + 470 * col;
            let oy = 125 + 350 * row;
            text(
                &mut s,
                ox,
                oy - 12,
                &format!(
                    "seq {} / {} / radius {radius} mm",
                    ui(&f.f["sequence"]),
                    if branch == 0 { "A" } else { "B" }
                ),
            );
            write!(s,"<defs><clipPath id='audit-{row}-{col}'><rect width='{}' height='{}'/></clipPath></defs><g transform='translate({ox},{oy})' clip-path='url(#audit-{row}-{col})'>",f.w,f.h)?;
            for y in 0..f.h {
                for x in 0..f.w {
                    let c = f.color[y * f.w + x] & 0xffffff;
                    write!(
                        s,
                        "<rect x='{x}' y='{y}' width='1' height='1' fill='#{c:06x}'/>"
                    )?;
                }
            }
            let color = if branch == 0 { "#33e6f2" } else { "#ffac48" };
            for y in 1..f.h - 1 {
                for x in 1..f.w - 1 {
                    let xx = (f.sx + x) as f64 + 0.5;
                    let yy = (f.sy + y) as f64 + 0.5;
                    if globe.hits(xx, yy)
                        && (!globe.hits(xx - 1., yy)
                            || !globe.hits(xx + 1., yy)
                            || !globe.hits(xx, yy - 1.)
                            || !globe.hits(xx, yy + 1.))
                    {
                        write!(
                            s,
                            "<rect x='{x}' y='{y}' width='1' height='1' fill='{color}'/>"
                        )?;
                    }
                }
            }
            let e = &f.f["ellipse"];
            let a = num(&e["a"]);
            let b = num(&e["b"]);
            let angle = num(&e["angle"]);
            let cx = num(&e["center_sensor_px"][0]);
            let cy = num(&e["center_sensor_px"][1]);
            write!(s,"<ellipse cx='{}' cy='{}' rx='{a}' ry='{b}' transform='rotate({} {} {})' fill='none' stroke='white' stroke-width='2'/>",cx-f.sx as f64,cy-f.sy as f64,angle.to_degrees(),cx-f.sx as f64,cy-f.sy as f64)?;
            let p = &f.f["poses"][branch];
            let center: V = std::array::from_fn(|i| num(&p[i]));
            let n = globe.normal;
            let length = n[0].hypot(n[2]);
            let u = [n[2] / length, 0., -n[0] / length];
            let v = [n[1] * u[2], n[2] * u[0] - n[0] * u[2], -n[1] * u[0]];
            let mut path = String::new();
            let mut max_error: f64 = 0.;
            let mut sum = 0.;
            for i in 0..=360 {
                let t = (i as f64).to_radians();
                let q = project(std::array::from_fn(|j| {
                    center[j] + 6. * (u[j] * t.cos() + v[j] * t.sin())
                }));
                write!(
                    path,
                    "{} {},{} ",
                    if i == 0 { "M" } else { "L" },
                    q[0] - f.sx as f64,
                    q[1] - f.sy as f64
                )?;
                let dx = q[0] - cx;
                let dy = q[1] - cy;
                let ex = dx * angle.cos() + dy * angle.sin();
                let ey = -dx * angle.sin() + dy * angle.cos();
                let implicit = ex * ex / (a * a) + ey * ey / (b * b) - 1.;
                let grad = 2. * (ex / (a * a)).hypot(ey / (b * b));
                let error = implicit.abs() / grad.max(1e-12);
                max_error = max_error.max(error);
                sum += error;
            }
            write!(s,"<path d='{path}' fill='none' stroke='#ff65d5' stroke-width='1' stroke-dasharray='4 4'/>")?;
            let c = project(globe.center);
            let x = c[0] - f.sx as f64;
            let y = c[1] - f.sy as f64;
            write!(
                s,
                "<path d='M {},{y} h 12 M {x},{} v 12' stroke='{color}' stroke-width='2'/></g>",
                x - 6.,
                y - 6.
            )?;
            text(
                &mut s,
                ox,
                oy + f.h + 22,
                &format!("Circle reprojection max: {max_error:.4} px"),
            );
            reports.push(json!({"sequence":f.f["sequence"],"branch":branch,"radius_mm":radius,"normal_norm":dot(n,n).sqrt(),"camera_facing_dot":dot(n,center),"globe_center_mm":globe.center,"iris_circle_reprojection_first_order_px_mean":sum/361.,"iris_circle_reprojection_first_order_px_max":max_error}));
        }
    }
    text(&mut s,20,865,"Geometric consistency is not anatomical accuracy: radius, intrinsics and the input ellipse remain assumptions.");
    text(&mut s,20,893,"Invented lids previously moved with the branch and falsely suggested anatomical evidence; actual lids stay fixed in these panels.");
    text(&mut s,20,921,"Neither branch is selected. No measured lid/canthus annotations, physical scale, corneal model or independent sign truth.");
    s.push_str("</g></svg>");
    fs::write(out, s)?;
    Ok(json!(reports))
}
fn main() -> Result<(), E> {
    let args = std::env::args().collect::<Vec<_>>();
    let lids = args.iter().any(|arg| arg == "--illustrative-lids");
    let globe_scale: f64 = match args.iter().position(|arg| arg == "--globe-scale") {
        Some(i) => args.get(i + 1).ok_or("missing globe scale")?.parse()?,
        None => 1.,
    };
    if !globe_scale.is_finite() || !(0.6..=2.).contains(&globe_scale) {
        return Err("globe scale must be finite and within 0.6..2".into());
    }
    let equatorial_radius = 12. * globe_scale;
    let input = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("outputs/limbus-sign-probe-20260916-neighbors/results.json");
    let out = Path::new(
        args.get(2)
            .map(String::as_str)
            .unwrap_or("outputs/sclera-specular-sign-20260916"),
    );
    fs::create_dir_all(out)?;
    let root: Value = serde_json::from_slice(&fs::read(input)?)?;
    let mut rows = vec![];
    let mut selected = [vec![], vec![]];
    for f in root["frames"].as_array().unwrap() {
        let input = &f["input"];
        let meta = &input["frame"];
        let w = ui(&meta["width"]);
        let h = ui(&meta["height"]);
        let sx = ui(&meta["sensor_x"]);
        let sy = ui(&meta["sensor_y"]);
        let mut file = fs::File::open(input["raw_file"].as_str().unwrap())?;
        file.seek(SeekFrom::Start(input["raw_offset"].as_u64().unwrap()))?;
        let mut bytes = vec![0; ui(&input["raw_length"])];
        file.read_exact(&mut bytes)?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(hash, f["raw_sha256"].as_str().unwrap());
        let raw = raw10::try_unpack_raw10(&bytes, w, h, ui(&meta["stride"]))?;
        let color = raw_preview::color_preview(&raw, w, h, sx as u32, sy as u32, 100, None);
        let map = specular_map::SpecularMapTracker::default()
            .observe(&color, w, h, sx as u32, sy as u32, None)
            .specular_map;
        let total = map
            .iter()
            .map(|c| ((*c & 255) as f64 / 255.).powi(2))
            .sum::<f64>();
        let mut shapes = vec![];
        for b in [10.5, 12., 13.5].map(|b| b * globe_scale) {
            let globes: [Globe; 2] =
                std::array::from_fn(|k| Globe::new(&f["poses"][k], equatorial_radius, b));
            let mut count = [0usize; 2];
            let mut mass = [0.; 2];
            let mut only = [0usize; 2];
            let mut only_mass = [0.; 2];
            let mut both = 0usize;
            let mut roi_globe = [0usize; 2];
            let mut full_globe = [0usize; 2];
            for k in 0..2 {
                let p = project(globes[k].center);
                let bound = equatorial_radius.max(b);
                let radius = 4000. * bound / (globes[k].center[2] - bound)
                    * (1.
                        + globes[k].center[0].abs().max(globes[k].center[1].abs())
                            / globes[k].center[2]);
                for y in (p[1] - radius).floor() as i32..=(p[1] + radius).ceil() as i32 {
                    for x in (p[0] - radius).floor() as i32..=(p[0] + radius).ceil() as i32 {
                        if globes[k].hits(x as f64 + 0.5, y as f64 + 0.5) {
                            full_globe[k] += 1;
                            if x >= sx as i32
                                && x < (sx + w) as i32
                                && y >= sy as i32
                                && y < (sy + h) as i32
                            {
                                roi_globe[k] += 1;
                            }
                        }
                    }
                }
            }
            for y in 0..h {
                for x in 0..w {
                    let xx = (sx + x) as f64 + 0.5;
                    let yy = (sy + y) as f64 + 0.5;
                    let mask = globes
                        .each_ref()
                        .map(|g| g.hits(xx, yy) && !iris(f, xx, yy));
                    let score = ((map[y * w + x] & 255) as f64 / 255.).powi(2);
                    for k in 0..2 {
                        if mask[k] {
                            count[k] += 1;
                            mass[k] += score;
                            if !mask[1 - k] {
                                only[k] += 1;
                                only_mass[k] += score;
                            }
                        }
                    }
                    if mask[0] && mask[1] {
                        both += 1;
                    }
                }
            }
            shapes.push(json!({"equatorial_radius_mm":equatorial_radius,"axial_radius_mm":b,"pivot_depth_mm":b*(1.-36./(equatorial_radius*equatorial_radius)).sqrt(),"centers_mm":globes.each_ref().map(|g|g.center),"observed_sclera_pixels":count,"mean_specular_heuristic":std::array::from_fn::<_,2,_>(|k|mass[k]/count[k].max(1) as f64),"fraction_roi_specular_mass":mass.map(|v|v/total.max(1e-12)),"unique_pixels":only,"unique_mean_score":std::array::from_fn::<_,2,_>(|k|only_mass[k]/only[k].max(1) as f64),"mask_iou_observed":both as f64/(count[0]+count[1]-both).max(1) as f64,"globe_coverage_roi":std::array::from_fn::<_,2,_>(|k|roi_globe[k] as f64/full_globe[k].max(1) as f64)}));
        }
        rows.push(json!({"capture":f["selection"]["capture"],"sequence":f["sequence"],"eye":f["eye"],"raw_sha256":hash,"shapes":shapes}));
        let seq = ui(&f["sequence"]);
        let eye = ui(&f["eye"]);
        let group = if eye == 1 && (seq == 60 || seq == 61) {
            Some(0)
        } else if eye == 2 && (seq == 1034 || seq == 1035) {
            Some(1)
        } else {
            None
        };
        if let Some(group) = group {
            let globes = std::array::from_fn(|k| {
                Globe::new(&f["poses"][k], equatorial_radius, equatorial_radius)
            });
            let masks = std::array::from_fn(|k| {
                (0..w * h)
                    .map(|i| {
                        globes[k].hits((sx + i % w) as f64 + 0.5, (sy + i / w) as f64 + 0.5)
                            && !iris(f, (sx + i % w) as f64 + 0.5, (sy + i / w) as f64 + 0.5)
                    })
                    .collect()
            });
            selected[group].push(Frame {
                f: f.clone(),
                w,
                h,
                sx,
                sy,
                color,
                map,
                masks,
                globes,
            });
        }
    }
    let mut projection_reports = vec![];
    for (k, frames) in selected.iter_mut().enumerate() {
        frames.sort_by_key(|f| ui(&f.f["sequence"]));
        assert_eq!(frames.len(), 2);
        sheet(frames, &out.join(format!("contact-{k}.svg")), lids)?;
        projection_reports.push(projection_audit(
            frames,
            &out.join(format!("projection-audit-{k}.svg")),
        )?);
    }
    fs::write(
        out.join("projection-audit.json"),
        serde_json::to_vec_pretty(&projection_reports)?,
    )?;
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"100 RAW exposures; evaluation only; cached historical poses. SpecularMapTracker no temporal registration, same per-frame map for both candidates. Quantized display score squared to undo sqrt presentation. No physical lighting/anatomy/occlusion model, no independent sign truth. Shape settings preserve iris circle radius6 at ellipsoid section. No SN-FEIDA or ellipse changes.","frames":rows}),
        )?,
    )?;
    println!("{} frames written to {}", rows.len(), out.display());
    Ok(())
}
