//! Branch-independent color/connectivity proxy for visible sclera containment.
//! Neither proxy pixels nor finite shape-grid preferences are anatomical/sign truth.
#![allow(dead_code)]
#[path = "../raw10.rs"]
mod raw10;
#[path = "../raw_preview.rs"]
mod raw_preview;
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
fn coordinates(f: &Value, x: f64, y: f64) -> (f64, f64, f64) {
    let e = &f["ellipse"];
    let dx = x - num(&e["center_sensor_px"][0]);
    let dy = y - num(&e["center_sensor_px"][1]);
    let t = num(&e["angle"]);
    let u = (dx * t.cos() + dy * t.sin()) / num(&e["a"]);
    let v = (-dx * t.sin() + dy * t.cos()) / num(&e["b"]);
    (u, v, u.hypot(v))
}
fn proxy(
    f: &Value,
    color: &[u32],
    w: usize,
    h: usize,
    sx: usize,
    sy: usize,
    strict: bool,
    fixed: bool,
) -> Vec<(usize, usize)> {
    let gw = w / 2;
    let gh = h / 2;
    let mut candidates = vec![false; gw * gh];
    for y in 0..gh {
        for x in 0..gw {
            let p = color[(y * 2) * w + x * 2];
            let r = ((p >> 16) & 255) as f64;
            let g = ((p >> 8) & 255) as f64;
            let b = (p & 255) as f64;
            let (u, v, rho) = coordinates(f, (sx + x * 2) as f64, (sy + y * 2) as f64);
            let e = &f["ellipse"];
            let a = num(&e["a"]);
            let minor = num(&e["b"]);
            let t = num(&e["angle"]);
            let hx = (a * a * t.cos().powi(2) + minor * minor * t.sin().powi(2)).sqrt();
            let hy = (a * a * t.sin().powi(2) + minor * minor * t.cos().powi(2)).sqrt();
            let dx = (sx + x * 2) as f64 - num(&e["center_sensor_px"][0]);
            let dy = (sy + y * 2) as f64 - num(&e["center_sensor_px"][1]);
            let lateral = if fixed {
                (dx / hx).abs() > 0.6 && (dy / hy).abs() < 0.6
            } else {
                u.abs() > 0.6 && v.abs() < 0.85
            };
            let light = (r + g + b) / 3.;
            let chroma = r.max(g).max(b) - r.min(g).min(b);
            candidates[y * gw + x] = rho > 1.04
                && rho < 1.85
                && lateral
                && light > 50.
                && light < 245.
                && chroma < if strict { 105. } else { 145. }
                && b > r * if strict { 1.12 } else { 0.98 }
                && g > r * if strict { 1.03 } else { 0.94 };
        }
    }
    let mut visited = vec![false; gw * gh];
    let mut result = vec![];
    for i in 0..candidates.len() {
        if !candidates[i] || visited[i] {
            continue;
        }
        let mut stack = vec![i];
        visited[i] = true;
        let mut component = vec![];
        let mut touches = false;
        while let Some(j) = stack.pop() {
            let x = j % gw;
            let y = j / gw;
            component.push((x * 2, y * 2));
            let (_, _, rho) = coordinates(f, (sx + x * 2) as f64, (sy + y * 2) as f64);
            touches |= rho < 1.19;
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let xx = x as i32 + dx;
                let yy = y as i32 + dy;
                if xx >= 0 && yy >= 0 && xx < gw as i32 && yy < gh as i32 {
                    let k = yy as usize * gw + xx as usize;
                    if candidates[k] && !visited[k] {
                        visited[k] = true;
                        stack.push(k);
                    }
                }
            }
        }
        if touches && component.len() >= 8 {
            result.extend(component);
        }
    }
    result
}
fn contains(g: &Globe, x: f64, y: f64) -> bool {
    [-3., 0., 3.]
        .into_iter()
        .any(|dx| [-3., 0., 3.].into_iter().any(|dy| g.hits(x + dx, y + dy)))
}
struct Show {
    f: Value,
    w: usize,
    h: usize,
    sx: usize,
    sy: usize,
    color: Vec<u32>,
    samples: Vec<(usize, usize)>,
    status: String,
}
fn sheet(frames: &[Show], out: &Path) -> Result<(), E> {
    let mut s=String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1000' height='1620'><rect width='100%' height='100%' fill='#161b23'/><g font-family='sans-serif' fill='white' font-size='16'>");
    write!(s,"<text x='20' y='28'>Branch-independent sclera-like samples: same yellow pixels on both hypotheses</text><text x='20' y='52'>Sphere radius12mm; cyan/orange contour = hypothetical globe; white = shared iris ellipse</text>")?;
    for (row, f) in frames.iter().enumerate() {
        for k in 0..2 {
            let ox = 20 + k * 490;
            let oy = 100 + row * 355;
            write!(s,"<text x='{ox}' y='{}'>{} eye{} seq{} / {} — {}</text><g transform='translate({ox},{oy})'>",oy-14,f.f["selection"]["capture"].as_str().unwrap(),ui(&f.f["eye"]),ui(&f.f["sequence"]),if k==0{"A"}else{"B"},f.status.replace("horizontal ","H ").replace("abstain:insufficient-proxy","insufficient").replace("conditional-","cue "))?;
            for y in (0..f.h).step_by(2) {
                for x in (0..f.w).step_by(2) {
                    write!(
                        s,
                        "<rect x='{x}' y='{y}' width='2' height='2' fill='#{:06x}'/>",
                        f.color[y * f.w + x]
                    )?;
                }
            }
            let g = Globe::new(&f.f["poses"][k], 12., 12.);
            let c = if k == 0 { "#20e7ee" } else { "#ffac40" };
            for y in 1..f.h - 1 {
                for x in 1..f.w - 1 {
                    let xx = (f.sx + x) as f64;
                    let yy = (f.sy + y) as f64;
                    if g.hits(xx, yy)
                        && (!g.hits(xx + 1., yy)
                            || !g.hits(xx - 1., yy)
                            || !g.hits(xx, yy + 1.)
                            || !g.hits(xx, yy - 1.))
                    {
                        write!(s, "<rect x='{x}' y='{y}' width='1' height='1' fill='{c}'/>")?;
                    }
                }
            }
            for &(x, y) in &f.samples {
                write!(
                    s,
                    "<rect x='{x}' y='{y}' width='1' height='1' fill='#ffff00'/>"
                )?;
            }
            let e = &f.f["ellipse"];
            let cx = num(&e["center_sensor_px"][0]) - f.sx as f64;
            let cy = num(&e["center_sensor_px"][1]) - f.sy as f64;
            write!(s,"<ellipse cx='{cx}' cy='{cy}' rx='{}' ry='{}' transform='rotate({} {cx} {cy})' fill='none' stroke='white'/></g>",num(&e["a"]),num(&e["b"]),num(&e["angle"]).to_degrees())?;
        }
    }
    write!(s,"<text x='20' y='1545'>Yellow = color/connectivity proxy, not human sclera labels. Skin and glints can pass; absence means unknown.</text><text x='20' y='1570'>Containment across sizes can refute a branch only IF samples really are sclera and the geometric family is valid.</text><text x='20' y='1595'>Boundary allowance3px; unilateral evidence allowed. Globe geometric center is not the eye rotation pivot.</text></g></svg>")?;
    fs::write(out, s)?;
    Ok(())
}
fn main() -> Result<(), E> {
    let out = Path::new("outputs/sclera-visibility-20260918");
    fs::create_dir_all(out)?;
    let root: Value = serde_json::from_slice(&fs::read(
        "outputs/limbus-sign-probe-20260916-neighbors/results.json",
    )?)?;
    let mut rows = vec![];
    let make_video = std::env::args().any(|a| a == "--video");
    let mut video_frames = vec![];
    let mut show = vec![];
    let mut failures = vec![];
    let mut failure_count = [0usize; 2];
    let mut fixed_show = vec![];
    let mut fixed_failures = vec![];
    let mut fixed_cases = vec![];
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
        let mut modes = vec![];
        let mut strict_samples = vec![];
        for (strict, fixed) in [(true, false), (false, false), (true, true), (false, true)] {
            let samples = proxy(f, &color, w, h, sx, sy, strict, fixed);
            let mut sides = [0usize; 2];
            for &(x, y) in &samples {
                let (u, _, _) = coordinates(f, (sx + x) as f64, (sy + y) as f64);
                sides[(if fixed {
                    (sx + x) as f64 > num(&f["ellipse"]["center_sensor_px"][0])
                } else {
                    u > 0.
                }) as usize] += 1;
            }
            let bilateral = sides.iter().all(|&n| n >= 12);
            let mut models = vec![];
            for ratio in [1.6, 2., 2.4] {
                for axial in [0.9, 1., 1.1] {
                    let a = 6. * ratio;
                    let b = a * axial;
                    let fractions: [f64; 2] = std::array::from_fn(|k| {
                        let g = Globe::new(&f["poses"][k], a, b);
                        samples
                            .iter()
                            .filter(|&&(x, y)| contains(&g, (sx + x) as f64, (sy + y) as f64))
                            .count() as f64
                            / samples.len().max(1) as f64
                    });
                    models.push(json!({"radius_to_iris":ratio,"axial_ratio":axial,"contained_fraction":fractions}));
                }
            }
            let best: [f64; 2] = std::array::from_fn(|k| {
                models
                    .iter()
                    .map(|m| num(&m["contained_fraction"][k]))
                    .fold(0., f64::max)
            });
            let status = if samples.len() < 12 {
                "abstain:insufficient-proxy"
            } else if best[0] >= 0.98 && best[1] < 0.90 {
                "conditional-A"
            } else if best[1] >= 0.98 && best[0] < 0.90 {
                "conditional-B"
            } else if best[0] < 0.90 && best[1] < 0.90 {
                "inconsistent-proxy-or-geometry"
            } else {
                "unresolved"
            };
            modes.push(json!({"strict":strict,"camera_horizontal":fixed,"side_samples":sides,"bilateral":bilateral,"selected_roi_pixels":samples,"models":models,"best_containment":best,"status":status}));
            if strict && !fixed {
                strict_samples = samples;
            }
        }
        let loose_status = modes[1]["status"].as_str().unwrap();
        if let Some(k) = match loose_status {
            "conditional-A" => Some(0),
            "conditional-B" => Some(1),
            _ => None,
        } {
            if failure_count[k] < 2 {
                failure_count[k] += 1;
                fixed_failures.push(Show {
                    f: f.clone(),
                    w,
                    h,
                    sx,
                    sy,
                    color: color.clone(),
                    samples: proxy(f, &color, w, h, sx, sy, false, true),
                    status: format!("horizontal {}", modes[3]["status"].as_str().unwrap()),
                });
                failures.push(Show {
                    f: f.clone(),
                    w,
                    h,
                    sx,
                    sy,
                    color: color.clone(),
                    samples: proxy(f, &color, w, h, sx, sy, false, false),
                    status: format!("loose {loose_status}"),
                });
            }
        }
        if make_video {
            video_frames.push(VideoFrame {
                f: f.clone(),
                color: color.clone(),
                modes: modes.clone(),
                hash: hash.clone(),
            });
        }
        let seq = ui(&f["sequence"]);
        let eye = ui(&f["eye"]);
        if (eye == 2 && (seq == 849 || seq == 850)) || (eye == 1 && (seq == 876 || seq == 877)) {
            fixed_cases.push(Show {
                f: f.clone(),
                w,
                h,
                sx,
                sy,
                color: color.clone(),
                samples: proxy(f, &color, w, h, sx, sy, true, true),
                status: format!("horizontal {}", modes[2]["status"].as_str().unwrap()),
            });
        }
        if (eye == 1 && (seq == 60 || seq == 61)) || (eye == 2 && (seq == 1034 || seq == 1035)) {
            fixed_show.push(Show {
                f: f.clone(),
                w,
                h,
                sx,
                sy,
                color: color.clone(),
                samples: proxy(f, &color, w, h, sx, sy, true, true),
                status: format!("horizontal {}", modes[2]["status"].as_str().unwrap()),
            });
            show.push(Show {
                f: f.clone(),
                w,
                h,
                sx,
                sy,
                color,
                samples: strict_samples,
                status: modes[0]["status"].as_str().unwrap().to_string(),
            });
        }
        rows.push(json!({"capture":f["selection"]["capture"],"sequence":seq,"eye":eye,"modes":modes,"raw_sha256":hash}));
    }
    show.sort_by_key(|f| ui(&f.f["sequence"]));
    sheet(&show, &out.join("contact.svg"))?;
    sheet(&failures, &out.join("loose-failures.svg"))?;
    fixed_show.sort_by_key(|f| ui(&f.f["sequence"]));
    sheet(&fixed_show, &out.join("horizontal-contact.svg"))?;
    sheet(&fixed_failures, &out.join("horizontal-loose-failures.svg"))?;
    fixed_cases.sort_by_key(|f| ui(&f.f["sequence"]));
    sheet(&fixed_cases, &out.join("horizontal-conditional-cases.svg"))?;
    let mut summary = serde_json::Map::new();
    for k in 0..4 {
        let mut counts = std::collections::BTreeMap::new();
        for r in &rows {
            *counts
                .entry(r["modes"][k]["status"].as_str().unwrap().to_string())
                .or_insert(0usize) += 1;
        }
        summary.insert(
            ["strict", "loose", "horizontal_strict", "horizontal_loose"][k].into(),
            json!(counts),
        );
    }
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"100 legacy RAW frames; no independent sign/sclera labels; current color evidence is branch-independent; thresholds heuristic; geometric globe center distinct from rotation pivot","frames":rows,"summary":summary}),
        )?,
    )?;
    println!("{}", json!(summary));
    if make_video {
        video_export(&mut video_frames, out)?;
    }
    Ok(())
}

struct VideoFrame {
    f: Value,
    color: Vec<u32>,
    modes: Vec<Value>,
    hash: String,
}
fn b64(bytes: &[u8]) -> String {
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in bytes.chunks(3) {
        let v = ((c[0] as u32) << 16)
            | ((c.get(1).copied().unwrap_or(0) as u32) << 8)
            | c.get(2).copied().unwrap_or(0) as u32;
        for j in 0..4 {
            s.push(if j > c.len() {
                '='
            } else {
                alphabet[((v >> (18 - j * 6)) & 63) as usize] as char
            });
        }
    }
    s
}
fn video_text(s: &mut String, x: usize, y: usize, size: usize, t: &str) {
    let t = t
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    write!(s, "<text x='{x}' y='{y}' font-size='{size}'>{t}</text>").unwrap();
}
fn video_export(frames: &mut [VideoFrame], out: &Path) -> Result<(), E> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let out = out.join("video");
    fs::create_dir_all(out.join("raw"))?;
    frames.sort_by_key(|f| {
        (
            f.f["selection"]["capture"].as_str().unwrap().to_string(),
            ui(&f.f["eye"]),
            f.f["input"]["frame"]["timestamp_ns"].as_u64().unwrap(),
        )
    });
    let mut encoder = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "420x280",
            "-framerate",
            "10",
            "-i",
            "pipe:0",
            "-c:v",
            "png",
            "-compression_level",
            "2",
            "-threads",
            "2",
            "-start_number",
            "0",
        ])
        .arg(out.join("raw/%03d.png"))
        .stdin(Stdio::piped())
        .spawn()?;
    let mut stdin = encoder.stdin.take().ok_or("PNG stdin missing")?;
    for f in frames.iter() {
        assert_eq!(f.color.len(), 420 * 280);
        let bytes = f
            .color
            .iter()
            .flat_map(|p| {
                [
                    ((p >> 16) & 255) as u8,
                    ((p >> 8) & 255) as u8,
                    (p & 255) as u8,
                ]
            })
            .collect::<Vec<_>>();
        stdin.write_all(&bytes)?;
    }
    drop(stdin);
    if !encoder.wait()?.success() {
        return Err("RAW PNG encoder failed".into());
    }
    let group_key = |f: &VideoFrame| {
        (
            f.f["selection"]["capture"].as_str().unwrap().to_string(),
            ui(&f.f["eye"]),
            ui(&f.f["selection"]["sequence"]),
        )
    };
    let mut groups: Vec<Vec<usize>> = vec![];
    for i in 0..frames.len() {
        if groups
            .last()
            .is_none_or(|g| group_key(&frames[g[0]]) != group_key(&frames[i]))
        {
            groups.push(vec![])
        }
        groups.last_mut().unwrap().push(i);
    }
    let mut concat = String::from("ffconcat version 1.0\n");
    let mut timeline = vec![];
    let mut at = 0.;
    let mut lastfile = String::new();
    for (gi, indices) in groups.iter().enumerate() {
        let first = &frames[indices[0]];
        let key = group_key(first);
        let ns0 = first.f["input"]["frame"]["timestamp_ns"].as_u64().unwrap();
        let mut dt = indices
            .windows(2)
            .map(|ij| {
                let a = frames[ij[0]].f["input"]["frame"]["timestamp_ns"]
                    .as_u64()
                    .unwrap();
                let b = frames[ij[1]].f["input"]["frame"]["timestamp_ns"]
                    .as_u64()
                    .unwrap();
                assert!(b > a);
                (b - a) as f64 / 1e9
            })
            .collect::<Vec<_>>();
        dt.sort_by(f64::total_cmp);
        let cadence = dt.get(dt.len() / 2).copied().unwrap_or(0.1);
        let mut cut=String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1800' height='900'><rect width='100%' height='100%' fill='#141923'/><g font-family='sans-serif' fill='white'>");
        video_text(
            &mut cut,
            90,
            320,
            45,
            &format!("Separate neighborhood {} / {}", gi + 1, groups.len()),
        );
        video_text(
            &mut cut,
            90,
            400,
            32,
            &format!(
                "{} · eye {} · sources {}–{}",
                key.0,
                key.1,
                ui(&first.f["sequence"]),
                ui(&frames[*indices.last().unwrap()].f["sequence"])
            ),
        );
        video_text(&mut cut,90,480,26,"Cut between eye/time intervals — not continuous footage. Next exposures play 4× slower.");
        video_text(
            &mut cut,
            90,
            535,
            26,
            "Conditional sclera geometry diagnostic; no verified gaze or sign labels.",
        );
        cut.push_str("</g></svg>");
        let cutfile = format!("cut-{gi:02}.svg");
        fs::write(out.join(&cutfile), cut)?;
        writeln!(concat, "file '{cutfile}'\nduration 0.800000000")?;
        timeline.push(json!({"file":cutfile,"kind":"labelled-cut","video_start_s":at,"duration_s":0.8,"next_capture":key.0,"next_eye":key.1,"next_source_ns":ns0}));
        at += 0.8;
        for (pos, &i) in indices.iter().enumerate() {
            let f = &frames[i];
            let meta = &f.f["input"]["frame"];
            let sx = ui(&meta["sensor_x"]);
            let sy = ui(&meta["sensor_y"]);
            let source_ns = meta["timestamp_ns"].as_u64().unwrap();
            let duration = if let Some(&j) = indices.get(pos + 1) {
                (frames[j].f["input"]["frame"]["timestamp_ns"]
                    .as_u64()
                    .unwrap()
                    - source_ns) as f64
                    / 1e9
                    * 4.
            } else {
                cadence * 4.
            };
            let mut s=String::from("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='1800' height='900'><rect width='100%' height='100%' fill='#141923'/><g font-family='sans-serif' fill='white'>");
            video_text(
                &mut s,
                30,
                38,
                29,
                &format!(
                    "{} · eye {} · source {} · {:.3}s into neighborhood · 4× slower",
                    key.0,
                    key.1,
                    ui(&f.f["sequence"]),
                    (source_ns - ns0) as f64 / 1e9
                ),
            );
            video_text(&mut s,30,77,24,"Same yellow camera-horizontal strict samples on both hypotheses — conditional, NOT verified gaze");
            let image = b64(&fs::read(out.join(format!("raw/{i:03}.png")))?);
            for k in 0..2 {
                let ox = 30 + k * 885;
                let oy = 125;
                let g = Globe::new(&f.f["poses"][k], 12., 12.);
                let color = if k == 0 { "#28e4ef" } else { "#ffac42" };
                video_text(
                    &mut s,
                    ox,
                    112,
                    24,
                    if k == 0 {
                        "Hypothesis A"
                    } else {
                        "Hypothesis B"
                    },
                );
                write!(s,"<clipPath id='clip{k}'><rect x='0' y='0' width='420' height='280'/></clipPath><g transform='translate({ox},{oy}) scale(2)' clip-path='url(#clip{k})'><image width='420' height='280' xlink:href='data:image/png;base64,{image}'/>")?;
                let mut edges = String::new();
                for y in 1..279 {
                    for x in 1..419 {
                        let xx = (sx + x) as f64;
                        let yy = (sy + y) as f64;
                        if g.hits(xx, yy)
                            && (!g.hits(xx + 1., yy)
                                || !g.hits(xx - 1., yy)
                                || !g.hits(xx, yy + 1.)
                                || !g.hits(xx, yy - 1.))
                        {
                            write!(edges, "M{x},{y}h1")?;
                        }
                    }
                }
                write!(s, "<path d='{edges}' stroke='{color}' stroke-width='1.2'/>")?;
                let mut points = String::new();
                for p in f.modes[2]["selected_roi_pixels"].as_array().unwrap() {
                    write!(points, "M{},{}h1", ui(&p[0]), ui(&p[1]))?;
                }
                write!(s, "<path d='{points}' stroke='#ffff00' stroke-width='.7'/>")?;
                let e = &f.f["ellipse"];
                let cx = num(&e["center_sensor_px"][0]) - sx as f64;
                let cy = num(&e["center_sensor_px"][1]) - sy as f64;
                write!(s,"<ellipse cx='{cx}' cy='{cy}' rx='{}' ry='{}' transform='rotate({} {cx} {cy})' fill='none' stroke='white' stroke-width='.7'/>",num(&e["a"]),num(&e["b"]),num(&e["angle"]).to_degrees())?;
                let center = project(g.center);
                let x = center[0] - sx as f64;
                let y = center[1] - sy as f64;
                write!(
                    s,
                    "<path d='M{},{}h12 M{},{}v12' stroke='{color}' stroke-width='1.4'/></g>",
                    x - 6.,
                    y,
                    x,
                    y - 6.
                )?;
                video_text(
                    &mut s,
                    ox,
                    720,
                    24,
                    &format!(
                        "Best-grid containment: {:.1}%",
                        100. * num(&f.modes[2]["best_containment"][k])
                    ),
                );
            }
            video_text(
                &mut s,
                30,
                762,
                24,
                &format!(
                    "Strict: iris-axis {} | horizontal {} | loose-horizontal {}",
                    f.modes[0]["status"].as_str().unwrap(),
                    f.modes[2]["status"].as_str().unwrap(),
                    f.modes[3]["status"].as_str().unwrap()
                ),
            );
            video_text(&mut s,30,800,22,&format!("Selected left/right: {}/{} · contour shown: 12mm sphere · + geometric globe center (not rotation pivot)",f.modes[2]["side_samples"][0],f.modes[2]["side_samples"][1]));
            video_text(&mut s,30,837,21,"Scores search nine assumed shapes; finite grid is not a continuous exclusion proof. Color proxy can include lids/skin.");
            video_text(&mut s,30,872,21,"Missing/occluded pixels are unknown. Exact source intervals slowed 4×; last exposure held for local median cadence.");
            s.push_str("</g></svg>");
            let file = format!("frame-{i:03}.svg");
            fs::write(out.join(&file), s)?;
            writeln!(concat, "file '{file}'\nduration {duration:.9}")?;
            timeline.push(json!({"file":file,"kind":"RAW-exposure","source_ns":source_ns,"source_sequence":f.f["sequence"],"source_key":meta["source_clock"]["source_key"],"capture":key.0,"eye":key.1,"raw_sha256":f.hash,"raw_input":f.f["input"],"ellipse":f.f["ellipse"],"poses":f.f["poses"],"video_start_s":at,"duration_s":duration,"timing_rule":if pos+1==indices.len(){"terminal hold:4x neighborhood median cadence"}else{"4x actual next-source interval"},"strict_baseline":f.modes[0]["status"],"horizontal_strict":f.modes[2]["status"],"best_containment":f.modes[2]["best_containment"]}));
            at += duration;
            lastfile = file;
        }
    }
    writeln!(concat, "file '{lastfile}'")?;
    fs::write(out.join("video.ffconcat"), concat)?;
    fs::write(
        out.join("timing-provenance.json"),
        serde_json::to_vec_pretty(
            &json!({"exposures":frames.len(),"neighborhoods":groups.len(),"nominal_duration_s":at,"speed_factor":4,"render_fps":25,"timestamp_quantization":"MP4 25fps; actual requested source-derived holds retained here. Export is trimmed to nominal duration; end time quantizes to the nearest25fps frame.","generator_sha256":format!("{:x}",Sha256::digest(include_bytes!("buttercup_sclera_visibility.rs"))),"frames":timeline}),
        )?,
    )?;
    let result = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "concat", "-safe", "0", "-i"])
        .arg(out.join("video.ffconcat"))
        .args([
            "-vf",
            "fps=25",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            "-movflags",
            "+faststart",
        ])
        .args(["-t", &format!("{at:.9}")])
        .arg(out.join("sclera-hypotheses.mp4"))
        .status()?;
    if !result.success() {
        return Err("MP4 export failed".into());
    }
    println!(
        "video {} exposures, {} neighborhoods, {:.3}s",
        frames.len(),
        groups.len(),
        at
    );
    Ok(())
}
