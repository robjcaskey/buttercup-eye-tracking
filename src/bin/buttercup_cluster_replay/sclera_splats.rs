//! Bounded CPU Gaussian surface-map pilot. Geometry is conditional on a stated
//! sphere; native sclera-only matches fit pose, never an iris ellipse or gaze.
use super::{cohorts, json, motion, quantiles, read_rows, BundleSource, Error, Value};
use buttercup_eye_tracking::raw10;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
    sync::Arc,
    time::Instant,
};
type Result<T> = std::result::Result<T, Error>;
#[path = "../../bootstrapability.rs"]
#[allow(dead_code)]
mod boot;
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
#[path = "splat_geometry.rs"]
mod geom;
#[path = "../../sclera_splat_input_recipe.rs"]
mod recipe;
use canvas::*;
use geom::{Pair, Pose, Splat, CENTER, RADIUS};
const W: usize = 420;
const H: usize = 280;
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn write(path: impl AsRef<Path>, v: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}
fn n(v: &Value) -> u64 {
    v.as_u64().expect("native integer metadata")
}
struct Frame {
    row: Value,
    raw: Arc<Vec<u16>>,
    rgb: Vec<u8>,
    core: Vec<bool>,
    support_stats: Value,
}
impl Frame {
    fn sx(&self) -> u32 {
        n(&self.row["source"]["sensor_x"]) as u32
    }
    fn sy(&self) -> u32 {
        n(&self.row["source"]["sensor_y"]) as u32
    }
    fn color(&self, j: usize) -> [f64; 3] {
        std::array::from_fn(|k| self.rgb[k * W * H + j] as f64)
    }
    fn support(&self, p: [f32; 2]) -> bool {
        let x = (p[0] - self.sx() as f32).round() as i32;
        let y = (p[1] - self.sy() as f32).round() as i32;
        if x < 8 || y < 8 || x >= W as i32 - 8 || y >= H as i32 - 8 {
            return false;
        }
        (-6..=6).all(|dy| (-6..=6).all(|dx| self.core[(y + dy) as usize * W + (x + dx) as usize]))
    }
    fn bgra(&self, masked: bool) -> Vec<u8> {
        (0..W * H)
            .flat_map(|j| {
                let c = self.color(j);
                let scale = if masked && !self.core[j] { 0.12 } else { 1. };
                [
                    (c[2] * scale) as u8,
                    (c[1] * scale) as u8,
                    (c[0] * scale) as u8,
                    255,
                ]
            })
            .collect()
    }
}
fn core(mask: &[u8], raw: &[u16], sx: u32, sy: u32) -> (Vec<bool>, Value) {
    // RAW colour channels clip independently. Losing red alone does not erase
    // measured green texture. Veto cells with >=6 of 8 clipped green sites,
    // using the physical Quad Bayer phase, before the boundary guard.
    let mut clipped = vec![true; W * H];
    let ox = (4 - sx as usize % 4) % 4;
    let oy = (4 - sy as usize % 4) % 4;
    for y in (oy..H - 3).step_by(4) {
        for x in (ox..W - 3).step_by(4) {
            let mut green = 0;
            for dy in 0..4 {
                for dx in 0..4 {
                    if (dx < 2) != (dy < 2) && raw[(y + dy) * W + x + dx] >= 1018 {
                        green += 1;
                    }
                }
            }
            for dy in 0..4 {
                for dx in 0..4 {
                    clipped[(y + dy) * W + x + dx] = green >= 6;
                }
            }
        }
    }
    let mut channels = [0usize; 3];
    let mut saturated = [0usize; 3];
    for y in 0..H {
        for x in 0..W {
            let j = y * W + x;
            if mask[j] >= 179 {
                let red = (sx as usize + x) % 4 < 2;
                let top = (sy as usize + y) % 4 < 2;
                let k = if red && top {
                    0
                } else if !red && !top {
                    2
                } else {
                    1
                };
                channels[k] += 1;
                saturated[k] += usize::from(raw[j] >= 1018);
            }
        }
    }
    let mut out = vec![false; W * H];
    for y in 4..H - 4 {
        for x in 4..W - 4 {
            out[y * W + x] = (-4..=4).all(|dy| {
                (-4..=4).all(|dx| {
                    let j = (y as i32 + dy) as usize * W + (x as i32 + dx) as usize;
                    mask[j] >= 179 && !clipped[j]
                })
            });
        }
    }
    let stats = json!({"mask_pixels":mask.iter().filter(|&&v|v>=179).count(),"retained_pixels":out.iter().filter(|&&v|v).count(),"raw_channel_samples_rgb":channels,"clipped_channel_samples_rgb":saturated});
    (out, stats)
}
fn pairs(a: &Frame, b: &Frame) -> Vec<Pair> {
    let mut tracker = motion::NativeGlobalSimilarityTracker::default();
    tracker.observe_diagnostic_where(
        a.raw.clone(),
        W,
        H,
        a.sx(),
        a.sy(),
        [40, 28],
        |_| true,
        |p| a.support(p),
    );
    tracker
        .observe_diagnostic_where(
            b.raw.clone(),
            W,
            H,
            b.sx(),
            b.sy(),
            [40, 28],
            |p| a.support(p),
            |p| b.support(p),
        )
        .into_iter()
        .filter(|m| m.photometric_score >= 0.70 && m.distinct_match_margin >= 0.04)
        .map(|m| Pair {
            a: [
                m.previous_sensor_px[0] as f64 - a.sx() as f64,
                m.previous_sensor_px[1] as f64 - a.sy() as f64,
            ],
            b: [
                m.current_sensor_px[0] as f64 - b.sx() as f64,
                m.current_sensor_px[1] as f64 - b.sy() as f64,
            ],
        })
        .collect()
}

struct GreenTexture {
    w: usize,
    h: usize,
    origin: [f64; 2],
    values: Vec<f64>,
}
impl GreenTexture {
    fn new(f: &Frame) -> Self {
        let ox = (4 - f.sx() as usize % 4) % 4;
        let oy = (4 - f.sy() as usize % 4) % 4;
        let w = (W - ox) / 4;
        let h = (H - oy) / 4;
        let mut g = [vec![0.; w * h], vec![0.; w * h]];
        for y in 0..h {
            for x in 0..w {
                for (k, (dx, dy)) in [(2, 0), (0, 2)].into_iter().enumerate() {
                    let j = (oy + y * 4 + dy) * W + ox + x * 4 + dx;
                    g[k][y * w + x] = (f.raw[j] as f64
                        + f.raw[j + 1] as f64
                        + f.raw[j + W] as f64
                        + f.raw[j + W + 1] as f64)
                        * 0.25;
                }
            }
        }
        let mut collocated = vec![0.; w * h];
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                for (k, (dx, dy)) in [(-1isize, 1isize), (1, -1)].into_iter().enumerate() {
                    let xx = x.checked_add_signed(dx).unwrap();
                    let yy = y.checked_add_signed(dy).unwrap();
                    collocated[y * w + x] += 0.5
                        * (0.5625 * g[k][y * w + x]
                            + 0.1875 * (g[k][y * w + xx] + g[k][yy * w + x])
                            + 0.0625 * g[k][yy * w + xx]);
                }
            }
        }
        let mut values = vec![0.; w * h];
        for y in 2..h - 2 {
            for x in 2..w - 2 {
                let mut mean = 0.;
                for yy in y - 1..=y + 1 {
                    for xx in x - 1..=x + 1 {
                        mean += collocated[yy * w + xx] / 9.;
                    }
                }
                values[y * w + x] = collocated[y * w + x] - mean;
            }
        }
        Self {
            w,
            h,
            origin: [ox as f64 + 1.5, oy as f64 + 1.5],
            values,
        }
    }
    fn sample(&self, p: [f64; 2]) -> Option<f64> {
        let x = (p[0] - self.origin[0]) / 4.;
        let y = (p[1] - self.origin[1]) / 4.;
        if x < 2. || y < 2. || x >= self.w as f64 - 3. || y >= self.h as f64 - 3. {
            return None;
        }
        let xx = x.floor() as usize;
        let yy = y.floor() as usize;
        let fx = x - xx as f64;
        let fy = y - yy as f64;
        Some(
            (1. - fy)
                * ((1. - fx) * self.values[yy * self.w + xx]
                    + fx * self.values[yy * self.w + xx + 1])
                + fy * ((1. - fx) * self.values[(yy + 1) * self.w + xx]
                    + fx * self.values[(yy + 1) * self.w + xx + 1]),
        )
    }
}

fn texture_translation(a: &Frame, b: &Frame) -> (Option<Pose>, Value) {
    let ga = GreenTexture::new(a);
    let gb = GreenTexture::new(b);
    let strict = (0..W * H)
        .map(|j| {
            b.support([
                b.sx() as f32 + (j % W) as f32,
                b.sy() as f32 + (j / W) as f32,
            ])
        })
        .collect::<Vec<_>>();
    let mut points = [vec![], vec![]];
    for y in (12..H - 12).step_by(4) {
        for x in (12..W - 12).step_by(4) {
            let p = [x as f64, y as f64];
            if a.support([a.sx() as f32 + x as f32, a.sy() as f32 + y as f32]) {
                if let Some(v) = ga.sample(p) {
                    points[(x / 16 + y / 16) % 2].push((p, v));
                }
            }
        }
    }
    let offset = [a.sx() as f64 - b.sx() as f64, a.sy() as f64 - b.sy() as f64];
    let score = |shift: [f64; 2], fold: usize| -> Option<(f64, usize, f64)> {
        let mut sums = [0.; 5];
        let mut count = 0;
        for &(p, v) in &points[fold] {
            let q = [p[0] + shift[0], p[1] + shift[1]];
            let x = q[0].round() as i32;
            let y = q[1].round() as i32;
            if x < 0
                || y < 0
                || x >= W as i32
                || y >= H as i32
                || !strict[y as usize * W + x as usize]
            {
                continue;
            }
            let Some(u) = gb.sample(q) else { continue };
            count += 1;
            sums[0] += v;
            sums[1] += u;
            sums[2] += v * v;
            sums[3] += u * u;
            sums[4] += v * u;
        }
        if count < 24 || count * 3 < points[fold].len() * 2 {
            return None;
        }
        let va = sums[2] - sums[0] * sums[0] / count as f64;
        let vb = sums[3] - sums[1] * sums[1] / count as f64;
        let texture = (va.min(vb) / count as f64).sqrt();
        if texture < 1. {
            return None;
        }
        Some((
            (sums[4] - sums[0] * sums[1] / count as f64) / (va * vb).sqrt(),
            count,
            texture,
        ))
    };
    let mut candidates = vec![];
    for y in -12..=12 {
        for x in -12..=12 {
            let shift = [offset[0] + x as f64, offset[1] + y as f64];
            if let Some((corr, count, texture)) = score(shift, 0) {
                candidates.push((corr, shift, count, texture));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let Some(&(corr, shift, count, texture)) = candidates.first() else {
        return (
            None,
            json!({"accepted":false,"reason":"insufficient common native green texture","fold_samples":[points[0].len(),points[1].len()]}),
        );
    };
    let second = candidates
        .iter()
        .skip(1)
        .find(|q| (q.1[0] - shift[0]).hypot(q.1[1] - shift[1]) >= 3.)
        .map(|q| q.0);
    let validation = score(shift, 1);
    let accepted = corr >= 0.85
        && validation.is_some_and(|v| v.0 >= 0.85)
        && second.is_some_and(|s| corr - s >= 0.015);
    let pose = Pose {
        translation: shift,
        ..Pose::default()
    };
    (
        accepted.then_some(pose),
        json!({"accepted":accepted,"method":"masked native green high-pass translation only","shift":shift,"fit_correlation":corr,"fit_samples":count,"native_texture_std":texture,"separated_candidate_correlation":second,"spatial_validation":validation,"rotation_observable":false,"fold_policy":"alternating 16px spatial blocks; fit fold 0, score fold 1 without parameter refit; correlated nearby samples are not independent anatomy labels"}),
    )
}
fn pose_evidence(pairs: &[Pair]) -> (Option<Pose>, Value) {
    let mut span = [0.; 2];
    for k in 0..2 {
        span[k] = pairs
            .iter()
            .map(|p| p.a[k])
            .fold(f64::NEG_INFINITY, f64::max)
            - pairs.iter().map(|p| p.a[k]).fold(f64::INFINITY, f64::min);
    }
    let fitted = geom::fit(pairs, RADIUS);
    let mut errors = fitted
        .map(|pose| {
            pairs
                .iter()
                .filter_map(|&p| geom::error(p, pose, RADIUS))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    errors.sort_by(f64::total_cmp);
    let inliers = errors.iter().filter(|&&e| e <= 2.5).count();
    let accepted = pairs.len() >= 8
        && span[0] >= 45.
        && span[1] >= 20.
        && inliers * 4 >= pairs.len() * 3
        && errors.get(errors.len() / 2).is_some_and(|&e| e <= 1.5);
    // Leave whole overlapping native patches out, not just the held center.
    let mut held = vec![];
    for (i, &p) in pairs.iter().enumerate() {
        let witnesses = pairs
            .iter()
            .enumerate()
            .filter(|&(j, q)| {
                j != i
                    && (q.a[0] - p.a[0]).hypot(q.a[1] - p.a[1]) > 20.
                    && (q.b[0] - p.b[0]).hypot(q.b[1] - p.b[1]) > 20.
            })
            .map(|(_, p)| *p)
            .collect::<Vec<_>>();
        let original = witnesses.iter().map(|p| (p.a, p.b)).collect::<Vec<_>>();
        let similarity =
            cohorts::predict(&original, p.a).map(|q| (q[0] - p.b[0]).hypot(q[1] - p.b[1]));
        let radius_errors = [180., 240., 320.]
            .map(|r| geom::fit(&witnesses, r).and_then(|pose| geom::error(p, pose, r)));
        held.push(json!({"point":p.a,"witnesses":witnesses.len(),"similarity_px":similarity,"sphere_px":radius_errors}));
    }
    (
        if accepted { fitted } else { None },
        json!({"accepted":accepted,"matches":pairs.len(),"span":span.map(|v|if v.is_finite(){v}else{0.}),"inliers_2_5px":inliers,"residual_px":quantiles(errors),"pose":fitted,"held_patch_errors":held,
        "matches_local":pairs.iter().map(|p|json!({"reference":p.a,"current":p.b})).collect::<Vec<_>>()}),
    )
}
/// Each canonical cell gets at most one sample from each fresh exposure.
/// Capped color updates limit ghosts, but are not specular/material truth.
fn fuse(map: &mut BTreeMap<(i32, i32), Splat>, frame: &Frame, pose: Pose) -> usize {
    let mut frame_samples = BTreeMap::new();
    for y in (5..H - 5).step_by(2) {
        for x in (5..W - 5).step_by(2) {
            let j = y * W + x;
            if !frame.core[j] {
                continue;
            }
            let Some(v) = geom::unproject([x as f64, y as f64], pose, RADIUS) else {
                continue;
            };
            let px = CENTER[0] + v[0] * RADIUS;
            let py = CENTER[1] + v[1] * RADIUS;
            if px < 4. || py < 4. || px >= W as f64 - 4. || py >= H as f64 - 4. {
                continue;
            }
            let key = ((px / 2.).round() as i32, (py / 2.).round() as i32);
            let Some(mean) = geom::lift([key.0 as f64 * 2., key.1 as f64 * 2.], CENTER, RADIUS)
            else {
                continue;
            };
            let distance = (px - key.0 as f64 * 2.).hypot(py - key.1 as f64 * 2.);
            let replace = frame_samples
                .get(&key)
                .is_none_or(|(old, _, _): &(f64, [f64; 3], [f64; 3])| distance < *old);
            if replace {
                frame_samples.insert(key, (distance, mean, frame.color(j)));
            }
        }
    }
    let count = frame_samples.len();
    for (key, (_, mean, color)) in frame_samples {
        if let Some(s) = map.get_mut(&key) {
            let delta = (0..3)
                .map(|k| (color[k] - s.rgb[k]).abs())
                .fold(0., f64::max);
            if delta > 45. {
                continue;
            }
            s.observations += 1;
            for k in 0..3 {
                s.rgb[k] += (color[k] - s.rgb[k]) / s.observations as f64;
            }
        } else {
            map.insert(key, Splat::new(mean, color, 1.1 / RADIUS));
        }
    }
    count
}
fn photo_error(
    frame: &Frame,
    base: &[u8],
    candidate: &[u8],
    bmass: &[f64],
    cmass: &[f64],
) -> Value {
    let mut b = 0.;
    let mut c = 0.;
    let mut common = 0;
    let mut cov = [0; 2];
    let mut mask = 0;
    for j in 0..W * H {
        if frame.core[j] {
            mask += 1;
            cov[0] += usize::from(bmass[j] >= 0.3);
            cov[1] += usize::from(cmass[j] >= 0.3);
            if bmass[j] >= 0.3 && cmass[j] >= 0.3 {
                common += 1;
                for k in 0..3 {
                    let actual = frame.rgb[k * W * H + j] as f64;
                    b += (actual - base[4 * j + 2 - k] as f64).abs();
                    c += (actual - candidate[4 * j + 2 - k] as f64).abs();
                }
            }
        }
    }
    json!({"masked_pixels":mask,"baseline_covered":cov[0],"map_covered":cov[1],"common_pixels":common,"baseline_rgb_mae":(common>0).then(||b/(3*common) as f64),"map_rgb_mae":(common>0).then(||c/(3*common) as f64),"role":"held-out color only; held-frame pose uses its RAW matches; no anatomical or correspondence truth"})
}
fn panel(
    frame: &Frame,
    map: &[Splat],
    baseline: &[Splat],
    pose: Option<Pose>,
    i: usize,
    eye: u64,
    split: usize,
    evidence: &Value,
) -> Result<Canvas> {
    let mut c = Canvas::new(1296, 790)?;
    c.clear();
    c.text(
        15.,
        30.,
        23.,
        WHITE,
        "SCLERA GAUSSIAN SURFACE PREVIEW | CPU prototype",
    );
    c.text(15.,58.,16.,MUTED,"Round surface is an assumed sphere. Blank regions are unobserved; these are not measured 3D anatomy.");
    c.text(
        15.,
        85.,
        17.,
        WHITE,
        &format!(
            "Eye {eye} | source sequence {} | {} | {} native patch matches",
            frame.row["source"]["sequence"],
            if i < split {
                "mapping exposure"
            } else {
                "held-out texture exposure"
            },
            evidence["matches"]
        ),
    );
    for (x, label) in [
        (10., "RAW color"),
        (438., "Sclera support (SAM + boundary/clipping guard)"),
        (866., "Map reprojection: requires accepted alignment"),
    ] {
        c.text(x, 115., 15., WHITE, label);
    }
    c.image(&frame.bgra(false), W, H, 10., 130., 420., 280.);
    c.image(&frame.bgra(true), W, H, 438., 130., 420., 280.);
    if let Some(pose) = pose {
        let (view, _) = geom::render(map, pose, RADIUS, W, H);
        c.image(&view, W, H, 866., 130., 420., 280.);
    } else {
        c.text(890., 260., 20., ORANGE, "No supported alignment");
    }
    let (reference, _) = geom::render(map, Pose::default(), RADIUS, W, H);
    let (base, _) = geom::render(baseline, Pose::default(), RADIUS, W, H);
    let (side, _) = geom::render(
        map,
        Pose {
            angle: [0., 0.25, 0.],
            ..Pose::default()
        },
        RADIUS,
        W,
        H,
    );
    for (x, label) in [
        (10., "Single-exposure Gaussian map"),
        (
            438.,
            if map.iter().any(|s| s.observations > 1) {
                "Combined map in reference coordinates"
            } else {
                "Reference retained: no fusion accepted"
            },
        ),
        (866., "Novel angle: conditional sphere visualization"),
    ] {
        c.text(x, 442., 15., WHITE, label);
    }
    c.image(&base, W, H, 10., 456., 420., 280.);
    c.image(&reference, W, H, 438., 456., 420., 280.);
    c.image(&side, W, H, 866., 456., 420., 280.);
    c.text(15.,765.,15.,MUTED,"No iris ellipse or gaze target. Mask errors, highlights and limited pose observability remain possible.");
    Ok(c)
}
fn html(path: &Path, maps: &[Value]) -> Result<()> {
    let data = serde_json::to_string(maps)?;
    let template = r#"<!doctype html><html><meta charset="utf-8"><title>Sclera splat maps</title><style>body{margin:30px;background:#0d141e;color:#e8eef5;font:17px system-ui}h1{font-size:26px}p{max-width:1000px;color:#bdcbd9}canvas{background:#121c29;width:min(94vw,1050px);height:auto;display:block;border:1px solid #415063;cursor:grab}label{margin-right:20px}input{width:260px}a{color:#84d2fa}</style><h1>Sclera Gaussian surface previews</h1><p>Drag to rotate the observed patches. The round shape is an assumed sphere, not measured anatomy. Only sclera-mask samples with accepted alignment enter the map. Dark gaps remain unknown.</p><label>Eye <select id="eye"><option value="0">1</option><option value="1">2</option></select></label><label><input id="baseline" type="checkbox">Single-exposure baseline</label><button id="reset">Reset view</button><p id="stats"></p><canvas id="view" width="840" height="560"></canvas><p><a href="review.mp4">Source-frame comparison video</a> · <a href="report.json">Measurements and assumptions</a></p><script>
const maps=DATA;const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');let yaw=0,pitch=0,drag=null;
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
function rotate(p){let x=Math.cos(yaw)*p[0]+Math.sin(yaw)*p[2],z=-Math.sin(yaw)*p[0]+Math.cos(yaw)*p[2];return [x,Math.cos(pitch)*p[1]-Math.sin(pitch)*z,Math.sin(pitch)*p[1]+Math.cos(pitch)*z]}
function draw(){let m=maps[+document.getElementById('eye').value],s=document.getElementById('baseline').checked?m.baseline:m.splats;ctx.fillStyle='#121c29';ctx.fillRect(0,0,840,560);let sum=new Float32Array(840*560*3),mass=new Float32Array(840*560);
for(let g of s){let p=rotate(g.mean);if(p[2]<=.15)continue;let u=rotate(g.tangent_u),v=rotate(g.tangent_v),k=(g.sigma*480)**2,a=k*(u[0]**2+v[0]**2)+.64,b=k*(u[0]*u[1]+v[0]*v[1]),c=k*(u[1]**2+v[1]**2)+.64,d=a*c-b*b,x=420+480*p[0],y=280+480*p[1];for(let j=Math.max(0,Math.floor(y-3*Math.sqrt(c)));j<=Math.min(559,Math.ceil(y+3*Math.sqrt(c)));j++)for(let i=Math.max(0,Math.floor(x-3*Math.sqrt(a)));i<=Math.min(839,Math.ceil(x+3*Math.sqrt(a)));i++){let dx=i-x,dy=j-y,q=(c*dx*dx-2*b*dx*dy+a*dy*dy)/d;if(q>9)continue;let w=Math.exp(-q/2),n=j*840+i;mass[n]+=w;for(let ch=0;ch<3;ch++)sum[3*n+ch]+=w*g.rgb[ch]}}
let image=ctx.createImageData(840,560);for(let i=0;i<mass.length;i++){for(let k=0;k<3;k++)image.data[4*i+k]=mass[i]>=.3?sum[3*i+k]/mass[i]:((Math.floor((i%840)/24)+Math.floor(Math.floor(i/840)/24))%2?31:23);image.data[4*i+3]=255}ctx.putImageData(image,0,0);document.getElementById('stats').textContent=`${m.accepted_training===1?"REFERENCE ONLY: multi-frame alignment failed. ":""}${s.length} Gaussian surface elements · ${m.accepted_training-1} additional frames aligned (reference excluded) · ${m.held_frames} held-out texture frames · drag changes the assumed viewing angle`;}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;yaw=Math.max(-.65,Math.min(.65,yaw+(e.clientX-drag[0])*.003));pitch=Math.max(-.45,Math.min(.45,pitch+(e.clientY-drag[1])*.003));drag=[e.clientX,e.clientY];draw()};canvas.onpointerup=()=>drag=null;document.getElementById('reset').onclick=()=>{yaw=pitch=0;draw()};document.getElementById('eye').onchange=draw;document.getElementById('baseline').onchange=draw;draw();</script></html>"#;
    fs::write(path, template.replace("DATA", &data))?;
    Ok(())
}
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("--sclera-splats VERIFIED_INPUT_DIR NEW_OUTPUT_DIR".into());
    }
    let input = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("new checked outputs directory required".into());
    }
    let summary: Value = serde_json::from_slice(&fs::read(input.join("summary.json"))?)?;
    let source = boot::current_source(Path::new("."))?;
    if summary["schema"] != "buttercup-sclera-splat-inputs-v1"
        || summary["complete"] != true
        || summary["device"] != "cpu"
        || summary["preparation_recipe_sha256"] != recipe::stamp()?
    {
        return Err(
            "completed CPU sclera inputs with unchanged preparation recipe required".into(),
        );
    }
    let bytes = fs::read(input.join("frames.jsonl"))?;
    if digest(&bytes) != summary["frames_sha256"] {
        return Err("input manifest hash mismatch".into());
    }
    let info: Value = serde_json::from_slice(&fs::read(input.join("inputs.json"))?)?;
    let bundle = BundleSource::open(Path::new(info["bundle"].as_str().ok_or("bundle")?))?;
    let mut groups = BTreeMap::<u64, Vec<Frame>>::new();
    for row in read_rows(&input.join("frames.jsonl"))? {
        let s = &row["source"];
        let packed = bundle.read_range(
            s["stream"].as_str().ok_or("stream")?,
            n(&s["offset"]),
            n(&s["length"]) as usize,
        )?;
        let rgb = fs::read(input.join(row["rgb"].as_str().ok_or("rgb")?))?;
        let mask = fs::read(input.join(row["mask"].as_str().ok_or("mask")?))?;
        if n(&s["width"]) != W as u64
            || n(&s["height"]) != H as u64
            || rgb.len() != W * H * 3
            || mask.len() != W * H
            || digest(&packed) != row["raw_sha256"]
            || digest(&rgb) != row["rgb_sha256"]
            || digest(&mask) != row["mask_sha256"]
            || row["prompt"] != "exposed white sclera"
            || row["input_adapter"] != "native-preview-100"
        {
            return Err("RAW/mask/adapter identity mismatch".into());
        }
        let raw = raw10::try_unpack_raw10(&packed, W, H, n(&s["stride"]) as usize)?;
        let (core, support_stats) = core(
            &mask,
            &raw,
            n(&s["sensor_x"]) as u32,
            n(&s["sensor_y"]) as u32,
        );
        groups.entry(n(&s["eye_id"])).or_default().push(Frame {
            row,
            raw: Arc::new(raw),
            rgb,
            core,
            support_stats,
        });
    }
    fs::create_dir(out)?;
    let graph = json!({"schema":boot::SCHEMA,"source":source,"targets":["surface_map"],"nodes":[
        {"id":"source","kind":"source","sha256":source.tree_sha256,"dependencies":[]},
        {"id":"raw","kind":"raw","sha256":digest(&fs::read(input.join("raw-inventory.json"))?),"dependencies":[]},
        {"id":"sam31","kind":"sam3","sha256":info["teacher"]["checkpoint_sha256"],"dependencies":[]},
        {"id":"export","kind":"export","sha256":info["teacher"]["model_sha256"],"dependencies":["sam31","source"]},
        {"id":"masks","kind":"derived_data","sha256":digest(&bytes),"dependencies":["raw","source","export"]},
        {"id":"surface_map","kind":"derived_data","planned":true,"sha256":null,"dependencies":["raw","source","masks"]}]});
    let parsed: boot::Manifest = serde_json::from_value(graph.clone())?;
    write(
        out.join("bootstrap-preflight.json"),
        &boot::validate(&parsed, &source).map_err(|e| format!("map preflight: {e:?}"))?,
    )?;
    write(out.join("bootstrap-graph.json"), &graph)?;
    let started = Instant::now();
    let mut reports = vec![];
    let mut maps = vec![];
    let mut video = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgra",
            "-s",
            "1296x790",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(out.join("review.mp4"))
        .stdin(Stdio::piped())
        .spawn()?;
    let mut stream = video.stdin.take().ok_or("ffmpeg stdin")?;
    for (eye, frames) in &groups {
        let split = frames.len() * 2 / 3;
        let mut map = BTreeMap::new();
        fuse(&mut map, &frames[0], Pose::default());
        let baseline = map.values().cloned().collect::<Vec<_>>();
        let mut evidence = vec![json!({"accepted":true,"reference":true,"matches":0})];
        let mut poses = vec![Some(Pose::default())];
        let mut accepted_training = 1;
        for i in 1..frames.len() {
            let same_clock = frames[i].row["source"]["source_clock"]["source_key"]["stream_epoch"]
                == frames[0].row["source"]["source_clock"]["source_key"]["stream_epoch"];
            if !same_clock
                || n(&frames[i].row["source"]["timestamp_ns"])
                    <= n(&frames[i - 1].row["source"]["timestamp_ns"])
            {
                return Err("changed or nonmonotonic source epoch".into());
            }
            let matches = pairs(&frames[0], &frames[i]);
            let (sphere_pose, mut ev) = pose_evidence(&matches);
            let (translation, texture) = texture_translation(&frames[0], &frames[i]);
            let pose = sphere_pose.or(translation);
            ev["sphere_accepted"] = json!(sphere_pose.is_some());
            ev["translation_evidence"] = texture;
            ev["accepted"] = json!(pose.is_some());
            ev["admitted_pose"] = json!(pose);
            ev["admitted_method"] = json!(if sphere_pose.is_some() {
                "sphere rotation + translation"
            } else if translation.is_some() {
                "texture translation only; out-of-plane rotation unresolved"
            } else {
                "none"
            });
            if i < split {
                if let Some(p) = pose {
                    fuse(&mut map, &frames[i], p);
                    accepted_training += 1;
                }
            }
            poses.push(pose);
            evidence.push(ev);
        }
        let splats = map.values().cloned().collect::<Vec<_>>();
        let mut held = vec![];
        for (i, frame) in frames.iter().enumerate() {
            let mut metrics = Value::Null;
            if i >= split {
                if let Some(pose) = poses[i] {
                    let (b, bm) = geom::render(&baseline, pose, RADIUS, W, H);
                    let (c, cm) = geom::render(&splats, pose, RADIUS, W, H);
                    metrics = photo_error(frame, &b, &c, &bm, &cm);
                }
                held.push(json!({"source":frame.row["source"],"alignment_accepted":poses[i].is_some(),"color":metrics}));
            }
            let mut canvas = panel(
                frame,
                &splats,
                &baseline,
                poses[i],
                i,
                *eye,
                split,
                &evidence[i],
            )?;
            canvas.png(&out.join(format!("eye-{eye}-{i:02}.png")))?;
            // Review repeats expose overlays at a readable pace; do not count
            // these display holds as additional source observations.
            for _ in 0..24 {
                stream.write_all(canvas.bytes())?;
            }
            write(
                out.join(format!("eye-{eye}-{i:02}.json")),
                &json!({"source":frame.row["source"],"raw_sha256":frame.row["raw_sha256"],"role":if i<split{"map"}else{"held-out color"},"pose_evidence":evidence[i],"photometry":metrics,"support":frame.support_stats}),
            )?;
        }
        maps.push(json!({"eye":eye,"splats":splats,"baseline":baseline,"accepted_training":accepted_training,"training_frames":split,"held_frames":frames.len()-split}));
        reports.push(json!({"eye":eye,"frames":frames.len(),"training_frames":split,"accepted_training":accepted_training,"aligned_additional_training":accepted_training-1,"map_status":if accepted_training==1{"single_reference_only"}else{"fused_surface_preview"},"baseline_splats":baseline.len(),"map_splats":splats.len(),"multi_observation_splats":splats.iter().filter(|s|s.observations>=2).count(),"held_out":held,"pose_evidence":evidence}));
        eprintln!(
            "SCLERA MAP eye={eye} splats={} accepted={accepted_training}/{split}",
            splats.len()
        );
    }
    drop(stream);
    if !video.wait()?.success() {
        return Err("review encoder failed".into());
    }
    write(out.join("maps.json"), &maps)?;
    html(&out.join("viewer.html"), &maps)?;
    if boot::current_source(Path::new("."))? != source {
        return Err("source changed during map build".into());
    }
    write(
        out.join("report.json"),
        &json!({"schema":"buttercup-sclera-gaussian-map-v1","complete":true,"input":fs::canonicalize(input)?,"source":source,"device":"cpu","seconds":started.elapsed().as_secs_f64(),"eyes":reports,
        "sphere_prior":{"projection":"orthographic","center_roi_px":CENTER,"radius_px":RADIUS,"radius_sensitivity_px":[180,240,320],"measured_geometry":false},
        "representation":"tangent Gaussian surface elements; normalized Gaussian kernel rasterization, not the unconstrained volumetric 3DGS optimizer",
        "validated_3d":false,"input_preparation_source":summary["source"],"input_preparation_recipe_sha256":summary["preparation_recipe_sha256"],
        "translation_admission":{"fit_and_spatial_check_minimum_correlation":0.85,"minimum_separated_peak_margin":0.015,"separated_peak_distance_px":3,"minimum_samples_each_fold":24,"minimum_overlap_fraction":0.6666666667,"minimum_native_texture_std":1,"maximum_search_offset_sensor_px":12,"geometry_role":"translation-only appearance alignment; cannot determine out-of-plane eye rotation"},
        "policy":{"mask_threshold":179,"mask_erosion_px":4,"additional_matching_patch_margin":6,"raw_clipping_threshold":1018,"clipping_veto":"6 of 8 native green sites in a physical 4x4 Quad Bayer cell; partial color clipping remains in appearance samples","native_match_score":0.70,"native_distinct_margin":0.04,"minimum_matches":8,"minimum_span_px":[45,20],"minimum_inlier_fraction":0.75,"inlier_residual_px":2.5,"maximum_median_residual_px":1.5,"grid_px":2,"splat_sigma_px":1.1,"maximum_color_update_difference":45,"training_fraction":"first two thirds","heldout_pose_uses_current_matches":true,"maximum_frames_per_eye":24},
        "limitations":["Unreviewed SAM masks may include wrong anatomy or omit sclera","Radius/center/depth/intrinsics are assumed, not recovered or independently scaled","Matching and rendering error are not vessel identity or anatomical accuracy","Each eye is a separate object, never a stereo reconstruction of the other eye","Review video uses slow display holds, not source-time playback","Single-user short-interval prototype, no live changes or cross-user evidence"],"sn_feida":null,"human_label_error":null,"measured_3d_error":null}),
    )?;
    fs::write(out.join("README.md"),"# Sclera Gaussian surface map pilot\n\nOpen viewer.html to rotate the two separately built eye maps and switch to their single-frame baselines. review.mp4 shows RAW, sclera support and projected maps. Gray checkerboard is unknown surface. The curved shape is a fixed orthographic sphere prior, not measured anatomy. Tangent Gaussian footprints rotate and foreshorten with that surface. This is normalized surface-kernel splatting, not a run of the official volumetric 3DGS optimizer.\n\nNative RAW is reopened and hashed. Fresh highest-score SAM `exposed white sclera` proposals supply support; no iris fit or gaze target is used. Native patch correspondences first attempt a sphere pose. When they fail, masked native green high-pass correlation may support translation only; out-of-plane rotation stays unresolved. Both align directly to the reference. Unsupported fits contribute no samples. The first two thirds of the interval build the map; remaining texture is never fused. Their poses still use current RAW matches. Baseline and candidate use identical poses and common observed pixels for color comparisons. Source identities and excluded fits remain in per-frame JSON.\n\nRun `buttercup_calibration_sign sclera-splat-inputs BUNDLE VERIFIED_CPU_EXPORT NEW_INPUTS SKIP COUNT`, then `buttercup-cluster-replay --sclera-splats NEW_INPUTS NEW_MAP`. Both use CPU only and retain native bootstrap graph preflight. Cached official SAM export verification is not a new cold-export proof. No custom learned ancestor is consumed.\n\nInspect report.json before interpreting quality: photometric fidelity and self-correspondence errors are not anatomical accuracy or durable vessel re-identification. No independent scale or human sclera correspondences are supplied.\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_native_texture_translation_recovers_shift_and_abstains_on_flat_image() {
        let make = |dx: i32, dy: i32, flat: bool| Frame {
            row: json!({"source":{"sensor_x":0,"sensor_y":0}}),
            raw: Arc::new(
                (0..W * H)
                    .map(|j| {
                        let x = (j % W) as i32 - dx;
                        let y = (j / W) as i32 - dy;
                        if flat || x < 0 || y < 0 || x >= W as i32 || y >= H as i32 {
                            450
                        } else {
                            200 + (((x as u64 / 4) * 73856093) ^ ((y as u64 / 4) * 19349663)) as u16
                                % 400
                        }
                    })
                    .collect(),
            ),
            rgb: vec![128; W * H * 3],
            core: vec![true; W * H],
            support_stats: Value::Null,
        };
        let (pose, _) = texture_translation(&make(0, 0, false), &make(4, -4, false));
        assert_eq!(pose.unwrap().translation, [4., -4.]);
        assert!(texture_translation(&make(0, 0, true), &make(4, -4, true))
            .0
            .is_none());
    }
    #[test]
    fn clipping_respects_independent_color_channels_and_sensor_phase() {
        for (sx, sy) in [(0, 0), (1, 3), (2, 2)] {
            let mask = vec![255; W * H];
            let raw = (0..W * H)
                .map(|j| {
                    if (j % W + sx as usize) % 4 < 2 && (j / W + sy as usize) % 4 < 2 {
                        1023
                    } else {
                        450
                    }
                })
                .collect::<Vec<_>>();
            let (supported, stats) = core(&mask, &raw, sx, sy);
            assert!(supported[140 * W + 210]);
            assert!(stats["clipped_channel_samples_rgb"][0].as_u64().unwrap() > 0);
            assert_eq!(stats["clipped_channel_samples_rgb"][1], 0);
            let (clipped, _) = core(&mask, &vec![1023; W * H], sx, sy);
            assert!(!clipped.iter().any(|&v| v));
            let (outside, _) = core(&vec![0; W * H], &raw, sx, sy);
            assert!(!outside.iter().any(|&v| v));
        }
    }
}
