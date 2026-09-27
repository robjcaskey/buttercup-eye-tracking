//! Offline, source-seeded texture tracking. Destination ellipses never gate tracks.
//! Reuses the native RAW/CFA preprocessing; no live solver or model changes.
#![allow(dead_code)]
#[path = "../glasses_parallax.rs"]
mod glasses_parallax;
#[path = "buttercup_iris_feature_motion/layers.rs"]
mod layers;
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
type P = [f64; 2];
fn n(v: &Value) -> f64 {
    v.as_f64().expect("numeric field")
}
fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}
fn dist(a: P, b: P) -> f64 {
    let d = sub(a, b);
    d[0].hypot(d[1])
}
fn med(v: Vec<f64>) -> f64 {
    glasses_parallax::median(v)
}
struct Frame {
    meta: Value,
    im: glasses_parallax::Image,
    rgb: Vec<u32>,
    w: usize,
    h: usize,
    origin: P,
    cell_origin: P,
}
impl Frame {
    fn load(v: &Value) -> Result<Self, E> {
        let input = &v["input"];
        let m = &input["frame"];
        let w = n(&m["width"]) as usize;
        let h = n(&m["height"]) as usize;
        let sx = n(&m["sensor_x"]) as usize;
        let sy = n(&m["sensor_y"]) as usize;
        let stride = n(&m["stride"]) as usize;
        let mut f = fs::File::open(input["raw_file"].as_str().unwrap())?;
        f.seek(SeekFrom::Start(input["raw_offset"].as_u64().unwrap()))?;
        let mut b = vec![0; input["raw_length"].as_u64().unwrap() as usize];
        f.read_exact(&mut b)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&b)),
            v["raw_sha256"].as_str().unwrap()
        );
        let im = glasses_parallax::decode(&b, w, h, stride, sx, sy)?;
        let raw = raw10::try_unpack_raw10(&b, w, h, stride)?;
        let rgb = raw_preview::color_preview(&raw, w, h, sx as u32, sy as u32, 100, None);
        Ok(Self {
            meta: v.clone(),
            im,
            rgb,
            w,
            h,
            origin: [sx as f64, sy as f64],
            cell_origin: [
                (sx + (4 - sx % 4) % 4) as f64 + 1.5,
                (sy + (4 - sy % 4) % 4) as f64 + 1.5,
            ],
        })
    }
    fn sensor(&self, p: P) -> P {
        [
            self.cell_origin[0] + 4. * p[0],
            self.cell_origin[1] + 4. * p[1],
        ]
    }
    fn cell(&self, p: P) -> P {
        [
            (p[0] - self.cell_origin[0]) / 4.,
            (p[1] - self.cell_origin[1]) / 4.,
        ]
    }
    fn radius(&self, p: P) -> f64 {
        let e = &self.meta["ellipse"];
        let a = n(&e["angle"]);
        let d = sub(
            p,
            [n(&e["center_sensor_px"][0]), n(&e["center_sensor_px"][1])],
        );
        ((a.cos() * d[0] + a.sin() * d[1]) / n(&e["a"]))
            .hypot((-a.sin() * d[0] + a.cos() * d[1]) / n(&e["b"]))
    }
}
fn sample(im: &glasses_parallax::Image, p: P, green: bool) -> Option<f64> {
    if p[0] < 0. || p[1] < 0. || p[0] + 1. >= im.w as f64 || p[1] + 1. >= im.h as f64 {
        return None;
    }
    let x = p[0].floor() as usize;
    let y = p[1].floor() as usize;
    let (u, v) = (p[0] - x as f64, p[1] - y as f64);
    let z = if green { &im.green } else { &im.pixels };
    Some(
        (1. - v) * ((1. - u) * z[y * im.w + x] + u * z[y * im.w + x + 1])
            + v * ((1. - u) * z[(y + 1) * im.w + x] + u * z[(y + 1) * im.w + x + 1]),
    )
}
fn patch(im: &glasses_parallax::Image, p: P, r: [i32; 2], green: bool) -> Option<Vec<f64>> {
    let mut v = Vec::new();
    for y in -r[1]..=r[1] {
        for x in -r[0]..=r[0] {
            v.push(sample(im, [p[0] + x as f64, p[1] + y as f64], green)?);
        }
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let norm = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>().sqrt();
    if !norm.is_finite() || norm / (v.len() as f64).sqrt() < 0.6 {
        return None;
    }
    for x in &mut v {
        *x = (*x - mean) / norm;
    }
    Some(v)
}
fn corr(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
#[derive(Clone, Copy)]
struct Match {
    p: P,
    ncc: f64,
    gap: f64,
    edge: bool,
}
fn locate(
    template: &[f64],
    im: &glasses_parallax::Image,
    center: P,
    r: [i32; 2],
    green: bool,
    search: i32,
) -> Option<Match> {
    let mut all = Vec::new();
    for y in -search..=search {
        for x in -search..=search {
            let p = [center[0] + x as f64, center[1] + y as f64];
            if let Some(v) = patch(im, p, r, green) {
                all.push((p, corr(template, &v)));
            }
        }
    }
    let &(mut p, mut c) = all.iter().max_by(|a, b| a.1.total_cmp(&b.1))?;
    let coarse = p;
    for step in [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625] {
        let base = p;
        for y in -1..=1 {
            for x in -1..=1 {
                let q = [base[0] + x as f64 * step, base[1] + y as f64 * step];
                if let Some(v) = patch(im, q, r, green) {
                    let s = corr(template, &v);
                    if s > c {
                        p = q;
                        c = s;
                    }
                }
            }
        }
    }
    let rival = all
        .iter()
        .filter(|a| dist(a.0, p) >= 2.)
        .map(|a| a.1)
        .fold(-1., f64::max);
    Some(Match {
        p,
        ncc: c,
        gap: c - rival,
        edge: (coarse[0] - center[0]).abs() >= search as f64
            || (coarse[1] - center[1]).abs() >= search as f64,
    })
}
#[derive(Clone)]
struct Seed {
    p: P,
    score: f64,
    kind: &'static str,
}
fn corner_score(f: &Frame, x: usize, y: usize) -> f64 {
    let (mut xx, mut yy, mut xy) = (0., 0., 0.);
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            let k = (y as isize + dy) as usize * f.im.w + (x as isize + dx) as usize;
            let gx = (f.im.pixels[k + 1] - f.im.pixels[k - 1]) / 2.;
            let gy = (f.im.pixels[k + f.im.w] - f.im.pixels[k - f.im.w]) / 2.;
            xx += gx * gx;
            yy += gy * gy;
            xy += gx * gy;
        }
    }
    (xx + yy - ((xx - yy).powi(2) + 4. * xy * xy).sqrt()) / 2.
}
fn seeds(f: &Frame, r: i32) -> (Vec<Seed>, f64) {
    let mut levels = Vec::new();
    for y in 0..f.im.h {
        for x in 0..f.im.w {
            if f.radius(f.sensor([x as f64, y as f64])) <= 1. {
                levels.push(f.im.green[y * f.im.w + x]);
            }
        }
    }
    levels.sort_by(f64::total_cmp);
    let middle = levels[levels.len() / 2];
    let high = levels[levels.len() * 4 / 5];
    // Bright-patch flag only: not calibrated specularity or material identity.
    let threshold = (middle + 25.).max(middle + 5. * (high - middle));
    let mut all = Vec::new();
    for y in 6..f.im.h - 6 {
        for x in 6..f.im.w - 6 {
            let p = [x as f64, y as f64];
            if f.radius(f.sensor(p)) > 1. {
                continue;
            }
            let score = corner_score(f, x, y);
            // Keep seed positions identical when comparing tracking patch sizes.
            if score < 2. || patch(&f.im, p, [3, 3], false).is_none() {
                continue;
            }
            let mut bright = false;
            let mut crosses = false;
            // Include smoothing and interpolation support, not just patch centers.
            // A reflection just beyond the sampled grid can otherwise dominate it.
            for dy in -(r + 2)..=(r + 2) {
                for dx in -(r + 2)..=(r + 2) {
                    let q = [p[0] + dx as f64, p[1] + dy as f64];
                    match sample(&f.im, q, true) {
                        Some(level) => bright |= level > threshold,
                        None => crosses = true,
                    }
                    crosses |= f.radius(f.sensor(q)) > 1.;
                }
            }
            let kind = if bright {
                "bright/reflection"
            } else if crosses {
                "rim/context"
            } else {
                "texture candidate"
            };
            all.push(Seed {
                p: f.sensor(p),
                score,
                kind,
            });
        }
    }
    all.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out: Vec<Seed> = Vec::new();
    for a in all {
        if out.iter().all(|q| dist(a.p, q.p) >= 12.) {
            out.push(a);
            if out.len() == 64 {
                break;
            }
        }
    }
    out.sort_by(|a, b| a.p[1].total_cmp(&b.p[1]).then(a.p[0].total_cmp(&b.p[0])));
    (out, threshold)
}
#[derive(Clone, Copy, PartialEq)]
enum TrackingPolicy {
    Iris,
    OuterBands,
}
fn track_pair(
    a: &Frame,
    b: &Frame,
    p: P,
    anchor: &Frame,
    anchor_p: P,
    r: [i32; 2],
    search: i32,
    policy: TrackingPolicy,
    search_hint: Option<P>,
) -> Value {
    let go = || -> Option<Value> {
        let t = patch(&a.im, a.cell(p), r, false)?;
        let m = locate(
            &t,
            &b.im,
            b.cell(search_hint.unwrap_or(p)),
            r,
            false,
            search,
        )?;
        let candidate = b.sensor(m.p);
        let u = patch(&b.im, m.p, r, false)?;
        let rev = locate(&u, &a.im, a.cell(p), r, false, search)?;
        let fb = dist(a.sensor(rev.p), p);
        let gt = patch(&a.im, a.cell(p), r, true)?;
        let gm = locate(&gt, &b.im, m.p, r, true, 2)?;
        let green_delta = dist(b.sensor(gm.p), candidate);
        let at = patch(&anchor.im, anchor.cell(anchor_p), r, false)?;
        let am = locate(&at, &b.im, m.p, r, false, 2)?;
        let anchor_delta = dist(b.sensor(am.p), candidate);
        let strict_reason = if m.edge {
            Some("search edge")
        } else if m.ncc < 0.90 || rev.ncc < 0.90 {
            Some("weak NCC")
        } else if m.gap < 0.025 {
            Some("ambiguous match")
        } else if fb > 1.25 {
            Some("forward/back mismatch")
        } else if gm.ncc < 0.85 || green_delta > 1.5 {
            Some("green-channel mismatch")
        } else if am.ncc < 0.90 || anchor_delta > 1.5 {
            Some("source-template drift")
        } else {
            None
        };
        // Narrow low-texture strips have substantially larger matching error
        // than the high-contrast iris patches. Preserve the strict result and
        // label the wider, bounded outer-band policy explicitly in the report.
        let outer = policy == TrackingPolicy::OuterBands;
        let reason = if !outer {
            strict_reason
        } else if m.edge {
            Some("search edge")
        } else if m.ncc < 0.85 || rev.ncc < 0.85 {
            Some("weak NCC")
        } else if m.gap < 0.025 {
            Some("ambiguous match")
        } else if fb > 3. {
            Some("forward/back mismatch")
        } else if gm.ncc < 0.80 || green_delta > 3. {
            Some("green-channel mismatch")
        } else if am.ncc < 0.85 || anchor_delta > 3. {
            Some("source-template drift")
        } else {
            None
        };
        Some(
            json!({"accepted":reason.is_none(),"reason":reason,"sensor":candidate,"ncc":m.ncc,"peak_gap":m.gap,
            "strict_iris_gate_passed":strict_reason.is_none(),
            "reverse_ncc":rev.ncc,"forward_back_px":fb,"green_ncc":gm.ncc,"green_agreement_px":green_delta,
            "anchor_ncc":am.ncc,"anchor_agreement_px":anchor_delta,
            "outside_current_ellipse":b.radius(candidate)>1.,"outside_source_ellipse":anchor.radius(candidate)>1.}),
        )
    };
    go().unwrap_or_else(|| json!({"accepted":false,"reason":"flat patch or outside recorded ROI"}))
}
fn track_seeds(
    frames: &[Frame],
    si: usize,
    seeds: &[Seed],
    r: [i32; 2],
    search: i32,
    policy: TrackingPolicy,
) -> Vec<Vec<Value>> {
    if policy == TrackingPolicy::OuterBands {
        let mut result = vec![
            vec![json!({"accepted":false,"reason":"not evaluated"}); frames.len()];
            seeds.len()
        ];
        for (i, s) in seeds.iter().enumerate() {
            result[i][si] = json!({"accepted":true,"sensor":s.p,"source_seed":true});
        }
        for direction in [-1isize, 1] {
            let mut prev = si;
            let mut common = [0.; 2];
            loop {
                let j = prev as isize + direction;
                if j < 0 || j >= frames.len() as isize {
                    break;
                }
                let j = j as usize;
                for (i, s) in seeds.iter().enumerate() {
                    let center = if result[i][prev]["accepted"] == true {
                        xy(&result[i][prev]["sensor"])
                    } else {
                        [s.p[0] + common[0], s.p[1] + common[1]]
                    };
                    result[i][j] = track_pair(
                        &frames[si],
                        &frames[j],
                        s.p,
                        &frames[si],
                        s.p,
                        r,
                        search,
                        policy,
                        Some(center),
                    );
                    result[i][j]["search_center_sensor"] = json!(center);
                }
                let good = seeds
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| result[*i][j]["accepted"] == true)
                    .map(|(i, s)| sub(xy(&result[i][j]["sensor"]), s.p))
                    .collect::<Vec<_>>();
                if good.len() >= 3 {
                    common = std::array::from_fn(|a| med(good.iter().map(|p| p[a]).collect()));
                }
                prev = j;
            }
        }
        return result;
    }
    seeds
        .iter()
        .map(|s| {
            let mut v = vec![json!({"accepted":false,"reason":"prior track lost"}); frames.len()];
            v[si] = json!({"accepted":true,"sensor":s.p,"source_seed":true});
            for direction in [-1isize, 1] {
                let mut prev = si;
                let mut p = s.p;
                loop {
                    let j = prev as isize + direction;
                    if j < 0 || j >= frames.len() as isize {
                        break;
                    }
                    let j = j as usize;
                    v[j] = track_pair(
                        &frames[prev],
                        &frames[j],
                        p,
                        &frames[si],
                        s.p,
                        r,
                        search,
                        policy,
                        None,
                    );
                    if v[j]["accepted"] != true {
                        break;
                    }
                    p = xy(&v[j]["sensor"]);
                    prev = j;
                }
            }
            v
        })
        .collect()
}

/// Invalidate every CFA cell whose actual smoothing support touches the middle
/// 75%. Matching, reverse checks and template checks then cannot use that region.
fn restrict_to_outer_bands(f: &mut Frame) {
    let start_y = f.cell_origin[1] - f.origin[1] - 1.5;
    for y in 0..f.im.h {
        let smooth = usize::from(y > 0 && y + 1 < f.im.h);
        let lo = start_y + 4. * (y - smooth) as f64;
        let hi = start_y + 4. * (y + smooth + 1) as f64;
        if !(hi <= f.h as f64 * 0.125 || lo >= f.h as f64 * 0.875) {
            for x in 0..f.im.w {
                f.im.pixels[y * f.im.w + x] = f64::NAN;
                f.im.green[y * f.im.w + x] = f64::NAN;
            }
        }
    }
}

fn outer_band_seeds(f: &Frame) -> Vec<Seed> {
    let mut all = Vec::new();
    for y in 2..f.im.h - 2 {
        for x in 3..f.im.w - 3 {
            let p = [x as f64, y as f64];
            let score = corner_score(f, x, y);
            if score.is_finite() && score >= 2. && patch(&f.im, p, [5, 1], false).is_some() {
                all.push(Seed {
                    p: f.sensor(p),
                    score,
                    kind: "outer band",
                });
            }
        }
    }
    all.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut counts = [0; 16];
    let mut selected: Vec<Seed> = vec![];
    // Equal spatial quotas keep a single textured corner from dominating.
    for s in all {
        let column = (((s.p[0] - f.origin[0]) / f.w as f64 * 8.) as usize).min(7);
        let row = usize::from(s.p[1] - f.origin[1] > f.h as f64 * 0.5);
        let bin = row * 8 + column;
        if counts[bin] < 3 && selected.iter().all(|q| dist(s.p, q.p) >= 16.) {
            counts[bin] += 1;
            selected.push(s);
        }
    }
    selected.sort_by(|a, b| a.p[1].total_cmp(&b.p[1]).then(a.p[0].total_cmp(&b.p[0])));
    selected
}

fn add_outer_bands(input: &str, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("output directory must be new".into());
    }
    if !out
        .parent()
        .ok_or("missing output parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let mut root: Value = serde_json::from_slice(&fs::read(input)?)?;
    let s = root["series"]
        .as_array_mut()
        .ok_or("missing tracked series")?
        .iter_mut()
        .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .ok_or("selected series missing")?;
    let mut frames = s["frames"]
        .as_array()
        .ok_or("missing frames")?
        .iter()
        .map(Frame::load)
        .collect::<Result<Vec<_>, _>>()?;
    let si = frames
        .iter()
        .position(|f| f.meta["sequence"] == s["source_sequence"])
        .ok_or("seed absent")?;
    for pair in frames.windows(2) {
        assert_eq!(
            pair[0].meta["input"]["clock_lineage"],
            pair[1].meta["input"]["clock_lineage"]
        );
        assert_eq!(
            pair[0].meta["sequence"].as_u64().unwrap() + 1,
            pair[1].meta["sequence"].as_u64().unwrap()
        );
    }
    for f in &mut frames {
        restrict_to_outer_bands(f);
    }
    let seeds = outer_band_seeds(&frames[si]);
    let tracks = track_seeds(&frames, si, &seeds, [5, 1], 6, TrackingPolicy::OuterBands);
    let rows = seeds.iter().enumerate().map(|(i,s)| json!({"id":i+1,"source_sensor":s.p,"corner_score":s.score,"kind":s.kind,"frames":tracks[i]})).collect::<Vec<_>>();
    let counts = (0..frames.len())
        .map(|j| tracks.iter().filter(|t| t[j]["accepted"] == true).count())
        .collect::<Vec<_>>();
    let persistent = rows
        .iter()
        .filter(|t| {
            t["frames"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v["accepted"] == true)
        })
        .map(|t| t["id"].clone())
        .collect::<Vec<_>>();
    s["outer_tracks"] = json!(rows);
    s["outer_band_tracking"] = json!({"input":input,"top_fraction":0.125,"bottom_fraction":0.125,"central_fraction_excluded":0.75,"patch_half_extents_cells":[5,1],"search_radius_cells":6,"source_seeds":seeds.len(),"accepted_per_frame":counts,"persistent_ids":persistent,"support":"All sampled CFA cells including smoothing and bilinear support lie wholly in the top or bottom 12.5% of each recorded image. Middle 75% is unavailable to matching. Wide 11x3-cell patches fit the narrow strips.","tracking_policy":"Lower-precision outer-band candidate; strict iris gate result also recorded per match. Not calibrated confidence.","thresholds":{"ncc":0.85,"reverse_ncc":0.85,"peak_gap":0.025,"forward_back_px":3.,"green_ncc":0.80,"green_agreement_px":3.,"anchor_ncc":0.85,"anchor_agreement_px":3.},"selection":"source corners, equal quotas in 8 columns per strip, >=16 native px spacing; fixed source IDs; no pose-based rejection"});
    println!(
        "{}",
        serde_json::to_string_pretty(&s["outer_band_tracking"])?
    );
    s["outer_band_tracking"]["search_transport"]=json!("Match each fresh exposure to its original source patch around the last validated position; missing tracks use the previous frame's robust group shift. Search radius is local, not a cap on total motion. Reverse/template checks still refer to the original source; predictions never count as observations.");
    fs::create_dir(out)?;
    fs::write(out.join("results.json"), serde_json::to_vec_pretty(&root)?)?;
    Ok(())
}
fn xy(v: &Value) -> P {
    [n(&v[0]), n(&v[1])]
}
fn rigid(points: &[(P, P)]) -> Option<(f64, P, P, Vec<f64>)> {
    if points.len() < 3 {
        return None;
    }
    let k = points.len() as f64;
    let a = [
        points.iter().map(|p| p.0[0]).sum::<f64>() / k,
        points.iter().map(|p| p.0[1]).sum::<f64>() / k,
    ];
    let b = [
        points.iter().map(|p| p.1[0]).sum::<f64>() / k,
        points.iter().map(|p| p.1[1]).sum::<f64>() / k,
    ];
    let mut dot = 0.;
    let mut cross = 0.;
    for &(p, q) in points {
        let p = sub(p, a);
        let q = sub(q, b);
        dot += p[0] * q[0] + p[1] * q[1];
        cross += p[0] * q[1] - p[1] * q[0];
    }
    let theta = cross.atan2(dot);
    let errors = points
        .iter()
        .map(|&(p, q)| {
            let p = sub(p, a);
            dist(
                [
                    b[0] + theta.cos() * p[0] - theta.sin() * p[1],
                    b[1] + theta.sin() * p[0] + theta.cos() * p[1],
                ],
                q,
            )
        })
        .collect();
    Some((theta, a, sub(b, a), errors))
}
fn summary(points: &[(P, P)]) -> Value {
    let Some((theta, a, t, errors)) = rigid(points) else {
        return json!({"count":points.len(),"fit":null});
    };
    let mut loo = Vec::new();
    if points.len() > 3 {
        for i in 0..points.len() {
            let p = points
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, p)| *p)
                .collect::<Vec<_>>();
            loo.push(rigid(&p).unwrap().0.to_degrees());
        }
    }
    let mut q = [false; 4];
    for &(p, _) in points {
        q[(p[0] >= a[0]) as usize + 2 * (p[1] >= a[1]) as usize] = true;
    }
    let max_radius = points.iter().map(|&(p, _)| dist(p, a)).fold(0., f64::max);
    json!({"count":points.len(),"rotation_deg_clockwise_image":theta.to_degrees(),"centroid_displacement_px":t,
        "centroid_sensor":a,"median_residual_px":med(errors.clone()),"max_residual_px":errors.iter().copied().fold(0.,f64::max),
        "median_displacement_px":med(points.iter().map(|&(p,q)|dist(p,q)).collect()),"span_radius_px":max_radius,
        "occupied_quadrants":q.iter().filter(|&&x|x).count(),"leave_one_out_angles_deg":loo,
        "interpretation":"descriptive 2D rigid fit, not 3D eye rotation; overlapping patches and no material ground truth"})
}
fn warp(a: &Value, p: P) -> P {
    std::array::from_fn(|i| n(&a[i][0]) * p[0] + n(&a[i][1]) * p[1] + n(&a[i][2]))
}
fn color(kind: &str) -> &'static str {
    match kind {
        "bright/reflection" => "#ffb15c",
        "rim/context" => "#e59bee",
        _ => "#57f5d2",
    }
}
fn background(
    svg: &mut String,
    f: &Frame,
    crop: [i32; 4],
    x: f64,
    y: f64,
    scale: f64,
    boost: bool,
) -> Result<(), E> {
    write!(
        svg,
        "<g transform='translate({x},{y}) scale({scale})' shape-rendering='crispEdges'>"
    )?;
    for yy in crop[1]..crop[3] {
        for xx in crop[0]..crop[2] {
            let rx = xx - f.origin[0] as i32;
            let ry = yy - f.origin[1] as i32;
            if rx < 0 || ry < 0 || rx >= f.w as i32 || ry >= f.h as i32 {
                continue;
            }
            let mut c = f.rgb[ry as usize * f.w + rx as usize];
            if boost {
                let map = |v: u32| ((v as f64 / 255.).sqrt() * 255.).round() as u32;
                c = (map((c >> 16) & 255) << 16) | (map((c >> 8) & 255) << 8) | map(c & 255);
            }
            write!(
                svg,
                "<rect x='{}' y='{}' width='1' height='1' fill='#{c:06x}'/>",
                xx - crop[0],
                yy - crop[1]
            )?;
        }
    }
    svg.push_str("</g>");
    Ok(())
}
fn ellipse(
    svg: &mut String,
    f: &Frame,
    crop: [i32; 4],
    x: f64,
    y: f64,
    scale: f64,
) -> Result<(), E> {
    let e = &f.meta["ellipse"];
    let c = xy(&e["center_sensor_px"]);
    let a = n(&e["angle"]);
    let pts = (0..=120)
        .map(|i| {
            let t = i as f64 * std::f64::consts::TAU / 120.;
            let dx = n(&e["a"]) * t.cos();
            let dy = n(&e["b"]) * t.sin();
            format!(
                "{:.2},{:.2}",
                x + (c[0] + a.cos() * dx - a.sin() * dy - crop[0] as f64) * scale,
                y + (c[1] + a.sin() * dx + a.cos() * dy - crop[1] as f64) * scale
            )
        })
        .collect::<Vec<_>>()
        .join(" ");
    write!(svg,"<polyline points='{pts}' fill='none' stroke='white' stroke-width='1.2' stroke-dasharray='5 4'/>")?;
    Ok(())
}
fn render(
    frames: &[Frame],
    seed_index: usize,
    seeds: &[Seed],
    tracks: &[Vec<Value>],
    out: &Path,
    name: &str,
) -> Result<(), E> {
    let source = &frames[seed_index];
    let e = &source.meta["ellipse"];
    let c = xy(&e["center_sensor_px"]);
    let half = n(&e["a"]).max(n(&e["b"])).ceil() as i32 + 25;
    let crop = [
        c[0].round() as i32 - half,
        c[1].round() as i32 - half,
        c[0].round() as i32 + half,
        c[1].round() as i32 + half,
    ];
    let scale = 440. / (2 * half) as f64;
    // Display a fixed, spatially separated subset. Analysis retains every seed.
    let mut candidates: Vec<usize> = (0..seeds.len())
        .filter(|&i| {
            tracks[i]
                .iter()
                .enumerate()
                .any(|(j, v)| j != seed_index && v["accepted"] == true)
        })
        .collect();
    let crosses = |i: usize| {
        tracks[i].iter().any(|v| {
            v["accepted"] == true
                && (v["outside_current_ellipse"] == true || v["outside_source_ellipse"] == true)
        })
    };
    candidates.sort_by(|&a, &b| {
        crosses(b)
            .cmp(&crosses(a))
            .then(seeds[b].score.total_cmp(&seeds[a].score))
    });
    let mut displayed: Vec<usize> = Vec::new();
    for i in candidates {
        if displayed
            .iter()
            .filter(|&&k| seeds[k].kind == seeds[i].kind)
            .count()
            < 8
            && displayed
                .iter()
                .all(|&k| dist(seeds[i].p, seeds[k].p) >= 20.)
        {
            displayed.push(i);
        }
    }
    let start="<svg xmlns='http://www.w3.org/2000/svg' width='1440' height='1210' viewBox='0 0 1440 1210'><rect width='100%' height='100%' fill='#171b24'/><g font-family='sans-serif' fill='white'>";
    let mut svg = start.to_string();
    write!(svg,"<text x='20' y='32' font-size='25'>Iris texture tracks · {name} · starting points selected once at source {}</text>",source.meta["sequence"])?;
    svg.push_str("<text x='20' y='63' font-size='17'>Destinations may cross either ellipse. Dashed white outlines are reference only; they never reject a match.</text>");
    for (j, f) in frames.iter().enumerate() {
        let x = 20. + (j % 3) as f64 * 475.;
        let y = 125. + (j / 3) as f64 * 505.;
        let accepted = tracks.iter().filter(|t| t[j]["accepted"] == true).count();
        write!(
            svg,
            "<text x='{x}' y='{}' font-size='20'>Source {}{} · {accepted}/{} retained</text>",
            y - 16.,
            f.meta["sequence"],
            if j == seed_index { " (seed)" } else { "" },
            seeds.len()
        )?;
        background(&mut svg, f, crop, x, y, scale, false)?;
        ellipse(&mut svg, f, crop, x, y, scale)?;
        for (id, s) in seeds.iter().enumerate() {
            let v = &tracks[id][j];
            if v["accepted"] != true || !displayed.contains(&id) {
                continue;
            }
            let p = xy(&v["sensor"]);
            let px = x + (p[0] - crop[0] as f64) * scale;
            let py = y + (p[1] - crop[1] as f64) * scale;
            let col = color(s.kind);
            let a = (id as f64 * 2.39996).sin();
            let lx = px + if a > 0. { 10. } else { -16. };
            let ly = py + if id % 2 == 0 { -12. } else { 18. };
            let star =
                if v["outside_current_ellipse"] == true || v["outside_source_ellipse"] == true {
                    "*"
                } else {
                    ""
                };
            write!(svg,"<circle cx='{px:.2}' cy='{py:.2}' r='4.5' fill='none' stroke='{col}' stroke-width='1.5'/><path d='M{px:.2},{py:.2} L{lx:.2},{ly:.2}' stroke='{col}' stroke-width='.8'/><text x='{lx:.2}' y='{ly:.2}' font-size='14' fill='{col}' stroke='#10141a' stroke-width='3' paint-order='stroke'>{}{star}</text>",id+1)?;
        }
        // Individual frames for a stable-layout animation; retain source pixels.
        let mut one=format!("<svg xmlns='http://www.w3.org/2000/svg' width='520' height='570'><rect width='100%' height='100%' fill='#171b24'/><g fill='white' font-family='sans-serif'><text x='20' y='29' font-size='21'>Source {} · {} / {} tracks</text>",f.meta["sequence"],accepted,seeds.len());
        background(&mut one, f, crop, 20., 55., 480. / (2 * half) as f64, false)?;
        ellipse(&mut one, f, crop, 20., 55., 480. / (2 * half) as f64)?;
        for (id, s) in seeds.iter().enumerate() {
            if tracks[id][j]["accepted"] != true || !displayed.contains(&id) {
                continue;
            }
            let p = xy(&tracks[id][j]["sensor"]);
            let px = 20. + (p[0] - crop[0] as f64) * 480. / (2 * half) as f64;
            let py = 55. + (p[1] - crop[1] as f64) * 480. / (2 * half) as f64;
            let col = color(s.kind);
            write!(one,"<circle cx='{px:.2}' cy='{py:.2}' r='4' stroke='{col}' fill='none' stroke-width='1.5'/><text x='{:.2}' y='{:.2}' font-size='14' fill='{col}' stroke='#111' stroke-width='3' paint-order='stroke'>{}</text>",px+7.,py-7.,id+1)?;
        }
        one.push_str("<text x='20' y='558' font-size='13'>Actual motion · no ellipse clipping or per-iris alignment</text></g></svg>");
        fs::write(out.join(format!("{name}-frame-{j}.svg")), one)?;
    }
    svg.push_str("<g font-size='18'><text x='975' y='665' fill='#57f5d2'>Cyan: interior texture candidates</text><text x='975' y='701' fill='#ffb15c'>Orange: bright light in patch support</text><text x='975' y='737' fill='#e59bee'>Purple: support crosses source rim</text><text x='975' y='790'>Same number = same source feature.</text><text x='975' y='826'>Missing number = failed validation.</text><text x='975' y='865'>* Crossed source or current ellipse.</text><text x='975' y='910'>Selected labels for clarity.</text><text x='975' y='946'>All matches are saved in JSON.</text><text x='975' y='995'>Color is a flag, not material identity.</text></g>");
    svg.push_str("<text x='20' y='1155' font-size='18'>Native CFA-cell preprocessing, subpixel normalized patch matching, reverse check, green-channel check and source-template check.</text><text x='20' y='1186' font-size='18'>No conic-based predictions. Pixel motion is measured; 2D rotation estimates are descriptive and do not resolve the 3D sign.</text></g></svg>");
    fs::write(out.join(format!("{name}-contact.svg")), svg)?;
    if name == "later-recording-eye2" {
        let alignment: Value = serde_json::from_slice(&fs::read(
            "outputs/globally-stabilized-motion-20260916-v2/stabilization.json",
        )?)?;
        let mut motion="<svg xmlns='http://www.w3.org/2000/svg' width='1440' height='630'><rect width='100%' height='100%' fill='#171b24'/><g fill='white' font-family='sans-serif'><text x='20' y='30' font-size='24'>Source 1034 to 1035 · measured feature displacements, magnified 8 times</text>".to_string();
        for (col, title) in [
            "Source image",
            "Measured motion · arrows ×8",
            "After outer alignment · arrows ×8",
        ]
        .iter()
        .enumerate()
        {
            let x = 20. + col as f64 * 475.;
            let y = 96.;
            write!(motion, "<text x='{x}' y='75' font-size='21'>{title}</text>")?;
            background(&mut motion, source, crop, x, y, scale, false)?;
            ellipse(&mut motion, source, crop, x, y, scale)?;
            if col == 0 {
                continue;
            }
            for &id in &displayed {
                let v = &tracks[id][seed_index + 1];
                if v["accepted"] != true {
                    continue;
                }
                let p = seeds[id].p;
                let q = xy(&v["sensor"]);
                let q = if col == 2 {
                    warp(&alignment["affine_second_to_first"], q)
                } else {
                    q
                };
                let a = [
                    x + (p[0] - crop[0] as f64) * scale,
                    y + (p[1] - crop[1] as f64) * scale,
                ];
                let b = [
                    a[0] + 8. * (q[0] - p[0]) * scale,
                    a[1] + 8. * (q[1] - p[1]) * scale,
                ];
                let color = color(seeds[id].kind);
                write!(motion,"<circle cx='{:.2}' cy='{:.2}' r='3' fill='{color}'/><path d='M{:.2},{:.2} L{:.2},{:.2}' stroke='{color}' stroke-width='2'/><circle cx='{:.2}' cy='{:.2}' r='2.8' fill='none' stroke='{color}'/><text x='{:.2}' y='{:.2}' font-size='13' fill='{color}' stroke='#111' stroke-width='3' paint-order='stroke'>{}</text>",a[0],a[1],a[0],a[1],b[0],b[1],b[0],b[1],a[0]+7.,a[1]-7.,id+1)?;
            }
        }
        motion.push_str("<text x='20' y='576' font-size='18'>Filled dot = start; hollow dot = magnified endpoint. Orange has bright light within matching support; cyan is darker interior.</text><text x='20' y='608' font-size='18'>Outer alignment comes from independent surrounding tracks. These short, noisy image motions do not establish 3D iris rotation.</text></g></svg>");
        fs::write(out.join(format!("{name}-motion.svg")), motion)?;
    }
    // Enlarge exact source/next patches separately from the fitted curves.
    let j = seed_index + 1;
    let mut shown: Vec<_> = seeds
        .iter()
        .enumerate()
        .filter(|(id, s)| s.kind != "bright/reflection" && tracks[*id][j]["accepted"] == true)
        .take(8)
        .collect();
    for (id, s) in seeds
        .iter()
        .enumerate()
        .filter(|(id, s)| s.kind == "bright/reflection" && tracks[*id][j]["accepted"] == true)
        .take(8 - shown.len())
    {
        shown.push((id, s));
    }
    let mut svg="<svg xmlns='http://www.w3.org/2000/svg' width='1260' height='910'><rect width='100%' height='100%' fill='#171b24'/><g fill='white' font-family='sans-serif'><text x='20' y='30' font-size='24'>Matched interior patches · left source / right next exposure</text><text x='20' y='61' font-size='17'>Same fixed display brightening on both images. Crops centered on measured matches; no iris-outline clipping.</text>".to_string();
    for (k, (id, s)) in shown.iter().enumerate() {
        let x = 20. + (k % 4) as f64 * 310.;
        let y = 125. + (k / 4) as f64 * 370.;
        let v = &tracks[*id][j];
        write!(
            svg,
            "<text x='{x}' y='{}' font-size='16'>#{} · NCC {:.3} · FB {:.2}px</text>",
            y - 14.,
            id + 1,
            n(&v["ncc"]),
            n(&v["forward_back_px"])
        )?;
        write!(
            svg,
            "<text x='{x}' y='{}' font-size='16' fill='{}'>{}</text>",
            y + 208.,
            color(s.kind),
            s.kind
        )?;
        for (col, fi, p) in [(0, seed_index, s.p), (1, j, xy(&v["sensor"]))] {
            let crop = [
                p[0].round() as i32 - 16,
                p[1].round() as i32 - 16,
                p[0].round() as i32 + 16,
                p[1].round() as i32 + 16,
            ];
            background(
                &mut svg,
                &frames[fi],
                crop,
                x + col as f64 * 148.,
                y,
                4.5,
                true,
            )?;
        }
        let p = xy(&v["sensor"]);
        let d = sub(p, s.p);
        write!(
            svg,
            "<text x='{x}' y='{}' font-size='17'>dx {:+.2}px · dy {:+.2}px</text>",
            y + 178.,
            d[0],
            d[1]
        )?;
    }
    svg.push_str("<text x='20' y='872' font-size='17'>These patches overlap: track count is not a count of independent anatomical landmarks.</text></g></svg>");
    fs::write(out.join(format!("{name}-patches.svg")), svg)?;
    Ok(())
}
fn main() -> Result<(), E> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() == 5 && args[1] == "--feature-layers" {
        return layers::run(&args[2], &args[3], Path::new(&args[4]));
    }
    if args.len() == 4 && args[1] == "--outer-transport" {
        return layers::outer_transport(&args[2], Path::new(&args[3]));
    }
    if args.len() == 8
        && (args[1] == "--extend-recording" || args[1] == "--extend-recording-reframed")
    {
        return extend_recording(
            &args[2],
            &args[3],
            Path::new(&args[4]),
            args[5].parse()?,
            args[6].parse()?,
            Path::new(&args[7]),
            args[1] == "--extend-recording-reframed",
        );
    }
    if args.len() == 4 && args[1] == "--outer-bands" {
        return add_outer_bands(&args[2], Path::new(&args[3]));
    }
    if args.len() < 3 {
        return Err("usage: buttercup_iris_feature_motion INPUT_RESULTS NEW_OUTPUT [patch-radius-cells] [search-radius-cells]".into());
    }
    let r: i32 = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(3);
    let search: i32 = args.get(4).map(|v| v.parse()).transpose()?.unwrap_or(6);
    if !(2..=5).contains(&r) || !(3..=16).contains(&search) {
        return Err("bounded patch/search settings required".into());
    }
    let out = Path::new(&args[2]);
    if out.exists() {
        return Err("output directory must be new".into());
    }
    let parent = out
        .parent()
        .ok_or("missing output parent")?
        .canonicalize()?;
    if !parent.starts_with("/mnt/bulk_data/buttercup-eye-tracking") {
        return Err("outputs must live under bulk runtime root".into());
    }
    fs::create_dir(out)?;
    let root: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut all = Vec::new();
    for (capture, eye, seq) in [("later-recording", 2, 1034), ("complete-nine", 1, 60)] {
        let mut selected = root["frames"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| {
                f["selection"]["capture"] == capture
                    && f["eye"] == eye
                    && (n(&f["sequence"]) - seq as f64).abs() <= 2.
            })
            .collect::<Vec<_>>();
        selected.sort_by_key(|f| f["sequence"].as_u64().unwrap());
        let frames = selected
            .into_iter()
            .map(Frame::load)
            .collect::<Result<Vec<_>, _>>()?;
        let si = frames
            .iter()
            .position(|f| f.meta["sequence"] == seq)
            .ok_or("source missing")?;
        assert_eq!(frames.len(), 5);
        for p in frames.windows(2) {
            assert_eq!(
                p[0].meta["input"]["clock_lineage"],
                p[1].meta["input"]["clock_lineage"]
            );
            assert_eq!(
                p[1].meta["sequence"].as_u64().unwrap(),
                p[0].meta["sequence"].as_u64().unwrap() + 1
            );
        }
        let (seeds, threshold) = seeds(&frames[si], r);
        let tracks = track_seeds(&frames, si, &seeds, [r, r], search, TrackingPolicy::Iris);
        let mut pairs = Vec::new();
        for j in 0..frames.len() {
            if j == si {
                continue;
            }
            let groups = [
                "texture candidate",
                "bright/reflection",
                "rim/context",
                "all nonbright source features",
            ]
            .map(|kind| {
                let points = seeds
                    .iter()
                    .enumerate()
                    .filter(|(i, s)| {
                        (s.kind == kind
                            || (kind == "all nonbright source features"
                                && s.kind != "bright/reflection"))
                            && tracks[*i][j]["accepted"] == true
                    })
                    .map(|(i, s)| (s.p, xy(&tracks[i][j]["sensor"])))
                    .collect::<Vec<_>>();
                let mut report = json!({"kind":kind,"image_motion":summary(&points)});
                if capture == "later-recording" && frames[j].meta["sequence"] == 1035 {
                    let s: Value = serde_json::from_slice(
                        &fs::read(
                            "outputs/globally-stabilized-motion-20260916-v2/stabilization.json",
                        )
                        .expect("outer stabilization required"),
                    )
                    .unwrap();
                    assert_eq!(s["raw_sha256"][0], frames[si].meta["raw_sha256"]);
                    assert_eq!(s["raw_sha256"][1], frames[j].meta["raw_sha256"]);
                    let aligned = points
                        .iter()
                        .map(|&(p, q)| (p, warp(&s["affine_second_to_first"], q)))
                        .collect::<Vec<_>>();
                    report["relative_to_outer_affine"] = summary(&aligned);
                    report["outer_alignment_source"] =
                        json!("globally-stabilized-motion-20260916-v2/stabilization.json");
                }
                report
            });
            pairs.push(json!({"sequence":frames[j].meta["sequence"],"dt_ms":(frames[j].meta["input"]["frame"]["timestamp_ns"].as_u64().unwrap() as i128-frames[si].meta["input"]["frame"]["timestamp_ns"].as_u64().unwrap() as i128) as f64/1e6,"groups":groups}));
        }
        let name = format!("{capture}-eye{eye}");
        render(&frames, si, &seeds, &tracks, out, &name)?;
        let rows=seeds.iter().enumerate().map(|(i,s)|json!({"id":i+1,"source_sensor":s.p,"corner_score":s.score,"kind":s.kind,"frames":tracks[i]})).collect::<Vec<_>>();
        let report = json!({"capture":capture,"eye":eye,"source_sequence":seq,"source_seed_count":seeds.len(),"bright_patch_threshold_raw_green":threshold,
            "frames":frames.iter().map(|f|json!({"sequence":f.meta["sequence"],"raw_sha256":f.meta["raw_sha256"],"input":f.meta["input"],"ellipse":f.meta["ellipse"]})).collect::<Vec<_>>(),"tracks":rows,"pairs":pairs});
        println!(
            "{name}: {} source features; next retained {}",
            seeds.len(),
            tracks
                .iter()
                .filter(|t| t[si + 1]["accepted"] == true)
                .count()
        );
        all.push(report);
    }
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"patch_radius_cells":r,"search_radius_cells":search,"native_px_per_cell":4,"analysis":"existing glasses_parallax CFA-cell luminance and green planes; NCC bilinear subpixel refinement (1/16 native px grid, not an accuracy claim)","source_gate":"center within source ellipse only; destination and patch footprint unconstrained by any ellipse","thresholds":{"ncc":0.90,"distinct_peak_gap":0.025,"forward_back_native_px":1.25,"green_ncc":0.85,"green_agreement_native_px":1.5,"anchor_ncc":0.90,"anchor_agreement_native_px":1.5},"scope":"10 existing RAW exposures from Rob, historical conics only seed regions; no training, anatomical labels or true 3D motion; fresh matches only","series":all}),
        )?,
    )?;
    Ok(())
}

/// Replay the existing source identities through a bounded native RAW interval.
/// The source ellipse only supplies the original selection/crop, never a new fit.
fn extend_recording(
    input: &str,
    cohorts: &str,
    index: &Path,
    first: u64,
    last: u64,
    out: &Path,
    allow_reframe: bool,
) -> Result<(), E> {
    if last < first || last - first > 300 {
        return Err("select at most 301 source exposures".into());
    }
    if out.exists() {
        return Err("output must be new".into());
    }
    if !out
        .parent()
        .ok_or("missing output parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let mut root: Value = serde_json::from_slice(&fs::read(input)?)?;
    let cohort: Value = serde_json::from_slice(&fs::read(cohorts)?)?;
    let ids = cohort["groups"][0]["ids"]
        .as_array()
        .ok_or("missing source cohort")?;
    let s = root["series"]
        .as_array_mut()
        .ok_or("missing series")?
        .iter_mut()
        .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .ok_or("missing selected series")?;
    let source_seq = s["source_sequence"]
        .as_u64()
        .ok_or("missing source sequence")?;
    let source = s["frames"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["sequence"] == source_seq)
        .ok_or("missing source frame")?
        .clone();
    let source_rows = s["tracks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| ids.contains(&t["id"]))
        .cloned()
        .collect::<Vec<_>>();
    let mut metadata = vec![];
    for (line_index, line) in fs::read_to_string(index)?.lines().enumerate() {
        let m: Value = serde_json::from_str(line)?;
        let seq = m["sequence"].as_u64().ok_or("missing archive sequence")?;
        if m["eye_id"] != 2 || seq < first || seq > last {
            continue;
        }
        assert_eq!(
            m["source_clock"]["source_key"]["stream_epoch"],
            source["input"]["clock_lineage"]
        );
        assert_eq!(
            m["region"]["session"],
            source["input"]["frame"]["region"]["session"]
        );
        assert_eq!(m["pixel_format"], "RAW10_LE40_1X1");
        // This bounded comparison intentionally keeps one sensor crop; metadata
        // generation changes alone are not camera motion or new stream epochs.
        for k in ["sensor_x", "sensor_y", "width", "height", "stride"] {
            if allow_reframe && (k == "sensor_x" || k == "sensor_y") {
                continue;
            }
            assert_eq!(
                m[k], source["input"]["frame"][k],
                "crop changed at {seq}: {k}"
            );
        }
        let path = index
            .parent()
            .ok_or("missing archive parent")?
            .join(m["stream"].as_str().ok_or("missing RAW stream")?)
            .canonicalize()?;
        assert_eq!(
            path,
            Path::new(source["input"]["raw_file"].as_str().unwrap()).canonicalize()?
        );
        let mut file = fs::File::open(&path)?;
        file.seek(SeekFrom::Start(m["offset"].as_u64().unwrap()))?;
        let mut bytes = vec![0; m["length"].as_u64().unwrap() as usize];
        file.read_exact(&mut bytes)?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        if seq == source_seq {
            assert_eq!(hash, source["raw_sha256"].as_str().unwrap());
        }
        metadata.push(json!({"sequence":seq,"ellipse":source["ellipse"],"ellipse_is_source_reference":true,"raw_sha256":hash,"input":{"frame":m,"index":line_index,"raw_file":path,"raw_offset":m["offset"],"raw_length":m["length"],"clock_lineage":source["input"]["clock_lineage"],"clock_attested":true}}));
    }
    metadata.sort_by_key(|m| m["sequence"].as_u64().unwrap());
    if metadata.len() != (last - first + 1) as usize {
        return Err("archive has missing/duplicate selected exposures".into());
    }
    for p in metadata.windows(2) {
        assert_eq!(
            p[0]["sequence"].as_u64().unwrap() + 1,
            p[1]["sequence"].as_u64().unwrap()
        );
        let a = p[0]["input"]["frame"]["timestamp_ns"].as_u64().unwrap();
        let b = p[1]["input"]["frame"]["timestamp_ns"].as_u64().unwrap();
        assert!(b > a && b - a < 250_000_000, "source time discontinuity");
    }
    let si = metadata
        .iter()
        .position(|f| f["sequence"] == source_seq)
        .ok_or("interval must contain original source")?;
    let frames = metadata
        .iter()
        .map(Frame::load)
        .collect::<Result<Vec<_>, _>>()?;
    let seeds = source_rows
        .iter()
        .map(|t| Seed {
            p: xy(&t["source_sensor"]),
            score: n(&t["corner_score"]),
            kind: "existing cohort",
        })
        .collect::<Vec<_>>();
    let tracks = track_seeds(&frames, si, &seeds, [3, 3], 6, TrackingPolicy::Iris);
    let rows = source_rows
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut row = t.clone();
            row["frames"] = json!(tracks[i]);
            row
        })
        .collect::<Vec<_>>();
    let counts = (0..frames.len())
        .map(|j| tracks.iter().filter(|t| t[j]["accepted"] == true).count())
        .collect::<Vec<_>>();
    let duration = (metadata.last().unwrap()["input"]["frame"]["timestamp_ns"]
        .as_u64()
        .unwrap()
        - metadata[0]["input"]["frame"]["timestamp_ns"]
            .as_u64()
            .unwrap()) as f64
        / 1e9;
    s["tracks"] = json!(rows);
    s["frames"] = json!(metadata);
    s["pairs"] = json!([]);
    s["source_seed_count"] = json!(seeds.len());
    s["extension"] = json!({"original_tracks":input,"original_cohort":cohorts,"frame_index":index,"first_sequence":first,"last_sequence":last,"duration_s":duration,"iris_accepted_per_frame":counts,"cohort_ids":ids,"method":"Existing source IDs, shared adjacent-frame tracker with original source-template checks; no reseeding or held tracks. Source ellipse is crop/selection reference only, not current conic evidence."});
    s["extension"]["roi_reframes_allowed"] = json!(allow_reframe);
    let group_report = json!({"frames":metadata,"source_seed":source_seq,"groups":[{"ids":ids,"size":ids.len()}],"scope":"Original five-frame cohort identities retained; longer interval is not reclustered and does not require survival to be displayed.","original_cohort":cohorts});
    println!(
        "{} exposures, {:.3}s, iris matches {:?}",
        frames.len(),
        duration,
        counts
    );
    fs::create_dir(out)?;
    fs::write(out.join("results.json"), serde_json::to_vec_pretty(&root)?)?;
    fs::write(
        out.join("cohorts.json"),
        serde_json::to_vec_pretty(&group_report)?,
    )?;
    Ok(())
}
