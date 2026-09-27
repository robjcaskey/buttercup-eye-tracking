//! Non-learned RAW conics for a conditional sign-supervision feasibility check.
//! No recorded predictions, masks, custom weights or calibration fit are read.
use super::{data, Result};
#[path = "canvas.rs"]
mod canvas;
#[path = "raw_conic.rs"]
mod raw_conic;
use buttercup_eye_tracking::{
    geometry::{projected_circle_candidates, Ellipse},
    raw10, raw_iris_focus as raw_fit,
    recorded_bundle::BundleSource,
};
use canvas::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeFit {
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub sam_score: Option<f32>,
    pub ellipse: [f64; 5],
    pub points: Vec<[f64; 2]>,
    pub normals: [[f64; 3]; 2],
    pub centers_per_radius: [[f64; 3]; 2],
    pub mean_residual_px: f64,
    pub origin: [u32; 2],
    #[serde(default)]
    pub outward_support_left_bottom_right: [f64; 3],
}
impl NativeFit {
    pub fn role(&self) -> &'static str {
        if self.provider == "sam31-single" {
            "fresh SAM mask conic"
        } else {
            "fresh classical RAW conic"
        }
    }
    pub fn admissible(&self) -> bool {
        self.points.len() >= 12
            && self.mean_residual_px <= 3.0
            && self.ellipse[2] >= 24.0
            && self.ellipse[3] / self.ellipse[2] >= 0.55
            && self
                .outward_support_left_bottom_right
                .iter()
                .all(|v| *v >= 0.5)
    }
    pub fn ellipse(&self) -> Ellipse {
        Ellipse {
            center: (self.ellipse[0], self.ellipse[1]),
            major_radius: self.ellipse[2],
            minor_radius: self.ellipse[3],
            angle: self.ellipse[4],
        }
    }
}
pub fn unpack(raw: &[u8], f: &Value) -> Result<Vec<u16>> {
    Ok(raw10::try_unpack_raw10(
        raw,
        data::num(&f["width"]).ok_or("width")? as usize,
        data::num(&f["height"]).ok_or("height")? as usize,
        data::num(&f["stride"]).ok_or("stride")? as usize,
    )?)
}
fn finish(e: Ellipse, points: Vec<[f64; 2]>, f: &Value) -> Option<NativeFit> {
    if points.len() < 8
        || e.minor_radius < 12.
        || e.major_radius > data::num(&f["height"])? as f64 * 0.6
    {
        return None;
    }
    let origin = [
        data::num(&f["sensor_x"])? as u32,
        data::num(&f["sensor_y"])? as u32,
    ];
    let p = projected_circle_candidates(e, origin, [4000.; 2], [4000., 3000.])?;
    let (s, c) = e.angle.sin_cos();
    let residual = points
        .iter()
        .map(|p| {
            let dx = p[0] - e.center.0;
            let dy = p[1] - e.center.1;
            let rho = ((c * dx + s * dy).powi(2) / e.major_radius.powi(2)
                + (-s * dx + c * dy).powi(2) / e.minor_radius.powi(2))
            .sqrt();
            (rho - 1.).abs() * e.minor_radius
        })
        .sum::<f64>()
        / points.len() as f64;
    Some(NativeFit {
        provider: "classical-raw".into(),
        sam_score: None,
        ellipse: [
            e.center.0,
            e.center.1,
            e.major_radius,
            e.minor_radius,
            e.angle,
        ],
        points,
        normals: p.map(|v| v.0),
        centers_per_radius: p.map(|v| v.1),
        mean_residual_px: residual,
        origin,
        outward_support_left_bottom_right: [0.; 3],
    })
}

/// RAW bytes alone do not identify a sensor ray. Keep crop/decoding metadata
/// in geometry cache keys even when feature images have identical content.
pub fn identity(hash: &str, frame: &Value) -> String {
    format!(
        "{hash}:{}:{}:{}:{}:{}",
        frame["sensor_x"], frame["sensor_y"], frame["width"], frame["height"], frame["stride"]
    )
}

/// Deterministic non-learned label proposal: bounded RAW-gradient RANSAC,
/// then check fresh RAW contrast along the completed sides and lower rim.
/// Missing/occluded support excludes supervision; it is not negative iris truth.
pub fn for_supervision(raw: &[u16], f: &Value) -> Option<NativeFit> {
    let w = data::num(&f["width"])? as usize;
    let h = data::num(&f["height"])? as usize;
    let (e, points) = raw_conic::fit(raw, w, h)?;
    from_segmentation(raw, f, e, points)
}

/// Shared native conic construction and RAW contrast audit for either a
/// classical proposal or the independently regenerated SAM mask contour.
pub fn from_segmentation(
    raw: &[u16],
    f: &Value,
    e: Ellipse,
    points: Vec<[f64; 2]>,
) -> Option<NativeFit> {
    let w = data::num(&f["width"])? as usize;
    let h = data::num(&f["height"])? as usize;
    if raw.len() != w.checked_mul(h)? || raw.is_empty() {
        return None;
    }
    let mut fit = finish(e, points, f)?;
    let mut values = raw.iter().step_by(16).copied().collect::<Vec<_>>();
    values.sort_unstable();
    let contrast = (values[values.len() * 95 / 100] - values[values.len() * 5 / 100]) as f64;
    let sample = |x: f64, y: f64| -> Option<f64> {
        let x = x.round() as isize;
        let y = y.round() as isize;
        if x < 4 || y < 4 || x + 4 >= w as isize || y + 4 >= h as isize {
            return None;
        }
        let mut sum = 0.;
        for dy in -4..4 {
            for dx in -4..4 {
                sum += raw[((y + dy) as usize) * w + (x + dx) as usize] as f64;
            }
        }
        Some(sum / 64.)
    };
    let mut counts = [[0usize; 2]; 3];
    let (s, c) = e.angle.sin_cos();
    for k in 0..96 {
        let t = k as f64 * std::f64::consts::TAU / 96.;
        let (st, ct) = t.sin_cos();
        let x = e.center.0 + c * e.major_radius * ct - s * e.minor_radius * st;
        let y = e.center.1 + s * e.major_radius * ct + c * e.minor_radius * st;
        let nx = c * ct / e.major_radius - s * st / e.minor_radius;
        let ny = s * ct / e.major_radius + c * st / e.minor_radius;
        let length = nx.hypot(ny);
        let (nx, ny) = (nx / length, ny / length);
        let zone = if nx.abs() > ny.abs() {
            if nx < 0. {
                0
            } else {
                2
            }
        } else if ny > 0. {
            1
        } else {
            continue;
        };
        if let (Some(inside), Some(outside)) = (
            sample(x - 7. * nx, y - 7. * ny),
            sample(x + 7. * nx, y + 7. * ny),
        ) {
            counts[zone][1] += 1;
            counts[zone][0] += usize::from(outside - inside > contrast * 0.04 && contrast > 20.);
        }
    }
    fit.outward_support_left_bottom_right = counts.map(|v| {
        if v[1] >= 8 {
            v[0] as f64 / v[1] as f64
        } else {
            0.
        }
    });
    Some(fit)
}
pub fn fit(raw: &[u16], f: &Value) -> [Option<NativeFit>; 4] {
    let w = data::num(&f["width"]).unwrap() as usize;
    let h = data::num(&f["height"]).unwrap() as usize;
    let coarse = raw_fit::score_stream_eye(raw, w, h);
    let points = coarse
        .points
        .iter()
        .map(|p| [p.x as f64, p.y as f64])
        .collect::<Vec<_>>();
    let a = finish(
        Ellipse {
            center: coarse.center,
            major_radius: coarse.radius,
            minor_radius: coarse.radius * coarse.axis_ratio,
            angle: coarse.axis_angle,
        },
        points.clone(),
        f,
    );
    let route = points.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>();
    let b = raw_fit::fit_sampled_limbus_route_ellipse(&route).and_then(|p| {
        finish(
            Ellipse {
                center: p.center,
                major_radius: p.major_radius,
                minor_radius: p.minor_radius,
                angle: p.angle,
            },
            p.evidence_points.iter().map(|p| [p.x, p.y]).collect(),
            f,
        )
    });
    let p = raw_fit::detect_outer_iris_boundary_between_eyelids_at_sensor(
        raw,
        w,
        h,
        data::num(&f["sensor_x"]).unwrap() as u32,
        data::num(&f["sensor_y"]).unwrap() as u32,
        &coarse,
        &[],
        &[],
    );
    let c = finish(
        Ellipse {
            center: p.center,
            major_radius: p.major_radius,
            minor_radius: p.minor_radius,
            angle: p.angle,
        },
        p.evidence_points.iter().map(|p| [p.x, p.y]).collect(),
        f,
    );
    let pupil = raw_fit::debug_drive_pupil_center(
        raw,
        w,
        h,
        data::num(&f["sensor_x"]).unwrap() as u32,
        data::num(&f["sensor_y"]).unwrap() as u32,
        &coarse,
    );
    let d = pupil.and_then(|pupil| {
        let seed = raw_fit::OuterIrisBoundary {
            center: pupil.center,
            major_radius: coarse.radius,
            minor_radius: coarse.radius,
            angle: 0.,
            ..Default::default()
        };
        let strip = raw_fit::debug_drive_limbus_perimeter_strip_with_pupil_lap_limit(
            raw,
            w,
            h,
            &seed,
            pupil.center,
            5.,
            96,
            2,
        )?;
        let measured = strip
            .samples
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                p.transition_score > 0.
                    && !p.inferred_occlusion
                    && !strip.conic_projected_rows.get(*i).copied().unwrap_or(false)
            })
            .map(|(_, p)| p.driven_point)
            .collect::<Vec<_>>();
        let e = raw_fit::fit_sampled_limbus_route_ellipse(&measured)?;
        finish(
            Ellipse {
                center: e.center,
                major_radius: e.major_radius,
                minor_radius: e.minor_radius,
                angle: e.angle,
            },
            e.evidence_points.iter().map(|p| [p.x, p.y]).collect(),
            f,
        )
    });
    [a, b, c, d]
}
pub fn preview(raw: &[u16], w: usize, h: usize) -> Vec<u8> {
    let (cw, ch) = (w / 4, h / 4);
    let mut p = Vec::with_capacity(cw * ch);
    for y in 0..ch {
        for x in 0..cw {
            let mut v = 0.;
            for dy in 0..4 {
                for dx in 0..4 {
                    v += raw[(y * 4 + dy) * w + x * 4 + dx] as f64;
                }
            }
            p.push(v / 16.);
        }
    }
    let mut sorted = p.clone();
    sorted.sort_by(f64::total_cmp);
    let (lo, hi) = (sorted[sorted.len() / 50], sorted[sorted.len() * 49 / 50]);
    p.into_iter()
        .flat_map(|v| {
            let q = (255. * ((v - lo) / (hi - lo).max(1.)).clamp(0., 1.)).round() as u8;
            [q, q, q, 255]
        })
        .collect()
}
pub fn run(args: &[String]) -> Result<()> {
    if !(3..=4).contains(&args.len()) {
        return Err("native CORPUS NEW_OUT [archive-fragment]".into());
    }
    let out = data::output(&args[2])?;
    let (sources, inventory) = data::scan(&args[1])?;
    data::write(out.join("inventory.json"), &inventory)?;
    let mut log = BufWriter::new(fs::File::create(out.join("native.jsonl"))?);
    let start = Instant::now();
    let mut counts = [0usize; 4];
    let mut total = 0;
    for (si, s) in sources
        .iter()
        .filter(|s| !s.eligible.is_empty() && args.get(3).is_none_or(|p| s.archive.contains(p)))
        .enumerate()
    {
        let bundle = BundleSource::open(Path::new(&s.archive))?;
        let mut page = Canvas::new(1800, 1640)?;
        page.clear();
        page.text(
            20.,
            28.,
            22.,
            WHITE,
            &format!(
                "Fresh non-learned conics | {}",
                Path::new(&s.archive).file_name().unwrap().to_string_lossy()
            ),
        );
        page.text(20.,56.,16.,MUTED,if args[0]=="native-conics" {"Cyan: RAW gradient ellipse. Orange: old seed-ray fit. Dots are observed gradients; curves are fitted completions."}else{"Cyan: coarse. Orange: seed rays. Pink: native boundary. Green: pupil-anchored measured road. Dots=samples, curves=fits."});
        let mut chosen = Vec::new();
        // Include short fixation periods as well as long ones, both eyes.
        for t in 0..s.spans.len() {
            for eye in 1..=2 {
                let ids: Vec<_> = (0..s.eligible.len())
                    .filter(|&j| {
                        s.targets[j] == t
                            && data::num(&s.frames[s.eligible[j][1]]["eye_id"]) == Some(eye)
                    })
                    .collect();
                if !ids.is_empty() {
                    chosen.push(ids[ids.len() / 2]);
                }
            }
        }
        for (k, &j) in chosen.iter().take(12).enumerate() {
            let fi = s.eligible[j][1];
            let f = &s.frames[fi];
            let bytes = bundle.read_range(
                f["stream"].as_str().ok_or("stream")?,
                data::num(&f["offset"]).ok_or("offset")?,
                data::num(&f["length"]).ok_or("length")? as usize,
            )?;
            let raw = unpack(&bytes, f)?;
            let variants = if args[0] == "native-conics" {
                vec![for_supervision(&raw, f), fit(&raw, f)[1].clone()]
            } else {
                fit(&raw, f).to_vec()
            };
            let w = data::num(&f["width"]).unwrap() as usize;
            let h = data::num(&f["height"]).unwrap() as usize;
            let x = 20. + (k % 3) as f64 * 600.;
            let y = 100. + (k / 3) as f64 * 380.;
            let scale = 560. / w as f64;
            let hh = h as f64 * scale;
            page.image(&preview(&raw, w, h), w / 4, h / 4, x, y, 560., hh);
            for (i, v) in variants.iter().enumerate() {
                if let Some(v) = v {
                    counts[i] += 1;
                    let color = [CYAN, ORANGE, PINK, GREEN][i];
                    let p = v
                        .ellipse()
                        .dense_points(121)
                        .into_iter()
                        .map(|(xx, yy)| [x + xx * scale, y + yy * scale])
                        .collect::<Vec<_>>();
                    page.clipped(x, y, 560., hh, |c| {
                        c.path(&p, 2., color);
                        for p in &v.points {
                            c.dot(x + p[0] * scale, y + p[1] * scale, 2., color, true);
                        }
                    });
                }
            }
            page.text(
                x,
                y - 10.,
                16.,
                WHITE,
                &format!(
                    "eye {} seq {} target {:?}",
                    f["eye_id"], f["sequence"], s.spans[s.targets[j]].uv
                ),
            );
            total += 1;
            serde_json::to_writer(
                &mut log,
                &json!({"archive":s.archive,"source":f,"raw_sha256":data::digest(&bytes),"target":s.spans[s.targets[j]],"variants":variants}),
            )?;
            log.write_all(b"\n")?;
        }
        page.png(&out.join(format!("native-{si:02}.png")))?;
        eprintln!("native probe {si}: {total} exposures, fits {counts:?}");
    }
    log.flush()?;
    data::write(
        out.join("result.json"),
        &json!({"frames":total,"fits":counts,"seconds":start.elapsed().as_secs_f64(),"limits":"Non-learned unverified conics; nominal K=4000px and principal=(4000,3000); not sign labels or human localization truth; native detector has runtime budgets; no held output"}),
    )
}
