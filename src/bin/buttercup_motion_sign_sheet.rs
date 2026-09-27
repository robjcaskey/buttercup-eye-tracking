//! Bounded evaluation: native regional RAW tracks versus candidate circle motion.
//! Historical conics are hypotheses, never sign labels. No live solver changes.
#![allow(dead_code)]
#[path = "../"]
mod native {
    pub mod conic_solver;
    pub mod eye_scene_model;
    pub mod geometry;
    pub mod outline_conic_segments;
    pub mod raw10;
    pub mod raw_iris_focus;
    pub mod raw_motion_octrees;
    pub mod roi_evidence;
    pub mod roi_visibility;
}
use native::{
    conic_solver, eye_scene_model, geometry, outline_conic_segments, raw10, raw_iris_focus,
    raw_motion_octrees, roi_evidence, roi_visibility,
};
use raw_motion_octrees::{FourMotionOctrees, IrisEllipseSeed, MotionOctreeOverlay};
#[path = "buttercup_motion_sign_sheet/clusters.rs"]
mod clusters;
#[path = "buttercup_motion_sign_sheet/eye_center.rs"]
mod eye_center;
#[path = "buttercup_motion_sign_sheet/vectors.rs"]
mod motion_vectors;
#[path = "../raw_preview.rs"]
mod raw_preview;
#[path = "buttercup_motion_sign_sheet/sign_video.rs"]
mod sign_video;
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
fn num(v: &Value) -> f64 {
    v.as_f64().expect("numeric corpus field")
}
fn uint(v: &Value) -> u64 {
    v.as_u64().expect("integer corpus field")
}
fn dot(a: V, b: V) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}
fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn mul(a: V, s: f64) -> V {
    a.map(|v| v * s)
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V) -> V {
    mul(a, 1. / norm(a))
}
fn rotate(v: V, axis: V, angle: f64) -> V {
    add(
        add(mul(v, angle.cos()), mul(cross(axis, v), angle.sin())),
        mul(axis, dot(axis, v) * (1. - angle.cos())),
    )
}
fn align(v: V, a: V, b: V) -> V {
    let axis = cross(a, b);
    let s = norm(axis);
    if s < 1e-10 {
        v
    } else {
        rotate(v, mul(axis, 1. / s), s.atan2(dot(a, b)))
    }
}
#[derive(Clone)]
struct Pose {
    c: V,
    n: V,
}
fn poses(f: &Value) -> Vec<Pose> {
    f["poses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| Pose {
            c: std::array::from_fn(|i| num(&p[i])),
            n: unit(std::array::from_fn(|i| num(&p[i + 3]))),
        })
        .collect()
}
fn project(x: V) -> [f64; 2] {
    [4000. + 4000. * x[0] / x[2], 3000. + 4000. * x[1] / x[2]]
}
fn predict(x: [f64; 2], a: &Pose, b: &Pose, twist: f64) -> Option<[f64; 2]> {
    let ray = [(x[0] - 4000.) / 4000., (x[1] - 3000.) / 4000., 1.];
    let d = dot(a.n, ray);
    if d.abs() < 1e-8 {
        return None;
    }
    let distance = dot(a.n, a.c) / d;
    if distance <= 0. {
        return None;
    }
    let local = sub(mul(ray, distance), a.c);
    let moved = add(b.c, rotate(align(local, a.n, b.n), b.n, twist));
    (moved[2] > 0.).then(|| project(moved))
}
#[derive(Clone)]
struct Track {
    id: u64,
    layer: usize,
    a: [f64; 2],
    b: [f64; 2],
    iris: bool,
}
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn median(mut a: Vec<f64>) -> Option<f64> {
    if a.is_empty() {
        None
    } else {
        a.sort_by(f64::total_cmp);
        Some(a[a.len() / 2])
    }
}
fn ellipse(f: &Value) -> IrisEllipseSeed {
    let e = &f["ellipse"];
    let fr = &f["input"]["frame"];
    IrisEllipseSeed {
        center: (
            num(&e["center_sensor_px"][0]) - num(&fr["sensor_x"]),
            num(&e["center_sensor_px"][1]) - num(&fr["sensor_y"]),
        ),
        major_radius: num(&e["a"]),
        minor_radius: num(&e["b"]),
        angle: num(&e["angle"]),
    }
}
fn radius(x: [f64; 2], f: &Value) -> f64 {
    let e = &f["ellipse"];
    let dx = x[0] - num(&e["center_sensor_px"][0]);
    let dy = x[1] - num(&e["center_sensor_px"][1]);
    let (s, c) = num(&e["angle"]).sin_cos();
    ((dx * c + dy * s) / num(&e["a"])).hypot((-dx * s + dy * c) / num(&e["b"]))
}
fn raw(f: &Value) -> Result<Vec<u16>, E> {
    let i = &f["input"];
    let fr = &i["frame"];
    let mut file = fs::File::open(i["raw_file"].as_str().unwrap())?;
    file.seek(SeekFrom::Start(uint(&i["raw_offset"])))?;
    let mut bytes = vec![0; uint(&i["raw_length"]) as usize];
    file.read_exact(&mut bytes)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        f["raw_sha256"].as_str().unwrap()
    );
    Ok(raw10::try_unpack_raw10(
        &bytes,
        uint(&fr["width"]) as usize,
        uint(&fr["height"]) as usize,
        uint(&fr["stride"]) as usize,
    )?)
}
fn tracks(o: &MotionOctreeOverlay, previous: &Value, current: &Value) -> Vec<Track> {
    let fr = &current["input"]["frame"];
    let origin = [num(&fr["sensor_x"]), num(&fr["sensor_y"])];
    o.trails
        .iter()
        .filter(|t| {
            t.matched_streak >= 1
                && t.points.len() >= 2
                && t.layer_evidence
                && !t.normal_flow_evidence
        })
        .map(|t| {
            let p = &t.points[t.points.len() - 2];
            let q = t.points.last().unwrap();
            let a = [p.x as f64 + origin[0], p.y as f64 + origin[1]];
            let b = [q.x as f64 + origin[0], q.y as f64 + origin[1]];
            let r0 = radius(a, previous);
            let r1 = radius(b, current);
            Track {
                id: t.id,
                layer: t.object,
                a,
                b,
                iris: t.object == raw_motion_octrees::PUPIL_LAYER
                    // Same non-reflection bound as SPECULAR_HOLD_SCORE in
                    // the shared tracker; this score is not a probability.
                    && t.specularity < 1.55
                    && (0.30..0.90).contains(&r0)
                    && (0.30..0.90).contains(&r1),
            }
        })
        .collect()
}
fn score(ts: &[Track], a: &Pose, b: &Pose, twist: f64) -> Option<f64> {
    median(
        ts.iter()
            .filter_map(|t| Some(dist(predict(t.a, a, b, twist)?, t.b)))
            .collect(),
    )
}
fn evaluate(ts: &[Track], a: &Value, b: &Value, overlay: &MotionOctreeOverlay) -> Value {
    let iris: Vec<_> = ts.iter().filter(|t| t.iris).cloned().collect();
    let train: Vec<_> = iris.iter().filter(|t| t.id % 2 == 0).cloned().collect();
    let held: Vec<_> = iris.iter().filter(|t| t.id % 2 == 1).cloned().collect();
    let pa = poses(a);
    let pb = poses(b);
    let mut rows = vec![];
    for i in 0..2 {
        for j in 0..2 {
            let mut best = None;
            if train.len() >= 3 {
                for k in -150..=150 {
                    let twist = (k as f64 * 0.1).to_radians();
                    if let Some(cost) = score(&train, &pa[i], &pb[j], twist) {
                        if best.is_none_or(|(_, v)| cost < v) {
                            best = Some((twist, cost));
                        }
                    }
                }
            }
            let (twist, cost) = best.unwrap_or((0., f64::NAN));
            rows.push(json!({"pair":format!("{}→{}",['A','B'][i],['A','B'][j]),"twist_degrees":best.map(|_|twist.to_degrees()),"training_median_px":best.map(|_|cost),"heldout_median_px":if best.is_some()&&held.len()>=3{score(&held,&pa[i],&pb[j],twist)}else{None},"heldout_zero_motion_px":median(held.iter().map(|t|dist(t.a,t.b)).collect::<Vec<_>>()),"twist_bound_hit":best.is_some()&&twist.abs().to_degrees()>14.95}));
        }
    }
    let mut ranked: Vec<_> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, r)| Some((i, r["heldout_median_px"].as_f64()?)))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let span = iris
        .iter()
        .map(|t| {
            let e = &a["ellipse"];
            (t.a[1] - num(&e["center_sensor_px"][1])).atan2(t.a[0] - num(&e["center_sensor_px"][0]))
        })
        .fold([false; 4], |mut q, angle| {
            q[((angle + std::f64::consts::PI) / (std::f64::consts::PI / 2.)).floor() as usize
                % 4] = true;
            q
        });
    // Reuse the live general-layer similarity at its actual fitting center.
    // This is a projected pivot transport diagnostic, not measured 3D motion.
    let fr = &b["input"]["frame"];
    let center = [
        num(&fr["sensor_x"]) + num(&fr["width"]) * 0.5,
        num(&fr["sensor_y"]) + num(&fr["height"]) * 0.5,
    ];
    let motion = overlay.motions[raw_motion_octrees::GENERAL_LAYER];
    let general = overlay.layers[raw_motion_octrees::GENERAL_LAYER];
    let reference_ready = motion.support >= 8
        && general.stable_frames >= 2
        && general.coherence >= 0.3
        && motion.residual.is_finite();
    for i in 0..2 {
        for j in 0..2 {
            let errors: Vec<Value> = [10.,30.,50.].into_iter().map(|depth| {
            let x = project(sub(pa[i].c,mul(pa[i].n,depth)));
            let target = project(sub(pb[j].c,mul(pb[j].n,depth)));
            let dx=x[0]-center[0]; let dy=x[1]-center[1];
            let predicted=[x[0]+motion.translation[0] as f64+motion.diagonal_coefficient_delta as f64*dx-motion.rotation_coefficient as f64*dy,
                           x[1]+motion.translation[1] as f64+motion.rotation_coefficient as f64*dx+motion.diagonal_coefficient_delta as f64*dy];
            json!({"pivot_behind_mm":depth,"residual_px":if reference_ready {Some(dist(predicted,target))}else{None}})
        }).collect();
            rows[i * 2 + j]["pivot_transport"] = json!(errors);
        }
    }
    let margin = if ranked.len() == 4 {
        Some(ranked[1].1 - ranked[0].1)
    } else {
        None
    };
    // Conservative diagnostic gate, not calibrated confidence or anatomical truth.
    let supported = ranked.len() == 4
        && train.len() >= 4
        && held.len() >= 4
        && span.into_iter().filter(|b| *b).count() >= 3
        && ranked[0].1 < 2.
        && margin.unwrap_or(0.) > 0.75
        && !rows[ranked[0].0]["twist_bound_hit"].as_bool().unwrap();
    json!({"fresh_material_tracks":ts.len(),"iris_tracks":iris.len(),"training_tracks":train.len(),"heldout_tracks":held.len(),"occupied_quadrants":span,"pairings":rows,"best_pair":ranked.first().map(|r|rows[r.0]["pair"].clone()),"margin_px":margin,"status":if supported{"conditional_motion_preference"}else{"unresolved"},"tracks":ts.iter().map(|t|json!({"id":t.id,"layer":t.layer,"previous_sensor":t.a,"current_sensor":t.b,"iris_vote":t.iris,"split":if t.id%2==0{"fit"}else{"heldout"}})).collect::<Vec<_>>()})
}
const COLORS: [&str; 4] = ["#44dce9", "#82ee69", "#f377d5", "#ffa45e"];
fn sheet(
    out: &Path,
    previous: &Value,
    current: &Value,
    raws: [&[u16]; 2],
    ts: &[Track],
    result: &Value,
) -> Result<(), E> {
    let mut svg=String::from("<svg xmlns='http://www.w3.org/2000/svg' width='1440' height='1160' viewBox='0 0 1440 1160'><rect width='1440' height='1160' fill='#161922'/><g fill='white' font-family='sans-serif'><text x='35' y='35' font-size='23'>Native Rust regional motion — consecutive RAW exposures</text>");
    let text = |s: &mut String, x: i32, y: i32, size: i32, t: &str| {
        let _ = write!(s, "<text x='{x}' y='{y}' font-size='{size}'>{t}</text>");
    };
    text(
        &mut svg,
        35,
        62,
        16,
        &format!(
            "{} · eye {} · source {} → {} · {}",
            previous["selection"]["capture"].as_str().unwrap(),
            previous["eye"],
            previous["sequence"],
            current["sequence"],
            result["status"].as_str().unwrap()
        ),
    );
    for row in 0..2 {
        let f = if row == 0 { previous } else { current };
        let fr = &f["input"]["frame"];
        let w = uint(&fr["width"]) as usize;
        let h = uint(&fr["height"]) as usize;
        let origin = [num(&fr["sensor_x"]), num(&fr["sensor_y"])];
        let ps = poses(f);
        for col in 0..2 {
            let x0 = 35 + col * 710;
            let y0 = 112 + row * 380;
            let scale = 1.2;
            text(
                &mut svg,
                x0 as i32,
                y0 as i32 - 12,
                17,
                &format!("Source {} · candidate {}", f["sequence"], ['A', 'B'][col]),
            );
            write!(svg, "<g transform='translate({x0},{y0}) scale({scale})'>")?;
            // Display only red CFA sites, matching the latest raw-only comparison.
            for y in 0..h {
                for x in 0..w {
                    if (x + origin[0] as usize) % 4 >= 2 || (y + origin[1] as usize) % 4 >= 2 {
                        continue;
                    }
                    let g = (raws[row][y * w + x] as f64 * 255. / 1023.).round() as u8;
                    write!(svg,"<rect x='{x}' y='{y}' width='1' height='1' fill='#{g:02x}{g:02x}{g:02x}'/>")?;
                }
            }
            let e = ellipse(f);
            write!(svg,"<ellipse cx='{}' cy='{}' rx='{}' ry='{}' transform='rotate({} {} {})' fill='none' stroke='white' stroke-width='.7'/>",e.center.0,e.center.1,e.major_radius,e.minor_radius,e.angle.to_degrees(),e.center.0,e.center.1)?;
            let p = project(ps[col].c);
            let q = project(add(ps[col].c, mul(ps[col].n, 7.)));
            write!(svg,"<path d='M {} {} L {} {}' stroke='#ffdf44' stroke-width='1.4'/><circle cx='{}' cy='{}' r='2' fill='#ffdf44'/>",p[0]-origin[0],p[1]-origin[1],q[0]-origin[0],q[1]-origin[1],q[0]-origin[0],q[1]-origin[1])?;
            for t in ts {
                let pt = if row == 0 { t.a } else { t.b };
                let color = COLORS[t.layer.min(3)];
                write!(svg,"<circle cx='{}' cy='{}' r='{}' stroke='{color}' stroke-width='.9' fill='none'/>",pt[0]-origin[0],pt[1]-origin[1],if t.iris{2.5}else{1.5})?;
                if row == 1 {
                    // Exaggerate only the display tail; endpoints and all
                    // scoring continue to use measured native coordinates.
                    write!(
                        svg,
                        "<path d='M {} {} L {} {}' stroke='{color}' stroke-width='.8'/>",
                        t.b[0] - 8.0 * (t.b[0] - t.a[0]) - origin[0],
                        t.b[1] - 8.0 * (t.b[1] - t.a[1]) - origin[1],
                        t.b[0] - origin[0],
                        t.b[1] - origin[1]
                    )?;
                }
            }
            svg.push_str("</g>");
        }
    }
    text(
        &mut svg,
        35,
        895,
        17,
        &format!(
            "Fresh material tracks: {} · iris votes: {} · fit/held-out: {}/{}",
            result["fresh_material_tracks"],
            result["iris_tracks"],
            result["training_tracks"],
            result["heldout_tracks"]
        ),
    );
    for (i, r) in result["pairings"].as_array().unwrap().iter().enumerate() {
        let cost = r["heldout_median_px"]
            .as_f64()
            .map(|x| format!("{x:.3} px"))
            .unwrap_or("insufficient tracks".into());
        text(
            &mut svg,
            35,
            928 + i as i32 * 27,
            17,
            &format!(
                "{}: texture {} · pivot errors at 10/30/50 mm: {}",
                r["pair"].as_str().unwrap(),
                cost,
                r["pivot_transport"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v["residual_px"]
                        .as_f64()
                        .map(|x| format!("{x:.2} px"))
                        .unwrap_or("unavailable".into()))
                    .collect::<Vec<_>>()
                    .join(" / ")
            ),
        );
    }
    let general = &result["layers"][0];
    text(&mut svg,35,1040,15,&format!("General reference: {} tracks; fit residual {:.2} px. Pivot errors assume a shared 2D material motion.",general["support"],general["residual_px"].as_f64().unwrap_or(0.)));
    text(&mut svg,35,1060,15,"Tracker groups: cyan general · green pupil/iris · magenta reflection · orange residual. Motion tails shown at 8×.");
    text(&mut svg,35,1086,15,"White: fitted conic; yellow: hypothetical normal. Red-site RAW backdrop; tracking uses the existing native RAW pipeline.");
    text(&mut svg,35,1112,15,"No known 3D motion or sign truth. Planar iris, fixed scale/intrinsics, and feature identity remain assumptions.");
    text(&mut svg,35,1138,15,"Unresolved is intentional when support or separation is insufficient; Z-layer values are not metric depth.");
    svg.push_str("</g></svg>");
    fs::write(out, svg)?;
    Ok(())
}
fn main() -> Result<(), E> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 4 && args[1] == "--sign-video" {
        return sign_video::run(Path::new(&args[2]), Path::new(&args[3]));
    }
    if args.len() == 4 && args[1] == "--eye-center-render" {
        return eye_center::render_saved(Path::new(&args[2]), Path::new(&args[3]));
    }
    if args.len() == 4 && args[1] == "--eye-center-corpus" {
        return eye_center::run(Path::new(&args[2]), Path::new(&args[3]));
    }
    if args.len() == 5 && args[1] == "--eye-center-transport" {
        return eye_center::transport_audit(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
        );
    }
    if args.len() == 6 && args[1] == "--animate-groups" {
        return clusters::replay(&args[2], &args[3], &args[4], Path::new(&args[5]));
    }
    if (args.len() == 4 || args.len() == 5) && args[1] == "--cluster-features" {
        return clusters::run(
            &args[2],
            Path::new(&args[3]),
            args.get(4).map(String::as_str),
        );
    }
    if args.len() != 3 {
        return Err("usage: buttercup_motion_sign_sheet INPUT_RESULTS_JSON NEW_OUTPUT_DIR".into());
    }
    let output = Path::new(&args[2]);
    if output.exists() {
        return Err("output must be new".into());
    }
    let parent = output.parent().ok_or("missing parent")?.canonicalize()?;
    if !parent.starts_with("/mnt/bulk_data/buttercup-eye-tracking") {
        return Err("output must be below checked bulk runtime root".into());
    }
    fs::create_dir(output)?;
    let d: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut frames: Vec<_> = d["frames"].as_array().unwrap().iter().collect();
    frames.sort_by_key(|f| {
        (
            f["selection"]["capture"].as_str().unwrap(),
            uint(&f["eye"]),
            uint(&f["sequence"]),
        )
    });
    let mut tracker = FourMotionOctrees::default();
    let mut prev: Option<(&Value, Vec<u16>)> = None;
    let mut reports = vec![];
    let mut pupil_regions = vec![];
    let mut sheets = 0;
    for f in frames {
        let fr = &f["input"]["frame"];
        let timestamp = uint(&fr["timestamp_ns"]);
        let contiguous = prev.as_ref().is_some_and(|(p, _)| {
            p["selection"]["capture"] == f["selection"]["capture"]
                && p["eye"] == f["eye"]
                && p["selection"]["clock"] == f["selection"]["clock"]
                && uint(&p["sequence"]) + 1 == uint(&f["sequence"])
                && timestamp > uint(&p["input"]["frame"]["timestamp_ns"])
                && timestamp - uint(&p["input"]["frame"]["timestamp_ns"]) <= 250_000_000
        });
        if !contiguous {
            tracker.clear();
        }
        let pixels = raw(f)?;
        let overlay = tracker.observe_with_iris_seed_at(
            &pixels,
            uint(&fr["width"]) as usize,
            uint(&fr["height"]) as usize,
            uint(&fr["sensor_x"]) as u32,
            uint(&fr["sensor_y"]) as u32,
            timestamp,
            None,
            true,
            Some(ellipse(f)),
        );
        pupil_regions.push(json!({"capture":f["selection"]["capture"],"eye":f["eye"],"sequence":f["sequence"],
            "pupil":overlay.nested_eye_boundaries.as_ref().map(|p|json!({"center_sensor":[p.pupil.center.0+num(&fr["sensor_x"]),p.pupil.center.1+num(&fr["sensor_y"])],"a":p.pupil.major_radius,"b":p.pupil.minor_radius,"angle":p.pupil.angle,"support":p.pupil_support,"confidence_heuristic":p.confidence}))}));
        if contiguous {
            let (p, pr) = prev.as_ref().unwrap();
            let ts = tracks(&overlay, p, f);
            let mut result = evaluate(&ts, p, f, &overlay);
            result["capture"] = f["selection"]["capture"].clone();
            result["eye"] = f["eye"].clone();
            result["previous_sequence"] = p["sequence"].clone();
            result["sequence"] = f["sequence"].clone();
            result["dt_ms"] =
                json!((timestamp - uint(&p["input"]["frame"]["timestamp_ns"])) as f64 / 1e6);
            result["raw_sha256"] = json!([p["raw_sha256"], f["raw_sha256"]]);
            result["layers"]=json!((0..4).map(|k|{let l=overlay.layers[k];let m=overlay.motions[k];json!({"layer":k,"tracks":l.persistent_tracks,"coherence":l.coherence,"differential":l.differential,"parallax_proxy":l.parallax,"translation":m.translation,"residual_px":m.residual,"support":m.support})}).collect::<Vec<_>>());
            if ((60..=61).contains(&uint(&p["sequence"]))
                && f["selection"]["capture"] == "complete-nine"
                && f["eye"] == 1)
                || (uint(&p["sequence"]) == 1034
                    && f["selection"]["capture"] == "later-recording"
                    && f["eye"] == 2)
            {
                let name = format!(
                    "contact-{}-eye{}-{}.svg",
                    f["selection"]["capture"].as_str().unwrap(),
                    f["eye"],
                    p["sequence"]
                );
                sheet(&output.join(&name), p, f, [pr, &pixels], &ts, &result)?;
                sheets += 1;
            }
            println!(
                "{} eye{} {}→{}: {} fresh {} iris; {}",
                f["selection"]["capture"],
                f["eye"],
                p["sequence"],
                f["sequence"],
                ts.len(),
                result["iris_tracks"],
                result["status"]
            );
            reports.push(result);
        }
        prev = Some((f, pixels));
    }
    fs::write(
        output.join("pupil-regions.json"),
        serde_json::to_vec_pretty(&pupil_regions)?,
    )?;
    let eligible = reports
        .iter()
        .filter(|r| r["pairings"][0]["heldout_median_px"].is_number())
        .count();
    let preferred = reports
        .iter()
        .filter(|r| r["status"] == "conditional_motion_preference")
        .count();
    fs::write(
        output.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"conditional RAW motion consistency; no independent sign truth; no live changes","input":args[1],"pairs":reports,"eligible_pairs":eligible,"conditional_preferences":preferred,"sheets":sheets}),
        )?,
    )?;
    println!("eligible={eligible} preferred={preferred} sheets={sheets}");
    Ok(())
}
