//! Full-recording, held-out-day appearance-model review, including failures.
use super::{bootstrapability as boot, data, native, Result};
#[path = "canvas.rs"]
mod canvas;
use buttercup_eye_tracking::{
    calibration_sign_model::{self as net, Model, Prediction},
    geometry::{projected_circle_candidates, Ellipse},
    recorded_bundle::BundleSource,
};
use canvas::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fs,
    io::{BufWriter, Write},
    path::Path,
    process::{Command, Stdio},
    time::Instant,
};
const W: usize = 1920;
const H: usize = 1080;
const FPS: usize = 25;
const EYE_COLORS: [[f64; 3]; 2] = [CYAN, PINK];
type Key = (String, u64, u64, u64);
fn n(v: &Value) -> u64 {
    data::num(v).unwrap_or(0)
}
fn key(f: &Value) -> Key {
    (
        f["source_clock"]["source_key"]["stream_epoch"]
            .as_str()
            .unwrap_or("")
            .to_owned(),
        n(&f["eye_id"]),
        n(&f["sequence"]),
        n(&f["timestamp_ns"]),
    )
}
fn source_key(epoch: &str, v: &Value) -> Key {
    (
        epoch.to_owned(),
        n(&v["roi_id"]),
        n(&v["sequence"]),
        n(&v["sensor_timestamp_ns"]),
    )
}
#[derive(Clone)]
struct Shape {
    ellipse: Ellipse,
    normals: [[f64; 3]; 2],
    centers: [[f64; 3]; 2],
    points: Vec<[f64; 2]>,
    origin: [u32; 2],
    size: [usize; 2],
    intrinsics: Value,
    admissible: bool,
    role: &'static str,
}
impl Shape {
    fn native(f: native::NativeFit, frame: &Value) -> Self {
        let role = f.role();
        Self {
            ellipse: f.ellipse(),
            normals: f.normals,
            centers: f.centers_per_radius,
            admissible: f.admissible(),
            points: f.points,
            origin: f.origin,
            size: [n(&frame["width"]) as usize, n(&frame["height"]) as usize],
            intrinsics: json!({"focal_px":[4000.,4000.],"principal_px":[4000.,3000.],"measured":false}),
            role,
        }
    }
}
fn shapes(bundle: &BundleSource) -> Result<(HashMap<Key, Shape>, Value)> {
    let bytes = match bundle.read_entry("predictions.jsonl") {
        Ok(v) => v,
        Err(e) => return Ok((HashMap::new(), json!({"unavailable":e}))),
    };
    let hash = data::digest(&bytes);
    let mut result = HashMap::new();
    let mut considered = 0;
    for line in std::str::from_utf8(&bytes)?.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(line)?;
        let epoch = row["source_clock"]["source_key"]["stream_epoch"]
            .as_str()
            .unwrap_or("");
        if epoch.is_empty() {
            continue;
        }
        let joint = &row["predictions"]["eye_candidate"]["centers_and_gaze"]["joint_conics"];
        let intr = &joint["intrinsics"];
        let pair = |v: &Value| Some([v[0].as_f64()?, v[1].as_f64()?]);
        let (Some(focal), Some(principal)) = (pair(&intr["focal_px"]), pair(&intr["principal_px"]))
        else {
            continue;
        };
        for eye in joint["source_projected_conics"]["eyes"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if eye["modeled_eye"] != true {
                continue;
            }
            let source = &eye["source"];
            let k = source_key(epoch, source);
            let e = &eye["ellipses"][0];
            let Some(c) = pair(&e["center"]) else {
                continue;
            };
            let (Some(a), Some(b), Some(angle)) = (
                e["major_radius"].as_f64(),
                e["minor_radius"].as_f64(),
                e["angle_rad"].as_f64(),
            ) else {
                continue;
            };
            let ellipse = Ellipse {
                center: (c[0], c[1]),
                major_radius: a,
                minor_radius: b,
                angle,
            };
            let origin = [
                n(&eye["sensor_origin_px"][0]) as u32,
                n(&eye["sensor_origin_px"][1]) as u32,
            ];
            let size = [
                n(&eye["dimensions_px"][0]) as usize,
                n(&eye["dimensions_px"][1]) as usize,
            ];
            let Some(poses) = projected_circle_candidates(ellipse, origin, focal, principal) else {
                continue;
            };
            let points = joint["arcs"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|a| a["kind"] == "OuterLimbus" && source_key(epoch, &a["source"]) == k)
                .flat_map(|a| {
                    a["points_roi_px"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(pair)
                })
                .collect();
            considered += 1;
            result.entry(k).or_insert(Shape {
                ellipse,
                normals: poses.map(|p| p.0),
                centers: poses.map(|p| p.1),
                points,
                origin,
                size,
                intrinsics: intr.clone(),
                admissible: false,
                role: "archived diagnostic conic",
            });
        }
    }
    Ok((
        result,
        json!({"predictions_sha256":hash,"candidate_records":considered,"role":"retrospective source-matched archived joint geometry only; NOT model inputs, supervision, or sign truth"}),
    ))
}
struct Eye {
    frame: Value,
    hash: String,
    preview: Vec<u8>,
    small: Option<Vec<f32>>,
    previous_small: Option<Vec<f32>>,
    prediction: Option<Prediction>,
    baseline: Option<Prediction>,
    shape: Option<Shape>,
    eligible: bool,
    reason: String,
    teacher: Option<usize>,
}
impl Eye {
    fn branch(&self) -> Option<net::BranchPreference> {
        self.prediction.as_ref().map(|p| {
            p.choose_native_branches(
                self.shape
                    .as_ref()
                    .filter(|s| s.admissible)
                    .map(|s| s.normals),
            )
        })
    }
}
fn bgra(pixels: &[f32]) -> Vec<u8> {
    pixels
        .iter()
        .flat_map(|v| {
            let p = ((v + 1.) * 127.5).clamp(0., 255.) as u8;
            [p, p, p, 255]
        })
        .collect()
}
fn native_preview(raw: &[u8], f: &Value) -> Result<Vec<u8>> {
    let (w, h, s) = (
        n(&f["width"]) as usize,
        n(&f["height"]) as usize,
        n(&f["stride"]) as usize,
    );
    let raw = buttercup_eye_tracking::raw10::try_unpack_raw10(raw, w, h, s)?;
    let mut cells = vec![0f32; (w / 4) * (h / 4)];
    for y in 0..h / 4 {
        for x in 0..w / 4 {
            let mut sum = 0.;
            for yy in 0..4 {
                for xx in 0..4 {
                    sum += raw[(y * 4 + yy) * w + x * 4 + xx] as f32;
                }
            }
            cells[y * (w / 4) + x] = sum / 16.;
        }
    }
    let mut sorted = cells.clone();
    sorted.sort_by(f32::total_cmp);
    let lo = sorted[sorted.len() / 200];
    let hi = sorted[sorted.len() * 199 / 200];
    Ok(cells
        .iter()
        .flat_map(|v| {
            let p = (255. * ((v - lo) / (hi - lo).max(1.)).clamp(0., 1.).powf(0.7)) as u8;
            [p, p, p, 255]
        })
        .collect())
}
fn target<'a>(s: &'a data::Source, time: u64) -> Option<&'a data::Span> {
    s.spans.iter().find(|v| v.start <= time && time < v.end)
}
fn active_class(p: &Prediction) -> usize {
    p.target_scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0
}
fn screen(c: &mut Canvas, eyes: &[Option<Eye>; 2], target: Option<&data::Span>) {
    c.text(
        1280.,
        110.,
        23.,
        WHITE,
        "Screen target and appearance prediction",
    );
    let (x, y, w, h) = (1290., 134., 580., 326.25);
    c.rect(x, y, w, h, [0.105, 0.14, 0.18]);
    for t in [0.1, 0.5, 0.9] {
        c.line([x + t * w, y], [x + t * w, y + h], 1., [0.17, 0.21, 0.26]);
        c.line([x, y + t * h], [x + w, y + t * h], 1., [0.17, 0.21, 0.26]);
    }
    if let Some(t) = target {
        c.cross(x + t.uv[0] as f64 * w, y + t.uv[1] as f64 * h, 13., WHITE);
        c.dot(
            x + t.uv[0] as f64 * w,
            y + t.uv[1] as f64 * h,
            17.,
            WHITE,
            false,
        );
    }
    for (i, e) in eyes.iter().enumerate() {
        if let Some(e) = e {
            if let Some(p) = &e.baseline {
                c.cross(x + p.uv[0] as f64 * w, y + p.uv[1] as f64 * h, 6., MUTED);
            }
            if let Some(p) = &e.prediction {
                c.dot(
                    x + p.uv[0] as f64 * w,
                    y + p.uv[1] as f64 * h,
                    8.,
                    EYE_COLORS[i],
                    true,
                );
            }
        }
    }
    c.text(
        1285.,
        493.,
        18.,
        WHITE,
        "White ring: displayed target (not measured fixation)",
    );
    c.text(
        1285.,
        519.,
        18.,
        MUTED,
        "Gray crosses: matched current-frame-only baseline",
    );
}
fn heatmap(c: &mut Canvas, p: Option<&Prediction>, x: f64, y: f64, color: [f64; 3]) {
    for row in 0..3 {
        for col in 0..3 {
            let v = p
                .map(|p| p.target_scores[row * 3 + col] as f64)
                .unwrap_or(0.);
            let tint = color.map(|a| 0.10 + v * 0.8 * a);
            c.rect(x + col as f64 * 76., y + row as f64 * 54., 72., 50., tint);
            c.text(
                x + col as f64 * 76. + 15.,
                y + row as f64 * 54. + 32.,
                20.,
                WHITE,
                &format!("{:>3.0}", v * 100.),
            );
        }
    }
}
type Trace = VecDeque<(f64, [Option<[f32; 2]>; 2], Option<[f32; 2]>, [bool; 2])>;
fn plot(c: &mut Canvas, trace: &Trace, t: f64, x: f64, y: f64, w: f64, h: f64, axis: usize) {
    c.rect(x, y, w, h, [0.085, 0.115, 0.15]);
    for v in [0.1, 0.5, 0.9] {
        c.line([x, y + v * h], [x + w, y + v * h], 1., [0.19, 0.22, 0.27]);
    }
    for layer in 0..3 {
        let color = if layer == 2 { WHITE } else { EYE_COLORS[layer] };
        let mut path = Vec::new();
        for (time, p, target, _) in trace {
            let v = if layer == 2 {
                target.map(|p| p[axis])
            } else {
                p[layer].map(|p| p[axis])
            };
            if let Some(v) = v {
                path.push([
                    x + ((*time - t + 8.) / 8.).clamp(0., 1.) * w,
                    y + v as f64 * h,
                ]);
            } else {
                c.path(&path, if layer == 2 { 1.5 } else { 2. }, color);
                path.clear();
            }
        }
        c.path(&path, if layer == 2 { 1.5 } else { 2. }, color);
    }
    c.text(
        x + 5.,
        y + 18.,
        15.,
        MUTED,
        if axis == 0 {
            "LAST 8s: horizontal screen coordinate"
        } else {
            "LAST 8s: vertical screen coordinate"
        },
    );
}
fn normals(c: &mut Canvas, eyes: &[Option<Eye>; 2]) {
    c.text(1280., 797., 22., WHITE, "Two physical conic hypotheses");
    c.text(
        1280.,
        820.,
        15.,
        MUTED,
        "Cyan A / orange B; +Z toward camera; uncalibrated intrinsics",
    );
    for (i, e) in eyes.iter().enumerate() {
        let cx = 1410. + i as f64 * 300.;
        let cy = 901.;
        c.dot(cx, cy, 76., [0.3, 0.37, 0.44], false);
        c.line([cx - 86., cy], [cx + 86., cy], 1., MUTED);
        c.line([cx, cy - 86.], [cx, cy + 86.], 1., MUTED);
        if let Some(s) = e.as_ref().and_then(|e| e.shape.as_ref()) {
            for j in 0..2 {
                let n = s.normals[j];
                let tip = [cx + 74. * n[0], cy + 74. * n[1]];
                c.arrow([cx, cy], tip, if j == 0 { CYAN } else { ORANGE });
                c.dot(
                    tip[0],
                    tip[1],
                    4. + n[2].max(0.) * 3.,
                    if j == 0 { CYAN } else { ORANGE },
                    true,
                );
                if e.as_ref()
                    .and_then(Eye::branch)
                    .is_some_and(|b| b.selected == Some(j))
                {
                    c.dot(tip[0], tip[1], 12., GREEN, false);
                }
            }
            let angle = s.normals[0]
                .iter()
                .zip(s.normals[1])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                .clamp(-1., 1.)
                .acos()
                .to_degrees();
            c.text(
                cx - 112.,
                991.,
                15.,
                MUTED,
                &format!("Eye {}: {:.1} deg apart", i + 1, angle),
            );
        } else {
            c.text(cx - 66., cy, 16., MUTED, "NO MATCHED FIT");
        }
        if let Some(e) = e {
            if let Some(branch) = e.branch() {
                let text = if branch.votes.iter().any(|p| *p > 0.) {
                    format!(
                        "{} | A {:.0}% B {:.0}%",
                        branch
                            .selected
                            .map(|j| if j == 0 { "PICK A" } else { "PICK B" })
                            .unwrap_or("ABSTAIN"),
                        100. * branch.votes[0],
                        100. * branch.votes[1]
                    )
                } else {
                    "ABSTAIN: no supported conic/head".into()
                };
                c.text(
                    cx - 128.,
                    1013.,
                    14.,
                    if branch.selected.is_some() {
                        GREEN
                    } else {
                        ORANGE
                    },
                    &text,
                );
            }
            if let Some((label, scores)) = e.teacher.zip(
                e.prediction
                    .as_ref()
                    .and_then(|p| p.conditional_branch_scores),
            ) {
                let good = usize::from(scores[1] > scores[0]) == label;
                c.text(
                    cx - 128.,
                    1032.,
                    13.,
                    if good { GREEN } else { RED },
                    if good {
                        "Agrees with conditional teacher"
                    } else {
                        "DISAGREES with conditional teacher"
                    },
                );
            } else {
                c.text(
                    cx - 128.,
                    1032.,
                    13.,
                    MUTED,
                    "No conditional teacher comparison",
                );
            }
        }
    }
}
fn render(
    c: &mut Canvas,
    s: &data::Source,
    record: usize,
    total: usize,
    elapsed: f64,
    ns: u64,
    host: u64,
    eyes: &[Option<Eye>; 2],
    trace: &Trace,
) {
    c.clear();
    c.text(
        20.,
        39.,
        29.,
        WHITE,
        "Two-frame appearance model / held-out corpus review",
    );
    let name = Path::new(&s.archive).file_name().unwrap().to_string_lossy();
    c.text(
        20.,
        74.,
        19.,
        MUTED,
        &format!(
            "Recording {record}/{total} | {} | day {} held out | source +{elapsed:6.2}s",
            name, s.day
        ),
    );
    let current_target = target(s, host);
    screen(c, eyes, current_target);
    for i in 0..2 {
        let x = 20. + 620. * i as f64;
        let color = EYE_COLORS[i];
        c.text(x, 113., 23., color, &format!("Eye {} / source RAW", i + 1));
        c.rect(x, 132., 600., 400., [0.025, 0.035, 0.05]);
        if let Some(e) = &eyes[i] {
            let w = n(&e.frame["width"]) as usize;
            let h = n(&e.frame["height"]) as usize;
            c.image(&e.preview, w / 4, h / 4, x, 132., 600., 400.);
            if let Some(s) = &e.shape {
                let sx = 600. / w as f64;
                let sy = 400. / h as f64;
                let path: Vec<_> = s
                    .ellipse
                    .dense_points(160)
                    .into_iter()
                    .map(|(a, b)| [x + a * sx, 132. + b * sy])
                    .collect();
                let mut closed = path.clone();
                if let Some(p) = path.first() {
                    closed.push(*p);
                }
                c.clipped(x, 132., 600., 400., |c| {
                    c.path(&closed, 2., WHITE);
                    for p in &s.points {
                        if (0.0..w as f64).contains(&p[0]) && (0.0..h as f64).contains(&p[1]) {
                            c.dot(x + p[0] * sx, 132. + p[1] * sy, 2., PINK, true);
                        }
                    }
                });
            }
            let age = ns.saturating_sub(n(&e.frame["timestamp_ns"])) / 1_000_000;
            if age > 200 {
                c.shade(x, 132., 600., 400., 0.65);
                c.text(
                    x + 35.,
                    345.,
                    27.,
                    WHITE,
                    &format!("No new RAW exposure for {age} ms"),
                );
            }
            c.text(
                x + 6.,
                520.,
                16.,
                WHITE,
                &format!(
                    "seq {} | {w}x{h} | {}",
                    n(&e.frame["sequence"]),
                    e.shape
                        .as_ref()
                        .map(|s| s.role)
                        .unwrap_or("no source-matched conic")
                ),
            );
            c.text(
                x,
                560.,
                18.,
                WHITE,
                "Previous frame       Current frame          Absolute change",
            );
            for (j, p) in [e.previous_small.as_ref(), e.small.as_ref()]
                .into_iter()
                .enumerate()
            {
                if let Some(p) = p {
                    c.image(
                        &bgra(p),
                        net::WIDTH,
                        net::HEIGHT,
                        x + j as f64 * 200.,
                        576.,
                        192.,
                        144.,
                    );
                } else {
                    c.rect(x + j as f64 * 200., 576., 192., 144., [0.10, 0.13, 0.17]);
                }
            }
            if let Some((a, b)) = e.previous_small.as_ref().zip(e.small.as_ref()) {
                let diff: Vec<_> = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| ((a - b).abs() * 2.).min(2.) - 1.)
                    .collect();
                c.image(
                    &bgra(&diff),
                    net::WIDTH,
                    net::HEIGHT,
                    x + 400.,
                    576.,
                    192.,
                    144.,
                );
            }
            let status = if e.eligible {
                if let Some(t) = target(s, n(&e.frame["host_arrival_unix_ns"])) {
                    if e.prediction
                        .as_ref()
                        .is_some_and(|p| active_class(p) == net::class(t.uv))
                    {
                        ("MATCHES DISPLAYED TARGET", GREEN)
                    } else {
                        ("DISAGREES WITH DISPLAYED TARGET", RED)
                    }
                } else {
                    ("NO TIMED TARGET", MUTED)
                }
            } else {
                ("NOT A SCORED CALIBRATION PAIR", ORANGE)
            };
            c.text(x, 750., 19., status.1, status.0);
            c.text(x, 778., 16., MUTED, &e.reason);
            if let Some(p) = &e.prediction {
                c.text(
                    x,
                    809.,
                    19.,
                    color,
                    &format!(
                        "Predicted screen ({:.2}, {:.2}) | top-bin support {:.0}%",
                        p.uv[0],
                        p.uv[1],
                        p.target_scores[active_class(p)] * 100.
                    ),
                );
            }
        } else {
            c.text(x + 65., 330., 28., MUTED, "Waiting for native ROI exposure");
        }
        c.text(
            1290. + i as f64 * 300.,
            560.,
            20.,
            EYE_COLORS[i],
            &format!("Eye {}: 3x3 target scores", i + 1),
        );
        heatmap(
            c,
            eyes[i].as_ref().and_then(|e| e.prediction.as_ref()),
            1290. + i as f64 * 300.,
            578.,
            EYE_COLORS[i],
        );
    }
    plot(c, trace, elapsed, 20., 839., 600., 168., 0);
    plot(c, trace, elapsed, 640., 839., 600., 168., 1);
    normals(c, eyes);
    c.text(20.,1044.,15.,WHITE,"White = fitted completion; pink = measured edge samples. Weak teacher assumes nominal optics / upright camera.");
    c.text(20.,1070.,17.,MUTED,"32x24 blurred inputs / CPU inference / no targets at inference. Learned sign scores are conditional, not independently measured 3D truth.");
    c.text(
        1280.,
        769.,
        16.,
        MUTED,
        "Scores are uncalibrated. Green ring = learned choice, if supported.",
    );
}

fn model(run: &Path, day: u64, arm: &str, report: &Value) -> Result<Model> {
    let name = format!("day-{day}-{arm}.json");
    let bytes = fs::read(run.join(&name))?;
    let entry = report["model_artifacts"]
        .as_array()
        .ok_or("model artifact manifest")?
        .iter()
        .find(|r| r["path"] == name)
        .ok_or("missing held-out model manifest entry")?;
    if entry["sha256"] != data::digest(&bytes) {
        return Err("held-out model hash changed".into());
    }
    let m: Model = serde_json::from_slice(&bytes)?;
    m.validate()?;
    if m.provenance["test_day"] != day || m.provenance["source"] != report["source"] {
        return Err("model is not the required held-out-day artifact".into());
    }
    Ok(m)
}
fn card(c: &mut Canvas, heading: &str, lines: &[String]) {
    c.clear();
    c.text(90., 160., 42., WHITE, heading);
    for (i, line) in lines.iter().enumerate() {
        c.text(
            90.,
            245. + i as f64 * 57.,
            27.,
            if i == 0 { CYAN } else { WHITE },
            line,
        );
    }
}
fn emit(c: &mut Canvas, w: &mut impl Write, count: usize, frame_count: &mut usize) -> Result<()> {
    let bytes = c.bytes();
    for _ in 0..count {
        w.write_all(bytes)?;
    }
    *frame_count += count;
    Ok(())
}
fn frame_record(e: &Eye) -> Value {
    json!({"frame":e.frame,"raw_sha256":e.hash,"prediction":e.prediction,"single_frame_baseline":e.baseline,"scored_pair":e.eligible,"admission":e.reason,"conditional_teacher":e.teacher,"conditional_choice":e.branch(),"geometry":e.shape.as_ref().map(|s|json!({"ellipse_roi":{"center":[s.ellipse.center.0,s.ellipse.center.1],"a":s.ellipse.major_radius,"b":s.ellipse.minor_radius,"angle":s.ellipse.angle},"normals":s.normals,"centers_per_radius":s.centers,"segment_samples":s.points,"intrinsics":s.intrinsics,"role":s.role,"admissible":s.admissible,"truth":false,"inference_input":false}))})
}

pub fn run(args: &[String]) -> Result<()> {
    if !(4..=5).contains(&args.len()) {
        return Err(
            "movie CALIBRATION_CORPUS TRAIN_RUN NEW_OUTPUT_DIR [PREVIEW_ARCHIVE_FRAGMENT]".into(),
        );
    }
    let start = Instant::now();
    let run = Path::new(&args[2]);
    let out = data::output(&args[3])?;
    let report: Value = serde_json::from_slice(&fs::read(run.join("results.json"))?)?;
    let source = boot::current_source(Path::new("."))?;
    if serde_json::to_value(&source)? != report["source"] {
        return Err("movie requires a cold run on the current stamped source tree".into());
    }
    let preview = args.get(4);
    let (all, inventory) = data::scan(&args[1])?;
    let sources: Vec<_> = all
        .into_iter()
        .filter(|s| !s.eligible.is_empty() && preview.is_none_or(|p| s.archive.contains(p)))
        .collect();
    if sources.is_empty() {
        return Err("no qualifying corpus recordings".into());
    }
    for s in &sources {
        let frozen = report["inventory"]["sources"]
            .as_array()
            .ok_or("frozen inventory")?
            .iter()
            .find(|v| v["session"] == s.session)
            .ok_or("corpus changed after training")?;
        if frozen["metadata_sha256"] != s.metadata_sha256
            || frozen["frame_index_sha256"] != s.frame_index_sha256
        {
            return Err("recording changed after training".into());
        }
    }
    let branches = !report["branch_label_audit"].is_null();
    let sam_geometry = report["branch_label_audit"]["geometry_provider"] == "sam31-single";
    let mut native_fits = HashMap::<String, Option<native::NativeFit>>::new();
    if branches {
        let bytes = fs::read(run.join("native-conics.jsonl"))?;
        if data::digest(&bytes)
            != report["branch_label_audit"]["native_conics_sha256"]
                .as_str()
                .ok_or("native conic hash")?
        {
            return Err("native geometry changed after cold run".into());
        }
        for line in std::str::from_utf8(&bytes)?.lines() {
            let row: Value = serde_json::from_str(line)?;
            let f: Option<native::NativeFit> = serde_json::from_value(row["fit"].clone())?;
            native_fits.insert(
                native::identity(
                    row["source"]["raw_sha256"]
                        .as_str()
                        .ok_or("native RAW hash")?,
                    &row["source"]["frame"],
                ),
                f,
            );
        }
    }
    let mut expected = HashMap::<String, ([f32; 9], Option<[f32; 2]>, Option<usize>)>::new();
    for day in report["days"].as_array().ok_or("model days")? {
        let day = n(day);
        for line in
            fs::read_to_string(run.join(format!("predictions-day-{day}-two-frame.jsonl")))?.lines()
        {
            let row: Value = serde_json::from_str(line)?;
            let pair = &row["source"]["raw_sha256"];
            let k = format!(
                "{}:{}",
                pair[0].as_str().unwrap(),
                pair[1].as_str().unwrap()
            );
            let p: [f32; 9] = serde_json::from_value(row["class_scores"].clone())?;
            expected.insert(
                k,
                (
                    p,
                    serde_json::from_value(row["conditional_branch_scores"].clone())?,
                    row["conditional_branch_label"].as_u64().map(|n| n as usize),
                ),
            );
        }
    }
    let mut c = Canvas::new(W, H)?;
    let staging = out.join("encoding.mp4");
    let mut ff = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgra",
            "-video_size",
            &format!("{W}x{H}"),
            "-framerate",
            &FPS.to_string(),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "8",
            "-movflags",
            "+faststart",
        ])
        .arg(&staging)
        .stdin(Stdio::piped())
        .stderr(fs::File::create(out.join("ffmpeg.log"))?)
        .spawn()?;
    let mut pipe = BufWriter::with_capacity(1024 * 1024, ff.stdin.take().ok_or("ffmpeg stdin")?);
    let mut evidence = BufWriter::new(fs::File::create(out.join("events.jsonl"))?);
    let mut encoded = 0usize;
    let mut chapters = Vec::new();
    let mut audits = Vec::new();
    let mut fresh_raw = 0;
    let mut scored = 0;
    let mut matched = 0;
    let mut max_diff = 0f32;
    let mut branch_counts = [0usize; 4]; // evaluated, matches, supported, supported matches
    card(
        &mut c,
        "Two blurred RAW frames: what the trained model actually learns",
        &[
            format!(
                "{} complete recordings; each recording's entire day was held out of its model.",
                sources.len()
            ),
            if sam_geometry {
                "Left: RAW eyes, regenerated SAM contours and fitted ellipses."
            } else if branches {
                "Left: RAW eyes, fresh classical edge fits and observed gradients."
            } else {
                "Left: RAW eyes, archived conic fits and their segment samples."
            }
            .into(),
            "Middle: previous/current 32x24 inputs, image change and 8-second traces.".into(),
            "Right: commanded target, appearance predictions, scores and both 3D branches.".into(),
            "RAW remains visible during settling, missing detections and target transitions."
                .into(),
            "White target is an instruction, not independently measured fixation.".into(),
            if branches {
                "The model also chooses a conic branch; green rings show supported choices."
            } else {
                "The model predicts SCREEN direction. Physical conic sign remains unverified."
            }
            .into(),
            "Branch comparisons use a conditional RAW/target teacher, not measured 3D truth."
                .into(),
            "No archived segmentation or conic decision is a training input.".into(),
        ],
    );
    c.png(&out.join("opening.png"))?;
    emit(&mut c, &mut pipe, FPS * 8, &mut encoded)?;
    for (record, s) in sources.iter().enumerate() {
        let m = model(run, s.day, "two-frame", &report)?;
        let baseline = model(run, s.day, "current-only", &report)?;
        let bundle = BundleSource::open(Path::new(&s.archive))?;
        let (geometry, geometry_audit) = if branches {
            (
                HashMap::new(),
                json!({"role":if sam_geometry {"fresh SAM contours for eligible-pair exposures; unprepared exposures have no conic; no legacy predictions read"}else{"fresh deterministic classical RAW fits; no legacy predictions read"}}),
            )
        } else {
            shapes(&bundle)?
        };
        let eligible: HashMap<usize, usize> = s.eligible.iter().map(|p| (p[1], p[0])).collect();
        let mut groups = Vec::<Vec<usize>>::new();
        let mut by_event = HashMap::new();
        for (i, f) in s.frames.iter().enumerate() {
            if !(1..=2).contains(&n(&f["eye_id"])) {
                continue;
            }
            let k = key(f);
            let ek = (k.0, k.3);
            let index = *by_event.entry(ek).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            });
            groups[index].push(i);
        }
        let mut seen_keys = BTreeSet::new();
        let mut eyes: [Option<Eye>; 2] = [None, None];
        let mut trace = Trace::new();
        let mut previous_epoch = String::new();
        let mut previous_ns = 0;
        let mut elapsed = 0.;
        let mut source_span = 0.;
        let mut timing_residual = 0.;
        let mut record_scored = 0;
        let mut record_raw = 0;
        let mut record_geometry = 0;
        let chapter_start = encoded;
        card(
            &mut c,
            &format!("Recording {} of {}", record + 1, sources.len()),
            &[
                Path::new(&s.archive)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                format!(
                    "Held-out day {} | {} native eye frames | {} scored pairs",
                    s.day,
                    s.frames.len(),
                    s.eligible.len()
                ),
                "All RAW exposures are replayed; training admission is shown separately.".into(),
                "Cyan = eye 1 prediction; pink = eye 2; gray = one-frame baseline.".into(),
            ],
        );
        emit(&mut c, &mut pipe, FPS * 2, &mut encoded)?;
        let raw_start = encoded;
        for (gi, indices) in groups.iter().enumerate() {
            let first = &s.frames[indices[0]];
            let epoch = key(first).0;
            let ns = n(&first["timestamp_ns"]);
            if gi > 0 && (epoch != previous_epoch || ns <= previous_ns) {
                eyes = [None, None];
                trace.clear();
                card(
                    &mut c,
                    "Source clock / stream boundary",
                    &["Previous eye images and temporal inputs were cleared.".into()],
                );
                emit(&mut c, &mut pipe, FPS, &mut encoded)?;
            }
            let step = if gi > 0 && epoch == previous_epoch && ns > previous_ns {
                (ns - previous_ns) as f64 / 1e9
            } else {
                0.
            };
            elapsed += step;
            source_span += step;
            previous_epoch = epoch.clone();
            previous_ns = ns;
            if preview.is_some() && elapsed > 12. {
                break;
            }
            let mut host = 0;
            let mut new_ids = Vec::new();
            for &fi in indices {
                let f = &s.frames[fi];
                let k = key(f);
                if !seen_keys.insert(k.clone()) {
                    return Err("duplicate native source key in movie".into());
                }
                let eye = (n(&f["eye_id"]) - 1) as usize;
                host = host.max(n(&f["host_arrival_unix_ns"]));
                let bytes = bundle.read_range(
                    f["stream"].as_str().ok_or("RAW stream")?,
                    n(&f["offset"]),
                    n(&f["length"]) as usize,
                )?;
                let hash = data::digest(&bytes);
                let small = net::image(
                    &bytes,
                    n(&f["width"]) as usize,
                    n(&f["height"]) as usize,
                    n(&f["stride"]) as usize,
                )
                .ok();
                let old = eyes[eye].take();
                let previous = old.as_ref().filter(|e| data::pair_clock(&e.frame, f));
                let previous_small = previous.and_then(|e| e.small.clone());
                let prediction = previous_small
                    .as_ref()
                    .zip(small.as_ref())
                    .map(|(a, b)| m.predict(a, b))
                    .transpose()?;
                let control = small.as_ref().map(|p| baseline.predict(p, p)).transpose()?;
                let is_eligible = eligible.contains_key(&fi);
                let mut teacher = None;
                if is_eligible {
                    let prior = previous.ok_or("scored exposure lost previous native frame")?;
                    let p = prediction
                        .as_ref()
                        .ok_or("scored exposure lost model input")?;
                    let lookup = format!("{}:{hash}", prior.hash);
                    let known = expected
                        .get(&lookup)
                        .ok_or("movie pair missing held-out prediction")?;
                    for j in 0..9 {
                        max_diff = max_diff.max((p.target_scores[j] - known.0[j]).abs());
                    }
                    if p.conditional_branch_scores.is_some() != known.1.is_some() {
                        return Err("movie branch head mismatch".into());
                    }
                    if let (Some(a), Some(b)) = (p.conditional_branch_scores, known.1) {
                        for j in 0..2 {
                            max_diff = max_diff.max((a[j] - b[j]).abs());
                        }
                        if let Some(label) = known.2 {
                            let cls = usize::from(a[1] > a[0]);
                            let good = cls == label;
                            branch_counts[0] += 1;
                            branch_counts[1] += usize::from(good);
                            if a[cls] >= 0.8 {
                                branch_counts[2] += 1;
                                branch_counts[3] += usize::from(good);
                            }
                        }
                    }
                    teacher = known.2;
                    if max_diff > 1e-6 {
                        return Err("movie inference differs from held-out evaluation".into());
                    }
                    record_scored += 1;
                    scored += 1;
                    let target =
                        target(s, n(&f["host_arrival_unix_ns"])).ok_or("scored target missing")?;
                    matched += usize::from(active_class(p) == net::class(target.uv));
                }
                let mut shape = if branches {
                    let fit = if let Some(fit) = native_fits.get(&native::identity(&hash, f)) {
                        fit.clone()
                    } else if sam_geometry {
                        None
                    } else {
                        native::for_supervision(&native::unpack(&bytes, f)?, f)
                    };
                    fit.map(|fit| Shape::native(fit, f))
                } else {
                    geometry
                        .get(&k)
                        .filter(|s| {
                            s.origin == [n(&f["sensor_x"]) as u32, n(&f["sensor_y"]) as u32]
                                && s.size == [n(&f["width"]) as usize, n(&f["height"]) as usize]
                        })
                        .cloned()
                };
                if let Some((old_shape, new_shape)) =
                    previous.and_then(|e| e.shape.as_ref()).zip(shape.as_mut())
                {
                    let dot =
                        |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
                    if dot(old_shape.normals[0], new_shape.normals[1])
                        + dot(old_shape.normals[1], new_shape.normals[0])
                        > dot(old_shape.normals[0], new_shape.normals[0])
                            + dot(old_shape.normals[1], new_shape.normals[1])
                    {
                        new_shape.normals.swap(0, 1);
                        new_shape.centers.swap(0, 1);
                    }
                }
                record_geometry += usize::from(shape.is_some());
                let reason = if small.is_none() {
                    "Insufficient RAW contrast / no appearance prediction".into()
                } else if prediction.is_none() {
                    "Waiting for consecutive fresh frames of this eye".into()
                } else if is_eligible {
                    "Settled target; hidden thumbnails; fresh pair".into()
                } else if let Some(t) = target(s, n(&f["host_arrival_unix_ns"])) {
                    let age = n(&f["host_arrival_unix_ns"]).saturating_sub(t.start) as f64 / 1e9;
                    format!("Target age {age:.2}s: settling / transition / visibility gate")
                } else {
                    "No unique timed calibration target: not scored".into()
                };
                let preview = native_preview(&bytes, f)?;
                eyes[eye] = Some(Eye {
                    frame: f.clone(),
                    hash,
                    preview,
                    small,
                    previous_small,
                    prediction,
                    baseline: control,
                    shape,
                    eligible: is_eligible,
                    reason,
                    teacher,
                });
                record_raw += 1;
                fresh_raw += 1;
                new_ids.push(eye + 1);
            }
            let current_target = target(s, host);
            trace.push_back((
                elapsed,
                std::array::from_fn(|i| {
                    eyes[i]
                        .as_ref()
                        .and_then(|e| e.prediction.as_ref().map(|p| p.uv))
                }),
                current_target.map(|t| t.uv),
                std::array::from_fn(|i| eyes[i].as_ref().is_some_and(|e| e.eligible)),
            ));
            while trace.front().is_some_and(|p| p.0 < elapsed - 8.) {
                trace.pop_front();
            }
            render(
                &mut c,
                s,
                record + 1,
                sources.len(),
                elapsed,
                ns,
                host,
                &eyes,
                &trace,
            );
            if gi == 0 || gi == groups.len() / 2 {
                c.png(&out.join(format!(
                    "review-{:02}-{}.png",
                    record + 1,
                    if gi == 0 { "start" } else { "middle" }
                )))?;
            }
            let next_ns = groups
                .get(gi + 1)
                .filter(|next| key(&s.frames[next[0]]).0 == epoch)
                .map(|next| n(&s.frames[next[0]]["timestamp_ns"]));
            let dt = next_ns
                .filter(|next| *next > ns)
                .map(|next| (next - ns) as f64 / 1e9)
                .unwrap_or(0.1);
            // Native cadence, quantized at 25fps; every exposure group receives
            // at least one encoded frame. Long no-RAW gaps get a labelled slate.
            let desired = dt.min(1.) * FPS as f64 + timing_residual;
            let count = desired.round().max(1.) as usize;
            timing_residual = desired - count as f64;
            let video_start = encoded;
            emit(&mut c, &mut pipe, count, &mut encoded)?;
            serde_json::to_writer(
                &mut evidence,
                &json!({"recording":record+1,"archive":s.archive,"event":gi,"source_epoch":epoch,"source_ns":ns,"source_elapsed_seconds":elapsed,"host_arrival_ns":host,"video_frame_start":video_start,"video_frames":count,"new_eyes":new_ids,"target":current_target,"eyes":eyes.iter().map(|e|e.as_ref().map(frame_record)).collect::<Vec<_>>(),"held_out_day":s.day,"prediction_input":"only two current native ROI images; archived geometry is display-only"}),
            )?;
            evidence.write_all(b"\n")?;
            if dt > 1. {
                card(&mut c,"No recorded RAW exposure in this interval",&[format!("Source gap: {dt:.3} seconds. No held prediction counts as a new observation."),"The remaining gap is summarized by this two-second slate.".into()]);
                emit(&mut c, &mut pipe, FPS * 2, &mut encoded)?;
            }
        }
        if preview.is_none() && record_scored != s.eligible.len() {
            return Err("movie omitted a scored pair".into());
        }
        chapters.push(json!({"recording":record+1,"archive":s.archive,"start_frame":chapter_start,"raw_start_frame":raw_start,"end_frame":encoded,"start_seconds":chapter_start as f64/FPS as f64,"end_seconds":encoded as f64/FPS as f64}));
        audits.push(json!({"archive":s.archive,"raw_frames":record_raw,"scored_pairs":record_scored,"source_matched_conics":record_geometry,"source_span_seconds":source_span,"geometry":geometry_audit}));
        eprintln!(
            "MOVIE {}/{}: {} RAW, {} scored pairs, {} matched conics; video {:.1} min",
            record + 1,
            sources.len(),
            record_raw,
            record_scored,
            record_geometry,
            encoded as f64 / FPS as f64 / 60.
        );
        data::write(
            out.join("progress.json"),
            &json!({"completed_recordings":record+1,"recordings":sources.len(),"encoded_frames":encoded,"raw_frames":fresh_raw,"scored_pairs":scored,"wall_seconds":start.elapsed().as_secs_f64()}),
        )?;
    }
    let branch_report = &report["pooled_two_frame"]["conditional_branch"];
    let percent = |v: &Value| {
        v.as_f64()
            .map(|v| format!("{:.1}%", v * 100.))
            .unwrap_or("unavailable".into())
    };
    card(&mut c,"Corpus outcome: held-out predictions, coverage and failures",&[
        format!("{fresh_raw} RAW eye exposures replayed; {scored} scored pairs; {matched} nine-target matches."),
        format!("Two-frame target-class agreement: {:.1}%. Current-frame-only control: {:.1}%.",report["pooled_two_frame"]["target_class_accuracy"].as_f64().unwrap()*100.,report["pooled_current_only"]["target_class_accuracy"].as_f64().unwrap()*100.),
        format!("Noncentral left/right target agreement: {:.1}%; up/down: {:.1}%.",report["pooled_two_frame"]["axes"][0]["accuracy"].as_f64().unwrap()*100.,report["pooled_two_frame"]["axes"][1]["accuracy"].as_f64().unwrap()*100.),
        format!("Conditional branch agreement: {}; evaluated label coverage: {}.",percent(&branch_report["agreement"]),percent(&branch_report["evaluated_coverage"])),
        format!("Conditional branch control: {} current-only; {} constant training-majority.",percent(&report["pooled_current_only"]["conditional_branch"]["agreement"]),percent(&branch_report["training_majority_agreement"])),
        format!("At support >=80%: {} conditional agreement on {} labeled pairs.",percent(&branch_report["supported_agreement"]),branch_report["support_ge_0_8"]),
        "Displayed targets are weak labels. Camera-relative branch truth is unavailable.".into(),
        "Both hypotheses remain visible; missing geometry or weak support causes abstention.".into(),
        "Source hashes, model hashes, timelines, omissions and scores accompany this movie.".into()]);
    emit(&mut c, &mut pipe, FPS * 9, &mut encoded)?;
    c.png(&out.join("outcome.png"))?;
    pipe.flush()?;
    drop(pipe);
    if !ff.wait()?.success() {
        return Err("movie encoder failed; inspect ffmpeg.log".into());
    }
    evidence.flush()?;
    if branches && preview.is_none() {
        for (i, key) in [
            "evaluated_labels",
            "matches",
            "support_ge_0_8",
            "supported_matches",
        ]
        .iter()
        .enumerate()
        {
            if branch_report[key].as_u64() != Some(branch_counts[i] as u64) {
                return Err("movie branch counts differ from held-out evaluation".into());
            }
        }
    }
    if preview.is_none()
        && (scored != report["training_pairs"].as_u64().unwrap() as usize
            || matched
                != report["pooled_two_frame"]["target_class_correct"]
                    .as_u64()
                    .unwrap() as usize)
    {
        return Err("movie counts differ from held-out corpus evaluation".into());
    }
    let mut meta = ";FFMETADATA1\n".to_owned();
    for chapter in &chapters {
        let a = chapter["start_frame"].as_u64().unwrap() * 1000 / FPS as u64;
        let b = chapter["end_frame"].as_u64().unwrap() * 1000 / FPS as u64;
        meta.push_str(&format!(
            "[CHAPTER]\nTIMEBASE=1/1000\nSTART={a}\nEND={b}\ntitle=Recording {}\n",
            chapter["recording"]
        ));
    }
    fs::write(out.join("chapters.ffmeta"), meta)?;
    let video = out.join("calibration-sign-corpus.mp4");
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&staging)
        .arg("-i")
        .arg(out.join("chapters.ffmeta"))
        .args([
            "-map_metadata",
            "1",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
        ])
        .arg(&video)
        .status()?;
    if !status.success() {
        return Err("chapter mux failed".into());
    }
    fs::remove_file(staging)?;
    let receipt = json!({"schema":"buttercup-calibration-sign-movie-v1","source":source,"train_run":run,"training_results_sha256":data::digest(&fs::read(run.join("results.json"))?),"model_artifacts":report["model_artifacts"],"inventory":inventory,"preview":preview,"recordings":sources.len(),"raw_exposures":fresh_raw,"scored_pairs":scored,"target_class_matches":matched,"held_out_prediction_max_absolute_difference":max_diff,"encoded_frames":encoded,"fps":FPS,"duration_seconds":encoded as f64/FPS as f64,"dimensions":[W,H],"chapters":chapters,"audits":audits,"wall_seconds":start.elapsed().as_secs_f64(),"video_sha256":data::digest(&fs::read(&video)?),"limits":["RAW is phase-neutral 4x4 CFA-cell display; network inputs are separate 32x24 blurred images.","All exposures in the 47 qualifying recordings are shown; other recordings are listed with training timing exclusions.","Archive geometry is retrospective, source-matched, conditional and evaluation-only; neither model feature nor branch truth.","Per-event source intervals are rounded to 25fps, minimum one output frame; gaps over 1s receive a labeled slate.","No held prediction counts as new evidence; scores and target agreement are not measured gaze or calibrated probabilities."]});
    if boot::current_source(Path::new("."))? != source {
        return Err("source tree changed during movie generation".into());
    }
    let mut receipt = receipt;
    receipt["conditional_branch_counts"] = json!(branch_counts);
    receipt["conditional_branch_label_audit"] = report["branch_label_audit"].clone();
    receipt["geometry_provider"] = report["branch_label_audit"]["geometry_provider"].clone();
    if sam_geometry {
        receipt["limits"][2] = json!("SAM contours are regenerated from RAW for eligible-pair exposures; they provide conditional teacher labels and display geometry, never neural inputs. Unprepared exposures remain visible with no conic; no classical or archived fallback.");
    }
    if branches {
        receipt["limits"][2]=json!("Fresh classical RAW geometry and nominal camera/display assumptions form weak labels; no independent sign, human localization or scale truth is present.");
    }
    data::write(out.join("movie.json"), &receipt)?;
    eprintln!("MOVIE DONE {}", video.display());
    Ok(())
}
