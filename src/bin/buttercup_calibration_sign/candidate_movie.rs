//! Presentation-only excerpt of a completed experiment, with both conic
//! directions projected through the same saved calibration map for each eye.
//! This consumes historical diagnostics, never trains or promotes a model.
use super::{bootstrapability as boot, branch_labels::Map, data, native, Result};
#[path = "canvas.rs"]
mod canvas;
use canvas::*;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{Read, Write},
    path::Path,
    process::{Command, Stdio},
};

const FPS: usize = 25;
const SLOW: usize = 3;
const SCREEN_PADDING: f64 = 0.20;
const SCREEN_ASPECT: f64 = 16. / 9.;
const COLORS: [[f64; 3]; 2] = [CYAN, PINK];

#[derive(Clone, Serialize)]
struct Eye {
    raw_sha256: String,
    sequence: u64,
    family: String,
    map: Map,
    near_best_maps: usize,
    normals: [[f64; 3]; 2],
    uv: [[f64; 2]; 2],
}
#[derive(Clone, Serialize)]
struct Event {
    start: usize,
    end: usize,
    archive: String,
    source_seconds: f64,
    target_id: String,
    target: [f64; 2],
    eyes: [Eye; 2],
}
fn integer(v: &Value) -> usize {
    data::num(v).unwrap_or(0) as usize
}
fn eye(
    row: &Value,
    index: usize,
    maps: &HashMap<String, Vec<Map>>,
    fits: &HashMap<String, native::NativeFit>,
) -> Result<Option<Eye>> {
    let e = &row["eyes"][index];
    if e["geometry"]["admissible"] != true {
        return Ok(None);
    }
    let family = format!(
        "{}:{}:{}",
        row["archive"].as_str().ok_or("archive")?,
        e["frame"]["eye_id"],
        e["frame"]["source_clock"]["source_key"]["stream_epoch"]
    );
    let Some(options) = maps.get(&family).filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    let map = options
        .iter()
        .min_by(|a, b| a.rmse.total_cmp(&b.rmse))
        .unwrap();
    let hash = e["raw_sha256"].as_str().ok_or("RAW identity")?;
    let fit = fits
        .get(&native::identity(hash, &e["frame"]))
        .ok_or("unmatched native fit")?;
    let normals: [[f64; 3]; 2] = serde_json::from_value(e["geometry"]["normals"].clone())?;
    let same = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
    if !fit.admissible()
        || !((same(normals[0], fit.normals[0]) && same(normals[1], fit.normals[1]))
            || (same(normals[0], fit.normals[1]) && same(normals[1], fit.normals[0])))
    {
        return Err("movie normals differ from source-matched native geometry".into());
    }
    let uv = normals.map(|n| map.project([-n[0].atan2(n[2]), n[1].clamp(-1., 1.).asin()]));
    if uv.iter().flatten().any(|v| !v.is_finite()) {
        return Err("nonfinite projected candidate".into());
    }
    Ok(Some(Eye {
        raw_sha256: hash.into(),
        sequence: integer(&e["frame"]["sequence"]) as u64,
        family,
        map: map.clone(),
        near_best_maps: options.len(),
        normals,
        uv,
    }))
}

fn marker(c: &mut Canvas, p: [f64; 2], eye: usize, branch: usize, radius: f64) {
    c.dot(p[0], p[1], radius, COLORS[eye], branch == 0);
    if branch == 1 {
        c.dot(p[0], p[1], radius + 3., COLORS[eye], false);
    }
}

#[derive(Serialize)]
struct CursorPlacement {
    position_uv: [f64; 2],
    boundary_clamped: bool,
    offscreen_distance_screen_heights: f64,
}

fn cursor_placement(uv: [f64; 2]) -> CursorPlacement {
    let boundary_clamped = uv
        .iter()
        .any(|v| !(-SCREEN_PADDING..=1. + SCREEN_PADDING).contains(v));
    let position_uv = if boundary_clamped {
        let d = [uv[0] - 0.5, uv[1] - 0.5];
        let radius = 0.5 + SCREEN_PADDING;
        let k = (radius / d[0].abs().max(1e-12)).min(radius / d[1].abs().max(1e-12));
        [0.5 + k * d[0], 0.5 + k * d[1]]
    } else {
        uv
    };
    // Shortest distance to the actual display rectangle, expressed in screen
    // heights. Horizontal normalized coordinates measure a full screen width.
    let dx = (uv[0] - uv[0].clamp(0., 1.)) * SCREEN_ASPECT;
    let dy = uv[1] - uv[1].clamp(0., 1.);
    CursorPlacement {
        position_uv,
        boundary_clamped,
        offscreen_distance_screen_heights: dx.hypot(dy),
    }
}

fn render(c: &mut Canvas, pixels: &[u8], e: &Event, clip: usize, total: usize, bounds: [f64; 4]) {
    c.clear();
    c.text(
        22.,
        42.,
        30.,
        WHITE,
        "Two possible gaze directions from each eye",
    );
    c.text(
        22.,
        73.,
        17.,
        MUTED,
        &format!(
            "Excerpt {}/{} | {} | source +{:.2}s | 3x slower",
            clip + 1,
            total,
            Path::new(&e.archive).file_name().unwrap().to_string_lossy(),
            e.source_seconds
        ),
    );
    for i in 0..2 {
        let y = 111. + i as f64 * 450.;
        c.clipped(20., y, 600., 400., |c| {
            c.image(
                pixels,
                1240,
                400,
                if i == 0 { 0. } else { -620. },
                y,
                1240.,
                400.,
            );
        });
        c.shade(20., y + 367., 600., 33., 0.9);
        c.text(
            30.,
            y + 390.,
            17.,
            COLORS[i],
            &format!(
                "Eye {} | RAW exposure {} | same ellipse, two directions",
                i + 1,
                e.eyes[i].sequence
            ),
        );
    }
    c.text(
        20.,
        1000.,
        18.,
        WHITE,
        "White curve: fitted ellipse. Pink points: SAM contour predictions.",
    );
    c.text(
        20.,
        1032.,
        17.,
        MUTED,
        "Native source identities and both original 3D normals are retained.",
    );

    let rect = [700., 147., 1152., 648.];
    c.rect(rect[0], rect[1], rect[2], rect[3], [0.065, 0.09, 0.12]);
    c.text(
        700.,
        122.,
        24.,
        WHITE,
        "Screen projection — both alternatives, equally visible",
    );
    let project = |p: [f64; 2]| {
        [
            rect[0] + (p[0] + SCREEN_PADDING) / (1. + 2. * SCREEN_PADDING) * rect[2],
            rect[1] + (p[1] + SCREEN_PADDING) / (1. + 2. * SCREEN_PADDING) * rect[3],
        ]
    };
    let top_left = project([0., 0.]);
    let bottom_right = project([1., 1.]);
    c.rect(
        top_left[0],
        top_left[1],
        bottom_right[0] - top_left[0],
        bottom_right[1] - top_left[1],
        [0.12, 0.17, 0.22],
    );
    c.path(
        &[
            top_left,
            project([1., 0.]),
            bottom_right,
            project([0., 1.]),
            top_left,
        ],
        2.,
        MUTED,
    );
    c.path(
        &[
            [rect[0], rect[1]],
            [rect[0] + rect[2], rect[1]],
            [rect[0] + rect[2], rect[1] + rect[3]],
            [rect[0], rect[1] + rect[3]],
            [rect[0], rect[1]],
        ],
        1.,
        [0.24, 0.30, 0.36],
    );
    c.text(top_left[0] + 12., top_left[1] + 24., 16., MUTED, "DISPLAY");
    c.text(
        rect[0] + 18.,
        rect[1] + 26.,
        17.,
        MUTED,
        "20% margin on each side",
    );
    for t in [0.1, 0.5, 0.9] {
        c.line(project([t, 0.]), project([t, 1.]), 1., [0.20, 0.25, 0.30]);
        c.line(project([0., t]), project([1., t]), 1., [0.20, 0.25, 0.30]);
    }
    let target = project(e.target);
    c.cross(target[0], target[1], 16., WHITE);
    c.dot(target[0], target[1], 23., WHITE, false);
    for i in 0..2 {
        for b in 0..2 {
            let uv = e.eyes[i].uv[b];
            let placement = cursor_placement(uv);
            let q = placement.position_uv;
            let p = project(q);
            if placement.boundary_clamped {
                let d = [q[0] - 0.5, q[1] - 0.5];
                let inner = project([q[0] - 0.07 * d[0], q[1] - 0.07 * d[1]]);
                c.arrow(inner, p, COLORS[i]);
            }
            marker(c, p, i, b, if b == 0 { 9. } else { 11. });
            let label = format!(
                "{}{}{}",
                i + 1,
                if b == 0 { 'A' } else { 'B' },
                if placement.boundary_clamped {
                    format!(" {:.2}H off", placement.offscreen_distance_screen_heights)
                } else {
                    String::new()
                },
            );
            // Label the eyes on opposite sides of their cursors so nearby
            // predictions remain readable, with separate boundary lanes.
            let text_width = label.chars().count() as f64 * 10.8;
            let label_x = if i == 0 {
                p[0] - 16. - text_width
            } else {
                p[0] + 16.
            }
            .clamp(rect[0] + 10., rect[0] + rect[2] - text_width - 10.);
            let dy = if p[1] > rect[1] + rect[3] - 65. {
                -22. - i as f64 * 26.
            } else if p[1] < rect[1] + 55. {
                30. + i as f64 * 26.
            } else if i == 0 {
                -18.
            } else {
                29.
            };
            c.text(label_x, p[1] + dy, 19., COLORS[i], &label);
        }
    }
    c.text(
        700.,
        826.,
        18.,
        WHITE,
        "White ring = target. H = screen height; off distance is measured from the display edge.",
    );
    c.text(
        700.,
        855.,
        18.,
        MUTED,
        "A: filled cursor   B: outlined cursor   Cyan: eye 1   Pink: eye 2",
    );
    for i in 0..2 {
        c.text(
            700.,
            902. + i as f64 * 35.,
            18.,
            COLORS[i],
            &format!(
                "Eye {}   A ({:+.2}, {:+.2})   B ({:+.2}, {:+.2})",
                i + 1,
                e.eyes[i].uv[0][0],
                e.eyes[i].uv[0][1],
                e.eyes[i].uv[1][0],
                e.eyes[i].uv[1][1]
            ),
        );
    }
    // Fixed overview includes actual out-of-screen endpoints, never clamped.
    let mini = [1460., 880., 390., 165.];
    let scale =
        (mini[2] / (16. * (bounds[2] - bounds[0]))).min(mini[3] / (9. * (bounds[3] - bounds[1])));
    let p = |uv: [f64; 2]| {
        [
            mini[0] + mini[2] / 2. + (uv[0] - (bounds[0] + bounds[2]) / 2.) * 16. * scale,
            mini[1] + mini[3] / 2. + (uv[1] - (bounds[1] + bounds[3]) / 2.) * 9. * scale,
        ]
    };
    c.rect(mini[0], mini[1], mini[2], mini[3], [0.08, 0.10, 0.13]);
    let a = p([0., 0.]);
    let z = p([1., 1.]);
    c.rect(a[0], a[1], z[0] - a[0], z[1] - a[1], [0.20, 0.24, 0.29]);
    let t = p(e.target);
    c.cross(t[0], t[1], 5., WHITE);
    for i in 0..2 {
        for b in 0..2 {
            marker(c, p(e.eyes[i].uv[b]), i, b, 4.);
        }
    }
    c.text(
        1460.,
        1069.,
        14.,
        MUTED,
        "Overview: actual endpoints; fixed scale across all excerpts",
    );
    c.text(
        700.,
        982.,
        16.,
        MUTED,
        "Both alternatives use the same saved per-eye calibration map.",
    );
    c.text(
        700.,
        1008.,
        16.,
        MUTED,
        "Best session fit; nominal camera optics. This is retrospective.",
    );
    c.text(
        700.,
        1034.,
        16.,
        MUTED,
        "Coordinates: screen top-left (0,0), bottom-right (1,1).",
    );
    c.text(
        700.,
        1061.,
        16.,
        MUTED,
        "Selected for available geometry, not independent gaze accuracy.",
    );
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 4 {
        return Err("candidate-movie TRAIN_RUN EXISTING_MOVIE_DIR NEW_OUT".into());
    }
    let run = Path::new(&args[1]);
    let movie = Path::new(&args[2]);
    let report: Value = serde_json::from_slice(&fs::read(run.join("results.json"))?)?;
    let receipt: Value = serde_json::from_slice(&fs::read(movie.join("movie.json"))?)?;
    if receipt["source"] != report["source"]
        || receipt["training_results_sha256"] != data::digest(&fs::read(run.join("results.json"))?)
    {
        return Err("historical movie/training provenance mismatch".into());
    }
    let video = movie.join("calibration-sign-corpus.mp4");
    if receipt["video_sha256"] != data::digest(&fs::read(&video)?) {
        return Err("source movie hash changed".into());
    }
    let conic_bytes = fs::read(run.join("native-conics.jsonl"))?;
    if report["branch_label_audit"]["native_conics_sha256"] != data::digest(&conic_bytes) {
        return Err("native conic hash changed".into());
    }
    let mut fits = HashMap::new();
    for line in std::str::from_utf8(&conic_bytes)?.lines() {
        let r: Value = serde_json::from_str(line)?;
        if let Some(f) = serde_json::from_value::<Option<native::NativeFit>>(r["fit"].clone())? {
            fits.insert(
                native::identity(
                    r["source"]["raw_sha256"].as_str().ok_or("RAW hash")?,
                    &r["source"]["frame"],
                ),
                f,
            );
        }
    }
    let audit = &report["branch_label_audit"];
    let mut maps = HashMap::new();
    for f in audit["families"].as_array().ok_or("map families")? {
        maps.insert(
            f["family"].as_str().ok_or("family key")?.to_owned(),
            serde_json::from_value::<Vec<Map>>(f["near_best_maps"].clone())?,
        );
    }
    let mut runs: Vec<Vec<Event>> = Vec::new();
    let mut active: Vec<Event> = Vec::new();
    let bytes = fs::read(movie.join("events.jsonl"))?;
    for line in bytes.split(|b| *b == b'\n').filter(|v| !v.is_empty()) {
        let row: Value = serde_json::from_slice(line)?;
        let pair = [eye(&row, 0, &maps, &fits)?, eye(&row, 1, &maps, &fits)?];
        let next = match pair {
            [Some(a), Some(b)] if !row["target"].is_null() => Some(Event {
                start: integer(&row["video_frame_start"]),
                end: integer(&row["video_frame_start"]) + integer(&row["video_frames"]),
                archive: row["archive"].as_str().ok_or("archive")?.into(),
                source_seconds: row["source_elapsed_seconds"]
                    .as_f64()
                    .ok_or("source time")?,
                target_id: row["target"]["id"].as_str().ok_or("target id")?.into(),
                target: serde_json::from_value(row["target"]["uv"].clone())?,
                eyes: [a, b],
            }),
            _ => None,
        };
        let continuous = next.as_ref().zip(active.last()).is_some_and(|(n, p)| {
            n.start == p.end
                && n.archive == p.archive
                && n.target_id == p.target_id
                && n.eyes[0].family == p.eyes[0].family
                && n.eyes[1].family == p.eyes[1].family
        });
        if !continuous && !active.is_empty() {
            runs.push(std::mem::take(&mut active));
        }
        if let Some(n) = next {
            active.push(n);
        }
    }
    if !active.is_empty() {
        runs.push(active);
    }
    // Longest continuous usable interval for each target; first three archives
    // with at least two such targets. No target error or learned score selection.
    let mut groups: BTreeMap<String, BTreeMap<String, Vec<Event>>> = BTreeMap::new();
    let length = |r: &Vec<Event>| r.last().unwrap().end - r[0].start;
    for r in runs.into_iter().filter(|r| length(r) >= 20) {
        let entry = groups
            .entry(r[0].archive.clone())
            .or_default()
            .entry(r[0].target_id.clone())
            .or_default();
        if entry.is_empty() || length(&r) > length(entry) {
            *entry = r;
        }
    }
    let mut clips = Vec::new();
    for (_, targets) in groups.into_iter().filter(|(_, t)| t.len() >= 2).take(3) {
        let mut values: Vec<_> = targets.into_values().collect();
        values.sort_by_key(|r| r[0].start);
        let choices = if values.len() > 3 {
            vec![0, values.len() / 2, values.len() - 1]
        } else {
            (0..values.len()).collect()
        };
        for i in choices {
            clips.push(values[i].clone());
        }
    }
    if clips.is_empty() {
        return Err("no continuous two-eye candidate excerpts".into());
    }
    let out = data::output(&args[3])?;
    let mut bounds = [0f64, 0., 1., 1.];
    for r in &clips {
        for e in r {
            for eye in &e.eyes {
                for uv in eye.uv {
                    for a in 0..2 {
                        bounds[a] = bounds[a].min(uv[a]);
                        bounds[a + 2] = bounds[a + 2].max(uv[a]);
                    }
                }
            }
        }
    }
    for a in 0..2 {
        let pad = (bounds[a + 2] - bounds[a]) * 0.1;
        bounds[a] -= pad;
        bounds[a + 2] += pad;
    }
    let mut encoder = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgra",
            "-video_size",
            "1920x1080",
            "-framerate",
            "25",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "8",
            "-movflags",
            "+faststart",
        ])
        .arg(out.join("two-candidates-short.mp4"))
        .stdin(Stdio::piped())
        .stderr(fs::File::create(out.join("encode.log"))?)
        .spawn()?;
    let mut sink = encoder.stdin.take().ok_or("encoder stdin")?;
    let mut c = Canvas::new(1920, 1080)?;
    let mut count = 0usize;
    let mut records = Vec::new();
    let mut events = fs::File::create(out.join("projected-events.jsonl"))?;
    for (ci, clip) in clips.iter().enumerate() {
        let first = clip[0].start;
        let end = clip.last().unwrap().end.min(first + 50);
        c.clear();
        c.text(
            100.,
            230.,
            42.,
            WHITE,
            &format!("Excerpt {} of {}", ci + 1, clips.len()),
        );
        c.text(
            100.,
            305.,
            28.,
            WHITE,
            "Two projections per eye. Filled A / outlined B.",
        );
        c.text(
            100.,
            360.,
            24.,
            MUTED,
            "Cyan: eye 1. Pink: eye 2. White ring: displayed target.",
        );
        c.text(
            100.,
            415.,
            24.,
            MUTED,
            "3x slower. Hard cuts separate recordings and fixation intervals.",
        );
        c.text(
            100.,
            470.,
            24.,
            MUTED,
            "The same saved calibration map projects both alternatives for each eye.",
        );
        c.text(
            100.,
            525.,
            24.,
            MUTED,
            "20% margin on each side. Farther cursors stop at its boundary.",
        );
        c.text(
            100.,
            580.,
            24.,
            MUTED,
            "H = screen height. Labels give total distance beyond the actual display.",
        );
        for _ in 0..25 {
            sink.write_all(c.bytes())?;
            count += 1;
        }
        let output_start = count;
        let mut decoder = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-ss",
                &format!("{:.6}", first as f64 / FPS as f64),
                "-i",
            ])
            .arg(&video)
            .args([
                "-frames:v",
                &(end - first).to_string(),
                "-vf",
                "crop=1240:400:0:132",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgra",
                "pipe:1",
            ])
            .stdout(Stdio::piped())
            .stderr(fs::File::create(out.join(format!("decode-{ci}.log")))?)
            .spawn()?;
        let mut stream = decoder.stdout.take().ok_or("decoder stdout")?;
        let mut pixels = vec![0; 1240 * 400 * 4];
        let mut idx = 0;
        for frame in first..end {
            stream.read_exact(&mut pixels)?;
            while idx + 1 < clip.len() && clip[idx].end <= frame {
                idx += 1;
            }
            let e = &clip[idx];
            if !(e.start <= frame && frame < e.end) {
                return Err("source movie/event alignment gap".into());
            }
            render(&mut c, &pixels, e, ci, clips.len(), bounds);
            if frame == first {
                c.png(&out.join(format!("excerpt-{:02}.png", ci + 1)))?;
            }
            serde_json::to_writer(
                &mut events,
                &json!({"source_movie_frame":frame,"output_frame_start":count,"output_frames":SLOW,"event":e,
                    "cursor_placements":e.eyes.iter().map(|eye|eye.uv.map(cursor_placement)).collect::<Vec<_>>()}),
            )?;
            events.write_all(b"\n")?;
            for _ in 0..SLOW {
                sink.write_all(c.bytes())?;
                count += 1;
            }
        }
        if !decoder.wait()?.success() {
            return Err("source movie decoding failed".into());
        }
        records.push(json!({"archive":clip[0].archive,"target_id":clip[0].target_id,"input_frames":[first,end],"output_frames":[output_start,count],"source_start_seconds":clip[0].source_seconds,"maps":clip[0].eyes.iter().map(|e|json!({"family":e.family,"map":e.map,"near_best_maps":e.near_best_maps})).collect::<Vec<_>>()}));
        eprintln!(
            "candidate excerpt {}/{}: {} original frames",
            ci + 1,
            clips.len(),
            end - first
        );
    }
    drop(sink);
    if !encoder.wait()?.success() {
        return Err("candidate movie encoding failed".into());
    }
    data::write(
        out.join("receipt.json"),
        &json!({
            "schema":"buttercup-two-candidate-movie-v1","render_source":boot::current_source(Path::new("."))?,"historical_training_source":report["source"],
            "role":"presentation-only historical diagnostic; no training, model promotion or new accuracy claim",
            "input_movie_sha256":receipt["video_sha256"],"input_events_sha256":data::digest(&bytes),"native_conics_sha256":audit["native_conics_sha256"],"training_results_sha256":receipt["training_results_sha256"],
            "projection":"Each original normal n -> (-atan2(nx,nz), asin(ny)); both candidates use the identical fixed saved lowest-RMSE map for that recording/eye/source epoch. Other near-best maps are not uncertainty-calibrated.",
            "selection":"First three archives with at least two targets with >=20 continuous movie frames where both eyes have admissible original geometry and saved maps. Longest run per target; first/middle/last target; at most 50 frames each. No target error or model confidence selection.",
            "slow_factor":SLOW,"fps":FPS,"frames":count,"seconds":count as f64/FPS as f64,"overview_bounds":bounds,"clips":records,
            "padding_each_side_fraction":SCREEN_PADDING,"main_view_bounds":[[-SCREEN_PADDING,-SCREEN_PADDING],[1.+SCREEN_PADDING,1.+SCREEN_PADDING]],
            "offscreen_distance":"Shortest Euclidean distance to the actual [0,1] display rectangle in screen heights H: hypot((u-clamp(u,0,1))*16/9, v-clamp(v,0,1)). Includes the 20% margin; not distance from the padded border. Far endpoints clip along a ray from display center to the padded boundary.",
            "video_sha256":data::digest(&fs::read(out.join("two-candidates-short.mp4"))?),
            "limits":["Retrospective session-fitted screen mapping and nominal optics; not independent gaze accuracy.","Endpoints within 20% padding stay at actual positions. Farther endpoints show boundary arrows and total off-screen distance in screen heights. Numerical coordinates and fixed-scale overview retain actual endpoints; no measured physical distance or visual angle is implied.","Samples show both available conics; no coverage estimate or invented missing directions.","No custom model is trained or executed; historical RAW video, conics and calibration maps are displayed."]
        }),
    )?;
    eprintln!(
        "CANDIDATE MOVIE DONE: {:.2}s, {}",
        count as f64 / FPS as f64,
        out.display()
    );
    Ok(())
}
