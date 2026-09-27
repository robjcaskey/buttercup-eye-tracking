//! Source-timed, source-verified review video. This renders completed runs;
//! it neither re-runs clustering nor feeds the commanded target to a model.
use super::{curve, json, motion, point, raw_preview, read_rows, BundleSource, Error, Value};
use buttercup_eye_tracking::raw10;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

const W: usize = 1296;
const H: usize = 864;
const WHITE: u32 = 0xe3e9f0;

fn line(pixels: &mut [u32], w: usize, h: usize, a: [f64; 2], b: [f64; 2], color: u32) {
    let steps = ((b[0] - a[0]).hypot(b[1] - a[1]).ceil() as usize).clamp(1, 800);
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        point(
            pixels,
            w,
            h,
            (a[0] * (1. - t) + b[0] * t, a[1] * (1. - t) + b[1] * t),
            color,
            0,
        );
    }
}
fn circle(pixels: &mut [u32], w: usize, h: usize, p: [f64; 2], color: u32) {
    for i in 0..24 {
        let t = i as f64 * std::f64::consts::TAU / 24.;
        point(
            pixels,
            w,
            h,
            (p[0] + 3. * t.cos(), p[1] + 3. * t.sin()),
            color,
            0,
        );
    }
}
fn xy(p: &Value) -> [f64; 2] {
    [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()]
}
fn native(raw: &[u16], r: &Value) -> Vec<u32> {
    let s = &r["source"];
    raw_preview::color_preview(
        raw,
        s["width"].as_u64().unwrap() as usize,
        s["height"].as_u64().unwrap() as usize,
        s["sensor_x"].as_u64().unwrap() as u32,
        s["sensor_y"].as_u64().unwrap() as u32,
        100,
        None,
    )
}

fn overlay(image: &mut [u32], r: &Value, candidate: bool, motion_only: bool) {
    let s = &r["source"];
    let (w, h) = (
        s["width"].as_u64().unwrap() as usize,
        s["height"].as_u64().unwrap() as usize,
    );
    let origin = [
        s["sensor_x"].as_f64().unwrap(),
        s["sensor_y"].as_f64().unwrap(),
    ];
    for p in r["tensor_points"].as_array().unwrap() {
        let q = xy(&p["current_sensor"]);
        point(
            image,
            w,
            h,
            (q[0] - origin[0], q[1] - origin[1]),
            0x708090,
            0,
        );
    }
    for (ci, c) in r["tensor_clusters"].as_array().unwrap().iter().enumerate() {
        let color = if motion_only {
            if c["persistent_nodes"].as_u64().unwrap_or(0) >= 4
                && c["persistent_edges"].as_u64().unwrap_or(0) >= 2
            {
                [0x63beff, 0xffb050, 0xd888ff, 0x7affe0][ci % 4]
            } else {
                0x84909d
            }
        } else if candidate {
            if c["motion_role"] == "carrier_compatible" {
                0x4aff80
            } else {
                [0xffb050, 0x63beff, 0xd888ff, 0xffb050][ci % 4]
            }
        } else {
            [0x63beff, 0xffb050, 0xd888ff, 0x7affe0][ci % 4]
        };
        for m in c["members"].as_array().unwrap() {
            let (p, q) = (xy(&m["previous_sensor"]), xy(&m["current_sensor"]));
            let a = [p[0] - origin[0], p[1] - origin[1]];
            let b = [q[0] - origin[0], q[1] - origin[1]];
            line(image, w, h, a, b, color);
            circle(image, w, h, b, color);
        }
        if motion_only && c["shared_origin_sensor"].is_array() {
            let o = xy(&c["shared_origin_sensor"]);
            let p = [o[0] - origin[0], o[1] - origin[1]];
            line(image, w, h, [p[0] - 5., p[1]], [p[0] + 5., p[1]], color);
            line(image, w, h, [p[0], p[1] - 5.], [p[0], p[1] + 5.], color);
        }
    }
    if !candidate {
        if !r["fit"].is_null() {
            let e = &r["fit"]["ellipse"];
            let c = xy(&e["center"]);
            curve(
                image,
                w,
                h,
                motion::IrisEllipseSeed {
                    center: (c[0], c[1]),
                    major_radius: e["major"].as_f64().unwrap(),
                    minor_radius: e["minor"].as_f64().unwrap(),
                    angle: e["angle"].as_f64().unwrap(),
                },
                0xff4050,
            );
        }
        for p in r["tensor_points"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p["normal_flow"] == true)
        {
            let q = xy(&p["current_sensor"]);
            let (x, y) = (q[0] - origin[0], q[1] - origin[1]);
            let corners = [
                [x, y - 4.],
                [x + 4., y],
                [x, y + 4.],
                [x - 4., y],
                [x, y - 4.],
            ];
            for pair in corners.windows(2) {
                line(image, w, h, pair[0], pair[1], 0xff60bc);
            }
        }
    }
}

fn digit(pixels: &mut [u32], x: usize, y: usize, d: usize) {
    let rows = match d {
        1 => [4, 12, 4, 4, 4, 4, 14],
        2 => [14, 17, 1, 2, 4, 8, 31],
        _ => [0; 7],
    };
    for (dy, bits) in rows.into_iter().enumerate() {
        for dx in 0..5 {
            if bits & (1 << (4 - dx)) != 0 {
                for yy in 0..3 {
                    for xx in 0..3 {
                        pixels[(y + 3 * dy + yy) * W + x + 3 * dx + xx] = WHITE;
                    }
                }
            }
        }
    }
}

fn filters(motion_only: bool) -> String {
    let texts = if motion_only {
        vec![
        (12,12,26,"MOTION GROUPS WITHOUT AN IRIS DETECTOR"),
        (12,48,18,"Same native RAW exposures. Removing iris priors also changes feature tracks; this is an ablation."),
        (12,77,18,"Right panel - no iris detector, ellipse seed, inferred iris region, anatomical naming or ellipse fit."),
        (12,106,18,"Colors = current groups with temporal support. Gray = weak groups. Colors are not tissue identities."),
        (12,134,16,"Crosses = inferred image-space origins. Persistent groups can still be reflections or eyelids."),
        (10,160,16,"EYE 1 - RAW"),(439,160,16,"EXISTING TRACKER + IRIS PRIOR"),(868,160,16,"MOTION ONLY - NO IRIS PRIOR"),
        (10,480,16,"EYE 2 - RAW"),(439,480,16,"EXISTING TRACKER + IRIS PRIOR"),(868,480,16,"MOTION ONLY - NO IRIS PRIOR"),
        (12,806,17,"Existing panel - red curve is an ellipse candidate; pink diamonds constrain only edge-normal motion."),
        (12,836,16,"No colored points = insufficient current temporal support. Playback follows recorded timing (~6.4 frames/s)."),
        (1115,124,14,"TARGET COMMAND"),(1115,144,12,"TIMING APPROXIMATE"),(1008,134,12,"SESSION"),
    ]
    } else {
        vec![
        (12,12,26,"CLUSTERING REVIEW - TWO RECORDED TARGETING SESSIONS"),
        (12,48,18,"Same RAW and identical tracked points. Experimental grouping; not a live viewer recording."),
        (12,77,18,"Common-point prediction error  1.27 to 1.13 px (session 1); 1.21 to 1.08 px (session 2)."),
        (12,106,18,"Experimental green = surrounding-motion compatible. Other colors = differential motion."),
        (12,134,16,"Motion groups are not confirmed sclera / iris / skin labels. No experimental ellipse solve."),
        (10,160,16,"EYE 1 - RAW"),(439,160,16,"EXISTING CLUSTERS + FIT"),(868,160,16,"EXPERIMENTAL MOTION GROUPS"),
        (10,480,16,"EYE 2 - RAW"),(439,480,16,"EXISTING CLUSTERS + FIT"),(868,480,16,"EXPERIMENTAL MOTION GROUPS"),
        (12,806,17,"Existing panel  red curve = ellipse candidate; pink diamonds = one-direction edge flow."),
        (12,836,16,"No experimental colors = insufficient current motion support. Playback follows recorded timing (~6.4 frames/s)."),
        (1115,124,14,"TARGET COMMAND"),(1115,144,12,"TIMING APPROXIMATE"),(1008,134,12,"SESSION"),
    ]
    };
    texts.iter().map(|(x,y,size,s)|format!("drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{s}':x={x}:y={y}:fontsize={size}:fontcolor=0xe3e9f0")).collect::<Vec<_>>().join(",")
}

pub fn run(args: &[String]) -> Result<(), Error> {
    if args.len() != 9 {
        return Err(
            "usage: --demo NEW_VIDEO BUNDLE1 BASELINE1 PARTITION1 BUNDLE2 BASELINE2 PARTITION2"
                .into(),
        );
    }
    let motion_only = args[1] == "--motion-demo";
    let out = Path::new(&args[2]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output needs parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("video must be a new checked runtime output".into());
    }
    let log = fs::File::create(out.with_extension("encode.log"))?;
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "warning",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "1296x864",
            "-framerate",
            "30",
            "-i",
            "pipe:0",
            "-an",
            "-vf",
        ])
        .arg(filters(motion_only))
        .args([
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-threads",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log))
        .spawn()?;
    let render = (|| -> Result<Value, Error> {
        let mut pipe = child.stdin.take().ok_or("encoder stdin unavailable")?;
        let mut total_frames = 0usize;
        let mut manifest = Vec::new();
        for (session, paths) in args[3..].chunks_exact(3).enumerate() {
            let bundle = BundleSource::open(Path::new(&paths[0]))?;
            let a = read_rows(&Path::new(&paths[1]).join("frames.jsonl"))?;
            let b = read_rows(&Path::new(&paths[2]).join("frames.jsonl"))?;
            if a.len() != b.len()
                || a.iter().zip(&b).any(|(a, b)| {
                    a["source"] != b["source"]
                        || a["raw_sha256"] != b["raw_sha256"]
                        || (!motion_only && a["tensor_points"] != b["tensor_points"])
                })
            {
                return Err(
                    "demo inputs must have identical RAW identities and, for partition comparisons, identical point correspondences"
                        .into(),
                );
            }
            if motion_only
                && b.iter().any(|r| {
                    r["motion_only"] != true
                        || !r["seed"].is_null()
                        || !r["fit"].is_null()
                        || !r["semantic_iris"].is_null()
                        || !r["nested_limbus"].is_null()
                        || r["iris_layer_identified"] == true
                        || r["relation_state"]["selector_calls"] != 0
                })
            {
                return Err(
                    "motion-only review requires audited anatomy-free candidate rows".into(),
                );
            }
            let mut pairs = BTreeMap::<u64, [Option<usize>; 2]>::new();
            for (i, r) in a.iter().enumerate() {
                let eye = r["source"]["eye_id"].as_u64().unwrap() as usize - 1;
                let stamp = r["source"]["timestamp_ns"].as_u64().unwrap();
                if pairs.entry(stamp).or_insert([None, None])[eye]
                    .replace(i)
                    .is_some()
                {
                    return Err("duplicate eye source timestamp".into());
                }
            }
            let pairs = pairs.into_iter().collect::<Vec<_>>();
            let start = pairs.first().ok_or("no source pairs")?.0;
            let video_start = total_frames;
            for (index, (stamp, ids)) in pairs.iter().enumerate() {
                let mut pixels = vec![0x121b25u32; W * H];
                let mut sources = Vec::new();
                for eye in 0..2 {
                    let id = ids[eye]
                        .ok_or("missing matched eye source; do not hold another exposure")?;
                    let r = &a[id];
                    let s = &r["source"];
                    let u = |k: &str| s[k].as_u64().ok_or("missing source metadata");
                    if u("width")? != 420 || u("height")? != 280 {
                        return Err(
                            "review layout requires native 420x280; no implicit resize".into()
                        );
                    }
                    let packed = bundle.read_range(
                        s["stream"].as_str().ok_or("source stream")?,
                        u("offset")?,
                        u("length")? as usize,
                    )?;
                    if format!("{:x}", Sha256::digest(&packed)) != r["raw_sha256"].as_str().unwrap()
                    {
                        return Err("RAW hash mismatch during video rendering".into());
                    }
                    let raw = raw10::try_unpack_raw10(&packed, 420, 280, u("stride")? as usize)?;
                    let preview = native(&raw, r);
                    for column in 0..3 {
                        let mut image = preview.clone();
                        if column > 0 {
                            overlay(
                                &mut image,
                                if column == 1 { r } else { &b[id] },
                                column == 2,
                                motion_only,
                            );
                        }
                        let (x, y) = (9 + 429 * column, 180 + 320 * eye);
                        for yy in 0..280 {
                            pixels[(y + yy) * W + x..(y + yy) * W + x + 420]
                                .copy_from_slice(&image[yy * 420..(yy + 1) * 420]);
                        }
                    }
                    sources.push(
                        json!({"source":s,"raw_sha256":r["raw_sha256"],"target":r["target"]}),
                    );
                }
                // A presentation command is shown as context only, with its
                // recorded host-arrival timing caveat visible throughout.
                for y in 10..116 {
                    for x in 1120..1282 {
                        pixels[y * W + x] = 0x273442;
                    }
                }
                for y in [29, 63, 97] {
                    for x in [1144, 1201, 1258] {
                        circle(&mut pixels, W, H, [x as f64, y as f64], 0x647383);
                    }
                }
                let target = &a[ids[0].unwrap()]["target"];
                if let Some(uv) = target["uv"].as_array() {
                    let p = [
                        1130. + 142. * uv[0].as_f64().unwrap(),
                        20. + 86. * uv[1].as_f64().unwrap(),
                    ];
                    point(&mut pixels, W, H, (p[0], p[1]), 0xffda56, 3);
                }
                digit(&mut pixels, 1080, 128, session + 1);
                let end = pairs.get(index + 1).map_or(stamp + 157_000_000, |p| p.0);
                let last_tick = ((end - start) as f64 / 1e9 * 30.).round() as usize;
                let repeat = last_tick.saturating_sub(total_frames - video_start).max(1);
                let rgb = pixels
                    .iter()
                    .flat_map(|&p| [(p >> 16) as u8, (p >> 8) as u8, p as u8])
                    .collect::<Vec<_>>();
                for _ in 0..repeat {
                    pipe.write_all(&rgb)?;
                    total_frames += 1;
                }
                manifest.push(json!({"session":session+1,"video_frame_start":total_frames-repeat,"video_frame_count":repeat,"source_ns":stamp,"eyes":sources}));
            }
        }
        drop(pipe);
        Ok(
            json!({"schema":"buttercup-cluster-review-video-v1","motion_only_ablation":motion_only,"video":out,"output_fps":30,"video_frames":total_frames,"source_pairs":manifest,
            "scope":if motion_only {"Two matched RAW sessions, original tracker versus motion-only ablation. Candidate rows verified free of iris seeds, inferred regions, anatomical selection and ellipse fits. Point correspondences can differ. Native 420x280 panels; source timing preserved with presentation repeats. Colors are current anonymous groups, not durable tissue names. No anatomical accuracy or live performance claim."} else {"Two completed source-matched offline clustering runs; all paired RAW hashes and point correspondences checked. Native 420x280 eye pixels; presentation color preview only. Frame holds preserve source cadence to 1/30s. No interpolated tracking observations, live performance, tissue truth, or experimental ellipse/gaze claim."}}),
        )
    })();
    if render.is_err() {
        let _ = child.kill();
    }
    let status = child.wait()?;
    let report = render?;
    if !status.success() {
        return Err(format!("encoder failed: {status}").into());
    }
    fs::write(
        out.with_extension("manifest.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}: {} frames at 30fps",
        out.display(),
        report["video_frames"]
    );
    Ok(())
}
