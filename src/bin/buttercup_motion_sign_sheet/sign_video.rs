//! Source-timed multi-recording RAW review. Reads frozen solver decisions;
//! eyelid cues are displayed separately and never overwrite that solver.
use super::{conic_solver, geometry::Ellipse, num, raw10, raw_preview, uint, E};
#[path = "eyelid_cue.rs"]
mod eyelid_cue;
use conic_solver::joint::{circle_pose_hypotheses, PinholeCamera};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs,
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write as _},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
type P = [f64; 2];
type N = [f64; 3];
fn rows(p: &Path) -> Result<Vec<Value>, E> {
    BufReader::new(fs::File::open(p)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}
fn e(v: &Value) -> Option<Ellipse> {
    Some(Ellipse {
        center: (
            v["center_sensor_px"][0].as_f64()?,
            v["center_sensor_px"][1].as_f64()?,
        ),
        major_radius: v["a"].as_f64()?,
        minor_radius: v["b"].as_f64()?,
        angle: v["angle"].as_f64()?,
    })
}
fn escaped(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
fn base64(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for b in bytes.chunks(3) {
        let n = ((b[0] as u32) << 16)
            | ((b.get(1).copied().unwrap_or(0) as u32) << 8)
            | b.get(2).copied().unwrap_or(0) as u32;
        s.push(A[(n >> 18) as usize] as char);
        s.push(A[((n >> 12) & 63) as usize] as char);
        s.push(if b.len() > 1 {
            A[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        s.push(if b.len() > 2 {
            A[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    s
}
fn text(svg: &mut String, x: f64, y: f64, size: usize, color: &str, s: &str) {
    write!(
        svg,
        "<text x='{x}' y='{y}' font-size='{size}' fill='{color}'>{}</text>",
        escaped(s)
    )
    .unwrap();
}
fn source_key(v: &Value) -> (u64, u64, u64) {
    let m = &v["input"]["frame"];
    (
        uint(&m["timestamp_ns"]),
        uint(&m["eye_id"]),
        uint(&m["sequence"]),
    )
}
fn boundaries(worker: &Value, source: &Value) -> Option<Vec<P>> {
    let q = worker["selected_query"].as_u64()?;
    let c = worker["candidates"]
        .as_array()?
        .iter()
        .find(|c| c["query"].as_u64() == Some(q))?;
    let original = e(&source["ellipse"])?;
    let m = &source["input"]["frame"];
    let fit = &c["baseline_ellipse"];
    let center = [
        fit["center"][0].as_f64()? + num(&m["sensor_x"]),
        fit["center"][1].as_f64()? + num(&m["sensor_y"]),
    ];
    if (center[0] - original.center.0).hypot(center[1] - original.center.1) > 1e-3
        || (fit["major_radius"].as_f64()? - original.major_radius).abs() > 1e-3
    {
        return None;
    }
    let mut out = vec![];
    for k in ["baseline_retained", "baseline_censored"] {
        for p in c[k].as_array().into_iter().flatten() {
            out.push([
                num(&p[0]) + num(&m["sensor_x"]),
                num(&p[1]) + num(&m["sensor_y"]),
            ]);
        }
    }
    Some(out)
}
struct EyeFrame {
    source: Value,
    decision: Value,
    cue: Value,
    boundary: Vec<P>,
    png: PathBuf,
    hash: String,
    poses: Option<[[f64; 3]; 2]>,
    directions: Option<[P; 2]>,
    centers: Option<[P; 2]>,
    display_order: [usize; 2],
    cue_choice: Option<usize>,
    cue_history: Value,
    gated: bool,
}
#[derive(Default)]
struct CueHistory {
    previous: Option<(N, u64, u64)>,
    opposing_since: Option<(u64, usize)>,
}
impl CueHistory {
    fn update(
        &mut self,
        ns: u64,
        normals: Option<[N; 2]>,
        cue: Option<usize>,
        admitted: bool,
    ) -> Value {
        let dot = |a: N, b: N| (0..3).map(|i| a[i] * b[i]).sum::<f64>();
        let Some(n) = normals.filter(|n| admitted && dot(n[0], n[1]) < 3_f64.to_radians().cos())
        else {
            self.previous = None;
            self.opposing_since = None;
            return json!({"choice":null,"reason":"no separated current admitted conic"});
        };
        let old = self.previous.and_then(|(normal, evidence, last)| {
            if ns <= last || ns - last > 500_000_000 || ns - evidence > 2_000_000_000 {
                return None;
            }
            let scores = n.map(|v| dot(normal, v));
            let k = usize::from(scores[1] > scores[0]);
            (scores[k] > 45_f64.to_radians().cos() && (scores[0] - scores[1]).abs() > 0.005)
                .then_some((k, evidence))
        });
        let mut choice = old.map(|o| o.0);
        let mut evidence = old.map(|o| o.1);
        if let Some(k) = cue {
            if old.is_none_or(|o| o.0 == k) {
                choice = Some(k);
                evidence = Some(ns);
                self.opposing_since = None;
            } else {
                let (start, count) = self.opposing_since.unwrap_or((ns, 0));
                self.opposing_since = Some((start, count + 1));
                if count + 1 >= 3 && ns - start >= 150_000_000 {
                    choice = Some(k);
                    evidence = Some(ns);
                    self.opposing_since = None;
                }
            }
        } else {
            self.opposing_since = None;
        }
        if let (Some(k), Some(t)) = (choice, evidence) {
            self.previous = Some((n[k], t, ns));
            json!({"choice":k,"evidence_source_ns":t,"evidence_age_ms":(ns-t)as f64/1e6,"fresh":t==ns,"current_cue":cue})
        } else {
            self.previous = None;
            json!({"choice":null,"reason":"no recent cue"})
        }
    }
}
fn raw(source: &Value) -> Result<(Vec<u32>, String), E> {
    let i = &source["input"];
    let m = &i["frame"];
    assert_eq!(source["sequence"], m["sequence"]);
    assert_eq!(source["eye"], m["eye_id"]);
    assert_eq!(source["source_ns"], m["timestamp_ns"]);
    let mut f = fs::File::open(i["raw_file"].as_str().ok_or("missing RAW path")?)?;
    f.seek(SeekFrom::Start(uint(&i["raw_offset"])))?;
    let mut bytes = vec![0; uint(&i["raw_length"]) as usize];
    f.read_exact(&mut bytes)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let pixels = raw10::try_unpack_raw10(
        &bytes,
        uint(&m["width"]) as usize,
        uint(&m["height"]) as usize,
        uint(&m["stride"]) as usize,
    )?;
    let rgb = raw_preview::color_preview(
        &pixels,
        uint(&m["width"]) as usize,
        uint(&m["height"]) as usize,
        uint(&m["sensor_x"]) as u32,
        uint(&m["sensor_y"]) as u32,
        100,
        None,
    );
    Ok((rgb, hash))
}
fn caption(status: &str) -> &str {
    match status {
        "conditional outward branch" => "Conditional direction selected",
        "ambiguous centre support" => "Uncertain: rotation centre is not established",
        "behind-distance hypothesis unsupported" => {
            "Uncertain: inferred pivot distance is implausible"
        }
        "centre model contradicted by current ellipse" => {
            "Uncertain: current ellipse contradicts the centre model"
        }
        "missing or quality-rejected ellipse" => "No accepted ellipse for this exposure",
        "insufficient past angular/time support" => "Uncertain: insufficient recent eye movement",
        _ => status,
    }
}
fn draw_eye(s: &mut String, f: Option<&EyeFrame>, eye: usize, provider: &str) -> Result<(), E> {
    let x = 30. + 900. * eye as f64;
    let y = 150.;
    let width = 840.;
    text(
        s,
        x,
        123.,
        24,
        "#eaf0f6",
        &format!(
            "{} eye  |  {provider}",
            if eye == 0 { "Right" } else { "Left" }
        ),
    );
    let Some(f) = f else {
        text(
            s,
            x,
            y + 200.,
            24,
            "#a8b2c2",
            "No ROI archived for this eye at this exposure",
        );
        return Ok(());
    };
    let m = &f.source["input"]["frame"];
    let w = num(&m["width"]);
    let h = num(&m["height"]);
    let scale = width / w;
    let origin = [num(&m["sensor_x"]), num(&m["sensor_y"])];
    let point = |p: P| {
        [
            x + (p[0] - origin[0]) * scale,
            y + (p[1] - origin[1]) * scale,
        ]
    };
    let data = base64(&fs::read(&f.png)?);
    write!(s,"<clipPath id='eye{eye}'><rect x='{x}' y='{y}' width='{width}' height='{}'/></clipPath><g clip-path='url(#eye{eye})'><image x='{x}' y='{y}' width='{width}' height='{}' xlink:href='data:image/png;base64,{data}'/>",h*scale,h*scale)?;
    if let Some(e) = e(&f.source["ellipse"]) {
        let c = point([e.center.0, e.center.1]);
        write!(s,"<ellipse cx='{}' cy='{}' rx='{}' ry='{}' transform='rotate({} {} {})' fill='none' stroke='{}' stroke-width='2' {}/>",c[0],c[1],e.major_radius*scale,e.minor_radius*scale,e.angle.to_degrees(),c[0],c[1],if f.gated{"#ffffff"}else{"#a4abb8"},if f.gated{""}else{"stroke-dasharray='7 6'"})?;
        for p in &f.boundary {
            let p = point(*p);
            write!(
                s,
                "<circle cx='{}' cy='{}' r='1.8' fill='#fd91cb' fill-opacity='.75'/>",
                p[0], p[1]
            )?;
        }
        // Sparse vertical ticks expose the actual overlap measurement, not a
        // fabricated continuous eyelid curve.
        for t in f.cue["traces"].as_array().into_iter().flatten().step_by(3) {
            for k in 0..2 {
                let a = point([num(&t["x"]), num(&t["observed"][k])]);
                let b = point([num(&t["x"]), num(&t["fitted"][k])]);
                write!(
                    s,
                    "<path d='M{},{} L{},{}' stroke='#edff8a' stroke-width='3'/>",
                    a[0], a[1], b[0], b[1]
                )?;
            }
        }
        for p in f.cue["white_gap"]["points"]
            .as_array()
            .into_iter()
            .flatten()
            .step_by(4)
        {
            let p = point([num(&p[0]), num(&p[1])]);
            write!(
                s,
                "<circle cx='{}' cy='{}' r='1.5' fill='#edff8a'/>",
                p[0], p[1]
            )?;
        }
        if let (Some(dirs), Some(centers)) = (f.directions, f.centers) {
            for display in 0..2 {
                let native = f.display_order[display];
                let color = if display == 0 { "#59ebdc" } else { "#ffad58" };
                let start = point(centers[native]);
                let u = dirs[native];
                let end = [start[0] + u[0] * 125., start[1] + u[1] * 125.];
                let chosen = f.decision["selected"].as_u64() == Some(native as u64);
                write!(s,"<path d='M{},{} L{},{}' stroke='#111824' stroke-width='{}'/><path d='M{},{} L{},{}' stroke='{color}' stroke-width='{}' opacity='{}'/><path d='M{},{} L{},{} L{},{} Z' fill='{color}' opacity='{}'/>",start[0],start[1],end[0],end[1],if chosen{8}else{4},start[0],start[1],end[0],end[1],if chosen{5}else{2},if chosen{1.}else{0.7},end[0],end[1],end[0]-u[0]*15.+u[1]*7.,end[1]-u[1]*15.-u[0]*7.,end[0]-u[0]*15.-u[1]*7.,end[1]-u[1]*15.+u[0]*7.,if chosen{1.}else{0.7})?;
                text(
                    s,
                    end[0] + 8.,
                    end[1] - 8.,
                    23,
                    color,
                    if display == 0 { "A" } else { "B" },
                );
            }
        }
    }
    s.push_str("</g>");
    let name = |native: usize| {
        if f.display_order[0] == native {
            "A"
        } else {
            "B"
        }
    };
    let temporal = f.decision["selected"]
        .as_u64()
        .map(|n| name(n as usize))
        .unwrap_or("uncertain");
    text(
        s,
        x,
        756.,
        27,
        "#edf3ff",
        &format!("Temporal solver: {temporal}"),
    );
    text(
        s,
        x,
        786.,
        18,
        "#adb9ca",
        caption(f.decision["status"].as_str().unwrap_or("no saved solve")),
    );
    let cue = if let Some(k) = f.cue_history["choice"].as_u64() {
        if f.cue_history["fresh"] == true {
            format!("{} (current cue)", name(k as usize))
        } else {
            format!(
                "{} (history, {:.0} ms old)",
                name(k as usize),
                num(&f.cue_history["evidence_age_ms"])
            )
        }
    } else {
        "no preference".to_owned()
    };
    text(
        s,
        x,
        829.,
        25,
        "#edff8a",
        &format!("Mask / eyelid hypothesis: {cue}"),
    );
    let metric = |v: &Value| {
        v.as_f64()
            .map(|x| format!("{:.1} px", x.max(0.)))
            .unwrap_or_else(|| "unavailable".to_owned())
    };
    text(
        s,
        x,
        861.,
        19,
        "#c6d0df",
        &format!(
            "Overlap   top: {}    bottom: {}",
            metric(&f.cue["upper_overlap_px"]),
            metric(&f.cue["lower_overlap_px"])
        ),
    );
    text(
        s,
        x,
        892.,
        19,
        "#c6d0df",
        &format!(
            "White gap   top: {}    bottom: {}",
            metric(&f.cue["white_gap"]["upper_px"]),
            metric(&f.cue["white_gap"]["lower_px"])
        ),
    );
    let reason = if f.gated {
        f.cue["status"]
            .as_str()
            .unwrap_or("No fresh boundary evidence")
    } else {
        "Ellipse rejected or absent: overlap is not trusted"
    };
    text(s, x, 921., 17, "#adb9ca", reason);
    text(
        s,
        x,
        956.,
        15,
        "#8d9aaf",
        &format!("Source {}  |  RAW {}", f.source["sequence"], &f.hash[..16]),
    );
    Ok(())
}
pub fn run(manifest: &Path, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("use a new output directory".into());
    }
    if !out
        .parent()
        .ok_or("missing output parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk output root".into());
    }
    let config: Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let clips = config["clips"].as_array().ok_or("clips required")?;
    let mut decisions = BTreeMap::new();
    for d in rows(Path::new(
        config["decisions"]
            .as_str()
            .ok_or("decisions path required")?,
    ))? {
        if d["variant"] == "static-2s" {
            decisions.insert(
                (
                    d["capture"].as_str().unwrap().to_owned(),
                    d["provider"].as_str().unwrap().to_owned(),
                    uint(&d["eye"]),
                    uint(&d["sequence"]),
                ),
                d,
            );
        }
    }
    fs::create_dir(out)?;
    fs::create_dir(out.join("raw"))?;
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    let mut concat = "ffconcat version 1.0\n".to_owned();
    let mut timeline = vec![];
    let mut audits = vec![];
    let mut video_time = 0.;
    let mut svg_count = 0;
    let mut summaries = vec![];
    for (ci, clip) in clips.iter().enumerate() {
        let capture = clip["capture"].as_str().ok_or("capture required")?;
        let provider = clip["provider"].as_str().ok_or("provider required")?;
        let sides = rows(Path::new(clip["sidecar"].as_str().unwrap()))?
            .into_iter()
            .map(|v| (uint(&v["id"]), v))
            .collect::<BTreeMap<_, _>>();
        let gated = rows(Path::new(clip["admitted_shapes"].as_str().unwrap()))?
            .into_iter()
            .map(|v| (uint(&v["id"]), v["ellipse"].is_object()))
            .collect::<BTreeMap<_, _>>();
        let mut workers = BTreeMap::new();
        for w in rows(Path::new(clip["worker"].as_str().unwrap()))? {
            workers.entry(source_key(&w)).or_insert(w);
        }
        let sources = rows(Path::new(clip["shapes"].as_str().unwrap()))?
            .into_iter()
            .filter(|s| s["capture"] == capture && s["provider"] == provider)
            .filter(|s| {
                clip["first_sequence"]
                    .as_u64()
                    .is_none_or(|n| uint(&s["sequence"]) >= n)
                    && clip["last_sequence"]
                        .as_u64()
                        .is_none_or(|n| uint(&s["sequence"]) <= n)
            })
            .collect::<Vec<_>>();
        if sources.is_empty() {
            return Err(format!("no sources for {capture}/{provider}").into());
        }
        let mut grouped = BTreeMap::<u64, [Option<EyeFrame>; 2]>::new();
        let first_meta = &sides[&uint(&sources[0]["id"])]["input"]["frame"];
        let width = uint(&first_meta["width"]);
        let height = uint(&first_meta["height"]);
        let pattern = out.join("raw").join(format!("clip{ci}-%05d.png"));
        let mut child = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                &format!("{width}x{height}"),
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
            .arg(&pattern)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()?;
        let mut stdin = child.stdin.take().ok_or("missing PNG encoder stdin")?;
        let mut ordering = BTreeMap::<u64, ([N; 2], u64)>::new();
        let mut cue_histories = BTreeMap::<u64, CueHistory>::new();
        let mut source_order = sources;
        source_order.sort_by_key(|s| (uint(&s["source_ns"]), uint(&s["eye"])));
        let camera = PinholeCamera {
            focal_px: [4000.; 2],
            principal_px: [4000., 3000.],
        };
        let mut cue_count = 0;
        let mut solver_count = 0;
        let mut both = 0;
        let mut agree = 0;
        let mut boundary_misses = 0;
        for (pi, mut source) in source_order.into_iter().enumerate() {
            let id = uint(&source["id"]);
            source["input"] = sides.get(&id).ok_or("sidecar missing")?["input"].clone();
            let meta = &source["input"]["frame"];
            assert_eq!(uint(&meta["width"]), width);
            assert_eq!(uint(&meta["height"]), height);
            assert_eq!(source["clock"], source["input"]["clock_lineage"]);
            let eye = uint(&source["eye"]);
            let seq = uint(&source["sequence"]);
            let ns = uint(&source["source_ns"]);
            let d = decisions
                .get(&(capture.to_owned(), provider.to_owned(), eye, seq))
                .ok_or("saved decision missing")?
                .clone();
            assert_eq!(d["source_ns"], source["source_ns"]);
            let (pixels, hash) = raw(&source)?;
            let bytes = pixels
                .iter()
                .flat_map(|v| [(v >> 16) as u8, (v >> 8) as u8, *v as u8])
                .collect::<Vec<_>>();
            stdin.write_all(&bytes)?;
            let png = out.join("raw").join(format!("clip{ci}-{pi:05}.png"));
            let boundary = workers
                .get(&source_key(&source))
                .filter(|w| w["input"]["clock_lineage"] == source["clock"])
                .and_then(|w| boundaries(w, &source))
                .unwrap_or_default();
            boundary_misses += usize::from(boundary.is_empty());
            let origin = [num(&meta["sensor_x"]), num(&meta["sensor_y"])];
            let cue = e(&source["ellipse"])
                .map(|e| {
                    eyelid_cue::measure(
                        e,
                        &boundary,
                        &pixels,
                        width as usize,
                        height as usize,
                        origin,
                    )
                })
                .unwrap_or_else(|| json!({"status":"no current ellipse"}));
            let mut display_order = [0, 1];
            let mut normals = None;
            let mut directions = None;
            let mut centers = None;
            let mut cue_choice = None;
            if let Some(poses) =
                e(&source["ellipse"]).and_then(|e| circle_pose_hypotheses(camera, e, [0, 0]))
            {
                let n = poses.map(|p| p.normal);
                let c = poses.map(|p| camera.project(p.center_per_radius).unwrap());
                let dirs = std::array::from_fn(|k| {
                    let q = camera
                        .project(std::array::from_fn(|i| {
                            poses[k].center_per_radius[i] + 0.05 * n[k][i]
                        }))
                        .unwrap();
                    let v = [q[0] - c[k][0], q[1] - c[k][1]];
                    let l = v[0].hypot(v[1]).max(1e-12);
                    v.map(|v| v / l)
                });
                if let Some((old, t)) = ordering
                    .get(&eye)
                    .filter(|(_, t)| ns > *t && ns - *t <= 500_000_000)
                {
                    let score =
                        |a: usize, b: usize| (0..3).map(|i| old[a][i] * n[b][i]).sum::<f64>();
                    if score(0, 1) + score(1, 0) > score(0, 0) + score(1, 1) {
                        display_order = [1, 0];
                    }
                    let _ = t;
                } else if dirs[0][1] > dirs[1][1] {
                    display_order = [1, 0];
                }
                ordering.insert(eye, ([n[display_order[0]], n[display_order[1]]], ns));
                if let Some(y) = cue["image_y_direction"].as_i64() {
                    let k = usize::from(dirs[1][1] * y as f64 > dirs[0][1] * y as f64);
                    let separation_dot = (0..3).map(|i| n[0][i] * n[1][i]).sum::<f64>();
                    if dirs[k][1] * y as f64 > 0.25
                        && gated[&id]
                        && separation_dot < 3_f64.to_radians().cos()
                    {
                        cue_choice = Some(k);
                    }
                }
                if let Some(saved) = d["branches"].as_array() {
                    for k in 0..2 {
                        let dot = (0..3)
                            .map(|i| n[k][i] * num(&saved[k]["normal_camera_right_down_toward"][i]))
                            .sum::<f64>();
                        assert!(
                            dot > 1. - 1e-6,
                            "saved sign geometry must match displayed RAW ellipse"
                        );
                    }
                }
                normals = Some(n);
                directions = Some(dirs);
                centers = Some(c);
            }
            let selected = d["selected"].as_u64().map(|v| v as usize);
            let cue_history = cue_histories
                .entry(eye)
                .or_default()
                .update(ns, normals, cue_choice, gated[&id]);
            solver_count += usize::from(selected.is_some());
            cue_count += usize::from(cue_choice.is_some());
            if let (Some(a), Some(b)) = (selected, cue_choice) {
                both += 1;
                agree += usize::from(a == b);
            }
            audits.push(json!({"clip":ci,"capture":capture,"provider":provider,"eye":eye,"sequence":seq,"source_ns":ns,"raw_sha256":hash,"input":source["input"],"ellipse":source["ellipse"],"ellipse_admitted":gated[&id],"temporal_selected":selected,"temporal_status":d["status"],"cue_selected":cue_choice,"cue_history":cue_history,"cue":cue,"native_normals":normals,"display_native_order":display_order,"boundary":boundary}));
            let frame = EyeFrame {
                source,
                decision: d,
                cue,
                boundary,
                png,
                hash,
                poses: normals,
                directions,
                centers,
                display_order,
                cue_choice,
                cue_history,
                gated: gated[&id],
            };
            let slot = &mut grouped.entry(ns).or_insert_with(|| [None, None])[(eye - 1) as usize];
            assert!(slot.is_none(), "duplicate source eye exposure");
            *slot = Some(frame);
        }
        drop(stdin);
        if !child.wait()?.success() {
            return Err("RAW PNG encoder failed".into());
        }
        let ordered = grouped.into_iter().collect::<Vec<_>>();
        let first = ordered[0].0;
        let last = ordered.last().unwrap().0;
        let slowdown = clip["slowdown"].as_f64().unwrap_or(1.5);
        if !(1. ..=5.).contains(&slowdown) {
            return Err("slowdown must be 1..5".into());
        }
        let mut steps = ordered
            .windows(2)
            .map(|p| (p[1].0 - p[0].0) as f64 / 1e9)
            .collect::<Vec<_>>();
        steps.sort_by(f64::total_cmp);
        let nominal = steps.get(steps.len() / 2).copied().unwrap_or(0.1).min(0.2);
        let start_video = video_time;
        for (i, (ns, pair)) in ordered.iter().enumerate() {
            let duration = ordered
                .get(i + 1)
                .map(|p| (p.0 - *ns) as f64 / 1e9)
                .unwrap_or(nominal);
            let time = (*ns - first) as f64 / 1e9;
            let title = clip["name"].as_str().unwrap_or(capture);
            let mut svg="<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='1800' height='1080'><rect width='100%' height='100%' fill='#151b25'/><g font-family='sans-serif'>".to_owned();
            text(
                &mut svg,
                30.,
                43.,
                29,
                "#f2f5fa",
                &format!("{}/{}  {title}", ci + 1, clips.len()),
            );
            text(&mut svg,30.,79.,20,"#b9c6d8",&format!("Source time {time:6.2} s  |  {slowdown:.1}x slower  |  thick arrow = selected temporal sign"));
            draw_eye(
                &mut svg,
                pair[0].as_ref(),
                0,
                clip["provider_label"].as_str().unwrap_or(provider),
            )?;
            draw_eye(
                &mut svg,
                pair[1].as_ref(),
                1,
                clip["provider_label"].as_str().unwrap_or(provider),
            )?;
            text(&mut svg,30.,1003.,18,"#c6d0df","White ellipse: fitted limbus. Pink points: recorded mask boundary. Yellow ticks: measured overlap / white-gap proxy.");
            text(&mut svg,30.,1035.,18,"#c6d0df","A/B colors follow continuous normal candidates. Eyelid preference is a separate unvalidated hypothesis; neither is gaze truth.");
            svg.push_str("</g></svg>");
            let name = format!("frame-{svg_count:05}.svg");
            svg_count += 1;
            fs::write(out.join(&name), svg)?;
            let fresh_duration = if duration > 0.35 { nominal } else { duration };
            let display = fresh_duration * slowdown;
            write!(concat, "file {name}\nduration {display:.9}\n")?;
            timeline.push(json!({"file":name,"clip":ci,"source_ns":ns,"video_time":video_time,"duration":display,"fresh_source":true}));
            video_time += display;
            if duration > 0.35 {
                let mut gap="<svg xmlns='http://www.w3.org/2000/svg' width='1800' height='1080'><rect width='100%' height='100%' fill='#151b25'/><g font-family='sans-serif'>".to_owned();
                text(&mut gap, 80., 120., 32, "#edf3ff", title);
                text(
                    &mut gap,
                    200.,
                    480.,
                    40,
                    "#adb9ca",
                    &format!(
                        "No archived exposure for {:.0} ms",
                        (duration - nominal) * 1000.
                    ),
                );
                text(
                    &mut gap,
                    200.,
                    545.,
                    23,
                    "#adb9ca",
                    "The gap is preserved; no held frame is presented as fresh evidence.",
                );
                gap.push_str("</g></svg>");
                let file = format!("frame-{svg_count:05}.svg");
                svg_count += 1;
                fs::write(out.join(&file), gap)?;
                let display = (duration - nominal) * slowdown;
                write!(concat, "file {file}\nduration {display:.9}\n")?;
                timeline.push(json!({"file":file,"clip":ci,"source_ns":ns,"video_time":video_time,"duration":display,"fresh_source":false}));
                video_time += display;
            }
        }
        let summary = json!({"clip":ci,"capture":capture,"provider":provider,"paired_timestamps":ordered.len(),"source_seconds":(last-first)as f64/1e9,"video_start_seconds":start_video,"video_end_seconds":video_time,"temporal_choices":solver_count,"eyelid_cue_choices":cue_count,"both_choose":both,"agree":agree,"missing_or_unmatched_boundaries":boundary_misses});
        println!("{summary}");
        summaries.push(summary);
    }
    if let Some(last) = timeline.last() {
        writeln!(concat, "file {}", last["file"].as_str().unwrap())?;
    }
    fs::write(out.join("video.ffconcat"), concat)?;
    fs::write(
        out.join("timeline.json"),
        serde_json::to_vec_pretty(&json!({"duration":video_time,"frames":timeline}))?,
    )?;
    let mut w = std::io::BufWriter::new(fs::File::create(out.join("evidence.jsonl"))?);
    for a in &audits {
        writeln!(w, "{a}")?;
    }
    w.flush()?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"Three physical recordings and four provider passes by default; not four independent videos. Frozen temporal solver, separate current-frame eyelid/color cue. RAW and all dropouts retained. No sign truth.","clips":summaries,"video_seconds":video_time,"source_exposures":audits.len(),"source_sha256":format!("{:x}",Sha256::digest(include_bytes!("sign_video.rs"))),"eyelid_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("eyelid_cue.rs")))}),
        )?,
    )?;
    Ok(())
}
