//! Corpus evaluation of shared projected eye-centre sign initialization.
//! Uses Buttercup's circle unprojection; no external tracker source or runtime.
use super::{conic_solver, geometry, num, raw10, raw_preview, uint, E};
use conic_solver::joint::{circle_pose_hypotheses, PinholeCamera};
use geometry::Ellipse;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Seek, SeekFrom, Write as _},
    path::Path,
};
#[path = "../../projected_eye_center.rs"]
mod projected_eye_center;
use projected_eye_center::{CenterFit, NormalLine};
type P = [f64; 2];
fn read(path: &Path) -> Result<Vec<Value>, E> {
    fs::read_to_string(path)?
        .lines()
        .map(|l| Ok(serde_json::from_str(l)?))
        .collect()
}
fn ellipse(v: &Value) -> Ellipse {
    Ellipse {
        center: (
            num(&v["center_sensor_px"][0]),
            num(&v["center_sensor_px"][1]),
        ),
        major_radius: num(&v["a"]),
        minor_radius: num(&v["b"]),
        angle: num(&v["angle"]),
    }
}
fn dot(a: P, b: P) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn projected_line(
    camera: PinholeCamera,
    pose: conic_solver::joint::CirclePoseSeed,
) -> Option<NormalLine> {
    let p = camera.project(pose.center_per_radius)?;
    let q = camera.project(std::array::from_fn(|i| {
        pose.center_per_radius[i] + 0.05 * pose.normal[i]
    }))?;
    let delta = [q[0] - p[0], q[1] - p[1]];
    let length = delta[0].hypot(delta[1]);
    if length < 1e-8 {
        return None;
    }
    Some(NormalLine {
        point: p,
        direction: delta.map(|v| v / length),
        seconds: 0.,
        weight: 1.,
    })
}
fn decision(camera: PinholeCamera, e: Ellipse, center: CenterFit) -> Value {
    let Some(poses) = circle_pose_hypotheses(camera, e, [0, 0]) else {
        return json!({"status":"invalid conic"});
    };
    let mut branches = vec![];
    for p in poses {
        let Some(line) = projected_line(camera, p) else {
            return json!({"status":"coalesced projected normal"});
        };
        let signed = dot(
            line.direction,
            [
                line.point[0] - center.center[0],
                line.point[1] - center.center[1],
            ],
        );
        let u = (center.center[0] - camera.principal_px[0]) / camera.focal_px[0];
        let v = (center.center[1] - camera.principal_px[1]) / camera.focal_px[1];
        let a = [p.normal[0] + u * p.normal[2], p.normal[1] + v * p.normal[2]];
        let b = [
            p.center_per_radius[0] + u * p.center_per_radius[2],
            p.center_per_radius[1] + v * p.center_per_radius[2],
        ];
        let distance_per_radius = dot(a, b) / dot(a, a).max(1e-18);
        branches.push(json!({"normal_camera_right_down_toward":p.normal,"center_per_radius":p.center_per_radius,"projected_circle_center":line.point,"projected_outward_unit":line.direction,"outward_margin_px":signed,"behind_distance_mm_assuming_6mm_iris":6.*distance_per_radius,"outward_supported":signed>center.support_radius_px}));
    }
    let separation = poses[0]
        .normal
        .iter()
        .zip(poses[1].normal)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-1., 1.)
        .acos()
        .to_degrees();
    let winners = (0..2)
        .filter(|&k| branches[k]["outward_supported"] == true)
        .collect::<Vec<_>>();
    let transverse = |i: usize| {
        let line = projected_line(camera, poses[i]).unwrap();
        let diff = [
            center.center[0] - line.point[0],
            center.center[1] - line.point[1],
        ];
        (line.direction[0] * diff[1] - line.direction[1] * diff[0]).abs()
    };
    let residual = (transverse(0) + transverse(1)) / 2.;
    let mut status = "ambiguous centre support";
    let mut selected = None;
    if separation < 3. {
        status = "near-coalescence";
    } else if center.rms_px > 6. || residual > center.support_radius_px.max(6.) {
        status = "centre model contradicted by current ellipse";
    } else if winners.len() == 1 {
        let k = winners[0];
        let d = num(&branches[k]["behind_distance_mm_assuming_6mm_iris"]);
        if !(10. ..=50.).contains(&d) {
            status = "behind-distance hypothesis unsupported";
        } else {
            status = "conditional outward branch";
            selected = Some(k);
        }
    }
    json!({"status":status,"selected":selected,"branches":branches,"branch_separation_degrees":separation,"current_line_residual_px":residual,"eye_center_sensor":center.center,"eye_center_velocity_px_s":center.velocity,"center_support_radius_px":center.support_radius_px,"history_rms_px":center.rms_px,"history_median_px":center.median_px,"history_condition":center.condition,"history_observations":center.observations})
}
pub fn run(base: &Path, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("use new output directory".into());
    }
    if !out
        .parent()
        .ok_or("no parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let mut sets = BTreeMap::<String, Vec<Value>>::new();
    for (tag, path, admitted) in [
        (
            "original",
            base.to_path_buf(),
            base.join("quality-gate/admitted-shapes.jsonl"),
        ),
        (
            "expanded",
            base.join("expanded-corpus"),
            base.join("expanded-corpus/admitted-shapes.jsonl"),
        ),
    ] {
        let sides = read(&path.join("evaluation-sidecar.jsonl"))?
            .into_iter()
            .map(|v| (uint(&v["id"]), v))
            .collect::<BTreeMap<_, _>>();
        for mut v in read(&admitted)? {
            let side = sides.get(&uint(&v["id"])).ok_or("missing source sidecar")?;
            v["input"] = side["input"].clone();
            v["retained"] = side["retained"].clone();
            v["dataset"] = json!(tag);
            let key = format!(
                "{tag}/{}/{}/eye{}",
                v["capture"].as_str().unwrap(),
                v["provider"].as_str().unwrap(),
                v["eye"]
            );
            sets.entry(key).or_default().push(v);
        }
    }
    fs::create_dir(out)?;
    let mut writer = std::io::BufWriter::new(fs::File::create(out.join("decisions.jsonl"))?);
    let mut summaries = vec![];
    let mut reviews = vec![];
    for (key, mut rows) in sets {
        rows.sort_by_key(|v| uint(&v["source_ns"]));
        for (name, focal, window, moving) in [
            ("static-2s", 4000., 2., false),
            ("static-5s", 4000., 5., false),
            ("linear-2s", 4000., 2., true),
            ("static-2s-f3200", 3200., 2., false),
            ("static-2s-f4800", 4800., 2., false),
        ] {
            let camera = PinholeCamera {
                focal_px: [focal; 2],
                principal_px: [4000., 3000.],
            };
            let mut history = vec![];
            let mut counts = BTreeMap::<String, usize>::new();
            let mut residuals = vec![];
            let mut previous_time = None;
            let mut previous_sequence = None;
            let mut previous_clock = String::new();
            let mut previous_region = Value::Null;
            let mut selected_count = 0;
            for source in &rows {
                let now = uint(&source["source_ns"]) as f64 * 1e-9;
                let clock = source["clock"].as_str().unwrap();
                let region = &source["input"]["frame"]["region"]["session"];
                if previous_time.is_some_and(|t| now - t > 0.5)
                    || previous_sequence.is_some_and(|s| uint(&source["sequence"]) < s)
                    || history
                        .last()
                        .is_some_and(|l: &NormalLine| now - l.seconds > 0.5)
                    || previous_clock != clock
                    || previous_region != *region
                {
                    history.clear();
                }
                previous_time = Some(now);
                previous_sequence = Some(uint(&source["sequence"]));
                previous_clock = clock.to_owned();
                previous_region = region.clone();
                history.retain(|l: &NormalLine| now - l.seconds <= window);
                let mut result = json!({"stream":key,"variant":name,"id":source["id"],"sequence":source["sequence"],"source_ns":source["source_ns"],"provider":source["provider"],"stage":source["stage"],"eye":source["eye"],"capture":source["capture"],"status":"missing or quality-rejected ellipse"});
                if source["ellipse"].is_object() {
                    let e = ellipse(&source["ellipse"]);
                    result["ellipse"] = source["ellipse"].clone();
                    result["status"] = json!("insufficient past angular/time support");
                    if let Some(center) = projected_eye_center::fit(&history, now, moving) {
                        let dec = decision(camera, e, center);
                        for (k, v) in dec.as_object().unwrap() {
                            result[k] = v.clone();
                        }
                        if let Some(error) = result["current_line_residual_px"].as_f64() {
                            residuals.push(error);
                        }
                        if result["selected"].is_u64() {
                            selected_count += 1;
                            if name == "static-2s"
                                && (selected_count == 1 || selected_count % 40 == 0)
                            {
                                let mut review = result.clone();
                                review["source"] = source.clone();
                                reviews.push(review);
                            }
                        }
                    }
                    if let Some(poses) = circle_pose_hypotheses(camera, e, [0, 0]) {
                        if let Some(mut line) = projected_line(camera, poses[0]) {
                            line.seconds = now;
                            let sep = poses[0]
                                .normal
                                .iter()
                                .zip(poses[1].normal)
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                .clamp(-1., 1.)
                                .acos();
                            line.weight = (sep.sin().powi(2) / 0.25).clamp(0.01, 1.);
                            history.push(line);
                        }
                    }
                }
                *counts
                    .entry(result["status"].as_str().unwrap().to_owned())
                    .or_default() += 1;
                writeln!(writer, "{}", result)?;
            }
            residuals.sort_by(f64::total_cmp);
            let summary = json!({"stream":key,"variant":name,"rows":rows.len(),"admitted_ellipses":rows.iter().filter(|v|v["ellipse"].is_object()).count(),"stage":rows[0]["stage"],"status_counts":counts,"current_line_residual_median_px":residuals.get(residuals.len()/2),"current_line_residual_p90_px":residuals.get(((residuals.len().saturating_sub(1)) as f64*0.9).ceil() as usize)});
            if name == "static-2s" {
                println!("{}", summary);
            }
            summaries.push(summary);
        }
    }
    writer.flush()?;
    comparisons(out)?;
    fs::write(
        out.join("review-selection.json"),
        serde_json::to_vec_pretty(&reviews)?,
    )?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"method":"Exact native circular-section poses -> projected normal lines -> robust past-only line intersection -> outward branch and behind-distance check. No Orlosky source code used.","data":"Four recordings, ten provider/eye streams; unchanged prior quality gate; joint-conditioned stream is separately labeled.","assumptions":"Static or constant-image-velocity effective eye center over 2/5 seconds; assumed focal and 6 mm iris radius; 10–50 mm behind-iris support. Current ellipse is excluded from center fitting. Bounds are engineering support, not probabilities.","limitations":"Historical outer limbus fits, not freshly detected pupil contours. No true gaze/sign labels, fresh independent head pose or metric scale. Conditional choices are not sign accuracy. Source pixel motion will be audited separately.","summaries":summaries}),
        )?,
    )?;
    render(&reviews, out)?;
    Ok(())
}
pub fn render_saved(input: &Path, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("use new output directory".into());
    }
    if !out
        .parent()
        .ok_or("missing parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let reviews: Vec<Value> = serde_json::from_slice(&fs::read(input)?)?;
    if reviews.is_empty() {
        return Err("no source-matched reviews to render".into());
    }
    fs::create_dir(out)?;
    fs::write(
        out.join("review-selection.json"),
        serde_json::to_vec_pretty(&reviews)?,
    )?;
    render(&reviews, out)
}

fn render(reviews: &[Value], out: &Path) -> Result<(), E> {
    use std::fmt::Write as _;
    let mut seen = std::collections::BTreeSet::new();
    let selected = reviews
        .iter()
        .filter(|r| seen.insert(r["stream"].as_str().unwrap_or("transport").to_owned()))
        .take(6)
        .collect::<Vec<_>>();
    let mut svg="<svg xmlns='http://www.w3.org/2000/svg' width='1800' height='1140'><rect width='100%' height='100%' fill='#171b24'/><g font-family='sans-serif' fill='white'><text x='25' y='34' font-size='25'>Projected eye-centre sign hypotheses on actual RAW</text><text x='25' y='65' font-size='17'>White ellipse = recorded fit · cyan/orange = both projected normals · yellow cross/circle = prior centre/support</text><text x='25' y='90' font-size='16'>The current ellipse is withheld from centre fitting. Chosen arrows are conditional, not verified gaze labels.</text>".to_owned();
    for (i, r) in selected.iter().enumerate() {
        let source = &r["source"];
        let input = &source["input"];
        let meta = &input["frame"];
        assert_eq!(source["sequence"], meta["sequence"]);
        assert_eq!(source["eye"], meta["eye_id"]);
        let (w, h) = (
            uint(&meta["width"]) as usize,
            uint(&meta["height"]) as usize,
        );
        let origin = [num(&meta["sensor_x"]), num(&meta["sensor_y"])];
        let mut file = fs::File::open(input["raw_file"].as_str().unwrap())?;
        file.seek(SeekFrom::Start(uint(&input["raw_offset"])))?;
        let mut bytes = vec![0; uint(&input["raw_length"]) as usize];
        file.read_exact(&mut bytes)?;
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let pixels = raw10::try_unpack_raw10(&bytes, w, h, uint(&meta["stride"]) as usize)?;
        let rgb = raw_preview::color_preview(
            &pixels,
            w,
            h,
            origin[0] as u32,
            origin[1] as u32,
            100,
            None,
        );
        let (x, y) = (25. + 600. * (i % 3) as f64, 155. + 500. * (i / 3) as f64);
        let scale = 550. / w as f64;
        let pt = |p: P| {
            [
                x + (p[0] - origin[0]) * scale,
                y + (p[1] - origin[1]) * scale,
            ]
        };
        write!(svg,"<text x='{x}' y='{}' font-size='14'>{} / eye{} / source {} / {}</text><clipPath id='clip{i}'><rect x='{x}' y='{y}' width='550' height='{}'/></clipPath><g clip-path='url(#clip{i})'><g transform='translate({x},{y}) scale({scale})' shape-rendering='crispEdges'>",y-17.,r["provider"].as_str().unwrap(),r["eye"],r["sequence"],r["variant"].as_str().unwrap_or("transport"),h as f64*scale)?;
        for yy in 0..h {
            for xx in 0..w {
                write!(
                    svg,
                    "<rect x='{xx}' y='{yy}' width='1' height='1' fill='#{:06x}'/>",
                    rgb[yy * w + xx]
                )?;
            }
        }
        svg.push_str("</g>");
        let e = ellipse(&source["ellipse"]);
        let c = pt([e.center.0, e.center.1]);
        write!(svg,"<ellipse cx='{}' cy='{}' rx='{}' ry='{}' transform='rotate({} {} {})' fill='none' stroke='white' stroke-width='1.5'/>",c[0],c[1],e.major_radius*scale,e.minor_radius*scale,e.angle.to_degrees(),c[0],c[1])?;
        for (j, b) in r["branches"].as_array().unwrap().iter().enumerate() {
            let p = pt([
                num(&b["projected_circle_center"][0]),
                num(&b["projected_circle_center"][1]),
            ]);
            let u = [
                num(&b["projected_outward_unit"][0]),
                num(&b["projected_outward_unit"][1]),
            ];
            let color = if j == 0 { "#62f4d3" } else { "#ffa74f" };
            write!(svg,"<path d='M{},{} l{},{}' fill='none' stroke='{color}' stroke-width='{}'/><circle cx='{}' cy='{}' r='3' fill='{color}'/>",p[0],p[1],u[0]*90.,u[1]*90.,if r["selected"]==j {3.}else{1.},p[0]+u[0]*90.,p[1]+u[1]*90.)?;
        }
        let p = pt([
            num(&r["eye_center_sensor"][0]),
            num(&r["eye_center_sensor"][1]),
        ]);
        let rad = num(&r["center_support_radius_px"]) * scale;
        let choice = r["selected"]
            .as_u64()
            .map(|k| format!("branch {k}"))
            .unwrap_or_else(|| "no branch chosen".to_owned());
        write!(svg,"<circle cx='{}' cy='{}' r='{rad}' fill='none' stroke='#ffe268' stroke-dasharray='5 4'/><path d='M{},{} h16 m-8,-8 v16' stroke='#ffe268' stroke-width='2'/></g><text x='{x}' y='{}' font-size='16'>{choice} · current line {:.1}px · centre support {:.1}px</text><text x='{x}' y='{}' font-size='13'>RAW {}</text><text x='{x}' y='{}' font-size='13'>{}</text>",p[0],p[1],p[0]-8.,p[1],y+h as f64*scale+27.,num(&r["current_line_residual_px"]),num(&r["center_support_radius_px"]),y+h as f64*scale+50.,&hash[..20],y+h as f64*scale+72.,r["status"].as_str().unwrap_or("unclassified"))?;
    }
    svg.push_str("</g></svg>");
    fs::write(out.join("raw-centre-choices.svg"), svg)?;
    Ok(())
}

fn comparisons(out: &Path) -> Result<(), E> {
    let all = read(&out.join("decisions.jsonl"))?;
    let mut table = BTreeMap::<(String, u64, u64), BTreeMap<String, Value>>::new();
    for v in all {
        table
            .entry((
                v["stream"].as_str().unwrap().to_owned(),
                uint(&v["sequence"]),
                uint(&v["source_ns"]),
            ))
            .or_default()
            .insert(v["variant"].as_str().unwrap().to_owned(), v);
    }
    let mut reports = vec![];
    for name in [
        "linear-2s",
        "static-5s",
        "static-2s-f3200",
        "static-2s-f4800",
    ] {
        let mut count = 0;
        let mut agree = 0;
        let mut disagreements = vec![];
        for ((stream, seq, source_ns), variants) in &table {
            let (Some(a), Some(b)) = (variants.get("static-2s"), variants.get(name)) else {
                continue;
            };
            let (Some(ai), Some(bi)) = (a["selected"].as_u64(), b["selected"].as_u64()) else {
                continue;
            };
            let normal = &b["branches"][bi as usize]["normal_camera_right_down_toward"];
            let score = |k: usize| {
                (0..3)
                    .map(|i| {
                        num(&a["branches"][k]["normal_camera_right_down_toward"][i])
                            * num(&normal[i])
                    })
                    .sum::<f64>()
            };
            let nearest = usize::from(score(1) > score(0));
            count += 1;
            if nearest == ai as usize {
                agree += 1;
            } else {
                disagreements.push(json!({"stream":stream,"sequence":seq,"source_ns":source_ns,"baseline_center":a["eye_center_sensor"],"other_center":b["eye_center_sensor"],"baseline_normal":a["branches"][ai as usize]["normal_camera_right_down_toward"],"other_normal":normal}));
            }
        }
        reports.push(json!({"variant":name,"both_select":count,"agree_after_normal_matching":agree,"disagreements":disagreements}));
    }
    fs::write(
        out.join("sensitivity.json"),
        serde_json::to_vec_pretty(
            &json!({"scope":"Agreement is sensitivity, not correctness; align actual normal candidates rather than array indices.","comparisons":reports}),
        )?,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
struct Transport {
    source: P,
    current: P,
    c: f64,
    s: f64,
    error: f64,
}
impl Transport {
    fn from(v: &Value) -> Option<Self> {
        let r = &v["outer"]["rigid_2d"];
        if uint(&v["outer"]["point_count"]) < 8 || !r.is_object() {
            return None;
        }
        let angle = num(&r["rotation_degrees"]).to_radians();
        Some(Self {
            source: [num(&r["source_centroid"][0]), num(&r["source_centroid"][1])],
            current: [
                num(&r["current_centroid"][0]),
                num(&r["current_centroid"][1]),
            ],
            c: angle.cos(),
            s: angle.sin(),
            error: num(&r["evaluation_median_euclidean_px"]),
        })
    }
    fn inverse(self, p: P) -> P {
        let p = [p[0] - self.current[0], p[1] - self.current[1]];
        [
            self.source[0] + self.c * p[0] + self.s * p[1],
            self.source[1] - self.s * p[0] + self.c * p[1],
        ]
    }
    fn forward(self, p: P) -> P {
        let p = [p[0] - self.source[0], p[1] - self.source[1]];
        [
            self.current[0] + self.c * p[0] - self.s * p[1],
            self.current[1] + self.s * p[0] + self.c * p[1],
        ]
    }
    fn line(self, mut l: NormalLine) -> NormalLine {
        l.point = self.inverse(l.point);
        l.direction = [
            self.c * l.direction[0] + self.s * l.direction[1],
            -self.s * l.direction[0] + self.c * l.direction[1],
        ];
        l
    }
}
/// Bounded source-matched audit using independent top/bottom texture motion.
/// This transports image lines; it never claims a measured 3D head rotation.
pub fn transport_audit(base: &Path, motion_path: &Path, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("use new output".into());
    }
    if !out
        .parent()
        .ok_or("missing parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let motion: Value = serde_json::from_slice(&fs::read(motion_path)?)?;
    let tracks: Value = serde_json::from_slice(&fs::read(
        motion["input"].as_str().ok_or("missing motion input")?,
    )?)?;
    let selected = tracks["series"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .ok_or("missing series")?;
    let metadata = selected["frames"].as_array().unwrap();
    let shapes = read(&base.join("expanded-corpus/admitted-shapes.jsonl"))?
        .into_iter()
        .filter(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .map(|s| (uint(&s["sequence"]), s))
        .collect::<BTreeMap<_, _>>();
    let camera = PinholeCamera {
        focal_px: [4000.; 2],
        principal_px: [4000., 3000.],
    };
    let mut results = vec![];
    let mut reviews = vec![];
    for (name, aligned) in [("matched-static-2s", false), ("outer-aligned-2s", true)] {
        let mut history = vec![];
        let mut errors = vec![];
        for (j, m) in motion["frames"].as_array().unwrap().iter().enumerate() {
            assert_eq!(m["raw_sha256"], metadata[j]["raw_sha256"]);
            assert_eq!(m["sequence"], metadata[j]["sequence"]);
            let seq = uint(&m["sequence"]);
            let source = shapes.get(&seq).ok_or("missing source shape")?;
            assert_eq!(source["clock"], metadata[j]["input"]["clock_lineage"]);
            let time = uint(&source["source_ns"]) as f64 * 1e-9;
            history.retain(|l: &NormalLine| time - l.seconds <= 2.);
            errors.retain(|(t, _): &(f64, f64)| time - t <= 2.);
            let mut r = json!({"stream":"later-recording/eye2","variant":name,"sequence":seq,"source_ns":source["source_ns"],"status":"missing independent transport or ellipse"});
            if source["ellipse"].is_object() {
                if let Some(t) = Transport::from(m) {
                    let e = ellipse(&source["ellipse"]);
                    if let Some(mut center) = projected_eye_center::fit(&history, time, false) {
                        if aligned {
                            center.center = t.forward(center.center);
                            center.support_radius_px += 2. * t.error
                                + 2. * errors.iter().map(|(_, e)| e).copied().fold(0., f64::max);
                        }
                        let d = decision(camera, e, center);
                        for (k, v) in d.as_object().unwrap() {
                            r[k] = v.clone();
                        }
                        r["outer_transport_median_error_px"] = json!(t.error);
                        if aligned && r["selected"].is_u64() {
                            let mut review = r.clone();
                            let mut s = source.clone();
                            s["input"] = metadata[j]["input"].clone();
                            review["source"] = s;
                            review["provider"] = source["provider"].clone();
                            review["eye"] = json!(2);
                            review["stream"] = json!(format!("source-{seq}"));
                            reviews.push(review);
                        }
                    } else {
                        r["status"] = json!("insufficient matched past support");
                    }
                    if let Some(poses) = circle_pose_hypotheses(camera, e, [0, 0]) {
                        if let Some(mut l) = projected_line(camera, poses[0]) {
                            l.seconds = time;
                            let sep = poses[0]
                                .normal
                                .iter()
                                .zip(poses[1].normal)
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                .clamp(-1., 1.)
                                .acos();
                            l.weight = (sep.sin().powi(2) / 0.25).clamp(0.01, 1.);
                            history.push(if aligned { t.line(l) } else { l });
                            errors.push((time, t.error));
                        }
                    }
                }
            }
            results.push(r);
        }
    }
    fs::create_dir(out)?;
    let mut writer = std::io::BufWriter::new(fs::File::create(out.join("decisions.jsonl"))?);
    for r in &results {
        writeln!(writer, "{}", r)?;
    }
    writer.flush()?;
    let summary = ["matched-static-2s", "outer-aligned-2s"].map(|name| {
        let mut counts = BTreeMap::new();
        for r in &results {
            if r["variant"] == name {
                *counts.entry(r["status"].as_str().unwrap()).or_insert(0) += 1;
            }
        }
        json!({"variant":name,"status_counts":counts})
    });
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"input":motion_path,"frame_count":metadata.len(),"scope":"Same RAW exposures and same past-only two-second ellipse histories; compare stationary image centre against lines transported by independently measured outer top/bottom image rotation and translation. Offline pixel tracks use a common reference template, including frames before that reference. No metric head pose and no true sign labels.","summaries":summary}),
        )?,
    )?;
    fs::write(
        out.join("review-selection.json"),
        serde_json::to_vec_pretty(&reviews)?,
    )?;
    render(&reviews, out)?;
    println!("{}", json!(summary));
    Ok(())
}
