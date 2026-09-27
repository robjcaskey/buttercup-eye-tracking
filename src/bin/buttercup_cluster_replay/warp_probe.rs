//! Isolated native-RAW patch-deformation/shared-3D-pivot experiment.
//! Source-matched ablations, rejection logs, synthetic truth and a RAW viewer.
//! This does not change production tracking or promote anatomical labels.
use super::{json, quantiles, raw_preview, BundleSource, Result, Value};
use buttercup_eye_tracking::raw10;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};
#[path = "warp_math.rs"]
mod math;
use math::*;
#[path = "iris_pivot.rs"]
mod iris_pivot;
pub fn z_motion3d_run(args: &[String]) -> Result<()> {
    iris_pivot::z_motion3d::run(args)
}
pub fn z_discovery_run(args: &[String]) -> Result<()> {
    iris_pivot::z_discovery::run(args)
}
pub fn layers_run(args: &[String]) -> Result<()> {
    iris_pivot::layers::run(args)
}
pub fn iris_run(args: &[String]) -> Result<()> {
    iris_pivot::run(args)
}
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
use canvas::*;
const NAMES: [&str; 3] = [
    "Translation patches",
    "Affine rescue",
    "Affine rescue + shared 3D",
];
#[derive(Clone)]
struct Track {
    p: P,
    velocity: P,
}
struct Tracker {
    tracks: Vec<Option<Track>>,
}
impl Tracker {
    fn new(seeds: &[P]) -> Self {
        Self {
            tracks: seeds
                .iter()
                .map(|&p| {
                    Some(Track {
                        p,
                        velocity: [0., 0.],
                    })
                })
                .collect(),
        }
    }
}
struct Input {
    image: Image,
    rgb: Option<[Image; 3]>,
    preview: Vec<u8>,
    meta: Value,
    hash: String,
    time: f64,
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn origin(v: &Value) -> P {
    [
        v["sensor_x"].as_f64().unwrap(),
        v["sensor_y"].as_f64().unwrap(),
    ]
}
fn load(bundle: &BundleSource, meta: &Value, first: u64) -> Result<Input> {
    let n = |k: &str| meta[k].as_u64().ok_or_else(|| format!("missing {k}"));
    let w = n("width")? as usize;
    let h = n("height")? as usize;
    if w * h > 1024 * 1024 || w < 64 || h < 64 {
        return Err("bounded native crop required".into());
    }
    let packed = bundle.read_range(
        meta["stream"].as_str().ok_or("stream")?,
        n("offset")?,
        n("length")? as usize,
    )?;
    let raw = raw10::try_unpack_raw10(&packed, w, h, n("stride")? as usize)?;
    let rgb = raw_preview::color_preview(
        &raw,
        w,
        h,
        n("sensor_x")? as u32,
        n("sensor_y")? as u32,
        100,
        None,
    );
    let preview = rgb
        .into_iter()
        .flat_map(|p| [p as u8, (p >> 8) as u8, (p >> 16) as u8, 255])
        .collect();
    Ok(Input {
        image: Image::raw(&raw, w, h),
        rgb: Some(iris_pivot::color::raw_rgb(
            &raw,
            w,
            h,
            n("sensor_x")? as usize,
            n("sensor_y")? as usize,
        )),
        preview,
        meta: meta.clone(),
        hash: digest(&packed),
        time: (n("timestamp_ns")? - first) as f64 / 1e9,
    })
}
fn image(out: &Path, frame: usize, f: &Input) -> Result<()> {
    let mut c = Canvas::new(f.image.w, f.image.h)?;
    c.image(
        &f.preview,
        f.image.w,
        f.image.h,
        0.,
        0.,
        f.image.w as f64,
        f.image.h as f64,
    );
    c.png(&out.join(format!("raw-{frame:03}.png")))
}
fn step(
    tracker: &mut Tracker,
    previous: &Input,
    current: &Input,
    variant: usize,
    min_ncc: f64,
) -> (Value, Vec<Sphere>) {
    let start = Instant::now();
    let a = origin(&previous.meta);
    let b = origin(&current.meta);
    let delta = [a[0] - b[0], a[1] - b[1]];
    let mut hits = Vec::new();
    let mut audits = Vec::new();
    for track in &tracker.tracks {
        if let Some(t) = track {
            let predicted = [
                t.p[0] + delta[0] + 0.65 * t.velocity[0],
                t.p[1] + delta[1] + 0.65 * t.velocity[1],
            ];
            let (h, a) = independent(
                &previous.image,
                &current.image,
                t.p,
                predicted,
                variant != 0,
                min_ncc,
            );
            hits.push(h);
            audits.push(Some(a));
        } else {
            hits.push(None);
            audits.push(None);
        }
    }
    let mut models = Vec::new();
    let mut field_errors = Vec::new();
    if variant == 2 {
        let pairs: Vec<_> = hits
            .iter()
            .enumerate()
            .filter_map(|(id, h)| {
                let h = h.as_ref()?;
                let p = tracker.tracks[id].as_ref()?.p;
                (h.ncc >= min_ncc.max(0.70) && h.fb < 0.6).then_some(Pair {
                    id,
                    p,
                    q: [h.q[0] - delta[0], h.q[1] - delta[1]],
                })
            })
            .collect();
        let mut unique: Vec<Pair> = Vec::new();
        for p in pairs {
            if unique.iter().all(|q| distance(p.q, q.q) >= 3.5) {
                unique.push(p);
            }
        }
        for parity in 0..2 {
            let base = sphere_models(&unique, previous.image.w, previous.image.h, parity);
            // Image origins are acquisition metadata, never an estimated motion.
            let shifted: Vec<_> = base
                .iter()
                .map(|s| Sphere {
                    shift: [s.shift[0] + delta[0], s.shift[1] + delta[1]],
                    ..*s
                })
                .collect();
            for (id, track) in tracker.tracks.iter().enumerate() {
                if id % 2 != parity {
                    continue;
                }
                let Some(t) = track else { continue };
                if let Some(h) = &hits[id] {
                    let errors: Vec<_> = shifted
                        .iter()
                        .filter_map(|s| s.map(t.p))
                        .map(|p| distance(p, h.q))
                        .collect();
                    if let Some(e) = errors.into_iter().min_by(f64::total_cmp) {
                        field_errors.push(e);
                    }
                }
                let (h, a) = guided(
                    &previous.image,
                    &current.image,
                    t.p,
                    hits[id].clone(),
                    &shifted,
                    min_ncc,
                );
                if h.as_ref().is_some_and(|h| h.source == "shared_3d")
                    || (hits[id].is_none() && !shifted.is_empty())
                {
                    hits[id] = h;
                    audits[id] = Some(a);
                }
            }
            models.extend(base);
        }
    }
    // One-to-one target assignments. Consistent duplicate destinations do not
    // become extra independent evidence merely because they have different IDs.
    let mut order: Vec<_> = hits
        .iter()
        .enumerate()
        .filter_map(|(i, h)| h.as_ref().map(|h| (i, h.ncc)))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut used = Vec::new();
    for (i, _) in order {
        let q = hits[i].as_ref().unwrap().q;
        if used.iter().any(|&p| distance(p, q) < 3.5) {
            hits[i] = None;
            if let Some(a) = &mut audits[i] {
                a.accepted = false;
                a.reason = "destination_collision".into();
            }
        } else {
            used.push(q);
        }
    }
    let mut rows = Vec::new();
    for (id, track) in tracker.tracks.iter_mut().enumerate() {
        let Some(old) = track.clone() else { continue };
        let audit = audits[id].as_ref().unwrap();
        let hit = hits[id].as_ref();
        rows.push(json!({"id":id,"from":old.p,"hit":hit,"audit":audit}));
        *track = hit.map(|h| Track {
            p: h.q,
            velocity: [
                0.5 * old.velocity[0] + 0.5 * (h.q[0] - old.p[0] - delta[0]),
                0.5 * old.velocity[1] + 0.5 * (h.q[1] - old.p[1] - delta[1]),
            ],
        });
    }
    (
        json!({"name":NAMES[variant],"rows":rows,"alive":tracker.tracks.iter().filter(|t|t.is_some()).count(),"ms":start.elapsed().as_secs_f64()*1000.,"independent_witness_prediction_error_px":quantiles(field_errors),"models":models}),
        models,
    )
}
fn frame_row(i: usize, f: &Input, variants: Vec<Value>) -> Value {
    json!({"index":i,"sequence":f.meta["sequence"],"timestamp_ns":f.meta["timestamp_ns"].as_u64().unwrap().to_string(),"time_s":f.time,"origin":origin(&f.meta),"raw_sha256":f.hash,"image":format!("raw-{i:03}.png"),"variants":variants})
}
fn start_rows(seeds: &[P]) -> Vec<Value> {
    (0..3).map(|v|json!({"name":NAMES[v],"alive":seeds.len(),"ms":0.,"models":[],"rows":seeds.iter().enumerate().map(|(id,p)|json!({"id":id,"from":p,"hit":{"q":p,"ncc":1.,"fb":0.,"source":"seed","witnesses":0},"audit":{"accepted":true,"reason":"seed"}})).collect::<Vec<_>>()})).collect()
}
fn summary(frames: &[Value], seed_count: usize) -> Value {
    let variants=(0..3).map(|v|{
        let mut rejects=std::collections::BTreeMap::<String,usize>::new();let mut correlations=Vec::new();let mut fb=Vec::new();let mut shared=0;let mut times=Vec::new();let mut observations=0;
        for f in frames.iter().skip(1){let r=&f["variants"][v];times.push(r["ms"].as_f64().unwrap());for p in r["rows"].as_array().unwrap(){if p["hit"].is_null(){*rejects.entry(p["audit"]["reason"].as_str().unwrap().into()).or_default()+=1;}else{observations+=1;shared+=usize::from(p["hit"]["source"]=="shared_3d");correlations.push(p["hit"]["ncc"].as_f64().unwrap());fb.push(p["hit"]["fb"].as_f64().unwrap());}}}
        json!({"name":NAMES[v],"initial":seed_count,"survive_all":frames.last().unwrap()["variants"][v]["alive"],"observations":observations,"shared_3d_observations":shared,"held_pixel_ncc":quantiles(correlations),"forward_backward_error_px":quantiles(fb),"rejections":rejects,"ms":quantiles(times)})
    }).collect::<Vec<_>>();
    json!(variants)
}
fn sheet(out: &Path, frames: &[Value], f: &Input, seeds: &[P], name: &str) -> Result<()> {
    let width = f.image.w;
    let height = f.image.h;
    let mut c = Canvas::new(width * 3 + 40, height + 135)?;
    c.clear();
    c.text(12., 24., 18., WHITE, name);
    let row = frames.last().unwrap();
    for v in 0..3 {
        let x = 10. + v as f64 * (width as f64 + 10.);
        c.text(
            x,
            52.,
            14.,
            WHITE,
            &format!(
                "{}: {}/{} survive",
                NAMES[v],
                row["variants"][v]["alive"],
                seeds.len()
            ),
        );
        c.image(
            &f.preview,
            width,
            height,
            x,
            68.,
            width as f64,
            height as f64,
        );
        for p in row["variants"][v]["rows"].as_array().unwrap() {
            let h = &p["hit"];
            if h.is_null() {
                continue;
            }
            let q = &h["q"];
            let color = if h["source"] == "shared_3d" {
                PINK
            } else {
                CYAN
            };
            c.dot(
                x + q[0].as_f64().unwrap(),
                68. + q[1].as_f64().unwrap(),
                2.7,
                color,
                false,
            );
        }
    }
    c.text(12.,height as f64+103.,13.,MUTED,"Measured patch matches. Pink: accepted shared-rotation proposal. IDs and photometric consistency are not tissue ground truth.");
    c.png(&out.join("final-overlay.png"))
}
fn write_review(
    out: &Path,
    report: &Value,
    frames: &[Value],
    seeds: &[P],
    w: usize,
    h: usize,
) -> Result<()> {
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(report)?)?;
    let data = json!({"report":report,"frames":frames,"seeds":seeds,"width":w,"height":h});
    fs::write(out.join("review.json"), serde_json::to_vec(&data)?)?;
    let html =
        include_str!("warp_viewer.html").replace("REVIEW_DATA", &serde_json::to_string(&data)?);
    fs::write(out.join("viewer.html"), html)?;
    Ok(())
}
fn matched_truth(frames: &[Value]) -> Value {
    let mut errors: [Vec<f64>; 3] = Default::default();
    let mut shared_errors = Vec::new();
    for f in frames.iter().skip(1) {
        let maps: Vec<std::collections::BTreeMap<usize, f64>> = (0..3)
            .map(|v| {
                f["variants"][v]["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|r| {
                        Some((r["id"].as_u64()? as usize, r["truth_error_px"].as_f64()?))
                    })
                    .collect()
            })
            .collect();
        for (&id, &e) in &maps[0] {
            if maps[1].contains_key(&id) && maps[2].contains_key(&id) {
                errors[0].push(e);
                errors[1].push(maps[1][&id]);
                errors[2].push(maps[2][&id]);
            }
        }
        for r in f["variants"][2]["rows"].as_array().unwrap() {
            if r["hit"]["source"] == "shared_3d" {
                if let Some(e) = r["truth_error_px"].as_f64() {
                    shared_errors.push(e);
                }
            }
        }
    }
    json!({"same_frame_and_seed_count":errors[0].len(),"variants":(0..3).map(|v|json!({"name":NAMES[v],"error_px":quantiles(errors[v].clone()),"wrong_over_2px":errors[v].iter().filter(|&&e|e>2.).count()})).collect::<Vec<_>>(),"shared_3d_only":{"scored":shared_errors.len(),"wrong_over_2px":shared_errors.iter().filter(|&&e|e>2.).count(),"error_px":quantiles(shared_errors)},"scope":"Compare identical visible seed/frame pairs observed by all three variants. Material outside the rendered sphere cutoff or image is excluded from coordinate error and counted separately as accepted_not_rendered. Additional candidate observations remain in the separate all-support report; this intersection alone cannot reward added false matches."})
}
pub fn run(args: &[String]) -> Result<()> {
    if !(7..=8).contains(&args.len()) {
        return Err("usage: --warp-probe BUNDLE NEW_OUTPUT EYE FIRST_FRAME LAST_FRAME [MIN_NCC=0.65] (inclusive, max 120 frames); BUNDLE=synthetic for a known-motion control".into());
    }
    let out = Path::new(&args[3]);
    let eye: usize = args[4].parse()?;
    let first: usize = args[5].parse()?;
    let last: usize = args[6].parse()?;
    let min_ncc: f64 = args.get(7).map(|s| s.parse()).transpose()?.unwrap_or(0.65);
    if out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
        || !(1..=2).contains(&eye)
        || last <= first
        || last - first > 119
        || !min_ncc.is_finite()
        || !(0.5..=0.95).contains(&min_ncc)
    {
        return Err("new checked output, eye 1/2 and 2..120 frames required".into());
    }
    fs::create_dir(out)?;
    if args[2] == "synthetic" {
        return synthetic(out, min_ncc);
    }
    let source = Path::new(&args[2]);
    let bundle = BundleSource::open(source)?;
    let text = String::from_utf8(bundle.read_entry("frames.jsonl")?)?;
    let metas = text
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<std::result::Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|v| v["eye_id"] == eye as u64)
        .collect::<Vec<_>>();
    if last >= metas.len() {
        return Err("source interval unavailable".into());
    }
    let stamp = metas[0]["timestamp_ns"].as_u64().ok_or("source clock")?;
    let mut previous = load(&bundle, &metas[first], stamp)?;
    let seeds = seeds(&previous.image, 180);
    let mut trackers: [Tracker; 3] = std::array::from_fn(|_| Tracker::new(&seeds));
    let mut frames = vec![frame_row(0, &previous, start_rows(&seeds))];
    image(out, 0, &previous)?;
    let mut log = fs::File::create(out.join("frames.jsonl"))?;
    writeln!(log, "{}", frames[0])?;
    let began = Instant::now();
    for i in first + 1..=last {
        let current = load(&bundle, &metas[i], stamp)?;
        let a = previous.meta["timestamp_ns"].as_u64().unwrap();
        let b = current.meta["timestamp_ns"].as_u64().unwrap();
        if b <= a
            || b - a > 100_000_000
            || current.image.w != previous.image.w
            || current.image.h != previous.image.h
            || previous.meta["source_clock"]["source_key"]["stream_epoch"]
                != current.meta["source_clock"]["source_key"]["stream_epoch"]
        {
            return Err("non-contiguous source interval".into());
        }
        let variants = (0..3)
            .map(|v| step(&mut trackers[v], &previous, &current, v, min_ncc).0)
            .collect();
        image(out, i - first, &current)?;
        let f = frame_row(i - first, &current, variants);
        writeln!(log, "{f}")?;
        frames.push(f);
        previous = current;
        if (i - first) % 10 == 0 {
            eprintln!(
                "eye {eye} frame {i}/{last}: alive {:?}",
                trackers
                    .iter()
                    .map(|t| t.tracks.iter().filter(|p| p.is_some()).count())
                    .collect::<Vec<_>>()
            );
        }
    }
    let report = json!({"affine_policy":"rescue_after_translation_failure","min_ncc":min_ncc,"schema":"raw-warp-shared-pivot-experiment-v1","bundle":source,"eye":eye,"first_frame":first,"last_frame":last,"frame_count":frames.len(),"start_s":frames[0]["time_s"],"end_s":frames.last().unwrap()["time_s"],"seeds":seeds.len(),"variants":summary(&frames,seeds.len()),"seconds":began.elapsed().as_secs_f64(),"provenance":{"recipe_sha256":digest(include_bytes!("warp_probe.rs")),"math_sha256":digest(include_bytes!("warp_math.rs")),"viewer_sha256":digest(include_bytes!("warp_viewer.html")),"frame_inventory_sha256":digest(text.as_bytes())},"scope":"Rob-only bounded monocular native RAW. Same spatially balanced initial corners, preprocessing, translation-search budget and image validation gates; affine rescue adds shape freedom only after both simple translation basins fail; shared variant proposes exact 3D sphere rotations from other identities. Two identity folds exclude each query from its own pivot evidence. Generic sphere centers/radii are nuisance hypotheses, not measured globe geometry. No iris detector, labels, learned weights, recorded predictions, calibration targets, or metric depth. No reseeding, held positions or reidentification counted as survival. New experimental translation baseline is not the production matcher. Pixel checkerboard withheld from optimization is correlated image evidence, not anatomical truth. No limbus estimate/independent scale: SN-FEIDA not applicable. No FPS or cross-user readiness claim."});
    sheet(
        out,
        &frames,
        &previous,
        &seeds,
        &format!(
            "Eye {eye} / native RAW / {:.3} to {:.3} s",
            frames[0]["time_s"].as_f64().unwrap(),
            previous.time
        ),
    )?;
    write_review(
        out,
        &report,
        &frames,
        &seeds,
        previous.image.w,
        previous.image.h,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["variants"])?);
    Ok(())
}
#[derive(Debug, PartialEq)]
enum SyntheticTruth {
    Visible(P),
    NotRendered,
    ExcludedInitialBoundary,
}
fn synthetic_truth(initial: P, s: Sphere, w: usize, h: usize) -> SyntheticTruth {
    let now_center = [s.center[0] + s.shift[0], s.center[1] + s.shift[1]];
    let in_image =
        |p: P| p[0] >= 0. && p[1] >= 0. && p[0] < (w - 1) as f64 && p[1] < (h - 1) as f64;
    if distance(initial, s.center) < s.radius * 0.88 {
        // The renderer deliberately cuts the hemisphere at 0.96r. A point
        // may still be geometrically on the front surface after leaving that
        // rendered support. Do not score its nonexistent image coordinate.
        match s.map(initial) {
            Some(p) if in_image(p) && distance(p, now_center) < s.radius * 0.96 => {
                SyntheticTruth::Visible(p)
            }
            _ => SyntheticTruth::NotRendered,
        }
    } else if distance(initial, s.center) > s.radius * 1.08 {
        if in_image(initial) && distance(initial, now_center) >= s.radius * 0.96 {
            SyntheticTruth::Visible(initial)
        } else {
            SyntheticTruth::NotRendered
        }
    } else {
        SyntheticTruth::ExcludedInitialBoundary
    }
}
fn synthetic(out: &Path, min_ncc: f64) -> Result<()> {
    let (w, h) = (420, 280);
    let center = [210., 154.];
    let radius = 151.2;
    let texture = |p: P| {
        0.45 + 0.10 * (p[0] * 0.15 + p[1] * 0.11).sin()
            + 0.09 * (p[0] * 0.34 - p[1] * 0.19).cos()
            + 0.08 * (p[0] * 0.065 + p[1] * 0.42).sin()
            + 0.035 * ((p[0] * 0.011).sin() * 35. + (p[1] * 0.013).cos() * 21.).sin()
    };
    let sphere = |i: usize| Sphere {
        center,
        radius,
        shift: [i as f64 * 0.22, i as f64 * (-0.12)],
        omega: [i as f64 * 0.014, i as f64 * (-0.027), i as f64 * 0.008],
        support: 0,
        residual: 0.,
    };
    let frame = |i: usize| {
        let s = sphere(i);
        let reverse = Sphere {
            center: [center[0] + s.shift[0], center[1] + s.shift[1]],
            shift: s.shift.map(|v| -v),
            omega: s.omega.map(|v| -v),
            ..s
        };
        let im = Image {
            w,
            h,
            v: (0..w * h)
                .map(|k| {
                    let p = [(k % w) as f64, (k / w) as f64];
                    if distance(p, reverse.center) < radius * 0.96 {
                        reverse.map(p).map(texture).unwrap_or(0.42)
                    } else {
                        texture([p[0] + 41., p[1] - 61.]) * 0.85
                    }
                })
                .collect::<Vec<_>>(),
        };
        let preview =
            im.v.iter()
                .flat_map(|v| {
                    let b = (v * 255.).clamp(0., 255.) as u8;
                    [b, b, b, 255]
                })
                .collect();
        Input {
            rgb: None,
            image: im,
            preview,
            hash: format!("synthetic-{i}"),
            time: i as f64 / 33.383,
            meta: json!({"sensor_x":0,"sensor_y":0,"sequence":i,"timestamp_ns":i as u64*30_000_000+1}),
        }
    };
    let mut previous = frame(0);
    let seeds = seeds(&previous.image, 180);
    let mut trackers: [Tracker; 3] = std::array::from_fn(|_| Tracker::new(&seeds));
    let mut frames = vec![frame_row(0, &previous, start_rows(&seeds))];
    image(out, 0, &previous)?;
    let mut errors: [Vec<f64>; 3] = Default::default();
    let mut wrong = [0usize; 3];
    let mut visible = [0usize; 3];
    let mut not_rendered = [0usize; 3];
    let mut shared_not_rendered = [0usize; 3];
    for i in 1..=18 {
        let current = frame(i);
        let mut variants = Vec::new();
        for v in 0..3 {
            let (mut result, _) = step(&mut trackers[v], &previous, &current, v, min_ncc);
            for r in result["rows"].as_array_mut().unwrap() {
                if r["hit"].is_null() {
                    continue;
                }
                let id = r["id"].as_u64().unwrap() as usize;
                let initial = seeds[id];
                let truth = synthetic_truth(initial, sphere(i), w, h);
                if let SyntheticTruth::Visible(p) = truth {
                    let q = [
                        r["hit"]["q"][0].as_f64().unwrap(),
                        r["hit"]["q"][1].as_f64().unwrap(),
                    ];
                    let e = distance(p, q);
                    r["truth_status"] = json!("visible");
                    r["truth_point"] = json!(p);
                    r["truth_error_px"] = json!(e);
                    errors[v].push(e);
                    wrong[v] += usize::from(e > 2.);
                    visible[v] += 1;
                } else if truth == SyntheticTruth::NotRendered {
                    r["truth_status"] = json!("not_rendered");
                    not_rendered[v] += 1;
                    shared_not_rendered[v] += usize::from(r["hit"]["source"] == "shared_3d");
                } else {
                    r["truth_status"] = json!("excluded_initial_boundary");
                }
            }
            variants.push(result);
        }
        image(out, i, &current)?;
        frames.push(frame_row(i, &current, variants));
        previous = current;
    }
    let report = json!({"affine_policy":"rescue_after_translation_failure","min_ncc":min_ncc,"schema":"raw-warp-shared-pivot-synthetic-v2","provenance":{"recipe_sha256":digest(include_bytes!("warp_probe.rs")),"math_sha256":digest(include_bytes!("warp_math.rs")),"viewer_sha256":digest(include_bytes!("warp_viewer.html"))},"matched_truth":matched_truth(&frames),"eye":"synthetic","seeds":seeds.len(),"variants":summary(&frames,seeds.len()),"truth":(0..3).map(|v|json!({"name":NAMES[v],"scored_observations":visible[v],"wrong_over_2px":wrong[v],"accepted_not_rendered":not_rendered[v],"shared_3d_accepted_not_rendered":shared_not_rendered[v],"endpoint_error_px":quantiles(errors[v].clone())})).collect::<Vec<_>>(),"scope":"Known out-of-plane sphere rotation plus separately stationary textured background. Initial boundary seeds excluded. Coordinate errors cover currently rendered, visible material only; accepted identities after material leaves the renderer cutoff/image or becomes occluded are separate failures, not successful fresh measurements. Procedural texture is not native RAW, does not reproduce eye optics, specularities or camera noise, and is not an anatomical validation. More tracks are useful only with low truth error."});
    sheet(
        out,
        &frames,
        &previous,
        &seeds,
        "Synthetic / known 3D rotation plus stationary background",
    )?;
    write_review(out, &report, &frames, &seeds, w, h)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod truth_tests {
    use super::*;
    #[test]
    fn synthetic_truth_respects_renderer_cutoff_and_occlusion() {
        let s = Sphere {
            center: [210., 154.],
            radius: 151.2,
            shift: [0.; 2],
            omega: [0.; 3],
            support: 0,
            residual: 0.,
        };
        assert_eq!(
            synthetic_truth(s.center, s, 420, 280),
            SyntheticTruth::Visible(s.center)
        );
        assert_eq!(
            synthetic_truth([350., 154.], s, 420, 280),
            SyntheticTruth::ExcludedInitialBoundary
        );
        let rotated = Sphere {
            omega: [0., -0.4, 0.],
            ..s
        };
        assert!(rotated.map([330., 154.]).is_some());
        assert_eq!(
            synthetic_truth([330., 154.], rotated, 420, 280),
            SyntheticTruth::NotRendered
        );
        assert_eq!(
            synthetic_truth(
                s.center,
                Sphere {
                    shift: [400., 0.],
                    ..s
                },
                420,
                280
            ),
            SyntheticTruth::NotRendered
        );
        assert_eq!(
            synthetic_truth(
                [400., 154.],
                Sphere {
                    shift: [150., 0.],
                    ..s
                },
                420,
                280
            ),
            SyntheticTruth::NotRendered
        );
    }
}
