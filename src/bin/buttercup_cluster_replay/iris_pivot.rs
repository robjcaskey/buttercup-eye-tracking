//! Bounded exhaustive iris-only pivot experiment; no live tracker changes.
use super::super::{json, quantiles, BundleSource, Value};
use super::{digest, image, load, math, origin, Canvas, Input, Result, CYAN, MUTED, PINK, WHITE};
use math::{distance, Image, Warp, P};
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, path::Path, time::Instant};
#[path = "iris_pivot_math.rs"]
mod geometry;
use geometry::*;
#[path = "iris_pivot_prior.rs"]
mod prior;
use prior::{MotionPrior, Search};
#[path = "iris_pivot_binary.rs"]
mod binary;
#[path = "iris_pivot_color.rs"]
pub(super) mod color;
#[path = "iris_layers.rs"]
pub(super) mod layers;
#[path = "z_discovery.rs"]
pub(super) mod z_discovery;
#[path = "z_motion3d.rs"]
pub(super) mod z_motion3d;
fn unit_scale() -> f64 {
    1.
}

#[derive(Deserialize, Serialize)]
struct Config {
    eye: usize,
    first: usize,
    last: usize,
    center: P,
    radii: P,
    fit_frames: Vec<usize>,
    #[serde(default)]
    smoothing_radius: usize,
    #[serde(default)]
    motion_prior: Option<MotionPrior>,
    #[serde(default)]
    binary_search: Option<binary::Config>,
    #[serde(default)]
    baseline_review: Option<String>,
    #[serde(default)]
    lenient_comparison: bool,
    #[serde(default = "unit_scale")]
    synthetic_motion_scale: f64,
    #[serde(default)]
    synthetic_radial_plane: bool,
    #[serde(default)]
    synthetic_translation_px: P,
}
#[derive(Clone, Serialize)]
struct Pose {
    omega: [f64; 3],
    warp: Affine,
    train: Score,
    held: Score,
    translation: P,
    plane_gradient: P,
    objective: f64,
    prior_penalty: f64,
    at_limit: bool,
    binary_trace: Vec<binary::Step>,
    binary_score_evaluations: usize,
    relative_color_score: Option<Score>,
}
#[derive(Clone, Serialize)]
struct Profile {
    pivot: [f64; 3],
    loss: f64,
    held_loss: f64,
    texture_loss: f64,
    pivot_prior_penalty: f64,
    poses: Vec<Pose>,
    stage: &'static str,
}
fn pose(
    train: &Samples,
    held: &Samples,
    f: &Features,
    c: [f64; 3],
    delta: P,
    start: Option<([f64; 3], P)>,
    angles: &[[f64; 3]],
    search: Option<&Search<'_>>,
    pyramid: Option<(&binary::Pyramid, usize)>,
) -> Pose {
    if let Some(search) = search {
        let (fit, binary_trace, binary_score_evaluations) = if let Some((p, frame)) = pyramid {
            p.fit(frame, search, train, f, c, delta, start)
        } else {
            (search.fit(train, f, c, delta, angles, start), Vec::new(), 0)
        };
        let warp = search.warp(c, fit.omega, delta, fit.translation);
        return Pose {
            omega: fit.omega,
            warp,
            train: geometry::score(train, f, warp),
            relative_color_score: pyramid
                .and_then(|(p, frame)| p.native_relative.as_ref().map(|r| r.score(frame, warp))),
            held: geometry::score(held, f, warp),
            translation: fit.translation,
            plane_gradient: search.gradient(c),
            objective: fit.objective,
            prior_penalty: fit.objective - fit.score.loss,
            at_limit: search.at_limit(fit.omega, fit.translation),
            binary_trace,
            binary_score_evaluations,
        };
    }
    let (omega, score) = match start {
        Some((w, _)) => refine_rotation(train, f, c, delta, w),
        None => brute_rotation(train, f, c, delta, angles),
    };
    let warp = project(c, omega, delta);
    Pose {
        omega,
        warp,
        train: score,
        held: geometry::score(held, f, warp),
        translation: [0.; 2],
        plane_gradient: [0.; 2],
        objective: score.loss,
        prior_penalty: 0.,
        at_limit: false,
        binary_trace: Vec::new(),
        binary_score_evaluations: 0,
        relative_color_score: None,
    }
}
fn delta(a: &Input, b: &Input) -> P {
    let a = origin(&a.meta);
    let b = origin(&b.meta);
    [a[0] - b[0], a[1] - b[1]]
}
fn profiles(
    train: &Samples,
    held: &Samples,
    inputs: &[Input],
    features: &[Features],
    fit: &[usize],
    candidates: &[([f64; 3], Option<Vec<([f64; 3], P)>>)],
    stage: &'static str,
    search: Option<&Search<'_>>,
    pyramid: Option<&binary::Pyramid>,
) -> Vec<Profile> {
    let angles = if pyramid.is_some() {
        Vec::new()
    } else {
        search.map_or_else(grid_angles, |s| s.prior.angles())
    };
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|worker| {
                let angles = &angles;
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for k in (worker..candidates.len()).step_by(2) {
                        let (pivot, start) = &candidates[k];
                        let poses: Vec<_> = fit
                            .iter()
                            .enumerate()
                            .map(|(j, &i)| {
                                let mut p = pose(
                                    train,
                                    held,
                                    &features[i],
                                    *pivot,
                                    delta(&inputs[0], &inputs[i]),
                                    start.as_ref().map(|s| s[j]),
                                    angles,
                                    search,
                                    pyramid.map(|p| (p, i)),
                                );
                                p.binary_trace.clear();
                                p
                            })
                            .collect();
                        let pivot_prior_penalty = search.map_or(0., |s| s.pivot_penalty(*pivot));
                        out.push(Profile {
                            pivot: *pivot,
                            loss: poses.iter().map(|p| p.objective).sum::<f64>()
                                / poses.len() as f64
                                + pivot_prior_penalty,
                            texture_loss: poses
                                .iter()
                                .map(|p| p.relative_color_score.unwrap_or(p.train).loss)
                                .sum::<f64>()
                                / poses.len() as f64,
                            pivot_prior_penalty,
                            held_loss: poses.iter().map(|p| p.held.loss).sum::<f64>()
                                / poses.len() as f64,
                            poses,
                            stage,
                        });
                        if worker == 0 && k % 100 == 0 {
                            eprintln!("{stage} pivots {k}/{}", candidates.len())
                        }
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("pivot worker panicked"))
            .collect()
    })
}
fn support_image(
    out: &Path,
    input: &Input,
    roi: Region,
    train: &Samples,
    held: &Samples,
    seeds: &[P],
) -> Result<()> {
    let (w, h) = (input.image.w, input.image.h);
    let mut c = Canvas::new(w * 2 + 30, h + 115)?;
    c.clear();
    c.text(
        10.,
        22.,
        17.,
        WHITE,
        "Iris-only support / reference exposure",
    );
    for j in 0..2 {
        c.image(
            &input.preview,
            w,
            h,
            10. + j as f64 * (w as f64 + 10.),
            48.,
            w as f64,
            h as f64,
        )
    }
    for &p in &train.points {
        c.dot(10. + p[0], 48. + p[1], 1.1, CYAN, true)
    }
    for &p in &held.points {
        c.dot(10. + p[0], 48. + p[1], 1.1, PINK, true)
    }
    for &p in seeds {
        c.dot(w as f64 + 20. + p[0], 48. + p[1], 3., CYAN, false)
    }
    c.text(
        10.,
        h as f64 + 74.,
        12.,
        MUTED,
        &format!(
            "Cyan: fit texture; pink: withheld texture. {} iris patch IDs. ROI {:?} / {:?}",
            seeds.len(),
            roi.center,
            roi.radii
        ),
    );
    c.text(10.,h as f64+96.,12.,MUTED,"Conservative visual ROI, not a human anatomical label. Void, bright-reflection halo and outer boundary excluded.");
    c.png(&out.join("reference-support.png"))
}
fn points(
    reference: &Input,
    target: &Input,
    target_features: &Features,
    seeds: &[P],
    m: Affine,
    deform: bool,
    truth: Option<Affine>,
    min_ncc: f64,
) -> Value {
    let rows:Vec<_>=seeds.iter().enumerate().map(|(id,&p)| {
        let predicted=m.map(p);
        let clean=[-7.,0.,7.].into_iter().all(|y|[-7.,0.,7.].into_iter().all(|x|target_features.sample(m.map([p[0]+x,p[1]+y])).is_some()));
        if !clean {return json!({"id":id,"p":p,"predicted":predicted,"hit":null,"reason":"reflection_or_bounds"})}
        let (mut hit,audit)=math::checked(&reference.image,&target.image,Warp{p,q:predicted,a:m.a,sphere:None},deform,if deform {"iris_affine_rescue"}else{"iris_translation"},min_ncc);
        let mut reason=audit.reason;
        if hit.as_ref().is_some_and(|h|distance(h.q,predicted)>2.5) {hit=None;reason="left_shared_prediction".into()}
        let truth_error=hit.as_ref().and_then(|h|truth.map(|t|distance(h.q,t.map(p))));
        json!({"id":id,"p":p,"predicted":predicted,"hit":hit,"reason":reason,"truth_error_px":truth_error})
    }).collect();
    // A fresh direct match to the original reference is required on every
    // frame, even after a dropout. No projected point is counted as observed.
    json!({"accepted":rows.iter().filter(|r|!r["hit"].is_null()).count(),"rows":rows})
}
fn value_noise(x: f64, y: f64, scale: f64) -> f64 {
    let x = x / scale;
    let y = y / scale;
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let a = x - x.floor();
    let b = y - y.floor();
    let hash = |x: i64, y: i64| {
        let mut s = (x as u64).wrapping_mul(0x9e3779b97f4a7c15)
            ^ (y as u64).wrapping_mul(0xbf58476d1ce4e5b9)
            ^ 73;
        s ^= s >> 30;
        s = s.wrapping_mul(0xbf58476d1ce4e5b9);
        s ^= s >> 27;
        (s & 65535) as f64 / 65535.
    };
    (1. - b) * ((1. - a) * hash(ix, iy) + a * hash(ix + 1, iy))
        + b * ((1. - a) * hash(ix, iy + 1) + a * hash(ix + 1, iy + 1))
}
fn signal_check(inputs: &[Input], roi: Region, bright_limit: f64) -> Value {
    let mut rows = Vec::new();
    for radius in [0, 2, 4, 6] {
        let images: Vec<_> = inputs
            .iter()
            .take(3)
            .map(|i| smooth(&i.image, radius))
            .collect();
        let fs: Vec<_> = images
            .iter()
            .map(|i| Features::new(i, bright_limit))
            .collect();
        let train = Samples::new(&fs[0], roi, 0);
        let held = Samples::new(&fs[0], roi, 1);
        for i in 1..images.len() {
            let raw_samples = |s: &Samples| Samples {
                points: s.points.clone(),
                center: s.center,
                values: s
                    .points
                    .iter()
                    .map(|&p| images[0].sample(p).unwrap())
                    .collect(),
            };
            let raw = Features {
                image: images[i].clone(),
                valid: fs[i].valid.clone(),
                center_hint: None,
            };
            let t = raw_samples(&train);
            let h = raw_samples(&held);
            let (raw_m, raw_score) = brute_translation(&t, &raw, delta(&inputs[0], &inputs[i]));
            let (texture_m, texture_score) =
                brute_translation(&train, &fs[i], delta(&inputs[0], &inputs[i]));
            rows.push(json!({"radius":radius,"frame":i,"intensity_fit":raw_score,"intensity_held":geometry::score(&h,&raw,raw_m),"texture_fit":texture_score,"texture_held":geometry::score(&held,&fs[i],texture_m)}));
        }
    }
    json!(rows)
}
fn synthetic(config: &Config) -> (Vec<Input>, Vec<Affine>, [f64; 3]) {
    let (w, h) = (420, 280);
    let c = [config.center[0] + 20., config.center[1] - 20., 180.];
    let mut frames = Vec::new();
    let mut truth = Vec::new();
    for i in 0..=config.last - config.first {
        let t = i as f64;
        let omega = [0.01 * t, -0.015 * t, 0.005 * t].map(|x| x * config.synthetic_motion_scale);
        let mut m = project(c, omega, [0., 0.]);
        if config.synthetic_radial_plane {
            let r = rotation(omega);
            for a in 0..2 {
                for b in 0..2 {
                    let g = (config.center[b] - c[b]) / c[2];
                    m.a[a][b] += r[a][2] * g;
                    m.t[a] -= r[a][2] * g * config.center[b];
                }
            }
        }
        for k in 0..2 {
            m.t[k] += config.synthetic_translation_px[k] * (t * 0.10).sin();
        }
        let inv = m.inverse().unwrap();
        let image = Image {
            w,
            h,
            v: (0..w * h)
                .map(|k| {
                    let p = [(k % w) as f64, (k / w) as f64];
                    let q = inv.map(p);
                    let radius = ((q[0] - config.center[0]) / config.radii[0])
                        .hypot((q[1] - config.center[1]) / config.radii[1]);
                    if (p[0] - config.center[0]).abs() < 20. && (p[1] - config.center[1]).abs() < 9.
                    {
                        0.8
                    } else if radius < 0.33 {
                        0.015
                    } else if radius < 1. {
                        0.06 + 0.07 * value_noise(q[0], q[1], 7.)
                            + 0.035 * value_noise(q[0] + 39., q[1] - 15., 19.)
                    } else {
                        0.25 + 0.06 * value_noise(p[0], p[1], 23.)
                    }
                })
                .collect(),
        };
        let preview = image
            .v
            .iter()
            .flat_map(|v| {
                let b = (v * 510.).clamp(0., 255.) as u8;
                [b, b, b, 255]
            })
            .collect();
        frames.push(Input{image,rgb:None,preview,meta:json!({"sensor_x":0,"sensor_y":0,"sequence":i,"timestamp_ns":i as u64*30_000_000+1}),hash:format!("procedural-iris-{i}"),time:i as f64*0.03});
        truth.push(m);
    }
    (frames, truth, c)
}
pub fn run(args: &[String]) -> Result<()> {
    if !(5..=6).contains(&args.len()) {
        return Err(
            "usage: --iris-pivot BUNDLE|synthetic NEW_OUTPUT CONFIG_JSON [--prepare]".into(),
        );
    }
    let config: Config = serde_json::from_str(&fs::read_to_string(&args[4])?)?;
    let out = Path::new(&args[3]);
    if out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
        || config.last <= config.first
        || config.last - config.first > 60
        || !(1..=2).contains(&config.eye)
        || config.fit_frames.is_empty()
        || config
            .fit_frames
            .iter()
            .any(|&i| i == 0 || i > config.last - config.first)
        || config
            .center
            .iter()
            .chain(config.radii.iter())
            .any(|v| !v.is_finite())
        || config.radii.iter().any(|&v| !(40. ..150.).contains(&v))
        || config.motion_prior.as_ref().is_some_and(|p| !p.valid())
        || config
            .binary_search
            .as_ref()
            .is_some_and(|b| !b.valid() || config.motion_prior.is_none())
        || !(0.1..=1.).contains(&config.synthetic_motion_scale)
        || config
            .synthetic_translation_px
            .iter()
            .any(|x| !x.is_finite() || x.abs() > 4.)
    {
        return Err("new checked output, bounded ROI and 2..61 source frames required".into());
    }
    fs::create_dir(out)?;
    fs::write(out.join("config.json"), serde_json::to_vec_pretty(&config)?)?;
    let mut true_pivot = None;
    let mut truth = Vec::new();
    let mut inputs = if args[2] == "synthetic" {
        let (a, b, c) = synthetic(&config);
        truth = b;
        true_pivot = Some(c);
        a
    } else {
        let bundle = BundleSource::open(Path::new(&args[2]))?;
        let rows = String::from_utf8(bundle.read_entry("frames.jsonl")?)?;
        let metas = rows
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|v| v["eye_id"] == config.eye as u64)
            .collect::<Vec<_>>();
        if config.last >= metas.len() {
            return Err("source interval missing".into());
        }
        let first = metas[0]["timestamp_ns"]
            .as_u64()
            .ok_or("source timestamp")?;
        let inputs = metas[config.first..=config.last]
            .iter()
            .map(|m| load(&bundle, m, first))
            .collect::<Result<Vec<_>>>()?;
        for pair in inputs.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            if b.time <= a.time
                || b.time - a.time > 0.1
                || a.image.w != b.image.w
                || a.image.h != b.image.h
                || a.meta["source_clock"]["source_key"]["stream_epoch"]
                    != b.meta["source_clock"]["source_key"]["stream_epoch"]
            {
                return Err("non-contiguous native source".into());
            }
        }
        inputs
    };
    let roi = Region {
        center: config.center,
        radii: config.radii,
        inner: 0.48,
        outer: 0.84,
    };
    let search = config
        .motion_prior
        .as_ref()
        .map(|p| Search { prior: p, roi });
    let mut levels = Vec::new();
    for y in 0..inputs[0].image.h {
        for x in 0..inputs[0].image.w {
            if roi.contains([x as f64, y as f64]) {
                levels.push(inputs[0].image.v[y * inputs[0].image.w + x])
            }
        }
    }
    levels.sort_by(f64::total_cmp);
    if levels.is_empty() {
        return Err("ROI outside native frame".into());
    }
    let bright_limit = (levels[levels.len() / 2] * 2.5 + 0.04).min(0.75);
    if config.smoothing_radius > 6 {
        return Err("smoothing radius must be at most 6 native pixels".into());
    }
    fs::write(
        out.join("pair-signal.json"),
        serde_json::to_vec_pretty(&signal_check(&inputs, roi, bright_limit))?,
    )?;
    let seed_image = inputs[0].image.clone();
    let broad_images: Vec<_> = inputs.iter().map(|f| smooth(&f.image, 6)).collect();
    let mut features: Vec<_> = inputs
        .iter_mut()
        .map(|f| {
            let raw = f.image.clone();
            f.image = smooth(&raw, config.smoothing_radius);
            Features::with_mask(&f.image, &raw, bright_limit)
        })
        .collect();
    let train = Samples::new(&features[0], roi, 0);
    let held = Samples::new(&features[0], roi, 1);
    // Localize only the coarse iris appearance to bound the target search.
    // This does not provide a rotation, pivot or point correspondence. The
    // same bound applies to all methods and never uses withheld samples.
    let broad_train = Samples {
        points: train.points.clone(),
        values: train
            .points
            .iter()
            .map(|&p| broad_images[0].sample(p).unwrap())
            .collect(),
        center: roi.center,
    };
    let mut hints = Vec::new();
    for i in 0..features.len() {
        let f = Features {
            image: broad_images[i].clone(),
            valid: features[i].valid.clone(),
            center_hint: None,
        };
        let (m, s) = brute_translation(&broad_train, &f, delta(&inputs[0], &inputs[i]));
        let intensity_center = m.map(roi.center);
        let (contrast_center, contrast) =
            iris_region_center(&broad_images[i], roi, delta(&inputs[0], &inputs[i]));
        let center = if contrast > 0.03 && distance(intensity_center, contrast_center) > 12. {
            contrast_center
        } else {
            intensity_center
        };
        features[i].center_hint = Some(center);
        hints.push(json!({"center":center,"intensity_center":intensity_center,"contrast_center":contrast_center,"lateral_contrast":contrast,"fit":s,"scope":"coarse iris intensity translation, corrected by bilateral lateral contrast if inconsistent; fixed-radius search bound only, not an anatomical measurement"}));
    }
    fs::write(
        out.join("iris-region-hints.json"),
        serde_json::to_vec_pretty(&hints)?,
    )?;
    let seeds = iris_seeds(&seed_image, &features[0], roi);
    if train.points.len() < 40 || held.points.len() < 40 || seeds.len() < 4 {
        return Err(format!(
            "insufficient iris-only support: fit {}, held {}, seeds {}",
            train.points.len(),
            held.points.len(),
            seeds.len()
        )
        .into());
    }
    support_image(out, &inputs[0], roi, &train, &held, &seeds)?;
    let support = json!({"roi":roi,"train":train.points,"held":held.points,"seeds":seeds,"bright_limit":bright_limit,"source_sequence":inputs[0].meta["sequence"],"raw_sha256":inputs[0].hash});
    fs::write(
        out.join("support.json"),
        serde_json::to_vec_pretty(&support)?,
    )?;
    for (i, f) in inputs.iter().enumerate() {
        image(out, i, f)?
    }
    eprintln!(
        "eye {}: {} fit samples / {} withheld / {} iris patch IDs, bright threshold {:.4}",
        config.eye,
        train.points.len(),
        held.points.len(),
        seeds.len(),
        bright_limit
    );
    if args.get(5).is_some_and(|s| s == "--prepare") {
        return Ok(());
    }
    let began = Instant::now();
    let pyramid = config.binary_search.as_ref().map(|p| {
        binary::Pyramid::new(
            p.clone(),
            &inputs,
            &features,
            &train,
            roi,
            config.smoothing_radius,
        )
    });
    if let Some(p) = &pyramid {
        for level in &p.levels {
            for (i, f) in level.frames.iter().enumerate() {
                let bytes: Vec<_> = f
                    .image
                    .v
                    .iter()
                    .zip(&f.valid)
                    .flat_map(|(&v, &ok)| {
                        let c = if ok {
                            (v.clamp(0., 1.).sqrt() * 255.) as u8
                        } else {
                            0
                        };
                        [c, c, c, 255]
                    })
                    .collect();
                let mut canvas = Canvas::new(f.image.w, f.image.h)?;
                canvas.image(
                    &bytes,
                    f.image.w,
                    f.image.h,
                    0.,
                    0.,
                    f.image.w as f64,
                    f.image.h as f64,
                );
                canvas.png(&out.join(format!("pyramid-{i:03}-{}.png", level.factor)))?;
            }
        }
    }
    if let Some(p) = &pyramid {
        for (factor, relative) in p
            .levels
            .iter()
            .filter_map(|l| l.relative.as_ref().map(|r| (l.factor, r)))
            .chain(p.native_relative.as_ref().map(|r| (1, r)))
        {
            for (i, f) in relative.frames.iter().enumerate() {
                let (w, h) = (f.channels[0].w, f.channels[0].h);
                let bytes: Vec<_> = (0..w * h)
                    .flat_map(|k| {
                        let rgb: [u8; 3] = std::array::from_fn(|c| {
                            if f.valid[k] {
                                (f.channels[c].v[k].clamp(0., 1.).sqrt() * 255.) as u8
                            } else {
                                0
                            }
                        });
                        [rgb[2], rgb[1], rgb[0], 255]
                    })
                    .collect();
                let mut canvas = Canvas::new(w, h)?;
                canvas.image(&bytes, w, h, 0., 0., w as f64, h as f64);
                canvas.png(&out.join(format!("color-{i:03}-{factor}.png")))?;
            }
        }
    }
    let coarse: Vec<_> = search
        .as_ref()
        .map_or_else(|| grid_pivots(config.center), Search::pivots)
        .into_iter()
        .map(|p| (p, None))
        .collect();
    let mut grid = profiles(
        &train,
        &held,
        &inputs,
        &features,
        &config.fit_frames,
        &coarse,
        "exhaustive",
        search.as_ref(),
        pyramid.as_ref(),
    );
    grid.sort_by(|a, b| a.loss.total_cmp(&b.loss));
    let mut bases = Vec::new();
    for p in &grid {
        if bases.iter().all(|&j: &usize| {
            let q = &grid[j];
            (0..3)
                .map(|i| (p.pivot[i] - q.pivot[i]).powi(2))
                .sum::<f64>()
                .sqrt()
                > 35.
        }) {
            bases.push(grid.iter().position(|q| std::ptr::eq(q, p)).unwrap());
            if bases.len() == 4 {
                break;
            }
        }
    }
    let mut fine = Vec::new();
    for b in bases {
        let xy_step = search.as_ref().map_or(10., |s| s.radius() * 0.15);
        let z_step = search.as_ref().map_or(15., |s| s.radius() * 0.15);
        for z in [-z_step, 0., z_step] {
            for y in [-xy_step, 0., xy_step] {
                for x in [-xy_step, 0., xy_step] {
                    if x == 0. && y == 0. && z == 0. {
                        continue;
                    }
                    let p = &grid[b];
                    let c = [p.pivot[0] + x, p.pivot[1] + y, p.pivot[2] + z];
                    if search.as_ref().is_some_and(|s| !s.contains_pivot(c)) {
                        continue;
                    }
                    if search.is_some()
                        && coarse
                            .iter()
                            .chain(fine.iter())
                            .any(|(q, _)| (0..3).all(|i| (q[i] - c[i]).abs() < 1e-7))
                    {
                        continue;
                    }
                    fine.push((
                        c,
                        Some(p.poses.iter().map(|p| (p.omega, p.translation)).collect()),
                    ));
                }
            }
        }
    }
    grid.extend(profiles(
        &train,
        &held,
        &inputs,
        &features,
        &config.fit_frames,
        &fine,
        "local-grid",
        search.as_ref(),
        pyramid.as_ref(),
    ));
    grid.sort_by(|a, b| a.loss.total_cmp(&b.loss));
    let mut pivot_refinement = Vec::new();
    let mut iterative_pivots = 0;
    if let Some(pyramid) = &pyramid {
        let search = search.as_ref().unwrap();
        for round in 0..pyramid.config.pivot_rounds {
            let step = search.radius() * 0.075 / 2f64.powi(round as i32);
            let before = grid[0].loss;
            let mut centers: Vec<&Profile> = Vec::new();
            for p in &grid {
                if centers.iter().all(|q| {
                    (0..3)
                        .map(|k| (p.pivot[k] - q.pivot[k]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        > 10.
                }) {
                    centers.push(p);
                    if centers.len() == 4 {
                        break;
                    }
                }
            }
            let mut tests = Vec::new();
            for p in centers {
                for axis in 0..3 {
                    for sign in [-1., 1.] {
                        let mut c = p.pivot;
                        c[axis] += sign * step;
                        if search.contains_pivot(c)
                            && !grid
                                .iter()
                                .any(|q| (0..3).all(|k| (q.pivot[k] - c[k]).abs() < 1e-7))
                        {
                            tests.push((
                                c,
                                Some(p.poses.iter().map(|p| (p.omega, p.translation)).collect()),
                            ));
                        }
                    }
                }
            }
            iterative_pivots += tests.len();
            grid.extend(profiles(
                &train,
                &held,
                &inputs,
                &features,
                &config.fit_frames,
                &tests,
                "iterative-binary-pivot",
                Some(search),
                Some(pyramid),
            ));
            grid.sort_by(|a, b| a.loss.total_cmp(&b.loss));
            pivot_refinement.push(json!({"round":round,"step_px":step,"tested":tests.len(),"before_objective":before,"after_objective":grid[0].loss,"chosen_pivot":grid[0].pivot}));
        }
    }
    let chosen = grid[0].clone();
    fs::write(
        out.join("pivot-grid.json"),
        serde_json::to_vec_pretty(&grid)?,
    )?;
    eprintln!(
        "chosen apparent pivot {:?}, fit loss {:.4}, withheld loss {:.4}",
        chosen.pivot, chosen.loss, chosen.held_loss
    );
    let angles = search
        .as_ref()
        .map_or_else(grid_angles, |s| s.prior.angles());
    let baseline = if let Some(path) = &config.baseline_review {
        let path = Path::new(path).canonicalize()?;
        if !path.starts_with(fs::canonicalize("outputs")?) {
            return Err("baseline must be a saved outputs review".into());
        }
        let data: Value = serde_json::from_slice(&fs::read(&path)?)?;
        if data["support"] != support
            || data["frames"].as_array().map(Vec::len) != Some(inputs.len())
        {
            return Err("baseline has different source support or frame count".into());
        }
        for (i, f) in inputs.iter().enumerate() {
            if data["frames"][i]["raw_sha256"] != f.hash
                || data["frames"][i]["timestamp_ns"]
                    != f.meta["timestamp_ns"].as_u64().unwrap().to_string()
            {
                return Err("baseline source hashes or timestamps differ".into());
            }
        }
        Some(data)
    } else {
        None
    };
    let variant_count =
        if baseline.is_some() { 4 } else { 3 } + usize::from(config.lenient_comparison);
    let mut frames = Vec::new();
    let mut log = fs::File::create(out.join("frames.jsonl"))?;
    for (i, current) in inputs.iter().enumerate() {
        let d = delta(&inputs[0], current);
        let (translation, ts) = brute_translation(&train, &features[i], d);
        let p = if i == 0 && search.is_some() {
            let warp = Affine::translation([0.; 2]);
            let s = geometry::score(&train, &features[i], warp);
            Pose {
                omega: [0.; 3],
                warp,
                train: s,
                held: geometry::score(&held, &features[i], warp),
                translation: [0.; 2],
                plane_gradient: search.as_ref().unwrap().gradient(chosen.pivot),
                objective: s.loss,
                prior_penalty: 0.,
                at_limit: false,
                binary_trace: Vec::new(),
                binary_score_evaluations: 0,
                relative_color_score: None,
            }
        } else {
            pose(
                &train,
                &held,
                &features[i],
                chosen.pivot,
                d,
                None,
                &angles,
                search.as_ref(),
                pyramid.as_ref().map(|p| (p, i)),
            )
        };
        // The free affine ablation gets both the translation and searched 3D
        // initialization. This compares the constraint, not unequal seeding.
        let a = refine_affine(&train, &features[i], translation, config.center, true);
        let b = refine_affine(&train, &features[i], p.warp, config.center, true);
        let (affine, ascore) = if a.1.loss < b.1.loss { a } else { b };
        let models = [translation, affine, p.warp];
        let scores = [ts, ascore, p.train];
        let mut variants: Vec<_> = (0..3)
            .map(|v| {
                let mut r = points(
                    &inputs[0],
                    current,
                    &features[i],
                    &seeds,
                    models[v],
                    v != 0,
                    truth.get(i).copied(),
                    0.65,
                );
                r["name"] = json!(
                    [
                        "Iris translation",
                        "Free iris affine",
                        if search.is_some() {
                            "Constrained 3D iris rescue"
                        } else {
                            "Brute 3D pivot + affine rescue"
                        }
                    ][v]
                );
                r["warp"] = json!(models[v]);
                r["fit"] = json!(scores[v]);
                r["held"] = json!(geometry::score(&held, &features[i], models[v]));
                if let Some(t) = truth.get(i) {
                    r["true_geometry_error_px"] = quantiles(
                        held.points
                            .iter()
                            .map(|&q| distance(models[v].map(q), t.map(q)))
                            .collect(),
                    )
                }
                r
            })
            .collect();
        if let Some(b) = &baseline {
            let mut previous = b["frames"][i]["variants"][2].clone();
            previous["name"] = json!("Previous 3D search");
            variants.push(previous);
        }
        if config.lenient_comparison {
            let mut relaxed = points(
                &inputs[0],
                current,
                &features[i],
                &seeds,
                p.warp,
                true,
                truth.get(i).copied(),
                0.55,
            );
            for key in ["warp", "fit", "held", "true_geometry_error_px"] {
                if let Some(value) = variants[2].get(key) {
                    relaxed[key] = value.clone();
                }
            }
            relaxed["name"] = json!("Same 3D fit, lenient NCC 0.55");
            variants.push(relaxed);
        }
        let row = json!({"index":i,"image":format!("raw-{i:03}.png"),"time_s":current.time,"sequence":current.meta["sequence"],"timestamp_ns":current.meta["timestamp_ns"].as_u64().unwrap().to_string(),"origin":origin(&current.meta),"raw_sha256":current.hash,"pivot_fit_frame":config.fit_frames.contains(&i),"iris_region_hint":hints[i],"pose":p,"variants":variants});
        writeln!(log, "{row}")?;
        frames.push(row);
        if i % 10 == 0 {
            eprintln!(
                "eye {} frame {i}: iris image matches {:?}",
                config.eye,
                frames.last().unwrap()["variants"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v["accepted"].as_u64().unwrap())
                    .collect::<Vec<_>>()
            )
        }
    }
    let near: Vec<_> = grid
        .iter()
        .filter(|p| p.loss <= chosen.loss + 0.015)
        .collect();
    let bounds: Vec<_> = (0..3)
        .map(|k| {
            [
                near.iter()
                    .map(|p| p.pivot[k])
                    .fold(f64::INFINITY, f64::min),
                near.iter()
                    .map(|p| p.pivot[k])
                    .fold(f64::NEG_INFINITY, f64::max),
            ]
        })
        .collect();
    let stats:Vec<_>=(0..variant_count).map(|v| {
        let counts:Vec<_>=frames.iter().skip(1).map(|f|f["variants"][v]["accepted"].as_u64().unwrap() as f64).collect();
        let errors=frames.iter().skip(1).flat_map(|f|f["variants"][v]["rows"].as_array().unwrap()).filter_map(|r|r["truth_error_px"].as_f64()).collect::<Vec<_>>();
        let all=seeds.iter().enumerate().filter(|(id,_)|frames.iter().skip(1).all(|f|!f["variants"][v]["rows"][*id]["hit"].is_null())).count();
        json!({"name":frames[0]["variants"][v]["name"],"accepted_per_frame":quantiles(counts.clone()),"observations_after_reference":counts.iter().sum::<f64>(),"at_end":frames.last().unwrap()["variants"][v]["accepted"],"every_frame":all,"held_texture_ncc":quantiles(frames.iter().skip(1).map(|f|f["variants"][v]["held"]["ncc"].as_f64().unwrap()).collect()),"true_match_error_px":quantiles(errors.clone()),"truth_scored":errors.len(),"wrong_over_2px":errors.iter().filter(|&&e|e>2.).count()})
    }).collect();
    let report = json!({"schema":"iris-only-pivot-brute-force-v1","bundle":args[2],"config":config,"start_s":inputs[0].time,"end_s":inputs.last().unwrap().time,"frames":inputs.len(),"initial_iris_ids":seeds.len(),"fit_samples":train.points.len(),"held_samples":held.points.len(),"chosen":chosen,"near_tie_loss_tolerance":0.015,"near_tied_pivots":near.len(),"near_tie_coordinate_ranges":bounds,"grid_pivots":coarse.len(),"angles_per_pivot_frame":angles.len(),"coarse_models_evaluated":coarse.len()*angles.len()*config.fit_frames.len(),"local_pivots":fine.len(),"seconds":began.elapsed().as_secs_f64(),"true_pivot_if_synthetic":true_pivot,"variants":stats,"provenance":{"recipe":digest(include_bytes!("iris_pivot.rs")),"geometry":digest(include_bytes!("iris_pivot_math.rs")),"viewer":digest(include_bytes!("iris_pivot_viewer.html")),"patch_math":digest(include_bytes!("warp_math.rs"))},"scope":"Motion fitting uses iris-interior appearance. A fixed-size lateral-contrast/intensity locator bounds the target region; it supplies no anatomical geometry or point correspondences. Conservative manually configured ROI is an experimental support mask, not an anatomical human label or model training material. Source fitting and held samples exclude the pupil void, bright-reflection halo and outer boundary, with 15-pixel context guards. Configured smoothing is identical across methods. Every frame independently matches the initial RAW reference, including re-found identities after dropouts. Dashed model points are never counted as observations. Plane z=0, orthographic projection, one fixed apparent pivot and bounded rotation; sensor crop translation is corrected from metadata, but unknown head/camera motion is not independently removed. Free per-frame translation would make the pivot exactly non-identifiable, so it is absent from the 3D search. Depth is in image-scale units, not metric anatomy; near ties are a sensitivity diagnostic, not confidence intervals. No measured limbus or independent scale: SN-FEIDA not applicable. No human tissue point ground truth for native runs and no live-FPS or cross-user claim."});
    let mut report = report;
    report["provenance"]["motion_prior"] = json!(digest(include_bytes!("iris_pivot_prior.rs")));
    report["provenance"]["relative_color"] = json!(digest(include_bytes!("iris_pivot_color.rs")));
    report["provenance"]["input_decode"] = json!(digest(include_bytes!("warp_probe.rs")));
    report["provenance"]["binary_search"] = json!(digest(include_bytes!("iris_pivot_binary.rs")));
    report["pose_diagnostics"] = json!({
        "rotation_degrees":quantiles(frames.iter().skip(1).map(|f| prior::rotation_difference_degrees([0.;3], serde_json::from_value(f["pose"]["omega"].clone()).unwrap())).collect()),
        "adjacent_rotation_degrees":quantiles(frames.windows(2).map(|f| prior::rotation_difference_degrees(serde_json::from_value(f[0]["pose"]["omega"].clone()).unwrap(),serde_json::from_value(f[1]["pose"]["omega"].clone()).unwrap())).collect()),
        "residual_translation_px":quantiles(frames.iter().skip(1).map(|f| { let t:P=serde_json::from_value(f["pose"]["translation"].clone()).unwrap();t[0].hypot(t[1]) }).collect()),
        "frames_at_motion_bound":frames.iter().skip(1).filter(|f|f["pose"]["at_limit"]==true).count(),
        "near_tie_objective_includes_prior":search.is_some(),
        "selection_uses_held_pixels":false,
    });
    if let Some(b) = &baseline {
        report["baseline_provenance"] = json!({"review_sha256":digest(&serde_json::to_vec(b)?),"report":b["report"],"same_source_hashes_timestamps_support":true});
    }
    if let Some(s) = &search {
        report["schema"] = json!("iris-only-pivot-constrained-v2");
        report["prior_units"] = json!({"fixed_reference_radius_px":s.radius(),"xy_offset_limit_px":s.prior.pivot_offset_radii*s.radius(),"depth_range_px":s.prior.pivot_depth_radii.map(|x|x*s.radius()),"translation_l2_limit_px":s.prior.translation_radii*s.radius(),"finest_angle_step_deg":0.03125,"finest_translation_step_px":0.0625});
        report["scope"] = json!("Same iris-interior support, bright RAW exclusion mask, matching gates and original-reference identities as the unconstrained experiment. The reference radius is a manually configured support size, not a measured anatomical radius or independent metric scale. Explicit heuristic bounds limit rotation, apparent pivot location and residual image translation; soft penalties prefer smaller motion and a central pivot within that range. Optional reference-plane tilt aligns its normal with the candidate pivot-to-iris-center direction. Recorded sensor crop shifts are corrected, but head/camera motion is not independently measured. Neither the reference-plane assumption nor the prior is calibrated for Rob or any population. Pivot score includes the stated priors; separate image-only fitting and withheld scores are retained. Small translation weakens pivot identifiability further. Frames are independently image-fitted: no temporal smoothing, interpolation or held prediction is counted as an observation. Bounds may reject real saccades outside this local interval. No native human tissue-point ground truth, measured limbus or independent scale; SN-FEIDA is not applicable and no cross-user, metric-depth or live-FPS claim is made.");
    }
    if let Some(p) = &pyramid {
        report["schema"] = json!("iris-only-iterative-binary-v5");
        report["coarse_models_evaluated"] = Value::Null;
        report["angles_per_pivot_frame"] = Value::Null;
        report["binary_search"] = json!({"config":p.config,
            "pyramid_levels":p.levels.iter().map(|l|json!({"factor":l.factor,"gaussian_sigma_native_px":l.factor as f64/2.,"post_downsample_sigma_px":if l.factor==p.config.coarse_factor {p.config.coarse_post_blur_sigma_px} else {0.},"used_for_search":l.relative.as_ref().map_or(l.samples.points.len(),color::Level::support)>=12,"source_fit_samples":l.relative.as_ref().map_or(l.samples.points.len(),color::Level::support),"source_fit_points_native":l.relative.as_ref().map_or_else(||l.samples.points.clone(),color::Level::points).iter().map(|p|p.map(|x|x*l.factor as f64)).collect::<Vec<_>>(),"width":l.frames[0].image.w,"height":l.frames[0].image.h})).collect::<Vec<_>>(),
            "scored_binary_nodes":grid.iter().flat_map(|p|&p.poses).map(|p|p.binary_score_evaluations).sum::<usize>()+frames.iter().map(|f|f["pose"]["binary_score_evaluations"].as_u64().unwrap() as usize).sum::<usize>(),
            "count_excludes_final_coordinate_refinement":true,"pivot_refinement":pivot_refinement,"iterative_pivots":iterative_pivots,
            "scope":"Every binary split tests both halves of one rotation or translation interval; a bounded beam retains multiple alternatives. Each child is probed at its midpoint and at the existing coarse iris-location estimate projected into its parameter bounds. The latter uses the pivot-depth small-angle displacement relation plus a bounded residual translation. It is an initialization hypothesis, not a measured rotation. Binary score counts count these representative-pose evaluations, not just child cells. Early cycles use the configured 4x or 8x then 2x decimated masked Gaussian intensity images; the coarsest image optionally receives additional Gaussian blur in downsampled pixels; later cycles and final coordinate refinement use native contrast or the explicitly selected relative-color objective. Native grayscale held correlation is still recorded for comparison. Blur is normalized over allowed pixels and requires at least 90% kernel weight. Masks exclude bright pixels and constrain texture to the conservative iris region with preprocessing support guards. Coarse target ROI comes from the same recorded intensity/contrast hint as the baseline, not anatomy truth. Levels with fewer than 12 source samples fall back to native scoring. Shared pivot coordinates are tested in both directions with successively halved step sizes, refitting per-frame rotation/translation for each proposed pivot. No monotonic-loss or global-optimum guarantee; coarse branch rejection can discard the right hypothesis. Withheld samples and point-acceptance counts never select a branch. Primary point checks retain the native grayscale gates and original-reference identities; any relaxed-threshold variant is separately labeled."});
    }
    report["matching_thresholds"] = json!({"primary_min_ncc":0.65,"lenient_comparison_min_ncc":if config.lenient_comparison {Some(0.55)} else {None},"forward_backward_max_px":0.9,"shared_projection_max_px":2.5});
    if let Some(p) = pyramid.as_ref().filter(|p| p.config.relative_color) {
        report["relative_color"] = json!({"native_reference_support":p.native_relative.as_ref().unwrap().support(),"native_source_points":p.native_relative.as_ref().unwrap().points(),"native_neighbor_radius_px":6,"minimum_coverage":0.4,"minimum_points":12,"cauchy_half_width":0.5,"contrast_floor_raw_units":2./1023.,
            "scope":"Phase-aware 4x4 Quad-Bayer channel averages from native RAW; no display white balance. Each channel is compared with four cardinal neighbors at max(6/factor,1) level pixels. Center subtraction removes additive channel offsets and local RMS normalization reduces multiplicative channel-gain changes above the contrast floor. The candidate affine warp moves both center and neighbors. Cauchy residuals limit the influence of changing neighbors; at least two valid neighbors, 12 source locations and 40 percent target coverage are required. Excluded pixels never contribute. Reference eligibility is reported per level; levels with fewer than 12 eligible locations fall back to native color. Binary trace ncc and pose.relative_color_score.ncc are legacy storage names for 1-minus-mean-robust-residual, not NCC or probability. Pose.train and pose.held remain native grayscale NCC diagnostics. Synthetic full-sequence controls are achromatic; separate unit tests exercise independent channel gains and offsets. Lenient point checks only lower native NCC to 0.55; backward and shared-projection limits remain unchanged."});
    }
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    let data = json!({"report":report,"frames":frames,"support":support,"pivots":grid,"width":inputs[0].image.w,"height":inputs[0].image.h});
    fs::write(out.join("review.json"), serde_json::to_vec(&data)?)?;
    fs::write(
        out.join("viewer.html"),
        include_str!("iris_pivot_viewer.html").replace("IRIS_DATA", &serde_json::to_string(&data)?),
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binary_intervals_search_motion_far_from_the_reference_region() {
        let mut prior = MotionPrior::default();
        prior.angle_limit_deg[2] = 4.;
        let config = Config {
            eye: 1,
            first: 0,
            last: 16,
            center: [190., 130.],
            radii: [90., 80.],
            fit_frames: vec![4, 8, 12],
            smoothing_radius: 0,
            motion_prior: Some(prior),
            binary_search: None,
            baseline_review: None,
            lenient_comparison: false,
            synthetic_motion_scale: 0.8,
            synthetic_radial_plane: true,
            synthetic_translation_px: [2.5, -1.5],
        };
        let (inputs, truth, pivot) = synthetic(&config);
        let mut features: Vec<_> = inputs
            .iter()
            .map(|i| Features::new(&i.image, 0.32))
            .collect();
        // Synthetic region observations supply location only, never a rotation.
        for (i, f) in features.iter_mut().enumerate() {
            f.center_hint = Some(truth[i].map(config.center));
        }
        assert!(distance(features[16].center_hint.unwrap(), config.center) > 30.);
        let roi = Region {
            center: config.center,
            radii: config.radii,
            inner: 0.48,
            outer: 0.84,
        };
        let train = Samples::new(&features[0], roi, 0);
        let held = Samples::new(&features[0], roi, 1);
        for (coarse_factor, coarse_post_blur_sigma_px, relative_color) in [
            (4, 0., false),
            (8, 0., false),
            (8, 0.5, false),
            (8, 0., true),
            (8, 0.5, true),
        ] {
            let pyramid = binary::Pyramid::new(
                binary::Config {
                    coarse_factor,
                    coarse_post_blur_sigma_px,
                    relative_color,
                    ..binary::Config::default()
                },
                &inputs,
                &features,
                &train,
                roi,
                0,
            );
            let search = Search {
                prior: config.motion_prior.as_ref().unwrap(),
                roi,
            };
            let p = pose(
                &train,
                &held,
                &features[16],
                pivot,
                [0.; 2],
                None,
                &[],
                Some(&search),
                Some((&pyramid, 16)),
            );
            let error = held
                .points
                .iter()
                .map(|&q| distance(p.warp.map(q), truth[16].map(q)))
                .sum::<f64>()
                / held.points.len() as f64;
            assert!(
                error < 0.5 && p.held.ncc > 0.95,
                "far-region error={error}, NCC={}",
                p.held.ncc
            );
            assert!(
                p.binary_trace.iter().any(|s| s.factor == coarse_factor),
                "coarse stage fell back to native"
            );
        }
    }
    #[test]
    fn constrained_search_recovers_tilted_iris_with_small_translation() {
        let config = Config {
            eye: 1,
            first: 0,
            last: 12,
            center: [190., 130.],
            radii: [90., 80.],
            fit_frames: vec![4, 8, 12],
            smoothing_radius: 0,
            motion_prior: Some(MotionPrior::default()),
            binary_search: None,
            baseline_review: None,
            lenient_comparison: false,
            synthetic_motion_scale: 0.5,
            synthetic_radial_plane: true,
            synthetic_translation_px: [2.5, -1.5],
        };
        let (inputs, truth, pivot) = synthetic(&config);
        let reference = Features::new(&inputs[0].image, 0.32);
        let target = Features::new(&inputs[12].image, 0.32);
        let roi = Region {
            center: config.center,
            radii: config.radii,
            inner: 0.48,
            outer: 0.84,
        };
        let train = Samples::new(&reference, roi, 0);
        let held = Samples::new(&reference, roi, 1);
        let search = Search {
            prior: config.motion_prior.as_ref().unwrap(),
            roi,
        };
        let p = pose(
            &train,
            &held,
            &target,
            pivot,
            [0.; 2],
            None,
            &search.prior.angles(),
            Some(&search),
            None,
        );
        let error = held
            .points
            .iter()
            .map(|&q| distance(p.warp.map(q), truth[12].map(q)))
            .sum::<f64>()
            / held.points.len() as f64;
        assert!(
            error < 0.5 && p.held.ncc > 0.95,
            "error={error}, ncc={}",
            p.held.ncc
        );
        assert!(p.translation[0].hypot(p.translation[1]) <= 3.6);
        let features: Vec<_> = inputs
            .iter()
            .map(|i| Features::new(&i.image, 0.32))
            .collect();
        let pyramid = binary::Pyramid::new(
            binary::Config::default(),
            &inputs,
            &features,
            &train,
            roi,
            0,
        );
        let p = pose(
            &train,
            &held,
            &target,
            pivot,
            [0.; 2],
            None,
            &[],
            Some(&search),
            Some((&pyramid, 12)),
        );
        let error = held
            .points
            .iter()
            .map(|&q| distance(p.warp.map(q), truth[12].map(q)))
            .sum::<f64>()
            / held.points.len() as f64;
        assert!(
            error < 0.5 && p.held.ncc > 0.95,
            "binary error={error}, ncc={}",
            p.held.ncc
        );
        assert_eq!(p.binary_trace.len(), 30);
        assert!(
            p.binary_trace.iter().any(|s| s.factor == 4)
                && p.binary_trace.iter().any(|s| s.factor == 2)
        );
        assert!(p.binary_trace.iter().all(|s| s.retained <= 32));
    }
    #[test]
    fn brute_rotation_recovers_motion_between_angular_grid_cells() {
        let config = Config {
            eye: 1,
            first: 0,
            last: 4,
            center: [190., 130.],
            radii: [90., 80.],
            fit_frames: vec![4],
            smoothing_radius: 0,
            motion_prior: None,
            binary_search: None,
            baseline_review: None,
            lenient_comparison: false,
            synthetic_motion_scale: 1.,
            synthetic_radial_plane: false,
            synthetic_translation_px: [0.; 2],
        };
        let (inputs, truth, pivot) = synthetic(&config);
        let reference = Features::new(&inputs[0].image, 0.32);
        let target = Features::new(&inputs[4].image, 0.32);
        let roi = Region {
            center: config.center,
            radii: config.radii,
            inner: 0.48,
            outer: 0.84,
        };
        let train = Samples::new(&reference, roi, 0);
        let held = Samples::new(&reference, roi, 1);
        let (w, s) = brute_rotation(&train, &target, pivot, [0., 0.], &grid_angles());
        let got = project(pivot, w, [0., 0.]);
        let error = held
            .points
            .iter()
            .map(|&p| distance(got.map(p), truth[4].map(p)))
            .sum::<f64>()
            / held.points.len() as f64;
        assert!(error < 0.4, "error={error}, rotation={w:?}, score={s:?}");
    }
}
