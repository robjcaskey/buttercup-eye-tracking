//! RAW/mask audit on area-admitted inputs. Conditional anatomy, not sign truth.
use super::{archive, Result};
#[path = "surface_sign/anatomy_masks.rs"]
mod anatomy_masks;
#[path = "surface_sign/evidence_audit.rs"]
pub(crate) mod evidence_audit;
#[path = "surface_sign/center_motion.rs"]
pub(super) mod center_motion;
#[path = "../buttercup_calibration_sign/canvas.rs"]
#[allow(dead_code)]
mod canvas;
#[path = "surface_sign/geometry.rs"]
mod geometry;
#[path = "surface_sign/lids.rs"]
mod lids;
#[path = "surface_sign/lid_circle.rs"]
pub(super) mod lid_circle;
#[path = "surface_sign/lid_review.rs"]
pub(crate) mod lid_review;
#[path = "surface_sign/photometry.rs"]
mod photometry;
#[path = "../../raw_preview.rs"]
#[allow(dead_code)]
mod preview;
#[path = "surface_sign/report.rs"]
pub(super) mod report;
use buttercup_eye_tracking::{
    focus_region::*, geometry::Ellipse, raw10, recorded_bundle::BundleSource,
};
use canvas::*;
use geometry::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

fn load(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn rows(p: &Path) -> Result<Vec<Value>> {
    BufReader::new(fs::File::open(p)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect()
}
fn count(c: &mut BTreeMap<String, usize>, s: String) {
    *c.entry(s).or_default() += 1;
}
fn n(v: &Value) -> u64 {
    v.as_u64().expect("required source identity")
}
fn shape(v: &Value) -> Result<Ellipse> {
    let e: [f64; 5] = serde_json::from_value(v.clone())?;
    Ok(Ellipse {
        center: (e[0], e[1]),
        major_radius: e[2],
        minor_radius: e[3],
        angle: e[4],
    })
}
fn rho(e: Ellipse, p: [f64; 2]) -> f64 {
    let (s, c) = e.angle.sin_cos();
    let x = p[0] - e.center.0;
    let y = p[1] - e.center.1;
    ((c * x + s * y) / e.major_radius).hypot((-s * x + c * y) / e.minor_radius)
}

/// Same observed mask for both poses. Probability and lid veto are unverified.
fn samples(masks: &[u8], frame: &Value, e: Ellipse, threshold: u8, erode: usize) -> Vec<[f64; 2]> {
    let selected = sample_mask(masks, frame, e, threshold, erode);
    let (w, h) = (n(&frame["width"]) as f64, n(&frame["height"]) as f64);
    let (sx, sy) = (n(&frame["sensor_x"]) as f64, n(&frame["sensor_y"]) as f64);
    selected
        .into_iter()
        .enumerate()
        .filter_map(|(j, yes)| {
            let (x, y) = (j % 384, j / 384);
            (yes && x % 3 == 0 && y % 3 == 0).then_some([
                sx + (x as f64 + 0.5) * w / 384. - 0.5,
                sy + (y as f64 + 0.5) * h / 256. - 0.5,
            ])
        })
        .collect()
}
fn sample_mask(masks: &[u8], frame: &Value, e: Ellipse, threshold: u8, erode: usize) -> Vec<bool> {
    let (w, h) = (n(&frame["width"]) as f64, n(&frame["height"]) as f64);
    let (wm, hm) = (384usize, 256usize);
    let plane = wm * hm;
    let sclera = &masks[3 * plane..4 * plane];
    let upper = &masks[4 * plane..5 * plane];
    let lower = &masks[5 * plane..6 * plane];
    let mut selected = vec![false; plane];
    for y in erode..hm - erode {
        for x in erode..wm - erode {
            let p = [
                (x as f64 + 0.5) * w / wm as f64 - 0.5,
                (y as f64 + 0.5) * h / hm as f64 - 0.5,
            ];
            if rho(e, p) < 1.04
                || sclera[y * wm + x] < threshold
                || upper[y * wm + x] > 128
                || lower[y * wm + x] > 128
            {
                continue;
            }
            selected[y * wm + x] = (y - erode..=y + erode)
                .all(|yy| (x - erode..=x + erode).all(|xx| sclera[yy * wm + xx] >= threshold));
        }
    }
    // Reject isolated islands disconnected from the lateral limbus neighborhood.
    let mut visited = vec![false; plane];
    let mut output = vec![false; plane];
    for i in 0..plane {
        if !selected[i] || visited[i] {
            continue;
        }
        let mut q = vec![i];
        visited[i] = true;
        let mut at = 0;
        let mut touches = false;
        while at < q.len() {
            let j = q[at];
            at += 1;
            let (x, y) = (j % wm, j / wm);
            let p = [
                (x as f64 + 0.5) * w / wm as f64 - 0.5,
                (y as f64 + 0.5) * h / hm as f64 - 0.5,
            ];
            touches |= rho(e, p) < 1.30;
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let xx = x as i32 + dx;
                let yy = y as i32 + dy;
                if xx >= 0 && yy >= 0 && xx < wm as i32 && yy < hm as i32 {
                    let k = yy as usize * wm + xx as usize;
                    if selected[k] && !visited[k] {
                        visited[k] = true;
                        q.push(k);
                    }
                }
            }
        }
        if touches && q.len() >= 64 {
            for j in q {
                output[j] = true;
            }
        }
    }
    output
}
fn select(a: &Value, b: &Value) -> Option<usize> {
    let (x, y) = (a["coverage"].as_f64()?, b["coverage"].as_f64()?);
    if x >= 0.98 && y <= 0.80 {
        Some(0)
    } else if y >= 0.98 && x <= 0.80 {
        Some(1)
    } else {
        None
    }
}
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn silhouette(g: V3, r: f64) -> Vec<[f64; 2]> {
    let s = norm(g);
    let axis = unit(g);
    let u = unit(cross(axis, [0., 1., 0.]));
    let v = cross(axis, u);
    let c = scale(g, 1. - r * r / (s * s));
    let rad = r * (1. - r * r / (s * s)).sqrt();
    (0..=200)
        .map(|i| {
            let t = i as f64 * std::f64::consts::TAU / 200.;
            project(add(
                c,
                scale(add(scale(u, t.cos()), scale(v, t.sin())), rad),
            ))
        })
        .collect()
}
fn render(
    row: &Value,
    raw: &[u16],
    e: Ellipse,
    points: &[[f64; 2]],
    rays: TheoreticalEllipseExplanations,
    result: &Value,
    out: &Path,
) -> Result<()> {
    let frame = &row["frame"];
    let w = n(&frame["width"]) as usize;
    let h = n(&frame["height"]) as usize;
    let origin = [n(&frame["sensor_x"]) as u32, n(&frame["sensor_y"]) as u32];
    let color = preview::color_preview(raw, w, h, origin[0], origin[1], 100, None);
    let bgra = color
        .into_iter()
        .flat_map(|v| {
            [
                (v & 255) as u8,
                ((v >> 8) & 255) as u8,
                ((v >> 16) & 255) as u8,
                255,
            ]
        })
        .collect::<Vec<_>>();
    let mut c = Canvas::new(1800, 730)?;
    c.clear();
    c.text(
        20.,
        32.,
        24.,
        WHITE,
        &format!(
            "Sclera surface support | {} | record {} | eye {} sequence {}",
            row["provider"].as_str().unwrap(),
            row["record"],
            row["eye"],
            row["sequence"]
        ),
    );
    c.text(20.,62.,17.,MUTED,"Area-consistent input. Both poses see the same predicted sclera samples. Radius is optimized continuously within the declared range.");
    for k in 0..3 {
        let x = 20. + 594. * k as f64;
        let y = 112.;
        let s = 570. / w as f64;
        let hh = h as f64 * s;
        c.text(
            x,
            99.,
            21.,
            WHITE,
            [
                "Observed RAW and fitted iris",
                "Hypothesis A: best supported globe",
                "Hypothesis B: best supported globe",
            ][k],
        );
        c.image(&bgra, w, h, x, y, 570., hh);
        c.clipped(x, y, 570., hh, |c| {
            c.path(
                &e.dense_points(200)
                    .iter()
                    .map(|&(u, v)| [x + s * u, y + s * v])
                    .collect::<Vec<_>>(),
                1.8,
                WHITE,
            );
            let pose = if k > 0 { Some(rays.rays[k - 1]) } else { None };
            let radius = if k > 0 {
                result["nominal"]["candidates"][k - 1]["radius_iris_units"].as_f64()
            } else {
                None
            };
            for &p in points {
                let inside = pose
                    .zip(radius)
                    .is_none_or(|(g, r)| hits(g, r, camera_ray(p)));
                c.dot(
                    x + (p[0] - origin[0] as f64) * s,
                    y + (p[1] - origin[1] as f64) * s,
                    1.6,
                    if inside { GREEN } else { ORANGE },
                    true,
                );
            }
            if let Some((pose, r)) = pose.zip(radius) {
                let g = center(pose, r);
                let center_px = project(g);
                let mapped = |p: [f64; 2]| {
                    [
                        x + (p[0] - origin[0] as f64) * s,
                        y + (p[1] - origin[1] as f64) * s,
                    ]
                };
                c.path(
                    &silhouette(g, r).into_iter().map(mapped).collect::<Vec<_>>(),
                    3.,
                    if k == 1 { CYAN } else { PINK },
                );
                let p = mapped(center_px);
                c.dot(p[0], p[1], 6., if k == 1 { CYAN } else { PINK }, false);
            }
        });
        if k > 0 {
            let v = &result["nominal"]["candidates"][k - 1];
            c.text(
                x,
                535.,
                18.,
                WHITE,
                &format!(
                    "Best containment {:.1}% | radius {:.3} iris units",
                    100. * v["coverage"].as_f64().unwrap_or(0.),
                    v["radius_iris_units"].as_f64().unwrap_or(0.)
                ),
            );
        }
    }
    c.text(
        20.,
        581.,
        19.,
        WHITE,
        &format!(
            "Nominal choice {} | robust across masks/radii {} | votes {}",
            result["nominal"]["choice"], result["robust_choice"], result["votes"]
        ),
    );
    c.text(20.,619.,17.,MUTED,"Green/orange dots are predicted sclera inside/outside each globe. White is the fitted iris; cyan/pink are hypothetical sphere silhouettes.");
    c.text(20.,654.,17.,MUTED,"Skin/glasses or wrong lid masks can invalidate containment. A conditional rejection assumes true sclera, valid camera rays and this sphere family.");
    c.text(20.,690.,17.,MUTED,"No physical sign labels or independent scale. Constant frontal-equivalent area does not certify localization. Display demosaicing only.");
    c.png(out)
}
pub fn run(area_dir: &str, fresh_dir: &str, output: &str) -> Result<()> {
    run_inner(area_dir, fresh_dir, output, None, None, None)
}
pub fn lighting_run(area_dir: &str, fresh_dir: &str, output: &str, limit: usize) -> Result<()> {
    run_inner(area_dir, fresh_dir, output, Some(limit), None, None)
}
pub fn temporal_run(area_dir: &str, fresh_dir: &str, output: &str, anatomy: Option<&str>) -> Result<()> {
    photometry::temporal::run(area_dir, fresh_dir, output, anatomy)
}
pub fn sclera_motion_run(area_dir: &str, fresh_dir: &str, output: &str, mode:&str) -> Result<()> {
    photometry::sclera_motion::run(area_dir, fresh_dir, output, mode)
}
pub fn lids_run(area_dir: &str, fresh_dir: &str, output: &str, limit: usize) -> Result<()> {
    run_inner(area_dir, fresh_dir, output, None, Some(limit), None)
}
pub fn anatomy_run(area_dir: &str, fresh_dir: &str, anatomy: &str, output: &str) -> Result<()> {
    run_inner(area_dir, fresh_dir, output, Some(0), None, Some(anatomy))
}
fn run_inner(
    area_dir: &str,
    fresh_dir: &str,
    output: &str,
    lighting: Option<usize>,
    lid_mode: Option<usize>,
    anatomy_dir: Option<&str>,
) -> Result<()> {
    let area = Path::new(area_dir);
    let fresh = Path::new(fresh_dir);
    let out = Path::new(output);
    if out.exists() {
        return Err("output exists".into());
    }
    let summary = load(&area.join("summary.json"))?;
    if summary["complete"] != true || summary["schema"] != "buttercup-area-first-focus-v1" {
        return Err("area-first input required".into());
    }
    let anatomy = anatomy_dir
        .map(|p| {
            anatomy_masks::AnatomyMasks::open(
                p,
                &archive::digest(&fs::read(area.join("summary.json"))?),
            )
        })
        .transpose()?;
    let fresh_bytes = fs::read(fresh.join("frames.jsonl"))?;
    if archive::digest(&fresh_bytes) != summary["fresh_frames_sha256"] {
        return Err("fresh masks/fits source changed".into());
    }
    let fresh_rows = BufReader::new(fresh_bytes.as_slice())
        .lines()
        .map(|l| Ok(serde_json::from_str::<Value>(&l?)?))
        .collect::<Result<Vec<_>>>()?;
    let fresh_by_id = fresh_rows
        .iter()
        .map(|r| (n(&r["record"]), r))
        .collect::<BTreeMap<_, _>>();
    let classes = rows(&area.join("classifications.jsonl"))?;
    let target_sources = classes
        .iter()
        .filter(|r| r["class"] == "multiple")
        .map(|r| n(&r["source"]))
        .collect::<BTreeSet<_>>();
    let by_id = classes
        .iter()
        .map(|r| {
            (
                (n(&r["record"]), r["provider"].as_str().unwrap().to_string()),
                r,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let inputs = rows(&area.join("retained-inputs.jsonl"))?;
    fs::create_dir(out)?;
    let controls = controls();
    fs::write(
        out.join("controls.json"),
        serde_json::to_vec_pretty(&controls)?,
    )?;
    let mut writer = BufWriter::new(fs::File::create(out.join("results.jsonl"))?);
    let mut bundles = BTreeMap::new();
    let mut counters = BTreeMap::new();
    let mut shown = BTreeSet::new();
    let mut index = vec![];
    let mut center_bounds = BTreeMap::<String, Vec<f64>>::new();
    let mut light_counts = BTreeMap::new();
    let mut light_writer = BufWriter::new(fs::File::create(out.join("photometry.jsonl"))?);
    let mut light_index = vec![];
    let mut light_controls = vec![];
    let mut paired = photometry::PairStream::default();
    let mut paired_writer = BufWriter::new(fs::File::create(out.join("paired-photometry.jsonl"))?);
    let mut paired_counts = BTreeMap::new();
    let mut lid_writer = BufWriter::new(fs::File::create(out.join("lids.jsonl"))?);
    let mut lid_counts = BTreeMap::new();
    let mut lid_index = vec![];
    let selected: Vec<_> = inputs
        .iter()
        .filter(|r| target_sources.contains(&n(&r["source"])))
        .filter(|r| anatomy.as_ref().is_none_or(|a| a.contains(n(&r["record"]))))
        .collect();
    if selected.is_empty() {
        return Err("no admitted rows with requested anatomy evidence".into());
    }
    // These are mask sensitivity levels, not equal calibrated confidence across
    // models. SAM's conventional zero-logit mask is 128; 153 checks erosion by
    // a stricter probability level. Preserve the historical Obelisk settings.
    let thresholds = if anatomy.is_some() {
        [128, 153]
    } else {
        [179, 230]
    };
    let step = lighting
        .or(lid_mode)
        .filter(|&x| x > 0)
        .map_or(1, |limit| selected.len().div_ceil(limit));
    let mut selected_index = 0usize;
    for row in &selected {
        assert_eq!(
            row["area_admission"]["accepted"], true,
            "non-admitted input reached surface analysis"
        );
        let id = n(&row["record"]);
        let name = row["provider"].as_str().unwrap();
        let class = by_id[&(id, name.to_string())]["class"].as_str().unwrap();
        let original = fresh_by_id[&id];
        let frame = &row["frame"];
        let fit = &row["fit"];
        let e = shape(&fit["ellipse"])?;
        if row["raw_sha256"] != original["raw_sha256"] || fit != &original[name]["fit"] {
            return Err("retained input no longer matches original fresh fit".into());
        }
        let masks = if let Some(a) = &anatomy {
            a.get(row)?
        } else {
            let masks = fs::read(fresh.join(original["obelisk"]["masks"].as_str().unwrap()))?;
            if masks.len() != 6 * 384 * 256
                || archive::digest(&masks) != original["obelisk"]["masks_sha256"]
            {
                return Err("mask identity mismatch".into());
            }
            masks
        };
        let mut sensor = e;
        sensor.center.0 += n(&frame["sensor_x"]) as f64;
        sensor.center.1 += n(&frame["sensor_y"]) as f64;
        let rays = TheoreticalEllipseExplanations::from_ellipse(sensor, [4000.; 2], [4000., 3000.])
            .ok_or("retained geometry missing")?;
        let mut variants = vec![];
        let mut nominal = Value::Null;
        let mut nominal_points = vec![];
        let mut votes = [0usize; 2];
        let mut available = 0usize;
        for threshold in thresholds {
            for erode in [0, 2] {
                let pixels = samples(&masks, frame, e, threshold, erode);
                for radii in [[1.8, 2.4], [1.5, 2.8]] {
                    let candidates = rays.rays.map(|p| best_coverage(p, &pixels, radii));
                    let choice = if pixels.len() >= 32 {
                        select(&candidates[0], &candidates[1])
                    } else {
                        None
                    };
                    if pixels.len() >= 32 {
                        available += 1;
                    }
                    if let Some(k) = choice {
                        votes[k] += 1;
                    }
                    let v = json!({"sclera_threshold":threshold,"erode_mask_pixels":erode,"samples":pixels.len(),"radius_range":radii,"candidates":candidates,"choice":choice});
                    if threshold == thresholds[0] && erode == 2 && radii[0] == 1.8 {
                        nominal = v.clone();
                        nominal_points = pixels.clone();
                    }
                    variants.push(v);
                }
            }
        }
        let robust = if available == 8 && votes[0] == 8 {
            Some(0)
        } else if available == 8 && votes[1] == 8 {
            Some(1)
        } else {
            None
        };
        let separation = center_separation(rays, [1.8, 2.4]);
        let result = json!({"record":id,"provider":name,"source":row["source"],"epoch":row["epoch"],"eye":row["eye"],"sequence":row["sequence"],"source_ns":row["source_ns"],"raw_sha256":row["raw_sha256"],"original_focus_class":class,"area_admission":row["area_admission"],"nominal":nominal,"variants":variants,"available_variants":available,"votes":votes,"robust_choice":robust,"center_separation":separation,"broad_center_separation":center_separation(rays,[1.5,2.8]),"physical_sign_truth":null});
        count(&mut counters, format!("{name}/{class}/total"));
        if result["nominal"]["choice"].is_number() {
            count(&mut counters, format!("{name}/{class}/nominal_choice"));
        }
        if robust.is_some() {
            count(&mut counters, format!("{name}/{class}/robust_choice"));
        }
        if class == "multiple" {
            center_bounds.entry(name.to_string()).or_default().push(
                separation["sufficient_independent_center_error_radius_px_strictly_less_than"]
                    .as_f64()
                    .unwrap(),
            );
        }
        let tag = format!(
            "{name}/{class}/{}",
            if robust.is_some() {
                "robust"
            } else if result["nominal"]["choice"].is_number() {
                "nominal"
            } else {
                "abstain"
            }
        );
        let show = shown.insert(tag.clone()) && index.len() < 16;
        let do_lighting = lighting.is_some() && selected_index % step == 0;
        let do_lids = lid_mode.is_some() && selected_index % step == 0;
        selected_index += 1;
        if show || do_lighting || do_lids {
            let path = row["raw_source"].as_str().unwrap();
            if !bundles.contains_key(path) {
                bundles.insert(path.to_string(), BundleSource::open(Path::new(path))?);
            }
            let bytes = bundles[path].read_range(
                row["stream_entry"].as_str().unwrap(),
                n(&frame["offset"]),
                n(&frame["length"]) as usize,
            )?;
            if archive::digest(&bytes) != row["raw_sha256"] {
                return Err("RAW hash mismatch".into());
            }
            let raw = raw10::try_unpack_raw10(
                &bytes,
                n(&frame["width"]) as usize,
                n(&frame["height"]) as usize,
                n(&frame["stride"]) as usize,
            )?;
            if show {
                let image = format!("surface-{id}-{name}.png");
                render(
                    row,
                    &raw,
                    e,
                    &nominal_points,
                    rays,
                    &result,
                    &out.join(&image),
                )?;
                index.push(
                    json!({"record":id,"provider":name,"class":class,"tag":tag,"image":image}),
                );
            }
            if do_lighting {
                let show_light = light_index.len() < 8;
                let (light, control) = photometry::analyze(
                    row, &raw, &masks, e, rays, out, show_light, true, thresholds,
                )?;
                count(&mut light_counts, format!("{name}/{class}/total"));
                for key in [
                    "usable",
                    "stable_nominal_preference",
                    "beats_gradient_all_variants",
                ] {
                    if light[key] == true {
                        count(&mut light_counts, format!("{name}/{class}/{key}"));
                    }
                }
                if show_light {
                    light_index.push(json!({"record":id,"provider":name,"image":format!("photometry-{id}-{name}.png")}));
                }
                if let Some(v) = control {
                    light_controls.push(v);
                }
                serde_json::to_writer(&mut light_writer, &light)?;
                writeln!(light_writer)?;
                if let Some(pair) = paired.push(row, &raw, &masks, e, rays, thresholds) {
                    count(&mut paired_counts, format!("{name}/exact_pairs"));
                    if pair["stable_preference"] == true {
                        count(&mut paired_counts, format!("{name}/stable_preference"));
                    }
                    if pair["stable_beats_gradient"] == true {
                        count(&mut paired_counts, format!("{name}/stable_beats_gradient"));
                    }
                    serde_json::to_writer(&mut paired_writer, &pair)?;
                    writeln!(paired_writer)?;
                }
                if selected_index % 25 == 0 {
                    eprintln!("PHOTOMETRY {selected_index}/{}", selected.len());
                }
            }
            if do_lids {
                let show_lids =
                    lid_index.len() < 4 || (selected_index - 1) % selected.len().div_ceil(12) == 0;
                let value = lids::analyze(row, &raw, &masks, e, rays, out, show_lids)?;
                if value["mask_semantic_check"]["passed"] != true {
                    count(
                        &mut lid_counts,
                        format!("{name}/{class}/invalid_lid_semantics"),
                    );
                }
                for key in [
                    "total",
                    "all_variants_have_uncropped_canthi",
                    "stable_40px_single_choice",
                ] {
                    if key == "total" || value[key] == true {
                        count(&mut lid_counts, format!("{name}/{class}/{key}"));
                    }
                }
                if show_lids {
                    lid_index.push(json!({"record":id,"provider":name,"image":format!("lids-{id}-{name}.png")}));
                }
                serde_json::to_writer(&mut lid_writer, &value)?;
                writeln!(lid_writer)?;
            }
        }
        serde_json::to_writer(&mut writer, &result)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    light_writer.flush()?;
    paired_writer.flush()?;
    lid_writer.flush()?;
    fs::write(
        out.join("lids-review.json"),
        serde_json::to_vec_pretty(&lid_index)?,
    )?;
    fs::write(
        out.join("photometry-controls.json"),
        serde_json::to_vec_pretty(&light_controls)?,
    )?;
    fs::write(
        out.join("photometry-review.json"),
        serde_json::to_vec_pretty(&light_index)?,
    )?;
    let bounds=center_bounds.into_iter().map(|(p,mut v)|{v.sort_by(f64::total_cmp);(p,json!({"frames":v.len(),"min_px":v[0],"median_px":v[v.len()/2],"max_px":v[v.len()-1],"p05_px":v[v.len()/20]}))}).collect::<BTreeMap<_,_>>();
    let result = json!({"complete":true,"counts":counters,"lid_counts":lid_counts,"photometry_counts":light_counts,"paired_photometry_counts":paired_counts,"unpaired_admitted_rows":paired.pending(),"photometry_sampling_stride":step,"center_error_radius_sufficiency":bounds,"target_sources":target_sources,"controls":controls,
        "anatomy_override":anatomy.as_ref().map(|a|&a.meta),"evaluated_provider_rows":selected.len(),"area_admitted_rows_without_requested_anatomy":inputs.iter().filter(|r|target_sources.contains(&n(&r["source"])) && anatomy.as_ref().is_some_and(|a|!a.contains(n(&r["record"])))).count(),
        "mask_source":if anatomy.is_some() {"fresh official SAM white of the eye; no lid veto"} else {"historical Obelisk sclera with lid-probability veto"},
        "mask_thresholds_uint8":thresholds,"threshold_interpretation":"uncalibrated model-specific mask sensitivity; values are not equivalent correctness probabilities across providers",
        "policy":"Only upstream area-admitted inputs; same predicted sclera mask for both poses, connected-to-limbus support. See mask_source and anatomy_override for mask/veto provenance. Continuous single shared sphere radius maximizes containment. Nominal rejection: winner >=98%, other <=80%, >=32 samples. Robust requires same choice for all 8 mask/radius variants. Conditional geometry only, unverified masks and sphere family, no physical sign accuracy.",
        "area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"fresh_frames_sha256":archive::digest(&fresh_bytes),"executable_sha256":archive::digest(&fs::read(std::env::current_exe()?)?)});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    fs::write(
        out.join("visual-review.json"),
        serde_json::to_vec_pretty(&index)?,
    )?;
    eprintln!(
        "SURFACE SUPPORT {}",
        serde_json::to_string(&result["counts"])?
    );
    Ok(())
}
