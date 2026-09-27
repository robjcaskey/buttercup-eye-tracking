//! Source-matched scleral texture availability and rigid-globe diagnostics.
//! Historical masks are proposals; a correlated patch is not a vessel label.
use super::*;
use buttercup_eye_tracking::raw_motion_octrees::{
    NativeGlobalSimilarityTracker, NativePatchCorrespondence,
};
use std::{sync::Arc, time::Instant};
#[path = "rigid_motion.rs"]
mod rigid;

struct Frame {
    row: Value,
    raw: Arc<Vec<u16>>,
    support: [Vec<bool>; 2],
    rays: TheoreticalEllipseExplanations,
    ns: u64,
}

fn patch_inside(f: &Frame, p: [f32; 2], threshold: usize) -> bool {
    let m = &f.row["frame"];
    let (w, h) = (n(&m["width"]) as i32, n(&m["height"]) as i32);
    let x = (p[0] as f64 - n(&m["sensor_x"]) as f64).round() as i32;
    let y = (p[1] as f64 - n(&m["sensor_y"]) as f64).round() as i32;
    if x < 0 || y < 0 || x >= w || y >= h {
        return false;
    }
    f.support[threshold][y as usize * w as usize + x as usize]
}

fn eroded_native(selected: &[bool], frame: &Value) -> Vec<bool> {
    let (w, h) = (n(&frame["width"]) as usize, n(&frame["height"]) as usize);
    let stride = w + 1;
    let mut prefix = vec![0u32; stride * (h + 1)];
    for y in 0..h {
        let mut row = 0;
        for x in 0..w {
            let mx = ((x as f64 + 0.5) * 384. / w as f64) as usize;
            let my = ((y as f64 + 0.5) * 256. / h as f64) as usize;
            row += u32::from(!selected[my * 384 + mx]);
            prefix[(y + 1) * stride + x + 1] = prefix[y * stride + x + 1] + row;
        }
    }
    let mut allowed = vec![false; w * h];
    // Radius-4 matcher + native CFA support + rounding/subpixel safety.
    for y in 8..h.saturating_sub(8) {
        for x in 8..w.saturating_sub(8) {
            allowed[y * w + x] = prefix[(y + 9) * stride + x + 9]
                + prefix[(y - 8) * stride + x - 8]
                == prefix[(y - 8) * stride + x + 9] + prefix[(y + 9) * stride + x - 8];
        }
    }
    allowed
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    (!values.is_empty()).then(|| values[values.len() / 2])
}

#[derive(serde::Serialize)]
struct PolarScore {
    radius: f64,
    median_absolute_polar_angle_drift_degrees: f64,
    maximum_degrees: f64,
}

fn polar_profiles(
    a: &Frame,
    b: &Frame,
    matches: &[NativePatchCorrespondence],
) -> [[Vec<PolarScore>; 2]; 2] {
    [0, 1].map(|sa| {
        [0, 1].map(|sb| {
            let mut radii = vec![];
            for ri in 0..=26 {
                let r = 1.5 + ri as f64 * 0.05;
                let ga = scale(center(a.rays.rays[sa], r), 1. / r);
                let gb = scale(center(b.rays.rays[sb], r), 1. / r);
                let angles = matches
                    .iter()
                    .map(|m| {
                        let u = normal(ga, camera_ray(m.previous_sensor_px.map(f64::from)))?;
                        let v = normal(gb, camera_ray(m.current_sensor_px.map(f64::from)))?;
                        Some(
                            (dot(u, a.rays.rays[sa].direction).clamp(-1., 1.).acos()
                                - dot(v, b.rays.rays[sb].direction).clamp(-1., 1.).acos())
                            .abs()
                            .to_degrees(),
                        )
                    })
                    .collect::<Option<Vec<_>>>();
                if let Some(errors) = angles.filter(|v| !v.is_empty()) {
                    radii.push(PolarScore {
                        radius: r,
                        median_absolute_polar_angle_drift_degrees: median(errors.clone()).unwrap(),
                        maximum_degrees: errors.iter().copied().fold(0f64, f64::max),
                    });
                }
            }
            radii
        })
    })
}

fn invariant(a: &Frame, b: &Frame, matches: &[NativePatchCorrespondence]) -> Value {
    let p = polar_profiles(a, b, matches);
    let families = [0, 1].map(|sa| {
        [0, 1].map(
            |sb| json!({"source_branch":sa,"target_branch":sb,"shared_radius_profile":p[sa][sb]}),
        )
    });
    json!({"matches":matches.len(),"families":families,"selected_sign":null,"interpretation":"Necessary invariant under rigid surface attachment and a spherical globe. Radius profiling is diagnostic, not a sign certificate; unknown feature identity, localization and conic errors remain."})
}

fn basis(axis: V3) -> [V3; 3] {
    let u = unit(cross(
        axis,
        if axis[1].abs() < 0.9 {
            [0., 1., 0.]
        } else {
            [1., 0., 0.]
        },
    ));
    [u, cross(axis, u), axis]
}

fn polar_controls(a: &Frame, b: &Frame, with_rigid: bool) -> Result<Vec<Value>> {
    let f = &a.row["frame"];
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let mut source = vec![];
    for y in (8..h - 8).step_by(12) {
        for x in (8..w - 8).step_by(12) {
            if a.support[0][y * w + x] {
                source.push([
                    (x + n(&f["sensor_x"]) as usize) as f32,
                    (y + n(&f["sensor_y"]) as usize) as f32,
                ]);
            }
        }
    }
    let mut cases = vec![];
    for sa in 0..2 {
        for sb in 0..2 {
            let r = 2.15;
            let ga = scale(center(a.rays.rays[sa], r), 1. / r);
            let gb = center(b.rays.rays[sb], r);
            let ba = basis(a.rays.rays[sa].direction);
            let bb = basis(b.rays.rays[sb].direction);
            let (s, c) = 0.05f64.sin_cos();
            let clean = source
                .iter()
                .filter_map(|&p| {
                    let u = normal(ga, camera_ray(p.map(f64::from)))?;
                    let xyz = ba.map(|v| dot(v, u));
                    let v = add(
                        add(
                            scale(bb[0], c * xyz[0] - s * xyz[1]),
                            scale(bb[1], s * xyz[0] + c * xyz[1]),
                        ),
                        scale(bb[2], xyz[2]),
                    );
                    let target = project(add(gb, scale(v, r)));
                    if dot(v, camera_ray(target)) >= -0.15 {
                        return None;
                    }
                    let q = target.map(|v| v as f32);
                    if !patch_inside(b, q, 0) {
                        return None;
                    }
                    Some(NativePatchCorrespondence {
                        previous_sensor_px: p,
                        current_sensor_px: q,
                        photometric_score: 1.,
                        distinct_match_margin: 1.,
                        global_similarity_inlier: false,
                    })
                })
                .collect::<Vec<_>>();
            for noise in [0., 1., 2.] {
                if clean.len() < 6 {
                    cases.push(json!({"source_branch":sa,"target_branch":sb,"noise_bound_per_axis_px":noise,"usable":false,"matches":clean.len(),"reason":"fewer than six synthetic visible surface points"}));
                    continue;
                }
                let matches = clean
                    .iter()
                    .enumerate()
                    .map(|(i, m)| {
                        let mut m = *m;
                        for k in 0..4 {
                            let value = (((i as u64 * 191
                                + k as u64 * 83
                                + n(&a.row["record"]) * 47)
                                % 101) as f64
                                - 50.)
                                / 50.
                                * noise;
                            if k < 2 {
                                m.previous_sensor_px[k] += value as f32;
                            } else {
                                m.current_sensor_px[k - 2] += value as f32;
                            }
                        }
                        m
                    })
                    .collect::<Vec<_>>();
                let profiles = polar_profiles(a, b, &matches);
                let true_nominal = profiles[sa][sb]
                    .iter()
                    .find(|p| (p.radius - r).abs() < 1e-8);
                let Some(true_nominal) = true_nominal else {
                    if noise == 0. {
                        return Err("noise-free synthetic true sphere missing".into());
                    }
                    cases.push(json!({"source_branch":sa,"target_branch":sb,"noise_bound_per_axis_px":noise,"usable":false,"matches":matches.len(),"reason":"pixel perturbation moves at least one ray outside the true nominal sphere"}));
                    continue;
                };
                if noise == 0. && true_nominal.maximum_degrees > 0.002 {
                    return Err("rigid polar-angle control failed".into());
                }
                let errors = [0, 1].map(|i| {
                    [0, 1].map(|j| {
                        profiles[i][j]
                            .iter()
                            .map(|p| p.median_absolute_polar_angle_drift_degrees)
                            .min_by(f64::total_cmp)
                    })
                });
                let mut choices = (0..2)
                    .flat_map(|i| (0..2).filter_map(move |j| errors[i][j].map(|v| (v, i, j))))
                    .collect::<Vec<_>>();
                choices.sort_by(|a, b| a.0.total_cmp(&b.0));
                let best = choices[0];
                let mut case = json!({"source_branch":sa,"target_branch":sb,"noise_bound_per_axis_px":noise,"usable":true,"matches":matches.len(),"true_nominal_maximum_drift_degrees":true_nominal.maximum_degrees,"profiled_median_errors_degrees":errors,"lowest_error_pair":[best.1,best.2],"lowest_error_pair_correct":best.1==sa&&best.2==sb,"lowest_error_target_correct":best.2==sb,"runner_up_gap_degrees":choices.get(1).map(|v|v.0-best.0),"sign_admission":null});
                if with_rigid {
                    let results = [16., 24.].map(|guard| {
                        let mut result = rigid::analyze(a, b, &matches, guard);
                        if let Some(object) = result.as_object_mut() {
                            object.remove("folds");
                            object.remove("full_data_fits_for_display_only");
                        }
                        result
                    });
                    case["rigid"] = json!(results);
                }
                cases.push(case);
            }
        }
    }
    Ok(cases)
}

fn render(a: &Frame, b: &Frame, matches: &[NativePatchCorrespondence], path: &Path) -> Result<()> {
    let mut c = Canvas::new(1800, 840)?;
    c.clear();
    c.text(
        22.,
        36.,
        26.,
        WHITE,
        &format!(
            "Scleral texture | {} | RAW {} to {} | {:.1} ms",
            a.row["provider"].as_str().unwrap(),
            a.row["record"],
            b.row["record"],
            (b.ns - a.ns) as f64 / 1e6
        ),
    );
    c.text(22.,72.,18.,MUTED,"Raw matches are selected without a 3D sign. Bright points pass both-frame sclera support and stricter photometric checks.");
    for (side, f) in [a, b].into_iter().enumerate() {
        let m = &f.row["frame"];
        let (w, h) = (n(&m["width"]) as usize, n(&m["height"]) as usize);
        let origin = [n(&m["sensor_x"]) as f64, n(&m["sensor_y"]) as f64];
        let rgb =
            preview::color_preview(&f.raw, w, h, origin[0] as u32, origin[1] as u32, 100, None);
        let bgra = rgb
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
        let x = 22. + side as f64 * 884.;
        let y = 116.;
        let s = 852. / w as f64;
        c.image(&bgra, w, h, x, y, 852., h as f64 * s);
        let e = shape(&f.row["fit"]["ellipse"])?;
        c.path(
            &e.dense_points(160)
                .iter()
                .map(|&(u, v)| [x + u * s, y + v * s])
                .collect::<Vec<_>>(),
            1.,
            WHITE,
        );
        for (i, t) in matches.iter().enumerate() {
            let p = if side == 0 {
                t.previous_sensor_px
            } else {
                t.current_sensor_px
            }
            .map(f64::from);
            let white =
                patch_inside(a, t.previous_sensor_px, 0) && patch_inside(b, t.current_sensor_px, 0);
            let strict = white && t.photometric_score >= 0.75 && t.distinct_match_margin >= 0.10;
            let color = if strict {
                ORANGE
            } else if white {
                [0.3, 0.95, 1.]
            } else {
                [0.5, 0.5, 0.5]
            };
            let q = [x + (p[0] - origin[0]) * s, y + (p[1] - origin[1]) * s];
            c.dot(q[0], q[1], if strict { 4. } else { 2. }, color, false);
            if white {
                c.text(q[0] + 6., q[1] - 5., 14., color, &format!("{}", i + 1));
                if side == 0 {
                    let d = [
                        t.current_sensor_px[0] - t.previous_sensor_px[0],
                        t.current_sensor_px[1] - t.previous_sensor_px[1],
                    ];
                    let delta = [d[0] as f64 * s * 20., d[1] as f64 * s * 20.];
                    let mut fraction = 1f64;
                    for axis in 0..2 {
                        let (lo, hi) = if axis == 0 {
                            (x, x + 852.)
                        } else {
                            (y, y + h as f64 * s)
                        };
                        if delta[axis] > 0. {
                            fraction = fraction.min((hi - q[axis]) / delta[axis]);
                        }
                        if delta[axis] < 0. {
                            fraction = fraction.min((lo - q[axis]) / delta[axis]);
                        }
                    }
                    c.path(
                        &[q, [q[0] + delta[0] * fraction, q[1] + delta[1] * fraction]],
                        1.5,
                        color,
                    );
                }
            }
        }
    }
    c.text(22.,718.,18.,WHITE,"Orange: strict interior matches. Cyan: below strict quality. Gray: other native matches. Motion x20, clipped to image.");
    c.text(22.,756.,18.,MUTED,"Full matching patches must stay outside the iris and inside both sclera proposals. Reflections and conjunctival motion remain possible.");
    c.text(22.,794.,18.,MUTED,"Same native CFA-neutral patch matcher as the scale diagnostic; no global affine is applied. These points do not establish physical sign truth.");
    c.png(path)
}

pub(crate) fn run(area_dir: &str, fresh_dir: &str, output: &str, mode: &str) -> Result<()> {
    if !["whole", "whole-dense", "sclera", "rigid"].contains(&mode) {
        return Err("mode must be whole, whole-dense, sclera or rigid".into());
    }
    let started = Instant::now();
    let area = Path::new(area_dir);
    let fresh = Path::new(fresh_dir);
    let out = Path::new(output);
    if out.exists() {
        return Err("output exists".into());
    }
    let input = lid_circle::admitted_with_context(area, fresh_dir, true)?;
    let originals = rows(&fresh.join("frames.jsonl"))?
        .into_iter()
        .map(|r| (n(&r["record"]), r))
        .collect::<BTreeMap<_, _>>();
    let classes = rows(&area.join("classifications.jsonl"))?
        .into_iter()
        .map(|r| {
            (
                (n(&r["record"]), r["provider"].as_str().unwrap().to_owned()),
                r["class"].clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::<_, Vec<Value>>::new();
    for row in input {
        for field in [
            "frame",
            "source_ns",
            "source",
            "epoch",
            "eye",
            "raw_source",
            "stream_entry",
        ] {
            if row[field] != originals[&n(&row["record"])][field] {
                return Err(format!("motion source mismatch: {field}").into());
            }
        }
        groups
            .entry((
                row["provider"].as_str().unwrap().to_owned(),
                n(&row["source"]),
                n(&row["epoch"]),
                n(&row["eye"]),
            ))
            .or_default()
            .push(row);
    }
    fs::create_dir(out)?;
    let mut cache = BTreeMap::<String, Arc<Vec<u16>>>::new();
    let mut bundles = BTreeMap::new();
    let mut writer = BufWriter::new(fs::File::create(out.join("tracks.jsonl"))?);
    let mut oracle = BufWriter::new(fs::File::create(out.join("polar-controls.jsonl"))?);
    let mut counts = BTreeMap::new();
    let mut control_counts = BTreeMap::new();
    let mut review = vec![];
    let mut shown = BTreeSet::new();
    let mut rigid_shown = BTreeSet::new();
    let mut total = 0;
    for (group, mut rows) in groups {
        rows.sort_by_key(|r| r["source_ns"].as_str().unwrap().parse::<u64>().unwrap());
        let mut tracker = NativeGlobalSimilarityTracker::default();
        tracker.retain_diagnostic_correspondences(true);
        let mut prior: Option<Frame> = None;
        for row in rows {
            let key = row["raw_sha256"].as_str().unwrap().to_owned();
            if !cache.contains_key(&key) {
                cache.insert(
                    key.clone(),
                    Arc::new(temporal::read_raw(&row, &mut bundles)?),
                );
            }
            let raw = cache[&key].clone();
            let original = &originals[&n(&row["record"])];
            let masks =
                fs::read(fresh.join(original["obelisk"]["masks"].as_str().ok_or("mask path")?))?;
            if masks.len() != 6 * 384 * 256
                || archive::digest(&masks) != original["obelisk"]["masks_sha256"]
            {
                return Err("mask hash/shape".into());
            }
            let f = &row["frame"];
            let e = shape(&row["fit"]["ellipse"])?;
            let support = [179, 230].map(|t| eroded_native(&sample_mask(&masks, f, e, t, 2), f));
            let mut sensor = e;
            sensor.center.0 += n(&f["sensor_x"]) as f64;
            sensor.center.1 += n(&f["sensor_y"]) as f64;
            let rays =
                TheoreticalEllipseExplanations::from_ellipse(sensor, [4000.; 2], [4000., 3000.])
                    .ok_or("native conic pair")?;
            let ns = row["source_ns"].as_str().unwrap().parse::<u64>()?;
            let fresh_pair = prior
                .as_ref()
                .is_some_and(|p| ns > p.ns && ns - p.ns <= 300_000_000);
            if !fresh_pair {
                tracker.clear();
            }
            let current = Frame {
                row,
                raw,
                support,
                rays,
                ns,
            };
            let mut variants = vec![];
            let f = &current.row["frame"];
            let matches = if mode == "whole" {
                tracker.observe(
                    current.raw.clone(),
                    n(&f["width"]) as usize,
                    n(&f["height"]) as usize,
                    n(&f["sensor_x"]) as u32,
                    n(&f["sensor_y"]) as u32,
                );
                tracker.diagnostic_correspondences().to_vec()
            } else {
                tracker.observe_diagnostic_where(
                    current.raw.clone(),
                    n(&f["width"]) as usize,
                    n(&f["height"]) as usize,
                    n(&f["sensor_x"]) as u32,
                    n(&f["sensor_y"]) as u32,
                    [20, 16],
                    |p| {
                        mode == "whole-dense"
                            || prior.as_ref().is_some_and(|a| patch_inside(a, p, 0))
                    },
                    |p| mode == "whole-dense" || patch_inside(&current, p, 0),
                )
            };
            if let Some(a) = prior.as_ref().filter(|_| fresh_pair) {
                if mode == "sclera" || mode == "rigid" {
                    let controls = polar_controls(a, &current, mode == "rigid")?;
                    for c in &controls {
                        let prefix = format!("{}/{}px", group.0, c["noise_bound_per_axis_px"]);
                        count(&mut control_counts, format!("{prefix}/cases"));
                        if c["usable"] == true {
                            count(&mut control_counts, format!("{prefix}/usable"));
                            count(
                                &mut control_counts,
                                format!(
                                    "{prefix}/{}",
                                    if c["lowest_error_pair_correct"] == true {
                                        "pair_rank_correct"
                                    } else {
                                        "pair_rank_wrong"
                                    }
                                ),
                            );
                            count(
                                &mut control_counts,
                                format!(
                                    "{prefix}/{}",
                                    if c["lowest_error_target_correct"] == true {
                                        "target_rank_correct"
                                    } else {
                                        "target_rank_wrong"
                                    }
                                ),
                            );
                        } else {
                            count(
                                &mut control_counts,
                                format!("{prefix}/{}", c["reason"].as_str().unwrap()),
                            );
                        }
                    }
                    writeln!(
                        oracle,
                        "{}",
                        serde_json::to_string(
                            &json!({"source_record":a.row["record"],"target_record":current.row["record"],"provider":group.0,"known_synthetic_radius":2.15,"known_synthetic_torsion_radians":0.05,"cases":controls,"synthetic_only":true,"scope":"Actual admitted conic/mask layouts; projected feature correspondences are synthetic, not tracked RAW texture or measured sign truth."})
                        )?
                    )?;
                }
                for t in 0..2 {
                    let interior = matches
                        .iter()
                        .copied()
                        .filter(|m| {
                            patch_inside(a, m.previous_sensor_px, t)
                                && patch_inside(&current, m.current_sensor_px, t)
                        })
                        .collect::<Vec<_>>();
                    let strict = interior
                        .iter()
                        .copied()
                        .filter(|m| m.photometric_score >= 0.75 && m.distinct_match_margin >= 0.10)
                        .collect::<Vec<_>>();
                    let span = [0, 1].map(|j| {
                        let lo = strict
                            .iter()
                            .map(|m| m.previous_sensor_px[j])
                            .fold(f32::INFINITY, f32::min);
                        let hi = strict
                            .iter()
                            .map(|m| m.previous_sensor_px[j])
                            .fold(f32::NEG_INFINITY, f32::max);
                        if strict.is_empty() {
                            0.
                        } else {
                            hi - lo
                        }
                    });
                    let mut variant = json!({"mask_threshold":([179,230][t]),"interior_matches":interior.len(),"strict_matches":strict.len(),"strict_source_span_px":span,"rigid_polar_invariant":invariant(a,&current,&strict),"matches":interior.iter().map(|m|json!({"source_sensor":m.previous_sensor_px,"target_sensor":m.current_sensor_px,"score":m.photometric_score,"margin":m.distinct_match_margin,"strict":m.photometric_score>=0.75&&m.distinct_match_margin>=0.10})).collect::<Vec<_>>()});
                    if mode == "rigid" {
                        let results =
                            [16., 24.].map(|guard| rigid::analyze(a, &current, &interior, guard));
                        let result = &results[0];
                        let conditional = !result["conditional_pair_preference"].is_null();
                        if t == 0
                            && result["usable"] == true
                            && result["full_data_fits_for_display_only"]
                                .as_array()
                                .is_some_and(|v| v.iter().all(Value::is_object))
                            && (conditional || rigid_shown.insert((group.0.clone(), group.3)))
                            && review.len() < 32
                        {
                            let name = format!(
                                "rigid-{}-{}-{}.png",
                                a.row["record"], current.row["record"], group.0
                            );
                            rigid::render_fit(a, &current, &interior, result, &out.join(&name))?;
                            review.push(json!({"image":name,"record":current.row["record"],"provider":group.0,"rigid":true,"conditional":conditional}));
                        }
                        variant["rigid"] = json!(results);
                    }
                    variants.push(variant);
                }
                let bucket = if n(&variants[0]["strict_matches"]) >= 6 {
                    "six_or_more"
                } else if n(&variants[0]["strict_matches"]) > 0 {
                    "one_to_five"
                } else {
                    "zero"
                };
                if shown.insert((group.0.clone(), group.3, bucket)) && review.len() < 18 {
                    let name = format!(
                        "texture-{}-{}-{}.png",
                        a.row["record"], current.row["record"], group.0
                    );
                    render(a, &current, &matches, &out.join(&name))?;
                    review.push(json!({"image":name,"record":current.row["record"],"provider":group.0,"bucket":bucket}));
                }
                count(&mut counts, format!("{}/{bucket}", group.0));
            } else {
                count(&mut counts, format!("{}/first_or_gap", group.0));
            }
            let class = &classes[&(n(&current.row["record"]), group.0.clone())];
            let mut result = json!({"record":current.row["record"],"provider":group.0,"class":class,"raw_sha256":current.row["raw_sha256"],"source_ns":current.row["source_ns"],"area_admission":current.row["area_admission"],"fresh_pair":fresh_pair,"previous_record":prior.as_ref().filter(|_|fresh_pair).map(|p|&p.row["record"]),"native_matches":matches.len(),"variants":variants,"sign_choice":null,"physical_sign_truth":null});
            if mode == "rigid" {
                let choices = variants
                    .iter()
                    .flat_map(|v| v["rigid"].as_array().into_iter().flatten())
                    .map(|v| &v["conditional_pair_preference"])
                    .collect::<Vec<_>>();
                result["stable_conditional_pair_preference"] = if choices.len() == 4
                    && !choices[0].is_null()
                    && choices.iter().all(|c| *c == choices[0])
                {
                    choices[0].clone()
                } else {
                    Value::Null
                };
            }
            writeln!(writer, "{}", serde_json::to_string(&result)?)?;
            prior = Some(current);
            total += 1;
            if total % 50 == 0 {
                eprintln!(
                    "SCLERA TEXTURE {total} rows, {:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
        }
    }
    writer.flush()?;
    oracle.flush()?;
    fs::write(out.join("review.json"), serde_json::to_vec_pretty(&review)?)?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"counts":counts,"synthetic_control_counts":control_counts,"rows":total,"unique_raws":cache.len(),"seconds":started.elapsed().as_secs_f64(),"area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"retained_inputs_sha256":archive::digest(&fs::read(area.join("retained-inputs.jsonl"))?),"executable_sha256":archive::digest(&fs::read("/proc/self/exe")?),"native_patch_radius":4,"support_margin_px":8,"strict_minimum_score":0.75,"strict_minimum_distinct_margin":0.10,"maximum_pair_gap_ms":300,"feature_selection":mode,"feature_grid":(if mode=="whole" {[10,8]} else {[20,16]}),"policy":"Same native matcher; explicit whole or sclera-restricted seeding. Full patch support in both exposures. Historical masks diagnostic only. No material truth, sign admission, temporal holds, independent scale or live changes."}),
        )?,
    )?;
    Ok(())
}
