//! Oracle study: are approximately planar lid arcs sufficient sign evidence?
//! Synthetic lid curves are NEVER described as detected or human-labelled lids.
use super::*;
use buttercup_eye_tracking::geometry::{fit_plane, PlaneFit};
#[path = "lid_proposals.rs"]
pub(crate) mod proposals;
#[path = "native_lids.rs"]
pub(crate) mod native;
#[path = "lid_visual.rs"]
pub(crate) mod visual;

#[derive(Clone)]
struct Curve {
    pixels: Vec<[f64; 2]>,
}
#[derive(Clone, serde::Serialize)]
struct Fit {
    radius: f64,
    rms_px: f64,
    maximum_px: f64,
    planes: Vec<([f64; 3], f64)>,
}

#[derive(Clone, serde::Serialize)]
struct SilhouetteWitness {
    /// Strictly positive values exclude every radius in the declared family
    /// if total point/camera/model image error is smaller than this bound.
    outside_margin_lower_bound_px: f64,
    point_sensor_px: [f64; 2],
    lipschitz_per_iris_offset: f64,
    grid_allowance_px: f64,
}
fn silhouette_witness(p: GazeRay, curves: &[Curve]) -> SilhouetteWitness {
    let (lo, hi) = ((1.5f64 * 1.5 - 1.).sqrt(), (2.8f64 * 2.8 - 1.).sqrt());
    let normal_length = norm(p.direction);
    let minimum_distance = norm(p.origin_iris_radii) - hi * normal_length;
    assert!(minimum_distance > 2.8);
    // |d angle(q,G)/dd| <= |n|/M. With r=sqrt(1+d²), |r'|<=1:
    // |d asin(r/|G|)/dd| <= (1/M+rmax*|n|/M²)/sqrt(1-(rmax/M)²).
    // This bounds interpolation over the CONTINUOUS radius family. It is not
    // an estimated real-world error bound on the native input conic/camera.
    let m = minimum_distance;
    let lipschitz = normal_length / m
        + (1. / m + 2.8 * normal_length / (m * m)) / (1. - (2.8 / m).powi(2)).sqrt();
    let grid_allowance = lipschitz * (hi - lo) / 64.;
    let family = (0..=32)
        .map(|j| {
            let d = lo + (hi - lo) * j as f64 / 32.;
            let g = sub(p.origin_iris_radii, scale(p.direction, d));
            (unit(g), ((1. + d * d).sqrt() / norm(g)).asin())
        })
        .collect::<Vec<_>>();
    let mut best = SilhouetteWitness {
        outside_margin_lower_bound_px: f64::NEG_INFINITY,
        point_sensor_px: [0.; 2],
        lipschitz_per_iris_offset: lipschitz,
        grid_allowance_px: grid_allowance * 4000.,
    };
    for xy in curves.iter().flat_map(|c| &c.pixels) {
        let q = camera_ray(*xy);
        let sampled = family
            .iter()
            .map(|(g, alpha)| dot(q, *g).clamp(-1., 1.).acos() - alpha)
            .fold(f64::INFINITY, f64::min);
        // Pixel-to-ray angular displacement is <= pixel displacement / f.
        let lower = (sampled - grid_allowance - 1e-12) * 4000.;
        if lower > best.outside_margin_lower_bound_px {
            best.outside_margin_lower_bound_px = lower;
            best.point_sensor_px = *xy;
        }
    }
    best
}
fn lift(p: GazeRay, r: f64, xy: [f64; 2]) -> Option<V3> {
    let q = camera_ray(xy);
    let g = center(p, r);
    let t = dot(q, g);
    let d = t * t - dot(g, g) + r * r;
    if t <= 0. || d < 0. {
        return None;
    }
    Some(scale(q, t - d.sqrt()))
}
fn conic_error(p: [f64; 2], g: V3, r: f64, plane: PlaneFit) -> f64 {
    // Project the plane/sphere intersection analytically as a conic. This
    // first-order pixel distance is a diagnostic, not calibrated likelihood.
    let q = [(p[0] - 4000.) / 4000., (p[1] - 3000.) / 4000., -1.];
    let m = plane.normal;
    let k = plane.offset;
    let a = dot(m, q);
    let b = dot(g, q);
    let c = dot(g, g) - r * r;
    let f = k * k * dot(q, q) - 2. * k * a * b + c * a * a;
    let du = (2. * k * k * q[0] - 2. * k * (m[0] * b + a * g[0]) + 2. * c * a * m[0]) / 4000.;
    let dv = (2. * k * k * q[1] - 2. * k * (m[1] * b + a * g[1]) + 2. * c * a * m[1]) / 4000.;
    f.abs() / du.hypot(dv).max(1e-12)
}
fn at_radius(p: GazeRay, r: f64, curves: &[Curve]) -> Option<Fit> {
    let g = center(p, r);
    let mut errors = vec![];
    let mut planes = vec![];
    for curve in curves {
        let points = curve
            .pixels
            .iter()
            .map(|&q| lift(p, r, q))
            .collect::<Option<Vec<_>>>()?;
        let plane = fit_plane(&points)?;
        errors.extend(curve.pixels.iter().map(|&xy| conic_error(xy, g, r, plane)));
        planes.push((plane.normal, plane.offset));
    }
    let rms_px = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    if !rms_px.is_finite() {
        return None;
    }
    Some(Fit {
        radius: r,
        rms_px,
        maximum_px: errors.into_iter().fold(0., f64::max),
        planes,
    })
}
fn profile(p: GazeRay, curves: &[Curve], radii: [f64; 2]) -> Option<Fit> {
    // A declared finite nuisance search, not a continuous exclusion proof.
    (0..=32)
        .filter_map(|i| at_radius(p, radii[0] + (radii[1] - radii[0]) * i as f64 / 32., curves))
        .min_by(|a, b| a.rms_px.total_cmp(&b.rms_px))
}
fn rotate_roll(p: V3, roll: f64) -> V3 {
    let (s, c) = roll.sin_cos();
    [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
}

fn visibility_agreement(row: &Value, curves: &[Curve]) -> Value {
    let origin = [
        n(&row["frame"]["sensor_x"]) as f64,
        n(&row["frame"]["sensor_y"]) as f64,
    ];
    let sorted = curves
        .iter()
        .map(|c| {
            let mut p = c.pixels.clone();
            p.sort_by(|a, b| a[0].total_cmp(&b[0]));
            p
        })
        .collect::<Vec<_>>();
    let interpolate = |p: &[[f64; 2]], x: f64| {
        let j = p.partition_point(|p| p[0] < x);
        if j == 0 || j == p.len() {
            return None;
        }
        let a = p[j - 1];
        let b = p[j];
        let t = (x - a[0]) / (b[0] - a[0]).max(1e-9);
        Some(a[1] + t * (b[1] - a[1]))
    };
    let points = row["fit"]["points"]
        .as_array()
        .expect("observed contour points");
    let mut compared = 0;
    let mut inside = 0;
    for p in points {
        let x = origin[0] + p[0].as_f64().unwrap();
        let y = origin[1] + p[1].as_f64().unwrap();
        if let Some((upper, lower)) = interpolate(&sorted[0], x).zip(interpolate(&sorted[1], x)) {
            compared += 1;
            inside += usize::from(y >= upper - 3. && y <= lower + 3.);
        }
    }
    let coverage = compared as f64 / points.len().max(1) as f64;
    let included = inside as f64 / compared.max(1) as f64;
    json!({"compared_points":compared,"recorded_visible_points":points.len(),"comparison_coverage":coverage,"fraction_inside_opening":included,"point_allowance_px":3.,"compatible":coverage>=0.7&&included>=0.9,"not_human_verified_visibility":true})
}
fn perturb(seed: u64, j: usize, axis: u64) -> f64 {
    let mut x = seed.wrapping_mul(0x9e3779b97f4a7c15) ^ (j as u64).wrapping_mul(0x85ebca6b) ^ axis;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    2. * ((x >> 11) as f64 / (1u64 << 53) as f64) - 1.
}
fn synthetic(
    p: GazeRay,
    frame: &Value,
    configuration: usize,
    noise: f64,
    deform: f64,
    seed: u64,
) -> Option<Vec<Curve>> {
    let r = 2.15;
    let g = center(p, r);
    let (upper, lower, roll, offset) = match configuration {
        0 => (35f64, 15f64, 0f64, 0.),
        1 => (55., 25., 12., 0.12),
        _ => (20., 40., -12., -0.12),
    };
    let origin = [n(&frame["sensor_x"]) as f64, n(&frame["sensor_y"]) as f64];
    let size = [n(&frame["width"]) as f64, n(&frame["height"]) as f64];
    let mut curves = vec![];
    for (lid, beta, sign) in [(0, upper, 1.), (1, lower, -1.)] {
        let beta = beta.to_radians();
        let roll = roll.to_radians();
        let m = rotate_roll([0., beta.cos(), sign * beta.sin()], roll);
        let u = rotate_roll([1., 0., 0.], roll);
        let v = unit(cross(m, u));
        let h = offset * r * sign;
        let rr = (r * r - h * h).sqrt();
        let mut pixels = vec![];
        for j in 0..=96 {
            let theta = std::f64::consts::PI * j as f64 / 96.;
            let relative = add(
                scale(m, h),
                scale(sub(scale(u, theta.cos()), scale(v, theta.sin())), rr),
            );
            // Nonplanarity bends the circle on the same sphere, rather than
            // moving the surface off it. This remains a declared oracle model.
            let relative = scale(
                unit(add(relative, scale(m, deform * (3. * theta).sin()))),
                r,
            );
            let point = add(g, relative);
            if dot(unit(relative), scale(unit(point), -1.)) < 0.1 {
                continue;
            }
            let mut xy = project(point);
            if !(0..2).all(|k| xy[k] >= origin[k] + 8. && xy[k] < origin[k] + size[k] - 8.) {
                continue;
            }
            xy[0] += noise * perturb(seed, j + lid * 97, 1);
            xy[1] += noise * perturb(seed, j + lid * 97, 2);
            pixels.push(xy);
        }
        if pixels.len() < 12 {
            return None;
        }
        let x_span = pixels
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max)
            - pixels.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        if x_span < 40. {
            return None;
        }
        curves.push(Curve { pixels });
    }
    Some(curves)
}
fn choose(fits: &[Option<Fit>; 2]) -> Option<usize> {
    let [Some(a), Some(b)] = fits else {
        return None;
    };
    if a.rms_px <= 2. && b.rms_px >= 5. && b.rms_px >= 2. * a.rms_px {
        Some(0)
    } else if b.rms_px <= 2. && a.rms_px >= 5. && a.rms_px >= 2. * b.rms_px {
        Some(1)
    } else {
        None
    }
}
fn render(
    row: &Value,
    raw: &[u16],
    rays: TheoreticalEllipseExplanations,
    curves: &[Curve],
    truth: Option<usize>,
    fits: &[Option<Fit>; 2],
    path: &Path,
) -> Result<()> {
    let frame = &row["frame"];
    let origin = [n(&frame["sensor_x"]) as f64, n(&frame["sensor_y"]) as f64];
    let (w, h) = (n(&frame["width"]) as usize, n(&frame["height"]) as usize);
    let color = preview::color_preview(raw, w, h, origin[0] as u32, origin[1] as u32, 100, None);
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
    let mut c = Canvas::new(1800, 800)?;
    c.clear();
    c.text(
        20.,
        35.,
        25.,
        WHITE,
        if truth.is_some() {
            "Planar-lid oracle study — SYNTHETIC LIDS, not detected boundaries"
        } else {
            "SAM opening boundary proposals — UNVERIFIED masks and anatomical assumption"
        },
    );
    c.text(
        20.,
        70.,
        18.,
        MUTED,
        &format!(
            "Native iris from {} record {} / {}",
            row["provider"].as_str().unwrap(),
            row["record"],
            truth.map_or("No physical sign truth".to_owned(), |j| format!(
                "synthetic globe {} / R=2.15 iris radii",
                if j == 0 { "A" } else { "B" }
            ))
        ),
    );
    c.text(
        20.,
        112.,
        20.,
        WHITE,
        "Actual RAW and unchanged fitted iris",
    );
    c.image(&bgra, w, h, 20., 130., 560., h as f64 * 560. / w as f64);
    let e = shape(&row["fit"]["ellipse"])?;
    c.path(
        &e.dense_points(180)
            .iter()
            .map(|&(x, y)| [20. + x * 560. / w as f64, 130. + y * 560. / w as f64])
            .collect::<Vec<_>>(),
        1.5,
        WHITE,
    );
    for pose in 0..2 {
        let x = 615. + pose as f64 * 590.;
        let y = 130.;
        c.text(
            x,
            112.,
            19.,
            WHITE,
            &format!(
                "{}: identical {} pixels and iris",
                if pose == 0 { "A" } else { "B" },
                if truth.is_some() {
                    "synthetic lid"
                } else {
                    "mask boundary"
                }
            ),
        );
        let xy = |p: [f64; 2]| {
            [
                x + (p[0] - origin[0]) * 560. / w as f64,
                y + (p[1] - origin[1]) * 560. / w as f64,
            ]
        };
        c.rect(x, y, 560., h as f64 * 560. / w as f64, [0.08, 0.11, 0.14]);
        if truth.is_none() {
            c.image(&bgra, w, h, x, y, 560., h as f64 * 560. / w as f64);
        }
        c.path(
            &e.dense_points(180)
                .iter()
                .map(|&(u, v)| xy([u + origin[0], v + origin[1]]))
                .collect::<Vec<_>>(),
            1.,
            MUTED,
        );
        for (i, curve) in curves.iter().enumerate() {
            for &p in &curve.pixels {
                let p = xy(p);
                c.dot(p[0], p[1], 2.4, if i == 0 { CYAN } else { PINK }, true);
            }
        }
        let center = xy(project(center(
            rays.rays[pose],
            fits[pose].as_ref().map_or(2.15, |f| f.radius),
        )));
        c.cross(center[0], center[1], 9., ORANGE);
        c.text(
            x,
            548.,
            18.,
            WHITE,
            &fits[pose]
                .as_ref()
                .map_or("No complete curve support in radius grid".into(), |f| {
                    format!(
                        "Best R={:.3}, planar-circle residual {:.3} px",
                        f.radius, f.rms_px
                    )
                }),
        );
        let witness = silhouette_witness(rays.rays[pose], curves);
        c.text(
            x,
            580.,
            16.,
            MUTED,
            &format!(
                "Outside-family margin lower bound: {:.1} px",
                witness.outside_margin_lower_bound_px
            ),
        );
    }
    let visibility = visibility_agreement(row, curves);
    c.text(
        20.,
        620.,
        19.,
        WHITE,
        &format!(
            "Opening agrees with observed contour: {} | coverage {:.0}% | contained {:.0}%",
            visibility["compatible"],
            100. * visibility["comparison_coverage"].as_f64().unwrap(),
            100. * visibility["fraction_inside_opening"].as_f64().unwrap()
        ),
    );
    c.text(20.,656.,18.,MUTED,if truth.is_some() {"Only the iris hypotheses come from RAW. Lid points are procedurally generated to test information and sensitivity."} else {"Cyan/pink dots are mask-derived upper/lower boundaries; curves may be iris, skin or sclera instead of true lid margins."});
    c.text(20.,680.,18.,MUTED,"Both hypotheses get identical lid points and one shared globe radius across upper/lower curves. Each lid has its own fitted plane.");
    c.text(20.,718.,18.,MUTED,if truth.is_some() {"Perfect synthetic circular lids do not establish anatomical planarity, real boundary accuracy, or sign accuracy on this recording."} else {"No sign is admitted: fitting a sphere to an unverified mask does not establish anatomical accuracy or lid planarity."});
    c.png(path)
}

fn admitted_inputs(area: &Path, fresh_dir: &str) -> Result<Vec<Value>> {
    admitted_with_context(area, fresh_dir, false)
}

pub(super) fn admitted_with_context(
    area: &Path,
    fresh_dir: &str,
    include_context: bool,
) -> Result<Vec<Value>> {
    let summary = load(&area.join("summary.json"))?;
    if summary["complete"] != true || summary["schema"] != "buttercup-area-first-focus-v1" {
        return Err("completed area-first input required".into());
    }
    let fresh_bytes = fs::read(Path::new(fresh_dir).join("frames.jsonl"))?;
    if archive::digest(&fresh_bytes) != summary["fresh_frames_sha256"] {
        return Err("fresh fit source changed".into());
    }
    let fresh = rows(&Path::new(fresh_dir).join("frames.jsonl"))?
        .into_iter()
        .map(|r| (n(&r["record"]), r))
        .collect::<BTreeMap<_, _>>();
    let classes = rows(&area.join("classifications.jsonl"))?
        .into_iter()
        .map(|r| {
            (
                (n(&r["record"]), r["provider"].as_str().unwrap().to_owned()),
                r,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let sources = classes
        .values()
        .filter(|r| r["class"] == "multiple")
        .map(|r| n(&r["source"]))
        .collect::<BTreeSet<_>>();
    let input = rows(&area.join("retained-inputs.jsonl"))?
        .into_iter()
        .filter(|r| {
            if include_context {
                sources.contains(&n(&r["source"]))
            } else {
                classes[&(n(&r["record"]), r["provider"].as_str().unwrap().to_owned())]["class"]
                    == "multiple"
            }
        })
        .collect::<Vec<_>>();
    for row in &input {
        let id = n(&row["record"]);
        let provider = row["provider"].as_str().ok_or("provider")?;
        if row["area_admission"]["accepted"] != true
            || row["raw_sha256"] != fresh[&id]["raw_sha256"]
            || row["fit"] != fresh[&id][provider]["fit"]
        {
            return Err("admission or source identity mismatch".into());
        }
    }
    Ok(input)
}

pub(crate) fn run(area_dir: &str, fresh_dir: &str, output: &str) -> Result<()> {
    let area = Path::new(area_dir);
    let out = Path::new(output);
    if out.exists() {
        return Err("output exists".into());
    }
    let input = admitted_inputs(area, fresh_dir)?;
    fs::create_dir(out)?;
    let mut file = BufWriter::new(fs::File::create(out.join("oracle.jsonl"))?);
    let mut counts = BTreeMap::new();
    let mut bundles = BTreeMap::new();
    let mut reviews = vec![];
    let mut shown = BTreeSet::new();
    let scenarios = [
        ("perfect", 0., 0.),
        ("pixel_noise_1", 1., 0.),
        ("pixel_noise_2", 2., 0.),
        ("pixel_noise_4", 4., 0.),
        ("bend_001", 0., 0.01),
        ("bend_005", 0., 0.05),
        ("bend_010", 0., 0.1),
    ];
    let mut maximum_true_perfect_error = 0f64;
    for row in &input {
        let id = n(&row["record"]);
        let provider = row["provider"].as_str().unwrap();
        let frame = &row["frame"];
        let mut e = shape(&row["fit"]["ellipse"])?;
        e.center.0 += n(&frame["sensor_x"]) as f64;
        e.center.1 += n(&frame["sensor_y"]) as f64;
        let rays = TheoreticalEllipseExplanations::from_ellipse(e, [4000.; 2], [4000., 3000.])
            .ok_or("circle solve")?;
        for truth in 0..2 {
            for configuration in 0..3 {
                for &(tag, noise, deform) in &scenarios {
                    let Some(curves) =
                        synthetic(rays.rays[truth], frame, configuration, noise, deform, id)
                    else {
                        count(
                            &mut counts,
                            format!("{provider}/{tag}/insufficient_visible_synthetic_arcs"),
                        );
                        writeln!(
                            file,
                            "{}",
                            serde_json::to_string(
                                &json!({"record":id,"provider":provider,"scenario":tag,"configuration":configuration,"synthetic_truth":truth,"outcome":"insufficient_visible_synthetic_arcs","synthetic_lids":true,"physical_sign_truth":null})
                            )?
                        )?;
                        continue;
                    };
                    if tag == "perfect" {
                        let true_fit = at_radius(rays.rays[truth], 2.15, &curves)
                            .ok_or("known synthetic sphere lost support")?;
                        maximum_true_perfect_error =
                            maximum_true_perfect_error.max(true_fit.rms_px);
                        if true_fit.rms_px > 1e-5 {
                            return Err(format!(
                                "exact planar-circle control failed: {id}/{provider}/{truth}: {}",
                                true_fit.rms_px
                            )
                            .into());
                        }
                    }
                    let fits = rays.rays.map(|p| profile(p, &curves, [1.5, 2.8]));
                    let visibility = visibility_agreement(row, &curves);
                    let witnesses = rays.rays.map(|p| silhouette_witness(p, &curves));
                    if witnesses[truth].outside_margin_lower_bound_px
                        > noise * std::f64::consts::SQRT_2 + 1e-6
                    {
                        return Err("continuous silhouette bound falsely excluded its known synthetic sphere".into());
                    }
                    let containment_sensitivity=[0.,2.,5.,10.,20.,40.].map(|error| {
                    let rejected=witnesses.each_ref().map(|w|w.outside_margin_lower_bound_px>error);
                    let choice=match rejected {[true,false]=>Some(1),[false,true]=>Some(0),_=>None};
                    let outcome=match choice {Some(j) if j==truth=>"correct",Some(_)=>"wrong",None if rejected[0]&&rejected[1]=>"both_rejected",None=>"ambiguous"};
                    count(&mut counts,format!("{provider}/{tag}/silhouette_{error}px/{outcome}"));
                    if visibility["compatible"]==true {count(&mut counts,format!("{provider}/{tag}/visibility_compatible_silhouette_{error}px/{outcome}"));}
                    json!({"total_error_allowance_px":error,"rejected":rejected,"conditional_choice":choice,"outcome":outcome})
                });
                    let choice = choose(&fits);
                    let status = match choice {
                        Some(j) if j == truth => "correct_conditional_choice",
                        Some(_) => "wrong_conditional_choice",
                        None if fits.iter().any(Option::is_none) => "missing_sphere_support",
                        None => "ambiguous",
                    };
                    count(&mut counts, format!("{provider}/{tag}/{status}"));
                    let result = json!({"record":id,"provider":provider,"eye":row["eye"],"source":row["source"],"raw_sha256":row["raw_sha256"],"area_admission":row["area_admission"],"scenario":tag,"configuration":configuration,"synthetic_truth":truth,"fits":fits,"conditional_choice":choice,"outcome":status,"curve_samples":curves.iter().map(|c|c.pixels.len()).collect::<Vec<_>>(),"silhouette_witnesses":witnesses,"containment_sensitivity":containment_sensitivity,"visible_iris_compatibility":visibility,"synthetic_lids":true,"physical_sign_truth":null});
                    writeln!(file, "{}", serde_json::to_string(&result)?)?;
                    let key = format!(
                        "{provider}/{}/{truth}/{status}/{}",
                        row["eye"], visibility["compatible"]
                    );
                    if tag == "perfect" && reviews.len() < 16 && shown.insert(key) {
                        let source = row["raw_source"].as_str().unwrap();
                        if !bundles.contains_key(source) {
                            bundles
                                .insert(source.to_owned(), BundleSource::open(Path::new(source))?);
                        }
                        let bytes = bundles[source].read_range(
                            row["stream_entry"].as_str().unwrap(),
                            n(&frame["offset"]),
                            n(&frame["length"]) as usize,
                        )?;
                        if archive::digest(&bytes) != row["raw_sha256"] {
                            return Err("RAW changed".into());
                        }
                        let raw = raw10::try_unpack_raw10(
                            &bytes,
                            n(&frame["width"]) as usize,
                            n(&frame["height"]) as usize,
                            n(&frame["stride"]) as usize,
                        )?;
                        let name = format!("oracle-{id}-{provider}-{truth}-{configuration}.png");
                        render(
                            row,
                            &raw,
                            rays,
                            &curves,
                            Some(truth),
                            &fits,
                            &out.join(&name),
                        )?;
                        reviews.push(json!({"image":name,"record":id,"provider":provider,"truth":truth,"outcome":status,"visible_iris_compatibility":visibility}));
                    }
                }
            }
        }
    }
    file.flush()?;
    fs::write(
        out.join("review.json"),
        serde_json::to_vec_pretty(&reviews)?,
    )?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"schema":"buttercup-planar-lid-oracle-v1","counts":counts,"real_ambiguous_provider_rows":input.len(),"synthetic_globe_radius":2.15,"searched_radius_range":[1.5,2.8],"radius_grid_points":33,"winner_maximum_rms_px":2.,"loser_minimum_rms_px":5.,"maximum_true_perfect_residual_px":maximum_true_perfect_error,"assumption":"Each synthetic lid is a planar circular section of one sphere, with separately declared on-sphere bending and pixel perturbations.","real_lid_boundaries_measured":false,"physical_sign_truth":null,"area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"retained_inputs_sha256":archive::digest(&fs::read(area.join("retained-inputs.jsonl"))?),"executable_sha256":archive::digest(&fs::read(std::env::current_exe()?)?)}),
        )?,
    )?;
    Ok(())
}
