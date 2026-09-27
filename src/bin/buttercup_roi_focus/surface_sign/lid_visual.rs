//! Provisional visual lid estimates, kept apart from canonical human labels.
//! Tests aperture containment without assuming planar anatomical lid curves.
use super::*;

fn neutral(raw: &[u16], w: usize, h: usize) -> Vec<u8> {
    // Symmetric period-four CFA suppression for display only. No resampling.
    let weights = [1., 2., 2., 2., 1.];
    let mut tmp = vec![0.; w * h];
    let mut smooth = vec![0.; w * h];
    for y in 0..h {
        for x in 0..w {
            tmp[y * w + x] = weights
                .iter()
                .enumerate()
                .map(|(j, a)| {
                    let xx = (x as isize + j as isize - 2).clamp(0, w as isize - 1) as usize;
                    a * raw[y * w + xx] as f64 / 8.
                })
                .sum();
        }
    }
    for y in 0..h {
        for x in 0..w {
            smooth[y * w + x] = weights
                .iter()
                .enumerate()
                .map(|(j, a)| {
                    let yy = (y as isize + j as isize - 2).clamp(0, h as isize - 1) as usize;
                    a * tmp[yy * w + x] / 8.
                })
                .sum();
        }
    }
    let mut sorted = smooth.clone();
    sorted.sort_by(f64::total_cmp);
    let (lo, hi) = (sorted[sorted.len() / 100], sorted[sorted.len() * 99 / 100]);
    smooth
        .into_iter()
        .flat_map(|v| {
            let c = (255. * ((v - lo) / (hi - lo).max(1.)).clamp(0., 1.).sqrt()) as u8;
            [c, c, c, 255]
        })
        .collect()
}

fn context_image(root: &Path, f: &Value) -> Result<Vec<u8>> {
    let bytes = fs::read(
        root.join("archive")
            .join(f["path"].as_str().ok_or("context path")?),
    )?;
    if archive::digest(&bytes) != f["raw_sha256"] {
        return Err("visual context RAW hash mismatch".into());
    }
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let raw = raw10::try_unpack_raw10(&bytes, w, h, n(&f["stride"]) as usize)?;
    Ok(neutral(&raw, w, h))
}

fn raw_sheet(root: &Path, target: &Value, out: &Path) -> Result<()> {
    let frames = target["context_frames"]
        .as_array()
        .ok_or("native context")?;
    if frames.len() != 3 || target["target_position"] != 1 {
        return Err("before/target/after required".into());
    }
    let images = frames
        .iter()
        .map(|f| context_image(root, f))
        .collect::<Result<Vec<_>>>()?;
    let mut c = Canvas::new(1340, 830)?;
    c.clear();
    c.text(
        24.,
        34.,
        24.,
        WHITE,
        &format!(
            "Native RAW review | record {} | eye {} | no predicted geometry",
            target["record"], target["eye"]
        ),
    );
    c.text(24., 67., 17., MUTED, "Neutral display only. Coordinates are native pixels; estimates are not human labels or physical sign truth.");
    c.image(&images[1], 420, 280, 40., 140., 840., 560.);
    for x in (0..=400).step_by(40) {
        c.text(34. + x as f64 * 2., 124., 16., MUTED, &format!("{x}"));
    }
    for y in (0..=280).step_by(40) {
        c.text(4., 145. + y as f64 * 2., 15., MUTED, &format!("{y}"));
    }
    for (i, pos) in [0, 2].into_iter().enumerate() {
        let y = 120. + i as f64 * 330.;
        c.text(
            900.,
            y - 14.,
            18.,
            WHITE,
            &format!(
                "{} | {:+.1} ms",
                if pos == 0 { "before" } else { "after" },
                frames[pos]["relative_timing_ms"].as_f64().unwrap()
            ),
        );
        c.image(&images[pos], 420, 280, 900., y, 420., 280.);
    }
    c.text(
        40.,
        759.,
        18.,
        MUTED,
        "Mark only visually identifiable upper/lower margins; preserve gaps and unclear portions.",
    );
    c.text(40., 797., 18., MUTED, "Review-set selection is fixed before these estimates. No source rejected by the upstream area gate is added.");
    c.png(out)
}

fn parse_curves(estimate: &Value, row: &Value) -> Result<Vec<Curve>> {
    if estimate["raw_sha256"] != row["raw_sha256"] || estimate["record"] != row["record"] {
        return Err("provisional estimate/source identity mismatch".into());
    }
    let f = &row["frame"];
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
    ["upper", "lower"]
        .iter()
        .map(|name| {
            let points: Vec<[f64; 2]> = serde_json::from_value(estimate[*name].clone())?;
            if points.windows(2).any(|p| p[0][0] >= p[1][0])
                || points.iter().any(|p| {
                    !p[0].is_finite()
                        || !p[1].is_finite()
                        || p[0] < 0.
                        || p[1] < 0.
                        || p[0] >= n(&f["width"]) as f64
                        || p[1] >= n(&f["height"]) as f64
                })
            {
                return Err(
                    "provisional coordinates must be finite, inside RAW and x-ordered".into(),
                );
            }
            Ok(Curve {
                pixels: points
                    .into_iter()
                    .map(|p| [p[0] + origin[0], p[1] + origin[1]])
                    .collect(),
            })
        })
        .collect()
}

fn evaluate(rays: TheoreticalEllipseExplanations, curves: &[Curve]) -> Value {
    let points = curves
        .iter()
        .flat_map(|c| &c.pixels)
        .copied()
        .collect::<Vec<_>>();
    let sensitivity = [0., 2., 5., 10., 20.].map(|error| {
        let expanded = points.iter().flat_map(|p| {
            [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)].map(|(dx, dy)| [p[0] + dx * error, p[1] + dy * error])
        }).collect::<Vec<_>>();
        let coverage = rays.rays.map(|p| best_coverage(p, &expanded, [1.5, 2.8]));
        let compatible = coverage.each_ref().map(|v| v["coverage"].as_f64() == Some(1.));
        json!({"coordinate_box_half_width_px":error,"candidates":coverage,"both_cover_every_box_corner":compatible==[true,true],"entire_box_covered":compatible})
    });
    let fits = rays.rays.map(|p| profile(p, curves, [1.5, 2.8]));
    let hinges = [0, 1].map(|i| fits[i].as_ref().map(|f| hinge(rays.rays[i], f)));
    json!({"sensitivity":sensitivity,"planar_lid_diagnostic":fits,"plane_intersection_diagnostic":hinges,"planarity_preference":choose(&fits),"admitted_sign":null,"physical_sign_truth":null,
        "interpretation":"One shared globe radius per candidate covers every proposed point box. Convex perspective sphere silhouettes then cover the boxes and all line segments between them. This is a conditional existence witness, not measured anatomy or real sign recovery."})
}

fn hinge(p: GazeRay, fit: &Fit) -> Value {
    let [(a, ha), (b, hb)] = fit.planes.as_slice() else {
        return Value::Null;
    };
    let g = center(p, fit.radius);
    let c = dot(*a, *b);
    let determinant = 1. - c * c;
    if determinant <= 1e-8 {
        return json!({"available":false,"reason":"near parallel planes"});
    }
    let (u, v) = (ha - dot(*a, g), hb - dot(*b, g));
    let delta = add(
        scale(*a, (u - c * v) / determinant),
        scale(*b, (v - c * u) / determinant),
    );
    let axis_origin = add(g, delta);
    let axis = unit(cross(*a, *b));
    let distance = norm(delta);
    let endpoints = (distance <= fit.radius).then(|| {
        let extent = (fit.radius * fit.radius - distance * distance)
            .max(0.)
            .sqrt();
        [
            sub(axis_origin, scale(axis, extent)),
            add(axis_origin, scale(axis, extent)),
        ]
    });
    json!({"available":true,"globe_radius_iris_units":fit.radius,"axis_origin":axis_origin,"axis_direction":axis,
        "distance_from_globe_center_in_globe_radii":distance/fit.radius,"sphere_intersections":endpoints,
        "projected_intersections_sensor_px":endpoints.map(|v|v.map(project)),
        "interpretation":"Intersection of independently best-fit lid planes only. It is not measured anatomy, a validated hinge constraint, or an exhaustive constrained fit."})
}

fn scene(c: &mut Canvas, p: GazeRay, radius: f64, curves: &[Curve], x: f64, color: [f64; 3]) {
    let g = center(p, radius);
    let yaw = 35f64.to_radians();
    let pitch = (-12f64).to_radians();
    let xy = |q: V3| {
        let q = sub(q, p.origin_iris_radii);
        let xx = q[0] * yaw.cos() + q[2] * yaw.sin();
        let zz = -q[0] * yaw.sin() + q[2] * yaw.cos();
        [
            x + 280. + 65. * xx,
            840. + 65. * (q[1] * pitch.cos() - zz * pitch.sin()),
        ]
    };
    for axis in 0..3 {
        let points = (0..=96)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 96.;
                let mut q = g;
                q[(axis + 1) % 3] += radius * t.cos();
                q[(axis + 2) % 3] += radius * t.sin();
                xy(q)
            })
            .collect::<Vec<_>>();
        c.path(&points, 0.8, [0.25, 0.32, 0.37]);
    }
    let u = unit(cross(
        p.direction,
        if p.direction[0].abs() < 0.9 {
            [1., 0., 0.]
        } else {
            [0., 1., 0.]
        },
    ));
    let v = unit(cross(p.direction, u));
    c.path(
        &(0..=96)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 96.;
                xy(add(
                    p.origin_iris_radii,
                    add(scale(u, t.cos()), scale(v, t.sin())),
                ))
            })
            .collect::<Vec<_>>(),
        2.,
        WHITE,
    );
    c.line(
        xy(p.origin_iris_radii),
        xy(add(p.origin_iris_radii, scale(p.direction, 2.))),
        2.,
        color,
    );
    let center = xy(g);
    c.dot(center[0], center[1], 4., color, false);
    for (lid, curve) in curves.iter().enumerate() {
        for &pixel in &curve.pixels {
            if let Some(q) = lift(p, radius, pixel) {
                let reprojection = project(q);
                assert!((reprojection[0] - pixel[0]).hypot(reprojection[1] - pixel[1]) < 1e-6);
                let q = xy(q);
                c.dot(q[0], q[1], 2.7, if lid == 0 { ORANGE } else { CYAN }, true);
            }
        }
    }
    c.text(
        x,
        1070.,
        16.,
        MUTED,
        "Oblique 3D view: native iris, globe and lifted points",
    );
    c.text(
        x,
        1097.,
        16.,
        MUTED,
        "35 degree yaw / -12 degree pitch; one common scale",
    );
}

fn overlay(
    row: &Value,
    raw: &[u8],
    rays: TheoreticalEllipseExplanations,
    curves: &[Curve],
    result: &Value,
    out: &Path,
) -> Result<()> {
    let f = &row["frame"];
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
    let e = shape(&row["fit"]["ellipse"])?;
    let mut c = Canvas::new(1800, 1360)?;
    c.clear();
    c.text(
        22.,
        36.,
        25.,
        WHITE,
        &format!(
            "Provisional visual lid check | {} record {}",
            row["provider"].as_str().unwrap(),
            row["record"]
        ),
    );
    c.text(22., 76., 18., MUTED, "Dots: assistant visual estimates, not human labels. White: fitted iris. Colored globes: alternative 3D hypotheses.");
    for panel in 0..3 {
        let (x, y, s) = (22. + panel as f64 * 598., 142., 560. / 420.);
        let title = if panel == 0 {
            "Same provisional boundary portions"
        } else if panel == 1 {
            "Candidate A"
        } else {
            "Candidate B"
        };
        c.text(x, 118., 22., WHITE, title);
        c.image(raw, 420, 280, x, y, 560., 280. * s);
        c.clipped(x, y, 560., 280. * s, |c| {
            c.path(
                &e.dense_points(160)
                    .iter()
                    .map(|&(u, v)| [x + u * s, y + v * s])
                    .collect::<Vec<_>>(),
                1.,
                WHITE,
            );
            if panel == 0 {
                for point in row["fit"]["points"].as_array().unwrap() {
                    c.dot(
                        x + point[0].as_f64().unwrap() * s,
                        y + point[1].as_f64().unwrap() * s,
                        1.5,
                        PINK,
                        true,
                    );
                }
            }
            for (lid, curve) in curves.iter().enumerate() {
                for p in &curve.pixels {
                    c.dot(
                        x + (p[0] - origin[0]) * s,
                        y + (p[1] - origin[1]) * s,
                        3.,
                        if lid == 0 { ORANGE } else { CYAN },
                        false,
                    );
                }
            }
            if panel != 0 {
                let k = panel - 1;
                let r = result["sensitivity"][3]["candidates"][k]["radius_iris_units"]
                    .as_f64()
                    .unwrap();
                let g = center(rays.rays[k], r);
                let color = if k == 0 { GREEN } else { PINK };
                c.path(
                    &silhouette(g, r)
                        .iter()
                        .map(|p| [x + (p[0] - origin[0]) * s, y + (p[1] - origin[1]) * s])
                        .collect::<Vec<_>>(),
                    2.,
                    color,
                );
                let p = project(g);
                c.dot(
                    x + (p[0] - origin[0]) * s,
                    y + (p[1] - origin[1]) * s,
                    5.,
                    color,
                    false,
                );
            }
        });
        if panel != 0 {
            let a = &result["sensitivity"][3]["candidates"][panel - 1];
            c.text(
                x,
                555.,
                19.,
                WHITE,
                &format!(
                    "10 px boxes: {:.1}% covered | R {:.3}",
                    100. * a["coverage"].as_f64().unwrap(),
                    a["radius_iris_units"].as_f64().unwrap()
                ),
            );
            scene(
                &mut c,
                rays.rays[panel - 1],
                a["radius_iris_units"].as_f64().unwrap(),
                curves,
                x,
                if panel == 1 { GREEN } else { PINK },
            );
        }
    }
    c.text(
        22.,
        609.,
        20.,
        WHITE,
        &format!(
            "Both candidates cover every 10 px test box: {}",
            result["sensitivity"][3]["both_cover_every_box_corner"]
        ),
    );
    c.text(22., 1180., 18., MUTED, "Visible lids alone do not constrain their hidden 3D shape. A planar-lid assumption is evaluated separately.");
    c.text(22., 1221., 18., MUTED, "The proposed points and anatomical sphere bounds are unverified; a compatible projection is not a correct sign.");
    c.text(22., 1263., 18., MUTED, "Native RAW identities and area eligibility match the prepared before/target/after set. No training labels are written.");
    c.text(22.,1305.,18.,MUTED,"Pink dots on left: model contour samples, not verified visible tissue. Sparse lid estimates do not establish the complete opening.");
    c.png(out)
}

pub(crate) fn run(
    area: &str,
    fresh: &str,
    review: &str,
    estimates: &str,
    output: &str,
) -> Result<()> {
    let (root, out) = (Path::new(review), Path::new(output));
    if out.exists() {
        return Err("output exists".into());
    }
    let manifest = load(&root.join("manifest.json"))?;
    if manifest["schema"] != "buttercup-native-lid-review-v1" || manifest["complete"] != true {
        return Err("native review manifest required".into());
    }
    if manifest["retained_inputs_sha256"]
        != archive::digest(&fs::read(Path::new(area).join("retained-inputs.jsonl"))?)
    {
        return Err("review area admission changed".into());
    }
    let targets = manifest["targets"].as_array().ok_or("review targets")?;
    let input = admitted_inputs(Path::new(area), fresh)?;
    let proposals = if estimates == "-" {
        None
    } else {
        let v = load(Path::new(estimates))?;
        if v["schema"] != "buttercup-assistant-visual-lid-estimates-v1"
            || v["human_reviewed"] != false
            || v["training_eligible"] != false
        {
            return Err("explicit provisional assistant estimates required".into());
        }
        Some(v)
    };
    fs::create_dir(out)?;
    let mut template = vec![];
    let mut results = vec![];
    let mut counts = BTreeMap::new();
    for target in targets {
        let record = n(&target["record"]);
        raw_sheet(root, target, &out.join(format!("raw-{record}.png")))?;
        template.push(json!({"record":record,"raw_sha256":target["raw_sha256"],"upper":[],"lower":[],"notes":"unreviewed"}));
        let Some(proposals) = &proposals else {
            continue;
        };
        let matching = proposals["estimates"]
            .as_array()
            .ok_or("estimates")?
            .iter()
            .filter(|v| v["record"] == record)
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err("exactly one estimate per selected target required".into());
        }
        let estimate = matching[0];
        for row in input.iter().filter(|r| r["record"] == record) {
            for field in ["raw_sha256", "source", "epoch", "eye", "sequence"] {
                if row[field] != target[field] {
                    return Err(format!("visual input identity mismatch {field}").into());
                }
            }
            let f = &row["frame"];
            if n(&f["width"]) != 420 || n(&f["height"]) != 280 {
                return Err("native 420x280 required".into());
            }
            let curves = parse_curves(estimate, row)?;
            if curves.iter().any(|c| c.pixels.len() < 6) {
                return Err("at least six provisional visible points per margin required for this diagnostic".into());
            }
            let mut e = shape(&row["fit"]["ellipse"])?;
            e.center.0 += n(&f["sensor_x"]) as f64;
            e.center.1 += n(&f["sensor_y"]) as f64;
            let rays = TheoreticalEllipseExplanations::from_ellipse(e, [4000.; 2], [4000., 3000.])
                .ok_or("conic")?;
            let mut result = evaluate(rays, &curves);
            result["visible_iris_compatibility"] = visibility_agreement(row, &curves);
            let provider = row["provider"].as_str().unwrap();
            count(&mut counts, format!("{provider}/rows"));
            for v in result["sensitivity"].as_array().unwrap() {
                if v["both_cover_every_box_corner"] == true {
                    count(
                        &mut counts,
                        format!(
                            "{provider}/both_cover/{}px",
                            v["coordinate_box_half_width_px"]
                        ),
                    );
                }
            }
            let raw = context_image(root, &target["context_frames"][1])?;
            overlay(
                row,
                &raw,
                rays,
                &curves,
                &result,
                &out.join(format!("visual-{record}-{provider}.png")),
            )?;
            results.push(json!({"record":record,"provider":provider,"raw_sha256":row["raw_sha256"],"source_ns":row["source_ns"],"frame":f,"area_admission":row["area_admission"],"estimate":estimate,"geometry":result}));
        }
    }
    fs::write(
        out.join("estimate-template.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"buttercup-assistant-visual-lid-estimates-v1","human_reviewed":false,"training_eligible":false,"physical_sign_truth":null,"estimates":template}),
        )?,
    )?;
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(&results)?,
    )?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"targets":targets.len(),"provider_rows":results.len(),"counts":counts,"scope":"Fixed review subset; provisional assistant estimates, not canonical human labels. No sign accuracy or corpus-wide recovery is asserted.","executable_sha256":archive::digest(&fs::read("/proc/self/exe")?),"review_manifest_sha256":archive::digest(&fs::read(root.join("manifest.json"))?),"estimates_sha256":if estimates=="-" {None} else {Some(archive::digest(&fs::read(estimates)?))},"retained_inputs_sha256":manifest["retained_inputs_sha256"]}),
        )?,
    )?;
    let mut report=String::from("# Provisional visual lid geometry check\n\nThese are assistant estimates from source-matched RAW before/target/after previews. They are sparse and unreviewed, and may confuse a fold with a margin. They are not canonical human labels, training material, measured globe centers or physical sign truth. The selected targets were fixed before the estimates.\n\n| Provider | Ambiguous rows in pilot | Both cover 0 px boxes | 2 px | 5 px | 10 px | 20 px |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    for provider in ["sam", "obelisk"] {
        let get = |key: String| counts.get(&key).copied().unwrap_or(0);
        report.push_str(&format!(
            "| {provider} | {} |",
            get(format!("{provider}/rows"))
        ));
        for error in [0., 2., 5., 10., 20.] {
            report.push_str(&format!(
                " {} |",
                get(format!("{provider}/both_cover/{error:.1}px"))
            ));
        }
        report.push('\n');
    }
    report.push_str("\nOne shared globe radius per candidate, continuously searched over 1.5–2.8 iris radii, must cover every point-box corner simultaneously. A sphere's perspective silhouette is convex, so it also covers the boxes and segments between them. The box widths are sensitivity assumptions, not calibrated annotation errors. Failure to cover an entire box does **not** reject a candidate: the actual point might lie in the part it covers.\n\nNo complete aperture or reliable canthus location is established. Comparisons with unreviewed model contour samples cannot identify which estimate is wrong when they conflict. Native planar-lid fits and their plane-intersection axes are diagnostics only. A near-central hinge is not assumed or validated. No sign is admitted.\n\nThe 3D panels use the native circle explanations and exact ray/sphere lifting; every drawn lifted point is checked to reproject within 0.000001 pixel. Camera intrinsics remain nominal and sphere-radius bounds are engineering assumptions. Input conics and their upstream area admission are unchanged; this is not an SN-FEIDA improvement.\n\n## Images\n\n");
    for target in targets {
        let id = n(&target["record"]);
        report.push_str(&format!("- Record {id}: [RAW triplet](raw-{id}.png)"));
        for r in results.iter().filter(|r| r["record"] == id) {
            let p = r["provider"].as_str().unwrap();
            report.push_str(&format!(", [{p} alternatives](visual-{id}-{p}.png)"));
        }
        report.push('\n');
    }
    report.push_str("\nThe retained RAW/fit cohort lacks independent physical sign truth, scale support and reviewed lid anatomy. These results do not demonstrate nearly-all-corpus disambiguation. The unresolved measurement is an independently accurate globe position/surface orientation or a validated anatomical relationship connecting visible lids to that position.\n");
    fs::write(out.join("README.md"), report)?;
    println!("{}", out.display());
    Ok(())
}
