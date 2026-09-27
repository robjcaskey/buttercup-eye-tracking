//! Lid aperture proposals and explicit center-prior sensitivity; not anatomy truth.
use super::*;
#[derive(Clone)]
struct Aperture {
    mask: Vec<bool>,
    upper: Vec<[f64; 2]>,
    lower: Vec<[f64; 2]>,
    canthi: [[f64; 2]; 2],
    canthi_mid: [f64; 2],
    aperture_mid: [f64; 2],
    cropped: bool,
}
fn mask_semantics(masks: &[u8]) -> Value {
    let plane = 384 * 256;
    assert_eq!(masks.len(), 6 * plane);
    let overlaps = [4, 5].map(|channel| {
        let mut intersection = 0;
        let mut union = 0;
        for i in 0..plane {
            let iris = masks[i] >= 128;
            let lid = masks[channel * plane + i] >= 128;
            intersection += usize::from(iris && lid);
            union += usize::from(iris || lid);
        }
        intersection as f64 / union.max(1) as f64
    });
    // These heads describe visible anatomy. A lid prediction duplicating most
    // of the iris cannot independently locate an aperture. Rejection is a
    // semantic consistency check, not proof that the remaining masks are right.
    let rejected = overlaps.iter().any(|&iou| iou > 0.5);
    json!({"upper_lower_iris_iou":overlaps,"maximum_allowed_iou":0.5,
        "passed":!rejected,"anatomical_accuracy_established":false,
        "reason":if rejected {Some("lid head substantially duplicates the iris; inferred aperture is unusable")} else {None}})
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn aperture(
    masks: &[u8],
    frame: &Value,
    e: Ellipse,
    eye_threshold: u8,
    lid_veto: u8,
) -> Option<Aperture> {
    let (w, h) = (384, 256);
    let plane = w * h;
    let mw = n(&frame["width"]) as f64;
    let mh = n(&frame["height"]) as f64;
    let origin = [n(&frame["sensor_x"]) as f64, n(&frame["sensor_y"]) as f64];
    let native = |x: usize, y: usize| {
        [
            origin[0] + (x as f64 + 0.5) * mw / w as f64 - 0.5,
            origin[1] + (y as f64 + 0.5) * mh / h as f64 - 0.5,
        ]
    };
    let mut possible = vec![false; plane];
    for i in 0..plane {
        possible[i] = masks[i].max(masks[3 * plane + i]) >= eye_threshold
            && masks[4 * plane + i] < lid_veto
            && masks[5 * plane + i] < lid_veto;
    }
    // Keep one connected aperture touching the independently fitted iris.
    let mut visited = vec![false; plane];
    let mut best = vec![];
    for seed in 0..plane {
        if !possible[seed] || visited[seed] {
            continue;
        }
        let mut q = vec![seed];
        visited[seed] = true;
        let mut j = 0;
        let mut inner = 0;
        while j < q.len() {
            let i = q[j];
            j += 1;
            let (x, y) = (i % w, i / w);
            let p = native(x, y);
            inner += usize::from(rho(e, [p[0] - origin[0], p[1] - origin[1]]) < 0.6);
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                if xx >= 0 && yy >= 0 && xx < w as i32 && yy < h as i32 {
                    let k = yy as usize * w + xx as usize;
                    if possible[k] && !visited[k] {
                        visited[k] = true;
                        q.push(k);
                    }
                }
            }
        }
        if inner >= 64 && q.len() > best.len() {
            best = q;
        }
    }
    if best.len() < 256 {
        return None;
    }
    let mut mask = vec![false; plane];
    for i in best {
        mask[i] = true;
    }
    let mut columns = vec![];
    for x in 0..w {
        let ys = (0..h).filter(|&y| mask[y * w + x]).collect::<Vec<_>>();
        if ys.len() >= 3 {
            columns.push((x, ys[0], ys[ys.len() - 1]));
        }
    }
    if columns.len() < 32 {
        return None;
    }
    let left = columns[0].0;
    let right = columns.last()?.0;
    let cropped = left < 4
        || right + 4 >= w
        || columns
            .iter()
            .any(|&(_, top, bottom)| top < 4 || bottom + 4 >= h);
    // Median endpoint strips reduce one-pixel tips. No interpolation across
    // unsupported columns; every displayed margin comes from a predicted mask.
    let edge = columns.len().div_ceil(20).max(3);
    let tip = |slice: &[(usize, usize, usize)]| {
        [
            median(slice.iter().map(|&(x, t, _)| native(x, t)[0]).collect()),
            median(
                slice
                    .iter()
                    .map(|&(x, t, b)| (native(x, t)[1] + native(x, b)[1]) * 0.5)
                    .collect(),
            ),
        ]
    };
    let canthi = [tip(&columns[..edge]), tip(&columns[columns.len() - edge..])];
    let mid = [
        (canthi[0][0] + canthi[1][0]) * 0.5,
        (canthi[0][1] + canthi[1][1]) * 0.5,
    ];
    let width = canthi[1][0] - canthi[0][0];
    let central = columns
        .iter()
        .filter(|&&(x, t, _)| (native(x, t)[0] - mid[0]).abs() < width * 0.1)
        .copied()
        .collect::<Vec<_>>();
    if central.len() < 4 {
        return None;
    }
    Some(Aperture {
        mask,
        upper: columns.iter().map(|&(x, t, _)| native(x, t)).collect(),
        lower: columns.iter().map(|&(x, _, b)| native(x, b)).collect(),
        canthi,
        canthi_mid: mid,
        aperture_mid: tip(&central),
        cropped,
    })
}
fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let u = [b[0] - a[0], b[1] - a[1]];
    let v = [p[0] - a[0], p[1] - a[1]];
    let t = ((u[0] * v[0] + u[1] * v[1]) / (u[0] * u[0] + u[1] * u[1]).max(1e-12)).clamp(0., 1.);
    (v[0] - t * u[0]).hypot(v[1] - t * u[1])
}
fn compatibility(anchor: [f64; 2], rays: TheoreticalEllipseExplanations, radii: [f64; 2]) -> Value {
    let distances = rays.rays.map(|p| {
        segment_distance(
            anchor,
            project(center(p, radii[0])),
            project(center(p, radii[1])),
        )
    });
    let sweep=[10.,20.,30.,40.,50.,60.,80.,100.].map(|epsilon| {
        let valid=distances.map(|d|d<=epsilon);let choices=valid.into_iter().filter(|v|*v).count();
        json!({"assumed_anchor_error_px":epsilon,"compatible":valid,"count":choices,"choice":if choices==1 {Some(usize::from(valid[1]))}else{None}})
    });
    json!({"anchor_sensor_px":anchor,"radius_range":radii,"minimum_distance_to_each_center_family_px":distances,"single_choice_error_interval_px":[distances[0].min(distances[1]),distances[0].max(distances[1])],"sensitivity":sweep,"condition":"An independent bound must justify the globe center being within epsilon of this lid-derived anchor. No such anatomical error bound has been measured here."})
}
fn render_lids(
    row: &Value,
    raw: &[u16],
    masks: &[u8],
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
    a: Option<&Aperture>,
    result: &Value,
    out: &Path,
) -> Result<()> {
    let f = &row["frame"];
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
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
    let mut c = Canvas::new(1800, 1170)?;
    c.clear();
    c.text(
        20.,
        34.,
        25.,
        WHITE,
        &format!(
            "Lid-boundary and center-prior audit | {} record {} | eye {} sequence {}",
            row["provider"].as_str().unwrap(),
            row["record"],
            row["eye"],
            row["sequence"]
        ),
    );
    c.text(20.,69.,17.,MUTED,"Existing Obelisk predictions, not human labels. The two hypotheses share the same observed aperture and unchanged iris ellipse.");
    for panel in 0..6 {
        let (px, py) = (panel % 3, panel / 3);
        let x = 20. + px as f64 * 594.;
        let y = 115. + py as f64 * 470.;
        let s = 570. / w as f64;
        c.text(
            x,
            y - 12.,
            20.,
            WHITE,
            [
                "Original RAW + proposed aperture",
                "Upper-lid probability",
                "Lower-lid probability",
                "Candidate globe centers + lid anchors",
                "Iris probability",
                "Sclera probability",
            ][panel],
        );
        if [0, 3].contains(&panel) {
            c.image(&bgra, w, h, x, y, 570., h as f64 * s);
        } else {
            let k = [0, 4, 5, 0, 0, 3][panel];
            let plane = 384 * 256;
            let p = &masks[k * plane..(k + 1) * plane];
            let display = p.iter().flat_map(|&v| [v, v, v, 255]).collect::<Vec<_>>();
            c.image(&display, 384, 256, x, y, 570., h as f64 * s);
        }
        c.clipped(x, y, 570., h as f64 * s, |c| {
            let xy = |p: [f64; 2]| [x + (p[0] - origin[0]) * s, y + (p[1] - origin[1]) * s];
            c.path(
                &e.dense_points(180)
                    .iter()
                    .map(|&(u, v)| [x + u * s, y + v * s])
                    .collect::<Vec<_>>(),
                1.3,
                WHITE,
            );
            if let Some(a) = a {
                if panel == 0 || panel == 3 {
                    c.path(
                        &a.upper.iter().copied().map(xy).collect::<Vec<_>>(),
                        2.,
                        CYAN,
                    );
                    c.path(
                        &a.lower.iter().copied().map(xy).collect::<Vec<_>>(),
                        2.,
                        PINK,
                    );
                }
                if panel == 3 {
                    c.line(xy(a.canthi[0]), xy(a.canthi[1]), 1.6, ORANGE);
                    for p in a.canthi {
                        let p = xy(p);
                        c.dot(p[0], p[1], 4., ORANGE, true);
                    }
                    for (p, col) in [(a.canthi_mid, ORANGE), (a.aperture_mid, GREEN)] {
                        let p = xy(p);
                        c.dot(p[0], p[1], 7., col, false);
                    }
                    for (i, pose) in rays.rays.iter().enumerate() {
                        let pts = [1.5, 2.8].map(|r| xy(project(center(*pose, r))));
                        let col = if i == 0 { CYAN } else { PINK };
                        c.line(pts[0], pts[1], 4., col);
                        let p = xy(project(center(*pose, 2.1)));
                        c.dot(p[0], p[1], 5., col, true);
                    }
                }
            }
        });
    }
    c.text(
        20.,
        1055.,
        18.,
        WHITE,
        &format!(
            "Lid/iris semantic check passed: {} | same conditional 40px choice: {}",
            result["mask_semantic_check"]["passed"], result["stable_40px_single_choice"]
        ),
    );
    c.text(20.,1091.,17.,MUTED,"Orange: inferred canthi and midpoint. Green: central aperture midpoint. Cyan/pink segments: A/B globe-center families, R/iris=1.5..2.8.");
    c.text(20.,1128.,17.,MUTED,"The anchor-to-globe error allowance is an unverified anatomical assumption. A selected branch is conditional evidence, not sign truth.");
    c.png(out)
}
pub(super) fn analyze(
    row: &Value,
    raw: &[u16],
    masks: &[u8],
    e: Ellipse,
    rays: TheoreticalEllipseExplanations,
    out: &Path,
    show: bool,
) -> Result<Value> {
    let semantic = mask_semantics(masks);
    let mut variants = vec![];
    let mut nominal = None;
    let mut choices = vec![];
    for (eye_threshold, lid_veto) in [(77, 179), (128, 128), (179, 77)] {
        let a = aperture(masks, &row["frame"], e, eye_threshold, lid_veto);
        let v = if let Some(a) = &a {
            let mut priors = vec![];
            for (kind, p) in [
                ("canthi_midpoint", a.canthi_mid),
                ("aperture_midpoint", a.aperture_mid),
            ] {
                for radii in [[1.8, 2.4], [1.5, 2.8]] {
                    let mut v = compatibility(p, rays, radii);
                    v["anchor_kind"] = json!(kind);
                    if !a.cropped {
                        choices.push(v["sensitivity"][3]["choice"].clone());
                    } else {
                        choices.push(Value::Null);
                    }
                    priors.push(v);
                }
            }
            json!({"available":true,"eye_threshold":eye_threshold,"lid_veto":lid_veto,"cropped":a.cropped,"aperture_mask_pixels":a.mask.iter().filter(|v|**v).count(),"upper_sensor_px":a.upper,"lower_sensor_px":a.lower,"canthi_sensor_px":a.canthi,"center_priors":priors})
        } else {
            choices.extend((0..4).map(|_| Value::Null));
            json!({"available":false,"eye_threshold":eye_threshold,"lid_veto":lid_veto})
        };
        if eye_threshold == 128 {
            nominal = a;
        }
        variants.push(v);
    }
    let usable = variants
        .iter()
        .all(|v| v["available"] == true && v["cropped"] == false);
    let stable = semantic["passed"] == true
        && choices.len() == 12
        && choices.iter().all(|v| v.is_number() && v == &choices[0]);
    let value = json!({"record":row["record"],"provider":row["provider"],"source":row["source"],"epoch":row["epoch"],"eye":row["eye"],"sequence":row["sequence"],"source_ns":row["source_ns"],"raw_sha256":row["raw_sha256"],"area_admission":row["area_admission"],"mask_semantic_check":semantic,"all_variants_have_uncropped_canthi":usable,"stable_40px_single_choice":stable,"conditional_choice":if stable {choices[0].clone()}else{Value::Null},"variants":variants,"physical_sign_truth":null});
    if show {
        render_lids(
            row,
            raw,
            masks,
            e,
            rays,
            nominal.as_ref(),
            &value,
            &out.join(format!(
                "lids-{}-{}.png",
                row["record"],
                row["provider"].as_str().unwrap()
            )),
        )?;
    }
    Ok(value)
}
