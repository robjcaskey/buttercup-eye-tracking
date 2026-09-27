//! Replay shared native lid measurements without interpreting them as labels.
//! The conic is a search seed only; no candidate sign selects any edge.
use super::*;
use buttercup_eye_tracking::raw_iris_focus::{
    self as native, BorderFocus, BorderPoint, OuterIrisBoundary,
};
use std::{sync::Arc, time::Instant};

const SEEDS: [[f64; 3]; 7] = [
    [0., 0., 1.],
    [-8., 0., 1.],
    [8., 0., 1.],
    [0., -8., 1.],
    [0., 8., 1.],
    [0., 0., 0.9],
    [0., 0., 1.1],
];

fn detect(row: &Value, raw: &[u16], kind: &str, seed: [f64; 3]) -> Result<(Vec<Curve>, Value)> {
    let frame = &row["frame"];
    let (w, h) = (n(&frame["width"]) as usize, n(&frame["height"]) as usize);
    let (sx, sy) = (n(&frame["sensor_x"]), n(&frame["sensor_y"]));
    let mut e = shape(&row["fit"]["ellipse"])?;
    e.center.0 += seed[0];
    e.center.1 += seed[1];
    e.major_radius *= seed[2];
    e.minor_radius *= seed[2];
    let start = Instant::now();
    let (margins, mut metadata) = if kind == "legacy" {
        let focus = BorderFocus {
            center: e.center,
            radius: e.major_radius,
            axis_ratio: e.minor_radius / e.major_radius,
            axis_angle: e.angle,
            ..Default::default()
        };
        (
            [
                native::detect_upper_eyelid_points(raw, w, h, sx as u32, sy as u32, &focus),
                native::detect_lower_eyelid_points(raw, w, h, sx as u32, sy as u32, &focus),
            ],
            json!({"detector":"native log-plane constrained arch"}),
        )
    } else {
        let outer = OuterIrisBoundary {
            center: e.center,
            major_radius: e.major_radius,
            minor_radius: e.minor_radius,
            angle: e.angle,
            ..Default::default()
        };
        let scene = native::discover_eyelid_scene_nautilus(raw, w, h, &outer, None);
        let metadata = json!({"detector":"native nautilus","status":[scene.upper_status.label(),scene.lower_status.label()],"clipped_occluder_counts":[scene.upper_clipped_occluder.len(),scene.lower_clipped_occluder.len()],"elapsed_us":scene.elapsed_us,"elapsed_at_or_above_live_budget":scene.elapsed_us>=8000,"policy":"Shared production deadline retained. Clipped occluders are not promoted to anatomical lids. No pupil hint is invented."});
        ([scene.upper_margin, scene.lower_margin], metadata)
    };
    metadata["wall_us"] = json!(start.elapsed().as_micros() as u64);
    metadata["seed_offset_scale"] = json!(seed);
    metadata["points"] = json!(margins.each_ref().map(|p| p
        .iter()
        .map(|p| [p.x as f64 + sx as f64, p.y as f64 + sy as f64, p.quality])
        .collect::<Vec<_>>()));
    let curves = margins
        .into_iter()
        .map(|points| Curve {
            pixels: points
                .into_iter()
                .map(|BorderPoint { x, y, .. }| [x as f64 + sx as f64, y as f64 + sy as f64])
                .collect(),
        })
        .collect();
    Ok((curves, metadata))
}

fn interpolate(points: &[[f64; 2]], x: f64) -> Option<f64> {
    let j = points.partition_point(|p| p[0] < x);
    if j == 0 {
        return points
            .first()
            .filter(|p| (p[0] - x).abs() < 1e-8)
            .map(|p| p[1]);
    }
    if j >= points.len() {
        return None;
    }
    let (a, b) = (points[j - 1], points[j]);
    if b[0] - a[0] > 16. || b[0] <= a[0] {
        return None;
    }
    Some(a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))
}
fn seed_disagreement(base: &[Curve], variant: &[Curve]) -> Value {
    let parts=[0,1].map(|lid| {
        let mut errors=vec![];
        for (a,b) in [(&base[lid],&variant[lid]),(&variant[lid],&base[lid])] {
            errors.extend(a.pixels.iter().filter_map(|p|interpolate(&b.pixels,p[0]).map(|y|(y-p[1]).abs())));
        }
        let total=base[lid].pixels.len()+variant[lid].pixels.len();
        if errors.len()<12 || total==0 {return json!({"stable":false,"reason":"insufficient common boundary support"});}
        errors.sort_by(f64::total_cmp);
        let p95=errors[((errors.len()-1) as f64*0.95).ceil() as usize];
        let maximum=*errors.last().unwrap();
        json!({"compared":errors.len(),"coverage":errors.len() as f64/total as f64,"p95_vertical_error_px":p95,"maximum_px":maximum,"stable":errors.len()*5>=total*4&&p95<=4.&&maximum<=8.})
    });
    json!({"lids":parts,"stable":parts.iter().all(|p|p["stable"]==true),"interpretation":"Sensitivity to search seed, not anatomical localization accuracy."})
}

fn evaluate(row: &Value, rays: TheoreticalEllipseExplanations, curves: &[Curve]) -> Value {
    let both = curves.iter().all(|c| c.pixels.len() >= 6);
    let observed = curves
        .iter()
        .filter(|c| !c.pixels.is_empty())
        .cloned()
        .collect::<Vec<_>>();
    let fits = if both {
        rays.rays.map(|p| profile(p, curves, [1.5, 2.8]))
    } else {
        [None, None]
    };
    let witnesses =
        (!observed.is_empty()).then(|| rays.rays.map(|p| silhouette_witness(p, &observed)));
    let sensitivity=[0.,2.,5.,10.,20.,40.].map(|error| {
        let Some(w)=witnesses.as_ref() else {return json!({"error_allowance_px":error,"outcome":"no_boundary"});};
        let rejected=w.each_ref().map(|w|w.outside_margin_lower_bound_px>error);
        let outcome=match rejected {[true,true]=>"both_rejected",[false,false]=>"ambiguous",_=>"conditional_single"};
        json!({"error_allowance_px":error,"outcome":outcome,"rejected":rejected,"conditional_preference":if outcome=="conditional_single" {Some(usize::from(rejected[0]))}else{None}})
    });
    json!({"point_counts":curves.iter().map(|c|c.pixels.len()).collect::<Vec<_>>(),"both_lids":both,"visible_iris_compatibility":if both {visibility_agreement(row,curves)} else {Value::Null},"fits":fits,"planarity_preference":choose(&fits),"silhouette_witnesses":witnesses,"containment_sensitivity":sensitivity,"admitted_sign":null})
}

fn render(
    row: &Value,
    raw: &[u16],
    rays: TheoreticalEllipseExplanations,
    panels: &[(String, Vec<Curve>, Value)],
    path: &Path,
) -> Result<()> {
    let f = &row["frame"];
    let (w, h) = (n(&f["width"]) as usize, n(&f["height"]) as usize);
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
    let rgba = preview::color_preview(raw, w, h, origin[0] as u32, origin[1] as u32, 100, None)
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
    let e = shape(&row["fit"]["ellipse"])?;
    let mut c = Canvas::new(1800, 1030)?;
    c.clear();
    c.text(
        22.,
        36.,
        25.,
        WHITE,
        &format!(
            "Native RAW lids | {} record {} | measurements are unverified proposals",
            row["provider"].as_str().unwrap(),
            row["record"]
        ),
    );
    c.text(22.,73.,18.,MUTED,"Dots: native edge samples. Lines: fitted iris / hypothetical globe. No synthetic lid points, completed margins, or sign labels.");
    for (i, (name, curves, result)) in panels.iter().enumerate() {
        let x = 22. + 884. * i as f64;
        let y = 132.;
        let s = 852. / w as f64;
        c.text(x, 110., 20., WHITE, name);
        c.image(&rgba, w, h, x, y, 852., h as f64 * s);
        c.clipped(x, y, 852., h as f64 * s, |c| {
            c.path(
                &e.dense_points(160)
                    .iter()
                    .map(|&(u, v)| [x + u * s, y + v * s])
                    .collect::<Vec<_>>(),
                1.,
                WHITE,
            );
            for (lid, curve) in curves.iter().enumerate() {
                let color = if lid == 0 { ORANGE } else { [0.3, 0.9, 1.] };
                for p in &curve.pixels {
                    c.dot(
                        x + (p[0] - origin[0]) * s,
                        y + (p[1] - origin[1]) * s,
                        2.5,
                        color,
                        false,
                    );
                }
            }
            for (j, ray) in rays.rays.iter().enumerate() {
                c.path(
                    &silhouette(center(*ray, 2.15), 2.15)
                        .iter()
                        .map(|p| [x + (p[0] - origin[0]) * s, y + (p[1] - origin[1]) * s])
                        .collect::<Vec<_>>(),
                    1.,
                    if j == 0 {
                        [0.6, 0.8, 0.4]
                    } else {
                        [0.85, 0.45, 0.8]
                    },
                );
            }
        });
        c.text(
            x,
            745.,
            19.,
            WHITE,
            &format!(
                "Upper / lower points: {}. Both observed: {}",
                result["point_counts"], result["both_lids"]
            ),
        );
        c.text(
            x,
            780.,
            18.,
            MUTED,
            &format!(
                "Recorded iris visibility compatible: {}",
                result["visible_iris_compatibility"]["compatible"]
            ),
        );
        c.text(
            x,
            815.,
            18.,
            MUTED,
            &format!(
                "10 px containment: {}",
                result["containment_sensitivity"][3]["outcome"]
            ),
        );
        c.text(
            x,
            850.,
            18.,
            MUTED,
            &format!("Planarity preference: {}", result["planarity_preference"]),
        );
    }
    c.text(22.,912.,18.,MUTED,"Both detectors use the same iris only to seed a bounded search. Seed-offset/radius checks are recorded separately.");
    c.text(22.,950.,18.,MUTED,"Contours and globe size are unreviewed assumptions; native edges may follow iris, lashes or skin. Neither detector establishes truth.");
    c.text(22.,988.,18.,MUTED,"Nautilus preserves its production 8 ms budget and keeps clipped occluders separate from anatomical margin proposals.");
    c.png(path)
}

pub(crate) fn run(area_dir: &str, fresh_dir: &str, output: &str) -> Result<()> {
    let (area, out) = (Path::new(area_dir), Path::new(output));
    if out.exists() {
        return Err("output exists".into());
    }
    let rows = admitted_inputs(area, fresh_dir)?;
    let originals = super::super::rows(&Path::new(fresh_dir).join("frames.jsonl"))?
        .into_iter()
        .map(|r| (n(&r["record"]), r))
        .collect::<BTreeMap<_, _>>();
    for row in &rows {
        for key in [
            "raw_sha256",
            "frame",
            "source_ns",
            "source",
            "epoch",
            "eye",
            "raw_source",
            "stream_entry",
        ] {
            if row[key] != originals[&n(&row["record"])][key] {
                return Err(format!("native lid source mismatch: {key}").into());
            }
        }
    }
    fs::create_dir(out)?;
    let start = Instant::now();
    let mut cache = BTreeMap::<String, Arc<Vec<u16>>>::new();
    let mut bundles = BTreeMap::new();
    let mut writer = BufWriter::new(fs::File::create(out.join("lids.jsonl"))?);
    let mut counts = BTreeMap::new();
    let mut reviews = vec![];
    for (index, row) in rows.iter().enumerate() {
        let hash = row["raw_sha256"].as_str().ok_or("RAW hash")?;
        if !cache.contains_key(hash) {
            cache.insert(
                hash.to_owned(),
                Arc::new(photometry::temporal::read_raw(row, &mut bundles)?),
            );
        }
        let raw = &cache[hash];
        let provider = row["provider"].as_str().unwrap();
        let f = &row["frame"];
        let mut e = shape(&row["fit"]["ellipse"])?;
        e.center.0 += n(&f["sensor_x"]) as f64;
        e.center.1 += n(&f["sensor_y"]) as f64;
        let rays = TheoreticalEllipseExplanations::from_ellipse(e, [4000.; 2], [4000., 3000.])
            .ok_or("native conic")?;
        let mut methods = vec![];
        let mut panels = vec![];
        for kind in ["legacy", "nautilus"] {
            let mut variants = vec![];
            let mut base: Option<Vec<Curve>> = None;
            for seed in SEEDS {
                let (curves, metadata) = detect(row, raw, kind, seed)?;
                let geometry = evaluate(row, rays, &curves);
                let stable = if let Some(base) = &base {
                    seed_disagreement(base, &curves)
                } else {
                    Value::Null
                };
                if base.is_none() {
                    panels.push((kind.to_owned(), curves.clone(), geometry.clone()));
                    base = Some(curves);
                }
                variants.push(json!({"metadata":metadata,"geometry":geometry,"difference_from_reference":stable}));
            }
            let seed_stable = variants
                .iter()
                .skip(1)
                .all(|v| v["difference_from_reference"]["stable"] == true);
            let reference = &variants[0]["geometry"];
            let p = &reference["containment_sensitivity"][3]["conditional_preference"];
            let stable_pref = (!p.is_null()
                && seed_stable
                && variants.iter().all(|v| {
                    v["geometry"]["visible_iris_compatibility"]["compatible"] == true
                        && v["geometry"]["containment_sensitivity"][3]["conditional_preference"]
                            == *p
                }))
            .then_some(p.clone());
            let prefix = format!("{provider}/{kind}");
            count(&mut counts, format!("{prefix}/rows"));
            for (name, pass) in [
                ("both_lids", reference["both_lids"] == true),
                (
                    "visibility_compatible",
                    reference["visible_iris_compatibility"]["compatible"] == true,
                ),
                ("seed_stable", seed_stable),
                ("stable_conditional_containment", stable_pref.is_some()),
                (
                    "conditional_planarity",
                    !reference["planarity_preference"].is_null(),
                ),
            ] {
                if pass {
                    count(&mut counts, format!("{prefix}/{name}"));
                }
            }
            count(
                &mut counts,
                format!(
                    "{prefix}/containment_10px/{}",
                    reference["containment_sensitivity"][3]["outcome"]
                        .as_str()
                        .unwrap()
                ),
            );
            methods.push(json!({"method":kind,"variants":variants,"seed_stable":seed_stable,"stable_conditional_containment_preference":stable_pref,"admitted_sign":null}));
        }
        let result = json!({"record":row["record"],"provider":provider,"source_ns":row["source_ns"],"raw_sha256":hash,"area_admission":row["area_admission"],"methods":methods,"measured_sign_truth":null});
        writeln!(writer, "{}", serde_json::to_string(&result)?)?;
        if index % 33 == 0
            || methods
                .iter()
                .any(|m| !m["stable_conditional_containment_preference"].is_null())
        {
            let name = format!("native-lids-{}-{provider}.png", row["record"]);
            render(row, raw, rays, &panels, &out.join(&name))?;
            reviews.push(json!({"image":name,"record":row["record"],"provider":provider}));
        }
        if (index + 1) % 50 == 0 {
            eprintln!(
                "NATIVE LIDS {} / {}, {:.1}s",
                index + 1,
                rows.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    writer.flush()?;
    fs::write(
        out.join("review.json"),
        serde_json::to_vec_pretty(&reviews)?,
    )?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"rows":rows.len(),"unique_raws":cache.len(),"seconds":start.elapsed().as_secs_f64(),"counts":counts,"seeds":SEEDS,"area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"retained_inputs_sha256":archive::digest(&fs::read(area.join("retained-inputs.jsonl"))?),"executable_sha256":archive::digest(&fs::read("/proc/self/exe")?),"method":"shared native RAW detectors and unchanged sphere/planarity geometry; no SAM mask input","physical_sign_accuracy":null,"human_lid_localization_error":null,"independent_scale":null,"policy":"Area gate before all RAW reads. Historical iris conics are unverified proposals; no training or live promotion. Search-seed stability is not anatomical accuracy. Production nautilus wall-time deadline may affect availability."}),
        )?,
    )?;
    Ok(())
}

/// Frozen reporting also rejects limbus-coincident margins as an independent
/// globe-center prior. This is not a declaration that a touching lid is false.
pub(crate) fn report(directory: &str, area_dir: &str) -> Result<()> {
    let (dir, area) = (Path::new(directory), Path::new(area_dir));
    let summary = load(&dir.join("summary.json"))?;
    if summary["complete"] != true
        || summary["retained_inputs_sha256"]
            != archive::digest(&fs::read(area.join("retained-inputs.jsonl"))?)
    {
        return Err("completed source-matched native lid run required".into());
    }
    if dir.join("evaluation.json").exists() || dir.join("README.md").exists() {
        return Err("native lid report exists".into());
    }
    let inputs = super::super::rows(&area.join("retained-inputs.jsonl"))?
        .into_iter()
        .map(|r| {
            (
                (n(&r["record"]), r["provider"].as_str().unwrap().to_owned()),
                r,
            )
        })
        .collect::<BTreeMap<_, _>>();
    let data = super::super::rows(&dir.join("lids.jsonl"))?;
    if data.len() as u64 != n(&summary["rows"]) {
        return Err("native lid row count mismatch".into());
    }
    let mut diagnostics = BufWriter::new(fs::File::create(dir.join("independence.jsonl"))?);
    let mut counts = BTreeMap::<String, usize>::new();
    let mut timings = BTreeMap::<String, Vec<u64>>::new();
    let mut seen = BTreeSet::new();
    for row in &data {
        let key = (
            n(&row["record"]),
            row["provider"].as_str().ok_or("provider")?.to_owned(),
        );
        if !seen.insert(key.clone()) {
            return Err("duplicate native lid row".into());
        }
        let original = inputs.get(&key).ok_or("native lid input missing")?;
        for name in ["raw_sha256", "area_admission", "source_ns"] {
            if row[name] != original[name] {
                return Err(format!("native lid receipt differs: {name}").into());
            }
        }
        let e = shape(&original["fit"]["ellipse"])?;
        let f = &original["frame"];
        let points = e
            .dense_points(2048)
            .into_iter()
            .map(|(x, y)| [x + n(&f["sensor_x"]) as f64, y + n(&f["sensor_y"]) as f64])
            .collect::<Vec<_>>();
        for method in row["methods"].as_array().ok_or("native methods")? {
            let name = method["method"].as_str().unwrap();
            let prefix = format!("{}/{name}", key.1);
            let reference = &method["variants"][0];
            let geometry = &reference["geometry"];
            let margins = reference["metadata"]["points"]
                .as_array()
                .ok_or("native margins")?;
            let contact=margins.iter().map(|p| {
                let p=p.as_array().unwrap();
                let close=p.iter().filter(|p| {
                    let xy=[p[0].as_f64().unwrap(),p[1].as_f64().unwrap()];
                    points.iter().any(|q|(q[0]-xy[0]).hypot(q[1]-xy[1])<=4.)
                }).count();
                json!({"points":p.len(),"within_4px_of_fitted_limbus":close,"limbus_coincident":p.len()>=6&&close*4>=p.len()*3})
            }).collect::<Vec<_>>();
            let aliases = contact
                .iter()
                .filter(|p| p["limbus_coincident"] == true)
                .count();
            let independent = geometry["both_lids"] == true
                && geometry["visible_iris_compatibility"]["compatible"] == true
                && method["seed_stable"] == true
                && aliases == 0;
            count(&mut counts, format!("{prefix}/rows"));
            count(
                &mut counts,
                format!(
                    "{prefix}/containment_10px/{}",
                    geometry["containment_sensitivity"][3]["outcome"]
                        .as_str()
                        .ok_or("containment outcome")?
                ),
            );
            for (label, pass) in [
                ("both_lids", geometry["both_lids"] == true),
                (
                    "visibility_compatible",
                    geometry["visible_iris_compatibility"]["compatible"] == true,
                ),
                ("seed_stable", method["seed_stable"] == true),
                ("limbus_coincident_margin", aliases > 0),
                (
                    "conditional_planarity",
                    !geometry["planarity_preference"].is_null(),
                ),
                ("passes_independence_diagnostics", independent),
                (
                    "stable_conditional_choice",
                    !method["stable_conditional_containment_preference"].is_null(),
                ),
            ] {
                if pass {
                    count(&mut counts, format!("{prefix}/{label}"));
                }
            }
            for variant in method["variants"].as_array().unwrap() {
                timings
                    .entry(name.to_owned())
                    .or_default()
                    .push(n(&variant["metadata"]["wall_us"]));
                if variant["metadata"]["elapsed_at_or_above_live_budget"] == true {
                    count(&mut counts, format!("{prefix}/at_live_budget"));
                }
            }
            writeln!(
                diagnostics,
                "{}",
                json!({"record":key.0,"provider":key.1,"method":name,"contact":contact,"passes_independence_diagnostics":independent,"anatomical_correctness_established":false,"policy":"Limbus coincidence with >=75% of at least six points within 4 px is not independent center evidence. It may be a touching lid, a duplicate limbus, or an erroneous conic; no anatomical truth is inferred."})
            )?;
        }
    }
    diagnostics.flush()?;
    let mut text=format!("# Native RAW lid measurements and independent sign evidence\n\nThis replay uses {} area-admitted ambiguous provider rows on {} unique RAW exposures. SAM and Obelisk supply the iris conics; the lid detectors are the same shared classical native algorithms for both. No learned anatomy mask is an input, and no live detector is modified. All source, time, native RAW hashes and area admissions are checked.\n\n| Conic provider / detector | Rows | Both margins returned | Compatible with recorded iris visibility | Stable across six seed perturbations | Limbus-coincident margin | Pass independence diagnostics |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",data.len(),summary["unique_raws"]);
    for provider in ["sam", "obelisk"] {
        for kind in ["legacy", "nautilus"] {
            let c = |field| {
                counts
                    .get(&format!("{provider}/{kind}/{field}"))
                    .copied()
                    .unwrap_or(0)
            };
            text += &format!(
                "| {provider} / {kind} | {} | {} | {} | {} | {} | {} |\n",
                c("rows"),
                c("both_lids"),
                c("visibility_compatible"),
                c("seed_stable"),
                c("limbus_coincident_margin"),
                c("passes_independence_diagnostics")
            );
        }
    }
    text+="\nThe reference conic only seeds each search. Replays independently shift it 8 pixels left/right/up/down or change both radii by ±10%; the candidate 3D conics remain unchanged. Stability requires both margins, at least 80% common sampled support, a 95th-percentile vertical disagreement <=4 px and maximum <=8 px for every perturbation. This measures dependence on the seed, not true localization accuracy.\n\nA margin with at least six points and >=75% within 4 pixels of the fitted limbus is rejected as independent center evidence. Distance is evaluated against 2,048 samples of the fitted curve. A real lid can touch the limbus: this diagnostic flags dependence or coincidence, not a proven anatomical mistake. Likewise the iris visibility comparison uses unreviewed predicted points, so conflicts do not by themselves establish which detector is wrong.\n\nThe receipt records containment outcomes, planarity preferences and independence checks separately. A conditional geometric preference does not establish physical sign recovery or anatomical accuracy. The nautilus production 8 ms deadline remains active, with clipped occluders kept separate; timing receipts identify whether it was reached.\n\nReview the RAW overlays before interpreting a returned path as anatomy: internal iris texture and the limbus can also yield coherent curves. These checks do not prove that true eyelid positions are uninformative. Masks, conics and the sphere model remain uncertain; independent scale and physical sign labels are missing. No model is trained or promoted, and no SN-FEIDA improvement is claimed.\n";
    let timing = timings
        .into_iter()
        .map(|(name, mut v)| {
            v.sort_unstable();
            (
                name,
                json!({"calls":v.len(),"median_us":v[v.len()/2],"maximum_us":v.last()}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let hashes = ["summary.json", "lids.jsonl", "independence.jsonl"]
        .into_iter()
        .map(|file| Ok((file, archive::digest(&fs::read(dir.join(file))?))))
        .collect::<Result<BTreeMap<_, _>>>()?;
    fs::write(
        dir.join("evaluation.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"counts":counts,"timings":timing,"input_hashes":hashes,"human_lid_localization_error":null,"physical_sign_accuracy":null}),
        )?,
    )?;
    fs::write(dir.join("README.md"), text)?;
    Ok(())
}
