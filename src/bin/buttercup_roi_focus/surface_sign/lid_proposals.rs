//! Conditional geometry of fixed-rank SAM opening proposals, never sign truth.
use super::*;

fn boundaries(mask: &[u8], threshold: u8, row: &Value) -> Option<(Vec<Curve>, Value)> {
    let (w, h) = (420usize, 280usize);
    if mask.len() != w * h {
        return None;
    }
    let e = shape(&row["fit"]["ellipse"]).ok()?;
    let mut seen = vec![false; mask.len()];
    let mut best = vec![];
    for seed in 0..mask.len() {
        if seen[seed] || mask[seed] < threshold {
            continue;
        }
        let mut component = vec![seed];
        let mut inner = 0;
        let mut cursor = 0;
        seen[seed] = true;
        while cursor < component.len() {
            let i = component[cursor];
            cursor += 1;
            let (x, y) = (i % w, i / w);
            inner += usize::from(rho(e, [x as f64, y as f64]) < 0.6);
            for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if !seen[j] && mask[j] >= threshold {
                    seen[j] = true;
                    component.push(j);
                }
            }
        }
        if inner >= 64 && component.len() > best.len() {
            best = component;
        }
    }
    if best.len() < 256 {
        return None;
    }
    let area = best.len();
    let mut top = vec![h; w];
    let mut bottom = vec![0; w];
    let mut cropped = false;
    for i in best {
        let (x, y) = (i % w, i / w);
        top[x] = top[x].min(y);
        bottom[x] = bottom[x].max(y);
        cropped |= x < 4 || y < 4 || x + 4 >= w || y + 4 >= h;
    }
    // Crop edges are censored, not observed lid boundaries. Keep the longest
    // uninterrupted run with two actual boundaries; never bridge missing x.
    let mut longest = vec![];
    let mut current = vec![];
    for x in 0..w {
        if x >= 4 && x + 4 < w && top[x] >= 4 && bottom[x] + 4 < h && top[x] + 3 <= bottom[x] {
            current.push(x);
        } else {
            if current.len() > longest.len() {
                longest = current.clone();
            }
            current.clear();
        }
    }
    if current.len() > longest.len() {
        longest = current;
    }
    if longest.len() < 40 {
        return None;
    }
    let f = &row["frame"];
    let origin = [n(&f["sensor_x"]) as f64, n(&f["sensor_y"]) as f64];
    let curves = [&top, &bottom]
        .map(|margin| Curve {
            pixels: longest
                .iter()
                .map(|&x| [origin[0] + x as f64, origin[1] + margin[x] as f64])
                .collect(),
        })
        .to_vec();
    let details = json!({"component_area_px":area,"component_touches_crop":cropped,
        "uncensored_contiguous_columns":longest.len(),"mask_threshold":threshold,
        "selection":"largest 4-connected component with at least 64 pixels inside rho<0.6 of the fixed fitted iris; no sign selects mask/component",
        "semantics_verified":false});
    Some((curves, details))
}

pub(crate) fn run(area_dir: &str, fresh_dir: &str, anatomy_dir: &str, output: &str) -> Result<()> {
    let (area, anatomy, out) = (
        Path::new(area_dir),
        Path::new(anatomy_dir),
        Path::new(output),
    );
    if out.exists() {
        return Err("output exists".into());
    }
    let input = admitted_inputs(area, fresh_dir)?;
    let summary = load(&anatomy.join("summary.json"))?;
    if summary["complete"] != true
        || summary["device"] != "cpu"
        || summary["teacher"]["schema"] != "buttercup-sam31-cold-export-v1"
    {
        return Err("completed CPU SAM anatomy run required".into());
    }
    let proposals = rows(&anatomy.join("frames.jsonl"))?
        .into_iter()
        .map(|r| (n(&r["record"]), r))
        .collect::<BTreeMap<_, _>>();
    fs::create_dir(out)?;
    let mut writer = BufWriter::new(fs::File::create(out.join("proposals.jsonl"))?);
    let mut counts = BTreeMap::new();
    let mut bundles = BTreeMap::new();
    let mut evaluated = 0;
    for row in &input {
        let id = n(&row["record"]);
        let provider = row["provider"].as_str().ok_or("provider")?;
        let Some(observation) = proposals.get(&id) else {
            continue;
        };
        for key in [
            "raw_sha256",
            "frame",
            "source",
            "epoch",
            "eye",
            "sequence",
            "source_ns",
        ] {
            if row[key] != observation[key] {
                return Err(format!("SAM/iris source mismatch {id}: {key}").into());
            }
        }
        if !observation["area_admitted_providers"]
            .as_array()
            .ok_or("admitted providers")?
            .iter()
            .any(|p| p == provider)
        {
            return Err("provider was not area-admitted before SAM inference".into());
        }
        let frame = &row["frame"];
        if n(&frame["width"]) != 420 || n(&frame["height"]) != 280 {
            return Err("native 420x280 required".into());
        }
        evaluated += 1;
        let detail = observation["prompts"]
            .as_array()
            .ok_or("prompts")?
            .iter()
            .find(|p| p["prompt"] == "open eye")
            .ok_or("fixed open eye prompt missing")?;
        let candidate = &detail["candidates"][0];
        let mask = if let Some(path) = candidate["mask"].as_str() {
            let bytes = fs::read(anatomy.join(path))?;
            if bytes.len() != 420 * 280 || archive::digest(&bytes) != candidate["mask_sha256"] {
                return Err("mask hash/shape mismatch".into());
            }
            Some(bytes)
        } else {
            None
        };
        let mut e = shape(&row["fit"]["ellipse"])?;
        e.center.0 += n(&frame["sensor_x"]) as f64;
        e.center.1 += n(&frame["sensor_y"]) as f64;
        let rays = TheoreticalEllipseExplanations::from_ellipse(e, [4000.; 2], [4000., 3000.])
            .ok_or("native circle solve")?;
        for threshold in [128, 153] {
            let Some((curves, selection)) =
                mask.as_ref().and_then(|m| boundaries(m, threshold, row))
            else {
                count(
                    &mut counts,
                    format!("{provider}/{threshold}/insufficient_boundary_support"),
                );
                writeln!(
                    writer,
                    "{}",
                    json!({"record":id,"provider":provider,"threshold":threshold,"outcome":"insufficient_boundary_support","admitted_sign":null})
                )?;
                continue;
            };
            let fits = rays.rays.map(|p| profile(p, &curves, [1.5, 2.8]));
            let witnesses = rays.rays.map(|p| silhouette_witness(p, &curves));
            let visibility = visibility_agreement(row, &curves);
            let planarity_choice = choose(&fits);
            count(
                &mut counts,
                format!(
                    "{provider}/{threshold}/planarity_{}",
                    if planarity_choice.is_some() {
                        "conditional_choice"
                    } else {
                        "inconclusive"
                    }
                ),
            );
            let sensitivity = [0., 2., 5., 10., 20., 40.].map(|error| {
                let rejected = witnesses
                    .each_ref()
                    .map(|w| w.outside_margin_lower_bound_px > error);
                let outcome = match rejected {
                    [true, true] => "both_rejected",
                    [false, false] => "ambiguous",
                    _ => "conditional_single",
                };
                count(
                    &mut counts,
                    format!("{provider}/{threshold}/silhouette_{error}px/{outcome}"),
                );
                json!({"error_allowance_px":error,"rejected":rejected,"outcome":outcome})
            });
            let result = json!({"record":id,"provider":provider,"raw_sha256":row["raw_sha256"],"area_admission":row["area_admission"],
                "selection":selection,"sam_candidate":candidate,"visible_iris_compatibility":visibility,"fits":fits,
                "planarity_choice_if_assumptions_hold":planarity_choice,"silhouette_witnesses":witnesses,"containment_sensitivity":sensitivity,
                "admitted_sign":null,"reason":"Unverified opening mask and unmeasured anatomical/camera error; no physical sign truth"});
            writeln!(writer, "{}", serde_json::to_string(&result)?)?;
            if threshold == 128 {
                let source = row["raw_source"].as_str().ok_or("RAW source")?;
                if !bundles.contains_key(source) {
                    bundles.insert(source.to_owned(), BundleSource::open(Path::new(source))?);
                }
                let bytes = bundles[source].read_range(
                    row["stream_entry"].as_str().ok_or("stream entry")?,
                    n(&frame["offset"]),
                    n(&frame["length"]) as usize,
                )?;
                if archive::digest(&bytes) != row["raw_sha256"] {
                    return Err("RAW changed".into());
                }
                let raw = raw10::try_unpack_raw10(&bytes, 420, 280, n(&frame["stride"]) as usize)?;
                render(
                    row,
                    &raw,
                    rays,
                    &curves,
                    None,
                    &fits,
                    &out.join(format!("proposal-{id}-{provider}.png")),
                )?;
            }
        }
    }
    writer.flush()?;
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"counts":counts,"evaluated_ambiguous_provider_rows":evaluated,
        "all_ambiguous_provider_rows":input.len(),"anatomy_summary":summary,"anatomy_frames_sha256":archive::digest(&fs::read(anatomy.join("frames.jsonl"))?),
        "area_summary_sha256":archive::digest(&fs::read(area.join("summary.json"))?),"executable_sha256":archive::digest(&fs::read(std::env::current_exe()?)?),
        "semantics_verified":false,"real_admitted_sign_choices":0,"physical_sign_accuracy":null}),
        )?,
    )?;
    Ok(())
}
