//! Reproducible reporting of frozen surface runs; no new inference or training.
use super::*;
fn stats(mut v: Vec<f64>) -> Value {
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return Value::Null;
    }
    json!({"count":v.len(),"min":v[0],"median":v[v.len()/2],"max":v[v.len()-1]})
}

pub fn sclera(dir: &str) -> Result<()> {
    let dir = Path::new(dir);
    let summary = load(&dir.join("summary.json"))?;
    if summary["complete"] != true {
        return Err("completed sclera motion run required".into());
    }
    if summary["feature_selection"] == "rigid" {
        return Err(
            "use surface-rigid-report BASELINE_RUN RIGID_RUN for a matched rigid comparison".into(),
        );
    }
    if dir.join("evaluation.json").exists() || dir.join("README.md").exists() {
        return Err("sclera report exists".into());
    }
    let data = rows(&dir.join("tracks.jsonl"))?;
    if data.len() as u64 != n(&summary["rows"])
        || data.iter().any(|r| r["area_admission"]["accepted"] != true)
    {
        return Err("sclera row/admission mismatch".into());
    }
    let indexed = data
        .iter()
        .map(|r| {
            (
                (r["provider"].as_str().unwrap().to_owned(), n(&r["record"])),
                r,
            )
        })
        .collect::<BTreeMap<_, _>>();
    if indexed.len() != data.len() {
        return Err("duplicate sclera records".into());
    }
    let mut real = BTreeMap::new();
    let mut report=String::from("# Scleral texture and rigid-surface sign evidence\n\nFor a material point on a rigid globe, its surface normal N and the iris normal n undergo the same rotation Q. Their dot product, (QN)·(Qn)=N·n, is invariant. This is a necessary constraint on competing iris interpretations. It depends on correct point identity, spherical geometry and conic estimates; matching a bright patch does not establish that it is a fixed scleral feature.\n\nThe current diagnostic uses area-admitted native RAW only. Each full matching patch stays inside both frames' proposed sclera and outside the fitted iris. There are no measured sign labels or independent scale measurements.\n\n| Iris-conic provider | Ambiguous targets | Targets with fresh admitted predecessor | At least one strict track | At least six strict tracks | Real sign selections |\n| --- | ---: | ---: | ---: | ---: | ---: |\n");
    for provider in ["sam", "obelisk"] {
        let cohort = data
            .iter()
            .filter(|r| r["provider"] == provider && r["class"] == "multiple")
            .collect::<Vec<_>>();
        let fresh = cohort
            .iter()
            .filter(|r| r["fresh_pair"] == true)
            .collect::<Vec<_>>();
        let one = fresh
            .iter()
            .filter(|r| n(&r["variants"][0]["strict_matches"]) > 0)
            .count();
        let six = fresh
            .iter()
            .filter(|r| n(&r["variants"][0]["strict_matches"]) >= 6)
            .count();
        report += &format!(
            "| {provider} | {} | {} | {one} | {six} | 0 |\n",
            cohort.len(),
            fresh.len()
        );
        real.insert(provider,json!({"ambiguous_targets":cohort.len(),"fresh_predecessor_targets":fresh.len(),"at_least_one_strict_track":one,"at_least_six_strict_tracks":six,"real_sign_selections":0,"interior_match_counts":stats(fresh.iter().map(|r|n(&r["variants"][0]["interior_matches"]) as f64).collect()),"strict_match_counts":stats(fresh.iter().map(|r|n(&r["variants"][0]["strict_matches"]) as f64).collect())}));
    }
    report+="\nSix points is the synthetic study's support requirement, not a proof that fewer can never help. Strict patch scores require native score >=0.75 and distinct-match margin >=0.10. These are heuristic filters, not calibrated correspondence error bounds. Sclera masks are historical unverified custom predictions used only as diagnostic proposals.\n\nThe oracle generates true rigid-globe correspondences on recorded conic/mask layouts, uses radius 2.15 and a 0.05-radian torsion, and profiles both candidate families over the shared radius grid. The image perturbations below are deterministic bounded per-axis errors, not measured RAW tracking errors. Ranking is forced among available families; it is not sign admission or real accuracy.\n\n| Provider / perturbation | Synthetic cases | Usable | Correct pair ranking | Correct target ranking | Unavailable |\n| --- | ---: | ---: | ---: | ---: | ---: |\n";
    let mut controls = BTreeMap::<String, Vec<Value>>::new();
    for row in rows(&dir.join("polar-controls.jsonl"))? {
        let key = (
            row["provider"].as_str().unwrap().to_owned(),
            n(&row["target_record"]),
        );
        let target = indexed
            .get(&key)
            .ok_or("oracle target missing from real cohort")?;
        if row["synthetic_only"] != true
            || target["fresh_pair"] != true
            || target["previous_record"] != row["source_record"]
        {
            return Err("oracle source correspondence mismatch".into());
        }
        for c in row["cases"].as_array().ok_or("oracle cases")? {
            let mut c = c.clone();
            c["ambiguous_target"] = json!(target["class"] == "multiple");
            controls
                .entry(format!("{}/{}px", key.0, c["noise_bound_per_axis_px"]))
                .or_default()
                .push(c);
        }
    }
    let mut oracle_metrics = BTreeMap::new();
    for (name, cases) in controls {
        let usable = cases
            .iter()
            .filter(|c| c["usable"] == true)
            .collect::<Vec<_>>();
        let correct = usable
            .iter()
            .filter(|c| c["lowest_error_pair_correct"] == true)
            .count();
        let target = usable
            .iter()
            .filter(|c| c["lowest_error_target_correct"] == true)
            .count();
        let mut reasons = BTreeMap::<String, usize>::new();
        for c in cases.iter().filter(|c| c["usable"] != true) {
            *reasons
                .entry(c["reason"].as_str().unwrap().to_owned())
                .or_default() += 1;
        }
        let ambiguous = cases
            .iter()
            .filter(|c| c["ambiguous_target"] == true)
            .collect::<Vec<_>>();
        report += &format!(
            "| {name} | {} | {} | {correct} | {target} | {} |\n",
            cases.len(),
            usable.len(),
            cases.len() - usable.len()
        );
        oracle_metrics.insert(name,json!({"cases":cases.len(),"usable":usable.len(),"pair_rank_correct":correct,"target_rank_correct":target,"unavailable_reasons":reasons,"runner_up_gap_degrees":stats(usable.iter().filter_map(|c|c["runner_up_gap_degrees"].as_f64()).collect()),"true_nominal_maximum_drift_degrees":stats(usable.iter().filter_map(|c|c["true_nominal_maximum_drift_degrees"].as_f64()).collect()),"ambiguous_target_subset":{"cases":ambiguous.len(),"usable":ambiguous.iter().filter(|c|c["usable"]==true).count(),"pair_rank_correct":ambiguous.iter().filter(|c|c["lowest_error_pair_correct"]==true).count(),"target_rank_correct":ambiguous.iter().filter(|c|c["lowest_error_target_correct"]==true).count()}}));
    }
    report+="\nThe noise-free result is a positive conditional geometry experiment. It does not show that the real image features can be measured to that precision, that their anatomical identity is correct, or that the true globe matches the assumed sphere. No conic, area value, model or live setting is changed by this diagnostic. The upstream area gate lacks independent scale and therefore does not establish SN-FEIDA.\n";
    let mut hashes = BTreeMap::new();
    for file in ["summary.json", "tracks.jsonl", "polar-controls.jsonl"] {
        hashes.insert(file, archive::digest(&fs::read(dir.join(file))?));
    }
    fs::write(
        dir.join("evaluation.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"real":real,"synthetic":oracle_metrics,"input_hashes":hashes,"measured_sign_accuracy":null,"reviewed_sclera_localization_error":null}),
        )?,
    )?;
    fs::write(dir.join("README.md"), report)?;
    Ok(())
}

pub fn sclera_rigid(baseline: &str, candidate: &str) -> Result<()> {
    let base = Path::new(baseline);
    let dir = Path::new(candidate);
    if dir.join("evaluation.json").exists() || dir.join("README.md").exists() {
        return Err("rigid report exists".into());
    }
    let summary = load(&dir.join("summary.json"))?;
    let before = load(&base.join("summary.json"))?;
    if summary["complete"] != true
        || before["complete"] != true
        || summary["feature_selection"] != "rigid"
        || before["feature_selection"] != "sclera"
        || summary["retained_inputs_sha256"] != before["retained_inputs_sha256"]
        || summary["area_summary_sha256"] != before["area_summary_sha256"]
    {
        return Err("completed source-matched sclera and rigid runs required".into());
    }
    let data = rows(&dir.join("tracks.jsonl"))?;
    let old = rows(&base.join("tracks.jsonl"))?;
    if data.len() != old.len() || data.len() as u64 != n(&summary["rows"]) {
        return Err("matched track row counts differ".into());
    }
    for (row, old) in data.iter().zip(&old) {
        let mut stripped = row.clone();
        stripped
            .as_object_mut()
            .ok_or("track object")?
            .remove("stable_conditional_pair_preference");
        for variant in stripped["variants"].as_array_mut().ok_or("variants")? {
            variant.as_object_mut().ok_or("variant")?.remove("rigid");
        }
        if stripped != *old || row["area_admission"]["accepted"] != true {
            return Err(
                "rigid run changed matched tracks, area admission, sources or prior polar results"
                    .into(),
            );
        }
    }
    let mut real = BTreeMap::new();
    let mut report = String::from("# Held-out rigid scleral-motion diagnostic\n\nAll native correspondences, RAW identities, source times, area admissions and prior polar-invariant results exactly match the sclera-seeded baseline. Parameters are fitted without the held patch or any overlapping patch in either exposure. Orange review points are native matches; blue crosses are full-data predictions for display only. No physical sign labels or reviewed sclera labels are available.\n\nA common sphere radius and proper rotation link the source and target iris hypotheses. Radius is profiled from 1.5 to 2.8 iris radii; torsion minimizes surface-normal chord error, and radius/source family are chosen by training image error. Scoring uses held-out image error. Translation and 2D similarity use identical folds. This is a finite-grid conditional model, not a physical sign certificate.\n\n| Provider | Ambiguous targets | Fresh predecessor | Usable reference setting | Conditional reference choices | Stable across both masks and guards |\n| --- | ---: | ---: | ---: | ---: | ---: |\n");
    for provider in ["sam", "obelisk"] {
        let cohort = data
            .iter()
            .filter(|r| r["provider"] == provider && r["class"] == "multiple")
            .collect::<Vec<_>>();
        let fresh = cohort
            .iter()
            .filter(|r| r["fresh_pair"] == true)
            .collect::<Vec<_>>();
        let stable = cohort
            .iter()
            .filter(|r| !r["stable_conditional_pair_preference"].is_null())
            .count();
        let mut variants = vec![];
        for mask in 0..2 {
            for guard in 0..2 {
                let results = fresh
                    .iter()
                    .map(|r| &r["variants"][mask]["rigid"][guard])
                    .collect::<Vec<_>>();
                let usable = results
                    .iter()
                    .copied()
                    .filter(|r| r["usable"] == true)
                    .collect::<Vec<_>>();
                let mut reasons = BTreeMap::new();
                for result in results.iter().filter(|r| r["usable"] != true) {
                    count(
                        &mut reasons,
                        result["reason"]
                            .as_str()
                            .ok_or("unavailable reason")?
                            .to_owned(),
                    );
                }
                let best = |r: &&Value| {
                    r["candidate_cv_rmse_px"][0]
                        .as_f64()
                        .unwrap()
                        .min(r["candidate_cv_rmse_px"][1].as_f64().unwrap())
                };
                variants.push(json!({"mask_threshold":([179,230][mask]),"guard_px":([16,24][guard]),"usable":usable.len(),"unavailable_reasons":reasons,"conditional_choices":usable.iter().filter(|r|!r["conditional_pair_preference"].is_null()).count(),"best_candidate_cv_rmse_px":stats(usable.iter().map(best).collect()),"translation_cv_rmse_px":stats(usable.iter().map(|r|r["baseline_cv_rmse_px"]["translation"].as_f64().unwrap()).collect()),"similarity_cv_rmse_px":stats(usable.iter().map(|r|r["baseline_cv_rmse_px"]["similarity"].as_f64().unwrap()).collect()),"best_candidate_beats_translation":usable.iter().filter(|r| best(r)<r["baseline_cv_rmse_px"]["translation"].as_f64().unwrap()).count(),"best_candidate_beats_similarity":usable.iter().filter(|r| best(r)<r["baseline_cv_rmse_px"]["similarity"].as_f64().unwrap()).count()}));
            }
        }
        report += &format!(
            "| {provider} | {} | {} | {} | {} | {stable} |\n",
            cohort.len(),
            fresh.len(),
            variants[0]["usable"],
            variants[0]["conditional_choices"]
        );
        real.insert(provider,json!({"ambiguous_targets":cohort.len(),"fresh_predecessor_targets":fresh.len(),"stable_conditional_choices":stable,"variants":variants}));
    }
    report += "\nReference: mask threshold 179, 16-pixel patch separation. Sensitivity checks use threshold 230 and 24-pixel separation. At least six held points and 80% coverage are required. Both candidate families must predict every covered point; unsupported geometry is an abstention. A conditional choice requires winner RMSE <=2 px, loser >=1 px and >=50% worse, winner within 0.5 px of the better baseline, and >=80% agreement among training-fold source/target choices. These fixed heuristic margins are not calibrated probabilities.\n\n| Provider / synthetic coordinate perturbation | Cases | Usable reference | Correct target ranking | Correct conditional pair | Wrong conditional pair | Stable conditional pair: correct / wrong |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    let controls = rows(&dir.join("polar-controls.jsonl"))?;
    let old_controls = rows(&base.join("polar-controls.jsonl"))?;
    if controls.len() != old_controls.len() {
        return Err("matched control counts differ".into());
    }
    let mut synthetic = BTreeMap::<String, Vec<Value>>::new();
    for (row, old) in controls.iter().zip(&old_controls) {
        let mut stripped = row.clone();
        for case in stripped["cases"].as_array_mut().ok_or("synthetic cases")? {
            case.as_object_mut().ok_or("case")?.remove("rigid");
        }
        if stripped != *old {
            return Err("prior synthetic control result changed".into());
        }
        for c in row["cases"].as_array().unwrap() {
            synthetic
                .entry(format!(
                    "{}/{}px",
                    row["provider"].as_str().unwrap(),
                    c["noise_bound_per_axis_px"]
                ))
                .or_default()
                .push(c.clone());
        }
    }
    let mut synthetic_metrics = BTreeMap::new();
    for (name, cases) in synthetic {
        let mut variants = vec![];
        for guard in 0..2 {
            let mut counts = [
                "usable",
                "target_rank_correct",
                "target_rank_wrong",
                "conditional_pair_correct",
                "conditional_pair_wrong",
            ]
            .into_iter()
            .map(|key| (key.to_owned(), 0usize))
            .collect::<BTreeMap<_, _>>();
            let mut truth_rmse = vec![];
            for c in &cases {
                let r = &c["rigid"][guard];
                if r["usable"] != true {
                    let reason = r["reason"]
                        .as_str()
                        .or_else(|| c["reason"].as_str())
                        .unwrap_or("unavailable");
                    count(&mut counts, format!("unavailable/{reason}"));
                    continue;
                }
                count(&mut counts, "usable".into());
                let winner = usize::from(
                    r["candidate_cv_rmse_px"][1].as_f64().unwrap()
                        < r["candidate_cv_rmse_px"][0].as_f64().unwrap(),
                );
                count(
                    &mut counts,
                    if winner as u64 == n(&c["target_branch"]) {
                        "target_rank_correct"
                    } else {
                        "target_rank_wrong"
                    }
                    .into(),
                );
                truth_rmse.push(
                    r["candidate_cv_rmse_px"][n(&c["target_branch"]) as usize]
                        .as_f64()
                        .unwrap(),
                );
                let pref = &r["conditional_pair_preference"];
                if !pref.is_null() {
                    count(
                        &mut counts,
                        if pref[0] == c["source_branch"] && pref[1] == c["target_branch"] {
                            "conditional_pair_correct"
                        } else {
                            "conditional_pair_wrong"
                        }
                        .into(),
                    );
                }
            }
            variants.push(json!({"guard_px":([16,24][guard]),"counts":counts,"true_target_cv_rmse_px":stats(truth_rmse)}));
        }
        let mut stable = [0usize; 2];
        for c in &cases {
            let p = &c["rigid"][0]["conditional_pair_preference"];
            if !p.is_null() && *p == c["rigid"][1]["conditional_pair_preference"] {
                stable[usize::from(p[0] != c["source_branch"] || p[1] != c["target_branch"])] += 1;
            }
        }
        let counts = &variants[0]["counts"];
        report += &format!(
            "| {name} | {} | {} | {} | {} | {} | {} / {} |\n",
            cases.len(),
            n(&counts["usable"]),
            n(&counts["target_rank_correct"]),
            n(&counts["conditional_pair_correct"]),
            n(&counts["conditional_pair_wrong"]),
            stable[0],
            stable[1]
        );
        synthetic_metrics.insert(name,json!({"cases":cases.len(),"variants":variants,"stable_conditional_correct":stable[0],"stable_conditional_wrong":stable[1]}));
    }
    report += "\nSynthetic correspondences are generated with known rigid geometry on recorded conic/mask layouts, across all four branch pairs. They do not establish real RAW point identity, conic accuracy or spherical anatomy. The radius and torsion here match the declared generator, not measured anatomy. Real masks remain unverified historical proposals used only for diagnostics. No model is trained or promoted. The upstream frontal-area gate lacks independent scale support; this experiment changes no conic or SN-FEIDA value. Physical sign accuracy and reviewed localization error remain unknown.\n";
    let mut hashes = BTreeMap::new();
    for (name, path) in [("baseline", base), ("candidate", dir)] {
        for file in ["summary.json", "tracks.jsonl", "polar-controls.jsonl"] {
            hashes.insert(
                format!("{name}/{file}"),
                archive::digest(&fs::read(path.join(file))?),
            );
        }
    }
    fs::write(
        dir.join("evaluation.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"native_track_and_polar_control_parity":true,"baseline":baseline,"input_hashes":hashes,"real":real,"synthetic":synthetic_metrics,"measured_sign_accuracy":null,"reviewed_sclera_localization_error":null}),
        )?,
    )?;
    fs::write(dir.join("README.md"), report)?;
    Ok(())
}

/// Compare completed temporal runs without rerunning or selecting new fits.
pub fn temporal(baseline: &str, candidate: &str, output: &str) -> Result<()> {
    let dirs = [Path::new(baseline), Path::new(candidate)];
    let summaries = [
        load(&dirs[0].join("summary.json"))?,
        load(&dirs[1].join("summary.json"))?,
    ];
    for s in &summaries {
        if s["complete"] != true {
            return Err("temporal report requires completed runs".into());
        }
    }
    for field in [
        "retained_inputs_sha256",
        "area_summary_sha256",
        "mask_thresholds",
        "light_directions",
        "radius_grid",
        "windows_ms",
        "maximum_source_gap_ms",
        "minimum_distinct_source_times",
        "anatomy",
    ] {
        if summaries[0][field] != summaries[1][field] {
            return Err(format!("temporal comparison differs in {field}").into());
        }
    }
    let rows = [
        rows(&dirs[0].join("temporal.jsonl"))?,
        rows(&dirs[1].join("temporal.jsonl"))?,
    ];
    let key = |r: &Value| {
        (
            r["provider"].as_str().unwrap_or("").to_owned(),
            n(&r["record"]),
        )
    };
    let indexed = rows
        .each_ref()
        .map(|rr| rr.iter().map(|r| (key(r), r)).collect::<BTreeMap<_, _>>());
    if indexed[0].len() != rows[0].len()
        || indexed[1].len() != rows[1].len()
        || indexed[0].keys().ne(indexed[1].keys())
    {
        return Err("temporal comparison has duplicate or unmatched source records".into());
    }
    for (k, a) in &indexed[0] {
        let b = indexed[1][k];
        for field in ["raw_sha256", "source_ns", "area_admission", "class"] {
            if a[field] != b[field] {
                return Err(format!("temporal source/admission mismatch: {k:?}/{field}").into());
            }
        }
        if a["area_admission"]["accepted"] != true
            || !a["physical_sign_truth"].is_null()
            || !b["physical_sign_truth"].is_null()
        {
            return Err(
                "temporal report requires admitted diagnostic rows without asserted real truth"
                    .into(),
            );
        }
        for v in 0..8 {
            for field in ["mask_threshold", "green_phase", "window_ms"] {
                if a["variants"][v][field] != b["variants"][v][field] {
                    return Err("temporal variant mismatch".into());
                }
            }
        }
    }
    let best = |v: &Value, field: &str| -> Option<f64> {
        v[field]
            .as_array()?
            .iter()
            .filter_map(Value::as_f64)
            .min_by(f64::total_cmp)
    };
    let mut metrics = BTreeMap::new();
    let mut text = String::from("# Shared temporal sclera-lighting comparison\n\nOnly area-admitted observations enter either run. Source identities, RAW hashes, source times, area admissions and fixed search settings match. These are single-user diagnostics, not measured sign accuracy. The report compares the same unknown-light model before and after enforcing unshadowed feasibility during optimization.\n\n| Provider | Ambiguous rows | Usable at 2 s / threshold 179 / green phase 0 | Baseline stable choices | Candidate stable choices | Candidate choices in any single setting |\n| --- | ---: | ---: | ---: | ---: | ---: |\n");
    for provider in ["sam", "obelisk"] {
        let cohort = rows[1]
            .iter()
            .filter(|r| r["provider"] == provider && r["class"] == "multiple")
            .collect::<Vec<_>>();
        let mut runs = vec![];
        for ri in 0..2 {
            let selected = cohort
                .iter()
                .map(|r| indexed[ri][&key(r)])
                .collect::<Vec<_>>();
            let mut reasons = BTreeMap::<String, usize>::new();
            for r in &selected {
                let v = &r["variants"][4];
                if v["usable"] != true {
                    *reasons
                        .entry(v["reason"].as_str().unwrap_or("unknown").to_owned())
                        .or_default() += 1;
                }
            }
            let usable = selected
                .iter()
                .filter(|r| r["variants"][4]["usable"] == true)
                .collect::<Vec<_>>();
            let ratios = usable
                .iter()
                .filter_map(|r| {
                    let v = &r["variants"][4];
                    let independent = best(&v["matched_independent_grid"], "candidate_cv_mse")?;
                    (independent > 1e-8)
                        .then(|| best(v, "shared_light_candidate_cv_mse").unwrap() / independent)
                })
                .collect();
            runs.push(json!({"run":dirs[ri],"ambiguous_rows":selected.len(),"stable_choices":selected.iter().filter(|r|r["stable_conditional_preference"].is_number()).count(),"any_variant_choice_rows":selected.iter().filter(|r|r["variants"].as_array().unwrap().iter().any(|v|v["conditional_preference"].is_number())).count(),"v4_usable":usable.len(),"v4_unavailable_reasons":reasons,"v4_shared_choices":usable.iter().filter(|r|r["variants"][4]["conditional_preference"].is_number()).count(),"v4_matched_independent_choices":usable.iter().filter(|r|r["variants"][4]["matched_independent_grid"]["conditional_preference"].is_number()).count(),"shared_vs_independent_best_cv_ratio":stats(ratios)}));
        }
        let relative = cohort
            .iter()
            .filter_map(|r| {
                let a = &indexed[0][&key(r)]["variants"][4];
                let b = &r["variants"][4];
                if a["usable"] != true || b["usable"] != true {
                    return None;
                }
                let av = best(a, "shared_light_candidate_cv_mse")?;
                (av > 1e-8).then(|| best(b, "shared_light_candidate_cv_mse").unwrap() / av)
            })
            .collect();
        text += &format!(
            "| {provider} | {} | {} | {} | {} | {} |\n",
            cohort.len(),
            runs[1]["v4_usable"],
            runs[0]["stable_choices"],
            runs[1]["stable_choices"],
            runs[1]["any_variant_choice_rows"]
        );
        metrics.insert(
            provider,
            json!({"runs":runs,"candidate_vs_baseline_best_shared_cv_ratio":stats(relative)}),
        );
    }
    let controls = load(&dirs[1].join("controls.json"))?;
    let mut control_metrics = vec![];
    text+="\nThe synthetic controls use actual retained sampling layouts, a known off-grid light, a spherical globe and bounded 3-code noise. Each scenario starts with forty layouts; insufficient temporal context remains an abstention.\n\n| Synthetic scenario | Correct selections | Wrong selections | Abstained |\n| --- | ---: | ---: | ---: |\n";
    for case in controls["cases"]
        .as_array()
        .ok_or("missing synthetic cases")?
    {
        let mut counts = BTreeMap::<String, usize>::new();
        for r in case["results"]
            .as_array()
            .ok_or("missing synthetic results")?
        {
            *counts
                .entry(r["outcome"].as_str().ok_or("missing outcome")?.to_owned())
                .or_default() += 1;
        }
        text += &format!(
            "| {} | {} | {} | {} |\n",
            case["scenario"].as_str().unwrap(),
            counts.get("correct").unwrap_or(&0),
            counts.get("wrong").unwrap_or(&0),
            counts.get("abstained").unwrap_or(&0)
        );
        control_metrics.push(json!({"scenario":case["scenario"],"generated_rows":case["generated_rows"],"outcomes":counts}));
    }
    text+="\nA stable selection must survive both mask thresholds, both native green photosite phases and both window lengths. A preference confined to one setting is not a recovery. Finite light/radius grids, historical unverified custom sclera masks, unknown actual lighting and sphere-model departures limit the inference. No reviewed lid/sclera localization, physical sign truth or independent scale was added. The area gate is unnormalized frontal-equivalent iris disk area, not full SN-FEIDA. Conics and live behavior are unchanged. See evaluation.json for availability, matched independent-light comparisons and numerical controls.\n";
    let mut hashes = BTreeMap::new();
    for (ri, dir) in dirs.iter().enumerate() {
        for file in ["summary.json", "temporal.jsonl", "controls.json"] {
            hashes.insert(
                format!("{ri}/{file}"),
                archive::digest(&fs::read(dir.join(file))?),
            );
        }
    }
    let result = json!({"complete":true,"matched_records":rows[0].len(),"metrics":metrics,"synthetic_controls":control_metrics,"maximum_direct_vs_stats_mse_error":controls["maximum_direct_vs_stats_mse_error"],"physical_feasibility_bound_direct_checks":controls["physical_feasibility_bound_direct_checks"],"input_hashes":hashes,"real_sign_accuracy":null,"reviewed_localization_error":null,"independent_scale":false});
    fs::create_dir(output)?;
    fs::write(
        Path::new(output).join("evaluation.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    fs::write(Path::new(output).join("README.md"), text)?;
    Ok(())
}

pub fn run(dir: &str) -> Result<()> {
    let dir = Path::new(dir);
    let summary = load(&dir.join("summary.json"))?;
    if summary["complete"] != true {
        return Err("report requires complete run".into());
    }
    if dir.join("README.md").exists() || dir.join("evaluation.json").exists() {
        return Err("surface report already exists".into());
    }
    let geometry = rows(&dir.join("results.jsonl"))?;
    let photo = rows(&dir.join("photometry.jsonl"))?;
    let pair = rows(&dir.join("paired-photometry.jsonl"))?;
    let oracle = load(&dir.join("photometry-controls.json"))?;
    let by_id = geometry
        .iter()
        .map(|r| ((n(&r["record"]), r["provider"].as_str().unwrap()), r))
        .collect::<BTreeMap<_, _>>();
    let mut provider_metrics = BTreeMap::new();
    let mut controlled: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    let mut ambient = vec![];
    for row in &photo {
        let g = by_id[&(n(&row["record"]), row["provider"].as_str().unwrap())];
        if row["raw_sha256"] != g["raw_sha256"]
            || row["area_admission"] != g["area_admission"]
            || row["area_admission"]["accepted"] != true
        {
            return Err("photometry/source admission identity mismatch".into());
        }
    }
    for provider in ["sam", "obelisk"] {
        let cohort = geometry
            .iter()
            .filter(|r| r["provider"] == provider && r["original_focus_class"] == "multiple")
            .collect::<Vec<_>>();
        let ids = cohort
            .iter()
            .map(|r| n(&r["record"]))
            .collect::<BTreeSet<_>>();
        let observed = photo
            .iter()
            .filter(|r| r["provider"] == provider && ids.contains(&n(&r["record"])))
            .collect::<Vec<_>>();
        let free_beats = observed
            .iter()
            .filter(|r| {
                r["variants"].as_array().unwrap().iter().all(|v| {
                    v["independent_center"]["nested_spatial_cv"][0]
                        .as_f64()
                        .zip(v["gradient_cv_mse"].as_f64())
                        .is_some_and(|(a, b)| a < 0.95 * b)
                })
            })
            .count();
        let spans = observed
            .iter()
            .filter_map(|r| {
                let p =
                    &r["variants"][0]["independent_center"]["profile_support"][1]["center_span_px"];
                p[0].as_f64().zip(p[1].as_f64()).map(|(x, y)| x.max(y))
            })
            .collect();
        let gaps = observed
            .iter()
            .flat_map(|r| r["variants"].as_array().unwrap())
            .filter_map(|v| {
                v["candidates"][0]["predictive"][0]
                    .as_f64()
                    .zip(v["candidates"][1]["predictive"][0].as_f64())
                    .zip(v["constant_cv_mse"].as_f64())
                    .map(|((a, b), c)| (a - b).abs() / c)
            })
            .collect();
        let nominal = cohort
            .iter()
            .filter_map(|r| {
                r["center_separation"]
                    ["sufficient_independent_center_error_radius_px_strictly_less_than"]
                    .as_f64()
            })
            .collect();
        let broad = cohort
            .iter()
            .filter_map(|r| {
                r["broad_center_separation"]
                    ["sufficient_independent_center_error_radius_px_strictly_less_than"]
                    .as_f64()
            })
            .collect();
        let pairs = pair
            .iter()
            .filter(|r| r["provider"] == provider)
            .collect::<Vec<_>>();
        provider_metrics.insert(provider,json!({"ambiguous_frames":cohort.len(),"photometry_frames":observed.len(),"stable_real_photometric_preferences":observed.iter().filter(|r|r["stable_nominal_preference"]==true).count(),"free_shape_beats_gradient_by5pct_all4variants":free_beats,"nominal_max_axis_profile_span_at5pct_constant_error_px":stats(spans),"absolute_pose_error_gap_over_constant_error_all_variants":stats(gaps),"nominal_sufficient_center_error_px":stats(nominal),"broad_sufficient_center_error_px":stats(broad),"exact_source_pairs":pairs.len(),"stable_paired_preferences":pairs.iter().filter(|r|r["stable_preference"]==true).count()}));
    }
    for row in oracle.as_array().ok_or("invalid control collection")? {
        let provider = row["provider"].as_str().ok_or("control provider missing")?;
        let g = by_id[&(n(&row["record"]), provider)];
        if g["original_focus_class"] != "multiple" {
            continue;
        }
        let bound = g["broad_center_separation"]
            ["sufficient_independent_center_error_radius_px_strictly_less_than"]
            .as_f64()
            .ok_or("broad geometric bound missing")?;
        for c in row["oracle"]["cases"]
            .as_array()
            .ok_or("oracle cases missing")?
        {
            if let Some(e) = c["center_error_px"].as_f64() {
                controlled
                    .entry(format!(
                        "{provider}/{}",
                        c["scenario"].as_str().unwrap_or("legacy-noise-only")
                    ))
                    .or_default()
                    .push((e, bound));
            }
        }
        if let Some(a) = row["oracle"]["uniform_ambient_counterexample"]["branch_cv_mse"].as_array()
        {
            ambient.extend(a.iter().filter_map(Value::as_f64));
        }
    }
    let control_metrics=controlled.into_iter().map(|(key,v)| {let passes=v.iter().filter(|(e,b)|e<b).count();(key,json!({"synthetic_candidate_cases":v.len(),"center_error_px":stats(v.iter().map(|v|v.0).collect()),"below_broad_separation_bound":passes,"real_measurements":false}))}).collect::<BTreeMap<_,_>>();
    let mut hash_map = BTreeMap::new();
    for file in [
        "summary.json",
        "results.jsonl",
        "photometry.jsonl",
        "paired-photometry.jsonl",
        "photometry-controls.json",
    ] {
        hash_map.insert(file, archive::digest(&fs::read(dir.join(file))?));
    }
    let evaluation = json!({"providers":provider_metrics,"controlled_lighting_sensitivity":control_metrics,"uniform_ambient_counterexample_mse":stats(ambient),"source_results_sha256":hash_map,"real_sign_labels":false,"independent_scale":false,"physical_sign_accuracy":null});
    let mut report=String::from("# Sclera shape and iris-sign experiment\n\nThe geometric target is promising: independently measuring the projected globe center precisely enough would separate every remaining candidate family in this subset. The current natural-light fits do not achieve that measurement. Controlled-light synthetic recovery does, under the declared sphere and radiometry assumptions.\n\n## Actual recordings\n\nOnly upstream area-admitted data are used. The original source remains 239, epoch 172; these are Rob-only observations. The 202 SAM and 193 Obelisk ambiguous provider rows overlap in RAW exposures and are not 395 different recordings. Additional one/zero-interpretation rows provide matched diagnostic comparisons.\n\n| Provider | Ambiguous frames evaluated | Stable natural-light choices | Free shape beats gradient, all mask/CFA settings | Exact-source eye pairs | Stable paired choices |\n| --- | ---: | ---: | ---: | ---: | ---: |\n");
    for provider in ["sam", "obelisk"] {
        let p = &evaluation["providers"][provider];
        report += &format!(
            "| {provider} | {} | {} | {} | {} | {} |\n",
            p["photometry_frames"],
            p["stable_real_photometric_preferences"],
            p["free_shape_beats_gradient_by5pct_all4variants"],
            p["exact_source_pairs"],
            p["stable_paired_preferences"]
        );
    }
    report+="\nThe free-center fit sometimes predicts brightness better than a flat gradient. That is useful shape evidence, but does not establish an accurate globe center. Near-best centers usually span much more than the sign-separation requirement. Negative ambient/unshadowed-directional violations, mask contamination, lid shadows and complex reflections remain material failures. The two-eye comparison fits shared illumination with separate ambient terms and checks fixed versus adjustable relative albedo. It requires identical provider/source/epoch/sequence/source-time and never uses a rejected eye frame as a partner.\n\n## Conditional geometric requirement\n\nFor a circular iris on a spherical globe, `G = C - sqrt(R²-1)n`. The two allowed projected center segments are disjoint here. An independent center error strictly less than half their minimum separation cannot match both, provided the true anatomy, conic and camera lie inside the declared model.\n\n| Provider | Smallest bound, R/iris 1.8–2.4 | Smallest bound, R/iris 1.5–2.8 |\n| --- | ---: | ---: |\n";
    for p in ["sam", "obelisk"] {
        report += &format!(
            "| {p} | {:.2} px | {:.2} px |\n",
            evaluation["providers"][p]["nominal_sufficient_center_error_px"]["min"]
                .as_f64()
                .unwrap(),
            evaluation["providers"][p]["broad_sufficient_center_error_px"]["min"]
                .as_f64()
                .unwrap()
        );
    }
    report+="\nThese are required error bounds, not measured center accuracy. Shape departures and camera/conic errors consume this allowance. Radius alone or two lid outlines do not determine the sign: both interpretations can have the same curvature and identical 2D iris/lid overlap.\n\n## Synthetic measurement assay\n\nThree linearly independent known lights recover per-sample normals after ambient subtraction; varying albedo cancels by normalizing the recovered vector. Exact perspective ray/normal equations then recover the globe center up to scale. The following uses actual retained sample layouts but SYNTHETIC brightness, separately for both candidate globes. It is not real-corpus sign accuracy.\n\n| Provider/scenario | Synthetic candidate cases | Worst center error | Cases below broad separation bound |\n| --- | ---: | ---: | ---: |\n";
    for (key, v) in evaluation["controlled_lighting_sensitivity"]
        .as_object()
        .unwrap()
    {
        report += &format!(
            "| {key} | {} | {:.3} px | {} |\n",
            v["synthetic_candidate_cases"],
            v["center_error_px"]["max"].as_f64().unwrap(),
            v["below_broad_separation_bound"]
        );
    }
    report+="\nThe light-direction offsets are one tested rotation axis, and the strength errors one specified pattern, not worst-case bounds over every possible calibration error. Noise is bounded deterministic variation, not a measured sensor-noise distribution. Ideal sphere shape, no glasses/refraction/cast shadows, known lighting correspondences, subtracted ambient and synchronized stable surface pose remain assumptions. Constant illumination reproduces both poses with essentially zero error on identical common samples, providing the opposing ambiguity control.\n\n## Review and reproduction\n\nInspect [RAW and free-center profile](photometry-90874-sam.png), [the matching Obelisk fit](photometry-90874-obelisk.png), [the other eye](photometry-90875-sam.png), and [its next exposure](photometry-90877-sam.png). The white curve is the fitted iris; dots are measured RAW green samples; colored centers/silhouettes are hypotheses. Display demosaicing does not enter numerical measurements. The broader pilot includes later exposures where the sphere fit follows a sclera/lid brightness gradient.\n\nNative commands and derivations are documented in `docs/sclera-shape-sign.md`. `evaluation.json` records counts, distributions and hashes of the frozen source results. Recreate this report with `buttercup_roi_focus surface-report RUN_DIR` before a report exists. Native source-tree audit and build are separate checks, not evidence of sign accuracy.\n\n## Next measurement\n\nRecord independently identifiable spatial lighting changes with source-aligned linear RAW, an ambient/reference condition, bounded eye motion between conditions, and camera-relative light direction/strength calibration or its uncertainty. Prefer several lights/patterns over the minimum to reject shadows and specular contamination and validate on held-out conditions. Review native RAW sclera/lid/limbus labels with the canonical labeler. Independently establish sign/center truth before training or claiming real accuracy. The current archived predictions and presumed mounting are not such truth.\n\nNo model was trained or promoted. Existing unverified Obelisk masks were used only for explicit historical diagnostic comparison. Input conics and frontal-equivalent area remain unchanged; no independent scale was added, so this makes no SN-FEIDA improvement claim.\n";
    fs::write(
        dir.join("evaluation.json"),
        serde_json::to_vec_pretty(&evaluation)?,
    )?;
    fs::write(dir.join("README.md"), report)?;
    eprintln!("SURFACE REPORT {}", dir.join("README.md").display());
    Ok(())
}
