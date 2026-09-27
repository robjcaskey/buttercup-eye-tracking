//! Appearance-constrained motion cohorts. Reflection-supported patches never
//! supply an iris-texture motion vote, even when their trajectories coincide.
use super::*;
fn bucket(t: &Value) -> usize {
    match t["kind"].as_str() {
        Some("bright/reflection") => 0,
        Some("texture candidate") => 1,
        _ => 2,
    }
}
const NAMES: [&str; 3] = [
    "Reflection-supported",
    "Iris-texture candidates",
    "Rim / context",
];
const COLORS: [&str; 3] = ["#ffa74f", "#62f4d3", "#d7a0ff"];
fn count(track: &Value) -> usize {
    track["frames"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["accepted"] == true && !f["source_seed"].as_bool().unwrap_or(false))
        .count()
}
fn distance(a: &Value, b: &Value, frames: &[Frame], train: bool) -> Option<f64> {
    let mut differences = vec![];
    let (pa, pb) = (xy(&a["source_sensor"]), xy(&b["source_sensor"]));
    let start = frames[0].meta["input"]["frame"]["timestamp_ns"]
        .as_u64()
        .unwrap();
    let end = frames.last().unwrap().meta["input"]["frame"]["timestamp_ns"]
        .as_u64()
        .unwrap();
    for j in 0..frames.len() {
        let time = frames[j].meta["input"]["frame"]["timestamp_ns"]
            .as_u64()
            .unwrap();
        if (time - start) * 2 >= end - start && train {
            continue;
        }
        if (time - start) * 2 < end - start && !train {
            continue;
        }
        if a["frames"][j]["accepted"] != true
            || b["frames"][j]["accepted"] != true
            || a["frames"][j]["source_seed"] == true
        {
            continue;
        }
        let da = sub(xy(&a["frames"][j]["sensor"]), pa);
        let db = sub(xy(&b["frames"][j]["sensor"]), pb);
        differences.push(dist(da, db));
    }
    if differences.len() < 3 {
        None
    } else {
        Some((differences.iter().map(|v| v * v).sum::<f64>() / differences.len() as f64).sqrt())
    }
}
fn cohorts(rows: &[Value], frames: &[Frame]) -> Vec<Value> {
    let mut output = vec![];
    for material in 0..3 {
        let mut order = (0..rows.len())
            .filter(|&i| bucket(&rows[i]) == material)
            .collect::<Vec<_>>();
        let start = uint_local(&frames[0].meta["input"]["frame"]["timestamp_ns"]);
        let end = uint_local(&frames.last().unwrap().meta["input"]["frame"]["timestamp_ns"]);
        order.sort_by_key(|&i| {
            std::cmp::Reverse(
                rows[i]["frames"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .filter(|(j, f)| {
                        let time = uint_local(&frames[*j].meta["input"]["frame"]["timestamp_ns"]);
                        (time - start) * 2 < end - start
                            && f["accepted"] == true
                            && f["source_seed"] != true
                    })
                    .count(),
            )
        });
        let mut groups: Vec<Vec<usize>> = vec![];
        for i in order {
            let mut best = None;
            for (g, members) in groups.iter().enumerate() {
                let distances = members
                    .iter()
                    .map(|&j| distance(&rows[i], &rows[j], frames, true))
                    .collect::<Option<Vec<_>>>();
                if let Some(d) = distances {
                    let maximum = d.into_iter().fold(0., f64::max);
                    if maximum <= 2. && best.is_none_or(|(_, v)| maximum < v) {
                        best = Some((g, maximum));
                    }
                }
            }
            if let Some((g, _)) = best {
                groups[g].push(i);
            } else {
                groups.push(vec![i]);
            }
        }
        for g in groups {
            let mut train = vec![];
            let mut held = vec![];
            for i in 0..g.len() {
                for j in 0..i {
                    if let Some(d) = distance(&rows[g[i]], &rows[g[j]], frames, true) {
                        train.push(d);
                    }
                    if let Some(d) = distance(&rows[g[i]], &rows[g[j]], frames, false) {
                        held.push(d);
                    }
                }
            }
            output.push(json!({"material_bucket":material,"name":NAMES[material],"ids":g.iter().map(|&i|rows[i]["id"].clone()).collect::<Vec<_>>(),"cohesive_training_group":g.len()>=3,"training_pair_rms_median_px":if train.is_empty(){None}else{Some(med(train))},"heldout_pair_rms_median_px":if held.is_empty(){None}else{Some(med(held.clone()))},"heldout_pair_count":held.len()}));
        }
    }
    output
}
pub fn run(long: &str, original: &str, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("use new output directory".into());
    }
    if !out
        .parent()
        .ok_or("no parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let extended: Value = serde_json::from_slice(&fs::read(long)?)?;
    let baseline: Value = serde_json::from_slice(&fs::read(original)?)?;
    let mut reports = vec![];
    fs::create_dir(out)?;
    for s in baseline["series"].as_array().unwrap() {
        let long_s = extended["series"]
            .as_array()
            .unwrap()
            .iter()
            .find(|l| l["capture"] == s["capture"] && l["eye"] == s["eye"]);
        let metadata = long_s.unwrap_or(s)["frames"].as_array().unwrap();
        let frames = metadata
            .iter()
            .map(Frame::load)
            .collect::<Result<Vec<_>, _>>()?;
        let si = metadata
            .iter()
            .position(|f| f["sequence"] == s["source_sequence"])
            .ok_or("source missing")?;
        let old = s["tracks"].as_array().unwrap();
        let mut rows = old.clone();
        for row in &mut rows {
            row["frames"] = json!(vec![
                json!({"accepted":false,"reason":"not evaluated"});
                frames.len()
            ]);
            row["frames"][si] =
                json!({"accepted":true,"source_seed":true,"sensor":row["source_sensor"]});
            row["material_bucket"] = json!(bucket(row));
        }
        for direction in [-1isize, 1] {
            let mut previous = si;
            let mut shift = [[0.; 2]; 3];
            loop {
                let j = previous as isize + direction;
                if j < 0 || j >= frames.len() as isize {
                    break;
                }
                let j = j as usize;
                for t in &mut rows {
                    let b = bucket(t);
                    let source = xy(&t["source_sensor"]);
                    let search_center = if t["frames"][previous]["accepted"] == true {
                        xy(&t["frames"][previous]["sensor"])
                    } else {
                        [source[0] + shift[b][0], source[1] + shift[b][1]]
                    };
                    let mut found = track_pair(
                        &frames[si],
                        &frames[j],
                        source,
                        &frames[si],
                        source,
                        [3, 3],
                        6,
                        TrackingPolicy::Iris,
                        Some(search_center),
                    );
                    found["search_center_sensor"] = json!(search_center);
                    t["frames"][j] = found;
                }
                for b in 0..3 {
                    let displacements = rows
                        .iter()
                        .filter(|t| bucket(t) == b && t["frames"][j]["accepted"] == true)
                        .map(|t| sub(xy(&t["frames"][j]["sensor"]), xy(&t["source_sensor"])))
                        .collect::<Vec<_>>();
                    if displacements.len() >= 3 {
                        shift[b] = std::array::from_fn(|a| {
                            med(displacements.iter().map(|v| v[a]).collect())
                        });
                    }
                }
                previous = j;
            }
        }
        let groups = cohorts(&rows, &frames);
        let counts=(0..3).map(|b|json!({"bucket":b,"name":NAMES[b],"source_candidates":rows.iter().filter(|t|bucket(t)==b).count(),"fresh_per_frame":(0..frames.len()).map(|j|rows.iter().filter(|t|bucket(t)==b&&t["frames"][j]["accepted"]==true&&j!=si).count()).collect::<Vec<_>>(),"ids":rows.iter().filter(|t|bucket(t)==b).map(|t|t["id"].clone()).collect::<Vec<_>>()})).collect::<Vec<_>>();
        let name = format!("{}-eye{}", s["capture"].as_str().unwrap(), s["eye"]);
        render(&name, &frames, si, &rows, &groups, out)?;
        println!(
            "{name}: {}",
            json!({"counts":counts,"cohesive_groups":groups.iter().filter(|g|g["cohesive_training_group"]==true).collect::<Vec<_>>()})
        );
        reports.push(json!({"capture":s["capture"],"eye":s["eye"],"source_sequence":s["source_sequence"],"frames":metadata,"tracks":rows,"buckets":counts,"groups":groups}));
    }
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"original":original,"extended":long,"method":"Classify source patch support first (highlight, nonbright interior, or rim), then complete-link cluster measured source displacements within each class. Require >=3 shared fresh exposures per comparison and <=2px pair RMS in first half. Second half is held out of group construction. Missingness stays explicit.","tracking":"All original 64 IDs, source-template matches rechecked on every exposure with unchanged strict NCC/reverse/channel gates. Search centers follow each material bucket separately. No ellipse clips destination motion and no held point becomes evidence.","limitations":"Photometric/footprint categories are conservative hypotheses, not anatomical labels. Patches overlap. No coherent nonbright group is invented if measured support is insufficient. Five-frame second recording has too little training overlap for motion grouping.","series":reports}),
        )?,
    )?;
    Ok(())
}
fn render(
    name: &str,
    frames: &[Frame],
    si: usize,
    rows: &[Value],
    groups: &[Value],
    out: &Path,
) -> Result<(), E> {
    let mut manifest = vec![];
    let start = uint_local(&frames[0].meta["input"]["frame"]["timestamp_ns"]);
    let labels = (0..3)
        .map(|b| {
            let mut ids = rows.iter().filter(|t| bucket(t) == b).collect::<Vec<_>>();
            ids.sort_by_key(|t| std::cmp::Reverse(count(t)));
            ids.into_iter()
                .take(8)
                .map(|t| uint_local(&t["id"]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    for (j, f) in frames.iter().enumerate() {
        let seq = &f.meta["sequence"];
        let time = uint_local(&f.meta["input"]["frame"]["timestamp_ns"]);
        let mut svg=format!("<svg xmlns='http://www.w3.org/2000/svg' width='1800' height='780'><rect width='100%' height='100%' fill='#171b24'/><g font-family='sans-serif' fill='white'><text x='25' y='35' font-size='26'>Reflection and iris-texture evidence kept separate</text><text x='25' y='70' font-size='18'>{name} · source {seq} · +{:.1} ms · 5× slower · same original 64 source IDs</text><text x='25' y='100' font-size='16'>Buckets use full patch support, not just dot brightness. Circles = current matches; gray crosses = unavailable source identities.</text><defs><g id='raw' shape-rendering='crispEdges'>",(time-start) as f64*1e-6);
        for y in 0..f.h {
            for x in 0..f.w {
                write!(
                    svg,
                    "<rect x='{x}' y='{y}' width='1' height='1' fill='#{:06x}'/>",
                    f.rgb[y * f.w + x]
                )?;
            }
        }
        svg.push_str("</g></defs>");
        for b in 0..3 {
            let (x, y) = (25. + 600. * b as f64, 165.);
            let scale = 545. / f.w as f64;
            let color = COLORS[b];
            let members = rows.iter().filter(|t| bucket(t) == b).collect::<Vec<_>>();
            let good = members
                .iter()
                .filter(|t| t["frames"][j]["accepted"] == true)
                .count();
            write!(svg,"<text x='{x}' y='139' font-size='21' fill='{color}'>{} · {good}/{} {}</text><use href='#raw' transform='translate({x},{y}) scale({scale})'/>",NAMES[b],members.len(),if j==si{"seeds"}else{"matches"})?;
            for t in &members {
                let valid = t["frames"][j]["accepted"] == true;
                let p = xy(if valid {
                    &t["frames"][j]["sensor"]
                } else {
                    &t["source_sensor"]
                });
                let p = [
                    x + (p[0] - f.origin[0]) * scale,
                    y + (p[1] - f.origin[1]) * scale,
                ];
                if valid {
                    write!(svg,"<circle cx='{}' cy='{}' r='3.4' fill='none' stroke='{color}' stroke-width='1.5'/>",p[0],p[1])?;
                } else {
                    write!(svg,"<path d='M{},{} l5,5 m-5,0 l5,-5' stroke='#90949b' stroke-width='1' opacity='.55'/>",p[0]-2.5,p[1]-2.5)?;
                }
                if let Some(slot) = labels[b].iter().position(|id| *id == uint_local(&t["id"])) {
                    let lx = x + 30. + (slot % 4) as f64 * 135.;
                    let ly = 555. + (slot / 4) as f64 * 30.;
                    let paint = if valid { color } else { "#828690" };
                    if valid {
                        write!(svg,"<path d='M{},{} L{lx},{}' stroke='{paint}' stroke-width='.65' opacity='.55' fill='none'/>",p[0],p[1],ly-9.)?;
                    }
                    write!(svg,"<rect x='{}' y='{}' width='30' height='20' rx='4' fill='#252a34'/><text x='{lx}' y='{ly}' text-anchor='middle' font-size='14' fill='{paint}'>{}</text>",lx-15.,ly-15.,t["id"])?;
                }
            }
            let coherent = groups
                .iter()
                .filter(|g| g["material_bucket"] == b && g["cohesive_training_group"] == true)
                .collect::<Vec<_>>();
            write!(svg,"<text x='{x}' y='629' font-size='17'>{} supported motion groups in first-half history</text>",coherent.len())?;
            let mut motions = vec![];
            for t in &members {
                if t["frames"][j]["accepted"] == true && j != si {
                    motions.push(sub(xy(&t["frames"][j]["sensor"]), xy(&t["source_sensor"])));
                }
            }
            if motions.len() >= 3 {
                let m: [f64; 2] =
                    std::array::from_fn(|a| med(motions.iter().map(|p| p[a]).collect()));
                write!(svg,"<text x='{x}' y='660' font-size='17' fill='{color}'>Fresh median shift: x {:+.2}px / y {:+.2}px</text>",m[0],m[1])?;
            } else {
                write!(svg,"<text x='{x}' y='660' font-size='17' fill='#9297a2'>Fewer than 3 fresh points: no group shift</text>")?;
            }
        }
        svg.push_str("<text x='25' y='720' font-size='17'>Similar motion cannot merge a reflection-supported patch into iris-texture evidence. These buckets do not prove material identity.</text><text x='25' y='751' font-size='17'>Source templates and original strict quality checks remain in force; failed tracks may be remeasured, never held as observations.</text></g></svg>");
        let filename = format!("{name}-{j:03}.svg");
        fs::write(out.join(&filename), svg)?;
        let duration = if j + 1 < frames.len() {
            (uint_local(&frames[j + 1].meta["input"]["frame"]["timestamp_ns"]) - time) as f64 * 1e-9
        } else {
            0.1
        };
        manifest.push(json!({"sequence":seq,"file":filename,"duration_s":duration*5.,"raw_sha256":f.meta["raw_sha256"]}));
    }
    fs::write(
        out.join(format!("{name}-animation.json")),
        serde_json::to_vec_pretty(&json!({"frames":manifest}))?,
    )?;
    Ok(())
}
fn uint_local(v: &Value) -> u64 {
    v.as_u64().unwrap()
}

/// Image transport only, avoiding thousands of unnecessary essential-matrix
/// perturbation fits when the downstream question needs a 2D alignment.
pub fn outer_transport(input: &str, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("use new output directory".into());
    }
    if !out
        .parent()
        .ok_or("no parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let root: Value = serde_json::from_slice(&fs::read(input)?)?;
    let s = root["series"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .ok_or("missing series")?;
    let frames = s["frames"].as_array().unwrap();
    let tracks = s["outer_tracks"].as_array().ok_or("missing outer tracks")?;
    let mut results = vec![];
    for (j, f) in frames.iter().enumerate() {
        let pairs = tracks
            .iter()
            .filter(|t| t["frames"][j]["accepted"] == true)
            .map(|t| (xy(&t["source_sensor"]), xy(&t["frames"][j]["sensor"])))
            .collect::<Vec<_>>();
        let mut r = json!({"sequence":f["sequence"],"raw_sha256":f["raw_sha256"],"source_sequence":s["source_sequence"],"reference":f["sequence"]==s["source_sequence"],"outer":{"point_count":pairs.len()}});
        if pairs.len() >= 8 {
            let mut inliers = pairs.clone();
            for _ in 0..4 {
                let Some((angle, a, shift, _)) = rigid(&inliers) else {
                    break;
                };
                // The shared fit returns centroid displacement, not destination position.
                let b = [a[0] + shift[0], a[1] + shift[1]];
                let (sn, cs) = angle.sin_cos();
                let mut scored = pairs
                    .iter()
                    .map(|&(p, q)| {
                        let v = sub(p, a);
                        let predicted =
                            [b[0] + cs * v[0] - sn * v[1], b[1] + sn * v[0] + cs * v[1]];
                        (dist(predicted, q), (p, q))
                    })
                    .collect::<Vec<_>>();
                scored.sort_by(|a, b| a.0.total_cmp(&b.0));
                inliers = scored
                    .into_iter()
                    .take((pairs.len() * 3 / 4).max(6))
                    .map(|v| v.1)
                    .collect();
            }
            if let Some((angle, a, shift, errors)) = rigid(&inliers) {
                let b = [a[0] + shift[0], a[1] + shift[1]];
                let (sn, cs) = angle.sin_cos();
                let all_errors = pairs
                    .iter()
                    .map(|&(p, q)| {
                        let v = sub(p, a);
                        dist(
                            [b[0] + cs * v[0] - sn * v[1], b[1] + sn * v[0] + cs * v[1]],
                            q,
                        )
                    })
                    .collect::<Vec<_>>();
                let evaluation_median = med(all_errors);
                if r["reference"] == true {
                    assert!(
                        evaluation_median < 1e-6,
                        "reference transform must be identity"
                    );
                }
                r["outer"]["rigid_2d"] = json!({"source_centroid":a,"current_centroid":b,"rotation_degrees":angle.to_degrees(),"evaluation_median_euclidean_px":evaluation_median,"fitted_median_px":med(errors),"inlier_count":inliers.len()});
            }
        }
        results.push(r);
    }
    fs::create_dir(out)?;
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"input":input,"scope":"Robust 2D rigid image alignment using original-source top/bottom patch matches only. No 3D motion is inferred.","frames":results}),
        )?,
    )?;
    Ok(())
}
