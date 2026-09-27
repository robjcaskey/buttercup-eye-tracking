//! Motion-only cohort diagnostic using the existing live signature kernel.
use super::{num, raw, raw_motion_octrees, raw_preview, uint, E};
use serde_json::{json, Value};
use std::{fmt::Write as _, fs, path::Path};
type P = [f64; 2];
fn xy(v: &Value) -> P {
    [num(&v[0]), num(&v[1])]
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn group(samples: &[Vec<P>], steps: usize, scale: f64) -> Vec<Vec<usize>> {
    let s = samples
        .iter()
        .map(|v| {
            v[..steps]
                .iter()
                .map(|p| [(p[0] * scale) as f32, (p[1] * scale) as f32])
                .collect()
        })
        .collect::<Vec<_>>();
    let mut groups = raw_motion_octrees::debug_cluster_motion_signatures(&s)
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<_>>();
    groups.sort_by_key(|g| std::cmp::Reverse(g.len()));
    groups
}
fn centroid(samples: &[Vec<P>], members: &[usize]) -> Vec<P> {
    (0..4)
        .map(|t| {
            std::array::from_fn(|j| {
                members.iter().map(|&i| samples[i][t][j]).sum::<f64>() / members.len() as f64
            })
        })
        .collect()
}
fn heldout(samples: &[Vec<P>], groups: &[Vec<usize>]) -> Value {
    let assigned = groups.iter().flatten().copied().collect::<Vec<_>>();
    let mut grouped = 0.;
    let mut common = 0.;
    let mut count = 0;
    for g in groups {
        if g.len() < 3 {
            continue;
        }
        for &i in g {
            for t in 2..4 {
                for j in 0..2 {
                    let mu = g
                        .iter()
                        .filter(|&&k| k != i)
                        .map(|&k| samples[k][t][j])
                        .sum::<f64>()
                        / (g.len() - 1) as f64;
                    let base = assigned
                        .iter()
                        .filter(|&&k| k != i)
                        .map(|&k| samples[k][t][j])
                        .sum::<f64>()
                        / (assigned.len() - 1) as f64;
                    grouped += (samples[i][t][j] - mu).powi(2);
                    common += (samples[i][t][j] - base).powi(2);
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        return json!({"eligible":false});
    }
    json!({"eligible":true,"points":assigned.len(),"grouped_rms_px_per_100ms":(grouped/count as f64).sqrt(),"common_rms_px_per_100ms":(common/count as f64).sqrt(),"skill_vs_one_group":1.-grouped/common.max(1e-12),"method":"groups assigned on first two increments only; last two increments scored with leave-one-feature-out group means"})
}
pub fn run(input: &str, out: &Path, motion_path: Option<&str>) -> Result<(), E> {
    if out.exists() {
        return Err("output must be new".into());
    }
    if !out
        .parent()
        .ok_or("missing output parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk output".into());
    }
    fs::create_dir(out)?;
    let root: Value = serde_json::from_slice(&fs::read(input)?)?;
    let s = root["series"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .ok_or("series missing")?;
    let frames = s["frames"].as_array().unwrap();
    assert_eq!(frames.len(), 5);
    let rows = s["tracks"].as_array().unwrap();
    let core = rows
        .iter()
        .filter(|t| {
            t["frames"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v["accepted"] == true)
        })
        .collect::<Vec<_>>();
    let mut motions = vec![vec![[0.; 2]; 4]; core.len()];
    for t in 0..4 {
        let dt = (uint(&frames[t + 1]["input"]["frame"]["timestamp_ns"])
            - uint(&frames[t]["input"]["frame"]["timestamp_ns"])) as f64
            / 1e9;
        if dt <= 0. || dt > 0.25 {
            return Err("gapped source time".into());
        }
        for (i, p) in core.iter().enumerate() {
            let a = xy(&p["frames"][t]["sensor"]);
            let b = xy(&p["frames"][t + 1]["sensor"]);
            motions[i][t] = [(b[0] - a[0]) * 0.1 / dt, (b[1] - a[1]) * 0.1 / dt];
        }
    }
    let common = (0..4)
        .map(|t| {
            [
                median(motions.iter().map(|p| p[t][0]).collect()),
                median(motions.iter().map(|p| p[t][1]).collect()),
            ]
        })
        .collect::<Vec<_>>();
    let residual = motions
        .iter()
        .map(|p| {
            p.iter()
                .zip(&common)
                .map(|(a, b)| [a[0] - b[0], a[1] - b[1]])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut variants = vec![];
    for scale in [1., 2., 4.] {
        let full = group(&residual, 4, scale);
        let train = group(&residual, 2, scale);
        let validation = heldout(&residual, &train);
        let mut null_skills = vec![];
        let mut state = 239u64;
        let ids = train.iter().flatten().copied().collect::<Vec<_>>();
        for _ in 0..100 {
            let mut shuffled = ids.clone();
            for i in (1..shuffled.len()).rev() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                shuffled.swap(i, (state as usize) % (i + 1));
            }
            let mut start = 0;
            let random = train
                .iter()
                .map(|g| {
                    let v = shuffled[start..start + g.len()].to_vec();
                    start += g.len();
                    v
                })
                .collect::<Vec<_>>();
            if let Some(skill) = heldout(&residual, &random)["skill_vs_one_group"].as_f64() {
                null_skills.push(skill);
            }
        }
        let describe = |groups: &[Vec<usize>]| {
            groups.iter().map(|g|json!({"ids":g.iter().map(|&i|core[i]["id"].clone()).collect::<Vec<_>>(),"size":g.len(),"mean_residual_motion_px_per_100ms":centroid(&residual,g),"mean_image_motion_px_per_100ms":centroid(&motions,g),"bright_flag_count":g.iter().filter(|&&i|core[i]["kind"]=="bright/reflection").count()})).collect::<Vec<_>>()
        };
        null_skills.sort_by(f64::total_cmp);
        let null_median = null_skills.get(null_skills.len() / 2).copied();
        let null_p95 = null_skills.get(null_skills.len() * 95 / 100).copied();
        variants.push(json!({"residual_multiplier":scale,"all_four_increment_groups":describe(&full),"first_two_increment_groups":describe(&train),"heldout":validation,"random_label_null_skill_median":null_median,"random_label_null_skill_p95":null_p95,"unassigned_ids":(0..core.len()).filter(|i|!full.iter().any(|g|g.contains(i))).map(|i|core[i]["id"].clone()).collect::<Vec<_>>()}));
    }
    let groups = &variants[0]["all_four_increment_groups"];
    let report = json!({"input":input,"scope":"motion cohorts, NOT physical depth or refractive-index estimates; no spatial or photometric labels used to create groups","frames":frames,"source_seed":1034,"next":1035,"common_motion_px_per_100ms":common,"persistent_core_count":core.len(),"primary_pair_count":rows.iter().filter(|t|t["frames"][3]["accepted"]==true).count(),"primary_pair_without_complete_history":rows.iter().filter(|t|t["frames"][3]["accepted"]==true&&!core.iter().any(|c|c["id"]==t["id"])).map(|t|t["id"].clone()).collect::<Vec<_>>(),"native_kernel":"raw_motion_octrees::cluster_signatures; max 4 groups; min 3 members; unchanged production thresholds","variants":variants,"groups":groups,"persistent_tracks":core.iter().enumerate().map(|(i,t)|json!({"id":t["id"],"residual_motion":residual[i],"image_motion":motions[i]})).collect::<Vec<_>>()});
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    let colors = ["#62f4d3", "#ffa74f", "#ed91ff", "#75adff"];
    let membership = |id: &Value| {
        report["groups"]
            .as_array()
            .unwrap()
            .iter()
            .position(|g| g["ids"].as_array().unwrap().contains(id))
    };
    let mut svg="<svg xmlns='http://www.w3.org/2000/svg' width='1420' height='1230'><rect width='100%' height='100%' fill='#171b24'/><g fill='white' font-family='sans-serif'><text x='25' y='36' font-size='25'>Motion cohorts from the same iris-region features</text><text x='25' y='69' font-size='18'>Existing live clustering kernel · four motion increments · colors indicate motion, not anatomy or optical material</text>".to_string();
    for (col, j) in [2, 3].iter().enumerate() {
        let f = &frames[*j];
        let m = &f["input"]["frame"];
        let (w, h) = (uint(&m["width"]) as usize, uint(&m["height"]) as usize);
        let (sx, sy) = (uint(&m["sensor_x"]), uint(&m["sensor_y"]));
        let raw = raw(f)?;
        let rgb = raw_preview::color_preview(&raw, w, h, sx as u32, sy as u32, 100, None);
        let scale = 650. / w as f64;
        let (x, y) = (25. + col as f64 * 700., 125.);
        write!(svg,"<text x='{x}' y='108' font-size='21'>Source {} · same IDs; gray = incomplete/unassigned</text><g transform='translate({x},{y}) scale({scale})' shape-rendering='crispEdges'>",f["sequence"])?;
        for yy in 0..h {
            for xx in 0..w {
                write!(
                    svg,
                    "<rect x='{xx}' y='{yy}' width='1' height='1' fill='#{:06x}'/>",
                    rgb[yy * w + xx]
                )?;
            }
        }
        svg.push_str("</g>");
        for t in rows.iter().filter(|t| t["frames"][3]["accepted"] == true) {
            let p = xy(&t["frames"][*j]["sensor"]);
            let (px, py) = (
                x + (p[0] - sx as f64) * scale,
                y + (p[1] - sy as f64) * scale,
            );
            let color = membership(&t["id"]).map(|k| colors[k]).unwrap_or("#b0b4bd");
            let id = uint(&t["id"]);
            write!(svg,"<circle cx='{px}' cy='{py}' r='4' fill='none' stroke='{color}' stroke-width='1.4'/><text x='{}' y='{}' font-size='12' fill='{color}' stroke='#111' stroke-width='2.5' paint-order='stroke'>{id}</text>",px+5.,py+if id%2==0{-8.}else{14.})?;
        }
    }
    for (i, g) in report["groups"].as_array().unwrap().iter().enumerate() {
        let x = 25. + i as f64 * 345.;
        let color = colors[i];
        write!(svg,"<text x='{x}' y='601' font-size='18' fill='{color}'>Group {}: {} persistent points</text>",i+1,g["size"])?;
    }
    // Residual trajectories are cumulative displacements after common translation.
    for axis in 0..2 {
        let x = 75. + axis as f64 * 700.;
        let y = 810.;
        let width = 570.;
        let gain = 45.;
        write!(svg,"<text x='{x}' y='655' font-size='19'>Residual {} displacement (native px)</text><path d='M{x},675 V960 M{x},{y} H{}' stroke='#78818e' fill='none'/>",if axis==0{"horizontal"}else{"vertical"},x+width)?;
        for level in -3..=3 {
            let yy = y - level as f64 * gain;
            write!(
                svg,
                "<text x='{}' y='{}' font-size='14'>{level}</text>",
                x - 25.,
                yy + 5.
            )?;
        }
        for (i, t) in core.iter().enumerate() {
            let color = membership(&t["id"]).map(|k| colors[k]).unwrap_or("#9398a2");
            let mut d = 0.;
            let mut points = format!("{x},{y}");
            for step in 0..4 {
                let dt = (uint(&frames[step + 1]["input"]["frame"]["timestamp_ns"])
                    - uint(&frames[step]["input"]["frame"]["timestamp_ns"]))
                    as f64
                    / 1e8;
                d += residual[i][step][axis] * dt;
                write!(
                    points,
                    " {},{}",
                    x + (step + 1) as f64 * width / 4.,
                    y - d * gain
                )?;
            }
            write!(svg,"<polyline points='{points}' stroke='{color}' stroke-width='1.2' opacity='.65' fill='none'/>")?;
        }
        for j in 0..5 {
            write!(
                svg,
                "<text x='{}' y='990' font-size='15'>{}</text>",
                x + j as f64 * width / 4. - 18.,
                frames[j]["sequence"]
            )?;
        }
    }
    let h = &report["variants"][0]["heldout"];
    write!(svg,"<text x='25' y='1050' font-size='19'>Chronological check: held-out motion prediction skill vs one group = {:.3}</text>",h["skill_vs_one_group"].as_f64().unwrap_or(0.))?;
    svg.push_str("<text x='25' y='1092' font-size='18'>Groups trained on the first two increments are checked on the last two using other features, not their own motion.</text><text x='25' y='1132' font-size='18'>Only 0.39 seconds of evidence; shared/overlapping patches and matching noise can imitate layers.</text><text x='25' y='1172' font-size='18'>Separate 3D fits are conditional experiments. Z scale and refractive indices are not measured by this clustering.</text></g></svg>");
    fs::write(out.join("motion-groups.svg"), svg)?;
    let motion = motion_path
        .map(|p| -> Result<Value, E> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
        .transpose()?
        .unwrap_or(Value::Null);
    if !motion.is_null() {
        assert_eq!(motion["frames"].as_array().unwrap().len(), frames.len());
        for (j, f) in frames.iter().enumerate() {
            assert_eq!(motion["frames"][j]["raw_sha256"], f["raw_sha256"]);
        }
    }
    animation(
        frames,
        rows,
        &report,
        out,
        s["outer_tracks"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[]),
        &motion,
    )?;
    println!("{}", serde_json::to_string_pretty(&report["variants"])?);
    Ok(())
}

pub fn replay(input: &str, cohort_path: &str, motion_path: &str, out: &Path) -> Result<(), E> {
    if out.exists() {
        return Err("output must be new".into());
    }
    if !out
        .parent()
        .ok_or("missing parent")?
        .canonicalize()?
        .starts_with("/mnt/bulk_data/buttercup-eye-tracking")
    {
        return Err("use checked bulk outputs".into());
    }
    let root: Value = serde_json::from_slice(&fs::read(input)?)?;
    let cohort: Value = serde_json::from_slice(&fs::read(cohort_path)?)?;
    let motion: Value = serde_json::from_slice(&fs::read(motion_path)?)?;
    let s = root["series"]
        .as_array()
        .ok_or("missing series")?
        .iter()
        .find(|s| s["capture"] == "later-recording" && s["eye"] == 2)
        .ok_or("missing series")?;
    let frames = s["frames"].as_array().unwrap();
    assert_eq!(frames.len(), motion["frames"].as_array().unwrap().len());
    assert_eq!(frames.len(), cohort["frames"].as_array().unwrap().len());
    for (j, f) in frames.iter().enumerate() {
        assert_eq!(f["raw_sha256"], motion["frames"][j]["raw_sha256"]);
        assert_eq!(f["raw_sha256"], cohort["frames"][j]["raw_sha256"]);
    }
    fs::create_dir(out)?;
    animation(
        frames,
        s["tracks"].as_array().unwrap(),
        &cohort,
        out,
        s["outer_tracks"].as_array().unwrap(),
        &motion,
    )
}

/// A measured-frame animation: fixed crop, IDs and callout slots, no pose warp.
fn animation(
    frames: &[Value],
    rows: &[Value],
    report: &Value,
    out: &Path,
    outer: &[Value],
    motion: &Value,
) -> Result<(), E> {
    let expanded = !motion.is_null();
    let ids = report["groups"][0]["ids"]
        .as_array()
        .ok_or("no coherent group to animate")?;
    let selected = rows
        .iter()
        .filter(|t| ids.contains(&t["id"]))
        .collect::<Vec<_>>();
    let source = if expanded {
        motion["source_index"]
            .as_u64()
            .ok_or("missing motion source index")? as usize
    } else {
        frames
            .iter()
            .position(|f| f["sequence"] == report["source_seed"])
            .unwrap_or(2)
    };
    let center = xy(&frames[source]["ellipse"]["center_sensor_px"]);
    let half = num(&frames[source]["ellipse"]["a"])
        .max(num(&frames[source]["ellipse"]["b"]))
        .ceil()
        + 20.;
    let zoom = 440. / (2. * half);
    let crop = [center[0] - half, center[1] - half];
    let mut ordered = (0..selected.len()).collect::<Vec<_>>();
    ordered.sort_by(|&a, &b| {
        num(&selected[a]["frames"][source]["sensor"][0])
            .total_cmp(&num(&selected[b]["frames"][source]["sensor"][0]))
    });
    let mut labels = vec![(false, 0usize, 0usize); selected.len()];
    let split = (ordered.len() + 1) / 2;
    for (right, mut side) in [
        (false, ordered[..split].to_vec()),
        (true, ordered[split..].to_vec()),
    ] {
        side.sort_by(|&a, &b| {
            num(&selected[a]["frames"][source]["sensor"][1])
                .total_cmp(&num(&selected[b]["frames"][source]["sensor"][1]))
        });
        for (k, &i) in side.iter().enumerate() {
            labels[i] = (right, k, side.len());
        }
    }
    let means = (0..frames.len())
        .map(|j| {
            if selected.iter().any(|t| t["frames"][j]["accepted"] != true) {
                return None;
            }
            Some(std::array::from_fn::<_, 2, _>(|axis| {
                selected
                    .iter()
                    .map(|t| num(&t["frames"][j]["sensor"][axis]))
                    .sum::<f64>()
                    / selected.len() as f64
            }))
        })
        .collect::<Vec<_>>();
    let mut paths = vec![(
        "Iris",
        "#62f4d3",
        means
            .iter()
            .map(|p| {
                p.map(|p| {
                    [
                        p[0] - means[source].unwrap()[0],
                        p[1] - means[source].unwrap()[1],
                    ]
                })
            })
            .collect::<Vec<_>>(),
    )];
    if expanded {
        paths = vec![];
        for (name, label, color) in [("iris", "Iris", "#62f4d3"), ("outer", "Outer", "#ffb15c")] {
            paths.push((
                label,
                color,
                motion["frames"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        let p = &v[name]["motion_at_fixed_roi_center_px"];
                        p.as_array().map(|_| xy(p))
                    })
                    .collect(),
            ));
        }
    }
    // One scale for the entire clip and both groups; no per-frame zoom jitter.
    let max_x = paths
        .iter()
        .flat_map(|p| p.2.iter().flatten())
        .map(|p| p[0].abs())
        .fold(0., f64::max);
    let max_y = paths
        .iter()
        .flat_map(|p| p.2.iter().flatten())
        .map(|p| p[1].abs())
        .fold(0., f64::max);
    let gain = 50f64.min(195. / max_x.max(1e-6)).min(58. / max_y.max(1e-6));
    let start = uint(&frames[0]["input"]["frame"]["timestamp_ns"]);
    let duration =
        (uint(&frames.last().unwrap()["input"]["frame"]["timestamp_ns"]) - start) as f64 / 1e9;
    let mut timing = vec![];
    for (j, f) in frames.iter().enumerate() {
        let m = &f["input"]["frame"];
        let (w, h) = (uint(&m["width"]) as usize, uint(&m["height"]) as usize);
        let (sx, sy) = (uint(&m["sensor_x"]), uint(&m["sensor_y"]));
        let rgb = raw_preview::color_preview(&raw(f)?, w, h, sx as u32, sy as u32, 100, None);
        let timestamp = uint(&m["timestamp_ns"]);
        let elapsed = (timestamp - start) as f64 / 1e6;
        let mut svg=format!("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='1600' height='{}'><defs><g id='raw' shape-rendering='crispEdges'>",if expanded{1290}else{900});
        for y in 0..h {
            for x in 0..w {
                write!(
                    svg,
                    "<rect x='{x}' y='{y}' width='1' height='1' fill='#{:06x}'/>",
                    rgb[y * w + x]
                )?;
            }
        }
        svg.push_str("</g><clipPath id='zoom'><rect x='1040' y='155' width='440' height='440'/></clipPath></defs><rect width='100%' height='100%' fill='#171b24'/><g font-family='sans-serif' fill='white'>");
        write!(svg,"<text x='30' y='45' font-size='29'>{}</text><text x='30' y='82' font-size='20'>Original sensor coordinates · before global affine stabilization · 5× slower review</text><text x='1560' y='44' text-anchor='end' font-family='monospace' font-size='25'>Source {}   +{:06.1} ms</text>",if expanded{"Outer bands and iris motion"}else{"Motion of the 25-patch group"},f["sequence"],elapsed)?;
        let iris_count = selected
            .iter()
            .filter(|t| t["frames"][j]["accepted"] == true)
            .count();
        write!(svg,"<text x='30' y='129' font-size='22'>{}</text><text x='1035' y='129' font-size='22'>Iris / reflection · {iris_count}/{} {}</text>",if expanded{"Outer reference: top and bottom 12.5% only"}else{"Original image"},selected.len(),if j==source{"source seeds"}else{"matches"})?;
        write!(svg,"<g transform='translate(30,155) scale(2)'><use xlink:href='#raw'/></g><g clip-path='url(#zoom)'><g transform='translate({},{}) scale({zoom})'><use xlink:href='#raw'/></g></g>",1040.+(sx as f64-crop[0])*zoom,155.+(sy as f64-crop[1])*zoom)?;
        if expanded {
            for by in [155., 155. + h as f64 * 2. * 0.875] {
                write!(svg,"<rect x='30' y='{by}' width='{}' height='{}' fill='#ffb15c' fill-opacity='.06' stroke='#ffb15c' stroke-dasharray='6 5'/>",w*2,h as f64*0.25)?;
            }
            for t in outer {
                let observed = t["frames"][j]["accepted"] == true;
                let origin = xy(&t["source_sensor"]);
                let p = if observed {
                    xy(&t["frames"][j]["sensor"])
                } else {
                    origin
                };
                let q = [
                    30. + 2. * (p[0] - sx as f64),
                    155. + 2. * (p[1] - sy as f64),
                ];
                let anchor = [
                    30. + 2. * (origin[0] - sx as f64),
                    155. + 2. * (origin[1] - sy as f64),
                ];
                let top = origin[1] - (sy as f64) < h as f64 * 0.5;
                let label = [anchor[0], anchor[1] + if top { 24. } else { -16. }];
                let color = if observed { "#ffb15c" } else { "#818590" };
                if observed {
                    write!(svg,"<circle cx='{}' cy='{}' r='4' fill='none' stroke='{color}' stroke-width='2'/><path d='M{},{} L{},{}' fill='none' stroke='{color}' stroke-width='1'/>",q[0],q[1],q[0],q[1],label[0],label[1]-5.)?;
                } else {
                    write!(
                        svg,
                        "<path d='M{},{} l6,6 m-6,0 l6,-6' stroke='{color}' opacity='.5'/>",
                        q[0] - 3.,
                        q[1] - 3.
                    )?;
                }
                write!(svg,"<text x='{}' y='{}' text-anchor='middle' font-size='13' fill='{color}' stroke='#171b24' stroke-width='3' paint-order='stroke'>G{}</text>",label[0],label[1],t["id"])?;
            }
        }
        for (i, t) in selected.iter().enumerate() {
            let accepted = t["frames"][j]["accepted"] == true;
            let (right, slot, count) = labels[i];
            let ly = 178. + slot as f64 * 390. / (count - 1).max(1) as f64;
            let (lx, knee) = if right { (1540., 1500.) } else { (981., 1020.) };
            if accepted {
                let p = xy(&t["frames"][j]["sensor"]);
                let (px, py) = (
                    1040. + (p[0] - crop[0]) * zoom,
                    155. + (p[1] - crop[1]) * zoom,
                );
                write!(svg,"<path d='M{px:.2},{py:.2} L{knee:.2},{py:.2} L{knee:.2},{ly:.2} L{lx:.2},{ly:.2}' fill='none' stroke='#62f4d3' stroke-width='1' opacity='.6'/><circle cx='{px:.2}' cy='{py:.2}' r='4.5' fill='none' stroke='#62f4d3' stroke-width='1.8'/>")?;
            }
            write!(svg,"<rect x='{}' y='{}' width='34' height='25' rx='6' fill='{}'/><text x='{lx}' y='{}' font-size='17' text-anchor='middle' fill='{}'>{}</text>",lx-17.,ly-13.,if accepted{"#213b3b"}else{"#262b35"},ly+6.,if accepted{"#81ffe0"}else{"#727985"},t["id"])?;
        }
        // Both source-to-current 2D fits are evaluated at the same image center.
        // The original single-group mode continues to plot measured centroids.
        let plot = |p: P| [1260. + gain * p[0], 715. + gain * p[1]];
        write!(
            svg,
            "<text x='1030' y='638' font-size='19'>{} · {gain:.1}×</text>",
            if expanded {
                "2D motion at image center"
            } else {
                "Group center path"
            }
        )?;
        for (pi, (label, color, path)) in paths.iter().enumerate() {
            let mut d = String::new();
            let mut connected = false;
            for &p in path {
                if let Some(p) = p {
                    let q = plot(p);
                    write!(
                        d,
                        "{}{:0.2},{:0.2} ",
                        if connected { "L" } else { "M" },
                        q[0],
                        q[1]
                    )?;
                    connected = true;
                } else {
                    connected = false;
                }
            }
            write!(
                svg,
                "<path d='{d}' fill='none' stroke='{color}' stroke-opacity='.6' stroke-width='2'/>"
            )?;
            for (k, p) in path.iter().enumerate() {
                let Some(p) = p else {
                    continue;
                };
                let q = plot(*p);
                write!(
                    svg,
                    "<circle cx='{}' cy='{}' r='{}' fill='{}'/>",
                    q[0],
                    q[1],
                    if k == j { 7 } else { 3 },
                    if k == j { color } else { "#879098" }
                )?;
            }
            if let Some(delta) = path[j] {
                write!(svg,"<text x='1010' y='{}' font-family='monospace' fill='{color}' font-size='17'>{label:5} dx {:+6.2}px  dy {:+6.2}px</text>",790+pi*25,delta[0],delta[1])?;
            } else {
                write!(svg,"<text x='1010' y='{}' font-family='monospace' fill='{color}' font-size='17'>{label:5} insufficient current matches</text>",790+pi*25)?;
            }
        }
        if expanded {
            let count = outer
                .iter()
                .filter(|t| t["frames"][j]["accepted"] == true)
                .count();
            write!(svg,"<text x='30' y='749' font-size='20' fill='#ffb15c'>Outer: {count}/{} {} · orange circles / gray crosses</text><text x='30' y='781' font-size='18'>Middle 75% excluded from every outer matching patch.</text><text x='30' y='810' font-size='18'>Fixed labels; gray marks remain at their source location when unmatched.</text>",outer.len(),if j==source{"source seeds"}else{"current matches"})?;
        } else {
            svg.push_str("<text x='30' y='762' font-size='21'>Fixed labels and crop throughout the sequence.</text><text x='30' y='797' font-size='20'>Actual recorded exposures; points are not a simulated 3D solve.</text>");
        }
        for k in 0..frames.len() {
            write!(
                svg,
                "<rect x='{}' y='841' width='{}' height='6' rx='3' fill='{}'/>",
                30. + k as f64 * 400. / frames.len() as f64,
                (400. / frames.len() as f64 - 2.).max(1.),
                if k == j { "#62f4d3" } else { "#424b57" }
            )?;
        }
        write!(svg,"<text x='465' y='850' font-size='18'>Sources {}–{} · {} exposures · {duration:.3} seconds recorded</text>",frames[0]["sequence"],frames.last().unwrap()["sequence"],frames.len())?;
        if expanded {
            crate::motion_vectors::panels(&mut svg, &motion["frames"][j])?;
        }
        svg.push_str("</g></svg>");
        fs::write(out.join(format!("group-frame-{j}.svg")), svg)?;
        let dt = if j + 1 < frames.len() {
            (uint(&frames[j + 1]["input"]["frame"]["timestamp_ns"]) - timestamp) as f64 / 1e9
        } else {
            0.098
        };
        timing.push(json!({"index":j,"sequence":f["sequence"],"elapsed_ms":elapsed,"source_duration_s":dt,"slow_duration_s":5.*dt}));
    }
    fs::write(
        out.join("animation.json"),
        serde_json::to_vec_pretty(
            &json!({"frames":timing,"group_ids":ids,"outer_ids":outer.iter().map(|t|t["id"].clone()).collect::<Vec<_>>(),"motion_diagnostics":motion,"display":"original unwarped sensor coordinates; shared RAW color preview; 5x slow review","motion_plot_pixels_per_native_pixel":gain,"centroid_source_sensor":means,"note":"final exposure held for 98ms source-equivalent; replay returns to first exposure; no interpolated RAW frames. 3D candidate vectors are conditional fits, not a metric trajectory."}),
        )?,
    )?;
    Ok(())
}
