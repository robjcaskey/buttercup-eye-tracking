//! Frozen-correspondence experiment: subtract an independent image carrier
//! before looking for differential populations. Targets never enter selection.
use super::{cohorts, json, point, ppm, quantiles, read_rows, Error, Value};
use std::{collections::BTreeSet, fs, io::Write, path::Path, time::Instant};
type P = [f64; 2];

fn xy(v: &Value) -> P {
    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]
}
fn dist(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn carrier_predict(c: &Value, p: P) -> P {
    let center = xy(&c["center_sensor"]);
    let t = c["candidate_tensor"].as_array().unwrap();
    let t = t.iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
    let (x, y) = (p[0] - center[0], p[1] - center[1]);
    [
        p[0] + t[0] + t[2] * x - t[3] * y,
        p[1] + t[1] + t[3] * x + t[2] * y,
    ]
}

fn pair_predict(a: (P, P), b: (P, P), p: P) -> Option<P> {
    let x = [b.0[0] - a.0[0], b.0[1] - a.0[1]];
    let y = [b.1[0] - a.1[0], b.1[1] - a.1[1]];
    let n = x[0] * x[0] + x[1] * x[1];
    if n < 144. {
        return None;
    }
    let s = (x[0] * y[0] + x[1] * y[1]) / n;
    let r = (x[0] * y[1] - x[1] * y[0]) / n;
    if (s - 1.).abs() > 0.12 || r.abs() > 0.12 {
        return None;
    }
    let q = [p[0] - a.0[0], p[1] - a.0[1]];
    Some([a.1[0] + s * q[0] - r * q[1], a.1[1] + r * q[0] + s * q[1]])
}

/// Deterministic two-witness proposals, corroborated by at least two more
/// points, followed by two bounded least-squares consensus updates.
fn differential_groups(pairs: &[(P, P)], available: &mut [bool], radius: f64) -> Vec<Vec<usize>> {
    let mut groups = Vec::new();
    while groups.len() < 3 {
        let ids = (0..pairs.len())
            .filter(|&i| available[i])
            .collect::<Vec<_>>();
        if ids.len() < 4 {
            break;
        }
        let mut best = Vec::new();
        let mut best_score = 0.;
        for (ai, &a) in ids.iter().enumerate() {
            for &b in &ids[ai + 1..] {
                let Some(_) = pair_predict(pairs[a], pairs[b], pairs[a].0) else {
                    continue;
                };
                let mut inliers = Vec::new();
                let mut error = 0.;
                for &i in &ids {
                    let e = dist(
                        pair_predict(pairs[a], pairs[b], pairs[i].0).unwrap(),
                        pairs[i].1,
                    );
                    if e <= radius {
                        inliers.push(i);
                        error += e * e;
                    }
                }
                if inliers.len() < 4 {
                    continue;
                }
                let score = inliers.len() as f64 - error / (radius * radius);
                if score > best_score {
                    best_score = score;
                    best = inliers;
                }
            }
        }
        if best.len() < 4 {
            break;
        }
        for _ in 0..2 {
            let witnesses = best.iter().map(|&i| pairs[i]).collect::<Vec<_>>();
            let next = ids
                .iter()
                .copied()
                .filter(|&i| {
                    cohorts::predict(&witnesses, pairs[i].0)
                        .is_some_and(|p| dist(p, pairs[i].1) <= radius)
                })
                .collect::<Vec<_>>();
            if next.len() < best.len() || next == best {
                break;
            }
            best = next;
        }
        for &i in &best {
            available[i] = false;
        }
        groups.push(best);
    }
    groups
}

fn make_cluster(
    points: &[&Value],
    indices: &[usize],
    role: &str,
    prior: &[BTreeSet<u64>],
    carrier: &Value,
) -> Value {
    let members = indices
        .iter()
        .map(|&i| points[i].clone())
        .collect::<Vec<_>>();
    let ids = members
        .iter()
        .map(|m| m["id"].as_u64().unwrap())
        .collect::<BTreeSet<_>>();
    let persistent = prior
        .iter()
        .map(|p| ids.intersection(p).count())
        .max()
        .unwrap_or(0);
    let pairs = members
        .iter()
        .map(|m| (xy(&m["previous_sensor"]), xy(&m["current_sensor"])))
        .collect::<Vec<_>>();
    let n = pairs.len() as f64;
    let center = std::array::from_fn::<_, 2, _>(|j| pairs.iter().map(|p| p.0[j]).sum::<f64>() / n);
    let at = cohorts::predict(&pairs, center).unwrap_or(center);
    let at_x = cohorts::predict(&pairs, [center[0] + 1., center[1]])
        .unwrap_or([center[0] + 1., center[1]]);
    let errors = pairs
        .iter()
        .filter_map(|p| Some(dist(cohorts::predict(&pairs, p.0)?, p.1)))
        .collect::<Vec<_>>();
    let mean = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
    let differential = pairs
        .iter()
        .map(|&(p, q)| dist(carrier_predict(carrier, p), q))
        .collect::<Vec<_>>();
    json!({"members":members,"center_sensor":center,
        "tensor":[at[0]-center[0],at[1]-center[1],at_x[0]-at[0]-1.,at_x[1]-at[1]],
        "residual":mean,"coherence":(-mean).exp(),"persistent_nodes":persistent,
        "persistent_edges":0,"selected_iris":false,"motion_role":role,
        "carrier_error_px":quantiles(differential),
        "identity":"anonymous motion population; carrier-compatible does not mean proven skin; differential does not mean proven sclera/iris"})
}

fn raw_ppm(path: &Path, w: usize, h: usize) -> Result<Vec<u32>, Error> {
    let bytes = fs::read(path)?;
    let header = format!("P6\n{w} {h}\n255\n");
    if !bytes.starts_with(header.as_bytes()) || bytes.len() != header.len() + w * h * 3 {
        return Err("source RAW preview geometry mismatch".into());
    }
    Ok(bytes[header.len()..]
        .chunks_exact(3)
        .map(|p| (u32::from(p[0]) << 16) | (u32::from(p[1]) << 8) | u32::from(p[2]))
        .collect())
}

pub fn run(args: &[String]) -> Result<(), Error> {
    if !(4..=5).contains(&args.len()) {
        return Err(
            "usage: --partition BASELINE_DIR NEW_OUTPUT [CARRIER_RADIUS_MULTIPLIER=1.5]".into(),
        );
    }
    let input = Path::new(&args[2]);
    let out = Path::new(&args[3]);
    let multiplier = args
        .get(4)
        .map(|s| s.parse::<f64>())
        .transpose()?
        .unwrap_or(1.5);
    if !(0.75..=3.).contains(&multiplier)
        || out.exists()
        || !out
            .parent()
            .ok_or("output needs parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("new checked output and finite multiplier in 0.75..=3 required".into());
    }
    let rows = read_rows(&input.join("frames.jsonl"))?;
    if rows
        .iter()
        .any(|r| !r["tensor_points"].is_array() || !r["outer_band_carrier"].is_object())
    {
        return Err("full native correspondences and outer-band carrier required".into());
    }
    fs::create_dir(out)?;
    let mut log = fs::File::create(out.join("frames.jsonl"))?;
    let mut previous: [Option<u64>; 2] = [None, None];
    let mut cohorts: [Vec<BTreeSet<u64>>; 2] = Default::default();
    let mut counts = [0usize; 2];
    let mut available_frames = 0;
    let mut group_count = [0usize; 2];
    let mut costs = Vec::new();
    for mut row in rows {
        let eye = row["source"]["eye_id"].as_u64().unwrap() as usize - 1;
        let stamp = row["source"]["timestamp_ns"].as_u64().unwrap();
        let c = &row["outer_band_carrier"];
        let contiguous = row["reset"] != true
            && previous[eye].is_some_and(|t| {
                stamp > t && stamp - t <= 250_000_000 && c["previous_timestamp_ns"] == t
            });
        if !contiguous {
            cohorts[eye].clear();
        }
        let started = Instant::now();
        let points = row["tensor_points"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| {
                p["normal_flow"] != true
                    && p["consecutive_matches_before"].as_u64().unwrap_or(0) >= 2
                    && p["score"].as_f64().unwrap_or(0.) >= 0.2
            })
            .collect::<Vec<_>>();
        let pairs = points
            .iter()
            .map(|p| (xy(&p["previous_sensor"]), xy(&p["current_sensor"])))
            .collect::<Vec<_>>();
        let mut clusters = Vec::new();
        if contiguous && c["reliable"] == true {
            available_frames += 1;
            let radius = (multiplier * c["residual"].as_f64().unwrap()).max(0.75);
            let mut unused = vec![true; points.len()];
            let compatible = pairs
                .iter()
                .enumerate()
                .filter_map(|(i, &(p, q))| (dist(carrier_predict(c, p), q) <= radius).then_some(i))
                .collect::<Vec<_>>();
            if compatible.len() >= 4 {
                for &i in &compatible {
                    unused[i] = false;
                }
                clusters.push(make_cluster(
                    &points,
                    &compatible,
                    "carrier_compatible",
                    &cohorts[eye],
                    c,
                ));
                group_count[0] += 1;
            }
            for g in differential_groups(&pairs, &mut unused, 0.60) {
                clusters.push(make_cluster(&points, &g, "differential", &cohorts[eye], c));
                group_count[1] += 1;
            }
        }
        let elapsed = started.elapsed().as_secs_f64() * 1000.;
        costs.push(elapsed);
        cohorts[eye] = clusters
            .iter()
            .map(|c| {
                c["members"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p["id"].as_u64().unwrap())
                    .collect()
            })
            .collect();
        previous[eye] = Some(stamp);
        row["tensor_clusters"] = json!(clusters);
        row["stages_ms"]["relation"] = json!(elapsed);
        row["fit"] = Value::Null;
        row["rejection"] =
            json!("frozen-correspondence partition experiment; no ellipse or 3D solve run");
        row["partition_experiment"] = json!({"carrier_radius_multiplier":multiplier,"differential_radius_px":0.60,
            "target_feedback":false,"baseline_correspondences_frozen":true,"tissue_labels":false,
            "persistence":"intersection of at least four current track IDs with one preceding anonymous cohort; not anatomical identity"});
        let src = &row["source"];
        let (w, h) = (
            src["width"].as_u64().unwrap() as usize,
            src["height"].as_u64().unwrap() as usize,
        );
        let name = format!("eye-{}-{:04}", eye + 1, counts[eye]);
        let raw = input.join(format!("{name}-raw.ppm"));
        if counts[eye] % 8 == 0 && raw.exists() {
            let mut image = raw_ppm(&raw, w, h)?;
            for (index, c) in row["tensor_clusters"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let color = if c["motion_role"] == "carrier_compatible" {
                    0x40ff80
                } else {
                    [0xff5050, 0xffc040, 0x60a0ff, 0xff50ff][index % 4]
                };
                for m in c["members"].as_array().unwrap() {
                    let p = xy(&m["current_sensor"]);
                    point(
                        &mut image,
                        w,
                        h,
                        (
                            p[0] - src["sensor_x"].as_f64().unwrap(),
                            p[1] - src["sensor_y"].as_f64().unwrap(),
                        ),
                        color,
                        1,
                    );
                }
            }
            ppm(&out.join(format!("{name}-partition.ppm")), &image, w, h)?;
        }
        serde_json::to_writer(&mut log, &row)?;
        writeln!(log)?;
        counts[eye] += 1;
    }
    let report = json!({"input":input,"counts":counts,"carrier_available_frames":available_frames,"groups":group_count,"partition_ms":quantiles(costs),
        "scope":"Frozen native point correspondences; targets reserved for evaluation; no anatomical labels, ellipse fits, measured gaze or live-throughput claim. Skin-band image carrier may include lid/glasses motion. Carrier radius is a heuristic residual multiple, not a probability bound."});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disjoint_motion_populations_need_independent_witnesses() {
        let mut pairs = Vec::new();
        for (offset, movement) in [(0., [3., 1.]), (90., [-4., 2.])] {
            for p in [[0., 0.], [20., 0.], [0., 20.], [20., 20.], [10., 32.]] {
                let p = [p[0] + offset, p[1]];
                pairs.push((p, [p[0] + movement[0], p[1] + movement[1]]));
            }
        }
        let groups = differential_groups(&pairs, &mut vec![true; pairs.len()], 0.6);
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|g| g.len() == 5));
        assert!(groups
            .iter()
            .all(|g| g.iter().all(|&i| i < 5) || g.iter().all(|&i| i >= 5)));
        assert!(differential_groups(&pairs[..3], &mut [true; 3], 0.6).is_empty());
    }
    #[test]
    fn carrier_prediction_includes_scale_rotation_and_sensor_origin() {
        let c = json!({"center_sensor":[3000.,1500.],"candidate_tensor":[4.,-2.,0.02,0.03]});
        assert!(dist(carrier_predict(&c, [3020., 1540.]), [3023.2, 1539.4]) < 1e-9);
    }
}
