//! Future-exposure corroboration for tensor partitions, grouped by commanded
//! target visit. Targets never choose a partition or fit its motion.
use super::{json, quantiles, read_rows, Error, Value};
use std::{collections::BTreeMap, fs, path::Path};
type P = [f64; 2];
fn point(v: &Value) -> P {
    [v[0].as_f64().unwrap(), v[1].as_f64().unwrap()]
}
fn distance(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn points(r: &Value) -> Vec<&Value> {
    if let Some(points) = r["tensor_points"].as_array() {
        points.iter().collect()
    } else {
        r["tensor_clusters"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|c| c["members"].as_array().unwrap())
            .collect()
    }
}

fn observations(r: &Value) -> BTreeMap<u64, P> {
    points(r)
        .into_iter()
        .filter(|p| p["normal_flow"] != true)
        .map(|p| (p["id"].as_u64().unwrap(), point(&p["current_sensor"])))
        .collect()
}

// Similarity fit of independent witnesses in sensor coordinates. The held
// feature is excluded from means, tensor estimation and conditioning checks.
pub(super) fn predict(pairs: &[(P, P)], held: P) -> Option<P> {
    if pairs.len() < 3 {
        return None;
    }
    let n = pairs.len() as f64;
    let a = std::array::from_fn::<_, 2, _>(|j| pairs.iter().map(|p| p.0[j]).sum::<f64>() / n);
    let b = std::array::from_fn::<_, 2, _>(|j| pairs.iter().map(|p| p.1[j]).sum::<f64>() / n);
    let mut norm = 0.;
    let mut dot = 0.;
    let mut cross = 0.;
    for &(x, y) in pairs {
        let x = [x[0] - a[0], x[1] - a[1]];
        let y = [y[0] - b[0], y[1] - b[1]];
        norm += x[0] * x[0] + x[1] * x[1];
        dot += x[0] * y[0] + x[1] * y[1];
        cross += x[0] * y[1] - x[1] * y[0];
    }
    if norm / n < 64. {
        return None;
    }
    let (scale, rotation) = (dot / norm, cross / norm);
    let x = [held[0] - a[0], held[1] - a[1]];
    Some([
        b[0] + scale * x[0] - rotation * x[1],
        b[1] + rotation * x[0] + scale * x[1],
    ])
}

#[derive(Clone)]
struct Prediction {
    p: P,
    q: P,
    error: f64,
    common_error: f64,
    visit: Value,
    cluster: usize,
}

fn evaluate(rows: &[Value]) -> BTreeMap<(u64, u64, u64), Prediction> {
    let mut result = BTreeMap::new();
    for eye in [1, 2] {
        let eye_rows = rows
            .iter()
            .filter(|r| r["source"]["eye_id"] == eye)
            .collect::<Vec<_>>();
        for pair in eye_rows.windows(2) {
            let (now, next) = (pair[0], pair[1]);
            let t = now["source"]["timestamp_ns"].as_u64().unwrap();
            let u = next["source"]["timestamp_ns"].as_u64().unwrap();
            if next["reset"] == true || u <= t || u - t > 250_000_000 {
                continue;
            }
            let current = observations(now);
            let future = observations(next);
            let adjacent = points(next)
                .into_iter()
                .map(|p| (p["id"].as_u64().unwrap(), p))
                .collect::<BTreeMap<_, _>>();
            let joined = current
                .iter()
                .filter_map(|(&id, &p)| {
                    let m = adjacent.get(&id)?;
                    // A recovered old ID is not an adjacent tissue track.
                    if m["previous_timestamp_ns"].as_u64().is_some_and(|s| s != t)
                        || distance(point(&m["previous_sensor"]), p) > 0.01
                    {
                        return None;
                    }
                    Some((id, (p, *future.get(&id)?)))
                })
                .collect::<BTreeMap<_, _>>();
            for (ci, c) in now["tensor_clusters"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                if c["persistent_nodes"].as_u64().unwrap() < 4 {
                    continue;
                }
                let ids = c["members"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|m| m["normal_flow"] != true)
                    .filter_map(|m| m["id"].as_u64())
                    .filter(|id| joined.contains_key(id))
                    .collect::<Vec<_>>();
                if ids.len() < 4 {
                    continue;
                }
                for &id in &ids {
                    let (p, q) = joined[&id];
                    let witnesses = ids
                        .iter()
                        .filter(|&&other| other != id)
                        .map(|other| joined[other])
                        .collect::<Vec<_>>();
                    let common = joined
                        .iter()
                        .filter(|(other, _)| **other != id)
                        .map(|(_, pq)| *pq)
                        .collect::<Vec<_>>();
                    if let (Some(estimate), Some(reference)) =
                        (predict(&witnesses, p), predict(&common, p))
                    {
                        result.insert(
                            (eye, t, id),
                            Prediction {
                                p,
                                q,
                                error: distance(estimate, q),
                                common_error: distance(reference, q),
                                visit: now["target"].clone(),
                                cluster: ci,
                            },
                        );
                    }
                }
            }
        }
    }
    result
}

fn inventory(rows: &[Value]) -> Value {
    let mut target_counts = BTreeMap::<String, usize>::new();
    let mut track_visits = BTreeMap::<(u64, u64, u64), std::collections::BTreeSet<u64>>::new();
    let mut sizes = Vec::new();
    let mut persistent = 0;
    let mut members = 0;
    for r in rows {
        if r["target"]["stable_in_sensitivity_window"] == true {
            *target_counts
                .entry(r["target"]["target"].to_string())
                .or_default() += 1;
        }
        for c in r["tensor_clusters"].as_array().unwrap() {
            let m = c["members"].as_array().unwrap();
            members += m.len();
            sizes.push(m.len() as f64);
            persistent += usize::from(c["persistent_nodes"].as_u64().unwrap() >= 4);
            for p in m {
                let key = (
                    r["source"]["eye_id"].as_u64().unwrap(),
                    r["tracking_epoch"].as_u64().unwrap(),
                    p["id"].as_u64().unwrap(),
                );
                let seen = track_visits.entry(key).or_default();
                if r["target"]["stable_in_sensitivity_window"] == true {
                    seen.insert(r["target"]["visit"].as_u64().unwrap());
                }
            }
        }
    }
    let mut lengths = BTreeMap::<usize, usize>::new();
    for seen in track_visits.values() {
        *lengths.entry(seen.len()).or_default() += 1;
    }
    json!({"frames":rows.len(),"components":sizes.len(),"persistent_components":persistent,"component_size":quantiles(sizes),
        "member_observations":members,"settled_target_frames":target_counts,"feature_target_visit_counts":lengths,
        "relation_ms":quantiles(rows.iter().map(|r|r["stages_ms"]["relation"].as_f64().unwrap()).collect())})
}

pub fn run(args: &[String]) -> Result<(), Error> {
    if args.len() != 5 {
        return Err("usage: --cohorts BASELINE_DIR CANDIDATE_DIR NEW_REPORT".into());
    }
    let a = read_rows(&Path::new(&args[2]).join("frames.jsonl"))?;
    let b = read_rows(&Path::new(&args[3]).join("frames.jsonl"))?;
    if a.len() != b.len()
        || a.iter()
            .zip(&b)
            .any(|(x, y)| x["source"] != y["source"] || x["raw_sha256"] != y["raw_sha256"])
    {
        return Err("source identity mismatch".into());
    }
    let out = Path::new(&args[4]);
    if out.exists()
        || !out
            .parent()
            .ok_or("report needs parent")?
            .canonicalize()?
            .starts_with(fs::canonicalize("outputs")?)
    {
        return Err("use new checked output".into());
    }
    let ea = evaluate(&a);
    let eb = evaluate(&b);
    let mut joined = Vec::new();
    let mut changed_correspondences = 0;
    for (key, x) in &ea {
        if let Some(y) = eb.get(key) {
            if distance(x.p, y.p) > 0.01 || distance(x.q, y.q) > 0.01 {
                changed_correspondences += 1;
                continue;
            }
            joined.push(json!({"eye":key.0,"source_ns":key.1,"id":key.2,"target":x.visit,"point_sensor":x.p,
                "baseline_error_px":x.error,"candidate_error_px":y.error,"baseline_common_error_px":x.common_error,
                "candidate_common_error_px":y.common_error,"baseline_cluster":x.cluster,"candidate_cluster":y.cluster}));
        }
    }
    let errors = |key: &str, settled: bool| {
        quantiles(
            joined
                .iter()
                .filter(|r| !settled || r["target"]["stable_in_sensitivity_window"] == true)
                .map(|r| r[key].as_f64().unwrap())
                .collect(),
        )
    };
    let full_correspondences = a.iter().chain(&b).all(|r| r["tensor_points"].is_array());
    let report = json!({"schema":"buttercup-tensor-cohort-eval-v2","baseline":inventory(&a),"candidate":inventory(&b),
        "future_membership_censors_availability":!full_correspondences,
        "baseline_queries":ea.len(),"candidate_queries":eb.len(),"common_exact_geometry_queries":joined.len(),
        "id_collisions_or_changed_matches_excluded":changed_correspondences,
        "common_errors":{"baseline":errors("baseline_error_px",false),"candidate":errors("candidate_error_px",false),
            "settled_baseline":errors("baseline_error_px",true),"settled_candidate":errors("candidate_error_px",true)},
        "queries":joined,"scope":"Previous-exposure partition predicts a held feature in the next source pair using at least three other cohort witnesses. Common queries require matching source, ID and both endpoints within 0.01px; track continuation requires next previous endpoint to match now. All adjacent matches supply future observations when tensor_points is present, otherwise legacy graph membership censors availability. Missing/changed correspondences excluded, not zero error. Target association is weak commanded-position evidence with unknown capture latency; not fixation or tissue ground truth. No anatomical accuracy claim from this statistic alone."});
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", report["common_errors"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_witnesses_recover_translation_rotation_and_scale() {
        let transform = |p: P| {
            [
                100. + 1.04 * p[0] - 0.15 * p[1],
                -23. + 0.15 * p[0] + 1.04 * p[1],
            ]
        };
        let pairs = [[-20., -10.], [20., -10.], [0., 30.]].map(|p| (p, transform(p)));
        let query = [62., 41.];
        assert!(distance(predict(&pairs, query).unwrap(), transform(query)) < 1e-10);
        assert!(predict(&pairs[..2], query).is_none());
        let compact = [[0., 0.], [1., 0.], [0., 1.]].map(|p| (p, transform(p)));
        assert!(
            predict(&compact, query).is_none(),
            "unconditioned witnesses cannot authorize extrapolation"
        );
    }

    #[test]
    fn next_partition_cannot_hide_a_matched_outlier() {
        let make = |time: u64, dx: f64| {
            let points = [[0., 0.], [20., 0.], [0., 20.], [20., 20.]]
                .into_iter()
                .enumerate()
                .map(|(id, p)| {
                    json!({"id":id,"previous_sensor":p,"current_sensor":[p[0]+dx,p[1]],
                    "normal_flow":false})
                })
                .collect::<Vec<_>>();
            json!({"source":{"eye_id":1,"timestamp_ns":time},"tensor_points":points,
                "tensor_clusters":[{"persistent_nodes":4,"members":points}],"target":null})
        };
        let now = make(1_000_000_000, 0.);
        let mut next = make(1_100_000_000, 2.);
        next["tensor_clusters"] = json!([]);
        next["tensor_points"][3]["current_sensor"] = json!([28., 20.]);
        let evaluated = evaluate(&[now.clone(), next.clone()]);
        assert_eq!(evaluated.len(), 4);
        assert!((evaluated[&(1, 1_000_000_000, 3)].error - 6.).abs() < 1e-9);
        next["tensor_points"][3]["previous_timestamp_ns"] = json!(900_000_000u64);
        assert!(
            evaluate(&[now, next]).is_empty(),
            "a recovered ID is not a fourth adjacent witness"
        );
    }
}
