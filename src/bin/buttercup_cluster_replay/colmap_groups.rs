//! Bounded offline grouping of measured 2-D COLMAP tracks, never the rigid 3-D
//! playback. Repeated, independently supported similarity tensors vote for or
//! against common membership. Anonymous candidate groups are not anatomical labels.
//! Even acquisition frames discover groups; odd frames check predictions.
use super::{json, quantiles, Error, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};
type P = [f64; 2];
type Frame = BTreeMap<usize, P>;
const LAGS: [usize; 5] = [4, 12, 24, 48, 96];
const MIN_GROUP: usize = 6;

#[derive(Clone, Copy)]
struct Match {
    id: usize,
    p: P,
    q: P,
}
#[derive(Clone, Copy, Debug)]
struct Similarity {
    a: P,
    b: P,
    s: f64,
    r: f64,
}
fn distance(a: P, b: P) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
impl Similarity {
    // Same centered translation/rotation/scale tensor as cohorts::predict.
    // No iris center, image brightness or spatial anatomical region is used.
    fn fit(v: &[Match]) -> Option<Self> {
        if v.len() < 2 {
            return None;
        }
        let n = v.len() as f64;
        let a = std::array::from_fn(|k| v.iter().map(|m| m.p[k]).sum::<f64>() / n);
        let b = std::array::from_fn(|k| v.iter().map(|m| m.q[k]).sum::<f64>() / n);
        let (mut norm, mut dot, mut cross) = (0., 0., 0.);
        for m in v {
            let x = [m.p[0] - a[0], m.p[1] - a[1]];
            let y = [m.q[0] - b[0], m.q[1] - b[1]];
            norm += x[0] * x[0] + x[1] * x[1];
            dot += x[0] * y[0] + x[1] * y[1];
            cross += x[0] * y[1] - x[1] * y[0];
        }
        if norm / n < 64. {
            return None;
        }
        let (s, r) = (dot / norm, cross / norm);
        (s.is_finite() && r.is_finite() && (0.5..=1.6).contains(&s.hypot(r))).then_some(Self {
            a,
            b,
            s,
            r,
        })
    }
    fn predict(self, p: P) -> P {
        let x = p[0] - self.a[0];
        let y = p[1] - self.a[1];
        [
            self.b[0] + self.s * x - self.r * y,
            self.b[1] + self.r * x + self.s * y,
        ]
    }
    fn error(self, m: &Match) -> f64 {
        distance(self.predict(m.p), m.q)
    }
}
fn random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn robust(v: &[Match], radius: f64, minimum: usize, seed: u64) -> Option<(Similarity, Vec<usize>)> {
    if v.len() < minimum {
        return None;
    }
    let mut state = seed | 1;
    let mut best = None;
    let mut best_score = 0.;
    let trials = (v.len() * (v.len() - 1) / 2).min(900);
    for _ in 0..trials {
        let a = random(&mut state) as usize % v.len();
        let b = random(&mut state) as usize % v.len();
        if a == b || distance(v[a].p, v[b].p) < 16. {
            continue;
        }
        let Some(model) = Similarity::fit(&[v[a], v[b]]) else {
            continue;
        };
        let score = v
            .iter()
            .map(|m| (1. - (model.error(m) / radius).powi(2)).max(0.))
            .sum::<f64>();
        if score > best_score {
            best_score = score;
            best = Some(model);
        }
    }
    let mut model = best?;
    for _ in 0..3 {
        let inside = v
            .iter()
            .copied()
            .filter(|m| model.error(m) <= radius)
            .collect::<Vec<_>>();
        if inside.len() < minimum {
            return None;
        }
        model = Similarity::fit(&inside)?;
    }
    let inside = v
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (model.error(m) <= radius).then_some(i))
        .collect::<Vec<_>>();
    (inside.len() >= minimum).then_some((model, inside))
}
fn partition(v: &[Match], radius: f64, seed: u64) -> Vec<Option<usize>> {
    let mut available = v.to_vec();
    let mut models = Vec::new();
    for k in 0..4 {
        let Some((model, inside)) = robust(&available, radius, MIN_GROUP, seed + k * 719) else {
            break;
        };
        let mut used = vec![false; available.len()];
        for i in inside {
            used[i] = true;
        }
        available = available
            .iter()
            .enumerate()
            .filter_map(|(i, m)| (!used[i]).then_some(*m))
            .collect();
        models.push(model);
    }
    if models.len() < 2 {
        return vec![None; v.len()];
    }
    let classify = |m: &Match, models: &[Similarity]| {
        let mut e = models
            .iter()
            .enumerate()
            .map(|(k, g)| (k, g.error(m)))
            .collect::<Vec<_>>();
        e.sort_by(|a, b| a.1.total_cmp(&b.1));
        (e[0].1 <= radius && e[1].1 - e[0].1 >= radius * 0.6).then_some(e[0].0)
    };
    // Refit simultaneously so a large first consensus cannot permanently
    // swallow boundary points. Ambiguous points cast no membership vote.
    for _ in 0..3 {
        let assignments = v.iter().map(|m| classify(m, &models)).collect::<Vec<_>>();
        for (k, model) in models.iter_mut().enumerate() {
            let support = v
                .iter()
                .zip(&assignments)
                .filter_map(|(m, g)| (*g == Some(k)).then_some(*m))
                .collect::<Vec<_>>();
            if support.len() >= MIN_GROUP {
                if let Some(next) = Similarity::fit(&support) {
                    *model = next;
                }
            }
        }
    }
    let mut assignments = v.iter().map(|m| classify(m, &models)).collect::<Vec<_>>();
    let counts = (0..models.len())
        .map(|k| assignments.iter().filter(|g| **g == Some(k)).count())
        .collect::<Vec<_>>();
    for a in &mut assignments {
        if a.is_some_and(|k| counts[k] < MIN_GROUP) {
            *a = None;
        }
    }
    if counts.iter().filter(|&&n| n >= MIN_GROUP).count() < 2 {
        assignments.fill(None);
    }
    assignments
}
fn joined(a: &Frame, b: &Frame) -> Vec<Match> {
    a.iter()
        .filter_map(|(&id, &p)| {
            Some(Match {
                id,
                p,
                q: *b.get(&id)?,
            })
        })
        .collect()
}
#[derive(Clone, Copy, Default)]
struct Votes {
    together: u32,
    apart: u32,
}
fn root(parent: &[usize], mut p: usize) -> usize {
    while parent[p] != p {
        p = parent[p];
    }
    p
}
struct Groups {
    labels: Vec<usize>,
    support: Vec<usize>,
    report: Value,
}
fn discover(frames: &[Frame], count: usize, radius: f64) -> Groups {
    let mut votes = BTreeMap::<(usize, usize), Votes>::new();
    let mut support = vec![0; count];
    let mut observed = vec![0; count];
    let mut pairs = 0;
    let mut distinct = 0;
    for a in (0..frames.len()).step_by(2) {
        for &id in frames[a].keys() {
            observed[id] += 1;
        }
        for lag in LAGS {
            let b = a + lag;
            if b >= frames.len() {
                continue;
            }
            let v = joined(&frames[a], &frames[b]);
            if v.len() < MIN_GROUP * 2 {
                continue;
            }
            pairs += 1;
            let g = partition(&v, radius, (a * 733 + b * 31 + 1) as u64);
            if g.iter().all(Option::is_none) {
                continue;
            }
            distinct += 1;
            for (i, m) in v.iter().enumerate() {
                let Some(group) = g[i] else {
                    continue;
                };
                support[m.id] += 1;
                for j in i + 1..v.len() {
                    let Some(other) = g[j] else {
                        continue;
                    };
                    let key = (m.id.min(v[j].id), m.id.max(v[j].id));
                    let vote = votes.entry(key).or_default();
                    if group == other {
                        vote.together += 1;
                    } else {
                        vote.apart += 1;
                    }
                }
            }
        }
    }
    let eligible = (0..count)
        .map(|i| observed[i] >= 4 && support[i] >= 3)
        .collect::<Vec<_>>();
    let mut neighbors = vec![Vec::new(); count];
    for (&(a, b), v) in &votes {
        neighbors[a].push((b, *v));
        neighbors[b].push((a, *v));
    }
    let mut edges = votes
        .iter()
        .filter(|((a, b), v)| {
            eligible[*a] && eligible[*b] && v.together >= 3 && v.together >= 3 * v.apart
        })
        .map(|(&(a, b), v)| (a, b, v.together as f64 / (1. + v.apart as f64)))
        .collect::<Vec<_>>();
    edges.sort_by(|a, b| {
        b.2.total_cmp(&a.2)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut parent = (0..count).collect::<Vec<_>>();
    let mut members = (0..count).map(|i| vec![i]).collect::<Vec<_>>();
    for (a, b, _) in edges {
        let (mut a, mut b) = (root(&parent, a), root(&parent, b));
        if a == b {
            continue;
        }
        if members[a].len() > members[b].len() {
            std::mem::swap(&mut a, &mut b);
        }
        let (mut positive, mut negative) = (0u64, 0u64);
        for &i in &members[a] {
            for &(j, v) in &neighbors[i] {
                if root(&parent, j) == b {
                    // Cap each pair's leverage: hundreds of repeated overlapping
                    // intervals must not overwhelm contradictions elsewhere.
                    let weight = 20.0 / (v.together + v.apart).max(20) as f64;
                    positive += (v.together as f64 * weight).round() as u64;
                    negative += (v.apart as f64 * weight).round() as u64;
                }
            }
        }
        if positive < 3 || positive < 4 * negative {
            continue;
        }
        parent[a] = b;
        let old = std::mem::take(&mut members[a]);
        members[b].extend(old);
    }
    let mut components = members
        .into_iter()
        .filter(|m| m.len() >= 8)
        .collect::<Vec<_>>();
    components.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a[0].cmp(&b[0])));
    let discarded = components.len().saturating_sub(8);
    components.truncate(8);
    let mut labels = vec![0; count];
    for (g, ids) in components.iter().enumerate() {
        for &id in ids {
            labels[id] = g + 1;
        }
    }
    // Signed graph refinement lets a track reconsider an early greedy merge.
    // Missing overlap contributes no vote; repeated contradictory motion has
    // double weight. This propagates motion evidence, never image position.
    let mut order = (0..count).filter(|&i| eligible[i]).collect::<Vec<_>>();
    order.sort_by_key(|&i| std::cmp::Reverse(neighbors[i].len()));
    for _ in 0..24 {
        let mut changed = 0;
        for &i in &order {
            let mut p = [0f64; 9];
            let mut n = [0f64; 9];
            for &(j, v) in &neighbors[i] {
                let g = labels[j];
                if g == 0 {
                    continue;
                }
                let weight = 20.0 / (v.together + v.apart).max(20) as f64;
                p[g] += v.together as f64 * weight;
                n[g] += v.apart as f64 * weight;
            }
            let mut scores = (1..=components.len())
                .map(|g| (g, p[g] - 2. * n[g]))
                .collect::<Vec<_>>();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            let next = scores
                .first()
                .filter(|&&(g, s)| {
                    s >= 3. && p[g] >= 3. * n[g] && s - scores.get(1).map_or(0., |v| v.1) > 1.
                })
                .map_or(0, |v| v.0);
            if next != labels[i] {
                labels[i] = next;
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }
    let mut sizes = (1..=components.len())
        .map(|g| (g, labels.iter().filter(|&&v| v == g).count()))
        .filter(|v| v.1 >= 8)
        .collect::<Vec<_>>();
    sizes.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let remap = sizes
        .iter()
        .enumerate()
        .map(|(i, &(g, _))| (g, i + 1))
        .collect::<BTreeMap<_, _>>();
    for g in &mut labels {
        *g = remap.get(g).copied().unwrap_or(0);
    }
    let report = json!({"candidate_pairs":pairs,"pairs_with_separated_consensus":distinct,
        "pairwise_vote_edges":votes.len(),"discarded_small_or_disconnected_groups_over_limit":discarded,
        "group_sizes":sizes.iter().map(|v|v.1).collect::<Vec<_>>(),
        "assigned_points":labels.iter().filter(|&&v|v>0).count(),"total_points":count,
        "radius_px":radius,"lags_frames":LAGS,"group_limit":8,
        "grouping_frames":"even acquisition indices only; odd-frame coordinates withheld",
        "policy":"No forced group count or anatomical prior. At least 4 even-frame observations, 3 discriminating pair votes, 8 members per retained component. Aggregate agreement seeds a signed graph refinement; contradictory votes have double weight; each pair contribution capped at 20."});
    Groups {
        labels,
        support,
        report,
    }
}
fn evaluate(frames: &[Frame], groups: &Groups, radius: f64) -> Value {
    let group_count = groups.labels.iter().copied().max().unwrap_or(0);
    let mut baseline = Vec::new();
    let mut candidate = Vec::new();
    let mut per = vec![Vec::<(f64, f64)>::new(); group_count];
    let (mut possible, mut improved, mut regressed, mut pairs) = (0, 0, 0, 0);
    for a in (1..frames.len()).step_by(4) {
        for lag in LAGS {
            let b = a + lag;
            if b >= frames.len() {
                continue;
            }
            let v = joined(&frames[a], &frames[b]);
            if v.len() < 12 {
                continue;
            }
            for fold in 0..2 {
                let witness = v
                    .iter()
                    .copied()
                    .filter(|m| ((m.id as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 62) % 2 == fold)
                    .collect::<Vec<_>>();
                let Some((common, _)) = robust(
                    &witness,
                    radius,
                    4,
                    (a * 71 + b * 133 + fold as usize + 1) as u64,
                ) else {
                    continue;
                };
                let models = (1..=group_count)
                    .map(|g| {
                        let rows = witness
                            .iter()
                            .copied()
                            .filter(|m| groups.labels[m.id] == g)
                            .collect::<Vec<_>>();
                        robust(&rows, radius, 4, (g * 73 + a * 113 + b + 1) as u64).map(|r| r.0)
                    })
                    .collect::<Vec<_>>();
                pairs += 1;
                for m in &v {
                    if ((m.id as u64).wrapping_mul(0x9e3779b97f4a7c15) >> 62) % 2 == fold {
                        continue;
                    }
                    possible += 1;
                    let g = groups.labels[m.id];
                    if g == 0 {
                        continue;
                    }
                    let Some(model) = models[g - 1] else {
                        continue;
                    };
                    let (base, error) = (common.error(m), model.error(m));
                    baseline.push(base);
                    candidate.push(error);
                    per[g - 1].push((base, error));
                    improved += usize::from(base - error > 0.5);
                    regressed += usize::from(error - base > 0.5);
                }
            }
        }
    }
    json!({"scope":"Within-recording withheld odd-frame coordinates plus disjoint witness/test point folds; COLMAP identities themselves were built from the full recording. Not anatomical ground truth or an independent recording.",
        "frame_pair_folds":pairs,"possible_test_observations":possible,"paired_test_observations":baseline.len(),
        "coverage":baseline.len() as f64/possible.max(1) as f64,
        "single_motion_error_px_on_same_support":quantiles(baseline),"group_motion_error_px":quantiles(candidate),
        "improved_over_half_pixel":improved,"regressed_over_half_pixel":regressed,
        "groups":per.iter().enumerate().map(|(g,p)|json!({"id":g+1,"tests":p.len(),
            "single_motion_error_px":quantiles(p.iter().map(|p|p.0).collect()),
            "group_motion_error_px":quantiles(p.iter().map(|p|p.1).collect())})).collect::<Vec<_>>()})
}
fn measured_frames(model: &Value) -> Result<(Vec<Frame>, usize), Error> {
    let mut dropped = 0;
    let mut result = Vec::new();
    let n = model["points"].as_array().ok_or("points")?.len();
    for f in model["frames"].as_array().ok_or("frames")? {
        let off = [
            f["offset"][0].as_f64().ok_or("offset")?,
            f["offset"][1].as_f64().ok_or("offset")?,
        ];
        let mut duplicates = BTreeMap::<usize, Vec<P>>::new();
        for o in f["observations"].as_array().ok_or("observations")? {
            let i = o[0].as_u64().ok_or("point index")? as usize;
            if i >= n {
                return Err("point index out of range".into());
            }
            let p = [
                o[1].as_f64().ok_or("x")? + off[0],
                o[2].as_f64().ok_or("y")? + off[1],
            ];
            if !p.iter().all(|v| v.is_finite()) {
                return Err("non-finite measured location".into());
            }
            duplicates.entry(i).or_default().push(p);
        }
        let mut frame = Frame::new();
        for (i, p) in duplicates {
            let center =
                std::array::from_fn(|k| p.iter().map(|p| p[k]).sum::<f64>() / p.len() as f64);
            if p.iter().any(|&p| distance(p, center) > 1.5) {
                dropped += 1;
                continue;
            }
            frame.insert(i, center);
        }
        result.push(frame);
    }
    if result.len() > 512 || n > 5000 {
        return Err("motion grouping exceeds bounded frame/point budget".into());
    }
    Ok((result, dropped))
}
// Preserve explicit continuity. A recovered tracker ID starts a new segment;
// merely reusing its number must never manufacture a long tissue trajectory.
fn native_frames(
    model: &mut Value,
    rows: &[Value],
    inventory: &Value,
) -> Result<(Vec<Frame>, Value), Error> {
    let eye = model["eye_id"].as_u64().ok_or("eye")?;
    let rows = rows
        .iter()
        .filter(|r| r["source"]["eye_id"] == eye)
        .collect::<Vec<_>>();
    let count = model["frames"].as_array().ok_or("frames")?.len();
    if rows.len() != count || inventory["images"].as_array().ok_or("inventory")?.len() != count {
        return Err("native track/source frame count mismatch".into());
    }
    let mut frames = vec![Frame::new(); count];
    let mut last = BTreeMap::<u64, (usize, u64, P)>::new();
    let mut names = Vec::new();
    let mut origins = Vec::new();
    let mut splits = 0;
    for (i, r) in rows.iter().enumerate() {
        let s = &r["source"];
        let stamp = s["timestamp_ns"].as_u64().ok_or("source time")?;
        if r["motion_only"] != true
            || s["sequence"] != model["frames"][i]["sequence"]
            || stamp.to_string()
                != model["frames"][i]["timestamp_ns"]
                    .as_str()
                    .ok_or("frame time")?
            || r["raw_sha256"] != inventory["images"][i]["raw_sha256"]
        {
            return Err("native/SfM RAW identity mismatch or non-generic tracker".into());
        }
        let origin = [
            s["sensor_x"].as_f64().ok_or("sensor origin")?,
            s["sensor_y"].as_f64().ok_or("sensor origin")?,
        ];
        origins.push(origin);
        if r["reset"] == true {
            last.clear();
        }
        for m in r["tensor_points"].as_array().ok_or("native points")? {
            if m["normal_flow"] == true {
                continue;
            }
            let id = m["id"].as_u64().ok_or("track id")?;
            let xy = |v: &Value| -> Result<P, Error> {
                Ok([
                    v[0].as_f64().ok_or("track x")?,
                    v[1].as_f64().ok_or("track y")?,
                ])
            };
            let p = xy(&m["previous_sensor"])?;
            let q = xy(&m["current_sensor"])?;
            let prev = m["previous_timestamp_ns"].as_u64().ok_or("previous time")?;
            let continuous = last.get(&id).filter(|(_, t, at)| {
                *t == prev
                    && distance(*at, p) < 0.02
                    && m["consecutive_matches_before"].as_u64().unwrap_or(0) > 0
            });
            let index = if let Some(&(k, _, _)) = continuous {
                k
            } else {
                if last.contains_key(&id) {
                    splits += 1;
                }
                let k = names.len();
                names.push(format!("R{id}.{k}"));
                if i > 0 && rows[i - 1]["source"]["timestamp_ns"].as_u64() == Some(prev) {
                    frames[i - 1].insert(k, p);
                }
                k
            };
            frames[i].insert(index, q);
            last.insert(id, (index, stamp, q));
        }
    }
    if names.len() > 8000 {
        return Err("native segment budget exceeded".into());
    }
    let mut tracks = vec![Vec::new(); names.len()];
    for (i, f) in frames.iter().enumerate() {
        model["frames"][i]["raw_motion_observations"] = json!(f
            .iter()
            .map(|(&id, p)| {
                tracks[id].push(i);
                json!([id, p[0] - origins[i][0], p[1] - origins[i][1]])
            })
            .collect::<Vec<_>>());
        model["frames"][i]["native_origin"] = json!(origins[i]);
    }
    Ok((
        frames,
        json!({"point_ids":names,"point_tracks":tracks,"reidentified_ids_split":splits,
        "identity":"Native RAW patch track segments; breaks and reidentification split identities; no COLMAP filtering"}),
    ))
}
fn transfer_to_map(model: &Value, frames: &[Frame], groups: &Groups) -> (Vec<usize>, Vec<usize>) {
    let count = model["points"].as_array().unwrap().len();
    let mut votes = vec![BTreeMap::<usize, usize>::new(); count];
    for i in (0..frames.len()).step_by(2) {
        let f = &model["frames"][i];
        let origin = [
            f["native_origin"][0].as_f64().unwrap(),
            f["native_origin"][1].as_f64().unwrap(),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for o in f["observations"].as_array().unwrap() {
            let id = o[0].as_u64().unwrap() as usize;
            if !seen.insert(id) {
                continue;
            }
            let p = [
                o[1].as_f64().unwrap() + origin[0],
                o[2].as_f64().unwrap() + origin[1],
            ];
            let closest = frames[i]
                .iter()
                .filter_map(|(&k, &q)| {
                    let d = distance(p, q);
                    (d <= 2.5 && groups.labels[k] > 0).then_some((groups.labels[k], d))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((g, _)) = closest {
                *votes[id].entry(g).or_default() += 1;
            }
        }
    }
    let mut labels = vec![0; count];
    let mut support = vec![0; count];
    for (i, v) in votes.iter().enumerate() {
        if let Some((&g, &n)) = v.iter().max_by_key(|(_, n)| *n) {
            let total = v.values().sum::<usize>();
            if n >= 3 && n as f64 / total as f64 >= 0.8 {
                labels[i] = g;
                support[i] = n;
            }
        }
    }
    (labels, support)
}
pub fn run(args: &[String]) -> Result<(), Error> {
    if !(4..=6).contains(&args.len()) {
        return Err(
            "--colmap-motion-groups MODEL_REVIEW_DIR NEW_OUTPUT_DIR [RADIUS_PX=1.5] [NATIVE_RAW_REPLAY]".into(),
        );
    }
    let input = Path::new(&args[2]).canonicalize()?;
    let out = Path::new(&args[3]);
    let root = Path::new("outputs").canonicalize()?;
    if !input.starts_with(&root)
        || out.exists()
        || !out
            .parent()
            .ok_or("output parent")?
            .canonicalize()?
            .starts_with(&root)
    {
        return Err("existing checked input and new checked output required".into());
    }
    let radius = args
        .get(4)
        .map(|s| s.parse::<f64>())
        .transpose()?
        .unwrap_or(1.5);
    if !radius.is_finite() || !(0.75..=3.).contains(&radius) {
        return Err("radius outside 0.75..3 native pixels".into());
    }
    let bytes = fs::read(input.join("models.json"))?;
    let mut document: Value = serde_json::from_slice(&bytes)?;
    let raw_bytes = args
        .get(5)
        .map(|p| -> Result<Vec<u8>, Error> {
            let p = Path::new(p).canonicalize()?.join("frames.jsonl");
            if !p.starts_with(&root) || fs::metadata(&p)?.len() > 128 * 1024 * 1024 {
                return Err("bounded checked native replay required".into());
            }
            Ok(fs::read(p)?)
        })
        .transpose()?;
    let raw_rows = raw_bytes
        .as_ref()
        .map(|b| -> Result<Vec<Value>, Error> {
            std::str::from_utf8(b)?
                .lines()
                .map(|l| Ok(serde_json::from_str(l)?))
                .collect()
        })
        .transpose()?;
    let probe = document["input"]
        .as_str()
        .ok_or("original probe")?
        .to_string();
    let mut reports = Vec::new();
    let start = Instant::now();
    for m in document["models"].as_array_mut().ok_or("models")? {
        let (frames, dropped, mut native) = if let Some(rows) = raw_rows.as_ref() {
            let inventory: Value = serde_json::from_slice(&fs::read(
                Path::new(&probe).join(format!("eye-{}/inventory.json", m["eye_id"])),
            )?)?;
            let (f, data) = native_frames(m, rows, &inventory)?;
            (f, 0, Some(data))
        } else {
            let (f, d) = measured_frames(m)?;
            (f, d, None)
        };
        let point_count = native.as_ref().map_or_else(
            || m["points"].as_array().unwrap().len(),
            |v| v["point_ids"].as_array().unwrap().len(),
        );
        let groups = discover(&frames, point_count, radius);
        let mut evaluation = evaluate(&frames, &groups, radius);
        if native.is_some() {
            evaluation["scope"]=json!("Within-recording withheld odd-frame coordinates and disjoint witness/test points. Native track extraction saw all frames; no COLMAP or anatomical filtering. Not anatomical ground truth or independent recording.");
        }
        let count = groups.labels.iter().copied().max().unwrap_or(0);
        let summaries=(1..=count).map(|g|json!({"id":g,"name":format!("Motion group {g}"),"points":groups.labels.iter().filter(|&&x|x==g).count(),"anatomy":"unassigned"})).collect::<Vec<_>>();
        let (labels, support) = if let Some(data) = native.as_mut() {
            data["labels"] = json!(groups.labels);
            data["support"] = json!(groups.support);
            transfer_to_map(m, &frames, &groups)
        } else {
            (groups.labels.clone(), groups.support.clone())
        };
        m["motion_groups"] = json!({"labels":labels,"support":support,"native":native,"groups":summaries,"discovery":groups.report,
            "validation":evaluation,"ambiguous_duplicate_observations_omitted":dropped,
            "status":"experimental candidate partition; no anatomical accuracy established",
            "coordinates":"Native 2-D observations plus fixed-canvas crop offsets; no 3-D point/pose/color/anatomy input",
            "view_note":"Colors group measured 2D motion. 3D locations and poses remain the original joint COLMAP solve."});
        if raw_rows.is_some() {
            m["motion_groups"]["coordinates"]=json!("Generic native RAW patch observations in absolute sensor coordinates; no 3-D point/pose/color/anatomy input");
            m["motion_groups"]["map_transfer"] = json!({"assigned_3d_points":labels.iter().filter(|&&v|v>0).count(),"policy":"Approximate 2D proximity association: <=2.5px in >=3 even frames, >=80% group agreement. Not proven identical physical features or separate 3D solves."});
        }
        reports.push(json!({"eye_id":m["eye_id"],"discovery":m["motion_groups"]["discovery"],"validation":m["motion_groups"]["validation"],"ambiguous_duplicate_observations_omitted":dropped}));
        println!("eye {}: {}", m["eye_id"], m["motion_groups"]["discovery"]);
    }
    fs::create_dir(out)?;
    // Share immutable, RAW-verified source previews without regenerating them.
    for entry in fs::read_dir(&input)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "png") {
            fs::hard_link(&path, out.join(entry.file_name()))?;
        }
    }
    document["motion_group_provenance"] = json!({"input_models_sha256":format!("{:x}",Sha256::digest(&bytes)),"input":input,"source_sha256":format!("{:x}",Sha256::digest(include_bytes!("colmap_groups.rs"))),"elapsed_seconds":start.elapsed().as_secs_f64(),
        "scope":"Classical offline grouping; no training, learned weights, labels, gaze targets, calibration reuse, or new 3D reconstruction",
        "native_replay_sha256":raw_bytes.as_ref().map(|b|format!("{:x}",Sha256::digest(b))),
        "viewer_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("colmap_viewer.html"))),
        "limitations":["Short/missing tracks may remain unassigned","Original 3D map filtered by COLMAP; optional native 2D tracks are independent of that filter","Similarity approximation can split one curved object or merge objects with indistinguishable motion","Group identity across disjoint visibility is unresolved","No anatomical labels, measured scale or independent 3D truth; SN-FEIDA is not applicable"]});
    fs::write(
        out.join("models.json"),
        serde_json::to_vec_pretty(&document)?,
    )?;
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"eyes":reports,"provenance":document["motion_group_provenance"]}),
        )?,
    )?;
    fs::write(
        out.join("viewer.html"),
        include_str!("colmap_viewer.html")
            .replace("MODEL_DATA", &serde_json::to_string(&document["models"])?),
    )?;
    println!(
        "motion-group review: {} ({:.2}s)",
        out.display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn synthetic(multiple: bool) -> (Vec<Frame>, Vec<usize>) {
        let mut state = 991u64;
        let mut origin = Vec::new();
        let mut truth = Vec::new();
        for i in 0..108 {
            origin.push([
                20. + (random(&mut state) % 380) as f64,
                20. + (random(&mut state) % 240) as f64,
            ]);
            truth.push(if i < 96 { i / 32 + 1 } else { 0 });
        }
        let mut frames = Vec::new();
        for f in 0..120 {
            let mut frame = Frame::new();
            let t = f as f64 * 0.05;
            for (i, &p) in origin.iter().enumerate() {
                if random(&mut state) % 7 == 0 {
                    continue;
                }
                let g = if multiple { truth[i] } else { 1 };
                let theta = match g {
                    1 => 0.018 * t,
                    2 => -0.09 * t,
                    3 => 0.06 * t,
                    _ => 0.,
                };
                let shift = match g {
                    1 => [3. * t, 2. * t],
                    2 => [-4. * t, 3. * (t * 0.6).sin()],
                    3 => [t, -4. * t],
                    _ => [
                        (random(&mut state) % 50) as f64,
                        (random(&mut state) % 50) as f64,
                    ],
                };
                let noise = |v: u64| (v % 101) as f64 / 250. - 0.2;
                frame.insert(
                    i,
                    [
                        p[0] * theta.cos() - p[1] * theta.sin()
                            + shift[0]
                            + noise(random(&mut state)),
                        p[0] * theta.sin()
                            + p[1] * theta.cos()
                            + shift[1]
                            + noise(random(&mut state)),
                    ],
                );
            }
            frames.push(frame);
        }
        (frames, truth)
    }
    #[test]
    fn separated_rotations_and_translations_survive_missing_tracks() {
        let (frames, truth) = synthetic(true);
        let g = discover(&frames, truth.len(), 1.0);
        let mut assigned = 0;
        let mut correct = 0;
        for label in 1..=g.labels.iter().copied().max().unwrap_or(0) {
            let mut n = [0; 4];
            for (i, &v) in g.labels.iter().enumerate() {
                if v == label {
                    n[truth[i]] += 1;
                    assigned += 1;
                }
            }
            correct += *n[1..].iter().max().unwrap();
        }
        assert!(assigned >= 65, "{}", g.report);
        assert!(correct as f64 / assigned as f64 > 0.9, "{}", g.report);
        let e = evaluate(&frames, &g, 1.0);
        assert!(e["coverage"].as_f64().unwrap() > 0.35, "{e}");
        assert!(
            e["group_motion_error_px"]["p50"].as_f64().unwrap() < 0.5,
            "{e}"
        );
    }
    #[test]
    fn common_motion_does_not_force_multiple_groups() {
        let (frames, truth) = synthetic(false);
        let g = discover(&frames, truth.len(), 1.0);
        assert!(g.labels.iter().all(|&g| g == 0), "{}", g.report);
    }
    #[test]
    fn withheld_coordinates_do_not_change_discovered_groups() {
        let (mut f, t) = synthetic(true);
        let a = discover(&f, t.len(), 1.0);
        for i in (1..f.len()).step_by(2) {
            for p in f[i].values_mut() {
                p[0] += i as f64 * 13.;
                p[1] *= 3.;
            }
        }
        let b = discover(&f, t.len(), 1.0);
        assert_eq!(a.labels, b.labels);
    }
    #[test]
    fn crop_offsets_and_duplicate_ambiguity_are_respected() {
        let m = json!({"points":[[],[]],"frames":[{"offset":[30,70],"observations":[[0,5,7],[0,5,7],[1,0,0],[1,10,10]]},{"offset":[34,72],"observations":[[0,1,5]]}]});
        let (f, d) = measured_frames(&m).unwrap();
        assert_eq!(f[0][&0], f[1][&0]);
        assert_eq!(d, 1);
        assert!(!f[0].contains_key(&1));
    }
    #[test]
    fn native_reidentification_splits_and_checks_source_identity() {
        let mut model = json!({"eye_id":1,"frames":[
            {"sequence":1,"timestamp_ns":"100"},
            {"sequence":2,"timestamp_ns":"200"},
            {"sequence":3,"timestamp_ns":"300"}],"points":[]});
        let rows=(0..3).map(|i|json!({"motion_only":true,"reset":i==0,"raw_sha256":"abc",
            "source":{"eye_id":1,"sequence":i+1,"timestamp_ns":(i+1)*100,"sensor_x":1000,"sensor_y":2000},
            "tensor_points":[{"id":9,"normal_flow":false,"previous_timestamp_ns":i*100,
                "consecutive_matches_before":if i==2{0}else{i},
                "previous_sensor":[1010+i,2020],"current_sensor":[1011+i,2020]}]})).collect::<Vec<_>>();
        let inventory =
            json!({"images":[{"raw_sha256":"abc"},{"raw_sha256":"abc"},{"raw_sha256":"abc"}]});
        let (_, native) = native_frames(&mut model, &rows, &inventory).unwrap();
        assert_eq!(native["point_ids"].as_array().unwrap().len(), 2);
        assert_eq!(native["point_tracks"][0], json!([0, 1]));
        assert_eq!(native["point_tracks"][1], json!([1, 2]));
        assert_eq!(native["reidentified_ids_split"], 1);
        let mut bad = rows.clone();
        bad[1]["raw_sha256"] = json!("different");
        assert!(native_frames(&mut model, &bad, &inventory).is_err());
    }
}
