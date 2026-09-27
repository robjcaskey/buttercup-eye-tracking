//! Descriptive continuity counts on the completed, retrospective focus replay.
//! Never promotes a compatible candidate to physical gaze ground truth.
use crate::Result;
use buttercup_eye_tracking::focus_region::{dot, TheoreticalEllipseExplanations};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

#[derive(Deserialize)]
struct Candidate {
    status: String,
    miss_degrees: Option<f64>,
}
#[derive(Deserialize)]
struct Row {
    record: usize,
    source: u32,
    epoch: u32,
    eye: u16,
    sequence: u64,
    source_ns: String,
    provider: u8,
    classification: String,
    explanations: Option<TheoreticalEllipseExplanations>,
    interpretations: Vec<Candidate>,
    disk_area: Value,
}
struct Frame {
    row: Row,
    ns: u64,
}
impl Frame {
    fn winner(&self) -> Option<usize> {
        if self.row.classification != "one" {
            return None;
        }
        let winners = self
            .row
            .interpretations
            .iter()
            .enumerate()
            .filter(|(_, c)| c.status == "inside" || c.status == "nearby")
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        (winners.len() == 1).then(|| winners[0])
    }
}
#[derive(Clone, Serialize)]
struct Match {
    prior_record: usize,
    prior_sequence: u64,
    prior_provider: u8,
    prior_branch: usize,
    prior_selected_region_miss_degrees: Option<f64>,
    prior_rejected_region_miss_degrees: Option<f64>,
    age_ms: f64,
    frames_since_prior: usize,
    intervening_classes: BTreeMap<String, usize>,
    current_branch: usize,
    candidate_distance_degrees: [f64; 2],
    near_degrees: f64,
    far_degrees: f64,
    margin_degrees: f64,
    strong_5_20: bool,
    unnormalized_area_change_percent: Option<f64>,
}
fn compare(rows: &[Frame], i: usize, p: usize) -> Option<Match> {
    let prior = &rows[p];
    let now = &rows[i];
    let branch = prior.winner()?;
    let old = prior.row.explanations?.rays[branch].direction;
    let rays = now.row.explanations?;
    let d = rays
        .rays
        .map(|r| dot(old, r.direction).clamp(-1., 1.).acos().to_degrees());
    let nearest = usize::from(d[1] < d[0]);
    let near = d[nearest];
    let far = d[1 - nearest];
    let mut intervening = BTreeMap::new();
    for f in &rows[p + 1..i] {
        *intervening.entry(f.row.classification.clone()).or_default() += 1;
    }
    let previous_area = prior.row.disk_area["frontal_equivalent_disk_area_px2"].as_f64();
    let area = now.row.disk_area["frontal_equivalent_disk_area_px2"].as_f64();
    Some(Match {
        prior_record: prior.row.record,
        prior_sequence: prior.row.sequence,
        prior_provider: prior.row.provider,
        prior_branch: branch,
        prior_selected_region_miss_degrees: prior.row.interpretations[branch].miss_degrees,
        prior_rejected_region_miss_degrees: prior.row.interpretations[1 - branch].miss_degrees,
        age_ms: (now.ns - prior.ns) as f64 * 1e-6,
        frames_since_prior: i - p,
        intervening_classes: intervening,
        current_branch: nearest,
        candidate_distance_degrees: d,
        near_degrees: near,
        far_degrees: far,
        margin_degrees: far - near,
        strong_5_20: near <= 5. && far >= 20.,
        unnormalized_area_change_percent: area
            .zip(previous_area)
            .filter(|(_, p)| *p > 0.)
            .map(|(a, p)| (a / p - 1.) * 100.),
    })
}
#[derive(Clone, Serialize)]
struct Case {
    record: usize,
    source: u32,
    epoch: u32,
    eye: u16,
    sequence: u64,
    source_ns: String,
    provider: u8,
    previous_indexed_class: Option<String>,
    previous_indexed_age_ms: Option<f64>,
    immediately_previous: Option<Match>,
    most_recent_unambiguous: Option<Match>,
    two_previous_unambiguous_agree_5_20: bool,
}
fn match_for<'a>(case: &'a Case, mode: &str) -> Option<&'a Match> {
    if mode == "immediately_previous" {
        case.immediately_previous.as_ref()
    } else {
        case.most_recent_unambiguous.as_ref()
    }
}
fn table(cases: &[Case], mode: &str, max_ms: f64, close: f64, far: f64) -> Value {
    let eligible = cases
        .iter()
        .filter_map(|c| {
            match_for(c, mode)
                .filter(|m| m.age_ms > 0. && m.age_ms <= max_ms)
                .map(|m| (c, m))
        })
        .collect::<Vec<_>>();
    let strong = eligible
        .iter()
        .filter(|(_, m)| m.near_degrees <= close && m.far_degrees >= far)
        .collect::<Vec<_>>();
    let quantiles = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            Value::Null
        } else {
            json!({"min":v[0],"median":v[(v.len()-1)/2],"p90":v[(v.len()-1)*9/10],"max":v[v.len()-1]})
        }
    };
    let mut providers = BTreeMap::<String, usize>::new();
    for (c, m) in &strong {
        *providers
            .entry(format!("{}->{}", m.prior_provider, c.provider))
            .or_default() += 1;
    }
    json!({"mode":mode,"maximum_age_ms":max_ms,"close_at_most_degrees":close,"other_at_least_degrees":far,
        "ambiguous_frames":cases.len(),"has_eligible_unambiguous_predecessor":eligible.len(),"strong_match":strong.len(),
        "strong_percent_of_all":strong.len() as f64*100./cases.len().max(1) as f64,
        "recordings_with_strong_match":strong.iter().map(|(c,_)|c.source).collect::<BTreeSet<_>>().len(),
        "near_degrees":quantiles(strong.iter().map(|(_,m)|m.near_degrees).collect()),
        "far_degrees":quantiles(strong.iter().map(|(_,m)|m.far_degrees).collect()),
        "age_ms":quantiles(strong.iter().map(|(_,m)|m.age_ms).collect()),
        "strong_provider_pairs":providers,
        "prior_rejected_region_miss_degrees":quantiles(strong.iter().filter_map(|(_,m)|m.prior_rejected_region_miss_degrees).collect()),
        "strong_with_prior_other_region_miss_at_least_5deg":strong.iter().filter(|(_,m)|m.prior_rejected_region_miss_degrees.is_some_and(|x|x>=5.)).count(),
        "strong_with_prior_other_region_miss_at_least_10deg":strong.iter().filter(|(_,m)|m.prior_rejected_region_miss_degrees.is_some_and(|x|x>=10.)).count(),
        "strong_with_no_intervening_unavailable_or_rejected_frames":strong.iter().filter(|(_,m)|m.intervening_classes.keys().all(|k|k=="one"||k=="multiple")).count(),
        "strong_with_unnormalized_area_change_at_most_10pct":strong.iter().filter(|(_,m)|m.unnormalized_area_change_percent.is_some_and(|a|a.abs()<=10.)).count()})
}
pub fn run(replay: &str, output: &str) -> Result<()> {
    let dir = Path::new(replay);
    let out = Path::new(output);
    if out.exists() {
        return Err("continuity output exists".into());
    }
    fs::create_dir_all(out)?;
    let summary: Value = serde_json::from_slice(&fs::read(dir.join("summary.json"))?)?;
    let source_ids = summary["per_group"]
        .as_array()
        .ok_or("per_group")?
        .iter()
        .filter(|g| g["classes"]["multiple"].as_u64().unwrap_or(0) > 0)
        .map(|g| g["source"].as_u64().unwrap() as u32)
        .collect::<BTreeSet<_>>();
    let selected_groups = summary["per_group"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|g| g["classes"]["multiple"].as_u64().unwrap_or(0) > 0)
        .map(|g| {
            (
                g["source"].as_u64().unwrap() as u32,
                g["epoch"].as_u64().unwrap() as u32,
            )
        })
        .collect::<BTreeSet<_>>();
    let mut groups = BTreeMap::<(u32, u32, u16), Vec<Frame>>::new();
    // This file is generated by our replay. JSON parsing (not text replacement)
    // binds the selected rows; the string field scan only avoids parsing large
    // histories from recordings that have no ambiguous exposures.
    let file = fs::File::open(dir.join("classifications.jsonl"))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let source = line
            .split_once("\"source\":")
            .and_then(|(_, s)| s.split([',', '}']).next())
            .and_then(|s| s.parse::<u32>().ok());
        let epoch = line
            .split_once("\"epoch\":")
            .and_then(|(_, s)| s.split([',', '}']).next())
            .and_then(|s| s.parse::<u32>().ok());
        let selected = source
            .zip(epoch)
            .is_some_and(|key| selected_groups.contains(&key));
        if !selected {
            continue;
        }
        let row: Row = serde_json::from_str(&line)?;
        let ns = row.source_ns.parse()?;
        groups
            .entry((row.source, row.epoch, row.eye))
            .or_default()
            .push(Frame { row, ns });
    }
    let mut cases = vec![];
    let mut preceding_classes = BTreeMap::<String, usize>::new();
    let mut duplicates = 0;
    for ((_, _, _), mut rows) in groups {
        rows.sort_by_key(|f| (f.ns, f.row.sequence, f.row.record));
        let n = rows.len();
        rows.dedup_by_key(|f| (f.ns, f.row.sequence));
        duplicates += n - rows.len();
        let mut last_one: Option<usize> = None;
        for (i, f) in rows.iter().enumerate() {
            if f.row.classification == "multiple" {
                if f.row.explanations.is_none() {
                    return Err("ambiguous frame has no rays".into());
                }
                let prev = i.checked_sub(1);
                let direct = prev.and_then(|p| compare(&rows, i, p));
                let recent = last_one.and_then(|p| compare(&rows, i, p));
                let corroborated = prev
                    .filter(|&p| p > 0)
                    .and_then(|p| compare(&rows, i, p - 1))
                    .zip(direct.as_ref())
                    .is_some_and(|(a, b)| {
                        a.age_ms <= 2000.
                            && b.age_ms > 0.
                            && a.strong_5_20
                            && b.strong_5_20
                            && a.current_branch == b.current_branch
                    });
                let previous_class = prev.map(|p| rows[p].row.classification.clone());
                *preceding_classes
                    .entry(previous_class.clone().unwrap_or("recording_start".into()))
                    .or_default() += 1;
                cases.push(Case {
                    record: f.row.record,
                    source: f.row.source,
                    epoch: f.row.epoch,
                    eye: f.row.eye,
                    sequence: f.row.sequence,
                    source_ns: f.row.source_ns.clone(),
                    provider: f.row.provider,
                    previous_indexed_class: previous_class,
                    previous_indexed_age_ms: prev.map(|p| (f.ns - rows[p].ns) as f64 * 1e-6),
                    immediately_previous: direct,
                    most_recent_unambiguous: recent,
                    two_previous_unambiguous_agree_5_20: corroborated,
                });
            }
            if f.winner().is_some() {
                last_one = Some(i);
            }
        }
    }
    if cases.len() as u64
        != summary["classes"]["multiple"]
            .as_u64()
            .ok_or("ambiguity count")?
    {
        return Err("temporal analysis lost ambiguous frames".into());
    }
    let mut output_cases = BufWriter::new(fs::File::create(out.join("cases.jsonl"))?);
    for c in &cases {
        serde_json::to_writer(&mut output_cases, c)?;
        writeln!(output_cases)?;
    }
    output_cases.flush()?;
    let mut tables = vec![];
    for mode in ["immediately_previous", "most_recent_unambiguous"] {
        for ms in [250., 500., 1000., 2000.] {
            for near in [3., 5., 10.] {
                tables.push(table(&cases, mode, ms, near, 20.));
            }
        }
    }
    let sam = cases
        .iter()
        .filter(|c| c.provider == 1)
        .cloned()
        .collect::<Vec<_>>();
    let mut sam_tables = vec![];
    for mode in ["immediately_previous", "most_recent_unambiguous"] {
        for ms in [250., 500., 1000., 2000.] {
            sam_tables.push(table(&sam, mode, ms, 5., 20.));
        }
    }
    let mut per_source = vec![];
    for source in source_ids {
        let subset = cases
            .iter()
            .filter(|c| c.source == source)
            .cloned()
            .collect::<Vec<_>>();
        per_source.push(json!({"source":source,"ambiguous":subset.len(),"immediate_250ms":table(&subset,"immediately_previous",250.,5.,20.),"recent_2s":table(&subset,"most_recent_unambiguous",2000.,5.,20.)}));
    }
    let report = json!({"schema":"buttercup-ambiguity-continuity-v1","replay":replay,"binary_sha256":summary["binary_sha256"],"ambiguous_frames":cases.len(),"duplicate_source_keys_dropped":duplicates,"immediately_preceding_class":preceding_classes,
        "strong_with_two_consecutive_unambiguous_predecessors":cases.iter().filter(|c|c.two_previous_unambiguous_agree_5_20).count(),"tables":tables,"sam_current_frames":sam.len(),"sam_current_tables":sam_tables,"per_source":per_source,
        "limits":["Predecessor is strictly earlier and in the same recording, source epoch and eye. Same timestamp is not fresh support.","Unambiguous means only one candidate fit retrospective inferred volumes; that candidate is not independently verified.","Angles compare 3-D unit normals in nominal camera coordinates, not calibrated screen gaze or motion-compensated physical directions.","Thresholds declared before calculation: close <=5 deg and other >=20 deg; sensitivity uses 3/10 deg and 0.25/0.5/1/2-second maximum ages.","Most recent unambiguous frame is used, not the most favorable historical frame. Intervening missing/rejected/zero frames are reported explicitly.","Archived contact-derived ellipses dominate this set. Providers and source recordings remain stratified.","Area-change diagnostic is unnormalized frontal-equivalent disk area; independent scale/SN-FEIDA and human sign/localization truth remain unavailable.","This is a descriptive continuity audit. It does not update candidate classifications or enable sign propagation in the live tracker."]});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"ambiguous":cases.len(),"previous_class":preceding_classes,"immediate_250ms":table(&cases,"immediately_previous",250.,5.,20.),"recent_2s":table(&cases,"most_recent_unambiguous",2000.,5.,20.),"two_consecutive_prior_ones":report["strong_with_two_consecutive_unambiguous_predecessors"]})
        )?
    );
    Ok(())
}
