//! Source-timed area admission, independent of rays or focus regions.
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(super) struct Observation {
    pub group: (u64, u64, u64), // recording, clock epoch, eye
    pub ns: u64,
    pub area_px2: Option<f64>, // fresh fit passing the RAW/fit gate
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Admission {
    pub accepted: bool,
    pub reason: &'static str,
    pub area_px2: Option<f64>,
    pub trimmed_neighbor_mean_px2: Option<f64>,
    pub area_ratio: Option<f64>,
    pub before: usize,
    pub after: usize,
    pub neighbor_count: usize,
    pub neighbor_span_ms: f64,
}

pub(super) fn assess(input: &[Observation]) -> Vec<Admission> {
    let mut groups = BTreeMap::<_, Vec<(u64, usize)>>::new();
    for (i, row) in input.iter().enumerate() {
        groups.entry(row.group).or_default().push((row.ns, i));
    }
    let mut output = input
        .iter()
        .map(|row| Admission {
            accepted: false,
            reason: "unusable_fit",
            area_px2: row.area_px2,
            trimmed_neighbor_mean_px2: None,
            area_ratio: None,
            before: 0,
            after: 0,
            neighbor_count: 0,
            neighbor_span_ms: 0.,
        })
        .collect::<Vec<_>>();
    for group in groups.values_mut() {
        group.sort_unstable();
        for &(ns, i) in group.iter() {
            let Some(area) = input[i].area_px2.filter(|a| a.is_finite() && *a > 0.) else {
                continue;
            };
            let lo = group.partition_point(|p| p.0 < ns.saturating_sub(1_000_000_000));
            let hi = group.partition_point(|p| p.0 <= ns.saturating_add(1_000_000_000));
            let neighbors = group[lo..hi]
                .iter()
                .filter(|(_, j)| *j != i)
                .filter_map(|&(t, j)| {
                    input[j]
                        .area_px2
                        .filter(|a| a.is_finite() && *a > 0.)
                        .map(|a| (t, a))
                })
                .collect::<Vec<_>>();
            let before = neighbors.iter().filter(|(t, _)| *t < ns).count();
            let after = neighbors.iter().filter(|(t, _)| *t > ns).count();
            let span = neighbors
                .last()
                .zip(neighbors.first())
                .map(|(a, b)| a.0 - b.0)
                .unwrap_or(0);
            let result = &mut output[i];
            result.before = before;
            result.after = after;
            result.neighbor_count = neighbors.len();
            result.neighbor_span_ms = span as f64 / 1e6;
            if neighbors.len() < 6 || before < 2 || after < 2 || span < 100_000_000 {
                result.reason = "insufficient_neighbors";
                continue;
            }
            let mut values = neighbors.iter().map(|(_, a)| *a).collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            let trim = values.len() / 5;
            let middle = &values[trim..values.len() - trim];
            let mean = middle.iter().sum::<f64>() / middle.len() as f64;
            let ratio = area / mean;
            result.trimmed_neighbor_mean_px2 = Some(mean);
            result.area_ratio = Some(ratio);
            result.accepted = (1. / 1.5..=1.5).contains(&ratio);
            result.reason = if result.accepted {
                "consistent_area"
            } else {
                "area_outlier"
            };
        }
    }
    output
}
