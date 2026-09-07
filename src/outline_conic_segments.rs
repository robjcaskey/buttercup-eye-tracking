//! Ordered outline evidence, occlusion-chord exclusion, and feedback from a conic fit.
//!
//! This preserves the existing flat-tire pipeline. It does not yet enumerate
//! alternative gradient bands or jointly segment several ROIs. Semantic mask
//! generation, tile/native coordinate conversion, and publication stay upstream.

use crate::conic_solver::{
    constrained_ransac_ellipse, ellipse_residual, plausible_ellipse, robust_contour_fit,
    ConicArcConstraints, NumpyPcg64, OuterContourScaleContext, LEGACY_FIT_WIDTH as FRAME_WIDTH,
};
use crate::geometry::{ellipse_axis_point, Ellipse};
use std::collections::HashMap;
use std::sync::Arc;

/// Offline-only temporal exclusion experiment; the stateless live fitter below
/// deliberately does not call this module.
pub(crate) mod recent_exclusion;
pub(crate) mod sparse_evidence;

/// Geometric evidence extracted from an ordered semantic-mask contour.
/// Points and ellipse share the caller's pixel frame (model space during
/// fitting; native RAW ROI space after upstream coordinate conversion).  A flat-tire
/// point belongs to a long, physically implausible low-curvature occlusion
/// chord at any orientation (including upper/lower lids) and is deliberately
/// excluded from the ellipse fit.
#[derive(Clone, Debug)]
pub struct ContourFitEvidence {
    pub ellipse: Ellipse,
    /// Area of the connected semantic-mask component in native ROI pixels.
    /// Comparing this with PI*a*b exposes fits extrapolated from a much
    /// larger eyelid/whole-eye object without assuming the projected limbus
    /// itself must be circular.
    pub source_component_area_px: f64,
    pub retained_points: Arc<Vec<(f64, f64)>>,
    /// Contiguous supported runs, indexing retained_points in contour order.
    /// No run crosses a rejected sample; these are observed arc supports,
    /// not independently solved ellipses or a completed stereo posterior.
    pub conic_segments: Arc<Vec<Vec<usize>>>,
    pub flat_tire_points: Arc<Vec<(f64, f64)>>,
    pub upper_flat_tire: bool,
    pub lower_flat_tire: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct BoundaryEdge {
    pub(crate) start: (i32, i32),
    pub(crate) end: (i32, i32),
    pub(crate) owner: usize,
}

fn supported_conic_runs(kept: &[bool]) -> Vec<Vec<usize>> {
    let mut runs = Vec::new();
    let mut run = Vec::new();
    let mut retained_index = 0;
    for &keep in kept {
        if keep { run.push(retained_index); retained_index += 1; }
        else if !run.is_empty() { runs.push(std::mem::take(&mut run)); }
    }
    if !run.is_empty() { runs.push(run); }
    // The contour is cyclic. Merge its end only when both boundary samples
    // were retained; rejected samples anywhere else still split the arc.
    if runs.len() > 1 && kept.first() == Some(&true) && kept.last() == Some(&true) {
        let mut last = runs.pop().unwrap();
        last.extend(std::mem::take(&mut runs[0]));
        runs[0] = last;
    }
    runs.into_iter().filter(|run| run.len() >= 3).collect()
}

#[cfg(test)]
mod segment_tests {
    use super::*;
    #[test]
    fn conic_segments_do_not_bridge_rejected_samples() {
        assert_eq!(supported_conic_runs(&[false,true,true,true,false,true,true,true,false]),
            vec![vec![0,1,2],vec![3,4,5]]);
        assert_eq!(supported_conic_runs(&[true,true,false,true,true,true]),vec![vec![2,3,4,0,1]]);
        assert!(supported_conic_runs(&[true,false,true,false]).is_empty());
        assert!(supported_conic_runs(&[]).is_empty());
    }
}

pub(crate) fn native_component_contour(
    component: &[usize],
    mask_width: usize,
    mask_height: usize,
) -> Vec<(f64, f64)> {
    if component.len() < 5 {
        return Vec::new();
    }
    let mut membership = vec![false; mask_width * mask_height];
    for &index in component {
        membership[index] = true;
    }
    let mut sorted_component = component.to_vec();
    sorted_component.sort_unstable();
    let mut edges = Vec::<BoundaryEdge>::new();
    for index in sorted_component {
        let x = index % mask_width;
        let y = index / mask_width;
        let x0 = x as i32;
        let y0 = y as i32;
        if y == 0 || !membership[index - mask_width] {
            edges.push(BoundaryEdge {
                start: (x0, y0),
                end: (x0 + 1, y0),
                owner: index,
            });
        }
        if x + 1 == mask_width || !membership[index + 1] {
            edges.push(BoundaryEdge {
                start: (x0 + 1, y0),
                end: (x0 + 1, y0 + 1),
                owner: index,
            });
        }
        if y + 1 == mask_height || !membership[index + mask_width] {
            edges.push(BoundaryEdge {
                start: (x0 + 1, y0 + 1),
                end: (x0, y0 + 1),
                owner: index,
            });
        }
        if x == 0 || !membership[index - 1] {
            edges.push(BoundaryEdge {
                start: (x0, y0 + 1),
                end: (x0, y0),
                owner: index,
            });
        }
    }
    let mut outgoing = HashMap::<(i32, i32), Vec<usize>>::new();
    for (index, edge) in edges.iter().enumerate() {
        outgoing.entry(edge.start).or_default().push(index);
    }
    let mut used = vec![false; edges.len()];
    let mut best_vertices = Vec::<(i32, i32)>::new();
    let mut best_owners = Vec::<usize>::new();
    let mut best_area = 0.0f64;
    for first in 0..edges.len() {
        if used[first] {
            continue;
        }
        let start = edges[first].start;
        let mut current = first;
        let mut vertices = Vec::new();
        let mut owners = Vec::new();
        for _ in 0..=edges.len() {
            if used[current] {
                break;
            }
            let edge = edges[current];
            used[current] = true;
            vertices.push(edge.start);
            owners.push(edge.owner);
            if edge.end == start {
                break;
            }
            let Some(candidates) = outgoing.get(&edge.end) else {
                break;
            };
            let incoming = (edge.end.0 - edge.start.0, edge.end.1 - edge.start.1);
            let next = candidates
                .iter()
                .copied()
                .filter(|&candidate| !used[candidate])
                .max_by_key(|&candidate| {
                    let candidate = edges[candidate];
                    let direction = (
                        candidate.end.0 - candidate.start.0,
                        candidate.end.1 - candidate.start.1,
                    );
                    let cross = incoming.0 * direction.1 - incoming.1 * direction.0;
                    let dot = incoming.0 * direction.0 + incoming.1 * direction.1;
                    match (cross.signum(), dot.signum()) {
                        (1, _) => 3,
                        (0, 1) => 2,
                        (-1, _) => 1,
                        _ => 0,
                    }
                });
            let Some(next) = next else {
                break;
            };
            current = next;
        }
        if vertices.len() < 5 || edges[current].end != start {
            continue;
        }
        let twice_area = vertices
            .iter()
            .zip(vertices.iter().cycle().skip(1))
            .take(vertices.len())
            .map(|(first, second)| {
                first.0 as f64 * second.1 as f64 - second.0 as f64 * first.1 as f64
            })
            .sum::<f64>();
        let area = 0.5 * twice_area.abs();
        if area > best_area {
            best_area = area;
            best_vertices = vertices;
            best_owners = owners;
        }
    }
    let _ = best_vertices;
    let mut contour = Vec::with_capacity(best_owners.len());
    for owner in best_owners {
        let point = ((owner % mask_width) as f64, (owner / mask_width) as f64);
        if contour.last().copied() != Some(point) {
            contour.push(point);
        }
    }
    if contour.len() >= 5 {
        if let Some(start) = (0..contour.len()).min_by(|&first, &second| {
            contour[first]
                .1
                .total_cmp(&contour[second].1)
                .then_with(|| contour[first].0.total_cmp(&contour[second].0))
        }) {
            contour.rotate_left(start);
        }
    }
    contour
}

pub(crate) fn sample_closed_contour(points: &[(f64, f64)], count: usize) -> Vec<(f64, f64)> {
    if points.len() <= 1 || count == 0 {
        return points.to_vec();
    }
    let mut cumulative = Vec::with_capacity(points.len() + 1);
    cumulative.push(0.0);
    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        let length = (points[next].0 - points[index].0).hypot(points[next].1 - points[index].1);
        cumulative.push(cumulative.last().copied().unwrap() + length);
    }
    let total = *cumulative.last().unwrap();
    if total <= 1e-9 {
        return points.iter().copied().take(count).collect();
    }
    (0..count)
        .map(|sample| {
            let target = (sample as f64 + 0.5) * total / count as f64;
            let segment = cumulative
                .partition_point(|value| *value <= target)
                .saturating_sub(1)
                .min(points.len() - 1);
            let next = (segment + 1) % points.len();
            let span = (cumulative[segment + 1] - cumulative[segment]).max(1e-9);
            let blend = (target - cumulative[segment]) / span;
            (
                (points[segment].0 * (1.0 - blend) + points[next].0 * blend) as f32 as f64,
                (points[segment].1 * (1.0 - blend) + points[next].1 * blend) as f32 as f64,
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FlatTireSide {
    Upper,
    Lower,
    Other,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FlatTireRun {
    pub(crate) side: FlatTireSide,
    pub(crate) start: usize,
    pub(crate) length: usize,
    pub(crate) chord_length: f64,
}

pub(crate) fn smooth_closed_contour(points: &[(f64, f64)], radius: usize) -> Vec<(f64, f64)> {
    if points.is_empty() || radius == 0 {
        return points.to_vec();
    }
    let count = points.len();
    (0..count)
        .map(|index| {
            let mut sum = (0.0, 0.0);
            for offset in 0..=(2 * radius) {
                let sample = (index + count + offset - radius) % count;
                sum.0 += points[sample].0;
                sum.1 += points[sample].1;
            }
            let denominator = (2 * radius + 1) as f64;
            (sum.0 / denominator, sum.1 / denominator)
        })
        .collect()
}

/// Find the longest nearly straight, predominantly major-axis-aligned run on
/// one vertical extreme of the selected disk contour.  A real ellipse may be
/// locally horizontal at its pole, but it bends away from its endpoint chord;
/// an eyelid-clipped mask instead contains a long low-sagitta chord.  Testing
/// the whole ordered run (rather than individual pixel tangents) is stable in
/// the presence of the one-pixel staircase left by mask rasterization.
pub(crate) fn best_flat_tire_run(
    smoothed: &[(f64, f64)],
    reference: Ellipse,
    side: FlatTireSide,
) -> Option<FlatTireRun> {
    let count = smoothed.len();
    if count < 24 {
        return None;
    }
    let local = smoothed
        .iter()
        .copied()
        .map(|point| ellipse_axis_point(point, reference))
        .collect::<Vec<_>>();
    let minimum_u = local
        .iter()
        .map(|point| point.0)
        .fold(f64::INFINITY, f64::min);
    let maximum_u = local
        .iter()
        .map(|point| point.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let minimum_v = local
        .iter()
        .map(|point| point.1)
        .fold(f64::INFINITY, f64::min);
    let maximum_v = local
        .iter()
        .map(|point| point.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let horizontal_span = maximum_u - minimum_u;
    let vertical_span = maximum_v - minimum_v;
    if horizontal_span < 30.0 || vertical_span < 20.0 {
        return None;
    }
    let in_extreme_band = |point: (f64, f64)| match side {
        FlatTireSide::Upper => point.1 <= minimum_v + 0.40 * vertical_span,
        FlatTireSide::Lower => point.1 >= maximum_v - 0.40 * vertical_span,
        FlatTireSide::Other => true,
    };
    // A rasterized ellipse has a short, superficially straight plateau at its
    // true top and bottom.  Only a chord spanning roughly a quarter of the
    // observed diameter is long enough to be treated as occlusion here; the
    // all-orientation physical-curvature pass below handles other cases.
    let minimum_chord = (0.24 * horizontal_span).max(20.0);
    let minimum_samples = (count / 24).max(6);
    let maximum_samples = count / 2;
    let mut best: Option<FlatTireRun> = None;

    for start in 0..count {
        for length in minimum_samples..=maximum_samples {
            let first = local[start];
            let last = local[(start + length - 1) % count];
            let chord = (last.0 - first.0, last.1 - first.1);
            let chord_length = chord.0.hypot(chord.1);
            if chord_length < minimum_chord || chord.0.abs() / chord_length.max(1.0e-9) < 0.76 {
                continue;
            }
            let residual_limit = (0.032 * chord_length).clamp(1.35, 4.0);
            let mut maximum_residual = 0.0f64;
            let mut mean_residual = 0.0f64;
            let mut valid = true;
            for offset in 0..length {
                let point = local[(start + offset) % count];
                if !in_extreme_band(point) {
                    valid = false;
                    break;
                }
                let residual = ((point.0 - first.0) * chord.1 - (point.1 - first.1) * chord.0)
                    .abs()
                    / chord_length.max(1.0e-9);
                maximum_residual = maximum_residual.max(residual);
                mean_residual += residual;
                if maximum_residual > residual_limit {
                    valid = false;
                    break;
                }
            }
            mean_residual /= length as f64;
            if !valid || mean_residual > residual_limit * 0.52 {
                continue;
            }
            let candidate = FlatTireRun {
                side,
                start,
                length,
                chord_length,
            };
            let replace = best.is_none_or(|incumbent| {
                candidate.chord_length > incumbent.chord_length + 1.0e-9
                    || ((candidate.chord_length - incumbent.chord_length).abs() <= 1.0e-9
                        && candidate.length > incumbent.length)
            });
            if replace {
                best = Some(candidate);
            }
        }
    }
    best
}

/// A conservative, orientation-independent curvature veto.  `reference` is
/// intentionally only a crude scale observation: by allowing a 0.30 axis
/// ratio, the comparison admits a much more oblique projected iris than is
/// usual in a useful eye ROI.  Even under that permissive projection, a long
/// ellipse arc has a minimum sagitta.  A contour run substantially straighter
/// than that cannot belong to an iris of this apparent size.
pub(crate) fn best_impossible_conic_run(
    smoothed: &[(f64, f64)],
    reference: Ellipse,
    already_censored: &[bool],
) -> Option<FlatTireRun> {
    let count = smoothed.len();
    if count < 24 || already_censored.len() != count {
        return None;
    }
    let apparent_radius = reference
        .major_radius
        .clamp(30.0, FRAME_WIDTH as f64 * 0.60);
    let maximum_plausible_curvature_radius = apparent_radius / 0.30;
    let minimum_samples = (count / 24).max(6);
    let maximum_samples = count / 2;
    let minimum_chord = (0.34 * apparent_radius).max(24.0);
    let mut best: Option<FlatTireRun> = None;

    for start in 0..count {
        for length in minimum_samples..=maximum_samples {
            // An earlier chord may share its two physical junctions with a
            // second occluder, but its interior must never be rediscovered.
            if (1..length.saturating_sub(1))
                .any(|offset| already_censored[(start + offset) % count])
            {
                continue;
            }
            let first = smoothed[start];
            let last = smoothed[(start + length - 1) % count];
            let chord = (last.0 - first.0, last.1 - first.1);
            let chord_length = chord.0.hypot(chord.1);
            if chord_length < minimum_chord
                || chord_length >= 1.90 * maximum_plausible_curvature_radius
            {
                continue;
            }
            let half_chord = 0.5 * chord_length;
            let minimum_ellipse_sagitta = maximum_plausible_curvature_radius
                - (maximum_plausible_curvature_radius.powi(2) - half_chord.powi(2)).sqrt();
            if minimum_ellipse_sagitta < 2.0 {
                continue;
            }
            // Permit raster stair-steps and a gently bowed real occluder, but
            // demand a large margin below the least-curved plausible iris arc.
            let residual_limit = (0.52 * minimum_ellipse_sagitta).clamp(1.35, 6.0);
            let mut maximum_residual = 0.0f64;
            let mut mean_residual = 0.0f64;
            let mut valid = true;
            for offset in 0..length {
                let point = smoothed[(start + offset) % count];
                let residual = ((point.0 - first.0) * chord.1 - (point.1 - first.1) * chord.0)
                    .abs()
                    / chord_length.max(1.0e-9);
                maximum_residual = maximum_residual.max(residual);
                mean_residual += residual;
                if maximum_residual > residual_limit {
                    valid = false;
                    break;
                }
            }
            mean_residual /= length as f64;
            if !valid || mean_residual > residual_limit * 0.55 {
                continue;
            }
            let candidate = FlatTireRun {
                side: FlatTireSide::Other,
                start,
                length,
                chord_length,
            };
            let replace = best.is_none_or(|incumbent| {
                candidate.chord_length > incumbent.chord_length + 1.0e-9
                    || ((candidate.chord_length - incumbent.chord_length).abs() <= 1.0e-9
                        && candidate.length > incumbent.length)
            });
            if replace {
                best = Some(candidate);
            }
        }
    }
    best
}

pub(crate) fn deflattened_mask_fit_with_context(
    contour: Vec<(f64, f64)>,
    reference: Ellipse,
    scale_context: Option<OuterContourScaleContext>,
) -> Option<ContourFitEvidence> {
    deflattened_mask_fit_with_noise(contour, reference, scale_context, 1.0, true)
}

pub(crate) fn deflattened_mask_fit_with_noise(
    contour: Vec<(f64, f64)>,
    reference: Ellipse,
    scale_context: Option<OuterContourScaleContext>,
    pixel_scale: f64,
    constrain_arcs: bool,
) -> Option<ContourFitEvidence> {
    // Preserve boundary order: polar sorting can jump between the true limbus
    // and an occluding lid chord and manufacture exactly the flattened conic
    // this stage is intended to reject.
    let samples = sample_closed_contour(&contour, 128);
    if samples.len() < 24 {
        return None;
    }
    let smoothed = smooth_closed_contour(&samples, 2);
    let upper = best_flat_tire_run(&smoothed, reference, FlatTireSide::Upper);
    let lower = best_flat_tire_run(&smoothed, reference, FlatTireSide::Lower);
    let mut flat_tire = vec![false; samples.len()];
    for run in [upper, lower].into_iter().flatten() {
        // Keep each junction sample: it is normally the last directly visible
        // limbus point on either side.  Only the chord interior is censored.
        for offset in 1..run.length.saturating_sub(1) {
            flat_tire[(run.start + offset) % samples.len()] = true;
        }
    }
    // A selected outer-disk component can still be clipped by a nose, jaw,
    // glasses rim, image boundary, or unrelated foreground edge.  Censor up
    // to four disjoint physically impossible-curvature runs, regardless of
    // their position or orientation.  A mostly polygonal false component is
    // consequently left without enough arc support and fails closed.
    for _ in 0..4 {
        let Some(run) = best_impossible_conic_run(&smoothed, reference, &flat_tire) else {
            break;
        };
        for offset in 1..run.length.saturating_sub(1) {
            flat_tire[(run.start + offset) % samples.len()] = true;
        }
    }
    let retained_indices = (0..samples.len())
        .filter(|&i| !flat_tire[i])
        .collect::<Vec<_>>();
    let retained = retained_indices
        .iter()
        .map(|&i| samples[i])
        .collect::<Vec<_>>();
    if retained.len() < 20 {
        return None;
    }
    let constraints = ConicArcConstraints {
        tangents: (0..samples.len())
            .filter(|&i| !flat_tire[i])
            .map(|i| {
                let n = samples.len();
                if (-2isize..=2)
                    .any(|d| flat_tire[(i as isize + d).rem_euclid(n as isize) as usize])
                {
                    return None;
                }
                let a = smoothed[(i + n - 2) % n];
                let b = smoothed[(i + 2) % n];
                Some((b.0 - a.0, b.1 - a.1))
            })
            .collect(),
    };
    let mut random = NumpyPcg64::baseline_fit_stream();
    let robust = constrained_ransac_ellipse(
        &retained,
        &mut random,
        scale_context,
        pixel_scale,
        constrain_arcs.then_some(&constraints),
    )?;
    // The adaptive RANSAC cutoff is useful for locating a basin but must not
    // turn every non-flat part of an eyelid-shaped mask into usable limbus.
    // Reclassify against a pixel-scale ceiling and require local contour
    // continuity.  This classification is frame-local and therefore
    // defeasible on the next exposure.
    let strict_cutoff = robust.cutoff.min(4.0 * pixel_scale);
    let mut usable = retained
        .iter()
        .enumerate()
        .map(|(i, &point)| {
            ellipse_residual(point, robust.ellipse) <= strict_cutoff
                && (!constrain_arcs || constraints.tangent_agrees(i, point, robust.ellipse))
        })
        .collect::<Vec<_>>();
    if usable.len() >= 3 {
        let snapshot = usable.clone();
        for index in 0..usable.len() {
            let previous = (index + usable.len() - 1) % usable.len();
            let next = (index + 1) % usable.len();
            let has_previous = snapshot[previous]
                && (retained_indices[previous] + 1) % samples.len() == retained_indices[index];
            let has_next = snapshot[next]
                && (retained_indices[index] + 1) % samples.len() == retained_indices[next];
            if snapshot[index] && !has_previous && !has_next {
                usable[index] = false;
            }
        }
    }
    let inlier_points = retained
        .iter()
        .zip(usable.iter())
        .filter_map(|(&point, &inlier)| inlier.then_some(point))
        .collect::<Vec<_>>();
    if inlier_points.len() < 12
        || (constrain_arcs
            && !constraints.admits(&retained, &usable, robust.ellipse, strict_cutoff))
    {
        return None;
    }
    // A final numerical polish must obey the same support constraints as the
    // search. It may remove support, but must never reinstate excluded points.
    let final_support = |ellipse| {
        retained
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                usable[i]
                    && ellipse_residual(p, ellipse) <= strict_cutoff
                    && (!constrain_arcs || constraints.tangent_agrees(i, p, ellipse))
            })
            .collect::<Vec<_>>()
    };
    let ellipse = robust_contour_fit(&inlier_points, robust.ellipse)
        .filter(|ellipse| scale_context.is_none_or(|context| context.admits(*ellipse)))
        .filter(|ellipse| {
            !constrain_arcs
                || constraints.admits(&retained, &final_support(*ellipse), *ellipse, strict_cutoff)
        })
        .unwrap_or(robust.ellipse);
    if !plausible_ellipse(ellipse) || scale_context.is_some_and(|context| !context.admits(ellipse))
    {
        return None;
    }
    let supported = final_support(ellipse);
    let retained_points = retained
        .iter()
        .zip(&supported)
        .filter_map(|(&p, &keep)| keep.then_some(p))
        .collect();
    let mut final_kept = vec![false; samples.len()];
    for (&index, &keep) in retained_indices.iter().zip(&supported) {
        final_kept[index] = keep;
    }
    let flat_tire_points = samples
        .iter()
        .zip(final_kept.iter())
        .filter_map(|(&point, &keep)| (!keep).then_some(point))
        .collect::<Vec<_>>();
    Some(ContourFitEvidence {
        ellipse,
        source_component_area_px: 0.0,
        retained_points: Arc::new(retained_points),
        conic_segments: Arc::new(supported_conic_runs(&final_kept)),
        flat_tire_points: Arc::new(flat_tire_points),
        upper_flat_tire: upper.is_some_and(|run| run.side == FlatTireSide::Upper),
        lower_flat_tire: lower.is_some_and(|run| run.side == FlatTireSide::Lower),
    })
}

pub(crate) fn deflattened_mask_fit(
    contour: Vec<(f64, f64)>,
    reference: Ellipse,
) -> Option<ContourFitEvidence> {
    deflattened_mask_fit_with_context(contour, reference, None)
}
