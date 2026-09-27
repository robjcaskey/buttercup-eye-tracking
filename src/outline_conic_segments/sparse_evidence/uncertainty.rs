//! Optical allowances from current native RAW at measured contour positions.
//!
//! A fitted ellipse, another exposure, and the solver residual are never inputs.
//! These bounded profile measurements do not establish that an edge is an iris
//! boundary. They describe its local optical support under an engineering noise
//! model; semantic mistakes and circle-model error remain possible.
use super::{luma, OwnedRoiEvidence};
use crate::roi_evidence::{BoundaryKind, BoundaryNormalObservation};

#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct PupilDirectionReport {
    pub(crate) samples: usize,
    pub(crate) measured: usize,
    pub(crate) angular_sigmas_radians: Vec<f64>,
    pub(crate) support_caps: Vec<PupilSupportCap>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct PupilSupportCap {
    pub(crate) group: u32,
    pub(crate) polyline_length_px: f64,
    pub(crate) tangent_length_cap_px: f64,
}

/// Explicit offline weight ablation, not a claim that less RAW was captured.
/// Reduce only pupil information mass; retain positions, directions, optical
/// widths, source identity and every other anatomical boundary unchanged.
pub(crate) fn scale_pupil_information(packet: &mut OwnedRoiEvidence, scale: f64) {
    assert!(scale.is_finite() && scale > 0.0 && scale <= 1.0);
    if scale == 1.0 {
        return;
    }
    for arc in packet
        .arcs
        .iter_mut()
        .filter(|a| a.kind == BoundaryKind::PupillaryBoundary)
    {
        let length = arc
            .points_roi_px
            .windows(2)
            .map(|p| (p[1].0 - p[0].0).hypot(p[1].1 - p[0].1))
            .sum::<f64>();
        if length.is_finite() && length > 0.0 {
            arc.support_length_cap_px = Some(
                arc.support_length_cap_px
                    .map_or(length, |cap| cap.min(length))
                    * scale,
            );
        }
    }
}

/// Bound support by displacement along the RAW edge tangent. Use the largest
/// projection allowed by either endpoint's +/-2-sigma orientation cone. An
/// unknown direction keeps the original segment budget. The cone is the same
/// engineering allowance as the direction residual, not calibrated coverage.
pub(crate) fn cap_pupil_tangent_support(packet: &mut OwnedRoiEvidence) -> Vec<PupilSupportCap> {
    let mut report = Vec::new();
    for arc in packet
        .arcs
        .iter_mut()
        .filter(|a| a.kind == BoundaryKind::PupillaryBoundary)
    {
        let Some(normals) = &arc.outward_normals_roi else {
            continue;
        };
        if normals.len() != arc.points_roi_px.len() {
            continue;
        }
        let mut polyline = 0.0;
        let mut tangent = 0.0;
        for (points, ns) in arc.points_roi_px.windows(2).zip(normals.windows(2)) {
            let d = [points[1].0 - points[0].0, points[1].1 - points[0].1];
            let length = d[0].hypot(d[1]);
            polyline += length;
            let cap = if length > 1.0e-9 {
                ns[0]
                    .zip(ns[1])
                    .map(|(a, b)| {
                        [a, b]
                            .map(|n| {
                                let projection = (d[0] * n.unit_outward_roi[1]
                                    - d[1] * n.unit_outward_roi[0])
                                    .abs()
                                    / length;
                                let angle = projection.clamp(0.0, 1.0).acos();
                                length * (angle - 2.0 * n.angular_sigma_radians).max(0.0).cos()
                            })
                            .into_iter()
                            .fold(0.0_f64, f64::max)
                    })
                    .unwrap_or(length)
            } else {
                length
            };
            tangent += cap;
        }
        if !polyline.is_finite() || polyline <= 1.0e-9 {
            continue;
        }
        let cap = tangent.min(polyline);
        arc.support_length_cap_px = Some(
            arc.support_length_cap_px
                .map_or(cap, |previous| previous.min(cap)),
        );
        report.push(PupilSupportCap {
            group: arc.evidence_group,
            polyline_length_px: polyline,
            tangent_length_cap_px: arc.support_length_cap_px.unwrap(),
        });
    }
    report
}

/// Current-image gradient at an observed position, without a conic/tangent
/// supplied by a fit. The five overlapping stencils are NOT independent votes.
/// Their full angular scatter widens the engineering allowance; no sqrt(N)
/// precision gain is claimed. Dark-to-bright RAW polarity sets the direction.
fn raw_edge_direction(
    raw: &[u16],
    width: usize,
    height: usize,
    point: (f64, f64),
    ceiling: f64,
) -> Option<BoundaryNormalObservation> {
    let sample = |x, y| luma(raw, width, height, x, y).filter(|v| *v <= ceiling);
    let mut gradients = [[0.0; 2]; 5];
    for (g, (dx, dy)) in
        gradients
            .iter_mut()
            .zip([(0.0, 0.0), (-4.0, 0.0), (4.0, 0.0), (0.0, -4.0), (0.0, 4.0)])
    {
        let (x, y) = (point.0 + dx, point.1 + dy);
        *g = [
            (sample(x + 3.0, y)? - sample(x - 3.0, y)?) / 6.0,
            (sample(x, y + 3.0)? - sample(x, y - 3.0)?) / 6.0,
        ];
    }
    let mean =
        std::array::from_fn::<_, 2, _>(|i| gradients.iter().map(|g| g[i]).sum::<f64>() / 5.0);
    let magnitude = mean[0].hypot(mean[1]);
    if magnitude < 7.0 / 6.0 {
        return None;
    }
    let normal = [mean[0] / magnitude, mean[1] / magnitude];
    let transverse_scatter = (gradients
        .iter()
        .map(|g| (g[0] * normal[1] - g[1] * normal[0]).powi(2))
        .sum::<f64>()
        / 5.0)
        .sqrt();
    let sigma = 15.0_f64
        .to_radians()
        .hypot(transverse_scatter.atan2(magnitude));
    if sigma > 60.0_f64.to_radians() {
        return None;
    }
    Some(BoundaryNormalObservation {
        unit_outward_roi: normal,
        angular_sigma_radians: sigma,
    })
}

/// Explicit offline pupil-direction experiment. Preserve every measured point,
/// alternative, source key, positional allowance and evidence group. Unknown or
/// glint-censored stencils add no direction; they do not delete position data.
/// No outer/true-inner observations or existing measured directions are changed.
pub(crate) fn measure_pupil_directions(
    packet: &mut OwnedRoiEvidence,
    raw: &[u16],
    maximum_luma_raw10: Option<f64>,
) -> PupilDirectionReport {
    let mut report = PupilDirectionReport::default();
    let [width, height] = packet.dimensions_px.map(|n| n as usize);
    let Some(ceiling) = maximum_luma_raw10
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| v.min(990.0))
    else {
        return report;
    };
    if width < 20 || height < 20 || width.checked_mul(height) != Some(raw.len()) {
        return report;
    }
    for arc in packet
        .arcs
        .iter_mut()
        .filter(|a| a.kind == BoundaryKind::PupillaryBoundary && a.outward_normals_roi.is_none())
    {
        report.samples += arc.points_roi_px.len();
        let directions = arc
            .points_roi_px
            .iter()
            .map(|&p| raw_edge_direction(raw, width, height, p, ceiling))
            .collect::<Vec<_>>();
        for direction in directions.iter().flatten() {
            report.measured += 1;
            report
                .angular_sigmas_radians
                .push(direction.angular_sigma_radians);
        }
        if directions.iter().any(Option::is_some) {
            arc.outward_normals_roi = Some(directions);
        }
    }
    report
}

const UNKNOWN_SIGMA_PX: f64 = 12.0;

fn profile_sigma(
    raw: &[u16],
    width: usize,
    height: usize,
    point: (f64, f64),
    tangent: (f64, f64),
) -> Option<f64> {
    let length = tangent.0.hypot(tangent.1);
    if !length.is_finite() || length < 1.0 {
        return None;
    }
    let t = [tangent.0 / length, tangent.1 / length];
    let n = [t[1], -t[0]];
    let sample = |offset: f64, along: f64| {
        luma(
            raw,
            width,
            height,
            point.0 + offset * n[0] + along * t[0],
            point.1 + offset * n[1] + along * t[1],
        )
        .filter(|v| *v < 990.0)
    };
    let mut contrast = [0.0_f64; 13];
    for (i, value) in contrast.iter_mut().enumerate() {
        let offset = (i as f64 - 6.0) * 2.0;
        // Contour ordering may differ by detector. Polarity/semantic admission
        // belongs to extraction, not this uncertainty-only measurement.
        *value = (sample(offset + 3.0, 0.0)? - sample(offset - 3.0, 0.0)?).abs();
    }
    let peak = (0..contrast.len()).max_by(|&a, &b| contrast[a].total_cmp(&contrast[b]))?;
    let signal = contrast[peak];
    if peak == 0 || peak + 1 == contrast.len() || signal < 7.0 {
        return None;
    }
    let mut lo = peak;
    let mut hi = peak;
    while lo > 0 && contrast[lo - 1] >= 0.5 * signal {
        lo -= 1;
    }
    while hi + 1 < contrast.len() && contrast[hi + 1] >= 0.5 * signal {
        hi += 1;
    }
    if lo == 0 || hi + 1 == contrast.len() {
        return None; // Blur extends outside the measurement window.
    }
    // A second plausible transition also makes position ambiguous. Do not
    // report the width of just the strongest glint/texture peak as certainty.
    let extent_lo = contrast.iter().position(|v| *v >= 0.5 * signal)?;
    let extent_hi = contrast.iter().rposition(|v| *v >= 0.5 * signal)?;
    let width_px = (extent_hi - extent_lo + 1) as f64 * 2.0;
    let offset_px = (peak as f64 - 6.0).abs() * 2.0;
    // Tangential variation of native cell means is a conservative texture/noise
    // allowance, not a sensor read-noise calibration or a model training score.
    let variability = [-8.0, 8.0]
        .into_iter()
        .map(|offset| Some((sample(offset, -4.0)? - sample(offset, 4.0)?).abs()))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .sum::<f64>()
        * 0.5;
    let noise_ratio = (2.0 + variability) / signal;
    Some(
        0.75_f64
            .hypot(0.25 * width_px * (1.0 + noise_ratio))
            .hypot(offset_px)
            .min(UNKNOWN_SIGMA_PX),
    )
}

/// Update only per-arc optical allowances. Preserve positions, alternatives,
/// sample counts, conic hints, full-ROI detail and source identity exactly.
/// Unknown/censored profiles weaken an arc rather than silently removing it
/// or treating it as a sharp observation. Missing RAW leaves the legacy
/// unknown-optics fallback in place.
pub(crate) fn measure(packet: &mut OwnedRoiEvidence, raw: &[u16]) {
    let [width, height] = packet.dimensions_px.map(|v| v as usize);
    if width < 12 || height < 12 || width.checked_mul(height) != Some(raw.len()) {
        return;
    }
    for arc in &mut packet.arcs {
        let points = &arc.points_roi_px;
        if points.len() < 3 {
            continue;
        }
        let count = points.len().min(16);
        let mut sigmas = (0..count)
            .map(|i| {
                let at = i * (points.len() - 1) / (count - 1);
                let a = points[at.saturating_sub(1)];
                let b = points[(at + 1).min(points.len() - 1)];
                profile_sigma(raw, width, height, points[at], (b.0 - a.0, b.1 - a.1))
                    .unwrap_or(UNKNOWN_SIGMA_PX)
            })
            .collect::<Vec<_>>();
        sigmas.sort_by(f64::total_cmp);
        // A few sharp samples must not certify an otherwise obscured run.
        arc.localization_sigma_px = Some(sigmas[(3 * (sigmas.len() - 1)).div_ceil(4)]);
    }
}

/// Diagnostic receipt for the explicit outer-only RAW spread experiment.
/// Unknown measurements retain the existing engineering allowance; they are
/// not observations of a sharp edge or evidence of calibrated precision.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct OuterSpreadArc {
    pub(crate) arc_index: usize,
    pub(crate) measured_profiles: usize,
    pub(crate) requested_profiles: usize,
    pub(crate) previous_optical_sigma_px: f64,
    pub(crate) applied_optical_sigma_px: Option<f64>,
    pub(crate) profile_rms_spread_px: Vec<f64>,
}

// Finite-window transition spread around an observed contour point. The
// derivative's second moment includes displacement from that point. It is an
// engineering support allowance, not the variance of an unbiased edge fit.
fn outer_profile_spread(
    raw: &[u16],
    width: usize,
    height: usize,
    point: (f64, f64),
    tangent: (f64, f64),
) -> Option<f64> {
    let length = tangent.0.hypot(tangent.1);
    if !length.is_finite() || length < 1.0 {
        return None;
    }
    let t = [tangent.0 / length, tangent.1 / length];
    let n = [t[1], -t[0]];
    let mut profile = [0.0; 17];
    for (i, value) in profile.iter_mut().enumerate() {
        let offset = 2.0 * i as f64 - 16.0;
        let mut samples = [0.0; 3];
        for (sample, along) in samples.iter_mut().zip([-4.0, 0.0, 4.0]) {
            *sample = luma(
                raw,
                width,
                height,
                point.0 + offset * n[0] + along * t[0],
                point.1 + offset * n[1] + along * t[1],
            )
            .filter(|v| *v < 990.0)?;
        }
        samples.sort_by(f64::total_cmp);
        *value = samples[1];
    }
    let rise = profile[16] - profile[0];
    let variation = profile.windows(2).map(|p| (p[1] - p[0]).abs()).sum::<f64>();
    if rise.abs() < 7.0 || variation <= 0.0 || rise.abs() / variation < 0.75 {
        return None;
    }
    let mut mass = 0.0;
    let mut second_moment = 0.0;
    for (i, p) in profile.windows(2).enumerate() {
        let weight = ((p[1] - p[0]) * rise.signum()).max(0.0);
        let midpoint = 2.0 * i as f64 - 15.0;
        mass += weight;
        second_moment += weight * midpoint * midpoint;
    }
    (mass > 0.0).then(|| (second_moment / mass).sqrt())
}

/// Explicit offline candidate: widen only outer-limbus allowances using
/// measurable RAW transition spread. Preserve pupil/true-inner factors, all
/// source coordinates, conic hints and correlation groups. No joint result,
/// fitted-ellipse normal, temporal source or full-ROI score is consulted.
/// Finite sampling can miss optical/semantic errors; unknown is reported and
/// keeps the baseline allowance rather than being assigned a made-up width.
pub(crate) fn measure_outer_spread(
    packet: &mut OwnedRoiEvidence,
    raw: &[u16],
) -> Vec<OuterSpreadArc> {
    let [width, height] = packet.dimensions_px.map(|v| v as usize);
    if width < 12 || height < 12 || width.checked_mul(height) != Some(raw.len()) {
        return Vec::new();
    }
    let detail = packet
        .detail_reliability
        .filter(|d| d.is_finite())
        .unwrap_or(0.25)
        .clamp(0.0, 1.0);
    let fallback = 0.75_f64.hypot(4.0 * (1.0 - detail));
    let mut report = Vec::new();
    for (arc_index, arc) in packet.arcs.iter_mut().enumerate() {
        if arc.kind != crate::roi_evidence::BoundaryKind::OuterLimbus || arc.points_roi_px.len() < 3
        {
            continue;
        }
        let points = &arc.points_roi_px;
        let count = points.len().min(16);
        let mut widths = (0..count)
            .filter_map(|i| {
                let at = i * (points.len() - 1) / (count - 1);
                let a = points[at.saturating_sub(1)];
                let b = points[(at + 1).min(points.len() - 1)];
                outer_profile_spread(raw, width, height, points[at], (b.0 - a.0, b.1 - a.1))
            })
            .collect::<Vec<_>>();
        widths.sort_by(f64::total_cmp);
        let previous = arc.localization_sigma_px.unwrap_or(fallback);
        let measured = (widths.len() >= 3 && widths.len() * 2 >= count)
            .then(|| widths[(3 * (widths.len() - 1)).div_ceil(4)].max(previous));
        if let Some(sigma) = measured {
            arc.localization_sigma_px = Some(sigma);
        }
        report.push(OuterSpreadArc {
            arc_index,
            measured_profiles: widths.len(),
            requested_profiles: count,
            previous_optical_sigma_px: previous,
            applied_optical_sigma_px: measured,
            profile_rms_spread_px: widths,
        });
    }
    report
}

/// Keep displacement from the measured contour separate from optical width.
/// A gradient centroid is only a local optical landmark, not a semantic label
/// or a calibrated estimate of the anatomical limbus position.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct OuterPositionProfile {
    point_index: usize,
    point_roi_px: (f64, f64),
    centroid_offset_px: f64,
    centered_transition_spread_px: f64,
    contrast_raw10: f64,
    endpoint_gradient_ratio: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct OuterPositionArc {
    pub(crate) arc_index: usize,
    pub(crate) measured_profiles: usize,
    pub(crate) requested_profiles: usize,
    pub(crate) previous_optical_sigma_px: f64,
    pub(crate) applied_optical_sigma_px: Option<f64>,
    profiles: Vec<OuterPositionProfile>,
    profile_failures: std::collections::BTreeMap<&'static str, usize>,
}

fn outer_profile_position(
    raw: &[u16],
    width: usize,
    height: usize,
    point: (f64, f64),
    tangent: (f64, f64),
    point_index: usize,
) -> Result<OuterPositionProfile, &'static str> {
    let length = tangent.0.hypot(tangent.1);
    if !length.is_finite() || length < 1.0 {
        return Err("invalid-tangent");
    }
    let t = [tangent.0 / length, tangent.1 / length];
    let n = [t[1], -t[0]];
    let mut profile = [0.0; 17];
    for (i, value) in profile.iter_mut().enumerate() {
        let offset = 2.0 * i as f64 - 16.0;
        let mut samples = [0.0; 3];
        for (sample, along) in samples.iter_mut().zip([-4.0, 0.0, 4.0]) {
            *sample = luma(
                raw,
                width,
                height,
                point.0 + offset * n[0] + along * t[0],
                point.1 + offset * n[1] + along * t[1],
            )
            .filter(|v| *v < 990.0)
            .ok_or("sampling-boundary-or-saturation")?;
        }
        samples.sort_by(f64::total_cmp);
        *value = samples[1];
    }
    let rise = profile[16] - profile[0];
    let variation = profile.windows(2).map(|p| (p[1] - p[0]).abs()).sum::<f64>();
    if rise.abs() < 7.0 || variation <= 0.0 {
        return Err("insufficient-transition");
    }
    if rise.abs() / variation < 0.75 {
        return Err("nonmonotone-transition");
    }
    let weights = std::array::from_fn::<_, 16, _>(|i| {
        ((profile[i + 1] - profile[i]) * rise.signum()).max(0.0)
    });
    let peak = weights.iter().copied().fold(0.0_f64, f64::max);
    let endpoint_gradient_ratio = weights[0].max(weights[15]) / peak;
    // A cropped ramp can have centroid zero without exposing either plateau.
    // Report it as unknown instead of claiming it is accurately centered.
    if endpoint_gradient_ratio > 0.2 {
        return Err("window-censored-transition");
    }
    let mass = weights.iter().sum::<f64>();
    let centroid_offset_px = weights
        .iter()
        .enumerate()
        .map(|(i, w)| w * (2.0 * i as f64 - 15.0))
        .sum::<f64>()
        / mass;
    let centered_transition_spread_px = (weights
        .iter()
        .enumerate()
        .map(|(i, w)| w * (2.0 * i as f64 - 15.0 - centroid_offset_px).powi(2))
        .sum::<f64>()
        / mass)
        .sqrt();
    Ok(OuterPositionProfile {
        point_index,
        point_roi_px: point,
        centroid_offset_px,
        centered_transition_spread_px,
        contrast_raw10: rise.abs(),
        endpoint_gradient_ratio,
    })
}

/// Offline position-discrepancy experiment. A broad but centered optical edge
/// does not automatically inherit its full transition width as position noise.
/// This is a bounded discrepancy allowance, not a fitted edge covariance; it
/// cannot establish the semantic identity of a transition. Unknown keeps the
/// engineering baseline. Pupil/true-inner evidence and all points stay exact.
pub(crate) fn measure_outer_position(
    packet: &mut OwnedRoiEvidence,
    raw: &[u16],
) -> Vec<OuterPositionArc> {
    let [width, height] = packet.dimensions_px.map(|v| v as usize);
    if width < 12 || height < 12 || width.checked_mul(height) != Some(raw.len()) {
        return Vec::new();
    }
    let detail = packet
        .detail_reliability
        .filter(|d| d.is_finite())
        .unwrap_or(0.25)
        .clamp(0.0, 1.0);
    let fallback = 0.75_f64.hypot(4.0 * (1.0 - detail));
    let mut report = Vec::new();
    for (arc_index, arc) in packet.arcs.iter_mut().enumerate() {
        if arc.kind != crate::roi_evidence::BoundaryKind::OuterLimbus || arc.points_roi_px.len() < 3
        {
            continue;
        }
        let points = &arc.points_roi_px;
        let count = points.len().min(16);
        let mut profiles = Vec::new();
        let mut failures = std::collections::BTreeMap::new();
        for i in 0..count {
            let at = i * (points.len() - 1) / (count - 1);
            let a = points[at.saturating_sub(1)];
            let b = points[(at + 1).min(points.len() - 1)];
            match outer_profile_position(raw, width, height, points[at], (b.0 - a.0, b.1 - a.1), at)
            {
                Ok(profile) => profiles.push(profile),
                Err(reason) => *failures.entry(reason).or_insert(0) += 1,
            }
        }
        let mut offsets = profiles
            .iter()
            .map(|p| p.centroid_offset_px.abs())
            .collect::<Vec<_>>();
        offsets.sort_by(f64::total_cmp);
        let previous = arc.localization_sigma_px.unwrap_or(fallback);
        let measured = (offsets.len() >= 3 && offsets.len() * 2 >= count)
            .then(|| offsets[(3 * (offsets.len() - 1)).div_ceil(4)].max(previous));
        if let Some(sigma) = measured {
            arc.localization_sigma_px = Some(sigma);
        }
        report.push(OuterPositionArc {
            arc_index,
            measured_profiles: profiles.len(),
            requested_profiles: count,
            previous_optical_sigma_px: previous,
            applied_optical_sigma_px: measured,
            profiles,
            profile_failures: failures,
        });
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pupil_weight_ablation_preserves_raw_and_all_other_boundary_weights() {
        let mut p = packet();
        let mut pupil = p.arcs[0].clone();
        pupil.kind = BoundaryKind::PupillaryBoundary;
        pupil.evidence_group = 100;
        pupil.support_length_cap_px = Some(40.0);
        p.arcs.push(pupil);
        let mut inner = p.arcs[0].clone();
        inner.kind = BoundaryKind::InnerLimbus;
        p.arcs.push(inner);
        let original = p.clone();
        scale_pupil_information(&mut p, 1.0);
        assert_eq!(format!("{p:?}"), format!("{original:?}"));
        scale_pupil_information(&mut p, 0.75);
        assert_eq!(p.arcs[1].support_length_cap_px, Some(30.0));
        p.arcs[1].support_length_cap_px = Some(40.0);
        assert_eq!(format!("{p:?}"), format!("{original:?}"));
    }

    #[test]
    fn pupil_tangent_cap_preserves_clean_support_and_limits_cross_edge_motion() {
        let mut p = packet();
        p.arcs[0].kind = BoundaryKind::PupillaryBoundary;
        p.arcs[0].outward_normals_roi = Some(vec![
            Some(BoundaryNormalObservation {
                unit_outward_roi: [1.0, 0.0],
                angular_sigma_radians: 15.0_f64.to_radians(),
            });
            p.arcs[0].points_roi_px.len()
        ]);
        let original = p.clone();
        let clean = cap_pupil_tangent_support(&mut p);
        assert!((clean[0].polyline_length_px - clean[0].tangent_length_cap_px).abs() < 1e-10);
        for (i, point) in p.arcs[0].points_roi_px.iter_mut().enumerate() {
            point.0 += if i % 2 == 0 { 6.0 } else { -6.0 };
        }
        p.arcs[0].support_length_cap_px = None;
        let before = p.clone();
        let capped = cap_pupil_tangent_support(&mut p);
        assert!(capped[0].tangent_length_cap_px < 0.8 * capped[0].polyline_length_px);
        let mut restored = p.clone();
        restored.arcs[0].support_length_cap_px = None;
        assert_eq!(format!("{restored:?}"), format!("{before:?}"));
        // A second application cannot further reduce the same evidence.
        assert_eq!(
            cap_pupil_tangent_support(&mut p)[0].tangent_length_cap_px,
            capped[0].tangent_length_cap_px
        );
        p.arcs[0].points_roi_px.reverse();
        assert!(
            (cap_pupil_tangent_support(&mut p)[0].tangent_length_cap_px
                - capped[0].tangent_length_cap_px)
                .abs()
                < 1e-10
        );
        // Unknown directions retain the measured span, not invented certainty.
        p = original;
        p.arcs[0].outward_normals_roi = Some(vec![None; p.arcs[0].points_roi_px.len()]);
        let unknown = cap_pupil_tangent_support(&mut p);
        assert_eq!(
            unknown[0].polyline_length_px,
            unknown[0].tangent_length_cap_px
        );
        p.arcs[0].kind = BoundaryKind::OuterLimbus;
        assert!(cap_pupil_tangent_support(&mut p).is_empty());
    }

    #[test]
    fn pupil_raw_direction_follows_image_polarity_and_not_polyline_order() {
        let width = 128;
        let height = 128;
        for angle in [0.0_f64, 0.4, 1.2, 2.3] {
            for polarity in [-1.0, 1.0] {
                let n = [angle.cos() * polarity, angle.sin() * polarity];
                let raw = (0..height)
                    .flat_map(|y| {
                        (0..width).map(move |x| {
                            let signed = (x as f64 - 63.5) * n[0] + (y as f64 - 63.5) * n[1];
                            (300.0 + 100.0 * (signed / 3.0).tanh()).round() as u16
                        })
                    })
                    .collect::<Vec<_>>();
                let direction =
                    raw_edge_direction(&raw, width, height, (63.5, 63.5), 800.0).unwrap();
                assert!(direction.valid());
                assert!(
                    direction.unit_outward_roi[0] * n[0] + direction.unit_outward_roi[1] * n[1]
                        > 0.99
                );
                assert!(direction.angular_sigma_radians >= 15.0_f64.to_radians());
            }
        }
        for raw in [vec![300; 128 * 128], vec![1023; 128 * 128]] {
            assert!(raw_edge_direction(&raw, 128, 128, (63.5, 63.5), 800.0).is_none());
        }
    }

    #[test]
    fn pupil_direction_attachment_preserves_positions_other_boundaries_and_source() {
        let mut p = packet();
        let outer = p.arcs[0].clone();
        let mut pupil = outer.clone();
        pupil.kind = BoundaryKind::PupillaryBoundary;
        pupil.evidence_group = 100;
        pupil.points_roi_px = vec![(63.5, 54.0), (63.5, 64.0), (63.5, 74.0)];
        p.arcs.push(pupil);
        let before = p.clone();
        let raw = (0..128 * 128)
            .map(|i| (300.0 + 100.0 * ((i % 128) as f64 / 3.0 - 63.5 / 3.0).tanh()).round() as u16)
            .collect::<Vec<_>>();
        let report = measure_pupil_directions(&mut p, &raw, Some(800.0));
        assert_eq!(report.measured, 3);
        assert_eq!(report.samples, 3);
        assert_eq!(p.exposure, before.exposure);
        assert_eq!(p.detail_reliability, before.detail_reliability);
        assert_eq!(format!("{:?}", p.arcs[0]), format!("{:?}", before.arcs[0]));
        let mut restored = p.clone();
        restored.arcs[1].outward_normals_roi = None;
        assert_eq!(format!("{restored:?}"), format!("{before:?}"));
        let normals = p.arcs[1].outward_normals_roi.clone();
        // Neither a made-up ellipse hint nor reversed arc order changes RAW polarity.
        p.conics.clear();
        p.arcs[1].points_roi_px.reverse();
        p.arcs[1].outward_normals_roi = None;
        measure_pupil_directions(&mut p, &raw, Some(800.0));
        let mut reversed = normals.unwrap();
        reversed.reverse();
        assert_eq!(p.arcs[1].outward_normals_roi.as_ref(), Some(&reversed));
        let mut unknown = before.clone();
        assert_eq!(
            measure_pupil_directions(&mut unknown, &raw, None).measured,
            0
        );
        assert_eq!(format!("{unknown:?}"), format!("{before:?}"));
        assert_eq!(
            measure_pupil_directions(&mut unknown, &[0; 4], Some(800.0)).measured,
            0
        );
        assert_eq!(format!("{unknown:?}"), format!("{before:?}"));
    }
    use crate::outline_conic_segments::sparse_evidence::OwnedBoundaryArc;
    use crate::roi_evidence::{BoundaryKind, ExposureKey, RoiId, SourceClock};

    fn packet() -> OwnedRoiEvidence {
        OwnedRoiEvidence {
            exposure: ExposureKey {
                roi: RoiId(1),
                clock: SourceClock {
                    domain: 1,
                    epoch: 1,
                },
                sequence: 1,
                timestamp_ns: 1,
            },
            dimensions_px: [128, 128],
            sensor_origin_px: [0, 0],
            conics: vec![],
            detail_reliability: None,
            arcs: vec![OwnedBoundaryArc {
                support_length_cap_px: None,
                sampling_support_px: None,
                level_sets_roi: None,
                evidence_group: 0,
                kind: BoundaryKind::OuterLimbus,
                points_roi_px: (0..16).map(|i| (64.0, 32.0 + 4.0 * i as f64)).collect(),
                outward_normals_roi: None,
                localization_sigma_px: None,
                normal_band_half_width_px: 1.5,
                detector_score: None,
            }],
        }
    }
    fn edge(blur: f64, amplitude: f64, offset: f64) -> Vec<u16> {
        (0..128 * 128)
            .map(|i| {
                (150.0
                    + amplitude * (1.0 + (((i % 128) as f64 - 64.0 - offset) / blur).tanh()) * 0.5)
                    as u16
            })
            .collect()
    }
    fn sigma(raw: &[u16]) -> f64 {
        let mut p = packet();
        measure(&mut p, raw);
        p.arcs[0].localization_sigma_px.unwrap()
    }
    #[test]
    fn raw_blur_and_misalignment_weaken_without_moving_points() {
        let sharp = sigma(&edge(0.5, 400.0, 0.0));
        assert!(sharp < 3.0, "{sharp}");
        assert!(sigma(&edge(5.0, 400.0, 0.0)) > sharp);
        assert!(sigma(&edge(0.5, 400.0, 6.0)) > 2.0 * sharp);
        assert!(sigma(&edge(0.5, 5.0, 0.0)) > sharp);
        let mut p = packet();
        let before = p.arcs[0].points_roi_px.clone();
        measure(&mut p, &edge(0.5, 400.0, 0.0));
        assert_eq!(p.arcs[0].points_roi_px, before);
        assert_eq!(p.detail_reliability, None);
    }
    #[test]
    fn unknown_and_saturation_are_not_precise_and_missing_raw_is_unknown() {
        assert_eq!(sigma(&vec![100; 128 * 128]), UNKNOWN_SIGMA_PX);
        assert_eq!(sigma(&vec![1023; 128 * 128]), UNKNOWN_SIGMA_PX);
        let mut p = packet();
        measure(&mut p, &[0; 4]);
        assert_eq!(p.arcs[0].localization_sigma_px, None);
    }
    #[test]
    fn crop_origin_winding_and_other_arc_quality_do_not_change_local_measurement() {
        let raw = edge(0.5, 400.0, 0.0);
        let mut p = packet();
        measure(&mut p, &raw);
        let expected = p.arcs[0].localization_sigma_px;
        p.sensor_origin_px = [1400, 2600];
        p.arcs[0].points_roi_px.reverse();
        let mut other = p.arcs[0].clone();
        other.kind = BoundaryKind::PupillaryBoundary;
        for point in &mut other.points_roi_px {
            point.0 -= 25.0;
        }
        p.arcs.push(other);
        measure(&mut p, &raw);
        assert_eq!(p.arcs[0].localization_sigma_px, expected);
        assert_eq!(p.arcs[1].localization_sigma_px, Some(UNKNOWN_SIGMA_PX));
    }

    #[test]
    fn outer_spread_distinguishes_blur_from_brightness_without_moving_observations() {
        let sample = |blur, amplitude| {
            let mut p = packet();
            let before = format!("{:?}", p.arcs[0].points_roi_px);
            let report = measure_outer_spread(&mut p, &edge(blur, amplitude, 0.0));
            assert_eq!(format!("{:?}", p.arcs[0].points_roi_px), before);
            assert!(report[0].measured_profiles >= 8);
            p.arcs[0].localization_sigma_px.unwrap()
        };
        let bright = sample(0.5, 400.0);
        let dim = sample(0.5, 30.0);
        let blurred = sample(8.0, 400.0);
        assert!(
            (bright - dim).abs() < 0.25,
            "brightness is not blur: {bright} / {dim}"
        );
        assert!(
            blurred > bright * 1.5,
            "optical spread must widen allowance: {bright} / {blurred}"
        );
    }

    #[test]
    fn outer_spread_does_not_transfer_outer_weakness_to_pupil_or_true_inner() {
        let mut p = packet();
        for kind in [BoundaryKind::PupillaryBoundary, BoundaryKind::InnerLimbus] {
            let mut arc = p.arcs[0].clone();
            arc.kind = kind;
            arc.localization_sigma_px = Some(0.8);
            p.arcs.push(arc);
        }
        let before = format!("{:?}", &p.arcs[1..]);
        let exposure = p.exposure;
        let report = measure_outer_spread(&mut p, &edge(8.0, 400.0, 0.0));
        assert_eq!(report.len(), 1);
        assert!(p.arcs[0].localization_sigma_px.unwrap() > 4.0);
        assert_eq!(format!("{:?}", &p.arcs[1..]), before);
        assert_eq!(p.exposure, exposure);
        assert_eq!(p.detail_reliability, None);
    }

    #[test]
    fn outer_spread_reports_unknown_and_preserves_winding_and_existing_allowances() {
        for raw in [vec![100; 128 * 128], vec![1023; 128 * 128]] {
            let mut p = packet();
            let before = format!("{p:?}");
            let report = measure_outer_spread(&mut p, &raw);
            assert_eq!(report[0].applied_optical_sigma_px, None);
            assert_eq!(report[0].measured_profiles, 0);
            assert_eq!(format!("{p:?}"), before);
        }
        let raw = edge(8.0, 400.0, 0.0);
        let mut p = packet();
        let first = measure_outer_spread(&mut p, &raw)[0]
            .applied_optical_sigma_px
            .unwrap();
        p.arcs[0].points_roi_px.reverse();
        p.sensor_origin_px = [1400, 2600];
        let second = measure_outer_spread(&mut p, &raw)[0]
            .applied_optical_sigma_px
            .unwrap();
        assert!((first - second).abs() < 1e-12);
        p.arcs[0].localization_sigma_px = Some(20.0);
        assert_eq!(
            measure_outer_spread(&mut p, &raw)[0].applied_optical_sigma_px,
            Some(20.0)
        );
        let before = format!("{p:?}");
        assert!(measure_outer_spread(&mut p, &[0; 4]).is_empty());
        assert_eq!(format!("{p:?}"), before);
    }

    #[test]
    fn outer_position_separates_centered_blur_and_brightness_from_displacement() {
        for (blur, amplitude) in [(0.5, 400.0), (0.5, 30.0), (8.0, 400.0)] {
            let mut p = packet();
            p.detail_reliability = Some(1.0);
            let before = p.arcs[0].points_roi_px.clone();
            let report = measure_outer_position(&mut p, &edge(blur, amplitude, 0.0));
            assert_eq!(report[0].applied_optical_sigma_px, Some(0.75), "{report:?}");
            // RAW10 quantization shifts the dim synthetic centroid by 0.267 px.
            // It remains inside the existing 0.75 px engineering allowance;
            // the optical width must not replace that allowance.
            assert!(
                report[0]
                    .profiles
                    .iter()
                    .all(|v| v.centroid_offset_px.abs() < 0.75),
                "{report:?}"
            );
            if blur > 4.0 {
                assert!(report[0].profiles[0].centered_transition_spread_px > 5.0);
            }
            assert_eq!(p.arcs[0].points_roi_px, before);
        }
        let mut p = packet();
        let bright = measure_outer_position(&mut p, &edge(0.5, 400.0, 8.0))[0]
            .applied_optical_sigma_px
            .unwrap();
        let mut dim = packet();
        let dim = measure_outer_position(&mut dim, &edge(0.5, 30.0, 8.0))[0]
            .applied_optical_sigma_px
            .unwrap();
        assert!(
            (bright - 8.0).abs() < 0.2 && (dim - bright).abs() < 0.2,
            "{bright} / {dim}"
        );
    }

    #[test]
    fn outer_position_preserves_independent_inner_evidence_and_native_identity() {
        let mut p = packet();
        for kind in [BoundaryKind::PupillaryBoundary, BoundaryKind::InnerLimbus] {
            let mut arc = p.arcs[0].clone();
            arc.kind = kind;
            arc.localization_sigma_px = Some(0.8);
            p.arcs.push(arc);
        }
        let before = format!("{:?}", &p.arcs[1..]);
        let exposure = p.exposure;
        let raw = edge(0.5, 400.0, 8.0);
        let first = measure_outer_position(&mut p, &raw)[0]
            .applied_optical_sigma_px
            .unwrap();
        assert!(first > 7.0);
        assert_eq!(format!("{:?}", &p.arcs[1..]), before);
        assert_eq!(p.exposure, exposure);
        assert_eq!(p.detail_reliability, None);
        p.arcs[0].points_roi_px.reverse();
        p.sensor_origin_px = [1400, 2600];
        let second = measure_outer_position(&mut p, &raw)[0]
            .applied_optical_sigma_px
            .unwrap();
        assert!((first - second).abs() < 1e-12);
        p.arcs[0].localization_sigma_px = Some(20.0);
        assert_eq!(
            measure_outer_position(&mut p, &raw)[0].applied_optical_sigma_px,
            Some(20.0)
        );
    }

    #[test]
    fn outer_position_rejects_censored_ramps_missing_signal_and_saturation() {
        let ramp = (0..128 * 128)
            .map(|i| 150 + 3 * (i % 128) as u16)
            .collect::<Vec<_>>();
        for raw in [vec![100; 128 * 128], vec![1023; 128 * 128], ramp] {
            let mut p = packet();
            let before = format!("{p:?}");
            let report = measure_outer_position(&mut p, &raw);
            assert_eq!(report[0].applied_optical_sigma_px, None);
            assert_eq!(report[0].measured_profiles, 0);
            assert_eq!(report[0].profile_failures.values().sum::<usize>(), 16);
            assert_eq!(format!("{p:?}"), before);
        }
        let mut p = packet();
        let before = format!("{p:?}");
        assert!(measure_outer_position(&mut p, &[0; 4]).is_empty());
        assert_eq!(format!("{p:?}"), before);
    }
}
