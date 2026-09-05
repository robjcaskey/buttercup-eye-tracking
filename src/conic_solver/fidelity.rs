//! Local, conditional translation information from sparse conic support.
//!
//! This is NOT a pose/sign probability or the covariance of a free ellipse fit.
//! Shape, boundary association and source alignment are conditioned on here.
//! A joint solver must marginalize their uncertainty before claiming absolute
//! accuracy. The live legacy fitter does not consume this helper yet.

use crate::geometry::Ellipse;
use crate::roi_evidence::RoiConicEvidence;
use std::collections::HashMap;

/// Engineering noise assumptions, expressed in the evidence's native pixels.
/// Detail is optical edge reliability, not lens actuator position, SAM score,
/// geometric residual, or probability of anatomical correctness.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LocalizationHeuristic {
    pub(crate) sharp_edge_sigma_px: f64,
    pub(crate) additional_defocus_sigma_px: f64,
    pub(crate) unknown_detail_reliability: f64,
    /// Shared model-error floor prevents dense/coherent arcs from implying
    /// unlimited precision. Its value requires corpus calibration.
    pub(crate) model_floor_px: f64,
}

impl LocalizationHeuristic {
    pub(crate) fn sigma_px(self, detail: Option<f64>) -> Option<f64> {
        let detail = detail.unwrap_or(self.unknown_detail_reliability);
        if !self.sharp_edge_sigma_px.is_finite()
            || self.sharp_edge_sigma_px <= 0.0
            || !self.additional_defocus_sigma_px.is_finite()
            || self.additional_defocus_sigma_px < 0.0
            || !detail.is_finite()
            || !(0.0..=1.0).contains(&detail)
            || !self.model_floor_px.is_finite()
            || self.model_floor_px < 0.0
        {
            return None;
        }
        Some(
            self.sharp_edge_sigma_px
                .hypot(self.additional_defocus_sigma_px * (1.0 - detail)),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ConditionalTranslationFidelity {
    /// Covariance proxy in px², CONDITIONAL on conic shapes/associations.
    /// Includes the declared isotropic model floor; not calibrated coverage.
    pub(crate) covariance_px2: [[f64; 2]; 2],
    pub(crate) localization_sigma_px: f64,
    pub(crate) independent_groups: usize,
}

impl ConditionalTranslationFidelity {
    pub(crate) fn worst_axis_sigma_px(self) -> f64 {
        let c = self.covariance_px2;
        let discriminant = (c[0][0] - c[1][1]).hypot(2.0 * c[0][1]);
        (0.5 * (c[0][0] + c[1][1] + discriminant)).sqrt()
    }
}

/// Unit normal of the implicit ellipse at a sparse support point.
pub(super) fn conic_normal(point: (f64, f64), ellipse: Ellipse) -> Option<[f64; 2]> {
    if !point.0.is_finite()
        || !point.1.is_finite()
        || !ellipse.center.0.is_finite()
        || !ellipse.center.1.is_finite()
        || !ellipse.angle.is_finite()
        || !ellipse.major_radius.is_finite()
        || !ellipse.minor_radius.is_finite()
        || ellipse.major_radius <= 0.0
        || ellipse.minor_radius <= 0.0
    {
        return None;
    }
    let (sin, cos) = ellipse.angle.sin_cos();
    let dx = point.0 - ellipse.center.0;
    let dy = point.1 - ellipse.center.1;
    let x = (cos * dx + sin * dy) / ellipse.major_radius.powi(2);
    let y = (-sin * dx + cos * dy) / ellipse.minor_radius.powi(2);
    let normal = [cos * x - sin * y, sin * x + cos * y];
    let length = normal[0].hypot(normal[1]);
    (length.is_finite() && length > 1e-12).then(|| [normal[0] / length, normal[1] / length])
}

pub(super) fn inverse_symmetric_2x2(m: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let determinant = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let trace = m[0][0] + m[1][1];
    // Relative conditioning check; avoid a spurious precise inverse from a
    // single tangent direction or a nearly singular short arc.
    if !trace.is_finite()
        || trace <= 0.0
        || !determinant.is_finite()
        || determinant <= trace * trace * 1e-8
    {
        return None;
    }
    Some([
        [m[1][1] / determinant, -m[0][1] / determinant],
        [-m[1][0] / determinant, m[0][0] / determinant],
    ])
}

/// Each evidence group has ONE total normal-information budget, irrespective
/// of contour resampling density. Derived conics and repeated support indices
/// cannot add independent measurements. Correlated alternatives reuse a group;
/// inconsistent associations are refused instead of selecting the most certain.
/// This helper evaluates ONE selected system: unresolved alternatives within
/// the same group must be separated into hypotheses by its caller.
pub(crate) fn estimate_conditional_translation_fidelity(
    evidence: RoiConicEvidence<'_>,
    heuristic: LocalizationHeuristic,
) -> Option<ConditionalTranslationFidelity> {
    let sigma = heuristic.sigma_px(evidence.detail_reliability)?;
    let mut used = HashMap::new();
    let mut information = [[0.0; 2]; 2];
    for conic in evidence.conics {
        for &index in conic.supporting_arc_indices {
            let arc = evidence.arcs.get(index)?;
            if arc.kind != conic.kind {
                return None;
            }
            if arc.points_roi_px.is_empty() {
                continue;
            }
            let association = (conic.kind, conic.ellipse_roi_px, arc.points_roi_px);
            if let Some(previous) = used.insert(arc.evidence_group, association) {
                if previous != association {
                    return None;
                }
                continue;
            }
            let weight = 1.0 / (arc.points_roi_px.len() as f64 * sigma * sigma);
            for &point in arc.points_roi_px {
                let normal = conic_normal(point, conic.ellipse_roi_px)?;
                for row in 0..2 {
                    for column in 0..2 {
                        information[row][column] += weight * normal[row] * normal[column];
                    }
                }
            }
        }
    }
    let mut covariance_px2 = inverse_symmetric_2x2(information)?;
    for axis in 0..2 {
        covariance_px2[axis][axis] += heuristic.model_floor_px.powi(2);
    }
    if !covariance_px2
        .iter()
        .flatten()
        .all(|value| value.is_finite())
    {
        return None;
    }
    Some(ConditionalTranslationFidelity {
        covariance_px2,
        localization_sigma_px: sigma,
        independent_groups: used.len(),
    })
}
