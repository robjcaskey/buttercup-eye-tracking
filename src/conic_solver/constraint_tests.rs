//! Executable numerical examples for the FUTURE joint solver's contracts.
//!
//! These tests use sparse synthetic projections and explicit noise/scene
//! assumptions, not recorded eye accuracy. The finite hypothesis enumerator,
//! translation fitter and Gaussian fusion below are test fixtures, NOT the
//! still-unimplemented live joint solver. Fidelity and timing helpers are the
//! shared implementation under test. In particular, a pupil arc only signs a
//! tilt here given bounded relative-depth/decentration assumptions.

use super::ellipse_residual;
use super::fidelity::{
    conic_normal, estimate_conditional_translation_fidelity, inverse_symmetric_2x2,
    ConditionalTranslationFidelity, LocalizationHeuristic,
};
use crate::geometry::Ellipse;
use crate::roi_evidence::timing::{RoiBufferTiming, SensorReadTiming, TimestampBandNs};
use crate::roi_evidence::{
    BoundaryArcObservation, BoundaryKind, ConicObservation, ExposureKey, RoiConicEvidence, RoiId,
    SourceClock,
};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

fn clock() -> SourceClock {
    SourceClock {
        domain: 5,
        epoch: 7,
    }
}

fn exposure(eye: u32) -> ExposureKey {
    ExposureKey {
        roi: RoiId(eye),
        clock: clock(),
        sequence: 100 + eye as u64,
        timestamp_ns: 1_000_000_000,
    }
}

// Weak perspective; +y image-down, +z toward the camera. Both signs are
// camera-facing (cos(theta)>0), not a concave/convex choice. depth_px is an
// explicit inward displacement from the limbus plane, NOT a measured universal
// anatomical constant. A movable pivot could be c - slice_depth * normal.
fn projected_ring(tilt_degrees: f64, radius_px: f64, depth_px: f64) -> Ellipse {
    let tilt = tilt_degrees.to_radians();
    assert!(tilt.cos() > 0.0);
    Ellipse {
        center: (80.0, 60.0 - tilt.sin() * depth_px),
        major_radius: radius_px,
        minor_radius: radius_px * tilt.cos(),
        angle: 0.0,
    }
}

fn arc(ellipse: Ellipse, begin: f64, end: f64, count: usize) -> Vec<(f64, f64)> {
    let (sin, cos) = ellipse.angle.sin_cos();
    (0..count)
        .map(|i| {
            let phase = begin + (end - begin) * (i as f64 + 0.5) / count as f64;
            let x = ellipse.major_radius * phase.cos();
            let y = ellipse.minor_radius * phase.sin();
            (
                ellipse.center.0 + cos * x - sin * y,
                ellipse.center.1 + sin * x + cos * y,
            )
        })
        .collect()
}

fn shift(mut ellipse: Ellipse, delta: [f64; 2]) -> Ellipse {
    ellipse.center.0 += delta[0];
    ellipse.center.1 += delta[1];
    ellipse
}

// One arc, one information budget; resampling the same arc is not new evidence.
fn arc_cost(points: &[(f64, f64)], ellipse: Ellipse, sigma_px: f64) -> f64 {
    points
        .iter()
        .map(|&p| ellipse_residual(p, ellipse).powi(2))
        .sum::<f64>()
        / (2.0 * sigma_px.powi(2) * points.len() as f64)
}

// Normalized profile-likelihood weights, not calibrated probabilities or a
// posterior marginalized over the depth/radius nuisance grid.
fn normalized_first_weight(costs: [f64; 2]) -> f64 {
    let base = costs[0].min(costs[1]);
    let weights = costs.map(|cost| (base - cost).exp());
    weights[0] / (weights[0] + weights[1])
}

// Profile nuisance radius/depth over a bounded grid; do not feed the ground
// truth's radius/depth to only the favored sign hypothesis.
fn pupil_sign_costs(points: &[(f64, f64)], magnitude: f64, sigma: f64) -> [f64; 2] {
    [-magnitude, magnitude].map(|signed_tilt| {
        [4.0, 6.0, 8.0]
            .into_iter()
            .flat_map(|depth| {
                [15.5, 16.0, 16.5].into_iter().map(move |radius| {
                    arc_cost(points, projected_ring(signed_tilt, radius, depth), sigma)
                })
            })
            .fold(f64::INFINITY, f64::min)
    })
}

// Soft vertical-vergence/fixation factor, supplied as an explicit assumption.
// Missing synchronization or settling means no simultaneous binocular factor.
fn coupled_first_weight(
    a: [f64; 2],
    a_angles: [f64; 2],
    b: [f64; 2],
    b_angles: [f64; 2],
    vertical_sigma_degrees: f64,
    synchronized_and_settled: bool,
) -> f64 {
    let mut joint = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let coupling = if synchronized_and_settled {
                0.5 * ((a_angles[i] - b_angles[j]) / vertical_sigma_degrees).powi(2)
            } else {
                0.0
            };
            joint[i][j] = a[i] + b[j] + coupling;
        }
    }
    let base = joint
        .iter()
        .flatten()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let marginal = joint.map(|row| row.into_iter().map(|cost| (base - cost).exp()).sum::<f64>());
    marginal[0] / marginal.iter().sum::<f64>()
}

#[test]
fn an_outer_ellipse_and_coplanar_inner_arcs_cannot_sign_the_surface() {
    for radius in [40.0, 36.0, 16.0] {
        let up = projected_ring(-25.0, radius, 0.0);
        let down = projected_ring(25.0, radius, 0.0);
        assert_eq!(up, down);
        let points = arc(up, 0.4, 2.5, 11);
        let costs = [arc_cost(&points, up, 0.2), arc_cost(&points, down, 0.2)];
        assert!((normalized_first_weight(costs) - 0.5).abs() < 1e-12);
    }
    // Distinct movable pivots explain the same projected limbus. A fixed
    // pivot would spuriously remove this ambiguity before seeing new evidence.
    let pivot_y = [-25.0_f64, 25.0].map(|angle| 60.0 - 60.0 * angle.to_radians().sin());
    assert!((pivot_y[0] - pivot_y[1]).abs() > 40.0);
}

#[test]
fn a_partial_pupil_arc_with_an_explicit_depth_prior_disambiguates_both_mirrors() {
    for true_tilt in [-25.0, 25.0] {
        let pupil = projected_ring(true_tilt, 16.0, 6.0);
        let points = arc(pupil, 0.5, 2.6, 9);
        let costs = pupil_sign_costs(&points, 25.0, 0.35);
        let p_up = normalized_first_weight(costs);
        eprintln!("pupil_arc tilt={true_tilt} conditional_up_weight={p_up:.6} costs={costs:?}");
        if true_tilt < 0.0 {
            assert!(p_up > 0.999);
        } else {
            assert!(p_up < 0.001);
        }
        // If an unrestricted pupil decentration is allowed, the opposite
        // hypothesis can reproduce the same image. Never hide this nuisance.
        let opposite = projected_ring(-true_tilt, 16.0, 6.0);
        let free_decentration = [0.0, pupil.center.1 - opposite.center.1];
        assert!(arc_cost(&points, shift(opposite, free_decentration), 0.35) < 1e-20);
    }
}

#[test]
fn a_strong_signed_second_eye_can_resolve_weak_opposing_pupil_evidence() {
    // First eye's outer AND inner limbuses are sign-ambiguous. Its weak pupil
    // arc slightly favors the wrong (down) branch. Second eye clearly sees up.
    for radius in [40.0, 36.0] {
        assert_eq!(
            projected_ring(-12.0, radius, 0.0),
            projected_ring(12.0, radius, 0.0)
        );
    }
    let weak_points = arc(projected_ring(12.0, 16.0, 6.0), 0.5, 2.6, 7);
    let strong_points = arc(projected_ring(-12.0, 16.0, 6.0), 0.5, 2.6, 9);
    let weak = pupil_sign_costs(&weak_points, 12.0, 6.0);
    let strong = pupil_sign_costs(&strong_points, 12.0, 0.2);
    let alone = normalized_first_weight(weak);
    let joint = coupled_first_weight(weak, [-12.0, 12.0], strong, [-12.0, 12.0], 6.0, true);
    let asynchronous = coupled_first_weight(weak, [-12.0, 12.0], strong, [-12.0, 12.0], 6.0, false);
    eprintln!("binocular sign: weak_alone={alone:.6} synchronized={joint:.6} unavailable_clock={asynchronous:.6}");
    assert!(alone < 0.5 && alone > 0.45);
    assert!(joint > 0.995);
    assert!((asynchronous - alone).abs() < 1e-12);
}

#[test]
fn high_quality_frontal_support_does_not_invent_strong_sign_information() {
    let weak_points = arc(projected_ring(-8.0, 16.0, 6.0), 0.5, 2.6, 9);
    let weak = pupil_sign_costs(&weak_points, 8.0, 2.0);
    let alone = normalized_first_weight(weak);
    for frontal_magnitude in [0.0, 0.05] {
        let points = arc(projected_ring(-frontal_magnitude, 16.0, 6.0), 0.0, TAU, 96);
        let strong_but_ambiguous = pupil_sign_costs(&points, frontal_magnitude, 0.15);
        let joint = coupled_first_weight(
            weak,
            [-8.0, 8.0],
            strong_but_ambiguous,
            [-frontal_magnitude, frontal_magnitude],
            8.0,
            true,
        );
        let alone_mean = -8.0 * alone + 8.0 * (1.0 - alone);
        let joint_mean = -8.0 * joint + 8.0 * (1.0 - joint);
        eprintln!("frontal control tilt={frontal_magnitude}: p={alone:.6}->{joint:.6} mean_change_deg={:.6}", joint_mean-alone_mean);
        assert!((joint - alone).abs() < 0.002);
        assert!((joint_mean - alone_mean).abs() < 0.032);
    }
}

#[derive(Clone)]
struct OwnedArc {
    group: u32,
    kind: BoundaryKind,
    reference: Ellipse,
    points: Vec<(f64, f64)>,
}

#[derive(Clone)]
struct EyeFixture {
    source: ExposureKey,
    detail: Option<f64>,
    arcs: Vec<OwnedArc>,
}

impl EyeFixture {
    fn rings(eye: u32, rings: usize, delta: [f64; 2], detail: Option<f64>) -> Self {
        let kinds = [
            BoundaryKind::OuterLimbus,
            BoundaryKind::InnerLimbus,
            BoundaryKind::PupillaryBoundary,
        ];
        let mut arcs = Vec::new();
        for ring in 0..rings {
            let reference = projected_ring(
                20.0,
                [40.0, 36.0, 16.0][ring],
                if ring == 2 { 6.0 } else { 0.0 },
            );
            for sector in 0..8 {
                let phase = sector as f64 * TAU / 8.0;
                arcs.push(OwnedArc {
                    group: (ring * 8 + sector) as u32,
                    kind: kinds[ring],
                    reference,
                    points: arc(shift(reference, delta), phase - 0.15, phase + 0.15, 5),
                });
            }
        }
        Self {
            source: exposure(eye),
            detail,
            arcs,
        }
    }

    fn fidelity(&self, heuristic: LocalizationHeuristic) -> Option<ConditionalTranslationFidelity> {
        let indices: Vec<[usize; 1]> = (0..self.arcs.len()).map(|i| [i]).collect();
        let arcs: Vec<_> = self
            .arcs
            .iter()
            .map(|a| BoundaryArcObservation {
                evidence_group: a.group,
                kind: a.kind,
                points_roi_px: &a.points,
                normal_band_half_width_px: None,
                detector_score: None,
            })
            .collect();
        let conics: Vec<_> = self
            .arcs
            .iter()
            .zip(&indices)
            .map(|(a, i)| ConicObservation {
                kind: a.kind,
                ellipse_roi_px: a.reference,
                supporting_arc_indices: i,
                residual_px: None,
            })
            .collect();
        estimate_conditional_translation_fidelity(
            RoiConicEvidence {
                exposure: self.source,
                sensor_origin_px: [0, 0],
                dimensions_px: [160, 120],
                arcs: &arcs,
                conics: &conics,
                detail_reliability: self.detail,
            },
            heuristic,
        )
    }

    // Bounded conditional common-translation fit. Shape/depth/association are
    // fixed; do not report this as a free 3D fit or anatomical center estimate.
    fn fit_translation(&self) -> [f64; 2] {
        let mut translation = [0.0; 2];
        for _ in 0..8 {
            let mut h = [[0.0; 2]; 2];
            let mut rhs = [0.0; 2];
            for support in &self.arcs {
                let predicted = shift(support.reference, translation);
                for &point in &support.points {
                    let n = conic_normal(point, predicted).unwrap();
                    let (sin, cos) = predicted.angle.sin_cos();
                    let dx = point.0 - predicted.center.0;
                    let dy = point.1 - predicted.center.1;
                    let x = cos * dx + sin * dy;
                    let y = -sin * dx + cos * dy;
                    let a2 = predicted.major_radius.powi(2);
                    let b2 = predicted.minor_radius.powi(2);
                    let residual = (x * x / a2 + y * y / b2 - 1.0) / (2.0 * (x / a2).hypot(y / b2));
                    let weight = 1.0 / support.points.len() as f64;
                    for i in 0..2 {
                        rhs[i] += weight * n[i] * residual;
                        for j in 0..2 {
                            h[i][j] += weight * n[i] * n[j];
                        }
                    }
                }
            }
            let inv = inverse_symmetric_2x2(h).unwrap();
            for i in 0..2 {
                translation[i] += inv[i][0] * rhs[0] + inv[i][1] * rhs[1];
            }
        }
        translation
    }
}

fn heuristic() -> LocalizationHeuristic {
    LocalizationHeuristic {
        sharp_edge_sigma_px: 0.25,
        additional_defocus_sigma_px: 2.0,
        unknown_detail_reliability: 0.25,
        model_floor_px: 0.05,
    }
}

#[derive(Clone, Copy, Debug)]
struct TranslationEstimate {
    mean: [f64; 2],
    covariance: [[f64; 2]; 2],
}

fn estimate(eye: &EyeFixture, noise: LocalizationHeuristic) -> TranslationEstimate {
    TranslationEstimate {
        mean: eye.fit_translation(),
        covariance: eye.fidelity(noise).unwrap().covariance_px2,
    }
}

fn fuse(estimates: &[TranslationEstimate]) -> TranslationEstimate {
    let mut information = [[0.0; 2]; 2];
    let mut rhs = [0.0; 2];
    for estimate in estimates {
        let inv = inverse_symmetric_2x2(estimate.covariance).unwrap();
        for i in 0..2 {
            rhs[i] += inv[i][0] * estimate.mean[0] + inv[i][1] * estimate.mean[1];
            for j in 0..2 {
                information[i][j] += inv[i][j];
            }
        }
    }
    let covariance = inverse_symmetric_2x2(information).unwrap();
    TranslationEstimate {
        mean: [
            covariance[0][0] * rhs[0] + covariance[0][1] * rhs[1],
            covariance[1][0] * rhs[0] + covariance[1][1] * rhs[1],
        ],
        covariance,
    }
}

fn error(mean: [f64; 2], truth: [f64; 2]) -> f64 {
    (mean[0] - truth[0]).hypot(mean[1] - truth[1])
}

fn shared_read_pair() -> [Arc<RoiBufferTiming>; 2] {
    let read = SensorReadTiming::new(clock(), 8, 1_000_000_000, None).unwrap();
    [0, 1].map(|eye| RoiBufferTiming::new(Arc::clone(&read), RoiId(eye), Some((0, 0))).unwrap())
}

fn fuse_exactly_synchronous_pair(
    estimates: [TranslationEstimate; 2],
    timing: &[Arc<RoiBufferTiming>; 2],
) -> Option<TranslationEstimate> {
    let dt = timing[1].relative_exposure_to(&timing[0])?;
    // This fixture has no velocity state; asynchronous evidence requires the
    // explicit motion/uncertainty transport tested separately below.
    (dt.minimum == 0 && dt.maximum == 0).then(|| fuse(&estimates))
}

#[test]
fn two_strong_same_clock_outer_limbuses_override_a_weak_single_image_position() {
    let timing = shared_read_pair();
    assert_eq!(
        timing[1]
            .relative_exposure_to(&timing[0])
            .unwrap()
            .seconds(),
        (0.0, 0.0)
    );
    let truth = [1.0, -1.5];
    // Independent eyes, already mapped into a COMMON local translation frame.
    // This is not raw image-center averaging or a calibrated stereo depth fit.
    let left = EyeFixture::rings(0, 1, [0.95, -1.45], Some(0.85));
    let right = EyeFixture::rings(1, 1, [1.05, -1.55], Some(0.85));
    let weak = TranslationEstimate {
        mean: [3.0, 2.0],
        covariance: [[9.0, 0.0], [0.0, 9.0]],
    };
    let a = estimate(&left, heuristic());
    let b = estimate(&right, heuristic());
    let pair = fuse_exactly_synchronous_pair([a, b], &timing).unwrap();
    let combined = fuse(&[weak, pair]);
    eprintln!(
        "same-clock pair: weak={:?} joint={:?} joint_error_px={:.6}",
        weak.mean,
        combined.mean,
        error(combined.mean, truth)
    );
    assert!(error(combined.mean, truth) < 0.02);
    assert!(error(combined.mean, truth) < error(weak.mean, truth) * 0.01);
    assert!(combined.covariance[0][0] < weak.covariance[0][0] * 0.01);
}

#[test]
fn quantified_multiconic_information_can_outweigh_two_good_limbuses() {
    let excellent = EyeFixture::rings(0, 3, [0.0, 0.0], Some(1.0));
    let mut stereo_a = EyeFixture::rings(0, 1, [0.35, -0.3], Some(0.70));
    let stereo_b = EyeFixture::rings(1, 1, [0.35, -0.3], Some(0.70));
    // The extra same-eye outer arcs occupy different angular patches. Their
    // group ids are disjoint from the excellent system's outer/inner/pupil
    // support; otherwise the comparison would count the same pixels twice.
    for (sector, support) in stereo_a.arcs.iter_mut().enumerate() {
        support.group += 100;
        let phase = sector as f64 * TAU / 8.0 + PI / 8.0;
        support.points = arc(
            shift(support.reference, [0.35, -0.3]),
            phase - 0.1,
            phase + 0.1,
            5,
        );
    }
    assert!(stereo_a
        .arcs
        .iter()
        .all(|a| excellent.arcs.iter().all(|b| a.group != b.group)));
    let excellent_fidelity = excellent.fidelity(heuristic()).unwrap();
    let limbus_fidelity = stereo_a.fidelity(heuristic()).unwrap();
    let mono = estimate(&excellent, heuristic());
    let stereo = fuse_exactly_synchronous_pair(
        [
            estimate(&stereo_a, heuristic()),
            estimate(&stereo_b, heuristic()),
        ],
        &shared_read_pair(),
    )
    .unwrap();
    // Independent patch noise and the specified per-packet model floors are
    // explicit assumptions, not evidence that real optical errors are independent.
    let combined = fuse(&[mono, stereo]);
    let mono_precision_fraction = (1.0 / mono.covariance[0][0])
        / (1.0 / mono.covariance[0][0] + 1.0 / stereo.covariance[0][0]);
    eprintln!("fidelity: multiconic_sigma_px={:.6} limbus_sigma_px={:.6} mono_x_information_fraction={:.6} joint={:?}",
        excellent_fidelity.worst_axis_sigma_px(),limbus_fidelity.worst_axis_sigma_px(),mono_precision_fraction,combined.mean);
    assert_eq!(excellent_fidelity.independent_groups, 24);
    assert_eq!(limbus_fidelity.independent_groups, 8);
    assert!(excellent_fidelity.worst_axis_sigma_px() < limbus_fidelity.worst_axis_sigma_px() / 3.0);
    assert!(mono_precision_fraction > 0.85);
    assert!(error(combined.mean, mono.mean) < error(combined.mean, stereo.mean) / 5.0);
}

#[test]
fn focus_heuristically_reduces_limbus_fidelity_and_influence_without_dropping_it() {
    let sharp = EyeFixture::rings(0, 1, [0.0, 0.0], Some(1.0));
    let mut blurred = sharp.clone();
    blurred.detail = Some(0.1);
    let mut unknown = sharp.clone();
    unknown.detail = None;
    let sharp_f = sharp.fidelity(heuristic()).unwrap();
    let blurred_f = blurred.fidelity(heuristic()).unwrap();
    let unknown_f = unknown.fidelity(heuristic()).unwrap();
    let competitor = TranslationEstimate {
        mean: [1.0, -1.0],
        covariance: [[0.25, 0.0], [0.0, 0.25]],
    };
    let with_sharp = fuse(&[competitor, estimate(&sharp, heuristic())]);
    let with_blur = fuse(&[competitor, estimate(&blurred, heuristic())]);
    eprintln!(
        "focus: sharp_sigma_px={:.6} blurred={:.6} unknown={:.6} sharp_joint={:?} blur_joint={:?}",
        sharp_f.worst_axis_sigma_px(),
        blurred_f.worst_axis_sigma_px(),
        unknown_f.worst_axis_sigma_px(),
        with_sharp.mean,
        with_blur.mean
    );
    assert!(blurred_f.worst_axis_sigma_px() > 5.0 * sharp_f.worst_axis_sigma_px());
    assert!(unknown_f.worst_axis_sigma_px() > sharp_f.worst_axis_sigma_px());
    assert!(error(with_blur.mean, competitor.mean) < error(with_sharp.mean, competitor.mean));
    assert!(error(with_blur.mean, competitor.mean) > 0.05); // useful, not ignored
    for detail in [f64::NAN, -0.01, 1.01] {
        assert!(heuristic().sigma_px(Some(detail)).is_none());
    }
}

#[test]
fn repeated_conics_or_resampled_pixels_do_not_increase_information() {
    let baseline = EyeFixture::rings(0, 3, [0.0, 0.0], Some(1.0));
    let original = baseline.fidelity(heuristic()).unwrap();
    let mut duplicate = baseline.clone();
    duplicate.arcs.extend(baseline.arcs.iter().cloned());
    assert_eq!(duplicate.fidelity(heuristic()).unwrap(), original);
    let mut dense = baseline.clone();
    for support in &mut dense.arcs {
        let points = support.points.clone();
        for _ in 0..15 {
            support.points.extend_from_slice(&points);
        }
    }
    let resampled = dense.fidelity(heuristic()).unwrap();
    assert_eq!(resampled.independent_groups, original.independent_groups);
    assert!((resampled.worst_axis_sigma_px() - original.worst_axis_sigma_px()).abs() < 1e-12);
    let mut conflicting = baseline.clone();
    let mut different_conic_same_pixels = baseline.arcs[0].clone();
    different_conic_same_pixels.reference.center.0 += 1.0;
    conflicting.arcs.push(different_conic_same_pixels);
    assert!(conflicting.fidelity(heuristic()).is_none());
    let mut unresolved = baseline.clone();
    let mut different_arc_same_group = baseline.arcs[0].clone();
    different_arc_same_group.points[0].0 += 1.0;
    unresolved.arcs.push(different_arc_same_group);
    assert!(unresolved.fidelity(heuristic()).is_none());
}

#[test]
fn fidelity_refuses_invalid_geometry_noise_and_overflow_instead_of_claiming_certainty() {
    let valid = EyeFixture::rings(0, 1, [0.0, 0.0], Some(1.0));
    let mut invalid = valid.clone();
    invalid.arcs[0].reference.major_radius = 0.0;
    assert!(invalid.fidelity(heuristic()).is_none());
    invalid = valid.clone();
    invalid.arcs[0].points[0].0 = f64::NAN;
    assert!(invalid.fidelity(heuristic()).is_none());
    for sharp_edge_sigma_px in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(valid
            .fidelity(LocalizationHeuristic {
                sharp_edge_sigma_px,
                ..heuristic()
            })
            .is_none());
    }
    assert!(valid
        .fidelity(LocalizationHeuristic {
            model_floor_px: 1e308,
            ..heuristic()
        })
        .is_none());
    assert!(valid
        .fidelity(LocalizationHeuristic {
            model_floor_px: -0.1,
            ..heuristic()
        })
        .is_none());
}

#[test]
fn a_single_tangent_direction_does_not_claim_a_precise_center() {
    let mut eye = EyeFixture::rings(0, 1, [0.0, 0.0], Some(1.0));
    eye.arcs.truncate(1);
    eye.arcs[0].points = vec![(120.0, 60.0); 20];
    assert!(eye.fidelity(heuristic()).is_none());
    eye.arcs[0].points = arc(eye.arcs[0].reference, -0.1, 0.1, 7);
    let short = eye.fidelity(heuristic()).unwrap();
    let complete = EyeFixture::rings(0, 1, [0.0, 0.0], Some(1.0))
        .fidelity(heuristic())
        .unwrap();
    assert!(short.worst_axis_sigma_px() > 20.0 * complete.worst_axis_sigma_px());
}

// Exact extrema of a constant-acceleration trajectory over a bounded source
// interval, including an internal turning point. Variance here models a uniform
// time interval only in tests that explicitly assume a uniform distribution.
fn position_band(position: f64, velocity: f64, acceleration: f64, dt: (f64, f64)) -> (f64, f64) {
    let at = |t: f64| position + velocity * t + 0.5 * acceleration * t * t;
    let mut low = at(dt.0).min(at(dt.1));
    let mut high = at(dt.0).max(at(dt.1));
    if acceleration.abs() > 1e-12 {
        let turning = -velocity / acceleration;
        if (dt.0..=dt.1).contains(&turning) {
            low = low.min(at(turning));
            high = high.max(at(turning));
        }
    }
    (low, high)
}

fn distance_to_band(value: f64, band: (f64, f64)) -> f64 {
    (band.0 - value).max(value - band.1).max(0.0)
}

#[test]
fn source_time_bounds_distinguish_motion_hypotheses_and_widen_uncertainty() {
    let previous = SensorReadTiming::new(
        clock(),
        8,
        1_000_000_000,
        TimestampBandNs::new(999_000_000, 1_001_000_000),
    )
    .unwrap();
    let current = SensorReadTiming::new(
        clock(),
        9,
        1_020_000_000,
        TimestampBandNs::new(1_018_000_000, 1_022_000_000),
    )
    .unwrap();
    let precise = current.relative_to(&previous).unwrap().seconds();
    let correct = position_band(0.0, 100.0, 0.0, precise);
    let wrong_direction = position_band(0.0, -100.0, 0.0, precise);
    let measured = 2.0;
    assert!((correct.0 - 1.7).abs() < 1e-12 && (correct.1 - 2.3).abs() < 1e-12);
    assert_eq!(distance_to_band(measured, correct), 0.0);
    assert!(distance_to_band(measured, wrong_direction) > 3.6);
    // Host/SAM completion lag is intentionally NOT a source exposure delta.
    let inference_completion_dt = (0.17, 0.17);
    assert!(
        distance_to_band(
            measured,
            position_band(0.0, 100.0, 0.0, inference_completion_dt)
        ) > 14.0
    );
    let broad = SensorReadTiming::new(
        clock(),
        9,
        1_020_000_000,
        TimestampBandNs::new(1_000_000_000, 1_040_000_000),
    )
    .unwrap();
    let broad_dt = broad.relative_to(&previous).unwrap().seconds();
    let fast_precise = position_band(0.0, 200.0, 0.0, precise);
    let fast_uncertain = position_band(0.0, 200.0, 0.0, broad_dt);
    assert!(distance_to_band(measured, fast_precise) > 1.3);
    assert_eq!(distance_to_band(measured, fast_uncertain), 0.0);
    eprintln!("source timing: correct_band_px={correct:?} wrong_direction_px={wrong_direction:?} broad_fast_band_px={fast_uncertain:?}");
    // A broad band supports less discrimination; it is not a fresh precise vote.
    let turning = position_band(0.0, 2.0, -4.0, (0.0, 1.0));
    assert_eq!(turning, (0.0, 0.5));
    assert_eq!(
        position_band(0.0, 100.0, 0.0, (-0.023, -0.017)),
        (-2.3, -1.7000000000000002)
    );
}

#[test]
fn wrong_epochs_and_unmeasured_row_phase_never_become_exact_stereo() {
    let read = SensorReadTiming::new(clock(), 1, 100, None).unwrap();
    let left = RoiBufferTiming::new(Arc::clone(&read), RoiId(0), None).unwrap();
    let right = RoiBufferTiming::new(read, RoiId(1), None).unwrap();
    assert!(right.relative_exposure_to(&left).is_none());
    let eye = EyeFixture::rings(0, 1, [0.0, 0.0], Some(1.0));
    let estimate = estimate(&eye, heuristic());
    assert!(fuse_exactly_synchronous_pair([estimate; 2], &[Arc::clone(&left), right]).is_none());
    let restarted = SensorReadTiming::new(
        SourceClock {
            epoch: 8,
            ..clock()
        },
        1,
        100,
        TimestampBandNs::new(100, 100),
    )
    .unwrap();
    assert!(restarted.relative_to(left.read()).is_none());
    // Real wire packets currently supply timestamps, not these uncertainty
    // attestations. These contracts cannot manufacture hardware timing bounds.
}
