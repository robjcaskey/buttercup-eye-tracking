//! Sparse conic fitting and geometric support/culling.
//!
//! This is the existing deterministic single-ellipse fitter, extracted intact.
//! Mixed-boundary/multi-ROI posterior solving is not implemented here yet.
//! Conditioning scores and scale intervals are heuristics, not calibrated probabilities.

use crate::geometry::{ellipse_axis_point, ellipse_coordinate, Ellipse};
use crate::roi_evidence::{BoundaryKind, ExposureKey, RoiConicEvidence};
use std::cmp::Ordering as CmpOrdering;
use std::f64::consts::PI;

pub(crate) mod fidelity;
#[cfg(test)]
mod constraint_tests;

/// Future joint constraint solve. Budgets must bound hypothesis expansion
/// and refinement, not just how many candidates happen to be returned.
/// The completed contract must condition nested limbus/pupil arcs on an
/// uncertain eye-scene state AND return source-keyed residual factors that
/// refine that state (including its effective pivot). This is a bounded joint
/// optimization, not a one-way fixed-pivot gate or repeated independent votes
/// for the same pixels. Pupil center/axis and rotation pivot are distinct.
/// The legacy single-ellipse path below is unchanged and does not consume
/// these settings.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointConicRequest<'a> {
    pub(crate) eyes: [Option<RoiConicEvidence<'a>>; 2],
    pub(crate) maximum_hypotheses: usize,
    pub(crate) maximum_refinements: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct SupportedConicSolution {
    pub(crate) exposure: ExposureKey,
    pub(crate) kind: BoundaryKind,
    pub(crate) ellipse_roi_px: Ellipse,
    pub(crate) supporting_arc_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JointConicUnavailable {
    NotImplemented,
}

pub(crate) fn solve_joint_conics(
    _request: JointConicRequest<'_>,
) -> Result<Vec<SupportedConicSolution>, JointConicUnavailable> {
    // Do not disguise independent ellipse fits as a joint posterior.
    Err(JointConicUnavailable::NotImplemented)
}

/// Conservative camera-and-anatomy envelope for treating an image conic as
/// the projection of the physical limbus.
///
/// This is intentionally not called a calibration.  The lower focal bound
/// sits below both the current relative solve (~3250 px) and the checkerboard
/// preview value (4737.6 px), while the upper bound sits above both.  Focal
/// uncertainty changes angular scale; it cannot by itself explain arbitrary
/// anisotropy on square sensor pixels.  `maximum_local_metric_anisotropy`
/// reserves a further six percent for residual central lens distortion,
/// pixel-aspect uncertainty, and use before a full-aperture calibration has
/// been accepted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentralCameraLimbusProjectionEnvelope {
    pub minimum_focal_length_px: f64,
    pub maximum_focal_length_px: f64,
    pub maximum_pixel_aspect_error: f64,
    pub maximum_local_metric_anisotropy: f64,
    pub maximum_anatomical_surface_tilt_radians: f64,
    pub uncalibrated_central_ray_slack_radians: f64,
    pub maximum_limbus_half_angle_radians: f64,
    pub absolute_minimum_minor_to_major: f64,
}

pub const PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE: CentralCameraLimbusProjectionEnvelope =
    CentralCameraLimbusProjectionEnvelope {
        minimum_focal_length_px: 3_000.0,
        maximum_focal_length_px: 5_200.0,
        maximum_pixel_aspect_error: 0.01,
        maximum_local_metric_anisotropy: 1.06,
        maximum_anatomical_surface_tilt_radians: 55.0 * PI / 180.0,
        uncalibrated_central_ray_slack_radians: 1.5 * PI / 180.0,
        maximum_limbus_half_angle_radians: 4.0 * PI / 180.0,
        absolute_minimum_minor_to_major: 0.47,
    };

/// Typed interpretation of a canonical projected-limbus ellipse.  The
/// implied tilt is the weak-perspective circle-plane tilt, `acos(b/a)`.
/// `minimum_minor_to_major` is more permissive than the nominal anatomical
/// bound because it already includes the worst focal, aperture, and local
/// camera-metric allowances above.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimbusProjectionAssessment {
    pub major_radius_px: f64,
    pub minor_radius_px: f64,
    pub minor_to_major: f64,
    /// `acos(b/a)` before undoing the bounded local camera-metric anisotropy.
    /// This is an image-derived tilt proxy, not a calibrated gaze angle.
    pub uncorrected_image_implied_tilt_radians: f64,
    /// Maximum physical plane tilt before the camera-metric allowance.
    pub maximum_supported_surface_tilt_radians: f64,
    /// Equivalent image-conic tilt after applying every conservative camera
    /// allowance. This is the value directly comparable to the uncorrected
    /// image proxy above.
    pub maximum_supported_image_tilt_radians: f64,
    pub minimum_minor_to_major: f64,
}

impl CentralCameraLimbusProjectionEnvelope {
    pub fn assess_axes(
        self,
        axis_a_radius_px: f64,
        axis_b_radius_px: f64,
    ) -> Option<LimbusProjectionAssessment> {
        if !axis_a_radius_px.is_finite()
            || !axis_b_radius_px.is_finite()
            || axis_a_radius_px <= 0.0
            || axis_b_radius_px <= 0.0
            || !self.minimum_focal_length_px.is_finite()
            || !self.maximum_focal_length_px.is_finite()
            || self.minimum_focal_length_px <= 0.0
            || self.maximum_focal_length_px < self.minimum_focal_length_px
            || !self.maximum_local_metric_anisotropy.is_finite()
            || self.maximum_local_metric_anisotropy < 1.0
        {
            return None;
        }
        let major_radius_px = axis_a_radius_px.max(axis_b_radius_px);
        let minor_radius_px = axis_a_radius_px.min(axis_b_radius_px);
        let minor_to_major = minor_radius_px / major_radius_px;
        let limbus_half_angle = (major_radius_px / self.minimum_focal_length_px)
            .atan()
            .min(self.maximum_limbus_half_angle_radians.max(0.0));
        let maximum_supported_tilt_radians = (self.maximum_anatomical_surface_tilt_radians
            + self.uncalibrated_central_ray_slack_radians
            + limbus_half_angle)
            .clamp(0.0, PI * 0.5 - 1.0e-6);
        let minimum_minor_to_major = (maximum_supported_tilt_radians.cos()
            / self.maximum_local_metric_anisotropy)
            .max(self.absolute_minimum_minor_to_major)
            .clamp(0.0, 1.0);
        Some(LimbusProjectionAssessment {
            major_radius_px,
            minor_radius_px,
            minor_to_major,
            uncorrected_image_implied_tilt_radians: minor_to_major.clamp(0.0, 1.0).acos(),
            maximum_supported_surface_tilt_radians: maximum_supported_tilt_radians,
            maximum_supported_image_tilt_radians: minimum_minor_to_major.acos(),
            minimum_minor_to_major,
        })
    }

    pub fn admits_axes(self, axis_a_radius_px: f64, axis_b_radius_px: f64) -> bool {
        self.assess_axes(axis_a_radius_px, axis_b_radius_px)
            .is_some_and(|assessment| {
                assessment.minor_to_major + 1.0e-12 >= assessment.minimum_minor_to_major
            })
    }

    pub fn maximum_major_to_minor(self, major_radius_px: f64) -> Option<f64> {
        let assessment = self.assess_axes(major_radius_px, major_radius_px)?;
        Some(1.0 / assessment.minimum_minor_to_major.max(1.0e-9))
    }
}

pub fn assess_projected_circular_limbus_axes(
    axis_a_radius_px: f64,
    axis_b_radius_px: f64,
) -> Option<LimbusProjectionAssessment> {
    PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.assess_axes(axis_a_radius_px, axis_b_radius_px)
}

pub fn projected_circular_limbus_axes_plausible(
    axis_a_radius_px: f64,
    axis_b_radius_px: f64,
) -> bool {
    PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE.admits_axes(axis_a_radius_px, axis_b_radius_px)
}

// Preserve the accepted model-space fit domain during extraction. These are
// not sensor dimensions or universal biological constraints.
pub(crate) const LEGACY_FIT_WIDTH: usize = 384;
pub(crate) const LEGACY_FIT_HEIGHT: usize = 256;
use LEGACY_FIT_HEIGHT as FRAME_HEIGHT;
use LEGACY_FIT_WIDTH as FRAME_WIDTH;

/// Frozen current-frame support for recovering a physical limbus from an
/// occluded semantic-mask contour.  Radii are the fronto-parallel radius of
/// the circular limbus (the larger projected semi-axis under weak
/// perspective), after transporting the previous frame by independent 2D
/// affine image scale.  The contour itself never widens this interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OuterContourScaleContext {
    pub estimated_fronto_parallel_radius_px: f64,
    pub minimum_fronto_parallel_radius_px: f64,
    pub maximum_fronto_parallel_radius_px: f64,
}

impl OuterContourScaleContext {
    pub fn new(estimate: f64, minimum: f64, maximum: f64) -> Option<Self> {
        (estimate.is_finite()
            && minimum.is_finite()
            && maximum.is_finite()
            && minimum >= 4.0
            && maximum > minimum
            && (minimum..=maximum).contains(&estimate))
        .then_some(Self {
            estimated_fronto_parallel_radius_px: estimate,
            minimum_fronto_parallel_radius_px: minimum,
            maximum_fronto_parallel_radius_px: maximum,
        })
    }

    pub(crate) fn admits(self, ellipse: Ellipse) -> bool {
        let radius = ellipse.major_radius.max(ellipse.minor_radius);
        radius.is_finite()
            && (self.minimum_fronto_parallel_radius_px..=self.maximum_fronto_parallel_radius_px)
                .contains(&radius)
    }
}

/// Deterministically refit an ellipse to contour samples admitted by an
/// offline semantic evidence gate.  The caller remains responsible for
/// enforcing its scale/pose prior before publishing the result.
pub fn fit_trusted_arc_points(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 10 {
        return None;
    }
    let mut random = NumpyPcg64::baseline_fit_stream();
    let robust = robust_ransac_ellipse(points, &mut random)?;
    let inliers = points
        .iter()
        .zip(robust.inliers.iter())
        .filter_map(|(&point, &inlier)| inlier.then_some(point))
        .collect::<Vec<_>>();
    let ellipse = robust_contour_fit(&inliers, robust.ellipse).unwrap_or(robust.ellipse);
    plausible_ellipse(ellipse).then_some(ellipse)
}

pub(crate) fn moments_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 20 {
        return None;
    }
    let count = points.len() as f64;
    let center = points
        .iter()
        .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
    let center = (center.0 / count, center.1 / count);
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    for &(x, y) in points {
        let dx = x - center.0;
        let dy = y - center.1;
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    xx /= count;
    xy /= count;
    yy /= count;
    let root = ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt();
    let lambda_major = ((xx + yy + root) * 0.5).max(1.0);
    let lambda_minor = ((xx + yy - root) * 0.5).max(1.0);
    Some(Ellipse {
        center,
        major_radius: 2.0 * lambda_major.sqrt(),
        minor_radius: 2.0 * lambda_minor.sqrt(),
        angle: 0.5 * (2.0 * xy).atan2(xx - yy),
    })
}

pub(crate) fn solve_five(mut matrix: [[f64; 5]; 5], mut rhs: [f64; 5]) -> Option<[f64; 5]> {
    for pivot in 0..5 {
        let row = (pivot..5).max_by(|&first, &second| {
            matrix[first][pivot]
                .abs()
                .partial_cmp(&matrix[second][pivot].abs())
                .unwrap_or(CmpOrdering::Equal)
        })?;
        if matrix[row][pivot].abs() < 1e-12 {
            return None;
        }
        if row != pivot {
            matrix.swap(row, pivot);
            rhs.swap(row, pivot);
        }
        let divisor = matrix[pivot][pivot];
        for column in pivot..5 {
            matrix[pivot][column] /= divisor;
        }
        rhs[pivot] /= divisor;
        for row in 0..5 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..5 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

pub(crate) fn robust_contour_fit(points: &[(f64, f64)], initial: Ellipse) -> Option<Ellipse> {
    if points.len() < 24 {
        return None;
    }
    let mut parameters = [
        initial.center.0,
        initial.center.1,
        initial.major_radius.max(8.0),
        initial.minor_radius.max(8.0),
        initial.angle,
    ];
    for _ in 0..14 {
        let (sin_angle, cos_angle) = parameters[4].sin_cos();
        let major = parameters[2].max(8.0);
        let minor = parameters[3].max(8.0);
        let major2 = major * major;
        let minor2 = minor * minor;
        let mut normal = [[0.0f64; 5]; 5];
        let mut gradient = [0.0f64; 5];
        for &(x, y) in points {
            let dx = x - parameters[0];
            let dy = y - parameters[1];
            let xp = cos_angle * dx + sin_angle * dy;
            let yp = -sin_angle * dx + cos_angle * dy;
            let residual = xp * xp / major2 + yp * yp / minor2 - 1.0;
            let weight = if residual.abs() <= 0.18 {
                1.0
            } else {
                0.18 / residual.abs()
            };
            let jacobian = [
                -2.0 * xp * cos_angle / major2 + 2.0 * yp * sin_angle / minor2,
                -2.0 * xp * sin_angle / major2 - 2.0 * yp * cos_angle / minor2,
                -2.0 * xp * xp / major.powi(3),
                -2.0 * yp * yp / minor.powi(3),
                2.0 * xp * yp * (1.0 / major2 - 1.0 / minor2),
            ];
            for row in 0..5 {
                gradient[row] += weight * jacobian[row] * residual;
                for column in 0..5 {
                    normal[row][column] += weight * jacobian[row] * jacobian[column];
                }
            }
        }
        for index in 0..5 {
            normal[index][index] += 1e-7 * normal[index][index].abs().max(1.0);
        }
        let mut step = solve_five(normal, gradient.map(|value| -value))?;
        step[0] = step[0].clamp(-5.0, 5.0);
        step[1] = step[1].clamp(-5.0, 5.0);
        step[2] = step[2].clamp(-8.0, 8.0);
        step[3] = step[3].clamp(-8.0, 8.0);
        step[4] = step[4].clamp(-0.15, 0.15);
        for index in 0..5 {
            parameters[index] += step[index];
        }
        parameters[2] = parameters[2].clamp(12.0, FRAME_WIDTH as f64 * 0.55);
        parameters[3] = parameters[3].clamp(12.0, FRAME_HEIGHT as f64 * 0.70);
        if step.iter().map(|value| value.abs()).sum::<f64>() < 1e-4 {
            break;
        }
    }
    let mut ellipse = Ellipse {
        center: (parameters[0], parameters[1]),
        major_radius: parameters[2],
        minor_radius: parameters[3],
        angle: parameters[4],
    };
    normalize_ellipse(&mut ellipse);
    plausible_ellipse(ellipse).then_some(ellipse)
}

pub(crate) fn normalize_ellipse(ellipse: &mut Ellipse) {
    if ellipse.minor_radius > ellipse.major_radius {
        std::mem::swap(&mut ellipse.major_radius, &mut ellipse.minor_radius);
        ellipse.angle += std::f64::consts::FRAC_PI_2;
    }
    ellipse.angle = (ellipse.angle + std::f64::consts::FRAC_PI_2).rem_euclid(std::f64::consts::PI)
        - std::f64::consts::FRAC_PI_2;
}

pub(crate) fn plausible_ellipse(ellipse: Ellipse) -> bool {
    ellipse.center.0.is_finite()
        && ellipse.center.1.is_finite()
        && ellipse.major_radius.is_finite()
        && ellipse.minor_radius.is_finite()
        && ellipse.center.0 > -(FRAME_WIDTH as f64) * 0.1
        && ellipse.center.0 < FRAME_WIDTH as f64 * 1.1
        && ellipse.center.1 > -(FRAME_HEIGHT as f64) * 0.1
        && ellipse.center.1 < FRAME_HEIGHT as f64 * 1.1
        && (30.0..=FRAME_WIDTH as f64 * 0.52).contains(&ellipse.major_radius)
        && (24.0..=FRAME_HEIGHT as f64 * 0.65).contains(&ellipse.minor_radius)
        && ellipse.minor_radius / ellipse.major_radius >= 0.28
}

pub(crate) fn direct_conic_fit(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 5 {
        return None;
    }
    let count = points.len() as f64;
    let center = points
        .iter()
        .fold((0.0, 0.0), |sum, point| (sum.0 + point.0, sum.1 + point.1));
    let center = (center.0 / count, center.1 / count);
    let scale = (points
        .iter()
        .map(|point| (point.0 - center.0).powi(2) + (point.1 - center.1).powi(2))
        .sum::<f64>()
        / count)
        .sqrt()
        .max(1.0);
    let mut normal = [[0.0f64; 5]; 5];
    let mut rhs = [0.0f64; 5];
    for &(point_x, point_y) in points {
        let x = (point_x - center.0) / scale;
        let y = (point_y - center.1) / scale;
        let terms = [x * x, x * y, y * y, x, y];
        for row in 0..5 {
            rhs[row] += terms[row];
            for column in 0..5 {
                normal[row][column] += terms[row] * terms[column];
            }
        }
    }
    for index in 0..5 {
        normal[index][index] += 1e-10;
    }
    let [quadratic_x, cross, quadratic_y, linear_x, linear_y] = solve_five(normal, rhs)?;
    let off_diagonal = cross * 0.5;
    let determinant = quadratic_x * quadratic_y - off_diagonal * off_diagonal;
    if determinant <= 1e-10 {
        return None;
    }
    let local_center_x = -0.5 * (quadratic_y * linear_x - off_diagonal * linear_y) / determinant;
    let local_center_y = -0.5 * (-off_diagonal * linear_x + quadratic_x * linear_y) / determinant;
    let level = 1.0
        + quadratic_x * local_center_x * local_center_x
        + 2.0 * off_diagonal * local_center_x * local_center_y
        + quadratic_y * local_center_y * local_center_y;
    let trace = quadratic_x + quadratic_y;
    let discriminant = ((quadratic_x - quadratic_y).powi(2) + 4.0 * off_diagonal.powi(2)).sqrt();
    let eigen_minimum = (trace - discriminant) * 0.5;
    let eigen_maximum = (trace + discriminant) * 0.5;
    if level <= 0.0 || eigen_minimum <= 1e-10 || eigen_maximum <= eigen_minimum {
        return None;
    }
    let major_vector = if off_diagonal.abs() > 1e-10 {
        (off_diagonal, eigen_minimum - quadratic_x)
    } else if quadratic_x <= quadratic_y {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let mut ellipse = Ellipse {
        center: (
            center.0 + local_center_x * scale,
            center.1 + local_center_y * scale,
        ),
        major_radius: scale * (level / eigen_minimum).sqrt(),
        minor_radius: scale * (level / eigen_maximum).sqrt(),
        angle: major_vector.1.atan2(major_vector.0),
    };
    normalize_ellipse(&mut ellipse);
    (ellipse.center.0.is_finite()
        && ellipse.center.1.is_finite()
        && ellipse.major_radius.is_finite()
        && ellipse.minor_radius.is_finite()
        && ellipse.major_radius < 2000.0
        && ellipse.minor_radius >= 2.0
        && ellipse.minor_radius / ellipse.major_radius >= 0.05)
        .then_some(ellipse)
}

#[cfg(feature = "sam31")]
pub(crate) fn least_squares_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    use opencv::core::{Point2f, Vector};

    if points.len() < 5 {
        return None;
    }
    let mut input = Vector::<Point2f>::with_capacity(points.len());
    for &(x, y) in points {
        input.push(Point2f::new(x as f32, y as f32));
    }
    let fitted = opencv::imgproc::fit_ellipse(&input).ok()?;
    let mut ellipse = Ellipse {
        center: (fitted.center.x as f64, fitted.center.y as f64),
        major_radius: fitted.size.width as f64 * 0.5,
        minor_radius: fitted.size.height as f64 * 0.5,
        angle: (fitted.angle as f64).to_radians(),
    };
    normalize_ellipse(&mut ellipse);
    (ellipse.center.0.is_finite()
        && ellipse.center.1.is_finite()
        && ellipse.major_radius.is_finite()
        && ellipse.minor_radius.is_finite()
        && ellipse.major_radius < 2000.0
        && ellipse.minor_radius >= 2.0
        && ellipse.minor_radius / ellipse.major_radius >= 0.05)
        .then_some(ellipse)
}

#[cfg(not(feature = "sam31"))]
pub(crate) fn least_squares_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    direct_conic_fit(points)
}

pub(crate) fn ellipse_residual(point: (f64, f64), ellipse: Ellipse) -> f64 {
    let (sin_angle, cos_angle) = ellipse.angle.sin_cos();
    let dx = point.0 - ellipse.center.0;
    let dy = point.1 - ellipse.center.1;
    let x = cos_angle * dx + sin_angle * dy;
    let y = -sin_angle * dx + cos_angle * dy;
    let radial = ((x / ellipse.major_radius.max(1e-6)).powi(2)
        + (y / ellipse.minor_radius.max(1e-6)).powi(2))
    .sqrt();
    (radial - 1.0).abs() * (ellipse.major_radius * ellipse.minor_radius).sqrt()
}

pub(crate) struct RobustFit {
    pub(crate) ellipse: Ellipse,
    pub(crate) residuals: Vec<f64>,
    pub(crate) inliers: Vec<bool>,
    pub(crate) cutoff: f64,
}

pub(crate) struct NumpyPcg64 {
    pub(crate) state: u128,
    pub(crate) increment: u128,
    pub(crate) buffered_upper: Option<u32>,
}

impl NumpyPcg64 {
    pub(crate) const MULTIPLIER: u128 =
        ((2_549_297_995_355_413_924u128) << 64) | 4_865_540_595_714_422_341u128;

    pub(crate) fn baseline_fit_stream() -> Self {
        // np.random.default_rng(20260812).bit_generator.state after NumPy's
        // SeedSequence expansion. Keeping this tiny PCG stream in Rust avoids
        // a Python dependency while preserving the accepted six-query fit.
        Self {
            state: 94_368_598_227_006_152_144_556_554_533_211_010_852u128,
            increment: 174_605_511_227_888_025_715_320_101_340_171_151_329u128,
            buffered_upper: None,
        }
    }

    pub(crate) fn vertical_cull_stream() -> Self {
        // np.random.default_rng(20260813).bit_generator.state.
        Self {
            state: 32_836_341_824_432_678_873_702_219_582_571_864_733u128,
            increment: 70_717_288_352_767_355_186_512_493_438_108_981_773u128,
            buffered_upper: None,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.increment);
        let high = (self.state >> 64) as u64;
        let low = self.state as u64;
        (high ^ low).rotate_right((self.state >> 122) as u32)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.buffered_upper.take() {
            return value;
        }
        let value = self.next_u64();
        self.buffered_upper = Some((value >> 32) as u32);
        value as u32
    }

    pub(crate) fn bounded_inclusive(&mut self, maximum: u32) -> u32 {
        if maximum == 0 {
            return 0;
        }
        let range = maximum + 1;
        let threshold = (u32::MAX - maximum) % range;
        loop {
            let product = self.next_u32() as u64 * range as u64;
            if product as u32 >= threshold {
                return (product >> 32) as u32;
            }
        }
    }

    pub(crate) fn choice_five(&mut self, population: usize) -> Option<[usize; 5]> {
        if population < 5 || population > u32::MAX as usize {
            return None;
        }
        // Generator.choice uses Floyd's algorithm here. For a five-element
        // draw its 1.2x hash table always rounds up to eight slots.
        let mut hash = [usize::MAX; 8];
        let mut choice = [0usize; 5];
        for (slot, candidate) in ((population - 5)..population).enumerate() {
            let value = self.bounded_inclusive(candidate as u32) as usize;
            let mut location = value & 7;
            while hash[location] != usize::MAX && hash[location] != value {
                location = (location + 1) & 7;
            }
            if hash[location] == usize::MAX {
                hash[location] = value;
                choice[slot] = value;
            } else {
                location = candidate & 7;
                while hash[location] != usize::MAX {
                    location = (location + 1) & 7;
                }
                hash[location] = candidate;
                choice[slot] = candidate;
            }
        }
        for index in (1..5).rev() {
            let other = self.bounded_inclusive(index as u32) as usize;
            choice.swap(index, other);
        }
        Some(choice)
    }
}

pub(crate) fn robust_ransac_ellipse_with_context(
    points: &[(f64, f64)],
    random: &mut NumpyPcg64,
    scale_context: Option<OuterContourScaleContext>,
) -> Option<RobustFit> {
    robust_ransac_ellipse_with_noise(points, random, scale_context, 1.0)
}

pub(crate) fn robust_ransac_ellipse_with_noise(
    points: &[(f64, f64)],
    random: &mut NumpyPcg64,
    scale_context: Option<OuterContourScaleContext>,
    pixel_scale: f64,
) -> Option<RobustFit> {
    constrained_ransac_ellipse(points, random, scale_context, pixel_scale, None)
}

/// Geometry from the ordered boundary. Only trusted arcs constrain the fit;
/// censored mask extensions into an eyelid must never attract the conic.
pub(crate) struct ConicArcConstraints {
    pub(crate) tangents: Vec<Option<(f64, f64)>>,
}

impl ConicArcConstraints {
    pub(crate) fn tangent_agrees(&self, index: usize, point: (f64, f64), ellipse: Ellipse) -> bool {
        let Some(tangent) = self.tangents[index] else {
            return false;
        };
        let p = ellipse_axis_point(point, ellipse);
        let (s, c) = ellipse.angle.sin_cos();
        let normal = (
            c * p.0 / ellipse.major_radius.powi(2) - s * p.1 / ellipse.minor_radius.powi(2),
            s * p.0 / ellipse.major_radius.powi(2) + c * p.1 / ellipse.minor_radius.powi(2),
        );
        // The observed tangent must be within 25 degrees of the conic's
        // tangent. Position agreement alone can join incompatible lid arcs.
        (normal.0 * tangent.0 + normal.1 * tangent.1).abs()
            <= 0.423 * normal.0.hypot(normal.1) * tangent.0.hypot(tangent.1)
    }

    pub(crate) fn admits(
        &self,
        points: &[(f64, f64)],
        inliers: &[bool],
        ellipse: Ellipse,
        tolerance: f64,
    ) -> bool {
        let mut phases = points
            .iter()
            .zip(inliers)
            .filter_map(|(&p, &keep)| {
                if !keep {
                    return None;
                }
                let p = ellipse_axis_point(p, ellipse);
                Some(
                    (p.1 / ellipse.minor_radius)
                        .atan2(p.0 / ellipse.major_radius)
                        .rem_euclid(std::f64::consts::TAU),
                )
            })
            .collect::<Vec<_>>();
        if phases.len() < 20 {
            return false;
        }
        phases.sort_unstable_by(f64::total_cmp);
        let max_gap = phases.windows(2).map(|p| p[1] - p[0]).fold(
            phases[0] + std::f64::consts::TAU - phases[phases.len() - 1],
            f64::max,
        );
        // One short arc admits many radically different conics. Require
        // observations spanning at least half the candidate's circumference.
        if max_gap > std::f64::consts::PI {
            return false;
        }
        // Linearized conic information, expressed in normal pixel distance.
        // Unlike an angle/radius parameterization this stays nonsingular for
        // circles. Opposed but nearly straight tiny arcs still leave the poles
        // unconstrained; counting points or occupied quadrants misses that.
        let row = |phase: f64| {
            let (v, u) = phase.sin_cos();
            let gradient = 2.0 * (u / ellipse.major_radius).hypot(v / ellipse.minor_radius);
            [u * u, v * v, 2.0 * u * v, u, v].map(|value| value / gradient)
        };
        let mut information = [[0.0; 5]; 5];
        for &phase in &phases {
            let h = row(phase);
            for i in 0..5 {
                for j in 0..5 {
                    information[i][j] += h[i] * h[j];
                }
            }
        }
        let mut inverse = [[0.0; 5]; 5];
        for column in 0..5 {
            let mut unit = [0.0; 5];
            unit[column] = 1.0;
            let Some(solution) = solve_five(information, unit) else {
                return false;
            };
            for i in 0..5 {
                inverse[i][column] = solution[i];
            }
        }
        // Use the residual ceiling as conservative effective noise, allowing
        // for correlation between adjacent contour samples. This is a local
        // conditioning test, not a calibrated statistical confidence interval.
        let limit = (0.10 * ellipse.minor_radius).max(tolerance);
        (0..24).all(|i| {
            let h = row(std::f64::consts::TAU * i as f64 / 24.0);
            let variance = (0..5)
                .map(|i| (0..5).map(|j| h[i] * inverse[i][j] * h[j]).sum::<f64>())
                .sum::<f64>();
            variance.is_finite()
                && variance >= -1e-8
                && tolerance * variance.max(0.0).sqrt() <= limit
        })
    }
}

pub(crate) fn constrained_ransac_ellipse(
    points: &[(f64, f64)],
    random: &mut NumpyPcg64,
    scale_context: Option<OuterContourScaleContext>,
    pixel_scale: f64,
    constraints: Option<&ConicArcConstraints>,
) -> Option<RobustFit> {
    if points.len() < 5 {
        return None;
    }
    let mut best: Option<(isize, f64, Ellipse, Vec<bool>)> = None;
    for _ in 0..4000 {
        let indices = random.choice_five(points.len())?;
        let subset = indices.map(|index| points[index]);
        let Some(ellipse) = least_squares_ellipse(&subset) else {
            continue;
        };
        if scale_context.is_some_and(|context| !context.admits(ellipse)) {
            continue;
        }
        let residuals = points
            .iter()
            .map(|&point| ellipse_residual(point, ellipse))
            .collect::<Vec<_>>();
        let inliers = residuals
            .iter()
            .enumerate()
            .map(|(i, residual)| {
                *residual <= 2.5 * pixel_scale
                    && constraints.is_none_or(|g| g.tangent_agrees(i, points[i], ellipse))
            })
            .collect::<Vec<_>>();
        let count = inliers.iter().filter(|&&inlier| inlier).count();
        if count < 5 {
            continue;
        }
        if constraints.is_some_and(|g| !g.admits(points, &inliers, ellipse, 4.0 * pixel_scale)) {
            continue;
        }
        // Occlusion removes disk area. A hypothesis that cuts inside other
        // uncensored arcs is less plausible than one completing behind them.
        // This is a ranking cost, not a veto based on the whole SAM mask:
        // explicitly censored lid/chord points never reach this search.
        let outside = if constraints.is_some() {
            points
                .iter()
                .zip(&residuals)
                .filter(|(p, r)| **r > 4.0 * pixel_scale && ellipse_coordinate(**p, ellipse) > 1.0)
                .count()
        } else {
            0
        };
        let score = count as isize - 2 * outside as isize;
        let sum = residuals
            .iter()
            .zip(inliers.iter())
            .filter_map(|(residual, inlier)| inlier.then_some(*residual))
            .sum::<f64>();
        if best.as_ref().is_none_or(|candidate| {
            score > candidate.0 || (score == candidate.0 && sum < candidate.1)
        }) {
            best = Some((score, sum, ellipse, inliers));
        }
    }
    let (_, _, best_ellipse, first_inliers) = best?;
    let first_points = points
        .iter()
        .zip(first_inliers.iter())
        .filter_map(|(&point, &inlier)| inlier.then_some(point))
        .collect::<Vec<_>>();
    let mut ellipse = least_squares_ellipse(&first_points)
        .filter(|ellipse| scale_context.is_none_or(|context| context.admits(*ellipse)))
        .filter(|ellipse| {
            constraints
                .is_none_or(|g| g.admits(points, &first_inliers, *ellipse, 4.0 * pixel_scale))
        })
        .unwrap_or(best_ellipse);
    let mut residuals = points
        .iter()
        .map(|&point| ellipse_residual(point, ellipse))
        .collect::<Vec<_>>();
    let residual_median = median(residuals.clone());
    let mad = median(
        residuals
            .iter()
            .map(|residual| (residual - residual_median).abs())
            .collect(),
    );
    // Outliers must not enlarge their own admission threshold without bound.
    // Tolerances follow source pixels when a small contour is normalized.
    let cutoff = (residual_median + 3.5 * (1.4826 * mad).max(0.5 * pixel_scale))
        .clamp(2.5 * pixel_scale, 4.0 * pixel_scale);
    let mut inliers = residuals
        .iter()
        .enumerate()
        .map(|(i, residual)| {
            *residual <= cutoff
                && constraints.is_none_or(|g| g.tangent_agrees(i, points[i], ellipse))
        })
        .collect::<Vec<_>>();
    let final_points = points
        .iter()
        .zip(inliers.iter())
        .filter_map(|(&point, &inlier)| inlier.then_some(point))
        .collect::<Vec<_>>();
    ellipse = least_squares_ellipse(&final_points)
        .filter(|ellipse| scale_context.is_none_or(|context| context.admits(*ellipse)))
        .filter(|ellipse| {
            constraints.is_none_or(|g| g.admits(points, &inliers, *ellipse, 4.0 * pixel_scale))
        })
        .unwrap_or(ellipse);
    residuals = points
        .iter()
        .map(|&point| ellipse_residual(point, ellipse))
        .collect();
    inliers = residuals
        .iter()
        .enumerate()
        .map(|(i, residual)| {
            *residual <= cutoff
                && constraints.is_none_or(|g| g.tangent_agrees(i, points[i], ellipse))
        })
        .collect();
    if constraints.is_some_and(|g| !g.admits(points, &inliers, ellipse, 4.0 * pixel_scale)) {
        return None;
    }
    Some(RobustFit {
        ellipse,
        residuals,
        inliers,
        cutoff,
    })
}

pub(crate) fn robust_ransac_ellipse(
    points: &[(f64, f64)],
    random: &mut NumpyPcg64,
) -> Option<RobustFit> {
    robust_ransac_ellipse_with_context(points, random, None)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EllipseSupportSummary {
    pub(crate) occupied_sectors: usize,
    pub(crate) largest_empty_run: usize,
    pub(crate) opposite_pairs: usize,
}

pub(crate) fn ellipse_support_summary(
    points: &[(f64, f64)],
    ellipse: Ellipse,
) -> EllipseSupportSummary {
    pub(crate) const SECTORS: usize = 16;
    let mut occupied = [false; SECTORS];
    let (sine, cosine) = ellipse.angle.sin_cos();
    for &(x, y) in points {
        let dx = x - ellipse.center.0;
        let dy = y - ellipse.center.1;
        let local_x = cosine * dx + sine * dy;
        let local_y = -sine * dx + cosine * dy;
        let phase = (local_y / ellipse.minor_radius.max(1.0))
            .atan2(local_x / ellipse.major_radius.max(1.0));
        let unit = (phase + std::f64::consts::PI) / std::f64::consts::TAU;
        let sector =
            ((unit * SECTORS as f64).floor() as isize).rem_euclid(SECTORS as isize) as usize;
        occupied[sector] = true;
    }
    let occupied_sectors = occupied.iter().filter(|&&value| value).count();
    let opposite_pairs = (0..SECTORS / 2)
        .filter(|&sector| occupied[sector] && occupied[sector + SECTORS / 2])
        .count();
    let mut largest_empty_run = 0;
    let mut run = 0;
    for index in 0..SECTORS * 2 {
        if occupied[index % SECTORS] {
            run = 0;
        } else {
            run += 1;
            largest_empty_run = largest_empty_run.max(run.min(SECTORS));
        }
    }
    EllipseSupportSummary {
        occupied_sectors,
        largest_empty_run,
        opposite_pairs,
    }
}

pub(crate) fn median(mut values: Vec<f64>) -> f64 {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(|first, second| first.total_cmp(second));
    if values.len() & 1 == 1 {
        values[values.len() / 2]
    } else {
        0.5 * (values[values.len() / 2 - 1] + values[values.len() / 2])
    }
}
