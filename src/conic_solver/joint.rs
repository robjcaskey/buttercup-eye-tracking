//! Sparse joint projected-circle satisfaction, with ONE latent fixation.
//!
//! Actual boundary samples are the observation factors. Prefitted ellipses
//! initialize hypotheses but never contribute a second vote for their pixels.
//! Each hypothesis optimizes the shared target and both eyes' bounded nuisance
//! geometry together. There are no independently solved gaze points to average.
//!
//! Camera/scale, coplanarity, pupil depth and decentration are explicit model
//! assumptions. Results and competing-hypothesis diagnostics are conditional on those
//! assumptions, not calibrated gaze probabilities or measured anatomy.

use crate::geometry::{add3, cross3, dot3, norm3, normalized3, scale3, sub3, Ellipse};
use crate::roi_evidence::{
    BoundaryKind, BoundaryLevelSetObservation, BoundaryNormalObservation, ExposureKey,
    RoiConicEvidence,
};

mod mask_levels;
pub(crate) mod posterior;
#[cfg(test)]
mod tests;
pub(crate) mod uncertainty;
use mask_levels::Selection;

const TARGET_PARAMETERS: usize = 3;
const EYE_PARAMETERS: usize = 11;
const PARAMETERS: usize = TARGET_PARAMETERS + 2 * EYE_PARAMETERS;
const MAX_GROUPS_PER_EYE: usize = 32;
const MAX_ALTERNATIVES_PER_GROUP: usize = 4;
const MAX_POINTS_PER_ARC: usize = 16;
const MAX_HYPOTHESES: usize = 24;
const MAX_REFINEMENTS: usize = 16;
const HUBER_TRANSITION: f64 = 3.0;
const MAXIMUM_GROUP_COST: f64 = 9.0;
/// Conservative native-image correlation length, not a count of detector
/// samples. Splitting/resampling a contour must not create more evidence.
const SUPPORT_CORRELATION_LENGTH_PX: f64 = 32.0;
/// Correlated contour direction is a compatibility cue, not an independent
/// precise angle measurement. This support allowance is not a statistical CI.
const BOUNDARY_DIRECTION_ALLOWANCE_SIGMAS: f64 = 2.0;

/// Bounded engineering support with a soft center. This is not a confidence
/// interval. All bounds are frozen before the candidate is evaluated.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScalarSupport {
    pub(crate) nominal: f64,
    pub(crate) minimum: f64,
    pub(crate) maximum: f64,
    pub(crate) sigma: f64,
}

impl ScalarSupport {
    fn valid(self) -> bool {
        [self.nominal, self.minimum, self.maximum, self.sigma]
            .into_iter()
            .all(f64::is_finite)
            && self.minimum <= self.nominal
            && self.nominal <= self.maximum
            && self.minimum < self.maximum
            && self.sigma > 0.0
    }
}

/// Optical center at the origin; sensor-right/down, +Z TOWARD the camera.
/// Visible scene points therefore have negative Z. Intrinsics are native full
/// sensor pixels, never thumbnail/model/ROI pixels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PinholeCamera {
    pub(crate) focal_px: [f64; 2],
    pub(crate) principal_px: [f64; 2],
}

impl PinholeCamera {
    pub(crate) fn project(self, p: [f64; 3]) -> Option<[f64; 2]> {
        if !p.into_iter().all(f64::is_finite) || p[2] >= -1.0e-6 {
            return None;
        }
        Some([
            self.principal_px[0] - self.focal_px[0] * p[0] / p[2],
            self.principal_px[1] - self.focal_px[1] * p[1] / p[2],
        ])
    }

    pub(crate) fn unproject(self, sensor_px: [f64; 2], distance_mm: f64) -> [f64; 3] {
        [
            (sensor_px[0] - self.principal_px[0]) * distance_mm / self.focal_px[0],
            (sensor_px[1] - self.principal_px[1]) * distance_mm / self.focal_px[1],
            -distance_mm,
        ]
    }

    fn valid(self) -> bool {
        self.focal_px.into_iter().all(|f| f.is_finite() && f > 0.0)
            && self.principal_px.into_iter().all(f64::is_finite)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum TransversePositionFrame {
    /// Independent metric XYZ support, e.g. a transported effective pivot.
    Cartesian,
    /// X/Y errors measured where the viewing ray crosses the nominal depth.
    /// Range uncertainty changes X/Y together; it must not pin a point to a
    /// fictitious Cartesian column when the camera is viewing off axis.
    AtNominalDepth,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PositionSupport {
    pub(crate) camera_mm: [f64; 3],
    pub(crate) sigma_mm: [f64; 3],
    pub(crate) maximum_displacement_mm: [f64; 3],
    pub(crate) transverse_frame: TransversePositionFrame,
}

impl PositionSupport {
    fn valid(self) -> bool {
        self.camera_mm.into_iter().all(f64::is_finite)
            && self.sigma_mm.into_iter().all(|v| v.is_finite() && v > 0.0)
            && self
                .maximum_displacement_mm
                .into_iter()
                .all(|v| v.is_finite() && v > 0.0)
    }

    fn displacement(self, point: [f64; 3]) -> [f64; 3] {
        let ratio = match self.transverse_frame {
            TransversePositionFrame::Cartesian => 1.0,
            TransversePositionFrame::AtNominalDepth => self.camera_mm[2] / point[2],
        };
        [
            point[0] * ratio - self.camera_mm[0],
            point[1] * ratio - self.camera_mm[1],
            point[2] - self.camera_mm[2],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SurfaceAxisAlignment {
    /// Small transverse angles from a fixation ray to the imaged iris normal.
    /// Not a calibrated clinical kappa measurement; includes model mismatch.
    pub(crate) nominal_radians: [f64; 2],
    pub(crate) sigma_radians: [f64; 2],
    pub(crate) maximum_deviation_radians: [f64; 2],
}

impl SurfaceAxisAlignment {
    fn valid(self) -> bool {
        self.nominal_radians.into_iter().all(f64::is_finite)
            && self
                .sigma_radians
                .into_iter()
                .all(|v| v.is_finite() && v > 0.0)
            && self
                .maximum_deviation_radians
                .into_iter()
                .all(|v| v.is_finite() && (0.0..=0.35).contains(&v))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EyeScenePrior {
    /// Limbus plane center, not pupil-aperture center or effective pivot.
    pub(crate) limbus_center: PositionSupport,
    /// Outer limbus, inner limbus, pupil aperture. Radius bounds are in mm.
    pub(crate) radii_mm: [ScalarSupport; 3],
    /// Positive inward distance from limbus to pupil plane. A zero/uncertain
    /// depth or free decentration cannot magically supply sign information.
    pub(crate) pupil_inward_depth_mm: ScalarSupport,
    pub(crate) pupil_decentration_sigma_mm: f64,
    pub(crate) pupil_maximum_decentration_mm: f64,
    /// Optional independently transported effective pivot, with soft mobility.
    /// Never construct this prior from the current candidate's chosen sign.
    pub(crate) effective_pivot: Option<PositionSupport>,
    pub(crate) limbus_to_pivot_mm: f64,
    /// None is the exact-coincidence idealization used in synthetic controls.
    /// Real uncalibrated eyes must not silently treat surface and visual axes
    /// as identical and deform observed boundaries to enforce that fiction.
    pub(crate) surface_axis_alignment: Option<SurfaceAxisAlignment>,
}

impl EyeScenePrior {
    fn valid(self) -> bool {
        self.limbus_center.valid()
            && self.limbus_center.camera_mm[2] < 0.0
            && self
                .radii_mm
                .into_iter()
                .all(|r| r.valid() && r.minimum > 0.0)
            && self.pupil_inward_depth_mm.valid()
            && self.pupil_inward_depth_mm.minimum >= 0.0
            && self.pupil_decentration_sigma_mm.is_finite()
            && self.pupil_decentration_sigma_mm > 0.0
            && self.pupil_maximum_decentration_mm.is_finite()
            && self.pupil_maximum_decentration_mm >= 0.0
            && self.effective_pivot.map_or(true, PositionSupport::valid)
            && self.limbus_to_pivot_mm.is_finite()
            && self.limbus_to_pivot_mm >= 0.0
            && self
                .surface_axis_alignment
                .map_or(true, SurfaceAxisAlignment::valid)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct JointScenePrior {
    pub(crate) camera: PinholeCamera,
    pub(crate) eyes: [Option<EyeScenePrior>; 2],
    /// Reference point only parameterizes the target; no observation vote.
    pub(crate) target_reference_camera_mm: [f64; 3],
    /// Positive axial distance from the reference along its reference-to-
    /// camera direction, NOT optical-axis Z separation or Euclidean range.
    /// Together with the viewpoint slope envelope this bounds target range.
    pub(crate) fixation_axial_distance_mm: ScalarSupport,
    /// Optional previous joint result used as a seed only, never as new pixels.
    pub(crate) target_seed_camera_mm: Option<[f64; 3]>,
    /// A distinct previous read may provide a better start than the newest
    /// weak/occluded one. This competes within the SAME bounded search; its
    /// target is never averaged with another target or added as a residual.
    pub(crate) secondary_target_seed_camera_mm: Option<[f64; 3]>,
    /// Tangent-plane ray slopes around the reference-to-camera direction,
    /// NOT slopes around the camera's optical +Z axis. A visible off-axis eye
    /// can have a large camera-axis slope with only modest foreshortening.
    /// This is an engineering search envelope, not a measured head-relative
    /// biological rotation limit (the head frame is not supplied here).
    pub(crate) maximum_gaze_slope: f64,
    /// Optional independently supported eye-center separation. A coarse IPD
    /// is a model prior, not a measurement of the current iris radii.
    pub(crate) interocular_distance_mm: Option<ScalarSupport>,
    /// Boundary localization noise MEASURED from this tracking session's own
    /// solved residuals, per boundary kind (outer limbus, inner limbus,
    /// pupil). When present it replaces the engineering optical/band
    /// allowances for that kind; timing uncertainty is still added. None
    /// keeps the engineering allowance (no residuals measured yet).
    pub(crate) measured_boundary_sigma_px: [Option<f64>; 3],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct JointConicRequest<'a> {
    pub(crate) eyes: [Option<RoiConicEvidence<'a>>; 2],
    pub(crate) scene: &'a JointScenePrior,
    pub(crate) maximum_hypotheses: usize,
    pub(crate) maximum_refinements: usize,
    pub(crate) maximum_source_skew_ns: u64,
    /// Source/rolling-row uncertainty supplied as an engineering allowance,
    /// NOT as an attested hardware timestamp bound.
    pub(crate) exposure_uncertainty_ns: u64,
    pub(crate) motion_bound_px_per_second: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JointConicUnavailable {
    InvalidRequest,
    IncompatibleClocks,
    ExcessiveSourceSkew,
    NoBoundaryEvidence,
    NoFeasibleInitialization,
    NoFeasibleHypothesis,
    MountingAssumptionConflict,
}

#[derive(Clone, Debug)]
pub(crate) struct ArcSupport {
    pub(crate) exposure: ExposureKey,
    pub(crate) evidence_group: u32,
    pub(crate) arc_index: usize,
    /// Exact bounded residual samples of the selected arc alternative and
    /// mask level, in this exposure's ROI coordinates. Diagnostic only.
    pub(crate) points_roi_px: Vec<(f64, f64)>,
    pub(crate) kind: BoundaryKind,
    pub(crate) rms_px: f64,
    pub(crate) sigma_px: f64,
    pub(crate) support_length_px: f64,
    pub(crate) evidence_weight: f64,
    pub(crate) boundary_normal_samples: usize,
    pub(crate) boundary_normal_rms_radians: Option<f64>,
    pub(crate) mask_level: Option<i8>,
    pub(crate) used: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct JointConicSolution {
    /// Test-only attribution of the exact native residual vector. It never
    /// changes the objective, factor selection, or published gaze.
    #[cfg(test)]
    pub(crate) factor_costs: Option<serde_json::Value>,
    /// Local shape/pose uncertainty after marginalizing nuisance geometry.
    /// Mirror alternatives and empirical model error remain separate.
    pub(crate) local_uncertainty: Option<uncertainty::LocalUncertainty>,
    pub(crate) posterior: Option<posterior::ModelPosterior>,
    pub(crate) mask_level_families: Vec<serde_json::Value>,
    pub(crate) arc_alternative_marginals: Vec<serde_json::Value>,
    pub(crate) target_camera_mm: [f64; 3],
    /// Viewpoint-chart diagnostics, not a gaze covariance or a quality vote.
    pub(crate) target_reference_camera_mm: [f64; 3],
    pub(crate) target_viewpoint_axial_distance_mm: f64,
    pub(crate) target_viewpoint_slopes: [f64; 2],
    pub(crate) target_viewpoint_slope_limit: f64,
    pub(crate) target_viewpoint_bounds_active: [bool; 2],
    pub(crate) eye_centers_camera_mm: [Option<[f64; 3]>; 2],
    pub(crate) eye_normals: [Option<[f64; 3]>; 2],
    pub(crate) eye_gaze_directions: [Option<[f64; 3]>; 2],
    pub(crate) surface_axis_alignment_radians: [Option<[f64; 2]>; 2],
    pub(crate) effective_pivots_camera_mm: [Option<[f64; 3]>; 2],
    /// Source-native conics resulting from the shared solution, outer/inner/pupil.
    pub(crate) ellipses_roi_px: [[Option<Ellipse>; 3]; 2],
    pub(crate) arcs: Vec<ArcSupport>,
    pub(crate) contributing_eyes: [bool; 2],
    /// Which ROI-to-eye associations this hypothesis models. False means
    /// unlocalized, not a closed eye, concave surface, or fabricated ellipse.
    pub(crate) modeled_eyes: [bool; 2],
    /// Full capped group cost paid for each explicitly unlocalized ROI.
    /// Model selection remains a heuristic, not a calibrated probability.
    pub(crate) unlocalized_eye_cost: [f64; 2],
    pub(crate) robust_cost: f64,
    /// Difference to a geometrically distinct optimized hypothesis. None means
    /// no independently explored competing basin, NOT certainty.
    pub(crate) alternative_cost_margin: Option<f64>,
    pub(crate) alternative_target_camera_mm: Option<[f64; 3]>,
    pub(crate) hypotheses_evaluated: usize,
    /// Refined starts for [both ROIs, right only, left only]. Includes failed
    /// starts; these counts share the single caller-supplied work budget.
    pub(crate) hypotheses_by_association: [usize; 3],
    pub(crate) refinement_steps: usize,
}

/// A homogeneous conic evaluated in native ROI pixels. Coefficients are
/// [x², xy, y², x, y, constant]; xy is the full coefficient, not half.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProjectedCircle(pub(crate) [f64; 6], Option<EllipseDistance>);

/// Prepared once per projected conic, not once per observation. This is an
/// offline distance-metric experiment; the live conic still uses Sampson.
#[derive(Clone, Copy, Debug)]
struct EllipseDistance {
    center: (f64, f64),
    cosine: f64,
    sine: f64,
    major: f64,
    minor: f64,
}

impl EllipseDistance {
    fn new(ellipse: Ellipse) -> Self {
        let (sine, cosine) = ellipse.angle.sin_cos();
        Self {
            center: ellipse.center,
            cosine,
            sine,
            major: ellipse.major_radius,
            minor: ellipse.minor_radius,
        }
    }

    fn residual_px(self, point: (f64, f64)) -> f64 {
        let dx = point.0 - self.center.0;
        let dy = point.1 - self.center.1;
        let x = (self.cosine * dx + self.sine * dy).abs();
        let y = (-self.sine * dx + self.cosine * dy).abs();
        let (a, b) = (self.major, self.minor);
        if a == b {
            return x.hypot(y) - a;
        }
        let z0 = x / a;
        let z1 = y / b;
        let signed_level = z0 * z0 + z1 * z1 - 1.0;
        if signed_level == 0.0 {
            return 0.0;
        }
        let distance = if y <= 8.0 * f64::EPSILON * a {
            // An interior major-axis point may be closest to an off-axis
            // boundary point. The center's distance is the minor radius.
            let denominator = a * a - b * b;
            let ratio = a * x / denominator;
            if ratio < 1.0 {
                (a * ratio - x).hypot(b * (1.0 - ratio * ratio).sqrt())
            } else {
                (a - x).abs()
            }
        } else if x == 0.0 {
            (b - y).abs()
        } else {
            // The closest point satisfies the ellipse constraint with one
            // Lagrange multiplier. Its root is monotone on this bracket,
            // for both interior and exterior observations. Safeguarded Newton
            // avoids a quartic solve and retains a finite bisection fallback.
            let r0 = (a / b).powi(2);
            let numerator = r0 * z0;
            let mut lower = z1 - 1.0;
            let mut upper = if signed_level < 0.0 {
                0.0
            } else {
                numerator.hypot(z1) - 1.0
            };
            let mut t = 0.0_f64.clamp(lower, upper);
            for _ in 0..64 {
                let u = numerator / (t + r0);
                let v = z1 / (t + 1.0);
                let value = u * u + v * v - 1.0;
                if value.abs() <= 8.0 * f64::EPSILON {
                    break;
                }
                if value > 0.0 {
                    lower = t;
                } else {
                    upper = t;
                }
                let derivative = -2.0 * (u * u / (t + r0) + v * v / (t + 1.0));
                let proposed = t - value / derivative;
                let next = if proposed > lower && proposed < upper {
                    proposed
                } else {
                    0.5 * (lower + upper)
                };
                if next == t {
                    break;
                }
                t = next;
            }
            let closest_x = r0 * x / (t + r0);
            let closest_y = y / (t + 1.0);
            (closest_x - x).hypot(closest_y - y)
        };
        distance.copysign(signed_level)
    }
}

impl ProjectedCircle {
    pub(crate) fn project(
        camera: PinholeCamera,
        center: [f64; 3],
        normal: [f64; 3],
        radius_mm: f64,
        origin_px: [u32; 2],
    ) -> Option<Self> {
        if !camera.valid()
            || !radius_mm.is_finite()
            || radius_mm <= 0.0
            || center[2] + radius_mm >= -1.0e-6
            || normal[2] <= 0.0
            || dot3(normal, scale3(center, -1.0)) <= 0.0
        {
            return None;
        }
        // Intersect an image ray r with n·X=k, then impose |X-C|²=radius².
        // The resulting matrix is k²I-k(Cn'+nC')+(C'C-radius²)nn'.
        let k = dot3(normal, center);
        let q = dot3(center, center) - radius_mm * radius_mm;
        let mut h = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = if i == j { k * k } else { 0.0 }
                    - k * (center[i] * normal[j] + normal[i] * center[j])
                    + q * normal[i] * normal[j];
            }
        }
        let rays = [
            [1.0 / camera.focal_px[0], 0.0, 0.0],
            [0.0, 1.0 / camera.focal_px[1], 0.0],
            [
                (origin_px[0] as f64 - camera.principal_px[0]) / camera.focal_px[0],
                (origin_px[1] as f64 - camera.principal_px[1]) / camera.focal_px[1],
                -1.0,
            ],
        ];
        let bilinear = |a: [f64; 3], b: [f64; 3]| -> f64 {
            (0..3)
                .map(|i| (0..3).map(|j| a[i] * h[i][j] * b[j]).sum::<f64>())
                .sum()
        };
        let coefficients = [
            bilinear(rays[0], rays[0]),
            2.0 * bilinear(rays[0], rays[1]),
            bilinear(rays[1], rays[1]),
            2.0 * bilinear(rays[0], rays[2]),
            2.0 * bilinear(rays[1], rays[2]),
            bilinear(rays[2], rays[2]),
        ];
        coefficients
            .into_iter()
            .all(f64::is_finite)
            .then_some(Self(coefficients, None))
    }

    fn with_exact_distance(mut self, enabled: bool) -> Option<Self> {
        self.1 = if enabled {
            Some(EllipseDistance::new(self.ellipse()?))
        } else {
            None
        };
        Some(self)
    }

    pub(crate) fn residual_px(self, point: (f64, f64)) -> f64 {
        if let Some(distance) = self.1 {
            return distance.residual_px(point);
        }
        let [a, b, c, d, e, f] = self.0;
        let (x, y) = point;
        let gradient = (2.0 * a * x + b * y + d).hypot(b * x + 2.0 * c * y + e);
        if gradient <= 1.0e-20 {
            return 1.0e6;
        }
        (a * x * x + b * x * y + c * y * y + d * x + e * y + f) / gradient
    }

    pub(crate) fn ellipse(self) -> Option<Ellipse> {
        let [a, b, c, d, e, f] = self.0;
        let determinant = 4.0 * a * c - b * b;
        if determinant <= 1.0e-24 || a <= 0.0 {
            return None;
        }
        let x = (b * e - 2.0 * c * d) / determinant;
        let y = (b * d - 2.0 * a * e) / determinant;
        let level = -(f + 0.5 * (d * x + e * y));
        let spread = (a - c).hypot(b);
        let small = (a + c - spread) * 0.5;
        let large = (a + c + spread) * 0.5;
        if small <= 0.0 || level <= 0.0 {
            return None;
        }
        let angle = if b.abs() + (a - c).abs() < 1.0e-20 {
            0.0
        } else {
            0.5 * b.atan2(a - c) + std::f64::consts::FRAC_PI_2
        };
        Some(Ellipse {
            center: (x, y),
            major_radius: (level / small).sqrt(),
            minor_radius: (level / large).sqrt(),
            angle,
        })
    }
}

fn boundary_index(kind: BoundaryKind) -> Option<usize> {
    match kind {
        BoundaryKind::OuterLimbus => Some(0),
        BoundaryKind::InnerLimbus => Some(1),
        BoundaryKind::PupillaryBoundary => Some(2),
        BoundaryKind::Unclassified => None,
    }
}

/// Two camera-facing circle-plane normals from a calibrated image conic.
/// This conditions on circularity and intrinsics, not on an inferred gaze
/// point. The ellipse center is NOT assumed to be the projected circle center.
pub(crate) fn circle_normal_hypotheses(
    camera: PinholeCamera,
    ellipse: Ellipse,
    origin: [u32; 2],
) -> Option<[[f64; 3]; 2]> {
    circle_pose_hypotheses(camera, ellipse, origin).map(|poses| poses.map(|p| p.normal))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CirclePoseSeed {
    pub(crate) normal: [f64; 3],
    pub(crate) center_per_radius: [f64; 3],
}

/// A conic determines two circular sections up to one metric scale. Keep that
/// scale conditional on the caller's frozen scene support; this is a start,
/// not another pixel factor or an independent scale measurement.
pub(crate) fn circle_pose_hypotheses(
    camera: PinholeCamera,
    ellipse: Ellipse,
    origin: [u32; 2],
) -> Option<[CirclePoseSeed; 2]> {
    if !camera.valid()
        || !ellipse.major_radius.is_finite()
        || !ellipse.minor_radius.is_finite()
        || ellipse.minor_radius <= 0.0
        || ellipse.major_radius < ellipse.minor_radius
        || !ellipse.angle.is_finite()
        || !ellipse.center.0.is_finite()
        || !ellipse.center.1.is_finite()
    {
        return None;
    }
    let (s, c) = ellipse.angle.sin_cos();
    let aa = ellipse.major_radius.powi(-2);
    let bb = ellipse.minor_radius.powi(-2);
    let a = c * c * aa + s * s * bb;
    let b = c * s * (aa - bb);
    let d = s * s * aa + c * c * bb;
    let x = ellipse.center.0;
    let y = ellipse.center.1;
    let q = [
        [a, b, -a * x - b * y],
        [b, d, -b * x - d * y],
        [
            -a * x - b * y,
            -b * x - d * y,
            a * x * x + 2.0 * b * x * y + d * y * y - 1.0,
        ],
    ];
    // Homogeneous image vector = M * [X,Y,Z] for +Z toward camera.
    let m = [
        [
            camera.focal_px[0],
            0.0,
            origin[0] as f64 - camera.principal_px[0],
        ],
        [
            0.0,
            camera.focal_px[1],
            origin[1] as f64 - camera.principal_px[1],
        ],
        [0.0, 0.0, -1.0],
    ];
    let mut cone = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            cone[i][j] = (0..3)
                .map(|r| (0..3).map(|t| m[r][i] * q[r][t] * m[t][j]).sum::<f64>())
                .sum();
        }
    }
    let (values, vectors) = symmetric_eigen_3x3(cone)?;
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| values[j].total_cmp(&values[i]));
    let [largest, middle, smallest] = order.map(|i| values[i]);
    if middle <= 0.0 || smallest >= 0.0 {
        return None;
    }
    let first = ((largest - middle) / (largest - smallest))
        .clamp(0.0, 1.0)
        .sqrt();
    let last = ((middle - smallest) / (largest - smallest))
        .clamp(0.0, 1.0)
        .sqrt();
    let mut poses = [CirclePoseSeed {
        normal: [0.0; 3],
        center_per_radius: [0.0; 3],
    }; 2];
    for (i, sign) in [-1.0, 1.0].into_iter().enumerate() {
        let n = std::array::from_fn(|r| {
            sign * first * vectors[r][order[0]] + last * vectors[r][order[2]]
        });
        let n = normalized3(n)?;
        let n = if n[2] < 0.0 { scale3(n, -1.0) } else { n };
        // For plane n.C=k, the middle cone eigenvalue is proportional to k².
        // Hn/k² = -C/k + (|C|²-r²)n/k². Its transverse part gives C/k;
        // n.C/k=1 then fixes its axial part. Choose k<0 to face the camera.
        let hn = std::array::from_fn(|r| (0..3).map(|c| cone[r][c] * n[c] / middle).sum::<f64>());
        let axial = dot3(n, hn);
        let center_per_plane_offset = sub3(scale3(n, 1.0 + axial), hn);
        let radius_squared = dot3(center_per_plane_offset, center_per_plane_offset) - 1.0 - axial;
        if radius_squared <= 0.0 || !radius_squared.is_finite() {
            return None;
        }
        let center_per_radius = scale3(center_per_plane_offset, -radius_squared.sqrt().recip());
        if center_per_radius[2] >= 0.0 {
            return None;
        }
        poses[i] = CirclePoseSeed {
            normal: n,
            center_per_radius,
        };
    }
    Some(poses)
}

pub(crate) fn symmetric_eigen_3x3(mut a: [[f64; 3]; 3]) -> Option<([f64; 3], [[f64; 3]; 3])> {
    if !a.into_iter().flatten().all(f64::is_finite) {
        return None;
    }
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..24 {
        let (p, q) = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .max_by(|&(p, q), &(r, s)| a[p][q].abs().total_cmp(&a[r][s].abs()))?;
        if a[p][q].abs() < 1.0e-14 * (a[0][0].abs() + a[1][1].abs() + a[2][2].abs()).max(1.0) {
            break;
        }
        let angle = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (s, c) = angle.sin_cos();
        let app = c * c * a[p][p] - 2.0 * s * c * a[p][q] + s * s * a[q][q];
        let aqq = s * s * a[p][p] + 2.0 * s * c * a[p][q] + c * c * a[q][q];
        for r in 0..3 {
            if r != p && r != q {
                let arp = c * a[r][p] - s * a[r][q];
                let arq = s * a[r][p] + c * a[r][q];
                a[r][p] = arp;
                a[p][r] = arp;
                a[r][q] = arq;
                a[q][r] = arq;
            }
        }
        a[p][p] = app;
        a[q][q] = aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for row in &mut vectors {
            let vp = c * row[p] - s * row[q];
            let vq = s * row[p] + c * row[q];
            row[p] = vp;
            row[q] = vq;
        }
    }
    Some(([a[0][0], a[1][1], a[2][2]], vectors))
}

#[derive(Clone, Debug)]
struct SparseArc {
    eye: usize,
    index: usize,
    kind: BoundaryKind,
    boundary: usize,
    group: u32,
    points: Vec<(f64, f64)>,
    sigma: f64,
    outward_normals: Vec<Option<BoundaryNormalObservation>>,
    level_sets: Vec<Option<BoundaryLevelSetObservation>>,
    quadrature: Vec<f64>,
    length_px: f64,
    weight: f64,
}

#[derive(Clone, Debug)]
struct ArcResiduals {
    position: [f64; MAX_POINTS_PER_ARC],
    direction: [f64; MAX_POINTS_PER_ARC],
    mean_cost: f64,
}

/// Reuse exact arithmetic at an unchanged projected conic. Borrowing the arc
/// prevents its evidence from changing while the cache exists. Pointer identity
/// also rejects a selection borrowed from a different/conditioned problem.
#[derive(Clone, Debug)]
struct CachedArcResiduals<'a> {
    arc: &'a SparseArc,
    conic_bits: [u64; 6],
    exact_distance: bool,
    values: ArcResiduals,
}

impl CachedArcResiduals<'_> {
    fn at(&self, arc: &SparseArc, conic: ProjectedCircle, level: usize) -> Option<&ArcResiduals> {
        (level == 1
            && std::ptr::eq(self.arc, arc)
            && self.conic_bits == conic.0.map(f64::to_bits)
            && self.exact_distance == conic.1.is_some())
        .then_some(&self.values)
    }
}

impl SparseArc {
    fn evaluated_residuals(&self, conic: ProjectedCircle) -> ArcResiduals {
        let mut result = ArcResiduals {
            position: [0.0; MAX_POINTS_PER_ARC],
            direction: [0.0; MAX_POINTS_PER_ARC],
            mean_cost: 0.0,
        };
        // Keep the original order of operations and summation. This cache is
        // not an approximate norm, reduced sample budget or changed objective.
        result.mean_cost = self
            .points
            .iter()
            .zip(&self.quadrature)
            .enumerate()
            .map(|(i, (_, &w))| {
                let p = self.point_at_level(i, 1);
                let position = robust_residual(conic.residual_px(p) / self.sigma);
                let direction = self.normal_residual_at_level(conic, i, 1);
                result.position[i] = position;
                result.direction[i] = direction;
                w * (position.powi(2) + direction.powi(2))
            })
            .sum();
        result
    }

    fn point_at_level(&self, index: usize, level: usize) -> (f64, f64) {
        let point = self.points[index];
        if let Some(offset) = self.level_sets[index] {
            let shift = if level < 3 {
                offset.displacement_px[level]
            } else {
                offset
                    .spatial_displacement_px
                    .map_or(0.0, |states| states[level - 3])
            };
            if shift != 0.0 {
                return (
                    point.0 + shift * offset.unit_normal_roi[0],
                    point.1 + shift * offset.unit_normal_roi[1],
                );
            }
        }
        point
    }

    fn mean_cost(&self, conic: ProjectedCircle) -> f64 {
        self.mean_cost_at_level(conic, 1)
    }

    fn mean_cost_at_level(&self, conic: ProjectedCircle, level: usize) -> f64 {
        self.points
            .iter()
            .zip(&self.quadrature)
            .enumerate()
            .map(|(i, (_, &w))| {
                let p = self.point_at_level(i, level);
                w * (robust_residual(conic.residual_px(p) / self.sigma).powi(2)
                    + self.normal_residual_at_level(conic, i, level).powi(2))
            })
            .sum()
    }

    fn normal_angle_error(&self, conic: ProjectedCircle, index: usize) -> Option<f64> {
        self.normal_angle_error_at_level(conic, index, 1)
    }

    fn normal_angle_error_at_level(
        &self,
        conic: ProjectedCircle,
        index: usize,
        level: usize,
    ) -> Option<f64> {
        let measured = self.outward_normals[index]?;
        let (x, y) = self.point_at_level(index, level);
        let [a, b, c, d, e, _] = conic.0;
        let gradient = [2.0 * a * x + b * y + d, b * x + 2.0 * c * y + e];
        if gradient[0].hypot(gradient[1]) <= 1.0e-20 {
            return Some(std::f64::consts::PI);
        }
        let n = measured.unit_outward_roi;
        Some(
            (gradient[0] * n[1] - gradient[1] * n[0])
                .atan2(gradient[0] * n[0] + gradient[1] * n[1]),
        )
    }

    fn normal_residual(&self, conic: ProjectedCircle, index: usize) -> f64 {
        self.normal_residual_at_level(conic, index, 1)
    }

    fn normal_residual_at_level(&self, conic: ProjectedCircle, index: usize, level: usize) -> f64 {
        self.normal_angle_error_at_level(conic, index, level)
            .map(|angle| {
                let sigma = self.outward_normals[index].unwrap().angular_sigma_radians;
                let excess = (angle.abs() / sigma - BOUNDARY_DIRECTION_ALLOWANCE_SIGMAS).max(0.0);
                robust_residual(angle.signum() * excess)
            })
            .unwrap_or(0.0)
    }
}

#[derive(Clone)]
struct Group {
    alternatives: Vec<SparseArc>,
    weight: f64,
}

/// Trapezoidal integration along the actual observed polyline. Duplicate
/// points add zero length. Point count itself supplies no information mass.
fn polyline_quadrature(points: &[(f64, f64)]) -> Option<(Vec<f64>, f64)> {
    let mut weights = vec![0.0; points.len()];
    for (i, pair) in points.windows(2).enumerate() {
        let length = (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1);
        weights[i] += 0.5 * length;
        weights[i + 1] += 0.5 * length;
    }
    let length = weights.iter().sum::<f64>();
    if !length.is_finite() || length <= 1.0e-9 {
        return None;
    }
    for weight in &mut weights {
        *weight /= length;
    }
    Some((weights, length))
}
type Parameters = [f64; PARAMETERS];

/// A viewpoint-relative ray chart: two tangent slopes and log axial distance
/// along the SAME reference-to-camera axis. Mixing viewpoint slopes with
/// optical-Z depth divides by a near-zero optical-Z ray component and admits
/// unbounded target ranges even though both configured bounds are finite.
/// Camera-facing surface checks still apply separately to each eye.
#[derive(Clone, Copy, Debug)]
struct ViewpointRayChart {
    origin_camera_mm: [f64; 3],
    right: [f64; 3],
    down: [f64; 3],
    toward_camera: [f64; 3],
}

impl ViewpointRayChart {
    fn new(origin_camera_mm: [f64; 3]) -> Option<Self> {
        if !origin_camera_mm.into_iter().all(f64::is_finite) || origin_camera_mm[2] >= -1.0e-6 {
            return None;
        }
        let toward_camera = normalized3(scale3(origin_camera_mm, -1.0))?;
        let right = normalized3(cross3([0.0, 1.0, 0.0], toward_camera))?;
        let down = cross3(toward_camera, right);
        Some(Self {
            origin_camera_mm,
            right,
            down,
            toward_camera,
        })
    }

    fn coordinates(self, target: [f64; 3]) -> Option<[f64; 3]> {
        let delta = sub3(target, self.origin_camera_mm);
        let forward = dot3(delta, self.toward_camera);
        if !delta.into_iter().all(f64::is_finite) || delta[2] <= 0.0 || forward <= 1.0e-9 {
            return None;
        }
        Some([
            dot3(delta, self.right) / forward,
            dot3(delta, self.down) / forward,
            forward.ln(),
        ])
    }

    fn target(self, coordinates: [f64; 3]) -> Option<[f64; 3]> {
        let ray = add3(
            self.toward_camera,
            add3(
                scale3(self.right, coordinates[0]),
                scale3(self.down, coordinates[1]),
            ),
        );
        let axial_distance = coordinates[2].exp();
        if !ray.into_iter().all(f64::is_finite)
            || ray[2] <= 1.0e-6
            || !axial_distance.is_finite()
            || axial_distance <= 0.0
        {
            return None;
        }
        let target = add3(self.origin_camera_mm, scale3(ray, axial_distance));
        target.into_iter().all(f64::is_finite).then_some(target)
    }
}

#[derive(Clone)]
struct Problem<'a> {
    request: JointConicRequest<'a>,
    target_chart: ViewpointRayChart,
    groups: Vec<Group>,
    present: [bool; 2],
    initial: Parameters,
    lower: Parameters,
    upper: Parameters,
    scales: Parameters,
    marginalize_arc_alternatives: bool,
    exact_conic_distances: bool,
}

impl<'a> Problem<'a> {
    fn with_exact_distances(mut self, enabled: bool) -> Self {
        self.exact_conic_distances = enabled;
        self
    }
    fn with_arc_marginalization(mut self, enabled: bool) -> Self {
        self.marginalize_arc_alternatives = enabled;
        self
    }
    fn unlocalized_costs(&self, modeled: [bool; 2]) -> [f64; 2] {
        let mut cost = [0.0; 2];
        for group in &self.groups {
            let eye = group.alternatives[0].eye;
            if !modeled[eye] {
                cost[eye] += group.weight * MAXIMUM_GROUP_COST;
            }
        }
        cost
    }

    fn new(request: JointConicRequest<'a>) -> Result<Self, JointConicUnavailable> {
        let scene = request.scene;
        if request.maximum_hypotheses == 0
            || request.maximum_refinements == 0
            || !scene.camera.valid()
            || !scene.fixation_axial_distance_mm.valid()
            || scene.fixation_axial_distance_mm.minimum <= 0.0
            || !scene.maximum_gaze_slope.is_finite()
            || scene.maximum_gaze_slope <= 0.0
            || !scene
                .target_reference_camera_mm
                .into_iter()
                .all(f64::is_finite)
            || !request.motion_bound_px_per_second.is_finite()
            || request.motion_bound_px_per_second < 0.0
            || scene
                .interocular_distance_mm
                .is_some_and(|s| !s.valid() || s.minimum <= 0.0)
        {
            return Err(JointConicUnavailable::InvalidRequest);
        }
        let skew = if let [Some(a), Some(b)] = request.eyes {
            if a.exposure.roi == b.exposure.roi {
                return Err(JointConicUnavailable::InvalidRequest);
            }
            let skew = a
                .exposure
                .separation_ns(b.exposure)
                .ok_or(JointConicUnavailable::IncompatibleClocks)?;
            if skew > request.maximum_source_skew_ns {
                return Err(JointConicUnavailable::ExcessiveSourceSkew);
            }
            skew
        } else {
            0
        };
        let timing_sigma = (skew.saturating_add(request.exposure_uncertainty_ns)) as f64
            * 1.0e-9
            * request.motion_bound_px_per_second;
        let mut groups: Vec<Group> = Vec::new();
        let mut present = [false; 2];
        for eye in 0..2 {
            let (Some(evidence), Some(prior)) = (request.eyes[eye], scene.eyes[eye]) else {
                continue;
            };
            if !prior.valid() || evidence.dimensions_px.contains(&0) {
                return Err(JointConicUnavailable::InvalidRequest);
            }
            let detail = evidence
                .detail_reliability
                .filter(|d| d.is_finite())
                .unwrap_or(0.25)
                .clamp(0.0, 1.0);
            let optical_sigma = 0.75_f64.hypot(4.0 * (1.0 - detail));
            let start = groups.len();
            for (index, arc) in evidence.arcs.iter().take(128).enumerate() {
                let Some(boundary) = boundary_index(arc.kind) else {
                    continue;
                };
                if arc.points_roi_px.len() < 3 {
                    continue;
                }
                if arc
                    .support_length_cap_px
                    .is_some_and(|v| !v.is_finite() || v < 0.0)
                {
                    return Err(JointConicUnavailable::InvalidRequest);
                }
                if arc.sampling_support_px.is_some_and(|support| {
                    support.len() != arc.points_roi_px.len()
                        || support.iter().any(|v| !v.is_finite() || *v < 0.0)
                        || !support.iter().sum::<f64>().is_finite()
                }) {
                    return Err(JointConicUnavailable::InvalidRequest);
                }
                if arc.outward_normals_roi.is_some_and(|normals| {
                    normals.len() != arc.points_roi_px.len()
                        || normals.iter().flatten().any(|n| !n.valid())
                }) {
                    return Err(JointConicUnavailable::InvalidRequest);
                }
                if arc.level_sets_roi.is_some_and(|levels| {
                    levels.len() != arc.points_roi_px.len()
                        || levels.iter().flatten().any(|level| !level.valid())
                }) {
                    return Err(JointConicUnavailable::InvalidRequest);
                }
                if arc
                    .localization_sigma_px
                    .is_some_and(|sigma| !sigma.is_finite() || sigma <= 0.0)
                {
                    return Err(JointConicUnavailable::InvalidRequest);
                }
                let count = arc.points_roi_px.len().min(MAX_POINTS_PER_ARC);
                let points = (0..count)
                    .map(|j| arc.points_roi_px[j * (arc.points_roi_px.len() - 1) / (count - 1)])
                    .collect::<Vec<_>>();
                let outward_normals = (0..count)
                    .map(|j| {
                        arc.outward_normals_roi.and_then(|normals| {
                            normals[j * (arc.points_roi_px.len() - 1) / (count - 1)]
                        })
                    })
                    .collect();
                let level_sets = (0..count)
                    .map(|j| {
                        arc.level_sets_roi.and_then(|levels| {
                            levels[j * (arc.points_roi_px.len() - 1) / (count - 1)]
                        })
                    })
                    .collect();
                if points
                    .iter()
                    .any(|&(x, y)| !x.is_finite() || !y.is_finite())
                {
                    continue;
                }
                let integrated = if let Some(support) = arc.sampling_support_px {
                    let retained = (0..count)
                        .map(|j| j * (arc.points_roi_px.len() - 1) / (count - 1))
                        .collect::<Vec<_>>();
                    let mut weights =
                        crate::roi_evidence::reduce_sampling_support(support, &retained)
                            .ok_or(JointConicUnavailable::InvalidRequest)?;
                    let length = weights.iter().sum::<f64>();
                    if length > 1.0e-9 {
                        for weight in &mut weights {
                            *weight /= length;
                        }
                        Some((weights, length))
                    } else {
                        None
                    }
                } else {
                    polyline_quadrature(&points)
                };
                let Some((quadrature, length_px)) = integrated else {
                    continue;
                };
                // A source-local tangent footprint may reduce the amount of
                // distinct edge support. Preserve positions and quadrature;
                // radial zigzags must not manufacture extra information mass.
                let length_px = arc
                    .support_length_cap_px
                    .map_or(length_px, |cap| length_px.min(cap));
                if length_px <= 1.0e-9 {
                    continue;
                }
                let existing = (start..groups.len())
                    .find(|&i| groups[i].alternatives[0].group == arc.evidence_group);
                if existing.is_none() && groups.len() - start >= MAX_GROUPS_PER_EYE {
                    continue;
                }
                let i = existing.unwrap_or_else(|| {
                    groups.push(Group {
                        alternatives: Vec::new(),
                        weight: 0.0,
                    });
                    groups.len() - 1
                });
                if groups[i].alternatives.len() >= MAX_ALTERNATIVES_PER_GROUP {
                    continue;
                }
                let band = arc
                    .normal_band_half_width_px
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .unwrap_or(0.0);
                let sigma = arc
                    .localization_sigma_px
                    .unwrap_or(optical_sigma)
                    .hypot(band)
                    .hypot(timing_sigma);
                let sigma = match request.scene.measured_boundary_sigma_px[boundary] {
                    Some(measured) if measured.is_finite() && measured > 0.0 => {
                        measured.hypot(timing_sigma)
                    }
                    _ => sigma,
                };
                let weight = (length_px / SUPPORT_CORRELATION_LENGTH_PX.max(8.0 * sigma)).min(16.0);
                // Corpus-only sensitivity trial: change information mass, not
                // measured positions, localization, or the hypothesis support.
                #[cfg(test)]
                let weight = if arc.kind == BoundaryKind::PupillaryBoundary
                    && std::env::var("BUTTERCUP_JOINT_PUPIL_HALF_WEIGHT").as_deref() == Ok("1")
                {
                    weight * 0.5
                } else {
                    weight
                };
                groups[i].weight = groups[i].weight.max(weight);
                groups[i].alternatives.push(SparseArc {
                    eye,
                    index,
                    kind: arc.kind,
                    boundary,
                    group: arc.evidence_group,
                    points,
                    outward_normals,
                    level_sets,
                    sigma,
                    quadrature,
                    length_px,
                    weight,
                });
                present[eye] = true;
            }
        }
        if !present.into_iter().any(|x| x) {
            return Err(JointConicUnavailable::NoBoundaryEvidence);
        }
        if groups.iter().any(|group| {
            group
                .alternatives
                .iter()
                .any(|a| a.level_sets.iter().flatten().any(|p| p.varies()))
                && group.alternatives.iter().any(|a| {
                    a.eye != group.alternatives[0].eye
                        || a.boundary != group.alternatives[0].boundary
                })
        }) {
            return Err(JointConicUnavailable::InvalidRequest);
        }
        let target_chart = ViewpointRayChart::new(scene.target_reference_camera_mm)
            .ok_or(JointConicUnavailable::InvalidRequest)?;
        let mut p = Self {
            request,
            target_chart,
            groups,
            present,
            initial: [0.0; PARAMETERS],
            lower: [0.0; PARAMETERS],
            upper: [0.0; PARAMETERS],
            scales: [1.0; PARAMETERS],
            marginalize_arc_alternatives: false,
            exact_conic_distances: false,
        };
        for i in 0..2 {
            p.lower[i] = -scene.maximum_gaze_slope;
            p.upper[i] = scene.maximum_gaze_slope;
            p.scales[i] = 0.1;
        }
        p.initial[2] = scene.fixation_axial_distance_mm.nominal.ln();
        p.lower[2] = scene.fixation_axial_distance_mm.minimum.ln();
        p.upper[2] = scene.fixation_axial_distance_mm.maximum.ln();
        p.scales[2] = 0.25;
        for eye in 0..2 {
            if !present[eye] {
                continue;
            }
            let prior = scene.eyes[eye].unwrap();
            let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
            for axis in 0..3 {
                p.initial[k + axis] = prior.limbus_center.camera_mm[axis];
                let maximum = prior.limbus_center.maximum_displacement_mm;
                let width = if axis < 2
                    && matches!(
                        prior.limbus_center.transverse_frame,
                        TransversePositionFrame::AtNominalDepth
                    ) {
                    maximum[axis]
                        + (p.initial[k + axis].abs() + maximum[axis]) * maximum[2]
                            / prior.limbus_center.camera_mm[2].abs()
                } else {
                    maximum[axis]
                };
                p.lower[k + axis] = p.initial[k + axis] - width;
                p.upper[k + axis] = p.initial[k + axis] + width;
                p.scales[k + axis] = prior.limbus_center.sigma_mm[axis].min(20.0);
            }
            for boundary in 0..3 {
                let r = prior.radii_mm[boundary];
                p.initial[k + 3 + boundary] = r.nominal;
                p.lower[k + 3 + boundary] = r.minimum;
                p.upper[k + 3 + boundary] = r.maximum;
                p.scales[k + 3 + boundary] = r.sigma;
            }
            for axis in 0..2 {
                p.lower[k + 6 + axis] = -prior.pupil_maximum_decentration_mm;
                p.upper[k + 6 + axis] = prior.pupil_maximum_decentration_mm;
                p.scales[k + 6 + axis] = prior.pupil_decentration_sigma_mm;
            }
            let d = prior.pupil_inward_depth_mm;
            p.initial[k + 8] = d.nominal;
            p.lower[k + 8] = d.minimum;
            p.upper[k + 8] = d.maximum;
            p.scales[k + 8] = d.sigma;
            if let Some(alignment) = prior.surface_axis_alignment {
                for axis in 0..2 {
                    p.initial[k + 9 + axis] = alignment.nominal_radians[axis];
                    p.lower[k + 9 + axis] =
                        alignment.nominal_radians[axis] - alignment.maximum_deviation_radians[axis];
                    p.upper[k + 9 + axis] =
                        alignment.nominal_radians[axis] + alignment.maximum_deviation_radians[axis];
                    p.scales[k + 9 + axis] = alignment.sigma_radians[axis];
                }
            }
            // An upstream conic may INITIALIZE a nuisance radius inside the
            // already-frozen scene bounds. It is not an extra residual or a
            // scale measurement, and never changes a bound or SN-FEIDA scale.
            if let Some(evidence) = request.eyes[eye] {
                for boundary in 0..3 {
                    if let Some(c) = evidence
                        .conics
                        .iter()
                        .find(|c| boundary_index(c.kind) == Some(boundary))
                    {
                        let initial_radius = c.ellipse_roi_px.major_radius * (-p.initial[k + 2])
                            / (scene.camera.focal_px[0] * scene.camera.focal_px[1]).sqrt();
                        if initial_radius.is_finite() {
                            p.initial[k + 3 + boundary] = initial_radius
                                .clamp(p.lower[k + 3 + boundary], p.upper[k + 3 + boundary]);
                        }
                    }
                }
            }
            // Initialize a feasible nested family even when only an outer
            // conic was observed. An unobserved nominal inner radius must not
            // make every hypothesis infeasible before any iteration runs.
            p.initial[k + 4] = p.initial[k + 4]
                .min(p.initial[k + 3] * 0.98)
                .max(p.lower[k + 4]);
            p.initial[k + 5] = p.initial[k + 5]
                .min(p.initial[k + 4] * 0.90)
                .max(p.lower[k + 5]);
        }
        let nominal_geometry = p.initial;
        for eye in 0..2 {
            if present[eye] {
                let hint = request.eyes[eye].and_then(|e| {
                    e.conics
                        .iter()
                        .find(|c| c.kind == BoundaryKind::OuterLimbus)
                        .and_then(|c| {
                            circle_pose_hypotheses(
                                scene.camera,
                                c.ellipse_roi_px,
                                e.sensor_origin_px,
                            )
                        })
                });
                if let Some(poses) = hint {
                    if let Some(initial) = p.seed_circle_geometry(p.initial, eye, poses[0]) {
                        p.initial = initial;
                    }
                }
            }
        }
        // Two individually plausible conic-scale starts can violate a joint
        // IPD bound. Move only the INITIAL nuisance parameters back toward the
        // frozen nominal scene until their pair is feasible. Otherwise every
        // target/sign start fails before the joint objective sees any pixels.
        if !p.interocular_feasible(&p.initial) {
            let conic_geometry = p.initial;
            for fraction in [0.5, 0.25, 0.125, 0.0] {
                for i in TARGET_PARAMETERS..PARAMETERS {
                    p.initial[i] =
                        nominal_geometry[i] + fraction * (conic_geometry[i] - nominal_geometry[i]);
                }
                if p.interocular_feasible(&p.initial) {
                    break;
                }
            }
        }
        Ok(p)
    }

    fn interocular_feasible(&self, p: &Parameters) -> bool {
        self.request
            .scene
            .interocular_distance_mm
            .filter(|_| self.present == [true, true])
            .is_none_or(|ipd| {
                let other = TARGET_PARAMETERS + EYE_PARAMETERS;
                let distance = norm3(sub3(
                    [p[3], p[4], p[5]],
                    [p[other], p[other + 1], p[other + 2]],
                ));
                distance >= ipd.minimum && distance <= ipd.maximum
            })
    }

    fn seed_circle_geometry(
        &self,
        mut p: Parameters,
        eye: usize,
        pose: CirclePoseSeed,
    ) -> Option<Parameters> {
        let prior = self.request.scene.eyes[eye]?;
        let position = prior.limbus_center;
        let radius = prior.radii_mm[0];
        let mut lower = radius.minimum.max(prior.radii_mm[1].minimum);
        let mut upper = radius.maximum;
        let mut numerator = radius.nominal / radius.sigma.powi(2);
        let mut denominator = radius.sigma.powi(2).recip();
        for axis in 0..3 {
            if axis < 2
                && matches!(
                    position.transverse_frame,
                    TransversePositionFrame::AtNominalDepth
                )
            {
                continue;
            }
            let slope = pose.center_per_radius[axis];
            if slope.abs() > 1.0e-12 {
                let endpoints = [
                    (position.camera_mm[axis] - position.maximum_displacement_mm[axis]) / slope,
                    (position.camera_mm[axis] + position.maximum_displacement_mm[axis]) / slope,
                ];
                lower = lower.max(endpoints[0].min(endpoints[1]));
                upper = upper.min(endpoints[0].max(endpoints[1]));
            }
            numerator += slope * position.camera_mm[axis] / position.sigma_mm[axis].powi(2);
            denominator += slope * slope / position.sigma_mm[axis].powi(2);
        }
        if lower > upper {
            return None;
        }
        let r = (numerator / denominator).clamp(lower, upper);
        let center = scale3(pose.center_per_radius, r);
        if position
            .displacement(center)
            .into_iter()
            .zip(position.maximum_displacement_mm)
            .any(|(d, b)| d.abs() > b + 1.0e-9)
        {
            return None;
        }
        let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
        let depth_ratio = center[2] / p[k + 2];
        p[k..k + 3].copy_from_slice(&center);
        p[k + 3] = r;
        // Keep inner/pupil hint starts on this depth scale, while respecting
        // the same fixed bounds and nesting. No prior or diagnostic is changed.
        for boundary in 1..3 {
            let observed = self.request.eyes[eye]?
                .conics
                .iter()
                .any(|c| boundary_index(c.kind) == Some(boundary));
            if observed {
                p[k + 3 + boundary] = (p[k + 3 + boundary] * depth_ratio)
                    .clamp(self.lower[k + 3 + boundary], self.upper[k + 3 + boundary]);
            }
        }
        p[k + 4] = p[k + 4].min(p[k + 3]).max(self.lower[k + 4]);
        p[k + 5] = p[k + 5].min(p[k + 4] - 1.0e-6).max(self.lower[k + 5]);
        Some(p)
    }

    fn target(&self, p: &Parameters) -> Option<[f64; 3]> {
        self.target_chart.target([p[0], p[1], p[2]])
    }

    fn project_step(&self, mut p: Parameters) -> Option<Parameters> {
        for i in 0..PARAMETERS {
            p[i] = p[i].clamp(self.lower[i], self.upper[i]);
        }
        for eye in 0..2 {
            if self.present[eye] {
                let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS + 3;
                let nested = project_nested_radii(
                    std::array::from_fn(|i| p[k + i]),
                    std::array::from_fn(|i| self.lower[k + i]),
                    std::array::from_fn(|i| self.upper[k + i]),
                    std::array::from_fn(|i| self.scales[k + i]),
                )?;
                p[k..k + 3].copy_from_slice(&nested);
            }
        }
        Some(p)
    }

    fn geometry(
        &self,
        p: &Parameters,
        eye: usize,
    ) -> Option<([f64; 3], [f64; 3], [ProjectedCircle; 3])> {
        let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
        let c = [p[k], p[k + 1], p[k + 2]];
        let position = self.request.scene.eyes[eye]?.limbus_center;
        if position
            .displacement(c)
            .into_iter()
            .zip(position.maximum_displacement_mm)
            .any(|(d, bound)| d.abs() > bound + 1.0e-9)
        {
            return None;
        }
        let gaze = normalized3(sub3(self.target(p)?, c))?;
        if gaze[2] <= 0.0 {
            return None;
        }
        let tangent_u = normalized3(cross3([0.0, 1.0, 0.0], gaze))?;
        let tangent_v = cross3(gaze, tangent_u);
        let n = normalized3(add3(
            gaze,
            add3(
                scale3(tangent_u, p[k + 9].tan()),
                scale3(tangent_v, p[k + 10].tan()),
            ),
        ))?;
        if n[2] <= 0.0
            || dot3(n, scale3(c, -1.0)) <= 0.0
            || p[k + 4] > p[k + 3]
            || p[k + 5] >= p[k + 4]
        {
            return None;
        }
        let u = normalized3(cross3([0.0, 1.0, 0.0], n))?;
        let v = cross3(n, u);
        let pupil = add3(
            add3(c, scale3(n, -p[k + 8])),
            add3(scale3(u, p[k + 6]), scale3(v, p[k + 7])),
        );
        let camera = self.request.scene.camera;
        let origin = self.request.eyes[eye]?.sensor_origin_px;
        let conics = [
            ProjectedCircle::project(camera, c, n, p[k + 3], origin)?
                .with_exact_distance(self.exact_conic_distances)?,
            ProjectedCircle::project(camera, c, n, p[k + 4], origin)?
                .with_exact_distance(self.exact_conic_distances)?,
            ProjectedCircle::project(camera, pupil, n, p[k + 5], origin)?
                .with_exact_distance(self.exact_conic_distances)?,
        ];
        let outer = conics[0].ellipse()?;
        if !super::PROVISIONAL_CENTRAL_CAMERA_LIMBUS_ENVELOPE
            .admits_axes(outer.major_radius, outer.minor_radius)
        {
            return None;
        }
        Some((c, n, conics))
    }

    fn conics(&self, p: &Parameters) -> Option<[[Option<ProjectedCircle>; 3]; 2]> {
        let mut conics = [[None; 3]; 2];
        for eye in 0..2 {
            if self.present[eye] {
                conics[eye] = self.geometry(p, eye)?.2.map(Some);
            }
        }
        if !self.interocular_feasible(p) {
            return None;
        }
        Some(conics)
    }

    fn select(&self, conics: &[[Option<ProjectedCircle>; 3]; 2]) -> Selection<'_> {
        let mut cached_arcs = Vec::with_capacity(self.groups.len());
        let choices = self
            .groups
            .iter()
            .map(|g| {
                g.alternatives
                    .iter()
                    .enumerate()
                    .map(|(i, a)| {
                        let c = conics[a.eye][a.boundary].unwrap();
                        let values = a.evaluated_residuals(c);
                        let mean = values.mean_cost;
                        let cost = a.weight * mean.min(MAXIMUM_GROUP_COST)
                            + (g.weight - a.weight) * MAXIMUM_GROUP_COST;
                        (i, cost, mean, values, c)
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.total_cmp(&b.2)))
                    .map(|(i, _, _, values, conic)| {
                        cached_arcs.push(CachedArcResiduals {
                            arc: &g.alternatives[i],
                            conic_bits: conic.0.map(f64::to_bits),
                            exact_distance: conic.1.is_some(),
                            values,
                        });
                        i
                    })
                    .unwrap()
            })
            .collect();
        let mut selected = mask_levels::select(self, conics, choices);
        selected.cached_arcs = cached_arcs;
        selected
    }

    fn residuals(&self, p: &Parameters, selected: &Selection) -> Option<Vec<f64>> {
        self.residuals_with_rejection(p, selected, None)
    }

    fn rejected_groups(
        &self,
        conics: &[[Option<ProjectedCircle>; 3]; 2],
        selected: &Selection,
    ) -> Vec<bool> {
        self.groups
            .iter()
            .zip(selected)
            .enumerate()
            .map(|(index, (group, &choice))| {
                let arc = &group.alternatives[choice];
                let conic = conics[arc.eye][arc.boundary].unwrap();
                selected
                    .cached_arcs
                    .get(index)
                    .and_then(|cache| cache.at(arc, conic, selected.levels[index]))
                    .map_or_else(
                        || arc.mean_cost_at_level(conic, selected.levels[index]),
                        |values| values.mean_cost,
                    )
                    >= MAXIMUM_GROUP_COST
            })
            .collect()
    }

    /// Freeze the current arc activity ONLY for a local numerical derivative.
    /// Switching between capped position-only residuals and uncapped mixed
    /// position/direction residuals rotates the residual vector discontinuously,
    /// even when their squared costs are almost equal. Differencing across that
    /// switch manufactures a force for a rejected arc. Actual trial objectives
    /// pass None and reconsider both alternatives and rejection every time.
    fn residuals_with_rejection(
        &self,
        p: &Parameters,
        selected: &Selection,
        fixed_rejection: Option<&[bool]>,
    ) -> Option<Vec<f64>> {
        let conics = self.conics(p)?;
        let refreshed;
        let selected = if fixed_rejection.is_none() && selected.has_marginalization() {
            // Fresh objectives marginalize at the trial geometry. Only a
            // local EM derivative freezes the previous responsibilities.
            refreshed = self.select(&conics);
            &refreshed
        } else {
            selected
        };
        let mut r = Vec::with_capacity(self.groups.len() * MAX_POINTS_PER_ARC * 2 + PARAMETERS + 8);
        for (group_index, (group, &choice)) in self.groups.iter().zip(selected).enumerate() {
            if selected.in_family[group_index] || selected.has_group_mixture(group_index) {
                continue;
            }
            let arc = &group.alternatives[choice];
            let conic = conics[arc.eye][arc.boundary]?;
            let cached = selected
                .cached_arcs
                .get(group_index)
                .and_then(|cache| cache.at(arc, conic, 1));
            let evaluated = (cached.is_none() && !fixed_rejection.is_some_and(|r| r[group_index]))
                .then(|| arc.evaluated_residuals(conic));
            let values = cached.or(evaluated.as_ref());
            let outlier = fixed_rejection.map_or_else(
                || values.unwrap().mean_cost >= MAXIMUM_GROUP_COST,
                |rejected| rejected[group_index],
            );
            for (index, &w) in arc.quadrature.iter().enumerate() {
                // Position and measured direction describe ONE correlated
                // contour, sharing its geometric mass and full outlier cap.
                // A rejected group has constant cost and zero pulling force.
                r.push(if outlier {
                    (MAXIMUM_GROUP_COST * arc.weight * w).sqrt()
                } else {
                    values.unwrap().position[index] * (arc.weight * w).sqrt()
                });
                if arc.outward_normals[index].is_some() {
                    r.push(if outlier {
                        0.0
                    } else {
                        values.unwrap().direction[index] * (arc.weight * w).sqrt()
                    });
                }
            }
            // A shorter correlated alternative cannot inherit the longer
            // one's weight, or win merely by omitting most of its coverage.
            r.push(((group.weight - arc.weight) * MAXIMUM_GROUP_COST).sqrt());
        }
        mask_levels::append_residuals(self, &conics, selected, &mut r);
        let scene = self.request.scene;
        // Scale/position priors are separate from arc residuals. A radius does
        // not calibrate its own SN-FEIDA normalization.
        r.push(
            (p[2].exp() - scene.fixation_axial_distance_mm.nominal)
                / scene.fixation_axial_distance_mm.sigma,
        );
        for eye in 0..2 {
            if self.present[eye] {
                let prior = scene.eyes[eye]?;
                let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
                let displacement = prior.limbus_center.displacement([p[k], p[k + 1], p[k + 2]]);
                for axis in 0..3 {
                    r.push(displacement[axis] / prior.limbus_center.sigma_mm[axis]);
                }
                for boundary in 0..3 {
                    r.push(
                        (p[k + 3 + boundary] - prior.radii_mm[boundary].nominal)
                            / prior.radii_mm[boundary].sigma,
                    );
                }
                r.push(p[k + 6] / prior.pupil_decentration_sigma_mm);
                r.push(p[k + 7] / prior.pupil_decentration_sigma_mm);
                r.push(
                    (p[k + 8] - prior.pupil_inward_depth_mm.nominal)
                        / prior.pupil_inward_depth_mm.sigma,
                );
                if let Some(alignment) = prior.surface_axis_alignment {
                    for axis in 0..2 {
                        r.push(
                            (p[k + 9 + axis] - alignment.nominal_radians[axis])
                                / alignment.sigma_radians[axis],
                        );
                    }
                }
                if let Some(pivot) = prior.effective_pivot {
                    let (center, normal, _) = self.geometry(p, eye)?;
                    let current = sub3(center, scale3(normal, prior.limbus_to_pivot_mm));
                    let displacement = pivot.displacement(current);
                    for axis in 0..3 {
                        if displacement[axis].abs() > pivot.maximum_displacement_mm[axis] {
                            return None;
                        }
                        r.push(displacement[axis] / pivot.sigma_mm[axis]);
                    }
                }
            }
        }
        if let Some(ipd) = scene
            .interocular_distance_mm
            .filter(|_| self.present == [true, true])
        {
            let other = TARGET_PARAMETERS + EYE_PARAMETERS;
            let distance = norm3(sub3(
                [p[3], p[4], p[5]],
                [p[other], p[other + 1], p[other + 2]],
            ));
            r.push((distance - ipd.nominal) / ipd.sigma);
        }
        r.iter().all(|v| v.is_finite()).then_some(r)
    }

    fn seeds(&self) -> Vec<Parameters> {
        let limit = self.request.maximum_hypotheses.min(MAX_HYPOTHESES);
        let mut seeds = Vec::new();
        let mut push = |mut p: Parameters, target: [f64; 3]| {
            if seeds.len() >= limit {
                return;
            }
            let Some(coordinates) = self.target_chart.coordinates(target) else {
                return;
            };
            p[..TARGET_PARAMETERS].copy_from_slice(&coordinates);
            for i in 0..PARAMETERS {
                p[i] = p[i].clamp(self.lower[i], self.upper[i]);
            }
            if !seeds.iter().any(|s: &Parameters| {
                (s[0] - p[0]).hypot(s[1] - p[1]) < 0.002
                    && (s[2] - p[2]).abs() < 0.02
                    && (TARGET_PARAMETERS..PARAMETERS)
                        .all(|i| (s[i] - p[i]).abs() < 0.01 * self.scales[i])
            }) {
                seeds.push(p);
            }
        };
        if let Some(target) = self
            .request
            .scene
            .target_seed_camera_mm
            .filter(|p| p.into_iter().all(|v| v.is_finite()))
        {
            push(self.initial, target);
        }
        if let Some(target) = self
            .request
            .scene
            .secondary_target_seed_camera_mm
            .filter(|p| p.into_iter().all(|v| v.is_finite()))
        {
            push(self.initial, target);
        }
        // Conic signs supply MULTIPLE starts for the SAME coupled objective.
        // No independent gaze is accepted or averaged here.
        for multiplier in [1.0, 0.5, 2.0] {
            // Interleave ROIs before considering their second/further hints.
            // Four low-confidence candidates in eye zero must not consume the
            // whole paired budget before eye one's first supported conic.
            for rank in 0..4 {
                for eye in 0..2 {
                    if !self.present[eye] {
                        continue;
                    }
                    let evidence = self.request.eyes[eye].unwrap();
                    let Some(conic) = evidence.conics.get(rank) else {
                        continue;
                    };
                    let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
                    let center = [self.initial[k], self.initial[k + 1], self.initial[k + 2]];
                    let toward = normalized3(scale3(center, -1.0)).unwrap();
                    let u = normalized3(cross3([0.0, 1.0, 0.0], toward)).unwrap();
                    let v = cross3(toward, u);
                    let e = conic.ellipse_roi_px;
                    if !e.major_radius.is_finite()
                        || !e.minor_radius.is_finite()
                        || !e.angle.is_finite()
                        || e.minor_radius <= 0.0
                        || e.minor_radius > e.major_radius
                    {
                        continue;
                    }
                    let cosine = (e.minor_radius / e.major_radius).clamp(0.1, 1.0);
                    let sine = (1.0 - cosine * cosine).sqrt();
                    let approximate = [-1.0, 1.0].map(|sign| {
                        add3(
                            scale3(toward, cosine),
                            add3(
                                scale3(u, -e.angle.sin() * sine * sign),
                                scale3(v, e.angle.cos() * sine * sign),
                            ),
                        )
                    });
                    let poses = circle_pose_hypotheses(
                        self.request.scene.camera,
                        e,
                        evidence.sensor_origin_px,
                    );
                    let normals = poses.map(|p| p.map(|p| p.normal)).unwrap_or(approximate);
                    for (branch, n) in normals.into_iter().enumerate() {
                        if n[2] <= 0.05 {
                            continue;
                        }
                        let mut geometry = if conic.kind == BoundaryKind::OuterLimbus {
                            poses
                                .and_then(|poses| {
                                    self.seed_circle_geometry(self.initial, eye, poses[branch])
                                })
                                .unwrap_or(self.initial)
                        } else {
                            self.initial
                        };
                        // Alternative conics must bring their own nuisance
                        // center/range, not just rotate a target around the
                        // first candidate's geometry. Keep every hard bound.
                        if !self.interocular_feasible(&geometry) {
                            let proposed = geometry;
                            for fraction in [0.5, 0.25, 0.125, 0.0] {
                                for i in TARGET_PARAMETERS..PARAMETERS {
                                    geometry[i] = self.initial[i]
                                        + fraction * (proposed[i] - self.initial[i]);
                                }
                                if self.interocular_feasible(&geometry) {
                                    break;
                                }
                            }
                        }
                        let center = [geometry[k], geometry[k + 1], geometry[k + 2]];
                        let axial = self.request.scene.fixation_axial_distance_mm;
                        let distance =
                            (axial.nominal * multiplier).clamp(axial.minimum, axial.maximum);
                        let center_offset = dot3(
                            sub3(center, self.target_chart.origin_camera_mm),
                            self.target_chart.toward_camera,
                        );
                        let ray_forward = dot3(n, self.target_chart.toward_camera);
                        if ray_forward <= 1.0e-6 || distance <= center_offset {
                            continue;
                        }
                        push(
                            geometry,
                            add3(center, scale3(n, (distance - center_offset) / ray_forward)),
                        );
                    }
                }
            }
        }
        if let Some(target) = self.target(&self.initial) {
            push(self.initial, target);
        }
        // Unfitted partial arcs are allowed. Without a complete ellipse hint,
        // non-frontal starts avoid the zero tilt derivative at a perfect circle.
        for (x, y) in [
            (0.0, -0.35),
            (0.0, 0.35),
            (-0.35, 0.0),
            (0.35, 0.0),
            (-0.35, -0.35),
            (-0.35, 0.35),
            (0.35, -0.35),
            (0.35, 0.35),
        ] {
            let mut candidate = self.initial;
            candidate[0] = x;
            candidate[1] = y;
            if let Some(target) = self.target(&candidate) {
                push(self.initial, target);
            }
        }
        seeds
    }

    fn refine(&self, mut p: Parameters) -> Option<(Parameters, f64, usize)> {
        let mut damping = 0.03;
        let mut steps = 0;
        for _ in 0..self.request.maximum_refinements.min(MAX_REFINEMENTS) {
            let conics = self.conics(&p)?;
            let selected = self.select(&conics);
            let rejected = self.rejected_groups(&conics, &selected);
            let residuals = self.residuals(&p, &selected)?;
            let cost = squared_norm(&residuals);
            let mut derivatives: Vec<Vec<f64>> = Vec::with_capacity(PARAMETERS);
            for parameter in 0..PARAMETERS {
                if self.lower[parameter] == self.upper[parameter] {
                    derivatives.push(vec![0.0; residuals.len()]);
                    continue;
                }
                let step = 1.0e-4 * self.scales[parameter];
                let mut q = p;
                q[parameter] += step;
                // A one-sided derivative at a hard boundary must point inward.
                let delta = if q[parameter] > self.upper[parameter] {
                    -step
                } else {
                    step
                };
                q[parameter] = p[parameter] + delta;
                let d = self
                    .residuals_with_rejection(&q, &selected, Some(&rejected))
                    .map(|rr| {
                        rr.iter()
                            .zip(&residuals)
                            .map(|(a, b)| (a - b) / delta * self.scales[parameter])
                            .collect()
                    })
                    .unwrap_or_else(|| vec![0.0; residuals.len()]);
                derivatives.push(d);
            }
            let mut matrix = [[0.0; PARAMETERS]; PARAMETERS];
            let mut rhs = [0.0; PARAMETERS];
            for i in 0..PARAMETERS {
                rhs[i] = -derivatives[i]
                    .iter()
                    .zip(&residuals)
                    .map(|(a, b)| a * b)
                    .sum::<f64>();
                for j in 0..=i {
                    let value = derivatives[i]
                        .iter()
                        .zip(&derivatives[j])
                        .map(|(a, b)| a * b)
                        .sum();
                    matrix[i][j] = value;
                    matrix[j][i] = value;
                }
            }
            let mut accepted = false;
            for _ in 0..4 {
                let mut regularized = matrix;
                for i in 0..PARAMETERS {
                    regularized[i][i] += damping * (matrix[i][i] + 1.0);
                }
                let Some(delta) = solve_dense(regularized, rhs) else {
                    damping *= 10.0;
                    continue;
                };
                let mut q = p;
                for i in 0..PARAMETERS {
                    q[i] = p[i] + delta[i] * self.scales[i];
                }
                // Project onto the bounded nested-radius set. Rejecting every
                // coupled step when an unobserved inner radius crosses the
                // observed outer radius can otherwise freeze BOTH eyes.
                let Some(q) = self.project_step(q) else {
                    damping *= 5.0;
                    continue;
                };
                let candidate_cost = self
                    .conics(&q)
                    .and_then(|c| self.residuals(&q, &self.select(&c)))
                    .map(|r| squared_norm(&r))
                    .unwrap_or(f64::INFINITY);
                steps += 1;
                if candidate_cost < cost {
                    p = q;
                    damping = (damping * 0.3).max(1.0e-6);
                    accepted = true;
                    break;
                }
                damping *= 5.0;
            }
            if !accepted {
                break;
            }
        }
        let selected = self.select(&self.conics(&p)?);
        let cost = squared_norm(&self.residuals(&p, &selected)?);
        Some((p, cost, steps))
    }

    fn solution(&self, p: &Parameters, cost: f64) -> Option<JointConicSolution> {
        let conics = self.conics(p)?;
        let mut result = JointConicSolution {
            #[cfg(test)]
            factor_costs: Some(self.factor_cost_diagnostic(p)?),
            local_uncertainty: None,
            posterior: None,
            mask_level_families: Vec::new(),
            arc_alternative_marginals: Vec::new(),
            target_camera_mm: self.target(p)?,
            target_reference_camera_mm: self.target_chart.origin_camera_mm,
            target_viewpoint_axial_distance_mm: p[2].exp(),
            target_viewpoint_slopes: [p[0], p[1]],
            target_viewpoint_slope_limit: self.request.scene.maximum_gaze_slope,
            target_viewpoint_bounds_active: [0, 1]
                .map(|i| (p[i].abs() - self.request.scene.maximum_gaze_slope).abs() <= 1.0e-5),
            eye_centers_camera_mm: [None; 2],
            eye_normals: [None; 2],
            effective_pivots_camera_mm: [None; 2],
            eye_gaze_directions: [None; 2],
            surface_axis_alignment_radians: [None; 2],
            ellipses_roi_px: [[None; 3]; 2],
            arcs: Vec::new(),
            contributing_eyes: [false; 2],
            modeled_eyes: self.present,
            unlocalized_eye_cost: [0.0; 2],
            robust_cost: cost,
            alternative_cost_margin: None,
            alternative_target_camera_mm: None,
            hypotheses_evaluated: 0,
            hypotheses_by_association: [0; 3],
            refinement_steps: 0,
        };
        let selection = self.select(&conics);
        result.mask_level_families = mask_levels::diagnostics(self, &selection);
        result.arc_alternative_marginals = mask_levels::arc_diagnostics(self, &selection);
        for (index, g) in self.groups.iter().enumerate() {
            let a = &g.alternatives[selection.choices[index]];
            let level = selection.levels[index];
            let c = conics[a.eye][a.boundary]?;
            let rms = (a
                .points
                .iter()
                .enumerate()
                .map(|(i, _)| c.residual_px(a.point_at_level(i, level)).powi(2))
                .sum::<f64>()
                / a.points.len() as f64)
                .sqrt();
            let group_cost = a.mean_cost_at_level(c, level);
            let used = group_cost < MAXIMUM_GROUP_COST;
            let normal_errors = (0..a.points.len())
                .filter_map(|i| a.normal_angle_error_at_level(c, i, level))
                .collect::<Vec<_>>();
            result.contributing_eyes[a.eye] |= used;
            result.arcs.push(ArcSupport {
                exposure: self.request.eyes[a.eye]?.exposure,
                evidence_group: a.group,
                arc_index: a.index,
                points_roi_px: (0..a.points.len())
                    .map(|i| a.point_at_level(i, level))
                    .collect(),
                kind: a.kind,
                rms_px: rms,
                sigma_px: a.sigma,
                support_length_px: a.length_px,
                evidence_weight: a.weight,
                used,
                mask_level: (selection.in_family[index] && level < 3).then_some(level as i8 - 1),
                boundary_normal_samples: normal_errors.len(),
                boundary_normal_rms_radians: (!normal_errors.is_empty()).then(|| {
                    (normal_errors.iter().map(|v| v * v).sum::<f64>() / normal_errors.len() as f64)
                        .sqrt()
                }),
            });
        }
        for eye in 0..2 {
            if self.present[eye] {
                let (center, normal, _) = self.geometry(p, eye)?;
                result.eye_centers_camera_mm[eye] = Some(center);
                result.eye_normals[eye] = Some(normal);
                result.eye_gaze_directions[eye] =
                    normalized3(sub3(result.target_camera_mm, center));
                let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
                result.surface_axis_alignment_radians[eye] = Some([p[k + 9], p[k + 10]]);
                result.effective_pivots_camera_mm[eye] = Some(sub3(
                    center,
                    scale3(normal, self.request.scene.eyes[eye]?.limbus_to_pivot_mm),
                ));
                result.ellipses_roi_px[eye] =
                    conics[eye].map(|c| c.and_then(ProjectedCircle::ellipse));
            }
        }
        result
            .contributing_eyes
            .into_iter()
            .any(|x| x)
            .then_some(result)
    }
}

/// Exact weighted least-squares projection onto outer >= inner > pupil within
/// frozen individual bounds. Three variables have only four contiguous active
/// block partitions; enumerate those rather than opening any anatomical bound.
fn project_nested_radii(
    values: [f64; 3],
    lower: [f64; 3],
    upper: [f64; 3],
    scales: [f64; 3],
) -> Option<[f64; 3]> {
    let offset = [0.0, 0.0, 1.0e-6];
    let values = std::array::from_fn::<_, 3, _>(|i| values[i] + offset[i]);
    let lower = std::array::from_fn::<_, 3, _>(|i| lower[i] + offset[i]);
    let upper = std::array::from_fn::<_, 3, _>(|i| upper[i] + offset[i]);
    let weights = scales.map(|s| s.powi(-2));
    let mut best = None;
    for cuts in 0..4 {
        let mut q = [0.0; 3];
        let mut start = 0;
        let mut feasible = true;
        for end in 0..3 {
            if end == 2 || cuts & (1 << end) != 0 {
                let lo = lower[start..=end]
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let hi = upper[start..=end]
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                if lo > hi {
                    feasible = false;
                    break;
                }
                let mean = (start..=end).map(|i| weights[i] * values[i]).sum::<f64>()
                    / weights[start..=end].iter().sum::<f64>();
                q[start..=end].fill(mean.clamp(lo, hi));
                start = end + 1;
            }
        }
        if !feasible || q[0] < q[1] || q[1] < q[2] {
            continue;
        }
        let cost = (0..3)
            .map(|i| weights[i] * (q[i] - values[i]).powi(2))
            .sum::<f64>();
        if best.is_none_or(|(_, old)| cost < old) {
            best = Some((q, cost));
        }
    }
    best.map(|(q, _)| std::array::from_fn(|i| q[i] - offset[i]))
}

fn robust_residual(value: f64) -> f64 {
    if value.abs() <= HUBER_TRANSITION {
        value
    } else {
        value.signum() * (2.0 * HUBER_TRANSITION * value.abs() - HUBER_TRANSITION.powi(2)).sqrt()
    }
}

fn squared_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum()
}

fn solve_dense(mut a: [[f64; PARAMETERS]; PARAMETERS], mut b: Parameters) -> Option<Parameters> {
    for col in 0..PARAMETERS {
        let pivot =
            (col..PARAMETERS).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1.0e-14 {
            return None;
        }
        a.swap(pivot, col);
        b.swap(pivot, col);
        for row in col + 1..PARAMETERS {
            let factor = a[row][col] / a[col][col];
            for j in col + 1..PARAMETERS {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; PARAMETERS];
    for i in (0..PARAMETERS).rev() {
        x[i] = (b[i] - (i + 1..PARAMETERS).map(|j| a[i][j] * x[j]).sum::<f64>()) / a[i][i];
    }
    x.into_iter().all(f64::is_finite).then_some(x)
}

pub(crate) fn solve_joint_conics(
    request: JointConicRequest<'_>,
) -> Result<JointConicSolution, JointConicUnavailable> {
    solve_joint_conic_hypotheses(request, 1).map(|mut hypotheses| hypotheses.remove(0))
}

/// Retain a bounded set of CURRENT, jointly optimized alternatives for
/// temporal diagnosis. Returning more hypotheses does not add optimizer work,
/// independent gaze averages, or evidence. The first is exactly the ordinary
/// single-frame winner. A missing alternative is not proof of observability.
pub(crate) fn solve_joint_conic_hypotheses(
    request: JointConicRequest<'_>,
    maximum_returned: usize,
) -> Result<Vec<JointConicSolution>, JointConicUnavailable> {
    solve_hypotheses(
        request,
        maximum_returned,
        posterior::IntegrationConfig {
            budget: 0,
            ..Default::default()
        },
    )
}

/// The same bounded MAP search, with source-local posterior integration.
/// Sampling does not move the selected fit or add temporal observations.
pub(crate) fn solve_joint_conic_distribution(
    request: JointConicRequest<'_>,
    maximum_returned: usize,
) -> Result<Vec<JointConicSolution>, JointConicUnavailable> {
    solve_hypotheses(
        request,
        maximum_returned,
        posterior::IntegrationConfig::live(),
    )
}

/// The operating prior belongs inside the solve: selecting a different mode
/// after integration would detach its geometry from the reported posterior.
pub(crate) fn solve_joint_conics_with_mount(
    request: JointConicRequest<'_>,
    maximum_returned: usize,
    camera_mount: super::camera_mount::CameraMount,
    probabilistic: bool,
) -> Result<Vec<JointConicSolution>, JointConicUnavailable> {
    solve_joint_conics_with_direction(
        request,
        maximum_returned,
        camera_mount,
        probabilistic,
        Default::default(),
    )
}

pub(crate) fn solve_joint_conics_with_direction(
    request: JointConicRequest<'_>,
    maximum_returned: usize,
    camera_mount: super::camera_mount::CameraMount,
    probabilistic: bool,
    direction_prior: super::camera_mount::DirectionPrior,
) -> Result<Vec<JointConicSolution>, JointConicUnavailable> {
    let mut integration = if probabilistic {
        posterior::IntegrationConfig::live()
    } else {
        posterior::IntegrationConfig {
            budget: 0,
            ..Default::default()
        }
    };
    integration.camera_mount = camera_mount;
    integration.direction_prior = direction_prior;
    solve_hypotheses(request, maximum_returned, integration)
}

#[cfg(test)]
pub(crate) fn solve_joint_conic_distribution_diagnostic(
    request: JointConicRequest<'_>,
    maximum_returned: usize,
    integration: posterior::IntegrationConfig,
) -> Result<Vec<JointConicSolution>, JointConicUnavailable> {
    solve_hypotheses(request, maximum_returned, integration)
}

/// Offline attribution only. Validate the complete source request before
/// computing the existing capped cost of omitting an ROI. This cost is not a
/// normalized model evidence or a probability for that association.
#[cfg(test)]
pub(crate) fn association_omission_cost_diagnostic(
    request: JointConicRequest<'_>,
    modeled: [bool; 2],
) -> Result<[f64; 2], JointConicUnavailable> {
    Ok(Problem::new(request)?.unlocalized_costs(modeled))
}

fn solve_hypotheses(
    request: JointConicRequest<'_>,
    maximum_returned: usize,
    integration: posterior::IntegrationConfig,
) -> Result<Vec<JointConicSolution>, JointConicUnavailable> {
    if maximum_returned == 0 {
        return Err(JointConicUnavailable::InvalidRequest);
    }
    // Validate the FULL request first. Omitting an unlocalized ROI must never
    // be a way around invalid source clocks, timing bounds or scene inputs.
    let problem = Problem::new(request)?
        .with_arc_marginalization(integration.marginalize_arc_alternatives)
        .with_exact_distances(integration.exact_conic_distances);
    let budget = request.maximum_hypotheses.min(MAX_HYPOTHESES);
    let reserve = if problem.present == [true, true] && budget >= 8 {
        (budget / 4).min(4)
    } else {
        0
    };
    let mut fits = Vec::<(Parameters, JointConicSolution)>::new();
    let mut hypotheses = 0;
    let mut steps = 0;
    let mut feasible = false;
    let mut associations = [0; 3];
    let mut optimize = |model: &Problem<'_>, seeds: &[Parameters], omitted: [f64; 2]| {
        let mut best = f64::INFINITY;
        // Seeds are refined independently; evaluate them in parallel and fold
        // the results in seed order so every counter and fit list is unchanged.
        let refined = crate::parallel_work::ordered_map(seeds, 1, |&seed| {
            model.refine(seed).map(|(p, cost, refinements)| {
                (p, cost, refinements, model.solution(&p, cost + omitted.iter().sum::<f64>()))
            })
        });
        for refined in refined {
            hypotheses += 1;
            associations[if model.present == [true, true] {
                0
            } else if model.present[0] {
                1
            } else {
                2
            }] += 1;
            let Some((p, _cost, refinements, solution)) = refined else {
                continue;
            };
            feasible = true;
            steps += refinements;
            let Some(mut solution) = solution else {
                continue;
            };
            solution.unlocalized_eye_cost = omitted;
            if (0..2).all(|eye| {
                solution.eye_normals[eye].is_none_or(|n| {
                    integration.camera_mount.supports(n[1])
                        && solution.eye_centers_camera_mm[eye]
                            .is_some_and(|c| integration.direction_prior.supports(eye, c, n))
                })
            }) {
                best = best.min(solution.robust_cost);
            }
            fits.push((p, solution));
        }
        best
    };
    let seeds = problem.seeds();
    let prefix = (budget - 2 * reserve).min(seeds.len());
    let mut used = prefix;
    let mut best = optimize(&problem, &seeds[..prefix], [0.0; 2]);
    if reserve > 0 {
        for modeled in [[true, false], [false, true]] {
            let omitted = problem.unlocalized_costs(modeled);
            // Every remaining residual is nonnegative. This is a LOWER bound on
            // the identical objective, not a confidence threshold: an association
            // that already costs more with perfect residuals cannot beat `best`.
            // Return its reserved starts to the joint search. In particular, good
            // stereo evidence must not lose half its search to impossible winners.
            if omitted.iter().sum::<f64>() > best {
                continue;
            }
            let subrequest = JointConicRequest {
                eyes: std::array::from_fn(|i| if modeled[i] { request.eyes[i] } else { None }),
                maximum_hypotheses: reserve,
                ..request
            };
            let model = Problem::new(subrequest)?
                .with_arc_marginalization(integration.marginalize_arc_alternatives)
                .with_exact_distances(integration.exact_conic_distances);
            let candidates = model.seeds();
            used += candidates.len();
            best = best.min(optimize(&model, &candidates, omitted));
        }
    }
    let end = (prefix + budget - used).min(seeds.len());
    if end > prefix {
        optimize(&problem, &seeds[prefix..end], [0.0; 2]);
    }
    // These are alternative ASSOCIATIONS in one robust boundary objective.
    // A one-ROI hypothesis pays the complete other ROI's capped evidence cost;
    // it does not obtain a free win by omitting pixels or average two targets.
    fits.sort_by(|a, b| a.1.robust_cost.total_cmp(&b.1.robust_cost));
    if integration.mask_state_refinement && !fits.is_empty() {
        // Retain every baseline candidate. Additional starts can improve the
        // original objective but cannot win by changing mask likelihoods or
        // dropping an eye's omission penalty. Share one extra bounded search
        // budget across the associations already explored by the base search.
        let mut order = Vec::new();
        for (_, solution) in &fits {
            if !order.contains(&solution.modeled_eyes) {
                order.push(solution.modeled_eyes);
            }
        }
        let mut remaining = budget;
        let mut additions = Vec::new();
        for (index, &modeled) in order.iter().enumerate() {
            let local_request = JointConicRequest {
                eyes: std::array::from_fn(|eye| {
                    if modeled[eye] {
                        request.eyes[eye]
                    } else {
                        None
                    }
                }),
                ..request
            };
            let model = Problem::new(local_request)?
                .with_arc_marginalization(integration.marginalize_arc_alternatives)
                .with_exact_distances(integration.exact_conic_distances);
            let mut starts: Vec<Parameters> = Vec::new();
            for (p, solution) in &fits {
                if solution.modeled_eyes == modeled
                    && !starts
                        .iter()
                        .any(|q| (q[0] - p[0]).hypot(q[1] - p[1]) <= 0.035)
                {
                    starts.push(*p);
                    if starts.len() == 2 {
                        break;
                    }
                }
            }
            let expanded = posterior::refine_mask_state_initializations(
                &model,
                &starts,
                remaining / (order.len() - index),
            );
            remaining -= expanded.attempts;
            hypotheses += expanded.attempts;
            associations[if modeled == [true, true] {
                0
            } else if modeled[0] {
                1
            } else {
                2
            }] += expanded.attempts;
            steps += expanded.steps;
            let omitted = problem.unlocalized_costs(modeled);
            for (p, cost) in expanded.fits {
                if let Some(mut solution) = model.solution(&p, cost + omitted.iter().sum::<f64>()) {
                    solution.unlocalized_eye_cost = omitted;
                    additions.push((p, solution));
                }
            }
        }
        fits.extend(additions);
        fits.sort_by(|a, b| a.1.robust_cost.total_cmp(&b.1.robust_cost));
    }
    if fits.is_empty() {
        return Err(if feasible {
            JointConicUnavailable::NoFeasibleHypothesis
        } else {
            JointConicUnavailable::NoFeasibleInitialization
        });
    }
    // Filter all explored optimized modes before the bounded retention and
    // posterior stages. A compatible mode must carry its own confidence; the
    // unconstrained winner's distribution cannot be copied onto another fit.
    fits.retain(|(_, fit)| {
        (0..2).all(|eye| {
            fit.eye_normals[eye].is_none_or(|n| {
                integration.camera_mount.supports(n[1])
                    && fit.eye_centers_camera_mm[eye]
                        .is_some_and(|c| integration.direction_prior.supports(eye, c, n))
            })
        })
    });
    if fits.is_empty() {
        return Err(JointConicUnavailable::MountingAssumptionConflict);
    }
    let mut retained = Vec::<(Parameters, JointConicSolution)>::new();
    for (slope, candidate) in &fits {
        if retained
            .iter()
            .any(|(q, _)| (q[0] - slope[0]).hypot(q[1] - slope[1]) <= 0.035)
        {
            continue;
        }
        let mut solution = candidate.clone();
        solution.hypotheses_evaluated = hypotheses;
        solution.refinement_steps = steps;
        solution.hypotheses_by_association = associations;
        let local_request = JointConicRequest {
            eyes: std::array::from_fn(|i| {
                if solution.modeled_eyes[i] {
                    request.eyes[i]
                } else {
                    None
                }
            }),
            ..request
        };
        if let Ok(local_model) = Problem::new(local_request) {
            solution.local_uncertainty = Some(
                local_model
                    .with_arc_marginalization(integration.marginalize_arc_alternatives)
                    .with_exact_distances(integration.exact_conic_distances)
                    .local_uncertainty(slope),
            );
        }
        if let Some((_, alternate)) = fits
            .iter()
            .find(|(q, _)| (q[0] - slope[0]).hypot(q[1] - slope[1]) > 0.035)
        {
            solution.alternative_cost_margin =
                Some((alternate.robust_cost - solution.robust_cost).max(0.0));
            solution.alternative_target_camera_mm = Some(alternate.target_camera_mm);
        }
        retained.push((*slope, solution));
        if retained.len()
            >= if integration.budget > 0 {
                4
            } else {
                maximum_returned.min(4)
            }
        {
            break;
        }
    }
    if integration.budget > 0 {
        let modeled = retained[0].1.modeled_eyes;
        let local_request = JointConicRequest {
            eyes: std::array::from_fn(|i| if modeled[i] { request.eyes[i] } else { None }),
            ..request
        };
        let model = Problem::new(local_request)?
            .with_arc_marginalization(integration.marginalize_arc_alternatives)
            .with_exact_distances(integration.exact_conic_distances);
        let distribution = posterior::integrate(&model, &retained, integration);
        if let Some(selection) = &distribution.supported_mode_selection {
            let index = selection.retained_index;
            assert_eq!(retained[index].1.modeled_eyes, modeled);
            if index != 0 {
                // Return the actual optimized geometry with its own arcs and
                // uncertainty. Keep the lower-cost MAP visible as a competitor.
                let map_target = retained[0].1.target_camera_mm;
                let map_cost = retained[0].1.robust_cost;
                retained.swap(0, index);
                retained[0].1.alternative_target_camera_mm = Some(map_target);
                retained[0].1.alternative_cost_margin = Some(map_cost - retained[0].1.robust_cost);
            }
        }
        retained[0].1.posterior = Some(distribution);
    }
    retained.truncate(maximum_returned.min(4));
    Ok(retained.into_iter().map(|(_, solution)| solution).collect())
}
