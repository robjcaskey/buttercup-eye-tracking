//! Gaze-to-target geometry, separate from binocular timing/vergence coordination.
//!
//! Currently owns the existing monocular ray/physical-display and affine
//! calibration math and a joint mixed-conic target solve.
//! UI admission, authority/epoch binding, held cursors, and target animation
//! stay in the viewer. Screen coordinates are fractions, not sensor pixels.

use crate::binocular_coordinator::BinocularFactors;
use crate::eye_scene_model::RelativeGazeVector;
use crate::geometry::{add3, cross3, dot3, norm3, normalized3, scale3, solve_3x3, sub3};
use crate::roi_evidence::RoiConicEvidence;
use crate::conic_solver::joint::{JointConicRequest, JointConicSolution, JointConicUnavailable, JointScenePrior};
pub(crate) mod joint_tracking;

/// Joint target solving is intentionally distinct from the migrated
/// monocular display mapping below. Do not choose a winning eye before this
/// boundary: a weak complementary arc can resolve a sharper eye's ambiguity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointGazeRequest<'a> {
    pub(crate) eyes: [Option<RoiConicEvidence<'a>>; 2],
    pub(crate) binocular: Option<BinocularFactors>,
    pub(crate) scene: &'a JointScenePrior,
    pub(crate) maximum_hypotheses: usize,
    pub(crate) maximum_refinements: usize,
    pub(crate) exposure_uncertainty_ns: u64,
    pub(crate) motion_bound_px_per_second: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JointGazeUnavailable {
    Conic(JointConicUnavailable),
}

/// Unlike a display cursor, this target has depth conditional on the request's
/// explicit metric scene support. A nominal prior is not measured anatomy;
/// callers must preserve that provenance. Camera-frame axes follow
/// RelativeGazeVector (+Z toward camera).
#[derive(Clone, Debug)]
pub(crate) struct JointGazeSolution {
    pub(crate) eye_directions: [Option<RelativeGazeVector>; 2],
    pub(crate) target_camera_frame_mm: Option<[f64; 3]>,
    /// Only a calibrated uncertainty model may populate this covariance.
    pub(crate) target_covariance_mm2: Option<[[f64; 3]; 3]>,
    pub(crate) conic_solution: JointConicSolution,
}

pub(crate) fn solve_joint_gaze_target(
    request: JointGazeRequest<'_>,
) -> Result<JointGazeSolution, JointGazeUnavailable> {
    // Coordination only grants a bounded source-time compatibility window.
    // The shared target is fitted to image evidence INSIDE the conic solve;
    // these two directions are outputs of that one solve, not its inputs.
    let maximum_source_skew_ns = request.binocular.map(|b| b.maximum_joint_skew_ns).unwrap_or(0);
    let conic_solution = crate::conic_solver::solve_joint_conics(JointConicRequest {
        eyes: request.eyes, scene: request.scene,
        maximum_hypotheses: request.maximum_hypotheses,
        maximum_refinements: request.maximum_refinements,
        maximum_source_skew_ns,
        exposure_uncertainty_ns: request.exposure_uncertainty_ns,
        motion_bound_px_per_second: request.motion_bound_px_per_second,
    }).map_err(JointGazeUnavailable::Conic)?;
    let eye_directions = std::array::from_fn(|eye| {
        if !conic_solution.contributing_eyes[eye] { return None; }
        let normal = conic_solution.eye_gaze_directions[eye]?;
        RelativeGazeVector::from_projected(normal[0],normal[1])
    });
    Ok(JointGazeSolution {
        eye_directions,
        target_camera_frame_mm: Some(conic_solution.target_camera_mm),
        // The present scene supports are defeasible, not calibrated noise.
        target_covariance_mm2: None,
        conic_solution,
    })
}

// Allow a coarse initial calibration from the central 20% target field.
// Coverage, affine conditioning, physical geometry and shared model support
// remain mandatory; these tolerances are fractions of the full screen.
pub(crate) const VIRTUAL_MOUSE_PLANE_INLIER_RESIDUAL: f64 = 0.10;
pub(crate) const VIRTUAL_MOUSE_PLANE_MAX_RMS: f64 = 0.075;
pub(crate) const VIRTUAL_MOUSE_AFFINE_INLIER_RESIDUAL: f64 = 0.09;
pub(crate) const VIRTUAL_MOUSE_AFFINE_MAX_RMS: f64 = 0.065;
pub(crate) const VIRTUAL_MOUSE_MODEL_MAX_DISAGREEMENT: f64 = 0.08;
pub(crate) const VIRTUAL_MOUSE_AFFINE_MIN_SINGULAR_GAIN: f64 = 0.05;
pub(crate) const VIRTUAL_MOUSE_AFFINE_MAX_SINGULAR_GAIN: f64 = 40.0;
pub(crate) const VIRTUAL_MOUSE_AFFINE_MAX_CONDITION: f64 = 30.0;
// Keep all fixation centers inside the display's central 20%. This reduces
// eyelid occlusion and extreme-angle SAM foreshortening during calibration;
// the fitted 3D plane/affine still maps gaze across the full display.
pub(crate) const VIRTUAL_MOUSE_CALIBRATION_INSET: f64 = 0.40;
pub(crate) const VIRTUAL_MOUSE_CALIBRATION_TARGETS: [(f64, f64); 9] = [
    (
        VIRTUAL_MOUSE_CALIBRATION_INSET,
        VIRTUAL_MOUSE_CALIBRATION_INSET,
    ),
    (
        1.0 - VIRTUAL_MOUSE_CALIBRATION_INSET,
        VIRTUAL_MOUSE_CALIBRATION_INSET,
    ),
    (0.50, 0.50),
    (
        1.0 - VIRTUAL_MOUSE_CALIBRATION_INSET,
        1.0 - VIRTUAL_MOUSE_CALIBRATION_INSET,
    ),
    (
        VIRTUAL_MOUSE_CALIBRATION_INSET,
        1.0 - VIRTUAL_MOUSE_CALIBRATION_INSET,
    ),
    (0.50, VIRTUAL_MOUSE_CALIBRATION_INSET),
    (1.0 - VIRTUAL_MOUSE_CALIBRATION_INSET, 0.50),
    (0.50, 1.0 - VIRTUAL_MOUSE_CALIBRATION_INSET),
    (VIRTUAL_MOUSE_CALIBRATION_INSET, 0.50),
];
pub(crate) const VIRTUAL_MOUSE_MIN_STABLE_TARGETS: usize = 7;
pub(crate) const NOMINAL_DISPLAY_DISTANCE_INCHES: f64 = 24.0;
pub(crate) const NOMINAL_DISPLAY_DIAGONAL_INCHES: f64 = 27.0;
pub(crate) const NOMINAL_DISPLAY_ASPECT_WIDTH: f64 = 16.0;
pub(crate) const NOMINAL_DISPLAY_ASPECT_HEIGHT: f64 = 9.0;
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GazeAffine {
    pub(crate) x: [f64; 3],
    pub(crate) y: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VirtualDisplayPlane {
    /// Display center relative to the calibrated eye, in inches. Axes follow
    /// RelativeGazeVector: camera-right, camera-down, and toward the camera.
    pub(crate) center_inches: [f64; 3],
    pub(crate) right_axis: [f64; 3],
    pub(crate) down_axis: [f64; 3],
    pub(crate) width_inches: f64,
    pub(crate) height_inches: f64,
}

impl VirtualDisplayPlane {
    /// Eye-relative pose explicitly selected as the default from accepted SAM
    /// session 1788765280-345175455. This is a development convenience, not
    /// a calibration for the current eye/sign epoch or a physical measurement.
    /// Keep the centered nominal model separate for solver priors and tests.
    pub(crate) fn development_default() -> Self {
        Self {
            center_inches: [-4.242625177491027, -16.718478032593996, 22.343548660299646],
            right_axis: [-0.9983132233961985, 0.01620869432679404, -0.05574931587483707],
            down_axis: [-0.053388187992544585, 0.12100934046335098, 0.9912146290806535],
            width_inches: 23.228346456692915,
            height_inches: 13.110236220472443,
        }
    }

    pub(crate) fn nominal() -> Self {
        let (width_inches, height_inches) = nominal_display_dimensions_inches();
        Self {
            center_inches: [0.0, 0.0, NOMINAL_DISPLAY_DISTANCE_INCHES],
            right_axis: [1.0, 0.0, 0.0],
            down_axis: [0.0, 1.0, 0.0],
            width_inches,
            height_inches,
        }
    }

    pub(crate) fn target(self, relative_gaze: RelativeGazeVector) -> Option<(f64, f64)> {
        let direction = relative_gaze.as_array();
        let normal = cross3(self.right_axis, self.down_axis);
        let denominator = dot3(direction, normal);
        if !denominator.is_finite() || denominator.abs() < 1.0e-8 {
            return None;
        }
        let distance = dot3(self.center_inches, normal) / denominator;
        if !distance.is_finite() || distance <= 0.0 {
            return None;
        }
        let hit = scale3(direction, distance);
        let offset = sub3(hit, self.center_inches);
        let target = (
            0.5 + dot3(offset, self.right_axis) / self.width_inches,
            0.5 + dot3(offset, self.down_axis) / self.height_inches,
        );
        (target.0.is_finite() && target.1.is_finite()).then_some(target)
    }

    pub(crate) fn distance_inches(self) -> f64 {
        norm3(self.center_inches)
    }
}

impl GazeAffine {
    pub(crate) fn map(self, feature: (f64, f64)) -> (f64, f64) {
        (
            self.x[0] * feature.0 + self.x[1] * feature.1 + self.x[2],
            self.y[0] * feature.0 + self.y[1] * feature.1 + self.y[2],
        )
    }
}

pub(crate) fn gaze_affine_linear_geometry_plausible(affine: GazeAffine) -> bool {
    let [a, b, _] = affine.x;
    let [c, d, _] = affine.y;
    let trace = a * a + b * b + c * c + d * d;
    let determinant_squared = (a * d - b * c).powi(2);
    let discriminant = (trace * trace - 4.0 * determinant_squared).max(0.0);
    let maximum = ((trace + discriminant.sqrt()) * 0.5).sqrt();
    let minimum = ((trace - discriminant.sqrt()) * 0.5).max(0.0).sqrt();
    minimum.is_finite()
        && maximum.is_finite()
        && minimum >= VIRTUAL_MOUSE_AFFINE_MIN_SINGULAR_GAIN
        && maximum <= VIRTUAL_MOUSE_AFFINE_MAX_SINGULAR_GAIN
        && maximum / minimum <= VIRTUAL_MOUSE_AFFINE_MAX_CONDITION
}

pub(crate) fn calibration_models_have_shared_support(
    plane: VirtualDisplayPlane,
    affine: GazeAffine,
    observations: &[((f64, f64), (f64, f64))],
) -> bool {
    if !gaze_affine_linear_geometry_plausible(affine) {
        return false;
    }
    let shared_targets = observations
        .iter()
        .filter_map(|observation| {
            let plane_prediction =
                RelativeGazeVector::from_projected(observation.0 .0, observation.0 .1)
                    .and_then(|gaze| plane.target(gaze))?;
            let affine_prediction = affine.map(observation.0);
            let plane_residual = (plane_prediction.0 - observation.1 .0)
                .hypot(plane_prediction.1 - observation.1 .1);
            let affine_residual = (affine_prediction.0 - observation.1 .0)
                .hypot(affine_prediction.1 - observation.1 .1);
            let model_disagreement = (plane_prediction.0 - affine_prediction.0)
                .hypot(plane_prediction.1 - affine_prediction.1);
            (plane_residual <= VIRTUAL_MOUSE_PLANE_INLIER_RESIDUAL
                && affine_residual <= VIRTUAL_MOUSE_AFFINE_INLIER_RESIDUAL
                && model_disagreement <= VIRTUAL_MOUSE_MODEL_MAX_DISAGREEMENT)
                .then_some(observation.1)
        })
        .collect::<Vec<_>>();
    calibration_targets_have_required_coverage(shared_targets)
}

/// Intersect the eye-relative gaze ray with the uncalibrated physical display
/// plane. The screen is centered on the camera-facing eye axis; calibration
/// may later learn a correction for the real screen/camera/observer offsets.
pub(crate) fn nominal_display_target(relative_gaze: RelativeGazeVector) -> Option<(f64, f64)> {
    VirtualDisplayPlane::nominal().target(relative_gaze)
}

pub(crate) fn nominal_display_dimensions_inches() -> (f64, f64) {
    let aspect_hypotenuse = NOMINAL_DISPLAY_ASPECT_WIDTH.hypot(NOMINAL_DISPLAY_ASPECT_HEIGHT);
    (
        NOMINAL_DISPLAY_DIAGONAL_INCHES * NOMINAL_DISPLAY_ASPECT_WIDTH / aspect_hypotenuse,
        NOMINAL_DISPLAY_DIAGONAL_INCHES * NOMINAL_DISPLAY_ASPECT_HEIGHT / aspect_hypotenuse,
    )
}

pub(crate) fn display_ray_range_residuals(
    ray_ranges_inches: [f64; 3],
    directions: [[f64; 3]; 3],
    squared_target_distances_in2: [f64; 3],
) -> ([f64; 3], [[f64; 3]; 3]) {
    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    let mut residuals = [0.0; 3];
    let mut jacobian = [[0.0; 3]; 3];
    for (row, (left, right)) in pairs.into_iter().enumerate() {
        let cosine = dot3(directions[left], directions[right]);
        let scale = squared_target_distances_in2[row].max(1.0);
        residuals[row] = (ray_ranges_inches[left] * ray_ranges_inches[left] + ray_ranges_inches[right] * ray_ranges_inches[right]
            - 2.0 * cosine * ray_ranges_inches[left] * ray_ranges_inches[right]
            - squared_target_distances_in2[row])
            / scale;
        jacobian[row][left] = (2.0 * ray_ranges_inches[left] - 2.0 * cosine * ray_ranges_inches[right]) / scale;
        jacobian[row][right] = (2.0 * ray_ranges_inches[right] - 2.0 * cosine * ray_ranges_inches[left]) / scale;
    }
    (residuals, jacobian)
}

pub(crate) fn solve_display_ray_ranges(
    directions: [[f64; 3]; 3],
    squared_target_distances_in2: [f64; 3],
) -> Vec<[f64; 3]> {
    const SEEDS: [f64; 8] = [8.0, 12.0, 18.0, 24.0, 32.0, 44.0, 64.0, 92.0];
    const PERTURBATIONS: [[f64; 3]; 7] = [
        [1.0, 1.0, 1.0],
        [0.82, 1.18, 1.0],
        [1.18, 0.82, 1.0],
        [0.90, 0.90, 1.15],
        [1.10, 1.10, 0.85],
        [0.80, 1.05, 1.20],
        [1.20, 0.95, 0.80],
    ];
    let mut solutions = Vec::<[f64; 3]>::new();
    for seed in SEEDS {
        for perturbation in PERTURBATIONS {
            let mut ray_ranges_inches = [
                seed * perturbation[0],
                seed * perturbation[1],
                seed * perturbation[2],
            ];
            for _ in 0..64 {
                let (residuals, jacobian) =
                    display_ray_range_residuals(ray_ranges_inches, directions, squared_target_distances_in2);
                let error = dot3(residuals, residuals);
                if error < 1.0e-18 {
                    break;
                }
                let Some(delta) = solve_3x3([
                    [
                        jacobian[0][0],
                        jacobian[0][1],
                        jacobian[0][2],
                        -residuals[0],
                    ],
                    [
                        jacobian[1][0],
                        jacobian[1][1],
                        jacobian[1][2],
                        -residuals[1],
                    ],
                    [
                        jacobian[2][0],
                        jacobian[2][1],
                        jacobian[2][2],
                        -residuals[2],
                    ],
                ]) else {
                    break;
                };
                let mut accepted = false;
                let mut step = 1.0;
                while step >= 1.0 / 128.0 {
                    let candidate = [
                        ray_ranges_inches[0] + step * delta[0],
                        ray_ranges_inches[1] + step * delta[1],
                        ray_ranges_inches[2] + step * delta[2],
                    ];
                    if candidate.iter().all(|ray_range_inches| (4.0..=144.0).contains(ray_range_inches)) {
                        let candidate_residuals =
                            display_ray_range_residuals(candidate, directions, squared_target_distances_in2).0;
                        if dot3(candidate_residuals, candidate_residuals) < error {
                            ray_ranges_inches = candidate;
                            accepted = true;
                            break;
                        }
                    }
                    step *= 0.5;
                }
                if !accepted {
                    break;
                }
            }
            let residuals = display_ray_range_residuals(ray_ranges_inches, directions, squared_target_distances_in2).0;
            let error = dot3(residuals, residuals);
            if error < 1.0e-10
                && !solutions.iter().any(|existing| {
                    existing
                        .iter()
                        .zip(ray_ranges_inches)
                        .all(|(left, right)| (left - right).abs() < 1.0e-4)
                })
            {
                solutions.push(ray_ranges_inches);
            }
        }
    }
    solutions
}

pub(crate) fn display_object_coordinates(
    target: (f64, f64),
    width_inches: f64,
    height_inches: f64,
) -> (f64, f64) {
    (
        (target.0 - 0.5) * width_inches,
        (target.1 - 0.5) * height_inches,
    )
}

pub(crate) fn display_plane_from_three_rays(
    directions: [[f64; 3]; 3],
    targets: [(f64, f64); 3],
    ray_ranges_inches: [f64; 3],
    width_inches: f64,
    height_inches: f64,
) -> Option<VirtualDisplayPlane> {
    let object =
        targets.map(|target| display_object_coordinates(target, width_inches, height_inches));
    let dx10 = object[1].0 - object[0].0;
    let dy10 = object[1].1 - object[0].1;
    let dx20 = object[2].0 - object[0].0;
    let dy20 = object[2].1 - object[0].1;
    let determinant = dx10 * dy20 - dx20 * dy10;
    if !determinant.is_finite() || determinant.abs() < width_inches * height_inches * 0.01 {
        return None;
    }
    let points = [
        scale3(directions[0], ray_ranges_inches[0]),
        scale3(directions[1], ray_ranges_inches[1]),
        scale3(directions[2], ray_ranges_inches[2]),
    ];
    let p10 = sub3(points[1], points[0]);
    let p20 = sub3(points[2], points[0]);
    let right_raw = scale3(
        sub3(scale3(p10, dy20), scale3(p20, dy10)),
        1.0 / determinant,
    );
    let down_raw = scale3(
        sub3(scale3(p20, dx10), scale3(p10, dx20)),
        1.0 / determinant,
    );
    let right_axis = normalized3(right_raw)?;
    let down_axis = normalized3(sub3(
        down_raw,
        scale3(right_axis, dot3(down_raw, right_axis)),
    ))?;
    let mut center_inches = [0.0; 3];
    for index in 0..3 {
        center_inches = add3(
            center_inches,
            sub3(
                points[index],
                add3(
                    scale3(right_axis, object[index].0),
                    scale3(down_axis, object[index].1),
                ),
            ),
        );
    }
    center_inches = scale3(center_inches, 1.0 / 3.0);
    let plane = VirtualDisplayPlane {
        center_inches,
        right_axis,
        down_axis,
        width_inches,
        height_inches,
    };
    display_plane_geometry_plausible(plane).then_some(plane)
}

pub(crate) fn display_plane_geometry_plausible(plane: VirtualDisplayPlane) -> bool {
    if !plane.center_inches.iter().all(|value| value.is_finite())
        || plane.center_inches[2] <= 2.0
        || !(4.0..=144.0).contains(&plane.distance_inches())
        || (norm3(plane.right_axis) - 1.0).abs() > 1.0e-5
        || (norm3(plane.down_axis) - 1.0).abs() > 1.0e-5
        || dot3(plane.right_axis, plane.down_axis).abs() > 1.0e-5
    {
        return false;
    }
    [-0.5, 0.5].into_iter().all(|x| {
        [-0.5, 0.5].into_iter().all(|y| {
            add3(
                plane.center_inches,
                add3(
                    scale3(plane.right_axis, x * plane.width_inches),
                    scale3(plane.down_axis, y * plane.height_inches),
                ),
            )[2] > 2.0
        })
    })
}

pub(crate) fn display_plane_residual(
    plane: VirtualDisplayPlane,
    observation: &((f64, f64), (f64, f64)),
) -> Option<[f64; 2]> {
    let gaze = RelativeGazeVector::from_projected(observation.0 .0, observation.0 .1)?;
    let mapped = plane.target(gaze)?;
    Some([mapped.0 - observation.1 .0, mapped.1 - observation.1 .1])
}

pub(crate) fn display_plane_robust_cost(
    plane: VirtualDisplayPlane,
    observations: &[((f64, f64), (f64, f64))],
) -> f64 {
    const HUBER: f64 = 0.055;
    observations
        .iter()
        .map(|observation| {
            let residual = display_plane_residual(plane, observation)
                .map(|value| value[0].hypot(value[1]))
                .unwrap_or(2.0);
            if residual <= HUBER {
                0.5 * residual * residual
            } else {
                HUBER * (residual - 0.5 * HUBER)
            }
        })
        .sum::<f64>()
        / observations.len().max(1) as f64
}

pub(crate) fn solve_6x6(mut matrix: [[f64; 7]; 6]) -> Option<[f64; 6]> {
    for pivot in 0..6 {
        let best = (pivot..6).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right][pivot].abs())
        })?;
        if matrix[best][pivot].abs() < 1.0e-10 {
            return None;
        }
        matrix.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for column in pivot..7 {
            matrix[pivot][column] /= divisor;
        }
        for row in 0..6 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..7 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    Some(std::array::from_fn(|row| matrix[row][6]))
}

pub(crate) fn apply_display_plane_delta(
    plane: VirtualDisplayPlane,
    delta: [f64; 6],
    scale: f64,
) -> Option<VirtualDisplayPlane> {
    let translation = [delta[0] * scale, delta[1] * scale, delta[2] * scale];
    let rotation_increment_rad = [delta[3] * scale, delta[4] * scale, delta[5] * scale];
    let right_axis = normalized3(add3(plane.right_axis, cross3(rotation_increment_rad, plane.right_axis)))?;
    let down_rotated = add3(plane.down_axis, cross3(rotation_increment_rad, plane.down_axis));
    let down_axis = normalized3(sub3(
        down_rotated,
        scale3(right_axis, dot3(down_rotated, right_axis)),
    ))?;
    Some(VirtualDisplayPlane {
        center_inches: add3(plane.center_inches, translation),
        right_axis,
        down_axis,
        ..plane
    })
}

pub(crate) fn refine_virtual_display_plane(
    mut plane: VirtualDisplayPlane,
    observations: &[((f64, f64), (f64, f64))],
) -> VirtualDisplayPlane {
    const HUBER: f64 = 0.055;
    for _ in 0..18 {
        let mut normal = [[0.0; 7]; 6];
        let mut used = 0usize;
        for observation in observations {
            let Some(residual) = display_plane_residual(plane, observation) else {
                continue;
            };
            let length = residual[0].hypot(residual[1]);
            let weight = if length <= HUBER || length <= 1.0e-12 {
                1.0
            } else {
                HUBER / length
            };
            let mut jacobian = [[0.0; 6]; 2];
            for parameter in 0..6 {
                let epsilon = if parameter < 3 { 1.0e-3 } else { 1.0e-5 };
                let mut step = [0.0; 6];
                step[parameter] = epsilon;
                let Some(perturbed) = apply_display_plane_delta(plane, step, 1.0) else {
                    continue;
                };
                let Some(next) = display_plane_residual(perturbed, observation) else {
                    continue;
                };
                jacobian[0][parameter] = (next[0] - residual[0]) / epsilon;
                jacobian[1][parameter] = (next[1] - residual[1]) / epsilon;
            }
            for row in 0..6 {
                for column in 0..6 {
                    normal[row][column] += weight
                        * (jacobian[0][row] * jacobian[0][column]
                            + jacobian[1][row] * jacobian[1][column]);
                }
                normal[row][6] -=
                    weight * (jacobian[0][row] * residual[0] + jacobian[1][row] * residual[1]);
            }
            used += 1;
        }
        if used < 4 {
            break;
        }
        for (index, row) in normal.iter_mut().enumerate() {
            row[index] += 1.0e-7;
        }
        let Some(mut delta) = solve_6x6(normal) else {
            break;
        };
        let translation_length = norm3([delta[0], delta[1], delta[2]]);
        if translation_length > 2.0 {
            let scale = 2.0 / translation_length;
            for value in &mut delta[..3] {
                *value *= scale;
            }
        }
        let rotation_length = norm3([delta[3], delta[4], delta[5]]);
        if rotation_length > 0.08 {
            let scale = 0.08 / rotation_length;
            for value in &mut delta[3..] {
                *value *= scale;
            }
        }
        if norm3([delta[0], delta[1], delta[2]]) < 1.0e-6
            && norm3([delta[3], delta[4], delta[5]]) < 1.0e-7
        {
            break;
        }
        let previous_cost = display_plane_robust_cost(plane, observations);
        let mut step_scale = 1.0;
        let mut accepted = None;
        while step_scale >= 1.0 / 64.0 {
            if let Some(candidate) = apply_display_plane_delta(plane, delta, step_scale) {
                if display_plane_geometry_plausible(candidate)
                    && display_plane_robust_cost(candidate, observations) < previous_cost
                {
                    accepted = Some(candidate);
                    break;
                }
            }
            step_scale *= 0.5;
        }
        let Some(candidate) = accepted else {
            break;
        };
        plane = candidate;
    }
    plane
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DisplayPlaneScore {
    pub(crate) inliers: usize,
    pub(crate) rms: f64,
    pub(crate) robust_cost: f64,
}

pub(crate) fn targets_are_distributed(targets: &[(f64, f64)]) -> bool {
    if targets.len() < 4 {
        return false;
    }
    let min_x = targets
        .iter()
        .map(|target| target.0)
        .fold(f64::INFINITY, f64::min);
    let max_x = targets
        .iter()
        .map(|target| target.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = targets
        .iter()
        .map(|target| target.1)
        .fold(f64::INFINITY, f64::min);
    let max_y = targets
        .iter()
        .map(|target| target.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let required_span = (1.0 - 2.0 * VIRTUAL_MOUSE_CALIBRATION_INSET) * 0.90;
    if max_x - min_x < required_span || max_y - min_y < required_span {
        return false;
    }
    for first in 0..targets.len() {
        for second in first + 1..targets.len() {
            for third in second + 1..targets.len() {
                let area = (targets[second].0 - targets[first].0)
                    * (targets[third].1 - targets[first].1)
                    - (targets[third].0 - targets[first].0)
                        * (targets[second].1 - targets[first].1);
                // Judge two-dimensional coverage relative to the target
                // region, including when calibration is confined to 20%.
                if area.abs() >= 0.36 * (1.0 - 2.0 * VIRTUAL_MOUSE_CALIBRATION_INSET).powi(2) {
                    return true;
                }
            }
        }
    }
    false
}

pub(crate) fn calibration_targets_have_required_coverage(
    targets: impl IntoIterator<Item = (f64, f64)>,
) -> bool {
    let targets = targets.into_iter().collect::<Vec<_>>();
    let contains = |expected: (f64, f64)| {
        targets.iter().any(|target| {
            (target.0 - expected.0).abs() <= 1.0e-9 && (target.1 - expected.1).abs() <= 1.0e-9
        })
    };
    let corner_count = [0usize, 1, 3, 4]
        .into_iter()
        .filter(|&index| contains(VIRTUAL_MOUSE_CALIBRATION_TARGETS[index]))
        .count();
    targets.len() >= VIRTUAL_MOUSE_MIN_STABLE_TARGETS
        && contains(VIRTUAL_MOUSE_CALIBRATION_TARGETS[2])
        && corner_count >= 3
        && targets_are_distributed(targets.as_slice())
}

pub(crate) fn score_virtual_display_plane(
    plane: VirtualDisplayPlane,
    observations: &[((f64, f64), (f64, f64))],
) -> Option<DisplayPlaneScore> {
    let mut squared = 0.0;
    let mut inlier_targets = Vec::new();
    for observation in observations {
        let Some(residual) = display_plane_residual(plane, observation) else {
            continue;
        };
        let length = residual[0].hypot(residual[1]);
        if length <= VIRTUAL_MOUSE_PLANE_INLIER_RESIDUAL {
            squared += length * length;
            inlier_targets.push(observation.1);
        }
    }
    if !calibration_targets_have_required_coverage(inlier_targets.iter().copied()) {
        return None;
    }
    let rms = (squared / inlier_targets.len() as f64).sqrt();
    Some(DisplayPlaneScore {
        inliers: inlier_targets.len(),
        rms,
        robust_cost: display_plane_robust_cost(plane, observations),
    })
}

pub(crate) fn fit_virtual_display_plane(
    observations: &[((f64, f64), (f64, f64))],
) -> Option<VirtualDisplayPlane> {
    fit_virtual_display_plane_with_dimensions(observations, nominal_display_dimensions_inches())
}

pub(crate) fn fit_virtual_display_plane_with_dimensions(
    observations: &[((f64, f64), (f64, f64))],
    dimensions: (f64, f64),
) -> Option<VirtualDisplayPlane> {
    if observations.len() < VIRTUAL_MOUSE_MIN_STABLE_TARGETS {
        return None;
    }
    let (width_inches, height_inches) = dimensions;
    if !width_inches.is_finite() || !height_inches.is_finite()
        || !(1.0..=150.0).contains(&width_inches) || !(1.0..=150.0).contains(&height_inches) {
        return None;
    }
    let mut best: Option<(VirtualDisplayPlane, DisplayPlaneScore)> = None;
    for first in 0..observations.len() {
        for second in first + 1..observations.len() {
            for third in second + 1..observations.len() {
                let selected = [
                    observations[first],
                    observations[second],
                    observations[third],
                ];
                let targets = selected.map(|observation| observation.1);
                let seed_object = targets
                    .map(|target| display_object_coordinates(target, width_inches, height_inches));
                let seed_determinant = (seed_object[1].0 - seed_object[0].0)
                    * (seed_object[2].1 - seed_object[0].1)
                    - (seed_object[2].0 - seed_object[0].0) * (seed_object[1].1 - seed_object[0].1);
                if seed_determinant.abs() < width_inches * height_inches * 0.01 {
                    continue;
                }
                let directions = selected.map(|observation| {
                    RelativeGazeVector::from_projected(observation.0 .0, observation.0 .1)
                        .map(RelativeGazeVector::as_array)
                });
                let Some(directions) = directions
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .and_then(|values| values.try_into().ok())
                else {
                    continue;
                };
                let object = targets
                    .map(|target| display_object_coordinates(target, width_inches, height_inches));
                let squared_distance = |left: usize, right: usize| {
                    (object[left].0 - object[right].0).powi(2)
                        + (object[left].1 - object[right].1).powi(2)
                };
                let squared_target_distances_in2 = [
                    squared_distance(0, 1),
                    squared_distance(0, 2),
                    squared_distance(1, 2),
                ];
                for ray_ranges_inches in solve_display_ray_ranges(directions, squared_target_distances_in2) {
                    let Some(seed) = display_plane_from_three_rays(
                        directions,
                        targets,
                        ray_ranges_inches,
                        width_inches,
                        height_inches,
                    ) else {
                        continue;
                    };
                    let candidate = refine_virtual_display_plane(seed, observations);
                    let Some(score) = score_virtual_display_plane(candidate, observations) else {
                        continue;
                    };
                    let replace = match best.as_ref() {
                        None => true,
                        Some((_, prior)) if score.inliers != prior.inliers => {
                            score.inliers > prior.inliers
                        }
                        Some((_, prior))
                            if (score.robust_cost - prior.robust_cost).abs() > 1.0e-12 =>
                        {
                            score.robust_cost < prior.robust_cost
                        }
                        Some((_, prior)) if (score.rms - prior.rms).abs() > 1.0e-12 => {
                            score.rms < prior.rms
                        }
                        Some((existing, _)) => {
                            (candidate.distance_inches() - NOMINAL_DISPLAY_DISTANCE_INCHES).abs()
                                < (existing.distance_inches() - NOMINAL_DISPLAY_DISTANCE_INCHES)
                                    .abs()
                        }
                    };
                    if replace {
                        best = Some((candidate, score));
                    }
                }
            }
        }
    }
    if let Some((_, score)) = best.as_ref() {
        eprintln!(
            "mouse calibration best 3D candidate: {} inliers rms={:.4} limit={:.4}",
            score.inliers, score.rms, VIRTUAL_MOUSE_PLANE_MAX_RMS
        );
    }
    best.filter(|(_, score)| {
        score.inliers >= VIRTUAL_MOUSE_MIN_STABLE_TARGETS
            && score.rms <= VIRTUAL_MOUSE_PLANE_MAX_RMS
    })
        .map(|(plane, score)| {
            eprintln!(
                "mouse calibration robust 3D fit accepted {}/{} targets rms={:.4} screen-fraction cost={:.6}",
                score.inliers,
                observations.len(),
                score.rms,
                score.robust_cost,
            );
            plane
        })
}

pub(crate) fn fit_affine_component(
    features: [(f64, f64); 4],
    outputs: [f64; 4],
) -> Option<[f64; 3]> {
    let mut normal = [[0.0; 4]; 3];
    for (feature, output) in features.into_iter().zip(outputs) {
        let row = [feature.0, feature.1, 1.0];
        for y in 0..3 {
            for x in 0..3 {
                normal[y][x] += row[y] * row[x];
            }
            normal[y][3] += row[y] * output;
        }
    }
    solve_3x3(normal)
}

pub(crate) fn fit_gaze_affine(
    features: [(f64, f64); 4],
    targets: [(f64, f64); 4],
) -> Option<GazeAffine> {
    if features
        .iter()
        .chain(targets.iter())
        .any(|point| !point.0.is_finite() || !point.1.is_finite())
    {
        return None;
    }
    let mean = features.iter().fold((0.0, 0.0), |sum, feature| {
        (sum.0 + feature.0, sum.1 + feature.1)
    });
    let mean = (
        mean.0 / features.len() as f64,
        mean.1 / features.len() as f64,
    );
    let scale = features.iter().fold((0.0, 0.0), |sum, feature| {
        (
            sum.0 + (feature.0 - mean.0).powi(2),
            sum.1 + (feature.1 - mean.1).powi(2),
        )
    });
    let scale = (
        (scale.0 / features.len() as f64).sqrt(),
        (scale.1 / features.len() as f64).sqrt(),
    );
    if !scale.0.is_finite() || !scale.1.is_finite() || scale.0 < 1.0e-9 || scale.1 < 1.0e-9 {
        return None;
    }
    let normalized = features.map(|feature| {
        (
            (feature.0 - mean.0) / scale.0,
            (feature.1 - mean.1) / scale.1,
        )
    });
    let normalized_x = fit_affine_component(normalized, targets.map(|target| target.0))?;
    let normalized_y = fit_affine_component(normalized, targets.map(|target| target.1))?;
    let restore = |component: [f64; 3]| {
        [
            component[0] / scale.0,
            component[1] / scale.1,
            component[2] - component[0] * mean.0 / scale.0 - component[1] * mean.1 / scale.1,
        ]
    };
    let x = restore(normalized_x);
    let y = restore(normalized_y);
    if x.iter()
        .chain(y.iter())
        .any(|coefficient| !coefficient.is_finite())
    {
        return None;
    }
    Some(GazeAffine { x, y })
}

pub(crate) fn fit_gaze_affine_least_squares(
    observations: &[((f64, f64), (f64, f64))],
) -> Option<GazeAffine> {
    if observations.len() < 3
        || observations.iter().any(|(feature, target)| {
            !feature.0.is_finite()
                || !feature.1.is_finite()
                || !target.0.is_finite()
                || !target.1.is_finite()
        })
    {
        return None;
    }
    let component = |axis: usize| {
        let mut normal = [[0.0; 4]; 3];
        for (feature, target) in observations {
            let row = [feature.0, feature.1, 1.0];
            let output = if axis == 0 { target.0 } else { target.1 };
            for y in 0..3 {
                for x in 0..3 {
                    normal[y][x] += row[y] * row[x];
                }
                normal[y][3] += row[y] * output;
            }
        }
        solve_3x3(normal)
    };
    let affine = GazeAffine {
        x: component(0)?,
        y: component(1)?,
    };
    affine
        .x
        .iter()
        .chain(affine.y.iter())
        .all(|coefficient| coefficient.is_finite())
        .then_some(affine)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GazeAffineScore {
    pub(crate) inliers: usize,
    pub(crate) rms: f64,
    pub(crate) robust_cost: f64,
}

pub(crate) fn gaze_affine_residual(
    affine: GazeAffine,
    observation: ((f64, f64), (f64, f64)),
) -> f64 {
    let mapped = affine.map(observation.0);
    (mapped.0 - observation.1 .0).hypot(mapped.1 - observation.1 .1)
}

pub(crate) fn score_gaze_affine(
    affine: GazeAffine,
    observations: &[((f64, f64), (f64, f64))],
) -> Option<GazeAffineScore> {
    let mut squared = 0.0;
    let mut robust_cost = 0.0;
    let mut inlier_targets = Vec::new();
    for &observation in observations {
        let residual = gaze_affine_residual(affine, observation);
        if !residual.is_finite() {
            continue;
        }
        let huber = VIRTUAL_MOUSE_AFFINE_INLIER_RESIDUAL;
        robust_cost += if residual <= huber {
            0.5 * residual * residual
        } else {
            huber * (residual - 0.5 * huber)
        };
        if residual <= VIRTUAL_MOUSE_AFFINE_INLIER_RESIDUAL {
            squared += residual * residual;
            inlier_targets.push(observation.1);
        }
    }
    if !calibration_targets_have_required_coverage(inlier_targets.iter().copied()) {
        return None;
    }
    Some(GazeAffineScore {
        inliers: inlier_targets.len(),
        rms: (squared / inlier_targets.len() as f64).sqrt(),
        robust_cost,
    })
}

pub(crate) fn fit_robust_gaze_affine(
    observations: &[((f64, f64), (f64, f64))],
) -> Option<GazeAffine> {
    if observations.len() < VIRTUAL_MOUSE_MIN_STABLE_TARGETS {
        return None;
    }
    // Exact three-point seeds amplify noise in the small fixation field.
    // Also consider the all-point least-squares fit, under the same gates.
    let mut best: Option<(GazeAffine, GazeAffineScore)> =
        fit_gaze_affine_least_squares(observations)
            .filter(|affine| gaze_affine_linear_geometry_plausible(*affine))
            .and_then(|affine| {
                score_gaze_affine(affine, observations).map(|score| (affine, score))
            });
    for first in 0..observations.len() {
        for second in first + 1..observations.len() {
            for third in second + 1..observations.len() {
                let seed_observations = [
                    observations[first],
                    observations[second],
                    observations[third],
                ];
                let Some(seed) = fit_gaze_affine_least_squares(&seed_observations) else {
                    continue;
                };
                let inliers = observations
                    .iter()
                    .copied()
                    .filter(|observation| {
                        gaze_affine_residual(seed, *observation)
                            <= VIRTUAL_MOUSE_AFFINE_INLIER_RESIDUAL
                    })
                    .collect::<Vec<_>>();
                if !calibration_targets_have_required_coverage(
                    inliers.iter().map(|observation| observation.1),
                ) {
                    continue;
                }
                let Some(refined) = fit_gaze_affine_least_squares(inliers.as_slice()) else {
                    continue;
                };
                let Some(score) = score_gaze_affine(refined, observations) else {
                    continue;
                };
                let replace = match best {
                    None => true,
                    Some((_, previous)) if score.inliers != previous.inliers => {
                        score.inliers > previous.inliers
                    }
                    Some((_, previous))
                        if (score.robust_cost - previous.robust_cost).abs() > 1.0e-12 =>
                    {
                        score.robust_cost < previous.robust_cost
                    }
                    Some((_, previous)) => score.rms < previous.rms,
                };
                if replace {
                    best = Some((refined, score));
                }
            }
        }
    }
    best.filter(|(affine, score)| {
        score.inliers >= VIRTUAL_MOUSE_MIN_STABLE_TARGETS
            && score.rms <= VIRTUAL_MOUSE_AFFINE_MAX_RMS
            && gaze_affine_linear_geometry_plausible(*affine)
    })
    .map(|(affine, score)| {
        eprintln!(
            "mouse calibration robust 2D affine accepted {}/{} targets rms={:.4} screen fraction",
            score.inliers,
            observations.len(),
            score.rms,
        );
        affine
    })
}
