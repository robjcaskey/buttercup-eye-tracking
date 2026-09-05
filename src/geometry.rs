//! Shared geometry primitives. Units and coordinate frames belong to their callers.
//!
//! Ellipse centers/radii are pixels in the caller's declared image frame;
//! angles are radians clockwise from image +x (image +y points down). This is
//! a shape, not an observation, anatomical label, covariance, or 3D eye pose.
//! Legacy callers still use untagged tuples; ROI evidence owns their provenance.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ellipse {
    pub center: (f64, f64),
    pub major_radius: f64,
    pub minor_radius: f64,
    pub angle: f64,
}

impl Ellipse {
    pub fn dense_points(self, count: usize) -> Vec<(f64, f64)> {
        let count = count.max(8);
        let (sin_angle, cos_angle) = self.angle.sin_cos();
        (0..count)
            .map(|index| {
                let phase = std::f64::consts::TAU * index as f64 / count as f64;
                let (sin_phase, cos_phase) = phase.sin_cos();
                let x = self.major_radius * cos_phase;
                let y = self.minor_radius * sin_phase;
                (
                    self.center.0 + cos_angle * x - sin_angle * y,
                    self.center.1 + sin_angle * x + cos_angle * y,
                )
            })
            .collect()
    }
}

pub(crate) fn ellipse_coordinate(point: (f64, f64), ellipse: Ellipse) -> f64 {
    let (angle_sine, angle_cosine) = ellipse.angle.sin_cos();
    let dx = point.0 - ellipse.center.0;
    let dy = point.1 - ellipse.center.1;
    let local_x = angle_cosine * dx + angle_sine * dy;
    let local_y = -angle_sine * dx + angle_cosine * dy;
    ((local_x / ellipse.major_radius.max(1.0)).powi(2)
        + (local_y / ellipse.minor_radius.max(1.0)).powi(2))
    .sqrt()
}

pub(crate) fn ellipse_axis_point(point: (f64, f64), reference: Ellipse) -> (f64, f64) {
    let (sine, cosine) = reference.angle.sin_cos();
    let dx = point.0 - reference.center.0;
    let dy = point.1 - reference.center.1;
    (cosine * dx + sine * dy, -sine * dx + cosine * dy)
}

pub(crate) fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

pub(crate) fn sub3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(crate) fn add3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub(crate) fn scale3(vector: [f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

pub(crate) fn cross3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

pub(crate) fn norm3(vector: [f64; 3]) -> f64 {
    dot3(vector, vector).sqrt()
}

pub(crate) fn normalized3(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm3(vector);
    (length.is_finite() && length > 1.0e-9).then(|| scale3(vector, 1.0 / length))
}

pub(crate) fn solve_3x3(mut matrix: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for pivot in 0..3 {
        let best = (pivot..3).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right][pivot].abs())
        })?;
        if matrix[best][pivot].abs() < 1.0e-9 {
            return None;
        }
        matrix.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for column in pivot..4 {
            matrix[pivot][column] /= divisor;
        }
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..4 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
        }
    }
    Some([matrix[0][3], matrix[1][3], matrix[2][3]])
}

pub(crate) fn blend_point(previous: (f64, f64), next: (f64, f64), alpha: f64) -> (f64, f64) {
    let alpha = alpha.clamp(0.0, 1.0);
    (
        previous.0 * (1.0 - alpha) + next.0 * alpha,
        previous.1 * (1.0 - alpha) + next.1 * alpha,
    )
}

pub(crate) fn wrapped_angle_distance(left: f64, right: f64) -> f64 {
    let mut delta = (left - right).abs() % std::f64::consts::PI;
    if delta > std::f64::consts::FRAC_PI_2 {
        delta = std::f64::consts::PI - delta;
    }
    delta
}

pub(crate) fn upper_median_or_zero(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}
