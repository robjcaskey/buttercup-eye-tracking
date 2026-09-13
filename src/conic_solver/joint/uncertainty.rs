//! Local uncertainty in the current robust conic model, not a confidence score.
//!
//! Invert the complete undamped information matrix, then project to the shared
//! target and each eye's angular tangent plane. This marginalizes center/range,
//! radii, pupil decentration/depth and axis alignment rather than freezing them
//! at their fitted values. Alternative associations/sign modes and unknown
//! model error are not represented by this local approximation.
use super::*;

#[derive(Clone, Debug)]
pub(crate) struct LocalUncertainty {
    pub(crate) status: &'static str,
    pub(crate) free_parameters: usize,
    /// Active box/nested/scene bounds invalidate an unconstrained Gaussian
    /// approximation. They never manufacture a zero-variance estimate.
    pub(crate) at_constraint: bool,
    pub(crate) target_covariance_mm2: Option<[[f64; 3]; 3]>,
    /// Small-angle covariance in orthonormal tangent coordinates, radians².
    pub(crate) gaze_tangent_covariance_rad2: [Option<[[f64; 2]; 2]>; 2],
    pub(crate) gaze_tangent_basis_camera: [Option<[[f64; 3]; 2]>; 2],
    /// Internal proposal geometry for posterior integration. Never serialized
    /// as calibrated information or added to the observation objective.
    pub(super) information: Option<Vec<Vec<f64>>>,
}

impl LocalUncertainty {
    pub(crate) fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "status": self.status, "free_parameters": self.free_parameters,
            "at_constraint": self.at_constraint,
            "target_covariance_mm2": self.target_covariance_mm2,
            "gaze_tangent_covariance_rad2": self.gaze_tangent_covariance_rad2,
            "gaze_tangent_basis_camera": self.gaze_tangent_basis_camera,
            "gaze_worst_axis_sigma_degrees": [self.worst_axis_sigma_degrees(0), self.worst_axis_sigma_degrees(1)],
            "nuisance_geometry": "marginalized, not fixed at fitted values",
            "contract": "local robust-model Gaussian approximation; conditional on selected arcs, association, camera/scale priors and one sign basin; not calibrated coverage or full multimodal uncertainty",
        })
    }

    pub(crate) fn worst_axis_sigma_degrees(&self, eye: usize) -> Option<f64> {
        let c = self
            .gaze_tangent_covariance_rad2
            .get(eye)
            .copied()
            .flatten()?;
        let largest = 0.5 * (c[0][0] + c[1][1] + (c[0][0] - c[1][1]).hypot(2.0 * c[0][1]));
        (largest.is_finite() && largest >= 0.0).then(|| largest.sqrt().to_degrees())
    }
}

/// Column-normalized Cholesky inverse. No Levenberg damping, ridge floor or
/// pseudoinverse that could turn an unobserved direction into zero variance.
pub(super) fn inverse_information(information: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = information.len();
    if n == 0 || information.iter().any(|row| row.len() != n) {
        return None;
    }
    let scale = (0..n).map(|i| information[i][i].sqrt()).collect::<Vec<_>>();
    if scale.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return None;
    }
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let a = information[i][j] / (scale[i] * scale[j]);
            let b = information[j][i] / (scale[i] * scale[j]);
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-8 {
                return None;
            }
            let value = 0.5 * (a + b) - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                // This is a numerical observability guard, not added data.
                if !value.is_finite() || value <= 1e-10 {
                    return None;
                }
                l[i][j] = value.sqrt();
            } else {
                l[i][j] = value / l[j][j];
            }
        }
    }
    let mut inverse = vec![vec![0.0; n]; n];
    for column in 0..n {
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = ((if i == column { 1.0 } else { 0.0 })
                - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>())
                / l[i][i];
        }
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = (y[i] - (i + 1..n).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i];
        }
        for i in 0..n {
            inverse[i][column] = x[i] / (scale[i] * scale[column]);
        }
    }
    // Verify the inverse in the normalized system, where tiny physical-unit
    // diagonals cannot hide a bad conditioning residual.
    for i in 0..n {
        for j in 0..n {
            let value = (0..n)
                .map(|k| information[i][k] * inverse[k][j])
                .sum::<f64>()
                * scale[j]
                / scale[i];
            if !value.is_finite() || (value - (if i == j { 1.0 } else { 0.0 })).abs() > 1e-5 {
                return None;
            }
        }
    }
    Some(inverse)
}

fn project_covariance<const N: usize>(
    jacobian: &[[f64; N]],
    covariance: &[Vec<f64>],
) -> Option<[[f64; N]; N]> {
    if jacobian.len() != covariance.len() {
        return None;
    }
    let mut output = [[0.0; N]; N];
    for a in 0..N {
        for b in 0..=a {
            let value = (0..jacobian.len())
                .map(|i| {
                    (0..jacobian.len())
                        .map(|j| jacobian[i][a] * covariance[i][j] * jacobian[j][b])
                        .sum::<f64>()
                })
                .sum::<f64>();
            if !value.is_finite() || (a == b && value < -1e-8) {
                return None;
            }
            output[a][b] = value;
            output[b][a] = value;
        }
    }
    Some(output)
}

impl Problem<'_> {
    fn uncertainty_outputs(&self, p: &Parameters) -> Option<([f64; 3], [Option<[f64; 3]>; 2])> {
        let target = self.target(p)?;
        let mut gazes = [None; 2];
        for eye in 0..2 {
            if self.present[eye] {
                let k = TARGET_PARAMETERS + eye * EYE_PARAMETERS;
                gazes[eye] = Some(normalized3(sub3(target, [p[k], p[k + 1], p[k + 2]]))?);
            }
        }
        Some((target, gazes))
    }

    pub(super) fn local_uncertainty(&self, p: &Parameters) -> LocalUncertainty {
        let active = (0..PARAMETERS)
            .filter(|&i| self.lower[i] < self.upper[i])
            .collect::<Vec<_>>();
        let mut result = LocalUncertainty {
            status: "linearization-unavailable",
            free_parameters: active.len(),
            at_constraint: false,
            target_covariance_mm2: None,
            gaze_tangent_covariance_rad2: [None; 2],
            gaze_tangent_basis_camera: [None; 2],
            information: None,
        };
        let Some(conics) = self.conics(p) else {
            return result;
        };
        let selected = self.select(&conics);
        if selected.has_marginalization() {
            // The frozen-responsibility EM metric is useful for optimization
            // and proposals, but omits uncertainty about the discrete mask
            // state. Only the full marginal distribution may report spread.
            result.status = if selected.group_mixtures.iter().any(Option::is_some) {
                "arc-alternative-mixture-requires-distribution"
            } else {"mask-level-mixture-requires-distribution"};
            return result;
        }
        let rejected = self.rejected_groups(&conics, &selected);
        let Some(residuals) = self.residuals_with_rejection(p, &selected, Some(&rejected)) else {
            return result;
        };
        let Some((_, gazes)) = self.uncertainty_outputs(p) else {
            return result;
        };
        for (eye, gaze) in gazes.iter().enumerate() {
            if let Some(gaze) = gaze {
                let reference = if gaze[1].abs() < 0.9 {
                    [0.0, 1.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let Some(u) = normalized3(cross3(reference, *gaze)) else {
                    return result;
                };
                result.gaze_tangent_basis_camera[eye] = Some([u, cross3(*gaze, u)]);
            }
        }
        let mut derivatives = Vec::with_capacity(active.len());
        let mut target_jacobian = Vec::with_capacity(active.len());
        let mut gaze_jacobians: [Vec<[f64; 2]>; 2] =
            std::array::from_fn(|_| Vec::with_capacity(active.len()));
        for &i in &active {
            let step = 1e-4 * self.scales[i];
            let mut before = *p;
            let mut after = *p;
            before[i] -= step;
            after[i] += step;
            // Refuse a local Gaussian at a constrained optimum. A one-sided
            // optimizer derivative is useful for search, not evidence for a
            // symmetric uncertainty distribution beyond a hard boundary.
            if before[i] < self.lower[i] || after[i] > self.upper[i] {
                result.status = "constraint-active";
                result.at_constraint = true;
                return result;
            }
            let (Some(a), Some(b)) = (
                self.residuals_with_rejection(&before, &selected, Some(&rejected)),
                self.residuals_with_rejection(&after, &selected, Some(&rejected)),
            ) else {
                result.status = "constraint-active";
                result.at_constraint = true;
                return result;
            };
            if a.len() != residuals.len() || b.len() != residuals.len() {
                return result;
            }
            let factor = self.scales[i] / (2.0 * step);
            derivatives.push(
                a.iter()
                    .zip(&b)
                    .map(|(a, b)| (b - a) * factor)
                    .collect::<Vec<_>>(),
            );
            let (Some((ta, ga)), Some((tb, gb))) = (
                self.uncertainty_outputs(&before),
                self.uncertainty_outputs(&after),
            ) else {
                return result;
            };
            target_jacobian.push(std::array::from_fn(|j| (tb[j] - ta[j]) * factor));
            for eye in 0..2 {
                let derivative = if let (Some(a), Some(b), Some(basis)) =
                    (ga[eye], gb[eye], result.gaze_tangent_basis_camera[eye])
                {
                    let difference = scale3(sub3(b, a), factor);
                    [dot3(difference, basis[0]), dot3(difference, basis[1])]
                } else {
                    [0.0; 2]
                };
                gaze_jacobians[eye].push(derivative);
            }
        }
        let n = derivatives.len();
        let information = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        derivatives[i]
                            .iter()
                            .zip(&derivatives[j])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let Some(covariance) = inverse_information(&information) else {
            result.status = "rank-deficient";
            return result;
        };
        result.information = Some(information);
        result.target_covariance_mm2 = project_covariance(&target_jacobian, &covariance);
        for eye in 0..2 {
            if self.present[eye] {
                result.gaze_tangent_covariance_rad2[eye] =
                    project_covariance(&gaze_jacobians[eye], &covariance);
            }
        }
        result.status = if result.target_covariance_mm2.is_some() {
            "local-conditional"
        } else {
            "linearization-unavailable"
        };
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marginalization_retains_nuisance_correlation_instead_of_freezing_it() {
        // y = target + offset with unit noise; offset has sigma=3 prior.
        // Freezing offset gives variance 1, whereas marginalizing gives 10.
        let c = inverse_information(&[vec![1.0, 1.0], vec![1.0, 1.0 + 1.0 / 9.0]]).unwrap();
        assert!((c[0][0] - 10.0).abs() < 1e-10);
        let target = project_covariance(&[[1.0], [0.0]], &c).unwrap();
        assert!((target[0][0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn rank_deficiency_is_unknown_not_a_precise_pseudoinverse_or_damped_solve() {
        assert!(inverse_information(&[vec![1.0, 1.0], vec![1.0, 1.0]]).is_none());
        assert!(inverse_information(&[vec![1.0, 0.0], vec![0.0, 0.0]]).is_none());
        assert!(inverse_information(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_none());
    }

    #[test]
    fn physical_units_do_not_change_information_or_projected_covariance() {
        let a = inverse_information(&[vec![4.0, 1.0], vec![1.0, 3.0]]).unwrap();
        let b = inverse_information(&[vec![4e-12, 1e-6], vec![1e-6, 3.0]]).unwrap();
        let pa = project_covariance(&[[1.0, 0.0], [0.0, 1.0]], &a).unwrap();
        let pb = project_covariance(&[[1e-6, 0.0], [0.0, 1.0]], &b).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!((pa[i][j] - pb[i][j]).abs() < 1e-10);
            }
        }
    }
}
