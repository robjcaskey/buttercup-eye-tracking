//! Cross-ROI timing/settling and user-specific vergence factors.
//!
//! Interface scaffold only; not called by live publication yet. The existing
//! coupled_eye_kinematics module couples pupil and surrounding tissue within
//! ONE eye, and must not be presented as binocular coordination.
//! This module supplies constraints; gaze_target_solver owns the final target.

use crate::roi_evidence::RoiConicEvidence;

/// Future coordination must accept either missing eye without inventing its
/// pose. Each present packet retains its own clock, exposure and optical state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BinocularRequest<'a> {
    pub(crate) eyes: [Option<RoiConicEvidence<'a>>; 2],
}

/// Settling is an observable motion proxy, not a measurement of lens
/// accommodation. Unknown values must not become a zero-delay/default IPD.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BinocularFactors {
    pub(crate) source_skew_ns: Option<u64>,
    pub(crate) settled: [Option<bool>; 2],
    pub(crate) vergence_angle_radians: Option<f64>,
    pub(crate) vergence_uncertainty_radians: Option<f64>,
    pub(crate) ipd_mm: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CoordinationUnavailable {
    NotImplemented,
}

#[derive(Default)]
pub(crate) struct BinocularCoordinator;

impl BinocularCoordinator {
    pub(crate) fn coordinate(
        &mut self,
        _request: BinocularRequest<'_>,
    ) -> Result<BinocularFactors, CoordinationUnavailable> {
        // No fake "settled" result and no equal-weight average of the eyes.
        Err(CoordinationUnavailable::NotImplemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_never_manufactures_binocular_factors() {
        assert_eq!(
            BinocularCoordinator.coordinate(BinocularRequest { eyes: [None, None] }),
            Err(CoordinationUnavailable::NotImplemented),
        );
        assert_eq!(BinocularFactors::default().settled, [None, None]);
        assert_eq!(BinocularFactors::default().ipd_mm, None);
    }

    #[test]
    fn joint_solver_scaffolds_abstain_with_a_missing_or_soft_second_eye() {
        use crate::roi_evidence::{ExposureKey, RoiId, SourceClock};
        let eye = RoiConicEvidence {
            exposure: ExposureKey {
                roi: RoiId(0),
                clock: SourceClock {
                    domain: 1,
                    epoch: 0,
                },
                sequence: 1,
                timestamp_ns: 1,
            },
            sensor_origin_px: [100, 200],
            dimensions_px: [420, 280],
            arcs: &[],
            conics: &[],
            detail_reliability: Some(0.2),
        };
        let soft_second_eye = RoiConicEvidence {
            exposure: ExposureKey {
                roi: RoiId(1),
                ..eye.exposure
            },
            detail_reliability: Some(0.05),
            ..eye
        };
        for eyes in [
            [Some(eye), None],
            [None, Some(soft_second_eye)],
            [Some(eye), Some(soft_second_eye)],
        ] {
            assert!(matches!(
                BinocularCoordinator.coordinate(BinocularRequest { eyes }),
                Err(CoordinationUnavailable::NotImplemented),
            ));
            assert!(matches!(
                crate::gaze_target_solver::solve_joint_gaze_target(
                    crate::gaze_target_solver::JointGazeRequest {
                        eyes,
                        binocular: None
                    }
                ),
                Err(crate::gaze_target_solver::JointGazeUnavailable::NotImplemented),
            ));
            assert!(matches!(
                crate::conic_solver::solve_joint_conics(crate::conic_solver::JointConicRequest {
                    eyes,
                    maximum_hypotheses: 16,
                    maximum_refinements: 4,
                }),
                Err(crate::conic_solver::JointConicUnavailable::NotImplemented),
            ));
        }
    }
}
