//! Cross-ROI timing/settling and user-specific vergence factors.
//!
//! Source-time compatibility; unmeasured settling remains unknown. The existing
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
    pub(crate) maximum_joint_skew_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CoordinationUnavailable {
    NoEyes,
    IncompatibleClocks,
}

#[derive(Default)]
pub(crate) struct BinocularCoordinator;

impl BinocularCoordinator {
    pub(crate) fn coordinate(
        &mut self,
        request: BinocularRequest<'_>,
    ) -> Result<BinocularFactors, CoordinationUnavailable> {
        match request.eyes {
            [None,None] => Err(CoordinationUnavailable::NoEyes),
            [Some(a),Some(b)] => {
                let skew = a.exposure.separation_ns(b.exposure)
                    .ok_or(CoordinationUnavailable::IncompatibleClocks)?;
                // No motion history is available at this boundary yet. Only
                // same-stamped reads are granted simultaneous fixation. The
                // optical row-time allowance remains explicit in the solver.
                Ok(BinocularFactors { source_skew_ns: Some(skew),
                    maximum_joint_skew_ns: 0, ..Default::default() })
            }
            _ => Ok(BinocularFactors::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_evidence_never_manufactures_settling_or_ipd() {
        assert_eq!(
            BinocularCoordinator.coordinate(BinocularRequest { eyes: [None, None] }),
            Err(CoordinationUnavailable::NoEyes),
        );
        assert_eq!(BinocularFactors::default().settled, [None, None]);
        assert_eq!(BinocularFactors::default().ipd_mm, None);
    }

    #[test]
    fn clock_policy_keeps_weak_or_missing_eyes_optional_without_inventing_gaze() {
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
            let factors = BinocularCoordinator.coordinate(BinocularRequest { eyes }).unwrap();
            assert_eq!(factors.maximum_joint_skew_ns,0);
            assert_eq!(factors.settled,[None,None]);
            assert_eq!(factors.vergence_angle_radians,None);
        }
    }
}
