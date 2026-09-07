//! Source-keyed real-time orchestration of the joint segment objective.
//! No independently solved gaze point enters this module. A previous joint
//! target is only an optimization start, never a new observation or smoothing.

use crate::binocular_coordinator::source_pairing::{PairingUnavailable, SourcePairer};
use crate::conic_solver::joint::{solve_joint_conics, JointConicRequest, JointConicSolution, JointConicUnavailable, PinholeCamera};
use crate::eye_scene_model::binocular_pose::{approximate_scene, CoarseBinocularScene, EyePoseInput};
use crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence;
use crate::roi_evidence::{ExposureKey, SourceClock};
use std::sync::Arc;

pub(crate) struct FrameEvidence {
    pub(crate) packet: OwnedRoiEvidence,
    pub(crate) pose: EyePoseInput,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conic_solver::joint::ProjectedCircle;
    use crate::geometry::{normalized3,sub3};
    use crate::outline_conic_segments::sparse_evidence::{OwnedBoundaryArc,OwnedConicHint};
    use crate::roi_evidence::{BoundaryKind,RoiId};

    fn camera()->PinholeCamera {PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]}}
    fn frame(eye:usize,time:u64)->FrameEvidence {
        let center=[if eye==0 {-32.0} else {32.0},0.0,-350.0];
        let origin=if eye==0 {[3420,2860]} else {[4100,2860]};
        let normal=normalized3(sub3([70.0,-130.0,250.0],center)).unwrap();
        let e=ProjectedCircle::project(camera(),center,normal,6.0,origin).unwrap().ellipse().unwrap();
        FrameEvidence {packet:OwnedRoiEvidence {
            exposure:ExposureKey {roi:RoiId(eye as u32+1),clock:SourceClock {domain:1,epoch:5},
                sequence:time/100_000_000+eye as u64*1000,timestamp_ns:time},
            sensor_origin_px:origin,dimensions_px:[420,280],detail_reliability:Some(1.0),
            arcs:vec![OwnedBoundaryArc {evidence_group:0,kind:BoundaryKind::OuterLimbus,
                points_roi_px:e.dense_points(32),outward_normals_roi:None,normal_band_half_width_px:1.0,detector_score:None}],
            conics:vec![OwnedConicHint {kind:BoundaryKind::OuterLimbus,ellipse_roi_px:e,supporting_arc_indices:vec![0]}]},
            pose:EyePoseInput {limbus_center_sensor_px:camera().project(center).unwrap(),pixels_per_10mm:Some([4000.0/35.0,100.0,130.0])}}
    }

    #[test]
    fn late_unique_partner_cannot_resurrect_geometry_cleared_by_a_newer_empty_source() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        assert!(tracker.observe(frame(0,time),camera()).unwrap().is_some());
        let mut missing=frame(0,time+100_000_000);missing.packet.arcs.clear();missing.packet.conics.clear();
        assert!(tracker.observe(missing,camera()).is_err());
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none());
        let late=tracker.observe(frame(1,time),camera()).unwrap().unwrap();
        assert_eq!(late.exposures[0].unwrap().timestamp_ns,time,"the old pair is legitimate historical output");
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none(),
            "historical completion must not restore a newer rejected eye's live geometry");
        assert!(tracker.latest(1,clock,time+100_000_000,500_000_000).is_some());
        tracker.observe(frame(0,time+200_000_000),camera()).unwrap();
        assert_eq!(tracker.latest(0,clock,time+200_000_000,500_000_000).unwrap().exposures[0].unwrap().timestamp_ns,time+200_000_000);
    }

    #[test]
    fn fresh_invalid_scene_support_clears_previous_geometry_as_a_conic_failure_does() {
        let time=1_000_000_000;
        let clock=frame(0,time).packet.exposure.clock;
        let mut tracker=JointTracker::default();tracker.begin(clock,1);
        tracker.observe(frame(0,time),camera()).unwrap();
        let mut invalid=frame(0,time+100_000_000);invalid.pose.limbus_center_sensor_px=[f64::NAN;2];
        assert!(matches!(tracker.observe(invalid,camera()),Err(TrackingUnavailable::NoSceneSupport)));
        assert!(tracker.latest(0,clock,time+100_000_000,500_000_000).is_none());
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PublishedJoint {
    pub(crate) exposures: [Option<ExposureKey>; 2],
    pub(crate) sensor_origins_px: [Option<[u32;2]>; 2],
    pub(crate) dimensions_px: [Option<[u32;2]>; 2],
    pub(crate) scene: CoarseBinocularScene,
    pub(crate) solution: JointConicSolution,
    pub(crate) source_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrackingUnavailable {
    Pairing(PairingUnavailable),
    NoSceneSupport,
    Conic(JointConicUnavailable),
}

#[derive(Default)]
pub(crate) struct JointTracker {
    lineage: Option<(SourceClock,u64)>,
    pairing: SourcePairer<FrameEvidence>,
    latest: [Option<Arc<PublishedJoint>>;2],
    /// Rejection is an observation too. Clearing an invalid fit must not erase
    /// this time floor and let a delayed historical pair resurrect old gaze.
    newest_observation_ns: [Option<u64>;2],
}

impl JointTracker {
    /// `generation` changes with provider/prompt configuration, not crop
    /// translation. The caller, not an arriving asynchronous proposal, owns it.
    pub(crate) fn begin(&mut self, clock:SourceClock, generation:u64) {
        if self.lineage == Some((clock,generation)) {return;}
        self.lineage=Some((clock,generation));self.pairing.begin(clock);self.latest=[None,None];
        self.newest_observation_ns=[None;2];
    }

    fn clear_current_sources(&mut self,sources:[Option<ExposureKey>;2]) {
        for eye in 0..2 {
            if sources[eye].is_some_and(|source|self.newest_observation_ns[eye]
                .is_none_or(|time|source.timestamp_ns>=time)) {self.latest[eye]=None;}
        }
    }

    pub(crate) fn observe(&mut self, frame:FrameEvidence, camera:PinholeCamera)
        ->Result<Option<Arc<PublishedJoint>>,TrackingUnavailable> {
        let source=frame.packet.exposure;
        let frames=match self.pairing.insert(source,frame) {
            Ok(frames)=>frames,
            Err(PairingUnavailable::DuplicateSource)=>return Ok(None),
            Err(e)=>return Err(TrackingUnavailable::Pairing(e)),
        };
        let seen=&mut self.newest_observation_ns[source.roi.0 as usize-1];
        *seen=Some(seen.unwrap_or(0).max(source.timestamp_ns));
        let exposures=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.exposure));
        let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
        let Some(mut scene)=approximate_scene(camera,poses) else {
            self.clear_current_sources(exposures);
            return Err(TrackingUnavailable::NoSceneSupport);
        };
        scene.prior.target_seed_camera_mm=self.latest.iter().flatten()
            .filter(|s|s.exposures.iter().flatten().all(|k|k.timestamp_ns<source.timestamp_ns))
            .max_by_key(|s|s.exposures.iter().flatten().map(|k|k.timestamp_ns).max())
            .map(|s|s.solution.target_camera_mm);
        let prepared=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.prepare()));
        let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
        let result=solve_joint_conics(JointConicRequest {eyes:evidence,scene:&scene.prior,
            maximum_hypotheses:16,maximum_refinements:12,maximum_source_skew_ns:0,
            // Engineering allowance, not a measured bound on rolling rows.
            exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0});
        let solution=match result {
            Ok(solution)=>solution,
            Err(error)=>{
                self.clear_current_sources(exposures);
                return Err(TrackingUnavailable::Conic(error));
            }
        };
        let publication=Arc::new(PublishedJoint {solution,scene,
            exposures,
            sensor_origins_px:frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.sensor_origin_px)),
            dimensions_px:frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.dimensions_px)),
            source_generation:self.lineage.map_or(0,|(_,g)|g)});
        for eye in 0..2 {if publication.exposures[eye].is_some_and(|key|self.newest_observation_ns[eye]
            .is_none_or(|time|key.timestamp_ns>=time)) {
            self.latest[eye]=Some(Arc::clone(&publication));
        }}
        Ok(Some(publication))
    }

    pub(crate) fn latest(&self, eye:usize, clock:SourceClock, current_timestamp_ns:u64, maximum_age_ns:u64)
        ->Option<Arc<PublishedJoint>> {
        let result=self.latest.get(eye)?.as_ref()?;
        let source=result.exposures[eye]?;
        (source.clock==clock && source.timestamp_ns<=current_timestamp_ns
            && self.newest_observation_ns[eye].is_none_or(|time|source.timestamp_ns>=time)
            && current_timestamp_ns-source.timestamp_ns<=maximum_age_ns).then(||Arc::clone(result))
    }
}
