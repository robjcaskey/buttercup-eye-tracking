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
}

impl JointTracker {
    /// `generation` changes with provider/prompt configuration, not crop
    /// translation. The caller, not an arriving asynchronous proposal, owns it.
    pub(crate) fn begin(&mut self, clock:SourceClock, generation:u64) {
        if self.lineage == Some((clock,generation)) {return;}
        self.lineage=Some((clock,generation));self.pairing.begin(clock);self.latest=[None,None];
    }

    pub(crate) fn observe(&mut self, frame:FrameEvidence, camera:PinholeCamera)
        ->Result<Option<Arc<PublishedJoint>>,TrackingUnavailable> {
        let source=frame.packet.exposure;
        let frames=match self.pairing.insert(source,frame) {
            Ok(frames)=>frames,
            Err(PairingUnavailable::DuplicateSource)=>return Ok(None),
            Err(e)=>return Err(TrackingUnavailable::Pairing(e)),
        };
        let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
        let mut scene=approximate_scene(camera,poses).ok_or(TrackingUnavailable::NoSceneSupport)?;
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
                for eye in 0..2 {if frames[eye].is_some() && self.latest[eye].as_ref()
                    .is_none_or(|p|p.exposures[eye].is_none_or(|k|k.timestamp_ns<=source.timestamp_ns)) {self.latest[eye]=None;}}
                return Err(TrackingUnavailable::Conic(error));
            }
        };
        let publication=Arc::new(PublishedJoint {solution,scene,
            exposures:frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.exposure)),
            sensor_origins_px:frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.sensor_origin_px)),
            dimensions_px:frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.dimensions_px)),
            source_generation:self.lineage.map_or(0,|(_,g)|g)});
        for eye in 0..2 {if publication.exposures[eye].is_some() && self.latest[eye].as_ref()
            .is_none_or(|p|p.exposures[eye].is_none_or(|k|k.timestamp_ns<=source.timestamp_ns)) {
            self.latest[eye]=Some(Arc::clone(&publication));
        }}
        Ok(Some(publication))
    }

    pub(crate) fn latest(&self, eye:usize, clock:SourceClock, current_timestamp_ns:u64, maximum_age_ns:u64)
        ->Option<Arc<PublishedJoint>> {
        let result=self.latest.get(eye)?.as_ref()?;
        let source=result.exposures[eye]?;
        (source.clock==clock && source.timestamp_ns<=current_timestamp_ns
            && current_timestamp_ns-source.timestamp_ns<=maximum_age_ns).then(||Arc::clone(result))
    }
}
