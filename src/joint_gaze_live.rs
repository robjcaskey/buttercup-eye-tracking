//! Viewer adapter for source-aligned mixed-ROI conic solving. The geometry
//! engine knows nothing about windows, mouse calibration, or rendered pixels.

use crate::conic_solver::joint::PinholeCamera;
use crate::eye_scene_model::{quantize_frontal_disk_area, SurfaceGazeSample, SurfaceSignDiagnostics, SurfaceSignEvidence};
use crate::eye_scene_model::binocular_pose::EyePoseInput;
use crate::gaze_target_solver::joint_tracking::{FrameEvidence, JointTracker, PublishedJoint};
use crate::outline_conic_segments::sparse_evidence::{append_raw_ring_arcs, append_retained_sam_arcs, OwnedRoiEvidence, RawArcConfig};
use crate::roi_evidence::{BoundaryKind, ExposureKey, RoiId, SourceClock};
use crate::{EyeFrame, RelativeGazeVector, SegmentationMode, SAM31_RESULT_MAX_AGE_NS};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Default)]
pub(crate) struct Bridge {
    enabled: bool,
    signature: Option<(SourceClock, [u64;2], u64)>,
    generation: u64,
    submitted: [Option<ExposureKey>;2],
    last_status: [Option<String>;2],
    tracker: JointTracker,
}

fn clock(epoch:&str)->SourceClock {
    SourceClock {domain:1,epoch:epoch.bytes().fold(14695981039346656037,|h,b|(h^b as u64).wrapping_mul(1099511628211))}
}

impl Bridge {
    pub(crate) fn set_enabled(&mut self, enabled:bool, authority_generations:&mut [u64;2]) {
        if self.enabled==enabled {return;}
        self.enabled=enabled;self.signature=None;self.submitted=[None,None];self.last_status=[None,None];self.tracker=JointTracker::default();
        // A monocular-surface calibration is not silently reused for a
        // different observation model, even though both are SAM providers.
        for generation in authority_generations {*generation=generation.wrapping_add(1);}
    }

    /// Consume every admitted completion, not just the latest display slot
    /// for whichever ROI happens to receive the next RAW callback.
    pub(crate) fn observe_proposal(&mut self, proposal:&crate::sam31_outer::ProposalMasks,
        stream_epoch:&str, authority_generations:[u64;2], pixels_per_10mm:Option<[f64;3]>,
        hub:&crate::recording_trace::Hub) {
        if !self.enabled || proposal.eye_index>=2 {return;}
        let current_clock=clock(stream_epoch);
        let signature=(current_clock,authority_generations,proposal.prompt_generation);
        if self.signature!=Some(signature) {
            self.signature=Some(signature);self.generation=self.generation.wrapping_add(1);
            self.submitted=[None,None];self.last_status=[None,None];self.tracker.begin(current_clock,self.generation);
        }
        let eye=proposal.eye_index;
        let source_frame=(|| {
            if proposal.source_width.checked_mul(proposal.source_height)!=Some(proposal.source_raw.len()) {return None;}
            let reference=hub.source_reference(eye as u32+1,Some(proposal.source_timestamp_ns));
            let key=&reference["key"];
            if reference["status"]!="exact-source-key" || key["stream_epoch"].as_str()!=Some(stream_epoch)
                || key["sequence"].as_str()?.parse::<u64>().ok()?!=proposal.source_sequence {return None;}
            let exposure=ExposureKey {roi:RoiId(eye as u32+1),clock:current_clock,
                sequence:proposal.source_sequence,timestamp_ns:proposal.source_timestamp_ns};
            if self.submitted[eye].is_some_and(|old|old.timestamp_ns>=exposure.timestamp_ns) {return None;}
            let mut packet=OwnedRoiEvidence {exposure,
                sensor_origin_px:[proposal.source_sensor_origin.0,proposal.source_sensor_origin.1],
                dimensions_px:[proposal.source_width as u32,proposal.source_height as u32],
                arcs:Vec::new(),conics:Vec::new(),detail_reliability:None};
            if let Some(review)=&proposal.outer_fit {
                append_retained_sam_arcs(&mut packet,review,0);
                if !crate::sam31_outer::proposal_raw_outer_admitted(proposal) {
                    for arc in &mut packet.arcs {arc.normal_band_half_width_px=5.0;}
                }
            }
            if let Some((pupil,config))=proposal.inner_pupil_fit.zip(proposal.outer_fit.as_ref()
                .and_then(|outer|RawArcConfig::for_pupil(&proposal.source_raw,
                    proposal.source_width,proposal.source_height,outer.ellipse))) {
                append_raw_ring_arcs(&mut packet,&proposal.source_raw,pupil.ellipse,
                    BoundaryKind::PupillaryBoundary,100,config);
            }
            let center=proposal.outer_fit.as_ref().map(|r|[r.ellipse.center.0,r.ellipse.center.1])
                .unwrap_or([proposal.source_width as f64*0.5,proposal.source_height as f64*0.5]);
            let pose=EyePoseInput {limbus_center_sensor_px:[center[0]+packet.sensor_origin_px[0] as f64,
                center[1]+packet.sensor_origin_px[1] as f64],
                pixels_per_10mm};
            Some(FrameEvidence {packet,pose})
        })();
        if let Some(evidence)=source_frame {
            self.submitted[eye]=Some(evidence.packet.exposure);
            // Uncalibrated, explicit native-sensor pinhole engineering prior.
            // Do not label checkerboard intrinsics as used until supplied here.
            let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
            self.last_status[eye]=self.tracker.observe(evidence,camera).err().map(|e|format!("{e:?}"));
        }
    }

    pub(crate) fn update(&mut self, frame:&mut EyeFrame, stream_epoch:&str,
        authority_generations:[u64;2], hub:&crate::recording_trace::Hub) {
        frame.joint_gaze_active=self.enabled && frame.segmentation_mode.uses_mask_geometry();
        if !frame.joint_gaze_active {return;}
        if let Some(proposal)=frame.sam31_proposal_masks.as_ref().filter(|p|
            p.eye_index==frame.eye_id.saturating_sub(1) as usize
                && Some(p.prompt_generation)==frame.gaze_authority_sam_prompt_generation) {
            self.observe_proposal(proposal,stream_epoch,authority_generations,
                frame.centimeter_scale.map(|s|[s.estimate_px,s.minimum_px,s.maximum_px]),hub);
        }
        frame.joint_conic_status=self.last_status.get(frame.eye_id.saturating_sub(1) as usize)
            .cloned().flatten().or_else(||Some("waiting-for-source-aligned-SAM-arcs".into()));
        self.install(frame,clock(stream_epoch));
    }

    pub(crate) fn install(&self, frame:&mut EyeFrame, current_clock:SourceClock) {
        if !frame.joint_gaze_active {return;}
        frame.joint_conic=self.tracker.latest(frame.eye_id.saturating_sub(1) as usize,current_clock,
            frame.timestamp_ns,SAM31_RESULT_MAX_AGE_NS);
        if frame.joint_conic.is_some() {frame.joint_conic_status=Some("conditional-shared-target".into());}
        frame.virtual_contact_surface_gaze=surface(frame,false);
    }

    pub(crate) fn refresh_partner(&self, frame:&mut EyeFrame, stream_epoch:&str) {
        if frame.recording_clock["source_key"]["stream_epoch"].as_str()!=Some(stream_epoch) {return;}
        self.install(frame,clock(stream_epoch));
    }
}

fn current(frame:&EyeFrame)->Option<(&PublishedJoint,usize)> {
    if !frame.joint_gaze_active {return None;}
    let eye=frame.eye_id.checked_sub(1).filter(|i|*i<2)? as usize;
    let publication=frame.joint_conic.as_deref()?;
    let source=publication.exposures[eye]?;
    (source.timestamp_ns<=frame.timestamp_ns && frame.timestamp_ns-source.timestamp_ns<=SAM31_RESULT_MAX_AGE_NS
        && publication.solution.contributing_eyes[eye]).then_some((publication,eye))
}

pub(crate) fn gaze(frame:&EyeFrame)->Option<RelativeGazeVector> {
    let (publication,eye)=current(frame)?;
    let direction=publication.solution.eye_gaze_directions[eye]?;
    RelativeGazeVector::from_projected(direction[0],direction[1])
}

pub(crate) fn ellipse(frame:&EyeFrame)->Option<crate::geometry::Ellipse> {
    let (publication,eye)=current(frame)?;
    if publication.dimensions_px[eye]!=Some([frame.width as u32,frame.height as u32]) {return None;}
    let mut ellipse=publication.solution.ellipses_roi_px[eye][0]?;
    let origin=publication.sensor_origins_px[eye]?;
    ellipse.center.0+=origin[0] as f64-frame.sensor_x as f64;
    ellipse.center.1+=origin[1] as f64-frame.sensor_y as f64;
    (ellipse.center.0+ellipse.major_radius>=0.0 && ellipse.center.1+ellipse.major_radius>=0.0
        && ellipse.center.0-ellipse.major_radius<frame.width as f64
        && ellipse.center.1-ellipse.major_radius<frame.height as f64).then_some(ellipse)
}

pub(crate) fn source_ellipse(frame:&EyeFrame)->Option<crate::geometry::Ellipse> {
    let (publication,eye)=current(frame)?;
    let proposal=frame.sam31_proposal_masks.as_ref()?;
    let exposure=publication.exposures[eye]?;
    (exposure.timestamp_ns==proposal.source_timestamp_ns && exposure.sequence==proposal.source_sequence
        && publication.sensor_origins_px[eye]==Some([proposal.source_sensor_origin.0,proposal.source_sensor_origin.1]))
        .then_some(publication.solution.ellipses_roi_px[eye][0]).flatten()
}

/// Calibration records one vote per physical exposure. An atomic two-ROI
/// request has a provisional first completion and a final paired solve; only
/// the latter may consume that vote. Missing/evicted, independently submitted
/// or legacy single-eye sources must not wait for a nonexistent partner.
pub(crate) fn calibration_source_group_complete(frame:&EyeFrame)->bool {
    if !frame.joint_gaze_active || !frame.segmentation_mode.uses_mask_geometry() {return true;}
    let Some(proposal)=frame.sam31_proposal_masks.as_deref() else {return false;};
    if proposal.source_group_roi_count!=2 {return true;}
    let Some((publication,eye))=current(frame) else {return false;};
    let Some(selected)=publication.exposures[eye] else {return false;};
    selected.sequence==proposal.source_sequence && selected.timestamp_ns==proposal.source_timestamp_ns
        && publication.exposures.iter().enumerate().all(|(index,exposure)|
            exposure.is_some_and(|exposure|exposure.roi==RoiId(index as u32+1)
                && exposure.clock==selected.clock && exposure.timestamp_ns==selected.timestamp_ns))
}

pub(crate) fn surface(frame:&EyeFrame, gaze_axis:bool)->Option<SurfaceGazeSample> {
    let (publication,eye)=current(frame)?;
    let solution=&publication.solution;
    let ellipse=ellipse(frame)?;
    let direction=if gaze_axis {solution.eye_gaze_directions[eye]?} else {solution.eye_normals[eye]?};
    let relative_gaze=RelativeGazeVector::from_projected(direction[0],direction[1])?;
    let area=std::f64::consts::PI*ellipse.major_radius.powi(2);
    let (area_bucket,representative_area)=quantize_frontal_disk_area(area)?;
    let radius=(representative_area/std::f64::consts::PI).sqrt();
    // An outer ellipse alone has mirrored circle-normal solutions. Do not
    // call a numerical basin margin independent sign evidence for that case.
    let noncoplanar=solution.arcs.iter().any(|a|a.used&&a.kind==BoundaryKind::PupillaryBoundary);
    let sign_resolved=(solution.contributing_eyes==[true,true]||noncoplanar)
        && solution.alternative_cost_margin.is_some_and(|margin|margin>2.0);
    Some(SurfaceGazeSample {source_timestamp_ns:Some(publication.exposures[eye]?.timestamp_ns),
        frontal_equivalent_disk_area_px2:area,area_bucket,quantized_frontal_disk_radius_px:radius,
        near_surface_point_sensor_px:(frame.sensor_x as f64+ellipse.center.0+relative_gaze.right*radius,
            frame.sensor_y as f64+ellipse.center.1+relative_gaze.down*radius),relative_gaze,sign_resolved,
        // The joint optimizer's cache generation can change because the
        // OTHER eye was evicted/readmitted. That is not a sign correction to
        // this eye's camera-frame coordinates. Bind calibration to this eye's
        // authority lineage; keep the joint evidence generation separately
        // in the recorded publication. This does not promote unsigned fits
        // or certify temporal continuity of the optimizer's chosen branch.
        sign_epoch:frame.gaze_authority_generation,kinematic_sign_correction:[false;2],
        sign_diagnostics:Some(SurfaceSignDiagnostics {evidence:if sign_resolved {SurfaceSignEvidence::JointConics} else {SurfaceSignEvidence::Unresolved},
            selected_branch:0,branch_residual_ema_px:[f64::NAN;2],source_motion_residual_px:None,
            temporal_margin_px:None,pending_anchor_votes:0,near_frontal_continuation:false})})
}

pub(crate) fn json(frame:&EyeFrame)->Value {
    let Some(publication)=frame.joint_conic.as_deref() else {
        return json!({"active":frame.joint_gaze_active,"status":frame.joint_conic_status,"target":null});
    };
    let s=&publication.solution;
    json!({"active":frame.joint_gaze_active,"status":frame.joint_conic_status,
        "source_group_roi_count":frame.sam31_proposal_masks.as_ref().map(|p|p.source_group_roi_count),
        "calibration_source_group_complete":calibration_source_group_complete(frame),
        "frame":"camera-optical-center-mm-v1","units":"mm","axes":["sensor-right","sensor-down","toward-camera"],
        "method":"one shared fixation fitted to raw boundary segments; not averaged gaze points",
        "target":s.target_camera_mm,"target_covariance":null,
        "target_search_chart":{"frame":"reference-to-camera-tangent-plane","slopes":s.target_viewpoint_slopes,
            "reference_camera_mm":s.target_reference_camera_mm,
            "axial_distance_mm":s.target_viewpoint_axial_distance_mm,"distance_axis":"reference-to-camera",
            "slope_limit":s.target_viewpoint_slope_limit,"active_bounds":s.target_viewpoint_bounds_active},
        "source_generation":publication.source_generation.to_string(),
        "scale_provenance":publication.scene.scale_provenance.map(|p|p.map(|p|format!("{p:?}"))),
        "intrinsics":{"focal_px":publication.scene.prior.camera.focal_px,"principal_px":publication.scene.prior.camera.principal_px,"provenance":"uncalibrated engineering prior"},
        "sources":publication.exposures.map(|e|e.map(|e|json!({"roi_id":e.roi.0,"clock_domain":e.clock.domain.to_string(),
            "clock_epoch":e.clock.epoch.to_string(),"sequence":e.sequence.to_string(),"sensor_timestamp_ns":e.timestamp_ns.to_string()}))),
        "eye_centers":s.eye_centers_camera_mm,"surface_normals":s.eye_normals,"gaze_directions":s.eye_gaze_directions,
        "contributing_eyes":s.contributing_eyes,"cost":s.robust_cost,"alternative_cost_margin":s.alternative_cost_margin,
        "modeled_eyes":s.modeled_eyes,"unlocalized_eye_cost":s.unlocalized_eye_cost,
        "hypotheses":s.hypotheses_evaluated,"hypotheses_by_association":s.hypotheses_by_association,
        "arcs":s.arcs.iter().map(|a|json!({"roi_id":a.exposure.roi.0,"group":a.evidence_group,"kind":format!("{:?}",a.kind),
            "used":a.used,"rms_px":a.rms_px,"sigma_px":a.sigma_px,
            "boundary_normal_samples":a.boundary_normal_samples,"boundary_normal_rms_radians":a.boundary_normal_rms_radians,
            "support_length_px":a.support_length_px,"evidence_weight":a.evidence_weight})).collect::<Vec<_>>(),
        "uncertainty":"conditional engineering supports; not calibrated probabilities or measured anatomical pose",
        "transform_to_legacy_monitor_frame":null})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::{normalized3,sub3};
    use crate::conic_solver::joint::ProjectedCircle;
    use crate::outline_conic_segments::sparse_evidence::{OwnedBoundaryArc,OwnedConicHint};

    fn evidence(eye:usize,time:u64)->FrameEvidence {
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let center=[if eye==0 {-32.0} else {32.0},0.0,-350.0];
        let normal=normalized3(sub3([70.0,-150.0,250.0],center)).unwrap();
        let origin=if eye==0 {[3424,2860]} else {[4160,2860]};
        let mut packet=OwnedRoiEvidence {exposure:ExposureKey {roi:RoiId(eye as u32+1),clock:clock("test:1"),
            sequence:if eye==0 {100} else {900},timestamp_ns:time},sensor_origin_px:origin,dimensions_px:[420,280],
            arcs:Vec::new(),conics:Vec::new(),detail_reliability:Some(1.0)};
        for (kind,radius,depth,group) in [(BoundaryKind::OuterLimbus,6.0,0.0,0),(BoundaryKind::PupillaryBoundary,2.4,0.6,10)] {
            let center=std::array::from_fn(|i|center[i]-depth*normal[i]);
            let ellipse=ProjectedCircle::project(camera,center,normal,radius,origin).unwrap().ellipse().unwrap();
            packet.arcs.push(OwnedBoundaryArc {evidence_group:group,kind,points_roi_px:ellipse.dense_points(32),
                outward_normals_roi:None,normal_band_half_width_px:0.0,detector_score:None});
            packet.conics.push(OwnedConicHint {kind,ellipse_roi_px:ellipse,supporting_arc_indices:vec![packet.arcs.len()-1]});
        }
        FrameEvidence {packet,pose:EyePoseInput {limbus_center_sensor_px:camera.project(center).unwrap(),
            pixels_per_10mm:Some([4000.0/35.0,100.0,130.0])}}
    }

    fn proposal(eye:usize,time:u64)->crate::sam31_outer::ProposalMasks {
        let packet=evidence(eye,time).packet;
        let ellipse=packet.conics[0].ellipse_roi_px;
        crate::sam31_outer::ProposalMasks {
            eye_index:eye,source_sequence:packet.exposure.sequence,source_timestamp_ns:time,
            source_sensor_origin:(packet.sensor_origin_px[0],packet.sensor_origin_px[1]),
            source_width:420,source_height:280,source_raw:Arc::new(vec![100;420*280]),
            outer_fit:Some(crate::sam31_outer::OuterMaskFitReview {
                ellipse,source_component_area_px:0.0,
                retained_points:Arc::new(packet.arcs[0].points_roi_px.clone()),
                conic_segments:Arc::new(vec![(0..32).collect()]),
                flat_tire_points:Arc::new(Vec::new()),upper_flat_tire:false,lower_flat_tire:false,
            }),..Default::default()
        }
    }

    #[test]
    fn completed_proposals_are_consumed_without_any_display_frame_callback() {
        let time=1_000_000_000;
        let hub=crate::recording_trace::Hub::default();
        let mut bridge=Bridge::default();let mut generations=[0;2];
        bridge.set_enabled(true,&mut generations);
        for eye in 0..2 {
            let proposal=proposal(eye,time);
            hub.raw_arrived(json!({"roi_id":eye+1,"sensor_timestamp_ns":time.to_string(),
                "sequence":proposal.source_sequence.to_string(),"stream_epoch":"test:1"}),
                std::time::Instant::now(),0);
            bridge.observe_proposal(&proposal,"test:1",generations,Some([114.0,100.0,130.0]),&hub);
        }
        let publication=bridge.tracker.latest(0,clock("test:1"),time,1).unwrap();
        assert!(publication.exposures.iter().all(|e|e.is_some_and(|e|e.timestamp_ns==time)));
        assert_eq!(publication.exposures[0].unwrap().sequence,100);
        assert_eq!(publication.exposures[1].unwrap().sequence,900);
        // Neither stale display callbacks nor missing source attestations may
        // turn held geometry into another observation or replace this pair.
        bridge.observe_proposal(&proposal(0,time),"test:1",generations,None,&hub);
        bridge.observe_proposal(&proposal(0,time+1),"test:1",generations,None,&hub);
        assert!(Arc::ptr_eq(&publication,&bridge.tracker.latest(0,clock("test:1"),time,1).unwrap()));
    }

    #[test]
    fn joint_evidence_generation_is_not_a_per_eye_sign_correction() {
        let time=1_000_000_000;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),15);
        tracker.observe(evidence(0,time),camera).unwrap();
        let publication=tracker.observe(evidence(1,time),camera).unwrap().unwrap();
        let mut frame=crate::tests::control_eye_frame(1);
        frame.eye_id=2;frame.timestamp_ns=time;frame.width=420;frame.height=280;
        frame.sensor_x=4160;frame.sensor_y=2860;frame.segmentation_mode=SegmentationMode::Sam31;
        frame.joint_gaze_active=true;frame.joint_conic=Some(publication);
        frame.gaze_authority_generation=84;
        let before=surface(&frame,true).unwrap();

        // The failed 1788826811 session retained eye 2 authority 84 while
        // joint cache generations 15 -> 16 -> 17 threw away its targets.
        // Changing evidence lineage alone must not invent a sign correction.
        for generation in [16,17] {
            Arc::make_mut(frame.joint_conic.as_mut().unwrap()).source_generation=generation;
            let after=surface(&frame,true).unwrap();
            assert_eq!(after.sign_epoch,before.sign_epoch);
            assert_eq!(after.relative_gaze,before.relative_gaze);
            assert_eq!(after.sign_resolved,before.sign_resolved);
            assert_eq!(json(&frame)["source_generation"],generation.to_string());
        }
        frame.gaze_authority_generation+=1;
        assert_ne!(surface(&frame,true).unwrap().sign_epoch,before.sign_epoch,
            "the selected eye's own source lineage remains visible to calibration");
    }

    #[test]
    fn calibration_safe_frame_uses_the_joint_fit_not_the_legacy_review_ellipse() {
        let time=1_000_000_000;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),7);
        tracker.observe(evidence(0,time),camera).unwrap();
        let publication=tracker.observe(evidence(1,time),camera).unwrap().unwrap();
        let mut frame=crate::tests::control_eye_frame(1);
        frame.eye_id=1;frame.timestamp_ns=time;frame.width=420;frame.height=280;
        frame.sensor_x=3424;frame.sensor_y=2860;frame.segmentation_mode=SegmentationMode::Sam31;
        frame.joint_gaze_active=true;frame.joint_conic=Some(publication);
        frame.gaze_authority_sam_prompt_generation=Some(0);
        let mut proposal=proposal(0,time);
        proposal.outer_fit.as_mut().unwrap().ellipse.center.0=-100.0;
        frame.sam31_proposal_masks=Some(Arc::new(proposal));
        let mut sample=surface(&frame,false).unwrap();sample.sign_resolved=true;
        assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),crate::CalibrationFrameState::Ready);
        Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).outer_fit.as_mut().unwrap().ellipse.center.0=210.0;
        Arc::make_mut(frame.joint_conic.as_mut().unwrap()).solution.ellipses_roi_px[0][0].as_mut().unwrap().center.0=0.0;
        assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),crate::CalibrationFrameState::OutsideSafeFrame);
    }

    #[test]
    fn replay_uses_the_selected_query_not_an_unselected_first_mask() {
        let case=json!({"selected_query":7,"candidates":[
            {"query":3,"baseline_ellipse":{"center":[1,2]}},
            {"query":7,"baseline_ellipse":{"center":[9,8]}}
        ]});
        assert_eq!(replay_selected_candidate(&case).unwrap()["query"],7);
        assert!(replay_selected_candidate(&json!({"candidates":[]})).is_none());
        // Old exports omitted selected_query; only an actual complete ellipse
        // can supply the explicit fallback, not an arbitrary semantic mask.
        let case=json!({"candidates":[{"query":3},
            {"query":7,"baseline_ellipse":{"center":[9,8],"major_radius":20,"minor_radius":15,"angle":0}}]});
        assert_eq!(replay_selected_candidate(&case).unwrap()["query"],7);
    }

    #[test]
    fn calibration_waits_only_for_explicit_same_exposure_source_groups() {
        use crate::CalibrationFrameState::{Ready,WaitingForSourcePartner};
        use std::time::{Duration,Instant};
        let time=1_000_000_000;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        for eye in 0..2 {
            let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),7);
            let mono=tracker.observe(evidence(eye,time),camera).unwrap().unwrap();
            let mut frame=crate::tests::control_eye_frame(1);
            frame.eye_id=eye as u32+1;frame.timestamp_ns=time;frame.width=420;frame.height=280;
            let proposal=proposal(eye,time);
            frame.sensor_x=proposal.source_sensor_origin.0;frame.sensor_y=proposal.source_sensor_origin.1;
            frame.segmentation_mode=SegmentationMode::Sam31;frame.joint_gaze_active=true;
            frame.joint_conic=Some(mono);frame.sam31_proposal_masks=Some(Arc::new(proposal));
            frame.gaze_authority_sam_prompt_generation=Some(0);
            // Isolate the source-completion barrier from the independent sign
            // confidence gate; recorded-worker tests exercise natural signs.
            let mut sample=surface(&frame,true).unwrap();sample.sign_resolved=true;
            for count in [0,1] {
                Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).source_group_roi_count=count;
                assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),Ready);
            }
            Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).source_group_roi_count=2;
            assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),WaitingForSourcePartner);
            assert_eq!(crate::gaze_frame_state(Some(&frame),Some(sample)),Ready,
                "completed absolute placement need not delay provisional rendering");

            let start=Instant::now();let mut mode=crate::VirtualMouseMode::new(start);
            mode.target_source_started_ns=Some(time-600_000_000);
            mode.frame_state=WaitingForSourcePartner;
            mode.observe_at_frame(start+Duration::from_millis(600),Some(time),None);
            assert!(mode.samples.iter().all(Vec::is_empty));

            let paired=tracker.observe(evidence(1-eye,time),camera).unwrap().unwrap();
            frame.joint_conic=Some(paired);
            assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),Ready);
            mode.frame_state=Ready;
            for elapsed in [700,750] {
                mode.observe_at_frame(start+Duration::from_millis(elapsed),Some(time),
                    Some((time,sample.relative_gaze.projected(),sample.sign_epoch)));
            }
            assert_eq!(mode.samples[0].len(),1,"paired source is exactly one calibration vote");

            // A completed but unusable partner is still a completed request.
            // It need not contribute geometry for a strong single-eye solve.
            Arc::make_mut(frame.joint_conic.as_mut().unwrap()).solution.contributing_eyes[1-eye]=false;
            assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),Ready);
            let good=frame.joint_conic.clone();
            for stale_clock in [false,true] {
                frame.joint_conic=good.clone();
                let partner=Arc::make_mut(frame.joint_conic.as_mut().unwrap()).exposures[1-eye].as_mut().unwrap();
                if stale_clock {partner.clock.epoch+=1;} else {partner.timestamp_ns-=1;}
                assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),WaitingForSourcePartner);
            }
        }
    }

    /// Legacy offered-load exports used atomic dual-ROI ingress and refused
    /// incomplete active-mask=3 groups. No such guarantee exists for older
    /// completion-paced or single-image exports. Keep this inference visible.
    fn replay_source_group(case:&Value)->(u8,&'static str) {
        if let Some(count)=case["source_group_roi_count"].as_u64().filter(|count|*count<=2) {
            return (count as u8,"proposal-metadata");
        }
        if case["replay"]["method"]=="live-video-worker-offered-load"
            && case["input"]["frame"]["region"]["active_mask"]==3 {
            (2,"legacy-atomic-offered-replay-contract")
        } else {(0,"unavailable")}
    }

    #[test]
    fn replay_does_not_invent_atomic_requests_from_active_roi_count() {
        let mut case=json!({"input":{"frame":{"region":{"active_mask":3}}}});
        assert_eq!(replay_source_group(&case),(0,"unavailable"));
        case["replay"]=json!({"method":"live-video-worker-offered-load"});
        assert_eq!(replay_source_group(&case),(2,"legacy-atomic-offered-replay-contract"));
        case["source_group_roi_count"]=json!(1);
        assert_eq!(replay_source_group(&case),(1,"proposal-metadata"));
    }

    fn replay_selected_candidate(case:&Value)->Option<&Value> {
        let candidates=case["candidates"].as_array()?;
        case["selected_query"].as_u64().and_then(|query|
            candidates.iter().find(|candidate|candidate["query"].as_u64()==Some(query)))
            .or_else(||candidates.iter().find(|candidate| {
                let e=&candidate["baseline_ellipse"];
                e["center"][0].as_f64().is_some() && e["center"][1].as_f64().is_some()
                    && ["major_radius","minor_radius","angle"].into_iter().all(|key|e[key].as_f64().is_some())
            }))
    }

    #[test]
    #[ignore = "requires fresh live-worker replay cache and native RAW corpus under outputs"]
    fn recorded_calibration_acquires_from_source_timed_live_worker_evidence() {
        use std::io::{BufRead,Read,Seek,Write};
        use std::time::{Duration,Instant};
        let input=std::env::var("BUTTERCUP_JOINT_CALIBRATION_CACHE").unwrap();
        let output=std::env::var("BUTTERCUP_JOINT_CALIBRATION_REPORT").unwrap();
        let require_ready=std::env::var("BUTTERCUP_JOINT_CALIBRATION_REQUIRE_READY").as_deref()==Ok("1");
        let ignore_source_group=std::env::var("BUTTERCUP_JOINT_CALIBRATION_IGNORE_SOURCE_GROUP").as_deref()==Ok("1");
        let calibration_eye=std::env::var("BUTTERCUP_JOINT_CALIBRATION_EYE").map(|v|v.parse::<usize>().unwrap()).unwrap_or(0);
        assert!(calibration_eye<2,"calibration eye must be zero-based 0 or 1");
        let mut writer=std::fs::OpenOptions::new().create_new(true).write(true).open(output).unwrap();
        let mut bridge=Bridge::default();let mut generations=[0;2];bridge.set_enabled(true,&mut generations);
        let hub=crate::recording_trace::Hub::default();
        let mut frames:[Option<EyeFrame>;2]=[None,None];
        let start=Instant::now();let mut first_time=None;let mut acquired_at=None;
        let mut acquisition=crate::calibration_acquisition::Acquisition::default();acquisition.start(start);
        let mut calibration=crate::VirtualMouseMode::new(start);
        let ellipse=|v:&Value|->Option<crate::geometry::Ellipse> {
            Some(crate::geometry::Ellipse {center:(v["center"][0].as_f64()?,v["center"][1].as_f64()?),
                major_radius:v["major_radius"].as_f64()?,minor_radius:v["minor_radius"].as_f64()?,angle:v["angle"].as_f64()?})
        };
        let mut count=0;
        for line in std::io::BufReader::new(std::fs::File::open(input).unwrap()).lines() {
            let case:Value=serde_json::from_str(&line.unwrap()).unwrap();let row=&case["input"];let meta=&row["frame"];
            let n=|key|meta[key].as_u64().unwrap();let eye=n("eye_id") as usize-1;
            let time=n("timestamp_ns");let sequence=n("sequence");
            assert_eq!(case["timestamp_ns"].as_u64(),Some(time));
            assert_eq!(case["sequence"].as_u64(),Some(sequence));
            let source_start=case["replay"]["first_source_ns"].as_str().map(|s|s.parse::<u64>().unwrap()).unwrap_or(time);
            let elapsed_ns=case["replay"]["ready_elapsed_ns"].as_str().map(|s|s.parse::<u64>().unwrap())
                .unwrap_or_else(||time-*first_time.get_or_insert(source_start));
            let now=start+Duration::from_nanos(elapsed_ns);
            let epoch=meta["source_clock"]["source_key"]["stream_epoch"].as_str().unwrap();
            let mut raw=std::fs::File::open(row["raw_file"].as_str().unwrap()).unwrap();
            raw.seek(std::io::SeekFrom::Start(row["raw_offset"].as_u64().unwrap())).unwrap();
            let mut packed=vec![0;row["raw_length"].as_u64().unwrap() as usize];raw.read_exact(&mut packed).unwrap();
            let pixels=crate::raw10::try_unpack_raw10(&packed,n("width") as usize,n("height") as usize,n("stride") as usize).unwrap();
            let candidate=replay_selected_candidate(&case);
            let (source_group_roi_count,source_group_metadata)=replay_source_group(&case);
            let source_group_roi_count=if ignore_source_group {0}else{source_group_roi_count};
            let proposal=crate::sam31_outer::ProposalMasks {
                eye_index:eye,source_sequence:sequence,source_timestamp_ns:time,
                source_group_roi_count,
                source_sensor_origin:(n("sensor_x") as u32,n("sensor_y") as u32),
                source_width:n("width") as usize,source_height:n("height") as usize,source_raw:Arc::new(pixels),
                outer_fit:candidate.and_then(|c|Some(crate::sam31_outer::OuterMaskFitReview {
                    ellipse:ellipse(&c["baseline_ellipse"] )?,source_component_area_px:0.0,
                    retained_points:Arc::new(serde_json::from_value(c["baseline_retained"].clone()).unwrap()),
                    conic_segments:Arc::new(serde_json::from_value(c["baseline_retained_segments"].clone()).unwrap()),
                    flat_tire_points:Arc::new(serde_json::from_value(c["baseline_censored"].clone()).unwrap()),
                    upper_flat_tire:false,lower_flat_tire:false,
                })),
                // The bridge extracts new RAW boundary evidence at this
                // measured pupil fit; it does not consume the support scalar.
                inner_pupil_fit:ellipse(&case["pupil_void"]["ellipse"]).map(|ellipse|
                    crate::sam31_outer::PupilVoidFitReview {ellipse,raw_support:Default::default()}),
                ..Default::default()
            };
            let scale=&row["scale_hint"];
            let scale=scale["pixels_per_10mm"].as_f64().and_then(|estimate|Some([estimate,
                scale["bounds_px_per_10mm"][0].as_f64()?,scale["bounds_px_per_10mm"][1].as_f64()?]));
            let stamp=hub.raw_arrived(meta["source_clock"]["source_key"].clone(),now,n("host_arrival_unix_ns"));
            bridge.observe_proposal(&proposal,epoch,generations,scale,&hub);
            let mut frame=crate::tests::control_eye_frame(sequence);
            frame.eye_id=eye as u32+1;frame.timestamp_ns=time;frame.width=proposal.source_width;frame.height=proposal.source_height;
            frame.sensor_x=proposal.source_sensor_origin.0;frame.sensor_y=proposal.source_sensor_origin.1;
            frame.segmentation_mode=SegmentationMode::Sam31;frame.recording_clock=stamp;
            frame.sam31_proposal_masks=Some(Arc::new(proposal));frame.gaze_authority_sam_prompt_generation=Some(0);
            bridge.update(&mut frame,epoch,generations,&hub);frames[eye]=Some(frame);
            for (eye,frame) in frames.iter_mut().enumerate() {
                if let Some(frame)=frame {
                    if let Some(meta)=case["replay"]["presentation_inputs"][eye].get("frame") {
                        frame.timestamp_ns=meta["timestamp_ns"].as_u64().unwrap();
                        frame.sensor_x=meta["sensor_x"].as_u64().unwrap() as u32;
                        frame.sensor_y=meta["sensor_y"].as_u64().unwrap() as u32;
                        frame.width=meta["width"].as_u64().unwrap() as usize;
                        frame.height=meta["height"].as_u64().unwrap() as usize;
                        frame.recording_clock=meta["source_clock"].clone();
                    }
                    bridge.refresh_partner(frame,epoch);
                }
            }
            // Use the same axis adapter as live mouse calibration. The
            // contact's optical normal can differ from the joint gaze axis.
            let focus=frames[calibration_eye].as_ref();let surface=focus.and_then(crate::mouse_gaze_surface);
            let state=crate::calibration_frame_state(focus,surface);
            let qualified=surface.filter(|_|state==crate::CalibrationFrameState::Ready).and_then(|s|
                Some(crate::calibration_acquisition::QualifiedSign {source_ns:s.source_timestamp_ns?,epoch:s.sign_epoch,
                    sustained_support:s.sign_diagnostics.is_some_and(|d|d.evidence.sustained_acquisition_support())}));
            let result=acquisition.observe(now,1,qualified);
            if result==crate::calibration_acquisition::Update::Ready {acquired_at.get_or_insert(elapsed_ns);}
            // Exercise the enclosing UI phase machine too. Testing only the
            // acquisition helper missed re-entry after its first success.
            let current_time=focus.map(|f|f.timestamp_ns);
            calibration.acquisition_source_generation=1;
            calibration.observe_frame_state(current_time,state);
            calibration.surface_gaze=surface;
            calibration.observe_at_frame(now,current_time,qualified.zip(surface).map(|(q,s)|
                (q.source_ns,s.relative_gaze.projected(),q.epoch)));
            let report=json!({"input_index":row["index"],"source_ns":time.to_string(),"elapsed_ms":elapsed_ns/1_000_000,
                "calibration_eye":calibration_eye,"selected_query":candidate.map(|c|&c["query"]),
                "source_group_roi_count":source_group_roi_count,"source_group_metadata":source_group_metadata,
                "source_group_ignored_for_control":ignore_source_group,
                "state":format!("{state:?}"),"acquisition":format!("{result:?}"),"votes":acquisition.ready_sources,
                "qualified_source_ns":qualified.map(|s|s.source_ns.to_string()),
                "joint":focus.map(super::json),"source_ellipse":focus.and_then(source_ellipse).map(|e|
                    [e.center.0,e.center.1,e.major_radius,e.minor_radius,e.angle]),
                "timing_validation":case["replay"]["timing_validation"],
                "ui_target_index":calibration.target_index,"ui_acquisition_episodes":calibration.sign_acquisition.episodes,
                "ui_sign_restarts":calibration.calibration_sign_restarts,
                "ui_unique_surface_updates":calibration.unique_surface_updates,
                "ui_target_sample_counts":calibration.samples.iter().map(Vec::len).collect::<Vec<_>>(),
                "method":"live-worker cache through actual bridge and acquisition; no monitor-accuracy validation"});
            serde_json::to_writer(&mut writer,&report).unwrap();writeln!(writer).unwrap();count+=1;
        }
        eprintln!("RECORDED_CALIBRATION frames={count} acquired_ms={:?}",acquired_at.map(|ns|ns/1_000_000));
        assert!(calibration.sign_acquisition.episodes<=1,"a fixed-lineage recording must not repeatedly re-enter acquisition");
        assert_eq!(calibration.calibration_sign_restarts,0,"unsigned frames cannot invalidate the stationary sequence");
        if require_ready {assert!(acquired_at.is_some(),"last calibration failed to acquire; inspect the source-timed report");}
    }

    #[test]
    fn fresh_empty_roi_clears_old_geometry_and_pairs_as_missing_not_as_a_held_eye() {
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let time=1_000_000_000;
        let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),7);
        tracker.observe(evidence(0,time),camera).unwrap();
        tracker.observe(evidence(1,time),camera).unwrap();
        let now=time+100_000_000;
        let mut missing=evidence(0,now);missing.packet.exposure.sequence+=1;
        missing.packet.arcs.clear();missing.packet.conics.clear();
        assert!(tracker.observe(missing,camera).is_err());
        assert!(tracker.latest(0,clock("test:1"),now,500_000_000).is_none());
        let mut other=evidence(1,now);other.packet.exposure.sequence+=1;
        let current=tracker.observe(other,camera).unwrap().unwrap();
        assert_eq!(current.solution.contributing_eyes,[false,true]);
        assert!(current.solution.ellipses_roi_px[0][0].is_none());
        assert!(current.exposures.into_iter().flatten().all(|e|e.timestamp_ns==now));
        // Replayed old output cannot replace the fresh missing-eye state.
        assert!(tracker.observe(evidence(0,time),camera).unwrap().is_none());
        let latest=tracker.latest(0,clock("test:1"),now,500_000_000).unwrap();
        assert_eq!(latest.exposures[0].unwrap().timestamp_ns,now);
        assert!(!latest.solution.contributing_eyes[0]);
    }

    #[test]
    fn live_publication_keeps_one_fixation_and_separate_surface_and_cursor_axes() {
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let time=1_000_000_000;
        let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),7);
        tracker.observe(evidence(0,time),camera).unwrap();
        let result=tracker.observe(evidence(1,time),camera).unwrap().unwrap();
        assert_eq!(result.solution.contributing_eyes,[true,true]);
        assert!(tracker.observe(evidence(1,time),camera).unwrap().is_none(),"held proposal must not re-solve");
        for eye in 0..2 {
            let mut frame=crate::tests::control_eye_frame(1);
            frame.eye_id=eye as u32+1;frame.timestamp_ns=time+50_000_000;
            frame.width=420;frame.height=280;frame.segmentation_mode=SegmentationMode::Sam31;
            let origin=result.sensor_origins_px[eye].unwrap();frame.sensor_x=origin[0];frame.sensor_y=origin[1];
            frame.joint_gaze_active=true;frame.joint_conic=Some(Arc::clone(&result));
            let contact=surface(&frame,false).unwrap();
            assert!((40.0..100.0).contains(&contact.quantized_frontal_disk_radius_px),"area must be converted back to radius");
            assert_eq!(contact.source_timestamp_ns,Some(time));
            let gaze=gaze(&frame).unwrap();
            assert_eq!(crate::mouse_gaze_surface(&frame).unwrap().relative_gaze,gaze);
            let exact=normalized3(sub3(result.solution.target_camera_mm,result.solution.eye_centers_camera_mm[eye].unwrap())).unwrap();
            assert!((gaze.right-exact[0]).abs()<1e-10&&(gaze.down-exact[1]).abs()<1e-10);
            frame.virtual_contact_surface_gaze=Some(contact);
            let pose=crate::virtual_contact_pose(&frame).expect("joint contact should render");
            assert_eq!(crate::pose_for_cursor(&frame,pose).unwrap().relative_gaze,gaze);
            let before=ellipse(&frame).unwrap();frame.sensor_y+=24;
            assert!((ellipse(&frame).unwrap().center.1-before.center.1+24.0).abs()<1e-10);
            frame.joint_conic=None;
            assert!(crate::mouse_gaze_surface(&frame).is_none(),"joint mode must not fall back to the old surface model");
        }
        assert!(tracker.latest(0,clock("test:2"),time,900_000_000).is_none());
        assert!(tracker.latest(0,clock("test:1"),time-1,900_000_000).is_none());
        tracker.begin(clock("test:1"),8);
        assert!(tracker.latest(0,clock("test:1"),time,900_000_000).is_none());
    }

    #[test]
    fn switching_joint_provider_invalidates_both_old_calibration_authorities_once() {
        let mut bridge=Bridge::default();let mut generations=[5,7];
        bridge.set_enabled(true,&mut generations);assert_eq!(generations,[6,8]);
        bridge.set_enabled(true,&mut generations);assert_eq!(generations,[6,8]);
        bridge.set_enabled(false,&mut generations);assert_eq!(generations,[7,9]);
    }
}
