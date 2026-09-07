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
    tracker: JointTracker,
}

fn clock(epoch:&str)->SourceClock {
    SourceClock {domain:1,epoch:epoch.bytes().fold(14695981039346656037,|h,b|(h^b as u64).wrapping_mul(1099511628211))}
}

impl Bridge {
    pub(crate) fn set_enabled(&mut self, enabled:bool, authority_generations:&mut [u64;2]) {
        if self.enabled==enabled {return;}
        self.enabled=enabled;self.signature=None;self.submitted=[None,None];self.tracker=JointTracker::default();
        // A monocular-surface calibration is not silently reused for a
        // different observation model, even though both are SAM providers.
        for generation in authority_generations {*generation=generation.wrapping_add(1);}
    }

    pub(crate) fn update(&mut self, frame:&mut EyeFrame, stream_epoch:&str,
        authority_generations:[u64;2], hub:&crate::recording_trace::Hub) {
        frame.joint_gaze_active=self.enabled && frame.segmentation_mode==SegmentationMode::Sam31;
        if !frame.joint_gaze_active {return;}
        let current_clock=clock(stream_epoch);
        let signature=(current_clock,authority_generations,frame.gaze_authority_sam_prompt_generation.unwrap_or(0));
        if self.signature!=Some(signature) {
            self.signature=Some(signature);self.generation=self.generation.wrapping_add(1);
            self.submitted=[None,None];self.tracker.begin(current_clock,self.generation);
        }
        let eye=frame.eye_id.saturating_sub(1) as usize;
        frame.joint_conic_status=Some("waiting-for-source-aligned-SAM-arcs".into());
        let source_frame=frame.sam31_proposal_masks.as_ref().and_then(|proposal| {
            if proposal.eye_index!=eye || Some(proposal.prompt_generation)!=frame.gaze_authority_sam_prompt_generation
                || proposal.source_width*proposal.source_height!=proposal.source_raw.len() {return None;}
            let reference=hub.source_reference(frame.eye_id,Some(proposal.source_timestamp_ns));
            let key=&reference["key"];
            if reference["status"]!="exact-source-key" || key["stream_epoch"].as_str()!=Some(stream_epoch)
                || key["sequence"].as_str()?.parse::<u64>().ok()?!=proposal.source_sequence {return None;}
            let exposure=ExposureKey {roi:RoiId(frame.eye_id),clock:current_clock,
                sequence:proposal.source_sequence,timestamp_ns:proposal.source_timestamp_ns};
            if self.submitted.get(eye).copied().flatten()==Some(exposure) {return None;}
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
            if let Some(pupil)=proposal.inner_pupil_fit {
                append_raw_ring_arcs(&mut packet,&proposal.source_raw,pupil.ellipse,
                    BoundaryKind::PupillaryBoundary,100,RawArcConfig::default());
            }
            let center=proposal.outer_fit.as_ref().map(|r|[r.ellipse.center.0,r.ellipse.center.1])
                .unwrap_or([proposal.source_width as f64*0.5,proposal.source_height as f64*0.5]);
            let pose=EyePoseInput {limbus_center_sensor_px:[center[0]+packet.sensor_origin_px[0] as f64,
                center[1]+packet.sensor_origin_px[1] as f64],
                pixels_per_10mm:frame.centimeter_scale.map(|s|[s.estimate_px,s.minimum_px,s.maximum_px])};
            Some(FrameEvidence {packet,pose})
        });
        if let Some(evidence)=source_frame {
            self.submitted[eye]=Some(evidence.packet.exposure);
            // Uncalibrated, explicit native-sensor pinhole engineering prior.
            // Do not label checkerboard intrinsics as used until supplied here.
            let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
            if let Err(error)=self.tracker.observe(evidence,camera) {
                frame.joint_conic_status=Some(format!("{error:?}"));
            }
        }
        self.install(frame,current_clock);
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
        sign_epoch:publication.source_generation,kinematic_sign_correction:[false;2],
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
        "frame":"camera-optical-center-mm-v1","units":"mm","axes":["sensor-right","sensor-down","toward-camera"],
        "method":"one shared fixation fitted to raw boundary segments; not averaged gaze points",
        "target":s.target_camera_mm,"target_covariance":null,
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
                normal_band_half_width_px:0.0,detector_score:None});
            packet.conics.push(OwnedConicHint {kind,ellipse_roi_px:ellipse,supporting_arc_indices:vec![packet.arcs.len()-1]});
        }
        FrameEvidence {packet,pose:EyePoseInput {limbus_center_sensor_px:camera.project(center).unwrap(),
            pixels_per_10mm:Some([4000.0/35.0,100.0,130.0])}}
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
