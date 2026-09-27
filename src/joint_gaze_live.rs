//! Viewer adapter for source-aligned mixed-ROI conic solving. The geometry
//! engine knows nothing about windows, mouse calibration, or rendered pixels.

use crate::conic_solver::joint::PinholeCamera;
use crate::eye_scene_model::{quantize_frontal_disk_area, SurfaceGazeSample, SurfaceSignDiagnostics, SurfaceSignEvidence};
use crate::eye_scene_model::binocular_pose::EyePoseInput;
use crate::gaze_target_solver::joint_tracking::{FrameEvidence, JointTracker, PublishedJoint};
#[cfg(test)]
use crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence;
use crate::roi_evidence::{BoundaryKind, ExposureKey, RoiId, SourceClock};
use crate::{EyeFrame, RelativeGazeVector, SegmentationMode, SAM31_RESULT_MAX_AGE_NS};
use serde_json::{json, Value};
use std::sync::Arc;

/// Explicit startup-only native-sensor pinhole configuration. This is an
/// engineering approximation, never an assertion of measured lens calibration.
pub(crate) fn parse_camera_intrinsics(value:&str)->Result<PinholeCamera,String> {
    let values:[f64;4]=serde_json::from_str(value)
        .map_err(|_|"joint camera intrinsics require JSON [fx,fy,cx,cy] in native sensor pixels")?;
    if !values.into_iter().all(f64::is_finite) || values[0]<=0.0 || values[1]<=0.0 {
        return Err("joint camera intrinsics require finite values and positive focal lengths".into());
    }
    Ok(PinholeCamera {focal_px:[values[0],values[1]],principal_px:[values[2],values[3]]})
}

/// Measured lens pinhole saved by the operator (e.g. from the checkerboard
/// collector). The environment variable still overrides it; a missing file
/// keeps the engineering default, and a malformed one is an error.
pub(crate) const JOINT_CAMERA_INTRINSICS_PATH:&str="outputs/settings/joint-camera-intrinsics.json";

pub(crate) fn load_saved_camera_intrinsics(path:&std::path::Path)->Result<Option<PinholeCamera>,String> {
    let bytes=match std::fs::read(path) {
        Ok(bytes)=>bytes,
        Err(error) if error.kind()==std::io::ErrorKind::NotFound=>return Ok(None),
        Err(error)=>return Err(format!("cannot read {}: {error}",path.display())),
    };
    let v:serde_json::Value=serde_json::from_slice(&bytes).map_err(|e|format!("{}: {e}",path.display()))?;
    if v["schema"]!="buttercup-joint-camera-intrinsics-v1" {
        return Err(format!("{}: unsupported joint camera intrinsics schema",path.display()));
    }
    parse_camera_intrinsics(&v["fx_fy_cx_cy_px"].to_string()).map(Some)
}

pub(crate) fn configured_camera()->Result<PinholeCamera,String> {
    static CAMERA:std::sync::OnceLock<Result<PinholeCamera,String>>=std::sync::OnceLock::new();
    CAMERA.get_or_init(||match std::env::var("BUTTERCUP_JOINT_CAMERA_INTRINSICS") {
        Ok(value)=>parse_camera_intrinsics(&value),
        Err(std::env::VarError::NotPresent)=>load_saved_camera_intrinsics(std::path::Path::new(JOINT_CAMERA_INTRINSICS_PATH))
            .map(|saved|saved.unwrap_or(PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]})),
        Err(error)=>Err(format!("invalid joint camera intrinsics environment: {error}")),
    }).clone()
}

#[derive(Default)]
pub(crate) struct Bridge {
    parallax_enabled: bool,
    parallax_clock: Option<SourceClock>,
    parallax: [crate::conic_solver::parallax_sign::Tracker;2],
    parallax_motion: [crate::raw_motion_octrees::NativeGlobalSimilarityTracker;2],
    continuous_gaze_diagnostic: bool,
    orientation_active: bool,
    orientation_frontier: Option<(SourceClock,u64)>,
    camera_mount: crate::eye_scene_model::CameraMount,
    enabled: bool,
    signature: Option<(SourceClock, [u64;2], u64)>,
    generation: u64,
    submitted: [Option<ExposureKey>;2],
    last_status: [Option<String>;2],
    tracker: JointTracker,
    #[cfg(test)]
    defer_explicit_pairs: bool,
    #[cfg(test)]
    buffered_pair_elisions: usize,
}

pub(super) fn clock(epoch:&str)->SourceClock {
    SourceClock {domain:1,epoch:epoch.bytes().fold(14695981039346656037,|h,b|(h^b as u64).wrapping_mul(1099511628211))}
}

impl Bridge {
    pub(crate) fn set_parallax(&mut self, enabled:bool) {
        if self.parallax_enabled!=enabled {
            self.parallax=Default::default();self.parallax_motion=Default::default();
            self.signature=None;
        }
        self.parallax_enabled=enabled;
    }
    pub(crate) fn parallax_status(&self)->serde_json::Value {
        serde_json::json!({"enabled":self.parallax_enabled,"eyes":self.parallax.each_ref().map(|p|p.json())})
    }
    fn install_monocular_parallax(&self, frame:&mut EyeFrame) {
        if !self.parallax_enabled || frame.joint_gaze_active {return;}
        let Some(surface)=frame.virtual_contact_surface_gaze.as_mut() else {return;};
        let Some(source)=surface.source_timestamp_ns else {return;};
        let eye=frame.eye_id.saturating_sub(1) as usize;
        let chosen=self.parallax[eye].normals(source);
        surface.sign_resolved=chosen.is_some();
        if let Some([normal,_])=chosen {
            if let Some(gaze)=RelativeGazeVector::from_projected(normal[0],normal[1]) {surface.relative_gaze=gaze;}
        }
    }

    pub(crate) fn set_continuous_gaze_diagnostic(&mut self, enabled:bool) {
        self.continuous_gaze_diagnostic=enabled;
        self.tracker.set_continuous_gaze_diagnostic(enabled);
    }
    pub(crate) fn set_orientation_active(&mut self, active:bool) {
        self.orientation_active=active;
        self.orientation_frontier=None;
        self.tracker.set_orientation_active(active);
    }
    /// On-screen stationary stimulus after orientation; see DirectionContinuation.
    pub(crate) fn set_screen_fixation(&mut self, active:bool) {
        self.tracker.set_screen_fixation(active);
    }
    /// Live routine tracking may re-establish direction from on-screen viewing.
    pub(crate) fn set_routine_acquisition(&mut self, enabled:bool) {
        self.tracker.set_routine_acquisition(enabled);
    }
    pub(crate) fn set_orientation_at(&mut self, active:bool, stream_epoch:&str, current_raw_source:u64) {
        let current_clock=clock(stream_epoch);
        if active && (!self.orientation_active || self.orientation_frontier.is_none_or(|(c,_)|c!=current_clock)) {
            self.orientation_frontier=Some((current_clock,current_raw_source));
        }
        if !active {self.orientation_frontier=None;}
        self.orientation_active=active;
        self.tracker.set_orientation_active(active);
    }
    pub(crate) fn set_camera_mount(&mut self, mode:crate::eye_scene_model::CameraMount) {
        if self.camera_mount==mode {return;}
        self.orientation_active=false;self.orientation_frontier=None;
        self.camera_mount=mode;
        self.signature=None;self.submitted=[None,None];self.last_status=[None,None];
        self.tracker=JointTracker::default();
        self.tracker.set_continuous_gaze_diagnostic(self.continuous_gaze_diagnostic);
        self.tracker.camera_mount=mode;
        self.tracker.legacy_mount_filter=false;
        self.tracker.set_probabilistic(true);
    }
    pub(crate) fn set_enabled(&mut self, enabled:bool, authority_generations:&mut [u64;2]) {
        if self.enabled==enabled {return;}
        self.orientation_active=false;self.orientation_frontier=None;
        self.enabled=enabled;self.signature=None;self.submitted=[None,None];self.last_status=[None,None];self.tracker=JointTracker::default();
        self.tracker.set_continuous_gaze_diagnostic(self.continuous_gaze_diagnostic);
        self.tracker.set_probabilistic(true);
        self.tracker.camera_mount=self.camera_mount;
        self.tracker.legacy_mount_filter=false;
        // A monocular-surface calibration is not silently reused for a
        // different observation model, even though both are SAM providers.
        for generation in authority_generations {*generation=generation.wrapping_add(1);}
    }

    /// Consume every admitted completion, not just the latest display slot
    /// for whichever ROI happens to receive the next RAW callback.
    pub(crate) fn observe_proposal(&mut self, proposal:&crate::sam31_outer::ProposalMasks,
        stream_epoch:&str, authority_generations:[u64;2], pixels_per_10mm:Option<[f64;3]>,
        hub:&crate::recording_trace::Hub) {
        self.observe_proposal_with_pair_wait(proposal,stream_epoch,authority_generations,pixels_per_10mm,hub,false);
    }

    /// Only adjacent, already-admitted completions can elide a provisional
    /// solve. No pending request, timeout or future partner is assumed here.
    pub(crate) fn observe_admitted_batch(&mut self, proposals:&[Arc<crate::sam31_outer::ProposalMasks>],
        stream_epoch:&str, authority_generations:[u64;2], scales:[Option<[f64;3]>;2],
        hub:&crate::recording_trace::Hub) {
        if !self.enabled && !self.parallax_enabled {return;}
        let mut index=0;
        while index<proposals.len() {
            let first=&proposals[index];
            let paired=proposals.get(index+1).is_some_and(|second| {
                first.eye_index<2 && second.eye_index==1-first.eye_index
                    && first.source_group_roi_count==2 && second.source_group_roi_count==2
                    && first.source_timestamp_ns==second.source_timestamp_ns
                    && first.prompt_generation==second.prompt_generation
                    && [first,second].into_iter().all(|p| {
                        if p.source_width.checked_mul(p.source_height)!=Some(p.source_raw.len()) {return false;}
                        let reference=hub.source_reference(p.eye_index as u32+1,Some(p.source_timestamp_ns));
                        let key=&reference["key"];
                        let signature=(clock(stream_epoch),authority_generations,p.prompt_generation);
                        reference["status"]=="exact-source-key" && key["stream_epoch"]==stream_epoch
                            && key["sequence"].as_str().and_then(|s|s.parse::<u64>().ok())==Some(p.source_sequence)
                            && (self.signature!=Some(signature) || self.submitted[p.eye_index]
                                .is_none_or(|old|old.timestamp_ns<p.source_timestamp_ns))
                    })
            });
            self.observe_proposal_with_pair_wait(first,stream_epoch,authority_generations,
                scales.get(first.eye_index).copied().flatten(),hub,paired);
            #[cfg(test)]
            if paired {self.buffered_pair_elisions+=1;}
            index+=1;
            if paired {
                let second=&proposals[index];
                self.observe_proposal(second,stream_epoch,authority_generations,scales[second.eye_index],hub);
                index+=1;
            }
        }
    }

    fn observe_proposal_with_pair_wait(&mut self, proposal:&crate::sam31_outer::ProposalMasks,
        stream_epoch:&str, authority_generations:[u64;2], pixels_per_10mm:Option<[f64;3]>,
        hub:&crate::recording_trace::Hub, wait_for_buffered_pair:bool) {
        if (!self.enabled && !self.parallax_enabled) || proposal.eye_index>=2 {return;}
        let current_clock=clock(stream_epoch);
        let signature=(current_clock,authority_generations,proposal.prompt_generation);
        if self.signature!=Some(signature) {
            self.signature=Some(signature);self.generation=self.generation.wrapping_add(1);
            self.submitted=[None,None];self.last_status=[None,None];self.tracker.begin(current_clock,self.generation);
        }
        if let Some((_,source))=self.orientation_frontier.filter(|(c,_)|*c==current_clock) {
            self.tracker.anchor_orientation_source(source);
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
            let packet=crate::sam31_outer::evidence_stage::joint_evidence(proposal, exposure)?;
            let center=proposal.outer_fit.as_ref().map(|r|[r.ellipse.center.0,r.ellipse.center.1])
                .unwrap_or([proposal.source_width as f64*0.5,proposal.source_height as f64*0.5]);
            let pose=EyePoseInput {limbus_center_sensor_px:[center[0]+packet.sensor_origin_px[0] as f64,
                center[1]+packet.sensor_origin_px[1] as f64],
                pixels_per_10mm};
            Some(FrameEvidence {packet,pose})
        })();
        if let Some(evidence)=source_frame {
            if self.parallax_enabled {
                if self.parallax_clock!=Some(current_clock) {
                    self.parallax_clock=Some(current_clock);
                    for p in &mut self.parallax {p.restart_stream();}
                    self.parallax_motion=Default::default();
                }
                if self.parallax[eye].last_source().is_none_or(|old|proposal.source_timestamp_ns>old) {
                    if let Some(fit)=proposal.outer_fit.as_ref() {
                        let mut exclusion=fit.ellipse;
                        exclusion.center.0+=proposal.source_sensor_origin.0 as f64;
                        exclusion.center.1+=proposal.source_sensor_origin.1 as f64;
                        let motion=self.parallax_motion[eye].observe_excluding(Arc::clone(&proposal.source_raw),
                            proposal.source_width,proposal.source_height,proposal.source_sensor_origin.0,
                            proposal.source_sensor_origin.1,Some(exclusion));
                        self.parallax[eye].observe(proposal.source_timestamp_ns,fit.ellipse,
                            [proposal.source_sensor_origin.0,proposal.source_sensor_origin.1],
                            configured_camera().expect("validated camera"),motion);
                    }
                }
            }
            self.tracker.parallax_enabled=self.parallax_enabled;
            self.tracker.parallax_normals=self.parallax.each_ref().map(|p|p.normals(proposal.source_timestamp_ns));
            if !self.enabled {return;}

            self.submitted[eye]=Some(evidence.packet.exposure);
            let camera=configured_camera().expect("joint intrinsics must be validated before processing");
            #[cfg(test)]
            let wait_for_pair=wait_for_buffered_pair || (self.defer_explicit_pairs && proposal.source_group_roi_count==2);
            #[cfg(not(test))]
            let wait_for_pair=wait_for_buffered_pair;
            self.last_status[eye]=self.tracker.observe_with_pair_wait(evidence,camera,wait_for_pair).err().map(|e|format!("{e:?}"));
        }
    }

    pub(crate) fn update(&mut self, frame:&mut EyeFrame, stream_epoch:&str,
        authority_generations:[u64;2], hub:&crate::recording_trace::Hub) {
        frame.joint_gaze_active=self.enabled && frame.segmentation_mode.uses_mask_geometry();
        if !frame.joint_gaze_active && !self.parallax_enabled {return;}
        if let Some(proposal)=frame.sam31_proposal_masks.as_ref().filter(|p|
            p.eye_index==frame.eye_id.saturating_sub(1) as usize
                && Some(p.prompt_generation)==frame.gaze_authority_sam_prompt_generation) {
            self.observe_proposal(proposal,stream_epoch,authority_generations,
                frame.centimeter_scale.map(|s|[s.estimate_px,s.minimum_px,s.maximum_px]),hub);
        }
        frame.joint_conic_status=self.last_status.get(frame.eye_id.saturating_sub(1) as usize)
            .cloned().flatten().or_else(||Some("waiting-for-source-aligned-SAM-arcs".into()));
        self.install(frame,clock(stream_epoch));
        self.install_monocular_parallax(frame);
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
    let orientation_ready=publication.orientation_ready.map(|ready|ready[eye]);
    let sign_resolved=orientation_ready.unwrap_or(solution.contributing_eyes==[true,true]||noncoplanar)
        && solution.posterior.as_ref().map_or_else(
            ||solution.alternative_cost_margin.is_some_and(|margin|margin>2.0),
            |posterior|posterior.supports_direction(eye));
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
        sign_diagnostics:Some(SurfaceSignDiagnostics {orientation_reference_ready:orientation_ready,continuous_gaze:None,evidence:if sign_resolved {
            if orientation_ready==Some(true) {SurfaceSignEvidence::MountingAssumption} else {SurfaceSignEvidence::JointConics}
        } else {SurfaceSignEvidence::Unresolved},
            selected_branch:0,branch_residual_ema_px:[f64::NAN;2],source_motion_residual_px:None,
            temporal_margin_px:None,pending_anchor_votes:0,near_frontal_continuation:false})})
}

/// Preserve fitted source conics directly. The contact's disk area and normal
/// are insufficient to recover image axes under off-axis perspective. These
/// are conditional solution outputs, not new independent boundary observations.
fn source_projected_conics_json(publication:&PublishedJoint)->Value {
    let ellipse_json=|ellipse:Option<crate::geometry::Ellipse>| {
        ellipse.filter(|e|e.center.0.is_finite() && e.center.1.is_finite()
            && e.major_radius.is_finite() && e.minor_radius.is_finite()
            && e.angle.is_finite() && e.minor_radius>0.0 && e.major_radius>=e.minor_radius)
            .map(|e|json!({"center":[e.center.0,e.center.1],"major_radius":e.major_radius,
                "minor_radius":e.minor_radius,"angle_rad":e.angle})).unwrap_or(Value::Null)
    };
    let eyes:[Value;2]=std::array::from_fn(|eye| {
        let Some((source,(origin,dimensions)))=publication.exposures[eye]
            .zip(publication.sensor_origins_px[eye].zip(publication.dimensions_px[eye])) else {
            return Value::Null;
        };
        json!({"source":{"roi_id":source.roi.0,"clock_domain":source.clock.domain.to_string(),
                "clock_epoch":source.clock.epoch.to_string(),"sequence":source.sequence.to_string(),
                "sensor_timestamp_ns":source.timestamp_ns.to_string()},
            "sensor_origin_px":origin,"dimensions_px":dimensions,
            "modeled_eye":publication.solution.modeled_eyes[eye],
            "contributing_eye":publication.solution.contributing_eyes[eye],
            "ellipses":publication.solution.ellipses_roi_px[eye].map(ellipse_json)})
    });
    json!({"coordinate_frame":"source-roi-pixels",
        "boundary_order":["OuterLimbus","InnerLimbus","PupillaryBoundary"],"eyes":eyes,
        "provenance":"conditional joint-conic fit; not independent observed boundary evidence"})
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
        "local_uncertainty":s.local_uncertainty.as_ref().map(|u|u.json()),
        "posterior":s.posterior.as_ref().map(|p|p.json()),
        "screen_orientation_ready":publication.orientation_ready,
        "continuous_gaze_sign":publication.continuous_gaze_sign.as_ref().map(|r|r.json()),
        "target_search_chart":{"frame":"reference-to-camera-tangent-plane","slopes":s.target_viewpoint_slopes,
            "reference_camera_mm":s.target_reference_camera_mm,
            "axial_distance_mm":s.target_viewpoint_axial_distance_mm,"distance_axis":"reference-to-camera",
            "slope_limit":s.target_viewpoint_slope_limit,"active_bounds":s.target_viewpoint_bounds_active},
        "source_generation":publication.source_generation.to_string(),
        "scale_provenance":publication.scene.scale_provenance.map(|p|p.map(|p|format!("{p:?}"))),
        "intrinsics":{"focal_px":publication.scene.prior.camera.focal_px,"principal_px":publication.scene.prior.camera.principal_px,"provenance":"uncalibrated pinhole approximation; explicit startup values or engineering default"},
        "sources":publication.exposures.map(|e|e.map(|e|json!({"roi_id":e.roi.0,"clock_domain":e.clock.domain.to_string(),
            "clock_epoch":e.clock.epoch.to_string(),"sequence":e.sequence.to_string(),"sensor_timestamp_ns":e.timestamp_ns.to_string()}))),
        "eye_centers":s.eye_centers_camera_mm,"surface_normals":s.eye_normals,"gaze_directions":s.eye_gaze_directions,
        "source_projected_conics":source_projected_conics_json(publication),
        "effective_pivots_camera_mm":s.effective_pivots_camera_mm,
        "surface_axis_alignment_radians":s.surface_axis_alignment_radians,
        "effective_pivot_provenance":"conditional fitted nuisance geometry; not an independent head/pivot measurement",
        "contributing_eyes":s.contributing_eyes,"cost":s.robust_cost,"alternative_cost_margin":s.alternative_cost_margin,
        "modeled_eyes":s.modeled_eyes,"unlocalized_eye_cost":s.unlocalized_eye_cost,
        "hypotheses":s.hypotheses_evaluated,"hypotheses_by_association":s.hypotheses_by_association,
        "arcs":s.arcs.iter().map(|a|json!({"roi_id":a.exposure.roi.0,"group":a.evidence_group,"kind":format!("{:?}",a.kind),
            "arc_index":a.arc_index,"points_roi_px":a.points_roi_px,
            "coordinate_frame":"source-roi-pixels",
            "point_provenance":"bounded residual samples of selected arc alternative at selected mask level; not full contour or fitted ellipse",
            "source":{"roi_id":a.exposure.roi.0,"clock_domain":a.exposure.clock.domain.to_string(),
                "clock_epoch":a.exposure.clock.epoch.to_string(),"sequence":a.exposure.sequence.to_string(),
                "sensor_timestamp_ns":a.exposure.timestamp_ns.to_string()},
            "mask_level":a.mask_level,
            "used":a.used,"rms_px":a.rms_px,"sigma_px":a.sigma_px,
            "boundary_normal_samples":a.boundary_normal_samples,"boundary_normal_rms_radians":a.boundary_normal_rms_radians,
            "support_length_px":a.support_length_px,"evidence_weight":a.evidence_weight})).collect::<Vec<_>>(),
        "uncertainty":"conditional engineering supports; not calibrated probabilities or measured anatomical pose",
        "transform_to_legacy_monitor_frame":null})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires explicit recorded conics; exports native two-circle pose diagnostic"]
    fn recorded_conic_two_pose_contact_sheet() {
        let input=std::env::var("BUTTERCUP_CONIC_POSE_INPUT").unwrap();
        let output=std::env::var("BUTTERCUP_CONIC_POSE_OUTPUT").unwrap();
        let rows:Vec<Value>=serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
        let reports=rows.into_iter().map(|row| {
            let camera=PinholeCamera {
                focal_px:serde_json::from_value(row["camera"]["focal_px"].clone()).unwrap(),
                principal_px:serde_json::from_value(row["camera"]["principal_px"].clone()).unwrap(),
            };
            let e=&row["ellipse"];
            let ellipse=crate::geometry::Ellipse {center:(e["center"][0].as_f64().unwrap(),e["center"][1].as_f64().unwrap()),
                major_radius:e["major_radius"].as_f64().unwrap(),minor_radius:e["minor_radius"].as_f64().unwrap(),angle:e["angle_rad"].as_f64().unwrap()};
            let origin=serde_json::from_value(row["origin"].clone()).unwrap();
            let poses=crate::conic_solver::joint::circle_pose_hypotheses(camera,ellipse,origin).unwrap();
            let alternatives=poses.map(|p| {
                let conic=ProjectedCircle::project(camera,p.center_per_radius,p.normal,1.0,origin).unwrap();
                let max_error=ellipse.dense_points(180).into_iter().map(|q|conic.residual_px(q).abs()).fold(0.0,f64::max);
                assert!(max_error<1e-5,"native circle reprojection differs from input conic: {max_error}");
                let facing=-crate::geometry::dot3(p.normal,crate::geometry::normalized3(p.center_per_radius).unwrap());
                assert!(facing>0.0);
                json!({"normal":p.normal,"center_per_radius":p.center_per_radius,"camera_facing_cosine":facing,
                    "maximum_reprojection_error_px":max_error,
                    "camera_below_in_unrotated_sensor_axes":p.center_per_radius[1]<0.0,
                    "current_below_eyes_gaze_prior_accepts":crate::conic_solver::camera_mount::CameraMount::BelowEyes.supports(p.normal[1])})
            });
            json!({"input":row,"poses":alternatives,
                "contract":"native perspective circular sections of one recorded fitted ellipse; not two retained joint posterior modes; scale in iris radii; sensor down is not gravity without camera attitude"})
        }).collect::<Vec<_>>();
        let mut f=std::fs::OpenOptions::new().create_new(true).write(true).open(output).unwrap();
        use std::io::Write;
        f.write_all(serde_json::to_string_pretty(&reports).unwrap().as_bytes()).unwrap();
    }
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
            packet.arcs.push(OwnedBoundaryArc { support_length_cap_px: None, sampling_support_px: None, level_sets_roi: None,evidence_group:group,kind,points_roi_px:ellipse.dense_points(32),
                outward_normals_roi:None,localization_sigma_px:None,normal_band_half_width_px:0.0,detector_score:None});
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
    fn admitted_batches_preserve_missing_invalid_and_interleaved_partner_behavior() {
        let time=1_000_000_000;
        for case in 0..10 {
            let mut inputs=vec![proposal(0,time),proposal(1,time)];
            for p in &mut inputs {p.source_group_roi_count=2;}
            match case {
                1=>{inputs.pop();},
                2=>inputs[1].outer_fit=None,
                3=>inputs[1].source_raw=Arc::new(vec![100]),
                5=>{
                    inputs=vec![proposal(0,time),proposal(0,time+100_000_000),
                        proposal(1,time),proposal(1,time+100_000_000)];
                    for p in &mut inputs {p.source_group_roi_count=2;}
                },
                6=>inputs.reverse(),
                7=>inputs[1].source_group_roi_count=1,
                9=>inputs[1].prompt_generation=1,
                _=>{},
            }
            let inputs=inputs.into_iter().map(Arc::new).collect::<Vec<_>>();
            let hub=crate::recording_trace::Hub::default();
            for (i,p) in inputs.iter().enumerate() {
                if case==4 && i==1 {continue;}
                hub.raw_arrived(json!({"roi_id":p.eye_index+1,"sensor_timestamp_ns":p.source_timestamp_ns.to_string(),
                    "sequence":p.source_sequence.to_string(),"stream_epoch":"test:1"}),std::time::Instant::now(),0);
            }
            let mut a=Bridge::default();let mut b=Bridge::default();let mut generations=[1,1];
            a.set_enabled(true,&mut generations);b.set_enabled(true,&mut generations);
            let scales=[Some([114.,100.,130.]);2];
            if case==8 {
                for bridge in [&mut a,&mut b] {bridge.observe_proposal(&inputs[0],"test:1",generations,scales[0],&hub);}
            }
            for p in &inputs {a.observe_proposal(p,"test:1",generations,scales[p.eye_index],&hub);}
            b.observe_admitted_batch(&inputs,"test:1",generations,scales,&hub);
            assert_eq!(b.buffered_pair_elisions,usize::from(matches!(case,0|2|6)),"case {case}");
            for eye in 0..2 {
                let latest=|bridge:&Bridge|bridge.tracker.latest(eye,clock("test:1"),time+100_000_000,500_000_000);
                assert_eq!(format!("{:?}",latest(&a)),format!("{:?}",latest(&b)),"case {case}, eye {eye}");
            }
            if matches!(case,1|3|4) {
                assert!(b.tracker.latest(0,clock("test:1"),time,1).is_some(),"missing or invalid partner lost single-eye fallback");
            }
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
    fn projected_conic_metadata_stays_with_its_actual_source_after_display_reframe() {
        let time=1_000_000_000;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),15);
        tracker.observe(evidence(0,time),camera).unwrap();
        let publication=tracker.observe(evidence(1,time),camera).unwrap().unwrap();
        let expected=publication.solution.ellipses_roi_px[0][0].unwrap();
        let mut frame=crate::tests::control_eye_frame(1);
        frame.timestamp_ns=time;frame.width=420;frame.height=280;
        frame.sensor_x=3424;frame.sensor_y=2860;frame.joint_gaze_active=true;
        frame.joint_conic=Some(publication);
        let before=json(&frame)["source_projected_conics"].clone();
        assert_eq!(before["coordinate_frame"],"source-roi-pixels");
        assert_eq!(before["eyes"][0]["source"]["sequence"],"100");
        assert_eq!(before["eyes"][1]["source"]["sequence"],"900");
        assert_eq!(before["eyes"][0]["sensor_origin_px"],json!([3424,2860]));
        assert_eq!(before["eyes"][0]["ellipses"][0]["center"],json!([expected.center.0,expected.center.1]));
        assert_eq!(before["eyes"][0]["ellipses"][0]["minor_radius"],expected.minor_radius);
        assert_eq!(before["eyes"][0]["ellipses"][0]["angle_rad"],expected.angle);
        frame.sensor_x+=64;frame.sensor_y-=32;frame.timestamp_ns+=100_000_000;
        assert_eq!(json(&frame)["source_projected_conics"],before,
            "moving/redrawing the display ROI cannot rebase or freshen fitted source conics");
        Arc::make_mut(frame.joint_conic.as_mut().unwrap()).exposures[1]=None;
        assert!(json(&frame)["source_projected_conics"]["eyes"][1].is_null(),
            "a missing source cannot export the other eye's model as a sourced observation");
        Arc::make_mut(frame.joint_conic.as_mut().unwrap()).sensor_origins_px[0]=None;
        assert!(json(&frame)["source_projected_conics"]["eyes"][0].is_null());
    }

    #[test]
    fn recorded_conic_segments_keep_exact_source_points_and_rejection_status() {
        let time=1_000_000_000;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let originals=[evidence(0,time),evidence(1,time)];
        let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),15);
        tracker.observe(evidence(0,time),camera).unwrap();
        let publication=tracker.observe(evidence(1,time),camera).unwrap().unwrap();
        let mut frame=crate::tests::control_eye_frame(1);
        frame.joint_gaze_active=true;frame.joint_conic=Some(publication);
        // Mark one diagnostic rejected: its coordinates must remain available.
        Arc::make_mut(frame.joint_conic.as_mut().unwrap()).solution.arcs[0].used=false;
        let report=json(&frame);
        let arcs=report["arcs"].as_array().unwrap();
        assert!(!arcs.is_empty());assert_eq!(arcs[0]["used"],false);
        for a in arcs {
            let eye=a["roi_id"].as_u64().unwrap() as usize-1;
            let packet=&originals[eye].packet;
            let original=&packet.arcs[a["arc_index"].as_u64().unwrap() as usize].points_roi_px;
            let points=a["points_roi_px"].as_array().unwrap();
            assert!(points.len()>=3 && points.len()<=original.len());
            for (i,p) in points.iter().enumerate() {
                let expected=original[i*(original.len()-1)/(points.len()-1)];
                assert_eq!(*p,json!([expected.0,expected.1]));
            }
            assert_eq!(a["source"]["sequence"],packet.exposure.sequence.to_string());
            assert_eq!(a["source"]["sensor_timestamp_ns"],time.to_string());
            assert_eq!(a["coordinate_frame"],"source-roi-pixels");
        }
        frame.sensor_x+=200;frame.timestamp_ns+=500_000_000;
        assert_eq!(json(&frame)["arcs"],report["arcs"],"presentation changes cannot relocate source segments");
    }

    #[test]
    fn projected_conic_metadata_leaves_missing_or_invalid_boundaries_unknown() {
        let time=1_000_000_000;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let mut tracker=JointTracker::default();tracker.begin(clock("test:1"),15);
        let mut publication=tracker.observe(evidence(0,time),camera).unwrap().unwrap();
        let solution=&mut Arc::make_mut(&mut publication).solution;
        solution.ellipses_roi_px[0][1]=None;
        solution.ellipses_roi_px[0][2]=Some(crate::geometry::Ellipse {
            center:(f64::NAN,0.0),major_radius:10.0,minor_radius:5.0,angle:0.0});
        let report=source_projected_conics_json(&publication);
        assert!(report["eyes"][0]["ellipses"][0].is_object());
        assert!(report["eyes"][0]["ellipses"][1].is_null());
        assert!(report["eyes"][0]["ellipses"][2].is_null());
        assert!(report["eyes"][1].is_null());
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
        let time=3_000_000_000;
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
            mode.target_source_started_ns=Some(time-2_200_000_000);
            let hidden=start+crate::VIRTUAL_MOUSE_TARGET_SETTLE;
            mode.calibration_presented(hidden,hidden);
            // Both completion revisions retain the same original RAW receipt;
            // inference completion never renews the calibration eligibility.
            mode.sample_source_arrived_at=Some(start+Duration::from_millis(2200));
            mode.frame_state=WaitingForSourcePartner;
            mode.observe_at_frame(start+Duration::from_millis(2200),Some(time),None);
            assert!(mode.samples.iter().all(Vec::is_empty));

            let paired=tracker.observe(evidence(1-eye,time),camera).unwrap().unwrap();
            frame.joint_conic=Some(paired);
            assert_eq!(crate::calibration_frame_state(Some(&frame),Some(sample)),Ready);
            mode.frame_state=Ready;
            for elapsed in [2300,2350] {
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
    #[ignore = "requires recorded worker ellipse cache; conditional branch diagnostic, no ground truth"]
    fn camera_mount_recorded_branch_trial() {
        use std::io::{BufRead,Write};
        use std::time::{Instant,Duration};
        use crate::eye_scene_model::{CameraMount,SurfaceGazeTracker};
        let input=std::env::var("BUTTERCUP_JOINT_CALIBRATION_CACHE").unwrap();
        let output=std::env::var("BUTTERCUP_CAMERA_MOUNT_REPORT").unwrap();
        let modes=[CameraMount::Flexible,CameraMount::BelowEyes,CameraMount::AboveEyes];
        let mut trackers:[[SurfaceGazeTracker;2];3]=std::array::from_fn(|m|
            std::array::from_fn(|_|SurfaceGazeTracker {camera_mount:modes[m],..Default::default()}));
        let now=Instant::now();let mut first=None;let mut seen=[None;2];
        let mut rows=Vec::new();let mut missing=0;
        for line in std::io::BufReader::new(std::fs::File::open(&input).unwrap()).lines() {
            let case:Value=serde_json::from_str(&line.unwrap()).unwrap();
            let meta=&case["input"]["frame"];let eye=meta["eye_id"].as_u64().unwrap() as usize-1;
            let time=meta["timestamp_ns"].as_u64().unwrap();
            if seen[eye].is_some_and(|old|old>=time) {continue;} seen[eye]=Some(time);
            let Some(c)=replay_selected_candidate(&case) else {missing+=1;continue;};
            let e=&c["baseline_ellipse"];
            let Some(major)=e["major_radius"].as_f64() else {missing+=1;continue;};
            let outer=crate::raw_iris_focus::OuterIrisBoundary {
                center:(e["center"][0].as_f64().unwrap(),e["center"][1].as_f64().unwrap()),
                major_radius:major,minor_radius:e["minor_radius"].as_f64().unwrap(),angle:e["angle"].as_f64().unwrap(),
                points:vec![crate::raw_iris_focus::OuterIrisPoint::default();8],..Default::default()};
            let origin=(meta["sensor_x"].as_u64().unwrap() as u32,meta["sensor_y"].as_u64().unwrap() as u32);
            let at=now+Duration::from_nanos(time-*first.get_or_insert(time));
            let results:Vec<_>=(0..3).map(|m|trackers[m][eye].observe_keyed_with_global_similarity(time,at,origin,None,&outer,None)
                .map(|s|json!({"resolved":s.sign_resolved,"gaze":s.relative_gaze.projected(),"area_px2":s.frontal_equivalent_disk_area_px2,
                    "source_ns":s.source_timestamp_ns,"epoch":s.sign_epoch}))).collect();
            for result in results.iter().flatten() {assert_eq!(result["source_ns"],json!(time));}
            if results.iter().all(Option::is_some) {
                assert_eq!(results[0].as_ref().unwrap()["area_px2"],results[1].as_ref().unwrap()["area_px2"]);
                assert_eq!(results[0].as_ref().unwrap()["area_px2"],results[2].as_ref().unwrap()["area_px2"]);
            }
            rows.push(json!({"eye":eye,"source_ns":time,"modes":results}));
        }
        assert!(!rows.is_empty());
        let counts:Vec<_>=(0..3).map(|m|json!({"mode":modes[m].label(),"resolved":rows.iter().filter(|r|r["modes"][m]["resolved"]==true).count(),
            "available":rows.iter().filter(|r|!r["modes"][m].is_null()).count()})).collect();
        let report=json!({"input":input,"rows":rows,"counts":counts,"missing_ellipse":missing,
            "limitations":"Rob-only recorded contour replay. Identical ellipse inputs, no motion/pupil cues in this ablation. No sign truth, human localization labels or independent scale. Raw FEIDA unchanged is not SN-FEIDA accuracy evidence. Wrong mounting can force the wrong sign."});
        std::fs::File::create(output).unwrap().write_all(serde_json::to_string_pretty(&report).unwrap().as_bytes()).unwrap();
    }

    #[test]
    #[ignore = "requires fresh live-worker replay cache and native RAW corpus under outputs"]
    fn recorded_calibration_acquires_from_source_timed_live_worker_evidence() {
        replay_recorded_calibration(false);
    }
    #[test]
    #[ignore = "Source-matched continuous-sign diagnostic; requires explicit recorded cache/report paths"]
    fn recorded_calibration_with_continuous_gaze_diagnostic() {
        replay_recorded_calibration(true);
    }
    fn replay_recorded_calibration(continuous_gaze:bool) {
        use std::io::{BufRead,Read,Seek,Write};
        use std::time::{Duration,Instant};
        let coupled_index=std::env::var("BUTTERCUP_JOINT_CALIBRATION_NATIVE_INDEX").ok();
        let input=std::env::var("BUTTERCUP_JOINT_CALIBRATION_CACHE").unwrap_or_default();
        let output=std::env::var("BUTTERCUP_JOINT_CALIBRATION_REPORT").unwrap();
        let require_ready=std::env::var("BUTTERCUP_JOINT_CALIBRATION_REQUIRE_READY").as_deref()==Ok("1");
        let omit_pupil=std::env::var("BUTTERCUP_JOINT_CALIBRATION_OMIT_PUPIL").as_deref()==Ok("1");
        let half_pupil_weight=std::env::var("BUTTERCUP_JOINT_PUPIL_HALF_WEIGHT").as_deref()==Ok("1");
        assert!(!(omit_pupil && half_pupil_weight),"choose one pupil ablation");
        let ignore_source_group=std::env::var("BUTTERCUP_JOINT_CALIBRATION_IGNORE_SOURCE_GROUP").as_deref()==Ok("1");
        let integration_recipe=std::env::var("BUTTERCUP_JOINT_CALIBRATION_INTEGRATION")
            .unwrap_or_else(|_|"live".into());
        let calibration_eye=std::env::var("BUTTERCUP_JOINT_CALIBRATION_EYE").map(|v|v.parse::<usize>().unwrap()).unwrap_or(0);
        assert!(calibration_eye<2,"calibration eye must be zero-based 0 or 1");
        let mut writer=std::fs::OpenOptions::new().create_new(true).write(true).open(output).unwrap();
        let mut native_cache_writer=std::env::var("BUTTERCUP_JOINT_CALIBRATION_NATIVE_CACHE_REPORT").ok()
            .map(|path|std::fs::OpenOptions::new().create_new(true).write(true).open(path).unwrap());
        let mut bridge=Bridge::default();let mut generations=[0;2];bridge.set_enabled(true,&mut generations);
        bridge.set_continuous_gaze_diagnostic(continuous_gaze);
        let camera_mount=match std::env::var("BUTTERCUP_CAMERA_MOUNT_TRIAL") {
            Ok(value)=>crate::eye_scene_model::CameraMount::parse(&value).expect("invalid legacy camera mount override"),
            Err(std::env::VarError::NotPresent)=>crate::eye_scene_model::CameraMount::for_offline_checks().unwrap(),
            Err(error)=>panic!("{error}"),
        };
        camera_mount.warn_if_not_below("recorded calibration trial");
        bridge.set_camera_mount(camera_mount);
        let screen_reference_mode=std::env::var("BUTTERCUP_SCREEN_REFERENCE_TRIAL").unwrap_or_default();
        assert!(matches!(screen_reference_mode.as_str(), ""|"0"|"1"|"recorded"),
            "screen reference trial must be 0, 1, or recorded");
        let recorded_screen_reference=screen_reference_mode=="recorded";
        let screen_reference_trial=screen_reference_mode=="1" || recorded_screen_reference;
        // Matches the live UI: stationary targets after acquisition are on-screen
        // fixations. Off by default so historical baselines replay unchanged.
        let screen_fixation_trial=std::env::var("BUTTERCUP_SCREEN_FIXATION_TRIAL").as_deref()==Ok("1");
        assert!(!screen_fixation_trial || screen_reference_trial,"screen fixation requires the screen-reference phase policy");
        // Preserve an explicit historical control for matched experiments.
        bridge.tracker.legacy_mount_filter=!screen_reference_trial;
        bridge.set_orientation_active(screen_reference_trial && !recorded_screen_reference);
        let camera_diagnostic=std::env::var("BUTTERCUP_JOINT_CALIBRATION_INTRINSICS").ok().map(|value| {
            let values: [f64;4]=serde_json::from_str(&value)
                .expect("camera trial requires JSON [fx,fy,cx,cy] in native sensor pixels");
            let camera=PinholeCamera {focal_px:[values[0],values[1]],principal_px:[values[2],values[3]]};
            bridge.tracker.set_camera_diagnostic(camera);
            values
        });
        let configured=configured_camera().unwrap();
        let reported_camera=camera_diagnostic.unwrap_or([configured.focal_px[0],configured.focal_px[1],
            configured.principal_px[0],configured.principal_px[1]]);
        let radial_camera_diagnostic=std::env::var("BUTTERCUP_JOINT_CALIBRATION_RADIAL_K1").ok().map(|value| {
            let k1=value.parse::<f64>().expect("radial camera trial requires a finite coefficient");
            bridge.tracker.set_radial_camera_diagnostic(k1);
            k1
        });
        let integration_budget=std::env::var("BUTTERCUP_JOINT_CALIBRATION_QUADRATURE_BUDGET").ok().map(|value| {
            let budget=value.parse::<usize>().expect("quadrature budget must be an integer");
            assert!((8192..=262144).contains(&budget),"offline quadrature budget outside bounded diagnostic range");
            budget
        });
        let mut integration=match integration_recipe.as_str() {
            "live"=>crate::conic_solver::joint::posterior::IntegrationConfig::live(),
            "baseline"=>crate::conic_solver::joint::posterior::IntegrationConfig::default(),
            _=>panic!("select live or baseline integration for the matched calibration diagnostic"),
        };
        if let Some(budget)=integration_budget {integration.budget=budget;}
        if integration_budget.is_some() || integration_recipe=="baseline" {
            bridge.tracker.set_posterior_diagnostic(integration);
        }
        let hub=crate::recording_trace::Hub::default();
        let mut frames:[Option<EyeFrame>;2]=[None,None];
        bridge.defer_explicit_pairs=std::env::var("BUTTERCUP_JOINT_CALIBRATION_DEFER_EXPLICIT_PAIRS").as_deref()==Ok("1");
        let start=Instant::now();let mut first_time=None;let mut acquired_at=None;
        let mut acquisition=crate::calibration_acquisition::Acquisition::default();acquisition.start(start);
        acquisition.use_screen_reference(screen_reference_trial);
        let mut calibration=crate::VirtualMouseMode::new(start);
        calibration.sign_acquisition.use_screen_reference(screen_reference_trial);
        let ellipse=|v:&Value|->Option<crate::geometry::Ellipse> {
            Some(crate::geometry::Ellipse {center:(v["center"][0].as_f64()?,v["center"][1].as_f64()?),
                major_radius:v["major_radius"].as_f64()?,minor_radius:v["minor_radius"].as_f64()?,angle:v["angle"].as_f64()?})
        };
        let mut count=0;
        let mut process=|case:Value,native:Option<&Arc<crate::sam31_outer::ProposalMasks>>| {
            let processing_started=Instant::now();
            let row=&case["input"];let meta=&row["frame"];
            let n=|key|meta[key].as_u64().unwrap();let eye=n("eye_id") as usize-1;
            let time=n("timestamp_ns");let sequence=n("sequence");
            assert_eq!(case["timestamp_ns"].as_u64(),Some(time));
            assert_eq!(case["sequence"].as_u64(),Some(sequence));
            let source_start=case["replay"]["first_source_ns"].as_str().map(|s|s.parse::<u64>().unwrap()).unwrap_or(time);
            let elapsed_ns=case["replay"]["ready_elapsed_ns"].as_str().map(|s|s.parse::<u64>().unwrap())
                .unwrap_or_else(||time-*first_time.get_or_insert(source_start));
            let now=start+Duration::from_nanos(elapsed_ns);
            let epoch=meta["source_clock"]["source_key"]["stream_epoch"].as_str().unwrap();
            if recorded_screen_reference {
                let active=recorded_orientation_active(&case,time,epoch);
                bridge.set_orientation_at(active,epoch,time);
            }
            let pixels=if native.is_some() {Vec::new()} else {
            let mut raw=std::fs::File::open(row["raw_file"].as_str().unwrap()).unwrap();
            raw.seek(std::io::SeekFrom::Start(row["raw_offset"].as_u64().unwrap())).unwrap();
            let mut packed=vec![0;row["raw_length"].as_u64().unwrap() as usize];raw.read_exact(&mut packed).unwrap();
            crate::raw10::try_unpack_raw10(&packed,n("width") as usize,n("height") as usize,n("stride") as usize).unwrap()
            };
            let candidate=replay_selected_candidate(&case);
            let (source_group_roi_count,source_group_metadata)=replay_source_group(&case);
            let source_group_roi_count=if ignore_source_group {0}else{source_group_roi_count};
            let mut proposal=if let Some(native)=native {(**native).clone()} else {crate::sam31_outer::ProposalMasks {
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
            }};
            proposal.source_group_roi_count=source_group_roi_count;
            // Test-only ablation: omit the measured pupil proposal before shared
            // RAW evidence extraction, never replace it with fitted points.
            if omit_pupil {proposal.inner_pupil_fit=None;}
            let scale=&row["scale_hint"];
            let scale=scale["pixels_per_10mm"].as_f64().and_then(|estimate|Some([estimate,
                scale["bounds_px_per_10mm"][0].as_f64()?,scale["bounds_px_per_10mm"][1].as_f64()?]));
            let native_processing_started=Instant::now();
            let stamp=hub.raw_arrived(meta["source_clock"]["source_key"].clone(),now,n("host_arrival_unix_ns"));
            bridge.observe_proposal(&proposal,epoch,generations,scale,&hub);
            let native_observe_ns=native_processing_started.elapsed().as_nanos() as u64;
            let mut frame=crate::tests::control_eye_frame(sequence);
            frame.eye_id=eye as u32+1;frame.timestamp_ns=time;frame.width=proposal.source_width;frame.height=proposal.source_height;
            frame.sensor_x=proposal.source_sensor_origin.0;frame.sensor_y=proposal.source_sensor_origin.1;
            frame.segmentation_mode=if native.is_some() {SegmentationMode::EyeStudent} else {SegmentationMode::Sam31};frame.recording_clock=stamp;
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
            let delivered_elapsed_ns=elapsed_ns+if native.is_some() {processing_started.elapsed().as_nanos() as u64} else {0};
            let delivered_now=start+Duration::from_nanos(delivered_elapsed_ns);
            let result=acquisition.observe(delivered_now,1,qualified);
            if result==crate::calibration_acquisition::Update::Ready {
                acquired_at.get_or_insert(delivered_elapsed_ns);
                if screen_reference_trial && !recorded_screen_reference {bridge.set_orientation_active(false);}
                if screen_fixation_trial {bridge.set_screen_fixation(true);}
            }
            // Exercise the enclosing UI phase machine too. Testing only the
            // acquisition helper missed re-entry after its first success.
            let current_time=focus.map(|f|f.timestamp_ns);
            calibration.acquisition_source_generation=1;
            calibration.observe_frame_state(current_time,state);
            calibration.surface_gaze=surface;
            calibration.observe_at_frame(delivered_now,current_time,qualified.zip(surface).map(|(q,s)|
                (q.source_ns,s.relative_gaze.projected(),q.epoch)));
            let native_processing_ns=native_processing_started.elapsed().as_nanos() as u64;
            let report=json!({"input_index":row["index"],"source_ns":time.to_string(),"elapsed_ms":elapsed_ns/1_000_000,
                "pupil_evidence_ablation":if omit_pupil {"omit-measured-pupil"} else if half_pupil_weight {"half-weight"} else {"none"},
                "coupled_native_processing":native.is_some(),
                "coupled_processing_contract":native.is_some().then_some(
                    "CPU workers remain active during scalar native completion processing; buffered-pair optimization not applied by this hook; synchronous diagnostic export also loads the pump; no renderer or scanout timing"),
                "native_ready_elapsed_ns":native.is_some().then(||delivered_elapsed_ns.to_string()),
                "worker_replay":case["replay"],
                "native_service_timing":{"observe_proposal_ns":native_observe_ns.to_string(),
                    "publication_and_admission_ns":native_processing_ns.saturating_sub(native_observe_ns).to_string(),
                    "total_ns":native_processing_ns.to_string(),
                    "worker_ready_elapsed_ns":case["replay"]["ready_elapsed_ns"],
                    "contract":if native.is_some() {"native bridge service measured while CPU model workers remain active; excludes diagnostic output and rendering"}
                        else {"measured serial native bridge, publication and admission work; excludes RAW decode, JSON I/O, worker inference and rendering; no coupled worker/solver scheduling claim"}},
                "camera_mount_assumption":camera_mount.label(),
                "screen_reference_trial":screen_reference_trial,"screen_fixation_trial":screen_fixation_trial,
                "recorded_orientation":case["recorded_orientation"],
                "screen_reference_trial_contract":screen_reference_trial.then_some(if recorded_screen_reference {
                    "recorded target visibility mapped by paired RAW host arrivals; source-time settling retained; not measured scanout or offered-load live latency"
                } else {"counterfactual initial screen fixation; recorded target/pose not certified; no claim of successful live calibration"}),
                "defer_explicit_pairs":bridge.defer_explicit_pairs,
                "integration_recipe":integration_recipe,
                "integration_budget_override":integration_budget,
                "intrinsics_diagnostic":reported_camera,
                "radial_camera_diagnostic":radial_camera_diagnostic,
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
                "ui_probe_contract":"unpresented stationary-target phase probe; does not submit hidden thumbnails or replay recorded target transitions; not an end-to-end recorded calibration",
                "ui_target_sample_counts":calibration.samples.iter().map(Vec::len).collect::<Vec<_>>(),
                "method":"live-worker cache through actual bridge and acquisition; no monitor-accuracy validation"});
            serde_json::to_writer(&mut writer,&report).unwrap();writeln!(writer).unwrap();count+=1;
            if native.is_some() {
                if let Some(cache)=native_cache_writer.as_mut() {
                    serde_json::to_writer(&mut *cache,&case).unwrap();writeln!(cache).unwrap();
                }
            }
        };
        if let Some(index)=coupled_index {
            let model=crate::sam31_outer::student::default_model_path();
            let sources=crate::offline_segmentation_replay::stereo::source_frames(std::path::Path::new(&index),0,usize::MAX).unwrap();
            crate::offline_segmentation_replay::stereo::visit_offered_frames_native(&model,"student",sources,
                crate::sam31_outer::Target::OuterLimbusAndInnerPupilVoid,|row,_,mut case,proposal| {
                    case["input"]=row;case["backend"]=json!("student");case["model"]=json!(model);
                    case["configuration"]=crate::sam31_outer::live_configuration();
                    process(case,Some(proposal));Ok(())
                }).unwrap();
        } else {
            for line in std::io::BufReader::new(std::fs::File::open(input).unwrap()).lines() {
                process(serde_json::from_str(&line.unwrap()).unwrap(),None);
            }
        }
        drop(process);
        eprintln!("RECORDED_CALIBRATION frames={count} acquired_ms={:?}",acquired_at.map(|ns|ns/1_000_000));
        // A fixed-stimulus comparison deliberately replays recorded recovery
        // targets. The unpresented UI probe does not control that schedule and
        // cannot certify that a counterfactual UI would avoid reorientation.
        if !recorded_screen_reference {
            assert!(calibration.sign_acquisition.episodes<=1,"a fixed-lineage recording must not repeatedly re-enter acquisition");
        }
        assert_eq!(calibration.calibration_sign_restarts,0,"unsigned frames cannot invalidate the stationary sequence");
        if require_ready {assert!(acquired_at.is_some(),"last calibration failed to acquire; inspect the source-timed report");}
    }

    fn recorded_orientation_active(case:&Value, source_ns:u64, epoch:&str)->bool {
        let schedule=&case["recorded_orientation"];
        assert_eq!(schedule["source_ns"].as_str(),Some(source_ns.to_string().as_str()),
            "orientation visibility must match the exact RAW source");
        assert_eq!(schedule["clock_lineage"].as_str(),Some(epoch),
            "orientation visibility cannot cross recording clocks");
        assert_eq!(schedule["timing_basis"],"paired-raw-host-arrival-vs-recorded-submit");
        schedule["active"].as_bool().expect("recorded orientation visibility is required")
    }

    #[test]
    fn recorded_orientation_schedule_requires_exact_source_and_clock() {
        let case=json!({"recorded_orientation":{"source_ns":"123","clock_lineage":"clip:1",
            "timing_basis":"paired-raw-host-arrival-vs-recorded-submit","active":true}});
        assert!(recorded_orientation_active(&case,123,"clip:1"));
        let mut off=case.clone();off["recorded_orientation"]["active"]=json!(false);
        assert!(!recorded_orientation_active(&off,123,"clip:1"));
        assert!(std::panic::catch_unwind(||recorded_orientation_active(&case,124,"clip:1")).is_err());
        assert!(std::panic::catch_unwind(||recorded_orientation_active(&case,123,"clip:2")).is_err());
        assert!(std::panic::catch_unwind(||recorded_orientation_active(&json!({}),123,"clip:1")).is_err());
    }

    #[test]
    fn explicit_joint_intrinsics_validate_native_pixel_values() {
        let camera=parse_camera_intrinsics("[5488,5712,4000,3000]").unwrap();
        assert_eq!(camera.focal_px,[5488.0,5712.0]);
        assert_eq!(camera.principal_px,[4000.0,3000.0]);
        for bad in ["[0,4000,4000,3000]","[-1,4000,4000,3000]","[4000,4000,3000]",
            "[4000,4000,4000,3000,1]","[1e999,4000,4000,3000]","null"] {
            assert!(parse_camera_intrinsics(bad).is_err(),"{bad}");
        }
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
            // Cursor authority requires the exact provider/prompt source,
            // independently of whether geometry exists in the publication.
            frame.sam31_proposal_masks=Some(Arc::new(proposal(eye,time)));
            frame.gaze_authority_sam_prompt_generation=Some(0);
            let contact=surface(&frame,false).unwrap();
            assert!((40.0..100.0).contains(&contact.quantized_frontal_disk_radius_px),"area must be converted back to radius");
            assert_eq!(contact.source_timestamp_ns,Some(time));
            let gaze=gaze(&frame).unwrap();
            assert_eq!(crate::mouse_gaze_surface(&frame).unwrap().relative_gaze,gaze);
            let exact=normalized3(sub3(result.solution.target_camera_mm,result.solution.eye_centers_camera_mm[eye].unwrap())).unwrap();
            assert!((gaze.right-exact[0]).abs()<1e-10&&(gaze.down-exact[1]).abs()<1e-10);
            frame.virtual_contact_surface_gaze=Some(contact);
            let pose=crate::virtual_contact_pose(&frame).expect("joint contact should render");
            assert_eq!(crate::pose_for_cursor(&frame,pose).is_some(),contact.sign_resolved,
                "an ambiguous optimizer result must retain the sign gate");
            // This is a routing fixture, not proof of statistical sign
            // acquisition. Exercise an explicitly admitted synthetic branch.
            Arc::make_mut(frame.joint_conic.as_mut().unwrap()).solution.alternative_cost_margin=Some(3.0);
            assert_eq!(crate::pose_for_cursor(&frame,pose).unwrap().relative_gaze,gaze);
            let mut no_source=frame.clone();no_source.sam31_proposal_masks=None;
            assert!(crate::pose_for_cursor(&no_source,pose).is_none(),"missing source cannot gain cursor authority");
            Arc::make_mut(frame.joint_conic.as_mut().unwrap()).solution.alternative_cost_margin=None;
            assert!(crate::pose_for_cursor(&frame,pose).is_none(),"unknown branch margin cannot gain cursor authority");
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
    #[ignore = "larger numerical references for the native synthetic publication fixture"]
    fn synthetic_publication_posterior_reference() {
        use crate::conic_solver::joint::posterior::IntegrationConfig;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let time=1_000_000_000;
        let budget=std::env::var("BUTTERCUP_PUBLICATION_REFERENCE_BUDGET")
            .map(|s|s.parse::<usize>().expect("integer draw ceiling")).unwrap_or(65536);
        assert!((8192..=1048576).contains(&budget));
        let mut selected=None;
        for seed in [0xd1b5_4a32_d192_ed03,0x94c5_09a1_814f_753d,0x419b_79df_a2c7_5301] {
            for (recipe,mut config) in [("baseline",IntegrationConfig::default()),("live",IntegrationConfig::live())] {
                config.seed=seed;config.budget=budget;config.early_stop=false;
                let mut tracker=JointTracker::default();tracker.set_probabilistic(true);tracker.begin(clock("test:1"),7);
                tracker.set_posterior_diagnostic(IntegrationConfig {budget:0,..config});
                tracker.observe(evidence(0,time),camera).unwrap();
                tracker.set_posterior_diagnostic(config);
                let start=std::time::Instant::now();
                let result=tracker.observe(evidence(1,time),camera).unwrap().unwrap();
                if let Some(target)=selected {assert_eq!(result.solution.target_camera_mm,target);}
                selected=Some(result.solution.target_camera_mm);
                assert_eq!(result.solution.contributing_eyes,[true,true]);
                eprintln!("publication-reference {}",serde_json::json!({
                    "recipe":recipe,"seed":seed.to_string(),"budget":budget,
                    "elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
                    "target_camera_mm":result.solution.target_camera_mm,
                    "posterior":result.solution.posterior.as_ref().unwrap().json(),
                    "contract":"Exact synthetic projected circles through native coarse scene priors; numerical reference, not empirical gaze accuracy."}));
            }
        }
    }

    #[test]
    fn publication_routes_only_numerically_supported_current_gaze() {
        use crate::conic_solver::joint::posterior::DirectionNumerics;
        let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
        let time=1_000_000_000;
        let mut tracker=JointTracker::default();tracker.set_probabilistic(true);tracker.begin(clock("test:1"),7);
        tracker.observe(evidence(0,time),camera).unwrap();
        let result=tracker.observe(evidence(1,time),camera).unwrap().unwrap();
        for eye in 0..2 {
            let mut frame=crate::tests::control_eye_frame(1);
            frame.eye_id=eye as u32+1;frame.timestamp_ns=time+50_000_000;
            frame.width=420;frame.height=280;frame.segmentation_mode=SegmentationMode::Sam31;
            let origin=result.sensor_origins_px[eye].unwrap();frame.sensor_x=origin[0];frame.sensor_y=origin[1];
            frame.joint_gaze_active=true;frame.joint_conic=Some(Arc::clone(&result));
            frame.sam31_proposal_masks=Some(Arc::new(proposal(eye,time)));
            frame.gaze_authority_sam_prompt_generation=Some(0);
            let uncertainty=result.solution.posterior.as_ref().unwrap();
            eprintln!("live posterior eye {eye}: {} {:?}",uncertainty.status,uncertainty.gaze_radius_90_degrees);
            // One-million-draw controls under these unchanged coarse anatomy
            // priors put only 85--90% near the selected ray. Perfect projected
            // circles do not make this particular native fixture identifiable.
            assert!(!uncertainty.supports_direction(eye));
            assert!(crate::gaze_output_direction(&frame).is_none());
            // Explicit posterior doubles test publication routing separately
            // from numerical integration. Production-entry conic tests cover
            // actual estimated positive cases; these values are not estimates
            // or a way to narrow the coarse fixture's posterior.
            for (mass,error,radius,status,batches,expected) in [
                (0.97,0.01,3.0,"estimated-conditional",1,true),
                (0.91,0.01,3.0,"estimated-conditional",1,false),
                (0.97,0.01,35.0,"estimated-conditional",1,false),
                (0.97,0.01,3.0,"insufficient-sampling",1,false),
                (0.97,0.01,3.0,"estimated-conditional",4,false),
            ] {
                let mut controlled=frame.clone();
                let solution=&mut Arc::make_mut(controlled.joint_conic.as_mut().unwrap()).solution;
                solution.alternative_cost_margin=Some(1000.0);
                let posterior=solution.posterior.as_mut().unwrap();
                posterior.status=status;
                posterior.require_numerical_margin=true;
                posterior.replicas=batches;
                posterior.replicate_direction_numerics=[None,None];
                posterior.direction_numerics[eye]=Some(DirectionNumerics {mass,standard_error:error});
                posterior.gaze_radius_90_degrees[eye]=Some(radius);
                assert_eq!(crate::gaze_output_direction(&controlled).is_some(),expected,
                    "a large MAP gap cannot overrule angular spread, precision, missing batches or failed integration");
            }
        }
    }

    #[test]
    fn switching_joint_provider_invalidates_both_old_calibration_authorities_once() {
        let mut bridge=Bridge::default();let mut generations=[5,7];
        bridge.set_enabled(true,&mut generations);assert_eq!(generations,[6,8]);
        bridge.set_enabled(true,&mut generations);assert_eq!(generations,[6,8]);
        bridge.set_enabled(false,&mut generations);assert_eq!(generations,[7,9]);
    }
}
