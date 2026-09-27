//! Shared gaze input for the independent pointer and window-focus outputs.
//! Hidden windows still get event-loop ticks. Preparation/mapping matches draw.
use crate::mouse_output::{Sample, Source};
use crate::{App, EyeFrame, GlobalGazePolicy, SegmentationMode, SharedState, VirtualContactAuthority, VirtualDisplayPlane};
use std::time::{Duration, Instant};

pub(crate) fn tick(app: &mut App) {
    let now = Instant::now();
    if now < app.desktop_gaze_next_tick {
        return;
    }
    app.desktop_gaze_next_tick = now + Duration::from_millis(10);
    let (mouse_generation, focus_generation, eyes_generation, cursor_generation) = app
        .shared
        .lock()
        .map(|s| {
            (
                s.mouse_output.enabled_generation(),
                s.gaze_focus.enabled_generation(),
                s.wleyes.enabled_generation(),
                s.gaze_cursor.enabled_generation(),
            )
        })
        .unwrap_or((None, None, None, None));
    if mouse_generation.is_none() && focus_generation.is_none() && eyes_generation.is_none()
        && cursor_generation.is_none() {
        return;
    }
    // refresh_viewer_frames updates each image only when its source changes.
    let sample = if crate::refresh_viewer_frames(app).is_err() {
        Err("paused: viewer state unavailable")
    } else {
        current_sample(app)
    };
    // The cartoon is a diagnostic preview, not an input-control consumer.
    // It may display the same source's conditional direction before sign is
    // accepted; that path never supplies a mouse/focus/calibration sample.
    let preview = eyes_generation.map(|_| cartoon_sample(app));
    let error = app.shared.lock().ok().and_then(|mut shared| {
        let now = Instant::now();
        // Projection happens outside this lock. A concurrent G/settings,
        // reference-eye or monitor change must not republish the old target
        // after the mutation synchronously canceled pending desktop work.
        let sample = sample.and_then(|(sample, selection)| {
            (selection == Selection::from_shared(&shared))
                .then_some(sample)
                .ok_or("paused: global gaze selection changed during projection")
        });
        if let Some(generation) = focus_generation {
            shared.gaze_focus.publish(generation, now, sample);
        }
        if let Some(generation) = cursor_generation {
            shared.gaze_cursor.publish(generation, sample);
        }
        if let Some(generation) = eyes_generation {
            let mut approximate=false;
            let input=preview.expect("enabled cartoon has a preview result").and_then(|(sample,selection,rough)|{
                approximate=rough;
                (selection==Selection::from_shared(&shared)).then_some(sample)
                    .ok_or("paused: global gaze selection changed during projection")
            });
            shared.wleyes.publish_preview(generation, input, approximate);
        }
        mouse_generation.and_then(|generation| shared.mouse_output.update(generation, now, sample))
    });
    if let Some(error) = error {
        eprintln!("{error}");
        // Only an asynchronous device failure needs a viewer notification.
        // Explicit enable errors are displayed by the shortcut client instead.
        std::thread::spawn(move || {
            let _ = std::process::Command::new("notify-send")
                .args([
                    "--app-name=Buttercup",
                    "--urgency=critical",
                    "--expire-time=10000",
                    "Buttercup mouse OFF",
                    &error,
                ])
                .status();
        });
    }
}

fn source(frame: &EyeFrame, prompt_generation: u64) -> Result<Source, &'static str> {
    let (source,surface)=preview_source(frame,prompt_generation)?;
    if !surface.sign_resolved {return Err("paused: unresolved gaze sign");}
    Ok(source)
}

fn preview_source(frame: &EyeFrame, prompt_generation: u64) -> Result<(Source,crate::SurfaceGazeSample), &'static str> {
    if let Some(reason) = frame.gaze_policy_error { return Err(reason); }
    if !frame.eye_identity_present {
        return Err("paused: eye not present");
    }
    let surface = crate::mouse_gaze_surface(frame).ok_or("paused: no gaze surface")?;
    let timestamp_ns = surface
        .source_timestamp_ns
        .ok_or("paused: no gaze source clock")?;
    if frame.segmentation_mode.uses_mask_geometry()
        && !frame.sam31_proposal_masks.as_ref().is_some_and(|p| {
            p.source_timestamp_ns == timestamp_ns
                && p.prompt_generation == prompt_generation
                && Some(p.prompt_generation) == frame.gaze_authority_sam_prompt_generation
        })
    {
        return Err("paused: SAM source or prompt mismatch");
    }
    Ok((Source {
        eye: frame.eye_id.saturating_sub(1) as usize,
        authority: frame.gaze_authority_generation,
        sign_epoch: surface.sign_epoch,
        timestamp_ns,
    },surface))
}

/// Shared pre-projection gaze evidence for desktop output and read-only
/// telemetry. This is not a screen target and does not require calibration.
/// The caller must pass the same globally prepared frame used for presentation;
/// preparation errors, unresolved signs and carried presentation poses all
/// remain unavailable rather than falling back to a different analysis mode.
#[derive(Clone, Debug)]
pub(crate) struct AuthorizedGaze {
    pub source: Source,
    pub receipt: crate::recording_trace::SourceReceipt,
    pub direction: crate::RelativeGazeVector,
}

pub(crate) fn authorized_gaze(
    frame: &EyeFrame,
    prompt_generation: u64,
    trace: &crate::recording_trace::Hub,
) -> Result<AuthorizedGaze, &'static str> {
    let source = source(frame, prompt_generation)?;
    let receipt = trace
        .source_receipt(frame.eye_id, source.timestamp_ns)
        .ok_or("paused: unknown or ambiguous source clock")?;
    if frame.presentation_pivot_held {
        return Err("paused: held contact is not a fresh observation");
    }
    let direction = crate::gaze_output_direction(frame).ok_or("paused: no current gaze ray")?;
    Ok(AuthorizedGaze { source, receipt, direction })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Selection {
    policy: GlobalGazePolicy,
    eye: usize,
    plane: VirtualDisplayPlane,
    object_search: bool,
}
impl Selection {
    fn from_shared(s: &SharedState) -> Self {
        Self {
            policy: GlobalGazePolicy::from_shared(s),
            eye: s.focus_eye,
            plane: s.monitor_location.effective_plane(),
            object_search: s.sam31_object_inspection,
        }
    }
}

fn prepared_frame(app: &App) -> Result<(EyeFrame,Selection,crate::recording_trace::Hub), &'static str> {
    // No desktop pointer movement while collecting targets, editing a prompt,
    // or doing non-eye object search. Neither pause toggles J or stops analysis.
    if app.virtual_mouse.is_some() || app.accuracy_requested || app.accuracy_check.is_some() {
        return Err("paused: calibration or accuracy screen");
    }
    if app.sam31_prompt_editor.is_some() {
        return Err("paused: prompt editor");
    }
    let (selection, trace) = {
        let s = app
            .shared
            .lock()
            .map_err(|_| "paused: viewer state unavailable")?;
        if s.sam31_object_inspection {
            return Err("paused: object search");
        }
        (
            Selection::from_shared(&s),
            s.recording_trace.clone(),
        )
    };
    let mut frame = app.eyes[selection.eye]
        .clone()
        .ok_or("paused: no eye frame")?;
    crate::prepare_current_contact_frame(app, selection.eye, &mut frame, selection.policy);
    Ok((frame,selection,trace))
}

fn current_sample(app: &App) -> Result<(Sample, Selection), &'static str> {
    let (frame,selection,trace)=prepared_frame(app)?;
    let gaze = authorized_gaze(&frame, selection.policy.prompt_generation, &trace)?;
    let calibration = app
        .calibrated_display
        .and_then(|c| c.for_frame(selection.eye, Some(&frame)));
    let target = crate::display_gaze_target(gaze.direction, calibration, selection.plane)
        .ok_or("paused: no forward monitor intersection")?;
    Ok((Sample {
        source: gaze.source,
        age: gaze.receipt.age_at(Instant::now())
            .ok_or("paused: unknown or ambiguous source clock")?,
        target,
    }, selection))
}

fn cartoon_sample(app:&App)->Result<(Sample,Selection,bool),&'static str>{
    let (frame,selection,trace)=prepared_frame(app)?;
    if frame.presentation_pivot_held {return Err("paused: held contact is not a fresh observation");}
    let (source,surface)=preview_source(&frame,selection.policy.prompt_generation)?;
    let receipt=trace.source_receipt(frame.eye_id,source.timestamp_ns)
        .ok_or("paused: unknown or ambiguous source clock")?;
    let age=receipt.age_at(Instant::now()).ok_or("paused: unknown or ambiguous source clock")?;
    let calibrated=app.calibrated_display.and_then(|c|c.for_frame(selection.eye,Some(&frame)))
        .and_then(|calibration|authorized_gaze(&frame,selection.policy.prompt_generation,&trace).ok()
            .and_then(|gaze|crate::display_gaze_target(gaze.direction,Some(calibration),selection.plane)));
    // Deliberately approximate camera-relative motion. A mirrored conic branch
    // can still be wrong; neither this estimate nor a stale screen mapping
    // becomes input-control or calibration evidence.
    let target=calibrated.unwrap_or((0.5-1.25*surface.relative_gaze.right,0.5+1.25*surface.relative_gaze.down));
    Ok((Sample{source,age,target},selection,calibrated.is_none()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn surface() -> crate::SurfaceGazeSample {
        crate::SurfaceGazeSample {
            source_timestamp_ns: Some(100),
            frontal_equivalent_disk_area_px2: 1000.0,
            area_bucket: 0,
            quantized_frontal_disk_radius_px: 30.0,
            near_surface_point_sensor_px: (100.0, 100.0),
            relative_gaze: crate::RelativeGazeVector::from_projected(0.1, -0.2).unwrap(),
            sign_resolved: true,
            sign_epoch: 7,
            kinematic_sign_correction: [false; 2],
            sign_diagnostics: None,
        }
    }

    #[test]
    fn cartoon_can_preview_uncertain_direction_without_authorizing_pointer() {
        let mut frame=crate::tests::control_eye_frame(1);
        frame.eye_identity_present=true;
        frame.segmentation_mode=SegmentationMode::Sam31;
        frame.gaze_authority_sam_prompt_generation=Some(3);
        let mut estimate=surface();estimate.sign_resolved=false;
        frame.virtual_contact_surface_gaze=Some(estimate);
        frame.sam31_proposal_masks=Some(std::sync::Arc::new(crate::sam31_outer::ProposalMasks {
            source_timestamp_ns:100,prompt_generation:3,..Default::default()
        }));
        assert!(preview_source(&frame,3).is_ok());
        assert_eq!(source(&frame,3).unwrap_err(),"paused: unresolved gaze sign");
        assert!(preview_source(&frame,4).is_err(),"cartoon must still reject another prompt/source");
    }

    #[test]
    fn every_gaze_consumer_rejects_previous_global_method_or_settings() {
        let mut state = SharedState::default();
        state.segmentation_mode = SegmentationMode::Sam31;
        state.segmentation_generation = 8;
        state.sam31_prompt_bundle_generation = 3;
        let mut frame = crate::tests::control_eye_frame(1);
        frame.eye_identity_present = true;
        frame.segmentation_mode = SegmentationMode::Sam31;
        frame.gaze_settings_generation = 8;
        frame.gaze_authority_sam_prompt_generation = Some(3);
        frame.virtual_contact_surface_gaze = Some(surface());
        frame.sam31_proposal_masks = Some(std::sync::Arc::new(crate::sam31_outer::ProposalMasks {
            source_timestamp_ns: 100,
            prompt_generation: 3,
            ..Default::default()
        }));
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame), None);
        assert!(source(&frame, 3).is_ok());

        for (method, revision, reason) in [
            (SegmentationMode::EyeStudent, 9, "paused: waiting for global gaze method"),
            // Same enum after a round-trip must not resurrect old settings.
            (SegmentationMode::Sam31, 10, "paused: waiting for global gaze settings"),
        ] {
            state.segmentation_mode = method;
            state.segmentation_generation = revision;
            frame.gaze_policy_error = GlobalGazePolicy::from_shared(&state).frame_error(&frame);
            assert_eq!(frame.gaze_policy_error, Some(reason));
            assert_eq!(source(&frame, 3).unwrap_err(), reason);
            assert!(crate::mouse_gaze_surface(&frame).is_none());
            assert!(crate::gaze_feature(&frame).is_none());
            assert!(crate::virtual_contact_pose(&frame).is_none());
        }
        // Invalidation is a freshness fence, not destructive tracker editing.
        assert_eq!(frame.virtual_contact_surface_gaze.unwrap().sign_epoch, 7);
        assert!(frame.virtual_contact_surface_gaze.unwrap().sign_resolved);
    }

    #[test]
    fn global_prompt_stereo_and_enabled_eye_are_part_of_the_source_contract() {
        let mut state = SharedState::default();
        state.segmentation_mode = SegmentationMode::Sam31;
        let mut frame = crate::tests::control_eye_frame(1);
        frame.segmentation_mode = SegmentationMode::Sam31;
        frame.gaze_authority_sam_prompt_generation = Some(0);
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame), None);
        state.sam31_prompt_bundle_generation = 1;
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame),
            Some("paused: waiting for global gaze prompt"));
        frame.gaze_authority_sam_prompt_generation = Some(1);
        state.second_roi_enabled = true;
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame), None);
        state.stereo_solver_enabled = true;
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame),
            Some("paused: waiting for global stereo setting"));
        frame.joint_gaze_active = true;
        frame.eye_id = 2;
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame), None);
        state.second_roi_enabled = false;
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame),
            Some("paused: reference eye analysis disabled"));
    }

    #[test]
    fn sizing_freshness_does_not_reset_sign_or_completed_calibration() {
        let mut state = SharedState::default();
        let mut frame = crate::tests::control_eye_frame(1);
        frame.surface_gaze = Some(surface());
        let calibration = crate::CalibratedDisplay {
            gaze_affine_input: crate::GazeAffineInput::ProjectedDirection,
            restored: false,
            eye: 0,
            segmentation_mode: frame.segmentation_mode,
            sam_prompt_generation: None,
            gaze_authority_generation: frame.gaze_authority_generation,
            sign_epoch: 7,
            plane: VirtualDisplayPlane::development_default(),
            gaze_affine: crate::GazeAffine { x: [1.0, 0.0, 0.0], y: [0.0, 1.0, 0.0] },
        };
        crate::bump_pupil_sizing_generation(&mut state);
        assert_eq!(state.gaze_input_generation, 0);
        frame.gaze_policy_error = GlobalGazePolicy::from_shared(&state).frame_error(&frame);
        assert!(calibration.for_frame(0, Some(&frame)).is_none());
        // Next RAW analysis uses the new sizing policy with the same sign/basis.
        frame.gaze_settings_generation = state.segmentation_generation;
        frame.gaze_policy_error = GlobalGazePolicy::from_shared(&state).frame_error(&frame);
        assert!(calibration.for_frame(0, Some(&frame)).is_some());
        assert_eq!(frame.surface_gaze.unwrap().sign_epoch, 7);
        crate::bump_iris_runtime_policy_generation(&mut state);
        assert_eq!(state.gaze_input_generation, 0);
        assert_eq!(GlobalGazePolicy::from_shared(&state).frame_error(&frame),
            Some("paused: waiting for global gaze settings"));
    }

    #[test]
    fn in_flight_desktop_projection_is_bound_to_global_selection_not_preview() {
        let mut state = SharedState::default();
        let captured = Selection::from_shared(&state);
        let mut preview = crate::viewer_ui::Workspace::default();
        preview.cycle_view(SegmentationMode::Sam31);
        preview.selected = 1;
        preview.scope = crate::viewer_ui::Scope::Linked;
        assert_eq!(captured, Selection::from_shared(&state));
        state.focus_eye = 1;
        assert_ne!(captured, Selection::from_shared(&state));
        state.focus_eye = 0;
        state.sam31_object_inspection = true;
        assert_ne!(captured, Selection::from_shared(&state));
        state.sam31_object_inspection = false;
        let mut plane = state.monitor_location.effective_plane();
        plane.center_inches[0] += 1.0;
        state.monitor_location.offer(plane);
        assert_ne!(captured, Selection::from_shared(&state));
    }

    #[test]
    fn cursor_and_laser_use_calibrations_signed_surface_not_a_divergent_contact_normal() {
        for method in [SegmentationMode::Native, SegmentationMode::Driving, SegmentationMode::Clusters] {
            let mut frame = crate::tests::control_eye_frame(1);
            frame.segmentation_mode = method;
            frame.surface_gaze = Some(surface());
            let contact_normal = crate::RelativeGazeVector::from_projected(-0.4, 0.5).unwrap();
            frame.virtual_contact_surface_gaze = Some(crate::SurfaceGazeSample {
                relative_gaze: contact_normal, ..surface()
            });
            let contact = crate::VirtualContactPose {
                rotation_center: (20.0, 30.0),
                rotation_center_z: Some(-40.0),
                relative_gaze: contact_normal,
                sphere_radius: Some(50.0),
                authority: VirtualContactAuthority::MotionLocked,
            };
            let cursor = crate::pose_for_cursor(&frame, contact).unwrap();
            assert_eq!(cursor.relative_gaze, surface().relative_gaze, "{method:?}");
            assert_eq!(crate::gaze_feature(&frame), Some(cursor.relative_gaze.projected()));
            assert_eq!(crate::gaze_output_direction(&frame), Some(cursor.relative_gaze));
            assert_eq!(cursor.rotation_center, contact.rotation_center);
            assert_eq!(cursor.sphere_radius, contact.sphere_radius);
            assert_eq!(contact.relative_gaze, contact_normal, "globe normal remains presentation geometry");

            frame.surface_gaze.as_mut().unwrap().sign_resolved = false;
            assert!(crate::pose_for_cursor(&frame, contact).is_none());
            frame.surface_gaze = Some(crate::SurfaceGazeSample { source_timestamp_ns: None, ..surface() });
            assert!(crate::pose_for_cursor(&frame, contact).is_none());
            frame.surface_gaze = Some(crate::SurfaceGazeSample {
                source_timestamp_ns: Some(frame.timestamp_ns + 1), ..surface()
            });
            assert!(crate::pose_for_cursor(&frame, contact).is_none());
            frame.surface_gaze = Some(surface());
            frame.gaze_policy_error = Some("paused: waiting for global gaze settings");
            assert!(crate::pose_for_cursor(&frame, contact).is_none());
        }
    }

    #[test]
    fn sam_output_direction_requires_exact_source_and_never_uses_native_fallback() {
        for method in [SegmentationMode::Sam31, SegmentationMode::EyeStudent] {
            let mut frame = crate::tests::control_eye_frame(1);
            frame.segmentation_mode = method;
            frame.surface_gaze = Some(surface());
            frame.virtual_contact_surface_gaze = Some(surface());
            frame.gaze_authority_sam_prompt_generation = Some(3);
            assert!(crate::gaze_output_direction(&frame).is_none());
            frame.sam31_proposal_masks = Some(std::sync::Arc::new(crate::sam31_outer::ProposalMasks {
                source_timestamp_ns: 101, prompt_generation: 3, ..Default::default()
            }));
            assert!(crate::gaze_output_direction(&frame).is_none());
            std::sync::Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).source_timestamp_ns = 100;
            assert_eq!(crate::gaze_output_direction(&frame), Some(surface().relative_gaze));
            frame.gaze_authority_sam_prompt_generation = Some(4);
            assert!(crate::gaze_output_direction(&frame).is_none());
            frame.gaze_authority_sam_prompt_generation = Some(3);
            frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_resolved = false;
            assert!(crate::gaze_output_direction(&frame).is_none());
        }
    }

    #[test]
    fn sam_requires_its_own_signed_surface_and_exact_prompt_source() {
        let mut frame = crate::tests::control_eye_frame(1);
        frame.eye_identity_present = true;
        frame.segmentation_mode = SegmentationMode::Sam31;
        frame.surface_gaze = Some(surface()); // Native geometry is not a fallback.
        frame.virtual_contact_surface_gaze = None;
        assert!(source(&frame, 3).is_err());
        frame.virtual_contact_surface_gaze = Some(surface());
        frame.gaze_authority_sam_prompt_generation = Some(3);
        frame.sam31_proposal_masks = Some(std::sync::Arc::new(crate::sam31_outer::ProposalMasks {
            source_timestamp_ns: 100,
            prompt_generation: 3,
            ..crate::sam31_outer::ProposalMasks::default()
        }));
        assert_eq!(source(&frame, 3).unwrap().timestamp_ns, 100);
        // Transported to a later RAW ROI still uses the actual solve's clock.
        frame.timestamp_ns = 200;
        assert_eq!(source(&frame, 3).unwrap().timestamp_ns, 100);
        assert!(source(&frame, 4).is_err());
        frame
            .virtual_contact_surface_gaze
            .as_mut()
            .unwrap()
            .sign_resolved = false;
        assert_eq!(
            source(&frame, 3).unwrap_err(),
            "paused: unresolved gaze sign"
        );
    }

    #[test]
    fn saved_calibration_retains_perspective_basis_and_migrates_legacy_coordinates() {
        let original = crate::CalibratedDisplay {
            restored: false, eye: 0, segmentation_mode: crate::SegmentationMode::EyeStudent,
            sam_prompt_generation: None, gaze_authority_generation: 0, sign_epoch: 0,
            plane: crate::VirtualDisplayPlane::development_default(),
            gaze_affine: crate::GazeAffine { x: [1.0,0.0,0.0], y: [0.0,1.0,0.0] },
            gaze_affine_input: crate::GazeAffineInput::DisplayIntersection,
        };
        let ray = crate::RelativeGazeVector::from_projected(-0.1,-0.5).unwrap();
        let restored = crate::CalibratedDisplay::from_json(&original.json()).unwrap();
        assert_eq!(restored.gaze_affine_input, original.gaze_affine_input);
        assert_eq!(restored.target(ray), original.plane.target(ray));
        let mut legacy = original.json();
        legacy["schema"] = serde_json::json!("buttercup-gaze-calibration-v1");
        legacy["gaze_affine"].as_object_mut().unwrap().remove("input");
        let restored = crate::CalibratedDisplay::from_json(&legacy).unwrap();
        assert_eq!(restored.gaze_affine_input, crate::GazeAffineInput::ProjectedDirection);
        assert_eq!(restored.target(ray), Some(ray.projected()));
        for input in [serde_json::Value::Null, serde_json::json!("unknown-space")] {
            let mut invalid = original.json(); invalid["gaze_affine"]["input"] = input;
            assert!(crate::CalibratedDisplay::from_json(&invalid).is_err());
        }
        let mut backward = original;
        backward.plane.center_inches = [0.0,0.0,-24.0];
        backward.plane.right_axis = [1.0,0.0,0.0];
        backward.plane.down_axis = [0.0,1.0,0.0];
        assert_eq!(backward.target(ray), None);
    }

    #[test]
    fn saved_gaze_calibration_reloads_without_session_generation_or_prompt_gates() {
        let mut frame=crate::tests::control_eye_frame(1);
        frame.surface_gaze=Some(surface());
        let original=crate::CalibratedDisplay {restored:false,eye:0,
            gaze_affine_input: crate::GazeAffineInput::ProjectedDirection,
            segmentation_mode:frame.segmentation_mode,sam_prompt_generation:Some(7),
            gaze_authority_generation:40,sign_epoch:7,
            plane:VirtualDisplayPlane::development_default(),
            gaze_affine:crate::GazeAffine{x:[1.0,0.1,0.4],y:[0.2,1.0,0.6]}};
        let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let path=std::path::PathBuf::from(format!("outputs/settings-tests/gaze-{}-{stamp}.json",std::process::id()));
        let recording=path.with_extension("tar");
        original.save_accepted(&path,Some(&recording)).unwrap();
        let restored=crate::CalibratedDisplay::load(&path).unwrap();
        let archive=recording.with_extension("gaze-calibration.json");
        assert_eq!(crate::CalibratedDisplay::load(&archive).unwrap(),restored);
        let mut replacement=original;
        replacement.gaze_affine.x[2]+=0.1;
        replacement.save_accepted(&path,Some(&path.with_extension("next.tar"))).unwrap();
        assert_eq!(crate::CalibratedDisplay::load(&archive).unwrap(),restored,
            "a replacement fit must preserve the previous recording's calibration");
        assert_eq!(crate::CalibratedDisplay::load(&path).unwrap().gaze_affine,replacement.gaze_affine);
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(archive).unwrap();
        std::fs::remove_file(recording.with_extension("next.gaze-calibration.json")).unwrap();
        for (actual,expected) in restored.plane.center_inches.into_iter()
            .chain(restored.plane.right_axis).chain(restored.plane.down_axis)
            .chain([restored.plane.width_inches,restored.plane.height_inches])
            .zip(original.plane.center_inches.into_iter().chain(original.plane.right_axis)
                .chain(original.plane.down_axis).chain([original.plane.width_inches,original.plane.height_inches])) {
            assert!((actual-expected).abs()<1e-12);
        }
        assert_eq!(restored.gaze_affine,original.gaze_affine);
        frame.gaze_authority_generation=900;
        frame.gaze_authority_sam_prompt_generation=Some(100);
        assert!(original.for_frame(0,Some(&frame)).is_some(),
            "a completed fit must resume after a tracking reset without requiring an app restart");
        assert!(restored.for_frame(0,Some(&frame)).is_some());
        assert!(restored.for_frame(1,Some(&frame)).is_none());
        frame.surface_gaze.as_mut().unwrap().sign_resolved=false;
        assert!(restored.for_frame(0,Some(&frame)).is_none());
        let mut bad=original.json();bad["gaze_affine"]["screen_x"]=serde_json::json!([1,2]);
        assert!(crate::CalibratedDisplay::from_json(&bad).is_err());
    }

    #[test]
    fn calibrated_screen_mapping_is_shared_between_mask_detectors() {
        let mut frame=crate::tests::control_eye_frame(1);
        frame.eye_identity_present=true;
        frame.virtual_contact_surface_gaze=Some(surface());
        frame.gaze_authority_sam_prompt_generation=Some(3);
        frame.sam31_proposal_masks=Some(std::sync::Arc::new(crate::sam31_outer::ProposalMasks {
            source_timestamp_ns:100,prompt_generation:3,..Default::default()
        }));
        let saved=crate::CalibratedDisplay {restored:true,eye:0,
            segmentation_mode:SegmentationMode::Sam31,sam_prompt_generation:Some(7),
            gaze_authority_generation:40,sign_epoch:7,plane:VirtualDisplayPlane::nominal(),
            gaze_affine_input:crate::GazeAffineInput::DisplayIntersection,
            gaze_affine:crate::GazeAffine {x:[0.97,0.03,-0.01],y:[0.01,1.0,-0.02]}};
        let expected=saved.target(surface().relative_gaze).unwrap();
        for method in [SegmentationMode::Sam31,SegmentationMode::EyeStudent] {
            frame.segmentation_mode=method;
            let calibration=saved.for_frame(0,Some(&frame)).expect("same camera-frame ray mapping");
            let ray=crate::gaze_output_direction(&frame).unwrap();
            assert_eq!(crate::display_gaze_target(ray,Some(calibration),VirtualDisplayPlane::nominal()),Some(expected));
            assert!(saved.for_frame(1,Some(&frame)).is_none(),"eye-specific mapping stays eye-specific");
        }
        frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_resolved=false;
        assert!(saved.for_frame(0,Some(&frame)).is_none());
        assert!(crate::gaze_output_direction(&frame).is_none());
        frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_resolved=true;
        frame.gaze_policy_error=Some("paused: changed gaze settings");
        assert!(saved.for_frame(0,Some(&frame)).is_none());
    }

    #[test]
    fn cursor_ray_does_not_require_the_illustrative_contact_surface() {
        let mut frame=crate::tests::control_eye_frame(1);
        frame.eye_identity_present=true;
        frame.surface_gaze=Some(surface());
        frame.virtual_contact_surface_gaze=Some(surface());
        frame.gaze_authority_sam_prompt_generation=Some(3);
        frame.sam31_proposal_masks=Some(std::sync::Arc::new(crate::sam31_outer::ProposalMasks {
            source_timestamp_ns:100,prompt_generation:3,..Default::default()
        }));
        let trace=crate::recording_trace::Hub::default();
        trace.raw_arrived(serde_json::json!({"roi_id":1,"sensor_timestamp_ns":"100",
            "stream_epoch":"cursor-ray-fixture"}),Instant::now(),1);
        for method in [SegmentationMode::Native,SegmentationMode::Sam31,SegmentationMode::EyeStudent] {
            frame.segmentation_mode=method;
            assert!(crate::virtual_contact_pose(&frame).is_none(),"fixture has no renderable cap");
            let gaze=authorized_gaze(&frame,3,&trace).expect("published ray can drive the cursor");
            assert_eq!(gaze.direction,surface().relative_gaze);
            let plane=VirtualDisplayPlane::nominal();
            assert_eq!(crate::display_gaze_target(gaze.direction,None,plane),plane.target(surface().relative_gaze));
        }
        frame.presentation_pivot_held=true;
        assert_eq!(authorized_gaze(&frame,3,&trace).unwrap_err(),"paused: held contact is not a fresh observation");
        frame.presentation_pivot_held=false;
        std::sync::Arc::make_mut(frame.sam31_proposal_masks.as_mut().unwrap()).source_timestamp_ns=99;
        assert_eq!(authorized_gaze(&frame,3,&trace).unwrap_err(),"paused: SAM source or prompt mismatch");
    }

    #[test]
    fn pointer_reuses_monitor_affine_and_calibration_basis_checks() {
        let mut frame = crate::tests::control_eye_frame(1);
        frame.surface_gaze = Some(surface());
        let plane = crate::VirtualDisplayPlane::development_default();
        let cal = crate::CalibratedDisplay {
            gaze_affine_input: crate::GazeAffineInput::ProjectedDirection,
            restored: false,
            eye: 0,
            segmentation_mode: frame.segmentation_mode,
            sam_prompt_generation: frame.gaze_authority_sam_prompt_generation,
            gaze_authority_generation: frame.gaze_authority_generation,
            sign_epoch: 7,
            plane,
            gaze_affine: crate::GazeAffine {
                x: [0.0, 0.0, 0.4],
                y: [0.0, 0.0, 0.6],
            },
        };
        // Use a ray which really hits the current plane in front of the eye.
        let ray = crate::RelativeGazeVector::from_projected(
            plane.center_inches[0] / plane.distance_inches(),
            plane.center_inches[1] / plane.distance_inches(),
        )
        .unwrap();
        assert!(plane.target(ray).is_some());
        assert!(cal.target(ray).is_some());
        assert_eq!(
            crate::display_gaze_target(ray, None, plane),
            plane.target(ray)
        );
        assert_eq!(
            crate::display_gaze_target(ray, Some(cal), plane),
            cal.target(ray)
        );
        assert!(cal.for_frame(0, Some(&frame)).is_some());
        assert!(cal.for_frame(1, Some(&frame)).is_none());
        frame.surface_gaze.as_mut().unwrap().sign_epoch += 1;
        assert!(cal.for_frame(0, Some(&frame)).is_some());
        frame.surface_gaze.as_mut().unwrap().sign_resolved = false;
        assert!(cal.for_frame(0, Some(&frame)).is_none());
        frame.surface_gaze.as_mut().unwrap().sign_resolved = true;
        frame.gaze_authority_generation += 1;
        assert!(cal.for_frame(0, Some(&frame)).is_some(),
            "fresh signed gaze may reuse the completed map after reacquisition");
    }

    #[test]
    fn sam_calibration_keeps_its_affine_for_current_camera_frame_rays() {
        let mut frame = crate::tests::control_eye_frame(1);
        frame.segmentation_mode = SegmentationMode::Sam31;
        frame.gaze_authority_sam_prompt_generation = Some(3);
        frame.virtual_contact_surface_gaze = Some(surface());
        let cal = crate::CalibratedDisplay {
            gaze_affine_input: crate::GazeAffineInput::ProjectedDirection,
            restored: false,
            eye: 0,
            segmentation_mode: SegmentationMode::Sam31,
            sam_prompt_generation: Some(3),
            gaze_authority_generation: frame.gaze_authority_generation,
            sign_epoch: 7,
            plane: crate::VirtualDisplayPlane::nominal(),
            gaze_affine: crate::GazeAffine {
                x: [1.0, 0.0, 0.5],
                y: [0.0, 1.0, 0.5],
            },
        };
        for epoch in [7, 8, 12] {
            frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_epoch = epoch;
            frame.gaze_authority_generation = epoch * 100;
            let active = cal.for_frame(0, Some(&frame)).unwrap();
            assert_eq!(active.sign_epoch, 7, "retain training provenance");
            for feature in [(0.1, -0.2), (-0.1, 0.2)] {
                let ray = crate::RelativeGazeVector::from_projected(feature.0, feature.1).unwrap();
                assert_eq!(active.target(ray), Some(cal.gaze_affine.map(feature)));
            }
        }
        assert!(cal.for_frame(0, None).is_none());
        assert!(cal.for_frame(1, Some(&frame)).is_none());
        frame.gaze_authority_sam_prompt_generation = Some(4);
        assert!(cal.for_frame(0, Some(&frame)).is_some(),
            "session-local prompt counters do not invalidate a completed map");
        frame.gaze_policy_error = Some("paused: waiting for global gaze prompt");
        assert!(cal.for_frame(0, Some(&frame)).is_none(),
            "persisted calibration cannot bypass current source-policy checks");
        frame.gaze_policy_error = None;
        frame.gaze_authority_sam_prompt_generation = Some(3);
        frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_resolved = false;
        assert!(cal.for_frame(0, Some(&frame)).is_none());
        frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_resolved = true;
        frame.segmentation_mode = SegmentationMode::Native;
        assert!(cal.for_frame(0, Some(&frame)).is_none());
        frame.surface_gaze=Some(surface());
        assert!(cal.for_frame(0,Some(&frame)).is_some(),
            "a different provider can reuse the mapping once it publishes its own signed ray");
    }

    #[test]
    fn mouse_off_is_available_during_camera_lease_without_toggling_laser() {
        let shared = std::sync::Arc::new(std::sync::Mutex::new(crate::SharedState::default()));
        crate::handle_control_command("LEASE CLAIM mouse-safety-test 5000", &shared);
        for command in ["MOUSE OUTPUT STATUS", "MOUSE OUTPUT OFF"] {
            let response: serde_json::Value =
                serde_json::from_str(&crate::handle_control_command(command, &shared)).unwrap();
            assert_eq!(response["ok"], true);
            assert_eq!(response["mouse_output"]["enabled"], false);
        }
        let state = shared.lock().unwrap();
        assert!(state.control_lease.is_some());
        assert!(!state.eye_laser_enabled);
        assert!(state.ui_action.is_none());
    }

    #[test]
    fn missing_eye_or_surface_cannot_drive_mouse() {
        let mut frame = crate::tests::control_eye_frame(1);
        frame.eye_identity_present = false;
        assert_eq!(source(&frame, 0).unwrap_err(), "paused: eye not present");
        frame.eye_identity_present = true;
        frame.surface_gaze = None;
        frame.virtual_contact_surface_gaze = None;
        assert_eq!(source(&frame, 0).unwrap_err(), "paused: no gaze surface");
    }

    #[test]
    fn shared_gaze_evidence_preserves_policy_sign_and_source_clock_rejections() {
        let trace = crate::recording_trace::Hub::default();
        let mut frame = crate::tests::control_eye_frame(1);
        frame.eye_identity_present = true;
        frame.segmentation_mode = SegmentationMode::Native;
        frame.surface_gaze = Some(surface());
        frame.gaze_policy_error = Some("paused: waiting for global gaze settings");
        assert_eq!(authorized_gaze(&frame, 0, &trace).unwrap_err(),
            "paused: waiting for global gaze settings");
        frame.gaze_policy_error = None;
        frame.surface_gaze.as_mut().unwrap().sign_resolved = false;
        assert_eq!(authorized_gaze(&frame, 0, &trace).unwrap_err(),
            "paused: unresolved gaze sign");
        frame.surface_gaze.as_mut().unwrap().sign_resolved = true;
        assert_eq!(authorized_gaze(&frame, 0, &trace).unwrap_err(),
            "paused: unknown or ambiguous source clock");
        for epoch in ["first", "reconnected"] {
            trace.raw_arrived(serde_json::json!({"roi_id":1,
                "sensor_timestamp_ns":"100", "stream_epoch":epoch}), Instant::now(), 1);
        }
        assert_eq!(authorized_gaze(&frame, 0, &trace).unwrap_err(),
            "paused: unknown or ambiguous source clock");
    }
}
