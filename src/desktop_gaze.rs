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
    let (mouse_generation, focus_generation) = app
        .shared
        .lock()
        .map(|s| {
            (
                s.mouse_output.enabled_generation(),
                s.gaze_focus.enabled_generation(),
            )
        })
        .unwrap_or((None, None));
    if mouse_generation.is_none() && focus_generation.is_none() {
        return;
    }
    // refresh_viewer_frames updates each image only when its source changes.
    let sample = if crate::refresh_viewer_frames(app).is_err() {
        Err("paused: viewer state unavailable")
    } else {
        current_sample(app)
    };
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
    if let Some(reason) = frame.gaze_policy_error { return Err(reason); }
    if !frame.eye_identity_present {
        return Err("paused: eye not present");
    }
    let surface = crate::mouse_gaze_surface(frame).ok_or("paused: no gaze surface")?;
    if !surface.sign_resolved {
        return Err("paused: unresolved gaze sign");
    }
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
    Ok(Source {
        eye: frame.eye_id.saturating_sub(1) as usize,
        authority: frame.gaze_authority_generation,
        sign_epoch: surface.sign_epoch,
        timestamp_ns,
    })
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

fn current_sample(app: &App) -> Result<(Sample, Selection), &'static str> {
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
    let source = source(&frame, selection.policy.prompt_generation)?;
    let age = trace
        .source_arrival_age(frame.eye_id, source.timestamp_ns)
        .ok_or("paused: unknown or ambiguous source clock")?;
    let pose = crate::virtual_contact_pose(&frame).ok_or("paused: no current virtual contact")?;
    if pose.authority == VirtualContactAuthority::MotionHeld {
        return Err("paused: held contact is not a fresh observation");
    }
    let pose=crate::pose_for_cursor(&frame,pose).ok_or("paused: no current joint gaze ray")?;
    let calibration = app
        .calibrated_display
        .and_then(|c| c.for_frame(selection.eye, Some(&frame)));
    let target = crate::display_gaze_target(pose, calibration, selection.plane)
        .ok_or("paused: no forward monitor intersection")?;
    Ok((Sample {
        source,
        age,
        target,
    }, selection))
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
    fn pointer_reuses_monitor_affine_and_calibration_basis_checks() {
        let mut frame = crate::tests::control_eye_frame(1);
        frame.surface_gaze = Some(surface());
        let plane = crate::VirtualDisplayPlane::development_default();
        let cal = crate::CalibratedDisplay {
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
        let pose = crate::VirtualContactPose {
            rotation_center: (0.0, 0.0),
            rotation_center_z: None,
            relative_gaze: ray,
            sphere_radius: Some(100.0),
            authority: VirtualContactAuthority::SamEllipseProvisional,
        };
        assert!(plane.target(ray).is_some());
        assert!(cal.target(ray).is_some());
        assert_eq!(
            crate::display_gaze_target(pose, None, plane),
            plane.target(ray)
        );
        assert_eq!(
            crate::display_gaze_target(pose, Some(cal), plane),
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
        assert!(cal.for_frame(0, Some(&frame)).is_none());
    }

    #[test]
    fn sam_calibration_keeps_its_affine_across_sign_changes_but_not_provider_changes() {
        let mut frame = crate::tests::control_eye_frame(1);
        frame.segmentation_mode = SegmentationMode::Sam31;
        frame.gaze_authority_sam_prompt_generation = Some(3);
        frame.virtual_contact_surface_gaze = Some(surface());
        let cal = crate::CalibratedDisplay {
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
        assert!(cal.for_frame(0, Some(&frame)).is_none());
        frame.gaze_authority_sam_prompt_generation = Some(3);
        frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_resolved = false;
        assert!(cal.for_frame(0, Some(&frame)).is_none());
        frame.virtual_contact_surface_gaze.as_mut().unwrap().sign_resolved = true;
        frame.segmentation_mode = SegmentationMode::Native;
        assert!(cal.for_frame(0, Some(&frame)).is_none());
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
}
