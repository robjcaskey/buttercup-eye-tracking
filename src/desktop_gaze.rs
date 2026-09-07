//! Shared gaze input for the independent pointer and window-focus outputs.
//! Hidden windows still get event-loop ticks. Preparation/mapping matches draw.
use crate::mouse_output::{Sample, Source};
use crate::{App, EyeFrame, SegmentationMode, VirtualContactAuthority};
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
    if frame.segmentation_mode == SegmentationMode::Sam31
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

fn current_sample(app: &App) -> Result<Sample, &'static str> {
    // No desktop pointer movement while collecting targets, editing a prompt,
    // or doing non-eye object search. Neither pause toggles J or stops analysis.
    if app.virtual_mouse.is_some() || app.accuracy_requested || app.accuracy_check.is_some() {
        return Err("paused: calibration or accuracy screen");
    }
    if app.sam31_prompt_editor.is_some() {
        return Err("paused: prompt editor");
    }
    let (prompt_generation, plane, trace) = {
        let s = app
            .shared
            .lock()
            .map_err(|_| "paused: viewer state unavailable")?;
        if s.sam31_object_inspection {
            return Err("paused: object search");
        }
        (
            s.sam31_prompt_bundle_generation,
            s.monitor_location.effective_plane(),
            s.recording_trace.clone(),
        )
    };
    let mut frame = app.eyes[app.focus_eye]
        .clone()
        .ok_or("paused: no eye frame")?;
    let source = source(&frame, prompt_generation)?;
    let age = trace
        .source_arrival_age(frame.eye_id, source.timestamp_ns)
        .ok_or("paused: unknown or ambiguous source clock")?;
    crate::prepare_current_contact_frame(app, app.focus_eye, &mut frame, prompt_generation);
    let pose = crate::virtual_contact_pose(&frame).ok_or("paused: no current virtual contact")?;
    if pose.authority == VirtualContactAuthority::MotionHeld {
        return Err("paused: held contact is not a fresh observation");
    }
    let pose=crate::pose_for_cursor(&frame,pose).ok_or("paused: no current joint gaze ray")?;
    let calibration = app
        .calibrated_display
        .and_then(|c| c.for_frame(app.focus_eye, Some(&frame)));
    let target = crate::display_gaze_target(pose, calibration, plane)
        .ok_or("paused: no forward monitor intersection")?;
    Ok(Sample {
        source,
        age,
        target,
    })
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
