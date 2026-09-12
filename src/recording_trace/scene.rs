//! Export existing model outputs, never infer metric camera/eye poses to fill a
//! visualization. The shared frame is the viewer's legacy eye-relative frame.
use super::Hub;
use crate::geometry::{add3, cross3, dot3, scale3, sub3};
use crate::{EyeFrame, RelativeGazeVector, SharedState, VirtualDisplayPlane};
use serde_json::{Value, json};
use std::io::Read;
use std::sync::Mutex;

pub(crate) const FRAME: &str = "viewer-eye-reference-inches-v1";

/// A change-cached host metadata snapshot; no filesystem reads on normal redraws.
#[derive(Default)]
pub(crate) struct HostMetadata {
    display_key: String,
    physical_size: Option<(f64, f64)>,
    intrinsics_key: String,
    intrinsics: Value,
}

impl HostMetadata {
    pub fn observe(
        &mut self,
        window: &winit::window::Window,
        checker: &crate::checkerboard_calibration::StatusSnapshot,
    ) -> (Value, Value) {
        let monitor = window.current_monitor();
        let name = monitor.as_ref().and_then(|m| m.name());
        let key = format!("{name:?}");
        if key != self.display_key {
            self.physical_size = name
                .as_deref()
                .and_then(crate::display_pose_wireframe::monitor_dimensions_inches);
            self.display_key = key;
        }
        let key = format!(
            "{}:{}:{:?}:{:?}:{}",
            checker.generation,
            checker.accepted_views,
            checker.focus_position,
            checker.output,
            checker.calibrated
        );
        if key != self.intrinsics_key {
            self.intrinsics_key = key;
            self.intrinsics = checker
                .output
                .as_ref()
                .filter(|_| checker.calibrated)
                .and_then(|p| {
                    let mut bytes = Vec::new();
                    std::fs::File::open(p.join("camera-intrinsics-checkerboard.json"))
                        .ok()?
                        .take(64 * 1024 + 1)
                        .read_to_end(&mut bytes)
                        .ok()?;
                    Some(bytes)
                })
                .filter(|b| b.len() <= 64 * 1024)
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .filter(|v| v["calibrated"] == true)
                .unwrap_or(Value::Null);
        }
        let display = json!({"id": name, "identity_source": "host-monitor-name-not-verified-saved-calibration-binding",
            "monitor_pixels": monitor.as_ref().map(|m| [m.size().width, m.size().height]),
            "monitor_desktop_origin_px": monitor.as_ref().map(|m| [m.position().x, m.position().y]),
            "viewport_desktop_origin_px": window.inner_position().ok().map(|p| [p.x, p.y]),
            "viewport_origin_unavailable_on_some_wayland_hosts": true,
            "window_scale_factor": window.scale_factor(),
            "monitor_scale_factor": monitor.as_ref().map(|m| m.scale_factor()),
            "physical_size_inches": self.physical_size,
            "physical_size_source": self.physical_size.map(|_| "selected-monitor-edid-dtd"),
            "compositor_rotation_degrees": null,
            "axes": "UV top-left [0,0] to bottom-right [1,1]; x right, y down",
            "pixel_centers": "UV * [width-1,height-1]; drawn centers rounded; predictions not clamped",
            "visibility": "submitted-to-window; physical-occlusion-and-scanout-not-measured"});
        let camera = json!({"sensor_size_px": [crate::SENSOR_WIDTH, crate::SENSOR_HEIGHT],
            "optical_center_scene": null, "camera_to_scene": null,
            "orientation_basis": "scene axes are defined from camera-image axes; optical pose is not solved",
            "pose_provenance": "unavailable", "intrinsics": self.intrinsics,
            "intrinsics_used_for_this_gaze": false,
            "sensor_to_host_clock_model": null,
            "target_exposure_alignment": "host-arrival/submission-approximate; offset/drift/scanout bounds unavailable"});
        (display, camera)
    }
}

pub fn geometry(plane: VirtualDisplayPlane, reference_eye: usize, camera: Value) -> Value {
    let right = scale3(plane.right_axis, plane.width_inches);
    let down = scale3(plane.down_axis, plane.height_inches);
    let corner = sub3(
        sub3(plane.center_inches, scale3(right, 0.5)),
        scale3(down, 0.5),
    );
    json!({"frame": {"id": FRAME, "units": "inches", "handedness": "left",
        "axes": ["camera-right", "camera-down", "toward-camera"],
        "origin": "fixed virtual ray origin of reference eye; not a measured camera-space pivot",
        "reference_roi_id": reference_eye + 1, "origin_provenance": "assumed",
        "head_translation_in_metric_scene": "not-estimated",
        "transform_convention": "row-major matrices multiplying column vectors; points include translation",
        "right_handed_export_conversion": "reflect Y for right/up/toward-camera; multiply inches by 25.4 for mm"},
        "camera": camera,
        "monitor": {"center": plane.center_inches, "right_axis": plane.right_axis,
            "down_axis": plane.down_axis, "size_inches": [plane.width_inches, plane.height_inches],
            "corners": [corner, add3(corner,right), add3(add3(corner,right),down), add3(corner,down)],
            "uv_to_scene_3x3": [[right[0],down[0],corner[0]], [right[1],down[1],corner[1]], [right[2],down[2],corner[2]]],
            "pose_provenance": "estimated-or-preset; see effective mapping kind",
            "pose_uncertainty": null, "room-level_orientation_measured": false},
        "field_provenance": {"contact_axis": "estimated conic/contact-surface normal; not kappa-calibrated visual axis",
            "prediction_ready": "completed host EyeFrame availability; not CUDA completion or physical display time",
            "freshness": "scene revision advances only on change; repeated analysis source keys are not new observations",
            "no_new_exposure_threshold_ms": 750,
            "visual_axis": "unavailable; affine screen mapping is not a solved 3D visual ray",
            "local_rotation_center_and_radius": "estimated model-space sensor pixels; never metric eye position/radius",
            "scale_hint": "MediaPipe iris projection plus assumed 12mm limbus; heuristic bounds, not calibrated probability",
            "other_eye_origin": "unavailable; no nominal IPD inserted",
            "binocular": "source-keyed conditional joint conics are exported separately in fused; no transform to this legacy monitor frame is asserted"}})
}

/// Reuses the same plane/ray convention as VirtualDisplayPlane::target, while
/// exposing why a hit is absent. It does not create a missing eye origin.
pub fn intersection(
    plane: VirtualDisplayPlane,
    direction: Option<RelativeGazeVector>,
    origin_known: bool,
) -> Value {
    if !origin_known {
        return json!({"status": "origin-unavailable", "uv": null, "point": null});
    }
    let Some(gaze) = direction else {
        return json!({"status": "direction-unavailable", "uv": null, "point": null});
    };
    let direction = gaze.as_array();
    let normal = cross3(plane.right_axis, plane.down_axis);
    let denominator = dot3(direction, normal);
    if !denominator.is_finite() || denominator.abs() < 1e-8 {
        return json!({"status": "parallel", "uv": null, "point": null});
    }
    let range = dot3(plane.center_inches, normal) / denominator;
    if !range.is_finite() || range <= 0.0 {
        return json!({"status": "behind-eye", "uv": null, "point": null});
    }
    let Some(uv) = plane.target(gaze) else {
        return json!({"status": "invalid", "uv": null, "point": null});
    };
    json!({"status": if (0.0..=1.0).contains(&uv.0) && (0.0..=1.0).contains(&uv.1) { "on-screen" } else { "off-screen" },
        "uv": uv, "point": scale3(direction,range), "range_inches": range})
}

pub struct Input {
    pub plane: VirtualDisplayPlane,
    pub reference_eye: usize,
    pub selected_ray: Option<RelativeGazeVector>,
    pub selected_prediction: Option<(f64, f64)>,
    pub calibration: Value,
    pub camera: Value,
}

pub fn snapshot(
    eyes: &[Option<EyeFrame>; 2],
    input: Input,
    shared: &Mutex<SharedState>,
    hub: &Hub,
) -> Value {
    let enabled = shared
        .lock()
        .map(|s| [true, s.second_roi_enabled])
        .unwrap_or([false, false]);
    let transport = hub.transport_state();
    let now = hub.stamp_json()["host_monotonic_ns"]
        .as_str()
        .and_then(|s| s.parse::<u128>().ok());
    let mut states = Vec::new();
    let samples: Vec<_> = (0..2).map(|index| {
        let frame = eyes[index].as_ref();
        let surface = frame.and_then(crate::mouse_gaze_surface);
        let pose = frame.and_then(crate::virtual_contact_pose);
        let resident = transport["region"]["active_mask"].as_u64().map(|m| m & (1<<index) != 0);
        let arrival = frame.and_then(|f| f.recording_clock["host_arrival_monotonic_ns"].as_str()).and_then(|s| s.parse::<u128>().ok());
        let stale = now.zip(arrival).is_some_and(|(now,at)| now.saturating_sub(at) > 750_000_000);
        let state = if !enabled[index] { "intentionally-disabled" }
            else if transport["global_capture_active"] == true { "global-sensor-capture" }
            else if resident == Some(false) { "temporarily-evicted-from-sensor-band" }
            else if frame.is_none() { "no-analysis-frame" }
            else if stale { "no-new-exposure-750ms" }
            else if !frame.is_some_and(|f| f.eye_identity_present) { "not-detected" }
            else if surface.is_none() { "analysis-surface-unavailable" }
            else if !surface.is_some_and(|s| s.sign_resolved) { "unresolved-sign" }
            else if pose.is_none() { "no-admissible-contact" }
            else { "available" };
        states.push(json!({"roi_id": index+1, "enabled": enabled[index], "sensor_resident": resident, "state": state}));
        let reference = index == input.reference_eye;
        let usable = enabled[index] && !stale && resident != Some(false) && transport["global_capture_active"] != true;
        let contact_surface=frame.and_then(|f|if f.segmentation_mode.uses_mask_geometry() {f.virtual_contact_surface_gaze} else {f.surface_gaze});
        let axis = contact_surface.map(|s| s.relative_gaze.as_array());
        let source = hub.source_reference(index as u32 + 1, surface.and_then(|s| s.source_timestamp_ns));
        let analysis_pending = transport["latest_ingress"][index]["source_key"].as_object().is_some()
            && frame.is_none_or(|f|transport["latest_ingress"][index]["source_key"] != f.recording_clock["source_key"]);
        let admitted = if reference {input.selected_ray} else {pose.map(|p| p.relative_gaze)}
            .filter(|_| usable && surface.is_some_and(|s| s.sign_resolved));
        json!({"roi_id": index+1, "role": if index==0 {"subject-right"} else {"subject-left"},
            "state": state, "raw_clock": frame.map(|f| &f.recording_clock),
            "prediction_ready": frame.map(|f| &f.recording_ready),
            "newer_ingress_than_displayed_analysis": analysis_pending,
            "freshness": if pose.is_some_and(|p|p.authority==crate::VirtualContactAuthority::MotionHeld) {"motion-held"}
                else if surface.is_none(){"unavailable"}
                else if frame.zip(surface).is_some_and(|(f,s)|s.source_timestamp_ns==Some(f.timestamp_ns)){"same-exposure; deduplicate-analysis-source-key"}
                else {"asynchronous-older-source; deduplicate-analysis-source-key"},
            "analysis_source": source,
            "basis": frame.map(|f| json!({"authority_generation": f.gaze_authority_generation.to_string(),
                "prompt_generation": f.gaze_authority_sam_prompt_generation.map(|n|n.to_string()),
                "sign_epoch": surface.map(|s|s.sign_epoch.to_string()), "sign_resolved":surface.map(|s|s.sign_resolved)})),
            "contact_axis_candidate": axis, "admitted_contact_axis": pose.filter(|_|usable&&contact_surface.is_some_and(|s|s.sign_resolved)).map(|p|p.relative_gaze.as_array()),
            "gaze_axis_candidate": surface.map(|s|s.relative_gaze.as_array()),
            "joint_conics":frame.filter(|f|f.joint_gaze_active).map(crate::joint_gaze_live::json),
            "contact_authority": pose.map(|p|p.authority.label()), "visual_axis": null,
            "held_geometry": pose.is_some_and(|p|p.authority == crate::VirtualContactAuthority::MotionHeld),
            "same_source_as_displayed_exposure": frame.zip(surface).map(|(f,s)| s.source_timestamp_ns==Some(f.timestamp_ns)),
            "ray_origin": if reference {json!([0.0,0.0,0.0])} else {Value::Null},
            "origin_provenance": if reference {"assumed-fixed-reference"} else {"unavailable"},
            "eye_to_scene": if reference {json!([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])} else {Value::Null},
            "eye_to_scene_semantics": "coordinate basis only; anatomical eye torsion not solved",
            "metric_radius": null,
            "monitor_intersection": intersection(input.plane, admitted, reference),
            "predicted_screen_uv": if reference {input.selected_prediction} else {None},
            "screen_prediction_kind": if reference {"selected-eye-current-mapping"} else {"unavailable-no-shared-origin-or-eye-specific-mapping"},
            "local_geometry": frame.map(|f| json!({"units":"sensor-pixels", "rotation_center_xy_roi": f.projected_rotation_center,
                "rotation_center_z_model": f.projected_rotation_center_z, "sphere_radius":f.projected_sphere_radius})),
            "scale_hint": frame.and_then(|f|f.centimeter_scale).map(|s| json!({"pixels_per_10mm":s.estimate_px,
                "bounds_px_per_10mm":[s.minimum_px,s.maximum_px],"reacquisitions":s.reacquisition_count})),
        })
    }).collect();
    let fused=eyes.iter().flatten().filter(|f|f.joint_gaze_active&&f.joint_conic.is_some())
        .max_by_key(|f|f.joint_conic.as_ref().and_then(|p|p.exposures.iter().flatten().map(|e|e.timestamp_ns).max()))
        .map(crate::joint_gaze_live::json).unwrap_or_else(||json!({"target":null,"status":"joint-conics-inactive-or-unavailable"}));
    json!({"geometry": geometry(input.plane,input.reference_eye,input.camera), "calibration": input.calibration,
        "eyes":samples,"roi_states":states,"fused":fused})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exported_uv_transform_and_intersections_match_the_actual_plane_in_inches() {
        for plane in [
            VirtualDisplayPlane::nominal(),
            VirtualDisplayPlane::development_default(),
        ] {
            let exported = geometry(plane, 0, Value::Null);
            assert_eq!(exported["frame"]["handedness"], "left");
            assert_eq!(exported["frame"]["units"], "inches");
            for uv in [[0.5, 0.5], [0.1, 0.7], [-0.3, 1.4]] {
                let m = &exported["monitor"]["uv_to_scene_3x3"];
                let point: [f64; 3] = std::array::from_fn(|r| {
                    m[r][0].as_f64().unwrap() * uv[0]
                        + m[r][1].as_f64().unwrap() * uv[1]
                        + m[r][2].as_f64().unwrap()
                });
                let norm = dot3(point, point).sqrt();
                let gaze =
                    RelativeGazeVector::from_projected(point[0] / norm, point[1] / norm).unwrap();
                let hit = intersection(plane, Some(gaze), true);
                for i in 0..2 {
                    assert!((hit["uv"][i].as_f64().unwrap() - uv[i]).abs() < 1e-8);
                }
                for i in 0..3 {
                    assert!((hit["point"][i].as_f64().unwrap() - point[i]).abs() < 1e-8);
                }
                assert!((hit["range_inches"].as_f64().unwrap() - norm).abs() < 1e-8);
                assert_eq!(
                    hit["status"],
                    if uv[0] < 0.0 {
                        "off-screen"
                    } else {
                        "on-screen"
                    }
                );
                assert_eq!(
                    plane.target(gaze).unwrap(),
                    (
                        hit["uv"][0].as_f64().unwrap(),
                        hit["uv"][1].as_f64().unwrap()
                    )
                );
            }
        }
    }

    #[test]
    fn absent_and_invalid_geometry_never_becomes_a_successful_monitor_hit() {
        let mut plane = VirtualDisplayPlane::nominal();
        let gaze = RelativeGazeVector::from_projected(0.0, 0.0).unwrap();
        assert_eq!(
            intersection(plane, Some(gaze), false)["status"],
            "origin-unavailable"
        );
        assert_eq!(
            intersection(plane, None, true)["status"],
            "direction-unavailable"
        );
        plane.center_inches[2] = -24.0;
        assert_eq!(
            intersection(plane, Some(gaze), true)["status"],
            "behind-eye"
        );
        plane.right_axis = [0.0, 0.0, 1.0];
        assert_eq!(intersection(plane, Some(gaze), true)["status"], "parallel");
    }
}
