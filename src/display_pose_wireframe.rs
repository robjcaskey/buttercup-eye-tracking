//! Presentation-only geometry for the completed eye-to-screen calibration.
//! Inches and axes come from the fitted plane, never the affine cursor map.
use crate::eye_scene_model::RelativeGazeVector;
use crate::gaze_target_solver::VirtualDisplayPlane;
use crate::geometry::{add3, cross3, dot3, norm3, scale3};

/// Physical dimensions from the selected host display, not camera metadata.
pub(crate) fn monitor_dimensions_inches(connector: &str) -> Option<(f64, f64)> {
    let suffix = format!("-{connector}");
    let mut matches = std::fs::read_dir("/sys/class/drm").ok()?.flatten()
        .filter(|entry| entry.file_name().to_string_lossy().ends_with(&suffix))
        .filter_map(|entry| std::fs::read(entry.path().join("edid")).ok())
        .filter_map(|bytes| edid_dimensions_inches(&bytes));
    let dimensions = matches.next()?;
    // Ambiguous connector names across GPUs are not a physical-size measurement.
    matches.next().is_none().then_some(dimensions)
}

fn edid_dimensions_inches(bytes: &[u8]) -> Option<(f64, f64)> {
    if bytes.len() < 128 || bytes[..8] != [0,255,255,255,255,255,255,0]
        || bytes[..128].iter().fold(0u8, |sum,b| sum.wrapping_add(*b)) != 0 { return None; }
    for dtd in bytes[54..126].chunks_exact(18) {
        if dtd[0] == 0 && dtd[1] == 0 { continue; }
        let w = u16::from(dtd[12]) | (u16::from(dtd[14] & 0xf0) << 4);
        let h = u16::from(dtd[13]) | (u16::from(dtd[14] & 0x0f) << 8);
        if w >= 100 && h >= 100 { return Some((f64::from(w)/25.4,f64::from(h)/25.4)); }
    }
    None // Rounded centimeter fields are deliberately not treated as precise.
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RayStatus { Unavailable, OnScreen, OffScreen, ParallelOrBehind }

#[derive(Clone, Debug)]
pub(crate) struct DisplayPoseWireframe {
    pub plane: VirtualDisplayPlane,
    /// TL, TR, BR, BL in physical display coordinates.
    pub corners: [[f64; 3]; 4],
    pub normal_end: [f64; 3],
    pub ray_end: Option<[f64; 3]>,
    pub hit: Option<[f64; 3]>,
    pub ray_status: RayStatus,
    /// Euler factorization Rz(roll) Ry(yaw) Rx(pitch); camera XYZ frame.
    pub pitch_yaw_roll_degrees: Option<[f64; 3]>,
}

impl DisplayPoseWireframe {
    pub fn new(plane: VirtualDisplayPlane, gaze: Option<RelativeGazeVector>) -> Option<Self> {
        if !plane.center_inches.iter().chain(plane.right_axis.iter()).chain(plane.down_axis.iter())
            .all(|x| x.is_finite() && x.abs() <= 10_000.0)
            || !plane.width_inches.is_finite() || !plane.height_inches.is_finite()
            || !(0.01..=1000.0).contains(&plane.width_inches)
            || !(0.01..=1000.0).contains(&plane.height_inches)
            || (norm3(plane.right_axis) - 1.0).abs() > 0.001
            || (norm3(plane.down_axis) - 1.0).abs() > 0.001
            || dot3(plane.right_axis, plane.down_axis).abs() > 0.001
        { return None; }
        let point = |u: f64, v: f64| add3(plane.center_inches,
            add3(scale3(plane.right_axis, (u - 0.5) * plane.width_inches),
                 scale3(plane.down_axis, (v - 0.5) * plane.height_inches)));
        let corners = [point(0.0, 0.0), point(1.0, 0.0), point(1.0, 1.0), point(0.0, 1.0)];
        let normal = cross3(plane.right_axis, plane.down_axis);
        let normal_end = add3(plane.center_inches, scale3(normal, plane.height_inches * 0.25));
        let horizontal = plane.right_axis[0].hypot(plane.right_axis[1]);
        // Near gimbal lock no unique pitch/roll pair exists. Draw the axes
        // instead of printing arbitrary Euler angles as measurements.
        let pitch_yaw_roll_degrees = (horizontal > 1e-5).then(|| [
            plane.down_axis[2].atan2(normal[2]).to_degrees(),
            (-plane.right_axis[2]).atan2(horizontal).to_degrees(),
            plane.right_axis[1].atan2(plane.right_axis[0]).to_degrees(),
        ]);
        let mut scene = Self { plane, corners, normal_end, ray_end: None, hit: None,
            ray_status: RayStatus::Unavailable, pitch_yaw_roll_degrees };
        if let Some(gaze) = gaze.filter(|g| g.is_camera_facing()) {
            let direction = gaze.as_array();
            let extent = (plane.distance_inches() + plane.width_inches.hypot(plane.height_inches)).max(1.0);
            if let Some((u, v)) = plane.target(gaze) {
                let hit = point(u, v);
                let on_screen = (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v);
                scene.ray_status = if on_screen { RayStatus::OnScreen } else { RayStatus::OffScreen };
                scene.hit = on_screen.then_some(hit);
                // Finite segment for an off-screen or nearly parallel ray;
                // do not invent an on-screen hit by clamping screen UV.
                scene.ray_end = Some(scale3(direction, norm3(hit).min(extent * 2.0)));
            } else {
                scene.ray_status = RayStatus::ParallelOrBehind;
                scene.ray_end = Some(scale3(direction, extent));
            }
        }
        Some(scene)
    }

    pub fn point(&self, u: f64, v: f64) -> [f64; 3] {
        add3(self.plane.center_inches,
            add3(scale3(self.plane.right_axis, (u - 0.5) * self.plane.width_inches),
                 scale3(self.plane.down_axis, (v - 0.5) * self.plane.height_inches)))
    }

    /// Fixed oblique orthographic view: no invented camera position. Fit uses
    /// static pose only, so eye motion cannot resize or rotate the diagram.
    pub fn projector(&self, rect: [f64; 4]) -> impl Fn([f64; 3]) -> [f64; 2] {
        let raw = |p: [f64; 3]| [0.8660254 * p[0] + 0.5 * p[2],
            0.1710101 * p[0] + 0.9396926 * p[1] - 0.2961981 * p[2]];
        let points = self.corners.into_iter().chain([[0.0; 3], self.normal_end, [0.0, 0.0, 6.0]]);
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        for p in points.map(raw) { for i in 0..2 { lo[i] = lo[i].min(p[i]); hi[i] = hi[i].max(p[i]); } }
        let scale = (rect[2] / (hi[0] - lo[0]).max(1.0)).min(rect[3] / (hi[1] - lo[1]).max(1.0)) * 0.82;
        let center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
        move |p| { let p = raw(p); [rect[0] + rect[2] * 0.5 + (p[0] - center[0]) * scale,
            rect[1] + rect[3] * 0.5 + (p[1] - center[1]) * scale] }
    }

    /// Presentation-only orbit around the eye/display scene. A static bounding
    /// sphere keeps scale constant across both orbit and gaze changes.
    pub fn orbit_projector(&self, rect: [f64;4], seconds: f64) -> impl Fn([f64;3]) -> [f64;2] {
        let [yaw_offset,pitch_offset] = orbit_offsets(seconds);
        let yaw = (30.0+yaw_offset).to_radians();
        let pitch = (20.0+pitch_offset).to_radians();
        let center = scale3(self.plane.center_inches,0.5);
        let radius = self.corners.into_iter().chain([[0.0;3],self.normal_end])
            .map(|p|norm3(crate::geometry::sub3(p,center))).fold(1.0_f64,f64::max);
        let scale = rect[2].min(rect[3]).max(1.0)*0.44/radius;
        move |p| {
            let p=crate::geometry::sub3(p,center);
            let x=yaw.cos()*p[0]+yaw.sin()*p[2];
            let z=-yaw.sin()*p[0]+yaw.cos()*p[2];
            let y=pitch.cos()*p[1]-pitch.sin()*z;
            [rect[0]+rect[2]*0.5+x*scale,rect[1]+rect[3]*0.5+y*scale]
        }
    }
}

fn orbit_offsets(seconds: f64) -> [f64;2] {
    let t=if seconds.is_finite() { seconds.max(0.0) } else { 0.0 };
    [45.0*(t*std::f64::consts::TAU/24.0).sin(),
     10.0*(t*std::f64::consts::TAU/32.0).sin()]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orbit_is_bounded_slow_and_does_not_change_geometry() {
        assert_eq!(orbit_offsets(0.0),[0.0,0.0]);
        assert!((orbit_offsets(6.0)[0]-45.0).abs()<1e-10);
        assert!((orbit_offsets(18.0)[0]+45.0).abs()<1e-10);
        assert!((orbit_offsets(8.0)[1]-10.0).abs()<1e-10);
        assert!((orbit_offsets(24.0)[1]+10.0).abs()<1e-10);
        let plane=VirtualDisplayPlane::nominal();
        let scene=DisplayPoseWireframe::new(plane,None).unwrap();
        let rect=[10.0,20.0,600.0,400.0];
        let before=scene.orbit_projector(rect,0.0)(scene.corners[0]);
        assert_ne!(before,scene.orbit_projector(rect,6.0)(scene.corners[0]));
        for t in 0..96 {
            let project=scene.orbit_projector(rect,t as f64);
            for p in scene.corners.into_iter().chain([[0.0;3],scene.normal_end]) {
                let p=project(p);
                assert!((10.0..=610.0).contains(&p[0]) && (20.0..=420.0).contains(&p[1]));
            }
        }
        assert_eq!(scene.plane,plane);
        assert_eq!(scene.pitch_yaw_roll_degrees,Some([0.0,0.0,0.0]));
    }
    #[test]
    fn wireframe_edid_dimensions_require_valid_detailed_timing() {
        let mut bytes = [0u8;128];
        bytes[..8].copy_from_slice(&[0,255,255,255,255,255,255,0]);
        bytes[54] = 1;
        bytes[66] = 0x4e; bytes[67] = 0x4d; bytes[68] = 0x21;
        bytes[127] = 0u8.wrapping_sub(bytes[..127].iter().fold(0u8,|a,b|a.wrapping_add(*b)));
        assert_eq!(edid_dimensions_inches(&bytes),Some((590.0/25.4,333.0/25.4)));
        bytes[20] ^= 1;
        assert!(edid_dimensions_inches(&bytes).is_none());
        assert!(edid_dimensions_inches(&bytes[..64]).is_none());
    }
    #[test]
    fn wireframe_uses_physical_dimensions_and_center_ray() {
        let plane = VirtualDisplayPlane::nominal();
        let scene = DisplayPoseWireframe::new(plane, Some(RelativeGazeVector {
            right: 0.0, down: 0.0, toward_camera: 1.0,
        })).unwrap();
        assert_eq!(scene.hit, Some(plane.center_inches));
        assert_eq!(scene.ray_status, RayStatus::OnScreen);
        assert_eq!(scene.pitch_yaw_roll_degrees, Some([0.0, 0.0, 0.0]));
        let diagonal = norm3(crate::geometry::sub3(scene.corners[2], scene.corners[0]));
        assert!((diagonal - 27.0).abs() < 1e-10);
    }
    #[test]
    fn wireframe_offscreen_ray_is_not_clamped_to_monitor() {
        let plane = VirtualDisplayPlane::nominal();
        let scene = DisplayPoseWireframe::new(plane, Some(RelativeGazeVector {
            right: 0.8, down: 0.0, toward_camera: 0.6,
        })).unwrap();
        assert_eq!(scene.ray_status, RayStatus::OffScreen);
        assert_eq!(scene.hit, None);
        let no_gaze = DisplayPoseWireframe::new(plane, None).unwrap();
        assert_eq!(no_gaze.ray_end, None);
        assert_eq!(scene.projector([0.0, 0.0, 600.0, 300.0])(plane.center_inches),
            no_gaze.projector([0.0, 0.0, 600.0, 300.0])(plane.center_inches));
    }
    #[test]
    fn wireframe_reports_rotation_and_rejects_invalid_pose() {
        let mut plane = VirtualDisplayPlane::nominal();
        let a = 30_f64.to_radians();
        plane.right_axis = [a.cos(), a.sin(), 0.0];
        plane.down_axis = [-a.sin(), a.cos(), 0.0];
        let scene = DisplayPoseWireframe::new(plane, None).unwrap();
        assert!((scene.pitch_yaw_roll_degrees.unwrap()[2] - 30.0).abs() < 1e-9);
        plane.center_inches[1] = f64::NAN;
        assert!(DisplayPoseWireframe::new(plane, None).is_none());
    }
}
