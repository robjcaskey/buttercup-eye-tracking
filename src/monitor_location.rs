//! Persistent physical monitor pose. Gaze-provider/sign epochs and the
//! eye-specific affine are intentionally session-local, never restored here.
use crate::gaze_target_solver::{display_plane_geometry_plausible, VirtualDisplayPlane};
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_PATH: &str = "outputs/settings/monitor-location.json";

pub(crate) fn plane_json(p: VirtualDisplayPlane) -> Value {
    json!({"center_inches":p.center_inches,"right_axis":p.right_axis,"down_axis":p.down_axis,
        "width_inches":p.width_inches,"height_inches":p.height_inches})
}

fn valid(p: VirtualDisplayPlane) -> bool {
    p.width_inches.is_finite()
        && p.height_inches.is_finite()
        && (1.0..=100.0).contains(&p.width_inches)
        && (1.0..=100.0).contains(&p.height_inches)
        && display_plane_geometry_plausible(p)
}

fn same_pose(a: VirtualDisplayPlane, b: VirtualDisplayPlane) -> bool {
    a.center_inches
        .into_iter()
        .chain(a.right_axis)
        .chain(a.down_axis)
        .chain([a.width_inches, a.height_inches])
        .zip(
            b.center_inches
                .into_iter()
                .chain(b.right_axis)
                .chain(b.down_axis)
                .chain([b.width_inches, b.height_inches]),
        )
        .all(|(a, b)| (a - b).abs() < 1e-10)
}

pub(crate) fn parse_plane(v: &Value) -> Result<VirtualDisplayPlane, String> {
    let number = |name: &str| {
        v[name]
            .as_f64()
            .ok_or_else(|| format!("missing monitor {name}"))
    };
    let axis = |name: &str| -> Result<[f64; 3], String> {
        let a = v[name]
            .as_array()
            .filter(|a| a.len() == 3)
            .ok_or_else(|| format!("invalid monitor {name}"))?;
        Ok([a[0].as_f64(), a[1].as_f64(), a[2].as_f64()].map(|v| v.unwrap_or(f64::NAN)))
    };
    let p = VirtualDisplayPlane {
        center_inches: axis("center_inches")?,
        right_axis: axis("right_axis")?,
        down_axis: axis("down_axis")?,
        width_inches: number("width_inches")?,
        height_inches: number("height_inches")?,
    };
    valid(p)
        .then_some(p)
        .ok_or_else(|| "invalid physical monitor pose".into())
}

#[derive(Clone, Debug)]
pub(crate) struct MonitorLocation {
    pub default_plane: VirtualDisplayPlane,
    /// Latest accepted calibration survives exiting M; offering it never saves.
    pub candidate: Option<VirtualDisplayPlane>,
    pub saved: bool,
    pub status: String,
    path: PathBuf,
}
impl Default for MonitorLocation {
    fn default() -> Self {
        Self {
            default_plane: VirtualDisplayPlane::development_default(),
            candidate: None,
            saved: false,
            status: "BUILT-IN MONITOR POSE".into(),
            path: DEFAULT_PATH.into(),
        }
    }
}
impl MonitorLocation {
    pub fn load(path: impl AsRef<Path>) -> Self {
        let mut state = Self {
            path: path.as_ref().to_path_buf(),
            ..Self::default()
        };
        let result = (|| {
            let data = std::fs::read(&state.path).map_err(|e| e.to_string())?;
            let v: Value = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
            if v["schema"] != "buttercup-monitor-location-v1" {
                return Err("unsupported monitor location version".into());
            }
            parse_plane(&v["pose"])
        })();
        match result {
            Ok(p) => {
                state.default_plane = p;
                state.saved = true;
                state.status = "SAVED MONITOR POSE - GAZE UNCALIBRATED".into();
            }
            Err(e) if state.path.exists() => {
                state.status = format!("MONITOR LOAD FAILED: {e}");
                eprintln!("{}", state.status);
            }
            Err(_) => {}
        }
        state
    }
    pub fn offer(&mut self, plane: VirtualDisplayPlane) {
        if valid(plane) && self.candidate != Some(plane) {
            self.candidate = Some(plane);
            self.status = if self.saved && self.default_plane == plane {
                "MONITOR POSE ALREADY SAVED"
            } else {
                "NEW MONITOR POSE IS SESSION ONLY - SAVE AFTER REVIEW"
            }
            .into();
        }
    }
    pub fn effective_plane(&self) -> VirtualDisplayPlane {
        self.candidate.unwrap_or(self.default_plane)
    }
    pub fn unsaved_candidate(&self) -> bool {
        self.candidate
            .is_some_and(|p| !same_pose(p, self.default_plane) || !self.saved)
    }
    pub fn save(&mut self) -> Result<(), String> {
        let plane = self.candidate.unwrap_or(self.default_plane);
        if !valid(plane) {
            return Err("invalid monitor pose; previous default preserved".into());
        }
        let value = json!({"schema":"buttercup-monitor-location-v1","pose":plane_json(plane),
            "coordinate_frame":"eye-relative inches; camera-right, camera-down, toward-camera",
            "scope":"physical pose only; no affine, provider, sign epoch or calibration validity"});
        write_json_atomic(&self.path, &value)?;
        self.default_plane = plane;
        self.saved = true;
        self.status = "MONITOR LOCATION SAVED AS STARTUP DEFAULT".into();
        Ok(())
    }
    pub fn snapshot(&self) -> Value {
        json!({"path":self.path,"saved":self.saved,"status":self.status,
            "default_pose":plane_json(self.default_plane),"latest_accepted_pose":self.candidate.map(plane_json),
            "session_pose":plane_json(self.effective_plane()),
            "unsaved_candidate":self.unsaved_candidate()})
    }
}

/// Same-directory atomic replacement: a failed write cannot truncate the old
/// settings. The temporary filename is exclusive and never follows a symlink.
pub(crate) fn write_json_atomic(path: &Path, value: &Value) -> Result<(), String> {
    let parent = path.parent().ok_or("settings path has no parent")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let temp = parent.join(format!(".monitor-{}-{stamp}.tmp", std::process::id()));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|e| e.to_string())?;
    let result = (|| {
        f.write_all(&serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&temp, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepted_calibration_is_session_only_until_explicit_save() {
        let dir = std::env::temp_dir().join(format!(
            "buttercup-monitor-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = dir.join("pose.json");
        let mut s = MonitorLocation::load(&path);
        let old = s.default_plane;
        let mut candidate = old;
        candidate.center_inches[0] += 1.0;
        s.offer(candidate);
        assert_eq!(s.default_plane, old);
        assert!(!path.exists());
        assert_eq!(s.effective_plane(), candidate);
        s.save().unwrap();
        assert!(same_pose(
            MonitorLocation::load(&path).default_plane,
            candidate
        ));
        let mut next = candidate;
        next.center_inches[0] += 1.0;
        s.offer(next);
        assert!(same_pose(
            MonitorLocation::load(&path).default_plane,
            candidate
        ));
        let mut invalid = next;
        invalid.right_axis = [0.0; 3];
        s.offer(invalid);
        assert_eq!(s.candidate, Some(next));
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }
    #[test]
    fn invalid_pose_and_unsupported_dimensions_are_rejected() {
        let mut v = plane_json(VirtualDisplayPlane::development_default());
        assert!(parse_plane(&v).is_ok());
        v["width_inches"] = json!(-1);
        assert!(parse_plane(&v).is_err());
        v = plane_json(VirtualDisplayPlane::development_default());
        v["center_inches"] = json!([0, 0, -24]);
        assert!(parse_plane(&v).is_err());
    }
}
