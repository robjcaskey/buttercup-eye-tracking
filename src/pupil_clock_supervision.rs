//! Optical screen-clock supervision for pupil calibration captures.
//!
//! The reflected spatial code identifies the display-code tick at each sensor
//! exposure.  This module reconstructs the fixation target from that identity
//! and the stimulus protocol; it never joins on host packet or compositor
//! timestamps.  The resulting target is supervision/evaluation data, not an
//! image-space pupil label.

use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScreenStimulusProtocol {
    pub code_hz: f64,
    pub warmup_seconds: f64,
    pub horizontal_period_seconds: f64,
    pub vertical_period_seconds: f64,
    pub x_range_normalized: [f64; 2],
    pub y_range_normalized: [f64; 2],
    pub screen_field_degrees: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StimulusPose {
    pub code_index: u64,
    pub elapsed_seconds: f64,
    pub center_normalized: [f64; 2],
    pub velocity_normalized_per_second: [f64; 2],
    pub gaze_tangent: [f64; 2],
    pub gaze_tangent_velocity_per_second: [f64; 2],
    pub moving: bool,
    /// Native ROI-local carrier selected by the whole-image optical clock.
    /// It establishes exposure identity only.  The carrier may be screen
    /// illumination on any visible surface, so it is explicitly *not* a
    /// corneal glint or an anatomical coordinate.
    pub clock_carrier_center: Option<[f64; 2]>,
    pub clock_carrier_extent: Option<[f64; 2]>,
    pub clock_carrier_score: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OpticalFrameIdentity {
    code_index: u64,
    clock_carrier_center: Option<[f64; 2]>,
    clock_carrier_extent: Option<[f64; 2]>,
    clock_carrier_score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpticalStimulusTrack {
    protocol: ScreenStimulusProtocol,
    identity_by_sensor_timestamp: HashMap<u64, OpticalFrameIdentity>,
    pub stream: String,
    pub session_id: String,
}

fn finite_number(value: &Value, field: &str) -> Result<f64, String> {
    let number = value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing numeric field {field}"))?;
    number
        .is_finite()
        .then_some(number)
        .ok_or_else(|| format!("non-finite numeric field {field}"))
}

fn finite_pair(value: &Value, field: &str) -> Result<[f64; 2], String> {
    let values = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing pair field {field}"))?;
    if values.len() != 2 {
        return Err(format!("field {field} must contain two numbers"));
    }
    let pair = [
        values[0]
            .as_f64()
            .ok_or_else(|| format!("invalid first value in {field}"))?,
        values[1]
            .as_f64()
            .ok_or_else(|| format!("invalid second value in {field}"))?,
    ];
    (pair[0].is_finite() && pair[1].is_finite())
        .then_some(pair)
        .ok_or_else(|| format!("non-finite value in {field}"))
}

fn parse_protocol(session: &Value) -> Result<ScreenStimulusProtocol, String> {
    if session.get("record_type").and_then(Value::as_str) != Some("session") {
        return Err("screen manifest does not begin with a session record".to_string());
    }
    let motion = session
        .get("motion")
        .ok_or_else(|| "screen session is missing motion protocol".to_string())?;
    let protocol = ScreenStimulusProtocol {
        code_hz: finite_number(session, "code_hz")?,
        warmup_seconds: finite_number(motion, "warmup_seconds")?,
        horizontal_period_seconds: finite_number(motion, "horizontal_period_seconds")?,
        vertical_period_seconds: finite_number(motion, "vertical_period_seconds")?,
        x_range_normalized: finite_pair(motion, "x_range_normalized")?,
        y_range_normalized: finite_pair(motion, "y_range_normalized")?,
        screen_field_degrees: finite_pair(session, "screen_field_degrees")?,
    };
    if protocol.code_hz <= 0.0
        || protocol.horizontal_period_seconds <= 0.0
        || protocol.vertical_period_seconds <= 0.0
        || protocol.warmup_seconds < 0.0
        || protocol.x_range_normalized[0] >= protocol.x_range_normalized[1]
        || protocol.y_range_normalized[0] >= protocol.y_range_normalized[1]
    {
        return Err("invalid screen stimulus protocol geometry".to_string());
    }
    Ok(protocol)
}

impl ScreenStimulusProtocol {
    pub fn pose_for_code(self, code_index: u64) -> StimulusPose {
        self.pose_for_identity(OpticalFrameIdentity {
            code_index,
            clock_carrier_center: None,
            clock_carrier_extent: None,
            clock_carrier_score: None,
        })
    }

    fn pose_for_identity(self, identity: OpticalFrameIdentity) -> StimulusPose {
        // The optical identity resolves a complete code interval.  Its
        // midpoint is the minimum-bias exposure time when no sub-tick rolling
        // timing is asserted by the sensor protocol.
        let code_index = identity.code_index;
        let elapsed_seconds = (code_index as f64 + 0.5) / self.code_hz;
        let moving = elapsed_seconds >= self.warmup_seconds;
        let (x, y, vx, vy) = if moving {
            let motion_time = elapsed_seconds - self.warmup_seconds;
            let x_center = 0.5 * (self.x_range_normalized[0] + self.x_range_normalized[1]);
            let y_center = 0.5 * (self.y_range_normalized[0] + self.y_range_normalized[1]);
            let x_amplitude = 0.5 * (self.x_range_normalized[1] - self.x_range_normalized[0]);
            let y_amplitude = 0.5 * (self.y_range_normalized[1] - self.y_range_normalized[0]);
            let x_omega = std::f64::consts::TAU / self.horizontal_period_seconds;
            let y_omega = std::f64::consts::TAU / self.vertical_period_seconds;
            (
                x_center + x_amplitude * (x_omega * motion_time).sin(),
                y_center + y_amplitude * (y_omega * motion_time).sin(),
                x_amplitude * x_omega * (x_omega * motion_time).cos(),
                y_amplitude * y_omega * (y_omega * motion_time).cos(),
            )
        } else {
            (0.5, 0.5, 0.0, 0.0)
        };
        let half_tangent = [
            (0.5 * self.screen_field_degrees[0].to_radians()).tan(),
            (0.5 * self.screen_field_degrees[1].to_radians()).tan(),
        ];
        StimulusPose {
            code_index,
            elapsed_seconds,
            center_normalized: [x, y],
            velocity_normalized_per_second: [vx, vy],
            gaze_tangent: [
                (2.0 * x - 1.0) * half_tangent[0],
                (2.0 * y - 1.0) * half_tangent[1],
            ],
            gaze_tangent_velocity_per_second: [
                2.0 * vx * half_tangent[0],
                2.0 * vy * half_tangent[1],
            ],
            moving,
            clock_carrier_center: identity.clock_carrier_center,
            clock_carrier_extent: identity.clock_carrier_extent,
            clock_carrier_score: identity.clock_carrier_score,
        }
    }
}

fn clock_carrier(record: &Value) -> Option<([f64; 2], [f64; 2], f64)> {
    let witness = record.get("direct_witness")?;
    if witness.get("valid").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    let corners = witness.get("quad_roi_corners")?.as_array()?;
    if corners.len() != 4 {
        return None;
    }
    let corner = |index: usize| -> Option<[f64; 2]> {
        let values = corners.get(index)?.as_array()?;
        Some([values.first()?.as_f64()?, values.get(1)?.as_f64()?])
    };
    let first = corner(0)?;
    let third = corner(2)?;
    let center = [0.5 * (first[0] + third[0]), 0.5 * (first[1] + third[1])];
    let extent = [(third[0] - first[0]).abs(), (third[1] - first[1]).abs()];
    let score = witness.get("proposal_score")?.as_f64()?;
    (center[0].is_finite()
        && center[1].is_finite()
        && extent[0].is_finite()
        && extent[1].is_finite()
        && extent[0] > 0.0
        && extent[1] > 0.0
        && score.is_finite())
    .then_some((center, extent, score))
}

impl OpticalStimulusTrack {
    pub fn from_jsonl(
        manifest_jsonl: &str,
        clock_jsonl: &str,
        stream: &str,
    ) -> Result<Self, String> {
        let session = manifest_jsonl
            .lines()
            .find(|line| !line.trim().is_empty())
            .ok_or_else(|| "empty screen presentation manifest".to_string())
            .and_then(|line| {
                serde_json::from_str::<Value>(line).map_err(|error| error.to_string())
            })?;
        let protocol = parse_protocol(&session)?;
        let session_id = session
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "screen session is missing session_id".to_string())?
            .to_string();
        let mut identity_by_sensor_timestamp = HashMap::new();
        for line in clock_jsonl.lines().filter(|line| !line.trim().is_empty()) {
            let record = serde_json::from_str::<Value>(line).map_err(|error| error.to_string())?;
            if record.get("record_type").and_then(Value::as_str) != Some("whole-roi-clock-frame")
                || record.get("stream").and_then(Value::as_str) != Some(stream)
                || record.get("status").and_then(Value::as_str) != Some("recovered")
            {
                continue;
            }
            let timestamp = record
                .get("sensor_timestamp_ns")
                .and_then(Value::as_u64)
                .ok_or_else(|| "recovered clock frame is missing sensor timestamp".to_string())?;
            let code_index = record
                .get("code_index")
                .and_then(Value::as_u64)
                .ok_or_else(|| "recovered clock frame is missing code index".to_string())?;
            let carrier = clock_carrier(&record);
            identity_by_sensor_timestamp.insert(
                timestamp,
                OpticalFrameIdentity {
                    code_index,
                    clock_carrier_center: carrier.map(|value| value.0),
                    clock_carrier_extent: carrier.map(|value| value.1),
                    clock_carrier_score: carrier.map(|value| value.2),
                },
            );
        }
        if identity_by_sensor_timestamp.is_empty() {
            return Err(format!("no recovered optical clock frames for {stream}"));
        }
        Ok(Self {
            protocol,
            identity_by_sensor_timestamp,
            stream: stream.to_string(),
            session_id,
        })
    }

    pub fn from_files(
        manifest_path: impl AsRef<Path>,
        clock_path: impl AsRef<Path>,
        stream: &str,
    ) -> Result<Self, String> {
        let manifest_path = manifest_path.as_ref();
        let clock_path = clock_path.as_ref();
        let manifest = fs::read_to_string(manifest_path)
            .map_err(|error| format!("read {}: {error}", manifest_path.display()))?;
        let clock = fs::read_to_string(clock_path)
            .map_err(|error| format!("read {}: {error}", clock_path.display()))?;
        Self::from_jsonl(&manifest, &clock, stream)
    }

    pub fn pose_at_sensor_timestamp(&self, sensor_timestamp_ns: u64) -> Option<StimulusPose> {
        self.identity_by_sensor_timestamp
            .get(&sensor_timestamp_ns)
            .copied()
            .map(|identity| self.protocol.pose_for_identity(identity))
    }

    pub fn recovered_frames(&self) -> usize {
        self.identity_by_sensor_timestamp.len()
    }

    pub fn protocol(&self) -> ScreenStimulusProtocol {
        self.protocol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"{"record_type":"session","session_id":"test","code_hz":30.0,"screen_field_degrees":[56.0,34.0],"motion":{"horizontal_period_seconds":9.0,"vertical_period_seconds":7.0,"warmup_seconds":2.0,"x_range_normalized":[0.07,0.93],"y_range_normalized":[0.1,0.9]}}
{"record_type":"presentation","present_commit_unix_ns":999999999}"#;
    const CLOCK: &str = r#"{"record_type":"whole-roi-clock-frame","stream":"subject-right","status":"recovered","sensor_timestamp_ns":100,"code_index":0,"direct_witness":{"valid":true,"proposal_score":1.6,"quad_roi_corners":[[10.0,20.0],[30.0,20.0],[30.0,44.0],[10.0,44.0]]}}
{"record_type":"whole-roi-clock-frame","stream":"subject-right","status":"stimulus-inactive","sensor_timestamp_ns":200,"code_index":null}
{"record_type":"whole-roi-clock-frame","stream":"subject-left","status":"recovered","sensor_timestamp_ns":100,"code_index":9}"#;

    #[test]
    fn optical_identity_not_host_time_selects_the_target_pose() {
        let track = OpticalStimulusTrack::from_jsonl(MANIFEST, CLOCK, "subject-right").unwrap();
        let pose = track.pose_at_sensor_timestamp(100).unwrap();
        assert_eq!(pose.code_index, 0);
        assert_eq!(pose.center_normalized, [0.5, 0.5]);
        assert_eq!(pose.velocity_normalized_per_second, [0.0, 0.0]);
        assert!(!pose.moving);
        assert_eq!(pose.clock_carrier_center, Some([20.0, 32.0]));
        assert_eq!(pose.clock_carrier_extent, Some([20.0, 24.0]));
        assert_eq!(pose.clock_carrier_score, Some(1.6));
        assert!(track.pose_at_sensor_timestamp(200).is_none());
    }

    #[test]
    fn code_midpoint_reconstructs_the_same_lissajous_protocol_as_the_renderer() {
        let track = OpticalStimulusTrack::from_jsonl(MANIFEST, CLOCK, "subject-right").unwrap();
        let pose = track.protocol().pose_for_code(60);
        let motion_time = pose.elapsed_seconds - 2.0;
        let expected_x = 0.5 + 0.43 * (std::f64::consts::TAU / 9.0 * motion_time).sin();
        let expected_y = 0.5 + 0.40 * (std::f64::consts::TAU / 7.0 * motion_time).sin();
        assert!((pose.center_normalized[0] - expected_x).abs() < 1.0e-12);
        assert!((pose.center_normalized[1] - expected_y).abs() < 1.0e-12);
        assert!(pose.moving);
    }
}
