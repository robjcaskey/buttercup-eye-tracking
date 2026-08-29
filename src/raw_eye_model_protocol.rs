use std::io::{Read, Write};
use std::sync::Arc;

pub const MODEL_STREAM_HEADER_BYTES: usize = 96;
pub const MODEL_STREAM_MAGIC: &[u8; 4] = b"OIR1";
pub const MODEL_STREAM_VERSION: u16 = 1;
pub const MODEL_STREAM_MAX_PAYLOAD_BYTES: usize = 64 * 1024 * 1024;

pub const FLAG_VERIFIED_RAW10_1X1: u32 = 1 << 0;
pub const FLAG_ANATOMY_VALID: u32 = 1 << 1;
pub const FLAG_EYE_BASIN_VALID: u32 = 1 << 2;
pub const FLAG_FOCUS_SETTLED: u32 = 1 << 3;
pub const FLAG_IDENTITY_PRESENT: u32 = 1 << 4;
pub const FLAG_MOTION_COMPARABLE: u32 = 1 << 5;

#[derive(Clone, Debug)]
pub struct RawModelFrame {
    pub eye_id: u32,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub flags: u32,
    pub focus_target: u16,
    pub focus_position: u16,
    pub focus_generation: u32,
    pub focus_score: f32,
    pub motion_score: f32,
    pub center_x: f32,
    pub center_y: f32,
    pub iris_radius: f32,
    pub axis_ratio: f32,
    pub axis_angle: f32,
    pub point_count: u32,
    pub payload: Arc<Vec<u8>>,
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("u16 field"))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64 field"))
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("f32 field"))
}

impl RawModelFrame {
    pub fn has_eye_geometry(&self) -> bool {
        self.center_x >= 0.0
            && self.center_y >= 0.0
            && self.center_x < self.width as f32
            && self.center_y < self.height as f32
            && self.iris_radius >= 4.0
            && (0.25..=4.0).contains(&self.axis_ratio)
    }

    pub fn anatomy_observable(&self) -> bool {
        self.flags & FLAG_ANATOMY_VALID != 0 && self.has_eye_geometry()
    }

    pub fn header(&self) -> Result<[u8; MODEL_STREAM_HEADER_BYTES], String> {
        self.validate()?;
        let mut bytes = [0u8; MODEL_STREAM_HEADER_BYTES];
        bytes[0..4].copy_from_slice(MODEL_STREAM_MAGIC);
        bytes[4..6].copy_from_slice(&MODEL_STREAM_VERSION.to_le_bytes());
        bytes[6..8].copy_from_slice(&(MODEL_STREAM_HEADER_BYTES as u16).to_le_bytes());
        bytes[8..12].copy_from_slice(&(self.payload.len() as u32).to_le_bytes());
        bytes[12..16].copy_from_slice(&self.eye_id.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.sequence.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.sensor_x.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.sensor_y.to_le_bytes());
        bytes[40..44].copy_from_slice(&self.width.to_le_bytes());
        bytes[44..48].copy_from_slice(&self.height.to_le_bytes());
        bytes[48..52].copy_from_slice(&self.stride.to_le_bytes());
        bytes[52..56].copy_from_slice(&self.flags.to_le_bytes());
        bytes[56..58].copy_from_slice(&self.focus_target.to_le_bytes());
        bytes[58..60].copy_from_slice(&self.focus_position.to_le_bytes());
        bytes[60..64].copy_from_slice(&self.focus_generation.to_le_bytes());
        bytes[64..68].copy_from_slice(&self.focus_score.to_le_bytes());
        bytes[68..72].copy_from_slice(&self.motion_score.to_le_bytes());
        bytes[72..76].copy_from_slice(&self.center_x.to_le_bytes());
        bytes[76..80].copy_from_slice(&self.center_y.to_le_bytes());
        bytes[80..84].copy_from_slice(&self.iris_radius.to_le_bytes());
        bytes[84..88].copy_from_slice(&self.axis_ratio.to_le_bytes());
        bytes[88..92].copy_from_slice(&self.axis_angle.to_le_bytes());
        bytes[92..96].copy_from_slice(&self.point_count.to_le_bytes());
        Ok(bytes)
    }

    pub fn write_to<W: Write>(&self, output: &mut W) -> Result<(), String> {
        let header = self.header()?;
        output
            .write_all(&header)
            .and_then(|()| output.write_all(self.payload.as_slice()))
            .map_err(|error| format!("write RAW model frame: {error}"))
    }

    pub fn read_from<R: Read>(input: &mut R) -> Result<Self, String> {
        let mut header = [0u8; MODEL_STREAM_HEADER_BYTES];
        input
            .read_exact(&mut header)
            .map_err(|error| format!("read RAW model header: {error}"))?;
        if &header[0..4] != MODEL_STREAM_MAGIC
            || read_u16(&header, 4) != MODEL_STREAM_VERSION
            || read_u16(&header, 6) as usize != MODEL_STREAM_HEADER_BYTES
        {
            return Err("unsupported RAW model stream header".to_string());
        }
        let payload_len = read_u32(&header, 8) as usize;
        if payload_len == 0 || payload_len > MODEL_STREAM_MAX_PAYLOAD_BYTES {
            return Err(format!("invalid RAW model payload length {payload_len}"));
        }
        let mut payload = vec![0u8; payload_len];
        input
            .read_exact(&mut payload)
            .map_err(|error| format!("read RAW model payload: {error}"))?;
        let frame = Self {
            eye_id: read_u32(&header, 12),
            sequence: read_u64(&header, 16),
            timestamp_ns: read_u64(&header, 24),
            sensor_x: read_u32(&header, 32),
            sensor_y: read_u32(&header, 36),
            width: read_u32(&header, 40),
            height: read_u32(&header, 44),
            stride: read_u32(&header, 48),
            flags: read_u32(&header, 52),
            focus_target: read_u16(&header, 56),
            focus_position: read_u16(&header, 58),
            focus_generation: read_u32(&header, 60),
            focus_score: read_f32(&header, 64),
            motion_score: read_f32(&header, 68),
            center_x: read_f32(&header, 72),
            center_y: read_f32(&header, 76),
            iris_radius: read_f32(&header, 80),
            axis_ratio: read_f32(&header, 84),
            axis_angle: read_f32(&header, 88),
            point_count: read_u32(&header, 92),
            payload: Arc::new(payload),
        };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(1..=2).contains(&self.eye_id) {
            return Err(format!("invalid semantic eye id {}", self.eye_id));
        }
        if self.width == 0 || self.height == 0 || self.width % 4 != 0 {
            return Err("invalid RAW model frame dimensions".to_string());
        }
        let expected_stride = self.width / 4 * 5;
        let expected_payload = expected_stride as usize * self.height as usize;
        if self.stride != expected_stride || self.payload.len() != expected_payload {
            return Err(format!(
                "RAW model geometry mismatch {}x{} stride={} payload={}",
                self.width,
                self.height,
                self.stride,
                self.payload.len(),
            ));
        }
        if self.flags & FLAG_VERIFIED_RAW10_1X1 == 0 {
            return Err("RAW model frame is not attested 1x1 RAW10".to_string());
        }
        let finite = [
            self.focus_score,
            self.motion_score,
            self.center_x,
            self.center_y,
            self.iris_radius,
            self.axis_ratio,
            self.axis_angle,
        ];
        if finite.iter().any(|value| !value.is_finite())
            || self.iris_radius < 0.0
            || self.axis_ratio < 0.0
        {
            return Err("RAW model frame contains invalid fit metadata".to_string());
        }
        if self.flags & FLAG_ANATOMY_VALID != 0 && !self.has_eye_geometry() {
            return Err(
                "RAW model frame claims observable anatomy without a usable eye fit".to_string(),
            );
        }
        if self.payload.len() > MODEL_STREAM_MAX_PAYLOAD_BYTES {
            return Err("RAW model frame exceeds the protocol limit".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample() -> RawModelFrame {
        RawModelFrame {
            eye_id: 1,
            sequence: 42,
            timestamp_ns: 77,
            sensor_x: 1000,
            sensor_y: 2000,
            width: 4,
            height: 1,
            stride: 5,
            flags: FLAG_VERIFIED_RAW10_1X1,
            focus_target: 520,
            focus_position: 520,
            focus_generation: 9,
            focus_score: 81.0,
            motion_score: 2.0,
            center_x: 2.0,
            center_y: 0.5,
            iris_radius: 1.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            point_count: 12,
            payload: Arc::new(vec![1, 2, 3, 4, 5]),
        }
    }

    #[test]
    fn frame_round_trips_through_fragment_agnostic_stream_io() {
        let frame = sample();
        let mut bytes = Vec::new();
        frame.write_to(&mut bytes).unwrap();
        let decoded = RawModelFrame::read_from(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(decoded.sequence, 42);
        assert_eq!(decoded.sensor_x, 1000);
        assert_eq!(decoded.payload.as_slice(), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn truncated_payload_is_rejected() {
        let frame = sample();
        let mut bytes = Vec::new();
        frame.write_to(&mut bytes).unwrap();
        bytes.pop();
        assert!(RawModelFrame::read_from(&mut Cursor::new(bytes)).is_err());
    }

    #[test]
    fn raw_roi_without_anatomical_fit_still_round_trips() {
        let mut frame = sample();
        frame.flags = FLAG_VERIFIED_RAW10_1X1;
        frame.center_x = 0.0;
        frame.center_y = 0.0;
        frame.iris_radius = 0.0;
        frame.axis_ratio = 0.0;
        frame.point_count = 0;
        let mut bytes = Vec::new();
        frame.write_to(&mut bytes).unwrap();
        let decoded = RawModelFrame::read_from(&mut Cursor::new(bytes)).unwrap();
        assert!(!decoded.has_eye_geometry());
        assert!(!decoded.anatomy_observable());
    }

    #[test]
    fn anatomy_flag_requires_usable_eye_geometry() {
        let mut frame = sample();
        frame.flags |= FLAG_ANATOMY_VALID;
        frame.iris_radius = 0.0;
        assert!(frame.write_to(&mut Vec::new()).is_err());
    }
}
