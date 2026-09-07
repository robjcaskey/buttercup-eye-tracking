//! Native camera thumbnail transport, not a display screenshot or re-encode.
use super::{MODEL_STREAM_MAX_PAYLOAD_BYTES, read_u16, read_u32, read_u64};
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::sync::Arc;

const ENVELOPE_BYTES: usize = 24;
const MAX_METADATA_BYTES: usize = 16 * 1024;
pub const CAMERA_HEADER_BYTES: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThumbnailKind {
    GlobalSensor,
    SensorBand,
}

impl ThumbnailKind {
    fn label(self) -> &'static str {
        match self {
            Self::GlobalSensor => "global_sensor",
            Self::SensorBand => "sensor_band",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CameraThumbnailFrame {
    pub kind: ThumbnailKind,
    /// Full physical sensor size and covered rectangle, NOT output dimensions.
    pub sensor_size_px: [u32; 2],
    pub sensor_rect_px: [u32; 4],
    /// Host completion time of the socket read, never substituted for exposure.
    pub host_received_unix_ns: u64,
    pub region_session: Option<u64>,
    pub region_generation: Option<u64>,
    pub camera_header: [u8; CAMERA_HEADER_BYTES],
    pub payload: Arc<Vec<u8>>,
}

impl CameraThumbnailFrame {
    pub fn validate(&self) -> Result<(), String> {
        let h = &self.camera_header;
        let width = read_u32(h, 40) as usize;
        let height = read_u32(h, 44) as usize;
        let stride = read_u32(h, 48) as usize;
        let expected_stride = match &h[..4] {
            b"ORT1"
                if self.kind == ThumbnailKind::GlobalSensor
                    && read_u32(h, 12) == 1
                    && read_u32(h, 56) == 32
                    && width % 4 == 0 =>
            {
                width.checked_div(4).and_then(|w| w.checked_mul(5))
            }
            b"OTH1" if read_u32(h, 12) == 3 && read_u32(h, 56) == 16 => width.checked_mul(2),
            _ => return Err("unsupported native thumbnail format".into()),
        };
        if read_u16(h, 4) != 1
            || read_u16(h, 6) as usize != CAMERA_HEADER_BYTES
            || read_u32(h, 8) != 0
            || width == 0
            || height == 0
            || expected_stride != Some(stride)
            || stride.checked_mul(height) != Some(self.payload.len())
            || read_u32(h, 52) as usize != self.payload.len()
            || self.payload.len() > MODEL_STREAM_MAX_PAYLOAD_BYTES
        {
            return Err("invalid native thumbnail geometry or payload length".into());
        }
        let [x, y, w, h] = self.sensor_rect_px;
        if w == 0
            || h == 0
            || width > w as usize
            || height > h as usize
            || x.checked_add(w).is_none_or(|v| v > self.sensor_size_px[0])
            || y.checked_add(h).is_none_or(|v| v > self.sensor_size_px[1])
            || x != read_u32(&self.camera_header, 32)
            || y != read_u32(&self.camera_header, 36)
            || (self.kind == ThumbnailKind::GlobalSensor
                && [x, y, w, h] != [0, 0, self.sensor_size_px[0], self.sensor_size_px[1]])
        {
            return Err("invalid thumbnail sensor coverage".into());
        }
        Ok(())
    }

    pub fn metadata(&self) -> Value {
        let h = &self.camera_header;
        let raw10 = &h[..4] == b"ORT1";
        json!({
            "schema": "buttercup-native-thumbnail-v1",
            "frame_kind": self.kind.label(),
            "encoding": if raw10 { "RAW10_LE40" } else { "GRAY16LE" },
            "sample_layout": if raw10 { "binned-global-rggb" } else { "camera-linear-raw-average" },
            "width_px": read_u32(h, 40), "height_px": read_u32(h, 44), "stride_bytes": read_u32(h, 48),
            "sensor_size_px": self.sensor_size_px, "sensor_rect_px": self.sensor_rect_px,
            "camera_magic": String::from_utf8_lossy(&h[..4]),
            "camera_format_id": read_u32(h, 12), "camera_pixel_format": read_u32(h, 56),
            "camera_flags": read_u32(h, 60),
            "sequence": read_u64(h, 16).to_string(),
            "sequence_semantics": if self.kind == ThumbnailKind::GlobalSensor { "capture-request-id" } else { "fine-stream-acquisition-sequence" },
            "source_timestamp_ns": read_u64(h, 24).to_string(),
            "source_clock": "camera-source-timestamp; hardware uncertainty and cross-connection epoch not supplied",
            "host_received_unix_ns": self.host_received_unix_ns.to_string(),
            "region_session": self.region_session.map(|v| v.to_string()),
            "region_generation": self.region_generation.map(|v| v.to_string()),
            "same_exposure_as_roi": Value::Null,
            "camera_header_bytes": CAMERA_HEADER_BYTES, "payload_bytes": self.payload.len(),
            "native_bytes_preserved": true
        })
    }

    pub fn write_to<W: Write>(&self, output: &mut W) -> Result<(), String> {
        self.validate()?;
        let metadata = serde_json::to_vec(&self.metadata()).map_err(|e| e.to_string())?;
        if metadata.len() > MAX_METADATA_BYTES {
            return Err("thumbnail metadata too large".into());
        }
        let mut header = [0u8; ENVELOPE_BYTES];
        header[..4].copy_from_slice(b"OIC1");
        header[4..6].copy_from_slice(&1u16.to_le_bytes());
        header[6..8].copy_from_slice(&(ENVELOPE_BYTES as u16).to_le_bytes());
        header[8..12].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
        header[12..16].copy_from_slice(&(CAMERA_HEADER_BYTES as u32).to_le_bytes());
        header[16..20].copy_from_slice(&(self.payload.len() as u32).to_le_bytes());
        output
            .write_all(&header)
            .and_then(|()| output.write_all(&metadata))
            .and_then(|()| output.write_all(&self.camera_header))
            .and_then(|()| output.write_all(self.payload.as_slice()))
            .map_err(|e| format!("write native thumbnail: {e}"))
    }

    pub(super) fn read_after_magic<R: Read>(input: &mut R) -> Result<Self, String> {
        let mut header = [0u8; ENVELOPE_BYTES];
        header[..4].copy_from_slice(b"OIC1");
        input
            .read_exact(&mut header[4..])
            .map_err(|e| e.to_string())?;
        let metadata_len = read_u32(&header, 8) as usize;
        let payload_len = read_u32(&header, 16) as usize;
        if read_u16(&header, 4) != 1
            || read_u16(&header, 6) as usize != ENVELOPE_BYTES
            || metadata_len == 0
            || metadata_len > MAX_METADATA_BYTES
            || read_u32(&header, 12) as usize != CAMERA_HEADER_BYTES
            || read_u32(&header, 20) != 0
            || payload_len == 0
            || payload_len > MODEL_STREAM_MAX_PAYLOAD_BYTES
        {
            return Err("invalid native thumbnail envelope".into());
        }
        let mut metadata = vec![0; metadata_len];
        input.read_exact(&mut metadata).map_err(|e| e.to_string())?;
        let metadata: Value = serde_json::from_slice(&metadata).map_err(|e| e.to_string())?;
        let mut camera_header = [0; CAMERA_HEADER_BYTES];
        input
            .read_exact(&mut camera_header)
            .map_err(|e| e.to_string())?;
        // Check the native length before allocating or reading its payload.
        if read_u32(&camera_header, 52) as usize != payload_len {
            return Err("thumbnail native length mismatch".into());
        }
        let mut payload = vec![0; payload_len];
        input.read_exact(&mut payload).map_err(|e| e.to_string())?;
        let array = |key: &str| -> Result<Vec<u32>, String> {
            metadata[key]
                .as_array()
                .ok_or_else(|| format!("missing {key}"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| format!("invalid {key}"))
                })
                .collect()
        };
        let optional_u64 = |key: &str| -> Result<Option<u64>, String> {
            if metadata[key].is_null() {
                return Ok(None);
            }
            metadata[key]
                .as_str()
                .and_then(|v| v.parse().ok())
                .map(Some)
                .ok_or_else(|| format!("invalid {key}"))
        };
        let frame = Self {
            kind: match metadata["frame_kind"].as_str() {
                Some("global_sensor") => ThumbnailKind::GlobalSensor,
                Some("sensor_band") => ThumbnailKind::SensorBand,
                _ => return Err("invalid thumbnail kind".into()),
            },
            sensor_size_px: array("sensor_size_px")?
                .try_into()
                .map_err(|_| "invalid sensor size")?,
            sensor_rect_px: array("sensor_rect_px")?
                .try_into()
                .map_err(|_| "invalid sensor rectangle")?,
            host_received_unix_ns: optional_u64("host_received_unix_ns")?
                .ok_or("missing receive time")?,
            region_session: optional_u64("region_session")?,
            region_generation: optional_u64("region_generation")?,
            camera_header,
            payload: Arc::new(payload),
        };
        frame.validate()?;
        if frame.metadata() != metadata {
            return Err("thumbnail metadata contradicts native header".into());
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw_eye_model_protocol::{FLAG_VERIFIED_RAW10_1X1, ModelStreamFrame, RawModelFrame};
    use std::io::Cursor;

    fn sample(kind: ThumbnailKind, raw10: bool) -> CameraThumbnailFrame {
        let mut h = [0u8; CAMERA_HEADER_BYTES];
        h[..4].copy_from_slice(if raw10 { b"ORT1" } else { b"OTH1" });
        h[4..6].copy_from_slice(&1u16.to_le_bytes());
        h[6..8].copy_from_slice(&64u16.to_le_bytes());
        for (offset, value) in [
            (12, if raw10 { 1u32 } else { 3 }),
            (40, 4),
            (44, 2),
            (48, if raw10 { 5 } else { 8 }),
            (52, if raw10 { 10 } else { 16 }),
            (56, if raw10 { 32 } else { 16 }),
            (60, 0x1234),
        ] {
            h[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        h[16..24].copy_from_slice(&(u64::MAX - 2).to_le_bytes());
        h[24..32].copy_from_slice(&(u64::MAX - 3).to_le_bytes());
        let y = if kind == ThumbnailKind::SensorBand {
            2400u32
        } else {
            0
        };
        h[36..40].copy_from_slice(&y.to_le_bytes());
        CameraThumbnailFrame {
            kind,
            sensor_size_px: [8000, 6000],
            sensor_rect_px: [0, y, 8000, if y == 0 { 6000 } else { 576 }],
            host_received_unix_ns: u64::MAX - 1,
            region_session: (y != 0).then_some(123),
            region_generation: (y != 0).then_some(7),
            camera_header: h,
            payload: Arc::new((0..if raw10 { 10 } else { 16 }).collect()),
        }
    }

    struct Fragmented(Cursor<Vec<u8>>);
    impl Read for Fragmented {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let count = out.len().min(3);
            self.0.read(&mut out[..count])
        }
    }

    #[test]
    fn native_global_and_band_payloads_and_metadata_round_trip_without_reencoding() {
        for (kind, raw10) in [
            (ThumbnailKind::GlobalSensor, true),
            (ThumbnailKind::GlobalSensor, false),
            (ThumbnailKind::SensorBand, false),
        ] {
            let frame = sample(kind, raw10);
            let mut bytes = Vec::new();
            frame.write_to(&mut bytes).unwrap();
            let header_start = ENVELOPE_BYTES + read_u32(&bytes, 8) as usize;
            assert_eq!(
                &bytes[header_start..header_start + CAMERA_HEADER_BYTES],
                frame.camera_header
            );
            assert_eq!(
                &bytes[header_start + CAMERA_HEADER_BYTES..],
                frame.payload.as_slice()
            );
            let ModelStreamFrame::Thumbnail(decoded) =
                ModelStreamFrame::read_from(&mut Fragmented(Cursor::new(bytes))).unwrap()
            else {
                panic!("wrong type")
            };
            assert_eq!(decoded.metadata(), frame.metadata());
            assert_eq!(decoded.payload, frame.payload);
            assert_eq!(
                decoded.metadata()["source_timestamp_ns"],
                (u64::MAX - 3).to_string()
            );
            assert!(decoded.metadata()["same_exposure_as_roi"].is_null());
        }
    }

    fn eye() -> RawModelFrame {
        RawModelFrame {
            eye_id: 1,
            sequence: 22,
            timestamp_ns: 23,
            sensor_x: 0,
            sensor_y: 2400,
            width: 4,
            height: 1,
            stride: 5,
            flags: FLAG_VERIFIED_RAW10_1X1,
            focus_target: 500,
            focus_position: 500,
            focus_generation: 2,
            focus_score: 0.0,
            motion_score: 0.0,
            center_x: 0.0,
            center_y: 0.0,
            iris_radius: 0.0,
            axis_ratio: 1.0,
            axis_angle: 0.0,
            point_count: 0,
            payload: Arc::new(vec![3; 5]),
        }
    }

    #[test]
    fn mixed_stream_preserves_roi_layout_and_eye_only_reader_skips_thumbnails_and_metadata() {
        let mut bytes = Vec::new();
        sample(ThumbnailKind::GlobalSensor, true)
            .write_to(&mut bytes)
            .unwrap();
        sample(ThumbnailKind::SensorBand, false)
            .write_to(&mut bytes)
            .unwrap();
        ModelStreamFrame::Metadata(Arc::new(json!({"schema":"test","event":"presentation"})))
            .write_to(&mut bytes)
            .unwrap();
        let offset = bytes.len();
        eye().write_to(&mut bytes).unwrap();
        assert_eq!(&bytes[offset..offset + 96], eye().header().unwrap());
        assert_eq!(
            RawModelFrame::read_from(&mut Cursor::new(bytes.clone()))
                .unwrap()
                .payload,
            eye().payload
        );
        let mut input = Cursor::new(bytes);
        assert!(matches!(
            ModelStreamFrame::read_from(&mut input).unwrap(),
            ModelStreamFrame::Thumbnail(_)
        ));
        assert!(matches!(
            ModelStreamFrame::read_from(&mut input).unwrap(),
            ModelStreamFrame::Thumbnail(_)
        ));
        assert!(matches!(
            ModelStreamFrame::read_from(&mut input).unwrap(),
            ModelStreamFrame::Metadata(_)
        ));
        assert!(matches!(
            ModelStreamFrame::read_from(&mut input).unwrap(),
            ModelStreamFrame::Eye(_)
        ));
    }

    #[test]
    fn rejects_truncation_oversized_lengths_and_native_metadata_contradictions() {
        let mut bytes = Vec::new();
        sample(ThumbnailKind::GlobalSensor, true)
            .write_to(&mut bytes)
            .unwrap();
        for length in [0, 4, 23, 40, bytes.len() - 1] {
            assert!(
                ModelStreamFrame::read_from(&mut Cursor::new(bytes[..length].to_vec())).is_err()
            );
        }
        let mut bad = bytes.clone();
        bad[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(ModelStreamFrame::read_from(&mut Cursor::new(bad)).is_err());
        let mut bad = bytes.clone();
        bad[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(ModelStreamFrame::read_from(&mut Cursor::new(bad)).is_err());
        // Metadata must not silently override source time from the camera header.
        let header_start = ENVELOPE_BYTES + read_u32(&bytes, 8) as usize;
        bytes[header_start + 24] ^= 1;
        assert!(ModelStreamFrame::read_from(&mut Cursor::new(bytes)).is_err());
        let mut bad = sample(ThumbnailKind::SensorBand, false);
        bad.sensor_rect_px[1] = 5900;
        assert!(bad.validate().is_err());
    }
}
