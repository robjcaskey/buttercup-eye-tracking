//! Immutable sensor-read / ROI-buffer provenance and bounded source timing.
//!
//! The current wire protocol supplies a source timestamp, not its accuracy or
//! rolling-shutter row offsets. Adapters must leave those bounds unknown unless
//! supplied by an explicit clock/exposure model. Host arrival and inference
//! completion times are never substitutes. Sharing a read reference does not
//! assert that different sensor rows were exposed at the same instant.

use super::{RoiId, SourceClock};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TimestampBandNs {
    minimum: u64,
    maximum: u64,
}

impl TimestampBandNs {
    pub(crate) fn new(minimum: u64, maximum: u64) -> Option<Self> {
        (minimum <= maximum).then_some(Self { minimum, maximum })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RelativeTimeBandNs {
    pub(crate) minimum: i128,
    pub(crate) maximum: i128,
}

impl RelativeTimeBandNs {
    pub(crate) fn seconds(self) -> (f64, f64) {
        (self.minimum as f64 * 1e-9, self.maximum as f64 * 1e-9)
    }
}

/// `read_id` identifies a physical sensor read, not a per-ROI sequence number.
/// The adapter that owns the source clock is responsible for this attestation.
#[derive(Debug)]
pub(crate) struct SensorReadTiming {
    clock: SourceClock,
    read_id: u64,
    timestamp_ns: u64,
    reference_band: Option<TimestampBandNs>,
}

impl SensorReadTiming {
    pub(crate) fn new(
        clock: SourceClock,
        read_id: u64,
        timestamp_ns: u64,
        reference_band: Option<TimestampBandNs>,
    ) -> Option<Arc<Self>> {
        if reference_band
            .is_some_and(|band| timestamp_ns < band.minimum || timestamp_ns > band.maximum)
        {
            return None;
        }
        Some(Arc::new(Self {
            clock,
            read_id,
            timestamp_ns,
            reference_band,
        }))
    }

    pub(crate) fn clock(&self) -> SourceClock {
        self.clock
    }
    pub(crate) fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    pub(crate) fn relative_to(&self, earlier: &Self) -> Option<RelativeTimeBandNs> {
        if self.clock != earlier.clock {
            return None;
        }
        if self.read_id == earlier.read_id {
            // An inconsistent reuse of a read id is not synchronization proof.
            return (self.timestamp_ns == earlier.timestamp_ns).then_some(RelativeTimeBandNs {
                minimum: 0,
                maximum: 0,
            });
        }
        let later = self.reference_band?;
        let earlier = earlier.reference_band?;
        Some(RelativeTimeBandNs {
            minimum: i128::from(later.minimum) - i128::from(earlier.maximum),
            maximum: i128::from(later.maximum) - i128::from(earlier.minimum),
        })
    }
}

/// One immutable RAW ROI buffer's timing; every image derived from that buffer
/// shares this object. Offsets express exposure time relative to the read's
/// clock reference. None means unknown (for example, unmeasured row timing).
#[derive(Debug)]
pub(crate) struct RoiBufferTiming {
    read: Arc<SensorReadTiming>,
    roi: RoiId,
    exposure_offset_ns: Option<(i64, i64)>,
}

impl RoiBufferTiming {
    pub(crate) fn new(
        read: Arc<SensorReadTiming>,
        roi: RoiId,
        exposure_offset_ns: Option<(i64, i64)>,
    ) -> Option<Arc<Self>> {
        if exposure_offset_ns.is_some_and(|(minimum, maximum)| minimum > maximum) {
            return None;
        }
        Some(Arc::new(Self {
            read,
            roi,
            exposure_offset_ns,
        }))
    }

    pub(crate) fn read(&self) -> &Arc<SensorReadTiming> {
        &self.read
    }
    pub(crate) fn roi(&self) -> RoiId {
        self.roi
    }

    pub(crate) fn relative_exposure_to(
        self: &Arc<Self>,
        earlier: &Arc<Self>,
    ) -> Option<RelativeTimeBandNs> {
        if Arc::ptr_eq(self, earlier) {
            // Same pixels: shared clock/row uncertainty cancels, it is not two
            // independent errors to add. Redraws are also not new exposures.
            return Some(RelativeTimeBandNs {
                minimum: 0,
                maximum: 0,
            });
        }
        let reference = self.read.relative_to(&earlier.read)?;
        let (later_minimum, later_maximum) = self.exposure_offset_ns?;
        let (earlier_minimum, earlier_maximum) = earlier.exposure_offset_ns?;
        Some(RelativeTimeBandNs {
            minimum: reference.minimum + i128::from(later_minimum) - i128::from(earlier_maximum),
            maximum: reference.maximum + i128::from(later_maximum) - i128::from(earlier_minimum),
        })
    }
}

/// A small host-side adapter contract, not a new wire format. Transforming
/// pixels cannot edit their timing. No live producer is migrated implicitly.
#[derive(Clone, Debug)]
pub(crate) struct TimedImage<T> {
    pixels: Arc<T>,
    timing: Arc<RoiBufferTiming>,
}

impl<T> TimedImage<T> {
    pub(crate) fn new(pixels: T, timing: Arc<RoiBufferTiming>) -> Self {
        Self {
            pixels: Arc::new(pixels),
            timing,
        }
    }
    pub(crate) fn pixels(&self) -> &T {
        &self.pixels
    }
    pub(crate) fn timing(&self) -> &Arc<RoiBufferTiming> {
        &self.timing
    }
    pub(crate) fn transformed<U>(&self, transform: impl FnOnce(&T) -> U) -> TimedImage<U> {
        TimedImage::new(transform(&self.pixels), Arc::clone(&self.timing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock() -> SourceClock {
        SourceClock {
            domain: 7,
            epoch: 2,
        }
    }

    #[test]
    fn images_from_one_sensor_read_inherit_the_same_clock_reference() {
        let read = SensorReadTiming::new(clock(), 17, 1_000_000, None).unwrap();
        let left = TimedImage::new(
            vec![123u16; 16],
            RoiBufferTiming::new(Arc::clone(&read), RoiId(0), None).unwrap(),
        );
        let right = TimedImage::new(
            vec![456u16; 16],
            RoiBufferTiming::new(Arc::clone(&read), RoiId(1), None).unwrap(),
        );
        assert!(Arc::ptr_eq(left.timing().read(), right.timing().read()));
        assert_eq!(left.timing().read().clock(), right.timing().read().clock());
        assert_eq!(
            left.timing().read().timestamp_ns(),
            right.timing().read().timestamp_ns()
        );
        assert_ne!(left.timing().roi(), right.timing().roi());
        assert_eq!(
            left.timing().read().relative_to(right.timing().read()),
            Some(RelativeTimeBandNs {
                minimum: 0,
                maximum: 0
            })
        );
        assert_eq!(left.timing().relative_exposure_to(right.timing()), None,
            "same read clock must not invent simultaneous exposure of different rolling-shutter rows");
    }

    #[test]
    fn derived_images_from_one_roi_buffer_keep_identical_timing() {
        let read = SensorReadTiming::new(clock(), 17, 1_000_000, None).unwrap();
        let raw = TimedImage::new(
            vec![100u16, 200, 300, 400],
            RoiBufferTiming::new(read, RoiId(0), None).unwrap(),
        );
        let color = raw.transformed(|raw| raw.iter().map(|value| [*value; 3]).collect::<Vec<_>>());
        let contrast =
            color.transformed(|rgb| rgb.iter().map(|pixel| pixel[0] * 2).collect::<Vec<_>>());
        assert_eq!(contrast.pixels(), &[200, 400, 600, 800]);
        assert!(Arc::ptr_eq(raw.timing(), color.timing()));
        assert!(Arc::ptr_eq(raw.timing(), contrast.timing()));
        assert_eq!(
            contrast.timing().relative_exposure_to(raw.timing()),
            Some(RelativeTimeBandNs {
                minimum: 0,
                maximum: 0
            })
        );
    }

    #[test]
    fn different_reads_have_signed_upper_and_lower_relative_time_bounds() {
        let a = SensorReadTiming::new(
            clock(),
            17,
            1_000_000_000,
            TimestampBandNs::new(999_000_000, 1_001_000_000),
        )
        .unwrap();
        let b = SensorReadTiming::new(
            clock(),
            18,
            1_020_000_000,
            TimestampBandNs::new(1_018_000_000, 1_022_000_000),
        )
        .unwrap();
        assert_eq!(
            b.relative_to(&a),
            Some(RelativeTimeBandNs {
                minimum: 17_000_000,
                maximum: 23_000_000
            })
        );
        assert_eq!(
            a.relative_to(&b),
            Some(RelativeTimeBandNs {
                minimum: -23_000_000,
                maximum: -17_000_000
            })
        );
        let left = RoiBufferTiming::new(Arc::clone(&a), RoiId(0), Some((0, 1_000_000))).unwrap();
        let right = RoiBufferTiming::new(b, RoiId(1), Some((2_000_000, 4_000_000))).unwrap();
        assert_eq!(
            right.relative_exposure_to(&left),
            Some(RelativeTimeBandNs {
                minimum: 18_000_000,
                maximum: 27_000_000
            })
        );
        let same_read_later_rows =
            RoiBufferTiming::new(a, RoiId(1), Some((2_000_000, 4_000_000))).unwrap();
        assert_eq!(
            same_read_later_rows.relative_exposure_to(&left),
            Some(RelativeTimeBandNs {
                minimum: 1_000_000,
                maximum: 4_000_000
            })
        );
    }

    #[test]
    fn timing_rejects_unknown_bounds_unrelated_epochs_and_invalid_intervals() {
        assert!(TimestampBandNs::new(5, 4).is_none());
        assert!(SensorReadTiming::new(clock(), 1, 10, TimestampBandNs::new(20, 30)).is_none());
        let a = SensorReadTiming::new(clock(), 1, 10, TimestampBandNs::new(9, 11)).unwrap();
        let unknown = SensorReadTiming::new(clock(), 2, 20, None).unwrap();
        assert_eq!(unknown.relative_to(&a), None);
        let restarted = SensorReadTiming::new(
            SourceClock {
                epoch: 3,
                ..clock()
            },
            1,
            10,
            TimestampBandNs::new(9, 11),
        )
        .unwrap();
        assert_eq!(restarted.relative_to(&a), None);
        let wrong_id = SensorReadTiming::new(clock(), 1, 20, None).unwrap();
        assert_eq!(wrong_id.relative_to(&a), None);
        assert!(RoiBufferTiming::new(a, RoiId(0), Some((2, 1))).is_none());
    }
}
