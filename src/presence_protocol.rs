//! Protocol-independent, read-only telemetry values. This module does not
//! acquire images, choose tracking modes, resolve gaze sign, or infer absence.
//! Callers must reuse the desktop gaze policy before supplying signed gaze.
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Source {
    /// Exact source-clock epoch, not the viewer process or publication epoch.
    pub stream_epoch: String,
    pub eye: u32,
    pub timestamp_ns: u64,
    pub authority: String,
    pub authority_generation: u64,
    pub sign_epoch: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Observation<T> {
    pub value: T,
    pub source: Source,
    /// Age of this exact source's host arrival when the snapshot was built.
    /// None means unavailable/ambiguous clock, never a zero age.
    pub source_arrival_age: Option<Duration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Freshness {
    Fresh,
    Stale,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Presence {
    Present,
    Unknown,
}

/// Direct eye evidence is kept separate from identity retention. A missing
/// eye in a selected ROI cannot establish that a person is absent.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum EyeEvidence {
    Direct(Observation<()>),
    Retained(Observation<()>),
    Unavailable(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SignedDirection {
    /// Camera-relative unit direction: right, down, toward camera.
    /// This is not a screen location or an assertion of calibration.
    right_down_toward_camera: [f64; 3],
}

impl SignedDirection {
    /// Validates representation only; the caller must separately establish
    /// signed authority through the existing desktop gaze policy.
    pub fn new(direction: [f64; 3]) -> Result<Self, &'static str> {
        let squared_norm: f64 = direction.iter().map(|value| value * value).sum();
        if !direction.iter().all(|value| value.is_finite())
            || direction[2] <= 0.0
            || (squared_norm - 1.0).abs() > 1e-6
        {
            return Err("invalid camera-facing unit gaze direction");
        }
        Ok(Self {
            right_down_toward_camera: direction,
        })
    }

    pub fn right_down_toward_camera(&self) -> [f64; 3] {
        self.right_down_toward_camera
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Snapshot {
    pub captured_at: Instant,
    pub eye: EyeEvidence,
    /// Err preserves the existing gaze policy's rejection reason, including
    /// unresolved sign. Eye presence remains independently reportable.
    pub gaze: Result<Observation<SignedDirection>, &'static str>,
    /// Process/protocol availability is not eye or person presence.
    pub availability: Result<(), &'static str>,
}

impl Snapshot {
    pub fn gaze_status(
        &self,
        now: Instant,
        maximum_source_age: Duration,
    ) -> Result<&Observation<SignedDirection>, &'static str> {
        self.availability?;
        let observation = self.gaze.as_ref().map_err(|reason| *reason)?;
        match self.freshness(observation, now, maximum_source_age) {
            Freshness::Fresh => Ok(observation),
            Freshness::Stale => Err("stale gaze source"),
            Freshness::Unknown => Err("unknown or ambiguous gaze source clock"),
        }
    }

    pub fn freshness<T>(
        &self,
        observation: &Observation<T>,
        now: Instant,
        maximum_source_age: Duration,
    ) -> Freshness {
        let Some(elapsed) = now.checked_duration_since(self.captured_at) else {
            return Freshness::Unknown;
        };
        match observation
            .source_arrival_age
            .and_then(|age| age.checked_add(elapsed))
        {
            Some(age) if age <= maximum_source_age => Freshness::Fresh,
            Some(_) => Freshness::Stale,
            None => Freshness::Unknown,
        }
    }

    /// Pass mouse_output::MAX_SOURCE_AGE at the integration boundary. There
    /// is deliberately no second freshness threshold defined here.
    pub fn presence(&self, now: Instant, maximum_source_age: Duration) -> (Presence, &'static str) {
        match &self.eye {
            EyeEvidence::Direct(observation) => {
                match self.freshness(observation, now, maximum_source_age) {
                    Freshness::Fresh => {
                        (Presence::Present, "direct eye observation in selected ROI")
                    }
                    Freshness::Stale => (Presence::Unknown, "stale eye source"),
                    Freshness::Unknown => {
                        (Presence::Unknown, "unknown or ambiguous eye source clock")
                    }
                }
            }
            EyeEvidence::Retained(_) => (
                Presence::Unknown,
                "retained identity is not a fresh eye observation",
            ),
            EyeEvidence::Unavailable(reason) => (Presence::Unknown, reason),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(age: Option<Duration>) -> Snapshot {
        Snapshot {
            captured_at: Instant::now(),
            eye: EyeEvidence::Direct(Observation {
                value: (),
                source: Source {
                    stream_epoch: "source-session-1".into(),
                    eye: 1,
                    timestamp_ns: 42,
                    authority: "native anatomy".into(),
                    authority_generation: 3,
                    sign_epoch: None,
                },
                source_arrival_age: age,
            }),
            gaze: Err("paused: unresolved gaze sign"),
            availability: Ok(()),
        }
    }

    #[test]
    fn unsigned_gaze_does_not_hide_direct_eye_evidence() {
        let s = snapshot(Some(Duration::from_millis(10)));
        assert_eq!(
            s.presence(s.captured_at, Duration::from_millis(500)).0,
            Presence::Present
        );
        assert_eq!(s.gaze, Err("paused: unresolved gaze sign"));
        assert_eq!(s.availability, Ok(()));
    }

    #[test]
    fn snapshot_rereads_do_not_renew_source_age() {
        let s = snapshot(Some(Duration::from_millis(490)));
        let limit = Duration::from_millis(500);
        assert_eq!(
            s.presence(s.captured_at + Duration::from_millis(10), limit)
                .0,
            Presence::Present
        );
        assert_eq!(
            s.presence(s.captured_at + Duration::from_millis(11), limit)
                .0,
            Presence::Unknown
        );
    }

    #[test]
    fn unknown_clock_and_clock_reversal_are_not_fresh() {
        let s = snapshot(None);
        assert_eq!(
            s.presence(s.captured_at, Duration::from_secs(1)).0,
            Presence::Unknown
        );
        let s = snapshot(Some(Duration::ZERO));
        assert_eq!(
            s.presence(
                s.captured_at - Duration::from_millis(1),
                Duration::from_secs(1)
            )
            .0,
            Presence::Unknown
        );
    }

    #[test]
    fn retained_or_missing_eye_never_proves_person_absence_or_fresh_presence() {
        let mut s = snapshot(Some(Duration::ZERO));
        let EyeEvidence::Direct(observation) = s.eye.clone() else {
            unreachable!()
        };
        s.eye = EyeEvidence::Retained(observation);
        assert_eq!(
            s.presence(s.captured_at, Duration::from_secs(1)).0,
            Presence::Unknown
        );
        s.eye = EyeEvidence::Unavailable("eye not observed in selected ROI");
        assert_eq!(
            s.presence(s.captured_at, Duration::from_secs(1)).0,
            Presence::Unknown
        );
    }

    #[test]
    fn each_product_ages_its_own_source() {
        let mut s = snapshot(Some(Duration::from_millis(10)));
        let EyeEvidence::Direct(eye) = &s.eye else {
            unreachable!()
        };
        let mut source = eye.source.clone();
        source.timestamp_ns = 7;
        source.sign_epoch = Some(4);
        s.gaze = Ok(Observation {
            value: SignedDirection::new([0.0, 0.0, 1.0]).unwrap(),
            source,
            source_arrival_age: Some(Duration::from_secs(1)),
        });
        assert_eq!(
            s.presence(s.captured_at, Duration::from_millis(500)).0,
            Presence::Present
        );
        assert_eq!(
            s.freshness(
                s.gaze.as_ref().unwrap(),
                s.captured_at,
                Duration::from_millis(500)
            ),
            Freshness::Stale
        );
        assert_eq!(
            s.gaze_status(s.captured_at, Duration::from_millis(500)),
            Err("stale gaze source")
        );
        s.gaze.as_mut().unwrap().source_arrival_age = Some(Duration::ZERO);
        assert!(s
            .gaze_status(s.captured_at, Duration::from_millis(500))
            .is_ok());
        s.availability = Err("tracking service unavailable");
        assert_eq!(
            s.gaze_status(s.captured_at, Duration::from_millis(500)),
            Err("tracking service unavailable")
        );
        assert_eq!(
            s.presence(s.captured_at, Duration::from_millis(500)).0,
            Presence::Present
        );
    }

    #[test]
    fn age_overflow_is_unknown() {
        let s = snapshot(Some(Duration::MAX));
        assert_eq!(
            s.presence(s.captured_at + Duration::from_millis(1), Duration::MAX)
                .0,
            Presence::Unknown
        );
    }

    #[test]
    fn direction_requires_finite_camera_facing_unit_vector() {
        for direction in [
            [f64::NAN, 0.0, 1.0],
            [f64::INFINITY, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0; 3],
        ] {
            assert!(SignedDirection::new(direction).is_err());
        }
        let direction = [0.6, 0.0, 0.8];
        assert_eq!(
            SignedDirection::new(direction)
                .unwrap()
                .right_down_toward_camera(),
            direction
        );
    }
}
