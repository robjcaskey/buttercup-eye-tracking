//! Source/identity continuity shared by live ingestion and offline replay.
//!
//! A crop origin is a coordinate window, not an eye identity or source clock.
//! This module decides only whether existing identity may survive. Overlap is
//! not proof of valid anatomy, motion, or visibility; those gates remain with
//! their evidence owners. No transported result is made fresh here.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoiSource {
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoiDiscontinuity {
    InvalidGeometry,
    DimensionsChanged,
    SourceGap,
    NoSensorOverlap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoiTransition {
    First,
    Continuous,
    CompatibleTranslation,
    /// Another presentation of one exposure, even if its crop has changed.
    Duplicate,
    /// A late/inconsistent packet cannot rewind either source clock.
    OutOfOrder,
    Discontinuity(RoiDiscontinuity),
}

impl RoiTransition {
    pub fn accepts_source(self) -> bool {
        !matches!(
            self,
            Self::Duplicate
                | Self::OutOfOrder
                | Self::Discontinuity(RoiDiscontinuity::InvalidGeometry)
        )
    }

    pub fn resets_tracking(self) -> bool {
        matches!(self, Self::First | Self::Discontinuity(_))
    }
}

pub fn classify_roi_transition(
    previous: Option<RoiSource>,
    current: RoiSource,
    maximum_gap_ns: u64,
) -> RoiTransition {
    if current.width == 0 || current.height == 0 {
        return RoiTransition::Discontinuity(RoiDiscontinuity::InvalidGeometry);
    }
    let Some(previous) = previous else {
        return RoiTransition::First;
    };
    // Equal timestamps are not a fresh exposure, including re-crops and a
    // producer that changed its packet sequence while reusing the ROI buffer.
    if current.timestamp_ns == previous.timestamp_ns {
        return RoiTransition::Duplicate;
    }
    if current.timestamp_ns < previous.timestamp_ns {
        return RoiTransition::OutOfOrder;
    }
    if current.timestamp_ns - previous.timestamp_ns > maximum_gap_ns {
        return RoiTransition::Discontinuity(RoiDiscontinuity::SourceGap);
    }
    if current.sequence <= previous.sequence {
        return RoiTransition::OutOfOrder;
    }
    if (current.width, current.height) != (previous.width, previous.height) {
        return RoiTransition::Discontinuity(RoiDiscontinuity::DimensionsChanged);
    }
    if (current.sensor_x, current.sensor_y) == (previous.sensor_x, previous.sensor_y) {
        return RoiTransition::Continuous;
    }
    // u128 also handles malformed dimensions without integer wrapping.
    let overlaps = u128::from(current.sensor_x)
        < u128::from(previous.sensor_x) + previous.width as u128
        && u128::from(previous.sensor_x) < u128::from(current.sensor_x) + current.width as u128
        && u128::from(current.sensor_y) < u128::from(previous.sensor_y) + previous.height as u128
        && u128::from(previous.sensor_y) < u128::from(current.sensor_y) + current.height as u128;
    if overlaps {
        RoiTransition::CompatibleTranslation
    } else {
        RoiTransition::Discontinuity(RoiDiscontinuity::NoSensorOverlap)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoiResultSource {
    pub tracking_epoch: u64,
    pub sequence: u64,
    pub timestamp_ns: u64,
}

/// One eye's logical tracking session; it does not manufacture a sensor clock.
#[derive(Clone, Debug, Default)]
pub struct RoiContinuitySession {
    epoch: u64,
    selection_started_ns: Option<u64>,
    last_source: Option<RoiSource>,
}

impl RoiContinuitySession {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn selection_started_ns(&self) -> Option<u64> {
        self.selection_started_ns
    }

    pub fn last_source(&self) -> Option<RoiSource> {
        self.last_source
    }

    /// Explicit deselection, prompt/identity change or camera reconnect.
    /// Invalidate immediately so queued results cannot reappear on re-entry.
    pub fn invalidate(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.selection_started_ns = None;
        self.last_source = None;
    }

    pub fn observe(&mut self, source: RoiSource, maximum_gap_ns: u64) -> RoiTransition {
        let transition = classify_roi_transition(self.last_source, source, maximum_gap_ns);
        if !transition.accepts_source() {
            return transition;
        }
        if transition.resets_tracking() {
            self.epoch = self.epoch.wrapping_add(1);
            self.selection_started_ns = Some(source.timestamp_ns);
        }
        self.last_source = Some(source);
        transition
    }

    /// Shared asynchronous-result gate. Coordinate origin is deliberately
    /// absent: downstream transport consumes the result's original geometry.
    /// `replaced` prevents late completions from replacing a newer observation.
    pub fn admits_result(
        &self,
        result: RoiResultSource,
        replaced: Option<RoiResultSource>,
        maximum_age_ns: u64,
    ) -> bool {
        let (Some(started), Some(current)) = (self.selection_started_ns, self.last_source) else {
            return false;
        };
        result.tracking_epoch == self.epoch
            && result.timestamp_ns >= started
            && result.timestamp_ns <= current.timestamp_ns
            && result.sequence <= current.sequence
            && current.timestamp_ns - result.timestamp_ns <= maximum_age_ns
            && replaced.is_none_or(|old| {
                old.tracking_epoch != result.tracking_epoch
                    || (result.timestamp_ns > old.timestamp_ns && result.sequence > old.sequence)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_AGE: u64 = 900_000_000;

    fn source(sequence: u64, x: u32, y: u32) -> RoiSource {
        RoiSource {
            sequence,
            timestamp_ns: sequence * 20_000_000,
            sensor_x: x,
            sensor_y: y,
            width: 420,
            height: 280,
        }
    }

    fn result(session: &RoiContinuitySession, source: RoiSource) -> RoiResultSource {
        RoiResultSource {
            tracking_epoch: session.epoch(),
            sequence: source.sequence,
            timestamp_ns: source.timestamp_ns,
        }
    }

    #[test]
    fn positive_negative_and_repeated_nudges_preserve_epoch_and_source_start() {
        let mut session = RoiContinuitySession::default();
        assert_eq!(
            session.observe(source(1, 3000, 2000), MAX_AGE),
            RoiTransition::First
        );
        let epoch = session.epoch();
        let start = session.selection_started_ns();
        for (sequence, x, y) in [(2, 3032, 2024), (3, 2984, 1980), (4, 3000, 2000)] {
            assert_eq!(
                session.observe(source(sequence, x, y), MAX_AGE),
                RoiTransition::CompatibleTranslation
            );
            assert_eq!(session.epoch(), epoch);
            assert_eq!(session.selection_started_ns(), start);
        }
    }

    #[test]
    fn one_eye_reframe_does_not_reset_the_other_or_change_the_shared_exposure_clock() {
        let mut sessions = [
            RoiContinuitySession::default(),
            RoiContinuitySession::default(),
        ];
        sessions[0].observe(source(1, 3000, 2000), MAX_AGE);
        sessions[1].observe(source(1, 4000, 2000), MAX_AGE);
        let epochs = sessions.each_ref().map(|session| session.epoch());
        sessions[0].observe(source(2, 3032, 2024), MAX_AGE);
        sessions[1].observe(source(2, 4000, 2000), MAX_AGE);
        assert_eq!(sessions.each_ref().map(|session| session.epoch()), epochs);
        assert_eq!(
            sessions[0].last_source().unwrap().timestamp_ns,
            sessions[1].last_source().unwrap().timestamp_ns
        );
        sessions[0].observe(source(3, 0, 0), MAX_AGE);
        assert_ne!(sessions[0].epoch(), epochs[0]);
        assert_eq!(sessions[1].epoch(), epochs[1]);
    }

    #[test]
    fn duplicate_recrop_and_out_of_order_packet_do_not_rewind_or_refresh_history() {
        let mut session = RoiContinuitySession::default();
        let current = source(4, 3000, 2000);
        session.observe(current, MAX_AGE);
        let epoch = session.epoch();
        for duplicate in [
            current,
            RoiSource {
                sensor_x: 3032,
                sequence: 5,
                ..current
            },
        ] {
            assert_eq!(
                session.observe(duplicate, MAX_AGE),
                RoiTransition::Duplicate
            );
        }
        for old in [
            source(3, 3000, 2000),
            RoiSource {
                timestamp_ns: current.timestamp_ns + 1,
                ..current
            },
        ] {
            assert_eq!(session.observe(old, MAX_AGE), RoiTransition::OutOfOrder);
        }
        assert_eq!(session.epoch(), epoch);
        assert_eq!(session.last_source(), Some(current));
    }

    #[test]
    fn true_discontinuities_and_explicit_identity_changes_reject_old_inflight_results() {
        for changed in [
            source(100, 3000, 2000),
            RoiSource {
                width: 416,
                ..source(2, 3000, 2000)
            },
            source(2, 3420, 2000),
        ] {
            let mut session = RoiContinuitySession::default();
            let before = source(1, 3000, 2000);
            session.observe(before, MAX_AGE);
            let in_flight = result(&session, before);
            assert!(session.observe(changed, MAX_AGE).resets_tracking());
            assert!(!session.admits_result(in_flight, None, MAX_AGE));
        }
        let mut session = RoiContinuitySession::default();
        let before = source(1, 3000, 2000);
        session.observe(before, MAX_AGE);
        let in_flight = result(&session, before);
        session.invalidate();
        assert!(!session.admits_result(in_flight, None, MAX_AGE));
        session.observe(source(2, 3000, 2000), MAX_AGE);
        assert!(!session.admits_result(in_flight, None, MAX_AGE));
    }

    #[test]
    fn advancing_source_gap_can_recover_a_restarted_sequence_counter() {
        let mut session = RoiContinuitySession::default();
        let before = source(100, 3000, 2000);
        session.observe(before, MAX_AGE);
        let in_flight = result(&session, before);
        let restarted = RoiSource {
            sequence: 1,
            timestamp_ns: before.timestamp_ns + MAX_AGE + 1,
            ..before
        };
        assert_eq!(
            session.observe(restarted, MAX_AGE),
            RoiTransition::Discontinuity(RoiDiscontinuity::SourceGap)
        );
        assert_eq!(session.last_source(), Some(restarted));
        assert!(!session.admits_result(in_flight, None, MAX_AGE));
    }

    #[test]
    fn move_in_flight_preserves_result_identity_without_renewing_its_age() {
        let mut session = RoiContinuitySession::default();
        let before = source(1, 3000, 2000);
        session.observe(before, MAX_AGE);
        let in_flight = result(&session, before);
        session.observe(source(2, 3032, 2024), MAX_AGE);
        assert!(session.admits_result(in_flight, None, MAX_AGE));
        assert_eq!(in_flight.timestamp_ns, before.timestamp_ns);
        assert!(!session.admits_result(in_flight, Some(in_flight), MAX_AGE));
        let future = result(&session, source(3, 3032, 2024));
        assert!(!session.admits_result(future, None, MAX_AGE));
        for sequence in 3..=47 {
            session.observe(
                source(sequence, 3032 + (sequence as u32 % 2) * 4, 2024),
                MAX_AGE,
            );
        }
        assert_eq!(session.epoch(), in_flight.tracking_epoch);
        assert!(!session.admits_result(in_flight, None, MAX_AGE));
    }
}
