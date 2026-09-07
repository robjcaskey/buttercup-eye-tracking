//! Recorded motion stimulus before stationary calibration, not another gaze
//! solver. Target coordinates NEVER choose a sign or enter the monitor fit.
use std::time::{Duration, Instant};

pub(crate) const MAX_DURATION: Duration = Duration::from_secs(20);
pub(crate) const REQUIRED_SOURCES: usize = 4;
const MIN_SOURCE_SPAN_NS: u64 = 300_000_000;
const MAX_SOURCE_GAP_NS: u64 = 750_000_000;

#[derive(Clone, Copy)]
pub(crate) struct QualifiedSign {
    pub(crate) source_ns: u64,
    pub(crate) epoch: u64,
    pub(crate) sustained_support: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Update {
    Waiting,
    Ready,
    TimedOut,
}

#[derive(Default)]
pub(crate) struct Acquisition {
    started: Option<Instant>,
    accumulated: Duration,
    pub(crate) episodes: u32,
    lineage: Option<u64>,
    last_source: Option<u64>,
    last_fresh_at: Option<Instant>,
    first_ready_source: Option<u64>,
    epoch: Option<u64>,
    pub(crate) ready_sources: usize,
    pub(crate) sustained_support: bool,
}

impl Acquisition {
    pub(crate) fn active(&self) -> bool {
        self.started.is_some()
    }

    pub(crate) fn start(&mut self, now: Instant) {
        if self.active() {
            return;
        }
        self.started = Some(now);
        self.episodes = self.episodes.saturating_add(1);
        self.lineage = None;
        self.last_source = None;
        self.last_fresh_at = None;
        self.clear_votes();
    }

    fn clear_votes(&mut self) {
        self.first_ready_source = None;
        self.epoch = None;
        self.ready_sources = 0;
        self.sustained_support = false;
    }

    pub(crate) fn elapsed(&self, now: Instant) -> Duration {
        self.started
            .map_or(Duration::ZERO, |t| now.saturating_duration_since(t))
    }

    pub(crate) fn total_elapsed(&self, now: Instant) -> Duration {
        self.accumulated + self.elapsed(now)
    }

    pub(crate) fn target(&self, now: Instant) -> (f64, f64) {
        let phase = self.elapsed(now).as_secs_f64() * std::f64::consts::TAU / 6.4;
        (0.5 + 0.095 * phase.sin(), 0.5 + 0.095 * (2.0 * phase).sin())
    }

    pub(crate) fn observe(
        &mut self,
        now: Instant,
        lineage: u64,
        sample: Option<QualifiedSign>,
    ) -> Update {
        if !self.active() {
            return Update::Ready;
        }
        if self.elapsed(now) >= MAX_DURATION {
            return Update::TimedOut;
        }
        if self.lineage != Some(lineage) {
            self.lineage = Some(lineage);
            self.last_source = None;
            self.last_fresh_at = None;
            self.clear_votes();
        }
        if self.last_fresh_at.is_some_and(|t| {
            now.saturating_duration_since(t).as_nanos() > MAX_SOURCE_GAP_NS as u128
        }) {
            self.clear_votes();
        }
        let Some(sample) = sample else {
            self.clear_votes();
            return Update::Waiting;
        };
        if self
            .last_source
            .is_some_and(|last| sample.source_ns <= last)
        {
            return Update::Waiting;
        }
        if self.epoch != Some(sample.epoch)
            || self
                .last_source
                .is_some_and(|last| sample.source_ns - last > MAX_SOURCE_GAP_NS)
        {
            self.clear_votes();
        }
        self.epoch = Some(sample.epoch);
        self.last_source = Some(sample.source_ns);
        self.last_fresh_at = Some(now);
        let first = *self.first_ready_source.get_or_insert(sample.source_ns);
        self.ready_sources = self.ready_sources.saturating_add(1);
        self.sustained_support |= sample.sustained_support;
        if self.ready_sources >= REQUIRED_SOURCES
            && self.sustained_support
            && sample.source_ns.saturating_sub(first) >= MIN_SOURCE_SPAN_NS
        {
            self.accumulated += self.elapsed(now);
            self.started = None;
            Update::Ready
        } else {
            Update::Waiting
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(ns: u64, epoch: u64, support: bool) -> Option<QualifiedSign> {
        Some(QualifiedSign {
            source_ns: ns,
            epoch,
            sustained_support: support,
        })
    }

    #[test]
    fn target_moves_smoothly_inside_middle_twenty_percent() {
        let now = Instant::now();
        let mut a = Acquisition::default();
        a.start(now);
        let mut previous = a.target(now);
        for ms in (0..20_000).step_by(10) {
            let p = a.target(now + Duration::from_millis(ms));
            assert!((0.4..=0.6).contains(&p.0) && (0.4..=0.6).contains(&p.1));
            assert!((p.0 - previous.0).hypot(p.1 - previous.1) < 0.003);
            previous = p;
        }
        assert_ne!(a.target(now), a.target(now + Duration::from_secs(1)));
    }

    #[test]
    fn acquisition_requires_supported_fresh_sources_not_elapsed_time_or_duplicates() {
        let now = Instant::now();
        let mut a = Acquisition::default();
        a.start(now);
        for i in 0..5 {
            let t = now + Duration::from_millis(i * 100);
            assert_eq!(
                a.observe(t, 1, sample(1 + i * 100_000_000, 9, false)),
                Update::Waiting
            );
        }
        assert_eq!(
            a.observe(
                now + Duration::from_millis(401),
                1,
                sample(400_000_001, 9, true)
            ),
            Update::Waiting
        );
        assert_eq!(a.ready_sources, 5);
        assert_eq!(
            a.observe(
                now + Duration::from_millis(500),
                1,
                sample(500_000_001, 9, true)
            ),
            Update::Ready
        );
        assert!(!a.active());
    }

    #[test]
    fn gaps_sign_changes_and_new_sensor_lineages_cannot_pool_votes() {
        let now = Instant::now();
        let mut a = Acquisition::default();
        a.start(now);
        for i in 0..3 {
            assert_eq!(
                a.observe(
                    now + Duration::from_millis(i * 100),
                    1,
                    sample(1 + i * 100_000_000, 0, true)
                ),
                Update::Waiting
            );
        }
        assert_eq!(
            a.observe(
                now + Duration::from_millis(300),
                1,
                sample(300_000_001, 1, true)
            ),
            Update::Waiting
        );
        assert_eq!(a.ready_sources, 1);
        assert_eq!(
            a.observe(now + Duration::from_millis(400), 2, sample(1, 1, true)),
            Update::Waiting
        );
        assert_eq!(a.ready_sources, 1);
        assert_eq!(
            a.observe(
                now + Duration::from_secs(2),
                2,
                sample(1_600_000_001, 1, true)
            ),
            Update::Waiting
        );
        assert_eq!(a.ready_sources, 1);
        a.observe(now + Duration::from_millis(2010), 2, None);
        a.observe(
            now + Duration::from_millis(2020),
            2,
            sample(1_600_000_001, 1, true),
        );
        assert_eq!(
            a.ready_sources, 0,
            "a held old sample cannot start another confirmation window"
        );
        assert_eq!(
            a.observe(now + MAX_DURATION, 99, sample(1, 1, true)),
            Update::TimedOut
        );
    }
}
