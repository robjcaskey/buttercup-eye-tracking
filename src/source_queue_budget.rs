//! Anticipatory host work admission for synchronized ROI reads.
//!
//! This does not relax source freshness, alter camera cadence, buffer images,
//! or replace the caller's hard queue limits. It can only reject extra work
//! before an already-late first ROI consumes the second ROI's remaining time.
use std::time::{Duration, Instant};

// A rejected head cannot supply a new work sample. Bound history in time as
// well as count so context/recording work above the fresh-start allowance
// cannot make one obsolete overrun a self-sustaining admission deadlock.
const WORK_SAMPLE_MAX_AGE: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ReadKey {
    pub(crate) sequence: u64,
    pub(crate) timestamp_ns: u64,
    pub(crate) region_session: u64,
    pub(crate) region_generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourcePart {
    First(ReadKey),
    Last(ReadKey),
    Independent,
    Auxiliary,
}

struct PendingRead {
    key: ReadKey,
    started: Instant,
    admitted: bool,
}

pub(crate) struct SourceQueueBudget {
    queue_limit: Duration,
    pending: Option<PendingRead>,
    first_roi_times: [Option<(Instant, Duration)>; 8],
    next_sample: usize,
}

impl SourceQueueBudget {
    pub(crate) fn new(queue_limit: Duration) -> Self {
        Self {
            queue_limit,
            pending: None,
            first_roi_times: [None; 8],
            next_sample: 0,
        }
    }

    /// `would_consume` comes from the existing age/warmup/control policy.
    /// A false result grants no permission: that original policy still wins.
    /// Only a same-read tail can measure work spent after admitting its head.
    pub(crate) fn should_shed(
        &mut self,
        part: SourcePart,
        now: Instant,
        queue_age: Duration,
        would_consume: bool,
    ) -> bool {
        match part {
            SourcePart::First(key) => {
                let observed = self
                    .first_roi_times
                    .iter()
                    .flatten()
                    .filter(|(at, _)| now.saturating_duration_since(*at) <= WORK_SAMPLE_MAX_AGE)
                    .map(|(_, elapsed)| *elapsed)
                    .max()
                    .unwrap_or(Duration::ZERO);
                let reserve = if observed.is_zero() {
                    Duration::ZERO
                } else {
                    observed.saturating_add(Duration::from_millis(2))
                };
                // Allow immediate genuinely fresh starts. History expiry also
                // permits recovery when recurring context/recording work
                // leaves every packet outside this small fresh-start window.
                let fresh_start = queue_age <= Duration::from_millis(20).min(self.queue_limit / 4);
                let shed = would_consume
                    && !fresh_start
                    && queue_age.saturating_add(reserve) > self.queue_limit;
                self.pending = Some(PendingRead {
                    key,
                    started: now,
                    admitted: would_consume && !shed,
                });
                shed
            }
            SourcePart::Last(key) => {
                let Some(pending) = self.pending.take().filter(|pending| pending.key == key) else {
                    return false;
                };
                if pending.admitted {
                    self.first_roi_times[self.next_sample] =
                        Some((now, now.saturating_duration_since(pending.started)));
                    self.next_sample = (self.next_sample + 1) % self.first_roi_times.len();
                    false
                } else {
                    // The first ROI was shed. A fresher-arriving companion is
                    // not a usable stereo read and must not waste another pass.
                    would_consume
                }
            }
            SourcePart::Independent => {
                self.pending = None;
                false
            }
            SourcePart::Auxiliary => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(sequence: u64) -> ReadKey {
        ReadKey {
            sequence,
            timestamp_ns: sequence * 100_000_000,
            region_session: 5,
            region_generation: 0,
        }
    }
    fn ms(value: u64) -> Duration {
        Duration::from_millis(value)
    }

    #[test]
    fn sustained_load_preserves_pairs_without_consuming_stale_packets() {
        let start = Instant::now();
        let replay = |anticipate: bool| {
            let mut budget = SourceQueueBudget::new(ms(200));
            let mut now = 0;
            let mut complete = 0;
            let mut partial = 0;
            let mut consumed = 0;
            for read in 0..40 {
                let arrival = read * 100;
                let mut admitted = 0;
                for first in [true, false] {
                    now = now.max(arrival);
                    let age = ms(now - arrival);
                    let ordinary = age <= ms(200);
                    let part = if first {
                        SourcePart::First(key(read))
                    } else {
                        SourcePart::Last(key(read))
                    };
                    let shed = budget.should_shed(part, start + ms(now), age, ordinary);
                    if ordinary && (!anticipate || !shed) {
                        assert!(age <= ms(200), "no freshness exception for the second ROI");
                        admitted += 1;
                        consumed += 1;
                        now += 100;
                    }
                }
                complete += usize::from(admitted == 2);
                partial += usize::from(admitted == 1);
            }
            (complete, partial, consumed)
        };
        let baseline = replay(false);
        let candidate = replay(true);
        assert!(
            baseline.1 > 20,
            "control reproduces one-sided starvation: {baseline:?}"
        );
        assert_eq!(
            candidate.1, 0,
            "steady bounded work admits or sheds whole reads"
        );
        assert!(
            candidate.0 > baseline.0 && candidate.2 > 0,
            "{baseline:?} -> {candidate:?}"
        );
    }

    #[test]
    fn independent_rois_and_mismatched_clocks_do_not_inherit_a_pair_drop() {
        let now = Instant::now();
        let mut budget = SourceQueueBudget::new(ms(200));
        assert!(!budget.should_shed(SourcePart::First(key(1)), now, ms(250), false));
        let mut other = key(1);
        other.region_generation += 1;
        assert!(!budget.should_shed(SourcePart::Last(other), now, ms(10), true));
        budget.should_shed(SourcePart::First(key(2)), now, ms(250), false);
        assert!(!budget.should_shed(SourcePart::Independent, now, ms(10), true));
        assert!(!budget.should_shed(SourcePart::Last(key(2)), now, ms(10), true));
        budget.should_shed(SourcePart::First(key(3)), now, ms(250), false);
        assert!(budget.should_shed(SourcePart::Last(key(3)), now, ms(10), true));
    }

    #[test]
    fn over_budget_work_never_earns_stale_grace_or_permanent_eviction() {
        let now = Instant::now();
        let mut budget = SourceQueueBudget::new(ms(200));
        budget.should_shed(SourcePart::First(key(1)), now, ms(0), true);
        assert!(!budget.should_shed(SourcePart::Last(key(1)), now + ms(400), ms(400), false));
        assert!(budget.should_shed(SourcePart::First(key(2)), now + ms(500), ms(100), true));
        assert!(
            !budget.should_shed(SourcePart::First(key(3)), now + ms(600), ms(0), true),
            "a fresh source can retire a past slow-work measurement"
        );
    }

    #[test]
    fn old_work_expires_even_when_context_work_prevents_a_fresh_start() {
        let now = Instant::now();
        let mut budget = SourceQueueBudget::new(ms(200));
        budget.should_shed(SourcePart::First(key(1)), now, ms(0), true);
        budget.should_shed(SourcePart::Last(key(1)), now + ms(400), ms(400), false);
        for read in 2..=11 {
            let visit = now + ms(300 + read * 100);
            assert!(budget.should_shed(SourcePart::First(key(read)), visit, ms(30), true));
            assert!(budget.should_shed(SourcePart::Last(key(read)), visit, ms(28), true));
        }
        // None of these packets is <=20 ms old: the fresh-start exception
        // alone would otherwise leave the pair permanently unobserved.
        assert!(!budget.should_shed(SourcePart::First(key(12)), now + ms(1500), ms(30), true));
        assert!(!budget.should_shed(SourcePart::Last(key(12)), now + ms(1550), ms(78), true));
        assert!(!budget.should_shed(SourcePart::First(key(13)), now + ms(1600), ms(30), true));
    }

    #[derive(Debug, Default)]
    struct ReplayCounts {
        pairs: usize,
        partial: usize,
        consumed: usize,
        stale_consumed: usize,
    }

    fn replay_variable_load(
        period_ms: u64,
        base_work_ms: u64,
        jitter_ms: u64,
        occasional_stall: bool,
        anticipate: bool,
    ) -> ReplayCounts {
        let start = Instant::now();
        let mut budget = SourceQueueBudget::new(ms(200));
        let mut counts = ReplayCounts::default();
        let mut now = 0;
        let mut random = 12345_u64;
        for read in 0..400 {
            let arrival = read * period_ms;
            let mut admitted = 0;
            // Context precedes the eye pair, as in the camera protocol. This
            // shared work and packet arrival skew are not fresh eye evidence.
            now = now.max(arrival);
            budget.should_shed(
                SourcePart::Auxiliary,
                start + ms(now),
                ms(now - arrival),
                now - arrival <= 200,
            );
            if now - arrival <= 200 {
                now += 8;
            }
            for first in [true, false] {
                let packet_arrival = arrival + if first { 1 } else { 3 };
                now = now.max(packet_arrival);
                let age = ms(now - packet_arrival);
                let ordinary = age <= ms(200);
                let part = if first {
                    SourcePart::First(key(read))
                } else {
                    SourcePart::Last(key(read))
                };
                let shed = budget.should_shed(part, start + ms(now), age, ordinary);
                // Draw costs for every source, even when shed, so both
                // schedulers see the same predefined workload/interruptions.
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                let work = base_work_ms
                    + (random >> 32) % (jitter_ms + 1)
                    + if occasional_stall && read % 83 == 20 && first {
                        180
                    } else {
                        0
                    };
                if ordinary && (!anticipate || !shed) {
                    counts.stale_consumed += usize::from(age > ms(200));
                    counts.consumed += 1;
                    admitted += 1;
                    now += work;
                }
            }
            counts.pairs += usize::from(admitted == 2);
            counts.partial += usize::from(admitted == 1);
        }
        counts
    }

    #[test]
    fn paired_work_budget_recovers_under_arrival_skew_jitter_and_occasional_stalls() {
        for (period, work, jitter, stalls) in [
            (200, 60, 20, false),
            (97, 80, 25, false),
            (97, 80, 25, true),
            (200, 60, 20, true),
            (97, 30, 20, true),
        ] {
            let baseline = replay_variable_load(period, work, jitter, stalls, false);
            let candidate = replay_variable_load(period, work, jitter, stalls, true);
            eprintln!("queue workload period={period} work={work}+0..{jitter} stalls={stalls}: {baseline:?} -> {candidate:?}");
            assert_eq!(candidate.stale_consumed, 0);
            assert!(
                candidate.partial <= baseline.partial,
                "one-sided starvation regressed"
            );
            if period == 97 && work == 80 {
                assert!(
                    candidate.pairs > baseline.pairs * 8,
                    "sustained overload should preserve many more complete reads"
                );
            } else {
                // The largest recent work sample intentionally reserves
                // headroom after a surprising stall. Bound, and expose, the
                // extra complete-read loss instead of claiming free recovery.
                assert!(
                    candidate.pairs + 10 >= baseline.pairs,
                    "more than two extra pairs shed per isolated stall"
                );
            }
            if period == 200 && !stalls {
                assert_eq!(candidate.pairs, 400);
            }
        }
    }

    #[test]
    fn temporarily_evicted_eye_cannot_leave_a_stale_paired_admission_pending() {
        let now = Instant::now();
        let mut budget = SourceQueueBudget::new(ms(200));
        budget.should_shed(SourcePart::First(key(1)), now, ms(250), false);
        for source in 2..40 {
            assert!(!budget.should_shed(
                SourcePart::Independent,
                now + ms(source * 100),
                ms(20),
                true
            ));
        }
        let mut readmitted = key(40);
        readmitted.region_generation = 2;
        assert!(!budget.should_shed(SourcePart::First(readmitted), now + ms(4000), ms(20), true));
        assert!(!budget.should_shed(SourcePart::Last(readmitted), now + ms(4050), ms(70), true));
    }
}
