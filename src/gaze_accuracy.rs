//! Fixed-target validation, never a calibration fit. Score every fresh finite
//! cursor observation, including off-screen results; never select a good cluster.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

pub(crate) const HOLD: Duration = Duration::from_millis(2000);
pub(crate) const SETTLE: Duration = Duration::from_millis(650);
pub(crate) const TARGETS: [(f64, f64); 20] = [
    (0.5, 0.15),
    (0.125, 0.85),
    (0.875, 0.3833),
    (0.3125, 0.6167),
    (0.6875, 0.15),
    (0.5, 0.85),
    (0.125, 0.3833),
    (0.875, 0.6167),
    (0.3125, 0.15),
    (0.6875, 0.85),
    (0.5, 0.3833),
    (0.125, 0.6167),
    (0.875, 0.15),
    (0.3125, 0.85),
    (0.6875, 0.3833),
    (0.5, 0.6167),
    (0.125, 0.15),
    (0.875, 0.85),
    (0.3125, 0.3833),
    (0.6875, 0.6167),
];

pub(crate) struct Session {
    pub size: (usize, usize),
    pub index: usize,
    pub finished: Option<String>,
    pub report_path: Option<String>,
    pub report_error: Option<String>,
    pub mapping: Value,
    started: Option<Instant>,
    last_tick: Option<Instant>,
    source_floor: Option<u64>,
    last_raw: Option<u64>,
    last_sample: Option<u64>,
    samples: Vec<Vec<(u64, (f64, f64))>>,
    frames: [usize; 20],
    reasons: BTreeMap<String, usize>,
}
impl Session {
    pub fn new(size: (usize, usize), mapping: Value) -> Self {
        Self {
            size,
            index: 0,
            finished: None,
            report_path: None,
            report_error: None,
            mapping,
            started: None,
            last_tick: None,
            source_floor: None,
            last_raw: None,
            last_sample: None,
            samples: vec![vec![]; 20],
            frames: [0; 20],
            reasons: BTreeMap::new(),
        }
    }
    pub fn abort(&mut self, reason: &str) {
        if self.finished.is_none() {
            self.finished = Some(reason.into());
        }
    }
    pub fn elapsed(&self, now: Instant) -> Duration {
        self.started
            .map_or(Duration::ZERO, |s| now.saturating_duration_since(s))
    }
    pub fn observe(
        &mut self,
        now: Instant,
        raw_source: Option<u64>,
        observation: Option<(u64, (f64, f64))>,
        reason: &str,
    ) {
        if self.finished.is_some() {
            return;
        }
        if self
            .last_tick
            .is_some_and(|last| now.saturating_duration_since(last) > Duration::from_millis(750))
        {
            self.abort("PRESENTATION INTERRUPTED");
            return;
        }
        self.last_tick = Some(now);
        self.started.get_or_insert(now);
        if raw_source
            .zip(self.last_raw)
            .is_some_and(|(current, previous)| current < previous)
        {
            self.abort("SOURCE CLOCK RESTARTED");
            return;
        }
        let fresh_raw = raw_source.is_some() && raw_source != self.last_raw;
        if raw_source.is_some() {
            self.last_raw = raw_source;
        }
        if self.elapsed(now) >= SETTLE {
            // Establish the source boundary AFTER settling, not merely when
            // a delayed SAM answer arrives at the new screen target.
            if self.source_floor.is_none() {
                self.source_floor = raw_source;
            }
            if fresh_raw {
                self.frames[self.index] += 1;
                if observation.is_none() {
                    *self.reasons.entry(reason.into()).or_default() += 1;
                }
            }
            if let Some((source, point)) = observation {
                if self.source_floor.is_some_and(|floor| source > floor)
                    && raw_source.is_some_and(|raw| source <= raw)
                    && self.last_sample.is_none_or(|last| source > last)
                    && point.0.is_finite()
                    && point.1.is_finite()
                {
                    self.samples[self.index].push((source, point));
                    self.last_sample = Some(source);
                }
            }
        }
        if self.elapsed(now) >= HOLD {
            self.index += 1;
            self.source_floor = None;
            self.started = Some(now);
            if self.index == TARGETS.len() {
                self.finished = Some("COMPLETE".into());
            }
        }
    }
    pub fn report(&self) -> Value {
        let mut all_errors = vec![];
        let mut target_mean_errors = vec![];
        let (w, h) = (
            self.size.0.saturating_sub(1) as f64,
            self.size.1.saturating_sub(1) as f64,
        );
        let targets:Vec<_>=TARGETS.iter().enumerate().map(|(i,&target)| {
            let samples=&self.samples[i];
            let errors:Vec<_>=samples.iter().map(|(_,p)|((p.0-target.0)*w).hypot((p.1-target.1)*h)).collect();
            all_errors.extend(&errors);
            let mean=(!errors.is_empty()).then(||errors.iter().sum::<f64>()/errors.len() as f64);
            if let Some(e)=mean{target_mean_errors.push(e);}
            let center=(!samples.is_empty()).then(|| {
                (samples.iter().map(|(_,p)|p.0).sum::<f64>()/samples.len() as f64,
                 samples.iter().map(|(_,p)|p.1).sum::<f64>()/samples.len() as f64)
            });
            let jitter=center.map(|c| (samples.iter().map(|(_,p)|((p.0-c.0)*w).powi(2)+((p.1-c.1)*h).powi(2))
                .sum::<f64>()/samples.len() as f64).sqrt());
            json!({"target":target,"fresh_samples":samples.len(),"sampling_window_raw_frames":self.frames[i],
                "mean_error_px":mean,"median_error_px":percentile(&errors,0.5),"mean_cursor":center,
                "bias_px":center.map(|p|[(p.0-target.0)*w,(p.1-target.1)*h]),"rms_jitter_px":jitter,
                "samples":samples.iter().map(|(clock,p)|json!({"source_ns":clock.to_string(),"cursor":p})).collect::<Vec<_>>()})
        }).collect();
        let mean = (!target_mean_errors.is_empty())
            .then(|| target_mean_errors.iter().sum::<f64>() / target_mean_errors.len() as f64);
        json!({"schema":"buttercup-gaze-accuracy-v1","outcome":self.finished,"resolution":self.size,
            "hold_ms":HOLD.as_millis(),"settle_ms":SETTLE.as_millis(),"mapping":self.mapping,
            "targets_with_samples":target_mean_errors.len(),"targets_total":TARGETS.len(),
            "well_sampled_targets":self.samples.iter().filter(|s|s.len()>=3).count(),
            "fresh_samples":all_errors.len(),"mean_target_error_px":mean,
            "mean_target_error_percent_diagonal":mean.map(|e|e/w.hypot(h)*100.0),
            "median_sample_error_px":percentile(&all_errors,0.5),"p95_sample_error_px":percentile(&all_errors,0.95),
            "missing_observation_reasons_by_raw_frame":self.reasons,"targets":targets,
            "limitations":"User fixation assumed, not independently verified. Missing targets are missing, not zero error. Raw frames and processed gaze samples have different rates. No refit, no clamping, no stability-based cherry-picking. Pixel metrics use this presentation buffer, not physical millimeters or calibrated angular error."})
    }
}
fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[((sorted.len() - 1) as f64 * fraction).round() as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn twenty_unique_targets_are_separate_from_calibration() {
        for (i, target) in TARGETS.iter().enumerate() {
            assert!(!TARGETS[..i].contains(target));
            assert!(!crate::gaze_target_solver::VIRTUAL_MOUSE_CALIBRATION_TARGETS.contains(target));
        }
    }
    fn replay(offset: (f64, f64), missing: bool) -> Session {
        let mut s = Session::new((1001, 1001), json!({"frozen":true}));
        let start = Instant::now();
        let mut clock = 1000;
        for i in 0..=400 {
            let target = TARGETS[s.index.min(19)];
            clock += 100;
            s.observe(
                start + Duration::from_millis(i * 100),
                Some(clock),
                (!missing).then_some((clock, (target.0 + offset.0, target.1 + offset.1))),
                "NO GAZE",
            );
        }
        s
    }
    #[test]
    fn perfect_and_known_offset_score_without_refitting() {
        let perfect = replay((0.0, 0.0), false);
        assert_eq!(perfect.report()["targets_with_samples"], 20);
        assert_eq!(perfect.report()["mean_target_error_px"], 0.0);
        let biased = replay((0.03, 0.04), false);
        assert!((biased.report()["mean_target_error_px"].as_f64().unwrap() - 50.0).abs() < 1e-8);
        assert_eq!(biased.mapping, json!({"frozen":true}));
        assert!(
            replay((2.0, 0.0), false).report()["mean_target_error_px"]
                .as_f64()
                .unwrap()
                > 1999.0
        );
    }
    #[test]
    fn missing_targets_are_not_perfect_accuracy() {
        let r = replay((0.0, 0.0), true).report();
        assert_eq!(r["targets_with_samples"], 0);
        assert!(r["mean_target_error_px"].is_null());
        assert_eq!(r["outcome"], "COMPLETE");
    }
    #[test]
    fn stale_duplicate_and_future_gaze_cannot_inflate_samples() {
        let mut s = Session::new((1001, 1001), json!({}));
        let now = Instant::now();
        for i in 0..12 {
            s.observe(
                now + Duration::from_millis(i * 100),
                Some(100 + i),
                Some((99, TARGETS[0])),
                "WAIT",
            );
        }
        assert_eq!(s.samples[0].len(), 0);
        s.observe(
            now + Duration::from_millis(1200),
            Some(112),
            Some((112, TARGETS[0])),
            "WAIT",
        );
        s.observe(
            now + Duration::from_millis(1250),
            Some(112),
            Some((112, TARGETS[0])),
            "WAIT",
        );
        s.observe(
            now + Duration::from_millis(1300),
            Some(113),
            Some((999, TARGETS[0])),
            "WAIT",
        );
        assert_eq!(s.samples[0].len(), 1);
    }
    #[test]
    fn pause_or_clock_restart_aborts_without_skipping_targets() {
        let now = Instant::now();
        let mut s = Session::new((100, 100), json!({}));
        s.observe(now, Some(100), None, "WAIT");
        s.observe(now + Duration::from_secs(10), Some(101), None, "WAIT");
        assert_eq!(s.index, 0);
        assert_eq!(s.finished.as_deref(), Some("PRESENTATION INTERRUPTED"));
        let mut s = Session::new((100, 100), json!({}));
        s.observe(now, Some(100), None, "WAIT");
        s.observe(now + Duration::from_millis(100), Some(99), None, "WAIT");
        assert_eq!(s.finished.as_deref(), Some("SOURCE CLOCK RESTARTED"));
    }
}
