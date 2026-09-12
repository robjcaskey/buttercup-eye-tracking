//! Opt-in, absolute desktop pointer output. No clicks, acceleration or easing.
//!
//! Device lifetime is the enable switch: OFF drops the uinput descriptor. An
//! open/setup/write failure is fail-closed; there is no permission escalation.
use serde_json::{json, Value};
use std::io;
use std::time::{Duration, Instant};

mod linux;
pub(crate) use linux::UinputPointer;

pub(crate) const DEVICE_NAME: &str = "Buttercup Gaze Pointer";
pub(crate) const DEVICE_IDENTIFIER: &str = "0:0:Buttercup_Gaze_Pointer";
pub(crate) const AXIS_MAX: i32 = 65_535;
// A liveness bound, not a measured sensor-to-host latency or gaze confidence.
pub(crate) const MAX_SOURCE_AGE: Duration = Duration::from_millis(500);

pub(crate) trait Pointer: Send {
    fn position(&mut self, point: [i32; 2]) -> io::Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Source {
    pub eye: usize,
    pub authority: u64,
    pub sign_epoch: u64,
    pub timestamp_ns: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct Sample {
    pub source: Source,
    /// Age of this solve's exact RAW source arrival, not its publication time.
    pub age: Duration,
    pub target: (f64, f64),
}

pub(crate) struct Controller<D = UinputPointer> {
    device: Option<D>,
    generation: u64,
    enabled_at: Option<Instant>,
    last_source: Option<Source>,
    last_target: Option<(f64, f64)>,
    emitted: u64,
    pause: &'static str,
    error: Option<String>,
}

impl<D> Default for Controller<D> {
    fn default() -> Self {
        Self {
            device: None,
            generation: 0,
            enabled_at: None,
            last_source: None,
            last_target: None,
            emitted: 0,
            pause: "off",
            error: None,
        }
    }
}

impl<D: Pointer> Controller<D> {
    pub(crate) fn enabled_generation(&self) -> Option<u64> {
        self.device.as_ref().map(|_| self.generation)
    }

    pub(crate) fn invalidate_global_settings(&mut self) {
        self.last_target = None;
        if self.device.is_some() {
            self.pause = "paused: global gaze settings changed";
        }
        // Preserve last_source: a new display setting or repeated RAW source
        // is not a fresh observation, nor a reason to reopen the uinput device.
    }

    fn disable(&mut self) {
        self.device = None;
        self.generation = self.generation.wrapping_add(1);
        self.enabled_at = None;
        self.last_source = None;
        self.last_target = None;
        self.pause = "off";
    }

    fn set_enabled_with(
        &mut self,
        enabled: bool,
        now: Instant,
        open: impl FnOnce() -> io::Result<D>,
    ) -> Result<(), String> {
        if !enabled {
            self.disable();
            self.error = None;
            return Ok(());
        }
        if self.device.is_some() {
            return Ok(());
        }
        match open() {
            Ok(device) => {
                self.device = Some(device);
                self.generation = self.generation.wrapping_add(1);
                self.enabled_at = Some(now);
                self.last_source = None;
                self.last_target = None;
                self.error = None;
                self.pause = "waiting for fresh signed gaze";
                Ok(())
            }
            Err(error) => {
                self.disable();
                let message = enable_error(&error);
                self.error = Some(message.clone());
                Err(message)
            }
        }
    }

    pub(crate) fn snapshot(&self) -> Value {
        json!({
            "enabled": self.device.is_some(), "device": "/dev/uinput",
            "device_name": DEVICE_NAME, "input_identifier": DEVICE_IDENTIFIER,
            "mode": "absolute", "clicks": false,
            "status": if self.error.is_some() { "error" } else { self.pause },
            "error": self.error, "emitted_positions": self.emitted,
            "last_unclamped_target": self.last_target,
            "last_source_timestamp_ns": self.last_source.map(|s| s.timestamp_ns.to_string()),
            "maximum_source_arrival_age_ms": MAX_SOURCE_AGE.as_millis(),
        })
    }

    pub(crate) fn label(&self) -> String {
        if let Some(error) = &self.error {
            format!("MOUSE OFF: {error}")
        } else if self.device.is_some() {
            format!("MOUSE ON: {}", self.pause)
        } else {
            "MOUSE OFF / Super+Shift+M".into()
        }
    }

    /// Caller snapshots the generation before projecting. A concurrent OFF (or
    /// OFF/ON) invalidates that work, even when a camera-control lease is held.
    pub(crate) fn update(
        &mut self,
        generation: u64,
        now: Instant,
        sample: Result<Sample, &'static str>,
    ) -> Option<String> {
        if self.device.is_none() || self.generation != generation {
            return None;
        }
        let sample = match sample {
            Ok(sample) => sample,
            Err(reason) => {
                self.pause = reason;
                return None;
            }
        };
        if sample.age > MAX_SOURCE_AGE {
            self.pause = "paused: stale gaze source";
            return None;
        }
        if now
            .checked_sub(sample.age)
            .zip(self.enabled_at)
            .is_none_or(|(arrived, enabled)| arrived < enabled)
        {
            self.pause = "waiting for gaze captured after enabling";
            return None;
        }
        if self.last_source.is_some_and(|last| {
            last.eye == sample.source.eye
                && last.authority == sample.source.authority
                && sample.source.timestamp_ns <= last.timestamp_ns
        }) {
            // Do not fight a physical mouse by re-emitting a held prediction.
            return None;
        }
        let Some(point) = absolute_position(sample.target) else {
            self.pause = "paused: non-finite monitor target";
            return None;
        };
        if let Err(error) = self.device.as_mut().expect("enabled").position(point) {
            self.disable();
            let message = format!("Mouse movement OFF: /dev/uinput write failed: {error}");
            self.error = Some(message.clone());
            return Some(message);
        }
        self.last_source = Some(sample.source);
        self.last_target = Some(sample.target);
        self.emitted += 1;
        self.pause = "tracking";
        None
    }
}

impl Controller<UinputPointer> {
    pub(crate) fn command(&mut self, action: &str) -> Value {
        let enabled = match action.to_ascii_uppercase().as_str() {
            "STATUS" => return json!({"ok": true, "mouse_output": self.snapshot()}),
            "ON" => true,
            "OFF" => false,
            "TOGGLE" => self.device.is_none(),
            _ => return json!({"ok": false, "error": "MOUSE OUTPUT ON|OFF|TOGGLE|STATUS"}),
        };
        let result = self.set_enabled_with(enabled, Instant::now(), UinputPointer::create);
        if let Err(error) = &result {
            eprintln!("{error}");
        }
        json!({"ok": result.is_ok(), "error": result.err(), "mouse_output": self.snapshot()})
    }
}

fn enable_error(error: &io::Error) -> String {
    if error.kind() == io::ErrorKind::PermissionDenied {
        "Mouse movement remains OFF: no write permission to /dev/uinput. Grant your user write access before enabling; Buttercup will not change permissions or use sudo.".into()
    } else {
        format!("Mouse movement remains OFF: cannot create /dev/uinput pointer: {error}")
    }
}

fn absolute_position(target: (f64, f64)) -> Option<[i32; 2]> {
    (target.0.is_finite() && target.1.is_finite()).then(|| {
        [
            (target.0.clamp(0.0, 1.0) * AXIS_MAX as f64).round() as i32,
            (target.1.clamp(0.0, 1.0) * AXIS_MAX as f64).round() as i32,
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Fake {
        points: Arc<Mutex<Vec<[i32; 2]>>>,
        fail: bool,
    }
    impl Pointer for Fake {
        fn position(&mut self, point: [i32; 2]) -> io::Result<()> {
            if self.fail {
                return Err(io::Error::from(io::ErrorKind::PermissionDenied));
            }
            self.points.lock().unwrap().push(point);
            Ok(())
        }
    }
    fn sample(timestamp: u64, age: Duration, target: (f64, f64)) -> Result<Sample, &'static str> {
        Ok(Sample {
            source: Source {
                eye: 0,
                authority: 1,
                sign_epoch: 1,
                timestamp_ns: timestamp,
            },
            age,
            target,
        })
    }

    #[test]
    fn defaults_off_and_off_never_opens_device() {
        let mut c = Controller::<Fake>::default();
        assert_eq!(c.snapshot()["enabled"], false);
        c.set_enabled_with(false, Instant::now(), || panic!("must not open"))
            .unwrap();
    }

    #[test]
    fn permission_denial_warns_and_leaves_output_off() {
        let mut c = Controller::<Fake>::default();
        let e = c
            .set_enabled_with(true, Instant::now(), || {
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            })
            .unwrap_err();
        assert!(e.contains("no write permission to /dev/uinput"));
        assert_eq!(c.snapshot()["enabled"], false);
        assert_eq!(c.snapshot()["error"], e);
        assert!(c.label().starts_with("MOUSE OFF"));
    }

    #[test]
    fn missing_device_warns_and_can_be_retried() {
        let mut c = Controller::<Fake>::default();
        let now = Instant::now();
        assert!(c
            .set_enabled_with(true, now, || Err(io::Error::from(io::ErrorKind::NotFound)))
            .is_err());
        c.set_enabled_with(true, now, || Ok(Fake::default()))
            .unwrap();
        assert_eq!(c.snapshot()["enabled"], true);
        assert!(c.snapshot()["error"].is_null());
        c.set_enabled_with(true, now, || panic!("ON is idempotent"))
            .unwrap();
    }

    #[test]
    fn absolute_plotter_jumps_without_easing_and_never_replays_old_sources() {
        let mut c = Controller::<Fake>::default();
        let now = Instant::now();
        let fake = Fake::default();
        let points = fake.points.clone();
        c.set_enabled_with(true, now, || Ok(fake)).unwrap();
        let generation = c.enabled_generation().unwrap();
        c.update(generation, now, sample(10, Duration::ZERO, (0.0, 0.0)));
        c.update(generation, now, sample(11, Duration::ZERO, (1.0, 1.0)));
        c.update(generation, now, sample(11, Duration::ZERO, (0.5, 0.5)));
        c.update(generation, now, sample(9, Duration::ZERO, (0.5, 0.5)));
        assert_eq!(*points.lock().unwrap(), [[0, 0], [AXIS_MAX, AXIS_MAX]]);
    }

    #[test]
    fn missing_stale_pre_enable_and_invalid_samples_do_not_move() {
        let mut c = Controller::<Fake>::default();
        let now = Instant::now();
        c.set_enabled_with(true, now, || Ok(Fake::default()))
            .unwrap();
        let generation = c.enabled_generation().unwrap();
        c.update(generation, now, Err("paused: no signed gaze"));
        c.update(
            generation,
            now,
            sample(1, Duration::from_millis(1), (0.5, 0.5)),
        );
        c.update(
            generation,
            now + Duration::from_secs(2),
            sample(2, Duration::from_secs(1), (0.5, 0.5)),
        );
        c.update(generation, now, sample(3, Duration::ZERO, (f64::NAN, 0.5)));
        assert_eq!(c.emitted, 0);
        c.update(generation, now, sample(4, Duration::ZERO, (0.5, 0.5)));
        assert_eq!(c.emitted, 1);
    }

    #[test]
    fn global_gaze_switch_pause_keeps_device_on_and_resumes_absolute_output() {
        let now = Instant::now();
        let mut c = Controller::<Fake>::default();
        let fake = Fake::default();
        let points = fake.points.clone();
        c.set_enabled_with(true, now, || Ok(fake)).unwrap();
        let generation = c.enabled_generation().unwrap();
        c.update(generation, now, sample(10, Duration::ZERO, (0.0, 0.0)));
        c.invalidate_global_settings();
        assert_eq!(c.enabled_generation(), Some(generation));
        assert!(c.snapshot()["last_unclamped_target"].is_null());
        assert_eq!(*points.lock().unwrap(), [[0, 0]]);
        let mut next = sample(11, Duration::ZERO, (1.0, 1.0)).unwrap();
        next.source.authority += 1;
        c.update(generation, now, Ok(next));
        // The new detector's first admissible point is not eased toward the
        // previous detector's point, and no re-enable/uinput reopen is needed.
        assert_eq!(*points.lock().unwrap(), [[0, 0], [AXIS_MAX, AXIS_MAX]]);
        c.update(generation, now, Ok(next));
        assert_eq!(points.lock().unwrap().len(), 2);
    }

    #[test]
    fn off_and_off_on_invalidate_in_flight_work() {
        let mut c = Controller::<Fake>::default();
        let now = Instant::now();
        c.set_enabled_with(true, now, || Ok(Fake::default()))
            .unwrap();
        let generation = c.enabled_generation().unwrap();
        c.set_enabled_with(false, now, || panic!("off")).unwrap();
        c.update(generation, now, sample(1, Duration::ZERO, (0.5, 0.5)));
        c.set_enabled_with(true, now, || Ok(Fake::default()))
            .unwrap();
        c.update(generation, now, sample(2, Duration::ZERO, (0.5, 0.5)));
        assert_eq!(c.emitted, 0);
    }

    #[test]
    fn write_error_disables_device_and_is_visible() {
        let mut c = Controller::<Fake>::default();
        let now = Instant::now();
        c.set_enabled_with(true, now, || {
            Ok(Fake {
                fail: true,
                ..Fake::default()
            })
        })
        .unwrap();
        let error = c
            .update(c.generation, now, sample(1, Duration::ZERO, (0.5, 0.5)))
            .unwrap();
        assert!(error.contains("write failed"));
        assert_eq!(c.snapshot()["enabled"], false);
    }

    #[test]
    fn clamping_is_at_device_boundary_not_in_prediction() {
        assert_eq!(absolute_position((-10.0, 10.0)), Some([0, AXIS_MAX]));
        assert_eq!(absolute_position((0.5, 0.5)), Some([32768, 32768]));
        assert_eq!(absolute_position((f64::INFINITY, 0.0)), None);
    }
}
