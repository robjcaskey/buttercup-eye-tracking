//! Explicitly enabled window focus from gaze, independent of pointer movement.
//! The worker only focuses existing, visible windows on the current workspace.
//! No uinput, pointer coordinates, clicks, workspace switches or window moves.
use crate::mouse_output::{Sample, Source, MAX_SOURCE_AGE};
use serde_json::{json, Value};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

mod sway;
#[cfg(test)]
mod tests;

const DWELL: Duration = Duration::from_millis(350);
const MIN_SAMPLES: u32 = 3;
const MAX_GAP: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

impl Rect {
    fn contains(self, p: (f64, f64), margin: f64) -> bool {
        p.0 >= self.x + margin
            && p.1 >= self.y + margin
            && p.0 < self.x + self.w - margin
            && p.1 < self.y + self.h - margin
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Target {
    id: u64,
    rect: Rect,
}

#[derive(Clone, Copy, Debug)]
struct Hit {
    workspace: u64,
    focused: u64,
    target: Option<Target>,
}

trait Backend: Send + 'static {
    fn hit(&mut self, point: (f64, f64)) -> Result<Hit, String>;
    fn focus(&mut self, target: Target) -> Result<(), String>;
}

#[derive(Default)]
struct Dwell {
    context: Option<(u64, u64)>,
    basis: Option<(usize, u64, u64)>,
    pending: Option<(Target, Instant, Instant, u32, u64)>,
}

impl Dwell {
    fn observe(&mut self, now: Instant, source: Source, hit: Hit) -> Option<Target> {
        let basis = (source.eye, source.authority, source.sign_epoch);
        let context = (hit.workspace, hit.focused);
        if self.basis != Some(basis) || self.context != Some(context) {
            self.pending = None;
        }
        self.basis = Some(basis);
        self.context = Some(context);
        let Some(target) = hit.target.filter(|t| t.id != hit.focused) else {
            self.pending = None;
            return None;
        };
        let (first, count, first_source_ns) = match self.pending {
            Some((previous, first, last, count, first_source_ns))
                if previous == target && now.saturating_duration_since(last) <= MAX_GAP =>
            {
                (first, count + 1, first_source_ns)
            }
            _ => (now, 1, source.timestamp_ns),
        };
        self.pending = Some((target, first, now, count, first_source_ns));
        (count >= MIN_SAMPLES
            && now.saturating_duration_since(first) >= DWELL
            && u128::from(source.timestamp_ns.saturating_sub(first_source_ns)) >= DWELL.as_nanos())
        .then_some(target)
    }
}

#[derive(Clone, Copy)]
struct Input {
    at: Instant,
    sample: Result<Sample, &'static str>,
}

impl Input {
    fn fresh(self, now: Instant, enabled_at: Instant) -> Result<Sample, &'static str> {
        let sample = self.sample?;
        let age = sample.age + now.saturating_duration_since(self.at);
        if age > MAX_SOURCE_AGE {
            return Err("paused: stale gaze source");
        }
        if self
            .at
            .checked_sub(sample.age)
            .is_none_or(|at| at < enabled_at)
        {
            return Err("waiting for gaze captured after enabling");
        }
        if ![sample.target.0, sample.target.1]
            .into_iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(&v))
        {
            return Err("paused: gaze outside monitor");
        }
        Ok(sample)
    }
}

#[derive(Default)]
struct State {
    enabled_at: Option<Instant>,
    generation: u64,
    revision: u64,
    /// Global input-policy changes do not turn the output off, but do discard
    /// dwell accumulated with a different analysis configuration.
    policy_revision: u64,
    input: Option<Input>,
    status: String,
    error: Option<String>,
    output: Option<String>,
    pending: Option<u64>,
    focused: Option<u64>,
    changes: u64,
}

#[derive(Default)]
struct Channel {
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Default)]
pub(crate) struct Controller {
    channel: Arc<Channel>,
    worker: Option<JoinHandle<()>>,
}

impl Controller {
    pub(crate) fn enabled_generation(&self) -> Option<u64> {
        let s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
        s.enabled_at.map(|_| s.generation)
    }

    pub(crate) fn snapshot(&self) -> Value {
        let s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
        json!({"enabled":s.enabled_at.is_some(), "mode":"window-focus-only", "moves_pointer":false,
            "status":if s.enabled_at.is_none() && s.error.is_none() {"off"} else {&s.status},
            "error":s.error,"output":s.output,"dwell_ms":DWELL.as_millis(),"minimum_fresh_samples":MIN_SAMPLES,
            "pending_window":s.pending,"last_focused_window":s.focused,"focus_changes":s.changes})
    }

    pub(crate) fn label(&self) -> String {
        let s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(error) = &s.error {
            format!("GAZE FOCUS OFF: {error}")
        } else if s.enabled_at.is_some() {
            format!("GAZE FOCUS ON: {}", s.status)
        } else {
            "GAZE FOCUS OFF / Super+Shift+F".into()
        }
    }

    pub(crate) fn publish(
        &self,
        generation: u64,
        now: Instant,
        sample: Result<Sample, &'static str>,
    ) {
        let mut s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.enabled_at.is_none() || s.generation != generation {
            return;
        }
        s.input = Some(Input { at: now, sample });
        s.revision = s.revision.wrapping_add(1);
        self.channel.wake.notify_one();
    }

    pub(crate) fn invalidate_global_settings(&self) {
        let mut s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
        s.policy_revision = s.policy_revision.wrapping_add(1);
        s.revision = s.revision.wrapping_add(1);
        s.input = None;
        s.pending = None;
        if s.enabled_at.is_some() {
            s.status = "paused: global gaze settings changed".into();
        }
        self.channel.wake.notify_one();
    }

    pub(crate) fn disable(&mut self) {
        {
            let mut s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
            s.enabled_at = None;
            s.generation = s.generation.wrapping_add(1);
            s.input = None;
            s.pending = None;
            s.error = None;
            s.status = "off".into();
            self.channel.wake.notify_one();
        }
        // The worker only touches its own channel, never SharedState. Joining
        // invalidates old work before OFF returns; IPC has bounded timeouts.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }

    fn start(&mut self, backend: impl Backend, output: String) -> Result<(), String> {
        self.disable();
        let generation = {
            let mut s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
            s.enabled_at = Some(Instant::now());
            s.output = Some(output);
            s.status = "waiting for fresh signed gaze".into();
            s.generation
        };
        let channel = Arc::clone(&self.channel);
        match std::thread::Builder::new()
            .name("gaze-focus".into())
            .spawn(move || run_worker(channel, generation, backend))
        {
            Ok(worker) => {
                self.worker = Some(worker);
                Ok(())
            }
            Err(e) => {
                self.disable();
                Err(format!("cannot start focus worker: {e}"))
            }
        }
    }

    pub(crate) fn command(&mut self, action: &str) -> Value {
        let enable = match action.to_ascii_uppercase().as_str() {
            "STATUS" => return json!({"ok":true,"gaze_focus":self.snapshot()}),
            "ON" => true,
            "OFF" => false,
            "TOGGLE" => self.enabled_generation().is_none(),
            _ => return json!({"ok":false,"error":"GAZE FOCUS ON|OFF|TOGGLE|STATUS"}),
        };
        let result = if !enable {
            self.disable();
            Ok(())
        } else if self.enabled_generation().is_some() {
            Ok(())
        } else {
            sway::Sway::connect().and_then(|backend| {
                let output = backend.output.clone();
                self.start(backend, output)
            })
        };
        if let Err(error) = &result {
            self.disable();
            let mut s = self.channel.state.lock().unwrap_or_else(|e| e.into_inner());
            s.error = Some(error.clone());
            s.status = "error".into();
        }
        json!({"ok":result.is_ok(),"error":result.err(),"gaze_focus":self.snapshot()})
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.disable();
    }
}

fn source_advanced(last: Option<Source>, current: Source) -> bool {
    last.is_none_or(|p| {
        (p.eye, p.authority, p.sign_epoch) != (current.eye, current.authority, current.sign_epoch)
            || current.timestamp_ns > p.timestamp_ns
    })
}

fn run_worker(channel: Arc<Channel>, generation: u64, mut backend: impl Backend) {
    let mut last_revision = 0;
    let mut last_policy_revision = None;
    let mut last_source = None;
    let mut dwell = Dwell::default();
    loop {
        let mut s = channel.state.lock().unwrap_or_else(|e| e.into_inner());
        while s.enabled_at.is_some() && s.generation == generation && s.revision == last_revision {
            s = channel
                .wake
                .wait_timeout(s, Duration::from_millis(100))
                .unwrap_or_else(|e| e.into_inner())
                .0;
            if s.input
                .is_some_and(|input| input.at.elapsed() > MAX_SOURCE_AGE)
            {
                dwell = Dwell::default();
                s.pending = None;
                s.status = "paused: viewer updates stopped".into();
            }
        }
        let Some(enabled_at) = s.enabled_at.filter(|_| s.generation == generation) else {
            return;
        };
        let policy_revision = s.policy_revision;
        if last_policy_revision != Some(policy_revision) {
            dwell = Dwell::default();
            last_policy_revision = Some(policy_revision);
        }
        last_revision = s.revision;
        let Some(input) = s.input else {
            continue;
        };
        let sample = match input.fresh(Instant::now(), enabled_at) {
            Ok(sample) => sample,
            Err(reason) => {
                dwell = Dwell::default();
                s.pending = None;
                s.status = reason.into();
                continue;
            }
        };
        if !source_advanced(last_source, sample.source) {
            continue;
        }
        last_source = Some(sample.source);
        drop(s);
        let hit = backend.hit(sample.target);
        let mut s = channel.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.enabled_at.is_none() || s.generation != generation {
            return;
        }
        if s.policy_revision != policy_revision {
            continue;
        }
        // A newer observation/missing-eye update arriving during IPC cancels
        // this result; never send a queued decision for the preceding gaze.
        if !s
            .input
            .and_then(|v| v.fresh(Instant::now(), enabled_at).ok())
            .is_some_and(|current| current.source == sample.source)
        {
            continue;
        }
        let result = match hit {
            Ok(hit) => {
                let decision = dwell.observe(Instant::now(), sample.source, hit);
                s.pending = dwell.pending.map(|p| p.0.id);
                s.status = if s.pending.is_some() {
                    "dwelling"
                } else {
                    "tracking"
                }
                .into();
                if let Some(target) = decision {
                    // OFF serializes with this final bounded command, after
                    // all source/generation gates. No cursor command exists.
                    backend.focus(target).map(|()| {
                        s.focused = Some(target.id);
                        s.changes += 1;
                        s.pending = None;
                        dwell = Dwell::default();
                    })
                } else {
                    Ok(())
                }
            }
            Err(error) => Err(error),
        };
        if let Err(error) = result {
            s.enabled_at = None;
            s.pending = None;
            s.error = Some(error.clone());
            s.status = "error".into();
            drop(s);
            eprintln!("GAZE FOCUS OFF: {error}");
            std::thread::spawn(move || {
                let _ = std::process::Command::new("notify-send")
                    .args([
                        "--app-name=Buttercup",
                        "--urgency=critical",
                        "--expire-time=10000",
                        "Buttercup gaze focus OFF",
                        &error,
                    ])
                    .status();
            });
            return;
        }
    }
}
