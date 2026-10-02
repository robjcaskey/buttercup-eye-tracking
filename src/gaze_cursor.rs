//! On-screen gaze cursor: a small click-through ring on the layer-shell
//! overlay (visible above fullscreen windows) at the calibrated screen gaze. It never moves the pointer or takes focus. Only the
//! accepted (sign-resolved, calibrated) desktop gaze sample drives it.
use crate::{mouse_output::Sample, App};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

mod overlay;

/// Logical window size; the ring is drawn once at this size. At least Sway's
/// default 75x50 floating minimum, so the ring stays centered in its window.
const SIZE: f64 = 76.0;
/// Re-place the window only when the target moves at least this far (logical px).
const MOVE_THRESHOLD: f64 = 3.0;

#[derive(Default)]
pub(crate) struct Controller {
    enabled: bool,
    generation: u64,
    target: Option<(f64, f64)>,
    /// When the shown target's source was captured. A brief tracking gap
    /// keeps showing (and clicking at) it while it is still fresh.
    target_captured: Option<std::time::Instant>,
    status: String,
}

impl Controller {
    pub fn enabled_generation(&self) -> Option<u64> { self.enabled.then_some(self.generation) }
    /// Gaze is tracked even with the ring hidden, so look-and-click can aim
    /// at any time; the ring only displays this target.
    pub fn tracking_generation(&self) -> u64 { self.generation }
    /// Latest fresh calibrated target (the ring shows it when enabled).
    pub fn target(&self) -> Option<(f64, f64)> { self.target }
    pub fn command(&mut self, action: &str) -> Value {
        let enabled = match action.to_ascii_uppercase().as_str() {
            "STATUS" => self.enabled,
            "ON" => true, "OFF" => false, "TOGGLE" => !self.enabled,
            _ => return json!({"ok":false,"error":"GAZE CURSOR ON|OFF|TOGGLE|STATUS"}),
        };
        if enabled != self.enabled {
            self.enabled = enabled;
            self.generation = self.generation.wrapping_add(1);
            if self.target.is_none() { self.status = "waiting for calibrated gaze".into(); }
        }
        json!({"ok":true,"gaze_cursor":{"enabled":self.enabled,"status":self.status,"target":self.target}})
    }
    pub fn publish(&mut self, generation: u64, input: Result<Sample, &'static str>) {
        if self.tracking_generation() != generation { return; }
        let now = std::time::Instant::now();
        let age = input.as_ref().map(|s| s.age).unwrap_or_default();
        match input.and_then(|s| {
            if s.age > std::time::Duration::from_nanos(crate::SAM31_RESULT_MAX_AGE_NS) { Err("waiting for fresh gaze") }
            else if !s.target.0.is_finite() || !s.target.1.is_finite() { Err("invalid gaze") }
            else { Ok(s.target) }
        }) {
            Ok(target) => {
                self.target = Some(target);
                self.target_captured = Some(now - age);
                self.status = "tracking".into();
            }
            Err(reason) => {
                // Same freshness bound as a live sample, measured from capture.
                let fresh = self.target_captured.is_some_and(|captured|
                    now.duration_since(captured) <= std::time::Duration::from_nanos(crate::SAM31_RESULT_MAX_AGE_NS));
                if !fresh { self.target = None; self.target_captured = None; }
                self.status = if fresh { format!("holding last gaze ({reason})") } else { reason.into() };
            }
        }
    }
}

/// Owned by the viewer event loop beside the wleyes overlay.
pub(crate) struct CursorWindow {
    overlay: overlay::CursorOverlay,
    sent: Option<Option<(f64, f64)>>,
}

pub(crate) fn sync_window(app: &mut App) {
    let (enabled, target) = app.shared.lock()
        .map(|s| (s.gaze_cursor.enabled, s.gaze_cursor.target)).unwrap_or((false, None));
    if !enabled { app.gaze_cursor_window = None; return; }
    let cursor = app.gaze_cursor_window.get_or_insert_with(|| {
        // Normalized targets refer to the focused Sway output.
        let output = sway_focused_output_name(&mut sway_connect());
        CursorWindow { overlay: overlay::CursorOverlay::spawn(output), sent: None }
    });
    if let Some(error) = cursor.overlay.error() {
        app.gaze_cursor_window = None;
        if let Ok(mut s) = app.shared.lock() { s.gaze_cursor.command("OFF"); s.gaze_cursor.status = error; }
        return;
    }
    if cursor.sent != Some(target) {
        cursor.overlay.show(target);
        cursor.sent = Some(target);
    }
}

/// Premultiplied ARGB ring: dark outline around a bright ring, clear center.
pub(crate) fn render_ring(pixels: &mut [u32], w: usize, h: usize) {
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let scale = w.min(h) as f64 / SIZE;
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy)) / scale;
            pixels[y * w + x] = if (17.0..=21.0).contains(&d) {
                0xf0ff_d23c // warm yellow ring
            } else if (15.5..17.0).contains(&d) || (21.0..=22.5).contains(&d) {
                0xc000_0000 // dark outline for contrast on light pages
            } else if d <= 2.0 {
                0xf0ff_d23c // center dot
            } else {
                0
            };
        }
    }
}

fn sway_connect() -> Option<UnixStream> {
    let path = std::env::var_os("SWAYSOCK")?;
    UnixStream::connect(path).ok()
}

/// i3/Sway IPC: "i3-ipc" magic, u32 length, u32 type, payload.
fn sway_request(stream: &mut Option<UnixStream>, kind: u32, payload: &str) -> Option<Vec<u8>> {
    let socket = stream.as_mut()?;
    let mut message = b"i3-ipc".to_vec();
    message.extend_from_slice(&(payload.len() as u32).to_ne_bytes());
    message.extend_from_slice(&kind.to_ne_bytes());
    message.extend_from_slice(payload.as_bytes());
    let reply = (|| {
        socket.write_all(&message).ok()?;
        let mut header = [0u8; 14];
        socket.read_exact(&mut header).ok()?;
        let length = u32::from_ne_bytes(header[6..10].try_into().ok()?) as usize;
        let mut body = vec![0u8; length];
        socket.read_exact(&mut body).ok()?;
        Some(body)
    })();
    if reply.is_none() { *stream = None; }
    reply
}

/// Name of the focused (else first active) Sway output.
fn sway_focused_output_name(stream: &mut Option<UnixStream>) -> Option<String> {
    let body = sway_request(stream, 3, "")?;
    let outputs: Value = serde_json::from_slice(&body).ok()?;
    let output = outputs.as_array()?.iter()
        .find(|o| o["focused"] == true && o["active"] == true)
        .or_else(|| outputs.as_array()?.iter().find(|o| o["active"] == true))?;
    output["name"].as_str().map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ring_is_transparent_in_the_center_gap_and_outside() {
        let (w, h) = (56, 56);
        let mut pixels = vec![1u32; w * h];
        render_ring(&mut pixels, w, h);
        let scale = 56.0 / SIZE;
        let at = |r: f64| pixels[28 * w + 28 + (r * scale).round() as usize];
        assert_eq!(pixels[0], 0, "corners stay transparent");
        assert_ne!(pixels[28 * w + 28], 0, "center dot");
        assert_eq!(at(10.0), 0, "gap between dot and ring");
        assert_ne!(at(19.0), 0, "ring");
    }
    fn sample_at(target: (f64, f64), age: std::time::Duration) -> Result<Sample, &'static str> {
        Ok(Sample {
            source: crate::mouse_output::Source { eye: 0, authority: 1, sign_epoch: 1, timestamp_ns: 1 },
            age,
            target,
        })
    }
    #[test]
    fn controller_ignores_stale_generations_and_clears_on_error() {
        let mut c = Controller::default();
        c.command("ON");
        let generation = c.enabled_generation().unwrap();
        c.publish(generation.wrapping_add(1), Err("x"));
        assert_eq!(c.status, "waiting for calibrated gaze");
        c.publish(generation, Err("paused"));
        assert!(c.target.is_none());
        c.publish(generation, sample_at((0.25, 0.75), std::time::Duration::ZERO));
        c.publish(generation, Err("paused: unresolved gaze sign"));
        assert_eq!(c.target(), Some((0.25, 0.75)), "a brief gap holds the fresh target");
        assert!(c.status.starts_with("holding"));
        c.publish(generation, sample_at((0.5, 0.5), std::time::Duration::from_secs(2)));
        assert_eq!(c.target(), Some((0.25, 0.75)), "a stale sample never replaces the shown target");
        c.publish(generation, sample_at((0.5, 0.5), std::time::Duration::from_millis(890)));
        std::thread::sleep(std::time::Duration::from_millis(20));
        c.publish(generation, Err("paused"));
        assert!(c.target().is_none(), "the hold ends when the shown sample goes stale");
        c.command("OFF");
        assert!(c.enabled_generation().is_none());
        c.publish(c.tracking_generation(), sample_at((0.1, 0.2), std::time::Duration::ZERO));
        assert_eq!(c.target(), Some((0.1, 0.2)), "click aim keeps tracking with the ring hidden");
    }
}
