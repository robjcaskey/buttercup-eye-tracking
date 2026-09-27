//! On-screen gaze cursor: a small click-through ring that Sway places at the
//! calibrated screen gaze. It never moves the pointer or takes focus. Only the
//! accepted (sign-resolved, calibrated) desktop gaze sample drives it.
use crate::{mouse_output::Sample, App};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use winit::{dpi::LogicalSize, event::WindowEvent, event_loop::ActiveEventLoop,
    platform::wayland::WindowAttributesExtWayland, window::{Window, WindowId}};

const APP_ID: &str = "buttercup-gaze-cursor";
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
    status: String,
}

impl Controller {
    pub fn enabled_generation(&self) -> Option<u64> { self.enabled.then_some(self.generation) }
    pub fn command(&mut self, action: &str) -> Value {
        let enabled = match action.to_ascii_uppercase().as_str() {
            "STATUS" => self.enabled,
            "ON" => true, "OFF" => false, "TOGGLE" => !self.enabled,
            _ => return json!({"ok":false,"error":"GAZE CURSOR ON|OFF|TOGGLE|STATUS"}),
        };
        if enabled != self.enabled {
            self.enabled = enabled;
            self.generation = self.generation.wrapping_add(1);
            self.target = None;
            self.status = if enabled {"waiting for calibrated gaze"} else {"off"}.into();
        }
        json!({"ok":true,"gaze_cursor":{"enabled":self.enabled,"status":self.status,"target":self.target}})
    }
    pub fn publish(&mut self, generation: u64, input: Result<Sample, &'static str>) {
        if self.enabled_generation() != Some(generation) { return; }
        match input.and_then(|s| {
            if s.age > std::time::Duration::from_nanos(crate::SAM31_RESULT_MAX_AGE_NS) { Err("waiting for fresh gaze") }
            else if !s.target.0.is_finite() || !s.target.1.is_finite() { Err("invalid gaze") }
            else { Ok(s.target) }
        }) {
            Ok(target) => { self.target = Some(target); self.status = "tracking".into(); }
            Err(reason) => { self.target = None; self.status = reason.into(); }
        }
    }
}

/// Owned by the viewer event loop beside the wleyes overlay.
pub(crate) struct CursorWindow {
    window: crate::wleyes::ArgbWindow,
    drawn: bool,
    placed: Option<(i32, i32)>,
    hidden: bool,
    output: Option<[f64; 4]>,
    sway: Option<UnixStream>,
}

pub(crate) fn sync_window(app: &mut App, event_loop: &ActiveEventLoop) {
    let (enabled, target) = app.shared.lock()
        .map(|s| (s.gaze_cursor.enabled, s.gaze_cursor.target)).unwrap_or((false, None));
    if !enabled { app.gaze_cursor_window = None; return; }
    if app.gaze_cursor_window.is_none() {
        // Never take keyboard focus: register the rule before the window maps,
        // and remember what was focused so it can be restored if needed.
        let mut sway = sway_connect();
        sway_run(&mut sway, &format!("no_focus [app_id=\"^{APP_ID}$\"]"));
        let previous_focus = sway_focused_container(&mut sway);
        let attributes = Window::default_attributes()
            .with_name(APP_ID, APP_ID)
            .with_title("Buttercup gaze cursor")
            .with_transparent(true)
            .with_decorations(false).with_resizable(false).with_active(false)
            .with_inner_size(LogicalSize::new(SIZE, SIZE));
        let created = event_loop.create_window(attributes).map_err(|e| e.to_string())
            .and_then(|window| {
                // Clicks and hover pass through to whatever is underneath.
                let _ = window.set_cursor_hittest(false);
                crate::wleyes::ArgbWindow::new(Arc::new(window))
            });
        match created {
            Ok(window) => {
                sway_run(&mut sway, &format!(
                    "[app_id=\"^{APP_ID}$\"] floating enable, border none, sticky enable"));
                if let Some(id) = previous_focus {
                    sway_run(&mut sway, &format!("[con_id={id}] focus"));
                }
                let output = sway_focused_output(&mut sway);
                app.gaze_cursor_window = Some(CursorWindow {
                    window, drawn: false, placed: None, hidden: false, output, sway,
                });
            }
            Err(error) => {
                eprintln!("gaze cursor window: {error}");
                if let Ok(mut s) = app.shared.lock() { s.gaze_cursor.command("OFF"); s.gaze_cursor.status = error; }
                return;
            }
        }
    }
    let Some(cursor) = app.gaze_cursor_window.as_mut() else { return };
    let Some([ox, oy, ow, oh]) = cursor.output else { return };
    match target {
        Some((x, y)) => {
            let place = (
                (ox + x.clamp(0.0, 1.0) * ow - SIZE / 2.0).round() as i32,
                (oy + y.clamp(0.0, 1.0) * oh - SIZE / 2.0).round() as i32,
            );
            let moved = cursor.placed.is_none_or(|(px, py)|
                f64::from((px - place.0).abs().max((py - place.1).abs())) >= MOVE_THRESHOLD);
            if moved || cursor.hidden {
                sway_run(&mut cursor.sway, &format!(
                    "[app_id=\"^{APP_ID}$\"] move absolute position {} {}", place.0, place.1));
                cursor.placed = Some(place);
                if cursor.hidden { cursor.hidden = false; cursor.drawn = false; }
            }
        }
        None if !cursor.hidden => { cursor.hidden = true; cursor.drawn = false; }
        None => {}
    }
    if !cursor.drawn { cursor.window.request_redraw(); }
}

pub(crate) fn window_event(app: &mut App, id: WindowId, event: &WindowEvent) -> bool {
    let Some(cursor) = app.gaze_cursor_window.as_mut().filter(|c| c.window.window.id() == id) else { return false };
    if let WindowEvent::RedrawRequested = event {
        let size = cursor.window.window.inner_size();
        if size.width == 0 || size.height == 0 { return true; }
        let mut pixels = vec![0u32; size.width as usize * size.height as usize];
        if !cursor.hidden { render_ring(&mut pixels, size.width as usize, size.height as usize); }
        match cursor.window.present(&pixels, size.width, size.height) {
            Ok(()) => cursor.drawn = true,
            Err(error) => eprintln!("gaze cursor present: {error}"),
        }
    }
    // Overlay input must never reach viewer shortcuts.
    true
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

fn sway_run(stream: &mut Option<UnixStream>, command: &str) {
    if stream.is_none() { *stream = sway_connect(); }
    if sway_request(stream, 0, command).is_none() {
        eprintln!("gaze cursor: sway command failed: {command}");
    }
}

fn sway_focused_container(stream: &mut Option<UnixStream>) -> Option<i64> {
    fn find(node: &Value) -> Option<i64> {
        if node["focused"] == true { return node["id"].as_i64(); }
        node["nodes"].as_array().into_iter().chain(node["floating_nodes"].as_array())
            .flatten().find_map(find)
    }
    if stream.is_none() { *stream = sway_connect(); }
    let body = sway_request(stream, 4, "")?;
    find(&serde_json::from_slice(&body).ok()?)
}

/// Logical rectangle [x, y, width, height] of the focused output.
fn sway_focused_output(stream: &mut Option<UnixStream>) -> Option<[f64; 4]> {
    let body = sway_request(stream, 3, "")?;
    let outputs: Value = serde_json::from_slice(&body).ok()?;
    let output = outputs.as_array()?.iter()
        .find(|o| o["focused"] == true && o["active"] == true)
        .or_else(|| outputs.as_array()?.iter().find(|o| o["active"] == true))?;
    let rect = &output["rect"];
    Some([rect["x"].as_f64()?, rect["y"].as_f64()?, rect["width"].as_f64()?, rect["height"].as_f64()?])
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
    #[test]
    fn controller_ignores_stale_generations_and_clears_on_error() {
        let mut c = Controller::default();
        c.command("ON");
        let generation = c.enabled_generation().unwrap();
        c.publish(generation.wrapping_add(1), Err("x"));
        assert_eq!(c.status, "waiting for calibrated gaze");
        c.publish(generation, Err("paused"));
        assert!(c.target.is_none());
        c.command("OFF");
        assert!(c.enabled_generation().is_none());
    }
}
