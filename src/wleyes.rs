//! A playful desktop gaze preview, owned by the existing viewer event loop.
//! No camera connections, pointer injection, focus changes or new executable.
use crate::{App, mouse_output::Sample};
use serde_json::{json, Value};
use std::sync::Arc;
mod argb;
pub(crate) use argb::ArgbWindow;
use winit::{dpi::LogicalSize, event::{WindowEvent, ElementState, MouseButton}, event_loop::ActiveEventLoop,
    platform::wayland::WindowAttributesExtWayland, window::{Window, WindowId}};

#[derive(Default)]
pub(crate) struct Controller {
    enabled: bool,
    generation: u64,
    target: Option<(f64, f64)>,
    status: String,
}

impl Controller {
    pub fn enabled_generation(&self) -> Option<u64> { self.enabled.then_some(self.generation) }
    pub fn clear(&mut self) { self.target = None; }
    pub fn command(&mut self, action: &str) -> Value {
        let enabled = match action.to_ascii_uppercase().as_str() {
            "STATUS" => self.enabled,
            "ON" => true, "OFF" => false, "TOGGLE" => !self.enabled,
            _ => return json!({"ok":false,"error":"WLEYES ON|OFF|TOGGLE|STATUS"}),
        };
        if enabled != self.enabled {
            self.enabled = enabled;
            self.generation = self.generation.wrapping_add(1);
            self.clear();
            self.status = if enabled {"waiting for gaze"} else {"off"}.into();
        }
        json!({"ok":true,"wleyes":{"enabled":self.enabled,"status":self.status,
            "target":self.target,"source":"shared desktop gaze","placement":"upper right"}})
    }
    pub fn publish(&mut self, generation: u64, input: Result<Sample, &'static str>) {
        self.publish_preview(generation,input,false);
    }
    pub fn publish_preview(&mut self, generation: u64, input: Result<Sample, &'static str>, approximate:bool) {
        if self.enabled_generation() != Some(generation) { return; }
        match input.and_then(|s| {
            if s.age > std::time::Duration::from_nanos(crate::SAM31_RESULT_MAX_AGE_NS) { Err("waiting for fresh gaze") }
            else if !s.target.0.is_finite() || !s.target.1.is_finite() { Err("invalid gaze") }
            else { Ok(s.target) }
        }) {
            Ok(target) => { self.target = Some(target); self.status = if approximate {"approximate gaze"} else {"looking"}.into(); }
            Err(reason) => { self.clear(); self.status = reason.into(); }
        }
    }
}

pub(crate) fn sync_window(app: &mut App, event_loop: &ActiveEventLoop) {
    let enabled = app.shared.lock().is_ok_and(|s| s.wleyes.enabled);
    if !enabled { app.wleyes_window = None; return; }
    if app.wleyes_window.is_none() {
        let attributes = Window::default_attributes()
            .with_name("buttercup-wleyes", "buttercup-wleyes")
            .with_title("Buttercup wleyes")
            .with_transparent(true)
            .with_decorations(false).with_resizable(false).with_active(false)
            .with_inner_size(LogicalSize::new(184.0, 112.0));
        let result = event_loop.create_window(attributes).map_err(|e| e.to_string())
            .and_then(|window| {
                ArgbWindow::new(Arc::new(window))
            });
        match result {
            Ok(window) => app.wleyes_window = Some(window),
            Err(error) => {
                eprintln!("wleyes window: {error}");
                if let Ok(mut s) = app.shared.lock() {
                    s.wleyes.command("OFF"); s.wleyes.status = error;
                }
            }
        }
    }
    if let Some(window) = &mut app.wleyes_window { window.request_redraw(); }
}

pub(crate) fn window_event(app: &mut App, id: WindowId, event: &WindowEvent) -> bool {
    if !app.wleyes_window.as_ref().is_some_and(|w| w.window.id() == id) { return false; }
    match event {
        WindowEvent::CloseRequested | WindowEvent::MouseInput {
            state: ElementState::Pressed, button: MouseButton::Right, ..
        } => {
            if let Ok(mut s) = app.shared.lock() { s.wleyes.command("OFF"); }
            app.wleyes_window = None;
        }
        WindowEvent::RedrawRequested => {
            let (target,approximate) = app.shared.lock().map(|s| (s.wleyes.target,s.wleyes.status=="approximate gaze")).unwrap_or((None,false));
            if let Some(w) = &mut app.wleyes_window {
                let size = w.window.inner_size();
                if size.width == 0 || size.height == 0 { return true; }
                let mut buffer=vec![0;size.width as usize*size.height as usize];
                render(&mut buffer, size.width as usize, size.height as usize, target);
                if approximate {caption(&mut buffer,size.width as usize,size.height as usize,"APPROXIMATE");}
                if let Err(error) = w.present(&buffer,size.width,size.height) { eprintln!("wleyes present: {error}"); }
            }
        }
        // Overlay input must never become viewer calibration/quit shortcuts.
        _ => {}
    }
    true
}

pub(crate) fn render(pixels: &mut [u32], w: usize, h: usize, target: Option<(f64, f64)>) {
    if w == 0 || h == 0 { return; }
    // Deliberately cartoony; screen-normalized gaze drives both pupils equally.
    let gaze = target.map(|(x,y)| ((x.clamp(0.,1.)-0.5)*2., (y.clamp(0.,1.)-0.5)*2.));
    for y in 0..h {
        for x in 0..w {
            let px = x as f64 / w as f64;
            let py = y as f64 / h as f64;
            let mut color = 0;
            for cx in [0.29, 0.71] {
                // Tall ovals, about 1.4 times as high as wide at the default size.
                let ex = (px-cx)/0.17; let ey = (py-0.45)/0.39;
                let disk = ex*ex+ey*ey;
                if disk <= 1. { color = 0x0007090e; }
                if disk < 0.85 {
                    color = 0x00f6f4ec;
                    if let Some((gx,gy)) = gaze {
                        let dx=(px-cx-gx*0.07)/0.067;
                        let dy=(py-0.45-gy*0.16)/0.11;
                        if dx*dx+dy*dy<1. { color=0x001d6670; }
                        if dx*dx+dy*dy<0.48 { color=0x0007090e; }
                        if (dx+0.32).powi(2)+(dy+0.32).powi(2)<0.065 {color=0x00ffffff;}
                    } else {
                        // A sleepy neutral eye makes missing tracking unambiguous.
                        if ey < -0.08 {color=0x00455468;}
                        if ex*ex/0.10+(ey-0.25).powi(2)/0.08<1. {color=0x00313b4b;}
                    }
                }
            }
            pixels[y*w+x]=if color==0 {0} else {color|0xff000000};
        }
    }
    if target.is_none() {
        caption(pixels,w,h,"WAITING FOR GAZE");
    }
}
fn caption(pixels:&mut[u32],w:usize,h:usize,label:&str){
    let scale=(w/92).max(1) as i32;
    let x=(w as i32-label.len() as i32*6*scale).max(0)/2;let y=h as i32-8*scale;
    // Tiny outline keeps status readable over either a light or dark desktop.
    for (dx,dy) in [(-1,0),(1,0),(0,-1),(0,1)] {crate::draw_text_scaled(pixels,w,h,x+dx,y+dy,label,0xff101820,scale);}
    crate::draw_text_scaled(pixels,w,h,x,y,label,0xffe0e7ee,scale);
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn off_and_generation_changes_reject_delayed_updates() {
        let mut c=Controller::default(); assert!(c.enabled_generation().is_none());
        c.command("ON"); let generation=c.enabled_generation().unwrap();
        c.command("OFF"); c.command("ON");
        c.publish(generation, Err("obsolete")); assert_ne!(c.status,"obsolete");
        assert!(c.target.is_none());
    }
    #[test] fn gaze_changes_pupils_and_missing_gaze_is_distinct() {
        let mut left=vec![0;184*112]; let mut right=left.clone(); let mut asleep=left.clone();
        render(&mut left,184,112,Some((0.,0.5)));
        render(&mut right,184,112,Some((1.,0.5)));
        render(&mut asleep,184,112,None);
        assert_ne!(left,right); assert_ne!(asleep,left);
        assert_eq!(left[0], right[0]); assert_eq!(left[0], asleep[0]);
        assert_eq!(left[0],0,"desktop outside the eyes must be transparent");
        assert!(left.iter().any(|p|p>>24==255),"eyes themselves must be opaque");
        if let Some(dir) = std::env::var_os("BUTTERCUP_UI_TEST_EXPORT") {
            crate::export_eye_ppm(&std::path::PathBuf::from(dir).join("wleyes.ppm"), &right, 184, 112).unwrap();
        }
    }
}
