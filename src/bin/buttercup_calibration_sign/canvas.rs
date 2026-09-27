//! CPU diagnostic canvas; system Cairo handles glyphs and antialiased geometry.
use super::super::Result;
use std::{
    ffi::{c_char, c_int, c_void, CString},
    path::Path,
};
type Ptr = *mut c_void;
#[link(name = "cairo")]
extern "C" {
    fn cairo_image_surface_create(format: c_int, w: c_int, h: c_int) -> Ptr;
    fn cairo_image_surface_create_for_data(
        data: *mut u8,
        format: c_int,
        w: c_int,
        h: c_int,
        stride: c_int,
    ) -> Ptr;
    fn cairo_image_surface_get_data(s: Ptr) -> *mut u8;
    fn cairo_image_surface_get_stride(s: Ptr) -> c_int;
    fn cairo_surface_status(s: Ptr) -> c_int;
    fn cairo_surface_flush(s: Ptr);
    fn cairo_surface_destroy(s: Ptr);
    fn cairo_surface_write_to_png(s: Ptr, p: *const c_char) -> c_int;
    fn cairo_create(s: Ptr) -> Ptr;
    fn cairo_destroy(c: Ptr);
    fn cairo_save(c: Ptr);
    fn cairo_restore(c: Ptr);
    fn cairo_new_path(c: Ptr);
    fn cairo_set_source_rgb(c: Ptr, r: f64, g: f64, b: f64);
    fn cairo_set_source_rgba(c: Ptr, r: f64, g: f64, b: f64, a: f64);
    fn cairo_set_source_surface(c: Ptr, s: Ptr, x: f64, y: f64);
    fn cairo_paint(c: Ptr);
    fn cairo_rectangle(c: Ptr, x: f64, y: f64, w: f64, h: f64);
    fn cairo_fill(c: Ptr);
    fn cairo_stroke(c: Ptr);
    fn cairo_set_line_width(c: Ptr, w: f64);
    fn cairo_move_to(c: Ptr, x: f64, y: f64);
    fn cairo_line_to(c: Ptr, x: f64, y: f64);
    fn cairo_arc(c: Ptr, x: f64, y: f64, r: f64, a: f64, b: f64);
    fn cairo_translate(c: Ptr, x: f64, y: f64);
    fn cairo_scale(c: Ptr, x: f64, y: f64);
    fn cairo_clip(c: Ptr);
    fn cairo_select_font_face(c: Ptr, f: *const c_char, slant: c_int, weight: c_int);
    fn cairo_set_font_size(c: Ptr, s: f64);
    fn cairo_show_text(c: Ptr, s: *const c_char);
}
pub const WHITE: [f64; 3] = [0.91, 0.94, 0.98];
pub const MUTED: [f64; 3] = [0.57, 0.65, 0.74];
pub const CYAN: [f64; 3] = [0.18, 0.84, 0.93];
pub const PINK: [f64; 3] = [1.0, 0.43, 0.71];
pub const ORANGE: [f64; 3] = [1.0, 0.64, 0.20];
pub const GREEN: [f64; 3] = [0.43, 0.92, 0.64];
pub const RED: [f64; 3] = [1.0, 0.36, 0.33];
pub struct Canvas {
    surface: Ptr,
    context: Ptr,
    pub w: usize,
    pub h: usize,
}
impl Canvas {
    pub fn new(w: usize, h: usize) -> Result<Self> {
        unsafe {
            let surface = cairo_image_surface_create(0, w as i32, h as i32);
            if surface.is_null() || cairo_surface_status(surface) != 0 {
                return Err("Cairo image allocation failed".into());
            }
            let context = cairo_create(surface);
            let c = Self {
                surface,
                context,
                w,
                h,
            };
            let font = CString::new("DejaVu Sans")?;
            cairo_select_font_face(context, font.as_ptr(), 0, 0);
            Ok(c)
        }
    }
    fn color(&self, c: [f64; 3]) {
        unsafe {
            cairo_set_source_rgb(self.context, c[0], c[1], c[2]);
        }
    }
    pub fn clear(&mut self) {
        self.rect(0., 0., self.w as f64, self.h as f64, [0.055, 0.075, 0.10]);
    }
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, c: [f64; 3]) {
        unsafe {
            self.color(c);
            cairo_rectangle(self.context, x, y, w, h);
            cairo_fill(self.context);
        }
    }
    pub fn text(&mut self, x: f64, y: f64, size: f64, c: [f64; 3], text: &str) {
        unsafe {
            cairo_new_path(self.context);
            self.color(c);
            cairo_set_font_size(self.context, size);
            cairo_move_to(self.context, x, y);
            let s = CString::new(text.replace('\0', " ")).unwrap();
            cairo_show_text(self.context, s.as_ptr());
            cairo_new_path(self.context);
        }
    }
    pub fn line(&mut self, a: [f64; 2], b: [f64; 2], width: f64, c: [f64; 3]) {
        self.path(&[a, b], width, c);
    }
    pub fn path(&mut self, p: &[[f64; 2]], width: f64, c: [f64; 3]) {
        if p.len() < 2 {
            return;
        }
        unsafe {
            self.color(c);
            cairo_set_line_width(self.context, width);
            cairo_move_to(self.context, p[0][0], p[0][1]);
            for p in &p[1..] {
                cairo_line_to(self.context, p[0], p[1]);
            }
            cairo_stroke(self.context);
        }
    }
    pub fn dot(&mut self, x: f64, y: f64, r: f64, c: [f64; 3], filled: bool) {
        unsafe {
            cairo_new_path(self.context);
            self.color(c);
            cairo_set_line_width(self.context, 2.);
            cairo_arc(self.context, x, y, r, 0., std::f64::consts::TAU);
            if filled {
                cairo_fill(self.context);
            } else {
                cairo_stroke(self.context);
            }
        }
    }
    pub fn cross(&mut self, x: f64, y: f64, r: f64, c: [f64; 3]) {
        self.line([x - r, y], [x + r, y], 2., c);
        self.line([x, y - r], [x, y + r], 2., c);
    }
    pub fn arrow(&mut self, a: [f64; 2], b: [f64; 2], c: [f64; 3]) {
        self.line(a, b, 3., c);
        let t = (b[1] - a[1]).atan2(b[0] - a[0]);
        for s in [-0.55, 0.55] {
            self.line(
                b,
                [b[0] - 9. * (t + s).cos(), b[1] - 9. * (t + s).sin()],
                3.,
                c,
            );
        }
    }
    pub fn image(&mut self, bgra: &[u8], sw: usize, sh: usize, x: f64, y: f64, w: f64, h: f64) {
        assert_eq!(bgra.len(), sw * sh * 4);
        unsafe {
            let surface = cairo_image_surface_create_for_data(
                bgra.as_ptr() as *mut u8,
                0,
                sw as i32,
                sh as i32,
                (sw * 4) as i32,
            );
            cairo_save(self.context);
            cairo_rectangle(self.context, x, y, w, h);
            cairo_clip(self.context);
            cairo_translate(self.context, x, y);
            cairo_scale(self.context, w / sw as f64, h / sh as f64);
            cairo_set_source_surface(self.context, surface, 0., 0.);
            cairo_paint(self.context);
            cairo_restore(self.context);
            cairo_surface_destroy(surface);
        }
    }
    pub fn shade(&mut self, x: f64, y: f64, w: f64, h: f64, alpha: f64) {
        unsafe {
            cairo_set_source_rgba(self.context, 0., 0., 0., alpha);
            cairo_rectangle(self.context, x, y, w, h);
            cairo_fill(self.context);
        }
    }
    pub fn clipped(&mut self, x: f64, y: f64, w: f64, h: f64, draw: impl FnOnce(&mut Self)) {
        unsafe {
            cairo_save(self.context);
            cairo_rectangle(self.context, x, y, w, h);
            cairo_clip(self.context);
        }
        draw(self);
        unsafe {
            cairo_restore(self.context);
        }
    }
    pub fn bytes(&mut self) -> &[u8] {
        unsafe {
            cairo_surface_flush(self.surface);
            let stride = cairo_image_surface_get_stride(self.surface) as usize;
            assert_eq!(stride, self.w * 4);
            std::slice::from_raw_parts(cairo_image_surface_get_data(self.surface), stride * self.h)
        }
    }
    pub fn png(&mut self, path: &Path) -> Result<()> {
        let path = CString::new(path.to_string_lossy().as_bytes())?;
        unsafe {
            cairo_surface_flush(self.surface);
            if cairo_surface_write_to_png(self.surface, path.as_ptr()) != 0 {
                return Err("Cairo PNG write failed".into());
            }
        }
        Ok(())
    }
}
impl Drop for Canvas {
    fn drop(&mut self) {
        unsafe {
            cairo_destroy(self.context);
            cairo_surface_destroy(self.surface);
        }
    }
}
