//! Main-screen information architecture. View navigation is presentation state;
//! camera acquisition is always a separate, explicit action.
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Scope {
    #[default]
    Roi,
    Linked,
    Global,
}
impl Scope {
    pub fn next(self) -> Self {
        match self {
            Self::Roi => Self::Linked,
            Self::Linked => Self::Global,
            Self::Global => Self::Roi,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Roi => "PREVIEW",
            Self::Linked => "LINKED VIEWS",
            Self::Global => "OVERVIEW",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum LinkedView {
    #[default]
    Compare,
    Timing,
    Contacts,
    TweakedContacts,
    StudentMaskOutline,
    StudentEllipseOnly,
    StudentPupilOnly,
}
impl LinkedView {
    fn available(method: SegmentationMode) -> &'static [Self] {
        match method {
            SegmentationMode::Sam31 => &[Self::Compare, Self::Timing, Self::Contacts, Self::TweakedContacts],
            SegmentationMode::EyeStudent => &[Self::Compare, Self::StudentEllipseOnly,
                Self::StudentMaskOutline, Self::StudentPupilOnly, Self::Contacts, Self::Timing],
            _ => &[Self::Compare, Self::Timing, Self::Contacts],
        }
    }
    fn next(self, method: SegmentationMode) -> Self {
        let views = Self::available(method);
        let index = views.iter().position(|view| *view == self).unwrap_or(0);
        views[(index + 1) % views.len()]
    }
    fn position_for(self, method: SegmentationMode) -> (usize, usize) {
        let views = Self::available(method);
        (views.iter().position(|view| *view == self).unwrap_or(0) + 1, views.len())
    }
    fn label(self) -> &'static str {
        match self {
            Self::Compare => "COMPARE",
            Self::Timing => "SOURCE TIMING",
            Self::Contacts => "CONTACT GEOMETRY",
            Self::TweakedContacts => "TWEAKED CONTACT GEOMETRY / EXPERIMENTAL",
            Self::StudentMaskOutline => "STUDENT MASK OUTLINES",
            Self::StudentEllipseOnly => "STUDENT FITTED LIMBUS ONLY",
            Self::StudentPupilOnly => "STUDENT PUPIL ONLY",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum GlobalView {
    #[default]
    Sensor,
    Objects,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Panel {
    #[default]
    Selection,
    Analysis,
    Camera,
    Diagnostics,
}
impl Panel {
    fn next(self) -> Self {
        match self {
            Self::Selection => Self::Analysis,
            Self::Analysis => Self::Camera,
            Self::Camera => Self::Diagnostics,
            Self::Diagnostics => Self::Selection,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Selection => "VIEW",
            Self::Analysis => "MODEL",
            Self::Camera => "CAM",
            Self::Diagnostics => "MORE",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RoiView {
    pub pixels: ViewMode,
    pub overlay: RoiOverlayMode,
}
impl Default for RoiView {
    fn default() -> Self {
        Self {
            pixels: ViewMode::QuadColor,
            overlay: RoiOverlayMode::SamOuterIrisMasks,
        }
    }
}
/// This controls presentation edits, never the detector or camera settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum PreviewEditScope {
    #[default]
    GlobalDefaults,
    SelectedPreview,
}
impl PreviewEditScope {
    fn next(self) -> Self {
        match self {
            Self::GlobalDefaults => Self::SelectedPreview,
            Self::SelectedPreview => Self::GlobalDefaults,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::GlobalDefaults => "GLOBAL PREVIEW DEFAULTS",
            Self::SelectedPreview => "SELECTED PREVIEW OVERRIDE",
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
struct PreviewOverrides {
    pixels: Option<ViewMode>,
    overlay: Option<RoiOverlayMode>,
}
impl PreviewOverrides {
    fn resolve(self, defaults: RoiView) -> RoiView {
        RoiView {
            pixels: self.pixels.unwrap_or(defaults.pixels),
            overlay: self.overlay.unwrap_or(defaults.overlay),
        }
    }
    fn label(self) -> &'static str {
        match (self.overlay.is_some(), self.pixels.is_some()) {
            (false, false) => "INHERITS GLOBAL PREVIEW DEFAULTS",
            (true, false) => "F OVERRIDE / V INHERITED",
            (false, true) => "F INHERITED / V OVERRIDE",
            (true, true) => "F + V OVERRIDDEN",
        }
    }
}
pub(super) struct Workspace {
    pub scope: Scope,
    pub selected: usize,
    pub preview_defaults: RoiView,
    pub preview_edit_scope: PreviewEditScope,
    preview_overrides: [PreviewOverrides; 2],
    pub linked: LinkedView,
    pub global: GlobalView,
    pub panel: Panel,
    pub scroll: usize,
    pub pointer: (f64, f64),
    pub hits: Vec<(Rect, Action)>,
}
impl Default for Workspace {
    fn default() -> Self {
        Self {
            scope: Scope::Roi,
            selected: 0,
            preview_defaults: RoiView::default(),
            preview_edit_scope: PreviewEditScope::default(),
            preview_overrides: [PreviewOverrides::default(); 2],
            linked: LinkedView::default(),
            global: GlobalView::default(),
            panel: Panel::default(),
            scroll: 0,
            pointer: (0.0, 0.0),
            hits: vec![],
        }
    }
}
impl Workspace {
    pub fn roi_view(&self, eye: usize) -> RoiView {
        self.preview_overrides[eye.min(1)].resolve(self.preview_defaults)
    }
    fn edit_view(&self) -> RoiView {
        match self.preview_edit_scope {
            PreviewEditScope::GlobalDefaults => self.preview_defaults,
            PreviewEditScope::SelectedPreview => self.roi_view(self.selected),
        }
    }
    fn set_overlay(&mut self, overlay: RoiOverlayMode) {
        match self.preview_edit_scope {
            PreviewEditScope::GlobalDefaults => self.preview_defaults.overlay = overlay,
            PreviewEditScope::SelectedPreview => self.preview_overrides[self.selected].overlay = Some(overlay),
        }
    }
    fn set_pixels(&mut self, pixels: ViewMode) {
        match self.preview_edit_scope {
            PreviewEditScope::GlobalDefaults => self.preview_defaults.pixels = pixels,
            PreviewEditScope::SelectedPreview => self.preview_overrides[self.selected].pixels = Some(pixels),
        }
    }
    fn reset_selected_preview(&mut self) {
        self.preview_overrides[self.selected] = PreviewOverrides::default();
    }
    pub fn cycle_view(&mut self, method: SegmentationMode) {
        match self.scope {
            Scope::Roi => {
                self.set_overlay(self.edit_view().overlay.cycled_for(method));
            }
            Scope::Linked => self.linked = self.linked.next(method),
            Scope::Global => {
                self.global = if self.global == GlobalView::Sensor {
                    GlobalView::Objects
                } else {
                    GlobalView::Sensor
                }
            }
        }
        self.scroll = 0;
    }
    pub fn title(&self, method: SegmentationMode) -> String {
        match self.scope {
            Scope::Roi => format!(
                "{} / {}",
                eye_name(self.selected),
                self.roi_view(self.selected)
                    .overlay
                    .normalized_for(method)
                    .label_for(method)
            ),
            Scope::Linked => self.linked.label().into(),
            Scope::Global => match self.global {
                GlobalView::Sensor => "SENSOR OVERVIEW",
                GlobalView::Objects => "PROMPTED OBJECT SEARCH",
            }
            .into(),
        }
    }
    pub fn object_view(&self) -> bool {
        self.scope == Scope::Global && self.global == GlobalView::Objects
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}
impl Rect {
    fn contains(self, p: (f64, f64)) -> bool {
        p.0 >= self.x as f64
            && p.1 >= self.y as f64
            && p.0 < (self.x + self.w) as f64
            && p.1 < (self.y + self.h) as f64
    }
    fn tuple(self) -> (usize, usize, usize, usize) {
        (self.x, self.y, self.w, self.h)
    }
    fn inset(self, n: usize) -> Self {
        let nx = n.min(self.w / 2);
        let ny = n.min(self.h / 2);
        Self {
            x: self.x + nx,
            y: self.y + ny,
            w: self.w - 2 * nx,
            h: self.h - 2 * ny,
        }
    }
}
pub(super) struct Layout {
    pub nav: Rect,
    pub detector: Rect,
    pub toolbar: Rect,
    pub canvas: Rect,
    pub inspector: Rect,
    pub footer: Rect,
}
impl Layout {
    pub fn new(w: usize, h: usize) -> Self {
        let nav_h = 40.min(h / 6);
        let detector_h = (if w >= 900 { 64 } else { 44 }).min(h / 4);
        let toolbar_h = 38.min(h / 6);
        let footer_h = 24.min(h / 8);
        let y = nav_h + detector_h + toolbar_h;
        let body_h = h.saturating_sub(y + footer_h);
        let body = Rect {
            x: 0,
            y,
            w,
            h: body_h,
        };
        let (canvas, inspector) = if w >= 1000 {
            let side = 320.min(w / 3);
            (
                Rect {
                    w: w - side,
                    ..body
                },
                Rect {
                    x: w - side,
                    w: side,
                    ..body
                },
            )
        } else {
            let panel = (body_h / 3).min(200);
            (
                Rect {
                    h: body_h - panel,
                    ..body
                },
                Rect {
                    y: y + body_h - panel,
                    h: panel,
                    ..body
                },
            )
        };
        Self {
            nav: Rect {
                x: 0,
                y: 0,
                w,
                h: nav_h,
            },
            detector: Rect {
                x: 0,
                y: nav_h,
                w,
                h: detector_h,
            },
            toolbar: Rect {
                x: 0,
                y: nav_h + detector_h,
                w,
                h: toolbar_h,
            },
            canvas,
            inspector,
            footer: Rect {
                x: 0,
                y: h - footer_h,
                w,
                h: footer_h,
            },
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Action {
    Scope(Scope),
    Select(usize),
    Panel(Panel),
    NextView,
    NextMethod,
    Search,
    FocusReference,
    SaveMonitor,
    AccuracyCheck,
    TogglePreviewEditScope,
    ResetPreviewOverrides,
}
pub(super) fn apply(app: &mut App, action: Action) {
    match action {
        Action::TogglePreviewEditScope => app.ui.preview_edit_scope = app.ui.preview_edit_scope.next(),
        Action::ResetPreviewOverrides => app.ui.reset_selected_preview(),
        Action::SaveMonitor => {
            if let Ok(mut s)=app.shared.lock() {
                if let Err(e)=s.monitor_location.save(){s.monitor_location.status=format!("MONITOR SAVE FAILED: {e}");}
            }
        }
        Action::AccuracyCheck => app.toggle_accuracy_check(),
        Action::Scope(scope) => app.ui.scope = scope,
        Action::Select(i) => {
            app.ui.selected = i.min(1);
            app.ui.scope = Scope::Roi;
        }
        Action::Panel(panel) => app.ui.panel = panel,
        Action::FocusReference => {
            if let Ok(mut s) = app.shared.lock() {
                if local_camera_control_allowed(&mut s, "FOCUS TARGET") {
                    s.focus_action = Some(FocusAction::SelectEye(app.ui.selected));
                }
            }
        }
        Action::NextView => {
            let method = app
                .shared
                .lock()
                .map(|s| s.segmentation_mode)
                .unwrap_or_default();
            app.ui.cycle_view(method);
        }
        Action::NextMethod => {
            if let Ok(mut state) = app.shared.lock() {
                cycle_segmentation_mode(&mut state, "detector selector");
            }
        }
        Action::Search => {
            if let Ok(mut s) = app.shared.lock() {
                if s.segmentation_mode==SegmentationMode::EyeStudent {
                    s.sam31_scene_prompt_status="STUDENT HAS FIXED EYE LABELS; SELECT SAM FOR OBJECT SEARCH".into();
                    return;
                }
                if s.segmentation_mode.uses_mask_geometry() && app.virtual_mouse.is_none() {
                    if !app.ui.object_view() && !s.sam31_object_inspection {
                        return;
                    }
                    if !s.sam31_object_inspection
                        && !local_camera_control_allowed(&mut s, "OBJECT SEARCH")
                    {
                        return;
                    }
                    if s.sam31_scene_prompt_bundle.is_none() {
                        s.sam31_scene_prompt_status =
                            "ENTER AND APPLY AN OBJECT PROMPT FIRST".into();
                        return;
                    }
                    invalidate_desktop_gaze_settings(&mut s);
                    s.sam31_object_inspection = !s.sam31_object_inspection;
                    s.sam31_scene_candidate = None;
                }
            }
        }
    }
    app.ui.scroll = 0;
    remember_roi(app);
    if let Ok(mut s) = app.shared.lock() {
        s.sam31_semantic_prompt = (s.segmentation_mode.uses_mask_geometry()).then_some(0);
    }
}
pub(super) fn cycle_pixels(app: &mut App) {
    if app.ui.scope != Scope::Global {
        app.ui.set_pixels(app.ui.edit_view().pixels.cycled());
        remember_roi(app);
    }
}
pub(super) fn select_pixels(app: &mut App, pixels: ViewMode) {
    app.ui.set_pixels(pixels);
    remember_roi(app);
}
pub(super) fn click(app: &mut App) {
    if let Some(action) = app
        .ui
        .hits
        .iter()
        .find(|(rect, _)| rect.contains(app.ui.pointer))
        .map(|(_, a)| *a)
    {
        apply(app, action);
    }
}
pub(super) fn next_panel(app: &mut App) {
    apply(app, Action::Panel(app.ui.panel.next()));
}
pub(super) fn remember_roi(app: &mut App) {
    let view = app.ui.roi_view(app.ui.selected);
    app.mode = view.pixels;
    app.roi_overlay_mode = view.overlay;
}
pub(super) fn eye_name(i: usize) -> &'static str {
    if i == 0 {
        "SUBJECT RIGHT"
    } else {
        "SUBJECT LEFT"
    }
}

const BG: u32 = 0x0009_101a;
const CARD: u32 = 0x0012_2030;
const INK: u32 = 0x00e5_ecf4;
const MUTED: u32 = 0x0097_aabd;
const ACCENT: u32 = 0x0062_d8c3;

struct Canvas<'a> {
    pixels: &'a mut [u32],
    w: usize,
    h: usize,
}
impl Canvas<'_> {
    fn fill(&mut self, r: Rect, c: u32) {
        fill_rect(
            self.pixels,
            self.w,
            self.h,
            r.x as i32,
            r.y as i32,
            r.w as i32,
            r.h as i32,
            c,
        );
    }
    fn text(&mut self, r: Rect, s: &str, color: u32) {
        // Text is clipped to its own surface, not merely the window edge.
        if r.w == 0 || r.h == 0 || r.x >= self.w || r.y >= self.h {
            return;
        }
        let mut p = vec![BG; r.w * r.h];
        let text = s.to_uppercase();
        let cols = r.w / 12;
        let text = if text.chars().count() > cols && cols >= 3 {
            text.chars().take(cols - 3).collect::<String>() + "..."
        } else {
            text
        };
        draw_text(&mut p, r.w, r.h, 0, 0, &text, color);
        for y in 0..r.h.min(self.h.saturating_sub(r.y)) {
            let len = r.w.min(self.w.saturating_sub(r.x));
            self.pixels[(r.y + y) * self.w + r.x..(r.y + y) * self.w + r.x + len]
                .copy_from_slice(&p[y * r.w..y * r.w + len]);
        }
    }
    fn large_text(&mut self, r: Rect, s: &str, color: u32) {
        // Integer glyph scaling keeps the detector readable on large displays.
        let (w,h)=(r.w/2,r.h/2);
        if w==0 || h==0 {return;}
        let mut pixels=vec![BG;w*h];
        Canvas {pixels:&mut pixels,w,h}.text(Rect {x:0,y:0,w,h},s,color);
        for y in 0..r.h.min(self.h.saturating_sub(r.y)) {
            for x in 0..r.w.min(self.w.saturating_sub(r.x)) {
                self.pixels[(r.y+y)*self.w+r.x+x]=pixels[(y/2).min(h-1)*w+(x/2).min(w-1)];
            }
        }
    }
    fn image(&mut self, r: Rect, p: &[u32], w: usize, h: usize) -> Rect {
        if r.w == 0 || r.h == 0 || w == 0 || h == 0 {
            return r;
        }
        let (x, y, w, h) = blit_scaled(self.pixels, self.w, self.h, p, w, h, r.x, r.y, r.w, r.h);
        Rect { x, y, w, h }
    }
    fn label(&mut self, r: Rect, title: &str) -> Rect {
        self.fill(r, CARD);
        let r = r.inset(10);
        self.text(
            Rect {
                h: 18.min(r.h),
                ..r
            },
            title,
            INK,
        );
        Rect {
            y: r.y + 24.min(r.h),
            h: r.h.saturating_sub(24),
            ..r
        }
    }
}

fn button(c: &mut Canvas, ui: &mut Workspace, r: Rect, label: &str, active: bool, action: Action) {
    c.fill(r, if active { 0x0025_4c55 } else { CARD });
    c.text(
        Rect {
            x: r.x + 6.min(r.w),
            y: r.y + 7.min(r.h),
            w: r.w.saturating_sub(12),
            h: 14.min(r.h),
        },
        label,
        if active { ACCENT } else { MUTED },
    );
    ui.hits.push((r, action));
}

fn selection_controls(c: &mut Canvas, ui: &mut Workspace, mut area: Rect, monitor_unsaved: bool) -> Rect {
    // A bottom inspector is short but wide. Use one row for its actions so
    // the controls cannot consume the entire scrollable status area.
    let compact = area.w >= 520 && area.h < 180;
    let overriding = ui.preview_edit_scope == PreviewEditScope::SelectedPreview;
    let edit_label = match (compact, overriding) {
        (true, false) => "S-TAB GLOBAL",
        (true, true) => "S-TAB OVERRIDE",
        (false, false) => "S-TAB: GLOBAL DEFAULTS",
        (false, true) => "S-TAB: PREVIEW OVERRIDE",
    };
    for (index, (label, action, active)) in [
        (edit_label, Action::TogglePreviewEditScope, overriding),
        (if compact { "SAVE MONITOR" } else { "SAVE MONITOR LOCATION" }, Action::SaveMonitor, monitor_unsaved),
        (if compact { "\\ ACCURACY" } else { "\\ ACCURACY CHECK - 20 TARGETS" }, Action::AccuracyCheck, false),
    ].into_iter().enumerate() {
        let rect = if compact {
            let column = area.w / 3;
            Rect { x: area.x + index * column, w: column.saturating_sub(4), h: 28.min(area.h), ..area }
        } else {
            Rect { h: 28.min(area.h), ..area }
        };
        if rect.h > 0 { button(c, ui, rect, label, active, action); }
        if !compact {
            area.y += 32.min(area.h);
            area.h = area.h.saturating_sub(32);
        }
    }
    if compact {
        area.y += 32.min(area.h);
        area.h = area.h.saturating_sub(32);
    }
    area
}

fn detector_label(method: SegmentationMode) -> String {
    format!("G {}/{} {}",method.ordinal(),SegmentationMode::COUNT,method.label().to_ascii_uppercase())
}

fn detector_scope(second: bool) -> &'static str {
    if second {"GLOBAL GAZE / BOTH ROIS"} else {"GLOBAL GAZE / RIGHT ROI"}
}

fn detector_bar(c: &mut Canvas, ui: &mut Workspace, r: Rect, method: SegmentationMode, second: bool) {
    let control=Rect {w:if r.w>=900 {540.min(r.w)} else {r.w},..r}.inset(4);
    c.fill(control,0x0025_4c55);
    let label=detector_label(method);
    let large=r.w>=900 && control.h>=48;
    let label_rect=Rect {x:control.x+6.min(control.w),y:control.y+3.min(control.h),
        w:control.w.saturating_sub(12),h:(if large {28}else{14}).min(control.h)};
    if large {c.large_text(label_rect,&label,ACCENT);} else {c.text(label_rect,&label,ACCENT);}
    c.text(Rect {x:label_rect.x,y:control.y+control.h.saturating_sub(16),
        w:label_rect.w,h:14.min(control.h)},detector_scope(second),INK);
    ui.hits.push((control,Action::NextMethod));
    if r.w>=900 {
        let hint=Rect {x:r.x+550,y:r.y+8,w:r.w.saturating_sub(562),h:18.min(r.h)};
        c.text(hint,"G: GLOBAL GAZE DETECTOR",INK);
        c.text(Rect {y:hint.y+24,..hint},"FOCUS + MOUSE + J + CALIBRATION",MUTED);
    }
}

fn roi_method_lines(method: SegmentationMode, frame_method: Option<SegmentationMode>, enabled: bool) -> [String;2] {
    [format!("G {}",method.label().to_ascii_uppercase()),
        if !enabled {"ANALYSIS OFF / 3 TO ENABLE".into()}
        else {match frame_method {
            Some(frame) if frame!=method=>format!("FRAME: {} / SWITCHING",frame.label().to_ascii_uppercase()),
            Some(frame)=>format!("FRAME: {}",frame.label().to_ascii_uppercase()),
            None=>"WAITING FOR FIRST FRAME".into(),
        }}]
}

fn overview(c: &mut Canvas, r: Rect, backdrop: Option<&Backdrop>, eyes: &[Option<EyeFrame>; 2]) {
    let body = c.label(r, "GLOBAL SNAPSHOT");
    if body.w == 0 || body.h == 0 {
        return;
    }
    if let Some(image) = backdrop {
        // Composite into the card's own clipped surface. In particular the
        // 1cm interval legend must not escape into a neighbouring inspector.
        let mut pixels = vec![CARD; body.w * body.h];
        let rect = blit_scaled(
            &mut pixels,
            body.w,
            body.h,
            &image.pixels,
            image.width,
            image.height,
            0,
            0,
            body.w,
            body.h,
        );
        draw_sensor_band_overlay(
            &mut pixels,
            body.w,
            body.h,
            rect,
            [eyes[0].as_ref(), eyes[1].as_ref()],
        );
        for (i, eye) in eyes.iter().enumerate() {
            if let Some(eye) = eye {
                let region =
                    project_sensor_rect((eye.sensor_x, eye.sensor_y, eye.width, eye.height), rect);
                draw_outline(
                    &mut pixels,
                    body.w,
                    body.h,
                    region,
                    2,
                    if i == 0 { ACCENT } else { 0x00e7_b577 },
                );
                draw_sensor_overview_eye_laser(&mut pixels, body.w, body.h, rect, eye);
            }
        }
        if let Some(scale) = eyes.iter().flatten().find_map(|eye| eye.centimeter_scale) {
            draw_centimeter_scale(
                &mut pixels,
                body.w,
                body.h,
                rect.0 as i32,
                rect.1 as i32,
                rect.3.min(86),
                rect.2 as f64 / PREVIEW_SENSOR_WIDTH.max(1) as f64,
                scale,
            );
        }
        c.image(body, &pixels, body.w, body.h);
    } else {
        c.text(body, "WAITING FOR GLOBAL CAPTURE", MUTED);
    }
}

fn roi_card(
    c: &mut Canvas,
    r: Rect,
    eye: Option<&EyeFrame>,
    i: usize,
    view: RoiView,
    method: SegmentationMode,
    enabled: bool,
    present: bool,
    checkerboard: &checkerboard_calibration::StatusSnapshot,
) {
    let heading = format!(
        "{} / {}",
        eye_name(i),
        if !enabled {
            "DISABLED"
        } else if present {
            "TRACKING"
        } else {
            "ID HELD"
        }
    );
    let mut body = c.label(r, &heading);
    let method_lines=roi_method_lines(method,eye.map(|f|f.segmentation_mode),enabled);
    for (line,color) in method_lines.iter().zip([ACCENT,MUTED]) {
        c.text(Rect {h:16.min(body.h),..body},line,color);
        let used=20.min(body.h);body.y+=used;body.h-=used;
    }
    if !enabled {
        c.text(body, "3 ENABLE SECOND ROI ANALYSIS", MUTED);
        return;
    }
    let Some(frame) = eye else {
        c.text(body, "NO CURRENT ROI FRAME", MUTED);
        return;
    };
    // Render native geometry once then fit the result into a bounded card.
    // No scaling decision may change the inference frame or source clock.
    let (source_width,source_height)=roi_card_source_size(frame,view.overlay);
    let w = source_width + 16;
    let h = source_height + 36;
    let mut pixels = vec![BG; w * h];
    let checkerboard = checkerboard.overlay.as_ref().filter(|overlay| {
        overlay.eye_index == i
            && overlay.sensor_origin == (frame.sensor_x, frame.sensor_y)
            && overlay.timestamp_ns.abs_diff(frame.timestamp_ns) <= 1_000_000_000
    });
    draw_eye_with_spatial_debug(
        &mut pixels,
        w,
        h,
        frame,
        8,
        28,
        view.pixels,
        "",
        false,
        present,
        1,
        view.overlay,
        checkerboard,
    );
    let image = Rect {
        h: body.h.saturating_sub(20),
        ..body
    };
    let mut image_pixels = Vec::with_capacity(source_width * source_height);
    for y in 28..28 + source_height {
        image_pixels.extend_from_slice(&pixels[y * w + 8..y * w + 8 + source_width]);
    }
    c.image(image, &image_pixels, source_width, source_height);
    let source = frame
        .sam31_proposal_masks
        .as_ref()
        .map_or(frame.sequence, |p| p.source_sequence);
    c.text(
        Rect {
            y: body.y + body.h.saturating_sub(16),
            h: 16.min(body.h),
            ..body
        },
        &format!(
            "LIVE {}  MASK {}  LAG {} FR",
            frame.sequence,
            source,
            frame.sequence.saturating_sub(source)
        ),
        MUTED,
    );
}

fn roi_card_source_size(frame:&EyeFrame,overlay:RoiOverlayMode)->(usize,usize) {
    if frame.segmentation_mode==SegmentationMode::EyeStudent && overlay.normalized_for(frame.segmentation_mode).uses_student_source() {
        return student_preview::source_dimensions(frame).unwrap_or((frame.width, frame.height));
    }
    if frame.segmentation_mode==SegmentationMode::Sam31 && overlay==RoiOverlayMode::SamTweakedContactGeometry {
        if let Some(p)=frame.sam31_proposal_masks.as_ref().filter(|p|p.source_width>0 && p.source_height>0) {
            return (p.source_width,p.source_height);
        }
    }
    (frame.width,frame.height)
}

fn split_pair(r: Rect) -> [Rect; 2] {
    let gap = 8.min(r.w / 4).min(r.h / 4);
    if r.w >= r.h {
        let w = (r.w - gap) / 2;
        [
            Rect { w, ..r },
            Rect {
                x: r.x + w + gap,
                w: r.w - w - gap,
                ..r
            },
        ]
    } else {
        let h = (r.h - gap) / 2;
        [
            Rect { h, ..r },
            Rect {
                y: r.y + h + gap,
                h: r.h - h - gap,
                ..r
            },
        ]
    }
}

#[derive(Clone)]
struct Snapshot {
    monitor_status:String,
    mouse_output_status: String,
    gaze_focus_status: String,
    gaze_settings_status: String,
    monitor_unsaved:bool,
    method: SegmentationMode,
    second: bool,
    object_running: bool,
    prompt: String,
    prompt_status: String,
    candidate: Option<sam31_outer::SceneCandidate>,
    recovery: Option<String>,
    laser: bool,
    follow: bool,
    exposure: Option<u16>,
    focus: Option<u16>,
    stacks: [EyePresenceStackStatus; 2],
    record: bool,
    eyes: [Option<EyeFrame>; 2],
    present: [bool; 2],
    backdrop: Option<Backdrop>,
    editor: Option<String>,
    focus_eye: usize,
    host_telemetry: Option<TelemetrySnapshot>,
    camera_telemetry: Option<TelemetrySnapshot>,
    checkerboard: checkerboard_calibration::StatusSnapshot,
    analysis_detail: String,
}

pub(super) fn render(
    app: &mut App,
    pixels: &mut [u32],
    w: usize,
    h: usize,
    presented_eyes: &[Option<EyeFrame>; 2],
) {
    let snapshot = {
        let s = app.shared.lock().unwrap();
        let (prompt, status) = if app.ui.object_view() {
            (&s.sam31_scene_prompt_text, &s.sam31_scene_prompt_status)
        } else {
            (&s.sam31_prompt_text, &s.sam31_prompt_status)
        };
        Snapshot {
            monitor_status:s.monitor_location.status.clone(),
            mouse_output_status: s.mouse_output.label(),
            gaze_focus_status: s.gaze_focus.label(),
            gaze_settings_status: presented_eyes[app.focus_eye].as_ref().map_or_else(
                || "GLOBAL GAZE: WAITING FOR REFERENCE EYE".into(),
                |frame| frame.gaze_policy_error.map_or_else(
                    || format!("GLOBAL GAZE: {} / {}",s.segmentation_mode.label(),eye_name(app.focus_eye)),
                    |reason| reason.to_ascii_uppercase())),
            monitor_unsaved:s.monitor_location.unsaved_candidate(),
            method: s.segmentation_mode,
            second: s.second_roi_enabled,
            object_running: s.sam31_object_inspection,
            prompt: prompt.clone(),
            prompt_status: status.clone(),
            candidate: s.sam31_scene_candidate.clone(),
            recovery: s.reacquire_status.clone(),
            laser: s.eye_laser_enabled,
            follow: s.region_follow_paused,
            exposure: s.exposure_actual,
            focus: s.focus_position,
            stacks: s.eye_presence_stacks.clone(),
            record: s
                .hotkey_record_result
                .as_ref()
                .is_some_and(|r| r.state == "recording"),
            eyes: presented_eyes.clone(),
            present: app.eye_identity_present,
            backdrop: app.backdrop.clone(),
            editor: app.sam31_prompt_editor.clone(),
            focus_eye: app.focus_eye,
            host_telemetry: app.host_telemetry.clone(),
            camera_telemetry: app.camera_telemetry.clone(),
            checkerboard: app.checkerboard_status.clone(),
            analysis_detail: s.segmentation_status.clone(),
        }
    };
    render_snapshot(&mut app.ui, pixels, w, h, &snapshot);
    if let Ok(mut s) = app.shared.lock() {
        s.ui_snapshot = serde_json::json!({
            "scope":app.ui.scope.label(),"view":app.ui.title(snapshot.method),"selected_roi":eye_name(app.ui.selected),
            "camera_focus_reference":eye_name(app.focus_eye),"panel":app.ui.panel.label(),
            "object_search_running":s.sam31_object_inspection,"iris_prompt":s.sam31_prompt_text,
            "object_prompt":s.sam31_scene_prompt_text,"prompt_editor":app.sam31_prompt_editor.is_some(),
            "iris_prompt_generation":s.sam31_prompt_bundle_generation,"object_prompt_generation":s.sam31_scene_prompt_generation,
            "object_prompt_status":s.sam31_scene_prompt_status,"recovery_status":s.reacquire_status,
            "roi_views":[app.ui.roi_view(0).overlay.label(),app.ui.roi_view(1).overlay.label()],
            "roi_pixels":[annotated_view_mode_name(app.ui.roi_view(0).pixels),annotated_view_mode_name(app.ui.roi_view(1).pixels)],
            "preview_edit_scope":app.ui.preview_edit_scope.label(),
            "preview_defaults":{"overlay":app.ui.preview_defaults.overlay.label(),"pixels":annotated_view_mode_name(app.ui.preview_defaults.pixels)},
            "preview_overrides":app.ui.preview_overrides.map(|r|serde_json::json!({
                "overlay":r.overlay.map(|o|o.label()),"pixels":r.pixels.map(annotated_view_mode_name)})),
            "global_gaze":{"detector":s.segmentation_mode.label(),"settings_generation":s.segmentation_generation,
                "reference_eye":eye_name(app.focus_eye),"outputs":"focus / uinput / J cursor / calibration / accuracy"},
            "monitor":s.monitor_location.snapshot(),
            "mouse_output":s.mouse_output.snapshot(),
            "gaze_focus":s.gaze_focus.snapshot(),
        });
    }
}

pub(super) fn configure_hotkeys(
    map: &mut keyboard_peeper::HotkeyMap,
    ui: &Workspace,
    editing: bool,
    search_running: bool,
) {
    for binding in &mut map.bindings {
        if editing {
            binding.enabled = binding.key == "Esc";
            if binding.key == "Esc" {
                binding.label = "Cancel prompt";
            }
            continue;
        }
        match binding.key {
            "V" => {
                binding.enabled = ui.scope != Scope::Global;
                binding.label = if ui.preview_edit_scope == PreviewEditScope::GlobalDefaults {
                    "Global preview pixels"
                } else { "Override preview pixels" };
            }
            "F" => binding.label = if ui.scope == Scope::Roi {
                if ui.preview_edit_scope == PreviewEditScope::GlobalDefaults { "Global preview overlay" }
                else { "Override preview overlay" }
            } else { "Workspace view only" },
            "G" => binding.label = "Global gaze detector",
            "Y" => binding.label = "Global pupil source",
            "Tab" => binding.label = "Preview / linked / overview",
            "Space" => {
                binding.enabled = ui.object_view() || search_running;
                binding.label = if search_running {
                    "Stop object search"
                } else {
                    "Start object search"
                };
            }
            "M" => binding.enabled = !search_running,
            "1" => binding.label = "Select left preview",
            "2" => binding.label = "Select right preview",
            _ => {}
        }
    }
    map.bindings.push(keyboard_peeper::Binding {
        // KPP/1 uses bit 1 for Shift (independent of winit's bit layout).
        modifiers: 1 << 1,
        enabled: !editing,
        key: "Tab",
        label: if ui.preview_edit_scope == PreviewEditScope::GlobalDefaults {
            "Edit selected preview"
        } else { "Edit global defaults" },
    });
    map.bindings.push(keyboard_peeper::Binding {
        modifiers: 0,
        enabled: !editing,
        key: "Backspace",
        label: "Preview: inherit defaults",
    });
    map.bindings.push(keyboard_peeper::Binding {
        modifiers: 0,
        enabled: editing || ui.object_view() || ui.scope == Scope::Roi,
        key: "Enter",
        label: if editing {
            "Apply prompt"
        } else {
            "Edit prompt"
        },
    });
}

fn render_snapshot(
    ui: &mut Workspace,
    pixels: &mut [u32],
    w: usize,
    h: usize,
    snapshot: &Snapshot,
) {
    let Snapshot {
        monitor_status,
        mouse_output_status,
        gaze_focus_status,
        gaze_settings_status,
        monitor_unsaved,
        method,
        second,
        object_running,
        prompt,
        prompt_status,
        candidate,
        recovery,
        laser,
        follow,
        exposure,
        focus,
        stacks,
        record,
        eyes,
        present,
        backdrop,
        editor,
        focus_eye,
        host_telemetry,
        camera_telemetry,
        checkerboard,
        analysis_detail,
    } = snapshot.clone();
    let layout = Layout::new(w, h);
    let mut c = Canvas { pixels, w, h };
    c.pixels.fill(BG);
    ui.hits.clear();
    if !LinkedView::available(method).contains(&ui.linked) {
        ui.linked=LinkedView::Contacts;
    }
    let tab_w = (layout.nav.w / 3).min(180);
    for (i, scope) in [Scope::Roi, Scope::Linked, Scope::Global]
        .into_iter()
        .enumerate()
    {
        let active = ui.scope == scope;
        button(
            &mut c,
            ui,
            Rect {
                x: i * tab_w,
                y: 4,
                w: tab_w.saturating_sub(4),
                h: layout.nav.h.saturating_sub(8),
            },
            scope.label(),
            active,
            Action::Scope(scope),
        );
    }
    if w >= 860 {
        for (i, label) in [(1, "1 LEFT"), (0, "2 RIGHT")] {
            let active = ui.selected == i;
            button(
                &mut c,
                ui,
                Rect {
                    x: 550 + (1 - i) * 100,
                    y: 4,
                    w: 96,
                    h: 28,
                },
                label,
                active,
                Action::Select(i),
            );
        }
    }
    if record && w >= 1000 {
        c.text(
            Rect {
                x: 760,
                y: 10,
                w: w.saturating_sub(770),
                h: 16,
            },
            "RECORDING RAW",
            0x00ff_897d,
        );
    }
    detector_bar(&mut c,ui,layout.detector,method,second);
    let (position,count)=match ui.scope {
        Scope::Roi=>ui.roi_view(ui.selected).overlay.position_for(method),
        Scope::Linked=>ui.linked.position_for(method),
        Scope::Global=>(if ui.global==GlobalView::Sensor {1}else{2},2),
    };
    let title = format!("F VIEW {position}/{count} / {}",ui.title(method));
    c.text(
        Rect {
            x: 10,
            y: layout.toolbar.y + 10,
            w: layout.toolbar.w.saturating_sub(120),
            h: 16,
        },
        &title,
        INK,
    );
    button(
        &mut c,
        ui,
        Rect {
            x: w.saturating_sub(104),
            y: layout.toolbar.y + 4,
            w: 100.min(w),
            h: 28.min(layout.toolbar.h),
        },
        "F NEXT",
        false,
        Action::NextView,
    );
    let area = layout.canvas.inset(8);
    let mut eyes = eyes;
    for frame in eyes.iter_mut().flatten() {
        frame.eye_laser_enabled = laser && !object_running;
    }
    match ui.scope {
        Scope::Roi => {
            let context_h = if area.h > area.w * 6 / 5 {
                area.h
                    .saturating_sub((area.w * 2 / 3 + 80).min(area.h * 2 / 3))
            } else {
                (area.h / 4).min(160)
            };
            let context = Rect {
                y: area.y + area.h - context_h,
                h: context_h,
                ..area
            };
            let primary = Rect {
                h: area.h.saturating_sub(context_h + 8),
                ..area
            };
            let i = ui.selected;
            roi_card(
                &mut c,
                primary,
                eyes[i].as_ref(),
                i,
                ui.roi_view(i),
                method,
                i == 0 || second,
                present[i],
                &checkerboard,
            );
            overview(&mut c, context, backdrop.as_ref(), &eyes);
        }
        Scope::Linked => {
            let info_h = 76.min(area.h / 3);
            let cards = split_pair(Rect {
                h: area.h.saturating_sub(info_h + 8),
                ..area
            });
            for i in 0..2 {
                let mut view = ui.roi_view(i);
                if ui.linked == LinkedView::Contacts {
                    view.overlay = RoiOverlayMode::SamDeflattenedVirtualContact;
                }
                if ui.linked == LinkedView::TweakedContacts {
                    view.overlay = RoiOverlayMode::SamTweakedContactGeometry;
                }
                if ui.linked == LinkedView::Timing {
                    view.overlay = RoiOverlayMode::Clean;
                }
                match ui.linked {
                    LinkedView::StudentMaskOutline => view.overlay = RoiOverlayMode::StudentMaskOutline,
                    LinkedView::StudentEllipseOnly => view.overlay = RoiOverlayMode::StudentEllipseOnly,
                    LinkedView::StudentPupilOnly => view.overlay = RoiOverlayMode::StudentPupilOnly,
                    _ => {}
                }
                roi_card(
                    &mut c,
                    cards[i],
                    eyes[i].as_ref(),
                    i,
                    view,
                    method,
                    i == 0 || second,
                    present[i],
                    &checkerboard,
                );
            }
            let info = Rect {
                y: area.y + area.h - info_h,
                h: info_h,
                ..area
            };
            let mut rows = vec![format!("SHARED DETECTOR: {} / ENABLED ROIS",method.label().to_ascii_uppercase())];
            match (&eyes[0], &eyes[1]) {
                (Some(a), Some(b)) if second => {
                    let delta = a.timestamp_ns.abs_diff(b.timestamp_ns);
                    rows.push(format!(
                        "RAW SOURCE DIFFERENCE {:.2} MS / {}",
                        delta as f64 / 1e6,
                        if delta == 0 {
                            "SAME TIMESTAMP"
                        } else {
                            "DIFFERENT EXPOSURES"
                        }
                    ));
                    rows.push(format!(
                        "MASK SOURCE R {} / L {}",
                        a.sam31_proposal_masks
                            .as_ref()
                            .map_or(0, |p| p.source_timestamp_ns),
                        b.sam31_proposal_masks
                            .as_ref()
                            .map_or(0, |p| p.source_timestamp_ns)
                    ));
                }
                _ => rows.push("WAITING FOR TWO ENABLED ROI STREAMS / 3 TOGGLE SECOND ROI".into()),
            }
            text_rows(&mut c, info, &rows, 0);
        }
        Scope::Global => {
            if ui.global == GlobalView::Sensor {
                overview(&mut c, area, backdrop.as_ref(), &eyes);
                if let Some(overlay) = checkerboard.overlay.as_ref().filter(|o| o.full_sensor) {
                    if area.w > 0 && area.h > 0 {
                        let mut preview = vec![BG; area.w * area.h];
                        draw_full_checkerboard_preview(
                            &mut preview,
                            area.w,
                            area.h,
                            area.w,
                            overlay,
                        );
                        c.image(area, &preview, area.w, area.h);
                    }
                }
            } else {
                let split = if candidate.is_some() {
                    split_pair(area)
                } else {
                    let note = 60.min(area.h / 4);
                    [
                        Rect {
                            h: area.h.saturating_sub(note + 8),
                            ..area
                        },
                        Rect {
                            y: area.y + area.h - note,
                            h: note,
                            ..area
                        },
                    ]
                };
                let mut scene = backdrop.clone();
                if let (Some(scene), Some(candidate)) = (&mut scene, &candidate) {
                    let mut p = (*scene.pixels).clone();
                    let (mw, mh) = candidate.mask_size;
                    if mw > 0 && mh > 0 && candidate.mask.len() == mw * mh {
                        for y in 0..scene.height {
                            for x in 0..scene.width {
                                if candidate.mask[y * mh / scene.height * mw + x * mw / scene.width]
                                    != 0
                                {
                                    let i = y * scene.width + x;
                                    p[i] = ((p[i] & 0x00fe_fefe) >> 1) + 0x0000_6040;
                                }
                            }
                        }
                    }
                    scene.pixels = Arc::new(p);
                }
                overview(&mut c, split[0], scene.as_ref(), &[None, None]);
                let body = c.label(split[1], "OBJECT CROP");
                if let (Some(scene), Some(candidate)) = (scene, candidate) {
                    let (x, y, cw, ch) =
                        scene_crop_bounds(candidate.bounds, scene.width, scene.height);
                    let mut crop = Vec::with_capacity(cw * ch);
                    for yy in y..y + ch {
                        crop.extend_from_slice(
                            &scene.pixels[yy * scene.width + x..yy * scene.width + x + cw],
                        );
                    }
                    c.image(body, &crop, cw, ch);
                } else {
                    c.text(
                        body,
                        if object_running {
                            "NO MATCH YET / GLOBAL SEARCH CONTINUES"
                        } else {
                            "SPACE START OBJECT SEARCH"
                        },
                        MUTED,
                    );
                }
            }
        }
    }
    let panel = layout.inspector.inset(8);
    let pw = panel.w / 4;
    for (i, p) in [
        Panel::Selection,
        Panel::Analysis,
        Panel::Camera,
        Panel::Diagnostics,
    ]
    .into_iter()
    .enumerate()
    {
        let active = ui.panel == p;
        button(
            &mut c,
            ui,
            Rect {
                x: panel.x + i * pw,
                y: panel.y,
                w: pw.saturating_sub(2),
                h: 28.min(panel.h),
            },
            p.label(),
            active,
            Action::Panel(p),
        );
    }
    let mut text_area = Rect {
        y: panel.y + 36.min(panel.h),
        h: panel.h.saturating_sub(36),
        ..panel
    };
    if ui.panel == Panel::Selection && ui.object_view() {
        button(
            &mut c,
            ui,
            Rect {
                h: 28.min(text_area.h),
                ..text_area
            },
            if object_running {
                "SPACE STOP SEARCH"
            } else {
                "SPACE START SEARCH"
            },
            object_running,
            Action::Search,
        );
        text_area.y += 32.min(text_area.h);
        text_area.h = text_area.h.saturating_sub(32);
    }
    let mut rows = vec![];
    if ui.panel==Panel::Selection && !ui.object_view() {
        text_area = selection_controls(&mut c, ui, text_area, monitor_unsaved);
        rows.push(gaze_settings_status.clone());
        rows.push(monitor_status);
        rows.push(mouse_output_status);
        rows.push(gaze_focus_status);
    }
    if object_running {
        rows.push("OBJECT SEARCH ACTIVE / SPACE STOP".into());
        rows.push("FINE EYE ANALYSIS PAUSED".into());
    }
    match ui.panel {
        Panel::Selection => {
            if ui.scope != Scope::Global {
                rows.push(format!("PREVIEW: {}", eye_name(ui.selected)));
                rows.push("1 left / 2 right (view only)".into());
                rows.push(ui.preview_overrides[ui.selected].label().into());
                rows.push(format!("SHIFT+TAB EDIT {}",ui.preview_edit_scope.label()));
                rows.push("BACKSPACE: INHERIT BOTH DEFAULTS".into());
                rows.push("F overlays / V pixels; no gaze changes".into());
                if ui.preview_edit_scope==PreviewEditScope::GlobalDefaults {
                    rows.push(format!("GLOBAL F: {}",ui.preview_defaults.overlay.normalized_for(method).label_for(method)));
                    rows.push(format!("GLOBAL V: {}",annotated_view_mode_name(ui.preview_defaults.pixels)));
                }
                rows.push(format!(
                    "V {}",
                    annotated_view_mode_name(ui.roi_view(ui.selected).pixels)
                ));
                rows.push(format!(
                    "J gaze overlays {}",
                    if laser { "ON" } else { "OFF" }
                ));
                rows.push(format!("M CALIBRATE {}",eye_name(focus_eye)));
                if ui.selected!=focus_eye {rows.push("F2 USE SELECTED EYE FOR AF + ALL GAZE OUTPUT".into());}
            }
            if ui.object_view() {
                if method==SegmentationMode::EyeStudent {
                    rows.push("OBJECT SEARCH REQUIRES SAM; G TO SWITCH".into());
                }
                rows.push(format!("PROMPT: {}", editor.as_deref().unwrap_or(&prompt)));
                rows.push(prompt_status);
                rows.push("Enter edit object prompt".into());
                rows.push("Object crops are not eye evidence.".into());
                rows.push("Global-image crop, not native ROI.".into());
            } else if method.uses_mask_geometry() && ui.scope==Scope::Roi {
                if method==SegmentationMode::EyeStudent {
                    rows.push("CUDA EYE STUDENT: FIXED EYE LABELS".into());
                    rows.push("SAM-trained masks; shared RAW + 3D solver".into());
                    rows.push("Experimental; G switches back to SAM".into());
                } else {
                    rows.push(format!("IRIS PROMPT: {prompt}"));
                    rows.push("Enter edit / Esc cancel".into());
                }
            }
            if let Some(status) = recovery {
                rows.push(status);
            }
        }
        Panel::Analysis => {
            rows.push("GLOBAL GAZE / ALL ENABLED EYES".into());
            rows.push(gaze_settings_status);
            rows.push(method.method_control_label());
            rows.push("Y global rough-center source".into());
            rows.push("Same source for focus, mouse, J, M and accuracy.".into());
            rows.push("F/V are previews only, never gaze input.".into());
            rows.push("No per-region analysis overrides.".into());
            rows.push(format!(
                "3 second ROI {}",
                if second { "ON" } else { "OFF" }
            ));
            rows.push("U size reticles / K edge method".into());
            rows.push("Iris bounds [ ] - = / 0 auto".into());
            rows.push("Pupil bounds: arrow keys".into());
            rows.extend(eye_presence_stack_text_rows(&stacks, "", second));
            rows.push(analysis_detail);
        }
        Panel::Camera => {
            rows.push("PHYSICAL CONTROLS / ENTIRE CAMERA".into());
            rows.push(format!(
                "Focus position {} / reference {}",
                focus.map_or("unknown".into(), |v| v.to_string()),
                eye_name(focus_eye)
            ));
            rows.push("F2 use selected ROI as AF reference".into());
            rows.push("; ' focus -/+ / L autofocus".into());
            rows.push(format!(
                "Exposure {} / A auto",
                exposure.map_or("unknown".into(), |v| v.to_string())
            ));
            rows.push("/ darker / . brighter".into());
            rows.push(format!(
                "W ROI + sensor follow {}",
                if follow { "PAUSED" } else { "ON" }
            ));
            rows.push("R eye reacquisition enable/disable".into());
            rows.push("B lightbox / N pattern".into());
            rows.push(checkerboard.one_line());
            rows.push("C camera geometry calibration".into());
            rows.push("X lighthouse / Z optical clock".into());
            rows.push("S/H raw recording / D right still".into());
            rows.push(format!(
                "Recording {}",
                if record { "ACTIVE" } else { "OFF" }
            ));
        }
        Panel::Diagnostics => {
            rows.push("TIMING / MICROSECONDS".into());
            for (stage, label) in [
                (HOST_STAGE_PACKET_INTERVAL, "PACKET"),
                (HOST_STAGE_RAW_SET_INTERVAL, "RAW SET"),
                (HOST_STAGE_PACKET_READ, "READ"),
                (HOST_STAGE_EYE_PROCESS, "EYE"),
                (HOST_STAGE_TRACK_AUTOFOCUS, "AF"),
                (HOST_STAGE_REACQUIRE, "REACQUIRE"),
                (HOST_STAGE_TEMPORAL_MATCH, "TEMPORAL MATCH"),
                (HOST_STAGE_CLUSTER_FIT, "CLUSTER FIT"),
            ] {
                rows.push(timing_metric(host_telemetry.as_ref(), stage, label, false));
            }
            for (stage, label) in [
                (CAMERA_STAGE_SENSOR_INTERVAL, "SENSOR"),
                (CAMERA_STAGE_SENSOR_ACQUIRE, "ACQUIRE"),
                (CAMERA_STAGE_CONTEXT_BUILD, "CONTEXT"),
                (CAMERA_STAGE_ROI_SLICE, "SLICE"),
                (CAMERA_STAGE_STREAM_WRITE, "SEND"),
                (CAMERA_STAGE_VCM_COMMAND, "VCM"),
                (CAMERA_STAGE_VCM_REMAINING, "SETTLE"),
            ] {
                rows.push(timing_metric(
                    camera_telemetry.as_ref(),
                    stage,
                    label,
                    false,
                ));
            }
            rows.extend(eye_presence_stack_text_rows(&stacks, "", second));
            rows.push("PgUp/PgDn or wheel to scroll".into());
        }
    }
    text_rows(&mut c, text_area, &rows, ui.scroll);
    c.text(
        layout.footer.inset(4),
        if w >= 1180 {
            "G GLOBAL GAZE   F/V PREVIEW   SHIFT+TAB EDIT SCOPE   BACKSPACE INHERIT   TAB WORKSPACE   , PANEL"
        } else if w >= 700 {
            "G GAZE  F/V VIEW  SHIFT+TAB SCOPE  BKSP INHERIT  TAB NAV"
        } else if w >= 520 {
            "G GAZE  F/V VIEW  S-TAB SCOPE  BKSP INHERIT"
        } else {
            "F/V VIEW  S-TAB SCOPE"
        },
        MUTED,
    );
    if let Some(editor) = editor {
        let r = Rect {
            x: 8,
            y: layout.toolbar.y,
            w: w.saturating_sub(16),
            h: 70.min(h.saturating_sub(layout.toolbar.y)),
        };
        c.fill(r, 0x0025_4c55);
        c.text(
            Rect {
                h: 18.min(r.h),
                ..r
            },
            if ui.object_view() {
                "EDIT OBJECT PROMPT / ENTER APPLY / ESC CANCEL"
            } else {
                "EDIT IRIS PROMPT / ENTER APPLY / ESC CANCEL"
            },
            ACCENT,
        );
        let cols = (r.w / 12).max(1);
        let text = editor
            .chars()
            .rev()
            .take(cols * 2)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>();
        text_rows(
            &mut c,
            Rect {
                y: r.y + 24,
                h: r.h.saturating_sub(24),
                ..r
            },
            &[text],
            0,
        );
    }
}

fn text_rows(c: &mut Canvas, r: Rect, rows: &[String], scroll: usize) {
    let cols = (r.w / 12).max(1);
    let lines = wrapped_lines(rows, cols);
    let capacity = r.h / 20;
    if capacity == 0 {
        return;
    }
    let overflow = lines.len() > capacity;
    let count = if overflow && capacity > 1 {
        capacity - 1
    } else {
        capacity
    };
    let start = scroll.min(lines.len().saturating_sub(count));
    for (i, line) in lines.iter().skip(start).take(count).enumerate() {
        c.text(
            Rect {
                y: r.y + i * 20,
                h: 16,
                ..r
            },
            line,
            INK,
        );
    }
    if overflow && capacity > 1 {
        c.text(
            Rect {
                y: r.y + (capacity - 1) * 20,
                h: 16,
                ..r
            },
            &format!(
                "PGUP/DN {}-{} OF {}",
                start + 1,
                (start + count).min(lines.len()),
                lines.len()
            ),
            ACCENT,
        );
    }
}

fn wrapped_lines(rows: &[String], cols: usize) -> Vec<String> {
    let cols = cols.max(1);
    let mut result = vec![];
    for row in rows {
        let mut line = String::new();
        for word in row.split_whitespace() {
            if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > cols {
                result.push(std::mem::take(&mut line));
            }
            let chars = word.chars().collect::<Vec<_>>();
            for chunk in chars.chunks(cols) {
                if !line.is_empty() {
                    line.push(' ');
                }
                line.extend(chunk);
                if line.chars().count() >= cols {
                    result.push(std::mem::take(&mut line));
                }
            }
        }
        if !line.is_empty() {
            result.push(line);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn example_snapshot() -> Snapshot {
        let mut frame = crate::tests::control_eye_frame(120);
        frame.width = 384;
        frame.height = 256;
        frame.segmentation_mode = SegmentationMode::Sam31;
        let raw = (0..384 * 256)
            .map(|i| {
                let x = (i % 384) as f64 - 192.0;
                let y = (i / 384) as f64 - 128.0;
                if x * x / 6400.0 + y * y / 3600.0 < 1.0 {
                    120u16
                } else {
                    650u16
                }
            })
            .collect::<Vec<_>>();
        frame.quad_color = Arc::new(color_preview(&raw, 384, 256, 0, 0, 100, None));
        let ellipse = geometry::Ellipse {
            center: (192.0, 128.0),
            major_radius: 80.0,
            minor_radius: 60.0,
            angle: 0.0,
        };
        frame.sam31_proposal_masks = Some(Arc::new(sam31_outer::ProposalMasks {
            source_width: 384,
            source_height: 256,
            source_sequence: 118,
            source_timestamp_ns: 100_000,
            source_raw: Arc::new(raw),
            outer_fit: Some(sam31_outer::OuterMaskFitReview {
                ellipse,
                source_component_area_px: 15000.0,
                retained_points: Arc::new(ellipse.dense_points(64)),
                conic_segments: Arc::new(vec![]),
                flat_tire_points: Arc::new(vec![]),
                upper_flat_tire: false,
                lower_flat_tire: false,
            }),
            ..sam31_outer::ProposalMasks::default()
        }));
        let mut left = frame.clone();
        left.sequence = 121;
        left.timestamp_ns += 10_000_000;
        left.sensor_x += 1200;
        Snapshot {
            method: SegmentationMode::Sam31,
            mouse_output_status: "MOUSE OFF / Super+Shift+M".into(),
            gaze_focus_status: "GAZE FOCUS OFF / Super+Shift+F".into(),
            gaze_settings_status: "GLOBAL GAZE: SAM31 / SUBJECT RIGHT".into(),
            second: true,
            object_running: false,
            prompt: "iris".into(),
            prompt_status: "READY".into(),
            candidate: None,
            recovery: Some("SYNTHETIC PRESENTATION FIXTURE / NOT A CAMERA FRAME".into()),
            laser: false,
            follow: false,
            exposure: Some(2000),
            focus: Some(564),
            stacks: std::array::from_fn(|_| EyePresenceStackStatus::default()),
            record: false,
            eyes: [Some(frame), Some(left)],
            present: [true, false],
            backdrop: Some(Backdrop {
                width: 400,
                height: 300,
                pixels: Arc::new(vec![0x0030_4050; 400 * 300]),
            }),
            editor: None,
            focus_eye: 0,
            host_telemetry: None,
            camera_telemetry: None,
            checkerboard: checkerboard_calibration::StatusSnapshot::default(),
            analysis_detail: "SYNTHETIC FIXTURE".into(),
            monitor_status:"MONITOR POSE SESSION ONLY".into(),
            monitor_unsaved:true,
        }
    }
    #[test]
    fn render_all_workspace_views_desktop_and_compact() {
        for (w, h) in [
            (320, 240),
            (640, 480),
            (1200, 850),
            (800, 1200),
            (904, 2048),
        ] {
            for (name, scope, linked, global) in [
                ("roi", Scope::Roi, LinkedView::Compare, GlobalView::Sensor),
                (
                    "linked",
                    Scope::Linked,
                    LinkedView::Compare,
                    GlobalView::Sensor,
                ),
                (
                    "timing",
                    Scope::Linked,
                    LinkedView::Timing,
                    GlobalView::Sensor,
                ),
                (
                    "contacts",
                    Scope::Linked,
                    LinkedView::Contacts,
                    GlobalView::Sensor,
                ),
                (
                    "global",
                    Scope::Global,
                    LinkedView::Compare,
                    GlobalView::Sensor,
                ),
                (
                    "objects",
                    Scope::Global,
                    LinkedView::Compare,
                    GlobalView::Objects,
                ),
            ] {
                let mut ui = Workspace {
                    scope,
                    linked,
                    global,
                    ..Workspace::default()
                };
                let mut pixels = vec![0; w * h];
                render_snapshot(&mut ui, &mut pixels, w, h, &example_snapshot());
                assert!(pixels.iter().any(|p| *p == INK));
                assert!(ui.hits.len() >= 7);
                if let Some(dir) = std::env::var_os("BUTTERCUP_UI_TEST_EXPORT") {
                    export_eye_ppm(
                        &PathBuf::from(dir).join(format!("{name}-{w}x{h}.ppm")),
                        &pixels,
                        w,
                        h,
                    )
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn detector_banner_is_visible_and_clickable_in_every_scope() {
        // The requested detector changes immediately, even while both frames
        // still belong to SAM. A heading must not quietly report the old mode.
        let mut snapshot = example_snapshot();
        snapshot.method = SegmentationMode::EyeStudent;
        for (w, h) in [(320, 240), (640, 480), (1200, 850), (904, 2048)] {
            let mut pixels = vec![0; w * h];
            let mut reference = vec![BG; w * h];
            let banner = Layout::new(w, h).detector;
            detector_bar(
                &mut Canvas { pixels: &mut reference, w, h },
                &mut Workspace::default(),
                banner,
                snapshot.method,
                snapshot.second,
            );
            for scope in [Scope::Roi, Scope::Linked, Scope::Global] {
                let mut ui = Workspace { scope, ..Workspace::default() };
                render_snapshot(&mut ui, &mut pixels, w, h, &snapshot);
                let controls: Vec<_> = ui.hits.iter()
                    .filter(|(_, action)| matches!(action, Action::NextMethod))
                    .collect();
                assert_eq!(controls.len(), 1);
                let (control, _) = controls[0];
                assert!(banner.contains((control.x as f64, control.y as f64)));
                assert!(control.y + control.h <= banner.y + banner.h);
                for y in banner.y..banner.y + banner.h {
                    assert_eq!(&pixels[y*w..(y+1)*w], &reference[y*w..(y+1)*w]);
                }
                if scope == Scope::Linked {
                    if let Some(dir) = std::env::var_os("BUTTERCUP_UI_TEST_EXPORT") {
                        export_eye_ppm(
                            &PathBuf::from(dir).join(format!("detector-switch-{w}x{h}.ppm")),
                            &pixels, w, h,
                        ).unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn detector_names_fit_and_identify_the_shared_scope() {
        assert_eq!(detector_label(SegmentationMode::EyeStudent), "G 3/6 EYE-STUDENT");
        assert_eq!(detector_scope(true), "GLOBAL GAZE / BOTH ROIS");
        assert_eq!(detector_scope(false), "GLOBAL GAZE / RIGHT ROI");
        let mut method = SegmentationMode::Native;
        for _ in 0..SegmentationMode::COUNT {
            // 520px at double-size and 300px at normal size: no ellipsis,
            // including the longest detector name, VESSEL-FEATURES.
            let glyphs = detector_label(method).chars().count();
            assert!(glyphs * 24 <= 520);
            assert!(glyphs * 12 <= 300);
            method = method.cycled();
        }
    }

    #[test]
    fn roi_detector_labels_distinguish_requested_frame_and_disabled_states() {
        let student = SegmentationMode::EyeStudent;
        assert_eq!(roi_method_lines(student, Some(SegmentationMode::Sam31), true),
            ["G EYE-STUDENT", "FRAME: SAM31 / SWITCHING"]);
        // An enum match is not proof of an accepted pupil/contact solution.
        assert_eq!(roi_method_lines(student, Some(student), true),
            ["G EYE-STUDENT", "FRAME: EYE-STUDENT"]);
        assert_eq!(roi_method_lines(student, None, true),
            ["G EYE-STUDENT", "WAITING FOR FIRST FRAME"]);
        assert_eq!(roi_method_lines(student, Some(SegmentationMode::Sam31), false),
            ["G EYE-STUDENT", "ANALYSIS OFF / 3 TO ENABLE"]);
    }

    #[test]
    fn detector_selector_and_keyboard_share_the_complete_cycle() {
        for source in ["G hotkey", "detector selector"] {
            let mut state = SharedState {
                segmentation_mode: SegmentationMode::Native,
                rough_pupil_center_mode: RoughPupilCenterMode::IrisGuided,
                ..SharedState::default()
            };
            for (index, expected) in [
                SegmentationMode::Sam31,
                SegmentationMode::EyeStudent,
                SegmentationMode::Clusters,
                SegmentationMode::Driving,
                SegmentationMode::ScleraRedCanny,
                SegmentationMode::Native,
            ].into_iter().enumerate() {
                cycle_segmentation_mode(&mut state, source);
                assert_eq!(state.segmentation_mode, expected);
                assert_eq!(state.iris_segmentation_generation, index as u64 + 1);
            }
        }
    }

    #[test]
    fn eye_student_renders_every_sam_roi_view_and_linked_contact_view() {
        let mut snapshot=example_snapshot();
        snapshot.method=SegmentationMode::EyeStudent;
        snapshot.prompt_status="FIXED EYE LABELS / CUDA STUDENT".into();
        for frame in snapshot.eyes.iter_mut().flatten() {
            frame.segmentation_mode=SegmentationMode::EyeStudent;
        }
        let mut ui=Workspace::default();
        let mut pixels=vec![0;1200*850];
        for &overlay in RoiOverlayMode::available(SegmentationMode::EyeStudent) {
            ui.preview_defaults.overlay=overlay;
            render_snapshot(&mut ui,&mut pixels,1200,850,&snapshot);
            assert!(pixels.iter().any(|p|*p==INK));
            assert!(ui.hits.len()>=7);
        }
        ui.scope=Scope::Linked;
        ui.linked=LinkedView::Contacts;
        render_snapshot(&mut ui,&mut pixels,1200,850,&snapshot);
        assert!(pixels.iter().any(|p|*p==INK));
        if let Some(dir)=std::env::var_os("BUTTERCUP_UI_TEST_EXPORT") {
            export_eye_ppm(&PathBuf::from(dir).join("eye-student-contacts-1200x850.ppm"),
                &pixels,1200,850).unwrap();
        }
    }
    #[test]
    fn tweaked_contact_is_a_sam_only_linked_view() {
        assert_eq!(LinkedView::Contacts.next(SegmentationMode::Sam31),LinkedView::TweakedContacts);
        assert_eq!(LinkedView::TweakedContacts.next(SegmentationMode::Sam31),LinkedView::Compare);
        assert_eq!(LinkedView::Contacts.next(SegmentationMode::EyeStudent),LinkedView::Timing);
        let mut snapshot=example_snapshot();
        snapshot.method=SegmentationMode::Sam31;
        let mut ui=Workspace::default();
        ui.scope=Scope::Linked;
        ui.linked=LinkedView::TweakedContacts;
        let mut pixels=vec![0;1200*850];
        render_snapshot(&mut ui,&mut pixels,1200,850,&snapshot);
        assert_eq!(ui.linked,LinkedView::TweakedContacts);
        snapshot.method=SegmentationMode::EyeStudent;
        render_snapshot(&mut ui,&mut pixels,1200,850,&snapshot);
        assert_eq!(ui.linked,LinkedView::Contacts);
    }

    #[test]
    fn tweaked_roi_card_preserves_source_size_across_a_newer_crop_resize() {
        let mut frame=crate::tests::control_eye_frame(12);
        frame.segmentation_mode=SegmentationMode::Sam31;
        frame.width=420;frame.height=280;
        frame.sam31_proposal_masks=Some(Arc::new(sam31_outer::ProposalMasks {
            source_width:384,source_height:256,..Default::default()}));
        assert_eq!(roi_card_source_size(&frame,RoiOverlayMode::SamTweakedContactGeometry),(384,256));
        assert_eq!(roi_card_source_size(&frame,RoiOverlayMode::Clean),(420,280));
        frame.segmentation_mode=SegmentationMode::EyeStudent;
        assert_eq!(roi_card_source_size(&frame,RoiOverlayMode::SamTweakedContactGeometry),(420,280));
    }

    #[test]
    fn absent_disabled_and_object_views_render_without_eye_evidence() {
        let mut snapshot = example_snapshot();
        snapshot.eyes = [None, None];
        snapshot.second = false;
        snapshot.backdrop = None;
        let mut ui = Workspace::default();
        let mut p = vec![0; 640 * 480];
        for scope in [Scope::Roi, Scope::Linked, Scope::Global] {
            ui.scope = scope;
            render_snapshot(&mut ui, &mut p, 640, 480, &snapshot);
        }
        snapshot.backdrop = example_snapshot().backdrop;
        snapshot.candidate = Some(sam31_outer::SceneCandidate {
            bounds: [0.2, 0.2, 0.8, 0.8],
            score: 0.8,
            mask: Arc::new(vec![1; 40 * 30]),
            mask_size: (40, 30),
        });
        ui.global = GlobalView::Objects;
        render_snapshot(&mut ui, &mut p, 640, 480, &snapshot);
        assert!(snapshot.eyes.iter().all(Option::is_none));
    }
    #[test]
    fn layout_regions_stay_bounded_and_do_not_overlap() {
        for (w, h) in [
            (1, 1),
            (320, 240),
            (640, 480),
            (900, 600),
            (1200, 850),
            (1920, 1080),
            (800, 1200),
        ] {
            let l = Layout::new(w, h);
            let rs = [l.nav, l.detector, l.toolbar, l.canvas, l.inspector, l.footer];
            for r in rs {
                assert!(r.x + r.w <= w && r.y + r.h <= h, "{w}x{h} {r:?}");
            }
            for (i, a) in rs.iter().enumerate() {
                for b in &rs[i + 1..] {
                    assert!(
                        a.x + a.w <= b.x
                            || b.x + b.w <= a.x
                            || a.y + a.h <= b.y
                            || b.y + b.h <= a.y
                    );
                }
            }
        }
    }
    #[test]
    fn diagnostic_wrapping_preserves_words_and_long_tokens() {
        assert_eq!(
            wrapped_lines(&["ROI remembers its view".into()], 13),
            vec!["ROI remembers", "its view"]
        );
        assert_eq!(
            wrapped_lines(&["abcdefgh".into()], 3),
            vec!["abc", "def", "gh"]
        );
    }
    #[test]
    fn hotkeys_follow_scope_and_prompt_editor() {
        let ui = Workspace {
            scope: Scope::Global,
            global: GlobalView::Objects,
            ..Workspace::default()
        };
        let mut map = keyboard_peeper::buttercup_map(false, false);
        configure_hotkeys(&mut map, &ui, false, true);
        assert!(!map.bindings.iter().find(|b| b.key == "V").unwrap().enabled);
        assert!(!map.bindings.iter().find(|b| b.key == "M").unwrap().enabled);
        assert_eq!(
            map.bindings
                .iter()
                .find(|b| b.key == "Space")
                .unwrap()
                .label,
            "Stop object search"
        );
        let mut map = keyboard_peeper::buttercup_map(false, false);
        configure_hotkeys(&mut map, &ui, true, false);
        assert!(map
            .bindings
            .iter()
            .filter(|b| b.enabled)
            .all(|b| matches!(b.key, "Enter" | "Esc")));
    }
    #[test]
    fn roi_views_are_independent_and_scopes_remember_selection() {
        let mut ui = Workspace::default();
        ui.preview_edit_scope = PreviewEditScope::SelectedPreview;
        let other = ui.roi_view(1).overlay;
        ui.cycle_view(SegmentationMode::Sam31);
        assert_eq!(ui.roi_view(1).overlay, other);
        let selected = ui.roi_view(0).overlay;
        ui.scope = Scope::Linked;
        ui.cycle_view(SegmentationMode::Sam31);
        ui.scope = Scope::Global;
        ui.cycle_view(SegmentationMode::Sam31);
        assert!(ui.object_view());
        assert_eq!(ui.roi_view(0).overlay, selected);
    }

    #[test]
    fn global_preview_defaults_inherit_per_setting_and_reset_is_local() {
        let mut ui = Workspace::default();
        assert_eq!(ui.preview_edit_scope, PreviewEditScope::GlobalDefaults);
        ui.cycle_view(SegmentationMode::Sam31);
        assert_eq!(ui.roi_view(0), ui.roi_view(1));
        let global_overlay = ui.preview_defaults.overlay;
        ui.preview_edit_scope = PreviewEditScope::SelectedPreview;
        ui.cycle_view(SegmentationMode::Sam31);
        let overridden_overlay = ui.roi_view(0).overlay;
        assert_ne!(overridden_overlay, global_overlay);
        assert!(ui.preview_overrides[0].pixels.is_none());
        ui.preview_edit_scope = PreviewEditScope::GlobalDefaults;
        ui.set_pixels(ViewMode::BlueFilter);
        assert_eq!(ui.roi_view(0).pixels, ViewMode::BlueFilter);
        assert_eq!(ui.roi_view(1).pixels, ViewMode::BlueFilter);
        assert_eq!(ui.roi_view(0).overlay, overridden_overlay);

        ui.selected = 1;
        ui.preview_edit_scope = PreviewEditScope::SelectedPreview;
        ui.set_pixels(ViewMode::RawColor);
        assert!(ui.preview_overrides[1].overlay.is_none());
        ui.preview_edit_scope = PreviewEditScope::GlobalDefaults;
        ui.set_overlay(RoiOverlayMode::Clean);
        assert_eq!(ui.roi_view(1).overlay, RoiOverlayMode::Clean);
        assert_eq!(ui.roi_view(1).pixels, ViewMode::RawColor);
        ui.selected = 0;
        ui.reset_selected_preview();
        assert_eq!(ui.roi_view(0), ui.preview_defaults);
        assert_eq!(ui.roi_view(1).pixels, ViewMode::RawColor);
    }

    #[test]
    fn linked_student_views_offer_sparse_layers_without_changing_preview_defaults() {
        let mut snapshot = example_snapshot();
        snapshot.method = SegmentationMode::EyeStudent;
        for frame in snapshot.eyes.iter_mut().flatten() {
            frame.segmentation_mode = SegmentationMode::EyeStudent;
        }
        let mut ui = Workspace { scope: Scope::Linked, ..Default::default() };
        let original = ui.preview_defaults;
        let mut pixels = vec![0; 1200 * 850];
        let views = LinkedView::available(snapshot.method);
        assert!(views.contains(&LinkedView::StudentEllipseOnly));
        assert!(views.contains(&LinkedView::StudentMaskOutline));
        assert!(views.contains(&LinkedView::StudentPupilOnly));
        for index in 0..views.len() {
            assert_eq!(ui.linked.position_for(snapshot.method), (index + 1, views.len()));
            render_snapshot(&mut ui, &mut pixels, 1200, 850, &snapshot);
            ui.cycle_view(snapshot.method);
        }
        assert_eq!(ui.linked, LinkedView::Compare);
        assert_eq!(ui.preview_defaults, original);
        assert_eq!(LinkedView::available(SegmentationMode::Sam31),
            &[LinkedView::Compare, LinkedView::Timing, LinkedView::Contacts, LinkedView::TweakedContacts]);
    }

    #[test]
    fn preview_scope_hotkeys_are_explicit_and_do_not_require_function_keys() {
        let mut ui = Workspace::default();
        let mut map = keyboard_peeper::buttercup_map(false, false);
        configure_hotkeys(&mut map, &ui, false, false);
        assert!(map.bindings.iter().any(|b| b.key == "Tab" && b.modifiers == 2 && b.enabled
            && b.label == "Edit selected preview"));
        assert!(map.bindings.iter().any(|b| b.key == "Backspace" && b.modifiers == 0 && b.enabled));
        assert_eq!(map.bindings.iter().find(|b| b.key == "G").unwrap().label, "Global gaze detector");
        ui.preview_edit_scope = PreviewEditScope::SelectedPreview;
        let mut map = keyboard_peeper::buttercup_map(false, false);
        configure_hotkeys(&mut map, &ui, false, false);
        assert_eq!(map.bindings.iter().find(|b| b.key == "F").unwrap().label, "Override preview overlay");
        let mut map = keyboard_peeper::buttercup_map(false, false);
        configure_hotkeys(&mut map, &ui, true, false);
        assert!(map.bindings.iter().filter(|b| b.enabled).all(|b| matches!(b.key, "Enter" | "Esc")));
    }

    #[test]
    fn compact_selection_actions_leave_room_for_scrollable_status() {
        let (w, h) = (640, 480);
        let panel = Layout::new(w, h).inspector.inset(8);
        let area = Rect { y: panel.y + 36, h: panel.h.saturating_sub(36), ..panel };
        let mut pixels = vec![0; w * h];
        let mut ui = Workspace::default();
        let remaining = selection_controls(&mut Canvas { pixels: &mut pixels, w, h }, &mut ui, area, false);
        assert!(remaining.h >= 20, "at least one status row must remain");
        assert_eq!(ui.hits.len(), 3);
        for (rect, _) in &ui.hits {
            assert_eq!(rect.y, area.y);
            assert!(rect.y + rect.h <= remaining.y);
            assert!(rect.x >= area.x && rect.x + rect.w <= area.x + area.w);
        }
    }
}
