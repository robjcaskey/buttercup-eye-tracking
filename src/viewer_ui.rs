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
            Self::Roi => "ROI",
            Self::Linked => "LINKED ROIS",
            Self::Global => "GLOBAL",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum LinkedView {
    #[default]
    Compare,
    Timing,
    Contacts,
}
impl LinkedView {
    fn next(self) -> Self {
        match self {
            Self::Compare => Self::Timing,
            Self::Timing => Self::Contacts,
            Self::Contacts => Self::Compare,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Compare => "COMPARE",
            Self::Timing => "SOURCE TIMING",
            Self::Contacts => "CONTACT GEOMETRY",
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
#[derive(Clone, Copy, Debug)]
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
pub(super) struct Workspace {
    pub scope: Scope,
    pub selected: usize,
    pub rois: [RoiView; 2],
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
            rois: [RoiView::default(); 2],
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
    pub fn cycle_view(&mut self, method: SegmentationMode) {
        match self.scope {
            Scope::Roi => {
                self.rois[self.selected].overlay =
                    self.rois[self.selected].overlay.cycled_for(method)
            }
            Scope::Linked => self.linked = self.linked.next(),
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
                self.rois[self.selected]
                    .overlay
                    .normalized_for(method)
                    .label()
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
    pub toolbar: Rect,
    pub canvas: Rect,
    pub inspector: Rect,
    pub footer: Rect,
}
impl Layout {
    pub fn new(w: usize, h: usize) -> Self {
        let nav_h = 40.min(h / 5);
        let toolbar_h = 38.min(h / 5);
        let footer_h = 24.min(h / 8);
        let y = nav_h + toolbar_h;
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
            toolbar: Rect {
                x: 0,
                y: nav_h,
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
    Search,
    FocusReference,
    SaveMonitor,
    AccuracyCheck,
}
pub(super) fn apply(app: &mut App, action: Action) {
    app.ui.rois[app.ui.selected] = RoiView {
        pixels: app.mode,
        overlay: app.roi_overlay_mode,
    };
    match action {
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
        Action::Search => {
            if let Ok(mut s) = app.shared.lock() {
                if s.segmentation_mode == SegmentationMode::Sam31 && app.virtual_mouse.is_none() {
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
                    s.sam31_object_inspection = !s.sam31_object_inspection;
                    s.sam31_scene_candidate = None;
                }
            }
        }
    }
    app.ui.scroll = 0;
    app.mode = app.ui.rois[app.ui.selected].pixels;
    app.roi_overlay_mode = app.ui.rois[app.ui.selected].overlay;
    if let Ok(mut s) = app.shared.lock() {
        s.sam31_semantic_prompt = (s.segmentation_mode == SegmentationMode::Sam31).then_some(0);
    }
}
pub(super) fn cycle_pixels(app: &mut App) {
    match app.ui.scope {
        Scope::Roi => {
            app.mode = app.mode.cycled();
            app.ui.rois[app.ui.selected].pixels = app.mode;
        }
        Scope::Linked => {
            let next = app.ui.rois[app.ui.selected].pixels.cycled();
            for view in &mut app.ui.rois {
                view.pixels = next;
            }
            app.mode = next;
        }
        Scope::Global => {}
    }
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
    app.ui.rois[app.ui.selected] = RoiView {
        pixels: app.mode,
        overlay: app.roi_overlay_mode,
    };
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
    let body = c.label(r, &heading);
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
    let w = frame.width + 16;
    let h = frame.height + 36;
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
    let mut image_pixels = Vec::with_capacity(frame.width * frame.height);
    for y in 28..28 + frame.height {
        image_pixels.extend_from_slice(&pixels[y * w + 8..y * w + 8 + frame.width]);
    }
    c.image(image, &image_pixels, frame.width, frame.height);
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
            "roi_views":app.ui.rois.map(|r|r.overlay.label()),"roi_pixels":app.ui.rois.map(|r|annotated_view_mode_name(r.pixels)),
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
            "V" => binding.enabled = ui.scope != Scope::Global,
            "Space" => {
                binding.enabled = ui.object_view() || search_running;
                binding.label = if search_running {
                    "Stop object search"
                } else {
                    "Start object search"
                };
            }
            "M" => binding.enabled = !search_running,
            "1" => binding.label = "Select left ROI",
            "2" => binding.label = "Select right ROI",
            _ => {}
        }
    }
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
    let title = ui.title(method);
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
                ui.rois[i],
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
                let mut view = ui.rois[i];
                if ui.linked == LinkedView::Contacts {
                    view.overlay = RoiOverlayMode::SamDeflattenedVirtualContact;
                }
                if ui.linked == LinkedView::Timing {
                    view.overlay = RoiOverlayMode::Clean;
                }
                roi_card(
                    &mut c,
                    cards[i],
                    eyes[i].as_ref(),
                    i,
                    view,
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
            let mut rows = vec!["LINKED PRESENTATION / JOINT STEREO SOLVER NOT IMPLEMENTED".into()];
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
        for (label,action,active) in [
            ("SAVE MONITOR LOCATION",Action::SaveMonitor,monitor_unsaved),
            ("\\ ACCURACY CHECK - 20 TARGETS",Action::AccuracyCheck,false),
        ] {
            button(&mut c,ui,Rect {h:28.min(text_area.h),..text_area},label,active,action);
            text_area.y+=32.min(text_area.h);text_area.h=text_area.h.saturating_sub(32);
        }
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
                rows.push(format!("SELECTED: {}", eye_name(ui.selected)));
                rows.push("1 left / 2 right".into());
                rows.push(format!(
                    "V {}",
                    annotated_view_mode_name(ui.rois[ui.selected].pixels)
                ));
                rows.push(format!(
                    "J gaze overlays {}",
                    if laser { "ON" } else { "OFF" }
                ));
                rows.push(format!("M CALIBRATE {}",eye_name(focus_eye)));
                if ui.selected!=focus_eye {rows.push("F2 USE SELECTED ROI FOR AF + CALIBRATION".into());}
            }
            if ui.object_view() {
                rows.push(format!("PROMPT: {}", editor.as_deref().unwrap_or(&prompt)));
                rows.push(prompt_status);
                rows.push("Enter edit object prompt".into());
                rows.push("Object crops are not eye evidence.".into());
                rows.push("Global-image crop, not native ROI.".into());
            } else if method == SegmentationMode::Sam31 && ui.scope==Scope::Roi {
                rows.push(format!("IRIS PROMPT: {prompt}"));
                rows.push("Enter edit / Esc cancel".into());
            }
            if let Some(status) = recovery {
                rows.push(status);
            }
        }
        Panel::Analysis => {
            rows.push("SHARED ANALYSIS / BOTH EYES".into());
            rows.push(method.method_control_label());
            rows.push("Y rough-center source".into());
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
        "TAB SCOPE   F VIEW   , INSPECTOR   PGUP/PGDN SCROLL   ESC CANCEL EDIT / Q QUIT",
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
            let rs = [l.nav, l.toolbar, l.canvas, l.inspector, l.footer];
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
        let other = ui.rois[1].overlay;
        ui.cycle_view(SegmentationMode::Sam31);
        assert_eq!(ui.rois[1].overlay, other);
        let selected = ui.rois[0].overlay;
        ui.scope = Scope::Linked;
        ui.cycle_view(SegmentationMode::Sam31);
        ui.scope = Scope::Global;
        ui.cycle_view(SegmentationMode::Sam31);
        assert!(ui.object_view());
        assert_eq!(ui.rois[0].overlay, selected);
    }
}
