//! Soft look-to-focus: every gaze sample is weighed as evidence for each
//! visible window instead of hard hit-testing one point.
//!
//! P(sample | looking at window i) = ∫ N(sample; x, σ²) w_i(x) dx over the
//! window's visible area, where w_i is where people actually look inside that
//! kind of window. A robust outlier term bounds any one sample's influence.
//! Evidence decays over a short horizon (a little averaging), and focus moves
//! only when another window's posterior odds over the focused one stay above
//! the threshold for the same dwell the hard rule uses.
//!
//! Legacy allowance: terminals are read and typed at the left and the bottom
//! (prompt, cursor); browsers are read from the left. A gaze landing on the far
//! right or top edge of such a window, next to another window, is therefore
//! weaker evidence for it than a plain hit test suggests. A semantic
//! focus-shaping layer outside Buttercup can replace these priors later.
use super::{Rect, Target, MAX_GAP};
use crate::mouse_output::Source;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Measured gaze error: the ground-truth stimulus session's 1.19° median
/// fixed-target error is σ·√(2 ln 2) for an isotropic Gaussian; at the
/// calibrated 23.23 in display and 36.57 in distance that is σ ≈ 0.029 of the
/// display width.
pub(super) const GAZE_SIGMA_WIDTH_FRACTION: f64 = 0.029;
/// Fraction of accepted samples that are unrelated to the fixation (saccade
/// transients, mismatched solves): the p90 error is about twice the median in
/// every truth replay, so a tenth of samples are treated as uninformative.
const OUTLIER_FRACTION: f64 = 0.1;
/// Required posterior odds (95%) of the candidate over the focused window.
const SWITCH_ODDS: f64 = 19.0;
/// Evidence half-life. There is no separate dwell: focus moves as soon as the
/// posterior odds favor another window, so a clear look acts on its first
/// sample (like pointer hover), while edge samples must accumulate and a
/// stray sample is absorbed by the focused window's recent evidence.
pub(super) const HORIZON: Duration = Duration::from_millis(125);
/// Integration grid step in logical pixels.
const CELL: f64 = 8.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowClass {
    Terminal,
    Browser,
    Other,
}

impl WindowClass {
    pub(super) fn from_app_id(app_id: &str) -> Self {
        let id = app_id.to_ascii_lowercase();
        const TERMINALS: &[&str] = &["alacritty", "foot", "footclient", "kitty", "org.wezfurlong.wezterm",
            "wezterm", "com.mitchellh.ghostty", "ghostty", "gnome-terminal-server", "org.gnome.console",
            "org.gnome.ptyxis", "konsole", "xterm", "urxvt", "st", "terminator", "tilix"];
        const BROWSERS: &[&str] = &["firefox", "firefox-esr", "librewolf", "chromium", "chromium-browser",
            "google-chrome", "brave-browser", "org.qutebrowser.qutebrowser", "epiphany", "org.gnome.epiphany"];
        if TERMINALS.contains(&id.as_str()) { WindowClass::Terminal }
        else if BROWSERS.contains(&id.as_str()) { WindowClass::Browser }
        else { WindowClass::Other }
    }

    /// Linear ramps (left-to-right, top-to-bottom) of the in-window look
    /// density: the left edge is nine times as likely as the right for
    /// terminals and browsers; the bottom nine times the top for terminals.
    fn ramps(self) -> (f64, f64) {
        match self {
            WindowClass::Terminal => (0.8, 0.8),
            WindowClass::Browser => (0.8, 0.0),
            WindowClass::Other => (0.0, 0.0),
        }
    }

    /// Density at normalized window coordinates; integrates to 1 over [0,1]².
    fn density(self, u: f64, v: f64) -> f64 {
        let (left, bottom) = self.ramps();
        (1.0 + left * (1.0 - 2.0 * u)) * (1.0 + bottom * (2.0 * v - 1.0))
    }

    pub(super) fn label(self) -> &'static str {
        match self { WindowClass::Terminal => "terminal", WindowClass::Browser => "browser", WindowClass::Other => "other" }
    }
}

#[derive(Clone, Debug)]
pub(super) struct SceneWindow {
    pub target: Target,
    pub floating: bool,
    pub class: WindowClass,
}

#[derive(Clone, Debug)]
pub(super) struct Scene {
    pub workspace: u64,
    pub focused: u64,
    pub output: Rect,
    pub windows: Vec<SceneWindow>,
}

impl Scene {
    /// Topmost window at a point; overlapping floating windows have
    /// ambiguous stacking in GET_TREE, so such points belong to no window.
    fn window_at(&self, p: (f64, f64)) -> Option<usize> {
        let mut floating = self.windows.iter().enumerate().filter(|(_, w)| w.floating && w.target.rect.contains(p, 0.0));
        if let Some((i, _)) = floating.next() {
            return floating.next().is_none().then_some(i);
        }
        self.windows.iter().position(|w| !w.floating && w.target.rect.contains(p, 0.0))
    }

    /// The window most likely looked at from one gaze sample (monitor fraction).
    pub(super) fn most_likely(&self, uv: (f64, f64)) -> Option<Target> {
        let p = (self.output.x + uv.0 * self.output.w, self.output.y + uv.1 * self.output.h);
        let ll = self.log_likelihoods(p, GAZE_SIGMA_WIDTH_FRACTION * self.output.w);
        let outlier = (OUTLIER_FRACTION / (self.output.w * self.output.h)).ln();
        self.windows.iter().zip(ll).filter(|(_, l)| *l > outlier + 1e-9)
            .max_by(|a, b| a.1.total_cmp(&b.1)).map(|(w, _)| w.target)
    }

    /// Per-window log-likelihood of observing gaze at `p` (logical px).
    fn log_likelihoods(&self, p: (f64, f64), sigma: f64) -> Vec<f64> {
        let mut mass = vec![0.0; self.windows.len()];
        let reach = 4.0 * sigma;
        let (x0, x1) = ((p.0 - reach).max(self.output.x), (p.0 + reach).min(self.output.x + self.output.w));
        let (y0, y1) = ((p.1 - reach).max(self.output.y), (p.1 + reach).min(self.output.y + self.output.h));
        let norm = CELL * CELL / (2.0 * std::f64::consts::PI * sigma * sigma);
        let mut y = y0 + CELL / 2.0;
        while y < y1 {
            let mut x = x0 + CELL / 2.0;
            while x < x1 {
                if let Some(i) = self.window_at((x, y)) {
                    let r = self.windows[i].target.rect;
                    let d2 = (x - p.0).powi(2) + (y - p.1).powi(2);
                    let density = self.windows[i].class.density((x - r.x) / r.w, (y - r.y) / r.h) / (r.w * r.h);
                    mass[i] += norm * (-d2 / (2.0 * sigma * sigma)).exp() * density;
                }
                x += CELL;
            }
            y += CELL;
        }
        // Uniform outlier density over the output.
        let outlier = OUTLIER_FRACTION / (self.output.w * self.output.h);
        mass.into_iter().map(|m| ((1.0 - OUTLIER_FRACTION) * m + outlier).ln()).collect()
    }
}

#[derive(Default)]
pub(super) struct SoftFocus {
    context: Option<(u64, u64)>,
    /// Eye and calibration authority; a sign-epoch change alone keeps the
    /// evidence (a mis-signed sample is just an outlier here).
    basis: Option<(usize, u64)>,
    evidence: HashMap<u64, f64>,
    last: Option<Instant>,
    /// Candidate currently beating the focused window by the switch odds.
    pub(super) pending: Option<(Target, Instant, u32, u64)>,
}

impl SoftFocus {
    pub(super) fn observe(&mut self, now: Instant, source: Source, uv: (f64, f64), scene: &Scene) -> Option<Target> {
        let basis = (source.eye, source.authority);
        let context = (scene.workspace, scene.focused);
        if self.basis != Some(basis) || self.context != Some(context) {
            *self = SoftFocus { basis: Some(basis), context: Some(context), ..Default::default() };
        }
        // Dwell continuity tolerates the same gap as the hard rule.
        if self.last.is_some_and(|last| now.saturating_duration_since(last) > MAX_GAP) {
            self.pending = None;
        }
        let decay = self.last.map_or(1.0, |last| {
            0.5f64.powf(now.saturating_duration_since(last).as_secs_f64() / HORIZON.as_secs_f64())
        });
        self.last = Some(now);
        let p = (scene.output.x + uv.0 * scene.output.w, scene.output.y + uv.1 * scene.output.h);
        let sigma = GAZE_SIGMA_WIDTH_FRACTION * scene.output.w;
        let ll = scene.log_likelihoods(p, sigma);
        let mut evidence = HashMap::with_capacity(scene.windows.len());
        for (w, l) in scene.windows.iter().zip(ll) {
            let previous = self.evidence.get(&w.target.id).copied().unwrap_or(0.0);
            evidence.insert(w.target.id, previous * decay + l);
        }
        self.evidence = evidence;
        let focused = self.evidence.get(&scene.focused).copied();
        let best = scene.windows.iter()
            .filter(|w| w.target.id != scene.focused)
            .filter_map(|w| self.evidence.get(&w.target.id).map(|e| (w.target, *e)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        // Compare against the focused window, or the runner-up when the
        // focused window is not on this output/workspace.
        let rival = focused.or_else(|| {
            let best_id = best.map(|b| b.0.id);
            scene.windows.iter().filter(|w| Some(w.target.id) != best_id)
                .filter_map(|w| self.evidence.get(&w.target.id).copied()).max_by(f64::total_cmp)
        });
        let margin = |e: f64| rival.map_or(f64::INFINITY, |r| e - r);
        // The dwell clock starts once the new window leads at all.
        let Some((target, e)) = best.filter(|(_, e)| margin(*e) > 0.0) else {
            self.pending = None;
            return None;
        };
        let (first, count, first_source_ns) = match self.pending {
            Some((previous, first, count, ns)) if previous.id == target.id => (first, count + 1, ns),
            _ => (now, 1, source.timestamp_ns),
        };
        self.pending = Some((target, first, count, first_source_ns));
        (margin(e) >= SWITCH_ODDS.ln()).then_some(target)
    }

    pub(super) fn odds(&self, scene_focused: u64) -> Option<f64> {
        let (target, ..) = self.pending?;
        let e = self.evidence.get(&target.id)?;
        Some((e - self.evidence.get(&scene_focused).copied().unwrap_or(0.0)).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(t: u64) -> Source {
        Source { eye: 0, authority: 1, sign_epoch: 1, timestamp_ns: t * 100_000_000 }
    }
    /// Five side-by-side tiles like the live desktop: 2560x1440, 512 px wide.
    fn tiles(class: WindowClass, focused: u64) -> Scene {
        Scene {
            workspace: 1,
            focused,
            output: Rect { x: 0.0, y: 0.0, w: 2560.0, h: 1440.0 },
            windows: (0..5).map(|i| SceneWindow {
                target: Target { id: i + 1, rect: Rect { x: i as f64 * 512.0, y: 0.0, w: 512.0, h: 1440.0 } },
                floating: false,
                class,
            }).collect(),
        }
    }
    /// Feed `n` samples 100 ms apart at logical point `p`; first focus result.
    fn look(f: &mut SoftFocus, scene: &Scene, p: (f64, f64), n: u64) -> Option<u64> {
        let start = Instant::now();
        (1..=n).find_map(|t| f.observe(start + Duration::from_millis(100 * t), source(t),
            (p.0 / scene.output.w, p.1 / scene.output.h), scene).map(|t| t.id))
    }

    #[test]
    fn a_clear_look_switches_at_once_and_a_fixated_focus_needs_one_more_sample() {
        let scene = tiles(WindowClass::Terminal, 3);
        let uv = |x: f64| (x / 2560.0, 700.0 / 1440.0);
        // A clear look with no competing evidence acts at once, like hover.
        let mut f = SoftFocus::default();
        assert_eq!(look(&mut f, &scene, (256.0 + 512.0, 700.0), 1), Some(2));
        // After fixating the focused tile, a single stray sample is absorbed
        // and a deliberate look switches on its second sample.
        let start = Instant::now();
        let mut f = SoftFocus::default();
        for t in 1..=5 { assert!(f.observe(start + Duration::from_millis(100 * t), source(t), uv(1280.0), &scene).is_none()); }
        let at = (6..=12).find(|&t| f.observe(start + Duration::from_millis(100 * t), source(t), uv(768.0), &scene).is_some());
        assert_eq!(at, Some(7));
    }

    #[test]
    fn terminal_right_edge_next_to_the_focused_tile_keeps_focus() {
        // Focus is on tile 3 (x 1024..1536). Gaze keeps landing 30 px inside
        // the right edge of tile 2, i.e. the left side of tile 3 read with
        // ordinary gaze error. For terminals that is evidence for tile 3.
        let p = (1024.0 - 30.0, 700.0);
        let mut f = SoftFocus::default();
        assert_eq!(look(&mut f, &tiles(WindowClass::Terminal, 3), p, 50), None);
        // Focus needs confidence in the window, not the exact point: 30 px
        // from a plain window's edge is still a coin flip at this accuracy,
        // while 100 px inside is clearly the neighbor.
        let mut f = SoftFocus::default();
        assert_eq!(look(&mut f, &tiles(WindowClass::Other, 3), p, 50), None);
        let mut f = SoftFocus::default();
        assert_eq!(look(&mut f, &tiles(WindowClass::Other, 3), (1024.0 - 100.0, 700.0), 10), Some(2));
        // Deep inside tile 2 is decisive for any class.
        let mut f = SoftFocus::default();
        assert_eq!(look(&mut f, &tiles(WindowClass::Terminal, 3), (1024.0 - 300.0, 700.0), 10), Some(2));
    }

    #[test]
    fn terminal_prior_shifts_the_effective_boundary_toward_the_left_tile() {
        let scene = tiles(WindowClass::Terminal, 1);
        let sigma = GAZE_SIGMA_WIDTH_FRACTION * 2560.0;
        let ll = |x: f64| { let l = scene.log_likelihoods((x, 700.0), sigma); l[2] - l[1] };
        // At the exact boundary between tiles 2 and 3, tile 3's left side is
        // more likely than tile 2's right side.
        assert!(ll(1024.0) > 0.5, "{}", ll(1024.0));
        let plain = tiles(WindowClass::Other, 1);
        let l = plain.log_likelihoods((1024.0, 700.0), sigma);
        assert!((l[2] - l[1]).abs() < 0.05);
    }

    #[test]
    fn one_shot_focus_picks_the_likely_window_with_reading_priors() {
        let scene = tiles(WindowClass::Terminal, 1);
        assert_eq!(scene.most_likely((768.0 / 2560.0, 0.5)).map(|t| t.id), Some(2));
        // 30 px inside tile 2's right edge: the left side of tile 3.
        assert_eq!(scene.most_likely((994.0 / 2560.0, 0.5)).map(|t| t.id), Some(3));
        let mut empty = scene.clone();
        empty.windows.clear();
        assert_eq!(empty.most_likely((0.5, 0.5)), None);
    }

    #[test]
    fn stacked_terminals_favor_the_bottom_of_the_upper_tile() {
        let mut scene = tiles(WindowClass::Terminal, 9);
        scene.windows = vec![
            SceneWindow { target: Target { id: 1, rect: Rect { x: 0.0, y: 0.0, w: 1280.0, h: 720.0 } }, floating: false, class: WindowClass::Terminal },
            SceneWindow { target: Target { id: 2, rect: Rect { x: 0.0, y: 720.0, w: 1280.0, h: 720.0 } }, floating: false, class: WindowClass::Terminal },
        ];
        let sigma = GAZE_SIGMA_WIDTH_FRACTION * 2560.0;
        let l = scene.log_likelihoods((640.0, 720.0), sigma);
        assert!(l[0] - l[1] > 0.5, "upper tile's prompt row wins at the shared edge: {l:?}");
    }

    #[test]
    fn sparse_samples_with_sign_epoch_changes_still_accumulate() {
        let scene = tiles(WindowClass::Terminal, 3);
        let mut f = SoftFocus::default();
        let start = Instant::now();
        let uv = ((512.0 + 256.0) / 2560.0, 0.5);
        // Samples every 250 ms (gaps from dropped solves), sign epoch flipping.
        let mut focused = None;
        for t in 1..=4u64 {
            let mut src = source(t * 2);
            src.sign_epoch = t;
            focused = focused.or(f.observe(start + Duration::from_millis(250 * t), src, uv, &scene));
        }
        assert_eq!(focused.map(|t| t.id), Some(2));
    }

    #[test]
    fn one_outlier_sample_cannot_switch_and_overlapping_floats_abstain() {
        let scene = tiles(WindowClass::Other, 3);
        let mut f = SoftFocus::default();
        let start = Instant::now();
        let center3 = (1280.0 / 2560.0, 0.5);
        for t in 1..=5 { assert!(f.observe(start + Duration::from_millis(100 * t), source(t), center3, &scene).is_none()); }
        assert!(f.observe(start + Duration::from_millis(600), source(6), (256.0 / 2560.0, 0.5), &scene).is_none());
        let mut scene = tiles(WindowClass::Other, 1);
        for id in [10, 11] {
            scene.windows.insert(0, SceneWindow { target: Target { id, rect: Rect { x: 1000.0, y: 400.0, w: 600.0, h: 600.0 } }, floating: true, class: WindowClass::Other });
        }
        assert_eq!(scene.window_at((1300.0, 700.0)), None);
    }
}
