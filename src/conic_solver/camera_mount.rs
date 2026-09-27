/// Shared operating policy, not a measured pose. Native BelowEyes tracking uses
/// screen-reference initialization followed by conditional continuation. The
/// low-level supports/branch methods retain legacy sensor-Y filters for explicit
/// historical controls; physical camera placement alone does not prove them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CameraMount {
    #[default]
    Flexible,
    BelowEyes,
    AboveEyes,
}

impl CameraMount {
    /// Temporary operating prior for Rob's upright, screen-facing rig. Keep
    /// Default flexible for geometry/unit experiments; production/offline
    /// callers choose this explicitly and record the resolved setting.
    pub(crate) const OPERATING_DEFAULT: Self = Self::BelowEyes;

    pub(crate) fn offline_override(value: Option<&str>) -> Result<Self, String> {
        match value {
            None => Ok(Self::OPERATING_DEFAULT),
            Some(value) => Self::parse(value).ok_or_else(|| {
                "BUTTERCUP_OFFLINE_CAMERA_MOUNT must be below-eyes, above-eyes or flexible".into()
            }),
        }
    }

    pub(crate) fn for_offline_checks() -> Result<Self, String> {
        let value = match std::env::var("BUTTERCUP_OFFLINE_CAMERA_MOUNT") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(error.to_string()),
        };
        let mode = Self::offline_override(value.as_deref())?;
        eprintln!(
            "CAMERA_MOUNT offline={} (operator assumption, not measured pose)",
            mode.label()
        );
        mode.warn_if_not_below("offline check");
        Ok(mode)
    }

    pub(crate) fn missing_below_warning(self) -> Option<&'static str> {
        (self != Self::BelowEyes).then_some(
            "below-eyes assumption is not enabled; this differs from the current signed-gaze operating prior. Use F8 in the viewer to choose the mounting assumption. This prior is temporary, not a guarantee of correct gaze.")
    }

    pub(crate) fn warn_if_not_below(self, context: &str) {
        if let Some(message) = self.missing_below_warning() {
            eprintln!(
                "WARNING CAMERA_MOUNT {context} mode={}: {message}",
                self.label()
            );
        }
    }

    pub(crate) fn next(self) -> Self {
        match self {
            Self::Flexible => Self::BelowEyes,
            Self::BelowEyes => Self::AboveEyes,
            Self::AboveEyes => Self::Flexible,
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Flexible => "flexible",
            Self::BelowEyes => "below-eyes",
            Self::AboveEyes => "above-eyes",
        }
    }
    /// Describe the constraint actually applied by the current solver. Physical
    /// camera placement alone does not imply the sign of sensor-space normal Y.
    pub(crate) fn description(self) -> &'static str {
        match self {
            Self::Flexible => "No fixed sensor-up/down gaze constraint",
            Self::BelowEyes => {
                "Stereo: look at the top plus to orient, then look freely (conditional tracking)"
            }
            Self::AboveEyes => "Favors sensor-down gaze continuously; assumes an upright camera",
        }
    }
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "flexible" => Some(Self::Flexible),
            "below-eyes" | "below" => Some(Self::BelowEyes),
            "above-eyes" | "above" => Some(Self::AboveEyes),
            _ => None,
        }
    }
    pub(crate) fn supports(self, down: f64) -> bool {
        down.is_finite()
            && match self {
                Self::Flexible => true,
                Self::BelowEyes => down < -0.05,
                Self::AboveEyes => down > 0.05,
            }
    }
    pub(crate) fn branch(self, down: [f64; 2]) -> Option<usize> {
        if self == Self::Flexible {
            return None;
        }
        match down.map(|y| self.supports(y)) {
            [true, false] => Some(0),
            [false, true] => Some(1),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_prior_is_explicit_and_invalid_overrides_are_not_flexible_fallbacks() {
        assert_eq!(
            CameraMount::offline_override(None).unwrap(),
            CameraMount::BelowEyes
        );
        assert_eq!(
            CameraMount::offline_override(Some("flexible")).unwrap(),
            CameraMount::Flexible
        );
        assert_eq!(
            CameraMount::offline_override(Some("ABOVE")).unwrap(),
            CameraMount::AboveEyes
        );
        assert!(CameraMount::offline_override(Some("belwo")).is_err());
        assert!(CameraMount::offline_override(Some("")).is_err());
        assert!(CameraMount::BelowEyes.missing_below_warning().is_none());
        assert!(CameraMount::Flexible.missing_below_warning().is_some());
        assert!(CameraMount::AboveEyes.missing_below_warning().is_some());
    }
}

/// A source-local restriction on the SAME density used for mode selection and
/// posterior integration. These are explicit operating priors, not new evidence.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum DirectionPrior {
    #[default]
    None,
    /// An old completion with no retained source-local prior must abstain;
    /// today's reference cannot authorize an unconditioned historical solve.
    Unavailable,
    Parallax([Option<[[f64; 3]; 2]>; 2]),
    ScreenReference,
    ScreenAndContinue([Option<[f64; 3]>; 2]),
    /// Routine tracking: an eye WITHOUT a reference may earn one only under the
    /// screen half-space and the same coherent vote as orientation; an eye with
    /// a reference continues unbounded, so off-screen gaze stays allowed.
    AcquireAndContinue([Option<[f64; 3]>; 2]),
    Continue([Option<[f64; 3]>; 2]),
}
impl DirectionPrior {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::None => "unconditioned",
            Self::Unavailable => "source-direction-prior-unavailable",
            Self::Parallax(_) => "parallax-latched-conditional",
            Self::ScreenReference => "screen-reference-conditional",
            Self::ScreenAndContinue(_) => "screen-reference-and-continuation-conditional",
            Self::AcquireAndContinue(_) => "screen-acquisition-then-continuation-conditional",
            Self::Continue(_) => "temporal-continuation-conditional",
        }
    }
    pub(crate) fn supports(self, eye: usize, center: [f64; 3], normal: [f64; 3]) -> bool {
        if !center.into_iter().chain(normal).all(f64::is_finite) {
            return false;
        }
        match self {
            Self::None => true,
            Self::Unavailable => false,
            Self::Parallax(pairs) => pairs[eye].is_none_or(|[chosen, other]| {
                (0..3)
                    .map(|i| normal[i] * (chosen[i] - other[i]))
                    .sum::<f64>()
                    >= 0.0
            }),
            Self::ScreenReference => {
                // +X sensor right, +Y sensor down, +Z toward camera.
                // Normal of the plane through camera, eye and camera-horizontal.
                // Allow 15 degrees alignment uncertainty + 5 degrees model error.
                // This is CONDITIONAL on that projected-axis error allowance;
                // physical roll alone does not certify it for arbitrary yaw.
                let length = center[1].hypot(center[2]);
                length > 1e-9
                    && (center[2] * normal[1] - center[1] * normal[2]) / length
                        >= -20.0f64.to_radians().sin()
            }
            Self::Continue(normals) => normals.get(eye).copied().flatten().is_none_or(|previous| {
                normal
                    .into_iter()
                    .zip(previous)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                    >= 45.0f64.to_radians().cos()
            }),
            // A presented on-screen stimulus bounds every normal, not just the
            // first vote. Otherwise a chain of 45-degree continuation steps
            // can walk a surviving reference onto the mirror branch.
            Self::AcquireAndContinue(normals) => {
                if normals[eye].is_some() {
                    Self::Continue(normals).supports(eye, center, normal)
                } else {
                    Self::ScreenReference.supports(eye, center, normal)
                }
            }
            Self::ScreenAndContinue(normals) => {
                Self::ScreenReference.supports(eye, center, normal)
                    && Self::Continue(normals).supports(eye, center, normal)
            }
        }
    }
}

/// Per-eye references from unique source exposures. A partner revision replaces
/// the same-time observation; it cannot cast an extra initialization vote.
#[derive(Default)]
pub(crate) struct DirectionContinuation {
    pub(crate) orientation_active: bool,
    /// A presented on-screen fixation stimulus after orientation, such as a
    /// stationary calibration target. It permits an eye whose reference expired
    /// to re-earn it under the same screen half-space and vote; unlike
    /// orientation, it never clears a surviving reference or its history.
    pub(crate) screen_fixation: bool,
    /// Outside orientation/calibration, let an eye without a reference earn one
    /// from ordinary on-screen viewing (AcquireAndContinue). Off by default so
    /// historical replays keep their unconditioned routine prior.
    pub(crate) routine_acquisition: bool,
    orientation_source_start: Option<u64>,
    history: [Vec<(u64, [f64; 3])>; 2],
    reference: [Option<(u64, [f64; 3])>; 2],
    // A delayed partner must reuse the restriction assigned to its physical
    // exposure, even after another ROI has advanced the source frontier.
    cached_priors: Vec<(u64, DirectionPrior)>,
    newest_prior_source: Option<u64>,
    contradiction_source: Option<u64>,
    contradiction_votes: usize,
}
impl DirectionContinuation {
    const MAX_GAP_NS: u64 = 1_500_000_000;
    pub(crate) fn anchor_source(&mut self, source: u64) {
        if self.orientation_active && self.orientation_source_start.is_none() {
            self.orientation_source_start = Some(source);
        }
    }
    pub(crate) fn set_orientation(&mut self, active: bool) {
        if active && !self.orientation_active {
            self.history = [Vec::new(), Vec::new()];
            self.reference = [None; 2];
            self.orientation_source_start = None;
            self.cached_priors.clear();
            self.newest_prior_source = None;
            self.contradiction_source = None;
            self.contradiction_votes = 0;
        }
        self.orientation_active = active;
    }
    pub(crate) fn set_screen_fixation(&mut self, active: bool) {
        self.screen_fixation = active;
    }
    pub(crate) fn set_routine_acquisition(&mut self, enabled: bool) {
        self.routine_acquisition = enabled;
    }
    pub(crate) fn prior(&mut self, source: u64) -> DirectionPrior {
        if let Some((_, prior)) = self.cached_priors.iter().find(|(at, _)| *at == source) {
            return *prior;
        }
        if self.newest_prior_source.is_some_and(|at| source < at) {
            return DirectionPrior::Unavailable;
        }
        self.newest_prior_source = Some(source);
        for eye in 0..2 {
            if self.reference[eye]
                .is_some_and(|(at, _)| source.saturating_sub(at) > Self::MAX_GAP_NS)
            {
                self.reference[eye] = None;
                self.history[eye].clear();
            }
        }
        let normals = self.reference.map(|r| r.map(|(_, n)| n));
        let prior = if self.orientation_active {
            let first = *self.orientation_source_start.get_or_insert(source);
            if source >= first.saturating_add(2_100_000_000) {
                DirectionPrior::ScreenAndContinue(normals)
            } else {
                DirectionPrior::None
            }
        } else if self.screen_fixation {
            // Surviving references continue unchanged; only an expired eye
            // falls back to the screen half-space and must win a fresh vote.
            DirectionPrior::ScreenAndContinue(normals)
        } else if self.routine_acquisition {
            DirectionPrior::AcquireAndContinue(normals)
        } else if normals.iter().any(Option::is_some) {
            DirectionPrior::Continue(normals)
        } else {
            DirectionPrior::None
        };
        // Bounded bookkeeping, not additional observations or longer sign life.
        self.cached_priors
            .retain(|(at, _)| source.saturating_sub(*at) <= Self::MAX_GAP_NS);
        self.cached_priors.push((source, prior));
        if self.cached_priors.len() > 64 {
            self.cached_priors.remove(0);
        }
        prior
    }
    pub(crate) fn ready(&self, eye: usize) -> bool {
        self.reference[eye].is_some()
    }
    pub(crate) fn contradiction_pending(&self) -> bool {
        self.contradiction_votes > 0
    }
    pub(crate) fn observe_contradiction(&mut self, source: u64, contradicts: bool) -> bool {
        if self.contradiction_source.is_some_and(|old| source <= old) {
            return false;
        }
        self.contradiction_source = Some(source);
        self.contradiction_votes = if contradicts {
            self.contradiction_votes + 1
        } else {
            0
        };
        if self.contradiction_votes < 3 {
            return false;
        }
        self.reference = [None; 2];
        self.history = [Vec::new(), Vec::new()];
        self.cached_priors.clear();
        self.contradiction_votes = 0;
        true
    }
    /// Keep an existing reference alive from a fresh continuation-conditioned
    /// fit that is too imprecise for calibration but still concentrated inside
    /// the cone. It cannot initialize a reference, and the periodic unrestricted
    /// audit still releases a reference after repeated strong contradictions.
    pub(crate) fn refresh(
        &mut self,
        source: u64,
        normals: [Option<[f64; 3]>; 2],
        continuing: [bool; 2],
        prior: DirectionPrior,
    ) {
        if self
            .newest_prior_source
            .is_some_and(|at| at.saturating_sub(source) > Self::MAX_GAP_NS)
        {
            return;
        }
        let referenced = match prior {
            DirectionPrior::Continue(normals)
            | DirectionPrior::ScreenAndContinue(normals)
            | DirectionPrior::AcquireAndContinue(normals) => normals,
            _ => return,
        };
        for eye in 0..2 {
            let Some(n) = normals[eye].filter(|_| continuing[eye] && referenced[eye].is_some()) else {
                continue;
            };
            if self.reference[eye].is_some_and(|(at, _)| source > at) {
                self.reference[eye] = Some((source, n));
            }
        }
    }
    pub(crate) fn observe(
        &mut self,
        source: u64,
        normals: [Option<[f64; 3]>; 2],
        supported: [bool; 2],
        prior: DirectionPrior,
    ) {
        // A late supported pair can close a gap discovered by a newer partial
        // read. It still expires by the SAME source-time bound; an old completion
        // cannot keep a reference alive indefinitely or roll a newer normal back.
        if self
            .newest_prior_source
            .is_some_and(|at| at.saturating_sub(source) > Self::MAX_GAP_NS)
        {
            return;
        }
        for eye in 0..2 {
            let Some(n) = normals[eye].filter(|_| supported[eye]) else {
                continue;
            };
            if self.reference[eye].is_some_and(|(at, _)| source < at) {
                continue;
            }
            let continued = match prior {
                DirectionPrior::Continue(normals)
                | DirectionPrior::ScreenAndContinue(normals)
                | DirectionPrior::AcquireAndContinue(normals) => {
                    normals[eye].is_some()
                }
                _ => false,
            };
            if continued {
                self.reference[eye] = Some((source, n));
                continue;
            }
            if !matches!(
                prior,
                DirectionPrior::ScreenReference
                    | DirectionPrior::ScreenAndContinue(_)
                    | DirectionPrior::AcquireAndContinue(_)
            ) {
                continue;
            }
            let h = &mut self.history[eye];
            if h.last()
                .is_some_and(|(at, _)| source.saturating_sub(*at) > Self::MAX_GAP_NS)
            {
                h.clear();
            }
            if let Some(old) = h.iter_mut().find(|(at, _)| *at == source) {
                *old = (source, n);
            } else if h.last().is_none_or(|(at, _)| source > *at) {
                h.push((source, n));
            }
            if h.len() > 12 {
                h.remove(0);
            }
            let near = |a: [f64; 3], b: [f64; 3]| {
                a.into_iter().zip(b).map(|(a, b)| a * b).sum::<f64>() >= 12.0f64.to_radians().cos()
            };
            let best = h
                .iter()
                .map(|(_, candidate)| {
                    h.iter()
                        .enumerate()
                        .filter(|(_, (_, n))| near(*candidate, *n))
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                })
                .max_by_key(Vec::len)
                .unwrap_or_default();
            let runner = h
                .iter()
                .enumerate()
                .filter(|(i, _)| !best.contains(i))
                .map(|(_, (_, candidate))| {
                    h.iter()
                        .enumerate()
                        .filter(|(i, (_, n))| !best.contains(i) && near(*candidate, *n))
                        .count()
                })
                .max()
                .unwrap_or(0);
            if best.len() >= 6
                && best.len() >= runner + 2
                && h[*best.last().unwrap()].0.saturating_sub(h[best[0]].0) >= 1_000_000_000
                && best.last().is_some_and(|&i| h[i].0 == source)
            {
                // Use the current observed normal, never a mean across modes.
                self.reference[eye] = Some((source, n));
            }
        }
    }
}

#[cfg(test)]
mod continuation_tests {
    use super::*;
    fn n(y: f64) -> [f64; 3] {
        [0.0, y, (1.0 - y * y).sqrt()]
    }
    #[test]
    fn direction_continuation_anchors_to_current_raw_not_a_delayed_inference() {
        let mut c = DirectionContinuation::default();
        c.set_orientation(true);
        c.anchor_source(10_000_000_000);
        assert!(matches!(c.prior(8_000_000_000), DirectionPrior::None));
        assert!(matches!(c.prior(12_099_999_999), DirectionPrior::None));
        assert!(matches!(
            c.prior(12_100_000_000),
            DirectionPrior::ScreenAndContinue(_)
        ));
    }
    #[test]
    fn direction_continuation_requires_three_fresh_contradictions_to_release() {
        let mut c = DirectionContinuation::default();
        let time = initialize(&mut c);
        assert!(!c.observe_contradiction(time + 1, true));
        assert!(!c.observe_contradiction(time + 1, true));
        assert!(!c.observe_contradiction(time + 2, true));
        assert!(c.ready(0));
        assert!(c.observe_contradiction(time + 3, true));
        assert!(!c.ready(0));
    }
    fn initialize(c: &mut DirectionContinuation) -> u64 {
        c.set_orientation(true);
        assert!(matches!(c.prior(0), DirectionPrior::None));
        assert!(matches!(c.prior(2_099_999_999), DirectionPrior::None));
        let mut last = 0;
        for i in 0..6 {
            last = 2_100_000_000 + i * 250_000_000;
            let p = c.prior(last);
            c.observe(last, [Some(n(-0.5)); 2], [true; 2], p);
            if i < 5 {
                assert!(!c.ready(0));
            }
        }
        assert!(c.ready(0) && c.ready(1));
        last
    }
    #[test]
    fn screen_fixation_reearns_expired_eye_without_clearing_surviving_reference() {
        let mut c = DirectionContinuation::default();
        let mut time = initialize(&mut c);
        c.set_orientation(false);
        c.set_screen_fixation(true);
        // Eye 1 keeps reporting; eye 0 drops out beyond the gap bound.
        for _ in 0..8 {
            time += 250_000_000;
            let p = c.prior(time);
            c.observe(time, [None, Some(n(-0.5))], [false, true], p);
        }
        assert!(!c.ready(0) && c.ready(1));
        let p = c.prior(time + 1);
        assert!(matches!(p, DirectionPrior::ScreenAndContinue([None, Some(_)])));
        // The screen half-space excludes the mirror branch for the expired eye only.
        let center = [0.0, -80.0, -500.0];
        assert!(p.supports(0, center, n(-0.5)) && !p.supports(0, center, n(0.8)));
        // The same six-source, one-second vote re-earns eye 0; eye 1 is untouched.
        for i in 0..6 {
            time += 250_000_000;
            let p = c.prior(time);
            c.observe(time, [Some(n(-0.5)), Some(n(-0.5))], [true; 2], p);
            assert_eq!(c.ready(0), i == 5);
            assert!(c.ready(1));
        }
        // Without a presented on-screen stimulus an expired eye stays unresolved.
        c.set_screen_fixation(false);
        assert!(matches!(c.prior(time + 1_500_000_001), DirectionPrior::None));
    }
    #[test]
    fn routine_acquisition_earns_on_screen_then_allows_off_screen_gaze() {
        let mut c = DirectionContinuation::default();
        c.set_routine_acquisition(true);
        let center = [0.0, -80.0, -500.0];
        // Without a reference, only the screen half-space is admitted.
        let p = c.prior(1);
        assert!(matches!(p, DirectionPrior::AcquireAndContinue([None, None])));
        assert!(p.supports(0, center, n(-0.5)) && !p.supports(0, center, n(0.8)));
        let mut time = 1;
        for _ in 0..6 {
            time += 250_000_000;
            let p = c.prior(time);
            c.observe(time, [Some(n(-0.5)); 2], [true; 2], p);
        }
        assert!(c.ready(0) && c.ready(1));
        // With a reference, gaze may leave the screen within the continuation cone.
        let below = n(0.6); // looking down past the camera
        assert!(!DirectionPrior::ScreenReference.supports(0, center, below));
        let continued = DirectionPrior::AcquireAndContinue([Some(n(0.4)), None]);
        assert!(continued.supports(0, center, below));
        assert!(!continued.supports(1, center, below), "an eye without a reference still acquires on screen");
        // Disabled (historical replays): no acquisition without a reference.
        let mut legacy = DirectionContinuation::default();
        assert!(matches!(legacy.prior(1), DirectionPrior::None));
    }
    #[test]
    fn on_screen_continuation_cannot_walk_onto_the_mirror_branch() {
        // Live stereo Obelisk replay 1790436833: eye center and a walked normal.
        let center = [94.96, -51.99, -290.57];
        let previous = [0.43, -0.01, 0.90];
        let mirror = [0.04, 0.62, 0.78];
        let on_screen = [0.2, -0.25, 0.95];
        let walk = DirectionPrior::Continue([None, Some(previous)]);
        assert!(walk.supports(1, center, mirror), "a 45-degree step alone admits it");
        let bounded = DirectionPrior::ScreenAndContinue([None, Some(previous)]);
        assert!(!bounded.supports(1, center, mirror));
        assert!(bounded.supports(1, center, on_screen));
    }
    #[test]
    fn imprecise_continuation_keeps_reference_alive_but_cannot_create_one() {
        let mut c = DirectionContinuation::default();
        let mut time = initialize(&mut c);
        c.set_orientation(false);
        // Three seconds of fits too imprecise for calibration support.
        for _ in 0..12 {
            time += 250_000_000;
            let p = c.prior(time);
            c.observe(time, [Some(n(-0.45)); 2], [false; 2], p);
            c.refresh(time, [Some(n(-0.45)), None], [true, false], p);
        }
        assert!(c.ready(0), "continuity evidence keeps eye 0");
        assert!(!c.ready(1), "an eye without fresh continuity still expires");
        // Once expired, continuity-only evidence never re-creates a reference.
        for _ in 0..8 {
            time += 250_000_000;
            let p = c.prior(time);
            c.refresh(time, [Some(n(-0.45)); 2], [true; 2], p);
        }
        assert!(c.ready(0) && !c.ready(1));
    }
    #[test]
    fn delayed_partner_reuses_source_prior_and_refreshes_before_gap_expiry() {
        let mut c = DirectionContinuation::default();
        let time = initialize(&mut c);
        c.set_orientation(false);
        let delayed = time + 1_100_000_000;
        let first = c.prior(delayed);
        c.observe(delayed, [Some(n(-0.5)); 2], [false; 2], first);
        let newer = c.prior(time + 1_300_000_000);
        c.observe(time + 1_300_000_000, [Some(n(-0.5)); 2], [false; 2], newer);
        let paired = c.prior(delayed);
        assert!(matches!(
            paired,
            DirectionPrior::Continue([Some(_), Some(_)])
        ));
        assert_eq!(
            paired.supports(0, [0., -100., -400.], n(0.8)),
            first.supports(0, [0., -100., -400.], n(0.8))
        );
        c.observe(delayed, [Some(n(-0.4)); 2], [true; 2], paired);
        assert!(matches!(
            c.prior(time + 1_600_000_000),
            DirectionPrior::Continue([Some(_), Some(_)])
        ));
        assert!(c.ready(0) && c.ready(1));
    }
    #[test]
    fn delayed_supported_pair_can_close_partial_read_gap_without_extending_timeout() {
        let mut c = DirectionContinuation::default();
        let time = initialize(&mut c);
        c.set_orientation(false);
        let delayed = time + 1_300_000_000;
        c.prior(delayed);
        assert!(matches!(
            c.prior(time + 1_600_000_000),
            DirectionPrior::None
        ));
        assert!(!c.ready(0));
        let prior = c.prior(delayed);
        c.observe(delayed, [Some(n(-0.4)); 2], [true; 2], prior);
        assert!(c.ready(0));
        assert!(matches!(
            c.prior(delayed + DirectionContinuation::MAX_GAP_NS + 1),
            DirectionPrior::None
        ));
        assert!(!c.ready(0));
    }
    #[test]
    fn delayed_pair_does_not_replace_newer_normal_or_survive_explicit_reset() {
        let mut c = DirectionContinuation::default();
        let time = initialize(&mut c);
        c.set_orientation(false);
        let old = time + 100_000_000;
        c.prior(old);
        let newer = old + 100_000_000;
        let p = c.prior(newer);
        c.observe(newer, [Some(n(-0.2)); 2], [true; 2], p);
        let p = c.prior(old);
        c.observe(old, [Some(n(-0.6)); 2], [true; 2], p);
        assert_eq!(c.reference[0], Some((newer, n(-0.2))));
        for i in 1..=3 {
            c.observe_contradiction(newer + i, true);
        }
        let p = c.prior(old);
        assert!(matches!(p, DirectionPrior::Unavailable));
        assert!(!p.supports(0, [0., -100., -400.], n(-0.6)));
        c.observe(old, [Some(n(-0.6)); 2], [true; 2], p);
        assert!(!c.ready(0));
        c.set_orientation(true);
        assert!(c.cached_priors.is_empty());
        assert!(!c.ready(0));
    }
    #[test]
    fn direction_continuation_initializes_then_allows_offscreen_gaze_without_screen_clamp() {
        let mut c = DirectionContinuation::default();
        let mut time = initialize(&mut c);
        c.set_orientation(false);
        for y in [-0.3, -0.1, 0.1, 0.3, 0.5] {
            time += 100_000_000;
            let p = c.prior(time);
            assert!(matches!(p, DirectionPrior::Continue(_)));
            assert!(p.supports(0, [0., -100., -400.], n(y)));
            c.observe(time, [Some(n(y)); 2], [true; 2], p);
        }
        assert!(
            !CameraMount::BelowEyes.supports(0.5),
            "crossed the legacy sensor-up gate"
        );
        let p = c.prior(time + 100_000_000);
        assert!(
            !p.supports(0, [0., -100., -400.], n(-0.8)),
            "a large unsupported switch is not continuity"
        );
    }
    #[test]
    fn direction_continuation_duplicates_bursts_and_unusable_signals_cannot_initialize() {
        let mut c = DirectionContinuation::default();
        c.set_orientation(true);
        c.prior(0);
        for i in 0..20 {
            let time = 2_100_000_000 + (i % 2) * 10_000_000;
            let p = c.prior(time);
            c.observe(time, [Some(n(-0.5)); 2], [true; 2], p);
        }
        assert!(!c.ready(0));
        for i in 1..10 {
            let time = 3_000_000_000 + i * 200_000_000;
            let p = c.prior(time);
            c.observe(time, [Some(n(-0.5)); 2], [false; 2], p);
        }
        assert!(!c.ready(0));
    }
    #[test]
    fn direction_continuation_gap_requires_a_new_reference_and_partner_uses_frozen_prior() {
        let mut c = DirectionContinuation::default();
        let time = initialize(&mut c);
        // Completion of source six must not turn that same source into its own prior.
        assert!(matches!(
            c.prior(time),
            DirectionPrior::ScreenAndContinue([None, None])
        ));
        c.set_orientation(false);
        assert!(matches!(
            c.prior(time + 1_500_000_001),
            DirectionPrior::None
        ));
        assert!(!c.ready(0));
        c.set_orientation(true);
        assert!(matches!(
            c.prior(time + 2_000_000_000),
            DirectionPrior::None
        ));
    }
    #[test]
    fn direction_continuation_competing_fixations_do_not_initialize_by_elapsed_time() {
        let mut c = DirectionContinuation::default();
        c.set_orientation(true);
        c.prior(0);
        for i in 0..12 {
            let time = 2_100_000_000 + i * 200_000_000;
            let p = c.prior(time);
            let sample = n(if i % 2 == 0 { -0.5 } else { 0.5 });
            c.observe(time, [Some(sample); 2], [true; 2], p);
        }
        assert!(!c.ready(0));
    }
    #[test]
    fn direction_prior_uses_eye_camera_line_and_near_coincident_directions_remain_allowed() {
        let center = [0., -200., -600.];
        assert!(
            DirectionPrior::ScreenReference.supports(0, center, n(0.1)),
            "positive sensor Y can still point above the eye-camera line"
        );
        let p = DirectionPrior::Continue([Some(n(0.)); 2]);
        assert!(p.supports(0, center, n(0.01)) && p.supports(0, center, n(-0.01)));
    }
}
