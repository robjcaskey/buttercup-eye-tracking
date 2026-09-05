//! Bounded, defeasible memory of independently detected occluding regions.
//!
//! This API is an offline experiment, not a live fit or an anatomical gate.
//! Call `begin_frame` once per physical exposure, score all alternatives from
//! that snapshot, then `observe_current` once using only frame-local evidence.
//! Never feed the memory's own rejected points back as fresh bad observations.

use crate::geometry::{ellipse_axis_point, Ellipse};
use crate::roi_evidence::ExposureKey;
use std::f64::consts::{PI, TAU};

pub(crate) const REGION_BINS: usize = 48;
const MAX_POINTS: usize = 128;
const MAX_HORIZON_NS: u64 = 2_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DecayFunction {
    Exponential,
    Linear,
    FiniteHorizon,
}

impl DecayFunction {
    pub(crate) fn value(self, age_ns: u64, decay_ns: u64, horizon_ns: u64) -> f64 {
        if decay_ns == 0 || horizon_ns == 0 || age_ns >= horizon_ns {
            return 0.0;
        }
        let t = age_ns as f64 / decay_ns as f64;
        match self {
            Self::Exponential => (-t).exp(),
            Self::Linear => (1.0 - t).max(0.0),
            Self::FiniteHorizon => f64::from(age_ns < decay_ns),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RecentExclusionConfig {
    pub(crate) decay: DecayFunction,
    pub(crate) decay_ns: u64,
    pub(crate) horizon_ns: u64,
    pub(crate) maximum_penalty: f64,
    /// None means soft weights only. Some((enter, leave)) permits temporary
    /// re-exclusion with hysteresis; fresh good arcs override either state.
    pub(crate) reexclude_thresholds: Option<(f64, f64)>,
}

impl RecentExclusionConfig {
    pub(crate) fn valid(self) -> bool {
        self.decay_ns > 0
            && self.decay_ns <= self.horizon_ns
            && self.horizon_ns <= MAX_HORIZON_NS
            && self.maximum_penalty.is_finite()
            && (0.0..=0.95).contains(&self.maximum_penalty)
            && self.reexclude_thresholds.is_none_or(|(enter, leave)| {
                enter.is_finite()
                    && leave.is_finite()
                    && 0.0 < leave
                    && leave < enter
                    && enter <= 1.0
            })
    }
}

/// A linear image-scale observation independent of the candidate limbus.
/// Relative scales need a stable reference/provenance ID; absolute scales can
/// use pixels per physical unit. Candidate radius is never an admissible scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IndependentScale {
    pub(crate) pixels_per_reference_unit: f64,
    pub(crate) provenance: u64,
}

/// Scale-normalized frontal-equivalent iris disk area (SN-FEIDA).
/// Under the circular weak-perspective limbus model: PI*a*b/(b/a)/s^2.
/// This is the unobstructed outer disk, not visible mask/tissue surface area.
pub(crate) fn sn_feida(ellipse: Ellipse, scale: IndependentScale) -> Option<f64> {
    let s = scale.pixels_per_reference_unit;
    if !valid_ellipse(ellipse) || !s.is_finite() || s <= 0.0 {
        return None;
    }
    let area = PI * (ellipse.major_radius / s).powi(2);
    (area.is_finite() && area > 0.0).then_some(area)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExclusionFrame {
    pub(crate) exposure: ExposureKey,
    /// Advance when crop sampling/calibration/association changes meaning.
    /// Ordinary crop translation is handled by `sensor_origin_px`.
    pub(crate) geometry_lineage: u64,
    pub(crate) sensor_origin_px: (f64, f64),
    pub(crate) reference: Ellipse,
    pub(crate) independent_scale: Option<IndependentScale>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameDisposition {
    Fresh,
    Reset,
    ReusedOrStale,
    Invalid,
}

#[derive(Clone, Copy, Debug)]
struct Region {
    timestamp_ns: u64,
    radius: f64,
    hard_excluded: bool,
}

pub(crate) struct RecentExclusionMemory {
    config: RecentExclusionConfig,
    regions: [Option<Region>; REGION_BINS],
    previous: Option<ExclusionFrame>,
    pending: bool,
}

impl RecentExclusionMemory {
    pub(crate) fn new(config: RecentExclusionConfig) -> Option<Self> {
        config.valid().then_some(Self {
            config,
            regions: [None; REGION_BINS],
            previous: None,
            pending: false,
        })
    }

    pub(crate) fn active_regions(&self) -> usize {
        self.regions.iter().flatten().count()
    }

    pub(crate) fn begin_frame(&mut self, frame: ExclusionFrame) -> FrameDisposition {
        self.pending = false;
        if !valid_frame(frame) {
            self.regions.fill(None);
            self.previous = None;
            return FrameDisposition::Invalid;
        }
        let mut reset = self.previous.is_none();
        if let Some(previous) = self.previous {
            let same_lineage = previous.exposure.roi == frame.exposure.roi
                && previous.exposure.clock == frame.exposure.clock
                && previous.geometry_lineage == frame.geometry_lineage;
            if same_lineage
                && (frame.exposure.sequence <= previous.exposure.sequence
                    || frame.exposure.timestamp_ns <= previous.exposure.timestamp_ns)
            {
                // A cached result cannot vote, clear, decay by host time, or
                // refresh the horizon. A stale result cannot rewind state.
                return FrameDisposition::ReusedOrStale;
            }
            reset = !same_lineage
                || frame
                    .exposure
                    .timestamp_ns
                    .saturating_sub(previous.exposure.timestamp_ns)
                    >= self.config.horizon_ns
                || !compatible_geometry(previous, frame);
        }
        if reset {
            self.regions.fill(None);
        }
        self.previous = Some(frame);
        self.pending = true;
        for slot in &mut self.regions {
            if let Some(region) = slot {
                let age = frame
                    .exposure
                    .timestamp_ns
                    .saturating_sub(region.timestamp_ns);
                let penalty = self.config.maximum_penalty
                    * self
                        .config
                        .decay
                        .value(age, self.config.decay_ns, self.config.horizon_ns);
                if penalty == 0.0 {
                    *slot = None;
                } else if let Some((_, leave)) = self.config.reexclude_thresholds {
                    if penalty < leave {
                        region.hard_excluded = false;
                    }
                }
            }
        }
        if reset {
            FrameDisposition::Reset
        } else {
            FrameDisposition::Fresh
        }
    }

    /// A current well-supported arc gets weight one even inside an old veto.
    /// The caller must determine `fresh_good` without consulting this memory.
    pub(crate) fn weight(&self, point: (f64, f64), fresh_good: bool) -> f64 {
        if !self.pending || fresh_good {
            return 1.0;
        }
        let Some(frame) = self.previous else {
            return 1.0;
        };
        let Some((phase, radius)) = canonical_point(point, frame.reference) else {
            return 1.0;
        };
        let bin = phase_bin(phase);
        let mut penalty = 0.0_f64;
        for offset in [-1_isize, 0, 1] {
            let index = (bin as isize + offset).rem_euclid(REGION_BINS as isize) as usize;
            let Some(region) = self.regions[index] else {
                continue;
            };
            if (radius - region.radius).abs() > 0.20 {
                continue;
            }
            let value = self.config.maximum_penalty
                * self.config.decay.value(
                    frame
                        .exposure
                        .timestamp_ns
                        .saturating_sub(region.timestamp_ns),
                    self.config.decay_ns,
                    self.config.horizon_ns,
                );
            if region.hard_excluded && value > 0.0 {
                return 0.0;
            }
            penalty = penalty.max(value);
        }
        (1.0 - penalty).clamp(0.05, 1.0)
    }

    /// Commit at most 128 independent current-frame bad/good samples once.
    /// Multiple correlated alternatives are NOT separate observations. Good
    /// evidence wins where current bad/good alternatives conflict.
    pub(crate) fn observe_current(&mut self, bad: &[(f64, f64)], good: &[(f64, f64)]) -> bool {
        if !self.pending {
            return false;
        }
        self.pending = false;
        let frame = self.previous.expect("pending frame");
        for &point in bad.iter().take(MAX_POINTS) {
            if let Some((phase, radius)) = canonical_point(point, frame.reference) {
                self.regions[phase_bin(phase)] = Some(Region {
                    timestamp_ns: frame.exposure.timestamp_ns,
                    radius,
                    hard_excluded: self
                        .config
                        .reexclude_thresholds
                        .is_some_and(|(enter, _)| self.config.maximum_penalty >= enter),
                });
            }
        }
        for &point in good.iter().take(MAX_POINTS) {
            if let Some((phase, radius)) = canonical_point(point, frame.reference) {
                let bin = phase_bin(phase);
                for offset in [-1_isize, 0, 1] {
                    let index = (bin as isize + offset).rem_euclid(REGION_BINS as isize) as usize;
                    if self.regions[index].is_some_and(|r| (r.radius - radius).abs() <= 0.20) {
                        self.regions[index] = None;
                    }
                }
            }
        }
        true
    }
}

fn valid_ellipse(e: Ellipse) -> bool {
    e.center.0.is_finite()
        && e.center.1.is_finite()
        && e.angle.is_finite()
        && e.major_radius.is_finite()
        && e.minor_radius.is_finite()
        && e.major_radius >= e.minor_radius
        && e.minor_radius > 0.0
}

fn valid_frame(frame: ExclusionFrame) -> bool {
    valid_ellipse(frame.reference)
        && frame.reference.minor_radius / frame.reference.major_radius >= 0.30
        && frame.sensor_origin_px.0.is_finite()
        && frame.sensor_origin_px.1.is_finite()
        && frame.independent_scale.is_none_or(|s| {
            s.pixels_per_reference_unit.is_finite() && s.pixels_per_reference_unit > 0.0
        })
}

fn compatible_geometry(a: ExclusionFrame, b: ExclusionFrame) -> bool {
    let scale_ratio = match (a.independent_scale, b.independent_scale) {
        (Some(a), Some(b)) if a.provenance == b.provenance => {
            b.pixels_per_reference_unit / a.pixels_per_reference_unit
        }
        (None, None) => 1.0,
        _ => return false,
    };
    if !(0.80..=1.25).contains(&scale_ratio) {
        return false;
    }
    let radius_ratio = b.reference.major_radius / a.reference.major_radius / scale_ratio;
    let aspect_ratio = (b.reference.minor_radius / b.reference.major_radius)
        / (a.reference.minor_radius / a.reference.major_radius);
    let shape = |e: Ellipse| {
        let anisotropy = e.major_radius / e.minor_radius - 1.0;
        (
            anisotropy * (2.0 * e.angle).cos(),
            anisotropy * (2.0 * e.angle).sin(),
        )
    };
    let shape_a = shape(a.reference);
    let shape_b = shape(b.reference);
    let dx =
        b.reference.center.0 + b.sensor_origin_px.0 - a.reference.center.0 - a.sensor_origin_px.0;
    let dy =
        b.reference.center.1 + b.sensor_origin_px.1 - a.reference.center.1 - a.sensor_origin_px.1;
    (0.80..=1.25).contains(&radius_ratio)
        && (0.80..=1.25).contains(&aspect_ratio)
        && (shape_b.0 - shape_a.0).hypot(shape_b.1 - shape_a.1) <= 0.35
        && dx.hypot(dy) <= 0.35 * a.reference.major_radius
}

/// Symmetric image-axis whitening R diag(1/a,1/b) R^T. Phase is stable under
/// the arbitrary pi-axis sign and near-circular major-axis angle changes.
/// This coordinate normalization is ONLY for region correspondence, never s.
pub(crate) fn canonical_point(point: (f64, f64), e: Ellipse) -> Option<(f64, f64)> {
    if !valid_ellipse(e) || !point.0.is_finite() || !point.1.is_finite() {
        return None;
    }
    let (u, v) = ellipse_axis_point(point, e);
    let (sin, cos) = e.angle.sin_cos();
    let x = cos * u / e.major_radius - sin * v / e.minor_radius;
    let y = sin * u / e.major_radius + cos * v / e.minor_radius;
    let radius = x.hypot(y);
    (radius.is_finite() && (0.20..=1.80).contains(&radius))
        .then(|| (y.atan2(x).rem_euclid(TAU), radius))
}

fn phase_bin(phase: f64) -> usize {
    ((phase.rem_euclid(TAU) / TAU * REGION_BINS as f64) as usize).min(REGION_BINS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi_evidence::{RoiId, SourceClock};

    fn config() -> RecentExclusionConfig {
        RecentExclusionConfig {
            decay: DecayFunction::Exponential,
            decay_ns: 100_000_000,
            horizon_ns: 500_000_000,
            maximum_penalty: 0.75,
            reexclude_thresholds: None,
        }
    }
    fn frame(sequence: u64, timestamp_ns: u64) -> ExclusionFrame {
        ExclusionFrame {
            exposure: ExposureKey {
                roi: RoiId(1),
                clock: SourceClock {
                    domain: 1,
                    epoch: 1,
                },
                sequence,
                timestamp_ns,
            },
            geometry_lineage: 1,
            sensor_origin_px: (0.0, 0.0),
            reference: Ellipse {
                center: (100.0, 100.0),
                major_radius: 80.0,
                minor_radius: 60.0,
                angle: 0.0,
            },
            independent_scale: None,
        }
    }
    #[test]
    fn cached_and_stale_results_never_refresh_or_vote() {
        let mut m = RecentExclusionMemory::new(config()).unwrap();
        m.begin_frame(frame(1, 1_000_000_000));
        assert!(m.observe_current(&[(180.0, 100.0)], &[]));
        assert!(!m.observe_current(&[(100.0, 160.0)], &[]));
        assert_eq!(
            m.begin_frame(frame(1, 1_000_000_000)),
            FrameDisposition::ReusedOrStale
        );
        assert!(!m.observe_current(&[(180.0, 100.0)], &[]));
        assert_eq!(
            m.begin_frame(frame(0, 900_000_000)),
            FrameDisposition::ReusedOrStale
        );
        m.begin_frame(frame(2, 1_499_000_000));
        assert!(m.weight((180.0, 100.0), false) < 1.0);
        m.observe_current(&[], &[]);
        m.begin_frame(frame(3, 1_500_000_000));
        assert_eq!(m.weight((180.0, 100.0), false), 1.0);
        assert_eq!(m.active_regions(), 0);
    }
    #[test]
    fn fresh_arc_reenters_and_clears_hysteresis() {
        let mut c = config();
        c.reexclude_thresholds = Some((0.6, 0.2));
        let mut m = RecentExclusionMemory::new(c).unwrap();
        m.begin_frame(frame(1, 1_000_000_000));
        m.observe_current(&[(180.0, 100.0)], &[]);
        m.begin_frame(frame(2, 1_010_000_000));
        assert_eq!(m.weight((180.0, 100.0), false), 0.0);
        assert_eq!(m.weight((180.0, 100.0), true), 1.0);
        m.observe_current(&[], &[(180.0, 100.0)]);
        m.begin_frame(frame(3, 1_020_000_000));
        assert_eq!(m.weight((180.0, 100.0), false), 1.0);
    }
    #[test]
    fn crop_transport_is_sensor_aware_but_lineage_and_geometry_changes_reset() {
        for change in 0..5 {
            let mut m = RecentExclusionMemory::new(config()).unwrap();
            m.begin_frame(frame(1, 1_000_000_000));
            m.observe_current(&[(180.0, 100.0)], &[]);
            let mut f = frame(2, 1_010_000_000);
            match change {
                0 => {
                    f.sensor_origin_px.0 = 20.0;
                    f.reference.center.0 -= 20.0;
                }
                1 => f.exposure.clock.epoch += 1,
                2 => f.exposure.roi = RoiId(2),
                3 => f.reference.major_radius *= 1.4,
                _ => f.geometry_lineage += 1,
            }
            let disposition = m.begin_frame(f);
            if change == 0 {
                assert_eq!(disposition, FrameDisposition::Fresh);
                assert!(m.weight((160.0, 100.0), false) < 1.0);
            } else {
                assert_eq!(disposition, FrameDisposition::Reset);
                assert_eq!(m.active_regions(), 0);
            }
        }
    }
    #[test]
    fn bounded_memory_and_no_feedback_from_weight_queries() {
        let mut m = RecentExclusionMemory::new(config()).unwrap();
        m.begin_frame(frame(1, 1));
        m.observe_current(&frame(1, 1).reference.dense_points(10000), &[]);
        assert!(m.active_regions() <= REGION_BINS);
        for i in 2..50 {
            m.begin_frame(frame(i, i * 10_000_000));
            m.weight((180.0, 100.0), false);
            m.observe_current(&[], &[]);
        }
        m.begin_frame(frame(50, 500_000_001));
        assert_eq!(m.active_regions(), 0);
    }
    #[test]
    fn all_decay_functions_expire_and_are_monotone() {
        for d in [
            DecayFunction::Exponential,
            DecayFunction::Linear,
            DecayFunction::FiniteHorizon,
        ] {
            let mut previous = 1.0;
            for t in 0..600 {
                let v = d.value(t, 100, 500);
                assert!((0.0..=previous).contains(&v));
                previous = v;
            }
            assert_eq!(d.value(500, 100, 500), 0.0);
        }
    }
    #[test]
    fn sn_feida_preserves_obliquity_and_independent_scale_invariance() {
        let mut e = frame(1, 1).reference;
        let s = IndependentScale {
            pixels_per_reference_unit: 2.0,
            provenance: 7,
        };
        let base = sn_feida(e, s).unwrap();
        e.minor_radius *= 0.5;
        assert_eq!(sn_feida(e, s), Some(base));
        e.major_radius *= 3.0;
        e.minor_radius *= 3.0;
        assert!(
            (sn_feida(
                e,
                IndependentScale {
                    pixels_per_reference_unit: 6.0,
                    ..s
                }
            )
            .unwrap()
                - base)
                .abs()
                < 1e-10
        );
        assert!(sn_feida(
            e,
            IndependentScale {
                pixels_per_reference_unit: 0.0,
                ..s
            }
        )
        .is_none());
    }
    #[test]
    fn circle_phase_does_not_follow_arbitrary_major_axis() {
        let mut e = frame(1, 1).reference;
        e.minor_radius = e.major_radius;
        let first = canonical_point((150.0, 150.0), e).unwrap();
        e.angle = PI * 0.49;
        let second = canonical_point((150.0, 150.0), e).unwrap();
        assert!((first.0 - second.0).abs() < 1e-12);
        assert!((first.1 - second.1).abs() < 1e-12);
    }
    #[test]
    fn axis_plus_pi_cannot_move_upper_exclusion_to_lower() {
        let mut m = RecentExclusionMemory::new(config()).unwrap();
        m.begin_frame(frame(1, 1_000_000_000));
        m.observe_current(&[(100.0, 40.0)], &[]);
        let mut next = frame(2, 1_010_000_000);
        next.reference.angle += PI;
        assert_eq!(m.begin_frame(next), FrameDisposition::Fresh);
        assert!(m.weight((100.0, 40.0), false) < 1.0);
        assert_eq!(m.weight((100.0, 160.0), false), 1.0);
        let a = canonical_point((100.0, 40.0), frame(1, 1).reference).unwrap();
        let b = canonical_point((100.0, 40.0), next.reference).unwrap();
        assert!((a.0 - b.0).abs() < 1e-12);
    }
    #[test]
    fn invalid_input_and_independent_scale_provenance_reset() {
        let mut m = RecentExclusionMemory::new(config()).unwrap();
        let mut f = frame(1, 1);
        f.independent_scale = Some(IndependentScale {
            pixels_per_reference_unit: 1.0,
            provenance: 1,
        });
        m.begin_frame(f);
        m.observe_current(&[(180.0, 100.0)], &[]);
        f.exposure.sequence = 2;
        f.exposure.timestamp_ns = 2;
        f.independent_scale.as_mut().unwrap().provenance = 2;
        assert_eq!(m.begin_frame(f), FrameDisposition::Reset);
        m.observe_current(&[(180.0, 100.0)], &[]);
        f.reference.angle = f64::NAN;
        assert_eq!(m.begin_frame(f), FrameDisposition::Invalid);
        assert_eq!(m.active_regions(), 0);
    }
    #[test]
    fn incompatible_oblique_orientation_resets_but_missing_scale_is_not_invented() {
        let mut m = RecentExclusionMemory::new(config()).unwrap();
        let mut f = frame(1, 1);
        f.reference.minor_radius = 40.0;
        m.begin_frame(f);
        m.observe_current(&[(180.0, 100.0)], &[]);
        f.exposure.sequence = 2;
        f.exposure.timestamp_ns = 2;
        f.reference.angle = PI * 0.25;
        assert_eq!(m.begin_frame(f), FrameDisposition::Reset);
        m.observe_current(&[(180.0, 100.0)], &[]);
        f.exposure.sequence = 3;
        f.exposure.timestamp_ns = 3;
        f.independent_scale = Some(IndependentScale {
            pixels_per_reference_unit: 1.0,
            provenance: 3,
        });
        assert_eq!(m.begin_frame(f), FrameDisposition::Reset);
        assert_eq!(m.active_regions(), 0);
    }
}
