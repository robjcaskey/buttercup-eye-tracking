//! Sensor-coordinate visibility and finite-lived support across ROI moves.
//!
//! A crop intersection is not an eye mask and an eye mask is not a newly
//! observed iris. Keep those three claims separate. Geometry here allocates
//! nothing; borrowed foreground masks are scanned once with a finite bound.
//! Support weights and clipping classes are engineering diagnostics, never
//! calibrated probabilities or authorization to synthesize missing pixels.

use crate::roi_evidence::ExposureKey;

/// Half-open native sensor-pixel rectangle, independent of display/model size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SensorRect {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl SensorRect {
    pub(crate) fn area(self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    pub(crate) fn contains(self, point: [f64; 2]) -> bool {
        point[0].is_finite()
            && point[1].is_finite()
            && point[0] >= f64::from(self.x)
            && point[1] >= f64::from(self.y)
            && point[0] < (u64::from(self.x) + u64::from(self.width)) as f64
            && point[1] < (u64::from(self.y) + u64::from(self.height)) as f64
    }

    pub(crate) fn intersection(self, other: Self) -> Option<Self> {
        let left = self.x.max(other.x);
        let top = self.y.max(other.y);
        let right = (u64::from(self.x) + u64::from(self.width))
            .min(u64::from(other.x) + u64::from(other.width));
        let bottom = (u64::from(self.y) + u64::from(self.height))
            .min(u64::from(other.y) + u64::from(other.height));
        if right <= u64::from(left) || bottom <= u64::from(top) {
            return None;
        }
        Some(Self {
            x: left,
            y: top,
            width: u32::try_from(right - u64::from(left)).ok()?,
            height: u32::try_from(bottom - u64::from(top)).ok()?,
        })
    }

    pub(crate) fn inset(self, margin: u32) -> Option<Self> {
        let diameter = margin.checked_mul(2)?;
        let width = self.width.checked_sub(diameter)?;
        let height = self.height.checked_sub(diameter)?;
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self {
            x: self.x.checked_add(margin)?,
            y: self.y.checked_add(margin)?,
            width,
            height,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SensorOverlap {
    pub(crate) source: SensorRect,
    pub(crate) current: SensorRect,
    pub(crate) intersection: Option<SensorRect>,
    /// Fraction of the old crop still represented by the current window.
    pub(crate) source_fraction: f64,
    /// Fraction of the new crop for which an old sensor sample exists.
    pub(crate) current_fraction: f64,
}

impl SensorOverlap {
    /// Zero-area input is malformed/unknown, unlike valid disjoint crops.
    pub(crate) fn between(source: SensorRect, current: SensorRect) -> Option<Self> {
        if source.area() == 0 || current.area() == 0 {
            return None;
        }
        let intersection = source.intersection(current);
        let common_area = intersection.map_or(0, SensorRect::area) as f64;
        Some(Self {
            source,
            current,
            intersection,
            source_fraction: common_area / source.area() as f64,
            current_fraction: common_area / current.area() as f64,
        })
    }

    /// Both frames must independently supply the patch, including its margin.
    /// Missing margins are not filled with padding or copied historical RAW.
    pub(crate) fn supports_margin(self, margin: u32) -> Option<SensorRect> {
        self.source
            .inset(margin)?
            .intersection(self.current.inset(margin)?)
    }

    pub(crate) fn pixel_coverage(self, sensor_point: [f64; 2]) -> PixelCoverage {
        match (
            self.source.contains(sensor_point),
            self.current.contains(sensor_point),
        ) {
            (true, true) => PixelCoverage::OverlapCandidate,
            (false, true) => PixelCoverage::CurrentOnlyNoHistory,
            (true, false) => PixelCoverage::HistoricalOnly,
            (false, false) => PixelCoverage::OutsideBoth,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PixelCoverage {
    /// Common sensor location, not proof of correspondence after physical motion.
    OverlapCandidate,
    CurrentOnlyNoHistory,
    HistoricalOnly,
    OutsideBoth,
}

/// Diagnostic division of the *observed foreground*, not the fitted full disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisibilityClass {
    Full,
    Partial,
    Severe,
    None,
}

impl VisibilityClass {
    pub(crate) fn from_fraction(fraction: f64) -> Option<Self> {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return None;
        }
        Some(if fraction >= 1.0 - 1.0e-12 {
            Self::Full
        } else if fraction >= 0.50 {
            Self::Partial
        } else if fraction > 0.0 {
            Self::Severe
        } else {
            Self::None
        })
    }

    /// Geometry-only recommendation. This is not admission: the caller must
    /// independently verify source identity, immutable age, focus and spatial
    /// support before retaining evidence. Naming this separately lets a worker
    /// without the camera's clock-domain key report clipping honestly.
    pub(crate) fn spatial_action(self) -> VisibilityAction {
        match self {
            Self::Full => VisibilityAction::RetainSupportedState,
            Self::Partial => VisibilityAction::RetainWithReducedSupport,
            Self::Severe | Self::None => VisibilityAction::Reassociate,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ForegroundVisibility {
    /// Number of nonzero source mask cells, before considering the new crop.
    source_cells: usize,
    /// Sum of each source foreground cell's area fraction in the new crop.
    /// Fractional boundary cells avoid treating a center-in-window test as
    /// proof that the complete source-supported area remained visible.
    visible_cell_equivalents: f64,
}

impl ForegroundVisibility {
    pub(crate) fn source_cells(self) -> usize {
        self.source_cells
    }

    pub(crate) fn visible_fraction(self) -> f64 {
        self.visible_cell_equivalents / self.source_cells as f64
    }

    pub(crate) fn class(self) -> VisibilityClass {
        VisibilityClass::from_fraction(self.visible_fraction()).unwrap_or(VisibilityClass::None)
    }

    /// Association is only an eligibility test: current RAW/shape/identity
    /// checks still apply. Keep existing caller thresholds unless evaluated.
    pub(crate) fn supports_association(self, minimum_fraction: f64) -> bool {
        minimum_fraction.is_finite()
            && (0.0..=1.0).contains(&minimum_fraction)
            && self.visible_fraction() >= minimum_fraction
    }
}

// Enough for the current 384x256 masks and native eye crops. This is a bounded
// scan, not a whole-sensor allocation or an unlimited dense-history operation.
const MAX_FOREGROUND_CELLS: usize = 1_048_576;

/// The caller must have independently admitted this source mask against RAW.
/// This only intersects that evidence's sensor-addressed cells with a crop;
/// it neither validates semantics nor fills the inferred hidden iris disk.
/// Empty/all-background/malformed masks are UNKNOWN (`None`), not a measured
/// disappearance (`Some` with zero visible area).
pub(crate) fn mask_foreground_visibility(
    mask: &[u8],
    mask_width: usize,
    mask_height: usize,
    source: SensorRect,
    current: SensorRect,
) -> Option<ForegroundVisibility> {
    let count = mask_width.checked_mul(mask_height)?;
    if count == 0
        || count > MAX_FOREGROUND_CELLS
        || count != mask.len()
        || source.area() == 0
        || current.area() == 0
    {
        return None;
    }
    let cell_width = f64::from(source.width) / mask_width as f64;
    let cell_height = f64::from(source.height) / mask_height as f64;
    let current_right = f64::from(current.x) + f64::from(current.width);
    let current_bottom = f64::from(current.y) + f64::from(current.height);
    let mut source_cells = 0usize;
    let mut visible_cell_equivalents = 0.0;
    for (index, &value) in mask.iter().enumerate() {
        if value == 0 {
            continue;
        }
        source_cells += 1;
        let left = f64::from(source.x) + (index % mask_width) as f64 * cell_width;
        let top = f64::from(source.y) + (index / mask_width) as f64 * cell_height;
        let overlap_width =
            ((left + cell_width).min(current_right) - left.max(f64::from(current.x))).max(0.0);
        let overlap_height =
            ((top + cell_height).min(current_bottom) - top.max(f64::from(current.y))).max(0.0);
        visible_cell_equivalents += (overlap_width / cell_width).clamp(0.0, 1.0)
            * (overlap_height / cell_height).clamp(0.0, 1.0);
    }
    (source_cells != 0).then_some(ForegroundVisibility {
        source_cells,
        visible_cell_equivalents: visible_cell_equivalents.clamp(0.0, source_cells as f64),
    })
}

/// One admitted source. Copying/transporting it must retain this original key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SupportSource {
    pub(crate) exposure: ExposureKey,
    pub(crate) identity_epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SupportKind {
    RawObservation,
    Transported,
    Held,
    Predicted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SupportReason {
    Supported,
    MissingForeground,
    IncompatibleIdentity,
    IncompatibleClock,
    ReversedSource,
    Expired,
    PoorFocus,
    InvalidFocusReliability,
    InsufficientSpatialCoverage,
}

impl SupportSource {
    pub(crate) fn age_at(self, current: Self, max_age_ns: u64) -> Result<u64, SupportReason> {
        if self.exposure.roi != current.exposure.roi
            || self.identity_epoch != current.identity_epoch
        {
            return Err(SupportReason::IncompatibleIdentity);
        }
        if self.exposure.clock != current.exposure.clock {
            return Err(SupportReason::IncompatibleClock);
        }
        // Same-exposure alternate crops are legal but remain a single source.
        // A changed sequence with unchanged time is not a new exposure.
        let same = self.exposure.sequence == current.exposure.sequence
            && self.exposure.timestamp_ns == current.exposure.timestamp_ns;
        if !same
            && (self.exposure.sequence >= current.exposure.sequence
                || self.exposure.timestamp_ns >= current.exposure.timestamp_ns)
        {
            return Err(SupportReason::ReversedSource);
        }
        let age = current.exposure.timestamp_ns - self.exposure.timestamp_ns;
        if age > max_age_ns {
            Err(SupportReason::Expired)
        } else {
            Ok(age)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisibilityAction {
    RetainSupportedState,
    RetainWithReducedSupport,
    /// Local current association is required; this does not itself demand a
    /// camera-wide search, erase the other eye, or assert a changed identity.
    Reassociate,
    Abstain,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct VisibilityAssessment {
    pub(crate) class: Option<VisibilityClass>,
    pub(crate) action: VisibilityAction,
    pub(crate) reason: SupportReason,
    pub(crate) age_ns: Option<u64>,
    /// A same-source observation can be current without being independent of
    /// another derived view. The ingestion owner must deduplicate exposure keys.
    pub(crate) current_raw_observation: bool,
    /// Relative engineering support, NOT calibrated fidelity. `None` preserves
    /// unknown focus instead of silently promoting it to sharp or failed.
    pub(crate) heuristic_weight: Option<f64>,
    pub(crate) weight_without_focus: f64,
}

/// Grade visible evidence without inventing fresh observations or letting
/// repeated reframe calls renew the immutable source age. `spatial_coverage`
/// must be based on actual samples/arc sectors, not the inferred ellipse area.
/// `focus_reliability` is an optional caller-normalized detail heuristic; RAW
/// camera focus scores are not probabilities and must not be passed as one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assess_support(
    visibility: Option<ForegroundVisibility>,
    source: SupportSource,
    current: SupportSource,
    kind: SupportKind,
    max_age_ns: u64,
    spatial_coverage: f64,
    focus_reliability: Option<f64>,
) -> VisibilityAssessment {
    let class = visibility.map(ForegroundVisibility::class);
    let mut result = VisibilityAssessment {
        class,
        action: VisibilityAction::Abstain,
        reason: SupportReason::Supported,
        age_ns: None,
        current_raw_observation: false,
        heuristic_weight: None,
        weight_without_focus: 0.0,
    };
    let age = match source.age_at(current, max_age_ns) {
        Ok(age) => age,
        Err(reason) => {
            result.reason = reason;
            if reason == SupportReason::Expired {
                result.action = VisibilityAction::Reassociate;
            }
            return result;
        }
    };
    result.age_ns = Some(age);
    let Some(visibility) = visibility else {
        result.reason = SupportReason::MissingForeground;
        result.action = VisibilityAction::Reassociate;
        return result;
    };
    if !spatial_coverage.is_finite() || spatial_coverage <= 0.0 || spatial_coverage > 1.0 {
        result.reason = SupportReason::InsufficientSpatialCoverage;
        return result;
    }
    if focus_reliability.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        // Malformed reported focus is not the same as a caller honestly
        // declaring that focus is unknown. Do not silently upgrade it.
        result.reason = SupportReason::InvalidFocusReliability;
        return result;
    }
    let age_weight = if max_age_ns == 0 {
        1.0
    } else {
        1.0 / (1.0 + age as f64 / max_age_ns as f64)
    };
    result.weight_without_focus = visibility.visible_fraction() * spatial_coverage * age_weight;
    let focus = focus_reliability;
    result.heuristic_weight = focus.map(|value| value * result.weight_without_focus);
    if focus == Some(0.0) {
        result.reason = SupportReason::PoorFocus;
        return result;
    }
    result.current_raw_observation = age == 0
        && kind == SupportKind::RawObservation
        && visibility.class() != VisibilityClass::None;
    result.action = visibility.class().spatial_action();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi_evidence::{RoiId, SourceClock};

    fn rect(x: u32, y: u32, width: u32, height: u32) -> SensorRect {
        SensorRect {
            x,
            y,
            width,
            height,
        }
    }

    fn source(sequence: u64, timestamp_ns: u64) -> SupportSource {
        SupportSource {
            exposure: ExposureKey {
                roi: RoiId(0),
                clock: SourceClock {
                    domain: 1,
                    epoch: 2,
                },
                sequence,
                timestamp_ns,
            },
            identity_epoch: 3,
        }
    }

    fn full_support() -> ForegroundVisibility {
        ForegroundVisibility {
            source_cells: 80,
            visible_cell_equivalents: 80.0,
        }
    }

    #[test]
    fn nudge_loses_crop_border_but_retains_all_validated_foreground() {
        let old = rect(3596, 2836, 420, 280);
        let new = rect(3628, 2860, 420, 280);
        let overlap = SensorOverlap::between(old, new).unwrap();
        assert_eq!(overlap.intersection, Some(rect(3628, 2860, 388, 256)));
        assert!((overlap.source_fraction - 0.844625850340136).abs() < 1.0e-12);
        assert_eq!(overlap.source_fraction, overlap.current_fraction);
        let mut mask = [0u8; 42 * 28];
        for y in 6..22 {
            for x in 8..34 {
                mask[y * 42 + x] = 1;
            }
        }
        let eye = mask_foreground_visibility(&mask, 42, 28, old, new).unwrap();
        assert_eq!(eye.class(), VisibilityClass::Full);
        assert_eq!(eye.visible_fraction(), 1.0);
        assert!(eye.supports_association(0.80));
        assert_eq!(
            overlap.pixel_coverage([3600.0, 2840.0]),
            PixelCoverage::HistoricalOnly
        );
        assert_eq!(
            overlap.pixel_coverage([4020.0, 3000.0]),
            PixelCoverage::CurrentOnlyNoHistory
        );
        assert_eq!(
            overlap.pixel_coverage([3800.0, 3000.0]),
            PixelCoverage::OverlapCandidate
        );
    }

    #[test]
    fn foreground_area_does_not_promote_a_partly_clipped_cell_to_fully_visible() {
        let eye =
            mask_foreground_visibility(&[1], 1, 1, rect(0, 0, 100, 100), rect(25, 0, 100, 100))
                .unwrap();
        assert_eq!(eye.visible_fraction(), 0.75);
        assert_eq!(eye.class(), VisibilityClass::Partial);
        assert!(!eye.supports_association(0.80));
    }

    #[test]
    fn tiny_and_disjoint_overlaps_cannot_supply_whole_eye_support() {
        let old = rect(0, 0, 420, 280);
        let thin = rect(416, 0, 420, 280);
        let overlap = SensorOverlap::between(old, thin).unwrap();
        assert!((overlap.source_fraction - 4.0 / 420.0).abs() < 1.0e-12);
        assert!(overlap.supports_margin(10).is_none());
        let eye = mask_foreground_visibility(&[1], 1, 1, old, thin).unwrap();
        assert_eq!(eye.class(), VisibilityClass::Severe);
        let none = mask_foreground_visibility(&[1], 1, 1, old, rect(420, 0, 420, 280)).unwrap();
        assert_eq!(none.class(), VisibilityClass::None);
        assert!(SensorOverlap::between(old, rect(420, 0, 420, 280))
            .unwrap()
            .intersection
            .is_none());
        assert!(SensorOverlap::between(old, rect(0, 0, 0, 280)).is_none());
    }

    #[test]
    fn asymmetric_crop_sizes_report_both_history_and_current_support_fractions() {
        let overlap = SensorOverlap::between(rect(0, 0, 100, 100), rect(25, 25, 50, 50)).unwrap();
        assert_eq!(overlap.source_fraction, 0.25);
        assert_eq!(overlap.current_fraction, 1.0);
        assert_eq!(overlap.supports_margin(10), Some(rect(35, 35, 30, 30)));
    }

    #[test]
    fn malformed_or_missing_foreground_is_unknown_not_disappearance() {
        let crop = rect(0, 0, 100, 100);
        assert!(mask_foreground_visibility(&[], 0, 0, crop, crop).is_none());
        assert!(mask_foreground_visibility(&[0], 1, 1, crop, crop).is_none());
        assert!(mask_foreground_visibility(&[1], 2, 2, crop, crop).is_none());
        assert!(mask_foreground_visibility(&[1], usize::MAX, 2, crop, crop).is_none());
        let assessment = assess_support(
            None,
            source(1, 10),
            source(2, 20),
            SupportKind::Held,
            100,
            1.0,
            Some(1.0),
        );
        assert_eq!(assessment.class, None);
        assert_eq!(assessment.reason, SupportReason::MissingForeground);
        assert_eq!(assessment.action, VisibilityAction::Reassociate);
        assert!(!assessment.current_raw_observation);
    }

    #[test]
    fn reframes_never_renew_original_age_or_make_transport_fresh() {
        let original = source(1, 100);
        let first = assess_support(
            Some(full_support()),
            original,
            source(2, 150),
            SupportKind::Transported,
            100,
            1.0,
            Some(1.0),
        );
        let later = assess_support(
            Some(full_support()),
            original,
            source(3, 190),
            SupportKind::Held,
            100,
            1.0,
            Some(1.0),
        );
        assert_eq!(first.age_ns, Some(50));
        assert_eq!(later.age_ns, Some(90));
        assert!(later.heuristic_weight.unwrap() < first.heuristic_weight.unwrap());
        assert!(!first.current_raw_observation && !later.current_raw_observation);
        let expired = assess_support(
            Some(full_support()),
            original,
            source(4, 201),
            SupportKind::Transported,
            100,
            1.0,
            Some(1.0),
        );
        assert_eq!(expired.reason, SupportReason::Expired);
        assert_eq!(expired.action, VisibilityAction::Reassociate);
        assert_eq!(expired.weight_without_focus, 0.0);
        for kind in [
            SupportKind::Transported,
            SupportKind::Held,
            SupportKind::Predicted,
        ] {
            let same = assess_support(
                Some(full_support()),
                original,
                original,
                kind,
                100,
                1.0,
                Some(1.0),
            );
            assert!(!same.current_raw_observation);
        }
    }

    #[test]
    fn wrong_eye_identity_clock_and_reversed_time_abstain() {
        let original = source(1, 100);
        let mut wrong_eye = source(2, 150);
        wrong_eye.exposure.roi = RoiId(1);
        let mut wrong_identity = source(2, 150);
        wrong_identity.identity_epoch += 1;
        let mut wrong_clock = source(2, 150);
        wrong_clock.exposure.clock.epoch += 1;
        for (current, reason) in [
            (wrong_eye, SupportReason::IncompatibleIdentity),
            (wrong_identity, SupportReason::IncompatibleIdentity),
            (wrong_clock, SupportReason::IncompatibleClock),
            (source(2, 100), SupportReason::ReversedSource),
            (source(1, 150), SupportReason::ReversedSource),
            (source(0, 90), SupportReason::ReversedSource),
        ] {
            let assessment = assess_support(
                Some(full_support()),
                original,
                current,
                SupportKind::RawObservation,
                100,
                1.0,
                Some(1.0),
            );
            assert_eq!(assessment.reason, reason);
            assert_eq!(assessment.action, VisibilityAction::Abstain);
            assert!(!assessment.current_raw_observation);
        }
    }

    #[test]
    fn unknown_focus_differs_from_known_poor_focus_and_partial_support() {
        let assess = |support, spatial_coverage, focus| {
            assess_support(
                support,
                source(1, 100),
                source(1, 100),
                SupportKind::RawObservation,
                100,
                spatial_coverage,
                focus,
            )
        };
        let unknown = assess(Some(full_support()), 1.0, None);
        let sharp = assess(Some(full_support()), 1.0, Some(1.0));
        let poor = assess(Some(full_support()), 1.0, Some(0.0));
        assert_eq!(unknown.heuristic_weight, None);
        assert_eq!(unknown.reason, SupportReason::Supported);
        assert_eq!(poor.heuristic_weight, Some(0.0));
        assert_eq!(poor.reason, SupportReason::PoorFocus);
        assert_eq!(poor.action, VisibilityAction::Abstain);
        let partial = assess(
            Some(ForegroundVisibility {
                source_cells: 80,
                visible_cell_equivalents: 48.0,
            }),
            0.5,
            Some(0.5),
        );
        assert_eq!(partial.class, Some(VisibilityClass::Partial));
        assert_eq!(partial.action, VisibilityAction::RetainWithReducedSupport);
        assert!((partial.heuristic_weight.unwrap() - 0.15).abs() < 1.0e-12);
        assert!(partial.heuristic_weight.unwrap() < sharp.heuristic_weight.unwrap());
        assert_eq!(
            assess(Some(full_support()), 0.0, Some(1.0)).reason,
            SupportReason::InsufficientSpatialCoverage
        );
    }

    #[test]
    fn large_jump_and_recovery_are_not_identity_changes_or_fresh_holds() {
        let old = rect(0, 144, 384, 256);
        let jumped = rect(144, 0, 384, 256);
        let visibility = mask_foreground_visibility(&[1], 1, 1, old, jumped).unwrap();
        assert_eq!(visibility.class(), VisibilityClass::Severe);
        let clipped = assess_support(
            Some(visibility),
            source(1, 100),
            source(2, 120),
            SupportKind::Held,
            100,
            0.3,
            Some(1.0),
        );
        assert_eq!(clipped.action, VisibilityAction::Reassociate);
        assert_eq!(clipped.reason, SupportReason::Supported);
        let recovery = assess_support(
            Some(full_support()),
            source(3, 150),
            source(3, 150),
            SupportKind::RawObservation,
            100,
            1.0,
            Some(1.0),
        );
        assert!(recovery.current_raw_observation);
        assert_eq!(recovery.action, VisibilityAction::RetainSupportedState);
    }

    #[test]
    fn malformed_focus_never_upgrades_to_unknown_or_supported() {
        for focus in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            let assessment = assess_support(
                Some(full_support()),
                source(1, 100),
                source(1, 100),
                SupportKind::RawObservation,
                100,
                1.0,
                Some(focus),
            );
            assert_eq!(assessment.reason, SupportReason::InvalidFocusReliability);
            assert_eq!(assessment.action, VisibilityAction::Abstain);
            assert_eq!(assessment.heuristic_weight, None);
            assert_eq!(assessment.weight_without_focus, 0.0);
            assert!(!assessment.current_raw_observation);
        }
    }

    #[test]
    fn extreme_rectangles_do_not_wrap_into_false_overlap_or_pixel_history() {
        let high = rect(u32::MAX - 9, u32::MAX - 9, 20, 20);
        let low = rect(0, 0, 20, 20);
        let overlap = SensorOverlap::between(high, low).unwrap();
        assert_eq!(overlap.source_fraction, 0.0);
        assert_eq!(overlap.current_fraction, 0.0);
        assert!(overlap.intersection.is_none());
        assert_eq!(
            overlap.pixel_coverage([0.0, 0.0]),
            PixelCoverage::CurrentOnlyNoHistory
        );
        for point in [
            [f64::NAN, 0.0],
            [0.0, f64::INFINITY],
            [f64::NEG_INFINITY, 0.0],
        ] {
            assert_eq!(overlap.pixel_coverage(point), PixelCoverage::OutsideBoth);
        }
        assert!(high.inset(u32::MAX).is_none());
        assert!(high.inset(10).is_none());
        assert_eq!(
            SensorOverlap::between(high, high).unwrap().source_fraction,
            1.0
        );
        for fraction in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            assert_eq!(VisibilityClass::from_fraction(fraction), None);
        }
    }

    #[test]
    fn disappeared_foreground_is_not_a_current_observation_even_on_same_exposure() {
        let vanished =
            mask_foreground_visibility(&[1], 1, 1, rect(0, 0, 100, 100), rect(100, 0, 100, 100))
                .unwrap();
        let assessment = assess_support(
            Some(vanished),
            source(1, 100),
            source(1, 100),
            SupportKind::RawObservation,
            100,
            1.0,
            Some(1.0),
        );
        assert_eq!(assessment.class, Some(VisibilityClass::None));
        assert_eq!(assessment.action, VisibilityAction::Reassociate);
        assert!(!assessment.current_raw_observation);
        assert_eq!(assessment.heuristic_weight, Some(0.0));
    }
}
