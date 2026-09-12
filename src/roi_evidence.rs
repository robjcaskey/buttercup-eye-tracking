//! ROI observations and source-clock motion evidence, independent of presentation.
//!
//! The TCP codec owns serialization; detectors own extraction. These records
//! distinguish fitted diagnostic motion from admitted transport. Each timeline
//! belongs to one physical ROI/session and never uses SAM completion time.

use std::collections::VecDeque;
use std::sync::Arc;

pub(crate) mod timing;

/// Physical ROI identity, not a motion-layer index or a presentation slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RoiId(pub(crate) u32);

/// Caller-owned source-clock/session identity. Reconnects that restart the
/// sensor clock must advance the epoch. Host receipt time is not this clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceClock {
    pub(crate) domain: u64,
    pub(crate) epoch: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExposureKey {
    pub(crate) roi: RoiId,
    pub(crate) clock: SourceClock,
    pub(crate) sequence: u64,
    pub(crate) timestamp_ns: u64,
}

impl ExposureKey {
    /// Cross-eye time comparison is meaningful only for explicitly co-clocked
    /// exposures. Sequence numbers remain local to their respective ROIs.
    pub(crate) fn separation_ns(self, other: Self) -> Option<u64> {
        (self.clock == other.clock).then(|| self.timestamp_ns.abs_diff(other.timestamp_ns))
    }
}

impl RawModelFrame {
    pub(crate) fn exposure_key(&self, clock: SourceClock) -> ExposureKey {
        ExposureKey {
            roi: RoiId(self.eye_id),
            clock,
            sequence: self.sequence,
            timestamp_ns: self.timestamp_ns,
        }
    }
}

/// Semantic alternatives must survive extraction, not be collapsed to a
/// universal "iris" ellipse before the joint solver sees the observations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BoundaryKind {
    OuterLimbus,
    InnerLimbus,
    PupillaryBoundary,
    Unclassified,
}

/// A measured outward IMAGE-boundary normal, not the eye's 3D surface normal.
/// It belongs to the same native-ROI sample and source exposure as its point.
/// RAW polarity establishes outward direction; a fitted ellipse must never
/// manufacture this observation. Its angular sigma is an engineering model,
/// not a calibrated posterior or an independent second vote for the contour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BoundaryNormalObservation {
    pub(crate) unit_outward_roi: [f64;2],
    pub(crate) angular_sigma_radians: f64,
}

impl BoundaryNormalObservation {
    pub(crate) fn valid(self)->bool {
        self.unit_outward_roi.into_iter().all(f64::is_finite)
            && (self.unit_outward_roi[0].hypot(self.unit_outward_roi[1])-1.0).abs()<1.0e-6
            && self.angular_sigma_radians.is_finite() && self.angular_sigma_radians>0.0
    }
}

/// Sparse native-ROI evidence contract for the future joint solvers.
/// A score is detector-specific ranking evidence, NEVER a posterior
/// probability. Missing normal localization is unknown, not zero noise.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundaryArcObservation<'a> {
    /// Stable within this exposure; correlated alternatives reuse the same
    /// evidence group so downstream solvers can avoid double-counting pixels.
    pub(crate) evidence_group: u32,
    pub(crate) kind: BoundaryKind,
    pub(crate) points_roi_px: &'a [(f64, f64)],
    /// When present, exactly one optional direction per point, before any
    /// downstream decimation. Missing directions supply no angular constraint.
    pub(crate) outward_normals_roi: Option<&'a [Option<BoundaryNormalObservation>]>,
    pub(crate) normal_band_half_width_px: Option<f64>,
    pub(crate) detector_score: Option<f64>,
}

/// A supported partial/full conic is not interchangeable with raw samples.
/// Its support indices refer to arcs in the same RoiConicEvidence packet.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConicObservation<'a> {
    pub(crate) kind: BoundaryKind,
    pub(crate) ellipse_roi_px: crate::geometry::Ellipse,
    pub(crate) supporting_arc_indices: &'a [usize],
    pub(crate) residual_px: Option<f64>,
}

/// Every borrowed arc/conic is in this exposure's native ROI pixel frame.
/// Already fitted weak conics may be supplied even when no sharp arcs survive.
/// Empty slices mean unavailable evidence; they do not describe a closed eye.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoiConicEvidence<'a> {
    pub(crate) exposure: ExposureKey,
    pub(crate) sensor_origin_px: [u32; 2],
    pub(crate) dimensions_px: [u32; 2],
    pub(crate) arcs: &'a [BoundaryArcObservation<'a>],
    pub(crate) conics: &'a [ConicObservation<'a>],
    /// Optical reliability is separate from geometric compatibility.
    pub(crate) detail_reliability: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct RawModelFrame {
    pub eye_id: u32,
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub sensor_x: u32,
    pub sensor_y: u32,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub flags: u32,
    pub focus_target: u16,
    pub focus_position: u16,
    pub focus_generation: u32,
    pub focus_score: f32,
    pub motion_score: f32,
    pub center_x: f32,
    pub center_y: f32,
    pub iris_radius: f32,
    pub axis_ratio: f32,
    pub axis_angle: f32,
    pub point_count: u32,
    pub payload: Arc<Vec<u8>>,
}

// Legacy within-ROI motion-layer indices. These are NOT physical eye IDs.
pub const OBJECTS: usize = 4;
pub const GENERAL_LAYER: usize = 0;
pub const PUPIL_LAYER: usize = 1;
pub const REFLECTION_LAYER: usize = 2;
pub const RESIDUAL_LAYER: usize = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct SimilarityMotion {
    pub translation: [f32; 2],
    /// Off-diagonal coefficient b of [[1+d, -b], [b, 1+d]], not an angle.
    pub rotation_coefficient: f32,
    /// d in that matrix, not the exact similarity-scale change. The exact
    /// angle is atan2(b, 1+d) and the scale is hypot(1+d, b).
    pub diagonal_coefficient_delta: f32,
    pub residual: f32,
    pub support: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MotionLayerStatus {
    /// Mean image-space position of the currently associated tracks.
    pub centroid: [f32; 2],
    /// Translation relative to the robust whole-frame motion.
    pub differential: [f32; 2],
    /// Signed coordinate on the learned dominant parallax axis. This is not
    /// metric depth, but remains directionally consistent across frames.
    pub parallax: f32,
    /// Temporal agreement of member tracks with this layer's motion model.
    pub coherence: f32,
    /// Mean RMS distance between member motion histories and the layer's
    /// multi-frame signature, in full-resolution pixels.
    pub trajectory_error: f32,
    pub signature_samples: usize,
    /// Distance in motion space to the nearest other supported layer.
    pub separation: f32,
    pub persistent_tracks: usize,
    pub stable_frames: u16,
}

impl SimilarityMotion {
    pub(crate) fn predict(self, point: [f32; 2], center: [f32; 2]) -> [f32; 2] {
        let x = point[0] - center[0];
        let y = point[1] - center[1];
        [
            point[0] + self.translation[0] + self.diagonal_coefficient_delta * x - self.rotation_coefficient * y,
            point[1] + self.translation[1] + self.rotation_coefficient * x + self.diagonal_coefficient_delta * y,
        ]
    }
}

/// One native patch correspondence. The enclosing source interval owns its
/// clocks; these coordinates are full-sensor pixels, not current ROI pixels.
/// A match is not an anatomical identity or a pure head-motion observation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NativePatchCorrespondence {
    pub(crate) previous_sensor_px: [f32; 2],
    pub(crate) current_sensor_px: [f32; 2],
    pub(crate) photometric_score: f32,
    pub(crate) distinct_match_margin: f32,
    pub(crate) global_similarity_inlier: bool,
}

/// Independent full-ROI evidence that an apparent radius change is supported
/// by coherent image scale elsewhere in the native eye frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeGlobalSimilarityEvidence {
    /// Motion authorized as a physical image-scale transport. This remains
    /// zero when the broadly distributed native matches fail any reliability
    /// gate, so downstream anatomy code cannot accidentally consume a merely
    /// diagnostic fit.
    pub motion: SimilarityMotion,
    /// Best robust fit before the whole-ROI reliability gates. Keeping this
    /// diagnostic separate is important: a zero authorized motion otherwise
    /// hides whether support, spatial coverage, residual, or excessive gross
    /// movement rejected an otherwise informative candidate.
    pub candidate_motion: SimilarityMotion,
    pub candidate_matches: usize,
    pub reliable: bool,
    pub stable_frames: u16,
    pub spatial_span: [f32; 2],
    pub occupied_quadrants: usize,
    /// Absolute sensor-space point about which `motion.translation` is
    /// defined. Keeping the center beside the fitted transform is essential
    /// when the sensor ROI moves: applying a scale/rotation about the current
    /// crop center would otherwise manufacture relative pupil motion.
    pub motion_center_sensor: [f32; 2],
}

// Covers more than the maximum accepted asynchronous SAM result age at the
// normal eye-stream cadence. Entries are tiny affine summaries, never image
// data, and are retained separately for each physical eye.
pub(crate) const GLOBAL_SIMILARITY_TIMELINE_STEPS: usize = 64;
#[derive(Clone, Copy, Debug)]
pub(crate) struct TimedGlobalSimilarity {
    pub(crate) from_timestamp_ns: u64,
    pub(crate) to_timestamp_ns: u64,
    pub(crate) evidence: NativeGlobalSimilarityEvidence,
}

/// Adjacent whole-ROI motion indexed in the sensor clock domain. SAM
/// finishes asynchronously, so the motion needed by its surface tracker is
/// the composition from the previous SAM source exposure to the new source
/// exposure, not the transform adjacent to whichever live frame happens to
/// receive the answer.
#[derive(Clone, Debug, Default)]
pub(crate) struct GlobalSimilarityTimeline {
    pub(crate) last_timestamp_ns: Option<u64>,
    pub(crate) steps: VecDeque<TimedGlobalSimilarity>,
}

impl GlobalSimilarityTimeline {
    pub(crate) fn observe_frame(
        &mut self,
        timestamp_ns: u64,
        evidence: NativeGlobalSimilarityEvidence,
    ) {
        let Some(from_timestamp_ns) = self.last_timestamp_ns.replace(timestamp_ns) else {
            return;
        };
        if timestamp_ns <= from_timestamp_ns {
            self.steps.clear();
            return;
        }
        if self
            .steps
            .back()
            .is_some_and(|step| step.to_timestamp_ns != from_timestamp_ns)
        {
            self.steps.clear();
        }
        self.steps.push_back(TimedGlobalSimilarity {
            from_timestamp_ns,
            to_timestamp_ns: timestamp_ns,
            evidence,
        });
        while self.steps.len() > GLOBAL_SIMILARITY_TIMELINE_STEPS {
            self.steps.pop_front();
        }
    }

    pub(crate) fn reliable_between(
        &self,
        from_timestamp_ns: u64,
        to_timestamp_ns: u64,
    ) -> Option<NativeGlobalSimilarityEvidence> {
        if to_timestamp_ns < from_timestamp_ns {
            return None;
        }
        if to_timestamp_ns == from_timestamp_ns {
            return Some(NativeGlobalSimilarityEvidence {
                reliable: true,
                ..NativeGlobalSimilarityEvidence::default()
            });
        }

        // Complex scalar `a + ib` plus translation represents the restricted
        // affine used by SimilarityMotion:
        //   x' = a*x - b*y + tx; y' = b*x + a*y + ty.
        let mut accumulated = (1.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut cursor = from_timestamp_ns;
        let mut residual = 0.0f32;
        let mut support = usize::MAX;
        let mut stable_frames = u16::MAX;
        let mut span = [f32::INFINITY; 2];
        let mut quadrants = usize::MAX;
        let mut used = 0usize;
        for step in self.steps.iter().filter(|step| {
            step.to_timestamp_ns > from_timestamp_ns && step.from_timestamp_ns < to_timestamp_ns
        }) {
            if step.from_timestamp_ns != cursor
                || step.to_timestamp_ns > to_timestamp_ns
                || !step.evidence.reliable
            {
                return None;
            }
            let motion = step.evidence.motion;
            let step_a = 1.0 + f64::from(motion.diagonal_coefficient_delta);
            let step_b = f64::from(motion.rotation_coefficient);
            let center_x = f64::from(step.evidence.motion_center_sensor[0]);
            let center_y = f64::from(step.evidence.motion_center_sensor[1]);
            let step_tx =
                f64::from(motion.translation[0]) + (1.0 - step_a) * center_x + step_b * center_y;
            let step_ty =
                f64::from(motion.translation[1]) - step_b * center_x + (1.0 - step_a) * center_y;
            let (a, b, tx, ty) = accumulated;
            accumulated = (
                step_a * a - step_b * b,
                step_b * a + step_a * b,
                step_a * tx - step_b * ty + step_tx,
                step_b * tx + step_a * ty + step_ty,
            );
            residual = residual.max(motion.residual);
            support = support.min(motion.support);
            stable_frames = stable_frames.min(step.evidence.stable_frames);
            span[0] = span[0].min(step.evidence.spatial_span[0]);
            span[1] = span[1].min(step.evidence.spatial_span[1]);
            quadrants = quadrants.min(step.evidence.occupied_quadrants);
            cursor = step.to_timestamp_ns;
            used += 1;
            if cursor == to_timestamp_ns {
                break;
            }
        }
        if cursor != to_timestamp_ns || used == 0 {
            return None;
        }
        let (a, b, tx, ty) = accumulated;
        if [a, b, tx, ty].into_iter().any(|value| !value.is_finite()) {
            return None;
        }
        let motion = SimilarityMotion {
            translation: [tx as f32, ty as f32],
            rotation_coefficient: b as f32,
            diagonal_coefficient_delta: (a - 1.0) as f32,
            residual,
            support,
        };
        Some(NativeGlobalSimilarityEvidence {
            motion,
            candidate_motion: motion,
            candidate_matches: support,
            reliable: true,
            stable_frames,
            spatial_span: span,
            occupied_quadrants: quadrants,
            // The composed translation is already expressed as an absolute
            // sensor-space affine offset, so downstream prediction is about
            // the origin rather than any one intermediate crop center.
            motion_center_sensor: [0.0; 2],
        })
    }
}

#[cfg(test)]
mod clock_contract_tests {
    use super::*;

    #[test]
    fn submitted_motion_snapshot_survives_live_eviction_and_reset() {
        let evidence = NativeGlobalSimilarityEvidence {
            reliable: true,
            motion: SimilarityMotion {
                translation: [2.0, -1.0],
                support: 12,
                residual: 0.5,
                ..SimilarityMotion::default()
            },
            ..NativeGlobalSimilarityEvidence::default()
        };
        let mut live = GlobalSimilarityTimeline::default();
        for timestamp in [100, 200, 300] {
            live.observe_frame(timestamp, evidence);
        }
        let submitted = live.clone();
        let assert_pinned = || {
            assert_eq!(submitted.last_timestamp_ns, Some(300));
            assert_eq!(submitted.steps.len(), 2);
            let retained = submitted.reliable_between(100, 300).unwrap();
            assert_eq!(retained.motion.translation, [4.0, -2.0]);
            assert_eq!(retained.motion.support, 12);
            assert!(submitted.reliable_between(100, 400).is_none());
        };
        assert_pinned();

        // Evict every originally submitted link from the mutable live queue.
        for index in 1..=GLOBAL_SIMILARITY_TIMELINE_STEPS + 1 {
            live.observe_frame(300 + index as u64 * 100, evidence);
        }
        let current = live.last_timestamp_ns.unwrap();
        assert_eq!(live.steps.len(), GLOBAL_SIMILARITY_TIMELINE_STEPS);
        assert!(live.reliable_between(100, 300).is_none());
        assert!(live.reliable_between(current - 100, current).is_some());
        assert!(submitted.reliable_between(100, current).is_none());
        assert_pinned();

        // A source-time restart clears the live links, and an explicit session
        // reset drops the live timeline entirely. Neither mutates the batch.
        live.observe_frame(50, evidence);
        assert!(live.steps.is_empty());
        assert_eq!(live.last_timestamp_ns, Some(50));
        assert_pinned();
        live = GlobalSimilarityTimeline::default();
        assert!(live.steps.is_empty());
        assert_eq!(live.last_timestamp_ns, None);
        assert_pinned();
    }

    #[test]
    fn co_clocked_rois_compare_timestamps_not_local_sequences() {
        let first = ExposureKey {
            roi: RoiId(0),
            clock: SourceClock {
                domain: 7,
                epoch: 2,
            },
            sequence: 900,
            timestamp_ns: 12_000,
        };
        let second = ExposureKey {
            roi: RoiId(1),
            sequence: 3,
            ..first
        };
        assert_eq!(first.separation_ns(second), Some(0));
        assert_eq!(
            first.separation_ns(ExposureKey {
                timestamp_ns: 12_015,
                ..second
            }),
            Some(15)
        );
    }

    #[test]
    fn unrelated_or_restarted_clocks_are_not_synchronized() {
        let first = ExposureKey {
            roi: RoiId(0),
            clock: SourceClock {
                domain: 7,
                epoch: 2,
            },
            sequence: 900,
            timestamp_ns: 12_000,
        };
        for clock in [
            SourceClock {
                domain: 8,
                epoch: 2,
            },
            SourceClock {
                domain: 7,
                epoch: 3,
            },
        ] {
            assert_eq!(first.separation_ns(ExposureKey { clock, ..first }), None);
        }
    }
}
