//! Sparse scheduling of sensor windows around projected, deformable eye pivots.
//!
//! Inputs are native sensor pixels, not metric 3D coordinates. A caller projects
//! its pivot, visible iris-cap support and supported head motion at the SAME source
//! exposure. Pupil motion, held presentation contacts and host completion clocks
//! are not observations. Bounds below are engineering allowances, not confidence
//! probabilities. Scheduling does not certify a limbus or optimize SN-FEIDA.

#[derive(Clone, Copy, Debug)]
pub(crate) struct Config {
    pub sensor_size: [u32; 2],
    pub band_height: u32,
    pub roi_size: [u32; 2],
    pub capacity: usize,
    pub max_tracks: usize,
    pub prediction_ns: u64,
    pub max_age_ns: u64,
    pub reentry_ns: u64,
    pub guard_px: f64,
    pub max_prediction_px: f64,
    pub uncertainty_growth_px_s: f64,
    pub residency_bonus: f64,
    /// Preferred spare scan lines around the projected-support-centered crops.
    pub band_guard_px: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sensor_size: [8000, 6000],
            band_height: 576,
            roi_size: [420, 280],
            capacity: 2,
            max_tracks: 32,
            prediction_ns: 60_000_000,
            max_age_ns: 500_000_000,
            reentry_ns: 150_000_000,
            guard_px: 12.0,
            max_prediction_px: 80.0,
            uncertainty_growth_px_s: 40.0,
            residency_bonus: 0.35,
            band_guard_px: 64,
        }
    }
}

impl Config {
    pub fn for_geometry(sensor_size: [u32; 2], band_height: u32, roi_size: [u32; 2]) -> Self {
        Self {
            sensor_size,
            band_height,
            roi_size,
            ..Self::default()
        }
    }

    fn valid(self) -> bool {
        self.sensor_size.iter().all(|v| *v > 0 && *v <= 100_000)
            && self.roi_size.iter().all(|v| *v > 0)
            && self.roi_size[0] <= self.sensor_size[0]
            && self.roi_size[1] <= self.band_height
            && self.band_height <= self.sensor_size[1]
            && self.roi_size[0] % 4 == 0
            && self.roi_size[1] % 2 == 0
            && self.band_height % 2 == 0
            && self.band_guard_px <= self.band_height / 2
            && self.sensor_size[0] % 4 == 0
            && self.sensor_size[1] % 2 == 0
            && self.capacity > 0
            && self.capacity <= self.max_tracks
            && self.max_tracks <= 64
            && self.prediction_ns <= 250_000_000
            && self.max_age_ns > 0
            && self.max_age_ns <= 2_000_000_000
            && self.reentry_ns <= self.max_age_ns
            && [
                self.guard_px,
                self.max_prediction_px,
                self.uncertainty_growth_px_s,
                self.residency_bonus,
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1000.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PivotObservation {
    pub identity: u64,
    pub source_lineage: u64,
    pub source_timestamp_ns: u64,
    pub effective_pivot_sensor_px: [f64; 2],
    /// Current iris-cap center minus the projected globe pivot. This offset is
    /// support geometry, never a head-translation observation.
    pub support_offset_px: [f64; 2],
    /// Zero means no independently supported motion prediction, never pupil velocity.
    pub velocity_sensor_px_s: [f64; 2],
    pub uncertainty_px: [f64; 2],
    /// Half extents of required iris-cap support, NOT the entire hidden globe.
    /// The projected pivot may lie outside this support and outside the ROI.
    pub envelope_half_extent_px: [f64; 2],
    pub quality: f64,
    pub priority: f64,
}

impl PivotObservation {
    fn valid(self) -> bool {
        self.source_timestamp_ns > 0
            && self
                .effective_pivot_sensor_px
                .iter()
                .all(|x| x.is_finite() && x.abs() <= 1e6)
            && self
                .support_offset_px
                .iter()
                .all(|x| x.is_finite() && x.abs() <= 1e4)
            && self
                .velocity_sensor_px_s
                .iter()
                .all(|x| x.is_finite() && x.abs() <= 10_000.0)
            && self
                .uncertainty_px
                .iter()
                .all(|x| x.is_finite() && *x >= 0.0 && *x <= 1e4)
            && self
                .envelope_half_extent_px
                .iter()
                .all(|x| x.is_finite() && *x > 0.0 && *x <= 1e4)
            && self.quality.is_finite()
            && (0.0..=1.0).contains(&self.quality)
            && self.priority.is_finite()
            && (0.0..=10.0).contains(&self.priority)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SensorRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Residency {
    Resident,
    EvictedBand,
    EvictedCapacity,
    Stale,
    OutsideSensor,
    EnvelopeTooLarge,
    CoolingDown,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RegionDecision {
    pub identity: u64,
    pub rect: Option<SensorRect>,
    pub status: Residency,
    pub source_age_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Schedule {
    pub band_y: u32,
    pub regions: Vec<RegionDecision>,
}

#[derive(Clone, Debug)]
struct Track {
    observation: PivotObservation,
    resident: bool,
    evicted_at: Option<u64>,
    rect: Option<SensorRect>,
}

#[derive(Clone, Copy)]
struct Placement {
    x: [u32; 2],
    y: [u32; 2],
    preferred: [u32; 2],
    band: [u32; 2],
    score: f64,
}

#[derive(Debug)]
pub(crate) struct PivotRegionScheduler {
    config: Config,
    tracks: Vec<Track>,
    last_plan: Option<(u64, Schedule)>,
    plan_dirty: bool,
    last_applied_timestamp_ns: Option<u64>,
}

impl PivotRegionScheduler {
    pub fn new(config: Config) -> Result<Self, &'static str> {
        if !config.valid() {
            return Err("invalid projected-pivot scheduler configuration");
        }
        Ok(Self {
            config,
            tracks: Vec::new(),
            last_plan: None,
            plan_dirty: false,
            last_applied_timestamp_ns: None,
        })
    }

    /// Explicit clock/identity reset; ordinary crop translations must not call this.
    pub fn reset(&mut self) {
        self.tracks.clear();
        self.last_plan = None;
        self.plan_dirty = false;
        self.last_applied_timestamp_ns = None;
    }

    /// Bounded identity table: missing/evicted tracks remain addressable. Admission
    /// of a new identity at capacity requires caller-directed reset/reassociation.
    pub fn observe(&mut self, observation: PivotObservation) -> bool {
        if !observation.valid() {
            return false;
        }
        if let Some(track) = self
            .tracks
            .iter_mut()
            .find(|t| t.observation.identity == observation.identity)
        {
            if observation.source_lineage != track.observation.source_lineage
                || observation.source_timestamp_ns <= track.observation.source_timestamp_ns
            {
                return false;
            }
            track.observation = observation;
        } else {
            if self.tracks.len() == self.config.max_tracks {
                return false;
            }
            self.tracks.push(Track {
                observation,
                resident: false,
                evicted_at: None,
                rect: None,
            });
        }
        self.plan_dirty = true;
        true
    }

    /// Commit only source-keyed, camera-acknowledged membership and geometry.
    /// Proposals never earn residency credit or start eviction cooldowns. The
    /// caller may retain a fallback ROI with no pivot track; it is validated but
    /// does not become new anatomical evidence. ACK does not refresh observation
    /// timestamps, quality, motion, or uncertainty.
    pub fn acknowledge_applied(
        &mut self,
        source_timestamp_ns: u64,
        band_y: u32,
        regions: &[(u64, SensorRect)],
    ) -> bool {
        let c = self.config;
        if source_timestamp_ns == 0
            || self
                .last_applied_timestamp_ns
                .is_some_and(|t| source_timestamp_ns <= t)
            || band_y % 2 != 0
            || band_y > c.sensor_size[1] - c.band_height
            || regions.len() > c.capacity
            || regions.iter().enumerate().any(|(i, (id, r))| {
                regions[..i].iter().any(|(other, _)| other == id)
                    || r.width != c.roi_size[0]
                    || r.height != c.roi_size[1]
                    || r.x % 4 != 0
                    || r.y % 2 != 0
                    || r.x > c.sensor_size[0] - r.width
                    || r.y < band_y
                    || r.y > band_y + c.band_height - r.height
            })
        {
            return false;
        }
        for track in &mut self.tracks {
            let applied = regions
                .iter()
                .find(|(id, _)| *id == track.observation.identity);
            if let Some((_, rect)) = applied {
                track.resident = true;
                track.evicted_at = None;
                track.rect = Some(*rect);
            } else {
                if track.resident {
                    track.evicted_at = Some(source_timestamp_ns);
                }
                track.resident = false;
            }
        }
        self.last_applied_timestamp_ns = Some(source_timestamp_ns);
        self.plan_dirty = true;
        true
    }

    fn placement(&self, track: &Track, now: u64) -> Result<Placement, Residency> {
        let c = self.config;
        let o = track.observation;
        let age = now
            .checked_sub(o.source_timestamp_ns)
            .ok_or(Residency::Stale)?;
        if age > c.max_age_ns || o.quality <= 0.0 {
            return Err(Residency::Stale);
        }
        if !track.resident
            && track
                .evicted_at
                .is_some_and(|t| now.saturating_sub(t) < c.reentry_ns)
        {
            return Err(Residency::CoolingDown);
        }
        let elapsed = age as f64 * 1e-9;
        let horizon = age.saturating_add(c.prediction_ns) as f64 * 1e-9;
        let mut ranges = [[0; 2]; 2];
        let mut preferred = [0; 2];
        for axis in 0..2 {
            let delta = o.velocity_sensor_px_s[axis] * horizon;
            // Saturating the displacement would understate where the eye can
            // be. Unsupported extrapolation expires instead of becoming a
            // falsely precise, stationary head prediction.
            if delta.abs() > c.max_prediction_px {
                return Err(Residency::Stale);
            }
            let extent = o.envelope_half_extent_px[axis]
                + o.uncertainty_px[axis]
                + c.guard_px
                + elapsed * c.uncertainty_growth_px_s;
            // Cover the swept interval, not only its endpoint. Head translation
            // may move the pivot; rotating the gaze does not drag the ROI.
            let support_center = o.effective_pivot_sensor_px[axis] + o.support_offset_px[axis];
            let lo = support_center + delta.min(0.0) - extent;
            let hi = support_center + delta.max(0.0) + extent;
            if lo < 0.0 || hi > c.sensor_size[axis] as f64 {
                return Err(Residency::OutsideSensor);
            }
            let align = if axis == 0 { 4.0 } else { 2.0 };
            let lower = ((hi - c.roi_size[axis] as f64).max(0.0) / align).ceil() * align;
            let upper =
                (lo.min((c.sensor_size[axis] - c.roi_size[axis]) as f64) / align).floor() * align;
            if lower > upper {
                return Err(Residency::EnvelopeTooLarge);
            }
            ranges[axis] = [lower as u32, upper as u32];
            // Prefer the pivot, shifting only as far toward the cap as needed
            // to retain its guarded support. Gaze offset never becomes velocity.
            let center = o.effective_pivot_sensor_px[axis] + delta * 0.5 - c.roi_size[axis] as f64 * 0.5;
            preferred[axis] = ((center / align).round() * align).clamp(lower, upper) as u32;
            if let Some(rect) = track.rect {
                let previous = if axis == 0 { rect.x } else { rect.y };
                // Eight-pixel deadband only while all guarded support still fits.
                if previous >= ranges[axis][0]
                    && previous <= ranges[axis][1]
                    && previous.abs_diff(preferred[axis]) <= 8
                {
                    preferred[axis] = previous;
                }
            }
        }
        Ok(Placement {
            x: ranges[0],
            y: ranges[1],
            preferred,
            band: [
                ranges[1][0].saturating_sub(c.band_height - c.roi_size[1]),
                ranges[1][1].min(c.sensor_size[1] - c.band_height),
            ],
            score: 1.0
                + o.priority
                + o.quality * (1.0 - age as f64 / c.max_age_ns as f64)
                + if track.resident {
                    c.residency_bonus
                } else {
                    0.0
                },
        })
    }

    /// Source time must advance even when no ROI has a fresh observation. A
    /// repeated/reordered scheduling call cannot renew ages. A new observation
    /// for another identity at the same exposure invalidates the cached plan.
    pub fn plan(&mut self, source_timestamp_ns: u64, current_band_y: u32) -> Schedule {
        if let Some((last, result)) = &self.last_plan {
            if source_timestamp_ns < *last || (source_timestamp_ns == *last && !self.plan_dirty) {
                return result.clone();
            }
        }
        let c = self.config;
        let current = current_band_y.min(c.sensor_size[1] - c.band_height) & !1;
        let placements: Vec<_> = self
            .tracks
            .iter()
            .map(|t| self.placement(t, source_timestamp_ns))
            .collect();
        let mut origins = vec![current];
        for p in placements.iter().filter_map(|p| p.as_ref().ok()) {
            origins.extend(p.band);
            origins.push(p.preferred[1].saturating_sub((c.band_height - c.roi_size[1]) / 2) & !1);
        }
        let top = placements
            .iter()
            .filter_map(|p| p.as_ref().ok().map(|p| p.preferred[1]))
            .min();
        let bottom = placements
            .iter()
            .filter_map(|p| p.as_ref().ok().map(|p| p.preferred[1] + c.roi_size[1]))
            .max();
        if let (Some(top), Some(bottom)) = (top, bottom) {
            origins.push(
                ((top + bottom).saturating_sub(c.band_height) / 2)
                    .min(c.sensor_size[1] - c.band_height)
                    & !1,
            );
        }
        origins.sort_unstable();
        origins.dedup();
        let mut best_score = -1.0;
        let mut best_band = current;
        let mut best_indices = Vec::new();
        let mut best_headroom = i64::MIN;
        for band in origins {
            if band > c.sensor_size[1] - c.band_height {
                continue;
            }
            let mut eligible: Vec<_> = placements
                .iter()
                .enumerate()
                .filter_map(|(i, p)| {
                    p.as_ref()
                        .ok()
                        .filter(|p| band >= p.band[0] && band <= p.band[1])
                        .map(|p| (i, p.score))
                })
                .collect();
            eligible.sort_by(|a, b| {
                b.1.total_cmp(&a.1).then_with(|| {
                    self.tracks[a.0]
                        .observation
                        .identity
                        .cmp(&self.tracks[b.0].observation.identity)
                })
            });
            eligible.truncate(c.capacity);
            let score: f64 = eligible.iter().map(|p| p.1).sum();
            // Preserve the current band while its preferred crops have guard
            // on both sides. Once that reserve is spent, move BEFORE physical
            // clipping. Capping this secondary objective supplies a deadband;
            // residency/priority always outrank additional spare scan lines.
            let headroom = eligible
                .iter()
                .map(|(i, _)| {
                    let p = placements[*i].as_ref().unwrap();
                    (p.preferred[1] as i64 - band as i64).min(
                        (band + c.band_height) as i64 - (p.preferred[1] + c.roi_size[1]) as i64,
                    )
                })
                .min()
                .unwrap_or(c.band_guard_px as i64)
                .min(c.band_guard_px as i64);
            if score > best_score + 1e-9
                || ((score - best_score).abs() <= 1e-9
                    && (headroom > best_headroom
                        || (headroom == best_headroom
                            && band.abs_diff(current) < best_band.abs_diff(current))))
            {
                best_score = score;
                best_headroom = headroom;
                best_band = band;
                best_indices = eligible.iter().map(|p| p.0).collect();
            }
        }
        let mut regions = Vec::with_capacity(self.tracks.len());
        for (i, track) in self.tracks.iter().enumerate() {
            let selected = best_indices.contains(&i);
            let status = if selected {
                Residency::Resident
            } else {
                match placements[i] {
                    Err(reason) => reason,
                    Ok(p) if best_band < p.band[0] || best_band > p.band[1] => {
                        Residency::EvictedBand
                    }
                    Ok(_) => Residency::EvictedCapacity,
                }
            };
            let rect = if selected {
                let p = placements[i].as_ref().unwrap();
                let lower = p.y[0].max(best_band);
                let upper = p.y[1].min(best_band + c.band_height - c.roi_size[1]);
                Some(SensorRect {
                    x: p.preferred[0].clamp(p.x[0], p.x[1]),
                    y: p.preferred[1].clamp(lower, upper),
                    width: c.roi_size[0],
                    height: c.roi_size[1],
                })
            } else {
                None
            };
            regions.push(RegionDecision {
                identity: track.observation.identity,
                rect,
                status,
                source_age_ns: source_timestamp_ns
                    .saturating_sub(track.observation.source_timestamp_ns),
            });
        }
        let result = Schedule {
            band_y: best_band,
            regions,
        };
        self.last_plan = Some((source_timestamp_ns, result.clone()));
        self.plan_dirty = false;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observation(id: u64, x: f64, y: f64, time: u64) -> PivotObservation {
        PivotObservation {
            identity: id,
            source_lineage: 7,
            source_timestamp_ns: time,
            effective_pivot_sensor_px: [x, y],
            support_offset_px: [0.0; 2],
            velocity_sensor_px_s: [0.0, 0.0],
            uncertainty_px: [4.0, 4.0],
            envelope_half_extent_px: [65.0, 55.0],
            quality: 0.8,
            priority: 0.0,
        }
    }
    fn scheduler() -> PivotRegionScheduler {
        PivotRegionScheduler::new(Config::default()).unwrap()
    }
    fn resident(result: &Schedule) -> usize {
        result
            .regions
            .iter()
            .filter(|r| r.status == Residency::Resident)
            .count()
    }
    fn apply(s: &mut PivotRegionScheduler, time: u64, plan: &Schedule) {
        let regions: Vec<_> = plan
            .regions
            .iter()
            .filter_map(|r| r.rect.map(|rect| (r.identity, rect)))
            .collect();
        assert!(s.acknowledge_applied(time, plan.band_y, &regions));
    }
    fn validate(result: &Schedule) {
        assert_eq!(result.band_y % 2, 0);
        assert!(result.band_y + 576 <= 6000);
        for r in &result.regions {
            assert_eq!(r.rect.is_some(), r.status == Residency::Resident);
            if let Some(rect) = r.rect {
                assert_eq!(rect.x % 4, 0);
                assert_eq!(rect.y % 2, 0);
                assert!(rect.x + rect.width <= 8000);
                assert!(rect.y >= result.band_y);
                assert!(rect.y + rect.height <= result.band_y + 576);
            }
        }
    }
    #[test]
    fn translations_use_sensor_headroom_not_old_band_clamping() {
        let mut s = scheduler();
        for frame in 1..26 {
            let time = frame * 20_000_000;
            for id in 0..2 {
                assert!(s.observe(observation(
                    id,
                    3100.0 + id as f64 * 700.0 + frame as f64 * 12.0,
                    2200.0 + frame as f64 * 25.0,
                    time
                )));
            }
            let p = s.plan(time, 2100);
            assert_eq!(resident(&p), 2);
            validate(&p);
            if frame == 25 {
                assert!(p.band_y > 2200);
            }
        }
    }

    #[test]
    fn band_recenters_before_guarded_crops_reach_physical_edge() {
        let mut s = scheduler();
        let mut band = 2200;
        s.observe(observation(0, 3000.0, 2488.0, 1));
        assert_eq!(s.plan(1, band).band_y, band);
        let mut moved = false;
        for i in 1..20 {
            let y = 2488.0 + i as f64 * 8.0;
            s.observe(observation(0, 3000.0, y, 1 + i * 10_000_000));
            let p = s.plan(1 + i * 10_000_000, band);
            if p.band_y != band {
                // Even the centered complete ROI still fits the OLD band:
                // this movement deliberately spends larger sensor headroom.
                assert!(y + 140.0 < (band + 576) as f64);
                assert!(y - 140.0 > band as f64);
                validate(&p);
                moved = true;
                break;
            }
            band = p.band_y;
        }
        assert!(moved);
    }
    #[test]
    fn opposing_tilt_explicitly_evicts_then_reinstates_same_identity() {
        let mut s = scheduler();
        s.observe(observation(0, 3000.0, 2400.0, 1));
        s.observe(observation(1, 3800.0, 2450.0, 1));
        let initial = s.plan(1, 2200);
        assert_eq!(resident(&initial), 2);
        apply(&mut s, 1, &initial);
        s.observe(observation(0, 3000.0, 2100.0, 20_000_001));
        s.observe(observation(1, 3800.0, 2800.0, 20_000_001));
        let p = s.plan(20_000_001, 2200);
        validate(&p);
        assert_eq!(resident(&p), 1);
        apply(&mut s, 20_000_001, &p);
        let lost = p
            .regions
            .iter()
            .find(|r| r.status != Residency::Resident)
            .unwrap()
            .identity;
        assert_eq!(p.regions[lost as usize].status, Residency::EvictedBand);
        s.observe(observation(0, 3000.0, 2400.0, 40_000_001));
        s.observe(observation(1, 3800.0, 2450.0, 40_000_001));
        let p = s.plan(40_000_001, p.band_y);
        assert_eq!(p.regions[lost as usize].status, Residency::CoolingDown);
        s.observe(observation(0, 3000.0, 2400.0, 200_000_001));
        s.observe(observation(1, 3800.0, 2450.0, 200_000_001));
        assert_eq!(resident(&s.plan(200_000_001, p.band_y)), 2);
        assert_eq!(s.tracks.len(), 2);
    }
    #[test]
    fn swept_predictive_envelope_covers_current_and_future_pivot() {
        let mut s = scheduler();
        let mut o = observation(0, 3000.0, 2400.0, 1);
        o.velocity_sensor_px_s = [1000.0, -400.0];
        s.observe(o);
        let p = s.plan(1, 2200);
        let r = p.regions[0].rect.unwrap();
        assert!(r.x as f64 <= 3000.0 - 81.0);
        assert!((r.x + r.width) as f64 >= 3060.0 + 81.0);
        assert!(r.y as f64 <= 2376.0 - 71.0);
        assert!((r.y + r.height) as f64 >= 2400.0 + 71.0);
        validate(&p);
    }

    #[test]
    fn excessive_extrapolation_expires_instead_of_saturating_motion() {
        let mut s = scheduler();
        let mut o = observation(0, 3000.0, 2400.0, 1);
        o.velocity_sensor_px_s = [1000.0, 0.0];
        s.observe(o);
        assert_eq!(resident(&s.plan(1, 2200)), 1);
        let p = s.plan(100_000_001, 2200);
        assert_eq!(p.regions[0].status, Residency::Stale);
        assert_eq!(p.band_y, 2200);
    }
    #[test]
    fn jitter_does_not_churn_rois_or_band() {
        let mut s = scheduler();
        s.observe(observation(0, 3000.0, 2400.0, 1));
        let first = s.plan(1, 2200);
        apply(&mut s, 1, &first);
        for i in 1..30 {
            let jitter = if i % 2 == 0 { 1.0 } else { -1.0 };
            s.observe(observation(
                0,
                3000.0 + jitter,
                2400.0 + jitter,
                1 + i * 10_000_000,
            ));
            let p = s.plan(1 + i * 10_000_000, first.band_y);
            assert_eq!(p.band_y, first.band_y);
            assert_eq!(p.regions[0].rect, first.regions[0].rect);
        }
    }
    #[test]
    fn missing_tracks_expire_without_motion_or_identity_loss() {
        let mut s = scheduler();
        s.observe(observation(7, 3000.0, 2400.0, 1));
        let first = s.plan(1, 2200);
        apply(&mut s, 1, &first);
        let expired = s.plan(600_000_001, first.band_y);
        assert_eq!(expired.band_y, first.band_y);
        assert_eq!(expired.regions[0].status, Residency::Stale);
        assert_eq!(expired.regions[0].source_age_ns, 600_000_000);
        apply(&mut s, 600_000_001, &expired);
        assert_eq!(s.tracks.len(), 1);
        s.observe(observation(7, 3010.0, 2400.0, 800_000_001));
        assert_eq!(resident(&s.plan(800_000_001, first.band_y)), 1);
    }
    #[test]
    fn timestamps_and_lineage_cannot_refresh_or_rewind() {
        let mut s = scheduler();
        let o = observation(0, 3000.0, 2400.0, 100);
        assert!(s.observe(o));
        assert!(!s.observe(o));
        let first = s.plan(100, 2200);
        assert_eq!(s.plan(99, 0), first);
        let mut old = o;
        old.source_timestamp_ns = 99;
        assert!(!s.observe(old));
        old.source_lineage = 8;
        old.source_timestamp_ns = 101;
        assert!(!s.observe(old));
        s.reset();
        assert!(s.observe(old));
        assert_eq!(resident(&s.plan(101, 2200)), 1);
    }
    #[test]
    fn invalid_input_cannot_poison_existing_support() {
        let mut s = scheduler();
        let o = observation(0, 3000.0, 2400.0, 1);
        s.observe(o);
        let mut invalid = o;
        invalid.source_timestamp_ns = 2;
        invalid.effective_pivot_sensor_px[0] = f64::NAN;
        assert!(!s.observe(invalid));
        invalid = o;
        invalid.quality = f64::INFINITY;
        assert!(!s.observe(invalid));
        invalid = o;
        invalid.uncertainty_px[0] = -1.0;
        assert!(!s.observe(invalid));
        assert_eq!(resident(&s.plan(2, 2200)), 1);
        assert!(PivotRegionScheduler::new(Config {
            band_height: 200,
            ..Config::default()
        })
        .is_err());
    }
    #[test]
    fn clipping_and_impossible_envelopes_abstain_explicitly() {
        let mut s = scheduler();
        s.observe(observation(0, 10.0, 2400.0, 1));
        let mut large = observation(1, 3800.0, 2400.0, 1);
        large.envelope_half_extent_px[1] = 150.0;
        s.observe(large);
        let p = s.plan(1, 2200);
        assert_eq!(p.regions[0].status, Residency::OutsideSensor);
        assert_eq!(p.regions[1].status, Residency::EnvelopeTooLarge);
        assert_eq!(p.band_y, 2200);
        assert_eq!(resident(&p), 0);
    }
    #[test]
    fn n_regions_honor_capacity_priority_and_residency() {
        let mut s = scheduler();
        for id in 0..4 {
            s.observe(observation(id, 3000.0 + id as f64 * 500.0, 2400.0, 1));
        }
        let p = s.plan(1, 2200);
        assert_eq!(resident(&p), 2);
        apply(&mut s, 1, &p);
        assert_eq!(p.regions[2].status, Residency::EvictedCapacity);
        for id in 0..4 {
            let mut o = observation(id, 3000.0 + id as f64 * 500.0, 2400.0, 2);
            o.quality = if id < 2 { 0.7 } else { 0.9 };
            s.observe(o);
        }
        let p = s.plan(2, 2200);
        assert_eq!(p.regions[0].status, Residency::Resident);
        let mut o = observation(3, 4500.0, 2400.0, 3);
        o.priority = 2.0;
        s.observe(o);
        let p = s.plan(3, 2200);
        assert_eq!(p.regions[3].status, Residency::Resident);
        let mut c = Config::default();
        c.capacity = 4;
        let mut s = PivotRegionScheduler::new(c).unwrap();
        for id in 0..4 {
            s.observe(observation(id, 3000.0 + id as f64 * 500.0, 2400.0, 1));
        }
        assert_eq!(resident(&s.plan(1, 2200)), 4);
    }
    #[test]
    fn far_sensor_edges_stay_aligned_and_bounded() {
        for (x, y, band) in [(82.0, 72.0, 0), (7918.0, 5928.0, 5424)] {
            let mut s = scheduler();
            s.observe(observation(0, x, y, 1));
            let p = s.plan(1, band);
            assert_eq!(resident(&p), 1);
            validate(&p);
        }
    }

    #[test]
    fn unacknowledged_proposals_never_earn_residency_or_cooldown() {
        let mut s = scheduler();
        for id in 0..3 {
            s.observe(observation(id, 3000.0 + id as f64 * 500.0, 2400.0, 1));
        }
        let p = s.plan(1, 2200);
        assert_eq!(resident(&p), 2);
        assert!(s
            .tracks
            .iter()
            .all(|t| !t.resident && t.rect.is_none() && t.evicted_at.is_none()));
        let mut o = observation(2, 4000.0, 2400.0, 2);
        o.quality = 0.9;
        s.observe(o);
        let p = s.plan(2, 2200);
        assert_eq!(p.regions[2].status, Residency::Resident);
        apply(&mut s, 2, &p);
        let applied: Vec<_> = s
            .tracks
            .iter()
            .map(|t| (t.resident, t.rect, t.evicted_at))
            .collect();
        s.observe(observation(2, 4000.0, 3200.0, 3));
        let _ = s.plan(3, 2200);
        assert_eq!(
            s.tracks
                .iter()
                .map(|t| (t.resident, t.rect, t.evicted_at))
                .collect::<Vec<_>>(),
            applied
        );
    }

    #[test]
    fn same_exposure_peer_observation_invalidates_cache_without_renewing_age() {
        let mut s = scheduler();
        s.observe(observation(0, 3000.0, 2400.0, 100));
        assert_eq!(resident(&s.plan(200, 2200)), 1);
        s.observe(observation(1, 3800.0, 2400.0, 100));
        let paired = s.plan(200, 2200);
        assert_eq!(resident(&paired), 2);
        assert!(paired.regions.iter().all(|r| r.source_age_ns == 100));
        assert!(!s.observe(observation(1, 3800.0, 2400.0, 100)));
        assert_eq!(s.plan(200, 0), paired);
        assert_eq!(s.plan(199, 0), paired);
    }

    #[test]
    fn acknowledgements_validate_geometry_order_and_do_not_refresh_evidence() {
        let mut s = scheduler();
        s.observe(observation(0, 3000.0, 2400.0, 1));
        let p = s.plan(1, 2200);
        let rect = p.regions[0].rect.unwrap();
        let mut bad = rect;
        bad.x += 1;
        assert!(!s.acknowledge_applied(2, p.band_y, &[(0, bad)]));
        assert!(!s.acknowledge_applied(2, p.band_y, &[(0, rect), (0, rect)]));
        assert!(s.acknowledge_applied(2, p.band_y, &[(0, rect), (99, rect)]));
        assert_eq!(s.tracks.len(), 1); // No invented support for fallback identity.
        assert!(!s.acknowledge_applied(2, p.band_y, &[]));
        assert!(!s.acknowledge_applied(1, p.band_y, &[]));
        assert!(s.tracks[0].resident);
        assert_eq!(s.tracks[0].observation.source_timestamp_ns, 1);
        assert!(s.acknowledge_applied(3, p.band_y, &[]));
        assert_eq!(s.tracks[0].evicted_at, Some(3));
        assert_eq!(
            s.plan(4, p.band_y).regions[0].status,
            Residency::CoolingDown
        );
    }

    #[test]
    fn iris_cap_fits_even_when_hidden_globe_and_pivot_do_not() {
        let mut s = scheduler();
        let mut o = observation(0, 3000.0, 2100.0, 1);
        o.support_offset_px = [0.0, 300.0];
        o.envelope_half_extent_px = [114.0, 114.0];
        s.observe(o);
        let p = s.plan(1, 2200);
        let r = p.regions[0].rect.expect("visible 228px cap fits 280px ROI");
        assert!(r.y as f64 > o.effective_pivot_sensor_px[1]);
        assert!(r.y as f64 <= 2400.0 - 130.0);
        assert!((r.y + r.height) as f64 >= 2400.0 + 130.0);
        validate(&p);
    }

    #[test]
    fn gaze_offset_does_not_translate_preferred_pivot_or_create_head_velocity() {
        let mut s = scheduler();
        let mut o = observation(0, 3000.0, 2400.0, 1);
        o.support_offset_px = [0.0, -20.0];
        s.observe(o);
        let first = s.plan(1, 2200);
        apply(&mut s, 1, &first);
        o.source_timestamp_ns = 20_000_001;
        o.support_offset_px = [0.0, 20.0];
        s.observe(o);
        let second = s.plan(20_000_001, first.band_y);
        assert_eq!(second.regions[0].rect, first.regions[0].rect);
        assert_eq!(s.tracks[0].observation.velocity_sensor_px_s, [0.0; 2]);
        o.source_timestamp_ns = 40_000_001;
        o.support_offset_px = [0.0, 100.0];
        s.observe(o);
        let third = s.plan(40_000_001, second.band_y);
        let r = third.regions[0].rect.unwrap();
        assert!(r.y > first.regions[0].rect.unwrap().y);
        assert!((r.y + r.height) as f64 >= 2500.0 + 71.0);
        assert_eq!(s.tracks[0].observation.velocity_sensor_px_s, [0.0; 2]);
    }
}
