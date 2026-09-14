//! Owned sparse packets and source-native boundary adapters for the joint solve.
//!
//! A projected ellipse is a SEARCH GUIDE, never observed arc evidence. The RAW
//! adapter searches bounded radial profiles and emits only current positive
//! intensity transitions. The SAM adapter retains actual non-flat-tired runs.
//! Neither interpolates across rejected/occluded runs to manufacture evidence.

use super::ContourFitEvidence;
use crate::geometry::Ellipse;
use crate::roi_evidence::{
    BoundaryArcObservation, BoundaryKind, BoundaryNormalObservation, BoundaryLevelSetObservation, ConicObservation, ExposureKey,
    RoiConicEvidence,
};

pub(crate) mod uncertainty;
pub(crate) mod outer_candidates;

#[derive(Clone, Debug)]
pub(crate) struct OwnedBoundaryArc {
    pub(crate) evidence_group: u32,
    pub(crate) kind: BoundaryKind,
    pub(crate) points_roi_px: Vec<(f64, f64)>,
    pub(crate) sampling_support_px: Option<Vec<f64>>,
    pub(crate) support_length_cap_px: Option<f64>,
    pub(crate) outward_normals_roi: Option<Vec<Option<BoundaryNormalObservation>>>,
    pub(crate) level_sets_roi: Option<Vec<Option<BoundaryLevelSetObservation>>>,
    pub(crate) localization_sigma_px: Option<f64>,
    pub(crate) normal_band_half_width_px: f64,
    pub(crate) detector_score: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct OwnedConicHint {
    pub(crate) kind: BoundaryKind,
    pub(crate) ellipse_roi_px: Ellipse,
    pub(crate) supporting_arc_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct OwnedRoiEvidence {
    pub(crate) exposure: ExposureKey,
    pub(crate) sensor_origin_px: [u32; 2],
    pub(crate) dimensions_px: [u32; 2],
    pub(crate) arcs: Vec<OwnedBoundaryArc>,
    pub(crate) conics: Vec<OwnedConicHint>,
    pub(crate) detail_reliability: Option<f64>,
}

pub(crate) struct PreparedRoiEvidence<'a> {
    source: &'a OwnedRoiEvidence,
    arcs: Vec<BoundaryArcObservation<'a>>,
    conics: Vec<ConicObservation<'a>>,
}

impl OwnedRoiEvidence {
    pub(crate) fn prepare(&self) -> PreparedRoiEvidence<'_> {
        PreparedRoiEvidence {
            source: self,
            arcs: self
                .arcs
                .iter()
                .map(|a| BoundaryArcObservation { support_length_cap_px: a.support_length_cap_px,
                    evidence_group: a.evidence_group,
                    kind: a.kind,
                    points_roi_px: &a.points_roi_px,
                    sampling_support_px: a.sampling_support_px.as_deref(),
                    outward_normals_roi: a.outward_normals_roi.as_deref(),
                    level_sets_roi: a.level_sets_roi.as_deref(),
                    localization_sigma_px: a.localization_sigma_px,
                    normal_band_half_width_px: Some(a.normal_band_half_width_px),
                    detector_score: a.detector_score,
                })
                .collect(),
            conics: self
                .conics
                .iter()
                .map(|c| ConicObservation {
                    kind: c.kind,
                    ellipse_roi_px: c.ellipse_roi_px,
                    supporting_arc_indices: &c.supporting_arc_indices,
                    residual_px: None,
                })
                .collect(),
        }
    }
}

impl PreparedRoiEvidence<'_> {
    pub(crate) fn evidence(&self) -> RoiConicEvidence<'_> {
        RoiConicEvidence {
            exposure: self.source.exposure,
            sensor_origin_px: self.source.sensor_origin_px,
            dimensions_px: self.source.dimensions_px,
            arcs: &self.arcs,
            conics: &self.conics,
            detail_reliability: self.source.detail_reliability,
        }
    }
}

/// Retain sparse samples of real SAM contour runs. A single connected run is
/// one correlated evidence group, not one vote per resampled pixel.
pub(crate) fn append_retained_sam_arcs(
    packet: &mut OwnedRoiEvidence,
    review: &ContourFitEvidence,
    group_base: u32,
) {
    append_retained_sam_arcs_with_direction_policy(packet, review, group_base, None);
}

/// Optional measured-contour direction experiment. Directions come from
/// neighbors within a retained run, never the completed ellipse's gradient.
/// Current RAW polarity must confirm outward direction. Their information
/// shares the existing arc budget in the joint objective.
pub(crate) fn append_retained_sam_arcs_with_direction_policy(
    packet: &mut OwnedRoiEvidence,
    review: &ContourFitEvidence,
    group_base: u32,
    raw: Option<&[u16]>,
) {
    append_retained_boundary_arcs(packet, review, group_base, BoundaryKind::OuterLimbus, raw);
}

/// Preserve the detector's retained runs for an explicitly identified boundary.
/// The completed ellipse remains only a hint; excluded gaps supply no samples.
/// Pupil use is currently an offline experiment, not additional independent RAW
/// support for the same model-selected contour.
pub(crate) fn append_retained_boundary_arcs(
    packet: &mut OwnedRoiEvidence,
    review: &ContourFitEvidence,
    group_base: u32,
    kind: BoundaryKind,
    raw: Option<&[u16]>,
) {
    let start = packet.arcs.len();
    let perimeter = &review.retained_points;
    let signed_area = perimeter
        .iter()
        .zip(perimeter.iter().cycle().skip(1))
        .take(perimeter.len())
        .map(|(a, b)| a.0 * b.1 - b.0 * a.1)
        .sum::<f64>();
    let [width, height] = packet.dimensions_px.map(|v| v as usize);
    let raw = raw
        .filter(|raw| width >= 12 && height >= 12 && width.checked_mul(height) == Some(raw.len()));
    let winding = (raw.is_some() && signed_area.is_finite() && signed_area.abs() > 1.0)
        .then(|| signed_area.signum());
    let native_jitter =
        1.5 * packet.dimensions_px[0] as f64 / crate::conic_solver::LEGACY_FIT_WIDTH as f64;
    for (run, indices) in review.conic_segments.iter().take(12).enumerate() {
        if indices.len() < 3 {
            continue;
        }
        let count = indices.len().min(16);
        let points = (0..count)
            .filter_map(|i| {
                review
                    .retained_points
                    .get(indices[i * (indices.len() - 1) / (count - 1)])
                    .copied()
            })
            .collect::<Vec<_>>();
        if points.len() < 3 {
            continue;
        }
        let normals = winding.filter(|_| points.len() == count).map(|winding| {
            (0..count)
                .map(|i| {
                    let at = i * (indices.len() - 1) / (count - 1);
                    let a = *perimeter.get(indices[at.saturating_sub(2)])?;
                    let b = *perimeter.get(indices[(at + 2).min(indices.len() - 1)])?;
                    let tangent = (b.0 - a.0, b.1 - a.1);
                    let length = tangent.0.hypot(tangent.1);
                    if !length.is_finite() || length < 1.0e-6 {
                        return None;
                    }
                    let normal = [winding * tangent.1 / length, -winding * tangent.0 / length];
                    let point = *perimeter.get(indices[at])?;
                    let inside = luma(
                        raw?,
                        width,
                        height,
                        point.0 - 3.0 * normal[0],
                        point.1 - 3.0 * normal[1],
                    )?;
                    let outside = luma(
                        raw?,
                        width,
                        height,
                        point.0 + 3.0 * normal[0],
                        point.1 + 3.0 * normal[1],
                    )?;
                    if inside >= 990.0 || outside >= 990.0 || outside - inside < 7.0 {
                        return None;
                    }
                    Some(BoundaryNormalObservation {
                        unit_outward_roi: normal,
                        angular_sigma_radians: 15.0f64
                            .to_radians()
                            .hypot((native_jitter * 2.0f64.sqrt()).atan2(length)),
                    })
                })
                .collect()
        });
        packet.arcs.push(OwnedBoundaryArc { support_length_cap_px: None, sampling_support_px: None, level_sets_roi: None,
            evidence_group: group_base + run as u32,
            kind,
            points_roi_px: points,
            outward_normals_roi: normals,
            localization_sigma_px: None,
            normal_band_half_width_px: 1.5,
            detector_score: None,
        });
    }
    packet.conics.push(OwnedConicHint {
        kind,
        ellipse_roi_px: review.ellipse,
        supporting_arc_indices: (start..packet.arcs.len()).collect(),
    });
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RawArcConfig {
    pub(crate) radial_search_px: f64,
    pub(crate) minimum_contrast_raw10: f64,
    /// A boundary-specific tissue ceiling, not a full-ROI brightness/focus
    /// score. Profiles touching brighter specular pixels are unavailable.
    pub(crate) maximum_profile_luma_raw10: Option<f64>,
    pub(crate) sampling: RawSampling,
    /// Offline experiment: integrate over the fixed search-profile footprint.
    pub(crate) profile_footprint: bool,
    /// Offline comparison: localize an existing peak within its sampled bracket.
    /// Does not average tangential pixels or associate different edge peaks.
    pub(crate) subpixel_peaks: bool,
    /// Offline comparison: measure this peak's connected half-height interval.
    /// Censored wings retain the historical band; they cannot justify tightening.
    pub(crate) connected_peak_width: bool,
    /// Offline appearance prerequisite; incomplete/glint-censored cores remain
    /// unknown. This never changes accepted peak positions or outer evidence.
    pub(crate) reject_weak_pupil_core: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum RawSampling {
    #[default]
    CfaGrid,
    /// Offline comparison: a complete 4x4 CFA window at every native pixel,
    /// instead of interpolation between windows four pixels apart.
    SlidingBox,
}

impl RawSampling {
    fn sample(self, raw: &[u16], width: usize, height: usize, x: f64, y: f64) -> Option<f64> {
        match self {
            Self::CfaGrid => luma(raw, width, height, x, y),
            Self::SlidingBox => sliding_luma(raw, width, height, x, y),
        }
    }
}

impl Default for RawArcConfig {
    fn default() -> Self {
        Self {
            radial_search_px: 8.0,
            minimum_contrast_raw10: 7.0,
            maximum_profile_luma_raw10: None,
            sampling: RawSampling::CfaGrid,
            profile_footprint: false,
            subpixel_peaks: false,
            connected_peak_width: false,
            reject_weak_pupil_core: false,
        }
    }
}

/// Shared RAW photometric policy with the SAM pupil fitter. The enclosing
/// limbus is only a sampling guide: this is not pupil evidence or a scale
/// estimate. A missing/too-small iris annulus cannot certify a glint boundary.
/// Keep the historical stride, annulus and robust estimator so moving this
/// policy out of the SAM adapter does not change its existing pupil fits.
pub(crate) fn iris_tissue_luma_ceiling(
    width: usize,
    height: usize,
    outer: Ellipse,
    sample: impl Fn(usize, usize) -> Option<f64>,
) -> Option<f64> {
    if !outer.center.0.is_finite()
        || !outer.center.1.is_finite()
        || !outer.major_radius.is_finite()
        || !outer.minor_radius.is_finite()
        || !outer.angle.is_finite()
        || outer.minor_radius <= 0.0
        || outer.major_radius < outer.minor_radius
    {
        return None;
    }
    let mut annulus = Vec::new();
    for y in (0..height).step_by(2) {
        for x in (0..width).step_by(2) {
            if !(0.55..=0.85).contains(&crate::geometry::ellipse_coordinate(
                (x as f64, y as f64),
                outer,
            )) {
                continue;
            }
            if let Some(value) = sample(x, y).filter(|v| v.is_finite()) {
                annulus.push(value);
            }
        }
    }
    if annulus.len() < 64 {
        return None;
    }
    let center = crate::conic_solver::median(annulus.clone());
    let deviation = crate::conic_solver::median(
        annulus
            .into_iter()
            .map(|value| (value - center).abs())
            .collect(),
    );
    Some(center + (4.0 * deviation).max(0.4 * center).max(12.0))
}

impl RawArcConfig {
    pub(crate) fn for_pupil(
        raw: &[u16],
        width: usize,
        height: usize,
        outer: Ellipse,
    ) -> Option<Self> {
        Self::for_pupil_with_sampling(raw, width, height, outer, RawSampling::CfaGrid)
    }

    pub(crate) fn for_pupil_with_sampling(
        raw: &[u16], width: usize, height: usize, outer: Ellipse, sampling: RawSampling,
    ) -> Option<Self> {
        if width < 12 || height < 12 || width.checked_mul(height) != Some(raw.len()) {
            return None;
        }
        let ceiling = iris_tissue_luma_ceiling(width, height, outer, |x, y| {
            sampling.sample(raw, width, height, x as f64, y as f64)
        })?;
        Some(Self {
            maximum_profile_luma_raw10: Some(ceiling),
            sampling,
            ..Self::default()
        })
    }
}

/// Average a CFA-aligned 4×4 cell (one complete quad-Bayer color cell).
/// Bilinear sampling these cell means avoids treating mosaic phase as an edge.
/// This is custom native RAW decoding/photometry, not a rendered RGB thumbnail.
pub(super) fn luma(raw: &[u16], width: usize, height: usize, x: f64, y: f64) -> Option<f64> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 1.5
        || y < 1.5
        || x > width as f64 - 6.5
        || y > height as f64 - 6.5
    {
        return None;
    }
    let gx = (x - 1.5) / 4.0;
    let gy = (y - 1.5) / 4.0;
    let ix = gx.floor() as usize;
    let iy = gy.floor() as usize;
    let wx = gx - ix as f64;
    let wy = gy - iy as f64;
    let cell = |cx: usize, cy: usize| {
        (0..4)
            .flat_map(|dy| (0..4).map(move |dx| raw[(cy * 4 + dy) * width + cx * 4 + dx] as f64))
            .sum::<f64>()
            / 16.0
    };
    Some(
        (1.0 - wy) * ((1.0 - wx) * cell(ix, iy) + wx * cell(ix + 1, iy))
            + wy * ((1.0 - wx) * cell(ix, iy + 1) + wx * cell(ix + 1, iy + 1)),
    )
}

/// Each integer crop translation selects the identical physical window. All
/// sixteen CFA pixels remain measured; there is no padding, conic projection
/// or previous-frame input. Keep the baseline border exclusion for comparison.
fn sliding_luma(raw: &[u16], width: usize, height: usize, x: f64, y: f64) -> Option<f64> {
    if width.checked_mul(height) != Some(raw.len()) || !x.is_finite() || !y.is_finite()
        || x < 1.5 || y < 1.5 || x > width as f64 - 6.5 || y > height as f64 - 6.5
    { return None; }
    let gx = x - 1.5;
    let gy = y - 1.5;
    let ix = gx.floor() as usize;
    let iy = gy.floor() as usize;
    let wx = gx - ix as f64;
    let wy = gy - iy as f64;
    let cell = |cx: usize, cy: usize| {
        (0..4).flat_map(|dy| (0..4).map(move |dx| raw[(cy + dy) * width + cx + dx] as f64))
            .sum::<f64>() / 16.0
    };
    Some((1.0 - wy) * ((1.0 - wx) * cell(ix, iy) + wx * cell(ix + 1, iy))
        + wy * ((1.0 - wx) * cell(ix, iy + 1) + wx * cell(ix + 1, iy + 1)))
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    point: (f64, f64),
    profile_index: usize,
    contrast: f64,
    width: f64,
    offset: f64,
}

/// Associate the two observed profile peaks by position, not changing contrast
/// rank. One sector still has ONE correlation group, including all its runs.
/// The four-pixel adjacent offset allowance is an engineering continuity gate;
/// it neither projects points onto the guide nor bridges a missing profile.
fn coherent_edge_runs(profiles: &[[Option<Edge>; 2]]) -> Vec<Vec<Edge>> {
    let mut tracks: [Vec<Edge>; 2] = [Vec::new(), Vec::new()];
    let mut runs = Vec::new();
    for row in profiles {
        let mut best = [None; 2];
        let mut best_count = 0;
        let mut best_cost = f64::INFINITY;
        for a in [None, Some(0), Some(1)] {
            for b in [None, Some(0), Some(1)] {
                if a.is_some() && a == b {
                    continue;
                }
                let matching = [a, b];
                let mut count = 0;
                let mut cost = 0.0;
                let mut valid = true;
                for (track, choice) in tracks.iter().zip(matching) {
                    if let Some(choice) = choice {
                        let Some((previous, current)) = track.last().zip(row[choice]) else {
                            valid = false;
                            break;
                        };
                        let delta = (current.offset - previous.offset).abs();
                        if delta > 4.0 {
                            valid = false;
                            break;
                        }
                        count += 1;
                        cost += delta * delta;
                    }
                }
                if valid && (count > best_count || (count == best_count && cost < best_cost)) {
                    best = matching;
                    best_count = count;
                    best_cost = cost;
                }
            }
        }
        let mut used = [false; 2];
        for (track, choice) in tracks.iter_mut().zip(best) {
            if let Some(choice) = choice {
                track.push(row[choice].unwrap());
                used[choice] = true;
            } else if !track.is_empty() {
                runs.push(std::mem::take(track));
            }
        }
        for (index, edge) in row.iter().enumerate() {
            if used[index] {
                continue;
            }
            if let Some(edge) = edge {
                if let Some(track) = tracks.iter_mut().find(|track| track.is_empty()) {
                    track.push(*edge);
                }
            }
        }
    }
    runs.extend(tracks.into_iter().filter(|track| !track.is_empty()));
    runs
}

/// Bounded alternative paths through observed RAW peaks. Unlike contrast-rank
/// joining, this cannot switch to a remote edge or retain a sharp reversal.
/// The angular limits are explicit engineering gates, not anatomical truth.
fn shape_edge_runs(profiles: &[[Option<Edge>; 2]]) -> Vec<Vec<Edge>> {
    fn turn(a:Edge,b:Edge,c:Edge)->f64 {
        let u=(b.point.0-a.point.0,b.point.1-a.point.1);
        let v=(c.point.0-b.point.0,c.point.1-b.point.1);
        (u.0*v.1-u.1*v.0).atan2(u.0*v.0+u.1*v.1)
    }
    fn cost(path:&[Edge])->f64 {
        path.windows(3).map(|p|turn(p[0],p[1],p[2]).powi(2)).sum()
    }
    fn rank(paths:&mut Vec<Vec<Edge>>) {
        paths.sort_by(|a,b|b.len().cmp(&a.len()).then_with(||cost(a).total_cmp(&cost(b))));
        paths.dedup_by(|a,b|a.len()==b.len() && a.iter().zip(b.iter()).all(|(a,b)|a.point==b.point));
    }
    let mut active:Vec<Vec<Edge>>=Vec::new();let mut done=Vec::new();
    for row in profiles {
        let mut next=Vec::new();
        for path in active.drain(..) {
            let mut extended=false;
            for edge in row.iter().flatten() {
                let last=*path.last().unwrap();
                if (edge.offset-last.offset).abs()>4.0 {continue;}
                if path.len()>=2 {
                    let angle=turn(path[path.len()-2],last,*edge).to_degrees();
                    if !(-20.0..=60.0).contains(&angle) {continue;}
                }
                let mut candidate=path.clone();candidate.push(*edge);next.push(candidate);extended=true;
            }
            if !extended && path.len()>=3 {done.push(path);}
        }
        // Starts are genuine peaks. Missing profiles terminate paths; no gaps
        // are interpolated and no point is projected onto the ellipse guide.
        next.extend(row.iter().flatten().map(|e|vec![*e]));
        rank(&mut next);next.truncate(16);active=next;
    }
    done.extend(active.into_iter().filter(|p|p.len()>=3));rank(&mut done);
    // The solver admits at most four alternatives for a correlation group.
    done.truncate(4);done
}

/// Offline association experiment. Smooth whole-pupil conics are proposals for
/// assigning the existing RAW peaks, never additional observations. The search
/// guide bounds this finite experiment; it does not certify anatomical truth.
/// Every available profile gets one measured peak per proposed path. Missing
/// profiles stay missing. Correlated paths keep their original sector budget.
fn conic_edge_paths(profiles: &[[Option<Edge>; 2]], guide: Ellipse, search: f64)
    -> Vec<Vec<Option<Edge>>>
{
    use crate::conic_solver::{direct_conic_fit, ellipse_residual};
    let mut quadrants=[0;4];
    for (i,row) in profiles.iter().enumerate() {
        if row.iter().any(Option::is_some) {quadrants[4*i/profiles.len()]+=1;}
    }
    if quadrants.iter().sum::<usize>()<16 || quadrants.iter().filter(|&&n|n>=3).count()<3 {
        return Vec::new(); // Partial support cannot determine this proposal.
    }
    let bounded=|e:Ellipse| {
        e.minor_radius>=4.0 && e.major_radius>=e.minor_radius && e.angle.is_finite()
            && (e.center.0-guide.center.0).hypot(e.center.1-guide.center.1)<=2.0*search
            && (e.major_radius-guide.major_radius).abs()<=2.0*search
            && (e.minor_radius-guide.minor_radius).abs()<=2.0*search
    };
    let assign=|ellipse:Ellipse| profiles.iter().map(|row|row.iter().flatten().copied()
        .min_by(|a,b|ellipse_residual(a.point,ellipse).total_cmp(&ellipse_residual(b.point,ellipse))
            .then_with(||a.offset.total_cmp(&b.offset)))).collect::<Vec<_>>();
    let mut seeds=vec![guide];
    // Radial order is invariant to changing contrast rank. Keep both genuine
    // inner/outer alternatives when distinct transitions remain plausible.
    for outer in [false,true] {
        let points=profiles.iter().filter_map(|row|row.iter().flatten()
            .min_by(|a,b|if outer {b.offset.total_cmp(&a.offset)} else {a.offset.total_cmp(&b.offset)})
            .map(|e|e.point)).collect::<Vec<_>>();
        if let Some(seed)=direct_conic_fit(&points).filter(|&e|bounded(e)) {seeds.push(seed);}
    }
    let mut paths=Vec::<(f64,Vec<Option<Edge>>)>::new();
    for mut ellipse in seeds {
        for _ in 0..4 {
            let points=assign(ellipse).into_iter().flatten().map(|e|e.point).collect::<Vec<_>>();
            let Some(next)=direct_conic_fit(&points).filter(|&e|bounded(e)) else {break;};
            ellipse=next;
        }
        let path=assign(ellipse);
        if paths.iter().any(|(_,previous)|previous.iter().zip(&path)
            .all(|(a,b)|a.map(|e|e.point)==b.map(|e|e.point))) {continue;}
        let cost=path.iter().flatten().map(|e| {
            let r=ellipse_residual(e.point,ellipse)/4.0;
            if r<=1.0 {r*r} else {2.0*r-1.0}
        }).sum::<f64>();
        paths.push((cost,path));
    }
    paths.sort_by(|a,b|a.0.total_cmp(&b.0));
    paths.into_iter().take(4).map(|(_,p)|p).collect()
}

/// At most 64 radial profiles and 17 offset candidates each. Out-of-frame,
/// saturated and non-positive profiles remain absent, not predicted points.
/// Retain separated positive edge peaks as CORRELATED alternatives in each
/// angular sector. The joint model can select a secondary plausible boundary.
pub(crate) fn append_raw_ring_arcs(
    packet: &mut OwnedRoiEvidence,
    raw: &[u16],
    guide: Ellipse,
    kind: BoundaryKind,
    group_base: u32,
    config: RawArcConfig,
) -> usize {
    append_raw_ring_arcs_with_cohesion(packet, raw, guide, kind, group_base, config, false)
}

/// Explicit offline comparison until matched corpus checks justify promotion.
pub(crate) fn append_raw_ring_arcs_with_cohesion(
    packet: &mut OwnedRoiEvidence,
    raw: &[u16],
    guide: Ellipse,
    kind: BoundaryKind,
    group_base: u32,
    config: RawArcConfig,
    coherent: bool,
) -> usize {
    append_raw_ring_arcs_paths(packet,raw,guide,kind,group_base,config,coherent,false,false,ConicPathPolicy::None)
}

/// Offline-only shape continuity candidate pending matched corpus evaluation.
pub(crate) fn append_shape_checked_raw_ring_arcs(
    packet:&mut OwnedRoiEvidence,raw:&[u16],guide:Ellipse,kind:BoundaryKind,
    group_base:u32,config:RawArcConfig,
)->usize {append_raw_ring_arcs_paths(packet,raw,guide,kind,group_base,config,false,true,false,ConicPathPolicy::None)}

/// Offline optical-measurement candidate. Average a short tangent footprint
/// before locating a subpixel RAW contrast maximum; never fit/project a curve.
pub(crate) fn append_optical_raw_ring_arcs(
    packet:&mut OwnedRoiEvidence,raw:&[u16],guide:Ellipse,kind:BoundaryKind,
    group_base:u32,config:RawArcConfig,
)->usize {append_raw_ring_arcs_paths(packet,raw,guide,kind,group_base,config,false,false,true,ConicPathPolicy::None)}

pub(crate) fn append_conic_associated_raw_ring_arcs(
    packet:&mut OwnedRoiEvidence,raw:&[u16],guide:Ellipse,kind:BoundaryKind,
    group_base:u32,config:RawArcConfig,
)->usize {append_raw_ring_arcs_paths(packet,raw,guide,kind,group_base,config,false,false,false,ConicPathPolicy::Replace)}

/// Offline comparison: keep every ranked observation and use otherwise vacant
/// alternative slots for coherent assignments of the SAME measured peaks.
/// The solver still has one information budget per original angular sector.
pub(crate) fn append_augmented_raw_ring_arcs(
    packet:&mut OwnedRoiEvidence,raw:&[u16],guide:Ellipse,kind:BoundaryKind,
    group_base:u32,config:RawArcConfig,
)->usize {append_raw_ring_arcs_paths(packet,raw,guide,kind,group_base,config,false,false,false,ConicPathPolicy::Augment)}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConicPathPolicy { None, Replace, Augment }

/// The width of one observed contrast ridge, excluding disconnected peaks.
/// Both half-height crossings must be observed. A glint gap or search limit
/// inside the interval leaves its width unknown, not artificially narrow.
fn connected_profile_width(
    profiles: &[Option<(f64, (f64, f64))>], peak: usize, step: f64,
) -> Option<f64> {
    let threshold = profiles.get(peak)?.as_ref()?.0 * 0.5;
    let mut left = peak;
    loop {
        let previous = left.checked_sub(1)?;
        if profiles[previous]?.0 < threshold { break; }
        left = previous;
    }
    let mut right = peak;
    loop {
        let next = right + 1;
        if profiles.get(next)?.as_ref()?.0 < threshold { break; }
        right = next;
    }
    Some((right - left + 1) as f64 * step)
}

fn append_distinct_arc_alternatives(arcs: &mut Vec<OwnedBoundaryArc>, candidates: Vec<OwnedBoundaryArc>) {
    for candidate in candidates {
        let same_group = |arc: &&OwnedBoundaryArc| arc.kind == candidate.kind
            && arc.evidence_group == candidate.evidence_group;
        let existing = arcs.iter().filter(same_group).collect::<Vec<_>>();
        // Match the existing solver budget. Never evict a baseline alternative
        // or manufacture an extra correlation group to admit a new path.
        if existing.len() >= 4 || existing.iter().any(|arc|
            arc.points_roi_px == candidate.points_roi_px) { continue; }
        arcs.push(candidate);
    }
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub(crate) struct PupilCoreSupport {
    pub(crate) measured_rays: usize,
    pub(crate) median_relative_contrast: Option<f64>,
}

impl PupilCoreSupport {
    pub(crate) fn observed_weak(self) -> bool {
        self.measured_rays >= 48
            && self.median_relative_contrast.is_some_and(|contrast| contrast < 0.05)
    }
}

/// Bounded native appearance diagnostic, not a pupil/eyelid classifier. The
/// guide defines sampling locations only. Censoring any of the four probes
/// makes that ray unknown; a reflection cannot manufacture a dark interior.
/// The 48/64 quorum and 5% contrast are engineering assumptions, not a CI.
pub(crate) fn pupil_core_support(raw: &[u16], width: usize, height: usize,
    guide: Ellipse, config: RawArcConfig,
) -> PupilCoreSupport {
    if width < 12 || height < 12 || width.checked_mul(height) != Some(raw.len())
        || ![guide.center.0,guide.center.1,guide.major_radius,guide.minor_radius,guide.angle]
            .into_iter().all(f64::is_finite)
        || guide.minor_radius < 4.0 || guide.major_radius < guide.minor_radius
    { return PupilCoreSupport::default(); }
    let Some(ceiling) = config.maximum_profile_luma_raw10
        .filter(|v|v.is_finite() && *v > 0.0).map(|v|v.min(990.0))
    else { return PupilCoreSupport::default(); };
    let (sine, cosine) = guide.angle.sin_cos();
    let mut contrasts = Vec::with_capacity(64);
    for index in 0..64 {
        let phase = std::f64::consts::TAU * index as f64 / 64.0;
        let values = [0.25,0.5,1.25,1.5].map(|radius| {
            let x = radius * guide.major_radius * phase.cos();
            let y = radius * guide.minor_radius * phase.sin();
            config.sampling.sample(raw,width,height,
                guide.center.0+cosine*x-sine*y,guide.center.1+sine*x+cosine*y)
                .filter(|v|v.is_finite() && *v <= ceiling)
        });
        let [Some(a),Some(b),Some(c),Some(d)] = values else {continue;};
        let core = (a+b)*0.5;
        let surround = (c+d)*0.5;
        contrasts.push((surround-core)/(0.5*(surround+core)).max(8.0));
    }
    PupilCoreSupport {measured_rays:contrasts.len(),
        median_relative_contrast:(!contrasts.is_empty()).then(||crate::conic_solver::median(contrasts))}
}

// Shared measurement kernel for the live adapter and explicit offline path
// audits. Path association never changes which native RAW peaks were measured.
fn raw_ring_profiles(raw: &[u16], width: usize, height: usize, guide: Ellipse,
    config: RawArcConfig, optical: bool,
) -> ([[Option<Edge>; 2]; 64], [(f64, f64); 64]) {
    let (sine, cosine) = guide.angle.sin_cos();
    let search = config.radial_search_px.min(12.0);
    let step = search / 8.0;
    let mut edges: [[Option<Edge>; 2]; 64] = [[None; 2]; 64];
    let mut profile_centers = [(0.0, 0.0); 64];
    for (index, result) in edges.iter_mut().enumerate() {
        let phase = std::f64::consts::TAU * index as f64 / 64.0;
        let x = guide.major_radius * phase.cos();
        let y = guide.minor_radius * phase.sin();
        let point = (
            guide.center.0 + cosine * x - sine * y,
            guide.center.1 + sine * x + cosine * y,
        );
        profile_centers[index] = point;
        let nx = phase.cos() / guide.major_radius;
        let ny = phase.sin() / guide.minor_radius;
        let norm = nx.hypot(ny);
        let normal = (
            (cosine * nx - sine * ny) / norm,
            (sine * nx + cosine * ny) / norm,
        );
        let ceiling = config.maximum_profile_luma_raw10.unwrap_or(990.0).min(990.0);
        let measure = |center:(f64,f64), side:f64| -> Option<f64> {
            let sample = |along:f64| config.sampling.sample(raw,width,height,
                center.0+side*normal.0-along*normal.1,
                center.1+side*normal.1+along*normal.0).filter(|v|*v<=ceiling);
            if optical {
                // Every sample must be uncensored; averaging cannot authorize
                // a profile that overlaps a glint or the image boundary.
                Some((sample(-4.0)?+2.0*sample(0.0)?+sample(4.0)?)*0.25)
            } else {sample(0.0)}
        };
        let mut profiles = Vec::with_capacity(17);
        for offset_index in -8..=8 {
            let offset = offset_index as f64 * step;
            let center = (point.0 + offset * normal.0, point.1 + offset * normal.1);
            let (Some(inner), Some(outer)) = (measure(center,-3.0),measure(center,3.0)) else {
                profiles.push(None);
                continue;
            };
            profiles.push(Some((outer - inner, center)));
        }
        let mut peaks = profiles
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let (contrast, point) = (*p)?;
                if contrast < config.minimum_contrast_raw10 {
                    return None;
                }
                // A maximum outside the search window is censored. Do not pin it
                // to the guide's allowed boundary and call that an observed edge.
                if i == 0 || i + 1 == profiles.len() {
                    return None;
                }
                // A censored adjacent profile is another unknown search limit,
                // not evidence that the current point is an intensity maximum.
                let (Some(before), Some(after)) = (profiles[i - 1], profiles[i + 1]) else {
                    return None;
                };
                if before.0 > contrast || after.0 >= contrast {
                    return None;
                }
                let refinement=if optical || config.subpixel_peaks {
                    // Quadratic maximum inside this observed three-sample
                    // bracket, not an ellipse or a temporal prediction.
                    (0.5*(before.0-after.0)/(before.0-2.0*contrast+after.0)).clamp(-0.5,0.5)*step
                } else {0.0};
                let supported_width = profiles
                    .iter()
                    .filter_map(|p| *p)
                    .filter(|p| p.0 >= contrast * 0.5)
                    .count() as f64
                    * step;
                let supported_width = if config.connected_peak_width {
                    connected_profile_width(&profiles, i, step).unwrap_or(supported_width)
                } else { supported_width };
                Some(Edge {
                    point:(point.0+refinement*normal.0,point.1+refinement*normal.1),
                    profile_index: index,
                    contrast,
                    width: supported_width,
                    offset: (i as f64 - 8.0) * step+refinement,
                })
            })
            .collect::<Vec<_>>();
        peaks.sort_by(|a, b| b.contrast.total_cmp(&a.contrast));
        for (slot, edge) in result.iter_mut().zip(peaks.into_iter().take(2)) {
            *slot = Some(edge);
        }
    }
    (edges, profile_centers)
}

#[cfg(test)]
pub(crate) fn raw_ring_path_audit(raw: &[u16], width: usize, height: usize,
    guide: Ellipse, config: RawArcConfig,
) -> serde_json::Value {
    assert_eq!(raw.len(), width * height);
    assert!(width >= 12 && height >= 12 && guide.minor_radius >= 4.0
        && guide.major_radius >= guide.minor_radius && guide.angle.is_finite());
    let (profiles, centers) = raw_ring_profiles(raw, width, height, guide, config, false);
    let paths = conic_edge_paths(&profiles, guide, config.radial_search_px.min(12.0));
    let edge = |e: Edge| serde_json::json!({"point":e.point,"profile":e.profile_index,
        "contrast":e.contrast,"width":e.width,"offset":e.offset});
    // Diagnostic appearance samples around the SAME search guide. These are
    // intensities, not observed boundary locations or an anatomical decision.
    // Retain above-ceiling values explicitly so censorship is inspectable.
    let radii = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5];
    let (sine, cosine) = guide.angle.sin_cos();
    let appearance = (0..64).map(|index| {
        let phase = std::f64::consts::TAU * index as f64 / 64.0;
        radii.map(|radius| {
            let x = radius * guide.major_radius * phase.cos();
            let y = radius * guide.minor_radius * phase.sin();
            let point = (guide.center.0 + cosine*x - sine*y,
                guide.center.1 + sine*x + cosine*y);
            serde_json::json!({"point":point,
                "luma":config.sampling.sample(raw,width,height,point.0,point.1)})
        })
    }).collect::<Vec<_>>();
    serde_json::json!({"profile_centers":centers.to_vec(),
        "profiles":profiles.iter().map(|row|row.map(|p|p.map(&edge))).collect::<Vec<_>>(),
        "paths":paths.iter().map(|path|path.iter().map(|p|p.map(&edge)).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "appearance":{"radial_fractions":radii,"samples":appearance,
            "tissue_ceiling":config.maximum_profile_luma_raw10},
        "core_support":pupil_core_support(raw,width,height,guide,config),
        "contract":"Existing whole-pupil paths assign the unchanged measured RAW peaks. They are detector-conditioned association hypotheses, not anatomical truth or extra observations."})
}

fn append_raw_ring_arcs_paths(
    packet:&mut OwnedRoiEvidence,raw:&[u16],guide:Ellipse,kind:BoundaryKind,
    group_base:u32,config:RawArcConfig,coherent:bool,shape:bool,optical:bool,conic_policy:ConicPathPolicy,
)->usize {
    let [width, height] = packet.dimensions_px.map(|v| v as usize);
    if width < 12
        || height < 12
        || width * height != raw.len()
        || !guide.major_radius.is_finite()
        || !guide.minor_radius.is_finite()
        || !guide.angle.is_finite()
        || guide.minor_radius < 4.0
        || guide.major_radius < guide.minor_radius
        || !config.radial_search_px.is_finite()
        || config.radial_search_px <= 0.0
        || !config.minimum_contrast_raw10.is_finite()
        || config.minimum_contrast_raw10 <= 0.0
        || config
            .maximum_profile_luma_raw10
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
    {
        return 0;
    }
    if kind == BoundaryKind::PupillaryBoundary && config.reject_weak_pupil_core
        && pupil_core_support(raw,width,height,guide,config).observed_weak()
    { return 0; }
    let (edges, profile_centers) = raw_ring_profiles(raw, width, height, guide, config, optical);
    let search = config.radial_search_px.min(12.0);
    let start = packet.arcs.len();
    let conic_paths=if conic_policy != ConicPathPolicy::None {conic_edge_paths(&edges,guide,search)} else {Vec::new()};
    let mut supplements = Vec::new();
    for sector in 0..8 {
        let append = |run: &[Edge], arcs: &mut Vec<OwnedBoundaryArc>| {
            if run.len() < 3 {
                return;
            }
            let mean_width = run.iter().map(|p| p.width).sum::<f64>() / run.len() as f64;
            let score = run.iter().map(|p| p.contrast).sum::<f64>() / run.len() as f64;
            arcs.push(OwnedBoundaryArc { support_length_cap_px: None, level_sets_roi: None,
                evidence_group: group_base + sector as u32,
                kind,
                points_roi_px: run.iter().map(|p| p.point).collect(),
                sampling_support_px: config.profile_footprint.then(|| {
                    let mut support = vec![0.0; run.len()];
                    for (i, pair) in run.windows(2).enumerate() {
                        let a = profile_centers[pair[0].profile_index];
                        let b = profile_centers[pair[1].profile_index];
                        let half = (b.0-a.0).hypot(b.1-a.1) * 0.5;
                        support[i] += half;
                        support[i+1] += half;
                    }
                    support
                }),
                outward_normals_roi: None,
                localization_sigma_px: None,
                normal_band_half_width_px: (mean_width * 0.25).clamp(0.75, 4.0),
                detector_score: Some(score),
            });
        };
        if !conic_paths.is_empty() {
            let destination = if conic_policy == ConicPathPolicy::Augment {
                &mut supplements
            } else { &mut packet.arcs };
            for path in &conic_paths {
                let mut run=Vec::new();
                for edge in &path[sector*8..(sector+1)*8] {
                    if let Some(edge)=edge {run.push(*edge);}
                    else {append(&run,destination);run.clear();}
                }
                append(&run,destination);
            }
            if conic_policy == ConicPathPolicy::Replace { continue; }
        }
        if shape {
            for run in shape_edge_runs(&edges[sector*8..(sector+1)*8]) {append(&run,&mut packet.arcs);}
            continue;
        }
        if coherent {
            for run in coherent_edge_runs(&edges[sector * 8..(sector + 1) * 8]) {
                append(&run, &mut packet.arcs);
            }
            continue;
        }
        for alternative in 0..2 {
            let mut run = Vec::new();
            // Do not cross missing-profile gaps. Each contiguous subrun in
            // this sector is an alternative for one information budget.
            let flush = |run: &mut Vec<Edge>, arcs: &mut Vec<OwnedBoundaryArc>| {
                append(run, arcs);
                run.clear();
            };
            for row in edges.iter().skip(sector * 8).take(8) {
                if let Some(edge) = row[alternative] {
                    run.push(edge);
                } else {
                    flush(&mut run, &mut packet.arcs);
                }
            }
            flush(&mut run, &mut packet.arcs);
        }
    }
    append_distinct_arc_alternatives(&mut packet.arcs, supplements);
    let count = packet.arcs.len() - start;
    if count > 0 {
        packet.conics.push(OwnedConicHint {
            kind,
            ellipse_roi_px: guide,
            supporting_arc_indices: (start..packet.arcs.len()).collect(),
        });
        // A pupil edge's optical width belongs to that edge's normal band.
        // It is NOT a full-ROI focus measurement: updating packet detail here
        // would silently reweight previously appended SAM limbus arcs whenever
        // an intermittent/reflected pupil happened to produce a RAW run.
        // Preserve any independently supplied full-ROI focus assessment.
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roi_evidence::{RoiId, SourceClock};

    #[test]
    fn pupil_peak_width_uses_only_its_observed_connected_ridge() {
        let profile = |values: &[f64]| values.iter()
            .map(|v| Some((*v, (0.0, 0.0)))).collect::<Vec<_>>();
        let separated = profile(&[0.0, 6.0, 10.0, 6.0, 0.0, 7.0, 12.0, 7.0, 0.0]);
        assert_eq!(connected_profile_width(&separated, 2, 1.0), Some(3.0));
        assert_eq!(connected_profile_width(&separated, 6, 1.5), Some(4.5));
        let broad = profile(&[0.0, 6.0, 8.0, 10.0, 8.0, 6.0, 0.0]);
        assert_eq!(connected_profile_width(&broad, 3, 1.0), Some(5.0));
    }

    #[test]
    fn pupil_peak_width_cannot_tighten_a_censored_interval() {
        let mut profiles = [0.0, 6.0, 10.0, 6.0, 0.0]
            .map(|v| Some((v, (0.0, 0.0))));
        assert_eq!(connected_profile_width(&profiles[1..], 1, 1.0), None);
        assert_eq!(connected_profile_width(&profiles[..4], 2, 1.0), None);
        profiles[0] = None;
        assert_eq!(connected_profile_width(&profiles, 2, 1.0), None);
        profiles[0] = Some((0.0, (0.0, 0.0)));
        profiles[4] = None;
        assert_eq!(connected_profile_width(&profiles, 2, 1.0), None);
    }

    #[test]
    fn wider_pupil_search_recovers_observed_edges_outside_an_inaccurate_guide() {
        let truth=Ellipse {center:(128.0,96.0),major_radius:50.0,minor_radius:50.0,angle:0.0};
        let guide=Ellipse {major_radius:40.0,minor_radius:40.0,..truth};
        let raw=(0..256*192).map(|i| {
            let radius=((i%256) as f64-truth.center.0).hypot((i/256) as f64-truth.center.1);
            (100.0+400.0*(1.0+((radius-50.0)/1.5).tanh())*0.5).round() as u16
        }).collect::<Vec<_>>();
        let mut variants=[packet(),packet()];
        for (p,radius) in variants.iter_mut().zip([8.0,12.0]) {
            append_raw_ring_arcs(p,&raw,guide,BoundaryKind::PupillaryBoundary,100,
                RawArcConfig {radial_search_px:radius,..Default::default()});
        }
        let points:Vec<Vec<_>>=variants.iter().map(|p|p.arcs.iter()
            .flat_map(|a|a.points_roi_px.iter().copied()).collect()).collect();
        assert!(points[1].len()>=48 && points[1].len()>points[0].len()+16,
            "the wider window must recover actual edge observations");
        let rms=(points[1].iter().map(|p|crate::conic_solver::ellipse_residual(*p,truth).powi(2))
            .sum::<f64>()/points[1].len() as f64).sqrt();
        assert!(rms<1.2,"localization against the known disk: {rms}");
        assert!(points[1].iter().all(|p|crate::conic_solver::ellipse_residual(*p,guide)>7.5),
            "search-guide points must never replace measured peaks");
        let mut unavailable=packet();
        append_raw_ring_arcs(&mut unavailable,&vec![100;256*192],guide,
            BoundaryKind::PupillaryBoundary,100,RawArcConfig {radial_search_px:12.0,..Default::default()});
        assert!(unavailable.arcs.is_empty() && unavailable.conics.is_empty());
        eprintln!("WIDE_PUPIL_SEARCH baseline_points={} wider_points={} wider_rms={rms}",points[0].len(),points[1].len());
    }

    #[test]
    fn subpixel_pupil_peaks_reduce_localization_error_on_translated_blurred_disks() {
        let mut squared_error=[0.0;2];let mut counts=[0;2];
        for shift in 0..12 {
            let e=Ellipse {center:(127.13+shift as f64/12.0,95.37+shift as f64/17.0),
                major_radius:56.7,minor_radius:35.4,angle:0.37};
            let (s,c)=e.angle.sin_cos();
            let raw=(0..256*192).map(|i| {
                let dx=(i%256) as f64-e.center.0;let dy=(i/256) as f64-e.center.1;
                let x=c*dx+s*dy;let y=-s*dx+c*dy;
                let distance=(x*x/e.major_radius.powi(2)+y*y/e.minor_radius.powi(2)-1.0)
                    /(2.0*(x/e.major_radius.powi(2)).hypot(y/e.minor_radius.powi(2))).max(1e-12);
                (100.0+400.0*(1.0+(distance/2.0).tanh())*0.5).round() as u16
            }).collect::<Vec<_>>();
            let mut variants=[packet(),packet()];
            for (enabled,p) in [false,true].into_iter().zip(&mut variants) {
                append_raw_ring_arcs(p,&raw,e,BoundaryKind::PupillaryBoundary,100,
                    RawArcConfig {subpixel_peaks:enabled,..Default::default()});
            }
            assert!(!variants[0].arcs.is_empty());
            assert_eq!(variants[0].arcs.len(),variants[1].arcs.len());
            for (a,b) in variants[0].arcs.iter().zip(&variants[1].arcs) {
                assert_eq!(a.points_roi_px.len(),b.points_roi_px.len());
                assert_eq!(a.evidence_group,b.evidence_group);
                assert_eq!(a.normal_band_half_width_px,b.normal_band_half_width_px);
                assert_eq!(a.detector_score,b.detector_score);
                for (p,q) in a.points_roi_px.iter().zip(&b.points_roi_px) {
                    assert!((p.0-q.0).hypot(p.1-q.1)<=0.5+1e-10);
                }
            }
            for (arm,p) in variants.iter().enumerate() {
                for point in p.arcs.iter().flat_map(|a|&a.points_roi_px) {
                    squared_error[arm]+=crate::conic_solver::ellipse_residual(*point,e).powi(2);
                    counts[arm]+=1;
                }
            }
        }
        assert_eq!(counts[0],counts[1]);
        eprintln!("SUBPIXEL_SYNTHETIC points={} rms_baseline={} rms_subpixel={}",counts[0],
            (squared_error[0]/counts[0] as f64).sqrt(),(squared_error[1]/counts[1] as f64).sqrt());
        assert!(squared_error[1]<squared_error[0],"subpixel interpolation must improve actual known boundary localization");
    }

    #[test]
    fn subpixel_pupil_peaks_preserve_glint_censorship_and_outer_evidence() {
        let e=ellipse();
        let mut raw=raw_disk();
        for y in 0..192 {for x in 0..256 {
            if x as f64>e.center.0 && (y as f64)<e.center.1 {raw[y*256+x]=1023;}
        }}
        let mut baseline=packet();
        append_retained_sam_arcs(&mut baseline,&review(e.dense_points(128)),0);
        let outer=baseline.arcs.len();let mut candidate=baseline.clone();
        for (enabled,p) in [(false,&mut baseline),(true,&mut candidate)] {
            append_raw_ring_arcs(p,&raw,e,BoundaryKind::PupillaryBoundary,100,
                RawArcConfig {subpixel_peaks:enabled,..Default::default()});
        }
        assert_eq!(format!("{:?}",&baseline.arcs[..outer]),format!("{:?}",&candidate.arcs[..outer]));
        assert_eq!(format!("{:?}",baseline.conics),format!("{:?}",candidate.conics));
        assert_eq!(baseline.arcs.len(),candidate.arcs.len());
        for (a,b) in baseline.arcs[outer..].iter().zip(&candidate.arcs[outer..]) {
            assert_eq!(a.evidence_group,b.evidence_group);
            assert_eq!(a.points_roi_px.len(),b.points_roi_px.len());
            assert_eq!(a.normal_band_half_width_px,b.normal_band_half_width_px);
            assert!(b.points_roi_px.iter().all(|p|p.0<=e.center.0 || p.1>=e.center.1));
        }
    }

    #[test]
    fn pupil_profile_footprint_preserves_raw_points_and_missing_runs() {
        let e=ellipse();
        let raw=(0..256*192).map(|i| {
            let x=(i%256) as f64-e.center.0;let y=(i/256) as f64-e.center.1;
            if x>8.0 && y < -8.0 {1023}
            else if (x/e.major_radius).hypot(y/e.minor_radius)<1.0 {150} else {550}
        }).collect::<Vec<_>>();
        let mut baseline=packet();let mut candidate=packet();
        for (packet,enabled) in [(&mut baseline,false),(&mut candidate,true)] {
            append_raw_ring_arcs(packet,&raw,e,BoundaryKind::PupillaryBoundary,100,
                RawArcConfig {profile_footprint:enabled,..Default::default()});
        }
        assert!(!candidate.arcs.is_empty());
        assert_eq!(baseline.arcs.len(),candidate.arcs.len());
        for (a,b) in baseline.arcs.iter().zip(&mut candidate.arcs) {
            let support=b.sampling_support_px.take().unwrap();
            assert_eq!(support.len(),b.points_roi_px.len());
            assert!(support.iter().all(|v|v.is_finite() && *v>0.0));
            assert_eq!(format!("{a:?}"),format!("{b:?}"));
        }
        assert_eq!(format!("{:?}",baseline.conics),format!("{:?}",candidate.conics));
    }

    #[test]
    fn pupil_sliding_photometry_is_invariant_to_native_crop_translation() {
        let raw = (0..64 * 64).map(|i| {
            let x = i % 64; let y = i / 64;
            ((x * x * 7 + y * y * 11 + x * y * 3) % 901) as u16
        }).collect::<Vec<_>>();
        let cropped = (0..48 * 48).map(|i| raw[(i / 48 + 2) * 64 + i % 48 + 2])
            .collect::<Vec<_>>();
        let mut baseline_difference = 0.0_f64;
        for y in 10..35 { for x in 10..35 {
            let (x, y) = (x as f64 + 0.25, y as f64 + 0.75);
            let original = sliding_luma(&raw, 64, 64, x, y).unwrap();
            let translated = sliding_luma(&cropped, 48, 48, x - 2.0, y - 2.0).unwrap();
            assert_eq!(original.to_bits(), translated.to_bits());
            baseline_difference = baseline_difference.max((luma(&raw, 64, 64, x, y).unwrap()
                - luma(&cropped, 48, 48, x - 2.0, y - 2.0).unwrap()).abs());
        }}
        assert!(baseline_difference > 1.0, "the control must expose crop-grid phase dependence");
        for (x,y) in [(f64::NAN, 12.0), (1.0, 12.0), (63.0, 12.0), (12.0, -1.0)] {
            assert!(sliding_luma(&raw, 64, 64, x, y).is_none());
        }
        assert!(sliding_luma(&raw[..10], 64, 64, 12.0, 12.0).is_none());
    }

    #[test]
    fn pupil_conic_association_keeps_raw_peaks_across_rank_changes_without_filling_gaps() {
        let guide=Ellipse {center:(100.0,100.0),major_radius:40.0,minor_radius:25.0,angle:0.0};
        let profiles=(0..64).map(|i| {
            let angle=std::f64::consts::TAU*i as f64/64.0;
            let a=Edge {profile_index: 0,point:(100.0+40.0*angle.cos(),100.0+25.0*angle.sin()),
                offset:0.0,contrast:15.0,width:2.0};
            let b=Edge {profile_index: 0,point:(100.0+45.0*angle.cos(),100.0+30.0*angle.sin()),
                offset:5.0,contrast:20.0,width:2.0};
            if i%2==0 {[Some(a),Some(b)]} else {[Some(b),Some(a)]}
        }).collect::<Vec<_>>();
        let paths=conic_edge_paths(&profiles,guide,8.0);
        assert!(paths.iter().any(|p|p.iter().flatten().all(|e|e.offset==0.0)));
        assert!(paths.iter().any(|p|p.iter().flatten().all(|e|e.offset==5.0)));
        for path in &paths {
            for (edge,row) in path.iter().zip(&profiles) {
                assert!(row.iter().flatten().any(|e|Some(e.point)==edge.map(|e|e.point)));
            }
        }
        let coordinates=|paths:Vec<Vec<Option<Edge>>>|paths.into_iter()
            .map(|p|p.into_iter().map(|e|e.map(|e|e.point)).collect::<Vec<_>>()).collect::<Vec<_>>();
        let swapped=profiles.iter().map(|p|[p[1],p[0]]).collect::<Vec<_>>();
        assert_eq!(coordinates(paths),coordinates(conic_edge_paths(&swapped,guide,8.0)));
        let mut missing=profiles.clone();missing[10]=[None,None];
        for path in conic_edge_paths(&missing,guide,8.0) {
            assert!(path[10].is_none());
            assert_eq!(path.iter().flatten().count(),63);
        }
        for row in &mut missing[12..] {*row=[None,None];}
        assert!(conic_edge_paths(&missing,guide,8.0).is_empty(),"insufficient coverage leaves baseline evidence intact");

        let arc = |group, points| OwnedBoundaryArc {
            evidence_group: group, kind: BoundaryKind::PupillaryBoundary,
            points_roi_px: points, outward_normals_roi: None, level_sets_roi: None,
            localization_sigma_px: None, support_length_cap_px: None, sampling_support_px: None,
            normal_band_half_width_px: 1.0, detector_score: None,
        };
        let baseline = (0..8).flat_map(|sector| {
            let profiles = &profiles;
            (0..2).map(move |rank| arc(100 + sector as u32,
                profiles[sector*8..(sector+1)*8].iter().map(|p|p[rank].unwrap().point).collect()))
        }).collect::<Vec<_>>();
        let paths = conic_edge_paths(&profiles,guide,8.0);
        let supplements = (0..8).flat_map(|sector| paths.iter().map(move |path|
            arc(100 + sector as u32,path[sector*8..(sector+1)*8].iter().flatten().map(|e|e.point).collect())))
            .collect::<Vec<_>>();
        let mut augmented = baseline.clone();
        append_distinct_arc_alternatives(&mut augmented,supplements.clone());
        assert_eq!(format!("{:?}",&augmented[..baseline.len()]),format!("{baseline:?}"),
            "every original measured path must remain available at its original index");
        assert!(augmented.len()>baseline.len(),"rank swaps must admit additional coherent paths");
        for sector in 0..8 {
            let group=augmented.iter().filter(|a|a.evidence_group==100+sector as u32).collect::<Vec<_>>();
            assert_eq!(group.len(),4,"one sector budget, with at most four alternatives");
            for candidate in group {
                for &point in &candidate.points_roi_px {
                    assert!(profiles[sector*8..(sector+1)*8].iter().flatten().flatten().any(|e|e.point==point),
                        "a coherent assignment never invents an observed position");
                }
            }
        }
        let once=format!("{augmented:?}");
        append_distinct_arc_alternatives(&mut augmented,supplements);
        assert_eq!(format!("{augmented:?}"),once,"duplicate proposals cannot create more evidence");
    }

    #[test]
    fn shape_paths_follow_observed_smooth_edges_across_rank_swaps_and_stop_at_gaps() {
        let good=(0..8).map(|i|Edge {profile_index: 0,point:(3.0*i as f64,0.08*(i*i) as f64),
            contrast:20.0,width:2.0,offset:0.0}).collect::<Vec<_>>();
        let profiles=good.iter().enumerate().map(|(i,e)| {
            let mut bad=*e;bad.point.1+=if i%2==0 {4.0} else {-4.0};bad.offset=if i%2==0 {4.0} else {-4.0};
            if i%2==0 {[Some(bad),Some(*e)]} else {[Some(*e),Some(bad)]}
        }).collect::<Vec<_>>();
        let paths=shape_edge_runs(&profiles);
        assert!(paths.iter().any(|p|p.len()==8 && p.iter().zip(&good).all(|(a,b)|a.point==b.point)));
        for path in &paths {for p in path {assert!(profiles.iter().flatten().flatten().any(|e|e.point==p.point));}}
        let mut gap=profiles.clone();gap[4]=[None,None];
        for path in shape_edge_runs(&gap) {
            assert!(!path.iter().any(|e|e.point.0<12.0)||!path.iter().any(|e|e.point.0>12.0));
            for p in path.windows(3) {
                let a=(p[1].point.0-p[0].point.0,p[1].point.1-p[0].point.1);
                let b=(p[2].point.0-p[1].point.0,p[2].point.1-p[1].point.1);
                let turn=(a.0*b.1-a.1*b.0).atan2(a.0*b.0+a.1*b.1).to_degrees();
                assert!((-20.0..=60.0).contains(&turn));
            }
        }
    }

    fn packet() -> OwnedRoiEvidence {
        OwnedRoiEvidence {
            exposure: ExposureKey {
                roi: RoiId(1),
                clock: SourceClock {
                    domain: 1,
                    epoch: 1,
                },
                sequence: 1,
                timestamp_ns: 1,
            },
            sensor_origin_px: [0, 0],
            dimensions_px: [256, 192],
            arcs: Vec::new(),
            conics: Vec::new(),
            detail_reliability: None,
        }
    }
    fn ellipse() -> Ellipse {
        Ellipse {
            center: (128.0, 96.0),
            major_radius: 60.0,
            minor_radius: 46.0,
            angle: 0.0,
        }
    }

    #[test]
    fn weak_pupil_core_rejects_a_local_dark_ring_without_reweighting_outer_arcs() {
        let guide=Ellipse {major_radius:30.0,minor_radius:24.0,..ellipse()};
        let raw=(0..256*192).map(|i| {
            let r=crate::geometry::ellipse_coordinate(((i%256) as f64,(i/256) as f64),guide);
            if (0.84..1.0).contains(&r) {50} else {150}
        }).collect::<Vec<u16>>();
        let baseline=RawArcConfig {maximum_profile_luma_raw10:Some(500.0),..Default::default()};
        let candidate=RawArcConfig {reject_weak_pupil_core:true,..baseline};
        let support=pupil_core_support(&raw,256,192,guide,candidate);
        assert_eq!(support.measured_rays,64);
        assert!(support.observed_weak());
        let mut a=packet();let mut b=packet();
        append_raw_ring_arcs(&mut a,&raw,guide,BoundaryKind::OuterLimbus,0,baseline);
        append_raw_ring_arcs(&mut b,&raw,guide,BoundaryKind::OuterLimbus,0,candidate);
        assert!(!a.arcs.is_empty());assert_eq!(format!("{a:?}"),format!("{b:?}"));
        let outer=format!("{b:?}");
        assert!(append_raw_ring_arcs(&mut a,&raw,guide,BoundaryKind::PupillaryBoundary,100,baseline)>0);
        assert_eq!(append_raw_ring_arcs(&mut b,&raw,guide,BoundaryKind::PupillaryBoundary,100,candidate),0);
        assert_eq!(format!("{b:?}"),outer,"reject only the new pupil family, including its hint");
    }

    #[test]
    fn pupil_core_keeps_strong_disks_and_treats_reflected_cores_as_unknown() {
        let guide=Ellipse {major_radius:30.0,minor_radius:24.0,..ellipse()};
        let baseline=RawArcConfig {maximum_profile_luma_raw10:Some(500.0),..Default::default()};
        let candidate=RawArcConfig {reject_weak_pupil_core:true,..baseline};
        for glint in [false,true] {
            let raw=(0..256*192).map(|i| {
                let r=crate::geometry::ellipse_coordinate(((i%256) as f64,(i/256) as f64),guide);
                if glint && r<0.65 {900} else if r<1.0 {50} else {150}
            }).collect::<Vec<u16>>();
            let support=pupil_core_support(&raw,256,192,guide,candidate);
            assert!(!support.observed_weak());
            if glint {assert_eq!(support.measured_rays,0);assert!(support.median_relative_contrast.is_none());}
            else {
                assert_eq!(support.measured_rays,64);
                assert!(support.median_relative_contrast.unwrap()>0.9);
                let doubled=raw.iter().map(|v|v*2).collect::<Vec<_>>();
                let other=pupil_core_support(&doubled,256,192,guide,
                    RawArcConfig {maximum_profile_luma_raw10:Some(1000.0),..candidate});
                assert!((other.median_relative_contrast.unwrap()-support.median_relative_contrast.unwrap()).abs()<1e-12);
            }
            let mut a=packet();let mut b=packet();
            append_raw_ring_arcs(&mut a,&raw,guide,BoundaryKind::PupillaryBoundary,100,baseline);
            append_raw_ring_arcs(&mut b,&raw,guide,BoundaryKind::PupillaryBoundary,100,candidate);
            assert!(!a.arcs.is_empty());assert_eq!(format!("{a:?}"),format!("{b:?}"));
        }
        assert!(!pupil_core_support(&[],256,192,guide,candidate).observed_weak());
        assert!(!pupil_core_support(&vec![100;256*192],256,192,guide,
            RawArcConfig::default()).observed_weak(),"no tissue ceiling is unknown");
    }
    fn review(points: Vec<(f64, f64)>) -> ContourFitEvidence {
        let count = points.len();
        ContourFitEvidence {
            ellipse: ellipse(),
            source_component_area_px: 0.0,
            retained_points: std::sync::Arc::new(points),
            conic_segments: std::sync::Arc::new(vec![(0..count).collect()]),
            flat_tire_points: std::sync::Arc::new(Vec::new()),
            upper_flat_tire: false,
            lower_flat_tire: false,
        }
    }
    fn raw_disk() -> Vec<u16> {
        (0..192)
            .flat_map(|y| {
                (0..256).map(move |x| {
                    if crate::geometry::ellipse_coordinate((x as f64, y as f64), ellipse()) <= 1.0 {
                        150
                    } else {
                        550
                    }
                })
            })
            .collect()
    }

    #[test]
    fn competing_raw_peaks_keep_their_paths_when_contrast_rank_swaps() {
        let profiles = (0..8)
            .map(|i| {
                let inner = Edge {profile_index: 0,
                    point: (i as f64, -5.0),
                    contrast: 10.0 + i as f64,
                    width: 4.0,
                    offset: -5.0,
                };
                let outer = Edge {profile_index: 0,
                    point: (i as f64, 5.0),
                    contrast: 17.0 - i as f64,
                    width: 4.0,
                    offset: 5.0,
                };
                if i < 4 {
                    [Some(outer), Some(inner)]
                } else {
                    [Some(inner), Some(outer)]
                }
            })
            .collect::<Vec<_>>();
        let runs = coherent_edge_runs(&profiles);
        assert_eq!(runs.len(), 2);
        assert!(runs
            .iter()
            .all(|run| run.len() == 8 && run.windows(2).all(|p| p[0].offset == p[1].offset)));
        assert_eq!(
            runs.iter().map(Vec::len).sum::<usize>(),
            16,
            "each measured peak belongs to exactly one path"
        );
    }

    #[test]
    fn raw_peak_paths_do_not_bridge_missing_profiles_or_jump_to_another_edge() {
        let edge = |x, offset| {
            Some(Edge {profile_index: 0,
                point: (x, offset),
                contrast: 20.0,
                width: 4.0,
                offset,
            })
        };
        let profiles = [
            [edge(0.0, -5.0), None],
            [edge(1.0, -5.0), None],
            [None, None],
            [edge(3.0, -5.0), None],
            [edge(4.0, 5.0), None],
            [edge(5.0, 5.0), None],
        ];
        let runs = coherent_edge_runs(&profiles);
        assert_eq!(runs.iter().map(Vec::len).collect::<Vec<_>>(), [2, 1, 2]);
        assert!(runs.iter().all(|run| run
            .windows(2)
            .all(|p| p[1].point.0 - p[0].point.0 == 1.0 && p[1].offset == p[0].offset)));
    }

    #[test]
    fn coherent_raw_paths_keep_smooth_offsets_and_outer_evidence() {
        let profiles = (0..8)
            .map(|i| {
                [
                    Some(Edge {profile_index: 0,
                        point: (i as f64, -3.0 + i as f64 * 0.75),
                        contrast: 20.0,
                        width: 4.0,
                        offset: -3.0 + i as f64 * 0.75,
                    }),
                    None,
                ]
            })
            .collect::<Vec<_>>();
        let runs = coherent_edge_runs(&profiles);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].len(), 8);
        for (edge, row) in runs[0].iter().zip(&profiles) {
            assert_eq!(edge.point, row[0].unwrap().point);
        }
        let raw = raw_disk();
        let evidence = review(ellipse().dense_points(128));
        let mut original = packet();
        append_retained_sam_arcs(&mut original, &evidence, 0);
        let mut candidate = original.clone();
        assert!(
            append_raw_ring_arcs_with_cohesion(
                &mut candidate,
                &raw,
                ellipse(),
                BoundaryKind::PupillaryBoundary,
                100,
                RawArcConfig::default(),
                true
            ) >= 4
        );
        assert_eq!(original.detail_reliability, candidate.detail_reliability);
        assert_eq!(
            format!("{:?}", original.arcs),
            format!("{:?}", &candidate.arcs[..original.arcs.len()])
        );
        assert!(candidate.arcs[original.arcs.len()..]
            .iter()
            .all(|arc| (100..108).contains(&arc.evidence_group)));
    }

    #[test]
    fn retained_outline_directions_are_measured_not_borrowed_from_the_fit() {
        let e = ellipse();
        let raw = raw_disk();
        for reverse in [false, true] {
            let mut points = e.dense_points(128);
            if reverse {
                points.reverse();
            }
            let mut evidence = review(points);
            let mut p = packet();
            append_retained_sam_arcs_with_direction_policy(&mut p, &evidence, 0, Some(&raw));
            assert!(!p.arcs.is_empty());
            for arc in &p.arcs {
                for (&(x, y), normal) in arc
                    .points_roi_px
                    .iter()
                    .zip(arc.outward_normals_roi.as_ref().unwrap())
                {
                    let normal = normal.unwrap();
                    assert!(normal.valid());
                    let gradient = [
                        (x - e.center.0) / e.major_radius.powi(2),
                        (y - e.center.1) / e.minor_radius.powi(2),
                    ];
                    assert!(
                        (gradient[0] * normal.unit_outward_roi[0]
                            + gradient[1] * normal.unit_outward_roi[1])
                            / gradient[0].hypot(gradient[1])
                            > 0.98
                    );
                }
            }
            evidence.ellipse = Ellipse {
                center: (22.0, 180.0),
                major_radius: 100.0,
                minor_radius: 12.0,
                angle: 1.7,
            };
            let mut changed = packet();
            append_retained_sam_arcs_with_direction_policy(&mut changed, &evidence, 0, Some(&raw));
            assert_eq!(p.arcs.len(), changed.arcs.len());
            for (a, b) in p.arcs.iter().zip(&changed.arcs) {
                assert_eq!(a.points_roi_px, b.points_roi_px);
                assert_eq!(
                    a.outward_normals_roi, b.outward_normals_roi,
                    "an ellipse hint is not measured direction"
                );
            }
            let mut legacy = packet();
            append_retained_sam_arcs(&mut legacy, &evidence, 0);
            assert!(
                legacy.arcs.iter().all(|a| a.outward_normals_roi.is_none()),
                "live remains the positional control"
            );
        }
    }

    #[test]
    fn retained_outline_directions_abstain_when_current_raw_cannot_confirm_the_side() {
        let evidence = review(ellipse().dense_points(128));
        for raw in [vec![100; 256 * 192], vec![1023; 256 * 192], vec![100; 10]] {
            let mut packet = packet();
            append_retained_sam_arcs_with_direction_policy(&mut packet, &evidence, 0, Some(&raw));
            assert!(
                !packet.arcs.is_empty(),
                "the existing measured positions remain available"
            );
            assert!(
                packet.arcs.iter().all(|arc| arc
                    .outward_normals_roi
                    .as_ref()
                    .is_none_or(|normals| normals.iter().all(Option::is_none))),
                "flat, saturated or missing RAW cannot manufacture directional support"
            );
        }
    }

    #[test]
    fn retained_outline_directions_do_not_bridge_an_occluded_gap() {
        let e = ellipse();
        let full = e.dense_points(128);
        let raw = raw_disk();
        let mut evidence = review(full[..20].iter().chain(&full[60..90]).copied().collect());
        evidence.conic_segments = std::sync::Arc::new(vec![(0..20).collect(), (20..50).collect()]);
        let mut p = packet();
        append_retained_sam_arcs_with_direction_policy(&mut p, &evidence, 0, Some(&raw));
        assert_eq!(p.arcs.len(), 2);
        for arc in &p.arcs {
            for (&(x, y), normal) in arc
                .points_roi_px
                .iter()
                .zip(arc.outward_normals_roi.as_ref().unwrap())
            {
                let n = normal.unwrap().unit_outward_roi;
                let gradient = [
                    (x - e.center.0) / e.major_radius.powi(2),
                    (y - e.center.1) / e.minor_radius.powi(2),
                ];
                assert!(
                    (gradient[0] * n[0] + gradient[1] * n[1]) / gradient[0].hypot(gradient[1])
                        > 0.98
                );
            }
        }
        let mut straight = review(vec![(10.0, 20.0), (20.0, 20.0), (30.0, 20.0)]);
        straight.ellipse = e;
        let mut p = packet();
        append_retained_sam_arcs_with_direction_policy(&mut p, &straight, 0, Some(&raw));
        assert!(
            p.arcs.iter().all(|a| a.outward_normals_roi.is_none()),
            "a line has no observed enclosed-side winding"
        );
    }
    #[test]
    fn no_raw_transition_means_no_arc_even_with_a_perfect_ellipse_guide() {
        for value in [0, 300, 1023] {
            let mut p = packet();
            assert_eq!(
                append_raw_ring_arcs(
                    &mut p,
                    &vec![value; 256 * 192],
                    ellipse(),
                    BoundaryKind::OuterLimbus,
                    0,
                    RawArcConfig::default()
                ),
                0
            );
            assert!(p.arcs.is_empty());
        }
    }
    #[test]
    fn raw_profiles_measure_the_edge_and_do_not_emit_the_guide_perimeter() {
        let e = ellipse();
        let raw = (0..192)
            .flat_map(|y| {
                (0..256).map(move |x| {
                    let r = ((x as f64 - e.center.0) / e.major_radius)
                        .hypot((y as f64 - e.center.1) / e.minor_radius);
                    if r <= 1.0 {
                        150
                    } else {
                        550
                    }
                })
            })
            .collect::<Vec<_>>();
        let mut guide = e;
        guide.center.0 += 3.0;
        let mut p = packet();
        let count = append_raw_ring_arcs(
            &mut p,
            &raw,
            guide,
            BoundaryKind::OuterLimbus,
            20,
            RawArcConfig::default(),
        );
        assert!(count >= 4, "{count}");
        let residuals = p
            .arcs
            .iter()
            .flat_map(|a| &a.points_roi_px)
            .map(|&p| crate::conic_solver::ellipse_residual(p, e))
            .collect::<Vec<_>>();
        assert!(residuals.iter().sum::<f64>() / (residuals.len() as f64) < 1.5);
        assert!(p.arcs.iter().all(|a| (20..28).contains(&a.evidence_group)));
    }

    #[test]
    fn pupil_edge_detail_does_not_reweight_preexisting_limbus_observations() {
        let e = ellipse();
        let raw = (0..192)
            .flat_map(|y| {
                (0..256).map(move |x| {
                    let r = ((x as f64 - e.center.0) / e.major_radius)
                        .hypot((y as f64 - e.center.1) / e.minor_radius);
                    if r <= 1.0 {
                        150
                    } else {
                        550
                    }
                })
            })
            .collect::<Vec<_>>();
        for supplied in [None, Some(0.1), Some(0.9)] {
            let mut p = packet();
            p.detail_reliability = supplied;
            p.arcs.push(OwnedBoundaryArc { support_length_cap_px: None, sampling_support_px: None, level_sets_roi: None,
                evidence_group: 0,
                kind: BoundaryKind::OuterLimbus,
                points_roi_px: vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
                outward_normals_roi: None,
                localization_sigma_px: None,
                normal_band_half_width_px: 2.75,
                detector_score: None,
            });
            assert!(
                append_raw_ring_arcs(
                    &mut p,
                    &raw,
                    e,
                    BoundaryKind::PupillaryBoundary,
                    100,
                    RawArcConfig::default()
                ) >= 4
            );
            assert_eq!(p.detail_reliability, supplied);
            assert_eq!(p.arcs[0].normal_band_half_width_px, 2.75);
            assert_eq!(
                p.arcs[0].points_roi_px,
                [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
            );
            assert!(p.arcs[1..]
                .iter()
                .all(|a| a.normal_band_half_width_px.is_finite()));
        }
    }

    #[test]
    fn pupil_arcs_do_not_reintroduce_unsaturated_screen_reflections() {
        let outer = Ellipse {
            major_radius: 90.0,
            minor_radius: 70.0,
            ..ellipse()
        };
        let pupil = Ellipse {
            major_radius: 32.0,
            minor_radius: 25.0,
            ..ellipse()
        };
        let raw = (0..192)
            .flat_map(|y| {
                (0..256).map(move |x| {
                    if (154..181).contains(&x) && (70..123).contains(&y) {
                        750
                    } else if crate::geometry::ellipse_coordinate((x as f64, y as f64), pupil)
                        <= 1.0
                    {
                        120
                    } else if crate::geometry::ellipse_coordinate((x as f64, y as f64), outer)
                        <= 1.0
                    {
                        220
                    } else {
                        480
                    }
                })
            })
            .collect::<Vec<_>>();
        let config = RawArcConfig::for_pupil(&raw, 256, 192, outer).unwrap();
        let ceiling = config.maximum_profile_luma_raw10.unwrap();
        assert!((ceiling - 308.0).abs() < 1.0e-9, "{ceiling}");
        let mut uncensored = packet();
        append_raw_ring_arcs(
            &mut uncensored,
            &raw,
            pupil,
            BoundaryKind::PupillaryBoundary,
            100,
            RawArcConfig::default(),
        );
        let mut censored = packet();
        assert!(
            append_raw_ring_arcs(
                &mut censored,
                &raw,
                pupil,
                BoundaryKind::PupillaryBoundary,
                100,
                config
            ) >= 4
        );
        let reflected = |p: &OwnedRoiEvidence| {
            p.arcs
                .iter()
                .flat_map(|a| &a.points_roi_px)
                .filter(|&&(x, y)| luma(&raw, 256, 192, x, y).is_some_and(|value| value > ceiling))
                .count()
        };
        assert!(
            reflected(&uncensored) > 0,
            "fixture must expose the old reflection edge"
        );
        assert_eq!(reflected(&censored), 0);
        let clear = censored
            .arcs
            .iter()
            .flat_map(|a| &a.points_roi_px)
            .filter(|&&(x, _)| x < 128.0)
            .copied()
            .collect::<Vec<_>>();
        assert!(
            clear.len() >= 16,
            "visible pupil side must remain available: {}",
            clear.len()
        );
        assert!(
            clear
                .iter()
                .map(|&point| crate::conic_solver::ellipse_residual(point, pupil))
                .sum::<f64>()
                / (clear.len() as f64)
                < 1.5
        );
        assert_eq!(censored.detail_reliability, None);
    }

    #[test]
    fn missing_iris_photometry_does_not_authorize_pupil_reflection_edges() {
        assert!(RawArcConfig::for_pupil(&[], 256, 192, ellipse()).is_none());
        assert!(iris_tissue_luma_ceiling(256, 192, ellipse(), |_, _| None).is_none());
        assert!(iris_tissue_luma_ceiling(256, 192, ellipse(), |_, _| Some(f64::NAN)).is_none());
        assert!(RawArcConfig::for_pupil(
            &vec![200; 256 * 192],
            256,
            192,
            Ellipse {
                center: (1000.0, 1000.0),
                ..ellipse()
            }
        )
        .is_none());
        for invalid in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            let mut p = packet();
            assert_eq!(
                append_raw_ring_arcs(
                    &mut p,
                    &vec![300; 256 * 192],
                    ellipse(),
                    BoundaryKind::PupillaryBoundary,
                    100,
                    RawArcConfig {
                        maximum_profile_luma_raw10: Some(invalid),
                        ..RawArcConfig::default()
                    }
                ),
                0
            );
        }
    }

    #[test]
    fn tissue_ceiling_is_raw_exposure_covariant_not_a_display_contrast_setting() {
        let e = ellipse();
        let sample = |x: usize, y: usize| 150.0 + (x % 11) as f64 + (y % 7) as f64;
        let base = iris_tissue_luma_ceiling(256, 192, e, |x, y| Some(sample(x, y))).unwrap();
        for gain in [0.5, 1.0, 2.0, 3.0] {
            let scaled =
                iris_tissue_luma_ceiling(256, 192, e, |x, y| Some(gain * sample(x, y))).unwrap();
            assert!((scaled - gain * base).abs() < 1.0e-9);
        }
    }
}
