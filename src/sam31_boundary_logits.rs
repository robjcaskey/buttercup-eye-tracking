//! Sensitivity of one selected semantic mask along its observed contour arcs.
//!
//! Logits are model activations, not calibrated boundary probabilities. The
//! sampled levels belong to ONE mask/source observation. They must never be
//! turned into independent contour votes or used to complete an occluded arc.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const HALF_WINDOW: i32 = 16;
const MAX_PROFILES: usize = 256;
const PROFILES_PER_ARC: usize = 256;
const LEVELS: [f64; 3] = [-1.0, 0.0, 1.0];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub eye_index: usize,
    pub sequence: u64,
    pub timestamp_ns: String,
    pub tracking_epoch: u64,
    pub prompt_generation: u64,
    pub sensor_origin: (u32, u32),
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub schema: String,
    pub source: Source,
    pub prompt_index: usize,
    pub query: usize,
    pub mask_size: (usize, usize),
    pub contour_refined: bool,
    pub offset_start_px: i32,
    pub offset_step_px: f64,
    pub samples_per_profile: usize,
    pub levels: [f64; 3],
    pub arcs: Vec<ArcProfiles>,
    pub contract: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArcProfiles {
    pub arc_index: usize,
    pub support_point_count: usize,
    pub profiles: Vec<Profile>,
    pub unavailable: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub point_index: usize,
    /// Position within this retained run. Version 1 exports did not store it.
    #[serde(default)]
    pub run_point_index: Option<usize>,
    pub point_roi_px: (f64, f64),
    /// Unit normal from adjacent measured samples within this run; endpoints
    /// use a one-sided tangent, without borrowing a point across a missing arc.
    /// Its sign follows contour order; it is not an inferred anatomical normal.
    pub normal_roi: (f64, f64),
    /// Uniform native-pixel offsets specified by the containing Evidence.
    /// None means outside the source image or nonfinite model support.
    pub logits: Vec<Option<f32>>,
    pub crossings: Vec<LevelCrossings>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LevelCrossings {
    pub level: f64,
    pub offsets_px: Vec<f64>,
    /// A threshold plateau is an interval, never an arbitrarily chosen root.
    pub plateaus_px: Vec<(f64, f64)>,
}

#[derive(Debug)]
pub(crate) struct Plane {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
}

impl Plane {
    fn sample_native(&self, x: f64, y: f64, source: &Source) -> Option<f32> {
        if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0
            || x > (source.width - 1) as f64 || y > (source.height - 1) as f64
        {
            return None;
        }
        // align_corners=false pixel centers, including edge clamping. This is
        // the inverse of the model/native contour conversion, not x * scale.
        let mx = ((x + 0.5) * self.width as f64 / source.width as f64 - 0.5)
            .clamp(0.0, (self.width - 1) as f64);
        let my = ((y + 0.5) * self.height as f64 / source.height as f64 - 0.5)
            .clamp(0.0, (self.height - 1) as f64);
        let x0 = mx.floor() as usize;
        let y0 = my.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = mx - x0 as f64;
        let ty = my - y0 as f64;
        let mut value = 0.0;
        for (px, py, weight) in [(x0, y0, (1.0-tx)*(1.0-ty)),
            (x1, y0, tx*(1.0-ty)), (x0, y1, (1.0-tx)*ty), (x1, y1, tx*ty)]
        {
            if weight == 0.0 { continue; }
            let sample = self.values[py * self.width + px];
            if !sample.is_finite() { return None; }
            value += f64::from(sample) * weight;
        }
        Some(value as f32)
    }
}

/// Sample only existing runs. Endpoint tangents are one-sided within the run;
/// no tangent bridges a censored gap. Published points and indices stay intact.
pub(crate) fn measure(
    plane: &Plane,
    source: Source,
    prompt_index: usize,
    query: usize,
    contour_refined: bool,
    points: &[(f64, f64)],
    runs: &[Vec<usize>],
) -> Result<Evidence, String> {
    if plane.width == 0 || plane.height == 0 || source.width == 0 || source.height == 0
        || plane.width.checked_mul(plane.height) != Some(plane.values.len())
    {
        return Err("selected-mask boundary logits have incompatible source/plane geometry".into());
    }
    let eligible = runs.iter().filter(|run| run.len() >= 3).count();
    // Bound the export without preferring a fitted or high-scoring arc. If a
    // pathological packet exceeds the budget, its omitted runs are explicit.
    let per_arc = (MAX_PROFILES / eligible.max(1)).clamp(1, PROFILES_PER_ARC);
    let mut remaining = MAX_PROFILES;
    let mut arcs = Vec::with_capacity(runs.len());
    for (arc_index, run) in runs.iter().enumerate() {
        let mut arc = ArcProfiles { arc_index, support_point_count: run.len(),
            profiles: Vec::new(), unavailable: BTreeMap::new() };
        let count = if run.len() < 3 {0} else {run.len().min(per_arc).min(remaining)};
        if count == 0 {
            arc.unavailable.insert(if run.len() < 3 { "no-within-run-tangent" }
                else { "profile-budget" }.into(), 1);
        }
        // Include the native adapter's 16-point quadrature before spending
        // the remaining budget on denser inspection profiles. An unrelated
        // evenly spaced grid can miss almost every point the solver retains.
        let mut indices = Vec::with_capacity(count);
        let native_count = run.len().min(16).min(count);
        if native_count > 0 {
            for slot in 0..native_count {
                indices.push(slot*(run.len()-1)/native_count.saturating_sub(1).max(1));
            }
            for slot in 0..count {
                let index = slot*(run.len()-1)/count.saturating_sub(1).max(1);
                if indices.len() == count { break; }
                if !indices.contains(&index) { indices.push(index); }
            }
            for index in 0..run.len() {
                if indices.len() == count { break; }
                if !indices.contains(&index) { indices.push(index); }
            }
        }
        indices.sort_unstable();
        for index in indices {
            remaining -= 1;
            let selected = [run[index.saturating_sub(1)], run[index], run[(index+1).min(run.len()-1)]];
            if selected.iter().any(|&i| i >= points.len()
                || !points[i].0.is_finite() || !points[i].1.is_finite())
            {
                *arc.unavailable.entry("invalid-measured-point".into()).or_default() += 1;
                continue;
            }
            let (before, point, after) = (points[selected[0]], points[selected[1]], points[selected[2]]);
            let (dx, dy) = (after.0-before.0, after.1-before.1);
            let length = dx.hypot(dy);
            if !length.is_finite() || length < 1e-6 {
                *arc.unavailable.entry("degenerate-within-run-tangent".into()).or_default() += 1;
                continue;
            }
            let normal = (-dy/length, dx/length);
            let logits = (-HALF_WINDOW..=HALF_WINDOW).map(|offset| {
                plane.sample_native(point.0+f64::from(offset)*normal.0,
                    point.1+f64::from(offset)*normal.1, &source)
            }).collect::<Vec<_>>();
            let crossings = LEVELS.into_iter().map(|level| level_crossings(&logits, level)).collect();
            arc.profiles.push(Profile { point_index: selected[1], run_point_index: Some(index), point_roi_px: point,
                normal_roi: normal, logits, crossings });
        }
        arcs.push(arc);
    }
    Ok(Evidence { schema: "buttercup-selected-mask-boundary-logits-v2".into(), source,
        prompt_index, query, mask_size: (plane.width, plane.height), contour_refined,
        offset_start_px: -HALF_WINDOW, offset_step_px: 1.0,
        samples_per_profile: (2*HALF_WINDOW+1) as usize, levels: LEVELS, arcs,
        contract: "One selected source mask; correlated sensitivity levels, not independent votes or calibrated probabilities. No contour motion or gap completion.".into() })
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct AttachmentReport {
    pub(crate) outer_points: usize,
    pub(crate) matched_profiles: usize,
    pub(crate) level_set_points: usize,
    pub(crate) unknown_profiles: usize,
    pub(crate) arcs_with_levels: usize,
}

/// Attach sensitivity to exact existing points. No point, conic hint, source
/// identity, noise allowance, arc budget or pupil evidence is replaced.
pub(crate) fn attach(
    packet: &mut crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence,
    evidence: &Evidence,
    group_base: u32,
) -> Result<AttachmentReport, String> {
    use crate::roi_evidence::{BoundaryKind, BoundaryLevelSetObservation};
    if !matches!(evidence.schema.as_str(), "buttercup-selected-mask-boundary-logits-v1" | "buttercup-selected-mask-boundary-logits-v2")
        || evidence.source.eye_index+1 != packet.exposure.roi.0 as usize
        || evidence.source.sequence != packet.exposure.sequence
        || evidence.source.timestamp_ns.parse::<u64>().ok() != Some(packet.exposure.timestamp_ns)
        || [evidence.source.sensor_origin.0,evidence.source.sensor_origin.1] != packet.sensor_origin_px
        || [evidence.source.width,evidence.source.height] != packet.dimensions_px.map(|n|n as usize)
        || evidence.prompt_index != 0 || evidence.levels != LEVELS
        || evidence.offset_start_px != -HALF_WINDOW || evidence.offset_step_px != 1.0
        || evidence.samples_per_profile != (2*HALF_WINDOW+1) as usize
    { return Err("mask-level evidence does not match the native source/coordinate contract".into()); }
    let mut seen_arcs=std::collections::BTreeSet::new();
    if evidence.arcs.iter().any(|a|!seen_arcs.insert(a.arc_index)) {
        return Err("duplicate retained arc identity in mask-level evidence".into());
    }
    let mut report=AttachmentReport::default();
    for arc in &mut packet.arcs {
        if arc.kind != BoundaryKind::OuterLimbus { continue; }
        report.outer_points += arc.points_roi_px.len();
        let Some(index)=arc.evidence_group.checked_sub(group_base) else {continue;};
        let Some(profiles)=evidence.arcs.iter().find(|a|a.arc_index==index as usize) else {continue;};
        let mut levels=Vec::with_capacity(arc.points_roi_px.len());
        if evidence.schema.ends_with("-v2") && arc.points_roi_px.len()!=profiles.support_point_count.min(16) {
            return Err("mask-level evidence must attach before native validation decimation".into());
        }
        for (slot,&point) in arc.points_roi_px.iter().enumerate() {
            let at=slot*profiles.support_point_count.saturating_sub(1)
                /arc.points_roi_px.len().saturating_sub(1).max(1);
            let matches=profiles.profiles.iter().filter(|p|p.point_roi_px==point
                && (!evidence.schema.ends_with("-v2") || p.run_point_index==Some(at))).collect::<Vec<_>>();
            if matches.len()>1 {return Err("duplicate native mask-level profile for an observed point".into());}
            let Some(profile)=matches.first() else {levels.push(None);continue;};
            report.matched_profiles+=1;
            let crossings=LEVELS.map(|level|level_crossings(&profile.logits,level));
            let single=profile.logits.len()==evidence.samples_per_profile
                && crossings.iter().all(|c|c.offsets_px.len()==1 && c.plateaus_px.is_empty());
            let offset=single.then(|| {
                let roots=crossings.each_ref().map(|c|c.offsets_px[0]);
                BoundaryLevelSetObservation {unit_normal_roi:[profile.normal_roi.0,profile.normal_roi.1],
                    displacement_px:[roots[0]-roots[1],0.0,roots[2]-roots[1]],spatial_displacement_px:None}
            }).filter(|level|level.valid());
            report.level_set_points+=offset.is_some() as usize;
            report.unknown_profiles+=offset.is_none() as usize;
            levels.push(offset);
        }
        if levels.iter().flatten().any(|p|p.varies()) {
            report.arcs_with_levels+=1;
            arc.level_sets_roi=Some(levels);
        }
    }
    Ok(report)
}

fn level_crossings(samples: &[Option<f32>], level: f64) -> LevelCrossings {
    let mut result = LevelCrossings { level, offsets_px: Vec::new(), plateaus_px: Vec::new() };
    for (i, pair) in samples.windows(2).enumerate() {
        let (Some(a), Some(b)) = (pair[0], pair[1]) else { continue; };
        let (a, b) = (f64::from(a)-level, f64::from(b)-level);
        let x = i as f64 - f64::from(HALF_WINDOW);
        if a == 0.0 && b == 0.0 {
            if let Some(last) = result.plateaus_px.last_mut().filter(|last| last.1 == x) {
                last.1 = x + 1.0;
            } else { result.plateaus_px.push((x, x+1.0)); }
        } else if a == 0.0 { result.offsets_px.push(x); }
        else if b == 0.0 { result.offsets_px.push(x+1.0); }
        else if a.signum() != b.signum() { result.offsets_px.push(x-a/(b-a)); }
    }
    result.offsets_px.dedup_by(|a, b| (*a-*b).abs() < 1e-9);
    result.offsets_px.retain(|x| !result.plateaus_px.iter().any(|&(lo, hi)| *x >= lo && *x <= hi));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(width: usize, height: usize) -> Source {
        Source { eye_index: 1, sequence: 160, timestamp_ns: "1690107949663163609".into(),
            tracking_epoch: 4, prompt_generation: 7, sensor_origin: (4512, 4656), width, height }
    }

    fn linear(slope: f32) -> Plane {
        Plane { width: 64, height: 64, values: (0..64*64)
            .map(|i| (i%64) as f32-32.0).map(|x| x*slope).collect() }
    }

    #[test]
    fn sharp_and_broad_logits_preserve_the_same_contour_and_one_observation() {
        let points = [(32.0, 20.0), (32.0, 32.0), (32.0, 44.0)];
        let runs = vec![vec![0, 1, 2]];
        for (slope, expected_width) in [(2.0, 1.0), (0.25, 8.0)] {
            let evidence = measure(&linear(slope), source(64, 64), 0, 0, false, &points, &runs).unwrap();
            assert_eq!(evidence.source, source(64, 64));
            assert_eq!(evidence.arcs.len(), 1);
            let arc = &evidence.arcs[0];
            assert_eq!(arc.support_point_count, 3);
            assert_eq!(arc.profiles.len(), 3);
            let profile = &arc.profiles[1];
            assert_eq!(profile.point_roi_px, points[1]);
            assert_eq!(profile.point_index, 1);
            assert_eq!(profile.normal_roi, (-1.0, 0.0));
            assert_eq!(profile.crossings[1].offsets_px, vec![0.0]);
            assert_eq!((profile.crossings[0].offsets_px[0]-profile.crossings[2].offsets_px[0]).abs(), expected_width);
        }
    }

    #[test]
    fn native_sampling_uses_pixel_centers_and_independent_axis_scales() {
        let plane = Plane { width: 4, height: 3,
            values: (0..12).map(|i| (i%4 + 10*(i/4)) as f32).collect() };
        let source = source(8, 9);
        for y in 0..3 { for x in 0..4 {
            let native = ((x as f64+0.5)*2.0-0.5, (y as f64+0.5)*3.0-0.5);
            assert_eq!(plane.sample_native(native.0, native.1, &source), Some((x+10*y) as f32));
        }}
        assert_eq!(plane.sample_native(0.0, 0.0, &source), Some(0.0));
        assert_eq!(plane.sample_native(7.0, 8.0, &source), Some(23.0));
        assert_eq!(plane.sample_native(-0.001, 0.0, &source), None);
    }

    #[test]
    fn disconnected_arcs_never_borrow_tangents_or_create_support() {
        let points = [(32., 10.), (32., 12.), (32., 14.), (10., 32.), (20., 32.)];
        let runs = vec![vec![0,1,2], vec![3,4]];
        let evidence = measure(&linear(1.0), source(64,64), 0, 0, false, &points, &runs).unwrap();
        assert_eq!(evidence.arcs[0].profiles.iter().map(|p|p.point_index).collect::<Vec<_>>(),vec![0,1,2]);
        assert!(evidence.arcs[0].profiles.iter().all(|p|p.normal_roi==(-1.0,0.0)));
        assert!(evidence.arcs[1].profiles.is_empty());
        assert_eq!(evidence.arcs[1].unavailable["no-within-run-tangent"], 1);
    }

    #[test]
    fn multiple_crossings_plateaus_and_missing_intervals_remain_explicit() {
        let values = vec![Some(-1.), Some(1.), None, Some(-1.), Some(0.), Some(0.), Some(1.), Some(-1.)];
        let roots = level_crossings(&values, 0.0);
        assert_eq!(roots.offsets_px, vec![-15.5, -9.5]);
        assert_eq!(roots.plateaus_px, vec![(-12.0, -11.0)]);
        assert!(level_crossings(&[Some(-1.), None, Some(1.)], 0.).offsets_px.is_empty());
    }

    #[test]
    fn absent_boundary_and_nonfinite_support_are_not_narrow_uncertainty() {
        let mut plane = linear(1.0);
        plane.values.fill(4.0);
        let points = [(2.,20.), (2.,32.), (2.,44.)];
        let runs = vec![vec![0,1,2]];
        let mut evidence = measure(&plane, source(64,64), 0, 0, false, &points, &runs).unwrap();
        let profile = &evidence.arcs[0].profiles[0];
        assert!(profile.logits.iter().any(Option::is_none));
        assert!(profile.crossings.iter().all(|c| c.offsets_px.is_empty()));
        plane.values.fill(f32::NAN);
        evidence = measure(&plane, source(64,64), 0, 0, false, &points, &runs).unwrap();
        assert!(evidence.arcs[0].profiles[0].logits.iter().all(Option::is_none));
        assert!(serde_json::to_string(&evidence).is_ok());
    }

    #[test]
    fn sampling_budget_and_invalid_geometry_are_bounded_and_reported() {
        let points = [(32.,20.), (32.,32.), (32.,44.)];
        let runs = vec![vec![0,1,2]; 300];
        let evidence = measure(&linear(1.0), source(64,64), 0, 0, false, &points, &runs).unwrap();
        assert_eq!(evidence.arcs.iter().map(|a| a.profiles.len()).sum::<usize>(), MAX_PROFILES);
        assert_eq!(evidence.arcs.last().unwrap().unavailable["profile-budget"], 1);
        assert!(measure(&linear(1.0), source(0,64), 0, 0, false, &points, &runs).is_err());
        let bad = measure(&linear(1.0), source(64,64), 0, 0, false, &points, &[vec![0,1,999]]).unwrap();
        assert_eq!(bad.arcs[0].unavailable["invalid-measured-point"], 2);
    }

    #[test]
    fn bounded_profiles_include_every_native_solver_sample_before_extra_samples() {
        let points=(0..97).map(|i|(32.0,8.0+i as f64*0.5)).collect::<Vec<_>>();
        let runs=vec![(0..97).collect::<Vec<_>>();12];
        let evidence=measure(&linear(1.0),source(64,64),0,0,false,&points,&runs).unwrap();
        assert!(evidence.arcs.iter().map(|a|a.profiles.len()).sum::<usize>()<=MAX_PROFILES);
        for arc in &evidence.arcs {
            for at in (0..16).map(|i|i*96/15) {
                let p=arc.profiles.iter().find(|p|p.run_point_index==Some(at)).unwrap();
                assert_eq!(p.point_roi_px,points[at]);
            }
        }
    }

    fn attachment_fixture() -> (crate::outline_conic_segments::sparse_evidence::OwnedRoiEvidence,Evidence) {
        use crate::outline_conic_segments::sparse_evidence::{OwnedRoiEvidence,OwnedBoundaryArc};
        use crate::roi_evidence::{BoundaryKind,ExposureKey,RoiId,SourceClock};
        let s=source(64,64);
        let points=vec![(32.0,20.0),(32.0,32.0),(32.0,44.0)];
        let evidence=measure(&linear(0.25),s.clone(),0,9,false,&points,&[vec![0,1,2]]).unwrap();
        let outer=OwnedBoundaryArc { support_length_cap_px: None, sampling_support_px: None,evidence_group:20,kind:BoundaryKind::OuterLimbus,
            points_roi_px:points,outward_normals_roi:None,level_sets_roi:None,
            localization_sigma_px:Some(2.0),normal_band_half_width_px:1.5,detector_score:Some(0.8)};
        let mut pupil=outer.clone();pupil.kind=BoundaryKind::PupillaryBoundary;pupil.evidence_group=100;
        (OwnedRoiEvidence {exposure:ExposureKey {roi:RoiId(2),clock:SourceClock {domain:1,epoch:11},
            sequence:s.sequence,timestamp_ns:s.timestamp_ns.parse().unwrap()},
            sensor_origin_px:[s.sensor_origin.0,s.sensor_origin.1],dimensions_px:[64,64],
            arcs:vec![outer,pupil],conics:vec![],detail_reliability:Some(0.6)},evidence)
    }

    #[test]
    fn attachment_preserves_native_points_pupil_identity_and_noise() {
        let (mut packet,evidence)=attachment_fixture();
        let before=format!("{packet:?}");
        let decoded:Evidence=serde_json::from_str(&serde_json::to_string(&evidence).unwrap()).unwrap();
        assert_eq!(decoded,evidence);
        let report=attach(&mut packet,&decoded,20).unwrap();
        assert_eq!((report.outer_points,report.matched_profiles,report.level_set_points,report.arcs_with_levels),(3,3,3,1));
        assert!(packet.arcs[0].level_sets_roi.as_ref().unwrap().iter().all(|p| {
            let p=p.unwrap();p.displacement_px==[4.0,0.0,-4.0] && p.unit_normal_roi==[-1.0,0.0]
        }));
        assert!(packet.arcs[1].level_sets_roi.is_none());
        packet.arcs[0].level_sets_roi=None;
        assert_eq!(format!("{packet:?}"),before);
    }

    #[test]
    fn attachment_refuses_wrong_sources_and_duplicate_profile_identities() {
        let (packet,evidence)=attachment_fixture();
        let mut wrong=evidence.clone();wrong.source.sequence+=1;
        assert!(attach(&mut packet.clone(),&wrong,20).is_err());
        wrong=evidence.clone();wrong.source.sensor_origin.0+=1;
        assert!(attach(&mut packet.clone(),&wrong,20).is_err());
        wrong=evidence.clone();let duplicate=wrong.arcs[0].profiles[0].clone();wrong.arcs[0].profiles.push(duplicate);
        assert!(attach(&mut packet.clone(),&wrong,20).is_err());
        wrong=evidence.clone();wrong.arcs.push(wrong.arcs[0].clone());
        assert!(attach(&mut packet.clone(),&wrong,20).is_err());
        let mut decimated=packet.clone();decimated.arcs[0].points_roi_px.pop();
        assert!(attach(&mut decimated,&evidence,20).is_err());
    }

    #[test]
    fn absent_or_ambiguous_profiles_do_not_invent_boundary_states() {
        let (mut packet,mut evidence)=attachment_fixture();
        evidence.arcs[0].profiles[0].logits.fill(None);
        evidence.arcs[0].profiles[1].logits.fill(Some(0.0));
        evidence.arcs[0].profiles.pop();
        let report=attach(&mut packet,&evidence,20).unwrap();
        assert_eq!((report.outer_points,report.matched_profiles,report.unknown_profiles,report.level_set_points),(3,2,2,0));
        assert!(packet.arcs.iter().all(|a|a.level_sets_roi.is_none()));
    }

    #[test]
    fn repeated_closed_run_coordinates_join_by_source_run_position() {
        let (mut packet,_)=attachment_fixture();
        let first=packet.arcs[0].points_roi_px[0];
        packet.arcs[0].points_roi_px.push(first);
        let evidence=measure(&linear(0.25),source(64,64),0,9,false,
            &packet.arcs[0].points_roi_px,&[vec![0,1,2,3]]).unwrap();
        let report=attach(&mut packet,&evidence,20).unwrap();
        assert_eq!(report.matched_profiles,4);
        assert_eq!(report.level_set_points,4);
    }
}
