//! Sparse outer-disk evidence without requiring a complete upstream ellipse.
//!
//! Callers supply only mandatory outer-iris SAM outlines from this exposure,
//! ordered by detector score. Other semantic prompts are NOT iris evidence.
//! Chords and unsupported RAW transitions are missing data, not filled rims.

use super::{censor_occluding_chords,sample_closed_contour};
use super::sparse_evidence::{luma,OwnedBoundaryArc,OwnedConicHint,OwnedRoiEvidence};
use crate::conic_solver::{direct_conic_fit,moments_ellipse,LEGACY_FIT_WIDTH};
use crate::roi_evidence::BoundaryKind;
use std::f64::consts::TAU;

const MAX_CANDIDATES:usize=4;
const SECTORS:usize=8;
const MAX_POINTS:usize=16;

pub(crate) struct OutlineCandidate<'a> {
    pub(crate) points_roi_px:&'a [(f64,f64)],
    pub(crate) detector_score:Option<f64>,
}

#[derive(Clone,Copy,Debug,Default)]
pub(crate) struct PartialOutlineReport {
    pub(crate) candidates:usize,
    pub(crate) censored_samples:usize,
    pub(crate) unsupported_samples:usize,
    pub(crate) emitted_arcs:usize,
}

fn length(points:&[(f64,f64)])->f64 {
    points.windows(2).map(|p|(p[1].0-p[0].0).hypot(p[1].1-p[0].1)).sum()
}

/// Maximum four candidate contours, 128 samples each, eight correlation groups
/// and 16 points per arc. Overlapping queries share the SAME sector budget.
/// The fixed association center is supplied before considering alternatives;
/// moving a candidate cannot change its own definition of independent groups.
pub(crate) fn append_unfitted_outline_arcs(packet:&mut OwnedRoiEvidence,raw:&[u16],
    candidates:&[OutlineCandidate<'_>],association_center:(f64,f64),group_base:u32)
    ->PartialOutlineReport {
    let mut report=PartialOutlineReport::default();
    let [width,height]=packet.dimensions_px.map(|v|v as usize);
    if width<16 || height<16 || width.checked_mul(height)!=Some(raw.len())
        || !association_center.0.is_finite() || !association_center.1.is_finite()
        || group_base.checked_add(SECTORS as u32).is_none() {return report;}
    // Keep the inherited curvature thresholds in their established model
    // coordinates. RAW observation and emitted evidence remain native pixels.
    let model_scale=LEGACY_FIT_WIDTH as f64/width as f64;
    for candidate in candidates.iter().take(MAX_CANDIDATES) {
        let outline=candidate.points_roi_px;
        if outline.len()<24 || outline.len()>1024 || outline.iter().any(|p|!p.0.is_finite()||!p.1.is_finite()) {
            continue;
        }
        report.candidates+=1;
        let model=outline.iter().map(|&(x,y)|((x+0.5)*model_scale-0.5,(y+0.5)*model_scale-0.5)).collect::<Vec<_>>();
        let coarse_samples=sample_closed_contour(&model,128);
        let Some(mut guide)=moments_ellipse(&coarse_samples) else {continue;};
        // Perimeter moments, unlike filled-disk moments, overestimate a
        // circle's radius by sqrt(2). This is only a chord scale, not a fit.
        guide.major_radius/=2.0f64.sqrt();guide.minor_radius/=2.0f64.sqrt();
        let Some(censored)=censor_occluding_chords(&model,guide) else {continue;};
        let count=censored.samples.len();
        let winding=censored.samples.iter().zip(censored.samples.iter().cycle().skip(1)).take(count)
            .map(|(a,b)|a.0*b.1-b.0*a.1).sum::<f64>().signum();
        if winding==0.0 {continue;}
        let mut observed=vec![None;count];
        for (i,slot) in observed.iter_mut().enumerate() {
            if censored.flat_tire[i] {report.censored_samples+=1;continue;}
            // A tangent straddling an excluded chord is not limbus support.
            if (-2isize..=2).any(|d|censored.flat_tire[(i as isize+d).rem_euclid(count as isize) as usize]) {
                report.censored_samples+=1;continue;
            }
            let a=censored.smoothed[(i+count-2)%count];
            let b=censored.smoothed[(i+2)%count];
            let tangent=(b.0-a.0,b.1-a.1);
            let norm=tangent.0.hypot(tangent.1);
            if norm<1.0e-6 {report.unsupported_samples+=1;continue;}
            let normal=(winding*tangent.1/norm,-winding*tangent.0/norm);
            let point=censored.samples[i];
            let point=((point.0+0.5)/model_scale-0.5,(point.1+0.5)/model_scale-0.5);
            let contrast=luma(raw,width,height,point.0+3.0*normal.0,point.1+3.0*normal.1)
                .zip(luma(raw,width,height,point.0-3.0*normal.0,point.1-3.0*normal.1))
                .filter(|&(outside,inside)|outside<990.0&&inside<990.0)
                .map(|(outside,inside)|outside-inside);
            if !contrast.is_some_and(|c|c>=7.0) {report.unsupported_samples+=1;continue;}
            let phase=(point.1-association_center.1).atan2(point.0-association_center.0).rem_euclid(TAU);
            *slot=Some(((phase/TAU*SECTORS as f64).floor() as usize%SECTORS,point));
        }
        // Keep the longest contiguous observed run in each sector for this
        // candidate. Never join two fragments across a missing sample.
        let mut sectors:[Vec<(f64,f64)>;SECTORS]=std::array::from_fn(|_|Vec::new());
        let mut runs:Vec<(usize,Vec<(f64,f64)>)>=Vec::new();
        for sample in &observed {
            if let Some((sector,point))=sample {
                if runs.last().is_none_or(|r|r.0!=*sector) {runs.push((*sector,Vec::new()));}
                runs.last_mut().unwrap().1.push(*point);
            } else if runs.last().is_some_and(|r|r.0!=SECTORS) {runs.push((SECTORS,Vec::new()));}
        }
        if runs.len()>1 && observed.first().is_some_and(Option::is_some)
            && observed.last().is_some_and(Option::is_some) && runs[0].0==runs.last().unwrap().0 {
            let (sector,mut end)=runs.pop().unwrap();end.append(&mut runs[0].1);runs[0]=(sector,end);
        }
        for (sector,run) in runs {
            if sector<SECTORS && run.len()>=3 && length(&run)>length(&sectors[sector]) {sectors[sector]=run;}
        }
        let start=packet.arcs.len();
        let seed_points=sectors.iter().flatten().copied().collect::<Vec<_>>();
        for (sector,run) in sectors.iter().enumerate().filter(|(_,r)|r.len()>=3&&length(r)>=6.0) {
            let count=run.len().min(MAX_POINTS);
            let points=(0..count).map(|i|run[i*(run.len()-1)/(count-1)]).collect();
            packet.arcs.push(OwnedBoundaryArc {evidence_group:group_base+sector as u32,kind:BoundaryKind::OuterLimbus,
                points_roi_px:points,normal_band_half_width_px:5.0,detector_score:candidate.detector_score});
        }
        if packet.arcs.len()>start {
            // A partial system can have no single-eye fit at all. When a
            // bounded fit exists it is a start only, never an observation.
            if let Some(ellipse)=direct_conic_fit(&seed_points).filter(|e|e.major_radius<=width as f64
                && e.minor_radius>=4.0 && e.center.0>=-(width as f64)*0.5 && e.center.0<=width as f64*1.5
                && e.center.1>=-(height as f64)*0.5 && e.center.1<=height as f64*1.5) {
                packet.conics.push(OwnedConicHint {kind:BoundaryKind::OuterLimbus,ellipse_roi_px:ellipse,
                    supporting_arc_indices:(start..packet.arcs.len()).collect()});
            }
            report.emitted_arcs+=packet.arcs.len()-start;
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Ellipse;
    use crate::roi_evidence::{ExposureKey,RoiId,SourceClock};
    fn ellipse()->Ellipse {Ellipse {center:(128.0,96.0),major_radius:60.0,minor_radius:46.0,angle:0.0}}
    fn packet()->OwnedRoiEvidence {OwnedRoiEvidence {exposure:ExposureKey {roi:RoiId(1),
        clock:SourceClock {domain:1,epoch:1},sequence:10,timestamp_ns:20},sensor_origin_px:[400,800],
        dimensions_px:[256,192],arcs:Vec::new(),conics:Vec::new(),detail_reliability:None}}
    fn raw()->Vec<u16> {
        let e=ellipse();
        (0..192).flat_map(|y|(0..256).map(move|x| {
            if ((x as f64-e.center.0)/e.major_radius).hypot((y as f64-e.center.1)/e.minor_radius)<=1.0 {150} else {550}
        })).collect()
    }
    fn extract(raw:&[u16],outlines:&[Vec<(f64,f64)>])->OwnedRoiEvidence {
        let candidates=outlines.iter().map(|points|OutlineCandidate {points_roi_px:points,detector_score:Some(0.7)}).collect::<Vec<_>>();
        let mut p=packet();append_unfitted_outline_arcs(&mut p,raw,&candidates,ellipse().center,20);p
    }
    #[test]
    fn flat_saturated_or_invalid_raw_does_not_turn_a_contour_into_evidence() {
        let outlines=[ellipse().dense_points(128)];
        for value in [0,300,1023] {assert!(extract(&vec![value;256*192],&outlines).arcs.is_empty());}
        assert!(extract(&[],&outlines).arcs.is_empty());
    }
    #[test]
    fn partial_circle_survives_but_an_occluding_chord_is_not_an_arc() {
        let e=ellipse();
        let outline=e.dense_points(128).into_iter().map(|(x,y)|(x.max(110.0),y)).collect::<Vec<_>>();
        let p=extract(&raw(),&[outline]);
        assert!(p.arcs.len()>=3,"{} arcs",p.arcs.len());
        assert!(p.arcs.iter().flat_map(|a|&a.points_roi_px).all(|&point|crate::conic_solver::ellipse_residual(point,e)<1.0));
        assert!(p.arcs.iter().flat_map(|a|&a.points_roi_px).all(|&(x,_)|x>112.0));
    }
    #[test]
    fn query_duplicates_share_groups_and_obey_a_fixed_budget() {
        let outline=ellipse().dense_points(128);
        let one=extract(&raw(),std::slice::from_ref(&outline));
        let many=extract(&raw(),&vec![outline;20]);
        assert!(one.arcs.len()>=4);
        assert_eq!(many.arcs.len(),one.arcs.len()*MAX_CANDIDATES);
        assert!(many.arcs.iter().all(|a|a.points_roi_px.len()<=MAX_POINTS&&(20..28).contains(&a.evidence_group)));
        for a in &one.arcs {assert_eq!(many.arcs.iter().filter(|b|a.evidence_group==b.evidence_group).count(),MAX_CANDIDATES);}
    }
    #[test]
    fn outline_winding_does_not_reverse_raw_boundary_polarity() {
        let outline=ellipse().dense_points(128);let mut reversed=outline.clone();reversed.reverse();
        let a=extract(&raw(),&[outline]);let b=extract(&raw(),&[reversed]);
        assert_eq!(a.arcs.len(),b.arcs.len());
        assert!((a.arcs.iter().map(|a|length(&a.points_roi_px)).sum::<f64>()-
            b.arcs.iter().map(|a|length(&a.points_roi_px)).sum::<f64>()).abs()<2.0);
    }

    #[test]
    fn an_inward_curving_bright_occluder_is_not_part_of_the_outer_limbus() {
        let e=ellipse();
        let mut raw=vec![550;256*192];let mut component=Vec::new();
        for y in 0..192 {for x in 0..256 {
            let inside=((x as f64-e.center.0)/e.major_radius).hypot((y as f64-e.center.1)/e.minor_radius)<=1.0;
            let reflection=(x as f64-165.0).hypot(y as f64-96.0)<32.0;
            if inside&&!reflection {raw[y*256+x]=150;component.push(y*256+x);}
        }}
        let outline=super::super::native_component_contour(&component,256,192);
        let p=extract(&raw,&[outline]);
        assert!(p.arcs.len()>=3,"the remaining actual limbus must survive");
        let worst=p.arcs.iter().flat_map(|a|&a.points_roi_px)
            .map(|&point|crate::conic_solver::ellipse_residual(point,e)).fold(0.0,f64::max);
        assert!(worst<5.0,"bright inward occlusion leaked into limbus evidence: {worst}px");
    }
}
