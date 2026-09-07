//! Sparse outer-disk evidence without requiring a complete upstream ellipse.
//!
//! Callers supply only mandatory outer-iris SAM outlines from this exposure,
//! ordered by detector score. Other semantic prompts are NOT iris evidence.
//! Chords and unsupported RAW transitions are missing data, not filled rims.

use super::{censor_occluding_chords,sample_closed_contour};
use super::sparse_evidence::{luma,OwnedBoundaryArc,OwnedConicHint,OwnedRoiEvidence};
use crate::conic_solver::{direct_conic_fit,moments_ellipse,LEGACY_FIT_WIDTH};
use crate::roi_evidence::{BoundaryKind,BoundaryNormalObservation};
use std::f64::consts::TAU;

const MAX_CANDIDATES:usize=4;
const SECTORS:usize=8;
const MAX_POINTS:usize=16;
// Engineering allowance for rasterization and semantic-boundary jitter in
// 384-wide model coordinates, not a calibrated anatomical confidence bound.
const INWARD_NOTCH_TOLERANCE_MODEL_PX:f64=3.0;

pub(crate) struct OutlineCandidate<'a> {
    pub(crate) points_roi_px:&'a [(f64,f64)],
    pub(crate) detector_score:Option<f64>,
}

#[derive(Clone,Copy,Debug,Default)]
pub(crate) struct PartialOutlineReport {
    pub(crate) candidates:usize,
    pub(crate) censored_samples:usize,
    pub(crate) inward_notch_samples:usize,
    pub(crate) unsupported_samples:usize,
    pub(crate) emitted_arcs:usize,
}

fn length(points:&[(f64,f64)])->f64 {
    points.windows(2).map(|p|(p[1].0-p[0].0).hypot(p[1].1-p[0].1)).sum()
}

#[derive(Clone,Copy)]
struct OutlineSample {point:(f64,f64),normal:BoundaryNormalObservation}

fn sample_length(samples:&[OutlineSample])->f64 {
    samples.windows(2).map(|p|(p[1].point.0-p[0].point.0).hypot(p[1].point.1-p[0].point.1)).sum()
}

/// Censor deep inward bites from a contour assumed to be a subset of a convex
/// projected disk. A bright curved occluder can have the correct RAW polarity
/// without being limbus. The hull is ONLY an exclusion witness: its vertices
/// and closing chords never become replacement observations. Outward semantic
/// errors can still invalidate this subset assumption; this is not iris ID.
/// Work is bounded by the existing 128 contour samples.
fn inward_notch_mask(samples:&[(f64,f64)],tolerance:f64)->Vec<bool> {
    if samples.len()<3 || samples.len()>128 || samples.iter().any(|p|!p.0.is_finite()||!p.1.is_finite()) {
        return vec![true;samples.len()];
    }
    let cross=|a:(f64,f64),b:(f64,f64),p:(f64,f64)|
        (b.0-a.0)*(p.1-a.1)-(b.1-a.1)*(p.0-a.0);
    let mut sorted=samples.to_vec();
    sorted.sort_by(|a,b|a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    sorted.dedup();
    let mut hull=Vec::new();
    for reverse in [false,true] {
        let mut chain=Vec::new();
        for i in 0..sorted.len() {
            let p=sorted[if reverse {sorted.len()-1-i} else {i}];
            while chain.len()>=2 && cross(chain[chain.len()-2],chain[chain.len()-1],p)<=0.0 {chain.pop();}
            chain.push(p);
        }
        chain.pop();hull.extend(chain);
    }
    if hull.len()<3 {return vec![true;samples.len()];}
    samples.iter().map(|&p|hull.iter().zip(hull.iter().cycle().skip(1)).take(hull.len())
        .all(|(&a,&b)|cross(a,b,p)>tolerance*(b.0-a.0).hypot(b.1-a.1))).collect()
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
        let inward=inward_notch_mask(&censored.samples,INWARD_NOTCH_TOLERANCE_MODEL_PX);
        report.inward_notch_samples+=inward.iter().filter(|&&excluded|excluded).count();
        let excluded=censored.flat_tire.iter().zip(&inward).map(|(&chord,&bite)|chord||bite).collect::<Vec<_>>();
        let winding=censored.samples.iter().zip(censored.samples.iter().cycle().skip(1)).take(count)
            .map(|(a,b)|a.0*b.1-b.0*a.1).sum::<f64>().signum();
        if winding==0.0 {continue;}
        let mut observed=vec![None;count];
        for (i,slot) in observed.iter_mut().enumerate() {
            // A tangent straddling an excluded chord or inward bite is not
            // limbus support. No arc can bridge the missing observations.
            if (-2isize..=2).any(|d|excluded[(i as isize+d).rem_euclid(count as isize) as usize]) {
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
            // Geometry is from the measured ordered contour, with polarity
            // checked against this RAW exposure, never from an ellipse seed.
            // A 15-degree floor plus 1.5-model-pixel endpoint jitter widens
            // short-baseline direction support. These are engineering sigmas.
            let angular_sigma_radians=15.0f64.to_radians().hypot((1.5*2.0f64.sqrt()).atan2(norm));
            let sample=OutlineSample {point,normal:BoundaryNormalObservation {
                unit_outward_roi:[normal.0,normal.1],angular_sigma_radians}};
            *slot=Some(((phase/TAU*SECTORS as f64).floor() as usize%SECTORS,sample));
        }
        // Keep the longest contiguous observed run in each sector for this
        // candidate. Never join two fragments across a missing sample.
        let mut sectors:[Vec<OutlineSample>;SECTORS]=std::array::from_fn(|_|Vec::new());
        let mut runs:Vec<(usize,Vec<OutlineSample>)>=Vec::new();
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
            if sector<SECTORS && run.len()>=3 && sample_length(&run)>sample_length(&sectors[sector]) {sectors[sector]=run;}
        }
        let start=packet.arcs.len();
        let seed_points=sectors.iter().flatten().map(|s|s.point).collect::<Vec<_>>();
        for (sector,run) in sectors.iter().enumerate().filter(|(_,r)|r.len()>=3&&sample_length(r)>=6.0) {
            let count=run.len().min(MAX_POINTS);
            let samples=(0..count).map(|i|run[i*(run.len()-1)/(count-1)]).collect::<Vec<_>>();
            packet.arcs.push(OwnedBoundaryArc {evidence_group:group_base+sector as u32,kind:BoundaryKind::OuterLimbus,
                points_roi_px:samples.iter().map(|s|s.point).collect(),
                outward_normals_roi:Some(samples.iter().map(|s|Some(s.normal)).collect()),
                normal_band_half_width_px:5.0,detector_score:candidate.detector_score});
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
        let e=ellipse();
        for packet in [&a,&b] {for arc in &packet.arcs {
            let normals=arc.outward_normals_roi.as_ref().unwrap();
            assert_eq!(normals.len(),arc.points_roi_px.len());
            for (&(x,y),normal) in arc.points_roi_px.iter().zip(normals) {
                let n=normal.unwrap();assert!(n.valid());
                let gradient=[(x-e.center.0)/e.major_radius.powi(2),(y-e.center.1)/e.minor_radius.powi(2)];
                let agreement=(gradient[0]*n.unit_outward_roi[0]+gradient[1]*n.unit_outward_roi[1])/gradient[0].hypot(gradient[1]);
                assert!(agreement>0.99,"winding must not invert the RAW-supported outward normal");
            }
        }}
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
        // The inherited resampler rounds to f32 in model coordinates; retain
        // that exact observation frame for the provenance assertion.
        let scale=LEGACY_FIT_WIDTH as f64/256.0;
        let model=outline.iter().map(|&(x,y)|((x+0.5)*scale-0.5,(y+0.5)*scale-0.5)).collect::<Vec<_>>();
        let original_samples=sample_closed_contour(&model,128).into_iter()
            .map(|(x,y)|((x+0.5)/scale-0.5,(y+0.5)/scale-0.5)).collect::<Vec<_>>();
        let p=extract(&raw,&[outline]);
        assert!(p.arcs.len()>=3,"the remaining actual limbus must survive");
        let worst=p.arcs.iter().flat_map(|a|&a.points_roi_px)
            .map(|&point|crate::conic_solver::ellipse_residual(point,e)).fold(0.0,f64::max);
        assert!(worst<5.0,"bright inward occlusion leaked into limbus evidence: {worst}px");
        assert!(p.arcs.iter().flat_map(|a|&a.points_roi_px).all(|&(x,y)|original_samples.iter()
            .any(|&(u,v)|(x-u).hypot(y-v)<1.0e-7)),"hull edges are never emitted as replacement observations");
    }

    #[test]
    fn inward_censor_is_winding_translation_and_rotation_covariant() {
        let e=ellipse();
        let points=e.dense_points(128).into_iter().map(|(x,y)|
            if x>165.0 {(x-25.0,y)} else {(x,y)}).collect::<Vec<_>>();
        let mask=inward_notch_mask(&points,3.0);
        assert!(mask.iter().any(|&v|v));assert!(mask.iter().any(|&v|!v));
        for angle in [0.0f64,0.7,1.8,3.4] {
            let (s,c)=angle.sin_cos();
            let mut transformed=points.iter().map(|&(x,y)|(c*x-s*y+710.0,s*x+c*y-310.0)).collect::<Vec<_>>();
            assert_eq!(inward_notch_mask(&transformed,3.0),mask);
            transformed.reverse();
            let mut reversed=inward_notch_mask(&transformed,3.0);reversed.reverse();assert_eq!(reversed,mask);
        }
    }

    #[test]
    fn raster_jitter_on_a_convex_ellipse_does_not_become_an_occlusion() {
        for angle in [0.0,0.6,1.2] {
            let e=Ellipse {angle,..ellipse()};
            let points=e.dense_points(128).into_iter().map(|(x,y)|(x.round(),y.round())).collect::<Vec<_>>();
            assert!(inward_notch_mask(&points,3.0).iter().all(|&v|!v));
        }
    }
}
