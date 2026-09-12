//! Owned sparse packets and source-native boundary adapters for the joint solve.
//!
//! A projected ellipse is a SEARCH GUIDE, never observed arc evidence. The RAW
//! adapter searches bounded radial profiles and emits only current positive
//! intensity transitions. The SAM adapter retains actual non-flat-tired runs.
//! Neither interpolates across rejected/occluded runs to manufacture evidence.

use crate::geometry::Ellipse;
use crate::roi_evidence::{BoundaryArcObservation,BoundaryKind,BoundaryNormalObservation,ConicObservation,ExposureKey,RoiConicEvidence};
use super::ContourFitEvidence;

#[derive(Clone,Debug)]
pub(crate) struct OwnedBoundaryArc {
    pub(crate) evidence_group:u32,
    pub(crate) kind:BoundaryKind,
    pub(crate) points_roi_px:Vec<(f64,f64)>,
    pub(crate) outward_normals_roi:Option<Vec<Option<BoundaryNormalObservation>>>,
    pub(crate) normal_band_half_width_px:f64,
    pub(crate) detector_score:Option<f64>,
}

#[derive(Clone,Debug)]
pub(crate) struct OwnedConicHint {
    pub(crate) kind:BoundaryKind,
    pub(crate) ellipse_roi_px:Ellipse,
    pub(crate) supporting_arc_indices:Vec<usize>,
}

#[derive(Clone,Debug)]
pub(crate) struct OwnedRoiEvidence {
    pub(crate) exposure:ExposureKey,
    pub(crate) sensor_origin_px:[u32;2],
    pub(crate) dimensions_px:[u32;2],
    pub(crate) arcs:Vec<OwnedBoundaryArc>,
    pub(crate) conics:Vec<OwnedConicHint>,
    pub(crate) detail_reliability:Option<f64>,
}

pub(crate) struct PreparedRoiEvidence<'a> {
    source:&'a OwnedRoiEvidence,
    arcs:Vec<BoundaryArcObservation<'a>>,
    conics:Vec<ConicObservation<'a>>,
}

impl OwnedRoiEvidence {
    pub(crate) fn prepare(&self) -> PreparedRoiEvidence<'_> {
        PreparedRoiEvidence {source:self,
            arcs:self.arcs.iter().map(|a| BoundaryArcObservation {
                evidence_group:a.evidence_group,kind:a.kind,points_roi_px:&a.points_roi_px,
                outward_normals_roi:a.outward_normals_roi.as_deref(),
                normal_band_half_width_px:Some(a.normal_band_half_width_px),detector_score:a.detector_score,
            }).collect(),
            conics:self.conics.iter().map(|c| ConicObservation {kind:c.kind,ellipse_roi_px:c.ellipse_roi_px,
                supporting_arc_indices:&c.supporting_arc_indices,residual_px:None}).collect(),
        }
    }
}

impl PreparedRoiEvidence<'_> {
    pub(crate) fn evidence(&self) -> RoiConicEvidence<'_> {
        RoiConicEvidence {exposure:self.source.exposure,sensor_origin_px:self.source.sensor_origin_px,
            dimensions_px:self.source.dimensions_px,arcs:&self.arcs,conics:&self.conics,
            detail_reliability:self.source.detail_reliability}
    }
}

/// Retain sparse samples of real SAM contour runs. A single connected run is
/// one correlated evidence group, not one vote per resampled pixel.
pub(crate) fn append_retained_sam_arcs(packet:&mut OwnedRoiEvidence, review:&ContourFitEvidence, group_base:u32) {
    append_retained_sam_arcs_with_direction_policy(packet,review,group_base,None);
}

/// Optional measured-contour direction experiment. Directions come from
/// neighbors within a retained run, never the completed ellipse's gradient.
/// Current RAW polarity must confirm outward direction. Their information
/// shares the existing arc budget in the joint objective.
pub(crate) fn append_retained_sam_arcs_with_direction_policy(packet:&mut OwnedRoiEvidence,
    review:&ContourFitEvidence,group_base:u32,raw:Option<&[u16]>) {
    let start = packet.arcs.len();
    let perimeter=&review.retained_points;
    let signed_area=perimeter.iter().zip(perimeter.iter().cycle().skip(1)).take(perimeter.len())
        .map(|(a,b)|a.0*b.1-b.0*a.1).sum::<f64>();
    let [width,height]=packet.dimensions_px.map(|v|v as usize);
    let raw=raw.filter(|raw|width>=12 && height>=12 && width.checked_mul(height)==Some(raw.len()));
    let winding=(raw.is_some() && signed_area.is_finite() && signed_area.abs()>1.0).then(||signed_area.signum());
    let native_jitter=1.5*packet.dimensions_px[0] as f64/crate::conic_solver::LEGACY_FIT_WIDTH as f64;
    for (run,indices) in review.conic_segments.iter().take(12).enumerate() {
        if indices.len()<3 {continue;}
        let count=indices.len().min(16);
        let points=(0..count).filter_map(|i| review.retained_points.get(indices[i*(indices.len()-1)/(count-1)]).copied()).collect::<Vec<_>>();
        if points.len()<3 {continue;}
        let normals=winding.filter(|_|points.len()==count).map(|winding|(0..count).map(|i| {
            let at=i*(indices.len()-1)/(count-1);
            let a=*perimeter.get(indices[at.saturating_sub(2)])?;
            let b=*perimeter.get(indices[(at+2).min(indices.len()-1)])?;
            let tangent=(b.0-a.0,b.1-a.1);let length=tangent.0.hypot(tangent.1);
            if !length.is_finite() || length<1.0e-6 {return None;}
            let normal=[winding*tangent.1/length,-winding*tangent.0/length];
            let point=*perimeter.get(indices[at])?;
            let inside=luma(raw?,width,height,point.0-3.0*normal[0],point.1-3.0*normal[1])?;
            let outside=luma(raw?,width,height,point.0+3.0*normal[0],point.1+3.0*normal[1])?;
            if inside>=990.0 || outside>=990.0 || outside-inside<7.0 {return None;}
            Some(BoundaryNormalObservation {unit_outward_roi:normal,
                angular_sigma_radians:15.0f64.to_radians().hypot((native_jitter*2.0f64.sqrt()).atan2(length))})
        }).collect());
        packet.arcs.push(OwnedBoundaryArc {evidence_group:group_base+run as u32,kind:BoundaryKind::OuterLimbus,
            points_roi_px:points,outward_normals_roi:normals,normal_band_half_width_px:1.5,detector_score:None});
    }
    packet.conics.push(OwnedConicHint {kind:BoundaryKind::OuterLimbus,ellipse_roi_px:review.ellipse,
        supporting_arc_indices:(start..packet.arcs.len()).collect()});
}

#[derive(Clone,Copy,Debug)]
pub(crate) struct RawArcConfig {
    pub(crate) radial_search_px:f64,
    pub(crate) minimum_contrast_raw10:f64,
    /// A boundary-specific tissue ceiling, not a full-ROI brightness/focus
    /// score. Profiles touching brighter specular pixels are unavailable.
    pub(crate) maximum_profile_luma_raw10:Option<f64>,
}

impl Default for RawArcConfig {
    fn default() -> Self {Self {radial_search_px:8.0,minimum_contrast_raw10:7.0,
        maximum_profile_luma_raw10:None}}
}

/// Shared RAW photometric policy with the SAM pupil fitter. The enclosing
/// limbus is only a sampling guide: this is not pupil evidence or a scale
/// estimate. A missing/too-small iris annulus cannot certify a glint boundary.
/// Keep the historical stride, annulus and robust estimator so moving this
/// policy out of the SAM adapter does not change its existing pupil fits.
pub(crate) fn iris_tissue_luma_ceiling(width:usize,height:usize,outer:Ellipse,
    sample:impl Fn(usize,usize)->Option<f64>)->Option<f64> {
    if !outer.center.0.is_finite() || !outer.center.1.is_finite()
        || !outer.major_radius.is_finite() || !outer.minor_radius.is_finite()
        || !outer.angle.is_finite() || outer.minor_radius<=0.0
        || outer.major_radius<outer.minor_radius {return None;}
    let mut annulus=Vec::new();
    for y in (0..height).step_by(2) {for x in (0..width).step_by(2) {
        if !(0.55..=0.85).contains(&crate::geometry::ellipse_coordinate((x as f64,y as f64),outer)) {continue;}
        if let Some(value)=sample(x,y).filter(|v|v.is_finite()) {annulus.push(value);}
    }}
    if annulus.len()<64 {return None;}
    let center=crate::conic_solver::median(annulus.clone());
    let deviation=crate::conic_solver::median(annulus.into_iter().map(|value|(value-center).abs()).collect());
    Some(center+(4.0*deviation).max(0.4*center).max(12.0))
}

impl RawArcConfig {
    pub(crate) fn for_pupil(raw:&[u16],width:usize,height:usize,outer:Ellipse)->Option<Self> {
        if width<12 || height<12 || width.checked_mul(height)!=Some(raw.len()) {return None;}
        let ceiling=iris_tissue_luma_ceiling(width,height,outer,
            |x,y|luma(raw,width,height,x as f64,y as f64))?;
        Some(Self {maximum_profile_luma_raw10:Some(ceiling),..Self::default()})
    }
}

/// Average a CFA-aligned 4×4 cell (one complete quad-Bayer color cell).
/// Bilinear sampling these cell means avoids treating mosaic phase as an edge.
/// This is custom native RAW decoding/photometry, not a rendered RGB thumbnail.
pub(super) fn luma(raw:&[u16],width:usize,height:usize,x:f64,y:f64) -> Option<f64> {
    if !x.is_finite() || !y.is_finite() || x<1.5 || y<1.5 || x>width as f64-6.5 || y>height as f64-6.5 {return None;}
    let gx=(x-1.5)/4.0; let gy=(y-1.5)/4.0;
    let ix=gx.floor() as usize; let iy=gy.floor() as usize;
    let wx=gx-ix as f64; let wy=gy-iy as f64;
    let cell=|cx:usize,cy:usize| (0..4).flat_map(|dy| (0..4).map(move |dx| raw[(cy*4+dy)*width+cx*4+dx] as f64)).sum::<f64>()/16.0;
    Some((1.0-wy)*((1.0-wx)*cell(ix,iy)+wx*cell(ix+1,iy))
        +wy*((1.0-wx)*cell(ix,iy+1)+wx*cell(ix+1,iy+1)))
}

#[derive(Clone,Copy)]
struct Edge {point:(f64,f64),contrast:f64,width:f64}

/// At most 64 radial profiles and 17 offset candidates each. Out-of-frame,
/// saturated and non-positive profiles remain absent, not predicted points.
/// Retain separated positive edge peaks as CORRELATED alternatives in each
/// angular sector. The joint model can select a secondary plausible boundary.
pub(crate) fn append_raw_ring_arcs(packet:&mut OwnedRoiEvidence,raw:&[u16],guide:Ellipse,
    kind:BoundaryKind,group_base:u32,config:RawArcConfig) -> usize {
    let [width,height]=packet.dimensions_px.map(|v|v as usize);
    if width<12 || height<12 || width*height!=raw.len()
        || !guide.major_radius.is_finite() || !guide.minor_radius.is_finite()
        || !guide.angle.is_finite() || guide.minor_radius<4.0
        || guide.major_radius<guide.minor_radius
        || !config.radial_search_px.is_finite() || config.radial_search_px<=0.0
        || !config.minimum_contrast_raw10.is_finite() || config.minimum_contrast_raw10<=0.0
        || config.maximum_profile_luma_raw10.is_some_and(|v|!v.is_finite() || v<=0.0) {return 0;}
    let (sine,cosine)=guide.angle.sin_cos();
    let search=config.radial_search_px.min(12.0);
    let step=search/8.0;
    let mut edges:[[Option<Edge>;2];64]=[[None;2];64];
    for (index,result) in edges.iter_mut().enumerate() {
        let phase=std::f64::consts::TAU*index as f64/64.0;
        let x=guide.major_radius*phase.cos(); let y=guide.minor_radius*phase.sin();
        let point=(guide.center.0+cosine*x-sine*y,guide.center.1+sine*x+cosine*y);
        let nx=phase.cos()/guide.major_radius; let ny=phase.sin()/guide.minor_radius;
        let norm=nx.hypot(ny);
        let normal=((cosine*nx-sine*ny)/norm,(sine*nx+cosine*ny)/norm);
        let mut profiles=Vec::with_capacity(17);
        for offset_index in -8..=8 {
            let offset=offset_index as f64*step;
            let center=(point.0+offset*normal.0,point.1+offset*normal.1);
            let (Some(inner),Some(outer))=(luma(raw,width,height,center.0-3.0*normal.0,center.1-3.0*normal.1),
                luma(raw,width,height,center.0+3.0*normal.0,center.1+3.0*normal.1)) else {profiles.push(None);continue;};
            let ceiling=config.maximum_profile_luma_raw10.unwrap_or(990.0).min(990.0);
            if inner>ceiling || outer>ceiling {profiles.push(None);continue;}
            profiles.push(Some((outer-inner,center)));
        }
        let mut peaks=profiles.iter().enumerate().filter_map(|(i,p)| {
            let (contrast,point)=(*p)?;
            if contrast<config.minimum_contrast_raw10 {return None;}
            // A maximum outside the search window is censored. Do not pin it
            // to the guide's allowed boundary and call that an observed edge.
            if i==0 || i+1==profiles.len() {return None;}
            // A censored adjacent profile is another unknown search limit,
            // not evidence that the current point is an intensity maximum.
            let (Some(before),Some(after))=(profiles[i-1],profiles[i+1]) else {return None;};
            if before.0>contrast || after.0>=contrast {return None;}
            let supported_width=profiles.iter().filter_map(|p|*p).filter(|p|p.0>=contrast*0.5).count() as f64*step;
            Some(Edge {point,contrast,width:supported_width})
        }).collect::<Vec<_>>();
        peaks.sort_by(|a,b|b.contrast.total_cmp(&a.contrast));
        for (slot,edge) in result.iter_mut().zip(peaks.into_iter().take(2)) {*slot=Some(edge);}
    }
    let start=packet.arcs.len();
    for sector in 0..8 {
        for alternative in 0..2 {
            let mut run=Vec::new();
            // Do not cross missing-profile gaps. Each contiguous subrun in
            // this sector is an alternative for one information budget.
            let flush=|run:&mut Vec<Edge>,arcs:&mut Vec<OwnedBoundaryArc>| {
                if run.len()>=3 {
                    let mean_width=run.iter().map(|p|p.width).sum::<f64>()/run.len() as f64;
                    let score=run.iter().map(|p|p.contrast).sum::<f64>()/run.len() as f64;
                    arcs.push(OwnedBoundaryArc {evidence_group:group_base+sector as u32,kind,
                        points_roi_px:run.iter().map(|p|p.point).collect(),
                        outward_normals_roi:None,
                        normal_band_half_width_px:(mean_width*0.25).clamp(0.75,4.0),detector_score:Some(score)});
                }
                run.clear();
            };
            for row in edges.iter().skip(sector*8).take(8) {
                if let Some(edge)=row[alternative] {run.push(edge);} else {flush(&mut run,&mut packet.arcs);}
            }
            flush(&mut run,&mut packet.arcs);
        }
    }
    let count=packet.arcs.len()-start;
    if count>0 {
        packet.conics.push(OwnedConicHint {kind,ellipse_roi_px:guide,supporting_arc_indices:(start..packet.arcs.len()).collect()});
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
    use crate::roi_evidence::{RoiId,SourceClock};
    fn packet()->OwnedRoiEvidence {OwnedRoiEvidence {
        exposure:ExposureKey {roi:RoiId(1),clock:SourceClock {domain:1,epoch:1},sequence:1,timestamp_ns:1},
        sensor_origin_px:[0,0],dimensions_px:[256,192],arcs:Vec::new(),conics:Vec::new(),detail_reliability:None}}
    fn ellipse()->Ellipse {Ellipse {center:(128.0,96.0),major_radius:60.0,minor_radius:46.0,angle:0.0}}
    fn review(points:Vec<(f64,f64)>)->ContourFitEvidence {
        let count=points.len();
        ContourFitEvidence {ellipse:ellipse(),source_component_area_px:0.0,
            retained_points:std::sync::Arc::new(points),conic_segments:std::sync::Arc::new(vec![(0..count).collect()]),
            flat_tire_points:std::sync::Arc::new(Vec::new()),upper_flat_tire:false,lower_flat_tire:false}
    }
    fn raw_disk()->Vec<u16> {
        (0..192).flat_map(|y|(0..256).map(move|x|
            if crate::geometry::ellipse_coordinate((x as f64,y as f64),ellipse())<=1.0 {150} else {550})).collect()
    }

    #[test]
    fn retained_outline_directions_are_measured_not_borrowed_from_the_fit() {
        let e=ellipse();
        let raw=raw_disk();
        for reverse in [false,true] {
            let mut points=e.dense_points(128);if reverse {points.reverse();}
            let mut evidence=review(points);let mut p=packet();
            append_retained_sam_arcs_with_direction_policy(&mut p,&evidence,0,Some(&raw));
            assert!(!p.arcs.is_empty());
            for arc in &p.arcs {for (&(x,y),normal) in arc.points_roi_px.iter().zip(arc.outward_normals_roi.as_ref().unwrap()) {
                let normal=normal.unwrap();assert!(normal.valid());
                let gradient=[(x-e.center.0)/e.major_radius.powi(2),(y-e.center.1)/e.minor_radius.powi(2)];
                assert!((gradient[0]*normal.unit_outward_roi[0]+gradient[1]*normal.unit_outward_roi[1])
                    /gradient[0].hypot(gradient[1])>0.98);
            }}
            evidence.ellipse=Ellipse {center:(22.0,180.0),major_radius:100.0,minor_radius:12.0,angle:1.7};
            let mut changed=packet();append_retained_sam_arcs_with_direction_policy(&mut changed,&evidence,0,Some(&raw));
            assert_eq!(p.arcs.len(),changed.arcs.len());
            for (a,b) in p.arcs.iter().zip(&changed.arcs) {
                assert_eq!(a.points_roi_px,b.points_roi_px);
                assert_eq!(a.outward_normals_roi,b.outward_normals_roi,"an ellipse hint is not measured direction");
            }
            let mut legacy=packet();append_retained_sam_arcs(&mut legacy,&evidence,0);
            assert!(legacy.arcs.iter().all(|a|a.outward_normals_roi.is_none()),"live remains the positional control");
        }
    }

    #[test]
    fn retained_outline_directions_abstain_when_current_raw_cannot_confirm_the_side() {
        let evidence=review(ellipse().dense_points(128));
        for raw in [vec![100;256*192],vec![1023;256*192],vec![100;10]] {
            let mut packet=packet();
            append_retained_sam_arcs_with_direction_policy(&mut packet,&evidence,0,Some(&raw));
            assert!(!packet.arcs.is_empty(),"the existing measured positions remain available");
            assert!(packet.arcs.iter().all(|arc|arc.outward_normals_roi.as_ref()
                .is_none_or(|normals|normals.iter().all(Option::is_none))),
                "flat, saturated or missing RAW cannot manufacture directional support");
        }
    }

    #[test]
    fn retained_outline_directions_do_not_bridge_an_occluded_gap() {
        let e=ellipse();let full=e.dense_points(128);let raw=raw_disk();
        let mut evidence=review(full[..20].iter().chain(&full[60..90]).copied().collect());
        evidence.conic_segments=std::sync::Arc::new(vec![(0..20).collect(),(20..50).collect()]);
        let mut p=packet();append_retained_sam_arcs_with_direction_policy(&mut p,&evidence,0,Some(&raw));
        assert_eq!(p.arcs.len(),2);
        for arc in &p.arcs {for (&(x,y),normal) in arc.points_roi_px.iter().zip(arc.outward_normals_roi.as_ref().unwrap()) {
            let n=normal.unwrap().unit_outward_roi;
            let gradient=[(x-e.center.0)/e.major_radius.powi(2),(y-e.center.1)/e.minor_radius.powi(2)];
            assert!((gradient[0]*n[0]+gradient[1]*n[1])/gradient[0].hypot(gradient[1])>0.98);
        }}
        let mut straight=review(vec![(10.0,20.0),(20.0,20.0),(30.0,20.0)]);
        straight.ellipse=e;
        let mut p=packet();append_retained_sam_arcs_with_direction_policy(&mut p,&straight,0,Some(&raw));
        assert!(p.arcs.iter().all(|a|a.outward_normals_roi.is_none()),"a line has no observed enclosed-side winding");
    }
    #[test]
    fn no_raw_transition_means_no_arc_even_with_a_perfect_ellipse_guide() {
        for value in [0,300,1023] {
            let mut p=packet();
            assert_eq!(append_raw_ring_arcs(&mut p,&vec![value;256*192],ellipse(),BoundaryKind::OuterLimbus,0,RawArcConfig::default()),0);
            assert!(p.arcs.is_empty());
        }
    }
    #[test]
    fn raw_profiles_measure_the_edge_and_do_not_emit_the_guide_perimeter() {
        let e=ellipse();
        let raw=(0..192).flat_map(|y|(0..256).map(move|x| {
            let r=((x as f64-e.center.0)/e.major_radius).hypot((y as f64-e.center.1)/e.minor_radius);
            if r<=1.0 {150} else {550}
        })).collect::<Vec<_>>();
        let mut guide=e;guide.center.0+=3.0;
        let mut p=packet();
        let count=append_raw_ring_arcs(&mut p,&raw,guide,BoundaryKind::OuterLimbus,20,RawArcConfig::default());
        assert!(count>=4,"{count}");
        let residuals=p.arcs.iter().flat_map(|a|&a.points_roi_px).map(|&p|crate::conic_solver::ellipse_residual(p,e)).collect::<Vec<_>>();
        assert!(residuals.iter().sum::<f64>()/(residuals.len() as f64)<1.5);
        assert!(p.arcs.iter().all(|a|(20..28).contains(&a.evidence_group)));
    }

    #[test]
    fn pupil_edge_detail_does_not_reweight_preexisting_limbus_observations() {
        let e=ellipse();
        let raw=(0..192).flat_map(|y|(0..256).map(move|x| {
            let r=((x as f64-e.center.0)/e.major_radius).hypot((y as f64-e.center.1)/e.minor_radius);
            if r<=1.0 {150} else {550}
        })).collect::<Vec<_>>();
        for supplied in [None,Some(0.1),Some(0.9)] {
            let mut p=packet();p.detail_reliability=supplied;
            p.arcs.push(OwnedBoundaryArc {evidence_group:0,kind:BoundaryKind::OuterLimbus,
                points_roi_px:vec![(1.0,2.0),(3.0,4.0),(5.0,6.0)],outward_normals_roi:None,
                normal_band_half_width_px:2.75,detector_score:None});
            assert!(append_raw_ring_arcs(&mut p,&raw,e,BoundaryKind::PupillaryBoundary,100,RawArcConfig::default())>=4);
            assert_eq!(p.detail_reliability,supplied);
            assert_eq!(p.arcs[0].normal_band_half_width_px,2.75);
            assert_eq!(p.arcs[0].points_roi_px,[(1.0,2.0),(3.0,4.0),(5.0,6.0)]);
            assert!(p.arcs[1..].iter().all(|a|a.normal_band_half_width_px.is_finite()));
        }
    }

    #[test]
    fn pupil_arcs_do_not_reintroduce_unsaturated_screen_reflections() {
        let outer=Ellipse {major_radius:90.0,minor_radius:70.0,..ellipse()};
        let pupil=Ellipse {major_radius:32.0,minor_radius:25.0,..ellipse()};
        let raw=(0..192).flat_map(|y|(0..256).map(move|x| {
            if (154..181).contains(&x) && (70..123).contains(&y) {750}
            else if crate::geometry::ellipse_coordinate((x as f64,y as f64),pupil)<=1.0 {120}
            else if crate::geometry::ellipse_coordinate((x as f64,y as f64),outer)<=1.0 {220}
            else {480}
        })).collect::<Vec<_>>();
        let config=RawArcConfig::for_pupil(&raw,256,192,outer).unwrap();
        let ceiling=config.maximum_profile_luma_raw10.unwrap();
        assert!((ceiling-308.0).abs()<1.0e-9,"{ceiling}");
        let mut uncensored=packet();
        append_raw_ring_arcs(&mut uncensored,&raw,pupil,BoundaryKind::PupillaryBoundary,100,RawArcConfig::default());
        let mut censored=packet();
        assert!(append_raw_ring_arcs(&mut censored,&raw,pupil,BoundaryKind::PupillaryBoundary,100,config)>=4);
        let reflected=|p:&OwnedRoiEvidence|p.arcs.iter().flat_map(|a|&a.points_roi_px)
            .filter(|&&(x,y)|luma(&raw,256,192,x,y).is_some_and(|value|value>ceiling)).count();
        assert!(reflected(&uncensored)>0,"fixture must expose the old reflection edge");
        assert_eq!(reflected(&censored),0);
        let clear=censored.arcs.iter().flat_map(|a|&a.points_roi_px)
            .filter(|&&(x,_)|x<128.0).copied().collect::<Vec<_>>();
        assert!(clear.len()>=16,"visible pupil side must remain available: {}",clear.len());
        assert!(clear.iter().map(|&point|crate::conic_solver::ellipse_residual(point,pupil)).sum::<f64>()
            /(clear.len() as f64)<1.5);
        assert_eq!(censored.detail_reliability,None);
    }

    #[test]
    fn missing_iris_photometry_does_not_authorize_pupil_reflection_edges() {
        assert!(RawArcConfig::for_pupil(&[],256,192,ellipse()).is_none());
        assert!(iris_tissue_luma_ceiling(256,192,ellipse(),|_,_|None).is_none());
        assert!(iris_tissue_luma_ceiling(256,192,ellipse(),|_,_|Some(f64::NAN)).is_none());
        assert!(RawArcConfig::for_pupil(&vec![200;256*192],256,192,
            Ellipse {center:(1000.0,1000.0),..ellipse()}).is_none());
        for invalid in [f64::NAN,f64::INFINITY,-1.0,0.0] {
            let mut p=packet();
            assert_eq!(append_raw_ring_arcs(&mut p,&vec![300;256*192],ellipse(),BoundaryKind::PupillaryBoundary,100,
                RawArcConfig {maximum_profile_luma_raw10:Some(invalid),..RawArcConfig::default()}),0);
        }
    }

    #[test]
    fn tissue_ceiling_is_raw_exposure_covariant_not_a_display_contrast_setting() {
        let e=ellipse();
        let sample=|x:usize,y:usize|150.0+(x%11) as f64+(y%7) as f64;
        let base=iris_tissue_luma_ceiling(256,192,e,|x,y|Some(sample(x,y))).unwrap();
        for gain in [0.5,1.0,2.0,3.0] {
            let scaled=iris_tissue_luma_ceiling(256,192,e,|x,y|Some(gain*sample(x,y))).unwrap();
            assert!((scaled-gain*base).abs()<1.0e-9);
        }
    }
}
