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
    let start = packet.arcs.len();
    for (run,indices) in review.conic_segments.iter().take(12).enumerate() {
        if indices.len()<3 {continue;}
        let count=indices.len().min(16);
        let points=(0..count).filter_map(|i| review.retained_points.get(indices[i*(indices.len()-1)/(count-1)]).copied()).collect::<Vec<_>>();
        if points.len()<3 {continue;}
        packet.arcs.push(OwnedBoundaryArc {evidence_group:group_base+run as u32,kind:BoundaryKind::OuterLimbus,
            points_roi_px:points,outward_normals_roi:None,normal_band_half_width_px:1.5,detector_score:None});
    }
    packet.conics.push(OwnedConicHint {kind:BoundaryKind::OuterLimbus,ellipse_roi_px:review.ellipse,
        supporting_arc_indices:(start..packet.arcs.len()).collect()});
}

#[derive(Clone,Copy,Debug)]
pub(crate) struct RawArcConfig {
    pub(crate) radial_search_px:f64,
    pub(crate) minimum_contrast_raw10:f64,
}

impl Default for RawArcConfig {
    fn default() -> Self {Self {radial_search_px:8.0,minimum_contrast_raw10:7.0}}
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
        || !config.minimum_contrast_raw10.is_finite() || config.minimum_contrast_raw10<=0.0 {return 0;}
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
            if inner>990.0 || outer>990.0 {profiles.push(None);continue;}
            profiles.push(Some((outer-inner,center)));
        }
        let mut peaks=profiles.iter().enumerate().filter_map(|(i,p)| {
            let (contrast,point)=(*p)?;
            if contrast<config.minimum_contrast_raw10 {return None;}
            // A maximum outside the search window is censored. Do not pin it
            // to the guide's allowed boundary and call that an observed edge.
            if i==0 || i+1==profiles.len() {return None;}
            if profiles[i-1].is_some_and(|p|p.0>contrast) || profiles[i+1].is_some_and(|p|p.0>=contrast) {return None;}
            let supported_width=profiles.iter().filter_map(|p|*p).filter(|p|p.0>=contrast*0.5).count() as f64*step;
            Some(Edge {point,contrast,width:supported_width})
        }).collect::<Vec<_>>();
        peaks.sort_by(|a,b|b.contrast.total_cmp(&a.contrast));
        for (slot,edge) in result.iter_mut().zip(peaks.into_iter().take(2)) {*slot=Some(edge);}
    }
    let start=packet.arcs.len();
    let mut detail=Vec::new();
    for sector in 0..8 {
        for alternative in 0..2 {
            let mut run=Vec::new();
            // Do not cross missing-profile gaps. Each contiguous subrun in
            // this sector is an alternative for one information budget.
            let flush=|run:&mut Vec<Edge>,arcs:&mut Vec<OwnedBoundaryArc>,detail:&mut Vec<f64>| {
                if run.len()>=3 {
                    let mean_width=run.iter().map(|p|p.width).sum::<f64>()/run.len() as f64;
                    let score=run.iter().map(|p|p.contrast).sum::<f64>()/run.len() as f64;
                    detail.push((4.0/mean_width.max(4.0)).clamp(0.0,1.0));
                    arcs.push(OwnedBoundaryArc {evidence_group:group_base+sector as u32,kind,
                        points_roi_px:run.iter().map(|p|p.point).collect(),
                        outward_normals_roi:None,
                        normal_band_half_width_px:(mean_width*0.25).clamp(0.75,4.0),detector_score:Some(score)});
                }
                run.clear();
            };
            for row in edges.iter().skip(sector*8).take(8) {
                if let Some(edge)=row[alternative] {run.push(edge);} else {flush(&mut run,&mut packet.arcs,&mut detail);}
            }
            flush(&mut run,&mut packet.arcs,&mut detail);
        }
    }
    let count=packet.arcs.len()-start;
    if count>0 {
        packet.conics.push(OwnedConicHint {kind,ellipse_roi_px:guide,supporting_arc_indices:(start..packet.arcs.len()).collect()});
        // Per-arc widths retain optical differences. This packet summary is a
        // heuristic and cannot by itself establish boundary correctness.
        detail.sort_by(f64::total_cmp);
        packet.detail_reliability=detail.get(detail.len()/2).copied();
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
}
