//! Matched joint-versus-monocular conic evaluation on immutable native SAM
//! contour exports. This is component validation, not measured gaze accuracy.
//! Human labels and screen target positions are deliberately not inputs.
#![allow(dead_code)]
#[path="../geometry.rs"] mod geometry;
#[path="../raw10.rs"] mod raw10;
#[path="../"] mod native {
    pub(crate) mod conic_solver;
    pub(crate) mod outline_conic_segments;
    pub(crate) mod roi_evidence;
    pub(crate) mod binocular_coordinator;
    pub(crate) mod eye_scene_model {pub(crate) mod binocular_pose;pub(crate) use crate::conic_solver::camera_mount::CameraMount;}
}
use native::{conic_solver,outline_conic_segments,roi_evidence};
use native::{binocular_coordinator,eye_scene_model};
#[path="../gaze_target_solver/joint_tracking.rs"] mod joint_tracking;
#[path="buttercup_stereo_conic_eval/source_order.rs"] mod source_order;
#[path="../sam31_boundary_logits.rs"] mod mask_boundary_logits;
use native::eye_scene_model::binocular_pose::{approximate_scene,EyePoseInput};
use conic_solver::joint::*;
use outline_conic_segments::sparse_evidence::*;
use outline_conic_segments::sparse_evidence::uncertainty as raw_optical_uncertainty;
use outline_conic_segments::partial_outline::{append_unfitted_outline_arcs,OutlineCandidate,PartialOutlineReport};
use roi_evidence::{BoundaryKind,ExposureKey,RoiId,SourceClock};
use serde_json::{json,Value};
use std::collections::HashMap;
use std::fs::{File,OpenOptions};
use std::io::{BufRead,BufReader,BufWriter,Read,Seek,SeekFrom,Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn number(v:&Value)->Option<u64> {v.as_u64().or_else(||v.as_str()?.parse().ok())}
fn integer(v:&Value,key:&str)->Result<u64,String> {number(&v[key]).ok_or_else(||format!("missing {key}"))}
fn ellipse(v:&Value)->Option<geometry::Ellipse> {Some(geometry::Ellipse {
    center:(v["center"][0].as_f64()?,v["center"][1].as_f64()?),major_radius:v["major_radius"].as_f64()?,
    minor_radius:v["minor_radius"].as_f64()?,angle:v["angle"].as_f64()?,})}
fn ellipse_json(e:Option<geometry::Ellipse>)->Value {e.map(|e|json!({"center":e.center,"major_radius":e.major_radius,
    "minor_radius":e.minor_radius,"angle":e.angle})).unwrap_or(Value::Null)}
fn points(v:&Value)->Vec<(f64,f64)> {v.as_array().map(|a|a.iter().filter_map(|p|Some((p[0].as_f64()?,p[1].as_f64()?))).collect()).unwrap_or_default()}
fn hash(s:&str)->u64 {s.bytes().fold(14695981039346656037,|h,b|(h^b as u64).wrapping_mul(1099511628211))}

fn sample_fingerprint(points:&[(f64,f64)])->String {
    let mut digest=14695981039346656037u64;
    for &(x,y) in points {for coordinate in [x,y] {for byte in coordinate.to_bits().to_le_bytes() {
        digest=(digest^byte as u64).wrapping_mul(1099511628211);
    }}}
    format!("{digest:016x}")
}

#[derive(Clone,Copy,Default)]
struct ExtractionPolicy {
    partial_outlines:bool,
    outline_directions:bool,
    pupil_raw_directions:bool,
    pupil_tangent_support:bool,
    conic_pupil_paths:bool,
    augmented_pupil_paths:bool,
    sliding_pupil_luma:bool,
    pupil_profile_footprint:bool,
    subpixel_pupil_peaks:bool,
    connected_pupil_width:bool,
    reject_weak_pupil_core:bool,
    semantic_pupil_contours:bool,
    pupil_search_radius_px:Option<f64>,
    pupil_weight_scale:Option<f64>,
    raw_uncertainty:bool,
    raw_outer_spread:bool,
    raw_outer_position:bool,
    without_pupil:bool,
    all_boundary_samples:bool,
    coherent_pupil_arcs:bool,
    shape_pupil_arcs:bool,
    optical_pupil_arcs:bool,
    mask_levels:bool,
    mask_spatial:bool,
    raw_outer_candidates:bool,
    withhold_rejected_outer:bool,
}

struct Frame {
    input:Value,
    conic_pupil_paths:bool,
    augmented_pupil_paths:bool,
    sliding_pupil_luma:bool,
    pupil_profile_footprint:bool,
    subpixel_pupil_peaks:bool,
    connected_pupil_width:bool,
    reject_weak_pupil_core:bool,
    pupil_core_support:Option<PupilCoreSupport>,
    semantic_pupil_contour:Option<Value>,
    pupil_search_radius_px:Option<f64>,
    pupil_weight_scale:Option<f64>,
    pupil_directions:Option<raw_optical_uncertainty::PupilDirectionReport>,
    outer_spread:Option<Vec<raw_optical_uncertainty::OuterSpreadArc>>,
    outer_position:Option<Vec<raw_optical_uncertainty::OuterPositionArc>>,
    mask_levels:Option<Value>,
    outer_candidates:Option<Value>,
    rejected_outer_ablation:Option<Value>,
    packet:OwnedRoiEvidence,
    pose:EyePoseInput,
    validation:Vec<(usize,u32,BoundaryKind,Vec<(f64,f64)>)>,
    baseline:Option<geometry::Ellipse>,
    selected_raw_admitted:bool,
    partial_outline:PartialOutlineReport,
    pupil_ablation:Value,
    all_boundary_samples:bool,
    coherent_pupil_arcs:bool,
    shape_pupil_arcs:bool,
    optical_pupil_arcs:bool,
}

/// Offline causal control. Remove both observed pupil arcs and their search
/// hints, but keep the already-extracted outer evidence, RAW policy and scale.
/// Remap surviving hints before training/validation decimation assigns indices.
fn remove_pupil_evidence(packet:&mut OwnedRoiEvidence)->Value {
    let mut remap=vec![None;packet.arcs.len()];
    let mut retained=0;
    let mut removed_groups=Vec::new();
    for (index,arc) in packet.arcs.iter().enumerate() {
        if arc.kind==BoundaryKind::PupillaryBoundary {removed_groups.push(arc.evidence_group);}
        else {remap[index]=Some(retained);retained+=1;}
    }
    packet.arcs.retain(|a|a.kind!=BoundaryKind::PupillaryBoundary);
    let hints=packet.conics.len();
    packet.conics.retain(|c|c.kind!=BoundaryKind::PupillaryBoundary);
    for conic in &mut packet.conics {
        conic.supporting_arc_indices=conic.supporting_arc_indices.iter()
            .filter_map(|&index|remap.get(index).copied().flatten()).collect();
    }
    json!({"removed_pupil_arc_groups":removed_groups,"removed_pupil_hints":hints-packet.conics.len(),
        "contract":"Pupil observations and initialization hints removed after shared extraction; all other observations, pose/scale inputs and source identities preserved."})
}

/// Offline causal control: the existing RAW rejection removes only the outer
/// observation family. Independently extracted pupil/inner support survives.
fn remove_outer_evidence(packet:&mut OwnedRoiEvidence)->Value {
    let mut remap=vec![None;packet.arcs.len()];
    let mut retained=0;
    let mut removed_groups=Vec::new();
    for (index,arc) in packet.arcs.iter().enumerate() {
        if arc.kind==BoundaryKind::OuterLimbus {removed_groups.push(arc.evidence_group);}
        else {remap[index]=Some(retained);retained+=1;}
    }
    packet.arcs.retain(|a|a.kind!=BoundaryKind::OuterLimbus);
    let hints=packet.conics.len();
    packet.conics.retain(|c|c.kind!=BoundaryKind::OuterLimbus);
    for conic in &mut packet.conics {
        conic.supporting_arc_indices=conic.supporting_arc_indices.iter()
            .filter_map(|&index|remap.get(index).copied().flatten()).collect();
    }
    json!({"applied":"selected-mask-failed-existing-raw-gate",
        "removed_outer_arc_groups":removed_groups,"removed_outer_hints":hints-packet.conics.len(),
        "preserved_arcs":packet.arcs.iter().map(|a|json!({"kind":format!("{:?}",a.kind),
            "group":a.evidence_group,"points_digest":sample_fingerprint(&a.points_roi_px),
            "points":a.points_roi_px.len()})).collect::<Vec<_>>(),
        "contract":"Remove only rejected outer observations and their conic search hints after native extraction. Pupil/inner observations, RAW policy, source identity and original scene initialization stay fixed. The RAW gate is an engineering admission rule, not calibrated anatomical identity."})
}

fn prepare(row:Value,partial_outlines:bool)->Result<Frame,String> {
    prepare_with_directions(row,partial_outlines,false)
}

fn prepare_with_directions(row:Value,partial_outlines:bool,outline_directions:bool)->Result<Frame,String> {
    prepare_with_policy(row,ExtractionPolicy {partial_outlines,outline_directions,..Default::default()})
}

/// Accept only the exact selected semantic contour from the native audit.
/// A recovered guide has no semantic retained-run provenance and uses the
/// original RAW adapter. Malformed or mismatched supplied evidence is an error.
fn semantic_pupil_review(row:&Value,raw:&[u16],guide:geometry::Ellipse)
    -> Result<Option<(outline_conic_segments::ContourFitEvidence,Value)>,String> {
    let Some(evidence)=row.get("selected_semantic_pupil_contour").filter(|v|!v.is_null()) else {return Ok(None);};
    let input=&row["input"];let selection=&evidence["selection"];let candidate=&evidence["candidate"];
    let source=&candidate["source"];let meta=&input["frame"];
    if &evidence["input"]!=input || &selection["input"]!=input || &candidate["input"]!=input
        || candidate["schema"]!="buttercup-semantic-pupil-contour-diagnostic-v1"
        || candidate["prompt"]!=2 || selection["semantic_fit_present"]!=true
        || selection["independent_observation"]!=true
        || selection["semantic_query"].as_u64().is_none()
        || selection["semantic_query"]!=candidate["query"]
        || candidate["source"]!=selection["source"]
        || number(&source["eye_index"]).and_then(|v|v.checked_add(1))!=number(&meta["eye_id"])
        || number(&source["sequence"])!=number(&meta["sequence"])
        || number(&source["timestamp_ns"])!=number(&meta["timestamp_ns"])
        || source["sensor_origin"]!=json!([meta["sensor_x"],meta["sensor_y"]])
        || source["dimensions"]!=json!([meta["width"],meta["height"]])
        || ["geometry_admitted","center_admitted","history_admitted","raw_admitted"]
            .iter().any(|k|candidate[*k]!=true)
        || ellipse(&candidate["ellipse"])!=Some(guide)
        || ellipse(&selection["selected_ellipse"])!=Some(guide) {
        return Err("semantic pupil contour is not the source-matched selected guide".into());
    }
    use sha2::{Digest,Sha256};
    let mut digest=Sha256::new();for pixel in raw {digest.update(pixel.to_le_bytes());}
    if candidate["source"]["native_u16le_sha256"]!=format!("{:x}",digest.finalize()) {
        return Err("semantic pupil contour has a different native RAW image".into());
    }
    let fit=&candidate["clipped_component_fit"];
    let retained:Vec<(f64,f64)>=serde_json::from_value(fit["retained_points_native"].clone())
        .map_err(|_|"invalid semantic pupil points")?;
    let censored:Vec<(f64,f64)>=serde_json::from_value(fit["flat_tire_points_native"].clone())
        .map_err(|_|"invalid semantic pupil censored points")?;
    let runs:Vec<Vec<usize>>=serde_json::from_value(fit["retained_runs"].clone())
        .map_err(|_|"invalid semantic pupil runs")?;
    let width=integer(&input["frame"],"width")? as f64;
    let height=integer(&input["frame"],"height")? as f64;
    let mut used=std::collections::HashSet::new();
    if retained.len()<3 || runs.is_empty()
        || retained.iter().chain(&censored).any(|&(x,y)|!x.is_finite()||!y.is_finite()
            || x<0.0 || y<0.0 || x>=width || y>=height)
        || runs.iter().flatten().any(|&index|index>=retained.len()||!used.insert(index)) {
        return Err("semantic pupil contour has invalid bounds or duplicated run support".into());
    }
    let report=json!({"selected_query":candidate["query"],"retained_points":retained.len(),
        "retained_runs":runs.len(),"sample_fingerprint":sample_fingerprint(&retained),
        "contract":"Selected same-exposure semantic retained runs replace RAW pupil arcs. Model predictions, not human labels or an independent second observation. Excluded points are not reinstated."});
    Ok(Some((outline_conic_segments::ContourFitEvidence {ellipse:guide,source_component_area_px:0.0,
        retained_points:Arc::new(retained),conic_segments:Arc::new(runs),
        flat_tire_points:Arc::new(censored),upper_flat_tire:false,lower_flat_tire:false},report)))
}

fn prepare_with_policy(row:Value,policy:ExtractionPolicy)->Result<Frame,String> {
    let ExtractionPolicy {partial_outlines,outline_directions,pupil_raw_directions,pupil_tangent_support,conic_pupil_paths,augmented_pupil_paths,sliding_pupil_luma,pupil_profile_footprint,subpixel_pupil_peaks,connected_pupil_width,reject_weak_pupil_core,semantic_pupil_contours,pupil_search_radius_px,pupil_weight_scale,raw_uncertainty,raw_outer_spread,raw_outer_position,without_pupil,all_boundary_samples,coherent_pupil_arcs,shape_pupil_arcs,optical_pupil_arcs,mask_levels,mask_spatial,raw_outer_candidates,withhold_rejected_outer}=policy;
    if semantic_pupil_contours && (pupil_raw_directions || pupil_tangent_support || coherent_pupil_arcs
        || shape_pupil_arcs || optical_pupil_arcs || conic_pupil_paths || augmented_pupil_paths
        || sliding_pupil_luma || pupil_profile_footprint || subpixel_pupil_peaks || connected_pupil_width
        || reject_weak_pupil_core || pupil_search_radius_px.is_some() || pupil_weight_scale.is_some()
        || raw_uncertainty || without_pupil) {
        return Err("compare semantic pupil contours separately from RAW pupil experiments".into());
    }
    if sliding_pupil_luma && (pupil_raw_directions || raw_uncertainty) {
        return Err("compare sliding pupil photometry separately from grid-based direction/uncertainty experiments".into());
    }
    if pupil_weight_scale.is_some_and(|s|!s.is_finite() || s<=0.0 || s>1.0) {return Err("pupil weight scale must be in (0, 1]".into());}
    if pupil_search_radius_px.is_some_and(|r|!r.is_finite() || r<=0.0 || r>12.0) {return Err("pupil search radius must be in (0, 12] native pixels".into());}
    if [coherent_pupil_arcs,shape_pupil_arcs,optical_pupil_arcs,conic_pupil_paths,augmented_pupil_paths].into_iter().filter(|v|*v).count()>1 {return Err("select one pupil path experiment".into());}
    if [raw_outer_candidates,partial_outlines,withhold_rejected_outer].into_iter().filter(|v|*v).count()>1 {
        return Err("compare RAW outer candidates, partial-mask extraction and outer withholding separately".into());
    }
    if mask_spatial && !mask_levels {return Err("spatial sensitivity requires native mask-level evidence".into());}
    if [raw_uncertainty,raw_outer_spread,raw_outer_position].into_iter().filter(|v|*v).count()>1 {
        return Err("select one RAW uncertainty experiment".into());
    }
    let input=row["input"].clone();
    let meta=&input["frame"];
    let eye=integer(meta,"eye_id")?;
    let origin=[integer(meta,"sensor_x")? as u32,integer(meta,"sensor_y")? as u32];
    let size=[integer(meta,"width")? as u32,integer(meta,"height")? as u32];
    let clock=input["clock_lineage"].as_str().ok_or("missing source lineage")?;
    let mut packet=OwnedRoiEvidence {exposure:ExposureKey {roi:RoiId(eye as u32),
        clock:SourceClock {domain:1,epoch:hash(clock)},sequence:integer(meta,"sequence")?,timestamp_ns:integer(meta,"timestamp_ns")?},
        sensor_origin_px:origin,dimensions_px:size,arcs:Vec::new(),conics:Vec::new(),detail_reliability:None};
    let candidates=row["candidates"].as_array().ok_or("missing candidates")?;
    let selected_query=row["selected_query"].as_u64();
    let selected=selected_query.and_then(|q|candidates.iter().find(|c|c["query"].as_u64()==Some(q)))
        .or_else(||candidates.iter().find(|c|ellipse(&c["baseline_ellipse"]).is_some()));
    let selected_raw_admitted=selected.is_some_and(|c|c["baseline_raw_admitted"]==true);
    let baseline=selected.and_then(|c|ellipse(&c["baseline_ellipse"]));
    let try_partial=partial_outlines&&(baseline.is_none()||!selected_raw_admitted);
    let pupil=ellipse(&row["pupil_void"]["ellipse"]);
    let raw=if raw_uncertainty || raw_outer_spread || raw_outer_position || outline_directions || pupil.is_some() || try_partial || raw_outer_candidates {
        let mut file=File::open(input["raw_file"].as_str().ok_or("missing RAW file")?).map_err(|e|e.to_string())?;
        file.seek(SeekFrom::Start(integer(&input,"raw_offset")?)).map_err(|e|e.to_string())?;
        let mut bytes=vec![0;integer(&input,"raw_length")? as usize];file.read_exact(&mut bytes).map_err(|e|e.to_string())?;
        Some(raw10::try_unpack_raw10(&bytes,size[0] as usize,size[1] as usize,integer(meta,"stride")? as usize)?)
    } else {None};
    // A rejected complete conic is not stronger evidence than an incomplete
    // outline. In this explicit experiment, re-extract RAW-supported measured
    // arcs instead of keeping the rejected fit's unchecked sections as well.
    let mut outer_candidates=None;
    if let Some((selected,baseline))=selected.zip(baseline).filter(|_|!try_partial) {
      if raw_outer_candidates && !selected_raw_admitted {
        let report=outline_conic_segments::sparse_evidence::outer_candidates::append(&mut packet,raw.as_deref().unwrap(),baseline,0);
        outer_candidates=Some(json!({"applied":"selected-mask-failed-existing-raw-gate","report":report,
            "arcs":packet.arcs.iter().map(|a|json!({"group":a.evidence_group,"points":a.points_roi_px,
                "band_px":a.normal_band_half_width_px,"raw_contrast":a.detector_score})).collect::<Vec<_>>(),
            "contract":"Four neighboring ellipse guides locate bounded native RAW searches. Only observed positive edge peaks become outer-iris candidates. Fixed image-angle sectors share alternative budgets. Rejected SAM points and their logit profiles are replaced; independently extracted pupil evidence and original scene initialization remain unchanged. No anatomical identity or calibrated confidence guarantee."}));
      } else {
        let retained=points(&selected["baseline_retained"]);
        let segments=selected["baseline_retained_segments"].as_array().map(|segments|segments.iter().filter_map(|run| {
            Some(run.as_array()?.iter().filter_map(|i|i.as_u64().map(|i|i as usize)).collect::<Vec<_>>())
        }).collect::<Vec<_>>()).unwrap_or_default();
        // Missing contour provenance is unavailable, not a dense ellipse.
        let review=outline_conic_segments::ContourFitEvidence {ellipse:baseline,source_component_area_px:0.0,
            retained_points:Arc::new(retained),conic_segments:Arc::new(segments),
            flat_tire_points:Arc::new(points(&selected["baseline_censored"])),upper_flat_tire:false,lower_flat_tire:false};
        append_retained_sam_arcs_with_direction_policy(&mut packet,&review,0,
            raw.as_deref().filter(|_|outline_directions));
        if !selected_raw_admitted {
            for arc in &mut packet.arcs {arc.normal_band_half_width_px=5.0;}
        }
      }
    }
    let mut pupil_directions=None;
    let mut pupil_core_support=None;
    let mut semantic_pupil_contour=None;
    let mut partial_outline=PartialOutlineReport::default();
    if let Some(raw)=raw.as_ref() {
        let sampling=if sliding_pupil_luma {RawSampling::SlidingBox} else {RawSampling::CfaGrid};
        if let Some((pupil,mut config))=pupil.zip(baseline.and_then(|outer|
            RawArcConfig::for_pupil_with_sampling(&raw,size[0] as usize,size[1] as usize,outer,sampling))) {
            config.profile_footprint=pupil_profile_footprint;
            config.subpixel_peaks=subpixel_pupil_peaks;
            config.connected_peak_width=connected_pupil_width;
            config.reject_weak_pupil_core=reject_weak_pupil_core;
            if reject_weak_pupil_core {
                pupil_core_support=Some(outline_conic_segments::sparse_evidence::pupil_core_support(
                    raw,size[0] as usize,size[1] as usize,pupil,config));
            }
            if let Some(radius)=pupil_search_radius_px {config.radial_search_px=radius;}
            let semantic=if semantic_pupil_contours {semantic_pupil_review(&row,raw,pupil)?} else {None};
            if let Some((review,report))=semantic {
                append_retained_boundary_arcs(&mut packet,&review,100,BoundaryKind::PupillaryBoundary,None);
                semantic_pupil_contour=Some(report);
            }
            else if augmented_pupil_paths {append_augmented_raw_ring_arcs(&mut packet,&raw,pupil,BoundaryKind::PupillaryBoundary,100,config);}
            else if conic_pupil_paths {append_conic_associated_raw_ring_arcs(&mut packet,&raw,pupil,BoundaryKind::PupillaryBoundary,100,config);}
            else if optical_pupil_arcs {append_optical_raw_ring_arcs(&mut packet,&raw,pupil,BoundaryKind::PupillaryBoundary,100,config);}
            else if shape_pupil_arcs {append_shape_checked_raw_ring_arcs(&mut packet,&raw,pupil,BoundaryKind::PupillaryBoundary,100,config);}
            else {append_raw_ring_arcs_with_cohesion(&mut packet,&raw,pupil,BoundaryKind::PupillaryBoundary,100,config,coherent_pupil_arcs);}
            if pupil_raw_directions {
                let mut report=raw_optical_uncertainty::measure_pupil_directions(&mut packet,&raw,config.maximum_profile_luma_raw10);
                if pupil_tangent_support {report.support_caps=raw_optical_uncertainty::cap_pupil_tangent_support(&mut packet);}
                pupil_directions=Some(report);
            }
        }
        if try_partial {
            let mut ranked=candidates.iter().filter(|c|c["semantic_score"].as_f64().is_some_and(f64::is_finite)).collect::<Vec<_>>();
            ranked.sort_by(|a,b|b["semantic_score"].as_f64().unwrap().total_cmp(&a["semantic_score"].as_f64().unwrap()));
            let outlines=ranked.iter().take(4).map(|c|(points(&c["outline"]),c["semantic_score"].as_f64())).collect::<Vec<_>>();
            let candidates=outlines.iter().map(|(points,score)|OutlineCandidate {points_roi_px:points,detector_score:*score}).collect::<Vec<_>>();
            partial_outline=append_unfitted_outline_arcs(&mut packet,&raw,&candidates,(size[0] as f64*0.5,size[1] as f64*0.5),200);
        }
    }
    if raw_uncertainty {
        if let Some(raw)=raw.as_deref() { outline_conic_segments::sparse_evidence::uncertainty::measure(&mut packet,raw); }
    }
    let outer_spread=raw_outer_spread.then(||raw_optical_uncertainty::measure_outer_spread(&mut packet,raw.as_deref().unwrap()));
    let outer_position=raw_outer_position.then(||raw_optical_uncertainty::measure_outer_position(&mut packet,raw.as_deref().unwrap()));
    let pupil_ablation=if without_pupil {remove_pupil_evidence(&mut packet)} else {Value::Null};
    let rejected_outer_ablation=(withhold_rejected_outer && baseline.is_some() && !selected_raw_admitted)
        .then(||remove_outer_evidence(&mut packet));
    let mask_levels=if mask_levels {
        Some(if outer_candidates.is_some() {
            json!({"source_evidence":"replaced-by-native-raw-candidates","attachment":"SAM mask profiles do not belong to the replacement observations"})
        } else if rejected_outer_ablation.is_some() {
            json!({"source_evidence":"rejected-outer-withheld","attachment":"No mask profiles attached to removed outer observations"})
        } else if let Some(value)=row.get("outer_boundary_logits").filter(|v|!v.is_null()) {
            let evidence:mask_boundary_logits::Evidence=serde_json::from_value(value.clone())
                .map_err(|error|format!("invalid native mask-boundary evidence: {error}"))?;
            if selected.and_then(|c|c["query"].as_u64()).is_some_and(|q|q!=evidence.query as u64) {
                return Err("mask-boundary evidence belongs to a different selected query".into());
            }
            let mut report=json!({"source_evidence":"present","attachment":mask_boundary_logits::attach(&mut packet,&evidence,0)?});
            if mask_spatial {
                let mut points=0;
                for arc in packet.arcs.iter_mut().filter(|a|a.kind==BoundaryKind::OuterLimbus) {
                    for profile in arc.level_sets_roi.iter_mut().flatten().flatten() {
                        *profile=profile.with_spatial_sensitivity();points+=1;
                    }
                }
                report["spatial_sensitivity"]=json!({"points":points,"modes":4,
                    "contract":"Coherent +/- cos(2 theta) and +/- sin(2 theta) threshold fields from measured contour-normal angles, bounded by each original logit-level envelope. No new observations or calibrated probabilities."});
            }
            report
        } else {json!({"source_evidence":"absent"})})
    } else {None};
    // These outer-observation experiments preserve the original scene prior to
    // isolate evidence changes. Its baseline-derived center is not independent
    // anatomical truth. Partial-outline mode separately uses the crop center.
    let center=baseline.filter(|_|!try_partial).map(|e|[e.center.0+origin[0] as f64,e.center.1+origin[1] as f64])
        .unwrap_or([origin[0] as f64+size[0] as f64*0.5,origin[1] as f64+size[1] as f64*0.5]);
    let scale=&input["scale_hint"];
    let scale=scale["pixels_per_10mm"].as_f64().zip(scale["bounds_px_per_10mm"].as_array()).and_then(|(n,b)|Some([n,b.first()?.as_f64()?,b.get(1)?.as_f64()?]));
    let pose=EyePoseInput {limbus_center_sensor_px:center,pixels_per_10mm:scale};
    let mut validation=Vec::new();
    for (index,arc) in packet.arcs.iter_mut().enumerate() {
        if !all_boundary_samples && arc.points_roi_px.len()>=6 {
            validation.push((index,arc.evidence_group,arc.kind,arc.points_roi_px.iter().skip(1).step_by(2).copied().collect()));
            arc.sampling_support_px=arc.sampling_support_px.take().map(|support|
                roi_evidence::reduce_sampling_support(&support,
                    &(0..arc.points_roi_px.len()).step_by(2).collect::<Vec<_>>())
                    .expect("extractor supplies valid fixed profile support"));
            arc.points_roi_px=arc.points_roi_px.iter().step_by(2).copied().collect();
            arc.outward_normals_roi=arc.outward_normals_roi.take().map(|normals|normals.into_iter().step_by(2).collect());
            arc.level_sets_roi=arc.level_sets_roi.take().map(|levels|levels.into_iter().step_by(2).collect());
        }
    }
    // Apply the ablation to the actual fitted samples, including in held-out
    // runs; the removed probes must not set the remaining information budget.
    if let Some(scale)=pupil_weight_scale {raw_optical_uncertainty::scale_pupil_information(&mut packet,scale);}
    if semantic_pupil_contours && row.get("selected_semantic_pupil_contour").is_some_and(|v|!v.is_null())
        && semantic_pupil_contour.is_none() {return Err("selected semantic contour did not reach pupil extraction".into());}
    Ok(Frame {input,conic_pupil_paths,augmented_pupil_paths,sliding_pupil_luma,pupil_profile_footprint,subpixel_pupil_peaks,connected_pupil_width,reject_weak_pupil_core,pupil_core_support,semantic_pupil_contour,pupil_search_radius_px,pupil_weight_scale,pupil_directions,outer_spread,outer_position,mask_levels,outer_candidates,rejected_outer_ablation,packet,pose,validation,baseline,selected_raw_admitted,partial_outline,pupil_ablation,all_boundary_samples,coherent_pupil_arcs,shape_pupil_arcs,optical_pupil_arcs})
}

fn heldout(solution:&JointConicSolution,frames:[Option<&Frame>;2])->[Value;2] {
    std::array::from_fn(|eye| {
        let Some(frame)=&frames[eye] else {return Value::Null;};
        let mut values=Vec::new();
        let mut supported=Vec::new();
        let mut groups=Vec::new();
        for (index,group,kind,points) in &frame.validation {
            let boundary=match kind {BoundaryKind::OuterLimbus=>0,BoundaryKind::InnerLimbus=>1,BoundaryKind::PupillaryBoundary=>2,BoundaryKind::Unclassified=>continue};
            let Some(e)=solution.ellipses_roi_px[eye][boundary] else {continue;};
            let residuals=points.iter().map(|&p|conic_solver::ellipse_residual(p,e)).collect::<Vec<_>>();
            let used=solution.arcs.iter().any(|a|a.exposure.roi==frame.packet.exposure.roi&&a.arc_index==*index&&a.used);
            if used {supported.extend_from_slice(&residuals);}
            groups.push(json!({"arc":index,"group":group,"kind":format!("{kind:?}"),"used":used,
                "points":residuals.len(),"sample_fingerprint":sample_fingerprint(points),
                "rms_px":(residuals.iter().map(|v|v*v).sum::<f64>()/residuals.len() as f64).sqrt()}));
            values.extend(residuals);
        }
        json!({"points":values.len(),"rms_px":(!values.is_empty()).then(||(values.iter().map(|v|v*v).sum::<f64>()/values.len() as f64).sqrt()),
            "supported_points":supported.len(),"supported_rms_px":(!supported.is_empty()).then(||(supported.iter().map(|v|v*v).sum::<f64>()/supported.len() as f64).sqrt()),
            "groups":groups})
    })
}

fn solution_json(result:Result<JointConicSolution,JointConicUnavailable>,frames:[Option<&Frame>;2],elapsed:f64)->Value {
    match result {
        Err(reason)=>json!({"available":false,"reason":format!("{reason:?}"),"elapsed_ms":elapsed}),
        Ok(solution)=> {
            let mut output=json!({"available":true,"target_camera_mm":solution.target_camera_mm,
            "local_uncertainty":solution.local_uncertainty.as_ref().map(|u|u.json()),
            "posterior":solution.posterior.as_ref().map(|p|p.json()),
            "target_search_chart":{"frame":"reference-to-camera-tangent-plane","slopes":solution.target_viewpoint_slopes,
                "reference_camera_mm":solution.target_reference_camera_mm,
                "axial_distance_mm":solution.target_viewpoint_axial_distance_mm,"distance_axis":"reference-to-camera",
                "slope_limit":solution.target_viewpoint_slope_limit,"active_bounds":solution.target_viewpoint_bounds_active},
            "eye_centers_camera_mm":solution.eye_centers_camera_mm,"eye_normals":solution.eye_normals,
            "eye_gaze_directions":solution.eye_gaze_directions,"surface_axis_alignment_radians":solution.surface_axis_alignment_radians,
            "contributing_eyes":solution.contributing_eyes,"cost":solution.robust_cost,
            "modeled_eyes":solution.modeled_eyes,"unlocalized_eye_cost":solution.unlocalized_eye_cost,
            "alternative_cost_margin":solution.alternative_cost_margin,
            "alternative_target_camera_mm":solution.alternative_target_camera_mm,
            "hypotheses":solution.hypotheses_evaluated,"refinement_steps":solution.refinement_steps,
            "hypotheses_by_association":solution.hypotheses_by_association,
            "outer_ellipses":solution.ellipses_roi_px.map(|e|ellipse_json(e[0])),
            "fitted_ellipses":solution.ellipses_roi_px.map(|e|e.map(ellipse_json)),
            "fitted_ellipses_contract":"Source-ROI model conics in outer/inner/pupil order; an unobserved boundary can have a latent model ellipse. Only source-matched used support is measured evidence.",
            "support":solution.arcs.iter().map(|a|json!({"roi":a.exposure.roi.0,"kind":format!("{:?}",a.kind),
                "group":a.evidence_group,"arc":a.arc_index,"rms_px":a.rms_px,"sigma_px":a.sigma_px,"used":a.used,
                "points_roi_px":a.points_roi_px,
                "source":{"roi_id":a.exposure.roi.0,"sequence":a.exposure.sequence.to_string(),
                    "timestamp_ns":a.exposure.timestamp_ns.to_string()},
                "boundary_normal_samples":a.boundary_normal_samples,"boundary_normal_rms_radians":a.boundary_normal_rms_radians,
                "support_length_px":a.support_length_px,"evidence_weight":a.evidence_weight})).collect::<Vec<_>>(),
            "withheld_sample_residuals":heldout(&solution,frames),"elapsed_ms":elapsed,
            "sn_feida_mm2":std::array::from_fn::<_,2,_>(|eye| {
                let frame=frames[eye].as_ref()?;let scale=frame.pose.pixels_per_10mm?[0]/10.0;
                if !solution.arcs.iter().any(|a|a.used&&a.exposure.roi==frame.packet.exposure.roi&&a.kind==BoundaryKind::OuterLimbus) {return None;}
                let e=solution.ellipses_roi_px[eye][0]?;
                Some(std::f64::consts::PI*(e.major_radius/scale).powi(2))
            }),
            });
            if !solution.mask_level_families.is_empty() {
                output["mask_level_families"]=json!(solution.mask_level_families);
                for (value,arc) in output["support"].as_array_mut().unwrap().iter_mut().zip(&solution.arcs) {
                    if let Some(level)=arc.mask_level {value["mask_level"]=json!(level);}
                }
            }
            if !solution.arc_alternative_marginals.is_empty() {
                output["arc_alternative_marginals"]=json!(solution.arc_alternative_marginals);
            }
            output
        },
    }
}

fn evaluate(frames:[Option<Frame>;2],export_sparse:bool)->Value {
    // Geometry unit/component fixtures deliberately retain their unconstrained prior.
    evaluate_with_distribution(frames,export_sparse,false,eye_scene_model::CameraMount::Flexible)
}

fn evaluate_with_distribution(frames:[Option<Frame>;2],export_sparse:bool,probabilistic:bool,
    camera_mount:eye_scene_model::CameraMount)->Value {
    let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
    let prepared=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.prepare()));
    let evidence=prepared.each_ref().map(|e|e.as_ref().map(|p|p.evidence()));
    let camera=PinholeCamera {focal_px:[4000.0,4000.0],principal_px:[4000.0,3000.0]};
    let base=json!({"inputs":frames.each_ref().map(|f|f.as_ref().map(|f|&f.input)),
        "pupil_ablation":frames.each_ref().map(|f|f.as_ref().map(|f|&f.pupil_ablation)),
        "pupil_raw_directions":frames.each_ref().map(|f|f.as_ref().and_then(|f|f.pupil_directions.as_ref())),
        "conic_pupil_paths":frames.each_ref().map(|f|f.as_ref().map(|f|f.conic_pupil_paths)),
        "augmented_pupil_paths":frames.each_ref().map(|f|f.as_ref().map(|f|f.augmented_pupil_paths)),
        "pupil_weight_scale":frames.each_ref().map(|f|f.as_ref().and_then(|f|f.pupil_weight_scale)),
        "all_boundary_samples":frames.each_ref().map(|f|f.as_ref().map(|f|f.all_boundary_samples)),
        "shape_pupil_arcs":frames.each_ref().map(|f|f.as_ref().map(|f|f.shape_pupil_arcs)),
        "optical_pupil_arcs":frames.each_ref().map(|f|f.as_ref().map(|f|f.optical_pupil_arcs)),
        "coherent_pupil_arcs":frames.each_ref().map(|f|f.as_ref().map(|f|f.coherent_pupil_arcs)),
        "baseline_sam_outer":frames.each_ref().map(|f|ellipse_json(f.as_ref().and_then(|f|f.baseline))),
        "raw_admitted":frames.each_ref().map(|f|f.as_ref().map(|f|f.selected_raw_admitted)),
        "observed_arc_groups":frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.arcs.len())),
        "partial_outline":frames.each_ref().map(|f|f.as_ref().map(|f|json!({"candidates":f.partial_outline.candidates,
            "censored_samples":f.partial_outline.censored_samples,"unsupported_samples":f.partial_outline.unsupported_samples,
            "inward_notch_samples":f.partial_outline.inward_notch_samples,
            "emitted_arcs":f.partial_outline.emitted_arcs}))),
        "contract":"shared latent fixation versus separate monocular optimizations of the SAME training arcs; no averaged gaze; held-out points condition on upstream detector segmentation/search and are absent in all-boundary-samples mode. Neither metric pose nor gaze accuracy is ground truth."});
    let mut row=base;
    row["camera_mount_assumption"]=json!(camera_mount.label());
    if frames.iter().flatten().any(|f|f.pupil_profile_footprint) {row["pupil_profile_footprint"]=json!(true);}
    if frames.iter().flatten().any(|f|f.subpixel_pupil_peaks) {row["subpixel_pupil_peaks"]=json!(true);}
    if frames.iter().flatten().any(|f|f.connected_pupil_width) {row["connected_pupil_width"]=json!(true);}
    if frames.iter().flatten().any(|f|f.reject_weak_pupil_core) {
        row["pupil_core_support"]=json!(frames.each_ref().map(|f|f.as_ref().and_then(|f|f.pupil_core_support)));
    }
    if frames.iter().flatten().any(|f|f.semantic_pupil_contour.is_some()) {
        row["semantic_pupil_contour"]=json!(frames.each_ref().map(|f|f.as_ref().and_then(|f|f.semantic_pupil_contour.as_ref())));
    }
    if frames.iter().flatten().any(|f|f.pupil_search_radius_px.is_some()) {row["pupil_search_radius_px"]=json!(frames.each_ref().map(|f|f.as_ref().and_then(|f|f.pupil_search_radius_px)));}
    if frames.iter().flatten().any(|f|f.sliding_pupil_luma) {
        row["sliding_pupil_luma"]=json!(true);
    }
    if frames.iter().flatten().any(|f|f.outer_spread.is_some()) {
        row["outer_raw_spread"]=json!(frames.each_ref().map(|f|f.as_ref().map(|f|&f.outer_spread)));
    }
    if frames.iter().flatten().any(|f|f.outer_position.is_some()) {
        row["outer_raw_position"]=json!(frames.each_ref().map(|f|f.as_ref().map(|f|&f.outer_position)));
    }
    if frames.iter().flatten().any(|f|f.mask_levels.is_some()) {
        row["mask_boundary_profiles"]=json!(frames.each_ref().map(|f|f.as_ref().map(|f|&f.mask_levels)));
    }
    if frames.iter().flatten().any(|f|f.outer_candidates.is_some()) {
        row["raw_outer_candidates"]=json!(frames.each_ref().map(|f|f.as_ref().map(|f|&f.outer_candidates)));
    }
    if frames.iter().flatten().any(|f|f.rejected_outer_ablation.is_some()) {
        row["rejected_outer_ablation"]=json!(frames.each_ref().map(|f|f.as_ref().map(|f|&f.rejected_outer_ablation)));
    }
    if export_sparse {
        row["sparse_evidence"]=json!(frames.each_ref().map(|f|f.as_ref().map(|f|json!({
            "arcs":f.packet.arcs.iter().map(|a|json!({"group":a.evidence_group,"kind":format!("{:?}",a.kind),
                "points":a.points_roi_px,"band_half_width_px":a.normal_band_half_width_px,
                "raw_localization_sigma_px":a.localization_sigma_px,
                "outward_normals":a.outward_normals_roi.as_ref().map(|normals|normals.iter().map(|n|n.map(|n|json!({
                    "unit_outward_roi":n.unit_outward_roi,"angular_sigma_radians":n.angular_sigma_radians}))).collect::<Vec<_>>())})).collect::<Vec<_>>(),
            "seeds":f.packet.conics.iter().map(|c|json!({"kind":format!("{:?}",c.kind),"ellipse":ellipse_json(Some(c.ellipse_roi_px))})).collect::<Vec<_>>()
        }))));
    }
    let Some(scene)=approximate_scene(camera,poses) else {row["error"]=json!("no coarse scene support");return row;};
    row["scale_provenance"]=json!(scene.scale_provenance.map(|p|p.map(|p|format!("{p:?}"))));
    row["independent_pixels_per_mm"]=json!(scene.independent_pixels_per_mm);
    for (name,mask) in [("joint",[true,true]),("monocular_right",[true,false]),("monocular_left",[false,true])] {
        let started=Instant::now();
        let request=JointConicRequest {
            eyes:std::array::from_fn(|eye|if mask[eye] {evidence[eye]} else {None}),scene:&scene.prior,
            maximum_hypotheses:16,maximum_refinements:12,maximum_source_skew_ns:0,
            exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0,
        };
        let count=if camera_mount==eye_scene_model::CameraMount::Flexible {1}else{16};
        let result=conic_solver::joint::solve_joint_conics_with_mount(request,count,camera_mount,probabilistic)
            .map(|mut hypotheses|hypotheses.remove(0));
        row[name]=solution_json(result,frames.each_ref().map(Option::as_ref),started.elapsed().as_secs_f64()*1000.0);
    }
    row
}

fn run()->Result<(),String> {
    let camera_mount=eye_scene_model::CameraMount::for_offline_checks()?;
    let mut args=std::env::args().skip(1);
    let output=PathBuf::from(args.next().ok_or("usage: buttercup_stereo_conic_eval OUTPUT.jsonl SAM_CACHE.jsonl...")?);
    let mut files=Vec::new();
    let mut maximum_frames_per_cache=usize::MAX;
    let mut extraction=ExtractionPolicy::default();
    let mut export_sparse=false;
    let mut extract_only=false;
    let mut source_order_replay=false;
    let mut export_hypotheses=false;
    let mut probabilistic=false;
    let mut arrival_delay_ns=[0u64;2];
    while let Some(arg)=args.next() {
        if arg=="--max-frames-per-cache" {
            maximum_frames_per_cache=args.next().ok_or("missing frame limit")?.parse::<usize>().map_err(|e|e.to_string())?;
            if maximum_frames_per_cache==0 {return Err("frame limit must be positive".into());}
        } else if arg=="--extract-only" {extract_only=true;}
        else if arg=="--partial-outlines" {extraction.partial_outlines=true;}
        else if arg=="--export-sparse-evidence" {export_sparse=true;}
        else if arg=="--source-order-replay" {source_order_replay=true;}
        else if arg=="--export-hypotheses" {export_hypotheses=true;}
        else if let Some(scale)=arg.strip_prefix("--pupil-weight-scale=") {extraction.pupil_weight_scale=Some(scale.parse().map_err(|_|"invalid pupil weight scale")?);}
        else if arg=="--conic-pupil-paths" {extraction.conic_pupil_paths=true;}
        else if arg=="--augment-pupil-paths" {extraction.augmented_pupil_paths=true;}
        else if arg=="--pupil-profile-footprint" {extraction.pupil_profile_footprint=true;}
        else if arg=="--subpixel-pupil-peaks" {extraction.subpixel_pupil_peaks=true;}
        else if arg=="--connected-pupil-width" {extraction.connected_pupil_width=true;}
        else if arg=="--reject-weak-pupil-core" {extraction.reject_weak_pupil_core=true;}
        else if let Some(radius)=arg.strip_prefix("--pupil-search-radius=") {extraction.pupil_search_radius_px=Some(radius.parse().map_err(|_|"invalid pupil search radius")?);}
        else if arg=="--sliding-pupil-luma" {extraction.sliding_pupil_luma=true;}
        else if arg=="--pupil-tangent-support" {extraction.pupil_raw_directions=true;extraction.pupil_tangent_support=true;}
        else if arg=="--pupil-raw-directions" {extraction.pupil_raw_directions=true;}
        else if arg=="--retained-outline-directions" {extraction.outline_directions=true;}
        else if arg=="--raw-boundary-uncertainty" {extraction.raw_uncertainty=true;}
        else if arg=="--raw-outer-spread" {extraction.raw_outer_spread=true;}
        else if arg=="--raw-outer-position" {extraction.raw_outer_position=true;}
        else if arg=="--raw-outer-candidates" {extraction.raw_outer_candidates=true;}
        else if arg=="--withhold-rejected-outer" {extraction.withhold_rejected_outer=true;}
        else if arg=="--probabilistic" {probabilistic=true;}
        else if arg=="--without-pupil" {extraction.without_pupil=true;}
        else if arg=="--all-boundary-samples" {extraction.all_boundary_samples=true;}
        else if arg=="--semantic-pupil-contours" {extraction.semantic_pupil_contours=true;}
        else if arg=="--optical-pupil-arcs" {extraction.optical_pupil_arcs=true;}
        else if arg=="--shape-pupil-arcs" {extraction.shape_pupil_arcs=true;}
        else if arg=="--coherent-pupil-arcs" {extraction.coherent_pupil_arcs=true;}
        else if arg=="--mask-levels" {extraction.mask_levels=true;}
        else if arg=="--mask-spatial" {extraction.mask_levels=true;extraction.mask_spatial=true;}
        else if arg=="--arrival-delay-ns" {
            let eye=args.next().ok_or("missing delayed ROI id (1 or 2)")?.parse::<usize>().map_err(|e|e.to_string())?;
            if !(1..=2).contains(&eye) {return Err("delayed ROI id must be 1 or 2".into());}
            arrival_delay_ns[eye-1]=args.next().ok_or("missing arrival delay")?.parse::<u64>().map_err(|e|e.to_string())?;
        }
        else {files.push(arg);}
    }
    if files.is_empty() {return Err("at least one SAM evidence cache is required".into());}
    let allowed=std::fs::canonicalize("outputs").map_err(|e|e.to_string())?;
    if !std::fs::canonicalize(output.parent().ok_or("missing output directory")?).map_err(|e|e.to_string())?.starts_with(allowed) {return Err("output must be under outputs".into());}
    let mut writer=BufWriter::new(OpenOptions::new().create_new(true).write(true).open(output).map_err(|e|e.to_string())?);
    if extract_only {
        if source_order_replay {return Err("extract-only and source-order-replay are separate diagnostics".into());}
        for path in &files {
            for line in BufReader::new(File::open(path).map_err(|e|e.to_string())?).lines().take(maximum_frames_per_cache) {
                let row=serde_json::from_str(&line.map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                let frame=prepare_with_policy(row,extraction)?;
                let mut report=json!({"input":frame.input,"shape_pupil_arcs":extraction.shape_pupil_arcs,
                    "conic_pupil_paths":extraction.conic_pupil_paths,
                    "augmented_pupil_paths":extraction.augmented_pupil_paths,
                    "pupil_weight_scale":extraction.pupil_weight_scale,
                    "coherent_pupil_arcs":extraction.coherent_pupil_arcs,"optical_pupil_arcs":extraction.optical_pupil_arcs,
                    "pupil_raw_directions":frame.pupil_directions,
                    "arcs":frame.packet.arcs.iter().map(|a|json!({"kind":format!("{:?}",a.kind),
                        "group":a.evidence_group,"points":a.points_roi_px,"normal_band_px":a.normal_band_half_width_px,"support_length_cap_px":a.support_length_cap_px,
                        "normals":a.outward_normals_roi.as_ref().map(|ns|ns.iter().map(|n|n.map(|n|json!({"unit":n.unit_outward_roi,"sigma_radians":n.angular_sigma_radians}))).collect::<Vec<_>>())})).collect::<Vec<_>>()});
                if extraction.pupil_profile_footprint {
                    report["pupil_profile_footprint"]=json!(true);
                    for (value,arc) in report["arcs"].as_array_mut().unwrap().iter_mut().zip(&frame.packet.arcs) {
                        value["sampling_support_px"]=json!(arc.sampling_support_px);
                    }
                }
                if extraction.sliding_pupil_luma {report["sliding_pupil_luma"]=json!(true);}
                if extraction.subpixel_pupil_peaks {report["subpixel_pupil_peaks"]=json!(true);}
                if extraction.connected_pupil_width {report["connected_pupil_width"]=json!(true);}
                if extraction.reject_weak_pupil_core {report["pupil_core_support"]=json!(frame.pupil_core_support);}
                if extraction.semantic_pupil_contours {report["semantic_pupil_contour"]=json!(frame.semantic_pupil_contour);}
                if let Some(radius)=extraction.pupil_search_radius_px {report["pupil_search_radius_px"]=json!(radius);}
                serde_json::to_writer(&mut writer,&report).map_err(|e|e.to_string())?;writer.write_all(b"\n").map_err(|e|e.to_string())?;
            }
        }
        return writer.flush().map_err(|e|e.to_string());
    }
    if source_order_replay {
        return source_order::run(&files,maximum_frames_per_cache,extraction,arrival_delay_ns,export_hypotheses,probabilistic,&mut writer);
    }
    if export_hypotheses { return Err("export-hypotheses requires source-order-replay".into()); }
    if arrival_delay_ns!=[0;2] {return Err("arrival-delay-ns requires source-order-replay".into());}
    let mut pending:HashMap<(String,u64),[Option<Frame>;2]>=HashMap::new();
    let mut count=0usize;
    let mut write=|frames|->Result<(),String> {
        let row=evaluate_with_distribution(frames,export_sparse,probabilistic,camera_mount);serde_json::to_writer(&mut writer,&row).map_err(|e|e.to_string())?;
        writer.write_all(b"\n").map_err(|e|e.to_string())?;count+=1;
        if count%500==0 {writer.flush().map_err(|e|e.to_string())?;eprintln!("stereo evaluation reads={count}");}
        Ok(())
    };
    for path in files {
        for (line_number,line) in BufReader::new(File::open(&path).map_err(|e|e.to_string())?).lines().take(maximum_frames_per_cache).enumerate() {
            let row=serde_json::from_str(&line.map_err(|e|e.to_string())?).map_err(|e|format!("{path}:{}: {e}",line_number+1))?;
            let frame=prepare_with_policy(row,extraction)?;
            let eye=frame.packet.exposure.roi.0.checked_sub(1).filter(|e|*e<2).ok_or("invalid ROI")? as usize;
            let key=(frame.input["clock_lineage"].as_str().ok_or("missing lineage")?.to_owned(),frame.packet.exposure.timestamp_ns);
            let slot=pending.entry(key.clone()).or_insert_with(||[None,None]);
            if slot[eye].is_some() {return Err("duplicate eye/source in supposedly deduplicated SAM export".into());}
            slot[eye]=Some(frame);
            if slot.iter().all(Option::is_some) {write(pending.remove(&key).unwrap())?;}
        }
    }
    // A source read with a missing ROI is still evaluated and counted. It is
    // never discarded merely because the joint path has less information.
    let mut remainder=pending.into_iter().collect::<Vec<_>>();
    remainder.sort_by(|a,b|a.0.cmp(&b.0));
    for (_,frames) in remainder {write(frames)?;}
    writer.flush().map_err(|e|e.to_string())?;
    eprintln!("stereo evaluation complete reads={count}");
    Ok(())
}

fn main() {if let Err(error)=run() {eprintln!("stereo evaluation error: {error}");std::process::exit(1);}}

#[cfg(test)]
mod extraction_tests {
    use super::*;

    // Synthetic RAW stays beneath the checked runtime link and is removed
    // only by the fixture that successfully created this unique file.
    struct RawFixture(PathBuf);
    impl Drop for RawFixture {fn drop(&mut self) {let _=std::fs::remove_file(&self.0);}}

    fn rejected_outline_fixture(raw:&[u16])->(RawFixture,Value) {
        let path=PathBuf::from("outputs").join(format!("joint-evaluator-test-{}-{}.raw10",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let mut file=OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        let fixture=RawFixture(path);
        assert_eq!(raw.len(),256*192);
        for p in raw.chunks_exact(4) {
            let packed=(p[0] as u64)|((p[1] as u64)<<10)|((p[2] as u64)<<20)|((p[3] as u64)<<30);
            file.write_all(&packed.to_le_bytes()[..5]).unwrap();
        }
        let e=geometry::Ellipse {center:(128.0,96.0),major_radius:60.0,minor_radius:46.0,angle:0.0};
        let points=e.dense_points(128);
        let row=json!({"input":{"index":0,"clock_lineage":"synthetic","raw_file":fixture.0,
                "raw_offset":0,"raw_length":256*192*5/4,
                "frame":{"eye_id":1,"sensor_x":400,"sensor_y":800,"width":256,"height":192,
                    "stride":320,"timestamp_ns":20,"sequence":10}},
            "selected_query":null,"pupil_void":null,
            "candidates":[{"query":0,"semantic_score":1.0,"outline":points,
                "baseline_raw_admitted":false,"baseline_ellipse":ellipse_json(Some(e)),
                "baseline_retained":points,"baseline_retained_segments":[(0..128).collect::<Vec<_>>()],
                "baseline_censored":[]}]});
        (fixture,row)
    }

    fn mask_profile_fixture()->(RawFixture,Value) {
        let (file,mut row)=rejected_outline_fixture(&vec![0;256*192]);
        row["candidates"][0]["baseline_raw_admitted"]=json!(true);
        let plane=mask_boundary_logits::Plane {width:256,height:192,values:(0..256*192).map(|i| {
            (1.0-(((i%256) as f64-128.0)/60.0).hypot(((i/256) as f64-96.0)/46.0)) as f32*32.0
        }).collect()};
        let source=mask_boundary_logits::Source {eye_index:0,sequence:10,timestamp_ns:"20".into(),
            tracking_epoch:7,prompt_generation:3,sensor_origin:(400,800),width:256,height:192};
        let evidence=mask_boundary_logits::measure(&plane,source,0,0,false,
            &points(&row["candidates"][0]["baseline_retained"]),&[(0..128).collect()]).unwrap();
        row["outer_boundary_logits"]=json!(evidence);
        (file,row)
    }

    #[test]
    fn semantic_pupil_contours_preserve_gaps_and_reject_stale_provenance() {
        use sha2::{Digest,Sha256};
        let raw=vec![100u16;256*192];
        let (_file,mut row)=rejected_outline_fixture(&raw);
        let e=geometry::Ellipse {center:(130.0,98.0),major_radius:20.0,minor_radius:15.0,angle:0.0};
        row["pupil_void"]=json!({"ellipse":ellipse_json(Some(e))});
        let mut digest=Sha256::new();for pixel in &raw {digest.update(pixel.to_le_bytes());}
        let source=json!({"eye_index":0,"sequence":10,"timestamp_ns":"20",
            "sensor_origin":[400,800],"dimensions":[256,192],
            "native_u16le_sha256":format!("{:x}",digest.finalize())});
        // Two disjoint observed runs; all missing middle samples stay missing.
        let retained=vec![(111.0,94.0),(110.0,98.0),(111.0,102.0),
            (149.0,94.0),(150.0,98.0),(149.0,102.0)];
        let evidence=json!({"input":row["input"],
            "selection":{"input":row["input"],"source":source,"semantic_fit_present":true,
                "independent_observation":true,"semantic_query":2,"selected_ellipse":ellipse_json(Some(e))},
            "candidate":{"schema":"buttercup-semantic-pupil-contour-diagnostic-v1",
                "input":row["input"],"source":source,"prompt":2,"query":2,"ellipse":ellipse_json(Some(e)),
                "geometry_admitted":true,"center_admitted":true,"history_admitted":true,"raw_admitted":true,
                "clipped_component_fit":{"retained_points_native":retained,
                    "flat_tire_points_native":[[130.0,83.0]],"retained_runs":[[0,1,2],[3,4,5]]}}});
        let policy=ExtractionPolicy {all_boundary_samples:true,semantic_pupil_contours:true,..Default::default()};
        let baseline=prepare_with_policy(row.clone(),ExtractionPolicy {all_boundary_samples:true,..Default::default()}).unwrap();
        let recovery=prepare_with_policy(row.clone(),policy).unwrap();
        assert_eq!(format!("{:?}",baseline.packet),format!("{:?}",recovery.packet));
        row["selected_semantic_pupil_contour"]=evidence;
        let candidate=prepare_with_policy(row.clone(),policy).unwrap();
        let pupil=candidate.packet.arcs.iter().filter(|a|a.kind==BoundaryKind::PupillaryBoundary).collect::<Vec<_>>();
        assert_eq!(pupil.len(),2);
        assert_eq!(pupil.iter().flat_map(|a|a.points_roi_px.iter().copied()).collect::<Vec<_>>(),retained);
        assert!(pupil.iter().all(|a|a.outward_normals_roi.is_none()));
        let outer=|p:&OwnedRoiEvidence|format!("{:?}",p.arcs.iter().filter(|a|a.kind==BoundaryKind::OuterLimbus).collect::<Vec<_>>());
        assert_eq!(outer(&baseline.packet),outer(&candidate.packet));
        assert_eq!(candidate.packet.conics.last().unwrap().supporting_arc_indices,
            (baseline.packet.arcs.len()..baseline.packet.arcs.len()+2).collect::<Vec<_>>());
        for (path,value) in [
            ("/candidate/source/sequence",json!(11)),
            ("/candidate/source/native_u16le_sha256",json!("different-image")),
            ("/candidate/query",json!(1)),
            ("/candidate/history_admitted",json!(false)),
            ("/selection/independent_observation",json!(false)),
            ("/candidate/clipped_component_fit/retained_runs",json!([[0,1,2],[2,3,4]])),
            ("/candidate/clipped_component_fit/retained_points_native/0",json!([-1.0,94.0])),
        ] {
            let mut invalid=row.clone();
            *invalid["selected_semantic_pupil_contour"].pointer_mut(path).unwrap()=value;
            assert!(prepare_with_policy(invalid,policy).is_err(),"must reject {path}");
        }
        let mut moved=row.clone();moved["pupil_void"]["ellipse"]["center"]=json!([131.0,98.0]);
        assert!(prepare_with_policy(moved,policy).is_err());
        let mut missing=row;missing["pupil_void"]=Value::Null;
        assert!(prepare_with_policy(missing,policy).is_err());
    }

    #[test]
    fn mask_profiles_join_native_samples_before_training_validation_decimation() {
        let (_file,row)=mask_profile_fixture();
        let original=prepare_with_policy(row.clone(),ExtractionPolicy::default()).unwrap();
        let split=prepare_with_policy(row.clone(),ExtractionPolicy {mask_levels:true,..Default::default()}).unwrap();
        let full=prepare_with_policy(row,ExtractionPolicy {mask_levels:true,all_boundary_samples:true,..Default::default()}).unwrap();
        assert_eq!(full.mask_levels.as_ref().unwrap()["attachment"]["matched_profiles"],16);
        assert_eq!(full.mask_levels.as_ref().unwrap()["attachment"]["level_set_points"],16);
        assert_eq!(original.input,split.input);
        assert_eq!(format!("{:?}",original.pose),format!("{:?}",split.pose));
        assert_eq!(original.validation,split.validation);
        let mut stripped=split.packet.clone();
        for arc in &mut stripped.arcs {arc.level_sets_roi=None;}
        assert_eq!(format!("{stripped:?}"),format!("{:?}",original.packet));
        for (a,b) in split.packet.arcs.iter().zip(&full.packet.arcs) {
            assert_eq!(a.points_roi_px,b.points_roi_px.iter().step_by(2).copied().collect::<Vec<_>>());
            assert_eq!(a.level_sets_roi.as_ref().unwrap(),
                &b.level_sets_roi.as_ref().unwrap().iter().step_by(2).copied().collect::<Vec<_>>());
        }
        assert!(full.validation.is_empty());
    }

    #[test]
    fn mask_level_experiment_refuses_mismatched_payloads_and_reports_missing_receipts() {
        let (_file,mut row)=mask_profile_fixture();
        let original=prepare_with_policy(row.clone(),ExtractionPolicy::default()).unwrap();
        row["outer_boundary_logits"]["source"]["sequence"]=json!(11);
        assert!(prepare_with_policy(row.clone(),ExtractionPolicy {mask_levels:true,..Default::default()}).is_err());
        row["outer_boundary_logits"]=json!("unparseable unused experimental receipt");
        let off=prepare_with_policy(row.clone(),ExtractionPolicy::default()).unwrap();
        assert_eq!(format!("{:?}",original.packet),format!("{:?}",off.packet));
        assert!(off.mask_levels.is_none());
        row.as_object_mut().unwrap().remove("outer_boundary_logits");
        let missing=prepare_with_policy(row,ExtractionPolicy {mask_levels:true,..Default::default()}).unwrap();
        assert_eq!(missing.mask_levels.unwrap()["source_evidence"],"absent");
        assert_eq!(format!("{:?}",original.packet),format!("{:?}",missing.packet));
    }

    #[test]
    fn a_rejected_complete_fit_cannot_bypass_partial_raw_boundary_checks() {
        let (_fixture,mut row)=rejected_outline_fixture(&vec![0;256*192]);
        row["candidates"][0]["baseline_ellipse"]["center"]=json!([10.0,20.0]);
        let ordinary=prepare(row.clone(),false).unwrap();
        assert!(ordinary.baseline.is_some()&&!ordinary.packet.arcs.is_empty(),"preserve the explicitly selected legacy control");
        assert_eq!(ordinary.pose.limbus_center_sensor_px,[410.0,820.0]);
        let partial=prepare(row,true).unwrap();
        assert_eq!(partial.partial_outline.candidates,1,"a rejected full fit is not an exemption from partial evidence extraction");
        assert!(partial.packet.arcs.is_empty()&&partial.packet.conics.is_empty(),"flat RAW supplies no observed boundary");
        assert_eq!(partial.pose.limbus_center_sensor_px,[528.0,896.0],"rejected center cannot remain hidden scene support");
    }

    #[test]
    fn accepted_control_extraction_is_unchanged_by_partial_mode() {
        let (_fixture,mut row)=rejected_outline_fixture(&vec![0;256*192]);
        row["candidates"][0]["baseline_raw_admitted"]=json!(true);
        let ordinary=prepare(row.clone(),false).unwrap();
        let partial=prepare(row,true).unwrap();
        assert_eq!(partial.partial_outline.candidates,0);
        assert_eq!(ordinary.packet.arcs.len(),partial.packet.arcs.len());
        for (a,b) in ordinary.packet.arcs.iter().zip(&partial.packet.arcs) {
            assert_eq!(a.points_roi_px,b.points_roi_px);assert_eq!(a.evidence_group,b.evidence_group);
        }
        assert_eq!(ordinary.pose.limbus_center_sensor_px,partial.pose.limbus_center_sensor_px);
    }

    #[test]
    fn pupil_ablation_keeps_native_outer_probes_pose_and_scale_identical() {
        let raw=(0..192).flat_map(|y|(0..256).map(move|x| {
            if ((x as f64-130.0)/20.0).hypot((y as f64-98.0)/15.0)<=1.0 {40}
            else if ((x as f64-128.0)/60.0).hypot((y as f64-96.0)/46.0)<=1.0 {250} else {600}
        })).collect::<Vec<_>>();
        let (_fixture,mut row)=rejected_outline_fixture(&raw);
        row["candidates"][0]["baseline_raw_admitted"]=json!(true);
        row["pupil_void"]=json!({"ellipse":{"center":[130.0,98.0],"major_radius":20.0,"minor_radius":15.0,"angle":0.0}});
        row["input"]["scale_hint"]=json!({"pixels_per_10mm":100.0,"bounds_px_per_10mm":[80.0,120.0]});
        let full=prepare_with_policy(row.clone(),ExtractionPolicy {outline_directions:true,..Default::default()}).unwrap();
        let ablated=prepare_with_policy(row.clone(),ExtractionPolicy {outline_directions:true,without_pupil:true,..Default::default()}).unwrap();
        assert!(full.packet.arcs.iter().any(|a|a.kind==BoundaryKind::PupillaryBoundary));
        assert!(full.packet.conics.iter().any(|c|c.kind==BoundaryKind::PupillaryBoundary));
        assert_eq!(full.input,ablated.input);
        assert_eq!(full.pose.limbus_center_sensor_px,ablated.pose.limbus_center_sensor_px);
        assert_eq!(full.pose.pixels_per_10mm,ablated.pose.pixels_per_10mm);
        assert_eq!(full.packet.detail_reliability,ablated.packet.detail_reliability);
        assert_eq!(full.packet.exposure,ablated.packet.exposure);
        assert_eq!(full.packet.sensor_origin_px,ablated.packet.sensor_origin_px);
        assert_eq!(full.packet.dimensions_px,ablated.packet.dimensions_px);
        let outer=full.packet.arcs.iter().filter(|a|a.kind!=BoundaryKind::PupillaryBoundary).collect::<Vec<_>>();
        assert_eq!(format!("{outer:?}"),format!("{:?}",ablated.packet.arcs));
        let probes=full.validation.iter().filter(|a|a.2!=BoundaryKind::PupillaryBoundary).collect::<Vec<_>>();
        assert_eq!(format!("{probes:?}"),format!("{:?}",ablated.validation));
        assert!(ablated.packet.conics.iter().all(|c|c.kind!=BoundaryKind::PupillaryBoundary));
        assert_eq!(ablated.pupil_ablation["removed_pupil_hints"],1);
        let live=prepare_with_policy(row,ExtractionPolicy {outline_directions:true,all_boundary_samples:true,..Default::default()}).unwrap();
        assert!(live.validation.is_empty(),"fitted samples must never be advertised as withheld probes");
        for (index,arc) in live.packet.arcs.iter().enumerate() {
            let held=full.validation.iter().find(|(i,_,_,_)|*i==index).map(|v|v.3.len()).unwrap_or(0);
            assert_eq!(arc.points_roi_px.len(),full.packet.arcs[index].points_roi_px.len()+held);
        }
    }

    #[test]
    fn removing_interleaved_pupil_arcs_preserves_other_conic_support_identity() {
        let (_fixture,row)=rejected_outline_fixture(&vec![0;256*192]);
        let mut packet=prepare(row,false).unwrap().packet;
        let original=packet.clone();
        let mut pupil=packet.arcs[0].clone();
        pupil.kind=BoundaryKind::PupillaryBoundary;
        packet.arcs.insert(0,pupil);
        for c in &mut packet.conics {for index in &mut c.supporting_arc_indices {*index+=1;}}
        packet.conics.insert(0,OwnedConicHint {kind:BoundaryKind::PupillaryBoundary,
            ellipse_roi_px:packet.conics[0].ellipse_roi_px,supporting_arc_indices:vec![0]});
        remove_pupil_evidence(&mut packet);
        assert_eq!(format!("{packet:?}"),format!("{original:?}"));
    }

    #[test]
    fn rejected_outer_withholding_preserves_native_pupil_and_original_scene_inputs() {
        let raw=(0..192).flat_map(|y|(0..256).map(move|x| {
            if ((x as f64-130.0)/20.0).hypot((y as f64-98.0)/15.0)<=1.0 {40}
            else if ((x as f64-128.0)/60.0).hypot((y as f64-96.0)/46.0)<=1.0 {250} else {600}
        })).collect::<Vec<_>>();
        let (_fixture,mut row)=rejected_outline_fixture(&raw);
        row["pupil_void"]=json!({"ellipse":{"center":[130.0,98.0],"major_radius":20.0,"minor_radius":15.0,"angle":0.0}});
        row["input"]["scale_hint"]=json!({"pixels_per_10mm":100.0,"bounds_px_per_10mm":[80.0,120.0]});
        for admitted in [false,true] {
            row["candidates"][0]["baseline_raw_admitted"]=json!(admitted);
            let policy=ExtractionPolicy {all_boundary_samples:true,mask_levels:true,..Default::default()};
            let full=prepare_with_policy(row.clone(),policy).unwrap();
            let candidate=prepare_with_policy(row.clone(),ExtractionPolicy {withhold_rejected_outer:true,..policy}).unwrap();
            assert!(full.packet.arcs.iter().any(|a|a.kind==BoundaryKind::OuterLimbus));
            assert!(full.packet.arcs.iter().any(|a|a.kind==BoundaryKind::PupillaryBoundary));
            assert_eq!(full.input,candidate.input);
            assert_eq!(full.pose.limbus_center_sensor_px,candidate.pose.limbus_center_sensor_px);
            assert_eq!(full.pose.pixels_per_10mm,candidate.pose.pixels_per_10mm);
            assert_eq!(full.packet.exposure,candidate.packet.exposure);
            assert_eq!(full.packet.sensor_origin_px,candidate.packet.sensor_origin_px);
            assert_eq!(full.packet.dimensions_px,candidate.packet.dimensions_px);
            assert_eq!(full.packet.detail_reliability,candidate.packet.detail_reliability);
            assert!(candidate.validation.is_empty());
            if admitted {
                assert_eq!(format!("{:?}",full.packet),format!("{:?}",candidate.packet));
                assert_eq!(full.mask_levels,candidate.mask_levels);
                assert!(candidate.rejected_outer_ablation.is_none());
            } else {
                let preserved=full.packet.arcs.iter().filter(|a|a.kind!=BoundaryKind::OuterLimbus).collect::<Vec<_>>();
                assert_eq!(format!("{preserved:?}"),format!("{:?}",candidate.packet.arcs));
                assert_eq!(candidate.mask_levels.as_ref().unwrap()["source_evidence"],"rejected-outer-withheld");
                assert!(!candidate.rejected_outer_ablation.as_ref().unwrap()["removed_outer_arc_groups"].as_array().unwrap().is_empty());
                for hint in &candidate.packet.conics {
                    assert_ne!(hint.kind,BoundaryKind::OuterLimbus);
                    assert!(!hint.supporting_arc_indices.is_empty());
                    assert!(hint.supporting_arc_indices.iter().all(|&i|candidate.packet.arcs[i].kind==hint.kind));
                }
            }
        }
    }

    #[test]
    fn rejected_outer_removal_preserves_interleaved_inner_and_pupil_hint_identity() {
        let (_fixture,row)=rejected_outline_fixture(&vec![0;256*192]);
        let mut packet=prepare(row,false).unwrap().packet;
        let original=packet.arcs[0].clone();
        let ellipse=packet.conics[0].ellipse_roi_px;
        packet.arcs=[BoundaryKind::OuterLimbus,BoundaryKind::InnerLimbus,
            BoundaryKind::OuterLimbus,BoundaryKind::PupillaryBoundary].into_iter()
            .enumerate().map(|(i,kind)|OwnedBoundaryArc {kind,evidence_group:i as u32,..original.clone()}).collect();
        packet.conics=packet.arcs.iter().enumerate().map(|(i,a)|OwnedConicHint {
            kind:a.kind,ellipse_roi_px:ellipse,supporting_arc_indices:vec![i]}).collect();
        let inner=packet.arcs[1].clone();let pupil=packet.arcs[3].clone();
        let report=remove_outer_evidence(&mut packet);
        assert_eq!(report["removed_outer_arc_groups"],json!([0,2]));
        assert_eq!(report["removed_outer_hints"],2);
        assert_eq!(format!("{:?}",packet.arcs),format!("{:?}",vec![inner,pupil]));
        assert_eq!(packet.conics[0].supporting_arc_indices,vec![0]);
        assert_eq!(packet.conics[1].supporting_arc_indices,vec![1]);
        assert_eq!(packet.conics[0].kind,BoundaryKind::InnerLimbus);
        assert_eq!(packet.conics[1].kind,BoundaryKind::PupillaryBoundary);
    }

    #[test]
    #[ignore = "requires frozen SAM/Student native caches and RAW archives under outputs; numerical diagnostic, not a gaze accuracy assertion"]
    fn recorded_posterior_proposal_diagnostic() {
        use conic_solver::joint::posterior::IntegrationConfig;
        let seeds=[0xd1b5_4a32_d192_ed03,0x94c5_09a1_814f_753d,0x419b_79df_a2c7_5301];
        for (cache,sequences) in [
            ("outputs/student-shadow.FkzvzE/sam-live.jsonl",vec![4396,4430]),
            ("outputs/student-shadow.FkzvzE/student-live.jsonl",vec![4397,4411,4418]),
            ("outputs/calibration-provider-fit.cVlx1Z/recent-offered-0-sam.jsonl",vec![329,611]),
        ] {
            let mut pending:HashMap<u64,[Option<Frame>;2]>=HashMap::new();
            for line in BufReader::new(File::open(cache).expect("native corpus cache required")).lines() {
                let row:Value=serde_json::from_str(&line.unwrap()).unwrap();
                let sequence=integer(&row["input"]["frame"],"sequence").unwrap();
                if !sequences.contains(&sequence) {continue;}
                let frame=prepare_with_policy(row,ExtractionPolicy {all_boundary_samples:true,..Default::default()}).unwrap();
                let eye=frame.packet.exposure.roi.0 as usize-1;
                pending.entry(sequence).or_insert_with(||[None,None])[eye]=Some(frame);
            }
            for sequence in sequences {
                let frames=pending.remove(&sequence).unwrap();assert!(frames.iter().all(Option::is_some));
                let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
                let prepared=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.prepare()));
                let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
                let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
                let scene=approximate_scene(camera,poses).unwrap();
                let request=JointConicRequest {eyes:evidence,scene:&scene.prior,maximum_hypotheses:16,maximum_refinements:12,
                    maximum_source_skew_ns:0,exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
                let baseline=solve_joint_conics(request).unwrap();
                for (budget,early_stop) in [(8192,true),(65536,false)] {
                  for (adaptive,global_proposal,conditional_nuisance) in [
                    (false,false,false),(true,false,false),(false,true,false),(true,true,false),(false,false,true),
                  ] {
                    for seed in seeds {
                        let started=Instant::now();
                        let result=solve_joint_conic_distribution_diagnostic(request,1,
                            IntegrationConfig {budget,seed,adaptive,global_proposal,conditional_nuisance,early_stop,trace_tail:true,..Default::default()}).unwrap().remove(0);
                        assert_eq!(result.target_camera_mm,baseline.target_camera_mm);
                        assert_eq!(result.robust_cost,baseline.robust_cost);
                        assert_eq!(result.ellipses_roi_px,baseline.ellipses_roi_px);
                        let posterior=result.posterior.as_ref().unwrap();
                        assert!(posterior.samples<=budget);
                        assert!(posterior.feasible_samples<=posterior.samples-posterior.pilot_samples);
                        eprintln!("posterior-proposal {}",json!({"cache":cache,"sequence":sequence,
                            "inputs":frames.each_ref().map(|f|f.as_ref().map(|f|&f.input)),
                            "budget":budget,"adaptive":adaptive,"global_proposal":global_proposal,"conditional_nuisance":conditional_nuisance,"early_stop":early_stop,"seed":seed.to_string(),
                            "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"posterior":posterior.json()}));
                    }
                  }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires frozen SAM/Student caches and native RAW archive under outputs; independent numerical reference, not measured gaze accuracy"]
    fn recorded_cross_eye_support_reference_diagnostic() {
        cross_eye_support_reference(false,false,false);
    }

    #[test]
    #[ignore = "million-draw references for the three observed early-stop disagreements; native RAW caches required"]
    fn recorded_cross_eye_support_tail_reference_diagnostic() {
        cross_eye_support_reference(true,true,false);
    }

    #[test]
    #[ignore = "million-draw joint references for the entire ten-read SAM/Student interval; native RAW caches required"]
    fn recorded_cross_eye_support_large_reference_diagnostic() {
        cross_eye_support_reference(true,false,false);
    }

    #[test]
    #[ignore = "native SAM/Student comparison of uncertainty in direction mass and stopping/admission decisions"]
    fn recorded_cross_eye_numerical_admission_diagnostic() {
        cross_eye_support_reference(false,false,true);
    }

    #[test]
    #[ignore = "full native source-order comparison; requires output directory and immutable SAM/Student/recent caches"]
    fn recorded_numerical_admission_source_replay() {
        use conic_solver::joint::posterior::{IntegrationConfig,TailProposal};
        let directory=PathBuf::from(std::env::var("BUTTERCUP_POSTERIOR_REPLAY_DIR").expect("set a new run directory under outputs"));
        assert!(directory.canonicalize().unwrap().starts_with(PathBuf::from("outputs").canonicalize().unwrap()));
        let selected=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_COHORT").ok();
        let proposal=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_PROPOSAL").ok();
        assert!(proposal.as_deref().is_none_or(|p|[
            "conditional","adaptive","adaptive-conditional","integrated-inner","integrated-inner-precision",
            "integrated-inner-paired","integrated-inner-paired-precision",
            "integrated-inner-global","integrated-inner-global-conditional","integrated-inner-global-adaptive",
            "integrated-inner-conditional","integrated-inner-adaptive",
            "integrated-inner-global-conditional-scene",
            "integrated-inner-global-conditional-scene-precision",
            "integrated-inner-global-conditional-scene-replicated-original",
            "integrated-inner-global-conditional-scene-replicated-original-precision",
            "integrated-inner-global-conditional-precision",
            "integrated-inner-global-conditional-replicated",
            "integrated-inner-global-conditional-replicated-precision",
            "integrated-inner-global-conditional-replicated-original",
            "integrated-inner-global-conditional-replicated-original-precision",
            "integrated-inner-global-conditional-tail-pilot-replicated-original-precision",
            "integrated-inner-global-conditional-tail-recenter-replicated-original-precision",
            "integrated-inner-global-conditional-tail-refit-replicated-original-precision",
            "integrated-inner-global-conditional-tail-conditional-refit-replicated-original-precision",
            "integrated-inner-global-conditional-tail-outlier-refit-replicated-original-precision",
            "integrated-inner-global-conditional-tail-boundary-refit-replicated-original-precision",
            "integrated-inner-global-conditional-tail-boundary-defensive-replicated-original-precision",
            "integrated-inner-global-conditional-tail-profile-affine-replicated-original-precision",
            "integrated-inner-global-conditional-tail-profile-quadratic-replicated-original-precision",
        ].contains(&p)));
        let seed=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_SEED").ok().map(|v|v.parse().unwrap()).unwrap_or(IntegrationConfig::default().seed);
        let budget=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_BUDGET").ok().map(|v|v.parse().unwrap()).unwrap_or(8192);
        let early_stop=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_FULL_BUDGET").ok().as_deref()!=Some("1");
        let mut ran=false;
        for (cohort,files) in [
            ("sam",vec!["outputs/student-shadow.FkzvzE/sam-live.jsonl".to_owned()]),
            ("student",vec!["outputs/student-shadow.FkzvzE/student-live.jsonl".to_owned()]),
            ("recent",(0..3).map(|i|format!("outputs/calibration-provider-fit.cVlx1Z/recent-offered-{i}-sam.jsonl")).collect()),
        ] {
            if selected.as_ref().is_some_and(|name|name!=cohort) {continue;}
            ran=true;
            let policies=if proposal.as_deref().is_some_and(|p|p.starts_with("integrated-inner")) {vec![proposal.as_deref().unwrap().ends_with("-precision")]}
                else if proposal.is_some() {vec![true]} else {vec![false,true]};
            for numerical_admission in policies {
                let arm=proposal.as_deref().unwrap_or(if numerical_admission {"precision"} else {"baseline"});
                let path=directory.join(format!("{cohort}-{arm}-source.jsonl"));
                let mut writer=BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path).unwrap());
                source_order::run_with_posterior_diagnostic(&files,
                    IntegrationConfig {numerical_admission,seed,budget,early_stop,
                        tail_proposal:match proposal.as_deref() {
                            Some(p) if p.contains("tail-pilot")=>TailProposal::PilotOnly,
                            Some(p) if p.contains("tail-recenter")=>TailProposal::Recenter,
                            Some(p) if p.contains("tail-conditional-refit")=>TailProposal::ConditionalRefit,
                            Some(p) if p.contains("tail-outlier-refit")=>TailProposal::OutlierRefit,
                            Some(p) if p.contains("tail-boundary-refit")=>TailProposal::BoundaryRefit,
                            Some(p) if p.contains("tail-boundary-defensive")=>TailProposal::BoundaryDefensive,
                            Some(p) if p.contains("tail-profile-affine")=>TailProposal::ProfileAffine,
                            Some(p) if p.contains("tail-profile-quadratic")=>TailProposal::ProfileQuadratic,
                            Some(p) if p.contains("tail-refit")=>TailProposal::Refit,
                            _=>TailProposal::Off,
                        },
                        conditional_nuisance:proposal.as_deref().is_some_and(|p|p.contains("conditional")),
                        adaptive:proposal.as_deref().is_some_and(|p|p.contains("adaptive")),
                        global_proposal:proposal.as_deref().is_some_and(|p|p.contains("global")),
                        conditional_global:proposal.as_deref().is_some_and(|p|p.contains("-scene")),
                        marginalize_unobserved_inner:proposal.as_deref().is_some_and(|p|p.starts_with("integrated-inner")),
                        preserve_marginal_draws:proposal.as_deref().is_some_and(|p|p.contains("-paired")),
                        replicas:if proposal.as_deref().is_some_and(|p|p.contains("replicated")) {4} else {1},
                        preserve_replica_draws:proposal.as_deref().is_some_and(|p|p.contains("replicated-original")),
                        trace_tail:true,
                        ..Default::default()},&mut writer).unwrap();
                writer.flush().unwrap();
            }
        }
        assert!(ran,"unknown corpus name");
    }

    #[test]
    #[ignore = "selected large posterior references preserve all native source-order context and MAP seeds"]
    fn recorded_selected_posterior_reference_replay() {
        use conic_solver::joint::posterior::IntegrationConfig;
        let directory=PathBuf::from(std::env::var("BUTTERCUP_POSTERIOR_REPLAY_DIR").expect("set reference output directory"));
        assert!(directory.canonicalize().unwrap().starts_with(PathBuf::from("outputs").canonicalize().unwrap()));
        let cohort=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_COHORT").expect("sam, student or recent");
        let budget=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_BUDGET").ok().map(|v|v.parse().unwrap()).unwrap_or(1048576);
        let proposal=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_PROPOSAL").ok();
        assert!(proposal.as_deref().is_none_or(|p|[
            "integrated-inner","integrated-inner-paired","integrated-inner-global-conditional-scene",
        ].contains(&p)));
        let seeds=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_SEED").ok().map(|v|vec![v.parse().unwrap()])
            .unwrap_or(vec![0xd1b5_4a32_d192_ed03,0x94c5_09a1_814f_753d,0x419b_79df_a2c7_5301]);
        let selected:Value=serde_json::from_reader(File::open(directory.join("reference-selection-v1.json")).unwrap()).unwrap();
        let indices=selected[&cohort]["reads"].as_array().unwrap().iter()
            .map(|row|row["final_event_input_index"].as_u64().unwrap()).collect::<std::collections::HashSet<_>>();
        assert!(!indices.is_empty());
        let files=match cohort.as_str() {
            "sam"=>vec!["outputs/student-shadow.FkzvzE/sam-live.jsonl".to_owned()],
            "student"=>vec!["outputs/student-shadow.FkzvzE/student-live.jsonl".to_owned()],
            "recent"=>(0..3).map(|i|format!("outputs/calibration-provider-fit.cVlx1Z/recent-offered-{i}-sam.jsonl")).collect(),
            _=>panic!("unknown native cohort"),
        };
        for seed in seeds {
            let path=directory.join(format!("{cohort}-reference-{seed}.jsonl"));
            let mut writer=BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path).unwrap());
            source_order::run_with_selected_posterior_diagnostic(&files,
                IntegrationConfig {budget,seed,early_stop:false,marginalize_unobserved_inner:proposal.is_some(),
                    global_proposal:proposal.as_deref().is_some_and(|p|p.ends_with("-scene")),
                    conditional_global:proposal.as_deref().is_some_and(|p|p.ends_with("-scene")),
                    conditional_nuisance:proposal.as_deref().is_some_and(|p|p.ends_with("-scene")),
                    preserve_marginal_draws:proposal.as_deref().is_some_and(|p|p.contains("-paired")),..Default::default()},&indices,&mut writer).unwrap();
            writer.flush().unwrap();
        }
    }

    #[test]
    #[ignore = "exact native factor attribution across frozen SAM3.1/Student source histories"]
    fn recorded_native_factor_diagnostic() {
        let directory=PathBuf::from(std::env::var("BUTTERCUP_FACTOR_DIAGNOSTIC_DIR").expect("set a new output directory"));
        assert!(directory.canonicalize().unwrap().starts_with(PathBuf::from("outputs").canonicalize().unwrap()));
        let cohorts = [
            ("sam", vec!["outputs/student-shadow.FkzvzE/sam-live.jsonl".to_owned()]),
            ("student", vec!["outputs/student-shadow.FkzvzE/student-live.jsonl".to_owned()]),
            ("recent", (0..3).map(|i|format!("outputs/calibration-provider-fit.cVlx1Z/recent-offered-{i}-sam.jsonl")).collect()),
        ];
        for (name, files) in cohorts {
            let mut writer=BufWriter::new(OpenOptions::new().write(true).create_new(true)
                .open(directory.join(format!("{name}-native-factors.jsonl"))).unwrap());
            source_order::run(&files,usize::MAX,ExtractionPolicy {all_boundary_samples:true,..Default::default()},
                [0;2],true,false,&mut writer).unwrap();
            writer.flush().unwrap();
        }
    }

    #[test]
    #[ignore = "matched native source-order experiment for supported retained-mode selection; explicit frozen inputs required"]
    fn recorded_supported_mode_replay() {
        use conic_solver::joint::posterior::IntegrationConfig;
        let directory=PathBuf::from(std::env::var("BUTTERCUP_SUPPORTED_MODE_DIR").expect("set output directory"));
        assert!(directory.canonicalize().unwrap().starts_with(PathBuf::from("outputs").canonicalize().unwrap()));
        let inputs:Vec<String>=serde_json::from_str(&std::env::var("BUTTERCUP_SUPPORTED_MODE_INPUTS")
            .expect("set frozen native inputs as a JSON array")).unwrap();
        assert!(!inputs.is_empty());
        let select_supported_mode=std::env::var("BUTTERCUP_SELECT_SUPPORTED_MODE").ok().as_deref()==Some("1");
        let exact_conic_distances=std::env::var("BUTTERCUP_EXACT_CONIC_DISTANCES").ok().as_deref()==Some("1");
        std::fs::write(directory.join("metric-contract.json"),serde_json::to_vec_pretty(&json!({
            "exact_conic_distances":exact_conic_distances,
            "contract":"Same projected-conic position metric throughout selection, fitting, posterior, uncertainty and reported residuals. Observations and all priors remain unchanged."})).unwrap()).unwrap();
        let subpixel_pupil_peaks=std::env::var("BUTTERCUP_SUBPIXEL_PUPIL_PEAKS").ok().as_deref()==Some("1");
        let all_boundary_samples=std::env::var("BUTTERCUP_WITHHOLD_BOUNDARY_SAMPLES").ok().as_deref()!=Some("1");
        let mut writer=BufWriter::new(OpenOptions::new().write(true).create_new(true)
            .open(directory.join("source.jsonl")).unwrap());
        source_order::run_with_supported_mode_diagnostic(&inputs,
            ExtractionPolicy {all_boundary_samples,subpixel_pupil_peaks,..Default::default()},
            IntegrationConfig {select_supported_mode,exact_conic_distances,..IntegrationConfig::live()},&mut writer).unwrap();
        writer.flush().unwrap();
    }

    #[test]
    #[ignore = "explicit native RAW audit of whole-pupil path membership; frozen inputs required"]
    fn recorded_pupil_path_association_audit() {
        let directory=PathBuf::from(std::env::var("BUTTERCUP_PATH_AUDIT_DIR").expect("set audit output directory"));
        assert!(directory.canonicalize().unwrap().starts_with(PathBuf::from("outputs").canonicalize().unwrap()));
        let inputs:Vec<String>=serde_json::from_str(&std::env::var("BUTTERCUP_PATH_AUDIT_INPUTS")
            .expect("set frozen input paths")).unwrap();
        let mut writer=BufWriter::new(OpenOptions::new().write(true).create_new(true)
            .open(directory.join("paths.jsonl")).unwrap());
        let mut count=0;
        for path in inputs {
            for line in BufReader::new(File::open(path).unwrap()).lines() {
                let row:Value=serde_json::from_str(&line.unwrap()).unwrap();
                let baseline=prepare_with_policy(row.clone(),ExtractionPolicy {all_boundary_samples:true,..Default::default()}).unwrap();
                let conic=prepare_with_policy(row.clone(),ExtractionPolicy {all_boundary_samples:true,conic_pupil_paths:true,..Default::default()}).unwrap();
                let describe=|frame:&Frame| frame.packet.arcs.iter().enumerate().map(|(index,a)|json!({
                    "arc":index,"group":a.evidence_group,"kind":format!("{:?}",a.kind),
                    "points":a.points_roi_px,"normal_band_px":a.normal_band_half_width_px,
                    "sampling_support":a.sampling_support_px,"score":a.detector_score})).collect::<Vec<_>>();
                let mut output=json!({"input":baseline.input,"baseline_arcs":describe(&baseline),"conic_arcs":describe(&conic),"path_audit":null});
                if let Some((guide,outer))=ellipse(&row["pupil_void"]["ellipse"]).zip(baseline.baseline) {
                    let input=&baseline.input;let meta=&input["frame"];
                    let [width,height]=baseline.packet.dimensions_px.map(|v|v as usize);
                    let mut file=File::open(input["raw_file"].as_str().unwrap()).unwrap();
                    file.seek(SeekFrom::Start(integer(input,"raw_offset").unwrap())).unwrap();
                    let mut bytes=vec![0;integer(input,"raw_length").unwrap() as usize];file.read_exact(&mut bytes).unwrap();
                    let raw=raw10::try_unpack_raw10(&bytes,width,height,integer(meta,"stride").unwrap() as usize).unwrap();
                    if let Some(config)=RawArcConfig::for_pupil(&raw,width,height,outer) {
                        output["path_audit"]=raw_ring_path_audit(&raw,width,height,guide,config);
                        output["pupil_guide"]=ellipse_json(Some(guide));
                        output["outer_guide"]=ellipse_json(Some(outer));
                    }
                }
                serde_json::to_writer(&mut writer,&output).unwrap();writer.write_all(b"\n").unwrap();count+=1;
            }
        }
        writer.flush().unwrap();assert!(count>0);eprintln!("PATH_AUDIT source_eye_records={count}");
    }

    #[test]
    #[ignore = "same native source-order likelihood with optional coherent-mask proposal conditioning"]
    fn recorded_mask_state_proposal_replay() {
        use conic_solver::joint::posterior::IntegrationConfig;
        let directory=PathBuf::from(std::env::var("BUTTERCUP_MASK_PROPOSAL_DIR").expect("set output directory"));
        assert!(directory.canonicalize().unwrap().starts_with(PathBuf::from("outputs").canonicalize().unwrap()));
        let input=std::env::var("BUTTERCUP_MASK_PROPOSAL_INPUT").expect("set frozen profile cache");
        let mask_levels=std::env::var("BUTTERCUP_MASK_LEVELS").ok().as_deref()==Some("1");
        let mask_state_proposals=std::env::var("BUTTERCUP_MASK_PROPOSALS").ok().as_deref()==Some("1");
        let mask_spatial=std::env::var("BUTTERCUP_MASK_SPATIAL").ok().as_deref()==Some("1");
        let mask_state_refinement=std::env::var("BUTTERCUP_MASK_REFINEMENT").ok().as_deref()==Some("1");
        let marginalize_arc_alternatives=std::env::var("BUTTERCUP_ARC_MARGINALIZATION").ok().as_deref()==Some("1");
        let populations=std::env::var("BUTTERCUP_POPULATION_PARTICLES").ok().map(|v|
            conic_solver::joint::posterior::populations::Config {particles:v.parse().unwrap(),
                steps:std::env::var("BUTTERCUP_POPULATION_STEPS").expect("set population steps").parse().unwrap(),
                populations:std::env::var("BUTTERCUP_POPULATION_COUNT").expect("set population count").parse().unwrap()});
        let seed=std::env::var("BUTTERCUP_MASK_PROPOSAL_SEED").ok().map(|v|v.parse().unwrap())
            .unwrap_or(IntegrationConfig::live().seed);
        let reference=std::env::var("BUTTERCUP_MASK_REFERENCE_PATHS").ok().map(|v|
            conic_solver::joint::posterior::annealed::Config {paths:v.parse().unwrap(),
                steps:std::env::var("BUTTERCUP_MASK_REFERENCE_STEPS").expect("set annealing steps").parse().unwrap()});
        let mut writer=BufWriter::new(OpenOptions::new().write(true).create_new(true)
            .open(directory.join("source.jsonl")).unwrap());
        source_order::run_with_mask_posterior_diagnostic(&[input],mask_levels,mask_spatial,
            IntegrationConfig {mask_state_proposals,mask_state_refinement,marginalize_arc_alternatives,populations,seed,annealed_reference:reference,..IntegrationConfig::live()},&mut writer).unwrap();
        writer.flush().unwrap();
    }

    #[test]
    #[ignore = "independent annealed paths on selected native publications, retaining full source history and ordinary posterior"]
    fn recorded_selected_annealed_reference_replay() {
        use conic_solver::joint::posterior::{annealed,IntegrationConfig,TailProposal};
        let directory=PathBuf::from(std::env::var("BUTTERCUP_POSTERIOR_REPLAY_DIR").expect("set output directory"));
        assert!(directory.canonicalize().unwrap().starts_with(PathBuf::from("outputs").canonicalize().unwrap()));
        let cohort=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_COHORT").expect("sam, student or recent");
        let seed=std::env::var("BUTTERCUP_POSTERIOR_REPLAY_SEED").expect("set numerical seed").parse().unwrap();
        let paths=std::env::var("BUTTERCUP_ANNEALED_PATHS").expect("set path count").parse().unwrap();
        let steps=std::env::var("BUTTERCUP_ANNEALED_STEPS").expect("set bridge steps").parse().unwrap();
        let selection:Value=serde_json::from_reader(File::open(directory.join("reference-selection-v1.json")).unwrap()).unwrap();
        let indices=selection[&cohort]["reads"].as_array().unwrap().iter()
            .map(|r|r["final_event_input_index"].as_u64().unwrap()).collect::<std::collections::HashSet<_>>();
        assert!(!indices.is_empty());
        let files=match cohort.as_str() {
            "sam"=>vec!["outputs/student-shadow.FkzvzE/sam-live.jsonl".to_owned()],
            "student"=>vec!["outputs/student-shadow.FkzvzE/student-live.jsonl".to_owned()],
            "recent"=>(0..3).map(|i|format!("outputs/calibration-provider-fit.cVlx1Z/recent-offered-{i}-sam.jsonl")).collect(),
            _=>panic!("unknown cohort"),
        };
        let path=directory.join(format!("{cohort}-annealed-{steps}-{paths}-{seed}.jsonl"));
        let mut writer=BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path).unwrap());
        source_order::run_with_selected_annealed_diagnostic(&files,IntegrationConfig {
            seed,budget:8192,early_stop:true,tail_proposal:TailProposal::ConditionalRefit,
            numerical_admission:true,marginalize_unobserved_inner:true,
            global_proposal:true,conditional_nuisance:true,replicas:4,preserve_replica_draws:true,
            annealed_reference:Some(annealed::Config {paths,steps}),..Default::default()
        },&indices,&mut writer).unwrap();
        writer.flush().unwrap();
    }

    fn cross_eye_support_reference(large_reference:bool,tail_only:bool,numerical_admission:bool) {
        use conic_solver::joint::posterior::IntegrationConfig;
        let seeds=[0xd1b5_4a32_d192_ed03,0x94c5_09a1_814f_753d,0x419b_79df_a2c7_5301];
        let mut same_source=HashMap::new();
        for cache in ["outputs/student-shadow.FkzvzE/sam-live.jsonl","outputs/student-shadow.FkzvzE/student-live.jsonl"] {
            let mut pending:HashMap<u64,[Option<Frame>;2]>=HashMap::new();
            for line in BufReader::new(File::open(cache).expect("native corpus cache required")).lines() {
                let row:Value=serde_json::from_str(&line.unwrap()).unwrap();
                let sequence=integer(&row["input"]["frame"],"sequence").unwrap();
                if !(4487..=4496).contains(&sequence) {continue;}
                let frame=prepare_with_policy(row,ExtractionPolicy {all_boundary_samples:true,..Default::default()}).unwrap();
                assert!(frame.validation.is_empty());
                assert_eq!(frame.input["clock_attested"],true);
                assert_eq!(frame.input["clock_lineage"],"stream:3721203-1789217645285126937:19");
                let eye=frame.packet.exposure.roi.0 as usize-1;
                if let Some(original)=same_source.insert((sequence,eye),frame.input.clone()) {
                    assert_eq!(original,frame.input,"both providers must describe the exact same native source");
                }
                assert!(pending.entry(sequence).or_insert_with(||[None,None])[eye].replace(frame).is_none());
            }
            assert_eq!(pending.len(),10);
            let mut previous_time=None;
            for sequence in 4487..=4496 {
                if tail_only && !((cache.ends_with("/sam-live.jsonl") && sequence==4492)
                    || (cache.ends_with("/student-live.jsonl") && [4488,4490].contains(&sequence))) {continue;}
                let frames=pending.remove(&sequence).unwrap();assert!(frames.iter().all(Option::is_some));
                let sources=frames.each_ref().map(|f|f.as_ref().unwrap().packet.exposure);
                assert_eq!(sources[0].clock,sources[1].clock);
                assert_eq!(sources[0].sequence,sources[1].sequence);
                assert_eq!(sources[0].timestamp_ns,sources[1].timestamp_ns);
                if let Some(previous)=previous_time {assert!((1..=500_000_000).contains(&(sources[0].timestamp_ns-previous)));}
                previous_time=Some(sources[0].timestamp_ns);
                let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
                let prepared=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.prepare()));
                let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
                let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
                // Freeze the same coarse scene even when one eye's conic evidence is withheld.
                let scene=approximate_scene(camera,poses).unwrap();
                for (arm,mask) in [("joint",[true,true]),("monocular_right",[true,false]),("monocular_left",[false,true])] {
                    if large_reference && arm!="joint" {continue;}
                    let request=JointConicRequest {eyes:std::array::from_fn(|eye|if mask[eye] {evidence[eye]} else {None}),
                        scene:&scene.prior,maximum_hypotheses:16,maximum_refinements:12,
                        maximum_source_skew_ns:0,exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
                    let baseline=solve_joint_conics(request).unwrap();
                    assert_eq!(baseline.modeled_eyes,mask);
                    assert_eq!(baseline.contributing_eyes,mask);
                    let budgets=if large_reference {vec![(1048576,false)]} else {vec![(8192,true),(65536,false)]};
                    for (budget,early_stop) in budgets {
                        for seed in seeds {
                            let result=solve_joint_conic_distribution_diagnostic(request,1,
                                IntegrationConfig {budget,seed,early_stop,numerical_admission,..Default::default()}).unwrap().remove(0);
                            assert_eq!(result.target_camera_mm,baseline.target_camera_mm);
                            assert_eq!(result.robust_cost,baseline.robust_cost);
                            assert_eq!(result.ellipses_roi_px,baseline.ellipses_roi_px);
                            let posterior=result.posterior.as_ref().unwrap();
                            for eye in 0..2 {if !mask[eye] {assert!(!posterior.supports_direction(eye));}}
                            eprintln!("cross-eye-reference {}",json!({"cache":cache,"sequence":sequence,"arm":arm,
                                "inputs":frames.each_ref().map(|f|f.as_ref().map(|f|&f.input)),
                                "scale_provenance":scene.scale_provenance.map(|p|p.map(|p|format!("{p:?}"))),
                                "independent_pixels_per_mm":scene.independent_pixels_per_mm,
                                "arc_fingerprints":frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.arcs.iter()
                                    .map(|a|json!({"kind":format!("{:?}",a.kind),"group":a.evidence_group,
                                        "samples":a.points_roi_px.len(),"fingerprint":sample_fingerprint(&a.points_roi_px)})).collect::<Vec<_>>())),
                                "budget":budget,"early_stop":early_stop,"seed":seed.to_string(),
                                "solution":solution_json(Ok(result.clone()),frames.each_ref().map(Option::as_ref),0.0)}));
                            if budget==65536 && arm!="joint" {
                                assert_eq!(posterior.status,"estimated-conditional");
                                assert!(!posterior.supports_direction(0));
                                assert!(!posterior.supports_direction(1));
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(same_source.len(),20);
    }

    #[test]
    #[ignore = "requires the immutable recent-offered-0 SAM cache and its native RAW archive under outputs"]
    fn recorded_pupil_conflict_search_diagnostic() {
        let cache="outputs/calibration-provider-fit.cVlx1Z/recent-offered-0-sam.jsonl";
        let mut pending:HashMap<u64,[Option<Frame>;2]>=HashMap::new();
        for line in BufReader::new(File::open(cache).expect("native corpus cache required")).lines() {
            let row:Value=serde_json::from_str(&line.unwrap()).unwrap();
            let sequence=integer(&row["input"]["frame"],"sequence").unwrap();
            if ![325,329,611].contains(&sequence) {continue;}
            let frame=prepare_with_policy(row,ExtractionPolicy {all_boundary_samples:true,..Default::default()}).unwrap();
            let eye=frame.packet.exposure.roi.0 as usize-1;
            pending.entry(sequence).or_insert_with(||[None,None])[eye]=Some(frame);
        }
        assert_eq!(pending.len(),3);
        for sequence in [325,329,611] {
            let frames=pending.remove(&sequence).unwrap();assert!(frames.iter().all(Option::is_some));
            let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
            let prepared=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.prepare()));
            let evidence=prepared.each_ref().map(|p|p.as_ref().map(|p|p.evidence()));
            let camera=PinholeCamera {focal_px:[4000.0;2],principal_px:[4000.0,3000.0]};
            let scene=approximate_scene(camera,poses).unwrap();
            for (hypotheses,refinements) in [(16,12),(24,16)] {
                let request=JointConicRequest {eyes:evidence,scene:&scene.prior,
                    maximum_hypotheses:hypotheses,maximum_refinements:refinements,
                    maximum_source_skew_ns:0,exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0};
                let result=solve_joint_conics(request);
                let row=solution_json(result,frames.each_ref().map(Option::as_ref),0.0);
                eprintln!("pupil-budget {}",json!({"sequence":sequence,"hypotheses":hypotheses,"refinements":refinements,
                    "available":row["available"],"cost":row["cost"],"areas":row["sn_feida_mm2"],
                    "ellipses":row["outer_ellipses"],"target":row["target_camera_mm"],"support":row["support"]}));
            }
        }
    }

    #[test]
    #[ignore = "requires the immutable recent-offered-0 SAM cache and its native RAW archive under outputs"]
    fn recorded_dark_clip_pupil_support_suppresses_one_area_spike() {
        // A real-source diagnostic regression, NOT proof of correct anatomy.
        // This interval has no human labels or fresh per-frame scale support;
        // a later interval in the SAME clip regresses with pupil support.
        let cache=std::env::var("BUTTERCUP_PUPIL_ABLATION_CACHE").unwrap_or_else(|_|
            "outputs/calibration-provider-fit.cVlx1Z/recent-offered-0-sam.jsonl".to_owned());
        let path=PathBuf::from("outputs").join(format!("pupil-corpus-case-{}-{}.jsonl",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let mut file=OpenOptions::new().write(true).create_new(true).open(&path).unwrap();
        let fixture=RawFixture(path);
        let mut sources=0;
        for line in BufReader::new(File::open(cache).expect("native corpus cache is required")).lines() {
            let line=line.unwrap();let row:Value=serde_json::from_str(&line).unwrap();let input=&row["input"];
            let sequence=integer(&input["frame"],"sequence").unwrap();
            if input["capture_entry"]!=0 || !(313..=333).contains(&sequence) {continue;}
            assert_eq!(input["clock_lineage"],"stream:1148508-1788826332876023270:35");
            assert_eq!(input["clock_attested"],true);
            file.write_all(line.as_bytes()).unwrap();file.write_all(b"\n").unwrap();sources+=1;
        }
        assert_eq!(sources,18,"nine exact source exposures from each eye, including source 329");
        drop(file);
        let mut series=Vec::new();
        for without_pupil in [false,true] {
            let mut bytes=Vec::new();
            // Full native boundary points; same real tracker and optimizer.
            // Posterior sampling does not change MAP geometry or history seeds.
            source_order::run(&[fixture.0.to_string_lossy().into_owned()],usize::MAX,
                ExtractionPolicy {without_pupil,all_boundary_samples:true,..Default::default()},[0;2],false,false,&mut bytes).unwrap();
            let mut values=Vec::new();
            let mut centers=Vec::new();
            for line in std::str::from_utf8(&bytes).unwrap().lines() {
                let row:Value=serde_json::from_str(line).unwrap();
                if !row["publication_inputs"].as_array().is_some_and(|a|a.iter().all(|i|!i.is_null())) {continue;}
                assert_eq!(row["duplicate_suppressed"],true);
                assert_eq!(row["joint"]["withheld_sample_residuals"][0]["points"],0);
                let time=integer(&row["input"]["frame"],"timestamp_ns").unwrap();
                let area=row["joint"]["sn_feida_mm2"][0].as_f64().expect("fresh accepted outer boundary and independent scale");
                let ellipse=&row["joint"]["outer_ellipses"][0];
                let input=&row["publication_inputs"][0]["frame"];
                centers.push([ellipse["center"][0].as_f64().unwrap()+integer(input,"sensor_x").unwrap() as f64,
                    ellipse["center"][1].as_f64().unwrap()+integer(input,"sensor_y").unwrap() as f64]);
                if !without_pupil {
                    let used=row["joint"]["support"].as_array().unwrap().iter()
                        .filter(|g|g["roi"]==1&&g["kind"]=="OuterLimbus"&&g["used"]==true).collect::<Vec<_>>();
                    assert!(!used.is_empty()&&used.iter().all(|g|g["rms_px"].as_f64().unwrap()<5.0),
                        "area alone cannot overrule current accepted contour support; this still does not certify the rejected rim");
                }
                values.push((time,area));
            }
            assert_eq!(values.len(),9,"neither arm may hide a dropout or count a repeated publication");
            for pair in values.windows(2) {assert!(pair[1].0>pair[0].0&&pair[1].0-pair[0].0<=500_000_000);}
            assert!(centers.windows(2).all(|p|(p[1][0]-p[0][0]).hypot(p[1][1]-p[0][1])>0.1),
                "this moving-source fixture must not pass by freezing the ellipse in sensor coordinates");
            series.push(values);
        }
        assert!(series[0].iter().zip(&series[1]).all(|(a,b)|a.0==b.0));
        let changes=series.iter().map(|values|values.windows(2)
            .map(|p|(p[1].1/p[0].1).ln().abs()).sum::<f64>()/(values.len()-1) as f64).collect::<Vec<_>>();
        println!("real clip 0 right eye 313..333, nine fresh frames per arm; mean absolute log SN-FEIDA step with/without pupil={changes:?}");
        assert!(changes[0]<changes[1]*0.5,"preserve this bounded causal area diagnostic, not a claim of localization accuracy");
    }

    #[test]
    fn partial_boundary_normals_follow_their_points_through_training_decimation() {
        let raw=(0..192).flat_map(|y|(0..256).map(move|x|
            if ((x as f64-128.0)/60.0).hypot((y as f64-96.0)/46.0)<=1.0 {150} else {550})).collect::<Vec<_>>();
        let (_fixture,row)=rejected_outline_fixture(&raw);
        let prepared=prepare(row,true).unwrap();
        assert!(prepared.packet.arcs.len()>=4);
        assert!(!prepared.validation.is_empty(),"exercise actual held-out/training subsampling");
        for arc in &prepared.packet.arcs {
            let normals=arc.outward_normals_roi.as_ref().unwrap();
            assert_eq!(normals.len(),arc.points_roi_px.len());
            for (&(x,y),normal) in arc.points_roi_px.iter().zip(normals) {
                let n=normal.unwrap();
                let gradient=[(x-128.0)/60.0f64.powi(2),(y-96.0)/46.0f64.powi(2)];
                assert!((gradient[0]*n.unit_outward_roi[0]+gradient[1]*n.unit_outward_roi[1])
                    /gradient[0].hypot(gradient[1])>0.99,"a normal must retain its original sample identity");
            }
        }
    }
}
