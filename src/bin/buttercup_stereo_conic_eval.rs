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
    pub(crate) mod eye_scene_model {pub(crate) mod binocular_pose;}
}
use native::{conic_solver,outline_conic_segments,roi_evidence};
use native::{binocular_coordinator,eye_scene_model};
#[path="../gaze_target_solver/joint_tracking.rs"] mod joint_tracking;
#[path="buttercup_stereo_conic_eval/source_order.rs"] mod source_order;
use native::eye_scene_model::binocular_pose::{approximate_scene,EyePoseInput};
use conic_solver::joint::*;
use outline_conic_segments::sparse_evidence::*;
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

struct Frame {
    input:Value,
    packet:OwnedRoiEvidence,
    pose:EyePoseInput,
    validation:Vec<(usize,u32,BoundaryKind,Vec<(f64,f64)>)>,
    baseline:Option<geometry::Ellipse>,
    selected_raw_admitted:bool,
    partial_outline:PartialOutlineReport,
}

fn prepare(row:Value,partial_outlines:bool)->Result<Frame,String> {
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
    if let Some((selected,baseline))=selected.zip(baseline) {
        let retained=points(&selected["baseline_retained"]);
        let segments=selected["baseline_retained_segments"].as_array().map(|segments|segments.iter().filter_map(|run| {
            Some(run.as_array()?.iter().filter_map(|i|i.as_u64().map(|i|i as usize)).collect::<Vec<_>>())
        }).collect::<Vec<_>>()).unwrap_or_default();
        // Missing contour provenance is unavailable, not a dense ellipse.
        let review=outline_conic_segments::ContourFitEvidence {ellipse:baseline,source_component_area_px:0.0,
            retained_points:Arc::new(retained),conic_segments:Arc::new(segments),
            flat_tire_points:Arc::new(points(&selected["baseline_censored"])),upper_flat_tire:false,lower_flat_tire:false};
        append_retained_sam_arcs(&mut packet,&review,0);
        if !selected_raw_admitted {
            for arc in &mut packet.arcs {arc.normal_band_half_width_px=5.0;}
        }
    }
    let pupil=ellipse(&row["pupil_void"]["ellipse"]);
    let try_partial=partial_outlines&&baseline.is_none();
    let mut partial_outline=PartialOutlineReport::default();
    if pupil.is_some()||try_partial {
        let mut file=File::open(input["raw_file"].as_str().ok_or("missing RAW file")?).map_err(|e|e.to_string())?;
        file.seek(SeekFrom::Start(integer(&input,"raw_offset")?)).map_err(|e|e.to_string())?;
        let mut bytes=vec![0;integer(&input,"raw_length")? as usize];file.read_exact(&mut bytes).map_err(|e|e.to_string())?;
        let raw=raw10::try_unpack_raw10(&bytes,size[0] as usize,size[1] as usize,integer(meta,"stride")? as usize)?;
        if let Some(pupil)=pupil {
            append_raw_ring_arcs(&mut packet,&raw,pupil,BoundaryKind::PupillaryBoundary,100,RawArcConfig::default());
        }
        if try_partial {
            let mut ranked=candidates.iter().filter(|c|c["semantic_score"].as_f64().is_some_and(f64::is_finite)).collect::<Vec<_>>();
            ranked.sort_by(|a,b|b["semantic_score"].as_f64().unwrap().total_cmp(&a["semantic_score"].as_f64().unwrap()));
            let outlines=ranked.iter().take(4).map(|c|(points(&c["outline"]),c["semantic_score"].as_f64())).collect::<Vec<_>>();
            let candidates=outlines.iter().map(|(points,score)|OutlineCandidate {points_roi_px:points,detector_score:*score}).collect::<Vec<_>>();
            partial_outline=append_unfitted_outline_arcs(&mut packet,&raw,&candidates,(size[0] as f64*0.5,size[1] as f64*0.5),200);
        }
    }
    let center=baseline.map(|e|[e.center.0+origin[0] as f64,e.center.1+origin[1] as f64])
        .unwrap_or([origin[0] as f64+size[0] as f64*0.5,origin[1] as f64+size[1] as f64*0.5]);
    let scale=&input["scale_hint"];
    let scale=scale["pixels_per_10mm"].as_f64().zip(scale["bounds_px_per_10mm"].as_array()).and_then(|(n,b)|Some([n,b.first()?.as_f64()?,b.get(1)?.as_f64()?]));
    let pose=EyePoseInput {limbus_center_sensor_px:center,pixels_per_10mm:scale};
    let mut validation=Vec::new();
    for (index,arc) in packet.arcs.iter_mut().enumerate() {
        if arc.points_roi_px.len()>=6 {
            validation.push((index,arc.evidence_group,arc.kind,arc.points_roi_px.iter().skip(1).step_by(2).copied().collect()));
            arc.points_roi_px=arc.points_roi_px.iter().step_by(2).copied().collect();
        }
    }
    Ok(Frame {input,packet,pose,validation,baseline,selected_raw_admitted,partial_outline})
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
        Ok(solution)=>json!({"available":true,"target_camera_mm":solution.target_camera_mm,
            "eye_centers_camera_mm":solution.eye_centers_camera_mm,"eye_normals":solution.eye_normals,
            "eye_gaze_directions":solution.eye_gaze_directions,"surface_axis_alignment_radians":solution.surface_axis_alignment_radians,
            "contributing_eyes":solution.contributing_eyes,"cost":solution.robust_cost,
            "modeled_eyes":solution.modeled_eyes,"unlocalized_eye_cost":solution.unlocalized_eye_cost,
            "alternative_cost_margin":solution.alternative_cost_margin,
            "alternative_target_camera_mm":solution.alternative_target_camera_mm,
            "hypotheses":solution.hypotheses_evaluated,"refinement_steps":solution.refinement_steps,
            "hypotheses_by_association":solution.hypotheses_by_association,
            "outer_ellipses":solution.ellipses_roi_px.map(|e|ellipse_json(e[0])),
            "support":solution.arcs.iter().map(|a|json!({"roi":a.exposure.roi.0,"kind":format!("{:?}",a.kind),
                "group":a.evidence_group,"arc":a.arc_index,"rms_px":a.rms_px,"sigma_px":a.sigma_px,"used":a.used,
                "support_length_px":a.support_length_px,"evidence_weight":a.evidence_weight})).collect::<Vec<_>>(),
            "withheld_sample_residuals":heldout(&solution,frames),"elapsed_ms":elapsed,
            "sn_feida_mm2":std::array::from_fn::<_,2,_>(|eye| {
                let frame=frames[eye].as_ref()?;let scale=frame.pose.pixels_per_10mm?[0]/10.0;
                if !solution.arcs.iter().any(|a|a.used&&a.exposure.roi==frame.packet.exposure.roi&&a.kind==BoundaryKind::OuterLimbus) {return None;}
                let e=solution.ellipses_roi_px[eye][0]?;
                Some(std::f64::consts::PI*(e.major_radius/scale).powi(2))
            }),
        }),
    }
}

fn evaluate(frames:[Option<Frame>;2],export_sparse:bool)->Value {
    let poses=frames.each_ref().map(|f|f.as_ref().map(|f|f.pose));
    let prepared=frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.prepare()));
    let evidence=prepared.each_ref().map(|e|e.as_ref().map(|p|p.evidence()));
    let camera=PinholeCamera {focal_px:[4000.0,4000.0],principal_px:[4000.0,3000.0]};
    let base=json!({"inputs":frames.each_ref().map(|f|f.as_ref().map(|f|&f.input)),
        "baseline_sam_outer":frames.each_ref().map(|f|ellipse_json(f.as_ref().and_then(|f|f.baseline))),
        "raw_admitted":frames.each_ref().map(|f|f.as_ref().map(|f|f.selected_raw_admitted)),
        "observed_arc_groups":frames.each_ref().map(|f|f.as_ref().map(|f|f.packet.arcs.len())),
        "partial_outline":frames.each_ref().map(|f|f.as_ref().map(|f|json!({"candidates":f.partial_outline.candidates,
            "censored_samples":f.partial_outline.censored_samples,"unsupported_samples":f.partial_outline.unsupported_samples,
            "emitted_arcs":f.partial_outline.emitted_arcs}))),
        "contract":"shared latent fixation versus separate monocular optimizations of the SAME training arcs; no averaged gaze; held-out points condition on upstream detector segmentation/search. Neither metric pose nor gaze accuracy is ground truth."});
    let mut row=base;
    if export_sparse {
        row["sparse_evidence"]=json!(frames.each_ref().map(|f|f.as_ref().map(|f|json!({
            "arcs":f.packet.arcs.iter().map(|a|json!({"group":a.evidence_group,"kind":format!("{:?}",a.kind),
                "points":a.points_roi_px,"band_half_width_px":a.normal_band_half_width_px})).collect::<Vec<_>>(),
            "seeds":f.packet.conics.iter().map(|c|json!({"kind":format!("{:?}",c.kind),"ellipse":ellipse_json(Some(c.ellipse_roi_px))})).collect::<Vec<_>>()
        }))));
    }
    let Some(scene)=approximate_scene(camera,poses) else {row["error"]=json!("no coarse scene support");return row;};
    row["scale_provenance"]=json!(scene.scale_provenance.map(|p|p.map(|p|format!("{p:?}"))));
    row["independent_pixels_per_mm"]=json!(scene.independent_pixels_per_mm);
    for (name,mask) in [("joint",[true,true]),("monocular_right",[true,false]),("monocular_left",[false,true])] {
        let started=Instant::now();
        let result=solve_joint_conics(JointConicRequest {
            eyes:std::array::from_fn(|eye|if mask[eye] {evidence[eye]} else {None}),scene:&scene.prior,
            maximum_hypotheses:16,maximum_refinements:12,maximum_source_skew_ns:0,
            exposure_uncertainty_ns:2_000_000,motion_bound_px_per_second:150.0,
        });
        row[name]=solution_json(result,frames.each_ref().map(Option::as_ref),started.elapsed().as_secs_f64()*1000.0);
    }
    row
}

fn run()->Result<(),String> {
    let mut args=std::env::args().skip(1);
    let output=PathBuf::from(args.next().ok_or("usage: buttercup_stereo_conic_eval OUTPUT.jsonl SAM_CACHE.jsonl...")?);
    let mut files=Vec::new();
    let mut maximum_frames_per_cache=usize::MAX;
    let mut partial_outlines=false;
    let mut export_sparse=false;
    let mut source_order_replay=false;
    let mut arrival_delay_ns=[0u64;2];
    while let Some(arg)=args.next() {
        if arg=="--max-frames-per-cache" {
            maximum_frames_per_cache=args.next().ok_or("missing frame limit")?.parse::<usize>().map_err(|e|e.to_string())?;
            if maximum_frames_per_cache==0 {return Err("frame limit must be positive".into());}
        } else if arg=="--partial-outlines" {partial_outlines=true;}
        else if arg=="--export-sparse-evidence" {export_sparse=true;}
        else if arg=="--source-order-replay" {source_order_replay=true;}
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
    if source_order_replay {
        return source_order::run(&files,maximum_frames_per_cache,partial_outlines,arrival_delay_ns,&mut writer);
    }
    if arrival_delay_ns!=[0;2] {return Err("arrival-delay-ns requires source-order-replay".into());}
    let mut pending:HashMap<(String,u64),[Option<Frame>;2]>=HashMap::new();
    let mut count=0usize;
    let mut write=|frames|->Result<(),String> {
        let row=evaluate(frames,export_sparse);serde_json::to_writer(&mut writer,&row).map_err(|e|e.to_string())?;
        writer.write_all(b"\n").map_err(|e|e.to_string())?;count+=1;
        if count%500==0 {writer.flush().map_err(|e|e.to_string())?;eprintln!("stereo evaluation reads={count}");}
        Ok(())
    };
    for path in files {
        for (line_number,line) in BufReader::new(File::open(&path).map_err(|e|e.to_string())?).lines().take(maximum_frames_per_cache).enumerate() {
            let row=serde_json::from_str(&line.map_err(|e|e.to_string())?).map_err(|e|format!("{path}:{}: {e}",line_number+1))?;
            let frame=prepare(row,partial_outlines)?;
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
